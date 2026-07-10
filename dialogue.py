"""
dialogue.py — Память диалогов Есет

Что делает:
  DialogueMemory  — хранит историю сообщений по чату в RAM
                    (тема, кто что говорил, реакции на бота)
  TopicTracker    — отслеживает смену тем, длительность, интерес участников
  OverloadDetector— понимает когда бот уже перебарщивает и надо заткнуться
  DialogueContext — итоговый срез для промпта и brain.py

Хранится в памяти процесса (не БД) — быстро, актуально.
При рестарте сбрасывается (не критично — контекст живёт внутри сессии).
"""

import time
import re
import logging
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════
# СТРУКТУРЫ
# ════════════════════════════════════════════════════════════════

@dataclass
class Turn:
    """Один ход диалога — одно сообщение."""
    user_id:    int
    user_name:  str
    text:       str
    timestamp:  float
    is_bot:     bool       = False
    topic:      str        = ""
    aggression: float      = 0.0
    sentiment:  float      = 0.0
    # Реакция людей на бот после этого сообщения (заполняется потом)
    reactions:  list       = field(default_factory=list)  # "ignored"|"replied"|"laughed"|"attacked"


@dataclass
class TopicSegment:
    """Отрезок разговора на одну тему."""
    topic:         str
    started_at:    float
    ended_at:      Optional[float] = None
    turns_count:   int             = 0
    bot_turns:     int             = 0      # сколько раз бот влез
    participants:  set             = field(default_factory=set)
    avg_aggression: float          = 0.0
    bot_ignored:   int             = 0      # сколько раз бота проигнорили в этой теме
    bot_welcomed:  int             = 0      # сколько раз ответили боту


@dataclass
class DialogueContext:
    """
    Итоговый срез состояния диалога — передаётся в brain и промпт.
    """
    # Текущая тема
    current_topic:       str   = ""
    topic_age:           float = 0.0   # сколько секунд живёт тема
    topic_turns:         int   = 0     # сообщений на эту тему
    topic_changed:       bool  = False # тема только что сменилась

    # Активность бота
    bot_recent_turns:    int   = 0     # сколько раз бот говорил за последние N сообщений
    bot_reply_rate:      float = 0.0   # доля сообщений бота в последних N (0..1)
    bot_overloading:     bool  = False # бот явно перебарщивает
    bot_cooling_down:    bool  = False # бот в режиме охлаждения (принудительное молчание)

    # Реакции людей на бота
    last_bot_reaction:   str   = ""    # "ignored"|"replied"|"laughed"|"attacked"|"none"
    bot_ignored_streak:  int   = 0     # сколько раз подряд проигнорили
    bot_welcomed_ratio:  float = 0.0   # доля позитивных реакций за сессию

    # Динамика разговора
    dialogue_active:     bool  = False # идёт ли живой диалог прямо сейчас
    who_talks_most:      str   = ""    # кто доминирует в разговоре
    silent_users:        list  = field(default_factory=list) # кто давно молчит

    # Для промпта
    summary:             str   = ""    # текстовое описание ситуации


# ════════════════════════════════════════════════════════════════
# ДЕТЕКТОР ТЕМЫ (быстрый, без LLM)
# ════════════════════════════════════════════════════════════════

_TOPIC_KEYWORDS = {
    "игры":      ["игра", "игры", "геймс", "cs", "дота", "майнкрафт", "стрим", "играем", "матч"],
    "работа":    ["работа", "работы", "офис", "начальник", "зарплата", "уволили", "найм", "проект"],
    "деньги":    ["деньги", "бабки", "бабло", "башли", "касса", "бюджет", "долг", "займ", "платить"],
    "отношения": ["девушка", "парень", "жена", "муж", "расстались", "познакомился", "свидание", "изменил"],
    "алкоголь":  ["пиво", "водка", "бухать", "пьём", "выпить", "бухло", "похмелье", "бар"],
    "еда":       ["еда", "жрать", "поесть", "готовить", "пицца", "суши", "доставка", "голодный"],
    "спорт":     ["футбол", "тренировка", "спортзал", "качалка", "качаться", "пробежка", "матч"],
    "машины":    ["машина", "тачка", "авто", "ехать", "дтп", "гибдд", "штраф", "права"],
    "политика":  ["политика", "путин", "правительство", "выборы", "закон", "страна", "власть"],
    "музыка":    ["музыка", "трек", "песня", "слушать", "альбом", "концерт", "артист"],
    "кино":      ["фильм", "сериал", "смотреть", "кино", "netflix", "ютуб", "видео"],
    "конфликт":  ["поругались", "разборка", "конфликт", "наехал", "разборки", "разборку"],
    "флуд":      [],
}

def _detect_topic_fast(text: str, prev_topic: str = "") -> str:
    """Быстрое определение темы по ключевым словам."""
    t = text.lower()
    scores = defaultdict(int)
    for topic, keywords in _TOPIC_KEYWORDS.items():
        for kw in keywords:
            if kw in t:
                scores[topic] += 1
    if not scores:
        return prev_topic or "разное"
    # Если новая тема набрала ≥2 совпадения — тема сменилась
    best = max(scores, key=scores.get)
    if scores[best] >= 1:
        return best
    return prev_topic or "разное"


# ════════════════════════════════════════════════════════════════
# ТРЕКЕР ТЕМ
# ════════════════════════════════════════════════════════════════

class TopicTracker:
    """
    Следит за сменой тем в чате.
    Тема считается сменившейся если:
    - 3+ подряд сообщения на новую тему
    - или резкий скачок (LLM-тема сильно отличается)
    """
    def __init__(self, min_turns_to_confirm: int = 2):
        self.current:    Optional[TopicSegment] = None
        self.history:    list[TopicSegment]     = []  # последние 10 тем
        self._pending:   str                    = ""
        self._pending_n: int                    = 0
        self._min_conf   = min_turns_to_confirm

    def push(self, topic: str, turn: Turn) -> bool:
        """
        Добавляет ход и обновляет тему.
        Возвращает True если тема сменилась.
        """
        changed = False

        if self.current is None:
            self.current = TopicSegment(topic=topic, started_at=turn.timestamp)
            self.current.participants.add(turn.user_id)
            self.current.turns_count = 1
            return False

        if turn.is_bot:
            self.current.bot_turns += 1
            self.current.avg_aggression = (
                self.current.avg_aggression * 0.8 + turn.aggression * 0.2
            )
            return False

        self.current.participants.add(turn.user_id)
        self.current.turns_count += 1
        self.current.avg_aggression = (
            self.current.avg_aggression * 0.8 + turn.aggression * 0.2
        )

        # Проверяем смену темы
        if topic != self.current.topic and topic not in ("разное", "флуд", ""):
            if topic == self._pending:
                self._pending_n += 1
            else:
                self._pending   = topic
                self._pending_n = 1

            if self._pending_n >= self._min_conf:
                # Подтверждаем смену темы
                self.current.ended_at = turn.timestamp
                self.history.append(self.current)
                if len(self.history) > 20:
                    self.history.pop(0)
                self.current = TopicSegment(
                    topic=topic,
                    started_at=turn.timestamp,
                    turns_count=1,
                )
                self.current.participants.add(turn.user_id)
                self._pending   = ""
                self._pending_n = 0
                changed = True
        else:
            self._pending   = ""
            self._pending_n = 0

        return changed

    def record_bot_reaction(self, reaction: str):
        """Записываем как люди отреагировали на бота."""
        if not self.current:
            return
        if reaction == "ignored":
            self.current.bot_ignored += 1
        elif reaction in ("replied", "laughed"):
            self.current.bot_welcomed += 1

    def topic_age(self) -> float:
        if not self.current:
            return 0.0
        return time.time() - self.current.started_at

    def topic_bot_ratio(self) -> float:
        """Доля сообщений бота в текущей теме."""
        if not self.current or self.current.turns_count == 0:
            return 0.0
        return self.current.bot_turns / max(self.current.turns_count, 1)


# ════════════════════════════════════════════════════════════════
# ДЕТЕКТОР ПЕРЕБОРА
# ════════════════════════════════════════════════════════════════

class OverloadDetector:
    """
    Следит за активностью бота и решает — не перебарщивает ли он.

    Перебор = бот слишком часто говорит В ОДНОМ РАЗГОВОРЕ,
    его игнорируют, или он доминирует над людьми.
    """

    def __init__(self):
        # Скользящее окно: последние 20 ходов (бот + люди)
        self._window:        deque = deque(maxlen=20)
        # Подряд игнорирований после ответа бота
        self._ignore_streak: int   = 0
        # Подряд ответов бота без ответной реакции людей
        self._bot_monologue: int   = 0
        # Режим охлаждения: бот молчит N сообщений
        self._cooling:       int   = 0
        self._cool_duration: int   = 0

    def push(self, is_bot: bool, reacted_to_bot: bool = False):
        """
        is_bot: True если это сообщение бота
        reacted_to_bot: True если это сообщение человека — ответ на бота
        """
        self._window.append(is_bot)

        if self._cooling > 0:
            if not is_bot:
                self._cooling -= 1
            return

        if is_bot:
            self._bot_monologue += 1
        else:
            if reacted_to_bot:
                self._ignore_streak  = 0
                self._bot_monologue  = 0
            else:
                # Человек написал не боту
                if self._bot_monologue > 0:
                    self._ignore_streak += 1

    def record_ignore(self):
        """Бота явно проигнорировали после его ответа."""
        self._ignore_streak += 1
        if self._ignore_streak >= 2:
            # Начинаем охлаждение
            self._start_cooling(turns=4 + self._ignore_streak)

    def record_engagement(self):
        """Человек ответил боту — хороший знак."""
        self._ignore_streak  = max(0, self._ignore_streak - 1)
        self._bot_monologue  = 0
        if self._cooling > 0:
            self._cooling = max(0, self._cooling - 2)

    def _start_cooling(self, turns: int):
        self._cooling      = turns
        self._cool_duration = turns
        logger.info(f"[OVERLOAD] Начало охлаждения на {turns} сообщений")

    # ── Метрики ──────────────────────────────────────────────────

    def bot_reply_rate(self, window: int = 10) -> float:
        """Доля сообщений бота в последних window ходах."""
        recent = list(self._window)[-window:]
        if not recent:
            return 0.0
        return sum(1 for x in recent if x) / len(recent)

    def is_overloading(self) -> bool:
        """Бот явно слишком много говорит."""
        rate = self.bot_reply_rate(10)
        if rate > 0.45:        # бот > 45% сообщений — перебор
            return True
        if self._ignore_streak >= 3:  # 3+ игнора подряд
            return True
        if self._bot_monologue >= 3:  # 3+ ответа бота без реакции
            return True
        return False

    def is_cooling(self) -> bool:
        return self._cooling > 0

    def cooling_progress(self) -> float:
        """0..1 — насколько охлаждение прошло."""
        if self._cool_duration == 0:
            return 1.0
        return 1.0 - (self._cooling / self._cool_duration)

    def ignore_streak(self) -> int:
        return self._ignore_streak

    def overload_penalty(self) -> float:
        """
        Штраф к вероятности ответа: 0..1.
        0 — нет штрафа, 1 — точно молчать.
        """
        if self.is_cooling():
            return 0.90  # охлаждение — очень большой штраф

        penalty = 0.0
        rate = self.bot_reply_rate(10)
        if rate > 0.30:
            penalty += (rate - 0.30) * 2.5  # растёт быстро после 30%
        penalty += min(self._ignore_streak * 0.15, 0.45)
        penalty += min(self._bot_monologue * 0.10, 0.30)
        return min(penalty, 0.85)


# ════════════════════════════════════════════════════════════════
# ПАМЯТЬ ДИАЛОГА (один чат)
# ════════════════════════════════════════════════════════════════

class DialogueMemory:
    """
    Полная память одного чата.
    Хранит повороты разговора, отслеживает темы, считает перебор.
    """
    def __init__(self, chat_id: int, maxlen: int = 80):
        self.chat_id   = chat_id
        self._turns:   deque[Turn]    = deque(maxlen=maxlen)
        self.topics    = TopicTracker(min_turns_to_confirm=2)
        self.overload  = OverloadDetector()
        self._last_bot_turn_idx: int  = -1   # индекс последнего хода бота

    # ── Добавление сообщений ─────────────────────────────────────

    def add_human(self, user_id: int, user_name: str, text: str,
                  topic: str = "", aggression: float = 0.0,
                  sentiment: float = 0.0) -> bool:
        """
        Добавляет сообщение человека.
        Возвращает True если тема сменилась.
        """
        now = time.time()
        # Определяем тему если не передана
        if not topic:
            prev = self.topics.current.topic if self.topics.current else ""
            topic = _detect_topic_fast(text, prev)

        turn = Turn(
            user_id=user_id, user_name=user_name,
            text=text, timestamp=now,
            topic=topic, aggression=aggression, sentiment=sentiment,
        )

        # Проверяем — ответ ли это на бота
        reacted = self._is_reaction_to_bot(text, user_id)
        if reacted:
            reaction_type = self._classify_reaction(text, aggression, sentiment)
            self.overload.record_engagement() if reaction_type != "attacked" else None
            self.topics.record_bot_reaction(reaction_type)
            # Обновляем реакцию в последнем ходе бота
            self._mark_last_bot_reaction(reaction_type)
        else:
            if self._last_bot_turn_idx >= 0:
                self.overload.push(is_bot=False, reacted_to_bot=False)
            else:
                self.overload.push(is_bot=False, reacted_to_bot=False)

        changed = self.topics.push(topic, turn)
        self._turns.append(turn)
        return changed

    def add_bot(self, text: str, topic: str = ""):
        """Добавляет сообщение бота."""
        now = time.time()
        if not topic and self.topics.current:
            topic = self.topics.current.topic
        turn = Turn(
            user_id=-1, user_name="Есет",
            text=text, timestamp=now,
            is_bot=True, topic=topic,
        )
        self.overload.push(is_bot=True)
        self.topics.push(topic, turn)
        self._turns.append(turn)
        self._last_bot_turn_idx = len(self._turns) - 1

    # ── Контекст для brain и промпта ─────────────────────────────

    def get_context(self) -> DialogueContext:
        ctx = DialogueContext()

        seg = self.topics.current
        if seg:
            ctx.current_topic   = seg.topic
            ctx.topic_age       = self.topics.topic_age()
            ctx.topic_turns     = seg.turns_count
            ctx.topic_changed   = False  # уже отработано в push

        ctx.bot_recent_turns = self._count_bot_recent(10)
        ctx.bot_reply_rate   = self.overload.bot_reply_rate(10)
        ctx.bot_overloading  = self.overload.is_overloading()
        ctx.bot_cooling_down = self.overload.is_cooling()
        ctx.bot_ignored_streak = self.overload.ignore_streak()

        ctx.last_bot_reaction  = self._get_last_bot_reaction()
        ctx.bot_welcomed_ratio = self._welcomed_ratio()
        ctx.dialogue_active    = self._is_dialogue_active()
        ctx.who_talks_most     = self._dominant_speaker()
        ctx.silent_users       = self._find_silent_users()

        ctx.summary = self._build_summary(ctx)
        return ctx

    def overload_penalty(self) -> float:
        """Штраф к вероятности ответа из-за перебора."""
        return self.overload.overload_penalty()

    def force_silence(self) -> bool:
        """True = бот ОБЯЗАН молчать (охлаждение или явный перебор)."""
        if self.overload.is_cooling():
            return True
        if self.overload.bot_reply_rate(6) > 0.60:  # >60% последних 6 — стоп
            return True
        return False

    # ── Внутренние методы ────────────────────────────────────────

    def _is_reaction_to_bot(self, text: str, user_id: int) -> bool:
        """Является ли это сообщение реакцией на бота?"""
        if not self._turns:
            return False
        # Смотрим последние 3 хода — был ли там бот
        recent = list(self._turns)[-3:]
        for t in reversed(recent):
            if t.is_bot:
                return True
            if t.user_id != user_id:
                return False  # между ботом и нами — другой человек
        return False

    def _classify_reaction(self, text: str, aggression: float,
                            sentiment: float) -> str:
        t = text.lower()
        if aggression > 0.55:
            return "attacked"
        if any(w in t for w in ["хаха", "лол", "смешно", "))))", "хахаха"]):
            return "laughed"
        if sentiment > 0.2:
            return "replied"
        if len(t.split()) <= 2 and sentiment < 0.0:
            return "ignored"
        return "replied"

    def _mark_last_bot_reaction(self, reaction: str):
        """Обновляем реакцию в последнем ходе бота."""
        for t in reversed(self._turns):
            if t.is_bot:
                t.reactions.append(reaction)
                break

    def _get_last_bot_reaction(self) -> str:
        """Последняя реакция на бота."""
        for t in reversed(self._turns):
            if t.is_bot and t.reactions:
                return t.reactions[-1]
        return "none"

    def _count_bot_recent(self, n: int) -> int:
        recent = list(self._turns)[-n:]
        return sum(1 for t in recent if t.is_bot)

    def _welcomed_ratio(self) -> float:
        """Доля позитивных реакций на бота за всю сессию."""
        reactions = []
        for t in self._turns:
            if t.is_bot:
                reactions.extend(t.reactions)
        if not reactions:
            return 0.5  # нет данных — нейтрально
        good = sum(1 for r in reactions if r in ("replied", "laughed"))
        return good / len(reactions)

    def _is_dialogue_active(self) -> bool:
        """Идёт ли живой разговор (сообщения в последние 3 мин)."""
        if not self._turns:
            return False
        last = list(self._turns)[-1]
        return (time.time() - last.timestamp) < 180

    def _dominant_speaker(self) -> str:
        """Кто больше всего говорит в последних 15 сообщениях."""
        recent = [t for t in list(self._turns)[-15:] if not t.is_bot]
        if not recent:
            return ""
        freq: dict[str, int] = defaultdict(int)
        for t in recent:
            freq[t.user_name] += 1
        return max(freq, key=freq.get) if freq else ""

    def _find_silent_users(self) -> list:
        """Кто давно не писал (>10 сообщений назад)."""
        recent_speakers = {t.user_name for t in list(self._turns)[-10:] if not t.is_bot}
        all_speakers    = {t.user_name for t in self._turns if not t.is_bot}
        return list(all_speakers - recent_speakers)

    def _build_summary(self, ctx: DialogueContext) -> str:
        parts = []

        if ctx.current_topic:
            age_str = f"{ctx.topic_age:.0f}с" if ctx.topic_age < 60 else f"{ctx.topic_age/60:.0f}мин"
            parts.append(f"Тема: {ctx.current_topic} ({age_str}, {ctx.topic_turns} сообщ.)")

        if ctx.bot_cooling_down:
            parts.append("🧊 БОТ В РЕЖИМЕ ОХЛАЖДЕНИЯ — молчит")
        elif ctx.bot_overloading:
            parts.append(f"⚠️ ПЕРЕБОР: бот говорил {ctx.bot_recent_turns}/10 сообщ. ({ctx.bot_reply_rate:.0%})")

        if ctx.bot_ignored_streak >= 2:
            parts.append(f"😶 проигнорили {ctx.bot_ignored_streak} раз подряд")

        if ctx.last_bot_reaction == "attacked":
            parts.append("😡 на последнее сообщение бота наехали")
        elif ctx.last_bot_reaction == "laughed":
            parts.append("😂 на бота смеялись — зашло")
        elif ctx.last_bot_reaction == "ignored":
            parts.append("🙄 бота проигнорили")

        if ctx.who_talks_most:
            parts.append(f"Доминирует: {ctx.who_talks_most}")

        if ctx.silent_users:
            parts.append(f"Молчат: {', '.join(ctx.silent_users[:3])}")

        return " | ".join(parts) if parts else "диалог спокойный"

    # ── История для промпта ──────────────────────────────────────

    def get_recent_turns(self, n: int = 8) -> list[Turn]:
        """Последние N ходов для промпта."""
        return list(self._turns)[-n:]

    def get_topic_history(self, n: int = 5) -> list[str]:
        """Последние N тем."""
        topics = [s.topic for s in self.topics.history[-n:]]
        if self.topics.current:
            topics.append(self.topics.current.topic)
        return topics

    def get_stats(self) -> dict:
        return {
            "total_turns":     len(self._turns),
            "bot_turns":       sum(1 for t in self._turns if t.is_bot),
            "current_topic":   self.topics.current.topic if self.topics.current else "",
            "topic_count":     len(self.topics.history) + (1 if self.topics.current else 0),
            "overload_penalty": self.overload.overload_penalty(),
            "is_cooling":      self.overload.is_cooling(),
        }


# ════════════════════════════════════════════════════════════════
# ГЛОБАЛЬНЫЙ РЕЕСТР ПАМЯТЕЙ
# ════════════════════════════════════════════════════════════════

_memories: dict[int, DialogueMemory] = {}

def get_dialogue(chat_id: int) -> DialogueMemory:
    """Возвращает или создаёт память диалога для chat_id."""
    if chat_id not in _memories:
        _memories[chat_id] = DialogueMemory(chat_id)
    return _memories[chat_id]


# ════════════════════════════════════════════════════════════════
# ТЕСТ
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("ТЕСТ dialogue.py")
    print("=" * 60)

    d = DialogueMemory(chat_id=1)

    # Симулируем разговор
    d.add_human(1, "Влад",   "ну что пацаны сегодня играем?",     "игры",  0.1)
    d.add_human(2, "Родион", "да давай дота или что",             "игры",  0.1)
    d.add_human(3, "Костя",  "ок щас зайду",                      "игры",  0.1)
    d.add_bot("ты уже зашёл Костя или опять тупишь")

    ctx = d.get_context()
    print(f"Тема: {ctx.current_topic}")
    print(f"Перебор: {ctx.bot_overloading}")
    print(f"Summary: {ctx.summary}")
    print()

    # Бота игнорят несколько раз
    d.add_human(1, "Влад",   "влад в игре",      "игры", 0.1)
    d.add_human(2, "Родион", "ок стартуем",      "игры", 0.1)
    d.add_bot("и без меня значит")
    d.add_human(1, "Влад",   "...",               "игры", 0.0)
    d.add_human(2, "Родион", "короче давай",      "игры", 0.1)
    d.add_bot("никто не ответил — продолжу")
    d.add_human(3, "Костя",  "стоп хватит уже",  "игры", 0.3)
    d.add_human(1, "Влад",   "да заткнись",       "флуд", 0.4)
    d.add_bot("ок понял")
    d.add_human(2, "Родион", "наконец-то",        "флуд", 0.2)
    d.add_human(1, "Влад",   "тишина",            "флуд", 0.1)
    d.add_bot("ещё скажу")
    d.add_human(3, "Костя",  "не надо",           "флуд", 0.2)

    ctx = d.get_context()
    print(f"После игнора:")
    print(f"  Перебор: {ctx.bot_overloading}")
    print(f"  Охлаждение: {ctx.bot_cooling_down}")
    print(f"  Игнор подряд: {ctx.bot_ignored_streak}")
    print(f"  Штраф: {d.overload_penalty():.2f}")
    print(f"  Принудит. молчание: {d.force_silence()}")
    print(f"  Summary: {ctx.summary}")
    print()

    # Смена темы
    d.add_human(1, "Влад",   "кстати пиво сегодня берём?",  "алкоголь", 0.1)
    d.add_human(2, "Родион", "да давай бухнём",             "алкоголь", 0.1)
    d.add_human(3, "Костя",  "я за пивом",                  "алкоголь", 0.1)

    ctx = d.get_context()
    print(f"После смены темы:")
    print(f"  Тема: {ctx.current_topic}")
    print(f"  История тем: {d.get_topic_history()}")
    print(f"  Summary: {ctx.summary}")

    stats = d.get_stats()
    print(f"\nСтатистика: {stats}")
