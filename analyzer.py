"""
analyzer.py — Глубокий AI-анализ сообщений в реальном времени

Что делает:
  - Понимает СМЫСЛ сообщения, а не только слова
  - Определяет кто к кому обращается (даже без имени)
  - Детектирует намерение за словами (провокация под видом вопроса и т.д.)
  - Хранит семантическую память — бот не повторяется
  - Строит полный контекст для промпта

Используется вместо старого analyze_message() или поверх него.
"""

import json
import re
import time
import hashlib
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# РЕЗУЛЬТАТ ГЛУБОКОГО АНАЛИЗА
# ════════════════════════════════════════════════════════════════

@dataclass
class DeepAnalysis:
    """Полный результат глубокого анализа одного сообщения."""

    # Базовые метрики (совместимость со старым analyze_message)
    sentiment:        float = 0.0
    aggression:       float = 0.0
    emotionality:     float = 0.3
    flood_score:      float = 0.0
    topic:            str   = "другое"
    subtopic:         str   = ""
    intent:           str   = "болтовня"
    directed_at_bot:  bool  = False
    directed_at_user: Optional[str] = None
    is_conflict:      bool  = False
    conflict_persons: list  = field(default_factory=list)
    topic_continuity: bool  = False
    rex_interest:     float = 0.3

    # Глубокое понимание (новое)
    real_meaning:     str   = ""   # что человек РЕАЛЬНО хочет сказать
    subtext:          str   = ""   # скрытый смысл / подтекст
    target_person:    str   = ""   # к кому обращаются (даже неявно)
    social_role:      str   = ""   # роль автора: доминант/жертва/провокатор/наблюдатель/миротворец
    power_dynamic:    str   = ""   # кто давит на кого
    sarcasm_detected: bool  = False
    is_bait:          bool  = False  # попытка спровоцировать бота
    is_repetition:    bool  = False  # человек повторяет мысль другими словами
    unique_content:   bool  = True   # есть ли что-то новое в сообщении

    # Для ответа бота
    best_response_angle: str = ""  # как лучше ответить: атака/ирония/вопрос/игнор
    what_bot_should_avoid: str = ""  # что НЕ говорить в ответе


# ════════════════════════════════════════════════════════════════
# СЕМАНТИЧЕСКАЯ ПАМЯТЬ — антиповтор
# ════════════════════════════════════════════════════════════════

@dataclass
class SemanticEntry:
    """Запись в семантической памяти."""
    text:       str
    meaning:    str    # смысл сообщения
    fingerprint: str   # хэш для быстрого сравнения
    timestamp:  float
    is_bot:     bool


class SemanticMemory:
    """
    Хранит смыслы сказанного — не дословно, а что имелось в виду.
    Позволяет боту не повторяться и понимать повторы от людей.
    """

    def __init__(self, maxlen: int = 60):
        self._entries: deque[SemanticEntry] = deque(maxlen=maxlen)
        # Последние N смыслов ответов бота — для антиповтора
        self._bot_meanings: deque[str] = deque(maxlen=15)
        self._bot_phrases:  deque[str] = deque(maxlen=30)  # конкретные фразы

    def add_human(self, text: str, meaning: str):
        fp = self._fingerprint(text)
        self._entries.append(SemanticEntry(
            text=text, meaning=meaning,
            fingerprint=fp, timestamp=time.time(), is_bot=False
        ))

    def add_bot(self, text: str, meaning: str = ""):
        fp = self._fingerprint(text)
        self._entries.append(SemanticEntry(
            text=text, meaning=meaning or text[:100],
            fingerprint=fp, timestamp=time.time(), is_bot=True
        ))
        self._bot_meanings.append(meaning or text[:100])
        # Сохраняем ключевые фразы (начала предложений, устойчивые выражения)
        for phrase in self._extract_phrases(text):
            self._bot_phrases.append(phrase)

    def is_human_repeating(self, text: str, meaning: str,
                            window: int = 10) -> bool:
        """Человек повторяет ту же мысль что уже писал?"""
        fp = self._fingerprint(text)
        recent = list(self._entries)[-window:]
        for e in recent:
            if e.is_bot:
                continue
            if e.fingerprint == fp:
                return True
            if meaning and e.meaning and self._semantic_overlap(meaning, e.meaning) > 0.7:
                return True
        return False

    def bot_already_said(self, candidate: str) -> bool:
        """Бот уже говорил что-то похожее?"""
        fp = self._fingerprint(candidate)
        for e in self._entries:
            if e.is_bot and e.fingerprint == fp:
                return True
        return False

    def get_bot_used_phrases(self) -> list[str]:
        return list(self._bot_phrases)

    def get_bot_recent_meanings(self, n: int = 5) -> list[str]:
        return list(self._bot_meanings)[-n:]

    def get_recent_human_topics(self, n: int = 8) -> list[str]:
        entries = [e for e in list(self._entries)[-n:] if not e.is_bot]
        return [e.meaning for e in entries if e.meaning]

    # ── Внутренние утилиты ───────────────────────────────────────

    def _fingerprint(self, text: str) -> str:
        """Нормализованный хэш текста."""
        normalized = re.sub(r'[^а-яёa-z0-9]', '', text.lower())
        return hashlib.md5(normalized.encode()).hexdigest()[:8]

    def _extract_phrases(self, text: str) -> list[str]:
        """Извлекает ключевые фразы из текста бота."""
        phrases = []
        # Первые слова каждого предложения
        sentences = re.split(r'[.!?]\s*', text)
        for s in sentences:
            words = s.strip().split()
            if len(words) >= 3:
                phrases.append(' '.join(words[:4]).lower())
        return phrases[:5]

    def _semantic_overlap(self, a: str, b: str) -> float:
        """Грубая оценка семантического совпадения по словам."""
        wa = set(re.findall(r'[а-яёa-z]{4,}', a.lower()))
        wb = set(re.findall(r'[а-яёa-z]{4,}', b.lower()))
        if not wa or not wb:
            return 0.0
        intersection = wa & wb
        union = wa | wb
        return len(intersection) / len(union)


# Глобальный реестр памятей по chat_id
_semantic_memories: dict[int, SemanticMemory] = {}

def get_semantic_memory(chat_id: int) -> SemanticMemory:
    if chat_id not in _semantic_memories:
        _semantic_memories[chat_id] = SemanticMemory()
    return _semantic_memories[chat_id]


# ════════════════════════════════════════════════════════════════
# ГЛУБОКИЙ АНАЛИЗАТОР
# ════════════════════════════════════════════════════════════════

class DeepAnalyzer:
    """
    Основной класс. Делает глубокий AI-анализ через LLM.

    Отличие от старого analyze_message:
    - Понимает СМЫСЛ и ПОДТЕКСТ, а не только классифицирует
    - Определяет социальную роль говорящего
    - Детектирует повторы и провокации
    - Строит инструкцию для ответа бота
    """

    def __init__(self, client, model: str = "llama-3.3-70b-versatile"):
        self.client = client
        self.model  = model

    def analyze(self,
                text:         str,
                chat_id:      int,
                user_name:    str,
                chat_context: list,    # последние сообщения из БД
                known_names:  list,
                bot_names:    list,
                chat_state    = None) -> DeepAnalysis:
        """
        Полный анализ одного сообщения в контексте чата.
        """
        mem = get_semantic_memory(chat_id)

        # Строим контекст для LLM
        ctx_lines = self._build_context(chat_context, known_names)
        bot_used  = mem.get_bot_used_phrases()
        bot_meanings = mem.get_bot_recent_meanings(5)
        bot_names_str = ", ".join(bot_names) if bot_names else "есет, eset"

        prompt = self._build_prompt(
            text=text,
            user_name=user_name,
            ctx_lines=ctx_lines,
            known_names=known_names,
            bot_names_str=bot_names_str,
            bot_used_phrases=bot_used[:10],
            bot_recent_meanings=bot_meanings,
            chat_state=chat_state,
        )

        raw = self._call_llm(prompt)
        result = self._parse(raw)
        da = self._to_dataclass(result)

        # Проверяем повтор от человека
        da.is_repetition = mem.is_human_repeating(text, da.real_meaning)

        # Сохраняем в семантическую память
        mem.add_human(text, da.real_meaning)

        return da

    def record_bot_reply(self, chat_id: int, reply_text: str):
        """Записываем ответ бота в семантическую память."""
        mem = get_semantic_memory(chat_id)
        # Краткий смысл ответа — первые 80 символов без мата
        meaning = reply_text[:80].strip()
        mem.add_bot(reply_text, meaning)

    # ── Построение промпта ───────────────────────────────────────

    def _build_context(self, chat_context: list, known_names: list) -> list[str]:
        lines = []
        if not chat_context:
            return lines
        for m in chat_context[-10:]:
            uid, msg_text, sentiment, topic_m, fname = m[0], m[1], m[2], m[3], m[4]
            if not msg_text:
                continue
            name = fname or "?"
            short = msg_text[:120].split("[ЗАМЕТКА:")[0].strip()
            aggr = float(m[5]) if len(m) > 5 and m[5] else 0.0
            prefix = "🤖 Есет" if uid is None or uid == -1 else name
            suffix = f" [агрессия:{aggr:.1f}]" if aggr > 0.4 else ""
            lines.append(f"  {prefix}: {short}{suffix}")
        return lines

    def _build_prompt(self, text: str, user_name: str,
                      ctx_lines: list, known_names: list,
                      bot_names_str: str, bot_used_phrases: list,
                      bot_recent_meanings: list, chat_state) -> str:

        ctx_block = "\n".join(ctx_lines) if ctx_lines else "  (нет истории)"
        names_block = ", ".join(known_names) if known_names else "неизвестны"

        state_block = ""
        if chat_state and len(chat_state) > 3:
            state_block = f"\nСОСТОЯНИЕ ЧАТА: тема={chat_state[1] or '?'} накал={float(chat_state[3] or 0):.2f}"

        phrases_block = ""
        if bot_used_phrases:
            phrases_block = f"\nУЖЕ СКАЗАННЫЕ БОТОМ ФРАЗЫ (не повторять): {'; '.join(bot_used_phrases)}"

        meanings_block = ""
        if bot_recent_meanings:
            meanings_block = f"\nПОСЛЕДНИЕ СМЫСЛЫ ОТВЕТОВ БОТА: {'; '.join(bot_recent_meanings)}"

        return f"""Ты — аналитик русскоязычного Telegram-чата. Глубокий анализ одного сообщения.
Верни ТОЛЬКО валидный JSON без markdown и пояснений.

УЧАСТНИКИ ЧАТА: {names_block}
БОТ (имена): {bot_names_str}
{state_block}{phrases_block}{meanings_block}

ИСТОРИЯ ПОСЛЕДНИХ СООБЩЕНИЙ:
{ctx_block}

НОВОЕ СООБЩЕНИЕ от {user_name}: "{text[:600]}"

ЗАДАЧА — понять:
1. Что человек РЕАЛЬНО имеет в виду (real_meaning) — не перефраз, а суть
2. Есть ли скрытый смысл/подтекст (subtext) — ирония, манипуляция, жалоба под видом вопроса
3. К кому РЕАЛЬНО обращается (target_person) — даже если не назван по имени
4. Социальная роль: доминант / жертва / провокатор / наблюдатель / миротворец / клоун
5. Кто на кого давит (power_dynamic) — например "Влад давит на Родиона"
6. Это повтор уже сказанного другими словами? (is_repetition)
7. Попытка спровоцировать бота? (is_bait)
8. Как ЛУЧШЕ ответить боту (best_response_angle): "атака" / "ирония" / "вопрос_в_лоб" / "поддеть_тихо" / "игнор_с_замечанием"
9. Что боту НЕ нужно говорить в ответе (what_bot_should_avoid) — конкретно

ПРАВИЛА КЛАССИФИКАЦИИ:
- flood_score > 0.7 только если реально бессмысленное ("+", "ок", стикер, одно слово без контекста)
- aggression > 0.6 только при явных оскорблениях конкретного человека
- directed_at_bot=true если обращение к боту — явное ИЛИ подразумеваемое (нет другого адресата)
- is_conflict=true только если ДВА РАЗНЫХ человека конфликтуют между собой
- sarcasm_detected=true если есть скрытая насмешка или противоречие смысла и тона
- unique_content=false если человек говорит то же самое что уже говорил недавно

JSON (строго):
{{
  "sentiment": <-1.0..1.0>,
  "aggression": <0.0..1.0>,
  "emotionality": <0.0..1.0>,
  "flood_score": <0.0..1.0>,
  "topic": "<флуд|спор|конфликт|оскорбление|мат|жалоба|юмор|похвала|вопрос|просьба|новость|угроза|провокация|другое>",
  "subtopic": "<1-2 слова>",
  "intent": "<болтовня|ссора|провокация|внимание|помощь|информация|жалоба|юмор|другое>",
  "directed_at_bot": <true|false>,
  "directed_at_user": "<имя или null>",
  "is_conflict": <true|false>,
  "conflict_persons": ["<имя>", "<имя>"],
  "topic_continuity": <true|false>,
  "rex_interest": <0.0..1.0>,
  "real_meaning": "<1-2 предложения — суть без пересказа>",
  "subtext": "<скрытый смысл или пустая строка>",
  "target_person": "<имя или 'бот' или 'все' или 'никто'>",
  "social_role": "<доминант|жертва|провокатор|наблюдатель|миротворец|клоун>",
  "power_dynamic": "<описание или пустая строка>",
  "sarcasm_detected": <true|false>,
  "is_bait": <true|false>,
  "is_repetition": <true|false>,
  "unique_content": <true|false>,
  "best_response_angle": "<атака|ирония|вопрос_в_лоб|поддеть_тихо|игнор_с_замечанием>",
  "what_bot_should_avoid": "<конкретная фраза или тема которую не надо повторять>"
}}"""

    # ── LLM вызов ────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        import time
        for attempt in range(2):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.05,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                msg = str(e).lower()
                if "429" in msg or "rate_limit" in msg or "rate limit" in msg:
                    wait = 2.0 * (2 ** attempt)  # 2s, 4s вместо 5/10/20
                    logger.warning(f"[analyzer] Rate limit, жду {wait:.0f}с (попытка {attempt+1}/2)")
                    time.sleep(wait)
                else:
                    logger.error(f"[analyzer] LLM error: {e}")
                    return "{}"
        logger.error("[analyzer] Rate limit: молчим")
        return "{}"

    # ── Парсинг ──────────────────────────────────────────────────

    def _parse(self, raw: str) -> dict:
        raw = raw.replace("```json", "").replace("```", "").strip()
        # Иногда LLM добавляет текст после JSON — обрезаем
        brace = raw.find("{")
        rbrace = raw.rfind("}")
        if brace != -1 and rbrace != -1:
            raw = raw[brace:rbrace+1]
        try:
            return json.loads(raw)
        except Exception as e:
            logger.warning(f"[analyzer] JSON parse error: {e} | raw: {raw[:200]}")
            return {}

    def _to_dataclass(self, d: dict) -> DeepAnalysis:
        """Конвертирует dict в DeepAnalysis с дефолтами."""
        return DeepAnalysis(
            sentiment        = float(d.get("sentiment", 0.0)),
            aggression       = float(d.get("aggression", 0.2)),
            emotionality     = float(d.get("emotionality", 0.3)),
            flood_score      = float(d.get("flood_score", 0.0)),
            topic            = d.get("topic", "другое"),
            subtopic         = d.get("subtopic", ""),
            intent           = d.get("intent", "болтовня"),
            directed_at_bot  = bool(d.get("directed_at_bot", False)),
            directed_at_user = d.get("directed_at_user"),
            is_conflict      = bool(d.get("is_conflict", False)),
            conflict_persons = d.get("conflict_persons", []),
            topic_continuity = bool(d.get("topic_continuity", False)),
            rex_interest     = float(d.get("rex_interest", 0.3)),
            real_meaning     = d.get("real_meaning", ""),
            subtext          = d.get("subtext", ""),
            target_person    = d.get("target_person", ""),
            social_role      = d.get("social_role", ""),
            power_dynamic    = d.get("power_dynamic", ""),
            sarcasm_detected = bool(d.get("sarcasm_detected", False)),
            is_bait          = bool(d.get("is_bait", False)),
            is_repetition    = bool(d.get("is_repetition", False)),
            unique_content   = bool(d.get("unique_content", True)),
            best_response_angle    = d.get("best_response_angle", "ирония"),
            what_bot_should_avoid  = d.get("what_bot_should_avoid", ""),
        )

    def to_dict(self, da: DeepAnalysis) -> dict:
        """Обратно в dict для совместимости со старым кодом."""
        return {
            "sentiment":        da.sentiment,
            "aggression":       da.aggression,
            "emotionality":     da.emotionality,
            "flood_score":      da.flood_score,
            "topic":            da.topic,
            "subtopic":         da.subtopic,
            "intent":           da.intent,
            "directed_at_bot":  da.directed_at_bot,
            "directed_at_user": da.directed_at_user,
            "is_conflict":      da.is_conflict,
            "conflict_persons": da.conflict_persons,
            "topic_continuity": da.topic_continuity,
            "rex_interest":     da.rex_interest,
            # Доп. поля
            "real_meaning":     da.real_meaning,
            "subtext":          da.subtext,
            "target_person":    da.target_person,
            "social_role":      da.social_role,
            "power_dynamic":    da.power_dynamic,
            "sarcasm_detected": da.sarcasm_detected,
            "is_bait":          da.is_bait,
            "is_repetition":    da.is_repetition,
            "unique_content":   da.unique_content,
            "best_response_angle":   da.best_response_angle,
            "what_bot_should_avoid": da.what_bot_should_avoid,
        }


# ════════════════════════════════════════════════════════════════
# СТРОИТЕЛЬ СИСТЕМНОГО ПРОМПТА ДЛЯ ОТВЕТА БОТА
# Берёт глубокий анализ и строит точные инструкции
# ════════════════════════════════════════════════════════════════

class ResponseInstructor:
    """
    Строит блок инструкций для промпта бота на основе DeepAnalysis.
    Вставляется в build_system() перед генерацией ответа.
    """

    def build(self, da: DeepAnalysis, mem: SemanticMemory,
              respond_mode: str) -> str:
        lines = []

        # Что происходит реально
        if da.real_meaning:
            lines.append(f"СМЫСЛ СООБЩЕНИЯ: {da.real_meaning}")
        if da.subtext:
            lines.append(f"ПОДТЕКСТ: {da.subtext}")
        if da.target_person and da.target_person not in ("никто", ""):
            lines.append(f"РЕАЛЬНО ОБРАЩАЕТСЯ К: {da.target_person}")
        if da.social_role:
            lines.append(f"РОЛЬ СОБЕСЕДНИКА: {da.social_role}")
        if da.power_dynamic:
            lines.append(f"ДИНАМИКА ВЛАСТИ: {da.power_dynamic}")

        # Флаги
        if da.sarcasm_detected:
            lines.append("⚡ САРКАЗМ в сообщении — ответь на скрытый смысл, не на слова")
        if da.is_bait:
            lines.append("🎣 ПРОВОКАЦИЯ — просто ответь резче обычного, без комментариев про провокацию")
        if da.is_repetition:
            lines.append("🔁 ПОВТОР — человек говорит то же самое снова. Укажи это прямо и язвительно")

        # Инструкция по углу ответа
        angle_map = {
            "атака":             "бей лично и конкретно — используй что знаешь об этом человеке",
            "ирония":            "ответь с холодной иронией — один точный укол без объяснений",
            "вопрос_в_лоб":      "задай один неудобный вопрос который обнажает суть",
            "поддеть_тихо":      "скажи что-то тихое но очень точное — пусть дойдёт потом",
            "игнор_с_замечанием":"сделай вид что тебе неинтересно, но мимоходом ужаль",
        }
        if da.best_response_angle in angle_map:
            lines.append(f"КАК ОТВЕЧАТЬ: {angle_map[da.best_response_angle]}")

        # Антиповтор — что НЕ говорить
        avoid_phrases = mem.get_bot_used_phrases()
        avoid_lines = []
        if da.what_bot_should_avoid:
            avoid_lines.append(da.what_bot_should_avoid)
        if avoid_phrases:
            avoid_lines.extend(avoid_phrases[:6])
        if avoid_lines:
            lines.append(f"НЕ ПОВТОРЯЙ ЭТИ ФРАЗЫ И СМЫСЛЫ: {' | '.join(avoid_lines)}")

        # Режим
        mode_map = {
            "attack":   "РЕЖИМ АТАКИ — жёстко, лично, с матом, без жалости",
            "conflict": "РЕЖИМ РАЗЖИГАТЕЛЯ — ты в чужом конфликте, подливай масла, выбирай сторону",
            "snark":    "РЕЖИМ САРКАЗМА — холодно, точно, одна убийственная фраза",
            "neutral":  "РЕЖИМ ЛЕНИВОГО — коротко, грубо, без энтузиазма",
        }
        if respond_mode in mode_map:
            lines.append(f"\n{mode_map[respond_mode]}")

        if not lines:
            return ""

        return "\n[АНАЛИЗ СИТУАЦИИ]\n" + "\n".join(lines) + "\n"


# ════════════════════════════════════════════════════════════════
# ГЛОБАЛЬНЫЙ ЭКЗЕМПЛЯР
# ════════════════════════════════════════════════════════════════

_analyzer: Optional[DeepAnalyzer] = None
_instructor = ResponseInstructor()

def init_analyzer(client, model: str = "llama-3.3-70b-versatile"):
    """Инициализировать анализатор с клиентом Groq. Вызвать при старте бота."""
    global _analyzer
    _analyzer = DeepAnalyzer(client=client, model=model)
    logger.info("[analyzer] DeepAnalyzer инициализирован")

def get_analyzer() -> Optional[DeepAnalyzer]:
    return _analyzer

def get_instructor() -> ResponseInstructor:
    return _instructor


def deep_analyze(text: str, chat_id: int, user_name: str,
                 chat_context: list, known_names: list,
                 bot_names: list, chat_state=None) -> dict:
    """
    Главная точка входа.
    Возвращает dict совместимый со старым analyze_message() + доп. поля.
    Если analyzer не инициализирован — возвращает пустой базовый dict.
    """
    if _analyzer is None:
        logger.warning("[analyzer] не инициализирован, fallback")
        return _fallback()

    try:
        da = _analyzer.analyze(
            text=text, chat_id=chat_id, user_name=user_name,
            chat_context=chat_context, known_names=known_names,
            bot_names=bot_names, chat_state=chat_state,
        )
        return _analyzer.to_dict(da)
    except Exception as e:
        logger.error(f"[analyzer] deep_analyze error: {e}")
        return _fallback()


def record_bot_reply(chat_id: int, reply_text: str):
    """
    Записываем ответ бота в семантическую память.
    Вызывать после каждого ответа бота.
    """
    if _analyzer:
        _analyzer.record_bot_reply(chat_id, reply_text)
    else:
        mem = get_semantic_memory(chat_id)
        mem.add_bot(reply_text)


def build_response_instructions(analysis: dict, chat_id: int,
                                 respond_mode: str = "neutral") -> str:
    """
    Строит блок инструкций для промпта бота.
    Принимает dict из deep_analyze().
    """
    mem = get_semantic_memory(chat_id)
    if _analyzer:
        da = _analyzer._to_dataclass(analysis)
    else:
        # Минимальный DeepAnalysis из dict
        da = DeepAnalysis(
            real_meaning = analysis.get("real_meaning", ""),
            subtext      = analysis.get("subtext", ""),
            target_person= analysis.get("target_person", ""),
            social_role  = analysis.get("social_role", ""),
            power_dynamic= analysis.get("power_dynamic", ""),
            sarcasm_detected  = analysis.get("sarcasm_detected", False),
            is_bait           = analysis.get("is_bait", False),
            is_repetition     = analysis.get("is_repetition", False),
            best_response_angle   = analysis.get("best_response_angle", "ирония"),
            what_bot_should_avoid = analysis.get("what_bot_should_avoid", ""),
        )
    return _instructor.build(da, mem, respond_mode)


def _fallback() -> dict:
    return {
        "sentiment": 0.0, "aggression": 0.2, "emotionality": 0.3,
        "flood_score": 0.5, "topic": "другое", "subtopic": "",
        "intent": "болтовня", "directed_at_bot": False,
        "directed_at_user": None, "is_conflict": False,
        "conflict_persons": [], "topic_continuity": False,
        "rex_interest": 0.3, "real_meaning": "", "subtext": "",
        "target_person": "", "social_role": "", "power_dynamic": "",
        "sarcasm_detected": False, "is_bait": False,
        "is_repetition": False, "unique_content": True,
        "best_response_angle": "ирония", "what_bot_should_avoid": "",
    }
