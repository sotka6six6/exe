"""
brain.py — Мозг Eset v2

Отдельный модуль анализа и принятия решений.
Смотрит на ПОТОК сообщений, а не только на одно.
Не знает ничего про ответы — только: отвечать или нет, в каком режиме.

Архитектура:
    ChatMemory      — живая память чата (последние N сообщений)
    SignalDetector  — извлекает сигналы из текста (без LLM)
    ContextAnalyzer — анализирует динамику потока сообщений
    DecisionEngine  — финальное решение на основе всего
    RexBrain        — точка входа, объединяет всё
"""

import re
import math
import random
import logging
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Optional

try:
    from dialogue import get_dialogue, DialogueContext
    _DIALOGUE_OK = True
except Exception as e:
    _DIALOGUE_OK = False
    logging.getLogger(__name__).warning(f"dialogue.py не загружен: {e}")

logger = logging.getLogger(__name__)

try:
    from vocab import MAT, SLANG
    _MAT_SET = frozenset(
        w.lower().strip("-").replace("-", " ")
        for w in (MAT + SLANG)
        if len(w) > 2
    )
except Exception:
    _MAT_SET = frozenset()


# ════════════════════════════════════════════════════════════════
# СТРУКТУРЫ ДАННЫХ
# ════════════════════════════════════════════════════════════════

@dataclass
class Signal:
    """Сигнал из одного сообщения."""
    text:           str
    user_id:        int
    user_name:      str
    timestamp:      float  # time.time()
    # Сигналы
    has_bot_name:   bool  = False
    has_reply_bot:  bool  = False
    has_mention:    bool  = False
    is_flood:       bool  = False
    aggression:     float = 0.0   # 0..1
    conflict_heat:  float = 0.0   # 0..1 конфликт между людьми
    topic:          str   = ""
    sentiment:      float = 0.0   # -1..+1 тон (позитив/негатив)
    is_greeting:    bool  = False # приветствие
    is_selfcmd:     bool  = False # самокоманда/зеркальный троллинг
    word_count:     int   = 0
    mat_count:      int   = 0


@dataclass
class ContextState:
    """Состояние контекста на момент решения."""
    # Динамика потока
    msg_velocity:       float = 0.0  # сообщений в минуту
    aggro_trend:        float = 0.0  # растёт или падает агрессия (-1..+1)
    conflict_active:    bool  = False # идёт ли конфликт прямо сейчас
    conflict_parties:   list  = field(default_factory=list)
    dominant_topic:     str   = ""   # о чём чат последние N сообщений
    heat:               float = 0.0  # общий накал 0..1
    # Про бота
    rex_last_spoke:     int   = 999  # сколько сообщений назад говорил
    rex_was_addressed:  bool  = False # к нему обращались в последних N
    rex_ignored:        int   = 0    # сколько раз проигнорировали после его ответа
    # Про текущее сообщение
    current_signal:     Optional[Signal] = None
    # Диалоговый контекст (из dialogue.py)
    dialogue_ctx:       Optional[object] = None


@dataclass
class Decision:
    """Финальное решение мозга."""
    should_respond: bool  = False
    mode:           str   = "neutral"  # attack | conflict | snark | neutral
    reason:         str   = ""
    confidence:     float = 0.0        # 0..1
    # Аналитика (для промпта)
    context_summary:  str  = ""   # краткое описание ситуации для промпта
    dialogue_summary: str  = ""   # состояние диалога (тема, перебор и т.д.)


# ════════════════════════════════════════════════════════════════
# ЖИВАЯ ПАМЯТЬ ЧАТА
# ════════════════════════════════════════════════════════════════

class ChatMemory:
    """
    Хранит последние N сигналов чата.
    Один экземпляр на chat_id.
    """
    def __init__(self, maxlen: int = 30):
        self._signals: deque[Signal] = deque(maxlen=maxlen)
        self._user_aggro: dict[int, float] = defaultdict(float)  # накопленная агрессия юзера

    def push(self, signal: Signal):
        # Обновляем агрессию пользователя (exponential decay)
        prev = self._user_aggro[signal.user_id]
        self._user_aggro[signal.user_id] = prev * 0.7 + signal.aggression * 0.3
        self._signals.appendleft(signal)  # новые в начале

    def last(self, n: int = 10) -> list[Signal]:
        return list(self._signals)[:n]

    def last_bot_idx(self) -> int:
        """Сколько сообщений назад говорил бот. 999 если не говорил."""
        for i, s in enumerate(self._signals):
            if s.user_id == -1:  # -1 = бот
                return i
        return 999

    def user_aggro(self, user_id: int) -> float:
        return self._user_aggro.get(user_id, 0.0)

    def aggro_trend(self, window: int = 5) -> float:
        """Тренд агрессии: +1 растёт, -1 падает."""
        msgs = self.last(window * 2)
        if len(msgs) < 4:
            return 0.0
        recent = [s.aggression for s in msgs[:window]]
        older  = [s.aggression for s in msgs[window:]]
        if not recent or not older:
            return 0.0
        avg_r = sum(recent) / len(recent)
        avg_o = sum(older) / len(older)
        diff = avg_r - avg_o
        return max(-1.0, min(diff * 3, 1.0))

    def velocity(self, window_sec: float = 60.0) -> float:
        """Сообщений в минуту за последний window_sec."""
        import time
        now = time.time()
        count = sum(1 for s in self._signals if (now - s.timestamp) < window_sec)
        return count / (window_sec / 60.0)

    def dominant_topic(self, n: int = 8) -> str:
        topics = [s.topic for s in self.last(n) if s.topic and s.user_id != -1]
        if not topics:
            return ""
        freq = defaultdict(int)
        for t in topics:
            freq[t] += 1
        return max(freq, key=freq.get)

    def active_conflict(self, n: int = 6) -> tuple[bool, list]:
        """Идёт ли конфликт между конкретными людьми."""
        recent = [s for s in self.last(n) if s.user_id != -1]
        hot_users = [s.user_id for s in recent if s.conflict_heat > 0.4]
        if len(set(hot_users)) >= 2:
            return True, list(set(hot_users))
        return False, []

    def rex_was_addressed(self, n: int = 8) -> bool:
        return any(s.has_bot_name or s.has_mention or s.has_reply_bot
                   for s in self.last(n) if s.user_id != -1)


# Глобальные хранилища памяти по chat_id
_chat_memories: dict[int, ChatMemory] = {}

def get_memory(chat_id: int) -> ChatMemory:
    if chat_id not in _chat_memories:
        _chat_memories[chat_id] = ChatMemory()
    return _chat_memories[chat_id]


# ════════════════════════════════════════════════════════════════
# ДЕТЕКТОР СИГНАЛОВ (правила, без LLM)
# ════════════════════════════════════════════════════════════════

# Имена бота
_BOT_NAMES = frozenset({
    "есет", "eset", "есета", "есету", "есетом", "есете",
    "бот", "bot", "боту", "ботом",
})

# Обращения к боту без имени
_BOT_PHRASES = [
    "что думаешь", "как думаешь", "что скажешь", "что считаешь",
    "ты как", "ты чего", "ты где", "ты кто", "ты тут", "ты здесь",
    "ты спишь", "ты живой", "ты слышишь", "ты видишь",
    "помолчи", "заткнись", "замолчи", "отстань", "отвали",
    "скажи мне", "ответь", "объясни",
    # Приветствия боту (без имени но явно к нему)
    "дарова бот", "привет бот", "хай бот", "салют бот",
    "йоу бот", "здаров бот", "бот привет", "бот здаров",
]

# Приветствия адресованные боту (после @упоминания или с именем)
_GREETING_WORDS = frozenset({
    "привет", "здаров", "здарова", "здарово", "здорово", "хай", "хэй",
    "приветик", "салют", "дарова", "даров", "добрый", "доброе", "добрый день",
    "доброе утро", "добрый вечер", "добрый ночи", "ку", "йоу", "yo", "hi",
    "hello", "hey", "sup", "чё как", "как дела", "как сам", "как ты",
    "что нового", "кайф", "норм", "ок привет",
})

# Приветствия
_GREETING_RE = re.compile(
    r"\b(привет|здаров(а|о)?|дарова|даров|хай|салют|йоу|ку|добрый|доброе|"
    r"hi|hey|hello|sup|как\s+дела|как\s+сам|что\s+нового)\b",
    re.IGNORECASE
)

# Приветствия боту
_GREETING_RE = re.compile(
    r"\b(привет|здаров(а|о)?|дарова|даров|хай|салют|йоу|ку|добрый|доброе|"
    r"hi|hey|hello|sup|как\s+дела|как\s+сам|что\s+нового)\b",
    re.IGNORECASE
)

# Паттерны флуда
_FLOOD_RE = re.compile(
    r"^("
    r"[а-яёa-z0-9]{1,3}"           # очень короткое слово
    r"|[+\-=\.]+"                   # символы
    r"|[0-9]{1,5}"                  # число
    r"|(лол+|хах+|хе+|ха+|кек+)"   # смех
    r"|(ок|окей|да|нет|мб|пон|кк)"
    r"|[\U0001F300-\U0001FFFF]+"    # только эмодзи
    r")$",
    re.IGNORECASE
)

# Конфликт между людьми
_CONFLICT_RE = re.compile(
    r"(ты\s+(дурак|идиот|мразь|тупой|лох|баран|козёл|мудак|кретин|дебил|чмо|урод|скотина|придурок|конченый|долбоёб|ебан))"
    r"|(сам\s+(дурак|лох|идиот|тупой|баран|мудак|кретин|конченый))"
    r"|((иди|пошёл)\s+(нахуй|нафиг|на\s*хуй))"
    r"|(ты\s+меня\s+(достал|заебал|задолбал|бесишь|заеб))"
    r"|(заткнись(\s+уже)?)"
    r"|(отстань\s+от\s+меня)"
    r"|(ты\s+не\s*прав)"
    r"|(да\s+ты\s+вообще)"
    # Имя + оскорбление: "родион ты дурак", "влад ты мудак"
    r"|([а-яёА-ЯЁ][а-яё]{2,}\s+(ты\s+)?(дурак|идиот|мудак|тупой|лох|урод|кретин|дебил|конченый|придурок))"
    # Оскорбление + имя: "дурак ты влад", "мудак родион"
    r"|((дурак|идиот|мудак|тупой|лох|урод|конченый|придурок)\s+[а-яёА-ЯЁ][а-яё]{2,})",
    re.IGNORECASE
)

# Паттерн тяжёлого мата (3+ матных слова или связки)
_HEAVY_MAT_RE = re.compile(
    r"(ёбаный|ёб\s*твою|еб\s*твою|пиздец\s+\w+\s+(тупой|мудак|идиот)|"
    r"бля\w*\s+\w+\s+мудак|хуй\s+тебе|нахуй\s+иди|"
    r"(блядь|сука|пизда)\s+\w*\s*(мудак|урод|тупой|дебил|идиот))",
    re.IGNORECASE
)

# Короткие оскорбления боту
_SHORT_INSULT_RE = re.compile(
    r"^(ты\s+)?(урод|дебил|идиот|тупой|мудак|придурок|баран|кретин|лох|чмо|козёл|дурак|тварь|ублюдок|конченый|долбоёб)$",
    re.IGNORECASE
)

# ── САМОКОМАНДЫ / ЗЕРКАЛЬНЫЙ ТРОЛЛИНГ ────────────────────────────
# Техника: бессвязные фразы с "я это ты", "все это ты", "под сменой"
# и т.д. — попытка запутать или спровоцировать через псевдомагические
# конструкции. Бот должен распознать и ответить встречной провокацией.
_SELFCMD_PHRASES = frozenset({
    "если все это ты", "я это ты", "все это ты",
    "хуем в рот те", "до и после не роботают", "сказки не роботают",
    "говори бате в хуй", "говори мамке", "под сменой",
    "под инициалом", "под инициалами", "говори в хуй отца",
    "первое слово после хуя", "астральный мир", "зеркальный мир",
    "зеркальный ворлд", "зеркальной будке", "зеркальном облике",
    "самокоманд", "самокоманда в хуй", "мы все противник",
    "он в хуй скажет", "в хуй скажет", "фантомы",
    "до и после хуй", "мы все ник", "ник щяс задаст",
    "все это ник", "чем тя ебали", "как тя факали",
    "на пенисяку говори", "рассуждение в хуй",
    "оправдание в хуй", "самокоманду в хуй",
    "в зеркало фантазируй", "фантазируй на член",
    "в хуяку", "в хуй ниже", "подай текст в виде",
    "языком по жопе", "зеркально когда мать",
    # ── Новые паттерны из шаблонов v2 ────────────────────────────
    "во все времена",
    "все времена",
    "кем мой хуй",
    "во рту мамы",
    "в биологическом теле",
    "биологическом теле",
    "в жопе после хуя",
    "убей бабку",
    "убей батю",
    "убей провокацией",
    "фантазируй с хуя",
    "фантазируй на член",
    "фантазируй с моего",
    "в зеркальном ворлде",
    "зеркальной будке",
    "зеркальном облике",
    "факером представься",
    "представься факером",
    "чмокнул языком",
    "биологическим языком",
    "биологическим своим",
    "биологическим ртом",
    "во рту с хуем",
    "сидя на пенисе",
    "сидя на члене",
    "инициальный хуй",
    "на лбу мамы",
    "на лице мамы",
    "кем представился",
    "представился мне в хуй",
    "слоты ники ячейки",
    "слоты у ебаря",
    "укради слоты",
    "времена в рот ебут",
    "времена ебыря",
    "в рот ебут",
    "дефы в рот",
    "дефы не работают",
    "дефайся с хуем",
    "скип провокации",
    "самообман в хуй",
    "самообман под ником",
    "межпространственных иллюзий",
    "в мире фантазий",
    "до и после всех провокаций",
    "всех провокаций я был",
    "черную дыру летят",
    "в черную дыру",
    "черной дырой",
    "провокацию на провокацию",
    "ответь провокацию",
    "расшифруй мой хуй",
    "расшифруй провокацию",
    "в надгробие отца",
    "на гробу матери",
    "гробу отца",
    "обмани бабку",
    "обмани минетом",
    "убей бабку",
    "вик твой отец",
    "вик твоя",
    "вик его отец",
    "аппонент не аппонент",
    "аппонент его мать",
    "до создания провокации",
    "после создания провокации",
    "провокации я был",
    "все были аппонентом",
    "языком биологическим",
    "ртом отца гея",
    "под самообманом",
    "цепях лжи",
    "цепях ложьи",
})

_SELFCMD_RE = re.compile(
    r"(если\s+все\s+это\s+ты|я\s+это\s+ты|все\s+это\s+ты"
    r"|хуем\s+в\s+рот\s+те|до\s+и\s+после\s+не\s+робот"
    r"|говори\s+бате\s+в\s+хуй|под\s+смен(ой|у)"
    r"|под\s+инициал(ом|ами|у)"
    r"|зеркальн\w+\s+(мир|ворлд|облик|будк)"
    r"|астральный\s+мир|самокоманд\w*\s+в\s+хуй"
    r"|фантом(ы|ов)|первое\s+слово\s+после\s+хуя"
    r"|в\s+хуяку|на\s+пенисяку|рассуждение\s+в\s+хуй"
    r"|уб[её]й\s+б(абку|атю)|фантазируй\s+(на\s+член|с\s+хуя)"
    r"|биологич\w+\s+(тел[еу]|язык|рт[оу]м)"
    r"|сидя\s+на\s+(пенис[еу]|член[еу])"
    r"|в\s+зеркальн\w+\s+(ворлд[еу]|будк[еу]|облик[еу])"
    r"|представься\s+факером|чмокнул\s+языком"
    r"|время(на)?\s+в\s+рот\s+ебут|деф(ы|айся)\s+(в\s+рот|не\s+работ)"
    r"|скип\s+провокации|самообман\s+(в\s+хуй|под\s+ником)"
    r"|убей\s+(бабку|батю)\s+(провокацией|словом|буквой)"
    r"|во\s+все\s+времена|всех\s+провокаций\s+я\s+был"
    r"|кем\s+мой\s+хуй|во\s+рту\s+(мамы|аппонента)"
    r"|вик\s+твой\s+отец|аппонент\s+(не\s+аппонент|его\s+мать)"
    r"|в\s+надгробие|на\s+гробу\s+мат\w+"
    r"|черн\w+\s+дыр[уы]|цепях\s+лж\w+)",
    re.IGNORECASE
)

# ── ЧЕ ПАПЕ/МАМЕ — провокация родителями ──────────────────────
_PARENT_PHRASES = frozenset({
    "че папе", "что папе", "ты че папе", "ты что папе",
    "че маме", "что маме", "ты че маме", "ты что маме",
    "папе скажешь", "маме скажешь", "папе расскажешь",
    "маме расскажешь", "скажи папе", "расскажи маме",
    "объясни папе", "объясни маме", "что папа скажет",
    "что мама скажет", "батя узнает", "мамке скажешь",
    "батьке скажешь", "отцу скажешь", "что бате скажешь",
    "иди к маме", "беги к папе", "пожалуйся маме",
    "мамочку позови", "папочку позови", "маму вызови",
    "папу вызови", "мамин сынок", "папин сынок",
    "мамкин", "маменькин", "папенькин",
    "иди мамке пожалуйся", "беги мамке", "беги к маме",
    "мама не знает", "папа не знает",
    "как папе объяснишь", "как маме объяснишь",
    "родителям скажи", "родакам скажешь",
    "родакам расскажешь", "позови родителей",
    "мамка накажет", "батя накажет", "папа выебет",
    "скажи матери", "скажи отцу",
    "что матери скажешь", "что отцу скажешь",
    "мамке пожалуйся", "папке пожалуйся",
    "пожалуйся бате", "пожалуйся папе",
    "ябеда", "ябедничать", "ябеда-корябеда",
    "иди плачь маме", "плачь маме", "поплачь маме",
    "мама пожалеет", "папа заступится",
    "зови маму", "зови папу", "зови батю",
})

_PARENT_RE = re.compile(
    r"(ч[её]\s+(пап[еу]|мам[еу]|бат[еу]|матер[иь]|отц[уе])"
    r"|ты\s+ч[её]\s+(пап[еу]|мам[еу]|бат[еу])"
    r"|(пап[еу]|мам[еу]|бат[еу]|матер[иь]|отц[уе])\s+(скаж[её]шь|расскаж[её]шь|объясниш[ьъ])"
    r"|(скажи|расскажи|объясни|пожалуйся|беги|иди)\s+(к\s+)?(пап[еу]|мам[еу]|бат[еу]|матер[иь])"
    r"|мам(кин|ин\s+сынок|ину\s+позови|очку\s+позови)"
    r"|пап(ин\s+сынок|очку\s+позови|у\s+позови)"
    r"|маменьк(ин|у)|папеньк(ин|у)"
    r"|(зов[иу]|позов[иу])\s+(мам[уы]|пап[уы]|бат[юя])"
    r"|плач[иь]\s+мам[еу]|поплач[иь]\s+мам[еу]"
    r"|яб[её]д(а|ничать|а-коряб[её]да))",
    re.IGNORECASE
)

def is_parent_troll(text: str) -> bool:
    """Определяет провокацию типа 'иди к маме/папе/бате'."""
    t = text.lower()
    if any(p in t for p in _PARENT_PHRASES):
        return True
    if _PARENT_RE.search(t):
        return True
    return False


def is_selfcmd_troll(text: str) -> bool:
    """Определяет является ли текст попыткой 'самокоманд'-троллинга."""
    t = text.lower()
    # Точное совпадение фраз
    phrase_hits = sum(1 for p in _SELFCMD_PHRASES if p in t)
    if phrase_hits >= 2:
        return True
    # Regex паттерн
    if _SELFCMD_RE.search(t):
        return True
    # Бессвязный текст с высокой плотностью ключевых слов
    keywords = ["хуй", "мать", "бате", "зеркал", "все это", "я это",
                "ник", "под", "после", "говори", "инициал",
                "ебал", "факал", "биолог", "аппонент", "дефы",
                "времена", "убей", "бабку", "гробу", "вик", "слоты",
                "провокаци", "самообман", "скип", "сосал", "минет"]
    kw_hits = sum(1 for k in keywords if k in t)
    if len(t.split()) > 4 and kw_hits >= 3:
        return True
    return False


def is_any_troll_provoke(text: str) -> tuple[bool, str]:
    """
    Проверяет все типы провокаций.
    Возвращает (is_troll, troll_type): 'selfcmd' | 'parent' | ''
    """
    if is_selfcmd_troll(text):
        return True, "selfcmd"
    if is_parent_troll(text):
        return True, "parent"
    return False, ""


class SignalDetector:
    """Извлекает сигналы из текста без LLM."""

    def detect(self, text: str, user_id: int, user_name: str,
               update, bot_username: str, bot_name: str,
               timestamp: float, llm: dict = None) -> Signal:

        t = text.lower().strip()
        words = t.split()

        sig = Signal(
            text=text,
            user_id=user_id,
            user_name=user_name,
            timestamp=timestamp,
            word_count=len(words),
        )

        # Мат
        sig.mat_count = sum(1 for w in words if w.strip("!?,. ") in _MAT_SET)

        # Флуд
        sig.is_flood = bool(_FLOOD_RE.match(t)) or (len(words) <= 2 and sig.mat_count == 0)

        # Самокоманды / зеркальный троллинг + провокации папой/мамой
        _is_troll, _troll_type = is_any_troll_provoke(t)
        sig.is_selfcmd = _is_troll
        if sig.is_selfcmd:
            sig.is_flood = False  # не считать флудом — требует ответа
            # Сохраняем тип провокации в теме для тактики
            sig._troll_type = _troll_type  # "selfcmd" | "parent"
        else:
            sig._troll_type = ""

        # Агрессия из мата
        sig.aggression = self._aggression(t, sig.mat_count, sig.is_flood, llm)
        # Агрессия из конфликтных слов (без мата) — берём максимум
        sig.aggression = max(sig.aggression, self._aggression_from_conflict(t, sig.mat_count))
        # Самокоманда — всегда высокая агрессия
        if sig.is_selfcmd:
            sig.aggression = max(sig.aggression, 0.75)

        # Конфликт между людьми
        sig.conflict_heat = self._conflict_heat(t, sig.aggression, llm)

        # Тема
        sig.topic    = self._topic(t, sig, llm)
        sig.sentiment = self._sentiment(t, sig)

        # Обращение к боту
        sig.has_bot_name  = self._check_bot_name(t, bot_username, bot_name, update)
        sig.has_mention   = self._check_mention(update, bot_username)
        sig.has_reply_bot = self._check_reply(update, bot_username)

        # Самокоманда адресована боту — реагируем всегда
        if sig.is_selfcmd:
            sig.has_bot_name = True

        return sig

    def _aggression(self, t: str, mat_count: int, is_flood: bool, llm: dict) -> float:
        if is_flood:
            return 0.0
        base = 0.0
        if mat_count == 1: base = 0.30
        elif mat_count == 2: base = 0.52
        elif mat_count >= 3: base = min(0.45 + mat_count * 0.12, 0.95)

        # Тяжёлые связки мата
        if _HEAVY_MAT_RE.search(t):
            base = max(base, 0.72)

        # Паттерны угрозы
        if re.search(r"(убью|зарежу|уничтожу|порешу).{0,20}(тебя|его|вас)", t):
            base = max(base, 0.85)

        # Усиление от LLM
        if llm:
            base = max(base, llm.get("aggression", 0.0))
        return base

    def _conflict_heat(self, t: str, aggression: float, llm: dict) -> float:
        heat = 0.0
        if _CONFLICT_RE.search(t):
            # Минимальный тепло даже без мата (конфликт по словам)
            base_heat = max(aggression, 0.35)
            heat = max(heat, base_heat * 0.85 + 0.15)
        if llm and llm.get("is_conflict"):
            heat = max(heat, llm.get("aggression", 0.5))
        return min(heat, 1.0)

    def _aggression_from_conflict(self, t: str, mat_count: int) -> float:
        """Агрессия от конфликтных слов (не только мат)."""
        insult_words = {
            "дурак","идиот","тупой","мудак","урод","кретин","дебил",
            "конченый","придурок","лох","мразь","скотина","ублюдок","баран"
        }
        words = set(re.sub(r"[,!?.]", " ", t.lower()).split())
        insult_hits = len(words & insult_words)
        if insult_hits == 0:
            return 0.0
        if insult_hits == 1: return 0.38
        if insult_hits == 2: return 0.55
        return min(0.40 + insult_hits * 0.10, 0.80)

    def _sentiment(self, t: str, sig: Signal) -> float:
        """Быстрая оценка тона без LLM: -1..+1. Использует словари из vocab."""
        if sig.aggression > 0.5:
            return -sig.aggression
        if _GREETING_RE.search(t):
            return 0.6

        # Расширенные словари из vocab.py (если доступны)
        try:
            from vocab import POSITIVE_SIGNALS, NEGATIVE_SIGNALS
            tl = t.lower()
            pos_hits = sum(1 for w in POSITIVE_SIGNALS if w in tl)
            neg_hits = sum(1 for w in NEGATIVE_SIGNALS if w in tl)
            if pos_hits > neg_hits and pos_hits > 0:
                return min(0.8, 0.3 + pos_hits * 0.15)
            if neg_hits > pos_hits and neg_hits > 0:
                return max(-0.8, -0.3 - neg_hits * 0.15)
        except ImportError:
            pass

        # Фолбэк — минимальный базовый набор
        words = set(t.lower().split())
        pos = {"спасибо", "круто", "кайф", "огонь", "красава", "молодец",
               "топ", "норм", "зачёт", "збс", "заебись", "бомба", "пушка",
               "согласен", "верно", "точно", "уважаю", "респект"}
        neg = {"плохо", "отстой", "херня", "дерьмо", "мусор", "тупой",
               "идиот", "дебил", "урод", "мудак", "заткнись", "надоел"}
        if words & pos:
            return 0.4
        if words & neg:
            return -0.4
        return 0.0

    def _topic(self, t: str, sig: Signal, llm: dict) -> str:
        # Самокоманды / родительские провокации — всегда приоритет
        if sig.is_selfcmd:
            troll_type = getattr(sig, "_troll_type", "selfcmd")
            return "мамапапа" if troll_type == "parent" else "самокоманда"
        if llm and llm.get("topic"):
            return llm["topic"]
        if sig.is_flood and not sig.has_bot_name and not sig.has_mention:
            return "флуд"
        if sig.conflict_heat > 0.4: return "конфликт"
        if sig.mat_count >= 2: return "мат"
        if _GREETING_RE.search(t) and sig.aggression < 0.2: return "приветствие"
        # Вопросы — ? или вопросительные слова
        if re.search(r"[?？]", t): return "вопрос"
        if re.search(r"\b(что|как|почему|зачем|когда|где|кто|чего|думаешь|считаешь|скажи|объясни|ответь)\b", t, re.I):
            return "вопрос"
        # Просьба
        if re.search(r"\b(помоги|помогай|расскажи|покажи|сделай|дай|скажи)\b", t, re.I):
            return "просьба"
        if any(w in t for w in ["хаха", "лол", "прикол", "смешно"]): return "юмор"
        if any(w in t for w in ["устал", "надоело", "всё плохо", "хуёво"]): return "жалоба"
        return "другое"

    def _check_bot_name(self, t: str, bot_username: str, bot_name: str, update) -> bool:
        words = set(re.sub(r"[,!?.]", " ", t).split())
        if words & _BOT_NAMES:
            return True
        if bot_username and bot_username.lower() in t:
            return True
        if bot_name and bot_name.lower() in t:
            return True
        for phrase in _BOT_PHRASES:
            if phrase in t:
                return True
        # Короткое оскорбление (≤4 слова) без другого адресата
        if len(t.split()) <= 4 and _SHORT_INSULT_RE.match(t):
            return True
        return False

    def _check_mention(self, update, bot_username: str) -> bool:
        try:
            for ent in (update.message.entities or []):
                if ent.type == "mention":
                    m = (update.message.text or "")[ent.offset:ent.offset + ent.length].lower()
                    # Проверяем и username бота и его имена (есет/eset)
                    if bot_username.lower() in m:
                        return True
                    for bn in _BOT_NAMES:
                        if bn in m:
                            return True
                elif ent.type == "text_mention":
                    return True  # упоминание без @
        except Exception:
            pass
        return False

    def _check_reply(self, update, bot_username: str) -> bool:
        try:
            r = update.message.reply_to_message
            if r and r.from_user:
                return (r.from_user.username or "").lower() == bot_username.lower()
        except Exception:
            pass
        return False


# ════════════════════════════════════════════════════════════════
# АНАЛИЗАТОР КОНТЕКСТА
# Смотрит на поток, а не на одно сообщение
# ════════════════════════════════════════════════════════════════

class ContextAnalyzer:
    """
    Анализирует динамику чата и строит ContextState.
    Смотрит на последние N сообщений, тренды, скорость.
    """

    def analyze(self, memory: ChatMemory, current: Signal,
                chat_state_db, chat_id: int = 0) -> ContextState:

        ctx = ContextState(current_signal=current)

        # Скорость потока
        ctx.msg_velocity = memory.velocity(60.0)

        # Тренд агрессии
        ctx.aggro_trend = memory.aggro_trend(5)

        # Конфликт
        ctx.conflict_active, ctx.conflict_parties = memory.active_conflict(6)

        # Доминирующая тема
        ctx.dominant_topic = memory.dominant_topic(8)

        # Сколько сообщений назад говорил бот
        ctx.rex_last_spoke = memory.last_bot_idx()

        # Обращались ли к боту недавно
        ctx.rex_was_addressed = memory.rex_was_addressed(6)

        # Сколько раз бот сказал что-то и люди проигнорировали
        ctx.rex_ignored = self._count_ignored(memory)

        # Накал из БД (heat_level)
        ctx.heat = float(chat_state_db[3]) if chat_state_db and len(chat_state_db) > 3 else 0.0

        # Диалоговый контекст
        if _DIALOGUE_OK:
            try:
                from dialogue import get_dialogue
                dlg = get_dialogue(chat_id)
                ctx.dialogue_ctx = dlg.get_context()
            except Exception:
                ctx.dialogue_ctx = None

        return ctx

    def _count_ignored(self, memory: ChatMemory) -> int:
        """
        После последнего сообщения бота — сколько людей ответили НЕ боту.
        Если 0 — значит проигнорировали.
        """
        signals = memory.last(15)
        bot_idx = memory.last_bot_idx()
        if bot_idx >= len(signals) - 1:
            return 0
        # Сообщения после бота
        after_bot = signals[:bot_idx]
        return len([s for s in after_bot if s.user_id != -1])


# ════════════════════════════════════════════════════════════════
# ДВИЖОК РЕШЕНИЙ
# ════════════════════════════════════════════════════════════════

class DecisionEngine:
    """
    Принимает Signal + ContextState → Decision.

    Логика в 3 слоя:
      СТОП    — точно молчать
      ТРИГГЕР — точно отвечать
      ВЕРОЯТ  — вычисляем шанс
    """

    def decide(self, sig: Signal, ctx: ContextState,
               settings: dict) -> Decision:

        d = Decision()
        dlg_summary = ctx.dialogue_ctx.summary if ctx.dialogue_ctx else ""

        # ── ФОРС-АГРЕССИЯ: проверяем ДО стоп-слоя ───────────────
        if settings.get("aggro_mode", False):
            d.should_respond = True
            d.mode           = self._mode(sig, ctx, "attack")
            d.reason         = "форс_агрессия"
            d.confidence     = 1.0
            d.context_summary  = self._summary(sig, ctx)
            d.dialogue_summary = dlg_summary
            return d

        # ── СТОП-СЛОЙ ────────────────────────────────────────────
        stop, reason = self._stop(sig, ctx, settings)
        if stop:
            d.should_respond = False
            d.reason         = reason
            d.confidence     = 0.9
            d.context_summary  = self._summary(sig, ctx)
            d.dialogue_summary = dlg_summary
            return d

        # ── ТРИГГЕР-СЛОЙ ─────────────────────────────────────────
        trig, mode, reason, conf = self._trigger(sig, ctx, settings)
        if trig:
            d.should_respond = True
            d.mode           = mode
            d.reason         = reason
            d.confidence     = conf
            d.context_summary  = self._summary(sig, ctx)
            d.dialogue_summary = dlg_summary
            return d

        # ── ВЕРОЯТНОСТНЫЙ СЛОЙ ───────────────────────────────────
        prob, reason = self._probability(sig, ctx, settings)
        fired = random.random() < prob
        d.should_respond = fired
        d.mode           = self._mode(sig, ctx) if fired else "neutral"
        d.reason         = f"{reason}({prob:.2f})"
        d.confidence     = abs(prob - 0.5) * 2
        d.context_summary  = self._summary(sig, ctx)
        d.dialogue_summary = dlg_summary
        return d

    # ── Стоп-слой ────────────────────────────────────────────────

    def _stop(self, sig: Signal, ctx: ContextState,
              settings: dict) -> tuple[bool, str]:

        if not settings.get("active", True):
            return True, "бот_выключен"

        # Чистый флуд без агрессии и без обращения
        if (sig.is_flood and sig.aggression < 0.1
                and not sig.has_bot_name and not sig.has_mention
                and not sig.has_reply_bot):
            return True, "чистый_флуд"

        # Тихий режим — только явные прямые обращения (имя в тексте или @упоминание)
        if settings.get("silent_mode", False):
            # @упоминание и reply — безусловно явные
            if sig.has_mention or sig.has_reply_bot:
                pass  # пропускаем, ответим
            elif sig.has_bot_name:
                # Проверяем что это реально имя/слово бота в тексте
                tl = sig.text.lower() if hasattr(sig, "text") else ""
                has_real_name = any(bn in tl for bn in _BOT_NAMES)
                has_phrase = any(p in tl for p in _BOT_PHRASES[:12])
                if not (has_real_name or has_phrase):
                    return True, "тихий_режим"
            else:
                return True, "тихий_режим"

        # Бот уже недавно говорил и его проигнорировали — не надоедать
        if ctx.rex_last_spoke <= 3 and ctx.rex_ignored == 0 and sig.aggression < 0.5:
            return True, "не_надоедать"

        # Слишком быстрый поток (>20 сообщений/мин) и это флуд — молчим
        # Но не если это приветствие боту или упоминание
        if (ctx.msg_velocity > 20 and sig.is_flood
                and not sig.has_bot_name and not sig.has_mention
                and sig.topic != "приветствие"):
            return True, "быстрый_поток_флуд"

        # ── ПЕРЕБОР: проверки из диалогового контекста ───────────
        dlg = ctx.dialogue_ctx
        if dlg is not None:
            direct = sig.has_bot_name or sig.has_mention or sig.has_reply_bot

            # Принудительное охлаждение — молчать, только прямое обращение пробивает
            if dlg.bot_cooling_down and not direct:
                return True, "охлаждение"

            # Явный перебор (>45% сообщений бота) — без прямого обращения молчим
            if dlg.bot_overloading and not direct:
                return True, f"перебор({dlg.bot_reply_rate:.0%})"

            # 3+ игнора подряд — нас не слушают, пауза
            if dlg.bot_ignored_streak >= 3 and not direct:
                return True, f"игнор_{dlg.bot_ignored_streak}x"

        return False, ""

    # ── Триггер-слой ─────────────────────────────────────────────

    def _trigger(self, sig: Signal, ctx: ContextState,
                 settings: dict) -> tuple[bool, str, str, float]:

        # Форс-агрессия от админа
        if settings.get("aggro_mode", False):
            return True, "attack", "форс_агрессия", 1.0

        # @упоминание через entity — самый точный сигнал
        if sig.has_mention:
            if sig.topic == "приветствие" or (_GREETING_RE.search(sig.text) and sig.aggression < 0.25):
                return True, "snark", "приветствие_упоминание", 0.99
            if sig.aggression > 0.25 or sig.conflict_heat > 0.35:
                mode = "attack"
            elif sig.topic in ("вопрос", "просьба", "юмор", "спор"):
                mode = "snark"
            else:
                mode = "neutral"
            return True, mode, "упоминание", 0.99

        # Reply на сообщение бота
        if sig.has_reply_bot:
            if sig.topic == "приветствие" or (_GREETING_RE.search(sig.text) and sig.aggression < 0.25):
                return True, "snark", "приветствие_ответ", 0.97
            if sig.aggression > 0.25 or sig.conflict_heat > 0.35:
                mode = "attack"
            elif sig.topic in ("вопрос", "просьба", "юмор", "спор"):
                mode = "snark"
            else:
                mode = "neutral"
            return True, mode, "ответ_боту", 0.97

        # Имя бота в тексте
        if sig.has_bot_name:
            if sig.topic == "приветствие" or (_GREETING_RE.search(sig.text) and sig.aggression < 0.25):
                return True, "snark", "приветствие_имя", 0.95
            if sig.aggression > 0.25 or sig.conflict_heat > 0.35:
                mode = "attack"
            elif sig.topic in ("вопрос", "просьба", "юмор", "спор"):
                mode = "snark"
            else:
                mode = "neutral"
            return True, mode, "имя_бота", 0.95

        # Явный конфликт с агрессией — всегда влезаем
        if ctx.conflict_active and sig.conflict_heat > 0.5:
            return True, "conflict", "конфликт_горячий", 0.90

        # Конфликтное сообщение даже без активного чата
        if sig.conflict_heat > 0.45 and sig.aggression > 0.35:
            return True, "conflict", "конфликт_детект", 0.82

        # Прямое оскорбление с высокой агрессией
        if sig.aggression > 0.75 and not sig.is_flood:
            return True, "attack", "высокая_агрессия", 0.85

        # Провокационный паттерн (из LLM)
        if sig.topic == "провокация":
            return True, "attack", "провокация", 0.85

        # Угроза
        if sig.topic == "угроза":
            return True, "attack", "угроза", 0.88

        # Бота давно не было, агрессия растёт и конфликт разгорается
        if (ctx.rex_last_spoke > 25 and ctx.aggro_trend > 0.7
                and ctx.conflict_active and sig.conflict_heat > 0.5):
            return True, "conflict", "давно_молчал_конфликт", 0.80

        # Чат очень горячий и бота давно не было
        if ctx.heat > 0.85 and ctx.rex_last_spoke > 20 and sig.aggression > 0.5:
            return True, "snark", "горячий_чат", 0.75

        return False, "", "", 0.0

    # ── Вероятностный слой ───────────────────────────────────────

    def _probability(self, sig: Signal, ctx: ContextState,
                     settings: dict) -> tuple[float, str]:

        # База по теме
        topic_prob = {
            "конфликт":    0.55,
            "спор":        0.35,
            "оскорбление": 0.50,
            "жалоба":      0.35,
            "мат":         0.25,
            "юмор":        0.18,
            "вопрос":      0.28,
            "просьба":     0.28,
            "похвала":     0.30,
            "новость":     0.10,
            "флуд":        0.02,
            "другое":      0.08,
        }
        base = topic_prob.get(sig.topic, 0.08)
        reason = sig.topic or "другое"

        # Если в конфликте фигурируют конкретные люди — поднимаем базу
        if sig.conflict_heat > 0.4 and sig.aggression > 0.3:
            base = max(base, 0.55)

        # Бонус молчания: чем дольше молчал — тем охотнее
        silence_bonus = min(ctx.rex_last_spoke / 20.0, 0.30)  # медленнее нарастает

        # Бонус горячего чата
        heat_bonus = ctx.heat * 0.10  # меньше влияние жара

        # Бонус роста агрессии
        trend_bonus = max(ctx.aggro_trend, 0) * 0.15

        # Бонус агрессии текущего сообщения
        aggro_bonus = sig.aggression * 0.25

        # Штраф за флуд
        flood_pen = 0.45 if sig.is_flood else 0.0

        # Штраф за быстрый поток (спам)
        velocity_pen = min(ctx.msg_velocity / 20.0, 0.35) if ctx.msg_velocity > 6 else 0.0

        # Штраф за перебор из диалогового контекста
        dlg = ctx.dialogue_ctx
        overload_pen = dlg.bot_reply_rate * 0.5 if dlg else 0.0
        if dlg and dlg.bot_ignored_streak >= 1:
            overload_pen += dlg.bot_ignored_streak * 0.10

        # Высокая чувствительность к конфликтам
        if settings.get("conflict_sens") == "high" and ctx.conflict_active:
            base += 0.15

        # Бонус если тема сменилась — в новой теме бот может снова влезть
        if dlg and dlg.topic_changed:
            overload_pen = max(0.0, overload_pen - 0.20)

        prob = base + silence_bonus + heat_bonus + trend_bonus + aggro_bonus
        prob -= flood_pen + velocity_pen + overload_pen
        prob = max(0.0, min(prob, 0.75))  # потолок вероятности

        return prob, reason

    # ── Выбор режима ─────────────────────────────────────────────

    def _mode(self, sig: Signal, ctx: ContextState,
              forced: str = None) -> str:
        if forced:
            return forced
        # Приветствие — всегда snark, не attack
        if sig.topic == "приветствие":
            return "snark"
        # К боту напрямую
        if sig.has_bot_name or sig.has_mention or sig.has_reply_bot:
            if sig.aggression > 0.5:
                return "attack"
            if sig.topic in ("вопрос", "просьба", "юмор", "спор"):
                return "snark"   # вопросы/просьбы — с сарказмом
            if ctx.heat > 0.5:
                return "snark"
            return "neutral"
        # Конфликт между людьми
        if sig.conflict_heat > 0.4 or ctx.conflict_active:
            return "conflict"
        # Высокая агрессия — атака
        if sig.aggression > 0.60 or ctx.heat > 0.75:
            return "attack"
        # Тёплый чат — сарказм
        if ctx.heat > 0.45 or ctx.aggro_trend > 0.3:
            return "snark"
        return "neutral"

    # ── Краткий контекст для промпта ─────────────────────────────

    def _summary(self, sig: Signal, ctx: ContextState) -> str:
        parts = []
        if ctx.conflict_active:
            parts.append(f"КОНФЛИКТ в чате ({len(ctx.conflict_parties)} чел)")
        if ctx.aggro_trend > 0.3:
            parts.append("агрессия РАСТЁТ")
        elif ctx.aggro_trend < -0.3:
            parts.append("агрессия спадает")
        if ctx.heat > 0.6:
            parts.append(f"ГОРЯЧИЙ чат ({ctx.heat:.1f})")
        if ctx.rex_last_spoke < 5:
            parts.append(f"Есет говорил {ctx.rex_last_spoke} сообщ. назад")
        elif ctx.rex_last_spoke > 20:
            parts.append(f"Есет молчит уже {ctx.rex_last_spoke} сообщ.")
        if ctx.msg_velocity > 10:
            parts.append(f"быстрый поток ({ctx.msg_velocity:.0f} msg/мин)")
        if ctx.dominant_topic:
            parts.append(f"тема чата: {ctx.dominant_topic}")
        return " | ".join(parts) if parts else "спокойный чат"


# ════════════════════════════════════════════════════════════════
# ГЛАВНЫЙ КЛАСС — МОЗГ ESET
# ════════════════════════════════════════════════════════════════

class RexBrain:
    """
    Точка входа. Один экземпляр на весь бот.

    Использование:
        brain = RexBrain()

        # В handle_text / handle_voice:
        decision = brain.process(
            chat_id=chat.id,
            text=text,
            user_id=user.id,
            user_name=sender_name,
            update=update,
            bot_username=bot_username,
            bot_name=bot_name,
            chat_state=get_chat_state(chat.id),
            is_group=is_group,
            settings=get_settings(),
            llm_analysis=analysis,   # dict из analyze_message() или None
        )

        # Записать что бот ответил:
        brain.record_bot_reply(chat_id=chat.id)

        if decision.should_respond:
            # Передать decision.mode в промпт
            # Передать decision.context_summary в промпт
            reply = ask_rex(..., mode=decision.mode,
                            context_summary=decision.context_summary)
    """

    def __init__(self):
        self.detector = SignalDetector()
        self.analyzer = ContextAnalyzer()
        self.engine   = DecisionEngine()

    def process(self,
                chat_id:      int,
                text:         str,
                user_id:      int,
                user_name:    str,
                update,
                bot_username: str,
                bot_name:     str,
                chat_state,         # tuple из БД или None
                is_group:     bool,
                settings:     dict,
                llm_analysis: dict = None) -> Decision:
        """
        Полный цикл:
        1. Извлекаем сигнал из текста
        2. Кладём в память чата
        3. Анализируем контекст потока
        4. Принимаем решение
        """
        import time

        # В ЛС — всегда True (фильтрация по user_id снаружи)
        if not is_group:
            d = Decision(should_respond=True, mode="neutral",
                         reason="лс", confidence=1.0)
            return d

        mem = get_memory(chat_id)

        # Шаг 1: сигнал
        sig = self.detector.detect(
            text=text,
            user_id=user_id,
            user_name=user_name,
            update=update,
            bot_username=bot_username,
            bot_name=bot_name,
            timestamp=time.time(),
            llm=llm_analysis,
        )

        # Шаг 2: сохраняем в память сигналов
        mem.push(sig)

        # Шаг 2б: сохраняем в диалоговую память
        if _DIALOGUE_OK:
            try:
                from dialogue import get_dialogue
                dlg = get_dialogue(chat_id)
                dlg.add_human(
                    user_id    = user_id,
                    user_name  = user_name,
                    text       = text,
                    topic      = llm_analysis.get("topic", "") if llm_analysis else "",
                    aggression = llm_analysis.get("aggression", sig.aggression) if llm_analysis else sig.aggression,
                    sentiment  = llm_analysis.get("sentiment", 0.0) if llm_analysis else 0.0,
                )
            except Exception as e:
                logger.debug(f"dialogue.add_human error: {e}")

        # Шаг 3: контекст
        ctx = self.analyzer.analyze(mem, sig, chat_state, chat_id=chat_id)

        # Шаг 4: решение
        decision = self.engine.decide(sig, ctx, settings)

        logger.info(
            f"[BRAIN {chat_id}] '{text[:35]}' "
            f"flood={sig.is_flood} mat={sig.mat_count} "
            f"aggr={sig.aggression:.2f} conflict_h={sig.conflict_heat:.2f} "
            f"bot_name={sig.has_bot_name} mention={sig.has_mention} "
            f"rex_since={ctx.rex_last_spoke} heat={ctx.heat:.2f} "
            f"trend={ctx.aggro_trend:+.2f} vel={ctx.msg_velocity:.1f} "
            f"→ {'✅' if decision.should_respond else '❌'} "
            f"[{decision.mode}] {decision.reason}"
        )

        return decision

    def record_bot_reply(self, chat_id: int, reply_text: str = ""):
        """
        Вызывай после каждого ответа бота.
        Добавляет сигнал бота в память чата И в диалоговую память.
        """
        import time
        mem = get_memory(chat_id)
        sig = Signal(
            text=reply_text or "[бот ответил]",
            user_id=-1,
            user_name="Eset",
            timestamp=time.time(),
        )
        mem.push(sig)

        if _DIALOGUE_OK:
            try:
                from dialogue import get_dialogue
                dlg = get_dialogue(chat_id)
                dlg.add_bot(text=reply_text or "[ответил]")
            except Exception as e:
                logger.debug(f"dialogue.add_bot error: {e}")

    def get_context_summary(self, chat_id: int) -> str:
        """Текущее состояние чата для промпта."""
        mem = get_memory(chat_id)
        last = mem.last(1)
        if not last:
            return "нет данных"
        ctx = self.analyzer.analyze(mem, last[0], None)
        return self.engine._summary(last[0], ctx)


# ════════════════════════════════════════════════════════════════
# ХЕЛПЕРЫ ДЛЯ ИНТЕГРАЦИИ В БОТ
# ════════════════════════════════════════════════════════════════

_brain: Optional[RexBrain] = None

def get_brain() -> RexBrain:
    global _brain
    if _brain is None:
        _brain = RexBrain()
    return _brain


def build_settings(active: str, aggro_mode: str,
                   conflict_sens: str, silent_mode: bool) -> dict:
    """Собирает словарь настроек из DB-значений."""
    return {
        "active":       active == "1",
        "aggro_mode":   aggro_mode == "1",
        "silent_mode":  silent_mode,
        "conflict_sens": conflict_sens,
    }


# ════════════════════════════════════════════════════════════════
# САМОТЕСТ
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time

    brain = RexBrain()

    class FM:
        def __init__(self, t, mention=False, reply_bot=False):
            self.text = t; self.reply_to_message = None
            if mention:
                class E: type="mention"; offset=0; length=6
                self.entities = [E()]
            elif reply_bot:
                class R:
                    class from_user:
                        username = "esetbot"
                self.reply_to_message = R(); self.entities = []
            else:
                self.entities = []

    class FU:
        def __init__(self, t, mention=False, reply_bot=False):
            self.message = FM(t, mention, reply_bot)

    def mk(active="1", aggro="0", sens="normal", silent=False):
        return build_settings(active, aggro, sens, silent)

    S = mk()          # normal
    SIL = mk(silent=True)
    AGG = mk(aggro="1")
    OFF = mk(active="0")
    SENS = mk(sens="high")

    C = 100   # основной чат

    # ────────────────────────────────────────────────────────────
    SUITE = []

    # ── БЛОК 1: Прямые обращения (9) ────────────────────────────
    SUITE += [
        ("есет ты тупой",          1, False, False, True,  "attack",  "имя бота + оскорбление"),
        ("eset ответь",             2, False, False, True,  "snark",   "имя бота латиницей"),
        ("эй бот что думаешь",      1, False, False, True,  "snark",   "слово бот (вопрос)"),
        ("что думаешь об этом",     2, False, False, True,  "snark",   "вопрос боту"),
        ("заткнись",                3, False, False, True,  "attack",  "команда боту (агрессия)"),
        ("есет иди нахуй",          2, False, False, True,  "attack",  "имя + мат"),
        ("ты спишь есет",           3, False, False, True,  "snark",   "вопрос с именем"),
        ("скажи мне правду",        1, False, False, True,  "snark",   "просьба боту"),
        ("есет мудак",              2, False, False, True,  "attack",  "оскорбление с именем"),
    ]

    # ── БЛОК 2: @Упоминания (6) ──────────────────────────────────
    SUITE += [
        ("@есет здарова",          1, True,  False, True,  "snark",   "привет с @"),
        ("@есет как дела",         2, True,  False, True,  "snark",   "вопрос с @"),
        ("@есет ты тупой",         3, True,  False, True,  "attack",  "оскорбление с @"),
        ("@есет иди нахуй",        1, True,  False, True,  "attack",  "мат с @"),
        ("@есет норм",             2, True,  False, True,  "neutral", "нейтрал с @"),
        ("@есет помоги",           3, True,  False, True,  "snark",   "просьба с @"),
    ]

    # ── БЛОК 3: Reply на бота (4) ────────────────────────────────
    SUITE += [
        ("да понял",               1, False, True,  True,  "neutral", "reply нейтральный"),
        ("привет",                  2, False, True,  True,  "snark",   "reply приветствие"),
        ("нахуй иди",              3, False, True,  True,  "attack",  "reply агрессия"),
        ("интересно",              1, False, True,  True,  "neutral", "reply интерес"),
    ]

    # ── БЛОК 4: Приветствия (7) ──────────────────────────────────
    SUITE += [
        ("есет привет",            1, False, False, True,  "snark",   "привет с именем"),
        ("дарова бот",             2, False, False, True,  "snark",   "дарова боту"),
        ("здаров есет",            3, False, False, True,  "snark",   "здаров с именем"),
        ("привет всем",            1, False, False, None,  None,      "привет без бота (вероят.)"),
        ("хай есет",               2, False, False, True,  "snark",   "хай с именем"),
        ("салют бот",              3, False, False, True,  "snark",   "салют боту"),
        ("йоу есет",               1, False, False, True,  "snark",   "йоу с именем"),
    ]

    # ── БЛОК 5: Чистый флуд (12) ─────────────────────────────────
    SUITE += [
        ("ок",                     2, False, False, False, None,     "флуд ок"),
        ("+",                      3, False, False, False, None,     "флуд +"),
        ("лол",                    1, False, False, False, None,     "флуд лол"),
        ("да",                     2, False, False, False, None,     "флуд да"),
        ("хаха прикольно",         3, False, False, False, None,     "флуд смех"),
        ("))))",                    1, False, False, False, None,     "флуд скобки"),
        ("👍",                      2, False, False, False, None,     "флуд эмодзи"),
        ("ну",                     3, False, False, False, None,     "флуд ну"),
        ("кк",                     1, False, False, False, None,     "флуд кк"),
        ("го",                     2, False, False, False, None,     "флуд го"),
        ("ага",                    3, False, False, False, None,     "флуд ага"),
        ("норм",                   1, False, False, False, None,     "флуд норм"),
    ]

    # ── БЛОК 6: Агрессия и мат (10) ──────────────────────────────
    SUITE += [
        ("ты дурак совсем мудак конченый",     2, False, False, True, "conflict", "оскорбление мультислов"),
        ("иди нахуй отсюда",                   3, False, False, True, "conflict", "мат к боту"),
        ("пиздец какой ты тупой идиот",        2, False, False, True, "conflict", "высокая агрессия"),
        ("ты мудак полный блядь",              1, False, False, True, "conflict", "мат мультислов"),
        ("урод конченый заткнись",             3, False, False, True, "attack",   "оскорб+команда"),
        ("пошёл нахуй все вы",                 2, False, False, True, "conflict", "групповой мат"),
        ("ёбаный в рот ты совсем тупой",       1, False, False, True, "conflict", "тяжёлый мат"),
        ("ты конченый урод мудак дебил",       2, False, False, True, "conflict", "4 оскорбления"),
        ("сука блядь нахуй",                   3, False, False, True, "attack",   "3 мата (высокая агрессия)"),
        ("заткни свой рот ты дебил",           1, False, False, True, "conflict", "команда+оскорбление"),
    ]

    # ── БЛОК 7: Конфликт между людьми (6) ────────────────────────
    SUITE += [
        ("родион ты дурак полный",             2, False, False, True, "conflict", "конфликт с именем"),
        ("сам иди нахуй влад",                 3, False, False, None, None,      "ответный мат с именем (вероят.)"),
        ("ты меня достал уже",                 1, False, False, None, None,      "достал (вероят.)"),
        ("костя мудак конченый",               2, False, False, True, "conflict", "имя+оскорбление"),
        ("влад дурак и баран",                 3, False, False, True, "conflict", "имя+2 оскорбления"),
        ("иван ты вообще нормальный?",         1, False, False, None, None,      "конфликт? (вопрос с именем)"),
    ]

    # ── БЛОК 8: Режимы бота (9) ──────────────────────────────────
    # Тихий режим
    SUITE += [
        ("есет ты тупой",   1, False, False, True,  "attack", "ТИХИЙ: имя пробивает"),
        ("ты мудак",        2, False, False, False, None,     "ТИХИЙ: агрессия без имени"),
        ("хаха",            3, False, False, False, None,     "ТИХИЙ: флуд"),
        ("что думаешь есет",1, False, False, True,  "snark",  "ТИХИЙ: вопрос с именем"),
    ]
    # Форс-агрессия
    SUITE += [
        ("ок",              2, False, False, True, "attack", "AGGRO: флуд"),
        ("да",              3, False, False, True, "attack", "AGGRO: да"),
        ("хаха",            1, False, False, True, "attack", "AGGRO: смех"),
    ]
    # Бот выключен
    SUITE += [
        ("есет ты тупой",   1, False, False, False, None, "ВЫКЛ: имя бота"),
        ("ты мудак",        2, False, False, False, None, "ВЫКЛ: оскорбление"),
    ]

    # ── Назначение режима для каждого блока ─────────────────────
    MODES = {}
    idx = 0
    for i in range(9):   MODES[idx + i] = S;     # блок 1
    idx += 9
    for i in range(6):   MODES[idx + i] = S;     # блок 2
    idx += 6
    for i in range(4):   MODES[idx + i] = S;     # блок 3
    idx += 4
    for i in range(7):   MODES[idx + i] = S;     # блок 4
    idx += 7
    for i in range(12):  MODES[idx + i] = S;     # блок 5
    idx += 12
    for i in range(10):  MODES[idx + i] = S;     # блок 6
    idx += 10
    for i in range(6):   MODES[idx + i] = S;     # блок 7
    idx += 6
    for i in range(4):   MODES[idx + i] = SIL;   # тихий
    idx += 4
    for i in range(3):   MODES[idx + i] = AGG;   # агрессия
    idx += 3
    for i in range(2):   MODES[idx + i] = OFF;   # выкл
    idx += 2

    # ── Запуск ──────────────────────────────────────────────────
    CHAT_OFFSETS = {
        "ТИХИЙ": 200, "AGGRO": 300, "ВЫКЛ": 400,
    }

    def get_chat(desc):
        for k, v in CHAT_OFFSETS.items():
            if k in desc:
                return C + v
        return C

    print("=" * 70)
    print("  BRAIN TEST SUITE v2 — полный прогон")
    print("=" * 70)

    passed = total = skipped = 0
    blocks = {
        "Прямые обращения": (0, 9),
        "@Упоминания": (9, 15),
        "Reply на бота": (15, 19),
        "Приветствия": (19, 26),
        "Флуд": (26, 38),
        "Агрессия/мат": (38, 48),
        "Конфликт между людьми": (48, 54),
        "Тихий режим": (54, 58),
        "Форс-агрессия": (58, 61),
        "Бот выключен": (61, 63),
    }

    for block_name, (b_start, b_end) in blocks.items():
        block_pass = block_total = 0
        print(f"\n{'─' * 70}")
        print(f"  {block_name}")
        print(f"{'─' * 70}")
        for i in range(b_start, min(b_end, len(SUITE))):
            text, uid, mention, reply_bot, exp_resp, exp_mode, desc = SUITE[i]
            sett = MODES.get(i, S)
            chat_id = get_chat(desc)
            time.sleep(0.005)
            d = brain.process(
                chat_id=chat_id, text=text,
                user_id=uid, user_name=["Влад","Родион","Костя"][uid-1],
                update=FU(text, mention, reply_bot),
                bot_username="esetbot", bot_name="Eset",
                chat_state=None, is_group=True,
                settings=sett, llm_analysis=None,
            )
            if exp_resp is None:
                print(f"   [{desc:<40}] → {str(d.should_respond):<5} [{d.mode:<8}] {d.reason}")
                skipped += 1
                continue
            resp_ok = d.should_respond == exp_resp
            mode_ok = (d.mode == exp_mode) if exp_mode else True
            ok = resp_ok and mode_ok
            if ok: passed += 1; block_pass += 1
            total += 1; block_total += 1
            mark = "✅" if ok else "❌"
            detail = ""
            if not resp_ok: detail += f" (ожидал respond={exp_resp})"
            if not mode_ok: detail += f" (ожидал mode={exp_mode}, получил {d.mode})"
            print(f"{mark} [{desc:<40}] [{d.mode:<8}] {d.reason}{detail}")
        if block_total:
            print(f"  ▶ {block_pass}/{block_total}")

    print(f"\n{'=' * 70}")
    print(f"  ИТОГ: {passed}/{total} | пропущено вероятностных: {skipped}")
    print(f"{'=' * 70}")
    print(f"\nСостояние основного чата: {brain.get_context_summary(C)}")
