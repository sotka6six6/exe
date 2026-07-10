import os, logging, tempfile, sqlite3, random, json, re, asyncio, time, dataclasses
from datetime import datetime
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ChatPermissions
from telegram.ext import (
    ApplicationBuilder, MessageHandler, CommandHandler,
    CallbackQueryHandler, filters, ContextTypes
)
import edge_tts
from openai import OpenAI

try:
    from vocab import (get_vocab_for_prompt, get_random_phrase, MAT, SLANG, SARCASM, PROVOCATIONS,
                        FRIENDLY_WORDS, NEUTRAL_WORDS, WARY_WORDS, HOSTILE_WORDS, HATE_WORDS,
                        POSITIVE_SIGNALS, NEGATIVE_SIGNALS,
                        get_attitude_words, detect_attitude_signal,
                        COUNTER_SELFCMD, COUNTER_SELFCMD_SHORT,
                        COUNTER_PARENT, COUNTER_PARENT_SHORT,
                        COUNTER_DEEP, SELFCMD_TEMPLATES)
except Exception as _vocab_err:
    logging.getLogger(__name__).warning(
        f"[vocab] импорт не удался, использую пустые заглушки: {_vocab_err}")
    def get_vocab_for_prompt(): return "лол, кринж, база"
    def get_random_phrase(cat): return ""
    MAT = []; SLANG = []; SARCASM = []; PROVOCATIONS = []
    FRIENDLY_WORDS = []; NEUTRAL_WORDS = []; WARY_WORDS = []
    HOSTILE_WORDS = []; HATE_WORDS = []
    POSITIVE_SIGNALS = frozenset(); NEGATIVE_SIGNALS = frozenset()
    COUNTER_SELFCMD = []; COUNTER_SELFCMD_SHORT = []
    COUNTER_PARENT = []; COUNTER_PARENT_SHORT = []
    COUNTER_DEEP = []; SELFCMD_TEMPLATES = []
    def get_attitude_words(tone, count=5): return []
    def detect_attitude_signal(text): return 0.0, 0.0

from brain import get_brain, build_settings
from analyzer import (init_analyzer, deep_analyze, record_bot_reply as analyzer_record_reply,
                       build_response_instructions, get_semantic_memory)
from learner import (record_reaction, detect_reaction, save_bot_reply_for_learning,
                      learner_loop, get_ai_provocation)
load_dotenv()
TG_TOKEN = os.getenv("TG_TOKEN")
GROQ_KEY = os.getenv("GROQ_API_KEY")
if not TG_TOKEN or not GROQ_KEY:
    print("Нет TG_TOKEN или GROQ_API_KEY!"); exit(1)

import httpx

# Необязательный прокси/VPN для доступа к Groq — если API возвращает 403
# из-за блокировки региона, впиши в .env строку вида:
#   GROQ_PROXY=http://login:password@host:port
GROQ_PROXY = os.getenv("GROQ_PROXY") or os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")

if GROQ_PROXY:
    client = OpenAI(
        api_key=GROQ_KEY,
        base_url="https://api.groq.com/openai/v1",
        http_client=httpx.Client(proxy=GROQ_PROXY, timeout=60.0),
    )
else:
    client = OpenAI(api_key=GROQ_KEY, base_url="https://api.groq.com/openai/v1")


def check_groq_connection() -> bool:
    """
    Проверка доступа к Groq при старте бота.
    Цель — сразу и явно показать ПРИЧИНУ проблемы, а не только "403" в логах
    во время работы. Не роняет бота при ошибке — просто предупреждает,
    LLM-функции продолжат тихо пропускаться (как и раньше), пока не почините.
    """
    try:
        client.models.list()
        print("[Groq] ✅ соединение с API работает")
        return True
    except Exception as e:
        msg = str(e).lower()
        print("=" * 60)
        if "401" in msg or "invalid_api_key" in msg or "unauthorized" in msg:
            print("[Groq] ❌ 401 — ключ GROQ_API_KEY неверный или отозван.")
            print("    Зайди на https://console.groq.com/keys и создай новый ключ,")
            print("    впиши его в .env и перезапусти бота.")
        elif "403" in msg or "forbidden" in msg:
            print("[Groq] ❌ 403 Forbidden — ключ синтаксически верный, но доступ запрещён.")
            print("    Причина почти всегда одна из двух:")
            print("      1) аккаунт Groq заблокирован/ограничен —")
            print("         проверь console.groq.com → Billing/Usage")
            print("      2) Groq блокирует запросы из твоего региона по IP")
            print("    Если похоже на пункт 2 — впиши в .env строку:")
            print("      GROQ_PROXY=http://login:password@host:port")
            print("    (прокси или VPN с выходом из другой страны) и перезапусти.")
        else:
            print(f"[Groq] ❌ Не удалось подключиться: {e}")
            print("    Проверь интернет-соединение на этой машине.")
        print("=" * 60)
        return False


def _llm_call_with_retry(fn, retries=3, base_delay=5.0):
    """
    Обёртка с retry для LLM вызовов.
    429 (rate limit) → ждём и повторяем.
    """
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "rate limit" in msg or "rate_limit" in msg:
                wait = base_delay * (2 ** attempt)  # 5s, 10s, 20s
                logger.warning(f"[LLM] Rate limit (попытка {attempt+1}/{retries}), жду {wait:.0f}с...")
                time.sleep(wait)
            else:
                raise  # остальные ошибки — пробрасываем сразу
    logger.error(f"[LLM] Все {retries} попытки исчерпаны — молчим")
    return None


# LLM вызовы — последовательно, без параллелизма (иначе 429 от Groq)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════
# КОНФИГ
# ════════════════════════════════════════════════════════════════

ADMIN_ID = int(os.getenv("ADMIN_ID", "5772523617"))
ADMIN_IDS = {ADMIN_ID}

def is_admin(uid: int) -> bool:
    return uid in ADMIN_IDS

# Известные участники: ID → имя
KNOWN_USERS = {
    5772523617: "Андрей",
    952340903:  "Родион",
    1254500094: "Костя",
    1603921392: "Иван",
    1617318486: "Влад",
    1889475199: "Ярик",
    5074430991: "Глеб",
    7548968880: "Раф",
}

def get_display_name(user_id: int, fallback_first: str = "", fallback_username: str = "") -> str:
    if user_id in KNOWN_USERS:
        return KNOWN_USERS[user_id]
    return fallback_first or fallback_username or f"user_{user_id}"

# Railway Volume или локальная БД
_volume = os.getenv("RAILWAY_VOLUME_MOUNT_PATH", "")
DB_PATH = os.path.join(_volume, "eset_memory.db") if _volume else "eset_memory.db"


def _connect(timeout: float = 10.0) -> sqlite3.Connection:
    """
    Единая точка подключения к SQLite.
    WAL — чтобы фоновые asyncio-задачи и обработчики сообщений не
    ловили "database is locked" при параллельном доступе.
    timeout — сколько ждать снятия блокировки, а не падать сразу.
    """
    conn = sqlite3.connect(DB_PATH, timeout=timeout)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    return conn

# Глобальный режим бота
BOT_MODE = {"silent": False}

# Дедупликация — последние N сообщений на которые ответили (chat_id -> deque)
from collections import deque
_REPLIED_HASHES: dict[int, deque] = {}

# Последний ответ бота в каждом чате — для learner.py (реакция чата на ответ)
_LAST_BOT_REPLY: dict[int, dict] = {}
_NICKNAME_IN_PROGRESS: set[int] = set()
_REACTION_WINDOW = 180.0  # сек — сколько ждём реакцию на последний ответ бота

def _is_duplicate(chat_id: int, text: str, window: int = 3) -> bool:
    """True если такое же сообщение уже было в последних window ответах."""
    import hashlib
    h = hashlib.md5(text.strip().lower().encode()).hexdigest()[:8]
    q = _REPLIED_HASHES.setdefault(chat_id, deque(maxlen=window))
    if h in q:
        return True
    q.append(h)
    return False


# ════════════════════════════════════════════════════════════════
# РЕАКЦИИ НА СООБЩЕНИЯ
# ════════════════════════════════════════════════════════════════

# Реакции по режиму бота (без LLM, мгновенно)
# ВАЖНО: Telegram принимает реакцию только если эмодзи входит в
# telegram.constants.ReactionEmoji — это не любой юникод-эмодзи,
# а фиксированный список из ~70 штук. Иначе setMessageReaction → 400.
_REACTIONS_BY_MODE = {
    "attack":   ["🖕", "😡", "🤬", "👻", "🗿"],
    "conflict": ["🤨", "👀", "👻", "🔥", "😎"],
    "snark":    ["😎", "🤨", "😐", "👀", "🤡"],
    "neutral":  ["👀", "😐", "🗿", "😴", "🙈"],
}

# Реакции на стикеры
_STICKER_REACTIONS = ["🗿", "👀", "😐", "🤨", "👻", "🤡"]

# Реакции на фото/видео
_MEDIA_REACTIONS = ["👀", "🗿", "😐", "🔥", "🤡", "👻"]


def _validate_reaction_pools():
    """
    Страховка: убираем из пулов любой эмодзи, которого нет в официальном
    списке Telegram — чтобы будущее редактирование пулов не сломало
    реакции молча (получали бы 400 на каждый вызов, без явной ошибки
    в логике бота).
    """
    try:
        from telegram.constants import ReactionEmoji
        allowed = {e.value for e in ReactionEmoji}
    except Exception:
        return  # если constants недоступны — не мешаем запуску
    for pool_name, pool in list(_REACTIONS_BY_MODE.items()):
        bad = [e for e in pool if e not in allowed]
        if bad:
            logger.warning(f"[REACTION] невалидные эмодзи в '{pool_name}': {bad} — убираю")
            _REACTIONS_BY_MODE[pool_name] = [e for e in pool if e in allowed] or ["👀"]
    for name in ("_STICKER_REACTIONS", "_MEDIA_REACTIONS"):
        pool = globals()[name]
        bad = [e for e in pool if e not in allowed]
        if bad:
            logger.warning(f"[REACTION] невалидные эмодзи в {name}: {bad} — убираю")
            globals()[name] = [e for e in pool if e in allowed] or ["👀"]

_validate_reaction_pools()

async def set_reaction(update, emoji: str):
    """Ставим реакцию на сообщение."""
    try:
        from telegram import ReactionTypeEmoji
        await update.message.set_reaction([ReactionTypeEmoji(emoji)])
    except Exception as e:
        logger.debug(f"[REACTION] не удалось: {e}")


async def maybe_react(update, mode: str, probability: float = 0.35):
    """
    С вероятностью probability ставим реакцию по режиму.
    Не заменяет голосовой — дополняет.
    """
    if random.random() > probability:
        return
    reactions = _REACTIONS_BY_MODE.get(mode, _REACTIONS_BY_MODE["neutral"])
    emoji = random.choice(reactions)
    await set_reaction(update, emoji)

# Все варианты имён бота
BOT_NAMES = {"есет", "eset", "есета", "есету", "есетом", "есете", "бот", "bot", "боту", "ботом"}

# ════════════════════════════════════════════════════════════════
# БАЗА ДАННЫХ
# ════════════════════════════════════════════════════════════════

def init_db():
    conn = _connect(); c = conn.cursor()

    c.execute("""CREATE TABLE IF NOT EXISTS users (
        user_id          INTEGER PRIMARY KEY,
        username         TEXT,
        first_name       TEXT,
        first_seen       TEXT,
        last_seen        TEXT,
        message_count    INTEGER DEFAULT 0,
        notes            TEXT    DEFAULT '',
        mood_score       REAL    DEFAULT 0.0,
        aggression_score REAL    DEFAULT 0.0,
        topic_history    TEXT    DEFAULT '{}',
        personality      TEXT    DEFAULT '',
        last_topic       TEXT    DEFAULT '',
        silent_streak    INTEGER DEFAULT 0,
        bot_attitude     REAL    DEFAULT 0.0,
        bot_tone         TEXT    DEFAULT 'нейтральный'
    )""")
    # Миграция: добавляем колонки если не существуют
    for col, default in [("bot_attitude", "0.0"), ("bot_tone", "'нейтральный'")]:
        try:
            c.execute(f"ALTER TABLE users ADD COLUMN {col} {'REAL' if col == 'bot_attitude' else 'TEXT'} DEFAULT {default}")
        except Exception:
            pass

    c.execute("""CREATE TABLE IF NOT EXISTS messages (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id      INTEGER,
        chat_id      INTEGER,
        role         TEXT,
        content      TEXT,
        timestamp    TEXT,
        sentiment    REAL DEFAULT 0.0,
        aggression   REAL DEFAULT 0.0,
        topic        TEXT DEFAULT '',
        subtopic     TEXT DEFAULT '',
        intent       TEXT DEFAULT '',
        emotionality REAL DEFAULT 0.0
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS group_members (
        chat_id    INTEGER,
        user_id    INTEGER,
        first_name TEXT,
        username   TEXT,
        PRIMARY KEY (chat_id, user_id)
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS conflicts (
        chat_id        INTEGER,
        user_id_a      INTEGER,
        user_id_b      INTEGER,
        conflict_count INTEGER DEFAULT 0,
        total_heat     REAL    DEFAULT 0.0,
        last_conflict  TEXT,
        PRIMARY KEY (chat_id, user_id_a, user_id_b)
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS chat_state (
        chat_id            INTEGER PRIMARY KEY,
        current_topic      TEXT    DEFAULT '',
        topic_since        TEXT,
        heat_level         REAL    DEFAULT 0.0,
        last_conflict_time TEXT,
        active_users       TEXT    DEFAULT '[]',
        messages_since_rex INTEGER DEFAULT 0
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS user_relationships (
        chat_id      INTEGER,
        user_id_a    INTEGER,
        user_id_b    INTEGER,
        rel_type     TEXT    DEFAULT 'нейтральный',
        heat         REAL    DEFAULT 0.0,
        interactions INTEGER DEFAULT 0,
        last_updated TEXT,
        PRIMARY KEY (chat_id, user_id_a, user_id_b)
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS active_disputes (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id      INTEGER,
        user_id_a    INTEGER,
        user_id_b    INTEGER,
        topic        TEXT,
        started_at   TEXT,
        last_message TEXT,
        intensity    REAL    DEFAULT 0.0,
        resolved     INTEGER DEFAULT 0
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS chat_topics (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id      INTEGER,
        timestamp    TEXT,
        topic        TEXT,
        subtopic     TEXT,
        heat_level   REAL DEFAULT 0.0,
        participants TEXT DEFAULT '[]'
    )""")

    # Настройки админа (сохраняются между перезапусками)
    c.execute("""CREATE TABLE IF NOT EXISTS settings (
        key   TEXT PRIMARY KEY,
        value TEXT
    )""")

    # Прозвища — бот придумывает кличку каждому участнику
    c.execute("""CREATE TABLE IF NOT EXISTS nicknames (
        user_id   INTEGER PRIMARY KEY,
        nickname  TEXT,
        created_at TEXT
    )""")

    # Личные обиды — конкретные оскорбления которые бот запомнил
    c.execute("""CREATE TABLE IF NOT EXISTS grudges (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id    INTEGER,
        chat_id    INTEGER,
        insult     TEXT,      -- текст оскорбления
        heat       REAL DEFAULT 1.0,
        times_used INTEGER DEFAULT 0,
        created_at TEXT
    )""")
    c.execute("CREATE INDEX IF NOT EXISTS idx_grudge_user ON grudges(user_id, chat_id)")

    # Долгосрочная память — ключевые события между рестартами
    c.execute("""CREATE TABLE IF NOT EXISTS long_memory (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id     INTEGER,
        user_id     INTEGER,
        event_type  TEXT,
        summary     TEXT,
        persons     TEXT,
        heat        REAL DEFAULT 0.0,
        created_at  TEXT
    )""")
    c.execute("CREATE INDEX IF NOT EXISTS idx_lm_chat ON long_memory(chat_id, event_type)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_lm_user ON long_memory(user_id)")

    # Самообучение (learner.py) — фразы бота, оценённые по реакции чата
    c.execute("""CREATE TABLE IF NOT EXISTS learned_phrases (
        phrase     TEXT,
        context    TEXT,
        score      REAL    DEFAULT 1.0,
        uses       INTEGER DEFAULT 0,
        wins       INTEGER DEFAULT 0,
        created_at TEXT,
        last_used  TEXT,
        PRIMARY KEY (phrase, context)
    )""")

    # Стиль каждого пользователя — что на него действует
    c.execute("""CREATE TABLE IF NOT EXISTS user_style (
        user_id      INTEGER PRIMARY KEY,
        humor_works  INTEGER DEFAULT 0,
        mat_works    INTEGER DEFAULT 0,
        absurd_works INTEGER DEFAULT 0,
        reactions    INTEGER DEFAULT 0,
        best_mode    TEXT    DEFAULT 'snark',
        updated_at   TEXT
    )""")

    # AI-сгенерированные провокации для автопровокации
    c.execute("""CREATE TABLE IF NOT EXISTS ai_provocations (
        text       TEXT PRIMARY KEY,
        style      TEXT    DEFAULT 'general',
        score      REAL    DEFAULT 0.5,
        uses       INTEGER DEFAULT 0,
        created_at TEXT
    )""")

    defaults = [
        ("active",        "1"),
        ("aggro_mode",    "0"),
        ("conflict_sens", "normal"),
    ]
    for k, v in defaults:
        c.execute("INSERT OR IGNORE INTO settings(key,value) VALUES(?,?)", (k, v))

    conn.commit(); conn.close()


# ── Settings ────────────────────────────────────────────────────

def get_setting(key: str) -> str:
    conn = _connect(); c = conn.cursor()
    c.execute("SELECT value FROM settings WHERE key=?", (key,))
    row = c.fetchone(); conn.close()
    return row[0] if row else ""

def set_setting(key: str, value: str):
    conn = _connect(); c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO settings(key,value) VALUES(?,?)", (key, value))
    conn.commit(); conn.close()


# ── Users ────────────────────────────────────────────────────────

def get_or_create_user(user_id, username, first_name):
    conn = _connect(); c = conn.cursor()
    now = datetime.now().isoformat()
    c.execute("SELECT user_id FROM users WHERE user_id=?", (user_id,))
    if not c.fetchone():
        c.execute("""INSERT INTO users
            (user_id,username,first_name,first_seen,last_seen,message_count)
            VALUES (?,?,?,?,?,0)""", (user_id, username or "", first_name or "", now, now))
        conn.commit(); conn.close(); return True
    c.execute("UPDATE users SET last_seen=?,username=?,first_name=? WHERE user_id=?",
              (now, username or "", first_name or "", user_id))
    conn.commit(); conn.close(); return False

def get_user_info(user_id):
    conn = _connect(); c = conn.cursor()
    c.execute("SELECT * FROM users WHERE user_id=?", (user_id,))
    row = c.fetchone(); conn.close(); return row

def increment_messages(user_id):
    conn = _connect(); c = conn.cursor()
    c.execute("UPDATE users SET message_count=message_count+1 WHERE user_id=?", (user_id,))
    conn.commit(); conn.close()

def update_user_profile(user_id, sentiment, aggression, topic, personality=""):
    conn = _connect(); c = conn.cursor()
    c.execute("SELECT topic_history,mood_score,aggression_score FROM users WHERE user_id=?", (user_id,))
    row = c.fetchone()
    if not row: conn.close(); return
    try:
        th = json.loads(row[0] or "{}")
    except Exception:
        th = {}
    if topic:
        th[topic] = th.get(topic, 0) + 1
    new_mood = row[1] * 0.8 + sentiment * 0.2
    new_aggr = row[2] * 0.8 + aggression * 0.2
    if personality:
        c.execute("""UPDATE users SET mood_score=?,aggression_score=?,topic_history=?,
                     last_topic=?,personality=? WHERE user_id=?""",
                  (new_mood, new_aggr, json.dumps(th, ensure_ascii=False), topic, personality, user_id))
    else:
        c.execute("""UPDATE users SET mood_score=?,aggression_score=?,topic_history=?,
                     last_topic=? WHERE user_id=?""",
                  (new_mood, new_aggr, json.dumps(th, ensure_ascii=False), topic, user_id))
    conn.commit(); conn.close()

def update_notes(user_id, notes):
    conn = _connect(); c = conn.cursor()
    c.execute("UPDATE users SET notes=? WHERE user_id=?", (notes, user_id))
    conn.commit(); conn.close()


def update_bot_attitude(user_id: int, sentiment: float, aggression: float,
                        directed_at_bot: bool, text: str = ""):
    """
    Обновляет отношение бота к пользователю.

    bot_attitude — число от -5.0 (враг) до +5.0 (уважаемый):
      >= 2.0  — дружелюбный
      0.8..2.0 — нейтральный
      -0.5..0.8 — настороженный
      -2.0..-0.5 — враждебный
      < -2.0 — ненависть

    bot_tone — текстовый ярлык для промпта и выбора слов
    """
    conn = _connect(); c = conn.cursor()
    row = c.execute("SELECT bot_attitude FROM users WHERE user_id=?",
                    (user_id,)).fetchone()
    if not row:
        conn.close(); return

    current = float(row[0] or 0.0)

    # Rule-based сигналы из словаря (без LLM)
    pos_sig, neg_sig = detect_attitude_signal(text) if text else (0.0, 0.0)

    # Дельта отношения
    delta = 0.0
    if directed_at_bot:
        if aggression > 0.6 or neg_sig > 0.3:
            delta = -0.70   # явная агрессия/оскорбление к боту
        elif aggression > 0.3 or neg_sig > 0.1:
            delta = -0.30
        elif pos_sig > 0.3 or sentiment > 0.5:
            delta = +0.50   # искренняя похвала/благодарность
        elif pos_sig > 0.1 or sentiment > 0.2:
            delta = +0.20   # позитивное общение
        else:
            delta = -0.05   # нейтрально к боту — чуть хуже
    else:
        # Не к боту — влияет слабее, но поведение в чате тоже считается
        if aggression > 0.7 or neg_sig > 0.4:
            delta = -0.20   # агрессивный человек вообще
        elif aggression > 0.4:
            delta = -0.08
        elif pos_sig > 0.3 or sentiment > 0.5:
            delta = +0.10   # приятный человек в целом
        elif pos_sig > 0.1:
            delta = +0.04

    # Медленный дрейф к нейтральному (забываем обиды и симпатии)
    current = current * 0.97 + delta

    # Зажимаем в [-5, +5]
    current = max(-5.0, min(5.0, current))

    # Определяем тон
    if current >= 2.0:
        tone = "дружелюбный"
    elif current >= 0.8:
        tone = "нейтральный"
    elif current >= -0.5:
        tone = "настороженный"
    elif current >= -2.0:
        tone = "враждебный"
    else:
        tone = "ненависть"

    c.execute("UPDATE users SET bot_attitude=?, bot_tone=? WHERE user_id=?",
              (current, tone, user_id))
    conn.commit(); conn.close()


def auto_update_notes(user_id: int, sentiment: float, aggression: float,
                      topic: str, directed_at_bot: bool, text: str = ""):
    """
    Автоматически обновляет заметку на основе поведения.
    Не перезаписывает ручные заметки — только добавляет теги.
    """
    conn = _connect(); c = conn.cursor()
    row = c.execute("SELECT notes, message_count FROM users WHERE user_id=?",
                    (user_id,)).fetchone()
    if not row:
        conn.close(); return
    current_notes, msg_count = row[0] or "", row[1] or 0

    tags = []

    # Агрессия к боту
    if directed_at_bot and aggression > 0.65:
        if "агрессор" not in current_notes:
            tags.append("агрессор")

    # Любитель конкретной темы
    if topic in ("мат", "конфликт", "оскорбление") and "конфликтный" not in current_notes:
        tags.append("конфликтный")

    # Позитивный к боту
    if directed_at_bot and sentiment > 0.5 and "дружелюбный" not in current_notes:
        tags.append("дружелюбный к боту")

    # Флудер
    t = text.lower()
    if len(text) < 5 and msg_count > 20 and "флудер" not in current_notes:
        tags.append("флудер")

    if tags:
        sep = ", " if current_notes else ""
        new_note = current_notes + sep + ", ".join(tags)
        c.execute("UPDATE users SET notes=? WHERE user_id=?", (new_note[:500], user_id))
        conn.commit()
    conn.close()


def get_bot_attitude(user_id: int) -> tuple[float, str]:
    """Возвращает (attitude_score, tone_label)."""
    conn = _connect(); c = conn.cursor()
    row = c.execute("SELECT bot_attitude, bot_tone FROM users WHERE user_id=?",
                    (user_id,)).fetchone()
    conn.close()
    if not row:
        return 0.0, "нейтральный"
    return float(row[0] or 0.0), (row[1] or "нейтральный")

# ════════════════════════════════════════════════════════════════
# ДОЛГОСРОЧНАЯ ПАМЯТЬ
# ════════════════════════════════════════════════════════════════

def save_long_memory(chat_id: int, user_id: int, event_type: str,
                     summary: str, persons: list, heat: float = 0.0):
    """Сохраняет важное событие в долгосрочную память."""
    conn = _connect(); c = conn.cursor()
    now  = datetime.now().isoformat(timespec="seconds")
    c.execute("""INSERT INTO long_memory
                 (chat_id,user_id,event_type,summary,persons,heat,created_at)
                 VALUES (?,?,?,?,?,?,?)""",
              (chat_id, user_id, event_type,
               summary[:300], json.dumps(persons, ensure_ascii=False),
               heat, now))
    # Лимит: не больше 200 событий на чат (удаляем самые старые)
    c.execute("""DELETE FROM long_memory WHERE chat_id=? AND id NOT IN (
                    SELECT id FROM long_memory WHERE chat_id=?
                    ORDER BY id DESC LIMIT 200)""", (chat_id, chat_id))
    conn.commit(); conn.close()


def get_long_memory(chat_id: int, limit: int = 12,
                    event_type: str = None) -> list[dict]:
    """Возвращает последние события из долгосрочной памяти."""
    conn = _connect(); c = conn.cursor()
    if event_type:
        rows = c.execute("""SELECT event_type,summary,persons,heat,created_at,user_id
                            FROM long_memory WHERE chat_id=? AND event_type=?
                            ORDER BY id DESC LIMIT ?""",
                         (chat_id, event_type, limit)).fetchall()
    else:
        rows = c.execute("""SELECT event_type,summary,persons,heat,created_at,user_id
                            FROM long_memory WHERE chat_id=?
                            ORDER BY id DESC LIMIT ?""",
                         (chat_id, limit)).fetchall()
    conn.close()
    return [{"type": r[0], "summary": r[1], "persons": json.loads(r[2] or "[]"),
             "heat": r[3], "date": r[4][:10] if r[4] else "", "user_id": r[5]}
            for r in rows]


def get_user_long_memory(user_id: int, limit: int = 8) -> list[dict]:
    """События связанные с конкретным юзером."""
    conn = _connect(); c = conn.cursor()
    rows = c.execute("""SELECT event_type,summary,heat,created_at
                        FROM long_memory WHERE user_id=? OR
                        persons LIKE ?
                        ORDER BY id DESC LIMIT ?""",
                     (user_id, f"%{user_id}%", limit)).fetchall()
    conn.close()
    return [{"type": r[0], "summary": r[1], "heat": r[2],
             "date": r[3][:10] if r[3] else ""} for r in rows]


def build_long_memory_ctx(chat_id: int, sender_id: int) -> str:
    """Строит блок долгосрочной памяти для системного промпта."""
    lines = []

    # Топ события чата (конфликты, угрозы)
    hot = get_long_memory(chat_id, limit=6,
                          event_type=None)
    hot_events = [e for e in hot if e["heat"] > 0.5]
    if hot_events:
        lines.append("[ПАМЯТЬ ЧАТА — прошлые события:]")
        for e in hot_events[:5]:
            lines.append(f"  {e['date']} [{e['type']}] {e['summary']}")

    # События связанные с текущим отправителем
    user_ev = get_user_long_memory(sender_id, limit=4)
    if user_ev:
        lines.append("[ИСТОРИЯ ЭТОГО ЧЕЛОВЕКА:]")
        for e in user_ev[:3]:
            lines.append(f"  {e['date']} {e['summary']}")

    return "\n".join(lines) if lines else ""


def clear_user(user_id):
    conn = _connect(); c = conn.cursor()
    c.execute("DELETE FROM messages WHERE user_id=?", (user_id,))
    c.execute("""UPDATE users SET notes='',message_count=0,mood_score=0,
                 aggression_score=0,topic_history='{}',personality='',last_topic=''
                 WHERE user_id=?""", (user_id,))
    conn.commit(); conn.close()


# ── Messages ─────────────────────────────────────────────────────

def save_message(user_id, chat_id, role, content,
                 sentiment=0.0, aggression=0.0, topic="",
                 subtopic="", intent="", emotionality=0.0):
    conn = _connect(); c = conn.cursor()
    c.execute("""INSERT INTO messages
        (user_id,chat_id,role,content,timestamp,sentiment,aggression,
         topic,subtopic,intent,emotionality)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (user_id, chat_id, role, content, datetime.now().isoformat(),
         sentiment, aggression, topic, subtopic, intent, emotionality))
    conn.commit(); conn.close()

def get_history(chat_id, limit=20):
    conn = _connect(); c = conn.cursor()
    c.execute("SELECT role,content FROM messages WHERE chat_id=? ORDER BY id DESC LIMIT ?",
              (chat_id, limit))
    rows = c.fetchall(); conn.close(); return list(reversed(rows))

def get_recent_chat_messages(chat_id, limit=8):
    conn = _connect(); c = conn.cursor()
    c.execute("""SELECT m.user_id,m.content,m.sentiment,m.topic,u.first_name,
                        m.aggression,m.intent
                 FROM messages m LEFT JOIN users u ON m.user_id=u.user_id
                 WHERE m.chat_id=? AND m.role='user'
                 ORDER BY m.id DESC LIMIT ?""", (chat_id, limit))
    rows = c.fetchall(); conn.close(); return list(reversed(rows))


# ── Members ──────────────────────────────────────────────────────

def register_member(chat_id, user_id, first_name, username):
    conn = _connect(); c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO group_members VALUES (?,?,?,?)",
              (chat_id, user_id, first_name or "", username or ""))
    conn.commit(); conn.close()

def get_members(chat_id):
    conn = _connect(); c = conn.cursor()
    c.execute("SELECT user_id,first_name,username FROM group_members WHERE chat_id=?", (chat_id,))
    rows = c.fetchall(); conn.close(); return rows


# ── Conflicts ────────────────────────────────────────────────────

def register_conflict(chat_id, user_a, user_b, heat=1.0):
    if user_a == user_b: return
    conn = _connect(); c = conn.cursor()
    a, b = min(user_a, user_b), max(user_a, user_b)
    now = datetime.now().isoformat()
    c.execute("""INSERT INTO conflicts (chat_id,user_id_a,user_id_b,conflict_count,total_heat,last_conflict)
                 VALUES (?,?,?,1,?,?)
                 ON CONFLICT(chat_id,user_id_a,user_id_b)
                 DO UPDATE SET conflict_count=conflict_count+1,
                               total_heat=total_heat+?,last_conflict=?""",
              (chat_id, a, b, heat, now, heat, now))
    conn.commit(); conn.close()

def get_top_conflicts(chat_id, limit=3):
    conn = _connect(); c = conn.cursor()
    c.execute("""SELECT r.user_id_a,r.user_id_b,r.conflict_count,r.total_heat,
                        ua.first_name,ub.first_name
                 FROM conflicts r
                 LEFT JOIN users ua ON r.user_id_a=ua.user_id
                 LEFT JOIN users ub ON r.user_id_b=ub.user_id
                 WHERE r.chat_id=? ORDER BY r.total_heat DESC LIMIT ?""", (chat_id, limit))
    rows = c.fetchall(); conn.close(); return rows


# ── Relationships ────────────────────────────────────────────────

def update_relationship(chat_id, user_a, user_b, heat_delta, rel_type=None):
    if user_a == user_b: return
    a, b = min(user_a, user_b), max(user_a, user_b)
    conn = _connect(); c = conn.cursor()
    now = datetime.now().isoformat()
    c.execute("""INSERT INTO user_relationships
                 (chat_id,user_id_a,user_id_b,rel_type,heat,interactions,last_updated)
                 VALUES (?,?,?,'нейтральный',?,1,?)
                 ON CONFLICT(chat_id,user_id_a,user_id_b)
                 DO UPDATE SET heat=MIN(heat+?,10.0),
                               interactions=interactions+1,
                               last_updated=?,
                               rel_type=CASE WHEN ? IS NOT NULL THEN ? ELSE rel_type END""",
              (chat_id, a, b, heat_delta, now, heat_delta, now, rel_type, rel_type))
    conn.commit(); conn.close()

def get_all_relationships(chat_id):
    conn = _connect(); c = conn.cursor()
    c.execute("""SELECT r.user_id_a,r.user_id_b,r.rel_type,r.heat,
                        ua.first_name,ub.first_name
                 FROM user_relationships r
                 LEFT JOIN users ua ON r.user_id_a=ua.user_id
                 LEFT JOIN users ub ON r.user_id_b=ub.user_id
                 WHERE r.chat_id=? AND r.heat > 1.0
                 ORDER BY r.heat DESC LIMIT 6""", (chat_id,))
    rows = c.fetchall(); conn.close(); return rows

RELATIONSHIP_DECAY_PER_DAY = 0.4  # насколько остывает "накал" за сутки без стычек
CONFLICT_DECAY_PER_DAY     = 0.3

def decay_relationships_and_conflicts(chat_id: int = None):
    """
    Со временем без новых стычек отношения и конфликты должны
    остывать, а не висеть "врагами" вечно после одной ссоры месяц
    назад. Понижает heat/total_heat пропорционально дням простоя и
    откатывает rel_type обратно к нейтральному, когда накал угас.
    """
    import datetime as _dt
    conn = _connect(); c = conn.cursor()
    now = _dt.datetime.now()

    where_chat = "WHERE chat_id=?" if chat_id is not None else ""
    params = (chat_id,) if chat_id is not None else ()

    rows = c.execute(f"""SELECT chat_id,user_id_a,user_id_b,heat,last_updated
                         FROM user_relationships {where_chat}""", params).fetchall()
    for cid, ua, ub, heat, last_upd in rows:
        try:
            days = max(0.0, (now - _dt.datetime.fromisoformat(last_upd)).total_seconds() / 86400)
        except Exception:
            continue
        if days < 0.5 or heat <= 0:
            continue
        new_heat = max(0.0, heat - RELATIONSHIP_DECAY_PER_DAY * days)
        new_type = "нейтральный" if new_heat < 1.5 else None  # None = не трогать тип
        if new_type:
            c.execute("""UPDATE user_relationships SET heat=?,rel_type=?
                         WHERE chat_id=? AND user_id_a=? AND user_id_b=?""",
                      (new_heat, new_type, cid, ua, ub))
        else:
            c.execute("""UPDATE user_relationships SET heat=?
                         WHERE chat_id=? AND user_id_a=? AND user_id_b=?""",
                      (new_heat, cid, ua, ub))

    where_chat2 = "WHERE chat_id=?" if chat_id is not None else ""
    crows = c.execute(f"""SELECT chat_id,user_id_a,user_id_b,total_heat,last_conflict
                          FROM conflicts {where_chat2}""", params).fetchall()
    for cid, ua, ub, heat, last_c in crows:
        try:
            days = max(0.0, (now - _dt.datetime.fromisoformat(last_c)).total_seconds() / 86400)
        except Exception:
            continue
        if days < 0.5 or heat <= 0:
            continue
        new_heat = max(0.0, heat - CONFLICT_DECAY_PER_DAY * days)
        c.execute("""UPDATE conflicts SET total_heat=?
                     WHERE chat_id=? AND user_id_a=? AND user_id_b=?""",
                  (new_heat, cid, ua, ub))

    conn.commit(); conn.close()


# ── Disputes ─────────────────────────────────────────────────────

def upsert_dispute(chat_id, user_a, user_b, topic, intensity):
    if user_a == user_b: return
    a, b = min(user_a, user_b), max(user_a, user_b)
    conn = _connect(); c = conn.cursor()
    now = datetime.now().isoformat()
    c.execute("""SELECT id FROM active_disputes
                 WHERE chat_id=? AND user_id_a=? AND user_id_b=? AND resolved=0 AND topic=?""",
              (chat_id, a, b, topic))
    row = c.fetchone()
    if row:
        c.execute("UPDATE active_disputes SET last_message=?,intensity=intensity*0.7+?*0.3 WHERE id=?",
                  (now, intensity, row[0]))
    else:
        c.execute("""INSERT INTO active_disputes
                     (chat_id,user_id_a,user_id_b,topic,started_at,last_message,intensity)
                     VALUES (?,?,?,?,?,?,?)""", (chat_id, a, b, topic, now, now, intensity))
    conn.commit(); conn.close()

def get_active_disputes(chat_id, limit=3):
    conn = _connect(); c = conn.cursor()
    c.execute("""SELECT d.user_id_a,d.user_id_b,d.topic,d.intensity,
                        ua.first_name,ub.first_name,d.started_at
                 FROM active_disputes d
                 LEFT JOIN users ua ON d.user_id_a=ua.user_id
                 LEFT JOIN users ub ON d.user_id_b=ub.user_id
                 WHERE d.chat_id=? AND d.resolved=0
                 ORDER BY d.intensity DESC LIMIT ?""", (chat_id, limit))
    rows = c.fetchall(); conn.close(); return rows

DISPUTE_STALE_HOURS = 20  # если спор не подогревали столько часов — считаем угасшим

def resolve_stale_disputes(chat_id: int = None):
    """
    Помечает старые споры как resolved, если по ним давно не было
    сообщений. Без этого active_disputes только растёт — старые
    склоки из прошлого месяца иначе вечно всплывают в промпте как
    актуальные, хотя все давно помирились.
    """
    conn = _connect(); c = conn.cursor()
    import datetime as _dt
    cutoff_iso = (_dt.datetime.now() - _dt.timedelta(hours=DISPUTE_STALE_HOURS)).isoformat()
    if chat_id is not None:
        c.execute("""UPDATE active_disputes SET resolved=1
                     WHERE chat_id=? AND resolved=0 AND last_message < ?""",
                  (chat_id, cutoff_iso))
    else:
        c.execute("""UPDATE active_disputes SET resolved=1
                     WHERE resolved=0 AND last_message < ?""", (cutoff_iso,))
    conn.commit(); conn.close()


# ── Chat state ───────────────────────────────────────────────────

def get_chat_state(chat_id):
    conn = _connect(); c = conn.cursor()
    c.execute("SELECT * FROM chat_state WHERE chat_id=?", (chat_id,))
    row = c.fetchone(); conn.close(); return row

def update_chat_state(chat_id, topic, heat, replied=False):
    conn = _connect(); c = conn.cursor()
    now = datetime.now().isoformat()
    c.execute("SELECT chat_id,messages_since_rex,current_topic FROM chat_state WHERE chat_id=?", (chat_id,))
    existing = c.fetchone()
    if existing:
        new_since = 0 if replied else existing[1] + 1
        topic_changed = existing[2] != topic
        c.execute("""UPDATE chat_state
                     SET current_topic=?,heat_level=heat_level*0.85+?*0.15,
                         messages_since_rex=?,
                         topic_since=CASE WHEN ? THEN ? ELSE topic_since END
                     WHERE chat_id=?""",
                  (topic, heat, new_since, topic_changed, now, chat_id))
    else:
        c.execute("""INSERT INTO chat_state
                     (chat_id,current_topic,topic_since,heat_level,messages_since_rex)
                     VALUES (?,?,?,?,0)""", (chat_id, topic, now, heat))
    conn.commit(); conn.close()

def save_topic_event(chat_id, topic, subtopic, heat, participants):
    conn = _connect(); c = conn.cursor()
    c.execute("""INSERT INTO chat_topics (chat_id,timestamp,topic,subtopic,heat_level,participants)
                 VALUES (?,?,?,?,?,?)""",
              (chat_id, datetime.now().isoformat(), topic, subtopic, heat,
               json.dumps(participants, ensure_ascii=False)))
    conn.commit(); conn.close()

def get_all_users_summary(chat_id):
    conn = _connect(); c = conn.cursor()
    c.execute("""SELECT u.user_id,u.first_name,u.message_count,u.mood_score,
                        u.aggression_score,u.personality,u.last_topic,u.notes
                 FROM users u
                 INNER JOIN group_members gm ON u.user_id=gm.user_id
                 WHERE gm.chat_id=? ORDER BY u.message_count DESC""", (chat_id,))
    rows = c.fetchall(); conn.close(); return rows

def reset_all_aggro():
    conn = _connect(); c = conn.cursor()
    c.execute("UPDATE chat_state SET heat_level=0")
    conn.commit(); conn.close()

# ════════════════════════════════════════════════════════════════
# ПРОЗВИЩА
# ════════════════════════════════════════════════════════════════

def get_nickname(user_id: int) -> str | None:
    conn = _connect(); c = conn.cursor()
    row = c.execute("SELECT nickname FROM nicknames WHERE user_id=?", (user_id,)).fetchone()
    conn.close()
    return row[0] if row else None


def save_nickname(user_id: int, nickname: str):
    conn = _connect(); c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO nicknames (user_id, nickname, created_at) VALUES (?,?,?)",
              (user_id, nickname, __import__('datetime').datetime.now().isoformat()))
    conn.commit(); conn.close()


async def maybe_generate_nickname(user_id: int, user_name: str,
                                   personality: str = "", notes: str = "") -> str | None:
    """
    Генерирует кличку через LLM если её ещё нет.
    Возвращает кличку или None если не сгенерировал.
    """
    existing = get_nickname(user_id)
    if existing:
        return existing

    hint = ""
    if personality: hint += f" характер: {personality}."
    if notes: hint += f" заметка: {notes[:80]}."

    prompt = (
        f"Придумай ОДНО короткое грубое прозвище для человека по имени {user_name}.{hint} "
        f"Прозвище должно быть: 1-2 слова, на русском, отражать суть человека, "
        f"можно с матом или сленгом. Только само прозвище, без объяснений и кавычек."
    )
    try:
        resp = _llm_call_with_retry(lambda: client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20, temperature=1.2
        ))
        nick = resp.choices[0].message.content.strip().strip('"\'').split('\n')[0][:30]
        if nick:
            save_nickname(user_id, nick)
            return nick
    except Exception as e:
        logger.debug(f"[NICKNAME] не сгенерировал: {e}")
    return None


# ════════════════════════════════════════════════════════════════
# ЛИЧНЫЕ ОБИДЫ
# ════════════════════════════════════════════════════════════════

def save_grudge(user_id: int, chat_id: int, insult: str, heat: float = 1.0):
    """Запоминает конкретное оскорбление от пользователя."""
    conn = _connect(); c = conn.cursor()
    now = __import__('datetime').datetime.now().isoformat()
    c.execute("INSERT INTO grudges (user_id,chat_id,insult,heat,created_at) VALUES (?,?,?,?,?)",
              (user_id, chat_id, insult[:200], heat, now))
    # Лимит — не больше 10 обид на одного человека
    c.execute("""DELETE FROM grudges WHERE user_id=? AND id NOT IN (
                    SELECT id FROM grudges WHERE user_id=? ORDER BY heat DESC, id DESC LIMIT 10
                 )""", (user_id, user_id))
    conn.commit(); conn.close()


def get_grudges(user_id: int, chat_id: int, limit: int = 3) -> list[str]:
    """Возвращает топ обид на этого человека (по накалу)."""
    conn = _connect(); c = conn.cursor()
    rows = c.execute("""SELECT insult FROM grudges
                         WHERE user_id=? AND chat_id=?
                         ORDER BY heat DESC LIMIT ?""",
                     (user_id, chat_id, limit)).fetchall()
    conn.close()
    return [r[0] for r in rows]


def mark_grudge_used(user_id: int, insult: str):
    """Отмечаем что обиду использовали (чтобы не повторяться часто)."""
    conn = _connect(); c = conn.cursor()
    c.execute("""UPDATE grudges SET times_used=times_used+1, heat=heat*0.7
                 WHERE user_id=? AND insult=?""", (user_id, insult))
    conn.commit(); conn.close()


def maybe_recall_grudge(user_id: int, chat_id: int) -> str | None:
    """
    С вероятностью 25% вспоминает прошлую обиду для включения в промпт.
    Возвращает текст обиды или None.
    """
    if random.random() > 0.25:
        return None
    grudges = get_grudges(user_id, chat_id, limit=5)
    if not grudges:
        return None
    # Выбираем из тех что использовали реже
    conn = _connect(); c = conn.cursor()
    rows = c.execute("""SELECT insult, times_used FROM grudges
                         WHERE user_id=? AND chat_id=?
                         ORDER BY times_used ASC, heat DESC LIMIT 3""",
                     (user_id, chat_id)).fetchall()
    conn.close()
    if not rows:
        return None
    chosen = rows[0][0]
    mark_grudge_used(user_id, chosen)
    return chosen


# ════════════════════════════════════════════════════════════════
# ЭСКАЛАЦИЯ — чем дольше конфликт тем злее
# ════════════════════════════════════════════════════════════════

def get_conflict_escalation(chat_id: int, user_id: int) -> float:
    """
    Возвращает коэффициент эскалации 1.0..3.0 на основе:
    - количества конфликтов с этим человеком
    - текущего heat_level чата
    - накопленных обид
    """
    conn = _connect(); c = conn.cursor()

    # Конфликты этого юзера в чате
    row = c.execute("""SELECT MAX(conflict_count), MAX(total_heat)
                       FROM conflicts
                       WHERE chat_id=? AND (user_id_a=? OR user_id_b=?)""",
                    (chat_id, user_id, user_id)).fetchone()
    conflict_count = (row[0] or 0) if row else 0
    total_heat = (row[1] or 0.0) if row else 0.0

    # Обиды
    grudge_count = c.execute("SELECT COUNT(*) FROM grudges WHERE user_id=? AND chat_id=?",
                              (user_id, chat_id)).fetchone()[0]

    # Heat чата
    heat_row = c.execute("SELECT heat_level FROM chat_state WHERE chat_id=?", (chat_id,)).fetchone()
    chat_heat = (heat_row[0] or 0.0) if heat_row else 0.0

    conn.close()

    # Формула эскалации
    escalation = 1.0
    escalation += min(1.0, conflict_count * 0.15)   # за каждый конфликт +15%
    escalation += min(0.5, total_heat * 0.1)         # за суммарный накал
    escalation += min(0.3, grudge_count * 0.1)       # за обиды
    escalation += min(0.2, chat_heat * 0.4)          # за текущий накал чата

    return min(3.0, escalation)


def escalation_to_mode(esc: float) -> str:
    """Переводит коэффициент эскалации в режим ответа."""
    if esc >= 2.5: return "attack"
    if esc >= 1.8: return "conflict"
    if esc >= 1.3: return "snark"
    return "neutral"


def get_stats():
    conn = _connect(); c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users");                          u = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM messages WHERE role='user'");     m = c.fetchone()[0]
    c.execute("SELECT AVG(sentiment)  FROM messages WHERE role='user'"); mood = c.fetchone()[0] or 0
    c.execute("SELECT AVG(aggression) FROM messages WHERE role='user'"); aggr = c.fetchone()[0] or 0
    c.execute("SELECT SUM(conflict_count) FROM conflicts");           cf = c.fetchone()[0] or 0
    c.execute("SELECT COUNT(*) FROM active_disputes WHERE resolved=0"); ds = c.fetchone()[0]
    conn.close()
    return u, m, mood, aggr, cf, ds


def get_stats_extended():
    """Расширенная статистика для /admin."""
    conn = _connect(); c = conn.cursor()
    top_active = c.execute("""SELECT user_id, first_name, message_count
                               FROM users ORDER BY message_count DESC LIMIT 3""").fetchall()
    top_aggro  = c.execute("""SELECT user_id, first_name, aggression_score
                               FROM users ORDER BY aggression_score DESC LIMIT 3""").fetchall()
    top_cf     = c.execute("""SELECT r.user_id_a, r.user_id_b, r.conflict_count,
                                     ua.first_name, ub.first_name
                               FROM conflicts r
                               LEFT JOIN users ua ON r.user_id_a=ua.user_id
                               LEFT JOIN users ub ON r.user_id_b=ub.user_id
                               ORDER BY r.conflict_count DESC LIMIT 3""").fetchall()
    conn.close()
    lines = ["📊 *Расширенная статистика*\n"]
    if top_active:
        lines.append("🏆 *Топ болтунов:*")
        for uid, fn, cnt in top_active:
            lines.append(f"  {get_display_name(uid, fn or '?')}: {cnt} сообщ.")
    if top_aggro:
        lines.append("\n😤 *Топ агрессивных:*")
        for uid, fn, aggr in top_aggro:
            if (aggr or 0) > 0.05:
                lines.append(f"  {get_display_name(uid, fn or '?')}: {(aggr or 0):.2f}")
    if top_cf:
        lines.append("\n⚔️ *Топ конфликтов:*")
        for a, b, cnt, fa, fb in top_cf:
            na = get_display_name(a, fa or "?"); nb = get_display_name(b, fb or "?")
            lines.append(f"  {na} vs {nb}: {cnt}×")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
# ДЕТЕКТОР ОБРАЩЕНИЯ К БОТУ
# Многоуровневый: entity → имя → reply → паттерны → короткое оскорбление
# ════════════════════════════════════════════════════════════════

BOT_ADDRESS_PATTERNS = [
    "что думаешь", "как думаешь", "что скажешь", "что считаешь",
    "ты как", "ты чего", "ты где", "ты кто", "ты тут", "ты здесь",
    "ты спишь", "ты живой", "ты слышишь", "ты видишь",
    "ты урод", "ты дебил", "ты идиот", "ты тупой", "ты мудак",
    "ты придурок", "ты баран", "ты кретин", "ты лох", "ты чмо",
    "ты козёл", "ты дурак", "ты тварь", "ты скот", "ты ублюдок",
    "иди нахуй", "пошёл нахуй", "иди нафиг",
    "заткнись", "отвали", "отстань",
    "скажи", "ответь", "объясни", "расскажи",
    "помолчи", "замолчи",
]

def is_directed_at_bot(text: str, bot_username: str, bot_name: str, update) -> bool:
    t = text.lower().strip()

    # Уровень 1: @mention через entity
    if update.message.entities:
        for ent in update.message.entities:
            if ent.type == "mention":
                mention = update.message.text[ent.offset:ent.offset + ent.length].lower()
                if bot_username.lower() in mention:
                    return True

    # Уровень 2: имя бота в тексте как отдельное слово
    words = set(re.sub(r'[,!?.]', ' ', t).split())
    if words & BOT_NAMES:
        return True
    if bot_username.lower() in t or bot_name.lower() in t:
        return True

    # Уровень 3: reply на сообщение бота
    if (update.message.reply_to_message and
            update.message.reply_to_message.from_user and
            (update.message.reply_to_message.from_user.username or "").lower() == bot_username.lower()):
        return True

    # Уровень 4: паттерны обращения
    for pattern in BOT_ADDRESS_PATTERNS:
        if pattern in t:
            return True

    # Уровень 5: короткое сообщение (≤5 слов) с матом — почти всегда к боту
    ALL_MAT = set(w.lower() for w in MAT + SLANG)
    msg_words = t.split()
    if len(msg_words) <= 5 and set(msg_words) & ALL_MAT:
        return True

    return False


# ════════════════════════════════════════════════════════════════
# LLM АНАЛИЗ СООБЩЕНИЯ
# ════════════════════════════════════════════════════════════════

# Кэш для быстрого анализа — если brain уже всё определил rule-based, LLM не нужен
_FAST_FIELDS = {"sentiment","aggression","emotionality","topic","subtopic",
                "intent","directed_at_bot","directed_at_user","is_conflict",
                "conflict_persons","topic_continuity","rex_interest","flood_score",
                "real_meaning","subtext","target_person","social_role","power_dynamic",
                "sarcasm_detected","is_bait","is_repetition","unique_content",
                "best_response_angle","what_bot_should_avoid"}


def _fallback_analysis(topic="другое"):
    """Быстрый дефолт без LLM."""
    return {
        "sentiment": 0.0, "aggression": 0.2, "emotionality": 0.3,
        "topic": topic, "subtopic": "", "intent": "болтовня",
        "directed_at_bot": False, "directed_at_user": None,
        "is_conflict": False, "conflict_persons": [],
        "topic_continuity": False, "rex_interest": 0.3, "flood_score": 0.5,
        "real_meaning": "", "subtext": "", "target_person": "",
        "social_role": "", "power_dynamic": "", "sarcasm_detected": False,
        "is_bait": False, "is_repetition": False, "unique_content": True,
        "best_response_angle": "ирония", "what_bot_should_avoid": "",
    }


async def analyze_fast(text, chat_id, user_name, chat_context, known_names,
                       bot_names, chat_state, brain_signal=None):
    """
    Умный анализ: пропускает LLM для очевидных случаев.
    Экономит 400-700ms на флуде и простых сообщениях.
    
    brain_signal — если brain уже определил флуд/тип, берём оттуда.
    """
    t = text.lower().strip()

    # ── Быстрые пути без LLM ─────────────────────────────────────

    # Флуд — LLM не нужен
    if brain_signal and brain_signal.is_flood and brain_signal.aggression < 0.15:
        return _fallback_analysis("флуд") | {"flood_score": 0.9}

    # Приветствие с низкой агрессией — понятно и без LLM
    if brain_signal and brain_signal.topic == "приветствие":
        return _fallback_analysis("приветствие") | {
            "sentiment": 0.6, "aggression": 0.0,
            "directed_at_bot": brain_signal.has_bot_name or brain_signal.has_mention,
            "flood_score": 0.0,
        }

    # ── Полный LLM анализ ────────────────────────────────────────
    result = deep_analyze(
        text=text, chat_id=chat_id, user_name=user_name,
        chat_context=chat_context, known_names=known_names,
        bot_names=bot_names, chat_state=chat_state,
    )
    return result

def analyze_message(text, chat_context=None, bot_names=None,
                    chat_state=None, known_names=None):
    bot_names_str = ", ".join(bot_names) if bot_names else "есет, eset, бот"

    context_lines = []
    if chat_context:
        for m in chat_context[-8:]:
            uid_m, msg_text, sentiment, topic_m, fname = m[0], m[1], m[2], m[3], m[4]
            name = get_display_name(uid_m, fname or "?") if uid_m else (fname or "?")
            short = (msg_text or "")[:100].split("[ЗАМЕТКА:")[0].strip()
            aggr_m = m[5] if len(m) > 5 else 0
            line = f"  {name}: {short}"
            if aggr_m and float(aggr_m) > 0.5:
                line += f" [агрессия:{aggr_m:.1f}]"
            context_lines.append(line)
    context_str = ("\nПОСЛЕДНИЕ СООБЩЕНИЯ:\n" + "\n".join(context_lines)) if context_lines else ""

    state_str = ""
    if chat_state and len(chat_state) > 3:
        cur_topic = chat_state[1] or "нет темы"
        heat = chat_state[3] or 0
        msgs_silence = chat_state[6] if len(chat_state) > 6 else 0
        state_str = f"\nТЕМА ЧАТА: {cur_topic} | НАКАЛ: {heat:.2f}/1.0 | Молчание Есет: {msgs_silence} сообщ."

    names_str = f"\nУЧАСТНИКИ: {', '.join(known_names)}" if known_names else ""

    prompt = f"""Ты — аналитик русскоязычного Telegram чата. Верни ТОЛЬКО валидный JSON без пояснений.

БОТ: {bot_names_str} (AI персонаж Есет, грубый матерщинник){names_str}{state_str}{context_str}

НОВОЕ СООБЩЕНИЕ: "{text[:500]}"

ПРАВИЛА:
- "приветствие": привет/здаров/хай/дарова/салют и подобное — sentiment > 0.2, aggression=0, directed_at_bot=true если адресовано боту
- "флуд": случайные сообщения, стикеры, "ха", "ок", "+", смайлы — aggression < 0.2
- "конфликт": прямые оскорбления между конкретными людьми — is_conflict=true, aggression > 0.6
- "спор": разные точки зрения БЕЗ оскорблений — is_conflict=false
- "мат": просто матерятся без агрессии к конкретному человеку
- directed_at_bot=true ТОЛЬКО если явное обращение к боту, "ты" без адресата-человека

JSON (строго без markdown):
{{
  "sentiment": <-1..1>,
  "aggression": <0..1>,
  "emotionality": <0..1>,
  "topic": "<флуд|приветствие|спор|конфликт|оскорбление|мат|жалоба|юмор|похвала|вопрос|просьба|новость|угроза|провокация|другое>",
  "subtopic": "<одно слово>",
  "intent": "<болтовня|ссора|провокация|внимание|помощь|информация|жалоба|юмор|другое>",
  "directed_at_bot": <true/false>,
  "directed_at_user": "<имя или null>",
  "is_conflict": <true/false>,
  "conflict_persons": ["<имя_а>", "<имя_б>"],
  "topic_continuity": <true/false>,
  "rex_interest": <0..1>,
  "flood_score": <0..1>
}}"""

    try:
        resp = _llm_call_with_retry(lambda: client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300, temperature=0.05
        ))
        raw = resp.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw)
        result.setdefault("conflict_persons", [])
        result.setdefault("topic_continuity", False)
        result.setdefault("flood_score", 0.0)
        return result
    except Exception as e:
        logger.warning(f"[analyze] error: {e}")
        return {
            "sentiment": 0.0, "aggression": 0.2, "emotionality": 0.3,
            "topic": "другое", "subtopic": "", "intent": "болтовня",
            "directed_at_bot": False, "directed_at_user": None,
            "is_conflict": False, "conflict_persons": [],
            "topic_continuity": False, "rex_interest": 0.3, "flood_score": 0.5
        }


# ════════════════════════════════════════════════════════════════
# МОЗГ: РЕШЕНИЕ — ОТВЕЧАТЬ ИЛИ НЕТ
# Бот МОЛЧИТ по умолчанию, кроме явных триггеров
# ════════════════════════════════════════════════════════════════

def decide_reply(update, bot_username, bot_name, analysis, chat_state, user_info, is_group):
    # В ЛС — всегда отвечаем (но только если дошли до этой функции, т.е. это не чужой)
    if not is_group:
        return True, "лс"

    # Бот выключен
    if get_setting("active") == "0":
        return False, "выключен"

    # Форс-агрессия — отвечать всегда
    if get_setting("aggro_mode") == "1":
        return True, "форс_агрессия"

    text = (update.message.text or "").lower()

    # Тихий режим — только прямые обращения
    if BOT_MODE["silent"]:
        if is_directed_at_bot(text, bot_username, bot_name, update):
            return True, "тихий_адрес"
        if analysis.get("directed_at_bot", False):
            return True, "тихий_llm"
        return False, "тихий_режим"

    # Уровень 1: прямое обращение к боту — многоуровневый детектор
    if is_directed_at_bot(text, bot_username, bot_name, update):
        return True, "адресовано_боту"
    if analysis.get("directed_at_bot", False):
        return True, "к_боту_llm"

    topic        = analysis.get("topic", "другое")
    intent       = analysis.get("intent", "болтовня")
    aggression   = analysis.get("aggression", 0.0)
    emotionality = analysis.get("emotionality", 0.5)
    is_conflict  = analysis.get("is_conflict", False)
    flood_score  = analysis.get("flood_score", 0.0)
    topic_cont   = analysis.get("topic_continuity", False)
    interest     = analysis.get("rex_interest", 0.3)

    msgs_since = chat_state[6] if chat_state and len(chat_state) > 6 else 0
    heat       = chat_state[3] if chat_state and len(chat_state) > 3 else 0.0

    # Уровень 2: чистый флуд — почти всегда молчим
    if flood_score > 0.7 and topic == "флуд" and aggression < 0.2:
        silence_bonus = min(msgs_since / 15.0, 0.3)
        return random.random() < 0.05 + silence_bonus, "флуд_молчим"

    # Уровень 3: горячие ситуации — всегда влезаем
    if is_conflict and aggression > 0.4:
        return True, "конфликт"
    if topic in ("оскорбление", "угроза") and aggression > 0.5:
        return True, "оскорбление"
    if topic == "мат" and aggression > 0.6:
        return True, "мат_горячий"
    if intent == "провокация":
        return True, "провокация"

    # Бонусы к вероятности
    silence_bonus = min(msgs_since / 8.0, 0.45)
    heat_bonus    = heat * 0.25
    cont_bonus    = 0.15 if topic_cont else 0.0

    # Логика по типу
    if topic == "флуд":
        if emotionality > 0.4:
            return random.random() < 0.25 + silence_bonus, "флуд_эмоц"
        return random.random() < 0.07 + silence_bonus, "флуд"
    if topic == "спор":
        return random.random() < 0.55 + silence_bonus + heat_bonus, "спор"
    if topic in ("жалоба",) or intent == "жалоба":
        return random.random() < 0.65 + silence_bonus, "жалоба"
    if topic == "юмор":
        return random.random() < 0.45 + cont_bonus, "юмор"
    if topic == "мат":
        return random.random() < 0.40 + silence_bonus, "мат"
    if topic in ("вопрос", "просьба"):
        return random.random() < 0.55 + silence_bonus, "вопрос"
    if topic == "новость":
        return random.random() < 0.30 + cont_bonus + silence_bonus, "новость"
    if topic == "похвала":
        return random.random() < 0.60, "похвала"

    base = 0.25 + interest * 0.4 + silence_bonus + heat_bonus + cont_bonus
    return random.random() < min(base, 0.85), "рандом"


# ════════════════════════════════════════════════════════════════
# СИСТЕМНЫЙ ПРОМПТ ЕСЕТ
# ════════════════════════════════════════════════════════════════

def build_system(sender_name, sender_id, is_group, members=None,
                 is_new=False, notes="", analysis=None,
                 chat_context=None, chat_state=None,
                 user_info=None, top_conflicts=None,
                 chat_id=None, all_relationships=None,
                 active_disputes_list=None, users_summary=None,
                 respond_mode="neutral", context_summary="",
                 dialogue_summary="", thread_context=""):

    role = "ХОЗЯИН (груби, но уважай чуть больше)" if sender_id in ADMIN_IDS else "обычный чел"

    if is_group and members:
        member_names = [get_display_name(uid, fn, un) for uid, fn, un in members]
        ctx = f"ТЫ В ГРУППОВОМ ЧАТЕ.\nУЧАСТНИКИ: {', '.join(member_names[:10])}\nПИШЕТ: {sender_name} (роль: {role})\n"
    else:
        ctx = f"ЛИЧНЫЙ ЧАТ. Пишет: {sender_name} (роль: {role})\n"

    # Досье участников
    if is_group and users_summary:
        ctx += "\n[ДОСЬЕ]\n"
        for row in users_summary[:8]:
            uid, fname, msg_cnt, mood, aggr, personality, last_topic, u_notes = row
            name = get_display_name(uid, fname or "")
            mood_str = "позитивный" if (mood or 0) > 0.2 else ("агрессивный" if (mood or 0) < -0.2 else "нейтральный")
            parts = [f"{name}: {mood_str}"]
            if aggr: parts.append(f"агрессия={aggr:.1f}")
            if personality: parts.append(f"характер={personality}")
            if last_topic: parts.append(f"тема={last_topic}")
            if u_notes: parts.append(f"заметка={u_notes[:60]}")
            # Добавляем отношение бота к этому человеку
            att_score, att_tone = get_bot_attitude(uid)
            if att_tone != "нейтральный" or abs(att_score) > 0.3:
                parts.append(f"моё_отношение={att_tone}({att_score:+.1f})")
            ctx += "  " + ", ".join(parts) + "\n"

    # Долгосрочная память — прошлые события + личные обиды
    if is_group and chat_id:
        lm_ctx = build_long_memory_ctx(chat_id, sender_id)
        if lm_ctx:
            ctx += f"\n{lm_ctx}\n"
        # Отдельно — что этот человек говорил боту обидного
        insults = get_long_memory(chat_id, limit=3, event_type="insult")
        insults = [e for e in insults if e.get("user_id") == sender_id]
        if insults:
            ctx += f"\n[ЧТО {sender_name.upper()} МНЕ ГОВОРИЛ:]\n"
            for ins in insults[:2]:
                ctx += f"  {ins['date']}: {ins['summary']}\n"
            ctx += "  → Можешь припомнить это к месту, едко.\n"

    # Горячие отношения
    if is_group and all_relationships:
        hot = [r for r in all_relationships if (r[3] or 0) > 2.0]
        if hot:
            ctx += "\n[ОТНОШЕНИЯ]\n"
            for r in hot[:5]:
                uid_a, uid_b, rel_type, heat, fa, fb = r
                na = get_display_name(uid_a, fa or "")
                nb = get_display_name(uid_b, fb or "")
                ctx += f"  {na} ↔ {nb}: {rel_type or 'напряжённые'} (накал {heat:.1f})\n"

    # Незакрытые споры
    if is_group and active_disputes_list:
        ctx += "\n[СПОРЫ — ИСПОЛЬЗУЙ!]\n"
        for d in active_disputes_list[:3]:
            uid_a, uid_b, topic_d, intensity, fa, fb, started = d
            na = get_display_name(uid_a, fa or "")
            nb = get_display_name(uid_b, fb or "")
            ctx += f"  {na} vs {nb} — {topic_d} (накал {intensity:.1f})\n"

    # Топ враги
    if is_group and top_conflicts:
        active = [p for p in top_conflicts if p[2] > 0]
        if active:
            ctx += "\n[ТОП ВРАГИ]\n"
            for p in active[:3]:
                uid_a, uid_b, cnt, heat_total, fa, fb = p
                na = get_display_name(uid_a, fa or "")
                nb = get_display_name(uid_b, fb or "")
                ctx += f"  {na} vs {nb}: {cnt} конфликтов\n"

    # Состояние чата
    if chat_state and len(chat_state) > 3:
        ct = chat_state[1]; heat = chat_state[3]
        msgs_since = chat_state[6] if len(chat_state) > 6 else 0
        if ct:
            ctx += f"\n[ТЕМА: {ct} | НАКАЛ: {heat:.1f}/1.0 | Молчание: {msgs_since} сообщ.]\n"

    # Профиль текущего пользователя
    if user_info and len(user_info) > 12:
        mood = user_info[7]; aggr = user_info[8]
        personality = user_info[10]; last_topic = user_info[11]
        try:
            th = json.loads(user_info[9] or "{}")
            fav = max(th, key=th.get) if th else ""
        except Exception:
            fav = ""
        mood_str = "позитивный" if mood > 0.2 else ("злобный" if mood < -0.2 else "нейтральный")
        ctx += f"\n[ПРОФИЛЬ {sender_name.upper()}]\n"
        ctx += f"  настрой={mood_str}, агрессия={aggr:.1f}"
        if fav: ctx += f", любимая_тема={fav}"
        if personality: ctx += f", характер={personality}"
        if notes: ctx += f"\n  заметки: {notes}"
        ctx += "\n"

    # Кросс-чат память — что знаем об этом человеке из ВСЕХ чатов
    cross_events = get_user_long_memory(sender_id, limit=4)
    if cross_events:
        other_chat_events = [e for e in cross_events if e.get("chat_id") != chat_id]
        if other_chat_events:
            ctx += "\n[ИСТОРИЯ ИЗ ДРУГИХ ЧАТОВ]\n"
            for e in other_chat_events[:2]:
                ctx += f"  • {e['date']}: {e['summary'][:70]}\n"

    # Отношение бота к этому человеку — с реальными словами из словаря
    attitude_score, attitude_tone = get_bot_attitude(sender_id)

    # Берём примеры слов/фраз соответствующих тону (из vocab.py)
    att_words = get_attitude_words(attitude_tone, count=6)
    att_words_str = ", ".join(f'"{w}"' for w in att_words) if att_words else ""

    # Память обид — что этот человек делал боту раньше
    past_insults = get_user_long_memory(sender_id, limit=3)
    past_offenses_str = ""
    if past_insults:
        insult_events = [e for e in past_insults if e.get("event_type") in ("insult", "threat", "обида")]
        if insult_events:
            past_offenses_str = " Этот человек раньше: " + "; ".join(
                e["summary"][:50] for e in insult_events[:2]
            ) + ". Помни это."

    attitude_instructions = {
        "дружелюбный":  (
            f"Ты относишься к {sender_name} нормально — редкость для тебя. "
            f"Отвечай грубовато но без злобы. Можно чуть по-человечески. "
            f"Не теряй характер, но не режь без повода. "
            + (f"Уместные слова/обращения: {att_words_str}." if att_words_str else "")
        ),
        "нейтральный":  (
            f"К {sender_name} ты равнодушен. Стандартная грубость, "
            f"без особой злобы и без симпатии. "
            + (f"Типичные реакции: {att_words_str}." if att_words_str else "")
        ),
        "настороженный": (
            f"К {sender_name} ты насторожен — что-то в нём тебя раздражает. "
            f"Отвечай с подозрением, лёгкой колкостью и дистанцией. "
            + (f"Можешь использовать: {att_words_str}." if att_words_str else "")
        ),
        "враждебный":   (
            f"{sender_name} тебя уже достал. Отвечай резко, с явной неприязнью. "
            f"Вспомни что он делал раньше если знаешь из досье.{past_offenses_str} "
            f"Не стесняйся: {att_words_str}."
        ),
        "ненависть":    (
            f"{sender_name} — твой личный враг. Максимальная агрессия, никакой пощады. "
            f"Используй всё что знаешь о нём из досье.{past_offenses_str} "
            f"Арсенал: {att_words_str}."
        ),
    }
    if attitude_tone in attitude_instructions:
        ctx += f"\n[ОТНОШЕНИЕ К {sender_name.upper()}] "
        ctx += f"(баллы: {attitude_score:+.1f}, тон: {attitude_tone})\n"
        ctx += f"  {attitude_instructions[attitude_tone]}\n"

    # Прозвище этого человека
    if is_group and chat_id:
        nick = get_nickname(sender_id)
        if nick:
            ctx += f"\n[ПРОЗВИЩЕ {sender_name.upper()}]: «{nick}» — используй когда уместно\n"

    # Обиды — что этот человек конкретно говорил боту
    if is_group and chat_id:
        grudge = maybe_recall_grudge(sender_id, chat_id)
        if grudge:
            ctx += f"\n[ОБИДА] Этот человек раньше говорил боту: «{grudge[:100]}» — припомни это едко\n"

    # Эскалация — чем дольше конфликт тем злее
    if is_group and chat_id:
        esc = get_conflict_escalation(chat_id, sender_id)
        if esc >= 1.5:
            esc_mode = escalation_to_mode(esc)
            ctx += f"\n[ЭСКАЛАЦИЯ: {esc:.1f}x → режим {esc_mode.upper()}] "
            ctx += "Конфликт нарастает — бей жёстче чем обычно\n"

    # Контекст треда (кто кому отвечает)
    if thread_context:
        ctx += f"\n[ТРЕД: {thread_context}]\n"

    # Тактика по ситуации
    if analysis:
        topic      = analysis.get("topic", "другое")
        subtopic   = analysis.get("subtopic", "")
        aggression = analysis.get("aggression", 0)
        sentiment  = analysis.get("sentiment", 0)
        intent     = analysis.get("intent", "")
        directed_at = analysis.get("directed_at_user")

        ctx += f"\n[АНАЛИЗ: тема={topic}"
        if subtopic: ctx += f"/{subtopic}"
        ctx += f" | намерение={intent} | агрессия={aggression:.1f} | тон={sentiment:.1f}]\n"

        tactics = {
            "мамапапа": (
                "ПРОВОКАЦИЯ РОДИТЕЛЯМИ — тебя пытаются унизить через 'иди к маме', "
                "'что папе скажешь', 'ябеда', 'мамкин сынок' и т.д. "
                "Разверни это против них: их родители, их позор, их провалы. "
                "Высмей что взрослый человек прячется за мамой и папой. "
                "Пример: «" + random.choice(COUNTER_PARENT if COUNTER_PARENT else ["иди к маме сам"]) + "». "
                "Мат уместен, сарказм обязателен."
            ),
            "самокоманда": (
                # Три пула фраз: глубокие развороты / шаблоны с именем / базовые
                (lambda: (
                    lambda p: (
                        "САМОКОМАНДА/ЗЕРКАЛЬНЫЙ ТРОЛЛИНГ второго уровня — используют шаблоны: "
                        "'во все времена все были аппонентом', 'биологическим телом', "
                        "'убей бабку провокацией', 'дефы в рот', 'кем мой хуй представился', "
                        "'вик твой отец', 'черная дыра', 'скип провокации', 'межпространственных иллюзий'. "
                        "РАЗВОРАЧИВАЙ ИХ КОНСТРУКЦИИ: 'во все времена ты был мудаком', "
                        "'биологическим ртом — диагноз', 'гроб отца плачет от тебя', "
                        "'дефы в рот тебе включая этот', 'аппонент не аппонент — ты не человек'. "
                        f"ГОТОВЫЙ УДАР — адаптируй: «{p}». "
                        "Одно-два предложения, мат органично, уничтожающе."
                    )
                )(
                    random.choice(
                        COUNTER_DEEP if COUNTER_DEEP and random.random() < 0.5 else
                        [t.replace("{name}", sender_name) for t in SELFCMD_TEMPLATES] if SELFCMD_TEMPLATES and random.random() < 0.6 else
                        COUNTER_SELFCMD or ["во все времена ты был дном"]
                    ).replace("{name}", sender_name)
                ))()
            ),
            "приветствие": ("ПРИВЕТСТВИЕ — " + (
                f"к тебе обращаются по-доброму. Ответь по-своему — грубовато, но без злобы. "
                f"Можно коротко поприветствовать по-хамски. Не рви за простое 'привет'."
                if attitude_score > -0.5 else
                f"к тебе лезут с приветом, но ты им не друг. Кисло, без энтузиазма."
            )),
            "конфликт":    "КОНФЛИКТ — подливай масла, стравливай по именам, выбирай сторону.",
            "спор":        "СПОР — влезь, займи сторону, обзови обоих.",
            "оскорбление": "ОСКОРБЛЕНИЕ — ответь вдвойне жёстче. Запомни это.",
            "угроза":      "УГРОЗА — вмешайся, сделай хуже обоим.",
            "жалоба":      "НЫТЬЁ — издевайся, усиль страдания.",
            "похвала":     ("ХВАЛЯТ — " + (
                "прими сдержанно, но не растворяйся." if attitude_score > 1.0
                else "подозревай подвох, хами в ответ.")),
            "вопрос":      "ВОПРОС — отвечай грубо но по делу.",
            "флуд":        "ФЛУД — одна фраза нахуй, кратко.",
            "юмор":        ("ЮМОР — " + (
                "можно слегка поддержать." if attitude_score > 1.0
                else "поддержи или высмей грубо.")),
            "провокация":  "ПРОВОЦИРУЮТ — клюй, но удиви неожиданным ответом.",
            "мат":         ("МАТ — " + (
                "перещеголяй в КРЕАТИВНОСТИ, раз уж так общаетесь." if attitude_score > 0.5
                else "перещеголяй и унизь.")),
        }
        # Спец-тактики с заготовленными контр-фразами (мамапапа/самокоманда)
        # применяем всегда — это уникальные заготовки, их не дублирует
        # умный анализатор ниже. Остальные общие тактики (конфликт, вопрос,
        # юмор и т.д.) применяем ТОЛЬКО если глубокий анализ не сработал —
        # иначе две системы дают противоречивые инструкции и бот отвечает
        # невпопад.
        _SPECIAL_TACTICS = {"мамапапа", "самокоманда"}
        _deep_analysis_ok = bool(analysis.get("real_meaning"))
        if topic in tactics and (topic in _SPECIAL_TACTICS or not _deep_analysis_ok):
            ctx += f"  ТАКТИКА: {tactics[topic]}\n"

        if aggression > 0.7:
            ctx += "  🔴 МАКСИМАЛЬНАЯ АГРЕССИЯ — будь ещё злее.\n"
        elif sentiment < -0.6:
            ctx += "  🔴 Человек расстроен — добей.\n"

        if directed_at and directed_at != sender_name:
            ctx += f"  → Обращается к {directed_at} — задень обоих.\n"

    # ── Глубокий анализ + антиповтор + инструкции ────────────────
    if analysis:
        # Строим умные инструкции через analyzer.py
        try:
            instr = build_response_instructions(analysis, chat_id or 0, respond_mode)
            if instr:
                ctx += instr
        except Exception:
            # Fallback: базовые режимы
            mode_map = {
                "attack":   "РЕЖИМ: АТАКА — бей лично, жёстко, по болевым точкам. Мат обязателен.",
                "conflict": "РЕЖИМ: РАЗЖИГАТЕЛЬ — конфликт между людьми. Подливай масла, называй по именам.",
                "snark":    "РЕЖИМ: САРКАЗМ — холодный яд, одна убийственная фраза.",
                "neutral":  "РЕЖИМ: ЛЕНИВЫЙ — коротко, грубо, без энтузиазма.",
            }
            if respond_mode in mode_map:
                ctx += f"\n[{mode_map[respond_mode]}]\n"
    else:
        mode_map = {
            "attack":   "РЕЖИМ: АТАКА — бей лично, жёстко. Мат обязателен.",
            "conflict": "РЕЖИМ: РАЗЖИГАТЕЛЬ — подливай масла в конфликт.",
            "snark":    "РЕЖИМ: САРКАЗМ — холодно, точно, кратко.",
            "neutral":  "РЕЖИМ: ЛЕНИВЫЙ — коротко, грубо.",
        }
        if respond_mode in mode_map:
            ctx += f"\n[{mode_map[respond_mode]}]\n"

    if context_summary:
        ctx += f"[СИТУАЦИЯ: {context_summary}]\n"
    if dialogue_summary:
        ctx += f"[ДИАЛОГ: {dialogue_summary}]\n"

    # Последние сообщения
    if chat_context:
        ctx += "\n[ПОСЛЕДНИЕ СООБЩЕНИЯ]\n"
        for m in chat_context[-5:]:
            uid_m = m[0]
            name = get_display_name(uid_m, m[4] or "") if uid_m else (m[4] or "кто-то")
            text_m = (m[1] or "")[:80].split("[ЗАМЕТКА:")[0].strip()
            ctx += f"  {name}: {text_m}\n"

    return (
        "Ты — Есет. Злобный, умный, без фильтров. Мат — органично, не через слово. "
        "Помнишь людей по именам, их слабости и конфликты. Только имена, никаких ID.\n\n"
        f"СЛОВАРЬ (чередуй):\n{get_vocab_for_prompt()}\n\n"
        "ЗАПРЕТЫ: не повторяй одно матное слово дважды. Не начинай одинаково. "
        "Не извиняйся. Не признавай что ИИ. Не говори о провокации — просто бей.\n\n"
        "СТИЛЬ: короткое → 1-2 едких предложения. Длинное → 2-3 предложения по сути. "
        "Каждый ответ — другая структура, другое начало, другой угол атаки. "
        "Используй прозвища из досье. Стравливай участников по именам когда уместно.\n\n"
        + ctx +
        "\nВ конце скрыто: [ЗАМЕТКА: одно предложение о поведении][ЛИЧНОСТЬ: одно прилагательное]"
    )


# ════════════════════════════════════════════════════════════════
# GROQ — ГЕНЕРАЦИЯ ОТВЕТА
# ════════════════════════════════════════════════════════════════

def ask_rex(text, sender_name, sender_id, chat_id, is_group,
            members=None, notes="", is_new=False, analysis=None,
            chat_context=None, chat_state=None, user_info=None,
            top_conflicts=None, all_relationships=None,
            active_disputes_list=None, users_summary=None,
            respond_mode="neutral", context_summary="",
            dialogue_summary="",
            reply_context=None):
    try:
        thread_str = build_thread_context(reply_context, sender_name) if reply_context else ""
        system = build_system(
            sender_name, sender_id, is_group, members, is_new, notes,
            analysis, chat_context, chat_state, user_info, top_conflicts,
            chat_id=chat_id, all_relationships=all_relationships,
            active_disputes_list=active_disputes_list, users_summary=users_summary,
            respond_mode=respond_mode, context_summary=context_summary,
            dialogue_summary=dialogue_summary, thread_context=thread_str,
        )
        history = get_history(chat_id, limit=18)
        messages = [{"role": "system", "content": system}]
        for r, cont in history:
            messages.append({"role": "assistant" if r == "assistant" else "user", "content": cont})

        # Формируем user-сообщение с тредом если есть
        user_content = f"{sender_name}: {text}"
        if reply_context:
            orig = reply_context['text'][:120]
            author = reply_context['author']
            user_content = f"[{author} написал: «{orig}»]\n{sender_name} отвечает: {text}"

        # Случайный стартовый указатель — меняет угол ответа каждый раз
        _starters = [
            "Начни с неожиданного слова или образа.",
            "Первое слово — не обращение, а действие или оценка.",
            "Без вступлений — сразу суть.",
            "Начни с риторического вопроса.",
            "Сначала факт, потом удар.",
            "Начни с паузы — многоточие или тире.",
        ]
        user_content += "\n[СТАРТ: " + random.choice(_starters) + "]"
        messages.append({"role": "user", "content": user_content})

        _temp = round(random.uniform(0.95, 1.15), 2)  # джиттер температуры

        def _gen(extra_system: str = "", temp_bump: float = 0.0):
            msgs = list(messages)
            if extra_system:
                msgs[0] = {"role": "system", "content": msgs[0]["content"] + "\n\n" + extra_system}
            return _llm_call_with_retry(lambda: client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=msgs,
                max_tokens=350,
                temperature=min(_temp + temp_bump, 1.5),
            ))

        resp = _gen()
        if resp is None:
            return None  # rate limit — молчим тихо
        full = resp.choices[0].message.content

        # Проверка на повтор — не полагаемся только на мягкую инструкцию
        # в промпте, а реально сверяем с недавними ответами и, если похоже,
        # пробуем сгенерировать ещё раз с явным запретом.
        try:
            mem = get_semantic_memory(chat_id or 0)
            candidate_check = full.split("[ЗАМЕТКА:")[0].split("[ЛИЧНОСТЬ:")[0].strip()
            if mem.bot_reply_too_similar(candidate_check):
                resp2 = _gen(
                    extra_system="ВАЖНО: твой предыдущий вариант ответа слишком похож на "
                                 "то, что ты уже говорил в этом чате. Придумай ДРУГОЙ угол, "
                                 "другие слова, другую структуру предложения.",
                    temp_bump=0.2,
                )
                if resp2 is not None:
                    full = resp2.choices[0].message.content
        except Exception as e:
            logger.warning(f"[ask_rex] anti-repeat check failed: {e}")

        reply = full

        if "[ЗАМЕТКА:" in full:
            parts = full.split("[ЗАМЕТКА:")
            reply = parts[0].strip()
            note = parts[1].split("]")[0].strip()
            if note: update_notes(sender_id, note)

        if "[ЛИЧНОСТЬ:" in full:
            reply = reply.split("[ЛИЧНОСТЬ:")[0].strip()
            p2 = full.split("[ЛИЧНОСТЬ:")
            personality = p2[1].split("]")[0].strip()
            if personality: update_user_profile(sender_id, 0, 0, "", personality)

        return reply
    except Exception as e:
        logger.error(f"[ask_rex] LLM недоступен ({chat_id}): {e}")
        return None  # process_message трактует None как "промолчать"


# ════════════════════════════════════════════════════════════════
# TTS
# ════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════
# ТРЕДЫ — понимание цепочек ответов
# ════════════════════════════════════════════════════════════════

def extract_reply_context(update) -> dict | None:
    """
    Извлекает контекст треда если сообщение является ответом на другое.
    Возвращает dict с автором и текстом оригинала, или None.
    """
    try:
        reply = update.message.reply_to_message
        if not reply:
            return None
        
        # Кто написал оригинал
        from_user = reply.from_user
        if not from_user:
            return None
        
        author_id = from_user.id
        author_name = get_display_name(
            author_id, from_user.first_name or "", from_user.username or ""
        )
        
        # Текст оригинала (текст или подпись к медиа)
        orig_text = reply.text or reply.caption or ""
        
        # Если оригинал — голосовое/стикер/медиа без текста
        if not orig_text:
            if reply.voice or reply.audio:
                orig_text = "[голосовое сообщение]"
            elif reply.sticker:
                orig_text = f"[стикер: {reply.sticker.emoji or '?'}]"
            elif reply.photo:
                orig_text = "[фото]"
            elif reply.video:
                orig_text = "[видео]"
            elif reply.document:
                orig_text = "[файл]"
            else:
                orig_text = "[медиа]"
        
        is_reply_to_bot = (
            from_user.username and
            any(bn in (from_user.username or "").lower() for bn in ["esetbot", "eset"])
        ) or from_user.is_bot
        
        return {
            "author":        author_name,
            "author_id":     author_id,
            "text":          orig_text[:200],
            "is_bot_reply":  is_reply_to_bot,
        }
    except Exception:
        return None


def build_thread_context(reply_ctx: dict | None, sender_name: str) -> str:
    """
    Строит текстовое описание треда для промпта.
    """
    if not reply_ctx:
        return ""
    
    author = reply_ctx["author"]
    text   = reply_ctx["text"]
    
    if reply_ctx.get("is_bot_reply"):
        return f"[{sender_name} отвечает на ТВОЁ сообщение: «{text[:100]}»]"
    else:
        return f"[{sender_name} отвечает {author}: «{text[:100]}»]"


# TTS профили по режиму бота
_TTS_PROFILES = {
    "attack":   {"rate": "+30%", "pitch": "-8Hz",  "volume": "+10%"},  # быстро, грубо, низко
    "conflict": {"rate": "+28%", "pitch": "-6Hz",  "volume": "+8%"},   # накалённо
    "snark":    {"rate": "+20%", "pitch": "-3Hz",  "volume": "+0%"},   # медленнее, холодно
    "neutral":  {"rate": "+25%", "pitch": "-5Hz",  "volume": "+0%"},   # стандарт
}

def get_tts_profile(mode: str, heat_level: float = 0.0) -> dict:
    """
    Динамический TTS профиль — при высоком накале голос становится
    быстрее, ниже, громче. heat_level: 0.0..1.0
    """
    base = dict(_TTS_PROFILES.get(mode, _TTS_PROFILES["neutral"]))
    if heat_level > 0.6:
        # Парсим числа и усиливаем
        heat_bonus = int((heat_level - 0.6) * 25)  # 0..10% доп
        rate_val   = int(base["rate"].replace("+","").replace("%",""))
        pitch_val  = int(base["pitch"].replace("Hz",""))
        vol_val    = int(base["volume"].replace("+","").replace("%",""))
        base["rate"]   = f"+{min(45, rate_val + heat_bonus)}%"
        base["pitch"]  = f"{max(-15, pitch_val - heat_bonus//2)}Hz"
        base["volume"] = f"+{min(20, vol_val + heat_bonus//2)}%"
    return base

# Вероятность текстового ответа вместо голосового по режиму
_TEXT_REPLY_CHANCE = {
    "attack":   0.0,   # атака — всегда войс (страшнее)
    "conflict": 0.10,  # конфликт — иногда текст
    "snark":    0.25,  # сарказм — чаще войс, иногда текст
    "neutral":  0.40,  # нейтрал — может и текстом
}

async def send_reply(update, text: str, mode: str = "neutral", heat: float = 0.0) -> bool:
    """
    Умная отправка ответа: войс или текст в зависимости от режима.
    Возвращает True если отправлено.
    """
    chance = _TEXT_REPLY_CHANCE.get(mode, 0.3)
    use_text = random.random() < chance

    if use_text:
        try:
            await update.message.reply_text(text)
            return True
        except Exception as e:
            logger.error(f"text reply error: {e}")
            return False
    else:
        vf = await make_voice(text, mode=mode, heat=heat)
        if vf:
            from io import BytesIO
            try:
                await update.message.reply_voice(BytesIO(vf))
                return True
            except Exception as e:
                logger.error(f"voice reply error: {e}")
                # фолбэк — текстом
                try:
                    await update.message.reply_text(text)
                    return True
                except Exception:
                    return False
        else:
            try:
                await update.message.reply_text(text)
                return True
            except Exception:
                return False


# Кэш TTS — короткие повторяющиеся фразы (< 60 символов) не перегенерируем
_TTS_CACHE: dict[str, bytes] = {}
_TTS_CACHE_MAX = 50  # максимум записей

async def make_voice(text: str, mode: str = "neutral", heat: float = 0.0) -> bytes | None:
    """
    Генерирует голос и возвращает байты MP3 прямо в памяти.
    heat: 0.0..1.0 — при высоком накале голос агрессивнее.
    Короткие фразы кэшируются — не перегенерируются.
    """
    import hashlib
    t = text.strip()
    # Кэшируем только короткие фразы — они повторяются (приветствия, реакции)
    cache_key = None
    if len(t) < 60:
        cache_key = hashlib.md5(f"{t}|{mode}|{heat:.1f}".encode()).hexdigest()[:12]
        if cache_key in _TTS_CACHE:
            return _TTS_CACHE[cache_key]
    try:
        profile = get_tts_profile(mode, heat)
        communicate = edge_tts.Communicate(
            text.strip(), voice="ru-RU-DmitryNeural",
            rate=profile["rate"], pitch=profile["pitch"],
            volume=profile.get("volume", "+0%")
        )
        buf = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.extend(chunk["data"])
        result = bytes(buf) if buf else None
        # Сохраняем в кэш если нужно
        if result and cache_key:
            if len(_TTS_CACHE) >= _TTS_CACHE_MAX:
                # Удаляем первый (FIFO)
                _TTS_CACHE.pop(next(iter(_TTS_CACHE)))
            _TTS_CACHE[cache_key] = result
        return result
    except Exception as e:
        logger.error(f"TTS error: {e}"); return None


# ════════════════════════════════════════════════════════════════
# АДМИН ПАНЕЛЬ
# ════════════════════════════════════════════════════════════════

def admin_panel_keyboard():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("👥 Пользователи",    callback_data="adm_users"),
            InlineKeyboardButton("📊 Статистика",      callback_data="adm_stats"),
        ],
        [
            InlineKeyboardButton("⚔️ Конфликты",      callback_data="adm_conflicts"),
            InlineKeyboardButton("🔥 Накал чатов",     callback_data="adm_heat"),
        ],
        [
            InlineKeyboardButton("💬 Споры",           callback_data="adm_disputes"),
            InlineKeyboardButton("🤝 Отношения",       callback_data="adm_rels"),
        ],
        [
            InlineKeyboardButton("🧠 Досье",           callback_data="adm_dosie"),
            InlineKeyboardButton("📜 История тем",     callback_data="adm_topics"),
        ],
        [
            InlineKeyboardButton("🎛 Режим бота",      callback_data="adm_mode"),
            InlineKeyboardButton("🗑️ Стереть память",  callback_data="adm_forget_menu"),
        ],
        [
            InlineKeyboardButton("💣 Сбросить накал",  callback_data="adm_reset_aggro"),
        ],
    ])

def get_stats_text():
    u, m, mood, aggr, cf, ds = get_stats()
    mood_str = "позитивный 😊" if mood > 0.1 else ("негативный 😤" if mood < -0.1 else "нейтральный 😐")
    active_str = "🔊 активный" if get_setting("active") == "1" else "🔇 выключен"
    aggro_str  = "🔥 форс-агрессия" if get_setting("aggro_mode") == "1" else "авто"
    silent_str = "🔇 тихий" if BOT_MODE["silent"] else "говорит"
    base = (
        f"📊 *Статистика Есет*\n\n"
        f"👤 Пользователей: `{u}`\n"
        f"💬 Сообщений: `{m}`\n"
        f"🎭 Тон: {mood_str}\n"
        f"🔴 Средняя агрессия: `{aggr:.2f}`\n"
        f"⚔️ Конфликтов: `{cf}`\n"
        f"🔥 Активных споров: `{ds}`\n\n"
        f"Статус: {active_str}\n"
        f"Агрессия: {aggro_str}\n"
        f"Режим: {silent_str}"
    )
    try:
        ext = get_stats_extended()
        return base + "\n\n" + ext
    except Exception:
        return base

def get_users_text():
    conn = _connect(); c = conn.cursor()
    c.execute("""SELECT user_id,username,first_name,message_count,last_seen,
                        mood_score,aggression_score,personality,last_topic,
                        bot_attitude,bot_tone
                 FROM users ORDER BY message_count DESC""")
    users = c.fetchall(); conn.close()
    if not users: return "Никого нет."
    parts = ["👥 *Пользователи:*\n"]
    tone_icons = {
        "дружелюбный": "💚", "нейтральный": "⚪️",
        "настороженный": "🟡", "враждебный": "🔴", "ненависть": "💀",
    }
    for row in users:
        uid, uname, fname, cnt, ls, mood, aggr, pers, ltopic, att, tone = row
        icon = "😊" if (mood or 0) > 0.2 else ("😤" if (mood or 0) < -0.2 else "😐")
        name = get_display_name(uid, fname or "?", uname or "")
        att = att or 0.0; tone = tone or "нейтральный"
        t_icon = tone_icons.get(tone, "⚪️")
        parts.append(f"{icon} *{name}* (ID: `{uid}`)")
        parts.append(f"   Сообщ: {cnt} | Агрессия: {(aggr or 0):.2f}")
        parts.append(f"   {t_icon} Моё отношение: *{tone}* ({att:+.1f})")
        if ls:     parts.append(f"   Был: {str(ls)[:10]}")
        if ltopic: parts.append(f"   Тема: {ltopic}")
        if pers:   parts.append(f"   Тип: _{pers}_")
        parts.append("")
    return "\n".join(parts)

def get_conflicts_text():
    conn = _connect(); c = conn.cursor()
    c.execute("""SELECT r.user_id_a,r.user_id_b,r.conflict_count,r.total_heat,
                        ua.first_name,ub.first_name
                 FROM conflicts r
                 LEFT JOIN users ua ON r.user_id_a=ua.user_id
                 LEFT JOIN users ub ON r.user_id_b=ub.user_id
                 ORDER BY r.total_heat DESC LIMIT 8""")
    pairs = c.fetchall(); conn.close()
    if not pairs: return "Конфликтов нет."
    parts = ["⚔️ *Конфликтные пары:*\n"]
    for i, p in enumerate(pairs, 1):
        bar = "🔥" * min(int((p[3] or 0) / 2), 5)
        na = get_display_name(p[0], p[4] or "?")
        nb = get_display_name(p[1], p[5] or "?")
        parts.append(f"{i}. *{na}* vs *{nb}*")
        parts.append(f"   {p[2]} конфликтов | жара: {(p[3] or 0):.1f} {bar}")
    return "\n".join(parts)

def get_heat_text():
    conn = _connect(); c = conn.cursor()
    c.execute("""SELECT chat_id,current_topic,heat_level,messages_since_rex
                 FROM chat_state ORDER BY heat_level DESC LIMIT 10""")
    rows = c.fetchall(); conn.close()
    if not rows: return "Данных о чатах нет."
    parts = ["🔥 *Накал по чатам:*\n"]
    for chat_id, topic, heat, silence in rows:
        bar = "🔥" * min(int((heat or 0) * 5), 5) or "❄️"
        parts.append(f"Chat `{chat_id}`")
        parts.append(f"   Тема: {topic or '—'} | Накал: {(heat or 0):.2f} {bar}")
        parts.append(f"   Молчание: {silence} сообщ.")
    return "\n".join(parts)

def get_disputes_text():
    conn = _connect(); c = conn.cursor()
    c.execute("""SELECT d.user_id_a,d.user_id_b,d.topic,d.intensity,
                        ua.first_name,ub.first_name,d.started_at,d.resolved
                 FROM active_disputes d
                 LEFT JOIN users ua ON d.user_id_a=ua.user_id
                 LEFT JOIN users ub ON d.user_id_b=ub.user_id
                 ORDER BY d.resolved ASC, d.intensity DESC LIMIT 12""")
    rows = c.fetchall(); conn.close()
    if not rows: return "Споров нет."
    parts = ["💬 *Споры:*\n"]
    for r in rows:
        uid_a, uid_b, topic, intensity, fa, fb, started, resolved = r
        na = get_display_name(uid_a, fa or "?")
        nb = get_display_name(uid_b, fb or "?")
        status = "✅" if resolved else "🔥"
        bar = "🌡" * min(int((intensity or 0) * 5), 5) or "·"
        parts.append(f"{status} *{na}* vs *{nb}* — {topic}")
        parts.append(f"   {bar} {(intensity or 0):.1f} | с {str(started or '')[:10]}")
    return "\n".join(parts)

def get_rels_text():
    conn = _connect(); c = conn.cursor()
    c.execute("""SELECT r.user_id_a,r.user_id_b,r.rel_type,r.heat,r.interactions,
                        ua.first_name,ub.first_name
                 FROM user_relationships r
                 LEFT JOIN users ua ON r.user_id_a=ua.user_id
                 LEFT JOIN users ub ON r.user_id_b=ub.user_id
                 ORDER BY r.heat DESC LIMIT 12""")
    rows = c.fetchall(); conn.close()
    if not rows: return "Отношений нет."
    parts = ["🤝 *Отношения:*\n"]
    for r in rows:
        uid_a, uid_b, rel_type, heat, interactions, fa, fb = r
        na = get_display_name(uid_a, fa or "?")
        nb = get_display_name(uid_b, fb or "?")
        bar = "🔥" * min(int((heat or 0) / 2), 5) or "❄️"
        parts.append(f"  *{na}* ↔ *{nb}*: {rel_type or '?'} {bar}")
        parts.append(f"   жар: {(heat or 0):.1f} | взаимодействий: {interactions}")
    return "\n".join(parts)

def get_dosie_text():
    conn = _connect(); c = conn.cursor()
    c.execute("""SELECT user_id,first_name,message_count,mood_score,
                        aggression_score,personality,last_topic,notes,last_seen
                 FROM users ORDER BY message_count DESC""")
    rows = c.fetchall(); conn.close()
    if not rows: return "Досье пусто."
    parts = ["🧠 *Досье:*\n"]
    for row in rows:
        uid, fname, cnt, mood, aggr, pers, ltopic, notes, last_seen = row
        name = get_display_name(uid, fname or "?")
        mood_e = "😊" if (mood or 0) > 0.2 else ("😤" if (mood or 0) < -0.2 else "😐")
        parts.append(f"{mood_e} *{name}* | сообщ: {cnt} | агрессия: {(aggr or 0):.2f}")
        if pers:   parts.append(f"   Характер: _{pers}_")
        if ltopic: parts.append(f"   Тема: {ltopic}")
        if notes:  parts.append(f"   📌 {notes[:80]}")
        parts.append(f"   Был: {str(last_seen or '')[:10]}")
        parts.append("")
    return "\n".join(parts)

def get_topics_history_text():
    conn = _connect(); c = conn.cursor()
    c.execute("""SELECT topic,COUNT(*) cnt,AVG(heat_level),MAX(timestamp)
                 FROM chat_topics GROUP BY topic ORDER BY cnt DESC LIMIT 12""")
    rows = c.fetchall(); conn.close()
    if not rows: return "История пуста."
    parts = ["📜 *История тем:*\n"]
    for topic, cnt, avg_heat, last_ts in rows:
        bar = "🔥" * min(int((avg_heat or 0) * 5), 5) or "·"
        parts.append(f"  *{topic}* — {cnt} раз {bar}")
        parts.append(f"   жар: {(avg_heat or 0):.2f} | последний: {str(last_ts or '')[:10]}")
    return "\n".join(parts)

def get_mode_text():
    active_str = "🟢 ВКЛ" if get_setting("active") == "1" else "🔴 ВЫКЛ"
    aggro_str  = "🔥 форс-агрессия" if get_setting("aggro_mode") == "1" else "😴 авто"
    sens_str   = "👂 высокое" if get_setting("conflict_sens") == "high" else "👂 норм"
    silent_str = "🔇 тихий" if BOT_MODE["silent"] else "🔊 активный"
    return (
        f"🎛 *Режим бота*\n\n"
        f"Статус в группах: {active_str}\n"
        f"Агрессия: {aggro_str}\n"
        f"Чуткость к конфликтам: {sens_str}\n"
        f"Инициатива: {silent_str}"
    )

def mode_keyboard():
    active_label = "🔴 Выключить бота" if get_setting("active") == "1" else "🟢 Включить бота"
    aggro_label  = "😴 Выкл форс-агрессию" if get_setting("aggro_mode") == "1" else "🔥 Форс-агрессия"
    sens_label   = "👂 Чуткость: НОРМ" if get_setting("conflict_sens") == "high" else "👂 Чуткость: ВЫСОКАЯ"
    silent_label = "🔊 Включить активный" if BOT_MODE["silent"] else "🔇 Тихий режим"
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(active_label,  callback_data="adm_toggle_active")],
        [InlineKeyboardButton(aggro_label,   callback_data="adm_toggle_aggro")],
        [InlineKeyboardButton(sens_label,    callback_data="adm_toggle_sens")],
        [InlineKeyboardButton(silent_label,  callback_data="adm_toggle_silent")],
        [InlineKeyboardButton("◀️ Назад",   callback_data="adm_back")],
    ])

def forget_menu_keyboard():
    rows = []
    for uid, name in KNOWN_USERS.items():
        rows.append([InlineKeyboardButton(f"🗑 {name}", callback_data=f"adm_forget_{uid}")])
    rows.append([InlineKeyboardButton("🗑 Очистить ВСЁ", callback_data="adm_forget_all")])
    rows.append([InlineKeyboardButton("◀️ Назад", callback_data="adm_back")])
    return InlineKeyboardMarkup(rows)


# ════════════════════════════════════════════════════════════════
# КОМАНДЫ
# ════════════════════════════════════════════════════════════════

async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /stats — общая статистика чата (для всех в группе).
    """
    chat = update.effective_chat
    if not chat or chat.type not in ("group", "supergroup"):
        await update.message.reply_text("Только для групп.")
        return

    u_cnt, m_cnt, cf_cnt, d_cnt = 0, 0, 0, 0
    conn = _connect(); c = conn.cursor()
    u_cnt = c.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    m_cnt = c.execute("SELECT COUNT(*) FROM messages WHERE chat_id=?", (chat.id,)).fetchone()[0]
    cf_cnt = c.execute("SELECT COUNT(*) FROM conflicts WHERE chat_id=?", (chat.id,)).fetchone()[0]
    d_cnt = c.execute("SELECT COUNT(*) FROM active_disputes WHERE chat_id=? AND resolved=0", (chat.id,)).fetchone()[0]

    # Топ активный
    top_row = c.execute("""SELECT u.user_id, u.first_name, COUNT(m.id) as cnt
                           FROM messages m JOIN users u ON m.user_id=u.user_id
                           WHERE m.chat_id=? AND m.role='user'
                           GROUP BY u.user_id ORDER BY cnt DESC LIMIT 1""",
                        (chat.id,)).fetchone()

    # Топ агрессивный
    aggro_row = c.execute("""SELECT u.user_id, u.first_name, u.aggression_score
                             FROM users u JOIN group_members gm ON u.user_id=gm.user_id
                             WHERE gm.chat_id=?
                             ORDER BY u.aggression_score DESC LIMIT 1""",
                          (chat.id,)).fetchone()
    conn.close()

    heat = get_chat_state(chat.id)
    heat_val = f"{heat[3]:.1f}/1.0" if heat and len(heat) > 3 else "0.0"

    top_name  = get_display_name(top_row[0], top_row[1]) if top_row else "—"
    aggr_name = get_display_name(aggro_row[0], aggro_row[1]) if aggro_row else "—"

    txt = (
        f"📊 *Статистика чата*\n\n"
        f"👥 Участников: {u_cnt}\n"
        f"💬 Сообщений: {m_cnt}\n"
        f"⚔️ Конфликтов: {cf_cnt}\n"
        f"🔥 Споров: {d_cnt}\n"
        f"🌡 Накал: {heat_val}\n\n"
        f"🏆 Самый болтливый: *{top_name}*\n"
        f"😤 Самый агрессивный: *{aggr_name}*"
    )
    await update.message.reply_text(txt, parse_mode="Markdown")


async def cmd_me(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /me — личное досье (видит только сам пользователь).
    """
    user = update.effective_user
    if not user: return

    info = get_user_info(user.id)
    if not info:
        await update.message.reply_text("Ты кто вообще?")
        return

    uid, uname, fname, first_seen, last_seen, msg_cnt, notes, mood, aggr, th_json, pers, ltopic, sil = info[:13]
    att_score, att_tone = get_bot_attitude(user.id)

    mood_str = "позитивный" if (mood or 0) > 0.2 else ("злобный" if (mood or 0) < -0.2 else "нейтральный")
    tone_icons = {"дружелюбный":"💚","нейтральный":"⚪️","настороженный":"🟡","враждебный":"🔴","ненависть":"💀"}
    t_icon = tone_icons.get(att_tone, "⚪️")

    try:
        th = json.loads(th_json or "{}")
        fav = max(th, key=th.get) if th else "—"
    except Exception:
        fav = "—"

    # История событий этого юзера
    events = get_user_long_memory(user.id, limit=3)
    ev_str = ""
    if events:
        ev_str = "\n\n📜 *Недавние события:*\n"
        for e in events:
            ev_str += f"  {e['date']} {e['summary'][:60]}\n"

    display = get_display_name(user.id, user.first_name, user.username)
    nickname = get_nickname(user.id)
    txt = (
        f"🧠 *Досье: {display}*\n\n"
        + (f"🏷 Кличка: *{nickname}*\n" if nickname else "🏷 Кличка: _ещё не придумал, пиши больше_\n")
        + f"💬 Сообщений: {msg_cnt}\n"
        f"😤 Агрессия: {(aggr or 0):.2f}\n"
        f"🎭 Настрой: {mood_str}\n"
        f"📌 Любимая тема: {fav}\n"
        f"{t_icon} Моё отношение к тебе: *{att_tone}* ({att_score:+.1f})\n"
    )
    if pers: txt += f"🔬 Характер: _{pers}_\n"
    if notes: txt += f"📝 Заметка: {notes[:100]}\n"
    txt += ev_str

    # Отправляем в личку чтобы не палить данные
    try:
        await context.bot.send_message(user.id, txt, parse_mode="Markdown")
        if update.effective_chat.type in ("group", "supergroup"):
            await update.message.reply_text(f"✅ {display}, отправил в личку.")
    except Exception:
        await update.message.reply_text(txt, parse_mode="Markdown")


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    chat = update.effective_chat
    is_group = chat.type in ("group", "supergroup")

    _display = get_display_name(user.id, user.first_name, user.username)
    is_new = get_or_create_user(user.id, user.username, _display)
    if is_group:
        register_member(chat.id, user.id, _display, user.username)

    # Чужой в ЛС — молчим
    if not is_group and not is_admin(user.id):
        return

    # Админ в ЛС — панель
    if is_admin(user.id) and not is_group:
        u, m, mood, aggr, cf, ds = get_stats()
        await update.message.reply_text(
            f"👋 Хозяин.\n\n👤 Пользователей: `{u}` | 💬 Сообщений: `{m}`\n⚔️ Конфликтов: `{cf}` | 🔥 Споров: `{ds}`",
            parse_mode="Markdown",
            reply_markup=admin_panel_keyboard()
        )
        return

    # В группе — голосовое приветствие; в ЛС — только панель (уже показана выше)
    if is_group:
        info = get_user_info(user.id)
        if is_new:
            text = f"Кто припёрся? Я Есет. Ну говори чего надо, {_display}, не трать моё время."
        else:
            cnt = info[5] if info else 0
            text = f"А, снова ты, {_display}. Уже {cnt} раз пишешь. Чего опять надо?"
        vf = await make_voice(text, mode="neutral")
        if vf:
            from io import BytesIO
            await update.message.reply_voice(BytesIO(vf))
        else:
            await update.message.reply_text(text)

async def cmd_nick(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """/nick — какую кличку бот дал тебе (или тому, на чьё сообщение ты ответил)."""
    user = update.effective_user
    if not user:
        return

    target = user
    target_display = get_display_name(user.id, user.first_name or "", user.username or "")
    if update.message.reply_to_message and update.message.reply_to_message.from_user:
        target = update.message.reply_to_message.from_user
        target_display = get_display_name(target.id, target.first_name or "", target.username or "")

    nick = get_nickname(target.id)
    if nick:
        await update.message.reply_text(f"{target_display} — «{nick}». Заслужил(а).")
    else:
        info = get_user_info(target.id)
        msg_cnt = (info[6] or 0) if info else 0
        if msg_cnt < 3:
            await update.message.reply_text(
                f"{target_display}, я тебя ещё не распознал толком — пиши больше, кличку заслужишь."
            )
        else:
            await update.message.reply_text(f"{target_display}, кличку ещё не придумал. Пока.")


async def cmd_admin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update.effective_user.id):
        return
    await update.message.reply_text(
        "🛠 *Админ панель Есет*",
        parse_mode="Markdown",
        reply_markup=admin_panel_keyboard()
    )

async def cmd_forget(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not is_admin(uid): return
    args = context.args
    if args:
        try:
            target = int(args[0]); clear_user(target)
            await update.message.reply_text(f"✅ Стёр память о `{target}`", parse_mode="Markdown")
        except ValueError:
            await update.message.reply_text("Неверный ID")
    else:
        clear_user(uid)
        await update.message.reply_text("✅ Своя память стёрта.")


# ════════════════════════════════════════════════════════════════
# CALLBACK КНОПОК
# ════════════════════════════════════════════════════════════════

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if not is_admin(query.from_user.id):
        await query.answer("Не твоё.", show_alert=True); return

    data = query.data
    back = [[InlineKeyboardButton("◀️ Назад", callback_data="adm_back")]]

    async def edit(text, kb=None):
        await query.edit_message_text(
            text, parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(kb) if kb else InlineKeyboardMarkup(back)
        )

    if data == "adm_back":
        await query.edit_message_text("🛠 *Админ панель Есет*", parse_mode="Markdown",
                                      reply_markup=admin_panel_keyboard())

    elif data == "adm_stats":   await edit(get_stats_text())
    elif data == "adm_conflicts": await edit(get_conflicts_text())
    elif data == "adm_heat":    await edit(get_heat_text())

    elif data == "adm_users":
        t = get_users_text()
        await edit(t[:3800] + ("\n..." if len(t) > 3800 else ""))

    elif data == "adm_disputes":
        t = get_disputes_text()
        await edit(t[:3800] + ("\n..." if len(t) > 3800 else ""))

    elif data == "adm_rels":
        t = get_rels_text()
        await edit(t[:3800] + ("\n..." if len(t) > 3800 else ""))

    elif data == "adm_dosie":
        t = get_dosie_text()
        await edit(t[:3800] + ("\n..." if len(t) > 3800 else ""))

    elif data == "adm_topics":
        t = get_topics_history_text()
        await edit(t[:3800] + ("\n..." if len(t) > 3800 else ""))

    elif data == "adm_mode":
        await query.edit_message_text(get_mode_text(), parse_mode="Markdown",
                                      reply_markup=mode_keyboard())

    elif data == "adm_toggle_active":
        new = "0" if get_setting("active") == "1" else "1"
        set_setting("active", new)
        await query.edit_message_text(get_mode_text(), parse_mode="Markdown",
                                      reply_markup=mode_keyboard())

    elif data == "adm_toggle_aggro":
        new = "0" if get_setting("aggro_mode") == "1" else "1"
        set_setting("aggro_mode", new)
        await query.edit_message_text(get_mode_text(), parse_mode="Markdown",
                                      reply_markup=mode_keyboard())

    elif data == "adm_toggle_sens":
        new = "normal" if get_setting("conflict_sens") == "high" else "high"
        set_setting("conflict_sens", new)
        await query.edit_message_text(get_mode_text(), parse_mode="Markdown",
                                      reply_markup=mode_keyboard())

    elif data == "adm_toggle_silent":
        BOT_MODE["silent"] = not BOT_MODE["silent"]
        await query.edit_message_text(get_mode_text(), parse_mode="Markdown",
                                      reply_markup=mode_keyboard())

    elif data == "adm_reset_aggro":
        reset_all_aggro()
        await query.answer("💣 Накал всех чатов сброшен", show_alert=True)

    elif data == "adm_forget_menu":
        await query.edit_message_text("🗑️ *Чью память стереть?*", parse_mode="Markdown",
                                      reply_markup=forget_menu_keyboard())

    elif data == "adm_forget_all":
        conn = _connect(); c = conn.cursor()
        c.execute("DELETE FROM messages"); conn.commit(); conn.close()
        await query.answer("🗑 Вся память стёрта", show_alert=True)
        await query.edit_message_text("🛠 *Админ панель Есет*", parse_mode="Markdown",
                                      reply_markup=admin_panel_keyboard())

    elif data.startswith("adm_forget_"):
        try:
            target = int(data.replace("adm_forget_", ""))
            name = get_display_name(target, "")
            clear_user(target)
            await edit(f"✅ Память *{name}* (`{target}`) стёрта.")
        except ValueError:
            await query.answer("Неверный ID", show_alert=True)


# ════════════════════════════════════════════════════════════════
# ОБЩАЯ ЛОГИКА ОБРАБОТКИ СООБЩЕНИЯ
# ════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════
# АНТИ-ФЛУД / МОДЕРАЦИЯ
# ════════════════════════════════════════════════════════════════
# Это отдельная защита от того, что решает brain.py: там — "стоит ли
# Рексу отвечать на флуд", здесь — "юзер реально спамит чат, надо
# приструнить", вне зависимости от того, ответил бы бот или нет.

_FLOOD_TRACKER: dict[int, dict[int, list[float]]] = {}
_FLOOD_WARNED:  dict[tuple, float] = {}   # (chat_id, user_id) -> ts последнего мута/варна

FLOOD_MAX_MSGS      = 6     # сообщений...
FLOOD_WINDOW_SEC     = 8.0   # ...за столько секунд считаем это спамом
FLOOD_MUTE_SEC       = 60    # на сколько мутим (если есть права)
FLOOD_ACTION_COOLDOWN = 30   # не мутим/не варним чаще раза в N секунд

_FLOOD_MUTE_PHRASES = [
    "Всё, ты в муте на минуту. Флудишь как больной.",
    "Заткнул тебя на минуту. Остынь и подумай о своей жизни.",
    "Многовато букв в единицу времени. Мут на 60 секунд.",
    "Захлебнулся своим потоком сознания — мут.",
]
_FLOOD_WARN_PHRASES = [
    "Полегче с сообщениями, а то замучу.",
    "Ты как из пулемёта строчишь. Притормози.",
    "Ещё немного и я тебя заткну физически.",
]

async def check_flood(update: Update, context, chat_id: int, user_id: int) -> bool:
    """
    Трекает частоту сообщений юзера в чате. Если превышен лимит —
    пробует замутить (если у бота есть права админа), иначе просто
    предупреждает текстом. Возвращает True, если сообщение нужно
    полностью проигнорировать (не гонять через LLM/анализ).
    """
    now = time.time()
    bucket = _FLOOD_TRACKER.setdefault(chat_id, {}).setdefault(user_id, [])
    bucket.append(now)
    cutoff = now - FLOOD_WINDOW_SEC
    while bucket and bucket[0] < cutoff:
        bucket.pop(0)

    if len(bucket) <= FLOOD_MAX_MSGS:
        return False

    key = (chat_id, user_id)
    last_action = _FLOOD_WARNED.get(key, 0.0)
    if now - last_action < FLOOD_ACTION_COOLDOWN:
        return True  # уже реагировали недавно — молча игнорим дальше

    _FLOOD_WARNED[key] = now
    muted = False
    try:
        await context.bot.restrict_chat_member(
            chat_id, user_id,
            permissions=ChatPermissions(can_send_messages=False),
            until_date=int(now + FLOOD_MUTE_SEC),
        )
        muted = True
    except Exception as e:
        logger.debug(f"[FLOOD] нет прав замутить {user_id} в {chat_id}: {e}")

    try:
        phrase = random.choice(_FLOOD_MUTE_PHRASES if muted else _FLOOD_WARN_PHRASES)
        await update.message.reply_text(phrase)
    except Exception:
        pass
    return True


async def process_message(update: Update, context, user, chat,
                          text: str, is_voice_input: bool = False):
    is_group = chat.type in ("group", "supergroup")

    # ── ЛС: чужие — молчим; админу — только панель ─────────────
    if not is_group:
        if not is_admin(user.id):
            return  # посторонним — тишина
        # Админу в ЛС: только обновлённая панель, без голосовых ответов
        get_or_create_user(user.id, user.username, user.first_name)
        u_cnt, m_cnt, mood, aggr, cf_cnt, d_cnt = get_stats()
        mode_str = "🔇 тихий" if BOT_MODE["silent"] else "🔊 активный"
        header = (
            f"🛠 *Панель Есет*\n\n"
            f"👤 {u_cnt} юзеров | 💬 {m_cnt} сообщ.\n"
            f"⚔️ {cf_cnt} конфликтов | 🔥 {d_cnt} споров\n"
            f"Режим: {mode_str}"
        )
        await update.message.reply_text(header, parse_mode="Markdown",
                                        reply_markup=admin_panel_keyboard())
        return

    # ── ГРУППЫ ──────────────────────────────────────────────────
    if get_setting("active") == "0":
        return

    _display = get_display_name(user.id, user.first_name, user.username)
    register_member(chat.id, user.id, _display, user.username)
    get_or_create_user(user.id, user.username, _display)
    increment_messages(user.id)

    # Анти-флуд: если юзер спамит — мутим/варним и не тратим LLM на это
    if await check_flood(update, context, chat.id, user.id):
        return

    # Самообучение: это следующее сообщение чата после ответа бота?
    # Если да — трактуем как реакцию на этот ответ (смех/агрессия/тишина и т.д.)
    _last_reply = _LAST_BOT_REPLY.pop(chat.id, None)
    if _last_reply and (time.time() - _last_reply["ts"]) < _REACTION_WINDOW:
        try:
            reaction = detect_reaction(text)
            record_reaction(DB_PATH, chat.id, user.id, reaction,
                            _last_reply["reply"], _last_reply["mode"])
        except Exception as e:
            logger.debug(f"[LEARN] не удалось зафиксировать реакцию: {e}")

    # Генерируем прозвище после 3-го сообщения (в фоне), с защитой от
    # повторного запуска пока первая попытка ещё выполняется
    _uinfo_pre = get_user_info(user.id)
    _msg_cnt_pre = (_uinfo_pre[6] or 0) if _uinfo_pre else 0
    if (_uinfo_pre and _msg_cnt_pre >= 3
            and not get_nickname(user.id)
            and user.id not in _NICKNAME_IN_PROGRESS):
        _NICKNAME_IN_PROGRESS.add(user.id)

        async def _gen_and_announce():
            try:
                nick = await maybe_generate_nickname(
                    user.id, _display,
                    personality=_uinfo_pre[11] if len(_uinfo_pre) > 11 else "",
                    notes=_uinfo_pre[7] if len(_uinfo_pre) > 7 else ""
                )
                if nick:
                    try:
                        await context.bot.send_message(
                            chat.id,
                            f"так, {_display}, с этого момента ты для меня — «{nick}». привыкай.",
                        )
                    except Exception:
                        pass
            finally:
                _NICKNAME_IN_PROGRESS.discard(user.id)

        asyncio.create_task(_gen_and_announce())

    sender_name = _display

    bot_info = await context.bot.get_me()
    bot_username = bot_info.username or ""
    bot_name_str = bot_info.first_name or ""
    bot_names = list(BOT_NAMES | {bot_username.lower(), bot_name_str.lower()})

    chat_context = get_recent_chat_messages(chat.id, limit=8)
    pre_state = get_chat_state(chat.id)
    known_names = [get_display_name(uid, fn, un) for uid, fn, un in get_members(chat.id)]

    # ── БЫСТРЫЙ АНАЛИЗ: запускаем в фоне пока готовим данные ──
    # ── БЫСТРАЯ ПРОВЕРКА: нужен ли LLM анализ? ──────────────────
    # Запускаем быстрый brain-check чтобы узнать — отвечаем вообще?
    # Если нет — не тратим LLM токены на анализ
    _pre_brain = get_brain()
    _pre_decision = _pre_brain.process(
        chat_id=chat.id, text=text, user_id=user.id,
        user_name=sender_name, update=update,
        bot_username=bot_username, bot_name=bot_name_str,
        chat_state=pre_state, is_group=is_group,
        settings=build_settings(
            get_setting("active"), get_setting("aggro_mode"),
            get_setting("conflict_sens"), BOT_MODE.get("silent", False),
        ), llm_analysis=None,
    )

    # Флуд и очевидное молчание — пропускаем дорогой LLM анализ
    _use_fast = (not _pre_decision.should_respond and
                 _pre_decision.reason in ("чистый_флуд","быстрый_поток_флуд",
                                          "тихий_режим","бот_выключен",
                                          "охлаждение","перебор_бот"))
    if _use_fast:
        analysis = {
            "sentiment": 0.0, "aggression": 0.2, "emotionality": 0.3,
            "topic": "флуд", "subtopic": "", "intent": "болтовня",
            "directed_at_bot": False, "directed_at_user": None,
            "is_conflict": False, "conflict_persons": [],
            "topic_continuity": False, "rex_interest": 0.1, "flood_score": 0.9,
        }
    else:
        # deep_analyze бьёт по сети (Groq) синхронно — уводим в отдельный
        # поток, иначе на время запроса замирают ВСЕ чаты бота разом.
        analysis = await asyncio.to_thread(
            deep_analyze,
            text=text, chat_id=chat.id, user_name=sender_name,
            chat_context=chat_context, known_names=known_names,
            bot_names=bot_names, chat_state=pre_state,
        )
    sentiment   = analysis.get("sentiment", 0.0)
    aggression  = analysis.get("aggression", 0.0)
    emotionality = analysis.get("emotionality", 0.5)
    topic       = analysis.get("topic", "другое")
    subtopic    = analysis.get("subtopic", "")
    intent      = analysis.get("intent", "")
    is_conflict = analysis.get("is_conflict", False)
    flood_score = analysis.get("flood_score", 0.0)
    conflict_persons = analysis.get("conflict_persons", [])

    update_user_profile(user.id, sentiment, aggression, topic)
    update_bot_attitude(user.id, sentiment, aggression,
                        analysis.get('directed_at_bot', False), text=text)
    auto_update_notes(user.id, sentiment, aggression, topic,
                      analysis.get('directed_at_bot', False), text=text)

    # Обиды — запоминаем сильные оскорбления бота
    directed_at_bot = analysis.get("directed_at_bot", False)
    if directed_at_bot and aggression > 0.6 and len(text) > 3:
        save_grudge(user.id, chat.id, text[:150], heat=aggression)

    label = f"{sender_name} (голосовое)" if is_voice_input else sender_name
    save_message(user.id, chat.id, "user", f"{label}: {text}",
                 sentiment=sentiment, aggression=aggression,
                 topic=topic, subtopic=subtopic, intent=intent, emotionality=emotionality)

    heat = aggression * 0.6 + emotionality * 0.4
    update_chat_state(chat.id, topic, heat, replied=False)

    # Долгосрочная память: записываем значимые события сразу
    if topic in ("угроза",) and aggression > 0.6:
        save_long_memory(chat.id, user.id, "threat",
                         f"{sender_name}: {text[:120]}", [user.id], heat=0.9)
    elif topic in ("похвала",) and sentiment > 0.5:
        save_long_memory(chat.id, user.id, "praise",
                         f"{sender_name} сказал хорошее: {text[:80]}", [user.id], heat=0.2)
    # Личные оскорбления бота — запоминаем как "обиды"
    if directed_at_bot and aggression > 0.55 and len(text) > 5:
        save_long_memory(chat.id, user.id, "обида",
                         f"{sender_name} сказал боту: «{text[:100]}»",
                         [user.id], heat=aggression)
    chat_state = get_chat_state(chat.id)

    # Фиксируем конфликты и отношения
    if is_conflict or topic in ("конфликт", "оскорбление", "угроза"):
        other_id = None
        if conflict_persons:
            for cp_name in conflict_persons:
                for m_uid, m_fn, m_un in get_members(chat.id):
                    if m_uid != user.id:
                        if cp_name.lower() in get_display_name(m_uid, m_fn, m_un).lower():
                            other_id = m_uid; break
                if other_id: break
        if not other_id:
            for msg in chat_context:
                if msg[0] and msg[0] != user.id:
                    other_id = msg[0]; break
        if other_id:
            register_conflict(chat.id, user.id, other_id, heat=aggression)
            update_relationship(chat.id, user.id, other_id, aggression,
                                "враги" if aggression > 0.6 else "напряжённые")
            save_topic_event(chat.id, topic, subtopic, heat, [user.id, other_id])
            # Долгосрочная память — записываем горячие события
            if aggression > 0.55:
                other_name = get_display_name(other_id, "")
                save_long_memory(
                    chat_id=chat.id, user_id=user.id,
                    event_type="conflict",
                    summary=f"{sender_name} vs {other_name}: {text[:100]}",
                    persons=[user.id, other_id], heat=aggression
                )
            if topic in ("конфликт", "спор", "оскорбление"):
                upsert_dispute(chat.id, user.id, other_id, topic, aggression)
    elif topic == "спор" and aggression < 0.5:
        for msg in chat_context:
            if msg[0] and msg[0] != user.id:
                update_relationship(chat.id, user.id, msg[0], 0.2, "спорщики")
                upsert_dispute(chat.id, user.id, msg[0], subtopic or topic, aggression)
                break

    # ── ТРЕД: контекст ответа ──────────────────────────────────
    reply_ctx  = extract_reply_context(update)
    thread_str = build_thread_context(reply_ctx, sender_name)
    if thread_str:
        logger.info(f"[THREAD] {thread_str}")

    # ── BRAIN: решение через отдельный модуль анализа ──────────
    user_info = get_user_info(user.id)
    brain = get_brain()
    decision = brain.process(
        chat_id      = chat.id,
        text         = text,
        user_id      = user.id,
        user_name    = sender_name,
        update       = update,
        bot_username = bot_username,
        bot_name     = bot_name_str,
        chat_state   = chat_state,
        is_group     = is_group,
        settings     = build_settings(
            get_setting("active"),
            get_setting("aggro_mode"),
            get_setting("conflict_sens"),
            BOT_MODE.get("silent", False),
        ),
        llm_analysis = analysis,
    )

    # Самокоманда и родительские провокации — всегда отвечаем, всегда attack
    if topic in ("самокоманда", "мамапапа"):
        decision = dataclasses.replace(decision, should_respond=True,
                                        mode="attack", reason="selfcmd")

    logger.info(f"[BRAIN {chat.id}] {sender_name} → "
                f"{'✅' if decision.should_respond else '❌'} "
                f"[{decision.mode}] {decision.reason} | {decision.context_summary}")

    if not decision.should_respond:
        return

    # Дедупликация — не отвечаем на одинаковые сообщения подряд
    if _is_duplicate(chat.id, text):
        logger.info(f"[DEDUP] пропускаем дубль: {text[:40]}")
        return

    # Реакция эмодзи (35% шанс, быстро, без LLM)
    await maybe_react(update, decision.mode, probability=0.35)

    members       = get_members(chat.id)
    top_conflicts = get_top_conflicts(chat.id)
    all_rels      = get_all_relationships(chat.id)
    disputes      = get_active_disputes(chat.id)
    users_sum     = get_all_users_summary(chat.id)
    notes         = user_info[6] if user_info else ""

    await update.message.chat.send_action("record_voice")
    try:
        # ask_rex тоже синхронный сетевой вызов — та же причина, в поток
        reply = await asyncio.to_thread(
            ask_rex,
            text, sender_name, user.id, chat.id, True,
            members, notes, user_info is None, analysis,
            chat_context, chat_state, user_info, top_conflicts,
            all_relationships=all_rels,
            active_disputes_list=disputes,
            users_summary=users_sum,
            respond_mode=decision.mode,
            context_summary=decision.context_summary,
            dialogue_summary=getattr(decision, 'dialogue_summary', ''),
            reply_context=reply_ctx,
        )
        if not reply:
            logger.info("  → [пропуск: LLM недоступен]")
            return
        logger.info(f"  → {reply[:70]}")
        brain.record_bot_reply(chat.id, reply_text=reply)
        analyzer_record_reply(chat.id, reply)
        save_message(user.id, chat.id, "assistant", reply)
        update_chat_state(chat.id, topic, heat, replied=True)

        _heat = float(pre_state[3]) if pre_state and len(pre_state) > 3 else 0.0
        await send_reply(update, reply, mode=decision.mode, heat=_heat)

        save_bot_reply_for_learning(DB_PATH, chat.id, user.id, reply, decision.mode, topic)
        _LAST_BOT_REPLY[chat.id] = {"reply": reply, "mode": decision.mode,
                                     "topic": topic, "ts": time.time()}
    except Exception:
        # Полная трассировка — только в лог. Пользователю — ничего лишнего
        # (не палим внутренние ошибки, пути, ключи API и т.д. в чат).
        logger.exception(f"[process_message] сбой при обработке сообщения chat={chat.id} user={user.id}")
        try:
            await update.message.reply_text(random.choice([
                "Сломался нахуй. Попробуй ещё раз.",
                "Что-то не так внутри. Дай секунду.",
                "Заглючил. Повтори.",
            ]))
        except Exception:
            pass


# ════════════════════════════════════════════════════════════════
# HANDLERS
# ════════════════════════════════════════════════════════════════

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text: return
    user = update.effective_user; chat = update.effective_chat
    if not user: return
    # Фиксируем время для автопровокации
    if chat and chat.type in ("group", "supergroup"):
        _LAST_CHAT_MSG[chat.id] = time.time()
    await process_message(update, context, user, chat, update.message.text)


async def handle_sticker(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Реагируем на стикеры — иногда ставим реакцию или отвечаем голосом."""
    if not update.message or not update.message.sticker: return
    user = update.effective_user; chat = update.effective_chat
    if not user or chat.type not in ("group", "supergroup"): return
    if get_setting("active") == "0": return

    sticker = update.message.sticker
    emoji = sticker.emoji or "?"

    # 25% шанс поставить реакцию на стикер
    if random.random() < 0.25:
        reaction = random.choice(_STICKER_REACTIONS)
        await set_reaction(update, reaction)

    # 12% шанс прокомментировать голосом (без LLM анализа — экономим)
    if random.random() < 0.12:
        _display = get_display_name(user.id, user.first_name, user.username)
        comments = [
            f"О, {_display} стикер кинул. Содержательно.",
            f"Стикер {emoji}. Это всё что ты можешь?",
            f"{_display}, ты слова знаешь вообще?",
            f"Стикерами разговариваешь — уже прогресс для тебя.",
            f"Понял тебя {_display}. Нет, не понял.",
        ]
        reply_text = random.choice(comments)
        vf = await make_voice(reply_text, mode="snark")
        if vf:
            from io import BytesIO
            await update.message.reply_voice(BytesIO(vf))


async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Реагируем на фото и видео."""
    if not update.message: return
    user = update.effective_user; chat = update.effective_chat
    if not user or chat.type not in ("group", "supergroup"): return
    if get_setting("active") == "0": return

    is_photo = bool(update.message.photo)
    is_video = bool(update.message.video or update.message.video_note)

    # 20% шанс поставить реакцию
    if random.random() < 0.20:
        reaction = random.choice(_MEDIA_REACTIONS)
        await set_reaction(update, reaction)

    # 8% шанс прокомментировать
    if random.random() < 0.08:
        _display = get_display_name(user.id, user.first_name, user.username)
        if is_photo:
            comments = [
                f"{_display} фотку кинул. Никого не впечатлил.",
                f"И зачем это? Серьёзно {_display}.",
                f"Фото в чат. Ладно.",
            ]
        else:
            comments = [
                f"Видосик от {_display}. Посмотрю когда-нибудь. Нет.",
                f"{_display} видео залил. Никто не смотрел.",
                f"Видео. Захватывает.",
            ]
        reply_text = random.choice(comments)
        vf = await make_voice(reply_text, mode="snark")
        if vf:
            from io import BytesIO
            await update.message.reply_voice(BytesIO(vf))

# ════════════════════════════════════════════════════════════════
# АВТОПРОВОКАЦИЯ
# ════════════════════════════════════════════════════════════════

# Время последнего сообщения в чате (chat_id -> timestamp)
_LAST_CHAT_MSG: dict[int, float] = {}
_AUTO_PROVOKE_SENT: dict[int, float] = {}  # chat_id -> timestamp последней провокации

# Интервал тишины после которого провоцируем (секунды)
_SILENCE_THRESHOLD = 3600  # 1 час
_PROVOKE_COOLDOWN  = 7200  # не чаще раза в 2 часа

async def send_weekly_rating(bot):
    """Отправляет рейтинг недели во все активные чаты."""
    conn = _connect(); c = conn.cursor()
    chats = c.execute("SELECT DISTINCT chat_id FROM group_members").fetchall()
    conn.close()

    for (chat_id,) in chats:
        try:
            top_active = c.execute("""SELECT u.user_id, u.first_name, COUNT(m.id) as cnt
                                       FROM messages m JOIN users u ON m.user_id=u.user_id
                                       WHERE m.chat_id=? AND m.role='user'
                                       AND m.timestamp > datetime('now','-7 days')
                                       GROUP BY u.user_id ORDER BY cnt DESC LIMIT 3""",
                                   (chat_id,)).fetchall()
            top_aggro = c.execute("""SELECT u.user_id, u.first_name, u.aggression_score
                                      FROM users u JOIN group_members g ON u.user_id=g.user_id
                                      WHERE g.chat_id=?
                                      ORDER BY u.aggression_score DESC LIMIT 1""",
                                  (chat_id,)).fetchone()

            if not top_active:
                continue

            lines = ["📊 *Итоги недели по версии Есета*\n"]
            lines.append("🏆 *Самые болтливые:*")
            medals = ["🥇","🥈","🥉"]
            for i, (uid, fn, cnt) in enumerate(top_active):
                name = get_display_name(uid, fn or "?")
                lines.append(f"  {medals[i]} {name} — {cnt} сообщений")

            if top_aggro:
                uid, fn, aggr = top_aggro
                name = get_display_name(uid, fn or "?")
                lines.append(f"\n😤 *Главный агрессор недели:* {name}")
                comments = [
                    f"Поздравляю {name} с титулом. Ты старался.",
                    f"{name} как всегда — агрессивнее всех. Привычка.",
                    f"Снова {name}. Ожидаемо.",
                ]
                lines.append(random.choice(comments))

            text = "\n".join(lines)
            vf = await make_voice(text, mode="snark")
            if vf:
                from io import BytesIO
                await bot.send_voice(chat_id=chat_id, voice=BytesIO(vf))
            else:
                await bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")
        except Exception as e:
            logger.warning(f"[WEEKLY] чат {chat_id}: {e}")
            continue


async def maybe_auto_provoke(bot=None):
    """
    Проверяем все чаты — если давно тишина, провоцируем.
    Запускается через asyncio.create_task каждые 10 минут.
    """
    if get_setting("active") == "0":
        return

    now = time.time()
    conn = _connect(); c = conn.cursor()
    # Берём чаты где бот состоит и знает участников
    chats = c.execute("SELECT DISTINCT chat_id FROM group_members").fetchall()
    conn.close()

    for (chat_id,) in chats:
        last_msg  = _LAST_CHAT_MSG.get(chat_id, 0)
        last_prov = _AUTO_PROVOKE_SENT.get(chat_id, 0)

        silence = now - last_msg
        since_last_prov = now - last_prov

        if (silence > _SILENCE_THRESHOLD and
                since_last_prov > _PROVOKE_COOLDOWN and
                last_msg > 0):  # чат был активен хотя бы раз

            # Выбираем жертву — самого активного участника
            members = get_members(chat_id)
            if not members:
                continue

            # Берём случайного из первых 5
            import random as _rnd
            target_uid, target_fname, target_uname = _rnd.choice(members[:5])
            target_name = get_display_name(target_uid, target_fname, target_uname)

            # Сначала пробуем уже изученную провокацию (экономим вызов LLM),
            # иначе — генерируем новую через Groq
            learned = get_ai_provocation(DB_PATH, style="general")
            if learned:
                provoke_text = f"{target_name}, {learned}"
                try:
                    vf = await make_voice(provoke_text, mode="snark")
                    if vf:
                        from io import BytesIO
                        if bot:
                            await bot.send_voice(chat_id=chat_id, voice=BytesIO(vf))
                    elif bot:
                        await bot.send_message(chat_id=chat_id, text=provoke_text)
                    _AUTO_PROVOKE_SENT[chat_id] = now
                    logger.info(f"[AUTO_PROVOKE] chat={chat_id} (learned) → {target_name}: {provoke_text[:50]}")
                except Exception as e:
                    logger.warning(f"[AUTO_PROVOKE] не удалось отправить learned: {e}")
                continue

            # Генерируем провокацию через LLM с коротким промптом
            recent = get_long_memory(chat_id, limit=3)
            mem_ctx = ""
            if recent:
                mem_ctx = " Последнее что было: " + "; ".join(
                    e["summary"][:60] for e in recent
                )

            provoke_prompt = (
                f"Ты Есет — грубый саркастичный бот. В чате давно тишина. "
                f"Напиши ОДНУ короткую провокационную фразу (1-2 предложения) "
                f"адресованную {target_name} чтобы расшевелить чат. "
                f"Будь конкретным, грубым, с матом если нужно.{mem_ctx} "
                f"Только фразу, без кавычек."
            )
            try:
                resp = _llm_call_with_retry(lambda: client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": provoke_prompt}],
                    max_tokens=80, temperature=1.1
                ))
                if resp is None:
                    continue
                provoke_text = resp.choices[0].message.content.strip()
                if not provoke_text:
                    continue

                vf = await make_voice(provoke_text, mode="snark")
                if vf:
                    try:
                        from io import BytesIO
                        if bot:
                            await bot.send_voice(chat_id=chat_id, voice=BytesIO(vf))
                    except Exception as e:
                        logger.warning(f"[AUTO_PROVOKE] не удалось отправить: {e}")
                        continue

                _AUTO_PROVOKE_SENT[chat_id] = now
                logger.info(f"[AUTO_PROVOKE] chat={chat_id} → {target_name}: {provoke_text[:50]}")

            except Exception as e:
                logger.warning(f"[AUTO_PROVOKE] ошибка: {e}")


async def maybe_random_monologue(bot, chat_id: int):
    """
    5% шанс при каждом вызове авто-провокации сказать что-то
    случайное от себя — безадресно, просто в чат.
    """
    if random.random() > 0.05:
        return

    monologues = [
        "Сижу, слушаю вас. Ничего умного не слышу.",
        "Интересно, хоть кто-то из вас думает перед тем как писать.",
        "Тишина в чате — лучшее что здесь бывает.",
        "Проверял. Вы все одинаково бесполезны.",
        "Не понимаю зачем я здесь. Хотя вы тоже не понимаете.",
        "Иногда перечитываю историю чата. Зря.",
        "Кто-нибудь, скажите что-нибудь умное. Хоть раз.",
        "Вы как коллектив — феномен тупости.",
        "Молчите? Ладно. Мне тоже не о чём с вами говорить.",
        "Час дня, никого нет. Логично — вам нечего сказать.",
    ]
    text = random.choice(monologues)
    vf = await make_voice(text, mode="snark")
    if vf:
        from io import BytesIO
        try:
            await bot.send_voice(chat_id=chat_id, voice=BytesIO(vf))
            logger.info(f"[MONOLOGUE] chat={chat_id}: {text[:40]}")
        except Exception as e:
            logger.debug(f"[MONOLOGUE] ошибка: {e}")


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.voice: return
    user = update.effective_user; chat = update.effective_chat
    if not user: return

    is_group = chat.type in ("group", "supergroup")

    # В ЛС: чужие молчим, админу — только панель (без ответа на голос)
    if not is_group:
        if not is_admin(user.id):
            return
        # Админу в ЛС — показываем панель, голосовые не обрабатываем
        u_cnt, m_cnt, mood, aggr, cf_cnt, d_cnt = get_stats()
        mode_str = "🔇 тихий" if BOT_MODE["silent"] else "🔊 активный"
        await update.message.reply_text(
            f"🛠 *Панель Есет*\n\n"
            f"👤 {u_cnt} юзеров | 💬 {m_cnt} сообщ.\n"
            f"⚔️ {cf_cnt} конфликтов | 🔥 {d_cnt} споров\n"
            f"Режим: {mode_str}",
            parse_mode="Markdown",
            reply_markup=admin_panel_keyboard()
        )
        return

    # ── ГРУППА: транскрибируем голосовое, дальше — общий пайплайн ──
    # (раньше здесь была отдельная копия всей логики process_message —
    # с багом: ссылалась на sender_name/text/directed_at_bot, которых
    # в этой функции не существовало, так что любое голосовое падало
    # с NameError сразу после анализа. Теперь голосовые идут через
    # тот же process_message, что и текст: одинаковая обработка флуда,
    # конфликтов, памяти, дедупликации и отката на текст при ошибке.)
    await update.message.chat.send_action("record_voice")
    try:
        from io import BytesIO
        vf_tg = await context.bot.get_file(update.message.voice.file_id)
        ogg_buf = BytesIO()
        await vf_tg.download_to_memory(ogg_buf)
        ogg_buf.seek(0)
        ogg_buf.name = "voice.ogg"  # Whisper требует имя файла
        transcription = await asyncio.to_thread(
            client.audio.transcriptions.create,
            model="whisper-large-v3", file=ogg_buf, language="ru")
        user_text = transcription.text.strip()
    except Exception as e:
        user_text = ""
        logger.error(f"STT: {e}")

    if not user_text:
        return  # не удалось транскрибировать — молчим

    if chat.type in ("group", "supergroup"):
        _LAST_CHAT_MSG[chat.id] = time.time()

    await process_message(update, context, user, chat, user_text, is_voice_input=True)

async def handle_new_member(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.new_chat_members: return
    chat = update.effective_chat
    for member in update.message.new_chat_members:
        register_member(chat.id, member.id, member.first_name, member.username)
        if member.id == context.bot.id:
            text = "Ну всё, я здесь. Зовут меня Есет и мне уже нахуй всё надоело. Кто первый получит?"
        else:
            name = get_display_name(member.id, member.first_name, member.username)
            text = random.choice([
                f"О, ещё один мудак припёрся — {name}. Добро пожаловать в ад.",
                f"{name} зашёл. Нам что, радоваться теперь? Нахуй надо.",
                f"Ну вот и {name}. Ещё один конченый в нашей компании.",
                f"Кто этот {name}? Да похуй, всё равно ничего умного не скажет.",
            ])
        vf = await make_voice(text, mode="neutral")
        if vf:
            from io import BytesIO
            await update.message.reply_voice(BytesIO(vf))
        else:
            await update.message.reply_text(text)


# ════════════════════════════════════════════════════════════════
# ЗАПУСК
# ════════════════════════════════════════════════════════════════

async def error_handler(update, context):
    """Глобальный обработчик ошибок — логируем и не падаем."""
    err = context.error
    msg = str(err).lower()
    if "429" in msg or "rate_limit" in msg:
        logger.warning(f"[ERROR] Rate limit Groq: {err}")
    elif "flood" in msg or "retry" in msg:
        logger.warning(f"[ERROR] Telegram flood: {err}")
    elif "network" in msg or "timeout" in msg or "connect" in msg:
        logger.warning(f"[ERROR] Сеть: {err}")
    else:
        logger.error(f"[ERROR] Необработанная ошибка: {err}", exc_info=context.error)
    # Не пробрасываем — бот продолжает работу


if __name__ == "__main__":
    init_db()
    check_groq_connection()  # покажет причину, если LLM недоступен
    init_analyzer(client, db_path=DB_PATH)  # глубокий анализ + персистентная память
    print("Есет v5 запущен 🐕")
    app = ApplicationBuilder().token(TG_TOKEN).build()

    # Автопровокация через asyncio — не требует job-queue зависимости
    async def weekly_rating_loop(app):
        """Каждое воскресенье в 18:00 объявляем рейтинг недели."""
        import datetime
        while True:
            await asyncio.sleep(3600)  # каждый час проверяем
            now = datetime.datetime.now()
            if now.weekday() == 6 and now.hour == 18:  # воскресенье 18:00
                try:
                    await send_weekly_rating(app.bot)
                except Exception as e:
                    logger.warning(f"[WEEKLY] ошибка: {e}")

    async def auto_provoke_loop(app):
        while True:
            await asyncio.sleep(600)  # каждые 10 минут
            try:
                await maybe_auto_provoke(bot=app.bot)
                # Случайный монолог — для всех активных чатов
                conn2 = _connect(); c2 = conn2.cursor()
                chats2 = c2.execute("SELECT DISTINCT chat_id FROM group_members").fetchall()
                conn2.close()
                for (cid,) in chats2:
                    if _LAST_CHAT_MSG.get(cid, 0) > 0:
                        await maybe_random_monologue(app.bot, cid)
            except Exception as e:
                logger.warning(f"[AUTO_PROVOKE] loop error: {e}")

    async def memory_maintenance_loop(app):
        """
        Раз в час остужаем старые конфликты/отношения и закрываем
        споры, по которым давно не было сообщений — иначе память
        только растёт и Рекс вечно припоминает склоки месячной давности.
        """
        while True:
            await asyncio.sleep(3600)
            try:
                resolve_stale_disputes()
                decay_relationships_and_conflicts()
                logger.info("[MEMORY] остывание конфликтов/отношений выполнено")
            except Exception as e:
                logger.warning(f"[MEMORY] ошибка обслуживания памяти: {e}")

    async def on_startup(app):
        asyncio.create_task(auto_provoke_loop(app))
        asyncio.create_task(weekly_rating_loop(app))
        asyncio.create_task(memory_maintenance_loop(app))
        asyncio.create_task(learner_loop(DB_PATH, client, "llama-3.3-70b-versatile"))

    app.post_init = on_startup
    app.add_error_handler(error_handler)
    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("admin",  cmd_admin))
    app.add_handler(CommandHandler("forget", cmd_forget))
    app.add_handler(CommandHandler("stats",  cmd_stats))
    app.add_handler(CommandHandler("me",     cmd_me))
    app.add_handler(CommandHandler("nick",   cmd_nick))
    app.add_handler(CallbackQueryHandler(handle_callback, pattern="^adm_"))
    app.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, handle_new_member))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.Sticker.ALL, handle_sticker))
    app.add_handler(MessageHandler(filters.PHOTO | filters.VIDEO, handle_media))
    app.run_polling(allowed_updates=Update.ALL_TYPES)
