"""
learner.py — Система самообучения бота Есет

Три компонента:
1. learn_from_reaction()  — фиксирует реакцию чата на ответ бота, обновляет стиль
2. generate_new_provocations() — Groq придумывает новые провокации на основе победных
3. get_learned_phrase()   — достаёт лучшую изученную фразу для контекста
4. analyze_effectiveness() — раз в час смотрит что сработало, обновляет веса
"""

import sqlite3
import logging
import random
import asyncio
import time
from datetime import datetime

logger = logging.getLogger(__name__)

# ── Подключение к БД ─────────────────────────────────────────────

def _db(db_path: str):
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.execute("PRAGMA busy_timeout=10000")
    conn.row_factory = sqlite3.Row
    return conn


# ════════════════════════════════════════════════════════════════
# 1. ФИКСАЦИЯ РЕАКЦИИ
# ════════════════════════════════════════════════════════════════

def save_bot_reply_for_learning(db_path: str, chat_id: int, user_id: int,
                                 reply_text: str, mode: str, topic: str):
    """Сохраняем ответ бота для последующей оценки."""
    conn = _db(db_path)
    c = conn.cursor()
    context = f"{mode}:{topic}"
    now = datetime.now().isoformat()
    c.execute("""
        INSERT OR IGNORE INTO learned_phrases(phrase, context, score, uses, wins, created_at, last_used)
        VALUES (?, ?, 1.0, 1, 0, ?, ?)
    """, (reply_text[:400], context, now, now))
    # Если уже есть — увеличиваем uses
    c.execute("""
        UPDATE learned_phrases SET uses = uses + 1, last_used = ?
        WHERE phrase = ? AND context = ?
    """, (now, reply_text[:400], context))
    conn.commit()
    conn.close()


def record_reaction(db_path: str, chat_id: int, user_id: int,
                    reaction_type: str, last_bot_reply: str, mode: str):
    """
    Фиксируем реакцию на последний ответ бота.
    reaction_type: 'laugh' | 'aggression' | 'silence' | 'reply' | 'emoji'
    """
    # Оцениваем качество реакции
    score_delta = {
        "laugh":      +0.4,   # смех/лол/хаха — отлично
        "emoji":      +0.2,   # эмодзи реакция — хорошо
        "aggression": +0.3,   # злой ответ — тоже зашло (спровоцировали)
        "reply":      +0.1,   # просто ответили — нейтрально
        "silence":    -0.1,   # проигнорили — плохо
    }.get(reaction_type, 0.0)

    conn = _db(db_path)
    c = conn.cursor()
    now = datetime.now().isoformat()

    # Обновляем фразу
    if last_bot_reply and score_delta != 0:
        wins_delta = 1 if score_delta > 0 else 0
        c.execute("""
            UPDATE learned_phrases
            SET score = MIN(5.0, MAX(0.1, score + ?)),
                wins = wins + ?
            WHERE phrase = ?
        """, (score_delta, wins_delta, last_bot_reply[:400]))

    # Обновляем стиль пользователя
    humor   = 1 if reaction_type == "laugh" else 0
    mat_hit = 1 if reaction_type in ("aggression", "emoji") else 0
    c.execute("""
        INSERT INTO user_style(user_id, humor_works, mat_works, reactions, updated_at)
        VALUES (?, ?, ?, 1, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            humor_works  = humor_works + excluded.humor_works,
            mat_works    = mat_works   + excluded.mat_works,
            reactions    = reactions   + 1,
            updated_at   = excluded.updated_at
    """, (user_id, humor, mat_hit, now))

    conn.commit()
    conn.close()
    logger.info(f"[LEARN] реакция {reaction_type} от user {user_id}, delta={score_delta:+.1f}")


def detect_reaction(text: str) -> str:
    """Rule-based определение типа реакции по тексту."""
    t = text.lower()
    laugh_marks = ["хах", "лол", "кек", "ахах", "ору", "😂", "🤣", "💀", "хаха", "лмао", "умер"]
    if any(m in t for m in laugh_marks):
        return "laugh"
    aggr_marks = ["иди нахуй", "пошёл", "заткнись", "мудак", "дебил", "тупой", "хуй", "пизд"]
    if any(m in t for m in aggr_marks):
        return "aggression"
    return "reply"


# ════════════════════════════════════════════════════════════════
# 2. ПОЛУЧЕНИЕ ЛУЧШИХ ИЗУЧЕННЫХ ФРАЗ
# ════════════════════════════════════════════════════════════════

def get_learned_phrase(db_path: str, context: str, user_id: int = None) -> str | None:
    """
    Возвращает лучшую изученную фразу для данного контекста.
    10% шанс что возьмёт случайную (exploration).
    """
    conn = _db(db_path)
    c = conn.cursor()

    # Exploitation: топ фразы по score
    if random.random() > 0.10:
        c.execute("""
            SELECT phrase FROM learned_phrases
            WHERE context LIKE ? AND wins > 0
            ORDER BY score DESC, wins DESC
            LIMIT 10
        """, (f"%{context.split(':')[0]}%",))
    else:
        # Exploration: случайная фраза
        c.execute("""
            SELECT phrase FROM learned_phrases
            WHERE context LIKE ?
            ORDER BY RANDOM()
            LIMIT 5
        """, (f"%{context.split(':')[0]}%",))

    rows = c.fetchall()
    conn.close()
    if rows:
        return random.choice(rows)["phrase"]
    return None


def get_user_best_mode(db_path: str, user_id: int) -> str:
    """Возвращает режим который лучше всего работает с этим пользователем."""
    conn = _db(db_path)
    c = conn.cursor()
    c.execute("SELECT * FROM user_style WHERE user_id=?", (user_id,))
    row = c.fetchone()
    conn.close()

    if not row or row["reactions"] < 3:
        return "snark"  # дефолт пока мало данных

    # Определяем лучший режим по статистике
    humor   = row["humor_works"]
    mat     = row["mat_works"]
    absurd  = row["absurd_works"]

    if humor > mat and humor > absurd:
        return "snark"   # юмор работает — используем сарказм
    if absurd > mat:
        return "attack"  # абсурд цепляет — идём в атаку
    if mat > 2:
        return "attack"  # мат работает — атака
    return "snark"


def get_ai_provocation(db_path: str, style: str = "general") -> str | None:
    """Возвращает AI-сгенерированную провокацию."""
    conn = _db(db_path)
    c = conn.cursor()
    c.execute("""
        SELECT text FROM ai_provocations
        WHERE style = ?
        ORDER BY score DESC, RANDOM()
        LIMIT 10
    """, (style,))
    rows = c.fetchall()
    conn.close()
    if rows:
        chosen = random.choice(rows)
        # Увеличиваем uses
        conn2 = _db(db_path)
        conn2.execute("UPDATE ai_provocations SET uses = uses + 1 WHERE text=?", (chosen["text"],))
        conn2.commit(); conn2.close()
        return chosen["text"]
    return None


# ════════════════════════════════════════════════════════════════
# 3. ГЕНЕРАЦИЯ НОВЫХ ПРОВОКАЦИЙ ЧЕРЕЗ GROQ
# ════════════════════════════════════════════════════════════════

async def generate_new_provocations(db_path: str, groq_client, model: str):
    """
    Groq придумывает 5 новых провокаций на основе победных фраз.
    Запускается раз в день через asyncio.
    """
    logger.info("[LEARN] генерирую новые провокации...")

    conn = _db(db_path)
    c = conn.cursor()

    # Берём топ победных фраз как примеры
    c.execute("""
        SELECT phrase FROM learned_phrases
        WHERE wins > 0
        ORDER BY score DESC
        LIMIT 15
    """)
    best = [r["phrase"] for r in c.fetchall()]

    # Также берём уже сгенерированные (чтобы не повторяться)
    c.execute("SELECT text FROM ai_provocations ORDER BY score DESC LIMIT 20")
    existing = [r["text"] for r in c.fetchall()]
    conn.close()

    examples = "\n".join(f"• {p[:100]}" for p in best[:8]) if best else "• иди нахуй\n• твоя мать плачет"

    prompt = f"""Ты — агрессивный русскоязычный бот по имени Есет. Твой стиль: жёсткий сарказм, мат, провокации, разворот слов собеседника против него.

Вот твои ЛУЧШИЕ фразы которые хорошо зашли в чате:
{examples}

Придумай РОВНО 5 НОВЫХ оригинальных провокационных фраз в таком же стиле.
Требования:
- Каждая фраза на отдельной строке
- Мат обязателен
- Короткие (1-2 предложения)
- Не повторяй эти уже существующие: {', '.join(existing[:5])}
- Никаких пояснений, только 5 фраз

Фразы:"""

    try:
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(None, lambda: groq_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=1.1,
        ))
        raw = resp.choices[0].message.content.strip()

        # Парсим строки
        lines = [l.strip().lstrip("•-0123456789.) ").strip()
                 for l in raw.split("\n") if len(l.strip()) > 10]

        conn = _db(db_path)
        now = datetime.now().isoformat()
        saved = 0
        for line in lines[:5]:
            if line and line not in existing:
                conn.execute("""
                    INSERT OR IGNORE INTO ai_provocations(text, style, score, uses, created_at)
                    VALUES (?, 'general', 0.5, 0, ?)
                """, (line, now))
                saved += 1
        conn.commit(); conn.close()
        logger.info(f"[LEARN] сгенерировано и сохранено {saved} новых провокаций")

    except Exception as e:
        logger.error(f"[LEARN] ошибка генерации: {e}")


# ════════════════════════════════════════════════════════════════
# 4. АНАЛИЗ ЭФФЕКТИВНОСТИ (раз в час)
# ════════════════════════════════════════════════════════════════

def analyze_effectiveness(db_path: str) -> dict:
    """
    Анализирует что работает а что нет.
    Возвращает статистику для логов.
    """
    conn = _db(db_path)
    c = conn.cursor()

    # Топ фраз
    c.execute("""
        SELECT phrase, score, wins, uses
        FROM learned_phrases
        WHERE uses > 0
        ORDER BY score DESC
        LIMIT 5
    """)
    top_phrases = c.fetchall()

    # Пользователи у которых мы уже поняли стиль
    c.execute("""
        SELECT user_id, best_mode, reactions, humor_works, mat_works
        FROM user_style
        WHERE reactions >= 3
        ORDER BY reactions DESC
        LIMIT 5
    """)
    user_styles = c.fetchall()

    # Удаляем фразы с очень низким score (мусор)
    c.execute("DELETE FROM learned_phrases WHERE uses > 5 AND wins = 0 AND score < 0.3")
    deleted = conn.total_changes

    # Обновляем best_mode у пользователей
    c.execute("SELECT user_id, humor_works, mat_works, absurd_works FROM user_style")
    for row in c.fetchall():
        uid = row["user_id"]
        if row["humor_works"] > row["mat_works"]:
            mode = "snark"
        elif row["mat_works"] > 2:
            mode = "attack"
        else:
            mode = "snark"
        c.execute("UPDATE user_style SET best_mode=? WHERE user_id=?", (mode, uid))

    conn.commit()

    stats = {
        "top_phrases": [(r["phrase"][:50], r["score"], r["wins"]) for r in top_phrases],
        "user_styles": [(r["user_id"], r["best_mode"], r["reactions"]) for r in user_styles],
        "deleted_ineffective": deleted,
    }
    conn.close()
    logger.info(f"[LEARN] анализ: топ={len(top_phrases)} фраз, стилей={len(user_styles)}, удалено={deleted}")
    return stats


# ════════════════════════════════════════════════════════════════
# 5. ФОНОВЫЕ ЗАДАЧИ
# ════════════════════════════════════════════════════════════════

_LAST_GENERATE = 0.0
_LAST_ANALYZE  = 0.0
_GENERATE_INTERVAL = 86400  # раз в сутки
_ANALYZE_INTERVAL  = 3600   # раз в час


async def learner_loop(db_path: str, groq_client, model: str):
    """Фоновый цикл самообучения. Запускается через asyncio."""
    global _LAST_GENERATE, _LAST_ANALYZE
    logger.info("[LEARN] фоновый цикл запущен")

    while True:
        await asyncio.sleep(300)  # проверка каждые 5 минут

        now = time.time()

        # Анализ раз в час
        if now - _LAST_ANALYZE > _ANALYZE_INTERVAL:
            try:
                analyze_effectiveness(db_path)
                _LAST_ANALYZE = now
            except Exception as e:
                logger.error(f"[LEARN] ошибка анализа: {e}")

        # Генерация раз в сутки
        if now - _LAST_GENERATE > _GENERATE_INTERVAL:
            try:
                await generate_new_provocations(db_path, groq_client, model)
                _LAST_GENERATE = now
            except Exception as e:
                logger.error(f"[LEARN] ошибка генерации: {e}")
