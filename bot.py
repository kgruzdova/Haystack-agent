"""
Telegram-бот-помощник с долговременной памятью на основе Pinecone.

Архитектура памяти:
- Краткосрочная (буфер): последние N сообщений диалога хранятся в памяти процесса.
- Долговременная (векторная): факты о пользователе сохраняются в Pinecone через PineconeManager.
  Перед записью выполняется дедупликация по косинусному сходству.

Все пользователи хранятся в едином индексе (namespace="").
Изоляция достигается через метаданные user_id и фильтры при записи и чтении.
"""

import os
import uuid
import logging
from collections import deque
from typing import Optional

import telebot
from dotenv import load_dotenv
from openai import OpenAI

from pinecone_manager import PineconeManager

load_dotenv()

# ---------------------------------------------------------------------------
# Настройки бота
# ---------------------------------------------------------------------------

# Количество последних сообщений, передаваемых в контекст LLM (краткосрочная память)
SHORT_TERM_MEMORY_SIZE: int = 20
# Сколько фактов из долговременной памяти добавлять в промпт
LONG_TERM_TOP_K: int = 5
# Модель LLM для диалога
CHAT_MODEL: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# ---------------------------------------------------------------------------
# Инициализация
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
if not bot_token:
    raise RuntimeError("Не задан TELEGRAM_BOT_TOKEN в .env")

bot = telebot.TeleBot(bot_token, parse_mode=None)

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    **({"base_url": os.getenv("OPENAI_BASE_URL")} if os.getenv("OPENAI_BASE_URL") else {}),
)

# Краткосрочная память: user_id → deque сообщений [{role, content}, ...]
short_term: dict[int, deque] = {}

# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Ты дружелюбный и внимательный ассистент. 
Ты помнишь всё, что пользователь рассказывал о себе раньше, и используешь эти знания в диалоге.
Обращайся к пользователю по имени, если оно тебе известно.
Отвечай на том языке, на котором пишет пользователь."""


def get_memory_manager() -> PineconeManager:
    """Создаёт экземпляр PineconeManager с единым индексом для всех пользователей."""
    return PineconeManager(namespace="")


def load_long_term_memory(user_id: int, query: str) -> str:
    """Ищет релевантные факты о пользователе в долговременной памяти."""
    try:
        manager = get_memory_manager()
        results = manager.query_by_text(
            text=query,
            top_k=LONG_TERM_TOP_K,
            include_metadata=True,
            filter={"user_id": {"$eq": str(user_id)}},
        )
        if not results:
            return ""
        facts = []
        for r in results:
            text = r.get("metadata", {}).get("text", "")
            if text:
                facts.append(f"- {text}")
        return "\n".join(facts)
    except Exception as exc:
        logger.warning("Ошибка чтения долговременной памяти для user %d: %s", user_id, exc)
        return ""


def save_to_long_term_memory(user_id: int, text: str, role: str = "user") -> None:
    """Сохраняет фрагмент текста в долговременную память пользователя."""
    try:
        manager = get_memory_manager()
        record_id = f"{user_id}_{uuid.uuid4().hex}"
        logger.info(
            "Pinecone | попытка записи | user=%d | role=%s | text=%.80r",
            user_id, role, text,
        )
        result = manager.upsert_document(
            id=record_id,
            text=text,
            metadata={
                "user_id": str(user_id),
                "role": role,
                "text": text,
            },
            # Фильтр гарантирует сравнение только с записями этого пользователя
            filter={"user_id": {"$eq": str(user_id)}},
        )
        upserted = result.get("upserted_count", 0) if isinstance(result, dict) else getattr(result, "upserted_count", "?")
        if upserted == 0:
            logger.info(
                "Pinecone | ПРОПУЩЕНО (дубликат/сходство >= порога) | user=%d | text=%.80r",
                user_id, text,
            )
        else:
            logger.info(
                "Pinecone | ЗАПИСАНО | id=%s | user=%d | text=%.80r",
                record_id, user_id, text,
            )
    except Exception as exc:
        logger.warning("Pinecone | ОШИБКА записи | user=%d: %s", user_id, exc)


def build_messages(user_id: int, user_text: str) -> list[dict]:
    """Формирует список сообщений для запроса к LLM."""
    long_term_facts = load_long_term_memory(user_id, user_text)

    system_content = SYSTEM_PROMPT
    if long_term_facts:
        system_content += f"\n\nИзвестные факты о пользователе из долговременной памяти:\n{long_term_facts}"

    messages = [{"role": "system", "content": system_content}]

    history = short_term.get(user_id, deque())
    messages.extend(list(history))

    messages.append({"role": "user", "content": user_text})
    return messages


def chat_completion(messages: list[dict]) -> str:
    """Выполняет запрос к LLM и возвращает текст ответа."""
    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
    )
    return response.choices[0].message.content or ""


def update_short_term(user_id: int, user_text: str, assistant_text: str) -> None:
    """Обновляет краткосрочный буфер диалога."""
    if user_id not in short_term:
        short_term[user_id] = deque(maxlen=SHORT_TERM_MEMORY_SIZE)
    buf = short_term[user_id]
    buf.append({"role": "user", "content": user_text})
    buf.append({"role": "assistant", "content": assistant_text})


# ---------------------------------------------------------------------------
# Обработчики команд
# ---------------------------------------------------------------------------

@bot.message_handler(commands=["start"])
def handle_start(message: telebot.types.Message) -> None:
    user = message.from_user
    name = user.first_name or "друг"
    bot.send_message(
        message.chat.id,
        f"Привет, {name}! Я твой персональный ассистент с памятью.\n"
        "Я запоминаю всё, что ты мне рассказываешь, и буду учитывать это в наших следующих беседах.\n\n"
        "Просто напиши мне что-нибудь 🙂\n\n"
        "Команды:\n"
        "/start — приветствие\n"
        "/clear — очистить краткосрочную память (текущий диалог)\n"
        "/forget — удалить всю долговременную память обо мне",
    )


@bot.message_handler(commands=["clear"])
def handle_clear(message: telebot.types.Message) -> None:
    user_id = message.from_user.id
    short_term.pop(user_id, None)
    bot.send_message(message.chat.id, "Краткосрочная память очищена. Начинаем диалог заново!")


@bot.message_handler(commands=["forget"])
def handle_forget(message: telebot.types.Message) -> None:
    user_id = message.from_user.id
    short_term.pop(user_id, None)
    try:
        manager = get_memory_manager()
        manager.delete(filter={"user_id": {"$eq": str(user_id)}})
        bot.send_message(
            message.chat.id,
            "Вся долговременная память о тебе удалена. Я тебя больше не помню 🙁\nМожем начать с чистого листа.",
        )
    except Exception as exc:
        logger.error("Ошибка удаления памяти для user %d: %s", user_id, exc)
        bot.send_message(message.chat.id, "Не удалось удалить долговременную память. Попробуй позже.")


# ---------------------------------------------------------------------------
# Основной обработчик сообщений
# ---------------------------------------------------------------------------

@bot.message_handler(func=lambda m: m.content_type == "text")
def handle_message(message: telebot.types.Message) -> None:
    user_id = message.from_user.id
    user_text = message.text.strip()

    if not user_text:
        return

    bot.send_chat_action(message.chat.id, "typing")

    try:
        messages = build_messages(user_id, user_text)
        reply = chat_completion(messages)
    except Exception as exc:
        logger.error("Ошибка LLM для user %d: %s", user_id, exc)
        bot.send_message(message.chat.id, "Произошла ошибка при обращении к AI. Попробуй ещё раз.")
        return

    bot.send_message(message.chat.id, reply)

    update_short_term(user_id, user_text, reply)

    # Сохраняем сообщение пользователя в долговременную память
    save_to_long_term_memory(user_id, user_text, role="user")


# ---------------------------------------------------------------------------
# Запуск
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Бот запущен. Ожидаю сообщений...")
    bot.infinity_polling(timeout=30, long_polling_timeout=20)
