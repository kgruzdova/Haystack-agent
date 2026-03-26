"""
Telegram-бот: персональный помощник на базе Haystack Agent + Pinecone (haystack-integrations).

Долговременный контекст: в Pinecone сохраняется только текст сообщений пользователя (не ответы ассистента).
Краткосрочный диалог: deque с ChatMessage для последних реплик.

Инструменты:
- catFactTool — факт о кошках (catfact.ninja), подробные логи;
- dogImageTool — URL случайного фото (dog.ceo);
- dogFactTool — факт о собаках (kinduff API);
- docImageAnalyzerTool — фото с dog.ceo, затем vision (OpenAI), в Telegram: фото + подпись.

Логи: loguru, цветной формат (как в консоли: время | уровень | файл:функция:строка — сообщение).

Запуск из корня проекта:
  python hay/hay-telegram-bot.py
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import uuid
from collections import deque
from copy import copy
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

import requests
import telebot
from dotenv import load_dotenv
from haystack import Document
from haystack.components.agents import Agent
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, TextContent, ToolCallResult
from haystack.document_stores.types import DuplicatePolicy
from haystack.tools import create_tool_from_function
from haystack.utils import Secret
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack_integrations.document_stores.pinecone.document_store import (
    METADATA_SUPPORTED_TYPES,
    PineconeDocumentStore,
)
from loguru import logger
from openai import OpenAI

load_dotenv()


def _setup_loguru() -> None:
    """Цветной вывод в стиле: время | INFO | модуль:функция:строка — сообщение."""
    logger.remove()
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        level="INFO",
        colorize=True,
    )


class _InterceptLoggingHandler(logging.Handler):
    """Переводит стандартный logging (Haystack и др.) в loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            logger.opt(depth=6, exception=record.exc_info).log(record.levelno, record.getMessage())
        except Exception:
            self.handleError(record)


def _setup_stdlib_logging_bridge() -> None:
    logging.captureWarnings(True)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(_InterceptLoggingHandler())
    root.setLevel(logging.INFO)


_setup_loguru()
_setup_stdlib_logging_bridge()

# --- Настройки ---
SHORT_TERM_MAX_MESSAGES: int = 24  # ChatMessage (user+assistant парами)
MEMORY_TOP_K: int = 8
CHAT_MODEL: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
PINECONE_INDEX: str = os.getenv("PINECONE_INDEX_NAME", "default")
# Отдельный namespace, чтобы не смешивать с «сырыми» векторами из pinecone_manager / bot.py
PINECONE_NAMESPACE: str = os.getenv("HAYSTACK_PINECONE_NAMESPACE", "haystack_memory")
PINECONE_DIMENSION: int = int(os.getenv("PINECONE_DIMENSION", "1536"))
TELEGRAM_MAX_LEN: int = 4096
TELEGRAM_CAPTION_MAX: int = 1024
DOG_CEO_RANDOM_IMAGE = "https://dog.ceo/api/breeds/image/random"
DOG_FACT_API = "https://dog-api.kinduff.com/api/facts?number=1"
CAT_FACT_API = "https://catfact.ninja/fact"
TOOL_NAME_DOC_IMAGE_ANALYZER = "docImageAnalyzerTool"
TOOL_NAME_CAT_FACT = "catFactTool"
TOOL_NAME_WEATHER = "weatherTool"

# OpenWeather (https://home.openweathermap.org) requires an API key (free tier is available).
OPENWEATHER_API_BASE_URL = os.getenv("OPENWEATHER_API_BASE_URL", "https://api.openweathermap.org/data/2.5")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
OPENWEATHER_CURRENT_BY_CITY = f"{OPENWEATHER_API_BASE_URL}/weather"

OPENAI_BASE = os.getenv("OPENAI_BASE_URL") or None


def user_message_mentions_cat_facts(text: str) -> bool:
    """Эвристика для расширенного логирования запросов про кошек и факты о них."""
    t = (text or "").lower()
    needles = (
        "кош",
        "кот",
        "котя",
        "коты",
        "коте",
        "котов",
        "кошк",
        "cat fact",
        "facts about cats",
        "факт о кош",
        "факт про кош",
        "факты о кош",
        "кошач",
    )
    return any(n in t for n in needles)


def log_tool_results_summary(messages: list[ChatMessage]) -> None:
    """Краткий лог по вызванным инструментам (в т.ч. catFactTool)."""
    for msg in messages:
        for tcr in msg.tool_call_results:
            name = tcr.origin.tool_name
            raw = _tool_call_result_as_str(tcr)
            logger.info(
                "handle_text: результат инструмента %s, длина вывода: %s",
                name,
                len(raw),
            )
            if name == TOOL_NAME_CAT_FACT:
                logger.info(
                    "handle_text: catFactTool — факт о кошках получен, превью: %s",
                    raw[:120] + ("..." if len(raw) > 120 else ""),
                )


def _metadata_value_ok(value: Any) -> bool:
    return isinstance(value, METADATA_SUPPORTED_TYPES) or (
        isinstance(value, list) and all(isinstance(i, str) for i in value)
    )


def _document_with_sanitized_meta(document: Document) -> Document:
    """
    Возвращает новый Document с отфильтрованным meta для Pinecone.
    Не мутирует исходный экземпляр (см. Haystack: avoid in-place mutation).
    https://docs.haystack.deepset.ai/docs/custom-components#requirements
    """
    if not document.meta:
        return document
    discarded_keys: list[str] = []
    new_meta: dict[str, Any] = {}
    for key, value in document.meta.items():
        if not _metadata_value_ok(value):
            discarded_keys.append(key)
        else:
            new_meta[key] = value
    if discarded_keys:
        logger.warning(
            "Document %s: отброшены поля meta неподдерживаемых типов %s (нужны str, int, bool, float, List[str]).",
            document.id,
            discarded_keys,
        )
    return replace(document, meta=new_meta)


class PineconeDocumentStoreSafe(PineconeDocumentStore):
    """
    Обход in-place присваивания ``document.meta`` в базовом PineconeDocumentStore
    при подготовке записи (иначе предупреждение Haystack о мутации Document).
    """

    def _convert_documents_to_pinecone_format(
        self, documents: list[Document]
    ) -> list[tuple[str, list[float], dict[str, Any]]]:
        logger.info(
            "run _convert_documents_to_pinecone_format: подготовка %s документов к записи в Pinecone",
            len(documents),
        )
        documents_for_pinecone: list[tuple[str, list[float], dict[str, Any]]] = []
        for document in documents:
            doc = _document_with_sanitized_meta(document) if document.meta else document
            embedding = copy(doc.embedding)
            if embedding is None:
                logger.warning(
                    "Document %s: нет embedding; подставляется dummy-вектор (может портить поиск).",
                    doc.id,
                )
                embedding = self._dummy_vector

            metadata = dict(doc.meta) if doc.meta else {}
            if doc.content is not None:
                metadata["content"] = doc.content
            if doc.blob is not None:
                logger.warning(
                    "Document %s: поле blob не сохраняется в Pinecone.",
                    doc.id,
                )
            if hasattr(doc, "sparse_embedding") and doc.sparse_embedding is not None:
                logger.warning(
                    "Document %s: sparse_embedding в Pinecone не сохраняется.",
                    doc.id,
                )

            documents_for_pinecone.append((doc.id, embedding, metadata))
        return documents_for_pinecone


def _openai_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        **({"base_url": OPENAI_BASE} if OPENAI_BASE else {}),
    )


def cat_fact_tool() -> str:
    """
    Fetch a random cat fact from the free catfact.ninja API.
    Use when the user asks for facts, trivia, or interesting information about cats.
    """
    logger.info("CatFactTool.run: Начало получения факта о кошках")
    logger.info("CatFactTool.run: Запрос к API {}", CAT_FACT_API)
    try:
        r = requests.get(CAT_FACT_API, timeout=20)
        r.raise_for_status()
        data = r.json() or {}
        fact = (data.get("fact") or "").strip()
    except Exception as exc:
        logger.exception("CatFactTool.run: ошибка API: {}", exc)
        return "Не удалось получить факт о кошках (ошибка сети или API)."
    if not fact:
        logger.warning("CatFactTool.run: пустой факт в ответе")
        return "Факт о кошках не пришёл."
    logger.info("CatFactTool.run: Факт успешно получен, длина: {}", len(fact))
    return f"Факт о кошках: {fact}"


def dog_image_tool() -> str:
    """
    Fetch a random dog image URL from the Dog CEO public API (no auth).
    Use when the user wants only a random dog picture URL.
    """
    logger.info("dogImageTool.run: запрос URL изображения ({})", DOG_CEO_RANDOM_IMAGE)
    r = requests.get(DOG_CEO_RANDOM_IMAGE, timeout=25)
    r.raise_for_status()
    image_url = (r.json() or {}).get("message", "").strip()
    if not image_url:
        logger.warning("dogImageTool.run: пустой URL в ответе API")
        return "Не удалось получить URL изображения собаки."
    logger.info("dogImageTool.run: получен URL длины={}", len(image_url))
    return f"Случайное изображение собаки: {image_url}"


def dog_fact_tool() -> str:
    """
    Fetch a random dog fact from the free Dog API (kinduff).
    Use when the user wants trivia or facts about dogs.
    """
    logger.info("DogFactTool.run: Начало получения факта о собаках")
    logger.info("DogFactTool.run: Запрос к API {}", DOG_FACT_API)
    try:
        r = requests.get(DOG_FACT_API, timeout=15)
        r.raise_for_status()
        data = r.json() or {}
        facts = data.get("facts") or []
        fact = (facts[0] if facts else "").strip()
    except Exception as exc:
        logger.exception("DogFactTool.run: ошибка API: {}", exc)
        return "Не удалось получить факт о собаках (ошибка сети или API)."
    if not fact:
        logger.warning("DogFactTool.run: пустой факт")
        return "Факт о собаках не пришёл."
    logger.info("DogFactTool.run: Факт успешно получен, длина: {}", len(fact))
    return f"Факт о собаках: {fact}"


def weather_openweather_tool(city_query: str) -> str:
    """
    Current weather by city name using OpenWeather API (free tier + API key required).

    Called as a tool by the agent. Returns a plain-text short weather summary in Russian.
    """
    place = (city_query or "").strip()
    if not place:
        logger.warning("{}.run: пустой запрос города", TOOL_NAME_WEATHER)
        return "Укажите название города для запроса погоды."

    logger.info("{}.run: начало запроса погоды, city={!r}", TOOL_NAME_WEATHER, place)

    if not OPENWEATHER_API_KEY:
        logger.error("{}.run: OPENWEATHER_API_KEY не задан", TOOL_NAME_WEATHER)
        return "Сервис погоды недоступен: не настроен `OPENWEATHER_API_KEY`."

    params = {
        "q": place,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
        "lang": "ru",
    }
    logger.info(
        "%s.run: запрос к API %s params(city=%s, units=metric, lang=ru)",
        TOOL_NAME_WEATHER,
        OPENWEATHER_CURRENT_BY_CITY,
        place,
    )

    try:
        r = requests.get(OPENWEATHER_CURRENT_BY_CITY, params=params, timeout=20)
        r.raise_for_status()
        data = r.json() or {}
    except Exception as exc:
        logger.exception("{}.run: ошибка OpenWeather API: {}", TOOL_NAME_WEATHER, exc)
        return "Не удалось получить погоду (ошибка сети/сервиса OpenWeather)."

    if data.get("cod") not in (200, "200"):
        msg = data.get("message") or "unknown error"
        logger.warning("{}.run: OpenWeather вернул cod={} message={}", TOOL_NAME_WEATHER, data.get("cod"), msg)
        return f"Не удалось получить погоду для «{place}»: {msg}."

    main = data.get("main") or {}
    weather_arr = data.get("weather") or []
    wind = data.get("wind") or {}

    temp = main.get("temp")
    feels = main.get("feels_like")
    humidity = main.get("humidity")

    wind_speed = wind.get("speed")
    wind_deg = wind.get("deg")

    weather_desc = ""
    if weather_arr:
        weather_desc = (weather_arr[0].get("description") or "").strip()

    city_name = data.get("name") or place
    country = (data.get("sys") or {}).get("country") or ""
    location_line = f"{city_name}" + (f" ({country})" if country else "")

    logger.info(
        "%s.run: данные получены: temp=%s feels=%s humidity=%s desc=%s wind=%s deg=%s",
        TOOL_NAME_WEATHER,
        temp,
        feels,
        humidity,
        weather_desc,
        wind_speed,
        wind_deg,
    )

    parts = [f"Погода в {location_line}:"]
    if weather_desc:
        parts.append(f"{weather_desc}.")
    if temp is not None:
        parts.append(f"Температура: {temp} °C.")
    if feels is not None:
        parts.append(f"Ощущается как: {feels} °C.")
    if humidity is not None:
        parts.append(f"Влажность: {humidity}%.")
    if wind_speed is not None:
        wind_str = f"{wind_speed} м/с"
        if wind_deg is not None:
            wind_str += f" (на {wind_deg}°)"
        parts.append(f"Ветер: {wind_str}.")

    parts.append("(Источник: OpenWeather, бесплатные данные тарифа.)")
    return " ".join(parts)


def doc_image_analyzer_tool() -> str:
    """
    Load a random dog image from Dog CEO, send that same URL to the vision/chat model,
    and return ONE JSON object for the app (image URL + caption text). The Telegram layer
    sends the photo with the model reply as caption.
    Use when the user wants breed identification, description, or analysis of a random dog photo.
    """
    logger.info("{}.run: шаг 1 — загрузка URL изображения", TOOL_NAME_DOC_IMAGE_ANALYZER)
    r = requests.get(DOG_CEO_RANDOM_IMAGE, timeout=25)
    r.raise_for_status()
    image_url = (r.json() or {}).get("message", "").strip()
    if not image_url:
        logger.warning("{}.run: пустой URL после API", TOOL_NAME_DOC_IMAGE_ANALYZER)
        return json.dumps(
            {"error": "no_image_url", "message": "Could not get a dog image URL."},
            ensure_ascii=False,
        )

    logger.info("{}.run: шаг 2 — URL получен, вызов vision (model={})", TOOL_NAME_DOC_IMAGE_ANALYZER, CHAT_MODEL)
    client = _openai_client()
    prompt = (
        "Look at the dog in this image. Name the most likely breed (or mix). "
        "Give 3–6 sentences: brief breed traits, geographic/historical origin, "
        "how the breed developed. Reply in the same language the Telegram user likely uses "
        "(if unclear, use Russian)."
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        max_tokens=600,
    )
    desc = (resp.choices[0].message.content or "").strip()
    logger.info(
        "%s.run: шаг 3 — ответ модели, длина подписи=%s; один JSON для Telegram",
        TOOL_NAME_DOC_IMAGE_ANALYZER,
        len(desc),
    )
    payload = {
        "_telegram": "send_photo",
        "photo_url": image_url,
        "caption": desc,
    }
    return json.dumps(payload, ensure_ascii=False)


def _tool_call_result_as_str(tcr: ToolCallResult) -> str:
    res = tcr.result
    if isinstance(res, str):
        return res
    if isinstance(res, list):
        parts: list[str] = []
        for part in res:
            if isinstance(part, TextContent):
                parts.append(part.text)
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return str(res)


def extract_doc_analyzer_photo_from_messages(messages: list[ChatMessage]) -> tuple[str | None, str | None]:
    """
    Ищет результат docImageAnalyzerTool (JSON с photo_url и caption) в цепочке сообщений агента.
    """
    for msg in messages:
        for tcr in msg.tool_call_results:
            if tcr.origin.tool_name != TOOL_NAME_DOC_IMAGE_ANALYZER:
                continue
            raw = _tool_call_result_as_str(tcr).strip()
            logger.info(
                "extract_doc_analyzer: найден результат инструмента %s, len=%s",
                TOOL_NAME_DOC_IMAGE_ANALYZER,
                len(raw),
            )
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("extract_doc_analyzer: не JSON, пробуем извлечь из markdown")
                m = re.search(r"\{[\s\S]*\"_telegram\"[\s\S]*\}", raw)
                if not m:
                    continue
                try:
                    obj = json.loads(m.group(0))
                except json.JSONDecodeError:
                    continue
            if obj.get("_telegram") != "send_photo":
                continue
            url, cap = obj.get("photo_url"), obj.get("caption")
            if url and cap is not None:
                return str(url), str(cap)
    return None, None


def build_document_store() -> PineconeDocumentStoreSafe:
    return PineconeDocumentStoreSafe(
        api_key=Secret.from_env_var("PINECONE_API_KEY"),
        index=PINECONE_INDEX,
        namespace=PINECONE_NAMESPACE,
        dimension=PINECONE_DIMENSION,
        metric="cosine",
    )


def build_embedders() -> tuple[OpenAITextEmbedder, OpenAIDocumentEmbedder]:
    common = dict(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        model=EMBEDDING_MODEL,
        dimensions=PINECONE_DIMENSION,
        api_base_url=OPENAI_BASE,
    )
    text_e = OpenAITextEmbedder(**common)
    doc_e = OpenAIDocumentEmbedder(**common)
    return text_e, doc_e


def retrieve_memory_context(
    user_id: int,
    query: str,
    text_embedder: OpenAITextEmbedder,
    retriever: PineconeEmbeddingRetriever,
) -> str:
    q_preview = query if len(query) <= 400 else query[:400] + "..."
    logger.info("ContextRetriever.run: Начало поиска. query: {}", q_preview)
    emb = text_embedder.run(text=query)["embedding"]
    logger.info("ContextRetriever.run: Эмбеддинг создан, размер: {}", len(emb))
    flt = {"field": "user_id", "operator": "==", "value": str(user_id)}
    logger.info("ContextRetriever.run: Поиск с фильтром: {}", flt)
    docs = retriever.run(query_embedding=emb, filters=flt, top_k=MEMORY_TOP_K)["documents"]
    logger.info("ContextRetriever.run: Найдено документов: {}", len(docs))
    if not docs:
        return ""
    lines = []
    for d in docs:
        content = (d.content or "").strip()
        if content:
            lines.append(f"- {content}")
    return "\n".join(lines)


def persist_turn(
    user_id: int,
    user_text: str,
    doc_embedder: OpenAIDocumentEmbedder,
    store: PineconeDocumentStore,
    username: str | None = None,
) -> None:
    """Сохраняет в Pinecone только текст текущего сообщения пользователя (без ответа ассистента)."""
    ts = datetime.now(timezone.utc).isoformat()
    logger.info(
        "ContextSaver.run: Начало сохранения (только user). text length: {}, user_id: {}, message_type: user_message",
        len(user_text),
        user_id,
    )
    uid = str(user_id)
    meta_user = {
        "user_id": uid,
        "type": "user_message",
        "timestamp": ts,
        "username": username or "",
    }
    logger.info("ContextSaver.run: Метаданные сформированы: {}", meta_user)
    docs = [
        Document(
            id=str(uuid.uuid4()),
            content=user_text,
            meta={**meta_user, "role": "user"},
        ),
    ]
    logger.info("ContextSaver.run: Документ создан (только сообщение пользователя), длина: {}", len(user_text))
    logger.info("ContextSaver.run: Создание эмбеддинга для документа")
    embed_out = doc_embedder.run(documents=docs)
    with_embeddings = embed_out["documents"]
    written = store.write_documents(with_embeddings, policy=DuplicatePolicy.OVERWRITE)
    logger.info("ContextSaver.run: Запись в Pinecone завершена, upserted: {}", written)


def forget_user_memory(store: PineconeDocumentStore, user_id: int) -> int:
    flt = {"field": "user_id", "operator": "==", "value": str(user_id)}
    return store.delete_by_filter(flt)


BASE_SYSTEM = """You are a warm, capable personal assistant in Telegram.
Stay coherent with the conversation: use short-term thread and long-term memory when provided.
Answer in the same language as the user.
Tools: catFactTool (random cat fact from catfact.ninja for questions about cats),
dogImageTool (random dog image URL), dogFactTool (random dog fact about dogs),
docImageAnalyzerTool (random dog photo + vision JSON for Telegram — do not rewrite the tool JSON),
weatherTool (current weather by city name using OpenWeather).
Use tools only when they fit the user's request.
If memory snippets are empty, rely on the current chat only."""


def make_system_prompt(memory_block: str) -> str:
    if memory_block.strip():
        return (
            f"{BASE_SYSTEM}\n\n"
            f"Long-term memory (relevant snippets, cosine-ranked):\n{memory_block}"
        )
    return f"{BASE_SYSTEM}\n\n(Long-term memory: no close matches for this query.)"


def split_telegram(text: str, limit: int = TELEGRAM_MAX_LEN) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    rest = text
    while rest:
        parts.append(rest[:limit])
        rest = rest[limit:]
    return parts


# --- Глобальная инициализация пайплайна памяти и агента ---
document_store = build_document_store()
text_embedder, document_embedder = build_embedders()
retriever = PineconeEmbeddingRetriever(document_store=document_store, top_k=MEMORY_TOP_K)

tools = [
    create_tool_from_function(cat_fact_tool, name=TOOL_NAME_CAT_FACT),
    create_tool_from_function(dog_image_tool, name="dogImageTool"),
    create_tool_from_function(dog_fact_tool, name="dogFactTool"),
    create_tool_from_function(doc_image_analyzer_tool, name=TOOL_NAME_DOC_IMAGE_ANALYZER),
    create_tool_from_function(weather_openweather_tool, name=TOOL_NAME_WEATHER),
]

chat_gen = OpenAIChatGenerator(
    api_key=Secret.from_env_var("OPENAI_API_KEY"),
    model=CHAT_MODEL,
    api_base_url=OPENAI_BASE,
)

agent = Agent(
    chat_generator=chat_gen,
    tools=tools,
    system_prompt=BASE_SYSTEM,
    exit_conditions=["text"],
    max_agent_steps=20,
    raise_on_tool_invocation_failure=False,
)

# Краткосрочная память: user_id -> deque[ChatMessage]
short_term: dict[int, deque] = {}

bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
if not bot_token:
    raise RuntimeError("Set TELEGRAM_BOT_TOKEN in .env")
bot = telebot.TeleBot(bot_token, parse_mode=None)


def get_history(user_id: int) -> deque:
    if user_id not in short_term:
        short_term[user_id] = deque(maxlen=SHORT_TERM_MAX_MESSAGES)
    return short_term[user_id]


@bot.message_handler(commands=["start"])
def handle_start(message: telebot.types.Message) -> None:
    name = (message.from_user.first_name if message.from_user else None) or "друг"
    bot.send_message(
        message.chat.id,
        f"Привет, {name}!\n\n"
        "Я твой умный персональный помощник с использованием Haystack Agent. Я запоминаю наши разговоры и "
        "использую эту информацию для более персонализированных ответов.\n\n"
        "Я умею:\n"
        "• Помнить контекст наших разговоров\n"
        "• Получать случайные факты о кошках 🐱 и о собаках 🐶\n"
        "• Показывать картинки собак и определять их породы 🖼️\n"
        "• Отвечать на вопросы с учетом истории общения\n\n"
        "Просто напиши мне что-нибудь, и я помогу тебе!\n\n"
        "Команды:\n"
        "/start — это приветствие\n"
        "/clear — очистить краткосрочный буфер диалога\n"
        "/forget — удалить долговременную память в Pinecone (этот бот, текущий namespace)",
    )


@bot.message_handler(commands=["clear"])
def handle_clear(message: telebot.types.Message) -> None:
    uid = message.from_user.id
    short_term.pop(uid, None)
    bot.send_message(message.chat.id, "Short-term buffer cleared.")


@bot.message_handler(commands=["forget"])
def handle_forget(message: telebot.types.Message) -> None:
    uid = message.from_user.id
    short_term.pop(uid, None)
    try:
        n = forget_user_memory(document_store, uid)
        bot.send_message(message.chat.id, f"Removed {n} long-term memory records for you in namespace '{PINECONE_NAMESPACE}'.")
    except Exception as exc:
        logger.exception("forget failed: {}", exc)
        bot.send_message(message.chat.id, "Could not erase long-term memory. Try again later.")


@bot.message_handler(func=lambda m: m.content_type == "text")
def handle_text(message: telebot.types.Message) -> None:
    user_id = message.from_user.id
    user_text = (message.text or "").strip()
    if not user_text:
        return

    bot.send_chat_action(message.chat.id, "typing")

    try:
        agent.warm_up()

        uname = (message.from_user.username if message.from_user else None) or ""
        if user_message_mentions_cat_facts(user_text):
            logger.info(
                "handle_text: запрос связан с кошками/фактами — рекомендуется catFactTool; query: %s",
                user_text[:300] + ("..." if len(user_text) > 300 else ""),
            )

        mem = retrieve_memory_context(user_id, user_text, text_embedder, retriever)
        system_prompt = make_system_prompt(mem)

        hist = list(get_history(user_id))
        runtime_messages = hist + [ChatMessage.from_user(user_text)]

        logger.info(
            "handle_text: запуск Agent, user_id=%s, history_msgs=%s, system_prompt_len=%s",
            user_id,
            len(runtime_messages),
            len(system_prompt),
        )
        result = agent.run(messages=runtime_messages, system_prompt=system_prompt)
        logger.info("handle_text: Agent завершён, ключи результата={}", list(result.keys()))
        final_messages = result.get("messages") or []
        if not final_messages:
            bot.send_message(message.chat.id, "No response from agent.")
            return
        reply = final_messages[-1].text or ""
        log_tool_results_summary(final_messages)
        logger.info("handle_text: Анализ ответа агента, длина: {}", len(reply))
        photo_url, photo_caption = extract_doc_analyzer_photo_from_messages(final_messages)

        if photo_url and photo_caption is not None:
            cap = photo_caption[:TELEGRAM_CAPTION_MAX]
            logger.info(
                "Telegram: send_photo url=%s caption_len=%s",
                photo_url[:80] + ("..." if len(photo_url) > 80 else ""),
                len(cap),
            )
            try:
                bot.send_photo(message.chat.id, photo_url, caption=cap)
            except Exception as exc:
                logger.exception("send_photo failed, fallback to text: {}", exc)
                bot.send_message(
                    message.chat.id,
                    f"{photo_url}\n\n{photo_caption}",
                )
            assistant_for_memory = photo_caption if photo_caption.strip() else reply
            if reply.strip() and reply.strip() != photo_caption.strip():
                for chunk in split_telegram(reply):
                    bot.send_message(message.chat.id, chunk)
                assistant_for_memory = f"{photo_caption}\n\n{reply}".strip()
        else:
            for chunk in split_telegram(reply):
                bot.send_message(message.chat.id, chunk)
            assistant_for_memory = reply

        get_history(user_id).append(ChatMessage.from_user(user_text))
        get_history(user_id).append(ChatMessage.from_assistant(assistant_for_memory))

        try:
            persist_turn(
                user_id,
                user_text,
                document_embedder,
                document_store,
                username=uname,
            )
        except Exception as persist_exc:
            logger.warning("Pinecone write failed (reply still sent): {}", persist_exc)
    except Exception as exc:
        logger.exception("handler error: {}", exc)
        bot.send_message(message.chat.id, "Something went wrong. Please try again.")


if __name__ == "__main__":
    logger.info(
        "Haystack Telegram bot | index=%s namespace=%s dim=%s model=%s",
        PINECONE_INDEX,
        PINECONE_NAMESPACE,
        PINECONE_DIMENSION,
        CHAT_MODEL,
    )
    bot.infinity_polling(timeout=30, long_polling_timeout=20)
