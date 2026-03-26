"""
Модуль для управления чтением и записью в векторную базу данных Pinecone.
Поддерживает различные варианты записи (документы, векторы) и чтения (по вектору, по тексту).

Перед записью выполняется проверка косинусного сходства с существующими записями:
- Низкое сходство (< порога) → новая информация → запоминаем
- Высокое сходство (>= порога) → дубликат/вариация → пропускаем или обновляем существующий слот
"""

import os
from typing import Any, Literal, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

# --- Настройки проверки косинусного сходства перед записью в долговременную память ---
# Порог: выше него — считаем дубликатом/вариацией, ниже — новой информацией
COSINE_SIMILARITY_THRESHOLD: float = 0.80
# Действие при высоком сходстве: "skip" — пропустить запись, "update" — обновить найденный слот
COSINE_DEDUP_ACTION: Literal["skip", "update"] = "skip"
# Включить/выключить проверку перед записью
COSINE_DEDUP_ENABLED: bool = True


class PineconeManager:
    """
    Класс для управления векторной базой данных Pinecone.
    Поддерживает запись документов и векторов, а также поиск по вектору и по тексту.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        embed_model: str = "text-embedding-3-small",
        namespace: str = "",
    ):
        """
        Инициализация менеджера Pinecone.

        Args:
            api_key: API ключ Pinecone (по умолчанию из env PINECONE_API_KEY)
            index_name: Имя индекса (по умолчанию из env PINECONE_INDEX_NAME)
            openai_api_key: API ключ OpenAI для эмбеддингов (по умолчанию из env OPENAI_API_KEY)
            openai_base_url: Базовый URL OpenAI API для прокси (по умолчанию из env OPENAI_BASE_URL)
            embed_model: Модель для создания эмбеддингов
            namespace: Пространство имён в индексе (пустая строка = default namespace)
        """
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME")
        self.namespace = namespace or ""
        self.embed_model = embed_model

        if not self.api_key or not self.index_name:
            raise ValueError("Требуются PINECONE_API_KEY и PINECONE_INDEX_NAME")

        self._pc = Pinecone(api_key=self.api_key)
        self._index = self._pc.Index(
            name=self.index_name,
            pool_threads=30,
            connection_pool_maxsize=30,
        )

        # Инициализация OpenAI для эмбеддингов (опционально)
        self._openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self._openai_base_url = openai_base_url or os.getenv("OPENAI_BASE_URL")

        self._openai_client: Optional[OpenAI] = None
        if self._openai_api_key:
            kwargs: dict[str, Any] = {"api_key": self._openai_api_key}
            base_url = self._openai_base_url or os.getenv("OPENAI_BASE_URL")
            if base_url:
                kwargs["base_url"] = base_url
            self._openai_client = OpenAI(**kwargs)

    @property
    def openai_client(self) -> Optional[OpenAI]:
        """Возвращает клиент OpenAI (если настроен)."""
        return self._openai_client

    def _get_embedding(self, text: str) -> list[float]:
        """Создаёт эмбеддинг текста через OpenAI."""
        if not self._openai_client:
            raise RuntimeError(
                "OpenAI не настроен. Укажите OPENAI_API_KEY для поиска по тексту и записи документов."
            )
        response = self._openai_client.embeddings.create(
            model=self.embed_model,
            input=text,
        )
        return response.data[0].embedding

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Создаёт эмбеддинги для списка текстов."""
        if not self._openai_client:
            raise RuntimeError(
                "OpenAI не настроен. Укажите OPENAI_API_KEY для поиска по тексту и записи документов."
            )
        base_url = os.getenv("OPENAI_BASE_URL")
        client = OpenAI(
            api_key=self._openai_api_key,
            **({"base_url": base_url} if base_url else {}),
        )
        response = client.embeddings.create(
            model=self.embed_model,
            input=texts,
        )
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

    def _check_similarity_before_store(
        self,
        vector: list[float],
        filter: Optional[dict[str, Any]] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        Проверяет косинусное сходство с уже сохранёнными фрагментами.

        Returns:
            (should_store, existing_id): should_store=True — новая информация, записать;
                False — дубликат/вариация; existing_id — id найденного слота (при action=update).
        """
        if not COSINE_DEDUP_ENABLED:
            return (True, None)

        matches = self.query_by_vector(
            vector=vector,
            top_k=1,
            include_metadata=False,
            include_values=False,
            filter=filter,
        )

        if not matches or matches[0].get("score") is None:
            return (True, None)

        score = float(matches[0]["score"])
        if score < COSINE_SIMILARITY_THRESHOLD:
            return (True, None)

        # Высокое сходство — дубликат или вариация
        existing_id = matches[0].get("id")
        if COSINE_DEDUP_ACTION == "update" and existing_id:
            return (False, existing_id)
        return (False, None)

    # --- Запись ---

    def upsert_vector(
        self,
        id: str,
        vector: list[float],
        metadata: Optional[dict[str, Any]] = None,
        check_similarity: Optional[bool] = None,
        filter: Optional[dict[str, Any]] = None,
    ) -> dict:
        """
        Записывает один вектор в индекс.
        Перед записью проверяет косинусное сходство (если COSINE_DEDUP_ENABLED).

        Args:
            id: Уникальный идентификатор записи
            vector: Вектор эмбеддинга
            metadata: Метаданные (опционально)
            check_similarity: Проверять сходство перед записью (None = использовать COSINE_DEDUP_ENABLED)
            filter: Фильтр при поиске похожих записей

        Returns:
            Результат операции upsert или пустой dict при пропуске
        """
        do_check = check_similarity if check_similarity is not None else COSINE_DEDUP_ENABLED
        if do_check:
            should_store, existing_id = self._check_similarity_before_store(vector, filter=filter)
            if not should_store:
                if existing_id and COSINE_DEDUP_ACTION == "update":
                    record = {"id": existing_id, "values": vector}
                    if metadata:
                        record["metadata"] = metadata
                    kwargs: dict[str, Any] = {"vectors": [record]}
                    if self.namespace:
                        kwargs["namespace"] = self.namespace
                    return self._index.upsert(**kwargs)
                return {"upserted_count": 0}

        record = {"id": id, "values": vector}
        if metadata:
            record["metadata"] = metadata

        kwargs = {"vectors": [record]}
        if self.namespace:
            kwargs["namespace"] = self.namespace

        return self._index.upsert(**kwargs)

    def upsert_vectors(
        self,
        vectors: list[dict[str, Any]],
        batch_size: int = 100,
        check_similarity: Optional[bool] = None,
        filter: Optional[dict[str, Any]] = None,
    ) -> dict:
        """
        Записывает несколько векторов в индекс.
        Опционально проверяет косинусное сходство перед записью каждого вектора.

        Args:
            vectors: Список записей формата [{"id": str, "values": list[float], "metadata": dict?}, ...]
            batch_size: Размер батча для пакетной записи
            check_similarity: Проверять сходство перед записью (None = использовать COSINE_DEDUP_ENABLED)
            filter: Фильтр при поиске похожих записей

        Returns:
            Результат операции upsert
        """
        do_check = check_similarity if check_similarity is not None else COSINE_DEDUP_ENABLED
        to_upsert: list[dict[str, Any]] = []

        for v in vectors:
            record_id = v.get("id", "")
            vec = v.get("values", [])
            meta = v.get("metadata")

            if do_check and vec:
                should_store, existing_id = self._check_similarity_before_store(vec, filter=filter)
                if not should_store:
                    if existing_id and COSINE_DEDUP_ACTION == "update":
                        rec = {"id": existing_id, "values": vec}
                        if meta is not None:
                            rec["metadata"] = meta
                        to_upsert.append(rec)
                    continue

            to_upsert.append(v)

        if not to_upsert:
            return {"upserted_count": 0}

        kwargs: dict[str, Any] = {"vectors": to_upsert, "batch_size": batch_size}
        if self.namespace:
            kwargs["namespace"] = self.namespace

        return self._index.upsert(**kwargs)

    def upsert_document(
        self,
        id: str,
        text: str,
        metadata: Optional[dict[str, Any]] = None,
        check_similarity: Optional[bool] = None,
        filter: Optional[dict[str, Any]] = None,
    ) -> dict:
        """
        Записывает документ (текст) в индекс. Текст преобразуется в эмбеддинг через OpenAI.
        Перед записью проверяет косинусное сходство с существующими фрагментами.

        Args:
            id: Уникальный идентификатор записи
            text: Текст документа
            metadata: Метаданные (опционально). Рекомендуется добавить "text" для хранения исходного текста
            check_similarity: Проверять сходство перед записью (None = использовать COSINE_DEDUP_ENABLED)
            filter: Фильтр при поиске похожих записей

        Returns:
            Результат операции upsert
        """
        vector = self._get_embedding(text)
        meta = metadata or {}
        if "text" not in meta:
            meta["text"] = text

        return self.upsert_vector(
            id=id, vector=vector, metadata=meta,
            check_similarity=check_similarity, filter=filter,
        )

    def upsert_documents(
        self,
        documents: list[dict[str, Any]],
        text_field: str = "text",
        id_field: str = "id",
        batch_size: int = 100,
        check_similarity: Optional[bool] = None,
        filter: Optional[dict[str, Any]] = None,
    ) -> dict:
        """
        Записывает несколько документов в индекс.
        Для каждого документа проверяется косинусное сходство перед записью.

        Args:
            documents: Список документов [{"id": str, "text": str, ...metadata}, ...]
            text_field: Поле с текстом для эмбеддинга
            id_field: Поле с идентификатором
            batch_size: Размер батча для пакетной записи
            check_similarity: Проверять сходство перед записью (None = использовать COSINE_DEDUP_ENABLED)
            filter: Фильтр при поиске похожих записей

        Returns:
            Результат операции upsert
        """
        texts = [doc[text_field] for doc in documents]
        embeddings = self._get_embeddings(texts)

        do_check = check_similarity if check_similarity is not None else COSINE_DEDUP_ENABLED
        vectors: list[dict[str, Any]] = []

        for i, doc in enumerate(documents):
            record_id = doc.get(id_field, doc.get("_id", str(i)))
            meta = {k: v for k, v in doc.items() if k not in (id_field, "_id")}
            if text_field not in meta and text_field in doc:
                meta[text_field] = doc[text_field]

            vec = embeddings[i]
            if do_check:
                should_store, existing_id = self._check_similarity_before_store(vec, filter=filter)
                if not should_store:
                    if existing_id and COSINE_DEDUP_ACTION == "update":
                        vectors.append({"id": str(existing_id), "values": vec, "metadata": meta})
                    continue

            vectors.append(
                {"id": str(record_id), "values": vec, "metadata": meta}
            )

        if not vectors:
            return {"upserted_count": 0}
        return self.upsert_vectors(vectors, batch_size=batch_size, check_similarity=False)

    # --- Чтение ---

    def query_by_vector(
        self,
        vector: list[float],
        top_k: int = 5,
        include_metadata: bool = True,
        include_values: bool = False,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """
        Поиск по вектору (семантический поиск).

        Args:
            vector: Вектор запроса
            top_k: Количество возвращаемых результатов
            include_metadata: Включать ли метаданные в ответ
            include_values: Включать ли векторы в ответ
            filter: Фильтр по метаданным (Pinecone filter syntax)

        Returns:
            Список найденных записей с id, score, metadata
        """
        kwargs: dict[str, Any] = {
            "vector": vector,
            "top_k": top_k,
            "include_metadata": include_metadata,
            "include_values": include_values,
        }
        if self.namespace:
            kwargs["namespace"] = self.namespace
        if filter:
            kwargs["filter"] = filter

        result = self._index.query(**kwargs)

        matches = []
        for m in result.matches:
            match_dict: dict[str, Any] = {
                "id": m.id,
                "score": m.score,
            }
            if include_metadata and m.metadata:
                match_dict["metadata"] = dict(m.metadata)
            if include_values and m.values:
                match_dict["values"] = list(m.values)
            matches.append(match_dict)

        return matches

    def query_by_text(
        self,
        text: str,
        top_k: int = 5,
        include_metadata: bool = True,
        include_values: bool = False,
        filter: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """
        Поиск по тексту. Текст преобразуется в эмбеддинг, затем выполняется поиск.

        Args:
            text: Текст запроса
            top_k: Количество возвращаемых результатов
            include_metadata: Включать ли метаданные в ответ
            include_values: Включать ли векторы в ответ
            filter: Фильтр по метаданным

        Returns:
            Список найденных записей
        """
        vector = self._get_embedding(text)
        return self.query_by_vector(
            vector=vector,
            top_k=top_k,
            include_metadata=include_metadata,
            include_values=include_values,
            filter=filter,
        )

    def fetch(
        self,
        ids: list[str],
        include_metadata: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """
        Получение записей по списку идентификаторов.

        Args:
            ids: Список идентификаторов
            include_metadata: Включать ли метаданные

        Returns:
            Словарь {id: {"id", "values"?, "metadata"?}}
        """
        kwargs: dict[str, Any] = {"ids": ids}
        if self.namespace:
            kwargs["namespace"] = self.namespace

        result = self._index.fetch(**kwargs)

        records: dict[str, dict[str, Any]] = {}
        for id_, record in (result.vectors or {}).items():
            records[id_] = {"id": id_}
            if record.values:
                records[id_]["values"] = list(record.values)
            if include_metadata and record.metadata:
                records[id_]["metadata"] = dict(record.metadata)

        return records

    def fetch_one(self, id: str, include_metadata: bool = True) -> Optional[dict[str, Any]]:
        """
        Получение одной записи по идентификатору.

        Returns:
            Запись или None, если не найдена
        """
        records = self.fetch(ids=[id], include_metadata=include_metadata)
        return records.get(id)

    # --- Удаление ---

    def delete(self, ids: Optional[list[str]] = None, filter: Optional[dict[str, Any]] = None) -> None:
        """
        Удаление записей по id или по фильтру.

        Args:
            ids: Список идентификаторов для удаления
            filter: Фильтр по метаданным (удаляются все записи, удовлетворяющие фильтру)

        Note:
            Необходимо указать ids или filter (хотя бы один параметр)
        """
        if not ids and not filter:
            raise ValueError("Необходимо указать ids или filter")

        kwargs: dict[str, Any] = {}
        if self.namespace:
            kwargs["namespace"] = self.namespace
        if ids:
            kwargs["ids"] = ids
        if filter:
            kwargs["filter"] = filter

        self._index.delete(**kwargs)

    def delete_all(self) -> None:
        """Удаляет все записи в текущем namespace."""
        kwargs: dict[str, Any] = {}
        if self.namespace:
            kwargs["namespace"] = self.namespace
        self._index.delete(**kwargs)

    # --- Статистика ---

    def stats(self) -> dict[str, Any]:
        """
        Возвращает статистику индекса (количество векторов и т.д.).
        """
        result = self._index.describe_index_stats()
        return {
            "dimension": result.dimension,
            "index_fullness": getattr(result, "index_fullness", None),
            "namespaces": dict(result.namespaces) if result.namespaces else {},
            "total_vector_count": result.total_vector_count,
        }

if __name__ == "__main__":
    import pprint

    SEARCH_TEXT = "Марс"
    SEARCH_TOP_K = 20

    # Получаем все namespace'ы из статистики индекса
    base = PineconeManager()
    stats = base.stats()
    namespaces = list(stats.get("namespaces", {}).keys()) or [""]

    print(f"Поиск: '{SEARCH_TEXT}' | namespace'ы: {namespaces}\n{'=' * 60}")

    all_results: list[dict] = []
    for ns in namespaces:
        manager = PineconeManager(namespace=ns)
        results = manager.query_by_text(text=SEARCH_TEXT, top_k=SEARCH_TOP_K)
        for r in results:
            r["_namespace"] = ns
        all_results.extend(results)

    # Сортируем по score (убывание)
    all_results.sort(key=lambda x: x.get("score", 0), reverse=True)

    if not all_results:
        print("Ничего не найдено.")
    else:
        print(f"Найдено записей: {len(all_results)}\n")
        pprint.pprint(all_results, width=120, sort_dicts=False)