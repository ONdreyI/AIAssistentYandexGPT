import torch
from langchain_chroma import Chroma
from loguru import logger
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import os

from app.config import settings


class ChromaVectorStore:
    def __init__(self):
        """
        Инициализирует пустой экземпляр хранилища векторов.
        Соединение с базой данных будет установлено позже с помощью метода init().
        """
        self._store: Chroma | None = None
        self.embeddings = None

    async def init(self):
        """
        Асинхронный метод для инициализации соединения с базой данных Chroma.
        Создает embeddings на основе модели из настроек, используя CUDA если доступно.
        """
        logger.info("🧠 Инициализация ChromaVectorStore...")
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"🚀 Используем устройство для эмбеддингов: {device}")

            self.embeddings = HuggingFaceEmbeddings(
                model_name=settings.LM_MODEL_NAME,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True},
            )

            self._store = Chroma(
                persist_directory=settings.STORAGE_CHROMA_PATH,
                embedding_function=self.embeddings,
                collection_name=settings.STORAGE_COLLECTION_NAME,
            )

            logger.success(
                f"✅ ChromaVectorStore успешно подключен к коллекции "
                f"'{settings.STORAGE_COLLECTION_NAME}' в '{settings.STORAGE_CHROMA_PATH}'"
            )
        except Exception as e:
            logger.exception(f"❌ Ошибка при инициализации ChromaVectorStore: {e}")
            raise

    async def asimilarity_search(self, query: str, with_score: bool, k: int = 3):
        """
        Асинхронный метод для поиска похожих документов в базе данных Chroma.

        Args:
            query (str): Текстовый запрос для поиска
            with_score (bool): Включать ли оценку релевантности в результаты
            k (int): Количество возвращаемых результатов

        Returns:
            list: Список найденных документов, возможно с оценками если with_score=True

        Raises:
            RuntimeError: Если хранилище не инициализировано
        """
        if not self._store:
            raise RuntimeError("ChromaVectorStore is not initialized.")
        logger.info(f"🔍 Поиск похожих документов по запросу: «{query}», top_k={k}")
        try:
            if with_score:
                # Используем асинхронный метод с правильными параметрами
                results = await self._store.asimilarity_search_with_score(
                    query=query, k=k
                )
            else:
                # Используем стандартный асинхронный метод
                results = await self._store.asimilarity_search(query=query, k=k)
            logger.debug(f"📄 Найдено {len(results)} результатов.")
            return results
        except Exception as e:
            logger.exception(f"❌ Ошибка при поиске: {e}")
            raise

    async def close(self):
        """
        Асинхронный метод для закрытия соединения с базой данных Chroma.
        В текущей реализации Chroma не требует явного закрытия,
        но метод добавлен для полноты API и возможных будущих изменений.
        """
        logger.info("🔌 Отключение ChromaVectorStore...")
        # Пока Chroma не требует явного закрытия, но в будущем может понадобиться
        # self._store.close() или подобный метод
        pass

    async def load_documents_to_chroma(self, directory_path: str):
        """
        Загружает документы из указанной директории в ChromaDB.
        """
        if not self.embeddings:
            raise RuntimeError("Embeddings not initialized. Call init() first.")

        logger.info(f"📚 Загрузка документов из директории: {directory_path}")
        documents = self._get_documents_from_directory(directory_path)

        if not documents:
            logger.warning("🤷 Документы для загрузки не найдены.")
            return

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHROMA_CHUNK_SIZE,
            chunk_overlap=settings.CHROMA_CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False,
        )

        texts = text_splitter.split_documents(documents)
        logger.info(f"✂️ Разделено на {len(texts)} чанков.")

        await self.add_documents(texts)

    async def add_documents(self, documents: list):
        """
        Добавляет список документов в существующую коллекцию ChromaDB.
        """
        if not self._store:
            raise RuntimeError(
                "ChromaVectorStore is not initialized. Call init() first."
            )
        if not self.embeddings:
            raise RuntimeError("Embeddings not initialized. Call init() first.")

        logger.info(f"➕ Добавление {len(documents)} документов в ChromaDB...")
        try:
            self._store.add_documents(documents)
            # Removed: self._store.persist() - no longer needed in ChromaDB 0.4.x and later
            logger.success("✅ Документы успешно добавлены и сохранены в ChromaDB.")
        except Exception as e:
            logger.exception(f"❌ Ошибка при добавлении документов в ChromaDB: {e}")
            raise

    def _get_documents_from_directory(self, directory_path: str):
        """
        Вспомогательный метод для загрузки JSON-файлов из директории.
        """
        documents = []
        for filename in os.listdir(directory_path):
            if filename.endswith(".json"):
                filepath = os.path.join(directory_path, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        # Предполагаем, что каждый JSON-файл содержит один документ
                        # с полями 'page_content' и 'metadata'
                        if "page_content" in data and "metadata" in data:
                            from langchain_core.documents import Document

                            documents.append(
                                Document(
                                    page_content=data["page_content"],
                                    metadata=data["metadata"],
                                )
                            )
                        else:
                            logger.warning(
                                f"⚠️ Файл {filename} не содержит 'page_content' или 'metadata'. Пропускаем."
                            )
                except json.JSONDecodeError as e:
                    logger.error(
                        f"❌ Ошибка декодирования JSON в файле {filename}: {e}"
                    )
                except Exception as e:
                    logger.error(
                        f"❌ Неизвестная ошибка при чтении файла {filename}: {e}"
                    )
        return documents

    def get_all_document_ids(self) -> set[str]:
        """
        Возвращает множество всех ID документов, хранящихся в ChromaDB.
        """
        if not self._store:
            raise RuntimeError("ChromaVectorStore is not initialized.")
        try:
            # ChromaDB хранит ID документов в своей коллекции. Мы можем получить их через _collection.get()
            # с include=['metadatas'] или include=['documents'] и извлечь ID.
            # Более прямой способ - получить все ID напрямую, если API это позволяет.
            # Если нет прямого метода, можно получить все документы и извлечь их ID.
            # Предполагаем, что ID доступны в метаданных или как часть возвращаемого объекта.
            # В Langchain Chroma, ID документов обычно являются частью возвращаемых результатов
            # при запросе или могут быть получены через внутренний доступ к коллекции.
            # Для получения всех ID, мы можем запросить все документы и извлечь их ID.
            # Это может быть неэффективно для очень больших коллекций.
            # Более эффективный способ - использовать _collection.get(ids=None, where=None, limit=None,
            # offset=None, include=[]) и извлечь 'ids'.

            # Получаем все ID из коллекции Chroma
            # Внимание: это может быть медленно для очень больших коллекций
            all_ids = self._store._collection.get(
                ids=None, where=None, limit=None, offset=None, include=[]
            )["ids"]
            logger.info(f"📊 Получено {len(all_ids)} ID документов из ChromaDB.")
            return set(all_ids)
        except Exception as e:
            logger.exception(
                f"❌ Ошибка при получении всех ID документов из ChromaDB: {e}"
            )
            raise

    def get_all_source_filenames(self) -> set[str]:
        """
        Возвращает множество имен исходных файлов всех документов, хранящихся в ChromaDB.
        Извлекает значения 'source_file' из метаданных документов.
        """
        if not self._store:
            raise RuntimeError("ChromaVectorStore is not initialized.")
        try:
            # Получаем все метаданные из коллекции Chroma
            all_metadatas = self._store._collection.get(include=["metadatas"])[
                "metadatas"
            ]
            source_files = set()

            # Извлекаем имена исходных файлов из метаданных
            for metadata in all_metadatas:
                if metadata and "source_file" in metadata:
                    source_files.add(metadata["source_file"])

            logger.info(
                f"📊 Получено {len(source_files)} имен исходных файлов из метаданных ChromaDB."
            )
            return source_files
        except Exception as e:
            logger.exception(
                f"❌ Ошибка при получении имен исходных файлов из ChromaDB: {e}"
            )
            raise


# Глобальный инстанс
chroma_vectorstore = ChromaVectorStore()


# Зависимость
def get_vectorstore() -> ChromaVectorStore:
    return chroma_vectorstore
