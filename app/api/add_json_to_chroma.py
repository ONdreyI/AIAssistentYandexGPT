from fastapi import APIRouter, HTTPException
from starlette.responses import JSONResponse
import os
import json
from loguru import logger
from langchain_core.documents import Document

from app.config import settings

from app.chroma_client.generate_chroma_db import ChromaVectorStore

router = APIRouter()


@router.post("/add_missing_jsons_to_chroma")
async def add_missing_jsons_to_chroma():
    try:
        # Убедимся, что ChromaManager инициализирован
        if not chroma_manager._store:
            await chroma_manager.init()

        parsed_json_path = settings.PARSED_JSON_PATH
        all_json_files = [
            f for f in os.listdir(parsed_json_path) if f.endswith(".json")
        ]

        # Получаем имена файлов существующих документов из метаданных ChromaDB
        existing_source_files = chroma_manager.get_all_source_filenames()
        logger.info(f"Найдено {len(existing_source_files)} документов в ChromaDB")

        documents_to_add = []
        added_files = []

        for json_file in all_json_files:
            # Проверяем, есть ли файл уже в БД по его имени в метаданных
            if json_file not in existing_source_files:
                file_path = os.path.join(parsed_json_path, json_file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        # Проверяем наличие необходимых полей
                        metadata = {}
                        page_content = None

                        # Проверяем различные варианты структуры JSON
                        if "metadata" in data:
                            metadata = data["metadata"]

                        if "page_content" in data:
                            page_content = data["page_content"]
                        elif "text" in data:
                            page_content = data["text"]

                        # Добавляем имя файла в метаданные для отслеживания
                        metadata["source_file"] = json_file

                        if page_content is not None:
                            documents_to_add.append(
                                Document(page_content=page_content, metadata=metadata)
                            )
                            added_files.append(json_file)
                            logger.info(
                                f"✅ Подготовлен документ из файла {json_file} для добавления в ChromaDB"
                            )
                        else:
                            logger.warning(
                                f"⚠️ Файл {json_file} не содержит 'page_content' или 'text'. Пропускаем."
                            )
                except json.JSONDecodeError as e:
                    logger.error(
                        f"❌ Ошибка декодирования JSON в файле {json_file}: {e}"
                    )
                except Exception as e:
                    logger.error(
                        f"❌ Неизвестная ошибка при чтении файла {json_file}: {e}"
                    )

        if documents_to_add:
            await chroma_manager.add_documents(documents_to_add)
            return JSONResponse(
                status_code=200,
                content={
                    "message": f"Successfully added {len(added_files)} new documents to ChromaDB.",
                    "added_files": added_files,
                },
            )
        else:
            return JSONResponse(
                status_code=200,
                content={"message": "All JSON documents are already in ChromaDB."},
            )

    except Exception as e:
        logger.error(f"Error adding missing JSONs to ChromaDB: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@router.get(
    "/list_documents", description="Получение списка всех документов в ChromaDB"
)
async def list_documents():
    try:
        # Убедимся, что ChromaManager инициализирован
        if not chroma_manager._store:
            await chroma_manager.init()

        # Получаем все метаданные и IDs из коллекции Chroma
        # IDs возвращаются по умолчанию при любом вызове get()
        result = chroma_manager._store._collection.get(include=["metadatas"])
        all_metadatas = result["metadatas"]
        all_ids = result["ids"]

        # Формируем список документов с их метаданными
        documents = []
        for i, metadata in enumerate(all_metadatas):
            doc_info = {
                "id": all_ids[i] if i < len(all_ids) else "unknown",
                "metadata": metadata,
            }
            documents.append(doc_info)

        return JSONResponse(
            status_code=200,
            content={"total_documents": len(documents), "documents": documents},
        )

    except Exception as e:
        logger.error(f"Error listing documents from ChromaDB: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
