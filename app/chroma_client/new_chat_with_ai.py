from typing import Any, Dict, List, Literal, Optional, AsyncGenerator
import datetime
import os
import torch
from langchain_chroma import Chroma
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from loguru import logger
import time
import json
import asyncio
from langchain_core.messages import HumanMessage, SystemMessage

from config import settings
from llama_client import LlamaClient
from yandexgpt_client import YandexGPT
from chroma_client.yandex_client_async import AsyncYandexGPTWrapper


class ChatWithAI:
    def __init__(
        self,
        provider: Literal["yandexgpt", "llama", "llama32"] = "yandexgpt",
        log_dialogs: bool = True,
        log_file_path: str = None,
    ):
        self.provider = provider
        self.log_dialogs = log_dialogs
        self.message_history = []

        if log_file_path is None:
            logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
            os.makedirs(logs_dir, exist_ok=True)
            self.log_file_path = os.path.join(
                logs_dir,
                f"dialog_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            )
        else:
            self.log_file_path = log_file_path

        if self.log_dialogs:
            with open(self.log_file_path, "w", encoding="utf-8") as f:
                f.write(
                    f"=== Диалог начат {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n"
                )

        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.LM_MODEL_NAME,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        if provider == "yandexgpt":
            self.llm = AsyncYandexGPTWrapper(
                folder_id=settings.YANDEX_IDENTIFICATOR, api_key=settings.YANDEX_API_KEY
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        self.chroma_db = Chroma(
            persist_directory=settings.STORAGE_CHROMA_PATH,
            embedding_function=self.embeddings,
            collection_name=settings.STORAGE_COLLECTION_NAME,
        )

    async def log_dialog(self, user_query: str, ai_response: str) -> None:
        if not self.log_dialogs:
            return
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] Пользователь: {user_query}\n")
                f.write(f"[{timestamp}] ИИ: {ai_response}\n\n")
        except Exception as e:
            logger.error(f"Ошибка при записи диалога в лог: {e}")

    async def get_relevant_context(
        self, query: str, k: int = 3
    ) -> List[Dict[str, Any]]:
        try:
            results = await self.chroma_db.asimilarity_search(query, k=k)
            return [
                {"text": doc.page_content, "metadata": doc.metadata} for doc in results
            ]
        except Exception as e:
            logger.error(f"Ошибка при получении контекста: {e}")
            return []

    def format_context(self, context: List[Dict[str, Any]]) -> str:
        formatted_context = []
        for item in context:
            metadata_str = "\n".join(f"{k}: {v}" for k, v in item["metadata"].items())
            formatted_context.append(
                f"Текст: {item['text']}\nМетаданные:\n{metadata_str}\n"
            )
        return "\n---\n".join(formatted_context)

    def _truncate_history(self):
        max_history = 10
        if len(self.message_history) > max_history * 2:
            self.message_history = self.message_history[-max_history * 2 :]

    async def astream_response(self, query: str) -> AsyncGenerator[str, None]:
        full_response = ""
        try:
            context = await self.get_relevant_context(query)
            if not context:
                response = "Извините, не удалось найти релевантный контекст для ответа."
                await self.log_dialog(query, response)
                yield response
                return

            formatted_context = self.format_context(context)

            system_message = SystemMessage(
                content="""Ты — внутренний менеджер компании Минко..."""  # Ваш текущий system prompt
            )
            human_message = HumanMessage(
                content=f"Вопрос: {query}\nКонтекст: {formatted_context}. Ответ форматируй в markdown!"
            )

            # Учитываем историю сообщений
            messages = self.message_history + [system_message, human_message]

            logger.info(
                f"Начинаем стриминг ответа для запроса: {query} (провайдер: {self.provider}, длина контекста: {len(formatted_context)} символов)"
            )

            async for chunk in self.llm.astream(messages):
                if chunk.content:
                    logger.debug(f"Получен чанк: {chunk.content[:50]}...")
                    full_response += chunk.content
                    yield chunk.content

            logger.info("Стриминг ответа завершен")
            await self.log_dialog(query, full_response)
            self.message_history.append({"role": "user", "content": query})
            self.message_history.append({"role": "assistant", "content": full_response})
            self._truncate_history()

        except Exception as e:
            error_message = f"Ошибка при стриминге ответа: {str(e)}"
            logger.error(error_message)
            yield error_message
            await self.log_dialog(query, error_message)


async def main():
    chat = ChatWithAI(provider="yandexgpt", log_dialogs=True)
    print(f"\n=== Чат с ИИ ===\n")
    print(f"Логи диалога сохраняются в: {chat.log_file_path}\n")

    while True:
        query = input("Вы: ")
        if query.lower() == "выход":
            print("\nДо свидания!")
            with open(chat.log_file_path, "a", encoding="utf-8") as f:
                f.write(
                    f"\n=== Диалог завершен {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n"
                )
            break

        print("\nИИ печатает...", end="\r")
        full_response_content = ""
        async for chunk in chat.astream_response(query):
            full_response_content += chunk
            print(chunk, end="")
        print(" " * 20, end="\r")
        print(f"ИИ: {full_response_content}\n")


if __name__ == "__main__":
    asyncio.run(main())
