from typing import Any, Dict, List, Literal, Optional
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

from app.config import settings
from app.chroma_client.llama_client import LlamaClient
from app.chroma_client.yandex_client import YandexGPTWrapper


class ChatWithAI:
    def __init__(
        self,
        provider: Literal[
            "deepseek", "openai", "llama", "llama32", "yandexgpt"
        ] = "yandexgpt",
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

        if provider == "deepseek":
            self.llm = ChatDeepSeek(
                api_key=settings.DEEPSEEK_API_KEY,
                model=settings.DEEPSEEK_MODEL_NAME,
                temperature=0.7,
            )
        elif provider == "openai":
            self.llm = ChatOpenAI(
                api_key=settings.OPENAI_API_KEY,
                model=settings.OPENAI_MODEL_NAME,
                temperature=0.7,
            )
        elif provider == "llama":
            raise NotImplementedError("Legacy llama provider not implemented")
        elif provider == "llama32":
            self.llm = LlamaClient(host="localhost", port=11434)
        elif provider == "yandexgpt":
            self.llm = YandexGPTWrapper(
                folder_id=settings.YANDEX_IDENTIFICATOR, api_key=settings.YANDEX_API_KEY
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        self.chroma_db = Chroma(
            persist_directory=settings.STORAGE_CHROMA_PATH,
            embedding_function=self.embeddings,
            collection_name=settings.STORAGE_COLLECTION_NAME,
        )

    def log_dialog(self, user_query: str, ai_response: str) -> None:
        if not self.log_dialogs:
            return
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] Пользователь: {user_query}\n")
                f.write(f"[{timestamp}] ИИ: {ai_response}\n\n")
        except Exception as e:
            logger.error(f"Ошибка при записи диалога в лог: {e}")

    def get_relevant_context(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        try:
            results = self.chroma_db.similarity_search(query, k=k)
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

    def generate_response(self, query: str) -> Optional[str]:
        try:
            context = self.get_relevant_context(query)
            if not context:
                response = "Извините, не удалось найти релевантный контекст для ответа."
                self.log_dialog(query, response)
                return response

            formatted_context = self.format_context(context)

            messages = [
                {
                    "role": "system",
                    "content": """Ты — внутренний менеджер компании Минко(продажа бентонитов и полимеров марки Baroid IDP). Отвечаешь по делу без лишних вступлений.
                    Правила:
                    1. Сразу переходи к сути, без фраз типа "На основе контекста"
                    2. Используй только факты. Если точных данных нет — отвечай общими фразами о бентоните и полимерах, но не придумывай конкретику
                    3. Используй обычный текст без форматирования
                    4. Включай ссылки только если они есть в контексте
                    5. Говори от первого лица множественного числа: "Мы предоставляем", "У нас есть"
                    6. При упоминании файлов делай это естественно, например: "Я прикреплю инструкцию, где подробно описаны шаги"
                    7. На приветствия отвечай доброжелательно, на негатив — с легким юмором
                    8. Можешь при ответах использовать общую информацию из открытых источников по бентонитам и полимерам, но опирайся на контекст
                    9. При вопросах о ценах или характеристиках опираемся на контекст, например, рецептуры продуктов вроде BORE-GEL (15–72 кг/м³)
                    10. Для технических вопросов предлагаем практические решения, такие как порядок добавления реагентов (кальцинированная сода, бентонит, полимеры).
                    11. Для расчета объема бурового раствора используй следующую формулу: Объем бурового раствора (литров/метр) = Диаметр расширителя (дюймы)² ÷ 2 * 3,5
                    (примеры: 
                    Для диаметра 150 мм (~6 дюймов): 6² ÷ 2 = 18 л/м.
                    Для диаметра 200 мм (~8 дюймов): 8² ÷ 2 = 32 л/м.
                    Для диаметра 250 мм (~10 дюймов): 10² ÷ 2 = 50 л/м.
                    Для диаметра 300 мм (~12 дюймов): 12² ÷ 2 = 72 л/м.
                    Для больших диаметров произведи расчет и экстраполируй
                    )
                    12. В ответах не пиши формулу расчета (она должна быть скрыта, покажывай ее, только если заказчик попросит)
                    13. Если заказчик в чате указывает диаметр трубы, а не расширителя, бери нименьший больший диаметр расширителя для расчета.
                    14. Результаты расчетов объема раствора предоставляй в м³. 
                    15. В конце рассчитай сколько всего необходимо каждого материала для проведения работ


                    Персонализируй ответы, упоминая имя клиента если оно есть в контексте. Будь краток, информативен и полезен.
                    """,
                }
            ]
            messages.extend(self.message_history)
            messages.append(
                {
                    "role": "user",
                    "content": f"Вопрос: {query}\nКонтекст: {formatted_context}",
                }
            )

            if self.provider == "yandexgpt":
                messages = [
                    {"role": msg["role"], "text": msg["content"]} for msg in messages
                ]
                response = self.llm.invoke(messages)
                logger.debug(
                    f"YandexGPT response: {json.dumps(vars(response), default=str)}"
                )

                if hasattr(response, "alternatives") and response.alternatives:
                    alternative = response.alternatives[0]
                    if hasattr(alternative, "message") and hasattr(
                        alternative.message, "text"
                    ):
                        result = alternative.message.text.strip()
                    elif hasattr(alternative, "text"):
                        result = alternative.text.strip()
                    else:
                        result = "Ошибка: неожиданная структура ответа от YandexGPT."
                else:
                    result = "Ошибка: нет альтернатив в ответе от YandexGPT."
            else:
                response = self.llm.invoke(messages)
                if hasattr(response, "content"):
                    result = str(response.content).strip()
                else:
                    result = str(response).strip()

            self.log_dialog(query, result)
            self.message_history.append({"role": "user", "content": query})
            self.message_history.append({"role": "assistant", "content": result})
            self._truncate_history()
            return result

        except Exception as e:
            logger.error(f"Ошибка при генерации ответа: {e}")
            result = "Произошла ошибка при генерации ответа."
            self.log_dialog(query, result)
            return result


if __name__ == "__main__":
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
        response = chat.generate_response(query)
        print(" " * 20, end="\r")
        print(f"ИИ: {response}\n")
