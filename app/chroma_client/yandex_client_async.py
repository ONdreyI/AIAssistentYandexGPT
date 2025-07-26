from typing import AsyncGenerator
from yandex_cloud_ml_sdk import AsyncYCloudML
from langchain_core.messages import AIMessage


class AsyncYandexGPTWrapper:
    def __init__(self, folder_id: str, api_key: str, model_name: str = "yandexgpt"):
        self.sdk = AsyncYCloudML(folder_id=folder_id, auth=api_key)
        self.model = self.sdk.models.completions(model_name)

    async def astream(self, messages) -> AsyncGenerator[AIMessage, None]:
        try:
            # Конвертируем сообщения LangChain в формат, подходящий для YandexGPT
            formatted_messages = [
                {
                    "role": "system" if msg.type == "system" else "user",
                    "text": msg.content,
                }
                for msg in messages
            ]

            # Настраиваем модель и запускаем запрос
            operation = await self.model.configure(temperature=0.7).run_deferred(
                formatted_messages
            )
            result = await operation

            # Проверяем структуру ответа
            if not hasattr(result, "alternatives") or not result.alternatives:
                yield AIMessage(
                    content="Ошибка: нет альтернатив в ответе от YandexGPT."
                )
                return

            alternative = result.alternatives[0]
            if hasattr(alternative, "message") and hasattr(alternative.message, "text"):
                text = alternative.message.text.strip()
            elif hasattr(alternative, "text"):
                text = alternative.text.strip()
            else:
                yield AIMessage(
                    content="Ошибка: неожиданная структура ответа от YandexGPT."
                )
                return

            # Эмуляция стриминга
            chunk_size = 50  # Увеличим размер чанка для большей плавности
            for i in range(0, len(text), chunk_size):
                chunk = text[i : i + chunk_size]
                yield AIMessage(content=chunk)

        except Exception as e:
            yield AIMessage(content=f"Ошибка при запросе к YandexGPT: {str(e)}")
