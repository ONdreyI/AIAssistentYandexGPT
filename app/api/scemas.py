from pydantic import BaseModel, Field
from typing import Literal, Optional


class SUserAuth(BaseModel):
    login: str
    password: str


class AskResponse(BaseModel):
    response: str
    k: int = Field(default=5, description="Количество документов для поиска")


class AskWithAIResponse(BaseModel):
    response: str
    provider: Literal["deepseek", "chatgpt", "yandexgpt", "llama", "llama32"] = "yandexgpt"
    k: int = Field(default=5, description="Количество документов для поиска")


class AskCombinedResponse(BaseModel):
    response: str
    provider: Literal["yandexgpt", "llama", "llama32"] = "yandexgpt"
    k: int = Field(default=5, description="Количество документов для поиска")
    include_documents: bool = Field(
        default=True, description="Включать ли документы в ответ"
    )


class ChatDialogRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    context: Optional[str] = None
