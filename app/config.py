import os
import re
import string

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    DEEPSEEK_API_KEY: SecretStr
    OPENAI_API_KEY: SecretStr
    YANDEX_API_KEY: str
    YANDEX_IDENTIFICATOR: str
    YANDEX_IDENTIFICATOR_KEY: str
    YANDEX_BASE_URL: str = "https://llm.api.cloud.yandex.net/v1"
    BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    DOCS_AMVERA_PATH: str = os.path.join(BASE_DIR, "docs_amvera")
    PARSED_JSON_PATH: str = os.path.join(BASE_DIR, "parsed_json")
    STORAGE_CHROMA_PATH: str = os.path.join(BASE_DIR, "chroma_db")
    STORAGE_COLLECTION_NAME: str = "storage_docs"
    MAX_CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    LM_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    DEEPSEEK_MODEL_NAME: str = "deepseek-chat"
    OPENAI_MODEL_NAME: str = "gpt-3.5-turbo"
    LLAMA_MODEL_PATH: str = os.path.join(BASE_DIR, "models", "llama-3-7b-chat.gguf")
    USERS: str = os.path.join(BASE_DIR, "..", "users.json")
    SECRET_KEY: str
    ALGORITHM: str
    model_config = SettingsConfigDict(env_file=f"{BASE_DIR}/.env")


settings = Config()  # type: ignore
