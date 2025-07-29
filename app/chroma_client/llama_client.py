import requests
from loguru import logger
from typing import List, Dict, Any


class LlamaClient:
    def __init__(self, host: str = "localhost", port: int = 11434):
        self.api_url = f"http://{host}:{port}/v1/chat/completions"
        self.max_retries = 3  # Добавляем повторные попытки
        self.timeout = 120  # Увеличиваем таймаут до 120 секунд

    def invoke(self, messages: List[Dict[str, Any]]) -> str:
        payload = {
            "model": "llama3:8b",  # Official model name format for Ollama
            "messages": messages,
        }
        logger.debug(
            f"Sending request to Llama at {self.api_url} with payload: {payload}"
        )

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=self.timeout,
                )
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                else:
                    logger.error(f"Attempt {attempt+1} failed: {response.text}")
            except (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
            ) as e:
                logger.warning(f"Attempt {attempt+1} timed out: {e}")
                if attempt == self.max_retries - 1:
                    raise Exception(f"All {self.max_retries} attempts failed") from e

        raise Exception("Unexpected error in LlamaClient")
