import logging

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, api_key: str, endpoint: str, model_id: str):
        self.api_key = api_key
        self.endpoint = endpoint
        self.model_id = model_id
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def generate(self, prompt: str) -> str:
        """Generate text using the Grok API."""
        try:
            data = {
                "model": self.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
            }
            response = requests.post(self.endpoint, headers=self.headers, json=data)
            response.raise_for_status()
            result = response.json()
            text = result["choices"][0]["message"]["content"]
            if not isinstance(text, str):
                logger.error(f"Invalid response format: {text}")
                raise ValueError("Invalid response format")
            return text
        except requests.exceptions.RequestException as e:
            logger.error(f"Grok API call failed: {str(e)}")
            raise
