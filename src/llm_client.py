import asyncio
import logging
from typing import Any, Dict

import aiohttp

logger = logging.getLogger(__name__)


class LLMClient:
    """Async client for LLM API calls."""

    def __init__(
        self,
        api_key: str = "",
        endpoint: str = "https://api-inference.huggingface.co",
        model_id: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
    ):
        """
        Initialize LLMClient.

        Args:
            api_key (str): API key for authentication.
            endpoint (str): API endpoint URL.
            model_id (str): Hugging Face model ID (e.g., 'mistralai/Mixtral-8x7B-Instruct-v0.1').
        """
        self.api_key = api_key
        self.endpoint = endpoint
        self.model_id = model_id

    async def call_llm(
        self, prompt: str, max_tokens: int = 1000, retries: int = 3
    ) -> Dict[str, Any]:
        """
        Make an async LLM API call with retries.

        Args:
            prompt (str): Input prompt.
            max_tokens (int): Maximum tokens in response.
            retries (int): Number of retry attempts for transient errors.

        Returns:
            Dict[str, Any]: API response.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        formatted_prompt = f"[INST] {prompt} [/INST]"
        payload = {
            "inputs": formatted_prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": 0.7,
                "return_full_text": False,
                "top_p": 0.9,
            },
        }

        for attempt in range(retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.endpoint}/models/{self.model_id}",
                        json=payload,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=30),  # 30s timeout
                    ) as response:
                        if response.status == 200:
                            return await response.json()
                        elif response.status == 429:  # Rate limit
                            logger.warning(
                                f"Rate limit hit, retrying {attempt + 1}/{retries}"
                            )
                            await asyncio.sleep(2**attempt)
                            continue
                        else:
                            error_text = await response.text()
                            raise Exception(
                                f"LLM API error: {response.status} - {error_text}"
                            )
            except aiohttp.ClientConnectorDNSError as e:
                logger.error(f"DNS error: {str(e)}")
                if attempt == retries - 1:
                    raise Exception(
                        f"DNS resolution failed after {retries} attempts: {str(e)}"
                    )
                logger.warning(f"Retrying {attempt + 1}/{retries} after DNS error")
                await asyncio.sleep(2**attempt)
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == retries - 1:
                    raise Exception(f"Failed after {retries} attempts: {str(e)}")
                logger.warning(f"Retrying {attempt + 1}/{retries}")
                await asyncio.sleep(2**attempt)
        raise Exception("Max retries reached for LLM API call")
