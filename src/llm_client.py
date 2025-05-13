import asyncio
import logging
from typing import Optional

import aiohttp

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LLMClient:
    """Async client to interact with an LLM API (e.g., Grok API)."""

    def __init__(self, api_key: str, endpoint: str, model_id: str) -> None:
        """
        Initialize LLM client.

        Args:
            api_key (str): API key for authentication.
            endpoint (str): API endpoint URL (e.g., https://api.groq.com/openai/v1/chat/completions).
            model_id (str): Model identifier (e.g., llama3-70b-8192).

        Raises:
            ValueError: If api_key or endpoint is empty/invalid.
        """
        if not api_key or api_key.startswith("gsk_") is False:
            logger.warning("API key is missing or invalid. Using mock responses.")
            self.api_key = None
        else:
            self.api_key = api_key

        if not endpoint:
            raise ValueError("Endpoint URL cannot be empty.")

        self.endpoint = endpoint
        self.model_id = model_id
        self.headers = (
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            if self.api_key
            else {}
        )

    async def call_llm(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        mock_response: Optional[str] = None,
    ) -> Optional[str]:
        """
        Call the LLM API with a given prompt using the OpenAI-compatible chat format.

        Args:
            prompt (str): Prompt to send to the LLM.
            max_tokens (int): Maximum tokens in the response.
            temperature (float): Sampling temperature for creativity.
            mock_response (Optional[str]): Mock response to return if API call fails (for testing).

        Returns:
            Optional[str]: Response from the LLM or mock_response if failed.
        """
        # If no valid API key, return mock response immediately
        if not self.api_key:
            logger.info("No valid API key provided. Returning mock response.")
            return (
                mock_response
                if mock_response is not None
                else "Mock LLM response: API key invalid."
            )

        # Construct payload in OpenAI-compatible chat format
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        for attempt in range(3):  # Retry up to 3 times
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.endpoint, json=payload, headers=self.headers
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            # Extract response from OpenAI-compatible format
                            return (
                                result.get("choices", [{}])[0]
                                .get("message", {})
                                .get("content", "")
                                .strip()
                            )
                        elif response.status == 403:
                            logger.error(
                                "API request failed with status 403: Invalid API key or insufficient permissions."
                            )
                            return mock_response if mock_response is not None else None
                        else:
                            logger.error(
                                f"API request failed with status {response.status}: {await response.text()}"
                            )
            except aiohttp.ClientError as e:
                logger.error(f"Request failed: {str(e)}")
                await asyncio.sleep(2**attempt)  # Exponential backoff
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                break

        logger.error("Max retries reached, returning mock response or None")
        return mock_response if mock_response is not None else None
