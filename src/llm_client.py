import aiohttp
import asyncio
from typing import Dict, Any

class LLMClient:
    """Async client for LLM API calls."""
    
    def __init__(self, api_key: str, endpoint: str = "https://api.example.com/v1"):
        """
        Initialize LLMClient.
        
        Args:
            api_key (str): API key for authentication.
            endpoint (str): API endpoint URL.
        """
        self.api_key = api_key
        self.endpoint = endpoint
    
    async def call_llm(self, prompt: str, max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Make an async LLM API call.
        
        Args:
            prompt (str): Input prompt.
            max_tokens (int): Maximum tokens in response.
            
        Returns:
            Dict[str, Any]: API response.
        """
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.endpoint}/completions",
                json=payload,
                headers=headers
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"LLM API error: {response.status}")
