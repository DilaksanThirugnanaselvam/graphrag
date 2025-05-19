import aiohttp


class LLMClient:
    def __init__(
        self,
        api_key: str,
        endpoint: str = "graphrag/configs/settings.yaml",
        model_id: str = "llama3-70b-8192",
    ):
        self.api_key = api_key
        self.endpoint = endpoint
        self.model_id = model_id

    async def generate(self, prompt: str) -> str:
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            data = {"model": self.model_id, "prompt": prompt}
            async with session.post(
                self.endpoint, json=data, headers=headers
            ) as response:
                result = await response.json()
                return result.get("text", f"Mock response for: {prompt}")
