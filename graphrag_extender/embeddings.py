import logging
from typing import List

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class Embeddings:
    def __init__(self, config: dict):
        self.client = AsyncOpenAI(api_key=config["embeddings"]["api_key"])
        self.model = config["embeddings"].get(
            "embedding_model", "text-embedding-ada-002"
        )
        logger.info("Initialized Embeddings with OpenAI API")

    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        try:
            response = await self.client.embeddings.create(
                input=text, model=self.model, encoding_format="float"
            )
            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding for text: {text[:50]}...")
            return embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise
