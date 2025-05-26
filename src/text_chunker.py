import logging
from typing import List

logger = logging.getLogger(__name__)


class TextChunker:
    def __init__(self, config: dict):
        self.chunk_size = config.get("chunk_size", 512)
        self.overlap = config.get("chunk_overlap", 50)

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks based on character length."""
        try:
            chunks = []
            text_length = len(text)
            start = 0

            while start < text_length:
                end = min(start + self.chunk_size, text_length)
                chunk = text[start:end]
                chunks.append(chunk)
                start += self.chunk_size - self.overlap

            logger.debug(
                f"Created {len(chunks)} chunks from text of length {text_length}"
            )
            return chunks
        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}")
            return [text]
