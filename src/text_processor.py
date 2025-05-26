class TextProcessor:
    def __init__(self, chunk_size: int, overlap: int):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> list:
        """Split text into chunks with specified size and overlap."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = " ".join(words[i : i + self.chunk_size])
            if chunk.strip():  # Skip empty chunks
                chunks.append({"text": chunk})
        return chunks
