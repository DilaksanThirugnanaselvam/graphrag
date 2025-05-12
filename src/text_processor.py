import re
from typing import List

class TextProcessor:
    """Handles text chunking and preprocessing for GraphRAG."""
    
    def __init__(self, chunk_size: int = 300, overlap: int = 50):
        """
        Initialize TextProcessor.
        
        Args:
            chunk_size (int): Number of tokens per chunk.
            overlap (int): Number of tokens to overlap between chunks.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text (str): Input text.
            
        Returns:
            List[str]: List of text chunks.
        """
        tokens = self.tokenize(text)
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunks.append(' '.join(chunk_tokens))
            start += self.chunk_size - self.overlap
        
        return chunks
