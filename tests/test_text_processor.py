import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from text_processor import TextProcessor


def test_chunk_text():
    processor = TextProcessor(chunk_size=5, overlap=2)
    text = "This is a sample text to test chunking"
    chunks = processor.chunk_text(text)
    assert len(chunks) > 0
    assert all(len(processor.tokenize(chunk)) <= 5 for chunk in chunks)
