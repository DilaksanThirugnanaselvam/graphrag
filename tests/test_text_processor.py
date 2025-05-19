import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from text_processor import TextProcessor


def test_chunk_text():
    processor = TextProcessor(chunk_size=3, overlap=1)
    text = "This is a test sentence to chunk"
    chunks = processor.chunk_text(text)

    expected = [
        "This is a",
        "is a test",
        "a test sentence",
        "test sentence to",
        "sentence to chunk",
    ]
    assert len(chunks) == 5
    assert chunks == expected


def test_chunk_text_no_overlap():
    processor = TextProcessor(chunk_size=3, overlap=0)
    text = "This is a test sentence"
    chunks = processor.chunk_text(text)

    expected = ["This is a", "test sentence"]
    assert len(chunks) == 2
    assert chunks == expected
