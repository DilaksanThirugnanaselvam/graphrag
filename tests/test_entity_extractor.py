import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from entity_extractor import EntityExtractor
from llm_client import LLMClient


@pytest.mark.asyncio
async def test_extract_entities():
    llm_client = LLMClient(api_key="test")
    extractor = EntityExtractor(llm_client)
    entities = await extractor.extract_entities("Alice went to Wonderland")
    assert len(entities) == 2
    assert entities[0] == ["Alice", "Person"]
