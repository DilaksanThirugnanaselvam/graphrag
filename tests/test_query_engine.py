import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import networkx as nx

from llm_client import LLMClient
from query_engine import QueryEngine


@pytest.mark.asyncio
async def test_global_query():
    llm_client = LLMClient(api_key="test")
    graph = nx.Graph()
    summaries = {0: "Mock summary"}
    query_engine = QueryEngine(llm_client, graph, summaries)
    result = await query_engine.global_query("What are the themes?")
    assert "Mock summary" in result
