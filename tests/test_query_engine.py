import os
import sys

import pytest
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from graphrag_extender.db import Database
from src.llm_client import LLMClient
from src.query_engine import QueryEngine


@pytest.mark.asyncio
async def test_global_query():
    config = {
        "db": {"conn_string": "postgresql://postgres@localhost:5433/graphrag"},
        "llm": {
            "api_key": "test",
            "endpoint": "https://api.groq.com/openai/v1/chat/completions",
            "model_id": "llama3-70b-8192",
        },
    }
    llm_client = LLMClient(
        api_key="test",
        endpoint="https://api.groq.com/openai/v1/chat/completions",
        model_id="llama3-70b-8192",
    )
    db = Database(config["db"]["conn_string"])
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Insert mock community summary
    mock_summary = "Mock summary about locations"
    summary_embedding = embedder.encode(mock_summary).tolist()
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO communities (id, nodes, summary, summary_embedding) VALUES (%s, %s, %s, %s)",
                (0, ["Rome", "Paris"], mock_summary, summary_embedding),
            )
            conn.commit()

    query_engine = QueryEngine(db, llm_client, embedder)
    result = await query_engine.global_query("What are the themes?")
    assert "Mock summary" in result
