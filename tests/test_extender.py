import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from graphrag_extender.extender import GraphExtender
from src.utils import load_config


@pytest.mark.asyncio
async def test_extend_graph():
    config = load_config("../configs/settings.yaml")
    extender = GraphExtender(config)
    document_path = "../data/input/sample.txt"
    await extender.extend_graph(document_path)
    nodes, edges = extender.db.load_graph()
    assert len(nodes) > 0, "No nodes created"
    assert len(edges) > 0, "No edges created"
    with extender.db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM communities")
            assert cur.fetchone()[0] > 0, "No communities created"


def test_database_connection():
    config = load_config("../configs/settings.yaml")
    db = GraphExtender(config).db
    with db.get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            assert cur.fetchone()[0] == 1, "Database connection failed"
