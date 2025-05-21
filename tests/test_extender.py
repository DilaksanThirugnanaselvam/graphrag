import os
import sys
import tempfile
from unittest.mock import AsyncMock

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from graphrag_extender.db import Database
from graphrag_extender.extender import GraphExtender
from src.utils import load_config


@pytest.fixture
async def test_db():
    """Fixture to create an in-memory PostgreSQL database for testing."""
    config = load_config("../configs/settings.yaml")
    # Use in-memory database for isolation
    test_conn_string = "postgresql://postgres:admin@localhost:5432/test_graphrag"
    db = Database(test_conn_string)
    await db.initialize()

    # Create schema
    with open("../schema.sql", "r") as f:
        schema = f.read()
    async with db.pool.acquire() as conn:
        await conn.execute(schema)

    yield db
    await db.close()


@pytest.fixture
def temp_input_file():
    """Fixture to create a temporary input file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Rome is historic. Rome and Venice are connected. Venice is beautiful.")
        f.flush()
        yield f.name
    os.unlink(f.name)


@pytest.mark.asyncio
async def test_extend_graph(test_db, temp_input_file, monkeypatch):
    """Test graph extension with a sample document."""
    config = load_config("../configs/settings.yaml")
    config["paths"]["input_dir"] = os.path.dirname(temp_input_file)

    # Mock LLM client to avoid real API calls
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = [
        ("Rome", "city"),
        ("Venice", "city"),
    ]  # Mock entity extraction
    monkeypatch.setattr("src.llm_client.LLMClient.generate", mock_llm.generate)

    extender = GraphExtender(config)
    extender.db = test_db  # Use test database
    await extender.extend_graph(os.path.dirname(temp_input_file))

    nodes, edges = await test_db.load_graph()
    assert len(nodes) >= 2, "Expected at least 2 nodes (Rome, Venice)"
    assert len(edges) >= 1, "Expected at least 1 edge (Rome-Venice)"

    async with test_db.pool.acquire() as conn:
        result = await conn.fetchval("SELECT COUNT(*) FROM communities")
        assert result > 0, "No communities created"

        # Verify documents table
        doc_count = await conn.fetchval(
            "SELECT COUNT(*) FROM documents WHERE processed = TRUE"
        )
        assert doc_count == 1, "Document not marked as processed"


@pytest.mark.asyncio
async def test_leiden_communities(test_db, temp_input_file, monkeypatch):
    """Test Leiden community detection."""
    config = load_config("../configs/settings.yaml")
    config["paths"]["input_dir"] = os.path.dirname(temp_input_file)

    # Mock LLM client
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = [("Rome", "city"), ("Venice", "city")]
    monkeypatch.setattr("src.llm_client.LLMClient.generate", mock_llm.generate)

    extender = GraphExtender(config)
    extender.db = test_db
    await extender.extend_graph(os.path.dirname(temp_input_file))

    async with test_db.pool.acquire() as conn:
        communities = await conn.fetch("SELECT id, nodes FROM communities")
        assert len(communities) > 0, "No communities detected"
        for community in communities:
            assert isinstance(community["nodes"], list), (
                f"Community {community['id']} nodes not stored as list"
            )
            assert len(community["nodes"]) > 0, (
                f"Community {community['id']} has no nodes"
            )


@pytest.mark.asyncio
async def test_database_connection():
    """Test database connection."""
    config = load_config("../configs/settings.yaml")
    db = Database(config["db"]["conn_string"])
    await db.initialize()
    async with db.pool.acquire() as conn:
        result = await conn.fetchval("SELECT 1")
        assert result == 1, "Database connection failed"
    await db.close()


@pytest.mark.asyncio
async def test_edge_weights(test_db, temp_input_file, monkeypatch):
    """Test edge weights based on shared chunks."""
    config = load_config("../configs/settings.yaml")
    config["paths"]["input_dir"] = os.path.dirname(temp_input_file)

    # Mock LLM client
    mock_llm = AsyncMock()
    mock_llm.generate.return_value = [("Rome", "city"), ("Venice", "city")]
    monkeypatch.setattr("src.llm_client.LLMClient.generate", mock_llm.generate)

    extender = GraphExtender(config)
    extender.db = test_db
    await extender.extend_graph(os.path.dirname(temp_input_file))

    async with test_db.pool.acquire() as conn:
        edges = await conn.fetch(
            "SELECT source_id, target_id, weight FROM edges WHERE relationship = 'connected'"
        )
        for edge in edges:
            source_name = await conn.fetchval(
                "SELECT name FROM nodes WHERE id = $1", edge["source_id"]
            )
            target_name = await conn.fetchval(
                "SELECT name FROM nodes WHERE id = $1", edge["target_id"]
            )
            shared_chunks = await conn.fetchval(
                """
                SELECT COUNT(DISTINCT ce1.chunk_id)
                FROM chunk_entities ce1
                JOIN chunk_entities ce2 ON ce1.chunk_id = ce2.chunk_id
                WHERE ce1.entity_id = $1 AND ce2.entity_id = $2
                """,
                edge["source_id"],
                edge["target_id"],
            )
            assert edge["weight"] == shared_chunks, (
                f"Edge weight mismatch for {source_name} to {target_name}: "
                f"expected {shared_chunks}, got {edge['weight']}"
            )
