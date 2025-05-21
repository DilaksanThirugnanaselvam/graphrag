import logging
from typing import List, Tuple

import asyncpg

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, conn_string: str):
        self.conn_string = conn_string
        self.pool = None

    async def initialize(self):
        self.pool = await asyncpg.create_pool(self.conn_string)
        logger.info("Database pool initialized")

    async def add_document(self, path: str) -> int:
        async with self.pool.acquire() as conn:
            document_id = await conn.fetchval(
                "INSERT INTO documents (path, processed) VALUES ($1, FALSE) "
                "ON CONFLICT (path) DO UPDATE SET processed = FALSE "
                "RETURNING id",
                path,
            )
        return document_id

    async def mark_document_processed(self, document_id: int):
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE documents SET processed = TRUE, last_updated = CURRENT_TIMESTAMP WHERE id = $1",
                document_id,
            )

    async def get_unprocessed_documents(self) -> List[Tuple[int, str]]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT id, path FROM documents WHERE processed = FALSE"
            )
        return [(row["id"], row["path"]) for row in rows]

    async def add_chunk(self, text: str, embedding: list, document_id: int) -> int:
        async with self.pool.acquire() as conn:
            chunk_id = await conn.fetchval(
                "INSERT INTO chunks (text, document_id, embedding) VALUES ($1, $2, $3) RETURNING id",
                text,
                document_id,
                embedding,
            )
        return chunk_id

    async def add_node(self, name: str, type_: str) -> int:
        async with self.pool.acquire() as conn:
            node_id = await conn.fetchval(
                "INSERT INTO nodes (name, type) VALUES ($1, $2) "
                "ON CONFLICT (name) DO UPDATE SET type = EXCLUDED.type "
                "RETURNING id",
                name,
                type_,
            )
        return node_id

    async def add_edge(
        self, source_id: int, target_id: int, relationship: str, weight: float
    ):
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO edges (source_id, target_id, relationship, weight) "
                "VALUES ($1, $2, $3, $4) "
                "ON CONFLICT (source_id, target_id, relationship) "
                "DO UPDATE SET weight = EXCLUDED.weight",
                source_id,
                target_id,
                relationship,
                weight,
            )

    async def link_chunk_entity(self, chunk_id: int, entity_id: int):
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO chunk_entities (chunk_id, entity_id) VALUES ($1, $2) "
                "ON CONFLICT DO NOTHING",
                chunk_id,
                entity_id,
            )

    async def add_community(
        self, community_id: int, nodes: list, summary: str, embedding: list
    ):
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO communities (id, nodes, summary, summary_embedding) "
                "VALUES ($1, $2, $3, $4) "
                "ON CONFLICT (id) DO UPDATE SET nodes = EXCLUDED.nodes, "
                "summary = EXCLUDED.summary, summary_embedding = EXCLUDED.summary_embedding",
                community_id,
                nodes,
                summary,
                embedding,
            )

    async def load_graph(self) -> Tuple[List[dict], List[dict]]:
        async with self.pool.acquire() as conn:
            nodes = await conn.fetch("SELECT id, name, type FROM nodes")
            edges = await conn.fetch(
                "SELECT source_id, target_id, relationship, weight FROM edges"
            )
        return nodes, edges

    async def close(self):
        if self.pool:
            await self.pool.close()
            logger.info("Database pool closed")
