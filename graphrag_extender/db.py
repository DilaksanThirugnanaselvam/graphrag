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
            doc_id = await conn.fetchval(
                "INSERT INTO documents (path) VALUES ($1) RETURNING id", path
            )
        return doc_id

    async def get_unprocessed_documents(self) -> List[Tuple[int, str]]:
        async with self.pool.acquire() as conn:
            return await conn.fetch(
                "SELECT id, path FROM documents WHERE processed = FALSE"
            )

    async def add_chunk(
        self, text: str, embedding: List[float], document_id: int
    ) -> int:
        embedding_str = f"[{', '.join(map(str, embedding))}]"
        async with self.pool.acquire() as conn:
            chunk_id = await conn.fetchval(
                "INSERT INTO chunks (text, embedding, document_id) VALUES ($1, $2::vector, $3) RETURNING id",
                text,
                embedding_str,
                document_id,
            )
        return chunk_id

    async def add_node(self, name: str, type: str) -> int:
        async with self.pool.acquire() as conn:
            node_id = await conn.fetchval(
                "INSERT INTO nodes (name, type) VALUES ($1, $2) ON CONFLICT (name) DO UPDATE SET type = EXCLUDED.type RETURNING id",
                name,
                type,
            )
        return node_id

    async def link_chunk_entity(self, chunk_id: int, entity_id: int):
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO chunk_entities (chunk_id, entity_id) VALUES ($1, $2) ON CONFLICT DO NOTHING",
                chunk_id,
                entity_id,
            )

    async def add_edge(
        self, source_id: int, target_id: int, relationship: str, weight: float
    ):
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO edges (source_id, target_id, relationship, weight) VALUES ($1, $2, $3, $4) ON CONFLICT DO NOTHING",
                source_id,
                target_id,
                relationship,
                weight,
            )

    async def get_document_entities(self, doc_id: int) -> List[Tuple[int, str]]:
        async with self.pool.acquire() as conn:
            return await conn.fetch(
                """
                SELECT DISTINCT n.id, n.name
                FROM nodes n
                JOIN chunk_entities ce ON n.id = ce.entity_id
                JOIN chunks c ON ce.chunk_id = c.id
                WHERE c.document_id = $1
                """,
                doc_id,
            )

    async def get_shared_chunks(self, source_id: int, target_id: int) -> int:
        async with self.pool.acquire() as conn:
            count = await conn.fetchval(
                """
                SELECT COUNT(DISTINCT ce1.chunk_id)
                FROM chunk_entities ce1
                JOIN chunk_entities ce2 ON ce1.chunk_id = ce2.chunk_id
                WHERE ce1.entity_id = $1 AND ce2.entity_id = $2
                """,
                source_id,
                target_id,
            )
        return count or 0

    async def mark_document_processed(self, doc_id: int):
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE documents SET processed = TRUE WHERE id = $1", doc_id
            )

    async def load_graph(self) -> Tuple[List[dict], List[dict]]:
        async with self.pool.acquire() as conn:
            nodes = await conn.fetch("SELECT id, name FROM nodes")
            edges = await conn.fetch(
                "SELECT source_id, target_id, weight FROM edges WHERE weight IS NOT NULL"
            )
        return nodes, edges

    async def add_community(
        self,
        comm_id: int,
        nodes: List[int],
        summary: str,
        summary_embedding: List[float],
    ):
        summary_embedding_str = f"[{', '.join(map(str, summary_embedding))}]"
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO communities (id, nodes, summary, summary_embedding) VALUES ($1, $2, $3, $4::vector)",
                comm_id,
                nodes,
                summary,
                summary_embedding_str,
            )

    async def close(self):
        if self.pool:
            await self.pool.close()
            logger.info("Database pool closed")
