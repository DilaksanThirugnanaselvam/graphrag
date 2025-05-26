from dataclasses import dataclass
from typing import List

import asyncpg


@dataclass
class Node:
    id: int
    name: str
    type: str

    @classmethod
    async def create(cls, conn: asyncpg.Connection, name: str, type: str) -> "Node":
        record = await conn.fetchrow(
            "INSERT INTO nodes (name, type) VALUES ($1, $2) ON CONFLICT (name) DO UPDATE SET type = $2 RETURNING id, name, type",
            name,
            type,
        )
        return cls(**record)


@dataclass
class Edge:
    source_id: int
    target_id: int
    type: str
    weight: float

    @classmethod
    async def create(
        cls,
        conn: asyncpg.Connection,
        source_id: int,
        target_id: int,
        type: str,
        weight: float,
    ) -> "Edge":
        await conn.execute(
            "INSERT INTO edges (source_id, target_id, type, weight) VALUES ($1, $2, $3, $4) ON CONFLICT DO NOTHING",
            source_id,
            target_id,
            type,
            weight,
        )
        return cls(source_id, target_id, type, weight)


@dataclass
class Chunk:
    id: int
    content: str
    embedding: List[float]
    document_id: int

    @classmethod
    async def create(
        cls,
        conn: asyncpg.Connection,
        content: str,
        embedding: List[float],
        document_id: int,
    ) -> "Chunk":
        record = await conn.fetchrow(
            "INSERT INTO chunks (content, embedding, document_id) VALUES ($1, $2, $3) RETURNING id, content, embedding, document_id",
            content,
            embedding,
            document_id,
        )
        return cls(**record)


@dataclass
class ChunkEntity:
    chunk_id: int
    entity_id: int

    @classmethod
    async def create(
        cls, conn: asyncpg.Connection, chunk_id: int, entity_id: int
    ) -> "ChunkEntity":
        await conn.execute(
            "INSERT INTO chunk_entities (chunk_id, entity_id) VALUES ($1, $2) ON CONFLICT DO NOTHING",
            chunk_id,
            entity_id,
        )
        return cls(chunk_id, entity_id)


@dataclass
class Community:
    id: int
    nodes: List[int]
    summary: str
    summary_embedding: List[float]

    @classmethod
    async def create_or_update(
        cls,
        conn: asyncpg.Connection,
        nodes: List[int],
        summary: str,
        summary_embedding: List[float],
    ) -> "Community":
        existing_id = await conn.fetchval(
            "SELECT id FROM communities WHERE nodes = $1", nodes
        )
        if existing_id is not None:
            await conn.execute(
                "UPDATE communities SET summary = $1, summary_embedding = $2 WHERE id = $3",
                summary,
                summary_embedding,
                existing_id,
            )
            return cls(existing_id, nodes, summary, summary_embedding)
        else:
            max_id = await conn.fetchval(
                "SELECT COALESCE(MAX(id), -1) FROM communities"
            )
            new_id = max_id + 1
            await conn.execute(
                "INSERT INTO communities (id, nodes, summary, summary_embedding) VALUES ($1, $2, $3, $4)",
                new_id,
                nodes,
                summary,
                summary_embedding,
            )
            return cls(new_id, nodes, summary, summary_embedding)


@dataclass
class Document:
    id: int
    file_path: str
    processed: bool = False

    @classmethod
    async def create(cls, conn: asyncpg.Connection, file_path: str) -> "Document":
        record = await conn.fetchrow(
            "INSERT INTO documents (file_path) VALUES ($1) RETURNING id, file_path, processed",
            file_path,
        )
        return cls(**record)

    async def mark_processed(self, conn: asyncpg.Connection):
        await conn.execute(
            "UPDATE documents SET processed = TRUE WHERE id = $1", self.id
        )
        self.processed = True
