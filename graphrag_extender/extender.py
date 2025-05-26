import logging
import os
import sys

import networkx as nx

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.entity_extractor import EntityExtractor
from src.text_chunker import TextChunker

from graphrag_extender.db import Database
from graphrag_extender.embeddings import Embeddings

logger = logging.getLogger(__name__)


class GraphExtender:
    def __init__(self, config: dict):
        self.config = config
        self.db = Database(config["db"]["conn_string"])
        self.chunker = TextChunker(config)
        self.extractor = EntityExtractor(config)
        self.embeddings = Embeddings(config)
        self.chunk_size = config.get("chunk_size", 512)

    async def initialize(self):
        await self.db.initialize()
        logger.info("GraphExtender initialized")

    async def extend_graph(self, input_dir: str):
        logger.info("Starting to process documents")
        for filename in os.listdir(input_dir):
            file_path = os.path.join(input_dir, filename)
            if os.path.isfile(file_path):
                doc_id = await self.db.add_document(file_path)
                logger.info(f"Added document to database: {file_path}")
                await self.process_document(file_path, doc_id)
                await self.db.mark_document_processed(doc_id)
        await self.update_communities()

    async def process_document(self, file_path: str, doc_id: int):
        logger.info(f"Processing document: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            logger.debug(f"Document preview: {text[:100]}")

            chunks = self.chunker.chunk_text(text)
            logger.info(f"Total chunks created: {len(chunks)}")

            for i, chunk in enumerate(chunks, 1):
                logger.info(f"Processing chunk {i}/{len(chunks)} in {file_path}")
                await self.process_chunk(chunk, doc_id)

            await self.calculate_edge_weights(doc_id)
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise

    async def process_chunk(self, chunk: str, doc_id: int):
        logger.debug(f"Chunk text: {chunk[:100]}...")
        try:
            embedding = await self.embeddings.generate_embedding(chunk)
            chunk_id = await self.db.add_chunk(chunk, embedding, doc_id)
            entities = await self.extractor.extract_entities(chunk)
            logger.debug(f"Extracted entities: {entities}")
            for entity in entities:
                entity_id = await self.db.add_node(entity["name"], entity["type"])
                await self.db.link_chunk_entity(chunk_id, entity_id)
        except Exception as e:
            logger.error(f"Failed to process chunk: {str(e)}")
            raise

    async def calculate_edge_weights(self, doc_id: int):
        logger.info(f"Calculating edge weights for document ID: {doc_id}")
        try:
            entities = await self.db.get_document_entities(doc_id)
            logger.debug(f"Entities for doc {doc_id}: {entities}")
            for i, (source_id, source_name) in enumerate(entities):
                for target_id, target_name in entities[i + 1 :]:
                    shared_chunks = await self.db.get_shared_chunks(
                        source_id, target_id
                    )
                    if shared_chunks > 0:
                        weight = shared_chunks / 2.0
                        await self.db.add_edge(source_id, target_id, "related", weight)
                        await self.db.add_edge(target_id, source_id, "related", weight)
            logger.info(f"Edge weights calculated for document ID: {doc_id}")
        except Exception as e:
            logger.error(f"Error calculating edge weights: {str(e)}")
            raise

    async def update_communities(self):
        logger.info("Updating communities")
        try:
            nodes, edges = await self.db.load_graph()
            logger.debug(f"Loaded {len(nodes)} nodes and {len(edges)} edges")
            if not nodes:
                logger.info("No nodes found for community detection")
                return

            G = nx.Graph()
            node_map = {node["id"]: i for i, node in enumerate(nodes)}
            for node in nodes:
                G.add_node(node_map[node["id"]], name=node["name"])
            for edge in edges:
                G.add_edge(
                    node_map[edge["source_id"]],
                    node_map[edge["target_id"]],
                    weight=edge["weight"],
                )

            communities = []
            if not edges:
                logger.info("No edges found, creating single community")
                communities = [[i for i in range(len(nodes))]]
            else:
                communities = list(
                    nx.algorithms.community.greedy_modularity_communities(
                        G, weight="weight"
                    )
                )
                logger.debug(f"Detected {len(communities)} communities")

            if not communities:
                logger.info("No communities detected, creating default community")
                communities = [[i for i in range(len(nodes))]]

            async with self.db.pool.acquire() as conn:
                max_id = await conn.fetchval(
                    "SELECT COALESCE(MAX(id), -1) FROM communities"
                )
                start_id = max_id + 1

                for idx, community in enumerate(communities, start=start_id):
                    community_nodes = [nodes[node]["id"] for node in community]
                    node_names = [nodes[node]["name"] for node in community]
                    summary = f"Community {idx} with nodes: {', '.join(node_names)}"
                    summary_embedding = await self.embeddings.generate_embedding(
                        summary
                    )

                    # Check if community with same nodes exists
                    existing_id = await conn.fetchval(
                        "SELECT id FROM communities WHERE nodes = $1", community_nodes
                    )
                    if existing_id is not None:
                        await conn.execute(
                            "UPDATE communities SET summary = $1, summary_embedding = $2 WHERE id = $3",
                            summary,
                            summary_embedding,
                            existing_id,
                        )
                        logger.debug(
                            f"Updated community {existing_id} with {len(community_nodes)} nodes"
                        )
                    else:
                        await self.db.add_community(
                            idx, community_nodes, summary, summary_embedding
                        )
                        logger.debug(
                            f"Added community {idx} with {len(community_nodes)} nodes"
                        )
            logger.info("Communities updated successfully")
        except Exception as e:
            logger.error(f"Error updating communities: {str(e)}")
            raise
