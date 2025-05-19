import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import networkx as nx
from db import Database
from embeddings import Embedder
from entity_extractor import EntityExtractor
from llm_client import LLMClient
from summarizer import Summarizer
from text_processor import TextProcessor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GraphExtender:
    def __init__(self, config: dict):
        self.db = Database(config["db"]["conn_string"])
        self.llm_client = LLMClient(
            api_key=config["llm"]["api_key"],
            endpoint=config["llm"]["endpoint"],
            model_id="llama3-70b-8192",
        )
        self.processor = TextProcessor(
            chunk_size=config["chunking"]["chunk_size"],
            overlap=config["chunking"]["overlap"],
        )
        self.extractor = EntityExtractor(self.llm_client)
        self.summarizer = Summarizer(self.llm_client)
        self.embedder = Embedder()

    async def extend_graph(self, document_path: str) -> None:
        """Add nodes, edges, and communities from a new document."""
        try:
            # Read new document
            with open(document_path, "r", encoding="utf-8") as f:
                text = f.read()
            logger.info(f"Processing new document: {document_path}")

            # Chunk text
            chunks = self.processor.chunk_text(text)
            logger.info(f"Generated {len(chunks)} chunks")

            # Process chunks
            relationships = []
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i + 1}/{len(chunks)}")
                try:
                    entities = await self.extractor.extract_entities(chunk)
                    chunk_id = self.db.add_chunk(chunk, self.embedder.embed(chunk))
                    for name, type_ in entities:
                        entity_id = self.db.add_node(name, type_)
                        self.db.link_chunk_entity(chunk_id, entity_id)
                    relationships.extend(
                        await self.extractor.extract_relationships(chunk, entities)
                    )
                except Exception as e:
                    logger.error(f"Error processing chunk {i + 1}: {str(e)}")
                    continue

            # Aggregate and add relationships
            logger.info("Aggregating relationships")
            rel_counts = {}
            for e1, e2, rel in relationships:
                key = (e1, e2, rel)
                rel_counts[key] = rel_counts.get(key, 0) + 1

            for (e1, e2, rel), count in rel_counts.items():
                source_id = self.db.add_node(e1, "Unknown")
                target_id = self.db.add_node(e2, "Unknown")
                self.db.add_edge(source_id, target_id, rel, float(count))

            # Rebuild graph for community detection
            logger.info("Rebuilding graph for community detection")
            nodes, edges = self.db.load_graph()
            graph = nx.Graph()
            for node in nodes:
                graph.add_node(node["name"], type=node["type"])
            for edge in edges:
                graph.add_edge(
                    next(n["name"] for n in nodes if n["id"] == edge["source_id"]),
                    next(n["name"] for n in nodes if n["id"] == edge["target_id"]),
                    relationship=edge["relationship"],
                    weight=edge["weight"],
                )

            # Cluster communities
            logger.info("Clustering communities")
            communities = []
            if graph.number_of_nodes() > 0:
                from networkx.algorithms.community import greedy_modularity_communities

                communities = list(greedy_modularity_communities(graph))
            logger.info(f"Found {len(communities)} communities")

            # Summarize communities
            for i, community in enumerate(communities):
                logger.info(f"Summarizing community {i + 1}/{len(communities)}")
                try:
                    community_nodes = list(community)
                    summary = await self.summarizer.summarize_community(
                        community_nodes, graph
                    )
                    summary_embedding = self.embedder.embed(summary)
                    self.db.add_community(
                        i, community_nodes, summary, summary_embedding
                    )
                    logger.info(f"Saved community {i} summary")
                except Exception as e:
                    logger.error(f"Error summarizing community {i + 1}: {str(e)}")
                    continue

            logger.info("Graph updated successfully")

        except Exception as e:
            logger.error(f"Failed to extend graph: {str(e)}")
            raise
