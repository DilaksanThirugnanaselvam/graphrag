import asyncio
import logging
import networkx as nx
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.text_processor import TextProcessor
from src.entity_extractor import EntityExtractor
from src.llm_client import LLMClient
from src.summarizer import Summarizer
from graphrag_extender.db import Database
from graphrag_extender.embeddings import Embedder
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
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

    async def extend_graph(self, input_dir: str) -> None:
        try:
            await self.db.initialize()
            if not os.path.exists(input_dir):
                logger.error(f"Input directory not found: {input_dir}")
                raise FileNotFoundError(f"Input directory not found: {input_dir}")

            # Register new documents
            for file_name in os.listdir(input_dir):
                if file_name.endswith(".txt"):
                    file_path = os.path.join(input_dir, file_name)
                    await self.db.add_document(file_path)
                    logger.info(f"Registered document: {file_path}")

            # Process unprocessed documents
            documents = await self.db.get_unprocessed_documents()
            if not documents:
                logger.info("No unprocessed documents found")
                return

            entity_to_id = {}
            async with self.db.pool.acquire() as conn:
                existing_nodes = await conn.fetch("SELECT id, name FROM nodes")
                for node in existing_nodes:
                    entity_to_id[node["name"]] = node["id"]

            for document_id, document_path in documents:
                logger.info(f"Processing document: {document_path}")
                try:
                    with open(document_path, "r", encoding="utf-8") as f:
                        text = f.read().strip()
                    if not text:
                        logger.warning(f"Empty document: {document_path}")
                        await self.db.mark_document_processed(document_id)
                        continue

                    chunks = self.processor.chunk_text(text)
                    if not chunks:
                        logger.warning(f"No chunks generated: {document_path}")
                        await self.db.mark_document_processed(document_id)
                        continue

                    chunk_entity_pairs = []
                    for i, chunk in enumerate(chunks):
                        logger.info(f"Processing chunk {i + 1}/{len(chunks)} in {document_path}")
                        try:
                            entities = await self.extractor.extract_entities(chunk)
                            chunk_id = await self.db.add_chunk(chunk, self.embedder.embed(chunk), document_id)
                            chunk_entities = []
                            for name, type_ in entities:
                                if name not in entity_to_id:
                                    entity_id = await self.db.add_node(name, type_)
                                    entity_to_id[name] = entity_id
                                await self.db.link_chunk_entity(chunk_id, entity_to_id[name])
                                chunk_entities.append(name)
                            chunk_entity_pairs.append((chunk_id, chunk_entities))
                        except Exception as e:
                            logger.error(f"Error processing chunk {i + 1}: {str(e)}")
                            continue

                    logger.info(f"Calculating edge weights for {document_path}")
                    edge_weights = defaultdict(int)
                    for _, entities in chunk_entity_pairs:
                        for i, e1 in enumerate(entities):
                            for e2 in entities[i + 1:]:
                                if e1 != e2:
                                    edge_weights[(e1, e2)] += 1
                                    edge_weights[(e2, e1)] += 1

                    for (e1, e2), weight in edge_weights.items():
                        source_id = entity_to_id[e1]
                        target_id = entity_to_id[e2]
                        await self.db.add_edge(source_id, target_id, "connected", float(weight))

                    await self.db.mark_document_processed(document_id)
                    logger.info(f"Completed processing: {document_path}")

            logger.info("Updating communities")
            nodes, edges = await self.db.load_graph()
            graph = nx.Graph()
            for node in nodes:
                graph.add_node(node["name"], type=node["type"])
            for edge in edges:
                graph.add_edge(
                    next(n["name"] for n in nodes if n["id"] == edge["source_id"]),
                    next(n["name"] for n in nodes if n["id"] == edge["target_id"]),
                    relationship=edge["relationship"],
                    weight=edge["weight"]
                )

            logger.info("Clustering communities using Leiden algorithm")
            communities = []
            if graph.number_of_nodes() > 0:
                try:
                    from networkx.algorithms.community.leiden import leiden_communities
                    communities = list(leiden_communities(graph, weight="weight", resolution=1.0, n_iterations=10))
                except ImportError as e:
                    logger.error(f"Leiden algorithm not available: {str(e)}")
                    raise ImportError("Please install leidenalg and igraph: pip install leidenalg igraph")
            logger.info(f"Found {len(communities)} communities")

            for i, community in enumerate(communities):
                logger.info(f"Summarizing community {i + 1}/{len(communities)}")
                try:
                    community_nodes = list(community)
                    summary = await self.summarizer.summarize_community(community_nodes, graph)
                    summary_embedding = self.embedder.embed(summary)
                    await self.db.add_community(i, community_nodes, summary, summary_embedding)
                    logger.info(f"Saved community {i} summary")
                except Exception as e:
                    logger.error(f"Error summarizing community {i + 1}: {str(e)}")
                    continue

            logger.info("Graph updated successfully")

        except Exception as e:
            logger.error(f"Failed to extend graph: {str(e)}")
            raise
        finally:
            await self.db.close()