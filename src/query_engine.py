import logging
from typing import Dict

import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class QueryEngine:
    """Handles global and local queries on the knowledge graph using embeddings and LLM."""

    def __init__(self, llm_client, graph: nx.Graph, summaries: Dict[int, str]) -> None:
        """
        Initialize QueryEngine.

        Args:
            llm_client: LLM client for generating answers.
            graph (nx.Graph): Knowledge graph.
            summaries (Dict[int, str]): Community summaries.
        """
        self.llm_client = llm_client
        self.graph = graph
        self.summaries = summaries
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Generate embeddings for summaries
        self.summary_texts = (
            [text for _, text in sorted(self.summaries.items())]
            if self.summaries
            else []
        )
        self.summary_embeddings = (
            self.model.encode(self.summary_texts)
            if self.summary_texts
            else np.array([])
        )

    async def global_query(self, query: str) -> str:
        """
        Answer a global query using community summaries and embeddings.

        Args:
            query (str): The query string.

        Returns:
            str: Answer to the query.
        """
        logger.info("Running global query")
        if not self.summary_texts:
            return "No summaries available to answer the query."

        # Encode the query
        query_embedding = self.model.encode(query)

        # Compute similarity with summaries
        similarities = np.dot(self.summary_embeddings, query_embedding) / (
            np.linalg.norm(self.summary_embeddings, axis=1)
            * np.linalg.norm(query_embedding)
        )

        # Find the most relevant summary
        most_relevant_idx = int(np.argmax(similarities))
        most_relevant_summary = self.summary_texts[most_relevant_idx]

        # Use LLM to generate a refined answer
        prompt = f"""
        Based on the following summary, answer the query.
        Summary: {most_relevant_summary}
        Query: {query}
        """
        mock_response = f"Answer to '{query}': {most_relevant_summary}"
        response = await self.llm_client.call_llm(prompt, mock_response=mock_response)
        return response

    async def local_query(self, query: str, entity: str) -> str:
        """
        Answer a local query focused on a specific entity.

        Args:
            query (str): The query string.
            entity (str): The target entity.

        Returns:
            str: Answer to the query.
        """
        logger.info("Running local query")
        if entity not in self.graph.nodes:
            return f"Entity '{entity}' not found in the graph."

        # Extract relationships involving the entity
        relationships = [
            (entity, neighbor, data.get("relationship", "related to"))
            for neighbor, data in self.graph[entity].items()
        ]

        # Use LLM to generate an answer
        prompt = f"""
        Based on the following entity and its relationships, answer the query.
        Entity: {entity}
        Relationships: {relationships}
        Query: {query}
        """
        mock_response = f"Answer to '{query}' about {entity}: Entity: {entity}, Relationships: {relationships}"
        response = await self.llm_client.call_llm(prompt, mock_response=mock_response)
        return response
