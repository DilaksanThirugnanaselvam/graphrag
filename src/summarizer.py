from typing import List

import networkx as nx  # <-- Add this line

from llm_client import LLMClient


class Summarizer:
    """Generates summaries for graph communities."""

    def __init__(self, llm_client: LLMClient):
        """
        Initialize Summarizer.

        Args:
            llm_client (LLMClient): Async LLM client.
        """
        self.llm_client = llm_client

    async def summarize_community(self, community: List[str], graph: nx.Graph) -> str:
        """
        Generate a summary for a community.

        Args:
            community (List[str]): List of entity names in the community.
            graph (nx.Graph): Knowledge graph.

        Returns:
            str: Community summary.
        """
        entities = ", ".join(community)
        prompt = f"""
        Summarize the relationships and entities in the following community.
        Entities: {entities}
        Provide a concise summary of their interactions.
        """
        response = await self.llm_client.call_llm(prompt)
        return f"Summary: {entities} are interconnected through various relationships."
