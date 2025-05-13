from typing import List

import networkx as nx

from llm_client import LLMClient


class Summarizer:
    """Generates summaries for communities in the graph using an LLM."""

    def __init__(self, llm_client: LLMClient) -> None:
        """
        Initialize Summarizer.

        Args:
            llm_client (LLMClient): Async LLM client.
        """
        self.llm_client = llm_client

    async def summarize_community(self, community: List[str], graph: nx.Graph) -> str:
        """
        Summarize a community using the LLM.

        Args:
            community (List[str]): List of entity names in the community.
            graph (nx.Graph): Knowledge graph.

        Returns:
            str: Summary of the community.
        """
        relationships = []
        for u, v, data in graph.edges(data=True):
            if u in community and v in community:
                relationships.append((u, v, data.get("relationship", "related to")))

        prompt = f"""
        Summarize the following community of entities and their relationships:
        Entities: {community}
        Relationships: {relationships}
        Provide a concise summary.
        """
        mock_response = f"Summary: {', '.join(community)} are interconnected through various relationships."
        response = await self.llm_client.call_llm(prompt, mock_response=mock_response)
        return response
