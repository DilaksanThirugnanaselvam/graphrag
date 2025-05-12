from typing import List, Dict
from llm_client import LLMClient
import networkx as nx
from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm

class QueryEngine:
    """Handles global and local queries using the knowledge graph."""
    
    def __init__(self, llm_client: LLMClient, graph: nx.Graph, summaries: Dict[int, str]):
        """
        Initialize QueryEngine.
        
        Args:
            llm_client (LLMClient): Async LLM client.
            graph (nx.Graph): Knowledge graph.
            summaries (Dict[int, str]): Community ID to summary mapping.
        """
        self.llm_client = llm_client
        self.graph = graph
        self.summaries = summaries
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        return dot(a, b) / (norm(a) * norm(b))
    
    async def global_query(self, query: str) -> str:
        """
        Handle a global query using community summaries.
        
        Args:
            query (str): User query.
            
        Returns:
            str: Answer based on summaries.
        """
        query_embedding = self.embedder.encode(query)
        best_summary = ""
        best_score = -1
        
        for summary in self.summaries.values():
            summary_embedding = self.embedder.encode(summary)
            score = self.cosine_similarity(query_embedding, summary_embedding)
            if score > best_score:
                best_score = score
                best_summary = summary
        
        prompt = f"""
        Using the following summary, answer the query: {query}
        Summary: {best_summary}
        """
        response = await self.llm_client.call_llm(prompt)
        # Mock response
        return f"Answer to '{query}': {best_summary}"
    
    async def local_query(self, query: str, entity: str) -> str:
        """
        Handle a local query focused on a specific entity.
        
        Args:
            query (str): User query.
            entity (str): Target entity.
            
        Returns:
            str: Answer based on entity's relationships.
        """
        if entity not in self.graph:
            return f"Entity {entity} not found."
        
        neighbors = list(self.graph.neighbors(entity))
        relationships = [self.graph[entity][n]['relationship'] for n in neighbors]
        context = f"Entity: {entity}, Relationships: {list(zip(neighbors, relationships))}"
        
        prompt = f"""
        Using the following context, answer the query: {query}
        Context: {context}
        """
        response = await self.llm_client.call_llm(prompt)
        # Mock response
        return f"Answer to '{query}' about {entity}: {context}"
