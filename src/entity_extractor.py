from typing import List, Tuple
from llm_client import LLMClient

class EntityExtractor:
    """Extracts entities and relationships from text using LLM."""
    
    def __init__(self, llm_client: LLMClient):
        """
        Initialize EntityExtractor.
        
        Args:
            llm_client (LLMClient): Async LLM client.
        """
        self.llm_client = llm_client
    
    async def extract_entities(self, chunk: str) -> List[Tuple[str, str]]:
        """
        Extract entities from a text chunk.
        
        Args:
            chunk (str): Text chunk.
            
        Returns:
            List[Tuple[str, str]]: List of (entity, type) pairs.
        """
        prompt = f"""
        Extract entities (e.g., people, organizations, locations) from the following text.
        Return as a JSON list of [entity, type] pairs.
        Text: {chunk}
        """
        response = await self.llm_client.call_llm(prompt)
        # Mock response parsing (replace with actual API response handling)
        return [["Alice", "Person"], ["Wonderland", "Location"]]
    
    async def extract_relationships(self, chunk: str, entities: List[Tuple[str, str]]) -> List[Tuple[str, str, str]]:
        """
        Extract relationships between entities.
        
        Args:
            chunk (str): Text chunk.
            entities (List[Tuple[str, str]]): List of entities.
            
        Returns:
            List[Tuple[str, str, str]]: List of (entity1, entity2, relationship) triples.
        """
        prompt = f"""
        Extract relationships between entities in the following text.
        Entities: {entities}
        Return as a JSON list of [entity1, entity2, relationship] triples.
        Text: {chunk}
        """
        response = await self.llm_client.call_llm(prompt)
        # Mock response
        return [["Alice", "Wonderland", "visited"]]
