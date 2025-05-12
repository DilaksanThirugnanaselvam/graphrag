import networkx as nx
from typing import List, Tuple
from collections import defaultdict

class GraphBuilder:
    """Builds and clusters a knowledge graph."""
    
    def __init__(self):
        """Initialize GraphBuilder with an empty graph."""
        self.graph = nx.Graph()
    
    def add_entities(self, entities: List[Tuple[str, str]]):
        """
        Add entities as nodes to the graph.
        
        Args:
            entities (List[Tuple[str, str]]): List of (entity, type) pairs.
        """
        for entity, entity_type in entities:
            self.graph.add_node(entity, type=entity_type)
    
    def add_relationships(self, relationships: List[Tuple[str, str, str]]):
        """
        Add relationships as edges to the graph.
        
        Args:
            relationships (List[Tuple[str, str, str]]): List of (entity1, entity2, relationship) triples.
        """
        for e1, e2, rel in relationships:
            self.graph.add_edge(e1, e2, relationship=rel)
    
    def cluster_communities(self) -> List[List[str]]:
        """
        Cluster graph into communities using a simplified Leiden-like approach.
        
        Returns:
            List[List[str]]: List of communities (lists of node names).
        """
        # Simplified clustering: group nodes by degree
        communities = defaultdict(list)
        for node in self.graph.nodes:
            degree = self.graph.degree[node]
            communities[degree // 5].append(node)  # Group by degree ranges
        return list(communities.values())
