from collections import defaultdict
from typing import List, Tuple

import igraph as ig
import leidenalg as la
import networkx as nx


class GraphBuilder:
    """Builds and clusters a knowledge graph."""

    def __init__(self) -> None:
        """Initialize GraphBuilder with an empty graph."""
        self.graph = nx.Graph()

    def add_entities(self, entities: List[Tuple[str, str]]) -> None:
        """
        Add entities as nodes to the graph.

        Args:
            entities (List[Tuple[str, str]]): List of (entity, type) pairs.
        """
        for entity, entity_type in entities:
            self.graph.add_node(entity, type=entity_type)

    def add_relationships(
        self, relationships: List[Tuple[str, str, str, float]]
    ) -> None:
        """
        Add relationships as weighted edges to the graph.

        Args:
            relationships (List[Tuple[str, str, str, float]]): List of (entity1, entity2, relationship, weight) quadruples.
        """
        for e1, e2, rel, weight in relationships:
            self.graph.add_edge(e1, e2, relationship=rel, weight=weight)

    def cluster_communities(self) -> List[List[str]]:
        """
        Cluster graph into communities using the Leiden algorithm.

        Returns:
            List[List[str]]: List of communities (lists of node names).
        """
        # Check if the graph has edges
        if not self.graph.edges:
            return [
                [node] for node in self.graph.nodes
            ]  # Each node is its own community

        # Convert networkx graph to igraph for Leiden
        g = ig.Graph.from_networkx(self.graph)

        # Run Leiden algorithm with weights
        partition = la.find_partition(g, la.ModularityVertexPartition, weights="weight")

        # Extract communities
        communities = [list(g.vs[community]["_nx_name"]) for community in partition]
        return communities

    def aggregate_relationships(
        self, relationships: List[Tuple[str, str, str]]
    ) -> List[Tuple[str, str, str, float]]:
        """
        Aggregate relationships and assign weights based on frequency.

        Args:
            relationships (List[Tuple[str, str, str]]): List of (entity1, entity2, relationship) triples.

        Returns:
            List[Tuple[str, str, str, float]]: List of (entity1, entity2, relationship, weight) quadruples.
        """
        # Count frequency of each relationship
        rel_counts = defaultdict(int)
        for e1, e2, rel in relationships:
            key = (e1, e2, rel)
            rel_counts[key] += 1

        # Convert to weighted relationships
        weighted_rels = [
            (e1, e2, rel, float(count)) for (e1, e2, rel), count in rel_counts.items()
        ]
        return weighted_rels
