import logging

import networkx as nx

logger = logging.getLogger(__name__)


class GraphBuilder:
    def __init__(self):
        self.graph = nx.Graph()

    def add_node(self, name: str, type_: str):
        """Add a node to the graph (legacy method)."""
        self.graph.add_node(name, type=type_)
        logger.info(f"Added node: {name} ({type_})")

    def add_edge(self, source: str, target: str, relationship: str, weight: float):
        """Add an edge to the graph (legacy method)."""
        self.graph.add_edge(source, target, relationship=relationship, weight=weight)
        logger.info(
            f"Added edge: {source} --{relationship}--> {target} (weight: {weight})"
        )

    def get_graph(self) -> nx.Graph:
        """Return the graph (legacy method)."""
        return self.graph
