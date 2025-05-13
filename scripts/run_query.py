import asyncio
import logging
import os
import sys

import networkx as nx

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add the absolute path to the src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from llm_client import LLMClient
from query_engine import QueryEngine
from utils import load_config


async def main():
    """Main function to run the GraphRAG query pipeline."""
    try:
        # Load configuration
        logger.info("Loading configuration")
        config = load_config("configs/settings.yaml")

        # Validate output directory
        output_dir = config["paths"]["output_dir"]
        if not os.path.exists(output_dir):
            logger.error(f"Output directory not found: {output_dir}")
            raise FileNotFoundError(f"Output directory not found: {output_dir}")

        # Initialize components
        logger.info("Initializing LLMClient")
        llm_client = LLMClient(
            api_key=config["llm"]["api_key"],
            endpoint=config["llm"]["endpoint"],
            model_id="llama3-70b-8192",
        )

        # Load graph
        graph_file = os.path.join(output_dir, "graph.txt")
        graph = nx.Graph()
        if os.path.exists(graph_file):
            logger.info(f"Loading graph from {graph_file}")
            with open(graph_file, "r", encoding="utf-8") as f:
                edges = eval(f.read())  # Assuming edges are stored as stringified list
                graph.add_edges_from(edges)
        else:
            logger.warning(f"Graph file not found: {graph_file}, using empty graph")

        # Load summaries
        summaries = {}
        for i in range(10):  # Assume up to 10 summaries
            summary_file = os.path.join(output_dir, f"summary_{i}.txt")
            if os.path.exists(summary_file):
                logger.info(f"Loading summary from {summary_file}")
                with open(summary_file, "r", encoding="utf-8") as f:
                    summaries[i] = f.read().strip()
            else:
                break
        if not summaries:
            logger.warning("No summaries found, using mock summary")
            summaries = {
                0: "Mock summary: Alice, Wonderland are interconnected through various relationships."
            }

        # Initialize QueryEngine with correct order: (llm_client, graph, summaries)
        logger.info("Initializing QueryEngine")
        query_engine = QueryEngine(
            graph=graph, llm_client=llm_client, summaries=summaries
        )

        # Example queries
        logger.info("Running global query")
        try:
            global_result = await query_engine.global_query("What are the main themes?")
            logger.info(f"Global Query Result: {global_result}")
            print("Global Query Result:", global_result)
        except Exception as e:
            logger.error(f"Global query failed: {str(e)}")

        logger.info("Running local query")
        try:
            local_result = await query_engine.local_query("Who is Alice?", "Alice")
            logger.info(f"Local Query Result: {local_result}")
            print("Local Query Result:", local_result)
        except Exception as e:
            logger.error(f"Local query failed: {str(e)}")

    except Exception as e:
        logger.error(f"Query pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
