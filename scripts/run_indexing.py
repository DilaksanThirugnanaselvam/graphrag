import asyncio
import logging
import os
import sys
from typing import Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add the absolute path to the src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from entity_extractor import EntityExtractor
from graph_builder import GraphBuilder
from llm_client import LLMClient
from summarizer import Summarizer
from text_processor import TextProcessor
from utils import load_config, save_to_file


async def main() -> None:
    """Main function to run the GraphRAG indexing pipeline."""
    try:
        # Load configuration
        logger.info("Loading configuration")
        config = load_config("configs/settings.yaml")

        # Validate paths
        input_path = os.path.join(config["paths"]["input_dir"], "sample.txt")
        output_dir = config["paths"]["output_dir"]
        if not os.path.exists(input_path):
            logger.error(f"Input file not found: {input_path}")
            raise FileNotFoundError(f"Input file not found: {input_path}")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Input file: {input_path}, Output dir: {output_dir}")

        # Initialize components
        llm_client = LLMClient(
            api_key=config["llm"]["api_key"],
            endpoint=config["llm"]["endpoint"],
            model_id="llama3-70b-8192",
        )
        processor = TextProcessor(
            chunk_size=config["chunking"]["chunk_size"],
            overlap=config["chunking"]["overlap"],
        )
        extractor = EntityExtractor(llm_client)
        graph_builder = GraphBuilder()
        summarizer = Summarizer(llm_client)
        logger.info("Components initialized")

        # Read input text
        logger.info("Reading input text")
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Process text
        logger.info("Chunking text")
        chunks = processor.chunk_text(text)
        logger.info(f"Generated {len(chunks)} chunks")

        # Extract entities and relationships
        all_relationships = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i + 1}/{len(chunks)}")
            try:
                entities = await extractor.extract_entities(chunk)
                logger.info(f"Extracted entities: {entities}")
                relationships = await extractor.extract_relationships(chunk, entities)
                logger.info(f"Extracted relationships: {relationships}")
                graph_builder.add_entities(entities)
                all_relationships.extend(relationships)
            except Exception as e:
                logger.error(f"Error processing chunk {i + 1}: {str(e)}")
                continue

        # Aggregate relationships and add to graph
        logger.info("Aggregating relationships")
        weighted_relationships = graph_builder.aggregate_relationships(
            all_relationships
        )
        logger.info(f"Weighted relationships: {weighted_relationships}")
        graph_builder.add_relationships(weighted_relationships)

        # Cluster communities
        logger.info("Clustering communities")
        communities = graph_builder.cluster_communities()
        logger.info(f"Found {len(communities)} communities")

        # Generate summaries
        summaries: Dict[int, str] = {}
        for i, community in enumerate(communities):
            logger.info(f"Summarizing community {i + 1}/{len(communities)}")
            try:
                summary = await summarizer.summarize_community(
                    community, graph_builder.graph
                )
                summaries[i] = summary
                output_file = os.path.join(output_dir, f"summary_{i}.txt")
                save_to_file(summary, output_file)
                logger.info(f"Saved summary to {output_file}")
            except Exception as e:
                logger.error(f"Error summarizing community {i + 1}: {str(e)}")
                continue

        # Save graph
        graph_output = str(graph_builder.graph.edges(data=True))
        graph_file = os.path.join(output_dir, "graph.txt")
        save_to_file(graph_output, graph_file)
        logger.info(f"Saved graph to {graph_file}")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
