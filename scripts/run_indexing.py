import asyncio
import logging
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add path to graphrag/ so that graphrag_extender and src are importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from graphrag_extender.extender import GraphExtender
from src.utils import load_config


async def main() -> None:
    """Main function to run the GraphRAG indexing pipeline."""
    try:
        # Load configuration
        logger.info("Loading configuration")
        config = load_config("graphrag/configs/settings.yaml")

        # Validate input path
        input_path = os.path.join(config["paths"]["input_dir"], "new_document.txt")
        if not os.path.exists(input_path):
            logger.error(f"Input file not found: {input_path}")
            raise FileNotFoundError(f"Input file not found: {input_path}")
        logger.info(f"Input file: {input_path}")

        # Initialize and run extender
        extender = GraphExtender(config)
        await extender.extend_graph(input_path)
        logger.info("Graph update complete")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
