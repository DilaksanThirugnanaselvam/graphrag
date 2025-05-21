import asyncio
import logging
import os
import sys
import traceback

import asyncpg

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from graphrag_extender.extender import GraphExtender
from src.utils import load_config


async def validate_schema(conn_string: str):
    """Validate that required database tables and pgvector extension exist."""
    pool = None
    try:
        pool = await asyncpg.create_pool(conn_string)
        async with pool.acquire() as conn:
            tables = await conn.fetch(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            )
            table_names = {t["table_name"] for t in tables}
            required = {
                "nodes",
                "edges",
                "chunks",
                "chunk_entities",
                "communities",
                "documents",
            }
            if not required.issubset(table_names):
                missing = required - table_names
                logger.error(f"Missing database tables: {missing}. Run schema.sql.")
                raise ValueError(f"Missing database tables: {missing}")
            ext = await conn.fetchval(
                "SELECT extname FROM pg_extension WHERE extname = 'vector'"
            )
            if not ext:
                logger.error(
                    "pgvector extension not installed. Run: CREATE EXTENSION vector;"
                )
                raise ValueError("pgvector extension not installed")
        logger.info("Database schema validated successfully")
    except Exception as e:
        logger.error(f"Schema validation failed: {str(e)}\n{traceback.format_exc()}")
        raise
    finally:
        if pool:
            await pool.close()


async def main() -> None:
    try:
        logger.info("Loading configuration")
        config = load_config("../configs/settings.yaml")

        logger.info("Validating database schema")
        await validate_schema(config["db"]["conn_string"])

        input_dir = config["paths"]["input_dir"]
        if not os.path.exists(input_dir):
            logger.error(f"Input directory not found: {input_dir}")
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        logger.info(f"Input directory: {input_dir}")

        extender = GraphExtender(config)
        await extender.extend_graph(input_dir)
        logger.info("Graph update complete")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}\n{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
