import asyncio
import logging
import os
import socket
import sys
import traceback

import asyncpg
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from graphrag_extender.extender import GraphExtender
from src.utils import load_config


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
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


async def clear_communities(conn_string: str):
    """Clear the communities table to avoid duplicate key errors."""
    pool = None
    try:
        pool = await asyncpg.create_pool(conn_string)
        async with pool.acquire() as conn:
            await conn.execute("TRUNCATE communities RESTART IDENTITY")
            logger.info("Cleared communities table")
    except Exception as e:
        logger.error(f"Failed to clear communities table: {str(e)}")
        raise
    finally:
        if pool:
            await pool.close()


async def main() -> None:
    try:
        logger.info("Loading configuration")
        config = load_config("configs/settings.yaml")

        host = os.getenv("POSTGRES_HOST", "postgres")
        conn_string = config["db"]["conn_string"].replace("${POSTGRES_HOST}", host)

        if not os.path.exists("/.dockerenv"):
            conn_string = config["db"]["conn_string"].replace(
                "${POSTGRES_HOST}", "localhost"
            )
            logger.info("Running locally, using localhost for PostgreSQL")
        else:
            try:
                container_ip = socket.gethostbyname("postgres")
                logger.info(
                    f"Running in Docker, resolved postgres to IP: {container_ip}"
                )
                conn_string = config["db"]["conn_string"].replace(
                    "${POSTGRES_HOST}", "postgres"
                )
            except socket.gaierror:
                logger.warning("Failed to resolve 'postgres', using fallback IP")
                container_ip = "172.19.0.2"
                conn_string = config["db"]["conn_string"].replace(
                    "${POSTGRES_HOST}", container_ip
                )
                logger.info(f"Using fallback IP: {container_ip}")

        logger.info("Validating database schema")
        await validate_schema(conn_string)

        logger.info("Clearing communities table")
        await clear_communities(conn_string)

        input_dir = config["paths"]["input_dir"]
        if not os.path.exists(input_dir):
            logger.error(f"Input directory not found: {input_dir}")
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        logger.info(f"Input directory: {input_dir}")

        extender = GraphExtender(config)
        await extender.initialize()
        await extender.extend_graph(input_dir)
        logger.info("Graph update complete")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}\n{traceback.format_exc()}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
