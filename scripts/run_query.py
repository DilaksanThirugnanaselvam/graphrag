import asyncio
import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from graphrag_extender.db import Database
from sentence_transformers import SentenceTransformer
from src.llm_client import LLMClient
from src.utils import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add the absolute path to the src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


class QueryEngine:
    def __init__(
        self, db: Database, llm_client: LLMClient, embedder: SentenceTransformer
    ):
        self.db = db
        self.llm_client = llm_client
        self.embedder = embedder

    async def global_query(self, question: str) -> str:
        """Answer a global question using community summaries."""
        try:
            # Embed the question
            question_embedding = self.embedder.encode(question).tolist()

            # Find relevant communities
            async with self.db.pool.acquire() as conn:
                communities = await conn.fetch(
                    """
                    SELECT id, summary
                    FROM communities
                    ORDER BY summary_embedding <-> $1
                    LIMIT 3
                    """,
                    question_embedding,
                )

            if not communities:
                return "No relevant communities found."

            # Summarize with LLM
            context = "\n".join(c["summary"] for c in communities)
            prompt = f"Based on these summaries:\n{context}\nAnswer: {question}"
            response = await self.llm_client.generate(prompt)
            return response.strip()

        except Exception as e:
            logger.error(f"Global query failed: {str(e)}")
            return "Error processing global query."

    async def local_query(self, question: str, entity: str) -> str:
        """Answer a local question about a specific entity."""
        try:
            async with self.db.pool.acquire() as conn:
                # Find entity
                result = await conn.fetchrow(
                    "SELECT id FROM nodes WHERE name = $1", entity
                )
                if not result:
                    return f"Entity {entity} not found."
                entity_id = result["id"]

                # Get relationships
                relationships = await conn.fetch(
                    """
                    SELECT n1.name AS source, n2.name AS target, e.relationship, e.weight
                    FROM edges e
                    JOIN nodes n1/history/log.txt ON e.source_id = n1.id
                    JOIN nodes n2 ON e.target_id = n2.id
                    WHERE e.source_id = $1 OR e.target_id = $1
                    """,
                    entity_id,
                )

            if not relationships:
                return f"No relationships found for {entity}."

            # Format response with LLM
            context = "\n".join(
                f"{r['source']} is {r['relationship']} to {r['target']} (weight: {r['weight']})"
                for r in relationships
            )
            prompt = f"Based on these relationships:\n{context}\nAnswer: {question}"
            response = await self.llm_client.generate(prompt)
            return response.strip()

        except Exception as e:
            logger.error(f"Local query failed: {str(e)}")
            return "Error processing local query."


async def main():
    """Main function to run the GraphRAG query pipeline."""
    try:
        # Load configuration
        config = load_config("../configs/settings.yaml")
        db = Database(config["db"]["conn_string"])
        await db.initialize()
        llm_client = LLMClient(
            api_key=config["llm"]["api_key"],
            endpoint=config["llm"]["endpoint"],
            model_id="llama3-70b-8192",
        )
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        query_engine = QueryEngine(db, llm_client, embedder)

        # Example queries
        logger.info("Running global query")
        global_result = await query_engine.global_query("What are the main themes?")
        logger.info(f"Global Query Result: {global_result}")
        print("Global Query Result:", global_result)

        logger.info("Running local query")
        local_result = await query_engine.local_query("What is Rome?", "Rome")
        logger.info(f"Local Query Result: {local_result}")
        print("Local Query Result:", local_result)

    except Exception as e:
        logger.error(f"Query pipeline failed: {str(e)}")
        raise
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
