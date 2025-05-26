import asyncio
import logging
import os
import sys

from tenacity import retry, stop_after_attempt, wait_exponential

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from graphrag_extender.db import Database
from graphrag_extender.embeddings import Embeddings
from src.llm_client import LLMClient
from src.utils import load_config

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class QueryEngine:
    def __init__(self, db: Database, llm_client: LLMClient, embeddings: Embeddings):
        self.db = db
        self.llm_client = llm_client
        self.embeddings = embeddings

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def generate_embedding(self, text: str) -> list:
        try:
            embedding = await self.embeddings.generate_embedding(text)
            if not isinstance(embedding, list) or not all(
                isinstance(x, float) for x in embedding
            ):
                logger.error(f"Invalid embedding format: {embedding}")
                raise ValueError("Invalid embedding format")
            return embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise

    async def global_query(self, question: str) -> str:
        try:
            question_embedding = await self.generate_embedding(question)
            embedding_str = f"[{', '.join(map(str, question_embedding))}]"
            async with self.db.pool.acquire() as conn:
                communities = await conn.fetch(
                    """
                    SELECT id, summary
                    FROM communities
                    ORDER BY summary_embedding <=> $1::vector
                    LIMIT 3
                    """,
                    embedding_str,
                )

            if not communities:
                return "No relevant communities found."

            context = "\n".join(c["summary"] for c in communities)
            prompt = f"Based on these summaries:\n{context}\nAnswer: {question}"
            response = await self.llm_client.generate(prompt)
            return response.strip() if response else "No response generated."

        except Exception as e:
            logger.error(f"Global query failed: {str(e)}")
            return "Error processing global query."

    async def local_query(self, question: str, entity: str) -> str:
        try:
            async with self.db.pool.acquire() as conn:
                result = await conn.fetchrow(
                    "SELECT id FROM nodes WHERE name = $1", entity
                )
                if not result:
                    return f"Entity {entity} not found."
                entity_id = result["id"]

                relationships = await conn.fetch(
                    """
                    SELECT n1.name AS source, n2.name AS target, e.relationship, e.weight
                    FROM edges e
                    JOIN nodes n1 ON e.source_id = n1.id
                    JOIN nodes n2 ON e.target_id = n2.id
                    WHERE e.source_id = $1 OR e.target_id = $1
                    """,
                    entity_id,
                )

            if not relationships:
                return f"No relationships found for {entity}."

            context = "\n".join(
                f"{r['source']} is {r['relationship']} to {r['target']} (weight: {r['weight']})"
                for r in relationships
            )
            prompt = f"Based on these relationships:\n{context}\nAnswer: {question}"
            response = await self.llm_client.generate(prompt)
            return response.strip() if response else "No response generated."

        except Exception as e:
            logger.error(f"Local query failed: {str(e)}")
            return "Error processing local query."


async def main():
    try:
        config = load_config("configs/settings.yaml")
        db = Database(config["db"]["conn_string"])
        await db.initialize()
        llm_client = LLMClient(
            api_key=config["llm"]["api_key"],
            endpoint=config["llm"]["endpoint"],
            model_id=config["llm"]["model_id"],
        )
        embeddings = Embeddings(config)
        query_engine = QueryEngine(db, llm_client, embeddings)

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
