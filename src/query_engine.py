import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from graphrag_extender.db import Database
from graphrag_extender.embeddings import Embeddings
from llm_client import LLMClient

logger = logging.getLogger(__name__)


class QueryEngine:
    def __init__(self, db: Database, llm_client: LLMClient, embedder: Embeddings):
        self.db = db
        self.llm_client = llm_client
        self.embedder = embedder

    async def global_query(self, question: str) -> str:
        """Answer a global question using community summaries."""
        try:
            question_embedding = await self.embedder.generate_embedding(question)
            with self.db.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, summary
                        FROM communities
                        ORDER BY summary_embedding <-> %s
                        LIMIT 3
                        """,
                        (question_embedding,),
                    )
                    communities = cur.fetchall()

            if not communities:
                return "No relevant communities found."

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
            with self.db.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT id FROM nodes WHERE name = %s", (entity,))
                    result = cur.fetchone()
                    if not result:
                        return f"Entity {entity} not found."
                    entity_id = result["id"]

                    cur.execute(
                        """
                        SELECT n1.name AS source, n2.name AS target, e.relationship, e.weight
                        FROM edges e
                        JOIN nodes n1 ON e.source_id = n1.id
                        JOIN nodes n2 ON e.target_id = n2.id
                        WHERE e.source_id = %s OR e.target_id = %s
                        """,
                        (entity_id, entity_id),
                    )
                    relationships = cur.fetchall()

            if not relationships:
                return f"No relationships found for {entity}."

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
