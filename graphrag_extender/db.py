import psycopg
from psycopg.rows import dict_row


class Database:
    def __init__(self, conn_string: str):
        self.conn_string = conn_string

    def get_conn(self):
        return psycopg.connect(self.conn_string, row_factory=dict_row)

    def add_node(self, name: str, type_: str) -> int:
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO nodes (name, type) VALUES (%s, %s) ON CONFLICT (name) DO NOTHING RETURNING id",
                    (name, type_),
                )
                result = cur.fetchone()
                if result:
                    return result["id"]
                cur.execute("SELECT id FROM nodes WHERE name = %s", (name,))
                return cur.fetchone()["id"]

    def add_edge(
        self, source_id: int, target_id: int, relationship: str, weight: float
    ):
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO edges (source_id, target_id, relationship, weight)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (source_id, target_id, relationship)
                    DO UPDATE SET weight = edges.weight + EXCLUDED.weight
                    """,
                    (source_id, target_id, relationship, weight),
                )

    def add_chunk(self, text: str, embedding: list) -> int:
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO chunks (text, embedding) VALUES (%s, %s) RETURNING id",
                    (text, embedding),
                )
                return cur.fetchone()["id"]

    def link_chunk_entity(self, chunk_id: int, entity_id: int):
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO chunk_entities (chunk_id, entity_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    (chunk_id, entity_id),
                )

    def add_community(
        self, community_id: int, nodes: list, summary: str, embedding: list
    ):
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO communities (id, nodes, summary, summary_embedding)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        nodes = EXCLUDED.nodes,
                        summary = EXCLUDED.summary,
                        summary_embedding = EXCLUDED.summary_embedding
                    """,
                    (community_id, nodes, summary, embedding),
                )

    def load_graph(self) -> tuple:
        with self.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id, name, type FROM nodes")
                nodes = cur.fetchall()
                cur.execute(
                    "SELECT source_id, target_id, relationship, weight FROM edges"
                )
                edges = cur.fetchall()
                return nodes, edges
