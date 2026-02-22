"""Vector index for memory embeddings (pgvector)."""
from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from cortex.memory.schema import Memory, memory_from_row


class VectorIndex:
    """Vector similarity search over memory embeddings using pgvector."""

    def __init__(self, db_connection=None, get_connection=None):
        """
        db_connection: optional sync connection (e.g. psycopg2).
        get_connection: optional callable that returns a connection (for context managers).
        """
        self._conn = db_connection
        self._get_conn = get_connection

    def _conn_or_get(self):
        if self._conn is not None:
            return self._conn
        if self._get_conn is not None:
            return self._get_conn()
        raise RuntimeError("VectorIndex: no db_connection or get_connection provided")

    def search(
        self,
        query_embedding: List[float],
        user_id: Optional[UUID] = None,
        k: int = 50,
        type_filter: Optional[str] = None,
    ) -> List[tuple[Memory, float]]:
        """
        Return top-k memories by cosine similarity.
        Returns list of (Memory, score) where score is similarity (higher = more similar).
        """
        conn = self._conn_or_get()
        cur = conn.cursor()
        try:
            # pgvector: <=> is cosine distance; 1 - distance = similarity
            # Cast %s to vector so Postgres gets vector type (not numeric[])
            q = """
                SELECT id, user_id, type, summary, raw_text, embedding, importance, emotion,
                       created_at, last_used, usage_count, mvn_score, entities, source,
                       1 - (embedding <=> %s::vector) AS score
                FROM memories
                WHERE embedding IS NOT NULL
            """
            params: list = [query_embedding]
            if user_id is not None:
                q += " AND user_id = %s"
                params.append(str(user_id))
            if type_filter is not None:
                q += " AND type = %s"
                params.append(type_filter)
            q += " ORDER BY embedding <=> %s::vector LIMIT %s"
            params.append(query_embedding)
            params.append(k)
            cur.execute(q, params)
            rows = cur.fetchall()
            colnames = [d[0] for d in cur.description]
            results = []
            for row in rows:
                r = dict(zip(colnames, row))
                score = r.pop("score", 0.0)
                results.append((memory_from_row(r), float(score)))
            return results
        finally:
            cur.close()

    def add(self, memory_id: UUID, embedding: List[float]) -> None:
        """Update embedding for an existing memory row."""
        conn = self._conn_or_get()
        cur = conn.cursor()
        try:
            cur.execute(
                "UPDATE memories SET embedding = %s::vector WHERE id = %s",
                (embedding, str(memory_id)),
            )
            conn.commit()
        finally:
            cur.close()
