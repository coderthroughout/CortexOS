"""Memory store: add, get, update, delete. Uses Postgres."""
from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from cortex.memory.schema import Memory, MemoryCreate, MemoryType, memory_from_row


class MemoryStore:
    """CRUD for atomic memories in Postgres."""

    def __init__(self, db_connection=None, get_connection=None):
        self._conn = db_connection
        self._get_conn = get_connection

    def _conn_or_get(self):
        if self._conn is not None:
            return self._conn
        if self._get_conn is not None:
            return self._get_conn()
        raise RuntimeError("MemoryStore: no db_connection or get_connection provided")

    def add_memory(self, mem: Memory | MemoryCreate, embedding: Optional[List[float]] = None) -> Memory:
        """Insert one memory. If MemoryCreate, returns created Memory with id."""
        conn = self._conn_or_get()
        cur = conn.cursor()
        try:
            if isinstance(mem, MemoryCreate):
                m = Memory(
                    **mem.model_dump(),
                    embedding=embedding,
                    raw_text=mem.text,
                )
            else:
                m = mem
                if embedding is not None:
                    m.embedding = embedding
            row = m.to_db_row()
            cur.execute(
                """
                INSERT INTO memories (id, user_id, type, summary, raw_text, embedding, importance, emotion, created_at, last_used, usage_count, mvn_score, entities, source)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    summary = EXCLUDED.summary,
                    raw_text = EXCLUDED.raw_text,
                    embedding = COALESCE(EXCLUDED.embedding, memories.embedding),
                    importance = EXCLUDED.importance,
                    emotion = EXCLUDED.emotion,
                    last_used = EXCLUDED.last_used,
                    usage_count = EXCLUDED.usage_count,
                    mvn_score = EXCLUDED.mvn_score,
                    entities = EXCLUDED.entities,
                    source = EXCLUDED.source
                """,
                (
                    str(row["id"]),
                    str(row["user_id"]),
                    row["type"],
                    row["summary"],
                    row["raw_text"],
                    row["embedding"],
                    row["importance"],
                    row["emotion"],
                    row["created_at"],
                    row["last_used"],
                    row["usage_count"],
                    row["mvn_score"],
                    row["entities"],
                    row["source"],
                ),
            )
            conn.commit()
            return m
        finally:
            cur.close()

    def get_memory(self, memory_id: UUID) -> Optional[Memory]:
        """Fetch a single memory by id."""
        conn = self._conn_or_get()
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT id, user_id, type, summary, raw_text, embedding, importance, emotion, created_at, last_used, usage_count, mvn_score, entities, source FROM memories WHERE id = %s",
                (str(memory_id),),
            )
            row = cur.fetchone()
            if not row:
                return None
            colnames = [d[0] for d in cur.description]
            return memory_from_row(dict(zip(colnames, row)))
        finally:
            cur.close()

    def get_user_memories(
        self,
        user_id: UUID,
        limit: int = 1000,
        type_filter: Optional[MemoryType] = None,
    ) -> List[Memory]:
        """Return memories for a user, optionally filtered by type."""
        conn = self._conn_or_get()
        cur = conn.cursor()
        try:
            q = "SELECT id, user_id, type, summary, raw_text, embedding, importance, emotion, created_at, last_used, usage_count, mvn_score, entities, source FROM memories WHERE user_id = %s"
            params: list = [str(user_id)]
            if type_filter is not None:
                q += " AND type = %s"
                params.append(type_filter.value)
            q += " ORDER BY created_at DESC LIMIT %s"
            params.append(limit)
            cur.execute(q, params)
            rows = cur.fetchall()
            colnames = [d[0] for d in cur.description]
            return [memory_from_row(dict(zip(colnames, r))) for r in rows]
        finally:
            cur.close()

    def update_usage(self, memory_id: UUID) -> None:
        """Increment usage_count and set last_used to now (for feedback/reflection)."""
        conn = self._conn_or_get()
        cur = conn.cursor()
        try:
            cur.execute(
                "UPDATE memories SET usage_count = usage_count + 1, last_used = NOW() WHERE id = %s",
                (str(memory_id),),
            )
            conn.commit()
        finally:
            cur.close()

    def update(self, memory_id: UUID, summary: Optional[str] = None, importance: Optional[float] = None) -> bool:
        """PATCH: update summary and/or importance. Returns True if updated."""
        conn = self._conn_or_get()
        cur = conn.cursor()
        try:
            if summary is not None:
                cur.execute("UPDATE memories SET summary = %s WHERE id = %s", (summary, str(memory_id)))
            if importance is not None:
                cur.execute("UPDATE memories SET importance = %s WHERE id = %s", (importance, str(memory_id)))
            conn.commit()
            return cur.rowcount > 0
        finally:
            cur.close()

    def delete(self, memory_id: UUID) -> bool:
        """Delete a memory. Returns True if a row was deleted."""
        conn = self._conn_or_get()
        cur = conn.cursor()
        try:
            cur.execute("DELETE FROM memories WHERE id = %s", (str(memory_id),))
            conn.commit()
            return cur.rowcount > 0
        finally:
            cur.close()
