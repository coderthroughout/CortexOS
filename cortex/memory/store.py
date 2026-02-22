"""Memory store: add, get, update, delete. Uses Postgres."""
from __future__ import annotations

import json
from typing import Dict, List, Optional
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

    def get_all_memory_summaries(self, limit: int = 50000) -> List[tuple[str, str]]:
        """Return (id, summary) for all memories, for BM25 index build. summary may be empty."""
        conn = self._conn_or_get()
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT id, COALESCE(summary, raw_text, '') FROM memories ORDER BY created_at DESC LIMIT %s",
                (limit,),
            )
            return [(str(r[0]), str(r[1] or "")) for r in cur.fetchall()]
        finally:
            cur.close()

    def get_user_ids(self, limit: int = 100) -> List[UUID]:
        """Distinct user_id from memories (for background consolidation)."""
        conn = self._conn_or_get()
        cur = conn.cursor()
        try:
            cur.execute("SELECT DISTINCT user_id FROM memories LIMIT %s", (limit,))
            return [UUID(r[0]) for r in cur.fetchall() if r and r[0]]
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

    def append_feedback_log(
        self,
        user_id: Optional[UUID],
        query: Optional[str],
        retrieved_memory_ids: List[str],
        used_memory_ids: List[str],
        reward: float,
    ) -> None:
        """Append one row to feedback_logs for MVN training."""
        conn = self._conn_or_get()
        cur = conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO feedback_logs (user_id, query, retrieved_memory_ids, used_memory_ids, reward)
                VALUES (%s, %s, %s::jsonb, %s::jsonb, %s)
                """,
                (
                    str(user_id) if user_id else None,
                    query or None,
                    json.dumps(list(retrieved_memory_ids)),
                    json.dumps(list(used_memory_ids)),
                    reward,
                ),
            )
            conn.commit()
        finally:
            cur.close()

    def get_feedback_logs(self, limit: int = 5000) -> List[dict]:
        """Read feedback_logs for MVN training. Returns list of {query, retrieved_memory_ids, used_memory_ids, reward}."""
        conn = self._conn_or_get()
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT query, retrieved_memory_ids, used_memory_ids, reward FROM feedback_logs ORDER BY created_at DESC LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()
            out = []
            for r in rows:
                query, ret, used, reward = r
                ret_ids = ret if isinstance(ret, list) else (json.loads(ret) if isinstance(ret, str) else [])
                used_ids = used if isinstance(used, list) else (json.loads(used) if isinstance(used, str) else [])
                # Normalize to str so UUID() and set membership work (e.g. if driver returns UUID from JSONB)
                ret_ids = [str(x) for x in ret_ids if x is not None]
                used_ids = [str(x) for x in used_ids if x is not None]
                out.append({
                    "query": query or "",
                    "retrieved_memory_ids": ret_ids,
                    "used_memory_ids": used_ids,
                    "reward": float(reward or 0.5),
                })
            return out
        finally:
            cur.close()

    def count_feedback_last_24h(self) -> int:
        """Count feedback_logs rows in the last 24 hours (for observability)."""
        conn = self._conn_or_get()
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT COUNT(*) FROM feedback_logs WHERE created_at > NOW() - INTERVAL '24 hours'"
            )
            return int(cur.fetchone()[0] or 0)
        finally:
            cur.close()

    def get_graph_metrics(self, memory_ids: List[UUID]) -> Dict[UUID, dict]:
        """Return {memory_id: {pagerank, degree}} for given ids. Missing ids omitted."""
        if not memory_ids:
            return {}
        conn = self._conn_or_get()
        cur = conn.cursor()
        try:
            placeholders = ",".join("%s" for _ in memory_ids)
            cur.execute(
                f"SELECT memory_id, pagerank, degree FROM graph_metrics WHERE memory_id IN ({placeholders})",
                [str(mid) for mid in memory_ids],
            )
            out = {}
            for row in cur.fetchall():
                mid, pr, deg = row
                out[UUID(mid)] = {"pagerank": float(pr or 0), "degree": int(deg or 0)}
            return out
        finally:
            cur.close()

    def set_graph_metrics_bulk(self, metrics: Dict[UUID, dict]) -> None:
        """Upsert graph_metrics. metrics: {memory_id: {pagerank, degree}}."""
        if not metrics:
            return
        conn = self._conn_or_get()
        cur = conn.cursor()
        try:
            for mid, vals in metrics.items():
                pr = float(vals.get("pagerank", 0))
                deg = int(vals.get("degree", 0))
                cur.execute(
                    """
                    INSERT INTO graph_metrics (memory_id, pagerank, degree, updated_at)
                    VALUES (%s, %s, %s, NOW())
                    ON CONFLICT (memory_id) DO UPDATE SET pagerank = EXCLUDED.pagerank, degree = EXCLUDED.degree, updated_at = NOW()
                    """,
                    (str(mid), pr, deg),
                )
            conn.commit()
        finally:
            cur.close()
