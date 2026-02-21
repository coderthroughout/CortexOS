"""Timeline: episodic memories -> chronological events by period."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, List
from uuid import UUID

from cortex.memory.schema import Memory, MemoryType
from cortex.memory.store import MemoryStore


def get_timeline(
    store: MemoryStore,
    user_id: UUID,
    limit: int = 200,
) -> List[dict]:
    """
    Return timeline of memories: chronological events, grouped by month or phase.
    Format: list of { "period": "Feb 2026", "events": [ { "id", "summary", "created_at", "type" } ] }.
    """
    memories = store.get_user_memories(user_id, limit=limit)
    # Prefer episodic for timeline; include semantic/procedural as secondary
    by_period: dict[str, List[dict]] = defaultdict(list)
    for m in memories:
        ts = m.created_at or datetime.utcnow()
        period = ts.strftime("%b %Y")
        by_period[period].append({
            "id": str(m.id),
            "summary": m.summary,
            "created_at": ts.isoformat(),
            "type": m.type.value,
        })
    # Sort periods (newest first)
    keys = sorted(by_period.keys(), key=lambda x: datetime.strptime(x, "%b %Y"), reverse=True)
    return [{"period": k, "events": by_period[k]} for k in keys]
