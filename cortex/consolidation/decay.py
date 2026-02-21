"""Retention score and policies for keep / compress / delete."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

from cortex.memory.schema import Memory


def compute_retention(memory: Memory, lambda_decay: float = 0.1) -> float:
    """
    RetentionScore = importance * recency * usage * MVN.
    recency = e^(-λ * time_since_last_used).
    """
    importance = memory.importance or 0.5
    usage = getattr(memory, "usage_count", 0) or getattr(memory, "access_count", 0)
    usage_factor = min(1.0, 0.3 + 0.7 * min(1.0, usage / 10.0))
    t = memory.last_accessed or getattr(memory, "last_used", None) or memory.created_at
    if not t:
        recency = 0.5
    else:
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        delta_days = (datetime.now(timezone.utc) - t).total_seconds() / 86400
        recency = math.exp(-lambda_decay * max(0, delta_days))
    mvn = memory.mvn_score if memory.mvn_score is not None else 0.5
    return importance * recency * usage_factor * max(0.2, mvn)


def should_compact(score: float, threshold: float = 0.2) -> bool:
    return score < threshold


def should_delete(score: float, threshold: float = 0.08) -> bool:
    return score < threshold
