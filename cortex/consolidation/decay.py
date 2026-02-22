"""Retention score and policies for keep / compress / delete.
Spec: Pi(m) = V_theta(m,q_bar)*(1 - D(m)) - C_storage(m); thresholds tau_compact, tau_delete.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

from cortex.memory.schema import Memory


def discount_factor(memory: Memory, lambda_decay: float = 0.1) -> float:
    """D(m): decay factor from age and usage. Returns value in [0, 1]; higher = more decayed."""
    t = memory.last_accessed or getattr(memory, "last_used", None) or memory.created_at
    if not t:
        return 0.5
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    delta_days = (datetime.now(timezone.utc) - t).total_seconds() / 86400
    age_decay = 1.0 - math.exp(-lambda_decay * max(0, delta_days))
    usage = getattr(memory, "usage_count", 0) or getattr(memory, "access_count", 0)
    usage_mod = 0.9 - 0.4 * min(1.0, usage / 10.0)  # more usage -> less discount
    return min(1.0, max(0.0, age_decay * usage_mod))


def storage_cost(memory: Memory) -> float:
    """C_storage(m): simple cost from text size. Non-negative."""
    text = (memory.summary or "") + (getattr(memory, "raw_text") or "")
    return min(1.0, 0.001 * max(0, len(text)))  # scale down so cost is small


def expected_value_proxy(memory: Memory) -> float:
    """Proxy for V_theta(m, q_bar) until MVN is trained. Uses mvn_score and importance."""
    mvn = memory.mvn_score if memory.mvn_score is not None else 0.5
    imp = memory.importance or 0.5
    return 0.6 * max(0.2, mvn) + 0.4 * imp


def compute_pi(
    memory: Memory,
    lambda_decay: float = 0.1,
    tau_compact: float = 0.2,
    tau_delete: float = 0.08,
) -> float:
    """
    Pi(m) = expected_value * (1 - D(m)) - C_storage(m).
    Used for retention decisions; compact if Pi < tau_compact, delete if Pi < tau_delete.
    """
    ev = expected_value_proxy(memory)
    d = discount_factor(memory, lambda_decay)
    c = storage_cost(memory)
    return max(0.0, ev * (1.0 - d) - c)


def compute_retention(memory: Memory, lambda_decay: float = 0.1) -> float:
    """
    Retention score for backward compatibility. Delegates to compute_pi.
    """
    return compute_pi(memory, lambda_decay=lambda_decay)


def should_compact(score: float, threshold: float = 0.2) -> bool:
    """True if retention Pi(m) < tau_compact."""
    return score < threshold


def should_delete(score: float, threshold: float = 0.08) -> bool:
    """True if retention Pi(m) < tau_delete."""
    return score < threshold
