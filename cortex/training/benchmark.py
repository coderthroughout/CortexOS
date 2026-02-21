"""Benchmark helpers: Recall@K, MRR, retrieval latency."""
from __future__ import annotations

import time
from typing import List, Optional
from uuid import UUID


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """1 if any relevant in top-k else 0."""
    top = set(retrieved_ids[:k])
    return 1.0 if any(rid in top for rid in relevant_ids) else 0.0


def mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Mean reciprocal rank: 1 / rank of first relevant."""
    rel_set = set(relevant_ids)
    for i, rid in enumerate(retrieved_ids):
        if rid in rel_set:
            return 1.0 / (i + 1)
    return 0.0


def run_retrieval_benchmark(
    query: str,
    user_id: Optional[UUID],
    retrieve_fn,
    k: int = 5,
) -> dict:
    """Run one query, measure latency; return latency_ms and top_k ids."""
    start = time.perf_counter()
    candidates = retrieve_fn(query, user_id=user_id, k=k)
    latency_ms = (time.perf_counter() - start) * 1000
    ids = [str(c.memory.id) for c in candidates] if hasattr(candidates[0], "memory") else [str(c.id) for c in candidates]
    return {"latency_ms": round(latency_ms, 2), "top_k_ids": ids}
