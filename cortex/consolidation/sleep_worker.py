"""Consolidation job: cluster episodic memories, create semantic, prune low retention. Trigger via POST /consolidate/run."""
from __future__ import annotations

from typing import Any, List, Optional
from uuid import UUID

from cortex.consolidation.clustering import cluster_memories
from cortex.consolidation.decay import compute_retention, should_delete
from cortex.consolidation.summarizer import create_semantic_memory
from cortex.memory.schema import Memory, MemoryType
from cortex.memory.store import MemoryStore
from cortex.memory.vector_index import VectorIndex


def run_consolidation(
    user_id: UUID,
    store: MemoryStore,
    vector_index: VectorIndex,
    graph_store: Optional[Any] = None,
    min_cluster_size: int = 2,
    distance_threshold: float = 0.35,
) -> dict:
    """
    Load user memories -> cluster episodic -> create semantic for each cluster -> optionally delete low-retention.
    """
    memories = store.get_user_memories(user_id, limit=2000)
    episodic = [m for m in memories if m.type == MemoryType.EPISODIC and (m.embedding is not None or getattr(m, "embedding", None))]
    if len(episodic) < min_cluster_size:
        return {"clusters": 0, "semantic_created": 0, "deleted": 0}
    embeddings = [getattr(m, "embedding", None) for m in episodic]
    if not all(embeddings):
        return {"clusters": 0, "semantic_created": 0, "deleted": 0}
    clusters = cluster_memories(episodic, embeddings=embeddings, distance_threshold=distance_threshold)
    semantic_created = 0
    for indices in clusters:
        if len(indices) < min_cluster_size:
            continue
        cluster = [episodic[i] for i in indices]
        sem = create_semantic_memory(cluster, store=store, vector_index=vector_index, graph_store=graph_store, user_id=user_id)
        if sem:
            semantic_created += 1
    deleted = 0
    for m in memories:
        score = compute_retention(m)
        if should_delete(score):
            store.delete(m.id)
            deleted += 1
    return {"clusters": len(clusters), "semantic_created": semantic_created, "deleted": deleted}
