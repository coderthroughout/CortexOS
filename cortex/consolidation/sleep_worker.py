"""Consolidation job: cluster episodic memories, create semantic, prune low retention. Trigger via POST /consolidate/run."""
from __future__ import annotations

from typing import Any, List, Optional
from uuid import UUID

from cortex.consolidation.clustering import cluster_memories
from cortex.consolidation.decay import compute_pi, compute_retention, should_delete
from cortex.consolidation.summarizer import create_semantic_memory
from cortex.memory.schema import Memory, MemoryType
from cortex.memory.store import MemoryStore
from cortex.memory.vector_index import VectorIndex


def _cluster_utility(cluster: List[Memory]) -> float:
    """Average Pi(m) over cluster for utility-aware summarization."""
    if not cluster:
        return 0.0
    return sum(compute_pi(m) for m in cluster) / len(cluster)


def run_consolidation(
    user_id: UUID,
    store: MemoryStore,
    vector_index: VectorIndex,
    graph_store: Optional[Any] = None,
    min_cluster_size: int = 2,
    distance_threshold: float = 0.35,
    cluster_utility_threshold: float = 0.15,
    max_clusters_to_summarize: int = 50,
) -> dict:
    """
    Load user memories -> cluster episodic -> create semantic only for clusters with utility above threshold -> delete low-retention.
    """
    memories = store.get_user_memories(user_id, limit=2000)
    episodic = [m for m in memories if m.type == MemoryType.EPISODIC and (m.embedding is not None or getattr(m, "embedding", None))]
    if len(episodic) < min_cluster_size:
        return {"clusters": 0, "semantic_created": 0, "deleted": 0}
    embeddings = [getattr(m, "embedding", None) for m in episodic]
    if not all(embeddings):
        return {"clusters": 0, "semantic_created": 0, "deleted": 0}
    clusters = cluster_memories(episodic, embeddings=embeddings, distance_threshold=distance_threshold)
    cluster_list = [(indices, [episodic[i] for i in indices]) for indices in clusters if len(indices) >= min_cluster_size]
    cluster_list.sort(key=lambda x: _cluster_utility(x[1]), reverse=True)
    semantic_created = 0
    for idx, (_, cluster) in enumerate(cluster_list):
        if semantic_created >= max_clusters_to_summarize:
            break
        if _cluster_utility(cluster) < cluster_utility_threshold:
            continue
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
