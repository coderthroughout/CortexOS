"""Cluster memories by embedding similarity (agglomerative)."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from cortex.memory.schema import Memory


def cluster_memories(
    memories: List[Memory],
    embeddings: Optional[List[List[float]]] = None,
    n_clusters: Optional[int] = None,
    distance_threshold: float = 0.3,
) -> List[List[int]]:
    """
    Return list of clusters; each cluster is list of indices into memories.
    If embeddings is None, use memory.embedding; else use provided.
    """
    if not memories:
        return []
    X = []
    for i, m in enumerate(memories):
        emb = embeddings[i] if embeddings and i < len(embeddings) else getattr(m, "embedding", None)
        if emb is None:
            return []
        X.append(emb)
    X = np.array(X, dtype=np.float32)
    if n_clusters is not None:
        model = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average")
    else:
        model = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, metric="cosine", linkage="average")
    labels = model.fit_predict(X)
    clusters: dict[int, List[int]] = {}
    for i, lab in enumerate(labels):
        clusters.setdefault(int(lab), []).append(i)
    return list(clusters.values())
