"""Hybrid retrieval: vector + BM25 + graph expansion, merged into candidate pool."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING, List, Optional, Set
from uuid import UUID

from cortex.utils.embeddings import embed

if TYPE_CHECKING:
    from cortex.graph.graph_store import GraphStore
    from cortex.memory.store import MemoryStore
    from cortex.memory.vector_index import VectorIndex
    from cortex.retrieval.bm25_index import BM25Index


def temporal_score(created_at: Optional[datetime], last_used: Optional[datetime], lambda_decay: float = 0.1) -> float:
    """Exponential time decay."""
    t = last_used or created_at
    if not t:
        return 0.5
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    delta_days = (now - t).total_seconds() / 86400
    return math.exp(-lambda_decay * max(0, delta_days))


def retrieve_candidates(
    query: str,
    user_id: Optional[UUID],
    vector_index: "VectorIndex",
    store: "MemoryStore",
    bm25_index: Optional["BM25Index"] = None,
    graph_store: Optional["GraphStore"] = None,
    top_k1: int = 50,
    top_k2: int = 30,
    merge_cap: int = 100,
) -> List[tuple]:
    """
    Union of vector search, BM25 search, and graph expansion. Returns list of (Memory, features_dict).
    features_dict includes: similarity, bm25_score, temporal_score, from_graph, etc.
    """
    # Entity names from query (simple: words that could be entities)
    from cortex.ingestion.entity_parser import extract_entities
    query_entities = extract_entities(query)
    memory_ids_seen: Set[str] = set()
    candidates: List[tuple] = []  # (memory, features)

    # 1) Vector search
    q_emb = embed(query)
    vec_results = vector_index.search(q_emb, user_id=user_id, k=top_k1)
    for mem, sim in vec_results:
        memory_ids_seen.add(str(mem.id))
        candidates.append((mem, {"similarity": sim, "bm25_score": 0.0, "from_graph": False}))

    # 2) BM25
    if bm25_index is not None:
        user_set = None
        if user_id:
            user_mems = store.get_user_memories(user_id, limit=5000)
            user_set = {str(m.id) for m in user_mems}
        for mid, bm25_s in bm25_index.search(query, top_k=top_k2, user_doc_ids=user_set):
            if mid in memory_ids_seen:
                for i, (m, feats) in enumerate(candidates):
                    if str(m.id) == mid:
                        feats["bm25_score"] = bm25_s
                        break
                continue
            memory_ids_seen.add(mid)
            mem = store.get_memory(UUID(mid))
            if mem:
                candidates.append((mem, {"similarity": 0.0, "bm25_score": bm25_s, "from_graph": False}))

    # 3) Graph expansion
    if graph_store is not None and query_entities:
        graph_ids = graph_store.traverse(query_entities, depth=2)
        for mid in graph_ids:
            if mid in memory_ids_seen:
                for i, (m, feats) in enumerate(candidates):
                    if str(m.id) == mid:
                        feats["from_graph"] = True
                        break
                continue
            memory_ids_seen.add(mid)
            mem = store.get_memory(UUID(mid))
            if mem and (user_id is None or str(mem.user_id) == str(user_id)):
                candidates.append((mem, {"similarity": 0.0, "bm25_score": 0.0, "from_graph": True}))

    # Cap and add temporal score
    for mem, feats in candidates[:merge_cap]:
        feats["temporal_score"] = temporal_score(mem.created_at, mem.last_accessed or mem.last_used)
        feats["importance"] = mem.importance or 0.5
    return candidates[:merge_cap]
