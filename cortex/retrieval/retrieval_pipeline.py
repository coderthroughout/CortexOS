"""Retrieval pipeline: hybrid candidates -> MVN scoring -> rerank -> top-k."""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, List, Optional
from uuid import UUID

from cortex.ranking.mvn_inference import score_candidates as mvn_score_candidates
from cortex.ranking.reranker import rerank
from cortex.retrieval.candidate_builder import Candidate, build_candidates
from cortex.retrieval.hybrid_search import retrieve_candidates
from cortex.retrieval.intent import detect_intent_simple

if TYPE_CHECKING:
    from cortex.graph.graph_store import GraphStore
    from cortex.memory.store import MemoryStore
    from cortex.memory.vector_index import VectorIndex
    from cortex.ranking.mvn_model import MVN
    from cortex.retrieval.bm25_index import BM25Index


def retrieve_with_hybrid(
    query: str,
    user_id: Optional[UUID],
    vector_index: "VectorIndex",
    store: "MemoryStore",
    bm25_index: Optional["BM25Index"] = None,
    graph_store: Optional["GraphStore"] = None,
    mvn_model: Optional["MVN"] = None,
    k: int = 10,
    use_intent: bool = True,
    use_reranker: bool = True,
    rerank_top_k: int = 5,
    timings: Optional[dict] = None,
) -> List[Candidate]:
    """
    Run hybrid retrieval: candidates -> MVN scoring (if model) -> rerank -> return top-k.
    Ranking formula when MVN present: 0.4*MVN + 0.2*similarity + 0.15*recency + 0.15*importance + 0.1*graph.
    If timings dict is provided, fills embed_ms, vector_ms, bm25_ms, graph_ms, build_ms, mvn_ms, rerank_ms.
    """
    raw = retrieve_candidates(
        query,
        user_id=user_id,
        vector_index=vector_index,
        store=store,
        bm25_index=bm25_index,
        graph_store=graph_store,
        timings=timings,
    )
    # Merge graph_metrics (pagerank, degree) from cache for graph_score in build_candidates
    memory_ids = [m.id for m, _ in raw]
    graph_metrics = store.get_graph_metrics(memory_ids)
    for m, feats in raw:
        gm = graph_metrics.get(m.id, {})
        feats["pagerank"] = gm.get("pagerank", 0.0)
        feats["degree"] = gm.get("degree", 0)
    t_build = time.perf_counter()
    candidates = build_candidates(raw)
    if timings is not None:
        timings["build_ms"] = round((time.perf_counter() - t_build) * 1000, 2)
    # MVN scoring
    t_mvn = time.perf_counter()
    candidates = mvn_score_candidates(query, candidates, model=mvn_model)
    if timings is not None:
        timings["mvn_ms"] = round((time.perf_counter() - t_mvn) * 1000, 2)
    # Combined score for sort when MVN used
    for c in candidates:
        if c.mvn_score is not None:
            c.final_score = (
                0.4 * c.mvn_score
                + 0.2 * c.similarity
                + 0.15 * (c.recency or c.temporal_score)
                + 0.15 * c.importance
                + 0.1 * c.graph_score
            )
        else:
            c.final_score = c.score
    candidates.sort(key=lambda x: x.final_score if x.final_score is not None else x.score, reverse=True)
    top_20 = candidates[:20]
    if use_reranker and top_20:
        t_rerank = time.perf_counter()
        out = rerank(top_20, top_k=min(rerank_top_k, k))
        if timings is not None:
            timings["rerank_ms"] = round((time.perf_counter() - t_rerank) * 1000, 2)
        return out
    if timings is not None:
        timings["rerank_ms"] = 0
    return candidates[:k]
