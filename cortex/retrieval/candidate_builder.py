"""Build Candidate objects (memory + similarity, recency, importance, graph_score, etc.) from raw results."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import UUID

from cortex.memory.schema import Memory


@dataclass
class Candidate:
    """One candidate memory with features for ranking (MVN, reranker)."""

    memory: Memory
    similarity: float = 0.0
    recency: float = 0.0
    importance: float = 0.5
    pagerank: float = 0.0
    entity_overlap: float = 0.0
    bm25_score: float = 0.0
    temporal_score: float = 0.0
    from_graph: bool = False
    graph_score: float = 0.0  # combined graph signal for reranker
    mvn_score: Optional[float] = None
    final_score: Optional[float] = None  # set by reranker
    features: Optional[Dict[str, Any]] = None  # raw feature dict

    @property
    def score(self) -> float:
        """Simple combined score when MVN not used: 0.4*sim + 0.2*recency + 0.2*importance + 0.2*temporal."""
        return (
            0.4 * self.similarity
            + 0.2 * self.recency
            + 0.2 * self.importance
            + 0.2 * (self.temporal_score or 0)
            + 0.1 * (1.0 if self.from_graph else 0.0)
        )


def _normalize(x: float, lo: float, hi: float) -> float:
    """Min-max normalize to [0,1]; if lo==hi return 0.5."""
    if hi is None or lo is None or hi <= lo:
        return 0.5
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def build_candidates(hybrid_results: List[tuple]) -> List[Candidate]:
    """
    hybrid_results: list of (Memory, features_dict) from hybrid_search.retrieve_candidates.
    feats may include pagerank, degree from graph_metrics cache; graph_score = α1*pagerank + α2*degree (normalized), fallback 0.3 if from_graph when missing.
    """
    alpha1, alpha2 = 0.6, 0.4  # weight pagerank vs degree
    pr_vals = [feats.get("pagerank", 0.0) for _, feats in hybrid_results]
    deg_vals = [feats.get("degree", 0) for _, feats in hybrid_results]
    pr_lo, pr_hi = (min(pr_vals), max(pr_vals)) if pr_vals else (0.0, 1.0)
    deg_lo, deg_hi = (min(deg_vals), max(deg_vals)) if deg_vals else (0, 1)

    out = []
    for mem, feats in hybrid_results:
        recency = feats.get("temporal_score", 0)
        importance = feats.get("importance", mem.importance or 0.5)
        sim = feats.get("similarity", 0)
        bm25 = feats.get("bm25_score", 0)
        from_graph = feats.get("from_graph", False)
        pr = feats.get("pagerank", 0.0)
        deg = feats.get("degree", 0)
        if pr_hi > pr_lo or deg_hi > deg_lo:
            npr = _normalize(pr, pr_lo, pr_hi)
            ndeg = _normalize(float(deg), deg_lo, deg_hi)
            graph_score = alpha1 * npr + alpha2 * ndeg
        else:
            graph_score = 0.3 if from_graph else 0.0
        out.append(
            Candidate(
                memory=mem,
                similarity=sim,
                recency=recency,
                importance=importance,
                pagerank=pr,
                entity_overlap=feats.get("entity_overlap", 0.0),
                bm25_score=bm25,
                temporal_score=recency,
                from_graph=from_graph,
                graph_score=graph_score,
                features=feats,
            )
        )
    return out
