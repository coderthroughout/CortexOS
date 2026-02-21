"""Rerank top candidates by weighted graph_score, MVN, and similarity."""
from __future__ import annotations

from typing import List

from cortex.retrieval.candidate_builder import Candidate


def rerank(candidates: List[Candidate], top_k: int = 5) -> List[Candidate]:
    """
    Final score = 0.5*reranker_score + 0.3*MVN + 0.2*similarity (doc).
    v1: reranker_score = graph_score (from graph features).
    """
    for c in candidates:
        mvn = c.mvn_score if c.mvn_score is not None else 0.0
        reranker_score = c.graph_score  # or future GNN score
        c.final_score = 0.5 * reranker_score + 0.3 * mvn + 0.2 * c.similarity
    candidates.sort(key=lambda x: getattr(x, "final_score", x.score), reverse=True)
    return candidates[:top_k]
