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


def build_candidates(hybrid_results: List[tuple]) -> List[Candidate]:
    """
    hybrid_results: list of (Memory, features_dict) from hybrid_search.retrieve_candidates.
    """
    out = []
    for mem, feats in hybrid_results:
        recency = feats.get("temporal_score", 0)
        importance = feats.get("importance", mem.importance or 0.5)
        sim = feats.get("similarity", 0)
        bm25 = feats.get("bm25_score", 0)
        from_graph = feats.get("from_graph", False)
        graph_score = 0.3 if from_graph else 0.0
        out.append(
            Candidate(
                memory=mem,
                similarity=sim,
                recency=recency,
                importance=importance,
                pagerank=feats.get("pagerank", 0.0),
                entity_overlap=feats.get("entity_overlap", 0.0),
                bm25_score=bm25,
                temporal_score=recency,
                from_graph=from_graph,
                graph_score=graph_score,
                features=feats,
            )
        )
    return out
