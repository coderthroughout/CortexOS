"""Build MVN feature vector from query, memory, and optional graph signals."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import math

from cortex.memory.schema import Memory
from cortex.retrieval.candidate_builder import Candidate
from cortex.utils.embeddings import embed


def _cosine_sim(a: List[float], b: List[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _recency_score(created_at: Optional[datetime], last_used: Optional[datetime], lambda_decay: float = 0.1) -> float:
    t = last_used or created_at
    if not t:
        return 0.0
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    delta_days = (now - t).total_seconds() / 86400
    return math.exp(-lambda_decay * max(0, delta_days))


def _entity_overlap(query_entities: List[str], memory_entities: List[str]) -> float:
    if not query_entities:
        return 0.0
    qset = set(e.lower() for e in query_entities)
    mset = set(e.lower() for e in memory_entities)
    inter = len(qset & mset)
    return inter / len(qset) if qset else 0.0


def build_mvn_features(
    query: str,
    query_embedding: Optional[List[float]] = None,
    candidate: Optional[Candidate] = None,
    memory: Optional[Memory] = None,
    pagerank: float = 0.0,
    graph_distance: float = 0.0,
    intent_type: int = 0,
) -> List[float]:
    """
    Build feature vector for MVN. Either pass Candidate (with .memory and .similarity etc) or Memory.
    Returns list of floats: similarity, recency, importance, usage_count, pagerank, entity_overlap,
    emotion_intensity (placeholder), topic_match (same as similarity), novelty (placeholder), graph_distance.
    """
    if candidate is not None:
        mem = candidate.memory
        similarity = candidate.similarity
        recency = candidate.recency or candidate.temporal_score
        importance = candidate.importance
        entity_overlap = candidate.entity_overlap
        if memory is not None:
            mem = memory
    elif memory is not None:
        mem = memory
        q_emb = query_embedding or embed(query)
        mem_emb = mem.embedding
        similarity = _cosine_sim(q_emb, mem_emb) if mem_emb else 0.0
        recency = _recency_score(mem.created_at, mem.last_accessed or getattr(mem, "last_used", None))
        importance = mem.importance or 0.5
        from cortex.ingestion.entity_parser import extract_entities
        query_entities = extract_entities(query)
        entity_overlap = _entity_overlap(query_entities, mem.entities or [])
    else:
        return []
    usage = getattr(mem, "usage_count", 0) or getattr(mem, "access_count", 0)
    usage_norm = min(1.0, usage / 10.0) if usage else 0.0
    emotion_intensity = 0.5 if (getattr(mem, "emotion", None) and mem.emotion) else 0.0
    novelty = 0.5  # placeholder: distance from other memories
    return [
        similarity,
        recency,
        importance,
        usage_norm,
        pagerank,
        entity_overlap,
        emotion_intensity,
        similarity,  # topic_match
        novelty,
        graph_distance,
        float(intent_type) / 4.0,
    ]


def build_mvn_feature_dim() -> int:
    """Return expected feature dimension (for model input_dim)."""
    return 11
