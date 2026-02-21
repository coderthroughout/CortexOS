"""Normalize and dedupe extracted memories; add session id, conversation id, confidence."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from cortex.ingestion.entity_parser import extract_entities, resolve_entities
from cortex.memory.schema import MemoryCreate, MemorySource, MemoryType
from cortex.utils.embeddings import embed


# Map extractor memory_type to schema type
_TYPE_MAP = {
    "event": MemoryType.EPISODIC,
    "preference": MemoryType.SEMANTIC,
    "goal": MemoryType.SEMANTIC,
    "relationship": MemoryType.SEMANTIC,
    "belief": MemoryType.SEMANTIC,
}


def normalize_memory(
    raw: dict,
    user_id: UUID,
    source: str = "chat",
    source_session: Optional[str] = None,
    conversation_id: Optional[str] = None,
    entity_aliases: Optional[dict] = None,
    compute_embedding: bool = True,
) -> tuple[MemoryCreate, Optional[List[float]], dict]:
    """
    Normalize one extracted memory dict into MemoryCreate + optional embedding + extra metadata.
    raw: from extract_memories() with memory_type, summary, entities, importance, emotion, etc.
    Returns (MemoryCreate, embedding or None, extra) where extra has source_session, conversation_id, confidence_score, relationships.
    """
    summary = (raw.get("summary") or "").strip()
    if not summary:
        raise ValueError("empty summary")
    entities = raw.get("entities") or []
    from_text = extract_entities(summary)
    entities = list(dict.fromkeys(entities + from_text))
    entities = resolve_entities(entities, entity_aliases)
    mem_type = _TYPE_MAP.get((raw.get("memory_type") or "event").lower(), MemoryType.EPISODIC)
    importance = float(raw.get("importance", 0.5))
    importance = max(0.0, min(1.0, importance))
    emotion = raw.get("emotion")
    if emotion and isinstance(emotion, str):
        emotion = emotion.strip() or None
    create = MemoryCreate(
        user_id=user_id,
        type=mem_type,
        text=summary,
        summary=summary,
        entities=entities,
        importance=importance,
        emotion=emotion,
        source=MemorySource(source),
    )
    emb = embed(summary) if compute_embedding else None
    extra = {
        "source_session": source_session,
        "conversation_id": conversation_id,
        "confidence_score": min(1.0, importance + 0.1),
        "relationships": raw.get("relationships") or [],
        "timestamp_reference": raw.get("timestamp_reference"),
    }
    return create, emb, extra


def deduplicate_memories(memories: List[MemoryCreate], threshold: float = 0.95) -> List[MemoryCreate]:
    """
    Simple dedupe by summary similarity: keep first of each near-duplicate cluster.
    Uses embedding cosine similarity if available; otherwise exact summary match.
    """
    if not memories:
        return []
    seen_embeddings: List[List[float]] = []
    out = []
    for m in memories:
        emb = embed(m.summary)
        is_dup = False
        for seen in seen_embeddings:
            if _cosine_sim(emb, seen) >= threshold:
                is_dup = True
                break
        if not is_dup:
            seen_embeddings.append(emb)
            out.append(m)
    return out


def _cosine_sim(a: List[float], b: List[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
