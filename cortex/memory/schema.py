"""Atomic memory schema: id, user_id, type, text, summary, embedding, entities, emotion, importance, timestamps, source."""
from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class MemorySource(str, Enum):
    CHAT = "chat"
    DOC = "doc"
    TOOL = "tool"


class MemoryBase(BaseModel):
    user_id: UUID
    type: MemoryType
    text: str = ""
    summary: str
    entities: list[str] = Field(default_factory=list)
    emotion: Optional[str] = None
    importance: float = Field(ge=0.0, le=1.0, default=0.5)
    source: MemorySource = MemorySource.CHAT


class MemoryCreate(MemoryBase):
    """Input for creating a memory. Embedding is computed on add."""

    pass


class Memory(MemoryBase):
    """Full memory record as stored and returned."""

    id: UUID = Field(default_factory=uuid4)
    embedding: Optional[list[float]] = None  # stored in DB as vector; here as list for JSON
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    last_used: Optional[datetime] = None  # alias used in doc
    usage_count: int = 0  # doc uses both access_count and usage_count; we use usage_count
    mvn_score: Optional[float] = None
    raw_text: Optional[str] = None  # doc: raw_text in DB
    graph_node_id: Optional[str] = None  # Neo4j node id
    provenance: list[str] = Field(default_factory=list)  # for semantic memories from consolidation
    source_session: Optional[str] = None
    conversation_id: Optional[str] = None
    confidence_score: Optional[float] = None

    class Config:
        from_attributes = True

    def to_db_row(self) -> dict[str, Any]:
        """For insertion into Postgres. Embedding as list; DB uses vector type."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "type": self.type.value,
            "summary": self.summary,
            "raw_text": self.raw_text or self.text,
            "embedding": self.embedding,
            "importance": self.importance,
            "emotion": self.emotion,
            "created_at": self.created_at,
            "last_used": self.last_accessed or self.last_used or self.created_at,
            "usage_count": self.usage_count or self.access_count,
            "mvn_score": self.mvn_score,
            "entities": self.entities or [],
            "source": self.source.value,
        }


def memory_from_row(row: Any) -> Memory:
    """Build Memory from DB row (e.g. dict or Row)."""
    if hasattr(row, "_mapping"):
        row = dict(row._mapping)
    elif hasattr(row, "keys"):
        row = dict(row)
    emb = row.get("embedding")
    if hasattr(emb, "tolist"):
        emb = emb.tolist()
    elif isinstance(emb, str):
        try:
            emb = json.loads(emb)
        except Exception:
            emb = None
    return Memory(
        id=UUID(row["id"]) if isinstance(row["id"], str) else row["id"],
        user_id=UUID(row["user_id"]) if isinstance(row["user_id"], str) else row["user_id"],
        type=MemoryType(row["type"]),
        text=row.get("raw_text") or row.get("summary") or "",
        summary=row.get("summary") or "",
        raw_text=row.get("raw_text"),
        embedding=emb,
        entities=row.get("entities") or [],
        emotion=row.get("emotion"),
        importance=float(row.get("importance") or 0.5),
        source=MemorySource(row.get("source") or "chat"),
        created_at=row.get("created_at") or datetime.utcnow(),
        last_accessed=row.get("last_used"),
        last_used=row.get("last_used"),
        usage_count=int(row.get("usage_count") or 0),
        access_count=int(row.get("usage_count") or 0),
        mvn_score=float(row["mvn_score"]) if row.get("mvn_score") is not None else None,
    )
