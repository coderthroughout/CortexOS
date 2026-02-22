"""Memory API: add, query, search, feedback, timeline, ingest, delete, patch."""
from __future__ import annotations

import time
from typing import Any, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

from cortex.ingestion.pipeline import ingest as ingest_pipeline
from cortex.ingestion.parser import parse_chat
from cortex.memory.schema import MemoryCreate, MemorySource, MemoryType
from cortex.memory.store import MemoryStore
from cortex.memory.timeline import get_timeline
from cortex.memory.vector_index import VectorIndex
from cortex.utils.embeddings import embed
from cortex.utils.observability import log_feedback, log_memory_add, log_retrieval


def get_store(request: Request) -> MemoryStore:
    return request.app.state.memory_store


def get_vector_index(request: Request) -> VectorIndex:
    return request.app.state.vector_index

router = APIRouter(prefix="/memory", tags=["memory"])


class AddMemoryBody(BaseModel):
    text: Optional[str] = None
    summary: str
    entities: List[str] = Field(default_factory=list)
    importance: float = Field(0.6, ge=0, le=1)
    type: str = "episodic"
    emotion: Optional[str] = None
    source: str = "chat"
    user_id: Optional[str] = None  # required in path or body


class MemoryResponse(BaseModel):
    id: str
    user_id: str
    type: str
    summary: str
    importance: float
    created_at: Optional[str] = None
    score: Optional[float] = None  # for query results
    score_breakdown: Optional[dict] = None  # explainability: mvn, semantic, recency, importance, graph

    class Config:
        from_attributes = True


class FeedbackBody(BaseModel):
    query_id: Optional[str] = None
    query: Optional[str] = None
    user_id: Optional[str] = None
    retrieved_memory_ids: List[str] = Field(default_factory=list)
    used_memory_ids: List[str] = Field(default_factory=list)
    reward: float = Field(ge=0, le=1)


@router.post("/add", response_model=MemoryResponse)
def add_memory(
    request: Request,
    body: AddMemoryBody,
    user_id: Optional[str] = Query(None, alias="user"),
    store: MemoryStore = Depends(get_store),
    vector_index: VectorIndex = Depends(get_vector_index),
):
    """Add a single memory. Embedding is computed and stored. Graph is updated if Neo4j is connected."""
    uid = user_id or body.user_id
    if not uid:
        raise HTTPException(422, "user_id required (query ?user= or body)")
    try:
        u = UUID(uid)
    except ValueError:
        raise HTTPException(422, "invalid user_id")
    try:
        mem_type = MemoryType(body.type)
    except ValueError:
        mem_type = MemoryType.EPISODIC
    try:
        src = MemorySource(body.source)
    except ValueError:
        src = MemorySource.CHAT
    create = MemoryCreate(
        user_id=u,
        type=mem_type,
        text=body.text or body.summary,
        summary=body.summary,
        entities=body.entities,
        importance=body.importance,
        emotion=body.emotion,
        source=src,
    )
    t0 = time.perf_counter()
    try:
        embedding = embed(create.summary or create.text)
    except Exception as e:
        raise HTTPException(500, detail=f"embedding: {str(e)[:200]}")
    try:
        memory = store.add_memory(create, embedding=embedding)
        vector_index.add(memory.id, embedding)
    except Exception as e:
        raise HTTPException(500, detail=f"store/vector: {str(e)[:300]}")
    graph_store = getattr(request.app.state, "graph_store", None)
    if graph_store:
        try:
            from cortex.graph.graph_builder import build_graph_for_memory
            build_graph_for_memory(
                graph_store,
                memory.id,
                memory.user_id,
                memory.summary or "",
                memory.type.value,
                body.entities,
                relationships=None,
            )
        except Exception:
            pass  # do not fail add if graph update fails
    latency_ms = (time.perf_counter() - t0) * 1000
    log_memory_add(str(memory.id), str(memory.user_id), latency_ms=latency_ms)
    return MemoryResponse(
        id=str(memory.id),
        user_id=str(memory.user_id),
        type=memory.type.value,
        summary=memory.summary,
        importance=memory.importance,
        created_at=memory.created_at.isoformat() if memory.created_at else None,
    )


@router.post("/ingest", response_model=List[MemoryResponse])
def ingest_memory(
    body: dict,
    request: Request,
    user_id: str = Query(..., alias="user"),
):
    """Ingest raw conversation: parse -> extract -> store -> graph. Body: { \"messages\": [{\"role\", \"content\"}] } or { \"content\": \"...\" }."""
    store = get_store(request)
    vector_index = get_vector_index(request)
    graph_store = getattr(request.app.state, "graph_store", None)
    try:
        uid = UUID(user_id)
    except ValueError:
        raise HTTPException(422, "invalid user_id")
    messages = body.get("messages")
    if messages:
        raw = parse_chat(messages, session_id=body.get("session_id"), conversation_id=body.get("conversation_id"))
    else:
        from cortex.ingestion.parser import RawInput
        raw = RawInput(source="chat", content=body.get("content", ""))
    created = ingest_pipeline(raw, user_id=uid, store=store, vector_index=vector_index, graph_store=graph_store)
    return [
        MemoryResponse(id=str(m.id), user_id=str(m.user_id), type=m.type.value, summary=m.summary, importance=m.importance, created_at=m.created_at.isoformat() if m.created_at else None)
        for m in created
    ]


@router.patch("/{memory_id}")
def patch_memory(
    memory_id: str,
    body: dict,
    store: MemoryStore = Depends(get_store),
):
    """PATCH /memory/{id}: update summary or importance (safety/corrections)."""
    try:
        uid = UUID(memory_id)
    except ValueError:
        raise HTTPException(422, "invalid memory id")
    summary = body.get("summary")
    importance = body.get("importance")
    if summary is None and importance is None:
        raise HTTPException(422, "provide summary and/or importance")
    ok = store.update(uid, summary=summary, importance=importance)
    if not ok:
        raise HTTPException(404, "memory not found")
    return {"ok": True}


@router.delete("/{memory_id}")
def delete_memory(memory_id: str, store: MemoryStore = Depends(get_store)):
    """DELETE /memory/{id}. Safety: allow human-editable delete."""
    try:
        ok = store.delete(UUID(memory_id))
        if not ok:
            raise HTTPException(404, "memory not found")
        return {"ok": True}
    except ValueError:
        raise HTTPException(422, "invalid memory id")


@router.post("/feedback")
def feedback(
    body: FeedbackBody,
    store: MemoryStore = Depends(get_store),
):
    """POST /memory/feedback: record used memories and reward for MVN training. Updates usage_count, last_used, and appends to feedback_logs."""
    for mid in body.used_memory_ids:
        try:
            store.update_usage(UUID(mid))
        except Exception:
            pass
    try:
        uid = UUID(body.user_id) if body.user_id else None
    except ValueError:
        uid = None
    store.append_feedback_log(
        user_id=uid,
        query=body.query,
        retrieved_memory_ids=body.retrieved_memory_ids or [],
        used_memory_ids=body.used_memory_ids,
        reward=body.reward,
    )
    log_feedback(body.reward, len(body.used_memory_ids))
    return {"ok": True, "reward": body.reward}


@router.post("/rebuild-bm25")
def rebuild_bm25(request: Request):
    """Rebuild in-memory BM25 index from store. Call after bulk adds if BM25 is used."""
    store = get_store(request)
    try:
        from cortex.retrieval.bm25_index import BM25Index
        pairs = store.get_all_memory_summaries()
        doc_ids = [p[0] for p in pairs]
        texts = [p[1] for p in pairs]
        bm25 = BM25Index()
        bm25.build(doc_ids, texts)
        request.app.state.bm25_index = bm25
        return {"ok": True, "doc_count": len(doc_ids)}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/graph", response_model=List[dict])
def graph(
    node: Optional[str] = Query(None),
    depth: int = Query(2, ge=1, le=5),
    request: Request = None,
):
    """GET /memory/graph?node=...&depth=2. Returns memory ids near entity (node)."""
    graph_store = getattr(request.app.state, "graph_store", None)
    if not graph_store or not node:
        return []
    ids = graph_store.traverse([node], depth=depth)
    return [{"memory_id": mid} for mid in ids]


@router.get("/timeline", response_model=List[dict])
def timeline(
    user: str = Query(..., description="user_id"),
    store: MemoryStore = Depends(get_store),
):
    """Timeline of memories by period. GET /memory/timeline?user=..."""
    try:
        uid = UUID(user)
    except ValueError:
        raise HTTPException(422, "invalid user_id")
    return get_timeline(store, uid)


@router.get("/query", response_model=List[MemoryResponse])
@router.get("/search", response_model=List[MemoryResponse])
def query_memory(
    request: Request,
    response: Response,
    q: str = Query(..., description="Search query"),
    user: Optional[str] = Query(None, description="user_id filter"),
    k: int = Query(10, ge=1, le=100),
    debug: bool = Query(False, description="Include timing headers"),
):
    """Query memories: hybrid retrieval + optional MVN + reranker. Returns ranked memories with score."""
    store = get_store(request)
    vector_index = get_vector_index(request)
    user_id = UUID(user) if user else None
    bm25 = getattr(request.app.state, "bm25_index", None)
    graph_store = getattr(request.app.state, "graph_store", None)
    mvn_model = getattr(request.app.state, "mvn_model", None)
    from cortex.retrieval.retrieval_pipeline import retrieve_with_hybrid
    timings: dict = {}
    t0 = time.perf_counter()
    candidates = retrieve_with_hybrid(
        query=q,
        user_id=user_id,
        vector_index=vector_index,
        store=store,
        bm25_index=bm25,
        graph_store=graph_store,
        mvn_model=mvn_model,
        k=k,
        use_reranker=True,
        rerank_top_k=5,
        timings=timings,
    )
    total_ms = (time.perf_counter() - t0) * 1000
    timings["total_ms"] = round(total_ms, 2)
    log_retrieval(total_ms, k, user, timings=timings)
    if debug:
        response.headers["X-Query-Total-Ms"] = str(round(total_ms, 2))
        for key in ("embed_ms", "vector_ms", "bm25_ms", "graph_ms", "build_ms", "mvn_ms", "rerank_ms"):
            if key in timings:
                response.headers[f"X-Query-{key.replace('_ms', '-Ms').replace('_', '-').title()}"] = str(timings[key])
    return [
        MemoryResponse(
            id=str(c.memory.id),
            user_id=str(c.memory.user_id),
            type=c.memory.type.value,
            summary=c.memory.summary,
            importance=c.memory.importance,
            created_at=c.memory.created_at.isoformat() if c.memory.created_at else None,
            score=round((c.final_score or c.score), 4),
            score_breakdown={
                "mvn": c.mvn_score,
                "similarity": round(c.similarity, 4),
                "recency": round(c.recency or 0, 4),
                "importance": round(c.importance, 4),
                "graph": round(c.graph_score, 4),
            },
        )
        for c in candidates
    ]
