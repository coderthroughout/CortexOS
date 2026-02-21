"""Consolidation API: POST /consolidate/run?user=..."""
from uuid import UUID
from fastapi import APIRouter, HTTPException, Query, Request
from cortex.memory.store import MemoryStore
from cortex.memory.vector_index import VectorIndex
from cortex.consolidation.sleep_worker import run_consolidation

router = APIRouter(prefix="/consolidate", tags=["consolidation"])

def get_store(request: Request) -> MemoryStore:
    return request.app.state.memory_store

def get_vector_index(request: Request) -> VectorIndex:
    return request.app.state.vector_index

@router.post("/run")
def run(request: Request, user: str = Query(..., description="user_id")):
    try:
        uid = UUID(user)
    except ValueError:
        raise HTTPException(422, "invalid user_id")
    store = get_store(request)
    vector_index = get_vector_index(request)
    graph_store = getattr(request.app.state, "graph_store", None)
    return run_consolidation(uid, store, vector_index, graph_store)
