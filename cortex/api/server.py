"""
CortexOS FastAPI server. Minimal Phase 1: memory add + query.
"""
from __future__ import annotations

import os
import sys

# Ensure cortex is on path when run as main
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2
from fastapi import FastAPI
from pgvector.psycopg2 import register_vector
from fastapi.middleware.cors import CORSMiddleware

from cortex.api.routes_memory import router as memory_router
from cortex.api.routes_consolidate import router as consolidate_router
from cortex.memory.store import MemoryStore
from cortex.memory.vector_index import VectorIndex
from cortex.utils.config import DATABASE_URL, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


def get_db_connection():
    return psycopg2.connect(DATABASE_URL)


app = FastAPI(
    title="CortexOS",
    description="Cognitive Memory System for AI Agents",
    version="0.1.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def startup():
    conn = get_db_connection()
    register_vector(conn)
    app.state.db_connection = conn
    app.state.memory_store = MemoryStore(db_connection=conn)
    app.state.vector_index = VectorIndex(db_connection=conn)
    try:
        from cortex.graph.graph_store import GraphStore
        app.state.graph_store = GraphStore(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)
    except Exception:
        app.state.graph_store = None
    app.state.bm25_index = None  # optional: build from store when needed
    try:
        from cortex.ranking.mvn_model import load_mvn
        from cortex.ranking.mvn_features import build_mvn_feature_dim
        mvn_path = os.environ.get("CORTEX_MVN_CHECKPOINT")
        app.state.mvn_model = load_mvn(path=mvn_path, input_dim=build_mvn_feature_dim()) if mvn_path else None
    except Exception:
        app.state.mvn_model = None


@app.on_event("shutdown")
def shutdown():
    if hasattr(app.state, "db_connection") and app.state.db_connection:
        app.state.db_connection.close()
    if getattr(app.state, "graph_store", None):
        app.state.graph_store.close()


app.include_router(memory_router)
app.include_router(consolidate_router)


@app.get("/health")
def health():
    return {"status": "ok", "service": "cortexos"}
