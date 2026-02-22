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

import logging

from cortex.api.routes_memory import router as memory_router
from cortex.api.routes_consolidate import router as consolidate_router
from cortex.api.routes_status import router as status_router
from cortex.memory.store import MemoryStore
from cortex.memory.vector_index import VectorIndex
from cortex.utils.config import DATABASE_URL, EMBEDDING_MODEL, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

# Ensure observability logs appear
_log = logging.getLogger("cortexos.observability")
if not _log.handlers:
    _log.setLevel(logging.INFO)
    _log.addHandler(logging.StreamHandler())


def get_db_connection():
    return psycopg2.connect(DATABASE_URL)


app = FastAPI(
    title="CortexOS",
    description="Cognitive Memory System for AI Agents",
    version="0.1.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def rollback_db_on_request(request, call_next):
    """Clear any aborted Postgres transaction so the shared connection can be reused."""
    conn = getattr(request.app.state, "db_connection", None)
    if conn is not None:
        try:
            conn.rollback()
        except Exception:
            pass
    return await call_next(request)


@app.on_event("startup")
def startup():
    conn = get_db_connection()
    try:
        register_vector(conn)
    except psycopg2.ProgrammingError as e:
        if "vector type not found" in str(e) or "vector" in str(e).lower():
            try:
                cur = conn.cursor()
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                conn.commit()
                cur.close()
                register_vector(conn)
            except Exception as e2:
                conn.close()
                raise RuntimeError(
                    "pgvector extension is required. Run in your database: CREATE EXTENSION IF NOT EXISTS vector;"
                ) from e2
        else:
            conn.close()
            raise
    app.state.db_connection = conn
    app.state.memory_store = MemoryStore(db_connection=conn)
    app.state.vector_index = VectorIndex(db_connection=conn)
    try:
        from cortex.graph.graph_store import GraphStore
        app.state.graph_store = GraphStore(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)
    except Exception:
        app.state.graph_store = None
    try:
        from cortex.retrieval.bm25_index import BM25Index
        pairs = app.state.memory_store.get_all_memory_summaries()
        if pairs:
            doc_ids = [p[0] for p in pairs]
            texts = [p[1] for p in pairs]
            bm25 = BM25Index()
            bm25.build(doc_ids, texts)
            app.state.bm25_index = bm25
        else:
            app.state.bm25_index = BM25Index()  # empty index
    except Exception:
        app.state.bm25_index = None
    try:
        from cortex.ranking.mvn_model import load_mvn
        from cortex.ranking.mvn_features import build_mvn_feature_dim
        mvn_path = os.environ.get("CORTEX_MVN_CHECKPOINT")
        app.state.mvn_model = load_mvn(path=mvn_path, input_dim=build_mvn_feature_dim()) if mvn_path else None
    except Exception:
        app.state.mvn_model = None
    # Preload embedder so first add/query does not pay ~8s model load
    try:
        from cortex.utils.embeddings import get_embedder
        get_embedder(EMBEDDING_MODEL)
    except Exception:
        pass  # first request will load on demand
    # Optional background jobs (consolidation + graph metrics every 6h)
    try:
        from cortex.background.scheduler import start_background_scheduler
        app.state.background_scheduler = start_background_scheduler(app)
    except Exception:
        app.state.background_scheduler = None


@app.on_event("shutdown")
def shutdown():
    if getattr(app.state, "background_scheduler", None):
        try:
            app.state.background_scheduler.shutdown(wait=False)
        except Exception:
            pass
    if hasattr(app.state, "db_connection") and app.state.db_connection:
        app.state.db_connection.close()
    if getattr(app.state, "graph_store", None):
        app.state.graph_store.close()


app.include_router(memory_router)
app.include_router(consolidate_router)
app.include_router(status_router)


@app.get("/health")
def health():
    return {"status": "ok", "service": "cortexos"}
