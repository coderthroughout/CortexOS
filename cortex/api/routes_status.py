"""Status/debug endpoint: component health and optional recent timings."""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Request

router = APIRouter(tags=["status"])


@router.get("/status")
def status(request: Request) -> Dict[str, Any]:
    """
    Component health: embedder_loaded, postgres_ok, neo4j_ok, mvn_loaded.
    Safe for production (no secrets).
    """
    out: Dict[str, Any] = {
        "embedder_loaded": False,
        "postgres_ok": False,
        "neo4j_ok": False,
        "mvn_loaded": False,
    }
    # Embedder
    try:
        from cortex.utils.embeddings import embed
        embed("ok")
        out["embedder_loaded"] = True
    except Exception:
        pass
    # Postgres: try existing conn, then fresh connection (for debugging deployment)
    try:
        conn = getattr(request.app.state, "db_connection", None)
        if conn:
            cur = conn.cursor()
            cur.execute("SELECT 1")
            cur.fetchone()
            cur.close()
            out["postgres_ok"] = True
    except Exception:
        try:
            from cortex.utils.config import DATABASE_URL
            import psycopg2
            fresh = psycopg2.connect(DATABASE_URL)
            fresh.cursor().execute("SELECT 1")
            fresh.close()
            out["postgres_ok"] = True
        except Exception as e:
            out["postgres_error"] = (str(e)[:120] or type(e).__name__)
    if not out.get("postgres_ok") and "postgres_error" not in out:
        out["postgres_error"] = "no db_connection and CORTEX_DATABASE_URL not tried"
    # Neo4j
    try:
        gs = getattr(request.app.state, "graph_store", None)
        if gs and gs._get_driver():
            with gs._get_driver().session() as session:
                session.run("RETURN 1")
            out["neo4j_ok"] = True
    except Exception:
        pass
    # MVN
    out["mvn_loaded"] = getattr(request.app.state, "mvn_model", None) is not None
    # Observability: feedback volume and last MVN training
    try:
        store = getattr(request.app.state, "memory_store", None)
        if store:
            out["feedback_events_last_24h"] = store.count_feedback_last_24h()
    except Exception:
        out["feedback_events_last_24h"] = None
    try:
        import os
        from pathlib import Path
        checkpoint_dir = Path(os.environ.get("CORTEX_MVN_CHECKPOINT", "checkpoints/mvn.pt")).parent
        stamp_file = checkpoint_dir / ".last_mvn_training"
        out["last_mvn_training_run"] = stamp_file.read_text().strip() if stamp_file.exists() else None
    except Exception:
        out["last_mvn_training_run"] = None
    return out
