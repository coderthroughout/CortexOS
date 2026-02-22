"""Optional in-process scheduler for consolidation and graph metrics. Enable with CORTEX_BACKGROUND_JOBS=1."""
from __future__ import annotations

import logging
import os
from typing import Any

_log = logging.getLogger("cortexos.background")


def _run_consolidation_job(app: Any) -> None:
    """Run consolidation for each user that has memories."""
    try:
        store = getattr(app.state, "memory_store", None)
        vector_index = getattr(app.state, "vector_index", None)
        graph_store = getattr(app.state, "graph_store", None)
        if not store or not vector_index:
            return
        from cortex.consolidation.sleep_worker import run_consolidation
        for user_id in store.get_user_ids(limit=50):
            try:
                run_consolidation(user_id, store, vector_index, graph_store)
            except Exception as e:
                _log.warning("Consolidation for user %s failed: %s", user_id, e)
    except Exception as e:
        _log.warning("Background consolidation job failed: %s", e)


def _run_graph_metrics_job(app: Any) -> None:
    """Update graph_metrics cache from Neo4j."""
    try:
        graph_store = getattr(app.state, "graph_store", None)
        store = getattr(app.state, "memory_store", None)
        if not graph_store or not store:
            return
        from cortex.graph.metrics import compute_graph_metrics
        from uuid import UUID
        metrics = compute_graph_metrics(graph_store)
        if not metrics:
            return
        by_uuid = {UUID(mid): v for mid, v in metrics.items()}
        store.set_graph_metrics_bulk(by_uuid)
        _log.info("Graph metrics updated for %d memories.", len(by_uuid))
    except Exception as e:
        _log.warning("Background graph metrics job failed: %s", e)


def start_background_scheduler(app: Any) -> Any:
    """
    Start APScheduler with consolidation and graph metrics jobs if CORTEX_BACKGROUND_JOBS=1.
    Returns the scheduler instance or None.
    """
    if os.environ.get("CORTEX_BACKGROUND_JOBS", "").strip() != "1":
        return None
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        scheduler = BackgroundScheduler()
        scheduler.add_job(
            _run_consolidation_job,
            "interval",
            hours=6,
            id="consolidation",
            args=[app],
            replace_existing=True,
        )
        scheduler.add_job(
            _run_graph_metrics_job,
            "interval",
            hours=6,
            id="graph_metrics",
            args=[app],
            replace_existing=True,
        )
        scheduler.start()
        _log.info("Background scheduler started (consolidation + graph_metrics every 6h).")
        return scheduler
    except Exception as e:
        _log.warning("Could not start background scheduler: %s", e)
        return None
