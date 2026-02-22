#!/usr/bin/env python3
"""Populate graph_metrics cache (pagerank, degree) from Neo4j. Run after consolidation or on a schedule."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

import psycopg2
from cortex.graph.graph_store import GraphStore
from cortex.graph.metrics import compute_graph_metrics
from cortex.memory.store import MemoryStore
from cortex.utils.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, DATABASE_URL


def main() -> int:
    graph_store = GraphStore(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)
    conn = psycopg2.connect(DATABASE_URL)
    store = MemoryStore(db_connection=conn)
    try:
        metrics = compute_graph_metrics(graph_store)
        if not metrics:
            print("No Memory nodes in graph; nothing to update.")
            return 0
        # Convert str keys to UUID for store
        by_uuid = {UUID(mid): v for mid, v in metrics.items()}
        store.set_graph_metrics_bulk(by_uuid)
        print(f"Updated graph_metrics for {len(by_uuid)} memories.")
    finally:
        graph_store.close()
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
