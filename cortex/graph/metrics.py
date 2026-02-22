"""Graph metrics (PageRank, degree) for Memory nodes. Run in background; cache in graph_metrics table."""
from __future__ import annotations

from typing import Any, Dict, List

from cortex.graph.graph_store import GraphStore


def compute_degree_per_memory(graph_store: GraphStore) -> Dict[str, int]:
    """Return {memory_id: degree} for each Memory node (relationship count)."""
    driver = graph_store._get_driver()
    out: Dict[str, int] = {}
    with driver.session() as session:
        result = session.run(
            """
            MATCH (m:Memory)
            OPTIONAL MATCH (m)-[r]-()
            WITH m, count(r) AS degree
            RETURN m.id AS mem_id, degree
            """
        )
        for r in result:
            mid = r.get("mem_id")
            if mid:
                out[str(mid)] = int(r.get("degree") or 0)
    return out


def compute_pagerank_memory(graph_store: GraphStore) -> Dict[str, float]:
    """
    PageRank over Memory nodes. Uses GDS if a 'memory-graph' is projected; else fallback to degree-normalized.
    Returns {memory_id: pagerank_score}.
    """
    driver = graph_store._get_driver()
    with driver.session() as session:
        try:
            # GDS requires a projected graph; if not present, fall through to degree fallback
            exists_result = session.run("CALL gds.graph.exists('memory-graph') YIELD exists")
            rec = exists_result.single()
            if rec and rec.get("exists"):
                pr_result = session.run(
                    """
                    CALL gds.pageRank.stream('memory-graph', { maxIterations: 20, dampingFactor: 0.85 })
                    YIELD nodeId, score
                    RETURN gds.util.asNode(nodeId).id AS mem_id, score
                    """
                )
                out = {str(r["mem_id"]): float(r.get("score") or 0) for r in pr_result if r.get("mem_id")}
                if out:
                    return out
        except Exception:
            pass
    # Fallback: use degree as proxy (normalize 0-1 by max degree)
    degree = compute_degree_per_memory(graph_store)
    if not degree:
        return {}
    max_deg = max(degree.values()) or 1
    return {mid: (d / max_deg) for mid, d in degree.items()}


def compute_graph_metrics(graph_store: GraphStore) -> Dict[str, dict]:
    """
    Compute pagerank and degree for each Memory node.
    Returns {memory_id: {"pagerank": float, "degree": int}}.
    """
    degree = compute_degree_per_memory(graph_store)
    pagerank = compute_pagerank_memory(graph_store)
    memory_ids = set(degree.keys()) | set(pagerank.keys())
    out = {}
    for mid in memory_ids:
        out[mid] = {
            "pagerank": pagerank.get(mid, 0.0),
            "degree": degree.get(mid, 0),
        }
    return out
