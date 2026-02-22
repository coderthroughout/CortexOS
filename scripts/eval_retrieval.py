#!/usr/bin/env python3
"""
Retrieval evaluation: Recall@K, MRR over (query, user_id, relevant_memory_ids).
Usage:
  python scripts/eval_retrieval.py queries.json [--regression] [--min-recall 0.5] [--min-mrr 0.3]
  Exit 1 if --regression and metrics below thresholds (for CI).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import UUID

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

from cortex.training.benchmark import recall_at_k, mrr
from cortex.retrieval.retrieval_pipeline import retrieve_with_hybrid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("queries_file", type=Path, help="JSON: { \"queries\": [ {\"query\", \"user_id\", \"relevant_memory_ids\"} ] }")
    parser.add_argument("--regression", action="store_true", help="Exit 1 if below thresholds")
    parser.add_argument("--min-recall", type=float, default=0.5)
    parser.add_argument("--min-mrr", type=float, default=0.3)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    data = json.loads(args.queries_file.read_text())
    queries = data.get("queries", data) if isinstance(data, dict) else data
    if not queries:
        print("No queries in file.")
        return 0 if not args.regression else 1

    import psycopg2
    from pgvector.psycopg2 import register_vector
    from cortex.utils.config import DATABASE_URL
    from cortex.memory.store import MemoryStore
    from cortex.memory.vector_index import VectorIndex

    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    store = MemoryStore(db_connection=conn)
    vector_index = VectorIndex(db_connection=conn)
    bm25_index = None
    try:
        from cortex.retrieval.bm25_index import BM25Index
        pairs = store.get_all_memory_summaries()
        bm25_index = BM25Index()
        if pairs:
            bm25_index.build([p[0] for p in pairs], [p[1] for p in pairs])
    except Exception:
        bm25_index = None
    graph_store = None
    try:
        from cortex.graph.graph_store import GraphStore
        from cortex.utils.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
        graph_store = GraphStore(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)
    except Exception:
        pass
    mvn_model = None
    try:
        import os
        from cortex.ranking.mvn_model import load_mvn
        from cortex.ranking.mvn_features import build_mvn_feature_dim
        p = os.environ.get("CORTEX_MVN_CHECKPOINT")
        if p:
            mvn_model = load_mvn(path=p, input_dim=build_mvn_feature_dim())
    except Exception:
        pass

    recall5_sum, recall10_sum, mrr_sum = 0.0, 0.0, 0.0
    n = len(queries)
    for q in queries:
        query = q.get("query", "")
        user_id = q.get("user_id")
        relevant = [str(x) for x in q.get("relevant_memory_ids", [])]
        if not query:
            continue
        uid = UUID(user_id) if user_id else None
        try:
            candidates = retrieve_with_hybrid(
                query, uid, vector_index, store,
                bm25_index=bm25_index, graph_store=graph_store, mvn_model=mvn_model,
                k=args.k,
            )
            retrieved = [str(c.memory.id) for c in candidates]
        except Exception as e:
            print(f"Query failed: {query[:50]}... -> {e}")
            retrieved = []
        recall5_sum += recall_at_k(retrieved, relevant, 5)
        recall10_sum += recall_at_k(retrieved, relevant, 10)
        mrr_sum += mrr(retrieved, relevant)

    conn.close()
    if graph_store:
        graph_store.close()

    recall5 = recall5_sum / n if n else 0
    recall10 = recall10_sum / n if n else 0
    mrr_avg = mrr_sum / n if n else 0
    print(f"Queries: {n}")
    print(f"Recall@5:  {recall5:.4f}")
    print(f"Recall@10: {recall10:.4f}")
    print(f"MRR:      {mrr_avg:.4f}")

    if args.regression:
        if recall5 < args.min_recall or mrr_avg < args.min_mrr:
            print(f"Regression: below threshold (min-recall={args.min_recall}, min-mrr={args.min_mrr})")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
