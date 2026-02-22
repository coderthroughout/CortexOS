#!/usr/bin/env python3
"""
Downstream evaluation: given query + top-k retrieved memories, build context, get LLM answer, run LLM judge.
Usage: python scripts/eval_downstream_judge.py queries.json [--model gpt-4o-mini]
Requires OPENAI_API_KEY. Output: average judge score over held-out set.
"""
from __future__ import annotations

import argparse
import json
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


def _build_context(memories: list) -> str:
    """Format memories as context string."""
    return "\n".join(f"- {getattr(m, 'summary', m) if hasattr(m, 'summary') else m}" for m in memories)


def _get_answer(query: str, context: str, model: str) -> str:
    """Call LLM to answer query given context."""
    try:
        import httpx
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            return ""
        resp = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "Answer based only on the provided memory context."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuery: {query}"},
                ],
                "max_tokens": 300,
            },
            timeout=30,
        )
        if resp.status_code != 200:
            return ""
        data = resp.json()
        return (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
    except Exception:
        return ""


def _judge(query: str, context: str, answer: str, model: str) -> float:
    """LLM judge: Did the answer correctly use the provided memories? Return 0-1."""
    try:
        import httpx
        key = os.environ.get("OPENAI_API_KEY")
        if not key or not answer:
            return 0.5
        prompt = (
            f"Query: {query}\n\nRelevant memories:\n{context}\n\nAnswer: {answer}\n\n"
            "Did the answer correctly use the provided memories? Reply with a number 0 (no) to 1 (yes)."
        )
        resp = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 10,
            },
            timeout=15,
        )
        if resp.status_code != 200:
            return 0.5
        text = (resp.json().get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
        for t in text.replace(",", ".").split():
            try:
                v = float(t)
                return max(0, min(1, v))
            except ValueError:
                continue
        return 0.5
    except Exception:
        return 0.5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("queries_file", type=Path)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    data = json.loads(args.queries_file.read_text())
    queries = data.get("queries", data) if isinstance(data, dict) else data
    if not queries:
        print("No queries.")
        return 0

    # Load retrieval pipeline in-process to get top-k for each query
    import psycopg2
    from pgvector.psycopg2 import register_vector
    from cortex.utils.config import DATABASE_URL
    from cortex.memory.store import MemoryStore
    from cortex.memory.vector_index import VectorIndex
    from cortex.retrieval.retrieval_pipeline import retrieve_with_hybrid

    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    store = MemoryStore(db_connection=conn)
    vector_index = VectorIndex(db_connection=conn)
    bm25_index, graph_store, mvn_model = None, None, None
    try:
        from cortex.retrieval.bm25_index import BM25Index
        pairs = store.get_all_memory_summaries()
        bm25_index = BM25Index()
        if pairs:
            bm25_index.build([p[0] for p in pairs], [p[1] for p in pairs])
    except Exception:
        pass
    try:
        from cortex.graph.graph_store import GraphStore
        from cortex.utils.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
        graph_store = GraphStore(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)
    except Exception:
        pass
    try:
        import os as _os
        from cortex.ranking.mvn_model import load_mvn
        from cortex.ranking.mvn_features import build_mvn_feature_dim
        if _os.environ.get("CORTEX_MVN_CHECKPOINT"):
            mvn_model = load_mvn(path=_os.environ["CORTEX_MVN_CHECKPOINT"], input_dim=build_mvn_feature_dim())
    except Exception:
        pass

    scores = []
    for q in queries:
        query = q.get("query", "")
        user_id = q.get("user_id")
        if not query:
            continue
        uid = UUID(user_id) if user_id else None
        try:
            candidates = retrieve_with_hybrid(
                query, uid, vector_index, store,
                bm25_index=bm25_index, graph_store=graph_store, mvn_model=mvn_model,
                k=args.k,
            )
            memories = [c.memory for c in candidates]
        except Exception:
            memories = []
        context = _build_context(memories)
        answer = _get_answer(query, context, args.model)
        score = _judge(query, context, answer, args.model)
        scores.append(score)

    conn.close()
    if graph_store:
        graph_store.close()

    avg = sum(scores) / len(scores) if scores else 0
    print(f"Queries: {len(scores)}")
    print(f"Downstream judge score (0-1): {avg:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
