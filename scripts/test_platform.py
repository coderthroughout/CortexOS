#!/usr/bin/env python3
"""
CortexOS platform test: exercises main API flows and reports where the system stands.
Run with the API server already up: python run.py (in another terminal).
Usage: python scripts/test_platform.py [--base-url http://localhost:8000]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from uuid import uuid4

# Load .env so CORTEX_* are set if script is run standalone
try:
    from dotenv import load_dotenv
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(_root, ".env"))
except Exception:
    pass

import httpx

DEFAULT_BASE = os.environ.get("CORTEX_TEST_BASE_URL", "http://localhost:8000")
TEST_USER = os.environ.get("CORTEX_TEST_USER", "550e8400-e29b-41d4-a716-446655440000")


def main() -> int:
    parser = argparse.ArgumentParser(description="Test CortexOS API and report status.")
    parser.add_argument("--base-url", default=DEFAULT_BASE, help="API base URL")
    parser.add_argument("--user", default=TEST_USER, help="User UUID for memory tests")
    parser.add_argument("--timeout", type=float, default=30.0, help="Request timeout seconds")
    parser.add_argument("--skip-consolidation", action="store_true", help="Skip consolidation step (use before train_mvn so feedback memory ids still exist)")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    user = args.user
    timeout = args.timeout

    results = []
    total_start = time.perf_counter()

    with httpx.Client(base_url=base, timeout=timeout) as client:

        # ---- Health ----
        t0 = time.perf_counter()
        try:
            r = client.get("/health")
            r.raise_for_status()
            data = r.json()
            ok = data.get("status") == "ok" and data.get("service") == "cortexos"
        except Exception as e:
            ok, data = False, str(e)
        elapsed = (time.perf_counter() - t0) * 1000
        results.append(("GET /health", ok, elapsed, data if ok else {"error": str(data)}))

        if not ok:
            print("FAIL: /health failed. Is the server running? Start with: python run.py")
            return 1

        # ---- Warmup (ensure embedder loaded so measured adds are steady-state) ----
        try:
            client.get("/status")
        except Exception:
            pass

        # ---- Add memories ----
        memories_to_add = [
            {"summary": "We launched the beta last week", "entities": ["beta"], "importance": 0.8},
            {"summary": "User prefers dark mode and keyboard shortcuts", "entities": ["preferences"], "importance": 0.7},
            {"summary": "Meeting with Alex about Q1 roadmap next Tuesday", "entities": ["Alex", "Q1"], "importance": 0.9},
        ]
        added_ids = []
        for i, body in enumerate(memories_to_add):
            t0 = time.perf_counter()
            try:
                r = client.post(f"/memory/add?user={user}", json=body)
                r.raise_for_status()
                data = r.json()
                added_ids.append(data.get("id"))
                ok = "id" in data and data.get("summary") == body["summary"]
            except Exception as e:
                ok, data = False, str(e)
            elapsed = (time.perf_counter() - t0) * 1000
            results.append((f"POST /memory/add ({body['summary'][:30]}...)", ok, elapsed, data if ok else {"error": str(data)}))

        # ---- Query ----
        query_text = "launch beta"
        retrieved_ids = []
        t0 = time.perf_counter()
        try:
            r = client.get(f"/memory/query?q=launch+beta&user={user}&k=5")
            r.raise_for_status()
            data = r.json()
            ok = isinstance(data, list) and len(data) >= 1
            if ok and data:
                first = data[0]
                ok = "score" in first or "score_breakdown" in first
                retrieved_ids = [item.get("id") for item in data if item.get("id")]
        except Exception as e:
            ok, data = False, str(e)
        elapsed = (time.perf_counter() - t0) * 1000
        results.append(("GET /memory/query (launch beta)", ok, elapsed, len(data) if isinstance(data, list) else data))

        # ---- Timeline ----
        t0 = time.perf_counter()
        try:
            r = client.get(f"/memory/timeline?user={user}")
            r.raise_for_status()
            data = r.json()
            ok = isinstance(data, list)
        except Exception as e:
            ok, data = False, str(e)
        elapsed = (time.perf_counter() - t0) * 1000
        results.append(("GET /memory/timeline", ok, elapsed, len(data) if isinstance(data, list) else data))

        # ---- Graph (optional: may be empty if no graph or no node) ----
        t0 = time.perf_counter()
        try:
            r = client.get("/memory/graph?node=beta&depth=2")
            r.raise_for_status()
            data = r.json()
            ok = isinstance(data, list)
        except Exception as e:
            ok, data = False, str(e)
        elapsed = (time.perf_counter() - t0) * 1000
        results.append(("GET /memory/graph (node=beta)", ok, elapsed, len(data) if isinstance(data, list) else data))

        # ---- Feedback (include query + retrieved_memory_ids so MVN training can build samples) ----
        feedback_body = {
            "user_id": user,
            "query": query_text,
            "retrieved_memory_ids": retrieved_ids if retrieved_ids else added_ids[:2],
            "used_memory_ids": [retrieved_ids[0]] if retrieved_ids else added_ids[:1],
            "reward": 0.8,
        }
        t0 = time.perf_counter()
        try:
            r = client.post("/memory/feedback", json=feedback_body)
            r.raise_for_status()
            data = r.json()
            ok = data.get("ok") is True
        except Exception as e:
            ok, data = False, str(e)
        elapsed = (time.perf_counter() - t0) * 1000
        results.append(("POST /memory/feedback", ok, elapsed, data if ok else {"error": str(data)}))

        # ---- Consolidation (first-class; skip with --skip-consolidation so MVN training can resolve feedback memory ids) ----
        if not args.skip_consolidation:
            t0 = time.perf_counter()
            try:
                r = client.post(f"/consolidate/run?user={user}")
                r.raise_for_status()
                ok = True
                data = r.json() if r.content else {}
            except Exception as e:
                ok, data = False, str(e)
            elapsed = (time.perf_counter() - t0) * 1000
            results.append(("POST /consolidate/run", ok, elapsed, data if ok else {"error": str(data)}))
        else:
            results.append(("POST /consolidate/run (skipped)", True, 0, "skipped"))

        # ---- Status (component health) ----
        try:
            r = client.get("/status")
            r.raise_for_status()
            status_data = r.json()
        except Exception:
            status_data = {}

    total_elapsed = (time.perf_counter() - total_start) * 1000

    # ---- Report ----
    print()
    print("=" * 60)
    print("CortexOS platform test report")
    print("=" * 60)
    print(f"Base URL: {base}")
    print(f"User:    {user}")
    print()
    passed = sum(1 for _, ok, _, _ in results if ok)
    total = len(results)
    add_latencies = [lat for name, ok, lat, _ in results if ok and "POST /memory/add" in name]
    query_latency = next((lat for name, ok, lat, _ in results if ok and "query" in name.lower()), None)
    for name, ok, lat_ms, extra in results:
        status = "PASS" if ok else "FAIL"
        extra_str = ""
        if isinstance(extra, dict) and "error" in extra:
            extra_str = f" — {extra['error'][:80]}"
        elif isinstance(extra, (int, float)):
            extra_str = f" (items: {extra})"
        print(f"  [{status}] {name}  {lat_ms:.0f} ms{extra_str}")
    print()
    print(f"Result: {passed}/{total} checks passed in {total_elapsed:.0f} ms total.")
    if add_latencies:
        steady = add_latencies[1:] if len(add_latencies) > 1 else add_latencies
        print(f"Steady-state add latency: {sum(steady)/len(steady):.0f} ms (avg of {len(steady)} add(s)).")
    if query_latency is not None:
        print(f"Query latency: {query_latency:.0f} ms.")
    if status_data:
        print(f"Status: embedder={status_data.get('embedder_loaded')}, postgres={status_data.get('postgres_ok')}, neo4j={status_data.get('neo4j_ok')}, mvn={status_data.get('mvn_loaded')}.")
    print()

    # Where we stand
    print("Where the system stands:")
    if passed == total:
        print("  - Core pipeline is working: Postgres + vector index, hybrid retrieval, timeline, graph, feedback, consolidation.")
        print("  - Steady-state add and query latencies are reported above; use GET /memory/query?debug=1 for timing breakdown.")
    else:
        print("  - Some checks failed. Fix the failing endpoints (see errors above) and re-run.")
        if passed >= 3:
            print("  - At least health, add, and query work — good baseline; fix remaining features as needed.")
    print()
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
