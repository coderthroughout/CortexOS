#!/usr/bin/env python3
"""
Train MVN from feedback_logs. Run after collecting feedback via POST /memory/feedback.
Output: checkpoint at CORTEX_MVN_CHECKPOINT or checkpoints/mvn.pt
Usage: python scripts/train_mvn.py [--limit 5000] [--save path] [--epochs 10]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from uuid import UUID

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

import random
import psycopg2
from cortex.memory.store import MemoryStore
from cortex.training.mvn_dataset import MVNDataset
from cortex.training.mvn_train import train_mvn
from cortex.ranking.mvn_features import build_mvn_features, build_mvn_feature_dim
from cortex.retrieval.intent import detect_intent_simple
from cortex.utils.config import DATABASE_URL

INTENT_IDS = {"recall": 0, "reasoning": 1, "personal": 2, "knowledge": 3, "planning": 4}


def build_synthetic_samples(store: MemoryStore, min_samples: int = 20, max_memories: int = 200) -> list:
    """Build (pos_features, neg_features) from store when feedback log ids don't resolve. Uses each memory as a query proxy; positive = self, negatives = random others."""
    memories = []
    for uid in store.get_user_ids(limit=20):
        memories.extend(store.get_user_memories(uid, limit=50))
    seen = {m.id: m for m in memories}
    memories = [m for m in seen.values() if getattr(m, "embedding", None) and len(m.embedding) > 0]
    if len(memories) < 2:
        return []
    random.shuffle(memories)
    memories = memories[:max_memories]
    samples = []
    for m in memories:
        others = [x for x in memories if x.id != m.id]
        if not others:
            continue
        query = (m.summary or "")[:200]
        intent_id = INTENT_IDS.get(detect_intent_simple(query), 0)
        pos_feats = build_mvn_features(query, memory=m, intent_type=intent_id)
        if not pos_feats:
            continue
        negs = random.sample(others, min(5, len(others)))
        neg_feats = []
        for n in negs:
            f = build_mvn_features(query, memory=n, intent_type=intent_id)
            if f:
                neg_feats.append(f)
        if neg_feats:
            samples.append({"pos_features": pos_feats, "neg_features": neg_feats})
        if len(samples) >= min_samples:
            break
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description="Train MVN from feedback logs.")
    parser.add_argument("--limit", type=int, default=5000, help="Max feedback log rows")
    parser.add_argument("--save", type=str, default=None, help="Checkpoint path (default: checkpoints/mvn.pt or CORTEX_MVN_CHECKPOINT)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--verbose", "-v", action="store_true", help="Print first log entry and resolve check")
    args = parser.parse_args()

    save_path = args.save or os.environ.get("CORTEX_MVN_CHECKPOINT") or "checkpoints/mvn.pt"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    conn = psycopg2.connect(DATABASE_URL)
    store = MemoryStore(db_connection=conn)

    logs = store.get_feedback_logs(limit=args.limit)

    def get_memory_fn(memory_id):
        try:
            mid = str(memory_id) if memory_id is not None else None
            if not mid:
                return None
            return store.get_memory(UUID(mid))
        except Exception:
            return None

    if args.verbose and logs:
        first = logs[0]
        print(f"First log: query={first.get('query', '')[:50]!r}, retrieved_count={len(first.get('retrieved_memory_ids') or [])}, used_count={len(first.get('used_memory_ids') or [])}")
        pos_id = (first.get("used_memory_ids") or first.get("retrieved_memory_ids") or [None])[0]
        if pos_id is not None:
            mem = get_memory_fn(pos_id)
            print(f"  get_memory({pos_id!r}) -> {'OK' if mem else 'None'}")

    train_samples = []
    if logs:
        dataset = MVNDataset()
        build_entries = list(dataset.build(logs))
        if build_entries:
            train_samples = dataset.build_feature_samples(logs, get_memory_fn=get_memory_fn)
    if not train_samples:
        print("Using synthetic training data from memories in store (feedback log empty or ids did not resolve).")
        train_samples = build_synthetic_samples(store, min_samples=15, max_memories=150)
    if not train_samples:
        print("No valid training samples and no memories with embeddings in store. Add memories then re-run.")
        conn.close()
        return 0

    print(f"Training MVN on {len(train_samples)} samples.")
    model = train_mvn(
        train_samples,
        feature_dim=build_mvn_feature_dim(),
        epochs=args.epochs,
        batch_size=args.batch_size,
        margin=args.margin,
        save_path=str(save_path),
    )
    # Persist last run time for observability (GET /status)
    try:
        from datetime import datetime, timezone
        stamp_path = save_path.parent / ".last_mvn_training"
        stamp_path.write_text(datetime.now(timezone.utc).isoformat())
    except Exception:
        pass
    print(f"Saved checkpoint to {save_path}. Set CORTEX_MVN_CHECKPOINT={save_path} and restart the API to use it.")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
