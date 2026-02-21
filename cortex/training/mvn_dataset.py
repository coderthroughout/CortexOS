"""Build MVN training samples (query, positive, negatives) from interaction logs."""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, List, Optional
from uuid import UUID

from cortex.ranking.mvn_features import build_mvn_features, build_mvn_feature_dim
from cortex.retrieval.candidate_builder import Candidate
from cortex.retrieval.intent import detect_intent_simple

INTENT_IDS = {"recall": 0, "reasoning": 1, "personal": 2, "knowledge": 3, "planning": 4}


def build_sample(
    query: str,
    positive_memory: Any,
    negative_memories: List[Any],
    intent_id: int = 0,
) -> Dict[str, Any]:
    """Build one training sample: query, pos_features, neg_features (list)."""
    pos_candidate = positive_memory if isinstance(positive_memory, Candidate) else None
    pos_memory = getattr(positive_memory, "memory", positive_memory)
    pos_feats = build_mvn_features(
        query,
        candidate=pos_candidate,
        memory=pos_memory if not pos_candidate else None,
        intent_type=intent_id,
    )
    neg_feats_list = []
    for neg in negative_memories:
        c = neg if isinstance(neg, Candidate) else None
        m = getattr(neg, "memory", neg)
        neg_feats_list.append(
            build_mvn_features(query, candidate=c, memory=m if not c else None, intent_type=intent_id)
        )
    return {"query": query, "pos_features": pos_feats, "neg_features": neg_feats_list}


class MVNDataset:
    """Create samples from interaction logs."""

    def __init__(self, feature_dim: Optional[int] = None):
        self.feature_dim = feature_dim or build_mvn_feature_dim()

    def build(
        self,
        logs: List[Dict],
    ) -> Iterator[Dict[str, Any]]:
        """
        logs: list of {
            "query": str,
            "used_memory_ids": list,
            "retrieved_memory_ids": list,
            "reward": float,
        }
        Yields samples with pos = used+high reward, neg = retrieved but not used or low reward.
        """
        for entry in logs:
            query = entry.get("query", "")
            used = set(entry.get("used_memory_ids") or [])
            retrieved = entry.get("retrieved_memory_ids") or []
            reward = float(entry.get("reward", 0.5))
            if not retrieved:
                continue
            # Positive: one of the used (or highest reward retrieved)
            positives = [r for r in retrieved if r in used]
            negatives = [r for r in retrieved if r not in used]
            if not positives and not negatives:
                continue
            if not positives:
                positives = [retrieved[0]]
            if not negatives:
                negatives = retrieved[1:2] or [retrieved[0]]
            intent_id = INTENT_IDS.get(detect_intent_simple(query), 0)
            yield {
                "query": query,
                "pos_memory_id": positives[0],
                "neg_memory_ids": negatives[:5],
                "reward": reward,
                "intent_id": intent_id,
            }

    def build_feature_samples(
        self,
        logs: List[Dict],
        get_memory_fn: Callable,
        get_candidate_fn: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        """
        From logs with memory ids, resolve to Memory/Candidate and build pos_features, neg_features.
        get_memory_fn(memory_id) -> Memory. get_candidate_fn(query, memory_id) -> optional Candidate.
        """
        out = []
        for entry in self.build(logs):
            query = entry["query"]
            pos_id = entry["pos_memory_id"]
            neg_ids = entry["neg_memory_ids"]
            intent_id = entry["intent_id"]
            pos_mem = get_memory_fn(pos_id)
            if not pos_mem:
                continue
            pos_feats = build_mvn_features(query, memory=pos_mem, intent_type=intent_id)
            neg_feats = []
            for nid in neg_ids:
                m = get_memory_fn(nid)
                if m:
                    neg_feats.append(build_mvn_features(query, memory=m, intent_type=intent_id))
            if neg_feats:
                out.append({"pos_features": pos_feats, "neg_features": neg_feats})
        return out
