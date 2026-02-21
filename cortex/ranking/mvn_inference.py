"""Score candidates with MVN; attach mvn_score to each candidate."""
from __future__ import annotations

from typing import List, Optional

import torch

from cortex.ranking.mvn_features import build_mvn_features, build_mvn_feature_dim
from cortex.ranking.mvn_model import MVN
from cortex.retrieval.candidate_builder import Candidate
from cortex.retrieval.intent import detect_intent_simple


INTENT_IDS = {"recall": 0, "reasoning": 1, "personal": 2, "knowledge": 3, "planning": 4}


def score_candidates(
    query: str,
    candidates: List[Candidate],
    model: Optional[MVN] = None,
    device: Optional[str] = None,
) -> List[Candidate]:
    """
    Build feature matrix for candidates, run MVN, attach mvn_score to each candidate.
    If model is None, leave mvn_score as None (pipeline will use simple score).
    """
    if not candidates:
        return candidates
    intent = detect_intent_simple(query)
    intent_id = INTENT_IDS.get(intent, 0)
    features_list = []
    for c in candidates:
        feats = build_mvn_features(
            query,
            candidate=c,
            pagerank=c.pagerank,
            graph_distance=c.graph_score,
            intent_type=intent_id,
        )
        features_list.append(feats)
    if not features_list:
        return candidates
    dim = build_mvn_feature_dim()
    for f in features_list:
        if len(f) != dim:
            f.extend([0.0] * (dim - len(f)))
    x = torch.tensor(features_list, dtype=torch.float32)
    if device:
        x = x.to(device)
    if model is not None:
        model.eval()
        with torch.no_grad():
            scores = model(x)
        if device and next(model.parameters()).is_cuda:
            scores = scores.cpu()
        for i, c in enumerate(candidates):
            c.mvn_score = float(scores[i].item())
    return candidates
