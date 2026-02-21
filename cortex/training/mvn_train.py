"""MVN training: pairwise ranking loss; optional bootstrap from heuristics."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from cortex.ranking.mvn_model import MVN
from cortex.training.mvn_dataset import MVNDataset, build_mvn_feature_dim


class MVNTrainDataset(Dataset):
    """In-memory dataset of (pos_features, neg_features) for pairwise loss."""

    def __init__(self, samples: List[Dict], feature_dim: int):
        self.feature_dim = feature_dim
        self.pairs: List[tuple] = []
        for s in samples:
            pos = s.get("pos_features")
            negs = s.get("neg_features", [])
            if not pos or not negs:
                continue
            for n in negs:
                if len(pos) == feature_dim and len(n) == feature_dim:
                    self.pairs.append((pos, n))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        pos, neg = self.pairs[i]
        return torch.tensor(pos, dtype=torch.float32), torch.tensor(neg, dtype=torch.float32)


def train_mvn(
    train_samples: List[Dict],
    feature_dim: Optional[int] = None,
    hidden_dim: int = 64,
    margin: float = 0.2,
    lr: float = 1e-3,
    epochs: int = 10,
    batch_size: int = 32,
    save_path: Optional[str] = None,
    device: Optional[str] = None,
) -> MVN:
    """
    train_samples: list of {"pos_features": [...], "neg_features": [[...], ...]}.
    Returns trained MVN model.
    """
    dim = feature_dim or build_mvn_feature_dim()
    dataset = MVNTrainDataset(train_samples, dim)
    if len(dataset) == 0:
        model = MVN(input_dim=dim, hidden_dim=hidden_dim)
        if save_path:
            torch.save({"state_dict": model.state_dict()}, save_path)
        return model
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = MVN(input_dim=dim, hidden_dim=hidden_dim)
    if device:
        model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0.0
        for pos, neg in loader:
            if device:
                pos, neg = pos.to(device), neg.to(device)
            score_pos = model(pos)
            score_neg = model(neg)
            loss = nn.functional.relu(margin - score_pos + score_neg).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict()}, save_path)
    return model
