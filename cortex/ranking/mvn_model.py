"""Memory Value Network (MVN): 2-layer MLP predicting utility score 0-1."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class MVN(nn.Module):
    """MVP: 2-layer MLP. Input: feature vector; output: utility score 0-1."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_mvn(path: Optional[str] = None, input_dim: int = 11, device: Optional[str] = None) -> MVN:
    """Load MVN from checkpoint or create new."""
    model = MVN(input_dim=input_dim)
    if path:
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state.get("state_dict", state))
    if device:
        model = model.to(device)
    return model
