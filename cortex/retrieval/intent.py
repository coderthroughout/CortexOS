"""Simple intent classification for query routing (recall, reasoning, personal, knowledge, planning)."""
from __future__ import annotations

import re
from typing import Literal

IntentType = Literal["recall", "reasoning", "personal", "knowledge", "planning"]


def detect_intent_simple(query: str) -> IntentType:
    """Heuristic intent from keywords. Can be replaced by LLM classifier."""
    q = query.lower().strip()
    if re.search(r"\b(why|how|what caused|reason|because)\b", q):
        return "reasoning"
    if re.search(r"\b(remember|recall|when did i|did i ever|last time)\b", q):
        return "recall"
    if re.search(r"\b(my|i |me |we |our)\b", q) or re.search(r"\b(stressed|worried|feel|want|prefer)\b", q):
        return "personal"
    if re.search(r"\b(plan|schedule|next|tomorrow|will i)\b", q):
        return "planning"
    return "knowledge"
