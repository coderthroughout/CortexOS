"""Entity extraction and resolution for graph and metadata."""
from __future__ import annotations

import re
from typing import List, Optional, Set


# Simple heuristic: capitalized phrases and known patterns. Can be replaced by NER/LLM.
def extract_entities(text: str, existing: Optional[List[str]] = None) -> List[str]:
    """
    Extract entity-like tokens from text. Returns deduplicated list.
    existing: optional list to merge with (e.g. from extractor output).
    """
    out: Set[str] = set()
    if existing:
        for e in existing:
            if isinstance(e, str) and e.strip():
                out.add(e.strip())

    # Words that are typically not entities
    stop = {"i", "me", "my", "we", "you", "the", "a", "an", "and", "or", "but", "is", "was", "are", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "can", "may", "might", "this", "that", "these", "those", "it", "its"}

    # Consecutive capitalized words (simple NER)
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text):
        candidate = m.group(1).strip()
        if candidate.lower() not in stop and len(candidate) > 1:
            out.add(candidate)

    # Single capitalized word (e.g. product names)
    for m in re.finditer(r"\b([A-Z][a-zA-Z0-9]{1,30})\b", text):
        candidate = m.group(1)
        if candidate.lower() not in stop:
            out.add(candidate)

    return sorted(out)


def resolve_entities(entities: List[str], known_aliases: Optional[dict] = None) -> List[str]:
    """
    Normalize entity names (e.g. map aliases to canonical form).
    known_aliases: optional dict mapping alias -> canonical name.
    """
    if not known_aliases:
        return list(dict.fromkeys(entities))
    return list(dict.fromkeys(known_aliases.get(e, e) for e in entities))
