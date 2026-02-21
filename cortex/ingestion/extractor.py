"""Memory extractor: conversation -> LLM -> parsed JSON memories."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, List, Optional

import httpx
from cortex.ingestion.parser import RawInput
from cortex.ingestion.prompts import EXTRACTION_SYSTEM, EXTRACTION_USER_TEMPLATE


def _llm_chat(system: str, user: str, api_base: Optional[str] = None, api_key: Optional[str] = None) -> str:
    """Call OpenAI-compatible chat API. Uses OPENAI_API_KEY and OPENAI_API_BASE env if not passed."""
    api_base = api_base or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set; cannot call LLM for extraction")
    url = f"{api_base.rstrip('/')}/chat/completions"
    payload = {
        "model": os.environ.get("CORTEX_EXTRACTION_MODEL", "gpt-4o-mini"),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
    }
    with httpx.Client(timeout=60.0) as client:
        r = client.post(url, json=payload, headers={"Authorization": f"Bearer {api_key}"})
        r.raise_for_status()
        data = r.json()
        return (data.get("choices") or [{}])[0].get("message", {}).get("content", "")


def _parse_json_array(text: str) -> List[dict]:
    """Extract a JSON array from LLM output (may be wrapped in markdown)."""
    text = text.strip()
    # Remove markdown code block if present
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1).strip()
    # Find [ ... ]
    start = text.find("[")
    if start == -1:
        return []
    depth = 0
    end = -1
    for i, c in enumerate(text[start:], start=start):
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        return []
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []


def extract_memories(
    conversation: str | RawInput,
    llm_fn: Optional[Callable[[str, str], str]] = None,
) -> List[dict]:
    """
    Extract atomic memories from conversation text.
    conversation: either raw string or RawInput (use .content).
    llm_fn: optional (system, user) -> response; else uses OpenAI-compatible API.
    Returns list of dicts with type, summary, entities, importance, emotion, time_reference, relationships.
    """
    if isinstance(conversation, RawInput):
        content = conversation.content
    else:
        content = conversation
    if not content.strip():
        return []
    user_prompt = EXTRACTION_USER_TEMPLATE.format(chat=content)
    if llm_fn is not None:
        response = llm_fn(EXTRACTION_SYSTEM, user_prompt)
    else:
        response = _llm_chat(EXTRACTION_SYSTEM, user_prompt)
    items = _parse_json_array(response)
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        rec = {
            "memory_type": item.get("type", "event"),
            "summary": item.get("summary", "").strip(),
            "entities": item.get("entities") or [],
            "importance": float(item.get("importance", 0.5)),
            "emotion": item.get("emotion"),
            "timestamp_reference": item.get("time_reference"),
            "relationships": item.get("relationships") or [],
        }
        if rec["summary"]:
            out.append(rec)
    return out
