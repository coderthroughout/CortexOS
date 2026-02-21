"""Stubs for synthetic conversation and query generation, LLM judge labeling."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import httpx


def generate_conversations(
    num_days: int = 10,
    api_key: Optional[str] = None,
) -> List[Dict]:
    """Generate multi-session chat logs (stub: returns empty; can call LLM to generate)."""
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return []
    prompt = f"Generate a {num_days}-day conversation history between a startup founder and an AI assistant. Include goals, stress, decisions, and events. Return as JSON array of {{day, messages: [{{role, content}}]}}."
    # Stub: real impl would call LLM and parse
    return []


def generate_queries(conversation_logs: List[Dict], num_queries: int = 20) -> List[Dict]:
    """Generate questions that require remembering past events. Stub."""
    return [{"query": "Why was I stressed last week?", "expected_memory_ids": []} for _ in range(min(num_queries, 5))]


def llm_judge_label(query: str, retrieved_ids: List[str], answer: str, api_key: Optional[str] = None) -> List[str]:
    """Which memory IDs were required to answer? Returns subset of retrieved_ids. Stub."""
    return retrieved_ids[:1] if retrieved_ids else []
