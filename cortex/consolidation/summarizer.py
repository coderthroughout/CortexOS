"""LLM summarization of memory clusters into semantic memory."""
from __future__ import annotations

import os
from typing import Any, List, Optional

import httpx
from cortex.memory.schema import Memory


def llm_summarize(texts: List[str], api_base: Optional[str] = None, api_key: Optional[str] = None) -> str:
    """Single summary from multiple texts. Prompt: 'Summarize these events into a stable long-term insight.'"""
    api_base = api_base or os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return " ".join(texts[:3])  # fallback: first 3
    content = "\n".join(f"- {t}" for t in texts[:20])
    prompt = f"Summarize these events into a stable long-term insight. Be concise.\n\n{content}"
    url = f"{api_base.rstrip('/')}/chat/completions"
    payload = {
        "model": os.environ.get("CORTEX_CONSOLIDATION_MODEL", "gpt-4o-mini"),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }
    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.post(url, json=payload, headers={"Authorization": f"Bearer {api_key}"})
            r.raise_for_status()
            return (r.json().get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
    except Exception:
        return " ".join(texts[:2])


def create_semantic_memory(
    cluster: List[Memory],
    store: Any,
    vector_index: Any,
    graph_store: Optional[Any] = None,
    user_id=None,
) -> Optional[Memory]:
    """Summarize cluster -> new semantic memory with provenance; store and link in graph."""
    from cortex.memory.schema import MemoryCreate, MemoryType
    from cortex.utils.embeddings import embed
    from cortex.graph.graph_builder import build_graph_for_memory
    texts = [m.summary or m.text for m in cluster]
    summary = llm_summarize(texts)
    if not summary:
        return None
    provenance = [str(m.id) for m in cluster]
    user_id = user_id or (cluster[0].user_id if cluster else None)
    if not user_id:
        return None
    create = MemoryCreate(
        user_id=user_id,
        type=MemoryType.SEMANTIC,
        text=summary,
        summary=summary,
        entities=[],
        importance=0.6,
        source=cluster[0].source if cluster else None,
    )
    emb = embed(summary)
    memory = store.add_memory(create, embedding=emb)
    vector_index.add(memory.id, emb)
    if graph_store:
        build_graph_for_memory(
            graph_store,
            memory_id=memory.id,
            user_id=user_id,
            summary=summary,
            memory_type="semantic",
            entities=[],
            relationships=[],  # could add DERIVED_FROM to cluster memories
        )
    return memory
