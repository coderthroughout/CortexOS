"""Full ingestion: parse -> extract -> normalize -> store -> graph update."""
from __future__ import annotations

from typing import Callable, List, Optional
from uuid import UUID

from cortex.graph.graph_builder import build_graph_for_memory
from cortex.graph.graph_store import GraphStore
from cortex.ingestion.extractor import extract_memories
from cortex.ingestion.normalize import normalize_memory
from cortex.ingestion.parser import RawInput, parse_chat, parse_document, parse_tool_usage
from cortex.memory.schema import Memory
from cortex.memory.store import MemoryStore
from cortex.memory.vector_index import VectorIndex


def ingest(
    raw: RawInput | str | List[dict],
    user_id: UUID,
    store: MemoryStore,
    vector_index: VectorIndex,
    graph_store: Optional[GraphStore] = None,
    llm_fn: Optional[Callable] = None,
) -> List[Memory]:
    """
    Full ingestion: parse -> extract -> normalize -> store -> vector index -> graph.
    raw: RawInput, or string (treated as chat content), or list of chat messages.
    Returns list of created Memory objects.
    """
    if isinstance(raw, list):
        raw = parse_chat(raw)
    elif isinstance(raw, str):
        raw = RawInput(source="chat", content=raw)
    memories_raw = extract_memories(raw, llm_fn=llm_fn)
    created: List[Memory] = []
    session_id = raw.metadata.get("session_id")
    conversation_id = raw.metadata.get("conversation_id")
    source = raw.source
    for rec in memories_raw:
        try:
            create, embedding, extra = normalize_memory(
                rec,
                user_id=user_id,
                source=source,
                source_session=session_id,
                conversation_id=conversation_id,
                compute_embedding=True,
            )
            memory = store.add_memory(create, embedding=embedding)
            if embedding:
                vector_index.add(memory.id, embedding)
            if graph_store:
                build_graph_for_memory(
                    graph_store,
                    memory_id=memory.id,
                    user_id=user_id,
                    summary=memory.summary,
                    memory_type=memory.type.value,
                    entities=memory.entities,
                    relationships=extra.get("relationships"),
                )
            created.append(memory)
        except Exception:
            continue
    return created
