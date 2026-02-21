"""On new memory: add node, link to entities, create relation edges from extractor output."""
from __future__ import annotations

from typing import Any, List, Optional
from uuid import UUID

from cortex.graph.graph_store import GraphStore


def build_graph_for_memory(
    graph: GraphStore,
    memory_id: UUID,
    user_id: UUID,
    summary: str,
    memory_type: str,
    entities: List[str],
    relationships: Optional[List[dict]] = None,
) -> None:
    """
    After a memory is stored, update the graph: add Memory node, link to User and Entities,
    and create relationship edges from extractor output (e.g. CAUSES, RELATES_TO).
    """
    graph.ensure_user(user_id)
    graph.add_memory_node(memory_id, user_id, summary, memory_type)
    graph.link_memory_entities(memory_id, entities)
    if not relationships:
        return
    for rel in relationships:
        if not isinstance(rel, dict):
            continue
        from_ent = rel.get("from") or rel.get("from_entity")
        to_ent = rel.get("to") or rel.get("to_entity")
        rel_type = rel.get("relation") or rel.get("type") or "RELATES_TO"
        if from_ent and to_ent:
            graph.link_relationship(str(from_ent), rel_type, str(to_ent))
