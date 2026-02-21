"""
Basic query retrieval: embed query, vector search, return top-k.
No MVN/reranker in this phase (Phase 1).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional
from uuid import UUID

from cortex.utils import embeddings as embed_utils

if TYPE_CHECKING:
    from cortex.memory.vector_index import VectorIndex


def retrieve(
    query: str,
    vector_index: "VectorIndex",
    user_id: Optional[UUID] = None,
    k: int = 10,
    type_filter: Optional[str] = None,
) -> List[tuple]:
    """
    Given a user query, embed it, run vector search, return top-k (Memory, score) pairs.
    """
    query_embedding = embed_utils.embed(query)
    return vector_index.search(
        query_embedding=query_embedding,
        user_id=user_id,
        k=k,
        type_filter=type_filter,
    )
