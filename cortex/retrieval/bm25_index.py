"""In-memory BM25 index over memory summaries."""
from __future__ import annotations

import re
from typing import List, Optional
from uuid import UUID

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """In-memory BM25 index over memory summaries. Rebuild when memories change or load from store."""

    def __init__(self):
        self._doc_ids: List[str] = []
        self._corpus: List[List[str]] = []
        self._bm25: Optional[BM25Okapi] = None

    def add(self, memory_id: str, text: str) -> None:
        self._doc_ids.append(memory_id)
        self._corpus.append(tokenize(text))
        self._bm25 = None

    def build(self, doc_ids: List[str], texts: List[str]) -> None:
        """Replace index with given docs."""
        self._doc_ids = list(doc_ids)
        self._corpus = [tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(self._corpus) if self._corpus else None

    def search(self, query: str, top_k: int = 50, user_doc_ids: Optional[set] = None) -> List[tuple[str, float]]:
        """
        Return list of (memory_id, score). user_doc_ids: optional set of ids to filter (e.g. by user).
        """
        if not self._corpus:
            return []
        if self._bm25 is None:
            self._bm25 = BM25Okapi(self._corpus)
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)
        out = [(self._doc_ids[i], float(scores[i])) for i in range(len(self._doc_ids))]
        if user_doc_ids is not None:
            out = [(mid, s) for mid, s in out if mid in user_doc_ids]
        out.sort(key=lambda x: -x[1])
        return out[:top_k]
