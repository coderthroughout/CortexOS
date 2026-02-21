"""
Embedding utility for memory text. Uses sentence-transformers (bge-small / e5-base style)
per ContexMemory.md. Dimension 384 for default model to match db_schema.sql.
"""
from __future__ import annotations

from typing import List, Union

# Default: small model, 384 dims. Override with CORTEX_EMBEDDING_MODEL env.
_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_embedder = None
_model_name = None


def get_embedder(model_name: str = ""):
    global _embedder, _model_name
    name = model_name or _DEFAULT_MODEL
    if _embedder is None or _model_name != name:
        try:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer(name)
            _model_name = name
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model {name}: {e}") from e
    return _embedder


def get_embedding_dimension(model_name: str = "") -> int:
    """Return embedding dimension for the given (or default) model."""
    m = get_embedder(model_name)
    return m.get_sentence_embedding_dimension()


def embed(text: Union[str, List[str]], model_name: str = "") -> Union[List[float], List[List[float]]]:
    """
    Embed one string or a list of strings. Returns list of floats or list of list of floats.
    """
    if isinstance(text, str):
        texts = [text]
        single = True
    else:
        texts = list(text)
        single = False
    if not texts:
        return [] if not single else []
    model = get_embedder(model_name)
    vecs = model.encode(texts, convert_to_numpy=True)
    if single:
        return vecs[0].tolist()
    return [v.tolist() for v in vecs]
