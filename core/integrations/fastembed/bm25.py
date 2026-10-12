"""Shared lazy BM25 embedding model for indexing and retrieval."""

from fastembed import SparseTextEmbedding

_bm25_model: SparseTextEmbedding | None = None


def get_bm25_model() -> SparseTextEmbedding:
    """Return the shared, lazily initialized BM25 model."""
    global _bm25_model
    if _bm25_model is None:
        _bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _bm25_model
