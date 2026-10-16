from typing import Any

from core.integrations.gemini.embeddings import embed
from core.integrations.fastembed.bm25 import get_bm25_model
from core.integrations.qdrant import reader

def search_summaries_and_facts(
    query: str,
    folder_ids: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    top_k: int = 15,
) -> list[dict[str, Any]]:
    """
    Semantic search over summaries and facts in Qdrant.
    Returns list of hits: doc_id, center_index, covered_range, score.
    Filters are applied before search; no fallback if results are empty.
    """
    vector = embed([query])[0]

    return reader.search_summaries_and_facts(vector, folder_ids, date_from, date_to, top_k)


def search_user_facts(
    query: str,
    top_k: int = 10,
    folder_ids: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict[str, Any]]:
    """Dense search over user_facts. Returns [{doc_id, score}]."""
    vector = embed([query])[0]

    return reader.search_user_facts(vector, top_k, folder_ids, date_from, date_to)


def search_summaries_for_doc(
    doc_id: str,
    query: str,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """Semantic search over summaries and facts for a specific document."""
    vector = embed([query])[0]

    return reader.search_summaries_for_doc(vector, doc_id, top_k)


def keyword_search_utterances(
    query: str,
    top_k: int = 10,
    folder_ids: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict[str, Any]]:
    """
    BM25 keyword search over utterances in Qdrant.
    Returns list of hits: doc_id, parent_id, covered_range (window around hit), score.
    Filters are applied before search; no fallback if results are empty.
    """
    embedding = next(iter(get_bm25_model().embed([query])))
    return reader.keyword_search_utterances(
        embedding.indices.tolist(), embedding.values.tolist(), top_k,
        folder_ids, date_from, date_to,
    )
