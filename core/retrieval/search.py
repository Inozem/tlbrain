from typing import Any

from core.integrations.gemini.embeddings import embed
from core.integrations.fastembed.bm25 import get_bm25_model
from core.integrations.qdrant import reader
from core.utils.retry import ProviderBoundaryError, RetryPolicy

def search_summaries_and_facts(
    query: str,
    folder_ids: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    top_k: int = 15,
    *,
    retry_policy: RetryPolicy | None = None,
) -> list[dict[str, Any]]:
    """
    Semantic search over summaries and facts in Qdrant.
    Returns list of hits: doc_id, center_index, covered_range, score.
    Filters are applied before search; no fallback if results are empty.
    """
    single_attempt = retry_policy is not None and retry_policy.attempts == 1
    try:
        vector = embed([query], **({} if retry_policy is None else {"retry_policy": retry_policy}))[0]
    except Exception as error:
        if single_attempt:
            raise ProviderBoundaryError("gemini", error) from error
        raise
    return reader.search_summaries_and_facts(
        vector, folder_ids, date_from, date_to, top_k,
        **({"single_attempt": True} if single_attempt else {}),
    )


def search_user_facts(
    query: str,
    top_k: int = 10,
    folder_ids: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    *,
    retry_policy: RetryPolicy | None = None,
) -> list[dict[str, Any]]:
    """Dense search over user_facts. Returns [{doc_id, score}]."""
    single_attempt = retry_policy is not None and retry_policy.attempts == 1
    try:
        vector = embed([query], **({} if retry_policy is None else {"retry_policy": retry_policy}))[0]
    except Exception as error:
        if single_attempt:
            raise ProviderBoundaryError("gemini", error) from error
        raise

    return reader.search_user_facts(
        vector, top_k, folder_ids, date_from, date_to,
        **({"single_attempt": True} if single_attempt else {}),
    )


def search_summaries_for_doc(
    doc_id: str,
    query: str,
    top_k: int = 20,
    *,
    retry_policy: RetryPolicy | None = None,
) -> list[dict[str, Any]]:
    """Semantic search over summaries and facts for a specific document."""
    single_attempt = retry_policy is not None and retry_policy.attempts == 1
    try:
        vector = embed([query], **({} if retry_policy is None else {"retry_policy": retry_policy}))[0]
    except Exception as error:
        if single_attempt:
            raise ProviderBoundaryError("gemini", error) from error
        raise

    return reader.search_summaries_for_doc(
        vector, doc_id, top_k, **({"single_attempt": True} if single_attempt else {}),
    )


def keyword_search_utterances(
    query: str,
    top_k: int = 10,
    folder_ids: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    *,
    retry_policy: RetryPolicy | None = None,
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
        **({"single_attempt": True} if retry_policy is not None and retry_policy.attempts == 1 else {}),
    )
