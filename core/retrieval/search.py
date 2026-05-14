from typing import Any

from fastembed import SparseTextEmbedding
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, Range, SparseVector

from core.config import get_root_folder_id
from core.gemini.embeddings import embed
from core.qdrant.client import get_client
from core.qdrant.schema import get_collection_name

_SEARCHABLE_TYPES = ["summary", "fact"]
_KEYWORD_HALF_WINDOW = 2

_bm25_model: SparseTextEmbedding | None = None


def _get_bm25_model() -> SparseTextEmbedding:
    global _bm25_model
    if _bm25_model is None:
        _bm25_model = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _bm25_model


def search_summaries_and_facts(
    query: str,
    client_name: str | None = None,
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

    must: list[FieldCondition] = [
        FieldCondition(key="type", match=MatchAny(any=_SEARCHABLE_TYPES)),
        FieldCondition(key="root_folder_id", match=MatchValue(value=get_root_folder_id())),
    ]

    if client_name is not None:
        must.append(FieldCondition(key="client_name", match=MatchValue(value=client_name)))

    date_range: dict[str, int] = {}
    if date_from is not None:
        date_range["gte"] = int(date_from.replace("-", ""))
    if date_to is not None:
        date_range["lte"] = int(date_to.replace("-", ""))
    if date_range:
        must.append(FieldCondition(key="dialog_date_num", range=Range(**date_range)))

    results = get_client().query_points(
        collection_name=get_collection_name(),
        query=vector,
        using="dense",
        query_filter=Filter(must=must),
        limit=top_k,
        with_payload=True,
    )

    hits = []
    for point in results.points:
        payload = point.payload or {}
        hits.append({
            "doc_id": payload.get("doc_id"),
            "client_name": payload.get("client_name"),
            "type": payload.get("type"),
            "center_index": payload.get("center_index"),
            "covered_range": payload.get("covered_range"),
            "score": point.score,
        })

    return hits


def search_user_facts(
    query: str,
    top_k: int = 10,
    client_name: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict[str, Any]]:
    """Dense search over user_facts. Returns [{doc_id, score}]."""
    vector = embed([query])[0]

    must: list[FieldCondition] = [
        FieldCondition(key="type", match=MatchValue(value="user_fact")),
        FieldCondition(key="root_folder_id", match=MatchValue(value=get_root_folder_id())),
    ]

    if client_name is not None:
        must.append(FieldCondition(key="client_name", match=MatchValue(value=client_name)))

    date_range: dict[str, int] = {}
    if date_from is not None:
        date_range["gte"] = int(date_from.replace("-", ""))
    if date_to is not None:
        date_range["lte"] = int(date_to.replace("-", ""))
    if date_range:
        must.append(FieldCondition(key="dialog_date_num", range=Range(**date_range)))

    results = get_client().query_points(
        collection_name=get_collection_name(),
        query=vector,
        using="dense",
        query_filter=Filter(must=must),
        limit=top_k,
        with_payload=True,
    )

    return [
        {"doc_id": (point.payload or {}).get("doc_id"), "score": point.score}
        for point in results.points
    ]


def search_summaries_for_doc(
    doc_id: str,
    query: str,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """Semantic search over summaries and facts for a specific document."""
    vector = embed([query])[0]

    must: list[FieldCondition] = [
        FieldCondition(key="type", match=MatchAny(any=_SEARCHABLE_TYPES)),
        FieldCondition(key="root_folder_id", match=MatchValue(value=get_root_folder_id())),
        FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
    ]

    results = get_client().query_points(
        collection_name=get_collection_name(),
        query=vector,
        using="dense",
        query_filter=Filter(must=must),
        limit=top_k,
        with_payload=True,
    )

    hits = []
    for point in results.points:
        payload = point.payload or {}
        hits.append({
            "doc_id": payload.get("doc_id"),
            "client_name": payload.get("client_name"),
            "type": payload.get("type"),
            "center_index": payload.get("center_index"),
            "covered_range": payload.get("covered_range"),
            "score": point.score,
        })
    return hits


def keyword_search_utterances(
    query: str,
    top_k: int = 10,
    client_name: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict[str, Any]]:
    """
    BM25 keyword search over utterances in Qdrant.
    Returns list of hits: doc_id, client_name, covered_range (window around hit), score.
    Filters are applied before search; no fallback if results are empty.
    """
    embedding = next(iter(_get_bm25_model().embed([query])))
    sparse_vec = SparseVector(
        indices=embedding.indices.tolist(),
        values=embedding.values.tolist(),
    )

    must: list[FieldCondition] = [
        FieldCondition(key="type", match=MatchValue(value="utterance")),
        FieldCondition(key="root_folder_id", match=MatchValue(value=get_root_folder_id())),
    ]

    if client_name is not None:
        must.append(FieldCondition(key="client_name", match=MatchValue(value=client_name)))

    date_range: dict[str, int] = {}
    if date_from is not None:
        date_range["gte"] = int(date_from.replace("-", ""))
    if date_to is not None:
        date_range["lte"] = int(date_to.replace("-", ""))
    if date_range:
        must.append(FieldCondition(key="dialog_date_num", range=Range(**date_range)))

    results = get_client().query_points(
        collection_name=get_collection_name(),
        query=sparse_vec,
        using="bm25",
        query_filter=Filter(must=must),
        limit=top_k,
        with_payload=True,
    )

    hits = []
    for point in results.points:
        payload = point.payload or {}
        order_index = payload.get("order_index", 0)
        hits.append({
            "doc_id": payload.get("doc_id"),
            "client_name": payload.get("client_name"),
            "covered_range": [max(0, order_index - _KEYWORD_HALF_WINDOW), order_index + _KEYWORD_HALF_WINDOW],
            "score": point.score,
        })

    return hits
