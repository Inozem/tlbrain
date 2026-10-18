"""Qdrant read operations used by retrieval orchestration."""

from typing import Any

from qdrant_client.models import SparseVector

from core.integrations.qdrant import filters
from core.integrations.qdrant.client import get_client
from core.integrations.qdrant.schema import get_collection_name
from core.utils.retry import with_retry

_KEYWORD_HALF_WINDOW = 2


def search_summaries_and_facts(vector: list[float], folder_ids: list[str] | None,
                               date_from: str | None, date_to: str | None,
                               top_k: int) -> list[dict[str, Any]]:
    results = get_client().query_points(
        collection_name=get_collection_name(), query=vector, using="dense",
        query_filter=filters.search_filter(folder_ids=folder_ids, date_from=date_from,
                                           date_to=date_to, searchable_types=True),
        limit=top_k, with_payload=True,
    )
    return _summary_hits(results.points)


def search_user_facts(vector: list[float], top_k: int, folder_ids: list[str] | None,
                      date_from: str | None, date_to: str | None) -> list[dict[str, Any]]:
    results = get_client().query_points(
        collection_name=get_collection_name(), query=vector, using="dense",
        query_filter=filters.search_filter(type_value="user_fact", folder_ids=folder_ids,
                                           date_from=date_from, date_to=date_to),
        limit=top_k, with_payload=True,
    )
    return [
        {"doc_id": (point.payload or {}).get("doc_id"), "score": point.score}
        for point in results.points
    ]


def search_summaries_for_doc(vector: list[float], doc_id: str,
                             top_k: int) -> list[dict[str, Any]]:
    results = get_client().query_points(
        collection_name=get_collection_name(), query=vector, using="dense",
        query_filter=filters.summaries_for_doc_filter(doc_id), limit=top_k, with_payload=True,
    )
    return _summary_hits(results.points)


def keyword_search_utterances(indices: list[int], values: list[float], top_k: int,
                              folder_ids: list[str] | None, date_from: str | None,
                              date_to: str | None) -> list[dict[str, Any]]:
    results = get_client().query_points(
        collection_name=get_collection_name(),
        query=SparseVector(indices=indices, values=values), using="bm25",
        query_filter=filters.search_filter(type_value="utterance", folder_ids=folder_ids,
                                           date_from=date_from, date_to=date_to),
        limit=top_k, with_payload=True,
    )
    hits = []
    for point in results.points:
        payload = point.payload or {}
        order_index = payload.get("order_index", 0)
        hits.append({
            "doc_id": payload.get("doc_id"),
            "parent_id": payload.get("parent_id"),
            "covered_range": [max(0, order_index - _KEYWORD_HALF_WINDOW),
                              order_index + _KEYWORD_HALF_WINDOW],
            "score": point.score,
        })
    return hits


def _summary_hits(points) -> list[dict[str, Any]]:
    hits = []
    for point in points:
        payload = point.payload or {}
        hits.append({
            "doc_id": payload.get("doc_id"),
            "parent_id": payload.get("parent_id"),
            "type": payload.get("type"),
            "center_index": payload.get("center_index"),
            "covered_range": payload.get("covered_range"),
            "score": point.score,
        })
    return hits


def fetch_utterances_in_ranges(doc_id: str, ranges: list[list[int]]) -> list[dict[str, Any]]:
    utterances: list[dict[str, Any]] = []
    for start, end in ranges:
        results, _ = get_client().scroll(
            collection_name=get_collection_name(),
            scroll_filter=filters.utterance_range_filter(doc_id, start, end),
            limit=1000, with_payload=True, with_vectors=False,
        )
        utterances.extend(point.payload for point in results if point.payload)
    return utterances


def fetch_utterances_by_doc_id(doc_id: str) -> list[dict[str, Any]]:
    return _scroll_all(filters.utterances_by_doc_filter(doc_id))


def fetch_utterances_by_folder(folder_ids: list[str], date_from: str | None,
                               date_to: str | None) -> list[dict[str, Any]]:
    return _scroll_all(filters.utterances_by_folder_filter(folder_ids, date_from, date_to))


def _scroll_all(scroll_filter) -> list[dict[str, Any]]:
    utterances: list[dict[str, Any]] = []
    offset = None
    while True:
        results, next_offset = get_client().scroll(
            collection_name=get_collection_name(), scroll_filter=scroll_filter,
            limit=1000, offset=offset, with_payload=True, with_vectors=False,
        )
        utterances.extend(point.payload for point in results if point.payload)
        if next_offset is None:
            break
        offset = next_offset
    return utterances


@with_retry
def fetch_utterances_by_order_indexes(
    doc_id: str, root_folder_id: str, order_indexes: list[int]
) -> list[dict[str, Any]]:
    if not order_indexes:
        return []
    results, _ = get_client().scroll(
        collection_name=get_collection_name(),
        scroll_filter=filters.utterances_by_order_indexes_filter(doc_id, root_folder_id, order_indexes),
        with_payload=True, with_vectors=False, limit=len(order_indexes),
    )
    return sorted([result.payload for result in results], key=lambda payload: payload.get("order_index", 0))
