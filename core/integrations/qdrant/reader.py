"""Qdrant read operations used by retrieval orchestration."""

from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, SparseVector

from core.integrations.qdrant import filters
from core.integrations.qdrant.client import get_client
from core.integrations.qdrant.schema import get_collection_name
from core.utils.retry import (
    LEGACY_RETRY_POLICY, ProviderBoundaryError, RetryPolicy,
    SINGLE_ATTEMPT_RETRY_POLICY, with_retry,
)

_KEYWORD_HALF_WINDOW = 2


def get_parent_ids_by_doc_ids(doc_ids: list[str], root_folder_id: str) -> dict[str, list[str | None]]:
    """Read and verify all root-scoped parent values for each document."""
    if not doc_ids:
        return {}
    wanted = set(doc_ids)
    parents: dict[str, list[str | None]] = {doc_id: [] for doc_id in wanted}
    offset = None
    while True:
        points, offset = get_client().scroll(
            collection_name=get_collection_name(),
            scroll_filter=Filter(must=[
                FieldCondition(key="doc_id", match=MatchAny(any=sorted(wanted))),
                FieldCondition(key="root_folder_id", match=MatchValue(value=root_folder_id)),
            ]),
            limit=128, offset=offset,
            with_payload=["doc_id", "parent_id"], with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            doc_id = payload.get("doc_id")
            if doc_id in wanted:
                parents[doc_id].append(payload.get("parent_id"))
        if offset is None:
            break
    return {
        doc_id: sorted(values, key=lambda value: (value is not None, value or ""))
        for doc_id, values in parents.items()
        if values
    }


def get_user_fact_payload(point_id: str) -> dict[str, Any] | None:
    """Read one user-fact point by stable UUID for mutation verification."""
    points = get_client().retrieve(
        collection_name=get_collection_name(), ids=[point_id], with_payload=True, with_vectors=False,
    )
    if not points:
        return None
    return dict(points[0].payload or {})


def search_summaries_and_facts(vector: list[float], folder_ids: list[str] | None,
                               date_from: str | None, date_to: str | None,
                               top_k: int, *, single_attempt: bool = False) -> list[dict[str, Any]]:
    results = _qdrant_read(single_attempt, lambda: _client(single_attempt).query_points(
        collection_name=get_collection_name(), query=vector, using="dense",
        query_filter=filters.search_filter(folder_ids=folder_ids, date_from=date_from,
                                           date_to=date_to, searchable_types=True),
        limit=top_k, with_payload=True,
    ))
    return _summary_hits(results.points)


def search_user_facts(vector: list[float], top_k: int, folder_ids: list[str] | None,
                      date_from: str | None, date_to: str | None, *,
                      single_attempt: bool = False) -> list[dict[str, Any]]:
    results = _qdrant_read(single_attempt, lambda: _client(single_attempt).query_points(
        collection_name=get_collection_name(), query=vector, using="dense",
        query_filter=filters.search_filter(type_value="user_fact", folder_ids=folder_ids,
                                           date_from=date_from, date_to=date_to),
        limit=top_k, with_payload=True,
    ))
    return [
        {"doc_id": (point.payload or {}).get("doc_id"), "score": point.score}
        for point in results.points
    ]


def search_summaries_for_doc(vector: list[float], doc_id: str,
                             top_k: int, *, single_attempt: bool = False) -> list[dict[str, Any]]:
    results = _qdrant_read(single_attempt, lambda: _client(single_attempt).query_points(
        collection_name=get_collection_name(), query=vector, using="dense",
        query_filter=filters.summaries_for_doc_filter(doc_id), limit=top_k, with_payload=True,
    ))
    return _summary_hits(results.points)


def keyword_search_utterances(indices: list[int], values: list[float], top_k: int,
                              folder_ids: list[str] | None, date_from: str | None,
                              date_to: str | None, *,
                              single_attempt: bool = False) -> list[dict[str, Any]]:
    results = _qdrant_read(single_attempt, lambda: _client(single_attempt).query_points(
        collection_name=get_collection_name(),
        query=SparseVector(indices=indices, values=values), using="bm25",
        query_filter=filters.search_filter(type_value="utterance", folder_ids=folder_ids,
                                           date_from=date_from, date_to=date_to),
        limit=top_k, with_payload=True,
    ))
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


def fetch_utterances_in_ranges(doc_id: str, ranges: list[list[int]], *,
                               single_attempt: bool = False) -> list[dict[str, Any]]:
    utterances: list[dict[str, Any]] = []
    for start, end in ranges:
        results, _ = _qdrant_read(single_attempt, lambda: _client(single_attempt).scroll(
            collection_name=get_collection_name(),
            scroll_filter=filters.utterance_range_filter(doc_id, start, end),
            limit=1000, with_payload=True, with_vectors=False,
        ))
        utterances.extend(point.payload for point in results if point.payload)
    return utterances


def fetch_utterances_by_doc_id(doc_id: str, *, single_attempt: bool = False) -> list[dict[str, Any]]:
    return _scroll_all(filters.utterances_by_doc_filter(doc_id), single_attempt=single_attempt)


def fetch_utterances_by_folder(folder_ids: list[str], date_from: str | None,
                               date_to: str | None, *, single_attempt: bool = False) -> list[dict[str, Any]]:
    return _scroll_all(filters.utterances_by_folder_filter(folder_ids, date_from, date_to),
                       single_attempt=single_attempt)


def _scroll_all(scroll_filter, *, single_attempt: bool = False) -> list[dict[str, Any]]:
    utterances: list[dict[str, Any]] = []
    offset = None
    while True:
        results, next_offset = _qdrant_read(single_attempt, lambda: _client(single_attempt).scroll(
            collection_name=get_collection_name(), scroll_filter=scroll_filter,
            limit=1000, offset=offset, with_payload=True, with_vectors=False,
        ))
        utterances.extend(point.payload for point in results if point.payload)
        if next_offset is None:
            break
        offset = next_offset
    return utterances


@with_retry
def fetch_utterances_by_order_indexes(
    doc_id: str, root_folder_id: str, order_indexes: list[int], *,
    retry_policy: RetryPolicy = LEGACY_RETRY_POLICY,
) -> list[dict[str, Any]]:
    if not order_indexes:
        return []
    single_attempt = retry_policy == SINGLE_ATTEMPT_RETRY_POLICY
    results, _ = _qdrant_read(single_attempt, lambda: _client(single_attempt).scroll(
        collection_name=get_collection_name(),
        scroll_filter=filters.utterances_by_order_indexes_filter(doc_id, root_folder_id, order_indexes),
        with_payload=True, with_vectors=False, limit=len(order_indexes),
    ))
    return sorted([result.payload for result in results], key=lambda payload: payload.get("order_index", 0))


def _qdrant_read(single_attempt: bool, callback):
    try:
        return callback()
    except Exception as error:
        if single_attempt:
            raise ProviderBoundaryError("qdrant", error) from error
        raise


def _client(single_attempt: bool):
    return get_client(single_attempt=True) if single_attempt else get_client()
