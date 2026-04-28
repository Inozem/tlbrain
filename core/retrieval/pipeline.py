from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchValue, Range

from core.qdrant.client import get_client
from core.qdrant.schema import get_collection_name

_HALF_WINDOW = 2


def merge_ranges(hits: list[dict[str, Any]]) -> dict[str, list[list[int]]]:
    """
    Groups hits by doc_id and merges overlapping covered_ranges.
    Facts hits without covered_range get a derived range [center-2, center+2].
    Returns {doc_id: [[start, end], ...]} with non-overlapping sorted ranges.
    """
    ranges_by_doc: dict[str, list[list[int]]] = {}

    for hit in hits:
        doc_id = hit["doc_id"]
        covered_range = hit.get("covered_range")

        if covered_range is None:
            center = hit["center_index"]
            covered_range = [center - _HALF_WINDOW, center + _HALF_WINDOW]

        ranges_by_doc.setdefault(doc_id, []).append(covered_range)

    merged: dict[str, list[list[int]]] = {}
    for doc_id, ranges in ranges_by_doc.items():
        merged[doc_id] = _merge(sorted(ranges, key=lambda r: r[0]))

    return merged


def fetch_utterances(doc_id: str, merged_ranges: list[list[int]]) -> list[dict[str, Any]]:
    """
    Fetches utterance payloads from Qdrant for all merged ranges of a single doc_id.
    Uses scroll without vectors; filters by type=utterance and order_index range.
    """
    utterances: list[dict[str, Any]] = []

    for start, end in merged_ranges:
        results, _ = get_client().scroll(
            collection_name=get_collection_name(),
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="type", match=MatchValue(value="utterance")),
                    FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                    FieldCondition(key="order_index", range=Range(gte=start, lte=end)),
                ]
            ),
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )
        for point in results:
            if point.payload:
                utterances.append(point.payload)

    return utterances


def dedup_and_sort(utterances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Removes duplicates by (doc_id, order_index) and sorts by order_index ASC.
    """
    seen: set[tuple[str, int]] = set()
    unique: list[dict[str, Any]] = []

    for u in utterances:
        key = (u["doc_id"], u["order_index"])
        if key not in seen:
            seen.add(key)
            unique.append(u)

    return sorted(unique, key=lambda u: u["order_index"])


def _merge(sorted_ranges: list[list[int]]) -> list[list[int]]:
    if not sorted_ranges:
        return []

    result = [sorted_ranges[0][:]]
    for start, end in sorted_ranges[1:]:
        if start <= result[-1][1] + 1:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])

    return result
