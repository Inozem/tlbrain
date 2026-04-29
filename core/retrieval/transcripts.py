from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchValue, Range

from core.config import get_root_folder_id
from core.qdrant.client import get_client
from core.qdrant.schema import get_collection_name
from core.retrieval.pipeline import dedup_and_sort
from core.retrieval.segments import build_segments


def get_transcripts(
    doc_id: str | None = None,
    client_name: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if doc_id is not None:
        utterances = _scroll_by_doc_id(doc_id)
    elif client_name is not None:
        utterances = _scroll_by_client(client_name, date_from, date_to)
    else:
        return [], {
            "truncated": False,
            "total_matches": 0,
            "returned_segments": 0,
            "limit_reason": "no_results",
            "suggestion": "Укажите doc_id или client_name",
        }

    if not utterances:
        return [], {
            "truncated": False,
            "total_matches": 0,
            "returned_segments": 0,
            "limit_reason": "no_results",
            "suggestion": "Нет данных за выбранный период или клиента",
        }

    docs: dict[str, list[dict[str, Any]]] = {}
    for u in utterances:
        docs.setdefault(u["doc_id"], []).append(u)

    sorted_docs = sorted(
        docs.items(),
        key=lambda x: x[1][0].get("dialog_date", "") if x[1] else "",
        reverse=True,
    )

    total_docs = len(sorted_docs)
    top_docs = sorted_docs[:limit]
    truncated = total_docs > limit

    result_segments = []
    for doc_id_key, doc_utterances in top_docs:
        sorted_utterances = dedup_and_sort(doc_utterances)
        if not sorted_utterances:
            continue
        min_idx = sorted_utterances[0]["order_index"]
        max_idx = sorted_utterances[-1]["order_index"]
        result_segments.append(build_segments(doc_id_key, [[min_idx, max_idx]], sorted_utterances))

    meta: dict[str, Any] = {
        "truncated": truncated,
        "total_matches": total_docs,
        "returned_segments": len(result_segments),
    }
    if truncated:
        meta["suggestion"] = "Используйте limit или уточните период для получения большего числа транскриптов"

    return result_segments, meta


def _scroll_by_doc_id(doc_id: str) -> list[dict[str, Any]]:
    return _scroll_all(Filter(must=[
        FieldCondition(key="type", match=MatchValue(value="utterance")),
        FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
        FieldCondition(key="root_folder_id", match=MatchValue(value=get_root_folder_id())),
    ]))


def _scroll_by_client(
    client_name: str,
    date_from: str | None,
    date_to: str | None,
) -> list[dict[str, Any]]:
    must: list[FieldCondition] = [
        FieldCondition(key="type", match=MatchValue(value="utterance")),
        FieldCondition(key="client_name", match=MatchValue(value=client_name)),
        FieldCondition(key="root_folder_id", match=MatchValue(value=get_root_folder_id())),
    ]

    date_range: dict[str, int] = {}
    if date_from:
        date_range["gte"] = int(date_from.replace("-", ""))
    if date_to:
        date_range["lte"] = int(date_to.replace("-", ""))
    if date_range:
        must.append(FieldCondition(key="dialog_date_num", range=Range(**date_range)))

    return _scroll_all(Filter(must=must))


def _scroll_all(scroll_filter: Filter) -> list[dict[str, Any]]:
    utterances: list[dict[str, Any]] = []
    offset = None

    while True:
        results, next_offset = get_client().scroll(
            collection_name=get_collection_name(),
            scroll_filter=scroll_filter,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in results:
            if point.payload:
                utterances.append(point.payload)
        if next_offset is None:
            break
        offset = next_offset

    return utterances
