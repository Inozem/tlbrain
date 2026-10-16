"""Qdrant filter construction for read operations."""

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, Range

from core.config import get_root_folder_id

_SEARCHABLE_TYPES = ["summary", "fact"]


def search_filter(
    type_value: str | None = None,
    folder_ids: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    *,
    searchable_types: bool = False,
) -> Filter:
    must: list[FieldCondition] = []
    if searchable_types:
        must.append(FieldCondition(key="type", match=MatchAny(any=_SEARCHABLE_TYPES)))
    elif type_value is not None:
        must.append(FieldCondition(key="type", match=MatchValue(value=type_value)))
    must.append(FieldCondition(key="root_folder_id", match=MatchValue(value=get_root_folder_id())))
    if folder_ids is not None:
        must.append(FieldCondition(key="parent_id", match=MatchAny(any=folder_ids)))
    date_range = _date_range(date_from, date_to)
    if date_range:
        must.append(FieldCondition(key="dialog_date_num", range=Range(**date_range)))
    return Filter(must=must)


def summaries_for_doc_filter(doc_id: str) -> Filter:
    return Filter(must=[
        FieldCondition(key="type", match=MatchAny(any=_SEARCHABLE_TYPES)),
        FieldCondition(key="root_folder_id", match=MatchValue(value=get_root_folder_id())),
        FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
    ])


def utterance_range_filter(doc_id: str, start: int, end: int) -> Filter:
    return Filter(must=[
        FieldCondition(key="type", match=MatchValue(value="utterance")),
        FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
        FieldCondition(key="order_index", range=Range(gte=start, lte=end)),
    ])


def utterances_by_doc_filter(doc_id: str) -> Filter:
    return Filter(must=[
        FieldCondition(key="type", match=MatchValue(value="utterance")),
        FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
        FieldCondition(key="root_folder_id", match=MatchValue(value=get_root_folder_id())),
    ])


def utterances_by_folder_filter(
    folder_ids: list[str], date_from: str | None, date_to: str | None
) -> Filter:
    must: list[FieldCondition] = [
        FieldCondition(key="type", match=MatchValue(value="utterance")),
        FieldCondition(key="parent_id", match=MatchAny(any=folder_ids)),
        FieldCondition(key="root_folder_id", match=MatchValue(value=get_root_folder_id())),
    ]
    # Transcript folder filtering historically treats empty dates as omitted.
    date_range = _date_range(date_from, date_to, ignore_empty=True)
    if date_range:
        must.append(FieldCondition(key="dialog_date_num", range=Range(**date_range)))
    return Filter(must=must)


def utterances_by_order_indexes_filter(
    doc_id: str, root_folder_id: str, order_indexes: list[int]
) -> Filter:
    return Filter(must=[
        FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
        FieldCondition(key="root_folder_id", match=MatchValue(value=root_folder_id)),
        FieldCondition(key="order_index", match=MatchAny(any=order_indexes)),
        FieldCondition(key="type", match=MatchValue(value="utterance")),
    ])


def _date_range(
    date_from: str | None, date_to: str | None, *, ignore_empty: bool = False
) -> dict[str, int]:
    date_range: dict[str, int] = {}
    if date_from is not None and (date_from or not ignore_empty):
        date_range["gte"] = int(date_from.replace("-", ""))
    if date_to is not None and (date_to or not ignore_empty):
        date_range["lte"] = int(date_to.replace("-", ""))
    return date_range
