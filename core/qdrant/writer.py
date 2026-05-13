from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, PointStruct, SparseVector

from core.qdrant.client import get_client
from core.qdrant.schema import get_collection_name
from core.utils.retry import with_retry


@with_retry
def upsert_utterances(utterances: list[dict[str, Any]], sparse_vectors: list[SparseVector]) -> None:
    if not utterances:
        return
    points = [
        PointStruct(
            id=_point_id("utterance", u["doc_id"], str(u["order_index"])),
            vector={"bm25": sv},
            payload=u,
        )
        for u, sv in zip(utterances, sparse_vectors)
    ]
    get_client().upsert(collection_name=get_collection_name(), points=points)


@with_retry
def upsert_summaries(summaries: list[dict[str, Any]], vectors: list[list[float]]) -> None:
    if not summaries:
        return
    points = [
        PointStruct(
            id=_point_id("summary", s["doc_id"], s["summary_id"]),
            vector={"dense": vector},
            payload=s,
        )
        for s, vector in zip(summaries, vectors)
    ]
    get_client().upsert(collection_name=get_collection_name(), points=points)


@with_retry
def upsert_facts(facts: list[dict[str, Any]], vectors: list[list[float]]) -> None:
    if not facts:
        return
    points = [
        PointStruct(
            id=_point_id("fact", f["doc_id"], f"{f['summary_id']}:{f['text']}"),
            vector={"dense": vector},
            payload=f,
        )
        for f, vector in zip(facts, vectors)
    ]
    get_client().upsert(collection_name=get_collection_name(), points=points)


@with_retry
def delete_old_versions(doc_id: str, new_version: str, root_folder_id: str) -> None:
    get_client().delete(
        collection_name=get_collection_name(),
        points_selector=Filter(
            must=[
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                FieldCondition(key="root_folder_id", match=MatchValue(value=root_folder_id)),
            ],
            must_not=[
                FieldCondition(key="version", match=MatchValue(value=new_version)),
            ],
        ),
    )


@with_retry
def delete_by_doc_id(doc_id: str, root_folder_id: str) -> None:
    get_client().delete(
        collection_name=get_collection_name(),
        points_selector=Filter(
            must=[
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
                FieldCondition(key="root_folder_id", match=MatchValue(value=root_folder_id)),
            ]
        ),
    )


@with_retry
def delete_utterances_by_order_indexes(
    doc_id: str, root_folder_id: str, order_indexes: list[int]
) -> None:
    if not order_indexes:
        return
    get_client().delete(
        collection_name=get_collection_name(),
        points_selector=Filter(must=[
            FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
            FieldCondition(key="root_folder_id", match=MatchValue(value=root_folder_id)),
            FieldCondition(key="order_index", match=MatchAny(any=order_indexes)),
            FieldCondition(key="type", match=MatchValue(value="utterance")),
        ]),
    )


@with_retry
def delete_summaries_by_center_indexes(
    doc_id: str, root_folder_id: str, center_indexes: list[int]
) -> None:
    if not center_indexes:
        return
    get_client().delete(
        collection_name=get_collection_name(),
        points_selector=Filter(must=[
            FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
            FieldCondition(key="root_folder_id", match=MatchValue(value=root_folder_id)),
            FieldCondition(key="center_index", match=MatchAny(any=center_indexes)),
            FieldCondition(key="type", match=MatchAny(any=["summary", "fact"])),
        ]),
    )


def _point_id(type_: str, doc_id: str, key: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"{type_}:{doc_id}:{key}"))
