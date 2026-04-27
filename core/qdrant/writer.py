from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct

from core.qdrant.client import get_client
from core.qdrant.schema import EMBEDDING_DIMENSIONS, get_collection_name

_ZERO_VECTOR = [0.0] * EMBEDDING_DIMENSIONS


def upsert_utterances(utterances: list[dict[str, Any]]) -> None:
    if not utterances:
        return
    points = [
        PointStruct(
            id=_point_id("utterance", u["doc_id"], str(u["order_index"])),
            vector=_ZERO_VECTOR,
            payload=u,
        )
        for u in utterances
    ]
    get_client().upsert(collection_name=get_collection_name(), points=points)


def upsert_summaries(summaries: list[dict[str, Any]], vectors: list[list[float]]) -> None:
    if not summaries:
        return
    points = [
        PointStruct(
            id=_point_id("summary", s["doc_id"], s["summary_id"]),
            vector=vector,
            payload=s,
        )
        for s, vector in zip(summaries, vectors)
    ]
    get_client().upsert(collection_name=get_collection_name(), points=points)


def upsert_facts(facts: list[dict[str, Any]], vectors: list[list[float]]) -> None:
    if not facts:
        return
    points = [
        PointStruct(
            id=_point_id("facts", f["doc_id"], f["summary_id"]),
            vector=vector,
            payload=f,
        )
        for f, vector in zip(facts, vectors)
    ]
    get_client().upsert(collection_name=get_collection_name(), points=points)


def delete_old_versions(doc_id: str, new_version: str, root_folder_id: str) -> None:
    """Delete all points for doc_id whose version differs from new_version."""
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


def _point_id(type_: str, doc_id: str, key: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"{type_}:{doc_id}:{key}"))
