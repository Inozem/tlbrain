from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, Range

from core.config import get_root_folder_id
from core.gemini.embeddings import embed
from core.qdrant.client import get_client
from core.qdrant.schema import get_collection_name

_SEARCHABLE_TYPES = ["summary", "facts"]


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

    date_range: dict[str, str] = {}
    if date_from is not None:
        date_range["gte"] = date_from
    if date_to is not None:
        date_range["lte"] = date_to
    if date_range:
        must.append(FieldCondition(key="dialog_date", range=Range(**date_range)))

    results = get_client().query_points(
        collection_name=get_collection_name(),
        query=vector,
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
