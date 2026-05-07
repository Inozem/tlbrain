from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchValue

from core.config import get_root_folder_id
from core.qdrant.client import get_client
from core.qdrant.schema import get_collection_name


def list_clients() -> list[dict[str, Any]]:
    """
    Returns all clients from the vector store, sorted by client_name.
    Each entry: {client_name, dialog_count, last_dialog_date}.
    """
    scroll_filter = Filter(must=[
        FieldCondition(key="type", match=MatchValue(value="utterance")),
        FieldCondition(key="root_folder_id", match=MatchValue(value=get_root_folder_id())),
    ])

    # Collect unique (client_name, doc_id, dialog_date) triples
    seen: dict[str, dict[str, str]] = {}  # client_name -> {doc_id -> dialog_date}
    offset = None

    while True:
        results, next_offset = get_client().scroll(
            collection_name=get_collection_name(),
            scroll_filter=scroll_filter,
            limit=1000,
            offset=offset,
            with_payload=["client_name", "doc_id", "dialog_date"],
            with_vectors=False,
        )
        for point in results:
            p = point.payload or {}
            client = p.get("client_name") or ""
            doc_id = p.get("doc_id") or ""
            date = p.get("dialog_date") or ""
            if not client or not doc_id:
                continue
            if client not in seen:
                seen[client] = {}
            if doc_id not in seen[client]:
                seen[client][doc_id] = date

        if next_offset is None:
            break
        offset = next_offset

    result = []
    for client_name in sorted(seen):
        docs = seen[client_name]
        entry: dict = {
            "client_name": client_name,
            "dialog_count": len(docs),
            "last_dialog_date": max(docs.values()) if docs else "",
        }
        if client_name == "_unassigned":
            entry["transcripts"] = sorted(
                [{"doc_id": doc_id, "dialog_date": date} for doc_id, date in docs.items()],
                key=lambda x: x["dialog_date"],
                reverse=True,
            )
        result.append(entry)

    return result
