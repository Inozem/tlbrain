"""Firestore access for transcript index records and transcript queries."""

import logging
from typing import Any

from google.cloud import firestore

from core.integrations.firestore.client import COLLECTION_NAME, get_client


logger = logging.getLogger(__name__)


def get_transcript_record(doc_id: str) -> dict | None:
    """Return a transcript_index record for a document, or None if absent."""
    document = get_client().collection(COLLECTION_NAME).document(doc_id).get()
    if not document.exists:
        return None
    return document.to_dict()


def load_index(doc_id: str) -> dict[str, Any] | None:
    """Load a transcript index record (legacy service API)."""
    result = get_transcript_record(doc_id)
    if result is not None:
        logger.info("Index loaded: %s", doc_id)
    return result


def save_index(doc_id: str, payload: dict[str, Any]) -> None:
    """Replace a transcript index record."""
    get_client().collection(COLLECTION_NAME).document(doc_id).set(payload)
    logger.info("Index saved: %s", doc_id)


def update_index(doc_id: str, fields: dict[str, Any]) -> None:
    """Update fields on a transcript index record, including dotted paths."""
    get_client().collection(COLLECTION_NAME).document(doc_id).update(fields)
    logger.info("Index fields updated: %s", doc_id)


def delete_index(doc_id: str) -> None:
    """Delete a transcript index record."""
    get_client().collection(COLLECTION_NAME).document(doc_id).delete()
    logger.info("Index deleted: %s", doc_id)


def list_all_index_ids() -> list[str]:
    """Return all transcript index document IDs in Firestore stream order."""
    docs = get_client().collection(COLLECTION_NAME).stream()
    return [doc.id for doc in docs]


def list_transcripts(
    folder_ids: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> dict:
    """Return paginated, date-descending transcripts, excluding placeholders."""
    transcripts = []
    for doc in get_client().collection(COLLECTION_NAME).stream():
        data = doc.to_dict() or {}
        if data.get("status") in ("queued", "downloading"):
            continue
        if folder_ids is not None and data.get("parent_id") not in folder_ids:
            continue
        dialog_date = data.get("dialog_date", "")
        if date_from and dialog_date < date_from:
            continue
        if date_to and dialog_date > date_to:
            continue
        transcripts.append({
            "doc_id": doc.id,
            "parent_id": data.get("parent_id"),
            "title": data.get("source_file", ""),
            "dialog_date": dialog_date,
            "status": data.get("status", ""),
        })

    transcripts.sort(key=lambda x: x["dialog_date"], reverse=True)
    total = len(transcripts)
    page = transcripts[offset: offset + limit]
    return {
        "total": total,
        "returned": len(page),
        "offset": offset,
        "limit": limit,
        "has_more": offset + len(page) < total,
        "transcripts": page,
    }


def list_imported() -> list[dict]:
    """Return all docs with status=imported, preserving stored record expansion."""
    docs = (
        get_client().collection(COLLECTION_NAME)
        .where(filter=firestore.FieldFilter("status", "==", "imported"))
        .stream()
    )
    return [{"doc_id": doc.id, **doc.to_dict()} for doc in docs]


def get_doc_ids_by_client(client_name: str) -> list[str]:
    """Return all transcript_index doc IDs for a client."""
    docs = (
        get_client().collection(COLLECTION_NAME)
        .where(filter=firestore.FieldFilter("client_name", "==", client_name))
        .select([])
        .stream()
    )
    return [doc.id for doc in docs]


def aggregate_transcripts_by_folder() -> dict[str, dict]:
    """Return synced transcript counts and latest dates grouped by parent folder."""
    agg: dict[str, dict] = {}
    for doc in get_client().collection(COLLECTION_NAME).where(
        filter=firestore.FieldFilter("status", "==", "synced")
    ).stream():
        data = doc.to_dict() or {}
        parent_id = data.get("parent_id")
        if not parent_id:
            continue
        dialog_date = data.get("dialog_date", "")
        entry = agg.setdefault(parent_id, {"count": 0, "last_date": None, "last_doc_id": None})
        entry["count"] += 1
        if not entry["last_date"] or dialog_date > entry["last_date"]:
            entry["last_date"] = dialog_date
            entry["last_doc_id"] = doc.id
    return agg
