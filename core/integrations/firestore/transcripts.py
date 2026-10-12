"""Firestore access for transcript index records and transcript queries."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from google.cloud import firestore

from core.integrations.firestore.client import COLLECTION_NAME, get_client


logger = logging.getLogger(__name__)

STALE_SYNCING_MINUTES = 15
_NO_RETRY_STAGES = {"import", "invalid_format"}


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


def acquire_for_syncing(doc_id: str) -> bool:
    """Transactionally acquire or recover a transcript for syncing."""
    db = get_client()
    ref = db.collection(COLLECTION_NAME).document(doc_id)
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=STALE_SYNCING_MINUTES)

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> bool:
        snapshot = ref.get(transaction=transaction)
        data = snapshot.to_dict() if snapshot.exists else {}

        if data.get("status") == "syncing":
            changed_at = data.get("status_changed_at")
            if changed_at and changed_at > cutoff:
                return False

        update = {"doc_id": doc_id, "status": "syncing", "status_changed_at": firestore.SERVER_TIMESTAMP}
        if snapshot.exists:
            transaction.update(ref, update)
        else:
            transaction.set(ref, update)
        return True

    result = _txn(db.transaction())
    if result:
        logger.info("Acquired for syncing: %s", doc_id)
    return result


def ensure_imported(doc_id: str, parent_id: str, source_file: str = "") -> bool:
    """Create an imported transcript record only if it does not exist."""
    db = get_client()
    ref = db.collection(COLLECTION_NAME).document(doc_id)

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> bool:
        if ref.get(transaction=transaction).exists:
            return False
        transaction.set(ref, {
            "doc_id": doc_id,
            "parent_id": parent_id,
            "source_file": source_file,
            "status": "imported",
            "status_changed_at": firestore.SERVER_TIMESTAMP,
        })
        return True

    result = _txn(db.transaction())
    if result:
        logger.info("Created imported record: %s (parent=%s)", doc_id, parent_id)
    return result


def mark_synced(doc_id: str) -> None:
    get_client().collection(COLLECTION_NAME).document(doc_id).update({
        "status": "synced",
        "error": None,
        "status_changed_at": firestore.SERVER_TIMESTAMP,
    })
    logger.info("Marked synced: %s", doc_id)


def mark_error(doc_id: str, error: str, error_stage: str = "vector_sync") -> None:
    get_client().collection(COLLECTION_NAME).document(doc_id).update({
        "status": "error",
        "error": error,
        "error_stage": error_stage,
        "status_changed_at": firestore.SERVER_TIMESTAMP,
    })
    logger.info("Marked error: %s — %s (stage=%s)", doc_id, error, error_stage)


def write_queued(meeting_id: str) -> bool:
    """Create a queued placeholder or reset one with an import error."""
    db = get_client()
    ref = db.collection(COLLECTION_NAME).document(f"tldv-{meeting_id}")

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> bool:
        snapshot = ref.get(transaction=transaction)
        if snapshot.exists:
            data = snapshot.to_dict() or {}
            if data.get("status") == "error" and data.get("error_stage") == "import":
                transaction.update(ref, {
                    "status": "queued",
                    "error": None,
                    "error_stage": None,
                    "status_changed_at": firestore.SERVER_TIMESTAMP,
                })
                return True
            return False
        transaction.set(ref, {
            "meeting_id": meeting_id,
            "status": "queued",
            "status_changed_at": firestore.SERVER_TIMESTAMP,
        })
        return True

    result = _txn(db.transaction())
    if result:
        logger.info("Queued: %s", meeting_id)
    return result


def mark_downloading(meeting_id: str) -> None:
    get_client().collection(COLLECTION_NAME).document(f"tldv-{meeting_id}").update({
        "status": "downloading",
        "status_changed_at": firestore.SERVER_TIMESTAMP,
    })
    logger.info("Marked downloading: %s", meeting_id)


def mark_download_error(meeting_id: str, error: str) -> None:
    get_client().collection(COLLECTION_NAME).document(f"tldv-{meeting_id}").update({
        "status": "error",
        "error": error,
        "error_stage": "import",
        "status_changed_at": firestore.SERVER_TIMESTAMP,
    })
    logger.info("Marked download error: %s — %s", meeting_id, error)


def delete_queued_placeholder(meeting_id: str) -> None:
    get_client().collection(COLLECTION_NAME).document(f"tldv-{meeting_id}").delete()
    logger.info("Deleted queued placeholder: %s", meeting_id)


def move_transcript_record(doc_id: str, new_parent_id: str) -> None:
    """Update parent_id, reset status to imported for reindexing.

    modifiedTime is deleted so the checker detects a change and re-enqueues sync,
    even though moving a file in Drive does not update its modifiedTime.
    """
    get_client().collection(COLLECTION_NAME).document(doc_id).update({
        "parent_id": new_parent_id,
        "status": "imported",
        "modifiedTime": firestore.DELETE_FIELD,
        "content_hash": firestore.DELETE_FIELD,
        "version": firestore.DELETE_FIELD,
        "error": None,
        "status_changed_at": firestore.SERVER_TIMESTAMP,
    })
    logger.info("Moved transcript record: %s → parent=%s", doc_id, new_parent_id)


def list_unassigned_transcripts(
    folder_id: str,
    db: firestore.Client | None = None,
) -> list[dict]:
    """Return visible transcripts in a resolved unassigned folder."""
    if db is None:
        db = get_client()
    transcripts = []
    for doc in db.collection(COLLECTION_NAME).where(
        filter=firestore.FieldFilter("parent_id", "==", folder_id)
    ).stream():
        data = doc.to_dict() or {}
        if data.get("status") not in ("queued", "downloading"):
            transcripts.append({"doc_id": doc.id, "dialog_date": data.get("dialog_date", "")})
    transcripts.sort(key=lambda x: x["dialog_date"], reverse=True)
    return transcripts


def get_stale_syncing() -> list[str]:
    """Return doc IDs stuck in syncing beyond the stale threshold."""
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=STALE_SYNCING_MINUTES)
    syncing_docs = get_client().collection(COLLECTION_NAME).where(
        filter=firestore.FieldFilter("status", "==", "syncing")
    ).stream()
    stale = []
    for doc in syncing_docs:
        changed_at = doc.to_dict().get("status_changed_at")
        if changed_at and changed_at < cutoff:
            stale.append(doc.id)
            logger.info("Stale syncing detected: %s", doc.id)
    return stale


def get_error_docs() -> list[str]:
    """Return retryable errors, excluding import and invalid-format failures."""
    error_docs = get_client().collection(COLLECTION_NAME).where(
        filter=firestore.FieldFilter("status", "==", "error")
    ).stream()
    result = []
    for doc in error_docs:
        if (doc.to_dict() or {}).get("error_stage") in _NO_RETRY_STAGES:
            continue
        result.append(doc.id)
        logger.info("Error doc detected: %s", doc.id)
    return result


def get_sync_status() -> dict:
    """Aggregate transcript_index counts by status in a single scan."""
    counts: dict[str, int] = {
        "queued": 0, "downloading": 0, "imported": 0,
        "syncing": 0, "synced": 0, "error": 0,
    }
    unassigned = 0
    for doc in get_client().collection(COLLECTION_NAME).stream():
        data = doc.to_dict()
        status = data.get("status", "")
        if status in counts:
            counts[status] += 1
        if data.get("client_name") == "_unassigned" and status not in ("queued", "downloading"):
            unassigned += 1
    return {"total": sum(counts.values()), **counts, "_unassigned_count": unassigned}


def update_skipped_utterances(doc_id: str, skipped_utterances: list[int]) -> None:
    get_client().collection(COLLECTION_NAME).document(doc_id).update({
        "skipped_utterances": sorted(set(skipped_utterances)),
    })


def get_docs_with_skipped_utterances() -> list[dict]:
    docs = get_client().collection(COLLECTION_NAME).where(
        filter=firestore.FieldFilter("skipped_utterances_count", ">", 0)
    ).stream()
    result = []
    for doc in docs:
        data = doc.to_dict() or {}
        result.append({
            "doc_id": doc.id,
            "client_name": data.get("client_name", ""),
            "skipped_utterances": data.get("skipped_utterances", []),
        })
    return result
