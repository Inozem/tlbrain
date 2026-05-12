import logging
from datetime import datetime, timezone, timedelta

from google.cloud import firestore

logger = logging.getLogger(__name__)

COLLECTION_NAME = "transcript_index"
CLIENTS_COLLECTION = "clients"
STALE_SYNCING_MINUTES = 15


def _get_db() -> firestore.Client:
    return firestore.Client()


def acquire_for_syncing(doc_id: str) -> bool:
    db = _get_db()
    ref = db.collection(COLLECTION_NAME).document(doc_id)

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> bool:
        snapshot = ref.get(transaction=transaction)
        if not snapshot.exists:
            return False
        if snapshot.to_dict().get("status") != "imported":
            return False
        transaction.update(ref, {
            "status": "syncing",
            "status_changed_at": firestore.SERVER_TIMESTAMP,
        })
        return True

    result = _txn(db.transaction())
    if result:
        logger.info("Acquired for syncing: %s", doc_id)
    return result


def mark_synced(doc_id: str) -> None:
    _get_db().collection(COLLECTION_NAME).document(doc_id).update({
        "status": "synced",
        "error": None,
        "status_changed_at": firestore.SERVER_TIMESTAMP,
    })
    logger.info("Marked synced: %s", doc_id)


def mark_error(doc_id: str, error: str, error_stage: str = "vector_sync") -> None:
    _get_db().collection(COLLECTION_NAME).document(doc_id).update({
        "status": "error",
        "error": error,
        "error_stage": error_stage,
        "status_changed_at": firestore.SERVER_TIMESTAMP,
    })
    logger.info("Marked error: %s — %s (stage=%s)", doc_id, error, error_stage)


def write_queued(meeting_id: str) -> bool:
    """Create a placeholder queued record for a meeting. Returns True if created, False if already existed."""
    db = _get_db()
    ref = db.collection(COLLECTION_NAME).document(f"tldv-{meeting_id}")

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> bool:
        if ref.get(transaction=transaction).exists:
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
    _get_db().collection(COLLECTION_NAME).document(f"tldv-{meeting_id}").update({
        "status": "downloading",
        "status_changed_at": firestore.SERVER_TIMESTAMP,
    })
    logger.info("Marked downloading: %s", meeting_id)


def delete_queued_placeholder(meeting_id: str) -> None:
    _get_db().collection(COLLECTION_NAME).document(f"tldv-{meeting_id}").delete()
    logger.info("Deleted queued placeholder: %s", meeting_id)


def move_transcript_record(doc_id: str, new_client_name: str, new_drive_folder: str) -> None:
    """Update client_name and drive_folder, reset status to imported for reindexing.

    modifiedTime is deleted so the checker detects a change and re-enqueues sync,
    even though moving a file in Drive does not update its modifiedTime.
    """
    _get_db().collection(COLLECTION_NAME).document(doc_id).update({
        "client_name": new_client_name,
        "drive_folder": new_drive_folder,
        "status": "imported",
        "modifiedTime": firestore.DELETE_FIELD,
        "content_hash": firestore.DELETE_FIELD,
        "version": firestore.DELETE_FIELD,
        "error": None,
        "status_changed_at": firestore.SERVER_TIMESTAMP,
    })
    logger.info("Moved transcript record: %s → %s", doc_id, new_client_name)


def get_unassigned() -> dict:
    """Return count and list of unassigned transcripts (past the import stage).

    Returns: {"count": int, "transcripts": [{"doc_id": str, "dialog_date": str}]}
    """
    db = _get_db()
    docs = (
        db.collection(COLLECTION_NAME)
        .where(filter=firestore.FieldFilter("client_name", "==", "_unassigned"))
        .stream()
    )
    transcripts = [
        {"doc_id": d.id, "dialog_date": d.to_dict().get("dialog_date", "")}
        for d in docs
        if d.to_dict().get("status") not in ("queued", "downloading")
    ]
    transcripts.sort(key=lambda x: x["dialog_date"], reverse=True)
    return {"count": len(transcripts), "transcripts": transcripts}


def list_imported() -> list[dict]:
    """Return all docs with status=imported."""
    db = _get_db()
    docs = (
        db.collection(COLLECTION_NAME)
        .where(filter=firestore.FieldFilter("status", "==", "imported"))
        .stream()
    )
    return [{"doc_id": doc.id, **doc.to_dict()} for doc in docs]


def recover_stale_syncing() -> list[str]:
    db = _get_db()
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=STALE_SYNCING_MINUTES)

    syncing_docs = (
        db.collection(COLLECTION_NAME)
        .where(filter=firestore.FieldFilter("status", "==", "syncing"))
        .stream()
    )

    recovered = []
    for doc in syncing_docs:
        changed_at = doc.to_dict().get("status_changed_at")
        if changed_at and changed_at < cutoff:
            db.collection(COLLECTION_NAME).document(doc.id).update({
                "status": "imported",
                "status_changed_at": firestore.SERVER_TIMESTAMP,
            })
            recovered.append(doc.id)
            logger.info("Recovered stale syncing: %s", doc.id)

    return recovered


def recover_errors() -> list[str]:
    """Reset error docs back to their pre-error status based on error_stage."""
    db = _get_db()
    error_docs = (
        db.collection(COLLECTION_NAME)
        .where(filter=firestore.FieldFilter("status", "==", "error"))
        .stream()
    )
    recovered = []
    for doc in error_docs:
        data = doc.to_dict()
        stage = data.get("error_stage", "vector_sync")
        reset_status = "imported" if stage == "vector_sync" else "queued"
        db.collection(COLLECTION_NAME).document(doc.id).update({
            "status": reset_status,
            "error": None,
            "error_stage": firestore.DELETE_FIELD,
            "status_changed_at": firestore.SERVER_TIMESTAMP,
        })
        recovered.append(doc.id)
        logger.info("Recovered error doc: %s (stage=%s) → %s", doc.id, stage, reset_status)
    return recovered



def increment_client_speakers(client_name: str, speakers: list[str]) -> None:
    """Increment speaker counts in clients/{client_name}.speakers."""
    if not speakers:
        return
    db = _get_db()
    ref = db.collection(CLIENTS_COLLECTION).document(client_name)
    updates = {f"speakers.{speaker}": firestore.Increment(1) for speaker in speakers}
    ref.set(updates, merge=True)
    logger.debug("Incremented speakers for %s: %s", client_name, speakers)


def migrate_speaker_index() -> int:
    """Populate clients.speakers from transcript_index docs that have speakers stored.

    Idempotent: skips docs where speakers_indexed=True.
    Returns count of migrated docs.
    """
    db = _get_db()
    migrated = 0
    for doc in db.collection(COLLECTION_NAME).stream():
        data = doc.to_dict()
        if data.get("status") in ("queued", "downloading"):
            continue
        if data.get("speakers_indexed"):
            continue
        speakers = data.get("speakers")
        if not speakers:
            continue
        client_name = data.get("client_name", "")
        if not client_name:
            continue
        increment_client_speakers(client_name, speakers)
        db.collection(COLLECTION_NAME).document(doc.id).update({"speakers_indexed": True})
        migrated += 1
        logger.info("Migrated speaker index: %s → %s (%d speakers)", doc.id, client_name, len(speakers))
    return migrated


def sync_clients_from_drive(folders: list[dict[str, str]]) -> int:
    """Upsert clients/{name} for each Drive folder. Updates folder_id if changed. Returns count of new records.

    Each folder dict must have: {name: str, id: str}
    """
    db = _get_db()
    created = 0
    for folder in folders:
        name = folder["name"]
        folder_id = folder["id"]
        ref = db.collection(CLIENTS_COLLECTION).document(name)
        snapshot = ref.get()
        if not snapshot.exists:
            ref.set({
                "status": "active",
                "folder_id": folder_id,
                "created_at": firestore.SERVER_TIMESTAMP,
            })
            logger.info("Auto-registered client from Drive: %s (%s)", name, folder_id)
            created += 1
        elif snapshot.to_dict().get("folder_id") != folder_id:
            ref.update({"folder_id": folder_id})
            logger.info("Updated folder_id for client: %s (%s)", name, folder_id)

    migrated = migrate_speaker_index()
    if migrated:
        logger.info("Migrated speaker index for %d doc(s)", migrated)

    return created


def get_all_client_names() -> list[str]:
    """Return all known client names from the clients collection."""
    db = _get_db()
    return sorted(
        doc.id for doc in db.collection(CLIENTS_COLLECTION).select([]).stream()
        if doc.id != "_unassigned"
    )


def get_client_folder_id(client_name: str) -> str | None:
    """Return Drive folder_id for a client, or None if not found."""
    doc = _get_db().collection(CLIENTS_COLLECTION).document(client_name).get()
    if not doc.exists:
        return None
    return doc.to_dict().get("folder_id")


def get_sync_status() -> dict:
    """Aggregate transcript_index counts by status in a single scan."""
    db = _get_db()
    counts: dict[str, int] = {
        "queued": 0, "downloading": 0, "imported": 0,
        "syncing": 0, "synced": 0, "error": 0,
    }
    unassigned = 0

    for doc in db.collection(COLLECTION_NAME).stream():
        data = doc.to_dict()
        status = data.get("status", "")
        if status in counts:
            counts[status] += 1
        if data.get("client_name") == "_unassigned" and status not in ("queued", "downloading"):
            unassigned += 1

    return {
        "total": sum(counts.values()),
        **counts,
        "_unassigned_count": unassigned,
    }


TOKENS_COLLECTION = "tokens"
DRIVE_SYNC_TOKEN_DOC = "drive_sync"


def get_drive_sync_token() -> str | None:
    doc = _get_db().collection(TOKENS_COLLECTION).document(DRIVE_SYNC_TOKEN_DOC).get()
    if not doc.exists:
        return None
    return doc.to_dict().get("page_token")


def set_drive_sync_token(token: str) -> None:
    _get_db().collection(TOKENS_COLLECTION).document(DRIVE_SYNC_TOKEN_DOC).set({
        "page_token": token,
        "updated_at": firestore.SERVER_TIMESTAMP,
    })


def get_client_name_by_folder_id(folder_id: str) -> str | None:
    """Reverse lookup: folder_id → client_name from clients collection."""
    db = _get_db()
    docs = (
        db.collection(CLIENTS_COLLECTION)
        .where(filter=firestore.FieldFilter("folder_id", "==", folder_id))
        .limit(1)
        .stream()
    )
    for doc in docs:
        return doc.id
    return None


def create_client(client_name: str, folder_id: str, description: str | None = None) -> bool:
    """Create clients/{client_name} record. Returns True if created, False if already existed."""
    db = _get_db()
    ref = db.collection(CLIENTS_COLLECTION).document(client_name)

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> bool:
        if ref.get(transaction=transaction).exists:
            return False
        data: dict = {
            "status": "active",
            "folder_id": folder_id,
            "created_at": firestore.SERVER_TIMESTAMP,
        }
        if description:
            data["description"] = description
        transaction.set(ref, data)
        return True

    result = _txn(db.transaction())
    if result:
        logger.info("Created client: %s (%s)", client_name, folder_id)
    return result
