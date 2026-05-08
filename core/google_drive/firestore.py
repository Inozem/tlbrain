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
            "syncing_started_at": firestore.SERVER_TIMESTAMP,
        })
        return True

    result = _txn(db.transaction())
    if result:
        logger.info("Acquired for syncing: %s", doc_id)
    return result


def mark_synced(doc_id: str) -> None:
    _get_db().collection(COLLECTION_NAME).document(doc_id).update({
        "status": "synced",
        "synced_at": firestore.SERVER_TIMESTAMP,
        "syncing_started_at": None,
        "error": None,
    })
    logger.info("Marked synced: %s", doc_id)


def mark_error(doc_id: str, error: str) -> None:
    _get_db().collection(COLLECTION_NAME).document(doc_id).update({
        "status": "error",
        "error": error,
        "syncing_started_at": None,
    })
    logger.info("Marked error: %s — %s", doc_id, error)


def write_queued(meeting_id: str) -> bool:
    """Create a placeholder queued record for a meeting. Returns True if created, False if already existed."""
    db = _get_db()
    ref = db.collection(COLLECTION_NAME).document(f"tldv-{meeting_id}")

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> bool:
        if ref.get(transaction=transaction).exists:
            return False
        transaction.set(ref, {
            "tldv_meeting_id": meeting_id,
            "status": "queued",
            "queued_at": firestore.SERVER_TIMESTAMP,
        })
        return True

    result = _txn(db.transaction())
    if result:
        logger.info("Queued: %s", meeting_id)
    return result


def mark_downloading(meeting_id: str) -> None:
    _get_db().collection(COLLECTION_NAME).document(f"tldv-{meeting_id}").update({
        "status": "downloading",
        "downloading_started_at": firestore.SERVER_TIMESTAMP,
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
        "synced_at": None,
        "syncing_started_at": None,
        "error": None,
    })
    logger.info("Moved transcript record: %s → %s", doc_id, new_client_name)


def count_unassigned() -> int:
    """Count transcripts in _unassigned that are past the import stage."""
    db = _get_db()
    docs = (
        db.collection(COLLECTION_NAME)
        .where(filter=firestore.FieldFilter("client_name", "==", "_unassigned"))
        .stream()
    )
    return sum(
        1 for d in docs
        if d.to_dict().get("status") not in ("queued", "downloading")
    )


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
        started_at = doc.to_dict().get("syncing_started_at")
        if started_at and started_at < cutoff:
            db.collection(COLLECTION_NAME).document(doc.id).update({
                "status": "imported",
                "syncing_started_at": None,
            })
            recovered.append(doc.id)
            logger.info("Recovered stale syncing: %s", doc.id)

    return recovered


def sync_clients_from_drive(client_names: list[str]) -> int:
    """Upsert clients/{name} for each Drive folder. Skips already registered. Returns count of new records."""
    db = _get_db()
    created = 0
    for name in client_names:
        ref = db.collection(CLIENTS_COLLECTION).document(name)
        if not ref.get().exists:
            ref.set({"status": "active", "created_at": firestore.SERVER_TIMESTAMP})
            logger.info("Auto-registered client from Drive: %s", name)
            created += 1
    return created


def create_client(client_name: str, description: str | None = None) -> bool:
    """Create clients/{client_name} record. Returns True if created, False if already existed."""
    db = _get_db()
    ref = db.collection(CLIENTS_COLLECTION).document(client_name)

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> bool:
        if ref.get(transaction=transaction).exists:
            return False
        data: dict = {
            "status": "active",
            "created_at": firestore.SERVER_TIMESTAMP,
        }
        if description:
            data["description"] = description
        transaction.set(ref, data)
        return True

    result = _txn(db.transaction())
    if result:
        logger.info("Created client: %s", client_name)
    return result
