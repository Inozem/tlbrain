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
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=STALE_SYNCING_MINUTES)

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> bool:
        snapshot = ref.get(transaction=transaction)
        data = snapshot.to_dict() if snapshot.exists else {}

        if data.get("status") == "syncing":
            changed_at = data.get("status_changed_at")
            if changed_at and changed_at > cutoff:
                return False  # recently acquired

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


def ensure_imported(doc_id: str, client_name: str, folder_id: str) -> bool:
    """Create transcript_index record with status=imported if it doesn't exist. Returns True if created."""
    db = _get_db()
    ref = db.collection(COLLECTION_NAME).document(doc_id)

    @firestore.transactional
    def _txn(transaction: firestore.Transaction) -> bool:
        if ref.get(transaction=transaction).exists:
            return False
        transaction.set(ref, {
            "doc_id": doc_id,
            "client_name": client_name,
            "drive_folder": folder_id,
            "status": "imported",
            "status_changed_at": firestore.SERVER_TIMESTAMP,
        })
        return True

    result = _txn(db.transaction())
    if result:
        logger.info("Created imported record from Drive: %s (client=%s)", doc_id, client_name)
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
    """Create a placeholder queued record for a meeting.

    Returns True if created or reset from import error, False if already existed.
    """
    db = _get_db()
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
    _get_db().collection(COLLECTION_NAME).document(f"tldv-{meeting_id}").update({
        "status": "downloading",
        "status_changed_at": firestore.SERVER_TIMESTAMP,
    })
    logger.info("Marked downloading: %s", meeting_id)


def mark_download_error(meeting_id: str, error: str) -> None:
    _get_db().collection(COLLECTION_NAME).document(f"tldv-{meeting_id}").update({
        "status": "error",
        "error": error,
        "error_stage": "import",
        "status_changed_at": firestore.SERVER_TIMESTAMP,
    })
    logger.info("Marked download error: %s — %s", meeting_id, error)


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


def update_transcript_client(
    doc_id: str,
    new_client_name: str,
    new_drive_folder: str,
    new_content_hash: str,
) -> None:
    """Update client_name, drive_folder and content_hash without resetting status or utterance_hashes."""
    _get_db().collection(COLLECTION_NAME).document(doc_id).update({
        "client_name": new_client_name,
        "drive_folder": new_drive_folder,
        "content_hash": new_content_hash,
    })
    logger.info("Updated transcript client: %s → %s", doc_id, new_client_name)


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


def get_stale_syncing() -> list[str]:
    """Return doc IDs stuck in syncing beyond the stale threshold."""
    db = _get_db()
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=STALE_SYNCING_MINUTES)

    syncing_docs = (
        db.collection(COLLECTION_NAME)
        .where(filter=firestore.FieldFilter("status", "==", "syncing"))
        .stream()
    )

    stale = []
    for doc in syncing_docs:
        changed_at = doc.to_dict().get("status_changed_at")
        if changed_at and changed_at < cutoff:
            stale.append(doc.id)
            logger.info("Stale syncing detected: %s", doc.id)

    return stale

    return recovered


def get_error_docs() -> list[str]:
    """Return doc IDs with error status, excluding import-stage errors (no Google Doc yet)."""
    db = _get_db()
    error_docs = (
        db.collection(COLLECTION_NAME)
        .where(filter=firestore.FieldFilter("status", "==", "error"))
        .stream()
    )
    result = []
    for doc in error_docs:
        if (doc.to_dict() or {}).get("error_stage") == "import":
            continue
        result.append(doc.id)
        logger.info("Error doc detected: %s", doc.id)
    return result



def _speaker_key(name: str) -> str:
    import hashlib
    return "s" + hashlib.md5(name.encode()).hexdigest()[:12]


def update_client_speakers(client_name: str, speakers: list[str], delta: int = 1) -> None:
    """Increment or decrement speaker counts in clients/{client_name}.speakers."""
    if not speakers:
        return
    db = _get_db()
    updates = {f"speakers.{_speaker_key(s)}": firestore.Increment(delta) for s in speakers}
    db.collection(CLIENTS_COLLECTION).document(client_name).set(updates, merge=True)
    logger.debug("Updated speakers for %s (delta=%d): %s", client_name, delta, speakers)


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
        update_client_speakers(client_name, speakers)
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


def rebuild_client_speakers() -> int:
    """Rebuild clients.speakers per client from scratch.
    Counts all synced docs in Python, then writes the final map in a single atomic update.
    Returns total number of unique speaker keys updated."""
    db = _get_db()
    updated = 0

    for client_doc in db.collection(CLIENTS_COLLECTION).stream():
        client_name = client_doc.id

        docs = (
            db.collection(COLLECTION_NAME)
            .where(filter=firestore.FieldFilter("client_name", "==", client_name))
            .where(filter=firestore.FieldFilter("status", "==", "synced"))
            .stream()
        )

        speaker_counts: dict[str, int] = {}
        for doc in docs:
            for speaker in (doc.to_dict() or {}).get("speakers", []):
                key = _speaker_key(speaker)
                speaker_counts[key] = speaker_counts.get(key, 0) + 1

        doc_ref = db.collection(CLIENTS_COLLECTION).document(client_name)
        doc_data = doc_ref.get().to_dict() or {}

        # New flat fields with computed counts
        new_flat = {f"speakers.{key}": count for key, count in speaker_counts.items()}

        # Delete stale flat fields (speakers no longer present in synced docs)
        to_delete = {
            k: firestore.DELETE_FIELD
            for k in doc_data
            if k.startswith("speakers.") and k not in new_flat
        }
        # Delete nested `speakers` map if it exists (left from wrong .update() calls)
        if "speakers" in doc_data and isinstance(doc_data.get("speakers"), dict):
            to_delete["speakers"] = firestore.DELETE_FIELD

        final = {**to_delete, **new_flat}
        if final:
            doc_ref.set(final, merge=True)

        logger.info("Rebuilt speakers for %s: %d unique speakers", client_name, len(speaker_counts))
        updated += len(speaker_counts)

    return updated


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


def get_transcript_record(doc_id: str) -> dict | None:
    """Return transcript_index record for a doc, or None if not found."""
    doc = _get_db().collection(COLLECTION_NAME).document(doc_id).get()
    if not doc.exists:
        return None
    return doc.to_dict()


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
