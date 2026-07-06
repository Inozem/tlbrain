import logging
from datetime import datetime, timezone, timedelta

from google.cloud import firestore

from core.config import get_root_folder_id

logger = logging.getLogger(__name__)

COLLECTION_NAME = "transcript_index"
CLIENTS_COLLECTION = "clients"
FOLDERS_COLLECTION = "folders"
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


def ensure_imported(doc_id: str, parent_id: str, source_file: str = "") -> bool:
    """Create transcript_index record with status=imported if it doesn't exist. Returns True if created."""
    db = _get_db()
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


def move_transcript_record(doc_id: str, new_parent_id: str) -> None:
    """Update parent_id, reset status to imported for reindexing.

    modifiedTime is deleted so the checker detects a change and re-enqueues sync,
    even though moving a file in Drive does not update its modifiedTime.
    """
    _get_db().collection(COLLECTION_NAME).document(doc_id).update({
        "parent_id": new_parent_id,
        "status": "imported",
        "modifiedTime": firestore.DELETE_FIELD,
        "content_hash": firestore.DELETE_FIELD,
        "version": firestore.DELETE_FIELD,
        "error": None,
        "status_changed_at": firestore.SERVER_TIMESTAMP,
    })
    logger.info("Moved transcript record: %s → parent=%s", doc_id, new_parent_id)



def update_transcript_source_file(doc_id: str, source_file: str) -> None:
    _get_db().collection(COLLECTION_NAME).document(doc_id).update({"source_file": source_file})
    logger.info("Updated source_file for %s: %r", doc_id, source_file)


def get_unassigned() -> dict:
    """Return count and list of unassigned transcripts (in _unassigned folder)."""
    db = _get_db()
    unassigned_folder_id = None
    for doc in db.collection(FOLDERS_COLLECTION).where(
        filter=firestore.FieldFilter("name", "==", "_unassigned")
    ).stream():
        unassigned_folder_id = doc.id
        break

    if not unassigned_folder_id:
        return {"count": 0, "transcripts": []}

    transcripts = []
    for d in db.collection(COLLECTION_NAME).where(
        filter=firestore.FieldFilter("parent_id", "==", unassigned_folder_id)
    ).stream():
        data = d.to_dict() or {}
        if data.get("status") not in ("queued", "downloading"):
            transcripts.append({"doc_id": d.id, "dialog_date": data.get("dialog_date", "")})
    transcripts.sort(key=lambda x: x["dialog_date"], reverse=True)
    return {"count": len(transcripts), "transcripts": transcripts}


def list_transcripts(
    folder_ids: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> dict:
    """Return paginated list of transcripts sorted by dialog_date desc.

    Skips placeholder records (queued / downloading).
    Returns: {total, returned, offset, limit, has_more, transcripts}
    Each transcript: {doc_id, parent_id, title, dialog_date, status}
    """
    db = _get_db()
    transcripts = []
    for doc in db.collection(COLLECTION_NAME).stream():
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


_NO_RETRY_STAGES = {"import", "invalid_format"}


def get_error_docs() -> list[str]:
    """Return doc IDs with error status, excluding stages that should not be auto-retried."""
    db = _get_db()
    error_docs = (
        db.collection(COLLECTION_NAME)
        .where(filter=firestore.FieldFilter("status", "==", "error"))
        .stream()
    )
    result = []
    for doc in error_docs:
        if (doc.to_dict() or {}).get("error_stage") in _NO_RETRY_STAGES:
            continue
        result.append(doc.id)
        logger.info("Error doc detected: %s", doc.id)
    return result


def update_skipped_utterances(doc_id: str, skipped_utterances: list[int]) -> None:
    _get_db().collection(COLLECTION_NAME).document(doc_id).update({
        "skipped_utterances": sorted(set(skipped_utterances)),
    })


def get_docs_with_skipped_utterances() -> list[dict]:
    docs = (
        _get_db().collection(COLLECTION_NAME)
        .where(filter=firestore.FieldFilter("skipped_utterances_count", ">", 0))
        .stream()
    )
    result = []
    for doc in docs:
        data = doc.to_dict() or {}
        result.append({
            "doc_id": doc.id,
            "client_name": data.get("client_name", ""),
            "skipped_utterances": data.get("skipped_utterances", []),
        })
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
    ref = db.collection(CLIENTS_COLLECTION).document(client_name)
    try:
        ref.update(updates)
    except Exception:
        ref.set({"speakers": {}}, merge=True)
        ref.update(updates)
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


def sync_clients_from_drive(folders: list[dict[str, str]]) -> tuple[int, list[dict[str, str]]]:
    """Upsert clients/{name} for each Drive folder. Updates folder_id if changed.

    Detects in-place folder renames (folder_id already registered under a different
    name) and returns them WITHOUT mutating — the checker dispatches a folder task
    (clients copy) plus per-doc file tasks to reconcile Qdrant and transcript_index.
    A renamed folder is NOT auto-registered, to avoid a duplicate folder_id mapping.

    Returns (count of new records, renames=[{"old", "new", "folder_id"}]).

    Each folder dict must have: {name: str, id: str}
    """
    db = _get_db()
    created = 0
    renames: list[dict[str, str]] = []
    for folder in folders:
        name = folder["name"]
        folder_id = folder["id"]
        ref = db.collection(CLIENTS_COLLECTION).document(name)
        snapshot = ref.get()
        if not snapshot.exists:
            old_name = get_client_name_by_folder_id(folder_id)
            if old_name and old_name != name:
                renames.append({"old": old_name, "new": name, "folder_id": folder_id})
                logger.info("Detected folder rename via Drive: %s → %s (%s)", old_name, name, folder_id)
            else:
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

    return created, renames


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

        # New nested paths with computed counts
        new_updates = {f"speakers.{key}": count for key, count in speaker_counts.items()}

        # Delete stale entries from nested speakers map (speakers no longer in any synced doc)
        existing_speaker_keys = set((doc_data.get("speakers") or {}).keys())
        stale_keys = existing_speaker_keys - set(speaker_counts.keys())
        to_delete = {f"speakers.{key}": firestore.DELETE_FIELD for key in stale_keys}

        final = {**to_delete, **new_updates}
        if final:
            doc_ref.update(final)

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


def rename_client_records(old_name: str, new_name: str, folder_id: str) -> list[str]:
    """Rename client in Firestore: copy clients record, delete old, update all transcript_index docs.

    Clears modifiedTime and content_hash so the checker re-enqueues reindexing.
    Returns list of affected transcript doc_ids.
    """
    db = _get_db()

    old_ref = db.collection(CLIENTS_COLLECTION).document(old_name)
    new_ref = db.collection(CLIENTS_COLLECTION).document(new_name)

    old_data = (old_ref.get().to_dict() or {}).copy()
    old_data["folder_id"] = folder_id
    new_ref.set(old_data)
    old_ref.delete()
    logger.info("Renamed client record: %s → %s", old_name, new_name)

    docs = (
        db.collection(COLLECTION_NAME)
        .where(filter=firestore.FieldFilter("client_name", "==", old_name))
        .stream()
    )

    doc_ids: list[str] = []
    batch = db.batch()
    batch_size = 0
    for doc in docs:
        batch.update(db.collection(COLLECTION_NAME).document(doc.id), {
            "client_name": new_name,
            "modifiedTime": firestore.DELETE_FIELD,
            "content_hash": firestore.DELETE_FIELD,
        })
        doc_ids.append(doc.id)
        batch_size += 1
        if batch_size == 500:
            batch.commit()
            batch = db.batch()
            batch_size = 0
    if batch_size:
        batch.commit()

    logger.info("Updated %d transcript_index docs: %s → %s", len(doc_ids), old_name, new_name)
    return doc_ids


def reconcile_client_record(old_name: str, new_name: str, folder_id: str) -> None:
    """Folder-rename reconciliation (folder task): ensure clients/{new_name} carries the
    folder mapping + metadata, and remove clients/{old_name}.

    Speaker counts are NOT copied — they self-assemble from per-doc increments, so a
    merge-set is used to avoid clobbering counts already written by file tasks. Idempotent
    and order-independent. Does NOT touch transcript_index (file tasks own client_name there).
    """
    db = _get_db()
    old_ref = db.collection(CLIENTS_COLLECTION).document(old_name)
    new_ref = db.collection(CLIENTS_COLLECTION).document(new_name)

    old_data = old_ref.get().to_dict() or {}
    record: dict = {"folder_id": folder_id, "status": old_data.get("status", "active")}
    if old_data.get("description"):
        record["description"] = old_data["description"]
    new_ref.set(record, merge=True)
    old_ref.delete()
    logger.info("Reconciled client record: %s → %s (%s)", old_name, new_name, folder_id)


def get_doc_ids_by_client(client_name: str) -> list[str]:
    """Return all transcript_index doc IDs for a client."""
    db = _get_db()
    docs = (
        db.collection(COLLECTION_NAME)
        .where(filter=firestore.FieldFilter("client_name", "==", client_name))
        .select([])
        .stream()
    )
    return [doc.id for doc in docs]


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


# ---------------------------------------------------------------------------
# folders/ collection (v2)
# ---------------------------------------------------------------------------

def upsert_folder(folder_id: str, name: str, parent_id: str | None) -> None:
    """Register or update a folder in folders/{folder_id}."""
    db = _get_db()
    ref = db.collection(FOLDERS_COLLECTION).document(folder_id)
    snapshot = ref.get()
    if snapshot.exists:
        data = snapshot.to_dict() or {}
        updates: dict = {}
        if data.get("name") != name:
            updates["name"] = name
        if data.get("parent_id") != parent_id:
            updates["parent_id"] = parent_id
        if updates:
            ref.update(updates)
            logger.info("Updated folder %s: %s", folder_id, updates)
    else:
        ref.set({
            "folder_id": folder_id,
            "name": name,
            "parent_id": parent_id,
            "speakers": {},
            "description": None,
            "created_at": firestore.SERVER_TIMESTAMP,
        })
        logger.info("Registered folder: %s name=%r parent=%s", folder_id, name, parent_id)


def get_folder_by_id(folder_id: str) -> dict | None:
    """Return folders/{folder_id} data, or None if not found."""
    doc = _get_db().collection(FOLDERS_COLLECTION).document(folder_id).get()
    if not doc.exists:
        return None
    return doc.to_dict()


def get_all_folders() -> dict[str, dict]:
    """Return {folder_id: data} for all registered folders."""
    db = _get_db()
    return {doc.id: (doc.to_dict() or {}) for doc in db.collection(FOLDERS_COLLECTION).stream()}


def folder_name_exists(name: str, parent_id: str) -> bool:
    """Return True if a folder with this name already exists under parent_id."""
    db = _get_db()
    docs = (
        db.collection(FOLDERS_COLLECTION)
        .where("name", "==", name)
        .where("parent_id", "==", parent_id)
        .limit(1)
        .stream()
    )
    return any(True for _ in docs)


def aggregate_transcripts_by_folder() -> dict[str, dict]:
    """Return {parent_id: {count, last_date, last_doc_id}} for all synced transcripts."""
    db = _get_db()
    agg: dict[str, dict] = {}
    for doc in db.collection(COLLECTION_NAME).where(
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


def expand_subtree(folder_id: str) -> list[str]:
    """Return folder_id plus all its descendants from the folders collection.

    BFS over Firestore data — no Drive calls. Includes the root folder_id itself.
    """
    all_folders = get_all_folders()
    children: dict[str, list[str]] = {}
    for fid, data in all_folders.items():
        pid = data.get("parent_id")
        if pid:
            children.setdefault(pid, []).append(fid)

    result: list[str] = []
    queue = [folder_id]
    while queue:
        fid = queue.pop(0)
        result.append(fid)
        queue.extend(children.get(fid, []))
    return result


def orphan_folder_children(folder_id: str) -> int:
    """Clear parent_id for all subfolders that are direct children of folder_id."""
    db = _get_db()
    count = 0
    for doc in db.collection(FOLDERS_COLLECTION).where("parent_id", "==", folder_id).stream():
        doc.reference.update({"parent_id": None})
        count += 1
    if count:
        logger.info("Orphaned %d subfolder(s) of %s", count, folder_id)
    return count


def orphan_transcript_children(folder_id: str) -> int:
    """Clear parent_id for all transcript_index docs that are direct children of folder_id."""
    db = _get_db()
    count = 0
    for doc in db.collection(COLLECTION_NAME).where("parent_id", "==", folder_id).stream():
        doc.reference.update({"parent_id": None})
        count += 1
    if count:
        logger.info("Orphaned %d transcript(s) of %s", count, folder_id)
    return count


def delete_folder(folder_id: str) -> None:
    """Remove a folder record from the folders collection."""
    _get_db().collection(FOLDERS_COLLECTION).document(folder_id).delete()
    logger.info("Deleted folder record: %s", folder_id)


def resolve_folder_path(path: list[str]) -> list[str]:
    """Resolve a path like ["Clients - Active", "Acme Corp"] to matching folder_ids.

    Handles duplicate folder names: collects all candidates at each segment.
    Returns folder_ids of the last path segment (not expanded to subtree).
    Returns empty list if path is empty or no match found.
    """
    if not path:
        return []

    root_folder_id = get_root_folder_id()
    all_folders = get_all_folders()

    children: dict[str, list[tuple[str, str]]] = {}
    for fid, data in all_folders.items():
        pid = data.get("parent_id")
        if pid:
            children.setdefault(pid, []).append((fid, data.get("name", "")))

    current: list[str] = [root_folder_id]
    for segment in path:
        next_candidates: list[str] = []
        for parent_id in current:
            for fid, name in children.get(parent_id, []):
                if name == segment:
                    next_candidates.append(fid)
        if not next_candidates:
            return []
        current = next_candidates
    return current
