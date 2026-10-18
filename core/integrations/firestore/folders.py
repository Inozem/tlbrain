"""Firestore repository helpers for the v2 Drive folders collection."""

import hashlib
import logging

from google.cloud import firestore

from core.integrations.firestore.client import FOLDERS_COLLECTION, get_client

logger = logging.getLogger(__name__)


def _db_or_client(db: firestore.Client | None = None) -> firestore.Client:
    return db if db is not None else get_client()


def speaker_key(name: str) -> str:
    """Return the stable key used to index a speaker name in folder documents."""
    return "s" + hashlib.md5(name.encode("utf-8")).hexdigest()[:12]


def upsert_folder(folder_id: str, name: str, parent_id: str | None, db: firestore.Client | None = None) -> None:
    """Register or update a folder at its stable Drive ID."""
    ref = _db_or_client(db).collection(FOLDERS_COLLECTION).document(folder_id)
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


def get_folder_by_id(folder_id: str, db: firestore.Client | None = None) -> dict | None:
    """Return a folder record, or None if it does not exist."""
    doc = _db_or_client(db).collection(FOLDERS_COLLECTION).document(folder_id).get()
    return doc.to_dict() if doc.exists else None


def get_all_folders(db: firestore.Client | None = None) -> dict[str, dict]:
    """Return all folder records keyed by their stable Drive IDs."""
    client = _db_or_client(db)
    return {doc.id: (doc.to_dict() or {}) for doc in client.collection(FOLDERS_COLLECTION).stream()}


def folder_name_exists(name: str, parent_id: str | None, db: firestore.Client | None = None) -> bool:
    """Return True when this name exists under the supplied nullable parent ID."""
    client = _db_or_client(db)
    docs = (
        client.collection(FOLDERS_COLLECTION)
        .where("name", "==", name)
        .where("parent_id", "==", parent_id)
        .limit(1)
        .stream()
    )
    return any(True for _ in docs)


def get_folder_id_by_name(name: str, db: firestore.Client | None = None) -> str | None:
    """Return the first folder ID with this name, for legacy unassigned lookup."""
    client = _db_or_client(db)
    docs = client.collection(FOLDERS_COLLECTION).where(
        filter=firestore.FieldFilter("name", "==", name)
    ).stream()
    for doc in docs:
        return doc.id
    return None


def orphan_folder_children(folder_id: str, db: firestore.Client | None = None) -> int:
    """Clear parent_id on all direct child folders."""
    client = _db_or_client(db)
    count = 0
    for doc in client.collection(FOLDERS_COLLECTION).where("parent_id", "==", folder_id).stream():
        doc.reference.update({"parent_id": None})
        count += 1
    if count:
        logger.info("Orphaned %d subfolder(s) of %s", count, folder_id)
    return count


def delete_folder(folder_id: str, db: firestore.Client | None = None) -> None:
    """Delete a folder record."""
    _db_or_client(db).collection(FOLDERS_COLLECTION).document(folder_id).delete()
    logger.info("Deleted folder record: %s", folder_id)


def get_folders_by_speaker(speaker: str, db: firestore.Client | None = None):
    """Query folder snapshots whose indexed count for speaker is positive."""
    client = _db_or_client(db)
    return client.collection(FOLDERS_COLLECTION).where(
        filter=firestore.FieldFilter(f"speakers.{speaker_key(speaker)}", ">", 0)
    ).stream()


def increment_folder_speakers(
    folder_id: str,
    speakers: list[str],
    delta: int = 1,
    db: firestore.Client | None = None,
) -> None:
    """Increment or decrement indexed speaker counts on a folder record."""
    if not speakers:
        return
    updates = {f"speakers.{speaker_key(speaker)}": firestore.Increment(delta) for speaker in speakers}
    _db_or_client(db).collection(FOLDERS_COLLECTION).document(folder_id).update(updates)
