import logging

from google.cloud import firestore as fs
from qdrant_client.models import FieldCondition, Filter, MatchValue

from core.google_drive.drive_client import scan_root_folder
from core.google_drive.firestore import (
    CLIENTS_COLLECTION,
    COLLECTION_NAME,
    FOLDERS_COLLECTION,
    upsert_folder,
)
from core.qdrant.client import get_client as get_qdrant_client
from core.qdrant.schema import get_collection_name

logger = logging.getLogger(__name__)


def _get_db() -> fs.Client:
    return fs.Client()


def step_build_folders(dry_run: bool = False) -> dict:
    """Populate folders/ from Drive structure; migrate speakers from clients/."""
    live_folders, _ = scan_root_folder()

    db = _get_db()
    folder_to_speakers: dict[str, dict] = {}
    for doc in db.collection(CLIENTS_COLLECTION).stream():
        data = doc.to_dict() or {}
        folder_id = data.get("folder_id")
        speakers = data.get("speakers")
        if folder_id and speakers:
            folder_to_speakers[folder_id] = speakers

    count = 0
    for f in live_folders:
        logger.info("[step1] folder %s (%r) parent=%s", f["id"], f["name"], f["parent_id"])
        if not dry_run:
            upsert_folder(f["id"], f["name"], f["parent_id"])
            speakers = folder_to_speakers.get(f["id"])
            if speakers:
                db.collection(FOLDERS_COLLECTION).document(f["id"]).update({"speakers": speakers})
        count += 1

    return {"folders_processed": count, "dry_run": dry_run}


def step_update_transcript_index(dry_run: bool = False) -> dict:
    """Set parent_id and remove client_name in transcript_index."""
    db = _get_db()

    client_to_folder: dict[str, str] = {}
    for doc in db.collection(CLIENTS_COLLECTION).stream():
        folder_id = (doc.to_dict() or {}).get("folder_id")
        if folder_id:
            client_to_folder[doc.id] = folder_id

    updated = skipped = 0
    batch = db.batch()
    batch_size = 0

    for doc in db.collection(COLLECTION_NAME).stream():
        data = doc.to_dict() or {}
        if "client_name" not in data:
            skipped += 1
            continue

        parent_id = data.get("parent_id") or client_to_folder.get(data["client_name"])
        if not parent_id:
            logger.warning("[step2] no folder_id for client_name=%r doc=%s — skipping", data["client_name"], doc.id)
            skipped += 1
            continue

        logger.info("[step2] %s client_name=%r → parent_id=%s", doc.id, data["client_name"], parent_id)
        if not dry_run:
            batch.update(doc.reference, {"parent_id": parent_id, "client_name": fs.DELETE_FIELD})
            batch_size += 1
            if batch_size == 500:
                batch.commit()
                batch = db.batch()
                batch_size = 0
        updated += 1

    if not dry_run and batch_size:
        batch.commit()

    return {"updated": updated, "skipped": skipped, "dry_run": dry_run}


def step_migrate_qdrant(dry_run: bool = False) -> dict:
    """Replace client_name with parent_id in Qdrant payload (no re-embed)."""
    import os
    from qdrant_client import QdrantClient

    db = _get_db()
    qdrant = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
        timeout=60,
    )
    collection = get_collection_name()

    client_to_folder: dict[str, str] = {}
    for doc in db.collection(CLIENTS_COLLECTION).stream():
        folder_id = (doc.to_dict() or {}).get("folder_id")
        if folder_id:
            client_to_folder[doc.id] = folder_id

    doc_to_parent: dict[str, str] = {}
    for doc in db.collection(COLLECTION_NAME).stream():
        data = doc.to_dict() or {}
        parent_id = data.get("parent_id") or client_to_folder.get(data.get("client_name", ""))
        if parent_id:
            doc_to_parent[doc.id] = parent_id

    updated = 0
    for doc_id, parent_id in doc_to_parent.items():
        f = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])
        logger.info("[step3] doc=%s parent_id=%s", doc_id, parent_id)
        if not dry_run:
            qdrant.set_payload(collection_name=collection, payload={"parent_id": parent_id}, points=f)
            qdrant.delete_payload(collection_name=collection, keys=["client_name"], points=f)
        updated += 1

    return {"docs_processed": updated, "dry_run": dry_run}


def step_delete_clients(dry_run: bool = False) -> dict:
    """Delete clients/ collection after migration."""
    db = _get_db()
    count = 0
    batch = db.batch()
    batch_size = 0

    for doc in db.collection(CLIENTS_COLLECTION).stream():
        logger.info("[step4] deleting clients/%s", doc.id)
        if not dry_run:
            batch.delete(doc.reference)
            batch_size += 1
            if batch_size == 500:
                batch.commit()
                batch = db.batch()
                batch_size = 0
        count += 1

    if not dry_run and batch_size:
        batch.commit()

    return {"deleted": count, "dry_run": dry_run}
