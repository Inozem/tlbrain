import logging
import os

import functions_framework
from google.cloud import firestore
from googleapiclient.errors import HttpError

from core.config import get_root_folder_id
from core.google_drive.drive_client import (
    get_drive_changes,
    get_start_page_token,
    scan_root_folder,
)
from core.google_drive.firestore import (
    COLLECTION_NAME,
    delete_folder,
    get_all_folders,
    get_drive_sync_token,
    get_error_docs,
    get_folder_by_id,
    get_stale_syncing,
    move_transcript_record,
    set_drive_sync_token,
    update_transcript_source_file,
    upsert_folder,
)
from core.utils.tasks import enqueue_task

from core.utils.logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

GOOGLE_DOC_MIME = "application/vnd.google-apps.document"
GOOGLE_FOLDER_MIME = "application/vnd.google-apps.folder"


@functions_framework.http
def checker(request):
    root_folder_id = get_root_folder_id()
    sync_url = os.environ["VECTOR_SYNC_URL"]
    queue_name = os.environ.get("VECTOR_SYNC_QUEUE", "tlbrain-vector-sync-queue")
    db = firestore.Client()

    body = request.get_json(silent=True) or {}
    doc_id = body.get("doc_id")

    if doc_id:
        enqueue_task(queue_name=queue_name, url=f"{sync_url}/sync/doc/{doc_id}")
        logger.info("Targeted resync: doc_id=%s", doc_id)
        return {"mode": "doc", "doc_id": doc_id, "queued": 1}, 200

    stale_syncing = get_stale_syncing()
    for doc_id in stale_syncing:
        enqueue_task(queue_name=queue_name, url=f"{sync_url}/sync/doc/{doc_id}")
    if stale_syncing:
        logger.info("Re-enqueued stale syncing: %d doc(s)", len(stale_syncing))

    error_docs = get_error_docs()
    for doc_id in error_docs:
        enqueue_task(queue_name=queue_name, url=f"{sync_url}/sync/doc/{doc_id}")
    if error_docs:
        logger.info("Re-enqueued error docs: %d doc(s)", len(error_docs))

    page_token = get_drive_sync_token()
    queued = 0

    if page_token:
        try:
            changes, new_token = get_drive_changes(page_token)
            queued = _process_changes(changes, sync_url, queue_name, db, root_folder_id)
            set_drive_sync_token(new_token)
            logger.info("Incremental sync: %d change(s) processed", len(changes))
        except HttpError as e:
            if e.resp.status == 400:
                logger.warning("Drive page token expired, falling back to full scan")
                queued = _full_scan(root_folder_id, sync_url, queue_name, db)
            else:
                raise
    else:
        logger.info("No page token found, running full scan")
        queued = _full_scan(root_folder_id, sync_url, queue_name, db)

    logger.info(
        "Checker done — queued=%d stale_syncing=%d error_docs=%d",
        queued, len(stale_syncing), len(error_docs),
    )
    return {
        "queued": queued,
        "stale_syncing": len(stale_syncing),
        "error_docs": len(error_docs),
    }, 200


def _full_scan(
    root_folder_id: str,
    sync_url: str,
    queue_name: str,
    db: firestore.Client,
) -> int:
    start_token = get_start_page_token()

    live_folders, live_docs = scan_root_folder()
    live_folder_ids = {f["id"] for f in live_folders}
    live_doc_ids = {doc["doc_id"] for doc in live_docs}

    # Register/update folders before processing docs
    for folder in live_folders:
        upsert_folder(folder["id"], folder["name"], folder["parent_id"])
    logger.info("Upserted %d folder(s) from Drive", len(live_folders))

    # Reconcile folders/ — remove records for folders no longer under ROOT
    for folder_id in get_all_folders():
        if folder_id not in live_folder_ids:
            delete_folder(folder_id)
            logger.info("Deleted orphaned folder record: %s", folder_id)

    queued = 0

    for file in live_docs:
        doc_id = file["doc_id"]
        snapshot = db.collection(COLLECTION_NAME).document(doc_id).get()
        existing = snapshot.to_dict() if snapshot.exists else None

        if existing and existing.get("status") in ("queued", "downloading"):
            continue

        if existing and file.get("name") and existing.get("source_file") != file["name"]:
            update_transcript_source_file(doc_id, file["name"])

        if existing and existing.get("parent_id") and existing.get("parent_id") != file["parent_id"]:
            move_transcript_record(doc_id, file["parent_id"])
            existing = None  # force re-enqueue

        if (
            existing
            and existing.get("modifiedTime") == file["modifiedTime"]
            and existing.get("parent_id") == file["parent_id"]
            and existing.get("status") == "synced"
        ):
            continue

        if (
            existing
            and existing.get("error_stage") == "invalid_format"
            and existing.get("modifiedTime") == file["modifiedTime"]
        ):
            continue

        if enqueue_task(queue_name=queue_name, url=f"{sync_url}/sync/doc/{doc_id}"):
            queued += 1

    # Enqueue orphaned transcript_index records for deletion
    for doc in db.collection(COLLECTION_NAME).select(["status", "error_stage"]).stream():
        if doc.id in live_doc_ids:
            continue
        data = doc.to_dict() or {}
        status = data.get("status", "")
        if status in ("queued", "downloading"):
            continue
        if status == "error" and data.get("error_stage") == "import":
            continue
        if enqueue_task(queue_name=queue_name, url=f"{sync_url}/sync/doc/{doc.id}"):
            queued += 1
            logger.info("Enqueued orphan for deletion: %s (status=%s)", doc.id, status)

    set_drive_sync_token(start_token)
    return queued


def _process_changes(
    changes: list[dict],
    sync_url: str,
    queue_name: str,
    db: firestore.Client,
    root_folder_id: str,
) -> int:
    queued = 0

    for change in changes:
        file_id = change.get("fileId")
        if not file_id:
            continue

        if change.get("removed"):
            # No mimeType available — check folders/ to distinguish folder vs doc
            if get_folder_by_id(file_id):
                delete_folder(file_id)
            else:
                if enqueue_task(queue_name=queue_name, url=f"{sync_url}/sync/doc/{file_id}"):
                    queued += 1
            continue

        file = change.get("file", {})
        if file.get("trashed"):
            if file.get("mimeType") == GOOGLE_FOLDER_MIME:
                delete_folder(file_id)
            else:
                if enqueue_task(queue_name=queue_name, url=f"{sync_url}/sync/doc/{file_id}"):
                    queued += 1
            continue

        mime_type = file.get("mimeType")

        if mime_type == GOOGLE_FOLDER_MIME:
            # Folder rename or move — update folders/ only, no cascade to docs
            name = file.get("name")
            parents = file.get("parents", [])
            if name and parents:
                upsert_folder(file_id, name, parents[0])
            continue

        if mime_type != GOOGLE_DOC_MIME:
            continue

        parents = file.get("parents", [])
        if not parents:
            continue

        parent_id = parents[0]
        if parent_id != root_folder_id and not get_folder_by_id(parent_id):
            continue

        snapshot = db.collection(COLLECTION_NAME).document(file_id).get()
        record = snapshot.to_dict() if snapshot.exists else None

        file_name = file.get("name")
        if record and file_name and record.get("source_file") != file_name:
            update_transcript_source_file(file_id, file_name)

        if record and record.get("parent_id") and record.get("parent_id") != parent_id:
            move_transcript_record(file_id, parent_id)

        if enqueue_task(queue_name=queue_name, url=f"{sync_url}/sync/doc/{file_id}"):
            queued += 1

    return queued
