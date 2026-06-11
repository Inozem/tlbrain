import logging
import os

import functions_framework
from google.cloud import firestore
from googleapiclient.errors import HttpError

from core.config import get_root_folder_id
from core.google_drive.drive_client import (
    get_drive_changes,
    get_start_page_token,
    list_client_folders,
    scan_root_folder,
)
from core.google_drive.firestore import (
    COLLECTION_NAME,
    get_client_name_by_folder_id,
    get_doc_ids_by_client,
    get_drive_sync_token,
    get_error_docs,
    get_stale_syncing,
    rebuild_client_speakers,
    set_drive_sync_token,
    sync_clients_from_drive,
    update_transcript_source_file,
)
from core.utils.tasks import enqueue_task

from core.utils.logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

GOOGLE_DOC_MIME = "application/vnd.google-apps.document"


@functions_framework.http
def checker(request):
    root_folder_id = get_root_folder_id()
    sync_url = os.environ["VECTOR_SYNC_URL"]
    queue_name = os.environ.get("VECTOR_SYNC_QUEUE", "tlbrain-vector-sync-queue")
    db = firestore.Client()

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

    folders = list_client_folders()
    clients_synced, renames = sync_clients_from_drive(folders)
    if clients_synced:
        logger.info("Auto-registered %d client(s) from Drive", clients_synced)
    rename_doc_ids: set[str] = set()
    rename_docs_queued = 0
    for r in renames:
        enqueue_task(
            queue_name=queue_name,
            url=f"{sync_url}/client/rename",
            task_id=f"rename-{r['folder_id']}",
            body={"old_client_name": r["old"], "new_client_name": r["new"], "folder_id": r["folder_id"]},
        )
        doc_ids = get_doc_ids_by_client(r["old"])
        for doc_id in doc_ids:
            enqueue_task(queue_name=queue_name, url=f"{sync_url}/sync/doc/{doc_id}")
            rename_doc_ids.add(doc_id)
        rename_docs_queued += len(doc_ids)
        logger.info(
            "Client rename %s → %s: folder task + %d file task(s)",
            r["old"], r["new"], len(doc_ids),
        )

    page_token = get_drive_sync_token()
    queued = 0

    if page_token:
        try:
            changes, new_token = get_drive_changes(page_token)
            queued = _process_changes(changes, sync_url, queue_name, db)
            set_drive_sync_token(new_token)
            logger.info("Incremental sync: %d change(s) processed", len(changes))
        except HttpError as e:
            if e.resp.status == 400:
                logger.warning("Drive page token expired, falling back to full scan")
                queued = _full_scan(root_folder_id, sync_url, queue_name, db, skip_doc_ids=rename_doc_ids)
            else:
                raise
    else:
        logger.info("No page token found, running full scan")
        queued = _full_scan(root_folder_id, sync_url, queue_name, db, skip_doc_ids=rename_doc_ids)

    logger.info(
        "Checker done — queued=%d renamed_folders=%d rename_docs_queued=%d stale_syncing=%d error_docs=%d",
        queued, len(renames), rename_docs_queued, len(stale_syncing), len(error_docs),
    )
    return {
        "queued": queued,
        "renamed_folders": len(renames),
        "rename_docs_queued": rename_docs_queued,
        "stale_syncing": len(stale_syncing),
        "error_docs": len(error_docs),
    }, 200


def _full_scan(
    root_folder_id: str,
    sync_url: str,
    queue_name: str,
    db: firestore.Client,
    skip_doc_ids: set[str] | None = None,
) -> int:
    """Full Drive scan: enqueue changed docs and orphaned Firestore records. Saves new page token.

    skip_doc_ids: docs already enqueued by the rename pass — skipped to avoid double-enqueue.
    """
    start_token = get_start_page_token()

    # Rebuild speaker counts BEFORE queuing new tasks — at this point all previously
    # synced docs are in status=synced and will be counted correctly.
    # Docs queued below will be handled incrementally by update_client_speakers when they sync.
    updated = rebuild_client_speakers()
    logger.info("Rebuilt client speakers: %d entries", updated)

    files = scan_root_folder()
    drive_doc_ids = {file["doc_id"] for file in files}
    queued = 0

    for file in files:
        doc_id = file["doc_id"]
        if skip_doc_ids and doc_id in skip_doc_ids:
            continue
        snapshot = db.collection(COLLECTION_NAME).document(doc_id).get()
        existing = snapshot.to_dict() if snapshot.exists else None

        if existing and file.get("name") and existing.get("source_file") != file["name"]:
            update_transcript_source_file(doc_id, file["name"])

        if (
            existing
            and existing.get("modifiedTime") == file["modifiedTime"]
            and existing.get("client_name") == file["client_name"]
            and existing.get("status") == "synced"
        ):
            continue

        if (
            existing
            and existing.get("error_stage") == "invalid_format"
            and existing.get("modifiedTime") == file["modifiedTime"]
        ):
            continue

        if existing and existing.get("status") in ("queued", "downloading"):
            continue

        if enqueue_task(queue_name=queue_name, url=f"{sync_url}/sync/doc/{doc_id}"):
            queued += 1

    for doc in db.collection(COLLECTION_NAME).select(["status", "error_stage"]).stream():
        if doc.id in drive_doc_ids:
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


def _process_changes(changes: list[dict], sync_url: str, queue_name: str, db: firestore.Client) -> int:
    """Enqueue tasks for incremental Drive changes. Returns count queued."""
    queued = 0

    for change in changes:
        doc_id = change.get("fileId")
        if not doc_id:
            continue

        if change.get("removed"):
            enqueue_task(queue_name=queue_name, url=f"{sync_url}/sync/doc/{doc_id}")
            queued += 1
            continue

        file = change.get("file", {})
        if file.get("trashed"):
            enqueue_task(queue_name=queue_name, url=f"{sync_url}/sync/doc/{doc_id}")
            queued += 1
            continue

        if file.get("mimeType") != GOOGLE_DOC_MIME:
            continue

        parents = file.get("parents", [])
        if not parents:
            continue

        if not get_client_name_by_folder_id(parents[0]):
            continue

        file_name = file.get("name")
        if file_name:
            snapshot = db.collection(COLLECTION_NAME).document(doc_id).get()
            record = snapshot.to_dict() if snapshot.exists else None
            if record and record.get("source_file") != file_name:
                update_transcript_source_file(doc_id, file_name)

        if enqueue_task(queue_name=queue_name, url=f"{sync_url}/sync/doc/{doc_id}"):
            queued += 1

    return queued
