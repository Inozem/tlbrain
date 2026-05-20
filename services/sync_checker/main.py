import logging
import os
from datetime import datetime, timedelta, timezone

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
    get_drive_sync_token,
    get_error_docs,
    get_stale_syncing,
    set_drive_sync_token,
    sync_clients_from_drive,
)
from core.utils.tasks import enqueue_task

from core.utils.logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

STALE_MINUTES = 15
GOOGLE_DOC_MIME = "application/vnd.google-apps.document"


@functions_framework.http
def checker(request):
    root_folder_id = get_root_folder_id()
    sync_url = os.environ["VECTOR_SYNC_URL"]
    queue_name = os.environ.get("VECTOR_SYNC_QUEUE", "tlbrain-vector-sync-queue")
    tldv_import_queue = os.environ.get("TLDV_IMPORT_QUEUE")
    tldv_import_url = os.environ.get("TLDV_IMPORT_SERVICE_URL")

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

    recovered_downloading = _recover_stale_downloading(db, tldv_import_queue, tldv_import_url)
    if recovered_downloading:
        logger.info("Recovered stale downloading: %d doc(s)", len(recovered_downloading))

    folders = list_client_folders()
    clients_synced = sync_clients_from_drive(folders)
    if clients_synced:
        logger.info("Auto-registered %d client(s) from Drive", clients_synced)

    page_token = get_drive_sync_token()
    queued = 0

    if page_token:
        try:
            changes, new_token = get_drive_changes(page_token)
            queued = _process_changes(changes, sync_url, queue_name)
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
        "Checker done — queued=%d stale_syncing=%d error_docs=%d recovered_downloading=%d",
        queued, len(stale_syncing), len(error_docs), len(recovered_downloading),
    )
    return {
        "queued": queued,
        "stale_syncing": len(stale_syncing),
        "error_docs": len(error_docs),
        "recovered_downloading": len(recovered_downloading),
    }, 200


def _full_scan(root_folder_id: str, sync_url: str, queue_name: str, db: firestore.Client) -> int:
    """Full Drive scan: enqueue changed docs. Saves new page token."""
    start_token = get_start_page_token()

    files = scan_root_folder()
    queued = 0

    for file in files:
        doc_id = file["doc_id"]
        snapshot = db.collection(COLLECTION_NAME).document(doc_id).get()
        existing = snapshot.to_dict() if snapshot.exists else None

        if (
            existing
            and existing.get("modifiedTime") == file["modifiedTime"]
            and existing.get("client_name") == file["client_name"]
            and existing.get("status") == "synced"
        ):
            continue

        if enqueue_task(queue_name=queue_name, url=f"{sync_url}/sync/doc/{doc_id}"):
            queued += 1

    set_drive_sync_token(start_token)
    return queued


def _process_changes(changes: list[dict], sync_url: str, queue_name: str) -> int:
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

        if enqueue_task(queue_name=queue_name, url=f"{sync_url}/sync/doc/{doc_id}"):
            queued += 1

    return queued


def _recover_stale_downloading(
    db: firestore.Client,
    tldv_import_queue: str | None,
    tldv_import_url: str | None,
) -> list[str]:
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=STALE_MINUTES)
    downloading_docs = (
        db.collection(COLLECTION_NAME)
        .where(filter=firestore.FieldFilter("status", "==", "downloading"))
        .stream()
    )
    recovered = []
    for doc in downloading_docs:
        data = doc.to_dict()
        changed_at = data.get("status_changed_at")
        if not changed_at or changed_at >= cutoff:
            continue

        meeting_id = data.get("meeting_id")
        provider = data.get("provider", "tldv")

        if provider == "tldv" and meeting_id and tldv_import_queue and tldv_import_url:
            enqueue_task(
                queue_name=tldv_import_queue,
                url=f"{tldv_import_url}/import",
                body={"meeting_id": meeting_id},
            )

        db.collection(COLLECTION_NAME).document(doc.id).update({
            "status": "queued",
            "status_changed_at": firestore.SERVER_TIMESTAMP,
        })
        recovered.append(doc.id)
        logger.info("Recovered stale downloading: %s (provider=%s)", doc.id, provider)

    return recovered
