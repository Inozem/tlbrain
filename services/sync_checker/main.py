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
    recover_errors,
    recover_stale_syncing,
    set_drive_sync_token,
    sync_clients_from_drive,
)
from core.utils.tasks import enqueue_task

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STALE_MINUTES = 15
GOOGLE_DOC_MIME = "application/vnd.google-apps.document"


@functions_framework.http
def checker(request):
    root_folder_id = get_root_folder_id()
    sync_url = os.environ["VECTOR_SYNC_URL"]
    queue_name = os.environ["VECTOR_SYNC_QUEUE"]
    tldv_import_queue = os.environ.get("TLDV_IMPORT_QUEUE")
    tldv_import_url = os.environ.get("TLDV_IMPORT_SERVICE_URL")

    db = firestore.Client()

    recovered = recover_stale_syncing()
    if recovered:
        logger.info("Recovered stale syncing: %d doc(s)", len(recovered))

    recovered_errors = recover_errors()
    if recovered_errors:
        logger.info("Recovered errors: %d doc(s)", len(recovered_errors))

    recovered_downloading = _recover_stale_downloading(db, tldv_import_queue, tldv_import_url)
    if recovered_downloading:
        logger.info("Recovered stale downloading: %d doc(s)", len(recovered_downloading))

    folders = list_client_folders()
    clients_synced = sync_clients_from_drive(folders)
    if clients_synced:
        logger.info("Auto-registered %d client(s) from Drive", clients_synced)

    page_token = get_drive_sync_token()
    queued = 0
    marked = 0

    if page_token:
        try:
            changes, new_token = get_drive_changes(page_token)
            marked, queued = _process_changes(changes, root_folder_id, sync_url, queue_name, db)
            set_drive_sync_token(new_token)
            logger.info("Incremental sync: %d change(s) processed", len(changes))
        except HttpError as e:
            if e.resp.status == 400:
                logger.warning("Drive page token expired, falling back to full scan")
                marked, queued = _full_scan(root_folder_id, sync_url, queue_name, db)
            else:
                raise
    else:
        logger.info("No page token found, running full scan")
        marked, queued = _full_scan(root_folder_id, sync_url, queue_name, db)

    queued += _enqueue_imported(sync_url, queue_name, db)

    logger.info(
        "Checker done — marked=%d queued=%d recovered_syncing=%d recovered_errors=%d recovered_downloading=%d",
        marked, queued, len(recovered), len(recovered_errors), len(recovered_downloading),
    )
    return {
        "marked": marked,
        "queued": queued,
        "recovered_syncing": len(recovered),
        "recovered_errors": len(recovered_errors),
        "recovered_downloading": len(recovered_downloading),
    }, 200


def _full_scan(root_folder_id: str, sync_url: str, queue_name: str, db: firestore.Client) -> tuple[int, int]:
    """Full Drive scan: mark changed docs as imported and enqueue. Saves new page token."""
    start_token = get_start_page_token()

    files = scan_root_folder()
    marked = 0
    queued = 0

    for file in files:
        doc_id = file["doc_id"]
        ref = db.collection(COLLECTION_NAME).document(doc_id)
        snapshot = ref.get()
        existing = snapshot.to_dict() if snapshot.exists else None

        if existing and existing.get("modifiedTime") == file["modifiedTime"]:
            continue

        ref.set({
            **(existing or {}),
            "doc_id": doc_id,
            "client_name": file["client_name"],
            "modifiedTime": file["modifiedTime"],
            "root_folder_id": root_folder_id,
            "status": "imported",
            "status_changed_at": firestore.SERVER_TIMESTAMP,
        })
        marked += 1

    set_drive_sync_token(start_token)
    return marked, queued


def _process_changes(
    changes: list[dict],
    root_folder_id: str,
    sync_url: str,
    queue_name: str,
    db: firestore.Client,
) -> tuple[int, int]:
    """Process incremental Drive changes. Returns (marked, queued)."""
    marked = 0
    queued = 0

    for change in changes:
        doc_id = change.get("fileId")
        if not doc_id:
            continue

        if change.get("removed"):
            _handle_deletion(doc_id, db)
            continue

        file = change.get("file", {})
        if file.get("trashed"):
            _handle_deletion(doc_id, db)
            continue

        if file.get("mimeType") != GOOGLE_DOC_MIME:
            continue

        parents = file.get("parents", [])
        if not parents:
            continue

        client_name = get_client_name_by_folder_id(parents[0])
        if not client_name:
            continue

        ref = db.collection(COLLECTION_NAME).document(doc_id)
        snapshot = ref.get()
        existing = snapshot.to_dict() if snapshot.exists else None

        ref.set({
            **(existing or {}),
            "doc_id": doc_id,
            "client_name": client_name,
            "root_folder_id": root_folder_id,
            "status": "imported",
            "status_changed_at": firestore.SERVER_TIMESTAMP,
        })
        marked += 1

    return marked, queued


def _handle_deletion(doc_id: str, db: firestore.Client) -> None:
    """Remove deleted file from Firestore and Qdrant."""
    from core.config import get_root_folder_id as _get_root
    from core.qdrant.writer import delete_by_doc_id
    try:
        delete_by_doc_id(doc_id, _get_root())
        db.collection(COLLECTION_NAME).document(doc_id).delete()
        logger.info("Deleted removed file: %s", doc_id)
    except Exception as e:
        logger.warning("Failed to delete %s: %s", doc_id, e)


def _enqueue_imported(sync_url: str, queue_name: str, db: firestore.Client) -> int:
    """Enqueue any docs stuck in imported status."""
    queued = 0
    imported_docs = (
        db.collection(COLLECTION_NAME)
        .where(filter=firestore.FieldFilter("status", "==", "imported"))
        .stream()
    )
    for doc in imported_docs:
        if enqueue_task(queue_name=queue_name, url=f"{sync_url}/sync/doc/{doc.id}"):
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
                task_id=f"tldv-import-{meeting_id}",
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
