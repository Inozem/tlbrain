import logging
import os
from datetime import datetime, timedelta, timezone

import functions_framework
from google.cloud import firestore

from core.config import get_root_folder_id
from core.google_drive.drive_client import scan_root_folder, list_client_folders
from core.google_drive.firestore import COLLECTION_NAME, recover_errors, recover_stale_syncing, sync_clients_from_drive
from core.utils.tasks import enqueue_task

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STALE_MINUTES = 15


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

        if enqueue_task(queue_name=queue_name, task_id=doc_id, url=f"{sync_url}/sync/doc/{doc_id}"):
            queued += 1

    imported_docs = (
        db.collection(COLLECTION_NAME)
        .where(filter=firestore.FieldFilter("status", "==", "imported"))
        .stream()
    )
    for doc in imported_docs:
        if enqueue_task(queue_name=queue_name, task_id=doc.id, url=f"{sync_url}/sync/doc/{doc.id}"):
            queued += 1

    logger.info(
        "Checker done — files=%d marked=%d queued=%d recovered_syncing=%d recovered_errors=%d recovered_downloading=%d",
        len(files), marked, queued, len(recovered), len(recovered_errors), len(recovered_downloading),
    )
    return {
        "files": len(files),
        "marked": marked,
        "queued": queued,
        "recovered_syncing": len(recovered),
        "recovered_errors": len(recovered_errors),
        "recovered_downloading": len(recovered_downloading),
    }, 200


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
