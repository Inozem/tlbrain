import logging
import os

import functions_framework
from google.cloud import firestore

from core.config import get_root_folder_id
from core.google_drive.drive_client import scan_root_folder
from core.google_drive.firestore import COLLECTION_NAME
from core.utils.tasks import enqueue_task

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@functions_framework.http
def checker(request):
    root_folder_id = get_root_folder_id()
    sync_url = os.environ["VECTOR_SYNC_URL"]
    queue_name = os.environ["VECTOR_SYNC_QUEUE"]

    db = firestore.Client()

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
        })
        marked += 1

        if enqueue_task(queue_name=queue_name, task_id=doc_id, url=f"{sync_url}/sync/doc/{doc_id}"):
            queued += 1

    logger.info("Checker done — files=%d marked=%d queued=%d", len(files), marked, queued)
    return {"files": len(files), "marked": marked, "queued": queued}, 200
