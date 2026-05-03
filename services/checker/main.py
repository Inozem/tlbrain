import logging
import os

import functions_framework
import google.api_core.exceptions
import google.auth
from google.cloud import firestore, tasks_v2

from core.config import get_root_folder_id
from core.google_drive.drive_client import scan_root_folder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_COLLECTION = "transcript_index"


@functions_framework.http
def checker(request):
    root_folder_id = get_root_folder_id()
    sync_url = os.environ["SYNC_URL"]
    _, project_id = google.auth.default()
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", project_id)
    region = os.environ["REGION"]
    queue_name = os.environ["CLOUD_TASKS_QUEUE"]

    db = firestore.Client()
    tasks_client = tasks_v2.CloudTasksClient()
    queue_path = tasks_client.queue_path(project_id, region, queue_name)

    files = scan_root_folder()
    marked = 0
    queued = 0

    for file in files:
        doc_id = file["doc_id"]
        ref = db.collection(_COLLECTION).document(doc_id)
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

        task = {
            "name": tasks_client.task_path(project_id, region, queue_name, doc_id),
            "http_request": {
                "http_method": tasks_v2.HttpMethod.POST,
                "url": f"{sync_url}/sync/doc/{doc_id}",
            },
        }
        try:
            tasks_client.create_task(parent=queue_path, task=task)
            queued += 1
        except google.api_core.exceptions.AlreadyExists:
            pass

    logger.info("Checker done — files=%d marked=%d queued=%d", len(files), marked, queued)
    return {"files": len(files), "marked": marked, "queued": queued}, 200
