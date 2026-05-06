import json
import logging
import os

import functions_framework

from core.google_drive.firestore import write_queued
from core.utils.logging import configure_logging
from core.utils.tasks import enqueue_task

configure_logging()
logger = logging.getLogger(__name__)


@functions_framework.http
def tldv_webhook(request):
    payload = request.get_json(silent=True) or {}
    logger.info("TL;DV webhook payload: %s", json.dumps(payload))

    meeting_id = payload.get("data", {}).get("meetingId")
    if not meeting_id:
        logger.warning("No meetingId in payload: %s", payload)
        return {"error": "missing meetingId"}, 400

    import_service_url = os.environ.get("TLDV_IMPORT_SERVICE_URL", "")
    queue_name = os.environ["TLDV_IMPORT_QUEUE"]

    if not import_service_url:
        logger.warning("TLDV_IMPORT_SERVICE_URL not set, skipping task for meeting_id=%s", meeting_id)
        return {"ok": True}, 200

    queued = enqueue_task(
        queue_name=queue_name,
        task_id=f"tldv-import-{meeting_id}",
        url=f"{import_service_url}/import",
        body={"meeting_id": meeting_id},
    )
    if queued:
        write_queued(meeting_id)
        logger.info("Task created for meeting_id=%s", meeting_id)
    else:
        logger.info("Task already exists for meeting_id=%s, skipping", meeting_id)

    return {"ok": True}, 200
