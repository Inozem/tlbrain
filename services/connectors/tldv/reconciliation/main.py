import logging
import os
from datetime import datetime, timezone

import functions_framework
from google.cloud import firestore

from core.google_drive.firestore import COLLECTION_NAME, write_queued
from core.utils.logging import configure_logging
from core.utils.tasks import enqueue_task
from tldv_client import get_meetings

configure_logging()
logger = logging.getLogger(__name__)


@functions_framework.http
def tldv_reconciliation(request):
    import_service_url = os.environ.get("TLDV_IMPORT_SERVICE_URL", "")
    queue_name = os.environ["TLDV_IMPORT_QUEUE"]

    body = request.get_json(silent=True) or {}
    since_str = body.get("since")
    if since_str:
        since = datetime.fromisoformat(since_str)
        logger.info("Fetching TL;DV meetings since %s", since.isoformat())
    else:
        since = None
        logger.info("Fetching all TL;DV meetings (no date filter)")

    meetings = get_meetings(since)
    logger.info("Found %d meetings in TL;DV", len(meetings))

    db = firestore.Client()
    existing = {
        doc.to_dict().get("tldv_meeting_id")
        for doc in db.collection(COLLECTION_NAME).stream()
        if doc.to_dict().get("tldv_meeting_id")
    }

    missing = [m for m in meetings if m["id"] not in existing]
    logger.info("Missing in Firestore: %d", len(missing))

    if not import_service_url:
        logger.warning("TLDV_IMPORT_SERVICE_URL not set, skipping task creation")
        return {"meetings": len(meetings), "missing": len(missing), "queued": 0}, 200

    queued = 0
    for meeting in missing:
        meeting_id = meeting["id"]
        if enqueue_task(
            queue_name=queue_name,
            task_id=f"tldv-import-{meeting_id}",
            url=f"{import_service_url}/import",
            body={"meeting_id": meeting_id},
        ):
            write_queued(meeting_id)
            queued += 1
            logger.info("Task created for meeting_id=%s", meeting_id)
        else:
            logger.info("Task already exists for meeting_id=%s, skipping", meeting_id)

    logger.info("Reconciliation done — meetings=%d missing=%d queued=%d", len(meetings), len(missing), queued)
    return {"meetings": len(meetings), "missing": len(missing), "queued": queued}, 200
