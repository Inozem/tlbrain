import logging
import os
from datetime import datetime, timedelta, timezone

import functions_framework
from google.cloud import firestore

from core.google_drive.firestore import COLLECTION_NAME, write_queued
from core.utils.logging import configure_logging
from core.utils.tasks import enqueue_task
from tldv_client import get_meetings, iter_meeting_pages

configure_logging()
logger = logging.getLogger(__name__)


@functions_framework.http
def tldv_reconciliation(request):
    import_service_url = os.environ.get("TLDV_IMPORT_SERVICE_URL", "")
    queue_name = os.environ["TLDV_IMPORT_QUEUE"]

    body = request.get_json(silent=True) or {}
    since_str = body.get("since")
    full_scan = body.get("full_scan", False)
    if since_str:
        since = datetime.fromisoformat(since_str)
        logger.info("Fetching TL;DV meetings since %s", since.isoformat())
    elif full_scan:
        since = None
        logger.info("Fetching all TL;DV meetings (full scan)")
    else:
        since = datetime.now(timezone.utc) - timedelta(hours=48)
        logger.info("Fetching TL;DV meetings since 48h ago (%s)", since.isoformat())

    limit = body.get("limit")

    db = firestore.Client()
    existing = {
        doc.to_dict().get("tldv_meeting_id")
        for doc in db.collection(COLLECTION_NAME).stream()
        if doc.to_dict().get("tldv_meeting_id")
    }

    missing = []
    meetings_count = 0
    stopped_early = False

    if limit:
        for page in iter_meeting_pages(since):
            meetings_count += len(page)
            for m in page:
                if m["id"] not in existing:
                    missing.append(m)
            if len(missing) >= limit:
                stopped_early = True
                break
    else:
        meetings = get_meetings(since)
        meetings_count = len(meetings)
        missing = [m for m in meetings if m["id"] not in existing]

    batch = missing[:limit] if limit else missing
    remaining = None if stopped_early else len(missing) - len(batch)
    logger.info("Found %d meetings (scanned), missing=%d, stopped_early=%s", meetings_count, len(missing), stopped_early)

    if not import_service_url:
        logger.warning("TLDV_IMPORT_SERVICE_URL not set, skipping task creation")
        return {"meetings": meetings_count, "missing": len(missing), "queued": 0, "remaining": remaining}, 200

    queued = 0
    for meeting in batch:
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

    logger.info("Reconciliation done — meetings=%d missing=%d queued=%d remaining=%s", meetings_count, len(missing), queued, remaining)
    return {"meetings": meetings_count, "missing": len(missing), "queued": queued, "remaining": remaining}, 200
