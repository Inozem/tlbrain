import logging
import os
from datetime import datetime, timedelta, timezone

import functions_framework
import requests
from google.cloud import firestore

from core.google_drive.firestore import COLLECTION_NAME, mark_download_error, write_queued
from core.utils.logging import configure_logging
from core.utils.tasks import enqueue_task
from tldv_client import get_meetings, iter_meeting_pages, tldv_get

configure_logging()
logger = logging.getLogger(__name__)


def _recover_stale_import_docs(
    db: firestore.Client,
    import_service_url: str,
    queue_name: str,
) -> tuple[int, int]:
    """Recover stale import docs by checking TL;DV API.

    Scans Firestore for docs with meeting_id stuck in queued/downloading/error+import
    between 1h and 48h old. For each:
    - transcript ready → reset to queued + enqueue_task (retry)
    - transcript still processing → skip
    - 403/404 from TL;DV → mark_download_error (finalize, stop retrying)

    Returns (recovered, finalized).
    """
    now = datetime.now(timezone.utc)
    stale_upper = now - timedelta(hours=1)   # newer → might still be in active import
    stale_lower = now - timedelta(hours=48)  # older → transcript likely gone from TL;DV

    recovered = 0
    finalized = 0

    for doc in db.collection(COLLECTION_NAME).stream():
        data = doc.to_dict() or {}

        meeting_id = data.get("meeting_id")
        if not meeting_id:
            continue

        status = data.get("status")
        error_stage = data.get("error_stage")
        eligible = (
            status in ("queued", "downloading")
            or (status == "error" and error_stage in (None, "import"))
        )
        if not eligible:
            continue

        changed_at = data.get("status_changed_at")
        if not changed_at or changed_at > stale_upper or changed_at < stale_lower:
            continue

        try:
            transcript_resp = tldv_get(f"/meetings/{meeting_id}/transcript")
            utterances = transcript_resp.get("results", transcript_resp.get("data", []))
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code in (403, 404):
                mark_download_error(meeting_id, f"HTTP {exc.response.status_code} in reconciliation recovery")
                finalized += 1
                logger.info(
                    "Finalized stale import doc: meeting_id=%s HTTP %s",
                    meeting_id, exc.response.status_code,
                )
            else:
                logger.warning("TL;DV API error for meeting_id=%s: %s", meeting_id, exc)
            continue
        except Exception as exc:
            logger.warning("Failed to check transcript for meeting_id=%s: %s", meeting_id, exc)
            continue

        if not utterances:
            logger.info("Stale import doc still processing: meeting_id=%s status=%s", meeting_id, status)
            continue

        # Transcript is ready — reset to queued and re-enqueue
        if status != "queued":
            db.collection(COLLECTION_NAME).document(doc.id).update({
                "status": "queued",
                "error": None,
                "error_stage": None,
                "status_changed_at": firestore.SERVER_TIMESTAMP,
            })

        if import_service_url:
            enqueued = enqueue_task(
                queue_name=queue_name,
                task_id=f"tldv-import-{meeting_id}",
                url=f"{import_service_url}/import",
                body={"meeting_id": meeting_id},
            )
            if enqueued:
                logger.info("Enqueued recovery task: meeting_id=%s (was status=%s)", meeting_id, status)
            else:
                logger.info("Recovery task already exists: meeting_id=%s (was status=%s)", meeting_id, status)
        else:
            logger.warning("TLDV_IMPORT_SERVICE_URL not set, skipping recovery task for meeting_id=%s", meeting_id)

        recovered += 1
        logger.info("Recovered stale import doc: meeting_id=%s (was status=%s)", meeting_id, status)

    return recovered, finalized


@functions_framework.http
def tldv_reconciliation(request):
    import_service_url = os.environ.get("TLDV_IMPORT_SERVICE_URL", "")
    queue_name = os.environ.get("TLDV_IMPORT_QUEUE", "tlbrain-tldv-import-queue")

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

    recovered, finalized = _recover_stale_import_docs(db, import_service_url, queue_name)
    if recovered:
        logger.info("Recovered stale import docs: %d", recovered)
    if finalized:
        logger.info("Finalized stale import docs (transcript gone): %d", finalized)

    stale_cutoff = datetime.now(timezone.utc) - timedelta(minutes=15)
    existing: set[str] = set()
    for doc in db.collection(COLLECTION_NAME).stream():
        data = doc.to_dict() or {}
        mid = data.get("meeting_id")
        if not mid:
            continue
        if data.get("status") == "error" and data.get("error_stage") == "import":
            continue  # allow reconciliation to retry import errors
        if data.get("status") == "queued":
            changed_at = data.get("status_changed_at")
            if not changed_at or changed_at < stale_cutoff:
                continue  # stale queued — allow reconciliation to re-enqueue
        existing.add(mid)

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
        write_queued(meeting_id)
        if enqueue_task(
            queue_name=queue_name,
            task_id=f"tldv-import-{meeting_id}",
            url=f"{import_service_url}/import",
            body={"meeting_id": meeting_id},
        ):
            queued += 1
            logger.info("Task created for meeting_id=%s", meeting_id)
        else:
            logger.info("Task already exists for meeting_id=%s, skipping", meeting_id)

    logger.info(
        "Reconciliation done — meetings=%d missing=%d queued=%d remaining=%s recovered=%d finalized=%d",
        meetings_count, len(missing), queued, remaining, recovered, finalized,
    )
    return {
        "meetings": meetings_count,
        "missing": len(missing),
        "queued": queued,
        "remaining": remaining,
        "recovered": recovered,
        "finalized": finalized,
    }, 200
