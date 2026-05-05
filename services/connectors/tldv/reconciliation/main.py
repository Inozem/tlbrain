import os
from datetime import datetime, timedelta, timezone

import functions_framework
from google.cloud import firestore

from core.google_drive.firestore import COLLECTION_NAME
from core.utils.tasks import enqueue_task
from tldv_client import get_meetings


@functions_framework.http
def tldv_reconciliation(request):
    import_service_url = os.environ.get("TLDV_IMPORT_SERVICE_URL", "")
    queue_name = os.environ["TLDV_IMPORT_QUEUE"]

    since = datetime.now(timezone.utc) - timedelta(hours=48)
    print(f"Fetching TL;DV meetings since {since.isoformat()}", flush=True)

    meetings = get_meetings(since)
    print(f"Found {len(meetings)} meetings in TL;DV", flush=True)

    db = firestore.Client()
    existing = {
        doc.to_dict().get("tldv_meeting_id")
        for doc in db.collection(COLLECTION_NAME).stream()
        if doc.to_dict().get("tldv_meeting_id")
    }

    missing = [m for m in meetings if m["id"] not in existing]
    print(f"Missing in Firestore: {len(missing)}", flush=True)

    if not import_service_url:
        print("TLDV_IMPORT_SERVICE_URL not set, skipping task creation", flush=True)
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
            queued += 1
            print(f"Task created for meeting_id={meeting_id}", flush=True)
        else:
            print(f"Task already exists for meeting_id={meeting_id}, skipping", flush=True)

    print(f"Reconciliation done — meetings={len(meetings)} missing={len(missing)} queued={queued}", flush=True)
    return {"meetings": len(meetings), "missing": len(missing), "queued": queued}, 200
