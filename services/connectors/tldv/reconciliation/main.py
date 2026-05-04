import json
import os
from datetime import datetime, timedelta, timezone

import functions_framework
import google.api_core.exceptions
import google.auth
import requests
from google.cloud import firestore, tasks_v2

_COLLECTION = "transcript_index"
_TLDV_API_BASE = "https://pasta.tldv.io/v1alpha1"


def _get_meetings(api_key: str, since: datetime) -> list[dict]:
    meetings = []
    page = 1

    while True:
        resp = requests.get(
            f"{_TLDV_API_BASE}/meetings",
            headers={"x-api-key": api_key},
            params={"page": page},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        total_pages = data.get("pages", 1)

        for m in results:
            happened_at = m.get("happenedAt", "")
            try:
                meeting_time = datetime.strptime(happened_at, "%a %b %d %Y %H:%M:%S GMT+0000 (Coordinated Universal Time)").replace(tzinfo=timezone.utc)
                if meeting_time >= since:
                    meetings.append(m)
            except ValueError:
                pass

        if page >= total_pages:
            break

        if results and meetings and meetings[-1].get("happenedAt"):
            last_time_str = results[-1].get("happenedAt", "")
            try:
                last_time = datetime.strptime(last_time_str, "%a %b %d %Y %H:%M:%S GMT+0000 (Coordinated Universal Time)").replace(tzinfo=timezone.utc)
                if last_time < since:
                    break
            except ValueError:
                pass

        page += 1

    return meetings


@functions_framework.http
def tldv_reconciliation(request):
    api_key = os.environ["TLDV_API_KEY"]
    import_service_url = os.environ.get("TLDV_IMPORT_SERVICE_URL", "")
    queue_name = os.environ["TLDV_IMPORT_QUEUE"]
    region = os.environ["REGION"]

    _, project_id = google.auth.default()
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", project_id)

    since = datetime.now(timezone.utc) - timedelta(hours=48)
    print(f"Fetching TL;DV meetings since {since.isoformat()}", flush=True)

    meetings = _get_meetings(api_key, since)
    print(f"Found {len(meetings)} meetings in TL;DV", flush=True)

    db = firestore.Client()
    existing = {
        doc.to_dict().get("tldv_meeting_id")
        for doc in db.collection(_COLLECTION).stream()
        if doc.to_dict().get("tldv_meeting_id")
    }

    missing = [m for m in meetings if m["id"] not in existing]
    print(f"Missing in Firestore: {len(missing)}", flush=True)

    if not import_service_url:
        print("TLDV_IMPORT_SERVICE_URL not set, skipping task creation", flush=True)
        return {"meetings": len(meetings), "missing": len(missing), "queued": 0}, 200

    tasks_client = tasks_v2.CloudTasksClient()
    queue_path = tasks_client.queue_path(project_id, region, queue_name)
    queued = 0

    for meeting in missing:
        meeting_id = meeting["id"]
        task_name = tasks_client.task_path(project_id, region, queue_name, f"tldv-import-{meeting_id}")
        task = {
            "name": task_name,
            "http_request": {
                "http_method": tasks_v2.HttpMethod.POST,
                "url": f"{import_service_url}/import",
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"meeting_id": meeting_id}).encode(),
            },
        }
        try:
            tasks_client.create_task(parent=queue_path, task=task)
            queued += 1
            print(f"Task created for meeting_id={meeting_id}", flush=True)
        except google.api_core.exceptions.AlreadyExists:
            print(f"Task already exists for meeting_id={meeting_id}, skipping", flush=True)

    print(f"Reconciliation done — meetings={len(meetings)} missing={len(missing)} queued={queued}", flush=True)
    return {"meetings": len(meetings), "missing": len(missing), "queued": queued}, 200
