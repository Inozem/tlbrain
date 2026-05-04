import json
import os

import functions_framework
import google.api_core.exceptions
import google.auth
from google.cloud import tasks_v2


@functions_framework.http
def tldv_webhook(request):
    payload = request.get_json(silent=True) or {}
    print("TL;DV webhook payload:", json.dumps(payload), flush=True)

    meeting_id = payload.get("data", {}).get("meetingId")
    if not meeting_id:
        print("No meetingId in payload:", payload, flush=True)
        return {"error": "missing meetingId"}, 400

    _, project_id = google.auth.default()
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", project_id)
    region = os.environ["REGION"]
    queue_name = os.environ["TLDV_IMPORT_QUEUE"]
    import_service_url = os.environ.get("TLDV_IMPORT_SERVICE_URL", "")

    if not import_service_url:
        print(f"TLDV_IMPORT_SERVICE_URL not set, skipping task for meeting_id={meeting_id}", flush=True)
        return {"ok": True}, 200

    tasks_client = tasks_v2.CloudTasksClient()
    queue_path = tasks_client.queue_path(project_id, region, queue_name)
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
        print(f"Task created for meeting_id={meeting_id}", flush=True)
    except google.api_core.exceptions.AlreadyExists:
        print(f"Task already exists for meeting_id={meeting_id}, skipping", flush=True)

    return {"ok": True}, 200
