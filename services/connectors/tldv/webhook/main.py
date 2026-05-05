import json
import os

import functions_framework

from core.utils.tasks import enqueue_task


@functions_framework.http
def tldv_webhook(request):
    payload = request.get_json(silent=True) or {}
    print("TL;DV webhook payload:", json.dumps(payload), flush=True)

    meeting_id = payload.get("data", {}).get("meetingId")
    if not meeting_id:
        print("No meetingId in payload:", payload, flush=True)
        return {"error": "missing meetingId"}, 400

    import_service_url = os.environ.get("TLDV_IMPORT_SERVICE_URL", "")
    queue_name = os.environ["TLDV_IMPORT_QUEUE"]

    if not import_service_url:
        print(f"TLDV_IMPORT_SERVICE_URL not set, skipping task for meeting_id={meeting_id}", flush=True)
        return {"ok": True}, 200

    queued = enqueue_task(
        queue_name=queue_name,
        task_id=f"tldv-import-{meeting_id}",
        url=f"{import_service_url}/import",
        body={"meeting_id": meeting_id},
    )
    if queued:
        print(f"Task created for meeting_id={meeting_id}", flush=True)
    else:
        print(f"Task already exists for meeting_id={meeting_id}, skipping", flush=True)

    return {"ok": True}, 200
