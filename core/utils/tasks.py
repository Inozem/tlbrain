import json
import os

import google.api_core.exceptions
import google.auth
from google.cloud import tasks_v2


def enqueue_task(queue_name: str, task_id: str, url: str, body: dict | None = None) -> bool:
    """Create a named Cloud Tasks HTTP task. Returns False if already exists."""
    _, project_id = google.auth.default()
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", project_id)
    region = os.environ["REGION"]

    client = tasks_v2.CloudTasksClient()
    queue_path = client.queue_path(project_id, region, queue_name)
    task_path = client.task_path(project_id, region, queue_name, task_id)

    http_request: dict = {
        "http_method": tasks_v2.HttpMethod.POST,
        "url": url,
    }
    if body is not None:
        http_request["headers"] = {"Content-Type": "application/json"}
        http_request["body"] = json.dumps(body).encode()

    try:
        client.create_task(parent=queue_path, task={"name": task_path, "http_request": http_request})
        return True
    except google.api_core.exceptions.AlreadyExists:
        return False
