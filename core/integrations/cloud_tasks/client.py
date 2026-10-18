import json
import os

import google.api_core.exceptions
import google.auth
from google.cloud import tasks_v2


def enqueue_task(queue_name: str, url: str, task_id: str | None = None, body: dict | None = None) -> bool:
    """Create a Cloud Tasks HTTP task.

    task_id: named task for idempotency (returns False if already exists).
             Omit for anonymous tasks that always create a new entry.
    """
    _, project_id = google.auth.default()
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", project_id)
    region = os.environ["REGION"]

    client = tasks_v2.CloudTasksClient()
    queue_path = client.queue_path(project_id, region, queue_name)

    http_request: dict = {
        "http_method": tasks_v2.HttpMethod.POST,
        "url": url,
    }
    if body is not None:
        http_request["headers"] = {"Content-Type": "application/json"}
        http_request["body"] = json.dumps(body).encode()

    task: dict = {"http_request": http_request}
    if task_id is not None:
        task["name"] = client.task_path(project_id, region, queue_name, task_id)

    try:
        client.create_task(parent=queue_path, task=task)
        return True
    except google.api_core.exceptions.AlreadyExists:
        return False
