import json
import os
from datetime import datetime, timedelta

import google.api_core.exceptions
import google.auth
from google.cloud import tasks_v2


def enqueue_task(
    queue_name: str,
    url: str,
    task_id: str | None = None,
    body: dict | None = None,
    *,
    schedule_time: datetime | None = None,
    service_account_email: str | None = None,
    audience: str | None = None,
    dispatch_deadline: timedelta | None = None,
) -> bool:
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
    oidc_token = {}
    if service_account_email is not None:
        oidc_token["service_account_email"] = service_account_email
    if audience is not None:
        oidc_token["audience"] = audience
    if oidc_token:
        http_request["oidc_token"] = oidc_token

    task: dict = {"http_request": http_request}
    if task_id is not None:
        task["name"] = client.task_path(project_id, region, queue_name, task_id)
    if schedule_time is not None:
        task["schedule_time"] = schedule_time
    if dispatch_deadline is not None:
        task["dispatch_deadline"] = dispatch_deadline

    try:
        client.create_task(parent=queue_path, task=task)
        return True
    except google.api_core.exceptions.AlreadyExists:
        return False
