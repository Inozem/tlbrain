import json
from datetime import datetime, timedelta, timezone

from google.cloud import tasks_v2

from core.integrations.cloud_tasks.client import enqueue_task


def test_enqueue_task_uses_adc_project_and_omits_optional_json(sdk_fakes, monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.setenv("REGION", "test-region")

    assert enqueue_task("queue", "https://worker.invalid/task", "named") is True

    call = next(entry for entry in sdk_fakes.trace if entry["operation"] == "create_task")
    project = "pytest-synthetic-project"
    parent = f"projects/{project}/locations/test-region/queues/queue"
    task = call["arguments"]["task"]
    assert call["arguments"]["parent"] == parent
    assert task["name"] == f"{parent}/tasks/named"
    assert task["http_request"] == {
        "http_method": tasks_v2.HttpMethod.POST,
        "url": "https://worker.invalid/task",
    }


def test_enqueue_task_preserves_positional_named_body_and_duplicate_behavior(sdk_fakes, monkeypatch):
    monkeypatch.setenv("REGION", "test-region")
    body = {"legacy": "payload"}

    assert enqueue_task("queue", "https://worker.invalid/task", "named", body) is True
    assert enqueue_task(
        queue_name="queue",
        url="https://worker.invalid/task",
        task_id="named",
        body=body,
    ) is False
    assert enqueue_task("queue", "https://worker.invalid/anonymous") is True

    calls = [entry for entry in sdk_fakes.trace if entry["operation"] == "create_task"]
    assert len(calls) == 3
    first_task = calls[0]["arguments"]["task"]
    assert first_task["http_request"]["headers"] == {"Content-Type": "application/json"}
    assert first_task["http_request"]["body"] == b'{"legacy": "payload"}'
    assert calls[1]["arguments"]["task"]["name"] == first_task["name"]
    anonymous_task = calls[2]["arguments"]["task"]
    assert "name" not in anonymous_task
    assert "headers" not in anonymous_task["http_request"]
    assert "body" not in anonymous_task["http_request"]


def test_enqueue_task_supports_schedule_deadline_and_oidc_fields(
    sdk_fakes, monkeypatch, caplog
):
    monkeypatch.setenv("REGION", "test-region")
    schedule_time = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    dispatch_deadline = timedelta(seconds=30)
    body = {
        "schema_version": 1,
        "context_id": "context-123",
        "operation_id": "operation-456",
        "outbox_generation": 7,
    }
    secret_markers = ("raw-dialog-marker", "raw-message-marker", "oauth-token-marker")

    assert enqueue_task(
        "queue",
        "https://worker.invalid/task",
        "stable-task-id",
        body,
        schedule_time=schedule_time,
        service_account_email="worker@example.invalid",
        audience="https://worker.invalid",
        dispatch_deadline=dispatch_deadline,
    ) is True

    call = next(entry for entry in sdk_fakes.trace if entry["operation"] == "create_task")
    task = call["arguments"]["task"]
    request = task["http_request"]
    assert task["schedule_time"] == schedule_time
    assert task["dispatch_deadline"] == dispatch_deadline
    assert request["oidc_token"] == {
        "service_account_email": "worker@example.invalid",
        "audience": "https://worker.invalid",
    }
    serialized_body = request["body"]
    assert serialized_body == json.dumps(body).encode()
    assert set(json.loads(serialized_body)) == {
        "schema_version",
        "context_id",
        "operation_id",
        "outbox_generation",
    }
    log_output = "\n".join(record.getMessage() for record in caplog.records)
    assert all(marker not in serialized_body.decode() for marker in secret_markers)
    assert all(marker not in log_output for marker in secret_markers)
