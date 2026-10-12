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
