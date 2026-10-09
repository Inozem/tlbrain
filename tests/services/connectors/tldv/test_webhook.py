from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import google.api_core.exceptions
import pytest
from google.cloud import tasks_v2


@pytest.fixture
def webhook(sdk_fakes):
    # conftest installs SDK boundary fakes while keeping the real handler modules.
    del sdk_fakes
    return importlib.import_module("services.connectors.tldv.webhook.main")


def _request(payload):
    return SimpleNamespace(get_json=lambda silent=True: payload)


def _task_events(sdk_fakes):
    return [
        event for event in sdk_fakes.trace
        if event["boundary"] == "tasks" and event["operation"] == "create_task"
    ]


def test_missing_meeting_id_returns_contract_error_without_side_effects(webhook, sdk_fakes):
    response = webhook.tldv_webhook(_request({"data": {}}))

    assert response == ({"error": "missing meetingId"}, 400)
    assert _task_events(sdk_fakes) == []


def test_missing_import_service_url_returns_ok_without_task_or_placeholder(
    webhook, sdk_fakes, firestore_fake, monkeypatch
):
    monkeypatch.delenv("TLDV_IMPORT_SERVICE_URL", raising=False)

    response = webhook.tldv_webhook(_request({"data": {"meetingId": "meeting-offline"}}))

    assert response == ({"ok": True}, 200)
    assert _task_events(sdk_fakes) == []
    assert "transcript_index/tldv-meeting-offline" not in firestore_fake.documents


def test_enqueue_creates_named_post_task_before_queued_placeholder(
    webhook, sdk_fakes, firestore_fake, monkeypatch
):
    meeting_id = "meeting-contract"
    monkeypatch.setenv("TLDV_IMPORT_SERVICE_URL", "https://import.synthetic.test")
    monkeypatch.setenv("TLDV_IMPORT_QUEUE", "synthetic-import-queue")
    effects = []
    original_enqueue_task = webhook.enqueue_task
    original_write_queued = webhook.write_queued

    def record_enqueue(**kwargs):
        effects.append("enqueue")
        return original_enqueue_task(**kwargs)

    def record_placeholder(meeting):
        effects.append("placeholder")
        return original_write_queued(meeting)

    monkeypatch.setattr(webhook, "enqueue_task", record_enqueue)
    monkeypatch.setattr(webhook, "write_queued", record_placeholder)

    response = webhook.tldv_webhook(_request({"data": {"meetingId": meeting_id}}))

    assert response == ({"ok": True}, 200)
    tasks = _task_events(sdk_fakes)
    assert len(tasks) == 1
    task = tasks[0]["arguments"]["task"]
    parent = tasks[0]["arguments"]["parent"]
    assert parent == (
        "projects/pytest-synthetic-project/locations/pytest-synthetic-region/"
        "queues/synthetic-import-queue"
    )
    assert task["name"] == (
        "projects/pytest-synthetic-project/locations/pytest-synthetic-region/"
        "queues/synthetic-import-queue/tasks/tldv-import-meeting-contract"
    )
    assert task["http_request"] == {
        "http_method": tasks_v2.HttpMethod.POST,
        "url": "https://import.synthetic.test/import",
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"meeting_id": meeting_id}).encode(),
    }
    assert effects == ["enqueue", "placeholder"]
    assert firestore_fake.documents["transcript_index/tldv-meeting-contract"]["status"] == "queued"


def test_already_existing_named_task_does_not_write_placeholder(
    webhook, sdk_fakes, firestore_fake, monkeypatch
):
    meeting_id = "meeting-duplicate"
    monkeypatch.setenv("TLDV_IMPORT_SERVICE_URL", "https://import.synthetic.test")
    from google.cloud import firestore

    firestore.Client().collection("transcript_index").document(
        f"tldv-{meeting_id}"
    ).set({"meeting_id": meeting_id, "status": "error", "error_stage": "import", "error": "prior failure"})
    sdk_fakes.queue(
        "tasks", "create_task", error=google.api_core.exceptions.AlreadyExists("duplicate")
    )
    writes = []
    monkeypatch.setattr(webhook, "write_queued", lambda value: writes.append(value))

    response = webhook.tldv_webhook(_request({"data": {"meetingId": meeting_id}}))

    assert response == ({"ok": True}, 200)
    assert len(_task_events(sdk_fakes)) == 1
    assert writes == []
    placeholder = firestore_fake.documents["transcript_index/tldv-meeting-duplicate"]
    assert placeholder == {
        "meeting_id": meeting_id,
        "status": "error",
        "error_stage": "import",
        "error": "prior failure",
    }
