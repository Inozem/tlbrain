from datetime import datetime, timedelta, timezone

import pytest

from core.integrations.firestore.outbox import (
    OutboxState,
    OutboxTargetKind,
    create_outbox_record,
    get_outbox_record,
)
from core.operations import outbox as publisher
from core.operations.outbox import TaskTargetConfig, publish_due_outbox_records


NOW = datetime(2026, 4, 1, tzinfo=timezone.utc)
TARGET = TaskTargetConfig(
    queue_name="telegram",
    url="https://worker.example.invalid/dispatch",
    service_account_email="worker@example.invalid",
    audience="https://worker.example.invalid",
    dispatch_deadline=timedelta(seconds=45),
)


def _create(client, outbox_id, *, kind=OutboxTargetKind.TELEGRAM_DISPATCH):
    return create_outbox_record(
        client,
        outbox_id,
        target_kind=kind,
        generation=1,
        schedule_at=NOW,
        now=NOW,
        operation_id=f"operation-{outbox_id}",
    )


def test_publisher_uses_trusted_target_and_ids_only_body(firestore_fake, monkeypatch):
    client = firestore_fake.client_class()
    record = _create(client, "outbox-1")
    operation_ref = client.collection("dialog_operations").document("operation-outbox-1")
    operation_ref.set({"state": "running", "status": "in_progress"})
    operation_before = operation_ref.get().to_dict()
    calls = []

    def enqueue(**kwargs):
        calls.append(kwargs)
        return True

    monkeypatch.setattr(publisher, "enqueue_task", enqueue)
    result = publish_due_outbox_records(
        client, {OutboxTargetKind.TELEGRAM_DISPATCH: TARGET}, now=NOW, limit=3
    )

    assert result.selected == result.published == 1
    assert result.already_exists == result.skipped == result.failed == 0
    assert calls == [{
        "queue_name": "telegram",
        "url": "https://worker.example.invalid/dispatch",
        "task_id": record.task_name,
        "body": {
            "schema_version": 1,
            "outbox_id": "outbox-1",
            "target_kind": "telegram_dispatch",
            "generation": 1,
            "operation_id": "operation-outbox-1",
        },
        "schedule_time": NOW,
        "service_account_email": "worker@example.invalid",
        "audience": "https://worker.example.invalid",
        "dispatch_deadline": timedelta(seconds=45),
    }]
    assert get_outbox_record(client, "outbox-1").state is OutboxState.DELIVERED
    assert operation_ref.get().to_dict() == operation_before


def test_duplicate_result_recovers_crash_after_cloud_tasks_acceptance(firestore_fake, monkeypatch):
    client = firestore_fake.client_class()
    record = _create(client, "outbox-crash")
    calls = []

    def enqueue(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            # The provider accepted the task, but the response was lost.
            raise RuntimeError("response contained https://secret.invalid/token")
        return False  # Cloud Tasks reports the deterministic name already exists.

    monkeypatch.setattr(publisher, "enqueue_task", enqueue)
    first = publish_due_outbox_records(
        client, {"telegram_dispatch": TARGET}, now=NOW, limit=5
    )
    assert first.failed == 1
    pending = get_outbox_record(client, "outbox-crash")
    assert pending.state is OutboxState.RETRY
    assert pending.delivery_attempts == 1
    assert pending.last_error == "cloud_tasks_enqueue_failed"

    second = publish_due_outbox_records(
        client, {"telegram_dispatch": TARGET}, now=NOW, limit=5
    )
    assert second.already_exists == 1
    assert second.failed == 0
    delivered = get_outbox_record(client, "outbox-crash")
    assert delivered.state is OutboxState.DELIVERED
    assert delivered.delivery_attempts == 2
    assert [call["task_id"] for call in calls] == [record.task_name, record.task_name]


def test_missing_target_stays_pending_and_sweep_continues_after_enqueue_error(
    firestore_fake, monkeypatch
):
    client = firestore_fake.client_class()
    _create(client, "one")
    _create(client, "two")
    _create(client, "unconfigured", kind=OutboxTargetKind.INDEX_DISPATCH)
    calls = []

    def enqueue(**kwargs):
        calls.append(kwargs)
        if kwargs["task_id"] == "task-one-g1":
            raise RuntimeError("private URL and body should not be persisted")
        return True

    monkeypatch.setattr(publisher, "enqueue_task", enqueue)
    result = publish_due_outbox_records(
        client,
        {OutboxTargetKind.TELEGRAM_DISPATCH: TARGET},
        now=NOW,
        limit=2,
    )
    assert result.selected == 2
    assert result.failed == 1
    assert result.published == 1
    assert len(calls) == 2
    assert get_outbox_record(client, "one").state is OutboxState.RETRY
    assert get_outbox_record(client, "one").last_error == "cloud_tasks_enqueue_failed"
    assert get_outbox_record(client, "two").state is OutboxState.DELIVERED

    skipped = publish_due_outbox_records(
        client,
        {OutboxTargetKind.INDEX_DISPATCH: None},
        now=NOW,
        limit=10,
    )
    assert skipped.selected == skipped.skipped == 2
    assert get_outbox_record(client, "one").state is OutboxState.RETRY
    assert get_outbox_record(client, "unconfigured").state is OutboxState.PENDING


def test_target_config_rejects_untrusted_or_malformed_url():
    with pytest.raises(ValueError, match="HTTPS"):
        TaskTargetConfig("queue", "http://worker.example.invalid/task")
    with pytest.raises(ValueError, match="credentials"):
        TaskTargetConfig("queue", "https://user:pass@worker.example.invalid/task")
