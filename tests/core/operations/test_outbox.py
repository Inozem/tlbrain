from datetime import datetime, timedelta, timezone

import pytest

from core.integrations.firestore.outbox import (
    OutboxState,
    OutboxTargetKind,
    create_outbox_record,
    get_outbox_record,
)
from core.integrations.firestore.operations import claim_operation, create_operation
from core.operations import outbox as publisher
from core.operations.models import OperationOrigin
from core.operations.outbox import (
    TaskTargetConfig,
    enqueue_stale_mcp_save_recoveries,
    publish_due_outbox_records,
)


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


def _create_operation(client, operation_id, *, now=None, **kwargs):
    values = {
        "operation_id": operation_id,
        "origin": OperationOrigin.MCP,
        "origin_operation_ref": f"request-{operation_id}",
        "caller_binding": "caller-1",
        "immutable_input_digest": "sha256:synthetic",
        "immutable_input_ref": "input-ref-1",
        "save_intent_id": f"save-{operation_id}",
        "now": now or NOW - timedelta(minutes=10),
    }
    values.update(kwargs)
    return create_operation(client, **values)


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


def test_notification_dispatch_publishes_identifier_only_without_changing_send_budget(
    firestore_fake, monkeypatch,
):
    client = firestore_fake.client_class()
    record = create_outbox_record(
        client,
        "notification-wakeup",
        target_kind=OutboxTargetKind.TELEGRAM_DISPATCH,
        generation=1,
        schedule_at=NOW,
        now=NOW,
        context_id="telegram-context",
        notification_id="notification-ack",
    )
    notification_ref = client.collection("notification_outbox").document("notification-ack")
    notification_ref.set({
        "notification_id": "notification-ack",
        "state": "pending",
        "send_attempts_started": 0,
        "edit_attempts_started": 0,
        "unknown_send_outcome": False,
        "in_flight_attempt_id": None,
    })
    ledger_before = notification_ref.get().to_dict()
    calls = []
    monkeypatch.setattr(publisher, "enqueue_task", lambda **kwargs: calls.append(kwargs) or True)

    result = publish_due_outbox_records(
        client, {OutboxTargetKind.TELEGRAM_DISPATCH: TARGET}, now=NOW, limit=1,
    )

    assert result.published == 1
    assert calls == [{
        "queue_name": TARGET.queue_name,
        "url": TARGET.url,
        "task_id": record.task_name,
        "body": {
            "schema_version": 1,
            "outbox_id": "notification-wakeup",
            "target_kind": "telegram_dispatch",
            "generation": 1,
            "context_id": "telegram-context",
            "notification_id": "notification-ack",
        },
        "schedule_time": NOW,
        "service_account_email": TARGET.service_account_email,
        "audience": TARGET.audience,
        "dispatch_deadline": TARGET.dispatch_deadline,
    }]
    assert notification_ref.get().to_dict() == ledger_before


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
    assert skipped.selected == skipped.skipped == 0
    assert get_outbox_record(client, "one").state is OutboxState.RETRY
    assert get_outbox_record(client, "unconfigured").state is OutboxState.PENDING


def test_publisher_filters_disabled_target_records_before_bounded_selection(firestore_fake, monkeypatch):
    client = firestore_fake.client_class()
    _create(client, "disabled", kind=OutboxTargetKind.INDEX_DISPATCH)
    _create(client, "enabled")
    calls = []
    monkeypatch.setattr(publisher, "enqueue_task", lambda **kwargs: calls.append(kwargs) or True)

    result = publish_due_outbox_records(
        client,
        {
            OutboxTargetKind.INDEX_DISPATCH: None,
            OutboxTargetKind.TELEGRAM_DISPATCH: TARGET,
        },
        now=NOW,
        limit=1,
    )

    assert result.selected == result.published == 1
    assert [call["task_id"] for call in calls] == ["task-enabled-g1"]
    assert get_outbox_record(client, "disabled").state is OutboxState.PENDING


def test_stale_mcp_recovery_filters_eligibility_and_reuses_fence_identity(firestore_fake):
    client = firestore_fake.client_class()
    old = NOW - timedelta(minutes=10)
    _create_operation(client, "pending", now=old)
    _create_operation(client, "no-save", now=old, save_intent_id=None)
    _create_operation(client, "legacy", now=old, origin_operation_ref=None)
    _create_operation(client, "telegram", now=old, origin=OperationOrigin.TELEGRAM)
    _create_operation(client, "terminal", now=old)
    client.collection("dialog_operations").document("terminal").update({"state": "failed"})
    _create_operation(client, "live", now=old)
    claim_operation(client, "live", owner="live-worker", now=NOW, lease_seconds=300)
    client.collection("dialog_operations").document("live").update({"updated_at": old})
    _create_operation(client, "expired", now=old)
    claim_operation(client, "expired", owner="expired-worker", now=old, lease_seconds=1)
    client.collection("dialog_operations").document("expired").update({"updated_at": old})
    _create_operation(client, "lease-less", now=old)
    client.collection("dialog_operations").document("lease-less").update({
        "state": "running", "updated_at": old, "lease_owner": None, "lease_expires_at": None,
    })

    first = enqueue_stale_mcp_save_recoveries(client, now=NOW, limit=50)
    second = enqueue_stale_mcp_save_recoveries(client, now=NOW + timedelta(seconds=30), limit=50)
    records = [
        data for path, data in firestore_fake.documents.items()
        if path.startswith("task_outbox/")
    ]

    assert first == 3
    assert second == 3
    assert {record["operation_id"] for record in records} == {"pending", "expired", "lease-less"}
    assert len(records) == 3
    assert {record["operation_id"]: record["generation"] for record in records} == {
        "pending": 1,
        "expired": 2,
        "lease-less": 1,
    }
    assert all(record["save_intent_id"] and "text" not in record for record in records)


def test_stale_mcp_recovery_advances_generation_only_with_fence(firestore_fake):
    client = firestore_fake.client_class()
    old = NOW - timedelta(minutes=10)
    _create_operation(client, "fenced", now=old)
    assert enqueue_stale_mcp_save_recoveries(client, now=NOW, limit=10) == 1
    first = next(data for path, data in firestore_fake.documents.items() if path.startswith("task_outbox/"))

    lease = claim_operation(client, "fenced", owner="worker", now=old, lease_seconds=1)
    assert lease is not None and lease.fence == 1
    client.collection("dialog_operations").document("fenced").update({"updated_at": old})
    assert enqueue_stale_mcp_save_recoveries(client, now=NOW, limit=10) == 1
    records = [data for path, data in firestore_fake.documents.items() if path.startswith("task_outbox/")]

    assert len(records) == 2
    assert {record["generation"] for record in records} == {1, 2}
    assert first["outbox_id"] != next(record["outbox_id"] for record in records if record["generation"] == 2)


def test_target_config_rejects_untrusted_or_malformed_url():
    with pytest.raises(ValueError, match="HTTPS"):
        TaskTargetConfig("queue", "http://worker.example.invalid/task")
    with pytest.raises(ValueError, match="credentials"):
        TaskTargetConfig("queue", "https://user:pass@worker.example.invalid/task")
