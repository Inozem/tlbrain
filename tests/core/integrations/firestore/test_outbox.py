from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from core.integrations.firestore.outbox import (
    OUTBOX_COLLECTION,
    OutboxState,
    OutboxTargetKind,
    create_outbox_record,
    create_outbox_record_in_transaction,
    get_outbox_record,
    list_due_outbox_records,
    record_outbox_delivery_result,
    task_payload,
)


NOW = datetime(2026, 4, 1, tzinfo=timezone.utc)


def _create(client, *, outbox_id="outbox-1", **kwargs):
    values = {
        "target_kind": OutboxTargetKind.TELEGRAM_DISPATCH,
        "generation": 1,
        "schedule_at": NOW,
        "operation_id": "op-1",
        "now": NOW,
    }
    values.update(kwargs)
    return create_outbox_record(client, outbox_id, **values)


def test_create_and_exact_replay_round_trip_fields_and_deterministic_name(firestore_fake):
    client = firestore_fake.client_class()
    created = _create(client)
    replay = _create(client)

    assert created == replay
    assert created.target_kind is OutboxTargetKind.TELEGRAM_DISPATCH
    assert created.state is OutboxState.PENDING
    assert created.task_name == "task-outbox-1-g1"
    assert created.delivery_attempts == 0
    assert created.last_error is None
    assert created.created_at == created.updated_at == NOW
    assert get_outbox_record(client, "outbox-1") == created
    assert firestore_fake.documents[f"{OUTBOX_COLLECTION}/outbox-1"]["schedule_at"] == NOW


def test_delivery_state_and_attempt_fields_round_trip(firestore_fake):
    client = firestore_fake.client_class()
    _create(client)
    client.collection(OUTBOX_COLLECTION).document("outbox-1").update({
        "state": OutboxState.RETRY.value,
        "delivery_attempts": 3,
        "last_error": "temporary failure",
        "last_attempt_at": NOW,
    })

    record = get_outbox_record(client, "outbox-1")
    assert record.state is OutboxState.RETRY
    assert record.delivery_attempts == 3
    assert record.last_error == "temporary failure"
    assert record.last_attempt_at == NOW


def test_replay_conflicts_on_target_identity_generation_or_schedule(firestore_fake):
    client = firestore_fake.client_class()
    _create(client)
    with pytest.raises(ValueError, match="different target identity"):
        _create(client, target_kind=OutboxTargetKind.INDEX_DISPATCH)
    with pytest.raises(ValueError, match="different target identity"):
        _create(client, generation=2)
    with pytest.raises(ValueError, match="different target identity"):
        _create(client, schedule_at=NOW + timedelta(seconds=1))


@pytest.mark.parametrize("target", [
    "telegram_dispatch", "telegram_status_check", "mcp_save_recovery", "index_dispatch",
])
def test_all_and_only_allowed_target_kinds_are_accepted(firestore_fake, target):
    client = firestore_fake.client_class()
    record = _create(client, target_kind=target)
    assert record.target_kind.value == target


@pytest.mark.parametrize("kwargs", [
    {"outbox_id": "bad/id"},
    {"generation": 0},
    {"generation": True},
    {"target_kind": "other"},
    {"operation_id": "bad/id"},
    {"operation_id": None, "context_id": None},
    {"schedule_at": datetime(2026, 4, 1)},
    {"now": datetime(2026, 4, 1)},
])
def test_invalid_identity_or_time_is_rejected(firestore_fake, kwargs):
    client = firestore_fake.client_class()
    outbox_id = kwargs.pop("outbox_id", "outbox-1")
    values = {
        "target_kind": OutboxTargetKind.TELEGRAM_DISPATCH,
        "generation": 1,
        "schedule_at": NOW,
        "operation_id": "op-1",
        "now": NOW,
    }
    values.update(kwargs)
    with pytest.raises(ValueError):
        create_outbox_record(client, outbox_id, **values)


def test_due_query_filters_state_and_schedule_and_applies_positive_bound(firestore_fake):
    client = firestore_fake.client_class()
    _create(client)
    _create(client, outbox_id="future", operation_id="op-future", schedule_at=NOW + timedelta(seconds=5))
    _create(client, outbox_id="later", operation_id="op-later", schedule_at=NOW)
    client.collection(OUTBOX_COLLECTION).document("later").update({"state": OutboxState.DELIVERED.value})
    client.collection(OUTBOX_COLLECTION).document("outbox-1").update({"state": OutboxState.RETRY.value})

    due = list_due_outbox_records(client, now=NOW, limit=1)
    assert len(due) == 1
    assert due[0].outbox_id == "outbox-1"
    assert due[0].state is OutboxState.RETRY
    assert list_due_outbox_records(client, now=NOW, limit=2)[0].outbox_id == "outbox-1"


def test_due_query_never_returns_more_than_requested_limit(firestore_fake):
    client = firestore_fake.client_class()
    for outbox_id in ("one", "two", "three"):
        _create(client, outbox_id=outbox_id, operation_id=f"op-{outbox_id}")

    assert len(list_due_outbox_records(client, now=NOW, limit=2)) == 2


def test_due_query_filters_enabled_target_kinds_before_applying_bound(firestore_fake):
    client = firestore_fake.client_class()
    _create(client, outbox_id="disabled", target_kind=OutboxTargetKind.INDEX_DISPATCH)
    _create(client, outbox_id="enabled", target_kind=OutboxTargetKind.MCP_SAVE_RECOVERY)

    due = list_due_outbox_records(
        client, now=NOW, limit=1, target_kinds=[OutboxTargetKind.MCP_SAVE_RECOVERY]
    )

    assert [record.outbox_id for record in due] == ["enabled"]
    assert list_due_outbox_records(client, now=NOW, limit=1, target_kinds=[]) == []


def test_delivery_results_are_transactional_and_delivered_state_is_monotonic(firestore_fake):
    client = firestore_fake.client_class()
    _create(client)

    retried = record_outbox_delivery_result(
        client, "outbox-1", delivered=False, now=NOW, safe_error="enqueue_failed"
    )
    assert retried.state is OutboxState.RETRY
    assert retried.delivery_attempts == 1
    assert retried.last_error == "enqueue_failed"

    delivered = record_outbox_delivery_result(client, "outbox-1", delivered=True, now=NOW)
    late_failure = record_outbox_delivery_result(
        client, "outbox-1", delivered=False, now=NOW, safe_error="enqueue_failed"
    )
    assert delivered.state is OutboxState.DELIVERED
    assert late_failure.state is OutboxState.DELIVERED
    assert late_failure.delivery_attempts == 3
    assert late_failure.last_error is None
    assert get_outbox_record(client, "outbox-1") == late_failure


def test_delivery_result_rejects_unsafe_diagnostic_and_missing_id(firestore_fake):
    client = firestore_fake.client_class()
    _create(client)
    with pytest.raises(ValueError, match="diagnostic code"):
        record_outbox_delivery_result(
            client, "outbox-1", delivered=False, now=NOW, safe_error="https://secret.invalid/token"
        )
    assert record_outbox_delivery_result(
        client, "missing", delivered=True, now=NOW
    ) is None


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, 501])
def test_due_query_rejects_nonpositive_or_unbounded_limits(firestore_fake, limit):
    client = firestore_fake.client_class()
    with pytest.raises(ValueError, match="limit"):
        list_due_outbox_records(client, now=NOW, limit=limit)


def test_task_payload_contains_only_schema_and_identifier_fields(firestore_fake):
    client = firestore_fake.client_class()
    record = _create(
        client,
        context_id="ctx-1",
        document_id="doc-1",
        save_intent_id="save-1",
    )
    payload = task_payload(record)

    assert payload == {
        "schema_version": 1,
        "outbox_id": "outbox-1",
        "target_kind": "telegram_dispatch",
        "generation": 1,
        "operation_id": "op-1",
        "context_id": "ctx-1",
        "document_id": "doc-1",
        "save_intent_id": "save-1",
    }
    assert not {"text", "command", "notification", "token", "payload"} & payload.keys()


def test_transaction_composable_creation_and_notification_reference(firestore_fake):
    from google.cloud import firestore

    client = firestore_fake.client_class()

    @firestore.transactional
    def create(transaction):
        return create_outbox_record_in_transaction(
            client, transaction, "outbox-ack", target_kind=OutboxTargetKind.TELEGRAM_DISPATCH,
            generation=1, schedule_at=NOW, now=NOW, context_id="context-1",
            notification_id="notification-1",
        )

    record = create(client.transaction())
    assert record.notification_id == "notification-1"
    assert task_payload(record)["notification_id"] == "notification-1"
    assert create(client.transaction()) == record


def test_firestore_indexes_are_valid_and_match_schema_collection_groups():
    config_path = Path(__file__).resolve().parents[4] / "infra" / "firestore" / "indexes.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["indexes"] == [
        {
            "collectionGroup": "task_outbox",
            "queryScope": "COLLECTION",
            "fields": [
                {"fieldPath": "state", "order": "ASCENDING"},
                {"fieldPath": "schedule_at", "order": "ASCENDING"},
            ],
        },
        {
            "collectionGroup": "task_outbox",
            "queryScope": "COLLECTION",
            "fields": [
                {"fieldPath": "state", "order": "ASCENDING"},
                {"fieldPath": "target_kind", "order": "ASCENDING"},
                {"fieldPath": "schedule_at", "order": "ASCENDING"},
            ],
        },
        {
            "collectionGroup": "dialog_operations",
            "queryScope": "COLLECTION",
            "fields": [
                {"fieldPath": "origin", "order": "ASCENDING"},
                {"fieldPath": "state", "order": "ASCENDING"},
                {"fieldPath": "updated_at", "order": "ASCENDING"},
            ],
        },
    ]
    overrides = {(item["collectionGroup"], item["fieldPath"]): item["indexes"]
                 for item in config["fieldOverrides"]}
    assert overrides == {
        ("messages", "text"): [],
        ("messages", "caption"): [],
        ("messages", "transcription_text"): [],
        ("prepared", "text"): [],
        ("turns", "text"): [],
        ("turns", "parts"): [],
        ("operations", "raw_command_envelope"): [],
        ("notification_outbox", "body"): [],
        ("input_parts", "payload"): [],
    }
