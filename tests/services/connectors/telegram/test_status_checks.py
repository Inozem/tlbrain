"""Acceptance-relative Telegram status checks and delivery recovery contract.

T083 supplies ``TelegramStatusChecks`` with this narrow interface:

* ``schedule_checks(context_id, operation_id, accepted_at)`` creates the three
  canonical status-check outbox records;
* ``handle_task(payload)`` reads the persisted operation and notification state
  using the identifiers in a canonical task payload.

These tests use only the Firestore fake, a controlled clock, and scripted
read/publish boundaries.
"""

from __future__ import annotations

import asyncio
import copy
import json
from datetime import datetime, timedelta, timezone

import pytest
from starlette.requests import Request

from core.integrations.firestore.outbox import (
    OUTBOX_COLLECTION,
    OutboxTargetKind,
    get_outbox_record,
    task_payload,
)
from services.connectors.telegram.app.main import create_app
from services.connectors.telegram.repository import TelegramRepository


pytestmark = pytest.mark.offline
ACCEPTED_AT = datetime(2026, 10, 22, 12, 0, tzinfo=timezone.utc)


class _Clock:
    def __init__(self, current: datetime = ACCEPTED_AT):
        self.current = current

    def __call__(self) -> datetime:
        return self.current

    def advance(self, seconds: int) -> None:
        self.current += timedelta(seconds=seconds)


class _StatusReader:
    def __init__(self, status: str = "running"):
        self.status = status
        self.calls: list[str] = []
        self.error: BaseException | None = None

    def __call__(self, operation_id: str):
        self.calls.append(operation_id)
        if self.error is not None:
            raise self.error
        return {"operation_id": operation_id, "state": self.status}


class _Publisher:
    def __init__(self):
        self.calls: list[str] = []

    async def publish(self, notification_id: str) -> str:
        self.calls.append(notification_id)
        return "delivered"


def _service(client, *, clock=None, reader=None, publisher=None):
    # T083 adds this module. Keep the import local so the full test file still
    # collects while that implementation is pending.
    from services.connectors.telegram.status_checks import TelegramStatusChecks

    return TelegramStatusChecks(
        client,
        status_reader=reader or _StatusReader(),
        notification_publisher=publisher or _Publisher(),
        clock=clock or _Clock(),
    )


def _scheduled_rows(client, operation_id="save-op"):
    rows = []
    for snapshot in client.collection(OUTBOX_COLLECTION).stream():
        record = get_outbox_record(client, snapshot.id)
        if record and record.operation_id == operation_id:
            rows.append(record)
    return sorted(rows, key=lambda row: row.schedule_at)


def _request(path, payload, token="valid"):
    headers = [(b"content-type", b"application/json")]
    if token is not None:
        headers.append((b"authorization", f"Bearer {token}".encode()))
    body = json.dumps(payload).encode()

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(
        {"type": "http", "method": "POST", "path": path,
         "headers": headers, "query_string": b""},
        receive,
    )


def _route(app, path):
    return next(route.endpoint for route in app.routes if getattr(route, "path", None) == path)


def _run(awaitable):
    return asyncio.run(awaitable)


async def _direct_to_thread(func, /, *args, **kwargs):
    return func(*args, **kwargs)


def test_checks_are_scheduled_at_immutable_acceptance_offsets_and_use_id_only_payloads(firestore_fake):
    client = firestore_fake.client_class()
    clock = _Clock()
    checks = _service(client, clock=clock)

    scheduled = checks.schedule_checks("ctx-1", "save-op", ACCEPTED_AT)
    assert [record.schedule_at for record in scheduled] == [
        ACCEPTED_AT + timedelta(seconds=60),
        ACCEPTED_AT + timedelta(seconds=120),
        ACCEPTED_AT + timedelta(seconds=300),
    ]
    assert all(record.target_kind is OutboxTargetKind.TELEGRAM_STATUS_CHECK for record in scheduled)
    assert all(record.operation_id == "save-op" and record.context_id == "ctx-1" for record in scheduled)

    # Progress is persisted between checks but is not an input to scheduling.
    operation_ref = client.collection("dialog_operations").document("save-op")
    operation_ref.set({"operation_id": "save-op", "context_id": "ctx-1", "state": "running",
                       "accepted_at": ACCEPTED_AT, "progress": "preparing"})
    clock.advance(45)
    operation_ref.update({"progress": "indexing"})
    assert checks.schedule_checks("ctx-1", "save-op", ACCEPTED_AT) == scheduled

    payloads = [task_payload(record) for record in scheduled]
    expected_keys = {"schema_version", "outbox_id", "target_kind", "generation",
                     "context_id", "operation_id"}
    assert all(set(payload) == expected_keys for payload in payloads)
    assert all(payload["target_kind"] == OutboxTargetKind.TELEGRAM_STATUS_CHECK.value
               for payload in payloads)
    assert all("text" not in payload and "offset" not in payload for payload in payloads)


@pytest.mark.parametrize("offset", [60, 120, 300])
def test_each_due_check_reads_the_same_operation_without_claiming_execution(firestore_fake, offset):
    client = firestore_fake.client_class()
    clock, reader, publisher = _Clock(), _StatusReader("running"), _Publisher()
    checks = _service(client, clock=clock, reader=reader, publisher=publisher)
    records = checks.schedule_checks("ctx-1", "save-op", ACCEPTED_AT)
    client.collection("dialog_operations").document("save-op").set(
        {"operation_id": "save-op", "context_id": "ctx-1", "state": "running",
         "accepted_at": ACCEPTED_AT, "attempts_started": 0, "failures": 0,
         "lease_owner": None, "notification_id": "progress-save-op"}
    )
    record = next(item for item in records if item.schedule_at == ACCEPTED_AT + timedelta(seconds=offset))
    clock.current = record.schedule_at

    assert checks.handle_task(task_payload(record)) == "running"
    assert reader.calls == ["save-op"]
    assert publisher.calls == ["progress-save-op"]
    assert _scheduled_rows(client) == records
    # A status read is not an execution attempt and does not take a queue lease.
    operation = client.collection("dialog_operations").document("save-op").get().to_dict()
    assert operation.get("attempts_started", 0) == 0
    assert operation.get("failures", 0) == 0
    assert operation.get("lease_owner") is None


def test_delivered_final_makes_remaining_checks_noops_without_new_send_or_execution(firestore_fake):
    client = firestore_fake.client_class()
    clock, reader, publisher = _Clock(), _StatusReader("completed"), _Publisher()
    checks = _service(client, clock=clock, reader=reader, publisher=publisher)
    records = checks.schedule_checks("ctx-1", "save-op", ACCEPTED_AT)
    operation_ref = client.collection("dialog_operations").document("save-op")
    operation_ref.set({"operation_id": "save-op", "context_id": "ctx-1", "state": "completed",
                       "accepted_at": ACCEPTED_AT, "attempts_started": 1, "failures": 0,
                       "lease_owner": None, "save_effects": ["effect-1"],
                       "final_notification_state": "delivered", "notification_id": "final-save-op"})
    clock.current = records[0].schedule_at

    assert checks.handle_task(task_payload(records[0])) == "no_op"
    assert checks.handle_task(task_payload(records[1])) == "no_op"
    assert checks.handle_task(task_payload(records[2])) == "no_op"
    assert reader.calls == []
    assert publisher.calls == []
    final_operation = operation_ref.get().to_dict()
    assert final_operation["state"] == "completed"
    assert final_operation["attempts_started"] == 1 and final_operation["failures"] == 0
    assert final_operation["save_effects"] == ["effect-1"]


def test_stale_progress_check_cannot_replace_delivered_final_or_reset_send_budget(firestore_fake):
    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    accepted = repo.accept_forward(
        31, chat_id=-10042, thread_id=5,
        message={"message_id": 41, "text": "synthetic source"}, now=ACCEPTED_AT,
    )
    final = repo.create_notification(
        "final-save-op", context_id=accepted.context_id, kind="final",
        body="completed final", operation_id="save-op", now=ACCEPTED_AT,
    )
    attempt = repo.reserve_notification_attempt(final["notification_id"],
                                               attempt_id="final-send", now=ACCEPTED_AT)
    repo.settle_notification_attempt(final["notification_id"], attempt_id="final-send",
                                     revision=attempt["attempt_revision"], outcome="sent",
                                     message_id=911, now=ACCEPTED_AT)
    final_before = repo.get_notification(final["notification_id"])
    operation_ref = client.collection("dialog_operations").document("save-op")
    operation_ref.set({"operation_id": "save-op", "context_id": accepted.context_id,
                       "state": "completed", "accepted_at": ACCEPTED_AT,
                       "attempts_started": 1, "failures": 0,
                       "notification_id": final["notification_id"],
                       "final_notification_state": "delivered"})
    checks = _service(client, clock=_Clock(), reader=_StatusReader("completed"), publisher=_Publisher())
    records = checks.schedule_checks(accepted.context_id, "save-op", ACCEPTED_AT)

    # Replayed +60/+120/+300 status tasks are older than the delivered final.
    assert [checks.handle_task(task_payload(record)) for record in records] == [
        "no_op", "no_op", "no_op",
    ]
    final_after = repo.get_notification(final["notification_id"])
    assert final_after["state"] == "delivered"
    assert final_after["body"] == final_before["body"]
    assert final_after["revision"] == final_before["revision"]
    assert final_after["message_id"] == 911
    assert final_after["send_attempts_started"] == final_before["send_attempts_started"] == 1
    operation = operation_ref.get().to_dict()
    assert operation["state"] == "completed"
    assert operation["attempts_started"] == 1 and operation["failures"] == 0


def test_pending_at_five_minutes_is_reported_without_failure_then_final_can_deliver(firestore_fake):
    client = firestore_fake.client_class()
    clock, reader, publisher = _Clock(), _StatusReader("running"), _Publisher()
    checks = _service(client, clock=clock, reader=reader, publisher=publisher)
    records = checks.schedule_checks("ctx-1", "save-op", ACCEPTED_AT)
    operation_ref = client.collection("dialog_operations").document("save-op")
    operation_ref.set({"operation_id": "save-op", "context_id": "ctx-1", "state": "running",
                       "accepted_at": ACCEPTED_AT, "attempts_started": 1, "failures": 0,
                       "lease_owner": "worker-1", "notification_id": "final-save-op"})
    clock.current = ACCEPTED_AT + timedelta(seconds=300)

    assert checks.handle_task(task_payload(records[2])) == "running"
    assert publisher.calls == ["final-save-op"]
    still_running = operation_ref.get().to_dict()
    assert still_running["state"] == "running"
    assert still_running["attempts_started"] == 1 and still_running["failures"] == 0
    assert still_running["lease_owner"] == "worker-1"

    # Completion uses the ordinary durable notification ID; the check does not
    # create a replacement notification or execute the save again.
    operation_ref.update({"state": "completed", "final_notification_state": "pending"})
    awaitable = publisher.publish("final-save-op")
    assert _run(awaitable) == "delivered"
    operation_ref.update({"final_notification_state": "delivered"})
    assert checks.handle_task(task_payload(records[2])) == "no_op"
    completed = operation_ref.get().to_dict()
    assert completed["state"] == "completed"
    assert completed["attempts_started"] == 1 and completed["failures"] == 0
    assert publisher.calls == ["final-save-op", "final-save-op"]


def test_status_read_failure_preserves_execution_focus_draft_and_save_state(firestore_fake):
    client = firestore_fake.client_class()
    clock, reader, publisher = _Clock(), _StatusReader("running"), _Publisher()
    checks = _service(client, clock=clock, reader=reader, publisher=publisher)
    record = checks.schedule_checks("ctx-1", "save-op", ACCEPTED_AT)[0]
    session_ref = client.collection("telegram_sessions").document("ctx-1")
    session_ref.set({"reserved_operation_id": "save-op", "active_draft_id": "draft-1",
                     "selected_dialog_number": 7, "lock_owner": "worker-1"})
    client.collection("telegram_sessions").document("ctx-1").collection("drafts").document("draft-1").set(
        {"draft_id": "draft-1", "state": "processing", "manifest_ref": "manifests/m1"}
    )
    operation_ref = client.collection("dialog_operations").document("save-op")
    operation_ref.set({"operation_id": "save-op", "context_id": "ctx-1", "state": "running",
                       "accepted_at": ACCEPTED_AT, "attempts_started": 2, "failures": 1,
                       "lease_owner": "worker-1", "save_effects": ["effect-1"]})
    before = copy.deepcopy(firestore_fake.documents)
    reader.error = OSError("synthetic status store outage")

    with pytest.raises(OSError, match="synthetic status store outage"):
        checks.handle_task(task_payload(record))

    assert firestore_fake.documents == before
    assert publisher.calls == []


def test_internal_status_check_accepts_only_canonical_server_payload(firestore_fake, monkeypatch):
    monkeypatch.setattr(asyncio, "to_thread", _direct_to_thread)
    client = firestore_fake.client_class()
    from core.integrations.firestore.outbox import create_outbox_record

    record = create_outbox_record(
        client, "status-task-1", target_kind=OutboxTargetKind.TELEGRAM_STATUS_CHECK,
        generation=1, schedule_at=ACCEPTED_AT + timedelta(seconds=60), now=ACCEPTED_AT,
        context_id="ctx-1", operation_id="save-op",
    )
    calls = []

    class Worker:
        async def handle_task(self, payload):
            calls.append(dict(payload))
            return "running"

    app = create_app(
        status_check_worker=Worker(),
        oidc_verifier=lambda token: type(
            "Identity", (), {"authorized": token == "valid"}
        )() if token in {"valid", "invalid"} else None,
    )
    monkeypatch.setattr("core.integrations.firestore.client.get_client", lambda: client)
    endpoint = _route(app, "/internal/status-check")
    payload = task_payload(record)
    with_extra_content = {**payload, "text": "caller controlled"}

    async def invoke():
        missing = await endpoint(_request("/internal/status-check", payload, token=None))
        invalid = await endpoint(_request("/internal/status-check", payload, token="invalid"))
        rejected = await endpoint(_request("/internal/status-check", with_extra_content, token="valid"))
        accepted = await endpoint(_request("/internal/status-check", payload, token="valid"))
        return missing, invalid, rejected, accepted

    missing, invalid, rejected, accepted = _run(invoke())
    assert missing.status_code == 401
    assert invalid.status_code == 403
    assert rejected.status_code == 400
    assert accepted == {"status": "running"}
    assert calls == [payload]
