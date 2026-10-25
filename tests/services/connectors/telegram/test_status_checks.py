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
import os
import socket
from datetime import datetime, timedelta, timezone

import pytest
from starlette.requests import Request

from core.integrations.firestore.outbox import (
    OUTBOX_COLLECTION,
    OutboxTargetKind,
    create_outbox_record,
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
    def __init__(self, outcome: str = "delivered"):
        self.calls: list[str] = []
        self.outcome = outcome

    async def publish(self, notification_id: str) -> str:
        self.calls.append(notification_id)
        return self.outcome


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


async def _direct_to_thread(func, /, *args, **kwargs):
    return func(*args, **kwargs)


def _run(awaitable):
    original_to_thread = asyncio.to_thread
    asyncio.to_thread = _direct_to_thread
    try:
        return asyncio.run(awaitable, loop_factory=_selector_loop)
    finally:
        asyncio.to_thread = original_to_thread


def _selector_loop():
    """Use loopback datagrams for asyncio's Windows self-pipe under the socket guard."""
    if os.name != "nt":
        return asyncio.SelectorEventLoop()
    original_socketpair = socket.socketpair
    pairs = []

    def local_pair(*_args, **_kwargs):
        reader = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        writer = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            reader.bind(("127.0.0.1", 0))
            writer.bind(("127.0.0.1", 0))
            writer.connect(reader.getsockname())
            reader.connect(writer.getsockname())
            pairs.append((reader, writer))
            return reader, writer
        except BaseException:
            reader.close()
            writer.close()
            raise

    socket.socketpair = local_pair
    try:
        return asyncio.SelectorEventLoop()
    finally:
        socket.socketpair = original_socketpair


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

    assert _run(checks.handle_task(task_payload(record))) == "running"
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
                       "final_delivery_state": "delivered", "notification_id": "final-save-op"})
    clock.current = records[0].schedule_at

    assert _run(checks.handle_task(task_payload(records[0]))) == "no_op"
    assert _run(checks.handle_task(task_payload(records[1]))) == "no_op"
    assert _run(checks.handle_task(task_payload(records[2]))) == "no_op"
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
                       "final_delivery_state": "delivered"})
    checks = _service(client, clock=_Clock(), reader=_StatusReader("completed"), publisher=_Publisher())
    records = checks.schedule_checks(accepted.context_id, "save-op", ACCEPTED_AT)

    # Replayed +60/+120/+300 status tasks are older than the delivered final.
    assert [_run(checks.handle_task(task_payload(record))) for record in records] == [
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

    assert _run(checks.handle_task(task_payload(records[2]))) == "running"
    assert publisher.calls == ["final-save-op"]
    still_running = operation_ref.get().to_dict()
    assert still_running["state"] == "running"
    assert still_running["attempts_started"] == 1 and still_running["failures"] == 0
    assert still_running["lease_owner"] == "worker-1"
    assert reader.calls == ["save-op"]

    # Completion uses the ordinary durable notification ID; the check does not
    # create a replacement notification or execute the save again.
    operation_ref.update({"state": "completed", "final_delivery_state": "pending"})
    assert _run(publisher.publish("final-save-op")) == "delivered"
    operation_ref.update({"final_delivery_state": "delivered"})
    assert _run(checks.handle_task(task_payload(records[2]))) == "no_op"
    completed = operation_ref.get().to_dict()
    assert completed["state"] == "completed"
    assert completed["attempts_started"] == 1 and completed["failures"] == 0
    assert publisher.calls == ["final-save-op", "final-save-op"]


def test_real_operation_without_notification_id_gets_one_stable_progress_notification(firestore_fake):
    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    accepted = repo.accept_forward(
        31, chat_id=-10042, thread_id=5,
        message={"message_id": 41, "text": "synthetic source"}, now=ACCEPTED_AT,
    )
    operation_id = "save-op-no-notification"
    context_ref = client.collection("telegram_sessions").document(accepted.context_id)
    context_ref.collection("operations").document(operation_id).set({
        "operation_id": operation_id,
        "input_ref": f"telegram_sessions/{accepted.context_id}/commands/1",
    })
    # The notification metadata comes from the accepted command's real draft.
    command_ref = client.collection("telegram_sessions").document(accepted.context_id).collection("commands").document("1")
    command_ref.set({"draft_id": accepted.draft_id})
    operation_ref = client.collection("dialog_operations").document(operation_id)
    operation_ref.set({"operation_id": operation_id, "context_id": accepted.context_id,
                       "state": "running", "final_delivery_state": "pending"})
    clock, reader, publisher = _Clock(), _StatusReader("running"), _Publisher()
    checks = _service(client, clock=clock, reader=reader, publisher=publisher)
    records = checks.schedule_checks(accepted.context_id, operation_id, ACCEPTED_AT)

    first = _run(checks.handle_task(task_payload(records[0])))
    notification_ids = [
        data["notification_id"] for snapshot in client.collection("notification_outbox").stream()
        if (data := snapshot.to_dict() or {}).get("kind") == "progress"
    ]
    first_revision = copy.deepcopy(repo.get_notification(notification_ids[0]))
    second = _run(checks.handle_task(task_payload(records[1])))
    third = _run(checks.handle_task(task_payload(records[2])))

    assert first == second == third == "running"
    assert len(notification_ids) == 1
    assert publisher.calls == [notification_ids[0]] * 3
    notification = repo.get_notification(notification_ids[0])
    assert notification["kind"] == "progress"
    assert notification["operation_id"] == operation_id
    assert notification["send_attempts_started"] == 0
    assert notification["revision"] == first_revision["revision"] + 2
    assert notification["body"] != first_revision["body"]
    assert "5" in notification["body"]


def test_queued_operation_status_works_before_shared_save_operation_exists(firestore_fake):
    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    accepted = repo.accept_forward(
        31, chat_id=-10042, thread_id=5,
        message={"message_id": 41, "text": "synthetic source"}, now=ACCEPTED_AT,
    )
    operation_id = "queued-op"
    context_ref = client.collection("telegram_sessions").document(accepted.context_id)
    context_ref.collection("operations").document(operation_id).set({
        "operation_id": operation_id, "state": "pending",
        "input_ref": f"telegram_sessions/{accepted.context_id}/commands/1",
    })
    context_ref.collection("commands").document("1").set({"draft_id": accepted.draft_id})
    publisher, clock = _Publisher(), _Clock()
    checks = _service(client, clock=clock, reader=lambda _: None, publisher=publisher)
    record = checks.schedule_checks(accepted.context_id, operation_id, ACCEPTED_AT)[0]

    assert _run(checks.handle_task(task_payload(record))) == "pending"
    rows = [row.to_dict() for row in client.collection("notification_outbox").stream()]
    progress = [row for row in rows if row.get("kind") == "progress"]
    assert len(progress) == 1
    assert progress[0]["operation_id"] == operation_id
    assert publisher.calls == [progress[0]["notification_id"]]


@pytest.mark.parametrize("selected_dialog_number", [7, None])
def test_regular_command_progress_does_not_require_a_draft_or_dialog_number(
    firestore_fake, selected_dialog_number,
):
    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    accepted = repo.accept_forward(
        31, chat_id=-10042, thread_id=5,
        message={"message_id": 41, "text": "synthetic source"}, now=ACCEPTED_AT,
    )
    operation_id = "regular-command"
    context_ref = client.collection("telegram_sessions").document(accepted.context_id)
    context_ref.update({"selected_dialog_number": selected_dialog_number})
    context_ref.collection("operations").document(operation_id).set({
        "operation_id": operation_id, "state": "pending",
    })
    # This ordinary command has no draft reference or dialog-number draft.
    publisher = _Publisher()
    checks = _service(client, reader=lambda _: None, publisher=publisher)
    record = checks.schedule_checks(accepted.context_id, operation_id, ACCEPTED_AT)[0]

    assert _run(checks.handle_task(task_payload(record))) == "pending"
    notification = repo.get_notification(publisher.calls[0])
    if selected_dialog_number is None:
        assert notification["dialog_number"] is None
        assert ":" not in notification["body"]
    else:
        assert notification["dialog_number"] == selected_dialog_number
        assert notification["body"].startswith(f"Dialog #{selected_dialog_number}:")


def test_terminal_check_uses_persisted_final_lifecycle_notification(firestore_fake):
    client = firestore_fake.client_class()
    publisher = _Publisher()
    reader = _StatusReader("completed")
    checks = _service(client, reader=reader, publisher=publisher)
    checks.terminal_notification_updater = lambda record: "final-save-op"
    record = checks.schedule_checks("ctx-1", "save-op", ACCEPTED_AT)[0]
    client.collection("dialog_operations").document("save-op").set(
        {"operation_id": "save-op", "context_id": "ctx-1", "state": "completed",
         "final_delivery_state": "pending"}
    )

    assert _run(checks.handle_task(task_payload(record))) == "completed"
    assert publisher.calls == ["final-save-op"]
    assert list(client.collection("notification_outbox").stream()) == []


def test_terminal_operation_without_draft_lifecycle_still_gets_status_notification(firestore_fake):
    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    accepted = repo.accept_forward(
        31, chat_id=-10042, thread_id=5,
        message={"message_id": 41, "text": "synthetic source"}, now=ACCEPTED_AT,
    )
    operation_id = "regular-command-terminal"
    context_ref = client.collection("telegram_sessions").document(accepted.context_id)
    context_ref.collection("operations").document(operation_id).set({
        "operation_id": operation_id, "state": "completed",
    })
    publisher = _Publisher()
    checks = _service(
        client, reader=lambda _: None, publisher=publisher,
        clock=_Clock(),
    )
    checks.terminal_notification_updater = lambda _: None
    record = checks.schedule_checks(accepted.context_id, operation_id, ACCEPTED_AT)[0]

    assert _run(checks.handle_task(task_payload(record))) == "completed"
    notification = repo.get_notification(publisher.calls[0])
    assert notification["kind"] == "final"
    assert notification["operation_id"] == operation_id
    assert notification["body"] != ""


@pytest.mark.parametrize(
    ("state", "kind"), [("completed", "final"), ("failed", "error")],
)
def test_terminal_fallback_persists_delivery_and_suppresses_remaining_checks(
    firestore_fake, state, kind,
):
    from services.connectors.telegram.notifications import TelegramNotificationPublisher

    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    accepted = repo.accept_forward(
        31, chat_id=-10042, thread_id=5,
        message={"message_id": 41, "text": "synthetic source"}, now=ACCEPTED_AT,
    )
    operation_id = f"terminal-{state}"
    client.collection("telegram_sessions").document(accepted.context_id).collection(
        "operations"
    ).document(operation_id).set({"operation_id": operation_id, "state": state})
    clock = _Clock()
    records = _service(
        client, reader=lambda _: {"state": state}, clock=clock,
        publisher=_Publisher(),
    ).schedule_checks(accepted.context_id, operation_id, ACCEPTED_AT)

    class FakeTelegram:
        def __init__(self):
            self.sent = []

        async def send_message(self, chat_id, text, *, message_thread_id=None):
            self.sent.append((chat_id, text, message_thread_id))
            return {"message_id": 912}

        async def edit_message_text(self, chat_id, message_id, text):
            raise AssertionError("terminal fallback should send only once")

    telegram = FakeTelegram()
    durable_publisher = TelegramNotificationPublisher(repo, telegram, clock=clock)

    class CountingPublisher:
        def __init__(self):
            self.calls = []

        async def publish(self, notification_id):
            self.calls.append(notification_id)
            return await durable_publisher.publish(notification_id)

    publisher = CountingPublisher()
    checks = _service(
        client, reader=lambda _: {"state": state}, clock=clock,
        publisher=publisher,
    )
    checks.terminal_notification_updater = lambda _: None

    assert _run(checks.handle_task(task_payload(records[0]))) == state
    assert not client.collection("dialog_operations").document(operation_id).get().exists
    terminal_id = publisher.calls[0]
    notification = repo.get_notification(terminal_id)
    assert notification["kind"] == kind
    assert notification["operation_id"] == operation_id
    assert notification["state"] == "delivered"
    assert len(telegram.sent) == 1

    assert [
        _run(checks.handle_task(task_payload(record))) for record in records[1:]
    ] == ["no_op", "no_op"]
    assert publisher.calls == [terminal_id]
    assert len(telegram.sent) == 1


def test_failed_terminal_fallback_renders_persisted_error_and_bound_dialog_after_selection_changes(
    firestore_fake,
):
    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    accepted = repo.accept_forward(
        31, chat_id=-10042, thread_id=5,
        message={"message_id": 41, "text": "synthetic source"}, now=ACCEPTED_AT,
    )
    context_ref = client.collection("telegram_sessions").document(accepted.context_id)
    context_ref.update({"selected_dialog_number": 9, "active_draft_id": "newer-draft",
                        "reserved_operation_id": "newer-operation", "lock_owner": "newer-worker"})
    context_before = context_ref.get().to_dict()
    operation_id = "failed-with-safe-error"
    context_ref.collection("operations").document(operation_id).set({
        "operation_id": operation_id,
        "state": "failed",
        "input_ref": f"telegram_sessions/{accepted.context_id}/commands/1",
    })
    context_ref.collection("commands").document("1").set({"draft_id": accepted.draft_id})
    client.collection("dialog_operations").document(operation_id).set({
        "operation_id": operation_id,
        "origin": "telegram",
        "caller_binding": "telegram:test",
        "immutable_input_digest": "digest",
        "immutable_input_ref": "telegram_sessions/test/drafts/input",
        "state": "failed",
        "stage": "indexing",
        "error": {
            "service": "google_drive",
            "stage": "indexing",
            "code": "provider_error",
            "category": "rate_limited",
            "sanitized_message": "Stage execution failed (RuntimeError)",
            "occurred_at": ACCEPTED_AT,
            "retryable": True,
            "failures": 3,
            "operation_id": operation_id,
        },
    })
    publisher = _Publisher()
    checks = _service(client, reader=lambda _: {"state": "failed"}, publisher=publisher)
    checks.terminal_notification_updater = lambda _: None
    record = checks.schedule_checks(accepted.context_id, operation_id, ACCEPTED_AT)[0]

    assert _run(checks.handle_task(task_payload(record))) == "failed"

    notification = repo.get_notification(publisher.calls[0])
    body = notification["body"]
    assert notification["kind"] == "error"
    assert notification["dialog_number"] == 1
    assert body.startswith("Dialog #1:")
    assert "Error in google_drive/indexing" in body
    assert "code=provider_error, category=rate_limited" in body
    assert "2026-10-22T12:00:00+00:00" in body
    assert f"operation_id={operation_id}" in body
    assert "Automatic retries have stopped" in body
    assert "administrator" in body and "logs" in body
    assert "billing" not in body.casefold()
    assert context_ref.get().to_dict() == context_before


def test_failed_queue_operation_without_shared_record_uses_failed_stage_error(firestore_fake):
    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    accepted = repo.accept_forward(
        31, chat_id=-10042, thread_id=5,
        message={"message_id": 41, "text": "synthetic source"}, now=ACCEPTED_AT,
    )
    context_ref = client.collection("telegram_sessions").document(accepted.context_id)
    operation_id = "queue-stage-terminal-failure"
    context_ref.collection("operations").document(operation_id).set({
        "operation_id": operation_id,
        "state": "failed",
        "input_ref": f"telegram_sessions/{accepted.context_id}/commands/1",
    })
    context_ref.collection("commands").document("1").set({"draft_id": accepted.draft_id})
    context_ref.collection("operations").document(operation_id).collection("stages").document(
        "telegram.audio.41"
    ).set({
        "stage_key": "telegram.audio.41",
        "state": "failed",
        "failures": 3,
        "last_error": {
            "service": "speech_provider",
            "stage": "telegram.audio.41",
            "code": "provider_error",
            "category": "rate_limited",
            "message": "Stage execution failed (RuntimeError)",
            "sanitized_message": "Stage execution failed (RuntimeError)",
            "occurred_at": ACCEPTED_AT,
            "retryable": True,
            "failures": 3,
            "operation_id": operation_id,
        },
    })
    publisher = _Publisher()
    checks = _service(client, reader=lambda _: {"state": "failed"}, publisher=publisher)
    checks.terminal_notification_updater = lambda _: None
    record = checks.schedule_checks(accepted.context_id, operation_id, ACCEPTED_AT)[0]

    assert _run(checks.handle_task(task_payload(record))) == "failed"
    assert not client.collection("dialog_operations").document(operation_id).get().exists

    notification = repo.get_notification(publisher.calls[0])
    body = notification["body"]
    assert notification["kind"] == "error"
    assert "Error in speech_provider/telegram.audio.41" in body
    assert "code=provider_error, category=rate_limited" in body
    assert "2026-10-22T12:00:00+00:00" in body
    assert f"operation_id={operation_id}" in body
    assert "Automatic retries have stopped" in body
    assert "administrator" in body and "logs" in body
    assert "billing" not in body.casefold()


def test_status_read_failure_preserves_execution_focus_draft_and_save_state(firestore_fake):
    client = firestore_fake.client_class()
    clock, reader, publisher = _Clock(), _StatusReader("running"), _Publisher()
    checks = _service(client, clock=clock, reader=reader, publisher=publisher)
    record = checks.schedule_checks("ctx-1", "save-op", ACCEPTED_AT)[0]
    session_ref = client.collection("telegram_sessions").document("ctx-1")
    session_ref.set({"chat_id": -10042, "reserved_operation_id": "save-op", "active_draft_id": "draft-1",
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
        _run(checks.handle_task(task_payload(record)))

    assert firestore_fake.documents == before
    assert publisher.calls == []
    assert reader.calls == ["save-op"]

    # The same durable task can be retried after the status provider recovers.
    reader.error = None
    assert _run(checks.handle_task(task_payload(record))) == "running"
    assert reader.calls == ["save-op", "save-op"]
    operation = operation_ref.get().to_dict()
    assert operation["attempts_started"] == 2 and operation["failures"] == 1
    assert operation["lease_owner"] == "worker-1"
    context = session_ref.get().to_dict()
    assert context["reserved_operation_id"] == "save-op"
    assert context["active_draft_id"] == "draft-1"
    assert context["selected_dialog_number"] == 7
    assert context["lock_owner"] == "worker-1"
    assert client.collection("telegram_sessions").document("ctx-1").collection(
        "drafts"
    ).document("draft-1").get().to_dict()["state"] == "processing"


@pytest.mark.parametrize("outcome", ["retry", "busy", "stale", "pending"])
def test_pending_status_delivery_is_retryable_and_does_not_ack_task(firestore_fake, outcome):
    client = firestore_fake.client_class()
    accepted = TelegramRepository(client).accept_forward(
        31, chat_id=-10042, thread_id=5,
        message={"message_id": 41, "text": "synthetic source"}, now=ACCEPTED_AT,
    )
    checks = _service(
        client, reader=_StatusReader("running"), publisher=_Publisher(outcome),
    )
    record = checks.schedule_checks(accepted.context_id, "save-op", ACCEPTED_AT)[0]
    client.collection("dialog_operations").document("save-op").set({
        "operation_id": "save-op", "context_id": accepted.context_id, "state": "running",
    })

    with pytest.raises(RuntimeError, match="delivery is pending"):
        _run(checks.handle_task(task_payload(record)))


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

    class RetryWorker:
        async def handle_task(self, payload):
            raise RuntimeError("notification still pending")

    app.state.telegram_status_check_worker = RetryWorker()
    retry = _run(endpoint(_request("/internal/status-check", payload, token="valid")))
    assert retry.status_code == 503


def test_dispatch_schedules_acceptance_relative_checks_before_queue_worker(firestore_fake, monkeypatch):
    monkeypatch.setattr(asyncio, "to_thread", _direct_to_thread)
    _configure_status_task_target(monkeypatch)
    client = firestore_fake.client_class()
    accepted_at = ACCEPTED_AT
    record = create_outbox_record(
        client, "tgcommand-operation-1", target_kind=OutboxTargetKind.TELEGRAM_DISPATCH,
        generation=1, schedule_at=ACCEPTED_AT + timedelta(hours=2), now=accepted_at,
        context_id="ctx-1", operation_id="operation-1",
    )
    events = []

    def enqueue_task(**kwargs):
        events.append(("enqueue", kwargs))
        # Cloud Tasks reports an existing deterministic task as False.
        return not kwargs["task_id"].endswith("-120-g1")

    monkeypatch.setattr("core.integrations.cloud_tasks.client.enqueue_task", enqueue_task)

    class QueueWorker:
        async def handle_task(self, payload):
            events.append(("queue", dict(payload)))
            return True

    class StatusWorker:
        def schedule_checks(self, context_id, operation_id, scheduled_at):
            events.append(("schedule", context_id, operation_id, scheduled_at))
            from services.connectors.telegram.status_checks import TelegramStatusChecks

            return TelegramStatusChecks(
                client, notification_publisher=_Publisher(), clock=lambda: ACCEPTED_AT,
            ).schedule_checks(context_id, operation_id, scheduled_at)

        async def handle_task(self, payload):
            raise AssertionError("dispatch must not execute a status check")

    app = create_app(
        queue_worker=QueueWorker(), status_check_worker=StatusWorker(),
        oidc_verifier=lambda token: type("Identity", (), {"authorized": token == "valid"})()
        if token == "valid" else None,
    )
    monkeypatch.setattr("core.integrations.firestore.client.get_client", lambda: client)
    monkeypatch.setattr("services.connectors.telegram.app.main._first_ack_state", lambda record: None)
    monkeypatch.setattr("services.connectors.telegram.app.main._update_terminal_notification", lambda record: None)
    endpoint = _route(app, "/internal/dispatch")

    response = _run(endpoint(_request("/internal/dispatch", task_payload(record), token="valid")))

    assert response == {"status": "processed"}
    assert [event[0] for event in events] == [
        "schedule", "enqueue", "enqueue", "enqueue", "queue",
    ]
    schedule_event = events[0]
    assert schedule_event == ("schedule", "ctx-1", "operation-1", record.created_at)
    assert schedule_event[3] != record.schedule_at
    scheduled = [
        item for item in _scheduled_rows(client, "operation-1")
        if item.target_kind is OutboxTargetKind.TELEGRAM_STATUS_CHECK
    ]
    assert len(scheduled) == 3
    assert [item.schedule_at for item in scheduled] == [
        record.created_at + timedelta(seconds=60),
        record.created_at + timedelta(seconds=120),
        record.created_at + timedelta(seconds=300),
    ]
    enqueued = [event[1] for event in events if event[0] == "enqueue"]
    assert [call["schedule_time"] for call in enqueued] == [item.schedule_at for item in scheduled]
    assert [call["task_id"] for call in enqueued] == [item.task_name for item in scheduled]
    assert [call["body"] for call in enqueued] == [task_payload(item) for item in scheduled]
    assert all(call["queue_name"] == "telegram-status-queue" for call in enqueued)
    assert all(call["url"] == "https://telegram.synthetic.test/internal/status-check" for call in enqueued)
    assert all(call["service_account_email"] == "worker@synthetic.test" for call in enqueued)
    assert all(call["audience"] == "https://telegram.synthetic.test" for call in enqueued)
    assert all(item.state.value == "delivered" for item in scheduled)


def _configure_status_task_target(monkeypatch):
    monkeypatch.setenv("TELEGRAM_CONTROL_URL", "https://telegram.synthetic.test/internal/dispatch")
    monkeypatch.setenv("TELEGRAM_CONTROL_QUEUE", "telegram-status-queue")
    monkeypatch.setenv("CLOUD_TASKS_OIDC_AUDIENCE", "https://telegram.synthetic.test")
    monkeypatch.setenv("OIDC_AUDIENCE", "https://telegram.synthetic.test")
    monkeypatch.setenv("CLOUD_TASKS_OIDC_SERVICE_ACCOUNT_EMAIL", "worker@synthetic.test")
    monkeypatch.setenv("OIDC_SERVICE_ACCOUNT_EMAIL", "worker@synthetic.test")


def test_dispatch_enqueue_failure_marks_status_row_retry_and_skips_queue(
    firestore_fake, monkeypatch,
):
    monkeypatch.setattr(asyncio, "to_thread", _direct_to_thread)
    _configure_status_task_target(monkeypatch)
    client = firestore_fake.client_class()
    record = create_outbox_record(
        client, "tgcommand-operation-failure", target_kind=OutboxTargetKind.TELEGRAM_DISPATCH,
        generation=1, schedule_at=ACCEPTED_AT + timedelta(hours=2), now=ACCEPTED_AT,
        context_id="ctx-1", operation_id="operation-failure",
    )
    events = []

    def enqueue_task(**kwargs):
        events.append(kwargs)
        raise RuntimeError("synthetic provider detail must not persist")

    monkeypatch.setattr("core.integrations.cloud_tasks.client.enqueue_task", enqueue_task)

    class QueueWorker:
        async def handle_task(self, payload):
            events.append("queue")
            return True

    class StatusWorker:
        def schedule_checks(self, context_id, operation_id, scheduled_at):
            from services.connectors.telegram.status_checks import TelegramStatusChecks

            return TelegramStatusChecks(
                client, notification_publisher=_Publisher(), clock=lambda: ACCEPTED_AT,
            ).schedule_checks(context_id, operation_id, scheduled_at)

    app = create_app(
        queue_worker=QueueWorker(), status_check_worker=StatusWorker(),
        oidc_verifier=lambda token: type("Identity", (), {"authorized": token == "valid"})()
        if token == "valid" else None,
    )
    monkeypatch.setattr("core.integrations.firestore.client.get_client", lambda: client)
    monkeypatch.setattr("services.connectors.telegram.app.main._first_ack_state", lambda _: None)
    monkeypatch.setattr("services.connectors.telegram.app.main._update_terminal_notification", lambda _: None)

    response = _run(_route(app, "/internal/dispatch")(
        _request("/internal/dispatch", task_payload(record), token="valid")
    ))

    assert response.status_code == 503
    assert len(events) == 1 and isinstance(events[0], dict)
    failed = get_outbox_record(client, events[0]["body"]["outbox_id"])
    assert failed is not None
    assert failed.state.value == "retry"
    assert failed.last_error == "cloud_tasks_enqueue_failed"
    rows = [item for item in _scheduled_rows(client, "operation-failure")
            if item.target_kind is OutboxTargetKind.TELEGRAM_STATUS_CHECK]
    assert len(rows) == 3
    assert sum(item.state.value == "retry" for item in rows) == 1
