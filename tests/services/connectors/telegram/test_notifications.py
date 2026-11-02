import asyncio
import importlib
import os
import socket
import threading
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from services.connectors.telegram.notifications import (
    _VerifiedTelegramLifecycle, TelegramNotificationPublisher, build_lifecycle_body,
)
from services.connectors.telegram.errors import format_terminal_error
from services.connectors.telegram.client import TelegramAPIError
from services.connectors.telegram.repository import TelegramRepository


NOW = datetime(2026, 10, 22, tzinfo=timezone.utc)


def _accepted(repo, update=1):
    return repo.accept_forward(
        update, chat_id=-10042, thread_id=5,
        message={"message_id": 10 + update, "text": "source"}, now=NOW,
    )


class _Clock:
    def __init__(self):
        self.now = NOW

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += timedelta(seconds=seconds)


class _Telegram:
    def __init__(self, *, failures=0, failure_category=None, retry_after=None):
        self.failures = failures
        self.failure_category = failure_category
        self.retry_after = retry_after
        self.sent = []
        self.edited = []

    async def send_message(self, chat_id, text, *, message_thread_id=None):
        self.sent.append((chat_id, text, message_thread_id))
        if self.failures:
            self.failures -= 1
            if self.failure_category is not None:
                raise TelegramAPIError(self.failure_category, retry_after=self.retry_after)
            raise TimeoutError("synthetic unknown outcome")
        return {"message_id": 901}

    async def edit_message_text(self, chat_id, message_id, text):
        self.edited.append((chat_id, message_id, text))
        return {"message_id": message_id}


def _run(awaitable):
    if os.name != "nt":
        return asyncio.run(awaitable)

    old_pair = socket.socketpair
    pairs = []

    def datagram_pair(*_args, **_kwargs):
        reader = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        writer = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        reader.bind(("127.0.0.1", 0))
        writer.bind(("127.0.0.1", 0))
        writer.connect(reader.getsockname())
        reader.connect(writer.getsockname())
        pairs.append((reader, writer))
        return reader, writer

    socket.socketpair = datagram_pair
    try:
        return asyncio.run(awaitable, loop_factory=asyncio.SelectorEventLoop)
    finally:
        socket.socketpair = old_pair
        for reader, writer in pairs:
            reader.close()
            writer.close()


def test_lifecycle_body_accepts_only_server_result_and_omits_unrequested_link():
    common = dict(dialog_number=12, operation_id="op-1", service="save", stage="verification",
                  reason="completion_unverified")
    without_proof = build_lifecycle_body(lifecycle=_VerifiedTelegramLifecycle(**common))
    assert "saved and indexed" not in without_proof
    assert "Processing or indexing is still in progress" not in without_proof
    assert "operation_id=op-1" in without_proof
    assert "Notes/2026" not in without_proof
    hidden_link = build_lifecycle_body(lifecycle=_VerifiedTelegramLifecycle(
        **{**common, "reason": "completed"}, folder_path="Notes/2026", completion_confirmed=True,
    ))
    assert "Notes/2026" in hidden_link and "Saved and indexed" in hidden_link
    no_path = build_lifecycle_body(lifecycle=_VerifiedTelegramLifecycle(
        **{**common, "reason": "completed"}, folder_path=None, completion_confirmed=True,
    ))
    assert "Saved and indexed" in no_path
    assert "Folder:" not in no_path
    assert "http" not in no_path
    with pytest.raises(TypeError):
        build_lifecycle_body(lifecycle={**common, "current_generation_proof": {"current": True}})


def test_verified_classification_outcome_is_bounded_and_user_visible():
    common = dict(dialog_number=4, operation_id="op-4", service="save", stage="complete",
                  reason="completed", folder_path="Clients/Alpha", completion_confirmed=True)
    ambiguous = build_lifecycle_body(lifecycle=_VerifiedTelegramLifecycle(
        **common, classification_outcome="ambiguous",
    ))
    unassigned = build_lifecycle_body(lifecycle=_VerifiedTelegramLifecycle(
        **{**common, "folder_path": "_unassigned"}, classification_outcome="unassigned",
    ))
    old_certificate = build_lifecycle_body(lifecycle=_VerifiedTelegramLifecycle(**common))
    invalid = build_lifecycle_body(lifecycle=_VerifiedTelegramLifecycle(
        **common, classification_outcome="raw user content",
    ))

    assert "The closest match was selected." in ambiguous
    assert "The folder could not be determined confidently; saved to _unassigned." in unassigned
    assert "The closest match was selected." not in old_certificate
    assert "raw user content" not in invalid


def test_terminal_error_has_safe_fields_exhaustion_advice_and_confirmed_only_billing_wording():
    common = dict(
        dialog_number=7, operation_id="op-error", service="google_drive",
        stage="telegram.audio.42", code="provider_error", occurred_at=NOW,
        sanitized_message="Stage execution failed (RuntimeError)", failures=3,
    )
    generic = format_terminal_error(**common, category="rate_limited")
    assert generic.startswith("Dialog #7: Error in google_drive/telegram.audio.42")
    assert "code=provider_error, category=rate_limited" in generic
    assert "2026-10-22T00:00:00+00:00" in generic
    assert "operation_id=op-error" in generic
    assert "temporarily limiting requests" in generic
    assert "billing" not in generic.casefold() and "funds" not in generic.casefold()
    assert "Automatic retries have stopped" in generic
    assert "administrator" in generic and "logs" in generic
    assert "RuntimeError" not in generic

    billing = format_terminal_error(**{**common, "failures": 1}, category="billing")
    gemini_billing = format_terminal_error(
        **{**common, "service": "gemini", "failures": 1}, category="billing",
    )
    quota = format_terminal_error(**{**common, "failures": 1}, category="quota")
    assert "confirmed a billing issue" in billing
    assert "confirmed a quota limit" in quota
    assert "Check provider billing" in billing
    assert "Check provider billing" not in quota
    assert "Prepay credits" not in billing
    assert "Prepay credits" in gemini_billing
    assert "https://aistudio.google.com/billing" in gemini_billing


def test_terminal_error_drops_untrusted_error_text_and_malformed_identifiers():
    message = format_terminal_error(
        dialog_number=None, operation_id="op-1?token=secret", service="https://evil.test",
        stage="save", code="bad code", category="provider_error",
        sanitized_message="raw provider body https://secret.test/token?key=secret",
        occurred_at="not a timestamp", failures=1,
    )
    assert "Dialog #" not in message
    assert "https://" not in message and "secret" not in message
    assert "service=unknown" not in message
    assert "Error in unknown/save" in message
    assert "code=operation_failed" in message
    assert "operation_id=unavailable" in message
    assert "Occurred: unavailable" in message


@pytest.mark.parametrize(
    "outcome,folder_path,proof_ok,expected_confirmed,expected_outcome",
    [
        ("ambiguous", ("Clients", "Alpha"), True, True, "ambiguous"),
        ("unassigned", ("_unassigned",), True, True, "unassigned"),
        ("unassigned", ("Clients", "Alpha"), True, True, None),
        (None, ("Clients", "Alpha"), True, True, None),
        ("ambiguous", ("Clients", "Alpha"), False, False, None),
    ],
)
def test_terminal_helper_extracts_only_verified_path_consistent_outcome(
    monkeypatch, outcome, folder_path, proof_ok, expected_confirmed, expected_outcome,
):
    terminal = importlib.import_module(
        "services.connectors.telegram.app.terminal_notifications"
    )
    operation = SimpleNamespace(
        state=SimpleNamespace(value="completed"), selected_folder_path=folder_path,
    )
    draft = {"completion_certificate": {"classification_outcome": outcome}}
    monkeypatch.setattr(
        terminal, "_load_completion_certificate",
        lambda client, operation, operation_id, draft: {"verified": True},
    )
    monkeypatch.setattr(
        terminal, "_completion_receipt_matches_index",
        lambda client, certificate: proof_ok,
    )

    confirmed, selected_path, extracted_outcome = terminal._verified_completion_details(
        object(), operation, "operation-1", draft,
    )

    assert confirmed is expected_confirmed
    if expected_confirmed:
        assert selected_path == "/".join(folder_path)
    else:
        assert selected_path is None
    assert extracted_outcome == expected_outcome


def test_slow_sync_reservation_does_not_block_event_loop_heartbeat():
    started = threading.Event()

    class SlowRepository:
        def reserve_notification_attempt(self, *args, **kwargs):
            started.set()
            time.sleep(0.12)
            return {"reserved": False, "state": "busy"}

    publisher = TelegramNotificationPublisher(SlowRepository(), _Telegram())
    beats = []

    async def run():
        async def beat():
            await asyncio.sleep(0.02)
            beats.append("alive")

        delivery = asyncio.create_task(publisher.publish("notify"))
        heartbeat = asyncio.create_task(beat())
        assert await delivery == "busy"
        await heartbeat

    _run(run())
    assert started.is_set()
    assert beats == ["alive"]


def test_ack_is_immutable_and_progress_send_then_edit_adds_no_conversation_turn(firestore_fake):
    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    accepted = _accepted(repo)
    row = repo.get_notification(accepted.notification_id)
    assert row["kind"] == "draft_ack"
    assert row["chat_id"] == -10042 and row["thread_id"] == 5
    assert row["message_id"] is None and row["send_attempts_started"] == 0
    assert row["body"] == "Dialog #1: Message received"

    clock, telegram = _Clock(), _Telegram()
    publisher = TelegramNotificationPublisher(repo, telegram, clock=clock)
    assert _run(publisher.publish(accepted.notification_id)) == "delivered"
    row = repo.get_notification(accepted.notification_id)
    assert row["message_id"] == 901 and row["send_attempts_started"] == 1
    assert telegram.sent == [(-10042, row["body"], 5)]
    with pytest.raises(ValueError, match="delivered draft acknowledgement"):
        repo.revise_notification(accepted.notification_id, body="updated control text", now=clock())

    progress_id = "progress-edit"
    repo.create_notification(progress_id, context_id=accepted.context_id, kind="progress",
                             body="working", dialog_number=1, now=clock())
    assert _run(publisher.publish(progress_id)) == "delivered"
    repo.revise_notification(progress_id, body="updated progress", now=clock())
    assert _run(publisher.publish(progress_id)) == "delivered"
    assert telegram.sent == [(-10042, row["body"], 5), (-10042, "working", 5)]
    assert telegram.edited == [(-10042, 901, "updated progress")]
    assert len(repo.list_messages(accepted.context_id, accepted.draft_id)) == 1


def test_create_notification_validates_kind_and_derives_persisted_target(firestore_fake):
    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    accepted = _accepted(repo)
    for kind in ("queued", "switch_notice", "progress", "final", "error"):
        row = repo.create_notification(f"n-{kind}", context_id=accepted.context_id,
                                       kind=kind, body="trusted", now=NOW)
        assert (row["chat_id"], row["thread_id"]) == (-10042, 5)
    linked = repo.create_notification("n-linked", context_id=accepted.context_id,
                                      kind="queued", body="queued", dialog_number=4,
                                      operation_id="save-4", now=NOW)
    assert (linked["dialog_number"], linked["operation_id"], linked["desired_status_revision"]) == (4, "save-4", 1)
    with pytest.raises(ValueError):
        repo.create_notification("n-invalid", context_id=accepted.context_id,
                                 kind="unknown", body="trusted", now=NOW)
    with pytest.raises(TypeError):
        repo.create_notification("n-target", context_id=accepted.context_id,
                                 kind="queued", body="trusted", chat_id=7, now=NOW)
    with pytest.raises(TypeError):
        repo.create_notification("n-body", context_id=accepted.context_id,
                                 kind="queued", now=NOW)


def test_terminal_notification_reuses_only_matching_linked_final_ledger(firestore_fake, monkeypatch):
    from types import SimpleNamespace
    from services.connectors.telegram.app.terminal_notifications import (
        _linked_final_notification_id,
    )

    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    accepted = _accepted(repo, update=902)
    terminal = importlib.import_module(
        "services.connectors.telegram.app.terminal_notifications"
    )
    monkeypatch.setattr("core.integrations.firestore.client.get_client", lambda: client)
    operation_id = "save-linked"
    operation_ref = (client.collection("telegram_sessions").document(accepted.context_id)
                     .collection("operations").document(operation_id))
    record = SimpleNamespace(context_id=accepted.context_id, operation_id=operation_id)

    repo.create_notification(
        "canonical-final", context_id=accepted.context_id, kind="final", body="Saved.",
        operation_id=operation_id, now=NOW,
    )
    operation_ref.set({"notification_id": "canonical-final"})
    final_ref = client.collection("notification_outbox").document("canonical-final")
    for transient_state in ("pending", "retry", "sending", "editing"):
        final_ref.update({"state": transient_state})
        assert terminal.update_terminal_notification(record) == "canonical-final"
        assert _linked_final_notification_id(client, record) == "canonical-final"

    repo.create_notification(
        "progress-pointer", context_id=accepted.context_id, kind="progress", body="Working.",
        operation_id=operation_id, now=NOW,
    )
    operation_ref.set({"notification_id": "progress-pointer"})
    assert _linked_final_notification_id(client, record) is None

    repo.create_notification(
        "foreign-final", context_id=accepted.context_id, kind="final", body="Other save.",
        operation_id="other-save", now=NOW,
    )
    operation_ref.set({"notification_id": "foreign-final"})
    assert _linked_final_notification_id(client, record) is None


def test_unknown_send_attempt_budget_survives_replay_and_becomes_delivery_unknown(firestore_fake):
    repo = TelegramRepository(firestore_fake.client_class())
    accepted = _accepted(repo)
    clock, telegram = _Clock(), _Telegram(failures=3)

    def publisher_after_restart():
        return TelegramNotificationPublisher(repo, telegram, clock=clock)

    assert _run(publisher_after_restart().publish(accepted.notification_id)) == "retry"
    clock.advance(1)
    assert _run(publisher_after_restart().publish(accepted.notification_id)) == "retry"
    clock.advance(2)
    assert _run(publisher_after_restart().publish(accepted.notification_id)) == "delivery_unknown"
    assert _run(publisher_after_restart().publish(accepted.notification_id)) == "delivery_unknown"
    row = repo.get_notification(accepted.notification_id)
    assert row["send_attempts_started"] == 3
    assert row["state"] == "delivery_unknown"
    assert len(telegram.sent) == 3


def test_telegram_timeout_error_is_unknown_and_consumes_send_budget(firestore_fake):
    repo = TelegramRepository(firestore_fake.client_class())
    accepted = _accepted(repo)
    clock, telegram = _Clock(), _Telegram(failures=1, failure_category="timeout")
    publisher = TelegramNotificationPublisher(repo, telegram, clock=clock)
    assert _run(publisher.publish(accepted.notification_id)) == "retry"
    row = repo.get_notification(accepted.notification_id)
    assert row["last_outcome"] == "unknown"
    assert row["unknown_send_outcome"] is True
    assert row["send_attempts_started"] == 1


def test_definitive_send_rejection_fails_without_retrying(firestore_fake):
    repo = TelegramRepository(firestore_fake.client_class())
    accepted = _accepted(repo)
    telegram = _Telegram(failures=1, failure_category="forbidden")
    publisher = TelegramNotificationPublisher(repo, telegram, clock=_Clock())

    assert _run(publisher.publish(accepted.notification_id)) == "failed"
    row = repo.get_notification(accepted.notification_id)
    assert row["state"] == "failed"
    assert row["send_attempts_started"] == 1
    assert row["last_outcome"] == "rejected"
    assert row["unknown_send_outcome"] is False

    assert _run(publisher.publish(accepted.notification_id)) == "failed"
    assert len(telegram.sent) == 1


def test_rate_limited_send_honors_retry_after_and_never_exceeds_three_attempts(firestore_fake):
    repo = TelegramRepository(firestore_fake.client_class())
    accepted = _accepted(repo)
    clock = _Clock()
    telegram = _Telegram(failures=3, failure_category="rate_limited", retry_after=7)
    publisher = TelegramNotificationPublisher(repo, telegram, clock=clock)

    assert _run(publisher.publish(accepted.notification_id)) == "retry"
    row = repo.get_notification(accepted.notification_id)
    assert row["last_outcome"] == "rate_limited"
    assert row["retry_not_before"] == NOW + timedelta(seconds=7)
    repo.revise_notification(accepted.notification_id, body="Dialog #1: Updated status", now=clock())
    assert _run(publisher.publish(accepted.notification_id)) == "busy"
    assert len(telegram.sent) == 1

    clock.advance(7)
    assert _run(publisher.publish(accepted.notification_id)) == "retry"
    clock.advance(7)
    assert _run(publisher.publish(accepted.notification_id)) == "failed"
    assert _run(publisher.publish(accepted.notification_id)) == "failed"
    row = repo.get_notification(accepted.notification_id)
    assert row["send_attempts_started"] == 3
    assert row["state"] == "failed"
    assert len(telegram.sent) == 3


def test_unknown_send_followed_by_rate_limits_keeps_delivery_unknown_semantics(firestore_fake):
    class MixedFailures(_Telegram):
        def __init__(self):
            super().__init__()
            self.outcomes = ["timeout", "rate_limited", "rate_limited"]

        async def send_message(self, chat_id, text, *, message_thread_id=None):
            self.sent.append((chat_id, text, message_thread_id))
            category = self.outcomes.pop(0)
            raise TelegramAPIError(category, retry_after=7 if category == "rate_limited" else None)

    repo = TelegramRepository(firestore_fake.client_class())
    accepted = _accepted(repo)
    clock, telegram = _Clock(), MixedFailures()
    publisher = TelegramNotificationPublisher(repo, telegram, clock=clock)

    assert _run(publisher.publish(accepted.notification_id)) == "retry"
    clock.advance(1)
    assert _run(publisher.publish(accepted.notification_id)) == "retry"
    assert _run(publisher.publish(accepted.notification_id)) == "busy"
    clock.advance(7)
    assert _run(publisher.publish(accepted.notification_id)) == "delivery_unknown"
    row = repo.get_notification(accepted.notification_id)
    assert row["send_attempts_started"] == 3
    assert row["unknown_send_outcome"] is True
    assert row["last_outcome"] == "rate_limited"
    assert len(telegram.sent) == 3


def test_stale_edit_rate_limit_waits_and_reuses_the_same_message(firestore_fake):
    class StaleEditTelegram(_Telegram):
        def __init__(self, repo, notification_id):
            super().__init__()
            self.repo = repo
            self.notification_id = notification_id
            self.once = True

        async def edit_message_text(self, chat_id, message_id, text):
            self.edited.append((chat_id, message_id, text))
            if self.once:
                self.once = False
                self.repo.revise_notification(self.notification_id, body="new body", now=clock())
                raise TelegramAPIError("rate_limited", retry_after=9)
            return {"message_id": message_id}

    repo = TelegramRepository(firestore_fake.client_class())
    accepted = _accepted(repo)
    notification_id = "stale-edit-rate-limit"
    repo.create_notification(notification_id, context_id=accepted.context_id, kind="progress",
                             body="initial", operation_id="edit-op", now=NOW)
    clock = _Clock()
    telegram = StaleEditTelegram(repo, notification_id)
    publisher = TelegramNotificationPublisher(repo, telegram, clock=clock)
    assert _run(publisher.publish(notification_id)) == "delivered"
    repo.revise_notification(notification_id, body="old revision", now=clock())

    assert _run(publisher.publish(notification_id)) == "pending"
    row = repo.get_notification(notification_id)
    assert row["last_outcome"] == "rate_limited"
    assert row["retry_not_before"] == clock() + timedelta(seconds=9)
    assert row["edit_attempts_started"] == 1
    assert _run(publisher.publish(notification_id)) == "busy"
    assert len(telegram.edited) == 1
    clock.advance(9)
    assert _run(publisher.publish(notification_id)) == "delivered"
    row = repo.get_notification(notification_id)
    assert row["edit_attempts_started"] == 2
    assert row["message_id"] == 901
    assert [edit[1] for edit in telegram.edited] == [901, 901]


@pytest.mark.parametrize("retry_after, expected", [
    (90_000, NOW + timedelta(seconds=90_000)),
    (10**100, datetime.max.replace(tzinfo=timezone.utc)),
])
def test_rate_limited_retry_deadline_does_not_truncate_or_overflow(firestore_fake, retry_after, expected):
    repo = TelegramRepository(firestore_fake.client_class())
    accepted = _accepted(repo)
    reservation = repo.reserve_notification_attempt(
        accepted.notification_id, attempt_id="long-retry", now=NOW,
    )

    settled = repo.settle_notification_attempt(
        accepted.notification_id, attempt_id="long-retry",
        revision=reservation["attempt_revision"], outcome="rate_limited",
        retry_after=retry_after, now=NOW,
    )

    assert settled["retry_not_before"] == expected


def test_default_thread_zero_is_persisted_as_no_thread_for_ack_and_new_notice(firestore_fake):
    repo = TelegramRepository(firestore_fake.client_class())
    accepted = repo.accept_forward(
        1, chat_id=-10042, thread_id=0,
        message={"message_id": 11, "text": "source"}, now=NOW,
    )
    ack = repo.get_notification(accepted.notification_id)
    assert ack["thread_id"] is None
    notice = repo.create_notification("default-thread", context_id=accepted.context_id,
                                      kind="switch_notice", body="switched", now=NOW)
    assert notice["thread_id"] is None
    clock, telegram = _Clock(), _Telegram()
    publisher = TelegramNotificationPublisher(repo, telegram, clock=clock)
    assert _run(publisher.publish("default-thread")) == "delivered"
    assert telegram.sent == [(-10042, "switched", None)]


def test_numbered_switch_notice_uses_the_durable_ledger_and_server_authored_body(firestore_fake):
    repo = TelegramRepository(firestore_fake.client_class())
    accepted = _accepted(repo)
    telegram = _Telegram()
    publisher = TelegramNotificationPublisher(repo, telegram, clock=_Clock())

    result = _run(publisher.send_switch_notice(
        context_id=accepted.context_id,
        operation_id="switch-operation",
        notification_id="switch-notification",
        target_doc_id="doc-target",
        dialog_number=12,
    ))

    assert result == "confirmed"
    row = repo.get_notification("switch-notification")
    assert row["kind"] == "switch_notice"
    assert row["operation_id"] == "switch-operation"
    assert row["dialog_number"] == 12
    assert row["body"] == "Starting to process dialog #12."
    assert row["state"] == "delivered"
    assert telegram.sent == [(-10042, "Starting to process dialog #12.", 5)]


@pytest.mark.parametrize("terminal", ["failed", "delivery_unknown", "delivered"])
def test_persisted_switch_notice_state_is_monotonic(firestore_fake, terminal):
    from core.integrations.firestore.operation_queue import claim_head
    from services.connectors.telegram.commands import parse_command
    from services.connectors.telegram.repository import SESSIONS

    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    accepted = repo.accept_command(
        1, chat_id=42, user_id=7, command=parse_command("open dialog 2"), now=NOW,
    )
    context_ref = client.collection(SESSIONS).document(accepted.context_id)
    lease = claim_head(client, context_ref, owner="switch-worker", now=NOW)
    assert lease is not None
    assert repo.persist_route(
        lease.operation_id,
        {"status": "ok", "kind": "pure_switch", "target_doc_id": "doc-2",
         "conversation_id": "conversation-2", "dialog_number": 2},
        context_id=accepted.context_id, owner=lease.owner, fence=lease.fence, now=NOW,
    ) is True
    set_state = lambda state: repo.set_switch_notice_state(
        accepted.context_id, lease.operation_id, state,
        owner=lease.owner, fence=lease.fence, now=NOW,
    )

    assert set_state("pending") is True
    assert set_state(terminal) is True
    assert set_state("pending") is False
    assert set_state("delivered" if terminal != "delivered" else "failed") is False
    assert repo.get_switch_notice_state(accepted.context_id, lease.operation_id) == terminal


def test_revision_keeps_active_attempt_and_stale_send_is_followed_by_edit(firestore_fake):
    repo = TelegramRepository(firestore_fake.client_class())
    accepted = _accepted(repo)
    reserved = repo.reserve_notification_attempt(accepted.notification_id, attempt_id="attempt-old", now=NOW)
    revised = repo.revise_notification(accepted.notification_id, body="new revision", now=NOW)
    assert revised["in_flight_attempt_id"] == "attempt-old"
    assert revised["revision"] == reserved["attempt_revision"] + 1
    busy = repo.reserve_notification_attempt(accepted.notification_id, attempt_id="attempt-racer", now=NOW)
    assert busy["reserved"] is False

    stale_success = repo.settle_notification_attempt(
        accepted.notification_id, attempt_id="attempt-old", revision=reserved["attempt_revision"],
        outcome="sent", message_id=111, now=NOW,
    )
    assert stale_success["settled"] is True
    assert stale_success["state"] == "pending"
    assert stale_success["message_id"] == 111
    next_attempt = repo.reserve_notification_attempt(
        accepted.notification_id, attempt_id="attempt-edit", now=NOW,
    )
    assert next_attempt["reserved"] is True and next_attempt["is_send"] is False
    assert next_attempt["message_id"] == 111 and next_attempt["body"] == "new revision"
    settled = repo.settle_notification_attempt(
        accepted.notification_id, attempt_id="attempt-edit", revision=next_attempt["attempt_revision"],
        outcome="edited", now=NOW,
    )
    assert settled["state"] == "delivered"
    assert repo.get_notification(accepted.notification_id)["send_attempts_started"] == 1


def test_stale_rejection_does_not_fail_newer_revision(firestore_fake):
    repo = TelegramRepository(firestore_fake.client_class())
    accepted = _accepted(repo)
    reserved = repo.reserve_notification_attempt(accepted.notification_id, attempt_id="old-reject", now=NOW)
    repo.revise_notification(accepted.notification_id, body="new desired body", now=NOW)
    result = repo.settle_notification_attempt(
        accepted.notification_id, attempt_id="old-reject", revision=reserved["attempt_revision"],
        outcome="rejected", now=NOW,
    )
    assert result["state"] == "pending"
    assert result["revision"] == reserved["attempt_revision"] + 1
    retry = repo.reserve_notification_attempt(accepted.notification_id, attempt_id="new-send", now=NOW)
    assert retry["reserved"] is True
    assert retry["body"] == "new desired body"


def test_terminal_unknown_delivery_is_recorded_on_operation(firestore_fake):
    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    accepted = _accepted(repo)
    operation_ref = client.collection("dialog_operations").document("save-unknown-notice")
    operation_ref.set({"operation_id": "save-unknown-notice", "state": "completed"})
    repo.create_notification("unknown-final", context_id=accepted.context_id, kind="final",
                             body="saved", operation_id="save-unknown-notice", now=NOW)
    clock = _Clock()
    telegram = _Telegram(failures=3)
    publisher = TelegramNotificationPublisher(repo, telegram, clock=clock)
    assert _run(publisher.publish("unknown-final")) == "retry"
    clock.advance(1)
    assert _run(publisher.publish("unknown-final")) == "retry"
    clock.advance(2)
    assert _run(publisher.publish("unknown-final")) == "delivery_unknown"
    operation = operation_ref.get().to_dict()
    assert operation["state"] == "completed"
    assert operation["final_delivery_state"] == "delivery_unknown"
    assert len(telegram.sent) == 3


def test_terminal_notification_delivery_is_durable_and_suppresses_stale_progress(firestore_fake):
    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    accepted = _accepted(repo)
    operation_ref = client.collection("dialog_operations").document("save-final")
    operation_ref.set({"operation_id": "save-final", "state": "completed"})
    final = repo.create_notification("final-save", context_id=accepted.context_id, kind="final",
                                     body="saved", operation_id="save-final", now=NOW)
    assert repo.create_notification("final-save", context_id=accepted.context_id, kind="final",
                                    body="saved", operation_id="save-final", now=NOW) == final
    telegram = _Telegram()
    publisher = TelegramNotificationPublisher(repo, telegram, clock=_Clock())
    assert _run(publisher.publish("final-save")) == "delivered"
    operation = operation_ref.get().to_dict()
    assert operation["state"] == "completed"
    assert operation["final_delivery_state"] == "delivered"

    repo.create_notification("later-error", context_id=accepted.context_id, kind="error",
                             body="later failure", operation_id="save-final", now=NOW)
    failing_publisher = TelegramNotificationPublisher(
        repo, _Telegram(failures=1, failure_category="forbidden"), clock=_Clock(),
    )
    assert _run(failing_publisher.publish("later-error")) == "failed"
    operation = operation_ref.get().to_dict()
    assert operation["state"] == "completed"
    assert operation["final_delivery_state"] == "delivered"

    progress = repo.create_notification("stale-progress", context_id=accepted.context_id,
                                        kind="progress", body="still working",
                                        operation_id="save-final", now=NOW)
    assert _run(publisher.publish(progress["notification_id"])) == "superseded"
    assert len(telegram.sent) == 1
    assert repo.get_notification("stale-progress")["send_attempts_started"] == 0
    with pytest.raises(ValueError, match="terminal notification"):
        repo.revise_notification("final-save", body="stale final", now=NOW)


def test_terminal_notification_failure_is_recorded_without_changing_operation_state(firestore_fake):
    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    accepted = _accepted(repo)
    operation_ref = client.collection("dialog_operations").document("save-failed-notice")
    operation_ref.set({"operation_id": "save-failed-notice", "state": "completed"})
    repo.create_notification("failed-final", context_id=accepted.context_id, kind="error",
                             body="save error", operation_id="save-failed-notice", now=NOW)
    publisher = TelegramNotificationPublisher(
        repo, _Telegram(failures=1, failure_category="forbidden"), clock=_Clock(),
    )
    assert _run(publisher.publish("failed-final")) == "failed"
    operation = operation_ref.get().to_dict()
    assert operation["state"] == "completed"
    assert operation["final_delivery_state"] == "failed"


def test_terminal_ack_failure_keeps_collecting_draft_and_sealed_identity(firestore_fake):
    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    collecting = _accepted(repo, 1)
    for i in range(3):
        attempt_id = f"collecting-{i}"
        reserved = repo.reserve_notification_attempt(collecting.notification_id, attempt_id=attempt_id,
                                                     now=NOW + timedelta(seconds=i * 3))
        result = repo.settle_notification_attempt(collecting.notification_id, attempt_id=attempt_id,
                                                  revision=reserved["attempt_revision"], outcome="unknown",
                                                  now=NOW + timedelta(seconds=i * 3))
    assert result["state"] == "delivery_unknown"
    context = repo.get_context(collecting.context_id)
    assert context["active_draft_id"] == collecting.draft_id
    assert context["lock_owner"] is None
    assert repo.get_draft(collecting.context_id, collecting.draft_id)["state"] == "collecting"

    sealed = repo.accept_forward(
        2, chat_id=-10043, thread_id=5,
        message={"message_id": 12, "text": "sealed source"}, now=NOW,
    )
    draft_ref = client.collection("telegram_sessions").document(sealed.context_id).collection("drafts").document(sealed.draft_id)
    original = {**repo.get_draft(sealed.context_id, sealed.draft_id),
                "state": "processing", "sealed_at": NOW, "manifest_ref": "manifests/m1",
                "manifest_digest": "digest-1", "active_save_operation_id": "save-child"}
    draft_ref.set(original)
    parent_ref = client.collection("telegram_sessions").document(sealed.context_id).collection("operations").document("fifo-parent")
    parent_ref.set({"operation_id": "fifo-parent", "sequence": 1, "state": "pending",
                    "target_draft_id": sealed.draft_id, "save_operation_id": "save-child",
                    "next_operation_id": "independent-successor"})
    successor_ref = client.collection("telegram_sessions").document(sealed.context_id).collection("operations").document("independent-successor")
    successor_ref.set({"operation_id": "independent-successor", "sequence": 2, "state": "pending"})
    save_ref = client.collection("dialog_operations").document("save-child")
    save_ref.set({"operation_id": "save-child", "state": "running",
                  "lease_owner": "worker-1", "lease_expires_at": NOW + timedelta(minutes=1)})
    ctx_ref = client.collection("telegram_sessions").document(sealed.context_id)
    ctx_ref.update({"reserved_operation_id": "fifo-parent", "head_sequence": 1,
                    "active_draft_id": sealed.draft_id, "draft_status": "collecting"})
    for i in range(3):
        attempt_id = f"sealed-{i}"
        reserved = repo.reserve_notification_attempt(sealed.notification_id, attempt_id=attempt_id,
                                                     now=NOW + timedelta(seconds=i * 3))
        result = repo.settle_notification_attempt(sealed.notification_id, attempt_id=attempt_id,
                                                  revision=reserved["attempt_revision"], outcome="unknown",
                                                  now=NOW + timedelta(seconds=i * 3))
    failed = repo.get_draft(sealed.context_id, sealed.draft_id)
    assert result["ack_failure_action"]["dialog_number"] == 1
    assert result["ack_failure_action"]["operation_id"] == "save-child"
    assert result["ack_failure_action"]["parent_operation_id"] == "fifo-parent"
    assert result["ack_failure_action"]["queue_release_required"] is False
    assert failed["state"] == "failed"
    assert failed["manifest_ref"] == original["manifest_ref"]
    assert failed["manifest_digest"] == original["manifest_digest"]
    assert failed["ordered_message_ids"] == original["ordered_message_ids"]
    context = repo.get_context(sealed.context_id)
    assert context["reserved_operation_id"] == "independent-successor"
    assert context["head_sequence"] == 2
    assert context["active_draft_id"] is None
    parent = parent_ref.get().to_dict()
    assert parent["state"] == "failed"
    assert parent["failure_code"] == "ack_delivery_failed"
    assert parent["next_operation_id"] == "independent-successor"
    successor = successor_ref.get().to_dict()
    assert successor["state"] == "pending"
    assert successor["sequence"] == 2
    child = save_ref.get().to_dict()
    assert child["state"] == "failed"
    assert child["error"]["code"] == "ack_delivery_failed"
    assert child["lease_owner"] is None and child["lease_expires_at"] is None


def test_delivered_ack_cannot_be_revised_and_edit_failure_does_not_fail_sealed_draft(firestore_fake):
    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    accepted = _accepted(repo)
    reserved = repo.reserve_notification_attempt(accepted.notification_id,
                                                 attempt_id="ack-send", now=NOW)
    repo.settle_notification_attempt(accepted.notification_id, attempt_id="ack-send",
                                     revision=reserved["attempt_revision"], outcome="sent",
                                     message_id=901, now=NOW)
    with pytest.raises(ValueError, match="delivered draft acknowledgement"):
        repo.revise_notification(accepted.notification_id, body="new acknowledgement", now=NOW)

    draft_ref = client.collection("telegram_sessions").document(accepted.context_id).collection("drafts").document(accepted.draft_id)
    draft_ref.update({"state": "processing", "sealed_at": NOW,
                      "manifest_ref": "manifests/m1", "manifest_digest": "digest-1"})
    notification_ref = client.collection("notification_outbox").document(accepted.notification_id)
    notification_ref.update({"state": "editing", "revision": 2, "desired_status_revision": 2,
                             "edit_attempts_started": 3, "in_flight_attempt_id": "edit-3",
                             "attempt_revision": 2, "attempt_expires_at": NOW})
    result = repo.settle_notification_attempt(accepted.notification_id, attempt_id="edit-3",
                                              revision=2, outcome="rejected", now=NOW)
    assert result["state"] == "failed"
    assert repo.get_draft(accepted.context_id, accepted.draft_id)["state"] == "processing"


def test_sealed_ack_failure_preserves_newer_collecting_pointer_without_exact_parent_link(firestore_fake):
    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    accepted = repo.accept_forward(
        1, chat_id=-10045, thread_id=5,
        message={"message_id": 21, "text": "sealed source"}, now=NOW,
    )
    draft_ref = client.collection("telegram_sessions").document(accepted.context_id).collection("drafts").document(accepted.draft_id)
    draft = {**repo.get_draft(accepted.context_id, accepted.draft_id), "state": "processing",
             "sealed_at": NOW, "manifest_ref": "manifests/old", "manifest_digest": "old-digest",
             "active_save_operation_id": "save-child"}
    draft_ref.set(draft)
    context_ref = client.collection("telegram_sessions").document(accepted.context_id)
    context_ref.update({"active_draft_id": "newer-draft", "draft_status": "collecting",
                        "active_conversation_id": f"{accepted.context_id}-2",
                        "selected_dialog_number": 2, "selection_sequence": 2,
                        "reserved_operation_id": "unrelated-parent", "head_sequence": 8})
    client.collection("telegram_sessions").document(accepted.context_id).collection("drafts").document("newer-draft").set(
        {"draft_id": "newer-draft", "state": "collecting"}
    )
    for i in range(3):
        attempt_id = f"unlinked-{i}"
        moment = NOW + timedelta(seconds=i * 3)
        reserved = repo.reserve_notification_attempt(accepted.notification_id, attempt_id=attempt_id, now=moment)
        result = repo.settle_notification_attempt(accepted.notification_id, attempt_id=attempt_id,
                                                  revision=reserved["attempt_revision"], outcome="unknown", now=moment)
    state = repo.get_context(accepted.context_id)
    assert result["ack_failure_action"]["queue_release_required"] is True
    assert state["active_draft_id"] == "newer-draft"
    assert state["draft_status"] == "collecting"
    assert state["active_conversation_id"] == f"{accepted.context_id}-2"
    assert state["selected_dialog_number"] == 2
    assert state["selection_sequence"] == 2
    assert state["reserved_operation_id"] == "unrelated-parent"
    assert repo.get_draft(accepted.context_id, accepted.draft_id)["manifest_digest"] == "old-digest"
