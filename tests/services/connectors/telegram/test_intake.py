"""Contract tests for Telegram intake before its production modules exist.

The test-facing boundary is ``TelegramRepository(client=...)`` in
``services.connectors.telegram.repository``. It owns Firestore transactions for
receipt deduplication, draft/message persistence, and operation reservations.
``TelegramIntake(repository, publisher, classifier, semantic_router)`` parses
updates and publishes the durable outbox item returned by the repository.
"""

from __future__ import annotations

import asyncio
import importlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from core.integrations.firestore.operation_queue import claim_head, complete_head
from services.connectors.telegram.commands import parse_command


pytestmark = pytest.mark.offline


NOW = datetime(2026, 10, 22, tzinfo=timezone.utc)


@pytest.mark.parametrize("command", [
    "save", "Сохрани", "guardar", "please save this", "do not save this",
    '"save"', "Сохрани это!", "保存してください",
])
def test_all_command_text_stays_freeform_for_structured_router(command):
    parsed = parse_command(command)

    assert parsed.kind == "freeform"
    assert parsed.text == command


def test_command_parser_preserves_complete_long_text():
    text = "Earlier context. " + ("detail " * 160) + "TAIL MUST SURVIVE"
    parsed = parse_command(text)
    assert parsed.text == text
    assert parsed.text.endswith("TAIL MUST SURVIVE")


def test_leading_text_and_long_burst_keep_source_ids_and_head_binding(firestore_fake):
    repository, client = _repository(firestore_fake)
    intake = _intake(repository, _Publisher(), clock=lambda: NOW)
    timestamp = int(NOW.timestamp())
    command = _command(101, "Мой диалог с мамой")
    command["message"]["date"] = timestamp
    intake.handle_update(command)
    results = []
    for index in range(100):
        forwarded = _forward(102 + index, message_id=202 + index)
        forwarded["message"]["date"] = timestamp
        results.append(intake.handle_update(forwarded))
    assert all(_value(result, "accepted") for result in results)
    context = repository.get_context("42_default")
    operation_id = context["reserved_operation_id"]
    binding = repository.get_ingress_binding("42_default", operation_id)
    assert binding["matched_message_ids"] == list(range(202, 302))
    receipt = repository.get_update_receipt("42_default", 102)
    group = _read_doc(
        client, f"telegram_sessions/42_default/receipt_groups/{receipt['receipt_group_id']}",
    )
    assert group["accepted_message_ids"] == list(range(102, 202))
    assert group["source_message_ids"] == list(range(202, 302))


def test_queued_text_does_not_take_admission_from_reserved_head(firestore_fake):
    repository, _ = _repository(firestore_fake)
    intake = _intake(repository, _Publisher(), clock=lambda: NOW)
    timestamp = int(NOW.timestamp())
    first = _command(201, "Мой диалог с мамой")
    first["message"].update(message_id=801, date=timestamp)
    intake.handle_update(first)
    first_source = _forward(202, message_id=802)
    first_source["message"]["date"] = timestamp
    intake.handle_update(first_source)
    second = _command(203, "a queued question")
    second["message"].update(message_id=900, date=timestamp)
    intake.handle_update(second)
    results = []
    for index in range(5):
        forwarded = _forward(204 + index, message_id=803 + index)
        forwarded["message"]["date"] = timestamp
        results.append(intake.handle_update(forwarded))
    assert all(_value(result, "accepted") for result in results)
    context = repository.get_context("42_default")
    binding = repository.get_ingress_binding("42_default", context["reserved_operation_id"])
    assert binding["candidate_message_id"] == 801
    assert binding["matched_message_ids"] == [802, 803, 804, 805, 806, 807]


def _repository(firestore_fake):
    repository_type = importlib.import_module(
        "services.connectors.telegram.repository"
    ).TelegramRepository
    client = firestore_fake.client_class()
    return repository_type(client=client), client


def _read_doc(client, path):
    parts = path.split("/")
    reference = client.collection(parts[0]).document(parts[1])
    for index in range(2, len(parts), 2):
        reference = reference.collection(parts[index]).document(parts[index + 1])
    snapshot = reference.get()
    return snapshot.to_dict() if snapshot.exists else None


def _paths_with_prefix(firestore_fake, prefix):
    return [path for path in firestore_fake.documents if path.startswith(prefix)]


def _direct_child_paths(firestore_fake, parent):
    prefix = parent.rstrip("/") + "/"
    return [
        path for path in firestore_fake.documents
        if path.startswith(prefix) and "/" not in path[len(prefix):]
    ]


def _message_path(context_id, draft_id, message_id):
    return f"telegram_sessions/{context_id}/drafts/{draft_id}/messages/{message_id}"


def _intake(repository, publisher, classifier=None, semantic_router=None, *, clock=None):
    intake_type = importlib.import_module(
        "services.connectors.telegram.intake"
    ).TelegramIntake
    return intake_type(
        repository=repository,
        publisher=publisher,
        classifier=classifier,
        semantic_router=semantic_router,
        clock=clock,
    )


def _forward(update_id: int, *, chat_id: int = 42, user_id: int = 7, message_id: int | None = None):
    return {
        "update_id": update_id,
        "message": {
            "message_id": message_id or update_id + 100,
            "date": 1_798_000_000 + update_id * 2,
            "chat": {"id": chat_id, "type": "private"},
            "from": {"id": user_id},
            "forward_origin": {"type": "user", "sender_user": {"id": 91}},
            "text": f"synthetic forwarded message {update_id}",
        },
    }


def _command(update_id: int, command: str):
    return {
        "update_id": update_id,
        "message": {
            "message_id": update_id + 100,
            "date": 1_798_000_000,
            "chat": {"id": 42, "type": "private"},
            "from": {"id": 7},
            "text": command,
        },
    }


def _value(record, name):
    return record[name] if isinstance(record, dict) else getattr(record, name)


class _Publisher:
    def __init__(self):
        self.outbox_ids = []
        self.failures = 0

    def publish(self, outbox_id):
        self.outbox_ids.append(outbox_id)
        if self.failures:
            self.failures -= 1
            raise RuntimeError("synthetic publish gap")


class _Classifier:
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        raise AssertionError("forwards must not be classified during intake")


def test_forward_atomically_persists_collecting_draft_message_and_draft_ack(
    firestore_fake, sdk_fakes,
):
    repository, client = _repository(firestore_fake)
    publisher = _Publisher()
    classifier = _Classifier()
    semantic_router = _Classifier()
    intake = _intake(repository, publisher, classifier, semantic_router)

    accepted = intake.handle_update(_forward(1))

    assert _value(accepted, "accepted") is True
    assert _value(accepted, "draft_id")
    assert _value(accepted, "dialog_number") > 0
    assert _value(accepted, "message_id") == 101
    assert _value(accepted, "notification_id")
    assert _value(accepted, "outbox_id")

    context_id = _value(accepted, "context_id")
    update_record = _read_doc(client, f"telegram_sessions/{context_id}/updates/1")
    draft = _read_doc(client, f"telegram_sessions/{context_id}/drafts/{_value(accepted, 'draft_id')}")
    draft_id = _value(accepted, "draft_id")
    message = _read_doc(client, _message_path(context_id, draft_id, 101))
    notification = _read_doc(client, f"notification_outbox/{_value(accepted, 'notification_id')}")
    outbox = _read_doc(client, f"task_outbox/{_value(accepted, 'outbox_id')}")
    assert update_record is not None
    assert draft["state"] == "collecting"
    assert draft["dialog_number"] == _value(accepted, "dialog_number")
    assert message["message_id"] == 101
    assert message["text"] == "synthetic forwarded message 1"
    assert notification["kind"] == "draft_ack"
    assert str(_value(accepted, "dialog_number")) in str(notification["body"])
    assert outbox["notification_id"] == _value(accepted, "notification_id")
    assert publisher.outbox_ids == [
        _value(accepted, "outbox_id"), *_value(accepted, "additional_outbox_ids"),
    ]
    context = repository.get_context(context_id)
    assert context["active_conversation_id"] == f"{context_id}-{_value(accepted, 'dialog_number')}"
    assert context["selected_dialog_number"] == _value(accepted, "dialog_number")
    assert context["selection_sequence"] == 1
    assert classifier.calls == []
    assert semantic_router.calls == []
    assert not any(event["boundary"] == "gemini" for event in sdk_fakes.trace)


def test_receipt_text_quiet_window_is_durable_and_stale_close_cannot_regress_latest_card(
    firestore_fake,
):
    repository, client = _repository(firestore_fake)
    publisher = _Publisher()
    current = [NOW]
    intake = _intake(repository, publisher, clock=lambda: current[0])

    first = intake.handle_update(_forward(50))
    current[0] = NOW.replace(microsecond=500_000)
    second_update = _forward(51)
    second_update["message"]["date"] = 1_798_000_000 + 50 * 2 + 1
    second = intake.handle_update(second_update)
    assert first.notification_id == second.notification_id
    assert first.outbox_id != second.outbox_id

    # The receipt binds the durable group even though the public result stays compact.
    first_receipt = repository.get_update_receipt(first.context_id, 50)
    group_id = first_receipt["receipt_group_id"]
    first_group = _read_doc(client, f"telegram_sessions/{first.context_id}/receipt_groups/{group_id}")
    assert first_group["accepted_message_ids"] == [50, 51]
    assert first_group["deadline_at"] == NOW + timedelta(seconds=1.5)

    # The deadline has passed, but the earlier close task has not run yet.
    current[0] = NOW + timedelta(seconds=1.6)
    third_update = _forward(52)
    third_update["message"]["date"] = 1_798_000_000 + 50 * 2 + 4
    third = intake.handle_update(third_update)
    assert third.notification_id != first.notification_id
    context = repository.get_context(first.context_id)
    assert context["pending_receipt_groups"] == 2

    current[0] = NOW + timedelta(seconds=3)
    assert repository.refresh_receipt_card(third.notification_id, now=current[0]) == (third.notification_id,)
    assert repository.get_context(first.context_id)["pending_receipt_groups"] == 1
    latest = repository.get_notification(third.notification_id)
    assert latest["reply_markup"] == {"inline_keyboard": []}
    assert "Thinking" in latest["body"]

    assert repository.refresh_receipt_card(first.notification_id, now=current[0]) == (
        first.notification_id, third.notification_id,
    )
    old_card = repository.get_notification(first.notification_id)
    assert "Received 2 forwarded message(s)" in old_card["body"]
    assert not old_card["reply_markup"]["inline_keyboard"]
    assert repository.get_context(first.context_id)["pending_receipt_groups"] == 0
    ready = repository.get_notification(third.notification_id)
    assert "3 messages received" in ready["body"]
    assert ready["reply_markup"]["inline_keyboard"][0][0]["text"] == "Save"
    token = ready["receipt_control_token"]
    revision = ready["revision"]
    repository.refresh_receipt_card(first.notification_id, now=current[0])
    replayed = repository.get_notification(third.notification_id)
    assert replayed["receipt_control_token"] == token
    assert replayed["revision"] == revision


def test_transport_timed_burst_reopens_its_closed_group_after_webhook_delay(firestore_fake):
    repository, client = _repository(firestore_fake)
    publisher = _Publisher()
    current = [NOW]
    intake = _intake(repository, publisher, clock=lambda: current[0])
    transport_second = 1_798_100_000

    first_update = _forward(54)
    first_update["message"]["date"] = transport_second
    first_update["message"]["forward_origin"]["date"] = transport_second - 100_000
    first = intake.handle_update(first_update)
    current[0] = NOW + timedelta(seconds=1.1)
    repository.refresh_receipt_card(first.notification_id, now=current[0])
    assert repository.get_context(first.context_id)["pending_receipt_groups"] == 0

    # Delivery took 9.7 seconds; the next member still identifies the same batch.
    current[0] = NOW + timedelta(seconds=9.7)
    second_update = _forward(55)
    second_update["message"]["date"] = transport_second + 1
    second = intake.handle_update(second_update)
    assert second.notification_id == first.notification_id
    assert repository.get_context(first.context_id)["pending_receipt_groups"] == 1
    assert repository.get_update_receipt(first.context_id, 55)["receipt_group_id"] == (
        repository.get_update_receipt(first.context_id, 54)["receipt_group_id"]
    )

    current[0] = NOW + timedelta(seconds=10.8)
    repository.refresh_receipt_card(first.notification_id, now=current[0])
    current[0] = NOW + timedelta(seconds=10.9)
    third_update = _forward(56)
    third_update["message"]["date"] = transport_second
    third = intake.handle_update(third_update)
    assert third.notification_id == first.notification_id
    current[0] = NOW + timedelta(seconds=12.0)
    repository.refresh_receipt_card(first.notification_id, now=current[0])
    current[0] = NOW + timedelta(seconds=12.1)
    fourth_update = _forward(57)
    fourth_update["message"]["date"] = transport_second + 1
    fourth = intake.handle_update(fourth_update)
    assert fourth.notification_id == first.notification_id

    current[0] = NOW + timedelta(seconds=13.2)
    repository.refresh_receipt_card(first.notification_id, now=current[0])
    ready = repository.get_notification(first.notification_id)
    group_id = repository.get_update_receipt(first.context_id, 54)["receipt_group_id"]
    group = _read_doc(client, f"telegram_sessions/{first.context_id}/receipt_groups/{group_id}")
    source = _read_doc(client, _message_path(first.context_id, first.draft_id, 154))
    assert group["accepted_message_ids"] == [54, 55, 56, 57]
    assert group["epoch"] == 4
    assert repository.get_context(first.context_id)["pending_receipt_groups"] == 0
    assert "4 messages received" in ready["body"]
    assert ready["reply_markup"]["inline_keyboard"][0][0]["text"] == "Save"
    assert source["original_date"] == datetime.fromtimestamp(
        transport_second - 100_000, tz=timezone.utc,
    )
    assert "transport_timestamp" not in source


def test_distinct_transport_bursts_and_legacy_timestamp_fallback(firestore_fake):
    repository, _ = _repository(firestore_fake)
    current = [NOW]
    intake = _intake(repository, _Publisher(), clock=lambda: current[0])
    first_update = _forward(58)
    first_update["message"]["date"] = 1_798_200_000
    first = intake.handle_update(first_update)
    current[0] = NOW + timedelta(milliseconds=100)
    distinct_update = _forward(59)
    distinct_update["message"]["date"] = 1_798_200_002
    distinct = intake.handle_update(distinct_update)
    assert distinct.notification_id != first.notification_id

    malformed_update = _forward(60)
    malformed_update["message"].pop("date")
    legacy = intake.handle_update(malformed_update)
    current[0] = NOW + timedelta(milliseconds=500)
    missing_update = _forward(61)
    missing_update["message"]["date"] = "not-a-timestamp"
    legacy_join = intake.handle_update(missing_update)
    assert legacy_join.notification_id == legacy.notification_id

    # When adjacent aliases point at two distinct groups, an intervening
    # timestamp creates its own group instead of bridging the existing groups.
    ambiguous_a = _forward(68, chat_id=44)
    ambiguous_a["message"]["date"] = 1_798_210_000
    group_a = intake.handle_update(ambiguous_a)
    ambiguous_b = _forward(69, chat_id=44)
    ambiguous_b["message"]["date"] = 1_798_210_004
    group_b = intake.handle_update(ambiguous_b)
    between_update = _forward(70, chat_id=44)
    between_update["message"]["date"] = 1_798_210_002
    between = intake.handle_update(between_update)
    ambiguous_update = _forward(71, chat_id=44)
    ambiguous_update["message"]["date"] = 1_798_210_003
    separate = intake.handle_update(ambiguous_update)
    assert len({group_a.notification_id, group_b.notification_id,
                between.notification_id, separate.notification_id}) == 4


def test_sliding_transport_gap_and_late_older_group_preserve_latest_card(firestore_fake):
    repository, client = _repository(firestore_fake)
    current = [NOW]
    intake = _intake(repository, _Publisher(), clock=lambda: current[0])
    base_second = 1_798_300_000

    first_a_update = _forward(62)
    first_a_update["message"]["date"] = base_second
    first_a = intake.handle_update(first_a_update)
    current[0] = NOW + timedelta(seconds=1.1)
    repository.refresh_receipt_card(first_a.notification_id, now=current[0])
    current[0] = NOW + timedelta(seconds=2)
    first_b_update = _forward(63)
    first_b_update["message"]["date"] = base_second + 5
    first_b = intake.handle_update(first_b_update)
    assert first_a.notification_id != first_b.notification_id

    # A(0), B(5), then a delayed A(0): the persisted timestamp lookup finds A
    # without regressing the latest-card pointer from B.
    current[0] = NOW + timedelta(seconds=3.1)
    repository.refresh_receipt_card(first_b.notification_id, now=current[0])
    ready_b = repository.get_notification(first_b.notification_id)
    assert ready_b["reply_markup"]["inline_keyboard"]
    current[0] = NOW + timedelta(seconds=9)
    delayed_a_update = _forward(64)
    delayed_a_update["message"]["date"] = base_second
    delayed_a = intake.handle_update(delayed_a_update)
    assert delayed_a.notification_id == first_a.notification_id
    assert delayed_a.additional_outbox_ids
    assert delayed_a.additional_outbox_ids[0] in intake.publisher.outbox_ids
    immediate_refresh = _read_doc(client, f"task_outbox/{delayed_a.additional_outbox_ids[0]}")
    assert immediate_refresh["notification_id"] == first_a.notification_id
    retire_task_id = next(
        outbox_id for outbox_id in delayed_a.additional_outbox_ids
        if _read_doc(client, f"task_outbox/{outbox_id}")["notification_id"] == first_b.notification_id
    )
    assert retire_task_id in intake.publisher.outbox_ids
    latest_before_close = repository.get_context(first_a.context_id)
    assert latest_before_close["latest_receipt_notification_id"] == first_b.notification_id
    assert latest_before_close["pending_receipt_groups"] == 1
    assert repository.get_notification(first_b.notification_id)["reply_markup"] == {
        "inline_keyboard": [],
    }

    current[0] = NOW + timedelta(seconds=10.1)
    repository.refresh_receipt_card(first_a.notification_id, now=current[0])
    latest_b = repository.get_notification(first_b.notification_id)
    assert latest_b["body"].startswith("Dialog #1: Ready - 3 messages received.")
    assert latest_b["reply_markup"]["inline_keyboard"][0][0]["text"] == "Save"

    # A continuous action spanning more than one timestamp second still groups
    # while each consecutive transport-second gap is at most one second.
    current[0] = NOW + timedelta(seconds=20)
    continuous_first = _forward(65, chat_id=43)
    continuous_first["message"]["date"] = base_second + 20
    one = intake.handle_update(continuous_first)
    continuous_second = _forward(66, chat_id=43)
    continuous_second["message"]["date"] = base_second + 21
    two = intake.handle_update(continuous_second)
    continuous_third = _forward(67, chat_id=43)
    continuous_third["message"]["date"] = base_second + 22
    three = intake.handle_update(continuous_third)
    assert one.notification_id == two.notification_id == three.notification_id


def test_transport_alias_filters_other_actor_before_resolving_adjacent_group(
    firestore_fake,
):
    repository, client = _repository(firestore_fake)
    current = [NOW]
    intake = _intake(repository, _Publisher(), clock=lambda: current[0])
    transport_second = 1_798_400_000
    first_update = _forward(72)
    first_update["message"]["date"] = transport_second
    first = intake.handle_update(first_update)
    first_group_id = repository.get_update_receipt(
        first.context_id, 72,
    )["receipt_group_id"]
    foreign_group_id = "receipt-foreign-owner"
    client.collection("telegram_sessions").document(first.context_id).collection(
        "receipt_groups").document(foreign_group_id).set({
            "group_id": foreign_group_id, "context_id": first.context_id,
            "draft_id": first.draft_id, "owner_user_id": 8,
            "media_group_id": None, "state": "closed",
            "transport_start_timestamp": transport_second + 1,
            "transport_end_timestamp": transport_second + 1,
        })
    client.collection("telegram_sessions").document(first.context_id).collection(
        "transport_receipt_seconds").document(str(transport_second + 1)).set({
            "transport_timestamp": transport_second + 1,
            "group_ids": [foreign_group_id],
        })

    second_update = _forward(73)
    second_update["message"]["date"] = transport_second + 1
    second = intake.handle_update(second_update)
    assert second.notification_id == first.notification_id
    assert repository.get_update_receipt(first.context_id, 73)["receipt_group_id"] == first_group_id
    alias = client.collection("telegram_sessions").document(first.context_id).collection(
        "transport_receipt_seconds").document(str(transport_second + 1)).get().to_dict()
    assert set(alias["group_ids"]) == {foreign_group_id, first_group_id}


def test_repository_discards_out_of_range_transport_metadata(firestore_fake):
    repository, client = _repository(firestore_fake)
    accepted = repository.accept_forward(
        74, chat_id=42, user_id=7,
        message={"message_id": 174, "text": "preserved source bytes"},
        transport_timestamp=10**100, now=NOW,
    )
    context = repository.get_context(accepted.context_id)
    group_id = repository.get_update_receipt(accepted.context_id, 74)["receipt_group_id"]
    group = _read_doc(
        client, f"telegram_sessions/{accepted.context_id}/receipt_groups/{group_id}",
    )
    source = _read_doc(client, _message_path(accepted.context_id, accepted.draft_id, 174))
    assert context["active_draft_id"] == accepted.draft_id
    assert "transport_start_timestamp" not in group
    assert source["text"] == "preserved source bytes"


def test_native_identity_reopens_closed_group_and_keeps_expired_open_epoch(firestore_fake):
    repository, _client = _repository(firestore_fake)
    current = [NOW]
    intake = _intake(repository, _Publisher(), clock=lambda: current[0])

    legacy_update = _forward(75)
    legacy_update["message"].pop("date")
    legacy_update["message"]["media_group_id"] = "legacy-native"
    legacy = intake.handle_update(legacy_update)
    current[0] = NOW + timedelta(seconds=1.1)
    repository.refresh_receipt_card(legacy.notification_id, now=current[0])
    current[0] = NOW + timedelta(seconds=9)
    next_legacy = _forward(76)
    next_legacy["message"].pop("date")
    next_legacy["message"]["media_group_id"] = "legacy-native"
    legacy_followup = intake.handle_update(next_legacy)
    assert legacy_followup.notification_id != legacy.notification_id

    timed_current = [NOW]
    timed_intake = _intake(repository, _Publisher(), clock=lambda: timed_current[0])
    timed_update = _forward(77, chat_id=43)
    timed_update["message"]["media_group_id"] = "timed-native"
    timed = timed_intake.handle_update(timed_update)
    timed_current[0] = NOW + timedelta(seconds=1.1)
    repository.refresh_receipt_card(timed.notification_id, now=timed_current[0])
    timed_context = repository.get_context(timed.context_id)
    timed_group_id = repository.get_update_receipt(timed.context_id, 77)["receipt_group_id"]
    timed_group = _read_doc(
        _client, f"telegram_sessions/{timed.context_id}/receipt_groups/{timed_group_id}",
    )
    assert timed_group["state"] == "closed"
    assert timed_context["pending_receipt_groups"] == 0
    timed_current[0] = NOW + timedelta(seconds=9)
    delayed_timed_update = _forward(78, chat_id=43)
    delayed_timed_update["message"]["media_group_id"] = "timed-native"
    delayed_timed_update["message"]["date"] = timed_update["message"]["date"] + 5
    delayed_timed = timed_intake.handle_update(delayed_timed_update)
    assert delayed_timed.notification_id == timed.notification_id
    assert repository.get_context(timed.context_id)["pending_receipt_groups"] == 1

    expired_current = [NOW]
    expired_intake = _intake(repository, _Publisher(), clock=lambda: expired_current[0])
    expired_update = _forward(79, chat_id=44)
    expired_update["message"]["media_group_id"] = "expired-native"
    expired = expired_intake.handle_update(expired_update)
    expired_current[0] = NOW + timedelta(seconds=2)
    expired_followup_update = _forward(80, chat_id=44)
    expired_followup_update["message"]["media_group_id"] = "expired-native"
    expired_followup_update["message"]["date"] = expired_update["message"]["date"] + 5
    expired_followup = expired_intake.handle_update(expired_followup_update)
    assert expired_followup.notification_id == expired.notification_id
    expired_group_id = repository.get_update_receipt(
        expired.context_id, 79,
    )["receipt_group_id"]
    expired_group = _read_doc(
        _client, f"telegram_sessions/{expired.context_id}/receipt_groups/{expired_group_id}",
    )
    assert expired_group["state"] == "open"
    assert expired_group["epoch"] == 1
    assert repository.get_context(expired.context_id)["pending_receipt_groups"] == 1


def test_unsupported_first_joins_supported_text_card_and_all_unsupported_never_get_save(
    firestore_fake,
):
    repository, client = _repository(firestore_fake)
    current = [NOW]
    intake = _intake(repository, _Publisher(), clock=lambda: current[0])

    unsupported = _forward(60)
    unsupported["message"].pop("text")
    unsupported["message"]["photo"] = [{"file_id": "photo-60"}]
    unsupported["message"]["media_group_id"] = "native-photo-group"
    skipped = intake.handle_update(unsupported)
    assert skipped.draft_id is None
    current[0] = NOW + timedelta(seconds=1.1)
    repository.refresh_receipt_card(skipped.notification_id, now=current[0])
    first_group_id = repository.get_update_receipt(skipped.context_id, 60)["receipt_group_id"]
    first_group = _read_doc(
        client, f"telegram_sessions/{skipped.context_id}/receipt_groups/{first_group_id}",
    )
    assert first_group["state"] == "closed"
    current[0] = NOW + timedelta(seconds=2)
    supported_update = _forward(61)
    supported_update["message"]["date"] = unsupported["message"]["date"] + 1
    supported = intake.handle_update(supported_update)
    assert supported.notification_id == skipped.notification_id
    assert supported.draft_id
    assert len(repository.get_draft(supported.context_id, supported.draft_id)["ordered_message_ids"]) == 1
    current[0] = NOW + timedelta(seconds=4)
    repository.refresh_receipt_card(supported.notification_id, now=current[0])
    card = repository.get_notification(supported.notification_id)
    assert "1 messages received" in card["body"]
    assert "Skipped 1 unsupported item" in card["body"]
    assert card["reply_markup"]["inline_keyboard"][0][0]["text"] == "Save"

    all_unsupported = _forward(62, chat_id=43)
    all_unsupported["message"].pop("text")
    all_unsupported["message"]["photo"] = [{"file_id": "photo-62"}]
    skipped_only = intake.handle_update(all_unsupported)
    current[0] = NOW + timedelta(seconds=4)
    repository.refresh_receipt_card(skipped_only.notification_id, now=current[0])
    skipped_card = repository.get_notification(skipped_only.notification_id)
    assert skipped_only.draft_id is None
    assert not skipped_card["reply_markup"]["inline_keyboard"]
    assert "skipped" in skipped_card["body"].lower()
    assert not _direct_child_paths(firestore_fake, "telegram_sessions/43_default/drafts")


def test_native_media_groups_stay_separate_and_text_joins_latest_open_group(firestore_fake):
    repository, client = _repository(firestore_fake)
    current = [NOW]
    intake = _intake(repository, _Publisher(), clock=lambda: current[0])
    first_update, second_update = _forward(70), _forward(71)
    first_update["message"]["media_group_id"] = "native-a"
    second_update["message"]["media_group_id"] = "native-b"
    first = intake.handle_update(first_update)
    second = intake.handle_update(second_update)
    assert first.notification_id != second.notification_id

    third_update = _forward(72)
    third_update["message"]["media_group_id"] = "native-a"
    third_update["message"]["date"] = first_update["message"]["date"] + 1
    third = intake.handle_update(third_update)
    assert third.notification_id == first.notification_id
    assert repository.get_update_receipt(first.context_id, 72)["receipt_group_id"] == (
        repository.get_update_receipt(first.context_id, 70)["receipt_group_id"]
    )

    standalone_update = _forward(73)
    standalone_update["message"]["date"] = second_update["message"]["date"] + 1
    standalone = intake.handle_update(standalone_update)
    assert standalone.notification_id == second.notification_id
    group_id = repository.get_update_receipt(first.context_id, 70)["receipt_group_id"]
    group = _read_doc(
        client,
        f"telegram_sessions/{first.context_id}/receipt_groups/{group_id}",
    )
    assert group["accepted_message_ids"] == [70, 72]
    second_group_id = repository.get_update_receipt(first.context_id, 71)["receipt_group_id"]
    second_group = _read_doc(
        client,
        f"telegram_sessions/{second.context_id}/receipt_groups/{second_group_id}",
    )
    assert second_group["accepted_message_ids"] == [71, 73]


def test_native_failed_card_is_not_revived_by_another_group_member(firestore_fake):
    repository, client = _repository(firestore_fake)
    current = [NOW]
    intake = _intake(repository, _Publisher(), clock=lambda: current[0])
    first_update = _forward(74)
    first_update["message"]["media_group_id"] = "native-terminal"
    first = intake.handle_update(first_update)
    old = repository.get_notification(first.notification_id)
    old.update({"state": "failed", "terminal_failure": "failed",
                "send_attempts_started": 1, "last_outcome": "rejected"})
    client.collection("notification_outbox").document(first.notification_id).set(old)
    prior_receipt = repository.get_update_receipt(first.context_id, 74)
    second_update = _forward(75)
    second_update["message"]["media_group_id"] = "native-terminal"
    second = intake.handle_update(second_update)

    assert second.notification_id != first.notification_id
    assert repository.get_notification(first.notification_id) == old
    group_id = repository.get_update_receipt(first.context_id, 75)["receipt_group_id"]
    assert group_id == prior_receipt["receipt_group_id"]
    group = _read_doc(client, f"telegram_sessions/{first.context_id}/receipt_groups/{group_id}")
    assert group["accepted_message_ids"] == [74, 75]
    assert repository.get_context(first.context_id)["pending_receipt_groups"] == 1


@pytest.mark.parametrize("old_card_state", ["saved_terminal", "failed_exhausted"])
def test_new_draft_never_revises_old_terminal_or_exhausted_receipt(
    firestore_fake, old_card_state,
):
    repository, client = _repository(firestore_fake)
    first = repository.accept_forward(
        76, chat_id=42, user_id=7,
        message={"message_id": 176, "text": "old source"}, now=NOW,
    )
    card = repository.get_notification(first.notification_id)
    if old_card_state == "saved_terminal":
        card.update({"state": "delivered", "receipt_terminal": True,
                     "receipt_terminal_kind": "final", "body": "Dialog #1: Saved and indexed.",
                     "save_operation_id": "old-save", "operation_id": "old-save",
                     "reply_markup": {"inline_keyboard": []},
                     "receipt_control_token": None})
    else:
        card.update({"state": "failed", "terminal_failure": "failed",
                     "send_attempts_started": 3})
    client.collection("notification_outbox").document(first.notification_id).set(card)
    context_ref = client.collection("telegram_sessions").document(first.context_id)
    context_ref.update({"active_draft_id": None, "draft_status": None})
    context_ref.collection("drafts").document(first.draft_id).update({"state": "cleaned"})
    before = repository.get_notification(first.notification_id)

    second = repository.accept_forward(
        77, chat_id=42, user_id=7,
        message={"message_id": 177, "text": "new source"}, now=NOW + timedelta(seconds=2),
    )

    assert second.draft_id != first.draft_id
    assert repository.get_notification(first.notification_id) == before
    assert repository.get_notification(first.notification_id)["state"] == card["state"]


def test_unsupported_forward_during_busy_save_does_not_create_receipt_group(firestore_fake):
    repository, client = _repository(firestore_fake)
    first = repository.accept_forward(
        78, chat_id=42, user_id=7,
        message={"message_id": 178, "text": "source"}, now=NOW,
    )
    before_context = repository.get_context(first.context_id)
    before_card = repository.get_notification(first.notification_id)
    repository.accept_command(
        79, chat_id=42, user_id=7, command=parse_command("save this"), now=NOW,
    )
    unsupported = repository.record_unsupported_forward(
        80, chat_id=42, user_id=7, message_id=180, now=NOW,
    )
    assert unsupported.accepted is False
    context_after = repository.get_context(first.context_id)
    assert context_after["latest_receipt_notification_id"] == before_context["latest_receipt_notification_id"]
    assert context_after["pending_receipt_groups"] == before_context["pending_receipt_groups"]
    assert repository.get_notification(first.notification_id) == before_card
    group_paths = _direct_child_paths(
        firestore_fake, f"telegram_sessions/{first.context_id}/receipt_groups",
    )
    assert len(group_paths) == 1


def _ready_receipt_callback(repository, *, current=None, update_id=90, user_id=7):
    current = current or [NOW]
    intake = _intake(repository, _Publisher(), clock=lambda: current[0])
    accepted = intake.handle_update(_forward(89))
    current[0] = NOW + timedelta(seconds=2)
    repository.refresh_receipt_card(accepted.notification_id, now=current[0])
    card = repository.get_notification(accepted.notification_id)
    card["message_id"] = 189
    repository.client.collection("notification_outbox").document(
        accepted.notification_id,
    ).set(card)
    callback = {
        "update_id": update_id,
        "callback_query": {
            "id": f"query-{update_id}", "from": {"id": user_id},
            "data": card["reply_markup"]["inline_keyboard"][0][0]["callback_data"],
            "message": {
                "message_id": 189, "date": 1793320100,
                "chat": {"id": 42, "type": "private"},
            },
        },
    }
    return intake, accepted, card, callback


def test_receipt_save_callback_reserves_typed_fifo_once_and_clears_same_card(
    firestore_fake,
):
    repository, client = _repository(firestore_fake)
    current = [NOW]
    intake, accepted, ready, callback = _ready_receipt_callback(
        repository, current=current,
    )

    result = intake.handle_callback(callback)
    assert result.accepted is True
    assert result.operation_id
    command = _read_doc(
        client, f"telegram_sessions/{accepted.context_id}/commands/90",
    )
    operation = _read_doc(
        client, f"telegram_sessions/{accepted.context_id}/operations/{result.operation_id}",
    )
    card = repository.get_notification(accepted.notification_id)
    assert command["kind"] == "save" and command["candidate"] == ""
    assert command["action_metadata"]["action"] == "receipt_save"
    assert operation["typed_action"] == command["action_metadata"]
    assert operation["input_ref"].endswith("/commands/90")
    assert "snapshot_message_ids" not in command or command["snapshot_message_ids"] == []
    assert card["body"] == f"Dialog #{ready['dialog_number']}: Saving…"
    assert card["reply_markup"] == {"inline_keyboard": []}
    assert card["receipt_control_token"] is None
    assert len(result.additional_outbox_ids) == 1
    assert _read_doc(client, f"notification_outbox/{accepted.notification_id}")["body"] == card["body"]
    intake.publish_result(result)
    assert result.additional_outbox_ids[0] in intake.publisher.outbox_ids
    assert _read_doc(
        client, f"telegram_sessions/{accepted.context_id}/receipt_controls/"
        f"{ready['receipt_control_token']}",
    )["state"] == "consumed"
    card_before_close = dict(card)
    repository.refresh_receipt_card(
        accepted.notification_id, now=current[0] + timedelta(seconds=2),
    )
    assert repository.get_context(accepted.context_id)["pending_receipt_groups"] == 0
    assert repository.get_notification(accepted.notification_id)["body"] == card_before_close["body"]
    assert repository.get_notification(accepted.notification_id)["reply_markup"] == {
        "inline_keyboard": [],
    }

    replay = intake.handle_callback(callback)
    assert replay.operation_id == result.operation_id
    assert replay.outbox_id == result.outbox_id
    second_click = {**callback, "update_id": 91,
                    "callback_query": {**callback["callback_query"], "id": "query-91"}}
    rejected = intake.handle_callback(second_click)
    assert rejected.accepted is False
    assert len(_direct_child_paths(firestore_fake,
                                   f"telegram_sessions/{accepted.context_id}/operations")) == 1


def test_receipt_callback_rejects_allowlisted_foreign_actor_and_wrong_message(
    firestore_fake,
):
    repository, _client = _repository(firestore_fake)
    current = [NOW]
    intake, _accepted, _ready, callback = _ready_receipt_callback(
        repository, current=current,
    )
    intake.allowed_user_ids = {7, 8}
    forged = {**callback, "update_id": 92,
              "callback_query": {**callback["callback_query"], "from": {"id": 8}}}
    assert intake.handle_callback(forged).accepted is False
    wrong_message = {**callback, "update_id": 93,
                     "callback_query": {**callback["callback_query"],
                                        "message": {"message_id": 999,
                                                    "date": 1793320100,
                                                    "chat": {"id": 42, "type": "private"}}}}
    assert intake.handle_callback(wrong_message).accepted is False
    inaccessible = {**callback, "update_id": 96,
                    "callback_query": {**callback["callback_query"], "id": "inaccessible",
                                       "message": {"message_id": 189, "date": 0,
                                                   "chat": {"id": 42, "type": "private"}}}}
    assert intake.handle_callback(inaccessible) is None
    malformed_thread = {**callback, "update_id": 97,
                        "callback_query": {**callback["callback_query"], "id": "bad-thread",
                                           "message": {"message_id": 189, "date": 1793320100,
                                                       "message_thread_id": False,
                                                       "chat": {"id": 42, "type": "private"}}}}
    assert intake.handle_callback(malformed_thread) is None


def test_receipt_callback_rejects_while_a_freeform_operation_reserves_the_context(
    firestore_fake,
):
    repository, _client = _repository(firestore_fake)
    current = [NOW]
    intake, accepted, ready, callback = _ready_receipt_callback(repository, current=current)
    pending = repository.accept_command(
        88, chat_id=42, user_id=7, command=parse_command("save this dialog"), now=current[0],
    )
    assert pending.accepted is True

    rejected = intake.handle_callback(callback)
    assert rejected.accepted is False
    card = repository.get_notification(accepted.notification_id)
    assert card["body"] == ready["body"]
    assert card["reply_markup"] == ready["reply_markup"]
    assert len(_direct_child_paths(
        firestore_fake, f"telegram_sessions/{accepted.context_id}/operations",
    )) == 1


def test_new_receipt_invalidates_an_older_ready_card_callback(firestore_fake):
    repository, _client = _repository(firestore_fake)
    current = [NOW]
    intake, _accepted, _ready, callback = _ready_receipt_callback(
        repository, current=current,
    )
    current[0] = NOW + timedelta(seconds=3)
    intake.handle_update(_forward(94))
    stale = {**callback, "update_id": 95,
             "callback_query": {**callback["callback_query"], "id": "stale-query"}}

    assert intake.handle_callback(stale).accepted is False
    context = repository.get_context("42_default")
    assert context["pending_receipt_groups"] == 1


def test_default_webhook_answers_save_callback_after_durable_reservation(firestore_fake):
    repository, _client = _repository(firestore_fake)
    current = [NOW]
    intake, accepted, ready, callback = _ready_receipt_callback(
        repository, current=current,
    )

    class Bot:
        def __init__(self):
            self.answered = []

        async def answer_callback_query(self, query_id, *, text=None):
            self.answered.append((query_id, text))
            return True

    bot = Bot()
    app_module = importlib.import_module("services.connectors.telegram.app.main")
    app = app_module.create_app(
        intake=intake, webhook_secret="synthetic-webhook-secret",
        telegram_enabled=True, telegram_client=bot,
    )

    async def post():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://telegram.test") as http:
            return await http.post(
                "/telegram/webhook",
                headers={"X-Telegram-Bot-Api-Secret-Token": "synthetic-webhook-secret"},
                json=callback,
            )

    response = asyncio.run(post())
    assert response.status_code == 200
    assert bot.answered == [(callback["callback_query"]["id"], None)]
    card = repository.get_notification(accepted.notification_id)
    assert card["body"] == f"Dialog #{ready['dialog_number']}: Saving…"
    receipt = repository.get_update_receipt(accepted.context_id, callback["update_id"])
    assert receipt["accepted"] is True


def test_webhook_persists_thinking_card_and_schedules_close_from_canonical_outbox(
    firestore_fake,
):
    repository, client = _repository(firestore_fake)
    publisher = _Publisher()
    current = [NOW]
    intake = _intake(repository, publisher, clock=lambda: current[0])
    app_module = importlib.import_module("services.connectors.telegram.app.main")
    app = app_module.create_app(
        intake=intake, webhook_secret="synthetic-webhook-secret", telegram_enabled=True,
    )

    async def post(update):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://telegram.test") as http:
            return await http.post(
                "/telegram/webhook",
                headers={"X-Telegram-Bot-Api-Secret-Token": "synthetic-webhook-secret"},
                json=update,
            )

    first = asyncio.run(post(_forward(80)))
    assert first.status_code == 200
    receipt = repository.get_update_receipt("42_default", 80)
    card = repository.get_notification(receipt["notification_id"])
    initial_task = _read_doc(client, f"task_outbox/{receipt['outbox_id']}")
    close_id = receipt["additional_outbox_ids"][0]
    close_task = _read_doc(client, f"task_outbox/{close_id}")
    assert "Thinking" in card["body"]
    assert initial_task["schedule_at"] == NOW
    assert close_task["schedule_at"] == NOW + timedelta(seconds=1)

    current[0] = NOW + timedelta(milliseconds=300)
    second_update = _forward(81)
    second_update["message"]["date"] = _forward(80)["message"]["date"] + 1
    second = asyncio.run(post(second_update))
    assert second.status_code == 200
    second_receipt = repository.get_update_receipt("42_default", 81)
    assert second_receipt["notification_id"] == receipt["notification_id"]
    group = _read_doc(
        client, f"telegram_sessions/42_default/receipt_groups/{receipt['receipt_group_id']}"
    )
    assert group["accepted_message_ids"] == [80, 81]
    latest_close = _read_doc(client, f"task_outbox/{second_receipt['outbox_id']}")
    assert latest_close["schedule_at"] == NOW + timedelta(seconds=1.3)
    assert repository.get_notification(receipt["notification_id"])["reply_markup"] == {
        "inline_keyboard": [],
    }


def test_distinct_forwards_reuse_the_unique_collecting_draft_and_dialog_number(
    firestore_fake,
):
    repository, client = _repository(firestore_fake)
    intake = _intake(repository, _Publisher(), _Classifier(), _Classifier())

    first = intake.handle_update(_forward(2))
    second = intake.handle_update(_forward(3))
    context_id = _value(first, "context_id")

    assert _value(first, "draft_id") == _value(second, "draft_id")
    assert _value(first, "dialog_number") == _value(second, "dialog_number")
    context = repository.get_context(context_id)
    assert context["active_conversation_id"] == f"{context_id}-{_value(first, 'dialog_number')}"
    assert context["selected_dialog_number"] == _value(first, "dialog_number")
    assert context["selection_sequence"] == 1
    assert len(_direct_child_paths(firestore_fake, f"telegram_sessions/{context_id}/drafts")) == 1
    draft_id = _value(first, "draft_id")
    assert _read_doc(client, _message_path(context_id, draft_id, 102))["message_id"] == 102
    assert _read_doc(client, _message_path(context_id, draft_id, 103))["message_id"] == 103


def test_channel_forward_persists_original_channel_metadata(firestore_fake):
    repository, client = _repository(firestore_fake)
    intake = _intake(repository, _Publisher(), _Classifier(), _Classifier())
    update = _forward(4)
    update["message"]["forward_origin"] = {
        "type": "channel",
        "chat": {"id": -1001234567890, "title": "Original channel"},
        "message_id": 765,
        "date": 1_798_000_000,
    }

    accepted = intake.handle_update(update)

    context_id = _value(accepted, "context_id")
    draft_id = _value(accepted, "draft_id")
    message = _read_doc(client, _message_path(context_id, draft_id, 104))
    assert message["source_chat_id"] == -1001234567890
    assert message["source_message_id"] == 765
    assert message["reliable_author_key"] == "telegram-chat:-1001234567890"
    assert message["display_speaker"] == "Original channel"


def test_duplicate_update_reuses_all_records_and_repairs_same_outbox_after_gap(
    firestore_fake,
):
    repository, client = _repository(firestore_fake)
    publisher = _Publisher()
    publisher.failures = 1
    intake = _intake(repository, publisher, _Classifier(), _Classifier())
    update = _forward(8)

    with pytest.raises(RuntimeError, match="synthetic publish gap"):
        intake.handle_update(update)
    replay = intake.handle_update(update)
    context_id = _value(replay, "context_id")
    draft_id = _value(replay, "draft_id")
    notification_id = _value(replay, "notification_id")
    outbox_id = _value(replay, "outbox_id")
    context = repository.get_context(context_id)
    assert context["active_conversation_id"] == f"{context_id}-{_value(replay, 'dialog_number')}"
    assert context["selected_dialog_number"] == _value(replay, "dialog_number")
    assert context["selection_sequence"] == 1
    draft = _read_doc(client, f"telegram_sessions/{context_id}/drafts/{draft_id}")
    update_record = _read_doc(client, f"telegram_sessions/{context_id}/updates/8")
    notification = _read_doc(client, f"notification_outbox/{notification_id}")
    outbox = _read_doc(client, f"task_outbox/{outbox_id}")
    assert draft is not None
    assert update_record["message_id"] == 108
    assert notification["notification_id"] == notification_id
    assert outbox["outbox_id"] == outbox_id
    assert publisher.outbox_ids == [
        outbox_id, outbox_id, *_value(replay, "additional_outbox_ids"),
    ]
    assert len(_direct_child_paths(firestore_fake, f"telegram_sessions/{context_id}/updates")) == 1
    assert len(_paths_with_prefix(
        firestore_fake,
        f"telegram_sessions/{context_id}/drafts/{draft_id}/messages/",
    )) == 1


def test_unsupported_contentless_forward_gets_one_durable_notice_without_draft(firestore_fake):
    repository, client = _repository(firestore_fake)
    publisher = _Publisher()
    intake = _intake(repository, publisher, _Classifier(), _Classifier())
    update = _forward(801)
    update["message"].pop("text")
    update["message"]["photo"] = [{"file_id": "synthetic-photo"}]

    first = intake.handle_update(update)
    replay = intake.handle_update(update)

    assert _value(first, "accepted") is True
    assert _value(first, "draft_id") is None
    assert _value(first, "notification_id") == _value(replay, "notification_id")
    assert _value(first, "outbox_id") == _value(replay, "outbox_id")
    context_id = _value(first, "context_id")
    receipt = _read_doc(client, f"telegram_sessions/{context_id}/updates/801")
    notice = _read_doc(client, f"notification_outbox/{_value(first, 'notification_id')}")
    outbox = _read_doc(client, f"task_outbox/{_value(first, 'outbox_id')}")
    assert receipt["outcome"] == "unsupported_forward_skipped"
    assert notice["kind"] == "draft_ack"
    assert "no supported text or audio" in notice["body"]
    assert outbox["notification_id"] == notice["notification_id"]
    assert not _direct_child_paths(firestore_fake, f"telegram_sessions/{context_id}/drafts")
    ids = [_value(first, "outbox_id"), *_value(first, "additional_outbox_ids")]
    assert publisher.outbox_ids == [*ids, *ids]


@pytest.mark.parametrize(("chat_type", "user_id"), [("group", 7), ("private", 999)])
def test_disallowed_chat_or_sender_is_acknowledged_without_persisting_content(
    firestore_fake, chat_type, user_id
):
    repository, client = _repository(firestore_fake)
    intake_type = importlib.import_module(
        "services.connectors.telegram.intake"
    ).TelegramIntake
    intake = intake_type(
        repository=repository,
        publisher=_Publisher(),
        classifier=_Classifier(),
        semantic_router=_Classifier(),
        allowed_user_ids={7},
    )
    update = _forward(9, user_id=user_id)
    update["message"]["chat"]["type"] = chat_type

    app_module = importlib.import_module("services.connectors.telegram.app.main")
    app = app_module.create_app(
        intake=intake,
        webhook_secret="synthetic-webhook-secret",
        telegram_enabled=True,
    )

    async def request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://telegram.test") as client:
            return await client.post(
                "/telegram/webhook",
                headers={"X-Telegram-Bot-Api-Secret-Token": "synthetic-webhook-secret"},
                json=update,
            )

    response = asyncio.run(request())

    assert response.status_code == 200
    assert not any("/drafts/" in path for path in firestore_fake.documents)
    assert not any("/messages/" in path for path in firestore_fake.documents)
    assert all(update["message"]["text"] not in str(value) for value in firestore_fake.documents.values())


def test_invalid_webhook_secret_is_rejected_before_intake_or_persistence(
    firestore_fake,
):
    repository, client = _repository(firestore_fake)
    publisher = _Publisher()
    intake = _intake(repository, publisher, _Classifier(), _Classifier())
    app_module = importlib.import_module("services.connectors.telegram.app.main")
    app = app_module.create_app(
        intake=intake,
        webhook_secret="synthetic-webhook-secret",
        telegram_enabled=True,
    )

    async def request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://telegram.test") as client:
            return await client.post(
                "/telegram/webhook",
                headers={"X-Telegram-Bot-Api-Secret-Token": "wrong-synthetic-secret"},
                json=_forward(10),
            )

    response = asyncio.run(request())
    assert response.status_code == 403
    assert publisher.outbox_ids == []
    assert firestore_fake.documents == {}


def test_malformed_json_is_rejected(firestore_fake):
    repository, _ = _repository(firestore_fake)
    intake = _intake(repository, _Publisher(), _Classifier(), _Classifier())
    app_module = importlib.import_module("services.connectors.telegram.app.main")
    app = app_module.create_app(
        intake=intake,
        webhook_secret="synthetic-webhook-secret",
        telegram_enabled=True,
    )

    async def request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://telegram.test") as client:
            return await client.post(
                "/telegram/webhook",
                headers={"X-Telegram-Bot-Api-Secret-Token": "synthetic-webhook-secret"},
                content="{invalid",
            )

    response = asyncio.run(request())
    assert response.status_code == 400
    assert firestore_fake.documents == {}


def test_publisher_failure_returns_service_unavailable(firestore_fake):
    repository, _ = _repository(firestore_fake)
    publisher = _Publisher()
    publisher.failures = 1
    intake = _intake(repository, publisher, _Classifier(), _Classifier())
    app_module = importlib.import_module("services.connectors.telegram.app.main")
    app = app_module.create_app(
        intake=intake,
        webhook_secret="synthetic-webhook-secret",
        telegram_enabled=True,
    )

    async def request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://telegram.test") as client:
            return await client.post(
                "/telegram/webhook",
                headers={"X-Telegram-Bot-Api-Secret-Token": "synthetic-webhook-secret"},
                json=_forward(11),
            )

    response = asyncio.run(request())
    assert response.status_code == 503
    assert len(publisher.outbox_ids) == 1
    assert any(path.endswith("/updates/11") for path in firestore_fake.documents)


def test_health_returns_only_safe_capability_status(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "secret-token-value")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USER_IDS", "7")
    app_module = importlib.import_module("services.connectors.telegram.app.main")
    app = app_module.create_app(webhook_secret="synthetic-webhook-secret")

    async def request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://telegram.test") as client:
            return await client.get("/")

    response = asyncio.run(request())
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "telegram_enabled": True}
    assert "secret-token-value" not in response.text
    assert "synthetic-webhook-secret" not in response.text


def test_command_reservation_blocks_forwards_until_all_fifo_heads_finish(
    firestore_fake,
):
    repository, client = _repository(firestore_fake)
    intake = _intake(repository, _Publisher(), _Classifier(), _Classifier())

    first = intake.handle_update(_command(20, "Сохрани"))
    second = intake.handle_update(_command(21, "Расскажи подробнее"))
    context_id = _value(first, "context_id")
    assert _value(first, "queue_position") == 1
    assert _value(second, "queue_position") == 2
    context = client.collection("telegram_sessions").document(context_id)
    assert context.get().to_dict()["reserved_operation_id"] == _value(first, "operation_id")

    first_lease = claim_head(client, context, owner="pytest-worker", now=NOW)
    assert first_lease is not None
    assert first_lease.operation_id == _value(first, "operation_id")
    assert complete_head(
        client, context, operation_id=first_lease.operation_id, owner="pytest-worker",
        fence=first_lease.fence, now=NOW,
    ) is True
    state = context.get().to_dict()
    assert state["reserved_operation_id"] == _value(second, "operation_id")
    assert _value(intake.handle_update(_forward(22)), "accepted") is False

    second_lease = claim_head(client, context, owner="pytest-worker-2", now=NOW)
    assert second_lease is not None
    assert second_lease.operation_id == _value(second, "operation_id")
    assert complete_head(
        client, context, operation_id=second_lease.operation_id, owner="pytest-worker-2",
        fence=second_lease.fence, now=NOW,
    ) is True
    assert context.get().to_dict()["reserved_operation_id"] is None
    accepted = intake.handle_update(_forward(23))
    assert _value(accepted, "accepted") is True


def test_busy_rejected_forward_does_not_change_focus_or_selection_sequence(firestore_fake):
    repository, client = _repository(firestore_fake)
    accepted = repository.accept_forward(
        1, chat_id=42,
        message={"message_id": 101, "text": "first"}, now=NOW,
    )
    context_ref = client.collection("telegram_sessions").document(accepted.context_id)
    context_ref.update({"reserved_operation_id": "busy-operation"})
    before = repository.get_context(accepted.context_id)

    rejected = repository.accept_forward(
        2, chat_id=42,
        message={"message_id": 102, "text": "rejected"}, now=NOW,
    )

    after = repository.get_context(accepted.context_id)
    assert rejected.accepted is False
    assert after["active_draft_id"] == before["active_draft_id"] == accepted.draft_id
    assert after["active_conversation_id"] == before["active_conversation_id"]
    assert after["selected_dialog_number"] == before["selected_dialog_number"]
    assert after["selection_sequence"] == before["selection_sequence"] == 1


def test_concurrent_command_and_forward_are_decided_by_repository_transaction(
    firestore_fake,
):
    repository, client = _repository(firestore_fake)
    intake = _intake(repository, _Publisher(), _Classifier(), _Classifier())
    command = _command(30, "Сохрани")
    forward = _forward(31)

    with ThreadPoolExecutor(max_workers=2) as executor:
        command_future = executor.submit(intake.handle_update, command)
        forward_future = executor.submit(intake.handle_update, forward)
        command_result = command_future.result(timeout=10)
        forward_result = forward_future.result(timeout=10)

    if _value(command_result, "accepted"):
        assert _value(forward_result, "accepted") is False
        context_id = _value(command_result, "context_id")
        assert not any(
            path.endswith("/messages/131")
            for path in firestore_fake.documents
            if path.startswith(f"telegram_sessions/{context_id}/drafts/")
        )
    else:
        assert _value(forward_result, "accepted") is True
        context_id = _value(forward_result, "context_id")
        draft_id = _value(forward_result, "draft_id")
        assert _read_doc(client, _message_path(context_id, draft_id, 131))["message_id"] == 131


def test_concurrent_forward_and_save_are_snapshot_included_or_rejected(
    firestore_fake,
):
    repository, client = _repository(firestore_fake)
    intake = _intake(repository, _Publisher(), _Classifier(), _Classifier())
    forward = _forward(41)
    save = _command(40, "Сохрани")

    with ThreadPoolExecutor(max_workers=2) as executor:
        forward_future = executor.submit(intake.handle_update, forward)
        save_future = executor.submit(intake.handle_update, save)
        forward_result = forward_future.result(timeout=10)
        save_result = save_future.result(timeout=10)

    accepted = _value(forward_result, "accepted")
    message_id = forward["message"]["message_id"]
    snapshot_ids = _value(save_result, "snapshot_message_ids")
    assert accepted is (message_id in snapshot_ids)
    context_id = _value(save_result, "context_id")
    if accepted:
        assert snapshot_ids.count(message_id) == 1
        draft_id = _value(forward_result, "draft_id")
        assert _read_doc(client, _message_path(context_id, draft_id, message_id)) is not None
    else:
        assert not any(
            path.endswith(f"/messages/{message_id}")
            for path in firestore_fake.documents
            if path.startswith(f"telegram_sessions/{context_id}/drafts/")
        )


def test_command_replay_reuses_receipt_and_queue_operation(firestore_fake):
    repository, client = _repository(firestore_fake)
    publisher = _Publisher()
    intake = _intake(repository, publisher, _Classifier(), _Classifier())
    update = _command(50, "Расскажи подробнее")

    first = intake.handle_update(update)
    replay = intake.handle_update(update)

    assert _value(first, "operation_id") == _value(replay, "operation_id")
    assert _value(first, "queue_position") == _value(replay, "queue_position") == 1
    assert _value(first, "outbox_id") == _value(replay, "outbox_id")
    assert publisher.outbox_ids == [_value(first, "outbox_id")] * 2
    context_id = _value(first, "context_id")
    assert len(_direct_child_paths(firestore_fake, f"telegram_sessions/{context_id}/operations")) == 1
    assert len(_direct_child_paths(firestore_fake, f"telegram_sessions/{context_id}/commands")) == 1
    outbox = _read_doc(client, f"task_outbox/{_value(first, 'outbox_id')}")
    assert outbox["target_kind"] == "telegram_dispatch"
    assert outbox["operation_id"] == _value(first, "operation_id")
    assert outbox["context_id"] == context_id


def test_save_seals_ordered_snapshot_and_rejects_later_forward(firestore_fake):
    repository, client = _repository(firestore_fake)
    publisher = _Publisher()
    intake = _intake(repository, publisher, _Classifier(), _Classifier())
    first = intake.handle_update(_forward(51))
    second = intake.handle_update(_forward(52))
    save = intake.handle_update(_command(53, "Сохрани"))
    save_replay = intake.handle_update(_command(53, "Сохрани"))

    assert _value(save, "snapshot_message_ids") == (151, 152)
    assert _value(save_replay, "operation_id") == _value(save, "operation_id")
    assert _value(save_replay, "outbox_id") == _value(save, "outbox_id")
    assert publisher.outbox_ids[-2:] == [_value(save, "outbox_id")] * 2
    context_id = _value(save, "context_id")
    draft_id = _value(first, "draft_id")
    context_ref = client.collection("telegram_sessions").document(context_id)
    assert repository.get_draft(context_id, draft_id)["state"] == "collecting"
    lease = claim_head(client, context_ref, owner="save-confirm", now=NOW)
    assert lease is not None
    assert repository.confirm_save_command(
        context_id, lease.operation_id, owner=lease.owner, fence=lease.fence, now=NOW,
    ) is True
    assert repository.confirm_save_command(
        context_id, lease.operation_id, owner=lease.owner, fence=lease.fence, now=NOW,
    ) is True
    draft = _read_doc(client, f"telegram_sessions/{context_id}/drafts/{draft_id}")
    assert draft["state"] == "processing"
    assert draft["sealed_at"] is not None
    assert draft["ordered_message_ids"] == [151, 152]
    assert draft["active_save_operation_id"] == _value(save, "operation_id")
    assert draft["manifest_ref"]
    assert draft["manifest_digest"]
    assert len(_direct_child_paths(firestore_fake, f"telegram_sessions/{context_id}/operations")) == 1
    assert len(_direct_child_paths(
        firestore_fake, f"telegram_sessions/{context_id}/drafts/{draft_id}/save_manifests"
    )) == 1
    assert _value(intake.handle_update(_forward(54)), "accepted") is False
    assert draft["manifest_digest"] == _read_doc(
        client, f"telegram_sessions/{context_id}/drafts/{draft_id}"
    )["manifest_digest"]


def test_empty_save_reserves_without_creating_draft_or_doc(firestore_fake):
    repository, client = _repository(firestore_fake)
    publisher = _Publisher()
    intake = _intake(repository, publisher, _Classifier(), _Classifier())

    result = intake.handle_update(_command(55, "Сохрани"))

    context_id = _value(result, "context_id")
    assert _value(result, "accepted") is True
    assert _value(result, "snapshot_message_ids") == ()
    assert not _direct_child_paths(firestore_fake, f"telegram_sessions/{context_id}/drafts")
    assert _value(result, "operation_id")
    context_ref = client.collection("telegram_sessions").document(context_id)
    lease = claim_head(client, context_ref, owner="empty-save", now=NOW)
    assert lease is not None
    assert repository.confirm_save_command(
        context_id, lease.operation_id, owner=lease.owner, fence=lease.fence, now=NOW,
    ) is False
    assert publisher.outbox_ids == [_value(result, "outbox_id")]
    assert _read_doc(client, f"task_outbox/{_value(result, 'outbox_id')}")["target_kind"] == "telegram_dispatch"


def test_freeform_command_candidate_contains_no_draft_source_text(firestore_fake):
    repository, client = _repository(firestore_fake)
    intake = _intake(repository, _Publisher(), _Classifier(), _Classifier())
    intake.handle_update(_forward(56))
    result = intake.handle_update(_command(57, "Суммируй это"))
    context_id = _value(result, "context_id")
    stored = _read_doc(client, f"telegram_sessions/{context_id}/commands/57")

    assert stored["candidate"] == "Суммируй это"
    assert stored["cutoff_message_ids"] == [156]
    assert "synthetic forwarded message 56" not in str(stored)
    assert "text" not in stored and "audio_file_id" not in stored


def test_malformed_forward_origin_is_not_intaken_as_a_command(firestore_fake):
    repository, _ = _repository(firestore_fake)
    intake = _intake(repository, _Publisher(), _Classifier(), _Classifier())
    update = _command(58, "Сохрани")
    update["message"]["forward_origin"] = "malformed-forward-metadata"

    assert intake.handle_update(update) is None
    assert firestore_fake.documents == {}
