"""Telegram per-thread state isolation and draft rollover contracts."""

from __future__ import annotations

import asyncio
import importlib
import os
import socket
from datetime import datetime, timezone

import pytest
from core.integrations.firestore.operation_queue import claim_head, complete_head
from services.connectors.telegram.models import PreparedUtterance


pytestmark = pytest.mark.offline

NOW = datetime(2026, 10, 22, tzinfo=timezone.utc)


def _selector_loop():
    if os.name != "nt":
        return asyncio.SelectorEventLoop()
    original_socketpair = socket.socketpair
    pairs = []

    def loopback_pair(*_args, **_kwargs):
        reader = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        writer = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        reader.bind(("127.0.0.1", 0))
        writer.bind(("127.0.0.1", 0))
        writer.connect(reader.getsockname())
        reader.connect(writer.getsockname())
        pairs.append((reader, writer))
        return reader, writer

    socket.socketpair = loopback_pair
    try:
        return asyncio.SelectorEventLoop()
    finally:
        socket.socketpair = original_socketpair


def _run(awaitable):
    return asyncio.run(awaitable, loop_factory=_selector_loop)


def _repository(firestore_fake):
    repository_type = importlib.import_module(
        "services.connectors.telegram.repository"
    ).TelegramRepository
    client = firestore_fake.client_class()
    return repository_type(client=client), client


class _Publisher:
    def __init__(self):
        self.outbox_ids = []

    def publish(self, outbox_id):
        self.outbox_ids.append(outbox_id)


def _intake(repository):
    intake_type = importlib.import_module(
        "services.connectors.telegram.intake"
    ).TelegramIntake
    return intake_type(repository=repository, publisher=_Publisher(), clock=lambda: NOW)


def _update(update_id, *, thread_id=None, text=None, forward=True):
    message = {
        "message_id": update_id + 100,
        "date": 1_798_000_000,
        "chat": {"id": 42, "type": "private"},
        "from": {"id": 7},
    }
    if thread_id is not None:
        message["message_thread_id"] = thread_id
    if forward:
        message.update({
            "forward_origin": {"type": "user", "sender_user": {"id": 91}},
            "text": text or f"synthetic source {update_id}",
        })
    else:
        message["text"] = text if text is not None else "Сохрани"
    return {"update_id": update_id, "message": message}


def _topic_event(update_id, *, thread_id, event="forum_topic_created", name=None):
    message = {
        "chat": {"id": 42, "type": "private"},
        "from": {"id": 7},
        "message_thread_id": thread_id,
        event: ({"name": name} if name is not None else {}),
    }
    return {"update_id": update_id, "message": message}


def _value(record, field):
    return record[field] if isinstance(record, dict) else getattr(record, field)


def test_default_and_thread_contexts_keep_drafts_sequences_and_queues_independent(
    firestore_fake,
):
    repository, client = _repository(firestore_fake)
    intake = _intake(repository)

    accepted = {}
    intake.handle_update(_topic_event(100, thread_id=11, name="Shared name"))
    intake.handle_update(_topic_event(101, thread_id=22, name="Shared name"))
    for update_id, thread_id in ((1, None), (2, 11), (3, 22)):
        accepted[thread_id] = intake.handle_update(_update(
            update_id, thread_id=thread_id,
        ))
    for update_id, thread_id in ((4, None), (5, 11), (6, 22)):
        accepted[(thread_id, "second")] = intake.handle_update(_update(
            update_id, thread_id=thread_id,
        ))

    expected_contexts = {None: "42_default", 11: "42_11", 22: "42_22"}
    for thread_id, context_id in expected_contexts.items():
        first = accepted[thread_id]
        second = accepted[(thread_id, "second")]
        assert _value(first, "context_id") == context_id
        assert _value(first, "draft_id") == _value(second, "draft_id")
        assert _value(first, "dialog_number") == _value(second, "dialog_number") == 1
        assert _value(first, "intake_sequence") == 1
        assert _value(second, "intake_sequence") == 2
        context = repository.get_context(context_id)
        assert context["next_intake_sequence"] == 3
        assert context["next_dialog_number"] == 2
        assert context["topic_name"] == ("Shared name" if thread_id is not None else None)

    draft_ids = {_value(row, "draft_id") for row in accepted.values()}
    assert len(draft_ids) == 3
    assert _value(accepted[11], "draft_id") != _value(accepted[22], "draft_id")

    # Queue counters are local to each context; taking one head leaves the other
    # two queues ready to make progress.
    commands = {}
    for update_id, thread_id in ((10, None), (12, 11), (14, 22)):
        commands[(thread_id, 1)] = intake.handle_update(_update(
            update_id, thread_id=thread_id, forward=False,
        ))
    for update_id, thread_id in ((11, None), (13, 11), (15, 22)):
        commands[(thread_id, 2)] = intake.handle_update(_update(
            update_id, thread_id=thread_id, forward=False,
        ))
    assert {_value(row, "queue_position") for row in commands.values()} == {1, 2}
    for thread_id, context_id in expected_contexts.items():
        assert repository.get_context(context_id)["next_operation_sequence"] == 3

    refs = {
        thread_id: client.collection("telegram_sessions").document(context_id)
        for thread_id, context_id in expected_contexts.items()
    }
    lease = claim_head(client, refs[11], owner="thread-11", now=NOW)
    assert lease is not None
    assert claim_head(client, refs[11], owner="thread-11-again", now=NOW) is None
    assert claim_head(client, refs[None], owner="default", now=NOW) is not None
    assert claim_head(client, refs[22], owner="thread-22", now=NOW) is not None
    assert complete_head(
        client, refs[11], operation_id=lease.operation_id,
        owner=lease.owner, fence=lease.fence, now=NOW,
    ) is True
    assert repository.get_context("42_11")["head_sequence"] == 2
    assert repository.get_context("42_11")["reserved_operation_id"] == _value(commands[(11, 2)], "operation_id")
    assert repository.get_context("42_default")["head_sequence"] == 1
    assert repository.get_context("42_default")["active_operation_id"] == _value(commands[(None, 1)], "operation_id")
    assert repository.get_context("42_22")["head_sequence"] == 1
    assert repository.get_context("42_22")["active_operation_id"] == _value(commands[(22, 1)], "operation_id")


def test_same_topic_name_does_not_merge_threads_and_rename_only_updates_its_hint(
    firestore_fake,
):
    repository, _ = _repository(firestore_fake)
    intake = _intake(repository)

    intake.handle_update(_topic_event(102, thread_id=11, name="Planning"))
    intake.handle_update(_topic_event(103, thread_id=22, name="Planning"))
    left = intake.handle_update(_update(20, thread_id=11))
    right = intake.handle_update(_update(21, thread_id=22))
    intake.handle_update(_topic_event(104, thread_id=11, event="forum_topic_edited", name="Renamed"))
    renamed = intake.handle_update(_update(22, thread_id=11))

    assert _value(left, "context_id") == "42_11"
    assert _value(right, "context_id") == "42_22"
    assert _value(left, "draft_id") != _value(right, "draft_id")
    assert _value(renamed, "draft_id") == _value(left, "draft_id")
    assert repository.get_context("42_11")["topic_name"] == "Renamed"
    assert repository.get_context("42_22")["topic_name"] == "Planning"
    assert repository.list_messages("42_11", _value(left, "draft_id"))[-1]["text"] == "synthetic source 22"
    assert repository.list_messages("42_22", _value(right, "draft_id"))[0]["text"] == "synthetic source 21"


def test_save_rollover_allocates_fresh_draft_and_second_processing_context_is_current_only(
    firestore_fake,
):
    repository, _ = _repository(firestore_fake)
    intake = _intake(repository)
    first = intake.handle_update(_update(30, thread_id=11, text="old private source"))
    first_context = _value(first, "context_id")
    first_draft = _value(first, "draft_id")
    repository.save_prepared_utterances(
        first_context, first_draft,
        [PreparedUtterance(0, "speaker", None, "old prepared content", 3, (130,))],
        required_audio_count=0, completed_audio_ids=(), context_capacity=20,
        preparation_digest="prepared-old", now=NOW,
    )
    save_first = intake.handle_update(_update(31, thread_id=11, forward=False))
    assert _value(save_first, "snapshot_message_ids") == (130,)
    context_ref = repository.client.collection("telegram_sessions").document(first_context)
    lease = claim_head(repository.client, context_ref, owner="save-worker", now=NOW)
    assert lease is not None
    assert complete_head(
        repository.client, context_ref, operation_id=lease.operation_id,
        owner=lease.owner, fence=lease.fence, now=NOW,
    ) is True

    second = intake.handle_update(_update(32, thread_id=11, text="new private source"))
    second_draft = _value(second, "draft_id")
    assert second_draft != first_draft
    assert _value(second, "dialog_number") == 2
    assert _value(second, "intake_sequence") == 2
    save_second = intake.handle_update(_update(33, thread_id=11, forward=False))
    assert _value(save_second, "snapshot_message_ids") == (132,)
    assert repository.get_draft(first_context, first_draft)["ordered_message_ids"] == [130]
    assert repository.get_draft(first_context, second_draft)["ordered_message_ids"] == [132]
    assert repository.get_draft(first_context, second_draft)["prepared_count"] == 0

    # Run the public save workflow against only the second sealed manifest. Any
    # accidental previous-draft lookup is represented by a conspicuous legacy
    # context in the adapter and must not leak into model input.
    workflow_type = importlib.import_module(
        "services.connectors.telegram.save"
    ).TelegramSaveWorkflow
    draft = repository.get_draft(first_context, second_draft)
    source = repository.list_messages(first_context, second_draft)
    sealed = {
        "draft_id": second_draft, "context_id": first_context, "status": "sealed",
        "manifest_digest": draft["manifest_digest"], "messages": source,
        "source_manifest": source,
    }

    class WorkflowRepository:
        context_id = first_context
        previous_dialogue_context = [{"role": "user", "content": "old private source"}]

        def load_sealed_draft(self, draft_id):
            assert draft_id == second_draft
            return sealed

        def load_current_guard(self, draft_id):
            return {"version": 1}

        def verify_save_receipt(self, receipt, *, draft_id, manifest_digest, guard):
            return (receipt["draft_id"] == draft_id
                    and receipt["manifest_digest"] == manifest_digest
                    and guard == {"version": 1})

        def commit_completion(self, certificate):
            return True

    contexts = []

    class Audio:
        async def process_messages(self, messages, *, draft_id, analyze):
            return await analyze(messages)

    async def analyze(messages, *, context):
        contexts.append(context)
        return {"messages": messages}

    async def save_pipeline(result, *, draft_id, manifest_digest, intent_id, operation_id, provider=None, dialog_date=None):
        return {"draft_id": draft_id, "manifest_digest": manifest_digest,
                "save_intent_id": intent_id, "operation_id": operation_id,
                "receipt_ref": "synthetic-receipt", "receipt_version": 1}

    workflow = workflow_type(
        repository=WorkflowRepository(), audio_transcriber=Audio(), analyze=analyze,
        classify=lambda result, *, draft_id: result, save=save_pipeline,
    )
    _run(workflow.save(
        draft_id=second_draft, intent_id="intent-second", operation_id="operation-second",
    ))

    context = contexts[0]
    assert [row["text"] for row in context["messages"]] == ["new private source"]
    assert "old private source" not in repr(context)
    assert "old prepared content" not in repr(context)


def test_private_default_intake_needs_no_forum_or_getme_capability_flags(firestore_fake):
    repository, _ = _repository(firestore_fake)
    intake = _intake(repository)

    accepted = intake.handle_update(_update(40))

    assert _value(accepted, "accepted") is True
    assert _value(accepted, "context_id") == "42_default"
    assert repository.get_context("42_default")["thread_id"] is None


@pytest.mark.parametrize("invalid_thread_id", [0, -1, "11", True, False, 1.5])
@pytest.mark.parametrize("forward", [True, False])
def test_invalid_explicit_thread_id_is_ignored_without_default_context_writes(
    firestore_fake, invalid_thread_id, forward,
):
    repository, client = _repository(firestore_fake)
    intake = _intake(repository)
    event = _update(41, forward=forward)
    event["message"]["message_thread_id"] = invalid_thread_id

    assert intake.handle_update(event) is None
    assert repository.get_context("42_default") is None
    assert not any("42_default" in path for path in firestore_fake.documents)


def test_missing_thread_id_on_topic_service_event_does_not_update_default(firestore_fake):
    repository, _ = _repository(firestore_fake)
    intake = _intake(repository)
    event = _topic_event(42, thread_id=None, name="Orphan topic")
    del event["message"]["message_thread_id"]

    assert intake.handle_update(event) is None
    assert repository.get_context("42_default") is None


def test_explicit_null_thread_id_routes_ordinary_update_to_default(firestore_fake):
    repository, _ = _repository(firestore_fake)
    intake = _intake(repository)
    event = _update(43)
    event["message"]["message_thread_id"] = None

    accepted = intake.handle_update(event)

    assert _value(accepted, "accepted") is True
    assert _value(accepted, "context_id") == "42_default"


@pytest.mark.parametrize(
    ("update_id", "topic_field", "topic_payload"),
    [
        (50, "forum_topic_created", {"name": "Created topic"}),
        (51, "forum_topic_edited", {"name": "Edited topic"}),
    ],
)
def test_topic_service_events_update_metadata_without_becoming_source_or_command(
    firestore_fake, update_id, topic_field, topic_payload,
):
    repository, client = _repository(firestore_fake)
    intake = _intake(repository)
    intake.handle_update(_topic_event(48, thread_id=11, name="Original topic"))
    seed = intake.handle_update(_update(49, thread_id=11))
    context_id = _value(seed, "context_id")
    draft_id = _value(seed, "draft_id")
    before_messages = repository.list_messages(context_id, draft_id)
    before_ordered_message_ids = repository.get_draft(context_id, draft_id)[
        "ordered_message_ids"
    ]
    side_effect_record_kinds = (
        "draft", "message", "dialog", "command", "operation",
        "notification", "task", "dispatch",
    )
    before_side_effect_paths = {
        path for path in firestore_fake.documents
        if any(
            kind in segment.lower().replace("_", "").replace("-", "")
            for kind in side_effect_record_kinds
            for segment in path.split("/")
        )
    }
    event = {
        "update_id": update_id,
        "message": {
            "message_id": update_id + 100,
            "chat": {"id": 42, "type": "private"},
            "from": {"id": 7},
            "message_thread_id": 11,
            topic_field: topic_payload,
        },
    }

    intake.handle_update(event)

    context = repository.get_context(context_id)
    assert context["next_dialog_number"] == 2
    assert context["next_operation_sequence"] == 1
    assert (
        repository.get_draft(context_id, draft_id)["ordered_message_ids"]
        == before_ordered_message_ids
        == [149]
    )
    assert repository.list_messages(context_id, draft_id) == before_messages
    after_side_effect_paths = {
        path for path in firestore_fake.documents
        if any(
            kind in segment.lower().replace("_", "").replace("-", "")
            for kind in side_effect_record_kinds
            for segment in path.split("/")
        )
    }
    assert after_side_effect_paths <= before_side_effect_paths
    assert client.collection("telegram_sessions").document("42_11").get().exists
    assert context["topic_name"] == topic_payload["name"]


@pytest.mark.parametrize("value, expected", [(True, True), (False, False), (None, None), ("true", None), (1, None)])
def test_optional_getme_topic_capability_is_pure_informational_normalization(value, expected):
    contexts = importlib.import_module("services.connectors.telegram.contexts")

    assert contexts.normalize_has_topics_enabled(value) is expected


def test_topic_event_before_first_forward_is_durable_and_replay_is_idempotent(firestore_fake):
    repository, _ = _repository(firestore_fake)
    publisher = _Publisher()
    intake = importlib.import_module("services.connectors.telegram.intake").TelegramIntake(
        repository=repository, publisher=publisher, allowed_user_ids={7}, clock=lambda: NOW,
    )
    event = _topic_event(60, thread_id=33, name="Early topic")

    first = intake.handle_update(event)
    replay = intake.handle_update(event)

    assert _value(first, "accepted") is False
    assert _value(replay, "accepted") is False
    assert _value(first, "context_id") == _value(replay, "context_id") == "42_33"
    receipt = repository.get_update_receipt("42_33", 60)
    assert receipt["event_kind"] == "forum_topic_created"
    assert receipt["outcome"] == "topic_metadata_updated"
    assert receipt["intake_sequence"] is None
    assert repository.get_context("42_33")["topic_name"] == "Early topic"
    assert publisher.outbox_ids == []
    assert not any("/drafts/" in path or "/operations/" in path or "/commands/" in path
                   for path in firestore_fake.documents)

    first_forward = intake.handle_update(_update(61, thread_id=33))
    assert _value(first_forward, "accepted") is True
    assert repository.get_context("42_33")["topic_name"] == "Early topic"
    assert repository.get_context("42_33")["next_intake_sequence"] == 2


def test_topic_edit_without_name_preserves_hint(firestore_fake):
    repository, _ = _repository(firestore_fake)
    intake = _intake(repository)
    intake.handle_update(_topic_event(62, thread_id=34, name="Known"))

    intake.handle_update(_topic_event(63, thread_id=34, event="forum_topic_edited"))

    assert repository.get_context("42_34")["topic_name"] == "Known"
    receipt = repository.get_update_receipt("42_34", 63)
    assert receipt["outcome"] == "topic_metadata_preserved"
