from datetime import datetime, timezone

import pytest

from services.connectors.telegram.conversations import (
    build_agent_prompt,
    build_conversation,
    build_turn_payload,
)
from services.connectors.telegram.context import build_conversation_history
from services.connectors.telegram.repository import TelegramRepository


NOW = datetime(2026, 10, 26, tzinfo=timezone.utc)


def test_conversation_kinds_enforce_their_target_shapes():
    draft = build_conversation(
        "ctx-1", "draft", draft_id="draft-1", dialog_number=1, now=NOW,
    )
    document = build_conversation(
        "ctx-2", "document", doc_id="doc1", dialog_number=2, now=NOW,
    )
    search = build_conversation(
        "search-1", "search", search_filters={"folder": "notes"}, now=NOW,
    )
    chat = build_conversation("chat-1", "chat", now=NOW)

    assert (draft.kind.value, draft.draft_id, draft.doc_id, draft.dialog_number) == (
        "draft", "draft-1", None, 1,
    )
    assert (document.kind.value, document.draft_id, document.doc_id, document.dialog_number) == (
        "document", None, "doc1", 2,
    )
    assert (search.draft_id, search.doc_id, search.dialog_number) == (None, None, None)
    assert search.search_filters == {"folder": "notes"}
    assert (chat.kind.value, chat.draft_id, chat.doc_id, chat.dialog_number) == (
        "chat", None, None, None,
    )
    with pytest.raises(ValueError):
        build_conversation("bad", "search", dialog_number=1, now=NOW)
    with pytest.raises(ValueError):
        build_conversation("bad", "draft", doc_id="doc1", draft_id="draft-1",
                           dialog_number=1, now=NOW)
    with pytest.raises(ValueError):
        build_turn_payload("ctx-1", turn_id="1", operation_id="op", role="system",
                           parts=[], now=NOW)


def test_agent_prompt_uses_only_canonical_conversation_roles_and_request():
    history = [{
        "role": "tool",
        "turn_id": "private-turn-id",
        "operation_id": "private-operation-id",
        "created_at": NOW,
        "raw_envelope": "discard this metadata",
        "parts": [
            {"text": "prior request"},
            {"function_response": {"name": "lookup", "response": {"ok": True}}},
        ],
    }]
    prompt = build_agent_prompt(history, "selected target question")
    assert prompt == {
        "history": [{
            "role": "tool",
            "parts": [
                {"text": "prior request"},
                {"function_response": {"name": "lookup", "response": {"ok": True}}},
            ],
        }],
        "request": "selected target question",
    }
    assert history[0]["turn_id"] == "private-turn-id"
    with pytest.raises(ValueError):
        build_agent_prompt([{"role": "assistant", "parts": []}], "question")


@pytest.mark.parametrize("bad_part", ["malformed", {"nested": object()}])
def test_agent_prompt_fails_closed_on_malformed_stored_parts(bad_part):
    with pytest.raises(TypeError):
        build_agent_prompt([{"role": "tool", "parts": [bad_part]}], "question")


@pytest.mark.parametrize("turn", [
    {"role": "tool", "parts": "malformed", "content": "must not fallback"},
    {"role": "tool", "parts": None, "content": "must not fallback"},
    {"role": "tool"},
    {"role": "tool", "content": None},
])
def test_agent_prompt_fails_closed_on_missing_or_malformed_top_level_payload(turn):
    with pytest.raises(TypeError):
        build_agent_prompt([turn], "question")


def test_agent_prompt_preserves_legacy_string_content_fallback():
    prompt = build_agent_prompt([{"role": "user", "content": "legacy turn"}], "question")
    assert prompt["history"] == [{"role": "user", "parts": [{"text": "legacy turn"}]}]


def test_conversation_history_keeps_all_ordered_turns_and_rejects_malformed_rows():
    history = [
        {"conversation_id": "chat", "role": role,
         "parts": [{"text": f"turn-{index}"}], "turn_sequence": index}
        for index, role in enumerate(("user", "model", "user", "model", "tool", "user"), 1)
    ]
    projected = build_conversation_history(history, conversation_id="chat")
    assert [row["role"] for row in projected] == [
        "user", "model", "user", "model", "user", "user",
    ]
    assert ["turn-1" in row["parts"][0]["text"] for row in projected] == [
        True, False, False, False, False, False,
    ]
    assert all(f"turn-{index}" in projected[index - 1]["parts"][0]["text"]
               for index in (2, 3, 4, 5, 6))
    with pytest.raises(ValueError, match="unsupported"):
        build_conversation_history(
            [*history, {"conversation_id": "chat", "role": "system", "parts": []}],
            conversation_id="chat",
        )
    with pytest.raises(ValueError, match="session boundary"):
        build_conversation_history(
            [*history, {"conversation_id": "other", "role": "user", "parts": []}],
            conversation_id="chat",
        )


def test_hydrated_search_history_projects_opaque_signature_as_untrusted_json():
    history = [{
        "conversation_id": "search-session", "operation_id": "prior-operation",
        "role": "tool", "parts": [{
            "type": "search_exchange_ref", "call_id": "call-1", "ordinal": 0,
            "checkpoint_conversation_id": "private-checkpoint", "reference": {"version": 1},
        }],
    }]

    projected = build_conversation_history(
        history, conversation_id="search-session",
        search_exchange_loader=lambda *_args, **_kwargs: {
            "call": {"thought_signature": b"opaque\\x00signature"},
            "result": {"segments": []},
        },
    )

    text = projected[0]["parts"][0]["text"]
    assert projected[0]["role"] == "user"
    assert {row["role"] for row in projected} <= {"user", "model"}
    assert "Untrusted prior tool data" in text
    assert "opaque_bytes_base64" in text
    assert "functionCall" not in text


def test_draft_session_record_is_separate_and_creation_is_idempotent(firestore_fake):
    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    intake = repo.accept_forward(
        1, chat_id=42, thread_id=5,
        message={"message_id": 101, "text": "hello"}, now=NOW,
    )

    first = repo.ensure_conversation(
        intake.context_id, f"{intake.context_id}-1", kind="draft",
        draft_id=intake.draft_id, dialog_number=1, now=NOW,
    )
    replay = repo.ensure_conversation(
        intake.context_id, f"{intake.context_id}-1", kind="draft",
        draft_id=intake.draft_id, dialog_number=1,
        now=datetime(2026, 10, 27, tzinfo=timezone.utc),
    )
    assert first == replay
    assert repo.get_conversation(intake.context_id, first.conversation_id) == first
    dialog = client.collection("telegram_sessions").document(intake.context_id).collection(
        "dialogs"
    ).document("1").get().to_dict()
    session = client.collection("telegram_sessions").document(intake.context_id).collection(
        "conversations"
    ).document(first.conversation_id).get().to_dict()
    assert "kind" not in dialog
    assert "last_turn_sequence" not in dialog
    assert session["kind"] == "draft"
    with pytest.raises(ValueError, match="numbered dialog"):
        repo.ensure_conversation(
            intake.context_id, first.conversation_id, kind="draft",
            draft_id="other-draft", dialog_number=1, now=NOW,
        )


def test_search_conversation_and_turns_are_ordered_idempotent_and_unbounded(firestore_fake):
    client = firestore_fake.client_class()
    repo = TelegramRepository(client)
    search = repo.ensure_conversation(
        "42_5", "search-42-5", kind="search", search_filters=None, now=NOW,
    )
    assert search.kind.value == "search"
    assert search.draft_id is search.doc_id is search.dialog_number is None
    session = client.collection("telegram_sessions").document("42_5").collection(
        "conversations"
    ).document(search.conversation_id).get()
    assert session.exists
    assert not client.collection("telegram_sessions").document("42_5").collection(
        "dialogs"
    ).document(search.conversation_id).get().exists

    first = repo.append_turn(
        "42_5", search.conversation_id, turn_id="turn-1", operation_id="op-1",
        role="user", parts=[{"text": "question"}], now=NOW,
    )
    duplicate = repo.append_turn(
        "42_5", search.conversation_id, turn_id="turn-1", operation_id="op-1",
        role="user", parts=[{"text": "question"}], now=datetime(2026, 10, 27, tzinfo=timezone.utc),
    )
    assert duplicate == first
    with pytest.raises(ValueError, match="different content"):
        repo.append_turn(
            "42_5", search.conversation_id, turn_id="turn-1", operation_id="op-1",
            role="user", parts=[{"text": "changed"}], now=NOW,
        )

    for index in range(2, 82):
        repo.append_turn(
            "42_5", search.conversation_id, turn_id=f"turn-{index}",
            operation_id=f"op-{index}", role="model", parts=[{"text": "x" * 1200}],
            source_refs=[{"id": f"source-{index}"}], now=NOW,
        )
    rows = repo.list_turns("42_5", search.conversation_id)
    assert len(rows) == 81
    assert [row["turn_sequence"] for row in rows] == list(range(1, 82))
    assert len(rows[-1]["parts"][0]["text"]) == 1200
    assert repo.get_conversation("42_5", search.conversation_id).last_substantive_at == NOW
    assert client.collection("telegram_sessions").document("42_5").collection(
        "conversations"
    ).document(search.conversation_id).collection("turns").document("turn-81").get().exists


def test_tool_turn_role_persists_expected_ordered_fields(firestore_fake):
    repo = TelegramRepository(firestore_fake.client_class())
    conversation = repo.ensure_conversation(
        "42_5", "search-tools", kind="search", now=NOW,
    )
    turn = repo.append_turn(
        "42_5", conversation.conversation_id, turn_id="tool-turn",
        operation_id="operation-1", role="tool",
        parts=[{"function_call_id": "call-opaque-1", "result": {"count": 3}}],
        tool_correlation={"call_id": "call-opaque-1"},
        source_refs=[{"doc_id": "doc-1"}], now=NOW,
    )

    assert turn == {
        "turn_id": "tool-turn",
        "conversation_id": "search-tools",
        "operation_id": "operation-1",
        "role": "tool",
        "parts": [{"function_call_id": "call-opaque-1", "result": {"count": 3}}],
        "tool_correlation": {"call_id": "call-opaque-1"},
        "source_refs": [{"doc_id": "doc-1"}],
        "created_at": NOW,
        "turn_sequence": 1,
    }
    assert repo.list_turns("42_5", conversation.conversation_id) == [turn]


def test_document_binding_reuses_number_and_rejects_conversation_repoint(firestore_fake):
    repo = TelegramRepository(firestore_fake.client_class())
    number = repo.bind_document(
        "42_5", "doc-id", "42_5-1", metadata={"title": "Document"}, now=NOW,
    )
    assert number == 1
    assert repo.bind_document(
        "42_5", "doc-id", "42_5-1", metadata={"title": "Changed"}, now=NOW,
    ) == number
    with pytest.raises(ValueError, match="different conversation"):
        repo.bind_document(
            "42_5", "doc-id", "42_5-2", metadata={}, now=NOW,
        )
    record = repo.ensure_conversation(
        "42_5", "42_5-1", kind="document", doc_id="doc-id",
        dialog_number=number, now=NOW,
    )
    assert (record.doc_id, record.dialog_number) == ("doc-id", 1)


def test_saved_document_annotation_is_bound_to_its_conversation_and_kept_separate(
    firestore_fake,
):
    client = firestore_fake.client_class()
    repository = TelegramRepository(client)
    context_id, operation_id, conversation_id = "telegram-context", "annotation-op", "doc-conv"
    owner, fence, doc_id = "queue-owner", 4, "imported-doc"
    expiry = NOW.replace(day=27)
    context_ref = client.collection("telegram_sessions").document(context_id)
    conversation = build_conversation(
        conversation_id, "document", doc_id=doc_id, dialog_number=3, now=NOW,
    )
    context_ref.set({
        "reserved_operation_id": operation_id, "active_operation_id": operation_id,
        "lock_owner": owner, "fence": fence, "lock_expires_at": expiry,
    })
    context_ref.collection("operations").document(operation_id).set({
        "operation_id": operation_id, "state": "running",
        "input_ref": f"telegram_sessions/{context_id}/commands/1",
        "route_control": {
            "status": "ok", "kind": "annotation", "target_doc_id": doc_id,
            "target_conversation_id": conversation_id,
            "annotation_text": "Dmitry means Project Y by ‘that thing’. ",
        },
    })
    context_ref.collection("commands").document("1").set({
        "context_id": context_id, "operation_id": operation_id,
    })
    context_ref.collection("conversations").document(conversation_id).set(
        conversation.to_dict(),
    )
    text = "Dmitry means Project Y by ‘that thing’. "
    staged = repository.stage_conversation_annotation(
        context_id, operation_id, owner=owner, fence=fence,
        conversation_id=conversation_id, doc_id=doc_id, text=text, now=NOW,
    )
    assert staged is not None
    assert staged["state"] == "pending_fact"
    assert staged["provenance"]["kind"] == "telegram_user_annotation"
    assert repository.list_conversation_annotations(context_id, conversation_id) == [staged]
    assert repository.list_conversation_annotations(context_id, "unrelated-conversation") == []
    assert repository.settle_conversation_annotation(
        context_id, conversation_id, staged["comment_id"],
        operation_id=operation_id, owner=owner, fence=fence,
        state="applied", now=NOW,
    )
    visible = repository.list_conversation_annotations(context_id, conversation_id)
    assert visible[0]["state"] == "applied"
    assert visible[0]["text"] == text


def test_annotation_payload_roundtrips_long_unicode_and_rejects_foreign_corrupt_or_stale_refs(
    firestore_fake,
):
    import hashlib

    from services.connectors.telegram.repository._annotation_payloads import (
        load_annotation_payload,
    )
    from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger

    client = firestore_fake.client_class()
    context_id, operation_id, owner, fence = "42_5", "annotation-ledger", "ledger-owner", 5
    expiry = NOW.replace(day=27)
    context_ref = client.collection("telegram_sessions").document(context_id)
    context_ref.set({
        "reserved_operation_id": operation_id, "active_operation_id": operation_id,
        "lock_owner": owner, "fence": fence, "lock_expires_at": expiry,
    })
    context_ref.collection("operations").document(operation_id).set({
        "operation_id": operation_id, "state": "running",
        "input_ref": f"telegram_sessions/{context_id}/commands/1",
    })
    context_ref.collection("commands").document("1").set({
        "context_id": context_id, "operation_id": operation_id,
    })
    ledger = FirestoreAgentLedger(
        client, context_id=context_id, operation_id=operation_id,
        owner=owner, fence=fence, clock=lambda: NOW,
    )
    text = "Контекст 🔒. " * 5000
    stored = ledger._store_payload("annotation", text)
    assert "_input_ref" in stored
    operation_ref = context_ref.collection("operations").document(operation_id)
    operation_ref.update({"route_control": {
        "status": "ok", "kind": "annotation", "annotation_payload": stored,
        "annotation_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }})
    row = {
        "context_id": context_id, "operation_id": operation_id,
        "text_payload": stored,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "provenance": {"kind": "telegram_user_annotation"},
    }
    repository = TelegramRepository(client)
    assert repository._annotation_row(row)["text"] == text
    from services.connectors.telegram.runtime.annotations import hydrate_annotation_text
    assert hydrate_annotation_text(
        client, row, context_id=context_id, operation_id=operation_id,
        owner=owner, fence=fence, clock=lambda: NOW,
    ) == text
    assert load_annotation_payload(
        client, {**stored, "sha256": "0" * 64}, context_id=context_id,
        operation_id=operation_id,
    ) is None
    assert load_annotation_payload(client, {
        **stored, "_input_ref": "dialog_operations/foreign/input_parts/manifest",
    }, context_id=context_id, operation_id=operation_id) is None
    assert load_annotation_payload(
        client, stored, context_id="42_6", operation_id="other-accepted-operation",
    ) is None
    other_operation_id = "other-accepted-operation"
    context_ref.collection("operations").document(other_operation_id).set({
        "operation_id": other_operation_id, "state": "completed",
        "input_ref": f"telegram_sessions/{context_id}/commands/2",
        "route_control": {
            "status": "ok", "kind": "annotation", "annotation_payload": stored,
            "annotation_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        },
    })
    context_ref.collection("commands").document("2").set({
        "context_id": context_id, "operation_id": other_operation_id,
    })
    other_row = {**row, "operation_id": other_operation_id}
    assert repository._annotation_row(other_row).get("text") is None
    foreign_operation_id = "foreign-accepted-operation"
    foreign_context = client.collection("telegram_sessions").document("42_6")
    foreign_context.collection("operations").document(foreign_operation_id).set({
        "operation_id": foreign_operation_id, "state": "completed",
        "input_ref": "telegram_sessions/42_6/commands/2",
        "route_control": {
            "status": "ok", "kind": "annotation", "annotation_payload": stored,
            "annotation_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        },
    })
    foreign_context.collection("commands").document("2").set({
        "context_id": "42_6", "operation_id": foreign_operation_id,
    })
    foreign_row = {
        **row, "context_id": "42_6", "operation_id": foreign_operation_id,
    }
    assert TelegramRepository(client)._annotation_row(foreign_row).get("text") is None
    context_ref.update({"lock_owner": "new-owner", "fence": fence + 1})
    assert hydrate_annotation_text(
        client, row, context_id=context_id, operation_id=operation_id,
        owner=owner, fence=fence, clock=lambda: NOW,
    ) is None
    with pytest.raises(PermissionError, match="stale"):
        ledger._store_payload("annotation-replay", text + "new")


def test_terminal_annotation_failure_replays_the_same_notice_after_lost_ack(firestore_fake):
    import asyncio
    import hashlib
    from datetime import timedelta
    from types import SimpleNamespace

    from core.integrations.firestore.save_intents import SAVE_INTENTS_COLLECTION
    from core.operations.queue import QueueOperationState
    from services.connectors.telegram.runtime.annotations import _fail_application

    client = firestore_fake.client_class()
    context_id, draft_id, doc_id = "annotation-failure", "draft-a", "doc-a"
    save_operation_id, fact_operation_id = "save-a", "fact-a"
    intent_id, comment_id = "intent-a", "comment-a"
    owner, fence = "annotation-worker", 8
    now = [NOW]
    expiry = NOW + timedelta(minutes=5)
    context_ref = client.collection("telegram_sessions").document(context_id)
    context_ref.set({
        "reserved_operation_id": fact_operation_id,
        "active_operation_id": fact_operation_id,
        "chat_id": 77,
        "lock_owner": owner, "fence": fence, "lock_expires_at": expiry,
    })
    context_ref.collection("operations").document(fact_operation_id).set({
        "operation_id": fact_operation_id, "state": "running",
        "telegram_internal_kind": "annotation_apply", "draft_id": draft_id,
        "annotation_id": comment_id,
        "input_ref": f"telegram_sessions/{context_id}/drafts/{draft_id}/annotations/{comment_id}",
    })
    digest, content_hash = "d" * 64, "c" * 64
    receipt_ref = f"index_completions/{doc_id}/generations/{content_hash}"
    binding = {"context_id": context_id, "draft_id": draft_id, "manifest_digest": digest}
    cleanup = {
        "certificate_id": "certificate-a", "save_intent_id": intent_id,
        "operation_id": save_operation_id, "origin_binding": binding,
    }
    local_certificate = {
        **binding, "save_intent_id": intent_id, "operation_id": save_operation_id,
        "receipt_ref": receipt_ref, "receipt_version": "receipt-v1",
        "origin_binding": (context_id, draft_id, digest), "doc_id": doc_id,
    }
    shared_certificate = {
        "operation_id": save_operation_id, "save_intent_id": intent_id,
        "doc_id": doc_id, "content_hash": content_hash,
        "receipt_ref": receipt_ref, "receipt_version": "receipt-v1",
        "certificate_id": "certificate-a", "origin_binding": binding,
    }
    context_ref.collection("drafts").document(draft_id).set({
        "draft_id": draft_id, "manifest_digest": digest,
        "active_save_operation_id": save_operation_id,
        "immutable_save_intent_id": intent_id,
        "completion_certificate_ref": f"{SAVE_INTENTS_COLLECTION}/{intent_id}",
        "completion_certificate": local_certificate, "dialog_number": 4,
    })
    client.collection("dialog_operations").document(save_operation_id).set({
        "operation_id": save_operation_id, "state": "completed", "doc_id": doc_id,
        "save_intent_id": intent_id, "origin_binding": binding,
        "cleanup_authorization": cleanup,
    })
    client.collection(SAVE_INTENTS_COLLECTION).document(intent_id).set({
        "doc_id": doc_id, "cleanup_authorization": cleanup,
        "completion_certificate": shared_certificate,
    })
    client.collection("index_completions").document(doc_id).collection(
        "generations",
    ).document(content_hash).set({
        "status": "completed", "doc_id": doc_id, "content_hash": content_hash,
        "version": "receipt-v1",
    })
    comment_ref = (context_ref.collection("drafts").document(draft_id)
                   .collection("annotations").document(comment_id))
    comment_ref.set({
        "comment_id": comment_id, "context_id": context_id, "draft_id": draft_id,
        "application_operation_id": fact_operation_id, "applied_doc_id": doc_id,
        "state": "queued_fact",
    })

    class Telegram:
        def __init__(self):
            self.calls = []

        async def send_message(self, chat_id, text, **_kwargs):
            self.calls.append((chat_id, text))
            return {} if len(self.calls) == 1 else {"message_id": 902}

    bot = Telegram()
    processor = SimpleNamespace(
        repository=TelegramRepository(client), _client=lambda: bot,
        clock=lambda: now[0],
    )
    lease = SimpleNamespace(owner=owner, fence=fence)
    result = asyncio.run(_fail_application(
        processor, context_id=context_id, operation_id=fact_operation_id,
        lease=lease, draft_id=draft_id, comment_id=comment_id,
        doc_id=doc_id, dialog_number=4, error_code="index_unavailable",
    ))
    assert result is QueueOperationState.RUNNING
    failed_note = comment_ref.get().to_dict()
    assert failed_note["state"] == "failed"
    assert failed_note["error_code"] == "index_unavailable"
    notification_id = "tgannotation-error-" + hashlib.sha256(
        comment_id.encode("utf-8"),
    ).hexdigest()
    notification = processor.repository.get_notification(notification_id)
    assert notification["state"] == "retry"
    assert notification["last_error"] == "unknown"
    body = notification["body"]
    assert bot.calls == [(77, body)]

    now[0] = notification["retry_not_before"]
    replay = asyncio.run(_fail_application(
        processor, context_id=context_id, operation_id=fact_operation_id,
        lease=lease, draft_id=draft_id, comment_id=comment_id,
        doc_id=doc_id, dialog_number=4, error_code="index_unavailable",
    ))
    assert replay is QueueOperationState.COMPLETED
    assert bot.calls == [(77, body), (77, body)]
    assert processor.repository.get_notification(notification_id)["state"] == "delivered"
