from datetime import datetime, timedelta, timezone
import hashlib

import pytest
from google.cloud import firestore

from services.connectors.telegram.conversations import ConversationKind, ConversationRecord
from services.connectors.telegram.commands import parse_command
from services.connectors.telegram.repository import TelegramRepository
from services.connectors.telegram.runtime.presentation import build_presentation_manifest
from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger
from tests.fixtures.telegram_contextual import guard_firestore_enumeration


NOW = datetime(2026, 11, 18, tzinfo=timezone.utc)

pytestmark = pytest.mark.offline


def test_legacy_conversation_defaults_keep_serialized_shape():
    legacy = {
        "conversation_id": "search-1",
        "kind": "search",
        "draft_id": None,
        "doc_id": None,
        "dialog_number": None,
        "search_filters": None,
        "created_at": NOW,
        "last_substantive_at": None,
    }

    record = ConversationRecord.from_dict(legacy)

    assert record.last_presented_list_id is None
    assert record.last_presentation_id is None
    assert record.to_dict() == {**legacy, "last_turn_sequence": 0}
    # Existing positional construction remains valid.
    positional = ConversationRecord(
        "search-1", ConversationKind.SEARCH, None, None, None, None, NOW, None,
    )
    assert positional.last_presented_list_id is None
    opaque_list_id = "🔎/weekly results 7"
    record_with_pointer = ConversationRecord.from_dict({
        **legacy, "last_presented_list_id": opaque_list_id,
        "last_presentation_id": "presentation-1",
    })
    assert record_with_pointer.last_presented_list_id == opaque_list_id
    assert record_with_pointer.to_dict()["last_presented_list_id"] == opaque_list_id


def _seed_presentation_context(client, *, context_id, conversation_id, operation_id,
                               owner, fence, body):
    context_ref = client.collection("telegram_sessions").document(context_id)
    context_ref.set({
        "reserved_operation_id": operation_id,
        "active_operation_id": operation_id,
        "lock_owner": owner,
        "fence": fence,
        "lock_expires_at": NOW + timedelta(minutes=5),
        "chat_id": 42,
        "thread_id": 5,
    })
    context_ref.collection("operations").document(operation_id).set({
        "operation_id": operation_id,
        "state": "running",
        "input_ref": f"telegram_sessions/{context_id}/commands/1",
    })
    context_ref.collection("commands").document("1").set({
        "context_id": context_id,
        "operation_id": operation_id,
        "user_id": 42,
        "kind": "freeform",
        "text": "show the results",
        "candidate": "show the results",
    })
    repository = TelegramRepository(client)
    conversation = repository.ensure_conversation(
        context_id, conversation_id, kind="search", now=NOW,
    )
    context_ref.update({"active_conversation_id": conversation_id})
    return context_ref, repository, conversation, body


def _seed_source_from_distinct_operation(client, repository, *, context_id,
                                         conversation_id, source_operation_id,
                                         source_user_id, turn_id, call_id):
    command_ref = client.collection("telegram_sessions").document(context_id).collection(
        "commands",
    ).document("2")
    command_ref.set({
        "context_id": context_id, "operation_id": source_operation_id,
        "user_id": source_user_id, "kind": "freeform", "candidate": "earlier search",
    })
    source_op_ref = client.collection("telegram_sessions").document(context_id).collection(
        "operations",
    ).document(source_operation_id)
    source_op_ref.set({
        "operation_id": source_operation_id, "context_id": context_id,
        "input_ref": f"telegram_sessions/{context_id}/commands/2",
    })
    checkpoint_conversation_id = "checkpoint-source-op"
    ledger = FirestoreAgentLedger(
        client, context_id=context_id, operation_id=source_operation_id,
        owner="source-owner", fence=1, clock=lambda: NOW,
    )
    call = {
        "id": call_id, "name": "list_recent_transcripts",
        "arguments": {"limit": 1, "offset": 0}, "ordinal": 0, "round": 0,
    }
    result = {
        "total": 1, "returned": 1, "offset": 0, "limit": 1,
        "has_more": False, "transcripts": [{"doc_id": "source-doc", "title": "Source"}],
    }
    reference = ledger.save_search_exchange(
        conversation_id, checkpoint_conversation_id=checkpoint_conversation_id,
        call=call, canonical_call_id=call_id, result=result,
    )
    checkpoint_ref = source_op_ref.collection("agent_checkpoints").document("turn")
    checkpoint_ref.set({
        "conversation_id": checkpoint_conversation_id,
        "tool_calls": {"_inline": [call]},
        "application_route": None,
    })
    checkpoint_ref.collection("tool_results").document("result-00000000").set({
        "ordinal": 0, "result": {"_inline": result},
    })
    repository.append_turn(
        context_id, conversation_id, turn_id=turn_id,
        operation_id=source_operation_id, role="tool",
        parts=[{"type": "search_exchange_ref", "reference": reference}], now=NOW,
    )
    exchange = ledger.load_search_exchange(
        reference, conversation_id=conversation_id, operation_id=source_operation_id,
        checkpoint_conversation_id=checkpoint_conversation_id,
        call_id=call_id, ordinal=0,
    )
    return reference, exchange


def test_manifest_roundtrip_reuses_answer_turn_and_is_immutable(firestore_fake):
    client = firestore_fake.client_class()
    context_id, conversation_id = "42_5", "search-42-5"
    operation_id, owner, fence = "present-op", "present-owner", 7
    answer_turn_id = "search-answer-" + hashlib.sha256(
        f"{operation_id}".encode("utf-8"),
    ).hexdigest()
    scope = {
        "context_id": context_id,
        "conversation_id": conversation_id,
        "producing_operation_id": operation_id,
        "presentation_id": "presentation-1",
        "answer_turn_id": answer_turn_id,
    }
    manifest = build_presentation_manifest(
        scope=scope,
        blocks=[{"type": "paragraph", "text": "Результаты 🔎", "source_refs": []}],
        verified_exchanges=[],
        notification_refs=[{"notification_id": "notice-1", "revision": 1}],
        revision_refs=[{"notification_id": "notice-1", "revision": 1}],
    ).to_dict()
    context_ref, repository, conversation, _ = _seed_presentation_context(
        client, context_id=context_id, conversation_id=conversation_id,
        operation_id=operation_id, owner=owner, fence=fence, body=manifest["body"],
    )
    turn = repository.append_turn(
        context_id, conversation_id, turn_id=scope["answer_turn_id"],
        operation_id=operation_id, role="model", parts=[{"text": manifest["body"]}],
        now=NOW,
    )
    assert turn["parts"] == [{"text": manifest["body"]}]

    saved = repository.save_presentation_manifest(
        context_id, manifest, queue_operation_id=operation_id,
        queue_owner=owner, queue_fence=fence, now=NOW, clock=lambda: NOW,
    )
    replay = repository.save_presentation_manifest(
        context_id, manifest, queue_operation_id=operation_id,
        queue_owner=owner, queue_fence=fence, now=NOW, clock=lambda: NOW,
    )

    assert saved == replay == manifest
    assert repository.get_presentation_manifest(
        context_id, conversation_id, "presentation-1",
    ) == manifest
    stored = context_ref.collection("conversations").document(conversation_id).collection(
        "presentations",
    ).document("presentation-1").get().to_dict()
    assert "body" not in stored
    assert stored["body_ref"] == {
        "turn_id": scope["answer_turn_id"],
        "turn_sequence": turn["turn_sequence"],
        "body_sha256": manifest["body_sha256"],
    }
    current = repository.get_conversation(context_id, conversation_id)
    assert current.last_presented_list_id is None
    assert current.last_presentation_id is None
    second_scope = {**scope, "presentation_id": "presentation-2",
                    "answer_turn_id": "answer-turn-2"}
    second_manifest = build_presentation_manifest(
        scope=second_scope,
        blocks=[{"type": "paragraph", "text": "A second answer", "source_refs": []}],
        verified_exchanges=[],
    ).to_dict()
    repository.append_turn(
        context_id, conversation_id, turn_id=second_scope["answer_turn_id"],
        operation_id=operation_id, role="model", parts=[{"text": second_manifest["body"]}],
        now=NOW,
    )
    with pytest.raises(ValueError, match="answer.turn|presentation|bound"):
        repository.save_presentation_manifest(
            context_id, second_manifest, queue_operation_id=operation_id,
            queue_owner=owner, queue_fence=fence, now=NOW, clock=lambda: NOW,
        )
    assert repository.get_presentation_manifest(
        context_id, conversation_id, "presentation-1",
    ) == manifest

    # Once linked, the canonical answer body cannot be rewritten by the
    # existing recovery path. The immutable artifact stays readable after its
    # producer is terminal and current selection moves elsewhere.
    with pytest.raises(ValueError, match="presentation"):
        repository.replace_model_turn_text(
            context_id, conversation_id, operation_id, "replacement body",
            queue_owner=owner, queue_fence=fence, now=NOW,
        )
    context_ref.collection("operations").document(operation_id).update({"state": "completed"})
    context_ref.update({"active_conversation_id": "another-conversation"})
    context_ref.collection("commands").document("1").update({
        "candidate": firestore.DELETE_FIELD,
    })
    assert repository.get_presentation_manifest(
        context_id, conversation_id, "presentation-1",
    ) == manifest

    changed_scope = {**scope, "answer_turn_id": "answer-turn-3"}
    conflicting = build_presentation_manifest(
        scope=changed_scope,
        blocks=[{"type": "paragraph", "text": "Different answer", "source_refs": []}],
        verified_exchanges=[],
    ).to_dict()
    repository.append_turn(
        context_id, conversation_id, turn_id=changed_scope["answer_turn_id"],
        operation_id=operation_id, role="model", parts=[{"text": conflicting["body"]}],
        now=NOW,
    )
    with pytest.raises(ValueError, match="immutable|already bound|presentation"):
        repository.save_presentation_manifest(
            context_id, conflicting, queue_operation_id=operation_id,
            queue_owner=owner, queue_fence=fence, now=NOW, clock=lambda: NOW,
        )

    # A malformed addressed value fails closed. It is never replaced with a
    # fresh interpretation of mutable conversation state.
    stored["manifest_payload"] = {"_inline": {"not": "a manifest"},
                                   "sha256": "0" * 64, "byte_length": 1}
    context_ref.collection("conversations").document(conversation_id).collection(
        "presentations",
    ).document("presentation-1").set(stored)
    with pytest.raises((TypeError, ValueError)):
        repository.get_presentation_manifest(
            context_id, conversation_id, "presentation-1",
        )


def test_real_accepted_queue_lifecycle_preserves_addressed_retry(firestore_fake):
    from core.integrations.firestore.operation_queue import claim_head, complete_head

    client = firestore_fake.client_class()
    repository = TelegramRepository(client)
    accepted = repository.accept_command(
        9001, chat_id=42, user_id=42, thread_id=5,
        command=parse_command("show a concise answer"), now=NOW,
    )
    context_ref = client.collection("telegram_sessions").document(accepted.context_id)
    lease = claim_head(client, context_ref, owner="real-presentation-worker", now=NOW)
    assert lease is not None
    conversation_id = "real-search-session"
    repository.ensure_conversation(
        accepted.context_id, conversation_id, kind="search",
        queue_operation_id=accepted.operation_id, queue_owner=lease.owner,
        queue_fence=lease.fence, select_if_unselected=True, now=NOW,
    )
    answer_turn_id = "search-answer-" + hashlib.sha256(
        accepted.operation_id.encode("utf-8"),
    ).hexdigest()
    manifest = build_presentation_manifest(
        scope={
            "context_id": accepted.context_id, "conversation_id": conversation_id,
            "producing_operation_id": accepted.operation_id,
            "presentation_id": "real-lifecycle-presentation",
            "answer_turn_id": answer_turn_id,
        },
        blocks=[{"type": "paragraph", "text": "A real accepted answer", "source_refs": []}],
        verified_exchanges=[],
    ).to_dict()
    repository.append_turn(
        accepted.context_id, conversation_id, turn_id=answer_turn_id,
        operation_id=accepted.operation_id, role="model", parts=[{"text": manifest["body"]}],
        queue_operation_id=accepted.operation_id, queue_owner=lease.owner,
        queue_fence=lease.fence, now=NOW,
    )
    repository.save_presentation_manifest(
        accepted.context_id, manifest, queue_operation_id=accepted.operation_id,
        queue_owner=lease.owner, queue_fence=lease.fence, now=NOW, clock=lambda: NOW,
    )
    context_ref.collection("operations").document(accepted.operation_id).update({
        "route_control": {"status": "ok"},
    })
    assert repository.clear_command_candidate(
        accepted.context_id, accepted.operation_id, owner=lease.owner,
        fence=lease.fence, now=NOW,
    )
    assert complete_head(
        client, context_ref, operation_id=accepted.operation_id,
        owner=lease.owner, fence=lease.fence, now=NOW,
    )
    context_ref.update({"active_conversation_id": "later-selection"})
    assert repository.save_presentation_manifest(
        accepted.context_id, manifest, queue_operation_id=accepted.operation_id,
        queue_owner=lease.owner, queue_fence=lease.fence, now=NOW,
    ) == manifest
    assert repository.get_presentation_manifest(
        accepted.context_id, conversation_id, "real-lifecycle-presentation",
    ) == manifest


def test_foreign_positive_source_actor_cannot_prove_manifest(firestore_fake):
    client = firestore_fake.client_class()
    context_id, conversation_id = "42_5", "foreign-source-search"
    operation_id, owner, fence = "current-producer", "producer-owner", 93
    context_ref, repository, _conversation, _ = _seed_presentation_context(
        client, context_id=context_id, conversation_id=conversation_id,
        operation_id=operation_id, owner=owner, fence=fence, body="",
    )
    source_turn_id = "arbitrary-source-tool-turn"
    reference, exchange = _seed_source_from_distinct_operation(
        client, repository, context_id=context_id, conversation_id=conversation_id,
        source_operation_id="prior-source-operation", source_user_id=99,
        turn_id=source_turn_id, call_id="safe-call-id",
    )
    verified = [{"reference": reference, "exchange": exchange, "source_filters": {}}]
    answer_turn_id = "answer-turn-current-producer"
    manifest = build_presentation_manifest(
        scope={
            "context_id": context_id, "conversation_id": conversation_id,
            "producing_operation_id": operation_id,
            "presentation_id": "foreign-source-presentation",
            "answer_turn_id": answer_turn_id,
        },
        blocks=[{"type": "paragraph", "text": "A safe answer", "source_refs": []}],
        verified_exchanges=verified,
    ).to_dict()
    repository.append_turn(
        context_id, conversation_id, turn_id=answer_turn_id,
        operation_id=operation_id, role="model", parts=[{"text": manifest["body"]}], now=NOW,
    )
    with pytest.raises((PermissionError, ValueError), match="actor|source"):
        repository.save_presentation_manifest(
            context_id, manifest, queue_operation_id=operation_id,
            queue_owner=owner, queue_fence=fence, now=NOW, clock=lambda: NOW,
            source_turn_refs={manifest["sources"][0]["source_id"]: source_turn_id},
        )
    assert not context_ref.collection("conversations").document(conversation_id).collection(
        "presentations",
    ).document("foreign-source-presentation").get().exists


def test_sourced_addressed_retry_uses_pinned_anchors_after_terminal(firestore_fake):
    client = firestore_fake.client_class()
    context_id, conversation_id = "42_5", "source-replay-search"
    operation_id, owner, fence = "source-replay-producer", "source-replay-owner", 95
    context_ref, repository, _conversation, _ = _seed_presentation_context(
        client, context_id=context_id, conversation_id=conversation_id,
        operation_id=operation_id, owner=owner, fence=fence, body="",
    )
    source_turn_id = "source-replay-tool-turn"
    reference, exchange = _seed_source_from_distinct_operation(
        client, repository, context_id=context_id, conversation_id=conversation_id,
        source_operation_id="source-replay-origin", source_user_id=42,
        turn_id=source_turn_id, call_id="source-replay-call",
    )
    verified = [{"reference": reference, "exchange": exchange, "source_filters": {}}]
    answer_turn_id = "source-replay-answer"
    manifest = build_presentation_manifest(
        scope={
            "context_id": context_id, "conversation_id": conversation_id,
            "producing_operation_id": operation_id,
            "presentation_id": "source-replay-presentation",
            "answer_turn_id": answer_turn_id,
        },
        blocks=[{"type": "paragraph", "text": "Sourced answer", "source_refs": []}],
        verified_exchanges=verified,
    ).to_dict()
    repository.append_turn(
        context_id, conversation_id, turn_id=answer_turn_id,
        operation_id=operation_id, role="model", parts=[{"text": manifest["body"]}], now=NOW,
    )
    anchors = {manifest["sources"][0]["source_id"]: source_turn_id}
    repository.save_presentation_manifest(
        context_id, manifest, queue_operation_id=operation_id,
        queue_owner=owner, queue_fence=fence, now=NOW, clock=lambda: NOW,
        verified_exchanges=verified, source_turn_refs=anchors,
    )
    context_ref.collection("operations").document(operation_id).update({"state": "completed"})
    context_ref.collection("commands").document("1").update({
        "candidate": firestore.DELETE_FIELD,
    })
    context_ref.update({"active_conversation_id": "different-selected-conversation"})
    assert repository.save_presentation_manifest(
        context_id, manifest, queue_operation_id=operation_id,
        queue_owner=owner, queue_fence=fence, now=NOW,
    ) == manifest


def test_malformed_manifest_creation_time_fails_addressed_read(firestore_fake):
    client = firestore_fake.client_class()
    context_id, conversation_id = "42_5", "malformed-created-search"
    operation_id, owner, fence = "malformed-created-op", "malformed-created-owner", 96
    context_ref, repository, _conversation, _ = _seed_presentation_context(
        client, context_id=context_id, conversation_id=conversation_id,
        operation_id=operation_id, owner=owner, fence=fence, body="",
    )
    answer_turn_id = "malformed-created-answer"
    manifest = build_presentation_manifest(
        scope={
            "context_id": context_id, "conversation_id": conversation_id,
            "producing_operation_id": operation_id,
            "presentation_id": "malformed-created-presentation",
            "answer_turn_id": answer_turn_id,
        },
        blocks=[{"type": "paragraph", "text": "Created later", "source_refs": []}],
        verified_exchanges=[],
    ).to_dict()
    repository.append_turn(
        context_id, conversation_id, turn_id=answer_turn_id,
        operation_id=operation_id, role="model", parts=[{"text": manifest["body"]}], now=NOW,
    )
    repository.save_presentation_manifest(
        context_id, manifest, queue_operation_id=operation_id,
        queue_owner=owner, queue_fence=fence, now=NOW, clock=lambda: NOW,
    )
    header_ref = context_ref.collection("conversations").document(conversation_id).collection(
        "presentations",
    ).document(manifest["presentation_id"])
    header_ref.update({"created_at": "not-an-aware-timestamp"})
    with pytest.raises(ValueError, match="created|timestamp|presentation"):
        repository.get_presentation_manifest(
            context_id, conversation_id, manifest["presentation_id"],
        )


def test_concurrent_identical_header_is_addressed_idempotent_result(firestore_fake, monkeypatch):
    client = firestore_fake.client_class()
    context_id, conversation_id = "42_5", "concurrent-presentation-search"
    operation_id, owner, fence = "concurrent-presentation-op", "concurrent-owner", 97
    context_ref, repository, _conversation, _ = _seed_presentation_context(
        client, context_id=context_id, conversation_id=conversation_id,
        operation_id=operation_id, owner=owner, fence=fence, body="",
    )
    answer_turn_id = "concurrent-answer-turn"
    manifest = build_presentation_manifest(
        scope={
            "context_id": context_id, "conversation_id": conversation_id,
            "producing_operation_id": operation_id,
            "presentation_id": "concurrent-identical-presentation",
            "answer_turn_id": answer_turn_id,
        },
        blocks=[{"type": "paragraph", "text": "Same concurrent body", "source_refs": []}],
        verified_exchanges=[],
    ).to_dict()
    turn = repository.append_turn(
        context_id, conversation_id, turn_id=answer_turn_id,
        operation_id=operation_id, role="model", parts=[{"text": manifest["body"]}], now=NOW,
    )
    conversation_ref = context_ref.collection("conversations").document(conversation_id)
    header_ref = conversation_ref.collection("presentations").document(
        manifest["presentation_id"],
    )
    turn_ref = conversation_ref.collection("turns").document(answer_turn_id)
    operation_ref = context_ref.collection("operations").document(operation_id)
    original_factory = repository._presentation_ledger
    injected = False

    def ledger_with_concurrent_save(**kwargs):
        nonlocal injected
        ledger = original_factory(**kwargs)
        original_store = ledger._store_payload

        def store_and_publish(kind, value, *, write_assertion=None):
            nonlocal injected
            stored_payload = original_store(kind, value, write_assertion=write_assertion)
            if not injected:
                injected = True
                scope = {key: manifest[key] for key in (
                    "context_id", "conversation_id", "producing_operation_id",
                    "presentation_id", "answer_turn_id",
                )}
                command = context_ref.collection("commands").document("1").get().to_dict()
                header_ref.set({
                    **scope, "schema_version": 1,
                    "producer_actor_id": command["user_id"],
                    "answer_turn_sequence": turn["turn_sequence"],
                    "body_ref": {
                        "turn_id": answer_turn_id,
                        "turn_sequence": turn["turn_sequence"],
                        "body_sha256": manifest["body_sha256"],
                    },
                    "body_sha256": manifest["body_sha256"],
                    "manifest_payload": stored_payload, "source_turn_refs": {},
                    "created_at": NOW,
                })
                turn_row = turn_ref.get().to_dict()
                turn_row["presentation_binding"] = {
                    "scope": scope, "body_sha256": manifest["body_sha256"],
                    "answer_turn_sequence": turn["turn_sequence"],
                    "manifest_payload": stored_payload, "source_turn_refs": {},
                    "created_at": NOW,
                }
                turn_ref.set(turn_row)
                operation_ref.update({"presentation_answer_turn_id": answer_turn_id})
            return stored_payload

        ledger._store_payload = store_and_publish
        return ledger

    monkeypatch.setattr(repository, "_presentation_ledger", ledger_with_concurrent_save)
    assert repository.save_presentation_manifest(
        context_id, manifest, queue_operation_id=operation_id,
        queue_owner=owner, queue_fence=fence, now=NOW, clock=lambda: NOW,
    ) == manifest
    assert injected
    assert repository.get_presentation_manifest(
        context_id, conversation_id, manifest["presentation_id"],
    ) == manifest


def test_fresh_clock_expiry_during_multipart_staging_has_no_authority(firestore_fake):
    client = firestore_fake.client_class()
    context_id, conversation_id = "42_5", "stage-expiry-search"
    operation_id, owner, fence = "stage-expiry-op", "stage-expiry-owner", 81
    context_ref, repository, _conversation, _ = _seed_presentation_context(
        client, context_id=context_id, conversation_id=conversation_id,
        operation_id=operation_id, owner=owner, fence=fence, body="",
    )
    answer_turn_id = "search-answer-" + hashlib.sha256(operation_id.encode()).hexdigest()
    manifest = build_presentation_manifest(
        scope={
            "context_id": context_id, "conversation_id": conversation_id,
            "producing_operation_id": operation_id,
            "presentation_id": "stage-expiry-presentation", "answer_turn_id": answer_turn_id,
        },
        blocks=[{"type": "paragraph", "text": "z" * (600 * 1024), "source_refs": []}],
        verified_exchanges=[],
    ).to_dict()
    turn = repository.append_turn(
        context_id, conversation_id, turn_id=answer_turn_id,
        operation_id=operation_id, role="model", parts=[{"text": manifest["body"]}], now=NOW,
    )
    clock_calls = 0

    def staging_clock():
        nonlocal clock_calls
        clock_calls += 1
        return NOW if clock_calls <= 2 else NOW + timedelta(minutes=10)

    with pytest.raises(PermissionError, match="stale"):
        repository.save_presentation_manifest(
            context_id, manifest, queue_operation_id=operation_id,
            queue_owner=owner, queue_fence=fence, now=NOW, clock=staging_clock,
        )
    header = context_ref.collection("conversations").document(conversation_id).collection(
        "presentations",
    ).document("stage-expiry-presentation").get()
    stored_turn = context_ref.collection("conversations").document(conversation_id).collection(
        "turns",
    ).document(answer_turn_id).get().to_dict()
    assert not header.exists
    assert stored_turn.get("presentation_binding") is None
    assert turn["parts"] == [{"text": manifest["body"]}]


def test_delivery_cannot_predate_manifest_storage(firestore_fake):
    client = firestore_fake.client_class()
    context_id, conversation_id = "42_5", "delivery-before-save"
    operation_id, owner, fence = "delivery-before-save-op", "delivery-before-save-owner", 94
    _context_ref, repository, _conversation, _ = _seed_presentation_context(
        client, context_id=context_id, conversation_id=conversation_id,
        operation_id=operation_id, owner=owner, fence=fence, body="",
    )
    answer_turn_id = "delivery-before-save-answer"
    notification_id = "delivery-before-save-note"
    manifest = build_presentation_manifest(
        scope={
            "context_id": context_id, "conversation_id": conversation_id,
            "producing_operation_id": operation_id,
            "presentation_id": "delivery-before-save-presentation",
            "answer_turn_id": answer_turn_id,
        },
        blocks=[{"type": "paragraph", "text": "Saved after delivery", "source_refs": []}],
        verified_exchanges=[],
        notification_refs=[{"notification_id": notification_id, "revision": 1}],
    ).to_dict()
    repository.append_turn(
        context_id, conversation_id, turn_id=answer_turn_id,
        operation_id=operation_id, role="model", parts=[{"text": manifest["body"]}], now=NOW,
    )
    repository.create_notification(
        notification_id, context_id=context_id, kind="final", body=manifest["body"],
        operation_id=operation_id, now=NOW,
    )
    reservation = repository.reserve_notification_attempt(
        notification_id, attempt_id="delivery-before-save-attempt", now=NOW,
    )
    assert reservation["reserved"]
    repository.settle_notification_attempt(
        notification_id, attempt_id="delivery-before-save-attempt", revision=1,
        outcome="sent", message_id=950, now=NOW,
    )
    saved_at = NOW + timedelta(minutes=1)
    repository.save_presentation_manifest(
        context_id, manifest, queue_operation_id=operation_id,
        queue_owner=owner, queue_fence=fence, now=NOW, clock=lambda: saved_at,
    )
    assert not repository.publish_presentation_delivery(
        context_id, conversation_id, manifest["presentation_id"],
        notification_id=notification_id, revision=1, body=manifest["body"],
        body_sha256=manifest["body_sha256"], message_id=950,
        delivered_at=NOW, now=NOW + timedelta(minutes=2),
    )
    assert not list(client.collection("notification_outbox").document(
        notification_id,
    ).collection("presentation_delivery_proofs").stream())


def test_identical_raw_chunks_keep_their_prepared_indexes(firestore_fake):
    client = firestore_fake.client_class()
    context_id, conversation_id = "42_5", "identical-chunks"
    operation_id, owner, fence = "identical-chunk-op", "chunk-owner", 91
    context_ref, repository, _conversation, _ = _seed_presentation_context(
        client, context_id=context_id, conversation_id=conversation_id,
        operation_id=operation_id, owner=owner, fence=fence, body="",
    )
    answer_turn_id = "arbitrary-answer-turn-91"
    ids = ["identical-note-0", "identical-note-1"]
    manifest = build_presentation_manifest(
        scope={
            "context_id": context_id, "conversation_id": conversation_id,
            "producing_operation_id": operation_id,
            "presentation_id": "identical-chunk-presentation",
            "answer_turn_id": answer_turn_id,
        },
        blocks=[{"type": "paragraph", "text": "x" * 8000, "source_refs": []}],
        verified_exchanges=[],
        notification_refs=[{"notification_id": value, "revision": 1} for value in ids],
    ).to_dict()
    chunks = [manifest["body"][i:i + 4000] for i in range(0, len(manifest["body"]), 4000)]
    assert chunks == ["x" * 4000, "x" * 4000]
    repository.append_turn(
        context_id, conversation_id, turn_id=answer_turn_id,
        operation_id=operation_id, role="model", parts=[{"text": manifest["body"]}], now=NOW,
    )
    repository.save_presentation_manifest(
        context_id, manifest, queue_operation_id=operation_id,
        queue_owner=owner, queue_fence=fence, now=NOW, clock=lambda: NOW,
    )
    for index in (1, 0):
        repository.create_notification(
            ids[index], context_id=context_id,
            kind="progress" if index == 0 else "final", body=chunks[index],
            operation_id=operation_id, now=NOW,
        )
        reserved = repository.reserve_notification_attempt(
            ids[index], attempt_id=f"identical-attempt-{index}", now=NOW,
        )
        assert reserved["reserved"]
        repository.settle_notification_attempt(
            ids[index], attempt_id=f"identical-attempt-{index}", revision=1,
            outcome="sent", message_id=900 + index, now=NOW,
        )
        assert repository.publish_presentation_delivery(
            context_id, conversation_id, manifest["presentation_id"],
            notification_id=ids[index], revision=1, body=chunks[index],
            body_sha256=hashlib.sha256(chunks[index].encode()).hexdigest(),
            message_id=900 + index, delivered_at=NOW, now=NOW,
        )
    proof_indexes = []
    for identity in ids:
        proof_rows = list(client.collection("notification_outbox").document(identity).collection(
            "presentation_delivery_proofs",
        ).stream())
        assert len(proof_rows) == 1
        proof_indexes.append(proof_rows[0].to_dict()["chunk_index"])
    assert proof_indexes == [0, 1]


def test_real_claim_is_reclaimed_during_part_staging_without_publication(
    firestore_fake, monkeypatch,
):
    from core.integrations.firestore.operation_queue import claim_head
    from services.connectors.telegram.runtime import ledger as ledger_module

    client = firestore_fake.client_class()
    repository = TelegramRepository(client)
    accepted = repository.accept_command(
        9101, chat_id=42, user_id=42, thread_id=5,
        command=parse_command("show a large search answer"), now=NOW,
    )
    context_ref = client.collection("telegram_sessions").document(accepted.context_id)
    lease = claim_head(
        client, context_ref, owner="first-stage-owner", now=NOW, lease_seconds=1,
    )
    assert lease is not None
    conversation_id = "reclaimed-search"
    repository.ensure_conversation(
        accepted.context_id, conversation_id, kind="search",
        queue_operation_id=accepted.operation_id, queue_owner=lease.owner,
        queue_fence=lease.fence, select_if_unselected=True, now=NOW,
    )
    answer_turn_id = "search-answer-" + hashlib.sha256(
        accepted.operation_id.encode("utf-8"),
    ).hexdigest()
    manifest = build_presentation_manifest(
        scope={
            "context_id": accepted.context_id, "conversation_id": conversation_id,
            "producing_operation_id": accepted.operation_id,
            "presentation_id": "reclaimed-presentation", "answer_turn_id": answer_turn_id,
        },
        blocks=[{"type": "paragraph", "text": "large" * (140 * 1024), "source_refs": []}],
        verified_exchanges=[],
    ).to_dict()
    repository.append_turn(
        accepted.context_id, conversation_id, turn_id=answer_turn_id,
        operation_id=accepted.operation_id, role="model", parts=[{"text": manifest["body"]}],
        queue_operation_id=accepted.operation_id, queue_owner=lease.owner,
        queue_fence=lease.fence, now=NOW,
    )
    original_write = ledger_module.write_input_parts
    stage_writes = 0
    reclaimed = []

    def write_with_reclaim(db, input_ref, payload, *, write_document):
        def write_and_maybe_reclaim(ref, data):
            nonlocal stage_writes
            result = write_document(ref, data)
            stage_writes += 1
            if stage_writes == 1:
                context_ref.update({"lock_expires_at": NOW + timedelta(seconds=2)})
                next_lease = claim_head(
                    client, context_ref, owner="second-stage-owner",
                    now=NOW + timedelta(seconds=3), lease_seconds=30,
                )
                assert next_lease is not None and next_lease.fence > lease.fence
                reclaimed.append(next_lease)
            return result
        return original_write(db, input_ref, payload, write_document=write_and_maybe_reclaim)

    monkeypatch.setattr(ledger_module, "write_input_parts", write_with_reclaim)
    with pytest.raises(PermissionError, match="stale"):
        repository.save_presentation_manifest(
            accepted.context_id, manifest, queue_operation_id=accepted.operation_id,
            queue_owner=lease.owner, queue_fence=lease.fence, now=NOW, clock=lambda: NOW,
        )
    assert stage_writes >= 1 and reclaimed
    conv_ref = context_ref.collection("conversations").document(conversation_id)
    header = conv_ref.collection("presentations").document("reclaimed-presentation").get()
    answer_turn = conv_ref.collection("turns").document(answer_turn_id).get().to_dict()
    assert not header.exists
    assert answer_turn.get("presentation_binding") is None


def test_large_manifest_is_multipart_without_copying_rendered_body(firestore_fake):
    client = firestore_fake.client_class()
    context_id, conversation_id = "42_5", "large-search"
    operation_id, owner, fence = "large-present-op", "present-owner", 8
    answer_turn_id = "telegram-" + hashlib.sha256(
        f"{operation_id}:model".encode("utf-8"),
    ).hexdigest()
    scope = {
        "context_id": context_id,
        "conversation_id": conversation_id,
        "producing_operation_id": operation_id,
        "presentation_id": "presentation-large",
        "answer_turn_id": answer_turn_id,
    }
    manifest = build_presentation_manifest(
        scope=scope,
        blocks=[{"type": "paragraph", "text": "Ж" * (600 * 1024), "source_refs": []}],
        verified_exchanges=[],
    ).to_dict()
    context_ref, repository, _conversation, _ = _seed_presentation_context(
        client, context_id=context_id, conversation_id=conversation_id,
        operation_id=operation_id, owner=owner, fence=fence, body=manifest["body"],
    )
    repository.append_turn(
        context_id, conversation_id, turn_id=scope["answer_turn_id"],
        operation_id=operation_id, role="model", parts=[{"text": manifest["body"]}],
        now=NOW,
    )

    repository.save_presentation_manifest(
        context_id, manifest, queue_operation_id=operation_id,
        queue_owner=owner, queue_fence=fence, now=NOW, clock=lambda: NOW,
    )
    header = context_ref.collection("conversations").document(conversation_id).collection(
        "presentations",
    ).document("presentation-large").get().to_dict()

    assert header["manifest_payload"]["_input_ref"].startswith("dialog_operations/")
    assert header["manifest_payload"]["byte_length"] > 512 * 1024
    assert "body" not in header
    assert repository.get_presentation_manifest(
        context_id, conversation_id, "presentation-large",
    ) == manifest
    header_ref = context_ref.collection("conversations").document(conversation_id).collection(
        "presentations",
    ).document("presentation-large")
    original_header = header_ref.get().to_dict()
    wrong_scope_header = dict(original_header)
    wrong_scope_payload = dict(original_header["manifest_payload"])
    wrong_scope_payload["_input_ref"] = "dialog_operations/foreign-context/input_parts/manifest"
    wrong_scope_header["manifest_payload"] = wrong_scope_payload
    header_ref.set(wrong_scope_header)
    turn_ref = context_ref.collection("conversations").document(conversation_id).collection(
        "turns",
    ).document(answer_turn_id)
    original_turn = turn_ref.get().to_dict()
    rebound_turn = dict(original_turn)
    rebound_binding = dict(original_turn["presentation_binding"])
    rebound_binding["manifest_payload"] = wrong_scope_payload
    rebound_turn["presentation_binding"] = rebound_binding
    turn_ref.set(rebound_turn)
    with pytest.raises(ValueError, match="path|scope|binding"):
        repository.get_presentation_manifest(
            context_id, conversation_id, "presentation-large",
        )
    header_ref.set(original_header)
    turn_ref.set(original_turn)
    input_ref = header["manifest_payload"]["_input_ref"]
    input_operation_id = input_ref.split("/")[1]
    part_ref = client.collection("dialog_operations").document(input_operation_id).collection(
        "input_parts",
    ).document("part-000000")
    part = part_ref.get().to_dict()
    part_ref.update({"payload": b"corrupted multipart presentation bytes"})
    with pytest.raises(ValueError, match="verify|integrity|corrupt"):
        repository.get_presentation_manifest(
            context_id, conversation_id, "presentation-large",
        )


def test_stale_fence_cannot_publish_header_or_bind_answer_turn(firestore_fake):
    client = firestore_fake.client_class()
    context_id, conversation_id = "42_5", "stale-search"
    operation_id, owner, fence = "stale-op", "stale-owner", 10
    answer_turn_id = "telegram-" + hashlib.sha256(
        f"{operation_id}:model".encode("utf-8"),
    ).hexdigest()
    context_ref, repository, _conversation, _ = _seed_presentation_context(
        client, context_id=context_id, conversation_id=conversation_id,
        operation_id=operation_id, owner=owner, fence=fence, body="",
    )
    manifest = build_presentation_manifest(
        scope={
            "context_id": context_id, "conversation_id": conversation_id,
            "producing_operation_id": operation_id, "presentation_id": "stale-presentation",
            "answer_turn_id": answer_turn_id,
        },
        blocks=[{"type": "paragraph", "text": "not sent", "source_refs": []}],
        verified_exchanges=[],
    ).to_dict()
    turn = repository.append_turn(
        context_id, conversation_id, turn_id=answer_turn_id,
        operation_id=operation_id, role="model", parts=[{"text": manifest["body"]}],
        now=NOW,
    )
    context_ref.update({"lock_expires_at": NOW - timedelta(seconds=1)})

    with pytest.raises(PermissionError, match="stale"):
        repository.save_presentation_manifest(
            context_id, manifest, queue_operation_id=operation_id,
            queue_owner=owner, queue_fence=fence, now=NOW, clock=lambda: NOW,
        )
    header = context_ref.collection("conversations").document(conversation_id).collection(
        "presentations",
    ).document("stale-presentation").get()
    stored_turn = context_ref.collection("conversations").document(conversation_id).collection(
        "turns",
    ).document(answer_turn_id).get().to_dict()
    assert not header.exists
    assert "presentation_binding" not in stored_turn
    assert stored_turn["turn_sequence"] == turn["turn_sequence"]


def test_part_stage_fault_leaves_only_unaddressable_parts(firestore_fake, monkeypatch):
    import services.connectors.telegram.runtime.ledger as ledger_module

    client = firestore_fake.client_class()
    context_id, conversation_id = "42_5", "part-fault-search"
    operation_id, owner, fence = "part-fault-op", "part-fault-owner", 11
    answer_turn_id = "telegram-" + hashlib.sha256(
        f"{operation_id}:model".encode("utf-8"),
    ).hexdigest()
    context_ref, repository, _conversation, _ = _seed_presentation_context(
        client, context_id=context_id, conversation_id=conversation_id,
        operation_id=operation_id, owner=owner, fence=fence, body="",
    )
    manifest = build_presentation_manifest(
        scope={
            "context_id": context_id, "conversation_id": conversation_id,
            "producing_operation_id": operation_id, "presentation_id": "part-fault-presentation",
            "answer_turn_id": answer_turn_id,
        },
        blocks=[{"type": "paragraph", "text": "body-" * 2000, "source_refs": []}],
        verified_exchanges=[],
    ).to_dict()
    repository.append_turn(
        context_id, conversation_id, turn_id=answer_turn_id,
        operation_id=operation_id, role="model", parts=[{"text": manifest["body"]}],
        now=NOW,
    )
    write_parts = ledger_module.write_input_parts

    def fail_after_parts(*args, **kwargs):
        write_parts(*args, **kwargs)
        raise RuntimeError("injected presentation part publication fault")

    monkeypatch.setattr(ledger_module, "write_input_parts", fail_after_parts)
    with pytest.raises(RuntimeError, match="part publication fault"):
        repository.save_presentation_manifest(
            context_id, manifest, queue_operation_id=operation_id,
            queue_owner=owner, queue_fence=fence, now=NOW, clock=lambda: NOW,
        )
    header = context_ref.collection("conversations").document(conversation_id).collection(
        "presentations",
    ).document("part-fault-presentation").get()
    turn = context_ref.collection("conversations").document(conversation_id).collection(
        "turns",
    ).document(answer_turn_id).get().to_dict()
    assert not header.exists
    assert "presentation_binding" not in turn


def test_header_commit_fault_keeps_manifest_and_body_binding_unpublished(firestore_fake):
    client = firestore_fake.client_class()
    context_id, conversation_id = "42_5", "header-fault-search"
    operation_id, owner, fence = "header-fault-op", "header-fault-owner", 12
    answer_turn_id = "telegram-" + hashlib.sha256(
        f"{operation_id}:model".encode("utf-8"),
    ).hexdigest()
    context_ref, repository, _conversation, _ = _seed_presentation_context(
        client, context_id=context_id, conversation_id=conversation_id,
        operation_id=operation_id, owner=owner, fence=fence, body="",
    )
    manifest = build_presentation_manifest(
        scope={
            "context_id": context_id, "conversation_id": conversation_id,
            "producing_operation_id": operation_id, "presentation_id": "header-fault-presentation",
            "answer_turn_id": answer_turn_id,
        },
        blocks=[{"type": "paragraph", "text": "body", "source_refs": []}],
        verified_exchanges=[],
    ).to_dict()
    repository.append_turn(
        context_id, conversation_id, turn_id=answer_turn_id,
        operation_id=operation_id, role="model", parts=[{"text": manifest["body"]}],
        now=NOW,
    )
    header_path = (
        f"telegram_sessions/{context_id}/conversations/{conversation_id}/"
        "presentations/header-fault-presentation"
    )
    client.fail_next("commit", path=header_path, phase="before",
                     exception=RuntimeError("injected presentation header fault"))

    with pytest.raises(RuntimeError, match="header fault"):
        repository.save_presentation_manifest(
            context_id, manifest, queue_operation_id=operation_id,
            queue_owner=owner, queue_fence=fence, now=NOW, clock=lambda: NOW,
        )
    header = context_ref.collection("conversations").document(conversation_id).collection(
        "presentations",
    ).document("header-fault-presentation").get()
    turn = context_ref.collection("conversations").document(conversation_id).collection(
        "turns",
    ).document(answer_turn_id).get().to_dict()
    assert not header.exists
    assert "presentation_binding" not in turn


def test_shown_list_pointer_requires_actual_matching_notification_delivery(firestore_fake):
    client = firestore_fake.client_class()
    context_id, conversation_id = "42_5", "search-delivery"
    operation_id, owner, fence = "delivery-op", "delivery-owner", 9
    answer_turn_id = "telegram-" + hashlib.sha256(
        f"{operation_id}:model".encode("utf-8"),
    ).hexdigest()
    context_ref, repository, _conversation, _ = _seed_presentation_context(
        client, context_id=context_id, conversation_id=conversation_id,
        operation_id=operation_id, owner=owner, fence=fence, body="",
    )
    context_ref.update({"chat_id": 42, "thread_id": 5})
    ledger = FirestoreAgentLedger(
        client, context_id=context_id, operation_id=operation_id,
        owner=owner, fence=fence, clock=lambda: NOW,
    )
    call = {
        "id": "lookup/call Ω", "name": "list_recent_transcripts",
        "arguments": {"limit": 1, "offset": 0}, "ordinal": 0, "round": 0,
    }
    result = {
        "total": 1, "returned": 1, "offset": 0, "limit": 1,
        "has_more": False, "transcripts": [{"doc_id": "doc-a", "title": "Project A"}],
    }
    reference = ledger.save_search_exchange(
        conversation_id,
        checkpoint_conversation_id="telegram-global-" + hashlib.sha256(
            f"{context_id}\0{operation_id}".encode("utf-8"),
        ).hexdigest(),
        call=call,
        canonical_call_id="lookup/call Ω", result=result,
    )
    checkpoint_conversation_id = reference["checkpoint_conversation_id"]
    ledger.save_turn(
        checkpoint_conversation_id, operation_id, model_response={}, tool_calls=[call],
    )
    ledger.save_tool_result(checkpoint_conversation_id, operation_id, 0, result)
    repository.append_turn(
        context_id, conversation_id,
        turn_id="search-history-anchor-0",
        operation_id=operation_id, role="tool",
        parts=[{"type": "search_exchange_ref", "reference": reference}], now=NOW,
    )
    exchange = ledger.load_search_exchange(
        reference, conversation_id=conversation_id, operation_id=operation_id,
        checkpoint_conversation_id=checkpoint_conversation_id,
        call_id="lookup/call Ω", ordinal=0,
    )
    verified = [{"reference": reference, "exchange": exchange, "source_filters": {}}]
    manifest = build_presentation_manifest(
        scope={
            "context_id": context_id, "conversation_id": conversation_id,
            "producing_operation_id": operation_id, "presentation_id": "shown-1",
            "answer_turn_id": answer_turn_id,
        },
        blocks=[{
            "type": "document_list", "list_id": "list-a", "entries": [{
                "doc_id": "doc-a", "label": "Project A",
                "source_refs": [{"doc_id": "doc-a", "call_id": "lookup/call Ω", "ordinal": 0}],
            }],
        }],
        verified_exchanges=verified,
        notification_refs=[{"notification_id": "answer-note", "revision": 1}],
    ).to_dict()
    repository.append_turn(
        context_id, conversation_id, turn_id=answer_turn_id,
        operation_id=operation_id, role="model", parts=[{"text": manifest["body"]}],
        now=NOW,
    )
    with guard_firestore_enumeration() as scan_events:
        repository.save_presentation_manifest(
            context_id, manifest, queue_operation_id=operation_id,
            queue_owner=owner, queue_fence=fence, now=NOW, clock=lambda: NOW,
            verified_exchanges=verified,
            source_turn_refs={manifest["sources"][0]["source_id"]: "search-history-anchor-0"},
        )
        assert repository.get_presentation_manifest(
            context_id, conversation_id, "shown-1",
        ) == manifest
    assert scan_events == []
    conversation_ref = context_ref.collection("conversations").document(conversation_id)
    # A forged pointer to a stored header without actual delivery proof is not
    # an ingress observation.
    conversation_ref.update({
        "last_presentation_id": "shown-1", "last_presented_list_id": "list-a",
    })
    forged_pointer = repository.accept_command(
        38, chat_id=42, user_id=42, thread_id=5,
        command=parse_command("move the shown conversation"), now=NOW,
    )
    assert forged_pointer.operation_id
    assert context_ref.collection("commands").document("38").get().to_dict()[
        "observed_presentation_id"
    ] is None
    conversation_ref.update({
        "last_presentation_id": firestore.DELETE_FIELD,
        "last_presented_list_id": firestore.DELETE_FIELD,
    })
    row = repository.create_notification(
        "answer-note", context_id=context_id, kind="final", body=manifest["body"],
        operation_id=operation_id, now=NOW,
    )
    assert row["state"] == "pending"
    short_body = manifest["body"][:12]
    assert not repository.publish_presentation_delivery(
        context_id, conversation_id, "shown-1", notification_id="answer-note",
        revision=1, body=short_body,
        body_sha256=hashlib.sha256(short_body.encode("utf-8")).hexdigest(),
        message_id=501, delivered_at=NOW, now=NOW,
    )
    assert not repository.publish_presentation_delivery(
        context_id, conversation_id, "shown-1", notification_id="wrong-prepared-note",
        revision=1, body=manifest["body"], body_sha256=manifest["body_sha256"],
        message_id=501, delivered_at=NOW, now=NOW,
    )
    record = repository.get_conversation(context_id, conversation_id)
    assert record.last_presentation_id is None
    assert record.last_presented_list_id is None
    assert not repository.publish_presentation_delivery(
        context_id, conversation_id, "shown-1", notification_id="answer-note",
        revision=1, body=manifest["body"], body_sha256=manifest["body_sha256"],
        message_id=501, delivered_at=NOW, now=NOW,
    )

    reserved = repository.reserve_notification_attempt("answer-note", attempt_id="attempt-1", now=NOW)
    assert reserved["reserved"] is True
    settled = repository.settle_notification_attempt(
        "answer-note", attempt_id="attempt-1", revision=1, outcome="sent",
        message_id=501, now=NOW,
    )
    assert settled["state"] == "delivered"
    conversation_path = f"telegram_sessions/{context_id}/conversations/{conversation_id}"
    client.fail_next(
        "commit", path=conversation_path, phase="before",
        exception=RuntimeError("injected delivery publication commit fault"),
    )
    with pytest.raises(RuntimeError, match="publication commit fault"):
        repository.publish_presentation_delivery(
            context_id, conversation_id, "shown-1", notification_id="answer-note",
            revision=1, body=manifest["body"], body_sha256=manifest["body_sha256"],
            message_id=501, delivered_at=NOW, now=NOW,
        )
    evidence_collection = client.collection("notification_outbox").document(
        "answer-note",
    ).collection("presentation_delivery_proofs")
    assert list(evidence_collection.stream()) == []
    record = repository.get_conversation(context_id, conversation_id)
    assert record.last_presentation_id is None
    assert record.last_presented_list_id is None

    assert repository.publish_presentation_delivery(
        context_id, conversation_id, "shown-1", notification_id="answer-note",
        revision=1, body=manifest["body"], body_sha256=manifest["body_sha256"],
        message_id=501, delivered_at=NOW, now=NOW,
    )
    record = repository.get_conversation(context_id, conversation_id)
    assert record.last_presentation_id == "shown-1"
    assert record.last_presented_list_id == "list-a"

    # The immutable proof remains after the mutable notification row advances.
    client.collection("notification_outbox").document("answer-note").update({
        "revision": 2, "body": "edited copy", "state": "pending", "delivered_at": None,
    })
    assert repository.publish_presentation_delivery(
        context_id, conversation_id, "shown-1", notification_id="answer-note",
        revision=1, body=manifest["body"], body_sha256=manifest["body_sha256"],
        message_id=501, delivered_at=NOW, now=NOW,
    )
    evidence = client.collection("notification_outbox").document("answer-note").collection(
        "presentation_delivery_proofs",
    ).stream()
    assert len(list(evidence)) == 1
    proof_row = list(client.collection("notification_outbox").document(
        "answer-note",
    ).collection("presentation_delivery_proofs").stream())[0].to_dict()
    assert "body" not in proof_row
    assert proof_row["chunk_index"] == 0
    assert proof_row["chunk_start"] == 0
    assert proof_row["chunk_end"] == len(manifest["body"])
    assert not client.collection("telegram_sessions").document(context_id).collection(
        "conversations",
    ).document(conversation_id).collection("presentations").document("shown-1").collection(
        "delivery_evidence",
    ).stream()
    proof_id = hashlib.sha256(
        f"{context_id}\0{conversation_id}\0shown-1\0{1}".encode(),
    ).hexdigest()
    proof_ref = client.collection("notification_outbox").document("answer-note").collection(
        "presentation_delivery_proofs",
    ).document(proof_id)
    proof = proof_ref.get().to_dict()
    proof_ref.update({"delivered_at": NOW + timedelta(minutes=1)})
    future_proof = repository.accept_command(
        39, chat_id=42, user_id=42, thread_id=5,
        command=parse_command("move the shown conversation"), now=NOW,
    )
    assert future_proof.operation_id
    assert context_ref.collection("commands").document("39").get().to_dict()[
        "observed_presentation_id"
    ] is None
    proof_ref.update({"delivered_at": proof["delivered_at"]})

    first = repository.accept_command(
        41, chat_id=42, user_id=42, thread_id=5,
        command=parse_command("move the shown conversation"), now=NOW,
    )
    command_ref = context_ref.collection("commands").document("41")
    command = command_ref.get().to_dict()
    assert command["observed_presentation_id"] == "shown-1"
    assert command["observed_presentation_id"] != "list-a"

    conversation_ref.update({"last_presentation_id": "unsupported-manifest"})
    unsupported = repository.accept_command(
        42, chat_id=42, user_id=42, thread_id=5,
        command=parse_command("move the shown conversation"), now=NOW,
    )
    assert unsupported.context_id == first.context_id
    assert context_ref.collection("commands").document("42").get().to_dict()[
        "observed_presentation_id"
    ] is None

    conversation_ref.update({"last_presentation_id": "shown-1"})
    foreign = repository.accept_command(
        43, chat_id=42, user_id=99, thread_id=5,
        command=parse_command("move the shown conversation"), now=NOW,
    )
    assert foreign.context_id == first.context_id
    assert context_ref.collection("commands").document("43").get().to_dict()[
        "observed_presentation_id"
    ] is None

    # A duplicate reuses the immutable ingress observation, even if the
    # mutable pointer changes before transport retries the same update.
    conversation_ref.update({
        "last_presentation_id": None, "last_presented_list_id": None,
    })
    replay = repository.accept_command(
        41, chat_id=42, user_id=42, thread_id=5,
        command=parse_command("move the shown conversation"), now=NOW,
    )
    assert replay.operation_id == first.operation_id
    assert command_ref.get().to_dict()["observed_presentation_id"] == "shown-1"
    context_ref.collection("operations").document(operation_id).update({"state": "completed"})
    context_ref.collection("commands").document("1").update({
        "candidate": firestore.DELETE_FIELD,
    })
    context_ref.update({"active_conversation_id": "changed-after-source-publication"})
    assert repository.save_presentation_manifest(
        context_id, manifest, queue_operation_id=operation_id,
        queue_owner=owner, queue_fence=fence, now=NOW,
    ) == manifest
    header_ref = conversation_ref.collection("presentations").document("shown-1")
    header = header_ref.get().to_dict()
    stored_payload = header["manifest_payload"]
    stored_payload["_inline"]["sources"][0]["call_name"] = "query"
    encoded_payload = FirestoreAgentLedger._json_bytes(stored_payload["_inline"])
    stored_payload["sha256"] = hashlib.sha256(encoded_payload).hexdigest()
    stored_payload["byte_length"] = len(encoded_payload)
    header_ref.update({"manifest_payload": stored_payload})
    with pytest.raises(ValueError):
        repository.get_presentation_manifest(context_id, conversation_id, "shown-1")


def test_list_spanning_notification_chunks_is_not_shown_until_all_covered(firestore_fake):
    client = firestore_fake.client_class()
    context_id, conversation_id = "42_5", "chunked-search"
    operation_id, owner, fence = "chunked-op", "chunked-owner", 13
    answer_turn_id = "telegram-" + hashlib.sha256(
        f"{operation_id}:model".encode("utf-8"),
    ).hexdigest()
    context_ref, repository, _conversation, _ = _seed_presentation_context(
        client, context_id=context_id, conversation_id=conversation_id,
        operation_id=operation_id, owner=owner, fence=fence, body="",
    )
    ledger = FirestoreAgentLedger(
        client, context_id=context_id, operation_id=operation_id,
        owner=owner, fence=fence, clock=lambda: NOW,
    )
    call = {
        "id": "many-call", "name": "list_recent_transcripts",
        "arguments": {"limit": 200, "offset": 0}, "ordinal": 0, "round": 0,
    }
    result = {
        "total": 200, "returned": 200, "offset": 0, "limit": 200,
        "has_more": False,
        "transcripts": [{"doc_id": f"doc-{index}", "title": f"Doc {index}"}
                        for index in range(200)],
    }
    checkpoint_conversation_id = "telegram-global-" + hashlib.sha256(
        f"{context_id}\0{operation_id}".encode("utf-8"),
    ).hexdigest()
    reference = ledger.save_search_exchange(
        conversation_id, checkpoint_conversation_id=checkpoint_conversation_id,
        call=call, canonical_call_id="many-call", result=result,
    )
    ledger.save_turn(checkpoint_conversation_id, operation_id, model_response={},
                     tool_calls=[call])
    ledger.save_tool_result(checkpoint_conversation_id, operation_id, 0, result)
    repository.append_turn(
        context_id, conversation_id,
        turn_id="search-history-anchor-long",
        operation_id=operation_id, role="tool",
        parts=[{"type": "search_exchange_ref", "reference": reference}], now=NOW,
    )
    exchange = ledger.load_search_exchange(
        reference, conversation_id=conversation_id, operation_id=operation_id,
        checkpoint_conversation_id=checkpoint_conversation_id,
        call_id="many-call", ordinal=0,
    )
    verified = [{"reference": reference, "exchange": exchange, "source_filters": {}}]
    entries = [{
        "doc_id": f"doc-{index}", "label": f"Document {index:03d} — " + "x" * 18,
        "source_refs": [{"doc_id": f"doc-{index}", "call_id": "many-call", "ordinal": 0}],
    } for index in range(200)]
    scope = {
        "context_id": context_id, "conversation_id": conversation_id,
        "producing_operation_id": operation_id, "presentation_id": "chunked-list",
        "answer_turn_id": answer_turn_id,
    }
    body_chunks = None
    # Build once to obtain the exact renderer body, then persist the prepared
    # chunk revision references in the immutable manifest.
    initial = build_presentation_manifest(
        scope=scope,
        blocks=[{"type": "document_list", "list_id": "🔎/long list/1", "entries": entries}],
        verified_exchanges=verified,
    ).to_dict()
    body_chunks = [initial["body"][index:index + 4000]
                   for index in range(0, len(initial["body"]), 4000)]
    assert len(body_chunks) == 2
    notification_refs = [
        {"notification_id": f"long-answer-{index}", "revision": 1}
        for index in range(2)
    ]
    manifest = build_presentation_manifest(
        scope=scope,
        blocks=[{"type": "document_list", "list_id": "🔎/long list/1", "entries": entries}],
        verified_exchanges=verified, notification_refs=notification_refs,
    ).to_dict()
    repository.append_turn(
        context_id, conversation_id, turn_id=answer_turn_id,
        operation_id=operation_id, role="model", parts=[{"text": manifest["body"]}],
        now=NOW,
    )
    repository.save_presentation_manifest(
        context_id, manifest, queue_operation_id=operation_id,
        queue_owner=owner, queue_fence=fence, now=NOW, clock=lambda: NOW,
        verified_exchanges=verified,
        source_turn_refs={manifest["sources"][0]["source_id"]: "search-history-anchor-long"},
    )
    for index, chunk in enumerate(body_chunks):
        notification_id = f"long-answer-{index}"
        repository.create_notification(
            notification_id, context_id=context_id, kind="progress" if index == 0 else "final",
            body=chunk, operation_id=operation_id, now=NOW,
        )

    def deliver(index):
        notification_id = f"long-answer-{index}"
        attempt_id = f"long-attempt-{index}"
        reservation = repository.reserve_notification_attempt(
            notification_id, attempt_id=attempt_id, now=NOW,
        )
        assert reservation["reserved"] is True
        settled = repository.settle_notification_attempt(
            notification_id, attempt_id=attempt_id, revision=1, outcome="sent",
            message_id=700 + index, now=NOW,
        )
        assert settled["state"] == "delivered"
        assert repository.publish_presentation_delivery(
            context_id, conversation_id, "chunked-list", notification_id=notification_id,
            revision=1, body=body_chunks[index],
            body_sha256=hashlib.sha256(body_chunks[index].encode("utf-8")).hexdigest(),
            message_id=700 + index, delivered_at=NOW, now=NOW,
        )

    deliver(0)
    record = repository.get_conversation(context_id, conversation_id)
    assert record.last_presentation_id is None
    assert record.last_presented_list_id is None
    deliver(1)
    record = repository.get_conversation(context_id, conversation_id)
    assert record.last_presentation_id == "chunked-list"
    assert record.last_presented_list_id == "🔎/long list/1"

    def next_producer(next_operation_id, command_id, next_owner, next_fence):
        context_ref.update({
            "reserved_operation_id": next_operation_id,
            "active_operation_id": next_operation_id,
            "lock_owner": next_owner,
            "fence": next_fence,
            "lock_expires_at": NOW + timedelta(minutes=5),
        })
        context_ref.collection("operations").document(next_operation_id).set({
            "operation_id": next_operation_id,
            "state": "running",
            "input_ref": f"telegram_sessions/{context_id}/commands/{command_id}",
        })
        context_ref.collection("commands").document(command_id).set({
            "context_id": context_id, "operation_id": next_operation_id,
            "user_id": 42, "kind": "freeform", "candidate": "continue",
        })

    no_list_op, no_list_owner, no_list_fence = "no-list-op", "no-list-owner", 14
    next_producer(no_list_op, "2", no_list_owner, no_list_fence)
    no_list_turn = "telegram-" + hashlib.sha256(
        f"{no_list_op}:model".encode("utf-8"),
    ).hexdigest()
    no_list_manifest = build_presentation_manifest(
        scope={
            "context_id": context_id, "conversation_id": conversation_id,
            "producing_operation_id": no_list_op, "presentation_id": "no-list-answer",
            "answer_turn_id": no_list_turn,
        },
        blocks=[{"type": "paragraph", "text": "An ordinary answer", "source_refs": []}],
        verified_exchanges=[],
        notification_refs=[{"notification_id": "no-list-note", "revision": 1}],
    ).to_dict()
    repository.append_turn(
        context_id, conversation_id, turn_id=no_list_turn,
        operation_id=no_list_op, role="model", parts=[{"text": no_list_manifest["body"]}],
        now=NOW,
    )
    repository.save_presentation_manifest(
        context_id, no_list_manifest, queue_operation_id=no_list_op,
        queue_owner=no_list_owner, queue_fence=no_list_fence, now=NOW, clock=lambda: NOW,
    )
    repository.create_notification(
        "no-list-note", context_id=context_id, kind="final", body=no_list_manifest["body"],
        operation_id=no_list_op, now=NOW,
    )
    reservation = repository.reserve_notification_attempt(
        "no-list-note", attempt_id="no-list-attempt", now=NOW,
    )
    assert reservation["reserved"] is True
    repository.settle_notification_attempt(
        "no-list-note", attempt_id="no-list-attempt", revision=1,
        outcome="sent", message_id=800, now=NOW,
    )
    assert repository.publish_presentation_delivery(
        context_id, conversation_id, "no-list-answer", notification_id="no-list-note",
        revision=1, body=no_list_manifest["body"],
        body_sha256=no_list_manifest["body_sha256"], message_id=800,
        delivered_at=NOW, now=NOW,
    )
    record = repository.get_conversation(context_id, conversation_id)
    assert record.last_presentation_id == "chunked-list"
    assert record.last_presented_list_id == "🔎/long list/1"

    multi_op, multi_owner, multi_fence = "multi-list-op", "multi-list-owner", 15
    next_producer(multi_op, "3", multi_owner, multi_fence)
    multi_turn = "telegram-" + hashlib.sha256(
        f"{multi_op}:model".encode("utf-8"),
    ).hexdigest()
    multi_note = "multi-list-note"
    multi_blocks = [
        {"type": "document_list", "list_id": list_id, "entries": [{
            "doc_id": doc_id, "label": label,
            "source_refs": [{"doc_id": doc_id, "call_id": "many-call", "ordinal": 0}],
        }]} for list_id, doc_id, label in (
            ("first-list", "doc-0", "First result"),
            ("second-list", "doc-1", "Second result"),
        )
    ]
    multi_manifest = build_presentation_manifest(
        scope={
            "context_id": context_id, "conversation_id": conversation_id,
            "producing_operation_id": multi_op, "presentation_id": "multi-list-answer",
            "answer_turn_id": multi_turn,
        },
        blocks=multi_blocks, verified_exchanges=verified,
        notification_refs=[{"notification_id": multi_note, "revision": 1}],
    ).to_dict()
    repository.append_turn(
        context_id, conversation_id, turn_id=multi_turn,
        operation_id=multi_op, role="model", parts=[{"text": multi_manifest["body"]}],
        now=NOW,
    )
    repository.save_presentation_manifest(
        context_id, multi_manifest, queue_operation_id=multi_op,
        queue_owner=multi_owner, queue_fence=multi_fence, now=NOW, clock=lambda: NOW,
        verified_exchanges=verified,
        source_turn_refs={multi_manifest["sources"][0]["source_id"]: "search-history-anchor-long"},
    )
    repository.create_notification(
        multi_note, context_id=context_id, kind="final", body=multi_manifest["body"],
        operation_id=multi_op, now=NOW,
    )
    reservation = repository.reserve_notification_attempt(
        multi_note, attempt_id="multi-list-attempt", now=NOW,
    )
    assert reservation["reserved"] is True
    repository.settle_notification_attempt(
        multi_note, attempt_id="multi-list-attempt", revision=1,
        outcome="sent", message_id=801, now=NOW,
    )
    assert repository.publish_presentation_delivery(
        context_id, conversation_id, "multi-list-answer", notification_id=multi_note,
        revision=1, body=multi_manifest["body"],
        body_sha256=multi_manifest["body_sha256"], message_id=801,
        delivered_at=NOW, now=NOW,
    )
    record = repository.get_conversation(context_id, conversation_id)
    assert record.last_presentation_id == "multi-list-answer"
    assert record.last_presented_list_id is None
    conversation_ref = context_ref.collection("conversations").document(conversation_id)
    conversation_ref.update({"last_presented_list_id": "first-list"})
    forged_multi_pointer = repository.accept_command(
        77, chat_id=42, user_id=42, thread_id=5,
        command=parse_command("move the shown conversation"), now=NOW,
    )
    assert forged_multi_pointer.operation_id
    assert context_ref.collection("commands").document("77").get().to_dict()[
        "observed_presentation_id"
    ] is None
    conversation_ref.update({"last_presented_list_id": None})

    # Simulate a delayed publication callback for an actually delivered older
    # chunk. Its new evidence is saved, but its lower turn sequence cannot
    # restore the prior list pointer.
    old_proof_id = hashlib.sha256(
        f"{context_id}\0{conversation_id}\0chunked-list\0{1}".encode(),
    ).hexdigest()
    old_proof_ref = client.collection("notification_outbox").document(
        "long-answer-0",
    ).collection("presentation_delivery_proofs").document(old_proof_id)
    old_proof_ref.delete()
    assert repository.publish_presentation_delivery(
        context_id, conversation_id, "chunked-list", notification_id="long-answer-0",
        revision=1, body=body_chunks[0],
        body_sha256=hashlib.sha256(body_chunks[0].encode("utf-8")).hexdigest(),
        message_id=700, delivered_at=NOW, now=NOW,
    )
    record = repository.get_conversation(context_id, conversation_id)
    assert record.last_presentation_id == "multi-list-answer"
    assert record.last_presented_list_id is None

