"""Acceptance tests for operation-scoped routing context capture."""

from __future__ import annotations

from copy import deepcopy
from datetime import timedelta
import hashlib

import pytest
from google.cloud import firestore

from core.integrations.firestore.client import COLLECTION_NAME
from core.integrations.firestore.input_parts import MAX_PART_BYTES
from core.integrations.firestore.operation_queue import claim_head, complete_head
from services.connectors.telegram.commands import parse_command
from services.connectors.telegram.repository import SESSIONS, TelegramRepository
from services.connectors.telegram.repository._common import _document_ref, _routing_json_bytes
from services.connectors.telegram.repository._common import NOTIFICATIONS
from services.connectors.telegram.models import build_context_id
from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger
from services.connectors.telegram.runtime.routing_context import (
    BoundRef, CommandSpan, DocumentEvidence, ExplicitReferenceKind,
    RoutingContextSnapshot, SourceKind, json_digest,
)
from services.connectors.telegram.runtime.store import TelegramRuntimeStore
from tests.fixtures.telegram_contextual import BASE_TIME
from tests.fixtures.telegram_contextual import guard_firestore_enumeration
from tests.fixtures.firestore_fake import FakeDocumentReference, FakeTransaction

pytestmark = pytest.mark.offline


def _setup(client, text="move doc-B by dialog №2", *, selected_doc="doc-A",
           chat_id=42017, thread_id=73, update_id=76001):
    repository = TelegramRepository(client=client)
    accepted = repository.accept_command(
        update_id, chat_id=chat_id, thread_id=thread_id, user_id=17,
        command=parse_command(text), now=BASE_TIME,
    )
    context_ref = client.collection(SESSIONS).document(accepted.context_id)
    lease = claim_head(client, context_ref, owner="capture-worker", now=BASE_TIME)
    conversation_id = "selected-conversation"
    for doc_id, number, conv_id in (
        (selected_doc, 1, conversation_id), ("doc-B", 2, "conversation-B"),
    ):
        context_ref.collection("dialogs").document(str(number)).set({
            "doc_id": doc_id, "dialog_number": number, "conversation_id": conv_id,
            "draft_id": None,
        })
        context_ref.collection("document_bindings").document(
            hashlib.sha256(doc_id.encode()).hexdigest()
        ).set({"doc_id": doc_id, "dialog_number": number,
               "conversation_id": conv_id})
        client.collection(COLLECTION_NAME).document(doc_id).set({
            "source_file": f"{doc_id}.doc", "dialog_date": "2026-04-03",
            "parent_id": "folder-root",
        })
    repository.ensure_conversation(
        accepted.context_id, conversation_id, kind="document", doc_id=selected_doc,
        dialog_number=1, queue_operation_id=accepted.operation_id,
        queue_owner=lease.owner, queue_fence=lease.fence,
        select_if_unselected=True, now=BASE_TIME,
    )
    repository.append_turn(
        accepted.context_id, conversation_id, turn_id="prior-1",
        operation_id="prior-op", role="user", parts=[{"text": "שלום 🌍 " * 80}],
        now=BASE_TIME,
    )
    repository.append_turn(
        accepted.context_id, conversation_id, turn_id="prior-2",
        operation_id="prior-op-2", role="model", parts=[{"text": "Earlier answer."}],
        now=BASE_TIME + timedelta(seconds=1),
    )
    store = TelegramRuntimeStore(
        repository, context_id=accepted.context_id, operation_id=accepted.operation_id,
    )
    return repository, accepted, lease, store, conversation_id


def _ledger_loader(client, context_id, operation_id, *, now=BASE_TIME, owner="capture-worker", fence=1):
    return FirestoreAgentLedger(
        client, context_id=context_id, operation_id=operation_id,
        owner=owner, fence=fence, clock=lambda: now,
    )

def _capture(store, lease, client, actor_id=17, *, now=BASE_TIME):
    ledger = _ledger_loader(client, store.context_id, store.operation_id,
                            now=now, owner=lease.owner, fence=lease.fence)
    return store.capture_routing_context(
        actor_id=actor_id, owner=lease.owner, fence=lease.fence,
        clock=lambda: now, search_exchange_loader=ledger.load_search_exchange,
    )


def test_capture_projects_complete_history_and_replay_ignores_mutable_selection_and_catalog(
        firestore_fake, monkeypatch):
    client = firestore_fake.client_class()
    repository, accepted, lease, store, conversation_id = _setup(client)
    with guard_firestore_enumeration() as streams:
        snapshot = _capture(store, lease, client)
    assert all(event["collection"].rsplit("/", 1)[-1] not in {
        COLLECTION_NAME, "dialogs", "telegram_sessions",
    } for event in streams)
    history = snapshot.full_history_ref.payload["turns"]
    assert [turn["role"] for turn in history] == ["user", "model"]
    assert "שלום 🌍 " * 80 in history[0]["parts"][0]["text"]
    assert snapshot.selected_doc_id == "doc-A"
    assert snapshot.document_evidence["doc-A"][0].source_kind is SourceKind.ACTIVE_DOCUMENT
    clone = snapshot.to_dict()
    clone["full_history_ref"]["payload"]["turns"][0]["parts"][0]["text"] = "mutated clone"
    assert snapshot.full_history_ref.payload["turns"][0]["parts"][0]["text"] != "mutated clone"
    assert snapshot.full_history_ref.payload["conversation"]["dialog_number"] == 1

    # Mutable changes and accepted candidate cleanup cannot alter a saved capture.
    operation_ref = client.collection(SESSIONS).document(accepted.context_id).collection(
        "operations").document(accepted.operation_id)
    command_ref = _document_ref(client, operation_ref.get().to_dict()["input_ref"])
    command_ref.update({"candidate": firestore.DELETE_FIELD})
    client.collection(SESSIONS).document(accepted.context_id).update({
        "active_conversation_id": "a-different-conversation",
    })
    repository.append_turn(
        accepted.context_id, conversation_id, turn_id="new-after-capture",
        operation_id="later", role="user", parts=[{"text": "changed"}],
        now=BASE_TIME + timedelta(seconds=2),
    )
    client.collection(COLLECTION_NAME).document("doc-A").set({"source_file": "changed"})

    original_get = FakeDocumentReference.get
    blocked = {f"{SESSIONS}/{accepted.context_id}", f"{COLLECTION_NAME}/doc-A"}

    def guarded_get(reference, *args, **kwargs):
        path = reference.path
        if (path in blocked or f"{SESSIONS}/{accepted.context_id}/conversations/" in path
                or f"{SESSIONS}/{accepted.context_id}/dialogs/" in path
                or f"{SESSIONS}/{accepted.context_id}/document_bindings/" in path):
            raise AssertionError(f"replay read mutable routing input: {path}")
        return original_get(reference, *args, **kwargs)

    monkeypatch.setattr(FakeDocumentReference, "get", guarded_get)
    client.collection(SESSIONS).document(accepted.context_id).update({
        "lock_expires_at": BASE_TIME - timedelta(seconds=1),
    })
    replay = _capture(store, lease, client)
    assert replay.to_dict() == snapshot.to_dict()
    assert replay.full_history_ref.payload["turns"] == history


def test_stable_unselected_chat_history_is_captured_and_invalid_selection_fails_closed(firestore_fake):
    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    accepted = repository.accept_command(
        76100, chat_id=43000, thread_id=73, user_id=17,
        command=parse_command("use the previous chat"), now=BASE_TIME,
    )
    context_ref = client.collection(SESSIONS).document(accepted.context_id)
    lease = claim_head(client, context_ref, owner="stable-chat-worker", now=BASE_TIME)
    stable_chat = "telegram-chat-" + hashlib.sha256(accepted.context_id.encode()).hexdigest()
    repository.ensure_conversation(accepted.context_id, stable_chat, kind="chat", now=BASE_TIME)
    repository.append_turn(
        accepted.context_id, stable_chat, turn_id="stable-chat-turn",
        operation_id="older-chat-op", role="user", parts=[{"text": "keep this history 🗒️"}],
        now=BASE_TIME,
    )
    store = TelegramRuntimeStore(repository, context_id=accepted.context_id,
                                 operation_id=accepted.operation_id)
    captured = _capture(store, lease, client)
    assert captured.selected_conversation_id == stable_chat
    assert captured.full_history_ref.payload["turns"][0]["parts"][0]["text"] == "keep this history 🗒️"

    # A non-null malformed pointer cannot be mistaken for absent selection.
    context_ref.update({"active_conversation_id": 42})
    next_accepted = repository.accept_command(
        76101, chat_id=43001, thread_id=73, user_id=17,
        command=parse_command("malformed pointer"), now=BASE_TIME + timedelta(minutes=1),
    )
    next_context_ref = client.collection(SESSIONS).document(next_accepted.context_id)
    next_lease = claim_head(client, next_context_ref, owner="malformed-pointer-worker",
                            now=BASE_TIME + timedelta(minutes=1))
    next_context_ref.update({"active_conversation_id": 42})
    next_store = TelegramRuntimeStore(repository, context_id=next_accepted.context_id,
                                      operation_id=next_accepted.operation_id)
    with pytest.raises(ValueError):
        _capture(next_store, next_lease, client, now=BASE_TIME + timedelta(minutes=1))


def test_genuinely_absent_stable_chat_has_empty_history(firestore_fake):
    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    accepted = repository.accept_command(
        76110, chat_id=43010, thread_id=73, user_id=17,
        command=parse_command("first chat command"), now=BASE_TIME,
    )
    context_ref = client.collection(SESSIONS).document(accepted.context_id)
    lease = claim_head(client, context_ref, owner="empty-chat-worker", now=BASE_TIME)
    store = TelegramRuntimeStore(repository, context_id=accepted.context_id,
                                 operation_id=accepted.operation_id)
    captured = _capture(store, lease, client)
    assert captured.selected_conversation_kind == "chat"
    assert not captured.full_history_ref.payload["turns"]
    assert captured.full_history_ref.payload["conversation"]["kind"] == "chat"


def test_replay_rejects_semantically_corrupt_saved_snapshot_without_recapture(firestore_fake):
    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    accepted = repository.accept_command(
        76111, chat_id=43011, thread_id=73, user_id=17,
        command=parse_command("replay integrity"), now=BASE_TIME,
    )
    context_ref = client.collection(SESSIONS).document(accepted.context_id)
    lease = claim_head(client, context_ref, owner="integrity-worker", now=BASE_TIME)
    store = TelegramRuntimeStore(repository, context_id=accepted.context_id,
                                 operation_id=accepted.operation_id)
    _capture(store, lease, client)
    checkpoint_ref = (context_ref.collection("operations").document(accepted.operation_id)
                      .collection("routing_checkpoints").document("context"))
    checkpoint = checkpoint_ref.get().to_dict()
    envelope = deepcopy(checkpoint["payload"]["_inline"])
    envelope["value"]["accepted_command_ref"]["payload"]["text"] = "forged replay text"
    encoded = _routing_json_bytes(envelope)
    digest = hashlib.sha256(encoded).hexdigest()
    checkpoint["digest"] = digest
    checkpoint["byte_length"] = len(encoded)
    checkpoint["payload"] = {
        "_inline": envelope, "sha256": digest, "byte_length": len(encoded),
    }
    checkpoint_ref.set(checkpoint)
    with pytest.raises(ValueError):
        _capture(store, lease, client)


def test_local_number_evidence_cannot_cross_conversation_without_canonical_dialog_anchor():
    digest = "a" * 64
    span = CommandSpan(ExplicitReferenceKind.LOCAL_NUMBER, 0, 3, "No2", 2)
    source = BoundRef("unanchored", digest, "ctx", 17, "foreign-conversation", "op")
    forged = DocumentEvidence(
        "doc-B", SourceKind.LOCAL_NUMBER, source, "ctx", 17,
        "foreign-conversation", "op", command_span=span,
    )
    with pytest.raises(ValueError, match="outside captured context"):
        RoutingContextSnapshot(
            context_id="ctx", actor_id=17, operation_id="op",
            selected_conversation_id="selected-conversation",
            selected_conversation_kind="chat", accepted_command="No2",
            accepted_command_ref=BoundRef(
                "telegram_sessions/ctx/commands/1", digest, "ctx", 17,
                "selected-conversation", "op", payload={"text": "No2"},
            ),
            full_history_ref=BoundRef(
                "telegram_sessions/ctx/conversations/selected-conversation/turns",
                digest, "ctx", 17, "selected-conversation", "op", payload={"turns": []},
            ),
            explicit_references=(span,), document_evidence={"doc-B": (forged,)},
        )


def test_explicit_id_number_and_url_values_must_match_their_accepted_spans(firestore_fake):
    client = firestore_fake.client_class()
    text = "move doc-B using No2"
    repository, accepted, lease, store, _ = _setup(
        client, text, chat_id=43002, update_id=76102,
    )
    client.collection(COLLECTION_NAME).document("doc-C").set({"source_file": "doc-C.doc"})
    id_start = text.index("doc-B")
    number_start = text.index("No2")
    forged_id = CommandSpan(ExplicitReferenceKind.ID, id_start, id_start + 5,
                            "doc-B", "doc-C")
    forged_number = CommandSpan(ExplicitReferenceKind.LOCAL_NUMBER, number_start,
                                number_start + 3, "No2", 3)
    third_dialog = client.collection(SESSIONS).document(accepted.context_id)
    third_dialog.collection("dialogs").document("3").set({
        "dialog_number": 3, "doc_id": "doc-C", "conversation_id": "conversation-C",
    })
    third_dialog.collection("document_bindings").document(
        hashlib.sha256(b"doc-C").hexdigest()
    ).set({"dialog_number": 3, "doc_id": "doc-C", "conversation_id": "conversation-C"})
    trusted_loader = _ledger_loader(client, accepted.context_id, accepted.operation_id,
                                    owner=lease.owner, fence=lease.fence)
    with pytest.raises(ValueError):
        store.capture_routing_context(
            actor_id=17, owner=lease.owner, fence=lease.fence, clock=lambda: BASE_TIME,
            search_exchange_loader=trusted_loader.load_search_exchange,
            explicit_references=(forged_id, forged_number),
        )

    spoofed_url = "https://docs.google.com.evil.example/document/d/doc-B/edit"
    url_command = "use " + spoofed_url
    repository2, accepted2, lease2, store2, _ = _setup(
        client, url_command, chat_id=43003, update_id=76103,
    )
    url_start = url_command.index(spoofed_url)
    url_span = CommandSpan(ExplicitReferenceKind.URL, url_start, len(url_command),
                           spoofed_url, spoofed_url)
    trusted_loader2 = _ledger_loader(client, accepted2.context_id, accepted2.operation_id,
                                     owner=lease2.owner, fence=lease2.fence)
    unresolved = store2.capture_routing_context(
        actor_id=17, owner=lease2.owner, fence=lease2.fence, clock=lambda: BASE_TIME,
        search_exchange_loader=trusted_loader2.load_search_exchange,
        explicit_references=(url_span,),
    )
    assert "doc-B" not in unresolved.document_evidence or all(
        evidence.source_kind is not SourceKind.EXPLICIT_REFERENCE
        for evidence in unresolved.document_evidence["doc-B"]
    )

    valid_url = "https://docs.google.com/document/d/doc-B/edit"
    valid_command = "use " + valid_url
    _, accepted3, lease3, store3, _ = _setup(
        client, valid_command, chat_id=43004, update_id=76104,
    )
    valid_start = valid_command.index(valid_url)
    valid_span = CommandSpan(ExplicitReferenceKind.URL, valid_start, len(valid_command),
                             valid_url, valid_url)
    trusted_loader3 = _ledger_loader(client, accepted3.context_id, accepted3.operation_id,
                                     owner=lease3.owner, fence=lease3.fence)
    resolved = store3.capture_routing_context(
        actor_id=17, owner=lease3.owner, fence=lease3.fence, clock=lambda: BASE_TIME,
        search_exchange_loader=trusted_loader3.load_search_exchange,
        explicit_references=(valid_span,),
    )
    assert any(
        evidence.source_kind is SourceKind.EXPLICIT_REFERENCE
        for evidence in resolved.document_evidence["doc-B"]
    )


def test_capture_projects_exact_id_and_local_number_without_catalog_enumeration_or_fallback(firestore_fake):
    client = firestore_fake.client_class()
    text = "😀 move doc-B by dialog №2"
    repository, _, lease, store, _ = _setup(client, text)
    accepted_text = parse_command(text).text
    id_start = accepted_text.index("doc-B")
    number_start = accepted_text.index("No2")
    spans = (
        CommandSpan(ExplicitReferenceKind.ID, id_start, id_start + 5, "doc-B", "doc-B"),
        CommandSpan(ExplicitReferenceKind.LOCAL_NUMBER, number_start, number_start + 3, "No2", 2),
    )
    trusted_loader = _ledger_loader(client, store.context_id, store.operation_id,
                                    owner=lease.owner, fence=lease.fence)
    snapshot = store.capture_routing_context(
        actor_id=17, owner=lease.owner, fence=lease.fence, clock=lambda: BASE_TIME,
        search_exchange_loader=trusted_loader.load_search_exchange,
        explicit_references=spans,
    )
    assert snapshot.selected_doc_id == "doc-A"
    assert snapshot.parsed_id == "doc-B" and snapshot.local_number == 2
    assert len(snapshot.document_evidence["doc-B"]) == 2
    assert {entry.source_kind for entry in snapshot.document_evidence["doc-B"]} == {
        SourceKind.EXPLICIT_REFERENCE, SourceKind.LOCAL_NUMBER,
    }
    assert "doc-A" in snapshot.document_evidence
    assert all("folder-root" == item.metadata["parent_id"]
               for item in snapshot.document_evidence["doc-B"])

    # A missing explicit ID is retained as a span but never substitutes the active doc.
    text2 = "move absent-doc"
    accepted2 = repository.accept_command(
        76002, chat_id=42017, thread_id=74, user_id=17,
        command=parse_command(text2), now=BASE_TIME + timedelta(minutes=1),
    )
    ctx_ref = client.collection(SESSIONS).document(accepted2.context_id)
    lease2 = claim_head(client, ctx_ref, owner="capture-worker-2", now=BASE_TIME + timedelta(minutes=1))
    repo2 = TelegramRepository(client=client)
    ctx_ref.collection("dialogs").document("1").set({
        "doc_id": "doc-A", "dialog_number": 1, "conversation_id": "selected-two",
        "draft_id": None,
    })
    ctx_ref.collection("document_bindings").document(
        hashlib.sha256(b"doc-A").hexdigest()
    ).set({"doc_id": "doc-A", "dialog_number": 1,
           "conversation_id": "selected-two"})
    repo2.ensure_conversation(
        accepted2.context_id, "selected-two", kind="document", doc_id="doc-A",
        dialog_number=1, queue_operation_id=accepted2.operation_id,
        queue_owner=lease2.owner, queue_fence=lease2.fence,
        select_if_unselected=True, now=BASE_TIME + timedelta(minutes=1),
    )
    store2 = TelegramRuntimeStore(repo2, context_id=accepted2.context_id,
                                  operation_id=accepted2.operation_id)
    span = CommandSpan(ExplicitReferenceKind.ID, 5, 15, "absent-doc", "absent-doc")
    trusted_loader2 = _ledger_loader(client, store2.context_id, store2.operation_id,
                                     owner=lease2.owner, fence=lease2.fence)
    missing = store2.capture_routing_context(
        actor_id=17, owner=lease2.owner, fence=lease2.fence,
        clock=lambda: BASE_TIME + timedelta(minutes=1),
        search_exchange_loader=trusted_loader2.load_search_exchange,
        explicit_references=(span,),
    )
    assert missing.selected_doc_id == "doc-A"
    assert "absent-doc" not in missing.document_evidence
    assert missing.explicit_references == (span,)


def test_capture_hydrates_real_verified_search_exchange_and_rejects_corruption(firestore_fake):
    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    prior = repository.accept_command(
        76010, chat_id=42017, thread_id=73, user_id=17,
        command=parse_command("prior search"), now=BASE_TIME,
    )
    context_ref = client.collection(SESSIONS).document(prior.context_id)
    prior_lease = claim_head(client, context_ref, owner="prior-search-worker", now=BASE_TIME)
    conversation_id = "verified-search-conversation"
    filters = {"keywords": ["price"], "folder_path": ["Records"],
               "date_from": "2026-01-01", "date_to": "2026-12-31"}
    repository.ensure_conversation(
        prior.context_id, conversation_id, kind="search", search_filters=filters,
        queue_operation_id=prior.operation_id, queue_owner=prior_lease.owner,
        queue_fence=prior_lease.fence, select_if_unselected=True, now=BASE_TIME,
    )
    ledger = FirestoreAgentLedger(
        client, context_id=prior.context_id, operation_id=prior.operation_id,
        owner=prior_lease.owner, fence=prior_lease.fence, clock=lambda: BASE_TIME,
    )
    exchange = {
        "call": {"id": "query-1", "name": "query", "arguments": {
                     "query": "цена", **filters,
                 },
                 "ordinal": 0, "round": 0},
        "result": {"segments": [{"doc_id": "doc-B", "text": "東京 🧾 €42 " * 40000}],
                   "total_matches": 1, "returned_segments": 1, "truncated": False},
    }
    client.collection(COLLECTION_NAME).document("doc-B").set({
        "doc_id": "doc-B", "source_file": "verified-search-doc.txt",
    })
    ref = ledger.save_search_exchange(
        conversation_id, checkpoint_conversation_id="search-checkpoint",
        call=exchange["call"], canonical_call_id="query-1", result=exchange["result"],
    )
    assert ref["payload"]["byte_length"] > MAX_PART_BYTES
    manifest = _document_ref(client, ref["payload"]["_input_ref"]).get().to_dict()
    assert manifest["part_count"] > 1
    repository.append_turn(
        prior.context_id, conversation_id, turn_id="search-tool-ref",
        operation_id=prior.operation_id, role="tool", parts=[{
            "type": "search_exchange_ref", "reference": ref, "call_id": "query-1",
            "ordinal": 0, "checkpoint_conversation_id": "search-checkpoint",
        }], tool_correlation={"name": "query", "arguments": {"query": "цена"}},
        now=BASE_TIME + timedelta(seconds=2),
    )
    assert complete_head(
        client, context_ref, operation_id=prior.operation_id,
        owner=prior_lease.owner, fence=prior_lease.fence,
        now=BASE_TIME + timedelta(seconds=3),
    )
    accepted = repository.accept_command(
        76011, chat_id=42017, thread_id=73, user_id=17,
        command=parse_command("follow up"), now=BASE_TIME + timedelta(seconds=4),
    )
    lease = claim_head(client, context_ref, owner="capture-worker", now=BASE_TIME + timedelta(seconds=4))
    store = TelegramRuntimeStore(repository, context_id=accepted.context_id,
                                 operation_id=accepted.operation_id)

    def load(reference, *, conversation_id, operation_id, checkpoint_conversation_id,
             call_id, ordinal):
        source = FirestoreAgentLedger(
            client, context_id=prior.context_id, operation_id=operation_id,
            owner=prior_lease.owner, fence=prior_lease.fence,
            clock=lambda: BASE_TIME + timedelta(seconds=4),
        )
        return source.load_search_exchange(
            reference, conversation_id=conversation_id, operation_id=operation_id,
            checkpoint_conversation_id=checkpoint_conversation_id,
            call_id=call_id, ordinal=ordinal,
        )

    snapshot = store.capture_routing_context(
        actor_id=17, owner=lease.owner, fence=lease.fence,
        clock=lambda: BASE_TIME + timedelta(seconds=4), search_exchange_loader=load,
    )
    serialized_history = snapshot.full_history_ref.payload["turns"]
    assert any("Untrusted prior search tool exchange data" in part.get("text", "")
               and "東京 🧾 €42" in part["text"]
               for turn in serialized_history for part in turn.get("parts", []))
    assert any(item.source_kind is SourceKind.VERIFIED_EXCHANGE
               for item in snapshot.document_evidence["doc-B"])

    ref_path = ref["payload"]["_input_ref"]
    manifest_ref = _document_ref(client, ref_path)
    manifest = manifest_ref.get().to_dict()
    manifest["sha256"] = "0" * 64
    manifest_ref.set(manifest)
    # No fallback to empty history if a source exchange becomes corrupt before capture.
    # Corrupting the mutable source after capture cannot alter replay.
    with pytest.raises((ValueError, PermissionError)):
        load(ref, conversation_id=conversation_id, operation_id=prior.operation_id,
               checkpoint_conversation_id="search-checkpoint", call_id="query-1", ordinal=0)
    replay = store.capture_routing_context(
        actor_id=17, owner=lease.owner, fence=lease.fence,
        clock=lambda: BASE_TIME + timedelta(seconds=4), search_exchange_loader=load,
    )
    assert replay.to_dict() == snapshot.to_dict()

    assert complete_head(
        client, context_ref, operation_id=accepted.operation_id,
        owner=lease.owner, fence=lease.fence, now=BASE_TIME + timedelta(seconds=5),
    )
    next_command = repository.accept_command(
        76012, chat_id=42017, thread_id=73, user_id=17,
        command=parse_command("capture after corruption"), now=BASE_TIME + timedelta(seconds=6),
    )
    next_lease = claim_head(client, context_ref, owner="capture-worker-next",
                            now=BASE_TIME + timedelta(seconds=6))
    next_store = TelegramRuntimeStore(
        repository, context_id=next_command.context_id, operation_id=next_command.operation_id,
    )
    with pytest.raises((ValueError, PermissionError)):
        next_store.capture_routing_context(
            actor_id=17, owner=next_lease.owner, fence=next_lease.fence,
            clock=lambda: BASE_TIME + timedelta(seconds=6), search_exchange_loader=load,
        )


def test_first_capture_requires_live_lease_and_exact_accepted_actor_scope(firestore_fake):
    client = firestore_fake.client_class()
    _, accepted, lease, store, _ = _setup(client)
    with pytest.raises(PermissionError):
        _capture(store, lease, client, actor_id=18)

    context_ref = client.collection(SESSIONS).document(accepted.context_id)
    context_ref.update({"lock_expires_at": BASE_TIME - timedelta(seconds=1)})
    with pytest.raises(PermissionError):
        _capture(store, lease, client)


def test_addressed_metadata_read_errors_propagate_without_empty_fallback(firestore_fake, monkeypatch):
    client = firestore_fake.client_class()
    _, accepted, lease, store, _ = _setup(client)
    original_get = FakeDocumentReference.get

    def failing_get(reference, *args, **kwargs):
        if reference.path == f"{COLLECTION_NAME}/doc-A":
            raise RuntimeError("metadata read failed")
        return original_get(reference, *args, **kwargs)

    monkeypatch.setattr(FakeDocumentReference, "get", failing_get)
    with pytest.raises(RuntimeError, match="metadata read failed"):
        _capture(store, lease, client)


def test_real_question_delivery_datetime_hint_survives_capture_and_replay(firestore_fake):
    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id = build_context_id(43020, 73)
    context_ref = client.collection(SESSIONS).document(context_id)
    context_ref.set({"context_id": context_id, "pending_clarification_id": "clarify-1",
                     "clarification_generation": 1, "active_conversation_id": None})
    body = "Which destination should I use?"
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    clarification_ref = context_ref.collection("clarifications").document("clarify-1")
    clarification_ref.set({
        "clarification_id": "clarify-1", "context_id": context_id, "protocol": "clarification-v1",
        "actor_id": 17, "state": "waiting", "generation": 1,
        "created_at": BASE_TIME - timedelta(minutes=5),
    })
    clarification_ref.collection("questions").document("1").set({
        "clarification_id": "clarify-1", "context_id": context_id, "actor_id": 17,
        "generation": 1, "prompt_notification_id": "question-notice-1",
        "prompt_notification_revision": 1, "prompt_digest": digest,
        "created_at": BASE_TIME - timedelta(minutes=3),
    })
    client.collection(NOTIFICATIONS).document("question-notice-1").set({
        "notification_id": "question-notice-1", "kind": "clarification",
        "context_id": context_id, "actor_id": 17, "clarification_id": "clarify-1",
        "generation": 1, "chat_id": 43020, "thread_id": 73,
        "revision": 1, "message_id": 123, "state": "delivered", "body": body,
        "delivered_at": BASE_TIME - timedelta(minutes=2),
    })
    accepted = repository.accept_command(
        76120, chat_id=43020, thread_id=73, user_id=17,
        command=parse_command("clarification answer"), message_id=900,
        transport_timestamp=int((BASE_TIME + timedelta(minutes=1)).timestamp()),
        reply_to_message_id=123, now=BASE_TIME + timedelta(minutes=1),
    )
    lease = claim_head(client, context_ref, owner="question-answer-worker",
                       now=BASE_TIME + timedelta(minutes=1))
    store = TelegramRuntimeStore(repository, context_id=context_id,
                                 operation_id=accepted.operation_id)
    captured = _capture(store, lease, client, now=BASE_TIME + timedelta(minutes=1))
    hint = captured.accepted_command_ref.payload["observed_question_delivery_ref"]
    assert hint["delivered_at"].endswith("+00:00")
    assert hint["observed_at"].endswith("+00:00")
    op_ref = context_ref.collection("operations").document(accepted.operation_id)
    input_ref = _document_ref(client, op_ref.get().to_dict()["input_ref"])
    input_ref.update({"candidate": firestore.DELETE_FIELD})
    context_ref.update({"lock_expires_at": BASE_TIME - timedelta(seconds=1)})
    replay = _capture(store, lease, client, now=BASE_TIME + timedelta(minutes=1))
    assert replay.to_dict() == captured.to_dict()


@pytest.mark.parametrize("native_call_id", (None, ""))
def test_inline_search_evidence_points_to_canonical_history_turn(firestore_fake, native_call_id):
    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    prior = repository.accept_command(
        76121, chat_id=43021, thread_id=73, user_id=17,
        command=parse_command("prior scoped search"), now=BASE_TIME,
    )
    context_ref = client.collection(SESSIONS).document(prior.context_id)
    prior_lease = claim_head(client, context_ref, owner="inline-search-prior", now=BASE_TIME)
    conversation_id = "inline-search-conversation"
    filters = {"keywords": ["budget"], "folder_path": ["Records"],
               "date_from": "2026-01-01", "date_to": "2026-12-31"}
    repository.ensure_conversation(
        prior.context_id, conversation_id, kind="search", search_filters=filters,
        queue_operation_id=prior.operation_id, queue_owner=prior_lease.owner,
        queue_fence=prior_lease.fence, select_if_unselected=True, now=BASE_TIME,
    )
    client.collection(COLLECTION_NAME).document("doc-inline").set({
        "doc_id": "doc-inline", "source_file": "inline-search.txt",
    })
    ledger = FirestoreAgentLedger(
        client, context_id=prior.context_id, operation_id=prior.operation_id,
        owner=prior_lease.owner, fence=prior_lease.fence, clock=lambda: BASE_TIME,
    )
    call = {"id": native_call_id, "name": "query", "arguments": {
        **filters, "query": "budget", "max_results": 10,
    }, "ordinal": 0, "round": 0}
    result = {"segments": [{"doc_id": "doc-inline", "text": "approved result"}],
              "total_matches": 1, "returned_segments": 1, "truncated": False}
    canonical_call_id = f"telegram-call-{prior.operation_id}-0"
    exchange_ref = ledger.save_search_exchange(
        conversation_id, checkpoint_conversation_id="inline-checkpoint",
        call=call, canonical_call_id=canonical_call_id, result=result,
    )
    repository.append_turn(
        prior.context_id, conversation_id, turn_id="inline-tool-turn",
        operation_id=prior.operation_id, role="tool", parts=[{
            "type": "search_exchange_ref", "reference": exchange_ref,
            "call_id": canonical_call_id, "ordinal": 0,
            "checkpoint_conversation_id": "inline-checkpoint",
        }], tool_correlation={"name": "query", "arguments": call["arguments"]},
        now=BASE_TIME + timedelta(seconds=1),
    )
    for ordinal, name, arguments, doc_id in (
        (1, "move_document", {"doc_id": "doc-unsupported"}, "doc-unsupported"),
        (2, "query", {"query": "budget", "folder_path": [],
                      "date_from": "2025-01-01", "date_to": "2027-01-01"},
         "doc-out-of-bounds"),
        (3, "query", {"query": "budget", **filters}, "doc-error-result"),
    ):
        client.collection(COLLECTION_NAME).document(doc_id).set({
            "doc_id": doc_id, "source_file": f"{doc_id}.txt",
        })
        negative_call = {"id": f"call-{ordinal}", "name": name,
                         "arguments": arguments, "ordinal": ordinal, "round": 0}
        negative_ref = ledger.save_search_exchange(
            conversation_id, checkpoint_conversation_id="inline-checkpoint",
            call=negative_call, canonical_call_id=negative_call["id"],
            result={"segments": [{"doc_id": doc_id, "text": "untrusted"}],
                    **({"error": "retrieval failed"} if ordinal == 3 else {})},
        )
        repository.append_turn(
            prior.context_id, conversation_id, turn_id=f"inline-negative-{ordinal}",
            operation_id=prior.operation_id, role="tool", parts=[{
                "type": "search_exchange_ref", "reference": negative_ref,
                "call_id": negative_call["id"], "ordinal": ordinal,
                "checkpoint_conversation_id": "inline-checkpoint",
            }], tool_correlation={"name": name, "arguments": arguments},
            now=BASE_TIME + timedelta(seconds=ordinal + 1),
        )
    assert complete_head(client, context_ref, operation_id=prior.operation_id,
                         owner=prior_lease.owner, fence=prior_lease.fence,
                         now=BASE_TIME + timedelta(seconds=2))
    accepted = repository.accept_command(
        76122, chat_id=43021, thread_id=73, user_id=17,
        command=parse_command("follow up"), now=BASE_TIME + timedelta(seconds=3),
    )
    lease = claim_head(client, context_ref, owner="inline-search-capture",
                       now=BASE_TIME + timedelta(seconds=3))
    store = TelegramRuntimeStore(repository, context_id=accepted.context_id,
                                 operation_id=accepted.operation_id)
    source_ledger = FirestoreAgentLedger(
        client, context_id=prior.context_id, operation_id=prior.operation_id,
        owner=prior_lease.owner, fence=prior_lease.fence,
        clock=lambda: BASE_TIME + timedelta(seconds=3),
    )
    captured = store.capture_routing_context(
        actor_id=17, owner=lease.owner, fence=lease.fence,
        clock=lambda: BASE_TIME + timedelta(seconds=3),
        search_exchange_loader=source_ledger.load_search_exchange,
    )
    proof = next(item for item in captured.document_evidence["doc-inline"]
                 if item.source_kind is SourceKind.VERIFIED_EXCHANGE)
    assert proof.source.ref == (
        f"{SESSIONS}/{prior.context_id}/conversations/{conversation_id}/turns/inline-tool-turn"
    )
    stored_turn = _document_ref(client, proof.source.ref).get().to_dict()
    assert stored_turn["parts"][0]["reference"] == exchange_ref
    assert proof.source.call_id == canonical_call_id
    assert proof.source.digest == json_digest(proof.source.payload["turn"])
    loaded = source_ledger.load_search_exchange(
        exchange_ref, conversation_id=conversation_id, operation_id=prior.operation_id,
        checkpoint_conversation_id="inline-checkpoint", call_id=canonical_call_id, ordinal=0,
    )
    assert loaded["call"]["id"] == native_call_id
    assert "doc-unsupported" not in captured.document_evidence
    assert "doc-out-of-bounds" not in captured.document_evidence
    assert "doc-error-result" not in captured.document_evidence


def test_accepted_collecting_draft_cutoff_stays_separate_from_selected_document(firestore_fake):
    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    context_id = build_context_id(43022, 73)
    context_ref = client.collection(SESSIONS).document(context_id)
    context_ref.set({"context_id": context_id, "active_draft_id": "draft-captured",
                     "next_intake_sequence": 5, "active_conversation_id": None})
    context_ref.collection("drafts").document("draft-captured").set({
        "draft_id": "draft-captured", "state": "collecting",
        "ordered_message_ids": [101, 102],
    })
    accepted = repository.accept_command(
        76123, chat_id=43022, thread_id=73, user_id=17,
        command=parse_command("continue this request"), now=BASE_TIME,
    )
    lease = claim_head(client, context_ref, owner="draft-cutoff-worker", now=BASE_TIME)
    conversation_id = "selected-document-independent-of-draft"
    context_ref.collection("dialogs").document("1").set({
        "dialog_number": 1, "doc_id": "doc-selected", "conversation_id": conversation_id,
    })
    context_ref.collection("document_bindings").document(
        hashlib.sha256(b"doc-selected").hexdigest()
    ).set({"dialog_number": 1, "doc_id": "doc-selected", "conversation_id": conversation_id})
    client.collection(COLLECTION_NAME).document("doc-selected").set({
        "doc_id": "doc-selected", "source_file": "selected.txt",
    })
    repository.ensure_conversation(
        context_id, conversation_id, kind="document", doc_id="doc-selected", dialog_number=1,
        queue_operation_id=accepted.operation_id, queue_owner=lease.owner,
        queue_fence=lease.fence, select_if_unselected=True, now=BASE_TIME,
    )
    store = TelegramRuntimeStore(repository, context_id=context_id,
                                 operation_id=accepted.operation_id)
    snapshot = _capture(store, lease, client)
    assert snapshot.selected_doc_id == "doc-selected"
    assert snapshot.full_history_ref.payload["conversation"]["doc_id"] == "doc-selected"
    assert snapshot.accepted_command_ref.payload["draft_id"] == "draft-captured"
    assert snapshot.accepted_command_ref.payload["cutoff_message_ids"] == (101, 102)
    assert snapshot.accepted_command_ref.payload["cutoff_intake_sequence"] == 4


def test_selected_draft_projection_and_malformed_selected_record_fail_closed(firestore_fake):
    client = firestore_fake.client_class()
    repository = TelegramRepository(client=client)
    accepted = repository.accept_command(
        76124, chat_id=43023, thread_id=73, user_id=17,
        command=parse_command("selected draft"), now=BASE_TIME,
    )
    context_ref = client.collection(SESSIONS).document(accepted.context_id)
    lease = claim_head(client, context_ref, owner="selected-draft-worker", now=BASE_TIME)
    conversation_id = "selected-draft-conversation"
    context_ref.collection("dialogs").document("7").set({
        "dialog_number": 7, "draft_id": "draft-selected", "conversation_id": conversation_id,
    })
    repository.ensure_conversation(
        accepted.context_id, conversation_id, kind="draft", draft_id="draft-selected",
        dialog_number=7, queue_operation_id=accepted.operation_id,
        queue_owner=lease.owner, queue_fence=lease.fence,
        select_if_unselected=True, now=BASE_TIME,
    )
    store = TelegramRuntimeStore(repository, context_id=accepted.context_id,
                                 operation_id=accepted.operation_id)
    selected = _capture(store, lease, client)
    assert selected.selected_conversation_kind == "draft"
    assert selected.full_history_ref.payload["conversation"]["draft_id"] == "draft-selected"
    assert selected.full_history_ref.payload["conversation"]["dialog_number"] == 7

    # A raw stored record that contradicts its kind cannot be projected as a document.
    assert complete_head(client, context_ref, operation_id=accepted.operation_id,
                         owner=lease.owner, fence=lease.fence,
                         now=BASE_TIME + timedelta(seconds=1))
    # Use a fresh accepted operation so the saved checkpoint cannot mask the malformed row.
    next_command = repository.accept_command(
        76125, chat_id=43023, thread_id=73, user_id=17,
        command=parse_command("malformed selected kind"), now=BASE_TIME + timedelta(minutes=1),
    )
    next_lease = claim_head(client, context_ref, owner="malformed-kind-worker",
                            now=BASE_TIME + timedelta(minutes=1))
    context_ref.collection("conversations").document(conversation_id).update({
        "kind": "chat", "doc_id": "doc-impossible", "draft_id": None,
        "dialog_number": None,
    })
    next_store = TelegramRuntimeStore(repository, context_id=accepted.context_id,
                                      operation_id=next_command.operation_id)
    with pytest.raises(ValueError):
        _capture(next_store, next_lease, client, now=BASE_TIME + timedelta(minutes=1))


@pytest.mark.parametrize("kind", (ExplicitReferenceKind.ID, ExplicitReferenceKind.LOCAL_NUMBER))
def test_raw_and_parsed_explicit_identity_mismatches_fail_independently(firestore_fake, kind):
    client = firestore_fake.client_class()
    text = "move doc-B No2"
    _, accepted, lease, store, _ = _setup(client, text, chat_id=43024,
                                          update_id=76126 if kind is ExplicitReferenceKind.ID else 76127)
    if kind is ExplicitReferenceKind.ID:
        start = text.index("doc-B")
        span = CommandSpan(kind, start, start + len("doc-B"), "doc-B", "doc-C")
    else:
        start = text.index("No2")
        span = CommandSpan(kind, start, start + len("No2"), "No2", 3)
    ledger = _ledger_loader(client, accepted.context_id, accepted.operation_id,
                            owner=lease.owner, fence=lease.fence)
    with pytest.raises(ValueError):
        store.capture_routing_context(
            actor_id=17, owner=lease.owner, fence=lease.fence,
            clock=lambda: BASE_TIME, search_exchange_loader=ledger.load_search_exchange,
            explicit_references=(span,),
        )


def test_scoped_capture_is_corpus_invariant_and_only_writes_checkpoint_artifacts(
        firestore_fake, monkeypatch):
    digests = []
    for count in (1, 324, 10_000):
        firestore_fake.reset()
        client = firestore_fake.client_class()
        _, accepted, lease, store, _ = _setup(client)
        batch = client.batch()
        for index in range(count):
            batch.set(client.collection(COLLECTION_NAME).document(f"unrelated-{index:05d}"), {
                "source_file": f"unrelated-{index:05d}.txt", "parent_id": "other-root",
            })
            if (index + 1) % 500 == 0:
                batch.commit()
                batch = client.batch()
        batch.commit()
        before_context = (client.collection(SESSIONS).document(accepted.context_id)
                          .get().to_dict())
        original_set, original_update = FakeDocumentReference.set, FakeDocumentReference.update
        original_tx_set = FakeTransaction.set
        writes = []

        def record_set(reference, *args, **kwargs):
            writes.append(reference.path)
            return original_set(reference, *args, **kwargs)

        def record_update(reference, *args, **kwargs):
            writes.append(reference.path)
            return original_update(reference, *args, **kwargs)

        def record_tx_set(transaction, reference, *args, **kwargs):
            writes.append(reference.path)
            return original_tx_set(transaction, reference, *args, **kwargs)

        monkeypatch.setattr(FakeDocumentReference, "set", record_set)
        monkeypatch.setattr(FakeDocumentReference, "update", record_update)
        monkeypatch.setattr(FakeTransaction, "set", record_tx_set)
        with guard_firestore_enumeration():
            snapshot = _capture(store, lease, client)
        digests.append(hashlib.sha256(_routing_json_bytes(snapshot.to_dict())).hexdigest())
        assert writes
        assert all("/routing_checkpoints/context" in path or
                   (path.startswith("dialog_operations/") and "/input_parts/" in path)
                   for path in writes)
        after_context = (client.collection(SESSIONS).document(accepted.context_id)
                         .get().to_dict())
        assert after_context.get("active_conversation_id") == before_context.get("active_conversation_id")
        assert after_context.get("last_doc_id") == before_context.get("last_doc_id")
        monkeypatch.setattr(FakeDocumentReference, "set", original_set)
        monkeypatch.setattr(FakeDocumentReference, "update", original_update)
        monkeypatch.setattr(FakeTransaction, "set", original_tx_set)
    assert len(set(digests)) == 1


def test_fresh_capture_rejects_changed_accepted_candidate_and_foreign_history(firestore_fake):
    client = firestore_fake.client_class()
    _, accepted, lease, store, conversation_id = _setup(client)
    operation = (client.collection(SESSIONS).document(accepted.context_id)
                 .collection("operations").document(accepted.operation_id).get().to_dict())
    command_ref = _document_ref(client, operation["input_ref"])
    command_ref.update({"candidate": "different accepted text"})
    with pytest.raises(ValueError, match="immutable input identity"):
        _capture(store, lease, client)

    client = firestore_fake.client_class()
    repository, accepted, lease, store, conversation_id = _setup(
        client, chat_id=42018, update_id=76021,
    )
    turn_ref = (client.collection(SESSIONS).document(accepted.context_id)
                .collection("conversations").document(conversation_id)
                .collection("turns").document("prior-1"))
    turn_ref.update({"conversation_id": "foreign-conversation"})
    with pytest.raises(ValueError, match="foreign or malformed"):
        _capture(store, lease, client)

    client = firestore_fake.client_class()
    repository, accepted, lease, store, conversation_id = _setup(
        client, chat_id=42019, update_id=76022,
    )
    (client.collection(SESSIONS).document(accepted.context_id)
     .collection("conversations").document(conversation_id)
     .collection("turns").document("prior-2").delete())
    with pytest.raises(ValueError, match="incomplete"):
        _capture(store, lease, client)

