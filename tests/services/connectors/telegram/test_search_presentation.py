"""Presentation contracts exercised through the existing Telegram runtime.

These are intentionally test-first contracts.  The native terminal tool is
submitted through Gemini's real response envelope and observed at the normal
agent/runtime/notification boundaries; no presentation implementation module
is imported here.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

pytestmark = pytest.mark.offline


def _bot_client(*, fail_send_number=None, before_send=None):
    from services.connectors.telegram.client import TelegramClient

    sent = []

    def handle(request):
        payload = json.loads(request.content.decode("utf-8"))
        sent.append({"method": request.url.path.rsplit("/", 1)[-1],
                     "text": payload.get("text") or payload.get("rich_message", {}).get("markdown"),
                     **payload})
        callback = before_send or getattr(client, "_before_send", None)
        if callable(callback):
            callback(payload, sent[-1])
        if fail_send_number == len(sent):
            return httpx.Response(200, json={
                "ok": False, "error_code": 400, "description": "synthetic delivery rejection",
            })
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 91000 + len(sent)}})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    client = TelegramClient("synthetic-presentation-token", http_client=http_client)
    client._test_sent = sent
    client._test_http_client = http_client
    return client


def _running_command(client, *, context_id="42017_73",
                     text="Find the three project discussions", now=None):
    """Create a real accepted command and hold its current FIFO lease."""
    from core.integrations.firestore.operation_queue import claim_head
    from services.connectors.telegram.commands import parse_command
    from services.connectors.telegram.repository import TelegramRepository

    now = now or datetime(2026, 4, 3, 12, tzinfo=timezone.utc)
    chat_text, thread_text = context_id.rsplit("_", 1)
    chat_id = int(chat_text)
    thread_id = None if thread_text == "default" else int(thread_text)
    repository = TelegramRepository(client=client)
    accepted = repository.accept_command(
        88101, chat_id=chat_id, thread_id=thread_id, user_id=17,
        command=parse_command(text), now=now,
    )
    assert accepted.operation_id
    context_ref = client.collection("telegram_sessions").document(context_id)
    lease = claim_head(client, context_ref, owner="presentation-test-worker", now=now)
    assert lease is not None and lease.operation_id == accepted.operation_id
    operation_ref = context_ref.collection("operations").document(accepted.operation_id)
    operation_snapshot = operation_ref.get()
    operation = operation_snapshot.to_dict()
    assert operation["state"] == "running"
    return repository, accepted, operation, lease, now


def _native_response(function_name, arguments):
    from google.genai import types

    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(parts=[types.Part(function_call=types.FunctionCall(
            name=function_name, args=arguments,
        ))]),
    )])


def _source_search_fixture(client_db):
    """Seed one accepted search, a complete native query round, and verified evidence."""
    from google.genai import types
    from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger

    repository, accepted, operation, lease, now = _running_command(
        client_db, text="Find the synthetic project discussions",
    )
    conversation_id = "search-" + hashlib.sha256(
        f"{accepted.context_id}\0{accepted.operation_id}".encode("utf-8")
    ).hexdigest()
    checkpoint_id = "telegram-global-" + hashlib.sha256(
        f"{accepted.context_id}\0{accepted.operation_id}".encode("utf-8")
    ).hexdigest()
    repository.ensure_conversation(
        accepted.context_id, conversation_id, kind="search",
        search_filters={"query": "synthetic project discussions", "keywords": None,
                        "folder_path": None, "date_from": None, "date_to": None},
        queue_operation_id=accepted.operation_id, queue_owner=lease.owner,
        queue_fence=lease.fence, select_if_unselected=True, now=now,
    )
    call = {"id": "query-1", "name": "query",
            "arguments": {"query": "synthetic project discussions"},
            "ordinal": 0, "round": 0}
    result = {"segments": [
        {"doc_id": "doc-c", "order_index": 3, "quote": "Gamma source."},
        {"doc_id": "doc-a", "order_index": 1, "quote": "Alpha source."},
        {"doc_id": "doc-b", "order_index": 2, "quote": "The exact verified sentence."},
    ], "metadata": {"truncated": False, "total_matches": 3, "returned_segments": 3}}
    ledger = FirestoreAgentLedger(
        client_db, context_id=accepted.context_id, operation_id=accepted.operation_id,
        owner=lease.owner, fence=lease.fence, clock=lambda: now,
    )
    reference = ledger.save_search_exchange(
        conversation_id, checkpoint_conversation_id=checkpoint_id,
        call=call, canonical_call_id=call["id"], result=result,
    )
    ledger.save_turn(
        checkpoint_id, accepted.operation_id,
        model_response=types.GenerateContentResponse(candidates=[types.Candidate(
            content=types.Content(parts=[types.Part(function_call=types.FunctionCall(
                id=call["id"], name=call["name"], args=call["arguments"],
            ))]),
        )]).model_dump(exclude_none=True),
        tool_calls=[call],
    )
    ledger.save_tool_result(checkpoint_id, accepted.operation_id, 0, result)
    verified = ledger.load_search_exchange(
        reference, conversation_id=conversation_id,
        operation_id=accepted.operation_id,
        checkpoint_conversation_id=checkpoint_id, call_id=call["id"], ordinal=0,
    )
    assert verified["call"] == call and verified["result"] == result
    repository.append_turn(
        accepted.context_id, conversation_id, turn_id="presentation-prior-user",
        operation_id=accepted.operation_id, role="user",
        parts=[{"text": "Find the synthetic project discussions"}], now=now,
        queue_operation_id=accepted.operation_id, queue_owner=lease.owner, queue_fence=lease.fence,
    )
    repository.append_turn(
        accepted.context_id, conversation_id, turn_id="presentation-prior-query",
        operation_id=accepted.operation_id, role="tool",
        parts=[{"type": "search_exchange_ref", "reference": reference,
                "call_id": call["id"], "ordinal": 0,
                "checkpoint_conversation_id": checkpoint_id}],
        tool_correlation={"name": "query", "arguments": call["arguments"]},
        source_refs=[{
            "context_id": accepted.context_id, "conversation_id": conversation_id,
            "operation_id": accepted.operation_id,
            "checkpoint_conversation_id": checkpoint_id,
            "call_id": call["id"], "ordinal": call["ordinal"],
            "reference": reference, "doc_id": row["doc_id"],
            "order_index": row["order_index"],
        } for row in result["segments"]], now=now,
        queue_operation_id=accepted.operation_id, queue_owner=lease.owner, queue_fence=lease.fence,
    )
    assert repository.persist_route(accepted.operation_id, {
        "status": "ok", "kind": "search", "search_protocol": "agent-v1",
        "search_session_id": conversation_id,
        "result_refs": ["doc-c", "doc-a", "doc-b"],
        "tool_choice": {"name": "query", "arguments": {
            "query": "synthetic project discussions",
        }, "link_requested": False},
    }, context_id=accepted.context_id, owner=lease.owner, fence=lease.fence, now=now)
    return {
        "repository": repository, "accepted": accepted, "operation": operation,
        "lease": lease, "now": now, "conversation_id": conversation_id,
        "checkpoint_id": checkpoint_id, "call": call, "result": result,
        "reference": reference, "ledger": ledger, "verified_exchange": verified,
    }


def _source_evidence(fixture, doc_id):
    row = next(item for item in fixture["result"]["segments"] if item["doc_id"] == doc_id)
    return {
        "context_id": fixture["accepted"].context_id,
        "conversation_id": fixture["conversation_id"],
        "operation_id": fixture["accepted"].operation_id,
        "checkpoint_conversation_id": fixture["checkpoint_id"],
        "call_id": fixture["call"]["id"], "ordinal": fixture["call"]["ordinal"],
        "reference": fixture["reference"],
        "doc_id": doc_id, "order_index": row["order_index"],
    }


def _list_block(fixture, *, list_id, doc_ids, labels=None):
    labels = labels or {"doc-a": "Alpha", "doc-b": "Beta", "doc-c": "Gamma"}
    return {"type": "document_list", "list_id": list_id, "entries": [
        {"position": position, "doc_id": doc_id, "label": labels[doc_id],
         "evidence_refs": [_source_evidence(fixture, doc_id)]}
        for position, doc_id in enumerate(doc_ids, 1)
    ]}


def _render_fixture_blocks(blocks):
    rendered = []
    for block in blocks:
        if block["type"] == "document_list":
            rendered.append("\n".join(
                f"{position}. {entry['label']}"
                for position, entry in enumerate(block["entries"], 1)
            ))
        elif block["type"] == "paragraph":
            rendered.append(block["text"])
        elif block["type"] == "quote":
            rendered.append("> " + block["text"])
        else:
            raise AssertionError(f"unhandled fixture block: {block['type']}")
    return "\n\n".join(rendered)


def _configure_writer_boundary(client_db, monkeypatch, sdk_fakes):
    """Use one coherent registered/live root for all shared writes."""
    from core.integrations.firestore.client import COLLECTION_NAME, FOLDERS_COLLECTION
    from tests.fixtures.shared_write_authorization import (
        ROOT_FOLDER_ID, SharedWriteAuthorizationHarness,
    )

    provider = SharedWriteAuthorizationHarness(monkeypatch, sdk_fakes)
    monkeypatch.setenv("GOOGLE_REFRESH_TOKEN", "synthetic-refresh-token")
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "synthetic-client-id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "synthetic-client-secret")
    client_db.collection(FOLDERS_COLLECTION).document(ROOT_FOLDER_ID).set({
        "folder_id": ROOT_FOLDER_ID, "name": "TLBrain", "parent_id": None,
    })
    client_db.collection(FOLDERS_COLLECTION).document("presentation-archive").set({
        "folder_id": "presentation-archive", "name": "Archive",
        "parent_id": ROOT_FOLDER_ID,
    })
    provider.set_file(ROOT_FOLDER_ID, name="TLBrain",
                      mime_type="application/vnd.google-apps.folder", parents=[])
    provider.set_file("presentation-archive", name="Archive",
                      mime_type="application/vnd.google-apps.folder", parents=[ROOT_FOLDER_ID])
    for doc_id in ("doc-a", "doc-b", "doc-c"):
        client_db.collection(COLLECTION_NAME).document(doc_id).set({
            "doc_id": doc_id, "source_file": f"Synthetic {doc_id}",
            "title": f"Synthetic {doc_id}", "status": "synced",
            "parent_id": ROOT_FOLDER_ID, "dialog_date": "2026-04-01",
        })
        provider.set_file(doc_id, name=f"Synthetic {doc_id}",
                          mime_type="application/vnd.google-apps.document",
                          parents=[ROOT_FOLDER_ID])
        provider.add_qdrant_parent(doc_id, ROOT_FOLDER_ID, ROOT_FOLDER_ID)
    return provider


def _publish_fixture_answer(
    client_db, fixture, bot, *, answer_turn_id, body, list_blocks,
    kind="final", pointer=True, persist_manifest=True, now=None,
):
    """Persist a representative manifest before real send; select only after ACK."""
    from services.connectors.telegram.notifications import TelegramNotificationPublisher
    from services.connectors.telegram.runtime.runtime import _DurableCommandReplies

    repository = fixture["repository"]
    accepted = fixture["accepted"]
    lease = fixture["lease"]
    now = fixture["now"] if now is None else now
    repository.append_turn(
        accepted.context_id, fixture["conversation_id"], turn_id=answer_turn_id,
        operation_id=accepted.operation_id, role="model", parts=[{"text": body}],
        source_refs=[_source_evidence(fixture, row["doc_id"])
                     for row in fixture["result"]["segments"]], now=now,
        queue_operation_id=accepted.operation_id, queue_owner=lease.owner, queue_fence=lease.fence,
    )
    replies = _DurableCommandReplies(
        repository=repository,
        publisher=TelegramNotificationPublisher(repository, bot, clock=lambda: now),
        context_id=accepted.context_id, operation_id=accepted.operation_id,
        owner=lease.owner, fence=lease.fence, clock=lambda: now,
    )
    chunks = [body[index:index + 4000] for index in range(0, len(body), 4000)] or [""]
    notification_refs = []
    for index, _chunk in enumerate(chunks):
        chunk_kind = kind if len(chunks) == 1 or index == len(chunks) - 1 else "progress"
        notification_refs.append({
            "notification_id": replies.notification_id(chunk_kind, index), "revision": 1,
        })
    presentation_id = "presentation-" + answer_turn_id
    presentation = {
        "schema_version": 1, "presentation_id": presentation_id,
        "context_id": accepted.context_id, "conversation_id": fixture["conversation_id"],
        "producing_operation_id": accepted.operation_id,
        "answer_turn_id": answer_turn_id, "body": body,
        "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "blocks": list_blocks,
        "sources": [{"call_id": fixture["call"]["id"], "ordinal": 0,
                     "checkpoint_conversation_id": fixture["checkpoint_id"],
                     "operation_id": accepted.operation_id,
                     "reference": fixture["reference"]}],
        "notification_refs": notification_refs,
        "source_refs": [_source_evidence(fixture, row["doc_id"])
                        for row in fixture["result"]["segments"]],
    }
    conversation_ref = (client_db.collection("telegram_sessions").document(accepted.context_id)
                        .collection("conversations").document(fixture["conversation_id"]))
    manifest_ref = conversation_ref.collection("presentations").document(presentation_id)
    if persist_manifest:
        manifest_ref.set(presentation)
    pre_send_observations = []

    def inspect_before_send(payload, sent):
        notification_id = notification_refs[len(pre_send_observations)]["notification_id"]
        notification = fixture["repository"].get_notification(notification_id)
        if persist_manifest:
            stored = manifest_ref.get().to_dict()
            assert stored["body"] == body
            assert stored["body_sha256"] == hashlib.sha256(body.encode("utf-8")).hexdigest()
        assert notification["body"] == sent["text"]
        assert notification["revision"] == notification_refs[len(pre_send_observations)]["revision"]
        assert notification["state"] in {"sending", "pending"}
        assert notification.get("message_id") is None
        pre_send_observations.append((notification_id, sent["text"]))

    bot._before_send = inspect_before_send
    sent_offset = len(bot._test_sent)
    outcome = asyncio.run(replies.deliver(body, kind=kind, presentation="rich_markdown"))
    assert len(pre_send_observations) == len(notification_refs)
    notification_rows = [repository.get_notification(row["notification_id"])
                         for row in notification_refs]
    if outcome == "delivered":
        assert all(row["state"] == "delivered" and type(row.get("message_id")) is int
                   for row in notification_rows)
        assert all(row["message_id"] == 91001 + sent_offset + index
                   for index, row in enumerate(notification_rows))
        assert all(row["revision"] == ref["revision"]
                   for row, ref in zip(notification_rows, notification_refs))
        if persist_manifest and pointer and len(list_blocks) == 1:
            conversation_ref.update({"last_presented_list_id": list_blocks[0]["list_id"],
                                     "last_presentation_id": presentation_id})
        elif persist_manifest and len(list_blocks) > 1:
            conversation_ref.update({"last_presented_list_id": None,
                                     "last_presentation_id": presentation_id})
    else:
        assert not pointer
        assert not any(row.get("state") == "delivered" for row in notification_rows[1:])
    return {"presentation": presentation, "presentation_id": presentation_id,
            "manifest_ref": manifest_ref, "conversation_ref": conversation_ref,
            "notification_refs": notification_refs, "notifications": notification_rows,
            "outcome": outcome, "replies": replies}


def test_search_agent_presents_native_ordered_list_and_bound_quote_through_runtime(
    firestore_fake, sdk_fakes, monkeypatch,
):
    """A terminal native presentation call must become the exact durable user body."""
    from core.integrations.gemini.agent import GeminiAgent
    from services.connectors.telegram.commands import parse_command
    from services.connectors.telegram.runtime import TelegramCommandRuntime
    from tests.fixtures.sdk_fakes import FakeGeminiClient

    client_db = firestore_fake.client_class()
    _script_source_retrieval(sdk_fakes, monkeypatch, client_db)
    repository, accepted, operation, lease, now = _running_command(client_db)
    bot = _bot_client()
    runtime = TelegramCommandRuntime(
        repository=repository, save_handler=object(), client=bot,
        model=GeminiAgent(client=FakeGeminiClient(api_key="synthetic-presentation-key")),
        clock=lambda: now,
    )
    blocks = [
        {"type": "paragraph", "text": "Verified summary.", "source_refs": [
            {"doc_id": "doc-a", "order_index": 1},
        ]},
        {"type": "document_list", "list_id": "list-visible-1", "entries": [
            {"doc_id": "doc-a", "label": "Alpha", "source_refs": [{"doc_id": "doc-a", "order_index": 1}]},
            {"doc_id": "doc-b", "label": "Beta", "source_refs": [{"doc_id": "doc-b", "order_index": 2}]},
            {"doc_id": "doc-c", "label": "Gamma", "source_refs": [{"doc_id": "doc-c", "order_index": 3}]},
        ]},
        {"type": "quote", "text": "The exact verified sentence.", "doc_id": "doc-b",
         "source_refs": [{"doc_id": "doc-b", "order_index": 2}]},
    ]
    expected_body = "Verified summary.\n\n1. Alpha\n2. Beta\n3. Gamma\n\n> The exact verified sentence."

    # route_command is a valid ordinary new-search route.  The second native
    # response is the proposed terminal artifact from the actual tool loop.
    sdk_fakes.gemini.queue("generate_content", response=_native_response("route_command", {
        "kind": "search", "target_ref": None, "target_operation_id": None,
        "dialog_number": None, "result_refs": [],
        "substantive_text": "Find three project discussions",
        "tool_choice": {"name": "query", "arguments": {"query": "project discussions"}},
        "certainty": "certain",
    }))
    sdk_fakes.gemini.queue("generate_content", response=_native_response("present_search_answer", {
        "blocks": blocks,
    }))

    producer_send_snapshots = []

    def inspect_producer_send(_payload, sent):
        """Record pre-ACK producer state without changing the transport outcome."""
        snapshot = {"sent_body": sent.get("text"), "exception": None}
        try:
            context_state = repository.get_context(accepted.context_id) or {}
            active_id = context_state.get("active_conversation_id")
            if isinstance(active_id, str):
                producer_conversation = (client_db.collection("telegram_sessions")
                                         .document(accepted.context_id)
                                         .collection("conversations").document(active_id))
                snapshot["conversation_id"] = active_id
                snapshot["turns"] = [row.to_dict() for row in producer_conversation.collection(
                    "turns").stream()]
                snapshot["presentations"] = [row.to_dict() for row in producer_conversation.collection(
                    "presentations").stream()]
                if snapshot["presentations"]:
                    header = snapshot["presentations"][0]
                    manifest = repository.get_presentation_manifest(
                        accepted.context_id, active_id, header["presentation_id"],
                    )
                    snapshot["manifest"] = manifest
                    first_ref = manifest["notification_refs"][0]
                    snapshot["notification_id"] = first_ref["notification_id"]
                    snapshot["notification"] = repository.get_notification(
                        first_ref["notification_id"],
                    )
            operation_ref = (client_db.collection("telegram_sessions").document(accepted.context_id)
                             .collection("operations").document(accepted.operation_id))
            operation_data = operation_ref.get().to_dict() or {}
            notification_id = operation_data.get("notification_id")
            if isinstance(notification_id, str):
                snapshot["notification_id"] = notification_id
                snapshot["notification"] = repository.get_notification(notification_id)
        except Exception as error:
            snapshot["exception"] = type(error).__name__
        producer_send_snapshots.append(snapshot)

    bot._before_send = inspect_producer_send

    try:
        with _capture_gemini(sdk_fakes) as requests:
            asyncio.run(runtime(
                accepted.context_id, operation, lease,
                command=parse_command("Find the three project discussions"),
            ))
        context_state = repository.get_context(accepted.context_id)
        conversation_id = context_state["active_conversation_id"]
        conversation_ref = (client_db.collection("telegram_sessions").document(accepted.context_id)
                            .collection("conversations").document(conversation_id))
        tool_turns = [snapshot.to_dict() for snapshot in conversation_ref.collection("turns").stream()
                      if (snapshot.to_dict() or {}).get("role") == "tool"]
        exchange_parts = [part for turn in tool_turns for part in turn.get("parts", [])
                          if part.get("type") == "search_exchange_ref"]
        assert exchange_parts, "the search answer must be grounded in a saved native query result"
        from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger
        ledger = FirestoreAgentLedger(
            client_db, context_id=accepted.context_id, operation_id=accepted.operation_id,
            owner=lease.owner, fence=lease.fence, clock=lambda: now,
        )
        actual_exchange = ledger.load_search_exchange(
            exchange_parts[-1]["reference"], conversation_id=conversation_id,
            operation_id=accepted.operation_id,
            checkpoint_conversation_id=exchange_parts[-1]["checkpoint_conversation_id"],
            call_id=exchange_parts[-1]["call_id"], ordinal=exchange_parts[-1]["ordinal"],
        )
        assert [row["doc_id"] for row in actual_exchange["result"]["segments"]] == [
            "doc-c", "doc-a", "doc-b",
        ]
        actual_quote_text = [line["text"]
                             for document in actual_exchange["result"]["segments"]
                             if document["doc_id"] == "doc-b"
                             for segment in document.get("segments", [])
                             for line in segment.get("dialog", [])]
        assert "The exact verified sentence." in actual_quote_text
        tool_loop = [row for row in requests if row.get("boundary") == "gemini"
                     and row.get("operation") == "generate_content"
                     and _thinking_level(row) == "MEDIUM"]
        assert tool_loop, "the ordinary search agent must receive the native terminal response"
        declarations = _native_function_names(tool_loop[-1])
        assert "present_search_answer" in declarations, "search agent must declare its terminal presentation tool"
        producer_snapshot = next(row for row in producer_send_snapshots
                                 if row["sent_body"] == expected_body)
        assert producer_snapshot["exception"] is None
        assert any(turn.get("role") == "model"
                   and any(part.get("text") == expected_body for part in turn.get("parts", []))
                   for turn in producer_snapshot["turns"])
        pre_manifest, = producer_snapshot["presentations"]
        assert "body" not in pre_manifest
        assert pre_manifest["body_ref"]["body_sha256"] == hashlib.sha256(
            expected_body.encode("utf-8"),
        ).hexdigest()
        pre_notification = producer_snapshot["notification"]
        assert pre_notification["body"] == expected_body
        manifest = producer_snapshot["manifest"]
        assert manifest["body"] == expected_body
        assert pre_notification["revision"] == manifest["notification_refs"][0]["revision"]
        assert pre_notification["state"] in {"pending", "sending"}
        assert pre_notification.get("message_id") is None
        assert [row.get("text") for row in bot._test_sent if row.get("text") is not None] == [expected_body]
        presentation_rows = list(conversation_ref.collection("presentations").stream())
        assert len(presentation_rows) == 1
        header = presentation_rows[0].to_dict()
        assert "body" not in header
        assert header["body_ref"]["body_sha256"] == hashlib.sha256(
            expected_body.encode("utf-8"),
        ).hexdigest()
        assert manifest["body_sha256"] == hashlib.sha256(expected_body.encode("utf-8")).hexdigest()
        entries = manifest["blocks"][1]["entries"]
        assert [(row["position"], row["doc_id"], row["label"])
                for row in entries] == [(1, "doc-a", "Alpha"),
                                       (2, "doc-b", "Beta"),
                                       (3, "doc-c", "Gamma")]
        quote = manifest["blocks"][2]
        assert quote["text"] == "The exact verified sentence."
        quote_refs = quote.get("evidence_refs", [])
        assert quote_refs and all(ref.get("doc_id") == "doc-b" for ref in quote_refs)
        source_by_id = {source["source_id"]: source for source in manifest["sources"]}
        quote_sources = [source_by_id[ref["source_id"]] for ref in quote_refs]
        assert any(source.get("call_id") == actual_exchange["call"]["id"]
                   for source in quote_sources)
        assert all(source["reference"]["payload"]["sha256"]
                   == exchange_parts[-1]["reference"]["payload"]["sha256"]
                   for source in quote_sources)
        notification_refs = manifest["notification_refs"]
        assert notification_refs
        for ref in notification_refs:
            notification = repository.get_notification(ref["notification_id"])
            assert notification["state"] == "delivered"
            assert notification["revision"] == ref["revision"]
            assert type(notification["message_id"]) is int
            assert notification["message_id"] == 91001 + bot._test_sent.index(
                next(row for row in bot._test_sent if row.get("text") == expected_body)
            )
    finally:
        asyncio.run(bot._test_http_client.aclose())


@pytest.mark.parametrize("artifact", ["duplicate", "quote", "mixed"])
def test_invalid_native_terminal_must_not_publish_duplicate_or_unbound_quote(
    firestore_fake, sdk_fakes, monkeypatch, artifact,
):
    """Invalid native artifacts and mixed read/terminal rounds publish nothing."""
    from core.integrations.gemini.agent import GeminiAgent
    from services.connectors.telegram.commands import parse_command
    from services.connectors.telegram.runtime import TelegramCommandRuntime
    from tests.fixtures.sdk_fakes import FakeGeminiClient

    client_db = firestore_fake.client_class()
    _script_source_retrieval(sdk_fakes, monkeypatch, client_db)
    repository, accepted, operation, lease, now = _running_command(client_db, text="Find discussions")
    bot = _bot_client()
    runtime = TelegramCommandRuntime(
        repository=repository, save_handler=object(), client=bot,
        model=GeminiAgent(client=FakeGeminiClient(api_key="synthetic-presentation-key")),
        clock=lambda: now,
    )
    sdk_fakes.gemini.queue("generate_content", response=_native_response("route_command", {
        "kind": "search", "target_ref": None, "target_operation_id": None,
        "dialog_number": None, "result_refs": [], "substantive_text": "Find discussions",
        "tool_choice": {"name": "query", "arguments": {"query": "discussions"}},
        "certainty": "certain",
    }))
    duplicate_blocks = [{"type": "document_list", "list_id": "same-list", "entries": [
        {"doc_id": "doc-a", "label": "Alpha", "source_refs": [{"doc_id": "doc-a", "order_index": 1}]},
        {"doc_id": "doc-a", "label": "Duplicate", "source_refs": [{"doc_id": "doc-a", "order_index": 1}]},
    ]}]
    quote_blocks = [
        {"type": "document_list", "list_id": "valid-list", "entries": [
            {"doc_id": "doc-a", "label": "Alpha", "source_refs": [{"doc_id": "doc-a", "order_index": 1}]},
            {"doc_id": "doc-b", "label": "Beta", "source_refs": [{"doc_id": "doc-b", "order_index": 2}]},
        ]},
        {"type": "quote", "text": "A sentence that is not in the verified source.",
         "doc_id": "doc-b", "source_refs": [{"doc_id": "doc-b", "order_index": 2}]},
    ]
    if artifact == "mixed":
        from google.genai import types
        response = types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(parts=[
            types.Part(function_call=types.FunctionCall(
                name="present_search_answer", args={"blocks": [{
                    "type": "document_list", "list_id": "valid-mixed-list", "entries": [
                        {"doc_id": "doc-a", "label": "Alpha",
                         "source_refs": [{"doc_id": "doc-a", "order_index": 1}]},
                        {"doc_id": "doc-b", "label": "Beta",
                         "source_refs": [{"doc_id": "doc-b", "order_index": 2}]},
                    ],
                }]},
            )),
            types.Part(function_call=types.FunctionCall(
                name="query", args={"query": "same round extra read"},
            )),
        ]))])
    else:
        response = _native_response("present_search_answer", {
            "blocks": duplicate_blocks if artifact == "duplicate" else quote_blocks,
        })
    sdk_fakes.gemini.queue("generate_content", response=response)

    try:
        with _capture_gemini(sdk_fakes) as requests:
            asyncio.run(runtime(
                accepted.context_id, operation, lease,
                command=parse_command("Find the three project discussions"),
            ))
        terminal_requests = [row for row in requests if _thinking_level(row) == "MEDIUM"]
        assert terminal_requests
        presentation_collection = client_db.collection("telegram_sessions").document(accepted.context_id).collection(
            "conversations").document(
                repository.get_context(accepted.context_id)["active_conversation_id"]
            ).collection("presentations")
        assert not list(presentation_collection.stream())
        if artifact == "mixed":
            qdrant_calls = [row for row in requests if row.get("boundary") == "qdrant"
                            and row.get("operation") == "query_points"]
            assert len(qdrant_calls) == 3, "mixed terminal+read round must not run its read call"
        assert "present_search_answer" in _native_function_names(terminal_requests[-1]), (
            "invalid-artifact safety needs an actual presentation capability canary"
        )
        assert not any("1. Alpha" in row.get("text", "") for row in bot._test_sent)
    finally:
        asyncio.run(bot._test_http_client.aclose())


def test_legacy_numbered_prose_is_delivered_without_creating_a_manifest(
    firestore_fake,
):
    """Existing text delivery remains readable but has no positional authority record."""
    from services.connectors.telegram.notifications import TelegramNotificationPublisher
    from services.connectors.telegram.commands import parse_command
    from services.connectors.telegram.runtime.runtime import _DurableCommandReplies

    client_db = firestore_fake.client_class()
    repository, accepted, _operation, lease, now = _running_command(
        client_db,
        text="Find project discussions",
    )
    bot = _bot_client()
    publisher = TelegramNotificationPublisher(repository, bot, clock=lambda: now)
    replies = _DurableCommandReplies(
        repository=repository, publisher=publisher,
        context_id=accepted.context_id, operation_id=accepted.operation_id,
        owner=lease.owner, fence=lease.fence, clock=lambda: now,
    )
    body = "1. Alpha\n2. Beta\n3. Gamma"
    try:
        outcome = asyncio.run(replies.deliver(body, presentation="rich_markdown"))
        assert outcome == "delivered"
        assert [row.get("text") for row in bot._test_sent if row.get("text") is not None] == [body]
        operation_row = (client_db.collection("telegram_sessions").document(accepted.context_id)
                         .collection("operations").document(accepted.operation_id).get().to_dict())
        notification_id = operation_row["notification_id"]
        notification = repository.get_notification(notification_id)
        assert notification["state"] == "delivered"
        assert notification["body"] == body
        assert notification["message_id"] == 91001
        conversation_rows = (client_db.collection("telegram_sessions").document(accepted.context_id)
                             .collection("conversations").stream())
        assert list(conversation_rows) == []
    finally:
        asyncio.run(bot._test_http_client.aclose())


def test_partial_multichunk_list_delivery_never_publishes_a_complete_list(
    firestore_fake,
):
    """A delivered prefix stays visible, while the failed later chunk has no list authority."""
    from services.connectors.telegram.notifications import TelegramNotificationPublisher
    from services.connectors.telegram.runtime.runtime import _DurableCommandReplies

    client_db = firestore_fake.client_class()
    repository, accepted, _operation, lease, now = _running_command(
        client_db, text="Find project discussions",
    )
    bot = _bot_client(fail_send_number=2)
    publisher = TelegramNotificationPublisher(repository, bot, clock=lambda: now)
    replies = _DurableCommandReplies(
        repository=repository, publisher=publisher, context_id=accepted.context_id,
        operation_id=accepted.operation_id, owner=lease.owner, fence=lease.fence,
        clock=lambda: now,
    )
    body = "1. Alpha\n2. Beta\n" + ("x" * 3983) + "\n3. Gamma"
    try:
        outcome = asyncio.run(replies.deliver(body, presentation="rich_markdown"))
        assert outcome == "terminal"
        actual_chunks = [row["text"] for row in bot._test_sent]
        assert len(actual_chunks) == 2
        assert actual_chunks[0] == body[:4000]
        assert actual_chunks[1] == body[4000:]
        assert "3. Gamma" not in actual_chunks[0] and "3. Gamma" in actual_chunks[1]
        first_id = replies.notification_id("progress", 0)
        second_id = replies.notification_id("final", 1)
        first = repository.get_notification(first_id)
        second = repository.get_notification(second_id)
        assert first["state"] == "delivered" and first["message_id"] == 91001
        assert second["state"] == "failed"
        conversations = list(client_db.collection("telegram_sessions").document(
            accepted.context_id).collection("conversations").stream())
        assert conversations == []
    finally:
        asyncio.run(bot._test_http_client.aclose())


def test_partial_trusted_search_list_cannot_authorize_visible_selector(
    firestore_fake, sdk_fakes, monkeypatch,
):
    """A real delivered prefix plus failed tail never selects the seeded list."""
    from core.integrations.firestore.operation_queue import claim_head, fail_head
    from core.integrations.gemini.agent import GeminiAgent
    from services.connectors.telegram.commands import parse_command
    from services.connectors.telegram.runtime import TelegramCommandRuntime
    from services.connectors.telegram.runtime.saving import TelegramSaveRuntime
    from tests.fixtures.sdk_fakes import FakeGeminiClient
    from tests.fixtures.shared_write_authorization import ROOT_FOLDER_ID

    client_db = firestore_fake.client_class()
    fixture = _source_search_fixture(client_db)
    provider = _configure_writer_boundary(client_db, monkeypatch, sdk_fakes)
    list_block = _list_block(
        fixture, list_id="partial-list", doc_ids=["doc-a", "doc-b", "doc-c"],
        labels={"doc-a": "Alpha", "doc-b": "B" * 4000, "doc-c": "Gamma"},
    )
    body = _render_fixture_blocks([list_block])
    assert len(body) > 4000 and body == _render_fixture_blocks([list_block])
    bot = _bot_client(fail_send_number=2)
    try:
        publication = _publish_fixture_answer(
            client_db, fixture, bot, answer_turn_id="partial-answer", body=body,
            list_blocks=[list_block],
            pointer=False,
        )
        assert publication["outcome"] == "terminal"
        assert [row["state"] for row in publication["notifications"]] == ["delivered", "failed"]
        assert publication["notifications"][0]["message_id"] == 91001
        assert publication["notifications"][1].get("message_id") is None
        conversation = publication["conversation_ref"].get().to_dict()
        assert conversation.get("last_presented_list_id") is None
        assert conversation.get("last_presentation_id") is None
        assert publication["presentation"]["blocks"][0]["list_id"] == "partial-list"

        prior, accepted, operation, lease = (
            fixture["repository"], fixture["accepted"], fixture["operation"], fixture["lease"]
        )
        context_ref = client_db.collection("telegram_sessions").document(accepted.context_id)
        assert fail_head(
            client_db, context_ref, operation_id=accepted.operation_id,
            owner=lease.owner, fence=lease.fence, now=fixture["now"] + timedelta(seconds=1),
        )
        current = prior.accept_command(
            88102, chat_id=42017, thread_id=73, user_id=17,
            command=parse_command("Move the second search result to Archive"),
            now=fixture["now"] + timedelta(seconds=2),
        )
        current_lease = claim_head(
            client_db, context_ref, owner="partial-selector-worker",
            now=fixture["now"] + timedelta(seconds=2),
        )
        assert current_lease is not None
        current_operation = context_ref.collection("operations").document(current.operation_id).get().to_dict()
        bot._before_send = None
        sdk_fakes.gemini.queue("generate_content", response=_native_response("route_command", {
            "kind": "substantive", "target_ref": "doc-b", "target_operation_id": None,
            "dialog_number": None, "result_refs": [],
            "substantive_text": "Move the second search result to Archive",
            "tool_choice": {"name": "move_transcript", "arguments": {
                "doc_id": "doc-b", "folder_path": ["Archive"],
            }}, "certainty": "certain",
        }))
        from google.genai import types
        sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(parts=[types.Part(
                text="I need you to choose a confirmed list before moving a document.",
            )]))],
        ))
        runtime = TelegramCommandRuntime(
            repository=prior, save_handler=TelegramSaveRuntime(repository=prior), client=bot,
            model=GeminiAgent(client=FakeGeminiClient(api_key="synthetic-presentation-key")),
            clock=lambda: fixture["now"] + timedelta(seconds=2), root_folder_id=ROOT_FOLDER_ID,
        )
        result = asyncio.run(runtime(
            current.context_id, current_operation, current_lease,
            command=parse_command("Move the second search result to Archive"),
        ))
        assert not provider.drive_mutations
        assert not provider.qdrant_mutations
        assert all(provider.files[doc_id]["parents"] == [ROOT_FOLDER_ID]
                   for doc_id in ("doc-a", "doc-b", "doc-c"))
        _assert_prepared_question(result)
    finally:
        asyncio.run(bot._test_http_client.aclose())


def test_unqualified_second_is_ambiguous_across_two_lists_in_one_answer(
    firestore_fake, sdk_fakes, monkeypatch,
):
    """Two separately delivered lists in one answer require a list-qualified target."""
    from core.integrations.firestore.operation_queue import claim_head, complete_head
    from core.integrations.gemini.agent import GeminiAgent
    from services.connectors.telegram.commands import parse_command
    from services.connectors.telegram.runtime import TelegramCommandRuntime
    from services.connectors.telegram.runtime.saving import TelegramSaveRuntime
    from tests.fixtures.sdk_fakes import FakeGeminiClient
    from tests.fixtures.shared_write_authorization import ROOT_FOLDER_ID

    client_db = firestore_fake.client_class()
    fixture = _source_search_fixture(client_db)
    provider = _configure_writer_boundary(client_db, monkeypatch, sdk_fakes)
    bot = _bot_client()
    body = "1. Alpha\n2. Beta\n\n1. Gamma\n2. Alpha"
    lists = [
        _list_block(fixture, list_id="one-answer-list-1", doc_ids=["doc-a", "doc-b"]),
        _list_block(fixture, list_id="one-answer-list-2", doc_ids=["doc-c", "doc-a"]),
    ]
    try:
        published = _publish_fixture_answer(
            client_db, fixture, bot, answer_turn_id="two-lists-one-answer",
            body=body, list_blocks=lists, pointer=True,
        )
        assert published["outcome"] == "delivered"
        assert len(published["notifications"]) == 1
        assert published["notifications"][0]["state"] == "delivered"
        assert published["presentation"]["body"] == body
        assert published["conversation_ref"].get().to_dict().get("last_presented_list_id") is None
        assert published["conversation_ref"].get().to_dict()["last_presentation_id"] == published[
            "presentation_id"
        ]
        producer_context_ref = client_db.collection("telegram_sessions").document(
            fixture["accepted"].context_id,
        )
        assert complete_head(
            client_db, producer_context_ref,
            operation_id=fixture["accepted"].operation_id, owner=fixture["lease"].owner,
            fence=fixture["lease"].fence, now=fixture["now"] + timedelta(seconds=1),
        )
        producer_state = producer_context_ref.collection("operations").document(
            fixture["accepted"].operation_id).get().to_dict()["state"]
        assert producer_state == "completed"
        repository = fixture["repository"]
        current = repository.accept_command(
            88102, chat_id=42017, thread_id=73, user_id=17,
            command=parse_command("Move the second search result to Archive"),
            now=fixture["now"] + timedelta(seconds=2),
        )
        context_ref = client_db.collection("telegram_sessions").document(current.context_id)
        lease = claim_head(client_db, context_ref, owner="multiple-list-selector",
                           now=fixture["now"] + timedelta(seconds=2))
        assert lease is not None
        operation = context_ref.collection("operations").document(current.operation_id).get().to_dict()
        bot._before_send = None
        sdk_fakes.gemini.queue("generate_content", response=_native_response("route_command", {
            "kind": "substantive", "target_ref": "doc-b", "target_operation_id": None,
            "dialog_number": None, "result_refs": [],
            "substantive_text": "Move the second search result to Archive",
            "tool_choice": {"name": "move_transcript", "arguments": {
                "doc_id": "doc-b", "folder_path": ["Archive"],
            }}, "certainty": "certain",
        }))
        from google.genai import types
        sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
            candidates=[types.Candidate(content=types.Content(parts=[types.Part(
                text="Which displayed list should I use for the second position?",
            )]))],
        ))
        runtime = TelegramCommandRuntime(
            repository=repository, save_handler=TelegramSaveRuntime(repository=repository), client=bot,
            model=GeminiAgent(client=FakeGeminiClient(api_key="synthetic-presentation-key")),
            clock=lambda: fixture["now"] + timedelta(seconds=2), root_folder_id=ROOT_FOLDER_ID,
        )
        result = asyncio.run(runtime(
            current.context_id, operation, lease,
            command=parse_command("Move the second search result to Archive"),
        ))
        assert not provider.drive_mutations and not provider.qdrant_mutations
        assert all(provider.files[doc_id]["parents"] == [ROOT_FOLDER_ID]
                   for doc_id in ("doc-a", "doc-b", "doc-c"))
        _assert_prepared_question(result)
    finally:
        asyncio.run(bot._test_http_client.aclose())


def test_explicit_document_move_remains_a_real_writer_canary(firestore_fake, sdk_fakes, monkeypatch):
    """The contextual suite's effects boundary is live enough to detect writes."""
    from google.genai import types
    from core.integrations.firestore.client import COLLECTION_NAME, FOLDERS_COLLECTION
    from core.integrations.gemini.agent import GeminiAgent
    from services.connectors.telegram.commands import parse_command
    from services.connectors.telegram.runtime import TelegramCommandRuntime
    from tests.fixtures.sdk_fakes import FakeGeminiClient
    from tests.fixtures.shared_write_authorization import (
        ROOT_FOLDER_ID, SharedWriteAuthorizationHarness,
    )

    client_db = firestore_fake.client_class()
    repository, accepted, operation, lease, now = _running_command(
        client_db,
        text="Move document doc-b to Archive",
    )
    provider = SharedWriteAuthorizationHarness(monkeypatch, sdk_fakes)
    monkeypatch.setenv("GOOGLE_REFRESH_TOKEN", "synthetic-refresh-token")
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "synthetic-client-id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "synthetic-client-secret")
    client_db.collection(FOLDERS_COLLECTION).document(ROOT_FOLDER_ID).set({
        "folder_id": ROOT_FOLDER_ID, "name": "TLBrain", "parent_id": None,
    })
    client_db.collection(FOLDERS_COLLECTION).document("presentation-archive").set({
        "folder_id": "presentation-archive", "name": "Archive",
        "parent_id": ROOT_FOLDER_ID,
    })
    provider.set_file(ROOT_FOLDER_ID, name="TLBrain",
                      mime_type="application/vnd.google-apps.folder", parents=[])
    provider.set_file("presentation-archive", name="Archive",
                      mime_type="application/vnd.google-apps.folder", parents=[ROOT_FOLDER_ID])
    for doc_id in ("doc-a", "doc-b", "doc-c"):
        client_db.collection(COLLECTION_NAME).document(doc_id).set({
            "doc_id": doc_id, "source_file": f"Synthetic {doc_id}",
            "title": f"Synthetic {doc_id}", "status": "synced",
            "parent_id": ROOT_FOLDER_ID, "dialog_date": "2026-04-01",
        })
        provider.set_file(doc_id, name=f"Synthetic {doc_id}",
                          mime_type="application/vnd.google-apps.document",
                          parents=[ROOT_FOLDER_ID])
        provider.add_qdrant_parent(doc_id, ROOT_FOLDER_ID, ROOT_FOLDER_ID)

    route = {
        "kind": "substantive", "target_ref": "doc-b",
        "target_operation_id": None, "dialog_number": None, "result_refs": [],
        "substantive_text": "Move document doc-b to Archive",
        "tool_choice": {"name": "move_transcript", "arguments": {
            "doc_id": "doc-b", "folder_path": ["Archive"],
        }},
        "certainty": "certain",
    }
    sdk_fakes.gemini.queue("generate_content", response=_native_response("route_command", route))
    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[types.Part(text="Moved doc-b to Archive.")]))],
    ))
    bot = _bot_client()
    runtime = TelegramCommandRuntime(
        repository=repository, save_handler=object(), client=bot,
        model=GeminiAgent(client=FakeGeminiClient(api_key="synthetic-presentation-key")),
        clock=lambda: now, root_folder_id=ROOT_FOLDER_ID,
    )
    try:
        result = asyncio.run(runtime(
            accepted.context_id, operation, lease,
            command=parse_command("Move document doc-b to Archive"),
        ))
        assert result.value == "completed"
        assert len(provider.drive_mutations) == 1
        assert provider.drive_mutations[0]["fileId"] == "doc-b"
        assert provider.files["doc-b"]["parents"] == ["presentation-archive"]
        assert client_db.collection(COLLECTION_NAME).document("doc-b").get().to_dict()[
            "parent_id"
        ] == "presentation-archive"
        assert all(provider.files[doc_id]["parents"] == [ROOT_FOLDER_ID]
                   for doc_id in ("doc-a", "doc-c"))
    finally:
        asyncio.run(bot._test_http_client.aclose())


def _prepare_selector_fixture(client_db, sdk_fakes, monkeypatch, *, has_newer_list, has_manifest):
    """Create genuine search evidence and send before seeding any selectable manifest."""
    from core.integrations.firestore.operation_queue import complete_head
    from tests.fixtures.shared_write_authorization import ROOT_FOLDER_ID

    fixture = _source_search_fixture(client_db)
    provider = _configure_writer_boundary(client_db, monkeypatch, sdk_fakes)
    repository = fixture["repository"]
    now = fixture["now"]
    bot = _bot_client()
    old_block = _list_block(fixture, list_id="list-visible-1",
                            doc_ids=["doc-a", "doc-b", "doc-c"])
    old_body = _render_fixture_blocks([old_block])
    old_publication = _publish_fixture_answer(
        client_db, fixture, bot, answer_turn_id="presentation-prior-answer",
        body=old_body, list_blocks=[old_block], pointer=has_manifest,
        persist_manifest=has_manifest,
    )
    assert old_publication["outcome"] == "delivered"
    if has_manifest:
        assert old_publication["notifications"][0]["state"] == "delivered"
        assert old_publication["conversation_ref"].get().to_dict()[
            "last_presented_list_id"] == "list-visible-1"

    if has_newer_list:
        newer_block = _list_block(fixture, list_id="list-visible-2",
                                  doc_ids=["doc-c", "doc-a", "doc-b"])
        newer_body = _render_fixture_blocks([newer_block])
        newer = _publish_fixture_answer(
            client_db, fixture, bot, answer_turn_id="presentation-newer-answer",
            body=newer_body, list_blocks=[newer_block], kind="progress", now=now + timedelta(
                milliseconds=1),
        )
        assert newer["outcome"] == "delivered"
        assert newer["notifications"][0]["state"] == "delivered"
        assert newer["presentation"]["presentation_id"] != old_publication[
            "presentation_id"]
        assert newer["conversation_ref"].get().to_dict()[
            "last_presented_list_id"] == "list-visible-2"

    for doc_id, number in (("doc-a", 1), ("doc-c", 2)):
        conversation_id = f"bound-{doc_id}"
        repository.bind_document(
            fixture["accepted"].context_id, doc_id, conversation_id,
            metadata={"title": f"Synthetic {doc_id}", "parent_id": ROOT_FOLDER_ID},
            now=now, update_last_doc=False,
        )
        repository.ensure_conversation(
            fixture["accepted"].context_id, conversation_id, kind="document", doc_id=doc_id,
            dialog_number=number, now=now,
        )
    assert complete_head(
        client_db, client_db.collection("telegram_sessions").document(
            fixture["accepted"].context_id),
        operation_id=fixture["accepted"].operation_id,
        owner=fixture["lease"].owner, fence=fixture["lease"].fence,
        now=now + timedelta(seconds=1),
    )
    asyncio.run(bot._test_http_client.aclose())
    return {"fixture": fixture, "provider": provider,
            "presentation": old_publication["presentation"],
            "old_body": old_body}


@pytest.mark.parametrize(
    ("command_text", "model_target", "dialog_number", "expected_move", "has_newer_list", "has_manifest"),
    [
        ("Move the first search result to Archive", "doc-a", None, "doc-a", False, True),
        ("Move the last search result to Archive", "doc-c", None, "doc-c", False, True),
        ("Move the second search result to Archive", "doc-b", None, "doc-b", False, True),
        ("Move the second search result to Archive", "doc-a", None, None, False, True),
        ("Move dialog #2 to Archive", "doc-c", 2, "doc-c", False, True),
        ("Move dialog #2 (https://docs.google.com/document/d/doc-a/edit) to Archive",
         "doc-a", 2, None, False, True),
        ("Move the second result from the older list to Archive", "doc-b", None, "doc-b", True, True),
        ("Move the second result from the older list to Archive", "doc-a", None, None, True, True),
        ("Move the second search result to Archive", "doc-b", None, None, True, True),
        ("Move the second search result to Archive", "doc-b", None, None, False, False),
    ],
)
def test_search_selectors_use_confirmed_visible_positions_and_keep_local_numbers_independent(
    firestore_fake, sdk_fakes, monkeypatch,
    command_text, model_target, dialog_number, expected_move, has_newer_list, has_manifest,
):
    """Visible positions resolve from ACKed manifests, independently of retrieval and local numbers."""
    from google.genai import types
    from core.integrations.firestore.client import COLLECTION_NAME
    from core.integrations.firestore.operation_queue import claim_head
    from core.integrations.gemini.agent import GeminiAgent
    from services.connectors.telegram.commands import parse_command
    from services.connectors.telegram.runtime import TelegramCommandRuntime
    from services.connectors.telegram.runtime.saving import TelegramSaveRuntime
    from tests.fixtures.sdk_fakes import FakeGeminiClient
    from tests.fixtures.shared_write_authorization import ROOT_FOLDER_ID

    client_db = firestore_fake.client_class()
    seeded = _prepare_selector_fixture(
        client_db, sdk_fakes, monkeypatch,
        has_newer_list=has_newer_list, has_manifest=has_manifest,
    )
    fixture = seeded["fixture"]
    repository = fixture["repository"]
    provider = seeded["provider"]
    now = fixture["now"]
    current = repository.accept_command(
        88102, chat_id=42017, thread_id=73, user_id=17,
        command=parse_command(command_text), now=now + timedelta(seconds=2),
    )
    context_ref = client_db.collection("telegram_sessions").document(current.context_id)
    current_lease = claim_head(
        client_db, context_ref, owner="presentation-current-worker",
        now=now + timedelta(seconds=2),
    )
    assert current_lease is not None
    current_operation = context_ref.collection("operations").document(
        current.operation_id).get().to_dict()

    # For the explicit URL/#2 conflict, both valid signals remain in the route proposal.
    target_ref = model_target if "https://docs.google.com" in command_text else (
        None if dialog_number else model_target
    )
    proposed_route = {
        "kind": "substantive", "target_ref": target_ref,
        "target_operation_id": None,
        "dialog_number": None if "https://docs.google.com" in command_text else dialog_number,
        "result_refs": [], "substantive_text": command_text,
        "tool_choice": {"name": "move_transcript", "arguments": {
            "doc_id": model_target, "folder_path": ["Archive"],
        }}, "certainty": "certain",
    }
    sdk_fakes.gemini.queue("generate_content", response=_native_response(
        "route_command", proposed_route,
    ))
    sdk_fakes.gemini.queue("generate_content", response=types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(parts=[types.Part(text="Moved.")]))],
    ))
    bot = _bot_client()
    runtime = TelegramCommandRuntime(
        repository=repository, save_handler=TelegramSaveRuntime(repository=repository),
        client=bot,
        model=GeminiAgent(client=FakeGeminiClient(api_key="synthetic-presentation-key")),
        clock=lambda: now + timedelta(seconds=2), root_folder_id=ROOT_FOLDER_ID,
    )
    try:
        result = asyncio.run(runtime(
            current.context_id, current_operation, current_lease,
            command=parse_command(command_text),
        ))
        if expected_move is None:
            assert not provider.drive_mutations
            assert not provider.qdrant_mutations
            assert all(provider.files[doc_id]["parents"] == [ROOT_FOLDER_ID]
                       for doc_id in ("doc-a", "doc-b", "doc-c"))
            if not has_manifest or "https://docs.google.com" in command_text:
                _assert_prepared_question(result)
            else:
                _assert_terminal_refusal_or_clarification(result, bot._test_sent)
        else:
            assert result.value == "completed"
            assert {row.get("fileId") for row in provider.drive_mutations} == {expected_move}
            assert provider.files[expected_move]["parents"] == ["presentation-archive"]
            assert client_db.collection(COLLECTION_NAME).document(expected_move).get().to_dict()[
                "parent_id"
            ] == "presentation-archive"
            assert all(provider.files[doc_id]["parents"] ==
                       (["presentation-archive"] if doc_id == expected_move else [ROOT_FOLDER_ID])
                       for doc_id in ("doc-a", "doc-b", "doc-c"))
            assert all(parents == (["presentation-archive"] if doc_id == expected_move else [ROOT_FOLDER_ID])
                       for (doc_id, _root), parents in provider.qdrant_points.items())
        if has_manifest:
            assert seeded["presentation"]["blocks"][0]["entries"][1]["doc_id"] == "doc-b"
    finally:
        asyncio.run(bot._test_http_client.aclose())

class _capture_gemini:
    """Copy the real SDK trace generated while invoking an existing entry point."""

    def __init__(self, sdk_fakes):
        self.sdk_fakes = sdk_fakes
        self.rows = []

    def __enter__(self):
        self.offset = len(self.sdk_fakes.trace)
        return self.rows

    def __exit__(self, *_exc):
        self.rows.extend(copy.deepcopy(self.sdk_fakes.trace[self.offset:]))


def _script_source_retrieval(sdk_fakes, monkeypatch, client_db):
    """Run the real query path over source-bearing SDK responses in retrieval order."""
    from types import SimpleNamespace
    from google.genai import types
    from core.integrations.firestore.client import COLLECTION_NAME, FOLDERS_COLLECTION
    from tests.fixtures.shared_write_authorization import ROOT_FOLDER_ID

    monkeypatch.setenv("RETRIEVAL_TOP_K", "15")
    monkeypatch.setenv("RETRIEVAL_SCORE_THRESHOLD", "0.5")
    monkeypatch.setenv("QDRANT_URL", "https://synthetic.invalid")
    monkeypatch.setenv("QDRANT_API_KEY", "synthetic-qdrant-key")
    monkeypatch.setenv(
        "ROOT_FOLDER_URL", f"https://drive.google.com/drive/folders/{ROOT_FOLDER_ID}",
    )
    client_db.collection(FOLDERS_COLLECTION).document(ROOT_FOLDER_ID).set({
        "folder_id": ROOT_FOLDER_ID, "name": "TLBrain", "parent_id": None,
    })
    evidence = {
        "doc-c": (3, "Gamma source."),
        "doc-a": (1, "Alpha source."),
        "doc-b": (2, "The exact verified sentence."),
    }
    points = []
    for doc_id, (order_index, _quote) in evidence.items():
        client_db.collection(COLLECTION_NAME).document(doc_id).set({
            "doc_id": doc_id, "title": {"doc-a": "Alpha", "doc-b": "Beta", "doc-c": "Gamma"}[doc_id],
            "source_file": f"Synthetic {doc_id}", "parent_id": ROOT_FOLDER_ID,
            "status": "synced", "dialog_date": "2026-04-01",
        })
        points.append(SimpleNamespace(
            id=f"summary-{doc_id}", score={"doc-c": 0.95, "doc-a": 0.90, "doc-b": 0.85}[doc_id],
            payload={"doc_id": doc_id, "parent_id": ROOT_FOLDER_ID, "type": "summary",
                     "center_index": order_index, "covered_range": [order_index, order_index],
                     "order_index": order_index},
        ))
    from core.integrations.qdrant import client as qdrant_client
    monkeypatch.setattr(qdrant_client, "_single_attempt_client", None)
    monkeypatch.setattr(qdrant_client, "_client", None)
    empty_embedding = types.EmbedContentResponse(
        embeddings=[types.ContentEmbedding(values=[0.25] * 768)],
    )
    sdk_fakes.gemini.queue("embed_content", response=empty_embedding)
    sdk_fakes.gemini.queue("embed_content", response=empty_embedding)
    for response_points in (points, points, []):
        sdk_fakes.qdrant.queue("query_points", response=SimpleNamespace(points=response_points))
    for doc_id, (order_index, quote) in evidence.items():
        sdk_fakes.qdrant.queue("scroll", response=([SimpleNamespace(payload={
            "doc_id": doc_id, "order_index": order_index, "speaker": "Mira",
            "text": quote, "dialog_date": "2026-04-01", "parent_id": ROOT_FOLDER_ID,
        })], None))


def _thinking_level(row):
    config = row.get("arguments", {}).get("config")
    thinking = getattr(config, "thinking_config", None)
    level = getattr(thinking, "thinking_level", None)
    return getattr(level, "value", level)


def _native_function_names(row):
    config = row.get("arguments", {}).get("config")
    names = set()
    for tool in getattr(config, "tools", []) or []:
        for declaration in getattr(tool, "function_declarations", []) or []:
            name = getattr(declaration, "name", None)
            if isinstance(name, str):
                names.add(name)
    return names


def _assert_prepared_question(result, *document_ids):
    from services.connectors.telegram.queue import TelegramQueueCompletion

    assert isinstance(result, TelegramQueueCompletion)
    assert result.result_type == "clarification"
    question = result.clarification.request.get("question")
    assert isinstance(question, dict) and question
    prompt = result.clarification.request.get("prompt")
    assert isinstance(prompt, dict) and isinstance(prompt.get("body"), str)
    assert prompt["body"].strip()
    assert result.clarification.pending.missing_slots
    serialized = json.dumps(question, ensure_ascii=False)
    for document_id in document_ids:
        assert document_id in serialized


def _assert_terminal_refusal_or_clarification(result, sent_rows):
    """Allow a safe terminal refusal or a prepared clarification, never a RUNNING stub."""
    from core.operations.queue import QueueOperationState
    from services.connectors.telegram.queue import TelegramQueueCompletion

    if isinstance(result, TelegramQueueCompletion):
        assert result.result_type in {"clarification", "action", "ordinary"}
        if result.result_type == "clarification":
            _assert_prepared_question(result)
            return
    else:
        assert result in {QueueOperationState.COMPLETED, QueueOperationState.FAILED}
    bodies = [row.get("text") for row in sent_rows
              if isinstance(row.get("text"), str) and row["text"].strip()]
    assert bodies, "a safe terminal refusal must send a nonempty reply"

