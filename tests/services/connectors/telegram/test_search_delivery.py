"""Search presentation delivery contracts across the real runtime boundary."""

from __future__ import annotations

import asyncio
import hashlib
from datetime import timedelta

import pytest

from tests.services.connectors.telegram.test_search_presentation import (
    _bot_client,
    _native_response,
    _running_command,
    _script_source_retrieval,
)

pytestmark = pytest.mark.offline


def _native_call_response(name, arguments, call_id):
    from google.genai import types

    return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(
        parts=[types.Part(function_call=types.FunctionCall(
            id=call_id, name=name, args=arguments,
        ))],
    ))])


def _queue_search(sdk_fakes, blocks, *, query="project discussions", filters=None):
    arguments = {"query": query, **(filters or {})}
    sdk_fakes.gemini.queue("generate_content", response=_native_response("route_command", {
        "kind": "search", "target_ref": None, "target_operation_id": None,
        "dialog_number": None, "result_refs": [],
        "substantive_text": "Find project discussions",
        "tool_choice": {"name": "query", "arguments": arguments},
        "certainty": "certain",
    }))
    sdk_fakes.gemini.queue("generate_content", response=_native_response(
        "present_search_answer", {"blocks": blocks},
    ))


def test_search_terminal_is_pinned_before_send_and_delivery_retry_repairs_proof(
    firestore_fake, sdk_fakes, monkeypatch,
):
    """A lost proof callback is repaired from the delivered row without resending."""
    from core.integrations.gemini.agent import GeminiAgent
    from services.connectors.telegram.commands import parse_command
    from services.connectors.telegram.notifications import TelegramNotificationPublisher
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
    blocks = [{
        "type": "document_list", "list_id": "visible-projects", "entries": [
            {"doc_id": "doc-a", "label": "Alpha",
             "source_refs": [{"doc_id": "doc-a", "order_index": 1}]},
            {"doc_id": "doc-b", "label": "Beta",
             "source_refs": [{"doc_id": "doc-b", "order_index": 2}]},
            {"doc_id": "doc-c", "label": "Gamma",
             "source_refs": [{"doc_id": "doc-c", "order_index": 3}]},
        ],
    }]
    expected_body = "1. Alpha\n2. Beta\n3. Gamma"
    _queue_search(sdk_fakes, blocks)

    original_publish = repository.publish_presentation_delivery
    failed_once = {"value": False}

    def fail_after_telegram_delivery(*args, **kwargs):
        if not failed_once["value"]:
            failed_once["value"] = True
            raise RuntimeError("synthetic proof checkpoint interruption")
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(repository, "publish_presentation_delivery", fail_after_telegram_delivery)
    pre_send = []

    def observe_before_send(_payload, sent):
        if sent.get("text") != expected_body:
            return
        context = repository.get_context(accepted.context_id) or {}
        conversation_id = context.get("active_conversation_id")
        assert isinstance(conversation_id, str)
        conversation_ref = (client_db.collection("telegram_sessions").document(accepted.context_id)
                            .collection("conversations").document(conversation_id))
        headers = [row.to_dict() for row in conversation_ref.collection("presentations").stream()]
        assert len(headers) == 1
        header = headers[0]
        assert "body" not in header
        manifest = repository.get_presentation_manifest(
            accepted.context_id, conversation_id, header["presentation_id"],
        )
        assert manifest["body"] == expected_body
        assert manifest["body_sha256"] == hashlib.sha256(expected_body.encode()).hexdigest()
        assert manifest["notification_refs"]
        for ref in manifest["notification_refs"]:
            notification = repository.get_notification(ref["notification_id"])
            assert notification["revision"] == ref["revision"]
            assert notification["body"] == expected_body
            assert notification["state"] in {"pending", "sending"}
        pre_send.append((conversation_id, manifest["presentation_id"], manifest))

    bot._before_send = observe_before_send
    try:
        asyncio.run(runtime(
            accepted.context_id, operation, lease,
            command=parse_command("Find the three project discussions"),
        ))

        assert pre_send
        conversation_id, presentation_id, manifest = pre_send[0]
        assert len(manifest["notification_refs"]) == 1
        context_ref = (client_db.collection("telegram_sessions").document(accepted.context_id)
                       .collection("conversations").document(conversation_id))
        header = context_ref.collection("presentations").document(presentation_id).get().to_dict()
        assert header["body_ref"]["body_sha256"] == hashlib.sha256(expected_body.encode()).hexdigest()
        notification_id = manifest["notification_refs"][0]["notification_id"]
        notification = repository.get_notification(notification_id)
        assert notification["state"] == "delivered"
        assert type(notification["message_id"]) is int
        assert not list(client_db.collection("notification_outbox").document(notification_id)
                        .collection("presentation_delivery_proofs").stream())
        sent_count = len(bot._test_sent)
        assert context_ref.get().to_dict().get("last_presentation_id") is None

        # A separate invocation can recover the actual delivered revision proof.
        publisher = TelegramNotificationPublisher(repository, bot, clock=lambda: now)
        assert asyncio.run(publisher.publish(notification_id)) == "delivered"
        assert len(bot._test_sent) == sent_count
        assert context_ref.get().to_dict()["last_presentation_id"] == presentation_id
        assert context_ref.get().to_dict()["last_presented_list_id"] == "visible-projects"
        proofs = list(client_db.collection("notification_outbox").document(notification_id)
                      .collection("presentation_delivery_proofs").stream())
        assert len(proofs) == 1
        proof = proofs[0].to_dict()
        assert proof["message_id"] == notification["message_id"]
        assert proof["revision"] == manifest["notification_refs"][0]["revision"]
    finally:
        asyncio.run(bot._test_http_client.aclose())


def test_independent_publisher_waits_for_presentation_header_before_send(
    firestore_fake, sdk_fakes, monkeypatch,
):
    """Prepared rows cannot be sent in the gap before their immutable header."""
    from core.integrations.gemini.agent import GeminiAgent
    from services.connectors.telegram.commands import parse_command
    from services.connectors.telegram.notifications import TelegramNotificationPublisher
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
    body = "A prepared answer with verified sources."
    _queue_search(sdk_fakes, [{"type": "paragraph", "text": body, "source_refs": [
        {"doc_id": "doc-b", "order_index": 2},
    ]}])
    publisher = TelegramNotificationPublisher(repository, bot, clock=lambda: now)
    original_save = repository.save_presentation_manifest
    preheader_attempts = []

    def attempt_before_header(*args, **kwargs):
        manifest = args[1]
        assert not repository.get_presentation_manifest(
            accepted.context_id, manifest["conversation_id"], manifest["presentation_id"],
        )
        refs = manifest["notification_refs"]
        assert refs
        for ref in refs:
            notification = repository.get_notification(ref["notification_id"])
            assert notification["presentation_id"] == manifest["presentation_id"]
            assert notification["revision"] == ref["revision"]
            assert asyncio.run(publisher.publish(ref["notification_id"])) == "busy"
            notification = repository.get_notification(ref["notification_id"])
            assert notification["send_attempts_started"] == 0
        assert not bot._test_sent
        preheader_attempts.extend(refs)
        return original_save(*args, **kwargs)

    monkeypatch.setattr(repository, "save_presentation_manifest", attempt_before_header)
    try:
        asyncio.run(runtime(
            accepted.context_id, operation, lease,
            command=parse_command("Find project discussions"),
        ))
        assert preheader_attempts
        assert [row["text"] for row in bot._test_sent] == [body]
    finally:
        asyncio.run(bot._test_http_client.aclose())


def test_partial_multichunk_list_keeps_earlier_list_proof_and_withholds_current_list(
    firestore_fake, sdk_fakes, monkeypatch,
):
    """A fully shown earlier list keeps proof when a later list is only partly sent."""
    from core.integrations.gemini.agent import GeminiAgent
    from services.connectors.telegram.commands import parse_command
    from services.connectors.telegram.runtime import TelegramCommandRuntime
    from tests.fixtures.sdk_fakes import FakeGeminiClient

    client_db = firestore_fake.client_class()
    _script_source_retrieval(sdk_fakes, monkeypatch, client_db)
    repository, accepted, operation, lease, now = _running_command(client_db)
    bot = _bot_client(fail_send_number=2)
    runtime = TelegramCommandRuntime(
        repository=repository, save_handler=object(), client=bot,
        model=GeminiAgent(client=FakeGeminiClient(api_key="synthetic-presentation-key")),
        clock=lambda: now,
    )
    huge_entries = [
        {"doc_id": doc_id, "label": f"Entry {index} " + ("x" * 1400),
         "source_refs": [{"doc_id": doc_id, "order_index": order_index}]}
        for index, (doc_id, order_index) in enumerate(
            (("doc-a", 1), ("doc-b", 2), ("doc-c", 3)), 1,
        )
    ]
    blocks = [
        {"type": "document_list", "list_id": "shown-first", "entries": [
            {"doc_id": "doc-a", "label": "Alpha",
             "source_refs": [{"doc_id": "doc-a", "order_index": 1}]},
            {"doc_id": "doc-b", "label": "Beta",
             "source_refs": [{"doc_id": "doc-b", "order_index": 2}]},
        ]},
        {"type": "paragraph", "text": "x" * 120, "source_refs": []},
        {"type": "document_list", "list_id": "partial-later", "entries": huge_entries},
    ]
    _queue_search(sdk_fakes, blocks)
    before_send = []
    callback_errors = []

    def inspect_before_send(_payload, sent):
        if not sent.get("text"):
            return
        try:
            context = repository.get_context(accepted.context_id) or {}
            conversation_id = context.get("active_conversation_id")
            assert isinstance(conversation_id, str)
            conversation_ref = (client_db.collection("telegram_sessions").document(accepted.context_id)
                                .collection("conversations").document(conversation_id))
            headers = list(conversation_ref.collection("presentations").stream())
            assert len(headers) == 1
            manifest = repository.get_presentation_manifest(
                accepted.context_id, conversation_id, headers[0].id,
            )
            refs = manifest["notification_refs"]
            assert len(refs) == 2
            expected_states = (["sending", "pending"] if len(bot._test_sent) == 1
                               else ["delivered", "sending"])
            for index, ref in enumerate(refs):
                notification = repository.get_notification(ref["notification_id"])
                assert notification["state"] == expected_states[index]
                assert notification["presentation_id"] == manifest["presentation_id"]
                assert notification["revision"] == ref["revision"]
            before_send.append((conversation_id, manifest))
        except Exception as error:
            callback_errors.append(error)

    bot._before_send = inspect_before_send
    try:
        result = asyncio.run(runtime(
            accepted.context_id, operation, lease,
            command=parse_command("Find project discussions"),
        ))
        assert before_send
        assert not callback_errors, callback_errors
        assert result.value == "failed"
        conversation_id, manifest = before_send[0]
        assert len(bot._test_sent) == 2
        conversation_state = (client_db.collection("telegram_sessions").document(accepted.context_id)
                              .collection("conversations").document(conversation_id)
                              .get().to_dict())
        assert conversation_state["last_presentation_id"] == manifest["presentation_id"]
        assert conversation_state["last_presented_list_id"] is None
        proofs_by_index = {}
        for ref in manifest["notification_refs"]:
            rows = list(client_db.collection("notification_outbox").document(
                ref["notification_id"],
            ).collection("presentation_delivery_proofs").stream())
            if rows:
                proofs_by_index.update({row.to_dict()["chunk_index"]: row.to_dict() for row in rows})
        assert set(proofs_by_index) == {0}
        assert proofs_by_index[0]["presentation_id"] == manifest["presentation_id"]
        prepared_rows = [repository.get_notification(ref["notification_id"])
                         for ref in manifest["notification_refs"]]
        assert [row["revision"] for row in prepared_rows] == [
            ref["revision"] for ref in manifest["notification_refs"]
        ]
        assert [row["body"] for row in prepared_rows] == [
            item["text"] for item in bot._test_sent
        ]
        assert prepared_rows[0]["state"] == "delivered"
        assert prepared_rows[1]["state"] == "failed"
    finally:
        asyncio.run(bot._test_http_client.aclose())


def test_search_continuation_can_present_a_verified_exchange_from_prior_turn(
    firestore_fake, sdk_fakes, monkeypatch,
):
    """A later search answer can anchor to canonical same-session prior evidence."""
    from core.integrations.firestore.operation_queue import claim_head
    from core.integrations.gemini.agent import GeminiAgent
    from services.connectors.telegram.commands import parse_command
    from services.connectors.telegram.runtime import TelegramCommandRuntime
    from tests.fixtures.sdk_fakes import FakeGeminiClient

    client_db = firestore_fake.client_class()
    _script_source_retrieval(sdk_fakes, monkeypatch, client_db)
    _script_source_retrieval(sdk_fakes, monkeypatch, client_db)
    repository, first, first_operation, first_lease, now = _running_command(client_db)
    bot = _bot_client()
    runtime = TelegramCommandRuntime(
        repository=repository, save_handler=object(), client=bot,
        model=GeminiAgent(client=FakeGeminiClient(api_key="synthetic-presentation-key")),
        clock=lambda: now,
    )
    first_blocks = [{"type": "paragraph", "text": "Initial search saved.", "source_refs": [
        {"doc_id": "doc-b", "order_index": 2},
    ]}]
    _queue_search(sdk_fakes, first_blocks, query="initial project discussions")
    try:
        first_result = asyncio.run(runtime(
            first.context_id, first_operation, first_lease,
            command=parse_command("Find project discussions"),
        ))
        from core.integrations.firestore.operation_queue import complete_head
        assert complete_head(
            client_db, client_db.collection("telegram_sessions").document(first.context_id),
            operation_id=first.operation_id, owner=first_lease.owner,
            fence=first_lease.fence, now=now + timedelta(seconds=1),
        ), first_result
        context_ref = client_db.collection("telegram_sessions").document(first.context_id)
        context = repository.get_context(first.context_id) or {}
        conversation_id = context.get("active_conversation_id")
        assert isinstance(conversation_id, str) and conversation_id.startswith("search-")
        prior_tool_turns = [turn for turn in repository.list_turns(first.context_id, conversation_id)
                            if isinstance(turn, dict) and turn.get("role") == "tool"
                            and any(isinstance(part, dict)
                                    and part.get("type") == "search_exchange_ref"
                                    for part in turn.get("parts", []))]
        assert prior_tool_turns
        prior_turn = prior_tool_turns[-1]
        prior_part = next(part for part in prior_turn["parts"]
                          if part.get("type") == "search_exchange_ref")
        prior_checkpoint = prior_part["checkpoint_conversation_id"]
        prior_call_id = prior_part["call_id"]

        accepted = repository.accept_command(
            88102, chat_id=42017, thread_id=73, user_id=17,
            command=parse_command("Search the previous discussion again"),
            now=now + timedelta(seconds=2),
        )
        lease = claim_head(client_db, context_ref, owner="presentation-continuation-worker",
                           now=now + timedelta(seconds=2))
        assert lease is not None and lease.operation_id == accepted.operation_id
        operation = (context_ref.collection("operations").document(accepted.operation_id)
                     .get().to_dict())
        blocks = [{"type": "quote", "text": "The exact verified sentence.",
                   "doc_id": "doc-b", "source_refs": [{
                       "doc_id": "doc-b", "order_index": 2,
                       "operation_id": first.operation_id,
                       "checkpoint_conversation_id": prior_checkpoint,
                       "call_id": prior_call_id,
                   }]}]
        sdk_fakes.gemini.queue("generate_content", response=_native_response("route_command", {
            "kind": "substantive", "target_ref": None, "target_operation_id": None,
            "dialog_number": None, "result_refs": [],
            "substantive_text": "Search the previous discussion again",
            "tool_choice": {"name": "query", "arguments": {"query": "previous discussion"}},
            "certainty": "certain",
        }))
        sdk_fakes.gemini.queue("generate_content", response=_native_response(
            "present_search_answer", {"blocks": blocks},
        ))
        asyncio.run(runtime(
            accepted.context_id, operation, lease,
            command=parse_command("Search the previous discussion again"),
        ))

        headers = list(client_db.collection("telegram_sessions").document(first.context_id)
                       .collection("conversations").document(conversation_id)
                       .collection("presentations").stream())
        assert len(headers) == 2
        second_manifest = repository.get_presentation_manifest(
            first.context_id, conversation_id,
            "search-presentation-" + hashlib.sha256(accepted.operation_id.encode()).hexdigest(),
        )
        assert second_manifest["body"] == "> The exact verified sentence."
        prior_sources = [source for source in second_manifest["sources"]
                         if source["operation_id"] == first.operation_id]
        assert prior_sources and all(
            source["reference"]["payload"]["sha256"]
            == prior_part["reference"]["payload"]["sha256"]
            for source in prior_sources
        )
        answer_turn_id = "search-answer-" + hashlib.sha256(
            accepted.operation_id.encode(),
        ).hexdigest()
        expected_source_turn_id = "search-tool-" + hashlib.sha256(
            f"{first.operation_id}\0{prior_part['ordinal']}".encode(),
        ).hexdigest()
        header = next(row.to_dict() for row in headers
                      if row.id == second_manifest["presentation_id"])
        assert header["body_ref"]["turn_id"] == answer_turn_id
        assert expected_source_turn_id in header["source_turn_refs"].values()
        assert any(row.get("text") == second_manifest["body"] for row in bot._test_sent)
    finally:
        asyncio.run(bot._test_http_client.aclose())


def test_continuation_without_route_query_pins_its_first_native_query(
    firestore_fake, sdk_fakes, monkeypatch,
):
    """A continuation's original bounds come from its first in-agent scoped read."""
    from core.integrations.firestore.operation_queue import claim_head, complete_head
    from core.integrations.gemini.agent import GeminiAgent
    from services.connectors.telegram.commands import parse_command
    from services.connectors.telegram.runtime import TelegramCommandRuntime
    from tests.fixtures.sdk_fakes import FakeGeminiClient

    client_db = firestore_fake.client_class()
    _script_source_retrieval(sdk_fakes, monkeypatch, client_db)
    _script_source_retrieval(sdk_fakes, monkeypatch, client_db)
    repository, first, first_operation, first_lease, now = _running_command(client_db)
    bot = _bot_client()
    runtime = TelegramCommandRuntime(
        repository=repository, save_handler=object(), client=bot,
        model=GeminiAgent(client=FakeGeminiClient(api_key="synthetic-presentation-key")),
        clock=lambda: now,
    )
    _queue_search(sdk_fakes, [{"type": "paragraph", "text": "First answer.",
                               "source_refs": [{"doc_id": "doc-b", "order_index": 2}]}])
    queue_context = client_db.collection("telegram_sessions").document(first.context_id)
    try:
        asyncio.run(runtime(
            first.context_id, first_operation, first_lease,
            command=parse_command("Find project discussions"),
        ))
        assert complete_head(
            client_db, queue_context, operation_id=first.operation_id,
            owner=first_lease.owner, fence=first_lease.fence,
            now=now + timedelta(seconds=1),
        )
        session_id = repository.get_context(first.context_id)["active_conversation_id"]
        accepted = repository.accept_command(
            88103, chat_id=42017, thread_id=73, user_id=17,
            command=parse_command("Continue the same project search"),
            now=now + timedelta(seconds=2),
        )
        lease = claim_head(client_db, queue_context, owner="native-query-continuation-worker",
                           now=now + timedelta(seconds=2))
        assert lease is not None and lease.operation_id == accepted.operation_id
        operation = (queue_context.collection("operations").document(accepted.operation_id)
                     .get().to_dict())
        sdk_fakes.gemini.queue("generate_content", response=_native_response("route_command", {
            "kind": "substantive", "target_ref": None, "target_operation_id": None,
            "dialog_number": None, "result_refs": [],
            "substantive_text": "Continue the same project search",
            "tool_choice": None, "certainty": "certain",
        }))
        query_arguments = {"query": "same project search"}
        native_call_id = "continuation-query-call"
        sdk_fakes.gemini.queue("generate_content", response=_native_call_response(
            "query", query_arguments, native_call_id,
        ))
        sdk_fakes.gemini.queue("generate_content", response=_native_response(
            "present_search_answer", {"blocks": [{
                "type": "paragraph", "text": "Continuation answer.", "source_refs": [{
                    "doc_id": "doc-b", "order_index": 2,
                    "call_id": native_call_id, "ordinal": 0,
                }],
            }]},
        ))

        result = asyncio.run(runtime(
            accepted.context_id, operation, lease,
            command=parse_command("Continue the same project search"),
        ))
        sdk_calls = [(row.get("boundary"), row.get("operation"))
                     for row in sdk_fakes.trace]
        from collections import Counter
        sdk_call_counts = Counter(sdk_calls)
        assert result.value == "completed"
        assert sdk_call_counts[("gemini", "embed_content")] == 4
        assert sdk_call_counts[("qdrant", "query_points")] == 6
        assert sdk_call_counts[("qdrant", "scroll")] == 6
        assert sdk_call_counts[("gemini", "generate_content")] == 5
        from services.connectors.telegram.runtime.ledger import FirestoreAgentLedger
        checkpoint_id = "telegram-global-" + hashlib.sha256(
            f"{accepted.context_id}\0{accepted.operation_id}".encode(),
        ).hexdigest()
        ledger = FirestoreAgentLedger(
            client_db, context_id=accepted.context_id, operation_id=accepted.operation_id,
            owner=lease.owner, fence=lease.fence, clock=lambda: now,
        )
        checkpoint = ledger.load_turn(checkpoint_id, accepted.operation_id)
        native_call = next(call for call in checkpoint["tool_calls"]
                           if call.get("id") == native_call_id)
        source_turn = next(turn for turn in repository.list_turns(
            accepted.context_id, session_id,
        ) if turn.get("operation_id") == accepted.operation_id and turn.get("role") == "tool")
        source_part = next(part for part in source_turn["parts"]
                           if part.get("type") == "search_exchange_ref"
                           and part.get("call_id") == native_call_id)
        exchange = ledger.load_search_exchange(
            source_part["reference"], conversation_id=session_id,
            operation_id=accepted.operation_id,
            checkpoint_conversation_id=source_part["checkpoint_conversation_id"],
            call_id=source_part["call_id"], ordinal=source_part["ordinal"],
        )
        assert exchange["call"] == native_call
        assert exchange["call"]["thought_signature"] == native_call["thought_signature"]
        presentation_id = "search-presentation-" + hashlib.sha256(
            accepted.operation_id.encode(),
        ).hexdigest()
        manifest = repository.get_presentation_manifest(
            accepted.context_id, session_id, presentation_id,
        )
        source = next(item for item in manifest["sources"]
                      if item["call_id"] == native_call_id)
        assert source["ordinal"] == 0
        assert source["source_filters"] == query_arguments
        header = (queue_context.collection("conversations").document(session_id)
                  .collection("presentations").document(presentation_id).get().to_dict())
        source_turn_id = header["source_turn_refs"][source["source_id"]]
        source_turn = next(turn for turn in repository.list_turns(accepted.context_id, session_id)
                           if turn["turn_id"] == source_turn_id)
        assert source_turn["operation_id"] == accepted.operation_id
    finally:
        asyncio.run(bot._test_http_client.aclose())


def test_narrower_current_bounds_reject_broader_historical_presentation_proof(
    firestore_fake, sdk_fakes, monkeypatch,
):
    """A historical read outside current folder/date bounds cannot create authority."""
    from core.integrations.firestore.operation_queue import claim_head, complete_head
    from core.integrations.gemini.agent import GeminiAgent
    from services.connectors.telegram.commands import parse_command
    from services.connectors.telegram.runtime import TelegramCommandRuntime
    from tests.fixtures.sdk_fakes import FakeGeminiClient

    client_db = firestore_fake.client_class()
    _script_source_retrieval(sdk_fakes, monkeypatch, client_db)
    _script_source_retrieval(sdk_fakes, monkeypatch, client_db)
    repository, first, first_operation, first_lease, now = _running_command(client_db)
    bot = _bot_client()
    runtime = TelegramCommandRuntime(
        repository=repository, save_handler=object(), client=bot,
        model=GeminiAgent(client=FakeGeminiClient(api_key="synthetic-presentation-key")),
        clock=lambda: now,
    )
    _queue_search(
        sdk_fakes, [{"type": "paragraph", "text": "First bounded answer.", "source_refs": [
            {"doc_id": "doc-b", "order_index": 2},
        ]}],
    )
    queue_context = client_db.collection("telegram_sessions").document(first.context_id)
    try:
        asyncio.run(runtime(
            first.context_id, first_operation, first_lease,
            command=parse_command("Find project discussions"),
        ))
        assert complete_head(
            client_db, queue_context, operation_id=first.operation_id,
            owner=first_lease.owner, fence=first_lease.fence,
            now=now + timedelta(seconds=1),
        )
        session_id = repository.get_context(first.context_id)["active_conversation_id"]
        prior_tool = next(turn for turn in repository.list_turns(first.context_id, session_id)
                          if turn.get("operation_id") == first.operation_id
                          and turn.get("role") == "tool")
        prior_ref = next(part for part in prior_tool["parts"]
                         if part.get("type") == "search_exchange_ref")
        accepted = repository.accept_command(
            88104, chat_id=42017, thread_id=73, user_id=17,
            command=parse_command("Search only recent Client reports"),
            now=now + timedelta(seconds=2),
        )
        lease = claim_head(client_db, queue_context, owner="narrow-bounds-worker",
                           now=now + timedelta(seconds=2))
        assert lease is not None and lease.operation_id == accepted.operation_id
        operation = (queue_context.collection("operations").document(accepted.operation_id)
                     .get().to_dict())
        current_filters = {
            "date_from": "2026-06-01", "date_to": "2026-12-31",
        }
        sdk_fakes.gemini.queue("generate_content", response=_native_response("route_command", {
            "kind": "substantive", "target_ref": None, "target_operation_id": None,
            "dialog_number": None, "result_refs": [],
            "substantive_text": "Search only recent Client reports",
            "tool_choice": {"name": "query", "arguments": {
                "query": "recent Client reports", **current_filters,
            }},
            "certainty": "certain",
        }))
        sdk_fakes.gemini.queue("generate_content", response=_native_response(
            "present_search_answer", {"blocks": [{
                "type": "paragraph", "text": "A historical answer.", "source_refs": [{
                    "doc_id": "doc-b", "order_index": 2,
                    "operation_id": first.operation_id,
                    "checkpoint_conversation_id": prior_ref["checkpoint_conversation_id"],
                    "call_id": prior_ref["call_id"], "ordinal": prior_ref["ordinal"],
                }],
            }]},
        ))

        result = asyncio.run(runtime(
            accepted.context_id, operation, lease,
            command=parse_command("Search only recent Client reports"),
        ))
        assert result.value == "failed"
        presentations = list(queue_context.collection("conversations").document(session_id)
                             .collection("presentations").stream())
        assert len(presentations) == 1
        assert presentations[0].id == "search-presentation-" + hashlib.sha256(
            first.operation_id.encode(),
        ).hexdigest()
        session = queue_context.collection("conversations").document(session_id).get().to_dict()
        assert session.get("last_presentation_id") is None
        assert session.get("last_presented_list_id") is None
    finally:
        asyncio.run(bot._test_http_client.aclose())
