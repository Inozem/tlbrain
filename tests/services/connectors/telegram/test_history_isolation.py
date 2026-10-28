"""Offline history isolation and switch notice barrier contracts."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from types import SimpleNamespace

import pytest

from services.connectors.telegram.models import TelegramContext


pytestmark = pytest.mark.offline


def _process(**kwargs):
    """Load the future operation seam lazily so this test module collects now."""
    from services.connectors.telegram.sessions import process_operation

    return asyncio.run(process_operation(**kwargs))


class _UnknownDelivery(Exception):
    """A send may have reached Telegram, but its outcome is unknown."""


class _Repository:
    """Minimal in-memory conversations, Docs, and search-index state."""

    def __init__(self, context, conversations, events):
        self.events = events
        self.context_id = context.context_id
        self.selection = {
            "active_conversation_id": context.active_conversation_id,
            "selected_dialog_number": context.selected_dialog_number,
            "selection_sequence": context.selection_sequence,
        }
        self.conversations = deepcopy(conversations)
        self.docs = {"doc-old": {"saved": "old-doc"}, "doc-target": {"saved": "target-doc"}}
        self.search_index = {"doc-old": ["old-index"], "doc-target": ["target-index"]}
        self.route_records = {}
        self.writes = []
        self.retry_requests = []
        self.retry_accepted = True
        self.retry_operations = []

    def snapshot(self):
        return deepcopy((self.selection, self.conversations, self.docs, self.search_index))

    def route_snapshot(self):
        return deepcopy((self.conversations, self.docs, self.search_index))

    def persist_route(self, operation_id, route):
        self.route_records[operation_id] = deepcopy(route)
        self.writes.append(("route", operation_id, deepcopy(route)))

    def retry_failed_draft_for_command(
        self, context_id, dialog_number, *, request_operation_id, now,
    ):
        self.retry_requests.append((context_id, dialog_number, request_operation_id))
        if not self.retry_accepted:
            return SimpleNamespace(accepted=False, operation_id=None)
        self.retry_operations.append("retry-operation")
        return SimpleNamespace(accepted=True, operation_id="retry-operation")

    def list_turns(self, context_id, conversation_id):
        assert context_id == self.context_id
        return self.conversations[conversation_id]

    def append_turn(
        self, context_id, conversation_id, *, turn_id, operation_id,
        role, parts, now,
    ):
        assert context_id == self.context_id
        turn = {
            "turn_id": turn_id, "operation_id": operation_id,
            "role": role, "parts": deepcopy(parts),
        }
        existing = next((row for row in self.conversations[conversation_id]
                         if row.get("turn_id") == turn_id), None)
        if existing is not None:
            assert {key: existing.get(key) for key in ("operation_id", "role", "parts")} == {
                key: turn[key] for key in ("operation_id", "role", "parts")
            }
            return existing
        self.conversations[conversation_id].append(turn)
        self.writes.append(("turn", conversation_id, deepcopy(turn)))
        self.events.append(("append", conversation_id, deepcopy(turn)))
        return turn

    def save_doc(self, target_doc_id, content):
        self.docs[target_doc_id] = content
        self.writes.append(("doc", target_doc_id, content))

    def update_search_index(self, target_doc_id, content):
        self.search_index[target_doc_id] = [content]
        self.writes.append(("index", target_doc_id, content))


class _EnvelopeRepository(_Repository):
    """Record the route/cleanup boundary used by the Firestore adapter."""

    def persist_route(self, operation_id, route, **lease):
        super().persist_route(operation_id, route)
        self.events.append(("route_persisted", operation_id))
        return True

    def clear_command_candidate(self, context_id, operation_id, **lease):
        self.events.append(("candidate_cleared", operation_id))
        return True


class _Router:
    def __init__(self, repository, route):
        self.repository = repository
        self.route = deepcopy(route)
        self.calls = []
        self.observed = []

    async def __call__(self, *, operation_id, context, text):
        self.calls.append({"operation_id": operation_id, "context": context, "text": text})
        self.observed.append(self.repository.route_snapshot())
        return deepcopy(self.route)


class _Agent:
    def __init__(self, events, result="synthetic answer", before_return=None):
        self.result = result
        self.prompts = []
        self.events = events
        self.before_return = before_return

    async def __call__(self, *, prompt, target_doc_id):
        self.prompts.append({"prompt": deepcopy(prompt), "target_doc_id": target_doc_id})
        self.events.append(("agent", target_doc_id, deepcopy(prompt)))
        if self.before_return is not None:
            await self.before_return()
        return self.result


class _Notifications:
    def __init__(self, events, outcomes=()):
        self.outcomes = list(outcomes)
        self.attempts = []
        self.events = events

    async def send_switch_notice(
        self, *, context_id=None, operation_id, notification_id, target_doc_id, dialog_number,
    ):
        while True:
            self.attempts.append({
                "operation_id": operation_id,
                "notification_id": notification_id,
                "target_doc_id": target_doc_id,
                "dialog_number": dialog_number,
                "context_id": context_id,
                "body": f"Starting to process dialog #{dialog_number}.",
            })
            self.events.append(("notice_attempt", operation_id, notification_id))
            outcome = self.outcomes.pop(0) if self.outcomes else "confirmed"
            if isinstance(outcome, Exception):
                raise outcome
            if outcome == "confirmed":
                self.events.append(("notice_confirmed", operation_id, notification_id))
                return outcome
            if outcome != "retry" or len(self.attempts) >= 3:
                return outcome


class _Queue:
    def __init__(self):
        self.released = []

    async def release_head(self, operation_id):
        self.released.append(operation_id)


def _context():
    return TelegramContext(
        chat_id=42,
        thread_id=7,
        active_conversation_id="conversation:old",
        selected_dialog_number=1,
        selection_sequence=4,
    )


def _harness(route, *, outcomes=()):
    context = _context()
    events = []
    repository = _Repository(context, {
        "conversation:old": [
            {"role": "user", "parts": [{"text": "old user turn"}]},
            {"role": "model", "parts": [{"text": "old answer"}]},
        ],
        "conversation:target": [
            {"role": "user", "parts": [{"text": "target user turn"}]},
            {"role": "model", "parts": [{"text": "target answer"}]},
        ],
    }, events)
    router = _Router(repository, route)
    agent = _Agent(events)
    notifications = _Notifications(events, outcomes)
    queue = _Queue()
    return context, repository, router, agent, notifications, queue


def _route(kind, *, target_ref="doc-target", substantive_text=None):
    return {
        "kind": kind,
        "target_ref": target_ref,
        "target_doc_id": target_ref,
        "conversation_id": "conversation:target",
        "dialog_number": 2,
        "result_refs": [],
        "substantive_text": substantive_text,
        "tool_choice": None,
        "certainty": "certain",
    }


def _call(operation_id, context, repository, router, agent, notifications, queue, text):
    return _process(
        operation_id=operation_id,
        context=context,
        text=text,
        repository=repository,
        router=router,
        agent=agent,
        notifications=notifications,
        queue=queue,
        selection_publisher=lambda op, *, conversation_id, dialog_number: _publish_selection(
            repository, op, conversation_id=conversation_id,
            dialog_number=dialog_number,
        ),
    )


def _publish_selection(repository, operation_id, *, conversation_id, dialog_number):
    repository.selection = {
        "active_conversation_id": conversation_id,
        "selected_dialog_number": dialog_number,
        "selection_sequence": repository.selection["selection_sequence"] + 1,
    }
    repository.writes.append(("selection", operation_id, deepcopy(repository.selection)))
    repository.events.append(("selection", operation_id, deepcopy(repository.selection)))


def _contains(value, fragment):
    return fragment in repr(value)


def _current_context(context, repository):
    selection = repository.selection
    return TelegramContext(
        chat_id=context.chat_id,
        thread_id=context.thread_id,
        active_conversation_id=selection["active_conversation_id"],
        selected_dialog_number=selection["selected_dialog_number"],
        selection_sequence=selection["selection_sequence"],
    )


def test_pure_switch_changes_focus_only_after_confirmed_notice_and_never_enters_history():
    control = "switch to dialog 2"
    switch_notice = "Dialog #2: target conversation"
    context, repository, router, agent, notifications, queue = _harness(
        _route("pure_switch"),
    )
    before_histories = deepcopy(repository.conversations)
    before_target_data = deepcopy((repository.docs, repository.search_index))

    result = _call("op-switch", context, repository, router, agent, notifications, queue, control)

    assert result["status"] in {"switched", "completed"}
    assert repository.selection == {
        "active_conversation_id": "conversation:target",
        "selected_dialog_number": 2,
        "selection_sequence": 5,
    }
    assert repository.conversations == before_histories
    assert (repository.docs, repository.search_index) == before_target_data
    assert "switch to dialog 2" not in repr(repository.route_records["op-switch"])
    assert "substantive_text" not in repository.route_records["op-switch"]
    assert "text" not in repository.route_records["op-switch"]
    assert "Dialog #2" not in repr(repository.route_records["op-switch"])
    assert agent.prompts == []
    assert [row[0] for row in repository.writes] == ["route", "selection"]
    assert notifications.attempts[0]["target_doc_id"] == "doc-target"
    assert notifications.attempts[0]["body"] == "Starting to process dialog #2."
    kinds = [event[0] for event in repository.events]
    assert kinds.index("notice_confirmed") < kinds.index("selection")
    assert "agent" not in kinds and "append" not in kinds

    # A later substantive prompt includes target history, never the control or
    # the switch notice. The second operation is routed independently.
    router.route = _route("substantive", substantive_text="continue with the plan")
    _call(
        "op-next", _current_context(context, repository), repository, router, agent, notifications, queue,
        "continue with the plan",
    )
    prompt = agent.prompts[-1]["prompt"]
    assert not _contains(prompt, control)
    assert not _contains(prompt, switch_notice)
    assert not _contains(prompt, "old user turn")
    assert not _contains(prompt, "old answer")


def test_mixed_switch_persists_only_substantive_text_and_result_in_target():
    control = "open dialog 2 and compare the two proposals"
    substantive = "compare the two proposals"
    switch_notice = "Dialog #2: target conversation"
    context, repository, router, agent, notifications, queue = _harness(
        _route("substantive", substantive_text=substantive),
    )
    before_old = deepcopy(repository.conversations["conversation:old"])
    before_target = deepcopy(repository.conversations["conversation:target"])
    before_saved_data = deepcopy((repository.docs, repository.search_index))

    _call("op-mixed", context, repository, router, agent, notifications, queue, control)

    assert repository.conversations["conversation:old"] == before_old
    target = repository.conversations["conversation:target"]
    additions = target[len(before_target):]
    assert [(row["role"], row["parts"]) for row in additions] == [
        ("user", [{"text": substantive}]),
        ("model", [{"text": "synthetic answer"}]),
    ]
    assert all(row["operation_id"] == "op-mixed" for row in additions)
    assert not _contains(target, "open dialog 2")
    assert not _contains(target, switch_notice)
    prompt = agent.prompts[0]["prompt"]
    assert not _contains(prompt, "open dialog 2")
    assert not _contains(prompt, switch_notice)
    assert not _contains(prompt, "old user turn")
    assert not _contains(prompt, "old answer")
    assert "open dialog 2" not in repr(repository.route_records["op-mixed"])
    assert "substantive_text" not in repository.route_records["op-mixed"]
    kinds = [event[0] for event in repository.events]
    notice_position = kinds.index("notice_confirmed")
    selection_position = kinds.index("selection")
    append_position = kinds.index("append")
    agent_position = kinds.index("agent")
    assert notice_position < selection_position < append_position < agent_position
    assert (repository.docs, repository.search_index) == before_saved_data
    assert not any(row[0] in {"doc", "index"} for row in repository.writes)


def test_automatic_semantic_target_change_preserves_entire_question_in_target():
    question = "How would this approach differ for a small research team?"
    context, repository, router, agent, notifications, queue = _harness(
        _route("substantive", substantive_text=question),
    )
    before_old = deepcopy(repository.conversations["conversation:old"])
    before_saved_data = deepcopy((repository.docs, repository.search_index))

    _call(
        "op-semantic", context, repository, router, agent, notifications, queue,
        "compare it with the research discussion",
    )

    assert repository.conversations["conversation:old"] == before_old
    assert repository.conversations["conversation:target"][-2]["role"] == "user"
    assert repository.conversations["conversation:target"][-2]["parts"] == [{"text": question}]
    assert (repository.docs, repository.search_index) == before_saved_data
    assert not any(row[0] in {"doc", "index"} for row in repository.writes)


def test_router_observes_unchanged_histories_before_target_resolution_or_append():
    context, repository, router, agent, notifications, queue = _harness(
        _route("substantive", substantive_text="new request"),
    )
    before = repository.route_snapshot()

    _call("op-order", context, repository, router, agent, notifications, queue, "new request")

    assert router.observed == [before]
    assert router.calls[0]["text"] == "new request"
    assert repository.writes[0][0] == "route"


def test_operation_replay_reuses_deterministic_turn_ids_and_skips_duplicate_agent_call():
    context, repository, router, agent, notifications, queue = _harness(
        _route("substantive", substantive_text="replay me once"),
    )
    selected_context = _current_context(context, repository)

    first = _call(
        "op-replay", selected_context, repository, router, agent, notifications, queue,
        "raw envelope",
    )
    turns_after_first = deepcopy(repository.conversations["conversation:target"])
    replay = _call(
        "op-replay", selected_context, repository, router, agent, notifications, queue,
        "raw envelope",
    )

    assert first["status"] == replay["status"] == "completed"
    assert repository.conversations["conversation:target"] == turns_after_first
    assert [turn["role"] for turn in turns_after_first[-2:]] == ["user", "model"]
    assert [turn["operation_id"] for turn in turns_after_first[-2:]] == ["op-replay", "op-replay"]
    assert len({turn["turn_id"] for turn in turns_after_first[-2:]}) == 2
    assert len(agent.prompts) == 1


def test_search_route_stays_in_its_separate_session_without_target_history_changes():
    context, repository, router, agent, notifications, queue = _harness({
        "status": "ok",
        "kind": "search",
        "result_refs": ["doc-target"],
        "search_session_id": "search-1",
    })
    before = repository.snapshot()

    result = _call(
        "op-search", context, repository, router, agent, notifications, queue,
        "find the target document",
    )

    assert result == {
        "status": "ok", "kind": "search",
        "result_refs": ["doc-target"], "search_session_id": "search-1",
    }
    assert repository.snapshot() == before
    assert repository.route_records["op-search"] == {
        "status": "ok", "kind": "search",
        "result_refs": ["doc-target"], "search_session_id": "search-1",
    }
    assert "find the target document" not in repr(repository.route_records["op-search"])
    assert agent.prompts == []
    assert notifications.attempts == []


def test_invalid_substantive_route_is_recorded_as_invalid_before_any_side_effect():
    route = _route("substantive", substantive_text=None)
    context, repository, router, agent, notifications, queue = _harness(route)
    before = repository.snapshot()

    result = _call(
        "op-invalid", context, repository, router, agent, notifications, queue,
        "raw command content",
    )

    assert result["status"] == "invalid_route"
    assert repository.route_records["op-invalid"]["status"] == "invalid_route"
    assert repository.snapshot() == before
    assert agent.prompts == []
    assert notifications.attempts == []
    assert "raw command content" not in repr(repository.route_records["op-invalid"])


@pytest.mark.parametrize("malformed", [
    {"status": [], "kind": "pure_switch"},
    {"status": "ok", "kind": []},
])
def test_unhashable_route_status_or_kind_fails_closed(malformed):
    context, repository, router, agent, notifications, queue = _harness(malformed)

    result = _call(
        "op-malformed", context, repository, router, agent, notifications, queue,
        "do not store this command",
    )

    assert result["status"] == "invalid_route"
    assert repository.route_records["op-malformed"]["status"] == "invalid_route"
    assert "do not store this command" not in repr(repository.route_records["op-malformed"])


def test_orchestration_clears_candidate_only_after_successful_route_persistence():
    context, base, router, agent, notifications, queue = _harness(
        _route("substantive", substantive_text="target request"),
    )
    repository = _EnvelopeRepository(context, base.conversations, base.events)

    result = _process(
        operation_id="op-cleanup", context=context, text="raw command envelope",
        repository=repository, router=router, agent=agent,
        notifications=notifications, queue=queue, owner="worker", fence=1,
        now=datetime.now().astimezone(),
    )

    event_kinds = [event[0] for event in repository.events]
    assert result["status"] == "completed"
    assert event_kinds.index("route_persisted") < event_kinds.index("candidate_cleared")
    assert event_kinds.index("append") < event_kinds.index("candidate_cleared")
    assert event_kinds.index("candidate_cleared") < event_kinds.index("agent")


def test_control_route_record_preserves_safe_dependency_operation_ids():
    route = _route("retry")
    route.update({
        "target_operation_id": "target-operation",
        "dependency_operation_id": "dependency-operation",
        "text": "raw command envelope",
    })
    context, repository, router, agent, notifications, queue = _harness(route)

    _call(
        "op-control-target", context, repository, router, agent, notifications, queue,
        "raw command envelope",
    )

    assert repository.route_records["op-control-target"]["target_operation_id"] == "target-operation"
    assert repository.route_records["op-control-target"]["dependency_operation_id"] == "dependency-operation"
    assert "raw command envelope" not in repr(repository.route_records["op-control-target"])


def test_orchestration_does_not_clear_candidate_for_invalid_route():
    context, base, router, agent, notifications, queue = _harness(
        _route("substantive", substantive_text=None),
    )
    repository = _EnvelopeRepository(context, base.conversations, base.events)

    result = _process(
        operation_id="op-invalid-cleanup", context=context,
        text="raw command envelope", repository=repository, router=router,
        agent=agent, notifications=notifications, queue=queue,
        owner="worker", fence=1, now=datetime.now().astimezone(),
    )

    assert result["status"] == "invalid_route"
    assert not any(event[0] == "candidate_cleared" for event in repository.events)


@pytest.mark.parametrize("route", [
    {"status": "ok", "kind": "save", "confirmed": True},
    {
        **_route("retry"),
        "status": "ok",
    },
])
def test_non_conversation_routes_pass_through_without_turns(route):
    context, repository, router, agent, notifications, queue = _harness(route)
    before = deepcopy(repository.conversations)

    result = _call(
        "op-control", context, repository, router, agent, notifications, queue,
        "control command",
    )

    assert result["kind"] == route["kind"]
    assert repository.conversations == before
    assert notifications.attempts == []
    assert agent.prompts == []
    assert "control command" not in repr(repository.route_records["op-control"])


def test_retry_route_uses_exact_number_and_command_identity_and_retry_work_replay_does_not_enqueue():
    context, repository, router, agent, notifications, queue = _harness({
        "kind": "retry", "dialog_number": 7,
    })
    before = deepcopy(repository.conversations)
    selection_before = deepcopy(repository.selection)

    result = _call(
        "accepted-command-operation", context, repository, router, agent,
        notifications, queue, "raw retry request",
    )

    assert result == {
        "status": "ok", "kind": "retry", "dialog_number": 7,
        "retry_operation_id": "retry-operation",
    }
    assert repository.retry_requests == [
        (context.context_id, 7, "accepted-command-operation"),
    ]
    assert repository.conversations == before
    assert repository.selection == selection_before


def test_rejected_numbered_retry_clears_candidate_without_registering_retry_operation():
    context, base, router, agent, notifications, queue = _harness({
        "kind": "retry", "dialog_number": 99,
    })
    repository = _EnvelopeRepository(context, base.conversations, base.events)
    repository.retry_accepted = False
    router.repository = repository

    result = _process(
        operation_id="accepted-command-operation", context=context,
        text="raw retry request", repository=repository, router=router,
        agent=agent, notifications=notifications, queue=queue,
        owner="worker", fence=3, now=datetime.now().astimezone(),
    )

    assert result == {"status": "not_found", "kind": "retry", "dialog_number": 99}
    assert repository.retry_requests == [
        (context.context_id, 99, "accepted-command-operation"),
    ]
    assert repository.retry_operations == []
    assert ("candidate_cleared", "accepted-command-operation") in repository.events
    assert repository.conversations == base.conversations
    assert repository.selection == base.selection
    assert agent.prompts == []
    assert notifications.attempts == []

    replay = _route("retry")
    replay.update({
        "status": "ok", "retry_request_operation_id": "accepted-command-operation",
    })
    context, repository, router, agent, notifications, queue = _harness(replay)
    before = deepcopy(repository.conversations)
    selection_before = deepcopy(repository.selection)
    replay_result = _call(
        "retry-work-operation", context, repository, router, agent,
        notifications, queue, "retry operation replay",
    )

    assert replay_result["status"] == "ok"
    assert replay_result["kind"] == "retry"
    assert repository.retry_requests == []
    assert repository.conversations == before
    assert repository.selection == selection_before


def test_successful_save_route_records_draft_target_without_command_text():
    context, repository, router, agent, notifications, queue = _harness({
        "status": "ok",
        "kind": "save",
        "confirmed": True,
        "target_draft_id": "draft-target",
        "text": "raw command envelope",
        "substantive_text": "private route payload",
    })

    result = _call(
        "op-save-route", context, repository, router, agent, notifications, queue,
        "raw command envelope",
    )

    assert result == {"status": "ok", "kind": "save", "confirmed": True}
    assert repository.route_records["op-save-route"] == {
        "status": "ok", "kind": "save", "target_draft_id": "draft-target",
    }
    assert "raw command envelope" not in repr(repository.route_records["op-save-route"])
    assert "private route payload" not in repr(repository.route_records["op-save-route"])


def test_route_message_notice_marker_does_not_bypass_durable_operation_barrier():
    route = _route("substantive", substantive_text="target request")
    route["selection_changed"] = True
    route["notice_sent"] = True
    context, repository, router, agent, notifications, queue = _harness(route)

    _call(
        "op-routed", context, repository, router, agent, notifications, queue,
        "raw route envelope",
    )

    assert len(notifications.attempts) == 1
    assert notifications.attempts[0]["body"] == "Starting to process dialog #2."
    assert repository.selection["active_conversation_id"] == "conversation:target"
    assert agent.prompts[0]["prompt"]["request"] == "target request"


def test_same_current_target_does_not_send_a_switch_notice():
    context, repository, router, agent, notifications, queue = _harness(
        _route("substantive", substantive_text="already here"),
    )
    context = replace(
        context, active_conversation_id="conversation:target",
        selected_dialog_number=2,
    )

    result = _call(
        "op-same-target", context, repository, router, agent, notifications, queue,
        "already here",
    )

    assert result["status"] == "completed"
    assert notifications.attempts == []


def test_three_unknown_notice_sends_release_head_and_same_operation_replay_is_suppressed():
    control = "open dialog 2"
    context, repository, router, agent, notifications, queue = _harness(
        _route("pure_switch"),
        outcomes=["retry", "retry", "delivery_unknown"],
    )
    before = repository.snapshot()

    first = _call("op-unknown", context, repository, router, agent, notifications, queue, control)
    router_call_count = len(router.calls)
    replay = _call("op-unknown", context, repository, router, agent, notifications, queue, control)

    assert first["status"] == replay["status"] == "switch_notice_undelivered"
    assert len(notifications.attempts) == 3
    assert len({row["notification_id"] for row in notifications.attempts}) == 1
    assert [row["operation_id"] for row in notifications.attempts] == ["op-unknown"] * 3
    assert queue.released == ["op-unknown"]
    assert router_call_count == len(router.calls) == 1
    assert repository.snapshot() == before
    assert agent.prompts == []
    assert all(row[0] != "selection" for row in repository.writes)
    assert not any(event[0] in {"selection", "append", "agent"} for event in repository.events)

    # A fresh explicit operation can retry selection with a new notice identity.
    retry = _call("op-retry", context, repository, router, agent, notifications, queue, control)
    assert retry["status"] in {"switched", "completed"}
    assert len(notifications.attempts) == 4
    assert notifications.attempts[-1]["notification_id"] != notifications.attempts[0]["notification_id"]


def test_busy_notice_releases_head_and_replay_resumes_pinned_route():
    context, repository, router, agent, notifications, queue = _harness(
        _route("pure_switch"), outcomes=["busy"],
    )
    first = _call("op-busy", context, repository, router, agent, notifications, queue, "open dialog 2")
    assert first["status"] == "switch_notice_pending"
    assert queue.released == ["op-busy"]
    assert repository.selection["active_conversation_id"] == "conversation:old"
    replay = _call("op-busy", context, repository, router, agent, notifications, queue, "open dialog 2")

    assert replay["status"] == "switched"
    assert [row["notification_id"] for row in notifications.attempts] == [
        "switch-op-busy", "switch-op-busy",
    ]
    assert len(router.calls) == 1
    assert queue.released == ["op-busy"]
    assert repository.selection["active_conversation_id"] == "conversation:target"
    assert agent.prompts == []


def test_late_completion_preserves_newer_focus_and_older_operation_pinned_target():
    context, repository, router, agent, notifications, queue = _harness(
        _route("substantive", substantive_text="pinned old operation"),
    )
    repository.conversations["conversation:newer"] = []
    repository.docs["doc-newer"] = {"saved": "newer-doc"}
    repository.search_index["doc-newer"] = ["newer-index"]
    old_history = deepcopy(repository.conversations["conversation:old"])
    target_history = deepcopy(repository.conversations["conversation:target"])
    saved_data = deepcopy((repository.docs, repository.search_index))
    newer_selection = []

    async def select_newer_during_old_agent_call():
        await notifications.send_switch_notice(
            operation_id="op-new",
            notification_id="notice-op-new",
            target_doc_id="doc-newer",
            dialog_number=9,
        )
        _publish_selection(
            repository, "op-new", conversation_id="conversation:newer", dialog_number=9,
        )
        newer_selection.append(deepcopy(repository.selection))

    agent.before_return = select_newer_during_old_agent_call
    result = _call(
        "op-old", context, repository, router, agent, notifications, queue,
        "old request",
    )

    assert result["status"] in {"completed", "processed"}
    assert agent.prompts[0]["target_doc_id"] == "doc-target"
    assert repository.route_records["op-old"]["conversation_id"] == "conversation:target"
    assert repository.conversations["conversation:target"][:len(target_history)] == target_history
    assert [(row["role"], row["parts"]) for row in repository.conversations["conversation:target"][len(target_history):]] == [
        ("user", [{"text": "pinned old operation"}]),
        ("model", [{"text": "synthetic answer"}]),
    ]
    assert repository.conversations["conversation:old"] == old_history
    assert repository.conversations["conversation:newer"] == []
    assert repository.selection == newer_selection[0]
    assert repository.selection["active_conversation_id"] == "conversation:newer"
    assert repository.selection["selected_dialog_number"] == 9
    assert repository.selection["selection_sequence"] > context.selection_sequence
    assert (repository.docs, repository.search_index) == saved_data
