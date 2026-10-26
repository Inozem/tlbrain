"""Offline history isolation and switch notice barrier contracts."""

from __future__ import annotations

import asyncio
from copy import deepcopy

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

    def snapshot(self):
        return deepcopy((self.selection, self.conversations, self.docs, self.search_index))

    def route_snapshot(self):
        return deepcopy((self.conversations, self.docs, self.search_index))

    def get_history(self, conversation_id):
        return self.conversations[conversation_id]

    def persist_route(self, operation_id, route):
        self.route_records[operation_id] = deepcopy(route)
        self.writes.append(("route", operation_id, deepcopy(route)))

    def update_selection(self, operation_id, *, conversation_id, dialog_number):
        self.selection = {
            "active_conversation_id": conversation_id,
            "selected_dialog_number": dialog_number,
            "selection_sequence": self.selection["selection_sequence"] + 1,
        }
        self.writes.append(("selection", operation_id, deepcopy(self.selection)))
        self.events.append(("selection", operation_id, deepcopy(self.selection)))

    def append_turn(self, conversation_id, *, role, content):
        turn = {"role": role, "content": content}
        self.conversations[conversation_id].append(turn)
        self.writes.append(("turn", conversation_id, deepcopy(turn)))
        self.events.append(("append", conversation_id, deepcopy(turn)))

    def save_doc(self, target_doc_id, content):
        self.docs[target_doc_id] = content
        self.writes.append(("doc", target_doc_id, content))

    def update_search_index(self, target_doc_id, content):
        self.search_index[target_doc_id] = [content]
        self.writes.append(("index", target_doc_id, content))


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
        self, *, operation_id, notification_id, target_doc_id, dialog_number,
    ):
        self.attempts.append({
            "operation_id": operation_id,
            "notification_id": notification_id,
            "target_doc_id": target_doc_id,
            "dialog_number": dialog_number,
        })
        self.events.append(("notice_attempt", operation_id, notification_id))
        outcome = self.outcomes.pop(0) if self.outcomes else "confirmed"
        if isinstance(outcome, Exception):
            raise outcome
        if outcome == "confirmed":
            self.events.append(("notice_confirmed", operation_id, notification_id))
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
            {"role": "user", "content": "old user turn"},
            {"role": "assistant", "content": "old answer"},
        ],
        "conversation:target": [
            {"role": "user", "content": "target user turn"},
            {"role": "assistant", "content": "target answer"},
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
    )


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
    assert agent.prompts == []
    assert [row[0] for row in repository.writes] == ["route", "selection"]
    assert notifications.attempts[0]["target_doc_id"] == "doc-target"
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
    assert target[len(before_target):] == [
        {"role": "user", "content": substantive},
        {"role": "assistant", "content": "synthetic answer"},
    ]
    assert not _contains(target, "open dialog 2")
    assert not _contains(target, switch_notice)
    prompt = agent.prompts[0]["prompt"]
    assert not _contains(prompt, "open dialog 2")
    assert not _contains(prompt, switch_notice)
    assert not _contains(prompt, "old user turn")
    assert not _contains(prompt, "old answer")
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
    assert repository.conversations["conversation:target"][-2] == {
        "role": "user", "content": question,
    }
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


def test_three_unknown_notice_sends_release_head_and_same_operation_replay_is_suppressed():
    control = "open dialog 2"
    context, repository, router, agent, notifications, queue = _harness(
        _route("pure_switch"),
        outcomes=[_UnknownDelivery(), _UnknownDelivery(), _UnknownDelivery()],
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
        repository.update_selection(
            "op-new", conversation_id="conversation:newer", dialog_number=9,
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
    assert repository.conversations["conversation:target"] == target_history + [
        {"role": "user", "content": "pinned old operation"},
        {"role": "assistant", "content": "synthetic answer"},
    ]
    assert repository.conversations["conversation:old"] == old_history
    assert repository.conversations["conversation:newer"] == []
    assert repository.selection == newer_selection[0]
    assert repository.selection["active_conversation_id"] == "conversation:newer"
    assert repository.selection["selected_dialog_number"] == 9
    assert repository.selection["selection_sequence"] > context.selection_sequence
    assert (repository.docs, repository.search_index) == saved_data
