"""Offline contract tests for structured Telegram dialog routing.

The routing seam is introduced by a later implementation task. Keep its import
inside ``_route`` so this module collects before that module exists.
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from core.operations.registry import OPERATIONS
from services.connectors.telegram.models import TelegramContext


pytestmark = pytest.mark.offline


def _route(**kwargs):
    """Load the future seam lazily and execute its async public entry point."""
    import asyncio
    from services.connectors.telegram.sessions import route_message

    return asyncio.run(route_message(**kwargs))


class FakeModel:
    def __init__(self, response=None, *, error=None):
        self.response = deepcopy(response)
        self.error = error
        self.calls = []

    async def route(self, *, text, candidates):
        self.calls.append({"text": text, "candidates": tuple(candidates)})
        if self.error is not None:
            raise self.error
        return deepcopy(self.response)


class FakeStore:
    """In-memory dialog bindings and search-session records scoped by context."""

    def __init__(self, docs):
        self.docs = {
            doc_id: {"conversation_id": f"conversation:{doc_id}", **doc}
            for doc_id, doc in docs.items()
        }
        self.document_bindings = {}
        self.dialogs = {}
        self.search_sessions = []
        self.get_calls = []
        self.created_docs = []
        self.created_drafts = []

    def existing_dialog_candidates(self):
        return tuple(self.docs)

    def get_doc(self, doc_id):
        self.get_calls.append(doc_id)
        return self.docs.get(doc_id)

    def bind_existing_numbers(self, context_id, numbers):
        for doc_id, number in numbers.items():
            self.document_bindings[(context_id, doc_id)] = number
            self.dialogs.setdefault(context_id, {})[number] = doc_id

    def get_or_assign_dialog_number(self, context_id, doc_id):
        binding_key = (context_id, doc_id)
        if binding_key not in self.document_bindings:
            local_numbers = self.dialogs.setdefault(context_id, {})
            number = max(local_numbers, default=0) + 1
            local_numbers[number] = doc_id
            self.document_bindings[binding_key] = number
        return self.document_bindings[binding_key]

    def create_search_session(self, *, context_id, query, result_refs, error=None):
        session_id = f"search-{len(self.search_sessions) + 1}"
        self.search_sessions.append({
            "session_id": session_id,
            "context_id": context_id,
            "query": query,
            "result_refs": list(result_refs),
            "error": error,
        })
        return session_id

    def create_doc(self, *args, **kwargs):
        self.created_docs.append((args, kwargs))

    def create_draft(self, *args, **kwargs):
        self.created_drafts.append((args, kwargs))


class FakeSearch:
    def __init__(self, results=(), *, error=None):
        self.results = tuple(results)
        self.error = error
        self.calls = []

    async def search(self, query):
        self.calls.append(query)
        if self.error is not None:
            raise self.error
        return self.results


class FakeNotices:
    def __init__(self):
        self.sent = []

    def send_switch_notice(self, *, doc_id, dialog_number):
        self.sent.append((doc_id, dialog_number))


class FakeActions:
    ALLOWED = frozenset(OPERATIONS)

    def __init__(self):
        self.calls = []

    async def execute(self, *, name, arguments, target_doc_id, link_requested=False):
        if name not in self.ALLOWED:
            raise AssertionError(f"operation is outside the shared registry: {name}")
        self.calls.append((name, deepcopy(arguments), target_doc_id, link_requested))
        return {"ok": True, "operation": name}


def _decision(kind="substantive", **overrides):
    value = {
        "kind": kind,
        "target_ref": None,
        "dialog_number": None,
        "result_refs": [],
        "substantive_text": None,
        "tool_choice": None,
        "certainty": "certain",
    }
    value.update(overrides)
    return value


def _make_context(*, current=None, selected_number=None, last=None):
    current_doc = current
    active_conversation_id = (
        None if current_doc is None else f"conversation:{current_doc}"
    )
    return TelegramContext(
        chat_id=42,
        thread_id=7,
        active_conversation_id=active_conversation_id,
        selected_dialog_number=selected_number,
        last_doc_id=last,
    )


def _harness(*, decision=None, docs=None, current=None, last=None, numbers=None,
             search_results=(), search_error=None, model_error=None):
    context = _make_context(
        current=current,
        selected_number=(numbers or {}).get(current),
        last=last,
    )
    store = FakeStore(docs or {})
    store.bind_existing_numbers(context.context_id, numbers or {})
    model = FakeModel(decision, error=model_error)
    search = FakeSearch(search_results, error=search_error)
    notices = FakeNotices()
    actions = FakeActions()
    return context, store, model, search, notices, actions


def _call(text, *, context, store, model, search, notices, actions):
    return _route(
        text=text,
        context=context,
        model=model,
        store=store,
        search=search,
        notices=notices,
        actions=actions,
    )


def test_current_topic_continues_without_notice_and_automatic_switch_uses_target_number():
    docs = {"doc-current": {"title": "Current"}, "doc-other": {"title": "Other"}}
    context, store, model, search, notices, actions = _harness(
        decision=_decision("substantive", target_ref="doc-current",
                           substantive_text="continue this topic"),
        docs=docs, current="doc-current", numbers={"doc-current": 2},
    )
    continued = _call("same topic", context=context, store=store, model=model,
                      search=search, notices=notices, actions=actions)
    assert continued["target_doc_id"] == "doc-current"
    assert notices.sent == []
    assert model.calls[0]["candidates"] == ("doc-current", "doc-other")

    model.response = _decision("pure_switch", target_ref="doc-other")
    switched = _call("unrelated thought", context=context, store=store, model=model,
                     search=search, notices=notices, actions=actions)
    assert switched["target_doc_id"] == "doc-other"
    expected_number = store.document_bindings[(context.context_id, "doc-other")]
    assert switched["dialog_number"] == expected_number
    assert notices.sent == [("doc-other", expected_number)]


def test_substantive_semantic_switch_preserves_question_and_notifies_target_number():
    question = "How would this approach differ for a small research team?"
    context, store, model, search, notices, actions = _harness(
        decision=_decision(
            "substantive",
            target_ref="doc-research",
            substantive_text=question,
        ),
        docs={"doc-current": {}, "doc-research": {}},
        current="doc-current",
        numbers={"doc-current": 3},
    )

    result = _call("compare it with the research discussion", context=context,
                   store=store, model=model, search=search, notices=notices,
                   actions=actions)

    expected_number = store.document_bindings[(context.context_id, "doc-research")]
    assert result["target_doc_id"] == "doc-research"
    assert result["dialog_number"] == expected_number
    assert result["substantive_text"] == question
    assert notices.sent == [("doc-research", expected_number)]


def test_explicit_dialog_number_wins_over_conflicting_semantic_target():
    context, store, model, search, notices, actions = _harness(
        decision=_decision("pure_switch", target_ref="doc-b", dialog_number=4),
        docs={"doc-a": {}, "doc-b": {}}, current="doc-b",
        numbers={"doc-a": 4, "doc-b": 7},
    )
    result = _call("open dialog 4", context=context, store=store, model=model,
                   search=search, notices=notices, actions=actions)
    assert result["target_doc_id"] == "doc-a"
    assert result["dialog_number"] == 4
    assert notices.sent == [("doc-a", 4)]


def test_unknown_explicit_number_is_not_found_without_any_target_fallback():
    context, store, model, search, notices, actions = _harness(
        decision=_decision("pure_switch", target_ref="doc-semantic", dialog_number=99),
        docs={"doc-current": {}, "doc-last": {}, "doc-semantic": {}},
        current="doc-current", last="doc-last", numbers={"doc-current": 1, "doc-last": 2},
    )
    result = _call("open dialog 99", context=context, store=store, model=model,
                   search=search, notices=notices, actions=actions)
    assert result["status"] == "not_found"
    assert result.get("target_doc_id") is None
    assert notices.sent == []
    assert store.get_calls == []


def test_last_target_uses_context_id_without_semantic_search():
    context, store, model, search, notices, actions = _harness(
        decision=_decision("substantive", target_ref="last", substantive_text="continue"),
        docs={"doc-last": {}, "doc-decoy": {}}, current="doc-decoy", last="doc-last",
        numbers={"doc-last": 6, "doc-decoy": 3},
    )
    result = _call("continue the last dialog", context=context, store=store, model=model,
                   search=search, notices=notices, actions=actions)
    assert result["target_doc_id"] == "doc-last"
    assert search.calls == []


def test_explicit_search_creates_only_search_session_then_selects_existing_result():
    context, store, model, search, notices, actions = _harness(
        decision=_decision("search", result_refs=["doc-imported"]),
        docs={"doc-current": {}, "doc-imported": {"title": "Found"}},
        current="doc-current", numbers={"doc-current": 1},
        search_results=["doc-imported"],
    )
    initial_selection = (context.active_conversation_id, context.selected_dialog_number)
    searched = _call("search for this topic", context=context, store=store, model=model,
                     search=search, notices=notices, actions=actions)
    assert searched["kind"] == "search"
    assert searched["result_refs"] == ["doc-imported"]
    assert searched["search_session_id"] == store.search_sessions[0]["session_id"]
    assert store.search_sessions[0]["context_id"] == context.context_id
    assert (context.active_conversation_id, context.selected_dialog_number) == initial_selection
    assert notices.sent == []
    assert store.created_docs == []
    assert store.created_drafts == []

    model.response = _decision("pure_switch", target_ref="doc-imported")
    first = _call("open that result", context=context, store=store, model=model,
                  search=search, notices=notices, actions=actions)
    assigned_number = first["dialog_number"]
    assert assigned_number == store.document_bindings[(context.context_id, "doc-imported")]
    assert store.created_docs == []
    assert store.created_drafts == []
    assert notices.sent == [("doc-imported", assigned_number)]

    # A new immutable TelegramContext for the same chat/thread still resolves
    # the imported Doc to its stored local number and avoids a repeated notice.
    recreated = _make_context(
        current="doc-imported", selected_number=assigned_number, last="doc-imported",
    )
    assert recreated.context_id == context.context_id
    model.response = _decision("substantive", target_ref="doc-imported",
                               substantive_text="continue here")
    repeated = _call("keep going", context=recreated, store=store, model=model,
                     search=search, notices=notices, actions=actions)
    assert repeated["dialog_number"] == assigned_number
    assert len([key for key in store.document_bindings
                if key == (context.context_id, "doc-imported")]) == 1
    assert notices.sent == [("doc-imported", assigned_number)]


@pytest.mark.parametrize(("search_results", "search_error", "expected_refs"), [
    (["doc-a"], None, ["doc-a"]),
    ([], None, []),
    ([], RuntimeError("search unavailable"), []),
])
def test_search_results_empty_results_and_errors_remain_search_results(
    search_results, search_error, expected_refs,
):
    context, store, model, search, notices, actions = _harness(
        decision=_decision("search", result_refs=[]), docs={"doc-a": {}},
        search_results=search_results, search_error=search_error,
    )
    result = _call("look for a matching dialog", context=context, store=store, model=model,
                   search=search, notices=notices, actions=actions)
    assert result["kind"] == "search"
    assert result["result_refs"] == expected_refs
    assert len(store.search_sessions) == 1
    assert (store.search_sessions[0]["error"] is None) == (search_error is None)
    assert notices.sent == []
    assert store.created_docs == []
    assert store.created_drafts == []


@pytest.mark.parametrize(("phrase", "decision", "expected_kind", "expected_operation"), [
    ("save this", _decision("save", target_ref="doc-a"), "save", None),
    ("\u0441\u043e\u0445\u0440\u0430\u043d\u0438 \u044d\u0442\u043e",
     _decision("save", target_ref="doc-a"), "save", None),
    ("please retry this", _decision("retry", target_ref="doc-a"), "retry", None),
    ("\u043f\u043e\u0432\u0442\u043e\u0440\u0438 \u043e\u043f\u0435\u0440\u0430\u0446\u0438\u044e",
     _decision("retry", target_ref="doc-a"), "retry", None),
    ("switch to another dialog", _decision("pure_switch", target_ref="doc-b"),
     "pure_switch", None),
    ("\u043f\u0435\u0440\u0435\u0439\u0434\u0438 \u043a \u0434\u0440\u0443\u0433\u043e\u043c\u0443 \u0434\u0438\u0430\u043b\u043e\u0433\u0443",
     _decision("pure_switch", target_ref="doc-b"), "pure_switch", None),
    ("find a related dialog", _decision("search", result_refs=[]), "search", None),
    ("\u043d\u0430\u0439\u0434\u0438 \u0441\u0432\u044f\u0437\u0430\u043d\u043d\u044b\u0439 \u0434\u0438\u0430\u043b\u043e\u0433",
     _decision("search", result_refs=[]), "search", None),
    ("continue the last dialog", _decision("substantive", target_ref="last",
                                            substantive_text="continue"),
     "substantive", None),
    ("\u043f\u0440\u043e\u0434\u043e\u043b\u0436\u0438 \u043f\u043e\u0441\u043b\u0435\u0434\u043d\u0438\u0439 \u0434\u0438\u0430\u043b\u043e\u0433",
     _decision("substantive", target_ref="last", substantive_text="continue"),
     "substantive", None),
    ("save nothing; the quoted text says 'retry'", _decision(
        "substantive", target_ref="doc-a", substantive_text="discuss the quote",
    ), "substantive", None),
    ("could you take care of this?", _decision("save", target_ref="doc-a"), "save", None),
])
def test_structured_intents_determine_outcome_across_languages(
    phrase, decision, expected_kind, expected_operation,
):
    context, store, model, search, notices, actions = _harness(
        decision=decision, docs={"doc-a": {}, "doc-b": {}}, current="doc-a",
        last="doc-a", numbers={"doc-a": 1, "doc-b": 2},
    )
    result = _call(phrase, context=context, store=store, model=model, search=search,
                   notices=notices, actions=actions)
    assert model.calls[0]["text"] == phrase
    assert result["kind"] == expected_kind
    assert [call[0] for call in actions.calls] == (
        [] if expected_operation is None else [expected_operation]
    )


_OPERATION_CASES = [
    ("query", {"query": "climate"}, "find climate transcripts",
     "\u043d\u0430\u0439\u0434\u0438 \u0442\u0440\u0430\u043d\u0441\u043a\u0440\u0438\u043f\u0442\u044b \u043e \u043a\u043b\u0438\u043c\u0430\u0442\u0435"),
    ("get_transcript", {"doc_id": "doc-a"}, "read this transcript",
     "\u043f\u0440\u043e\u0447\u0438\u0442\u0430\u0439 \u044d\u0442\u043e\u0442 \u0442\u0440\u0430\u043d\u0441\u043a\u0440\u0438\u043f\u0442"),
    ("list_recent_transcripts", {}, "show recent transcripts",
     "\u043f\u043e\u043a\u0430\u0436\u0438 \u043d\u0435\u0434\u0430\u0432\u043d\u0438\u0435 \u0442\u0440\u0430\u043d\u0441\u043a\u0440\u0438\u043f\u0442\u044b"),
    ("list_folders", {}, "list my folders",
     "\u043f\u0435\u0440\u0435\u0447\u0438\u0441\u043b\u0438 \u043c\u043e\u0438 \u043f\u0430\u043f\u043a\u0438"),
    ("sync_status", {}, "check sync status",
     "\u043f\u0440\u043e\u0432\u0435\u0440\u044c \u0441\u0442\u0430\u0442\u0443\u0441 \u0441\u0438\u043d\u0445\u0440\u043e\u043d\u0438\u0437\u0430\u0446\u0438\u0438"),
    ("move_transcript", {"doc_id": "doc-a", "folder_path": ["Work"]},
     "move this transcript to Work", "\u043f\u0435\u0440\u0435\u043c\u0435\u0441\u0442\u0438 \u044d\u0442\u043e\u0442 \u0442\u0440\u0430\u043d\u0441\u043a\u0440\u0438\u043f\u0442 \u0432 Work"),
    ("rename_transcript", {"doc_id": "doc-a", "new_title": "New title"},
     "rename this transcript New title", "\u043f\u0435\u0440\u0435\u0438\u043c\u0435\u043d\u0443\u0439 \u044d\u0442\u043e\u0442 \u0442\u0440\u0430\u043d\u0441\u043a\u0440\u0438\u043f\u0442 New title"),
    ("move_folder", {"folder_path": ["Old"], "new_parent_path": ["Archive"]},
     "move folder Old into Archive", "\u043f\u0435\u0440\u0435\u043c\u0435\u0441\u0442\u0438 \u043f\u0430\u043f\u043a\u0443 Old \u0432 Archive"),
    ("rename_folder", {"folder_path": ["Old"], "new_name": "New"},
     "rename folder Old to New", "\u043f\u0435\u0440\u0435\u0438\u043c\u0435\u043d\u0443\u0439 \u043f\u0430\u043f\u043a\u0443 Old \u0432 New"),
    ("create_folder", {"name": "Research"}, "create a Research folder",
     "\u0441\u043e\u0437\u0434\u0430\u0439 \u043f\u0430\u043f\u043a\u0443 Research"),
    ("add_fact", {"doc_id": "doc-a", "text": "The meeting is Friday."},
     "add this fact to the transcript", "\u0434\u043e\u0431\u0430\u0432\u044c \u044d\u0442\u043e\u0442 \u0444\u0430\u043a\u0442 \u0432 \u0442\u0440\u0430\u043d\u0441\u043a\u0440\u0438\u043f\u0442"),
    ("add_dialog", {"provider": "test", "utterances": [
        {"speaker": "Ada", "text": "Hello."},
    ]}, "add this dialog", "\u0434\u043e\u0431\u0430\u0432\u044c \u044d\u0442\u043e\u0442 \u0434\u0438\u0430\u043b\u043e\u0433"),
    ("sync_changes", {"doc_id": "doc-a"}, "sync changes for this transcript",
     "\u0441\u0438\u043d\u0445\u0440\u043e\u043d\u0438\u0437\u0438\u0440\u0443\u0439 \u0438\u0437\u043c\u0435\u043d\u0435\u043d\u0438\u044f \u044d\u0442\u043e\u0433\u043e \u0442\u0440\u0430\u043d\u0441\u043a\u0440\u0438\u043f\u0442\u0430"),
]


@pytest.mark.parametrize(("operation", "arguments", "english", "russian"), _OPERATION_CASES)
@pytest.mark.parametrize("language_index", (2, 3), ids=("english", "russian"))
def test_all_shared_operations_are_selected_from_structured_results(
    operation, arguments, english, russian, language_index,
):
    phrase = (english, russian)[language_index - 2]
    decision = _decision(
        "substantive",
        target_ref="doc-a",
        substantive_text="perform the requested operation",
        tool_choice={
            "name": operation,
            "arguments": arguments,
        },
    )
    context, store, model, search, notices, actions = _harness(
        decision=decision, docs={"doc-a": {}}, current="doc-a", numbers={"doc-a": 1},
    )
    result = _call(phrase, context=context, store=store, model=model, search=search,
                   notices=notices, actions=actions)
    assert operation in OPERATIONS
    assert model.calls[0]["text"] == phrase
    assert result["kind"] == "substantive"
    assert [call[0] for call in actions.calls] == [operation]
    assert actions.calls[0][1] == arguments
    assert actions.calls[0][3] is False


@pytest.mark.parametrize(("phrase", "link_phrase"), [
    ("link this transcript", "link_requested"),
    ("\u0441\u0432\u044f\u0436\u0438 \u044d\u0442\u043e\u0442 \u0442\u0440\u0430\u043d\u0441\u043a\u0440\u0438\u043f\u0442", "link_requested"),
])
def test_link_is_a_structured_flag_around_get_transcript(phrase, link_phrase):
    decision = _decision(
        "substantive",
        target_ref="doc-a",
        substantive_text="link the transcript",
        tool_choice={
            "name": "get_transcript",
            "arguments": {"doc_id": "doc-a"},
            link_phrase: True,
        },
    )
    context, store, model, search, notices, actions = _harness(
        decision=decision, docs={"doc-a": {}}, current="doc-a", numbers={"doc-a": 1},
    )
    result = _call(phrase, context=context, store=store, model=model, search=search,
                   notices=notices, actions=actions)
    assert result["kind"] == "substantive"
    assert [call[0] for call in actions.calls] == ["get_transcript"]
    assert actions.calls[0][3] is True


def test_identical_wording_can_produce_different_results_from_different_structured_routes():
    phrase = "save this"
    harness = _harness(
        decision=_decision("substantive", target_ref="doc-a", substantive_text="discuss wording"),
        docs={"doc-a": {}}, current="doc-a", numbers={"doc-a": 1},
    )
    context, store, model, search, notices, actions = harness
    ignored = _call(phrase, context=context, store=store, model=model, search=search,
                    notices=notices, actions=actions)
    assert ignored["kind"] == "substantive"
    assert actions.calls == []

    model.response = _decision("save", target_ref="doc-a")
    saved = _call(phrase, context=context, store=store, model=model, search=search,
                  notices=notices, actions=actions)
    assert saved["kind"] == "save"
    assert actions.calls == []


@pytest.mark.parametrize(("response", "model_error"), [
    (None, RuntimeError("model unavailable")),
    ({"kind": "freeform", "target_ref": "doc-a"}, None),
    (_decision("substantive", target_ref="doc-a", substantive_text="do it",
               tool_choice={"name": "delete_everything", "arguments": {}}), None),
    (_decision("search", certainty="uncertain", result_refs=["doc-a"]), None),
    ({"kind": "save", "target_ref": "doc-a", "tool_choice": "save"}, None),
])
def test_invalid_or_uncertain_route_fails_closed_without_side_effects(response, model_error):
    context, store, model, search, notices, actions = _harness(
        decision=response, model_error=model_error,
        docs={"doc-a": {}}, current="doc-a", last="doc-a", numbers={"doc-a": 1},
        search_results=["doc-a"],
    )
    result = _call("please save this or search for a result", context=context, store=store,
                   model=model, search=search, notices=notices, actions=actions)
    assert result["status"] in {"invalid_route", "uncertain", "error"}
    assert search.calls == []
    assert store.search_sessions == []
    assert store.created_docs == []
    assert store.created_drafts == []
    assert actions.calls == []
    assert notices.sent == []
