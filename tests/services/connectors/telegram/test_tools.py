"""Contract tests for the Telegram adapter over the shared operation registry.

The Telegram adapter is introduced in T115. Imports stay inside test bodies so
this module remains collectable while that adapter is being implemented.
"""

from __future__ import annotations

import importlib
from copy import deepcopy

import pytest

from core.operations import registry
from services.mcp.app.mcp.tool_catalog import get_tool_catalog as get_mcp_tool_catalog


ALLOWED_TOOLS = {
    "query": {"query": "launch plans", "keywords": "Acme", "folder_path": ["Clients"],
              "date_from": "2025-01-01", "date_to": "2025-02-01"},
    "get_transcript": {"doc_id": "doc-1", "folder_path": ["Clients"], "date_from": "2025-01-01",
                        "date_to": "2025-02-01", "limit": 4},
    "list_recent_transcripts": {"folder_path": ["Clients"], "date_from": "2025-01-01",
                                 "date_to": "2025-02-01", "limit": 8, "offset": 2},
    "list_folders": {},
    "sync_status": {},
    "move_transcript": {"doc_id": "doc-1", "folder_path": ["Archive"]},
    "rename_transcript": {"doc_id": "doc-1", "new_title": "Planning"},
    "move_folder": {"folder_path": ["Clients"], "new_parent_path": ["Archive"]},
    "rename_folder": {"folder_path": ["Clients"], "new_name": "Customers"},
    "create_folder": {"name": "Projects", "parent_path": ["Clients"]},
    "add_fact": {"doc_id": "doc-1", "text": "Acme approved the plan"},
    "add_dialog": {"provider": "synthetic", "utterances": [{"speaker": "Alex", "text": "Hello"}]},
    "sync_changes": {"doc_id": "doc-1"},
}


def _adapter():
    """Resolve lazily so absent T115 implementation causes test failures, not collection errors."""
    return importlib.import_module("services.connectors.telegram.tools")


def _schema_from_registry():
    """Project the registry's advertised field names, JSON types, and required fields."""
    result = {}
    for name, operation in registry.OPERATIONS.items():
        properties = {}
        required = []
        for key, field in operation.arguments.items():
            json_type = "array" if field.kind in {"path", "utterances"} else field.kind
            if field.kind == "date":
                json_type = "string"
            properties[key] = {"type": json_type}
            if field.required:
                required.append(key)
        schema = {"properties": properties}
        if required:
            schema["required"] = required
        result[name] = schema
    return result


def _without_descriptions(value):
    """Drop descriptive text while retaining every functional schema constraint."""
    if isinstance(value, dict):
        return {
            key: _without_descriptions(item)
            for key, item in value.items()
            if key != "description"
        }
    if isinstance(value, list):
        return [_without_descriptions(item) for item in value]
    return value


def test_catalog_has_exactly_the_shared_registry_tools_and_schemas():
    tools = _adapter().get_tool_catalog()["tools"]
    names = [tool["name"] for tool in tools]
    assert set(names) == set(registry.OPERATIONS)
    assert len(names) == len(set(names)) == 13
    assert "import_all_transcripts" not in names
    assert "delete_transcript" not in names

    expected = _schema_from_registry()
    actual = {tool["name"]: tool["inputSchema"] for tool in tools}
    assert {
        name: {key: prop["type"] for key, prop in schema["properties"].items()}
        | ({"required": schema["required"]} if "required" in schema else {})
        for name, schema in actual.items()
    } == {
        name: {
            key: prop["type"] for key, prop in schema["properties"].items()
        } | ({"required": schema["required"]} if "required" in schema else {})
        for name, schema in expected.items()
    }

    mcp_tools = {
        tool["name"]: tool["inputSchema"]
        for tool in get_mcp_tool_catalog()["tools"]
        if tool["name"] != "import_all_transcripts"
    }
    # MCP's legacy sync_status schema exposes operation_id, which is outside
    # the shared registry contract and must not expand the Telegram surface.
    mcp_tools["sync_status"]["properties"].pop("operation_id", None)
    assert {name: _without_descriptions(schema) for name, schema in actual.items()} == {
        name: _without_descriptions(schema) for name, schema in mcp_tools.items()
    }


def test_all_tools_are_exposed_and_invoke_without_an_active_draft(monkeypatch):
    tool_names = {tool["name"] for tool in _adapter().get_tool_catalog()["tools"]}
    assert tool_names == set(registry.OPERATIONS) == set(ALLOWED_TOOLS)

    calls = []

    def dispatch(name, arguments, *, context=None):
        calls.append((name, deepcopy(arguments), context))
        return {"ok": True, "operation": name}

    monkeypatch.setattr(registry, "invoke_operation", dispatch)
    adapter = _adapter()
    # Invoke every tool without supplying active-draft state. Typed write-context
    # enforcement is covered by the core registry tests; this isolates adapter routing.
    for name, arguments in ALLOWED_TOOLS.items():
        assert adapter.invoke_tool(name, deepcopy(arguments)) == {"ok": True, "operation": name}

    assert [call[0] for call in calls] == list(registry.OPERATIONS)
    assert {name: arguments for name, arguments, _ in calls} == {
        name: ALLOWED_TOOLS[name] for name in registry.OPERATIONS
    }


def test_last_dialog_uses_context_last_doc_id_for_direct_transcript_lookup(monkeypatch):
    calls = []

    def dispatch(name, arguments, *, context=None):
        calls.append((name, deepcopy(arguments), context))
        return {"segments": [{"speaker": "Alex", "text": "Most recent dialog"}]}

    monkeypatch.setattr(registry, "invoke_operation", dispatch)
    result = _adapter().invoke_tool("get_transcript", {}, context={"last_doc_id": "doc-last"})

    assert result == {"segments": [{"speaker": "Alex", "text": "Most recent dialog"}]}
    assert calls == [("get_transcript", {"doc_id": "doc-last"}, None)]
    assert all(name != "query" for name, _, _ in calls)


@pytest.mark.parametrize(("name", "arguments"), [
    ("move_transcript", {"folder_path": ["Archive"]}),
    ("rename_transcript", {"new_title": "Renamed"}),
])
def test_last_dialog_is_used_for_direct_move_and_rename(monkeypatch, name, arguments):
    calls = []

    def dispatch(operation, supplied, *, context=None):
        calls.append((operation, deepcopy(supplied), context))
        return {"ok": True}

    monkeypatch.setattr(registry, "invoke_operation", dispatch)
    result = _adapter().invoke_tool(name, arguments, context={"last_doc_id": "doc-last"})

    assert result == {"ok": True}
    assert calls == [(name, {**arguments, "doc_id": "doc-last"}, None)]


def test_read_preserves_provider_neutral_filters_segments_and_truncation_metadata(monkeypatch):
    raw = (
        [
            {"provider": "meet", "doc_id": "meet-7", "speaker": "Mina", "text": "Original words"},
            {"provider": "zoom", "doc_id": "zoom-3", "speaker": "Lee", "text": "More words"},
        ],
        {"truncated": True, "total_matches": 9, "returned_segments": 2},
    )
    calls = []

    def dispatch(name, arguments, *, context=None):
        calls.append((name, deepcopy(arguments), context))
        return deepcopy(raw)

    monkeypatch.setattr(registry, "invoke_operation", dispatch)
    arguments = {"folder_path": ["Clients", "Acme"], "date_from": "2025-01-01",
                 "date_to": "2025-01-31", "limit": 2}
    result = _adapter().invoke_tool("get_transcript", arguments)

    assert calls == [("get_transcript", arguments, None)]
    assert result == raw
    assert result[0] == raw[0]
    assert result[1] == {"truncated": True, "total_matches": 9, "returned_segments": 2}


def test_links_are_omitted_by_default_and_included_only_when_requested(monkeypatch):
    raw = {"doc_id": "doc-1", "title": "Planning", "link": "https://docs.invalid/doc-1"}

    def dispatch(name, arguments, *, context=None):
        return deepcopy(raw)

    monkeypatch.setattr(registry, "invoke_operation", dispatch)
    adapter = _adapter()
    default_result = adapter.invoke_tool("get_transcript", {"doc_id": "doc-1"})
    requested_result = adapter.invoke_tool(
        "get_transcript", {"doc_id": "doc-1"}, context={"include_link": True}
    )

    assert "link" not in default_result
    assert requested_result["link"] == raw["link"]


def test_structured_transcript_link_request_survives_missing_execution_context():
    from services.connectors.telegram.agent import TelegramAgent

    class Ledger:
        def load_turn(self, *_args):
            return None

        def save_turn(self, *_args, **_kwargs):
            pass

        def save_tool_result(self, *_args):
            pass

    received_contexts = []

    class Executor:
        def execute(self, **_kwargs):
            return {"link": "https://docs.invalid/doc-1"}

    def executor_factory(_target_doc_id, _delivery_id, context):
        received_contexts.append(context)
        return Executor()

    agent = TelegramAgent(model=None, ledger=Ledger(), tool_executor=Executor())
    result = agent.dispatch_tool_choice(
        name="get_transcript", arguments={"doc_id": "doc-1"}, target_doc_id="doc-1",
        conversation_id="conversation-1", delivery_id="delivery-1",
        executor_factory=executor_factory, link_requested=True,
    )

    assert received_contexts == [{"include_link": True}]
    assert result == {"link": "https://docs.invalid/doc-1"}


def test_sync_status_preserves_reported_status_and_path(monkeypatch):
    reported = {
        "status": "synced",
        "path": ["Clients", "Acme"],
        "last_sync_at": "2025-01-31T10:30:00Z",
    }
    calls = []

    def dispatch(name, arguments, *, context=None):
        calls.append((name, arguments, context))
        return deepcopy(reported)

    monkeypatch.setattr(registry, "invoke_operation", dispatch)
    result = _adapter().invoke_tool("sync_status", {})

    assert calls == [("sync_status", {}, None)]
    assert result == reported


@pytest.mark.parametrize("name", [
    "import_all_transcripts", "delete_transcript", "delete_document", "import_transcript",
    "getattr", "read_file", "storage_get", "arbitrary_module_call",
])
def test_forbidden_or_arbitrary_operations_are_rejected(name):
    adapter = _adapter()
    with pytest.raises(registry.OperationValidationError):
        adapter.invoke_tool(name, {})


@pytest.mark.parametrize("arguments", [
    {"query": "find this", "module": "os", "function": "system"},
    {"query": "find this", "storage_path": ["private"]},
    {"query": "find this", "create_document": True},
])
def test_extra_capabilities_and_unrequested_document_creation_are_rejected(arguments):
    adapter = _adapter()
    with pytest.raises(registry.OperationValidationError):
        adapter.invoke_tool("query", arguments)


def test_search_does_not_create_a_document_without_explicit_add_dialog(monkeypatch):
    calls = []

    def dispatch(name, arguments, *, context=None):
        calls.append(name)
        return []

    monkeypatch.setattr(registry, "invoke_operation", dispatch)
    _adapter().invoke_tool("query", {"query": "meeting notes"})
    assert calls == ["query"]
    assert "add_dialog" not in calls
    assert "create_document" not in registry.OPERATIONS
