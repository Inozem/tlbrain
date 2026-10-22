from __future__ import annotations

from types import MappingProxyType

import pytest

from core.operations import read, registry, write


EXPECTED = {
    "query", "get_transcript", "list_recent_transcripts", "list_folders", "sync_status",
    "move_transcript", "rename_transcript", "move_folder", "rename_folder", "create_folder",
    "add_fact", "add_dialog", "sync_changes",
}


def test_registry_is_exact_static_immutable_allowlist():
    assert set(registry.OPERATIONS) == EXPECTED
    assert isinstance(registry.OPERATIONS, MappingProxyType)
    with pytest.raises(TypeError):
        registry.OPERATIONS["arbitrary"] = registry.OPERATIONS["query"]


@pytest.mark.parametrize("arguments", [
    {"query": "secret", "module": "os"},
    {"query": 3},
    {"query": True},
    {"query": "  "},
    {"query": "ok", "folder_path": ["A", " "]},
    {"query": "ok", "limit": 1},
])
def test_validation_rejects_bad_fields_and_strict_types_without_echoing_values(arguments):
    with pytest.raises(registry.OperationValidationError) as raised:
        registry.validate_arguments("query", arguments)
    assert "secret" not in repr(raised.value)
    assert "os" not in repr(raised.value)
    assert "query" not in repr(raised.value) or "field query" in str(raised.value)


@pytest.mark.parametrize("arguments", [
    {"doc_id": "d", "folder_path": []},
    {"doc_id": "d", "folder_path": ["a"], "context": "spoofed"},
    {"doc_id": "d", "folder_path": ["a"], "operation_id": "spoofed"},
    {"doc_id": "d", "folder_path": ["a"], "client": "spoofed"},
])
def test_write_validation_rejects_empty_target_and_internal_control_fields(arguments):
    with pytest.raises(registry.OperationValidationError):
        registry.validate_arguments("move_transcript", arguments)


@pytest.mark.parametrize(("name", "arguments"), [
    ("query", {"query": "q", "folder_path": "A"}),
    ("get_transcript", {"limit": True}),
    ("get_transcript", {"limit": 0}),
    ("list_recent_transcripts", {"offset": -1}),
    ("list_recent_transcripts", {"limit": 2.0}),
    ("move_folder", {"folder_path": ["A"], "new_parent_path": [""]}),
    ("create_folder", {"name": "ok", "parent_path": [False]}),
])
def test_validation_rejects_malformed_shapes_and_integer_bounds(name, arguments):
    with pytest.raises(registry.OperationValidationError):
        registry.validate_arguments(name, arguments)


def test_unknown_operation_diagnostic_does_not_echo_name_or_url():
    injected = "https://secret.invalid/path"
    with pytest.raises(registry.OperationValidationError) as raised:
        registry.validate_arguments(injected, {})
    assert injected not in repr(raised.value)


def test_read_invocation_reuses_shared_query_and_returns_raw_result(monkeypatch):
    expected = ([], {"total_matches": 0, "returned_segments": 0, "truncated": True})
    calls = []
    monkeypatch.setattr(read, "run_retrieval", lambda **kwargs: calls.append(kwargs) or expected)
    arguments = {
        "query": "ask", "keywords": "raw term", "folder_path": [" Group ", " Folder "],
        "date_from": "2025-01-01", "date_to": "2025-12-31",
    }
    assert registry.invoke_operation("query", arguments) is expected
    assert calls == [{
        "query": "ask", "keywords": "raw term", "folder_path": [" Group ", " Folder "],
        "date_from": "2025-01-01", "date_to": "2025-12-31",
    }]


def test_write_invocation_requires_and_forwards_trusted_context(monkeypatch):
    context = write.WriteContext("operation", "effect", "owner", 1)
    seen = []
    monkeypatch.setattr(write, "_context", lambda supplied, arguments: seen.append(supplied) or supplied)
    monkeypatch.setattr(write, "_execute", lambda supplied, arguments, prepare: "raw-write-result")
    assert registry.invoke_operation(
        "move_transcript", {"doc_id": "doc", "folder_path": ["A"]}, context=context,
    ) == "raw-write-result"
    assert seen == [context]

    with pytest.raises(registry.OperationValidationError):
        registry.invoke_operation("move_transcript", {"doc_id": "doc", "folder_path": ["A"]})
    with pytest.raises(registry.OperationValidationError):
        registry.invoke_operation("move_transcript", {"doc_id": "doc", "folder_path": ["A"]}, context=object())


def test_add_dialog_dispatch_requires_trusted_typed_acceptance_context(monkeypatch):
    from datetime import datetime, timezone
    from core.operations.models import OperationOrigin

    arguments = {"provider": "manual", "utterances": [{"speaker": "A", "text": "words"}]}
    with pytest.raises(registry.OperationValidationError):
        registry.invoke_operation("add_dialog", arguments)
    with pytest.raises(registry.OperationValidationError):
        registry.invoke_operation("add_dialog", arguments, context=object())
    spoofed = registry.AddDialogContext(object(), "op", "caller", OperationOrigin.MCP,
                                        datetime.now(timezone.utc))
    with pytest.raises(registry.OperationValidationError):
        registry.invoke_operation("add_dialog", arguments, context=spoofed)

    accepted = object()
    seen = []
    monkeypatch.setattr(registry.saving, "accept_save", lambda *args, **kwargs: seen.append((args, kwargs)) or accepted)
    context = registry._new_add_dialog_context(
        client="client", operation_id="op", caller_binding="caller",
        origin=OperationOrigin.MCP, now=datetime.now(timezone.utc),
    )
    assert registry.invoke_operation("add_dialog", arguments, context=context) is accepted
    assert seen[0][0] == ("client", arguments)
    assert seen[0][1] == {
        "operation_id": "op", "caller_binding": "caller",
        "origin": OperationOrigin.MCP, "now": context.now,
    }


@pytest.mark.parametrize(("name", "arguments", "expected"), [
    ("move_transcript", {"doc_id": " doc ", "folder_path": [" A "]}, {"doc_id": "doc", "folder_path": [" A "]}),
    ("rename_transcript", {"doc_id": " doc ", "new_title": " title "}, {"doc_id": "doc", "new_title": "title"}),
    ("create_folder", {"name": " folder ", "parent_path": [" Parent "]}, {"name": "folder", "parent_path": [" Parent "]}),
    ("rename_folder", {"folder_path": [" Folder "], "new_name": " New "}, {"folder_path": [" Folder "], "new_name": "New"}),
])
def test_write_invocation_trims_only_legacy_normalized_fields_without_mutating_input(
    monkeypatch, name, arguments, expected,
):
    original = {key: list(value) if isinstance(value, list) else value for key, value in arguments.items()}
    context = write.WriteContext("operation", "effect", "owner", 1)
    seen = []
    monkeypatch.setattr(write, "_context", lambda supplied, values: supplied)
    monkeypatch.setattr(write, "_execute", lambda _context, values, _prepare: seen.append(values) or "raw")

    result = registry.invoke_operation(name, arguments, context=context)
    assert result == (None if name == "rename_transcript" else "raw")
    assert seen == [expected]
    assert arguments == original


def test_non_write_operation_rejects_context_before_dispatch(monkeypatch):
    monkeypatch.setattr(read, "run_retrieval", lambda **_kwargs: pytest.fail("must not dispatch"))
    with pytest.raises(registry.OperationValidationError):
        registry.invoke_operation("query", {"query": "q"}, context=object())
