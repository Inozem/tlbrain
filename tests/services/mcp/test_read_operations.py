from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from services.mcp.app.mcp import handler
from services.mcp.app.mcp.schemas import JSONRPCRequest


def _request(name: str, arguments: dict | None = None) -> JSONRPCRequest:
    return JSONRPCRequest(**{
        "jsonrpc": "2.0",
        "id": "read-test",
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments or {}},
    })


def test_sync_status_presentation_does_not_mutate_core_result(monkeypatch):
    domain_status = {"total": 4, "synced": 3, "_unassigned_count": 2}
    monkeypatch.setattr(handler.read_operations, "sync_status", lambda: domain_status)

    result = handler._handle_sync_status(_request("sync_status"))

    assert domain_status == {"total": 4, "synced": 3, "_unassigned_count": 2}
    assert "2 transcript(s) are currently unassigned" in result["result"]["content"][0]["text"]


def test_sync_status_operation_lookup_projects_safe_read_only_fields(monkeypatch):
    changed_at = datetime(2026, 10, 22, 12, 34, tzinfo=timezone.utc)
    started_at = datetime(2026, 10, 22, 12, 30, tzinfo=timezone.utc)
    operation = SimpleNamespace(
        operation_id="operation-123",
        state=SimpleNamespace(value="running"),
        stage="create_document",
        status_updated_at=changed_at,
        started_at=started_at,
        finished_at=None,
        stage_failure_counts=(("create_document", 2),),
        doc_id=None,
        selected_folder_path=("Projects", "Acme"),
        provider="tldv",
        result=None,
        error=SimpleNamespace(
            message="raw internal error", sanitized_message=None, code="write_failed",
            category="provider", service="drive", stage="create_document", retryable=True, failures=2,
            occurred_at=changed_at,
        ),
        caller_binding="secret-caller",
        immutable_input_digest="secret-digest",
        lease_owner="secret-owner",
        fence=7,
    )
    client = object()
    calls = []
    monkeypatch.setattr(handler, "get_client", lambda: client)
    monkeypatch.setattr(
        handler, "get_operation", lambda passed_client, operation_id: (
            calls.append((passed_client, operation_id)) or operation
        ),
    )

    response = handler._handle_sync_status(
        _request("sync_status"), {"operation_id": "operation-123"}
    )

    import json

    payload = json.loads(response["result"]["content"][0]["text"])
    assert calls == [(client, "operation-123")]
    assert payload == {
        "operation_id": "operation-123",
        "operation_status": "running",
        "stage": "create_document",
        "status_changed_at": "2026-10-22T12:34:00+00:00",
        "started_at": "2026-10-22T12:30:00+00:00",
        "finished_at": None,
        "stage_failure_counts": {"create_document": 2},
        "doc_id": None,
        "folder_path": ["Projects", "Acme"],
        "provider": "tldv",
        "result": None,
        "error": {
            "message": "Operation failed",
            "code": "write_failed",
            "category": "provider",
            "service": "drive",
            "stage": "create_document",
            "retryable": True,
            "failures": 2,
            "occurred_at": "2026-10-22T12:34:00+00:00",
        },
    }
    serialized = response["result"]["content"][0]["text"]
    for secret in ("raw internal error", "secret-caller", "secret-digest", "secret-owner"):
        assert secret not in serialized


def test_sync_status_operation_lookup_uses_persisted_result_fields(monkeypatch):
    operation = SimpleNamespace(
        operation_id="completed-operation",
        state="completed",
        stage="index",
        status_updated_at=None,
        started_at=None,
        finished_at=None,
        stage_failure_counts=(),
        doc_id="document-1",
        selected_folder_path=None,
        provider=None,
        result=SimpleNamespace(
            document_id="document-1",
            title="Meeting notes",
            resource_url="https://docs.example/document-1",
            skipped_utterances=(3, 8),
        ),
        error=None,
    )
    monkeypatch.setattr(handler, "get_client", lambda: object())
    monkeypatch.setattr(handler, "get_operation", lambda *_args: operation)

    response = handler._handle_sync_status(
        _request("sync_status"), {"operation_id": "completed-operation"}
    )

    import json

    payload = json.loads(response["result"]["content"][0]["text"])
    assert payload["result"] == {
        "document_id": "document-1",
        "title": "Meeting notes",
        "skipped_utterances": [3, 8],
    }
    assert "resource_url" not in payload["result"]
    assert "https://docs.example/document-1" not in response["result"]["content"][0]["text"]
    assert payload["doc_id"] == "document-1"


@pytest.mark.parametrize("lookup_result", [None, RuntimeError("private firestore detail")])
def test_sync_status_operation_lookup_errors_are_sanitized(monkeypatch, lookup_result):
    def lookup(*_args):
        if isinstance(lookup_result, Exception):
            raise lookup_result
        return lookup_result

    monkeypatch.setattr(handler, "get_client", lambda: object())
    monkeypatch.setattr(handler, "get_operation", lookup)
    response = handler._handle_sync_status(
        _request("sync_status"), {"operation_id": "missing-operation"}
    )

    assert response["error"]["message"] == (
        "Operation not found" if lookup_result is None else "Failed to get operation status"
    )
    assert "private firestore detail" not in str(response)


@pytest.mark.parametrize("operation_id", ["", "  ", None, 123, "bad/id"])
def test_sync_status_rejects_malformed_operation_selector_without_lookup(monkeypatch, operation_id):
    monkeypatch.setattr(
        handler, "get_operation", lambda *_args: pytest.fail("invalid ID reached the repository")
    )
    response = handler._handle_sync_status(
        _request("sync_status"), {"operation_id": operation_id}
    )

    assert response["error"] == {
        "code": -32602,
        "message": "operation_id must be a non-empty string",
    }


def test_sync_status_operation_repository_lookup_performs_no_writes(monkeypatch, firestore_fake):
    from core.integrations.firestore.operations import create_operation
    from core.operations.models import OperationOrigin

    client = firestore_fake.client_class()
    create_operation(
        client,
        operation_id="pre-create-operation",
        origin=OperationOrigin.MCP,
        caller_binding="caller",
        immutable_input_digest="digest",
        immutable_input_ref="ref",
        now=datetime(2026, 10, 22, 12, 0, tzinfo=timezone.utc),
    )
    committed_before = [
        entry for entry in firestore_fake.trace() if entry.get("outcome") == "committed"
    ]
    monkeypatch.setattr(handler, "get_client", lambda: client)
    monkeypatch.setattr(
        handler.read_operations, "sync_status", lambda: pytest.fail("aggregate path was called")
    )

    response = handler._handle_sync_status(
        _request("sync_status"), {"operation_id": "pre-create-operation"}
    )

    import json

    payload = json.loads(response["result"]["content"][0]["text"])
    committed_after = [
        entry for entry in firestore_fake.trace() if entry.get("outcome") == "committed"
    ]
    assert payload["doc_id"] is None
    assert payload["operation_status"] == "pending"
    assert committed_after == committed_before


def test_recent_transcript_repository_failure_keeps_legacy_error(monkeypatch):
    def fail(**kwargs):
        raise handler.read_operations.TranscriptListingError("storage unavailable")

    monkeypatch.setattr(handler.read_operations, "list_recent_transcripts", fail)
    result = handler._handle_list_recent_transcripts(_request("list_recent_transcripts"), {})

    assert result["error"] == {
        "code": -32603,
        "message": "Failed to list transcripts",
        "details": "storage unavailable",
    }


def test_recent_transcript_resolution_errors_still_propagate(monkeypatch):
    failure = RuntimeError("folder lookup unavailable")
    monkeypatch.setattr(
        handler.read_operations,
        "list_recent_transcripts",
        lambda **kwargs: (_ for _ in ()).throw(failure),
    )

    with pytest.raises(RuntimeError) as raised:
        handler._handle_list_recent_transcripts(_request("list_recent_transcripts"), {"folder_path": ["Folder"]})
    assert raised.value is failure


def test_recent_transcript_missing_folder_uses_legacy_invalid_params_error(monkeypatch):
    from core.integrations.google_drive.folder_service import FolderNotFound

    monkeypatch.setattr(
        handler.read_operations,
        "list_recent_transcripts",
        lambda **kwargs: (_ for _ in ()).throw(FolderNotFound(["Missing", "Folder"], "folder")),
    )
    result = handler._handle_list_recent_transcripts(_request("list_recent_transcripts"), {
        "folder_path": ["Missing", "Folder"]
    })

    assert result["error"] == {
        "code": -32602,
        "message": "Folder not found: Missing/Folder",
        "details": "Use list_folders to see available folders.",
    }
