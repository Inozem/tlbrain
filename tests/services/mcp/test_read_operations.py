from __future__ import annotations

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
