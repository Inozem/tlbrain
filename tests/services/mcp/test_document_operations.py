from __future__ import annotations

import importlib
import json


def _request(name: str):
    request_type = importlib.import_module("services.mcp.app.mcp.schemas").JSONRPCRequest
    return request_type(
        jsonrpc="2.0",
        method="tools/call",
        id="operation-test",
        params={"name": name, "arguments": {}},
    )


def _payload(response):
    return json.loads(response["result"]["content"][0]["text"])


def test_move_transcript_adapter_preserves_success_response(monkeypatch):
    handler = importlib.import_module("services.mcp.app.mcp.handler")
    outcome_type = handler.document_operations.MoveTranscriptResult
    monkeypatch.setattr(
        handler.document_operations,
        "move_transcript",
        lambda doc_id, folder_path: outcome_type([{"doc_id": "left"}]),
    )

    response = handler._handle_move_transcript(
        _request("move_transcript"),
        {"doc_id": " doc-1 ", "folder_path": ["Folder"]},
    )

    assert _payload(response) == {
        "status": "ok",
        "doc_id": "doc-1",
        "folder_path": ["Folder"],
        "unassigned_remaining": 1,
        "unassigned_transcripts": [{"doc_id": "left"}],
        "suggestion": "1 transcript(s) are still unassigned. Show each one using get_transcript(doc_id='...') and move it using move_transcript.",
    }


def test_move_transcript_adapter_maps_path_errors_and_generic_failures(monkeypatch):
    handler = importlib.import_module("services.mcp.app.mcp.handler")
    monkeypatch.setattr(
        handler.document_operations,
        "move_transcript",
        lambda *_args: (_ for _ in ()).throw(handler.document_operations.FolderPathNotFound(["missing"])),
    )
    missing = handler._handle_move_transcript(
        _request("move_transcript"), {"doc_id": "doc-1", "folder_path": ["missing"]}
    )
    assert missing["error"] == {
        "code": -32602,
        "message": "Folder not found: missing",
        "details": "Use list_folders to see available folders.",
    }

    monkeypatch.setattr(
        handler.document_operations,
        "move_transcript",
        lambda *_args: (_ for _ in ()).throw(
            handler.document_operations.AmbiguousFolderPathError(["same"], 2)
        ),
    )
    ambiguous = handler._handle_move_transcript(
        _request("move_transcript"), {"doc_id": "doc-1", "folder_path": ["same"]}
    )
    assert ambiguous["error"] == {
        "code": -32602,
        "message": "Ambiguous folder path: same matches 2 folders.",
        "details": "Use a more specific path.",
    }

    monkeypatch.setattr(
        handler.document_operations,
        "move_transcript",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("write failed")),
    )
    failed = handler._handle_move_transcript(
        _request("move_transcript"), {"doc_id": "doc-1", "folder_path": ["Folder"]}
    )
    assert failed["error"] == {
        "code": -32603,
        "message": "Failed to move transcript",
        "details": "write failed",
    }


def test_rename_transcript_adapter_maps_missing_record_and_success(monkeypatch):
    handler = importlib.import_module("services.mcp.app.mcp.handler")
    monkeypatch.setattr(
        handler.document_operations,
        "rename_transcript",
        lambda *_args: (_ for _ in ()).throw(handler.document_operations.TranscriptNotFound("doc-1")),
    )
    missing = handler._handle_rename_transcript(
        _request("rename_transcript"), {"doc_id": " doc-1 ", "new_title": "New title"}
    )
    assert missing["error"] == {"code": -32602, "message": "Document not found: doc-1"}

    calls = []
    monkeypatch.setattr(
        handler.document_operations,
        "rename_transcript",
        lambda doc_id, title: calls.append((doc_id, title)),
    )
    renamed = handler._handle_rename_transcript(
        _request("rename_transcript"), {"doc_id": " doc-1 ", "new_title": " New title "}
    )
    assert calls == [("doc-1", "New title")]
    assert _payload(renamed) == {"status": "ok", "doc_id": "doc-1", "new_title": "New title"}

    monkeypatch.setattr(
        handler.document_operations,
        "rename_transcript",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("drive failed")),
    )
    failed = handler._handle_rename_transcript(
        _request("rename_transcript"), {"doc_id": "doc-1", "new_title": "New title"}
    )
    assert failed["error"] == {
        "code": -32603,
        "message": "Failed to rename transcript",
        "details": "drive failed",
    }


def test_transcript_operation_adapters_validate_required_inputs():
    handler = importlib.import_module("services.mcp.app.mcp.handler")
    move = handler._handle_move_transcript(_request("move_transcript"), {"doc_id": " ", "folder_path": []})
    rename = handler._handle_rename_transcript(
        _request("rename_transcript"), {"doc_id": "doc-1", "new_title": " "}
    )
    assert move["error"] == {"code": -32602, "message": "doc_id and folder_path are required"}
    assert rename["error"] == {"code": -32602, "message": "doc_id and new_title are required"}
