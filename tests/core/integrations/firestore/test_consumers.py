from __future__ import annotations

import importlib
import json

from core.google_drive import firestore as drive_firestore
from core.integrations.firestore import folders, transcripts


def test_vector_sync_and_mcp_bind_canonical_repositories(sdk_fakes):
    del sdk_fakes
    vector_main = importlib.import_module("services.vector_sync.app.main")
    processor = importlib.import_module("services.vector_sync.app.processor")
    handler = importlib.import_module("services.mcp.app.mcp.handler")

    assert vector_main.ensure_imported is transcripts.ensure_imported
    assert vector_main.mark_error is transcripts.mark_error
    assert vector_main.delete_index is transcripts.delete_index
    assert vector_main.load_index is transcripts.load_index
    assert vector_main.get_folder_by_id is folders.get_folder_by_id

    assert processor.acquire_for_syncing is transcripts.acquire_for_syncing
    assert processor.mark_error is transcripts.mark_error
    assert processor.mark_synced is transcripts.mark_synced
    assert processor.update_skipped_utterances is transcripts.update_skipped_utterances
    assert processor.load_index is transcripts.load_index
    assert processor.update_index is transcripts.update_index

    assert handler.get_sync_status is transcripts.get_sync_status
    assert handler.get_transcript_record is transcripts.get_transcript_record
    assert handler.list_transcripts is transcripts.list_transcripts
    assert handler.move_transcript_record is transcripts.move_transcript_record
    assert handler.update_index is transcripts.update_index
    assert handler.get_folder_by_id is folders.get_folder_by_id
    assert handler.folder_name_exists is folders.folder_name_exists
    assert handler.get_folder_id_by_name is folders.get_folder_id_by_name
    assert handler.list_unassigned_transcripts is transcripts.list_unassigned_transcripts

    assert handler.resolve_folder_path.__module__ == drive_firestore.__name__
    assert handler.expand_subtree.__module__ == drive_firestore.__name__


def test_mcp_move_transcript_uses_canonical_unassigned_repositories(monkeypatch):
    handler = importlib.import_module("services.mcp.app.mcp.handler")
    request = importlib.import_module("services.mcp.app.mcp.schemas").JSONRPCRequest(
        jsonrpc="2.0",
        method="tools/call",
        id="move-test",
        params={"name": "move_transcript", "arguments": {"doc_id": "doc-1", "folder_path": ["folder"]}},
    )
    calls = []
    monkeypatch.setattr(handler, "resolve_folder_path", lambda _path: ["folder-id"])
    monkeypatch.setattr(handler, "get_root_folder_id", lambda: "root")
    monkeypatch.setattr(handler, "move_file_to_folder", lambda *_args: None)
    monkeypatch.setattr(handler, "move_transcript_record", lambda *_args: None)
    monkeypatch.setattr(handler, "set_payload_parent_id", lambda *_args: None)
    client = object()
    monkeypatch.setattr(handler, "get_client", lambda: client)
    monkeypatch.setattr(
        handler,
        "get_folder_id_by_name",
        lambda name, db: calls.append(("folder", name, db)) or "unassigned-id",
    )
    monkeypatch.setattr(
        handler,
        "list_unassigned_transcripts",
        lambda folder_id, db: calls.append(("transcripts", folder_id, db)) or [{"doc_id": "left", "dialog_date": "2026-01-01"}],
    )

    result = handler._handle_move_transcript(request, {"doc_id": "doc-1", "folder_path": ["folder"]})

    assert calls == [("folder", "_unassigned", client), ("transcripts", "unassigned-id", client)]
    assert json.loads(result["result"]["content"][0]["text"]) == {
        "status": "ok",
        "doc_id": "doc-1",
        "folder_path": ["folder"],
        "unassigned_remaining": 1,
        "unassigned_transcripts": [{"doc_id": "left", "dialog_date": "2026-01-01"}],
        "suggestion": "1 transcript(s) are still unassigned. Show each one using get_transcript(doc_id='...') and move it using move_transcript.",
    }

