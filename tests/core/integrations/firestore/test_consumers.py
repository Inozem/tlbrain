from __future__ import annotations

import importlib
from core.google_drive import firestore as drive_firestore
from core.integrations.firestore import folders, transcripts
from core.integrations.google_drive import folder_service


def test_vector_sync_and_mcp_bind_canonical_repositories(sdk_fakes):
    del sdk_fakes
    vector_main = importlib.import_module("services.vector_sync.app.main")
    processor = importlib.import_module("core.indexing.processor")
    handler = importlib.import_module("services.mcp.app.mcp.handler")
    operations = importlib.import_module("core.documents.operations")

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

    assert handler.list_transcripts is transcripts.list_transcripts
    assert handler.folder_service is folder_service
    assert handler.get_sync_status is transcripts.get_sync_status
    assert operations.transcripts is transcripts
    assert operations.folders is folders
    assert operations.firestore_client.get_client is importlib.import_module("core.integrations.firestore.client").get_client

    assert handler.resolve_folder_path.__module__ == drive_firestore.__name__
    assert handler.expand_subtree.__module__ == drive_firestore.__name__


def test_mcp_move_transcript_uses_canonical_unassigned_repositories(monkeypatch):
    operations = importlib.import_module("core.documents.operations")
    calls = []
    monkeypatch.setattr(operations.folders, "get_all_folders", lambda: {"folder-id": {"name": "folder"}})
    monkeypatch.setattr(operations, "get_root_folder_id", lambda: "root")
    monkeypatch.setattr(operations, "resolve_folder_path", lambda *_args: ["folder-id"])
    monkeypatch.setattr(operations, "move_file_to_folder", lambda *_args: None)
    monkeypatch.setattr(operations.transcripts, "move_transcript_record", lambda *_args: None)
    monkeypatch.setattr(operations, "set_payload_parent_id", lambda *_args: None)
    client = object()
    monkeypatch.setattr(operations.firestore_client, "get_client", lambda: client)
    monkeypatch.setattr(
        operations.folders,
        "get_folder_id_by_name",
        lambda name, db: calls.append(("folder", name, db)) or "unassigned-id",
    )
    monkeypatch.setattr(
        operations.transcripts,
        "list_unassigned_transcripts",
        lambda folder_id, db: calls.append(("transcripts", folder_id, db)) or [{"doc_id": "left", "dialog_date": "2026-01-01"}],
    )

    result = operations.move_transcript("doc-1", ["folder"])

    assert calls == [("folder", "_unassigned", client), ("transcripts", "unassigned-id", client)]
    assert result.unassigned_transcripts == [{"doc_id": "left", "dialog_date": "2026-01-01"}]

