from __future__ import annotations

import pytest

from core.documents import operations


def test_move_transcript_preserves_cross_store_order_and_shared_unassigned_client(monkeypatch):
    events = []
    client = object()
    monkeypatch.setattr(operations.folders, "get_all_folders", lambda: {"folder-id": {"name": "Folder"}})
    monkeypatch.setattr(operations, "get_root_folder_id", lambda: "root")
    monkeypatch.setattr(
        operations,
        "resolve_folder_path",
        lambda path, all_folders, root: events.append(("resolve", path, all_folders, root)) or ["folder-id"],
    )
    monkeypatch.setattr(operations, "move_file_to_folder", lambda *args: events.append(("drive", *args)))
    monkeypatch.setattr(
        operations.transcripts,
        "move_transcript_record",
        lambda *args: events.append(("firestore-move", *args)),
    )
    monkeypatch.setattr(
        operations,
        "set_payload_parent_id",
        lambda *args: events.append(("qdrant", *args)),
    )
    monkeypatch.setattr(operations.firestore_client, "get_client", lambda: client)
    monkeypatch.setattr(
        operations.folders,
        "get_folder_id_by_name",
        lambda name, db: events.append(("unassigned-folder", name, db)) or "unassigned-id",
    )
    monkeypatch.setattr(
        operations.transcripts,
        "list_unassigned_transcripts",
        lambda folder_id, db: events.append(("unassigned-list", folder_id, db)) or [{"doc_id": "left"}],
    )

    result = operations.move_transcript("doc-1", ["Folder"])

    assert [event[0] for event in events] == [
        "resolve", "drive", "firestore-move", "qdrant", "unassigned-folder", "unassigned-list",
    ]
    assert events[0] == ("resolve", ["Folder"], {"folder-id": {"name": "Folder"}}, "root")
    assert events[1:4] == [
        ("drive", "doc-1", "folder-id"),
        ("firestore-move", "doc-1", "folder-id"),
        ("qdrant", "doc-1", "root", "folder-id"),
    ]
    assert events[4:] == [
        ("unassigned-folder", "_unassigned", client),
        ("unassigned-list", "unassigned-id", client),
    ]
    assert result.unassigned_count == 1
    assert result.unassigned_transcripts == [{"doc_id": "left"}]


@pytest.mark.parametrize(
    ("matches", "error_type", "message"),
    [
        ([], operations.FolderPathNotFound, "Folder not found: missing"),
        (["one", "two"], operations.AmbiguousFolderPathError, "Ambiguous folder path: missing matches 2 folders."),
    ],
)
def test_move_transcript_rejects_non_unique_paths_before_mutations(monkeypatch, matches, error_type, message):
    effects = []
    monkeypatch.setattr(operations.folders, "get_all_folders", lambda: {})
    monkeypatch.setattr(operations, "get_root_folder_id", lambda: "root")
    monkeypatch.setattr(operations, "resolve_folder_path", lambda *_args: matches)
    monkeypatch.setattr(operations, "move_file_to_folder", lambda *_args: effects.append("drive"))
    monkeypatch.setattr(operations.transcripts, "move_transcript_record", lambda *_args: effects.append("firestore"))
    monkeypatch.setattr(operations, "set_payload_parent_id", lambda *_args: effects.append("qdrant"))

    with pytest.raises(error_type, match=message.replace(".", r"\.")):
        operations.move_transcript("doc-1", ["missing"])

    assert effects == []


def test_rename_transcript_checks_record_then_writes_drive_and_canonical_index(monkeypatch):
    effects = []
    monkeypatch.setattr(
        operations.transcripts,
        "get_transcript_record",
        lambda doc_id: effects.append(("lookup", doc_id)) or {"doc_id": doc_id},
    )
    monkeypatch.setattr(operations, "rename_file", lambda *args: effects.append(("drive", *args)))
    monkeypatch.setattr(
        operations.transcripts,
        "update_index",
        lambda *args: effects.append(("firestore", *args)),
    )

    operations.rename_transcript("doc-1", "Title")

    assert effects == [
        ("lookup", "doc-1"),
        ("drive", "doc-1", "Title"),
        ("firestore", "doc-1", {"source_file": "Title"}),
    ]


def test_rename_transcript_missing_record_fails_before_drive(monkeypatch):
    effects = []
    monkeypatch.setattr(operations.transcripts, "get_transcript_record", lambda _doc_id: None)
    monkeypatch.setattr(operations, "rename_file", lambda *_args: effects.append("drive"))

    with pytest.raises(operations.TranscriptNotFound, match="Document not found: doc-1"):
        operations.rename_transcript("doc-1", "Title")

    assert effects == []
