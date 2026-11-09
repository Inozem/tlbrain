from __future__ import annotations

import pytest

from core.operations import read
from core.integrations.google_drive.folder_service import FolderNotFound


def test_query_and_transcript_forward_legacy_arguments(monkeypatch):
    query_result = ([{"doc_id": "doc", "text": "raw"}], {"total_matches": 1})
    transcript_result = ([{"doc_id": "doc", "text": "transcript"}], {"truncated": False})
    calls = []
    monkeypatch.setattr(read, "run_retrieval", lambda **kwargs: calls.append(("query", kwargs)) or query_result)
    monkeypatch.setattr(read, "get_transcripts", lambda **kwargs: calls.append(("transcript", kwargs)) or transcript_result)

    assert read.query("ask", keywords="term", folder_path=["A"], date_from="from", date_to="to") == query_result
    assert read.get_transcript(doc_id="doc", folder_path=None, limit="0") == transcript_result
    assert calls == [
        ("query", {"query": "ask", "keywords": "term", "folder_path": ["A"], "date_from": "from", "date_to": "to"}),
        ("transcript", {"doc_id": "doc", "folder_path": None, "date_from": None, "date_to": None, "limit": "0"}),
    ]


def test_folder_and_status_reads_delegate_without_transforming_results(monkeypatch):
    folders = {"folders": [{"id": "f", "name": "Folder"}], "extra": "preserved"}
    status = {"total": 3, "synced": 2, "_unassigned_count": 1}
    monkeypatch.setattr(read.folder_service, "list_folders", lambda: folders)
    monkeypatch.setattr(read, "get_repository_sync_status", lambda: status)

    assert read.list_folders() is folders
    assert read.sync_status() is status


def test_list_recent_resolves_all_matches_and_passes_expansion_order(monkeypatch):
    calls = []
    monkeypatch.setattr(read.folder_service, "resolve_registered_folder_path", lambda path: ["first", "second"])
    monkeypatch.setattr(read.folder_service, "expand_registered_subtree", lambda folder: {
        "first": ["first", "child", "child"], "second": ["second", "last"]
    }[folder])
    expected = {"total": 5, "returned": 2, "transcripts": [{"doc_id": "new"}, {"doc_id": "older"}]}
    monkeypatch.setattr(read, "list_repository_transcripts", lambda **kwargs: calls.append(kwargs) or expected)

    assert read.list_recent_transcripts(["Duplicate"], "2025-01-01", "2025-02-01", 2, 3) is expected
    assert calls == [{
        "folder_ids": ["first", "child", "child", "second", "last"],
        "date_from": "2025-01-01", "date_to": "2025-02-01", "limit": 2, "offset": 3,
    }]


def test_list_recent_keeps_resolution_failures_outside_repository_error_boundary(monkeypatch):
    repository_calls = []
    monkeypatch.setattr(read, "list_repository_transcripts", lambda **kwargs: repository_calls.append(kwargs))
    monkeypatch.setattr(read.folder_service, "resolve_registered_folder_path", lambda _path: [])
    with pytest.raises(FolderNotFound) as missing:
        read.list_recent_transcripts(["Missing"])
    assert missing.value.path == ["Missing"]
    assert repository_calls == []

    resolution_error = RuntimeError("folder repository unavailable")
    monkeypatch.setattr(read.folder_service, "resolve_registered_folder_path", lambda _path: (_ for _ in ()).throw(resolution_error))
    with pytest.raises(RuntimeError, match="folder repository unavailable") as raised:
        read.list_recent_transcripts(["Folder"])
    assert raised.value is resolution_error


def test_list_recent_wraps_only_transcript_repository_failures(monkeypatch):
    failure = RuntimeError("transcript repository unavailable")
    monkeypatch.setattr(read, "list_repository_transcripts", lambda **kwargs: (_ for _ in ()).throw(failure))

    with pytest.raises(read.TranscriptListingError, match="transcript repository unavailable") as raised:
        read.list_recent_transcripts(limit=5, offset=2)
    assert raised.value.__cause__ is failure
