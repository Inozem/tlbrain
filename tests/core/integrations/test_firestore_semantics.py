"""Offline characterization of Firestore write semantics used by core helpers."""

from datetime import datetime, timedelta, timezone

import pytest

from core.google_drive import firestore as drive_firestore
from core.integrations.firestore import transcripts as transcript_repository
from services.vector_sync.app import index_store


INDEX = "transcript_index"


def _client(firestore_fake):
    return firestore_fake.client_class()


def _writes(firestore_fake, path):
    return [event for event in firestore_fake.trace() if event.get("path") == path]


def test_save_index_replaces_update_index_merges_and_dotted_paths(firestore_fake):
    db = _client(firestore_fake)
    path = f"{INDEX}/synthetic-index"
    ref = db.collection(INDEX).document("synthetic-index")
    ref.set({"keep": "initial", "removed": True, "utterance_hashes": {"0": "old", "1": "sibling"}})
    firestore_fake.set_clock(lambda: datetime(2026, 1, 2, tzinfo=timezone.utc))

    index_store.save_index("synthetic-index", {"status": "imported", "version": "v2.0.0"})
    assert firestore_fake.documents[path] == {"status": "imported", "version": "v2.0.0"}
    assert _writes(firestore_fake, path)[-1]["operation"] == "set"

    ref.set({
        "status": "reimported", "version": "v2.0.0", "retained": True,
        "utterance_hashes": {"0": "old", "1": "sibling"},
    })
    index_store.update_index("synthetic-index", {"status": "synced", "utterance_hashes.0": "new"})
    assert firestore_fake.documents[path] == {
        "status": "synced", "version": "v2.0.0", "retained": True,
        "utterance_hashes": {"0": "new", "1": "sibling"},
    }
    assert _writes(firestore_fake, path)[-1]["operation"] == "update"

    ref.set({"status": "replaced", "nested": {"written": True}})
    ref.set({"nested": {"added": True}}, merge=True)
    assert firestore_fake.documents[path] == {
        "status": "replaced", "nested": {"written": True, "added": True}
    }
    assert [event["operation"] for event in _writes(firestore_fake, path)][-2:] == ["set", "set"]


def test_transcript_repository_crud_and_legacy_get_exports_share_implementation(firestore_fake):
    db = _client(firestore_fake)
    ref = db.collection(INDEX).document("repo-record")
    ref.set({"value": 1, "optional": None})

    assert index_store.load_index is transcript_repository.load_index
    assert drive_firestore.get_transcript_record is transcript_repository.get_transcript_record
    assert index_store.load_index("repo-record") == {"value": 1, "optional": None}
    assert drive_firestore.get_transcript_record("absent") is None

    transcript_repository.save_index("repo-record", {"replacement": True})
    assert firestore_fake.documents[f"{INDEX}/repo-record"] == {"replacement": True}
    transcript_repository.update_index("repo-record", {"nested.path": "updated"})
    assert firestore_fake.documents[f"{INDEX}/repo-record"] == {
        "replacement": True, "nested": {"path": "updated"},
    }
    ref.set({"status": "synced"})
    transcript_repository.delete_index("repo-record")
    assert f"{INDEX}/repo-record" not in firestore_fake.documents
    assert transcript_repository.list_all_index_ids() == []


def test_transcript_repository_lists_filters_and_aggregates(firestore_fake):
    db = _client(firestore_fake)
    records = {
        "old": {"status": "synced", "parent_id": "folder-a", "dialog_date": "2026-01-01", "source_file": "old.docx"},
        "latest": {"status": "synced", "parent_id": "folder-a", "dialog_date": "2026-01-03", "source_file": "latest.docx"},
        "tie": {"status": "synced", "parent_id": "folder-a", "dialog_date": "2026-01-03"},
        "other": {"status": "imported", "parent_id": "folder-b", "dialog_date": "2026-01-02", "source_file": "other.docx", "doc_id": "stored-id"},
        "placeholder": {"status": "queued", "parent_id": "folder-a", "dialog_date": "2026-01-04"},
        "unparented": {"status": "synced", "dialog_date": "2026-01-05"},
    }
    for doc_id, data in records.items():
        db.collection(INDEX).document(doc_id).set(data)

    result = transcript_repository.list_transcripts(
        folder_ids=["folder-a"], date_from="2026-01-01", date_to="2026-01-03", limit=1, offset=1,
    )
    assert result == {
        "total": 3, "returned": 1, "offset": 1, "limit": 1, "has_more": True,
        "transcripts": [{
            "doc_id": "tie", "parent_id": "folder-a", "title": "", "dialog_date": "2026-01-03", "status": "synced",
        }],
    }
    assert transcript_repository.list_imported() == [{
        "doc_id": "stored-id", "status": "imported", "parent_id": "folder-b",
        "dialog_date": "2026-01-02", "source_file": "other.docx",
    }]
    assert transcript_repository.get_doc_ids_by_client("missing") == []
    db.collection(INDEX).document("client-doc").set({"client_name": "Acme"})
    assert transcript_repository.get_doc_ids_by_client("Acme") == ["client-doc"]
    assert transcript_repository.aggregate_transcripts_by_folder() == {
        "folder-a": {"count": 3, "last_date": "2026-01-03", "last_doc_id": "latest"},
    }


def test_fake_faults_propagate_and_capture_before_after_effects(firestore_fake):
    db = _client(firestore_fake)
    ref = db.collection(INDEX).document("synthetic-fault")
    path = f"{INDEX}/synthetic-fault"
    ref.set({"value": "before"})
    firestore_fake.fail_next("update", path=path, phase="before", exception=RuntimeError("before"))

    with pytest.raises(RuntimeError, match="before"):
        ref.update({"value": "blocked"})
    assert firestore_fake.documents[path] == {"value": "before"}
    assert _writes(firestore_fake, path)[-1] == {
        "operation": "update", "path": path, "outcome": "failed", "phase": "before"
    }

    firestore_fake.fail_next("update", path=path, phase="after", exception=RuntimeError("after"))
    with pytest.raises(RuntimeError, match="after"):
        ref.update({"value": "committed"})
    assert firestore_fake.documents[path] == {"value": "committed"}
    assert _writes(firestore_fake, path)[-1] == {
        "operation": "update", "path": path, "outcome": "failed", "phase": "after"
    }


def test_acquire_for_syncing_observes_exact_fifteen_minute_boundary(firestore_fake, monkeypatch):
    now = datetime(2026, 3, 4, 12, 0, tzinfo=timezone.utc)

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now if tz else now.replace(tzinfo=None)

    monkeypatch.setattr(transcript_repository, "datetime", FrozenDateTime)
    firestore_fake.set_clock(lambda: now)
    db = _client(firestore_fake)
    docs = db.collection(INDEX)
    docs.document("fresh").set({"status": "syncing", "status_changed_at": now - timedelta(minutes=14, seconds=59), "extra": 1})
    docs.document("boundary").set({"status": "syncing", "status_changed_at": now - timedelta(minutes=15), "extra": 2})
    docs.document("old").set({"status": "syncing", "status_changed_at": now - timedelta(minutes=16), "extra": 3})
    docs.document("no-time").set({"status": "syncing", "extra": 4})
    docs.document("other-status").set({"status": "error", "error": "synthetic", "extra": 5})

    assert drive_firestore.acquire_for_syncing("fresh") is False
    assert drive_firestore.acquire_for_syncing("boundary") is True
    assert drive_firestore.acquire_for_syncing("old") is True
    assert drive_firestore.acquire_for_syncing("no-time") is True
    assert drive_firestore.acquire_for_syncing("other-status") is True
    assert drive_firestore.acquire_for_syncing("missing") is True

    snapshots = firestore_fake.documents
    assert snapshots[f"{INDEX}/fresh"] == {"status": "syncing", "status_changed_at": now - timedelta(minutes=14, seconds=59), "extra": 1}
    for doc_id, extra in (("boundary", 2), ("old", 3), ("no-time", 4), ("other-status", 5), ("missing", None)):
        data = snapshots[f"{INDEX}/{doc_id}"]
        assert data["doc_id"] == doc_id and data["status"] == "syncing"
        assert data["status_changed_at"] == now
        if extra is not None:
            assert data["extra"] == extra
    assert snapshots[f"{INDEX}/missing"] == {
        "doc_id": "missing", "status": "syncing", "status_changed_at": now,
    }
    events = [event for event in firestore_fake.trace() if event.get("operation") == "commit"]
    assert [(event["path"], event["outcome"]) for event in events] == [
        (f"{INDEX}/boundary", "committed"), (f"{INDEX}/old", "committed"),
        (f"{INDEX}/no-time", "committed"), (f"{INDEX}/other-status", "committed"),
        (f"{INDEX}/missing", "committed"),
    ]


def test_ensure_imported_creates_only_missing_document(firestore_fake):
    firestore_fake.set_clock(lambda: datetime(2026, 4, 5, tzinfo=timezone.utc))
    db = _client(firestore_fake)
    db.collection(INDEX).document("existing-import").set({"status": "synced", "keep": True})

    assert drive_firestore.ensure_imported("new-import", "synthetic-parent", "fixture.docx") is True
    assert drive_firestore.ensure_imported("existing-import", "ignored-parent", "ignored.docx") is False
    snapshots = firestore_fake.documents
    assert snapshots[f"{INDEX}/new-import"] == {
        "doc_id": "new-import", "parent_id": "synthetic-parent", "source_file": "fixture.docx",
        "status": "imported", "status_changed_at": datetime(2026, 4, 5, tzinfo=timezone.utc),
    }
    assert snapshots[f"{INDEX}/existing-import"] == {"status": "synced", "keep": True}
    commits = [event for event in firestore_fake.trace() if event.get("operation") == "commit"]
    assert commits == [{"operation": "commit", "path": f"{INDEX}/new-import", "outcome": "committed"}]


def test_tldv_placeholder_stage_fields_and_delete(firestore_fake):
    now = datetime(2026, 5, 6, tzinfo=timezone.utc)
    firestore_fake.set_clock(lambda: now)

    assert drive_firestore.write_queued("meeting-a") is True
    placeholder_path = f"{INDEX}/tldv-meeting-a"
    assert firestore_fake.documents[placeholder_path] == {
        "meeting_id": "meeting-a", "status": "queued", "status_changed_at": now,
    }
    before_repeat_snapshot = firestore_fake.documents[placeholder_path]
    before_repeat_trace = firestore_fake.trace()
    assert drive_firestore.write_queued("meeting-a") is False
    assert firestore_fake.documents[placeholder_path] == before_repeat_snapshot
    assert firestore_fake.trace() == before_repeat_trace

    drive_firestore.mark_downloading("meeting-a")
    assert firestore_fake.documents[placeholder_path] == {
        "meeting_id": "meeting-a", "status": "downloading", "status_changed_at": now,
    }
    drive_firestore.mark_download_error("meeting-a", "synthetic download failure")
    assert firestore_fake.documents[placeholder_path] == {
        "meeting_id": "meeting-a", "status": "error", "error": "synthetic download failure",
        "error_stage": "import", "status_changed_at": now,
    }
    assert drive_firestore.write_queued("meeting-a") is True
    assert firestore_fake.documents[placeholder_path] == {
        "meeting_id": "meeting-a", "status": "queued", "error": None, "error_stage": None,
        "status_changed_at": now,
    }
    drive_firestore.delete_queued_placeholder("meeting-a")
    assert placeholder_path not in firestore_fake.documents
    relevant = _writes(firestore_fake, placeholder_path)
    assert [event["operation"] for event in relevant] == ["commit", "update", "update", "commit", "delete"]


def test_sync_error_and_move_writers_preserve_or_clear_fields_explicitly(firestore_fake):
    now = datetime(2026, 6, 7, tzinfo=timezone.utc)
    firestore_fake.set_clock(lambda: now)
    db = _client(firestore_fake)
    ref = db.collection(INDEX).document("synthetic-move")
    ref.set({
        "doc_id": "synthetic-move", "status": "error", "error": "old", "error_stage": "parse",
        "parent_id": "old-parent", "parent_id_hash": "parent-hash", "modifiedTime": "old-time",
        "content_hash": "old-content", "version": "v2.0.0", "utterance_hashes": {"0": "a"},
        "other": "preserved",
    })

    drive_firestore.mark_synced("synthetic-move")
    after_synced = firestore_fake.documents[f"{INDEX}/synthetic-move"]
    assert after_synced["status"] == "synced" and after_synced["error"] is None
    assert after_synced["error_stage"] == "parse" and after_synced["other"] == "preserved"

    drive_firestore.mark_error("synthetic-move", "synthetic vector failure")
    after_error = firestore_fake.documents[f"{INDEX}/synthetic-move"]
    assert after_error["status"] == "error" and after_error["error"] == "synthetic vector failure"
    assert after_error["error_stage"] == "vector_sync" and after_error["other"] == "preserved"

    drive_firestore.move_transcript_record("synthetic-move", "new-parent")
    moved = firestore_fake.documents[f"{INDEX}/synthetic-move"]
    assert moved["parent_id"] == "new-parent" and moved["status"] == "imported"
    assert "modifiedTime" not in moved and "content_hash" not in moved and "version" not in moved
    assert moved["error"] is None and moved["parent_id_hash"] == "parent-hash"
    assert moved["utterance_hashes"] == {"0": "a"} and moved["other"] == "preserved"
    assert [event["operation"] for event in _writes(firestore_fake, f"{INDEX}/synthetic-move")] == ["set", "update", "update", "update"]


def test_lifecycle_compatibility_exports_and_recovery_selectors(firestore_fake, monkeypatch):
    now = datetime(2026, 6, 8, 12, 0, tzinfo=timezone.utc)

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now if tz else now.replace(tzinfo=None)

    monkeypatch.setattr(transcript_repository, "datetime", FrozenDateTime)
    db = _client(firestore_fake)
    records = {
        "stale": {"status": "syncing", "status_changed_at": now - timedelta(minutes=16)},
        "fresh": {"status": "syncing", "status_changed_at": now - timedelta(minutes=2)},
        "missing-time": {"status": "syncing"},
        "retry": {"status": "error", "error_stage": "vector_sync"},
        "legacy-error": {"status": "error"},
        "import-error": {"status": "error", "error_stage": "import"},
        "format-error": {"status": "error", "error_stage": "invalid_format"},
        "not-error": {"status": "synced"},
        "unassigned-imported": {"status": "imported", "client_name": "_unassigned"},
        "unassigned-queued": {"status": "queued", "client_name": "_unassigned"},
    }
    for doc_id, data in records.items():
        db.collection(INDEX).document(doc_id).set(data)

    assert drive_firestore.STALE_SYNCING_MINUTES == 15
    assert drive_firestore.get_stale_syncing is transcript_repository.get_stale_syncing
    assert drive_firestore.get_error_docs is transcript_repository.get_error_docs
    assert drive_firestore.get_sync_status is transcript_repository.get_sync_status
    assert transcript_repository.get_stale_syncing() == ["stale"]
    assert transcript_repository.get_error_docs() == ["legacy-error", "retry"]
    assert transcript_repository.get_sync_status() == {
        "total": 10, "queued": 1, "downloading": 0, "imported": 1,
        "syncing": 3, "synced": 1, "error": 4, "_unassigned_count": 1,
    }


def test_skipped_utterance_repository_reads_and_writes_keep_legacy_shape(firestore_fake):
    db = _client(firestore_fake)
    db.collection(INDEX).document("has-skips").set({
        "client_name": "Acme", "skipped_utterances_count": 2, "skipped_utterances": [5, 9],
    })
    db.collection(INDEX).document("count-only").set({"skipped_utterances_count": 1})
    db.collection(INDEX).document("no-skips").set({"skipped_utterances": [3]})

    transcript_repository.update_skipped_utterances("has-skips", [9, 2, 9, 5])
    assert firestore_fake.documents[f"{INDEX}/has-skips"]["skipped_utterances"] == [2, 5, 9]
    assert drive_firestore.update_skipped_utterances is transcript_repository.update_skipped_utterances
    assert drive_firestore.get_docs_with_skipped_utterances is transcript_repository.get_docs_with_skipped_utterances
    assert transcript_repository.get_docs_with_skipped_utterances() == [
        {"doc_id": "count-only", "client_name": "", "skipped_utterances": []},
        {"doc_id": "has-skips", "client_name": "Acme", "skipped_utterances": [2, 5, 9]},
    ]


def test_unassigned_filter_is_delegated_and_preserves_facade_response(firestore_fake, monkeypatch):
    db = _client(firestore_fake)
    db.collection("folders").document("unassigned-folder").set({"name": "_unassigned"})
    records = {
        "first": {"parent_id": "unassigned-folder", "status": "synced", "dialog_date": "2026-01-02"},
        "second": {"parent_id": "unassigned-folder", "status": "imported", "dialog_date": "2026-01-03"},
        "tie": {"parent_id": "unassigned-folder", "status": "error", "dialog_date": "2026-01-03"},
        "queued": {"parent_id": "unassigned-folder", "status": "queued", "dialog_date": "2026-01-04"},
        "downloading": {"parent_id": "unassigned-folder", "status": "downloading", "dialog_date": "2026-01-05"},
        "other-folder": {"parent_id": "other", "status": "synced", "dialog_date": "2026-01-06"},
    }
    for doc_id, data in records.items():
        db.collection(INDEX).document(doc_id).set(data)

    assert transcript_repository.list_unassigned_transcripts("unassigned-folder") == [
        {"doc_id": "second", "dialog_date": "2026-01-03"},
        {"doc_id": "tie", "dialog_date": "2026-01-03"},
        {"doc_id": "first", "dialog_date": "2026-01-02"},
    ]
    delegated_clients = []
    original_helper = transcript_repository.list_unassigned_transcripts

    def track_client(folder_id, client):
        delegated_clients.append(client)
        return original_helper(folder_id, client)

    monkeypatch.setattr(drive_firestore, "_get_db", lambda: db)
    monkeypatch.setattr(drive_firestore, "list_unassigned_transcripts", track_client)
    assert drive_firestore.get_unassigned() == {
        "count": 3,
        "transcripts": [
            {"doc_id": "second", "dialog_date": "2026-01-03"},
            {"doc_id": "tie", "dialog_date": "2026-01-03"},
            {"doc_id": "first", "dialog_date": "2026-01-02"},
        ],
    }
    assert delegated_clients == [db]


def test_unassigned_without_folder_keeps_empty_legacy_result(firestore_fake):
    assert drive_firestore.get_unassigned() == {"count": 0, "transcripts": []}


def test_drive_sync_token_uses_exact_path_and_replacement_set(firestore_fake):
    now = datetime(2026, 7, 8, tzinfo=timezone.utc)
    firestore_fake.set_clock(lambda: now)
    db = _client(firestore_fake)
    ref = db.collection("tokens").document("drive_sync")

    assert drive_firestore.get_drive_sync_token() is None
    ref.set({"unrelated": True})
    assert drive_firestore.get_drive_sync_token() is None
    drive_firestore.set_drive_sync_token("synthetic-token-a")
    assert drive_firestore.get_drive_sync_token() == "synthetic-token-a"
    assert firestore_fake.documents["tokens/drive_sync"] == {"page_token": "synthetic-token-a", "updated_at": now}
    drive_firestore.set_drive_sync_token("synthetic-token-b")
    assert firestore_fake.documents["tokens/drive_sync"] == {"page_token": "synthetic-token-b", "updated_at": now}
    assert [event["operation"] for event in _writes(firestore_fake, "tokens/drive_sync")] == ["set", "set", "set"]
