"""Offline characterization of Firestore write semantics used by core helpers."""

import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from core.integrations.firestore import folders as folder_repository
from core.integrations.firestore import sync_state as sync_state_repository
from core.integrations.firestore import transcripts as transcript_repository


INDEX = "transcript_index"
FOLDERS = "folders"


def _client(firestore_fake):
    return firestore_fake.client_class()


def _writes(firestore_fake, path):
    return [event for event in firestore_fake.trace() if event.get("path") == path]


def test_folder_repository_preserves_create_and_update_semantics(firestore_fake, monkeypatch):
    db = _client(firestore_fake)
    now = datetime(2026, 8, 9, tzinfo=timezone.utc)
    firestore_fake.set_clock(lambda: now)
    monkeypatch.setattr(folder_repository, "get_client", lambda: db)

    folder_repository.upsert_folder("stable-drive-id", "New folder", None)
    path = f"{FOLDERS}/stable-drive-id"
    assert firestore_fake.documents[path] == {
        "folder_id": "stable-drive-id", "name": "New folder", "parent_id": None,
        "speakers": {}, "description": None, "created_at": now,
    }
    assert _writes(firestore_fake, path)[-1]["operation"] == "set"

    db.collection(FOLDERS).document("existing-folder").set({
        "folder_id": "existing-folder", "name": "Old name", "parent_id": "old-parent",
        "speakers": {"s123": 4}, "description": "Keep this", "created_at": "original-time",
        "unrelated": "preserved",
    })
    folder_repository.upsert_folder("existing-folder", "Renamed", None)
    existing_path = f"{FOLDERS}/existing-folder"
    assert firestore_fake.documents[existing_path] == {
        "folder_id": "existing-folder", "name": "Renamed", "parent_id": None,
        "speakers": {"s123": 4}, "description": "Keep this", "created_at": "original-time",
        "unrelated": "preserved",
    }
    assert _writes(firestore_fake, existing_path)[-1]["operation"] == "update"

    before = firestore_fake.trace()
    folder_repository.upsert_folder("existing-folder", "Renamed", None)
    assert firestore_fake.trace() == before


def test_folder_repository_crud_nullable_parent_and_orphaning(firestore_fake, monkeypatch):
    db = _client(firestore_fake)
    monkeypatch.setattr(folder_repository, "get_client", lambda: db)
    folders = db.collection(FOLDERS)
    folders.document("root-a").set({"folder_id": "root-a", "name": "Root", "parent_id": None})
    folders.document("child-a").set({"folder_id": "child-a", "name": "Child", "parent_id": "root-a"})
    folders.document("child-b").set({"folder_id": "child-b", "name": "Child", "parent_id": "root-a"})
    folders.document("other").set({"folder_id": "other", "name": "Root", "parent_id": "elsewhere"})

    assert folder_repository.folder_name_exists("Root", None)
    assert not folder_repository.folder_name_exists("Root", "missing")
    assert folder_repository.get_folder_by_id("root-a")["name"] == "Root"
    assert folder_repository.get_folder_by_id("missing") is None
    assert folder_repository.get_all_folders()["child-a"]["parent_id"] == "root-a"
    assert folder_repository.orphan_folder_children("root-a") == 2
    assert firestore_fake.documents[f"{FOLDERS}/child-a"]["parent_id"] is None
    folder_repository.delete_folder("other")
    assert f"{FOLDERS}/other" not in firestore_fake.documents


def test_folder_speaker_index_and_repository_operations(firestore_fake, monkeypatch):
    db = _client(firestore_fake)
    monkeypatch.setattr(folder_repository, "get_client", lambda: db)
    name = "Zoë Speaker"
    key = "s" + hashlib.md5(name.encode("utf-8")).hexdigest()[:12]
    assert folder_repository.speaker_key(name) == key

    db.collection(FOLDERS).document("speaker-folder").set({
        "folder_id": "speaker-folder", "name": "Speaker folder", "parent_id": None,
        "speakers": {key: 1},
    })
    db.collection(FOLDERS).document("other-folder").set({"name": "Other", "speakers": {key: 0}})
    matched = list(folder_repository.get_folders_by_speaker(name, db))
    assert [doc.id for doc in matched] == ["speaker-folder"]

    folder_repository.increment_folder_speakers("speaker-folder", [name], db=db)
    assert firestore_fake.documents[f"{FOLDERS}/speaker-folder"]["speakers"][key] == 2
    assert _writes(firestore_fake, f"{FOLDERS}/speaker-folder")[-1]["operation"] == "update"

    folder_repository.upsert_folder("facade-folder", "Facade", None, db)
    assert folder_repository.get_folder_by_id("facade-folder", db)["folder_id"] == "facade-folder"
    assert folder_repository.get_all_folders(db)["facade-folder"]["name"] == "Facade"
    assert folder_repository.folder_name_exists("Facade", None, db)
    assert folder_repository.orphan_folder_children("speaker-folder", db) == 0
    folder_repository.delete_folder("facade-folder", db)
    assert f"{FOLDERS}/facade-folder" not in firestore_fake.documents


def test_folder_repository_default_client_is_lazy(firestore_fake, monkeypatch):
    calls = []
    db = _client(firestore_fake)
    monkeypatch.setattr(folder_repository, "get_client", lambda: calls.append(True) or db)

    assert folder_repository.speaker_key("speaker").startswith("s")
    assert calls == []
    assert folder_repository.get_all_folders() == {}
    assert calls == [True]


def test_save_index_replaces_update_index_merges_and_dotted_paths(firestore_fake):
    db = _client(firestore_fake)
    path = f"{INDEX}/synthetic-index"
    ref = db.collection(INDEX).document("synthetic-index")
    ref.set({"keep": "initial", "removed": True, "utterance_hashes": {"0": "old", "1": "sibling"}})
    firestore_fake.set_clock(lambda: datetime(2026, 1, 2, tzinfo=timezone.utc))

    transcript_repository.save_index("synthetic-index", {"status": "imported", "version": "v2.0.0"})
    assert firestore_fake.documents[path] == {"status": "imported", "version": "v2.0.0"}
    assert _writes(firestore_fake, path)[-1]["operation"] == "set"

    ref.set({
        "status": "reimported", "version": "v2.0.0", "retained": True,
        "utterance_hashes": {"0": "old", "1": "sibling"},
    })
    transcript_repository.update_index("synthetic-index", {"status": "synced", "utterance_hashes.0": "new"})
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


def test_save_index_uses_injected_client_without_creating_another(firestore_fake, monkeypatch):
    db = _client(firestore_fake)
    monkeypatch.setattr(
        transcript_repository,
        "get_client",
        lambda: pytest.fail("injected client should be used"),
    )

    transcript_repository.save_index("injected-client", {"status": "imported"}, db=db)

    assert firestore_fake.documents[f"{INDEX}/injected-client"] == {"status": "imported"}


def test_transcript_reads_and_updates_use_injected_client(firestore_fake, monkeypatch):
    db = _client(firestore_fake)
    db.collection(INDEX).document("injected-read").set({"status": "imported", "keep": True})
    monkeypatch.setattr(
        transcript_repository,
        "get_client",
        lambda: pytest.fail("injected client should be used"),
    )

    assert transcript_repository.get_transcript_record("injected-read", db=db) == {
        "status": "imported", "keep": True,
    }
    transcript_repository.update_index("injected-read", {"status": "synced"}, db=db)

    assert firestore_fake.documents[f"{INDEX}/injected-read"] == {
        "status": "synced", "keep": True,
    }


def test_transcript_orphaning_updates_direct_children(firestore_fake, monkeypatch):
    db = _client(firestore_fake)
    records = {
        "child-a": {"parent_id": "removed-folder", "status": "synced", "keep": 1},
        "child-b": {"parent_id": "removed-folder", "status": "imported"},
        "other": {"parent_id": "other-folder", "status": "synced"},
    }
    for doc_id, data in records.items():
        db.collection(INDEX).document(doc_id).set(data)

    assert transcript_repository.orphan_transcript_children("removed-folder", db=db) == 2
    assert firestore_fake.documents[f"{INDEX}/child-a"] == {
        "parent_id": None, "status": "synced", "keep": 1,
    }
    assert firestore_fake.documents[f"{INDEX}/child-b"]["parent_id"] is None
    assert firestore_fake.documents[f"{INDEX}/other"]["parent_id"] == "other-folder"
    assert [event["operation"] for event in _writes(firestore_fake, f"{INDEX}/child-a")] == ["set", "update"]
    assert [event["operation"] for event in _writes(firestore_fake, f"{INDEX}/child-b")] == ["set", "update"]

    assert transcript_repository.orphan_transcript_children("other-folder", db) == 1
    assert firestore_fake.documents[f"{INDEX}/other"]["parent_id"] is None


def test_index_stream_preserves_lazy_projected_query():
    events = []

    class Stream:
        def __iter__(self):
            events.append("iterated")
            return iter(())

    stream = Stream()

    class Query:
        def select(self, fields):
            events.append(("select", fields))
            return self

        def stream(self):
            events.append("stream")
            return stream

    class Client:
        def collection(self, collection):
            events.append(("collection", collection))
            return Query()

    result = transcript_repository.stream_index_records(["status", "error_stage"], db=Client())

    assert result is stream
    assert events == [
        ("collection", INDEX),
        ("select", ["status", "error_stage"]),
        "stream",
    ]
    iter(result)
    assert events[-1] == "iterated"


def test_transcript_repository_crud_and_legacy_get_exports_share_implementation(firestore_fake):
    db = _client(firestore_fake)
    ref = db.collection(INDEX).document("repo-record")
    ref.set({"value": 1, "optional": None})

    assert transcript_repository.load_index("repo-record") == {"value": 1, "optional": None}
    assert transcript_repository.get_transcript_record("absent") is None

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


def test_transcript_repository_queries_by_meeting_id(firestore_fake):
    db = _client(firestore_fake)
    db.collection(INDEX).document("matching-a").set({"meeting_id": "meeting-a", "status": "queued"})
    db.collection(INDEX).document("matching-b").set({"meeting_id": "meeting-a"})
    db.collection(INDEX).document("other-meeting").set({"meeting_id": "meeting-b", "status": "imported"})

    matches = transcript_repository.get_index_records_by_meeting_id("meeting-a", db)

    assert [(doc.id, doc.to_dict()) for doc in matches] == [
        ("matching-a", {"meeting_id": "meeting-a", "status": "queued"}),
        ("matching-b", {"meeting_id": "meeting-a"}),
    ]


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

    assert transcript_repository.acquire_for_syncing("fresh") is False
    assert transcript_repository.acquire_for_syncing("boundary") is True
    assert transcript_repository.acquire_for_syncing("old") is True
    assert transcript_repository.acquire_for_syncing("no-time") is True
    assert transcript_repository.acquire_for_syncing("other-status") is True
    assert transcript_repository.acquire_for_syncing("missing") is True

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

    assert transcript_repository.ensure_imported("new-import", "synthetic-parent", "fixture.docx") is True
    assert transcript_repository.ensure_imported("existing-import", "ignored-parent", "ignored.docx") is False
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

    assert transcript_repository.write_queued("meeting-a") is True
    placeholder_path = f"{INDEX}/tldv-meeting-a"
    assert firestore_fake.documents[placeholder_path] == {
        "meeting_id": "meeting-a", "status": "queued", "status_changed_at": now,
    }
    before_repeat_snapshot = firestore_fake.documents[placeholder_path]
    before_repeat_trace = firestore_fake.trace()
    assert transcript_repository.write_queued("meeting-a") is False
    assert firestore_fake.documents[placeholder_path] == before_repeat_snapshot
    assert firestore_fake.trace() == before_repeat_trace

    transcript_repository.mark_downloading("meeting-a")
    assert firestore_fake.documents[placeholder_path] == {
        "meeting_id": "meeting-a", "status": "downloading", "status_changed_at": now,
    }
    transcript_repository.mark_download_error("meeting-a", "synthetic download failure")
    assert firestore_fake.documents[placeholder_path] == {
        "meeting_id": "meeting-a", "status": "error", "error": "synthetic download failure",
        "error_stage": "import", "status_changed_at": now,
    }
    assert transcript_repository.write_queued("meeting-a") is True
    assert firestore_fake.documents[placeholder_path] == {
        "meeting_id": "meeting-a", "status": "queued", "error": None, "error_stage": None,
        "status_changed_at": now,
    }
    transcript_repository.delete_queued_placeholder("meeting-a")
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

    transcript_repository.mark_synced("synthetic-move")
    after_synced = firestore_fake.documents[f"{INDEX}/synthetic-move"]
    assert after_synced["status"] == "synced" and after_synced["error"] is None
    assert after_synced["error_stage"] == "parse" and after_synced["other"] == "preserved"

    transcript_repository.mark_error("synthetic-move", "synthetic vector failure")
    after_error = firestore_fake.documents[f"{INDEX}/synthetic-move"]
    assert after_error["status"] == "error" and after_error["error"] == "synthetic vector failure"
    assert after_error["error_stage"] == "vector_sync" and after_error["other"] == "preserved"

    transcript_repository.move_transcript_record("synthetic-move", "new-parent")
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

    assert transcript_repository.STALE_SYNCING_MINUTES == 15
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
def test_drive_sync_token_uses_exact_path_and_replacement_set(firestore_fake):
    now = datetime(2026, 7, 8, tzinfo=timezone.utc)
    firestore_fake.set_clock(lambda: now)
    db = _client(firestore_fake)
    ref = db.collection("tokens").document("drive_sync")

    assert sync_state_repository.get_drive_sync_token() is None
    ref.set({"unrelated": True})
    assert sync_state_repository.get_drive_sync_token() is None
    sync_state_repository.set_drive_sync_token("synthetic-token-a")
    assert sync_state_repository.get_drive_sync_token() == "synthetic-token-a"
    assert firestore_fake.documents["tokens/drive_sync"] == {"page_token": "synthetic-token-a", "updated_at": now}
    assert _writes(firestore_fake, "tokens/drive_sync")[-1]["operation"] == "set"
    sync_state_repository.set_drive_sync_token("synthetic-token-b")
    assert firestore_fake.documents["tokens/drive_sync"] == {"page_token": "synthetic-token-b", "updated_at": now}
    assert [event["operation"] for event in _writes(firestore_fake, "tokens/drive_sync")] == ["set", "set", "set"]
