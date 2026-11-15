from __future__ import annotations

import importlib
from itertools import permutations

import pytest
from google.cloud import firestore


ROOT = "pytest-synthetic-root"


@pytest.fixture
def checker_main(sdk_fakes):
    del sdk_fakes
    return importlib.import_module("services.sync_checker.main")


def _folder_change(folder_id, parent_id, name=None):
    return {
        "fileId": folder_id,
        "file": {
            "mimeType": "application/vnd.google-apps.folder",
            "name": name or folder_id,
            "parents": [parent_id],
        },
    }


def _seed_folder(db, folder_id, parent_id, **fields):
    db.collection("folders").document(folder_id).set({
        "name": folder_id, "parent_id": parent_id, **fields,
    })


def _folders(db):
    return {row.id: row.to_dict() for row in db.collection("folders").stream()}


def _process(checker_main, db, changes):
    return checker_main._process_changes(changes, "https://sync.synthetic.test", "queue", db, ROOT)


def test_only_root_descendants_are_registered_and_external_records_are_preserved(checker_main):
    db = firestore.Client()
    _seed_folder(db, "inside", ROOT)
    _seed_folder(db, "outside", "drive-root", description="Keep existing metadata")
    _seed_folder(db, "cycle-a", "cycle-b")
    _seed_folder(db, "cycle-b", "cycle-a")
    original = _folders(db)

    assert _process(checker_main, db, [
        _folder_change("new-inside", "inside"),
        _folder_change("new-outside", "drive-root"),
        _folder_change("outside", "drive-root", "External rename"),
        _folder_change("outside-child", "outside"),
        _folder_change("unknown-child", "missing-parent"),
        _folder_change("cycle-child", "cycle-a"),
    ]) == 0

    actual = _folders(db)
    assert set(actual) == set(original) | {"new-inside"}
    assert actual["new-inside"]["parent_id"] == "inside"
    for folder_id, data in original.items():
        assert actual[folder_id] == data


@pytest.mark.parametrize("order", list(permutations(range(3))))
def test_new_nested_folders_are_registered_in_any_event_order(checker_main, order):
    db = firestore.Client()
    changes = [
        _folder_change("parent", ROOT),
        _folder_change("child", "parent"),
        _folder_change("grandchild", "child"),
    ]

    assert _process(checker_main, db, [changes[index] for index in order]) == 0

    actual = _folders(db)
    assert set(actual) == {"parent", "child", "grandchild"}
    assert actual["parent"]["parent_id"] == ROOT
    assert actual["child"]["parent_id"] == "parent"
    assert actual["grandchild"]["parent_id"] == "child"


def test_move_out_is_tracked_without_deleting_existing_records(checker_main):
    db = firestore.Client()
    _seed_folder(db, "parent", ROOT, description="Retain saved metadata")
    _seed_folder(db, "old-child", "parent")
    db.collection("transcript_index").document("saved-doc").set({
        "parent_id": "old-child", "status": "synced",
    })

    assert _process(checker_main, db, [
        _folder_change("new-child", "parent"),
        _folder_change("parent", "drive-root"),
    ]) == 0
    assert _process(checker_main, db, [
        _folder_change("parent", "drive-root", "Later external rename"),
        _folder_change("new-child", "parent"),
    ]) == 0

    actual = _folders(db)
    assert set(actual) == {"parent", "old-child"}
    assert actual["parent"] == {
        "name": "parent", "parent_id": "drive-root", "description": "Retain saved metadata",
    }
    assert actual["old-child"]["parent_id"] == "parent"
    assert db.collection("transcript_index").document("saved-doc").get().to_dict() == {
        "parent_id": "old-child", "status": "synced",
    }


def test_move_into_root_admits_nested_changes_before_parent_event(checker_main):
    db = firestore.Client()
    _seed_folder(db, "parent", "drive-root", description="Retain metadata")

    assert _process(checker_main, db, [
        _folder_change("child", "parent"),
        _folder_change("parent", ROOT, "Moved into root"),
    ]) == 0

    actual = _folders(db)
    assert actual["parent"] == {
        "name": "Moved into root", "parent_id": ROOT, "description": "Retain metadata",
    }
    assert actual["child"]["parent_id"] == "parent"


@pytest.mark.parametrize("removed", [True, False])
def test_external_deletions_do_not_clean_existing_records(checker_main, removed):
    db = firestore.Client()
    _seed_folder(db, "outside", "drive-root")
    _seed_folder(db, "child", "outside")
    original = _folders(db)
    change = _folder_change("outside", "drive-root")
    if removed:
        change = {"fileId": "outside", "removed": True}
    else:
        change["file"]["trashed"] = True

    assert _process(checker_main, db, [change]) == 0

    assert _folders(db) == original


@pytest.mark.parametrize("removed", [True, False])
def test_in_root_deletion_keeps_existing_orphan_handling(checker_main, removed):
    db = firestore.Client()
    _seed_folder(db, "parent", ROOT)
    _seed_folder(db, "child", "parent")
    db.collection("transcript_index").document("saved-doc").set({
        "parent_id": "parent", "status": "synced",
    })
    change = _folder_change("parent", ROOT)
    if removed:
        change = {"fileId": "parent", "removed": True}
    else:
        change["file"]["trashed"] = True

    assert _process(checker_main, db, [change]) == 0

    actual = _folders(db)
    assert set(actual) == {"child"}
    assert actual["child"]["parent_id"] is None
    assert db.collection("transcript_index").document("saved-doc").get().to_dict() == {
        "parent_id": None, "status": "synced",
    }


def test_root_folder_metadata_is_still_synchronized(checker_main):
    db = firestore.Client()

    assert _process(checker_main, db, [_folder_change(ROOT, "drive-root", "Root name")]) == 0

    assert _folders(db)[ROOT]["name"] == "Root name"


def test_existing_document_change_behavior_is_unchanged(checker_main, monkeypatch):
    db = firestore.Client()
    _seed_folder(db, "outside", "drive-root")
    queued = []
    monkeypatch.setattr(checker_main, "enqueue_task", lambda **kwargs: queued.append(kwargs) or True)
    monkeypatch.setattr(checker_main, "get_all_folders", lambda *args: pytest.fail(
        "document-only changes must not require a folder-tree scan"
    ))

    assert _process(checker_main, db, [{
        "fileId": "existing-doc",
        "file": {"mimeType": "application/vnd.google-apps.document", "parents": ["outside"]},
    }]) == 1

    assert queued == [{
        "queue_name": "queue", "url": "https://sync.synthetic.test/sync/doc/existing-doc",
    }]
