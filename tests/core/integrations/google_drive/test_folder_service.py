"""Offline tests for coordinated folder operations."""

from __future__ import annotations

import pytest

from core.integrations.google_drive import folder_service


@pytest.mark.offline
def test_create_folder_preserves_drive_id_and_effect_order(monkeypatch):
    events = []
    monkeypatch.setattr(folder_service, "_resolve", lambda path, role: events.append(("resolve", path, role)) or "parent-id")
    monkeypatch.setattr(
        folder_service.folder_repository,
        "folder_name_exists",
        lambda name, parent: events.append(("conflict", name, parent)) or False,
    )
    monkeypatch.setattr(
        folder_service.drive_client,
        "create_folder",
        lambda name, parent: events.append(("drive", name, parent)) or "stable-drive-id",
    )
    monkeypatch.setattr(
        folder_service.folder_repository,
        "upsert_folder",
        lambda *args: events.append(("firestore", *args)),
    )

    result = folder_service.create_folder("New", ["Parent"])

    assert result == "stable-drive-id"
    assert events == [
        ("resolve", ["Parent"], "parent"),
        ("conflict", "New", "parent-id"),
        ("drive", "New", "parent-id"),
        ("firestore", "stable-drive-id", "New", "parent-id"),
    ]


@pytest.mark.offline
def test_rename_checks_sibling_before_drive_and_updates_same_id(monkeypatch):
    events = []
    monkeypatch.setattr(folder_service, "_resolve", lambda path, role: events.append(("resolve", path, role)) or "folder-id")
    monkeypatch.setattr(
        folder_service.folder_repository,
        "get_folder_by_id",
        lambda folder_id: events.append(("get", folder_id)) or {"parent_id": "parent-id"},
    )
    monkeypatch.setattr(
        folder_service.folder_repository,
        "folder_name_exists",
        lambda name, parent: events.append(("conflict", name, parent)) or False,
    )
    monkeypatch.setattr(
        folder_service.drive_client,
        "rename_folder",
        lambda *args: events.append(("drive", *args)),
    )
    monkeypatch.setattr(
        folder_service.folder_repository,
        "upsert_folder",
        lambda *args: events.append(("firestore", *args)),
    )

    assert folder_service.rename_folder(["Old"], "New") == "folder-id"
    assert events == [
        ("resolve", ["Old"], "folder"),
        ("get", "folder-id"),
        ("conflict", "New", "parent-id"),
        ("drive", "folder-id", "New"),
        ("firestore", "folder-id", "New", "parent-id"),
    ]


@pytest.mark.offline
def test_create_raises_typed_parent_path_error(monkeypatch):
    monkeypatch.setattr(folder_service, "get_root_folder_id", lambda: "root-id")
    monkeypatch.setattr(folder_service.folder_repository, "get_all_folders", lambda: {})

    with pytest.raises(folder_service.FolderNotFound) as error:
        folder_service.create_folder("New", ["Missing"])

    assert error.value.path == ["Missing"]
    assert error.value.role == "parent"


@pytest.mark.offline
def test_move_folder_preserves_id_and_effect_order(monkeypatch):
    events = []

    def resolve(path, role):
        events.append(("resolve", path, role))
        return {("Source",): "folder-id", ("Destination",): "destination-id"}[tuple(path)]

    def get_all_folders():
        events.append(("cycle_snapshot",))
        return {
            "folder-id": {"name": "Source", "parent_id": "root-id"},
            "child-id": {"name": "Child", "parent_id": "folder-id"},
            "destination-id": {"name": "Destination", "parent_id": "root-id"},
        }

    def get_folder_by_id(folder_id):
        events.append(("get_folder", folder_id))
        return {"name": "Source", "parent_id": "root-id"}

    monkeypatch.setattr(folder_service, "_resolve", resolve)
    monkeypatch.setattr(
        folder_service.folder_repository, "get_all_folders", get_all_folders
    )
    monkeypatch.setattr(
        folder_service.folder_repository, "get_folder_by_id", get_folder_by_id
    )
    monkeypatch.setattr(
        folder_service.drive_client,
        "move_folder_in_drive",
        lambda *args: events.append(("drive_move", *args)),
    )
    monkeypatch.setattr(
        folder_service.folder_repository,
        "upsert_folder",
        lambda *args: events.append(("firestore_upsert", *args)),
    )

    result = folder_service.move_folder(["Source"], ["Destination"])

    assert result == ("folder-id", "destination-id")
    assert events == [
        ("resolve", ["Source"], "folder"),
        ("resolve", ["Destination"], "parent"),
        ("cycle_snapshot",),
        ("get_folder", "folder-id"),
        ("drive_move", "folder-id", "destination-id"),
        ("firestore_upsert", "folder-id", "Source", "destination-id"),
    ]

