"""Offline request-shape tests for folder operations in the Drive client."""

from __future__ import annotations

import pytest

from core.integrations.google_drive import client


class _Request:
    def __init__(self, result=None):
        self.result = result or {}

    def execute(self):
        return self.result


class _Files:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(("create", kwargs))
        return _Request({"id": "stable-drive-id"})

    def get(self, **kwargs):
        self.calls.append(("get", kwargs))
        return _Request({"parents": ["old-parent", "shared-parent"]})

    def update(self, **kwargs):
        self.calls.append(("update", kwargs))
        return _Request({"id": "folder-id"})


class _Service:
    def __init__(self):
        self.api = _Files()

    def files(self):
        return self.api


@pytest.mark.offline
def test_create_folder_uses_requested_parent_and_returns_stable_id(monkeypatch):
    service = _Service()
    monkeypatch.setattr(client, "build_drive_service_rw", lambda: service)

    assert client.create_folder("New", "parent-id") == "stable-drive-id"
    assert service.api.calls == [("create", {
        "body": {
            "name": "New",
            "mimeType": "application/vnd.google-apps.folder",
            "parents": ["parent-id"],
        },
        "fields": "id",
    })]


@pytest.mark.offline
def test_move_folder_removes_existing_parents_and_rename_updates_name(monkeypatch):
    service = _Service()
    monkeypatch.setattr(client, "build_drive_service_rw", lambda: service)

    client.move_folder_in_drive("folder-id", "new-parent")
    client.rename_folder("folder-id", "Renamed")

    assert service.api.calls == [
        ("get", {"fileId": "folder-id", "fields": "parents"}),
        ("update", {
            "fileId": "folder-id",
            "addParents": "new-parent",
            "removeParents": "old-parent,shared-parent",
            "fields": "id,parents",
        }),
        ("update", {
            "fileId": "folder-id",
            "body": {"name": "Renamed"},
            "fields": "id",
        }),
    ]

