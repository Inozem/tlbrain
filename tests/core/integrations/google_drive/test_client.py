"""Offline request-shape tests for folder operations in the Drive client."""

from __future__ import annotations

import pytest

from core.integrations.google_drive import client
from core.integrations.google_drive import changes, files, scanner


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


@pytest.mark.offline
def test_drive_file_metadata_trash_move_and_rename_keep_request_shapes(monkeypatch):
    service = _Service()
    service.api.get_results = [
        {"name": "Doc", "parents": ["parent", "second"], "trashed": False},
        {"name": "Deleted", "parents": ["trash"], "trashed": True},
        {"parents": ["old", "shared"]},
    ]

    def get(**kwargs):
        service.api.calls.append(("get", kwargs))
        return _Request(service.api.get_results.pop(0))

    monkeypatch.setattr(service.api, "get", get)
    read_builds = []
    write_builds = []
    monkeypatch.setattr(files, "build_drive_service", lambda: read_builds.append(True) or service)
    monkeypatch.setattr(files, "build_drive_service_rw", lambda: write_builds.append(True) or service)

    assert files.get_file_parent_folder_id("doc") == ("parent", False, "Doc")
    assert files.get_file_parent_folder_id("deleted") == (None, True, None)
    files.move_file_to_folder("doc", "new")
    files.rename_file("doc", "Renamed")

    assert read_builds == [True, True]
    assert write_builds == [True, True]
    assert service.api.calls == [
        ("get", {"fileId": "doc", "fields": "name,parents,trashed"}),
        ("get", {"fileId": "deleted", "fields": "name,parents,trashed"}),
        ("get", {"fileId": "doc", "fields": "parents"}),
        ("update", {
            "fileId": "doc", "addParents": "new", "removeParents": "old,shared", "fields": "id",
        }),
        ("update", {"fileId": "doc", "body": {"name": "Renamed"}, "fields": "id"}),
    ]


@pytest.mark.offline
def test_changes_pages_accumulate_and_return_latest_cursor(monkeypatch):
    class Changes:
        def __init__(self):
            self.calls = []
            self.responses = [
                {"changes": [{"fileId": "a"}], "nextPageToken": "next"},
                {"changes": [{"fileId": "b"}], "newStartPageToken": "cursor-3"},
            ]

        def getStartPageToken(self):
            return _Request({"startPageToken": "cursor-1"})

        def list(self, **kwargs):
            self.calls.append(kwargs)
            return _Request(self.responses.pop(0))

    class ChangesService:
        def __init__(self):
            self.api = Changes()

        def changes(self):
            return self.api

    service = ChangesService()
    builds = []
    monkeypatch.setattr(changes, "build_drive_service", lambda: builds.append(True) or service)

    assert changes.get_start_page_token() == "cursor-1"
    assert changes.get_drive_changes("cursor-1") == ([{"fileId": "a"}, {"fileId": "b"}], "cursor-3")
    assert builds == [True, True]
    assert service.api.calls == [
        {
            "pageToken": "cursor-1",
            "fields": "nextPageToken,newStartPageToken,changes(removed,fileId,file(id,name,mimeType,modifiedTime,parents,trashed))",
            "spaces": "drive",
            "includeItemsFromAllDrives": False,
        },
        {
            "pageToken": "next",
            "fields": "nextPageToken,newStartPageToken,changes(removed,fileId,file(id,name,mimeType,modifiedTime,parents,trashed))",
            "spaces": "drive",
            "includeItemsFromAllDrives": False,
        },
    ]


@pytest.mark.offline
def test_drive_scan_paginates_and_skips_folder_cycles(monkeypatch):
    class Listing:
        def __init__(self):
            self.calls = []

        def list(self, **kwargs):
            self.calls.append(kwargs)
            query = kwargs["q"]
            page = kwargs.get("pageToken")
            if "'root'" in query and "mimeType='application/vnd.google-apps.folder'" in query:
                response = (
                    {"files": [{"id": "a", "name": "A"}], "nextPageToken": "folder-next"}
                    if page is None
                    else {"files": [{"id": "b", "name": "B"}, {"id": "a", "name": "Cycle"}]}
                )
            elif "mimeType='application/vnd.google-apps.document'" in query:
                response = (
                    {"files": [{"id": "d1", "name": "D1", "createdTime": "c1", "modifiedTime": "m1"}], "nextPageToken": "doc-next"}
                    if page is None and "'root'" in query
                    else {"files": [{"id": "d2", "name": "D2", "createdTime": "c2", "modifiedTime": "m2"}]}
                    if page == "doc-next"
                    else {"files": []}
                )
            else:
                response = {"files": []}
            return _Request(response)

    class ScanService:
        def __init__(self):
            self.api = Listing()

        def files(self):
            return self.api

    service = ScanService()
    monkeypatch.setattr(scanner, "build_drive_service", lambda: service)
    monkeypatch.setattr(scanner, "get_root_folder_id", lambda: "root")

    assert scanner.scan_root_folder() == (
        [
            {"id": "a", "name": "A", "parent_id": "root"},
            {"id": "b", "name": "B", "parent_id": "root"},
        ],
        [
            {"doc_id": "d1", "name": "D1", "parent_id": "root", "createdTime": "c1", "modifiedTime": "m1"},
            {"doc_id": "d2", "name": "D2", "parent_id": "root", "createdTime": "c2", "modifiedTime": "m2"},
        ],
    )
    assert [call.get("pageToken") for call in service.api.calls[:2]] == [None, "folder-next"]
    assert any(call.get("pageToken") == "doc-next" for call in service.api.calls)

