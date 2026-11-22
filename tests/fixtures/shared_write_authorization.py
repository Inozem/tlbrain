"""Stateful external SDK boundaries for shared write authorization tests."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

from googleapiclient.errors import HttpError
import httplib2

ROOT_FOLDER_ID = "pytest-synthetic-root"


def _copy(value):
    return copy.deepcopy(value)


def _filter_values(filter_, key):
    values = None
    for condition in getattr(filter_, "must", []) or []:
        if getattr(condition, "key", None) != key:
            continue
        match = getattr(condition, "match", None)
        any_values = getattr(match, "any", None)
        value = getattr(match, "value", None)
        if any_values is not None:
            values = set(any_values) if values is None else values & set(any_values)
        elif value is not None:
            values = {value} if values is None else values & {value}
    return values


class SharedWriteAuthorizationHarness:
    """Model Drive and Qdrant effects while leaving shared/domain code real."""

    def __init__(self, monkeypatch, sdk_fakes):
        from tests.fixtures import sdk_fakes as sdk_module

        self.files: dict[str, dict[str, Any]] = {}
        monkeypatch.setenv(
            "ROOT_FOLDER_URL",
            f"https://drive.google.com/drive/folders/{ROOT_FOLDER_ID}",
        )
        self.drive_mutations: list[dict[str, Any]] = []
        self.drive_reads: list[str] = []
        self.drive_errors: list[tuple[str, int]] = []
        self.on_drive_read = None
        self.qdrant_points: dict[tuple[str, str], list[str | None]] = {}
        self.qdrant_mutations: list[dict[str, Any]] = []
        self.set_file(ROOT_FOLDER_ID, name="Root", mime_type="application/vnd.google-apps.folder", parents=[])
        self.set_file("outside-root", name="Outside", mime_type="application/vnd.google-apps.folder",
                      parents=["outside-root-parent"])
        self.set_file("outside-root-parent", name="Outside ancestor",
                      mime_type="application/vnd.google-apps.folder", parents=[])
        original_execute = sdk_module._Request.execute

        def execute(request, **kwargs):
            if request.boundary == "drive" and request.operation == "files.get":
                file_id = request.args["fileId"]
                self.drive_reads.append(file_id)
                value = self.files.get(file_id)
                if self.on_drive_read is not None:
                    self.on_drive_read(file_id, value, len(self.drive_reads))
                if isinstance(value, BaseException):
                    status = getattr(value, "status_code", None)
                    if isinstance(status, int):
                        self.drive_errors.append((file_id, status))
                    raise value
                if value is None:
                    error = HttpError(httplib2.Response({"status": "404"}), b"missing")
                    self.drive_errors.append((file_id, 404))
                    raise error
                return _copy(value)
            if request.boundary == "drive" and request.operation == "files.update":
                self.drive_mutations.append(_copy(request.args))
                file_id = request.args["fileId"]
                file = self.files.setdefault(file_id, {"id": file_id, "parents": []})
                body = request.args.get("body") or {}
                file.update(body)
                if request.args.get("addParents"):
                    file["parents"] = [request.args["addParents"]]
                return _copy(file)
            if request.boundary == "drive" and request.operation == "files.create":
                self.drive_mutations.append(_copy(request.args))
                body = _copy(request.args.get("body") or {})
                file_id = body.get("id", "created-folder")
                body["id"] = file_id
                body.setdefault("trashed", False)
                self.files[file_id] = body
                return {"id": file_id}
            return original_execute(request, **kwargs)

        monkeypatch.setattr(sdk_module._Request, "execute", execute)

        from tests.fixtures.sdk_fakes import FakeQdrantClient

        def scroll(_client, **kwargs):
            scroll_filter = kwargs.get("scroll_filter")
            doc_ids = _filter_values(scroll_filter, "doc_id")
            root_ids = _filter_values(scroll_filter, "root_folder_id")
            points = []
            for (doc_id, root_id), parents in sorted(self.qdrant_points.items()):
                if doc_ids is not None and doc_id not in doc_ids:
                    continue
                if root_ids is not None and root_id not in root_ids:
                    continue
                for parent_id in parents:
                    points.append(SimpleNamespace(payload={
                        "doc_id": doc_id, "root_folder_id": root_id, "parent_id": parent_id,
                    }))
            return points, None

        def set_payload(_client, **kwargs):
            self.qdrant_mutations.append({"operation": "set_payload", **_copy(kwargs)})
            payload = kwargs.get("payload", {})
            points_filter = kwargs.get("points")
            doc_ids = _filter_values(points_filter, "doc_id")
            root_ids = _filter_values(points_filter, "root_folder_id")
            for (doc_id, root_id), parents in list(self.qdrant_points.items()):
                if doc_ids is not None and doc_id not in doc_ids:
                    continue
                if root_ids is not None and root_id not in root_ids:
                    continue
                self.qdrant_points[(doc_id, root_id)] = [payload.get("parent_id", p) for p in parents]
            return True

        def other_qdrant_mutation(operation, **kwargs):
            self.qdrant_mutations.append({"operation": operation, **_copy(kwargs)})
            return True

        original_getattr = FakeQdrantClient.__getattr__

        def qdrant_method(client, operation):
            if operation == "scroll":
                return lambda **kwargs: scroll(client, **kwargs)
            if operation == "set_payload":
                return lambda **kwargs: set_payload(client, **kwargs)
            if operation in {"upsert", "delete"}:
                return lambda **kwargs: other_qdrant_mutation(operation, **kwargs)
            return original_getattr(client, operation)

        monkeypatch.setattr(FakeQdrantClient, "__getattr__", qdrant_method)
        # The fixture's API remains the real SDK fake controller; this assertion
        # catches accidental construction with an unrelated harness.
        assert sdk_fakes is not None

    def set_file(self, file_id: str, *, name: str, mime_type: str, parents: list[str],
                 trashed: bool = False) -> dict[str, Any]:
        value = {"id": file_id, "name": name, "mimeType": mime_type,
                 "parents": list(parents), "trashed": trashed, "appProperties": {}}
        self.files[file_id] = value
        return value

    def add_qdrant_parent(self, doc_id: str, parent_id: str | None,
                          root_id: str = ROOT_FOLDER_ID) -> None:
        self.qdrant_points.setdefault((doc_id, root_id), []).append(parent_id)

    def snapshot_domain(self, client) -> dict[str, Any]:
        def records(collection):
            return {doc.id: _copy(doc.to_dict() or {}) for doc in client.collection(collection).stream()}

        drive_state = {}
        for file_id, value in self.files.items():
            if isinstance(value, BaseException):
                drive_state[file_id] = {
                    "error_type": type(value).__name__,
                    "status": getattr(value, "status_code", None),
                }
            else:
                drive_state[file_id] = _copy(value)

        return {
            "drive": drive_state,
            "transcript_index": records("transcript_index"),
            "folders": records("folders"),
            "qdrant": _copy(self.qdrant_points),
        }
