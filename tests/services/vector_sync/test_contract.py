from __future__ import annotations

import importlib

import httpx
import pytest
from googleapiclient.errors import HttpError
from qdrant_client.models import FieldCondition, Filter, MatchValue

from core.integrations.qdrant import client as qdrant_client
from tests.fixtures.sdk_fakes import FakeQdrantClient


def _run(coroutine):
    import sniffio

    token = sniffio.current_async_library_cvar.set("asyncio")
    try:
        try:
            coroutine.send(None)
        except StopIteration as completed:
            return completed.value
    finally:
        sniffio.current_async_library_cvar.reset(token)
        coroutine.close()
    raise AssertionError("in-process ASGI coroutine unexpectedly suspended")


def _request(app, doc_id="doc-1"):
    async def request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://vector-sync.test") as client:
            return await client.post(f"/sync/doc/{doc_id}")

    return _run(request())


def _main():
    return importlib.import_module("services.vector_sync.app.main")


def _seed(collection, doc_id, data):
    from google.cloud import firestore

    firestore.Client().collection(collection).document(doc_id).set(data)


def _index_exists(firestore_fake, doc_id):
    return f"transcript_index/{doc_id}" in firestore_fake.documents


@pytest.fixture
def vector_main(sdk_fakes):
    # The external SDK boundary fake is installed by conftest before this import.
    del sdk_fakes
    return _main()


@pytest.mark.parametrize("result", [
    "processed",
    "skipped",
    "not_acquired",
])
def test_route_wraps_each_success_result(vector_main, monkeypatch, result):
    monkeypatch.setattr(vector_main, "load_index", lambda _doc_id: None)
    monkeypatch.setattr(vector_main, "get_file_parent_folder_id", lambda _doc_id: ("folder-1", False, "sample.docx"))
    monkeypatch.setattr(vector_main, "get_folder_by_id", lambda _folder_id: {"folder_id": "folder-1"})
    monkeypatch.setattr(vector_main, "read_google_doc", lambda _doc_id: "synthetic transcript")
    monkeypatch.setattr(vector_main, "is_valid_format", lambda _text: True)
    monkeypatch.setattr(vector_main, "ensure_imported", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(vector_main, "process_one", lambda *_args, **_kwargs: result)

    response = _request(vector_main.app)

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "result": result}


def test_process_one_exception_has_controlled_http_500(vector_main, monkeypatch):
    monkeypatch.setattr(vector_main, "load_index", lambda _doc_id: None)
    monkeypatch.setattr(vector_main, "get_file_parent_folder_id", lambda _doc_id: ("folder-1", False, "sample.docx"))
    monkeypatch.setattr(vector_main, "get_folder_by_id", lambda _folder_id: {"folder_id": "folder-1"})
    monkeypatch.setattr(vector_main, "read_google_doc", lambda _doc_id: "synthetic transcript")
    monkeypatch.setattr(vector_main, "is_valid_format", lambda _text: True)
    monkeypatch.setattr(vector_main, "ensure_imported", lambda *_args, **_kwargs: None)

    def fail(*_args, **_kwargs):
        raise RuntimeError("scripted processor failure")

    monkeypatch.setattr(vector_main, "process_one", fail)
    response = _request(vector_main.app)
    assert response.status_code == 500
    assert response.json() == {"status": "error", "details": "scripted processor failure"}


def _drive_missing(_doc_id):
    raise HttpError(type("Response", (), {"status": 404, "reason": "Not Found"})(), b"missing")


@pytest.mark.parametrize("trigger", ["drive_404", "trashed"])
def test_drive_deletion_removes_scoped_qdrant_points_and_firestore_index(
    vector_main, sdk_fakes, firestore_fake, monkeypatch, trigger
):
    doc_id = "doc-to-delete"
    _seed("transcript_index", doc_id, {"doc_id": doc_id, "status": "synced"})
    monkeypatch.setattr(vector_main, "load_index", lambda _doc_id: {"doc_id": doc_id})
    if trigger == "drive_404":
        monkeypatch.setattr(vector_main, "get_file_parent_folder_id", _drive_missing)
    else:
        monkeypatch.setattr(vector_main, "get_file_parent_folder_id", lambda _doc_id: ("folder-1", True, "gone.docx"))

    response = _request(vector_main.app, doc_id)

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "result": "deleted"}
    assert not _index_exists(firestore_fake, doc_id)
    deletion = next(entry for entry in sdk_fakes.trace if entry["boundary"] == "qdrant" and entry["operation"] == "delete")
    selector = deletion["arguments"]["points_selector"]
    assert isinstance(selector, Filter)
    assert selector.must == [
        FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
        FieldCondition(key="root_folder_id", match=MatchValue(value="pytest-synthetic-root")),
    ]
    assert selector.must_not is None


@pytest.mark.parametrize("trigger", ["drive_404", "trashed"])
def test_drive_deletion_without_index_does_not_write(vector_main, sdk_fakes, monkeypatch, trigger):
    monkeypatch.setattr(vector_main, "load_index", lambda _doc_id: None)
    if trigger == "drive_404":
        monkeypatch.setattr(vector_main, "get_file_parent_folder_id", _drive_missing)
    else:
        monkeypatch.setattr(vector_main, "get_file_parent_folder_id", lambda _doc_id: ("folder-1", True, "gone.docx"))

    response = _request(vector_main.app)

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "result": "deleted"}
    assert not [entry for entry in sdk_fakes.trace if entry["boundary"] == "qdrant" and entry["operation"] == "delete"]


@pytest.mark.parametrize("has_index", [False, True])
def test_unregistered_parent_skips_or_deletes_existing_index(vector_main, firestore_fake, sdk_fakes, monkeypatch, has_index):
    doc_id = "doc-outside-folder"
    if has_index:
        _seed("transcript_index", doc_id, {"doc_id": doc_id, "status": "synced"})
    monkeypatch.setattr(vector_main, "load_index", lambda _doc_id: {"doc_id": doc_id} if has_index else None)
    monkeypatch.setattr(vector_main, "get_file_parent_folder_id", lambda _doc_id: ("unknown-folder", False, "sample.docx"))

    response = _request(vector_main.app, doc_id)

    expected = "deleted" if has_index else "skipped_no_folder"
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "result": expected}
    assert not _index_exists(firestore_fake, doc_id)
    deletes = [entry for entry in sdk_fakes.trace if entry["boundary"] == "qdrant" and entry["operation"] == "delete"]
    assert bool(deletes) is has_index


def test_unregistered_root_parent_is_also_skipped(vector_main, monkeypatch):
    monkeypatch.setattr(vector_main, "load_index", lambda _doc_id: None)
    monkeypatch.setattr(vector_main, "get_file_parent_folder_id", lambda _doc_id: (vector_main.get_root_folder_id(), False, "sample.docx"))
    response = _request(vector_main.app)
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "result": "skipped_no_folder"}


@pytest.mark.parametrize("has_index", [False, True])
def test_invalid_format_only_marks_error_for_existing_index(vector_main, firestore_fake, monkeypatch, has_index):
    doc_id = "doc-invalid"
    if has_index:
        _seed("transcript_index", doc_id, {"doc_id": doc_id, "status": "synced"})
    monkeypatch.setattr(vector_main, "load_index", lambda _doc_id: {"doc_id": doc_id} if has_index else None)
    monkeypatch.setattr(vector_main, "get_file_parent_folder_id", lambda _doc_id: ("folder-1", False, "sample.docx"))
    monkeypatch.setattr(vector_main, "get_folder_by_id", lambda _folder_id: {"folder_id": "folder-1"})
    monkeypatch.setattr(vector_main, "read_google_doc", lambda _doc_id: "bad format")
    monkeypatch.setattr(vector_main, "is_valid_format", lambda _text: False)

    response = _request(vector_main.app, doc_id)

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "result": "skipped_invalid_format"}
    index = firestore_fake.documents.get(f"transcript_index/{doc_id}")
    if has_index:
        assert index["status"] == "error"
        assert index["error"] == "invalid TLBrain document format"
        assert index["error_stage"] == "invalid_format"
    else:
        assert index is None


def test_import_runs_real_collection_setup_with_sdk_fake(sdk_fakes, monkeypatch):
    monkeypatch.setenv("QDRANT_COLLECTION", "vector-sync-import-test")
    monkeypatch.setattr(qdrant_client, "_client", None)
    sdk_fakes.reset()

    vector_main = importlib.reload(_main())

    assert vector_main.app is not None
    assert [entry["operation"] for entry in sdk_fakes.trace[:4]] == [
        "client", "get_collections", "create_collection", "get_collection"
    ]
    assert any(entry["operation"] == "create_payload_index" for entry in sdk_fakes.trace)
    assert isinstance(qdrant_client.get_client(), FakeQdrantClient)




