from __future__ import annotations

import importlib
import json

import httpx
import pytest

from core.indexing.hashing import sha256_text, sha256_utterance
from core.parsing.parser import parse_document


DOC_ID = "synthetic-vector-sync-parent-move"
OLD_PARENT_ID = "synthetic-folder-before-move"
NEW_PARENT_ID = "synthetic-folder-after-move"
INDEX_PATH = f"transcript_index/{DOC_ID}"


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


def _request(app):
    async def request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://vector-sync.test") as client:
            return await client.post(f"/sync/doc/{DOC_ID}")

    return _run(request())


def _index_record(raw: str) -> dict:
    metadata, utterances = parse_document(raw)
    content_hash = sha256_text(raw)
    return {
        "doc_id": DOC_ID,
        "status": "imported",
        "parent_id": OLD_PARENT_ID,
        "parent_id_hash": sha256_text(OLD_PARENT_ID),
        "content_hash": content_hash,
        "version": content_hash,
        "dialog_date": metadata.get("date", ""),
        "provider": metadata.get("provider", ""),
        "utterance_hashes": {
            str(item["order_index"]): sha256_utterance(item["order_index"], item["speaker"], item["text"])
            for item in utterances
        },
    }


def _write_events(sdk_fakes):
    return [
        event for event in sdk_fakes.trace
        if event["boundary"] == "qdrant" and event["operation"] in {"set_payload", "upsert", "delete"}
    ]


def _embedding_events(sdk_fakes):
    return [
        event for event in sdk_fakes.trace
        if (event["boundary"] == "gemini" and event["operation"] in {"embed_content", "generate_content"})
        or event["boundary"] == "sparse"
    ]


@pytest.fixture
def vector_main(sdk_fakes):
    del sdk_fakes
    return importlib.import_module("services.vector_sync.app.main")


def test_parent_move_recovery_and_repeat_through_service_route(
    app_root, vector_main, sdk_fakes, firestore_fake, monkeypatch,
):
    from google.cloud import firestore

    raw = json.loads((app_root / "tests/fixtures/transcripts.json").read_text(encoding="utf-8"))["cases"]["valid_full"]["raw"]
    firestore.Client().collection("transcript_index").document(DOC_ID).set(_index_record(raw))
    monkeypatch.setattr(
        vector_main,
        "get_file_parent_folder_id",
        lambda _doc_id: (NEW_PARENT_ID, False, "sample.docx"),
    )
    monkeypatch.setattr(vector_main, "get_folder_by_id", lambda _parent_id: {"folder_id": NEW_PARENT_ID})
    monkeypatch.setattr(vector_main, "read_google_doc", lambda _doc_id: raw)

    first_response = _request(vector_main.app)

    assert first_response.status_code == 200
    assert first_response.json() == {"status": "ok", "result": "skipped"}
    first_writes = _write_events(sdk_fakes)
    assert len(first_writes) == 1
    assert first_writes[0]["operation"] == "set_payload"
    assert first_writes[0]["arguments"]["payload"] == {"parent_id": NEW_PARENT_ID}
    assert firestore_fake.documents[INDEX_PATH]["parent_id"] == NEW_PARENT_ID
    assert firestore_fake.documents[INDEX_PATH]["parent_id_hash"] == sha256_text(NEW_PARENT_ID)
    assert _embedding_events(sdk_fakes) == []

    second_response = _request(vector_main.app)

    assert second_response.status_code == 200
    assert second_response.json() == {"status": "ok", "result": "skipped"}
    assert _write_events(sdk_fakes) == first_writes
    assert _embedding_events(sdk_fakes) == []
