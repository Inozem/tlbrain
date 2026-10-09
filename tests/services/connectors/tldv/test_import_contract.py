from __future__ import annotations

import importlib
from types import SimpleNamespace

import httpx
import pytest
from google.auth.credentials import AnonymousCredentials
from google.oauth2.credentials import Credentials
from google.cloud import tasks_v2
import requests


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


def _request(app, payload):
    async def request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://tldv-import.test") as client:
            return await client.post("/import", json=payload)

    return _run(request())


@pytest.fixture
def import_main(app_root, monkeypatch, sdk_fakes):
    del sdk_fakes  # conftest installs the Google SDK boundary fakes before import.
    monkeypatch.syspath_prepend(str(app_root / "services" / "connectors" / "tldv"))
    return importlib.import_module("services.connectors.tldv.import_service.app.main")


def _seed(firestore_fake, collection, doc_id, data):
    from google.cloud import firestore

    firestore.Client().collection(collection).document(doc_id).set(data)


def _placeholder(firestore_fake, meeting_id):
    return firestore_fake.documents.get(f"transcript_index/tldv-{meeting_id}")


def _http_error(status_code):
    return requests.exceptions.HTTPError(
        f"scripted HTTP {status_code}",
        response=SimpleNamespace(status_code=status_code),
    )


@pytest.mark.parametrize("meeting_id", [None, ""])
def test_import_requires_truthy_meeting_id(import_main, monkeypatch, meeting_id):
    monkeypatch.setattr(import_main, "tldv_get", lambda _path: pytest.fail("provider fetch must not run"))

    response = _request(import_main.app, {"meeting_id": meeting_id})

    assert response.status_code == 400
    assert response.json() == {"error": "missing meeting_id"}


def test_import_skips_existing_nonqueued_record_without_fetch(
    import_main, firestore_fake, monkeypatch
):
    _seed(firestore_fake, "transcript_index", "existing-doc", {"meeting_id": "meeting-1", "status": "imported"})
    monkeypatch.setattr(import_main, "tldv_get", lambda _path: pytest.fail("provider fetch must not run"))

    response = _request(import_main.app, {"meeting_id": "meeting-1"})

    assert response.status_code == 200
    assert response.json() == {"ok": True, "skipped": True}


@pytest.mark.parametrize(
    ("endpoint", "status", "expected", "placeholder_after"),
    [
        ("meeting", 403, {"ok": True, "skipped": True, "reason": "meeting_http_403"},
         {"status": "error", "error": "HTTP 403 fetching meeting", "error_stage": "import"}),
        ("meeting", 404, {"ok": True, "skipped": True, "reason": "meeting_not_found"}, None),
        ("transcript", 403, {"ok": True, "skipped": True, "reason": "transcript_http_403"},
         {"status": "error", "error": "HTTP 403 fetching transcript", "error_stage": "import"}),
        ("transcript", 404, {"ok": True, "skipped": True, "reason": "transcript_http_404"},
         {"status": "error", "error": "HTTP 404 fetching transcript", "error_stage": "import"}),
    ],
)
def test_import_http_errors_update_or_remove_queued_placeholder(
    import_main, firestore_fake, monkeypatch, endpoint, status, expected, placeholder_after
):
    meeting_id = f"meeting-{endpoint}-{status}"
    _seed(firestore_fake, "transcript_index", f"tldv-{meeting_id}", {"meeting_id": meeting_id, "status": "queued"})
    calls = []

    def provider(path):
        calls.append(path)
        if endpoint == "meeting" or len(calls) == 2:
            raise _http_error(status)
        return {"name": "Synthetic meeting", "happenedAt": "2025-02-03T10:15:00Z"}

    monkeypatch.setattr(import_main, "tldv_get", provider)
    monkeypatch.delenv("VECTOR_SYNC_URL", raising=False)
    monkeypatch.delenv("VECTOR_SYNC_QUEUE", raising=False)

    response = _request(import_main.app, {"meeting_id": meeting_id})

    assert response.status_code == 200
    assert response.json() == expected
    expected_calls = [f"/meetings/{meeting_id}"]
    if endpoint == "transcript":
        expected_calls.append(f"/meetings/{meeting_id}/transcript")
    assert calls == expected_calls
    actual_placeholder = _placeholder(firestore_fake, meeting_id)
    if placeholder_after is None:
        assert actual_placeholder is None
    else:
        for key, value in placeholder_after.items():
            assert actual_placeholder[key] == value


def test_import_empty_transcript_marks_placeholder_error(import_main, firestore_fake, monkeypatch):
    meeting_id = "meeting-empty"
    _seed(firestore_fake, "transcript_index", f"tldv-{meeting_id}", {"meeting_id": meeting_id, "status": "queued"})
    monkeypatch.setattr(import_main, "tldv_get", lambda path: (
        {"name": "Synthetic meeting", "happenedAt": "2025-02-03T10:15:00Z"}
        if path == f"/meetings/{meeting_id}" else {"results": []}
    ))
    monkeypatch.delenv("VECTOR_SYNC_URL", raising=False)
    monkeypatch.delenv("VECTOR_SYNC_QUEUE", raising=False)

    response = _request(import_main.app, {"meeting_id": meeting_id})

    assert response.status_code == 200
    assert response.json() == {"ok": True, "skipped": True, "reason": "empty_transcript"}
    assert _placeholder(firestore_fake, meeting_id)["status"] == "error"
    assert _placeholder(firestore_fake, meeting_id)["error"] == "empty transcript"
    assert _placeholder(firestore_fake, meeting_id)["error_stage"] == "import"


def test_build_credentials_refreshes_oauth_credentials(import_main, monkeypatch, sdk_fakes):
    monkeypatch.setenv("GOOGLE_REFRESH_TOKEN", "synthetic-refresh-token")
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "synthetic-client-id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "synthetic-client-secret")

    credentials = import_main._build_credentials()

    assert isinstance(credentials, Credentials)
    assert credentials.refresh_token == "synthetic-refresh-token"
    assert credentials.client_id == "synthetic-client-id"
    assert credentials.client_secret == "synthetic-client-secret"
    assert credentials.token == "pytest-synthetic-access-token"
    assert [(entry["boundary"], entry["operation"]) for entry in sdk_fakes.trace] == [
        ("auth", "credentials.refresh")
    ]


def test_build_credentials_uses_adc_when_refresh_token_is_absent(import_main, monkeypatch, sdk_fakes):
    monkeypatch.delenv("GOOGLE_REFRESH_TOKEN", raising=False)

    credentials = import_main._build_credentials()

    assert isinstance(credentials, AnonymousCredentials)
    assert [(entry["boundary"], entry["operation"]) for entry in sdk_fakes.trace] == [("auth", "default")]
    assert sdk_fakes.trace[0]["arguments"]["scopes"] == import_main._SCOPES


def _record_success_effect_order(monkeypatch, doc_id, folder_id, meeting_id):
    from tests.fixtures import firestore_fake as firestore_module
    from tests.fixtures import sdk_fakes as sdk_module

    effects = []
    original_record = sdk_module._record

    def record(boundary, operation, **arguments):
        if boundary == "drive" and operation == "files.create":
            if arguments.get("body", {}).get("mimeType") == "application/vnd.google-apps.document":
                effects.append("drive.doc_create")
        elif boundary == "docs" and operation == "documents.batchUpdate":
            effects.append("docs.batch_update")
        elif boundary == "tasks" and operation == "create_task":
            effects.append("tasks.create")
        original_record(boundary, operation, **arguments)

    monkeypatch.setattr(sdk_module, "_record", record)

    for method_name in ("set", "update", "delete"):
        original = getattr(firestore_module.FakeDocumentReference, method_name)

        def wrapped(self, *args, _method=method_name, _original=original, **kwargs):
            if self.path == f"transcript_index/{doc_id}" and _method == "set":
                effects.append("firestore.transcript_set")
            elif self.path == f"folders/{folder_id}" and _method == "update":
                effects.append("firestore.speaker_increment")
            elif self.path == f"transcript_index/tldv-{meeting_id}" and _method == "delete":
                effects.append("firestore.placeholder_delete")
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(firestore_module.FakeDocumentReference, method_name, wrapped)
    return effects


def test_success_import_writes_canonical_doc_and_enqueues_named_sync_task(
    import_main, firestore_fake, sdk_fakes, monkeypatch
):
    meeting_id = "meeting-success"
    doc_id = "doc-success"
    folder_id = "folder-alex"
    root_id = "pytest-synthetic-root"
    speaker = "Alex"
    speaker_key = import_main._speaker_key(speaker)
    _seed(firestore_fake, "transcript_index", f"tldv-{meeting_id}", {"meeting_id": meeting_id, "status": "queued"})
    _seed(firestore_fake, "folders", folder_id, {
        "folder_id": folder_id,
        "name": "Alex folder",
        "parent_id": root_id,
        "speakers": {speaker_key: 4},
    })
    sdk_fakes.drive.queue("files.create", response={"id": doc_id, "modifiedTime": "2025-02-03T10:20:00Z"})
    sdk_fakes.docs.queue("documents.batchUpdate", response={})
    monkeypatch.setenv("VECTOR_SYNC_URL", "https://sync.synthetic.test")
    monkeypatch.setenv("VECTOR_SYNC_QUEUE", "sync-queue")
    meeting = {"id": meeting_id, "name": "Contract meeting", "happenedAt": "2025-02-03T10:15:00Z"}
    utterances = [
        {"speaker": "Alex::", "text": "First line."},
        {"speaker": "Alex", "text": "Second line."},
    ]
    monkeypatch.setattr(import_main, "tldv_get", lambda path: meeting if path == f"/meetings/{meeting_id}" else {"results": utterances})
    effects = _record_success_effect_order(monkeypatch, doc_id, folder_id, meeting_id)

    response = _request(import_main.app, {"meeting_id": meeting_id})

    assert response.status_code == 200
    assert response.json() == {"ok": True, "doc_id": doc_id, "meeting_id": meeting_id}
    expected_text = (
        "DATE: 2025-02-03\n"
        "TIME: 10:15\n"
        "PROVIDER: tldv\n"
        "---\n"
        "Alex :: First line.\n\n"
        "Alex :: Second line."
    )
    docs_call = next(
        event for event in sdk_fakes.trace
        if event["boundary"] == "docs" and event["operation"] == "documents.batchUpdate"
    )
    requests_body = docs_call["arguments"]["body"]["requests"]
    assert requests_body == [
        {"insertText": {"location": {"index": 1}, "text": expected_text}},
        {"updateTextStyle": {
            "range": {"startIndex": 49, "endIndex": 53},
            "textStyle": {"bold": True},
            "fields": "bold",
        }},
        {"updateTextStyle": {
            "range": {"startIndex": 70, "endIndex": 74},
            "textStyle": {"bold": True},
            "fields": "bold",
        }},
    ]
    doc_call = next(
        event for event in sdk_fakes.trace
        if event["boundary"] == "drive" and event["operation"] == "files.create"
        and event["arguments"]["body"].get("mimeType") == "application/vnd.google-apps.document"
    )
    assert doc_call["arguments"]["body"] == {
        "name": "Contract meeting",
        "mimeType": "application/vnd.google-apps.document",
        "parents": [folder_id],
    }
    index = firestore_fake.documents[f"transcript_index/{doc_id}"]
    assert {key: index[key] for key in (
        "doc_id", "parent_id", "dialog_date", "provider", "source_file", "meeting_id",
        "speakers", "modifiedTime", "status", "error",
    )} == {
        "doc_id": doc_id,
        "parent_id": folder_id,
        "dialog_date": "2025-02-03",
        "provider": "tldv",
        "source_file": "Contract meeting",
        "meeting_id": meeting_id,
        "speakers": [speaker],
        "modifiedTime": "2025-02-03T10:20:00Z",
        "status": "imported",
        "error": None,
    }
    assert index["status_changed_at"] is not None
    assert _placeholder(firestore_fake, meeting_id) is None
    assert firestore_fake.documents[f"folders/{folder_id}"]["speakers"][speaker_key] == 5
    task_event = next(
        event for event in sdk_fakes.trace
        if event["boundary"] == "tasks" and event["operation"] == "create_task"
    )
    assert task_event["arguments"]["parent"] == "projects/pytest-synthetic-project/locations/pytest-synthetic-region/queues/sync-queue"
    assert task_event["arguments"]["task"] == {
        "http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            "url": f"https://sync.synthetic.test/sync/doc/{doc_id}",
        },
        "name": "projects/pytest-synthetic-project/locations/pytest-synthetic-region/queues/sync-queue/tasks/doc-success",
    }
    assert effects == [
        "drive.doc_create",
        "docs.batch_update",
        "firestore.transcript_set",
        "firestore.speaker_increment",
        "firestore.placeholder_delete",
        "tasks.create",
    ]


def test_success_without_sync_configuration_keeps_imported_doc_and_skips_task(
    import_main, firestore_fake, sdk_fakes, monkeypatch
):
    meeting_id = "meeting-no-sync"
    doc_id = "doc-no-sync"
    folder_id = "folder-no-sync"
    _seed(firestore_fake, "transcript_index", f"tldv-{meeting_id}", {"meeting_id": meeting_id, "status": "queued"})
    _seed(firestore_fake, "folders", folder_id, {
        "folder_id": folder_id,
        "name": "_unassigned",
        "parent_id": "pytest-synthetic-root",
        "speakers": {},
    })
    sdk_fakes.drive.queue("files.create", response={"id": doc_id, "modifiedTime": "synthetic-time"})
    sdk_fakes.docs.queue("documents.batchUpdate", response={})
    monkeypatch.delenv("VECTOR_SYNC_URL", raising=False)
    monkeypatch.delenv("VECTOR_SYNC_QUEUE", raising=False)
    monkeypatch.setattr(import_main, "tldv_get", lambda path: (
        {"id": meeting_id, "name": "No sync", "happenedAt": "2025-02-03T10:15:00Z"}
        if path == f"/meetings/{meeting_id}"
        else {"results": [{"speaker": "Speaker 1", "text": "Text."}]}
    ))

    response = _request(import_main.app, {"meeting_id": meeting_id})

    assert response.status_code == 200
    assert response.json() == {"ok": True, "doc_id": doc_id, "meeting_id": meeting_id}
    assert firestore_fake.documents[f"transcript_index/{doc_id}"]["status"] == "imported"
    assert not [entry for entry in sdk_fakes.trace if entry["boundary"] == "tasks" and entry["operation"] == "create_task"]
