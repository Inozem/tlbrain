from __future__ import annotations

import importlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import httplib2
import pytest
from google.auth.credentials import AnonymousCredentials
from google.oauth2.credentials import Credentials
from google.auth.exceptions import RefreshError
from google.cloud import tasks_v2
from googleapiclient.errors import HttpError
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


def _google_api_error(status):
    return HttpError(
        httplib2.Response({"status": str(status), "reason": "synthetic provider detail"}),
        b'{"error":{"message":"synthetic-provider-body"}}',
        uri="https://google.synthetic.test/path?token=synthetic-url-secret",
    )


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


@pytest.mark.parametrize("provider", ["tldv", None])
def test_import_deduplicates_tldv_and_legacy_records(import_main, firestore_fake, monkeypatch, provider):
    record = {"meeting_id": "meeting-1", "status": "imported"}
    if provider is not None:
        record["provider"] = provider
    _seed(firestore_fake, "transcript_index", "existing-doc", record)
    monkeypatch.setattr(import_main, "tldv_get", lambda _path: pytest.fail("provider fetch must not run"))

    response = _request(import_main.app, {"meeting_id": "meeting-1"})

    assert response.status_code == 200
    assert response.json() == {"ok": True, "skipped": True}


def test_import_ignores_foreign_matching_record_and_preserves_reserved_collision(
    import_main, firestore_fake, sdk_fakes, monkeypatch
):
    meeting_id = "foreign-matching"
    foreign = {"meeting_id": meeting_id, "provider": "telegram", "status": "imported", "keep": True}
    _seed(firestore_fake, "transcript_index", f"tldv-{meeting_id}", foreign)
    _seed(firestore_fake, "folders", "folder-import", {
        "folder_id": "folder-import", "name": "_unassigned", "parent_id": "pytest-synthetic-root",
        "speakers": {},
    })
    sdk_fakes.drive.queue("files.create", response={"id": "doc-imported", "modifiedTime": "synthetic-time"})
    sdk_fakes.docs.queue("documents.batchUpdate", response={})
    monkeypatch.delenv("VECTOR_SYNC_URL", raising=False)
    monkeypatch.delenv("VECTOR_SYNC_QUEUE", raising=False)
    monkeypatch.setattr(import_main, "tldv_get", lambda path: (
        {"id": meeting_id, "name": "Foreign collision", "happenedAt": "2025-02-03T10:15:00Z"}
        if path == f"/meetings/{meeting_id}"
        else {"results": [{"speaker": "Speaker 1", "text": "Transcript."}]}
    ))

    response = _request(import_main.app, {"meeting_id": meeting_id})

    assert response.status_code == 200
    assert response.json() == {"ok": True, "doc_id": "doc-imported", "meeting_id": meeting_id}
    assert firestore_fake.documents[f"transcript_index/tldv-{meeting_id}"] == foreign
    assert firestore_fake.documents["transcript_index/doc-imported"]["provider"] == "tldv"


def test_import_preserves_imported_record_when_doc_id_matches_reserved_placeholder(
    import_main, firestore_fake, sdk_fakes, monkeypatch
):
    meeting_id = "reserved-doc-id"
    doc_id = f"tldv-{meeting_id}"
    _seed(firestore_fake, "transcript_index", doc_id, {
        "meeting_id": meeting_id, "provider": "tldv", "status": "queued",
    })
    _seed(firestore_fake, "folders", "folder-reserved", {
        "folder_id": "folder-reserved", "name": "_unassigned", "parent_id": "pytest-synthetic-root",
        "speakers": {},
    })
    sdk_fakes.drive.queue("files.create", response={"id": doc_id, "modifiedTime": "synthetic-time"})
    sdk_fakes.docs.queue("documents.batchUpdate", response={})
    monkeypatch.delenv("VECTOR_SYNC_URL", raising=False)
    monkeypatch.delenv("VECTOR_SYNC_QUEUE", raising=False)
    monkeypatch.setattr(import_main, "tldv_get", lambda path: (
        {"id": meeting_id, "name": "Reserved ID", "happenedAt": "2025-02-03T10:15:00Z"}
        if path == f"/meetings/{meeting_id}"
        else {"results": [{"speaker": "Speaker 1", "text": "Transcript."}]}
    ))

    response = _request(import_main.app, {"meeting_id": meeting_id})

    assert response.status_code == 200
    assert response.json() == {"ok": True, "doc_id": doc_id, "meeting_id": meeting_id}
    record = firestore_fake.documents[f"transcript_index/{doc_id}"]
    assert record["doc_id"] == doc_id
    assert record["meeting_id"] == meeting_id
    assert record["provider"] == "tldv"
    assert record["status"] == "imported"


def test_auth_refresh_failure_is_safe_retryable_and_deduplicated(
    import_main, firestore_fake, sdk_fakes, monkeypatch, caplog
):
    meeting_id = "meeting-auth-recovery"
    _seed(firestore_fake, "transcript_index", f"tldv-{meeting_id}", {
        "meeting_id": meeting_id, "provider": "tldv", "status": "queued",
    })
    _seed(firestore_fake, "folders", "folder-auth-recovery", {
        "folder_id": "folder-auth-recovery", "name": "_unassigned",
        "parent_id": "pytest-synthetic-root", "speakers": {},
    })
    sdk_fakes.drive.queue("files.create", response={"id": "doc-auth-recovery", "modifiedTime": "synthetic-time"})
    sdk_fakes.docs.queue("documents.batchUpdate", response={})
    monkeypatch.setattr(import_main, "tldv_get", lambda path: (
        {"name": "Synthetic meeting", "happenedAt": "2025-02-03T10:15:00Z"}
        if path == f"/meetings/{meeting_id}"
        else {"results": [{"speaker": "Speaker 1", "text": "Synthetic transcript."}]}
    ))
    monkeypatch.delenv("VECTOR_SYNC_URL", raising=False)
    monkeypatch.delenv("VECTOR_SYNC_QUEUE", raising=False)
    monkeypatch.setenv("GOOGLE_REFRESH_TOKEN", "synthetic-refresh-token")
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "synthetic-client-id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "synthetic-client-secret")
    original_refresh = Credentials.refresh
    refresh_attempts = []

    def refresh(credentials, request):
        refresh_attempts.append(None)
        if len(refresh_attempts) == 1:
            raise RefreshError("synthetic refresh failure", {"error": "invalid_grant", "description": "synthetic-secret source-text"})
        return original_refresh(credentials, request)

    monkeypatch.setattr(Credentials, "refresh", refresh)

    failed = _request(import_main.app, {"meeting_id": meeting_id})

    assert failed.status_code == 500
    assert failed.json() == {
        "ok": False,
        "error": "Google Drive authorization expired or was revoked (invalid_grant); refresh the configured credentials.",
    }
    assert "synthetic-secret" not in failed.text and "source-text" not in failed.text
    assert "synthetic-secret" not in caplog.text and "source-text" not in caplog.text
    failed_placeholder = _placeholder(firestore_fake, meeting_id)
    assert failed_placeholder["status"] == "error"
    assert failed_placeholder["error_stage"] == "import"
    assert "invalid_grant" in failed_placeholder["error"]
    assert "synthetic-secret" not in failed_placeholder["error"]

    repaired = _request(import_main.app, {"meeting_id": meeting_id})
    replay = _request(import_main.app, {"meeting_id": meeting_id})

    assert repaired.status_code == 200
    assert repaired.json() == {"ok": True, "doc_id": "doc-auth-recovery", "meeting_id": meeting_id}
    assert len(refresh_attempts) >= 2
    assert replay.status_code == 200 and replay.json() == {"ok": True, "skipped": True}
    created = [event for event in sdk_fakes.trace if event["boundary"] == "drive"
               and event["operation"] == "files.create"
               and event["arguments"]["body"].get("mimeType") == "application/vnd.google-apps.document"]
    assert len(created) == 1


def test_refresh_error_classification_requires_structured_oauth_code(import_main):
    assert import_main._safe_failure(
        "authorizing Google Drive",
        RefreshError("invalid_grant", {"error": "invalid_grant", "description": "synthetic-secret"}),
    ) == "Google Drive authorization expired or was revoked (invalid_grant); refresh the configured credentials."
    assert import_main._safe_failure(
        "authorizing Google Drive", RefreshError("temporary refresh failure; invalid_grant text only")
    ) == "Google credentials refresh failed (RefreshError)."
    assert import_main._safe_failure(
        "fetching meeting", RuntimeError("invalid_grant synthetic-secret")
    ) == "TL;DV import failed during fetching meeting (RuntimeError)."


def test_google_http_error_formatter_uses_only_validated_status(import_main):
    error = HttpError(
        SimpleNamespace(status="403", reason="synthetic provider body"),
        b"synthetic-body-marker",
        uri="https://google.synthetic.test/?token=synthetic-url-secret",
    )
    message = import_main._safe_failure("selecting destination folder", error)
    assert "HttpError" in message and "synthetic" not in message and "403" not in message


@pytest.mark.parametrize(
    ("failure_point", "error_type"),
    [
        (point, error_type)
        for point in ("fetch", "drive", "docs")
        for error_type in (RuntimeError, TimeoutError, ConnectionError)
    ],
)
def test_unexpected_import_errors_are_safe_non_success_and_persisted(
    import_main, firestore_fake, sdk_fakes, monkeypatch, failure_point, error_type, caplog
):
    meeting_id = f"meeting-unexpected-{failure_point}-{error_type.__name__}"
    _seed(firestore_fake, "transcript_index", f"tldv-{meeting_id}", {
        "meeting_id": meeting_id, "provider": "tldv", "status": "queued",
    })
    if failure_point == "docs":
        _seed(firestore_fake, "folders", "folder-unexpected-docs", {
            "folder_id": "folder-unexpected-docs", "name": "_unassigned",
            "parent_id": "pytest-synthetic-root", "speakers": {},
        })
        sdk_fakes.drive.queue("files.create", response={"id": "doc-unexpected-docs", "modifiedTime": "synthetic-time"})
        sdk_fakes.docs.queue(
            "documents.batchUpdate", error=error_type("synthetic-secret raw source text")
        )
    elif failure_point == "drive":
        sdk_fakes.drive.queue("files.list", error=error_type("synthetic-secret raw source text"))

    def provider(path):
        if failure_point == "fetch":
            raise error_type("synthetic-secret raw source text")
        return {"name": "Synthetic", "happenedAt": "2025-02-03T10:15:00Z"} if path == f"/meetings/{meeting_id}" else {
            "results": [{"speaker": "Speaker 1", "text": "Synthetic transcript."}],
        }

    monkeypatch.setattr(import_main, "tldv_get", provider)
    monkeypatch.delenv("VECTOR_SYNC_URL", raising=False)
    monkeypatch.delenv("VECTOR_SYNC_QUEUE", raising=False)

    response = _request(import_main.app, {"meeting_id": meeting_id})

    assert response.status_code == 500
    assert response.json()["ok"] is False
    assert "synthetic-secret" not in response.text and "raw source text" not in response.text
    assert "synthetic-secret" not in caplog.text and "raw source text" not in caplog.text
    placeholder = _placeholder(firestore_fake, meeting_id)
    assert placeholder["status"] == "error" and placeholder["error_stage"] == "import"
    assert "synthetic-secret" not in placeholder["error"]
    if failure_point == "docs":
        assert "write outcome unknown" in placeholder["error"]
        replay = _request(import_main.app, {"meeting_id": meeting_id})
        assert replay.status_code == 500
        assert "reconcile the saved document" in replay.json()["error"]
        creates = [event for event in sdk_fakes.trace if event["boundary"] == "drive"
                   and event["operation"] == "files.create"
                   and event["arguments"]["body"].get("mimeType") == "application/vnd.google-apps.document"]
        assert len(creates) == 1


@pytest.mark.parametrize(
    ("operation", "status", "unknown_write"),
    [
        ("files.list", 401, False),
        ("files.list", 403, False),
        ("files.create", 404, True),
        ("files.create", 429, True),
        ("documents.batchUpdate", 500, True),
        ("documents.batchUpdate", 503, True),
    ],
)
def test_google_api_http_failures_are_safe_visible_and_retry_according_to_write_boundary(
    import_main, firestore_fake, sdk_fakes, monkeypatch, caplog, operation, status, unknown_write
):
    meeting_id = f"meeting-google-http-{operation.replace('.', '-')}-{status}"
    _seed(firestore_fake, "transcript_index", f"tldv-{meeting_id}", {
        "meeting_id": meeting_id, "provider": "tldv", "status": "queued",
    })
    if operation != "files.list":
        _seed(firestore_fake, "folders", "folder-google-http", {
            "folder_id": "folder-google-http", "name": "_unassigned",
            "parent_id": "pytest-synthetic-root", "speakers": {},
        })
    if operation == "files.create":
        sdk_fakes.drive.queue("files.create", error=_google_api_error(status))
    elif operation == "documents.batchUpdate":
        sdk_fakes.drive.queue(
            "files.create", response={"id": f"doc-google-http-{status}", "modifiedTime": "synthetic-time"}
        )
        sdk_fakes.docs.queue("documents.batchUpdate", error=_google_api_error(status))
    elif operation == "files.list":
        sdk_fakes.drive.queue("files.list", error=_google_api_error(status))
        sdk_fakes.drive.queue("files.list", response={"files": []})

    def provider(path):
        return {"name": "Synthetic", "happenedAt": "2025-02-03T10:15:00Z"} if path == f"/meetings/{meeting_id}" else {
            "results": [{"speaker": "Speaker 1", "text": "synthetic-source-marker"}],
        }

    monkeypatch.setattr(import_main, "tldv_get", provider)
    monkeypatch.delenv("VECTOR_SYNC_URL", raising=False)
    monkeypatch.delenv("VECTOR_SYNC_QUEUE", raising=False)

    failed = _request(import_main.app, {"meeting_id": meeting_id})

    assert failed.status_code == 500 and failed.json()["ok"] is False
    error = failed.json()["error"]
    assert f"HTTP {status}" in error
    expected_stage = "selecting destination folder" if operation == "files.list" else "creating Google Doc"
    assert f"during {expected_stage}" in error
    assert "synthetic-provider-body" not in failed.text
    assert "synthetic provider detail" not in failed.text
    assert "synthetic-url-secret" not in failed.text
    assert "synthetic-source-marker" not in failed.text
    assert "synthetic-provider-body" not in caplog.text
    assert "synthetic-url-secret" not in caplog.text
    assert "synthetic-source-marker" not in caplog.text
    record = _placeholder(firestore_fake, meeting_id)
    assert record["status"] == "error" and record["error_stage"] == "import"
    assert f"HTTP {status}" in record["error"]
    assert all(secret not in record["error"] for secret in (
        "synthetic-provider-body", "synthetic provider detail", "synthetic-url-secret", "synthetic-source-marker",
    ))

    if unknown_write:
        assert "write outcome unknown" in error
        assert _request(import_main.app, {"meeting_id": meeting_id}).status_code == 500
    else:
        _seed(firestore_fake, "folders", "folder-google-http-repaired", {
            "folder_id": "folder-google-http-repaired", "name": "_unassigned",
            "parent_id": "pytest-synthetic-root", "speakers": {},
        })
        sdk_fakes.drive.queue(
            "files.create", response={"id": f"doc-google-http-repaired-{status}", "modifiedTime": "synthetic-time"}
        )
        sdk_fakes.docs.queue("documents.batchUpdate", response={})
        retried = _request(import_main.app, {"meeting_id": meeting_id})
        assert retried.status_code == 200 and retried.json().get("doc_id")
    create_events = [event for event in sdk_fakes.trace if event["boundary"] == "drive"
                     and event["operation"] == "files.create"]
    assert len(create_events) == 1


def test_fresh_downloading_is_busy_but_stale_download_can_retry(
    import_main, firestore_fake, sdk_fakes, monkeypatch
):
    now = datetime.now(timezone.utc)
    fresh_id, stale_id = "meeting-fresh-download", "meeting-stale-download"
    _seed(firestore_fake, "transcript_index", f"tldv-{fresh_id}", {
        "meeting_id": fresh_id, "provider": "tldv", "status": "downloading",
        "status_changed_at": now - timedelta(minutes=59, seconds=59),
    })
    _seed(firestore_fake, "transcript_index", f"tldv-{stale_id}", {
        "meeting_id": stale_id, "provider": "tldv", "status": "downloading",
        "status_changed_at": now - timedelta(hours=1),
    })
    _seed(firestore_fake, "folders", "folder-stale-download", {
        "folder_id": "folder-stale-download", "name": "_unassigned",
        "parent_id": "pytest-synthetic-root", "speakers": {},
    })
    sdk_fakes.drive.queue("files.create", response={"id": "doc-stale-download", "modifiedTime": "synthetic-time"})
    sdk_fakes.docs.queue("documents.batchUpdate", response={})
    monkeypatch.setattr(import_main, "tldv_get", lambda path: (
        {"name": "Synthetic", "happenedAt": "2025-02-03T10:15:00Z"}
        if path == f"/meetings/{stale_id}"
        else {"results": [{"speaker": "Speaker 1", "text": "Synthetic transcript."}]}
    ))
    monkeypatch.delenv("VECTOR_SYNC_URL", raising=False)
    monkeypatch.delenv("VECTOR_SYNC_QUEUE", raising=False)

    fresh = _request(import_main.app, {"meeting_id": fresh_id})
    stale = _request(import_main.app, {"meeting_id": stale_id})

    assert fresh.status_code == 503 and fresh.json()["ok"] is False
    assert _placeholder(firestore_fake, fresh_id)["status"] == "downloading"
    assert stale.status_code == 200 and stale.json()["doc_id"] == "doc-stale-download"


def test_import_status_persistence_failure_never_returns_success(import_main, firestore_fake, monkeypatch):
    meeting_id = "meeting-status-write-failure"
    path = f"transcript_index/tldv-{meeting_id}"
    _seed(firestore_fake, "transcript_index", f"tldv-{meeting_id}", {
        "meeting_id": meeting_id, "provider": "tldv", "status": "queued",
    })
    def provider(_path):
        firestore_fake.fail_next(
            "commit", path=path, phase="before", exception=RuntimeError("synthetic-secret status payload"),
        )
        raise RuntimeError("synthetic-secret fetch payload")

    monkeypatch.setattr(import_main, "tldv_get", provider)

    response = _request(import_main.app, {"meeting_id": meeting_id})

    assert response.status_code == 500
    assert response.json() == {
        "ok": False,
        "error": "TL;DV import failed and its error status could not be recorded.",
    }
    assert "synthetic-secret" not in response.text
    assert _placeholder(firestore_fake, meeting_id)["status"] == "downloading"


@pytest.mark.parametrize("failure_phase", ["initial_lookup", "diagnostic_lookup"])
def test_firestore_lookup_failures_return_safe_non_success(
    import_main, firestore_fake, monkeypatch, caplog, failure_phase
):
    meeting_id = f"meeting-{failure_phase}-failure"
    calls = 0

    def lookup(_meeting_id, _db):
        nonlocal calls
        calls += 1
        if failure_phase == "initial_lookup" or calls > 1:
            raise RuntimeError("synthetic-secret Firestore response body")
        return []

    monkeypatch.setattr(import_main, "get_index_records_by_meeting_id", lookup)
    if failure_phase == "diagnostic_lookup":
        _seed(firestore_fake, "folders", "folder-read-failure", {
            "folder_id": "folder-read-failure", "name": "_unassigned",
            "parent_id": "pytest-synthetic-root", "speakers": {},
        })
        monkeypatch.setattr(import_main, "_fetch_meeting", lambda *_args: (_ for _ in ()).throw(
            RuntimeError("synthetic-secret provider body")
        ))

    response = _request(import_main.app, {"meeting_id": meeting_id})

    assert response.status_code == 500
    assert "synthetic-secret" not in response.text and "Firestore response body" not in response.text
    assert "synthetic-secret" not in caplog.text and "Firestore response body" not in caplog.text
    if failure_phase == "diagnostic_lookup":
        record = _placeholder(firestore_fake, meeting_id)
        assert record["status"] == "error" and record["error_stage"] == "import"
        assert "synthetic-secret" not in record["error"]


def test_saved_doc_failure_settlement_and_read_errors_stay_safe(
    import_main, firestore_fake, monkeypatch, caplog
):
    meeting_id, doc_id = "meeting-saved-settlement-failure", "doc-saved-settlement-failure"
    _seed(firestore_fake, "transcript_index", doc_id, {
        "doc_id": doc_id, "meeting_id": meeting_id, "provider": "tldv", "status": "imported",
    })
    original_lookup = import_main.get_index_records_by_meeting_id
    calls = 0

    def lookup(mid, db):
        nonlocal calls
        calls += 1
        if calls == 1:
            return []
        return original_lookup(mid, db)

    monkeypatch.setattr(import_main, "get_index_records_by_meeting_id", lookup)
    monkeypatch.setattr(import_main, "_run_new_import", lambda *_args: (_ for _ in ()).throw(
        RuntimeError("synthetic-secret provider failure")
    ))
    monkeypatch.setattr(import_main, "settle_tldv_dispatch", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("synthetic-secret settlement failure")
    ))
    monkeypatch.setattr(import_main, "_saved_doc_has_advanced", lambda *_args: (_ for _ in ()).throw(
        RuntimeError("synthetic-secret diagnostic failure")
    ))

    response = _request(import_main.app, {"meeting_id": meeting_id})

    assert response.status_code == 500
    assert response.json() == {
        "ok": False,
        "error": "TL;DV import failed and its saved Doc status could not be settled safely.",
    }
    assert "synthetic-secret" not in response.text and "synthetic-secret" not in caplog.text


def test_providerless_error_without_stage_retries_through_asgi(
    import_main, firestore_fake, sdk_fakes, monkeypatch
):
    meeting_id = "meeting-legacy-error-no-stage"
    _seed(firestore_fake, "transcript_index", f"tldv-{meeting_id}", {
        "meeting_id": meeting_id, "status": "error", "error": "legacy safe failure",
    })
    _seed(firestore_fake, "folders", "folder-legacy-error-retry", {
        "folder_id": "folder-legacy-error-retry", "name": "_unassigned",
        "parent_id": "pytest-synthetic-root", "speakers": {},
    })
    sdk_fakes.drive.queue("files.create", response={"id": "doc-legacy-error-retry", "modifiedTime": "synthetic-time"})
    sdk_fakes.docs.queue("documents.batchUpdate", response={})
    monkeypatch.setattr(import_main, "tldv_get", lambda path: (
        {"name": "Synthetic", "happenedAt": "2025-02-03T10:15:00Z"}
        if path == f"/meetings/{meeting_id}"
        else {"results": [{"speaker": "Speaker 1", "text": "Synthetic transcript."}]}
    ))
    monkeypatch.delenv("VECTOR_SYNC_URL", raising=False)
    monkeypatch.delenv("VECTOR_SYNC_QUEUE", raising=False)

    response = _request(import_main.app, {"meeting_id": meeting_id})

    assert response.status_code == 200 and response.json()["doc_id"] == "doc-legacy-error-retry"
    assert firestore_fake.documents["transcript_index/doc-legacy-error-retry"]["status"] == "imported"


def test_mismatched_saved_doc_identity_fails_before_enqueue_or_source_fetch(
    import_main, firestore_fake, sdk_fakes, monkeypatch
):
    meeting_id = "meeting-mismatched-saved-doc"
    _seed(firestore_fake, "transcript_index", "doc-row-with-conflict", {
        "doc_id": "different-saved-doc", "meeting_id": meeting_id, "provider": "tldv", "status": "imported",
    })
    monkeypatch.setattr(import_main, "tldv_get", lambda _path: pytest.fail("conflicting Doc identity must fail before fetch"))

    response = _request(import_main.app, {"meeting_id": meeting_id})

    assert response.status_code == 500 and response.json()["ok"] is False
    assert "different-saved-doc" not in response.text
    assert not [event for event in sdk_fakes.trace if event["boundary"] in ("drive", "docs", "tasks")]
    assert firestore_fake.documents["transcript_index/doc-row-with-conflict"]["doc_id"] == "different-saved-doc"


def test_failed_dispatch_reuses_saved_doc_on_retry(import_main, firestore_fake, sdk_fakes, monkeypatch):
    meeting_id = "meeting-dispatch-retry"
    _seed(firestore_fake, "transcript_index", f"tldv-{meeting_id}", {
        "meeting_id": meeting_id, "provider": "tldv", "status": "queued",
    })
    _seed(firestore_fake, "folders", "folder-dispatch-retry", {
        "folder_id": "folder-dispatch-retry", "name": "_unassigned",
        "parent_id": "pytest-synthetic-root", "speakers": {},
    })
    sdk_fakes.drive.queue("files.create", response={"id": "doc-dispatch-retry", "modifiedTime": "synthetic-time"})
    sdk_fakes.docs.queue("documents.batchUpdate", response={})
    sdk_fakes.tasks.queue("create_task", error=RuntimeError("synthetic-secret dispatch body"))
    monkeypatch.setenv("VECTOR_SYNC_URL", "https://sync.synthetic.test")
    monkeypatch.setenv("VECTOR_SYNC_QUEUE", "sync-queue")
    monkeypatch.setattr(import_main, "tldv_get", lambda path: (
        {"name": "Synthetic", "happenedAt": "2025-02-03T10:15:00Z"}
        if path == f"/meetings/{meeting_id}"
        else {"results": [{"speaker": "Speaker 1", "text": "Synthetic transcript."}]}
    ))

    failed = _request(import_main.app, {"meeting_id": meeting_id})
    after_failure = firestore_fake.documents["transcript_index/doc-dispatch-retry"]
    retried = _request(import_main.app, {"meeting_id": meeting_id})

    assert failed.status_code == 500 and "synthetic-secret" not in failed.text
    assert after_failure["status"] == "error" and after_failure["error_stage"] == "import"
    assert after_failure["doc_id"] == "doc-dispatch-retry"
    assert retried.status_code == 200 and retried.json() == {"ok": True, "skipped": True}
    creates = [event for event in sdk_fakes.trace if event["boundary"] == "drive"
               and event["operation"] == "files.create"
               and event["arguments"]["body"].get("mimeType") == "application/vnd.google-apps.document"]
    tasks = [event for event in sdk_fakes.trace if event["boundary"] == "tasks" and event["operation"] == "create_task"]
    assert len(creates) == 1 and len(tasks) == 2
    assert firestore_fake.documents["transcript_index/doc-dispatch-retry"]["status"] == "imported"


def test_imported_canonical_replay_resumes_named_dispatch_without_fetch(
    import_main, firestore_fake, sdk_fakes, monkeypatch
):
    meeting_id, doc_id = "meeting-imported-dispatch-resume", "doc-imported-dispatch-resume"
    _seed(firestore_fake, "transcript_index", doc_id, {
        "doc_id": doc_id, "meeting_id": meeting_id, "provider": "tldv", "status": "imported",
    })
    sdk_fakes.tasks.queue("create_task", response={})
    monkeypatch.setenv("VECTOR_SYNC_URL", "https://sync.synthetic.test")
    monkeypatch.setenv("VECTOR_SYNC_QUEUE", "sync-queue")
    monkeypatch.setattr(import_main, "tldv_get", lambda _path: pytest.fail("saved Doc replay must not fetch source"))

    response = _request(import_main.app, {"meeting_id": meeting_id})

    assert response.status_code == 200 and response.json() == {"ok": True, "skipped": True}
    tasks = [event for event in sdk_fakes.trace if event["boundary"] == "tasks" and event["operation"] == "create_task"]
    assert len(tasks) == 1
    assert firestore_fake.documents[f"transcript_index/{doc_id}"]["status"] == "imported"


@pytest.mark.parametrize(
    ("advanced_status", "enqueue_error"),
    [
        ("syncing", False), ("synced", False), ("error", False),
        ("syncing", True), ("synced", True), ("error", True),
    ],
)
def test_dispatch_settlement_preserves_faster_vector_sync_state(
    import_main, firestore_fake, sdk_fakes, monkeypatch, advanced_status, enqueue_error
):
    from tests.fixtures import sdk_fakes as sdk_module

    meeting_id = f"meeting-dispatch-race-{advanced_status}-{enqueue_error}"
    doc_id = f"doc-dispatch-race-{advanced_status}-{enqueue_error}"
    _seed(firestore_fake, "transcript_index", doc_id, {
        "doc_id": doc_id, "meeting_id": meeting_id, "provider": "tldv", "status": "imported",
    })
    if enqueue_error:
        sdk_fakes.tasks.queue("create_task", error=RuntimeError("synthetic-secret acknowledgement"))
    else:
        sdk_fakes.tasks.queue("create_task", response={})
    monkeypatch.setenv("VECTOR_SYNC_URL", "https://sync.synthetic.test")
    monkeypatch.setenv("VECTOR_SYNC_QUEUE", "sync-queue")
    original_create_task = sdk_module.FakeCloudTasksClient.create_task

    def advance_then_create_task(self, *, parent, task, **kwargs):
        state = {"status": advanced_status}
        if advanced_status == "error":
            state["error_stage"] = "vector_sync"
            state["error"] = "synthetic vector error"
        from google.cloud import firestore
        firestore.Client().collection("transcript_index").document(doc_id).update(state)
        return original_create_task(self, parent=parent, task=task, **kwargs)

    monkeypatch.setattr(sdk_module.FakeCloudTasksClient, "create_task", advance_then_create_task)
    monkeypatch.setattr(import_main, "tldv_get", lambda _path: pytest.fail("saved Doc replay must not fetch source"))

    response = _request(import_main.app, {"meeting_id": meeting_id})

    assert response.status_code == 200 and response.json() == {"ok": True, "skipped": True}
    record = firestore_fake.documents[f"transcript_index/{doc_id}"]
    assert record["status"] == advanced_status
    if advanced_status == "error":
        assert record["error_stage"] == "vector_sync"
        assert record["error"] == "synthetic vector error"
    assert "synthetic-secret" not in response.text


def test_drive_returned_foreign_reserved_id_is_not_overwritten(
    import_main, firestore_fake, sdk_fakes, monkeypatch
):
    meeting_id = "meeting-returned-foreign-reserved-id"
    reserved_id = f"tldv-{meeting_id}"
    foreign = {"meeting_id": meeting_id, "provider": "telegram", "status": "imported", "keep": True}
    _seed(firestore_fake, "transcript_index", reserved_id, foreign)
    _seed(firestore_fake, "folders", "folder-foreign-reserved", {
        "folder_id": "folder-foreign-reserved", "name": "_unassigned",
        "parent_id": "pytest-synthetic-root", "speakers": {},
    })
    sdk_fakes.drive.queue("files.create", response={"id": reserved_id, "modifiedTime": "synthetic-time"})
    sdk_fakes.docs.queue("documents.batchUpdate", response={})
    monkeypatch.delenv("VECTOR_SYNC_URL", raising=False)
    monkeypatch.delenv("VECTOR_SYNC_QUEUE", raising=False)
    monkeypatch.setattr(import_main, "tldv_get", lambda path: (
        {"name": "Synthetic", "happenedAt": "2025-02-03T10:15:00Z"}
        if path == f"/meetings/{meeting_id}"
        else {"results": [{"speaker": "Speaker 1", "text": "Synthetic transcript."}]}
    ))

    response = _request(import_main.app, {"meeting_id": meeting_id})

    assert response.status_code == 500
    assert "write outcome unknown" in response.json()["error"]
    assert firestore_fake.documents[f"transcript_index/{reserved_id}"] == foreign


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

    original_commit = firestore_module.FakeTransaction._commit

    def record_transaction_commit(self, *args, **kwargs):
        if any(
            operation == "set" and reference.path == f"transcript_index/{doc_id}"
            for operation, reference, _data, _merge in self._writes
        ):
            effects.append("firestore.transcript_set")
        return original_commit(self, *args, **kwargs)

    monkeypatch.setattr(firestore_module.FakeTransaction, "_commit", record_transaction_commit)

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
    speaker_key = import_main.speaker_key(speaker)
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
    docs_build_index = next(
        index for index, event in enumerate(sdk_fakes.trace)
        if event["operation"] == "build" and event["arguments"].get("api") == "docs"
    )
    drive_create_index = next(
        index for index, event in enumerate(sdk_fakes.trace)
        if event["boundary"] == "drive" and event["operation"] == "files.create"
        and event["arguments"]["body"].get("mimeType") == "application/vnd.google-apps.document"
    )
    batch_update_index = next(
        index for index, event in enumerate(sdk_fakes.trace)
        if event["boundary"] == "docs" and event["operation"] == "documents.batchUpdate"
    )
    assert docs_build_index < drive_create_index < batch_update_index


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


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", ("", "")),
        ("not-a-date", ("", "")),
        ("2025-02-03T10:15:00+03:00", ("2025-02-03", "10:15")),
        ("2025-02-03T10:15:00Z", ("2025-02-03", "10:15")),
    ],
)
def test_provider_date_parsing_preserves_iso_fields_without_timezone_conversion(import_main, value, expected):
    assert import_main._parse_happened_at(value) == expected


def test_provider_formatting_preserves_distinct_label_and_storage_semantics(import_main):
    meeting = {"happenedAt": "invalid"}
    utterances = [
        {"speaker": "  Alex::  ", "text": "spoken"},
        {"speakerName": " Pat:: ", "transcript": "fallback"},
        {"speaker": "ignored", "text": "", "transcript": ""},
    ]

    text, ranges = import_main._format_transcript(meeting, utterances)

    assert text == (
        "DATE: \nTIME: \nPROVIDER: tldv\n---\n"
        "  Alex   :: spoken\n\n Pat  :: fallback"
    )
    assert [text[start:end] for start, end in ranges] == ["  Alex  ", " Pat "]
    assert import_main._get_speakers(utterances) == ["Alex"]


def test_placeholder_speakers_match_supported_english_and_russian_names(import_main):
    selection = importlib.import_module("services.connectors.tldv.folder_selection")

    assert selection._is_placeholder_speaker("Speaker 1")
    assert selection._is_placeholder_speaker("СПИКЕР 2")
    assert selection._is_placeholder_speaker("participant   3")
    assert selection._is_placeholder_speaker("Участник4")
    assert not selection._is_placeholder_speaker("speaker")
    assert not selection._is_placeholder_speaker("Alex 1")


def test_speaker_folder_candidates_stop_at_five_unique_ids(import_main, monkeypatch):
    selection = importlib.import_module("services.connectors.tldv.folder_selection")
    queried = []

    class Snapshot:
        def __init__(self, doc_id):
            self.id = doc_id

        def to_dict(self):
            return {"name": "Client"}

    def query(speaker, db):
        queried.append((speaker, db))
        return [Snapshot(f"folder-{index}") for index in range(7)]

    monkeypatch.setattr(selection, "get_folders_by_speaker", query)
    db = object()

    candidates, all_ids = selection._get_folders_by_speakers(db, ["Alex"])

    assert queried == [("Alex", db)]
    assert candidates == []
    assert all_ids == [f"folder-{index}" for index in range(5)]


def test_folder_detection_uses_title_then_first_fifteen_transcript_utterances(import_main, monkeypatch):
    selection = importlib.import_module("services.connectors.tldv.folder_selection")
    db = object()
    folders = {"folder-a": {"name": "Alpha"}, "folder-b": {"name": "Beta"}}
    seen_db = []
    prompts = []

    def all_folders(supplied_db):
        seen_db.append(supplied_db)
        return folders

    def gemini(prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            return {"folder_name": "Alpha", "confidence": 0.79}
        return {"folder_name": "Beta", "confidence": 0.8}

    monkeypatch.setattr(selection, "get_all_folders", all_folders)
    monkeypatch.setattr(selection, "get_folders_by_speaker", lambda _speaker, _db: [])
    utterances = [
        {"speaker": "Alex", "text": f"line-{index}"} for index in range(15)
    ] + [
        {"speaker": "Alex", "text": "line-15"},
        {"speaker": "Alex", "text": ""},
    ]

    result = selection._detect_target_folder_id(
        db, {"name": "Weekly sync"}, utterances, object(), "root", gemini_json=gemini
    )

    assert result == "folder-b"
    assert seen_db == [db]
    assert len(prompts) == 2
    assert "Meeting name: Weekly sync" in prompts[0]
    assert "Transcript excerpt:" not in prompts[0]
    assert "Transcript excerpt:" in prompts[1]
    assert "Alex: line-0" in prompts[1]
    assert "Alex: line-14" in prompts[1]
    assert "line-15" not in prompts[1]
