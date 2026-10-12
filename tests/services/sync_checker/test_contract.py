from __future__ import annotations

import importlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from google.cloud import tasks_v2
from googleapiclient.errors import HttpError


@pytest.fixture
def checker_main(sdk_fakes):
    # Keep the production handler and SDK symbols; conftest installs offline boundaries.
    del sdk_fakes
    return importlib.import_module("services.sync_checker.main")


def _request(payload=None):
    return SimpleNamespace(get_json=lambda silent=True: payload)


def _task_events(sdk_fakes):
    return [
        event for event in sdk_fakes.trace
        if event["boundary"] == "tasks" and event["operation"] == "create_task"
    ]


def _seed(firestore_fake, collection, doc_id, data):
    from google.cloud import firestore

    firestore.Client().collection(collection).document(doc_id).set(data)


def test_targeted_doc_queues_anonymous_task_and_skips_checker_queries(
    checker_main, sdk_fakes, monkeypatch
):
    monkeypatch.setenv("VECTOR_SYNC_URL", "https://sync.synthetic.test")
    monkeypatch.setenv("VECTOR_SYNC_QUEUE", "targeted-queue")
    for name in (
        "get_stale_syncing", "get_error_docs", "get_drive_sync_token",
        "get_drive_changes", "get_start_page_token", "scan_root_folder",
    ):
        monkeypatch.setattr(checker_main, name, lambda *args, _name=name, **kwargs: pytest.fail(
            f"targeted request must skip {_name}"
        ))

    response = checker_main.checker(_request({"doc_id": "doc-targeted"}))

    assert response == ({"mode": "doc", "doc_id": "doc-targeted", "queued": 1}, 200)
    tasks = _task_events(sdk_fakes)
    assert len(tasks) == 1
    assert tasks[0]["arguments"]["parent"].endswith("/queues/targeted-queue")
    assert "name" not in tasks[0]["arguments"]["task"]
    assert tasks[0]["arguments"]["task"]["http_request"] == {
        "http_method": tasks_v2.HttpMethod.POST,
        "url": "https://sync.synthetic.test/sync/doc/doc-targeted",
    }


def test_stale_and_error_retries_are_queued_in_order_before_changes(
    checker_main, sdk_fakes, firestore_fake, monkeypatch
):
    monkeypatch.setenv("VECTOR_SYNC_URL", "https://sync.synthetic.test")
    now = datetime.now(timezone.utc)
    _seed(firestore_fake, "transcript_index", "stale-a", {
        "status": "syncing", "status_changed_at": now - timedelta(minutes=30),
    })
    _seed(firestore_fake, "transcript_index", "stale-b", {
        "status": "syncing", "status_changed_at": now - timedelta(minutes=20),
    })
    _seed(firestore_fake, "transcript_index", "fresh-syncing", {
        "status": "syncing", "status_changed_at": now - timedelta(minutes=5),
    })
    _seed(firestore_fake, "transcript_index", "error-a", {
        "status": "error", "error_stage": "vector_sync",
    })
    _seed(firestore_fake, "transcript_index", "error-import", {
        "status": "error", "error_stage": "import",
    })
    _seed(firestore_fake, "transcript_index", "error-invalid-format", {
        "status": "error", "error_stage": "invalid_format",
    })
    calls = []
    original_enqueue = checker_main.enqueue_task

    def enqueue(**kwargs):
        calls.append(("task", kwargs["url"]))
        return original_enqueue(**kwargs)

    monkeypatch.setattr(checker_main, "enqueue_task", enqueue)
    monkeypatch.setattr(checker_main, "get_drive_sync_token", lambda: calls.append(("cursor",)) or "page-old")
    monkeypatch.setattr(checker_main, "get_drive_changes", lambda token: calls.append(("changes", token)) or ([], "page-new"))
    monkeypatch.setattr(checker_main, "set_drive_sync_token", lambda token: calls.append(("save", token)))

    response = checker_main.checker(_request())

    assert response == ({"queued": 0, "stale_syncing": 2, "error_docs": 1}, 200)
    assert calls == [
        ("task", "https://sync.synthetic.test/sync/doc/stale-a"),
        ("task", "https://sync.synthetic.test/sync/doc/stale-b"),
        ("task", "https://sync.synthetic.test/sync/doc/error-a"),
        ("cursor",),
        ("changes", "page-old"),
        ("save", "page-new"),
    ]
    task_urls = [event["arguments"]["task"]["http_request"]["url"] for event in _task_events(sdk_fakes)]
    assert task_urls == [
        "https://sync.synthetic.test/sync/doc/stale-a",
        "https://sync.synthetic.test/sync/doc/stale-b",
        "https://sync.synthetic.test/sync/doc/error-a",
    ]
    assert all("name" not in event["arguments"]["task"] for event in _task_events(sdk_fakes))


def test_nonexpired_drive_provider_error_propagates(checker_main, monkeypatch):
    monkeypatch.setenv("VECTOR_SYNC_URL", "https://sync.synthetic.test")
    monkeypatch.setattr(checker_main, "get_stale_syncing", lambda db=None: [])
    monkeypatch.setattr(checker_main, "get_error_docs", lambda db=None: [])
    monkeypatch.setattr(checker_main, "get_drive_sync_token", lambda: "page-token")
    error = HttpError(SimpleNamespace(status=503, reason="Unavailable"), b"provider unavailable")

    def fail_changes(_token):
        raise error

    monkeypatch.setattr(checker_main, "get_drive_changes", fail_changes)

    with pytest.raises(HttpError) as raised:
        checker_main.checker(_request())

    assert raised.value is error


def test_expired_cursor_falls_back_to_scan_and_enqueues_only_eligible_docs(
    checker_main, sdk_fakes, firestore_fake, monkeypatch
):
    monkeypatch.setenv("VECTOR_SYNC_URL", "https://sync.synthetic.test")
    monkeypatch.setattr(checker_main, "get_stale_syncing", lambda db=None: [])
    monkeypatch.setattr(checker_main, "get_error_docs", lambda db=None: [])
    monkeypatch.setattr(checker_main, "get_drive_sync_token", lambda: "expired-page")
    calls = []

    def expired(_token):
        calls.append("changes")
        raise HttpError(SimpleNamespace(status=400, reason="Bad Request"), b"expired page token")

    monkeypatch.setattr(checker_main, "get_drive_changes", expired)
    monkeypatch.setattr(checker_main, "get_start_page_token", lambda: calls.append("start-token") or "fresh-start")
    monkeypatch.setattr(checker_main, "scan_root_folder", lambda: calls.append("scan") or (
        [{"id": "live-folder", "name": "Live", "parent_id": "pytest-synthetic-root"}],
        [
            {"doc_id": "live-queued", "name": "queued", "parent_id": "live-folder", "modifiedTime": "new"},
            {"doc_id": "live-downloading", "name": "downloading", "parent_id": "live-folder", "modifiedTime": "new"},
            {"doc_id": "live-new", "name": "new", "parent_id": "live-folder", "modifiedTime": "new"},
            {"doc_id": "live-unchanged", "name": "same", "parent_id": "live-folder", "modifiedTime": "same-time"},
        ],
    ))
    _seed(firestore_fake, "transcript_index", "live-queued", {"status": "queued"})
    _seed(firestore_fake, "transcript_index", "live-downloading", {"status": "downloading"})
    _seed(firestore_fake, "transcript_index", "live-unchanged", {
        "status": "synced", "modifiedTime": "same-time", "parent_id": "live-folder",
    })
    _seed(firestore_fake, "transcript_index", "orphan-queued", {"status": "queued"})
    _seed(firestore_fake, "transcript_index", "orphan-downloading", {"status": "downloading"})
    _seed(firestore_fake, "transcript_index", "orphan-import-error", {
        "status": "error", "error_stage": "import",
    })
    _seed(firestore_fake, "transcript_index", "orphan-synced", {"status": "synced"})
    _seed(firestore_fake, "transcript_index", "orphan-retry-error", {
        "status": "error", "error_stage": "vector_sync",
    })
    events = []
    monkeypatch.setattr(checker_main, "upsert_folder", lambda *args: events.append(("folder", args)))
    monkeypatch.setattr(checker_main, "get_all_folders", lambda: {"old-folder": {}})
    monkeypatch.setattr(checker_main, "delete_folder", lambda folder_id: events.append(("delete-folder", folder_id)))
    original_enqueue = checker_main.enqueue_task

    def enqueue(**kwargs):
        events.append(("task", kwargs["url"]))
        return original_enqueue(**kwargs)

    monkeypatch.setattr(checker_main, "enqueue_task", enqueue)
    monkeypatch.setattr(checker_main, "set_drive_sync_token", lambda token: events.append(("save-token", token)))

    response = checker_main.checker(_request())

    assert response == ({"queued": 3, "stale_syncing": 0, "error_docs": 0}, 200)
    assert calls == ["changes", "start-token", "scan"]
    assert events[0] == ("folder", ("live-folder", "Live", "pytest-synthetic-root"))
    assert events[1] == ("delete-folder", "old-folder")
    assert [item for item in events if item[0] == "task"] == [
        ("task", "https://sync.synthetic.test/sync/doc/live-new"),
        ("task", "https://sync.synthetic.test/sync/doc/orphan-retry-error"),
        ("task", "https://sync.synthetic.test/sync/doc/orphan-synced"),
    ]
    assert events[-1] == ("save-token", "fresh-start")
    task_urls = [event["arguments"]["task"]["http_request"]["url"] for event in _task_events(sdk_fakes)]
    assert task_urls == [url for kind, url in events if kind == "task"]
    assert all("name" not in event["arguments"]["task"] for event in _task_events(sdk_fakes))
