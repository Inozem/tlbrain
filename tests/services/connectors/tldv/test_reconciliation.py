from __future__ import annotations

import importlib
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import google.api_core.exceptions
import pytest
import requests
from google.cloud import tasks_v2


@pytest.fixture
def reconciliation(app_root, monkeypatch, sdk_fakes):
    del sdk_fakes  # conftest installs the SDK fakes before importing the handler.
    monkeypatch.syspath_prepend(str(app_root / "services" / "connectors" / "tldv"))
    return importlib.import_module("services.connectors.tldv.reconciliation.main")


def _request(payload):
    return SimpleNamespace(get_json=lambda silent=True: payload)


def _seed(firestore_fake, doc_id, data):
    from google.cloud import firestore

    firestore.Client().collection("transcript_index").document(doc_id).set(data)


def _fixed_now(monkeypatch, reconciliation, now):
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now if tz is not None else now.replace(tzinfo=None)

    monkeypatch.setattr(reconciliation, "datetime", FixedDatetime)


def _http_error(status_code):
    return requests.exceptions.HTTPError(
        f"scripted HTTP {status_code}",
        response=SimpleNamespace(status_code=status_code),
    )


def _task_events(sdk_fakes):
    return [
        event for event in sdk_fakes.trace
        if event["boundary"] == "tasks" and event["operation"] == "create_task"
    ]


def test_recovery_uses_inclusive_age_window_and_only_eligible_statuses(
    reconciliation, firestore_fake, monkeypatch, sdk_fakes
):
    now = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)
    _fixed_now(monkeypatch, reconciliation, now)
    monkeypatch.setenv("TLDV_IMPORT_SERVICE_URL", "https://import.synthetic.test")
    sdk_fakes.queue(
        "tasks", "create_task",
        error=google.api_core.exceptions.AlreadyExists("synthetic recovery duplicate"),
    )
    stale = now - timedelta(hours=1)
    oldest = now - timedelta(hours=48)
    candidates = {
        "queued-at-new-edge": {
            "status": "queued", "status_changed_at": stale,
            "error": "retained queued error", "error_stage": "import",
        },
        "downloading-at-old-edge": {"status": "downloading", "status_changed_at": oldest},
        "error-no-stage": {"status": "error", "status_changed_at": stale},
        "error-import": {"status": "error", "error_stage": "import", "status_changed_at": stale},
        "queued-outside-new-edge": {
            "status": "queued", "status_changed_at": stale + timedelta(seconds=1)
        },
        "queued-outside-old-edge": {
            "status": "queued", "status_changed_at": oldest - timedelta(seconds=1)
        },
        "error-other-stage": {
            "status": "error", "error_stage": "vector_sync", "status_changed_at": stale
        },
        "imported": {"status": "imported", "status_changed_at": stale},
        "missing-time": {"status": "queued"},
    }
    for meeting_id, data in candidates.items():
        _seed(firestore_fake, f"tldv-{meeting_id}", {"meeting_id": meeting_id, **data})

    provider_calls = []

    def provider(path):
        provider_calls.append(path)
        return {"results": [{"text": "synthetic transcript"}]}

    monkeypatch.setattr(reconciliation, "tldv_get", provider)
    monkeypatch.setattr(reconciliation, "get_meetings", lambda _since: [])
    writes = []
    write_queued = reconciliation.write_queued
    monkeypatch.setattr(reconciliation, "write_queued", lambda mid: writes.append(mid) or write_queued(mid))

    result, status_code = reconciliation.tldv_reconciliation(_request({}))

    eligible = ["queued-at-new-edge", "downloading-at-old-edge", "error-no-stage", "error-import"]
    assert status_code == 200
    assert result == {
        "meetings": 0, "missing": 0, "queued": 0, "remaining": 0,
        "recovered": len(eligible), "finalized": 0,
    }
    assert sorted(provider_calls) == sorted(f"/meetings/{meeting_id}/transcript" for meeting_id in eligible)
    assert writes == []
    for meeting_id in eligible:
        record = firestore_fake.documents[f"transcript_index/tldv-{meeting_id}"]
        if candidates[meeting_id]["status"] == "queued":
            assert record["status"] == "queued"
            assert record["error"] == "retained queued error"
            assert record["error_stage"] == "import"
        else:
            assert record["status"] == "queued"
            assert record["error"] is None
            assert record["error_stage"] is None
    assert len(_task_events(sdk_fakes)) == len(eligible)
    oldest_task = next(
        event["arguments"] for event in _task_events(sdk_fakes)
        if event["arguments"]["task"]["name"].endswith("/tasks/tldv-import-downloading-at-old-edge")
    )
    assert oldest_task["task"]["http_request"] == {
        "http_method": tasks_v2.HttpMethod.POST,
        "url": "https://import.synthetic.test/import",
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"meeting_id": "downloading-at-old-edge"}).encode(),
    }
    assert "transcript_index/tldv-queued-outside-new-edge" in firestore_fake.documents


def test_recovery_empty_transcript_and_nonfinal_http_error_leave_records_unchanged(
    reconciliation, firestore_fake, monkeypatch, sdk_fakes
):
    now = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)
    _fixed_now(monkeypatch, reconciliation, now)
    monkeypatch.setenv("TLDV_IMPORT_SERVICE_URL", "https://import.synthetic.test")
    for meeting_id in ("still-processing", "provider-error"):
        _seed(firestore_fake, f"tldv-{meeting_id}", {
            "meeting_id": meeting_id,
            "status": "downloading",
            "status_changed_at": now - timedelta(hours=2),
            "error": "keep this",
        })

    def provider(path):
        if "provider-error" in path:
            raise _http_error(500)
        return {"results": []}

    monkeypatch.setattr(reconciliation, "tldv_get", provider)
    monkeypatch.setattr(reconciliation, "get_meetings", lambda _since: [])

    result, status_code = reconciliation.tldv_reconciliation(_request({}))

    assert status_code == 200
    assert result["recovered"] == 0
    assert result["finalized"] == 0
    assert firestore_fake.documents["transcript_index/tldv-still-processing"]["status"] == "downloading"
    assert firestore_fake.documents["transcript_index/tldv-provider-error"]["error"] == "keep this"
    assert _task_events(sdk_fakes) == []


@pytest.mark.parametrize("status", [403, 404])
def test_recovery_403_and_404_finalize_import_record(
    reconciliation, firestore_fake, monkeypatch, status
):
    now = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)
    _fixed_now(monkeypatch, reconciliation, now)
    meeting_id = f"gone-{status}"
    _seed(firestore_fake, f"tldv-{meeting_id}", {
        "meeting_id": meeting_id,
        "status": "error",
        "error_stage": "import",
        "status_changed_at": now - timedelta(hours=2),
    })
    monkeypatch.setenv("TLDV_IMPORT_SERVICE_URL", "https://import.synthetic.test")
    monkeypatch.setattr(reconciliation, "tldv_get", lambda _path: (_ for _ in ()).throw(_http_error(status)))
    monkeypatch.setattr(reconciliation, "get_meetings", lambda _since: [])

    result, status_code = reconciliation.tldv_reconciliation(_request({}))

    record = firestore_fake.documents[f"transcript_index/tldv-{meeting_id}"]
    assert status_code == 200
    assert result["recovered"] == 0
    assert result["finalized"] == 1
    assert record["status"] == "error"
    assert record["error"] == f"HTTP {status} in reconciliation recovery"
    assert record["error_stage"] == "import"


def test_scan_respects_queued_freshness_and_retries_import_errors(
    reconciliation, firestore_fake, monkeypatch, sdk_fakes
):
    now = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)
    _fixed_now(monkeypatch, reconciliation, now)
    cutoff = now - timedelta(minutes=15)
    _seed(firestore_fake, "fresh", {
        "meeting_id": "fresh", "status": "queued", "status_changed_at": cutoff
    })
    _seed(firestore_fake, "old", {
        "meeting_id": "old", "status": "queued", "status_changed_at": cutoff - timedelta(seconds=1)
    })
    _seed(firestore_fake, "untimed", {"meeting_id": "untimed", "status": "queued"})
    _seed(firestore_fake, "retry", {
        "meeting_id": "retry", "status": "error", "error_stage": "import"
    })
    meetings = [{"id": mid} for mid in ("fresh", "old", "untimed", "retry", "imported")]
    _seed(firestore_fake, "imported", {"meeting_id": "imported", "status": "imported"})
    monkeypatch.setattr(reconciliation, "tldv_get", lambda _path: {"results": []})
    monkeypatch.setattr(reconciliation, "get_meetings", lambda _since: meetings)
    monkeypatch.setenv("TLDV_IMPORT_SERVICE_URL", "https://import.synthetic.test")

    result, status_code = reconciliation.tldv_reconciliation(_request({}))

    assert status_code == 200
    assert result == {
        "meetings": 5, "missing": 3, "queued": 3, "remaining": 0,
        "recovered": 0, "finalized": 0,
    }
    assert [event["arguments"]["task"]["name"].rsplit("/", 1)[-1] for event in _task_events(sdk_fakes)] == [
        "tldv-import-old", "tldv-import-untimed", "tldv-import-retry"
    ]


@pytest.mark.parametrize(
    ("payload", "expected_since"),
    [
        ({"since": "2026-10-08T10:30:00+00:00", "full_scan": True}, datetime(2026, 10, 8, 10, 30, tzinfo=timezone.utc)),
        ({"full_scan": True}, None),
    ],
)
def test_scan_passes_since_or_full_scan_to_provider(
    reconciliation, monkeypatch, payload, expected_since
):
    captured = []
    monkeypatch.setattr(reconciliation, "tldv_get", lambda _path: {"results": []})
    monkeypatch.setattr(reconciliation, "get_meetings", lambda since: captured.append(since) or [])
    monkeypatch.delenv("TLDV_IMPORT_SERVICE_URL", raising=False)

    reconciliation.tldv_reconciliation(_request(payload))

    assert captured == [expected_since]


def test_limit_paginates_until_enough_missing_meetings_and_returns_unknown_remaining(
    reconciliation, monkeypatch
):
    requested_since = datetime(2026, 10, 8, 10, 30, tzinfo=timezone.utc)
    captured = []
    pages = [[{"id": "existing"}], [{"id": "new-1"}, {"id": "new-2"}], [{"id": "not-visited"}]]

    def iter_pages(since):
        captured.append(since)
        yield from pages

    monkeypatch.setattr(reconciliation, "tldv_get", lambda _path: {"results": []})
    monkeypatch.setattr(reconciliation, "iter_meeting_pages", iter_pages)
    monkeypatch.setattr(reconciliation, "get_meetings", lambda _since: pytest.fail("paged scan expected"))
    from google.cloud import firestore

    firestore.Client().collection("transcript_index").document("existing").set(
        {"meeting_id": "existing", "status": "imported"}
    )
    monkeypatch.delenv("TLDV_IMPORT_SERVICE_URL", raising=False)

    result, status_code = reconciliation.tldv_reconciliation(
        _request({"since": requested_since.isoformat(), "limit": 2})
    )

    assert captured == [requested_since]
    assert status_code == 200
    assert result["meetings"] == 3
    assert result["missing"] == 2
    assert result["remaining"] is None


def test_missing_import_url_omits_recovery_counters_even_after_recovery(
    reconciliation, firestore_fake, monkeypatch, sdk_fakes
):
    now = datetime(2026, 10, 9, 12, tzinfo=timezone.utc)
    _fixed_now(monkeypatch, reconciliation, now)
    _seed(firestore_fake, "tldv-recovered", {
        "meeting_id": "recovered", "status": "downloading",
        "status_changed_at": now - timedelta(hours=2),
    })
    monkeypatch.setattr(reconciliation, "tldv_get", lambda _path: {"results": [{"text": "ready"}]})
    monkeypatch.setattr(reconciliation, "get_meetings", lambda _since: [{"id": "new"}])
    monkeypatch.delenv("TLDV_IMPORT_SERVICE_URL", raising=False)

    result, status_code = reconciliation.tldv_reconciliation(_request({}))

    assert status_code == 200
    assert result == {"meetings": 1, "missing": 1, "queued": 0, "remaining": 0}
    assert firestore_fake.documents["transcript_index/tldv-recovered"]["status"] == "queued"
    assert _task_events(sdk_fakes) == []


def test_batch_writes_placeholder_before_named_task_and_already_exists_is_not_counted(
    reconciliation, firestore_fake, monkeypatch, sdk_fakes
):
    from google.api_core.exceptions import AlreadyExists
    from google.cloud import firestore

    meeting_id = "batch-meeting"
    monkeypatch.setenv("TLDV_IMPORT_SERVICE_URL", "https://import.synthetic.test")
    monkeypatch.setenv("TLDV_IMPORT_QUEUE", "synthetic-import-queue")
    monkeypatch.setattr(reconciliation, "tldv_get", lambda _path: {"results": []})
    monkeypatch.setattr(reconciliation, "get_meetings", lambda _since: [{"id": meeting_id}])
    sdk_fakes.queue("tasks", "create_task", error=AlreadyExists("synthetic duplicate"))
    effects = []
    write_queued = reconciliation.write_queued
    enqueue_task = reconciliation.enqueue_task

    def record_write(mid):
        effects.append("placeholder")
        return write_queued(mid)

    def record_enqueue(**kwargs):
        effects.append("task")
        return enqueue_task(**kwargs)

    monkeypatch.setattr(reconciliation, "write_queued", record_write)
    monkeypatch.setattr(reconciliation, "enqueue_task", record_enqueue)

    result, status_code = reconciliation.tldv_reconciliation(_request({}))

    assert status_code == 200
    assert result["queued"] == 0
    assert firestore_fake.documents["transcript_index/tldv-batch-meeting"]["status"] == "queued"
    assert effects == ["placeholder", "task"]
    task_event = _task_events(sdk_fakes)[0]["arguments"]
    assert task_event["parent"] == (
        "projects/pytest-synthetic-project/locations/pytest-synthetic-region/"
        "queues/synthetic-import-queue"
    )
    assert task_event["task"] == {
        "name": f"{task_event['parent']}/tasks/tldv-import-{meeting_id}",
        "http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            "url": "https://import.synthetic.test/import",
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"meeting_id": meeting_id}).encode(),
        },
    }
