"""Offline tests for recoverable native Google Doc Drive helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from googleapiclient.errors import HttpError

from core.integrations.google_drive.files import (
    GOOGLE_DOC_MIME_TYPE,
    create_native_doc,
    find_native_docs_by_save_intent,
    read_native_doc_metadata,
)


MARKER = "save-intent-123"
PARENT = "resolved-parent"


class _Request:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    def execute(self):
        if self.error:
            raise self.error
        return self.result


class _Files:
    def __init__(self):
        self.calls = []
        self.responses = []
        self.errors = {}

    def create(self, **kwargs):
        self.calls.append(("create", kwargs))
        return self._request("create", {"id": "created-doc"})

    def list(self, **kwargs):
        self.calls.append(("list", kwargs))
        return self._request("list", self.responses.pop(0))

    def get(self, **kwargs):
        self.calls.append(("get", kwargs))
        return self._request("get", {"id": kwargs["fileId"], "mimeType": GOOGLE_DOC_MIME_TYPE})

    def _request(self, operation, default):
        error = self.errors.get(operation)
        if error:
            return _Request(error=error)
        return _Request(default)


class _Service:
    def __init__(self):
        self.api = _Files()

    def files(self):
        return self.api


def _candidate(file_id, *, marker=MARKER, mime=GOOGLE_DOC_MIME_TYPE, parents=None, trashed=False, props=None):
    return {
        "id": file_id,
        "mimeType": mime,
        "parents": [PARENT] if parents is None else parents,
        "trashed": trashed,
        "appProperties": {"tlbrain_save_id": marker} if props is None else props,
    }


@pytest.mark.offline
def test_create_native_doc_uses_one_exact_marked_create_request():
    service = _Service()

    assert create_native_doc(service, "Transcript", PARENT, MARKER) == "created-doc"
    assert service.api.calls == [("create", {
        "body": {
            "name": "Transcript",
            "mimeType": GOOGLE_DOC_MIME_TYPE,
            "parents": [PARENT],
            "appProperties": {"tlbrain_save_id": MARKER},
        },
        "fields": "id",
    })]


@pytest.mark.offline
def test_marker_lookup_filters_candidates_and_paginates_without_followup_gets():
    service = _Service()
    service.api.responses = [
        {"files": [
            _candidate("valid-one"),
            _candidate("wrong-marker", marker="other"),
            _candidate("wrong-parent", parents=["outside"]),
            _candidate("trashed", trashed=True),
            _candidate("wrong-type", mime="text/plain"),
            {"id": "bad-marker", "mimeType": GOOGLE_DOC_MIME_TYPE, "parents": [PARENT], "trashed": False, "appProperties": "bad"},
            _candidate("bad-parent-shape", parents=[PARENT, "another"]),
        ], "nextPageToken": "next"},
        {"files": [_candidate("valid-two")]},
    ]

    found = find_native_docs_by_save_intent(service, MARKER, PARENT)

    assert [item["id"] for item in found] == ["valid-one", "valid-two"]
    assert len(service.api.calls) == 2
    assert all(call[0] == "list" for call in service.api.calls)
    first_args = service.api.calls[0][1]
    assert "appProperties has" in first_args["q"]
    assert "name" not in first_args["q"] and "fullText" not in first_args["q"]
    assert first_args["fields"] == "nextPageToken,files(id,mimeType,parents,trashed,appProperties)"
    assert "pageToken" not in first_args
    assert service.api.calls[1][1]["pageToken"] == "next"


@pytest.mark.offline
def test_marker_lookup_returns_empty_and_multiple_matches_as_read_only_results():
    service = _Service()
    service.api.responses = [{"files": []}, {"files": [_candidate("one"), _candidate("two")]}]

    assert find_native_docs_by_save_intent(service, MARKER, PARENT) == []
    assert [item["id"] for item in find_native_docs_by_save_intent(service, MARKER, PARENT)] == ["one", "two"]
    assert all(call[0] == "list" for call in service.api.calls)


@pytest.mark.offline
def test_marker_lookup_fails_closed_when_later_page_is_malformed():
    service = _Service()
    service.api.responses = [{"files": [_candidate("valid-first-page")], "nextPageToken": "next"}, {"files": None}]

    with pytest.raises(RuntimeError, match="Drive marker lookup returned malformed metadata"):
        find_native_docs_by_save_intent(service, MARKER, PARENT)

    assert len(service.api.calls) == 2


@pytest.mark.offline
def test_marker_lookup_fails_closed_on_non_record_page_item():
    service = _Service()
    service.api.responses = [{"files": [_candidate("valid"), "malformed record"]}]

    with pytest.raises(RuntimeError, match="Drive marker lookup returned malformed metadata"):
        find_native_docs_by_save_intent(service, MARKER, PARENT)


@pytest.mark.offline
@pytest.mark.parametrize("response", [None, [], {}, {"files": None}, {"files": [], "nextPageToken": 17}, {"files": [], "nextPageToken": ""}])
def test_marker_lookup_rejects_malformed_page_shapes_and_tokens(response):
    service = _Service()
    service.api.responses = [response]

    with pytest.raises(RuntimeError, match="Drive marker lookup returned malformed metadata"):
        find_native_docs_by_save_intent(service, MARKER, PARENT)


@pytest.mark.offline
def test_marker_lookup_hides_provider_error_details():
    service = _Service()
    secret = "sensitive provider reason and response body"
    service.api.errors["list"] = HttpError(SimpleNamespace(status=403, reason=secret), secret.encode())

    with pytest.raises(RuntimeError, match="Drive marker lookup is unavailable") as raised:
        find_native_docs_by_save_intent(service, MARKER, PARENT)

    assert secret not in str(raised.value)


@pytest.mark.offline
def test_known_id_readback_requests_only_validation_metadata():
    service = _Service()

    metadata = read_native_doc_metadata(service, "known-doc")

    assert metadata["id"] == "known-doc"
    assert service.api.calls == [("get", {
        "fileId": "known-doc",
        "fields": "id,mimeType,parents,trashed,appProperties,modifiedTime",
    })]


@pytest.mark.offline
@pytest.mark.parametrize("status,outcome", [(403, "rejected"), (408, "unknown"), (429, "unknown"), (503, "unknown")])
def test_create_failure_classification_is_safe(status, outcome):
    service = _Service()
    secret = "private provider response and transcript"
    service.api.errors["create"] = HttpError(SimpleNamespace(status=status, reason=secret), secret.encode())

    with pytest.raises(RuntimeError) as raised:
        create_native_doc(service, "private title", PARENT, MARKER)

    assert raised.value.outcome == outcome
    assert secret not in str(raised.value)
    assert "private title" not in str(raised.value)


@pytest.mark.offline
def test_create_malformed_response_is_unknown_without_exposing_content():
    service = _Service()
    service.api.create = lambda **kwargs: _Request({"private": "transcript text"})

    with pytest.raises(RuntimeError) as raised:
        create_native_doc(service, "private title", PARENT, MARKER)

    assert raised.value.outcome == "unknown"
    assert "transcript text" not in str(raised.value)


@pytest.mark.offline
def test_readback_hides_provider_error_details():
    service = _Service()
    service.api.errors["get"] = RuntimeError("credential and document body")

    with pytest.raises(RuntimeError, match="Drive document metadata is unavailable") as raised:
        read_native_doc_metadata(service, "known-doc")

    assert "credential" not in str(raised.value)
    assert "document body" not in str(raised.value)
