from __future__ import annotations

import httpx
import pytest

from core.operations import sync


class Response:
    def __init__(self, payload):
        self.payload = payload
        self.checked = False

    def raise_for_status(self):
        self.checked = True

    def json(self):
        return self.payload


@pytest.mark.parametrize(("doc_id", "expected_body"), [
    (None, None), ("", None), (False, None), (0, None), (" doc-1 ", {"doc_id": "doc-1"}),
    ("   ", {"doc_id": ""}),
])
def test_sync_changes_uses_trusted_url_timeout_and_legacy_body(monkeypatch, doc_id, expected_body):
    monkeypatch.setenv("SYNC_CHECKER_URL", "  https://checker.test/run  ")
    response = Response({"queued": 3})
    calls = []
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: calls.append((args, kwargs)) or response)
    result = sync.sync_changes(doc_id)
    assert result == {"queued": 3}
    assert response.checked
    assert calls == [(("  https://checker.test/run  ",), {"json": expected_body, "timeout": 300})]


def test_missing_config_is_typed_and_never_posts(monkeypatch):
    monkeypatch.delenv("SYNC_CHECKER_URL", raising=False)
    monkeypatch.setattr(httpx, "post", lambda *_args, **_kwargs: pytest.fail("must not post"))
    with pytest.raises(sync.SyncConfigurationError) as raised:
        sync.sync_changes("doc")
    assert str(raised.value) == "SYNC_CHECKER_URL is not configured"


def test_falsey_config_is_missing_but_nonempty_configuration_is_passed_unchanged(monkeypatch):
    for value in ("",):
        monkeypatch.setenv("SYNC_CHECKER_URL", value)
        with pytest.raises(sync.SyncConfigurationError):
            sync.sync_changes()
    monkeypatch.setenv("SYNC_CHECKER_URL", " ")
    monkeypatch.setattr(httpx, "post", lambda url, **kwargs: Response({"url": url}))
    assert sync.sync_changes() == {"url": " "}


def test_truthy_non_string_doc_id_fails_at_body_preparation(monkeypatch):
    monkeypatch.setenv("SYNC_CHECKER_URL", "https://checker.test/run")
    monkeypatch.setattr(httpx, "post", lambda *_args, **_kwargs: pytest.fail("must fail before post"))
    with pytest.raises(AttributeError):
        sync.sync_changes(7)


def test_transport_failure_propagates_same_exception(monkeypatch):
    monkeypatch.setenv("SYNC_CHECKER_URL", "https://checker.test/run")
    failure = httpx.ConnectError(
        "synthetic connection failure", request=httpx.Request("POST", "https://checker.test/run"),
    )
    calls = []

    def post(*_args, **_kwargs):
        calls.append("post")
        raise failure

    monkeypatch.setattr(httpx, "post", post)
    with pytest.raises(httpx.ConnectError) as raised:
        sync.sync_changes()
    assert raised.value is failure
    assert calls == ["post"]


def test_http_status_failure_uses_real_response_and_skips_json_decode(monkeypatch):
    monkeypatch.setenv("SYNC_CHECKER_URL", "https://checker.test/run")
    request = httpx.Request("POST", "https://checker.test/run")
    response = httpx.Response(503, request=request, content=b"unavailable")
    calls = []
    real_raise_for_status = response.raise_for_status

    def raise_for_status():
        calls.append("raise_for_status")
        return real_raise_for_status()

    response.raise_for_status = raise_for_status
    response.json = lambda: calls.append("json") or pytest.fail("status failures must not decode JSON")
    monkeypatch.setattr(httpx, "post", lambda *_args, **_kwargs: response)

    with pytest.raises(httpx.HTTPStatusError) as raised:
        sync.sync_changes()
    assert raised.value.response is response
    assert calls == ["raise_for_status"]


def test_json_decode_failure_happens_after_successful_status_check(monkeypatch):
    monkeypatch.setenv("SYNC_CHECKER_URL", "https://checker.test/run")
    request = httpx.Request("POST", "https://checker.test/run")
    response = httpx.Response(200, request=request, content=b"not-json")
    calls = []
    decode_errors = []
    real_raise_for_status = response.raise_for_status
    real_json = response.json

    def raise_for_status():
        calls.append("raise_for_status")
        return real_raise_for_status()

    def decode_json():
        calls.append("json")
        try:
            return real_json()
        except Exception as error:
            decode_errors.append(error)
            raise

    response.raise_for_status = raise_for_status
    response.json = decode_json
    monkeypatch.setattr(httpx, "post", lambda *_args, **_kwargs: response)

    with pytest.raises(ValueError) as raised:
        sync.sync_changes()
    assert decode_errors == [raised.value]
    assert calls == ["raise_for_status", "json"]


def test_sync_returns_malformed_decoded_json_without_validating_it(monkeypatch):
    monkeypatch.setenv("SYNC_CHECKER_URL", "https://checker.test/run")
    payload = ["raw", "checker", "json"]

    class MalformedResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return payload

    monkeypatch.setattr(httpx, "post", lambda *_args, **_kwargs: MalformedResponse())
    assert sync.sync_changes() is payload
