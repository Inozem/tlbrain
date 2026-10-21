from __future__ import annotations

import httpx
import pytest

from services.mcp.app.mcp import handler
from services.mcp.app.mcp.schemas import JSONRPCRequest


def _request(arguments):
    return JSONRPCRequest(**{
        "jsonrpc": "2.0", "id": "sync-test", "method": "tools/call",
        "params": {"name": "sync_changes", "arguments": arguments},
    })


class Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def test_handler_preserves_success_envelope_and_checker_status_override(monkeypatch):
    monkeypatch.setenv("SYNC_CHECKER_URL", "https://checker.test/run")
    posted = []
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: posted.append((args, kwargs)) or Response({
        "status": "checker-status", "queued": 2,
    }))
    result = handler._handle_sync_changes(_request({}), {})
    assert posted == [(("https://checker.test/run",), {"json": None, "timeout": 300})]
    assert result["result"]["content"][0]["text"] == '{"status": "checker-status", "queued": 2}'


def test_handler_preserves_missing_config_error_without_details(monkeypatch):
    monkeypatch.delenv("SYNC_CHECKER_URL", raising=False)
    monkeypatch.setattr(httpx, "post", lambda *_args, **_kwargs: pytest.fail("must not post"))
    result = handler._handle_sync_changes(_request({}), {})
    assert result == {
        "jsonrpc": "2.0", "id": "sync-test",
        "error": {"code": -32603, "message": "SYNC_CHECKER_URL is not configured"},
    }


@pytest.mark.parametrize(("doc_id", "expected_body"), [
    (None, None), ("", None), (0, None), ("  ", {"doc_id": ""}), (" d1 ", {"doc_id": "d1"}),
])
def test_handler_preserves_falsey_and_whitespace_doc_id_behavior(monkeypatch, doc_id, expected_body):
    monkeypatch.setenv("SYNC_CHECKER_URL", "https://checker.test/run")
    posted = []
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: posted.append(kwargs) or Response({}))
    handler._handle_sync_changes(_request({}), {"doc_id": doc_id})
    assert posted[-1] == {"json": expected_body, "timeout": 300}


def test_handler_preserves_http_and_decode_failure_details(monkeypatch):
    monkeypatch.setenv("SYNC_CHECKER_URL", "https://checker.test/run")
    failure = httpx.HTTPStatusError("synthetic status failure", request=None, response=None)

    class BadResponse:
        def raise_for_status(self):
            raise failure

    monkeypatch.setattr(httpx, "post", lambda *_args, **_kwargs: BadResponse())
    status_error = handler._handle_sync_changes(_request({}), {})
    assert status_error == {
        "jsonrpc": "2.0", "id": "sync-test",
        "error": {"code": -32603, "message": "Failed to trigger sync", "details": str(failure)},
    }

    class DecodeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            raise ValueError("synthetic decode failure")

    monkeypatch.setattr(httpx, "post", lambda *_args, **_kwargs: DecodeResponse())
    decode_error = handler._handle_sync_changes(_request({}), {})
    assert decode_error == {
        "jsonrpc": "2.0", "id": "sync-test",
        "error": {
            "code": -32603, "message": "Failed to trigger sync",
            "details": "synthetic decode failure",
        },
    }


def test_handler_keeps_truthy_non_string_failure_inside_generic_boundary(monkeypatch):
    monkeypatch.setenv("SYNC_CHECKER_URL", "https://checker.test/run")
    result = handler._handle_sync_changes(_request({}), {"doc_id": 3})
    assert result["error"]["message"] == "Failed to trigger sync"
    assert result["error"]["details"] == "'int' object has no attribute 'strip'"


def test_handler_keeps_malformed_decoded_result_failure_after_http_boundary(monkeypatch):
    monkeypatch.setenv("SYNC_CHECKER_URL", "https://checker.test/run")
    monkeypatch.setattr(httpx, "post", lambda *_args, **_kwargs: Response(["raw", "json"]))
    with pytest.raises(AttributeError, match="get"):
        handler._handle_sync_changes(_request({}), {})
