"""Offline logging and OAuth redaction characterization for the MCP service."""

from __future__ import annotations

import json
import logging

import httpx

from core.utils.logging import configure_logging
from services.mcp.app.mcp import handler
from services.mcp.app.mcp.schemas import JSONRPCRequest


def _json_events(captured: str) -> list[dict]:
    return [json.loads(line) for line in captured.splitlines() if line.strip()]


def test_nested_proxy_exception_markers_are_absent_from_serialized_logs(capsys):
    configure_logging()
    logger = logging.getLogger("services.mcp.logging.security")
    nested_marker = "synthetic-proxy-client-secret-marker"
    exception_marker = "synthetic-sdk-refresh-token-marker"

    proxy_error = httpx.ConnectError(
        f"synthetic proxy failure containing {exception_marker}",
        request=httpx.Request("POST", "https://proxy.example.test/token"),
    )
    try:
        raise proxy_error
    except httpx.ConnectError:
        logger.exception(
            "synthetic proxy request failed",
            extra={"proxy_error": {"response": {"credential": nested_marker}}},
        )

    serialized = capsys.readouterr().err
    [event] = _json_events(serialized)
    assert event["severity"] == "ERROR"
    assert event["message"] == "synthetic proxy request failed"
    assert "exception" in event
    leaked = [marker for marker in (nested_marker, exception_marker) if marker in serialized]
    assert leaked == []


def test_quoted_json_secret_in_log_message_is_redacted(capsys):
    configure_logging()
    logger = logging.getLogger("services.mcp.logging.json_secret")
    marker = "synthetic-opaque-value-73"

    logger.error(f'proxy response: {{"access_token": "{marker}"}}')

    [event] = _json_events(capsys.readouterr().err)
    assert marker not in event["message"]
    assert event["message"] == 'proxy response: {"access_token": "[REDACTED]"}'


def test_mcp_query_log_omits_client_query_and_folder_path_but_keeps_safe_counts(caplog, monkeypatch):
    query_marker = "synthetic-free-form-client-query"
    folder_marker = "synthetic-private-folder-name"
    monkeypatch.setattr(
        handler.read_operations,
        "query",
        lambda **_kwargs: (
            [{"doc_id": "synthetic-doc-41", "speaker": "Synthetic", "text": "synthetic transcript"}],
            {"total_matches": 3, "returned_segments": 1, "truncated": False},
        ),
    )
    caplog.set_level(logging.INFO, logger=handler.__name__)

    handler._handle_query(
        JSONRPCRequest(jsonrpc="2.0", id=1, method="tools/call"),
        {"query": query_marker, "folder_path": [folder_marker]},
    )

    [record] = [record for record in caplog.records if record.name == handler.__name__]
    serialized_record = json.dumps(record.__dict__, default=str)
    assert query_marker not in serialized_record
    assert folder_marker not in serialized_record
    assert record.tool == "query"
    assert record.hits_total == 3
    assert record.docs_returned == 1
    assert record.segments_returned == 1
