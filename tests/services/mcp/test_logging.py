"""Offline logging and OAuth redaction characterization for the MCP service."""

from __future__ import annotations

import json
import logging

import httpx

from core.utils.logging import configure_logging


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
