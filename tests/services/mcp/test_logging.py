"""Offline logging and OAuth redaction characterization for the MCP service."""

from __future__ import annotations

import json
import logging
from urllib.parse import parse_qs, urlparse

import httpx

from core.utils.logging import configure_logging
from services.mcp.app import main


def _run(coroutine):
    import sniffio

    context_token = sniffio.current_async_library_cvar.set("asyncio")
    try:
        try:
            coroutine.send(None)
        except StopIteration as completed:
            return completed.value
    finally:
        sniffio.current_async_library_cvar.reset(context_token)
        coroutine.close()
    raise AssertionError("in-process ASGI coroutine unexpectedly suspended")


def _get(path: str):
    async def request():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://mcp.test") as client:
            return await client.get(path)

    return _run(request())


def _json_events(captured: str) -> list[dict]:
    return [json.loads(line) for line in captured.splitlines() if line.strip()]


def test_authorize_forwards_oauth_params_without_logging_synthetic_secrets(capsys):
    configure_logging()
    params = {
        "client_id": "synthetic-client-id",
        "redirect_uri": "https://client.example.test/callback",
        "state": "synthetic-state-marker",
        "scope": "openid email",
        "access_token": "synthetic-access-token-marker",
        "refresh_token": "synthetic-refresh-token-marker",
        "id_token": "synthetic-id-token-marker",
        "client_secret": "synthetic-client-secret-marker",
        "Authorization": "Bearer synthetic-authorization-marker",
        "tracking_marker": "synthetic-additional-query-marker",
    }
    response = _get("/authorize?" + str(httpx.QueryParams(params)))

    assert response.status_code == 307
    assert parse_qs(urlparse(response.headers["location"]).query) == {
        key: [value] for key, value in params.items()
    }

    serialized = capsys.readouterr().err
    events = _json_events(serialized)
    [event] = [event for event in events if event["message"] == "authorize request"]
    assert event["severity"] == "INFO"
    sensitive_values = (
        params["access_token"],
        params["refresh_token"],
        params["id_token"],
        params["client_secret"],
        params["Authorization"],
        params["state"],
        params["tracking_marker"],
    )
    leaked = [value for value in sensitive_values if value in serialized]
    assert leaked == []


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
