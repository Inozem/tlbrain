"""Offline tests for the narrow Telegram Bot API client."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from services.connectors.telegram.client import TelegramAPIError, TelegramClient


pytestmark = pytest.mark.offline

TOKEN = "synthetic-secret-token"
API = "https://api.telegram.org"


def _client(handler, *, timeout=30):
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport)
    return TelegramClient(TOKEN, http_client=http, timeout=timeout), http


def test_send_edit_and_get_file_use_fixed_origin_and_single_http_attempts():
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.path.endswith("sendMessage"):
            return httpx.Response(200, json={"ok": True, "result": {"message_id": 11}})
        if request.url.path.endswith("editMessageText"):
            return httpx.Response(200, json={"ok": True, "result": {"message_id": 11}})
        return httpx.Response(
            200,
            json={"ok": True, "result": {"file_id": "provider-id", "file_path": "voice/clip.ogg", "file_size": 4}},
        )

    async def run():
        client, http = _client(handler)
        try:
            sent = await client.send_message(10, "hello", message_thread_id=7)
            edited = await client.edit_message_text(10, 11, "updated")
            file = await client.get_file("provider-id")
            return sent, edited, file
        finally:
            await http.aclose()

    sent, edited, file = asyncio.run(run())
    assert sent == {"message_id": 11}
    assert edited == {"message_id": 11}
    assert file.file_id == "provider-id"
    assert file.file_path == "voice/clip.ogg"
    assert len(requests) == 3
    assert all(request.url.scheme == "https" and request.url.host == "api.telegram.org" for request in requests)
    assert all(TOKEN in request.url.path for request in requests)
    assert requests[0].content == b'{"chat_id":10,"text":"hello","message_thread_id":7}'


@pytest.mark.parametrize("result", [True, False])
def test_delete_webhook_confirms_true_result_without_dropping_pending_updates(result):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"ok": True, "result": result})

    async def run():
        client, http = _client(handler)
        try:
            return await client.delete_webhook()
        finally:
            await http.aclose()

    if result:
        assert asyncio.run(run()) == {"ok": True, "result": True}
    else:
        with pytest.raises(TelegramAPIError) as caught:
            asyncio.run(run())
        assert caught.value.category == "invalid_response"
    assert len(requests) == 1
    assert requests[0].url.path == f"/bot{TOKEN}/deleteWebhook"
    assert requests[0].content == b"{}"


def test_download_uses_only_validated_get_file_path_and_returns_memory_payload():
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.path.startswith("/bot"):
            return httpx.Response(200, json={"ok": True, "result": {"file_path": "voice/clip.ogg"}})
        return httpx.Response(200, content=b"audio", headers={"content-type": "audio/ogg; charset=binary"})

    async def run():
        client, http = _client(handler)
        try:
            result = await client.download_audio("provider-file-id")
            await client.cleanup_download(result)
            return result
        finally:
            await http.aclose()

    result = asyncio.run(run())
    assert result == {"data": b"audio", "mime_type": "audio/ogg", "size": 5}
    assert len(requests) == 2
    assert requests[0].url.path.endswith("/getFile")
    assert requests[1].url.path == f"/file/bot{TOKEN}/voice/clip.ogg"
    assert TOKEN not in repr(result)
    assert "url" not in result and "temporary_url" not in result


@pytest.mark.parametrize("path", ["../secret", "/foreign/path", "https://evil.test/file", "a%2f..%2fsecret", "a\\b"])
def test_invalid_get_file_paths_never_trigger_download(path):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"ok": True, "result": {"file_path": path}})

    async def run():
        client, http = _client(handler)
        try:
            await client.download_audio("provider-file-id")
        finally:
            await http.aclose()

    with pytest.raises(TelegramAPIError) as caught:
        asyncio.run(run())
    assert caught.value.category == "invalid_response"
    assert len(requests) == 1
    assert TOKEN not in str(caught.value)
    assert "api.telegram.org" not in str(caught.value)


def test_retry_after_is_exposed_safely_and_each_call_has_one_attempt():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            429,
            json={
                "ok": False,
                "error_code": 429,
                "description": f"secret body contains {TOKEN}",
                "parameters": {"retry_after": 17},
            },
        )

    async def run():
        client, http = _client(handler, timeout=120)
        try:
            await client.send_message(10, "hello")
        finally:
            await http.aclose()

    with pytest.raises(TelegramAPIError) as caught:
        asyncio.run(run())
    assert caught.value.category == "rate_limited"
    assert caught.value.retry_after == 17
    assert TOKEN not in str(caught.value)
    assert "secret body" not in str(caught.value)
    assert len(requests) == 1
    configured_timeouts = requests[0].extensions["timeout"]
    assert configured_timeouts
    assert all(value is None or value <= 90 for value in configured_timeouts.values())


def test_redirect_is_not_followed_and_error_does_not_expose_location_url():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(302, headers={"location": "https://evil.test/"})

    async def run():
        client, http = _client(handler)
        try:
            await client.get_file("provider-file-id")
        finally:
            await http.aclose()

    with pytest.raises(TelegramAPIError) as caught:
        asyncio.run(run())
    assert len(requests) == 1
    assert caught.value.category == "provider_error"
    assert "evil.test" not in str(caught.value)
