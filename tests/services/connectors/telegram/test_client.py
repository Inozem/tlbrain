"""Offline tests for the narrow Telegram Bot API client."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from services.connectors.telegram.client import TelegramAPIError, TelegramClient
from services.connectors.telegram.formatting import rich_markdown_safe_chunk_indices


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


def test_rich_send_and_edit_use_native_payload_without_parse_mode():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 11}})

    async def run():
        client, http = _client(handler)
        try:
            body = "📁 **Root**\n- `file_name` & <value>"
            await client.send_message(
                10, body, message_thread_id=7, presentation="rich_markdown",
            )
            await client.edit_message_text(
                10, 11, body, presentation="rich_markdown",
                reply_markup={"inline_keyboard": []},
            )
        finally:
            await http.aclose()

    asyncio.run(run())
    assert [request.url.path.rsplit("/", 1)[-1] for request in requests] == [
        "sendRichMessage", "editMessageText",
    ]
    assert requests[0].content == (
        '{"chat_id":10,"rich_message":{"markdown":"📁 **Root**\\n- `file_name` & <value>"},'
        '"message_thread_id":7}'
    ).encode()
    assert requests[1].content == (
        '{"chat_id":10,"message_id":11,"rich_message":{"markdown":"📁 **Root**\\n- `file_name` & <value>"},'
        '"reply_markup":{"inline_keyboard":[]}}'
    ).encode()


@pytest.mark.parametrize("kwargs", [
    {"parse_mode": "MarkdownV2"}, {"reply_markup": {"inline_keyboard": []}},
])
def test_rich_send_rejects_unsupported_options_before_http(kwargs):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"ok": True, "result": {"message_id": 11}})

    async def run():
        client, http = _client(handler)
        try:
            with pytest.raises(ValueError):
                await client.send_message(10, "**bold**", presentation="rich_markdown", **kwargs)
        finally:
            await http.aclose()

    asyncio.run(run())
    assert requests == []


def test_rich_chunk_safety_keeps_lists_and_falls_back_for_split_constructs():
    tree = "* **Root & Co**\n  * **Nested_Folder**\n* **Another**"
    assert rich_markdown_safe_chunk_indices([(0, tree)]) == {0}
    assert rich_markdown_safe_chunk_indices([(
        0, "> **Quote**\n1) **Folder**\n- **_unassigned**",
    )]) == {0}
    assert rich_markdown_safe_chunk_indices([(0, "Literal <tag> text")]) == set()

    assert rich_markdown_safe_chunk_indices([
        (0, "**" + "a" * 3998), (1, "b" * 3998 + "**"),
    ]) == set()
    assert rich_markdown_safe_chunk_indices([
        (0, "[label](https://example.test/"), (1, "path)") ,
    ]) == set()
    assert rich_markdown_safe_chunk_indices([
        (0, "```python\n" + "x" * 3989),
        (1, "y" * 4000),
        (2, "z" * 3989 + "\n```"),
    ]) == set()


def test_callback_methods_validate_boolean_results_and_markup_is_sent():
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.path.endswith(("sendMessage", "editMessageText")):
            return httpx.Response(200, json={"ok": True, "result": {"message_id": 11}})
        return httpx.Response(200, json={"ok": True, "result": True})

    async def run():
        client, http = _client(handler)
        try:
            markup = {"inline_keyboard": [[{"text": "Save", "callback_data": "save:token"}]]}
            await client.send_message(10, "ready", reply_markup=markup)
            await client.edit_message_text(10, 11, "saving", reply_markup={"inline_keyboard": []})
            assert await client.answer_callback_query("query-id") is True
            assert await client.send_chat_action(10, "typing") is True
        finally:
            await http.aclose()

    asyncio.run(run())
    assert len(requests) == 4
    assert b'"reply_markup":{"inline_keyboard":[[{"text":"Save","callback_data":"save:token"}]]}' in requests[0].content
    assert b'"reply_markup":{"inline_keyboard":[]}' in requests[1].content
    assert requests[2].url.path.endswith("/answerCallbackQuery")
    assert requests[3].url.path.endswith("/sendChatAction")


def test_boolean_method_rejects_non_boolean_result():
    def handler(_request):
        return httpx.Response(200, json={"ok": True, "result": {}})

    async def run():
        client, http = _client(handler)
        try:
            await client.answer_callback_query("query-id")
        finally:
            await http.aclose()

    with pytest.raises(TelegramAPIError) as caught:
        asyncio.run(run())
    assert caught.value.category == "invalid_response"


@pytest.mark.parametrize("description", [
    "Bad Request: message is not modified",
    ("Bad Request: message is not modified: specified new message content and reply "
     "markup are exactly the same as a current content and reply markup of the message"),
])
@pytest.mark.parametrize("status_code", [200, 400])
def test_message_not_modified_is_only_applied_to_known_edits(description, status_code):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status_code, json={
            "ok": False, "error_code": 400,
            "description": description,
        })

    async def run():
        client, http = _client(handler)
        try:
            with pytest.raises(TelegramAPIError) as edited:
                await client.edit_message_text(10, 11, "same")
            with pytest.raises(TelegramAPIError) as sent:
                await client.send_message(10, "same")
            with pytest.raises(TelegramAPIError) as answered:
                await client.answer_callback_query("query-id")
            return edited.value.category, sent.value.category, answered.value.category
        finally:
            await http.aclose()

    assert asyncio.run(run()) == ("not_modified", "provider_error", "provider_error")
    assert [request.url.path.rsplit("/", 1)[-1] for request in requests] == [
        "editMessageText", "sendMessage", "answerCallbackQuery",
    ]


def test_not_modified_lookalike_description_is_not_treated_as_applied_edit():
    def handler(_request):
        return httpx.Response(400, json={
            "ok": False, "error_code": 400,
            "description": "Bad Request: message is not modified because the target changed",
        })

    async def run():
        client, http = _client(handler)
        try:
            await client.edit_message_text(10, 11, "same")
        finally:
            await http.aclose()

    with pytest.raises(TelegramAPIError) as caught:
        asyncio.run(run())
    assert caught.value.category == "provider_error"


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


@pytest.mark.parametrize("use_async_guard", [False, True])
@pytest.mark.parametrize("stop_at", ["before_get_file", "before_download"])
def test_download_audio_checks_call_local_guard_around_get_file(use_async_guard, stop_at):
    requests = []
    checks = []

    def handler(request):
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(200, json={"ok": True, "result": {"file_path": "voice/clip.ogg"}})
        return httpx.Response(200, content=b"audio", headers={"content-type": "audio/ogg"})

    async def run():
        client, http = _client(handler)
        async def async_guard_cb():
            checks.append(len(requests))
            if len(checks) == (1 if stop_at == "before_get_file" else 2):
                raise PermissionError("stale parent")
        def sync_guard():
            checks.append(len(requests))
            if len(checks) == (1 if stop_at == "before_get_file" else 2):
                raise PermissionError("stale parent")
        try:
            with pytest.raises(PermissionError, match="stale parent"):
                await client.download_audio(
                    "provider-id", assert_current=async_guard_cb if use_async_guard else sync_guard,
                )
        finally:
            await http.aclose()

    asyncio.run(run())
    assert len(requests) == (0 if stop_at == "before_get_file" else 1)
    assert checks == ([0] if stop_at == "before_get_file" else [0, 1])


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
