from __future__ import annotations

import httpx
import pytest

pytest_plugins = ("tests.fixtures.telegram",)


def test_telegram_http_fake_traces_requests_and_scripts_response_or_error(telegram_http_fake):
    telegram_http_fake.queue_response(status_code=200, json={"ok": True, "result": {"message_id": 7}})
    with httpx.Client(transport=telegram_http_fake.transport) as client:
        response = client.post(
            "https://api.telegram.org/botpytest-synthetic-token/sendMessage",
            json={"chat_id": 42, "text": "synthetic message"},
        )
    assert response.json() == {"ok": True, "result": {"message_id": 7}}
    assert telegram_http_fake.trace[0]["method"] == "POST"
    assert telegram_http_fake.trace[0]["url"].endswith("/sendMessage")
    assert b"synthetic message" in telegram_http_fake.trace[0]["content"]

    telegram_http_fake.queue_error(httpx.ConnectError("scripted Telegram failure"))
    with httpx.Client(transport=telegram_http_fake.transport) as client:
        with pytest.raises(httpx.ConnectError, match="scripted Telegram failure"):
            client.get("https://api.telegram.org/botpytest-synthetic-token/getMe")
    assert len(telegram_http_fake.trace) == 2
