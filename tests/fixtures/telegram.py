"""Scriptable Telegram HTTP transport for offline connector tests."""

from __future__ import annotations

from collections import deque
from typing import Any

import httpx
import pytest


class TelegramHTTPFake:
    """A transport-only fake that records requests and returns scripted results."""

    def __init__(self) -> None:
        self.trace: list[dict[str, Any]] = []
        self._responses: deque[httpx.Response | BaseException] = deque()
        self.transport = httpx.MockTransport(self._handle)

    def queue_response(
        self,
        *,
        status_code: int = 200,
        json: Any = None,
        content: bytes | None = None,
    ) -> None:
        self._responses.append(httpx.Response(status_code, json=json, content=content))

    def queue_error(self, error: BaseException) -> None:
        self._responses.append(error)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.trace.append({
            "method": request.method,
            "url": str(request.url),
            "headers": dict(request.headers),
            "content": request.content,
        })
        if not self._responses:
            raise RuntimeError("No scripted Telegram HTTP response")
        result = self._responses.popleft()
        if isinstance(result, BaseException):
            raise result
        return result


@pytest.fixture
def telegram_http_fake() -> TelegramHTTPFake:
    return TelegramHTTPFake()
