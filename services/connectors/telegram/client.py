"""Narrow asynchronous Telegram Bot API client.

Telegram file identifiers are the durable reference. Download URLs are built
only from Telegram's validated ``getFile`` response and are never returned.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import httpx


_API_ORIGIN = "https://api.telegram.org"
_FILE_PATH_SEGMENT = re.compile(r"^[A-Za-z0-9_.-]+$")


class TelegramAPIError(Exception):
    """Safe provider failure with a stable category and optional retry delay."""

    def __init__(self, category: str, *, retry_after: int | None = None):
        self.category = category
        self.retry_after = retry_after
        super().__init__(f"Telegram request failed ({category})")


@dataclass(frozen=True)
class TelegramFile:
    file_id: str
    file_path: str
    file_size: int | None = None


class TelegramClient:
    """One-request-per-call Telegram client; retries belong to the stage governor."""

    def __init__(
        self,
        bot_token: str,
        *,
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
    ) -> None:
        if not isinstance(bot_token, str) or not bot_token or "/" in bot_token:
            raise ValueError("A valid Telegram bot token is required")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self._bot_token = bot_token
        self._timeout = min(float(timeout), 90.0)
        self._http = http_client or httpx.AsyncClient(follow_redirects=False)
        self._owns_http = http_client is None

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def send_message(
        self,
        chat_id: int | str,
        text: str,
        *,
        message_thread_id: int | None = None,
        parse_mode: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"chat_id": chat_id, "text": text}
        if message_thread_id is not None:
            payload["message_thread_id"] = message_thread_id
        if parse_mode is not None:
            payload["parse_mode"] = parse_mode
        return await self._call("sendMessage", payload)

    async def delete_webhook(self) -> dict[str, Any]:
        """Delete this bot's webhook while retaining Telegram's pending updates."""
        url = f"{_API_ORIGIN}/bot{self._bot_token}/deleteWebhook"
        response = await self._request("POST", url, json={})
        if response.status_code >= 300:
            raise self._http_error(response)
        try:
            envelope = response.json()
        except (ValueError, UnicodeDecodeError):
            raise TelegramAPIError("invalid_response") from None
        if not isinstance(envelope, dict) or envelope.get("ok") is not True:
            raise TelegramAPIError("provider_error")
        if envelope.get("result") is not True:
            raise TelegramAPIError("invalid_response")
        return {"ok": True, "result": True}

    async def edit_message_text(
        self,
        chat_id: int | str,
        message_id: int,
        text: str,
        *,
        parse_mode: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"chat_id": chat_id, "message_id": message_id, "text": text}
        if parse_mode is not None:
            payload["parse_mode"] = parse_mode
        return await self._call(
            "editMessageText",
            payload,
        )

    async def get_file(self, file_id: str) -> TelegramFile:
        if not isinstance(file_id, str) or not file_id:
            raise ValueError("file_id is required")
        result = await self._call("getFile", {"file_id": file_id})
        file_path = result.get("file_path")
        if not isinstance(file_path, str) or not self._valid_file_path(file_path):
            raise TelegramAPIError("invalid_response")
        result_file_id = result.get("file_id", file_id)
        if not isinstance(result_file_id, str) or not result_file_id:
            raise TelegramAPIError("invalid_response")
        size = result.get("file_size")
        if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size < 0):
            raise TelegramAPIError("invalid_response")
        return TelegramFile(file_id=result_file_id, file_path=file_path, file_size=size)

    async def download_audio(self, file_id: str) -> dict[str, Any]:
        """Download by provider file ID and return the in-memory audio payload."""
        file = await self.get_file(file_id)
        url = f"{_API_ORIGIN}/file/bot{self._bot_token}/{file.file_path}"
        response = await self._request("GET", url)
        if response.status_code >= 300:
            raise self._http_error(response)
        mime_type = response.headers.get("content-type", "application/octet-stream").split(";", 1)[0].strip()
        data = response.content
        return {"data": data, "mime_type": mime_type, "size": len(data)}

    async def cleanup_download(self, item: Any) -> None:
        """Audio downloads are memory-only, so cleanup has no external work."""
        return None

    async def _call(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{_API_ORIGIN}/bot{self._bot_token}/{method}"
        response = await self._request("POST", url, json=payload)
        if response.status_code >= 300:
            raise self._http_error(response)
        try:
            envelope = response.json()
        except (ValueError, UnicodeDecodeError):
            raise TelegramAPIError("invalid_response") from None
        if not isinstance(envelope, dict):
            raise TelegramAPIError("invalid_response")
        if envelope.get("ok") is not True:
            code = envelope.get("error_code")
            retry_after = self._retry_after(envelope)
            if code == 429 or retry_after is not None:
                category = "rate_limited"
            elif code in (401, 403):
                category = "authorization"
            elif code == 404:
                category = "not_found"
            else:
                category = "provider_error"
            raise TelegramAPIError(category, retry_after=retry_after)
        result = envelope.get("result")
        if not isinstance(result, dict):
            raise TelegramAPIError("invalid_response")
        return result

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Perform one bounded request without retries or redirect following."""
        try:
            response = await self._http.request(
                method,
                url,
                timeout=self._timeout,
                follow_redirects=False,
                **kwargs,
            )
        except httpx.TimeoutException:
            raise TelegramAPIError("timeout") from None
        except httpx.HTTPError:
            raise TelegramAPIError("transport") from None
        if not self._is_api_origin(response.url) or response.history:
            raise TelegramAPIError("invalid_response")
        return response

    def _http_error(self, response: httpx.Response) -> TelegramAPIError:
        retry_after = self._retry_after_response(response)
        if response.status_code == 429 or retry_after is not None:
            category = "rate_limited"
        elif response.status_code in (401, 403):
            category = "authorization"
        elif response.status_code == 404:
            category = "not_found"
        elif response.status_code >= 500:
            category = "unavailable"
        else:
            category = "provider_error"
        return TelegramAPIError(category, retry_after=retry_after)

    @staticmethod
    def _retry_after(envelope: dict[str, Any]) -> int | None:
        parameters = envelope.get("parameters")
        value = parameters.get("retry_after") if isinstance(parameters, dict) else None
        return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None

    @classmethod
    def _retry_after_response(cls, response: httpx.Response) -> int | None:
        try:
            envelope = response.json()
        except (ValueError, UnicodeDecodeError):
            return None
        return cls._retry_after(envelope) if isinstance(envelope, dict) else None

    @staticmethod
    def _is_api_origin(url: httpx.URL) -> bool:
        return url.scheme == "https" and url.host == "api.telegram.org" and url.port in (None, 443)

    @classmethod
    def _valid_file_path(cls, path: str) -> bool:
        if not path or path.startswith("/") or "\\" in path or "%" in path or "?" in path or "#" in path:
            return False
        segments = path.split("/")
        return all(
            segment not in ("", ".", "..") and _FILE_PATH_SEGMENT.fullmatch(segment)
            for segment in segments
        )

