"""Safe Telegram webhook registration for deployment tooling."""

from __future__ import annotations

import http.client
import json
from typing import Any

from core.integrations.firestore.telegram_secret import (
    get_or_create_telegram_webhook_secret,
)


def register_telegram_webhook(
    *,
    client: Any,
    bot_token: str,
    webhook_url: str,
    telegram_client: Any | None = None,
) -> bool:
    """Store the webhook secret and register the messages webhook safely.

    An injected Telegram client remains available for callers that already have
    one. Deploy callers can instead provide only the bot token and webhook URL;
    this path makes a bounded synchronous Bot API request. Provider details and
    exceptions are deliberately reduced to a boolean result so request URLs or
    credentials cannot escape through this helper.
    """
    try:
        secret = get_or_create_telegram_webhook_secret(client)
        if telegram_client is not None:
            response = telegram_client.set_webhook(
                url=webhook_url,
                secret_token=secret,
                allowed_updates=["message"],
            )
        else:
            connection = http.client.HTTPSConnection("api.telegram.org", timeout=10)
            try:
                payload = json.dumps({
                    "url": webhook_url,
                    "secret_token": secret,
                    "allowed_updates": ["message"],
                })
                connection.request(
                    "POST",
                    f"/bot{bot_token}/setWebhook",
                    body=payload,
                    headers={"Content-Type": "application/json"},
                )
                api_response = connection.getresponse()
                if not 200 <= api_response.status < 300:
                    return False
                response_body = api_response.read(1_048_577)
                if len(response_body) > 1_048_576:
                    return False
                response = json.loads(response_body.decode("utf-8"))
            finally:
                connection.close()
        if isinstance(response, dict):
            return response.get("ok") is True
        return response is True
    except Exception:
        return False


__all__ = ["register_telegram_webhook"]
