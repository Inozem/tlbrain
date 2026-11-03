"""Safe Telegram webhook registration for deployment tooling."""

from __future__ import annotations

import http.client
import json
import os
import re
import shutil
import subprocess
from typing import Any

from core.integrations.firestore.telegram_secret import (
    get_or_create_telegram_webhook_secret,
)


_GCLOUD_TIMEOUT_SECONDS = 20
_GCLOUD_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9._~+/-]+=*\Z")
_DEPLOY_AUTH_EXCLUDED_ENV = frozenset({
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_ALLOWED_USER_IDS",
    "TLDV_API_KEY",
    "TL_DV_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "QDRANT_API_KEY",
    "GOOGLE_REFRESH_TOKEN",
    "GOOGLE_CLIENT_SECRET",
})


def _find_gcloud() -> str | None:
    """Find the active Google Cloud CLI, including native Windows batch shims."""
    names = ("gcloud.cmd", "gcloud", "gcloud.exe") if os.name == "nt" else ("gcloud",)
    for name in names:
        executable = shutil.which(name)
        if executable:
            return executable
    return None


def _gcloud_environment() -> dict[str, str]:
    """Copy the process environment without Telegram or owner API secrets."""
    return {
        name: value
        for name, value in os.environ.items()
        if name not in _DEPLOY_AUTH_EXCLUDED_ENV
    }


def _active_gcloud_access_token() -> str:
    """Read a short-lived token from the active CLI without exposing failures."""
    executable = _find_gcloud()
    if not executable:
        raise RuntimeError("gcloud authentication is unavailable")

    arguments = [executable, "auth", "print-access-token", "--quiet"]
    try:
        result = subprocess.run(
            arguments,
            check=False,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            stdin=subprocess.DEVNULL,
            timeout=_GCLOUD_TIMEOUT_SECONDS,
            env=_gcloud_environment(),
            shell=os.name == "nt" and executable.lower().endswith(".cmd"),
        )
    except (OSError, subprocess.TimeoutExpired):
        raise RuntimeError("gcloud authentication is unavailable") from None

    token = result.stdout.strip()
    if result.returncode != 0 or not _GCLOUD_TOKEN_PATTERN.fullmatch(token):
        raise RuntimeError("gcloud authentication is unavailable")
    return token


def create_deployment_firestore_client(*, project: str) -> Any:
    """Create a deploy-only Firestore client using the active gcloud identity.

    This deliberately avoids ADC discovery: webhook registration runs as the
    identity already selected by the deployment's gcloud configuration.
    """
    if not isinstance(project, str) or not project.strip():
        raise ValueError("an explicit Firestore project is required")

    token = _active_gcloud_access_token()
    from google.cloud import firestore
    from google.oauth2.credentials import Credentials

    credentials = Credentials(token=token)
    return firestore.Client(project=project, credentials=credentials)


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
                allowed_updates=["message", "callback_query"],
            )
        else:
            connection = http.client.HTTPSConnection("api.telegram.org", timeout=10)
            try:
                payload = json.dumps({
                    "url": webhook_url,
                    "secret_token": secret,
                    "allowed_updates": ["message", "callback_query"],
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


__all__ = ["create_deployment_firestore_client", "register_telegram_webhook"]
