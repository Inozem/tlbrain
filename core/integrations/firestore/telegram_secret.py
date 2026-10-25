"""Transactional storage for the Telegram webhook secret."""

from __future__ import annotations

import re
import secrets
from collections.abc import Callable
from typing import Any

from google.cloud import firestore

from core.integrations.firestore.client import TOKENS_COLLECTION


TELEGRAM_WEBHOOK_SECRET_DOC = "telegram_webhook"
_SECRET_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,256}$")


def _valid_secret(value: Any) -> bool:
    return isinstance(value, str) and _SECRET_PATTERN.fullmatch(value) is not None


def get_or_create_telegram_webhook_secret(
    client: Any,
    secret_factory: Callable[[], str] | None = None,
) -> str:
    """Return the persisted webhook secret, creating it once transactionally."""
    factory = secret_factory or (lambda: secrets.token_urlsafe(32))
    document = client.collection(TOKENS_COLLECTION).document(TELEGRAM_WEBHOOK_SECRET_DOC)

    @firestore.transactional
    def _get_or_create(transaction: Any) -> str:
        snapshot = document.get(transaction=transaction)
        if snapshot.exists:
            data = snapshot.to_dict() or {}
            value = data.get("secret")
            if _valid_secret(value):
                return value
            raise ValueError("Stored Telegram webhook secret is invalid")

        value = factory()
        if not _valid_secret(value):
            raise ValueError("Generated Telegram webhook secret is invalid")
        transaction.set(document, {"secret": value})
        return value

    return _get_or_create(client.transaction())


__all__ = ["TELEGRAM_WEBHOOK_SECRET_DOC", "get_or_create_telegram_webhook_secret"]
