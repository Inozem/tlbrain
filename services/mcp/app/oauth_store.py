"""Shared OAuth records for Cloud Run instances; bearer values are stored as hashes."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Callable, Sequence

from google.cloud import firestore

from core.integrations.firestore.client import get_client


COLLECTION = "mcp_oauth"


def _reference(db, kind: str, secret: str):
    digest = hashlib.sha256(secret.encode()).hexdigest()
    return db.collection(COLLECTION).document(f"{kind}-{digest}")


def record(data: dict, lifetime: int | None = None) -> dict:
    value = dict(data)
    if lifetime is not None:
        value["expires_at"] = datetime.now(timezone.utc) + timedelta(seconds=lifetime)
    return value


def _valid(value: dict | None) -> bool:
    return value is not None and (
        "expires_at" not in value or value["expires_at"] > datetime.now(timezone.utc)
    )


def save(kind: str, secret: str, value: dict) -> None:
    _reference(get_client(), kind, secret).set(value)


def load(kind: str, secret: str) -> dict | None:
    if not secret or len(secret) > 512:
        return None
    snapshot = _reference(get_client(), kind, secret).get()
    value = snapshot.to_dict() if snapshot.exists else None
    return value if _valid(value) else None


def exchange(
    kind: str,
    secret: str,
    accepts: Callable[[dict], bool],
    replacements: Sequence[tuple[str, str, dict]] = (),
) -> dict | None:
    """Consume a code/state/refresh token and write replacements atomically."""
    if not secret or len(secret) > 512:
        return None
    db = get_client()
    reference = _reference(db, kind, secret)

    @firestore.transactional
    def consume(transaction):
        snapshot = reference.get(transaction=transaction)
        value = snapshot.to_dict() if snapshot.exists else None
        if not _valid(value) or not accepts(value):
            return None
        transaction.delete(reference)
        for next_kind, next_secret, next_value in replacements:
            transaction.set(_reference(db, next_kind, next_secret), next_value)
        return value

    return consume(db.transaction())
