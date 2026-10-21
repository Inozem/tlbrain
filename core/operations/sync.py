"""Protocol-neutral sync-checker use case."""

from __future__ import annotations

import os
from typing import Any


class SyncConfigurationError(RuntimeError):
    """The trusted sync-checker endpoint is not configured."""


def sync_changes(doc_id: Any = None) -> Any:
    """Trigger the configured checker, preserving its raw JSON result."""
    checker_url = os.environ.get("SYNC_CHECKER_URL", "")
    if not checker_url:
        raise SyncConfigurationError("SYNC_CHECKER_URL is not configured") from None

    body: dict[str, Any] = {}
    if doc_id:
        body["doc_id"] = doc_id.strip()

    import httpx

    response = httpx.post(checker_url, json=body or None, timeout=300)
    response.raise_for_status()
    return response.json()
