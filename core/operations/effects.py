"""Safe allocation API for stable external resource identifiers."""

from __future__ import annotations

from datetime import datetime
import re
from typing import Callable
from uuid import uuid4

from core.integrations.firestore.effects import (
    AllocationConflict,
    allocate_effect_resource,
    get_effect_allocation,
    validate_effect_allocation_lease,
)
from core.integrations.google_drive.client import generate_folder_id

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _allocate(client, *, operation_id, effect_id, kind, intent_digest, owner, fence, now, factory, clock=None):
    """Reuse a durable allocation before generating a candidate; Firestore picks the winner."""
    if (
        not isinstance(operation_id, str) or not operation_id or "/" in operation_id
        or not isinstance(effect_id, str) or not effect_id or "/" in effect_id
        or kind not in {"user_fact", "drive_folder"}
        or not isinstance(intent_digest, str) or not _DIGEST.fullmatch(intent_digest)
        or not isinstance(owner, str) or not owner
        or isinstance(fence, bool) or not isinstance(fence, int) or fence < 1
        or not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None
        or not callable(factory)
    ):
        raise ValueError("invalid effect resource allocation")
    try:
        existing = get_effect_allocation(client, operation_id, effect_id)
    except Exception:
        raise RuntimeError("resource ID allocation is unresolved") from None
    if existing is not None:
        if (existing.get("operation_id") != operation_id or existing.get("effect_id") != effect_id
                or existing.get("kind") != kind or existing.get("intent_digest") != intent_digest):
            raise AllocationConflict("effect resource identity conflicts")
        try:
            return allocate_effect_resource(
                client, operation_id=operation_id, effect_id=effect_id, kind=kind,
                intent_digest=intent_digest, owner=owner, fence=fence, now=now,
                candidate=existing.get("resource_id"), clock=clock,
            )
        except AllocationConflict:
            raise
        except Exception:
            raise RuntimeError("resource ID allocation is unresolved") from None
    try:
        if not validate_effect_allocation_lease(
            client, operation_id, owner=owner, fence=fence, now=now,
        ):
            raise AllocationConflict("current operation lease is required")
    except AllocationConflict:
        raise
    except Exception:
        raise RuntimeError("resource ID allocation is unresolved") from None
    try:
        candidate = factory()
    except Exception:
        raise RuntimeError("resource ID allocation failed") from None
    try:
        return allocate_effect_resource(
            client, operation_id=operation_id, effect_id=effect_id, kind=kind,
            intent_digest=intent_digest, owner=owner, fence=fence, now=now, candidate=candidate,
            clock=clock,
        )
    except AllocationConflict:
        raise
    except Exception:
        raise RuntimeError("resource ID allocation is unresolved") from None


def allocate_fact_id(
    client, *, operation_id: str, effect_id: str, intent_digest: str,
    owner: str, fence: int, now: datetime, id_factory: Callable[[], str] = lambda: str(uuid4()),
    clock: Callable[[], datetime] | None = None,
) -> str:
    """Return the persisted UUID point ID allocated for one effect."""
    return _allocate(client, operation_id=operation_id, effect_id=effect_id, kind="user_fact",
                     intent_digest=intent_digest, owner=owner, fence=fence, now=now,
                     factory=id_factory, clock=clock)["resource_id"]


def allocate_folder_id(
    client, *, operation_id: str, effect_id: str, intent_digest: str,
    owner: str, fence: int, now: datetime, id_factory: Callable[[], str] = generate_folder_id,
    clock: Callable[[], datetime] | None = None,
) -> dict[str, str]:
    """Return the persisted Drive folder ID and its stable hashed intent marker."""
    allocation = _allocate(client, operation_id=operation_id, effect_id=effect_id, kind="drive_folder",
                           intent_digest=intent_digest, owner=owner, fence=fence, now=now,
                           factory=id_factory, clock=clock)
    return {"resource_id": allocation["resource_id"], "intent_marker": allocation["intent_marker"]}
