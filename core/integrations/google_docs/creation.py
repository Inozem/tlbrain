"""Recoverable native Google Doc creation helpers."""

from __future__ import annotations

from dataclasses import dataclass

from core.integrations.google_drive.files import (
    create_native_doc,
    find_native_docs_by_save_intent,
)
from core.operations.models import OperationResult


@dataclass(frozen=True, slots=True)
class GoogleDocCreation:
    """Outcome of a marker lookup, including a confirmed document when found."""

    state: str
    document_id: str | None = None


def create_marked_google_doc(
    drive,
    *,
    save_intent_id: str,
    parent_id: str,
    title: str,
) -> OperationResult:
    """Issue the one-shot native Docs create request for a durable save intent."""
    document_id = create_native_doc(drive, title, parent_id, save_intent_id)
    return OperationResult(document_id=document_id, title=title)


def reconcile_google_doc_by_marker(
    drive,
    *,
    save_intent_id: str,
    expected_parent_id: str,
) -> GoogleDocCreation:
    """Read-only lookup for a previously dispatched native Doc create."""
    matches = find_native_docs_by_save_intent(drive, save_intent_id, expected_parent_id)
    if not matches:
        return GoogleDocCreation(state="creation_outcome_unknown")
    if len(matches) > 1:
        raise ValueError("multiple native Docs match the save intent")
    return GoogleDocCreation(state="confirmed", document_id=matches[0]["id"])
