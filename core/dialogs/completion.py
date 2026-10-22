"""Guarded finalization of confirmed Telegram dialog saves."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from core.integrations.firestore import save_intents
from core.integrations.firestore.mutations import (
    acquire_completion_guard,
    release_completion_guard,
)
from core.integrations.firestore.operations import OPERATIONS_COLLECTION, _result_from


def _db(client: Any | None) -> Any:
    if client is not None:
        return client
    from core.integrations.firestore.client import get_client

    return get_client()


def _aware(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    return value.astimezone(timezone.utc)


def _binding(value: dict[str, str]) -> dict[str, str]:
    fields = ("context_id", "draft_id", "manifest_digest")
    if not isinstance(value, dict) or any(not isinstance(value.get(key), str) or not value[key] for key in fields):
        raise ValueError("origin_binding must include context_id, draft_id, and manifest_digest")
    return {key: value[key] for key in fields}


def _replay(db: Any, operation_id: str, save_intent_id: str, origin_binding: dict[str, str]):
    operation_snapshot = db.collection(OPERATIONS_COLLECTION).document(operation_id).get()
    intent_snapshot = db.collection(save_intents.SAVE_INTENTS_COLLECTION).document(save_intent_id).get()
    if not operation_snapshot.exists or not intent_snapshot.exists:
        return None
    operation = operation_snapshot.to_dict() or {}
    intent = intent_snapshot.to_dict() or {}
    certificate = intent.get("completion_certificate") or {}
    result = _result_from(operation.get("result"))
    doc_id = operation.get("doc_id")
    if (operation.get("state") == "completed"
            and operation.get("save_intent_id") == save_intent_id
            and intent.get("save_intent_id") == save_intent_id
            and intent.get("dispatch_operation_id") == operation_id
            and intent.get("doc_id") == doc_id
            and certificate.get("operation_id") == operation_id
            and certificate.get("save_intent_id") == save_intent_id
            and certificate.get("doc_id") == doc_id
            and certificate.get("origin_binding") == origin_binding
            and result is not None and result.document_id == doc_id):
        return result
    return None


def finalize_save(
    client: Any | None,
    *,
    operation_id: str,
    save_intent_id: str,
    owner: str,
    fence: int,
    origin_binding: dict[str, str],
    now: datetime,
    clock: Callable[[], datetime] | None = None,
):
    """Read the saved document under a shared resource guard, then certify it atomically."""
    now = _aware(now)
    clock = clock or (lambda: datetime.now(timezone.utc))
    origin_binding = _binding(origin_binding)
    db = _db(client)
    replay = _replay(db, operation_id, save_intent_id, origin_binding)
    if replay is not None:
        return replay

    operation_snapshot = db.collection(OPERATIONS_COLLECTION).document(operation_id).get()
    intent_snapshot = db.collection(save_intents.SAVE_INTENTS_COLLECTION).document(save_intent_id).get()
    if not operation_snapshot.exists or not intent_snapshot.exists:
        return None
    operation = operation_snapshot.to_dict() or {}
    intent = intent_snapshot.to_dict() or {}
    doc_id = intent.get("doc_id")
    if (not doc_id or operation.get("save_intent_id") != save_intent_id
            or intent.get("dispatch_operation_id") != operation_id
            or intent.get("state") != save_intents.SaveIntentState.CONFIRMED.value):
        return None

    resource = f"document:{doc_id}"
    reservation = acquire_completion_guard(
        db, operation_id=operation_id, owner=owner, fence=fence,
        resource=resource, now=now, clock=clock,
    )
    if reservation is None:
        return None
    resource_epoch, reservation_id = reservation

    try:
        from core.indexing import processor
        from core.integrations.google_docs.reader import read_google_doc
        from core.integrations.google_drive.files import get_file_metadata

        raw_text = read_google_doc(doc_id)
        content_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        drive = get_file_metadata(doc_id)
        parents = drive.get("parents") or []
        if (drive.get("trashed") or drive.get("mimeType") != "application/vnd.google-apps.document"
                or len(parents) != 1 or not isinstance(parents[0], str)):
            return None
        parent_id = parents[0]
        transcript_snapshot = db.collection("transcript_index").document(doc_id).get()
        if not transcript_snapshot.exists:
            return None
        transcript = transcript_snapshot.to_dict() or {}
        metadata, _ = processor.parse_document(raw_text)
        provider = metadata.get("provider", "")
        if (not provider or transcript.get("provider") != provider
                or transcript.get("parent_id") != parent_id
                or transcript.get("resource_epoch", 0) != resource_epoch
                or not processor.verify_current_generation(
                    doc_id, parent_id, transcript.get("root_folder_id", ""), raw_text,
                )):
            return None
        return save_intents.finalize_save_completion(
            db, operation_id=operation_id, save_intent_id=save_intent_id,
            owner=owner, fence=fence, origin_binding=origin_binding, now=now,
            verified_content_hash=content_hash, verified_parent_id=parent_id,
            verified_provider=provider, resource_epoch=resource_epoch,
            guard_reservation_id=reservation_id, certificate_id=uuid.uuid4().hex,
            verified_at=_aware(clock()), clock=clock,
        )
    finally:
        release_completion_guard(
            db, resource=resource, operation_id=operation_id, owner=owner,
            fence=fence, reservation_id=reservation_id, now=_aware(clock()),
        )
