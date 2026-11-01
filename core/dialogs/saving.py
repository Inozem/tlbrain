"""Shared checkpointed dialog save pipeline for protocol adapters."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from core.dialogs.classification import FolderSelectionRequest, classify_dialog_folder
from core.dialogs.formatting import format_dialog
from core.dialogs.models import DialogUtterance, normalize_dialog_input
from core.integrations.firestore.input_parts import read_input_parts, write_input_parts
from core.integrations.firestore.operations import (
    OPERATIONS_COLLECTION, create_operation, get_operation, validate_operation_lease,
)
from core.integrations.firestore.outbox import (
    OutboxTargetKind, create_outbox_record, get_outbox_record,
)
from core.integrations.firestore.save_intents import (
    SaveIntentState, bind_save_intent_parent, create_save_intent, dispatch_save_intent,
    get_save_intent, reconcile_save_intent, update_save_intent_checkpoints,
)
from core.integrations.google_docs.creation import create_marked_google_doc, reconcile_google_doc_by_marker
from core.integrations.google_docs.fill import fill_google_doc_safely
from core.integrations.google_drive.files import read_native_doc_metadata
from core.operations.models import OperationOrigin, OperationResult
from core.operations.retries import RetryGovernor, UnknownOutcome


@dataclass(frozen=True, slots=True)
class AcceptedSave:
    operation: Any
    save_intent_id: str
    immutable_input_digest: str
    immutable_input_ref: str


@dataclass(frozen=True, slots=True)
class SaveProgress:
    operation_id: str
    save_intent_id: str
    document_id: str | None
    stages: tuple[str, ...]


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    return value.astimezone(timezone.utc)


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _folder_path(folder_id: str, folders: Mapping[str, Mapping[str, Any]], root_id: str) -> tuple[str, ...]:
    names: list[str] = []
    current = folder_id
    visited: set[str] = set()
    while current != root_id:
        if current in visited or current not in folders:
            raise ValueError("selected folder is not under the configured root")
        visited.add(current)
        row = folders[current]
        names.append(str(row.get("name", current)))
        current = row.get("parent_id", "")
    return tuple(reversed(names))


def _input_ref(operation_id: str) -> str:
    return f"dialog_operations/{operation_id}/input_parts/manifest"


def _intent_id(operation_id: str) -> str:
    return "dialog-save-" + hashlib.sha256(operation_id.encode("utf-8")).hexdigest()


def accept_save(
    client: Any,
    arguments: Mapping[str, object],
    *,
    operation_id: str,
    caller_binding: str,
    origin: OperationOrigin | str,
    now: datetime,
    origin_operation_ref: str | None = None,
    dialog_time: str = "",
    save_intent_id: str | None = None,
    write_document: Callable[[Any, dict[str, Any]], Any] | None = None,
) -> AcceptedSave:
    """Validate and durably accept a dialog without classification or provider I/O."""
    now = _aware(now)
    existing = get_operation(client, operation_id)
    acceptance_time = existing.created_at if existing is not None else now
    normalized = normalize_dialog_input(arguments, accepted_at=acceptance_time)
    content, speaker_ranges = format_dialog(normalized, dialog_time=dialog_time)
    # This is the complete immutable request, including generated defaults and exact content.
    serialized = _json_bytes({
        "version": 1, "provider": normalized.provider, "title": normalized.title,
        "date": normalized.dialog_date, "folder_path": normalized.folder_path,
        "folder_id": normalized.folder_id,
        "utterances": [{"speaker": item.speaker, "text": item.text} for item in normalized.utterances],
        "dialog_time": dialog_time, "desired_content": content,
        "speaker_ranges": speaker_ranges,
    })
    digest = hashlib.sha256(serialized).hexdigest()
    input_ref = _input_ref(operation_id)
    intent_id = save_intent_id or _intent_id(operation_id)
    if not intent_id or "/" in intent_id:
        raise ValueError("save_intent_id must be a non-empty Firestore document ID")
    write_input_parts(client, input_ref, serialized, write_document=write_document)
    # Verify before publishing the operation that points to the accepted input.
    if read_input_parts(client, input_ref) != serialized:
        raise ValueError("accepted immutable input failed verification")
    create_operation(
        client, operation_id=operation_id, origin=OperationOrigin(origin),
        origin_operation_ref=origin_operation_ref, caller_binding=caller_binding,
        immutable_input_digest=digest, immutable_input_ref=input_ref, provider=normalized.provider,
        title=normalized.title, date=normalized.dialog_date,
        explicit_folder_id=normalized.folder_id, save_intent_id=intent_id, now=now,
    )
    create_save_intent(
        client, intent_id, operation_id=operation_id,
        desired_content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        desired_parent_id_hash=None, now=now,
    )
    return AcceptedSave(get_operation(client, operation_id), intent_id, digest, input_ref)


def _load_accepted(client: Any, operation_id: str) -> dict[str, Any]:
    operation = get_operation(client, operation_id)
    if operation is None:
        raise ValueError("operation does not exist")
    raw = read_input_parts(client, operation.immutable_input_ref)
    if hashlib.sha256(raw).hexdigest() != operation.immutable_input_digest:
        raise ValueError("immutable input digest does not match operation")
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict) or data.get("version") != 1:
        raise ValueError("immutable dialog input has an unsupported shape")
    return data


def _store_selected_folder(
    client: Any, operation_id: str, folder_id: str, folder_path: tuple[str, ...],
    *, owner: str, fence: int, now: datetime,
) -> None:
    from google.cloud import firestore

    ref = client.collection(OPERATIONS_COLLECTION).document(operation_id)

    @firestore.transactional
    def _txn(transaction: Any) -> None:
        snapshot = ref.get(transaction=transaction)
        if not snapshot.exists:
            raise ValueError("operation does not exist")
        data = snapshot.to_dict() or {}
        expiry = data.get("lease_expires_at")
        if (data.get("state") != "running" or data.get("lease_owner") != owner
                or data.get("fence") != fence or not isinstance(expiry, datetime) or expiry <= now):
            raise RuntimeError("operation lease is unavailable")
        selected = data.get("selected_folder_id")
        if selected is not None and selected != folder_id:
            raise ValueError("operation folder is already resolved to a different folder")
        transaction.update(ref, {
            "selected_folder_id": folder_id, "selected_folder_path": list(folder_path), "updated_at": now,
        })

    _txn(client.transaction())


def _store_created_doc(
    client: Any, operation_id: str, document_id: str, *, owner: str, fence: int, now: datetime,
) -> None:
    """Persist confirmed create identity for status readers under the current fence."""
    from google.cloud import firestore

    ref = client.collection(OPERATIONS_COLLECTION).document(operation_id)

    @firestore.transactional
    def _txn(transaction: Any) -> None:
        snapshot = ref.get(transaction=transaction)
        if not snapshot.exists:
            raise ValueError("operation does not exist")
        data = snapshot.to_dict() or {}
        expiry = data.get("lease_expires_at")
        if (data.get("state") != "running" or data.get("lease_owner") != owner
                or data.get("fence") != fence or not isinstance(expiry, datetime) or expiry <= now):
            raise RuntimeError("operation lease is unavailable")
        current = data.get("doc_id")
        if current is not None and current != document_id:
            raise ValueError("operation is already bound to a different document")
        outcome = data.get("create_outcome")
        if outcome not in (None, "confirmed"):
            raise ValueError("operation create outcome conflicts with confirmed document")
        transaction.update(ref, {
            "doc_id": document_id, "create_outcome": "confirmed", "updated_at": now,
        })

    _txn(client.transaction())


@dataclass(frozen=True, slots=True)
class _SaveStageContext:
    client: Any
    operation_id: str
    owner: str
    fence: int
    now: datetime
    intent_id: str
    folders: Mapping[str, Mapping[str, Any]]
    root_folder_id: str
    speaker_index_lookup: Callable[[str], Any]
    suggestion: Callable[..., Any]
    drive: Any
    docs: Any
    accepted: Mapping[str, Any]
    content: str
    ranges: list[tuple[Any, ...]]
    ensure_unassigned: Callable[[str], str] | None


def _validate_save_input(client: Any, operation_id: str, immutable_input_digest: str) -> dict[str, str]:
    _load_accepted(client, operation_id)
    return {"immutable_input_digest": immutable_input_digest}


def _reconcile_save_input(client: Any, operation_id: str, immutable_input_digest: str) -> Any:
    try:
        _load_accepted(client, operation_id)
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return {"immutable_input_digest": immutable_input_digest}


def _classify_save_folder(
    context: _SaveStageContext, intent: Any,
) -> dict[str, Any]:
    accepted = context.accepted
    if intent.desired_parent_id is not None:
        request = FolderSelectionRequest(
            title=accepted["title"], utterances=(), folder_id=intent.desired_parent_id,
        )
        result = classify_dialog_folder(
            request, context.folders, context.root_folder_id,
            speaker_index_lookup=context.speaker_index_lookup, suggestion=context.suggestion,
        )
    else:
        request = FolderSelectionRequest(
            title=accepted["title"],
            utterances=tuple(DialogUtterance(**row) for row in accepted["utterances"]),
            reliable_speakers=tuple(dict.fromkeys(row["speaker"] for row in accepted["utterances"])),
            folder_path=tuple(accepted["folder_path"]) if accepted["folder_path"] is not None else None,
            folder_id=accepted.get("folder_id"),
        )
        result = classify_dialog_folder(
            request, context.folders, context.root_folder_id,
            speaker_index_lookup=context.speaker_index_lookup, suggestion=context.suggestion,
        )
    folder_snapshot = dict(context.folders)
    folder_id = intent.desired_parent_id or result.folder_id
    registered_unassigned = next((key for key, value in folder_snapshot.items()
                                  if value.get("name") == "_unassigned"
                                  and value.get("parent_id") == context.root_folder_id), None)
    if folder_id is None and result.requires_unassigned:
        folder_id = registered_unassigned
        if folder_id is None:
            if context.ensure_unassigned is None:
                raise ValueError("classification requires a registered root _unassigned folder")
            folder_id = context.ensure_unassigned(context.root_folder_id)
            if not isinstance(folder_id, str) or not folder_id:
                raise ValueError("ensure_unassigned did not return a folder ID")
            existing = folder_snapshot.get(folder_id)
            if existing is not None and (existing.get("name") != "_unassigned"
                                         or existing.get("parent_id") != context.root_folder_id):
                raise ValueError("ensure_unassigned returned a conflicting folder ID")
            folder_snapshot[folder_id] = {"name": "_unassigned", "parent_id": context.root_folder_id}
    if folder_id is None:
        raise ValueError("classification did not resolve a folder")
    path = _folder_path(folder_id, folder_snapshot, context.root_folder_id)
    _store_selected_folder(
        context.client, context.operation_id, folder_id, path,
        owner=context.owner, fence=context.fence, now=context.now,
    )
    parent_hash = hashlib.sha256(folder_id.encode("utf-8")).hexdigest()
    if bind_save_intent_parent(
        context.client, context.intent_id, operation_id=context.operation_id,
        desired_parent_id_hash=parent_hash, desired_parent_id=folder_id,
        owner=context.owner, fence=context.fence, now=context.now,
    ) is None:
        raise RuntimeError("could not bind save intent parent")
    return {"folder_id": folder_id, "folder_path": list(path)}


def _reconcile_save_folder(context: _SaveStageContext) -> Any:
    current = get_operation(context.client, context.operation_id)
    current_intent = get_save_intent(context.client, context.intent_id)
    if current is None or current_intent is None:
        return None
    selected_id = current.selected_folder_id
    if current_intent.desired_parent_id is not None:
        resolved = classify_dialog_folder(
            FolderSelectionRequest(
                title=context.accepted["title"], utterances=(), folder_id=current_intent.desired_parent_id,
            ), context.folders, context.root_folder_id,
            speaker_index_lookup=context.speaker_index_lookup, suggestion=context.suggestion,
        )
        if selected_id is not None and selected_id != resolved.folder_id:
            raise ValueError("operation folder conflicts with immutable save-intent parent")
        selected_id = resolved.folder_id
        path = _folder_path(selected_id, context.folders, context.root_folder_id)
        _store_selected_folder(
            context.client, context.operation_id, selected_id, path,
            owner=context.owner, fence=context.fence, now=context.now,
        )
        current = get_operation(context.client, context.operation_id)
    if selected_id is None:
        return None
    if current_intent.desired_parent_id_hash is None:
        bound = bind_save_intent_parent(
            context.client, context.intent_id, operation_id=context.operation_id,
            desired_parent_id_hash=hashlib.sha256(selected_id.encode("utf-8")).hexdigest(),
            desired_parent_id=selected_id,
            owner=context.owner, fence=context.fence, now=context.now,
        )
        if bound is None:
            return None
    return {
        "folder_id": selected_id,
        "folder_path": list(current.selected_folder_path or _folder_path(
            selected_id, context.folders, context.root_folder_id,
        )),
    }


def _reconcile_save_create(context: _SaveStageContext, parent_id: str) -> Any:
    found = reconcile_google_doc_by_marker(
        context.drive, save_intent_id=context.intent_id, expected_parent_id=parent_id,
    )
    if found.state != "confirmed":
        return None
    result = OperationResult(document_id=found.document_id, title=context.accepted["title"])
    reconcile_save_intent(
        context.client, context.intent_id, owner=context.owner, fence=context.fence,
        now=context.now, external_result=result, operation_id=context.operation_id,
    )
    return {"document_id": result.document_id, "title": result.title}


def _create_save_document(context: _SaveStageContext, parent_id: str) -> dict[str, Any]:
    current = get_save_intent(context.client, context.intent_id)
    if current is None:
        raise ValueError("save intent is missing")
    if current.state in {SaveIntentState.UNKNOWN, SaveIntentState.DISPATCHED}:
        result = _reconcile_save_create(context, parent_id)
        if result is None:
            raise UnknownOutcome("document create outcome remains unresolved")
        return result
    result = dispatch_save_intent(
        context.client, context.intent_id, operation_id=context.operation_id,
        owner=context.owner, fence=context.fence, now=context.now,
        create_callback=lambda: create_marked_google_doc(
            context.drive, save_intent_id=context.intent_id,
            parent_id=parent_id, title=context.accepted["title"],
        ),
    )
    if result is None:
        raise UnknownOutcome("document create is not currently dispatchable")
    return {"document_id": result.document_id, "title": result.title}


def _fill_save_document(context: _SaveStageContext, document_id: str) -> dict[str, Any]:
    current_intent = get_save_intent(context.client, context.intent_id)
    if current_intent is None:
        raise ValueError("save intent is missing")
    if current_intent.fill_checkpoint_ref:
        checkpoint = json.loads(current_intent.fill_checkpoint_ref)
        if checkpoint.get("document_id") != document_id:
            raise ValueError("shared fill checkpoint belongs to a different document")
        return checkpoint
    document = context.docs.documents().get(documentId=document_id).execute()
    revision = document.get("revisionId") if isinstance(document, dict) else None
    if not isinstance(revision, str) or not revision:
        raise RuntimeError("document revision is unavailable")
    outcome = fill_google_doc_safely(
        context.docs, drive=context.drive, document_id=document_id, text=context.content,
        speaker_ranges=context.ranges, expected_revision_id=revision,
    )
    if outcome.state != "confirmed":
        raise RuntimeError(f"document fill was not confirmed ({outcome.state})")
    result = {"document_id": document_id, "revision_id": outcome.revision_id}
    checkpoint_ref = json.dumps(result, separators=(",", ":"), sort_keys=True)
    if update_save_intent_checkpoints(
        context.client, context.intent_id, operation_id=context.operation_id,
        owner=context.owner, fence=context.fence, now=context.now,
        fill_checkpoint_ref=checkpoint_ref,
    ) is None:
        raise RuntimeError("could not persist shared fill checkpoint")
    return result


def _reconcile_save_fill(context: _SaveStageContext, document_id: str) -> Any:
    try:
        return _fill_save_document(context, document_id)
    except Exception:
        return None


def _read_post_fill_metadata(context: _SaveStageContext, document_id: str, parent_id: str) -> dict[str, Any]:
    """Read metadata after the fill checkpoint so modifiedTime reflects the final write."""
    value = read_native_doc_metadata(context.drive, document_id)
    if value.get("mimeType") != "application/vnd.google-apps.document" or not value.get("modifiedTime"):
        raise RuntimeError("post-fill document metadata is unavailable")
    return {"modified_time": value["modifiedTime"], "parent_id": parent_id}


def _reconcile_post_fill_metadata(
    context: _SaveStageContext, document_id: str, parent_id: str,
) -> Any:
    try:
        return _read_post_fill_metadata(context, document_id, parent_id)
    except Exception:
        return None


def _dispatch_save_index(
    context: _SaveStageContext, document_id: str, parent_id: str, metadata: Mapping[str, Any],
) -> dict[str, str]:
    """Update import-owned index fields additively and persist the deterministic outbox checkpoint."""
    fields = {
        "doc_id": document_id, "parent_id": parent_id, "source_file": context.accepted["title"],
        "dialog_date": context.accepted["date"], "provider": context.accepted["provider"],
        "modified_time": metadata["modified_time"],
    }
    ref = context.client.collection("transcript_index").document(document_id)

    from google.cloud import firestore

    @firestore.transactional
    def _upsert(transaction: Any) -> None:
        snapshot = ref.get(transaction=transaction)
        if snapshot.exists:
            transaction.update(ref, fields)
        else:
            transaction.set(ref, {**fields, "status": "imported", "status_changed_at": context.now})

    _upsert(context.client.transaction())
    outbox_id = "index-dispatch-" + hashlib.sha256(
        f"{context.operation_id}\0{context.intent_id}\0{document_id}\0"
        f"{context.accepted['desired_content']}".encode("utf-8")
    ).hexdigest()
    existing_outbox = get_outbox_record(context.client, outbox_id)
    if existing_outbox is not None:
        if (existing_outbox.target_kind is not OutboxTargetKind.INDEX_DISPATCH
                or existing_outbox.document_id != document_id
                or existing_outbox.save_intent_id != context.intent_id
                or existing_outbox.generation != 1):
            raise ValueError("stable index outbox identity conflicts with this save")
    else:
        existing_outbox = create_outbox_record(
            context.client, outbox_id, target_kind=OutboxTargetKind.INDEX_DISPATCH, generation=1,
            schedule_at=context.now, now=context.now, operation_id=context.operation_id,
            document_id=document_id, save_intent_id=context.intent_id,
        )
    if update_save_intent_checkpoints(
        context.client, context.intent_id, operation_id=context.operation_id,
        owner=context.owner, fence=context.fence, now=context.now,
        index_checkpoint_ref=outbox_id,
    ) is None:
        raise RuntimeError("could not persist shared index checkpoint")
    return {"outbox_id": outbox_id}


def _reconcile_save_index_dispatch(
    context: _SaveStageContext, document_id: str, parent_id: str, metadata: Mapping[str, Any],
) -> Any:
    try:
        return _dispatch_save_index(context, document_id, parent_id, metadata)
    except Exception:
        return None


def advance_save(
    client: Any,
    operation_id: str,
    *,
    owner: str,
    fence: int,
    now: datetime,
    folders: Mapping[str, Mapping[str, Any]],
    root_folder_id: str,
    speaker_index_lookup: Callable[[str], Any],
    suggestion: Callable[..., Any],
    drive: Any,
    docs: Any,
    governor: RetryGovernor | None = None,
    ensure_unassigned: Callable[[str], str] | None = None,
) -> SaveProgress:
    """Advance accepted work through durable checkpoints under the caller's lease."""
    now = _aware(now)
    if not validate_operation_lease(client, operation_id, owner=owner, fence=fence, now=now):
        raise RuntimeError("operation lease is unavailable")
    operation = get_operation(client, operation_id)
    if operation is None:
        raise ValueError("operation does not exist")
    intent_id = operation.save_intent_id
    if not intent_id:
        raise ValueError("operation has no stable save intent ID")
    governor = governor or RetryGovernor(client)
    governor.run_stage(
        operation_id, "validate",
        lambda: _validate_save_input(client, operation_id, operation.immutable_input_digest),
        now=now, owner=owner, fence=fence,
        reconcile=lambda: _reconcile_save_input(client, operation_id, operation.immutable_input_digest),
    )
    stages: list[str] = ["validate"]
    accepted = _load_accepted(client, operation_id)
    intent = get_save_intent(client, intent_id)
    if intent is None:
        raise ValueError("accepted save intent is missing")
    content = accepted["desired_content"]
    ranges = [tuple(row) for row in accepted["speaker_ranges"]]
    context = _SaveStageContext(
        client=client, operation_id=operation_id, owner=owner, fence=fence, now=now,
        intent_id=intent_id, folders=folders, root_folder_id=root_folder_id,
        speaker_index_lookup=speaker_index_lookup, suggestion=suggestion, drive=drive, docs=docs,
        accepted=accepted, content=content, ranges=ranges,
        ensure_unassigned=ensure_unassigned,
    )
    selected = governor.run_stage(
        operation_id, "classify", lambda: _classify_save_folder(context, intent),
        now=now, owner=owner, fence=fence,
        reconcile=lambda: _reconcile_save_folder(context),
    )
    stages.append("classify")
    parent_id = selected["folder_id"]
    created = governor.run_stage(
        operation_id, "create", lambda: _create_save_document(context, parent_id),
        now=now, owner=owner, fence=fence,
        reconcile=lambda: _reconcile_save_create(context, parent_id),
    )
    stages.append("create")
    document_id = created.get("document_id")
    if not document_id:
        raise ValueError("create checkpoint has no document ID")
    _store_created_doc(client, operation_id, document_id, owner=owner, fence=fence, now=now)
    governor.run_stage(
        operation_id, "fill", lambda: _fill_save_document(context, document_id),
        now=now, owner=owner, fence=fence,
        reconcile=lambda: _reconcile_save_fill(context, document_id),
    )
    stages.append("fill")
    meta = governor.run_stage(
        operation_id, "metadata",
        lambda: _read_post_fill_metadata(context, document_id, parent_id),
        now=now, owner=owner, fence=fence,
        reconcile=lambda: _reconcile_post_fill_metadata(context, document_id, parent_id),
    )
    stages.append("metadata")
    governor.run_stage(
        operation_id, "index_dispatch",
        lambda: _dispatch_save_index(context, document_id, parent_id, meta),
        now=now, owner=owner, fence=fence,
        reconcile=lambda: _reconcile_save_index_dispatch(context, document_id, parent_id, meta),
    )
    stages.append("index_dispatch")
    return SaveProgress(operation_id, intent_id, document_id, tuple(stages))
