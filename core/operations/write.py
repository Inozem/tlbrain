"""Protocol-neutral, leased write use cases shared by tool adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Callable
from uuid import uuid4

from core.config import get_root_folder_id
from core.documents import operations as document_operations
from core.indexing import user_facts
from core.integrations.firestore import client as firestore_client
from core.integrations.firestore import folders as folder_repository, transcripts
from core.integrations.firestore.operations import (
    claim_operation, complete_operation, create_operation, fail_operation,
    validate_operation_lease,
)
from core.integrations.firestore.write_plans import (
    WritePlanConflict, digest, get_write_plan, persist_write_plan,
)
from core.integrations.google_drive import client as drive_client
from core.integrations.google_drive import files as drive_files
from core.integrations.google_drive import folder_service
from core.integrations.qdrant import writer as qdrant_writer
from core.integrations.qdrant import reader as qdrant_reader
from core.operations.effects import allocate_fact_id, allocate_folder_id
from core.operations.models import OperationError, OperationOrigin, OperationResult
from core.operations.mutations import (
    MutationConflict, MutationNotSent, MutationOutcomeUnknown, MutationPolicy,
    MutationSnapshot, execute_mutation,
)


class WriteError(RuntimeError):
    """Safe, protocol-neutral failure for a guarded shared write."""


@dataclass(frozen=True, slots=True)
class SystemClock:
    """Default injectable UTC clock."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class WriteContext:
    """Caller-owned durable operation/effect identity and current lease."""

    operation_id: str
    effect_id: str
    owner: str
    fence: int
    client: Any | None = None
    clock: Any = field(default_factory=SystemClock)
    _internal: bool = False

    def now(self) -> datetime:
        value = self.clock.now() if hasattr(self.clock, "now") else self.clock()
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("write clock must return a timezone-aware datetime")
        return value


def _json_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _client(context: WriteContext) -> Any:
    return context.client if context.client is not None else firestore_client.get_client()


def _lease_checkpoint(context: WriteContext, *, definitely_unsent: bool = False) -> None:
    """Stop provider bookkeeping when this callback no longer owns the lease."""
    try:
        valid = validate_operation_lease(
            _client(context), context.operation_id, owner=context.owner,
            fence=context.fence, now=context.now(),
        )
    except Exception:
        valid = False
    if valid:
        return
    error = WriteError("write operation lease is no longer current")
    if definitely_unsent:
        raise MutationNotSent(error) from None
    raise error from None


def _document_ids_in_folders(folder_ids: list[str], db=None) -> list[str]:
    wanted = set(folder_ids)
    return sorted(
        snapshot.id for snapshot in transcripts.stream_index_records(db=db)
        if (snapshot.to_dict() or {}).get("parent_id") in wanted
    )


def _context(context: WriteContext | None, arguments: dict[str, Any]) -> WriteContext:
    if context is not None:
        if (not isinstance(context, WriteContext) or not context.operation_id or not context.effect_id
                or not context.owner or isinstance(context.fence, bool) or context.fence < 1):
            raise ValueError("a valid leased write context is required")
        context.now()
        return context
    client = firestore_client.get_client()
    now = datetime.now(timezone.utc)
    operation_id = uuid4().hex
    owner = uuid4().hex
    input_digest = _json_digest(arguments)
    try:
        create_operation(
            client, operation_id=operation_id, origin=OperationOrigin.MCP,
            caller_binding="mcp-internal-write", immutable_input_digest=input_digest,
            immutable_input_ref="mcp-write:" + operation_id, now=now,
        )
        leased = claim_operation(
            client, operation_id, owner=owner, now=now, lease_seconds=300,
        )
    except Exception:
        raise WriteError("write operation could not be claimed") from None
    if leased is None:
        raise WriteError("write operation could not be claimed")
    return WriteContext(
        operation_id=operation_id, effect_id="mcp-write", owner=owner,
        fence=leased.fence, client=client, _internal=True,
    )


def _finish(context: WriteContext, metadata: dict[str, Any]) -> None:
    if not context._internal:
        return
    document_id = metadata.get("doc_id") if metadata.get("write_type") in {
        "move_transcript", "rename_transcript", "add_fact",
    } else None
    value = OperationResult(
        document_id=document_id,
        title=None,
    )
    if not complete_operation(
        _client(context), context.operation_id, owner=context.owner,
        fence=context.fence, result=value, now=context.now(),
    ):
        raise WriteError("write operation lease expired before completion")


def _fail(context: WriteContext, error: Exception) -> None:
    """Persist only a bounded domain error or a generic safe write failure."""
    if not context._internal or isinstance(error, MutationOutcomeUnknown):
        return
    message = _safe_error_message(error)
    try:
        fail_operation(
            _client(context), context.operation_id, owner=context.owner, fence=context.fence,
            error=OperationError(
                code=getattr(error, "code", "write_failed"), category="write",
                message=message, sanitized_message=message, retryable=False,
            ),
            now=context.now(),
        )
    except Exception:
        return


def _is_domain_error(error: Exception) -> bool:
    return isinstance(error, (
        document_operations.FolderPathNotFound,
        document_operations.AmbiguousFolderPathError,
        document_operations.TranscriptNotFound,
        folder_service.FolderOperationError,
        user_facts.AddFactInputError,
        user_facts.AddFactDocumentNotFound,
        user_facts.AddFactTranscriptNotSynced,
        MutationConflict,
        MutationOutcomeUnknown,
        WritePlanConflict,
    ))


def _safe_error_message(error: Exception) -> str:
    return str(error)[:500] if _is_domain_error(error) else "write failed"


def _read_plan(context: WriteContext, arguments_digest: str) -> dict[str, Any] | None:
    plan = get_write_plan(_client(context), context.operation_id, context.effect_id)
    if plan is not None and plan.get("arguments_digest") != arguments_digest:
        raise WritePlanConflict("effect plan belongs to different arguments")
    return plan


def _execute(
    context: WriteContext,
    arguments: dict[str, Any],
    prepare: Callable[[], dict[str, Any]],
) -> Any:
    try:
        return _execute_inner(context, arguments, prepare)
    except Exception as error:
        safe_error = error if _is_domain_error(error) else WriteError("write outcome is unresolved")
        _fail(context, safe_error)
        if safe_error is error:
            raise
        raise safe_error from None


def _execute_inner(
    context: WriteContext,
    arguments: dict[str, Any],
    prepare: Callable[[], dict[str, Any]],
) -> Any:
    """Persist a metadata plan and execute through the callbacks rebuilt from its winner."""
    arguments_digest = _json_digest(arguments)
    planned = _read_plan(context, arguments_digest)
    if planned is None:
        prepared_metadata = prepare()
        before_digest = prepared_metadata.pop("_expected_pre_state", None) or digest(
            _read_snapshot(prepared_metadata, context)
        )
        plan = {
            "arguments_digest": arguments_digest,
            "metadata": prepared_metadata,
            "expected_pre_state": before_digest,
        }
        planned = persist_write_plan(
            _client(context), operation_id=context.operation_id, effect_id=context.effect_id,
            owner=context.owner, fence=context.fence, now=context.now(),
            arguments_digest=arguments_digest, plan=plan, clock=context.now,
        )
    metadata = planned["plan"]["metadata"]
    metadata, read_state, apply = _prepare_from_plan(metadata, arguments, context)
    expected_pre_state = planned["plan"]["expected_pre_state"]
    desired_state = metadata["desired_state"]
    desired_digest = digest(desired_state)
    resources = tuple(metadata["resources"])
    result_kind = metadata["result_kind"]

    def safe_read() -> MutationSnapshot:
        state = read_state()
        return MutationSnapshot(digest(state))

    def result_from_snapshot(snapshot: MutationSnapshot) -> Any:
        if snapshot.state_digest != desired_digest:
            raise RuntimeError("provider state does not match planned result")
        if result_kind == "move_transcript":
            unassigned_id = folder_repository.get_folder_id_by_name("_unassigned", _client(context))
            unassigned = (transcripts.list_unassigned_transcripts(unassigned_id, _client(context))
                          if unassigned_id else [])
            return {"unassigned_transcripts": unassigned}
        if result_kind == "move_folder":
            return [metadata["folder_id"], metadata["parent_id"]]
        if result_kind == "none":
            return None
        return metadata.get("result_value")

    result = execute_mutation(
            _client(context), operation_id=context.operation_id, effect_id=context.effect_id,
            owner=context.owner, fence=context.fence, kind=metadata["kind"], resources=resources,
            payload_digest=arguments_digest, desired_state_digest=desired_digest,
            expected_pre_state=expected_pre_state, expected_revision=None,
            policy=MutationPolicy.DRIVE_METADATA, now=context.now(), read_back=safe_read,
            mutation_callback=lambda _revision: _normalize_result(result_kind, apply(metadata)),
            result_from_snapshot=result_from_snapshot, clock=context.now,
        )
    result = _decode_result(result_kind, result)
    _finish(context, metadata)
    return result


def _normalize_result(kind: str, result: Any) -> Any:
    if kind == "move_transcript":
        return {"unassigned_transcripts": result.unassigned_transcripts}
    if kind == "move_folder":
        return list(result)
    return result


def _decode_result(kind: str, result: Any) -> Any:
    if kind == "move_transcript":
        return document_operations.MoveTranscriptResult(result.get("unassigned_transcripts", []))
    if kind == "move_folder":
        return tuple(result)
    return result


def _snapshot_move_transcript(metadata: dict[str, Any], context: WriteContext) -> dict[str, Any]:
    drive = drive_files.get_file_metadata(metadata["doc_id"])
    record = transcripts.get_transcript_record(metadata["doc_id"], _client(context)) or {}
    qparents = qdrant_reader.get_parent_ids_by_doc_ids(
        [metadata["doc_id"]], metadata["root_id"],
    ).get(metadata["doc_id"])
    return {"drive_parent": (drive.get("parents") or [None])[0],
            "firestore_parent": record.get("parent_id"),
            "firestore_status": record.get("status"),
            "firestore_modified_time": record.get("modifiedTime"),
            "firestore_content_hash": record.get("content_hash"),
            "firestore_version": record.get("version"), "qdrant_parents": qparents}


def _snapshot_rename_transcript(metadata: dict[str, Any], context: WriteContext) -> dict[str, Any]:
    drive = drive_files.get_file_metadata(metadata["doc_id"])
    current = transcripts.get_transcript_record(metadata["doc_id"], _client(context)) or {}
    return {"drive_name": drive.get("name"), "firestore_name": current.get("source_file")}


def _snapshot_move_folder(metadata: dict[str, Any], context: WriteContext) -> dict[str, Any]:
    current_desc = sorted(set(folder_service.expand_registered_subtree(metadata["folder_id"], _client(context))))
    current_docs = _document_ids_in_folders(current_desc, _client(context))
    drive = drive_files.get_file_metadata(metadata["folder_id"])
    data = folder_repository.get_folder_by_id(metadata["folder_id"], _client(context)) or {}
    return {"drive_parent": (drive.get("parents") or [None])[0],
            "firestore_parent": data.get("parent_id"), "descendants": current_desc,
            "docs": current_docs}


def _snapshot_rename_folder(metadata: dict[str, Any], context: WriteContext) -> dict[str, Any]:
    drive = drive_files.get_file_metadata(metadata["folder_id"])
    current = folder_repository.get_folder_by_id(metadata["folder_id"], _client(context)) or {}
    current_desc = sorted(set(folder_service.expand_registered_subtree(metadata["folder_id"], _client(context))))
    current_docs = _document_ids_in_folders(current_desc, _client(context))
    return {"drive_name": drive.get("name"), "firestore_name": current.get("name"),
            "descendants": current_desc, "docs": current_docs}


def _snapshot_create_folder(metadata: dict[str, Any], context: WriteContext) -> dict[str, Any]:
    drive = drive_files.get_file_metadata_if_exists(metadata["folder_id"])
    if drive is not None:
        drive = {
            "id": drive.get("id"), "name": drive.get("name"),
            "mimeType": drive.get("mimeType"), "parents": list(drive.get("parents") or []),
            "trashed": drive.get("trashed"),
            "marker": (drive.get("appProperties") or {}).get("tlbrain_effect_intent"),
        }
    record = folder_repository.get_folder_by_id(metadata["folder_id"], _client(context))
    if record is not None:
        record = {"id": metadata["folder_id"], "name": record.get("name"),
                  "parent": record.get("parent_id")}
    return {"drive": drive, "firestore": record}


def _snapshot_add_fact(metadata: dict[str, Any], context: WriteContext) -> dict[str, Any]:
    current = transcripts.get_transcript_record(metadata["doc_id"], _client(context)) or {}
    point = qdrant_reader.get_user_fact_payload(metadata["point_id"])
    return {"status": current.get("status"), "parent_id": current.get("parent_id"),
            "dialog_date": current.get("dialog_date", ""),
            "point_digest": _json_digest(point) if point is not None else None}


def _read_snapshot(metadata: dict[str, Any], context: WriteContext) -> dict[str, Any]:
    """Read one domain snapshot using the canonical provider readback for its write kind."""
    readers = {
        "move_transcript": _snapshot_move_transcript,
        "rename_transcript": _snapshot_rename_transcript,
        "move_folder": _snapshot_move_folder,
        "rename_folder": _snapshot_rename_folder,
        "create_folder": _snapshot_create_folder,
        "add_fact": _snapshot_add_fact,
    }
    try:
        reader = readers[metadata["write_type"]]
    except KeyError:
        raise WriteError("persisted write plan is invalid") from None
    return reader(metadata, context)


def _prepare_from_plan(metadata: dict[str, Any], arguments: dict[str, Any], context: WriteContext):
    """Rebuild mutation callbacks from durable IDs without resolving old paths."""
    kind = metadata["write_type"]
    if kind == "move_transcript":
        apply = lambda m: document_operations.apply_move_transcript(
            m["doc_id"], m["target_id"], m["root_id"], single_attempt=True,
            db=_client(context), before_bookkeeping=lambda: _lease_checkpoint(context),
        )
    elif kind == "rename_transcript":
        apply = lambda m: document_operations.apply_rename_transcript(
            m["doc_id"], m["new_title"], _client(context),
            before_bookkeeping=lambda: _lease_checkpoint(context),
        )
    elif kind == "move_folder":
        apply = lambda m: folder_service.apply_move_folder(
            m["folder_id"], m["parent_id"], db=_client(context),
            before_bookkeeping=lambda: _lease_checkpoint(context),
        )
    elif kind == "rename_folder":
        apply = lambda m: _apply_rename_folder_guarded(m, context)
    elif kind == "create_folder":
        apply = lambda m: _apply_create_folder(m, context)
    elif kind == "add_fact":
        text = arguments["text"].strip()
        apply = lambda m: _apply_fact(m, text, context)
    else:
        raise WriteError("persisted write plan is invalid")
    return metadata, lambda: _read_snapshot(metadata, context), apply


def move_transcript(doc_id: str, folder_path: list[str], *, context: WriteContext | None = None):
    arguments = {"doc_id": doc_id, "folder_path": folder_path}
    ctx = _context(context, arguments)

    def prepare():
        target_id = document_operations.resolve_move_target(folder_path, _client(ctx))
        root_id = get_root_folder_id()
        metadata = {"kind": "parent_move", "write_type": "move_transcript",
                    "result_kind": "move_transcript", "doc_id": doc_id, "target_id": target_id,
                    "root_id": root_id, "resources": [f"document:{doc_id}", f"tree:{root_id}"],
                    "desired_state": None}
        before = _read_snapshot(metadata, ctx)
        metadata["_expected_pre_state"] = digest(before)
        qdrant_before = before["qdrant_parents"]
        metadata["desired_state"] = {
            "drive_parent": target_id, "firestore_parent": target_id,
            "firestore_status": "imported", "firestore_modified_time": None,
            "firestore_content_hash": None, "firestore_version": None,
            "qdrant_parents": [target_id] * len(qdrant_before) if qdrant_before is not None else None,
        }
        return metadata
    return _execute(ctx, arguments, prepare)


def rename_transcript(doc_id: str, new_title: str, *, context: WriteContext | None = None) -> None:
    arguments = {"doc_id": doc_id, "new_title": new_title}
    ctx = _context(context, arguments)
    def prepare():
        record = transcripts.get_transcript_record(doc_id, _client(ctx))
        if not record:
            raise document_operations.TranscriptNotFound(doc_id)
        metadata = {"kind": "content_update", "write_type": "rename_transcript",
                    "result_kind": "none", "doc_id": doc_id, "new_title": new_title,
                    "resources": [f"document:{doc_id}"], "desired_state": None,
                    }
        metadata["_expected_pre_state"] = digest(_read_snapshot(metadata, ctx))
        metadata["desired_state"] = {"drive_name": new_title, "firestore_name": new_title}
        return metadata
    _execute(ctx, arguments, prepare)


def move_folder(folder_path: list[str], new_parent_path: list[str] | None = None, *, context: WriteContext | None = None):
    arguments = {"folder_path": folder_path, "new_parent_path": new_parent_path or []}
    ctx = _context(context, arguments)
    def prepare():
        folder_id, parent_id = folder_service.prepare_move_folder(
            folder_path, new_parent_path, db=_client(ctx),
        )
        root_id = get_root_folder_id()
        descendants = sorted(set(folder_service.expand_registered_subtree(folder_id, _client(ctx))))
        docs = _document_ids_in_folders(descendants, _client(ctx))
        resources = sorted({f"tree:{root_id}", *(f"folder:{value}" for value in descendants),
                            *(f"document:{value}" for value in docs)})
        metadata = {"kind": "parent_move", "write_type": "move_folder",
                    "result_kind": "move_folder", "folder_id": folder_id, "parent_id": parent_id,
                    "root_id": root_id, "descendants": descendants, "docs": docs,
                    "resources": resources, "desired_state": None}
        before = _read_snapshot(metadata, ctx)
        if before["descendants"] != descendants or before["docs"] != docs:
            raise MutationConflict("folder subtree changed during write planning")
        metadata["_expected_pre_state"] = digest(before)
        metadata["desired_state"] = {**before, "drive_parent": parent_id, "firestore_parent": parent_id}
        return metadata
    return _execute(ctx, arguments, prepare)


def rename_folder(folder_path: list[str], new_name: str, *, context: WriteContext | None = None) -> str:
    arguments = {"folder_path": folder_path, "new_name": new_name}
    ctx = _context(context, arguments)
    def prepare():
        folder_id = folder_service.resolve_registered_folder_path(folder_path, _client(ctx))
        if not folder_id:
            raise folder_service.FolderNotFound(folder_path, "folder")
        if len(folder_id) > 1:
            raise folder_service.AmbiguousFolderPath(folder_path, "folder", len(folder_id))
        folder_id = folder_id[0]
        data = folder_repository.get_folder_by_id(folder_id, _client(ctx)) or {}
        parent_id = data.get("parent_id", "")
        if folder_repository.folder_name_exists(new_name, parent_id, _client(ctx)):
            raise folder_service.FolderNameConflict(new_name)
        root_id = get_root_folder_id()
        descendants = sorted(set(folder_service.expand_registered_subtree(folder_id, _client(ctx))))
        docs = _document_ids_in_folders(descendants, _client(ctx))
        metadata = {"kind": "content_update", "write_type": "rename_folder",
                    "result_kind": "string", "folder_id": folder_id, "new_name": new_name,
                    "parent_id": parent_id, "descendants": descendants, "docs": docs,
                    "resources": sorted({f"tree:{root_id}", f"folder:{folder_id}",
                                          *(f"folder:{value}" for value in descendants),
                                          *(f"document:{value}" for value in docs)}),
                    "desired_state": None,
                    "result_value": folder_id}
        before = _read_snapshot(metadata, ctx)
        if before["descendants"] != descendants or before["docs"] != docs:
            raise MutationConflict("folder subtree changed during write planning")
        metadata["_expected_pre_state"] = digest(before)
        if metadata["_expected_pre_state"] != digest(_read_snapshot(metadata, ctx)):
            raise MutationConflict("folder subtree changed during write planning")
        metadata["desired_state"] = {**before, "drive_name": new_name, "firestore_name": new_name}
        return metadata
    return _execute(ctx, arguments, prepare)


def create_folder(name: str, parent_path: list[str] | None = None, *, context: WriteContext | None = None) -> str:
    arguments = {"name": name, "parent_path": parent_path or []}
    ctx = _context(context, arguments)
    def prepare():
        parent_id = folder_service.prepare_create_folder(name, parent_path, db=_client(ctx))
        intent = _json_digest({"name": name, "parent_id": parent_id})
        allocation = allocate_folder_id(
            _client(ctx), operation_id=ctx.operation_id, effect_id=ctx.effect_id,
            intent_digest=intent, owner=ctx.owner, fence=ctx.fence, now=ctx.now(),
            clock=ctx.now,
        )
        folder_id, marker = allocation["resource_id"], allocation["intent_marker"]
        root_id = get_root_folder_id()
        metadata = {"kind": "provider_call", "write_type": "create_folder",
                    "result_kind": "string", "name": name, "parent_id": parent_id,
                    "folder_id": folder_id, "marker": marker,
                    "resources": [f"tree:{root_id}", f"folder:{parent_id}", f"folder:{folder_id}"],
                    "desired_state": {"drive": {"id": folder_id, "name": name,
                                                   "mimeType": "application/vnd.google-apps.folder",
                                                   "parents": [parent_id], "trashed": False, "marker": marker},
                                      "firestore": {"id": folder_id, "name": name, "parent": parent_id}},
                    "result_value": folder_id}
        metadata["_expected_pre_state"] = digest(_read_snapshot(metadata, ctx))
        return metadata
    return _execute(ctx, arguments, prepare)


def _apply_create_folder(metadata: dict[str, Any], context: WriteContext | None = None) -> str:
    db = _client(context) if context is not None else None
    try:
        folder_service.validate_folder_name_available(metadata["name"], metadata["parent_id"], db=db)
    except folder_service.FolderOperationError as error:
        raise MutationNotSent(error) from None
    except Exception:
        raise MutationNotSent(WriteError("folder validation is unavailable")) from None
    folder_id = drive_client.create_folder(
        metadata["name"], metadata["parent_id"], folder_id=metadata["folder_id"],
        intent_marker=metadata["marker"],
    )
    if context is not None:
        _lease_checkpoint(context)
    folder_repository.upsert_folder(folder_id, metadata["name"], metadata["parent_id"], db)
    return folder_id


def _apply_rename_folder_guarded(metadata: dict[str, Any], context: WriteContext | None = None) -> str:
    db = _client(context) if context is not None else None
    try:
        folder_service.validate_folder_name_available(
            metadata["new_name"], metadata["parent_id"], exclude_id=metadata["folder_id"], db=db,
        )
    except folder_service.FolderOperationError as error:
        raise MutationNotSent(error) from None
    except Exception:
        raise MutationNotSent(WriteError("folder validation is unavailable")) from None
    return folder_service.apply_rename_folder(
        metadata["folder_id"], metadata["new_name"], validate_name=False, db=db,
        before_bookkeeping=(lambda: _lease_checkpoint(context)) if context is not None else None,
    )


def add_fact(doc_id: str, text: str, *, context: WriteContext | None = None) -> str:
    """Validate trimmed fact input and store a metadata-only plan before indexing."""
    arguments = {"doc_id": doc_id, "text": text}
    clean_doc_id = doc_id.strip()
    clean_text = text.strip()
    if not clean_doc_id or not clean_text:
        raise user_facts.AddFactInputError("doc_id and text are required")
    ctx = _context(context, {"doc_id": clean_doc_id, "text": clean_text})
    def prepare():
        clean_id, clean_fact, payload = user_facts.prepare_user_fact(
            clean_doc_id, clean_text, db=_client(ctx),
        )
        root_id = payload["root_folder_id"]
        intent = _json_digest({"doc_id": clean_doc_id, "text_digest": _json_digest(clean_text),
                               "root_id": root_id, "parent_id": payload["parent_id"]})
        preserve_dedup = ctx._internal
        id_factory = (lambda: qdrant_writer.user_fact_point_id(clean_doc_id, clean_text)) if preserve_dedup else None
        allocation_kwargs = {"id_factory": id_factory} if id_factory is not None else {}
        point_id = allocate_fact_id(
            _client(ctx), operation_id=ctx.operation_id, effect_id=ctx.effect_id,
            intent_digest=intent, owner=ctx.owner, fence=ctx.fence, now=ctx.now(),
            clock=ctx.now, **allocation_kwargs,
        )
        metadata = {"kind": "provider_call", "write_type": "add_fact",
                    "result_kind": "string", "doc_id": clean_doc_id, "point_id": point_id,
                    "root_id": root_id, "parent_id": payload["parent_id"],
                    "dialog_date": payload["dialog_date"], "dialog_date_num": payload["dialog_date_num"],
                    "payload_digest": _json_digest(payload),
                    "resources": [f"document:{clean_doc_id}"],
                    "desired_state": {"payload_digest": _json_digest(payload)},
                    "result_value": clean_doc_id}
        metadata["_expected_pre_state"] = digest(_read_snapshot(metadata, ctx))
        metadata["desired_state"] = {"status": "synced", "parent_id": payload["parent_id"],
                                     "dialog_date": payload["dialog_date"],
                                     "point_digest": _json_digest(payload)}
        return metadata
    return _execute(ctx, arguments, prepare)


def _apply_fact(metadata: dict[str, Any], text: str, context: WriteContext | None = None) -> str:
    payload = {"type": "user_fact", "doc_id": metadata["doc_id"], "text": text,
               "root_folder_id": metadata["root_id"], "parent_id": metadata["parent_id"],
               "dialog_date": metadata["dialog_date"], "dialog_date_num": metadata["dialog_date_num"]}
    user_facts.apply_user_fact_payload(
        payload, point_id=metadata["point_id"], single_attempt=True,
        before_write=(lambda: _lease_checkpoint(context, definitely_unsent=True))
        if context is not None else None,
    )
    return metadata["doc_id"]
