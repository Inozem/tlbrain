"""Static validated allowlist for common protocol-neutral operations."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

from core.operations import read, sync, write
from core.dialogs import saving


class OperationValidationError(ValueError):
    """Safe input diagnostic that never retains or displays supplied values."""


_ADD_DIALOG_CONTEXT_TOKEN = object()


@dataclass(frozen=True, slots=True)
class AddDialogContext:
    """Trusted acceptance data required by the dialog save use case."""

    client: Any
    operation_id: str
    caller_binding: str
    origin: Any
    now: Any
    _token: object | None = None


def _new_add_dialog_context(*, client: Any, operation_id: str, caller_binding: str,
                            origin: Any, now: Any) -> AddDialogContext:
    """Create the internal-only context used to accept a dialog operation."""
    return AddDialogContext(client, operation_id, caller_binding, origin, now, _ADD_DIALOG_CONTEXT_TOKEN)


@dataclass(frozen=True, slots=True)
class Field:
    kind: str
    required: bool = False
    minimum: int | None = None


@dataclass(frozen=True, slots=True)
class Operation:
    name: str
    arguments: Mapping[str, Field]
    function: Callable[..., Any]
    write: bool = False
    trim_fields: frozenset[str] = frozenset()


def _fields(**items: Field) -> Mapping[str, Field]:
    return MappingProxyType(items)


_STRING = Field("string")
_NONBLANK = Field("string", required=True)
_PATH = Field("path")
_REQUIRED_NONEMPTY_PATH = Field("path", required=True)
_DATE = Field("string")
_LIMIT = Field("integer", minimum=1)
_OFFSET = Field("integer", minimum=0)
_UTTERANCES = Field("utterances", required=True)
_DATE_ONLY = Field("date", minimum=10)


def _accept_add_dialog(arguments: dict[str, Any], *, context: AddDialogContext) -> Any:
    if (not isinstance(context, AddDialogContext) or context._token is not _ADD_DIALOG_CONTEXT_TOKEN
            or not context.operation_id or not context.caller_binding):
        _fail("a trusted dialog acceptance context is required")
    return saving.accept_save(
        context.client, arguments, operation_id=context.operation_id,
        caller_binding=context.caller_binding, origin=context.origin, now=context.now,
    )


OPERATIONS: Mapping[str, Operation] = MappingProxyType({
    "query": Operation("query", _fields(
        query=_NONBLANK, keywords=_STRING, folder_path=_PATH, date_from=_DATE, date_to=_DATE,
    ), read.query),
    "get_transcript": Operation("get_transcript", _fields(
        doc_id=_STRING, folder_path=_PATH, date_from=_DATE, date_to=_DATE,
        limit=_LIMIT,
    ), read.get_transcript),
    "list_recent_transcripts": Operation("list_recent_transcripts", _fields(
        folder_path=_PATH, date_from=_DATE, date_to=_DATE, limit=_LIMIT, offset=_OFFSET,
    ), read.list_recent_transcripts),
    "list_folders": Operation("list_folders", _fields(), read.list_folders),
    "sync_status": Operation("sync_status", _fields(), read.sync_status),
    "move_transcript": Operation("move_transcript", _fields(
        doc_id=_NONBLANK, folder_path=_REQUIRED_NONEMPTY_PATH,
    ), write.move_transcript, write=True, trim_fields=frozenset({"doc_id"})),
    "rename_transcript": Operation("rename_transcript", _fields(
        doc_id=_NONBLANK, new_title=_NONBLANK,
    ), write.rename_transcript, write=True, trim_fields=frozenset({"doc_id", "new_title"})),
    "move_folder": Operation("move_folder", _fields(
        folder_path=_REQUIRED_NONEMPTY_PATH, new_parent_path=_PATH,
    ), write.move_folder, write=True),
    "rename_folder": Operation("rename_folder", _fields(
        folder_path=_REQUIRED_NONEMPTY_PATH, new_name=_NONBLANK,
    ), write.rename_folder, write=True, trim_fields=frozenset({"new_name"})),
    "create_folder": Operation("create_folder", _fields(
        name=_NONBLANK, parent_path=_PATH,
    ), write.create_folder, write=True, trim_fields=frozenset({"name"})),
    "add_fact": Operation("add_fact", _fields(
        doc_id=_NONBLANK, text=_NONBLANK,
    ), write.add_fact, write=True),
    "add_dialog": Operation("add_dialog", _fields(
        provider=_NONBLANK, utterances=_UTTERANCES, title=_STRING,
        dialog_date=_DATE_ONLY, folder_path=_PATH,
    ), _accept_add_dialog, write=True),
    "sync_changes": Operation("sync_changes", _fields(doc_id=_STRING), sync.sync_changes),
})


def _fail(message: str) -> None:
    raise OperationValidationError(message) from None


def validate_arguments(name: str, arguments: Any) -> dict[str, Any]:
    """Validate one allowlisted operation's JSON-shaped argument object."""
    if not isinstance(name, str) or name not in OPERATIONS:
        _fail("unknown operation")
    if not isinstance(arguments, dict):
        _fail("arguments must be an object")

    operation = OPERATIONS[name]
    for key in arguments:
        if not isinstance(key, str) or key not in operation.arguments:
            _fail("unknown argument field")
    for key, field in operation.arguments.items():
        if field.required and key not in arguments:
            _fail(f"missing required field: {key}")
        if key not in arguments:
            continue
        value = arguments[key]
        if field.kind == "string":
            if not isinstance(value, str):
                _fail(f"field {key} must be a string")
            if field.required and not value.strip():
                _fail(f"field {key} must be nonblank")
        elif field.kind == "date":
            from datetime import date

            if (not isinstance(value, str) or len(value) != 10
                    or value[4:5] != "-" or value[7:8] != "-"):
                _fail(f"field {key} must be an ISO date")
            try:
                date.fromisoformat(value)
            except ValueError:
                _fail(f"field {key} must be an ISO date")
        elif field.kind == "utterances":
            if not isinstance(value, list) or not value:
                _fail(f"field {key} must be a nonempty list")
            for item in value:
                if not isinstance(item, dict) or set(item) != {"speaker", "text"}:
                    _fail(f"field {key} contains an invalid item")
                for item_key in ("speaker", "text"):
                    item_value = item[item_key]
                    if not isinstance(item_value, str) or not item_value.strip():
                        _fail(f"field {key} contains an invalid item")
                if "\n" in item["speaker"] or "\r" in item["speaker"]:
                    _fail(f"field {key} contains an invalid item")
                if not item["speaker"].replace("::", "").strip():
                    _fail(f"field {key} contains an invalid item")
        elif field.kind == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                _fail(f"field {key} must be an integer")
            if field.minimum is not None and value < field.minimum:
                _fail(f"field {key} is outside the allowed range")
        elif field.kind == "path":
            if not isinstance(value, list) or any(not isinstance(segment, str) for segment in value):
                _fail(f"field {key} must be a list of strings")
            if field.required and not value:
                _fail(f"field {key} must be nonempty")
            if any(not segment.strip() for segment in value):
                _fail(f"field {key} contains a blank path element")
            if name == "add_dialog" and any(not segment for segment in value):
                _fail(f"field {key} contains a blank path element")
    return dict(arguments)


def invoke_operation(name: str, arguments: Any, *, context: write.WriteContext | AddDialogContext | None = None,
                     retry_policy: Any | None = None) -> Any:
    """Validate and invoke exactly one shared operation, returning its raw result."""
    validated = validate_arguments(name, arguments)
    operation = OPERATIONS[name]
    if retry_policy is not None:
        from core.utils.retry import SINGLE_ATTEMPT_RETRY_POLICY
        if retry_policy is not SINGLE_ATTEMPT_RETRY_POLICY or name not in {
            "query", "get_transcript", "list_recent_transcripts", "list_folders", "sync_status",
        }:
            _fail("retry policy is not accepted for this operation")
        return operation.function(**validated, retry_policy=retry_policy)
    if name == "add_dialog":
        if not isinstance(context, AddDialogContext):
            _fail("a trusted dialog acceptance context is required")
        return operation.function(validated, context=context)
    if operation.write:
        if not isinstance(context, write.WriteContext):
            _fail("a trusted write context is required")
        normalized = {
            key: value.strip() if key in operation.trim_fields else value
            for key, value in validated.items()
        }
        return operation.function(**normalized, context=context)
    if context is not None:
        _fail("write context is not accepted for this operation")
    return operation.function(**validated)
