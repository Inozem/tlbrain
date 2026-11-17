"""Validated, private data contracts for context-v2 Telegram routing.

These records preserve provenance and scope.  They are not capabilities: a
reference and digest still need to be resolved and verified by the caller.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from services.connectors.telegram.conversations import ConversationKind


PROTOCOL = "context-v2"
SCHEMA_VERSION = 1
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "/" in value:
        raise ValueError(f"{name} must be a non-empty identifier without '/' ")
    return value


def _opaque_ref(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty reference")
    return value


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _items(value: Any, name: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be an array")
    return tuple(value)


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _DIGEST_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _freeze_json(value: Any, name: str) -> Any:
    """Validate JSON data and recursively freeze caller-owned containers."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        import math

        if math.isfinite(value):
            return value
        raise TypeError(f"{name} must contain finite JSON numbers")
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{name} mapping keys must be strings")
        return MappingProxyType({key: _freeze_json(item, name) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, name) for item in value)
    raise TypeError(f"{name} must contain only JSON-safe values")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _enum(enum_type: type[StrEnum], value: Any, name: str) -> Any:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unsupported {name}: {value!r}") from exc


def _only_keys(value: Mapping[str, Any], allowed: set[str], name: str) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{name} contains unknown fields: {sorted(unknown)}")


class SourceKind(StrEnum):
    ACTIVE_DOCUMENT = "active_document"
    EXPLICIT_REFERENCE = "explicit_reference"
    LOCAL_NUMBER = "local_number"
    VERIFIED_EXCHANGE = "verified_exchange"
    PRESENTATION_ENTRY = "presentation_entry"
    CONFIRMED_SAVE = "confirmed_save"


class ExplicitReferenceKind(StrEnum):
    ID = "id"
    URL = "url"
    LOCAL_NUMBER = "local_number"


class Disposition(StrEnum):
    EXECUTE = "execute"
    CLARIFY = "clarify"
    CANCEL = "cancel"
    REPLACE = "replace"
    DELEGATE = "delegate"


class Action(StrEnum):
    MOVE_TRANSCRIPT = "move_transcript"
    CREATE_FOLDER = "create_folder"
    CREATE_THEN_MOVE = "create_then_move"
    PURE_SWITCH = "pure_switch"
    SEARCH = "search"
    SUBSTANTIVE = "substantive"
    SAVE = "save"
    RETRY = "retry"
    ANNOTATION = "annotation"


class MissingSlot(StrEnum):
    DOCUMENT = "document"
    DESTINATION = "destination"
    PARENT = "parent"
    FOLDER_NAME = "folder_name"
    EXPLICIT_CONFLICT = "explicit_conflict"


class Certainty(StrEnum):
    CERTAIN = "certain"
    UNCERTAIN = "uncertain"


class DocumentSelectorKind(StrEnum):
    EXPLICIT_REFERENCE = "explicit_reference"
    DIALOG_NUMBER = "dialog_number"
    ACTIVE_DOCUMENT = "active_document"
    EVIDENCE = "evidence"
    SHOWN_POSITION = "shown_position"


class FolderSelectorKind(StrEnum):
    EXPLICIT_PATH = "explicit_path"
    EXPLICIT_ID = "explicit_id"
    NAME = "name"
    ROOT_DEFAULT = "root_default"
    CREATE_STEP_OUTPUT = "create_step_output"


@dataclass(frozen=True, slots=True)
class BoundRef:
    """Reference and digest with its server-captured scope."""

    ref: str
    digest: str
    context_id: str
    actor_id: int
    conversation_id: str
    operation_id: str
    checkpoint_id: str | None = None
    call_id: str | None = None
    ordinal: int | None = None
    payload: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _opaque_ref(self.ref, "ref")
        for name in ("context_id", "conversation_id", "operation_id"):
            _identifier(getattr(self, name), name)
        _digest(self.digest, "digest")
        _positive_int(self.actor_id, "actor_id")
        fragment_fields = (self.checkpoint_id, self.call_id, self.ordinal)
        if any(item is not None for item in fragment_fields):
            if self.checkpoint_id is None or self.call_id is None or self.ordinal is None:
                raise ValueError("fragment refs require checkpoint_id, call_id, and ordinal")
            _identifier(self.checkpoint_id, "checkpoint_id")
            _identifier(self.call_id, "call_id")
            _nonnegative_int(self.ordinal, "ordinal")
        if self.payload is not None:
            if not isinstance(self.payload, Mapping):
                raise TypeError("reference payload must be an object or None")
            object.__setattr__(self, "payload", _freeze_json(self.payload, "reference payload"))

    @property
    def is_fragment(self) -> bool:
        return self.checkpoint_id is not None

    def to_dict(self) -> dict[str, Any]:
        return {"ref": self.ref, "digest": self.digest, "context_id": self.context_id,
                "actor_id": self.actor_id, "conversation_id": self.conversation_id,
                "operation_id": self.operation_id, "checkpoint_id": self.checkpoint_id,
                "call_id": self.call_id, "ordinal": self.ordinal,
                "payload": _thaw_json(self.payload)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BoundRef":
        _only_keys(value, {"ref", "digest", "context_id", "actor_id", "conversation_id", "operation_id",
                           "checkpoint_id", "call_id", "ordinal", "payload"}, "BoundRef")
        return cls(**{key: value.get(key) for key in (
            "ref", "digest", "context_id", "actor_id", "conversation_id", "operation_id",
            "checkpoint_id", "call_id", "ordinal", "payload")})


@dataclass(frozen=True, slots=True)
class CommandSpan:
    """Python Unicode code-point [start,end) span into accepted_command."""

    kind: ExplicitReferenceKind
    start: int
    end: int
    raw_text: str
    parsed_value: str | int

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(ExplicitReferenceKind, self.kind, "reference kind"))
        if type(self.start) is not int or self.start < 0 or type(self.end) is not int or self.end <= self.start:
            raise ValueError("span must have non-negative start and end > start")
        if not isinstance(self.raw_text, str) or not self.raw_text:
            raise ValueError("raw_text must be non-empty")
        if self.kind is ExplicitReferenceKind.LOCAL_NUMBER:
            _positive_int(self.parsed_value, "parsed local number")
        elif not isinstance(self.parsed_value, str) or not self.parsed_value:
            raise ValueError("parsed ID/URL must be non-empty text")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "start": self.start, "end": self.end,
                "raw_text": self.raw_text, "parsed_value": self.parsed_value}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CommandSpan":
        _only_keys(value, {"kind", "start", "end", "raw_text", "parsed_value"}, "CommandSpan")
        return cls(**{key: value[key] for key in ("kind", "start", "end", "raw_text", "parsed_value")})


@dataclass(frozen=True, slots=True)
class DocumentEvidence:
    doc_id: str
    source_kind: SourceKind
    source: BoundRef
    context_id: str
    actor_id: int
    conversation_id: str
    operation_id: str
    fragment_refs: tuple[BoundRef, ...] = ()
    command_span: CommandSpan | None = None
    predecessor_operation_id: str | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _identifier(self.doc_id, "doc_id")
        object.__setattr__(self, "source_kind", _enum(SourceKind, self.source_kind, "source_kind"))
        for name in ("context_id", "conversation_id", "operation_id"):
            _identifier(getattr(self, name), name)
        _positive_int(self.actor_id, "actor_id")
        object.__setattr__(self, "fragment_refs", _items(self.fragment_refs, "fragment_refs"))
        if any(not isinstance(ref, BoundRef) or not ref.is_fragment for ref in self.fragment_refs):
            raise ValueError("fragment_refs must contain checkpoint/call/ordinal fragment refs")
        if not isinstance(self.source, BoundRef):
            raise TypeError("source must be a BoundRef")
        if self.command_span is not None and not isinstance(self.command_span, CommandSpan):
            raise TypeError("command_span must be a CommandSpan or None")
        if self.source_kind in (SourceKind.EXPLICIT_REFERENCE, SourceKind.LOCAL_NUMBER):
            expected_kind = (ExplicitReferenceKind.LOCAL_NUMBER
                             if self.source_kind is SourceKind.LOCAL_NUMBER else None)
            if self.command_span is None:
                raise ValueError("explicit provenance requires an exact command span")
            if expected_kind is not None and self.command_span.kind is not expected_kind:
                raise ValueError("local-number provenance requires a local-number span")
            if expected_kind is None and self.command_span.kind is ExplicitReferenceKind.LOCAL_NUMBER:
                raise ValueError("explicit-reference provenance requires an ID or URL span")
        if self.source_kind is SourceKind.CONFIRMED_SAVE:
            _identifier(self.predecessor_operation_id, "predecessor_operation_id")
        elif self.predecessor_operation_id is not None:
            raise ValueError("predecessor_operation_id is only valid for confirmed_save")
        if self.metadata is not None:
            if not isinstance(self.metadata, Mapping):
                raise TypeError("metadata must be an object or None")
            object.__setattr__(self, "metadata", _freeze_json(self.metadata, "metadata"))

    def validate_scope(self, *, context_id: str, actor_id: int,
                       conversation_ids: set[str], predecessor_operation_ids: set[str],
                       predecessor_refs: tuple[BoundRef, ...], selected_doc_id: str | None,
                       explicit_references: tuple[CommandSpan, ...],
                       presentation_refs: tuple[BoundRef, ...]) -> None:
        """Check anchored scope, allowing only explicit same-context predecessors."""
        if self.context_id != context_id or self.actor_id != actor_id:
            raise ValueError("evidence crosses context or actor scope")
        if self.source.context_id != context_id or self.source.actor_id != actor_id:
            raise ValueError("evidence source crosses context or actor scope")
        if self.source.conversation_id != self.conversation_id or self.source.operation_id != self.operation_id:
            raise ValueError("evidence source binding does not match evidence identity")
        if self.source_kind is SourceKind.CONFIRMED_SAVE:
            if self.predecessor_operation_id not in predecessor_operation_ids:
                raise ValueError("confirmed save lacks an explicit predecessor binding")
            if self.operation_id != self.predecessor_operation_id:
                raise ValueError("confirmed save operation does not match predecessor binding")
            if self.source not in predecessor_refs:
                raise ValueError("confirmed save source is not an accepted predecessor ref")
        elif self.conversation_id not in conversation_ids:
            raise ValueError("evidence conversation is outside captured context")
        if self.source_kind is SourceKind.ACTIVE_DOCUMENT and self.doc_id != selected_doc_id:
            raise ValueError("active_document evidence must match selected_doc_id")
        if self.source_kind in (SourceKind.EXPLICIT_REFERENCE, SourceKind.LOCAL_NUMBER):
            if self.command_span not in explicit_references:
                raise ValueError("evidence command span is not a captured explicit reference")
        if self.source_kind is SourceKind.PRESENTATION_ENTRY and self.source not in presentation_refs:
            raise ValueError("presentation evidence is not anchored to a captured presentation ref")
        if self.source_kind is SourceKind.VERIFIED_EXCHANGE and not (
            self.source.is_fragment or any(ref.is_fragment for ref in self.fragment_refs)
        ):
            raise ValueError("verified_exchange requires checkpoint/call/ordinal identity")
        for ref in self.fragment_refs:
            if (ref.context_id != context_id or ref.actor_id != actor_id
                    or ref.conversation_id not in conversation_ids):
                raise ValueError("fragment ref crosses captured scope")

    def to_dict(self) -> dict[str, Any]:
        return {"doc_id": self.doc_id, "source_kind": self.source_kind.value,
                "source": self.source.to_dict(), "context_id": self.context_id,
                "actor_id": self.actor_id, "conversation_id": self.conversation_id,
                "operation_id": self.operation_id,
                "fragment_refs": [item.to_dict() for item in self.fragment_refs],
                "command_span": None if self.command_span is None else self.command_span.to_dict(),
                "predecessor_operation_id": self.predecessor_operation_id,
                "metadata": _thaw_json(self.metadata)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DocumentEvidence":
        _only_keys(value, {"doc_id", "source_kind", "source", "context_id", "actor_id", "conversation_id",
                           "operation_id", "fragment_refs", "command_span", "predecessor_operation_id", "metadata"}, "DocumentEvidence")
        return cls(doc_id=value["doc_id"], source_kind=value["source_kind"],
                   source=BoundRef.from_dict(value["source"]), context_id=value["context_id"],
                   actor_id=value["actor_id"], conversation_id=value["conversation_id"],
                   operation_id=value["operation_id"],
                   fragment_refs=tuple(BoundRef.from_dict(item) for item in _items(value.get("fragment_refs", []), "fragment_refs")),
                   command_span=None if value.get("command_span") is None else CommandSpan.from_dict(value["command_span"]),
                   predecessor_operation_id=value.get("predecessor_operation_id"),
                   metadata=value.get("metadata"))


@dataclass(frozen=True, slots=True)
class RoutingContextSnapshot:
    context_id: str
    actor_id: int
    operation_id: str
    selected_conversation_id: str
    selected_conversation_kind: str
    accepted_command: str
    accepted_command_ref: BoundRef
    full_history_ref: BoundRef
    selected_doc_id: str | None = None
    explicit_references: tuple[CommandSpan, ...] = ()
    local_number: int | None = None
    parsed_id: str | None = None
    parsed_url: str | None = None
    document_evidence: Mapping[str, tuple[DocumentEvidence, ...]] | None = None
    presentation_refs: tuple[BoundRef, ...] = ()
    confirmed_predecessor_saves: tuple[BoundRef, ...] = ()
    clarification_id: str | None = None
    clarification_generation: int | None = None
    protocol: str = PROTOCOL
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.protocol != PROTOCOL:
            raise ValueError(f"protocol must be {PROTOCOL!r}")
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported schema_version")
        for name in ("context_id", "operation_id", "selected_conversation_id", "selected_conversation_kind"):
            _identifier(getattr(self, name), name)
        try:
            ConversationKind(self.selected_conversation_kind)
        except ValueError as exc:
            raise ValueError("unsupported selected conversation kind") from exc
        _positive_int(self.actor_id, "actor_id")
        if not isinstance(self.accepted_command, str):
            raise TypeError("accepted_command must be text")
        if self.selected_doc_id is not None:
            _identifier(self.selected_doc_id, "selected_doc_id")
        for name in ("accepted_command_ref", "full_history_ref"):
            ref = getattr(self, name)
            if not isinstance(ref, BoundRef):
                raise TypeError(f"{name} must be a BoundRef")
            if ref.context_id != self.context_id or ref.actor_id != self.actor_id:
                raise ValueError(f"{name} crosses context or actor scope")
        if self.accepted_command_ref.conversation_id != self.selected_conversation_id:
            raise ValueError("accepted command must belong to selected conversation")
        if self.accepted_command_ref.operation_id != self.operation_id:
            raise ValueError("accepted command must belong to current operation")
        if self.full_history_ref.conversation_id != self.selected_conversation_id:
            raise ValueError("full history ref must identify selected conversation")
        spans = _items(self.explicit_references, "explicit_references")
        if any(not isinstance(span, CommandSpan) for span in spans):
            raise TypeError("explicit_references must contain CommandSpan values")
        object.__setattr__(self, "explicit_references", spans)
        if self.local_number is not None:
            _positive_int(self.local_number, "local_number")
        for name in ("parsed_id", "parsed_url"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be non-empty text or None")
        for name in ("presentation_refs", "confirmed_predecessor_saves"):
            values = _items(getattr(self, name), name)
            if any(not isinstance(item, BoundRef) for item in values):
                raise TypeError(f"{name} must contain BoundRef values")
            for item in values:
                if item.context_id != self.context_id or item.actor_id != self.actor_id:
                    raise ValueError(f"{name} crosses context or actor scope")
            object.__setattr__(self, name, values)
        if any(item.conversation_id != self.selected_conversation_id for item in self.presentation_refs):
            raise ValueError("presentation refs must belong to the selected conversation")
        if self.clarification_id is None:
            if self.clarification_generation is not None:
                raise ValueError("clarification generation requires clarification ID")
        else:
            _identifier(self.clarification_id, "clarification_id")
            _positive_int(self.clarification_generation, "clarification_generation")
        if self.document_evidence is not None and not isinstance(self.document_evidence, Mapping):
            raise TypeError("document_evidence must be an object or None")
        evidence: dict[str, tuple[DocumentEvidence, ...]] = {}
        predecessor_refs = self.confirmed_predecessor_saves
        for key, raw_items in (self.document_evidence or {}).items():
            if isinstance(raw_items, DocumentEvidence):
                items = (raw_items,)
            else:
                items = _items(raw_items, "document_evidence entries")
            if not items:
                raise ValueError("document_evidence entries must not be empty")
            if not isinstance(key, str) or not key:
                raise ValueError("document_evidence keys must be document IDs")
            for item in items:
                if not isinstance(item, DocumentEvidence):
                    raise TypeError("document_evidence entries must be DocumentEvidence")
                if key != item.doc_id:
                    raise ValueError("document_evidence key must equal evidence doc_id")
                item.validate_scope(
                    context_id=self.context_id, actor_id=self.actor_id,
                    conversation_ids={self.selected_conversation_id},
                    predecessor_operation_ids={ref.operation_id for ref in predecessor_refs},
                    predecessor_refs=predecessor_refs, selected_doc_id=self.selected_doc_id,
                    explicit_references=spans, presentation_refs=self.presentation_refs,
                )
            evidence[key] = items
        object.__setattr__(self, "document_evidence", MappingProxyType(evidence))
        self.validate_command_spans()

    def validate_command_spans(self, accepted_command: str | None = None) -> None:
        """Validate spans against the actual captured command (Unicode code points)."""
        command = self.accepted_command if accepted_command is None else accepted_command
        if not isinstance(command, str):
            raise TypeError("accepted_command must be text")
        for span in self.explicit_references:
            if span.end > len(command) or command[span.start:span.end] != span.raw_text:
                raise ValueError("explicit reference span does not match accepted command")
        for entries in self.document_evidence.values():
            for item in entries:
                span = item.command_span
                if span is not None and (
                    span.end > len(command) or command[span.start:span.end] != span.raw_text
                ):
                    raise ValueError("evidence command span does not match accepted command")

    def to_dict(self) -> dict[str, Any]:
        return {"protocol": self.protocol, "schema_version": self.schema_version,
                "context_id": self.context_id, "actor_id": self.actor_id,
                "operation_id": self.operation_id,
                "selected_conversation_id": self.selected_conversation_id,
                "selected_conversation_kind": self.selected_conversation_kind,
                "selected_doc_id": self.selected_doc_id, "accepted_command": self.accepted_command,
                "accepted_command_ref": self.accepted_command_ref.to_dict(),
                "explicit_references": [item.to_dict() for item in self.explicit_references],
                "local_number": self.local_number, "parsed_id": self.parsed_id,
                "parsed_url": self.parsed_url, "full_history_ref": self.full_history_ref.to_dict(),
                "document_evidence": {key: [item.to_dict() for item in items]
                                      for key, items in self.document_evidence.items()},
                "presentation_refs": [item.to_dict() for item in self.presentation_refs],
                "confirmed_predecessor_saves": [item.to_dict() for item in self.confirmed_predecessor_saves],
                "clarification_id": self.clarification_id,
                "clarification_generation": self.clarification_generation}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RoutingContextSnapshot":
        _only_keys(value, {"protocol", "schema_version", "context_id", "actor_id", "operation_id",
                           "selected_conversation_id", "selected_conversation_kind", "selected_doc_id",
                           "accepted_command", "accepted_command_ref", "explicit_references", "local_number",
                           "parsed_id", "parsed_url", "full_history_ref", "document_evidence", "presentation_refs",
                           "confirmed_predecessor_saves", "clarification_id", "clarification_generation"}, "snapshot")
        evidence = value.get("document_evidence", {})
        if not isinstance(evidence, Mapping):
            raise TypeError("document_evidence must be an object")
        return cls(context_id=value["context_id"], actor_id=value["actor_id"],
                   operation_id=value["operation_id"],
                   selected_conversation_id=value["selected_conversation_id"],
                   selected_conversation_kind=value["selected_conversation_kind"],
                   accepted_command=value["accepted_command"],
                   accepted_command_ref=BoundRef.from_dict(value["accepted_command_ref"]),
                   full_history_ref=BoundRef.from_dict(value["full_history_ref"]),
                   selected_doc_id=value.get("selected_doc_id"),
                   explicit_references=tuple(CommandSpan.from_dict(x) for x in _items(value.get("explicit_references", []), "explicit_references")),
                   local_number=value.get("local_number"), parsed_id=value.get("parsed_id"),
                   parsed_url=value.get("parsed_url"),
                   document_evidence={key: tuple(DocumentEvidence.from_dict(item) for item in _items(items, "document_evidence entries"))
                                      for key, items in evidence.items()},
                   presentation_refs=tuple(BoundRef.from_dict(x) for x in _items(value.get("presentation_refs", []), "presentation_refs")),
                   confirmed_predecessor_saves=tuple(BoundRef.from_dict(x) for x in _items(value.get("confirmed_predecessor_saves", []), "confirmed_predecessor_saves")),
                   clarification_id=value.get("clarification_id"),
                   clarification_generation=value.get("clarification_generation"),
                   protocol=value.get("protocol"), schema_version=value.get("schema_version"))


@dataclass(frozen=True, slots=True)
class DocumentSelector:
    kind: DocumentSelectorKind
    raw_value: str | None = None
    span: CommandSpan | None = None
    dialog_number: int | None = None
    doc_id: str | None = None
    evidence_refs: tuple[str, ...] = ()
    list_id: str | None = None
    position: int | str | None = None

    def __post_init__(self) -> None:
        kind = _enum(DocumentSelectorKind, self.kind, "document selector kind")
        object.__setattr__(self, "kind", kind)
        evidence_refs = _items(self.evidence_refs, "evidence_refs")
        if any(not isinstance(item, str) for item in evidence_refs):
            raise TypeError("evidence_refs must contain reference strings")
        object.__setattr__(self, "evidence_refs", tuple(_opaque_ref(item, "evidence ref") for item in evidence_refs))
        if kind is DocumentSelectorKind.EXPLICIT_REFERENCE:
            if not isinstance(self.raw_value, str) or not self.raw_value.strip():
                raise ValueError("explicit_reference raw_value must be non-empty text")
            if not isinstance(self.span, CommandSpan):
                raise ValueError("explicit_reference requires raw_value and exact command span")
            if self.span.kind not in (ExplicitReferenceKind.ID, ExplicitReferenceKind.URL):
                raise ValueError("explicit_reference requires an ID or URL command span")
            if self.span.parsed_value != self.raw_value:
                raise ValueError("explicit_reference must match its exact command span")
            if any(x is not None for x in (self.dialog_number, self.doc_id, self.list_id, self.position)) or self.evidence_refs:
                raise ValueError("explicit_reference contains unrelated selector fields")
        elif kind is DocumentSelectorKind.DIALOG_NUMBER:
            _positive_int(self.dialog_number, "dialog_number")
            if not isinstance(self.span, CommandSpan) or self.span.kind is not ExplicitReferenceKind.LOCAL_NUMBER:
                raise ValueError("dialog_number requires a local-number command span")
            if self.span.parsed_value != self.dialog_number:
                raise ValueError("dialog_number must match its exact command span")
            if any(x is not None for x in (self.raw_value, self.doc_id, self.list_id, self.position)) or self.evidence_refs:
                raise ValueError("dialog_number contains unrelated selector fields")
        elif kind is DocumentSelectorKind.ACTIVE_DOCUMENT:
            if any(x is not None for x in (self.raw_value, self.span, self.dialog_number, self.doc_id, self.list_id, self.position)) or self.evidence_refs:
                raise ValueError("active_document takes no caller-supplied target")
        elif kind is DocumentSelectorKind.EVIDENCE:
            _identifier(self.doc_id, "doc_id")
            if not self.evidence_refs:
                raise ValueError("evidence selector requires evidence refs")
            if any(x is not None for x in (self.raw_value, self.span, self.dialog_number, self.list_id, self.position)):
                raise ValueError("evidence selector contains unrelated selector fields")
        else:
            _identifier(self.list_id, "list_id")
            if self.position != "last":
                _positive_int(self.position, "position")
            if any(x is not None for x in (self.raw_value, self.span, self.dialog_number, self.doc_id)) or self.evidence_refs:
                raise ValueError("shown_position contains unrelated selector fields")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "raw_value": self.raw_value,
                "span": None if self.span is None else self.span.to_dict(),
                "dialog_number": self.dialog_number, "doc_id": self.doc_id,
                "evidence_refs": list(self.evidence_refs), "list_id": self.list_id,
                "position": self.position}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DocumentSelector":
        _only_keys(value, {"kind", "raw_value", "span", "dialog_number", "doc_id", "evidence_refs", "list_id", "position"}, "DocumentSelector")
        return cls(kind=value["kind"], raw_value=value.get("raw_value"),
                   span=None if value.get("span") is None else CommandSpan.from_dict(value["span"]),
                   dialog_number=value.get("dialog_number"), doc_id=value.get("doc_id"),
                   evidence_refs=_items(value.get("evidence_refs", []), "evidence_refs"),
                   list_id=value.get("list_id"), position=value.get("position"))


@dataclass(frozen=True, slots=True)
class FolderSelector:
    kind: FolderSelectorKind
    value: str | None = None
    parent: "FolderSelector | None" = None

    def __post_init__(self) -> None:
        kind = _enum(FolderSelectorKind, self.kind, "folder selector kind")
        object.__setattr__(self, "kind", kind)
        if self.parent is not None and not isinstance(self.parent, FolderSelector):
            raise TypeError("folder parent must be a FolderSelector or None")
        if kind is FolderSelectorKind.ROOT_DEFAULT:
            if self.value is not None or self.parent is not None:
                raise ValueError("root_default takes no value or parent")
        elif kind is FolderSelectorKind.CREATE_STEP_OUTPUT:
            if self.value not in (None, "step0") or self.parent is not None:
                raise ValueError("create_step_output only references step0")
        elif not isinstance(self.value, str) or not self.value:
            raise ValueError(f"{kind.value} requires a non-empty value")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "value": self.value,
                "parent": None if self.parent is None else self.parent.to_dict()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FolderSelector":
        _only_keys(value, {"kind", "value", "parent"}, "FolderSelector")
        return cls(kind=value["kind"], value=value.get("value"),
                   parent=None if value.get("parent") is None else cls.from_dict(value["parent"]))


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    disposition: Disposition
    action: Action
    substantive_text: str
    document_selector: DocumentSelector | None = None
    folder_selector: FolderSelector | None = None
    folder_name: str | None = None
    explicit_references: tuple[CommandSpan, ...] = ()
    missing_slots: tuple[MissingSlot, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    clarification_id: str | None = None
    clarification_generation: int | None = None
    certainty: Certainty = Certainty.UNCERTAIN

    def __post_init__(self) -> None:
        object.__setattr__(self, "disposition", _enum(Disposition, self.disposition, "disposition"))
        object.__setattr__(self, "action", _enum(Action, self.action, "action"))
        object.__setattr__(self, "certainty", _enum(Certainty, self.certainty, "certainty"))
        if not isinstance(self.substantive_text, str):
            raise TypeError("substantive_text must be text")
        if self.document_selector is not None and not isinstance(self.document_selector, DocumentSelector):
            raise TypeError("document_selector must be typed or None")
        if self.folder_selector is not None and not isinstance(self.folder_selector, FolderSelector):
            raise TypeError("folder_selector must be typed or None")
        if self.folder_name is not None and (not isinstance(self.folder_name, str) or not self.folder_name.strip()):
            raise ValueError("folder_name must be non-empty text or None")
        object.__setattr__(self, "explicit_references", _items(self.explicit_references, "explicit_references"))
        if any(not isinstance(item, CommandSpan) for item in self.explicit_references):
            raise TypeError("explicit_references must contain CommandSpan values")
        object.__setattr__(self, "missing_slots", tuple(_enum(MissingSlot, x, "missing slot")
                                                          for x in _items(self.missing_slots, "missing_slots")))
        evidence_refs = _items(self.evidence_refs, "evidence_refs")
        if any(not isinstance(item, str) for item in evidence_refs):
            raise TypeError("evidence_refs must contain reference strings")
        object.__setattr__(self, "evidence_refs", tuple(_opaque_ref(x, "evidence ref") for x in evidence_refs))
        if self.clarification_id is None:
            if self.clarification_generation is not None:
                raise ValueError("clarification generation requires clarification ID")
        else:
            _identifier(self.clarification_id, "clarification_id")
            _positive_int(self.clarification_generation, "clarification_generation")

    def validate_command_spans(self, accepted_command: str) -> None:
        if not isinstance(accepted_command, str):
            raise TypeError("accepted_command must be text")
        selector_span = self.document_selector.span if self.document_selector is not None else None
        spans = list(self.explicit_references)
        if selector_span is not None:
            if not isinstance(selector_span, CommandSpan):
                raise TypeError("document selector span must be a CommandSpan")
            if spans and selector_span not in spans:
                raise ValueError("document selector span is inconsistent with explicit references")
            spans.append(selector_span)
        for span in spans:
            if (not isinstance(span, CommandSpan) or span.end > len(accepted_command)
                    or accepted_command[span.start:span.end] != span.raw_text):
                raise ValueError("decision span does not match accepted command")

    def to_dict(self) -> dict[str, Any]:
        return {"disposition": self.disposition.value,
                "action": self.action.value, "substantive_text": self.substantive_text,
                "document_selector": None if self.document_selector is None else self.document_selector.to_dict(),
                "folder_selector": None if self.folder_selector is None else self.folder_selector.to_dict(),
                "folder_name": self.folder_name,
                "explicit_references": [x.to_dict() for x in self.explicit_references],
                "missing_slots": [x.value for x in self.missing_slots],
                "evidence_refs": list(self.evidence_refs), "clarification_id": self.clarification_id,
                "clarification_generation": self.clarification_generation,
                "certainty": self.certainty.value}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RoutingDecision":
        _only_keys(value, {"disposition", "action", "substantive_text", "document_selector",
                           "folder_selector", "folder_name", "explicit_references", "missing_slots", "evidence_refs",
                           "clarification_id", "clarification_generation", "certainty"}, "RoutingDecision")
        return cls(disposition=value["disposition"], action=value["action"],
                   substantive_text=value["substantive_text"],
                   document_selector=None if value.get("document_selector") is None else DocumentSelector.from_dict(value["document_selector"]),
                   folder_selector=None if value.get("folder_selector") is None else FolderSelector.from_dict(value["folder_selector"]),
                   folder_name=value.get("folder_name"),
                   explicit_references=tuple(CommandSpan.from_dict(x) for x in _items(value.get("explicit_references", []), "explicit_references")),
                   missing_slots=_items(value.get("missing_slots", []), "missing_slots"),
                   evidence_refs=_items(value.get("evidence_refs", []), "evidence_refs"),
                   clarification_id=value.get("clarification_id"),
                   clarification_generation=value.get("clarification_generation"),
                   certainty=value.get("certainty", Certainty.UNCERTAIN.value))
