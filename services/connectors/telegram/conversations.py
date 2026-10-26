"""Validated records for durable Telegram conversation history."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Mapping


class ConversationKind(StrEnum):
    DRAFT = "draft"
    DOCUMENT = "document"
    SEARCH = "search"


class TurnRole(StrEnum):
    USER = "user"
    MODEL = "model"
    TOOL = "tool"


def _json_value(value: Any, name: str) -> Any:
    """Validate and copy JSON-safe data without changing its contents."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, list):
        return [_json_value(item, name) for item in value]
    if isinstance(value, tuple):
        return [_json_value(item, name) for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{name} mapping keys must be strings")
        return {key: _json_value(item, name) for key, item in value.items()}
    raise TypeError(f"{name} must contain only JSON-safe values")


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or "/" in value:
        raise ValueError(f"{name} must be a non-empty identifier without '/' ")
    return value


def _timestamp(value: Any, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


@dataclass(frozen=True, slots=True)
class ConversationRecord:
    conversation_id: str
    kind: ConversationKind
    draft_id: str | None
    doc_id: str | None
    dialog_number: int | None
    search_filters: Any
    created_at: datetime
    last_substantive_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "kind": self.kind.value,
            "draft_id": self.draft_id,
            "doc_id": self.doc_id,
            "dialog_number": self.dialog_number,
            "search_filters": _json_value(self.search_filters, "search_filters"),
            "created_at": self.created_at,
            "last_substantive_at": self.last_substantive_at,
            "last_turn_sequence": 0,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ConversationRecord":
        return cls(
            conversation_id=value["conversation_id"],
            kind=ConversationKind(value["kind"]),
            draft_id=value.get("draft_id"),
            doc_id=value.get("doc_id"),
            dialog_number=value.get("dialog_number"),
            search_filters=_json_value(value.get("search_filters"), "search_filters"),
            created_at=value["created_at"],
            last_substantive_at=value.get("last_substantive_at"),
        )


@dataclass(frozen=True, slots=True)
class TurnRecord:
    turn_id: str
    conversation_id: str
    operation_id: str
    role: TurnRole
    parts: list[Any]
    turn_sequence: int
    created_at: datetime
    tool_correlation: Any = None
    source_refs: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "conversation_id": self.conversation_id,
            "operation_id": self.operation_id,
            "role": self.role.value,
            "parts": _json_value(self.parts, "parts"),
            "tool_correlation": _json_value(self.tool_correlation, "tool_correlation"),
            "source_refs": _json_value(self.source_refs, "source_refs"),
            "turn_sequence": self.turn_sequence,
            "created_at": self.created_at,
        }


def build_conversation(
    conversation_id: str,
    kind: ConversationKind | str,
    *,
    draft_id: str | None = None,
    doc_id: str | None = None,
    dialog_number: int | None = None,
    search_filters: Any = None,
    now: datetime,
) -> ConversationRecord:
    conversation_id = _identifier(conversation_id, "conversation_id")
    try:
        kind = ConversationKind(kind)
    except (TypeError, ValueError) as exc:
        raise ValueError("kind must be 'draft', 'document', or 'search'") from exc
    now = _timestamp(now, "now")
    if kind is ConversationKind.DRAFT:
        draft_id = _identifier(draft_id, "draft_id")
        if doc_id is not None or type(dialog_number) is not int or dialog_number < 1:
            raise ValueError("draft conversations require a dialog number and no document")
        search_filters = None
    elif kind is ConversationKind.DOCUMENT:
        if draft_id is not None or type(dialog_number) is not int or dialog_number < 1:
            raise ValueError("document conversations require a dialog number and no draft")
        doc_id = _identifier(doc_id, "doc_id")
        search_filters = None
    else:
        if draft_id is not None or doc_id is not None or dialog_number is not None:
            raise ValueError("search conversations cannot target a draft, document, or dialog")
        search_filters = _json_value(search_filters, "search_filters")
    return ConversationRecord(
        conversation_id, kind, draft_id, doc_id, dialog_number,
        search_filters, now, None,
    )


def build_turn_payload(
    conversation_id: str,
    *,
    turn_id: str,
    operation_id: str,
    role: TurnRole | str,
    parts: Any,
    tool_correlation: Any = None,
    source_refs: Any = None,
    now: datetime,
) -> dict[str, Any]:
    conversation_id = _identifier(conversation_id, "conversation_id")
    turn_id = _identifier(turn_id, "turn_id")
    operation_id = _identifier(operation_id, "operation_id")
    try:
        role = TurnRole(role)
    except (TypeError, ValueError) as exc:
        raise ValueError("role must be 'user', 'model', or 'tool'") from exc
    if not isinstance(parts, (list, tuple)):
        raise TypeError("parts must be a list")
    return {
        "turn_id": turn_id,
        "conversation_id": conversation_id,
        "operation_id": operation_id,
        "role": role.value,
        "parts": _json_value(parts, "parts"),
        "tool_correlation": _json_value(tool_correlation, "tool_correlation"),
        "source_refs": _json_value(source_refs, "source_refs"),
        "created_at": _timestamp(now, "now"),
    }
