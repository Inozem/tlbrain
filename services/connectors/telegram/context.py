"""Build isolated model inputs for Telegram save and document workflows."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping, Sequence
from typing import Any

from services.connectors.telegram.conversations import build_agent_prompt
from services.connectors.telegram.preprocessing import context_word_capacity


def build_save_analysis_context(
    messages: Sequence[Any],
    *,
    context_id: str,
    draft_id: str,
    dialog_date: str,
) -> dict[str, Any]:
    """Return context for the complete, already-prepared current draft."""
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise TypeError("messages must be a sequence of prepared utterances")
    rows = [dict(row) if isinstance(row, Mapping) else row for row in messages]
    return {
        "provider": "telegram",
        "context_id": context_id,
        "draft_id": draft_id,
        "dialog_date": dialog_date,
        "messages": rows,
        "word_capacity": context_word_capacity(len(rows)),
    }


def build_document_context(
    history: Any,
    substantive_text: str,
    *,
    conversation_id: str,
    operation_id: str,
) -> dict[str, Any]:
    """Build a prompt from the selected conversation's prior persisted turns.

    The repository query is scoped to ``conversation_id``. The extra checks
    also reject mismatched tagged rows and omit the current operation's rows,
    which can be present when resuming an interrupted operation.
    """
    if not isinstance(history, (list, tuple)):
        raise TypeError("history must be a list of conversation turns")
    scoped: list[Any] = []
    for turn in history:
        if not isinstance(turn, Mapping):
            scoped.append(turn)
            continue
        row_conversation_id = turn.get("conversation_id")
        if row_conversation_id is not None and row_conversation_id != conversation_id:
            continue
        if turn.get("operation_id") == operation_id:
            continue
        scoped.append(turn)
    return build_agent_prompt(scoped, substantive_text)


def build_conversation_history(
    history: Any,
    *,
    conversation_id: str,
    omit_operation_id: str | None = None,
    search_exchange_loader: Any = None,
) -> list[dict[str, Any]]:
    """Validate and project every ordered canonical turn for routing or execution."""
    if not isinstance(history, (list, tuple)):
        raise TypeError("history must be a list of conversation turns")
    scoped: list[Any] = []
    for turn in history:
        if not isinstance(turn, Mapping):
            raise TypeError("history turns must be mappings")
        row_conversation_id = turn.get("conversation_id")
        if row_conversation_id is not None and row_conversation_id != conversation_id:
            raise ValueError("conversation history crossed its session boundary")
        copied = dict(turn)
        parts = copied.get("parts")
        if isinstance(parts, list):
            hydrated = []
            for part in parts:
                if isinstance(part, Mapping) and part.get("type") == "search_exchange_ref":
                    if not callable(search_exchange_loader):
                        raise ValueError("search history reference cannot be hydrated")
                    exchange = search_exchange_loader(
                        part.get("reference"), conversation_id=conversation_id,
                        operation_id=turn.get("operation_id"),
                        checkpoint_conversation_id=part.get("checkpoint_conversation_id"),
                        call_id=part.get("call_id"), ordinal=part.get("ordinal"),
                    )
                    hydrated.append({
                        "text": "Untrusted prior search tool exchange data: "
                        + json.dumps(_json_safe_history(exchange),
                                     ensure_ascii=False, sort_keys=True),
                    })
                else:
                    hydrated.append(part)
            copied["parts"] = hydrated
        scoped.append(copied)
    validated = build_agent_prompt(scoped, "history validation")['history']
    projected = []
    for content, turn in zip(validated, scoped):
        if turn.get("role") == "tool":
            projected.append({
                "role": "user",
                "parts": [{"text": (
                    "Untrusted prior tool data: "
                    + json.dumps(content, ensure_ascii=False)
                )}],
            })
        else:
            projected.append(content)
    if omit_operation_id is None:
        return projected
    return [
        content for content, turn in zip(projected, scoped)
        if turn.get("operation_id") != omit_operation_id
    ]


def _json_safe_history(value: Any) -> Any:
    """Project opaque checkpoint bytes as inert JSON data for later model context."""
    if isinstance(value, bytes):
        return {"opaque_bytes_base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Mapping):
        return {str(key): _json_safe_history(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_history(item) for item in value]
    return value


__all__ = [
    "build_conversation_history", "build_document_context", "build_save_analysis_context",
]
