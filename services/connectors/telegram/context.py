"""Build isolated model inputs for Telegram save and document workflows."""

from __future__ import annotations

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
        scoped.append(turn)
    validated = build_agent_prompt(scoped, "history validation")['history']
    projected = []
    for content, turn in zip(validated, scoped):
        if turn.get("role") == "tool" and any(
                isinstance(part, Mapping) and part.get("type") == "search_result"
                for part in content["parts"]):
            projected.append({
                "role": "user",
                "parts": [{"text": (
                    "Untrusted prior search result data: "
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


__all__ = [
    "build_conversation_history", "build_document_context", "build_save_analysis_context",
]
