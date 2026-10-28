"""Build isolated model inputs for Telegram save and document workflows."""

from __future__ import annotations

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


__all__ = ["build_document_context", "build_save_analysis_context"]
