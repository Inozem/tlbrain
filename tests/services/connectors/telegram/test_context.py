from copy import deepcopy

import pytest

from services.connectors.telegram.context import (
    build_document_context,
    build_save_analysis_context,
)


@pytest.mark.parametrize(("count", "capacity"), [(0, 30), (10, 30), (11, 31), (40, 60)])
def test_save_context_keeps_every_prepared_row_and_uses_deterministic_capacity(
    count, capacity,
):
    rows = [
        {
            "order_index": index,
            "speaker": "Speaker",
            "text": f"draft row {index}",
            "word_count": 3,
            "source_message_ids": [index + 1],
        }
        for index in range(count)
    ]
    original = deepcopy(rows)

    context = build_save_analysis_context(
        rows, context_id="ctx-current", draft_id="draft-current",
        dialog_date="2026-10-28",
    )

    assert context == {
        "provider": "telegram",
        "context_id": "ctx-current",
        "draft_id": "draft-current",
        "dialog_date": "2026-10-28",
        "messages": rows,
        "word_capacity": capacity,
    }
    assert len(context["messages"]) == count
    assert rows == original


def test_document_context_uses_selected_history_and_keeps_permitted_tool_shape():
    history = [
        {
            "conversation_id": "document-current",
            "operation_id": "prior-op",
            "role": "user",
            "parts": [{"text": "prior document question"}],
        },
        {
            "conversation_id": "document-current",
            "operation_id": "prior-tool-op",
            "role": "tool",
            "parts": [{"function_response": {"name": "lookup", "response": {"ok": True}}}],
        },
        {
            "conversation_id": "document-other",
            "operation_id": "other-op",
            "role": "user",
            "parts": [{"text": "other conversation secret"}],
        },
        {
            "conversation_id": "document-current",
            "operation_id": "current-op",
            "role": "user",
            "parts": [{"text": "replayed current request"}],
        },
    ]
    original = deepcopy(history)

    prompt = build_document_context(
        history, "current substantive question",
        conversation_id="document-current", operation_id="current-op",
    )

    assert prompt == {
        "history": [
            {"role": "user", "parts": [{"text": "prior document question"}]},
            {
                "role": "tool",
                "parts": [{"function_response": {"name": "lookup", "response": {"ok": True}}}],
            },
        ],
        "request": "current substantive question",
    }
    assert history == original
