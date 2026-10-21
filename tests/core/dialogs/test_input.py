"""Acceptance and normalization contract for dialog input."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest


def _normalize(arguments: dict, accepted_at: datetime):
    # Runtime import keeps collection working before T032 supplies this seam.
    from core.dialogs.models import normalize_dialog_input

    return normalize_dialog_input(arguments, accepted_at=accepted_at)


ACCEPTED_AT = datetime(2026, 3, 1, 23, 30, tzinfo=timezone.utc)
TEXT = (
    "  intro  \r\n\t\n```py\nprint('x')\n```\n| a | b |\n|---|---|\n|1|2|\n"
    + r"\n"
    + " :: literal  \U0001f600  "
)


def test_normalization_preserves_provider_utterance_content_and_optional_path():
    normalized = _normalize(
        {
            "provider": "custom-origin",
            "utterances": [
                {"speaker": "  Ada :: Lovelace ", "text": TEXT},
                {"speaker": "Grace", "text": "second utterance"},
            ],
        },
        ACCEPTED_AT,
    )

    assert normalized.provider == "custom-origin"
    assert normalized.title.strip()
    assert normalized.dialog_date == "2026-03-01"
    assert normalized.folder_path is None
    assert len(normalized.utterances) == 2
    assert normalized.utterances[0].speaker == "Ada  Lovelace"
    assert normalized.utterances[0].text == TEXT
    assert (normalized.utterances[1].speaker, normalized.utterances[1].text) == (
        "Grace",
        "second utterance",
    )


def test_normalization_keeps_explicit_date_title_and_empty_folder_path():
    normalized = _normalize(
        {
            "provider": "chatgpt",
            "title": "  Saved title  ",
            "dialog_date": "2024-02-29",
            "folder_path": [],
            "utterances": [{"speaker": "\U0001f9d1\u200d\U0001f4bb", "text": "hello"}],
        },
        ACCEPTED_AT,
    )

    assert normalized.provider == "chatgpt"
    assert normalized.title == "Saved title"
    assert normalized.dialog_date == "2024-02-29"
    assert normalized.folder_path == []
    assert normalized.utterances[0].speaker == "\U0001f9d1\u200d\U0001f4bb"


@pytest.mark.parametrize(
    "provider",
    ["claude", "telegram", "tldv", "manual", "custom", "gemini-2.5-pro"],
)
def test_provider_is_an_open_origin_label_and_is_not_replaced(provider: str):
    normalized = _normalize(
        {"provider": provider, "utterances": [{"speaker": "A", "text": "x"}]},
        ACCEPTED_AT,
    )
    assert normalized.provider == provider


@pytest.mark.parametrize(
    "arguments",
    [
        {"provider": "  ", "utterances": [{"speaker": "A", "text": "x"}]},
        {"utterances": [{"speaker": "A", "text": "x"}]},
        {"provider": "manual"},
        {"provider": "custom\nPROVIDER: forged", "utterances": [{"speaker": "A", "text": "x"}]},
        {"provider": "custom", "utterances": []},
        {"provider": "custom", "utterances": "A :: x"},
        {"provider": "custom", "utterances": [{"speaker": "   ", "text": "x"}]},
        {"provider": "custom", "utterances": [{"speaker": "A\r\nPROVIDER: forged", "text": "x"}]},
        {"provider": "custom", "utterances": [{"speaker": "::", "text": "x"}]},
        {"provider": "custom", "utterances": [{"speaker": "A", "text": " \t "}]},
        {"provider": "custom", "utterances": [{"speaker": 3, "text": "x"}]},
        {"provider": "custom", "utterances": [{"speaker": "A", "text": None}]},
        {"provider": "custom", "utterances": [{"speaker": "A", "text": "x", "other": 1}]},
        {"provider": "custom", "utterances": [{"speaker": "A", "text": "x"}], "extra": True},
        {"provider": "custom", "dialog_date": "2026-02-30", "utterances": [{"speaker": "A", "text": "x"}]},
        {"provider": "custom", "dialog_date": "not-a-date", "utterances": [{"speaker": "A", "text": "x"}]},
        {"provider": "custom", "title": " \t ", "utterances": [{"speaker": "A", "text": "x"}]},
        {"provider": "custom", "folder_path": "folder/name", "utterances": [{"speaker": "A", "text": "x"}]},
        {"provider": "custom", "folder_path": ["one", " "], "utterances": [{"speaker": "A", "text": "x"}]},
        {"provider": "custom", "folder_path": ["one", 2], "utterances": [{"speaker": "A", "text": "x"}]},
    ],
)
def test_malformed_or_injected_input_is_rejected(arguments: dict, sdk_fakes, firestore_fake):
    with pytest.raises((TypeError, ValueError)):
        _normalize(arguments, ACCEPTED_AT)
    assert sdk_fakes.trace == []
    assert firestore_fake.trace() == []
    assert firestore_fake.documents == {}


def test_naive_acceptance_time_is_rejected():
    with pytest.raises((TypeError, ValueError)):
        _normalize(
            {"provider": "manual", "utterances": [{"speaker": "A", "text": "x"}]},
            datetime(2026, 3, 1, 12),
        )


def test_default_date_uses_installation_timezone_and_acceptance_time(monkeypatch):
    arguments = {"provider": "manual", "utterances": [{"speaker": "A", "text": "x"}]}
    monkeypatch.setenv("TZ", "Asia/Jerusalem")
    first = _normalize(arguments, ACCEPTED_AT)
    later = _normalize(arguments, ACCEPTED_AT)

    assert first.dialog_date == "2026-03-02"
    assert later.dialog_date == first.dialog_date
    assert first.dialog_date in first.title
