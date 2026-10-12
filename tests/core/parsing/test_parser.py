"""Characterization tests for the canonical transcript parser."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from core.parsing.parser import is_valid_format, parse_document


def _utterance(speaker: str, text: str, order_index: int) -> dict[str, Any]:
    return {"speaker": speaker, "text": text, "order_index": order_index}


_CASES: dict[str, tuple[dict[str, str], list[dict[str, Any]], bool]] = {
    "valid_full": (
        {"date": "2026-01-12", "time": "09:00", "provider": "tldv"},
        [
            _utterance("Fixture Speaker A", "We will review the sample schedule.", 0),
            _utterance("Fixture Speaker B", "The draft has three sections.", 1),
            _utterance("Fixture Speaker A", "Each section needs a clear owner.", 2),
            _utterance("Fixture Speaker B", "We can track changes in a small table.", 3),
            _utterance("Fixture Speaker A", "The first review is on Tuesday.", 4),
            _utterance("Fixture Speaker B", "Notes should include the decision.", 5),
            _utterance("Fixture Speaker A", "A second pass can check the wording.", 6),
            _utterance("Fixture Speaker B", "We will keep the example data invented.", 7),
            _utterance("Fixture Speaker A", "The final copy can stay brief.", 8),
            _utterance("Fixture Speaker B", "We will record the next step.", 9),
        ],
        True,
    ),
    "valid_minimally_changed": (
        {"date": "2026-01-12", "time": "09:00", "provider": "tldv"},
        [
            _utterance("Fixture Speaker A", "We will review the sample schedule.", 0),
            _utterance("Fixture Speaker B", "The draft has three sections.", 1),
            _utterance("Fixture Speaker A", "Each section needs a clear owner.", 2),
            _utterance("Fixture Speaker B", "We can track changes in a small table.", 3),
            _utterance("Fixture Speaker A", "The first review is on Wednesday.", 4),
            _utterance("Fixture Speaker B", "Notes should include the decision.", 5),
            _utterance("Fixture Speaker A", "A second pass can check the wording.", 6),
            _utterance("Fixture Speaker B", "We will keep the example data invented.", 7),
            _utterance("Fixture Speaker A", "The final copy can stay brief.", 8),
            _utterance("Fixture Speaker B", "We will record the next step.", 9),
        ],
        True,
    ),
    "valid_without_time": (
        {"date": "2026-01-12", "provider": "tldv"},
        [_utterance("Fixture Speaker A", "This transcript has no time header.", 0)],
        True,
    ),
    "empty_raw_text": ({}, [], False),
    "headers_only": ({"date": "2026-01-12", "provider": "tldv"}, [], False),
    "missing_date": (
        {"provider": "tldv"},
        [_utterance("Fixture Speaker A", "This has no date field.", 0)],
        False,
    ),
    "missing_provider": (
        {"date": "2026-01-12"},
        [_utterance("Fixture Speaker A", "This has no provider field.", 0)],
        False,
    ),
    "missing_separator": (
        {},
        [_utterance("Fixture Speaker A", "This has no separator line.", 0)],
        False,
    ),
    "mixed_invalid_body_lines": (
        {"date": "2026-01-12", "provider": "tldv"},
        [
            _utterance("Fixture Speaker B", "This accepted line has order zero.", 0),
            _utterance("Fixture Speaker A", "This accepted line has order one.", 1),
        ],
        True,
    ),
}


@pytest.mark.parametrize("case_name", tuple(_CASES))
def test_supplied_transcript_cases_have_exact_parse_and_validation(
    app_root: Path, case_name: str
) -> None:
    cases = json.loads((app_root / "tests/fixtures/transcripts.json").read_text(encoding="utf-8"))["cases"]
    expected_metadata, expected_utterances, expected_valid = _CASES[case_name]
    raw = cases[case_name]["raw"]

    assert parse_document(raw) == (expected_metadata, expected_utterances)
    assert is_valid_format(raw) is expected_valid


@pytest.mark.parametrize(
    ("raw", "expected_metadata", "expected_utterances", "expected_valid"),
    [
        (
            " DATE :   not-ISO  \nTIME:  \nPROVIDER:   \nIGNORED: value\n---\n  Zoë 🧪  ::  café :: tea  \n",
            {"date": "not-ISO", "time": "", "provider": ""},
            [_utterance("Zoë 🧪", "café :: tea", 0)],
            True,
        ),
        (
            "DATE: \nPROVIDER: \n---\nSpeaker :: accepted\n",
            {"date": "", "provider": ""},
            [_utterance("Speaker", "accepted", 0)],
            True,
        ),
        (
            "date: 2026-01-12\nprovider: tldv\n---\nSpeaker :: text\n",
            {},
            [_utterance("Speaker", "text", 0)],
            False,
        ),
        (
            "DATE: 2026-01-12\nPROVIDER: tldv\n --- \nSpeaker :: text\n",
            {},
            [_utterance("Speaker", "text", 0)],
            False,
        ),
        (
            "DATE: 2026-01-12\nPROVIDER: tldv\n----\nSpeaker :: text\n",
            {},
            [_utterance("Speaker", "text", 0)],
            False,
        ),
        (
            "DATE: 2026-01-12\nPROVIDER: tldv\n---\n"
            "bad line\n :: no speaker\nSpeaker ::   \n"
            "Speaker A :: first :: retained delimiter\n\nSpeaker B :: second\n",
            {"date": "2026-01-12", "provider": "tldv"},
            [
                _utterance("Speaker A", "first :: retained delimiter", 0),
                _utterance("Speaker B", "second", 1),
            ],
            True,
        ),
        (
            "DATE: sample\nPROVIDER: synthetic\n---\n"
            "Before::ignored\nA :: first\nBetween ::ignored\n"
            "B :: second\nAfter:: ignored\n",
            {"date": "sample", "provider": "synthetic"},
            [
                _utterance("A", "first", 0),
                _utterance("B", "second", 1),
            ],
            True,
        ),
    ],
    ids=[
        "empty-and-non-iso-header-values",
        "empty-date-provider-values-are-present-keys",
        "case-sensitive-header-keys",
        "spaced-separator",
        "near-separator",
        "accepted-lines-and-first-delimiter",
        "literal-spaced-delimiter-and-interspersed-invalid-lines",
    ],
)
def test_supported_parser_edges_are_preserved(
    raw: str,
    expected_metadata: dict[str, str],
    expected_utterances: list[dict[str, Any]],
    expected_valid: bool,
) -> None:
    assert parse_document(raw) == (expected_metadata, expected_utterances)
    assert is_valid_format(raw) is expected_valid
