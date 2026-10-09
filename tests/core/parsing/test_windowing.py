"""Characterization tests for default and clipped transcript windows."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from core.parsing.parser import parse_document
from core.parsing.windowing import generate_windows


def _utterances(records: list[tuple[str, str]]) -> list[dict[str, Any]]:
    return [
        {"speaker": speaker, "text": text, "order_index": index}
        for index, (speaker, text) in enumerate(records)
    ]


def _window(
    center: int, first: int, last: int, utterances: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "center_index": center,
        "covered_range": [first, last],
        "utterances": utterances,
    }


_FULL_RECORDS = [
    ("Fixture Speaker A", "We will review the sample schedule."),
    ("Fixture Speaker B", "The draft has three sections."),
    ("Fixture Speaker A", "Each section needs a clear owner."),
    ("Fixture Speaker B", "We can track changes in a small table."),
    ("Fixture Speaker A", "The first review is on Tuesday."),
    ("Fixture Speaker B", "Notes should include the decision."),
    ("Fixture Speaker A", "A second pass can check the wording."),
    ("Fixture Speaker B", "We will keep the example data invented."),
    ("Fixture Speaker A", "The final copy can stay brief."),
    ("Fixture Speaker B", "We will record the next step."),
]


def test_default_windows_match_full_fixture_centers_ranges_and_contents(app_root: Path) -> None:
    fixture = json.loads((app_root / "tests/fixtures/transcripts.json").read_text(encoding="utf-8"))
    _, utterances = parse_document(fixture["cases"]["valid_full"]["raw"])
    assert utterances == _utterances(_FULL_RECORDS)
    before = deepcopy(utterances)
    expected = [
        _window(0, 0, 2, [
            {"speaker": "Fixture Speaker A", "text": "We will review the sample schedule.", "order_index": 0},
            {"speaker": "Fixture Speaker B", "text": "The draft has three sections.", "order_index": 1},
            {"speaker": "Fixture Speaker A", "text": "Each section needs a clear owner.", "order_index": 2},
        ]),
        _window(3, 1, 5, [
            {"speaker": "Fixture Speaker B", "text": "The draft has three sections.", "order_index": 1},
            {"speaker": "Fixture Speaker A", "text": "Each section needs a clear owner.", "order_index": 2},
            {"speaker": "Fixture Speaker B", "text": "We can track changes in a small table.", "order_index": 3},
            {"speaker": "Fixture Speaker A", "text": "The first review is on Tuesday.", "order_index": 4},
            {"speaker": "Fixture Speaker B", "text": "Notes should include the decision.", "order_index": 5},
        ]),
        _window(6, 4, 8, [
            {"speaker": "Fixture Speaker A", "text": "The first review is on Tuesday.", "order_index": 4},
            {"speaker": "Fixture Speaker B", "text": "Notes should include the decision.", "order_index": 5},
            {"speaker": "Fixture Speaker A", "text": "A second pass can check the wording.", "order_index": 6},
            {"speaker": "Fixture Speaker B", "text": "We will keep the example data invented.", "order_index": 7},
            {"speaker": "Fixture Speaker A", "text": "The final copy can stay brief.", "order_index": 8},
        ]),
        _window(9, 7, 9, [
            {"speaker": "Fixture Speaker B", "text": "We will keep the example data invented.", "order_index": 7},
            {"speaker": "Fixture Speaker A", "text": "The final copy can stay brief.", "order_index": 8},
            {"speaker": "Fixture Speaker B", "text": "We will record the next step.", "order_index": 9},
        ]),
    ]

    assert fixture["cases"]["valid_full"]["expected_anchor_indices"] == [0, 3, 6, 9]
    assert fixture["cases"]["valid_full"]["expected_window_ranges"] == [[0, 2], [1, 5], [4, 8], [7, 9]]
    assert generate_windows(utterances) == expected
    assert utterances == before


def test_empty_and_short_inputs_return_exact_clipped_windows() -> None:
    assert generate_windows([]) == []

    one = _utterances([("A", "only")])
    assert generate_windows(one) == [
        _window(0, 0, 0, [{"speaker": "A", "text": "only", "order_index": 0}])
    ]

    short = _utterances([("A", "zero"), ("B", "one"), ("A", "two"), ("B", "three")])
    assert generate_windows(short) == [
        _window(0, 0, 2, [
            {"speaker": "A", "text": "zero", "order_index": 0},
            {"speaker": "B", "text": "one", "order_index": 1},
            {"speaker": "A", "text": "two", "order_index": 2},
        ]),
        _window(3, 1, 3, [
            {"speaker": "B", "text": "one", "order_index": 1},
            {"speaker": "A", "text": "two", "order_index": 2},
            {"speaker": "B", "text": "three", "order_index": 3},
        ]),
    ]


def test_custom_supported_anchor_step_and_half_window_are_passed_through() -> None:
    utterances = _utterances([("A", "zero"), ("B", "one"), ("A", "two"), ("B", "three"), ("A", "four")])
    assert generate_windows(utterances, anchor_step=2, half_window=1) == [
        _window(0, 0, 1, [
            {"speaker": "A", "text": "zero", "order_index": 0},
            {"speaker": "B", "text": "one", "order_index": 1},
        ]),
        _window(2, 1, 3, [
            {"speaker": "B", "text": "one", "order_index": 1},
            {"speaker": "A", "text": "two", "order_index": 2},
            {"speaker": "B", "text": "three", "order_index": 3},
        ]),
        _window(4, 3, 4, [
            {"speaker": "B", "text": "three", "order_index": 3},
            {"speaker": "A", "text": "four", "order_index": 4},
        ]),
    ]
