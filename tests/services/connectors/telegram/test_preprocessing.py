"""Pure text preprocessing contracts for Telegram source messages.

These tests define the public boundary expected from
``services.connectors.telegram.preprocessing``. They use synthetic records and
do not exercise Telegram, storage, transcription, or network integrations.
"""

from __future__ import annotations

import importlib

import pytest


pytestmark = pytest.mark.offline


def _message(
    message_id: int,
    text: str,
    *,
    intake_sequence: int | None = None,
    reliable_author_key: str | None = "telegram-user:7",
    display_name: str | None = "Alex",
):
    return {
        "message_id": message_id,
        "intake_sequence": message_id if intake_sequence is None else intake_sequence,
        "text": text,
        "reliable_author_key": reliable_author_key,
        "display_name": display_name,
    }


def _words(count: int, prefix: str = "word") -> str:
    return " ".join(f"{prefix}{index}" for index in range(count))


def _field(record, name):
    return record[name] if isinstance(record, dict) else getattr(record, name)


def _preprocess(messages):
    module = importlib.import_module("services.connectors.telegram.preprocessing")
    return module.preprocess_messages(messages)


def _context_capacity(message_count: int) -> int:
    module = importlib.import_module("services.connectors.telegram.preprocessing")
    return module.context_word_capacity(message_count)


def test_restores_intake_order_and_keeps_source_message_ids():
    messages = [
        _message(303, "third", intake_sequence=3, reliable_author_key="user:3"),
        _message(101, "first", intake_sequence=1, reliable_author_key="user:1"),
        _message(202, "second", intake_sequence=2, reliable_author_key="user:2"),
    ]

    prepared = _preprocess(messages)

    assert [_field(item, "text") for item in prepared] == ["first", "second", "third"]
    assert [_field(item, "source_message_ids") for item in prepared] == [[101], [202], [303]]


@pytest.mark.parametrize(
    ("counts", "expected_count"),
    [((18, 24, 31), 1), ((12, 43), 1)],
)
def test_merges_consecutive_short_same_author_messages(counts, expected_count):
    messages = [_message(index + 1, _words(count)) for index, count in enumerate(counts)]

    prepared = _preprocess(messages)

    assert len(prepared) == expected_count
    assert [_field(item, "source_message_ids") for item in prepared] == [
        [message["message_id"] for message in messages]
    ]
    assert [len(_field(item, "text").split()) for item in prepared] == [sum(counts)]


def test_same_display_name_does_not_merge_without_a_reliable_author_key():
    messages = [
        _message(1, "from an unknown sender", reliable_author_key=None, display_name="Alex"),
        _message(2, "also an unknown sender", reliable_author_key=None, display_name="Alex"),
    ]

    prepared = _preprocess(messages)

    assert [_field(item, "source_message_ids") for item in prepared] == [[1], [2]]


def test_author_change_ends_a_block_even_when_display_names_match():
    messages = [
        _message(1, _words(8), reliable_author_key="user:1", display_name="Alex"),
        _message(2, _words(9), reliable_author_key="user:1", display_name="Alex"),
        _message(3, _words(7), reliable_author_key="user:2", display_name="Alex"),
        _message(4, _words(6), reliable_author_key="user:2", display_name="Alex"),
        _message(5, _words(5), reliable_author_key="user:1", display_name="Alex"),
    ]

    prepared = _preprocess(messages)

    assert [_field(item, "source_message_ids") for item in prepared] == [[1, 2], [3, 4], [5]]


def test_long_source_message_stays_standalone_and_short_remainder_is_kept():
    messages = [
        _message(1, _words(10, "before")),
        _message(2, _words(50, "long")),
        _message(3, _words(6, "remainder")),
    ]

    prepared = _preprocess(messages)

    assert [_field(item, "source_message_ids") for item in prepared] == [[1], [2], [3]]
    assert [len(_field(item, "text").split()) for item in prepared] == [10, 50, 6]


def test_49_word_source_can_merge_to_50_but_50_word_source_is_standalone():
    prepared = _preprocess([
        _message(1, _words(49, "short")),
        _message(2, _words(1, "short")),
        _message(3, _words(50, "boundary")),
        _message(4, _words(1, "after")),
    ])

    assert [_field(item, "source_message_ids") for item in prepared] == [[1, 2], [3], [4]]
    assert [len(_field(item, "text").split()) for item in prepared] == [50, 50, 1]


def test_short_messages_merge_to_exactly_100_words():
    prepared = _preprocess([
        _message(1, _words(49, "first")),
        _message(2, _words(49, "second")),
        _message(3, _words(2, "third")),
    ])

    assert [_field(item, "source_message_ids") for item in prepared] == [[1, 2, 3]]
    assert len(_field(prepared[0], "text").split()) == 100


def test_word_overflow_starts_a_new_block_and_keeps_the_short_remainder():
    prepared = _preprocess([
        _message(1, _words(49, "first")),
        _message(2, _words(49, "second")),
        _message(3, _words(3, "overflow")),
    ])

    assert [_field(item, "source_message_ids") for item in prepared] == [[1, 2], [3]]
    assert [len(_field(item, "text").split()) for item in prepared] == [98, 3]


def test_preserves_source_text_whitespace_and_does_not_mutate_input_records():
    exact_text = "  leading spaces\n\n\tindented line\ntrailing spaces  "
    message = _message(1, exact_text)
    original = message.copy()

    prepared = _preprocess([message])

    assert _field(prepared[0], "text") == exact_text
    assert message == original


def test_merged_text_preserves_multiline_and_indented_source_fragments_in_order():
    first_text = "  first fragment\n\n\tfirst indentation"
    second_text = "second fragment\n  second indentation\n\n"

    prepared = _preprocess([
        _message(1, first_text, reliable_author_key="user:7"),
        _message(2, second_text, reliable_author_key="user:7"),
    ])

    assert len(prepared) == 1
    assert _field(prepared[0], "source_message_ids") == [1, 2]
    merged_text = _field(prepared[0], "text")
    assert merged_text.index(first_text) < merged_text.index(second_text)


@pytest.mark.parametrize(
    ("message_count", "expected_capacity"),
    [(5, 30), (10, 30), (20, 40), (50, 70), (100, 120)],
)
def test_context_capacity_scales_and_preprocessing_retains_every_current_message(
    message_count, expected_capacity
):
    messages = [
        _message(
            index + 1,
            f"current{index}",
            intake_sequence=index + 1,
            reliable_author_key=f"user:{index + 1}",
        )
        for index in range(message_count)
    ]

    prepared = _preprocess(messages)

    assert _context_capacity(message_count) == expected_capacity
    assert len(prepared) == message_count
    assert [
        message_id
        for utterance in prepared
        for message_id in _field(utterance, "source_message_ids")
    ] == list(range(1, message_count + 1))
