from __future__ import annotations

import pytest

from core.dialogs.classification import (
    AmbiguousFolderPathError,
    FolderPathNotFoundError,
    FolderSelectionRequest,
    classify_dialog_folder,
)
from core.dialogs.models import DialogUtterance


ROOT = "root"
FOLDERS = {
    "alpha": {"name": "Alpha", "parent_id": ROOT},
    "beta": {"name": "Beta", "parent_id": ROOT},
    "nested": {"name": "Nested", "parent_id": "alpha"},
    "unassigned": {"name": "_unassigned", "parent_id": ROOT},
}


def _request(**overrides):
    values = {
        "title": "Planning",
        "utterances": (DialogUtterance("Alex", "line"),),
        "reliable_speakers": ("Alex",),
    }
    values.update(overrides)
    return FolderSelectionRequest(**values)


def test_explicit_path_wins_without_callbacks():
    def forbidden(*_args):
        pytest.fail("automatic classification callback should not run")

    result = classify_dialog_folder(
        _request(folder_path=("Alpha", "Nested")), FOLDERS, ROOT,
        speaker_index_lookup=forbidden, suggestion=forbidden,
    )

    assert result.folder_id == "nested"
    assert not result.requires_unassigned


def test_explicit_root_selects_root_without_callbacks():
    def forbidden(*_args):
        pytest.fail("automatic classification callback should not run")

    result = classify_dialog_folder(
        _request(folder_path=()), FOLDERS, ROOT,
        speaker_index_lookup=forbidden, suggestion=forbidden,
    )

    assert result.folder_id == ROOT


def test_explicit_missing_or_ambiguous_paths_fail():
    duplicate_folders = dict(FOLDERS)
    duplicate_folders["alpha-copy"] = {"name": "Alpha", "parent_id": ROOT}

    with pytest.raises(FolderPathNotFoundError, match="not found"):
        classify_dialog_folder(
            _request(folder_path=("Missing",)), FOLDERS, ROOT,
            speaker_index_lookup=lambda _speaker: (), suggestion=lambda *_args: None,
        )
    with pytest.raises(AmbiguousFolderPathError, match="ambiguous"):
        classify_dialog_folder(
            _request(folder_path=("Alpha",)), duplicate_folders, ROOT,
            speaker_index_lookup=lambda _speaker: (), suggestion=lambda *_args: None,
        )


def test_explicit_folder_id_must_be_under_configured_root():
    snapshot = dict(FOLDERS)
    snapshot["other-root"] = {"name": "Outside", "parent_id": "another-root"}

    with pytest.raises(FolderPathNotFoundError, match="under the configured root"):
        classify_dialog_folder(
            _request(folder_id="other-root"), snapshot, ROOT,
            speaker_index_lookup=lambda _speaker: (), suggestion=lambda *_args: None,
        )


def test_unique_speaker_candidate_returns_without_suggestion():
    def forbidden(*_args):
        pytest.fail("unique speaker match should bypass suggestions")

    result = classify_dialog_folder(
        _request(), FOLDERS, ROOT,
        speaker_index_lookup=lambda speaker: ["alpha"] if speaker == "Alex" else [],
        suggestion=forbidden,
    )

    assert result.folder_id == "alpha"
    assert result.speaker_candidate_ids == ("alpha",)
    assert not result.speaker_candidates_ambiguous


def test_multiple_speaker_candidates_try_title_then_transcript_and_keep_ambiguity():
    utterances = tuple(DialogUtterance("Alex", f"line-{i}") for i in range(18))
    calls = []

    def suggest(stage, candidate_ids, title, excerpt, topic_hint):
        calls.append((stage, candidate_ids, title, excerpt, topic_hint))
        if stage == "title":
            return "Alpha", 0.79
        return "Beta", 0.8

    result = classify_dialog_folder(
        _request(utterances=utterances), FOLDERS, ROOT,
        speaker_index_lookup=lambda _speaker: ["alpha", "beta"], suggestion=suggest,
    )

    assert result.folder_id == "beta"
    assert result.speaker_candidate_ids == ("alpha", "beta")
    assert result.speaker_candidates_ambiguous
    assert [call[0] for call in calls] == ["title", "transcript"]
    assert calls[0][2] == "Planning"
    assert calls[1][3] == utterances[:15]


def test_topic_hint_is_passed_to_suggestion_unchanged():
    topic_hint = "  Finance / Q3  "
    received_hints = []

    def suggest(_stage, _candidate_ids, _title, _excerpt, received_hint):
        received_hints.append(received_hint)
        return "Alpha", 1.0

    result = classify_dialog_folder(
        _request(topic_hint=topic_hint), FOLDERS, ROOT,
        speaker_index_lookup=lambda _speaker: [], suggestion=suggest,
    )

    assert result.folder_id == "alpha"
    assert received_hints == [topic_hint]


def test_suggestion_requires_exact_name_and_confidence_at_least_point_eight():
    responses = iter([("alpha", 1.0), ("Alpha", 0.8)])
    stages = []

    def suggest(stage, *_args):
        stages.append(stage)
        return next(responses)

    result = classify_dialog_folder(
        _request(), FOLDERS, ROOT,
        speaker_index_lookup=lambda _speaker: ["alpha", "beta"], suggestion=suggest,
    )

    assert result.folder_id == "alpha"
    assert stages == ["title", "transcript"]


@pytest.mark.parametrize("speaker", ["Speaker 1", "СПИКЕР 2", "participant   3", "УЧАСТНИК4"])
def test_placeholder_speakers_are_not_queried(speaker):
    queried = []

    def lookup(value):
        queried.append(value)
        return ["alpha"]

    result = classify_dialog_folder(
        _request(reliable_speakers=(speaker,)), FOLDERS, ROOT,
        speaker_index_lookup=lookup,
        suggestion=lambda *_args: ("Beta", 1.0),
    )

    assert queried == []
    assert result.folder_id == "beta"
    assert not result.speaker_candidates_ambiguous


def test_no_valid_suggestion_returns_terminal_unassigned_result():
    result = classify_dialog_folder(
        _request(), FOLDERS, ROOT,
        speaker_index_lookup=lambda _speaker: [],
        suggestion=lambda *_args: ("_unassigned", 1.0),
    )

    assert result.folder_id is None
    assert result.requires_unassigned
