"""Provider-neutral folder classification for normalized dialogs."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import re

from core.dialogs.models import DialogUtterance
from core.integrations.google_drive.folders import resolve_folder_path


_CONFIDENCE_THRESHOLD = 0.8
_SPEAKER_MIN_CLIENTS = 1
_SPEAKER_MAX_CLIENTS = 4
_PLACEHOLDER_SPEAKER_RE = re.compile(
    r"^(speaker|спикер|participant|участник)\s*\d+$", re.IGNORECASE
)


class FolderPathNotFoundError(ValueError):
    """An explicitly requested folder path or ID is not under the root."""


class AmbiguousFolderPathError(ValueError):
    """An explicitly requested folder path identifies multiple folders."""


@dataclass(frozen=True, slots=True)
class FolderSelectionRequest:
    """Immutable normalized data used to classify one dialog folder."""

    title: str
    utterances: tuple[DialogUtterance, ...]
    reliable_speakers: tuple[str, ...] = ()
    folder_path: tuple[str, ...] | None = None
    folder_id: str | None = None
    topic_hint: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "utterances", tuple(self.utterances))
        object.__setattr__(self, "reliable_speakers", tuple(self.reliable_speakers))
        if self.folder_path is not None:
            object.__setattr__(self, "folder_path", tuple(self.folder_path))
        if self.folder_path is not None and self.folder_id is not None:
            raise ValueError("specify either folder_path or folder_id, not both")


@dataclass(frozen=True, slots=True)
class FolderSelectionResult:
    """Selected folder and speaker-match metadata without side effects."""

    folder_id: str | None
    speaker_candidate_ids: tuple[str, ...] = ()
    requires_unassigned: bool = False

    @property
    def speaker_candidates_ambiguous(self) -> bool:
        return len(self.speaker_candidate_ids) >= 2


SpeakerIndexLookup = Callable[[str], Iterable[str]]
FolderSuggestion = Callable[
    [str, tuple[str, ...], str, tuple[DialogUtterance, ...], str | None],
    tuple[str | None, float] | None,
]


def _is_placeholder_speaker(name: str) -> bool:
    return bool(_PLACEHOLDER_SPEAKER_RE.fullmatch(name))


def _is_under_root(folder_id: str, all_folders: Mapping[str, Mapping], root_folder_id: str) -> bool:
    current = folder_id
    visited: set[str] = set()
    while current and current != root_folder_id and current not in visited:
        visited.add(current)
        data = all_folders.get(current)
        if data is None:
            return False
        current = data.get("parent_id", "")
    return current == root_folder_id


def _select_suggestion(
    stage: str,
    candidate_ids: tuple[str, ...],
    request: FolderSelectionRequest,
    all_folders: Mapping[str, Mapping],
    suggestion: FolderSuggestion,
) -> str | None:
    folder_names = tuple(
        all_folders.get(folder_id, {}).get("name", folder_id)
        for folder_id in candidate_ids
    )
    name_to_ids: dict[str, list[str]] = {}
    for folder_id, folder_name in zip(candidate_ids, folder_names):
        name_to_ids.setdefault(folder_name, []).append(folder_id)

    try:
        result = suggestion(
            stage,
            candidate_ids,
            request.title,
            request.utterances[:15],
            request.topic_hint,
        )
        if result is None:
            return None
        folder_name, confidence = result
        if (
            folder_name
            and folder_name in name_to_ids
            and confidence >= _CONFIDENCE_THRESHOLD
        ):
            return name_to_ids[folder_name][0]
    except Exception:
        # Suggestion providers are advisory. Failure at one stage falls through.
        return None
    return None


def classify_dialog_folder(
    request: FolderSelectionRequest,
    all_folders: Mapping[str, Mapping],
    root_folder_id: str,
    *,
    speaker_index_lookup: SpeakerIndexLookup,
    suggestion: FolderSuggestion,
) -> FolderSelectionResult:
    """Choose a folder from a supplied snapshot using injected pure callbacks."""

    if request.folder_id is not None:
        if request.folder_id == root_folder_id:
            return FolderSelectionResult(folder_id=root_folder_id)
        if request.folder_id not in all_folders or not _is_under_root(
            request.folder_id, all_folders, root_folder_id
        ):
            raise FolderPathNotFoundError(
                f"folder ID {request.folder_id!r} was not found under the configured root"
            )
        return FolderSelectionResult(folder_id=request.folder_id)

    if request.folder_path is not None:
        if not request.folder_path:
            return FolderSelectionResult(folder_id=root_folder_id)
        matches = resolve_folder_path(list(request.folder_path), dict(all_folders), root_folder_id)
        if not matches:
            raise FolderPathNotFoundError(
                f"folder path {list(request.folder_path)!r} was not found under the configured root"
            )
        if len(matches) > 1:
            raise AmbiguousFolderPathError(
                f"folder path {list(request.folder_path)!r} is ambiguous under the configured root"
            )
        return FolderSelectionResult(folder_id=matches[0])

    all_indexed_ids: set[str] = set()
    speaker_candidates: set[str] = set()
    for speaker in request.reliable_speakers:
        if _is_placeholder_speaker(speaker):
            continue
        indexed_ids: list[str] = []
        seen: set[str] = set()
        for folder_id in speaker_index_lookup(speaker):
            if folder_id not in seen:
                seen.add(folder_id)
                if all_folders.get(folder_id, {}).get("name") != "_unassigned":
                    indexed_ids.append(folder_id)
            if len(indexed_ids) >= 5:
                break
        all_indexed_ids.update(indexed_ids)
        if _SPEAKER_MIN_CLIENTS <= len(indexed_ids) <= _SPEAKER_MAX_CLIENTS:
            speaker_candidates.update(indexed_ids)

    candidates = tuple(sorted(speaker_candidates))
    all_candidates = tuple(sorted(all_indexed_ids))
    if len(candidates) == 1:
        return FolderSelectionResult(folder_id=candidates[0], speaker_candidate_ids=candidates)

    if not all_candidates:
        all_candidates = tuple(
            folder_id
            for folder_id, data in all_folders.items()
            if data.get("name") != "_unassigned"
        )

    if not all_candidates:
        return FolderSelectionResult(
            folder_id=None,
            speaker_candidate_ids=candidates,
            requires_unassigned=True,
        )

    ids_to_check = candidates if candidates else all_candidates
    selected = _select_suggestion("title", ids_to_check, request, all_folders, suggestion)
    if selected is None:
        selected = _select_suggestion("transcript", ids_to_check, request, all_folders, suggestion)
    if selected is not None:
        return FolderSelectionResult(folder_id=selected, speaker_candidate_ids=candidates)
    return FolderSelectionResult(
        folder_id=None,
        speaker_candidate_ids=candidates,
        requires_unassigned=True,
    )
