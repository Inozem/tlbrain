"""Protocol-neutral read use cases over the canonical retrieval repositories."""

from typing import Any

from core.integrations.firestore.transcripts import (
    get_sync_status as get_repository_sync_status,
    list_transcripts as list_repository_transcripts,
)
from core.integrations.google_drive import folder_service
from core.integrations.google_drive.folder_service import FolderNotFound
from core.retrieval.run import run_retrieval
from core.retrieval.transcripts import get_transcripts


class TranscriptListingError(RuntimeError):
    """A failure while reading the transcript list repository."""


def query(
    query: str,
    keywords: str | None = None,
    folder_path: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the existing semantic, keyword, and pinned-fact retrieval flow."""
    return run_retrieval(
        query=query,
        keywords=keywords,
        folder_path=folder_path,
        date_from=date_from,
        date_to=date_to,
    )


def get_transcript(
    doc_id: str | None = None,
    folder_path: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Retrieve transcript segments through the established retrieval flow."""
    return get_transcripts(
        doc_id=doc_id,
        folder_path=folder_path,
        date_from=date_from,
        date_to=date_to,
        limit=limit,
    )


def list_folders() -> dict[str, Any]:
    """Return the canonical folder tree and its transcript aggregates."""
    return folder_service.list_folders()


def sync_status() -> dict[str, Any]:
    """Return the read-only transcript sync status aggregate."""
    return get_repository_sync_status()


def list_recent_transcripts(
    folder_path: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> dict[str, Any]:
    """List date-descending transcripts, optionally within folder subtrees.

    Folder resolution and expansion happen before the repository error boundary,
    preserving their distinct domain and infrastructure failures.
    """
    folder_ids = None
    if folder_path:
        terminal_ids = folder_service.resolve_registered_folder_path(folder_path)
        if not terminal_ids:
            raise FolderNotFound(folder_path, "folder")
        folder_ids = [
            folder_id
            for terminal_id in terminal_ids
            for folder_id in folder_service.expand_registered_subtree(terminal_id)
        ]

    try:
        return list_repository_transcripts(
            folder_ids=folder_ids,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
            offset=offset,
        )
    except Exception as error:
        raise TranscriptListingError(str(error)) from error
