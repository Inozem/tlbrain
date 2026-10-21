"""Cross-store operations for transcript documents."""

from dataclasses import dataclass
from typing import Callable

from core.config import get_root_folder_id
from core.integrations.firestore import folders, transcripts
from core.integrations.firestore import client as firestore_client
from core.integrations.google_drive.files import move_file_to_folder, rename_file
from core.integrations.google_drive.folders import resolve_folder_path
from core.integrations.qdrant.writer import delete_by_doc_id, set_payload_parent_id
from core.parsing.parser import is_valid_format


class FolderPathNotFound(ValueError):
    def __init__(self, path: list[str]):
        self.path = path
        super().__init__(f"Folder not found: {'/'.join(path)}")


class AmbiguousFolderPathError(ValueError):
    def __init__(self, path: list[str], count: int):
        self.path = path
        self.count = count
        super().__init__(f"Ambiguous folder path: {'/'.join(path)} matches {count} folders.")


class TranscriptNotFound(ValueError):
    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        super().__init__(f"Document not found: {doc_id}")


@dataclass(frozen=True)
class MoveTranscriptResult:
    unassigned_transcripts: list[dict]

    @property
    def unassigned_count(self) -> int:
        return len(self.unassigned_transcripts)


def move_transcript(doc_id: str, folder_path: list[str]) -> MoveTranscriptResult:
    """Move a transcript and update its canonical indexes in established order."""
    terminal_ids = resolve_folder_path(folder_path, folders.get_all_folders(), get_root_folder_id())
    if not terminal_ids:
        raise FolderPathNotFound(folder_path)
    if len(terminal_ids) > 1:
        raise AmbiguousFolderPathError(folder_path, len(terminal_ids))
    new_folder_id = terminal_ids[0]

    return apply_move_transcript(doc_id, new_folder_id, get_root_folder_id())


def resolve_move_target(folder_path: list[str], db=None) -> str:
    """Resolve and validate a unique target folder for guarded move planning."""
    folder_data = folders.get_all_folders() if db is None else folders.get_all_folders(db)
    terminal_ids = resolve_folder_path(folder_path, folder_data, get_root_folder_id())
    if not terminal_ids:
        raise FolderPathNotFound(folder_path)
    if len(terminal_ids) > 1:
        raise AmbiguousFolderPathError(folder_path, len(terminal_ids))
    return terminal_ids[0]


def apply_move_transcript(
    doc_id: str, new_folder_id: str, root_folder_id: str, *, single_attempt: bool = False,
    db=None, before_bookkeeping: Callable[[], None] | None = None,
) -> MoveTranscriptResult:
    """Apply a previously resolved move in the established Drive/Firestore/Qdrant order."""
    move_file_to_folder(doc_id, new_folder_id)
    if before_bookkeeping is not None:
        before_bookkeeping()
    if db is None:
        transcripts.move_transcript_record(doc_id, new_folder_id)
    else:
        transcripts.move_transcript_record(doc_id, new_folder_id, db)
    if before_bookkeeping is not None:
        before_bookkeeping()
    if single_attempt:
        from core.utils.retry import SINGLE_ATTEMPT_RETRY_POLICY
        set_payload_parent_id(doc_id, root_folder_id, new_folder_id, retry_policy=SINGLE_ATTEMPT_RETRY_POLICY)
    else:
        set_payload_parent_id(doc_id, root_folder_id, new_folder_id)

    db = db if db is not None else firestore_client.get_client()
    unassigned_folder_id = folders.get_folder_id_by_name("_unassigned", db)
    unassigned = (
        transcripts.list_unassigned_transcripts(unassigned_folder_id, db)
        if unassigned_folder_id else []
    )
    return MoveTranscriptResult(unassigned_transcripts=unassigned)


def rename_transcript(doc_id: str, new_title: str) -> None:
    """Rename a transcript in Drive and update its canonical transcript record."""
    if not transcripts.get_transcript_record(doc_id):
        raise TranscriptNotFound(doc_id)
    apply_rename_transcript(doc_id, new_title)


def apply_rename_transcript(
    doc_id: str, new_title: str, db=None, *, before_bookkeeping: Callable[[], None] | None = None,
) -> None:
    """Apply a title after the caller has validated and planned the transcript."""
    rename_file(doc_id, new_title)
    if before_bookkeeping is not None:
        before_bookkeeping()
    if db is None:
        transcripts.update_index(doc_id, {"source_file": new_title})
    else:
        transcripts.update_index(doc_id, {"source_file": new_title}, db)


def validate_document_format(doc_id: str, raw_text: str, indexed: bool) -> bool:
    """Validate a transcript and mark an existing index when its format is invalid."""
    if is_valid_format(raw_text):
        return True
    if indexed:
        transcripts.mark_error(doc_id, "invalid TLBrain document format", error_stage="invalid_format")
    return False


def delete_indexed_transcript(doc_id: str, root_folder_id: str, indexed: bool) -> None:
    """Delete only this document's vector points within the configured root, then its index."""
    if not indexed:
        return
    delete_by_doc_id(doc_id, root_folder_id)
    transcripts.delete_index(doc_id)
