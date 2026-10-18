"""Domain operation for adding a user-authored fact to a transcript."""

from core.config import get_root_folder_id
from core.indexing.payloads import _date_to_num
from core.integrations.firestore.transcripts import get_transcript_record
from core.integrations.gemini.embeddings import embed
from core.integrations.qdrant.writer import upsert_user_facts


class AddFactInputError(ValueError):
    """The supplied document ID or fact text is empty after trimming."""


class AddFactDocumentNotFound(LookupError):
    """The requested transcript has no canonical record."""

    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        super().__init__(f"Document not found: {doc_id}")


class AddFactTranscriptNotSynced(ValueError):
    """The requested transcript has not finished syncing."""

    def __init__(self, status: object):
        self.status = status
        super().__init__(
            f"Document is not synced yet (status={status}). Try again after sync completes."
        )


def add_user_fact(doc_id: str, text: str) -> str:
    """Validate and index a trimmed user fact, returning its trimmed document ID."""
    doc_id = doc_id.strip()
    text = text.strip()
    if not doc_id or not text:
        raise AddFactInputError("doc_id and text are required")

    record = get_transcript_record(doc_id)
    if not record:
        raise AddFactDocumentNotFound(doc_id)
    status = record.get("status")
    if status != "synced":
        raise AddFactTranscriptNotSynced(status)

    dialog_date = record.get("dialog_date", "")
    payload = {
        "type": "user_fact",
        "doc_id": doc_id,
        "text": text,
        "root_folder_id": get_root_folder_id(),
        "parent_id": record.get("parent_id", ""),
        "dialog_date": dialog_date,
        "dialog_date_num": _date_to_num(dialog_date) if dialog_date else 0,
    }

    vector = embed([text])[0]
    upsert_user_facts([payload], [vector])
    return doc_id
