"""Domain operation for adding a user-authored fact to a transcript."""

from core.config import get_root_folder_id
from core.indexing.payloads import _date_to_num
from core.integrations.firestore.transcripts import get_transcript_record
from core.integrations.gemini.embeddings import embed
from typing import Callable

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


def add_user_fact(
    doc_id: str, text: str, *, point_id: str | None = None, single_attempt: bool = False,
) -> str:
    """Validate and index a trimmed user fact, returning its trimmed document ID."""
    doc_id, text, payload = prepare_user_fact(doc_id, text, point_id=point_id)
    apply_user_fact_payload(payload, point_id=point_id, single_attempt=single_attempt)
    return doc_id


def prepare_user_fact(
    doc_id: str, text: str, *, point_id: str | None = None, db=None,
) -> tuple[str, str, dict]:
    """Validate canonical transcript state and build the stable point payload."""
    doc_id = doc_id.strip()
    text = text.strip()
    if not doc_id or not text:
        raise AddFactInputError("doc_id and text are required")
    if point_id is not None:
        from uuid import UUID
        try:
            UUID(point_id)
        except (ValueError, AttributeError, TypeError):
            raise AddFactInputError("point_id must be a UUID") from None

    record = get_transcript_record(doc_id) if db is None else get_transcript_record(doc_id, db)
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

    return doc_id, text, payload


def apply_user_fact_payload(
    payload: dict, *, point_id: str | None = None, single_attempt: bool = False,
    before_write: Callable[[], None] | None = None,
) -> None:
    """Embed and write an already validated fact payload once or with legacy retries."""
    text = payload["text"]
    if single_attempt:
        from core.utils.retry import SINGLE_ATTEMPT_RETRY_POLICY
        try:
            vector = embed([text], retry_policy=SINGLE_ATTEMPT_RETRY_POLICY)[0]
        except Exception:
            from core.operations.mutations import MutationNotSent
            raise MutationNotSent() from None
        if before_write is not None:
            before_write()
        if point_id is None:
            upsert_user_facts([payload], [vector], retry_policy=SINGLE_ATTEMPT_RETRY_POLICY)
        else:
            upsert_user_facts([payload], [vector], point_ids=[point_id], retry_policy=SINGLE_ATTEMPT_RETRY_POLICY)
        return
    vector = embed([text])[0]
    if point_id is None:
        upsert_user_facts([payload], [vector])
    else:
        upsert_user_facts([payload], [vector], point_ids=[point_id])
