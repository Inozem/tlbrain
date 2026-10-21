"""Safe, revision-guarded filling of a known native Google Doc."""

from __future__ import annotations

from dataclasses import dataclass

from core.integrations.google_drive.files import read_native_doc_metadata

_DOC_MIME_TYPE = "application/vnd.google-apps.document"


@dataclass(frozen=True, slots=True)
class GoogleDocFill:
    state: str
    document_id: str
    revision_id: str | None = None
    modified_time: str | None = None


def _document_text(document: object) -> tuple[str | None, str | None]:
    """Return full text and revision only for a supported paragraph body."""
    if not isinstance(document, dict):
        return None, None
    revision = document.get("revisionId")
    if not isinstance(revision, str) or not revision:
        revision = None
    body = document.get("body")
    content = body.get("content") if isinstance(body, dict) else None
    if not isinstance(content, list) or not content:
        return None, revision

    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict) or not isinstance(item.get("paragraph"), dict):
            return None, revision
        paragraph = item.get("paragraph")
        elements = paragraph.get("elements") if isinstance(paragraph, dict) else None
        if not isinstance(elements, list):
            return None, revision
        for element in elements:
            if not isinstance(element, dict):
                return None, revision
            text_run = element.get("textRun")
            if not isinstance(text_run, dict) or not isinstance(text_run.get("content"), str):
                return None, revision
            parts.append(text_run["content"])
    text = "".join(parts)
    # Docs terminates every paragraph with a structural newline. Remove only
    # the final one so user-provided trailing blank lines remain intact.
    if text.endswith("\n"):
        text = text[:-1]
    return text, revision


def _confirmed_result(drive, document_id: str, revision_id: str | None) -> GoogleDocFill:
    try:
        metadata = read_native_doc_metadata(drive, document_id)
        modified_time = metadata.get("modifiedTime")
        if (
            metadata.get("mimeType") != _DOC_MIME_TYPE
            or not isinstance(modified_time, str)
            or not modified_time.strip()
        ):
            raise RuntimeError("metadata unavailable")
    except Exception:
        return GoogleDocFill("fill_outcome_unknown", document_id, revision_id)
    return GoogleDocFill("confirmed", document_id, revision_id, modified_time)


def _read_document(docs, document_id: str) -> tuple[str | None, str | None]:
    try:
        document = docs.documents().get(documentId=document_id).execute()
    except Exception:
        return None, None
    return _document_text(document)


def fill_google_doc_safely(
    docs,
    *,
    drive,
    document_id: str,
    text: str,
    speaker_ranges: list[tuple[int, int]],
    expected_revision_id: str,
) -> GoogleDocFill:
    """Fill a verified empty native Doc once and reconcile uncertain writes."""
    current_text, current_revision = _read_document(docs, document_id)
    if current_text == text:
        return _confirmed_result(drive, document_id, current_revision)
    if current_text is None:
        return GoogleDocFill("fill_outcome_unknown", document_id, current_revision)
    if current_text not in {"", "\n"}:
        return GoogleDocFill("revision_conflict", document_id, current_revision)
    if not current_revision or current_revision != expected_revision_id:
        return GoogleDocFill("revision_conflict", document_id, current_revision)

    requests = [{"insertText": {"location": {"index": 1}, "text": text}}]
    for start, end in speaker_ranges:
        requests.append({
            "updateTextStyle": {
                "range": {"startIndex": start + 1, "endIndex": end + 1},
                "textStyle": {"bold": True},
                "fields": "bold",
            }
        })
    try:
        response = docs.documents().batchUpdate(
            documentId=document_id,
            body={"requests": requests},
            writeControl={"requiredRevisionId": expected_revision_id},
        ).execute()
    except Exception:
        recovered_text, recovered_revision = _read_document(docs, document_id)
        if recovered_text == text:
            return _confirmed_result(drive, document_id, recovered_revision)
        if recovered_text is not None and recovered_text not in {"", "\n"}:
            return GoogleDocFill("revision_conflict", document_id, recovered_revision)
        return GoogleDocFill("fill_outcome_unknown", document_id, recovered_revision)

    revision = None
    if isinstance(response, dict):
        control = response.get("writeControl")
        if isinstance(control, dict):
            revision = control.get("requiredRevisionId")
        if not isinstance(revision, str) or not revision:
            revision = response.get("revisionId")
        if not isinstance(revision, str) or not revision:
            revision = None
    return _confirmed_result(drive, document_id, revision)
