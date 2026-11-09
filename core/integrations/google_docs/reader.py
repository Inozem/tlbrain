"""Strict-OAuth Google Docs text reader."""

from core.integrations.google_docs.client import build_docs_service
from core.integrations.google_docs.fill import _document_text
from core.utils.retry import LEGACY_RETRY_POLICY, RetryPolicy, with_retry

SCOPES = ["https://www.googleapis.com/auth/documents.readonly"]


def _build_docs_service(retry_policy: RetryPolicy = LEGACY_RETRY_POLICY):
    return build_docs_service(SCOPES, retry_policy=retry_policy)


@with_retry
def read_google_doc(doc_id: str, *, retry_policy: RetryPolicy = LEGACY_RETRY_POLICY) -> str:
    service = _build_docs_service(retry_policy)
    doc = service.documents().get(documentId=doc_id).execute()
    return _extract_text(doc)


def _extract_text(doc: dict) -> str:
    legacy_text = _extract_legacy_text(doc)
    if _declares_json_string_v1(legacy_text):
        text, _revision = _document_text(doc)
        if text is None:
            raise ValueError("Unsupported or malformed content in encoded Google Doc")
        return text
    return legacy_text


def _extract_legacy_text(doc: dict) -> str:
    lines = []
    for element in doc.get("body", {}).get("content", []):
        paragraph = element.get("paragraph")
        if not paragraph:
            continue
        parts = []
        for elem in paragraph.get("elements", []):
            text_run = elem.get("textRun")
            if text_run:
                parts.append(text_run.get("content", ""))
        line = "".join(parts).rstrip("\n")
        if line.strip():
            lines.append(line)
    return "\n".join(lines)


def _declares_json_string_v1(text: str) -> bool:
    """Check the parser-compatible encoding declaration in the document header."""
    lines = text.splitlines()
    try:
        separator = lines.index("---")
    except ValueError:
        return False

    encoding = None
    for line in lines[:separator]:
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        if key.strip() == "TEXT_ENCODING":
            encoding = value.strip()
    return encoding == "json-string-v1"
