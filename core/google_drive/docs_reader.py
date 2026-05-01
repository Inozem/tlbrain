import google.auth
from googleapiclient.discovery import build

from core.utils.retry import with_retry

SCOPES = ["https://www.googleapis.com/auth/documents.readonly"]


def _build_docs_service():
    credentials, _ = google.auth.default(scopes=SCOPES)
    return build("docs", "v1", credentials=credentials)


@with_retry
def read_google_doc(doc_id: str) -> str:
    service = _build_docs_service()
    doc = service.documents().get(documentId=doc_id).execute()
    return _extract_text(doc)


def _extract_text(doc: dict) -> str:
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
