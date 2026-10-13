"""Compatibility exports for the canonical Google Docs reader."""

from core.integrations.google_docs.reader import (
    SCOPES,
    _build_docs_service,
    _extract_text,
    read_google_doc,
)

__all__ = ["SCOPES", "_build_docs_service", "_extract_text", "read_google_doc"]
