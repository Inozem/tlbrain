"""Google Docs discovery client construction."""

from collections.abc import Sequence

from googleapiclient.discovery import build

from core.integrations.google_auth.credentials import credentials


def build_docs_service(
    scopes: Sequence[str], *, allow_adc_fallback: bool = False
):
    """Build a Docs v1 service using the caller's explicit auth policy."""
    return build(
        "docs",
        "v1",
        credentials=credentials(scopes, allow_adc_fallback=allow_adc_fallback),
    )
