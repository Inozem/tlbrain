"""Shared Google credential construction for integration callers."""

import os
from collections.abc import Sequence


def credentials(scopes: Sequence[str], allow_adc_fallback: bool = False):
    """Build refreshed user OAuth credentials, optionally falling back to ADC.

    Callers must opt in to Application Default Credentials explicitly. This
    keeps user-data readers from silently changing identity when OAuth is not
    configured.
    """
    refresh_token = os.environ.get("GOOGLE_REFRESH_TOKEN")
    if refresh_token:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials

        creds = Credentials(
            token=None,
            refresh_token=refresh_token,
            client_id=os.environ["GOOGLE_CLIENT_ID"],
            client_secret=os.environ["GOOGLE_CLIENT_SECRET"],
            token_uri="https://oauth2.googleapis.com/token",
            scopes=list(scopes),
        )
        creds.refresh(Request())
        return creds

    if allow_adc_fallback:
        import google.auth

        creds, _ = google.auth.default(scopes=list(scopes))
        return creds

    raise RuntimeError(
        "GOOGLE_REFRESH_TOKEN is not set — Drive access requires user OAuth credentials"
    )
