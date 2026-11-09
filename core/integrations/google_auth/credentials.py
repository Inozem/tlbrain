"""Shared Google credential construction for integration callers."""

import os
from collections.abc import Sequence

from core.utils.retry import LEGACY_RETRY_POLICY, RetryPolicy, SINGLE_ATTEMPT_RETRY_POLICY


def _single_attempt_credentials(*, token: str | None, refresh_token: str,
                                client_id: str, client_secret: str, token_uri: str):
    """Build credentials whose installed google-auth refresh disables retries."""
    from google.oauth2.credentials import Credentials

    class SingleAttemptCredentials(Credentials):
        def _perform_refresh_token(self, request):
            from google.oauth2 import _client

            scopes = self._scopes if self._scopes is not None else self._default_scopes
            access_token, refreshed_token, expiry, grant_response = _client.refresh_grant(
                request, self._token_uri, self._refresh_token, self._client_id,
                self._client_secret, scopes=scopes, rapt_token=self._rapt_token,
                can_retry=False,
            )
            self.token = access_token
            self.expiry = expiry
            self._refresh_token = refreshed_token
            self._id_token = grant_response.get("id_token")
            self._rapt_token = None
            if scopes and "scope" in grant_response:
                self._granted_scopes = grant_response["scope"].split()

    return SingleAttemptCredentials(
        token=token, refresh_token=refresh_token, client_id=client_id,
        client_secret=client_secret, token_uri=token_uri,
    )


def credentials(scopes: Sequence[str], allow_adc_fallback: bool = False, *,
                retry_policy: RetryPolicy = LEGACY_RETRY_POLICY):
    """Build refreshed user OAuth credentials, optionally falling back to ADC.

    Callers must opt in to Application Default Credentials explicitly. This
    keeps user-data readers from silently changing identity when OAuth is not
    configured.
    """
    refresh_token = os.environ.get("GOOGLE_REFRESH_TOKEN")
    if refresh_token:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials

        # Preserve the existing user token grant; requested scopes apply to ADC.
        if retry_policy == SINGLE_ATTEMPT_RETRY_POLICY:
            import requests
            from requests.adapters import HTTPAdapter

            session = requests.Session()
            session.mount("https://", HTTPAdapter(max_retries=0))
            session.mount("http://", HTTPAdapter(max_retries=0))
            request = Request(session=session)
            creds = _single_attempt_credentials(
                token=None, refresh_token=refresh_token,
                client_id=os.environ["GOOGLE_CLIENT_ID"],
                client_secret=os.environ["GOOGLE_CLIENT_SECRET"],
                token_uri="https://oauth2.googleapis.com/token",
            )
        else:
            request = Request()
            creds = Credentials(
                token=None,
                refresh_token=refresh_token,
                client_id=os.environ["GOOGLE_CLIENT_ID"],
                client_secret=os.environ["GOOGLE_CLIENT_SECRET"],
                token_uri="https://oauth2.googleapis.com/token",
            )
        creds.refresh(request)
        return creds

    if allow_adc_fallback:
        import google.auth

        creds, _ = google.auth.default(scopes=list(scopes))
        return creds

    raise RuntimeError(
        "GOOGLE_REFRESH_TOKEN is not set — Drive access requires user OAuth credentials"
    )
