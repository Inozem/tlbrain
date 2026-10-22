"""Google-issued OIDC token verification shared by trusted callbacks."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OIDCIdentity:
    email: str
    authorized: bool


class OIDCConfigurationError(ValueError):
    """Required trusted OIDC settings are unavailable or invalid."""


def _configuration() -> tuple[str, str]:
    audience = os.environ.get("OIDC_AUDIENCE")
    expected_email = os.environ.get("OIDC_SERVICE_ACCOUNT_EMAIL")
    if (not isinstance(audience, str) or not audience.strip()
            or not isinstance(expected_email, str) or not expected_email.strip()
            or "@" not in expected_email):
        raise OIDCConfigurationError("OIDC verifier is not configured")
    return audience, expected_email


def verify_google_id_token(token: str | None) -> OIDCIdentity | None:
    """Verify a Google ID token and bind it to the configured task identity.

    Returns ``None`` for an invalid token or a valid token from a different
    service account. Configuration errors are distinct so callers can return
    a retryable service-unavailable response.
    """
    audience, expected_email = _configuration()
    if not isinstance(token, str) or not token:
        return None

    try:
        from google.auth.transport.requests import Request
        from google.oauth2 import id_token

        claims = id_token.verify_oauth2_token(token, Request(), audience=audience)
    except Exception:
        return None

    if not isinstance(claims, dict):
        return None
    email = claims.get("email")
    if claims.get("email_verified") is not True or not isinstance(email, str):
        return None
    return OIDCIdentity(email=email, authorized=email == expected_email)

