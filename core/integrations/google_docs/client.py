"""Google Docs discovery client construction."""

from collections.abc import Sequence

from googleapiclient.discovery import build
import google_auth_httplib2
import httplib2

from core.integrations.google_auth.credentials import credentials
from core.utils.retry import LEGACY_RETRY_POLICY, RetryPolicy, SINGLE_ATTEMPT_RETRY_POLICY


def build_docs_service(
    scopes: Sequence[str], *, allow_adc_fallback: bool = False,
    retry_policy: RetryPolicy = LEGACY_RETRY_POLICY,
):
    """Build a Docs v1 service using the caller's explicit auth policy."""
    if retry_policy == SINGLE_ATTEMPT_RETRY_POLICY:
        from core.integrations.google_auth.transport import SingleAttemptHttp
        from core.utils.retry import ProviderBoundaryError
        try:
            creds = credentials(scopes, allow_adc_fallback=allow_adc_fallback,
                                retry_policy=retry_policy)
        except Exception as error:
            raise ProviderBoundaryError("google_docs", error) from error
        return build(
            "docs", "v1",
            http=google_auth_httplib2.AuthorizedHttp(
                creds, http=SingleAttemptHttp(timeout=60), max_refresh_attempts=0,
            ),
            num_retries=0,
        )
    return build(
        "docs",
        "v1",
        credentials=credentials(scopes, allow_adc_fallback=allow_adc_fallback),
    )
