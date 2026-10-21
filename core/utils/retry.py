import logging
import time
from dataclasses import dataclass
from functools import wraps
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

RETRIES = 3
BACKOFFS = [1, 2, 4]
RATE_LIMIT_BACKOFFS = [30, 60, 120]

F = TypeVar("F", bound=Callable)


@dataclass(frozen=True)
class RetryPolicy:
    """Controls how many times a decorated provider call is attempted."""

    attempts: int

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError("retry policy attempts must be at least one")


LEGACY_RETRY_POLICY = RetryPolicy(attempts=RETRIES)
SINGLE_ATTEMPT_RETRY_POLICY = RetryPolicy(attempts=1)


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    try:
        from googleapiclient.errors import HttpError
        if isinstance(exc, HttpError) and exc.status_code >= 500:
            return True
    except ImportError:
        pass
    try:
        from qdrant_client.http.exceptions import UnexpectedResponse
        if isinstance(exc, UnexpectedResponse) and exc.status_code >= 500:
            return True
    except ImportError:
        pass
    try:
        from google.genai.errors import ServerError
        if isinstance(exc, ServerError):
            return True
    except ImportError:
        pass
    try:
        import requests.exceptions
        if isinstance(exc, requests.exceptions.HTTPError):
            if exc.response is not None and exc.response.status_code >= 500:
                return True
    except ImportError:
        pass
    return False


def _is_rate_limit(exc: Exception) -> bool:
    try:
        from google.genai.errors import ClientError
        if isinstance(exc, ClientError) and getattr(exc, "code", None) == 429:
            return True
    except ImportError:
        pass
    return False


def with_retry(fn: F) -> F:
    @wraps(fn)
    def wrapper(*args, **kwargs):
        retry_policy = kwargs.pop("retry_policy", LEGACY_RETRY_POLICY)
        if not isinstance(retry_policy, RetryPolicy):
            raise TypeError("retry_policy must be a RetryPolicy")
        last_exc: Exception | None = None
        for attempt in range(retry_policy.attempts):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                if _is_rate_limit(exc):
                    backoffs = RATE_LIMIT_BACKOFFS
                elif _is_transient(exc):
                    backoffs = BACKOFFS
                else:
                    raise
                last_exc = exc
                logger.warning(
                    "Transient error on attempt %d/%d for %s: %s",
                    attempt + 1, retry_policy.attempts, fn.__name__, exc,
                )
                if attempt < retry_policy.attempts - 1:
                    time.sleep(backoffs[min(attempt, len(backoffs) - 1)])
                elif retry_policy.attempts == 1:
                    raise
        raise RuntimeError(
            f"{fn.__name__} failed after {retry_policy.attempts} attempts"
        ) from last_exc

    return wrapper  # type: ignore[return-value]
