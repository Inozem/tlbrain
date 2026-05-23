import logging
import time
from functools import wraps
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

RETRIES = 3
BACKOFFS = [1, 2, 4]

F = TypeVar("F", bound=Callable)


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
        from google.genai.errors import ClientError
        if isinstance(exc, ClientError) and exc.status_code == 429:
            return True
    except ImportError:
        pass
    return False


def with_retry(fn: F) -> F:
    @wraps(fn)
    def wrapper(*args, **kwargs):
        last_exc: Exception | None = None
        for attempt in range(RETRIES):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                if not _is_transient(exc):
                    raise
                last_exc = exc
                logger.warning(
                    "Transient error on attempt %d/%d for %s: %s",
                    attempt + 1, RETRIES, fn.__name__, exc,
                )
                if attempt < RETRIES - 1:
                    time.sleep(BACKOFFS[attempt])
        raise RuntimeError(
            f"{fn.__name__} failed after {RETRIES} attempts"
        ) from last_exc

    return wrapper  # type: ignore[return-value]
