import os
from qdrant_client import QdrantClient

_client: QdrantClient | None = None
_single_attempt_client: QdrantClient | None = None


def get_client(*, single_attempt: bool = False) -> QdrantClient:
    """Return the legacy client or an isolated client without its startup probe."""
    global _client, _single_attempt_client
    if single_attempt:
        if _single_attempt_client is None:
            _single_attempt_client = QdrantClient(
                url=os.environ["QDRANT_URL"],
                api_key=os.environ["QDRANT_API_KEY"],
                check_compatibility=False,
            )
        return _single_attempt_client
    if _client is None:
        _client = QdrantClient(
            url=os.environ["QDRANT_URL"],
            api_key=os.environ["QDRANT_API_KEY"],
        )
    return _client
