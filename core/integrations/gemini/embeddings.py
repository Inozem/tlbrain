from google import genai
from google.genai import types

from .client import make_client
from core.utils.retry import LEGACY_RETRY_POLICY, RetryPolicy, with_retry

_EMBEDDING_MODEL = "gemini-embedding-2"
_OUTPUT_DIMS = 768
_BATCH_SIZE = 100


def embed(
    texts: list[str], client: genai.Client | None = None, *,
    retry_policy: RetryPolicy = LEGACY_RETRY_POLICY,
) -> list[list[float]]:
    if client is None:
        client = make_client(single_attempt=retry_policy.attempts == 1)
    results: list[list[float]] = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i: i + _BATCH_SIZE]
        results.extend(_embed_batch(client, batch, retry_policy=retry_policy))
    return results


@with_retry
def _embed_batch(client: genai.Client, texts: list[str]) -> list[list[float]]:
    response = client.models.embed_content(
        model=_EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(output_dimensionality=_OUTPUT_DIMS),
    )
    return [e.values for e in response.embeddings]
