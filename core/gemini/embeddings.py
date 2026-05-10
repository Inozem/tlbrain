import os

from google import genai
from google.genai import types

from core.utils.retry import with_retry

_EMBEDDING_MODEL = "gemini-embedding-2"
_OUTPUT_DIMS = 768
_BATCH_SIZE = 100


def make_client() -> genai.Client:
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def embed(texts: list[str], client: genai.Client | None = None) -> list[list[float]]:
    if client is None:
        client = make_client()
    results: list[list[float]] = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i: i + _BATCH_SIZE]
        results.extend(_embed_batch(client, batch))
    return results


@with_retry
def _embed_batch(client: genai.Client, texts: list[str]) -> list[list[float]]:
    return [
        client.models.embed_content(
            model=_EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(output_dimensionality=_OUTPUT_DIMS),
        ).embeddings[0].values
        for text in texts
    ]
