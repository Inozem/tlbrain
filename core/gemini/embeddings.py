import os
import time

from google import genai
from google.genai import types

_EMBEDDING_MODEL = "gemini-embedding-2"
_OUTPUT_DIMS = 768
_BATCH_SIZE = 100
_RETRIES = 3
_BACKOFFS = [1, 2, 4]


def embed(texts: list[str]) -> list[list[float]]:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    results: list[list[float]] = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        results.extend(_embed_batch_with_retry(client, batch))
    return results


def _embed_batch_with_retry(client: genai.Client, texts: list[str]) -> list[list[float]]:
    last_exc: Exception | None = None
    for attempt in range(_RETRIES):
        try:
            vectors = []
            for text in texts:
                response = client.models.embed_content(
                    model=_EMBEDDING_MODEL,
                    contents=text,
                    config=types.EmbedContentConfig(output_dimensionality=_OUTPUT_DIMS),
                )
                vectors.append(response.embeddings[0].values)
            return vectors
        except Exception as exc:
            last_exc = exc
            time.sleep(_BACKOFFS[attempt])
    raise RuntimeError(f"Gemini embedding failed after {_RETRIES} attempts") from last_exc
