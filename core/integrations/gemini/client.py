"""Shared Gemini SDK client construction."""

import os

from google import genai
from google.genai import types


def make_client(*, single_attempt: bool = False) -> genai.Client:
    """Create a Gemini client using the configured API key."""
    api_key = os.environ["GEMINI_API_KEY"]
    if single_attempt:
        return genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
    return genai.Client(api_key=api_key)
