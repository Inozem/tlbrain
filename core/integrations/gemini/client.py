"""Shared Gemini SDK client construction."""

import os

from google import genai


def make_client() -> genai.Client:
    """Create a Gemini client using the configured API key."""
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])
