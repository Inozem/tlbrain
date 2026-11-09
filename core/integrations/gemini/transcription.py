"""Gemini audio transcription boundary."""

from __future__ import annotations

from google.genai import types


TRANSCRIPTION_MODEL = "gemini-3.5-transcribe"


def transcription_config() -> types.GenerateContentConfig:
    """Build the fixed verbatim, automatically detected transcription config."""
    return types.GenerateContentConfig(
        audio_transcription_config=types.AudioTranscriptionConfig(
            mode="VERBATIM",
            language_codes=[],
        ),
    )


async def transcribe_part(client, part: types.Part) -> str:
    """Submit exactly one audio part and return the provider text unchanged."""
    response = await client.aio.models.generate_content(
        model=TRANSCRIPTION_MODEL,
        contents=part,
        config=transcription_config(),
    )
    text = getattr(response, "text", None)
    if not isinstance(text, str) or not text:
        raise ValueError("Gemini returned no transcription text")
    return text
