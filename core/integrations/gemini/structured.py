"""Transport options for Gemini's forward-compatible structured JSON format."""

from copy import deepcopy

from google.genai import types


def json_response_options(schema: dict) -> types.HttpOptions:
    """Return request options for native Developer API JSON schema output."""
    return types.HttpOptions(extra_body={
        "generationConfig": {
            "responseFormat": {
                "text": {
                    "mimeType": "APPLICATION_JSON",
                    "schema": deepcopy(schema),
                },
            },
        },
    })
