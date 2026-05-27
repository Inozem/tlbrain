import json
import logging
import os
from typing import Any

from google import genai
from google.genai import types

from core.utils.retry import with_retry

logger = logging.getLogger(__name__)

_LLM_MODEL = "gemini-2.5-flash"

_PROMPT = """\
You are analyzing a fragment of a dialogue. Produce two things:

1. A summary: 2-3 sentences capturing the essence of this fragment for future search and retrieval.
   Cover in order (skip only if genuinely absent):
   - The main topic or activity being discussed.
   - Any specific details worth remembering: prices, numbers, deadlines, product names, objections, decisions.
   - Any agreements, next steps, or unresolved issues.
   Do not retell the dialogue — write only the essence.

2. Facts: notable facts mentioned in the fragment, ordered from most to least important using this priority:
   decisions and agreements first, then specific numbers and deadlines,
   then pain points and objections, then next steps, then key people,
   then business context, then general details.
   A fact is any specific, concrete piece of information worth remembering: contacts, interests, industries,
   decisions, numbers, problems, relationships, plans, or anything unusual.
   Write each fact as a short noun phrase or brief clause — no attribution, no "X said that", just the fact itself.
   Use speaker names only when the person themselves (not their statement) is the fact.
   Empty list if nothing notable found.

Rules:
- Use speaker names exactly as they appear — do not infer or rename roles.
- Always reply in English regardless of the dialogue language.

Dialogue:
{dialog}"""

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "facts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "facts"],
}


_CLIENT_DETECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "folder_name": {"type": "string", "nullable": True},
        "confidence": {"type": "number"},
    },
    "required": ["folder_name", "confidence"],
}


def call_gemini_json(prompt: str) -> dict:
    client = _client()
    return _generate_structured(client, prompt, schema=_CLIENT_DETECTION_SCHEMA)


def generate_summary_and_facts(utterances: list[dict[str, Any]]) -> tuple[str, list[str]]:
    client = _client()
    dialog = _format_dialog(utterances)
    prompt = _PROMPT.format(dialog=dialog)
    data = _generate_structured(client, prompt)
    facts = data["facts"]
    if len(facts) > 10:
        logger.debug("Truncating facts from %d to 10", len(facts))
        facts = facts[:10]
    return data["summary"], facts


def _client() -> genai.Client:
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def _format_dialog(utterances: list[dict[str, Any]]) -> str:
    return "\n".join(f"{u['speaker']}: {u['text']}" for u in utterances)


@with_retry
def _generate_text(client: genai.Client, prompt: str) -> str:
    response = client.models.generate_content(
        model=_LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            top_p=1,
        ),
    )
    return response.text.strip()


@with_retry
def _generate_structured(client: genai.Client, prompt: str, schema: dict = _RESPONSE_SCHEMA) -> dict:
    response = client.models.generate_content(
        model=_LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            top_p=1,
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )
    return json.loads(response.text)
