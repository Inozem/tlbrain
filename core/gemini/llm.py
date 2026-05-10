import json
import logging
import os
import time
from typing import Any

from google import genai
from google.genai import types

from core.utils.retry import BACKOFFS, RETRIES, with_retry

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
- Reply with valid JSON only, no markdown, no explanation.

Required JSON structure:
{{
  "summary": "...",
  "facts": []
}}

Dialogue:
{dialog}"""


def call_gemini_json(prompt: str) -> dict:
    client = _client()
    last_exc: Exception | None = None
    for attempt in range(RETRIES):
        try:
            text = _generate_content(client, prompt)
            return json.loads(_strip_fences(text))
        except (ValueError, KeyError) as exc:
            last_exc = exc
            logger.warning("JSON parse error on attempt %d/%d: %s", attempt + 1, RETRIES, exc)
            if attempt < RETRIES - 1:
                time.sleep(BACKOFFS[attempt])
    raise RuntimeError(f"Gemini JSON call failed after {RETRIES} attempts") from last_exc


def generate_summary_and_facts(utterances: list[dict[str, Any]]) -> tuple[str, list[str]]:
    client = _client()
    dialog = _format_dialog(utterances)
    prompt = _PROMPT.format(dialog=dialog)
    raw = _call_with_validation_retry(client, prompt)
    data = json.loads(_strip_fences(raw))
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
def _generate_content(client: genai.Client, prompt: str) -> str:
    response = client.models.generate_content(
        model=_LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            top_p=1,
        ),
    )
    return response.text.strip()


def _call_with_validation_retry(client: genai.Client, prompt: str) -> str:
    last_exc: Exception | None = None
    for attempt in range(RETRIES):
        try:
            text = _generate_content(client, prompt)
            _validate(text)
            return text
        except ValueError as exc:
            last_exc = exc
            logger.warning(
                "Validation error on attempt %d/%d: %s",
                attempt + 1, RETRIES, exc,
            )
            if attempt < RETRIES - 1:
                time.sleep(BACKOFFS[attempt])
    raise RuntimeError(f"LLM validation failed after {RETRIES} attempts") from last_exc


def _strip_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:])
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned[: cleaned.rstrip().rfind("```")]
    return cleaned.strip()


def _validate(text: str) -> None:
    data = json.loads(_strip_fences(text))
    if "summary" not in data:
        raise ValueError("Missing 'summary' field")
    if not data["summary"]:
        raise ValueError("Empty summary")
    if "facts" not in data:
        raise ValueError("Missing 'facts' field")
    if not isinstance(data["facts"], list):
        raise ValueError("'facts' must be a list")
