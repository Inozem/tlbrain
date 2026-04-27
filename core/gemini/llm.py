import json
import os
import time
from typing import Any

from google import genai
from google.genai import types

_LLM_MODEL = "gemini-2.5-flash"
_RETRIES = 3
_BACKOFFS = [1, 2, 4]

_SUMMARY_PROMPT = """\
You are analyzing a fragment of a dialogue. Write exactly one sentence summarizing what is discussed.
Use speaker names exactly as they appear in the dialogue — do not infer or rename roles.
Always reply in English regardless of the dialogue language.
Reply with plain text only, no JSON, no markdown.

Dialogue:
{dialog}"""

_FACTS_PROMPT = """\
You are analyzing a fragment of a dialogue. Extract up to 10 notable facts mentioned in it.
A fact is any specific, concrete piece of information worth remembering: contacts, interests, industries, \
decisions, numbers, problems, relationships, plans, or anything unusual.
Write each fact as a short noun phrase or brief clause — no attribution, no "X said that", just the fact itself.
Use speaker names only when the person themselves (not their statement) is the fact.
Always reply in English regardless of the dialogue language.
Reply with valid JSON only, no markdown, no explanation.

Required JSON structure:
{{
  "facts": []
}}

Empty list if nothing notable found.

Dialogue:
{dialog}"""

_REQUIRED_FACTS_FIELDS = {"facts"}


def generate_summary(utterances: list[dict[str, Any]]) -> str:
    client = _client()
    dialog = _format_dialog(utterances)
    prompt = _SUMMARY_PROMPT.format(dialog=dialog)
    return _call_with_retry(client, prompt, _validate_summary)


def generate_facts(utterances: list[dict[str, Any]]) -> list[str]:
    client = _client()
    dialog = _format_dialog(utterances)
    prompt = _FACTS_PROMPT.format(dialog=dialog)
    raw = _call_with_retry(client, prompt, _validate_facts)
    return json.loads(_strip_fences(raw))["facts"]


def _client() -> genai.Client:
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def _format_dialog(utterances: list[dict[str, Any]]) -> str:
    return "\n".join(f"{u['speaker']}: {u['text']}" for u in utterances)


def _call_with_retry(
    client: genai.Client,
    prompt: str,
    validate,
) -> str:
    last_exc: Exception | None = None
    for attempt in range(_RETRIES):
        try:
            response = client.models.generate_content(
                model=_LLM_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    top_p=1,
                ),
            )
            text = response.text.strip()
            validate(text)
            return text
        except Exception as exc:
            last_exc = exc
            time.sleep(_BACKOFFS[attempt])
    raise RuntimeError(f"Gemini LLM failed after {_RETRIES} attempts") from last_exc


def _strip_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:])
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned[: cleaned.rstrip().rfind("```")]
    return cleaned.strip()


def _validate_summary(text: str) -> None:
    if not text:
        raise ValueError("Empty summary")


def _validate_facts(text: str) -> None:
    data = json.loads(_strip_fences(text))
    if "facts" not in data:
        raise ValueError("Missing 'facts' field in response")
    if not isinstance(data["facts"], list):
        raise ValueError("'facts' must be a list")
    if len(data["facts"]) > 10:
        raise ValueError(f"Too many facts: {len(data['facts'])}")
