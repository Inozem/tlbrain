"""Narrow Gemini SDK boundary for Telegram agent model calls."""

from __future__ import annotations

from collections.abc import Mapping

from google.genai import types

from core.integrations.gemini import client as gemini_client


AGENT_MODEL = "gemini-3.5-flash-lite"
_THINKING_LEVELS = {
    "routing": types.ThinkingLevel.MINIMAL,
    "tool_loop": types.ThinkingLevel.MEDIUM,
}


class GeminiAgent:
    """Submit Telegram agent turns to Gemini without interpreting responses."""

    def __init__(self, *, client=None):
        self._client = client if client is not None else gemini_client.make_client(single_attempt=True)

    def generate(self, *, contents, mode: str, tools=None):
        """Make one generation request and return the SDK response unchanged."""
        try:
            thinking_level = _THINKING_LEVELS[mode]
        except (KeyError, TypeError):
            raise ValueError(f"unsupported Gemini agent mode: {mode!r}") from None

        sdk_tools = tools
        if tools and all(isinstance(tool, dict) and "name" in tool for tool in tools):
            sdk_tools = [
                types.Tool(
                    function_declarations=[types.FunctionDeclaration(**tool) for tool in tools]
                )
            ]

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
            tools=sdk_tools,
            tool_config=(types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="ANY")
            ) if mode == "routing" and sdk_tools else None),
        )
        return self._client.models.generate_content(
            model=AGENT_MODEL,
            contents=contents,
            config=config,
        )

    @staticmethod
    def make_function_response(*, call, response):
        """Build a Gemini function response with the SDK-required object payload."""
        if not isinstance(response, Mapping):
            response = {"result": response}
        return types.Part(
            function_response=types.FunctionResponse(
                id=call.id,
                name=call.name,
                response=response,
            )
        )
