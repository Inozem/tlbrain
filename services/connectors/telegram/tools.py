"""Telegram tool catalog and dispatch over the shared operation registry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.operations import registry
from core.operations.write import WriteContext


def _argument_schema(name: str) -> dict[str, Any]:
    """Build the Telegram schema from the shared static operation definition."""
    operation = registry.OPERATIONS[name]
    properties: dict[str, Any] = {}
    required: list[str] = []
    for field_name, field in operation.arguments.items():
        if field.kind in {"path", "utterances"}:
            prop: dict[str, Any] = {"type": "array"}
            if field.kind == "path":
                item_schema = {"type": "string"}
                if name == "add_dialog":
                    item_schema["minLength"] = 1
                prop["items"] = item_schema
            else:
                prop.update({
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["speaker", "text"],
                        "additionalProperties": False,
                        "properties": {
                            "speaker": {"type": "string", "minLength": 1},
                            "text": {"type": "string", "minLength": 1},
                        },
                    },
                })
        else:
            prop = {"type": "integer" if field.kind == "integer" else "string"}
            if name == "add_dialog" and field_name == "provider":
                prop["minLength"] = 1
            elif name == "add_dialog" and field_name == "title":
                prop["minLength"] = 1
            elif name == "add_dialog" and field_name == "dialog_date":
                prop["format"] = "date"
        properties[field_name] = prop
        if field.required:
            required.append(field_name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    if name == "add_dialog":
        schema["additionalProperties"] = False
    return schema


def get_tool_catalog() -> dict[str, Any]:
    """Return an isolated catalog containing precisely the shared operations."""
    return {
        "tools": [
            {
                "name": name,
                "description": f"Run the {name} operation.",
                "parameters_json_schema": _argument_schema(name),
            }
            for name in registry.OPERATIONS
        ]
    }


def _context_metadata(context: Any) -> Mapping[str, Any]:
    return context if isinstance(context, Mapping) else {}


def _strip_unrequested_links(value: Any) -> Any:
    """Remove provider links recursively while preserving result shape and metadata."""
    if isinstance(value, Mapping):
        return {key: _strip_unrequested_links(item) for key, item in value.items()
                if key != "link"}
    if isinstance(value, list):
        return [_strip_unrequested_links(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_strip_unrequested_links(item) for item in value)
    return value


def invoke_tool(
    name: str,
    arguments: Any,
    *,
    context: Any = None,
    execution_context: WriteContext | registry.AddDialogContext | None = None,
    retry_policy: Any = None,
) -> Any:
    """Validate and dispatch once, before resolving trusted context.

    This ordering makes unknown operations and model-supplied capability fields
    fail closed.
    """
    metadata = _context_metadata(context)

    if (name in {"get_transcript", "move_transcript", "rename_transcript"}
            and isinstance(arguments, dict) and "doc_id" not in arguments):
        last_doc_id = metadata.get("last_doc_id")
        if isinstance(last_doc_id, str) and last_doc_id:
            arguments = {**arguments, "doc_id": last_doc_id}

    validated = registry.validate_arguments(name, arguments)

    trusted_context = execution_context
    if trusted_context is None and isinstance(context, (WriteContext, registry.AddDialogContext)):
        trusted_context = context

    if retry_policy is None:
        result = registry.invoke_operation(name, validated, context=trusted_context)
    else:
        result = registry.invoke_operation(
            name, validated, context=trusted_context, retry_policy=retry_policy,
        )
    if metadata.get("include_link") is True:
        return result
    return _strip_unrequested_links(result)
