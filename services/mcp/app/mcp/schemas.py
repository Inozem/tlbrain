from typing import Any, Optional
from pydantic import BaseModel


ADD_DIALOG_INPUT_SCHEMA = {
    "type": "object",
    "required": ["provider", "utterances"],
    "additionalProperties": False,
    "properties": {
        "provider": {"type": "string", "minLength": 1},
        "utterances": {
            "type": "array", "minItems": 1,
            "items": {
                "type": "object",
                "required": ["speaker", "text"],
                "additionalProperties": False,
                "properties": {
                    "speaker": {"type": "string", "minLength": 1},
                    "text": {"type": "string", "minLength": 1},
                },
            },
        },
        "title": {"type": "string", "minLength": 1},
        "dialog_date": {"type": "string", "format": "date"},
        "folder_path": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
    },
}


def validate_add_dialog_schema(arguments: Any) -> None:
    """Apply the advertised add_dialog JSON schema's structural constraints."""
    def validate(value: Any, schema: dict[str, Any]) -> None:
        kind = schema["type"]
        if kind == "object":
            if not isinstance(value, dict):
                raise ValueError("expected object")
            if any(key not in schema["properties"] for key in value):
                raise ValueError("unexpected property")
            if any(key not in value for key in schema.get("required", ())):
                raise ValueError("missing property")
            for key, child in schema["properties"].items():
                if key in value:
                    validate(value[key], child)
        elif kind == "array":
            if not isinstance(value, list) or len(value) < schema.get("minItems", 0):
                raise ValueError("expected nonempty array")
            for item in value:
                validate(item, schema["items"])
        elif kind == "string":
            if not isinstance(value, str) or len(value) < schema.get("minLength", 0):
                raise ValueError("expected nonempty string")
            if schema.get("format") == "date":
                from datetime import date
                try:
                    date.fromisoformat(value)
                except ValueError:
                    raise ValueError("expected date") from None
    validate(arguments, ADD_DIALOG_INPUT_SCHEMA)


# JSON-RPC layer
class JSONRPCRequest(BaseModel):
    jsonrpc: str
    id: int | str | None = None
    method: str
    params: Optional[dict[str, Any]] = None


class JSONRPCResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: Optional[int | str]
    result: Optional[Any] = None
    error: Optional[dict[str, Any]] = None


# TLBrain layer
class TLBrainMeta(BaseModel):
    truncated: bool
    total_matches: int
    returned_segments: int
    limit_reason: Optional[str] = None
    suggestion: Optional[str] = None
    other_matches: Optional[list[dict[str, Any]]] = None


class TLBrainPayload(BaseModel):
    segments: list[dict[str, Any]]
    meta: TLBrainMeta
