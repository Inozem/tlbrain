from typing import Any, Optional
from pydantic import BaseModel


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


class TLBrainPayload(BaseModel):
    segments: list[dict[str, Any]]
    meta: TLBrainMeta
