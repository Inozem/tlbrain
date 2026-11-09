"""Request-scoped trusted identity and initialized MCP session state."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any, Literal


SessionState = Literal["absent", "active", "invalid"]


@dataclass(frozen=True)
class MCPRequestContext:
    identity: Any
    session_state: SessionState = "absent"
    session_id: str | None = None
    session: Any | None = None


_current: ContextVar[MCPRequestContext | None] = ContextVar("mcp_request_context", default=None)


def get_request_context() -> MCPRequestContext | None:
    """Return context for the currently executing MCP request, if any."""
    return _current.get()


def set_request_context(value: MCPRequestContext) -> Token:
    """Set context and return a token that must be reset in a finally block."""
    return _current.set(value)


def reset_request_context(token: Token) -> None:
    _current.reset(token)
