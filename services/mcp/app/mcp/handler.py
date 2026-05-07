import logging
import time

from services.mcp.app.mcp.schemas import (
    JSONRPCRequest,
    JSONRPCResponse,
    TLBrainPayload,
)
from services.mcp.app.mcp.tools import (
    build_mcp_content,
    build_jsonrpc_result,
)
from core.retrieval.run import run_retrieval
from core.retrieval.transcripts import get_transcripts
from core.retrieval.clients import list_clients

logger = logging.getLogger(__name__)


def build_jsonrpc_error(
    request_id,
    code: int,
    message: str,
    details: str | None = None,
) -> dict:
    error = {
        "code": code,
        "message": message,
    }

    if details is not None:
        error["details"] = details

    return JSONRPCResponse(
        id=request_id,
        error=error,
    ).model_dump(exclude_none=True)


async def handle_mcp_request(request_dict: dict) -> dict:
    try:
        request = JSONRPCRequest(**request_dict)
    except Exception as e:
        return build_jsonrpc_error(
            request_id=None,
            code=-32600,
            message="Invalid Request",
            details=str(e),
        )

    method = request.method

    if method == "initialize":
        return handle_initialize(request)

    if method == "notifications/initialized":
        return {}

    if method == "tools/list":
        return handle_tools_list(request)

    if method == "tools/call":
        return handle_tools_call(request)

    return build_jsonrpc_error(
        request_id=request.id,
        code=-32601,
        message="Method not found",
    )


def handle_initialize(request: JSONRPCRequest) -> dict:
    return build_jsonrpc_result(
        request.id,
        {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": "tlbrain",
                "version": "1.0.0",
            },
        },
    )


def handle_tools_list(request: JSONRPCRequest) -> dict:
    return build_jsonrpc_result(
        request.id,
        {
            "tools": [
                {
                    "name": "query",
                    "description": "Search through client conversation transcripts. Always translate the query to English before calling this tool — the index is stored in English and translation ensures the best retrieval quality.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query in English",
                            },
                            "date_from": {
                                "type": "string",
                                "description": "ISO date, optional",
                            },
                            "date_to": {
                                "type": "string",
                                "description": "ISO date, optional",
                            },
                            "client_name": {
                                "type": "string",
                                "description": "Client name filter",
                            },
                        },
                        "required": ["query"],
                    },
                },
                {
                    "name": "get_transcript",
                    "description": "Retrieve full conversation transcripts without semantic search. Use when you need the complete text of a specific dialog or the most recent dialogs for a client.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "doc_id": {
                                "type": "string",
                                "description": "Specific document ID. If provided, all other params are ignored.",
                            },
                            "client_name": {
                                "type": "string",
                                "description": "Filter by client name",
                            },
                            "date_from": {
                                "type": "string",
                                "description": "ISO date, optional",
                            },
                            "date_to": {
                                "type": "string",
                                "description": "ISO date, optional",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Max number of most recent transcripts to return (default: 1)",
                            },
                        },
                    },
                },
                {
                    "name": "list_clients",
                    "description": "List all clients in the knowledge base with their dialog count and last dialog date. Call this first to discover available clients before querying. If the response contains a 'suggestion' field, present it to the user.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                    },
                },
            ]
        },
    )


def handle_tools_call(request: JSONRPCRequest) -> dict:
    params = request.params or {}
    tool_name = params.get("name")
    arguments = params.get("arguments", {})

    if tool_name == "query":
        return _handle_query(request, arguments)

    if tool_name == "get_transcript":
        return _handle_get_transcript(request, arguments)

    if tool_name == "list_clients":
        return _handle_list_clients(request)

    return build_jsonrpc_error(
        request_id=request.id,
        code=-32602,
        message="Invalid tool",
    )


def _handle_query(request: JSONRPCRequest, arguments: dict) -> dict:
    query = arguments.get("query", "")
    client_name = arguments.get("client_name") or None
    date_from = arguments.get("date_from") or None
    date_to = arguments.get("date_to") or None

    t0 = time.monotonic()
    try:
        segments, meta = run_retrieval(
            query=query,
            client_name=client_name,
            date_from=date_from,
            date_to=date_to,
        )
    except Exception as e:
        return build_jsonrpc_error(
            request_id=request.id,
            code=-32603,
            message="Retrieval failed",
            details=str(e),
        )

    latency_ms = int((time.monotonic() - t0) * 1000)
    docs_returned = len({s["doc_id"] for s in segments})
    logger.info(
        "tool call: query",
        extra={
            "tool": "query",
            "query": query,
            "client_name": client_name,
            "hits_total": meta.get("total_matches", 0),
            "docs_returned": docs_returned,
            "segments_returned": meta.get("returned_segments", 0),
            "truncated": meta.get("truncated", False),
            "latency_ms": latency_ms,
        },
    )

    content = build_mcp_content(TLBrainPayload(segments=segments, meta=meta).model_dump(exclude_none=True))
    return build_jsonrpc_result(request.id, content)


def _handle_get_transcript(request: JSONRPCRequest, arguments: dict) -> dict:
    doc_id = arguments.get("doc_id") or None
    client_name = arguments.get("client_name") or None
    date_from = arguments.get("date_from") or None
    date_to = arguments.get("date_to") or None
    limit = arguments.get("limit") or 1

    t0 = time.monotonic()
    try:
        segments, meta = get_transcripts(
            doc_id=doc_id,
            client_name=client_name,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
        )
    except Exception as e:
        return build_jsonrpc_error(
            request_id=request.id,
            code=-32603,
            message="Transcript retrieval failed",
            details=str(e),
        )

    latency_ms = int((time.monotonic() - t0) * 1000)
    docs_returned = len({s["doc_id"] for s in segments})
    logger.info(
        "tool call: get_transcript",
        extra={
            "tool": "get_transcript",
            "client_name": client_name,
            "hits_total": meta.get("total_matches", 0),
            "docs_returned": docs_returned,
            "segments_returned": meta.get("returned_segments", 0),
            "truncated": meta.get("truncated", False),
            "latency_ms": latency_ms,
        },
    )

    content = build_mcp_content(TLBrainPayload(segments=segments, meta=meta).model_dump(exclude_none=True))
    return build_jsonrpc_result(request.id, content)


def _handle_list_clients(request: JSONRPCRequest) -> dict:
    t0 = time.monotonic()
    try:
        clients = list_clients()
    except Exception as e:
        return build_jsonrpc_error(
            request_id=request.id,
            code=-32603,
            message="Failed to list clients",
            details=str(e),
        )

    latency_ms = int((time.monotonic() - t0) * 1000)
    logger.info(
        "tool call: list_clients",
        extra={
            "tool": "list_clients",
            "docs_returned": len(clients),
            "latency_ms": latency_ms,
        },
    )

    payload: dict = {"clients": clients}
    unassigned = next((c for c in clients if c["client_name"] == "_unassigned"), None)
    if unassigned and unassigned.get("dialog_count", 0) > 0:
        payload["suggestion"] = (
            f"{unassigned['dialog_count']} transcript(s) are unassigned. "
            f"Ask the user to review and assign them to a client. "
            f"Show each one using get_transcript(doc_id='...') and move it using move_transcript(doc_id='...', new_client_name='...')."
        )
    content = build_mcp_content(payload)
    return build_jsonrpc_result(request.id, content)
