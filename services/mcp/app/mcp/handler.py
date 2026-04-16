from services.mcp.app.mcp.schemas import (
    JSONRPCRequest,
    JSONRPCResponse,
    TLBrainPayload,
)
from services.mcp.app.mcp.tools import (
    build_mcp_content,
    build_jsonrpc_result,
)
from core.domain.mock_data import query_handler


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
                    "description": "Search through client conversation transcripts",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query",
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
                }
            ]
        },
    )


def handle_tools_call(request: JSONRPCRequest) -> dict:
    params = request.params or {}
    tool_name = params.get("name")
    arguments = params.get("arguments", {})
    query = arguments.get("query", "")

    if tool_name != "query":
        return build_jsonrpc_error(
            request_id=request.id,
            code=-32602,
            message="Invalid tool",
        )

    segments, meta = query_handler(query)

    tlbrain_payload = TLBrainPayload(
        segments=segments,
        meta=meta,
    ).model_dump(exclude_none=True)

    content = build_mcp_content(tlbrain_payload)

    return build_jsonrpc_result(request.id, content)
