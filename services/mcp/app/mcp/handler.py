from services.mcp.app.mcp.schemas import (
    JSONRPCRequest,
    JSONRPCResponse,
    TLBrainMeta,
    TLBrainPayload,
)
from services.mcp.app.mcp.tools import (
    build_mcp_content,
    build_jsonrpc_result,
)
from core.domain.mock_data import query_handler


async def handle_mcp_request(request_dict: dict) -> dict:
    try:
        request = JSONRPCRequest(**request_dict)
    except Exception as e:
        return JSONRPCResponse(
            id=None,
            error={
                "code": -32600,
                "message": "Invalid Request",
                "details": str(e),
            },
        ).model_dump(exclude_none=True)

    method = request.method

    if method == "initialize":
        return handle_initialize(request)

    if method == "tools/list":
        return handle_tools_list(request)

    if method == "tools/call":
        return handle_tools_call(request)

    return JSONRPCResponse(
        id=request.id,
        error={
            "code": -32601,
            "message": "Method not found",
        },
    ).model_dump(exclude_none=True)


def handle_initialize(request: JSONRPCRequest) -> dict:
    return build_jsonrpc_result(
        request.id,
        {
            "status": "ok",
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
                    "input_schema": {
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

    if tool_name != "query":
        return JSONRPCResponse(
            id=request.id,
            error={
                "code": -32602,
                "message": "Invalid tool",
            },
        ).model_dump(exclude_none=True)

    meta = TLBrainMeta(
        truncated=False,
        total_matches=0,
        returned_segments=0,
        limit_reason="no_results",
        suggestion=None,
    )

    tlbrain_payload = TLBrainPayload(
        segments=[],
        meta=meta,
    ).model_dump(exclude_none=True)

    content = build_mcp_content(tlbrain_payload)

    return build_jsonrpc_result(request.id, content)
