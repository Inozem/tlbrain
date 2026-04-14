from services.mcp.app.mcp.schemas import JSONRPCRequest, JSONRPCResponse
from services.mcp.app.mcp.tools import (
    build_mcp_content,
    build_jsonrpc_result
)
from core.domain.mock_data import get_mock_segments

async def handle_mcp_request(request_dict: dict) -> dict:
    try:
        request = JSONRPCRequest(**request_dict)
    except Exception as e:
        return JSONRPCResponse(
            id=None,
            error={
                "code": -32600,
                "message": "Invalid Request",
                "details": str(e)
            }
        ).model_dump(exclude_none=True)

    method = request.method

    if method == "initialize":
        return handle_initialize(request)

    elif method == "tools/list":
        return handle_tools_list(request)

    elif method == "tools/call":
        return handle_tools_call(request)

    else:
        return JSONRPCResponse(
            id=request.id,
            error={
                "code": -32601,
                "message": "Method not found"
            }
        ).model_dump(exclude_none=True)

def handle_initialize(request: JSONRPCRequest) -> dict:
    return JSONRPCResponse(
        id=request.id,
        result={
            "status": "ok"
        }
    ).model_dump(exclude_none=True)

def handle_tools_call(request: JSONRPCRequest) -> dict:
    params = request.params or {}
    tool_name = params.get("name")
    arguments = params.get("arguments", {})

    if tool_name != "query":
        return JSONRPCResponse(
            id=request.id,
            error={
                "code": -32602,
                "message": "Invalid tool"
            }
        ).model_dump(exclude_none=True)

    segments = get_mock_segments()

    tlbrain_payload = {
        "segments": segments,
        "meta": {
            "truncated": False,
            "total_matches": len(segments),
            "returned_segments": len(segments),
            "limit_reason": None,
            "suggestion": None
        }
    }

    content = build_mcp_content(tlbrain_payload)

    return build_jsonrpc_result(
        request.id,
        content
    )
