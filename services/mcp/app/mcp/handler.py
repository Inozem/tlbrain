from services.mcp.app.mcp.schemas import JSONRPCRequest, JSONRPCResponse
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

def handle_tools_list(request: JSONRPCRequest) -> dict:
    return JSONRPCResponse(
        id=request.id,
        result={
            "tools": [
                {
                    "name": "query",
                    "description": "Search through client conversation transcripts",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "date_from": {
                                "type": "string",
                                "description": "ISO date, optional"
                            },
                            "date_to": {
                                "type": "string",
                                "description": "ISO date, optional"
                            },
                            "client_name": {
                                "type": "string",
                                "description": "Client name filter"
                            }
                        },
                        "required": ["query"]
                    }
                }
            ]
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

    # mock response
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

    # MCP wrapper
    return JSONRPCResponse(
        id=request.id,
        result={
            "content": [
                {
                    "type": "json",
                    "json": tlbrain_payload
                }
            ]
        }
    ).model_dump(exclude_none=True)
