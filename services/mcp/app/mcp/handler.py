from app.mcp.schemas import JSONRPCRequest, JSONRPCResponse


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
    return JSONRPCResponse(
        id=request.id,
        result={}
    ).model_dump(exclude_none=True)
