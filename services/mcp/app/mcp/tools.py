from services.mcp.app.mcp.schemas import JSONRPCResponse


def build_mcp_content(payload: dict) -> dict:
    return {
        "content": [
            {
                "type": "json",
                "json": payload
            }
        ]
    }


def build_jsonrpc_result(request_id, result: dict) -> dict:
    return JSONRPCResponse(
        id=request_id,
        result=result
    ).model_dump(exclude_none=True)
