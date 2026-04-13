from .schemas import JSONRPCResponse, JSONRPCRequest

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
        ).model_dump()

    response = JSONRPCResponse(
        id=request.id,
        result={}
    )

    return response.model_dump(exclude_none=True)
