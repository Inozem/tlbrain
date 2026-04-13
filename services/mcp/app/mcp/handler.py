async def handle_mcp_request(request: dict) -> dict:

    return {
        "jsonrpc": "2.0",
        "id": request.get("id"),
        "result": {}
    }
