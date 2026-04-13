from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.mcp.handler import handle_mcp_request

app = FastAPI()


@app.get("/")
async def health():
    return {"status": "ok"}


@app.post("/mcp")
async def mcp_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            content={
                "error": "Invalid JSON body"
            },
            status_code=400
        )

    print("Incoming MCP request:", body)

    response = await handle_mcp_request(body)

    return JSONResponse(content=response)
