"""OAuth callback server. Runs inside Docker container.
Exchanges the code for a refresh token and prints it to stdout as TOKEN:<value>.
"""
import http.server
import os
import urllib.parse
import urllib.request
import urllib.error
import json
import sys

PORT = int(os.environ.get("OAUTH_PORT", "8085"))
TOKEN_URI = "https://oauth2.googleapis.com/token"
REDIRECT_URI = f"http://localhost:{PORT}"

CLIENT_ID = os.environ["GOOGLE_CLIENT_ID"]
CLIENT_SECRET = os.environ["GOOGLE_CLIENT_SECRET"]


def exchange_code(code: str) -> str:
    data = urllib.parse.urlencode({
        "code": code,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code",
    }).encode()
    req = urllib.request.Request(TOKEN_URI, data=data, method="POST")
    with urllib.request.urlopen(req) as resp:
        tokens = json.loads(resp.read())
    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        raise RuntimeError(f"No refresh_token in response: {tokens}")
    return refresh_token


def save_token(refresh_token: str) -> None:
    print(f"TOKEN:{refresh_token}", flush=True)


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)

        if "code" not in qs:
            self._respond(400, b"Missing code parameter.")
            return

        code = qs["code"][0]
        try:
            refresh_token = exchange_code(code)
            save_token(refresh_token)
            self._respond(200, b"<h2>Authorized! You can close this tab.</h2>")
            print("OK: GOOGLE_REFRESH_TOKEN received", flush=True)
        except Exception as e:
            self._respond(500, f"<pre>Error: {e}</pre>".encode())
            print(f"ERROR: {e}", file=sys.stderr, flush=True)
            sys.exit(1)

    def _respond(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


httpd = http.server.HTTPServer(("0.0.0.0", PORT), Handler)
print("READY", flush=True)
httpd.handle_request()
