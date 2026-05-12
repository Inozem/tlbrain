#!/usr/bin/env bash
# One-time setup: authorize Google Drive access and save refresh token to .env
# Requires: curl + one of: powershell.exe (Windows) | python3 (macOS/Linux)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"

REDIRECT_URI="http://localhost:8085"
SCOPE="https://www.googleapis.com/auth/drive https://www.googleapis.com/auth/documents"
OAUTH_PORT=8085

# ─── Load .env ────────────────────────────────────────────────────────────────

if [ -f "$ENV_FILE" ]; then
  set -a; source "$ENV_FILE"; set +a
fi

CLIENT_ID="${GOOGLE_CLIENT_ID:-}"
CLIENT_SECRET="${GOOGLE_CLIENT_SECRET:-}"

if [ -z "$CLIENT_ID" ] || [ -z "$CLIENT_SECRET" ]; then
  echo ""
  echo "ERROR: GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET must be set in .env"
  echo ""
  echo "  1. Google Cloud Console → APIs & Services → Credentials"
  echo "  2. Create OAuth 2.0 Client ID → Desktop app"
  echo "  3. Add to .env: GOOGLE_CLIENT_ID=... GOOGLE_CLIENT_SECRET=..."
  echo ""
  exit 1
fi

SCOPE_ENC="${SCOPE// /%20}"
AUTH_URL="https://accounts.google.com/o/oauth2/v2/auth?client_id=${CLIENT_ID}&redirect_uri=${REDIRECT_URI}&response_type=code&scope=${SCOPE_ENC}&access_type=offline&prompt=consent"

# ─── Start local HTTP listener and open browser ───────────────────────────────

echo "Opening browser for Google authorization..."

CODE=""

if command -v powershell.exe > /dev/null 2>&1; then
  echo "Using PowerShell HttpListener (Windows)"
  export _TLBRAIN_AUTH_URL="$AUTH_URL"
  CODE=$(powershell.exe -NoProfile -Command '
    $port = 8085
    $listener = New-Object System.Net.HttpListener
    $listener.Prefixes.Add("http://localhost:$port/")
    $listener.Start()
    Start-Process $env:_TLBRAIN_AUTH_URL
    $code = ""
    while ($code -eq "") {
      $ctx = $listener.GetContext()
      $query = $ctx.Request.Url.Query
      if ($query -match "[?&]code=([^&]+)") {
        $code = $Matches[1]
        $buf = [System.Text.Encoding]::UTF8.GetBytes("<h2>Authorized! You can close this tab.</h2>")
        $ctx.Response.ContentLength64 = $buf.Length
        $ctx.Response.OutputStream.Write($buf, 0, $buf.Length)
      } else {
        $ctx.Response.StatusCode = 204
      }
      $ctx.Response.OutputStream.Close()
    }
    $listener.Stop()
    Write-Output $code
  ' 2>/dev/null | tr -d '\r')

elif command -v python3 > /dev/null 2>&1; then
  echo "Using Python3 http.server (macOS/Linux)"
  if command -v xdg-open > /dev/null 2>&1; then
    xdg-open "$AUTH_URL" 2>/dev/null &
  elif command -v open > /dev/null 2>&1; then
    open "$AUTH_URL" 2>/dev/null &
  else
    echo "Open this URL in your browser:"
    echo "  $AUTH_URL"
  fi
  CODE=$(python3 - "$OAUTH_PORT" <<'PYEOF'
import http.server, urllib.parse, sys

port = int(sys.argv[1])
_code = []

class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        if "code" in qs:
            _code.append(qs["code"][0])
            body = b"<h2>Authorized! You can close this tab.</h2>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(204)
            self.end_headers()
    def log_message(self, *a): pass

httpd = http.server.HTTPServer(("", port), H)
while not _code:
    httpd.handle_request()
print(_code[0])
PYEOF
)

else
  echo "ERROR: Neither powershell.exe nor python3 found. Cannot start OAuth listener."
  exit 1
fi

CODE=$(echo "$CODE" | tr -d '[:space:]')

if [ -z "$CODE" ]; then
  echo "ERROR: No authorization code received."
  exit 1
fi

# ─── Exchange code for refresh token ─────────────────────────────────────────

echo "Exchanging code for refresh token..."

RESPONSE=$(curl -s -X POST "https://oauth2.googleapis.com/token" \
  --data-urlencode "code=${CODE}" \
  --data-urlencode "client_id=${CLIENT_ID}" \
  --data-urlencode "client_secret=${CLIENT_SECRET}" \
  --data-urlencode "redirect_uri=${REDIRECT_URI}" \
  --data-urlencode "grant_type=authorization_code")

REFRESH_TOKEN=$(echo "$RESPONSE" | grep -o '"refresh_token": *"[^"]*"' | grep -o '"[^"]*"$' | tr -d '"' || true)

if [ -z "$REFRESH_TOKEN" ]; then
  echo "ERROR: No refresh_token in response:"
  echo "$RESPONSE"
  exit 1
fi

# ─── Save to .env ─────────────────────────────────────────────────────────────

# sed -i is unreliable on Windows Git Bash — rewrite file manually
if [ -f "$ENV_FILE" ]; then
  TMP_FILE="${ENV_FILE}.tmp"
  grep -v '^GOOGLE_REFRESH_TOKEN=' "$ENV_FILE" > "$TMP_FILE" || true
  mv "$TMP_FILE" "$ENV_FILE"
fi
echo "GOOGLE_REFRESH_TOKEN=${REFRESH_TOKEN}" >> "$ENV_FILE"

if grep -q "^GOOGLE_REFRESH_TOKEN=" "$ENV_FILE"; then
  echo ""
  echo "GOOGLE_REFRESH_TOKEN saved to .env"
else
  echo "ERROR: Failed to write token to .env"
  exit 1
fi
