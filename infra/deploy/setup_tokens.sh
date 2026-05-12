#!/usr/bin/env bash
# One-time setup: authorize Google Drive access and save refresh token to .env
# Requires: docker, curl

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"

AUTH_URI="https://accounts.google.com/o/oauth2/v2/auth"
REDIRECT_URI="http://localhost:8085"
SCOPE="https://www.googleapis.com/auth/drive https://www.googleapis.com/auth/documents"
OAUTH_PORT=8085
IMAGE="tlbrain-setup-tokens"
CONTAINER="tlbrain-setup-tokens"

# Load .env
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

# ─── Build image ──────────────────────────────────────────────────────────────

echo "Building OAuth setup container..."
docker build -q \
  -f "${REPO_ROOT}/infra/docker/Dockerfile.setup_tokens" \
  -t "$IMAGE" \
  "$REPO_ROOT" > /dev/null

# ─── Start container ──────────────────────────────────────────────────────────

# Clean up any leftover container
docker rm -f "$CONTAINER" 2>/dev/null || true

docker run -d \
  --name "$CONTAINER" \
  -p "${OAUTH_PORT}:${OAUTH_PORT}" \
  -e GOOGLE_CLIENT_ID="$CLIENT_ID" \
  -e GOOGLE_CLIENT_SECRET="$CLIENT_SECRET" \
  "$IMAGE"

# Wait for server to be ready
echo "Waiting for server to start..."
READY=0
for i in $(seq 1 15); do
  if docker logs "$CONTAINER" 2>/dev/null | grep -q "READY"; then READY=1; break; fi
  sleep 1
done

if [ "$READY" -eq 0 ]; then
  echo "ERROR: Container failed to start. Logs:"
  docker logs "$CONTAINER" 2>&1
  docker rm -f "$CONTAINER" > /dev/null 2>&1 || true
  exit 1
fi

# ─── Open browser ─────────────────────────────────────────────────────────────

SCOPE_ENC="${SCOPE// /%20}"
AUTH_URL="${AUTH_URI}?client_id=${CLIENT_ID}&redirect_uri=${REDIRECT_URI}&response_type=code&scope=${SCOPE_ENC}&access_type=offline&prompt=consent"

echo "Opening browser for Google authorization..."
if command -v powershell.exe > /dev/null 2>&1; then
  # Windows (Git Bash)
  powershell.exe -NoProfile -Command "Start-Process '$AUTH_URL'" 2>/dev/null || true
elif command -v xdg-open > /dev/null 2>&1; then
  xdg-open "$AUTH_URL" 2>/dev/null || true
elif command -v open > /dev/null 2>&1; then
  open "$AUTH_URL" 2>/dev/null || true
else
  echo "Could not open browser automatically. Open this URL manually:"
  echo "  $AUTH_URL"
fi

echo "Waiting for authorization (up to 2 min)..."

# ─── Wait for token ───────────────────────────────────────────────────────────

REFRESH_TOKEN=""
for i in $(seq 1 120); do
  LOGS=$(docker logs "$CONTAINER" 2>/dev/null)
  if echo "$LOGS" | grep -q "^TOKEN:"; then
    REFRESH_TOKEN=$(echo "$LOGS" | grep "^TOKEN:" | tail -1 | cut -d: -f2-)
    break
  fi
  if echo "$LOGS" | grep -q "^ERROR:"; then
    echo "$LOGS" >&2
    docker rm -f "$CONTAINER" > /dev/null
    exit 1
  fi
  sleep 1
done

docker rm -f "$CONTAINER" > /dev/null

# ─── Save to .env ─────────────────────────────────────────────────────────────

if [ -z "$REFRESH_TOKEN" ]; then
  echo ""
  echo "ERROR: Token was not received. Try again."
  exit 1
fi

# Remove old value if present, then append
if [ -f "$ENV_FILE" ]; then
  sed -i '/^GOOGLE_REFRESH_TOKEN=/d' "$ENV_FILE"
fi
echo "GOOGLE_REFRESH_TOKEN=${REFRESH_TOKEN}" >> "$ENV_FILE"

echo ""
echo "GOOGLE_REFRESH_TOKEN saved to .env"
