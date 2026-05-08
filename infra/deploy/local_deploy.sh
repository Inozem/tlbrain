#!/usr/bin/env bash

set -euo pipefail

# =========================
# Load .env if exists
# =========================
if [ -f .env ]; then
  echo "Loading .env file..."
  export $(grep -v '^#' .env | xargs)
fi

# =========================
# Defaults
# =========================
PROJECT_ID=${PROJECT_ID:-tlbrain-prod}
REGION=${REGION:-europe-west1}

MCP_SERVICE_NAME=${MCP_SERVICE_NAME:-tlbrain-mcp}
VECTOR_SYNC_SERVICE_NAME=${VECTOR_SYNC_SERVICE_NAME:-tlbrain-vector-sync}
VECTOR_SYNC_CHECKER_NAME=${VECTOR_SYNC_CHECKER_NAME:-tlbrain-vector-sync-checker}
VECTOR_SYNC_QUEUE=${VECTOR_SYNC_QUEUE:-tlbrain-vector-sync-queue}
CLOUD_TASKS_MAX_CONCURRENT=${CLOUD_TASKS_MAX_CONCURRENT:-2}
VECTOR_SYNC_CHECKER_SCHEDULE=${VECTOR_SYNC_CHECKER_SCHEDULE:-"*/15 * * * *"}
TLDV_WEBHOOK_FUNCTION_NAME=${TLDV_WEBHOOK_FUNCTION_NAME:-tlbrain-tldv-webhook}
GOOGLE_CLIENT_ID=${GOOGLE_CLIENT_ID:-}
GOOGLE_CLIENT_SECRET=${GOOGLE_CLIENT_SECRET:-}
GOOGLE_REFRESH_TOKEN=${GOOGLE_REFRESH_TOKEN:-}
GEMINI_API_KEY=${GEMINI_API_KEY:-}
QDRANT_URL=${QDRANT_URL:-}
QDRANT_API_KEY=${QDRANT_API_KEY:-}
TLDV_API_KEY=${TLDV_API_KEY:-}
ROOT_FOLDER_URL=${ROOT_FOLDER_URL:-}
ALLOWED_EMAIL=${ALLOWED_EMAIL:-}

mask() { local v="$1"; local l=${#v}; if [ $l -le 8 ]; then echo "****"; else echo "${v:0:4}****${v: -4}"; fi; }

# =========================
# Show config
# =========================
echo
echo "========================================="
echo "Deploy config"
echo "========================================="
echo "  Project:        ${PROJECT_ID}"
echo "  Region:         ${REGION}"
echo ""
echo "  Root folder:    ${ROOT_FOLDER_URL}"
echo "  Allowed email:  ${ALLOWED_EMAIL}"
echo "  Qdrant URL:     ${QDRANT_URL}"
echo ""
echo "  Gemini key:     $(mask "${GEMINI_API_KEY}")"
echo "  Qdrant key:     $(mask "${QDRANT_API_KEY}")"
echo "  TL;DV key:      $(mask "${TLDV_API_KEY}")"
echo "  Client ID:      $(mask "${GOOGLE_CLIENT_ID}")"
echo "  Client Secret:  $(mask "${GOOGLE_CLIENT_SECRET}")"
if [ -n "${GOOGLE_REFRESH_TOKEN}" ]; then echo "  Refresh token:  $(mask "${GOOGLE_REFRESH_TOKEN}")"; else echo "  Refresh token:  (not set — will be generated on first run)"; fi
echo

# =========================
# Confirm
# =========================
read -p "Continue deploy? (y/n): " CONFIRM

if [ "$CONFIRM" != "y" ]; then
  echo "Cancelled."
  exit 0
fi

# =========================
# Set project
# =========================
gcloud config set project "${PROJECT_ID}"

# =========================
# Enable required APIs
# =========================
echo "Enabling required APIs..."
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  drive.googleapis.com \
  docs.googleapis.com \
  firestore.googleapis.com \
  cloudtasks.googleapis.com \
  cloudfunctions.googleapis.com \
  cloudscheduler.googleapis.com

# =========================
# Deploy core (except MCP)
# =========================
bash infra/deploy/core/local_deploy_vector_sync.sh

# =========================
# Deploy TL;DV Connector
# =========================
SKIP_CONFIRM=1 bash infra/deploy/connectors/local_deploy_tldv.sh

# =========================
# Deploy MCP (last — needs URLs from services above)
# =========================
bash infra/deploy/core/local_deploy_mcp.sh

# =========================
# Final URLs
# =========================
MCP_URL=$(gcloud run services describe "${MCP_SERVICE_NAME}" \
  --region "${REGION}" \
  --format='value(status.url)')

VECTOR_SYNC_URL=$(gcloud run services describe "${VECTOR_SYNC_SERVICE_NAME}" \
  --region "${REGION}" \
  --format='value(status.url)')

CHECKER_URL=$(gcloud functions describe "${VECTOR_SYNC_CHECKER_NAME}" \
  --region "${REGION}" \
  --format='value(serviceConfig.uri)')

WEBHOOK_URL=$(gcloud functions describe "${TLDV_WEBHOOK_FUNCTION_NAME}" \
  --region "${REGION}" \
  --format='value(serviceConfig.uri)')

echo
echo "========================================="
echo "Deploy complete"
echo "========================================="
echo
echo "┌─ Connect to Claude ──────────────────────────────────────"
echo "│"
echo "│  1. Claude Settings → Integrations → Add Integration:"
echo "│     ${MCP_URL}/mcp"
echo "│"
echo "│  2. When prompted for OAuth credentials:"
echo "│     Client ID:     $(mask "${GOOGLE_CLIENT_ID}")"
echo "│     Client Secret: $(mask "${GOOGLE_CLIENT_SECRET}")"
echo "│"
echo "└──────────────────────────────────────────────────────────"
echo
echo "┌─ TL;DV Webhook ──────────────────────────────────────────"
echo "│"
echo "│  TL;DV Settings → Webhooks → Add:"
echo "│     POST ${WEBHOOK_URL}"
echo "│"
echo "└──────────────────────────────────────────────────────────"
echo
echo "Internal:"
echo "  Sync trigger:  POST ${VECTOR_SYNC_URL}/sync"
echo "  Checker:       ${CHECKER_URL}"
echo
echo "Grant Drive access to sync service account:"
gcloud run services describe "${VECTOR_SYNC_SERVICE_NAME}" \
  --region "${REGION}" \
  --format='value(spec.template.spec.serviceAccountName)'
