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
VERSION=${VERSION:-}

if [ -z "${VERSION}" ]; then
  echo "Error: VERSION is not set. Add VERSION=v0.10 to your .env file."
  exit 1
fi

DOCKERHUB_USERNAME=inozem

PROJECT_ID=${PROJECT_ID:-tlbrain-prod}
REGION=${REGION:-europe-west1}

MCP_SERVICE_NAME=${MCP_SERVICE_NAME:-tlbrain-mcp}
VECTOR_SYNC_SERVICE_NAME=${VECTOR_SYNC_SERVICE_NAME:-tlbrain-vector-sync}
VECTOR_SYNC_CHECKER_NAME=${VECTOR_SYNC_CHECKER_NAME:-tlbrain-vector-sync-checker}
VECTOR_SYNC_QUEUE=${VECTOR_SYNC_QUEUE:-tlbrain-vector-sync-queue}
CLOUD_TASKS_MAX_CONCURRENT=${CLOUD_TASKS_MAX_CONCURRENT:-2}
SYNC_INTERVAL_MINUTES=${SYNC_INTERVAL_MINUTES:-15}
TLDV_WEBHOOK_FUNCTION_NAME=${TLDV_WEBHOOK_FUNCTION_NAME:-tlbrain-tldv-webhook}
GOOGLE_CLIENT_ID=${GOOGLE_CLIENT_ID:-}
GOOGLE_CLIENT_SECRET=${GOOGLE_CLIENT_SECRET:-}

MCP_IMAGE="${DOCKERHUB_USERNAME}/tlbrain-mcp:${VERSION}"
SYNC_IMAGE="${DOCKERHUB_USERNAME}/tlbrain-vector-sync:${VERSION}"

mask() { local v="$1"; local l=${#v}; if [ $l -le 8 ]; then echo "****"; else echo "${v:0:4}****${v: -4}"; fi; }

# =========================
# Show config
# =========================
echo
echo "Deploy config:"
echo "VERSION=${VERSION}"
echo "MCP_IMAGE=${MCP_IMAGE}"
echo "SYNC_IMAGE=${SYNC_IMAGE}"
echo "PROJECT_ID=${PROJECT_ID}"
echo "REGION=${REGION}"
echo "MCP_SERVICE_NAME=${MCP_SERVICE_NAME}"
echo "VECTOR_SYNC_SERVICE_NAME=${VECTOR_SYNC_SERVICE_NAME}"
echo "VECTOR_SYNC_CHECKER_NAME=${VECTOR_SYNC_CHECKER_NAME}"
echo "VECTOR_SYNC_QUEUE=${VECTOR_SYNC_QUEUE}"
echo "SYNC_INTERVAL_MINUTES=${SYNC_INTERVAL_MINUTES}"
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
# Deploy MCP
# =========================
gcloud run deploy "${MCP_SERVICE_NAME}" \
  --image "${MCP_IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars ROOT_FOLDER_URL="${ROOT_FOLDER_URL}",GEMINI_API_KEY="${GEMINI_API_KEY}",QDRANT_URL="${QDRANT_URL}",QDRANT_API_KEY="${QDRANT_API_KEY}",QDRANT_COLLECTION="${QDRANT_COLLECTION}",RETRIEVAL_TOP_K="${RETRIEVAL_TOP_K}",RETRIEVAL_SCORE_THRESHOLD="${RETRIEVAL_SCORE_THRESHOLD}",ALLOWED_EMAIL="${ALLOWED_EMAIL:-}"

# =========================
# Deploy Sync
# =========================
gcloud run deploy "${VECTOR_SYNC_SERVICE_NAME}" \
  --image "${SYNC_IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars ROOT_FOLDER_URL="${ROOT_FOLDER_URL}",GEMINI_API_KEY="${GEMINI_API_KEY}",QDRANT_URL="${QDRANT_URL}",QDRANT_API_KEY="${QDRANT_API_KEY}",QDRANT_COLLECTION="${QDRANT_COLLECTION}"

# =========================
# Get Sync URL
# =========================
VECTOR_SYNC_URL=$(gcloud run services describe "${VECTOR_SYNC_SERVICE_NAME}" \
  --region "${REGION}" \
  --format='value(status.url)')

# =========================
# Create Cloud Tasks queue
# =========================
echo "Creating Cloud Tasks queue..."
gcloud tasks queues create "${VECTOR_SYNC_QUEUE}" \
  --location="${REGION}" \
  --max-concurrent-dispatches="${CLOUD_TASKS_MAX_CONCURRENT}" \
  2>/dev/null || \
gcloud tasks queues update "${VECTOR_SYNC_QUEUE}" \
  --location="${REGION}" \
  --max-concurrent-dispatches="${CLOUD_TASKS_MAX_CONCURRENT}"

# =========================
# Deploy Checker (Cloud Function)
# =========================
bash infra/deploy/deploy_vector_sync_checker.sh

# =========================
# Deploy TL;DV Connector
# =========================
SKIP_CONFIRM=1 bash infra/deploy/connectors/deploy_tldv.sh

# =========================
# Get Checker URL
# =========================
CHECKER_URL=$(gcloud functions describe "${VECTOR_SYNC_CHECKER_NAME}" \
  --region "${REGION}" \
  --format='value(serviceConfig.uri)')

# =========================
# Create Cloud Scheduler job
# =========================
echo "Creating Cloud Scheduler job..."
gcloud scheduler jobs create http "${VECTOR_SYNC_CHECKER_NAME}-schedule" \
  --location="${REGION}" \
  --schedule="every ${SYNC_INTERVAL_MINUTES} minutes" \
  --uri="${CHECKER_URL}" \
  --http-method=POST \
  2>/dev/null || \
gcloud scheduler jobs update http "${VECTOR_SYNC_CHECKER_NAME}-schedule" \
  --location="${REGION}" \
  --schedule="every ${SYNC_INTERVAL_MINUTES} minutes" \
  --uri="${CHECKER_URL}" \
  --http-method=POST

# =========================
# Final URLs
# =========================
MCP_URL=$(gcloud run services describe "${MCP_SERVICE_NAME}" \
  --region "${REGION}" \
  --format='value(status.url)')

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
