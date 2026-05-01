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
SYNC_SERVICE_NAME=${SYNC_SERVICE_NAME:-tlbrain-sync}
CHECKER_FUNCTION_NAME=${CHECKER_FUNCTION_NAME:-tlbrain-sync-checker}
CLOUD_TASKS_QUEUE=${CLOUD_TASKS_QUEUE:-tlbrain-sync-queue}
CLOUD_TASKS_MAX_CONCURRENT=${CLOUD_TASKS_MAX_CONCURRENT:-2}
SYNC_INTERVAL_MINUTES=${SYNC_INTERVAL_MINUTES:-15}

MCP_IMAGE="gcr.io/${PROJECT_ID}/${MCP_SERVICE_NAME}"
SYNC_IMAGE="gcr.io/${PROJECT_ID}/${SYNC_SERVICE_NAME}"

# =========================
# Show config
# =========================
echo
echo "Deploy config:"
echo "PROJECT_ID=${PROJECT_ID}"
echo "REGION=${REGION}"
echo "MCP_SERVICE_NAME=${MCP_SERVICE_NAME}"
echo "SYNC_SERVICE_NAME=${SYNC_SERVICE_NAME}"
echo "CHECKER_FUNCTION_NAME=${CHECKER_FUNCTION_NAME}"
echo "CLOUD_TASKS_QUEUE=${CLOUD_TASKS_QUEUE}"
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
# Build MCP
# =========================
docker build -f infra/docker/Dockerfile.mcp -t "${MCP_IMAGE}" .
docker push "${MCP_IMAGE}"

# =========================
# Build Sync
# =========================
docker build -f infra/docker/Dockerfile.sync -t "${SYNC_IMAGE}" .
docker push "${SYNC_IMAGE}"

# =========================
# Deploy MCP
# =========================
gcloud run deploy "${MCP_SERVICE_NAME}" \
  --image "${MCP_IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars ROOT_FOLDER_URL="${ROOT_FOLDER_URL}",GEMINI_API_KEY="${GEMINI_API_KEY}",QDRANT_URL="${QDRANT_URL}",QDRANT_API_KEY="${QDRANT_API_KEY}",QDRANT_COLLECTION="${QDRANT_COLLECTION}",RETRIEVAL_TOP_K="${RETRIEVAL_TOP_K}",RETRIEVAL_SCORE_THRESHOLD="${RETRIEVAL_SCORE_THRESHOLD}"

# =========================
# Deploy Sync
# =========================
gcloud run deploy "${SYNC_SERVICE_NAME}" \
  --image "${SYNC_IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars ROOT_FOLDER_URL="${ROOT_FOLDER_URL}",GEMINI_API_KEY="${GEMINI_API_KEY}",QDRANT_URL="${QDRANT_URL}",QDRANT_API_KEY="${QDRANT_API_KEY}",QDRANT_COLLECTION="${QDRANT_COLLECTION}"

# =========================
# Get Sync URL
# =========================
SYNC_URL=$(gcloud run services describe "${SYNC_SERVICE_NAME}" \
  --region "${REGION}" \
  --format='value(status.url)')

# =========================
# Create Cloud Tasks queue
# =========================
echo "Creating Cloud Tasks queue..."
gcloud tasks queues create "${CLOUD_TASKS_QUEUE}" \
  --location="${REGION}" \
  --max-concurrent-dispatches="${CLOUD_TASKS_MAX_CONCURRENT}" \
  2>/dev/null || \
gcloud tasks queues update "${CLOUD_TASKS_QUEUE}" \
  --location="${REGION}" \
  --max-concurrent-dispatches="${CLOUD_TASKS_MAX_CONCURRENT}"

# =========================
# Deploy Checker (Cloud Function)
# =========================
gcloud functions deploy "${CHECKER_FUNCTION_NAME}" \
  --gen2 \
  --runtime python312 \
  --region "${REGION}" \
  --source services/checker \
  --entry-point checker \
  --trigger-http \
  --allow-unauthenticated \
  --set-env-vars ROOT_FOLDER_URL="${ROOT_FOLDER_URL}",SYNC_URL="${SYNC_URL}",CLOUD_TASKS_QUEUE="${CLOUD_TASKS_QUEUE}",REGION="${REGION}",GOOGLE_CLOUD_PROJECT="${PROJECT_ID}"

# =========================
# Get Checker URL
# =========================
CHECKER_URL=$(gcloud functions describe "${CHECKER_FUNCTION_NAME}" \
  --region "${REGION}" \
  --format='value(serviceConfig.uri)')

# =========================
# Create Cloud Scheduler job
# =========================
echo "Creating Cloud Scheduler job..."
gcloud scheduler jobs create http "${CHECKER_FUNCTION_NAME}-schedule" \
  --location="${REGION}" \
  --schedule="every ${SYNC_INTERVAL_MINUTES} minutes" \
  --uri="${CHECKER_URL}" \
  --http-method=POST \
  2>/dev/null || \
gcloud scheduler jobs update http "${CHECKER_FUNCTION_NAME}-schedule" \
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

echo
echo "========================================="
echo "Deploy complete"
echo "========================================="
echo
echo "MCP endpoint:   ${MCP_URL}/mcp"
echo "Sync trigger:   POST ${SYNC_URL}/sync"
echo "Sync health:    GET  ${SYNC_URL}/"
echo "Checker:        ${CHECKER_URL}"
echo
echo "Grant Drive access to sync service account:"
gcloud run services describe "${SYNC_SERVICE_NAME}" \
  --region "${REGION}" \
  --format='value(spec.template.spec.serviceAccountName)'
