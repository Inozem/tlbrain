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
  --allow-unauthenticated

# =========================
# Deploy Sync
# =========================
gcloud run deploy "${SYNC_SERVICE_NAME}" \
  --image "${SYNC_IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars ROOT_FOLDER_URL="${ROOT_FOLDER_URL}"

# =========================
# Final URLs
# =========================
echo
echo "MCP URL:"
gcloud run services describe "${MCP_SERVICE_NAME}" \
  --region "${REGION}" \
  --format='value(status.url)'

echo
echo "SYNC URL:"
gcloud run services describe "${SYNC_SERVICE_NAME}" \
  --region "${REGION}" \
  --format='value(status.url)'
