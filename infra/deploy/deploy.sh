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
# Enable required APIs
# =========================
echo "Enabling required APIs..."
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  drive.googleapis.com \
  firestore.googleapis.com

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
MCP_URL=$(gcloud run services describe "${MCP_SERVICE_NAME}" \
  --region "${REGION}" \
  --format='value(status.url)')

SYNC_URL=$(gcloud run services describe "${SYNC_SERVICE_NAME}" \
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
echo
echo "Grant Drive access to sync service account:"
gcloud run services describe "${SYNC_SERVICE_NAME}" \
  --region "${REGION}" \
  --format='value(spec.template.spec.serviceAccountName)'
