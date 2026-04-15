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
# Ask missing values
# =========================

if [ -z "${PROJECT_ID:-}" ]; then
  read -p "Enter PROJECT_ID: " PROJECT_ID
fi

if [ -z "${REGION:-}" ]; then
  read -p "Enter REGION [europe-west1]: " REGION
  REGION=${REGION:-europe-west1}
fi

if [ -z "${SERVICE_NAME:-}" ]; then
  read -p "Enter SERVICE_NAME [tlbrain-mcp]: " SERVICE_NAME
  SERVICE_NAME=${SERVICE_NAME:-tlbrain-mcp}
fi

IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# =========================
# Show config
# =========================
echo
echo "Deploy config:"
echo "PROJECT_ID=${PROJECT_ID}"
echo "REGION=${REGION}"
echo "SERVICE_NAME=${SERVICE_NAME}"
echo "IMAGE_NAME=${IMAGE_NAME}"
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
# Deploy
# =========================
gcloud config set project "${PROJECT_ID}"

gcloud builds submit \
  --config infra/cloudbuild.yaml \
  --substitutions _SERVICE_NAME="${SERVICE_NAME}" \
  .

gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE_NAME}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated

# =========================
# Final URL
# =========================
echo
echo "Deploy completed."

gcloud run services describe "${SERVICE_NAME}" \
  --region "${REGION}" \
  --format='value(status.url)'
