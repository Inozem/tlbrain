#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [ -f "${REPO_ROOT}/.env" ]; then
  export $(grep -v '^#' "${REPO_ROOT}/.env" | xargs)
fi

PROJECT_ID=${PROJECT_ID:-tlbrain-prod}
REGION=${REGION:-europe-west1}
VECTOR_SYNC_SERVICE_NAME=${VECTOR_SYNC_SERVICE_NAME:-tlbrain-vector-sync}
VECTOR_SYNC_CHECKER_NAME=${VECTOR_SYNC_CHECKER_NAME:-tlbrain-vector-sync-checker}
VECTOR_SYNC_QUEUE=${VECTOR_SYNC_QUEUE:-tlbrain-vector-sync-queue}
VECTOR_SYNC_MAX_INSTANCES=${VECTOR_SYNC_MAX_INSTANCES:-2}

SYNC_IMAGE="inozem/tlbrain-vector-sync:${VERSION}"

echo "--- Deploying Vector Sync Service ---"

gcloud run deploy "${VECTOR_SYNC_SERVICE_NAME}" \
  --image "${SYNC_IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --timeout 3600 \
  --max-instances "${VECTOR_SYNC_MAX_INSTANCES}" \
  --set-env-vars ROOT_FOLDER_URL="${ROOT_FOLDER_URL}",GEMINI_API_KEY="${GEMINI_API_KEY}",QDRANT_URL="${QDRANT_URL}",QDRANT_API_KEY="${QDRANT_API_KEY}",QDRANT_COLLECTION="${QDRANT_COLLECTION}",GOOGLE_REFRESH_TOKEN="${GOOGLE_REFRESH_TOKEN}",GOOGLE_CLIENT_ID="${GOOGLE_CLIENT_ID}",GOOGLE_CLIENT_SECRET="${GOOGLE_CLIENT_SECRET}"

VECTOR_SYNC_URL=$(gcloud run services describe "${VECTOR_SYNC_SERVICE_NAME}" \
  --region "${REGION}" \
  --format='value(status.url)')

echo "Creating Cloud Tasks queue..."
gcloud tasks queues create "${VECTOR_SYNC_QUEUE}" \
  --location="${REGION}" \
  --max-concurrent-dispatches="${VECTOR_SYNC_MAX_INSTANCES}" \
  2>/dev/null || \
gcloud tasks queues update "${VECTOR_SYNC_QUEUE}" \
  --location="${REGION}" \
  --max-concurrent-dispatches="${VECTOR_SYNC_MAX_INSTANCES}"

echo
echo "Vector Sync: POST ${VECTOR_SYNC_URL}/sync"
