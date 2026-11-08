#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [ -f "${REPO_ROOT}/.env" ]; then
  export $(grep -v '^#' "${REPO_ROOT}/.env" | xargs)
fi
export -n TELEGRAM_BOT_TOKEN TELEGRAM_ALLOWED_USER_IDS 2>/dev/null || true

PROJECT_ID=${PROJECT_ID:-tlbrain-prod}
REGION=${REGION:-europe-west1}
VECTOR_SYNC_SERVICE_NAME=${VECTOR_SYNC_SERVICE_NAME:-tlbrain-vector-sync}
VECTOR_SYNC_CHECKER_NAME=${VECTOR_SYNC_CHECKER_NAME:-tlbrain-vector-sync-checker}
VECTOR_SYNC_QUEUE=${VECTOR_SYNC_QUEUE:-tlbrain-vector-sync-queue}
VECTOR_SYNC_MAX_INSTANCES=${VECTOR_SYNC_MAX_INSTANCES:-2}

SYNC_IMAGE="gcr.io/${PROJECT_ID}/${VECTOR_SYNC_SERVICE_NAME}"

echo "--- Building Vector Sync image ---"
docker build --platform linux/amd64 -f "${REPO_ROOT}/infra/docker/Dockerfile.vector_sync" -t "${SYNC_IMAGE}" "${REPO_ROOT}"
bash "${REPO_ROOT}/infra/build/preflight.sh" --image-mode \
  --image "${SYNC_IMAGE}" \
  --service vector_sync \
  --platform linux/amd64
docker push "${SYNC_IMAGE}"

echo "--- Deploying Vector Sync Service ---"

gcloud run deploy "${VECTOR_SYNC_SERVICE_NAME}" \
  --image "${SYNC_IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --timeout 3600 \
  --concurrency 1 \
  --max-instances "${VECTOR_SYNC_MAX_INSTANCES}" \
  --set-env-vars ROOT_FOLDER_URL="${ROOT_FOLDER_URL}",GEMINI_API_KEY="${GEMINI_API_KEY}",QDRANT_URL="${QDRANT_URL}",QDRANT_API_KEY="${QDRANT_API_KEY}",QDRANT_COLLECTION="${QDRANT_COLLECTION}",GOOGLE_REFRESH_TOKEN="${GOOGLE_REFRESH_TOKEN}",GOOGLE_CLIENT_ID="${GOOGLE_CLIENT_ID}",GOOGLE_CLIENT_SECRET="${GOOGLE_CLIENT_SECRET}",REGION="${REGION}",GOOGLE_CLOUD_PROJECT="${PROJECT_ID}"

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

if [[ "${DEPLOY_DEFER_DIALOG_READINESS:-0}" != "1" ]] \
  && MCP_URL=$(gcloud run services describe "${MCP_SERVICE_NAME:-tlbrain-mcp}" \
  --project "${PROJECT_ID}" --region "${REGION}" --format='value(status.url)' 2>/dev/null) \
  && [[ -n "${MCP_URL}" ]]; then
  DIALOG_OPERATIONS_CONFIGURE_VECTOR_SYNC=1 \
    bash "${REPO_ROOT}/infra/deploy/core/helpers/deploy_dialog_operations.sh" "${MCP_URL}" >/dev/null
fi

echo
echo "Vector Sync: POST ${VECTOR_SYNC_URL}/sync"
