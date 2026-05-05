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
CLOUD_TASKS_MAX_CONCURRENT=${CLOUD_TASKS_MAX_CONCURRENT:-2}
SYNC_CHECKER_SCHEDULE=${SYNC_CHECKER_SCHEDULE:-"0 4 * * *"}
SYNC_IMAGE="inozem/tlbrain-vector-sync:${VERSION}"

echo "--- Deploying Vector Sync Service ---"

gcloud run deploy "${VECTOR_SYNC_SERVICE_NAME}" \
  --image "${SYNC_IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars ROOT_FOLDER_URL="${ROOT_FOLDER_URL}",GEMINI_API_KEY="${GEMINI_API_KEY}",QDRANT_URL="${QDRANT_URL}",QDRANT_API_KEY="${QDRANT_API_KEY}",QDRANT_COLLECTION="${QDRANT_COLLECTION}"

VECTOR_SYNC_URL=$(gcloud run services describe "${VECTOR_SYNC_SERVICE_NAME}" \
  --region "${REGION}" \
  --format='value(status.url)')

echo "Creating Cloud Tasks queue..."
gcloud tasks queues create "${VECTOR_SYNC_QUEUE}" \
  --location="${REGION}" \
  --max-concurrent-dispatches="${CLOUD_TASKS_MAX_CONCURRENT}" \
  2>/dev/null || \
gcloud tasks queues update "${VECTOR_SYNC_QUEUE}" \
  --location="${REGION}" \
  --max-concurrent-dispatches="${CLOUD_TASKS_MAX_CONCURRENT}"

bash "${SCRIPT_DIR}/deploy_vector_sync_checker.sh"

CHECKER_URL=$(gcloud functions describe "${VECTOR_SYNC_CHECKER_NAME}" \
  --region "${REGION}" \
  --format='value(serviceConfig.uri)')

echo "Creating Cloud Scheduler job..."
gcloud scheduler jobs create http "${VECTOR_SYNC_CHECKER_NAME}-schedule" \
  --location="${REGION}" \
  --schedule="${SYNC_CHECKER_SCHEDULE}" \
  --uri="${CHECKER_URL}" \
  --http-method=POST \
  2>/dev/null || \
gcloud scheduler jobs update http "${VECTOR_SYNC_CHECKER_NAME}-schedule" \
  --location="${REGION}" \
  --schedule="${SYNC_CHECKER_SCHEDULE}" \
  --uri="${CHECKER_URL}" \
  --http-method=POST

echo
echo "Vector Sync: POST ${VECTOR_SYNC_URL}/sync"
echo "Checker:     ${CHECKER_URL}"
