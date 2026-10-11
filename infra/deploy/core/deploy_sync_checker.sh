#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [ -f "${REPO_ROOT}/.env" ]; then
  export $(grep -v '^#' "${REPO_ROOT}/.env" | xargs)
fi

PROJECT_ID=${PROJECT_ID:-tlbrain-prod}
REGION=${REGION:-europe-west1}
SYNC_CHECKER_NAME=${SYNC_CHECKER_NAME:-tlbrain-sync-checker}
VECTOR_SYNC_QUEUE=${VECTOR_SYNC_QUEUE:-tlbrain-vector-sync-queue}
VECTOR_SYNC_SERVICE_NAME=${VECTOR_SYNC_SERVICE_NAME:-tlbrain-vector-sync}
SYNC_CHECKER_SCHEDULE=${SYNC_CHECKER_SCHEDULE:-"*/15 * * * *"}
echo "--- Deploying Sync Checker ---"

VECTOR_SYNC_URL=$(gcloud run services describe "${VECTOR_SYNC_SERVICE_NAME}" \
  --region "${REGION}" \
  --format='value(status.url)')

ARTIFACT="${REPO_ROOT}/.build/sync_checker"
"${REPO_ROOT}/infra/build/preflight.sh" \
  --service sync_checker \
  --source-root "${REPO_ROOT}" \
  --output "${ARTIFACT}"

gcloud functions deploy "${SYNC_CHECKER_NAME}" \
  --gen2 \
  --runtime python312 \
  --region "${REGION}" \
  --source "${ARTIFACT}" \
  --entry-point checker \
  --trigger-http \
  --allow-unauthenticated \
  --max-instances=1 \
  --set-env-vars ROOT_FOLDER_URL="${ROOT_FOLDER_URL}",VECTOR_SYNC_URL="${VECTOR_SYNC_URL}",VECTOR_SYNC_QUEUE="${VECTOR_SYNC_QUEUE}",REGION="${REGION}",GOOGLE_CLOUD_PROJECT="${PROJECT_ID}",GOOGLE_REFRESH_TOKEN="${GOOGLE_REFRESH_TOKEN}",GOOGLE_CLIENT_ID="${GOOGLE_CLIENT_ID}",GOOGLE_CLIENT_SECRET="${GOOGLE_CLIENT_SECRET}"

CHECKER_URL=$(gcloud functions describe "${SYNC_CHECKER_NAME}" \
  --region "${REGION}" \
  --format='value(serviceConfig.uri)')

echo
echo "Checker: ${CHECKER_URL}"

gcloud scheduler jobs create http "${SYNC_CHECKER_NAME}-schedule" \
  --location="${REGION}" \
  --schedule="${SYNC_CHECKER_SCHEDULE}" \
  --uri="${CHECKER_URL}" \
  --http-method=POST \
  2>/dev/null || \
gcloud scheduler jobs update http "${SYNC_CHECKER_NAME}-schedule" \
  --location="${REGION}" \
  --schedule="${SYNC_CHECKER_SCHEDULE}" \
  --uri="${CHECKER_URL}" \
  --http-method=POST

echo "Scheduler set: ${SYNC_CHECKER_SCHEDULE}"
