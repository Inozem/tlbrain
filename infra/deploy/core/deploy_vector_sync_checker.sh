#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [ -f "${REPO_ROOT}/.env" ]; then
  export $(grep -v '^#' "${REPO_ROOT}/.env" | xargs)
fi

PROJECT_ID=${PROJECT_ID:-tlbrain-prod}
REGION=${REGION:-europe-west1}
VECTOR_SYNC_CHECKER_NAME=${VECTOR_SYNC_CHECKER_NAME:-tlbrain-vector-sync-checker}
VECTOR_SYNC_QUEUE=${VECTOR_SYNC_QUEUE:-tlbrain-vector-sync-queue}
VECTOR_SYNC_SERVICE_NAME=${VECTOR_SYNC_SERVICE_NAME:-tlbrain-vector-sync}

echo "--- Deploying Vector Sync Checker ---"

VECTOR_SYNC_URL=$(gcloud run services describe "${VECTOR_SYNC_SERVICE_NAME}" \
  --region "${REGION}" \
  --format='value(status.url)')

STAGE=$(mktemp -d)
trap "rm -rf ${STAGE}" EXIT

cp "${REPO_ROOT}/services/vector_sync_checker/main.py" "${STAGE}/"
cat "${REPO_ROOT}/services/vector_sync_checker/requirements.txt" \
    "${REPO_ROOT}/core/google_drive/requirements.txt" > "${STAGE}/requirements.txt"

mkdir -p "${STAGE}/core/google_drive"
cp "${REPO_ROOT}/core/__init__.py" "${STAGE}/core/"
cp "${REPO_ROOT}/core/config.py" "${STAGE}/core/"
cp "${REPO_ROOT}/core/google_drive/__init__.py" "${STAGE}/core/google_drive/"
cp "${REPO_ROOT}/core/google_drive/drive_client.py" "${STAGE}/core/google_drive/"

gcloud functions deploy "${VECTOR_SYNC_CHECKER_NAME}" \
  --gen2 \
  --runtime python312 \
  --region "${REGION}" \
  --source "${STAGE}" \
  --entry-point checker \
  --trigger-http \
  --allow-unauthenticated \
  --set-env-vars ROOT_FOLDER_URL="${ROOT_FOLDER_URL}",VECTOR_SYNC_URL="${VECTOR_SYNC_URL}",VECTOR_SYNC_QUEUE="${VECTOR_SYNC_QUEUE}",REGION="${REGION}",GOOGLE_CLOUD_PROJECT="${PROJECT_ID}"

CHECKER_URL=$(gcloud functions describe "${VECTOR_SYNC_CHECKER_NAME}" \
  --region "${REGION}" \
  --format='value(serviceConfig.uri)')

echo
echo "Checker: ${CHECKER_URL}"
