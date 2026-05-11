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
TLDV_IMPORT_SERVICE_NAME=${TLDV_IMPORT_SERVICE_NAME:-tlbrain-tldv-import}
TLDV_IMPORT_QUEUE=${TLDV_IMPORT_QUEUE:-tlbrain-tldv-import-queue}

echo "--- Deploying Sync Checker ---"

VECTOR_SYNC_URL=$(gcloud run services describe "${VECTOR_SYNC_SERVICE_NAME}" \
  --region "${REGION}" \
  --format='value(status.url)')

TLDV_IMPORT_SERVICE_URL=$(gcloud run services describe "${TLDV_IMPORT_SERVICE_NAME}" \
  --region "${REGION}" \
  --format='value(status.url)' 2>/dev/null || echo "")

STAGE=$(mktemp -d)
trap "rm -rf ${STAGE}" EXIT

cp "${REPO_ROOT}/services/sync_checker/main.py" "${STAGE}/"
cat "${REPO_ROOT}/services/sync_checker/requirements.txt" \
    "${REPO_ROOT}/core/google_drive/requirements.txt" \
    "${REPO_ROOT}/core/utils/requirements.txt" \
    "${REPO_ROOT}/core/qdrant/requirements.txt" > "${STAGE}/requirements.txt"

mkdir -p "${STAGE}/core/google_drive"
mkdir -p "${STAGE}/core/utils"
mkdir -p "${STAGE}/core/qdrant"
cp "${REPO_ROOT}/core/__init__.py" "${STAGE}/core/"
cp "${REPO_ROOT}/core/config.py" "${STAGE}/core/"
cp "${REPO_ROOT}/core/google_drive/__init__.py" "${STAGE}/core/google_drive/"
cp "${REPO_ROOT}/core/google_drive/drive_client.py" "${STAGE}/core/google_drive/"
cp "${REPO_ROOT}/core/google_drive/firestore.py" "${STAGE}/core/google_drive/"
cp "${REPO_ROOT}/core/utils/__init__.py" "${STAGE}/core/utils/"
cp "${REPO_ROOT}/core/utils/tasks.py" "${STAGE}/core/utils/"
cp "${REPO_ROOT}/core/qdrant/__init__.py" "${STAGE}/core/qdrant/"
cp "${REPO_ROOT}/core/qdrant/client.py" "${STAGE}/core/qdrant/"
cp "${REPO_ROOT}/core/qdrant/writer.py" "${STAGE}/core/qdrant/"

gcloud functions deploy "${SYNC_CHECKER_NAME}" \
  --gen2 \
  --runtime python312 \
  --region "${REGION}" \
  --source "${STAGE}" \
  --entry-point checker \
  --trigger-http \
  --allow-unauthenticated \
  --set-env-vars ROOT_FOLDER_URL="${ROOT_FOLDER_URL}",VECTOR_SYNC_URL="${VECTOR_SYNC_URL}",VECTOR_SYNC_QUEUE="${VECTOR_SYNC_QUEUE}",TLDV_IMPORT_QUEUE="${TLDV_IMPORT_QUEUE}",TLDV_IMPORT_SERVICE_URL="${TLDV_IMPORT_SERVICE_URL}",REGION="${REGION}",GOOGLE_CLOUD_PROJECT="${PROJECT_ID}"

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
