#!/usr/bin/env bash

set -euo pipefail

if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

PROJECT_ID=${PROJECT_ID:-tlbrain-prod}
REGION=${REGION:-europe-west1}
VECTOR_SYNC_CHECKER_NAME=${VECTOR_SYNC_CHECKER_NAME:-tlbrain-vector-sync-checker}
VECTOR_SYNC_QUEUE=${VECTOR_SYNC_QUEUE:-tlbrain-vector-sync-queue}
VECTOR_SYNC_SERVICE_NAME=${VECTOR_SYNC_SERVICE_NAME:-tlbrain-vector-sync}
SYNC_INTERVAL_MINUTES=${SYNC_INTERVAL_MINUTES:-15}

VECTOR_SYNC_URL=$(gcloud run services describe "${VECTOR_SYNC_SERVICE_NAME}" \
  --region "${REGION}" \
  --format='value(status.url)')

# Build staging dir with checker source + required core modules
STAGE=$(mktemp -d)
trap "rm -rf ${STAGE}" EXIT

cp services/checker/main.py "${STAGE}/"
cp services/checker/requirements.txt "${STAGE}/"

mkdir -p "${STAGE}/core/google_drive"
cp core/__init__.py "${STAGE}/core/"
cp core/config.py "${STAGE}/core/"
cp core/google_drive/__init__.py "${STAGE}/core/google_drive/"
cp core/google_drive/drive_client.py "${STAGE}/core/google_drive/"

gcloud functions deploy "${VECTOR_SYNC_CHECKER_NAME}" \
  --gen2 \
  --runtime python312 \
  --region "${REGION}" \
  --source "${STAGE}" \
  --entry-point checker \
  --trigger-http \
  --allow-unauthenticated \
  --set-env-vars ROOT_FOLDER_URL="${ROOT_FOLDER_URL}",VECTOR_VECTOR_SYNC_URL="${VECTOR_SYNC_URL}",VECTOR_SYNC_QUEUE="${VECTOR_SYNC_QUEUE}",REGION="${REGION}",GOOGLE_CLOUD_PROJECT="${PROJECT_ID}"

echo "Checker deployed: ${VECTOR_SYNC_CHECKER_NAME}"
