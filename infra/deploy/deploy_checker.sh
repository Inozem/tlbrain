#!/usr/bin/env bash

set -euo pipefail

if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

PROJECT_ID=${PROJECT_ID:-tlbrain-prod}
REGION=${REGION:-europe-west1}
CHECKER_FUNCTION_NAME=${CHECKER_FUNCTION_NAME:-tlbrain-sync-checker}
CLOUD_TASKS_QUEUE=${CLOUD_TASKS_QUEUE:-tlbrain-sync-queue}
SYNC_SERVICE_NAME=${SYNC_SERVICE_NAME:-tlbrain-sync}
SYNC_INTERVAL_MINUTES=${SYNC_INTERVAL_MINUTES:-15}

SYNC_URL=$(gcloud run services describe "${SYNC_SERVICE_NAME}" \
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

gcloud functions deploy "${CHECKER_FUNCTION_NAME}" \
  --gen2 \
  --runtime python312 \
  --region "${REGION}" \
  --source "${STAGE}" \
  --entry-point checker \
  --trigger-http \
  --allow-unauthenticated \
  --set-env-vars ROOT_FOLDER_URL="${ROOT_FOLDER_URL}",SYNC_URL="${SYNC_URL}",CLOUD_TASKS_QUEUE="${CLOUD_TASKS_QUEUE}",REGION="${REGION}",GOOGLE_CLOUD_PROJECT="${PROJECT_ID}"

echo "Checker deployed: ${CHECKER_FUNCTION_NAME}"
