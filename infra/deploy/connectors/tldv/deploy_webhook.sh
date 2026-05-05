#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

if [ -f "${REPO_ROOT}/.env" ]; then
  export $(grep -v '^#' "${REPO_ROOT}/.env" | xargs)
fi

PROJECT_ID=${PROJECT_ID:-tlbrain-prod}
REGION=${REGION:-europe-west1}
TLDV_WEBHOOK_FUNCTION_NAME=${TLDV_WEBHOOK_FUNCTION_NAME:-tlbrain-tldv-webhook}
TLDV_IMPORT_QUEUE=${TLDV_IMPORT_QUEUE:-tlbrain-tldv-import-queue}
TLDV_IMPORT_SERVICE_NAME=${TLDV_IMPORT_SERVICE_NAME:-tlbrain-tldv-import}

echo "--- Deploying TL;DV Webhook Function ---"

IMPORT_SERVICE_URL=$(gcloud run services describe "${TLDV_IMPORT_SERVICE_NAME}" \
  --region "${REGION}" \
  --format='value(status.url)')

STAGE=$(mktemp -d)
trap "rm -rf ${STAGE}" EXIT

cp "${REPO_ROOT}/services/connectors/tldv/webhook/main.py" "${STAGE}/"
cat "${REPO_ROOT}/services/connectors/tldv/webhook/requirements.txt" \
    "${REPO_ROOT}/core/utils/requirements.txt" > "${STAGE}/requirements.txt"

mkdir -p "${STAGE}/core/utils"
cp "${REPO_ROOT}/core/__init__.py" "${STAGE}/core/"
cp "${REPO_ROOT}/core/utils/__init__.py" "${STAGE}/core/utils/"
cp "${REPO_ROOT}/core/utils/tasks.py" "${STAGE}/core/utils/"
cp "${REPO_ROOT}/core/utils/logging.py" "${STAGE}/core/utils/"

gcloud functions deploy "${TLDV_WEBHOOK_FUNCTION_NAME}" \
  --gen2 \
  --runtime python312 \
  --region "${REGION}" \
  --source "${STAGE}" \
  --entry-point tldv_webhook \
  --trigger-http \
  --allow-unauthenticated \
  --set-env-vars TLDV_IMPORT_QUEUE="${TLDV_IMPORT_QUEUE}",TLDV_IMPORT_SERVICE_URL="${IMPORT_SERVICE_URL}",REGION="${REGION}",GOOGLE_CLOUD_PROJECT="${PROJECT_ID}"

WEBHOOK_URL=$(gcloud functions describe "${TLDV_WEBHOOK_FUNCTION_NAME}" \
  --region "${REGION}" \
  --format='value(serviceConfig.uri)')

echo
echo "Webhook deployed: ${WEBHOOK_URL}"
echo "Configure in TL;DV Settings → Webhooks:"
echo "  POST ${WEBHOOK_URL}"
