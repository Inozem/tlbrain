#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

if [ -f "${REPO_ROOT}/.env" ]; then
  export $(grep -v '^#' "${REPO_ROOT}/.env" | xargs)
fi

PROJECT_ID=${PROJECT_ID:-tlbrain-prod}
REGION=${REGION:-europe-west1}
TLDV_RECONCILIATION_FUNCTION_NAME=${TLDV_RECONCILIATION_FUNCTION_NAME:-tlbrain-tldv-reconciliation}
TLDV_IMPORT_QUEUE=${TLDV_IMPORT_QUEUE:-tlbrain-tldv-import-queue}
TLDV_IMPORT_SERVICE_NAME=${TLDV_IMPORT_SERVICE_NAME:-tlbrain-tldv-import}
TLDV_API_KEY=${TLDV_API_KEY:-}
TLDV_RECONCILIATION_SCHEDULE=${TLDV_RECONCILIATION_SCHEDULE:-"0 3 * * *"}

echo "--- Deploying TL;DV Reconciliation Function ---"

IMPORT_SERVICE_URL=$(gcloud run services describe "${TLDV_IMPORT_SERVICE_NAME}" \
  --region "${REGION}" \
  --format='value(status.url)')

STAGE=$(mktemp -d)
trap "rm -rf ${STAGE}" EXIT

cp "${REPO_ROOT}/services/connectors/tldv/reconciliation/main.py" "${STAGE}/"
cp "${REPO_ROOT}/services/connectors/tldv/reconciliation/requirements.txt" "${STAGE}/"

gcloud functions deploy "${TLDV_RECONCILIATION_FUNCTION_NAME}" \
  --gen2 \
  --runtime python312 \
  --region "${REGION}" \
  --source "${STAGE}" \
  --entry-point tldv_reconciliation \
  --trigger-http \
  --allow-unauthenticated \
  --set-env-vars TLDV_API_KEY="${TLDV_API_KEY}",TLDV_IMPORT_QUEUE="${TLDV_IMPORT_QUEUE}",TLDV_IMPORT_SERVICE_URL="${IMPORT_SERVICE_URL}",REGION="${REGION}",GOOGLE_CLOUD_PROJECT="${PROJECT_ID}"

RECONCILIATION_URL=$(gcloud functions describe "${TLDV_RECONCILIATION_FUNCTION_NAME}" \
  --region "${REGION}" \
  --format='value(serviceConfig.uri)')

echo
echo "Reconciliation deployed: ${RECONCILIATION_URL}"

gcloud scheduler jobs create http "${TLDV_RECONCILIATION_FUNCTION_NAME}-schedule" \
  --location="${REGION}" \
  --schedule="${TLDV_RECONCILIATION_SCHEDULE}" \
  --uri="${RECONCILIATION_URL}" \
  --http-method=POST \
  2>/dev/null || \
gcloud scheduler jobs update http "${TLDV_RECONCILIATION_FUNCTION_NAME}-schedule" \
  --location="${REGION}" \
  --schedule="${TLDV_RECONCILIATION_SCHEDULE}" \
  --uri="${RECONCILIATION_URL}" \
  --http-method=POST

echo "Scheduler set: ${TLDV_RECONCILIATION_SCHEDULE}"
