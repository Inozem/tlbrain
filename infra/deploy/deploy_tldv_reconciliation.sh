#!/usr/bin/env bash

set -euo pipefail

if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

PROJECT_ID=${PROJECT_ID:-tlbrain-prod}
REGION=${REGION:-europe-west1}
TLDV_RECONCILIATION_FUNCTION_NAME=${TLDV_RECONCILIATION_FUNCTION_NAME:-tlbrain-tldv-reconciliation}
TLDV_IMPORT_QUEUE=${TLDV_IMPORT_QUEUE:-tlbrain-tldv-import-queue}
TLDV_IMPORT_SERVICE_URL=${TLDV_IMPORT_SERVICE_URL:-}
TLDV_API_KEY=${TLDV_API_KEY:-}
TLDV_RECONCILIATION_SCHEDULE=${TLDV_RECONCILIATION_SCHEDULE:-"0 3 * * *"}

# =========================
# Deploy Reconciliation Function
# =========================
STAGE=$(mktemp -d)
trap "rm -rf ${STAGE}" EXIT

cp services/connectors/tldv/reconciliation/main.py "${STAGE}/"
cp services/connectors/tldv/reconciliation/requirements.txt "${STAGE}/"

gcloud functions deploy "${TLDV_RECONCILIATION_FUNCTION_NAME}" \
  --gen2 \
  --runtime python312 \
  --region "${REGION}" \
  --source "${STAGE}" \
  --entry-point tldv_reconciliation \
  --trigger-http \
  --allow-unauthenticated \
  --set-env-vars TLDV_API_KEY="${TLDV_API_KEY}",TLDV_IMPORT_QUEUE="${TLDV_IMPORT_QUEUE}",TLDV_IMPORT_SERVICE_URL="${TLDV_IMPORT_SERVICE_URL}",REGION="${REGION}",GOOGLE_CLOUD_PROJECT="${PROJECT_ID}"

RECONCILIATION_URL=$(gcloud functions describe "${TLDV_RECONCILIATION_FUNCTION_NAME}" \
  --region "${REGION}" \
  --format='value(serviceConfig.uri)')

echo
echo "Reconciliation deployed: ${RECONCILIATION_URL}"
echo "Trigger manually: POST ${RECONCILIATION_URL}"

# =========================
# Create Cloud Scheduler job
# =========================
echo "Creating Cloud Scheduler job (${TLDV_RECONCILIATION_SCHEDULE})..."
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

# =========================
# Run initial reconciliation
# =========================
echo "Running initial reconciliation..."
curl -s -X POST "${RECONCILIATION_URL}" | python3 -m json.tool
echo "Initial reconciliation done."
