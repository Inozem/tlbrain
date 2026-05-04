#!/usr/bin/env bash

set -euo pipefail

if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

PROJECT_ID=${PROJECT_ID:-tlbrain-prod}
REGION=${REGION:-europe-west1}
TLDV_WEBHOOK_FUNCTION_NAME=${TLDV_WEBHOOK_FUNCTION_NAME:-tlbrain-tldv-webhook}
TLDV_IMPORT_QUEUE=${TLDV_IMPORT_QUEUE:-tlbrain-tldv-import-queue}
TLDV_IMPORT_SERVICE_URL=${TLDV_IMPORT_SERVICE_URL:-}

# =========================
# Create Cloud Tasks queue
# =========================
echo "Creating Cloud Tasks queue ${TLDV_IMPORT_QUEUE}..."
gcloud tasks queues create "${TLDV_IMPORT_QUEUE}" \
  --location="${REGION}" \
  --max-concurrent-dispatches=2 \
  2>/dev/null || echo "Queue already exists, skipping."

# =========================
# Deploy Webhook Function
# =========================
STAGE=$(mktemp -d)
trap "rm -rf ${STAGE}" EXIT

cp services/connectors/tldv/webhook/main.py "${STAGE}/"
cp services/connectors/tldv/webhook/requirements.txt "${STAGE}/"

gcloud functions deploy "${TLDV_WEBHOOK_FUNCTION_NAME}" \
  --gen2 \
  --runtime python312 \
  --region "${REGION}" \
  --source "${STAGE}" \
  --entry-point tldv_webhook \
  --trigger-http \
  --allow-unauthenticated \
  --set-env-vars TLDV_IMPORT_QUEUE="${TLDV_IMPORT_QUEUE}",TLDV_IMPORT_SERVICE_URL="${TLDV_IMPORT_SERVICE_URL}",REGION="${REGION}",GOOGLE_CLOUD_PROJECT="${PROJECT_ID}"

WEBHOOK_URL=$(gcloud functions describe "${TLDV_WEBHOOK_FUNCTION_NAME}" \
  --region "${REGION}" \
  --format='value(serviceConfig.uri)')

echo
echo "Webhook deployed: ${WEBHOOK_URL}"
echo "Configure in TL;DV Settings → Webhooks:"
echo "  POST ${WEBHOOK_URL}"
