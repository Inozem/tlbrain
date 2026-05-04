#!/usr/bin/env bash

set -euo pipefail

if [ -f .env ]; then
  echo "Loading .env file..."
  export $(grep -v '^#' .env | xargs)
fi

PROJECT_ID=${PROJECT_ID:-tlbrain-prod}
REGION=${REGION:-europe-west1}
TLDV_API_KEY=${TLDV_API_KEY:-}
ROOT_FOLDER_URL=${ROOT_FOLDER_URL:-}
TLDV_RECONCILIATION_SCHEDULE=${TLDV_RECONCILIATION_SCHEDULE:-"0 3 * * *"}

TLDV_IMPORT_SERVICE_NAME=${TLDV_IMPORT_SERVICE_NAME:-tlbrain-tldv-import}
TLDV_IMPORT_IMAGE="gcr.io/${PROJECT_ID}/${TLDV_IMPORT_SERVICE_NAME}"
TLDV_IMPORT_QUEUE=${TLDV_IMPORT_QUEUE:-tlbrain-tldv-import-queue}
TLDV_WEBHOOK_FUNCTION_NAME=${TLDV_WEBHOOK_FUNCTION_NAME:-tlbrain-tldv-webhook}
TLDV_RECONCILIATION_FUNCTION_NAME=${TLDV_RECONCILIATION_FUNCTION_NAME:-tlbrain-tldv-reconciliation}

VECTOR_SYNC_SERVICE_NAME=${VECTOR_SYNC_SERVICE_NAME:-tlbrain-vector-sync}
VECTOR_SYNC_QUEUE=${VECTOR_SYNC_QUEUE:-tlbrain-vector-sync-queue}

GOOGLE_CLIENT_ID=${GOOGLE_CLIENT_ID:-}
GOOGLE_CLIENT_SECRET=${GOOGLE_CLIENT_SECRET:-}
GOOGLE_REFRESH_TOKEN=${GOOGLE_REFRESH_TOKEN:-}

# =========================
# Google Drive auth setup
# =========================
if [ -z "${GOOGLE_REFRESH_TOKEN}" ]; then
  if [ -z "${GOOGLE_CLIENT_ID}" ] || [ -z "${GOOGLE_CLIENT_SECRET}" ]; then
    echo ""
    echo "ERROR: Google OAuth credentials not found in .env"
    echo ""
    echo "One-time setup:"
    echo "  1. Go to Google Cloud Console → APIs & Services → Credentials"
    echo "  2. Create Credentials → OAuth 2.0 Client ID → Desktop app"
    echo "  3. Add to .env:"
    echo "       GOOGLE_CLIENT_ID=..."
    echo "       GOOGLE_CLIENT_SECRET=..."
    echo "  4. Re-run this script"
    echo ""
    exit 1
  fi

  echo "Google Drive credentials not found, running setup (browser will open)..."
  python3 -m pip install -q requests
  python3 setup_tokens.py
  export $(grep -v '^#' .env | xargs)
fi

echo
echo "=== TL;DV Connector Local Deploy ==="
echo "Project: ${PROJECT_ID} / Region: ${REGION}"
echo "Image:   ${TLDV_IMPORT_IMAGE}"
echo

read -p "Continue? (y/n): " CONFIRM
[ "$CONFIRM" = "y" ] || { echo "Cancelled."; exit 0; }

gcloud config set project "${PROJECT_ID}"

# =========================
# Cloud Tasks queue
# =========================
echo
echo "Creating Cloud Tasks queue ${TLDV_IMPORT_QUEUE}..."
gcloud tasks queues create "${TLDV_IMPORT_QUEUE}" \
  --location="${REGION}" \
  --max-concurrent-dispatches=2 \
  2>/dev/null || echo "Queue already exists, skipping."

# =========================
# 1. Build & push Import Service
# =========================
echo
echo "--- Building Import Service image ---"
docker build -f infra/docker/Dockerfile.tldv_import -t "${TLDV_IMPORT_IMAGE}" .
docker push "${TLDV_IMPORT_IMAGE}"

# =========================
# 2. Deploy Import Service
# =========================
echo
echo "--- Deploying Import Service ---"

VECTOR_SYNC_URL=$(gcloud run services describe "${VECTOR_SYNC_SERVICE_NAME}" \
  --region "${REGION}" \
  --format='value(status.url)')

gcloud run deploy "${TLDV_IMPORT_SERVICE_NAME}" \
  --image "${TLDV_IMPORT_IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars TLDV_API_KEY="${TLDV_API_KEY}",ROOT_FOLDER_URL="${ROOT_FOLDER_URL}",VECTOR_SYNC_URL="${VECTOR_SYNC_URL}",VECTOR_SYNC_QUEUE="${VECTOR_SYNC_QUEUE}",TLDV_IMPORT_QUEUE="${TLDV_IMPORT_QUEUE}",REGION="${REGION}",GOOGLE_CLOUD_PROJECT="${PROJECT_ID}",GOOGLE_CLIENT_ID="${GOOGLE_CLIENT_ID}",GOOGLE_CLIENT_SECRET="${GOOGLE_CLIENT_SECRET}",GOOGLE_REFRESH_TOKEN="${GOOGLE_REFRESH_TOKEN}"

IMPORT_SERVICE_URL=$(gcloud run services describe "${TLDV_IMPORT_SERVICE_NAME}" \
  --region "${REGION}" \
  --format='value(status.url)')

echo "Import Service: ${IMPORT_SERVICE_URL}"

# =========================
# 3. Deploy Webhook Function
# =========================
echo
echo "--- Deploying Webhook Function ---"

STAGE_WEBHOOK=$(mktemp -d)
trap "rm -rf ${STAGE_WEBHOOK}" EXIT

cp services/connectors/tldv/webhook/main.py "${STAGE_WEBHOOK}/"
cp services/connectors/tldv/webhook/requirements.txt "${STAGE_WEBHOOK}/"

gcloud functions deploy "${TLDV_WEBHOOK_FUNCTION_NAME}" \
  --gen2 \
  --runtime python312 \
  --region "${REGION}" \
  --source "${STAGE_WEBHOOK}" \
  --entry-point tldv_webhook \
  --trigger-http \
  --allow-unauthenticated \
  --set-env-vars TLDV_IMPORT_QUEUE="${TLDV_IMPORT_QUEUE}",TLDV_IMPORT_SERVICE_URL="${IMPORT_SERVICE_URL}",REGION="${REGION}",GOOGLE_CLOUD_PROJECT="${PROJECT_ID}"

WEBHOOK_URL=$(gcloud functions describe "${TLDV_WEBHOOK_FUNCTION_NAME}" \
  --region "${REGION}" \
  --format='value(serviceConfig.uri)')

echo "Webhook Function: ${WEBHOOK_URL}"

# =========================
# 4. Deploy Reconciliation Function
# =========================
echo
echo "--- Deploying Reconciliation Function ---"

STAGE_RECONCILIATION=$(mktemp -d)
trap "rm -rf ${STAGE_RECONCILIATION}" EXIT

cp services/connectors/tldv/reconciliation/main.py "${STAGE_RECONCILIATION}/"
cp services/connectors/tldv/reconciliation/requirements.txt "${STAGE_RECONCILIATION}/"

gcloud functions deploy "${TLDV_RECONCILIATION_FUNCTION_NAME}" \
  --gen2 \
  --runtime python312 \
  --region "${REGION}" \
  --source "${STAGE_RECONCILIATION}" \
  --entry-point tldv_reconciliation \
  --trigger-http \
  --allow-unauthenticated \
  --set-env-vars TLDV_API_KEY="${TLDV_API_KEY}",TLDV_IMPORT_QUEUE="${TLDV_IMPORT_QUEUE}",TLDV_IMPORT_SERVICE_URL="${IMPORT_SERVICE_URL}",REGION="${REGION}",GOOGLE_CLOUD_PROJECT="${PROJECT_ID}"

RECONCILIATION_URL=$(gcloud functions describe "${TLDV_RECONCILIATION_FUNCTION_NAME}" \
  --region "${REGION}" \
  --format='value(serviceConfig.uri)')

echo "Reconciliation Function: ${RECONCILIATION_URL}"

# =========================
# Cloud Scheduler
# =========================
echo
echo "Configuring Cloud Scheduler (${TLDV_RECONCILIATION_SCHEDULE})..."
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

# =========================
# Summary
# =========================
echo
echo "========================================="
echo "Deploy complete"
echo "========================================="
echo
echo "Import Service:   ${IMPORT_SERVICE_URL}"
echo "Webhook Function: ${WEBHOOK_URL}"
echo "Reconciliation:   ${RECONCILIATION_URL}"
echo
echo "Configure in TL;DV Settings → Webhooks:"
echo "  POST ${WEBHOOK_URL}"
echo
echo "Grant Drive access to service account:"
gcloud run services describe "${TLDV_IMPORT_SERVICE_NAME}" \
  --region "${REGION}" \
  --format='value(spec.template.spec.serviceAccountName)'
