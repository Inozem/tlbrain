#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

if [ -f "${REPO_ROOT}/.env" ]; then
  export $(grep -v '^#' "${REPO_ROOT}/.env" | xargs)
fi

PROJECT_ID=${PROJECT_ID:-tlbrain-prod}
REGION=${REGION:-europe-west1}
TLDV_IMPORT_SERVICE_NAME=${TLDV_IMPORT_SERVICE_NAME:-tlbrain-tldv-import}
TLDV_IMPORT_IMAGE="inozem/tlbrain-tldv-import:${VERSION}"
TLDV_IMPORT_QUEUE=${TLDV_IMPORT_QUEUE:-tlbrain-tldv-import-queue}
VECTOR_SYNC_SERVICE_NAME=${VECTOR_SYNC_SERVICE_NAME:-tlbrain-vector-sync}
VECTOR_SYNC_QUEUE=${VECTOR_SYNC_QUEUE:-tlbrain-vector-sync-queue}
TLDV_API_KEY=${TLDV_API_KEY:-}
ROOT_FOLDER_URL=${ROOT_FOLDER_URL:-}
GOOGLE_CLIENT_ID=${GOOGLE_CLIENT_ID:-}
GOOGLE_CLIENT_SECRET=${GOOGLE_CLIENT_SECRET:-}
GOOGLE_REFRESH_TOKEN=${GOOGLE_REFRESH_TOKEN:-}

echo "--- Deploying TL;DV Import Service ---"

VECTOR_SYNC_URL=$(gcloud run services describe "${VECTOR_SYNC_SERVICE_NAME}" \
  --region "${REGION}" \
  --format='value(status.url)')

gcloud run deploy "${TLDV_IMPORT_SERVICE_NAME}" \
  --image "${TLDV_IMPORT_IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars TLDV_API_KEY="${TLDV_API_KEY}",ROOT_FOLDER_URL="${ROOT_FOLDER_URL}",VECTOR_SYNC_URL="${VECTOR_SYNC_URL}",VECTOR_SYNC_QUEUE="${VECTOR_SYNC_QUEUE}",TLDV_IMPORT_QUEUE="${TLDV_IMPORT_QUEUE}",REGION="${REGION}",GOOGLE_CLOUD_PROJECT="${PROJECT_ID}",GOOGLE_CLIENT_ID="${GOOGLE_CLIENT_ID}",GOOGLE_CLIENT_SECRET="${GOOGLE_CLIENT_SECRET}",GOOGLE_REFRESH_TOKEN="${GOOGLE_REFRESH_TOKEN}"

IMPORT_URL=$(gcloud run services describe "${TLDV_IMPORT_SERVICE_NAME}" \
  --region "${REGION}" \
  --format='value(status.url)')

echo
echo "Import Service: ${IMPORT_URL}"
echo "Health:  GET  ${IMPORT_URL}/"
echo "Import:  POST ${IMPORT_URL}/import"
echo
echo "Grant Drive access to service account:"
gcloud run services describe "${TLDV_IMPORT_SERVICE_NAME}" \
  --region "${REGION}" \
  --format='value(spec.template.spec.serviceAccountName)'
