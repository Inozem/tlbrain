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

ARTIFACT="${REPO_ROOT}/.build/tldv_webhook"
if PREFLIGHT_RESULT=$(bash "${REPO_ROOT}/infra/build/preflight.sh" \
  --service tldv_webhook \
  --source-root "${REPO_ROOT}" \
  --output "${ARTIFACT}"); then
  :
else
  PREFLIGHT_STATUS=$?
  printf '%s\n' "${PREFLIGHT_RESULT}"
  exit "${PREFLIGHT_STATUS}"
fi
printf '%s\n' "${PREFLIGHT_RESULT}"
ARTIFACT_FINGERPRINT=$(printf '%s\n' "${PREFLIGHT_RESULT}" | sed -n 's/^Artifact fingerprint: //p')
if [[ ! "${ARTIFACT_FINGERPRINT}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "preflight: missing successful artifact fingerprint; refusing Function upload" >&2
  exit 1
fi
bash "${REPO_ROOT}/infra/build/preflight.sh" --verify-source \
  --artifact "${ARTIFACT}" \
  --fingerprint "${ARTIFACT_FINGERPRINT}"

gcloud functions deploy "${TLDV_WEBHOOK_FUNCTION_NAME}" \
  --gen2 \
  --runtime python312 \
  --region "${REGION}" \
  --source "${ARTIFACT}" \
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
