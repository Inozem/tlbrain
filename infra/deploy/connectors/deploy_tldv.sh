#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [ -f "${REPO_ROOT}/.env" ]; then
  export $(grep -v '^#' "${REPO_ROOT}/.env" | xargs)
fi

PROJECT_ID=${PROJECT_ID:-tlbrain-prod}
REGION=${REGION:-europe-west1}
TLDV_IMPORT_SERVICE_NAME=${TLDV_IMPORT_SERVICE_NAME:-tlbrain-tldv-import}
TLDV_IMPORT_IMAGE="inozem/tlbrain-tldv-import:${VERSION}"
TLDV_IMPORT_QUEUE=${TLDV_IMPORT_QUEUE:-tlbrain-tldv-import-queue}
TLDV_WEBHOOK_FUNCTION_NAME=${TLDV_WEBHOOK_FUNCTION_NAME:-tlbrain-tldv-webhook}
TLDV_RECONCILIATION_FUNCTION_NAME=${TLDV_RECONCILIATION_FUNCTION_NAME:-tlbrain-tldv-reconciliation}
GOOGLE_CLIENT_ID=${GOOGLE_CLIENT_ID:-}
GOOGLE_CLIENT_SECRET=${GOOGLE_CLIENT_SECRET:-}
GOOGLE_REFRESH_TOKEN=${GOOGLE_REFRESH_TOKEN:-}

mask() { local v="$1"; local l=${#v}; if [ $l -le 8 ]; then echo "****"; else echo "${v:0:4}****${v: -4}"; fi; }

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
    echo "  2. Create Credentials → OAuth 2.0 Client ID → Web application"
    echo "  3. Add to .env:"
    echo "       GOOGLE_CLIENT_ID=..."
    echo "       GOOGLE_CLIENT_SECRET=..."
    echo "  4. Re-run this script"
    echo ""
    exit 1
  fi

  echo "Google Drive credentials not found, running setup (browser will open)..."
  python3 -m pip install -q requests
  python3 "${REPO_ROOT}/infra/deploy/setup_tokens.py"
  export $(grep -v '^#' "${REPO_ROOT}/.env" | xargs)
fi

echo
echo "=== TL;DV Connector Deploy ==="
echo "Project: ${PROJECT_ID} / Region: ${REGION}"
echo "Image:   ${TLDV_IMPORT_IMAGE}"
echo

if [ "${SKIP_CONFIRM:-}" != "1" ]; then
  read -p "Continue? (y/n): " CONFIRM
  [ "$CONFIRM" = "y" ] || { echo "Cancelled."; exit 0; }
fi

gcloud config set project "${PROJECT_ID}"

# =========================
# Cloud Tasks queue
# =========================
echo "Creating Cloud Tasks queue ${TLDV_IMPORT_QUEUE}..."
gcloud tasks queues create "${TLDV_IMPORT_QUEUE}" \
  --location="${REGION}" \
  --max-concurrent-dispatches=2 \
  2>/dev/null || echo "Queue already exists, skipping."

# =========================
# Deploy components
# =========================
bash "${SCRIPT_DIR}/tldv/deploy_import.sh"
bash "${SCRIPT_DIR}/tldv/deploy_webhook.sh"
bash "${SCRIPT_DIR}/tldv/deploy_reconciliation.sh"

# =========================
# Summary
# =========================
IMPORT_URL=$(gcloud run services describe "${TLDV_IMPORT_SERVICE_NAME}" \
  --region "${REGION}" \
  --format='value(status.url)')
WEBHOOK_URL=$(gcloud functions describe "${TLDV_WEBHOOK_FUNCTION_NAME}" \
  --region "${REGION}" \
  --format='value(serviceConfig.uri)')
RECONCILIATION_URL=$(gcloud functions describe "${TLDV_RECONCILIATION_FUNCTION_NAME}" \
  --region "${REGION}" \
  --format='value(serviceConfig.uri)')

echo
echo "========================================="
echo "TL;DV deploy complete"
echo "========================================="
echo
echo "┌─ TL;DV Webhook ──────────────────────────────────────────"
echo "│"
echo "│  TL;DV Settings → Webhooks → Add:"
echo "│     POST ${WEBHOOK_URL}"
echo "│"
echo "└──────────────────────────────────────────────────────────"
echo
echo "┌─ Google OAuth credentials ───────────────────────────────"
echo "│"
echo "│  Client ID:     $(mask "${GOOGLE_CLIENT_ID}")"
echo "│  Client Secret: $(mask "${GOOGLE_CLIENT_SECRET}")"
echo "│"
echo "└──────────────────────────────────────────────────────────"
echo
echo "Internal:"
echo "  Import Service:  ${IMPORT_URL}"
echo "  Reconciliation:  ${RECONCILIATION_URL}"
echo
echo "Grant Drive access to service account:"
gcloud run services describe "${TLDV_IMPORT_SERVICE_NAME}" \
  --region "${REGION}" \
  --format='value(spec.template.spec.serviceAccountName)'
