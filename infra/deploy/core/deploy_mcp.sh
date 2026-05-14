#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [ -f "${REPO_ROOT}/.env" ]; then
  export $(grep -v '^#' "${REPO_ROOT}/.env" | xargs)
fi

PROJECT_ID=${PROJECT_ID:-tlbrain-prod}
REGION=${REGION:-europe-west1}
MCP_SERVICE_NAME=${MCP_SERVICE_NAME:-tlbrain-mcp}
MCP_IMAGE="inozem/tlbrain-mcp:${VERSION}"
ALLOWED_EMAIL=${ALLOWED_EMAIL:-}
VECTOR_SYNC_SERVICE_NAME=${VECTOR_SYNC_SERVICE_NAME:-tlbrain-vector-sync}
TLDV_RECONCILIATION_NAME=${TLDV_RECONCILIATION_NAME:-tlbrain-tldv-reconciliation}
SYNC_CHECKER_NAME=${SYNC_CHECKER_NAME:-tlbrain-sync-checker}

echo "--- Deploying MCP Service ---"

VECTOR_SYNC_URL=$(gcloud run services describe "${VECTOR_SYNC_SERVICE_NAME}" \
  --region "${REGION}" \
  --format='value(status.url)')

TLDV_RECONCILIATION_URL=$(gcloud functions describe "${TLDV_RECONCILIATION_NAME}" \
  --region "${REGION}" \
  --format='value(serviceConfig.uri)')

SYNC_CHECKER_URL=$(gcloud functions describe "${SYNC_CHECKER_NAME}" \
  --region "${REGION}" \
  --format='value(serviceConfig.uri)')

gcloud run deploy "${MCP_SERVICE_NAME}" \
  --image "${MCP_IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars ROOT_FOLDER_URL="${ROOT_FOLDER_URL}",GEMINI_API_KEY="${GEMINI_API_KEY}",QDRANT_URL="${QDRANT_URL}",QDRANT_API_KEY="${QDRANT_API_KEY}",QDRANT_COLLECTION="${QDRANT_COLLECTION}",RETRIEVAL_TOP_K="${RETRIEVAL_TOP_K}",RETRIEVAL_SCORE_THRESHOLD="${RETRIEVAL_SCORE_THRESHOLD}",ALLOWED_EMAIL="${ALLOWED_EMAIL}",VECTOR_SYNC_URL="${VECTOR_SYNC_URL}",TLDV_RECONCILIATION_URL="${TLDV_RECONCILIATION_URL}",SYNC_CHECKER_URL="${SYNC_CHECKER_URL}"

MCP_URL=$(gcloud run services describe "${MCP_SERVICE_NAME}" \
  --region "${REGION}" \
  --format='value(status.url)')

echo
echo "MCP Service: ${MCP_URL}/mcp"
