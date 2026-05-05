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
MCP_IMAGE="gcr.io/${PROJECT_ID}/${MCP_SERVICE_NAME}"
ALLOWED_EMAIL=${ALLOWED_EMAIL:-}

echo "--- Building MCP image ---"
docker build -f "${REPO_ROOT}/infra/docker/Dockerfile.mcp" -t "${MCP_IMAGE}" "${REPO_ROOT}"
docker push "${MCP_IMAGE}"

echo "--- Deploying MCP Service ---"

gcloud run deploy "${MCP_SERVICE_NAME}" \
  --image "${MCP_IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars ROOT_FOLDER_URL="${ROOT_FOLDER_URL}",GEMINI_API_KEY="${GEMINI_API_KEY}",QDRANT_URL="${QDRANT_URL}",QDRANT_API_KEY="${QDRANT_API_KEY}",QDRANT_COLLECTION="${QDRANT_COLLECTION}",RETRIEVAL_TOP_K="${RETRIEVAL_TOP_K}",RETRIEVAL_SCORE_THRESHOLD="${RETRIEVAL_SCORE_THRESHOLD}",ALLOWED_EMAIL="${ALLOWED_EMAIL}"

MCP_URL=$(gcloud run services describe "${MCP_SERVICE_NAME}" \
  --region "${REGION}" \
  --format='value(status.url)')

echo
echo "MCP Service: ${MCP_URL}/mcp"
