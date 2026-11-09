#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [ -f "${REPO_ROOT}/.env" ]; then
  export $(grep -v '^#' "${REPO_ROOT}/.env" | xargs)
fi
export -n TELEGRAM_BOT_TOKEN TELEGRAM_ALLOWED_USER_IDS 2>/dev/null || true

TELEGRAM_BOT_TOKEN_CHECK=${TELEGRAM_BOT_TOKEN:-}
TELEGRAM_BOT_TOKEN_CHECK=${TELEGRAM_BOT_TOKEN_CHECK//[[:space:]]/}
TELEGRAM_ALLOWED_USER_IDS_CHECK=${TELEGRAM_ALLOWED_USER_IDS:-}
TELEGRAM_ALLOWED_USER_IDS_CHECK=${TELEGRAM_ALLOWED_USER_IDS_CHECK//[[:space:]]/}
if [[ -n "${TELEGRAM_BOT_TOKEN_CHECK}" && -z "${TELEGRAM_ALLOWED_USER_IDS_CHECK}" ]] || \
  [[ -z "${TELEGRAM_BOT_TOKEN_CHECK}" && -n "${TELEGRAM_ALLOWED_USER_IDS_CHECK}" ]]; then
  echo "TELEGRAM_BOT_TOKEN and TELEGRAM_ALLOWED_USER_IDS must be configured together" >&2
  exit 1
fi

PROJECT_ID=${PROJECT_ID:-tlbrain-prod}
REGION=${REGION:-europe-west1}
MCP_SERVICE_NAME=${MCP_SERVICE_NAME:-tlbrain-mcp}
MCP_IMAGE="gcr.io/${PROJECT_ID}/${MCP_SERVICE_NAME}"
ALLOWED_EMAIL=${ALLOWED_EMAIL:-}
RETRIEVAL_TOP_K=${RETRIEVAL_TOP_K:-15}
RETRIEVAL_SCORE_THRESHOLD=${RETRIEVAL_SCORE_THRESHOLD:-0.6}
VECTOR_SYNC_SERVICE_NAME=${VECTOR_SYNC_SERVICE_NAME:-tlbrain-vector-sync}
VECTOR_SYNC_QUEUE=${VECTOR_SYNC_QUEUE:-tlbrain-vector-sync-queue}
TLDV_RECONCILIATION_NAME=${TLDV_RECONCILIATION_NAME:-tlbrain-tldv-reconciliation}
SYNC_CHECKER_NAME=${SYNC_CHECKER_NAME:-tlbrain-sync-checker}
MCP_MAX_INSTANCES=${MCP_MAX_INSTANCES:-1}
TLDV_ENABLED=${TLDV_API_KEY:-}
TLDV_ENABLED=${TLDV_ENABLED//[[:space:]]/}

echo "--- Building MCP image ---"
docker build --platform linux/amd64 -f "${REPO_ROOT}/infra/docker/Dockerfile.mcp" -t "${MCP_IMAGE}" "${REPO_ROOT}"
bash "${REPO_ROOT}/infra/build/preflight.sh" --image-mode \
  --image "${MCP_IMAGE}" \
  --service mcp \
  --platform linux/amd64
docker push "${MCP_IMAGE}"

echo "--- Deploying MCP Service ---"

VECTOR_SYNC_URL=$(gcloud run services describe "${VECTOR_SYNC_SERVICE_NAME}" \
  --region "${REGION}" \
  --format='value(status.url)')

SYNC_CHECKER_URL=$(gcloud functions describe "${SYNC_CHECKER_NAME}" \
  --region "${REGION}" \
  --format='value(serviceConfig.uri)')

MCP_ENV_VARS=("ROOT_FOLDER_URL=${ROOT_FOLDER_URL}" "GEMINI_API_KEY=${GEMINI_API_KEY}" \
  "QDRANT_URL=${QDRANT_URL}" "QDRANT_API_KEY=${QDRANT_API_KEY}" \
  "QDRANT_COLLECTION=${QDRANT_COLLECTION}" "RETRIEVAL_TOP_K=${RETRIEVAL_TOP_K}" \
  "RETRIEVAL_SCORE_THRESHOLD=${RETRIEVAL_SCORE_THRESHOLD}" "VECTOR_SYNC_URL=${VECTOR_SYNC_URL}" \
  "REGION=${REGION}" "VECTOR_SYNC_QUEUE=${VECTOR_SYNC_QUEUE}")
if [[ -n "${TLDV_ENABLED}" ]]; then
  TLDV_RECONCILIATION_URL=$(gcloud functions describe "${TLDV_RECONCILIATION_NAME}" \
    --region "${REGION}" \
    --format='value(serviceConfig.uri)')
  MCP_ENV_VARS+=("TLDV_RECONCILIATION_URL=${TLDV_RECONCILIATION_URL}")
fi
MCP_ENV_VARS+=("SYNC_CHECKER_URL=${SYNC_CHECKER_URL}" \
  "GOOGLE_REFRESH_TOKEN=${GOOGLE_REFRESH_TOKEN}" "GOOGLE_CLIENT_ID=${GOOGLE_CLIENT_ID}" \
  "GOOGLE_CLIENT_SECRET=${GOOGLE_CLIENT_SECRET}" "VERSION=${VERSION}")
MCP_ENV_VARS+=("ALLOWED_EMAIL=${ALLOWED_EMAIL}")

gcloud run deploy "${MCP_SERVICE_NAME}" \
  --image "${MCP_IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --max-instances "${MCP_MAX_INSTANCES}" \
  --set-env-vars "^|^$(IFS='|'; echo "${MCP_ENV_VARS[*]}")"

MCP_URL=$(gcloud run services describe "${MCP_SERVICE_NAME}" \
  --region "${REGION}" \
  --format='value(status.url)')

if [[ -z "${MCP_URL}" ]]; then
  echo "Could not determine deployed MCP service URL" >&2
  exit 1
fi

if [[ "${DEPLOY_DEFER_DIALOG_READINESS:-}" != "1" ]]; then
  DIALOG_SETTINGS_OUTPUT=$(bash "${REPO_ROOT}/infra/deploy/core/helpers/deploy_dialog_operations.sh" "${MCP_URL}")
  DIALOG_ENV_VARS=()
  declare -A DIALOG_SETTING_SEEN=()
  while IFS= read -r setting; do
    case "${setting}" in
      INTERNAL_DIALOG_OPERATIONS_URL=*|DIALOG_OPERATIONS_QUEUE=*|CLOUD_TASKS_OIDC_AUDIENCE=*|CLOUD_TASKS_OIDC_SERVICE_ACCOUNT_EMAIL=*|OIDC_AUDIENCE=*|OIDC_SERVICE_ACCOUNT_EMAIL=*)
        setting_name=${setting%%=*}
        if [[ -n "${DIALOG_SETTING_SEEN[${setting_name}]:-}" ]]; then
          echo "Duplicate dialog operations setting from helper: ${setting_name}" >&2
          exit 1
        fi
        DIALOG_SETTING_SEEN["${setting_name}"]=1
        DIALOG_ENV_VARS+=("${setting}") ;;
    esac
  done <<< "${DIALOG_SETTINGS_OUTPUT}"
  if [[ ${#DIALOG_ENV_VARS[@]} -ne 6 ]]; then
    echo "Dialog operations helper returned incomplete readiness settings" >&2
    exit 1
  fi
  DIALOG_ENV_VARS_JOINED=$(IFS='|'; echo "${DIALOG_ENV_VARS[*]}")
  gcloud run services update "${MCP_SERVICE_NAME}" \
    --region "${REGION}" \
    --update-env-vars "^|^${DIALOG_ENV_VARS_JOINED}"
fi

echo
echo "MCP Service: ${MCP_URL}/mcp"
