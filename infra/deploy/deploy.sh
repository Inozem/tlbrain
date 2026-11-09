#!/usr/bin/env bash

set -euo pipefail

# =========================
# Load .env if exists
# =========================
if [ -f .env ]; then
  echo "Loading .env file..."
  export $(grep -v '^#' .env | xargs)
fi
TELEGRAM_BOT_TOKEN_VALUE=${TELEGRAM_BOT_TOKEN:-}
TELEGRAM_ALLOWED_USER_IDS_VALUE=${TELEGRAM_ALLOWED_USER_IDS:-}
export -n TELEGRAM_BOT_TOKEN TELEGRAM_ALLOWED_USER_IDS 2>/dev/null || true

# =========================
# Defaults
# =========================
VERSION=${VERSION:-latest}

DOCKERHUB_USERNAME=inozem

PROJECT_ID=${PROJECT_ID:-tlbrain-prod}
REGION=${REGION:-europe-west1}

MCP_SERVICE_NAME=${MCP_SERVICE_NAME:-tlbrain-mcp}
VECTOR_SYNC_SERVICE_NAME=${VECTOR_SYNC_SERVICE_NAME:-tlbrain-vector-sync}
SYNC_CHECKER_NAME=${SYNC_CHECKER_NAME:-tlbrain-sync-checker}
VECTOR_SYNC_QUEUE=${VECTOR_SYNC_QUEUE:-tlbrain-vector-sync-queue}
CLOUD_TASKS_MAX_CONCURRENT=${CLOUD_TASKS_MAX_CONCURRENT:-2}
SYNC_CHECKER_SCHEDULE=${SYNC_CHECKER_SCHEDULE:-"*/15 * * * *"}
TLDV_WEBHOOK_FUNCTION_NAME=${TLDV_WEBHOOK_FUNCTION_NAME:-tlbrain-tldv-webhook}
GOOGLE_CLIENT_ID=${GOOGLE_CLIENT_ID:-}
GOOGLE_CLIENT_SECRET=${GOOGLE_CLIENT_SECRET:-}
GOOGLE_REFRESH_TOKEN=${GOOGLE_REFRESH_TOKEN:-}
GEMINI_API_KEY=${GEMINI_API_KEY:-}
QDRANT_URL=${QDRANT_URL:-}
QDRANT_API_KEY=${QDRANT_API_KEY:-}
TLDV_API_KEY=${TLDV_API_KEY:-}
ROOT_FOLDER_URL=${ROOT_FOLDER_URL:-}
ALLOWED_EMAIL=${ALLOWED_EMAIL:-}
TELEGRAM_BOT_TOKEN_CHECK=${TELEGRAM_BOT_TOKEN_VALUE}
TELEGRAM_BOT_TOKEN_CHECK=${TELEGRAM_BOT_TOKEN_CHECK//[[:space:]]/}
TELEGRAM_ALLOWED_USER_IDS_CHECK=${TELEGRAM_ALLOWED_USER_IDS_VALUE}
TELEGRAM_ALLOWED_USER_IDS_CHECK=${TELEGRAM_ALLOWED_USER_IDS_CHECK//[[:space:]]/}
TLDV_API_KEY_CHECK=${TLDV_API_KEY}
TLDV_API_KEY_CHECK=${TLDV_API_KEY_CHECK//[[:space:]]/}
if [[ -n "${TELEGRAM_BOT_TOKEN_CHECK}" && -z "${TELEGRAM_ALLOWED_USER_IDS_CHECK}" ]] || \
  [[ -z "${TELEGRAM_BOT_TOKEN_CHECK}" && -n "${TELEGRAM_ALLOWED_USER_IDS_CHECK}" ]]; then
  echo "TELEGRAM_BOT_TOKEN and TELEGRAM_ALLOWED_USER_IDS must be configured together" >&2
  exit 1
fi
TELEGRAM_ENABLED=0
if [[ -n "${TELEGRAM_BOT_TOKEN_CHECK}" ]]; then
  IFS=',' read -r -a TELEGRAM_USER_IDS <<< "${TELEGRAM_ALLOWED_USER_IDS_VALUE}"
  for telegram_user_id in "${TELEGRAM_USER_IDS[@]}"; do
    if [[ ! "${telegram_user_id}" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]]; then
      echo "TELEGRAM_ALLOWED_USER_IDS must be a comma-separated list of positive decimal IDs" >&2
      exit 1
    fi
    telegram_user_id=${telegram_user_id//[[:space:]]/}
    if [[ ! "${telegram_user_id}" =~ [1-9] ]]; then
      echo "TELEGRAM_ALLOWED_USER_IDS must be a comma-separated list of positive decimal IDs" >&2
      exit 1
    fi
  done
  TELEGRAM_ENABLED=1
fi
TLDV_ENABLED=0
if [[ -n "${TLDV_API_KEY_CHECK}" ]]; then TLDV_ENABLED=1; fi

prompt_if_empty() {
  local var="$1"; local label="$2"
  if [ -z "${!var}" ]; then
    read -p "${label}: " val
    export "$var"="$val"
  fi
}

prompt_if_empty ROOT_FOLDER_URL    "ROOT_FOLDER_URL (Google Drive folder URL)"
prompt_if_empty GEMINI_API_KEY     "GEMINI_API_KEY"
prompt_if_empty QDRANT_URL         "QDRANT_URL"
prompt_if_empty QDRANT_API_KEY     "QDRANT_API_KEY"
prompt_if_empty QDRANT_COLLECTION  "QDRANT_COLLECTION"
prompt_if_empty GOOGLE_CLIENT_ID   "GOOGLE_CLIENT_ID"
prompt_if_empty GOOGLE_CLIENT_SECRET "GOOGLE_CLIENT_SECRET"

mask() { local v="$1"; local l=${#v}; if [ $l -le 8 ]; then echo "****"; else echo "${v:0:4}****${v: -4}"; fi; }

# =========================
# Show config
# =========================
echo
echo "========================================="
echo "Deploy config"
echo "========================================="
echo "  Project:        ${PROJECT_ID}"
echo "  Region:         ${REGION}"
echo "  Version:        ${VERSION}"
echo ""
echo "  Root folder:    ${ROOT_FOLDER_URL}"
echo "  Allowed emails: ${ALLOWED_EMAIL}"
echo "  Qdrant URL:     ${QDRANT_URL}"
echo ""
echo "  Gemini key:     $(mask "${GEMINI_API_KEY}")"
echo "  Qdrant key:     $(mask "${QDRANT_API_KEY}")"
echo "  TL;DV key:      $(mask "${TLDV_API_KEY}")"
echo "  Client ID:      $(mask "${GOOGLE_CLIENT_ID}")"
echo "  Client Secret:  $(mask "${GOOGLE_CLIENT_SECRET}")"
if [ -n "${GOOGLE_REFRESH_TOKEN}" ]; then echo "  Refresh token:  $(mask "${GOOGLE_REFRESH_TOKEN}")"; else echo "  Refresh token:  (not set)"; fi
echo

# =========================
# Google OAuth setup (if needed)
# =========================
if [ -z "${GOOGLE_REFRESH_TOKEN}" ]; then
  if [ -z "${GOOGLE_CLIENT_ID}" ] || [ -z "${GOOGLE_CLIENT_SECRET}" ]; then
    echo "ERROR: GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET must be set in .env"
    exit 1
  fi
  echo "GOOGLE_REFRESH_TOKEN not set — running OAuth setup (browser will open)..."
  bash infra/deploy/setup_tokens.sh
  export $(grep -v '^#' .env | xargs)
  TELEGRAM_BOT_TOKEN_VALUE=${TELEGRAM_BOT_TOKEN:-}
  TELEGRAM_ALLOWED_USER_IDS_VALUE=${TELEGRAM_ALLOWED_USER_IDS:-}
  export -n TELEGRAM_BOT_TOKEN TELEGRAM_ALLOWED_USER_IDS 2>/dev/null || true
  echo "Token saved. Continuing deploy..."
  echo
fi

# =========================
# Confirm
# =========================
read -p "Continue deploy? (y/n): " CONFIRM

if [ "$CONFIRM" != "y" ]; then
  echo "Cancelled."
  exit 0
fi

# =========================
# Set project
# =========================
gcloud config set project "${PROJECT_ID}"

# =========================
# Enable required APIs
# =========================
echo "Enabling required APIs..."
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  drive.googleapis.com \
  docs.googleapis.com \
  firestore.googleapis.com \
  cloudtasks.googleapis.com \
  cloudfunctions.googleapis.com \
  cloudscheduler.googleapis.com

bash infra/deploy/core/helpers/deploy_firestore_indexes.sh

# =========================
# Deploy core (except MCP)
# =========================
DEPLOY_DEFER_DIALOG_READINESS=1 bash infra/deploy/core/deploy_vector_sync.sh
echo "rollout_phase:decoder_compatible"

# =========================
# Deploy TL;DV Connector
# =========================
if [[ "${TLDV_ENABLED}" == "1" ]]; then
  SKIP_CONFIRM=1 bash infra/deploy/connectors/deploy_tldv.sh
fi

# =========================
# Deploy Checker (after TL;DV — needs TLDV_IMPORT_SERVICE_URL)
# =========================
bash infra/deploy/core/deploy_sync_checker.sh
echo "rollout_phase:guards_compatible"
DEPLOY_DEFER_DIALOG_READINESS=1 bash infra/deploy/core/deploy_mcp.sh

echo "Confirm that old workers and their external I/O are drained or terminated. This script does not drain workers."
if ! read -r -p "Confirm old workers and external I/O are drained/terminated? (y/n): " DRAIN_CONFIRMED; then
  echo "Drain confirmation is required before writer activation." >&2
  exit 1
fi
if [[ "${DRAIN_CONFIRMED}" != "y" ]]; then
  echo "Drain confirmation declined; writer activation cancelled." >&2
  exit 1
fi
echo "rollout_gate:old_workers_drained (operator attestation)"

MCP_URL=$(gcloud run services describe "${MCP_SERVICE_NAME}" \
  --region "${REGION}" --format='value(status.url)')
if [[ -z "${MCP_URL}" ]]; then
  echo "Could not determine deployed MCP service URL" >&2
  exit 1
fi
DIALOG_SETTINGS_OUTPUT=$(bash infra/deploy/core/helpers/deploy_dialog_operations.sh "${MCP_URL}")
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
      DIALOG_ENV_VARS+=("${setting}")
      case "${setting}" in
        INTERNAL_DIALOG_OPERATIONS_URL=*) VECTOR_DIALOG_URL=${setting#*=} ;;
        DIALOG_OPERATIONS_QUEUE=*) VECTOR_DIALOG_QUEUE=${setting#*=} ;;
        CLOUD_TASKS_OIDC_AUDIENCE=*) VECTOR_DIALOG_AUDIENCE=${setting#*=} ;;
        CLOUD_TASKS_OIDC_SERVICE_ACCOUNT_EMAIL=*) VECTOR_DIALOG_INVOKER=${setting#*=} ;;
      esac ;;
  esac
done <<< "${DIALOG_SETTINGS_OUTPUT}"
if [[ ${#DIALOG_ENV_VARS[@]} -ne 6 ]]; then
  echo "Dialog operations helper returned incomplete readiness settings" >&2
  exit 1
fi
DIALOG_ENV_VARS_JOINED=$(IFS='|'; echo "${DIALOG_ENV_VARS[*]}")
echo "rollout_phase:writers_enabled"
gcloud run services update "${MCP_SERVICE_NAME}" \
  --region "${REGION}" --update-env-vars "^|^${DIALOG_ENV_VARS_JOINED}"
VECTOR_DIALOG_ENV="INTERNAL_DIALOG_OPERATIONS_URL=${VECTOR_DIALOG_URL}|DIALOG_OPERATIONS_QUEUE=${VECTOR_DIALOG_QUEUE}|CLOUD_TASKS_OIDC_AUDIENCE=${VECTOR_DIALOG_AUDIENCE}|CLOUD_TASKS_OIDC_SERVICE_ACCOUNT_EMAIL=${VECTOR_DIALOG_INVOKER}|REGION=${REGION}|GOOGLE_CLOUD_PROJECT=${PROJECT_ID}"
gcloud run services update "${VECTOR_SYNC_SERVICE_NAME:-tlbrain-vector-sync}" \
  --region "${REGION}" --update-env-vars "^|^${VECTOR_DIALOG_ENV}"

if [[ "${TELEGRAM_ENABLED}" == "1" ]]; then
  TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN_VALUE}" \
    TELEGRAM_ALLOWED_USER_IDS="${TELEGRAM_ALLOWED_USER_IDS_VALUE}" \
    bash infra/deploy/connectors/deploy_telegram.sh
else
  bash infra/deploy/connectors/telegram_deploy_common.sh off
fi

# =========================
# Deploy MCP (last — needs URLs from services above)
# =========================
# MCP was deployed in compatible-reader mode before guard consumers.

# =========================
# Final URLs
# =========================
MCP_URL=$(gcloud run services describe "${MCP_SERVICE_NAME}" \
  --region "${REGION}" \
  --format='value(status.url)')

VECTOR_SYNC_URL=$(gcloud run services describe "${VECTOR_SYNC_SERVICE_NAME}" \
  --region "${REGION}" \
  --format='value(status.url)')

CHECKER_URL=$(gcloud functions describe "${SYNC_CHECKER_NAME}" \
  --region "${REGION}" \
  --format='value(serviceConfig.uri)')

if [[ "${TLDV_ENABLED}" == "1" ]]; then
  WEBHOOK_URL=$(gcloud functions describe "${TLDV_WEBHOOK_FUNCTION_NAME}" \
    --region "${REGION}" \
    --format='value(serviceConfig.uri)')
fi

echo
echo "========================================="
echo "Deploy complete"
echo "========================================="
echo
echo "┌─ Connect to Claude ──────────────────────────────────────"
echo "│"
echo "│  1. Claude Settings → Integrations → Add Integration:"
echo "│     ${MCP_URL}/mcp"
echo "│"
echo "│  2. When prompted for OAuth credentials:"
echo "│     Client ID:     $(mask "${GOOGLE_CLIENT_ID}")"
echo "│     Client Secret: $(mask "${GOOGLE_CLIENT_SECRET}")"
echo "│"
echo "└──────────────────────────────────────────────────────────"
echo
if [[ "${TLDV_ENABLED}" == "1" ]]; then
  echo "┌─ TL;DV Webhook ──────────────────────────────────────────"
  echo "│"
  echo "│  TL;DV Settings → Webhooks → Add:"
  echo "│     POST ${WEBHOOK_URL}"
  echo "│"
  echo "└──────────────────────────────────────────────────────────"
  echo
fi
echo "Internal:"
echo "  Sync trigger:  POST ${VECTOR_SYNC_URL}/sync"
echo "  Checker:       ${CHECKER_URL}"
echo
