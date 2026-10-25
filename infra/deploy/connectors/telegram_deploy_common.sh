#!/usr/bin/env bash

set +x
set -euo pipefail
umask 077

MODE=${1:?deployment mode required}
if [[ "${MODE}" != normal && "${MODE}" != local ]]; then
  echo "Unknown Telegram deployment mode" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

# Load the local configuration before validating it, but never enable shell tracing.
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

BOT_TOKEN=${TELEGRAM_BOT_TOKEN:-}
ALLOWED_USER_IDS=${TELEGRAM_ALLOWED_USER_IDS:-}
# These values may have arrived exported from the caller or through `set -a`
# while sourcing .env. Keep them in the shell and pass them only to the two
# Python commands that consume them.
export -n TELEGRAM_BOT_TOKEN TELEGRAM_ALLOWED_USER_IDS 2>/dev/null || true
BOT_TOKEN_CHECK=${BOT_TOKEN//[[:space:]]/}
ALLOWED_USER_IDS_CHECK=${ALLOWED_USER_IDS//[[:space:]]/}
if [[ -n "${BOT_TOKEN_CHECK}" && -z "${ALLOWED_USER_IDS_CHECK}" ]] || \
  [[ -z "${BOT_TOKEN_CHECK}" && -n "${ALLOWED_USER_IDS_CHECK}" ]]; then
  echo "TELEGRAM_BOT_TOKEN and TELEGRAM_ALLOWED_USER_IDS must be configured together" >&2
  exit 1
fi

PROJECT_ID=${PROJECT_ID:-tlbrain-prod}
REGION=${REGION:-europe-west1}
SERVICE_NAME=${TELEGRAM_SERVICE_NAME:-tlbrain-telegram}
REPOSITORY=${TELEGRAM_IMAGE_REPOSITORY:-inozem/tlbrain-telegram}
TASK_INVOKER_ACCOUNT_ID=${TELEGRAM_TASK_INVOKER_ACCOUNT_ID:-tlbrain-telegram-task-invoker}
TASK_INVOKER_EMAIL="${TASK_INVOKER_ACCOUNT_ID}@${PROJECT_ID}.iam.gserviceaccount.com"
CORE_QUEUE=tlbrain-dialog-operations
CONTROL_QUEUE=tlbrain-telegram-control
RUNTIME_TARGET=services.connectors.telegram.app.main:app
PLATFORM=linux/amd64

TEMP_ENV_FILE=""
cleanup() {
  if [[ -n "${TEMP_ENV_FILE}" ]]; then
    rm -f -- "${TEMP_ENV_FILE}"
  fi
}
trap cleanup EXIT

if [[ "${MODE}" == normal ]]; then
  IMAGE_DIGEST=$(bash "${REPO_ROOT}/infra/build/preflight.sh" --verify-mode \
    --repository "${REPOSITORY}" --version "${VERSION:?VERSION is required}" \
    --service telegram --target "${RUNTIME_TARGET}" --platform "${PLATFORM}" \
    --required-check compile --required-check imports --required-check startup)
  IMAGE="${REPOSITORY}@${IMAGE_DIGEST}"
else
  IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"
  docker build --platform "${PLATFORM}" -f "${REPO_ROOT}/infra/docker/Dockerfile.telegram" \
    -t "${IMAGE}" "${REPO_ROOT}"
  bash "${REPO_ROOT}/infra/build/preflight.sh" --image-mode --image "${IMAGE}" \
    --service telegram --platform "${PLATFORM}"
  docker push "${IMAGE}"
fi

if [[ -n "${BOT_TOKEN_CHECK}" ]]; then
  TEMP_ENV_FILE=$(mktemp "${TMPDIR:-/tmp}/tlbrain-telegram-env.XXXXXX")
  chmod 600 "${TEMP_ENV_FILE}"
  # JSON is a valid YAML env-vars file. Values travel to the writer via its
  # environment, never through argv or stdout.
  TELEGRAM_BOT_TOKEN="${BOT_TOKEN}" TELEGRAM_ALLOWED_USER_IDS="${ALLOWED_USER_IDS}" \
    python3 -c 'import json, os, sys; json.dump({"TELEGRAM_BOT_TOKEN": os.environ["TELEGRAM_BOT_TOKEN"], "TELEGRAM_ALLOWED_USER_IDS": os.environ["TELEGRAM_ALLOWED_USER_IDS"]}, open(sys.argv[1], "w", encoding="utf-8"))' "${TEMP_ENV_FILE}"
fi

DEPLOY_ARGS=(run deploy "${SERVICE_NAME}" --project "${PROJECT_ID}" --image "${IMAGE}"
  --region "${REGION}" --platform managed --allow-unauthenticated
  --min 0 --max 2 --concurrency 8 --timeout 540)
if [[ -n "${TEMP_ENV_FILE}" ]]; then
  DEPLOY_ARGS+=(--env-vars-file "${TEMP_ENV_FILE}")
fi
gcloud "${DEPLOY_ARGS[@]}"

SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" --project "${PROJECT_ID}" \
  --region "${REGION}" --format='value(status.url)')
if [[ ! "${SERVICE_URL}" =~ ^https://[A-Za-z0-9.-]+(:[0-9]+)?$ ]]; then
  echo "Could not determine a valid deployed Telegram service URL" >&2
  exit 1
fi

RUNTIME_SERVICE_ACCOUNT=$(gcloud run services describe "${SERVICE_NAME}" \
  --project "${PROJECT_ID}" --region "${REGION}" \
  --format='value(spec.template.spec.serviceAccountName)')
PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')
if [[ -z "${PROJECT_NUMBER}" ]]; then
  echo "Could not determine project number" >&2
  exit 1
fi
if [[ -z "${RUNTIME_SERVICE_ACCOUNT}" ]]; then
  RUNTIME_SERVICE_ACCOUNT="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
fi
TASKS_SERVICE_AGENT="service-${PROJECT_NUMBER}@gcp-sa-cloudtasks.iam.gserviceaccount.com"

if ! gcloud iam service-accounts describe "${TASK_INVOKER_EMAIL}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud iam service-accounts create "${TASK_INVOKER_ACCOUNT_ID}" --project "${PROJECT_ID}" \
    --display-name="TLBrain Telegram task invoker"
fi

for queue_spec in "${CORE_QUEUE}:2" "${CONTROL_QUEUE}:4"; do
  queue=${queue_spec%%:*}
  concurrency=${queue_spec##*:}
  if gcloud tasks queues describe "${queue}" --project "${PROJECT_ID}" \
    --location "${REGION}" >/dev/null 2>&1; then
    gcloud tasks queues update "${queue}" --project "${PROJECT_ID}" --location "${REGION}" \
      --max-concurrent-dispatches="${concurrency}"
  else
    gcloud tasks queues create "${queue}" --project "${PROJECT_ID}" --location "${REGION}" \
      --max-concurrent-dispatches="${concurrency}"
  fi
done

# The task identity is scoped to this service; the runtime can enqueue only on
# its two required queues and can act as only this task identity.
gcloud run services add-iam-policy-binding "${SERVICE_NAME}" --project "${PROJECT_ID}" \
  --region "${REGION}" --member="serviceAccount:${TASK_INVOKER_EMAIL}" --role=roles/run.invoker
gcloud iam service-accounts add-iam-policy-binding "${TASK_INVOKER_EMAIL}" \
  --project "${PROJECT_ID}" --member="serviceAccount:${TASKS_SERVICE_AGENT}" \
  --role=roles/iam.serviceAccountTokenCreator
for queue in "${CORE_QUEUE}" "${CONTROL_QUEUE}"; do
  gcloud tasks queues add-iam-policy-binding "${queue}" --project "${PROJECT_ID}" \
    --location "${REGION}" --member="serviceAccount:${RUNTIME_SERVICE_ACCOUNT}" \
    --role=roles/cloudtasks.enqueuer
done
gcloud iam service-accounts add-iam-policy-binding "${TASK_INVOKER_EMAIL}" \
  --project "${PROJECT_ID}" --member="serviceAccount:${RUNTIME_SERVICE_ACCOUNT}" \
  --role=roles/iam.serviceAccountUser
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${RUNTIME_SERVICE_ACCOUNT}" --role=roles/datastore.user --condition=None

OIDC_AUDIENCE=${SERVICE_URL}
SETTINGS=("TELEGRAM_CONTROL_URL=${SERVICE_URL}/internal/dispatch"
  "TELEGRAM_DELIVERY_URL=${SERVICE_URL}/internal/deliver"
  "TELEGRAM_CONTROL_QUEUE=${CONTROL_QUEUE}"
  "CLOUD_TASKS_OIDC_AUDIENCE=${OIDC_AUDIENCE}"
  "CLOUD_TASKS_OIDC_SERVICE_ACCOUNT_EMAIL=${TASK_INVOKER_EMAIL}"
  "OIDC_AUDIENCE=${OIDC_AUDIENCE}"
  "OIDC_SERVICE_ACCOUNT_EMAIL=${TASK_INVOKER_EMAIL}")
TZ_VALUE=${TZ:-}
TZ_CHECK=${TZ_VALUE//[[:space:]]/}
if [[ -n "${TZ_CHECK}" ]]; then
  SETTINGS+=("TZ=${TZ_VALUE}")
fi
SETTINGS_JOINED=$(IFS='|'; printf '%s' "${SETTINGS[*]}")
gcloud run services update "${SERVICE_NAME}" --project "${PROJECT_ID}" --region "${REGION}" \
  --update-env-vars "^|^${SETTINGS_JOINED}"

if [[ -n "${BOT_TOKEN_CHECK}" ]]; then
  # The helper returns only a boolean and deliberately suppresses provider details.
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    TELEGRAM_BOT_TOKEN="${BOT_TOKEN}" TELEGRAM_WEBHOOK_URL="${SERVICE_URL}/telegram/webhook" python3 -c \
    'import os; from core.integrations.firestore.client import get_client; from infra.deploy.connectors.telegram.webhook_admin import register_telegram_webhook; ok = False
try: ok = register_telegram_webhook(client=get_client(), bot_token=os.environ["TELEGRAM_BOT_TOKEN"], webhook_url=os.environ["TELEGRAM_WEBHOOK_URL"])
except Exception: pass
print("Telegram webhook registration succeeded" if ok else "Telegram webhook registration failed"); raise SystemExit(0 if ok else 1)'
fi

echo "Telegram service deployed: ${SERVICE_URL}"
