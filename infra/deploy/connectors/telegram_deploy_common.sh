#!/usr/bin/env bash

set +x
set -euo pipefail
umask 077

MODE=${1:?deployment mode required}
if [[ "${MODE}" != normal && "${MODE}" != local && "${MODE}" != off ]]; then
  echo "Unknown Telegram deployment mode" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

resolve_python() {
  local selected
  if ! selected=$(bash "${REPO_ROOT}/infra/build/preflight.sh" --python-mode -c 'import sys; print(sys.executable)'); then
    echo "Could not discover an installed Python interpreter" >&2
    return 1
  fi
  selected=${selected%$'\r'}
  if [[ "${selected:0:1}" == '"' && "${selected: -1}" == '"' ]]; then
    selected=${selected:1:${#selected}-2}
  fi
  if [[ "${selected}" =~ ^[A-Za-z]:[\\/].* ]] && command -v cygpath >/dev/null 2>&1; then
    selected=$(cygpath -u "${selected}")
  fi
  if [[ -z "${selected}" ]]; then
    echo "Could not discover an installed Python interpreter" >&2
    return 1
  fi
  PYTHON_EXECUTABLE=${selected}
}

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
if [[ "${MODE}" != off && -n "${BOT_TOKEN_CHECK}" && -z "${ALLOWED_USER_IDS_CHECK}" ]] || \
  [[ "${MODE}" != off && -z "${BOT_TOKEN_CHECK}" && -n "${ALLOWED_USER_IDS_CHECK}" ]]; then
  echo "TELEGRAM_BOT_TOKEN and TELEGRAM_ALLOWED_USER_IDS must be configured together" >&2
  exit 1
fi

PROJECT_ID=${PROJECT_ID:-tlbrain-prod}
REGION=${REGION:-europe-west1}
GOOGLE_CLOUD_PROJECT=${PROJECT_ID}
ROOT_FOLDER_URL=${ROOT_FOLDER_URL:-}
GEMINI_API_KEY=${GEMINI_API_KEY:-}
QDRANT_URL=${QDRANT_URL:-}
QDRANT_API_KEY=${QDRANT_API_KEY:-}
QDRANT_COLLECTION=${QDRANT_COLLECTION:-}
RETRIEVAL_TOP_K=${RETRIEVAL_TOP_K:-15}
RETRIEVAL_SCORE_THRESHOLD=${RETRIEVAL_SCORE_THRESHOLD:-0.6}
GOOGLE_REFRESH_TOKEN=${GOOGLE_REFRESH_TOKEN:-}
GOOGLE_CLIENT_ID=${GOOGLE_CLIENT_ID:-}
GOOGLE_CLIENT_SECRET=${GOOGLE_CLIENT_SECRET:-}
VECTOR_SYNC_SERVICE_NAME=${VECTOR_SYNC_SERVICE_NAME:-tlbrain-vector-sync}
SYNC_CHECKER_NAME=${SYNC_CHECKER_NAME:-tlbrain-sync-checker}
SERVICE_NAME=${TELEGRAM_SERVICE_NAME:-tlbrain-telegram}
REPOSITORY=${TELEGRAM_IMAGE_REPOSITORY:-inozem/tlbrain-telegram}
TASK_INVOKER_ACCOUNT_ID=${TELEGRAM_TASK_INVOKER_ACCOUNT_ID:-tlbrain-telegram-task-invoker}
TASK_INVOKER_EMAIL="${TASK_INVOKER_ACCOUNT_ID}@${PROJECT_ID}.iam.gserviceaccount.com"
DEPLOYER_ACCOUNT_ID=tlbrain-telegram-deployer
DEPLOYER_EMAIL="${DEPLOYER_ACCOUNT_ID}@${PROJECT_ID}.iam.gserviceaccount.com"
CORE_QUEUE=tlbrain-dialog-operations
CONTROL_QUEUE=tlbrain-telegram-control
RUNTIME_TARGET=services.connectors.telegram.app.main:app
PLATFORM=linux/amd64

if [[ "${MODE}" == off ]]; then
  if ! SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" --project "${PROJECT_ID}" \
    --region "${REGION}" --format='value(status.url)' 2>&1); then
    case "${SERVICE_URL}" in
      *NOT_FOUND*|*not\ found*|*does\ not\ exist*)
        echo "Telegram service is absent; nothing to disable"
        exit 0 ;;
      *)
        echo "Could not determine whether the Telegram service exists" >&2
        exit 1 ;;
    esac
  fi
  if [[ ! "${SERVICE_URL}" =~ ^https://[A-Za-z0-9.-]+(:[0-9]+)?$ ]]; then
    echo "Could not determine a valid deployed Telegram service URL" >&2
    exit 1
  fi
  if ! ID_TOKEN=$(gcloud auth print-identity-token --include-email --audiences="${SERVICE_URL}" \
    --impersonate-service-account="${DEPLOYER_EMAIL}" 2>/dev/null) || [[ -z "${ID_TOKEN}" ]]; then
    echo "Could not authenticate as the Telegram deployment identity; credentials were retained" >&2
    exit 1
  fi
  resolve_python
  if ! TELEGRAM_DISABLE_ID_TOKEN="${ID_TOKEN}" TELEGRAM_DISABLE_URL="${SERVICE_URL}/internal/disable-webhook" \
    "${PYTHON_EXECUTABLE}" -c 'import json, os, urllib.error, urllib.request
request = urllib.request.Request(os.environ["TELEGRAM_DISABLE_URL"], data=b"{}", method="POST", headers={"Authorization": "Bearer " + os.environ["TELEGRAM_DISABLE_ID_TOKEN"], "Content-Type": "application/json"})
try:
 response = urllib.request.urlopen(request, timeout=20)
 body = response.read(65537)
 if len(body) > 65536 or response.status >= 300: raise ValueError()
 result = json.loads(body.decode("utf-8"))
 if not isinstance(result, dict) or result.get("status") != "disabled": raise ValueError()
except Exception:
 raise SystemExit(1)'; then
    echo "Telegram webhook disable was not confirmed; credentials were retained" >&2
    exit 1
  fi
  if ! gcloud run services update "${SERVICE_NAME}" --project "${PROJECT_ID}" --region "${REGION}" \
    --remove-env-vars TELEGRAM_BOT_TOKEN,TELEGRAM_ALLOWED_USER_IDS >/dev/null; then
    echo "Webhook deletion was confirmed, but Telegram credentials could not be removed" >&2
    exit 1
  fi
  echo "Telegram webhook disabled and runtime credentials removed"
  exit 0
fi

if [[ -n "${BOT_TOKEN_CHECK}" ]]; then
  missing_runtime_config=()
  for setting in REGION GOOGLE_CLOUD_PROJECT ROOT_FOLDER_URL GEMINI_API_KEY QDRANT_URL \
    QDRANT_API_KEY QDRANT_COLLECTION RETRIEVAL_TOP_K RETRIEVAL_SCORE_THRESHOLD \
    GOOGLE_REFRESH_TOKEN GOOGLE_CLIENT_ID GOOGLE_CLIENT_SECRET; do
    setting_value=${!setting}
    setting_value=${setting_value//[[:space:]]/}
    if [[ -z "${setting_value}" ]]; then
      missing_runtime_config+=("${setting}")
    fi
  done
  if ((${#missing_runtime_config[@]})); then
    printf 'Telegram runtime configuration is missing: %s\n' "${missing_runtime_config[*]}" >&2
    exit 1
  fi
fi

TEMP_ENV_FILE=""
cleanup() {
  if [[ -n "${TEMP_ENV_FILE}" ]]; then
    rm -f -- "${TEMP_ENV_FILE}"
  fi
}
trap cleanup EXIT

if [[ -n "${BOT_TOKEN_CHECK}" ]]; then
  # Resolve shared service endpoints before any image build, push, or deployment.
VECTOR_SYNC_URL=$(gcloud run services describe "${VECTOR_SYNC_SERVICE_NAME}" \
  --project "${PROJECT_ID}" --region "${REGION}" --format='value(status.url)')
SYNC_CHECKER_URL=$(gcloud functions describe "${SYNC_CHECKER_NAME}" \
  --project "${PROJECT_ID}" --region "${REGION}" --format='value(serviceConfig.uri)')
if [[ ! "${VECTOR_SYNC_URL}" =~ ^https://[A-Za-z0-9.-]+(:[0-9]+)?$ ]] || \
  [[ ! "${SYNC_CHECKER_URL}" =~ ^https://[A-Za-z0-9.-]+(:[0-9]+)?$ ]]; then
  echo "Could not determine configured shared service URLs" >&2
  exit 1
fi

TEMP_ENV_FILE=$(mktemp "${TMPDIR:-/tmp}/tlbrain-telegram-env.XXXXXX")
chmod 600 "${TEMP_ENV_FILE}"
  resolve_python
# All runtime settings travel through a private JSON env file. Bot credentials
# are passed only to this writer and the later webhook-registration helper.
REGION="${REGION}" GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT}" \
  ROOT_FOLDER_URL="${ROOT_FOLDER_URL}" GEMINI_API_KEY="${GEMINI_API_KEY}" \
  QDRANT_URL="${QDRANT_URL}" QDRANT_API_KEY="${QDRANT_API_KEY}" \
  QDRANT_COLLECTION="${QDRANT_COLLECTION}" RETRIEVAL_TOP_K="${RETRIEVAL_TOP_K}" \
  RETRIEVAL_SCORE_THRESHOLD="${RETRIEVAL_SCORE_THRESHOLD}" \
  GOOGLE_REFRESH_TOKEN="${GOOGLE_REFRESH_TOKEN}" GOOGLE_CLIENT_ID="${GOOGLE_CLIENT_ID}" \
  GOOGLE_CLIENT_SECRET="${GOOGLE_CLIENT_SECRET}" VECTOR_SYNC_URL="${VECTOR_SYNC_URL}" \
  SYNC_CHECKER_URL="${SYNC_CHECKER_URL}" DIALOG_OPERATIONS_QUEUE="${CORE_QUEUE}" \
  TELEGRAM_BOT_TOKEN="${BOT_TOKEN}" \
  TELEGRAM_ALLOWED_USER_IDS="${ALLOWED_USER_IDS}" \
  "${PYTHON_EXECUTABLE}" -c 'import json, os, sys; names = ("REGION", "GOOGLE_CLOUD_PROJECT", "ROOT_FOLDER_URL", "GEMINI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY", "QDRANT_COLLECTION", "RETRIEVAL_TOP_K", "RETRIEVAL_SCORE_THRESHOLD", "GOOGLE_REFRESH_TOKEN", "GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "VECTOR_SYNC_URL", "SYNC_CHECKER_URL", "DIALOG_OPERATIONS_QUEUE"); data = {name: os.environ[name] for name in names};
if os.environ.get("TELEGRAM_BOT_TOKEN"): data.update(TELEGRAM_BOT_TOKEN=os.environ["TELEGRAM_BOT_TOKEN"], TELEGRAM_ALLOWED_USER_IDS=os.environ["TELEGRAM_ALLOWED_USER_IDS"])
json.dump(data, open(sys.argv[1], "w", encoding="utf-8"))' "${TEMP_ENV_FILE}"
fi

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
if ! gcloud iam service-accounts describe "${DEPLOYER_EMAIL}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud iam service-accounts create "${DEPLOYER_ACCOUNT_ID}" --project "${PROJECT_ID}" \
    --display-name="TLBrain Telegram deployment disable identity"
fi
DEPLOY_CALLER=$(gcloud config get-value account 2>/dev/null)
if [[ -z "${DEPLOY_CALLER}" || "${DEPLOY_CALLER}" == "(unset)" ]]; then
  echo "Could not determine the deployment caller identity" >&2
  exit 1
fi
case "${DEPLOY_CALLER}" in
  *@*.iam.gserviceaccount.com) DEPLOY_CALLER_MEMBER="serviceAccount:${DEPLOY_CALLER}" ;;
  *) DEPLOY_CALLER_MEMBER="user:${DEPLOY_CALLER}" ;;
esac

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
gcloud run services add-iam-policy-binding "${SERVICE_NAME}" --project "${PROJECT_ID}" \
  --region "${REGION}" --member="serviceAccount:${DEPLOYER_EMAIL}" --role=roles/run.invoker
gcloud iam service-accounts add-iam-policy-binding "${DEPLOYER_EMAIL}" --project "${PROJECT_ID}" \
  --member="${DEPLOY_CALLER_MEMBER}" --role=roles/iam.serviceAccountTokenCreator
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
  "OIDC_SERVICE_ACCOUNT_EMAIL=${TASK_INVOKER_EMAIL}"
  "TELEGRAM_DISABLE_OIDC_SERVICE_ACCOUNT_EMAIL=${DEPLOYER_EMAIL}")
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
  GOOGLE_CLOUD_PROJECT="${PROJECT_ID}" TELEGRAM_BOT_TOKEN="${BOT_TOKEN}" \
    TELEGRAM_WEBHOOK_URL="${SERVICE_URL}/telegram/webhook" \
    "${PYTHON_EXECUTABLE}" "${REPO_ROOT}/infra/build/bootstrap_runner.py" --run-helper \
    --source-root "${REPO_ROOT}" \
    --requirements "${REPO_ROOT}/core/integrations/firestore/requirements.txt" \
    --helper-code 'import os; from infra.deploy.connectors.telegram.webhook_admin import create_deployment_firestore_client, register_telegram_webhook; ok = False
try: ok = register_telegram_webhook(client=create_deployment_firestore_client(project=os.environ["GOOGLE_CLOUD_PROJECT"]), bot_token=os.environ["TELEGRAM_BOT_TOKEN"], webhook_url=os.environ["TELEGRAM_WEBHOOK_URL"])
except Exception: pass
print("Telegram webhook registration succeeded" if ok else "Telegram webhook registration failed"); raise SystemExit(0 if ok else 1)'
fi

echo "Telegram service deployed: ${SERVICE_URL}"
