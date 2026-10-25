#!/usr/bin/env bash

set -euo pipefail

PROJECT_ID=${PROJECT_ID:-tlbrain-prod}
REGION=${REGION:-europe-west1}
MCP_SERVICE_NAME=${MCP_SERVICE_NAME:-tlbrain-mcp}
QUEUE_NAME=tlbrain-dialog-operations
TASK_INVOKER_ACCOUNT_ID=${DIALOG_OPERATIONS_TASK_INVOKER_ACCOUNT_ID:-tlbrain-dialog-operations-invoker}

usage() {
  echo "Usage: $0 <deploy-confirmed-mcp-base-url>" >&2
  exit 2
}

[[ $# -eq 1 ]] || usage
MCP_BASE_URL=${1%/}
# Only accept an HTTPS origin/base URL, never a callback path, query, fragment,
# credentials, or a value containing whitespace supplied from a request.
if [[ ! "${MCP_BASE_URL}" =~ ^https://[A-Za-z0-9.-]+(:[0-9]+)?$ ]]; then
  echo "Invalid deploy-confirmed MCP base URL: ${MCP_BASE_URL}" >&2
  exit 2
fi
CALLBACK_URL="${MCP_BASE_URL}/internal/dialog-operations"
OIDC_AUDIENCE=${MCP_BASE_URL}

TASK_INVOKER_EMAIL="${TASK_INVOKER_ACCOUNT_ID}@${PROJECT_ID}.iam.gserviceaccount.com"

# Discover the runtime identity and project number from deployed Google Cloud
# resources. These values are never derived from request or Host headers.
RUNTIME_SERVICE_ACCOUNT=$(gcloud run services describe "${MCP_SERVICE_NAME}" \
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
  gcloud iam service-accounts create "${TASK_INVOKER_ACCOUNT_ID}" \
    --project "${PROJECT_ID}" \
    --display-name="TLBrain dialog operations task invoker"
fi

if gcloud tasks queues describe "${QUEUE_NAME}" --project "${PROJECT_ID}" \
  --location "${REGION}" >/dev/null 2>&1; then
  gcloud tasks queues update "${QUEUE_NAME}" --project "${PROJECT_ID}" \
    --location "${REGION}" --max-concurrent-dispatches=2
else
  gcloud tasks queues create "${QUEUE_NAME}" --project "${PROJECT_ID}" \
    --location "${REGION}" --max-concurrent-dispatches=2
fi

# The task invoker can call only the target MCP service. Cloud Tasks may mint
# OIDC tokens only for this dedicated identity; the MCP runtime may enqueue on
# this queue and act as this identity.
gcloud run services add-iam-policy-binding "${MCP_SERVICE_NAME}" \
  --project "${PROJECT_ID}" --region "${REGION}" \
  --member="serviceAccount:${TASK_INVOKER_EMAIL}" --role=roles/run.invoker
gcloud iam service-accounts add-iam-policy-binding "${TASK_INVOKER_EMAIL}" \
  --project "${PROJECT_ID}" --member="serviceAccount:${TASKS_SERVICE_AGENT}" \
  --role=roles/iam.serviceAccountTokenCreator
gcloud tasks queues add-iam-policy-binding "${QUEUE_NAME}" \
  --project "${PROJECT_ID}" --location "${REGION}" \
  --member="serviceAccount:${RUNTIME_SERVICE_ACCOUNT}" --role=roles/cloudtasks.enqueuer
gcloud iam service-accounts add-iam-policy-binding "${TASK_INVOKER_EMAIL}" \
  --project "${PROJECT_ID}" --member="serviceAccount:${RUNTIME_SERVICE_ACCOUNT}" \
  --role=roles/iam.serviceAccountUser
# Firestore data access only; no project owner/editor or datastore admin role.
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${RUNTIME_SERVICE_ACCOUNT}" --role=roles/datastore.user \
  --condition=None

printf 'INTERNAL_DIALOG_OPERATIONS_URL=%s\n' "${CALLBACK_URL}"
printf 'DIALOG_OPERATIONS_QUEUE=%s\n' "${QUEUE_NAME}"
printf 'CLOUD_TASKS_OIDC_AUDIENCE=%s\n' "${OIDC_AUDIENCE}"
printf 'CLOUD_TASKS_OIDC_SERVICE_ACCOUNT_EMAIL=%s\n' "${TASK_INVOKER_EMAIL}"
printf 'OIDC_AUDIENCE=%s\n' "${OIDC_AUDIENCE}"
printf 'OIDC_SERVICE_ACCOUNT_EMAIL=%s\n' "${TASK_INVOKER_EMAIL}"
