#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"
PY_HELPER="${SCRIPT_DIR}/firestore_indexes.py"
[[ -f "${REPO_ROOT}/infra/firestore/indexes.json" ]] || { echo "Missing Firestore index source: ${REPO_ROOT}/infra/firestore/indexes.json" >&2; exit 1; }

if [[ -n "${PYTHON:-}" ]]; then
  exec "${PYTHON}" "${PY_HELPER}"
fi
exec bash "${REPO_ROOT}/infra/build/preflight.sh" --python-mode "${PY_HELPER}"
