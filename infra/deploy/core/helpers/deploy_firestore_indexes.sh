#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../" && pwd)"
INDEX_FILE="${REPO_ROOT}/infra/firestore/indexes.json"
PROJECT_ID=${PROJECT_ID:-tlbrain-prod}
[[ -f "${INDEX_FILE}" ]] || { echo "Missing Firestore index source: ${INDEX_FILE}" >&2; exit 1; }

INDEX_ROWS=$("${PYTHON:-python}" -c 'import json,sys; d=json.load(open(sys.argv[1],encoding="utf-8")); [print("\t".join((i["collectionGroup"],i.get("queryScope","COLLECTION"),"|".join("field-path=%s,order=%s"%(f["fieldPath"],f["order"].lower()) if "order" in f else "field-path=%s,array-config=contains"%f["fieldPath"] for f in i.get("fields",[]))))) for i in d.get("indexes",[])]' "${INDEX_FILE}")
while IFS=$'\t' read -r collection_group query_scope encoded_fields; do
  [[ -n "${collection_group}" ]] || continue
  encoded_fields=${encoded_fields%$'\r'}
  IFS='|' read -r -a field_definitions <<< "${encoded_fields}"
  command=(gcloud firestore indexes composite create --project "${PROJECT_ID}" --database='(default)' --collection-group "${collection_group}" --query-scope="${query_scope,,}")
  for field_definition in "${field_definitions[@]}"; do
    command+=(--field-config="${field_definition}")
  done
  if error=$("${command[@]}" 2>&1); then
    :
  else
    status=$?
    if [[ "${error}" =~ [Aa][Ll][Rr][Ee][Aa][Dd][Yy][_[:space:]]?[Ee][Xx][Ii][Ss][Tt][Ss] ]]; then
      continue
    fi
    printf '%s\n' "${error}" >&2
    exit "${status}"
  fi
done <<< "${INDEX_ROWS}"

EXEMPTION_ROWS=$("${PYTHON:-python}" -c 'import json,sys; d=json.load(open(sys.argv[1],encoding="utf-8")); [print("\t".join((o["collectionGroup"],o["fieldPath"]))) for o in d.get("fieldOverrides",[]) if not o.get("indexes")]' "${INDEX_FILE}")
while IFS=$'\t' read -r collection_group field_path; do
  [[ -n "${collection_group}" ]] || continue
  field_path=${field_path%$'\r'}
  gcloud firestore indexes fields update "${field_path}" --project "${PROJECT_ID}" --database='(default)' --collection-group="${collection_group}" --disable-indexes
done <<< "${EXEMPTION_ROWS}"

