"""Offline checks for Function deploy source assembly and preflight gating."""

from __future__ import annotations

import re
from pathlib import Path

import pytest


APP_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_SCRIPTS = {
    "sync_checker": APP_ROOT / "infra/deploy/core/deploy_sync_checker.sh",
    "tldv_webhook": APP_ROOT / "infra/deploy/connectors/tldv/deploy_webhook.sh",
    "tldv_reconciliation": APP_ROOT / "infra/deploy/connectors/tldv/deploy_reconciliation.sh",
}


@pytest.mark.parametrize("service_id,script_path", DEPLOY_SCRIPTS.items())
def test_function_deploy_uses_preflight_artifact_and_fails_closed(
    service_id: str, script_path: Path
) -> None:
    source = script_path.read_text(encoding="utf-8")
    artifact = f'${{REPO_ROOT}}/.build/{service_id}'
    assert f'ARTIFACT="{artifact}"' in source
    preflight_at = source.index('PREFLIGHT_RESULT=$("${REPO_ROOT}/infra/build/preflight.sh"')
    preflight_end = source.index('  --output "${ARTIFACT}")', preflight_at)
    preflight = source[preflight_at:preflight_end]
    assert f"--service {service_id}" in preflight
    assert '--source-root "${REPO_ROOT}"' in preflight
    assert '--output "${ARTIFACT}"' in source[preflight_end:preflight_end + 64]

    fingerprint_parse = source.index(
        'ARTIFACT_FINGERPRINT=$(printf \'%s\\n\' "${PREFLIGHT_RESULT}" | sed -n \'s/^Artifact fingerprint: //p\')'
    )
    verify_at = source.index('preflight.sh" --verify-source \\', fingerprint_parse)
    verify_end = source.index('  --fingerprint "${ARTIFACT_FINGERPRINT}"', verify_at)
    deploy_at = source.index("gcloud functions deploy ", verify_end)
    assert preflight_at < fingerprint_parse < verify_at < verify_end < deploy_at
    verify = source[verify_at:verify_end + len('  --fingerprint "${ARTIFACT_FINGERPRINT}"')]
    assert '--artifact "${ARTIFACT}"' in verify

    deploy = source[deploy_at:].split("\n\n", 1)[0]
    assert '--source "${ARTIFACT}"' in deploy
    assert "--runtime python312" in deploy
    assert re.search(rf"--entry-point {re.escape(_ENTRY_POINTS[service_id])}", deploy)
    assert "set -euo pipefail" in source
    assert not re.search(r"^\s*(?:cp|cat)\s", source, re.MULTILINE)
    assert "mktemp -d" not in source


_ENTRY_POINTS = {
    "sync_checker": "checker",
    "tldv_webhook": "tldv_webhook",
    "tldv_reconciliation": "tldv_reconciliation",
}


@pytest.mark.parametrize(
    "service_id,required_fragments",
    [
        (
            "sync_checker",
            [
                'SYNC_CHECKER_NAME=${SYNC_CHECKER_NAME:-tlbrain-sync-checker}',
                'VECTOR_SYNC_QUEUE=${VECTOR_SYNC_QUEUE:-tlbrain-vector-sync-queue}',
                'VECTOR_SYNC_SERVICE_NAME=${VECTOR_SYNC_SERVICE_NAME:-tlbrain-vector-sync}',
                'SYNC_CHECKER_SCHEDULE=${SYNC_CHECKER_SCHEDULE:-"*/15 * * * *"}',
                "--trigger-http",
                "--allow-unauthenticated",
                "--max-instances=1",
                "ROOT_FOLDER_URL=",
                "VECTOR_SYNC_URL=",
                "VECTOR_SYNC_QUEUE=",
                'gcloud scheduler jobs create http "${SYNC_CHECKER_NAME}-schedule"',
                'gcloud scheduler jobs update http "${SYNC_CHECKER_NAME}-schedule"',
            ],
        ),
        (
            "tldv_webhook",
            [
                'TLDV_WEBHOOK_FUNCTION_NAME=${TLDV_WEBHOOK_FUNCTION_NAME:-tlbrain-tldv-webhook}',
                'TLDV_IMPORT_QUEUE=${TLDV_IMPORT_QUEUE:-tlbrain-tldv-import-queue}',
                'TLDV_IMPORT_SERVICE_NAME=${TLDV_IMPORT_SERVICE_NAME:-tlbrain-tldv-import}',
                "--trigger-http",
                "--allow-unauthenticated",
                "TLDV_IMPORT_QUEUE=",
                "TLDV_IMPORT_SERVICE_URL=",
            ],
        ),
        (
            "tldv_reconciliation",
            [
                'TLDV_RECONCILIATION_FUNCTION_NAME=${TLDV_RECONCILIATION_FUNCTION_NAME:-tlbrain-tldv-reconciliation}',
                'TLDV_IMPORT_QUEUE=${TLDV_IMPORT_QUEUE:-tlbrain-tldv-import-queue}',
                'TLDV_IMPORT_SERVICE_NAME=${TLDV_IMPORT_SERVICE_NAME:-tlbrain-tldv-import}',
                'TLDV_RECONCILIATION_SCHEDULE=${TLDV_RECONCILIATION_SCHEDULE:-"0 3 * * *"}',
                "TLDV_API_KEY=${TLDV_API_KEY:-}",
                "TLDV_API_KEY=",
                "--trigger-http",
                "--allow-unauthenticated",
                'gcloud scheduler jobs create http "${TLDV_RECONCILIATION_FUNCTION_NAME}-schedule"',
                'gcloud scheduler jobs update http "${TLDV_RECONCILIATION_FUNCTION_NAME}-schedule"',
            ],
        ),
    ],
)
def test_function_resource_and_schedule_settings_are_preserved(
    service_id: str, required_fragments: list[str]
) -> None:
    source = DEPLOY_SCRIPTS[service_id].read_text(encoding="utf-8")
    assert 'PROJECT_ID=${PROJECT_ID:-tlbrain-prod}' in source
    assert 'REGION=${REGION:-europe-west1}' in source
    assert all(fragment in source for fragment in required_fragments)

