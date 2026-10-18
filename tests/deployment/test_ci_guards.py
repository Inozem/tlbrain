"""Static safeguards for the deployment validation workflow."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


APP_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = APP_ROOT / ".github" / "workflows" / "build.yml"
REQUIRED_PATHS = {
    "core/**",
    "services/**",
    "infra/**",
    "tests/**",
    "migrations/**",
    "requirements*.txt",
    "**/requirements*.txt",
    "pytest.ini",
    ".github/workflows/**",
}
SOURCE_SERVICES = {
    "mcp",
    "vector_sync",
    "tldv_import",
    "sync_checker",
    "tldv_webhook",
    "tldv_reconciliation",
}
IMAGE_DOCKERFILES = {
    "mcp": "infra/docker/Dockerfile.mcp",
    "vector_sync": "infra/docker/Dockerfile.vector_sync",
    "tldv_import": "infra/docker/Dockerfile.tldv_import",
}
PUBLISH_JOBS = {
    "build-mcp": ("mcp", "inozem/tlbrain-mcp"),
    "build-vector-sync": ("vector_sync", "inozem/tlbrain-vector-sync"),
    "build-tldv-import": ("tldv_import", "inozem/tlbrain-tldv-import"),
}
GATES = {
    "offline-regression",
    "source-validation",
    "image-validation",
    "native-bootstrap-smoke",
}


def _workflow() -> dict[str, object]:
    """Use BaseLoader so YAML 1.1 does not coerce the `on` key to True."""
    return yaml.load(WORKFLOW_PATH.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _steps(job: dict[str, object]) -> list[dict[str, object]]:
    return job["steps"]


def _requirement_include_closure(requirements_file: Path) -> set[Path]:
    included: set[Path] = set()
    pending = [requirements_file.resolve()]
    while pending:
        current = pending.pop()
        if current in included:
            continue
        included.add(current)
        for raw_line in current.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] in {"-r", "--requirement"}:
                assert len(parts) == 2, f"unsupported requirements include: {line}"
                pending.append((current.parent / parts[1]).resolve())
    return included


def test_workflow_keeps_on_as_a_mapping_and_validates_relevant_changes() -> None:
    workflow = _workflow()
    triggers = workflow["on"]
    assert isinstance(triggers, dict)
    assert set(triggers) == {"pull_request", "push"}

    pull_request = triggers["pull_request"]
    push = triggers["push"]
    assert pull_request["branches"] == ["main"]
    assert set(pull_request["paths"]) == REQUIRED_PATHS
    assert push["branches"] == ["main"]
    assert set(push["paths"]) == REQUIRED_PATHS
    assert push["tags"] == ["v*"]


def test_source_validation_matrix_covers_every_supported_service() -> None:
    job = _workflow()["jobs"]["source-validation"]
    assert job["runs-on"] == "ubuntu-latest"
    assert job["strategy"]["matrix"]["service"] == sorted(SOURCE_SERVICES, key=("mcp", "vector_sync", "tldv_import", "sync_checker", "tldv_webhook", "tldv_reconciliation").index)
    assert any(step.get("with", {}).get("python-version") == "3.12" for step in _steps(job))
    invocation = " ".join(str(step.get("run", "")) for step in _steps(job))
    assert "infra/build/preflight.sh" in invocation
    assert "--service \"${{ matrix.service }}\"" in invocation
    assert "--source-root" in invocation
    assert "--output" in invocation


def test_image_matrix_builds_and_gates_each_final_image_before_capture() -> None:
    job = _workflow()["jobs"]["image-validation"]
    assert job["runs-on"] == "ubuntu-latest"
    rows = job["strategy"]["matrix"]["include"]
    assert {row["service"]: row["dockerfile"] for row in rows} == IMAGE_DOCKERFILES
    steps = _steps(job)
    names = [step.get("name") for step in steps]
    build_index = names.index("Build final Linux amd64 image")
    gate_index = names.index("Validate actual final image")
    save_index = names.index("Save gated image")
    upload_index = names.index("Upload gated image artifact")
    assert build_index < gate_index < save_index < upload_index

    build_command = steps[build_index]["run"]
    assert "docker build --platform linux/amd64" in build_command
    assert "${{ matrix.dockerfile }}" in build_command
    gate_command = steps[gate_index]["run"]
    assert "preflight.sh --image-mode" in gate_command
    assert "--image \"tlbrain-gated:${{ matrix.service }}\"" in gate_command
    assert "--service \"${{ matrix.service }}\"" in gate_command
    assert "--platform linux/amd64" in gate_command
    assert "docker save \"tlbrain-gated:${{ matrix.service }}\"" in steps[save_index]["run"]
    assert steps[upload_index]["uses"] == "actions/upload-artifact@v4"
    assert steps[upload_index]["with"]["name"] == "gated-image-${{ matrix.service }}"


def test_all_publish_jobs_wait_for_gates_and_push_only_gated_artifacts() -> None:
    jobs = _workflow()["jobs"]
    for name, (service, repository) in PUBLISH_JOBS.items():
        job = jobs[name]
        assert set(job["needs"]) == GATES
        assert job["if"] == "github.event_name == 'push'"
        steps = _steps(job)
        uses = {step.get("uses") for step in steps}
        assert "actions/checkout@v4" in uses
        assert "docker/login-action@v3" in uses
        assert "docker/metadata-action@v5" in uses

        metadata = next(step for step in steps if step.get("uses") == "docker/metadata-action@v5")
        assert metadata["with"]["images"] == repository
        tags = metadata["with"]["tags"]
        assert "type=sha" in tags
        assert "type=raw,value=latest,enable=${{ github.ref == 'refs/heads/main' }}" in tags
        assert "type=semver,pattern={{version}}" in tags

        download = next(step for step in steps if step.get("uses") == "actions/download-artifact@v4")
        assert download["with"]["name"] == f"gated-image-{service}"
        publish = next(step for step in steps if step.get("name") == "Load, tag, and push gated image")
        command = publish["run"]
        assert f"docker load --input \"${{{{ runner.temp }}}}/gated-image/{service}.tar\"" in command
        assert f"docker tag \"tlbrain-gated:{service}\" \"$tag\"" in command
        assert "docker push \"$tag\"" in command
        assert "docker build" not in command
        assert not any(step.get("uses") == "docker/build-push-action@v5" for step in steps)


def test_offline_regression_uses_declared_dev_requirements() -> None:
    job = _workflow()["jobs"]["offline-regression"]
    steps = _steps(job)
    assert job["runs-on"] == "ubuntu-latest"
    assert any(step.get("with", {}).get("python-version") == "3.12" for step in steps)
    install = next(step for step in steps if step.get("name") == "Install declared development requirements")
    assert "-r requirements-dev.txt" in install["run"]
    regression = next(step for step in steps if step.get("name") == "Run offline regression suite")
    assert regression["run"] == "python -m pytest -q"


def test_dev_requirements_include_all_service_and_provider_roots() -> None:
    included = _requirement_include_closure(APP_ROOT / "requirements-dev.txt")
    manifests = json.loads((APP_ROOT / "infra" / "build" / "service-manifests.json").read_text(encoding="utf-8"))
    providers = json.loads((APP_ROOT / "infra" / "build" / "dependency-providers.json").read_text(encoding="utf-8"))
    required = {
        (APP_ROOT / path).resolve()
        for service in manifests["services"].values()
        for path in service["requirements_roots"]
    }
    required.update(
        (APP_ROOT / path).resolve()
        for provider in providers["providers"]
        for path in provider["requirements_files"]
    )
    assert required <= included, sorted(path.relative_to(APP_ROOT).as_posix() for path in required - included)


def test_native_smoke_uses_only_synthetic_bootstrap_checks_and_one_function_preflight() -> None:
    job = _workflow()["jobs"]["native-bootstrap-smoke"]
    assert job["runs-on"] == "${{ matrix.os }}"
    assert set(job["strategy"]["matrix"]["os"]) == {
        "ubuntu-latest", "macos-latest", "windows-latest",
    }
    steps = _steps(job)
    assert any(step.get("with", {}).get("python-version") == "3.12" for step in steps)
    install = next(step for step in steps if step.get("name") == "Install smoke test runner")
    assert '"pytest>=8.0"' in install["run"]
    assert "requirements-dev.txt" not in install["run"]
    bootstrap = next(step for step in steps if step.get("name") == "Exercise synthetic refresh-token gcloud fallback")
    command = bootstrap["run"]
    assert "tests/deployment/test_runner_bootstrap.py" in command
    assert "--confcutdir=tests/deployment" in command
    assert "discovery_fallback_uses_only_gcloud_python_location" in command
    assert "discovery_refresh_token_uses_gcloud_fallback_without_setup_and_preserves_inputs" in command
    compatibility = next(
        step for step in steps
        if step.get("name") == "Exercise runner compatibility through ordinary deploy routes"
    )
    assert compatibility["env"]["TLBRAIN_RUN_BASH_TESTS_ON_WINDOWS"] == "1"
    assert "tests/deployment/test_runner_compatibility.py" in compatibility["run"]
    assert "--confcutdir=tests/deployment" in compatibility["run"]
    preflight = next(step for step in steps if step.get("name") == "Validate a lightweight Function on the prepared host")
    assert preflight["shell"] == "bash"
    assert "--service sync_checker" in preflight["run"]
    assert "--source-root" in preflight["run"]
    assert "--output" in preflight["run"]
    assert "secrets." not in str(job)
