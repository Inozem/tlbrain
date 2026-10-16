"""Synthetic end-to-end coverage for ordinary and local deployment routes.

The shell scenarios run from an isolated checkout with fake cloud and Docker
commands. The checkout contains no .env file, and the real source assembler,
offline validator, and image evidence reader/preflight are exercised by the
fake preflight boundary.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


APP_ROOT = Path(__file__).resolve().parents[2]
DEPLOY = APP_ROOT / "infra" / "deploy"
FUNCTION_ROUTES = {
    "sync_checker": ("core/deploy_sync_checker.sh", "tlbrain-sync-checker", "checker"),
    "tldv_webhook": ("connectors/tldv/deploy_webhook.sh", "tlbrain-tldv-webhook", "tldv_webhook"),
    "tldv_reconciliation": (
        "connectors/tldv/deploy_reconciliation.sh",
        "tlbrain-tldv-reconciliation",
        "tldv_reconciliation",
    ),
}
def _bash() -> str:
    if os.name == "nt" and os.environ.get("TLBRAIN_RUN_BASH_TESTS_ON_WINDOWS") != "1":
        pytest.skip("Bash route execution is unavailable on native Windows")
    bash = shutil.which("bash")
    if not bash:
        pytest.skip("bash is unavailable")
    return bash


def _copy_checkout(root: Path) -> None:
    root.mkdir(parents=True)
    for name in ("infra", "core", "services"):
        shutil.copytree(APP_ROOT / name, root / name, ignore=shutil.ignore_patterns(
            ".env", ".env.*", ".build", "__pycache__", ".pytest_cache", ".venv*",
        ))
    for name in ("requirements-common.txt",):
        shutil.copy2(APP_ROOT / name, root / name)
    assert not list(root.rglob(".env"))
    assert not list(root.rglob(".env.*"))


_GATE_HELPER = r'''from __future__ import annotations
import hashlib, json, os, subprocess, sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
build = root / "infra" / "build"
args = sys.argv[1:]
events = Path(os.environ["ROUTE_EVENT_LOG"])

def record(*parts):
    with events.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(list(map(str, parts))) + "\n")

def run(argv):
    return subprocess.run(argv, cwd=root, text=True, capture_output=True, check=False)

def assemble_and_validate(service, artifact):
    assembled = run([sys.executable, str(build / "assemble.py"), "--service", service,
                     "--source-root", str(root), "--output", str(artifact)])
    if assembled.returncode:
        raise SystemExit(assembled.returncode)
    result_file = root / ".build" / (service + "-route-validation.json")
    result_file.parent.mkdir(parents=True, exist_ok=True)
    checked = run([sys.executable, "-I", str(build / "validate.py"), "--service", service,
                   "--artifact", str(artifact), "--offline", "--result", str(result_file)])
    if checked.returncode:
        raise SystemExit(checked.returncode)
    report = json.loads((artifact / "assembly-report.json").read_text(encoding="utf-8"))
    validation = json.loads(result_file.read_text(encoding="utf-8"))
    record("validated", service, artifact, report["runtime_target"],
           validation.get("status"), validation.get("artifact_fingerprint", ""))
    return report

def make_registry(metadata, repository, version, service, target, checks):
    def compact(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    def digest(value):
        return "sha256:" + hashlib.sha256(value).hexdigest()
    config = {
        "os": "linux", "architecture": "amd64",
        "rootfs": {"type": "layers", "diff_ids": ["sha256:" + "1" * 64]},
        "config": {"Entrypoint": None, "Cmd": ["python", "-m", "uvicorn"],
                   "WorkingDir": "/app", "User": "1000", "Env": ["PORT=8080"], "Labels": {}},
    }
    evidence = {
        "schema_version": 1, "service": service, "target": target,
        "python_version": "3.12", "platform": "linux/amd64",
        "source_fingerprint": "sha256:" + "2" * 64,
        "requirements_fingerprint": "sha256:" + "3" * 64,
        "runtime_fingerprint": metadata.runtime_fingerprint(config),
        "checks": checks, "status": "passed",
        "workflow": {"repository": "synthetic/route-tests", "run_id": "1", "attempt": "1"},
    }
    config["config"]["Labels"][metadata.VALIDATION_LABEL] = json.dumps(evidence, sort_keys=True, separators=(",", ":"))
    blob = compact(config)
    blob_digest = digest(blob)
    manifest = {"schemaVersion": 2, "mediaType": "application/vnd.oci.image.manifest.v1+json",
                "config": {"mediaType": "application/vnd.oci.image.config.v1+json",
                           "digest": blob_digest, "size": len(blob)}, "layers": []}
    manifest_bytes = compact(manifest)
    manifest_digest = digest(manifest_bytes)
    index = {"schemaVersion": 2, "mediaType": "application/vnd.oci.image.index.v1+json",
             "manifests": [{"mediaType": "application/vnd.oci.image.manifest.v1+json",
                            "digest": manifest_digest, "size": len(manifest_bytes),
                            "platform": {"os": "linux", "architecture": "amd64"}}]}
    index_bytes = compact(index)
    index_digest = digest(index_bytes)
    class Registry:
        def __init__(self):
            self.tag_digest = index_digest
            self.manifests = {index_digest: index_bytes, manifest_digest: manifest_bytes}
            self.blobs = {blob_digest: blob}
        def resolve_tag(self, repo, tag):
            assert (repo, tag) == (repository, version)
            record("verified-version", repo, tag)
            return self.tag_digest
        def get_manifest(self, repo, requested):
            assert repo == repository and requested in self.manifests
            return self.manifests[requested]
        def get_blob(self, repo, requested):
            assert repo == repository and requested in self.blobs
            return self.blobs[requested]
    return Registry()

if args and args[0] == "--verify-mode":
    values = dict(zip(args[1::2], args[2::2]))
    service, repository, version, target = (values["--service"], values["--repository"],
                                            values["--version"], values["--target"])
    checks = [args[i + 1] for i, value in enumerate(args) if value == "--required-check"]
    sys.path.insert(0, str(root))
    from infra.build import image_metadata as metadata
    registry = make_registry(metadata, repository, version, service, target, checks)
    selected = metadata.read_verified_image(registry, repository, version, None, service,
                                             target, values["--platform"], "3.12", checks)
    record("verified-digest", service, repository, version, selected, target, ",".join(checks))
    if os.environ.get("ROUTE_FAIL_SERVICE") == service:
        raise SystemExit(44)
    print(selected)
    raise SystemExit(0)

if args and args[0] == "--image-mode":
    values = dict(zip(args[1::2], args[2::2]))
    image, service, platform = values["--image"], values["--service"], values["--platform"]
    artifact = root / ".build" / (service + "-local-route")
    report = assemble_and_validate(service, artifact)
    sys.path.insert(0, str(root))
    from infra.build import image_metadata as metadata
    from types import SimpleNamespace
    config = {"image_id": "sha256:" + "b" * 64, "os": "linux", "architecture": "amd64",
              "rootfs": {"type": "layers", "diff_ids": ["sha256:" + "4" * 64]},
              "config": {"Entrypoint": None, "Cmd": metadata._SERVICE_CMDS[service],
                         "WorkingDir": "/app", "User": "1000", "Env": ["PORT=8080"], "Labels": {}}}
    class Docker:
        def inspect_image(self, requested):
            assert requested in (image, config["image_id"])
            return json.loads(json.dumps(config))
        def run(self, requested, command, mounts):
            assert requested == config["image_id"]
            if "assembly-report.json" in command[-1]:
                return SimpleNamespace(returncode=0, stdout=json.dumps(report), stderr="")
            if any(str(item).endswith("validate.py") for item in command):
                result_mount = next(
                    host for host, destination, _readonly in mounts
                    if destination == "/tlbrain-validation-result"
                )
                (result_mount / "result.json").write_text(
                    json.dumps({"status": "passed", "service": service,
                                "target": report["runtime_target"]}),
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        def attach_label(self, requested, key, value):
            assert requested == image
            config["config"]["Labels"][key] = value
    result = metadata.run_image_preflight(Docker(), image, service,
        {"repository": "synthetic/route-tests", "run_id": "1", "attempt": "1"},
        artifact_path="/app", platform=platform, build_tools=build)
    record("validated-image", service, image, platform, report["runtime_target"],
           json.loads(result["config"]["Labels"][metadata.VALIDATION_LABEL])["source_fingerprint"])
    if os.environ.get("ROUTE_FAIL_SERVICE") == service:
        raise SystemExit(45)
    raise SystemExit(0)

if len(args) == 6 and args[0] == "--service":
    service = args[1]
    artifact = Path(args[5])
    if (args[0] != "--service" or args[1] != service or args[2] != "--source-root"
            or args[4] != "--output" or Path(args[3]).resolve() != root.resolve()
            or Path(args[5]).resolve() != artifact.resolve()):
        raise SystemExit("Function deploy passed unexpected preflight arguments: " + repr(args))
    report = assemble_and_validate(service, artifact)
    record("validated-function", service, artifact, report["runtime_target"])
    validation = json.loads((root / ".build" / (service + "-route-validation.json")).read_text(encoding="utf-8"))
    print("Artifact fingerprint: " + validation["artifact_fingerprint"])
    if os.environ.get("ROUTE_FAIL_SERVICE") == service:
        raise SystemExit(43)
    raise SystemExit(0)
raise SystemExit("unexpected preflight route: " + repr(args))
'''


def _install_fakes(root: Path, tmp_path: Path) -> tuple[Path, Path]:
    build = root / "infra" / "build"
    helper = build / "route_gate.py"
    helper.write_text(_GATE_HELPER, encoding="utf-8")
    preflight = build / "preflight.sh"
    preflight.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "if [[ \"${1:-}\" == '--verify-source' ]]; then\n"
        "  exec \"$ROUTE_PYTHON\" \"${BASH_SOURCE[0]%/*}/bootstrap_runner.py\" \"$@\"\n"
        "fi\n"
        "\"$ROUTE_PYTHON\" \"${BASH_SOURCE[0]%/*}/route_gate.py\" \"$@\"\n",
        encoding="utf-8",
    )
    preflight.chmod(0o755)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gcloud = bin_dir / "gcloud"
    gcloud.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "printf 'gcloud\\t%s\\n' \"$*\" >> \"$ROUTE_EVENT_LOG\"\n"
        "case \"$*\" in *'services describe'*) printf 'https://synthetic.example/service\\n' ;; "
        "*'functions describe'*) printf 'https://synthetic.example/function\\n' ;; "
        "esac\n",
        encoding="utf-8",
    )
    gcloud.chmod(0o755)
    docker = bin_dir / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "printf 'docker\\t%s\\n' \"$*\" >> \"$ROUTE_EVENT_LOG\"\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    return bin_dir, helper


def _read_route_events(log: Path) -> list[list[str]]:
    events: list[list[str]] = []
    if not log.exists():
        return events
    for line in log.read_text(encoding="utf-8").splitlines():
        if line.startswith(("gcloud\t", "docker\t")):
            kind, payload = line.split("\t", 1)
            events.append([kind, payload])
        else:
            events.append(json.loads(line))
    return events


def _bash_path(path: Path) -> str:
    value = path.resolve().as_posix()
    if os.name == "nt" and len(value) >= 3 and value[1:3] == ":/":
        return "/" + value[0].lower() + value[2:]
    return value


def _run_route(
    tmp_path: Path, command: str, *, fail_service: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], list[list[str]], Path]:
    bash = _bash()
    root = tmp_path / "synthetic checkout"
    _copy_checkout(root)
    bin_dir, _helper = _install_fakes(root, tmp_path)
    log = tmp_path / "route-events.jsonl"
    python_dir = str(Path(sys.executable).parent)
    environment = {
        **{
            name: os.environ[name]
            for name in ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT")
            if name in os.environ
        },
        "PATH": os.pathsep.join((str(bin_dir), python_dir, os.environ.get("PATH", ""))),
        "TEMP": str(tmp_path),
        "TMP": str(tmp_path),
        "ROUTE_PYTHON": sys.executable,
        "ROUTE_EVENT_LOG": str(log),
        "ROUTE_FAIL_SERVICE": fail_service or "",
        "PROJECT_ID": "synthetic-project",
        "REGION": "synthetic-region",
        "VERSION": "v-synthetic-42",
        "ROOT_FOLDER_URL": "https://synthetic.example/root",
        "GEMINI_API_KEY": "synthetic-gemini-key",
        "QDRANT_URL": "https://synthetic.example/qdrant",
        "QDRANT_API_KEY": "synthetic-qdrant-key",
        "QDRANT_COLLECTION": "synthetic-collection",
        "GOOGLE_CLIENT_ID": "synthetic-client-id",
        "GOOGLE_CLIENT_SECRET": "synthetic-client-secret",
        "GOOGLE_REFRESH_TOKEN": "synthetic-refresh-token",
        "TLDV_API_KEY": "synthetic-tldv-key",
        "ALLOWED_EMAIL": "synthetic@example.invalid",
        "SKIP_CONFIRM": "1",
        "MCP_SERVICE_NAME": "synthetic-mcp",
        "VECTOR_SYNC_SERVICE_NAME": "synthetic-vector-sync",
        "SYNC_CHECKER_NAME": "synthetic-sync-checker",
        "VECTOR_SYNC_QUEUE": "synthetic-vector-queue",
        "TLDV_IMPORT_SERVICE_NAME": "synthetic-tldv-import",
        "TLDV_IMPORT_QUEUE": "synthetic-tldv-queue",
        "TLDV_WEBHOOK_FUNCTION_NAME": "synthetic-tldv-webhook",
        "TLDV_RECONCILIATION_FUNCTION_NAME": "synthetic-tldv-reconciliation",
        "CLOUD_TASKS_MAX_CONCURRENT": "3",
        "VECTOR_SYNC_MAX_INSTANCES": "2",
        "MCP_MAX_INSTANCES": "1",
        "TLDV_IMPORT_MAX_INSTANCES": "2",
        "SYNC_CHECKER_SCHEDULE": "*/15 * * * *",
        "TLDV_RECONCILIATION_SCHEDULE": "0 3 * * *",
        "DOCKER_CONFIG": str(tmp_path / "docker-config"),
    }
    script = root / "infra" / "deploy" / command
    answer = b"y\n" if command in {"deploy.sh", "local_deploy.sh"} else b""
    completed = subprocess.run([bash, str(script)], cwd=root, env=environment, input=answer,
                               capture_output=True, check=False, timeout=180)
    result = subprocess.CompletedProcess(
        completed.args,
        completed.returncode,
        completed.stdout.decode("utf-8", errors="replace"),
        completed.stderr.decode("utf-8", errors="replace"),
    )
    return result, _read_route_events(log), root


def test_route_log_reader_normalizes_shell_and_gate_events(tmp_path: Path) -> None:
    log = tmp_path / "mixed-route-events.jsonl"
    log.write_text(
        'gcloud\tconfig set project synthetic-project\n'
        'docker\tbuild --platform linux/amd64 -t synthetic-image .\n'
        '["validated-function", "sync_checker", "/synthetic/.build/sync_checker", "checker"]\n',
        encoding="utf-8",
    )
    assert _read_route_events(log) == [
        ["gcloud", "config set project synthetic-project"],
        ["docker", "build --platform linux/amd64 -t synthetic-image ."],
        ["validated-function", "sync_checker", "/synthetic/.build/sync_checker", "checker"],
    ]


def test_route_entrypoints_keep_the_ordinary_and_local_operator_commands() -> None:
    assert (DEPLOY / "deploy.sh").is_file()
    assert (DEPLOY / "local_deploy.sh").is_file()
    assert (DEPLOY / "connectors" / "deploy_tldv.sh").is_file()
    assert (DEPLOY / "connectors" / "local_deploy_tldv.sh").is_file()
    assert all((DEPLOY / path).is_file() for path, _name, _entry in FUNCTION_ROUTES.values())
    ordinary = (DEPLOY / "deploy.sh").read_text(encoding="utf-8")
    local = (DEPLOY / "local_deploy.sh").read_text(encoding="utf-8")
    assert "bash infra/deploy/core/deploy_vector_sync.sh" in ordinary
    assert "bash infra/deploy/connectors/deploy_tldv.sh" in ordinary
    assert "bash infra/deploy/core/deploy_sync_checker.sh" in ordinary
    assert "bash infra/deploy/core/deploy_mcp.sh" in ordinary
    assert "bash infra/deploy/core/local_deploy_vector_sync.sh" in local
    assert "bash infra/deploy/connectors/local_deploy_tldv.sh" in local
    assert "bash infra/deploy/core/deploy_sync_checker.sh" in local
    assert "bash infra/deploy/core/local_deploy_mcp.sh" in local


@pytest.mark.parametrize(
    ("route", "local_images"),
    [("deploy.sh", False), ("local_deploy.sh", True)],
)
def test_top_level_route_hands_only_validated_sources_and_images_forward(
    tmp_path: Path, route: str, local_images: bool,
) -> None:
    result, events, root = _run_route(tmp_path, route)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "Cancelled." not in result.stdout
    assert not list(root.rglob(".env"))

    validated_functions = [event for event in events if event[0] == "validated-function"]
    assert {(event[1], Path(event[2]).name) for event in validated_functions} == {
        (service, service) for service in FUNCTION_ROUTES
    }
    function_sends = [event for event in events if event[0] == "gcloud" and event[1].startswith("functions deploy ")]
    assert len(function_sends) == 3
    for service, function_name, entrypoint in (
        ("sync_checker", "synthetic-sync-checker", "checker"),
        ("tldv_webhook", "synthetic-tldv-webhook", "tldv_webhook"),
        ("tldv_reconciliation", "synthetic-tldv-reconciliation", "tldv_reconciliation"),
    ):
        sent = next(event[1] for event in function_sends if event[1].startswith(f"functions deploy {function_name} "))
        artifact_path = root / ".build" / service
        artifact_shell = _bash_path(artifact_path)
        assert f"--source {artifact_shell}" in sent
        assert f"--entry-point {entrypoint}" in sent
        assert "--runtime python312" in sent
        assert "--gen2" in sent and "--trigger-http" in sent and "--allow-unauthenticated" in sent
        assert "--region synthetic-region" in sent
        checked = next(event for event in validated_functions if event[1] == service)
        assert Path(checked[2]).resolve() == artifact_path.resolve()
        checked_index = events.index(checked)
        sent_index = next(index for index, event in enumerate(events)
                          if event[0] == "gcloud" and event[1].startswith(f"functions deploy {function_name} "))
        assert checked_index < sent_index

    docker_events = [event for event in events if event[0] == "docker"]
    if local_images:
        assert len([event for event in docker_events if event[1].startswith("build ")]) == 3
        assert len([event for event in docker_events if event[1].startswith("push ")]) == 3
        gates = [event for event in events if event[0] == "validated-image"]
        assert {event[1] for event in gates} == {"mcp", "vector_sync", "tldv_import"}
        expected_local_images = {
            "mcp": ("gcr.io/synthetic-project/synthetic-mcp", "services.mcp.app.main:app", "Dockerfile.mcp"),
            "vector_sync": ("gcr.io/synthetic-project/synthetic-vector-sync", "services.vector_sync.app.main:app", "Dockerfile.vector_sync"),
            "tldv_import": ("gcr.io/synthetic-project/synthetic-tldv-import", "app.main:app", "Dockerfile.tldv_import"),
        }
        for event in gates:
            service, image, platform, target = event[1:5]
            expected_image, expected_target, dockerfile = expected_local_images[service]
            assert (image, platform, target) == (expected_image, "linux/amd64", expected_target)
            assert any(docker[1] == f"push {image}" for docker in docker_events)
            assert any(cloud[0] == "gcloud" and "run deploy " in cloud[1] and f"--image {image}" in cloud[1]
                       for cloud in events)
            built = next(docker[1] for docker in docker_events if docker[1].startswith("build ") and f"-t {image} " in docker[1])
            dockerfile_path = _bash_path(root / "infra" / "docker" / dockerfile)
            assert f"build --platform linux/amd64 -f {dockerfile_path} -t {image} {_bash_path(root)}" == built
            gate_index = events.index(event)
            push_index = events.index(["docker", f"push {image}"])
            deploy_index = next(index for index, cloud in enumerate(events)
                                if cloud[0] == "gcloud" and cloud[1].startswith("run deploy ")
                                and f"--image {image}" in cloud[1])
            assert gate_index < push_index < deploy_index
    else:
        assert docker_events == []
        verified = [event for event in events if event[0] == "verified-digest"]
        assert {event[1] for event in verified} == {"mcp", "vector_sync", "tldv_import"}
        assert all(event[3] == "v-synthetic-42" and event[4].startswith("sha256:") for event in verified)
        expected_registry_routes = {
            "mcp": ("inozem/tlbrain-mcp", "services.mcp.app.main:app", "compile,imports,startup,cache"),
            "vector_sync": ("inozem/tlbrain-vector-sync", "services.vector_sync.app.main:app", "compile,imports,startup,cache"),
            "tldv_import": ("inozem/tlbrain-tldv-import", "app.main:app", "compile,imports,startup"),
        }
        for event in verified:
            service, repository, version, digest, target, checks = event[1:]
            assert (repository, target, checks) == expected_registry_routes[service]
            deploy_name = {
                "mcp": "synthetic-mcp", "vector_sync": "synthetic-vector-sync",
                "tldv_import": "synthetic-tldv-import",
            }[event[1]]
            sent = next(cloud[1] for cloud in events if cloud[0] == "gcloud" and cloud[1].startswith(f"run deploy {deploy_name} "))
            assert f"--image {repository}@{digest}" in sent
            assert f"{repository}:{version}" not in sent
            gate_index = events.index(event)
            deploy_index = next(index for index, cloud in enumerate(events)
                                if cloud[0] == "gcloud" and cloud[1].startswith(f"run deploy {deploy_name} "))
            assert gate_index < deploy_index

    cloud_text = [event[1] for event in events if event[0] == "gcloud"]
    vector_deploy = next(i for i, command in enumerate(cloud_text)
                         if command.startswith("run deploy synthetic-vector-sync "))
    import_deploy = next(i for i, command in enumerate(cloud_text)
                         if command.startswith("run deploy synthetic-tldv-import "))
    checker_deploy = next(i for i, command in enumerate(cloud_text)
                          if command.startswith("functions deploy synthetic-sync-checker "))
    mcp_deploy = next(i for i, command in enumerate(cloud_text)
                      if command.startswith("run deploy synthetic-mcp "))
    assert vector_deploy < import_deploy
    assert checker_deploy < mcp_deploy


@pytest.mark.parametrize(
    ("route", "service", "expected_boundary"),
    [
        ("deploy.sh", "vector_sync", "verified-digest"),
        ("local_deploy.sh", "vector_sync", "validated-image"),
        ("deploy.sh", "tldv_webhook", "validated-function"),
    ],
)
def test_failed_gate_stops_before_its_cloud_or_registry_send(
    tmp_path: Path, route: str, service: str, expected_boundary: str,
) -> None:
    result, events, _root = _run_route(tmp_path, route, fail_service=service)
    assert result.returncode != 0
    assert "Cancelled." not in result.stdout
    assert any(event[0] == expected_boundary and event[1] == service for event in events)
    if expected_boundary == "validated-image":
        assert any(event[0] == "docker" and event[1].startswith("build ") for event in events)
        assert not any(event[0] == "docker" and event[1].startswith("push ") for event in events)
        assert not any(event[0] == "gcloud" and event[1].startswith("run deploy ") for event in events)
    elif expected_boundary == "verified-digest":
        assert not any(event[0] == "gcloud" and event[1].startswith("run deploy ") for event in events)
    else:
        assert not any(event[0] == "gcloud" and event[1].startswith("functions deploy ") for event in events)


@pytest.mark.parametrize("script", ["deploy.sh", "local_deploy.sh"])
def test_all_function_routes_are_assembled_before_their_upload(script: str) -> None:
    root = APP_ROOT / "infra" / "deploy"
    top = (root / script).read_text(encoding="utf-8")
    assert "setup_tokens.sh" in top
    for service, (relative, _function_name, entrypoint) in FUNCTION_ROUTES.items():
        source = (root / relative).read_text(encoding="utf-8")
        assert f'--service {service}' in source
        assert f'ARTIFACT="${{REPO_ROOT}}/.build/{service}"' in source
        assert f'"${{REPO_ROOT}}/.build/{service}"' in source
        assert f"--entry-point {entrypoint}" in source
        assert source.index("preflight.sh") < source.index("gcloud functions deploy")
