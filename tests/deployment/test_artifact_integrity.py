"""Integration checks for the bytes handed to deployment gates.

All inputs are temporary synthetic checkouts. The assembler and validator are
the production implementations; no registry, cloud, or Docker service is used.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


APP_ROOT = Path(__file__).resolve().parents[2]
BUILD = APP_ROOT / "infra" / "build"
SERVICES = {
    "mcp": ("services/mcp/app/main.py", "services.mcp.app.main:app", "asgi", {}),
    "vector_sync": ("services/vector_sync/app/main.py", "services.vector_sync.app.main:app", "asgi", {}),
    "tldv_import": ("services/tldv_import/app/main.py", "app.main:app", "asgi", {"services/tldv_import": ""}),
    "sync_checker": ("services/sync_checker/main.py", "checker", "function", {"services/sync_checker": ""}),
    "tldv_webhook": ("services/tldv_webhook/main.py", "tldv_webhook", "function", {"services/tldv_webhook": ""}),
    "tldv_reconciliation": (
        "services/tldv_reconciliation/main.py", "tldv_reconciliation", "function",
        {"services/tldv_reconciliation": ""},
    ),
}


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _checkout(root: Path) -> None:
    build = root / "infra" / "build"
    build.mkdir(parents=True)
    for name in ("assemble.py", "validate.py", "smoke_support.py"):
        shutil.copy2(BUILD / name, build / name)
    services: dict[str, dict[str, object]] = {}
    for service, (entry, target, kind, remaps) in SERVICES.items():
        callable_name = target.split(":", 1)[-1] if ":" in target else target
        if kind == "asgi":
            source = (
                "async def app(scope, receive, send):\n"
                "    await receive()\n"
                "    await send({'type': 'http.response.start', 'status': 200, 'headers': []})\n"
                "    await send({'type': 'http.response.body', 'body': b'{\"status\":\"ok\"}', 'more_body': False})\n"
            )
        else:
            source = f"def {callable_name}(request):\n    return ({{}}, 200)\n"
        _write(root / entry, source)
        runtime: dict[str, str] = {"kind": kind, "target": target}
        services[service] = {
            "entry_roots": [entry], "runtime": runtime, "path_remaps": remaps,
            "module_aliases": {}, "requirements_roots": [], "extra_modules": [],
            "resources": [], "model_preparation": [],
        }
    _write(build / "service-manifests.json", json.dumps({"schema_version": 1, "services": services}))
    _write(build / "dependency-providers.json", json.dumps({"schema_version": 1, "providers": {}, "optional_imports": {}}))


def _run_assembly(root: Path, service: str, artifact: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-I", str(root / "infra/build/assemble.py"), "--service", service,
         "--source-root", str(root), "--output", str(artifact)],
        cwd=root, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _inventory(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*")) if path.is_file()
    }


@pytest.mark.parametrize("service", SERVICES)
def test_assembler_stages_only_declared_service_graph(tmp_path: Path, service: str) -> None:
    root = tmp_path / "checkout"
    _checkout(root)
    artifact = root / ".build" / service

    _run_assembly(root, service, artifact)

    report = json.loads((artifact / "assembly-report.json").read_text(encoding="utf-8"))
    assert report["service"] == service
    assert report["runtime_target"] == SERVICES[service][1]
    staged = {path.relative_to(artifact).as_posix() for path in artifact.rglob("*") if path.is_file()}
    assert "assembly-report.json" in staged
    assert "requirements.txt" in staged
    assert not any(path.startswith(("tests/", "infra/", ".build/")) for path in staged)
    assert all(entry["sha256"] for entry in report["source_files"])
    source_payloads = {entry["runtime_path"]: entry["sha256"] for entry in report["source_files"]}
    assert staged == {"assembly-report.json", "requirements.txt", *source_payloads}
    assert {
        relative: hashlib.sha256((artifact / relative).read_bytes()).hexdigest()
        for relative in source_payloads
    } == source_payloads


def test_validator_uses_artifact_as_sole_app_root_and_preserves_its_bytes(tmp_path: Path) -> None:
    root = tmp_path / "checkout"
    _checkout(root)
    source_origin = root / "services/mcp/app/main.py"
    source_origin.write_text(
        source_origin.read_text(encoding="utf-8") + "from origin import VALUE\nassert VALUE == 'artifact'\n",
        encoding="utf-8",
    )
    _write(root / "origin.py", "VALUE = 'artifact'\n")
    artifact = root / ".build" / "mcp"
    _run_assembly(root, "mcp", artifact)

    # Same-named modules outside the artifact would reveal a leaked cwd, user
    # site, or PYTHONPATH entry if the isolated validator imported them.
    poison = tmp_path / "poison"
    _write(poison / "origin.py", "VALUE = 'poison'\n")
    user_site = poison / "userbase" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    _write(user_site / "origin.py", "VALUE = 'user-site poison'\n")
    result_file = tmp_path / "external-result" / "checked.json"
    harness = tmp_path / "external-harness"
    harness.mkdir()
    for name in ("validate.py", "smoke_support.py"):
        shutil.copy2(BUILD / name, harness / name)
    before = _inventory(artifact)
    environment = {
        **{key: os.environ[key] for key in ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT") if key in os.environ},
        "PATH": str(Path(sys.executable).parent),
        "TEMP": str(tmp_path),
        "TMP": str(tmp_path),
        "PYTHONPATH": str(poison),
        "PYTHONUSERBASE": str(poison / "userbase"),
        "PYTHONNOUSERSITE": "0",
    }
    completed = subprocess.run(
        [sys.executable, "-I", str((harness / "validate.py").resolve()), "--service", "mcp",
         "--artifact", str(artifact.resolve()), "--offline", "--result", str(result_file.resolve())],
        cwd=poison, env=environment, capture_output=True, text=True, check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (harness / "validate.py").is_absolute()
    assert root.resolve() not in (harness / "validate.py").resolve().parents
    assert result_file.is_file() and not result_file.is_relative_to(artifact)
    result = json.loads(result_file.read_text(encoding="utf-8"))
    assert result["status"] == "passed"
    assert result["artifact_fingerprint"]
    assert all(path.resolve().parent == harness.resolve() for path in harness.iterdir())
    assert _inventory(artifact) == before
    assert not list(artifact.rglob("*.pyc"))
    assert not any(path.is_dir() and path.name == "__pycache__" for path in artifact.rglob("*"))


@pytest.mark.parametrize("service", ("sync_checker", "tldv_webhook", "tldv_reconciliation"))
@pytest.mark.parametrize("mutation", ("modified", "added", "removed"))
def test_changed_validated_function_artifact_is_not_uploaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, service: str, mutation: str,
) -> None:
    """Mutate only after the real assembler and validator have passed."""
    route_tests_path = Path(__file__).with_name("test_deploy_routes.py")
    spec = importlib.util.spec_from_file_location("artifact_route_test_helpers", route_tests_path)
    assert spec is not None and spec.loader is not None
    route_helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(route_helpers)
    marker = 'record("validated-function", service, artifact, report["runtime_target"])'
    mutation_code = (
        "\n    paths = sorted(artifact.rglob('*.py'))\n"
        f"    if {mutation!r} == 'modified': paths[0].write_text(paths[0].read_text() + '# changed after validation\\n')\n"
        f"    elif {mutation!r} == 'added': (artifact / 'late.py').write_text('VALUE = 2\\n')\n"
        "    else: paths[0].unlink()\n"
    )
    monkeypatch.setattr(route_helpers, "_GATE_HELPER", route_helpers._GATE_HELPER.replace(marker, marker + mutation_code))

    scripts = {
        "sync_checker": "core/deploy_sync_checker.sh",
        "tldv_webhook": "connectors/tldv/deploy_webhook.sh",
        "tldv_reconciliation": "connectors/tldv/deploy_reconciliation.sh",
    }
    completed, events, _root = route_helpers._run_route(tmp_path, scripts[service])

    assert any(event[0] == "validated-function" and event[1] == service for event in events)
    names = {
        "sync_checker": "synthetic-sync-checker",
        "tldv_webhook": "synthetic-tldv-webhook",
        "tldv_reconciliation": "synthetic-tldv-reconciliation",
    }
    assert not any(
        event[0] == "gcloud" and event[1].startswith(f"functions deploy {names[service]} ")
        for event in events
    )
    assert completed.returncode != 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("service", ("mcp", "vector_sync", "tldv_import"))
@pytest.mark.parametrize("case", ("evidence", "rootfs", "cmd", "digest", "platform"))
def test_ordinary_image_route_rejects_bad_oci_evidence_before_deploy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, service: str, case: str,
) -> None:
    """Exercise the production image reader through the ordinary deploy route."""
    route_tests_path = Path(__file__).with_name("test_deploy_routes.py")
    spec = importlib.util.spec_from_file_location("artifact_image_route_helpers", route_tests_path)
    assert spec is not None and spec.loader is not None
    route_helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(route_helpers)
    marker = "registry = make_registry(metadata, repository, version, service, target, checks)"
    corrupt = f"""
if service == {service!r}:
    def compact(value): return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    def digest(value): return 'sha256:' + hashlib.sha256(value).hexdigest()
    index = json.loads(registry.manifests[registry.tag_digest])
    descriptor = next(item for item in index['manifests'] if item['platform']['architecture'] == 'amd64')
    manifest = json.loads(registry.manifests[descriptor['digest']])
    if {case!r} == 'digest':
        registry.manifests[descriptor['digest']] += b' '
    elif {case!r} == 'platform':
        descriptor['platform']['architecture'] = 'arm64'
        index_bytes = compact(index)
        registry.tag_digest = digest(index_bytes)
        registry.manifests[registry.tag_digest] = index_bytes
    else:
        config_descriptor = manifest['config']
        config = json.loads(registry.blobs[config_descriptor['digest']])
        if {case!r} == 'evidence':
            evidence = json.loads(config['config']['Labels'][metadata.VALIDATION_LABEL])
            evidence['target'] = 'unexpected.entry:app'
            config['config']['Labels'][metadata.VALIDATION_LABEL] = json.dumps(evidence)
        elif {case!r} == 'rootfs':
            config['rootfs']['diff_ids'].append('sha256:' + 'f' * 64)
        else:
            config['config']['Cmd'] = ['python', '-c', 'bad runtime']
        config_bytes = compact(config)
        config_digest = digest(config_bytes)
        registry.blobs[config_digest] = config_bytes
        manifest['config'].update(digest=config_digest, size=len(config_bytes))
        manifest_bytes = compact(manifest)
        manifest_digest = digest(manifest_bytes)
        registry.manifests[manifest_digest] = manifest_bytes
        descriptor.update(digest=manifest_digest, size=len(manifest_bytes))
        index_bytes = compact(index)
        registry.tag_digest = digest(index_bytes)
        registry.manifests[registry.tag_digest] = index_bytes
"""
    indented_corruption = "\n".join("    " + line for line in corrupt.strip().splitlines())
    monkeypatch.setattr(
        route_helpers, "_GATE_HELPER",
        route_helpers._GATE_HELPER.replace(marker, marker + "\n" + indented_corruption),
    )

    completed, events, _root = route_helpers._run_route(tmp_path, "deploy.sh")

    assert completed.returncode != 0, completed.stdout + completed.stderr
    assert any(event[0] == "verified-version" and event[1] == "inozem/tlbrain-" + service.replace("_", "-") for event in events)
    names = {"mcp": "synthetic-mcp", "vector_sync": "synthetic-vector-sync", "tldv_import": "synthetic-tldv-import"}
    assert not any(
        event[0] == "gcloud" and event[1].startswith(f"run deploy {names[service]} ")
        for event in events
    )


@pytest.mark.parametrize("service", ("mcp", "vector_sync", "tldv_import"))
def test_ordinary_image_route_pins_resolved_digest_when_version_moves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, service: str,
) -> None:
    route_tests_path = Path(__file__).with_name("test_deploy_routes.py")
    spec = importlib.util.spec_from_file_location("artifact_moving_tag_route_helpers", route_tests_path)
    assert spec is not None and spec.loader is not None
    route_helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(route_helpers)
    original = "            return self.tag_digest\n"
    moved = (
        "            resolved = self.tag_digest\n"
        "            self.tag_digest = 'sha256:' + 'f' * 64\n"
        "            return resolved\n"
    )
    monkeypatch.setattr(route_helpers, "_GATE_HELPER", route_helpers._GATE_HELPER.replace(original, moved))

    completed, events, _root = route_helpers._run_route(tmp_path, "deploy.sh")

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert any(event[0] == "verified-version" and event[1] == {
        "mcp": "inozem/tlbrain-mcp",
        "vector_sync": "inozem/tlbrain-vector-sync",
        "tldv_import": "inozem/tlbrain-tldv-import",
    }[service] for event in events)
    selected = next(event[4] for event in events if event[0] == "verified-digest" and event[1] == service)
    repositories = {
        "mcp": "inozem/tlbrain-mcp",
        "vector_sync": "inozem/tlbrain-vector-sync",
        "tldv_import": "inozem/tlbrain-tldv-import",
    }
    names = {"mcp": "synthetic-mcp", "vector_sync": "synthetic-vector-sync", "tldv_import": "synthetic-tldv-import"}
    deploy = next(
        event[1] for event in events
        if event[0] == "gcloud" and event[1].startswith(f"run deploy {names[service]} ")
    )
    assert f"{repositories[service]}@{selected}" in deploy
    assert f"{repositories[service]}:v-synthetic-42" not in deploy
