"""Verify immutable OCI image metadata and deployment validation evidence.

The reader accepts any object implementing ``resolve_tag``, ``get_manifest``
and ``get_blob``.  The command line uses the standard-library HTTP registry
client below; tests can inject a byte-oriented registry without network I/O.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen


VALIDATION_LABEL = "org.tlbrain.validation"
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_EXECUTION_FIELDS = ("Entrypoint", "Cmd", "WorkingDir", "User", "Env")
_MANIFEST_ACCEPT = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _valid_digest(value: Any) -> bool:
    return isinstance(value, str) and _DIGEST_RE.fullmatch(value) is not None


def _valid_check_name(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value == value.strip()
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", value) is not None
    )


def _validate_required_checks(required_checks: Iterable[str]) -> tuple[str, ...]:
    if isinstance(required_checks, (str, bytes)):
        raise ValueError("at least one valid required check must be specified")
    try:
        checks = tuple(required_checks)
    except TypeError as exc:
        raise ValueError("required checks must be a collection of check names") from exc
    if not checks or any(not _valid_check_name(check) for check in checks):
        raise ValueError("at least one valid required check must be specified")
    return checks


def _secure_http_url(url: str, description: str) -> Any:
    parsed = urlparse(url)
    if parsed.scheme == "https" and parsed.hostname:
        return parsed
    if parsed.scheme == "http" and parsed.hostname:
        hostname = parsed.hostname.rstrip(".").lower()
        is_loopback = hostname == "localhost"
        if not is_loopback:
            try:
                is_loopback = ipaddress.ip_address(hostname).is_loopback
            except ValueError:
                is_loopback = False
        if is_loopback:
            return parsed
    raise ValueError(f"{description} must use HTTPS (HTTP is allowed only for loopback)")


def _verify_payload(payload: bytes, digest: Any, size: Any, description: str) -> None:
    if not isinstance(payload, bytes):
        raise ValueError(f"{description} response was not bytes")
    if not _valid_digest(digest):
        raise ValueError(f"{description} descriptor has an invalid sha256 digest")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ValueError(f"{description} descriptor has an invalid size")
    if len(payload) != size:
        raise ValueError(f"{description} size does not match its descriptor")
    if _sha256(payload) != digest:
        raise ValueError(f"{description} digest does not match its descriptor")


def _parse_json(payload: bytes, description: str) -> dict[str, Any]:
    try:
        parsed = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{description} is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{description} must be a JSON object")
    return parsed


def runtime_fingerprint(config: dict[str, Any]) -> str:
    """Hash the layer diff IDs and execution settings, excluding all labels."""
    if not isinstance(config, dict):
        raise ValueError("image config must be an object")
    rootfs = config.get("rootfs")
    if not isinstance(rootfs, dict) or not isinstance(rootfs.get("diff_ids"), list):
        raise ValueError("image config is missing rootfs.diff_ids")
    container = config.get("config")
    if not isinstance(container, dict):
        raise ValueError("image config is missing execution config")
    execution = {
        "rootfs_diff_ids": rootfs["diff_ids"],
        "os": config.get("os"),
        "architecture": config.get("architecture"),
        "config": {name: container.get(name) for name in _EXECUTION_FIELDS},
    }
    return _sha256(_canonical_bytes(execution))


def _validate_final_evidence(evidence: Any, config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(evidence, dict):
        raise ValueError("validation evidence must be an object")
    fields = {
        "schema_version", "service", "target", "python_version", "platform",
        "source_fingerprint", "requirements_fingerprint", "runtime_fingerprint",
        "checks", "status", "workflow",
    }
    if set(evidence) != fields:
        raise ValueError("validation evidence fields do not match schema v1")
    if evidence["schema_version"] != 1:
        raise ValueError("validation evidence schema version is unsupported")
    if evidence["service"] not in {"mcp", "vector_sync", "tldv_import"}:
        raise ValueError("validation evidence service is invalid")
    if not isinstance(evidence["target"], str) or not evidence["target"]:
        raise ValueError("validation evidence target is invalid")
    if evidence["python_version"] != "3.12":
        raise ValueError("validation evidence requires Python 3.12")
    os_name, architecture, _variant = _platform_parts(evidence["platform"])
    if os_name != "linux" or config.get("os") != "linux" or config.get("architecture") != architecture:
        raise ValueError("validation evidence requires a Linux image")
    for field in ("source_fingerprint", "requirements_fingerprint", "runtime_fingerprint"):
        if not _valid_digest(evidence[field]):
            raise ValueError(f"validation evidence {field} is invalid")
    if evidence["runtime_fingerprint"] != runtime_fingerprint(config):
        raise ValueError("validation evidence runtime fingerprint does not match image")
    checks = evidence["checks"]
    required = {"compile", "imports", "startup"}
    if evidence["service"] in {"mcp", "vector_sync"}:
        required.add("cache")
    if (not isinstance(checks, list) or any(not _valid_check_name(item) for item in checks)
            or not required.issubset(checks) or len(set(checks)) != len(checks)):
        raise ValueError("validation evidence checks are invalid or incomplete")
    if evidence["status"] != "passed":
        raise ValueError("validation evidence did not pass")
    workflow = evidence["workflow"]
    if not isinstance(workflow, dict) or set(workflow) != {"repository", "run_id", "attempt"}:
        raise ValueError("validation evidence workflow identity is invalid")
    if any(not isinstance(workflow[key], str) or not workflow[key].strip() for key in workflow):
        raise ValueError("validation evidence workflow identity is invalid")
    return evidence


def finalize_image(
    docker: Any, image: str, evidence: dict[str, Any], *, expected_image_id: str | None = None,
    artifact_path: str = "/app", build_tools: Path | None = None,
) -> dict[str, Any]:
    """Attach compact validation evidence after proving runtime identity is stable."""
    if not isinstance(image, str) or not image:
        raise ValueError("image reference is required")
    before = docker.inspect_image(image)
    if expected_image_id is not None and before.get("image_id") != expected_image_id:
        raise RuntimeError("image tag moved before validation evidence finalization")
    before_fingerprint = runtime_fingerprint(before)
    _validate_final_evidence(evidence, before)
    label = json.dumps(evidence, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if expected_image_id is not None:
        # DockerCLI creates the metadata-only replacement from this immutable
        # source even if the user-facing tag moves between inspection calls.
        docker._pinned_image_id = expected_image_id
    docker.attach_label(image, VALIDATION_LABEL, label)
    after = docker.inspect_image(image)
    after_fingerprint = runtime_fingerprint(after)
    labels = after.get("config", {}).get("Labels") if isinstance(after.get("config"), dict) else None
    if not isinstance(labels, dict) or labels.get(VALIDATION_LABEL) != label:
        raise RuntimeError("validation evidence label was not attached")
    if after_fingerprint != before_fingerprint:
        image_id = after.get("image_id")
        if not isinstance(image_id, str) or not image_id:
            raise RuntimeError("Docker did not provide an immutable image ID for revalidation")
        changed_evidence = dict(evidence, runtime_fingerprint=after_fingerprint)
        _validate_final_evidence(changed_evidence, after)
        _validate_service_runtime(after, changed_evidence["service"])
        _run_image_runtime_validation(
            docker, image_id, changed_evidence, artifact_path=artifact_path, build_tools=build_tools
        )
        current = docker.inspect_image(image)
        if current.get("image_id") != image_id or runtime_fingerprint(current) != after_fingerprint:
            raise RuntimeError("image runtime fingerprint changed during revalidation")
        # Pin the retry to the exact image inspected above, even for direct
        # callers that did not supply an earlier expected ID.
        docker._pinned_image_id = image_id
        label = json.dumps(changed_evidence, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        docker.attach_label(image, VALIDATION_LABEL, label)
        final = docker.inspect_image(image)
        if runtime_fingerprint(final) != after_fingerprint:
            raise RuntimeError("image runtime fingerprint changed while attaching revalidated evidence")
        final_labels = final.get("config", {}).get("Labels") if isinstance(final.get("config"), dict) else None
        if not isinstance(final_labels, dict) or final_labels.get(VALIDATION_LABEL) != label:
            raise RuntimeError("revalidated evidence label was not attached")
        return final
    return after


class DockerCLI:
    """Small Docker CLI boundary; kept replaceable for offline tests."""

    def _run(self, args: list[str], *, capture: bool = True) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(["docker", *args], check=False, text=True,
                                  stdout=subprocess.PIPE if capture else None,
                                  stderr=subprocess.PIPE if capture else None)
        except OSError as exc:
            raise RuntimeError("Docker image gate requires an available Docker CLI") from exc

    def inspect_image(self, image: str) -> dict[str, Any]:
        result = self._run(["image", "inspect", image])
        if result.returncode:
            raise RuntimeError("Docker could not inspect the image")
        try:
            raw = json.loads(result.stdout)[0]
            config = raw["Config"]
            rootfs = raw["RootFS"]
            return {
                "image_id": raw["Id"],
                "os": raw["Os"], "architecture": raw["Architecture"],
                "rootfs": {"diff_ids": rootfs["Layers"]},
                "config": {**config, "Labels": config.get("Labels") or {}},
            }
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Docker returned incomplete image metadata") from exc

    def attach_label(self, image: str, key: str, value: str) -> None:
        created = self._run(["create", getattr(self, "_pinned_image_id", image)])
        if created.returncode:
            raise RuntimeError("Docker could not create image metadata container")
        container = created.stdout.strip()
        temporary_tag = f"tlbrain-validation-{os.getpid()}:{hashlib.sha256(value.encode()).hexdigest()[:12]}"
        try:
            committed = self._run(["commit", "--change", f"LABEL {key}={json.dumps(value)}", container, temporary_tag])
            if committed.returncode:
                raise RuntimeError("Docker could not attach validation evidence")
            tagged = self._run(["tag", temporary_tag, image])
            if tagged.returncode:
                raise RuntimeError("Docker could not replace image with finalized metadata")
        finally:
            self._run(["rm", "--force", container])
            self._run(["image", "rm", "--force", temporary_tag])

    def run(self, image: str, command: list[str], mounts: list[tuple[Path, str, bool]]) -> subprocess.CompletedProcess[str]:
        args = ["run", "--rm", "--network", "none"]
        for host, destination, readonly in mounts:
            args.extend(["--mount", f"type=bind,src={host},dst={destination}" + (",readonly" if readonly else "")])
        args.extend(["--entrypoint", command[0], image, *command[1:]])
        return self._run(args)


def _fingerprint_report_section(value: Any) -> str:
    return _sha256(_canonical_bytes(value))


_SERVICE_CMDS = {
    "mcp": ["python", "-m", "uvicorn", "services.mcp.app.main:app", "--host", "0.0.0.0", "--port", "8080"],
    "vector_sync": ["python", "-m", "uvicorn", "services.vector_sync.app.main:app", "--host", "0.0.0.0", "--port", "8080"],
    "tldv_import": ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"],
}


def _bm25_cache_probe_code() -> str:
    """Return an offline probe that validates a usable sparse BM25 vector."""
    return (
        "from fastembed import SparseTextEmbedding; "
        "vectors=SparseTextEmbedding(model_name='Qdrant/bm25').embed(['offline cache probe']); "
        "vector=next(iter(vectors), None); "
        "assert vector is not None; "
        "assert len(vector.indices) > 0 and len(vector.indices) == len(vector.values)"
    )


def _validate_service_runtime(config: dict[str, Any], service: str) -> None:
    container = config.get("config")
    if not isinstance(container, dict) or container.get("Cmd") != _SERVICE_CMDS[service]:
        raise ValueError("image command does not match the service runtime")
    environment = container.get("Env")
    if not isinstance(environment, list) or "PORT=8080" not in environment:
        raise ValueError("image runtime must configure PORT=8080")


def _validate_report_inventory(records: Any, description: str) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        raise RuntimeError(f"image assembly report {description} inventory is invalid")
    for record in records:
        if not isinstance(record, dict):
            raise RuntimeError(f"image assembly report {description} inventory is invalid")
        if any(not isinstance(record.get(key), str) or not record[key] for key in ("runtime_path", "source_path")):
            raise RuntimeError(f"image assembly report {description} inventory is invalid")
        digest = record.get("sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise RuntimeError(f"image assembly report {description} inventory is invalid")
    return records


def _run_image_runtime_validation(
    docker: Any, image_id: str, evidence: dict[str, Any], *,
    artifact_path: str = "/app", build_tools: Path | None = None,
) -> None:
    """Recheck the complete runtime and report identity of a pinned image."""
    service = evidence["service"]
    report_text = docker.run(
        image_id,
        ["python", "-I", "-c", f"from pathlib import Path; print(Path({artifact_path!r}, 'assembly-report.json').read_text())"],
        [],
    )
    if report_text.returncode:
        raise RuntimeError("image assembly report could not be read during revalidation")
    try:
        report = json.loads(report_text.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("image assembly report is invalid during revalidation") from exc
    if not isinstance(report, dict) or report.get("service") != service:
        raise RuntimeError("image assembly report service does not match during revalidation")
    target = report.get("runtime_target")
    source_files = _validate_report_inventory(report.get("source_files"), "source_files")
    resources = _validate_report_inventory(report.get("resources"), "resources")
    requirements = report.get("requirements_graph")
    if not isinstance(target, str) or not target or not isinstance(requirements, dict):
        raise RuntimeError("image assembly report is missing source or requirements identity")
    if target != evidence["target"]:
        raise RuntimeError("image runtime target changed during revalidation")
    source_fingerprint = _fingerprint_report_section({"source_files": source_files, "resources": resources})
    if source_fingerprint != evidence["source_fingerprint"]:
        raise RuntimeError("image source or resources changed during revalidation")
    if _fingerprint_report_section(requirements) != evidence["requirements_fingerprint"]:
        raise RuntimeError("image requirements changed during revalidation")

    if build_tools is None:
        build_tools = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="tlbrain-image-validation-") as temporary:
        external = Path(temporary)
        harness = external / "harness"
        harness.mkdir()
        shutil.copy2(build_tools / "validate.py", harness / "validate.py")
        shutil.copy2(build_tools / "smoke_support.py", harness / "smoke_support.py")
        result_file = external / "result.json"
        completed = docker.run(
            image_id,
            ["python", "-I", "/tlbrain-validation/validate.py", "--service", service,
             "--artifact", artifact_path, "--offline", "--result", "/tlbrain-validation-result/result.json"],
            [(harness, "/tlbrain-validation", True), (external, "/tlbrain-validation-result", False)],
        )
        if completed.returncode:
            raise RuntimeError("image compile/import/startup revalidation failed")
        try:
            validation = json.loads(result_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("image revalidation result is missing or invalid") from exc
        if validation.get("status") != "passed" or validation.get("service") != service or validation.get("target") != target:
            raise RuntimeError("image revalidation did not pass for the reported target")
        if service in {"mcp", "vector_sync"}:
            cache = docker.run(image_id, ["python", "-I", "-c", _bm25_cache_probe_code()], [])
            if cache.returncode:
                raise RuntimeError("image BM25 cache is missing or unusable offline")


def run_image_preflight(
    docker: Any, image: str, service: str, workflow: dict[str, str], *,
    artifact_path: str = "/app", platform: str = "linux/amd64",
    build_tools: Path | None = None,
) -> dict[str, Any]:
    """Run validation in the final image, then attach evidence as the last step."""
    if service not in {"mcp", "vector_sync", "tldv_import"}:
        raise ValueError("image validation supports mcp, vector_sync, and tldv_import")
    before = docker.inspect_image(image)
    image_id = before.get("image_id")
    if not isinstance(image_id, str) or not image_id:
        raise RuntimeError("Docker did not provide an immutable image ID")
    _validate_service_runtime(before, service)
    os_name, architecture, _variant = _platform_parts(platform)
    if os_name != "linux" or before.get("os") != os_name or before.get("architecture") != architecture:
        raise ValueError("image platform does not match requested Linux target")
    # The report comes from the built image itself, so the evidence identifies
    # the exact source and dependency graph assembled into that runtime.
    report_text = docker.run(image_id, ["python", "-I", "-c", f"from pathlib import Path; print(Path({artifact_path!r}, 'assembly-report.json').read_text())"], [])
    if report_text.returncode:
        raise RuntimeError("image assembly report could not be read")
    try:
        report = json.loads(report_text.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("image assembly report is invalid") from exc
    if not isinstance(report, dict) or report.get("service") != service:
        raise RuntimeError("image assembly report service does not match")
    target = report.get("runtime_target")
    source_files = _validate_report_inventory(report.get("source_files"), "source_files")
    resources = _validate_report_inventory(report.get("resources"), "resources")
    requirements = report.get("requirements_graph")
    if not isinstance(target, str) or not target or not isinstance(requirements, dict):
        raise RuntimeError("image assembly report is missing source or requirements identity")
    if build_tools is None:
        build_tools = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="tlbrain-image-validation-") as temporary:
        external = Path(temporary)
        harness = external / "harness"
        harness.mkdir()
        shutil.copy2(build_tools / "validate.py", harness / "validate.py")
        shutil.copy2(build_tools / "smoke_support.py", harness / "smoke_support.py")
        result_file = external / "result.json"
        # Bind the harness read-only and the result file separately. Neither
        # helper nor fake SDK boundary is copied into the image.
        completed = docker.run(image_id, ["python", "-I", "/tlbrain-validation/validate.py", "--service", service,
                                       "--artifact", artifact_path, "--offline", "--result", "/tlbrain-validation-result/result.json"],
                              [(harness, "/tlbrain-validation", True), (external, "/tlbrain-validation-result", False)])
        if completed.returncode:
            raise RuntimeError("image compile/import/startup validation failed")
        try:
            validation = json.loads(result_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("image validation result is missing or invalid") from exc
        if validation.get("status") != "passed" or validation.get("service") != service or validation.get("target") != target:
            raise RuntimeError("image validation did not pass for the reported target")
        if service in {"mcp", "vector_sync"}:
            cache = docker.run(image_id, ["python", "-I", "-c", _bm25_cache_probe_code()], [])
            if cache.returncode:
                raise RuntimeError("image BM25 cache is missing or unusable offline")
    checks = ["compile", "imports", "startup"]
    if service in {"mcp", "vector_sync"}:
        checks.append("cache")
    evidence = {
        "schema_version": 1, "service": service, "target": target,
        "python_version": "3.12", "platform": platform,
        "source_fingerprint": _fingerprint_report_section({"source_files": source_files, "resources": resources}),
        "requirements_fingerprint": _fingerprint_report_section(requirements),
        "runtime_fingerprint": runtime_fingerprint(before), "checks": checks,
        "status": "passed", "workflow": workflow,
    }
    current = docker.inspect_image(image)
    if current.get("image_id") != image_id or runtime_fingerprint(current) != runtime_fingerprint(before):
        raise RuntimeError("image tag moved or runtime changed during validation")
    return finalize_image(
        docker, image, evidence, expected_image_id=image_id,
        artifact_path=artifact_path, build_tools=build_tools,
    )


def _platform_parts(platform: str) -> tuple[str, str, str | None]:
    if not isinstance(platform, str):
        raise ValueError("platform must be a string")
    parts = platform.split("/")
    if len(parts) not in (2, 3) or any(not part for part in parts):
        raise ValueError("platform must be os/architecture[/variant]")
    return parts[0], parts[1], parts[2] if len(parts) == 3 else None


def _validation_record(config: dict[str, Any]) -> dict[str, Any]:
    container = config.get("config")
    labels = container.get("Labels") if isinstance(container, dict) else None
    if not isinstance(labels, dict) or VALIDATION_LABEL not in labels:
        raise ValueError("image is missing validation evidence")
    raw = labels[VALIDATION_LABEL]
    if not isinstance(raw, str):
        raise ValueError("validation evidence label must contain JSON text")
    try:
        evidence = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("validation evidence is malformed") from exc
    if not isinstance(evidence, dict):
        raise ValueError("validation evidence must be a JSON object")
    return evidence


def _check_evidence(
    config: dict[str, Any],
    *,
    service: str,
    target: str,
    platform: str,
    python_version: str,
    required_checks: Iterable[str],
) -> None:
    required_checks = _validate_required_checks(required_checks)
    os_name, architecture, variant = _platform_parts(platform)
    if os_name != "linux" or python_version != "3.12":
        raise ValueError("deployment verification requires Linux and Python 3.12")
    if config.get("os") != os_name or config.get("architecture") != architecture:
        raise ValueError("image config platform does not match the requested platform")
    evidence = _validation_record(config)
    required = {
        "schema_version",
        "service",
        "target",
        "python_version",
        "platform",
        "source_fingerprint",
        "requirements_fingerprint",
        "runtime_fingerprint",
        "checks",
        "status",
        "workflow",
    }
    if not required.issubset(evidence):
        raise ValueError("validation evidence is missing required fields")
    if evidence.get("schema_version") != 1:
        raise ValueError("validation evidence schema version is unsupported")
    if evidence.get("service") != service or evidence.get("target") != target:
        raise ValueError("validation evidence service or target does not match")
    if evidence.get("python_version") != "3.12" or evidence.get("python_version") != python_version:
        raise ValueError("validation evidence Python version does not match")
    evidence_platform = evidence.get("platform")
    if not isinstance(evidence_platform, str):
        raise ValueError("validation evidence platform is invalid")
    evidence_parts = _platform_parts(evidence_platform)
    if evidence_parts != (os_name, architecture, variant):
        raise ValueError("validation evidence platform does not match")
    if evidence.get("status") != "passed":
        raise ValueError("validation evidence did not pass")
    checks = evidence.get("checks")
    if not isinstance(checks, list) or any(not _valid_check_name(item) for item in checks):
        raise ValueError("validation evidence checks are invalid")
    if not set(required_checks).issubset(checks):
        raise ValueError("validation evidence is missing required checks")
    if not _valid_digest(evidence.get("source_fingerprint")):
        raise ValueError("validation evidence source fingerprint is invalid")
    if not _valid_digest(evidence.get("requirements_fingerprint")):
        raise ValueError("validation evidence requirements fingerprint is invalid")
    if evidence.get("runtime_fingerprint") != runtime_fingerprint(config):
        raise ValueError("validation evidence runtime fingerprint does not match")
    workflow = evidence.get("workflow")
    if not isinstance(workflow, dict) or not all(
        isinstance(workflow.get(key), str) and workflow[key]
        for key in ("repository", "run_id", "attempt")
    ):
        raise ValueError("validation evidence workflow identity is invalid")


def read_verified_image(
    registry: Any,
    repository: str,
    version: str,
    expected_digest: str | None,
    service: str,
    target: str,
    platform: str,
    python_version: str,
    required_checks: Iterable[str],
) -> str:
    """Resolve a tag once, verify selected image bytes and evidence, return digest."""
    if not repository or not version:
        raise ValueError("repository and version are required")
    required_checks = _validate_required_checks(required_checks)
    resolved_digest = registry.resolve_tag(repository, version)
    if not _valid_digest(resolved_digest):
        raise ValueError("registry returned an invalid resolved digest")
    if expected_digest is not None:
        if not _valid_digest(expected_digest):
            raise ValueError("expected digest is invalid")
        if expected_digest != resolved_digest:
            raise ValueError("resolved digest does not match the expected digest")

    index_bytes = registry.get_manifest(repository, resolved_digest)
    if not isinstance(index_bytes, bytes) or _sha256(index_bytes) != resolved_digest:
        raise ValueError("resolved manifest digest does not match its bytes")
    top = _parse_json(index_bytes, "resolved manifest")
    if "manifests" in top:
        entries = top.get("manifests")
        if not isinstance(entries, list):
            raise ValueError("manifest index has an invalid manifests field")
        wanted_os, wanted_arch, wanted_variant = _platform_parts(platform)
        matching = []
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("manifest index contains an invalid descriptor")
            entry_platform = entry.get("platform")
            if not isinstance(entry_platform, dict):
                continue
            candidate = (
                entry_platform.get("os"),
                entry_platform.get("architecture"),
                entry_platform.get("variant"),
            )
            if candidate[:2] == (wanted_os, wanted_arch) and (
                wanted_variant is None or candidate[2] == wanted_variant
            ):
                matching.append(entry)
        if len(matching) != 1:
            raise ValueError("manifest index does not contain exactly one requested platform")
        descriptor = matching[0]
        selected_digest = descriptor.get("digest")
        if not _valid_digest(selected_digest):
            raise ValueError("platform manifest descriptor digest is invalid")
        manifest_bytes = registry.get_manifest(repository, selected_digest)
        _verify_payload(manifest_bytes, selected_digest, descriptor.get("size"), "platform manifest")
    else:
        selected_digest = resolved_digest
        manifest_bytes = index_bytes

    manifest = _parse_json(manifest_bytes, "platform manifest")
    if "manifests" in manifest:
        raise ValueError("selected platform descriptor points to another manifest index")
    config_descriptor = manifest.get("config")
    if not isinstance(config_descriptor, dict):
        raise ValueError("platform manifest is missing its config descriptor")
    config_digest = config_descriptor.get("digest")
    if not _valid_digest(config_digest):
        raise ValueError("image config descriptor digest is invalid")
    config_bytes = registry.get_blob(repository, config_digest)
    _verify_payload(config_bytes, config_digest, config_descriptor.get("size"), "image config")
    config = _parse_json(config_bytes, "image config")
    _check_evidence(
        config,
        service=service,
        target=target,
        platform=platform,
        python_version=python_version,
        required_checks=required_checks,
    )
    return selected_digest


class HTTPRegistry:
    """Small anonymous OCI Distribution client using only urllib."""

    def __init__(self, registry: str) -> None:
        parsed = _secure_http_url(
            registry if "://" in registry else "https://" + registry,
            "registry URL",
        )
        if not parsed.netloc:
            raise ValueError("registry must be an HTTP(S) registry URL")
        self.base = parsed.geturl().rstrip("/")
        self._token: str | None = None

    def _url(self, repository: str, suffix: str) -> str:
        return f"{self.base}/v2/{quote(repository, safe='/')}/{suffix}"

    def _request(self, url: str, *, accept: str | None = None) -> tuple[bytes, Any]:
        headers = {"User-Agent": "tlbrain-image-metadata/1"}
        if accept:
            headers["Accept"] = accept
        if self._token:
            headers["Authorization"] = "Bearer " + self._token
        request = Request(url, headers=headers)
        try:
            response = urlopen(request, timeout=30)
            with response:
                return response.read(), response.headers
        except HTTPError as exc:
            challenge = exc.headers.get("WWW-Authenticate", "")
            if exc.code == 401 and challenge.lower().startswith("bearer ") and not self._token:
                self._token = self._fetch_bearer_token(challenge[7:])
                return self._request(url, accept=accept)
            raise RuntimeError(f"registry request failed with HTTP {exc.code}") from None
        except (URLError, TimeoutError, OSError):
            raise RuntimeError("registry request failed") from None

    def _fetch_bearer_token(self, challenge: str) -> str:
        values = dict(re.findall(r'([A-Za-z_]+)="([^"]*)"', challenge))
        realm = values.get("realm")
        try:
            _secure_http_url(realm or "", "registry token realm")
        except ValueError:
            raise RuntimeError("registry authentication challenge is invalid") from None
        query = {key: values[key] for key in ("service", "scope") if key in values}
        separator = "&" if "?" in realm else "?"
        try:
            with urlopen(realm + (separator + urlencode(query) if query else ""), timeout=30) as response:
                token_data = json.loads(response.read().decode("utf-8"))
        except Exception:
            raise RuntimeError("registry authentication failed") from None
        token = token_data.get("token") or token_data.get("access_token") if isinstance(token_data, dict) else None
        if not isinstance(token, str) or not token:
            raise RuntimeError("registry authentication returned no token")
        return token

    def resolve_tag(self, repository: str, tag: str) -> str:
        url = self._url(repository, "manifests/" + quote(tag, safe=""))
        payload, headers = self._request(url, accept=_MANIFEST_ACCEPT)
        digest = headers.get("Docker-Content-Digest")
        if not _valid_digest(digest):
            # Some registries omit the digest header; derive the immutable
            # address from the bytes returned by this one tag resolution.
            digest = _sha256(payload)
        return digest

    def get_manifest(self, repository: str, digest: str) -> bytes:
        payload, headers = self._request(
            self._url(repository, "manifests/" + quote(digest, safe="")),
            accept=_MANIFEST_ACCEPT,
        )
        advertised = headers.get("Docker-Content-Digest")
        if advertised is not None and advertised != digest:
            raise ValueError("registry manifest digest header does not match")
        return payload

    def get_blob(self, repository: str, digest: str) -> bytes:
        payload, _ = self._request(
            self._url(repository, "blobs/" + quote(digest, safe=""))
        )
        return payload


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Validate image evidence or verify an OCI image's deployment evidence.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    image = subparsers.add_parser("image-preflight", help="run offline checks in a final Linux image and attach evidence")
    image.add_argument("--image", required=True)
    image.add_argument("--service", required=True, choices=("mcp", "vector_sync", "tldv_import"))
    image.add_argument("--platform", default="linux/amd64")
    image.add_argument("--artifact-path", default="/app")
    image.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY", "local"))
    image.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", "local"))
    image.add_argument("--attempt", default=os.environ.get("GITHUB_RUN_ATTEMPT", "1"))
    image.add_argument("--workflow-repository")
    image.add_argument("--workflow-run-id")
    image.add_argument("--workflow-attempt")
    verify = subparsers.add_parser("verify", help="verify an image tag and its platform evidence")
    verify.add_argument("--repository", required=True)
    verify.add_argument("--version", required=True)
    verify.add_argument("--service", required=True)
    verify.add_argument("--target", required=True)
    verify.add_argument("--platform", required=True)
    verify.add_argument("--registry", default="https://registry-1.docker.io")
    verify.add_argument("--expected-digest")
    verify.add_argument("--required-check", action="append", required=True)
    args = parser.parse_args()
    try:
        if args.command == "image-preflight":
            workflow = {
                "repository": args.workflow_repository or args.repository,
                "run_id": args.workflow_run_id or args.run_id,
                "attempt": args.workflow_attempt or args.attempt,
            }
            result = run_image_preflight(
                DockerCLI(), args.image, args.service, workflow,
                artifact_path=args.artifact_path, platform=args.platform,
            )
            print(json.dumps(result.get("config", {}).get("Labels", {}).get(VALIDATION_LABEL), separators=(",", ":")))
            return 0
        digest = read_verified_image(
            HTTPRegistry(args.registry),
            repository=args.repository,
            version=args.version,
            expected_digest=args.expected_digest,
            service=args.service,
            target=args.target,
            platform=args.platform,
            python_version="3.12",
            required_checks=args.required_check,
        )
    except Exception as exc:
        # Registry exceptions can contain URLs or response data; emit only a
        # short error category/message assembled by this module.
        message = str(exc)
        if "token" in message.lower() or "authorization" in message.lower():
            message = "registry authentication failed"
        print(f"image verification failed: {message}", file=sys.stderr)
        return 1
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
