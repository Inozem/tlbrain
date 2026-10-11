"""Offline contract tests for immutable image evidence metadata.

The production helper is introduced later.  Its deliberately small test API is
documented at the call sites below; importing it only inside test bodies keeps
this file collectable before that implementation exists.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import pytest


FIXTURE = Path(__file__).parent / "fixtures" / "registry_images.json"
VALIDATION_LABEL = "org.tlbrain.validation"
# For this synthetic MCP target, evidence covers import, startup, and cache
# checks in addition to compile; source-only smoke evidence is not sufficient.
REQUIRED_CHECKS = ["compile", "imports", "startup", "cache"]


class FakeRegistry:
    """A byte-oriented OCI registry double; it never performs network I/O."""

    def __init__(
        self,
        tag_digest: str,
        manifests: dict[str, bytes],
        blobs: dict[str, bytes],
    ) -> None:
        self.tag_digest = tag_digest
        self.manifests = manifests
        self.blobs = blobs
        self.resolutions = 0
        self.tag_queries: list[tuple[str, str]] = []
        self.manifest_reads: list[str] = []
        self.blob_reads: list[str] = []
        self.move_tag_after_first_read: str | None = None
        self.fail_reads = False

    def resolve_tag(self, repository: str, tag: str) -> str:
        self.resolutions += 1
        self.tag_queries.append((repository, tag))
        return self.tag_digest

    def get_manifest(self, repository: str, digest: str) -> bytes:
        self.manifest_reads.append(digest)
        if self.fail_reads:
            raise OSError("synthetic registry failure")
        if digest not in self.manifests:
            raise KeyError(f"unknown synthetic manifest digest: {digest}")
        body = self.manifests[digest]
        if self.move_tag_after_first_read and len(self.manifest_reads) == 1:
            self.tag_digest = self.move_tag_after_first_read
        return body

    def get_blob(self, repository: str, digest: str) -> bytes:
        self.blob_reads.append(digest)
        if self.fail_reads:
            raise OSError("synthetic registry failure")
        if digest not in self.blobs:
            raise KeyError(f"unknown synthetic blob digest: {digest}")
        return self.blobs[digest]


def _metadata() -> Any:
    # Contract assumption: infra.build.image_metadata exposes the three
    # functions exercised here: runtime_fingerprint(config),
    # read_verified_image(registry, ...), and finalize_image(docker, ...).
    return importlib.import_module("infra.build.image_metadata")


def _forbid_process_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reader contracts must stay in the in-memory registry double."""
    import subprocess

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("registry reader must not invoke a subprocess")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _source_fixture() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _registry_image(metadata: Any) -> tuple[dict[str, Any], FakeRegistry, str]:
    """Build byte-accurate OCI index, manifest, and config responses."""
    fixture = _source_fixture()
    manifests: dict[str, bytes] = {}
    blobs: dict[str, bytes] = {}
    index = dict(fixture["index"])
    index["manifests"] = []

    for architecture, image in fixture["images"].items():
        manifest = dict(image["manifest"])
        config = json.loads(json.dumps(image["config"]))
        evidence = json.loads(config["config"]["Labels"][VALIDATION_LABEL])
        evidence["runtime_fingerprint"] = metadata.runtime_fingerprint(config)
        config["config"]["Labels"][VALIDATION_LABEL] = json.dumps(
            evidence, sort_keys=True, separators=(",", ":")
        )
        config_bytes = _json_bytes(config)
        config_digest = _sha256(config_bytes)
        blobs[config_digest] = config_bytes
        manifest["config"] = {
            "mediaType": "application/vnd.oci.image.config.v1+json",
            "digest": config_digest,
            "size": len(config_bytes),
        }
        manifest_bytes = _json_bytes(manifest)
        manifest_digest = _sha256(manifest_bytes)
        manifests[manifest_digest] = manifest_bytes
        index["manifests"].append(
            {
                "mediaType": "application/vnd.oci.image.manifest.v1+json",
                "digest": manifest_digest,
                "size": len(manifest_bytes),
                "platform": {"os": "linux", "architecture": architecture},
            }
        )

    index_bytes = _json_bytes(index)
    index_digest = _sha256(index_bytes)
    manifests[index_digest] = index_bytes
    registry = FakeRegistry(index_digest, manifests, blobs)
    return fixture, registry, index_digest


def _replace_amd64_config(registry: FakeRegistry, index_digest: str, payload: bytes) -> None:
    """Replace a config and refresh its parent descriptors, preserving OCI byte hashes."""
    index = json.loads(registry.manifests[index_digest])
    descriptor = next(
        item for item in index["manifests"]
        if item["platform"]["architecture"] == "amd64"
    )
    manifest = json.loads(registry.manifests[descriptor["digest"]])
    config_digest = _sha256(payload)
    registry.blobs[config_digest] = payload
    manifest["config"].update(digest=config_digest, size=len(payload))
    manifest_bytes = _json_bytes(manifest)
    manifest_digest = _sha256(manifest_bytes)
    registry.manifests[manifest_digest] = manifest_bytes
    descriptor.update(digest=manifest_digest, size=len(manifest_bytes))
    index_bytes = _json_bytes(index)
    new_index_digest = _sha256(index_bytes)
    registry.manifests[new_index_digest] = index_bytes
    registry.tag_digest = new_index_digest


def _replace_amd64_manifest(registry: FakeRegistry, index_digest: str, payload: bytes) -> None:
    """Replace a platform manifest and refresh the index's exact-byte descriptor."""
    index = json.loads(registry.manifests[index_digest])
    descriptor = next(
        item for item in index["manifests"]
        if item["platform"]["architecture"] == "amd64"
    )
    digest = _sha256(payload)
    registry.manifests[digest] = payload
    descriptor.update(digest=digest, size=len(payload))
    index_bytes = _json_bytes(index)
    new_index_digest = _sha256(index_bytes)
    registry.manifests[new_index_digest] = index_bytes
    registry.tag_digest = new_index_digest


def _read(registry: FakeRegistry, fixture: dict[str, Any], **overrides: Any) -> str:
    """Reader contract: return the selected immutable platform-manifest digest."""
    metadata = _metadata()
    arguments = {
        "repository": fixture["repository"],
        "version": fixture["version"],
        "expected_digest": None,
        "service": fixture["service"],
        "target": fixture["target"],
        "platform": fixture["platform"],
        "python_version": fixture["python_version"],
        "required_checks": REQUIRED_CHECKS,
    }
    arguments.update(overrides)
    return metadata.read_verified_image(registry, **arguments)


def test_registry_reader_resolves_version_selects_platform_and_checks_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same checked immutable digest is returned for deployment pinning."""
    _forbid_process_boundaries(monkeypatch)
    metadata = _metadata()
    fixture, registry, index_digest = _registry_image(metadata)
    selected_digest = _read(registry, fixture)

    assert registry.tag_queries == [(fixture["repository"], fixture["version"])]
    assert selected_digest in registry.manifests
    assert selected_digest in registry.manifest_reads
    assert index_digest in registry.manifest_reads
    assert _sha256(registry.manifests[index_digest]) == index_digest
    assert _sha256(registry.manifests[selected_digest]) == selected_digest
    selected_manifest = json.loads(registry.manifests[selected_digest])
    config_digest = selected_manifest["config"]["digest"]
    assert config_digest in registry.blob_reads
    assert _sha256(registry.blobs[config_digest]) == config_digest
    selected_config = json.loads(registry.blobs[config_digest])
    assert selected_config["architecture"] == "amd64"
    assert registry.resolutions == 1


@pytest.mark.parametrize(
    "case",
    [
        "missing_manifest",
        "malformed_manifest",
        "missing_config",
        "malformed_config",
        "missing_evidence",
        "malformed_evidence",
        "wrong_service",
        "wrong_target",
        "not_passed",
        "missing_check",
        "runtime_evidence_mismatch",
        "wrong_python_version",
        "wrong_platform",
        "wrong_os",
        "registry_error",
        "absent_platform",
        "bad_manifest_bytes",
        "bad_config_bytes",
    ],
)
def test_registry_reader_rejects_invalid_or_unavailable_image(
    case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_process_boundaries(monkeypatch)
    metadata = _metadata()
    fixture, registry, index_digest = _registry_image(metadata)
    if case == "registry_error":
        registry.fail_reads = True
    elif case == "missing_manifest":
        index = json.loads(registry.manifests[index_digest])
        missing = index["manifests"][0]["digest"]
        registry.manifests.pop(missing)
    elif case == "absent_platform":
        fixture["platform"] = "linux/s390x"
    elif case in {"malformed_manifest", "missing_config", "malformed_config", "missing_evidence", "malformed_evidence", "wrong_service", "wrong_target", "not_passed", "missing_check", "runtime_evidence_mismatch", "wrong_python_version", "wrong_platform", "wrong_os", "bad_manifest_bytes", "bad_config_bytes"}:
        index = json.loads(registry.manifests[index_digest])
        child_digest = next(
            item["digest"] for item in index["manifests"]
            if item["platform"]["architecture"] == "amd64"
        )
        manifest = json.loads(registry.manifests[child_digest])
        if case == "malformed_manifest":
            _replace_amd64_manifest(registry, index_digest, b"{")
        elif case == "missing_config":
            registry.blobs.pop(manifest["config"]["digest"])
        elif case == "malformed_config":
            _replace_amd64_config(registry, index_digest, b"not-json")
        elif case in {"bad_manifest_bytes", "bad_config_bytes"}:
            if case == "bad_manifest_bytes":
                registry.manifests[child_digest] += b" "
            else:
                config_digest = manifest["config"]["digest"]
                registry.blobs[config_digest] += b" "
        else:
            config_digest = manifest["config"]["digest"]
            config = json.loads(registry.blobs[config_digest])
            if case in {"missing_evidence", "malformed_evidence"}:
                if case == "missing_evidence":
                    config["config"]["Labels"].pop(VALIDATION_LABEL)
                else:
                    config["config"]["Labels"][VALIDATION_LABEL] = "{"
            else:
                evidence = json.loads(config["config"]["Labels"][VALIDATION_LABEL])
                if case == "wrong_service":
                    evidence["service"] = "other-service"
                elif case == "wrong_target":
                    evidence["target"] = "staging"
                elif case == "not_passed":
                    evidence["status"] = "failed"
                elif case == "missing_check":
                    evidence["checks"] = ["compile", "imports", "startup"]
                elif case == "wrong_python_version":
                    evidence["python_version"] = "3.11"
                elif case == "wrong_platform":
                    evidence["platform"] = "linux/arm64"
                else:
                    evidence["runtime_fingerprint"] = "sha256:stale-runtime"
                if case == "wrong_os":
                    config["os"] = "windows"
                    evidence["runtime_fingerprint"] = metadata.runtime_fingerprint(config)
                config["config"]["Labels"][VALIDATION_LABEL] = json.dumps(evidence)
            _replace_amd64_config(registry, index_digest, _json_bytes(config))

    with pytest.raises((ValueError, KeyError, OSError, RuntimeError)):
        _read(registry, fixture)


def test_registry_reader_pins_safe_tag_move_and_rejects_stale_expected_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _forbid_process_boundaries(monkeypatch)
    metadata = _metadata()
    fixture, registry, first_digest = _registry_image(metadata)
    # A mutable VERSION tag may move after resolution. The reader must use the
    # digest it resolved for every subsequent index/manifest/config request.
    registry.move_tag_after_first_read = "sha256:" + "f" * 64
    selected = _read(registry, fixture)
    assert selected in registry.manifests
    assert registry.resolutions == 1
    assert registry.tag_queries == [(fixture["repository"], fixture["version"])]
    assert first_digest in registry.manifest_reads
    assert all(digest != registry.tag_digest for digest in registry.manifest_reads[1:])

    fixture, registry, _ = _registry_image(metadata)
    with pytest.raises((ValueError, RuntimeError)):
        _read(registry, fixture, expected_digest="sha256:" + "0" * 64)


def test_fingerprint_is_stable_and_covers_rootfs_and_execution_config() -> None:
    metadata = _metadata()
    fixture = _source_fixture()
    config = fixture["images"]["amd64"]["config"]
    baseline = metadata.runtime_fingerprint(config)
    assert baseline == metadata.runtime_fingerprint(json.loads(json.dumps(config)))
    reordered = json.loads(json.dumps(config, sort_keys=True))
    assert metadata.runtime_fingerprint(reordered) == baseline

    variants = []
    for mutate in (
        lambda value: value["rootfs"]["diff_ids"].append("sha256:another-layer"),
        lambda value: value.update(os="windows"),
        lambda value: value.update(architecture="arm64"),
        lambda value: value["config"].update(Entrypoint=["/other"]),
        lambda value: value["config"].update(Cmd=["--other"]),
        lambda value: value["config"].update(WorkingDir="/other"),
        lambda value: value["config"].update(User="10002"),
        lambda value: value["config"].update(Env=["LANG=C"]),
    ):
        changed = json.loads(json.dumps(config))
        mutate(changed)
        variants.append(metadata.runtime_fingerprint(changed))
    assert all(value != baseline for value in variants)

    label_only = json.loads(json.dumps(config))
    label_only["config"]["Labels"][VALIDATION_LABEL] = '{"status":"different"}'
    assert metadata.runtime_fingerprint(label_only) == baseline


def test_registry_reader_rejects_empty_required_checks_before_tag_resolution() -> None:
    metadata = _metadata()
    fixture, registry, _ = _registry_image(metadata)

    with pytest.raises(ValueError):
        _read(registry, fixture, required_checks=[])

    assert registry.resolutions == 0
    assert registry.tag_queries == []


def test_http_registry_rejects_insecure_remote_urls_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _metadata()
    calls: list[Any] = []

    def forbidden(*args: Any, **kwargs: Any) -> None:
        calls.append((args, kwargs))
        raise AssertionError("URL policy checks must happen before network access")

    monkeypatch.setattr(metadata, "urlopen", forbidden)
    with pytest.raises(ValueError):
        metadata.HTTPRegistry("http://registry.example.invalid")

    registry = metadata.HTTPRegistry("https://registry.example.invalid")
    with pytest.raises(RuntimeError, match="authentication challenge is invalid"):
        registry._fetch_bearer_token(
            'realm="http://auth.example.invalid/token",service="registry"'
        )
    assert calls == []


def test_http_registry_allows_loopback_http_without_network() -> None:
    metadata = _metadata()
    registry = metadata.HTTPRegistry("http://127.0.0.1:5000")
    assert registry.base == "http://127.0.0.1:5000"


def test_finalizer_writes_compact_record_without_changing_runtime_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finalizer contract: inspect, attach one label, then re-inspect and compare."""
    _forbid_process_boundaries(monkeypatch)
    metadata = _metadata()
    fixture, registry, _ = _registry_image(metadata)
    image_digest = _read(registry, fixture)
    manifest = json.loads(registry.manifests[image_digest])
    config = json.loads(registry.blobs[manifest["config"]["digest"]])
    # Finalization must attach fresh evidence; the input image has no prior
    # validation record, so returning it unchanged cannot satisfy this test.
    config["config"]["Labels"].pop(VALIDATION_LABEL)
    before_fingerprint = metadata.runtime_fingerprint(config)
    before_rootfs = json.loads(json.dumps(config["rootfs"]))
    before_execution = {
        "architecture": config["architecture"],
        "os": config["os"],
        **{
            name: config["config"].get(name)
            for name in ("Entrypoint", "Cmd", "WorkingDir", "User", "Env")
        },
    }
    evidence = {
        "schema_version": 1,
        "service": fixture["service"],
        "target": fixture["target"],
        "python_version": fixture["python_version"],
        "platform": fixture["platform"],
        "source_fingerprint": "sha256:" + "2" * 64,
        "requirements_fingerprint": "sha256:" + "3" * 64,
        "runtime_fingerprint": before_fingerprint,
        "checks": REQUIRED_CHECKS.copy(),
        "status": "passed",
        "workflow": fixture["workflow"],
    }

    class FakeDocker:
        def __init__(self, image_config: dict[str, Any], *, alter_cmd: bool = False) -> None:
            self.config = json.loads(json.dumps(image_config))
            self.alter_cmd = alter_cmd
            self.attachments: list[tuple[str, str, str]] = []

        def inspect_image(self, image: str) -> dict[str, Any]:
            return json.loads(json.dumps(self.config))

        def attach_label(self, image: str, key: str, value: str) -> dict[str, Any]:
            self.attachments.append((image, key, value))
            self.config["config"].setdefault("Labels", {})[key] = value
            if self.alter_cmd:
                self.config["config"]["Cmd"] = ["--changed"]
            return self.inspect_image(image)

    docker = FakeDocker(config)
    finalized = metadata.finalize_image(docker, image_digest, evidence)
    assert len(docker.attachments) == 1
    written_image, written_key, written_value = docker.attachments[0]
    assert written_image == image_digest
    assert written_key == VALIDATION_LABEL
    assert written_value == json.dumps(evidence, sort_keys=True, separators=(",", ":"))
    record = json.loads(written_value)
    assert finalized["config"]["Labels"][VALIDATION_LABEL] == written_value
    assert set(record) == set(evidence)
    assert "manifest_digest" not in record and "config_digest" not in record
    assert metadata.runtime_fingerprint(finalized) == before_fingerprint
    assert finalized["rootfs"] == before_rootfs
    assert {
        "architecture": finalized["architecture"],
        "os": finalized["os"],
        **{
            name: finalized["config"].get(name)
            for name in ("Entrypoint", "Cmd", "WorkingDir", "User", "Env")
        },
    } == before_execution

    with pytest.raises((ValueError, RuntimeError)):
        metadata.finalize_image(FakeDocker(config, alter_cmd=True), image_digest, evidence)
