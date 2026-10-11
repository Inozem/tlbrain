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
import re
import sys
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
    parser = argparse.ArgumentParser(description="Verify an OCI image's deployment evidence and print its platform digest.")
    subparsers = parser.add_subparsers(dest="command", required=True)
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
