"""Offline validation for one assembled service artifact.

This file is copied beside ``smoke_support.py`` and is intended to run under
the runtime-only venv prepared by the caller.  It never consults the checkout.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
import sysconfig
from pathlib import Path
from typing import Any


SUPPORTED_SERVICES = {
    "mcp", "vector_sync", "tldv_import", "sync_checker", "tldv_webhook",
    "tldv_reconciliation",
}


class ValidationImportError(ImportError):
    """An actionable import failure constructed without source contents."""


def _fingerprint(artifact: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        (p for p in artifact.rglob("*") if p.is_file()),
        key=lambda p: p.relative_to(artifact).as_posix(),
    )
    for path in files:
        relative = path.relative_to(artifact).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        if path.name.lower().startswith(".env"):
            # Invalid artifacts still get a stable useful fingerprint without
            # ever reading environment-file contents.
            digest.update(b"forbidden-environment-file-content-omitted")
        else:
            digest.update(hashlib.sha256(path.read_bytes()).digest())
        digest.update(b"\n")
    return digest.hexdigest()


def _has_forbidden_environment_file(artifact: Path) -> bool:
    return any(
        path.is_file() and path.name.lower().startswith(".env")
        for path in artifact.rglob("*")
    )


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _interpreter_paths() -> list[str]:
    roots: set[Path] = set()
    for key in ("stdlib", "platstdlib", "purelib", "platlib"):
        value = sysconfig.get_path(key)
        if value:
            roots.add(Path(value).resolve())
    for value in (sys.prefix, sys.base_prefix):
        if value:
            roots.add(Path(value).resolve())
    paths: list[str] = []
    for entry in sys.path:
        if not entry:
            continue
        try:
            candidate = Path(entry).resolve()
        except (OSError, RuntimeError):
            continue
        if any(_inside(candidate, root) for root in roots):
            value = str(candidate)
            if value not in paths:
                paths.append(value)
    # Some venv layouts do not initially expose their purelib in sys.path.
    for key in ("purelib", "platlib"):
        value = sysconfig.get_path(key)
        if value and Path(value).is_dir() and value not in paths:
            paths.append(value)
    return paths


def _module_for_target(target: str, service: str, report: dict[str, Any]) -> tuple[str, str, str]:
    runtime = report.get("runtime")
    if isinstance(runtime, dict):
        kind = runtime.get("kind")
        target = runtime.get("target") or target
    else:
        kind = {"mcp": "asgi", "vector_sync": "asgi", "tldv_import": "asgi"}.get(service, "function")
    if not isinstance(target, str) or not target:
        raise ValueError("assembly report has no runtime target")
    if ":" in target:
        module, symbol = target.split(":", 1)
    elif kind == "function":
        module, symbol = "main", target
    else:
        raise ValueError("ASGI runtime target must use module:symbol form")
    if not module or not symbol or kind not in {"asgi", "function"}:
        raise ValueError("assembly report has an unsupported runtime target")
    return module, symbol, target


def _resolve_external_reference(qualified: str) -> None:
    """Import a qualified module or resolve its tail as exported symbols."""
    parts = qualified.split(".")
    for index in range(len(parts), 0, -1):
        module_name = ".".join(parts[:index])
        try:
            imported = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            missing = exc.name or ""
            if missing == module_name or module_name.startswith(missing + "."):
                continue
            raise ValidationImportError(
                f"required external import {qualified} has missing dependency {missing}"
            ) from None
        except ImportError as exc:
            raise ValidationImportError(f"required external import {qualified} failed: {exc.name or module_name}") from None

        current: Any = imported
        for name in parts[index:]:
            if hasattr(current, name):
                current = getattr(current, name)
                continue
            parent_name = getattr(current, "__name__", module_name)
            child_name = f"{parent_name}.{name}"
            try:
                current = importlib.import_module(child_name)
            except ModuleNotFoundError as exc:
                missing = exc.name or ""
                if missing == child_name or child_name.startswith(missing + "."):
                    raise ValidationImportError(f"required external import {qualified} is missing") from None
                raise ValidationImportError(
                    f"required external import {qualified} has missing dependency {missing}"
                ) from None
        return
    raise ValidationImportError(f"required external import {qualified} is missing")


def _validate_imports(artifact: Path, report: dict[str, Any], checks: list[dict[str, str]]) -> None:
    local_modules: set[str] = set()
    local_paths: dict[str, Path] = {}
    source_files = report.get("source_files", [])
    if not isinstance(source_files, list):
        raise ValueError("assembly report source_files must be a list")
    for record in source_files:
        if not isinstance(record, dict) or not isinstance(record.get("runtime_path"), str):
            raise ValueError("assembly report contains an invalid source_files entry")
        relative = record["runtime_path"]
        path = (artifact / relative).resolve()
        if not _inside(path, artifact) or not path.is_file():
            raise ValueError(f"assembly report references an invalid source file: {relative}")
        if path.suffix != ".py":
            continue
        module = relative.removesuffix(".py").replace("/", ".")
        if module.endswith(".__init__"):
            module = module[:-len(".__init__")]
        local_modules.add(module)
        local_paths[module] = path
        segments = module.split(".")
        for index in range(1, len(segments)):
            local_modules.add(".".join(segments[:index]))

    local_definitions: dict[str, set[str]] = {}
    for module, path in local_paths.items():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
        except (OSError, SyntaxError):
            continue
        definitions: set[str] = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                definitions.add(node.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                definitions.update(target.id for target in targets if isinstance(target, ast.Name))
            elif isinstance(node, ast.Import):
                definitions.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                definitions.update(alias.asname or alias.name for alias in node.names)
        local_definitions[module] = definitions

    for record in source_files:
        runtime_path = record["runtime_path"]
        if not runtime_path.endswith(".py"):
            continue
        path = (artifact / runtime_path).resolve()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=runtime_path)
        package = runtime_path.replace("\\", "/").removesuffix(".py").replace("/", ".")
        if package.endswith(".__init__"):
            package = package[: -len(".__init__")]
        elif "." in package:
            package = package.rpartition(".")[0]
        else:
            package = ""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                # External imports are checked only when the assembly report
                # marks them required. The smoke import tests imports reached
                # by application startup.
                continue
            elif isinstance(node, ast.ImportFrom):
                try:
                    module = importlib.util.resolve_name("." * node.level + (node.module or ""), package) if node.level else (node.module or "")
                except ImportError:
                    raise ValidationImportError(f"{runtime_path}:{node.lineno}: invalid relative import")
                if module:
                    if module in local_modules:
                        for alias in node.names:
                            if alias.name == "*":
                                continue
                            child_module = f"{module}.{alias.name}"
                            if child_module in local_modules:
                                continue
                            if alias.name not in local_definitions.get(module, set()):
                                raise ValidationImportError(f"missing required name {module}.{alias.name}")
    # The assembly report is authoritative for qualified external references.
    # Resolve them before the application import, without importing local app
    # modules from the source AST.
    required = report.get("required_external_imports", [])
    if isinstance(required, list):
        for item in required:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                _resolve_external_reference(item["name"])
    checks.append({"name": "source_imports", "status": "passed"})


def _compile_sources(artifact: Path, checks: list[dict[str, str]]) -> None:
    compiled = 0
    for source in sorted(artifact.rglob("*.py")):
        relative = source.relative_to(artifact).as_posix()
        compile(source.read_bytes(), relative, "exec", dont_inherit=True)
        compiled += 1
    if compiled == 0:
        raise ValueError("assembly report contains no Python source files")
    checks.append({"name": "compile_sources", "status": "passed", "count": str(compiled)})


def _load_smoke_helper() -> Any:
    helper_path = Path(__file__).resolve().with_name("smoke_support.py")
    spec = importlib.util.spec_from_file_location("_tlbrain_smoke_support", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load sibling smoke_support.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        version = distribution.version
        if name and version:
            versions[name] = version
    return dict(sorted(versions.items(), key=lambda pair: pair[0].lower()))


def _result(args: argparse.Namespace, status: str, fingerprint: str, target: str | None, checks: list[dict[str, str]], error: str | None = None, fingerprint_state: str = "complete") -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": 1,
        "service": args.service,
        "artifact_fingerprint": fingerprint,
        "artifact_fingerprint_state": fingerprint_state,
        "target": target,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "installed_versions": _versions(),
        "checks": checks,
        "status": status,
    }
    if error:
        value["error"] = error
    return value


def _write_result(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", required=True, choices=sorted(SUPPORTED_SERVICES))
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--offline", action="store_true", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args(argv)
    artifact = Path(args.artifact)
    result_path = Path(args.result)
    checks: list[dict[str, str]] = []
    fingerprint = ""
    target: str | None = None
    try:
        if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 12):
            raise RuntimeError("validator requires the caller-provided CPython 3.12 runtime")
        if not artifact.is_absolute() or not result_path.is_absolute():
            raise ValueError("artifact and result paths must be absolute")
        artifact = artifact.resolve()
        result_path = result_path.resolve()
        if not artifact.is_dir() or _inside(result_path, artifact):
            raise ValueError("artifact must be a directory and result must be outside it")
        report_path = artifact / "assembly-report.json"
        if not report_path.is_file():
            raise FileNotFoundError("artifact assembly-report.json is missing")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(report, dict) or report.get("service") != args.service:
            raise ValueError("assembly report service does not match requested service")
        _module_name, _symbol_name, target = _module_for_target(report.get("runtime_target"), args.service, report)
        fingerprint = _fingerprint(artifact)
        checks.append({"name": "python_runtime", "status": "passed"})
        checks.append({"name": "artifact_fingerprint_before", "status": "passed", "fingerprint": fingerprint})
        if _has_forbidden_environment_file(artifact):
            raise ValueError("artifact contains a forbidden environment file")
        _compile_sources(artifact, checks)
        # The artifact is the only application root. -I has already disabled
        # user site and ignored PYTHONPATH; replace sys.path to remove cwd and
        # the validator's own directory as additional application sources.
        sys.dont_write_bytecode = True
        sys.path[:] = [str(artifact), *_interpreter_paths()]
        os.environ.pop("PYTHONPATH", None)
        os.environ.pop("PYTHONUSERBASE", None)
        _validate_imports(artifact, report, checks)
        _load_smoke_helper().run_smoke(args.service, str(artifact))
        checks.append({"name": "runtime_target_and_smoke", "status": "passed", "target": target})
        checks.append({"name": "offline_smoke", "status": "passed"})
        after = _fingerprint(artifact)
        if after != fingerprint:
            raise RuntimeError("artifact changed during validation")
        checks.append({"name": "artifact_immutability", "status": "passed", "fingerprint": after})
        _write_result(result_path, _result(args, "passed", after, target, checks))
        return 0
    except BaseException as exc:
        if isinstance(exc, ValidationImportError):
            message = f"ImportError: {exc}"
        elif isinstance(exc, ImportError):
            message = f"ImportError: missing required module {exc.name or 'unknown'}"
        else:
            message = f"{type(exc).__name__}: validation failed"
        after = ""
        if artifact.is_dir():
            try:
                after = _fingerprint(artifact)
            except OSError:
                pass
            except ValueError:
                # The inventory contains a forbidden environment file. Its
                # contents are intentionally never read for a failure report.
                pass
        if fingerprint:
            checks.append({
                "name": "artifact_immutability",
                "status": "passed" if after == fingerprint else "failed",
                "fingerprint_before": fingerprint,
                "fingerprint_after": after,
            })
            if after and after != fingerprint:
                message = "RuntimeError: artifact changed during validation"
        elif artifact.is_dir():
            checks.append({
                "name": "artifact_immutability",
                "status": "passed" if after else "unavailable",
                "fingerprint_after": after,
            })
        checks.append({"name": "validation", "status": "failed"})
        try:
            if result_path.is_absolute() and (not artifact.is_absolute() or not _inside(result_path, artifact)):
                state = "forbidden_environment_file_contents_omitted" if _has_forbidden_environment_file(artifact) else "complete"
                _write_result(result_path, _result(args, "failed", after or fingerprint, target, checks, message, state))
        except OSError:
            pass
        print(message, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
