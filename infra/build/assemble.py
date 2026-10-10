#!/usr/bin/env python3
"""Assemble a service's statically reachable Python source files."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shlex
import shutil
import sys
from pathlib import Path, PurePosixPath
from typing import Any


class AssemblyError(Exception):
    """A source graph or path configuration error suitable for a CLI user."""


EXCLUDED_PARTS = {
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".tox", ".venv", "venv", "tests", "test", "fixtures", "fixture",
    "dist", "build", "cache", ".cache",
}

_REQUIREMENT_INCLUDE = re.compile(
    r"^(?P<prefix>\s*)(?P<option>-r|--requirement|-c|--constraint)"
    r"(?P<space>\s+)(?P<target>[^\s#]+)(?P<suffix>.*)$"
)
_LOCAL_PATH = re.compile(r"^(?:[./\\]|[A-Za-z]:[\\/]|[A-Za-z0-9_.-]+[\\/])")
_LOCAL_ARCHIVE = re.compile(r"(?i)(?:\.whl|\.egg|\.zip|\.tar(?:\.(?:gz|bz2|xz|zst))?|\.tgz|\.gz|\.bz2|\.xz)$")


def _safe_relative(value: str, *, label: str) -> PurePosixPath:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or not path.parts or any(p in ("", ".", "..") for p in path.parts):
        raise AssemblyError(f"{label} must be a contained relative path: {value!r}")
    if re.match(r"^[A-Za-z]:", value):
        raise AssemblyError(f"{label} must be a contained relative path: {value!r}")
    return path


def _excluded(path: Path, source_root: Path) -> bool:
    try:
        parts = path.relative_to(source_root).parts
    except ValueError:
        return True
    return any(part in EXCLUDED_PARTS or part.startswith(".env") for part in parts)


def _module_for_path(path: Path, root: Path) -> str:
    relative = path.relative_to(root).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _registry_prefixes(source_root: Path) -> set[str]:
    registry_path = source_root / "infra" / "build" / "dependency-providers.json"
    if not registry_path.is_file():
        return set()
    try:
        data = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssemblyError(f"cannot read dependency provider registry: {exc}") from exc
    prefixes: set[str] = set()
    for provider in data.get("providers", []):
        for prefix in provider.get("import_prefixes", []):
            if isinstance(prefix, str):
                prefixes.add(prefix)
    for optional in data.get("optional_imports", []):
        prefix = optional.get("import_prefix")
        if isinstance(prefix, str):
            prefixes.add(prefix)
    return prefixes


class Assembler:
    def __init__(self, source_root: Path, service_name: str, config: dict[str, Any]):
        self.root = source_root.resolve()
        self.service_name = service_name
        self.config = config
        self.aliases = config.get("module_aliases", {})
        self.remaps = config.get("path_remaps", {})
        self.provider_prefixes = _registry_prefixes(self.root)
        self.sources: dict[Path, dict[str, Any]] = {}
        self.edges: list[dict[str, Any]] = []
        self.external: dict[tuple[str, str], dict[str, Any]] = {}
        self.module_cache: dict[str, Path | None] = {}
        self.local_modules: set[str] = set()
        self._index_local_modules()

    def _index_local_modules(self) -> None:
        # Build a metadata-only module index; paths in excluded trees are ignored.
        for base, dirs, files in os.walk(self.root):
            base_path = Path(base)
            dirs[:] = [d for d in dirs if d not in EXCLUDED_PARTS and not d.startswith(".env")]
            for filename in files:
                if filename.endswith(".py") and not filename.startswith(".env"):
                    path = base_path / filename
                    if not _excluded(path, self.root):
                        self.local_modules.add(_module_for_path(path, self.root))

    def _alias(self, module: str) -> str:
        for alias in sorted(self.aliases, key=len, reverse=True):
            if module == alias or module.startswith(alias + "."):
                return self.aliases[alias] + module[len(alias):]
        return module

    def _candidate(self, module: str) -> Path | None:
        module = self._alias(module)
        if module in self.module_cache:
            return self.module_cache[module]
        if not module:
            self.module_cache[module] = None
            return None
        rel = Path(*module.split("."))
        candidates = (self.root / rel.with_suffix(".py"), self.root / rel / "__init__.py")
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
                if resolved.is_file() and resolved.is_relative_to(self.root) and not _excluded(resolved, self.root):
                    self.module_cache[module] = resolved
                    return resolved
            except (FileNotFoundError, OSError):
                pass
        self.module_cache[module] = None
        return None

    def _package_exists(self, module: str) -> bool:
        module = self._alias(module)
        directory = self.root.joinpath(*module.split("."))
        return directory.is_dir() and directory.resolve().is_relative_to(self.root) and not _excluded(directory, self.root)

    def _parent_initializers(self, module: str) -> list[Path]:
        pieces = module.split(".")
        found: list[Path] = []
        for i in range(1, len(pieces) + 1):
            parent = ".".join(pieces[:i])
            init = self._candidate(parent)
            if init is not None and init.name == "__init__.py" and init not in found:
                found.append(init)
        return found

    def _external_name(self, module: str) -> str:
        return module.split(".", 1)[0]

    def _is_stdlib(self, module: str) -> bool:
        return self._external_name(module) in sys.stdlib_module_names

    def _is_qualified_provider(self, module: str) -> bool:
        return any(
            module == prefix
            or module.startswith(prefix + ".")
            or ("." in module and prefix.startswith(module + "."))
            for prefix in self.provider_prefixes
        )

    def _is_external(self, module: str) -> bool:
        if self._is_stdlib(module) or self._is_qualified_provider(module):
            return True
        # A service manifest with requirements roots treats otherwise unknown
        # imports as third-party requirements. Synthetic/legacy manifests with
        # no declared requirements keep unresolved names actionable as errors.
        if self.config.get("requirements_roots"):
            return True
        # A missing descendant below an indexed local module is a local typo.
        return False

    def _has_local_ancestor(self, module: str) -> bool:
        pieces = module.split(".")
        return any(
            self._candidate(".".join(pieces[:i])) is not None
            or self._package_exists(".".join(pieces[:i]))
            for i in range(1, len(pieces))
        )

    def _record_external(self, module: str, source: Path, line: int, chain: list[str]) -> None:
        if not module or self._is_stdlib(module):
            return
        source_rel = source.relative_to(self.root).as_posix()
        key = (module, source_rel)
        self.external.setdefault(key, {
            "name": module,
            "source_path": source_rel,
            "line": line,
            "dependency_chain": chain + [source_rel],
        })

    def _fail(self, source: Path, line: int, missing: str, chain: list[str]) -> None:
        rel = source.relative_to(self.root).as_posix()
        chain_text = " -> ".join(chain + [rel])
        raise AssemblyError(
            f"service {self.service_name}: {rel}: line {line}: missing local import {missing!r}; "
            f"dependency chain: {chain_text}"
        )

    def _enqueue(self, path: Path, chain: list[str]) -> None:
        if path in self.sources:
            return
        if not path.is_relative_to(self.root) or _excluded(path, self.root):
            raise AssemblyError(f"service {self.service_name}: source escapes or is excluded from source root: {path}")
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError) as exc:
            raise AssemblyError(f"service {self.service_name}: cannot parse {path.relative_to(self.root)}: {exc}") from exc
        rel = path.relative_to(self.root).as_posix()
        self.sources[path] = {"tree": tree, "chain": chain, "module": _module_for_path(path, self.root)}

    def _resolve(self, source: Path, node: ast.AST, chain: list[str]) -> list[Path]:
        line = getattr(node, "lineno", 1)
        resolved: list[Path] = []
        source_module = _module_for_path(source, self.root)
        package = source_module if source.name == "__init__.py" else source_module.rpartition(".")[0]

        def add_module(
            module: str,
            missing_name: str,
            *,
            record_external: bool = True,
            allow_external: bool = False,
        ) -> Path | None:
            module = self._alias(module)
            candidate = self._candidate(module)
            if candidate is not None:
                for init in self._parent_initializers(module):
                    if init not in resolved:
                        resolved.append(init)
                if candidate not in resolved:
                    resolved.append(candidate)
                return candidate
            if self._package_exists(module):
                for init in self._parent_initializers(module):
                    if init not in resolved:
                        resolved.append(init)
                return None
            # Qualified provider declarations must be allowed through a local
            # namespace ancestor (for example a local ``google/`` tree).
            if self._is_stdlib(module) or self._is_qualified_provider(module) or allow_external:
                if record_external:
                    self._record_external(module, source, line, chain)
                return None
            if any(local.startswith(module + ".") for local in self.local_modules) or self._has_local_ancestor(module):
                self._fail(source, line, missing_name, chain)
            if self.config.get("requirements_roots"):
                if record_external:
                    self._record_external(module, source, line, chain)
                return None
            self._fail(source, line, missing_name, chain)

        if isinstance(node, ast.Import):
            for alias in node.names:
                module = self._alias(alias.name)
                add_module(module, alias.name)
                self.edges.append({"source_path": source.relative_to(self.root).as_posix(), "line": line,
                                   "import": alias.name, "resolved_module": module})
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                package_parts = package.split(".") if package else []
                keep = len(package_parts) - node.level + 1
                if keep <= 0:
                    self._fail(source, line, node.module or "." * node.level, chain)
                base = ".".join(package_parts[:keep] + (node.module.split(".") if node.module else []))
            else:
                base = node.module or ""
            base = self._alias(base)
            qualified_provider_child = any(
                alias.name != "*" and self._is_external(self._alias(f"{base}.{alias.name}" if base else alias.name))
                for alias in node.names
            )
            base_path = add_module(
                base,
                base or node.module or "<relative import>",
                record_external=False,
                allow_external=qualified_provider_child,
            ) if base else None
            base_is_package = self._package_exists(base) or (base_path is not None and base_path.name == "__init__.py")
            base_is_external = (
                base_path is None
                and not base_is_package
                and (self._is_external(base) or bool(self.config.get("requirements_roots")))
            )
            # For `from pkg import child`, Python can load a submodule. Only probe
            # children when the base is a package; other names may be attributes.
            for alias in node.names:
                if alias.name == "*":
                    if base_path is not None and base_path.name == "__init__.py":
                        exported_names = self._literal_all(base_path)
                        if exported_names is not None:
                            for exported_name in exported_names:
                                child = self._alias(f"{base}.{exported_name}")
                                child_path = self._candidate(child)
                                if child_path is not None:
                                    for init in self._parent_initializers(child):
                                        if init not in resolved:
                                            resolved.append(init)
                                    if child_path not in resolved:
                                        resolved.append(child_path)
                    elif base_path is None and base_is_external:
                        self._record_external(base, source, line, chain)
                    continue
                child = f"{base}.{alias.name}" if base else alias.name
                child = self._alias(child)
                child_path = self._candidate(child) if base_is_package else None
                if child_path is not None:
                    for init in self._parent_initializers(child):
                        if init not in resolved:
                            resolved.append(init)
                    if child_path not in resolved:
                        resolved.append(child_path)
                elif base_path is not None:
                    if not self._has_symbol(base_path, alias.name):
                        self._fail(source, line, alias.name, chain)
                elif self._is_external(child):
                    self._record_external(child, source, line, chain)
                elif base_is_package:
                    # A namespace package can have attributes only through loaded
                    # submodules; a missing name is a local unresolved import.
                    self._fail(source, line, f"{base}.{alias.name}", chain)
                elif base_is_external:
                    self._record_external(base, source, line, chain)
            self.edges.append({"source_path": source.relative_to(self.root).as_posix(), "line": line,
                               "import": f"from {'.' * node.level}{node.module or ''} import "
                                         f"{', '.join(alias.name for alias in node.names)}",
                               "resolved_module": base})
        return resolved

    @staticmethod
    def _has_symbol(path: Path, name: str) -> bool:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError):
            return False
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__getattr__":
                return True
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
                return True
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                    return True
            if isinstance(node, ast.Import):
                if any((alias.asname or alias.name.split(".")[0]) == name for alias in node.names):
                    return True
            if isinstance(node, ast.ImportFrom):
                if any((alias.asname or alias.name) == name for alias in node.names):
                    return True
        return False

    @staticmethod
    def _literal_all(path: Path) -> list[str] | None:
        """Return a literal module __all__, without evaluating source code."""
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeError, SyntaxError):
            return None
        value: ast.AST | None = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
            ):
                value = node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "__all__":
                value = node.value
        if value is None:
            return None
        try:
            exports = ast.literal_eval(value)
        except (ValueError, TypeError, SyntaxError):
            return None
        if isinstance(exports, (list, tuple)) and all(isinstance(name, str) for name in exports):
            return list(exports)
        return None

    def build(self) -> list[tuple[Path, list[str]]]:
        pending: list[tuple[Path, list[str]]] = []
        for entry in self.config.get("entry_roots", []):
            if not isinstance(entry, str):
                raise AssemblyError(f"service {self.service_name}: entry root must be a path string")
            rel = _safe_relative(entry, label="entry root")
            path = self.root.joinpath(*rel.parts).resolve()
            if not path.is_relative_to(self.root) or not path.is_file() or path.suffix != ".py" or _excluded(path, self.root):
                raise AssemblyError(f"service {self.service_name}: invalid entry root {entry!r}")
            pending.append((path, []))
        for extra in self.config.get("extra_modules", []):
            if not isinstance(extra, str):
                continue
            rel = _safe_relative(extra, label="extra module") if extra.endswith(".py") or "/" in extra or "\\" in extra else None
            path = self.root.joinpath(*rel.parts).resolve() if rel else self._candidate(extra)
            if path is None or not path.is_file() or not path.is_relative_to(self.root) or _excluded(path, self.root):
                self._fail(self.root / (self.config.get("entry_roots") or ["<entry>"])[0], 1, extra, [])
            pending.append((path, []))

        while pending:
            path, chain = pending.pop(0)
            if path in self.sources:
                continue
            self._enqueue(path, chain)
            tree = self.sources[path]["tree"]
            # ast.walk sees imports inside functions, classes, and control flow.
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for dependency in self._resolve(path, node, chain):
                        pending.append((dependency, chain + [path.relative_to(self.root).as_posix()]))
        return sorted(((path, info["chain"]) for path, info in self.sources.items()),
                      key=lambda item: item[0].relative_to(self.root).as_posix())

    def runtime_path(self, source: Path) -> str:
        source_rel = source.relative_to(self.root).as_posix()
        best_key = ""
        target = source_rel
        for key, value in self.remaps.items():
            key_path = _safe_relative(key, label="path remap source")
            key_str = key_path.as_posix()
            if source_rel == key_str or source_rel.startswith(key_str.rstrip("/") + "/"):
                if len(key_str) > len(best_key):
                    best_key = key_str
                    suffix = source_rel[len(key_str):].lstrip("/")
                    target = "/".join(part for part in (value.replace("\\", "/").strip("/"), suffix) if part)
        if target:
            target_path = _safe_relative(target, label="runtime path")
            return target_path.as_posix()
        raise AssemblyError(f"service {self.service_name}: remap maps {source_rel!r} to the output root")

    def requirements_graph(self) -> tuple[str, dict[str, str], dict[str, Path]]:
        """Resolve requirement roots and return the top-level file and staged graph."""
        staged: dict[str, str] = {}
        resources: dict[str, Path] = {}
        active: list[Path] = []
        visited: set[Path] = set()

        def checked_path(declaring: Path, value: str, label: str) -> Path:
            if not value or "\x00" in value:
                raise AssemblyError(f"service {self.service_name}: invalid {label} path {value!r}")
            raw = Path(value.replace("\\", "/"))
            normalized_value = value.replace("\\", "/")
            if raw.is_absolute() or normalized_value.startswith("/") or re.match(r"^[A-Za-z]:", value):
                raise AssemblyError(f"service {self.service_name}: unsafe {label} path {value!r}")
            candidate = declaring.parent / raw
            try:
                # Non-strict resolution still normalizes existing symlinked
                # components, while avoiding Windows access-denied failures
                # from strict resolution on readable paths.
                resolved = candidate.resolve()
                if not candidate.exists():
                    raise FileNotFoundError(value)
            except (OSError, RuntimeError) as exc:
                raise AssemblyError(
                    f"service {self.service_name}: {declaring.relative_to(self.root).as_posix()}: "
                    f"cannot resolve {label} {value!r}: {exc}"
                ) from exc
            if not resolved.is_relative_to(self.root) or _excluded(resolved, self.root):
                raise AssemblyError(
                    f"service {self.service_name}: unsafe {label} path {value!r} "
                    f"from {declaring.relative_to(self.root).as_posix()}"
                )
            return resolved

        def staged_name(path: Path) -> str:
            return ".requirements/" + path.relative_to(self.root).as_posix()

        def is_local_target(target: str, declaring: Path) -> bool:
            # URLs and VCS references belong to pip's network resolver. Keep
            # them intact even when their final component resembles an archive.
            if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", target):
                return False
            if target in (".", "..") or _LOCAL_PATH.match(target):
                return True
            if _LOCAL_ARCHIVE.search(target):
                return True
            if "/" not in target and "\\" not in target:
                try:
                    return (declaring.parent / target).exists()
                except OSError:
                    return False
            return False

        def rewrite_reference(text: str, source_file: Path, staged_file: str, token: str, start: int, end: int) -> str:
            resource = checked_path(source_file, token, "local requirement resource")
            if not resource.is_file() and not resource.is_dir():
                raise AssemblyError(
                    f"service {self.service_name}: local requirement resource {token!r} "
                    f"from {source_file.relative_to(self.root).as_posix()} is not a file or directory"
                )
            dest = staged_name(resource)
            if resource.is_dir():
                for base, dirs, files in os.walk(resource):
                    base_path = Path(base)
                    kept_dirs: list[str] = []
                    for dirname in dirs:
                        directory = base_path / dirname
                        if dirname in EXCLUDED_PARTS or dirname.startswith(".env"):
                            continue
                        if directory.is_symlink():
                            raise AssemblyError(
                                f"service {self.service_name}: unsafe symlink in local requirement resource "
                                f"{directory.relative_to(self.root).as_posix()}"
                            )
                        kept_dirs.append(dirname)
                    dirs[:] = kept_dirs
                    for filename in files:
                        candidate = base_path / filename
                        if candidate.is_symlink():
                            raise AssemblyError(
                                f"service {self.service_name}: unsafe symlink in local requirement resource "
                                f"{candidate.relative_to(self.root).as_posix()}"
                            )
                        if not _excluded(candidate, self.root):
                            resources[staged_name(candidate)] = candidate
            else:
                resources[dest] = resource
            staged_parent = Path(staged_file).parent
            relative_dest = os.path.relpath(dest, staged_parent.as_posix()).replace("\\", "/")
            return text[:start] + relative_dest + text[end:]

        def process_file(path: Path) -> None:
            if path in active:
                cycle = " -> ".join(p.relative_to(self.root).as_posix() for p in active + [path])
                raise AssemblyError(f"service {self.service_name}: requirements include cycle: {cycle}")
            if path in visited:
                return
            active.append(path)
            try:
                try:
                    contents = path.read_text(encoding="utf-8")
                except (OSError, UnicodeError) as exc:
                    raise AssemblyError(f"service {self.service_name}: cannot read requirements file {path}: {exc}") from exc
                out_lines: list[str] = []
                staged_file = staged_name(path)
                for line in contents.splitlines(keepends=True):
                    body = line.rstrip("\r\n")
                    ending = line[len(body):]
                    match = _REQUIREMENT_INCLUDE.match(body)
                    if match:
                        target = match.group("target")
                        included = checked_path(path, target, "requirements include")
                        if not included.is_file():
                            raise AssemblyError(
                                f"service {self.service_name}: requirements include {target!r} "
                                f"from {path.relative_to(self.root).as_posix()} is not a file"
                            )
                        target_staged = staged_name(included)
                        relative_target = os.path.relpath(target_staged, Path(staged_file).parent.as_posix()).replace("\\", "/")
                        rewritten = body[:match.start("target")] + relative_target + body[match.end("target"):]
                        out_lines.append(rewritten + ending)
                        process_file(included)
                        continue

                    # Local path requirements are either a path-only declaration,
                    # an editable argument, or a direct-reference URL after `@`.
                    spans: list[tuple[int, int]] = []
                    try:
                        tokens = shlex.shlex(body, posix=True, punctuation_chars="@")
                        tokens.whitespace_split = True
                        tokens.commenters = "#"
                        parsed = list(tokens)
                    except ValueError:
                        parsed = []
                    if parsed:
                        stripped = body.lstrip()
                        offset = len(body) - len(stripped)
                        first = parsed[0]
                        if first in ("-e", "--editable") and len(parsed) > 1:
                            target = parsed[1]
                            pos = body.find(target, offset + len(first))
                            if is_local_target(target, path):
                                spans.append((pos, pos + len(target)))
                        elif is_local_target(first, path):
                            pos = body.find(first, offset)
                            spans.append((pos, pos + len(first)))
                        else:
                            for i, token in enumerate(parsed[:-1]):
                                if token == "@" and is_local_target(parsed[i + 1], path):
                                    target = parsed[i + 1]
                                    pos = body.find(target, body.find("@") + 1)
                                    spans.append((pos, pos + len(target)))
                                    break
                    for start, end in reversed(spans):
                        body = rewrite_reference(body, path, staged_file, body[start:end], start, end)
                    out_lines.append(body + ending)
                staged[staged_file] = "".join(out_lines)
            finally:
                active.pop()
            visited.add(path)

        roots: list[str] = []
        for root_value in self.config.get("requirements_roots", []):
            if not isinstance(root_value, str):
                raise AssemblyError(f"service {self.service_name}: requirements root must be a path string")
            root = checked_path(self.root / "_requirements_root", root_value, "requirements root")
            if not root.is_file():
                raise AssemblyError(f"service {self.service_name}: requirements root {root_value!r} is not a file")
            roots.append(staged_name(root))
            process_file(root)

        top_lines = []
        for root in roots:
            relative = os.path.relpath(root, ".").replace("\\", "/")
            top_lines.append(f"-r {relative}\n")
        return "".join(top_lines), staged, resources


def assemble(service_name: str, output: Path, source_root: Path) -> None:
    source_root = source_root.resolve()
    config_path = source_root / "infra" / "build" / "service-manifests.json"
    try:
        manifest = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssemblyError(f"cannot read service manifest {config_path}: {exc}") from exc
    service = manifest.get("services", {}).get(service_name)
    if not isinstance(service, dict):
        raise AssemblyError(f"unknown service {service_name!r} in {config_path}")
    assembler = Assembler(source_root, service_name, service)
    sources = assembler.build()
    requirements_root, requirements_files, requirements_resources = assembler.requirements_graph()
    output = output.resolve()
    if output == source_root or output.is_relative_to(source_root):
        raise AssemblyError("output directory must be outside the source root")
    output.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, str]] = []
    for source, _chain in sources:
        source_rel = source.relative_to(source_root).as_posix()
        runtime_rel = assembler.runtime_path(source)
        destination = (output / Path(*PurePosixPath(runtime_rel).parts)).resolve()
        if not destination.is_relative_to(output):
            raise AssemblyError(f"service {service_name}: destination escapes output root: {runtime_rel}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_edges = [
            edge for edge in assembler.edges if edge["source_path"] == source_rel
        ]
        entries.append({
            "source_path": source_rel,
            "runtime_path": runtime_rel,
            "import_edges": source_edges,
        })
    (output / "requirements.txt").write_text(requirements_root, encoding="utf-8")
    for relative, source in requirements_resources.items():
        destination = (output / Path(*PurePosixPath(relative).parts)).resolve()
        if not destination.is_relative_to(output):
            raise AssemblyError(f"service {service_name}: requirements resource destination escapes output root: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for relative, contents in requirements_files.items():
        destination = (output / Path(*PurePosixPath(relative).parts)).resolve()
        if not destination.is_relative_to(output):
            raise AssemblyError(f"service {service_name}: requirements destination escapes output root: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(contents, encoding="utf-8")
    report = {
        "schema_version": int(manifest.get("schema_version", 1)),
        "service": service_name,
        "source_files": entries,
        "import_edges": assembler.edges,
        "required_external_imports": sorted(
            assembler.external.values(),
            key=lambda item: (item["name"], item["source_path"], item["line"]),
        ),
    }
    report_path = output / "assembly-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args(argv)
    try:
        assemble(args.service, args.output, args.source_root)
    except AssemblyError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
