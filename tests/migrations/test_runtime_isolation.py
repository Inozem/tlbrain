"""Static checks that current service code does not depend on old migrations.

This test reads only Python source files reachable from the service entrypoints.
It does not import project modules or inspect migration source files.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIGRATION_MODULE_PREFIX = "migrations.v1_to_v2"
ENTRYPOINTS = (
    "services/mcp/app/main.py",
    "services/vector_sync/app/main.py",
    "services/connectors/tldv/import_service/app/main.py",
    "services/sync_checker/main.py",
    "services/connectors/tldv/webhook/main.py",
    "services/connectors/tldv/reconciliation/main.py",
)

# The deployed import name is top-level, while the source lives under services.
CURRENT_MODULE_ALIASES = {
    "tldv_client": "services/connectors/tldv/tldv_client.py",
}


@dataclass(frozen=True)
class DiscoveredModule:
    name: str
    path: Path


def _module_path(
    module_name: str,
    root: Path,
    aliases: dict[str, str],
) -> Path | None:
    """Resolve a project-local module path without importing or reading it."""
    alias_path = aliases.get(module_name)
    if alias_path is not None:
        candidate = root / alias_path
        return candidate if candidate.is_file() else None

    relative = Path(*module_name.split("."))
    package_init = root / relative / "__init__.py"
    if package_init.is_file():
        return package_init

    module_file = root / relative.with_suffix(".py")
    if module_file.is_file():
        return module_file

    # Namespace packages have no __init__.py. Resolving the directory itself
    # avoids searching it; children are considered only when imports name them.
    package_dir = root / relative
    if package_dir.is_dir():
        return package_dir
    return None


def _package_name(module_name: str, path: Path, root: Path) -> str:
    """Return the package context used to resolve relative imports."""
    if path.name == "__init__.py":
        return module_name
    if path.is_dir():
        return module_name
    return module_name.rpartition(".")[0]


def _resolve_relative(node: ast.ImportFrom, package_name: str) -> str:
    if node.level == 0:
        return node.module or ""
    parts = package_name.split(".") if package_name else []
    keep = len(parts) - (node.level - 1)
    if keep < 0:
        return ""
    prefix = parts[:keep]
    if node.module:
        prefix.extend(node.module.split("."))
    return ".".join(prefix)


def _scan_local_dependencies(
    entrypoint: Path,
    root: Path,
    aliases: dict[str, str] | None = None,
) -> list[DiscoveredModule]:
    """Follow local imports using AST only; external modules are ignored."""
    aliases = aliases or {}
    root = root.resolve()
    entrypoint = entrypoint.resolve()
    start_name = ".".join(entrypoint.relative_to(root).with_suffix("").parts)

    pending: list[DiscoveredModule] = []
    discovered: dict[Path, DiscoveredModule] = {}

    def enqueue(module_name: str) -> None:
        if not module_name:
            return

        # Python loads each package parent before the requested child module.
        names = module_name.split(".")
        for end in range(1, len(names) + 1):
            parent_name = ".".join(names[:end])
            path = _module_path(parent_name, root, aliases)
            if (
                path is None
                and (
                    parent_name == MIGRATION_MODULE_PREFIX
                    or parent_name.startswith(MIGRATION_MODULE_PREFIX + ".")
                )
            ):
                # Preserve evidence of a forbidden import even when no local
                # source file exists for it.
                path = root.joinpath(*parent_name.split("."))
            if path is not None and path not in discovered:
                pending.append(DiscoveredModule(parent_name, path))

    # Resolve package initializers for the entrypoint's import path as Python
    # would, as well as the entrypoint source itself.
    enqueue(start_name)

    while pending:
        current = pending.pop()
        if current.path in discovered:
            continue
        discovered[current.path] = current
        migration_root = (root / "migrations" / "v1_to_v2").resolve()
        try:
            current.path.resolve().relative_to(migration_root)
            reached_migration = True
        except ValueError:
            reached_migration = (
                current.name == MIGRATION_MODULE_PREFIX
                or current.name.startswith(MIGRATION_MODULE_PREFIX + ".")
            )
        if reached_migration:
            # Record the boundary and stop. Historical migration source is
            # never read, parsed, imported, or executed by this test.
            continue
        if current.path.is_dir():
            # Namespace package imports are represented by their child modules;
            # no source belongs to the directory itself.
            continue

        source = current.path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(current.path))
        package = _package_name(current.name, current.path, root)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for imported in node.names:
                    enqueue(imported.name)
            elif isinstance(node, ast.ImportFrom):
                base_name = _resolve_relative(node, package)
                enqueue(base_name)

                # `from package import child` loads child when it is a local
                # submodule. Looking for a source path is static and harmless
                # when the imported name is instead an attribute.
                base_path = _module_path(base_name, root, aliases) if base_name else None
                if base_path is not None and (base_path.name == "__init__.py" or base_path.is_dir()):
                    for imported in node.names:
                        if imported.name != "*":
                            enqueue(f"{base_name}.{imported.name}")

    return sorted(discovered.values(), key=lambda module: (module.name, str(module.path)))


def _migration_dependencies(modules: list[DiscoveredModule], root: Path) -> list[DiscoveredModule]:
    migration_root = (root / "migrations" / "v1_to_v2").resolve()
    found = []
    for module in modules:
        try:
            module.path.resolve().relative_to(migration_root)
        except ValueError:
            if module.name == MIGRATION_MODULE_PREFIX or module.name.startswith(MIGRATION_MODULE_PREFIX + "."):
                found.append(module)
        else:
            found.append(module)
    return found


def test_service_runtime_dependency_graph_excludes_v1_to_v2() -> None:
    reachable: list[str] = []
    violations: list[str] = []
    alias_expectation = (PROJECT_ROOT / CURRENT_MODULE_ALIASES["tldv_client"]).resolve()
    alias_resolutions: dict[str, list[Path]] = {}

    for entrypoint_name in ENTRYPOINTS:
        entrypoint = PROJECT_ROOT / entrypoint_name
        modules = _scan_local_dependencies(entrypoint, PROJECT_ROOT, CURRENT_MODULE_ALIASES)
        reachable.extend(f"{module.name} ({module.path.relative_to(PROJECT_ROOT)})" for module in modules)
        violations.extend(
            f"{module.name} ({module.path.relative_to(PROJECT_ROOT)})"
            for module in _migration_dependencies(modules, PROJECT_ROOT)
        )
        if entrypoint_name in {
            "services/connectors/tldv/import_service/app/main.py",
            "services/connectors/tldv/reconciliation/main.py",
        }:
            alias_resolutions[entrypoint_name] = [
                module.path.resolve()
                for module in modules
                if module.name == "tldv_client"
            ]

    assert not violations, (
        "A current service runtime reaches migrations/v1_to_v2. "
        f"Migration dependencies: {sorted(set(violations))}. "
        f"Discovered runtime modules (including tldv_client alias): {sorted(set(reachable))}"
    )
    assert alias_resolutions == {
        "services/connectors/tldv/import_service/app/main.py": [alias_expectation],
        "services/connectors/tldv/reconciliation/main.py": [alias_expectation],
    }, f"The tldv_client deployment alias was not resolved as expected: {alias_resolutions}"


def test_scanner_detects_synthetic_runtime_migration_dependency_without_execution() -> None:
    with TemporaryDirectory(dir=Path(__file__).parent) as temporary_directory:
        root = Path(temporary_directory)
        runtime = root / "services" / "runtime.py"
        migration = root / "migrations" / "v1_to_v2" / "runner.py"
        runtime.parent.mkdir(parents=True)
        migration.parent.mkdir(parents=True)
        runtime.write_text(
            "import migrations.v1_to_v2.runner as old_migration\n",
            encoding="utf-8",
        )
        migration.write_text(
            "raise RuntimeError('static scanner executed migration source')\n",
            encoding="utf-8",
        )

        modules = _scan_local_dependencies(runtime, root)
        violations = _migration_dependencies(modules, root)

        assert any(module.name == "migrations.v1_to_v2.runner" for module in violations)
        assert any(module.path == migration for module in modules)

        missing_runtime = root / "services" / "missing_runtime.py"
        missing_runtime.write_text(
            "import migrations.v1_to_v2.missing_module as absent\n",
            encoding="utf-8",
        )
        missing_modules = _scan_local_dependencies(missing_runtime, root)
        missing_violations = _migration_dependencies(missing_modules, root)
        assert any(
            module.name == "migrations.v1_to_v2.missing_module"
            for module in missing_violations
        )
