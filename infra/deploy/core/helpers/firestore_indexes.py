#!/usr/bin/env python3
"""Apply and verify the repository's declared Firestore indexes."""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


class DeployError(RuntimeError):
    pass


COMMAND_TIMEOUT = 60.0


def _run(argv: list[str], timeout: float | None = None) -> str:
    command = list(argv)
    executable = shutil.which(command[0])
    if executable:
        command[0] = executable
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False,
                                timeout=min(COMMAND_TIMEOUT, timeout) if timeout else COMMAND_TIMEOUT)
    except subprocess.TimeoutExpired as exc:
        raise DeployError("gcloud command timed out") from exc
    if result.returncode:
        if "ALREADY_EXISTS" in (result.stderr or ""):
            raise DeployError("ALREADY_EXISTS")
        raise DeployError(f"gcloud command failed ({result.returncode}): {' '.join(argv[:4])}")
    return result.stdout


def _json(argv: list[str]) -> Any:
    output = _run(argv)
    try:
        value = json.loads(output)
    except (TypeError, json.JSONDecodeError) as exc:
        raise DeployError(f"malformed JSON from {' '.join(argv[:4])}") from exc
    if not isinstance(value, (list, dict)):
        raise DeployError("unexpected JSON response type")
    return value


def _records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list) and all(isinstance(row, dict) for row in value):
        return value
    if isinstance(value, dict):
        for key in ("indexes", "fields", "operations"):
            if key in value and isinstance(value[key], list) and all(isinstance(row, dict) for row in value[key]):
                return value[key]
    raise DeployError("malformed resource list")


def _field_sig(fields: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(fields, list):
        return ()
    result = []
    for field in fields:
        if not isinstance(field, dict) or not isinstance(field.get("fieldPath"), str):
            raise DeployError("malformed index field")
        path = field["fieldPath"]
        config = field.get("order", field.get("arrayConfig"))
        if path == "__name__":
            continue
        if config not in ("ASCENDING", "DESCENDING", "CONTAINS"):
            raise DeployError("unsupported index field configuration")
        result.append((path, config))
    return tuple(result)


def _resource_group(row: dict[str, Any]) -> str | None:
    group = row.get("collectionGroup")
    if isinstance(group, str):
        return group
    name = row.get("name")
    if isinstance(name, str):
        parts = name.split("/")
        try:
            return parts[parts.index("collectionGroups") + 1]
        except (ValueError, IndexError):
            return None
    return None


def _composite_matches(row: dict[str, Any], wanted: dict[str, Any]) -> bool:
    scope = wanted.get("queryScope", "COLLECTION")
    fields = _field_sig(wanted.get("fields", []))
    return (_resource_group(row) == wanted["collectionGroup"]
            and row.get("queryScope", "COLLECTION") == scope
            and _field_sig(row.get("fields", [])) == fields)


def _composite_ready(rows: list[dict[str, Any]], wanted: dict[str, Any]) -> bool:
    return any(_composite_matches(row, wanted) and row.get("state") == "READY" for row in rows)


def _field_config_ready(rows: list[dict[str, Any]], project: str, group: str, path: str) -> bool:
    target = f"collectionGroups/{group}/fields/{path}"
    for row in rows:
        name = row.get("name")
        if name == f"projects/{project}/databases/(default)/{target}" or (
            row.get("collectionGroup") == group and row.get("fieldPath") == path
        ):
            config = row.get("indexConfig", {})
            if not isinstance(config, dict):
                raise DeployError("malformed field indexConfig")
            if config.get("usesAncestorConfig") is True or config.get("reverting") is True:
                return False
            indexes = config.get("indexes")
            if indexes == [] and config.get("usesAncestorConfig") is not True:
                return True
            if indexes is None and isinstance(config.get("ancestorField"), str):
                return True
            if indexes is not None and not isinstance(indexes, list):
                raise DeployError("malformed field index configuration")
    return False


def _metadata_values(operation: dict[str, Any], key: str) -> list[str]:
    values: list[str] = []
    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for name, child in value.items():
                if name == key and isinstance(child, str):
                    values.append(child)
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(operation)
    return values


def _operation_relevant(operation: dict[str, Any], project: str, group: str,
                        path: str | None = None, index_name: str | None = None) -> bool:
    exact_field = f"projects/{project}/databases/(default)/collectionGroups/{group}/fields/{path}" if path else None
    if exact_field and exact_field in _metadata_values(operation, "field"):
        return True
    if index_name and index_name in _metadata_values(operation, "index"):
        return True
    return False


def _done(operation: dict[str, Any]) -> bool:
    states = [str(value).upper() for value in _metadata_values(operation, "state")]
    if any(state in {"FAILED", "ERROR", "CANCELLED", "CANCELED"} for state in states):
        raise DeployError("Firestore operation entered a failed state")
    if operation.get("done") is not True:
        return False
    if operation.get("error"):
        raise DeployError("Firestore operation completed with an error")
    return True


def _settings() -> tuple[float, float]:
    try:
        timeout = float(os.environ.get("FIRESTORE_INDEX_TIMEOUT_SECONDS", "3600"))
        interval = float(os.environ.get("FIRESTORE_INDEX_POLL_SECONDS", "10"))
        command_timeout = float(os.environ.get("FIRESTORE_INDEX_COMMAND_TIMEOUT_SECONDS", "60"))
    except ValueError as exc:
        raise DeployError("invalid Firestore index timeout setting") from exc
    if not all(math.isfinite(x) and x > 0 for x in (timeout, interval, command_timeout)):
        raise DeployError("Firestore index timeout settings must be finite and positive")
    global COMMAND_TIMEOUT
    COMMAND_TIMEOUT = min(command_timeout, timeout)
    return timeout, interval


def _load_declarations(index_file: Path) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    source = json.loads(index_file.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise DeployError("malformed canonical Firestore declaration")
    composites = source.get("indexes", [])
    overrides = source.get("fieldOverrides", [])
    if not isinstance(composites, list) or not isinstance(overrides, list):
        raise DeployError("malformed canonical Firestore declaration")
    for index in composites:
        if not isinstance(index, dict) or not isinstance(index.get("collectionGroup"), str):
            raise DeployError("malformed canonical composite declaration")
        if index.get("queryScope", "COLLECTION") not in ("COLLECTION", "COLLECTION_GROUP"):
            raise DeployError("unsupported composite query scope")
        fields = index.get("fields")
        if not isinstance(fields, list) or not fields:
            raise DeployError("malformed canonical composite fields")
        _field_sig(fields)
    exemptions: list[tuple[str, str]] = []
    for row in overrides:
        if not isinstance(row, dict) or not isinstance(row.get("collectionGroup"), str) or not isinstance(row.get("fieldPath"), str):
            raise DeployError("malformed canonical field override")
        indexes = row.get("indexes")
        if indexes == []:
            exemptions.append((row["collectionGroup"], row["fieldPath"]))
        elif not isinstance(indexes, list) or indexes:
            raise DeployError("malformed canonical field override indexes")
    return composites, exemptions


def _snapshot(project: str, database: str, fields: list[tuple[str, str]]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    composites = _records(_json(["gcloud", "firestore", "indexes", "composite", "list",
                                 "--project", project, "--database", database, "--format=json"]))
    operations = _records(_json(["gcloud", "firestore", "operations", "list",
                                 "--project", project, "--database", database,
                                 "--filter=done=false", "--limit=unlimited", "--format=json"]))
    field_rows: dict[str, list[dict[str, Any]]] = {}
    for group, _path in fields:
        if group not in field_rows:
            field_rows[group] = _records(_json(["gcloud", "firestore", "indexes", "fields", "list",
                                                "--project", project, "--database", database,
                                                f"--collection-group={group}", "--format=json"]))
    return composites, field_rows, operations


def _active(operations: list[dict[str, Any]], project: str, group: str,
            path: str | None = None, index_name: str | None = None) -> list[dict[str, Any]]:
    return [op for op in operations if not op.get("done") and _operation_relevant(op, project, group, path, index_name)]


def _plan(project: str, indexes: list[dict[str, Any]], fields: list[tuple[str, str]],
          composites: list[dict[str, Any]], field_rows: dict[str, list[dict[str, Any]]],
          operations: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[tuple[str, str]], set[str]]:
    create: list[dict[str, Any]] = []
    wait_composites: list[dict[str, Any]] = []
    wait_fields: list[tuple[str, str]] = []
    op_names: set[str] = set()
    for wanted in indexes:
        matches = [row for row in composites if _composite_matches(row, wanted)]
        if any(row.get("state") == "READY" for row in matches):
            continue
        if matches:
            if not any(row.get("state") == "CREATING" for row in matches):
                raise DeployError("composite index is in failed or unsupported state")
            wait_composites.append(wanted)
            for row in matches:
                if row.get("state") == "CREATING":
                    full_name = row.get("name")
                    for op in operations:
                        if not op.get("done") and isinstance(full_name, str) and full_name in _metadata_values(op, "index"):
                            if isinstance(op.get("name"), str):
                                op_names.add(op["name"])
        else:
            create.append(wanted)
    for group, path in fields:
        active = _active(operations, project, group, path=path)
        configured = _field_config_ready(field_rows[group], project, group, path)
        if active:
            wait_fields.append((group, path))
            op_names.update(op["name"] for op in active if isinstance(op.get("name"), str))
        elif not configured:
            wait_fields.append((group, path))
    return create, wait_composites, wait_fields, op_names


def _operation_response(value: Any) -> str:
    if not isinstance(value, dict) or not isinstance(value.get("name"), str) or not value["name"].strip():
        raise DeployError("async mutation returned no operation name")
    return value["name"]


def _composite_submission_reference(value: Any, project: str, index: dict[str, Any]) -> str:
    if isinstance(value, dict) and isinstance(value.get("name"), str) and value["name"].strip():
        return value["name"]
    metadata = value.get("metadata", value) if isinstance(value, dict) else None
    resource = metadata.get("index") if isinstance(metadata, dict) else None
    state = metadata.get("state") if isinstance(metadata, dict) else None
    prefix = (f"projects/{project}/databases/(default)/collectionGroups/"
              f"{index['collectionGroup']}/indexes/")
    if isinstance(resource, str) and isinstance(state, str) and state.strip() and resource.startswith(prefix):
        identifier = resource[len(prefix):]
        if identifier and "/" not in identifier:
            return resource
    raise DeployError("async mutation returned no operation name or valid index reference")


def _submit(project: str, database: str, indexes: list[dict[str, Any]],
            fields: list[tuple[str, str]], operations: set[str]) -> tuple[set[str], set[str]]:
    names = set(operations)
    index_references: set[str] = set()
    for index in indexes:
        args = ["gcloud", "firestore", "indexes", "composite", "create", "--project", project,
                "--database", database, "--collection-group", index["collectionGroup"],
                f"--query-scope={index.get('queryScope', 'COLLECTION').lower()}"]
        for field in index["fields"]:
            if "order" in field:
                args.append(f"--field-config=field-path={field['fieldPath']},order={field['order'].lower()}")
            elif field.get("arrayConfig") == "CONTAINS":
                args.append(f"--field-config=field-path={field['fieldPath']},array-config=contains")
            else:
                raise DeployError("unsupported canonical composite field config")
        args += ["--async", "--format=json"]
        try:
            response = _json(args)
            try:
                names.add(_operation_response(response))
            except DeployError:
                index_references.add(_composite_submission_reference(response, project, index))
        except DeployError as exc:
            if str(exc) != "ALREADY_EXISTS":
                raise
            print(f"Composite index create raced for {index['collectionGroup']}; rechecking state", flush=True)
        else:
            print(f"Submitted composite index for {index['collectionGroup']} asynchronously", flush=True)
    for group, path in fields:
        args = ["gcloud", "firestore", "indexes", "fields", "update", path, "--project", project,
                "--database", database, f"--collection-group={group}", "--disable-indexes", "--async", "--format=json"]
        try:
            names.add(_operation_response(_json(args)))
        except DeployError as exc:
            if str(exc) != "ALREADY_EXISTS":
                raise
            print(f"Field exemption update raced for {group}/{path}; rechecking state", flush=True)
        else:
            print(f"Submitted field exemption {group}/{path} asynchronously", flush=True)
    return names, index_references


def _resolve_submitted_indexes(project: str, database: str, references: set[str],
                               active_operations: list[dict[str, Any]], names: set[str]) -> None:
    """Resolve metadata-only create responses to their authoritative operation names."""
    unresolved = set(references)
    for reference in references:
        matches = [op for op in active_operations if reference in _metadata_values(op, "index")]
        if len(matches) > 1:
            raise DeployError(f"multiple active operations reference submitted index {reference}")
        if matches:
            operation_name = matches[0].get("name")
            if not isinstance(operation_name, str) or not operation_name.strip():
                raise DeployError(f"active operation for submitted index has no name: {reference}")
            names.add(operation_name)
            unresolved.discard(reference)
    if not unresolved:
        return
    history = _records(_json(["gcloud", "firestore", "operations", "list",
                              "--project", project, "--database", database,
                              "--limit=unlimited", "--format=json"]))
    for reference in unresolved:
        matches = [op for op in history if reference in _metadata_values(op, "index")]
        if len(matches) != 1:
            raise DeployError(f"could not uniquely identify operation for submitted index {reference}")
        operation_name = matches[0].get("name")
        if not isinstance(operation_name, str) or not operation_name.strip():
            raise DeployError(f"operation for submitted index has no name: {reference}")
        names.add(operation_name)


def _describe(project: str, database: str, name: str) -> dict[str, Any]:
    value = _json(["gcloud", "firestore", "operations", "describe", name,
                   "--project", project, "--database", database, "--format=json"])
    if not isinstance(value, dict):
        raise DeployError("malformed operation describe response")
    return value


def _wait(project: str, database: str, names: set[str], indexes: list[dict[str, Any]],
          fields: list[tuple[str, str]], timeout: float, interval: float,
          ) -> None:
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise DeployError("timed out waiting for Firestore index readiness")
        composites, field_rows, operations = _snapshot(project, database, fields)
        listed = {op.get("name"): op for op in operations if isinstance(op.get("name"), str)}
        operations_done = True
        for name in names:
            op = listed.get(name)
            if op is None:
                op = _describe(project, database, name)
            if _done(op):
                continue
            operations_done = False
        pending_composites = []
        for wanted in indexes:
            matches = [row for row in composites if _composite_matches(row, wanted)]
            if not any(row.get("state") == "READY" for row in matches):
                if matches and any(row.get("state") not in ("CREATING",) for row in matches):
                    raise DeployError("composite index entered a failed or unsupported state")
                pending_composites.append(wanted)
            elif any(_active(operations, project, wanted["collectionGroup"], index_name=row.get("name"))
                     for row in matches if isinstance(row.get("name"), str)):
                pending_composites.append(wanted)
        pending_fields = []
        for group, path in fields:
            configured = _field_config_ready(field_rows[group], project, group, path)
            active = _active(operations, project, group, path=path)
            if not configured or active:
                pending_fields.append((group, path))
        total = len(indexes) + len(fields)
        ready = total - len(pending_composites) - len(pending_fields)
        print(f"Firestore indexes: {ready}/{total} ready; {len(names)} tracked operations", flush=True)
        if operations_done and not pending_composites and not pending_fields:
            return
        time.sleep(min(interval, max(0.01, deadline - time.monotonic())))


def _collect_active_operations(project: str, indexes: list[dict[str, Any]],
                               composites: list[dict[str, Any]], operations: list[dict[str, Any]],
                               fields: list[tuple[str, str]], names: set[str]) -> None:
    """Track unfinished operations that affect any declared index resource."""
    for index in indexes:
        for op in operations:
            for row in composites:
                if (_composite_matches(row, index) and not op.get("done")
                        and _operation_relevant(op, project, index["collectionGroup"], index_name=row.get("name"))
                        and isinstance(op.get("name"), str)):
                    names.add(op["name"])
    for group, path in fields:
        names.update(op["name"] for op in _active(operations, project, group, path=path)
                     if isinstance(op.get("name"), str))


def _refresh_active_operations(project: str, indexes: list[dict[str, Any]],
                               composites: list[dict[str, Any]], operations: list[dict[str, Any]],
                               fields: list[tuple[str, str]], names: set[str]) -> None:
    """Add operation names discovered after submissions or an ALREADY_EXISTS race."""
    for index in indexes:
        matching = [row for row in composites if _composite_matches(row, index)]
        if any(row.get("state") not in ("CREATING", "READY") for row in matching):
            raise DeployError("composite index is in failed state")
    _collect_active_operations(project, indexes, composites, operations, fields, names)


def deploy(project: str, index_file: Path) -> None:
    timeout, interval = _settings()
    database = "(default)"
    indexes, fields = _load_declarations(index_file)
    composites, field_rows, operations = _snapshot(project, database, fields)
    create, waiting_composites, waiting_fields, names = _plan(
        project, indexes, fields, composites, field_rows, operations,
    )
    _collect_active_operations(project, indexes, composites, operations, fields, names)
    if not create and not waiting_composites and not waiting_fields and not names:
        print(f"Firestore indexes: {len(indexes) + len(fields)}/{len(indexes) + len(fields)} ready; no changes needed")
        return
    field_submissions = [(group, path) for group, path in waiting_fields
                         if not _active(operations, project, group, path=path)]
    names, index_references = _submit(project, database, create, field_submissions, names)
    composites, _field_rows, operations = _snapshot(project, database, fields)
    _resolve_submitted_indexes(project, database, index_references, operations, names)
    _refresh_active_operations(project, indexes, composites, operations, fields, names)
    _wait(project, database, names, indexes, fields, timeout, interval)
    print(f"Firestore indexes: {len(indexes) + len(fields)}/{len(indexes) + len(fields)} ready")


def main() -> int:
    try:
        root = Path(__file__).resolve().parents[4]
        deploy(os.environ.get("PROJECT_ID", "tlbrain-prod"), root / "infra" / "firestore" / "indexes.json")
        return 0
    except (DeployError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Firestore index deployment failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
