"""Offline contract tests for async Firestore index deployment."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "infra/deploy/core/helpers/firestore_indexes.py"
INDEX_FILE = ROOT / "infra/firestore/indexes.json"


def _module():
    spec = importlib.util.spec_from_file_location("firestore_indexes_under_test", HELPER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeGcloud:
    def __init__(self, *, ready: bool = False, fail_mutation: bool = False,
                 malformed: bool = False, operation_error: bool = False,
                 stuck: bool = False, final_mismatch: bool = False,
                 missing_response: bool = False, hide_operations: bool = False,
                 metadata_only_composite: bool = False, omit_index_operation: bool = False,
                 complete_before_discovery: bool = False):
        source = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        self.declarations = source
        self.events: list[list[str]] = []
        self.fail_mutation = fail_mutation
        self.malformed = malformed
        self.operation_error = operation_error
        self.stuck = stuck
        self.final_mismatch = final_mismatch
        self.missing_response = missing_response
        self.hide_operations = hide_operations
        self.metadata_only_composite = metadata_only_composite
        self.omit_index_operation = omit_index_operation
        self.complete_before_discovery = complete_before_discovery
        self.operation_list_calls = 0
        self.operations: list[dict] = []
        self.composites = []
        self.fields: dict[str, list[dict]] = {}
        if ready:
            self.composites = [dict(row, name=(
                f"projects/synthetic-project/databases/(default)/collectionGroups/{row['collectionGroup']}/indexes/i{n}"
            ), fields=[*row["fields"], {"fieldPath": "__name__", "order": "DESCENDING"}], state="READY")
                for n, row in enumerate(source["indexes"])]
            for override in source["fieldOverrides"]:
                group, path = override["collectionGroup"], override["fieldPath"]
                self.fields.setdefault(group, []).append({
                    "name": f"projects/synthetic-project/databases/(default)/collectionGroups/{group}/fields/{path}",
                    "indexConfig": {"indexes": [], "usesAncestorConfig": False},
                })

    def run(self, argv, **_kwargs):
        self.events.append(list(argv))
        command = argv[1:5]
        if self.malformed and argv[1:3] == ["firestore", "indexes"]:
            return SimpleNamespace(returncode=0, stdout="{invalid", stderr="")
        if argv[1:4] == ["firestore", "indexes", "composite"] and argv[4] == "list":
            return SimpleNamespace(returncode=0, stdout=json.dumps(self.composites), stderr="")
        if argv[1:4] == ["firestore", "indexes", "fields"] and argv[4] == "list":
            group = next(arg.split("=", 1)[1] for arg in argv if arg.startswith("--collection-group="))
            return SimpleNamespace(returncode=0, stdout=json.dumps(self.fields.get(group, [])), stderr="")
        if argv[1:3] == ["firestore", "operations"]:
            if argv[3] == "describe":
                name = argv[4]
                row = next((op for op in self.operations if op["name"] == name), None)
                if row and not row.get("done") and not self.stuck:
                    row["done"] = True
                    row["metadata"]["state"] = "SUCCESSFUL"
                    row["error"] = None
                    if not self.final_mismatch:
                        for composite in self.composites:
                            composite["state"] = "READY"
                return SimpleNamespace(returncode=0, stdout=json.dumps(row or {}), stderr="")
            active_only = "--filter=done=false" in argv
            if active_only:
                self.operation_list_calls += 1
            if self.hide_operations:
                return SimpleNamespace(returncode=0, stdout="[]", stderr="")
            if (active_only and self.operation_list_calls > 2
                    and any(op.get("done") is False for op in self.operations) and not self.stuck):
                for op in self.operations:
                    op["done"] = True
                    op["error"] = {"code": 7, "message": "synthetic failure"} if self.operation_error else None
                    op["metadata"]["state"] = "FAILED" if self.operation_error else "SUCCESSFUL"
                if not self.final_mismatch:
                    for row in self.composites:
                        row["state"] = "READY"
            rows = [op for op in self.operations if not op.get("done")] if active_only else self.operations
            return SimpleNamespace(returncode=0, stdout=json.dumps(rows), stderr="")
        if argv[1:5] == ["firestore", "indexes", "composite", "create"]:
            if self.fail_mutation:
                return SimpleNamespace(returncode=1, stdout="", stderr="PERMISSION_DENIED")
            group = argv[argv.index("--collection-group") + 1]
            scope = next(arg.split("=", 1)[1].upper() for arg in argv if arg.startswith("--query-scope="))
            fields = []
            for arg in argv:
                if arg.startswith("--field-config="):
                    conf = dict(part.split("=", 1) for part in arg.split("=", 1)[1].split(","))
                    fields.append({"fieldPath": conf["field-path"], "order": conf["order"].upper()} if "order" in conf
                                  else {"fieldPath": conf["field-path"], "arrayConfig": "CONTAINS"})
            index_name = f"projects/synthetic-project/databases/(default)/collectionGroups/{group}/indexes/i{len(self.composites)}"
            self.composites.append({"name": index_name,
                                    "queryScope": scope, "fields": fields, "state": "CREATING"})
            name = f"operations/composite-{len(self.operations)}"
            if not self.omit_index_operation:
                self.operations.append({"name": name, "done": self.complete_before_discovery,
                                        "metadata": {"index": index_name,
                                                     "state": "SUCCESSFUL" if self.complete_before_discovery
                                                     else "PROCESSING"}})
            if self.complete_before_discovery:
                self.composites[-1]["state"] = "READY"
            if self.missing_response:
                response = {}
            elif self.metadata_only_composite:
                response = {"index": index_name, "state": "INITIALIZING"}
            else:
                response = {"name": name}
            return SimpleNamespace(returncode=0, stdout=json.dumps(response), stderr="")
        if argv[1:5] == ["firestore", "indexes", "fields", "update"]:
            if self.fail_mutation:
                return SimpleNamespace(returncode=1, stdout="", stderr="PERMISSION_DENIED")
            group = next(arg.split("=", 1)[1] for arg in argv if arg.startswith("--collection-group="))
            path = argv[5]
            self.fields.setdefault(group, []).append({
                "name": f"projects/synthetic-project/databases/(default)/collectionGroups/{group}/fields/{path}",
                "indexConfig": {"indexes": [], "usesAncestorConfig": False},
            })
            name = f"operations/field-{len(self.operations)}"
            self.operations.append({"name": name, "done": False,
                                    "metadata": {"field": f"projects/synthetic-project/databases/(default)/collectionGroups/{group}/fields/{path}",
                                                 "state": "PROCESSING"}})
            return SimpleNamespace(returncode=0, stdout="{}" if self.missing_response else json.dumps({"name": name}), stderr="")
        raise AssertionError(argv)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def _run(monkeypatch, fake: FakeGcloud):
    module = _module()
    clock = FakeClock()
    monkeypatch.setattr(module.subprocess, "run", fake.run)
    monkeypatch.setattr(module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(module.time, "sleep", clock.sleep)
    monkeypatch.setenv("FIRESTORE_INDEX_POLL_SECONDS", "0.2")
    monkeypatch.setenv("FIRESTORE_INDEX_TIMEOUT_SECONDS", "2")
    module.deploy("synthetic-project", INDEX_FILE)
    return fake.events


def test_all_missing_declarations_submit_async_before_collective_poll(monkeypatch):
    fake = FakeGcloud()
    events = _run(monkeypatch, fake)
    mutations = [event for event in events if event[1:5] in (
        ["firestore", "indexes", "composite", "create"],
        ["firestore", "indexes", "fields", "update"],
    )]
    assert len(mutations) == 15
    assert all("--async" in event and "--project" in event and "--database" in event for event in mutations)
    operation_reads = [i for i, event in enumerate(events) if event[1:3] == ["firestore", "operations"]]
    assert len(operation_reads) >= 2  # initial active-operation read, then collective polling
    assert all(events.index(mutation) < operation_reads[1] for mutation in mutations)
    assert all(op["done"] is True for op in fake.operations)


def test_metadata_only_composite_response_resolves_active_operation(monkeypatch):
    fake = FakeGcloud(metadata_only_composite=True)
    events = _run(monkeypatch, fake)
    assert len([event for event in events if event[1:5] == ["firestore", "indexes", "composite", "create"]]) == 6
    assert all(op["done"] is True for op in fake.operations)
    assert any(event[1:4] == ["firestore", "operations", "list"] and "--filter=done=false" in event
               for event in events)


def test_metadata_only_composite_response_resolves_completed_operation_from_history(monkeypatch):
    fake = FakeGcloud(metadata_only_composite=True, complete_before_discovery=True)
    events = _run(monkeypatch, fake)
    history_reads = [event for event in events if event[1:4] == ["firestore", "operations", "list"]
                     and "--filter=done=false" not in event]
    assert history_reads and "--limit=unlimited" in history_reads[0]


def test_metadata_only_composite_response_fails_when_operation_cannot_be_identified(monkeypatch):
    fake = FakeGcloud(metadata_only_composite=True, omit_index_operation=True)
    module = _module()
    monkeypatch.setattr(module.subprocess, "run", fake.run)
    with pytest.raises(module.DeployError, match="could not uniquely identify operation"):
        module.deploy("synthetic-project", INDEX_FILE)
    history_reads = [event for event in fake.events if event[1:4] == ["firestore", "operations", "list"]
                     and "--filter=done=false" not in event]
    assert history_reads


def test_ready_state_is_idempotent(monkeypatch):
    fake = FakeGcloud(ready=True)
    events = _run(monkeypatch, fake)
    assert not any(event[1:4] == ["firestore", "indexes", "composite"] and event[4] == "create" for event in events)
    assert not any(event[1:4] == ["firestore", "indexes", "fields"] and event[4] == "update" for event in events)


def test_ready_name_only_composites_with_implicit_name_segment_are_recognized(monkeypatch):
    fake = FakeGcloud(ready=True)
    for row in fake.composites:
        row.pop("collectionGroup")
        row["name"] += "/unused-name-segment"
    _run(monkeypatch, fake)
    assert not any(event[1:5] == ["firestore", "indexes", "composite", "create"] for event in fake.events)


def test_field_exemption_accepts_omitted_false_but_rejects_inherited_or_reverting_config():
    module = _module()
    project, group, path = "synthetic-project", "messages", "text"
    name = f"projects/{project}/databases/(default)/collectionGroups/{group}/fields/{path}"
    assert module._field_config_ready([{"name": name, "indexConfig": {"indexes": []}}], project, group, path)
    assert not module._field_config_ready([{"name": name, "indexConfig": {"indexes": [], "usesAncestorConfig": True}}], project, group, path)
    assert not module._field_config_ready([{"name": name, "indexConfig": {"indexes": [], "reverting": True}}], project, group, path)


def test_run_uses_windows_cmd_path_returned_by_shutil_which(monkeypatch, tmp_path):
    module = _module()
    cmd_path = str(tmp_path / "gcloud.cmd")
    called = {}

    def fake_run(command, **kwargs):
        called["command"] = command
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(module.shutil, "which", lambda name: cmd_path if name == "gcloud" else None)
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert module._run(["gcloud", "info"]) == "ok"
    assert called["command"] == [cmd_path, "info"]


def test_nonempty_field_override_indexes_fail_before_any_cloud_mutation(tmp_path, monkeypatch):
    module = _module()
    source = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    source["fieldOverrides"][0]["indexes"] = [{"order": "ASCENDING"}]
    declaration = tmp_path / "indexes.json"
    declaration.write_text(json.dumps(source), encoding="utf-8")
    fake = FakeGcloud()
    monkeypatch.setattr(module.subprocess, "run", fake.run)
    with pytest.raises(module.DeployError, match="field override indexes"):
        module.deploy("synthetic-project", declaration)
    assert fake.events == []


def test_creating_composite_and_visible_field_config_reuse_active_operation(monkeypatch):
    fake = FakeGcloud()
    creating = fake.declarations["indexes"][-1]
    creating_name = f"projects/synthetic-project/databases/(default)/collectionGroups/{creating['collectionGroup']}/indexes/creating"
    fake.composites.append({**creating, "name": creating_name, "state": "CREATING"})
    fake.operations.append({"name": "operations/existing-composite", "metadata": {
        "index": creating_name, "state": "PROCESSING",
    }})
    group, path = "messages", "text"
    fake.fields[group] = [{
        "name": f"projects/synthetic-project/databases/(default)/collectionGroups/{group}/fields/{path}",
        "indexConfig": {"ancestorField": "projects/p/databases/(default)/collectionGroups/__default__/fields/*"},
    }]
    active_name = f"projects/synthetic-project/databases/(default)/collectionGroups/{group}/fields/{path}"
    fake.operations.append({"name": "operations/existing-field", "metadata": {
        "@type": "type.googleapis.com/google.firestore.admin.v1.FieldOperationMetadata",
        "field": active_name, "state": "PROCESSING",
    }})
    events = _run(monkeypatch, fake)
    assert not any(event[1:5] == ["firestore", "indexes", "composite", "create"]
                   and event[event.index("--collection-group") + 1] == creating["collectionGroup"]
                   for event in events)
    assert not any(event[1:5] == ["firestore", "indexes", "fields", "update"]
                   and event[5] == path and f"--collection-group={group}" in event
                   for event in events)
    assert fake.operations[0]["done"] is True
    assert fake.operations[1]["done"] is True


def test_mutation_failure_fails_closed(monkeypatch):
    fake = FakeGcloud(fail_mutation=True)
    module = _module()
    monkeypatch.setattr(module.subprocess, "run", fake.run)
    with pytest.raises(module.DeployError, match="gcloud command failed"):
        module.deploy("synthetic-project", INDEX_FILE)
    assert len([event for event in fake.events if event[1:3] == ["firestore", "operations"]]) == 1


@pytest.mark.parametrize("fake, message", [
    (FakeGcloud(malformed=True), "malformed JSON"),
    (FakeGcloud(operation_error=True), "failed state|completed with an error"),
    (FakeGcloud(stuck=True), "timed out waiting"),
    (FakeGcloud(final_mismatch=True), "timed out waiting"),
    (FakeGcloud(missing_response=True), "no operation name"),
], ids=["invalid-json", "operation-error", "timeout", "final-mismatch", "missing-operation-name"])
def test_malformed_or_incomplete_remote_state_fails_closed(monkeypatch, fake, message):
    module = _module()
    clock = FakeClock()
    monkeypatch.setattr(module.subprocess, "run", fake.run)
    monkeypatch.setattr(module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(module.time, "sleep", clock.sleep)
    monkeypatch.setenv("FIRESTORE_INDEX_TIMEOUT_SECONDS", "0.5")
    monkeypatch.setenv("FIRESTORE_INDEX_POLL_SECONDS", "0.1")
    with pytest.raises(module.DeployError, match=message):
        module.deploy("synthetic-project", INDEX_FILE)


def test_tracked_operations_are_described_when_list_omits_them(monkeypatch):
    fake = FakeGcloud(hide_operations=True)
    events = _run(monkeypatch, fake)
    assert any(event[1:4] == ["firestore", "operations", "describe"] for event in events)
    assert all("--database" in event for event in events if event[1:3] == ["firestore", "operations"])
    assert any("--filter=done=false" in event and "--limit=unlimited" in event for event in events
               if event[1:4] == ["firestore", "operations", "list"])


def test_already_exists_race_is_reread_and_verified(monkeypatch):
    class RacingGcloud(FakeGcloud):
        raced = False

        def run(self, argv, **kwargs):
            result = super().run(argv, **kwargs)
            if not self.raced and argv[1:5] == ["firestore", "indexes", "composite", "create"]:
                self.raced = True
                return SimpleNamespace(returncode=1, stdout="", stderr="ALREADY_EXISTS")
            return result

    fake = RacingGcloud()
    events = _run(monkeypatch, fake)
    assert fake.raced
    assert any(event[1:5] == ["firestore", "indexes", "composite", "list"] for event in events)
    assert all(op["done"] is True for op in fake.operations)


def test_preflight_python_mode_executes_candidate_arguments():
    source = (ROOT / "infra/build/preflight.sh").read_text(encoding="utf-8")
    assert '"${1:-}" == "--python-mode"' in source
    assert 'exec "$candidate" "$@"' in source
    assert "WindowsApps" in source
