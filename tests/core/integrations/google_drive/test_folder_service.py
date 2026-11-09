"""Offline tests for coordinated folder operations."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from googleapiclient.errors import HttpError

from core.integrations.firestore.effects import get_effect_allocation
from core.integrations.firestore.operations import claim_operation, create_operation
from core.integrations.google_drive import folder_service
from core.integrations.firestore import folders as folder_repository
from core.operations.effects import allocate_folder_id
from core.operations.models import OperationOrigin
from tests.fixtures.sdk_fakes import fake_discovery_build


class _Request:
    def __init__(self, response):
        self.response = response

    def execute(self):
        return self.response


class _DriveFiles:
    def __init__(self, events, list_response=None, create_response=None):
        self.events = events
        self.list_response = list_response or {"files": []}
        self.create_response = create_response or {"id": "created-folder-id"}

    def list(self, **kwargs):
        self.events.append(("list", kwargs))
        return _Request(self.list_response)

    def create(self, **kwargs):
        self.events.append(("create", kwargs))
        return _Request(self.create_response)


class _Drive:
    def __init__(self, events, **kwargs):
        self._files = _DriveFiles(events, **kwargs)

    def files(self):
        return self._files


@pytest.mark.offline
def test_registered_folder_helpers_preserve_empty_path_and_repository_order(monkeypatch):
    calls = []
    folders = {
        "first": {"name": "Same", "parent_id": "root"},
        "second": {"name": "Same", "parent_id": "root"},
        "child": {"name": "Child", "parent_id": "first"},
    }

    def fail_root():
        calls.append("root")
        return "root"

    monkeypatch.setattr(folder_service, "get_root_folder_id", fail_root)
    monkeypatch.setattr(
        folder_service.folder_repository,
        "get_all_folders",
        lambda: calls.append("folders") or folders,
    )

    assert folder_service.resolve_registered_folder_path([]) == []
    assert calls == []
    assert folder_service.resolve_registered_folder_path(["Same"]) == ["first", "second"]
    assert calls == ["root", "folders"]
    calls.clear()
    assert folder_service.expand_registered_subtree("first") == ["first", "child"]
    assert calls == ["folders"]


@pytest.mark.offline
def test_unassigned_registry_hit_has_no_drive_or_firestore_effects(monkeypatch):
    events = []
    monkeypatch.setattr(
        folder_service.folder_repository,
        "get_all_folders",
        lambda: events.append(("registry",)) or {
            "unassigned-id": {"name": "_unassigned", "parent_id": "root-id"},
            "other-id": {"name": "_unassigned", "parent_id": "another-root"},
        },
    )
    monkeypatch.setattr(
        folder_service.folder_repository,
        "upsert_folder",
        lambda *args: events.append(("upsert", *args)),
    )

    result = folder_service.get_unassigned_folder_id(_Drive(events), "root-id")

    assert result == "unassigned-id"
    assert events == [("registry",)]


@pytest.mark.offline
@pytest.mark.parametrize(
    ("list_response", "create_response", "expected_drive_events", "stable_id"),
    [
        ({"files": [{"id": "existing-drive-id"}]}, None, 1, "existing-drive-id"),
        ({"files": []}, {"id": "created-drive-id"}, 2, "created-drive-id"),
    ],
)
def test_unassigned_drive_result_is_registered_with_stable_id_and_exact_request_shape(
    monkeypatch, list_response, create_response, expected_drive_events, stable_id
):
    events = []
    monkeypatch.setattr(
        folder_service.folder_repository,
        "get_all_folders",
        lambda: events.append(("registry",)) or {},
    )
    monkeypatch.setattr(
        folder_service.folder_repository,
        "upsert_folder",
        lambda *args: events.append(("upsert", *args)),
    )
    drive = _Drive(
        events,
        list_response=list_response,
        create_response=create_response or {"id": "unused-created-id"},
    )

    result = folder_service.get_unassigned_folder_id(drive, "root-id")

    assert result == stable_id
    expected_events = [
        ("registry",),
        (
        "list",
        {
            "q": "name='_unassigned' and mimeType='application/vnd.google-apps.folder' and 'root-id' in parents and trashed=false",
            "fields": "files(id)",
            "pageSize": 1,
        },
        ),
    ]
    if expected_drive_events == 2:
        expected_events.append((
            "create",
            {
                "body": {
                    "name": "_unassigned",
                    "mimeType": "application/vnd.google-apps.folder",
                    "parents": ["root-id"],
                },
                "fields": "id",
            },
        ))
    expected_events.append(("upsert", stable_id, "_unassigned", "root-id"))
    assert events == expected_events


@pytest.mark.offline
def test_create_folder_preserves_drive_id_and_effect_order(monkeypatch):
    events = []
    monkeypatch.setattr(folder_service, "_resolve", lambda path, role: events.append(("resolve", path, role)) or "parent-id")
    monkeypatch.setattr(
        folder_service.folder_repository,
        "folder_name_exists",
        lambda name, parent: events.append(("conflict", name, parent)) or False,
    )
    monkeypatch.setattr(
        folder_service.drive_client,
        "create_folder",
        lambda name, parent: events.append(("drive", name, parent)) or "stable-drive-id",
    )
    monkeypatch.setattr(
        folder_service.folder_repository,
        "upsert_folder",
        lambda *args: events.append(("firestore", *args)),
    )

    result = folder_service.create_folder("New", ["Parent"])

    assert result == "stable-drive-id"
    assert events == [
        ("resolve", ["Parent"], "parent"),
        ("conflict", "New", "parent-id"),
        ("drive", "New", "parent-id"),
        ("firestore", "stable-drive-id", "New", "parent-id"),
    ]


@pytest.mark.offline
def test_stable_folder_creation_excludes_its_own_registry_record_and_registers_after_drive(monkeypatch):
    events = []
    monkeypatch.setattr(folder_service, "_resolve", lambda path, role: "parent-id")
    monkeypatch.setattr(
        folder_service.folder_repository, "folder_name_exists",
        lambda name, parent, *, exclude_id=None: events.append(("conflict", name, parent, exclude_id)) or False,
    )
    monkeypatch.setattr(
        folder_service.drive_client, "create_folder",
        lambda name, parent, *, folder_id, intent_marker: events.append(
            ("drive", name, parent, folder_id, intent_marker)
        ) or folder_id,
    )
    monkeypatch.setattr(
        folder_service.folder_repository, "upsert_folder",
        lambda *args: events.append(("register", *args)),
    )

    result = folder_service.create_folder(
        "New", ["Parent"], folder_id="planned-folder-id", intent_marker="v1:" + "a" * 64,
    )

    assert result == "planned-folder-id"
    assert events == [
        ("conflict", "New", "parent-id", "planned-folder-id"),
        ("drive", "New", "parent-id", "planned-folder-id", "v1:" + "a" * 64),
        ("register", "planned-folder-id", "New", "parent-id"),
    ]


@pytest.mark.offline
def test_repository_exclusion_ignores_only_own_folder_and_keeps_sibling_conflicts(firestore_fake):
    client = firestore_fake.client_class()
    client.collection("folders").document("own-folder").set({
        "folder_id": "own-folder", "name": "Same", "parent_id": "parent",
    })

    assert folder_repository.folder_name_exists("Same", "parent", client, exclude_id="own-folder") is False
    client.collection("folders").document("other-folder").set({
        "folder_id": "other-folder", "name": "Same", "parent_id": "parent",
    })
    assert folder_repository.folder_name_exists("Same", "parent", client, exclude_id="own-folder") is True


@pytest.mark.offline
def test_stable_folder_replay_still_rejects_another_registered_sibling(firestore_fake, monkeypatch):
    client = firestore_fake.client_class()
    client.collection("folders").document("own-folder").set({
        "folder_id": "own-folder", "name": "Same", "parent_id": "root-id",
    })
    client.collection("folders").document("other-folder").set({
        "folder_id": "other-folder", "name": "Same", "parent_id": "root-id",
    })
    drive_calls = []
    monkeypatch.setattr(folder_service, "get_root_folder_id", lambda: "root-id")
    monkeypatch.setattr(folder_service.drive_client, "create_folder", lambda *args, **kwargs: drive_calls.append(True))

    with pytest.raises(folder_service.FolderNameConflict):
        folder_service.create_folder("Same", folder_id="own-folder", intent_marker="v1:" + "a" * 64)
    assert drive_calls == []


@pytest.mark.offline
def test_stable_folder_replay_recovers_same_id_after_lost_create_response_and_repairs_registry(
    firestore_fake, sdk_fakes, monkeypatch,
):
    client = firestore_fake.client_class()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    create_operation(
        client, operation_id="folder-recovery-op", origin=OperationOrigin.MCP,
        caller_binding="caller", immutable_input_digest="sha256:" + "a" * 64,
        immutable_input_ref="input-ref", now=now,
    )
    lease = claim_operation(client, "folder-recovery-op", owner="worker", now=now, lease_seconds=60)
    allocation = allocate_folder_id(
        client, operation_id="folder-recovery-op", effect_id="folder-effect",
        intent_digest="sha256:" + "b" * 64, owner="worker", fence=lease.fence, now=now,
        id_factory=lambda: "planned-folder-id",
    )
    marker = allocation["intent_marker"]
    accepted_folder = {
        "id": "planned-folder-id", "mimeType": "application/vnd.google-apps.folder", "name": "Recovered",
        "parents": ["root-id"], "trashed": False,
        "appProperties": {"tlbrain_effect_intent": marker},
    }
    sdk_fakes.drive.queue("files.create", error=TimeoutError("private timeout detail"))
    sdk_fakes.drive.queue("files.create", error=HttpError(SimpleNamespace(status=409, reason="Conflict"), b"conflict"))
    sdk_fakes.drive.queue("files.get", response=accepted_folder)
    monkeypatch.setattr(folder_service, "get_root_folder_id", lambda: "root-id")
    monkeypatch.setattr(folder_service.drive_client, "build_drive_service_rw", lambda: fake_discovery_build("drive", "v3"))

    real_create = folder_service.drive_client.create_folder
    def check_allocation_before_remote_create(*args, **kwargs):
        persisted = get_effect_allocation(client, "folder-recovery-op", "folder-effect")
        assert persisted["resource_id"] == kwargs["folder_id"]
        return real_create(*args, **kwargs)
    monkeypatch.setattr(folder_service.drive_client, "create_folder", check_allocation_before_remote_create)

    with pytest.raises(RuntimeError, match="Drive folder creation outcome is unresolved") as first_error:
        folder_service.create_folder("Recovered", folder_id=allocation["resource_id"], intent_marker=marker)
    assert "private timeout detail" not in str(first_error.value)
    assert folder_repository.get_folder_by_id("planned-folder-id", client) is None

    replay = allocate_folder_id(
        client, operation_id="folder-recovery-op", effect_id="folder-effect",
        intent_digest="sha256:" + "b" * 64, owner="worker", fence=lease.fence, now=now,
        id_factory=lambda: pytest.fail("replay must reuse the persisted folder ID"),
    )
    assert replay == allocation
    assert folder_service.create_folder(
        "Recovered", folder_id=replay["resource_id"], intent_marker=replay["intent_marker"],
    ) == "planned-folder-id"
    assert folder_repository.get_folder_by_id("planned-folder-id", client)["name"] == "Recovered"
    events = [event["operation"] for event in sdk_fakes.trace if event["boundary"] == "drive"]
    assert events.count("files.create") == 2
    assert events.count("files.get") == 1
    assert "files.generateIds" not in events
    assert "files.list" not in events


@pytest.mark.offline
@pytest.mark.parametrize("readback", [
    {"id": "wrong-id"},
    {"mimeType": "application/vnd.google-apps.document"},
    {"name": "Different"},
    {"parents": ["wrong-parent"]},
    {"parents": "root-id"},
    {"parents": ["root-id", "other-parent"]},
    {"trashed": True},
    {"appProperties": {"tlbrain_effect_intent": "v1:" + "0" * 64}},
    {"appProperties": "malformed"},
    None,
])
def test_stable_folder_mismatching_or_malformed_readback_never_registers(
    sdk_fakes, monkeypatch, readback,
):
    marker = "v1:" + "a" * 64
    accepted = {
        "id": "planned-folder-id", "mimeType": "application/vnd.google-apps.folder", "name": "Recovered",
        "parents": ["root-id"], "trashed": False,
        "appProperties": {"tlbrain_effect_intent": marker},
    }
    if readback is None:
        accepted = None
    else:
        accepted.update(readback)
    sdk_fakes.drive.queue("files.create", error=HttpError(SimpleNamespace(status=409, reason="Conflict"), b"conflict"))
    sdk_fakes.drive.queue("files.get", response=accepted)
    registrations = []
    monkeypatch.setattr(folder_service, "get_root_folder_id", lambda: "root-id")
    monkeypatch.setattr(folder_service.drive_client, "build_drive_service_rw", lambda: fake_discovery_build("drive", "v3"))
    monkeypatch.setattr(folder_service.folder_repository, "folder_name_exists", lambda *args, **kwargs: False)
    monkeypatch.setattr(folder_service.folder_repository, "upsert_folder", lambda *args: registrations.append(args))

    with pytest.raises(RuntimeError):
        folder_service.create_folder("Recovered", folder_id="planned-folder-id", intent_marker=marker)
    assert registrations == []
    operations = [event["operation"] for event in sdk_fakes.trace if event["boundary"] == "drive"]
    assert operations.count("files.create") == 1
    assert operations.count("files.get") == 1


@pytest.mark.offline
def test_stable_folder_unreadable_readback_never_registers(sdk_fakes, monkeypatch):
    marker = "v1:" + "a" * 64
    sdk_fakes.drive.queue("files.create", error=HttpError(SimpleNamespace(status=409, reason="Conflict"), b"conflict"))
    sdk_fakes.drive.queue("files.get", error=RuntimeError("private provider detail"))
    registrations = []
    monkeypatch.setattr(folder_service, "get_root_folder_id", lambda: "root-id")
    monkeypatch.setattr(folder_service.drive_client, "build_drive_service_rw", lambda: fake_discovery_build("drive", "v3"))
    monkeypatch.setattr(folder_service.folder_repository, "folder_name_exists", lambda *args, **kwargs: False)
    monkeypatch.setattr(folder_service.folder_repository, "upsert_folder", lambda *args: registrations.append(args))

    with pytest.raises(RuntimeError, match="Drive folder creation outcome is unresolved") as error:
        folder_service.create_folder("Recovered", folder_id="planned-folder-id", intent_marker=marker)
    assert "private provider detail" not in str(error.value)
    assert registrations == []


@pytest.mark.offline
def test_after_commit_registry_failure_replays_own_folder_and_repairs_safely(
    firestore_fake, sdk_fakes, monkeypatch,
):
    client = firestore_fake.client_class()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    create_operation(
        client, operation_id="folder-register-replay-op", origin=OperationOrigin.MCP,
        caller_binding="caller", immutable_input_digest="sha256:" + "a" * 64,
        immutable_input_ref="input-ref", now=now,
    )
    lease = claim_operation(client, "folder-register-replay-op", owner="worker", now=now, lease_seconds=60)
    allocation = allocate_folder_id(
        client, operation_id="folder-register-replay-op", effect_id="folder-effect",
        intent_digest="sha256:" + "b" * 64, owner="worker", fence=lease.fence, now=now,
        id_factory=lambda: "registered-after-commit-id",
    )
    response = {"id": allocation["resource_id"]}
    readback = {
        "id": allocation["resource_id"], "mimeType": "application/vnd.google-apps.folder", "name": "Recovered",
        "parents": ["root-id"], "trashed": False,
        "appProperties": {"tlbrain_effect_intent": allocation["intent_marker"]},
    }
    sdk_fakes.drive.queue("files.create", response=response)
    sdk_fakes.drive.queue("files.create", error=HttpError(SimpleNamespace(status=409, reason="Conflict"), b"conflict"))
    sdk_fakes.drive.queue("files.get", response=readback)
    monkeypatch.setattr(folder_service, "get_root_folder_id", lambda: "root-id")
    monkeypatch.setattr(folder_service.drive_client, "build_drive_service_rw", lambda: fake_discovery_build("drive", "v3"))

    row_path = f"folders/{allocation['resource_id']}"
    firestore_fake.fail_next(
        "set", path=row_path, phase="after", exception=RuntimeError("private Firestore detail"),
    )
    real_create = folder_service.drive_client.create_folder
    def check_allocation_before_remote_create(*args, **kwargs):
        stored = get_effect_allocation(client, "folder-register-replay-op", "folder-effect")
        assert stored["resource_id"] == kwargs["folder_id"]
        return real_create(*args, **kwargs)
    monkeypatch.setattr(folder_service.drive_client, "create_folder", check_allocation_before_remote_create)

    with pytest.raises(RuntimeError, match="Drive folder registration outcome is unresolved") as first_error:
        folder_service.create_folder(
            "Recovered", folder_id=allocation["resource_id"], intent_marker=allocation["intent_marker"],
        )
    assert "private Firestore detail" not in str(first_error.value)
    assert folder_repository.get_folder_by_id(allocation["resource_id"], client)["name"] == "Recovered"

    replay = allocate_folder_id(
        client, operation_id="folder-register-replay-op", effect_id="folder-effect",
        intent_digest="sha256:" + "b" * 64, owner="worker", fence=lease.fence, now=now,
        id_factory=lambda: pytest.fail("replay must reuse the persisted folder ID"),
    )
    assert folder_service.create_folder(
        "Recovered", folder_id=replay["resource_id"], intent_marker=replay["intent_marker"],
    ) == allocation["resource_id"]
    assert folder_repository.get_folder_by_id(allocation["resource_id"], client)["name"] == "Recovered"
    drive_ops = [event["operation"] for event in sdk_fakes.trace if event["boundary"] == "drive"]
    assert drive_ops.count("files.create") == 2
    assert drive_ops.count("files.get") == 1
    assert "files.generateIds" not in drive_ops
    assert "files.list" not in drive_ops


@pytest.mark.offline
def test_create_folder_rejects_sibling_duplicate_without_drive_or_firestore_effects(monkeypatch):
    events = []
    monkeypatch.setattr(folder_service, "get_root_folder_id", lambda: "root-id")
    monkeypatch.setattr(
        folder_service.folder_repository,
        "folder_name_exists",
        lambda name, parent: events.append(("conflict", name, parent)) or True,
    )
    monkeypatch.setattr(
        folder_service.drive_client,
        "create_folder",
        lambda *args: events.append(("drive_create", *args)),
    )
    monkeypatch.setattr(
        folder_service.folder_repository,
        "upsert_folder",
        lambda *args: events.append(("firestore_upsert", *args)),
    )

    with pytest.raises(folder_service.FolderNameConflict):
        folder_service.create_folder("Existing", None)

    assert events == [("conflict", "Existing", "root-id")]


@pytest.mark.offline
def test_rename_checks_sibling_before_drive_and_updates_same_id(monkeypatch):
    events = []
    monkeypatch.setattr(folder_service, "_resolve", lambda path, role: events.append(("resolve", path, role)) or "folder-id")
    monkeypatch.setattr(
        folder_service.folder_repository,
        "get_folder_by_id",
        lambda folder_id: events.append(("get", folder_id)) or {"parent_id": "parent-id"},
    )
    monkeypatch.setattr(
        folder_service.folder_repository,
        "folder_name_exists",
        lambda name, parent: events.append(("conflict", name, parent)) or False,
    )
    monkeypatch.setattr(
        folder_service.drive_client,
        "rename_folder",
        lambda *args: events.append(("drive", *args)),
    )
    monkeypatch.setattr(
        folder_service.folder_repository,
        "upsert_folder",
        lambda *args: events.append(("firestore", *args)),
    )

    assert folder_service.rename_folder(["Old"], "New") == "folder-id"
    assert events == [
        ("resolve", ["Old"], "folder"),
        ("get", "folder-id"),
        ("conflict", "New", "parent-id"),
        ("drive", "folder-id", "New"),
        ("firestore", "folder-id", "New", "parent-id"),
    ]


@pytest.mark.offline
def test_create_raises_typed_parent_path_error(monkeypatch):
    monkeypatch.setattr(folder_service, "get_root_folder_id", lambda: "root-id")
    monkeypatch.setattr(folder_service.folder_repository, "get_all_folders", lambda: {})

    with pytest.raises(folder_service.FolderNotFound) as error:
        folder_service.create_folder("New", ["Missing"])

    assert error.value.path == ["Missing"]
    assert error.value.role == "parent"


@pytest.mark.offline
def test_move_folder_preserves_id_and_effect_order(monkeypatch):
    events = []

    def resolve(path, role):
        events.append(("resolve", path, role))
        return {("Source",): "folder-id", ("Destination",): "destination-id"}[tuple(path)]

    def get_all_folders():
        events.append(("cycle_snapshot",))
        return {
            "folder-id": {"name": "Source", "parent_id": "root-id"},
            "child-id": {"name": "Child", "parent_id": "folder-id"},
            "destination-id": {"name": "Destination", "parent_id": "root-id"},
        }

    def get_folder_by_id(folder_id):
        events.append(("get_folder", folder_id))
        return {"name": "Source", "parent_id": "root-id"}

    monkeypatch.setattr(folder_service, "_resolve", resolve)
    monkeypatch.setattr(
        folder_service.folder_repository, "get_all_folders", get_all_folders
    )
    monkeypatch.setattr(
        folder_service.folder_repository, "get_folder_by_id", get_folder_by_id
    )
    monkeypatch.setattr(
        folder_service.drive_client,
        "move_folder_in_drive",
        lambda *args: events.append(("drive_move", *args)),
    )
    monkeypatch.setattr(
        folder_service.folder_repository,
        "upsert_folder",
        lambda *args: events.append(("firestore_upsert", *args)),
    )

    result = folder_service.move_folder(["Source"], ["Destination"])

    assert result == ("folder-id", "destination-id")
    assert events == [
        ("resolve", ["Source"], "folder"),
        ("resolve", ["Destination"], "parent"),
        ("cycle_snapshot",),
        ("get_folder", "folder-id"),
        ("drive_move", "folder-id", "destination-id"),
        ("firestore_upsert", "folder-id", "Source", "destination-id"),
    ]

