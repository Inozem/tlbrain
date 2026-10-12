"""Characterization tests for the SDK-free Google Drive folder helpers."""

from __future__ import annotations

import pytest


ROOT_ID = "root"


@pytest.fixture
def folders():
    return {
        "alpha": {"name": "Alpha", "parent_id": ROOT_ID},
        "alpha-child": {"name": "Child", "parent_id": "alpha"},
        "alpha-child-duplicate": {"name": "Child", "parent_id": "alpha"},
        "alpha-second-child": {"name": "Second", "parent_id": "alpha"},
        "alpha-grandchild": {"name": "Grandchild", "parent_id": "alpha-child"},
        "beta": {"name": "Beta", "parent_id": ROOT_ID},
        "beta-child": {"name": "Child", "parent_id": "beta"},
        "zeta": {"name": "Zeta", "parent_id": ROOT_ID},
        "unassigned": {"name": "_unassigned", "parent_id": ROOT_ID},
        "outside": {"name": "Outside", "parent_id": "another-root"},
    }


@pytest.mark.offline
def test_resolve_folder_path_returns_every_duplicate_terminal_match(folders):
    from core.integrations.google_drive.folders import resolve_folder_path

    assert set(resolve_folder_path(["Alpha", "Child"], folders, ROOT_ID)) == {
        "alpha-child",
        "alpha-child-duplicate",
    }


@pytest.mark.offline
def test_resolve_folder_path_returns_empty_for_unknown_or_empty_path(folders):
    from core.integrations.google_drive.folders import resolve_folder_path

    assert resolve_folder_path([], folders, ROOT_ID) == []
    assert resolve_folder_path(["Missing"], folders, ROOT_ID) == []
    assert resolve_folder_path(["Alpha", "Missing"], folders, ROOT_ID) == []


@pytest.mark.offline
def test_expand_subtree_includes_starting_folder_and_descendants(folders):
    from core.integrations.google_drive.folders import expand_subtree

    assert set(expand_subtree("alpha", folders)) == {
        "alpha",
        "alpha-child",
        "alpha-child-duplicate",
        "alpha-second-child",
        "alpha-grandchild",
    }
    assert expand_subtree("missing", folders) == ["missing"]


@pytest.mark.offline
def test_folder_tree_shape_sorting_aggregates_and_unassigned_suggestion(folders):
    from core.integrations.google_drive.folders import build_folder_tree

    result = build_folder_tree(
        folders,
        ROOT_ID,
        {
            "alpha": {"count": 2, "last_date": "2026-10-10", "last_doc_id": "doc-a"},
            "alpha-child": {"count": 1, "last_date": "2026-10-11", "last_doc_id": "doc-child"},
            "unassigned": {"count": 3},
        },
    )

    assert [node["name"] for node in result["folders"]] == [
        "Alpha",
        "Beta",
        "Zeta",
        "_unassigned",
    ]
    alpha = result["folders"][0]
    assert alpha == {
        "name": "Alpha",
        "dialog_count": 2,
        "last_dialog_date": "2026-10-10",
        "last_dialog_doc_id": "doc-a",
        "children": [
            {
                "name": "Child",
                "dialog_count": 1,
                "last_dialog_date": "2026-10-11",
                "last_dialog_doc_id": "doc-child",
                "children": [
                    {
                        "name": "Grandchild",
                        "dialog_count": 0,
                        "last_dialog_date": None,
                        "last_dialog_doc_id": None,
                        "children": [],
                    }
                ],
            },
            {
                "name": "Child",
                "dialog_count": 0,
                "last_dialog_date": None,
                "last_dialog_doc_id": None,
                "children": [],
            },
            {
                "name": "Second",
                "dialog_count": 0,
                "last_dialog_date": None,
                "last_dialog_doc_id": None,
                "children": [],
            },
        ],
    }
    assert [child["name"] for child in alpha["children"]] == [
        "Child",
        "Child",
        "Second",
    ]
    assert result["folders"][1]["children"][0]["name"] == "Child"
    assert result["suggestion"] == (
        '3 transcript(s) are unassigned. '
        'Call get_transcript with folder_path=["_unassigned"] to review them, '
        "then move each using move_transcript(doc_id=..., folder_path=[...])."
    )


@pytest.mark.offline
def test_display_path_walks_parent_links_and_stops_on_cycles(folders):
    from core.integrations.google_drive.folders import make_folder_path_resolver

    class LookupOnceDict(dict):
        def __init__(self, values):
            super().__init__(values)
            self.lookups = set()

        def _lookup(self, key):
            assert key not in self.lookups, f"folder looked up repeatedly: {key}"
            self.lookups.add(key)

        def get(self, key, default=None):
            self._lookup(key)
            return super().get(key, default)

        def __getitem__(self, key):
            self._lookup(key)
            return super().__getitem__(key)

    cyclic_folders = LookupOnceDict(
        {
            **folders,
            "cycle-a": {"name": "Cycle A", "parent_id": "cycle-b"},
            "cycle-b": {"name": "Cycle B", "parent_id": "cycle-a"},
        }
    )
    resolve_path = make_folder_path_resolver(cyclic_folders)

    assert resolve_path("cycle-a") in {"Cycle A/Cycle B", "Cycle B/Cycle A"}
    assert resolve_path("alpha-grandchild") == "Alpha/Child/Grandchild"
