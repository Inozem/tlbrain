"""Pure helpers for resolving and presenting a completed Drive folder snapshot."""

from collections import deque
from typing import Any, Callable


def resolve_folder_path(
    path: list[str],
    all_folders: dict[str, dict],
    root_folder_id: str,
) -> list[str]:
    """Return every folder ID matching ``path`` below the supplied root."""
    if not path:
        return []

    children: dict[str, list[tuple[str, str]]] = {}
    for folder_id, data in all_folders.items():
        parent_id = data.get("parent_id")
        if parent_id:
            children.setdefault(parent_id, []).append((folder_id, data.get("name", "")))

    current = [root_folder_id]
    for segment in path:
        current = [
            folder_id
            for parent_id in current
            for folder_id, name in children.get(parent_id, [])
            if name == segment
        ]
        if not current:
            return []
    return current


def expand_subtree(folder_id: str, all_folders: dict[str, dict]) -> list[str]:
    """Return the supplied folder ID and its descendants in breadth-first order."""
    children: dict[str, list[str]] = {}
    for child_id, data in all_folders.items():
        parent_id = data.get("parent_id")
        if parent_id:
            children.setdefault(parent_id, []).append(child_id)

    result: list[str] = []
    visited: set[str] = set()
    queue = deque([folder_id])
    while queue:
        current_id = queue.popleft()
        if current_id in visited:
            continue
        visited.add(current_id)
        result.append(current_id)
        queue.extend(children.get(current_id, []))
    return result


def make_folder_path_resolver(all_folders: dict[str, dict]) -> Callable[[str], str]:
    """Build a resolver that follows parent links into a slash-joined display path."""

    def resolve(parent_id: str) -> str:
        path_parts: list[str] = []
        folder_id = parent_id
        visited: set[str] = set()
        while folder_id and folder_id not in visited:
            visited.add(folder_id)
            data = all_folders.get(folder_id, {})
            name = data.get("name", "")
            if name:
                path_parts.append(name)
            folder_id = data.get("parent_id", "")
        return "/".join(reversed(path_parts))

    return resolve


def build_folder_tree(
    all_folders: dict[str, dict],
    root_folder_id: str,
    aggregates: dict[str, dict],
) -> dict[str, Any]:
    """Build the existing folder listing response from a snapshot and aggregates."""
    children: dict[str, list[str]] = {}
    for folder_id, data in all_folders.items():
        parent_id = data.get("parent_id")
        if parent_id:
            children.setdefault(parent_id, []).append(folder_id)

    unassigned_folder_id: str | None = None
    for folder_id, data in all_folders.items():
        if data.get("name") == "_unassigned":
            unassigned_folder_id = folder_id
            break

    def build_node(folder_id: str, ancestors: frozenset[str] = frozenset()) -> dict[str, Any]:
        data = all_folders[folder_id]
        aggregate = aggregates.get(folder_id, {})
        next_ancestors = ancestors | {folder_id}
        child_ids = [
            child_id
            for child_id in children.get(folder_id, [])
            if child_id not in next_ancestors
        ]
        return {
            "name": data["name"],
            "dialog_count": aggregate.get("count", 0),
            "last_dialog_date": aggregate.get("last_date") or None,
            "last_dialog_doc_id": aggregate.get("last_doc_id") or None,
            "children": [
                build_node(child_id, next_ancestors)
                for child_id in sorted(
                    child_ids,
                    key=lambda child_id: all_folders[child_id].get("name", ""),
                )
            ],
        }

    root_ids = [
        folder_id
        for folder_id, data in all_folders.items()
        if data.get("parent_id") == root_folder_id
    ]
    result: dict[str, Any] = {
        "folders": [
            build_node(folder_id)
            for folder_id in sorted(
                root_ids,
                key=lambda folder_id: all_folders[folder_id].get("name", ""),
            )
        ]
    }

    if unassigned_folder_id:
        unassigned_count = aggregates.get(unassigned_folder_id, {}).get("count", 0)
        if unassigned_count > 0:
            result["suggestion"] = (
                f"{unassigned_count} transcript(s) are unassigned. "
                f'Call get_transcript with folder_path=["_unassigned"] to review them, '
                "then move each using move_transcript(doc_id=..., folder_path=[...])."
            )
    return result
