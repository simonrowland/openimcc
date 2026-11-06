"""Small JSON parsing helpers shared by the package file loaders."""

from __future__ import annotations

from typing import Any


class DuplicateJsonKeyError(ValueError):
    """Raised when a JSON object repeats a key."""


def reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a JSON object while refusing ambiguous duplicate keys."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKeyError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result
