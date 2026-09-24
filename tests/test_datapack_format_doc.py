"""Keep the datapack specification synchronized with shipped JSON fields.

This check does not prove that a documented meaning is correct, that a key in
an unrelated table's first cell describes the field, or that prose-only
mentions document a key. It also has no reverse check: the specification's
value tables (for example, the provenance classes) would create false
positives for names that are documented but are not JSON keys.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PACK_DIR = ROOT / "src" / "openimcc" / "data" / "packs"
DOC = ROOT / "docs" / "datapack-format.md"

# The specification uses the first cell of its top-level and row-field tables
# for field names. Restricting this parser to identifier-shaped cells avoids
# counting table headings and prose.
IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
PLACEHOLDER_DESCRIPTIONS = {
    "",
    "-",
    "—",
    "–",
    "?",
    "na",
    "n/a",
    "n.a.",
    "none",
    "null",
    "not applicable",
    "pending",
    "tbd",
    "todo",
    "unknown",
}


def _shipped_packs() -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(PACK_DIR.glob("*.json"))
        if path.name != "MANIFEST.json"
    ]


def _documented_rows(document: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in document.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2 or not IDENTIFIER.fullmatch(cells[0]):
            continue
        rows.append((cells[0], cells[-1]))
    return rows


def _add_object_keys(keys: set[str], value: object) -> None:
    if isinstance(value, dict):
        keys.update(value)


def _is_placeholder(description: str) -> bool:
    normalized = " ".join(description.split()).casefold()
    return normalized in PLACEHOLDER_DESCRIPTIONS or (
        bool(normalized) and set(normalized) <= {"-", "—", "–", "?"}
    )


def _nested_keys(pack: dict) -> set[str]:
    """Return keys in the shipped nested schema records.

    The selected object records are schema-bearing containers. Mapping keys
    inside their values (for example parent names in ``nu``) are data labels,
    not additional pack fields. Extension rows are included because they are
    rows nested under ``sp_extension`` and carry the provenance objects that
    the loader validates.
    """

    nested: set[str] = set()
    for field in ("sp_extension", "merge_semantics", "provenance_split"):
        _add_object_keys(nested, pack.get(field))
    for field in ("screens", "gas_only_inventory"):
        for entry in pack.get(field, []):
            _add_object_keys(nested, entry)

    for row in pack.get("rows", []):
        if isinstance(row, dict):
            _add_object_keys(nested, row.get("provenance"))

    sp_extension = pack.get("sp_extension")
    if isinstance(sp_extension, dict):
        _add_object_keys(nested, sp_extension.get("provenance"))
        for row in sp_extension.get("rows", []):
            if isinstance(row, dict):
                nested.update(row)
                _add_object_keys(nested, row.get("provenance"))
    return nested


def _missing_keys(document: str) -> dict[str, list[str]]:
    documented_rows = _documented_rows(document)
    documented = {key for key, _ in documented_rows}
    packs = _shipped_packs()
    top_level = set().union(*(pack.keys() for pack in packs))
    row_level = set().union(
        *((row.keys() for pack in packs for row in pack.get("rows", []))),
    )
    nested = set().union(*(_nested_keys(pack) for pack in packs))
    invalid_descriptions = sorted(
        key
        for key, description in documented_rows
        if _is_placeholder(description)
    )
    return {
        "top-level": sorted(top_level - documented),
        "row": sorted(row_level - documented),
        "nested": sorted(nested - documented),
        "empty-or-placeholder descriptions": invalid_descriptions,
    }


def test_every_shipped_datapack_key_is_documented() -> None:
    missing = _missing_keys(DOC.read_text(encoding="utf-8"))
    assert not any(missing.values()), f"undocumented datapack keys: {missing}"
