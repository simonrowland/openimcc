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
import tomllib
from pathlib import Path
from typing import Mapping

from openimcc import load_datapack


ROOT = Path(__file__).resolve().parent.parent
PACK_DIR = ROOT / "src" / "openimcc" / "data" / "packs"
DOC = ROOT / "docs" / "datapack-format.md"
PYPROJECT = ROOT / "pyproject.toml"
SHIPPED_TEXT_SUFFIXES = frozenset(
    {
        ".cff",
        ".csv",
        ".json",
        ".md",
        ".py",
        ".sh",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
)
RETIREMENT_NOTE = ROOT / "docs" / "datapack-format.md"
PRIVATE_PATH_MARKERS = ("/" + "Users/", "Cloud" + "Storage/", "drop" + "box/")
RETIRED_PACK_NAMES = tuple(
    "imcc-sf04-" + suffix for suffix in ("ext-v1", "ext-v2", "ext-v3")
)
# Both packs carry the published SF04 core, whose hash the loader verifies, and
# that core contains the documented DOI erratum; NOTICE explains why it cannot
# be rewritten.
FROZEN_DOI_PACKS = frozenset(
    {
        "src/openimcc/data/packs/imcc-sf04-v1.0.2.json",
        "src/openimcc/data/packs/imcc-sf04-ext-v4.json",
    }
)
WRONG_SF04_DOI = "10.1016/j.icarus." + "2003.11.023"
ERRATUM_START = "The packs `imcc-sf04-v1.0.2.json`"
ERRATUM_END = "make every pin of the current identity unloadable."
RETIREMENT_START = "The ext-v1, ext-v2, and ext-v3 records are retired"
RETIREMENT_END = "research records; the shipped loadable packs are v1.0.2 and ext-v4."

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


def _files_under(path: Path) -> set[Path]:
    if path.is_file():
        return {path}
    if path.is_dir():
        return {candidate for candidate in path.rglob("*") if candidate.is_file()}
    return set()


def _shipped_text_files() -> tuple[Path, ...]:
    """Derive the text files from build metadata, not a directory list."""
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    paths: set[Path] = {PYPROJECT}

    readme = project.get("project", {}).get("readme")
    if isinstance(readme, str):
        paths.add(ROOT / readme)
    elif isinstance(readme, dict) and isinstance(readme.get("file"), str):
        paths.add(ROOT / readme["file"])

    license_files = project.get("project", {}).get("license-files", [])
    for entry in license_files:
        if isinstance(entry, str):
            paths.add(ROOT / entry)

    wheel = (
        project.get("tool", {})
        .get("hatch", {})
        .get("build", {})
        .get("targets", {})
        .get("wheel", {})
    )
    for package in wheel.get("packages", []):
        if isinstance(package, str):
            paths.update(_files_under(ROOT / package))

    sdist = (
        project.get("tool", {})
        .get("hatch", {})
        .get("build", {})
        .get("targets", {})
        .get("sdist", {})
    )
    for include in sdist.get("include", []):
        if not isinstance(include, str):
            continue
        matches = list(ROOT.glob(include.rstrip("/")))
        if not matches:
            matches = [ROOT / include.rstrip("/")]
        for match in matches:
            paths.update(_files_under(match))

    return tuple(
        sorted(
            path
            for path in paths
            if path.is_file()
            and (
                path.suffix in SHIPPED_TEXT_SUFFIXES
                or path.name in license_files
                or path.name == "py.typed"
            )
        )
    )


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


def test_shipped_pack_inventory_excludes_retired_and_private_records() -> None:
    names = {
        path.name
        for path in PACK_DIR.glob("*.json")
        if path.name != "MANIFEST.json"
    }
    assert names == {
        "imcc-sf04-ext-v4.json",
        "imcc-sf04-v1.0.2.json",
    }

    ext4 = json.loads(
        (PACK_DIR / "imcc-sf04-ext-v4.json").read_text(encoding="utf-8")
    )
    assert ext4["imcc_sf04_datapack_version"] == "1.0.2-ext-sp-2"
    assert ext4["sp_extension"]["certification"] == "denied"
    assert all(
        "local_mirror_path" not in row.get("provenance", {})
        for row in ext4["sp_extension"]["rows"]
    )


def _span(text: str, start_marker: str, end_marker: str) -> tuple[int, int] | None:
    start = text.find(start_marker)
    if start < 0:
        return None
    end = text.find(end_marker, start)
    if end < 0:
        return None
    return start, end + len(end_marker)


def _inside(span: tuple[int, int] | None, offset: int) -> bool:
    return span is not None and span[0] <= offset < span[1]


def _hygiene_violations(files: Mapping[Path, str]) -> list[str]:
    violations: list[str] = []
    for path, text in files.items():
        relative = path.relative_to(ROOT).as_posix()
        for marker in PRIVATE_PATH_MARKERS:
            if marker in text:
                violations.append(f"{relative}: {marker}")
        retirement_span = _span(text, RETIREMENT_START, RETIREMENT_END)
        for pack_name in RETIRED_PACK_NAMES:
            for match in re.finditer(re.escape(pack_name), text):
                if path != RETIREMENT_NOTE or not _inside(
                    retirement_span, match.start()
                ):
                    violations.append(f"{relative}: {pack_name}")
        erratum_span = _span(text, ERRATUM_START, ERRATUM_END)
        for match in re.finditer(re.escape(WRONG_SF04_DOI), text):
            frozen_pack = relative in FROZEN_DOI_PACKS
            erratum_file = path.name in {"README.md", "NOTICE"}
            if not frozen_pack and not (
                erratum_file and _inside(erratum_span, match.start())
            ):
                violations.append(f"{relative}: wrong DOI outside erratum")
    return violations


def _shipped_text() -> dict[Path, str]:
    return {
        path: path.read_text(encoding="utf-8") for path in _shipped_text_files()
    }


def test_shipped_tree_excludes_private_paths_and_retired_pack_references() -> None:
    violations = _hygiene_violations(_shipped_text())
    assert not violations, "shipped-tree hygiene violations: " + "; ".join(violations)


def test_frozen_pack_doi_erratum_is_explicit() -> None:
    files = _shipped_text()
    occurrences = {
        path.relative_to(ROOT).as_posix()
        for path, text in files.items()
        if WRONG_SF04_DOI in text
    }
    assert occurrences == FROZEN_DOI_PACKS | {"README.md", "NOTICE"}
    assert not _hygiene_violations(files)


def test_notice_erratum_pins_frozen_names_and_freeze_explanation() -> None:
    notice = " ".join((ROOT / "NOTICE").read_text(encoding="utf-8").split())
    assert (
        "The packs `imcc-sf04-v1.0.2.json` and `imcc-sf04-ext-v4.json` "
        f"contain `{WRONG_SF04_DOI}` in their `sources.SF04.citation`."
    ) in notice
    assert (
        "The citation sits inside the published SF04 core, which both packs "
        "carry byte-identically and the loader verifies by hash, so the core "
        "is deliberately left unchanged: correcting it would change the "
        "published digest and make every pin of the current identity "
        "unloadable."
    ) in notice


def test_hygiene_controls_go_red_then_restore() -> None:
    original = _shipped_text()
    controls = (
        ("NOTICE private path", ROOT / "NOTICE", "/" + "Users/example/private"),
        ("CITATION private path", ROOT / "CITATION.cff", "/" + "Users/example/private"),
        ("README wrong DOI", ROOT / "README.md", WRONG_SF04_DOI),
        ("retired pack outside retirement note", DOC, RETIRED_PACK_NAMES[0]),
    )
    for label, path, addition in controls:
        mutated = dict(original)
        mutated[path] += "\n" + addition
        assert _hygiene_violations(mutated), label
        assert not _hygiene_violations(original), f"not restored after {label}"


def test_every_shipped_pack_loads() -> None:
    # packmanifest --check compares whole-file sha256 only.  The loader also
    # verifies the hash-pinned SF04 core that every shipped pack embeds, so a
    # pack can match the manifest and still be refused.  Load each one.
    manifest = json.loads((PACK_DIR / "MANIFEST.json").read_text(encoding="utf-8"))
    names = sorted(Path(entry["file"]).name for entry in manifest["packs"])
    assert names == ["imcc-sf04-ext-v4.json", "imcc-sf04-v1.0.2.json"]
    for name in names:
        load_datapack(PACK_DIR / name)
