#!/usr/bin/env python3
"""Build and verify the datapack manifest.

WHY THIS EXISTS. The simulator that spawned this package will consume it as a
pinned dependency. A pinned version number protects the CODE, but the numbers
this engine produces are set just as much by the DATAPACK -- the reaction rows
and their Gibbs coefficients. A pack edited in place under an unchanged version
string would silently move every activity the engine reports, and every
downstream residual with it.

So the manifest is the drift detector: a content hash per pack, checked in. A
consumer pins (package version, manifest hash). If either moves, the consumer
finds out at CI time rather than by noticing its residuals drifted three months
later.

    python tools/packmanifest.py --write     # regenerate after an intended edit
    python tools/packmanifest.py --check     # CI gate; nonzero on drift
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

PACK_DIR = Path(__file__).resolve().parent.parent / "src" / "openimcc" / "data" / "packs"
MANIFEST = PACK_DIR / "MANIFEST.json"
ROLE_METADATA_FIELDS = ("family", "role", "superseded_by")
PACK_ROLES = {"default", "opt-in", "historical", "reproduction"}


def _existing_role_metadata() -> dict[str, dict]:
    """Keep lifecycle metadata when --write refreshes pack digests."""
    if not MANIFEST.exists():
        return {}
    try:
        recorded = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(recorded, dict) or not isinstance(
        recorded.get("packs"), list
    ):
        return {}
    metadata = {}
    for entry in recorded["packs"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
            continue
        fields = {
            field: entry[field]
            for field in ROLE_METADATA_FIELDS
            if field in entry
        }
        if fields:
            metadata[entry["file"]] = fields
    return metadata


def _digest(path: Path) -> str:
    """SHA-256 of the raw bytes.

    Deliberately hashes bytes rather than parsed-then-recanonicalised JSON: a
    reformat that does not change a single number is still an edit we want to
    see, and canonicalising first would hide exactly the class of change (key
    reordering, float repr) that indicates someone regenerated a pack with a
    different tool.
    """
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _summarise(path: Path) -> dict:
    """One manifest entry.

    Reads the pack's own ``rows`` list -- the active reaction rows -- and NOT
    ``screens`` or ``gas_only_inventory``, which hold candidates that were
    considered and deliberately not activated. Counting those as content would
    make two packs with identical active chemistry look different.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("rows") or []
    provenance = Counter(
        str(r.get("provenance_class") or "unspecified") for r in rows
    )
    return {
        "file": path.name,
        "sha256": _digest(path),
        "bytes": path.stat().st_size,
        "model": raw.get("model"),
        "datapack_version": raw.get("imcc_sf04_datapack_version", raw.get("version")),
        "n_rows": len(rows),
        "n_screened_not_activated": len(raw.get("screens") or []),
        "provenance_classes": dict(sorted(provenance.items())),
        "provenance_split": raw.get("provenance_split"),
    }


def build() -> dict:
    packs = sorted(p for p in PACK_DIR.glob("*.json") if p.name != MANIFEST.name)
    role_metadata = _existing_role_metadata()
    entries = []
    for path in packs:
        entry = _summarise(path)
        entry.update(role_metadata.get(path.name, {}))
        entries.append(entry)
    return {
        "schema": "openimcc-pack-manifest.v1",
        "packs": entries,
    }


def _component(filename: str) -> str | None:
    if filename.startswith("openimcc-redox-"):
        return "redox"
    if filename.endswith(".json"):
        return "melt"
    return None


def _validate_roles(
    manifest: dict, runtime_default_versions: dict[str, str]
) -> list[str]:
    """Validate lifecycle metadata without changing runtime pack selection."""
    entries = manifest.get("packs")
    if not isinstance(entries, list):
        return ["manifest packs must be a list"]

    issues: list[str] = []
    by_name = {
        entry["file"]: entry
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("file"), str)
    }
    defaults: dict[str, list[dict]] = {"melt": [], "redox": []}
    families: dict[str, list[dict]] = {}
    superseded_edges: dict[str, str] = {}

    for entry in entries:
        if not isinstance(entry, dict):
            issues.append("each pack entry must be an object")
            continue
        filename = entry.get("file")
        role = entry.get("role")
        if not isinstance(role, str) or role not in PACK_ROLES:
            issues.append(f"{filename!r} has invalid or missing role {role!r}")
            continue

        component = _component(filename) if isinstance(filename, str) else None
        if role == "default":
            if component not in defaults:
                issues.append(f"default pack {filename!r} has an unknown component")
            else:
                defaults[component].append(entry)
        elif role == "historical":
            superseded_by = entry.get("superseded_by")
            if not isinstance(superseded_by, str) or not superseded_by:
                issues.append(f"historical pack {filename!r} must name superseded_by")
            elif superseded_by not in by_name:
                issues.append(
                    f"historical pack {filename!r} superseded_by "
                    f"{superseded_by!r}, which is not in the manifest"
                )
            elif filename == superseded_by:
                issues.append(
                    f"historical pack {filename!r} cannot supersede itself"
                )
            else:
                family = entry.get("family")
                successor = by_name[superseded_by]
                if (
                    isinstance(family, str)
                    and successor.get("family") != family
                ):
                    issues.append(
                        f"historical pack {filename!r} superseded_by "
                        f"must stay within candidate family {family!r}"
                    )
                if isinstance(filename, str):
                    superseded_edges[filename] = superseded_by

        family = entry.get("family")
        if role == "opt-in" and (not isinstance(family, str) or not family.strip()):
            issues.append(f"opt-in pack {filename!r} must name a candidate family")
        elif family is not None:
            if role not in {"opt-in", "historical"}:
                issues.append(f"{filename!r} has family metadata for role {role!r}")
            if not isinstance(family, str) or not family.strip():
                issues.append(f"{filename!r} has an invalid candidate family")
            else:
                families.setdefault(family, []).append(entry)

    for component, component_defaults in defaults.items():
        if len(component_defaults) != 1:
            issues.append(
                f"manifest must have exactly one {component} default; "
                f"found {len(component_defaults)}"
            )
            continue
        default = component_defaults[0]
        expected_version = runtime_default_versions.get(component)
        if default.get("datapack_version") != expected_version:
            issues.append(
                f"{component} default {default.get('file')!r} does not match "
                f"the runtime default version {expected_version!r}"
            )

    for family, members in families.items():
        current = [entry for entry in members if entry.get("role") == "opt-in"]
        if len(current) != 1:
            issues.append(
                f"candidate family {family!r} must have exactly one current opt-in; "
                f"found {len(current)}"
            )

    finished: set[str] = set()
    for start in superseded_edges:
        chain: list[str] = []
        positions: dict[str, int] = {}
        current = start
        while current in superseded_edges and current not in finished:
            if current in positions:
                cycle = chain[positions[current]:] + [current]
                issues.append(
                    "historical superseded_by cycle: " + " -> ".join(cycle)
                )
                break
            positions[current] = len(chain)
            chain.append(current)
            successor = superseded_edges[current]
            if (
                successor not in by_name
                or by_name[successor].get("role") != "historical"
            ):
                break
            current = successor
        finished.update(chain)
    return issues


def _runtime_default_versions() -> dict[str, str]:
    """Read the defaults through the same loaders used by package callers."""
    source = str(Path(__file__).resolve().parent.parent / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from openimcc.model import load_datapack
    from openimcc.redox_pack import load_redox_pack

    return {
        "melt": load_datapack().version,
        "redox": load_redox_pack().version,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="regenerate MANIFEST.json")
    group.add_argument("--check", action="store_true", help="fail if any pack drifted")
    args = ap.parse_args(argv)

    current = build()

    if args.write:
        MANIFEST.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {MANIFEST} ({len(current['packs'])} packs)")
        return 0

    if not MANIFEST.exists():
        print(f"error: {MANIFEST} is missing; run --write", file=sys.stderr)
        return 2

    recorded = json.loads(MANIFEST.read_text(encoding="utf-8"))
    recorded_packs = recorded.get("packs", [])
    filenames = [pack["file"] for pack in recorded_packs]
    duplicates = sorted(
        name for name, count in Counter(filenames).items() if count > 1
    )
    if duplicates:
        print(
            "error: duplicate filename(s) in manifest: " + ", ".join(duplicates),
            file=sys.stderr,
        )
        return 1
    if len(recorded_packs) != len(current["packs"]):
        print(
            f"error: manifest has {len(recorded_packs)} pack record(s); "
            f"found {len(current['packs'])} pack(s) on disk",
            file=sys.stderr,
        )
        return 1

    by_name = {p["file"]: p for p in recorded_packs}
    drift: list[str] = []

    for pack in current["packs"]:
        was = by_name.pop(pack["file"], None)
        if was is None:
            drift.append(f"NEW      {pack['file']} (not in manifest)")
        elif was["sha256"] != pack["sha256"]:
            drift.append(
                f"CHANGED  {pack['file']}\n"
                f"           manifest {was['sha256'][:16]}  n_rows={was['n_rows']}\n"
                f"           on disk  {pack['sha256'][:16]}  n_rows={pack['n_rows']}"
            )
    for missing in by_name:
        drift.append(f"REMOVED  {missing} (in manifest, not on disk)")

    if drift:
        print("datapack drift detected:\n  " + "\n  ".join(drift), file=sys.stderr)
        print("\nIf the edit was intended, re-run with --write and say why in the "
              "commit message; consumers pin this hash.", file=sys.stderr)
        return 1

    role_issues = _validate_roles(recorded, _runtime_default_versions())
    if role_issues:
        print("invalid pack lifecycle metadata:\n  " + "\n  ".join(role_issues),
              file=sys.stderr)
        return 1

    if not current["packs"]:
        # Zero packs on disk and in the manifest is not a match; it means the
        # check looked in the wrong place or the data are missing.
        print("datapack check found no packs; refusing to report a vacuous match",
              file=sys.stderr)
        return 1
    print(f"ok: {len(current['packs'])} pack(s) match the manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
