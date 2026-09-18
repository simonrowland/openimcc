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
        "datapack_version": raw.get("imcc_sf04_datapack_version"),
        "n_rows": len(rows),
        "n_screened_not_activated": len(raw.get("screens") or []),
        "provenance_classes": dict(sorted(provenance.items())),
        "provenance_split": raw.get("provenance_split"),
    }


def build() -> dict:
    packs = sorted(p for p in PACK_DIR.glob("*.json") if p.name != MANIFEST.name)
    return {
        "schema": "openimcc-pack-manifest.v1",
        "packs": [_summarise(p) for p in packs],
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
    by_name = {p["file"]: p for p in recorded.get("packs", [])}
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

    print(f"ok: {len(current['packs'])} pack(s) match the manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
