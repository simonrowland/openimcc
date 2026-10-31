"""Keep row-level provenance for legacy melt packs outside frozen pack bytes."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PACK_DIR = ROOT / "src" / "openimcc" / "data" / "packs"
SIDECAR_PATH = ROOT / "src" / "openimcc" / "data" / "source-refs-sidecar.json"
MANIFEST_SHA256 = "505237efb0502c0696921e996e9056b224ba3cdc8003b17e95816f52771e1cfa"


def _pack_data() -> dict[str, dict[str, Any]]:
    return {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(PACK_DIR.glob("imcc-sf04-*.json"))
    }


def _legacy_packs(packs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        name: pack
        for name, pack in packs.items()
        if pack.get("rows")
        and any("source_refs" not in row for row in pack["rows"])
    }


def _identity(row: dict[str, Any]) -> tuple[int, str]:
    return int(row["row"]), str(row["complex"])


def _row_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("row"),
        row.get("complex"),
        row.get("reaction"),
        json.dumps(row.get("nu"), sort_keys=True),
        row.get("A"),
        row.get("B"),
    )


def test_pack_manifest_is_unchanged() -> None:
    manifest_bytes = (PACK_DIR / "MANIFEST.json").read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == MANIFEST_SHA256


def test_legacy_melt_rows_have_one_valid_sidecar_entry() -> None:
    packs = _pack_data()
    legacy = _legacy_packs(packs)
    manifest = json.loads((PACK_DIR / "MANIFEST.json").read_text(encoding="utf-8"))
    manifest_files = {entry["file"] for entry in manifest["packs"]}
    sidecar = json.loads(SIDECAR_PATH.read_text(encoding="utf-8"))

    assert sidecar["schema"] == "melt_source_refs_sidecar.v1"
    assert set(sidecar["packs"]) == set(legacy)
    assert set(legacy) <= manifest_files

    source_vocabulary = {
        source_id
        for pack in packs.values()
        if pack.get("rows")
        and pack["rows"]
        and all("source_refs" in row for row in pack["rows"])
        for source_id in pack.get("sources", {})
    }
    assert set(sidecar["sources"]) <= source_vocabulary

    unknown_entries = []
    for name, pack in legacy.items():
        entries = sidecar["packs"][name]
        actual_identities = Counter(
            (int(entry["row"]), str(entry["complex"])) for entry in entries
        )
        expected_identities = Counter(_identity(row) for row in pack["rows"])
        assert actual_identities == expected_identities, name

        for entry in entries:
            if entry["source_refs"] is None:
                assert isinstance(entry.get("unknown_reason"), str)
                assert entry["unknown_reason"].strip()
                unknown_entries.append((name, _identity(entry)))
                continue

            assert isinstance(entry["source_refs"], list) and entry["source_refs"]
            assert len(entry["source_refs"]) == len(set(entry["source_refs"]))
            assert set(entry["source_refs"]) <= set(sidecar["sources"])
            assert set(entry["source_refs"]) <= source_vocabulary

    assert unknown_entries == []


def test_unchanged_rows_match_later_pack_source_refs() -> None:
    packs = _pack_data()
    legacy = _legacy_packs(packs)
    later_packs = [
        pack
        for pack in packs.values()
        if pack.get("rows")
        and pack["rows"]
        and all("source_refs" in row for row in pack["rows"])
    ]
    sidecar = json.loads(SIDECAR_PATH.read_text(encoding="utf-8"))

    comparisons = 0
    for name, pack in legacy.items():
        entries = {
            _identity(entry): entry for entry in sidecar["packs"][name]
        }
        for row in pack["rows"]:
            matching_rows = [
                later_row
                for later_pack in later_packs
                for later_row in later_pack["rows"]
                if _row_signature(later_row) == _row_signature(row)
            ]
            for later_row in matching_rows:
                assert entries[_identity(row)]["source_refs"] == later_row[
                    "source_refs"
                ], (name, _identity(row))
                comparisons += 1

    assert comparisons > 0
