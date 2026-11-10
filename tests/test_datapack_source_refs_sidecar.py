"""Keep row-level provenance for legacy melt packs outside frozen pack bytes."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from test_melt_source_transcriptions import (
    _assert_pack_reaction_matches_source,
    _assert_transform_reason,
    _matches_source_coefficients,
    _recorded_transforms,
    _source_row_for_pack_row,
    _source_rows,
)


ROOT = Path(__file__).resolve().parents[1]
PACK_DIR = ROOT / "src" / "openimcc" / "data" / "packs"
SIDECAR_PATH = ROOT / "src" / "openimcc" / "data" / "source-refs-sidecar.json"
# Re-pinned for the opt-in Ga/In KEMS condensate table.
MANIFEST_SHA256 = "363113ac09a28f5904209c37c774045064ade04896dd53798b36de73c5f44fa8"


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


def _later_rows_with_same_signature(
    row: dict[str, Any], later_packs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return [
        later_row
        for later_pack in later_packs
        for later_row in later_pack["rows"]
        if _row_signature(later_row) == _row_signature(row)
    ]


def test_pack_manifest_digest_is_pinned() -> None:
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


def test_sidecar_source_assignments_have_transcription_or_metadata_support() -> None:
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
    fc87_rows, sf04_rows = _source_rows()

    expected = {
        (name, _identity(row))
        for name, pack in legacy.items()
        for row in pack["rows"]
    }
    transcription_matches = set()
    conversion_metadata = set()
    later_matches = set()

    for name, pack in legacy.items():
        entries = {
            _identity(entry): entry for entry in sidecar["packs"][name]
        }
        for row in pack["rows"]:
            key = (name, _identity(row))
            entry = entries[_identity(row)]
            source_row = _source_row_for_pack_row(row["row"], fc87_rows, sf04_rows)
            source_id = "SF04" if isinstance(source_row["row"], int) else "FC87"

            _assert_pack_reaction_matches_source(row, source_row)
            if _matches_source_coefficients(row, source_row):
                assert source_id in entry["source_refs"], (name, _identity(row))
                transcription_matches.add(key)
            else:
                transforms = _recorded_transforms(row)
                if transforms:
                    assert source_id in entry["source_refs"], (name, _identity(row))
                    assert source_id in row.get("source", ""), (name, _identity(row))
                    for transform in transforms:
                        _assert_transform_reason(row, transform)
                        if transform == "k_translation":
                            metadata = row["k_translation"]
                            assert source_id == "FC87"
                            assert "FC87 Table 1" in metadata["basis"]
                            assert "JANAF K-012 K2O(l)" in metadata["new_reference"]
                    conversion_metadata.add(key)

            if _later_rows_with_same_signature(row, later_packs):
                later_matches.add(key)

    covered = transcription_matches | conversion_metadata | later_matches
    assert covered == expected, sorted(expected - covered)


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
            matching_rows = _later_rows_with_same_signature(row, later_packs)
            for later_row in matching_rows:
                assert entries[_identity(row)]["source_refs"] == later_row[
                    "source_refs"
                ], (name, _identity(row))
                comparisons += 1

    assert comparisons > 0
