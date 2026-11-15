from __future__ import annotations

import csv
import hashlib
import json
from importlib import resources
from pathlib import Path
import subprocess
import sys

from openimcc import coefficient_ancestry
from openimcc.janaf_crystals import (
    evaluate_janaf_apparent_gibbs,
    evaluate_janaf_crystal,
)
from openimcc.profiles import load_profile
from openimcc.standards import pack_liquid_standards


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_ROLES = {"derived", "fitted", "selected", "anchor", "diagnostic"}


def _json_resource(path: str) -> dict:
    return json.loads(
        resources.files("openimcc")
        .joinpath(*path.split("/"))
        .read_text(encoding="utf-8")
    )


def _artifact(entries: list[dict], identity: str) -> dict:
    return next(entry for entry in entries if entry["identity"] == identity)


def _expected_row_ids(artifact: dict) -> set[str]:
    file = artifact["file"].removeprefix("data/")
    path = resources.files("openimcc").joinpath("data", file)
    if file == "packs/imcc-sf04-d066-v4.json":
        content = json.loads(path.read_text(encoding="utf-8"))
        return {f"{row['row']}:{row['complex']}" for row in content["rows"]}
    if file in {
        "gas/gas-shomate-default-v1.csv",
        "gas/condensate-default-v1.csv",
    }:
        rows = csv.DictReader(path.read_text(encoding="utf-8").splitlines())
        return {
            f"{index}:{row['species_name']}"
            + (f":{row['T_interval']}" if row.get("T_interval") else "")
            for index, row in enumerate(rows, start=1)
        }
    if file == "packs/openimcc-redox-v6.json":
        content = json.loads(path.read_text(encoding="utf-8"))
        reaction_ids = {
            f"reaction:{index}"
            for index, row in enumerate(content["rows"], start=1)
        }
        species_ids = {
            f"thermo_species:{name}" for name in content["thermo_species"]
        }
        return reaction_ids | species_ids
    if file == "liquid-standards.json":
        content = json.loads(path.read_text(encoding="utf-8"))
        row_ids: set[str] = set()
        for digest, pack in content["packs"].items():
            row_ids.update(
                f"{digest}:standard:{standard}"
                for standard in pack.get("standards", {})
            )
            row_ids.update(
                f"{digest}:override:{complex_name}:{standard}"
                for complex_name, overrides in pack.get("row_overrides", {}).items()
                for standard in overrides
            )
            row_ids.update(
                f"{digest}:diagnostic:{complex_name}"
                for complex_name in pack.get("row_diagnostics", {})
            )
        return row_ids
    if file == "diagnostics/janaf-pure-oxide-crystals.json":
        content = json.loads(path.read_text(encoding="utf-8"))
        return {f"phases:{row['id']}" for row in content["phases"]}
    if file == "diagnostics/janaf-pure-oxide-liquid-pairs.json":
        content = json.loads(path.read_text(encoding="utf-8"))
        return {f"rows:{row['table_id']}" for row in content["rows"]}
    raise AssertionError(f"unexpected packaged ancestry artifact: {file}")


def test_default_artifacts_match_the_loaded_package_data() -> None:
    entries = coefficient_ancestry()
    manifest = _json_resource("data/coefficient-ancestry.json")
    profile = load_profile("openimcc")

    assert manifest["schema"] == "openimcc-coefficient-ancestry.v1"
    assert len(entries) == 7
    for entry in entries:
        raw = resources.files("openimcc").joinpath(*entry["file"].split("/")).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry["digest"]
        assert coefficient_ancestry(entry["identity"]) == entry
        assert coefficient_ancestry(entry["digest"]) == entry

    melt = _artifact(entries, f"{profile.melt_pack.model_id}@{profile.melt_pack.version}")
    assert melt["runtime_digest"] == profile.melt_pack.binding_digest

    gas = _artifact(entries, "gas-shomate-default-v1")
    assert gas["runtime_digest"] == profile.gas_datapack.gas_table_digest

    condensate = _artifact(entries, "condensate-default-v1")
    assert condensate["runtime_digest"] == profile.gas_datapack.condensate_table_digest

    redox = _artifact(entries, profile.redox_pack.version)
    assert redox["runtime_digest"] == profile.redox_pack.canonical_digest

    registry = _artifact(entries, "liquid-standards-v2")
    registry_data = _json_resource("data/liquid-standards.json")
    assert registry["runtime_digest"] == registry["digest"]
    loaded_standards = pack_liquid_standards(profile.melt_pack)
    declaration = registry_data["packs"][profile.melt_pack.binding_digest]
    assert dict(loaded_standards.standards) == declaration["standards"]

    crystals = _artifact(entries, "openimcc-janaf-apparent-crystals-v2")
    crystal_data = _json_resource(crystals["file"])
    phase = crystal_data["phases"][0]
    temperature = max(298.15, phase["source_temperature_K"][0])
    result = evaluate_janaf_crystal(phase["id"], temperature)
    assert result.provenance["resource_sha256"] == crystals["runtime_digest"]

    liquid_pairs = _artifact(entries, "openimcc-janaf-apparent-liquid-pairs-v1")
    pair_data = _json_resource(liquid_pairs["file"])
    pair = pair_data["rows"][0]
    temperature = max(298.15, pair["source_temperature_K"][0])
    result = evaluate_janaf_apparent_gibbs(pair["table_id"], temperature)
    assert result.provenance["resource_sha256"] == liquid_pairs["runtime_digest"]


def test_every_default_row_has_mapped_or_unmapped_lineage() -> None:
    entries = coefficient_ancestry()
    for entry in entries:
        expected = _expected_row_ids(entry)
        actual_rows = entry["rows"]
        actual = {row["id"] for row in actual_rows}
        assert actual == expected
        assert len(actual_rows) == len(actual)
        unmapped_row_ids = {item["row"] for item in entry["unmapped"]}
        for row in actual_rows:
            assert row["sources"] or row["id"] in unmapped_row_ids
            assert all(source["role"] in ALLOWED_ROLES for source in row["sources"])
        assert all(item["row"] in actual for item in entry["unmapped"])
        assert entry["counts"]["rows"] == len(expected)
        assert entry["complete"] is (not entry["unmapped"])


def test_manifest_work_ids_match_the_recorded_vocabulary() -> None:
    manifest = _json_resource("data/coefficient-ancestry.json")
    snapshot = json.loads(
        (ROOT / "tests" / "data" / "coefficient-ancestry-vocabulary.json").read_text(
            encoding="utf-8"
        )
    )
    metadata = manifest["vocabulary_snapshot"]
    encoded_ids = json.dumps(
        snapshot["work_ids"], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    assert metadata["commit"] == snapshot["commit"]
    assert metadata["directories"] == snapshot["directories"]
    assert metadata["work_id_count"] == len(snapshot["work_ids"])
    assert metadata["sha256"] == hashlib.sha256(encoded_ids).hexdigest()

    vocabulary = set(snapshot["work_ids"])
    for entry in manifest["artifacts"]:
        used_ids = {
            source["work_id"]
            for row in entry["rows"]
            for source in row["sources"]
        }
        assert used_ids <= vocabulary
        assert entry["work_ids"] == sorted(used_ids)


def test_manifest_regeneration_is_byte_identical() -> None:
    result = subprocess.run(
        [sys.executable, "tools/build_coefficient_ancestry.py", "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "byte-identical" in result.stdout
