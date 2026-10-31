"""Coverage for the split between fit residuals and accuracy bands."""

from __future__ import annotations

import json
import subprocess
import tarfile
from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml

from openimcc.redox_pack import canonical_redox_digest
from tools.migrate_band_schema import migrate

ROOT = Path(__file__).resolve().parents[1]
FIELDS = {"fit_residual", "accuracy_band", "band_basis", "independent_check"}
BASIS = {
    "source_uncertainty", "measured_construction", "independent_measurement",
    "fit_only", "none",
}
CHECK = {
    "status", "equilibrium", "primary_source", "conditions", "measured_value",
    "predicted_value", "miss", "miss_to_band_ratio", "reason",
}


def _validate_row(row: Mapping, label: str) -> None:
    assert FIELDS <= row.keys(), label
    assert not ({
        "band_kj_mol", "band_method", "uncertainty_band",
        "max_residual_J_per_mol", "max_residual_J_per_mol_by_interval",
        "max_residual_log10_K", "additional_uncertainty_bands",
    } & row.keys()), label
    basis = row["band_basis"]
    assert basis["kind"] in BASIS, label
    assert basis["explanation"], label
    check = row["independent_check"]
    assert CHECK <= check.keys(), label
    assert check["status"] in {"tested", "lead", "not_found", "not_applicable"}, label
    for field in ("fit_residual", "accuracy_band"):
        value = row[field]
        if value is not None:
            assert isinstance(value, dict), label
            assert "value" in value and value.get("unit"), label
            fields = (
                {"value", "unit", "target_source", "comparison_method", "temperature_range", "held_out_or_in_sample"}
                if field == "fit_residual"
                else {"value", "unit", "temperature_points_or_range", "coverage"}
            )
            assert fields <= value.keys(), label
            if field == "fit_residual":
                assert value["temperature_range"] not in (
                    "T_domain_K", "T_range_K or T_interval",
                ), label
            if field == "fit_residual" and isinstance(value.get("value"), Mapping):
                units = value["unit"]
                for field_name in value["value"]:
                    if field_name.endswith("_J_per_mol") or field_name.endswith("_J_per_mol_by_interval"):
                        declared = units.get(field_name) if isinstance(units, Mapping) else units
                        assert declared == "J mol^-1", f"{label}/{field_name} unit mismatch"
    if row["accuracy_band"] is not None and row["fit_residual"] is not None:
        assert row["accuracy_band"]["value"] != row["fit_residual"]["value"] or basis["kind"] == "independent_measurement", label


def test_every_published_row_uses_the_band_schema() -> None:
    redox_paths = [
        ROOT / "src/openimcc/data/packs/openimcc-redox-v3.json",
    ]
    redox_paths.extend(sorted((ROOT / "src/openimcc/data/packs").glob("openimcc-redox-*-candidate*.json")))
    for path in redox_paths:
        redox = json.loads(path.read_text())
        for key, row in redox["thermo_species"].items():
            _validate_row(row, f"{path.name}/thermo_species/{key}")
        for index, row in enumerate(redox["rows"]):
            _validate_row(row, f"{path.name}/rows/{index}")

    source = json.loads((ROOT / "data-src/redox-thermochemistry.json").read_text())
    for key, row in source["tables"].items():
        _validate_row(row, f"redox-source/tables/{key}")
    _validate_row(source["hematite_crystal_table"], "redox-source/hematite_crystal_table")

    gas = yaml.safe_load((ROOT / "src/openimcc/data/gas/PROVENANCE.yaml").read_text())
    for index, row in enumerate(gas["rows"]):
        _validate_row(row, f"gas/rows/{index}")

    liquid = json.loads((ROOT / "data-src/liquid-from-solid-inputs.json").read_text())
    for key, row in liquid["rows"].items():
        _validate_row(row, f"liquid-inputs/{key}")

    for path in sorted((ROOT / "src/openimcc/data/packs").glob("imcc-sf04*.json")):
        pack = json.loads(path.read_text())
        for index, row in enumerate(pack["rows"]):
            _validate_row(row, f"{path.name}/rows/{index}")
            assert row["band_basis"]["kind"] == "none"
            assert row["accuracy_band"] is None
        for index, row in enumerate(pack.get("sp_extension", {}).get("rows", [])):
            _validate_row(row, f"{path.name}/sp_extension/rows/{index}")
            assert row["band_basis"]["kind"] == "none"
            assert row["accuracy_band"] is None


def test_k2o_liquid_band_units_and_interval_fit_residual_are_explicit() -> None:
    gas = yaml.safe_load((ROOT / "src/openimcc/data/gas/PROVENANCE.yaml").read_text())
    row = next(
        row for row in gas["rows"]
        if row.get("table") == "condensate" and row.get("species_name") == "K2O(l)"
    )
    residual = row["fit_residual"]
    assert residual["value"]["max_residual_J_per_mol_by_interval"] == [
        9.56932283600327, 3.19386472436599, 1.54911242239177,
    ]
    assert residual["unit"]["max_residual_J_per_mol_by_interval"] == "J mol^-1"
    assert residual["temperature_range"] == [
        [1000, 1200], [1200, 2000], [2000, 3000],
    ]
    assert row["accuracy_band"]["unit"] == "kJ mol^-1"
    assert row["band_basis"]["kind"] == "measured_construction"
    assert "Tier-2" in row["band_basis"]["explanation"]


def test_sp_extension_held_out_diagnostics_are_migrated() -> None:
    for filename in ("imcc-sf04-ext-v4.json", "imcc-sf04-d066-ext-v1.json"):
        pack = json.loads((ROOT / "src/openimcc/data/packs" / filename).read_text())
        rows = pack["sp_extension"]["rows"]
        assert len(rows) == 5
        assert {row["fit_residual"]["held_out_or_in_sample"] for row in rows} == {"held_out"}
        assert all(row["fit_residual"]["value"] for row in rows)
        assert all(row["accuracy_band"] is None for row in rows)
        assert all(row["band_basis"]["kind"] == "none" for row in rows)


def test_accuracy_band_never_repeats_its_own_fit_residual() -> None:
    def walk(value, label):
        if isinstance(value, Mapping):
            if FIELDS <= value.keys() and value["accuracy_band"] is not None and value["fit_residual"] is not None:
                assert value["accuracy_band"]["value"] != value["fit_residual"]["value"], label
            for key, child in value.items():
                walk(child, f"{label}/{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{label}/{index}")

    for path in (
        ROOT / "src/openimcc/data/packs/openimcc-redox-v2.json",
        ROOT / "src/openimcc/data/packs/openimcc-redox-v1.json",
        ROOT / "src/openimcc/data/gas/PROVENANCE.yaml",
        ROOT / "data-src/liquid-from-solid-inputs.json",
        *sorted((ROOT / "src/openimcc/data/packs").glob("imcc-sf04*.json")),
    ):
        raw = yaml.safe_load(path.read_text()) if path.suffix == ".yaml" else json.loads(path.read_text())
        walk(raw, path.name)


@pytest.mark.slow
def test_migration_reproduces_committed_outputs_from_baseline(tmp_path: Path) -> None:
    paths = [
        "src/openimcc/data/packs/openimcc-redox-v1.json",
        "data-src/redox-thermochemistry.json",
        "src/openimcc/data/gas/PROVENANCE.yaml",
        "data-src/liquid-from-solid-inputs.json",
    ]
    paths.extend(
        path.relative_to(ROOT).as_posix()
        for path in sorted((ROOT / "src/openimcc/data/packs").glob("imcc-sf04*.json"))
        # Candidate packs and corrected D066 v2-v4 packs were authored after
        # the band migration, so the f013bba baseline has no copy to migrate.
        if "-candidate" not in path.name
        and path.name not in {
            "imcc-sf04-d066-v2.json",
            "imcc-sf04-d066-v3.json",
            "imcc-sf04-d066-v4.json",
            "imcc-sf04-d066-v5.json",
        }
    )
    archive = subprocess.run(
        ["git", "archive", "f013bba"], cwd=ROOT, check=True, capture_output=True
    ).stdout
    with tarfile.open(fileobj=__import__("io").BytesIO(archive)) as source:
        source.extractall(tmp_path, filter="data")

    def row_key(row: dict) -> tuple:
        return (
            row.get("table"), row.get("species_name"), row.get("table_id"),
            row.get("T_interval"),
        )

    ion_species = {
        "Au+(g)", "Au-(g)", "Hg+(g)", "Pd+(g)", "Pt+(g)", "Pt-(g)",
    }
    current_gas_provenance = ROOT / "src/openimcc/data/gas/PROVENANCE.yaml"
    baseline_gas_provenance = tmp_path / "src/openimcc/data/gas/PROVENANCE.yaml"
    current_document = yaml.safe_load(current_gas_provenance.read_text())
    current_inputs = ROOT / "data-src/liquid-from-solid-inputs.json"
    (tmp_path / "data-src/liquid-from-solid-inputs.json").write_bytes(
        current_inputs.read_bytes()
    )
    current_construction_rows = [
        row for row in current_document["rows"]
        if row.get("construction_inputs_path") == "data-src/liquid-from-solid-inputs.json"
    ]
    construction_keys = {row_key(row) for row in current_construction_rows}
    baseline_document = yaml.safe_load(baseline_gas_provenance.read_text())
    baseline_document["rows"] = [
        row for row in baseline_document["rows"]
        if row_key(row) not in construction_keys
    ]
    pgm_rows = [
        row for row in current_document["rows"]
        if row.get("species_name") in ion_species
    ]
    saha_rows = [
        row for row in current_document["rows"]
        if str(row.get("table_id", "")).startswith("NIST-ASD-Saha-")
    ]
    assert len(pgm_rows) == len(ion_species)
    assert saha_rows
    baseline_document["rows"].extend(
        (*pgm_rows, *saha_rows, *current_construction_rows)
    )
    baseline_rows = baseline_document["rows"]
    baseline_document["rows"] = []
    baseline_text = yaml.safe_dump(
        baseline_document, allow_unicode=True, sort_keys=False
    )
    serialized_rows = "".join(
        "  - " + yaml.safe_dump(
            row, allow_unicode=True, sort_keys=False,
            default_flow_style=True, width=1_000_000,
        ).strip() + "\n"
        for row in baseline_rows
    )
    assert "rows: []\n" in baseline_text
    baseline_gas_provenance.write_text(
        baseline_text.replace("rows: []\n", "rows:\n" + serialized_rows, 1),
        encoding="utf-8",
    )
    migrate(tmp_path)
    gas_provenance = "src/openimcc/data/gas/PROVENANCE.yaml"
    for relative in paths:
        migrated_path = tmp_path / relative
        committed_path = ROOT / relative
        if relative != gas_provenance:
            assert migrated_path.read_bytes() == committed_path.read_bytes(), relative
            continue

        migrated_provenance = yaml.safe_load(migrated_path.read_text())
        committed_provenance = yaml.safe_load(committed_path.read_text())
        for document in (migrated_provenance, committed_provenance):
            for note in (
                "phosphorus_parent_channels",
                "negative_ion_screen",
                "constructed_saha_ions",
            ):
                document["source_notes"].pop(note, None)
            document["rows"] = [
                row
                for row in document["rows"]
                if row.get("species_name") not in {"PS(g)", "P2O5(l)"}
                and not (
                    row.get("table") == "gas"
                    and row.get("species_name") in {
                        "H2O(g)", "Zn(g)", "Cd(g)", "Pb(g)",
                        "PbO(g)", "PbO2(g)",
                    }
                )
                and not (
                    row.get("table") == "condensate"
                    and row.get("species_name") in {
                        "Na2O(l)", "K2O(l)", "CaO(l)", "MgO(l)",
                        "SiO2(l)", "Al2O3(l)",
                    }
                )
            ]
        committed_provenance["rows"] = [
            row
            for row in committed_provenance["rows"]
            if row.get("source_path") not in {
                "data-src/published-thermo/burcat-os.json",
                "data-src/published-thermo/pankratz-bulletin-677-re.json",
            }
        ]
        wave3_rows = {
            ("gas", "SbO(g)", "Kelley-584-SbO", 1),
            ("gas", "SbO(g)", "Kelley-584-SbO", 2),
            ("gas", "TeO(g)", "NUREG-4308-TeO", 1),
            ("gas", "TeO(g)", "NUREG-4308-TeO", 2),
            ("gas", "HCl(g)", "Cl-026", 2),
            ("gas", "HF(g)", "F-005", 2),
            ("gas", "NaCl(g)", "Cl-056", 2),
            ("gas", "(NaCl)2(g)", "Cl-098", 2),
            ("gas", "KCl(g)", "Cl-039", 2),
            ("gas", "(KCl)2(g)", "Cl-091", 2),
            ("gas", "NaF(g)", "F-036", 2),
            ("gas", "(NaF)2(g)", "F-082", 2),
            ("gas", "KF(g)", "F-020", 2),
            ("gas", "(KF)2(g)", "F-069", 2),
            ("gas", "Hg(g)", "BU-2587", 3),
        }
        migrated_rows = {row_key(row): row for row in migrated_provenance["rows"]}
        committed_rows = {row_key(row): row for row in committed_provenance["rows"]}
        assert set(committed_rows) - set(migrated_rows) == wave3_rows
        assert set(migrated_rows) <= set(committed_rows)
        selected_feo = ("condensate", "FeO(l)", None, None)
        post_migration_tier_rows = {
            ("gas", "Hg(g)", "BU-2587", 1),
            ("gas", "Hg(g)", "BU-2587", 2),
        }
        assert committed_rows[selected_feo]["authority"] == (
            "primary_measurement_reanchored"
        )
        assert migrated_rows[selected_feo]["authority"] == "janaf_transcribed"

        def migration_comparable_row(row: dict, key: tuple) -> dict:
            normalized = dict(row)
            normalized.pop("T_range_K", None)
            normalized.pop("fit_intervals_K", None)
            for field in (
                "table_id", "source_table_id", "runtime_ref", "source_path",
                "source_sha256", "upstream_source_sha256",
            ):
                normalized.pop(field, None)
            if key in post_migration_tier_rows:
                normalized.pop("tier", None)
            for field in ("fit_residual", "accuracy_band"):
                value = normalized.get(field)
                if isinstance(value, dict):
                    normalized[field] = {
                        key: item for key, item in value.items()
                        if key != "value"
                        and not (field == "fit_residual" and key == "temperature_range")
                    }
            return normalized

        mismatched_keys = {
            key for key, row in migrated_rows.items()
            if key != selected_feo
            and migration_comparable_row(committed_rows[key], key)
            != migration_comparable_row(row, key)
        }
        assert not mismatched_keys
        for key in set(migrated_provenance) - {"rows", "source_notes"}:
            assert committed_provenance[key] == migrated_provenance[key]
        assert set(committed_provenance["source_notes"]) == set(
            migrated_provenance["source_notes"]
        )
        for key, value in migrated_provenance["source_notes"].items():
            if key != "trace_channel_additions":
                assert committed_provenance["source_notes"][key] == value
    first = {relative: (tmp_path / relative).read_bytes() for relative in paths}
    migrate(tmp_path)
    second = {relative: (tmp_path / relative).read_bytes() for relative in paths}
    assert first == second


def test_redox_loader_rejects_legacy_band_field_with_typed_error(tmp_path: Path) -> None:
    from openimcc.redox_pack import RedoxSchemaError, load_redox_pack

    path = ROOT / "src/openimcc/data/packs/openimcc-redox-v1.json"
    content = json.loads(path.read_text())
    content["thermo_species"]["O2(g)"]["band_kj_mol"] = 0.0
    content["canonical_content_digest"] = canonical_redox_digest(content)
    candidate = tmp_path / "legacy-band-field.json"
    candidate.write_text(json.dumps(content))
    try:
        load_redox_pack(candidate)
    except RedoxSchemaError as exc:
        assert "legacy band fields" in str(exc)
    else:
        raise AssertionError("legacy band field was accepted")


def test_melt_loader_rejects_legacy_band_field_with_typed_error(tmp_path: Path) -> None:
    from openimcc.model import ImccMalformedDatapackError, load_datapack

    path = ROOT / "src/openimcc/data/packs/imcc-sf04-d066-v1.json"
    content = json.loads(path.read_text())
    content["rows"][0]["band_kj_mol"] = 0.0
    candidate = tmp_path / "legacy-melt-band-field.json"
    candidate.write_text(json.dumps(content))
    try:
        load_datapack(candidate)
    except ImccMalformedDatapackError as exc:
        assert "legacy fields" in str(exc)
    else:
        raise AssertionError("legacy band field was accepted")


def test_melt_loader_rejects_legacy_field_on_sp_extension_row(tmp_path: Path) -> None:
    from openimcc.model import ImccMalformedDatapackError, load_datapack

    path = ROOT / "src/openimcc/data/packs/imcc-sf04-ext-v4.json"
    content = json.loads(path.read_text())
    content["sp_extension"]["rows"][0]["band_kj_mol"] = 0.0
    candidate = tmp_path / "legacy-extension-band-field.json"
    candidate.write_text(json.dumps(content))
    try:
        load_datapack(candidate)
    except ImccMalformedDatapackError as exc:
        assert "legacy fields" in str(exc)
    else:
        raise AssertionError("legacy band field on S/P extension row was accepted")
