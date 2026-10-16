"""Coverage for the split between fit residuals and accuracy bands."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path

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
    assert not ({"band_kj_mol", "band_method", "uncertainty_band", "max_residual_J_per_mol", "max_residual_log10_K"} & row.keys()), label
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
    if row["accuracy_band"] is not None and row["fit_residual"] is not None:
        assert row["accuracy_band"]["value"] != row["fit_residual"]["value"] or basis["kind"] == "independent_measurement", label


def test_every_published_row_uses_the_band_schema() -> None:
    redox = json.loads((ROOT / "src/openimcc/data/packs/openimcc-redox-v1.json").read_text())
    for key, row in redox["thermo_species"].items():
        _validate_row(row, f"thermo_species/{key}")
    for index, row in enumerate(redox["rows"]):
        _validate_row(row, f"redox/rows/{index}")

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
        ROOT / "src/openimcc/data/packs/openimcc-redox-v1.json",
        ROOT / "src/openimcc/data/gas/PROVENANCE.yaml",
        ROOT / "data-src/liquid-from-solid-inputs.json",
        *sorted((ROOT / "src/openimcc/data/packs").glob("imcc-sf04*.json")),
    ):
        raw = yaml.safe_load(path.read_text()) if path.suffix == ".yaml" else json.loads(path.read_text())
        walk(raw, path.name)


def test_migration_is_idempotent(tmp_path: Path) -> None:
    paths = [
        "src/openimcc/data/packs/openimcc-redox-v1.json",
        "data-src/redox-thermochemistry.json",
        "src/openimcc/data/gas/PROVENANCE.yaml",
        "data-src/liquid-from-solid-inputs.json",
    ]
    paths.extend(
        path.relative_to(ROOT).as_posix()
        for path in sorted((ROOT / "src/openimcc/data/packs").glob("imcc-sf04*.json"))
    )
    for relative in paths:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    migrate(tmp_path)
    first = {relative: hashlib.sha256((tmp_path / relative).read_bytes()).digest() for relative in paths}
    migrate(tmp_path)
    second = {relative: hashlib.sha256((tmp_path / relative).read_bytes()).digest() for relative in paths}
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
