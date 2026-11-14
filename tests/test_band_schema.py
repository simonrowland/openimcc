"""Coverage for the split between fit residuals and accuracy bands."""

from __future__ import annotations

import json
import subprocess
import tarfile
from copy import deepcopy
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml

from openimcc.redox_pack import canonical_redox_digest
from openimcc.model import ImccMalformedDatapackError, _validate_band_schema_rows
from openimcc.redox_pack import RedoxSchemaError, _validate_band_schema
from tools.migrate_band_schema import (
    _LIQUID_INPUT_BAND_CLASS_BY_KIND,
    _MELT_DIAGNOSTIC_SPLIT_BY_KEY,
    _REDOX_REACTION_BAND_CLASS_BY_EVIDENCE,
    _REDOX_SPECIES_BAND_CLASS_BY_EVIDENCE,
    _SOURCE_BAND_CLASS_BY_KIND,
    _liquid_input_band_class,
    _melt_held_out_split,
    _normalized_controlled_value,
    _redox,
    _redox_reaction_band_class,
    _redox_species_band_class,
    _source_band_kind,
    migrate,
    migrate_melt_row,
)

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
_HG_DUPLICATE_INTERVALS = {
    ("gas", "Hg(g)", "BU-2587", 1, (1500.0, 3000.0)),
    ("gas", "Hg(g)", "BU-2587", 2, (1200.0, 1500.0)),
}


def _committed_document(revision: str, path: str) -> dict:
    content = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return yaml.safe_load(content) if path.endswith((".yaml", ".yml")) else json.loads(content)


def _baseline_band_record() -> dict:
    return {
        "fit_residual": None,
        "accuracy_band": None,
        "band_basis": {
            "kind": "none",
            "explanation": "no separate band applies",
            "provenance": "test fixture",
        },
        "independent_check": {
            "status": "not_found",
            "equilibrium": None,
            "primary_source": None,
            "conditions": None,
            "measured_value": None,
            "predicted_value": None,
            "miss": None,
            "miss_to_band_ratio": None,
            "reason": "no check was located",
        },
    }


def test_source_band_kind_controlled_vocabulary() -> None:
    assert len(_SOURCE_BAND_CLASS_BY_KIND) == 30
    for kind, expected in _SOURCE_BAND_CLASS_BY_KIND.items():
        assert _source_band_kind({"kind": kind})[0] == expected
    assert _source_band_kind({"kind": "measured fit residual"})[0] is None
    historical_documents = (
        _committed_document(
            "b5f1172^", "src/openimcc/data/gas/PROVENANCE.yaml"
        ),
        _committed_document(
            "f013bba", "src/openimcc/data/gas/PROVENANCE.yaml"
        ),
    )
    kinds = set()
    for document in historical_documents:
        for row in document["rows"]:
            band = row.get("uncertainty_band")
            if isinstance(band, dict) and band.get("kind"):
                kinds.add(_normalized_controlled_value(band["kind"]))
    assert kinds == set(_SOURCE_BAND_CLASS_BY_KIND)


def test_liquid_input_band_kind_controlled_vocabulary() -> None:
    assert len(_LIQUID_INPUT_BAND_CLASS_BY_KIND) == 3
    for kind, expected in _LIQUID_INPUT_BAND_CLASS_BY_KIND.items():
        assert _liquid_input_band_class(kind) == expected
    assert _liquid_input_band_class("unlisted measured construction") is None
    kinds = set()
    for revision in ("b5f1172^",):
        document = _committed_document(
            revision, "data-src/liquid-from-solid-inputs.json"
        )
        for row in document["rows"].values():
            band = row.get("measured_band")
            if isinstance(band, dict) and band.get("kind"):
                kinds.add(_normalized_controlled_value(band["kind"]))
    current = json.loads(
        (ROOT / "data-src/liquid-from-solid-inputs.json").read_text()
    )
    for row in current["rows"].values():
        band = row.get("accuracy_band")
        value = band.get("value") if isinstance(band, dict) else None
        if isinstance(value, dict) and value.get("kind"):
            kinds.add(_normalized_controlled_value(value["kind"]))
    assert kinds == set(_LIQUID_INPUT_BAND_CLASS_BY_KIND)


def test_redox_species_band_controlled_vocabulary() -> None:
    assert len(_REDOX_SPECIES_BAND_CLASS_BY_EVIDENCE) == 17
    for key, expected in _REDOX_SPECIES_BAND_CLASS_BY_EVIDENCE.items():
        tier, rationale, construction, evaluator = key
        row = {
            "provenance": {"tier": tier, "selection_rationale": rationale},
            "construction": construction,
            "evaluator": evaluator,
        }
        assert _redox_species_band_class(row) == expected
    key = next(
        key
        for key, expected in _REDOX_SPECIES_BAND_CLASS_BY_EVIDENCE.items()
        if expected == "measured_construction"
    )
    tier, rationale, _, evaluator = key
    unlisted_construction = {
        "provenance": {"tier": tier, "selection_rationale": rationale},
        "evaluator": evaluator,
    }
    for value in (None, "false", "not construction", 1):
        assert _redox_species_band_class(
            {**unlisted_construction, "construction": value}
        ) == "fit_only"
    legacy = _committed_document(
        "b5f1172^", "src/openimcc/data/packs/openimcc-redox-v1.json"
    )
    evidence = set()
    for row in legacy["thermo_species"].values():
        if "band_kj_mol" not in row:
            continue
        provenance = row.get("provenance", {})
        evidence.add((
            _normalized_controlled_value(provenance.get("tier")),
            _normalized_controlled_value(provenance.get("selection_rationale")),
            bool(row.get("construction")),
            _normalized_controlled_value(row.get("evaluator")),
        ))
    assert evidence == set(_REDOX_SPECIES_BAND_CLASS_BY_EVIDENCE)


def test_redox_reaction_band_controlled_vocabulary() -> None:
    assert len(_REDOX_REACTION_BAND_CLASS_BY_EVIDENCE) == 6
    for (method, note), expected in _REDOX_REACTION_BAND_CLASS_BY_EVIDENCE.items():
        assert _redox_reaction_band_class(method, note) == expected
    for method, note in (
        ("not construction", ""),
        ("measured fit residual without correction", ""),
        ("construction", "formerly measured; now fit-only"),
    ):
        assert _redox_reaction_band_class(method, note) == "fit_only"
    legacy = _committed_document(
        "b5f1172^", "src/openimcc/data/packs/openimcc-redox-v1.json"
    )
    evidence = {
        (
            _normalized_controlled_value(row.get("band_method")),
            _normalized_controlled_value(row.get("uncertainty_note")),
        )
        for row in legacy["rows"]
        if "band_kj_mol" in row
    }
    assert evidence == set(_REDOX_REACTION_BAND_CLASS_BY_EVIDENCE)


def test_melt_diagnostic_split_controlled_vocabulary() -> None:
    assert len(_MELT_DIAGNOSTIC_SPLIT_BY_KEY) == 7
    for key, expected in _MELT_DIAGNOSTIC_SPLIT_BY_KEY.items():
        assert _melt_held_out_split([key]) == expected
    for key in (
        "not.heldout_residual",
        "not_heldout_residual",
        "not-heldout-residual",
        "not: measured fit residual",
        "isn't measured",
        "withheldout_residual",
    ):
        assert _melt_held_out_split([key]) == "held_out_or_candidate"
    actual = set()
    for path in sorted((ROOT / "src/openimcc/data/packs").glob("imcc-sf04*.json")):
        document = json.loads(path.read_text())
        for row in document["rows"] + document.get("sp_extension", {}).get("rows", []):
            replayed = deepcopy(row)
            migrate_melt_row(replayed)
            fit = replayed.get("fit_residual")
            value = fit.get("value") if isinstance(fit, dict) else None
            if isinstance(value, dict):
                actual.update(_normalized_controlled_value(key) for key in value)
    assert actual == set(_MELT_DIAGNOSTIC_SPLIT_BY_KEY)


@pytest.mark.parametrize(
    "rationale",
    [
        "unmeasured fit residual",
        "not directly measured fit residual",
        "never measured fit residual",
        "non-measured fit residual",
        "no measured fit residual",
        "without measured support, fit residual",
        "un-measured fit residual",
        "not: measured fit residual",
        "isn't measured",
        "measured fit residual without correction",
        "formerly measured fit residual",
    ],
)
def test_redox_migration_requires_affirmative_measured_evidence(
    tmp_path: Path,
    rationale: str,
) -> None:
    pack_path = tmp_path / "redox.json"
    pack = {
        "units": {},
        "thermo_species": {
            "FeO(l)": {
                "band_kj_mol": 1.0,
                "T_domain_K": [1000.0, 2000.0],
                "provenance": {
                    "tier": "tier3_flagged_estimate",
                    "selection_rationale": rationale,
                },
            },
            "Fe(metal)": {},
            "Fe(l)": {},
            "Fe(cr)": {},
            "O2(g)": {},
            "MnO(l)": {},
            "MnSiO3(l)": {},
            "Mn2SiO4(l)": {},
            "MnAl2O4(l)": {},
        },
        "rows": [
            {
                "band_kj_mol": 1.0,
                "band_method": "not construction",
                "uncertainty_note": "",
                "row": 2,
                "reaction": "Fe + 1/2 O2 = FeO",
                "T_domain_K": [1000.0, 2000.0],
            }
        ],
    }
    pack_path.write_text(json.dumps(pack), encoding="utf-8")

    _redox(tmp_path, pack_path)

    migrated = json.loads(pack_path.read_text(encoding="utf-8"))
    feo = migrated["thermo_species"]["FeO(l)"]
    assert feo["accuracy_band"] is None
    assert feo["fit_residual"]["value"] == 1.0
    assert feo["band_basis"]["kind"] == "fit_only"
    assert migrated["rows"][0]["band_basis"]["kind"] == "fit_only"


def test_redox_migration_accepts_affirmative_measured_evidence(tmp_path: Path) -> None:
    key = next(
        key
        for key, band_class in _REDOX_SPECIES_BAND_CLASS_BY_EVIDENCE.items()
        if band_class == "measured_construction"
    )
    tier, rationale, construction, evaluator = key
    pack_path = tmp_path / "redox.json"
    pack = {
        "units": {},
        "thermo_species": {
            "FeO(l)": {
                "band_kj_mol": 1.0,
                "T_domain_K": [1000.0, 2000.0],
                "provenance": {
                    "tier": tier,
                    "selection_rationale": rationale,
                },
                "construction": construction,
                "evaluator": evaluator,
            },
            "Fe(metal)": {},
            "Fe(l)": {},
            "Fe(cr)": {},
            "O2(g)": {},
            "MnO(l)": {},
            "MnSiO3(l)": {},
            "Mn2SiO4(l)": {},
            "MnAl2O4(l)": {},
        },
        "rows": [
            {
                "band_kj_mol": 1.0,
                "row": 2,
                "reaction": "Fe + 1/2 O2 = FeO",
                "T_domain_K": [1000.0, 2000.0],
            }
        ],
    }
    pack_path.write_text(json.dumps(pack), encoding="utf-8")

    _redox(tmp_path, pack_path)

    migrated = json.loads(pack_path.read_text(encoding="utf-8"))
    assert migrated["thermo_species"]["FeO(l)"]["band_basis"]["kind"] == (
        "measured_construction"
    )


@pytest.mark.parametrize(
    ("diagnostic_key", "expected"),
    [
        ("provenance/max_abs_heldout_log10K_residual", "held_out"),
        ("heldout_residual", "held_out_or_candidate"),
        ("notheldout_residual", "held_out_or_candidate"),
        ("withheldout_residual", "held_out_or_candidate"),
        ("not_heldout_residual", "held_out_or_candidate"),
        ("not-heldout-residual", "held_out_or_candidate"),
        ("non_heldout_residual", "held_out_or_candidate"),
        ("never_heldout_residual", "held_out_or_candidate"),
        ("un_heldout_residual", "held_out_or_candidate"),
        ("no-heldout-residual", "held_out_or_candidate"),
    ],
)
def test_melt_migration_requires_heldout_token(
    diagnostic_key: str,
    expected: str,
) -> None:
    row = {"T_domain_K": [1000.0, 2000.0], diagnostic_key: 1.0}

    migrate_melt_row(row)

    assert row["fit_residual"]["held_out_or_in_sample"] == expected


def test_melt_band_validator_f0_baseline_contract() -> None:
    rejected_fields = (
        "band_kj_mol", "band_method", "uncertainty_band",
        "max_residual_J_per_mol", "max_residual_J_per_mol_by_interval",
        "max_residual_log10_K", "additional_uncertainty_bands",
        "max_abs_heldout_log10K_residual",
    )
    cases = [
        ("valid", lambda row: None, None),
        (
            "missing key",
            lambda row: row.pop("accuracy_band"),
            "sample 0 is missing fields ['accuracy_band']",
        ),
        (
            "NaN fit residual is unchecked",
            lambda row: row.update(fit_residual={"value": float("nan"), "unit": "K"}),
            None,
        ),
        (
            "negative numeric fit residual is unchecked",
            lambda row: row.update(fit_residual={"value": -1.0, "unit": "K"}),
            None,
        ),
        (
            "invalid fit residual unit is unchecked",
            lambda row: row.update(fit_residual={"value": 1.0, "unit": ""}),
            None,
        ),
        (
            "negative accuracy band is rejected as non-null",
            lambda row: row.update(accuracy_band={"value": -1.0, "unit": "K"}),
            "sample 0 complex rows must have accuracy_band null",
        ),
        (
            "basis wrong type",
            lambda row: row.update(band_basis=[]),
            "sample 0 has an invalid band_basis",
        ),
        (
            "numeric explanation remains accepted",
            lambda row: row["band_basis"].update(explanation=1),
            None,
        ),
        (
            "missing provenance remains accepted",
            lambda row: row["band_basis"].pop("provenance"),
            None,
        ),
        (
            "independent check wrong type",
            lambda row: row.update(independent_check=[]),
            "sample 0 has an incomplete independent_check",
        ),
        (
            "numeric not-found reason remains accepted",
            lambda row: row["independent_check"].update(reason=1),
            None,
        ),
        (
            "competing legacy and missing-key errors",
            lambda row: (row.pop("accuracy_band"), row.update(band_kj_mol=1)),
            "sample 0 contains legacy fields ['band_kj_mol']",
        ),
        (
            "competing accuracy and basis errors",
            lambda row: (row.update(accuracy_band=1), row.update(band_basis=[])),
            "sample 0 complex rows must have accuracy_band null",
        ),
    ]
    cases.extend(
        (
            f"legacy field {field}",
            lambda row, field=field: row.update({field: 1}),
            f"sample 0 contains legacy fields ['{field}']",
        )
        for field in rejected_fields
    )

    for name, change, expected_error in cases:
        row = _baseline_band_record()
        change(row)
        if expected_error is None:
            _validate_band_schema_rows({"rows": [row]}, row_label="sample")
        else:
            with pytest.raises(ImccMalformedDatapackError) as raised:
                _validate_band_schema_rows({"rows": [row]}, row_label="sample")
            assert str(raised.value) == expected_error, name

    with pytest.raises(ImccMalformedDatapackError) as raised:
        _validate_band_schema_rows({"rows": None}, row_label="sample")
    assert str(raised.value) == "datapack rows must be a list"
    with pytest.raises(ImccMalformedDatapackError) as raised:
        _validate_band_schema_rows({"rows": [None]}, row_label="sample")
    assert str(raised.value) == "sample 0 must be an object"


def test_redox_band_validator_f0_baseline_contract() -> None:
    def residual(value: float, unit: str = "K") -> dict:
        return {
            "value": value,
            "unit": unit,
            "target_source": "test",
            "comparison_method": "test",
            "temperature_range": [100.0, 200.0],
            "held_out_or_in_sample": "held-out",
        }

    def accuracy(value: float, unit: str = "K") -> dict:
        return {
            "value": value,
            "unit": unit,
            "temperature_points_or_range": [100.0, 200.0],
            "coverage": "test",
        }

    rejected_fields = ("band_kj_mol", "band_method", "uncertainty_band")
    permitted_legacy_fields = (
        "max_residual_J_per_mol", "max_residual_J_per_mol_by_interval",
        "max_residual_log10_K", "additional_uncertainty_bands",
        "max_abs_heldout_log10K_residual",
    )
    cases = [
        ("valid", lambda row: None, None),
        (
            "missing key",
            lambda row: row.pop("accuracy_band"),
            "sample is missing band schema fields: ['accuracy_band']",
        ),
        (
            "wrong type",
            lambda row: row.update(fit_residual=1),
            "sample/fit_residual must be an object or null",
        ),
        (
            "negative numeric band",
            lambda row: row.update(accuracy_band=accuracy(-1.0)),
            "sample/accuracy_band/value must be non-negative",
        ),
        (
            "invalid unit string",
            lambda row: row.update(fit_residual=residual(1.0, "")),
            "sample/fit_residual/unit must be a non-empty string",
        ),
        (
            "basis wrong type",
            lambda row: row.update(band_basis=[]),
            "sample/band_basis/kind is not an allowed value",
        ),
        (
            "missing provenance",
            lambda row: row["band_basis"].pop("provenance"),
            "sample/band_basis needs an explanation",
        ),
        (
            "numeric explanation",
            lambda row: row["band_basis"].update(explanation=1),
            "sample/band_basis needs an explanation",
        ),
        (
            "wrong not-found reason type",
            lambda row: row["independent_check"].update(reason=1),
            "sample/independent_check/reason must explain not_found",
        ),
        (
            "competing legacy and missing-key errors",
            lambda row: (row.pop("accuracy_band"), row.update(band_kj_mol=1)),
            "sample contains legacy band fields: ['band_kj_mol']",
        ),
        (
            "competing residual and accuracy type errors",
            lambda row: (row.update(fit_residual=1), row.update(accuracy_band=1)),
            "sample/fit_residual must be an object or null",
        ),
    ]
    cases.extend(
        (
            f"legacy field {field}",
            lambda row, field=field: row.update({field: 1}),
            f"sample contains legacy band fields: ['{field}']",
        )
        for field in rejected_fields
    )
    cases.extend(
        (f"permitted legacy field {field}", lambda row, field=field: row.update({field: 1}), None)
        for field in permitted_legacy_fields
    )

    for name, change, expected_error in cases:
        row = _baseline_band_record()
        row["fit_residual"] = residual(1.0)
        change(row)
        if expected_error is None:
            _validate_band_schema(row, "sample")
        else:
            with pytest.raises(RedoxSchemaError) as raised:
                _validate_band_schema(row, "sample")
            assert str(raised.value) == expected_error, name

    row = _baseline_band_record()
    row["fit_residual"] = residual(float("nan"))
    with pytest.raises(ValueError) as raised:
        _validate_band_schema(row, "sample")
    assert str(raised.value) == "sample/fit_residual/value must be a finite number"


def _migration_row_identity(row: dict) -> tuple:
    temperature_range = row.get("T_range_K")
    return (
        row.get("table"), row.get("species_name"), row.get("table_id"),
        row.get("T_interval"),
        tuple(float(value) for value in temperature_range)
        if temperature_range is not None else None,
    )


def _keyed_migration_rows(rows: list[dict], label: str) -> list[tuple[tuple, dict]]:
    occurrences = Counter()
    first_payload_by_identity = {}
    keyed = []
    for row in rows:
        identity = _migration_row_identity(row)
        comparable = dict(row)
        if identity in _HG_DUPLICATE_INTERVALS:
            comparable.pop("tier", None)
        if identity in first_payload_by_identity:
            assert first_payload_by_identity[identity] == comparable, (
                f"differing duplicate {label} row payload: {identity}"
            )
        else:
            first_payload_by_identity[identity] = comparable
        occurrence = occurrences[identity]
        occurrences[identity] += 1
        keyed.append(((*identity, occurrence), row))
    keys = [key for key, _ in keyed]
    assert len(keys) == len(set(keys)), f"duplicate {label} row key"
    return keyed


def test_migration_duplicate_payload_corruption_is_rejected() -> None:
    first = {
        "table": "gas",
        "species_name": "Hg(g)",
        "table_id": "BU-2587",
        "T_interval": 1,
        "T_range_K": [1500.0, 3000.0],
        "tier": 3,
        "fit_residual": {"value": 1.0},
    }
    second = {key: value for key, value in first.items() if key != "tier"}
    assert [key[-1] for key, _ in _keyed_migration_rows(
        [first, second], "Hg"
    )] == [0, 1]

    corrupted = {**second, "fit_residual": {"value": 999}}
    with pytest.raises(AssertionError, match="differing duplicate Hg row payload"):
        _keyed_migration_rows([first, corrupted], "Hg")


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
    assert residual["value"]["max_residual_J_per_mol_by_interval"] == pytest.approx(
        [9.56932283600327, 3.19386472436599, 1.54911242239177],
        rel=1.0e-13,
        abs=0.0,
    )
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

    def fixture_key(value: list) -> tuple:
        assert len(value) == 6
        temperature_range = value[4]
        return (
            *value[:4],
            tuple(float(item) for item in temperature_range)
            if temperature_range is not None else None,
            int(value[5]),
        )

    # Re-pointed to cold-extension refs after the band-migration baseline.
    post_migration_runtime_refs = {
        (
            "gas", "As(g)", "P672-As-g", 2, (1200.0, 1500.0), 0,
        ): "P672-As-g-CP-EXT-1200",
        (
            "gas", "As2(g)", "P672-As2-g", 2, (1200.0, 1500.0), 0,
        ): "P672-As2-g-CP-EXT-1200",
        (
            "gas", "As4(g)", "P672-As4-g", 2, (1200.0, 1500.0), 0,
        ): "P672-As4-g-CP-EXT-1200",
        (
            "gas", "As4O6(g)", "P672-As4O6-g", 2, (1200.0, 1500.0), 0,
        ): "P672-As4O6-g-CP-EXT-1000",
    }
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
    construction_keys = {
        _migration_row_identity(row) for row in current_construction_rows
    }
    baseline_document = yaml.safe_load(baseline_gas_provenance.read_text())
    baseline_document["rows"] = [
        row for row in baseline_document["rows"]
        if _migration_row_identity(row) not in construction_keys
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
    baseline_keyed_rows = _keyed_migration_rows(baseline_rows, "baseline")
    baseline_rows_by_key = dict(baseline_keyed_rows)
    baseline_runtime_refs = {
        key: row["runtime_ref"]
        for key, row in baseline_keyed_rows
        if key in post_migration_runtime_refs
    }
    assert set(baseline_runtime_refs) == set(post_migration_runtime_refs)
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
                "lamoreaux_1987",
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
        migrated_keyed_rows = _keyed_migration_rows(
            migrated_provenance["rows"], "migrated"
        )
        committed_keyed_rows = _keyed_migration_rows(
            committed_provenance["rows"], "committed"
        )
        migrated_rows = dict(migrated_keyed_rows)
        raw_committed_rows = dict(committed_keyed_rows)
        exemption_fixture = yaml.safe_load(
            (ROOT / "tests/fixtures/migration_field_exemptions.yaml").read_text()
        )

        def fixture_entries(groups: list[dict]) -> dict[tuple, dict]:
            result = {}
            for group in groups:
                assert group["reason"]
                for entry in group["rows"]:
                    key = fixture_key(entry["row_key"])
                    assert key not in result
                    result[key] = entry["fields"]
            return result

        def fixture_keys(groups: list[dict]) -> set[tuple]:
            values = []
            for group in groups:
                assert group["reason"]
                values.extend(fixture_key(key) for key in group["row_keys"])
            assert len(values) == len(set(values))
            return set(values)

        post_migration_field_exemptions = fixture_entries(
            exemption_fixture["field_exemptions"]
        )
        expected_added_intervals = fixture_keys(
            exemption_fixture["added_intervals"]
        )
        expected_removed_intervals = fixture_keys(
            exemption_fixture["removed_intervals"]
        )

        committed_range_aliases = {}
        for key, fields in post_migration_field_exemptions.items():
            if "T_range_K" not in fields:
                continue
            expected_range = tuple(float(value) for value in fields["T_range_K"])
            committed_key = (*key[:4], expected_range, key[5])
            assert key in baseline_rows_by_key
            assert key in migrated_rows
            assert committed_key in raw_committed_rows
            assert tuple(float(value) for value in migrated_rows[key]["T_range_K"]) == key[4]
            assert raw_committed_rows[committed_key]["T_range_K"] == fields["T_range_K"]
            assert committed_key not in committed_range_aliases
            committed_range_aliases[committed_key] = key

        committed_rows = {}
        for key, row in committed_keyed_rows:
            comparison_key = committed_range_aliases.get(key, key)
            assert comparison_key not in committed_rows, (
                f"duplicate committed comparison key: {comparison_key}"
            )
            committed_rows[comparison_key] = row
        # The new default combination adds the Si2O2/AlSiO molecules and the
        # O-035 silica crystal interval after this migration baseline.
        expected_added_intervals |= {
            key for key in committed_rows
            if key[:2] in {
                ("gas", "Si2O2(g)"),
                ("gas", "AlSiO(g)"),
            }
            or key[:3] == ("condensate", "SiO2(cr)", "O-035")
        }
        # The O-035 crystal row replaces the historical LAM1987 interval.
        expected_removed_intervals |= {
            key for key in migrated_rows
            if key[:2] == ("condensate", "SiO2(cr)")
        }
        assert set(committed_rows) - set(migrated_rows) == set(
            expected_added_intervals
        )
        assert set(migrated_rows) - set(committed_rows) == expected_removed_intervals

        for key, expected_runtime_ref in post_migration_runtime_refs.items():
            assert committed_rows[key]["runtime_ref"] == expected_runtime_ref
            assert migrated_rows[key]["runtime_ref"] == baseline_runtime_refs[key]
        selected_feo = (
            "condensate", "FeO(l)", None, None, (1000.0, 5000.0), 0,
        )
        post_migration_tier_rows = {
            ("gas", "Hg(g)", "BU-2587", 1, (1500.0, 3000.0), 0),
            ("gas", "Hg(g)", "BU-2587", 2, (1200.0, 1500.0), 0),
        }
        lh87_silica_row = (
            "condensate", "SiO2(cr)", None, None, (1000.0, 1996.0), 0,
        )
        assert committed_rows[selected_feo]["authority"] == (
            "primary_measurement_reanchored"
        )
        assert migrated_rows[selected_feo]["authority"] == "janaf_transcribed"

        # 675031e raised eight condensate fit starts and Tl2O to close seams.
        # 675031e seam refits recomputed gas and condensate residual values.
        # 675031e recalculated parent-scale bands after refitting liquid parents.
        def migration_field_value(row: dict, field: str):
            if field in {"T_range_K", "fit_intervals_K"}:
                return row.get(field)
            container, member = field.split(".", 1)
            value = row.get(container)
            return value.get(member) if isinstance(value, dict) else None

        def baseline_migration_field_value(row: dict, field: str):
            if field in {"T_range_K", "fit_intervals_K"}:
                return row.get(field)
            if field == "fit_residual.value":
                fit = row.get("fit_residual")
                if isinstance(fit, dict) and "value" in fit:
                    return fit["value"]
                residual_fields = (
                    "max_residual_J_per_mol",
                    "max_residual_J_per_mol_by_interval",
                    "max_residual_log10_K",
                )
                return {name: row[name] for name in residual_fields if name in row} or None
            if field == "accuracy_band.value":
                band = row.get("accuracy_band")
                if isinstance(band, dict) and "value" in band:
                    return band["value"]
                if "combined_band" in row:
                    old_band = {
                        "combined_band": row["combined_band"],
                        "source_uncertainty": row.get("source_uncertainty"),
                    }
                else:
                    old_band = row.get("uncertainty_band")
                additional_bands = row.get("additional_uncertainty_bands")
                if old_band is not None:
                    return (
                        {**old_band, "additional_uncertainty_bands": additional_bands}
                        if additional_bands else old_band
                    )
                if additional_bands:
                    return {"additional_uncertainty_bands": additional_bands}
                return None
            raise AssertionError(f"unsupported pinned migration field: {field}")

        def migration_comparable_row(
            row: dict, key: tuple, migrated_row: dict
        ) -> dict:
            normalized = dict(row)
            exempted_fields = post_migration_field_exemptions.get(key, {})
            for field in ("T_range_K", "fit_intervals_K"):
                if field in exempted_fields:
                    normalized.pop(field, None)
            if "fit_residual.value" in exempted_fields:
                fit = normalized.get("fit_residual")
                if isinstance(fit, dict):
                    normalized["fit_residual"] = {**fit}
                    normalized["fit_residual"].pop("value", None)
            if "fit_residual.temperature_range" in exempted_fields:
                fit = normalized.get("fit_residual")
                if isinstance(fit, dict):
                    normalized["fit_residual"] = {**fit}
                    normalized["fit_residual"].pop("temperature_range", None)
            if "accuracy_band.value" in exempted_fields:
                band = normalized.get("accuracy_band")
                if isinstance(band, dict):
                    normalized["accuracy_band"] = {**band}
                    normalized["accuracy_band"].pop("value", None)
            for field in (
                "table_id", "source_table_id", "runtime_ref", "source_path",
                "source_sha256", "upstream_source_sha256", "source_data",
            ):
                if field not in migrated_row:
                    normalized.pop(field, None)
            if key in post_migration_runtime_refs:
                normalized.pop("runtime_ref", None)
            if key in post_migration_tier_rows:
                normalized.pop("tier", None)
            if key == lh87_silica_row:
                # The migration baseline predates the vendored transcription;
                # only SiO2(cr)'s transcription pointer differs across versions.
                source = dict(normalized["source"])
                source.pop("transcription", None)
                normalized["source"] = source
            return normalized

        compared_fields = (
            "T_range_K", "fit_intervals_K", "fit_residual.temperature_range",
            "fit_residual.value", "accuracy_band.value",
        )
        differing_field_pairs = {
            (key, field)
            for key in set(migrated_rows) & set(committed_rows)
            if key != selected_feo
            for field in compared_fields
            if migration_field_value(migrated_rows[key], field)
            != migration_field_value(committed_rows[key], field)
        }
        pinned_field_pairs = {
            (key, field)
            for key, fields in post_migration_field_exemptions.items()
            for field in fields
        }
        assert differing_field_pairs == pinned_field_pairs
        for key, fields in post_migration_field_exemptions.items():
            assert key in baseline_rows_by_key
            for field, expected_committed_value in fields.items():
                migrated_value = migration_field_value(migrated_rows[key], field)
                assert migration_field_value(
                    committed_rows[key], field
                ) == expected_committed_value
                assert migrated_value == baseline_migration_field_value(
                    baseline_rows_by_key[key], field
                )
                assert migrated_value != expected_committed_value

        mismatched_keys = {
            key for key, row in migrated_rows.items()
            if key in committed_rows
            and key != selected_feo
            and migration_comparable_row(committed_rows[key], key, row)
            != migration_comparable_row(row, key, row)
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
