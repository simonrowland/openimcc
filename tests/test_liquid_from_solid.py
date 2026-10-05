"""Tests for generator-side liquid Gibbs constructions from crystal tables."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from openimcc.gas import _lamor_gibbs
from tools.build_gas_tables import _janaf_text_source_rows, _load_record
from tools.liquid_from_solid import _crystal_state, liquid_from_solid


ROOT = Path(__file__).resolve().parents[1]
JANAF = ROOT / "data-src" / "janaf"
INPUTS_PATH = ROOT / "data-src" / "liquid-from-solid-inputs.json"
INPUTS = json.loads(INPUTS_PATH.read_text(encoding="utf-8"))
R_J_MOL_K = 8.314462618


def _source_rows(table_id: str) -> dict[float, dict[str, float]]:
    record_path = JANAF / f"{table_id}.yaml"
    if not record_path.exists():
        return _janaf_text_source_rows(JANAF, table_id)
    record = _load_record(record_path)
    fields = (
        "temperature",
        "heat_capacity",
        "entropy",
        "enthalpy_increment",
        "formation_enthalpy",
    )
    rows = {}
    for source in record["table"]["values"]:
        row = {}
        for field in fields:
            cell = source.get(field)
            value = cell.get("value") if isinstance(cell, dict) else cell
            if value is not None:
                row[field] = float(value)
        if "temperature" in row:
            rows[row["temperature"]] = row
    for ambiguity in record["table"].get("parse_ambiguities", []):
        raw_line = ambiguity.get("raw_line", "")
        if "<-->" not in raw_line:
            continue
        cells = raw_line.split("\t")
        try:
            temperature, cp, entropy, _, enthalpy = map(float, cells[:5])
        except ValueError:
            continue
        rows[temperature] = {
            "temperature": temperature,
            "heat_capacity": cp,
            "entropy": entropy,
            "enthalpy_increment": enthalpy,
        }
    return rows


def _validation_construction(
    species: str, fusion_entropy: float, liquid_cp: float
):
    row = INPUTS["rows"][species]
    crystal = _source_rows(row["crystal_table_id"])
    inputs = {
        "fusion_temperature": {
            "value": row["inputs"]["fusion_temperature"]["value"]
        },
        "fusion_entropy": {"value": fusion_entropy},
        "liquid_heat_capacity": {"value": liquid_cp},
    }
    return liquid_from_solid(species, {"values": list(crystal.values())}, inputs)


def _reference_gibbs_kj_mol(
    rows: dict[float, dict[str, float]], temperature_k: float
) -> float:
    temperatures = sorted(rows)
    lower = max(t for t in temperatures if t <= temperature_k)
    upper = min(t for t in temperatures if t >= temperature_k)
    if lower == upper:
        enthalpy = rows[lower]["enthalpy_increment"]
        entropy = rows[lower]["entropy"]
    else:
        fraction = (temperature_k - lower) / (upper - lower)
        enthalpy = rows[lower]["enthalpy_increment"] + fraction * (
            rows[upper]["enthalpy_increment"]
            - rows[lower]["enthalpy_increment"]
        )
        entropy = rows[lower]["entropy"] + fraction * (
            rows[upper]["entropy"] - rows[lower]["entropy"]
        )
    dfh298 = rows[298.15]["formation_enthalpy"]
    return dfh298 + enthalpy - temperature_k * entropy / 1000.0


def _tier_estimate(row: dict, key: str, tier: int, training: list[dict]):
    if key == "fusion_entropy":
        if tier == 1:
            return row["inputs"]["fusion_entropy"]["value"]
        if tier == 2:
            family = [
                other
                for other in training
                if other["family"] == row["family"] and other is not row
            ]
            if not family:
                return None
            return sum(
                x["inputs"]["fusion_entropy"]["value"] for x in family
            ) / len(family)
        per_atom = [
            other["inputs"]["fusion_entropy"]["value"] / other["total_atoms"]
            for other in training
            if other is not row
        ]
        return sum(per_atom) / len(per_atom) * row["total_atoms"]

    if tier == 1:
        return row["inputs"]["liquid_heat_capacity"]["value"]
    if tier == 2:
        components = INPUTS["partial_molar_cp_j_mol_k"]
        tier2 = next(
            entry
            for entry in row["inputs"]["liquid_heat_capacity"]["ladder"]
            if entry["tier"] == 2
        )
        return sum(components[name]["value"] for name in tier2["components"])
    crystal = _source_rows(row["crystal_table_id"])
    crystal_rows = list(crystal.values())
    return _crystal_state(
        crystal_rows, row["inputs"]["fusion_temperature"]["value"]
    )[2]


def _measured_error_band(input_key: str, tier: int) -> dict[str, list[float]]:
    rows = [
        row
        for row in INPUTS["rows"].values()
        if "liquid_table_id" in row and "family" in row
    ]
    if input_key == "fusion_entropy" and tier == 2:
        rows = [row for row in rows if row["family"] == "alkali_metasilicate"]
    errors_by_offset = {offset: [] for offset in (0, 300, 800)}
    dex_by_offset = {offset: [] for offset in (0, 300, 800)}
    for row in rows:
        ds = _tier_estimate(row, "fusion_entropy", tier, rows)
        cp = _tier_estimate(row, "liquid_heat_capacity", 1, rows)
        if input_key == "fusion_entropy":
            if ds is None:
                continue
        else:
            ds = row["inputs"]["fusion_entropy"]["value"]
            cp = _tier_estimate(row, "liquid_heat_capacity", tier, rows)
        construction = _validation_construction(row["formula"], ds, cp)
        liquid_rows = _source_rows(row["liquid_table_id"])
        for offset in (0, 300, 800):
            temperature = row["inputs"]["fusion_temperature"]["value"] + offset
            error = construction.at(temperature).gibbs_kj_mol - _reference_gibbs_kj_mol(
                liquid_rows, temperature
            )
            errors_by_offset[offset].append(error)
            dex_by_offset[offset].append(
                abs(error)
                * 1000.0
                / (
                    R_J_MOL_K
                    * temperature
                    * math.log(10.0)
                    * row["metal_atoms"]
                )
            )
    return {
        "sample_size": len(errors_by_offset[0]),
        "max": [max(map(abs, errors_by_offset[offset])) for offset in (0, 300, 800)],
        "rms": [
            math.sqrt(
                sum(error**2 for error in errors_by_offset[offset])
                / len(errors_by_offset[offset])
            )
            for offset in (0, 300, 800)
        ],
        "dex_max_per_metal_atom": [
            max(dex_by_offset[offset]) for offset in (0, 300, 800)
        ],
        "dex_rms_per_metal_atom": [
            math.sqrt(
                sum(error**2 for error in dex_by_offset[offset])
                / len(dex_by_offset[offset])
            )
            for offset in (0, 300, 800)
        ],
    }


def test_vendor_records_hash_the_tab_delimited_files():
    table_ids = ("Na-016", "Na-017", "K-014", "K-015", "Mg-012", "Mg-013")
    for table_id in table_ids:
        record = json.loads((JANAF / f"{table_id}.yaml").read_text(encoding="utf-8"))
        source = (JANAF / f"{table_id}.txt").read_bytes()
        assert record["table"]["table_id"] == table_id
        assert record["extraction"]["user_agent"] == "openimcc-janaf-vendor/1.0"
        assert hashlib.sha256(source).hexdigest() == record["extraction"]["source_sha256"]


def test_validation_rows_record_each_input_tier_and_source():
    rows = [
        row
        for row in INPUTS["rows"].values()
        if "liquid_table_id" in row
    ]
    for row in rows:
        for key in ("fusion_temperature", "fusion_entropy", "liquid_heat_capacity"):
            value = row["inputs"][key]
            assert value["source"]
            assert value["tier"] == 1
            assert [entry["tier"] for entry in value["ladder"]] == (
                [1] if key == "fusion_temperature" else [1, 2, 3]
            )
            assert all(entry["source"] for entry in value["ladder"])
        for key, tier in (
            ("fusion_entropy", 2),
            ("fusion_entropy", 3),
            ("liquid_heat_capacity", 2),
            ("liquid_heat_capacity", 3),
        ):
            entry = next(
                candidate
                for candidate in row["inputs"][key]["ladder"]
                if candidate["tier"] == tier
            )
            estimated = _tier_estimate(row, key, tier, rows)
            if estimated is None:
                assert entry["value"] is None
            else:
                assert entry["value"] == pytest.approx(estimated)
            if tier == 3:
                assert entry["flag"] == "red"


@pytest.mark.parametrize(
    ("temperature_k", "expected_kj_mol"),
    [
        (1200, -555.309542381300),
        (1300, -582.045759429439),
        (1500, -637.722335504321),
        (2000, -787.694271591751),
        (2500, -950.015529158349),
        (3000, -1122.178018972846),
    ],
)
def test_k2o_tier2_central_values_are_pinned(temperature_k, expected_kj_mol):
    row = INPUTS["rows"]["K2O"]
    crystal = json.loads((JANAF / "K-012.yaml").read_text(encoding="utf-8"))
    construction = liquid_from_solid("K2O", crystal, row["inputs"])
    assert construction.at(temperature_k).gibbs_kj_mol == pytest.approx(
        expected_kj_mol, abs=1e-9
    )


@pytest.mark.parametrize(
    ("temperature_k", "expected_kj_mol"),
    [(1300, -582.301144218524), (2000, -790.252485679271)],
)
def test_k2o_tier3_cp_mechanism_check_keeps_original_values(
    temperature_k, expected_kj_mol
):
    row = INPUTS["rows"]["K2O"]
    crystal = json.loads((JANAF / "K-012.yaml").read_text(encoding="utf-8"))
    inputs = {**row["inputs"]}
    inputs["liquid_heat_capacity"] = {
        **row["inputs"]["liquid_heat_capacity"],
        "value": 104.6,
        "tier": 3,
        "spread": 0.0,
    }
    construction = liquid_from_solid("K2O", crystal, inputs)
    assert construction.at(temperature_k).gibbs_kj_mol == pytest.approx(
        expected_kj_mol, abs=0.01
    )
    assert construction.diagnostics["negative_fusion_delta_cp"] is True
    assert "crystal high-temperature Cp" in construction.diagnostics[
        "negative_fusion_delta_cp_note"
    ]


def test_negative_fusion_delta_cp_is_in_diagnostics_for_tier2_central():
    row = INPUTS["rows"]["K2O"]
    crystal = json.loads((JANAF / "K-012.yaml").read_text(encoding="utf-8"))
    construction = liquid_from_solid("K2O", crystal, row["inputs"])
    assert construction.diagnostics["fusion_delta_cp_j_mol_k"] == pytest.approx(
        -16.28681
    )
    assert construction.diagnostics["negative_fusion_delta_cp"] is True


def test_na2sio3_compound_example_matches_its_janaf_liquid_at_fusion():
    row = INPUTS["rows"]["Na2SiO3"]
    crystal_rows = _source_rows(row["crystal_table_id"])
    liquid_rows = _source_rows(row["liquid_table_id"])
    temperature = row["inputs"]["fusion_temperature"]["value"]
    inputs = {
        "fusion_temperature": {"value": temperature, "tier": 1, "spread": 0.0},
        "fusion_entropy": {
            "value": row["inputs"]["fusion_entropy"]["value"],
            "tier": 1,
            "spread": 0.001,
        },
        "liquid_heat_capacity": {
            "value": row["inputs"]["liquid_heat_capacity"]["value"],
            "tier": 1,
            "spread": 0.001,
        },
        "condensate_fit": row["inputs"]["condensate_fit"],
    }
    construction = liquid_from_solid(
        "Na2SiO3", {"values": list(crystal_rows.values())}, inputs
    )
    liquid_at_fusion = liquid_rows[temperature]
    reference_g = (
        liquid_rows[298.15]["formation_enthalpy"]
        + liquid_at_fusion["enthalpy_increment"]
        - temperature * liquid_at_fusion["entropy"] / 1000.0
    )
    assert construction.at(temperature).gibbs_kj_mol == pytest.approx(
        reference_g, abs=0.003
    )
    assert construction.diagnostics["negative_fusion_delta_cp"] is True


def test_fitted_condensate_row_uses_the_existing_runtime_evaluator():
    row = INPUTS["rows"]["Na2SiO3"]
    crystal_rows = _source_rows(row["crystal_table_id"])
    inputs = {
        "fusion_temperature": {
            "value": row["inputs"]["fusion_temperature"]["value"]
        },
        "fusion_entropy": {
            "value": row["inputs"]["fusion_entropy"]["value"]
        },
        "liquid_heat_capacity": {
            "value": row["inputs"]["liquid_heat_capacity"]["value"]
        },
        "condensate_fit": row["inputs"]["condensate_fit"],
    }
    construction = liquid_from_solid(
        "Na2SiO3", {"values": list(crystal_rows.values())}, inputs
    )
    fitted = construction.fit_condensate_row()
    runtime_row = {
        key: float(fitted[key])
        for key in ("dG_A", "dG_B", "dG_C", "dG_D", "dG_E", "dH298_R")
    }
    for temperature in (1200, 1362, 1500, 1800, 2000):
        assert _lamor_gibbs(temperature, runtime_row) / 1000.0 == pytest.approx(
            construction.at(temperature).gibbs_kj_mol, abs=0.005
        )


def test_below_fusion_is_labeled_as_supercooled_extrapolation():
    row = INPUTS["rows"]["K2O"]
    crystal = json.loads((JANAF / "K-012.yaml").read_text(encoding="utf-8"))
    construction = liquid_from_solid("K2O", crystal, row["inputs"])
    assert construction.at(900).branch == "supercooled_extrapolation"
    assert construction.at(1200).branch == "liquid"


@pytest.mark.parametrize(
    ("input_key", "tier", "expected"),
    [
        (
            "fusion_entropy",
            2,
            {
                "sample_size": 2,
                "max": [0.001551, 0.778006, 1.832280],
                "rms": [0.001361, 0.658539, 1.735373],
                "dex_max_per_metal_atom": [0.000022, 0.008150, 0.014756],
                "dex_rms_per_metal_atom": [0.000018, 0.007055, 0.014322],
            },
        ),
        (
            "fusion_entropy",
            3,
            {
                "sample_size": 6,
                "max": [1.154923, 8.071990, 19.581036],
                "rms": [0.471497, 4.144969, 10.424638],
                "dex_max_per_metal_atom": [0.035569, 0.211237, 0.409771],
                "dex_rms_per_metal_atom": [0.014521, 0.091497, 0.180945],
            },
        ),
        (
            "liquid_heat_capacity",
            2,
            {
                "sample_size": 6,
                "max": [1.154923, 1.052749, 4.913977],
                "rms": [0.471497, 0.613449, 2.769430],
                "dex_max_per_metal_atom": [0.035569, 0.027550, 0.048429],
                "dex_rms_per_metal_atom": [0.014521, 0.012452, 0.028353],
            },
        ),
        (
            "liquid_heat_capacity",
            3,
            {
                "sample_size": 6,
                "max": [1.154923, 0.922182, 6.578776],
                "rms": [0.471497, 0.581916, 3.183846],
                "dex_max_per_metal_atom": [0.035569, 0.022441, 0.054946],
                "dex_rms_per_metal_atom": [0.014521, 0.010385, 0.028462],
            },
        ),
    ],
)
def test_tier_error_bands_are_recomputed_and_pinned(input_key, tier, expected):
    key = f"{input_key}_tier{tier}"
    recorded = INPUTS["tier_error_bands"][key]
    recomputed = _measured_error_band(input_key, tier)
    assert recomputed["sample_size"] == expected["sample_size"]
    assert recorded == expected
    for metric in (
        "max",
        "rms",
        "dex_max_per_metal_atom",
        "dex_rms_per_metal_atom",
    ):
        assert recomputed[metric] == pytest.approx(expected[metric], abs=1e-6)


def test_k2o_band_adds_the_selected_tier_bands_and_input_spread():
    row = INPUTS["rows"]["K2O"]
    crystal = json.loads((JANAF / "K-012.yaml").read_text(encoding="utf-8"))
    construction = liquid_from_solid("K2O", crystal, row["inputs"])
    entropy_band = INPUTS["tier_error_bands"]["fusion_entropy_tier2"]["max"]
    cp_band = INPUTS["tier_error_bands"]["liquid_heat_capacity_tier2"]["max"]
    for i, offset in enumerate((0, 300, 800)):
        measured_band = entropy_band[i] + cp_band[i]
        expected_band = measured_band + construction.input_spread_kj_mol(
            construction.fusion_temperature_k + offset
        )
        assert construction.band_kj_mol(
            construction.fusion_temperature_k + offset, measured_band
        ) == pytest.approx(expected_band)
