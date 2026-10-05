"""Tests for generator-side liquid Gibbs constructions from crystal tables."""

from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path

import pytest

from openimcc.gas import _lamor_gibbs
from tools.build_gas_tables import _janaf_text_source_rows, _load_record
from tools.liquid_from_solid import _crystal_rows, _crystal_state, liquid_from_solid


ROOT = Path(__file__).resolve().parents[1]
JANAF = ROOT / "data-src" / "janaf"
INPUTS_PATH = ROOT / "data-src" / "liquid-from-solid-inputs.json"
INPUTS = json.loads(INPUTS_PATH.read_text(encoding="utf-8"))
R_J_MOL_K = 8.314462618


@lru_cache(maxsize=None)
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
            return next(
                entry["value"]
                for entry in row["inputs"]["fusion_entropy"]["ladder"]
                if entry["tier"] == 1
            )
        if tier == 2:
            family = row.get("family")
            if family is None:
                return None
            family_rows = [
                other
                for other in training
                if other.get("family") == family and other is not row
            ]
            values = [
                other["inputs"]["fusion_entropy"]["value"]
                for other in family_rows
                if other["inputs"]["fusion_entropy"]["tier"] == 1
            ]
            values.extend(
                member["value"]
                for member in INPUTS.get(
                    "fusion_entropy_family_reference_values", {}
                ).get(row["family"], [])
                if member["formula"] != row["formula"]
            )
            if not values:
                return None
            return sum(values) / len(values)
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
        if not tier2.get("components"):
            return None
        return sum(components[name]["value"] for name in tier2["components"])
    crystal = _load_record(JANAF / f"{row['crystal_table_id']}.yaml")
    crystal_rows = _crystal_rows(crystal)
    return _crystal_state(
        crystal_rows, row["inputs"]["fusion_temperature"]["value"]
    )[2]


def _measured_error_band(input_key: str, tier: int) -> dict[str, object]:
    rows = [
        row
        for row in INPUTS["rows"].values()
        if "liquid_table_id" in row
    ]
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
            if cp is None:
                continue
        construction = _validation_construction(row["formula"], ds, cp)
        liquid_rows = _source_rows(row["liquid_table_id"])
        for offset in (0, 300, 800):
            temperature = row["inputs"]["fusion_temperature"]["value"] + offset
            if temperature > max(liquid_rows):
                continue
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
        "sample_size_by_offset": [
            len(errors_by_offset[offset]) for offset in (0, 300, 800)
        ],
        "max": [
            max(map(abs, errors_by_offset[offset]))
            if errors_by_offset[offset]
            else None
            for offset in (0, 300, 800)
        ],
        "rms": [
            math.sqrt(
                sum(error**2 for error in errors_by_offset[offset])
                / len(errors_by_offset[offset])
            )
            if errors_by_offset[offset]
            else None
            for offset in (0, 300, 800)
        ],
        "dex_max_per_metal_atom": [
            max(dex_by_offset[offset]) if dex_by_offset[offset] else None
            for offset in (0, 300, 800)
        ],
        "dex_rms_per_metal_atom": [
            math.sqrt(
                sum(error**2 for error in dex_by_offset[offset])
                / len(dex_by_offset[offset])
            )
            if dex_by_offset[offset]
            else None
            for offset in (0, 300, 800)
        ],
    }


def test_pair_records_have_canonical_source_provenance():
    table_ids = sorted(
        {
            table_id
            for row in INPUTS["rows"].values()
            if "liquid_table_id" in row
            for table_id in (row["crystal_table_id"], row["liquid_table_id"])
        }
    )
    assert len(table_ids) == 96
    for table_id in table_ids:
        text = (JANAF / f"{table_id}.yaml").read_text(encoding="utf-8")
        assert re.search(
            rf'(?m)^\s*["\']?table_id["\']?\s*:\s*["\']?{re.escape(table_id)}',
            text,
        )
        assert re.search(
            r'(?m)^\s*["\']?user_agent["\']?\s*:\s*["\']?openimcc-janaf-vendor/1\.0',
            text,
        )
        assert re.search(
            r'(?m)^\s*["\']?source_sha256["\']?\s*:\s*["\']?[0-9a-f]{64}',
            text,
        )
        assert not re.search(r'(?m)^\s*["\']?source_cache_path["\']?\s*:', text)


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
            elif key == "fusion_entropy":
                assert all(member["source"] and member["references"] for member in entry["training_members"])
            elif entry["value"] is not None:
                assert entry["components"]
                assert entry["provenance_class"] == "secondary_transcription_unverified_primary"


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


def test_k2o_tier2_cp_is_the_cited_mean_of_both_partial_molar_values():
    row = INPUTS["rows"]["K2O"]["inputs"]["liquid_heat_capacity"]["ladder"][1]
    component = INPUTS["partial_molar_cp_j_mol_k"]["K2O"]
    assert component["source_values"] == [98.5, 97.0]
    assert component["value"] == pytest.approx(97.75)
    assert row["value"] == pytest.approx(97.75)
    assert row["provenance_class"] == "secondary_transcription_unverified_primary"
    assert "Navrotsky (1995), Table 3, p. 130" in row["source"]
    assert row["references"] == [
        "https://doi.org/10.2138/rmg.1995.32.5",
        "https://doi.org/10.1007/BF00381840",
        "https://doi.org/10.1007/BF00310746",
    ]
    crystal = _load_record(JANAF / "K-012.yaml")
    crystal_cp = _crystal_state(_crystal_rows(crystal), 1013.0)[2]
    tier3 = INPUTS["rows"]["K2O"]["inputs"]["liquid_heat_capacity"]["ladder"][2]
    assert tier3["value"] == pytest.approx(crystal_cp)
    assert tier3["flag"] == "red"


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
            "fusion_temperature",
            1,
            {
                "sample_size_by_offset": [48, 48, 45],
                "max": [1.154923273, 1.164968030, 4.239951354],
                "rms": [0.166759609, 0.194222855, 0.686915566],
                "dex_max_per_metal_atom": [0.035569471, 0.030486221, 0.024323595],
                "dex_rms_per_metal_atom": [0.005134229, 0.004585847, 0.004784863],
            },
        ),
        (
            "fusion_entropy",
            1,
            {
                "sample_size_by_offset": [48, 48, 45],
                "max": [1.154923273, 1.164968030, 4.239951354],
                "rms": [0.166759609, 0.194222855, 0.686915566],
                "dex_max_per_metal_atom": [0.035569471, 0.030486221, 0.024323595],
                "dex_rms_per_metal_atom": [0.005134229, 0.004585847, 0.004784863],
            },
        ),
        (
            "fusion_entropy",
            2,
            {
                "sample_size_by_offset": [32, 32, 31],
                "max": [0.002385000, 7.356668197, 19.755915428],
                "rms": [0.000932845, 2.946475966, 8.139030435],
                "dex_max_per_metal_atom": [0.000046593, 0.133429696, 0.293724484],
                "dex_rms_per_metal_atom": [0.000016979, 0.040421986, 0.087489552],
            },
        ),
        (
            "fusion_entropy",
            3,
            {
                "sample_size_by_offset": [48, 48, 45],
                "max": [1.154923273, 16.463144105, 43.472805070],
                "rms": [0.166759609, 5.276230136, 13.853989157],
                "dex_max_per_metal_atom": [0.035569471, 0.232041418, 0.454135746],
                "dex_rms_per_metal_atom": [0.005134229, 0.064308831, 0.124734186],
            },
        ),
        (
            "liquid_heat_capacity",
            1,
            {
                "sample_size_by_offset": [48, 48, 45],
                "max": [1.154923273, 1.164968030, 4.239951354],
                "rms": [0.166759609, 0.194222855, 0.686915566],
                "dex_max_per_metal_atom": [0.035569471, 0.030486221, 0.024323595],
                "dex_rms_per_metal_atom": [0.005134229, 0.004585847, 0.004784863],
            },
        ),
        (
            "liquid_heat_capacity",
            2,
            {
                "sample_size_by_offset": [15, 15, 15],
                "max": [1.154923273, 1.674095117, 10.777037464],
                "rms": [0.298202531, 0.863964153, 5.144294318],
                "dex_max_per_metal_atom": [0.035569471, 0.027549534, 0.066705022],
                "dex_rms_per_metal_atom": [0.009184022, 0.010219438, 0.038084465],
            },
        ),
        (
            "liquid_heat_capacity",
            3,
            {
                "sample_size_by_offset": [48, 48, 45],
                "max": [1.154923273, 6.710438257, 45.281589063],
                "rms": [0.166759609, 1.613900384, 11.006661141],
                "dex_max_per_metal_atom": [0.035569471, 0.042529719, 0.208794255],
                "dex_rms_per_metal_atom": [0.005134229, 0.013327996, 0.064922836],
            },
        ),
    ],
)
def test_tier_error_bands_are_recomputed_and_pinned(input_key, tier, expected):
    key = f"{input_key}_tier{tier}"
    recorded = INPUTS["tier_error_bands"][key]
    recomputed = _measured_error_band(input_key, tier)
    assert recorded["sample_size_by_offset"] == expected["sample_size_by_offset"]
    assert recomputed["sample_size_by_offset"] == expected["sample_size_by_offset"]
    for metric in (
        "max",
        "rms",
        "dex_max_per_metal_atom",
        "dex_rms_per_metal_atom",
    ):
        assert recorded[metric] == pytest.approx(expected[metric], abs=1e-9)
        assert recomputed[metric] == pytest.approx(expected[metric], abs=1e-9)


def test_k2o_band_adds_the_selected_tier_bands_and_input_spread():
    row = INPUTS["rows"]["K2O"]
    crystal = json.loads((JANAF / "K-012.yaml").read_text(encoding="utf-8"))
    construction = liquid_from_solid("K2O", crystal, row["inputs"])
    entropy_band = INPUTS["tier_error_bands"]["fusion_entropy_tier2"]["max"]
    cp_band = INPUTS["tier_error_bands"]["liquid_heat_capacity_tier2"]["max"]
    assert row["band_validation_sample_by_tier"] == {
        "fusion_entropy_tier2": [32, 32, 31],
        "liquid_heat_capacity_tier2": [15, 15, 15],
    }
    for i, offset in enumerate((0, 300, 800)):
        measured_band = entropy_band[i] + cp_band[i]
        expected_band = measured_band + construction.input_spread_kj_mol(
            construction.fusion_temperature_k + offset
        )
        assert construction.band_kj_mol(
            construction.fusion_temperature_k + offset, measured_band
        ) == pytest.approx(expected_band)
