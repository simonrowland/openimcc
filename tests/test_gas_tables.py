"""Data-source, provenance, fit, and cross-check gates for the gas tables."""

from __future__ import annotations

import csv
import io
import json
import hashlib
import math
import os
import re
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tools import build_gas_tables
from openimcc import default_gas_channels, species_thermo
from openimcc.gas import (
    _GAS_IONIZATION_PAIRS,
    _GAS_PROVENANCE_AUTHORITY,
    _OXIDE_PROVENANCE_AUTHORITY,
    _OXIDE_SOURCE_TABLE_IDS,
    _formula_atoms,
    ImccGasTemperatureOutsideDomainError,
    IMCC_GAS_CHANNEL_SPECIES,
    IMCC_SF04_WORKBOOK_GRID_K,
    R_J_MOL_K,
    _SF04_REACTIONS,
    SpeciesThermo,
    _default_reactions,
    evaluate_gas,
    _janaf_gibbs,
    _lamor_gibbs,
    _nearest_interval_row,
    load_gas_datapack,
)


ROOT = Path(__file__).resolve().parents[1]
GAS_DATA = ROOT / "src" / "openimcc" / "data" / "gas"
JANAF_DATA = ROOT / "data-src" / "janaf"
NASA_DATA = ROOT / "data-src" / "nasa-glenn"
PUBLISHED_DATA = ROOT / "data-src" / "published-thermo"
LH84_DATA = ROOT / "data-src" / "lh84"
PROVENANCE_PATH = GAS_DATA / "PROVENANCE.yaml"
_BASE_INTERVAL_1_SHA256 = "ea3a8db1419140329e36be433453fdc67969a7e3826d9af5a310d284a1526c0a"
_ION_GAS_PROVENANCE_SPECIES = tuple(
    source[0]
    for source in (
        *build_gas_tables.ION_GAS_TEXT_SOURCES,
        *build_gas_tables.ION_GAS_YAML_SOURCES,
        *build_gas_tables.TRACE_ION_GAS_TEXT_SOURCES,
        *build_gas_tables.TRACE_ION_GAS_NASA_SOURCES,
        *build_gas_tables.TRACE_ION_PUBLISHED_SOURCES,
        *build_gas_tables.TRACE_ION_NASA_SOURCES,
        *build_gas_tables.TRACE_ION_GAS_YAML_SOURCES,
    )
)

GAS_TABLE_IDS = {
    "Na": "Na-005",
    "K": "K-005",
    "SiO": "O-012",
    "Fe": "Fe-008",
    "FeO": "Fe-021",
    "Mg": "Mg-005",
    "MgO": "Mg-011",
    "SiO2": "O-040",
    "O": "O-001",
    "AlO": "Al-074",
    "AlO2": "Al-077",
    "Al2O": "Al-092",
    "Al2O2": "Al-094",
    "Na2": "Na-011",
    "NaO": "Na-008",
    "K2": "K-011",
    "KO": "K-008",
    "Si": "Si-005",
    "Al": "Al-005",
    "CaO": "Ca-030",
    "Ca": "Ca-006",
    "O2": "O-029",
    "Ti": "Ti-006",
    "TiO": "O-022",
    "TiO2": "O-046",
    "Al2": "Al-080",
    "Si2": "Si-008",
    "Si3": "Si-009",
    "Cr": "Cr-005",
    "CrO": "Cr-010",
    "CrO2": "Cr-011",
    "CrO3": "Cr-012",
    "V": "V-005",
    "VO": "O-026",
    "VO2": "O-076",
    "Nb": "Nb-005",
    "NbO": "Nb-011",
    "NbO2": "Nb-015",
    "Mn": "Mn-005",
    "Ni": "Ni-005",
    "Co": "Co-005",
    "P": "P-008",
    "P2": "P-012",
    "P4": "P-013",
    "PO": "O-004",
    "PO2": "O-032",
    "P4O6": "O-087",
    "P4O10": "O-095",
    "S": "S-006",
    "S2": "S-012",
    "S3": "S-016",
    "S4": "S-017",
    "S5": "S-018",
    "S6": "S-019",
    "S7": "S-020",
    "S8": "S-021",
    "SO": "O-010",
    "SSO": "O-011",
    "SO2": "O-034",
    "SO3": "O-058",
}

NASA_TABLE_IDS = {"Na2O": "NG-0905", "K2O": "NG-0760"}

PARENT_TABLE_IDS = {
    "Na2O": "Na-013",
    "K2O": "K-012",  # JANAF has crystal K2O; the liquid row uses its construction.
    "MgO": "Mg-009",
    "CaO": "Ca-028",
    "Al2O3": "Al-100",
    "SiO2": "O-038",
    "FeO": "Fe-019",
    "TiO2": "O-044",
    "Cr2O3": "Cr-015",
    "V2O3": "O-063",
    "NbO2": "Nb-013",
}

# Fitted, not transcribed, condensate rows: species -> JANAF table ID.
FITTED_CONDENSATE_TABLE_IDS = {
    "TiO2(l)": "O-044",
    "Cr2O3(l)": "Cr-015",
    "V2O3(l)": "O-063",
    "NbO2(l)": "Nb-013",
    "Na2O(l)": "Na-013",
}

JANAF_CONDENSATE_TABLE_IDS = {
    "FeO(l)": "Fe-019",
    "Na2O(l)": "Na-013",
    **FITTED_CONDENSATE_TABLE_IDS,
}

# Rounded just above the measured worst residual across all 101 packaged
# JANAF-backed gas interval rows and 1402 in-interval source nodes. The S
# maximum is P4O10(g) interval 2 at 500 K.
GAS_PROPERTY_RESIDUAL_LIMITS = {
    "Cp_J_molK": 0.129,
    "S_J_molK": 0.014,
    "H_app_kJ_mol": 0.009,
    "G_kJ_mol": 0.0021,
}

# Measured maximum absolute Gibbs residuals at complete source nodes, in
# kJ/mol. Cp/S/H residuals are derivative-fit information and are not gates.
JANAF_CONDENSATE_G_RESIDUALS = {
    ("FeO(l)", 1000.0): 2.273853686498944,
    ("TiO2(l)", 1500.0): 0.00687822891632095,
    ("Cr2O3(l)", 1900.0): 0.001551177412737161,
    ("V2O3(l)", 1700.0): 0.00129979831143282,
    ("NbO2(l)", 1200.0): 3.4924596548080445e-13,
    ("NbO2(l)", 1500.0): 0.001405104221543297,
    ("Na2O(l)", 1200.0): 0.0004841441633179784,
    ("Na2O(l)", 1500.0): 0.005218239571899176,
}

# G source disagreements are model-minus-JANAF, measured over complete liquid
# branch nodes in each row's overlap. Derivative residuals of LAM G fits are
# fit-implied and are not source-quantity gate pins.
LAM_PARENT_SOURCE_EXCEPTIONS = {
    "Al2O3(l)": {
        "table_id": "Al-100",
        "sources": "LH87 Tables 2/3 vs NIST-JANAF 4th-edition Al-100",
        "reason": "Both anchors refer to the stable 298 K solid; the LH87 and JANAF Al-100 liquid G fits differ without a source-stated reconciliation.",
        "ranges": {
            "G_kJ_mol": (-0.130343, 0.101552, 0.002),
        },
    },
    "SiO2(l)": {
        "table_id": "O-038",
        "sources": "LH87 Table 2 vs NIST-JANAF 4th-edition O-038",
        "reason": "LH87 Table 2 and JANAF O-038 give different liquid G thermochemistry with no source-stated reconciliation.",
        "ranges": {
            "G_kJ_mol": (-2.972323, -2.302783, 0.002),
        },
    },
    "MgO(l)": {
        "table_id": "Mg-009",
        "sources": "LH87 Table 2 vs NIST-JANAF 4th-edition Mg-009",
        "reason": "The complete JANAF nodes inside the packaged LH87 liquid interval are included; the sources do not reconcile their G difference.",
        "ranges": {
            "G_kJ_mol": (-0.998335, -0.356769, 0.000001),
        },
    },
    "CaO(l)": {
        "table_id": "Ca-028",
        "sources": "LH87 Table 2 vs NIST-JANAF 4th-edition Ca-028",
        "reason": "The complete JANAF nodes inside the packaged LH87 liquid interval are included; the sources do not reconcile their G difference.",
        "ranges": {
            "G_kJ_mol": (-6.185, -1.210793, 0.000001),
        },
    },
}

CANDIDATE_LIQUID_TABLE_IDS = {
    "VO(l)": "O-024",
    "V2O3(l)": "O-063",
    "V2O4(l)": "O-074",
    "V2O5(l)": "O-085",
    "NbO(l)": "Nb-009",
    "NbO2(l)": "Nb-013",
    "Nb2O5(l)": "Nb-017",
}

CANDIDATE_LIQUID_TRANSITIONS_K = {
    "VO(l)": 2063.0,
    "V2O3(l)": 2340.0,
    "V2O4(l)": 1818.0,
    "V2O5(l)": 943.0,
    "NbO(l)": 2210.0,
    "NbO2(l)": 2175.0,
    "Nb2O5(l)": 1785.0,
}

_SOURCE_FIELDS = (
    "temperature",
    "heat_capacity",
    "entropy",
    "enthalpy_increment",
    "formation_enthalpy",
    "formation_gibbs_energy",
)


@lru_cache(maxsize=None)
def _record(table_id: str) -> dict:
    text = (JANAF_DATA / f"{table_id}.yaml").read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return yaml.safe_load(text)


@lru_cache(maxsize=None)
def _nasa_record(table_id: str) -> dict:
    return json.loads((NASA_DATA / f"{table_id}.json").read_text(encoding="utf-8"))


def _value(row: dict, field: str):
    return row[field]["value"]


@lru_cache(maxsize=None)
def _complete_rows(table_id: str) -> list[dict[str, float]]:
    rows = []
    for raw in _record(table_id)["table"]["values"]:
        values = {field: _value(raw, field) for field in _SOURCE_FIELDS}
        if all(value is not None for value in values.values()):
            rows.append({field: float(value) for field, value in values.items()})
    return rows


@lru_cache(maxsize=None)
def _ambiguous_temperatures(table_id: str) -> set[float]:
    result = set()
    for ambiguity in _record(table_id)["table"].get("parse_ambiguities", []):
        token = str(ambiguity.get("raw_line", "")).strip().split("\t", 1)[0]
        try:
            result.add(float(token))
        except ValueError:
            continue
    return result


def _gas_row_for_interval(pack, species: str, interval: int) -> pd.Series:
    rows = pack.gas_df.loc[pack.gas_df.index == f"{species}(g)"]
    return rows.loc[rows["T_interval"].astype(int) == interval].iloc[0]


def _shomate_entropy(row: pd.Series, temperature: float) -> float:
    t = temperature / 1000.0
    return float(
        row["A"] * math.log(t)
        + row["B"] * t
        + row["C"] * t**2 / 2.0
        + row["D"] * t**3 / 3.0
        - row["E"] / (2.0 * t**2)
        + row["G"]
    )


def _synthetic_source_row(temperature: float) -> dict[str, dict[str, float]]:
    return {
        field: {"value": temperature if field == "temperature" else 1.0}
        for field in build_gas_tables.REQUIRED_FIELDS
    }


def test_usable_rows_requires_exact_fit_interval_and_declared_gaps() -> None:
    complete_temperatures = [
        float(temperature)
        for temperature in range(
            int(build_gas_tables.FIT_T_MIN),
            int(build_gas_tables.FIT_T_MAX) + 1,
            int(build_gas_tables.JANAF_GRID_STEP_K),
        )
    ]
    incomplete_tail = {
        "values": [
            _synthetic_source_row(temperature)
            for temperature in complete_temperatures[:-1]
        ],
        "parse_ambiguities": [],
    }
    with pytest.raises(ValueError, match="tail") as tail_error:
        build_gas_tables._usable_rows(incomplete_tail, "tail")
    assert "tail" in str(tail_error.value)

    unaccounted_gap = {
        "values": [
            _synthetic_source_row(temperature)
            for temperature in complete_temperatures
            if temperature != 2200.0
        ],
        "parse_ambiguities": [],
    }
    with pytest.raises(ValueError, match="gap") as gap_error:
        build_gas_tables._usable_rows(unaccounted_gap, "gap")
    assert "gap" in str(gap_error.value)

    disclosed_gap = {
        **unaccounted_gap,
        "parse_ambiguities": [{"raw_line": "2200\tambiguous"}],
    }
    assert len(build_gas_tables._usable_rows(disclosed_gap, "disclosed")) == 15

    # Two adjacent ambiguous rows open a 300 K hole: a coverage gap, not one
    # unreadable row, even though both rows are declared.
    wide_gap = {
        "values": [
            _synthetic_source_row(temperature)
            for temperature in complete_temperatures
            if temperature not in (2200.0, 2300.0)
        ],
        "parse_ambiguities": [
            {"raw_line": "2200\tambiguous"},
            {"raw_line": "2300\tambiguous"},
        ],
    }
    # On the 1500-3000 K, 100 K grid the row minimum trips first; the
    # one-skipped-point bound guards a wider fit interval.  Either refuses.
    with pytest.raises(
        ValueError, match="at most one skipped grid point|does not cover"
    ):
        build_gas_tables._usable_rows(wide_gap, "wide")

    repeated_node = {
        "values": [
            _synthetic_source_row(temperature)
            for temperature in [*complete_temperatures, 2000.0]
        ],
        "parse_ambiguities": [],
    }
    with pytest.raises(ValueError, match="repeats"):
        build_gas_tables._usable_rows(repeated_node, "repeat")

    low_temperatures = list(
        range(
            int(build_gas_tables.LOW_FIT_T_MIN),
            int(build_gas_tables.LOW_FIT_T_MAX) + 1,
            int(build_gas_tables.JANAF_GRID_STEP_K),
        )
    )
    assert len(low_temperatures) == 11
    low_with_disclosed_gap = {
        "values": [
            _synthetic_source_row(temperature)
            for temperature in low_temperatures
            if temperature != 1100
        ],
        "parse_ambiguities": [{"raw_line": "1100\tambiguous"}],
    }
    assert len(
        build_gas_tables._usable_rows(
            low_with_disclosed_gap,
            "low-disclosed",
            fit_t_min=build_gas_tables.LOW_FIT_T_MIN,
            fit_t_max=build_gas_tables.LOW_FIT_T_MAX,
            minimum_rows=build_gas_tables.LOW_FIT_MINIMUM_ROWS,
        )
    ) == 10

    low_unreported_gap = {
        **low_with_disclosed_gap,
        "parse_ambiguities": [],
    }
    with pytest.raises(ValueError, match="gap"):
        build_gas_tables._usable_rows(
            low_unreported_gap,
            "low-unreported",
            fit_t_min=build_gas_tables.LOW_FIT_T_MIN,
            fit_t_max=build_gas_tables.LOW_FIT_T_MAX,
            minimum_rows=build_gas_tables.LOW_FIT_MINIMUM_ROWS,
        )

    low_ambiguous_row = {
        "values": [_synthetic_source_row(temperature) for temperature in low_temperatures],
        "parse_ambiguities": [{"raw_line": "1100\tambiguous"}],
    }
    with pytest.raises(ValueError, match="parse-ambiguous row"):
        build_gas_tables._usable_rows(
            low_ambiguous_row,
            "low-ambiguous",
            fit_t_min=build_gas_tables.LOW_FIT_T_MIN,
            fit_t_max=build_gas_tables.LOW_FIT_T_MAX,
            minimum_rows=build_gas_tables.LOW_FIT_MINIMUM_ROWS,
        )


def test_fitter_validates_formula_and_emitted_phase() -> None:
    with pytest.raises(ValueError, match="formula"):
        build_gas_tables._fit_row(JANAF_DATA, "K(g)", "Na-005", "K", 1, 0)
    mixed_phase = {
        "table": {
            "table_id": "Cr-016",
            "index_entry": {"formula": "Cr2O3", "state": "cr,l"},
        }
    }
    assert mixed_phase["table"]["index_entry"]["state"] == "cr,l"
    with pytest.raises(ValueError, match="mixed-phase"):
        build_gas_tables._validate_record_identity(
            mixed_phase, "Cr2O3(l)", "Cr-016", "l"
        )


def test_supercooled_liquid_fitter_rejects_parse_ambiguous_anchor() -> None:
    with pytest.raises(ValueError, match="anchor at 1600 K is parse-ambiguous"):
        build_gas_tables._fit_supercooled_liquid_row(
            JANAF_DATA, "V2O3(l)", "O-063", "V", 2, 3, 1600.0, 1500.0
        )


def test_cr_atomic_source_uses_only_its_declared_fit_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _record("Cr-005")
    rows = build_gas_tables._usable_rows(record["table"], "Cr-005")
    assert rows[0]["temperature"] == 1500.0
    assert rows[-1]["temperature"] == 2900.0

    monkeypatch.setitem(build_gas_tables._FIT_T_MAX_BY_TABLE, "Cr-005", 3000.0)
    with pytest.raises(ValueError):
        build_gas_tables._usable_rows(record["table"], "Cr-005")


def test_cr_3000_thermal_cells_recover_g_app_without_refitting_coefficients() -> None:
    record = _record("Cr-005")
    raw_line = next(
        ambiguity["raw_line"]
        for ambiguity in record["table"]["parse_ambiguities"]
        if ambiguity["raw_line"].startswith("3000\t")
    )
    cells = raw_line.split("\t")
    assert cells[5] == "0. 0. 0."
    cp_source, entropy_source, enthalpy_increment_source = map(
        float, (cells[1], cells[2], cells[4])
    )
    reference_h = next(
        row["formation_enthalpy"]
        for row in _complete_rows("Cr-005")
        if row["temperature"] == 298.15
    )
    source_g = (reference_h + enthalpy_increment_source) * 1000.0 - 3000.0 * entropy_source

    packaged = next(
        row
        for row in csv.DictReader(
            io.StringIO((GAS_DATA / "gas-shomate.csv").read_text(encoding="utf-8"))
        )
        if row["species_name"] == "Cr(g)" and row["T_interval"] == "1"
    )
    t = 3.0
    A, B, C, D, E, F, G = (float(packaged[key]) for key in "ABCDEFG")
    # Premise: dfG(298) is ambiguous at the elemental boiling transition, but
    # the species' dfH298 and thermal cells still define its G_app. Algebra:
    # evaluate Cp, S, and H_app from the current Shomate fit, then compare with
    # dfH298 + (H-H298) - T*S. Unit check: H_app kJ/mol is converted to J/mol.
    # Sanity: all four residuals stay below the gas fit's existing gates.
    cp_fit = A + B * t + C * t**2 + D * t**3 + E / t**2
    entropy_fit = (
        A * math.log(t) + B * t + C * t**2 / 2.0
        + D * t**3 / 3.0 - E / (2.0 * t**2) + G
    )
    h_app_fit = (
        A * t + B * t**2 / 2.0 + C * t**3 / 3.0
        + D * t**4 / 4.0 - E / t + F
    )
    g_fit = h_app_fit * 1000.0 - 3000.0 * entropy_fit
    assert reference_h == 397.480
    assert source_g == pytest.approx(-215491.0, abs=1.0e-9)
    assert cp_fit - cp_source == pytest.approx(-0.039868, abs=1.0e-6)
    assert entropy_fit - entropy_source == pytest.approx(-0.000671, abs=1.0e-6)
    assert (h_app_fit - reference_h - enthalpy_increment_source) * 1000.0 == pytest.approx(
        -2.068, abs=1.0e-3
    )
    assert g_fit - source_g == pytest.approx(-0.054033, abs=1.0e-6)
    assert float(packaged["T_max"]) == 3000.0


def test_public_sources_contain_no_private_paths_or_tooling_names() -> None:
    # Vendored records carry harvesting metadata; only the neutral tool name
    # may ship, and no file may carry a local filesystem path.
    agents = {}
    expected_agents = {}
    for source_root, neutral_agent, pattern in (
        (ROOT / "data-src" / "janaf", "openimcc-janaf-vendor/1.0", "*.yaml"),
        (NASA_DATA, "neutral-source-vendor/1.0", "PROVENANCE.yaml"),
        (LH84_DATA, "neutral-source-vendor/1.0", "*.yaml"),
    ):
        for path in sorted(source_root.glob(pattern)):
            text = path.read_text(encoding="utf-8")
            agents[str(path.relative_to(ROOT))] = re.findall(
                r'"?user_agent"?:\s*"?([^",\n]+)"?', text
            )
            expected_agents[str(path.relative_to(ROOT))] = [neutral_agent]
    assert agents
    assert agents == expected_agents

    forbidden = (b"/users/", b"/private/", b"docs-private")
    offenders = {}
    for root in (ROOT / "data-src", ROOT / "src", ROOT / "tools"):
        for path in sorted(root.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            contents = path.read_bytes().lower()
            matches = [marker.decode() for marker in forbidden if marker in contents]
            if matches:
                offenders[str(path.relative_to(ROOT))] = matches
    assert offenders == {}


def test_vendored_source_hashes_and_provenance_are_row_complete() -> None:
    provenance = yaml.safe_load(PROVENANCE_PATH.read_text(encoding="utf-8"))
    rows = provenance["rows"]
    gas_records = [row for row in rows if row["table"] == "gas"]
    ion_species = set(_ION_GAS_PROVENANCE_SPECIES)
    gas_rows = {
        row["species_name"]: row
        for row in gas_records
        if row["T_range_K"][0] >= 1500 and row["species_name"] not in ion_species
    }
    low_gas_rows = {
        row["species_name"].removesuffix("(g)"): row
        for row in gas_records
        if row.get("T_interval") == 2 and row["species_name"] not in ion_species
    }
    oxide_rows = {}
    for row in rows:
        if row["table"] != "condensate":
            continue
        current = oxide_rows.get(row["species_name"])
        if current is None or row["T_range_K"][1] > current["T_range_K"][1]:
            oxide_rows[row["species_name"]] = row
    public_gas_channels = set(IMCC_GAS_CHANNEL_SPECIES)
    assert set(gas_rows) == {f"{name}(g)" for name in public_gas_channels}
    assert set(low_gas_rows) == (
        build_gas_tables.LOW_T_GAS_SPECIES
        | {"Na2O", "K2O"}
        | build_gas_tables.TRACE_GAS_SPECIES
    )
    assert len(oxide_rows) == 40

    ion_species = set(_ION_GAS_PROVENANCE_SPECIES)
    ion_records = {}
    for row in gas_records:
        if row["species_name"] in ion_species:
            ion_records.setdefault(row["species_name"], []).append(row)
    assert set(ion_records) == ion_species
    convention = provenance["source_notes"]["janaf"]["ion_reference_convention"]
    assert "0.1 MPa (1 bar)" in convention
    assert "monatomic ideal-gas" in convention
    assert "elemental reference states" in convention
    nasa_ion_species = {
        source[0]
        for source in (
            *build_gas_tables.TRACE_ION_GAS_NASA_SOURCES,
            *build_gas_tables.TRACE_ION_NASA_SOURCES,
        )
    }
    split_ion_species = nasa_ion_species | {
        "Zn+(g)", "Zn-(g)", "Bi+(g)", "Mn+(g)",
        "Mo+(g)", "Mo-(g)", "W+(g)", "W-(g)",
        "Ba+(g)", "Sr+(g)", "Zr+(g)", "Zr-(g)",
        "Sc+(g)",
        "Ni+(g)", "Ni-(g)", "Co+(g)", "Co-(g)",
    }
    nasa_atomic_ion_species = {
        source[0] for source in build_gas_tables.TRACE_ION_GAS_NASA_SOURCES
    }
    for species, records in ion_records.items():
        if species in split_ion_species:
            assert {tuple(row["T_range_K"]) for row in records} == {
                (1500, 3000), (1200, 1500),
            }
        else:
            assert len(records) == 1
            assert records[0]["T_range_K"] == [1200, 3000]
        for row in records:
            source_path = ROOT / row["source_path"]
            assert source_path.is_file()
            assert row["source_sha256"] == hashlib.sha256(
                source_path.read_bytes()
            ).hexdigest()
            assert row["method"] == "fitted"
            if species in nasa_atomic_ion_species:
                assert row["authority"] == "nasa_glenn_fitted_ionisation"
                source_record = _nasa_record(row["table_id"])
                assert row["source"]["source_record"].endswith(row["table_id"])
                locator = source_record["source_locator"]
                expected_locator = (
                    f"thermo.inp lines {locator['name_line']}-{locator['end_line']}"
                    if isinstance(locator, dict)
                    else locator
                )
                assert (
                    row["source"]["source_locator"] == expected_locator
                    or row["source"]["source_locator"] == locator
                )
                assert row["source"]["upstream_source_sha256"] == (
                    "fa7746572952d74e249e818a82a35c113829742fb421a308e167185528884363"
                )
            elif species in nasa_ion_species:
                assert row["authority"] == "nasa_glenn_fitted"
                assert row["source_path"].startswith("data-src/nasa-glenn/")
                assert row["source_locator"].startswith("thermo.inp lines ")
            elif species == "Bi+(g)":
                assert row["authority"] == "nist_partition_fitted_ionisation"
                assert row["source_path"] == (
                    "data-src/published-thermo/bismuth-ion-sources.json"
                )
            else:
                assert row["authority"] == "janaf_fitted_ionisation"
                assert row["source_url"] == (
                    f"https://janaf.nist.gov/tables/{row['table_id']}.html"
                )
                assert row["download_url"] == (
                    f"https://janaf.nist.gov/tables/{row['table_id']}.txt"
                )
                if row["table_id"] in {
                    "Na-007", "K-007", "O-003", "Al-007", "Fe-010",
                    "Si-007", "Ti-008", "Al-076", "Na-009",
                    "Li-006", "Rb-006", "Pb-006", "Ga-006", "B-006",
                }:
                    assert row["retrieval_date"] == "2026-10-01"

    for species, table_id in GAS_TABLE_IDS.items():
        source = _record(table_id)
        row = gas_rows[f"{species}(g)"]
        assert row["table_id"] == table_id
        assert row["source_path"] == f"data-src/janaf/{table_id}.yaml"
        assert (ROOT / row["source_path"]).is_file()
        assert row["source_sha256"] == source["extraction"]["source_sha256"]
        assert row["authority"] == "janaf_fitted"
        assert row["method"] == "fitted"
        expected_t_max = build_gas_tables._DECLARED_T_MAX_BY_TABLE.get(
            table_id,
            build_gas_tables._FIT_T_MAX_BY_TABLE.get(
                table_id, build_gas_tables.FIT_T_MAX
            ),
        )
        assert row["T_range_K"] == [1500, int(expected_t_max)]
        assert source["source"]["doi"] == "10.18434/T42S31"

    for species in (*build_gas_tables.LOW_T_GAS_SPECIES, "Na2O", "K2O"):
        row = low_gas_rows[species]
        assert row["species_name"] == f"{species}(g)"
        assert row["method"] == "fitted"
        assert row["T_range_K"] == [
            int(build_gas_tables.LOW_FIT_T_MIN),
            int(build_gas_tables.LOW_FIT_T_MAX),
        ]
        if species in NASA_TABLE_IDS:
            table_id = NASA_TABLE_IDS[species]
            source_path = NASA_DATA / f"{table_id}.json"
            assert row["table_id"] == table_id
            assert row["source_path"] == f"data-src/nasa-glenn/{table_id}.json"
            assert row["source_sha256"] == hashlib.sha256(
                source_path.read_bytes()
            ).hexdigest()
            assert row["authority"] == "nasa_glenn_fitted"
            assert row["T_interval"] == 2
            assert row["breakpoint_checks"][0]["temperature_K"] == 1000
        else:
            table_id = GAS_TABLE_IDS[species]
            source = _record(table_id)
            assert row["table_id"] == table_id
            assert row["source_path"] == f"data-src/janaf/{table_id}.yaml"
            assert (ROOT / row["source_path"]).is_file()
            assert row["source_sha256"] == source["extraction"]["source_sha256"]
            assert row["authority"] == "janaf_fitted"
            assert source["source"]["doi"] == "10.18434/T42S31"

    for species, table_id in NASA_TABLE_IDS.items():
        source_path = NASA_DATA / f"{table_id}.json"
        source = _nasa_record(table_id)
        row = gas_rows[f"{species}(g)"]
        assert row["table_id"] == table_id
        assert row["source_path"] == f"data-src/nasa-glenn/{table_id}.json"
        assert row["source_sha256"] == hashlib.sha256(source_path.read_bytes()).hexdigest()
        assert row["authority"] == "nasa_glenn_fitted"
        assert row["method"] == "fitted"
        assert row["T_interval"] == 1
        assert row["T_range_K"] == [1500, 3000]
        # The note must name the primary anchor, the 1 atm -> 1 bar entropy
        # conversion, and the open LH84-vs-card certification item.
        for phrase in ("Lamoreaux & Hildenbrand (1984)", "1 atm", "R ln 1.01325", "15 kJ/mol"):
            assert phrase in row["note"], phrase
        assert row["source"]["upstream_source_sha256"] == (
            "fa7746572952d74e249e818a82a35c113829742fb421a308e167185528884363"
        )
        assert source["record_id"] == table_id
        assert source["phase"] == "gas"

    lh84 = yaml.safe_load((LH84_DATA / "lh84.yaml").read_text(encoding="utf-8"))
    assert re.fullmatch(r"[0-9a-f]{64}", lh84["source"]["pdf_sha256"])
    assert lh84["source"]["doi"] == "10.1063/1.555706"
    assert lh84["source"]["verified_by"] == (
        "visual check of PDF page 7 (scanned; no text layer)"
    )
    expected_lh84 = {
        "Na2O(g)": (-3.82, 1.00, 31.35, 0.75, 1.64),
        "K2O(g)": (-7.05, 0.26, 34.17, 0.75, 1.64),
    }
    for row in lh84["rows"]:
        expected = expected_lh84[row["species_name"]]
        assert (
            row["dfH_over_R_kK"]["value"],
            row["dfH_over_R_kK"]["uncertainty"],
            row["S_over_R"]["value"],
            row["S_over_R"]["uncertainty"],
            row["Hinc_over_R_kK"]["value"],
        ) == expected

    for species, table_id in FITTED_CONDENSATE_TABLE_IDS.items():
        source = _record(table_id)
        row = oxide_rows[species]
        assert source["table"]["index_entry"]["state"] == "l"
        assert row["table_id"] == table_id
        assert row["source_path"] == f"data-src/janaf/{table_id}.yaml"
        assert row["source_sha256"] == source["extraction"]["source_sha256"]
        assert row["authority"] == "janaf_fitted"
        assert row["method"] == "fitted"
        expected_t_min = build_gas_tables._FIT_T_MIN_BY_TABLE.get(
            table_id, build_gas_tables.FIT_T_MIN
        )
        runtime_t_min = 1500 if species == "Na2O(l)" else int(expected_t_min)
        assert row["T_range_K"] == [runtime_t_min, 3000]
        assert source["source"]["doi"] == "10.18434/T42S31"

    na_high = oxide_rows["Na2O(l)"]
    assert na_high["source_fit_range_K"] == [1500, 3000]
    assert na_high["supersedes"]["T_range_K"] == [1405, 3000]
    assert na_high["supersedes"]["dH298_R"] == -50.17
    assert na_high["supersedes"]["dG_coefficients"] == [7.67, 6.193, 0, 0, 0]
    assert na_high["supersedes"]["reason"].startswith("Replaces the LH84 Table 2 Na2O(l)")
    assert na_high["supersedes"]["measured_LH84_minus_JANAF_G_kJ_mol"][
        "values"
    ] == {1600: 13.44, 2000: 25.57, 2400: 29.64, 3000: 17.66}

    nb_low = next(
        row
        for row in rows
        if row["table"] == "condensate"
        and row["species_name"] == "NbO2(l)"
        and row["T_range_K"] == [1200, 1500]
    )
    nb_source = _record("Nb-013")
    assert nb_low["source_path"] == "data-src/janaf/Nb-013.yaml"
    assert nb_low["source_sha256"] == nb_source["extraction"]["source_sha256"]
    assert nb_low["source_fit_range_K"] == [1100, 1500]
    assert nb_low["authority"] == "janaf_fitted"
    assert "liquid Cp" in nb_low["note"]

    na_low = next(
        row
        for row in rows
        if row["table"] == "condensate"
        and row["species_name"] == "Na2O(l)"
        and row["T_range_K"] == [1200, 1500]
    )
    na_source = _record("Na-013")
    assert na_low["source_path"] == "data-src/janaf/Na-013.yaml"
    assert na_low["source_sha256"] == na_source["extraction"]["source_sha256"]
    assert na_low["source_fit_range_K"] == [1000, 1500]
    assert na_low["authority"] == "janaf_fitted"
    assert "Cp = 104.600" in na_low["note"]

    for species in (
        "Ti",
        "TiO",
        "TiO2",
        "Cr",
        "CrO",
        "CrO2",
        "CrO3",
        "V",
        "VO",
        "VO2",
        "Nb",
        "NbO",
        "NbO2",
        "Mn",
        "Ni",
        "Co",
    ):
        assert _record(GAS_TABLE_IDS[species])["table"]["index_entry"]["state"] == "g"

    k_row = oxide_rows["K2O(l)"]
    assert k_row["authority"] == "estimated_systematic"
    assert k_row["method_class"] == "estimated_systematic"
    assert k_row["method_rule"] == "liquid_from_solid"
    assert k_row["source_table_id"] == "K-012"
    assert k_row["source_sha256"] == _record("K-012")["extraction"]["source_sha256"]
    assert k_row["inputs"]["fusion_temperature"]["band_K"] == [919, 1190]
    assert k_row["inputs"]["liquid_heat_capacity"]["band_includes_J_per_mol_K"] == [104.6]
    assert k_row["combined_band"]["sample_temperatures_K"] == [1013, 1313, 1813]


def test_trace_rows_have_complete_source_hashes_and_fit_intervals() -> None:
    provenance = yaml.safe_load(PROVENANCE_PATH.read_text(encoding="utf-8"))
    rows = provenance["rows"]
    gas_records = [row for row in rows if row["table"] == "gas"]

    for source in build_gas_tables.TRACE_JANAF_GAS_SOURCES:
        species_name, table_id, *_ = source
        source_path = JANAF_DATA / f"{table_id}.txt"
        expected_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
        for interval, runtime_range, source_range in (
            (1, [1500, 3000], [1500, 3000]),
            (2, [1200, 1500], [500, 1500]),
        ):
            record = next(
                row for row in gas_records
                if row["species_name"] == species_name
                and row.get("T_interval") == interval
            )
            assert record["table_id"] == table_id
            assert record["source_path"] == f"data-src/janaf/{table_id}.txt"
            assert record["source_sha256"] == expected_sha
            assert record["authority"] == "janaf_fitted"
            assert record["method"] == "fitted"
            assert record["T_range_K"] == runtime_range
            assert record.get("source_fit_range_K", runtime_range) == source_range
            assert record["source_url"] == (
                f"https://janaf.nist.gov/tables/{table_id}.html"
            )
            assert record["download_url"] == (
                f"https://janaf.nist.gov/tables/{table_id}.txt"
            )

    for species_name, table_id, *_ in build_gas_tables.TRACE_JANAF_YAML_GAS_SOURCES:
        source_path = JANAF_DATA / f"{table_id}.yaml"
        source_record = _record(table_id)
        expected_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
        for interval, runtime_range, source_range in (
            (1, [1500, 3000], [1500, 3000]),
            (2, [1200, 1500], [500, 1500]),
        ):
            record = next(
                row for row in gas_records
                if row["species_name"] == species_name
                and row.get("T_interval") == interval
            )
            assert record["table_id"] == table_id
            assert record["source_path"] == f"data-src/janaf/{table_id}.yaml"
            assert record["source_sha256"] == expected_sha
            assert record["upstream_source_sha256"] == source_record[
                "extraction"
            ]["source_sha256"]
            assert record["authority"] == "janaf_fitted"
            assert record["T_range_K"] == runtime_range
            assert record["source_fit_range_K"] == source_range

    for source in (
        *build_gas_tables.TRACE_NASA_GAS_SOURCES,
        *build_gas_tables.TRACE_ION_GAS_NASA_SOURCES,
        *build_gas_tables.TRACE_ION_NASA_SOURCES,
    ):
        species_name, table_id, *_ = source
        source_path = NASA_DATA / f"{table_id}.json"
        expected_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
        nasa_source = _nasa_record(table_id)
        assert nasa_source["record_id"] == table_id
        assert nasa_source["phase"] == "gas"
        source_row = next(
            row
            for row in gas_records
            if row["species_name"] == species_name
            and row.get("T_interval") == 1
        )
        assert source_row.get(
            "upstream_source_sha256",
            source_row.get("source", {}).get("upstream_source_sha256"),
        ) == (
            "fa7746572952d74e249e818a82a35c113829742fb421a308e167185528884363"
        )
        for interval, runtime_range, source_range in (
            (1, [1500, 3000], [1500, 3000]),
            (2, [1200, 1500], [1200, 1500] if species_name == "NiO(g)" else [500, 1500]),
        ):
            record = next(
                row for row in gas_records
                if row["species_name"] == species_name
                and row.get("T_interval") == interval
            )
            assert record["table_id"] == table_id
            assert record["source_path"] == f"data-src/nasa-glenn/{table_id}.json"
            assert record["source_sha256"] == expected_sha
            assert record["authority"] == (
                "nasa_glenn_fitted_ionisation"
                if source in build_gas_tables.TRACE_ION_GAS_NASA_SOURCES
                else "nasa_glenn_fitted"
            )
            assert record["method"] == "fitted"
            assert record["T_range_K"] == runtime_range
            assert record.get("source_fit_range_K", runtime_range) == source_range

    condensate_records = [row for row in rows if row["table"] == "condensate"]
    for record in condensate_records:
        species_name = record["species_name"]
        if species_name not in {
            "Li2O(l)", "Rb2O(l)", "PbO(l)",
            "Cs2O(l)", "Cu2O(l)", "SnO(l)",
            "ZnO(l)", "CdO(l)", "Bi2O3(l)",
            "MnO(l)",
        }:
            continue
        source_path = ROOT / record["source_path"]
        assert record["source_sha256"] == hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest()
        assert record["authority"] in {
            "janaf_fitted", "nasa_glenn_fitted", "published_fitted",
            "liquid_from_solid_estimate",
        }
        assert record["method"] in {
            "fitted",
            "generated_constant_cp_extrapolation_fit",
            "janaf_anchored_nasa_tail_fit",
            "liquid_from_solid_construction",
        }
        if species_name == "PbO(l)" and record["T_range_K"] == [1500, 3000]:
            tail_path = ROOT / record["tail_source_path"]
            assert record["tail_source_sha256"] == hashlib.sha256(
                tail_path.read_bytes()
            ).hexdigest()
        if species_name == "Cu2O(l)" and "tail_source_path" in record:
            tail_path = ROOT / record["tail_source_path"]
            assert record["tail_source_sha256"] == hashlib.sha256(
                tail_path.read_bytes()
            ).hexdigest()


def test_trace_gas_fits_match_every_declared_source_node() -> None:
    provenance = yaml.safe_load(PROVENANCE_PATH.read_text(encoding="utf-8"))
    rows = {
        (row["species_name"], row.get("T_interval", 1)): row
        for row in provenance["rows"]
        if row["table"] == "gas"
    }
    pack = load_gas_datapack()

    for species_name, table_id, *_ in build_gas_tables.TRACE_JANAF_GAS_SOURCES:
        source_rows = build_gas_tables._janaf_text_source_rows(JANAF_DATA, table_id)
        reference = source_rows[298.15]["formation_enthalpy"]
        for interval, t_min, t_max, source_min in (
            (1, 1500.0, 3000.0, 1500.0),
            (2, 1200.0, 1500.0, 500.0),
        ):
            row = _gas_row_for_interval(pack, species_name.removesuffix("(g)"), interval)
            residuals = []
            log_residuals = []
            for temperature, source in source_rows.items():
                if not source_min <= temperature <= t_max:
                    continue
                # JANAF Hf(298) + [H(T)-H(298)] - T*S gives J/mol after
                # converting its two enthalpy columns from kJ/mol.
                source_g = (
                    (reference + source["enthalpy_increment"]) * 1000.0
                    - temperature * source["entropy"]
                )
                residual = _janaf_gibbs(temperature, row) - source_g
                residuals.append(abs(residual))
                log_residuals.append(
                    abs(residual)
                    / (R_J_MOL_K * temperature * math.log(10.0))
                )
            record = rows[(species_name, interval)]
            assert residuals
            assert max(residuals) == pytest.approx(
                record["max_residual_J_per_mol"], abs=1.0e-7
            )
            assert max(log_residuals) == pytest.approx(
                record["max_residual_log10_K"], abs=1.0e-10
            )

    for species_name, table_id, *_ in (
        *build_gas_tables.TRACE_NASA_GAS_SOURCES,
        *build_gas_tables.TRACE_ION_GAS_NASA_SOURCES,
        *build_gas_tables.TRACE_ION_NASA_SOURCES,
    ):
        record = _nasa_record(table_id)
        for interval, t_min, t_max in (
            (1, 1500.0, 3000.0),
            (2, 1200.0 if species_name == "NiO(g)" else 500.0, 1500.0),
        ):
            row = _gas_row_for_interval(pack, species_name.removesuffix("(g)"), interval)
            residuals = []
            log_residuals = []
            at_298 = build_gas_tables._nasa7_properties(record, 298.15)
            reference = float(record["delta_f_H_298_15"]["value"])
            for temperature in np.arange(t_min, t_max + 50.0, 100.0):
                properties = build_gas_tables._nasa7_properties(
                    record, float(temperature)
                )
                enthalpy = reference + R_J_MOL_K * (
                    float(temperature) * properties["h_rt"]
                    - 298.15 * at_298["h_rt"]
                )
                entropy = R_J_MOL_K * properties["s_R"]
                source_g = enthalpy - float(temperature) * entropy
                residual = _janaf_gibbs(float(temperature), row) - source_g
                residuals.append(abs(residual))
                log_residuals.append(
                    abs(residual)
                    / (R_J_MOL_K * float(temperature) * math.log(10.0))
                )
            provenance_row = rows[(species_name, interval)]
            assert max(residuals) == pytest.approx(
                provenance_row["max_residual_J_per_mol"],
                abs=1.0e-7,
            ), f"{species_name} interval {interval}"
            assert max(log_residuals) == pytest.approx(
                provenance_row["max_residual_log10_K"], abs=1.0e-10
            )



    for species_name, table_id, *_ in build_gas_tables.TRACE_JANAF_YAML_GAS_SOURCES:
        source_values = _record(table_id)["table"]["values"]
        for interval, source_min, source_max in (
            (1, 1500.0, 3000.0),
            (2, 500.0, 1500.0),
        ):
            row = _gas_row_for_interval(
                pack, species_name.removesuffix("(g)"), interval
            )
            residuals = [
                abs(
                    _janaf_gibbs(float(value["temperature"]["value"]), row)
                    - _janaf_apparent_gibbs(
                        table_id, float(value["temperature"]["value"])
                    )
                )
                for value in source_values
                if source_min <= float(value["temperature"]["value"]) <= source_max
            ]
            record = rows[(species_name, interval)]
            assert residuals
            assert max(residuals) == pytest.approx(
                record["max_residual_J_per_mol"], abs=1.0e-6
            )

    published_dir = ROOT / "data-src" / "published-thermo"
    for species_name, table_id, *_rest in build_gas_tables.TRACE_PUBLISHED_GAS_SOURCES:
        source_limit = _rest[-1]
        split = (
            min(max(source_limit, 1500.0), 2000.0)
            if table_id.startswith(("P672-Te", "P672-Se", "P672-As", "P672-Sb"))
            else 1500.0
        )
        high_fit_start = math.floor(split / 100.0) * 100.0
        for interval, t_min, t_max, side in (
            (1, high_fit_start, 3000.0, "high"),
            (2, 1200.0, split, "low"),
        ):
            source_rows, reference, _source_limit = (
                build_gas_tables._published_gas_source_rows(
                    published_dir, table_id, breakpoint_side=side
                )
            )
            source_rows = build_gas_tables._constant_cp_rows_through(
                source_rows, t_max
            )
            row = _gas_row_for_interval(
                pack, species_name.removesuffix("(g)"), interval
            )
            residuals = []
            log_residuals = []
            for temperature in np.arange(t_min, t_max + 50.0, 100.0):
                source = source_rows[float(temperature)]
                source_g = (
                    reference + source["enthalpy_increment"]
                ) * 1000.0 - float(temperature) * source["entropy"]
                residual = _janaf_gibbs(float(temperature), row) - source_g
                residuals.append(abs(residual))
                log_residuals.append(
                    abs(residual)
                    / (R_J_MOL_K * float(temperature) * math.log(10.0))
                )
            provenance_row = rows[(species_name, interval)]
            assert max(residuals) == pytest.approx(
                provenance_row["max_residual_J_per_mol"], abs=1.0e-7
            ), f"published {species_name} interval {interval}"
            assert max(log_residuals) == pytest.approx(
                provenance_row["max_residual_log10_K"], abs=1.0e-10
            )


def test_trace_continuation_bands_are_truncated_at_each_source_limit() -> None:
    provenance = yaml.safe_load(PROVENANCE_PATH.read_text(encoding="utf-8"))
    nasa_cards = sorted(
        path
        for path in (ROOT / "data-src" / "nasa-glenn").glob("NG-*.json")
        if path.stem not in {
            # NG-0074/0085/0086 are the Ag atom/ion cards; the other listed
            # records are the Mo/W channel cards. These added channels are not
            # inputs to the frozen envelope made from the original 44 cards.
            "NG-0074", "NG-0085", "NG-0086",
            "NG-0824", "NG-0825", "NG-0826", "NG-0827", "NG-0828",
            "NG-0829", "NG-0831", "NG-0832", "NG-0833", "NG-0834",
            "NG-1245", "NG-1246", "NG-1247", "NG-1249", "NG-1251",
            "NG-1253", "NG-1255", "NG-1256", "NG-1257", "NG-1258",
            "NG-1701", "NG-2013",
            "NG-0165", "NG-0166", "NG-0178", "NG-1168", "NG-1169",
            "NG-1181", "NG-1263", "NG-1264", "NG-1267", "NG-1269",
            "NG-1325", "NG-1917", "NG-2029",
            "NG-0926",
        "NG-1087", "NG-1088", "NG-1090", "NG-1092", "NG-1206",
        "NG-1224", "NG-1230", "NG-1235", "NG-1856", "NG-1857",
        "NG-1976", "NG-1977",
        }
    )
    expected_by_limit = {}
    for limit in (1000.0, 1200.0, 1500.0, 1860.0, 2000.0):
        maximum = {}
        counts = {}
        for card_path in nasa_cards:
            record = build_gas_tables._load_record(card_path)
            try:
                at_limit = build_gas_tables._nasa7_properties(record, limit)
            except (KeyError, ValueError):
                continue
            cp = R_J_MOL_K * at_limit["cp_R"]
            enthalpy = R_J_MOL_K * limit * at_limit["h_rt"]
            entropy = R_J_MOL_K * at_limit["s_R"]
            for temperature in (1200, 1500, 1800, 2000, 2500, 3000):
                if temperature <= limit:
                    continue
                try:
                    actual = build_gas_tables._nasa7_properties(record, temperature)
                except (KeyError, ValueError):
                    continue
                if at_limit.get("region") != actual.get("region"):
                    continue
                held_g = (
                    enthalpy
                    + cp * (temperature - limit)
                    - temperature
                    * (entropy + cp * math.log(temperature / limit))
                )
                actual_g = R_J_MOL_K * temperature * (
                    actual["h_rt"] - actual["s_R"]
                )
                maximum[temperature] = max(
                    maximum.get(temperature, 0.0), abs(held_g - actual_g)
                )
                counts[temperature] = counts.get(temperature, 0) + 1
        assert all(count == 44 for count in counts.values())
        expected_by_limit[limit] = maximum

    # Truncate each published card's Cp function at the row's own source
    # limit, hold Cp there, and compare G(T). Zero belongs only at the limit.
    for row in provenance["rows"]:
        band = row.get("uncertainty_band")
        if (
            row["table"] not in {"gas", "condensate"}
            or band is None
            or "44 published NASA Glenn cards" not in band.get("method", "")
        ):
            continue
        limit = float(row["extrapolation_source_limit_K"])
        assert limit in expected_by_limit
        values = band["values_J_per_mol"]
        assert values[int(limit)] == 0.0
        assert set(values) == {int(limit), *expected_by_limit[limit]}
        for temperature, expected in expected_by_limit[limit].items():
            assert values[temperature] == pytest.approx(expected, abs=0.002)


def test_published_source_data_flags_follow_source_limits() -> None:
    provenance = yaml.safe_load(PROVENANCE_PATH.read_text(encoding="utf-8"))
    for row in provenance["rows"]:
        limit = row.get("extrapolation_source_limit_K")
        bounds = row.get("T_range_K")
        if (
            not row.get("source_path", "").startswith("data-src/published-thermo/")
            or limit is None
            or bounds is None
            or float(bounds[0]) < float(limit)
        ):
            continue
        assert row["source_data"] is False, (
            f"{row['species_name']} interval {bounds} starts at {bounds[0]} K, "
            f"at or above its {limit} K source limit"
        )


def test_trace_nasa_card_thermal_increments_are_zero_at_the_formation_anchor() -> None:
    for _species_name, table_id, *_ in build_gas_tables.TRACE_NASA_GAS_SOURCES:
        record = _nasa_record(table_id)
        thermal = build_gas_tables._nasa_card_thermal_rows(record, [298.15])
        assert thermal[0]["enthalpy_increment"] == pytest.approx(0.0, abs=1.0e-12)


def test_trace_condensate_fits_match_janaf_and_nasa_source_nodes() -> None:
    provenance = yaml.safe_load(PROVENANCE_PATH.read_text(encoding="utf-8"))
    rows = [row for row in provenance["rows"] if row["table"] == "condensate"]
    pack = load_gas_datapack()

    def fitted_row(species_name: str, t_min: float) -> pd.Series:
        matches = pack.oxide_df.loc[
            (pack.oxide_df.index == species_name)
            & (pack.oxide_df["T_min"].astype(float) == t_min)
        ]
        assert len(matches) == 1
        return matches.iloc[0]

    def source_janaf_g(table_id: str, temperature: float) -> float:
        if (JANAF_DATA / f"{table_id}.yaml").is_file():
            return _janaf_apparent_gibbs(table_id, temperature)
        source_rows = build_gas_tables._janaf_text_source_rows(JANAF_DATA, table_id)
        reference = source_rows[298.15]["formation_enthalpy"]
        source = source_rows[temperature]
        return (
            (reference + source["enthalpy_increment"]) * 1000.0
            - temperature * source["entropy"]
        )

    def assert_residuals(
        species_name: str,
        row: pd.Series,
        source_nodes: list[tuple[float, float]],
    ) -> float:
        residual = max(
            abs(_lamor_gibbs(temperature, row) - source_g)
            for temperature, source_g in source_nodes
        )
        provenance_row = next(
            item
            for item in rows
            if item["species_name"] == species_name
            and item["T_range_K"]
            == [int(float(row["T_min"])), int(float(row["T_max"]))]
        )
        assert residual == pytest.approx(
            provenance_row["max_residual_J_per_mol"], abs=1.0e-6
        )
        return residual

    li_source = {row["temperature"]: row for row in _complete_rows("Li-015")}
    for t_min, t_max in ((1200.0, 1400.0), (1400.0, 2000.0), (2000.0, 3000.0)):
        li_nodes = [
            (temperature, source_janaf_g("Li-015", temperature))
            for temperature in li_source
            if max(700.0, t_min) <= temperature <= t_max
        ]
        assert_residuals("Li2O(l)", fitted_row("Li2O(l)", t_min), li_nodes)

    b_source = {row["temperature"]: row for row in _complete_rows("B-096")}
    for t_min, t_max, source_min in (
        (1200.0, 1500.0, 800.0),
        (1500.0, 3000.0, 1500.0),
    ):
        nodes = [
            (temperature, source_janaf_g("B-096", temperature))
            for temperature in b_source
            if source_min <= temperature <= t_max
        ]
        assert_residuals("B2O3(l)", fitted_row("B2O3(l)", t_min), nodes)

    te_source, te_reference, _te_limit = build_gas_tables._published_gas_source_rows(
        PUBLISHED_DATA, "P672-TeO2-l"
    )
    te_source = build_gas_tables._constant_cp_rows_through(te_source, 3000.0)
    for t_min, t_max in ((1200.0, 1500.0), (1500.0, 3000.0)):
        nodes = []
        for temperature, source in te_source.items():
            if t_min <= temperature <= t_max:
                source_g = (
                    te_reference + source["enthalpy_increment"]
                ) * 1000.0 - temperature * source["entropy"]
                nodes.append((temperature, source_g))
        assert assert_residuals(
            "TeO2(l)", fitted_row("TeO2(l)", t_min), nodes
        ) < 10.0

    se_config = json.loads(
        (ROOT / "data-src" / "liquid-from-solid-inputs.json").read_text(
            encoding="utf-8"
        )
    )["rows"]["SeO2"]
    _se_rows, se_construction, _se_alternate = (
        build_gas_tables._fit_liquid_from_solid_parent(
            PUBLISHED_DATA, "SeO2", se_config
        )
    )
    for t_min, t_max in ((1200.0, 1600.0), (1600.0, 3000.0)):
        nodes = [
            (float(t), se_construction.at(float(t)).gibbs_kj_mol * 1000.0)
            for t in np.arange(t_min, t_max + 100.0, 100.0)
        ]
        assert assert_residuals(
            "SeO2(l)", fitted_row("SeO2(l)", t_min), nodes
        ) < 10.0

    as_config = json.loads(
        (ROOT / "data-src" / "liquid-from-solid-inputs.json").read_text(
            encoding="utf-8"
        )
    )["rows"]["As2O3"]
    _as_rows, as_construction, _as_alternate = (
        build_gas_tables._fit_liquid_from_solid_parent(
            PUBLISHED_DATA, "As2O3", as_config
        )
    )
    for t_min, t_max in ((1200.0, 1600.0), (1600.0, 3000.0)):
        nodes = [
            (float(t), as_construction.at(float(t)).gibbs_kj_mol * 1000.0)
            for t in np.arange(t_min, t_max + 100.0, 100.0)
        ]
        assert assert_residuals(
            "As2O3(l)", fitted_row("As2O3(l)", t_min), nodes
        ) < 10.0

    rb_record = _nasa_record("NG-1841")
    for t_min, t_max in (
        (1200.0, 1500.0),
        (1500.0, 2000.0),
        (2000.0, 3000.0),
    ):
        card_nodes = []
        for temperature in np.arange(t_min, t_max + 50.0, 100.0):
            properties = build_gas_tables._nasa7_properties(
                rb_record, float(temperature)
            )
            source_g = R_J_MOL_K * float(temperature) * (
                properties["h_rt"] - properties["s_R"]
            )
            card_nodes.append((float(temperature), source_g))
        assert_residuals(
            "Rb2O(l)", fitted_row("Rb2O(l)", t_min), card_nodes
        )

    pb_low_nodes = [
        (temperature, source_janaf_g("O-007", temperature))
        for temperature in {
            row["temperature"] for row in _complete_rows("O-007")
        }
        if 1200.0 <= temperature <= 1500.0
    ]
    assert_residuals("PbO(l)", fitted_row("PbO(l)", 1200.0), pb_low_nodes)

    pb_high_nodes = [
        (temperature, source_janaf_g("O-007", temperature))
        for temperature in {
            row["temperature"] for row in _complete_rows("O-007")
        }
        if 1500.0 <= temperature <= 2500.0
    ]
    pb_record = _nasa_record("NG-1801")
    janaf_rows = build_gas_tables._janaf_text_source_rows(JANAF_DATA, "O-007")
    anchor = janaf_rows[2500.0]
    nasa_anchor = build_gas_tables._nasa7_properties(pb_record, 2500.0)
    for temperature in np.arange(2600.0, 3000.0 + 50.0, 100.0):
        properties = build_gas_tables._nasa7_properties(
            pb_record, float(temperature)
        )
        enthalpy_increment = anchor["enthalpy_increment"] + (
            R_J_MOL_K
            * (
                float(temperature) * properties["h_rt"]
                - 2500.0 * nasa_anchor["h_rt"]
            )
            / 1000.0
        )
        entropy = anchor["entropy"] + R_J_MOL_K * (
            properties["s_R"] - nasa_anchor["s_R"]
        )
        reference = janaf_rows[298.15]["formation_enthalpy"]
        source_g = (
            (reference + enthalpy_increment) * 1000.0
            - float(temperature) * entropy
        )
        pb_high_nodes.append((float(temperature), source_g))
    assert_residuals("PbO(l)", fitted_row("PbO(l)", 1500.0), pb_high_nodes)

    for table_id, species_name in (("Mo-015", "MoO3(l)"), ("O-066", "WO3(l)")):
        source_values = _record(table_id)["table"]["values"]
        complete_source = {
            row["temperature"]: row for row in _complete_rows(table_id)
        }
        if table_id == "Mo-015":
            complete_source = build_gas_tables._constant_cp_rows_through(
                complete_source, 3000.0
            )
        source_anchor = next(
            float(value["formation_enthalpy"]["value"])
            for value in source_values
            if float(value["temperature"]["value"]) == 298.15
        )
        condensate_rows = load_gas_datapack().oxide_df
        assert all(
            float(row["dH298_R"]) * R_J_MOL_K
            == pytest.approx(source_anchor, abs=1.0e-12)
            for _, row in condensate_rows.loc[
                condensate_rows.index == species_name
            ].iterrows()
        )
        for provenance_row in rows:
            if provenance_row["species_name"] != species_name:
                continue
            t_min, t_max = map(float, provenance_row["T_range_K"])
            nodes = [
                (
                    temperature,
                    (source_anchor + value["enthalpy_increment"]) * 1000.0
                    - temperature * value["entropy"],
                )
                for temperature, value in complete_source.items()
                if t_min <= temperature <= t_max
            ]
            assert assert_residuals(
                species_name, fitted_row(species_name, t_min), nodes
            ) < 10.0

    for (
        species_name,
        table_id,
        _formula,
        _cation,
        _cat_num,
        _oxy_num,
        _source_t_min,
        _continuation_intervals,
        fit_intervals,
    ) in build_gas_tables.TRACE_NASA_CONDENSATE_SOURCES:
        record = _nasa_record(table_id)
        for t_min, t_max in fit_intervals:
            card_nodes = []
            for temperature in np.arange(t_min, t_max + 50.0, 100.0):
                properties = build_gas_tables._nasa7_properties(
                    record, float(temperature)
                )
                card_gibbs = R_J_MOL_K * float(temperature) * (
                    properties["h_rt"] - properties["s_R"]
                )
                card_nodes.append((float(temperature), card_gibbs))
            assert_residuals(
                species_name,
                fitted_row(species_name, t_min),
                card_nodes,
            )
    sn_record = _nasa_record("NG-1848")
    sn_nodes = []
    for temperature in np.arange(1250.0, 2000.0, 100.0):
        properties = build_gas_tables._nasa7_properties(
            sn_record, float(temperature)
        )
        source_g = R_J_MOL_K * float(temperature) * (
            properties["h_rt"] - properties["s_R"]
        )
        sn_nodes.append((float(temperature), source_g))
    assert_residuals("SnO(l)", fitted_row("SnO(l)", 1250.0), sn_nodes)


def test_pankratz_liquid_construction_inputs_and_band_are_independent() -> None:
    inputs = json.loads(
        (ROOT / "data-src" / "liquid-from-solid-inputs.json").read_text(
            encoding="utf-8"
        )
    )["rows"]
    source = build_gas_tables._published_table_record(
        PUBLISHED_DATA, "P672-SeO2-c"
    )
    se = inputs["SeO2"]
    fusion_t = se["inputs"]["fusion_temperature"]["value"]
    fusion_h = se["inputs"]["fusion_entropy"]["value"] * fusion_t
    assert fusion_t == 660.0
    assert fusion_h == pytest.approx(22000.0, abs=1e-9)
    assert se["inputs"]["fusion_entropy"]["spread"] == pytest.approx(
        5000.0 / 660.0
    )
    assert "calculated triple point" in se["inputs"]["fusion_temperature"]["source"]
    assert "662.5 +/- 0.5 K" in se["inputs"]["fusion_temperature"]["source"]
    assert source["formation_enthalpy_298_kcal_mol"] == -53.860
    source_602 = next(row for row in source["nodes"] if row[0] == 602)
    assert source_602 == [602, 18.776, 27.358, 5.030]
    cp_equation = source["cp_equation_cal_mol_K"]
    cp_660 = (
        cp_equation["a"]
        + cp_equation["b_per_K"] * 660.0
        - cp_equation["c_K2"] / 660.0**2
    )
    assert cp_660 * 4.184 == pytest.approx(
        se["inputs"]["liquid_heat_capacity"]["value"], abs=1e-9
    )

    arsenic = inputs["As2O3"]
    as_source = build_gas_tables._published_table_record(
        PUBLISHED_DATA, "P672-As2O3-c"
    )
    assert as_source["formation_enthalpy_298_kcal_mol"] == -157.020
    as_melt = next(row for row in as_source["nodes"] if row[0] == 551)
    assert as_melt == [551, 29.732, 41.902, 6.766]
    as_cp = as_source["cp_equation_cal_mol_K"]
    as_equation_cp = (
        as_cp["a"]
        + as_cp["b_per_K"] * 551.0
        - as_cp["c_K2"] / 551.0**2
    )
    assert as_equation_cp == pytest.approx(29.716220235, abs=1e-9)
    assert arsenic["inputs"]["fusion_temperature"]["value"] == 551.0
    assert arsenic["inputs"]["liquid_heat_capacity"]["value"] == pytest.approx(
        as_equation_cp * 4.184, abs=1e-9
    )

    _rows, construction, held_cp = build_gas_tables._fit_liquid_from_solid_parent(
        PUBLISHED_DATA, "SeO2", se
    )
    provenance = yaml.safe_load(PROVENANCE_PATH.read_text(encoding="utf-8"))
    band = next(
        row["uncertainty_band"]["values_J_per_mol"]
        for row in provenance["rows"]
        if row["table"] == "condensate"
        and row["species_name"] == "SeO2(l)"
    )
    for temperature, expected in band.items():
        temperature = float(temperature)
        # At 2000 K the entropy term is |(660-2000) K| *
        # (5000 J/mol / 660 K) = 10151.515 J/mol. The heat-capacity term is
        # zero because the fallback has no stated Cp spread. The bounded
        # fusion-temperature endpoint term is 341.553 J/mol, evaluated from
        # the source crystal state. Finally the 602->660 K crystal-Cp model
        # spread contributes |G(source Cp)-G(held Cp)| = 234.011 J/mol.
        # Sum: 10151.515 + 341.553 + 234.011 = 10727.080 J/mol at 2000 K.
        propagated = construction.input_spread_kj_mol(temperature) * 1000.0
        crystal_extension = abs(
            construction.at(temperature).gibbs_kj_mol
            - held_cp.at(temperature).gibbs_kj_mol
        ) * 1000.0
        assert propagated + crystal_extension == pytest.approx(
            expected, abs=0.002
        )
    assert band[1200] > 0.0
    assert band[2000] == pytest.approx(10727.079668, abs=0.002)


def test_constructed_parent_fit_nodes_are_only_self_fit_checks() -> None:
    rows = build_gas_tables.build_condensate_rows(JANAF_DATA)
    selected = [
        row
        for row in rows
        if row["species_name"] in {
            "Li2O(l)", "Rb2O(l)", "PbO(l)", "B2O3(l)",
            "Ga2O3(l)", "GeO2(l)", "In2O3(l)",
            "Li2O(l)", "Rb2O(l)", "PbO(l)",
            "Cs2O(l)", "Cu2O(l)", "SnO(l)", "K2O(l)",
            "ZnO(l)", "CdO(l)", "Bi2O3(l)",
            "MnO(l)", "TeO2(l)", "Sb2O3(l)", "Ag2O(l)",
        }
    ]
    direct_fits = [
        row for row in selected
        if not row["Ref"].endswith(build_gas_tables.SUPERCOOLED_LIQUID_REF_SUFFIX)
    ]
    assert direct_fits
    assert all(float(row["_max_residual_J_per_mol"]) < 10.0 for row in direct_fits)


def test_mno_liquid_fit_matches_the_redox_construction_and_band() -> None:
    inputs_path = ROOT / "data-src" / "liquid-from-solid-inputs.json"
    inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    redox_path = ROOT / "src" / "openimcc" / "data" / "packs" / (
        "openimcc-redox-v1.json"
    )
    redox = json.loads(redox_path.read_text(encoding="utf-8"))
    config = inputs["rows"]["MnO"]
    rows, construction, alternate = build_gas_tables._fit_liquid_from_solid_parent(
        ROOT / "data-src" / "published-thermo", "MnO", config
    )
    redox_species = redox["thermo_species"]["MnO(l)"]
    redox_intervals = redox_species["intervals"]
    redox_construction = redox_species["construction"]
    middle_index = next(
        index
        for index, row in enumerate(rows)
        if (float(row["T_min"]), float(row["T_max"])) == (1600.0, 2200.0)
    )
    middle_gas_row = rows[middle_index]
    middle_redox_row = redox_intervals[middle_index]["row"]
    residual = float(middle_gas_row["_max_residual_J_per_mol"])

    assert alternate is None
    assert construction.inputs == config["inputs"]
    assert redox_construction["crystal_source_table"] == config["crystal_table_id"]
    assert redox_construction["fusion_temperature_K"] == config["inputs"][
        "fusion_temperature"
    ]["value"]
    assert redox_construction["fusion_temperature_spread_K"] == config["inputs"][
        "fusion_temperature"
    ]["spread"]
    assert redox_construction["fusion_entropy_j_mol_K"] == config["inputs"][
        "fusion_entropy"
    ]["value"]
    assert redox_construction[
        "fusion_entropy_spread_j_mol_K"
    ] == config["inputs"]["fusion_entropy"]["spread"]
    assert redox_construction["liquid_heat_capacity_j_mol_K"] == config["inputs"][
        "liquid_heat_capacity"
    ]["value"]
    assert redox_construction[
        "liquid_heat_capacity_spread_j_mol_K"
    ] == config["inputs"]["liquid_heat_capacity"]["spread"]
    assert [(float(row["T_min"]), float(row["T_max"])) for row in rows] == [
        (1200.0, 1600.0), (1600.0, 2200.0), (2200.0, 3000.0),
    ]
    assert [tuple(interval["T_domain_K"]) for interval in redox_intervals] == [
        (1200, 1600), (1600, 2200), (2200, 3000),
    ]
    assert all(row["species_name"] == "MnO(l)" for row in rows)
    assert all(row["Ref"] == "LFS[MnO-IVTAN-1435;Tf=2120]" for row in rows)
    assert residual == pytest.approx(0.186390915536322, abs=1e-12)
    assert max(float(row["_max_residual_J_per_mol"]) for row in rows) < 10.0
    for gas_row, redox_interval in zip(rows, redox_intervals):
        assert all(
            float(gas_row[key]) == pytest.approx(
                float(redox_interval["row"][key]), abs=1e-12
            )
            for key in ("dH298_R", "dG_A", "dG_B", "dG_C", "dG_D", "dG_E")
        )
    assert config["inputs"]["condensate_fit"] == {
        "fit_t_min": 1600.0,
        "fit_t_max": 2200.0,
        "runtime_t_min": 1673.0,
        "runtime_t_max": 2000.0,
        "table_id": "MnO-liquid-construction",
        "cation": "Mn",
        "cat_num": 1,
        "oxy_num": 1,
        "ref": "MnO crystal function plus systematic liquid construction",
    }
    provenance = yaml.safe_load(PROVENANCE_PATH.read_text(encoding="utf-8"))
    condensate_provenance = next(
        item
        for item in provenance["rows"]
        if item["table"] == "condensate" and item["species_name"] == "MnO(l)"
    )
    inputs_digest = hashlib.sha256(inputs_path.read_bytes()).hexdigest()
    assert condensate_provenance["construction_inputs_path"] == (
        "data-src/liquid-from-solid-inputs.json"
    )
    assert condensate_provenance["construction_inputs_sha256"] == inputs_digest
    assert condensate_provenance["T_range_K"] == [1200, 3000]
    assert condensate_provenance["source_fit_range_K"] == [1200, 3000]
    assert condensate_provenance["fit_intervals_K"] == [
        [1200, 1600], [1600, 2200], [2200, 3000],
    ]
    assert condensate_provenance["source"]["construction_input_record"] == (
        "data-src/liquid-from-solid-inputs.json rows.MnO"
    )
    assert "tier-4 liquid_from_solid construction" in condensate_provenance["flags"]
    assert any("supercooled-liquid" in flag for flag in condensate_provenance["flags"])
    assert any("tier-3" in flag for flag in condensate_provenance["flags"])
    expected_band = {
        temperature: construction.below_fusion_band_kj_mol(
            temperature,
            inputs["tier_error_bands"]["fusion_entropy_tier3"]["max"][0],
        )
        + residual / 1000.0
        for temperature in (1673.0, 2000.0)
    }
    for temperature, value in expected_band.items():
        assert condensate_provenance["uncertainty_band"]["values_J_per_mol"][
            int(temperature)
        ] == pytest.approx(value * 1000.0, abs=1e-6)

    source_manifest = yaml.safe_load(
        (ROOT / "data-src" / "published-thermo" / "PROVENANCE.yaml").read_text(
            encoding="utf-8"
        )
    )
    source_records = {
        record["record_id"]: record for record in source_manifest["records"]
    }
    mno_source = json.loads(
        (ROOT / "data-src" / "published-thermo" / "tsiv-table-1436.json").read_text(
            encoding="utf-8"
        )
    )
    assert mno_source["accuracy_class"] == "4-E"
    assert mno_source["formation_enthalpy_298_kj_mol"] == 155.620
    assert mno_source["formation_enthalpy_0_kj_mol"] == 156.096
    assert mno_source["dissociation_enthalpy_kj_mol"] == {"value": 374.0}
    assert "not quantify a numeric uncertainty band" in mno_source["table_note"]
    for source_id, source_path in (
        ("TSIV-1435", ROOT / "data-src" / "crystal-thermochemistry.json"),
        ("TSIV-1436", ROOT / "data-src" / "published-thermo" / "tsiv-table-1436.json"),
    ):
        assert source_records[source_id]["source_sha256"] == hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest()

    mn_gas_provenance = [
        item
        for item in provenance["rows"]
        if item["table"] == "gas" and item["species_name"] in {"MnO(g)", "Mn+(g)"}
    ]
    assert {(item["species_name"], item["T_interval"]) for item in mn_gas_provenance} == {
        ("MnO(g)", 1), ("MnO(g)", 2), ("Mn+(g)", 1), ("Mn+(g)", 2),
    }
    for item in mn_gas_provenance:
        if item["species_name"] == "MnO(g)":
            assert item["table_id"] == "TSIV-1436"
            assert item["tier"] == 3
            assert item["uncertainty_band"]["kind"] == (
                "Table 1436 prints accuracy class 4-E; numeric uncertainty band is "
                "not quantified by the cited page"
            )
            assert "phi_uncertainty_J_mol_K" not in item["uncertainty_band"]
        else:
            assert item["table_id"] == "Mn-006"
            assert item["tier"] == 1

    pack = load_gas_datapack()
    from openimcc.gas import _oxide_row_for_T
    from openimcc.redox_pack import load_redox_pack, species_gibbs_j_mol

    redox_pack = load_redox_pack(redox_path)
    assert len(pack.oxide_df.loc["MnO(l)"]) == 3
    for temperature in range(1200, 3001, 100):
        gas_row = _oxide_row_for_T(pack.oxide_df, "MnO(l)", float(temperature))
        gas_gibbs = _condensate_gibbs(gas_row, float(temperature))
        redox_gibbs = species_gibbs_j_mol(
            redox_pack, "MnO(l)", float(temperature)
        )
        constructed_gibbs = construction.at(
            float(temperature)
        ).gibbs_kj_mol * 1000.0
        assert gas_gibbs == pytest.approx(redox_gibbs, abs=1e-9)
        assert abs(gas_gibbs - constructed_gibbs) <= 10.0


def test_constructed_condensate_bands_match_liquid_construction_inputs() -> None:
    inputs_path = ROOT / "data-src" / "liquid-from-solid-inputs.json"
    inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(inputs_path.read_bytes()).hexdigest()
    provenance = yaml.safe_load(PROVENANCE_PATH.read_text(encoding="utf-8"))
    constructed = [
        row
        for row in provenance["rows"]
        if row.get("table") == "condensate"
        and row.get("method") == "liquid_from_solid_construction"
    ]
    assert constructed
    # These rows store source parameters or a derived envelope, not a value table.
    other_input_forms = {
        "MnO(l)": "fusion and fit-residual band checked by the MnO construction test",
        "SeO2(l)": "crystal-Cp extrapolation parameters",
        "CdO(l)": "crystal-Cp extrapolation parameters",
        "ZnO(l)": "TSIV uncertainties and crystal-Cp extension parameters",
    }
    seen_other_forms: set[str] = set()
    for row in constructed:
        species = row["species_name"]
        input_row = inputs["rows"][species.removesuffix("(l)")]
        assert row["construction_inputs_sha256"] == digest, species
        band = input_row.get("measured_band")
        if band and isinstance(band.get("values_J_per_mol"), dict):
            expected = band["values_J_per_mol"]
            actual = row["uncertainty_band"]["values_J_per_mol"]
            actual_values = {str(key): value for key, value in actual.items()}
            assert set(actual_values) == {str(key) for key in expected}, species
            for temperature, value in expected.items():
                assert actual_values[str(temperature)] == pytest.approx(value, abs=0.002), (
                    f"{species} at {temperature} K"
                )
            continue
        assert species in other_input_forms, f"unhandled input band for {species}"
        seen_other_forms.add(species)
        assert row["uncertainty_band"]["values_J_per_mol"], species
        if species in {"SeO2(l)", "CdO(l)"}:
            assert input_row["crystal_cp_extrapolation_band"]
        elif species == "ZnO(l)":
            assert input_row["measured_band"].get("phi_uncertainty_j_mol_k")
        else:
            assert "fusion_entropy" in input_row["inputs"]
    assert seen_other_forms == set(other_input_forms), sorted(seen_other_forms)


def test_k2o_from_solid_builder_fits_the_full_runtime_interval() -> None:
    rows = [
        row
        for row in build_gas_tables.build_condensate_rows(JANAF_DATA)
        if row["species_name"] == "K2O(l)"
    ]
    assert [(int(row["T_min"]), int(row["T_max"])) for row in rows] == [
        (1200, 2000),
        (2000, 3000),
    ]
    assert all(row["Ref"] == "K-012" for row in rows)
    assert all(float(row["_max_residual_J_per_mol"]) < 10.0 for row in rows)


def test_trace_parent_reactions_balance_every_element() -> None:
    trace_channels = {
        "Li", "LiO", "Li2O", "Li2O2",
        "Rb", "RbO", "Rb2O",
        "Pb", "PbO", "PbO2",
        "Ga", "GaO", "Ga2O", "Ge", "GeO", "GeO2",
        "B", "BO", "BO2", "B2O", "B2O2", "B2O3",
        "In", "InO", "In2O",
        "Cs", "CsO", "Cs2O",
        "Cu", "CuO", "Cu2",
        "Sn", "SnO", "SnO2",
        "Zn", "ZnO", "Cd", "CdO", "Bi", "Bi2", "BiO", "MnO",
        "Te", "Te2", "TeO2", "Se", "Se2", "SeO", "SeO2",
        "As", "As2", "As4", "As4O6",
        "Sb", "Sb2", "Sb4", "Sb4O6", "Ag",
        "Mo", "MoO", "MoO2", "MoO3", "(MoO3)2", "(MoO3)3",
        "(MoO3)4", "(MoO3)5", "W", "WO", "WO2", "WO3",
        "(WO3)2", "(WO3)3", "(WO3)4", "W3O8",
        "(WO3)5",
        "Ba", "BaO", "Sr", "SrO", "Zr", "ZrO", "ZrO2",
        "NiO", "CoO",
        "Sc", "ScO", "ScO2", "U", "UO", "UO2", "UO3",
    }
    assert trace_channels == set(build_gas_tables.TRACE_GAS_SPECIES)

    def atom_counts(formula: str) -> dict[str, int]:
        return _formula_atoms(formula)

    for species in trace_channels:
        parent, n_gas, n_o2 = _SF04_REACTIONS[species]
        parent_atoms = atom_counts(parent)
        gas_atoms = atom_counts(species)
        element = next(
            name for name in (
                "Li", "Rb", "Pb", "Cs", "Cu", "Sn", "Ga", "Ge", "B",
                "In", "Zn", "Cd", "Bi", "Mn", "Te", "Se", "As", "Sb", "Ag",
                "Mo", "W", "Ba", "Sr", "Zr", "Ni", "Co", "Sc", "U",
            )
            if name in parent_atoms
        )
        assert parent_atoms[element] == pytest.approx(n_gas * gas_atoms[element])
        parent_oxygen = parent_atoms.get("O", 0)
        product_oxygen = n_gas * gas_atoms.get("O", 0) + 2.0 * n_o2
        assert parent_oxygen == pytest.approx(product_oxygen)


def test_all_candidate_liquid_sources_cover_the_fit_interval() -> None:
    for species, table_id in CANDIDATE_LIQUID_TABLE_IDS.items():
        table = _record(table_id)["table"]
        assert table["index_entry"]["state"] == "l"
        temperatures = {
            row["temperature"]
            for row in _complete_rows(table_id)
            if 1500.0 <= row["temperature"] <= 3000.0
        }
        assert min(temperatures) == 1500.0
        assert max(temperatures) == 3000.0
        assert CANDIDATE_LIQUID_TRANSITIONS_K[species] in {
            float(str(ambiguity["raw_line"]).split("\t", 1)[0])
            for ambiguity in table.get("parse_ambiguities", [])
            if str(ambiguity.get("raw_line", "")).split("\t", 1)[0]
        }


# Reference solids vendored for solid-to-liquid standard-state conversion of
# oxide activities. They are not runtime inputs.
REFERENCE_SOLID_TABLE_IDS = {
    "Al2O3": ("Al-096", "Al-100"),
    "CaO": ("Ca-027", "Ca-028"),
    "SiO2": ("O-035", "O-038"),
}


def _printed_formation_gibbs_kj(table_id: str, temperature: float) -> float:
    for raw in _record(table_id)["table"]["values"]:
        if _value(raw, "temperature") == temperature:
            return float(_value(raw, "formation_gibbs_energy"))
    raise AssertionError(f"{table_id} has no complete {temperature} K row")


# NIST Atomic Spectra Database first ionisation energies:
# B https://physics.nist.gov/PhysRefData/Handbook/Tables/borontable1_a.htm
# Ga https://physics.nist.gov/PhysRefData/Handbook/Tables/galliumtable1_a.htm
# Ge https://physics.nist.gov/PhysRefData/Handbook/Tables/germaniumtable1_a.htm
# In https://physics.nist.gov/PhysRefData/Handbook/Tables/indiumtable1.htm
@pytest.mark.parametrize(
    ("oxide", "temperature", "expected_kj"),
    (
        ("Al2O3", 1900.0, 18.101),
        ("Al2O3", 2000.0, 14.299),
        ("CaO", 1900.0, 31.948),
        ("CaO", 2000.0, 29.537),
        ("CaO", 3200.0, 0.0),
        ("SiO2", 1900.0, 0.427),
        ("SiO2", 2000.0, -0.025),
    ),
)
def test_reference_solid_tables_give_printed_fusion_gibbs(
    oxide: str, temperature: float, expected_kj: float
) -> None:
    # Premise: the crystal and liquid tables of one oxide print formation
    # Gibbs energies from the same reference elements at the same T, so the
    # element terms cancel in the difference:
    #   dG_fus(T) = DfG(l, T) - DfG(cr, T)        [kJ/mol - kJ/mol = kJ/mol]
    # An activity on the solid standard state then follows from one on the
    # liquid standard state as
    #   log10 a(cr ref) = log10 a(l ref) + dG_fus(T) / (R T ln 10).
    # Sanity: dG_fus is positive below the melting point, zero at it (CaO,
    # 3200 K, where both tables print the same value) and negative above it
    # (SiO2 at 2000 K, above 1996 K).
    solid_id, liquid_id = REFERENCE_SOLID_TABLE_IDS[oxide]
    assert _record(solid_id)["table"]["index_entry"]["state"] == "cr"
    assert _record(liquid_id)["table"]["index_entry"]["state"] == "l"
    fusion = _printed_formation_gibbs_kj(
        liquid_id, temperature
    ) - _printed_formation_gibbs_kj(solid_id, temperature)
    assert fusion == pytest.approx(expected_kj, abs=5.0e-4)


def test_runtime_provenance_mirror_matches_yaml() -> None:
    provenance = yaml.safe_load(PROVENANCE_PATH.read_text(encoding="utf-8"))
    expected_gas = {}
    expected_oxide = {}
    expected_table_ids = {}
    for row in provenance["rows"]:
        if row["table"] == "gas":
            if row["species_name"] in build_gas_tables.DECLINED_ION_GAS_SPECIES:
                continue
            expected_gas[row["species_name"].removesuffix("(g)")] = row["authority"]
        else:
            if (
                row.get("extrapolation") is True
                and row["species_name"] not in {
                    "ZnO(l)", "CdO(l)", "Bi2O3(l)", "MnO(l)", "TeO2(l)", "SeO2(l)", "As2O3(l)", "Sb2O3(l)", "Ag2O(l)", "UO2(l)"
                }
            ):
                continue
            name = row["species_name"]
            expected_oxide[name.removesuffix("(l)")] = row["authority"]
            expected_table_ids[name] = row.get(
                "source_table_id",
                row.get("table_id", (row.get("source") or {}).get("table_id")),
            )

    # This is the deliberate source-of-truth boundary: the runtime remains
    # free of a YAML dependency, while this test compares every default gas
    # and oxide provenance row against the checked-in YAML. SiO2(cr) remains
    # available for explicit solid-phase thermochemistry queries.
    public_runtime_gas = dict(_GAS_PROVENANCE_AUTHORITY)
    public_runtime_oxide = {
        species: authority
        for species, authority in _OXIDE_PROVENANCE_AUTHORITY.items()
        if authority != "external_datapack"
    }
    assert public_runtime_gas == expected_gas
    assert public_runtime_oxide == expected_oxide
    assert _OXIDE_SOURCE_TABLE_IDS == {
        species.removesuffix("(l)").removesuffix("(cr)"): table_id
        for species, table_id in expected_table_ids.items()
        if species.endswith(("(l)", "(cr)"))
        and species.removesuffix("(l)").removesuffix("(cr)")
        in _OXIDE_SOURCE_TABLE_IDS
    }


def test_fitted_gas_rows_reproduce_every_complete_janaf_g_app_row() -> None:
    pack = load_gas_datapack()
    maxima: dict[str, tuple[float, float]] = {}
    for species, table_id in GAS_TABLE_IDS.items():
        source_rows = _complete_rows(table_id)
        reference = next(
            row["formation_enthalpy"]
            for row in source_rows
            if row["temperature"] == 298.15
        )
        fitted_row = _gas_row_for_interval(pack, species, 1)
        residuals = []
        log_residuals = []
        for source_row in source_rows:
            temperature = source_row["temperature"]
            if (
                not 1500.0
                <= temperature
                <= build_gas_tables._FIT_T_MAX_BY_TABLE.get(
                    table_id, build_gas_tables.FIT_T_MAX
                )
                or temperature in _ambiguous_temperatures(table_id)
            ):
                continue
            # Premise: the source apparent Gibbs value uses the 298.15 K
            # formation enthalpy plus H-H(298) minus T*S. Algebra: this is the
            # independent value against which the fitted Shomate row is tested.
            # Unit check: kJ/mol is converted to J/mol before T(K)*S(J/mol/K).
            # Sanity: the maximum residual must stay far below 0.01 log10 K.
            source_g = (
                reference + source_row["enthalpy_increment"]
            ) * 1000.0 - temperature * source_row["entropy"]
            fitted_g = _janaf_gibbs(temperature, fitted_row)
            residual = abs(fitted_g - source_g)
            residuals.append(residual)
            log_residuals.append(
                residual / (R_J_MOL_K * temperature * math.log(10.0))
            )
        maxima[species] = (max(residuals), max(log_residuals))

    assert max(value[1] for value in maxima.values()) <= 0.01
    assert max(value[0] for value in maxima.values()) < 2.1


def test_low_gas_rows_reproduce_janaf_nodes_and_generated_coefficients() -> None:
    pack = load_gas_datapack()
    provenance = yaml.safe_load(PROVENANCE_PATH.read_text(encoding="utf-8"))
    low_provenance = {
        row["species_name"]: row
        for row in provenance["rows"]
        if row["table"] == "gas"
        and row.get("T_interval") == 2
        and row["species_name"]
        not in {source[0] for source in build_gas_tables.TRACE_ION_NASA_SOURCES}
    }
    source_by_species = {
        source[0].removesuffix("(g)"): source
        for source in build_gas_tables.GAS_SOURCES
    }
    packaged_low_rows = {
        row["species_name"]: row
        for row in csv.DictReader(
            io.StringIO(
                (GAS_DATA / "gas-shomate.csv").read_text(encoding="utf-8")
            )
        )
        if row["T_interval"] == "2"
    }

    assert set(low_provenance) == {
        f"{species}(g)" for species in build_gas_tables.LOW_T_GAS_SPECIES
    } | {"Na2O(g)", "K2O(g)"} | {
        f"{species}(g)" for species in build_gas_tables.TRACE_GAS_SPECIES
    } | {
        source[0] for source in build_gas_tables.TRACE_ION_GAS_NASA_SOURCES
    } | {
        source[0] for source in build_gas_tables.TRACE_ION_GAS_YAML_SOURCES
    } | {"Zn+(g)", "Zn-(g)", "Bi+(g)"}
    for species in build_gas_tables.LOW_T_GAS_SPECIES:
        table_id = GAS_TABLE_IDS[species]
        row = _gas_row_for_interval(pack, species, 2)
        expected = build_gas_tables._fit_row(
            JANAF_DATA,
            *source_by_species[species],
            t_interval=2,
            fit_t_min=build_gas_tables.LOW_FIT_T_MIN,
            fit_t_max=build_gas_tables.LOW_FIT_T_MAX,
            minimum_rows=build_gas_tables.LOW_FIT_MINIMUM_ROWS,
        )
        assert int(row["T_interval"]) == 2
        assert float(row["T_min"]) == build_gas_tables.LOW_FIT_T_MIN
        assert float(row["T_max"]) == build_gas_tables.LOW_FIT_T_MAX
        assert row["Ref"] == table_id
        packaged_row = packaged_low_rows[f"{species}(g)"]
        assert packaged_row["species_name"] == expected["species_name"]
        assert packaged_row["T_interval"] == expected["T_interval"]
        assert packaged_row["T_min"] == expected["T_min"]
        assert packaged_row["T_max"] == expected["T_max"]
        assert packaged_row["Ref"] == expected["Ref"]
        _assert_fit_gibbs_matches(_shomate_g, packaged_row, expected)

        ambiguous = _ambiguous_temperatures(table_id)
        fit_rows = [
            source_row
            for source_row in _complete_rows(table_id)
            if (
                build_gas_tables.LOW_FIT_T_MIN
                <= source_row["temperature"]
                <= build_gas_tables.LOW_FIT_T_MAX
                and source_row["temperature"] not in ambiguous
            )
        ]
        assert len(fit_rows) >= build_gas_tables.LOW_FIT_MINIMUM_ROWS
        assert fit_rows[0]["temperature"] == build_gas_tables.LOW_FIT_T_MIN
        assert fit_rows[-1]["temperature"] == build_gas_tables.LOW_FIT_T_MAX
        residuals = []
        log_residuals = []
        for source_row in fit_rows:
            temperature = source_row["temperature"]
            residual = abs(
                _janaf_gibbs(temperature, row)
                - _janaf_apparent_gibbs(table_id, temperature)
            )
            residuals.append(residual)
            log_residuals.append(
                residual / (R_J_MOL_K * temperature * math.log(10.0))
            )

        recorded = low_provenance[f"{species}(g)"]
        assert max(residuals) == pytest.approx(
            float(recorded["max_residual_J_per_mol"]), abs=1.0e-8
        )
        assert max(log_residuals) == pytest.approx(
            float(recorded["max_residual_log10_K"]), abs=1.0e-12
        )
        assert max(residuals) < 2.1
        assert max(log_residuals) < 0.01


def test_low_and_high_gas_fits_are_continuous_at_1500_k() -> None:
    pack = load_gas_datapack()
    jumps = {}
    for species in build_gas_tables.LOW_T_GAS_SPECIES:
        low = _gas_row_for_interval(pack, species, 2)
        high = _gas_row_for_interval(pack, species, 1)
        jumps[species] = (
            abs(_janaf_gibbs(1500.0, low) - _janaf_gibbs(1500.0, high)),
            abs(_shomate_entropy(low, 1500.0) - _shomate_entropy(high, 1500.0)),
        )

    # Retain the measured-value gate used for the original low-temperature rows.
    assert max(g_jump for g_jump, _ in jumps.values()) < 1.0
    assert max(s_jump for _, s_jump in jumps.values()) < 0.01


def test_k_gas_apparent_gibbs_uses_its_298_k_anchor_across_1032_k() -> None:
    table_id = "K-005"
    source_rows = _complete_rows(table_id)
    reference = next(row for row in source_rows if row["temperature"] == 298.15)
    assert reference["formation_enthalpy"] == 89.0

    at_1000 = next(row for row in source_rows if row["temperature"] == 1000.0)
    at_1200 = next(row for row in source_rows if row["temperature"] == 1200.0)
    assert at_1200["formation_enthalpy"] == 0.0
    expected_1200 = (
        reference["formation_enthalpy"] + at_1200["enthalpy_increment"]
    ) * 1000.0 - 1200.0 * at_1200["entropy"]
    assert _janaf_apparent_gibbs(table_id, 1200.0) == pytest.approx(
        expected_1200, abs=1.0e-9
    )

    pack = load_gas_datapack()
    low = _gas_row_for_interval(pack, "K", 2)
    for source_row in (at_1000, at_1200):
        temperature = source_row["temperature"]
        assert _janaf_gibbs(temperature, low) == pytest.approx(
            _janaf_apparent_gibbs(table_id, temperature), abs=2.1
        )


def test_original_gas_interval_1_rows_match_52db3a9_bytes_and_order() -> None:
    lines = (GAS_DATA / "gas-shomate.csv").read_bytes().splitlines(keepends=True)
    retained = [lines[0]]
    intervals = []
    ion_species = {
        name.removesuffix("(g)") for name in _ION_GAS_PROVENANCE_SPECIES
    } | {
        source[0].removesuffix("(g)")
        for source in build_gas_tables.TRACE_ION_GAS_TEXT_SOURCES
    }
    ps_table_ids = {GAS_TABLE_IDS[species] for species in (
        "P", "P2", "P4", "PO", "PO2", "P4O6", "P4O10",
        "S", "S2", "S3", "S4", "S5", "S6", "S7", "S8",
        "SO", "SSO", "SO2", "SO3",
    )}
    for line in lines[1:]:
        values = next(csv.reader([line.decode("utf-8")]))
        species = values[0].removesuffix("(g)")
        if species in ion_species or species in build_gas_tables.TRACE_GAS_SPECIES:
            continue
        interval = values[2]
        intervals.append(interval)
        if interval == "1" and values[-1] not in ps_table_ids:
            if values[0] == "Cr(g)":
                values[7] = "2900"
                retained.append(",".join(values).encode("utf-8") + b"\n")
                continue
            retained.append(line)

    assert intervals[: intervals.index("2")] == ["1"] * 62
    assert all(interval == "2" for interval in intervals[intervals.index("2") :])
    assert hashlib.sha256(b"".join(retained)).hexdigest() == _BASE_INTERVAL_1_SHA256


def test_existing_gas_coefficient_rows_are_text_identical_to_base() -> None:
    baseline = subprocess.run(
        [
            "git",
            "show",
            "e724b6b:src/openimcc/data/gas/gas-shomate.csv",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    baseline_rows = list(csv.DictReader(io.StringIO(baseline)))
    current_rows = list(
        csv.DictReader(
            io.StringIO(
                (GAS_DATA / "gas-shomate.csv").read_text(encoding="utf-8")
            )
        )
    )
    current_by_key = {
        (
            row["species_name"],
            row["state"],
            row["T_min"],
            row["T_interval"],
        ): row
        for row in current_rows
    }
    for row in baseline_rows:
        key = (
            row["species_name"],
            row["state"],
            row["T_min"],
            row["T_interval"],
        )
        current = current_by_key[key]
        if row["species_name"] == "Cr(g)" and row["T_interval"] == "1":
            current = {**current, "T_max": row["T_max"]}
        assert current == row


def _hand_nasa7_properties(record: dict, temperature: float) -> tuple[float, float, float]:
    interval = next(
        (
            item
            for item in record["intervals"]
            if item["T_min_K"]["value"] <= temperature <= item["T_max_K"]["value"]
        ),
        None,
    )
    if interval is None and temperature < record["intervals"][0]["T_min_K"]["value"]:
        interval = record["intervals"][0]
    assert interval is not None
    a = [float(item["value"]) for item in interval["a_coefficients"]]
    a1, a2, a3, a4, a5, a6, a7 = a
    cp_R = (
        a1 / temperature**2
        + a2 / temperature
        + a3
        + a4 * temperature
        + a5 * temperature**2
        + a6 * temperature**3
        + a7 * temperature**4
    )
    h_rt = (
        -a1 / temperature**2
        + a2 * math.log(temperature) / temperature
        + a3
        + a4 * temperature / 2.0
        + a5 * temperature**2 / 3.0
        + a6 * temperature**3 / 4.0
        + a7 * temperature**4 / 5.0
        + float(interval["b1"]["value"]) / temperature
    )
    s_R = (
        -a1 / (2.0 * temperature**2)
        - a2 / temperature
        + a3 * math.log(temperature)
        + a4 * temperature
        + a5 * temperature**2 / 2.0
        + a6 * temperature**3 / 3.0
        + a7 * temperature**4 / 4.0
        + float(interval["b2"]["value"])
    )
    return cp_R, h_rt, s_R


def test_nasa_rows_match_anchored_card_evaluation_and_2000_k_hand_check() -> None:
    pack = load_gas_datapack()
    lh84 = yaml.safe_load((LH84_DATA / "lh84.yaml").read_text(encoding="utf-8"))
    lh84_rows = {row["species_name"]: row for row in lh84["rows"]}
    provenance = yaml.safe_load(PROVENANCE_PATH.read_text(encoding="utf-8"))
    recorded = {
        (row["species_name"], int(row.get("T_interval", 1))): row
        for row in provenance["rows"]
        if row["species_name"] in {"Na2O(g)", "K2O(g)"}
    }

    for species, table_id in NASA_TABLE_IDS.items():
        cation_table_id = "Na-005" if species == "Na2O" else "K-005"
        rows, checks = build_gas_tables._nasa_source_rows(
            _nasa_record(table_id),
            lh84_rows[species + "(g)"],
            {cation_table_id: _record(cation_table_id), "O-029": _record("O-029")},
            cation_table_id,
        )
        assert [row["temperature"] for row in rows] == list(
            np.arange(1500.0, 3000.1, 100.0)
        )

        fitted = _gas_row_for_interval(pack, species, 1)
        residuals = [
            abs(
                _janaf_gibbs(row["temperature"], fitted)
                - row["formation_gibbs_energy"] * 1000.0
            )
            for row in rows
        ]
        assert max(residuals) == pytest.approx(
            float(recorded[(species + "(g)", 1)]["max_residual_J_per_mol"]),
            abs=1e-9,
        )
        assert max(residuals) < 1e-3

        card = _nasa_record(table_id)
        cp_R, h_rt, s_R = _hand_nasa7_properties(card, 2000.0)
        assert cp_R == pytest.approx(
            build_gas_tables._nasa7_properties(card, 2000.0)["cp_R"]
        )
        hand_card_g = R_J_MOL_K * 2000.0 * (h_rt - s_R)
        assert hand_card_g == pytest.approx(
            checks["card_gibbs_check_J_per_mol"], abs=1e-8
        )

        lh = lh84_rows[species + "(g)"]
        dfh_over_R = float(lh["dfH_over_R_kK"]["value"]) * 1000.0
        hinc_over_R = float(lh["Hinc_over_R_kK"]["value"]) * 1000.0
        card_298 = _hand_nasa7_properties(card, 298.15)
        h_over_R = (
            dfh_over_R
            - hinc_over_R
            + hinc_over_R
            + 2000.0 * h_rt
            - 298.15 * card_298[1]
        )
        # LH84's 1 atm S° to the 1 bar runtime standard state: +ln(1.01325).
        entropy = R_J_MOL_K * (
            float(lh["S_over_R"]["value"]) + math.log(1.01325) + s_R - card_298[2]
        )
        hand_anchored_g = R_J_MOL_K * h_over_R - 2000.0 * entropy
        assert hand_anchored_g == pytest.approx(
            checks["anchored_card_gibbs_check_J_per_mol"], abs=1e-8
        )


def test_nasa_low_rows_use_piecewise_cards_and_lh84_anchors() -> None:
    pack = load_gas_datapack()
    lh84 = yaml.safe_load((LH84_DATA / "lh84.yaml").read_text(encoding="utf-8"))
    lh84_rows = {row["species_name"]: row for row in lh84["rows"]}
    provenance = yaml.safe_load(PROVENANCE_PATH.read_text(encoding="utf-8"))
    recorded = {
        row["species_name"]: row
        for row in provenance["rows"]
        if row["table"] == "gas"
        and row["species_name"] in {"Na2O(g)", "K2O(g)"}
        and int(row.get("T_interval", 1)) == 2
    }

    for species, table_id in NASA_TABLE_IDS.items():
        cation_table_id = "Na-005" if species == "Na2O" else "K-005"
        source_rows, checks = build_gas_tables._nasa_source_rows(
            _nasa_record(table_id),
            lh84_rows[species + "(g)"],
            {cation_table_id: _record(cation_table_id), "O-029": _record("O-029")},
            cation_table_id,
            fit_t_min=500.0,
            fit_t_max=1500.0,
        )
        assert [row["temperature"] for row in source_rows] == list(
            np.arange(500.0, 1500.1, 100.0)
        )
        low = _gas_row_for_interval(pack, species, 2)
        assert (int(low["T_min"]), int(low["T_max"])) == (500, 1500)
        residuals = [
            abs(
                _janaf_gibbs(source["temperature"], low)
                - source["formation_gibbs_energy"] * 1000.0
            )
            for source in source_rows
        ]
        assert max(residuals) == pytest.approx(
            float(recorded[species + "(g)"]["max_residual_J_per_mol"]), abs=1e-9
        )
        assert max(residuals) < 2.0e-2
        assert checks["check_temperature_K"] == 1500.0
        assert checks["card_gibbs_check_J_per_mol"] is not None
        assert checks["breakpoint_checks"]
        assert checks["breakpoint_checks"][0]["temperature_K"] == 1000.0
        assert abs(checks["breakpoint_checks"][0]["delta_H_J_per_mol"]) < 1.0e-3
        assert abs(checks["breakpoint_checks"][0]["delta_S_J_per_mol_K"]) < 1.0e-6


def test_na2o_liquid_to_gas_reaction_uses_new_gas_row_and_existing_condensate() -> None:
    from openimcc.gas import _oxide_row_for_T

    pack = load_gas_datapack()
    gas_row = _gas_row_for_interval(pack, "Na2O", 1)
    temperature = 2000.0
    oxide_row = _oxide_row_for_T(pack.oxide_df, "Na2O(l)", temperature)
    tau = temperature / 1000.0
    oxide_poly = sum(
        float(oxide_row[column]) * tau**power
        for power, column in enumerate(("dG_A", "dG_B", "dG_C", "dG_D", "dG_E"))
    )
    hand_liquid_g = (
        -R_J_MOL_K * temperature * oxide_poly
        + R_J_MOL_K * float(oxide_row["dH298_R"]) * 1000.0
    )
    assert hand_liquid_g == pytest.approx(_lamor_gibbs(temperature, oxide_row))

    lh84 = yaml.safe_load((LH84_DATA / "lh84.yaml").read_text(encoding="utf-8"))
    lh84_na = next(row for row in lh84["rows"] if row["species_name"] == "Na2O(g)")
    source_rows, _ = build_gas_tables._nasa_source_rows(
        _nasa_record("NG-0905"),
        lh84_na,
        {"Na-005": _record("Na-005"), "O-029": _record("O-029")},
        "Na-005",
    )
    source_2000 = next(row for row in source_rows if row["temperature"] == temperature)
    gas_g = _janaf_gibbs(temperature, gas_row)
    hand_reaction_g = source_2000["formation_gibbs_energy"] * 1000.0 - hand_liquid_g
    assert gas_g - hand_liquid_g == pytest.approx(hand_reaction_g, abs=1e-3)


def test_na_supercooled_parent_liquid_covers_low_gas_and_melt_activity() -> None:
    from openimcc import evaluate as evaluate_imcc

    composition = {
        "SiO2": 51.85068,
        "MgO": 4.78527,
        "FeO": 13.77307,
        "CaO": 9.02862,
        "Al2O3": 14.80572,
        "TiO2": 1.73824,
        "Na2O": 3.23108,
        "K2O": 0.78732,
    }
    melt = evaluate_imcc(composition, 1800.0, basis_type="wt")
    assert melt.activity("Na2O") > 0.0

    gas = evaluate_gas(
        {"Na2O": melt.activity("Na2O")},
        1200.0,
        1.0e-10,
        load_gas_datapack(),
        gas_species=("Na2O",),
    )
    assert gas["Na2O"] > 0.0
    assert gas.domain_flags["Na2O"] is None


def test_fitted_condensate_rows_reproduce_every_complete_janaf_g_app_row() -> None:
    """Fitted liquid rows reproduce their JANAF source, not just their fit.

    Premise: the runtime condensate form ``1000*R*dH298_R - R*T*P(T/1000)``
    must equal the source apparent Gibbs energy
    ``(dfH(298) + H - H(298))*1000 - T*S`` at every complete source node in
    each row's declared fit interval. Unit check: both sides are J/mol.
    Sanity: the quartic in T/1000 cannot carry the exact constant-Cp ``ln T``
    and ``1/T`` terms, so every fitted liquid stays below the residual gate.
    """
    from openimcc.gas import _lamor_gibbs

    pack = load_gas_datapack()
    provenance = yaml.safe_load(PROVENANCE_PATH.read_text(encoding="utf-8"))
    recorded = {
        (row["species_name"], tuple(row["T_range_K"])): row
        for row in provenance["rows"]
        if row["table"] == "condensate"
    }
    for species, table_id in FITTED_CONDENSATE_TABLE_IDS.items():
        fit_t_min = build_gas_tables._FIT_T_MIN_BY_TABLE.get(
            table_id, build_gas_tables.FIT_T_MIN
        )
        runtime_t_min = 1500.0 if species == "Na2O(l)" else fit_t_min
        row = pack.oxide_df.loc[[species]]
        row = row.loc[row["T_min"].astype(float) == runtime_t_min].iloc[0]
        maximum_j = 0.0
        maximum_log10 = 0.0
        nodes = 0
        source_rows = _complete_rows(table_id)
        if species == "Na2O(l)":
            source_rows = [
                row for row in source_rows if row["temperature"] != 1500.0
            ]
            source_rows.append({
                "temperature": 1500.0,
                "heat_capacity": 104.600,
                "entropy": 260.601,
                "enthalpy_increment": 125.714,
            })
        for source_row in source_rows:
            temperature = source_row["temperature"]
            if (
                not fit_t_min <= temperature <= 3000.0
                or (
                    temperature in _ambiguous_temperatures(table_id)
                    and not (species == "Na2O(l)" and temperature == 1500.0)
                )
            ):
                continue
            residual = abs(
                _lamor_gibbs(temperature, row)
                - _source_g_app_from_row(table_id, source_row)
            )
            maximum_j = max(maximum_j, residual)
            maximum_log10 = max(
                maximum_log10,
                residual / (R_J_MOL_K * temperature * math.log(10.0)),
            )
            nodes += 1
        expected_nodes = {
            "TiO2(l)": 15,
            "Cr2O3(l)": 11,
            "V2O3(l)": 13,
            "NbO2(l)": 15,
            "Na2O(l)": 16,
        }[species]
        assert nodes == expected_nodes
        max_log10_limit = {
            "TiO2(l)": 0.001,
            "Cr2O3(l)": 0.001,
            "V2O3(l)": 0.001,
            "NbO2(l)": 0.001,
            "Na2O(l)": 0.001,
        }[species]
        max_j_limit = {
            "TiO2(l)": 10.0,
            "Cr2O3(l)": 10.0,
            "V2O3(l)": 10.0,
            "NbO2(l)": 10.0,
            "Na2O(l)": 10.0,
        }[species]
        assert maximum_log10 <= max_log10_limit
        assert maximum_j < max_j_limit
        assert maximum_j == pytest.approx(
            recorded[(species, (int(runtime_t_min), 3000))][
                "max_residual_J_per_mol"
            ],
            abs=1e-6,
        )
        assert maximum_log10 == pytest.approx(
            recorded[(species, (int(runtime_t_min), 3000))]["max_residual_log10_K"],
            abs=1e-9,
        )


def test_ba_sr_zr_janaf_liquids_fit_each_printed_node() -> None:
    """Direct liquid fits stay below the 10 J/mol gate on every source node."""
    from openimcc.gas import _lamor_gibbs

    pack = load_gas_datapack()
    source_by_parent = {
        "BaO(l)": "Ba-035",
        "SrO(l)": "O-014",
        "ZrO2(l)": "O-053",
    }
    provenance = yaml.safe_load(PROVENANCE_PATH.read_text(encoding="utf-8"))
    recorded = {
        (row["species_name"], tuple(row["T_range_K"])): row
        for row in provenance["rows"]
        if row["table"] == "condensate"
    }
    for species, table_id in source_by_parent.items():
        source_rows = _complete_rows(table_id)
        source_rows.extend(
            build_gas_tables._recover_thermal_transition_rows(
                _record(table_id)["table"]
            )
        )
        nodes = sorted({
            row["temperature"] for row in source_rows
            if 1200.0 <= row["temperature"] <= 3000.0
        })
        intervals = sorted(
            pack.oxide_df.loc[species, "T_min"].astype(float).unique().tolist()
        )
        assert intervals == [1200.0, 1500.0, 2000.0, 2500.0]
        for start in intervals:
            row = pack.oxide_df.loc[[species]]
            row = row.loc[row["T_min"].astype(float) == start].iloc[0]
            fit_end = float(row["T_max"])
            in_interval = [T for T in nodes if start <= T <= fit_end]
            assert in_interval
            residuals = [
                abs(_lamor_gibbs(T, row) - _janaf_apparent_gibbs(table_id, T))
                for T in in_interval
            ]
            assert max(residuals) < 10.0
            provenance_row = recorded[(species, (start, fit_end))]
            assert provenance_row["max_residual_J_per_mol"] < 10.0
            assert provenance_row["max_residual_J_per_mol"] == pytest.approx(
                max(residuals), abs=1e-5
            )


def test_nickel_cobalt_tsiv_liquid_fits_each_printed_node() -> None:
    """TSIV liquid rows reproduce their tabulated nodes within 10 J/mol."""
    from openimcc.gas import _lamor_gibbs

    pack = load_gas_datapack()
    cases = (
        ("NiO(l)", "Ni", "TSIV-2026", -239.700, 2228.0),
        ("CoO(l)", "Co", "TSIV-1334", -238.000, 2090.0),
    )
    provenance = yaml.safe_load(PROVENANCE_PATH.read_text(encoding="utf-8"))
    recorded = {
        (row["species_name"], tuple(row["T_range_K"])): row
        for row in provenance["rows"]
        if row["table"] == "condensate"
    }
    for species, _element, table_id, anchor_kj, melt_k in cases:
        source_path = (
            ROOT / "data-src/published-thermo"
            / f"tsiv-table-{table_id.split('-')[1]}.json"
        )
        source = json.loads(source_path.read_text(encoding="utf-8"))
        assert source["formation_enthalpy_298_kj_mol"]["value"] == anchor_kj
        assert source["fusion_temperature_K"]["value"] == melt_k
        source_rows = build_gas_tables._tsiv_liquid_source_rows(source)
        for interval_start, interval_end in ((1200.0, melt_k), (melt_k, 3000.0)):
            csv_row = pack.oxide_df.loc[[species]]
            csv_row = csv_row.loc[
                csv_row["T_min"].astype(float) == interval_start
            ].iloc[0]
            assert float(csv_row["T_max"]) == interval_end
            nodes = [
                row for temperature, row in source_rows.items()
                if interval_start <= temperature <= interval_end
            ]
            residuals = [
                    abs(
                        _lamor_gibbs(temperature, csv_row)
                        - (
                            1000.0 * source_row["formation_enthalpy"]
                            + 1000.0 * source_row["enthalpy_increment"]
                            - temperature * source_row["entropy"]
                        )
                    )
                    for temperature, source_row in (
                        (row["temperature"], row) for row in nodes
                    )
            ]
            assert residuals
            assert max(residuals) < 10.0
            item = recorded[(species, (interval_start, interval_end))]
            assert item["tier"] == 3
            assert item["max_residual_J_per_mol"] == pytest.approx(
                max(residuals), abs=1e-5
            )
            assert item["source_data"] is (interval_start == melt_k)
            band = item["uncertainty_band"]
            source_phi = source["phi_uncertainty_j_mol_k"]
            phi_knots = sorted(
                (float(t), float(value)) for t, value in source_phi.items()
            )
            band_values = {
                float(t): float(value)
                for t, value in band["values_J_per_mol"].items()
            }
            expected_knots = sorted(
                {
                    interval_start,
                    interval_end,
                    *(
                        t for t in (1500.0, 1800.0, 2000.0, 2500.0, 3000.0)
                        if interval_start < t < interval_end
                    ),
                }
            )
            assert sorted(band_values) == expected_knots
            for temperature in expected_knots:
                lower = max(t for t, _ in phi_knots if t <= temperature)
                upper = min(t for t, _ in phi_knots if t >= temperature)
                low_sigma = dict(phi_knots)[lower]
                high_sigma = dict(phi_knots)[upper]
                fraction = (
                    0.0 if lower == upper
                    else (temperature - lower) / (upper - lower)
                )
                sigma = low_sigma + fraction * (high_sigma - low_sigma)
                fusion_band = 0.0
                if species == "NiO(l)" and interval_start < melt_k:
                    fusion = source["fusion_enthalpy_kj_mol"]
                    assert fusion["uncertainty"] == 5.0
                    fusion_band = (
                        float(fusion["uncertainty"])
                        * 1000.0
                        * (1.0 - temperature / melt_k)
                    )
                    assert "maximum" in band["kind"]
                elif species == "CoO(l)":
                    assert "no stated numeric uncertainty" in band["kind"]
                assert band_values[temperature] == pytest.approx(
                    max(temperature * sigma, fusion_band), abs=0.001
                )


def test_nickel_cobalt_janaf_298_k_anchors_are_pinned() -> None:
    expected_kj_mol = {
        "Ni-005": 430.115, "Ni-006": 1179.003, "Ni-007": 311.764,
        "Co-005": 426.676, "Co-006": 1191.597, "Co-007": 356.649,
    }
    for table_id, expected in expected_kj_mol.items():
        anchor = next(
            row for row in _record(table_id)["table"]["values"]
            if _value(row, "temperature") == 298.15
        )
        assert _value(anchor, "formation_enthalpy") == pytest.approx(expected)


def test_ba_sr_zr_janaf_298_k_formation_anchors_are_pinned() -> None:
    expected_kj_mol = {
        "Ba-005": 179.075, "Ba-006": 688.126, "Ba-037": -123.846,
        "Ba-035": -491.627, "Sr-006": 164.000, "Sr-007": 719.666,
        "O-016": -13.389, "O-014": -517.292, "Zr-006": 610.027,
        "Zr-007": 1277.880, "O-028": 58.576, "O-055": -286.186,
        "O-053": -1023.155,
    }
    for table_id, expected in expected_kj_mol.items():
        anchor = next(
            row for row in _record(table_id)["table"]["values"]
            if _value(row, "temperature") == 298.15
        )
        assert _value(anchor, "formation_enthalpy") == pytest.approx(expected)


def test_no_fitted_row_consumes_a_parse_ambiguous_source_row() -> None:
    consumed_ambiguous_rows = set()
    for table_id in (*GAS_TABLE_IDS.values(), *FITTED_CONDENSATE_TABLE_IDS.values()):
        rows = build_gas_tables._usable_rows(_record(table_id)["table"], table_id)
        consumed_ambiguous_rows.update(
            (table_id, row["temperature"])
            for row in rows
            if row["temperature"] in _ambiguous_temperatures(table_id)
        )
    assert consumed_ambiguous_rows == {("Na-013", 1500.0)}


def test_na_condensate_recovers_1500_thermal_cells_for_both_intervals() -> None:
    from openimcc.gas import _lamor_gibbs, _oxide_row_for_T

    source = _record("Na-013")
    assert source["table"]["index_entry"]["state"] == "l"
    ambiguous = _ambiguous_temperatures("Na-013")
    assert ambiguous == {1405.2, 1500.0}
    low = build_gas_tables._usable_rows(
        source["table"], "Na-013", fit_t_min=1000.0, fit_t_max=1500.0,
        minimum_rows=6,
    )
    high = build_gas_tables._usable_rows(
        source["table"], "Na-013", fit_t_min=1500.0, fit_t_max=3000.0,
        minimum_rows=16,
    )
    assert [row["temperature"] for row in low] == [
        1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0
    ]
    assert [row["temperature"] for row in high] == list(
        np.arange(1500.0, 3000.1, 100.0)
    )
    recovered = low[-1]
    raw_1500_line = next(
        ambiguity["raw_line"]
        for ambiguity in source["table"]["parse_ambiguities"]
        if ambiguity["raw_line"].startswith("1500\t")
    )
    raw_1500_cells = raw_1500_line.split("\t")
    assert recovered["heat_capacity"] == float(raw_1500_cells[1])
    assert recovered["entropy"] == float(raw_1500_cells[2])
    assert recovered["enthalpy_increment"] == float(raw_1500_cells[4])
    assert {row["heat_capacity"] for row in low + high} == {
        recovered["heat_capacity"]
    }
    assert "formation_enthalpy" not in recovered
    assert "formation_gibbs_energy" not in recovered

    # With constant Cp, S(T2)=S(T1)+Cp*ln(T2/T1) and
    # Hinc(T2)=Hinc(T1)+Cp*(T2-T1)/1000. Continue from each adjacent node,
    # then compare apparent G in J/mol to the recovered source row.
    complete = _complete_rows("Na-013")
    reference_h = next(
        row["formation_enthalpy"] for row in complete
        if row["temperature"] == 298.15
    )
    node_1400 = next(row for row in complete if row["temperature"] == 1400.0)
    node_1600 = next(row for row in complete if row["temperature"] == 1600.0)
    expected_g = (
        reference_h + recovered["enthalpy_increment"]
    ) * 1000.0 - 1500.0 * recovered["entropy"]
    for neighbor in (node_1400, node_1600):
        delta_t = 1500.0 - neighbor["temperature"]
        continued_h = neighbor["enthalpy_increment"] + recovered["heat_capacity"] * delta_t / 1000.0
        continued_s = neighbor["entropy"] + recovered["heat_capacity"] * np.log(
            1500.0 / neighbor["temperature"]
        )
        continued_g = (reference_h + continued_h) * 1000.0 - 1500.0 * continued_s
        assert abs(continued_g - expected_g) < 1.0

    pack = load_gas_datapack()
    na_rows = pack.oxide_df.loc[["Na2O(l)"]]
    low_row = na_rows.loc[na_rows["T_min"].astype(float) == 1200.0].iloc[0]
    high_row = na_rows.loc[na_rows["T_min"].astype(float) == 1500.0].iloc[0]
    selected = _oxide_row_for_T(pack.oxide_df, "Na2O(l)", 1500.0)
    assert float(selected["T_min"]) == 1500.0
    assert species_thermo("Na2O", "l", 1500.0, pack).G_J_mol == pytest.approx(
        expected_g, abs=10.0
    )
    low_g = _lamor_gibbs(1500.0, low_row)
    high_g = _lamor_gibbs(1500.0, high_row)
    assert abs(low_g - expected_g) < 10.0
    assert abs(high_g - expected_g) < 10.0
    assert abs(low_g - high_g) == pytest.approx(2.779451693408191, abs=1e-9)


def _janaf_apparent_gibbs(table_id: str, T: float) -> float:
    """G_app = dfH(298) + [H(T) - H(298)] - T*S(T), in J/mol, from JANAF cells."""
    rows = _complete_rows(table_id)
    reference = next(row for row in rows if row["temperature"] == 298.15)
    row = next((row for row in rows if row["temperature"] == T), None)
    if row is None:
        # Some printed source rows have parse-ambiguous formation columns but
        # complete thermal cells. Keep their printed H/S nodes on the species'
        # independently parsed 298 K formation anchor.
        recovered = build_gas_tables._recover_thermal_transition_rows(
            _record(table_id)["table"]
        )
        source = next(row for row in recovered if row["temperature"] == T)
        return (
            reference["formation_enthalpy"] + source["enthalpy_increment"]
        ) * 1000.0 - T * source["entropy"]
    return (
        reference["formation_enthalpy"] + row["enthalpy_increment"]
    ) * 1000.0 - T * row["entropy"]


def test_trace_source_comparisons_use_apparent_gibbs() -> None:
    provenance = yaml.safe_load(PROVENANCE_PATH.read_text(encoding="utf-8"))
    comparisons = provenance["source_comparisons"]
    assert comparisons["temperatures_K"] == [1800, 2000, 2500, 3000]
    compared_species = {row["species_name"] for row in comparisons["rows"]}
    assert {
        "Mo(g)", "Mo+(g)", "Mo-(g)", "MoO(g)", "MoO2(g)", "MoO3(g)",
        "MoO3(l)", "W(g)", "W+(g)", "W-(g)", "WO(g)", "WO2(g)",
        "WO3(g)", "WO3(l)", "(WO3)2(g)", "(WO3)3(g)", "(WO3)4(g)",
        "Ba(g)", "Ba+(g)", "BaO(g)", "BaO(l)", "Sr(g)", "Sr+(g)",
        "SrO(g)", "SrO(l)", "Zr(g)", "Zr+(g)", "ZrO(g)",
        "ZrO2(g)", "ZrO2(l)", "Sc(g)", "Sc2O3(cr)", "Sc2O3(l)", "U(g)", "UO2(cr)",
    } <= compared_species
    for comparison in comparisons["rows"]:
        if "janaf_298K_element_reference_table" in comparison:
            assert comparison["janaf_298K_element_reference_table"] == (
                comparison["janaf_table_id"]
            )
        if comparison["species_name"] == "NiO(g)":
            nasa = _nasa_record(comparison["selected_nasa_record_id"])
            bulletin = json.loads(
                (PUBLISHED_DATA / "nio-bulletin-672.json").read_text(
                    encoding="utf-8"
                )
            )
            nasa_h298 = R_J_MOL_K * 298.15 * build_gas_tables._nasa7_properties(
                nasa, 298.15
            )["h_rt"]
            for value in comparison["values"]:
                temperature = float(value["T_K"])
                node = next(row for row in bulletin["nodes"] if row[0] == temperature)
                bulletin_g = (
                    float(bulletin["formation_enthalpy_298_kcal_mol"]) * 4184.0
                    + float(node[3]) * 4184.0
                    - temperature * float(node[2]) * 4.184
                )
                nasa_properties = build_gas_tables._nasa7_properties(
                    nasa, temperature
                )
                nasa_g = float(nasa["delta_f_H_298_15"]["value"]) + (
                    R_J_MOL_K * temperature * nasa_properties["h_rt"] - nasa_h298
                ) - temperature * R_J_MOL_K * nasa_properties["s_R"]
                assert value["bulletin_672_G_app_J_mol"] == pytest.approx(
                    bulletin_g, abs=0.001
                )
                assert value["nasa_G_app_J_mol"] == pytest.approx(
                    nasa_g, abs=0.001
                )
                assert value["nasa_minus_bulletin_672_J_mol"] == pytest.approx(
                    nasa_g - bulletin_g, abs=0.001
                )
            continue
        nasa = _nasa_record(comparison["nasa_record_id"])
        if "pankratz_table_id" in comparison:
            source_path = ROOT / comparison.get(
                "pankratz_source_path",
                "data-src/published-thermo/pankratz-bulletin-672.json",
            )
            source = json.loads(source_path.read_text(encoding="utf-8"))["tables"][
                comparison["pankratz_table_id"]
            ]
            if comparison["pankratz_table_id"] in {
                "P672-Sc-g", "P672-Sc2O3-l", "P672-U-g", "P672-UO2-c"
            }:
                anchor = _nasa_record(
                    "NG-1856" if comparison["species_name"].startswith("Sc2O3")
                    else comparison["nasa_record_id"]
                )
                anchor_h298 = R_J_MOL_K * 298.15 * build_gas_tables._nasa7_properties(
                    anchor, 298.15
                )["h_rt"]
                for value in comparison["values"]:
                    temperature = float(value["T_K"])
                    node = next(row for row in source["nodes"] if row[0] == temperature)
                    pankratz_g_minus_h298 = -temperature * float(node[1]) * 4.184
                    nasa = _nasa_record(comparison["nasa_record_id"])
                    nasa_properties = build_gas_tables._nasa7_properties(nasa, temperature)
                    nasa_g_minus_h298 = R_J_MOL_K * temperature * (
                        nasa_properties["h_rt"] - nasa_properties["s_R"]
                    ) - anchor_h298
                    assert value["pankratz_G_minus_H298_J_mol"] == pytest.approx(
                        pankratz_g_minus_h298, abs=0.001
                    )
                    assert value["nasa_G_minus_H298_J_mol"] == pytest.approx(
                        nasa_g_minus_h298, abs=0.001
                    )
                    assert value["nasa_minus_pankratz_J_mol"] == pytest.approx(
                        nasa_g_minus_h298 - pankratz_g_minus_h298, abs=0.001
                    )
                continue
            for value in comparison["values"]:
                temperature = float(value["T_K"])
                node = next(row for row in source["nodes"] if row[0] == temperature)
                pankratz_g = (
                    float(source["formation_enthalpy_298_kcal_mol"]) * 4184.0
                    + float(node[3]) * 4184.0
                    - temperature * float(node[2]) * 4.184
                )
                formation_h = float(nasa["delta_f_H_298_15"]["value"])
                card_h298 = (
                    R_J_MOL_K * 298.15
                    * build_gas_tables._nasa7_properties(nasa, 298.15)["h_rt"]
                )
                nasa_properties = build_gas_tables._nasa7_properties(nasa, temperature)
                if nasa.get("phase") in {"liquid", "L"}:
                    nasa_g = R_J_MOL_K * temperature * (
                        nasa_properties["h_rt"] - nasa_properties["s_R"]
                    )
                elif comparison["nasa_record_id"] == "NG-0074":
                    nasa_g = R_J_MOL_K * temperature * nasa_properties["h_rt"] - (
                        temperature * R_J_MOL_K * nasa_properties["s_R"]
                    )
                else:
                    nasa_h = formation_h + (
                        R_J_MOL_K * temperature * nasa_properties["h_rt"] - card_h298
                    )
                    nasa_g = nasa_h - temperature * R_J_MOL_K * nasa_properties["s_R"]
                assert value["pankratz_G_app_J_mol"] == pytest.approx(pankratz_g, abs=0.001)
                assert value["nasa_G_app_J_mol"] == pytest.approx(nasa_g, abs=0.001)
                assert value["nasa_minus_pankratz_J_mol"] == pytest.approx(
                    nasa_g - pankratz_g, abs=0.001
                )
            continue
        if (JANAF_DATA / f"{comparison['janaf_table_id']}.yaml").is_file():
            janaf_gibbs = lambda temperature: _janaf_apparent_gibbs(
                comparison["janaf_table_id"], temperature
            )
        else:
            janaf = build_gas_tables._janaf_text_source_rows(
                JANAF_DATA, comparison["janaf_table_id"]
            )
            formation_h_298 = janaf[298.15]["formation_enthalpy"]
            janaf_gibbs = lambda temperature: (
                formation_h_298
                + janaf[temperature]["enthalpy_increment"]
            ) * 1000.0 - temperature * janaf[temperature]["entropy"]
        formation_h = float(nasa["delta_f_H_298_15"]["value"])
        card_h298 = (
            R_J_MOL_K
            * 298.15
            * build_gas_tables._nasa7_properties(nasa, 298.15)["h_rt"]
        )
        for value in comparison["values"]:
            temperature = float(value["T_K"])
            janaf_g = janaf_gibbs(temperature)
            nasa_properties = build_gas_tables._nasa7_properties(
                nasa, temperature
            )
            card_g = R_J_MOL_K * temperature * (
                nasa_properties["h_rt"] - nasa_properties["s_R"]
            )
            if comparison["nasa_record_id"] in {
                "NG-1701", "NG-2013", "NG-1325", "NG-1917", "NG-2029"
            }:
                nasa_g = card_g
                header_g = formation_h + (
                    R_J_MOL_K * temperature * nasa_properties["h_rt"]
                    - card_h298
                ) - temperature * R_J_MOL_K * nasa_properties["s_R"]
                assert value["nasa_header_anchored_G_app_J_mol"] == pytest.approx(
                    header_g, abs=0.001
                )
                assert value["nasa_card_G_app_J_mol"] == pytest.approx(
                    card_g, abs=0.001
                )
                assert value["card_G_nasa_minus_janaf_J_mol"] == pytest.approx(
                    card_g - janaf_g, abs=0.001
                )
            else:
                nasa_h = formation_h + (
                    R_J_MOL_K * temperature * nasa_properties["h_rt"] - card_h298
                )
                nasa_g = nasa_h - temperature * R_J_MOL_K * nasa_properties["s_R"]
            assert value["janaf_G_app_J_mol"] == pytest.approx(
                janaf_g, abs=0.001
            )
            comparison_g = (
                value["nasa_header_anchored_G_app_J_mol"]
                if comparison["nasa_record_id"] in {
                    "NG-1701", "NG-2013", "NG-1325", "NG-1917", "NG-2029"
                }
                else value["nasa_G_app_J_mol"]
            )
            assert comparison_g == pytest.approx(
                nasa_g
                if comparison["nasa_record_id"] not in {
                    "NG-1701", "NG-2013", "NG-1325", "NG-1917", "NG-2029"
                }
                else header_g,
                abs=0.001,
            )
            assert value["nasa_minus_janaf_J_mol"] == pytest.approx(
                comparison_g - janaf_g, abs=0.001
            )


def test_sc2o3_source_disagreement_bands_recompute_from_published_nodes() -> None:
    provenance = yaml.safe_load(PROVENANCE_PATH.read_text(encoding="utf-8"))
    comparison = next(
        row for row in provenance["source_comparisons"]["rows"]
        if row["species_name"] == "Sc2O3(l)"
    )
    differences = {
        int(row["T_K"]): abs(float(row["nasa_minus_pankratz_J_mol"]))
        for row in comparison["values"]
    }
    rows = [
        row for row in provenance["rows"]
        if row["table"] == "condensate"
        and row["species_name"] == "Sc2O3(l)"
        and row.get("uncertainty_band", {}).get("kind")
        == "measured_G_app_difference_from_Pankratz_Bulletin_672"
    ]
    assert len(rows) == 1
    for row in rows:
        values = row["uncertainty_band"]["values_J_per_mol"]
        low, high = map(int, row["T_range_K"])
        assert set(map(int, values)) == {
            temperature for temperature in differences
            if low <= temperature <= high
        }
        for temperature, value in values.items():
            assert float(value) == pytest.approx(differences[int(temperature)], abs=0.001)


def test_nio_gas_uncertainty_band_recomputes_from_bulletin_and_nasa() -> None:
    provenance = yaml.safe_load(PROVENANCE_PATH.read_text(encoding="utf-8"))
    bulletin = json.loads(
        (PUBLISHED_DATA / "nio-bulletin-672.json").read_text(encoding="utf-8")
    )
    nasa = _nasa_record("NG-0926")
    nasa_h298 = R_J_MOL_K * 298.15 * build_gas_tables._nasa7_properties(
        nasa, 298.15
    )["h_rt"]
    records = {
        row["T_interval"]: row
        for row in provenance["rows"]
        if row["table"] == "gas" and row["species_name"] == "NiO(g)"
    }
    for interval, temperatures in {
        1: (1500, 1800, 2000, 2500, 3000),
        2: (1200, 1300, 1400, 1500),
    }.items():
        band = records[interval]["uncertainty_band"]["values_J_per_mol"]
        assert tuple(map(int, band)) == temperatures
        for temperature in temperatures:
            comparison_temperature = min(temperature, 2000)
            node = next(
                row for row in bulletin["nodes"]
                if row[0] == comparison_temperature
            )
            bulletin_g = (
                float(bulletin["formation_enthalpy_298_kcal_mol"]) * 4184.0
                + float(node[3]) * 4184.0
                - comparison_temperature * float(node[2]) * 4.184
            )
            nasa_properties = build_gas_tables._nasa7_properties(
                nasa, comparison_temperature
            )
            nasa_g = float(nasa["delta_f_H_298_15"]["value"]) + (
                R_J_MOL_K * comparison_temperature * nasa_properties["h_rt"]
                - nasa_h298
            ) - comparison_temperature * R_J_MOL_K * nasa_properties["s_R"]
            gap = abs(nasa_g - bulletin_g)
            assert band[temperature] == pytest.approx(gap, abs=0.001)


def test_ni_co_mixed_source_parent_bands_recompute_from_compilations() -> None:
    """The disclosure bands follow the independent gas compilation values.

    Premise: the same TSIV liquid parent appears on both sides of each source
    comparison and cancels, leaving the selected gas G_app minus its TSIV
    counterpart. JANAF/NASA use 1 bar; TSIV uses 1 atm, with no conversion in
    these stored bands because the runtime source values retain those printed
    conventions. Algebra: G_app = dfH298 + (H(T)-H(298.15))-T*S(T). The TSIV
    Phi equation is checked against each printed knot; its tabulated H/S nodes
    supply G_app. Unit check: all gaps are J/mol. Sanity: the 2000 K gaps are
    about 5.9, 9.5, and 2.5 kJ/mol for Ni, NiO, and Co respectively.
    """
    provenance = yaml.safe_load(PROVENANCE_PATH.read_text(encoding="utf-8"))
    gas_rows = [
        row for row in provenance["rows"]
        if row["table"] == "gas" and row["species_name"] in {
            "Ni(g)", "NiO(g)", "Co(g)"
        }
    ]
    source_rows = {
        "Ni(g)": ("tsiv-table-2023.json", "janaf", "Ni-005"),
        "Co(g)": ("tsiv-table-1332.json", "janaf", "Co-005"),
        "NiO(g)": ("tsiv-table-2027.json", "nasa", "NG-0926"),
    }
    knots = (1200, 1500, 1800, 2000, 2500, 3000)
    published_provenance = yaml.safe_load(
        (PUBLISHED_DATA / "PROVENANCE.yaml").read_text(encoding="utf-8")
    )

    for species, (tsiv_filename, selected_kind, selected_id) in source_rows.items():
        tsiv_path = PUBLISHED_DATA / tsiv_filename
        tsiv = json.loads(tsiv_path.read_text(encoding="utf-8"))
        source_metadata = next(
            item for item in published_provenance["records"]
            if item["record_id"] == tsiv["record_id"]
        )
        assert source_metadata["path"] == f"data-src/published-thermo/{tsiv_filename}"
        assert source_metadata["source_sha256"] == hashlib.sha256(
            tsiv_path.read_bytes()
        ).hexdigest()
        tsiv_nodes = {float(row[0]): row for row in tsiv["nodes"]}
        h298_h0 = float(tsiv_nodes[298.15][4])

        def tsiv_g(temperature: float) -> float:
            node = tsiv_nodes[float(temperature)]
            # TSIV tabulates H-H(0); subtract its 298 K value to get H-H(298).
            return (
                float(tsiv["formation_enthalpy_298_kj_mol"]["value"])
                + float(node[4]) - h298_h0
            ) * 1000.0 - temperature * float(node[3])

        def tsiv_phi(temperature: float) -> float:
            function = next(
                item for item in tsiv["phi_functions"]
                if float(item["T_range_K"][0]) <= temperature
                <= float(item["T_range_K"][1])
            )
            c = list(map(float, function["coefficients"]))
            x = temperature * 1.0e-4
            return (
                c[0] + c[1] * math.log(x) + c[2] / x**2 + c[3] / x
                + c[4] * x + c[5] * x**2 + c[6] * x**3
            )

        if selected_kind == "janaf":
            selected_g = lambda temperature: _janaf_apparent_gibbs(
                selected_id, temperature
            )
        else:
            card = _nasa_record(selected_id)
            card_h298 = R_J_MOL_K * 298.15 * build_gas_tables._nasa7_properties(
                card, 298.15
            )["h_rt"]

            def selected_g(temperature: float) -> float:
                properties = build_gas_tables._nasa7_properties(card, temperature)
                return float(card["delta_f_H_298_15"]["value"]) + (
                    R_J_MOL_K * temperature * properties["h_rt"] - card_h298
                ) - temperature * R_J_MOL_K * properties["s_R"]

        rows = [row for row in gas_rows if row["species_name"] == species]
        assert len(rows) == 2
        for row in rows:
            low, high = map(float, row["T_range_K"])
            row_knots = tuple(t for t in knots if low <= t <= high)
            regular = row.get("uncertainty_band", {})
            bands = [regular, *row.get("additional_uncertainty_bands", [])]
            band = next(
                item for item in bands
                if item.get("kind") == "mixed_source_parent_scale_inconsistency"
            )
            assert band["comparison_record"] == tsiv["record_id"]
            assert set(map(int, band["values_J_per_mol"])) == set(row_knots)
            assert "1 bar" in band["standard_state_note"]
            assert "1 atm" in band["standard_state_note"]
            for temperature in row_knots:
                # Phi fits are printed to limited precision; the source table
                # H/S nodes remain the values used in the Gibbs comparison.
                assert tsiv_phi(temperature) == pytest.approx(
                    float(tsiv_nodes[float(temperature)][2]), abs=0.01
                )
                gap = abs(selected_g(temperature) - tsiv_g(temperature))
                assert band["values_J_per_mol"][temperature] == pytest.approx(
                    gap, abs=0.001
                )


def test_na_k_gas_anchor_and_parent_scale_bands_recompute() -> None:
    provenance = yaml.safe_load(PROVENANCE_PATH.read_text(encoding="utf-8"))
    gas_rows = [
        row for row in provenance["rows"]
        if row["table"] == "gas" and row["species_name"] in {"Na2O(g)", "K2O(g)"}
    ]
    lh84 = yaml.safe_load((LH84_DATA / "lh84.yaml").read_text(encoding="utf-8"))
    lh84_rows = {row["species_name"]: row for row in lh84["rows"]}
    temperatures = (1500, 1800, 2000, 2500, 3000)

    na_janaf = _record("Na-013")
    na_source_nodes = _complete_rows("Na-013")
    na_reference = next(row for row in na_source_nodes if row["temperature"] == 298.15)
    na_thermal_nodes = build_gas_tables._recover_thermal_transition_rows(
        na_janaf["table"]
    )

    def janaf_na_g(temperature: float) -> float:
        row = next(
            (row for row in na_source_nodes if row["temperature"] == temperature),
            None,
        )
        if row is None:
            row = next(
                row for row in na_thermal_nodes if row["temperature"] == temperature
            )
        return (
            na_reference["formation_enthalpy"] + row["enthalpy_increment"]
        ) * 1000.0 - temperature * row["entropy"]

    def nasa_card_g(record_id: str, temperature: float) -> float:
        card = _nasa_record(record_id)
        properties = build_gas_tables._nasa7_properties(card, temperature)
        card_h298 = R_J_MOL_K * 298.15 * build_gas_tables._nasa7_properties(
            card, 298.15
        )["h_rt"]
        return float(card["delta_f_H_298_15"]["value"]) + (
            R_J_MOL_K * temperature * properties["h_rt"] - card_h298
        ) - temperature * R_J_MOL_K * properties["s_R"]

    def lh84_anchored_card_g(species: str, record_id: str, temperature: float) -> float:
        card = _nasa_record(record_id)
        anchor = lh84_rows[species]
        at_temperature = build_gas_tables._nasa7_properties(card, temperature)
        at_298 = build_gas_tables._nasa7_properties(card, 298.15)
        h_increment = R_J_MOL_K * (
            temperature * at_temperature["h_rt"] - 298.15 * at_298["h_rt"]
        )
        entropy = R_J_MOL_K * (
            anchor["S_over_R"]["value"]
            + at_temperature["s_R"] - at_298["s_R"]
            + math.log(1.01325)
        )
        return (
            R_J_MOL_K * 1000.0 * anchor["dfH_over_R_kK"]["value"]
            + h_increment - temperature * entropy
        )

    na_liquid_row = next(
        row for row in provenance["rows"]
        if row["table"] == "condensate"
        and row["species_name"] == "Na2O(l)"
        and row.get("supersedes")
    )
    lh84_liquid = na_liquid_row["supersedes"]
    liquid_coefficients = lh84_liquid["dG_coefficients"]

    def lh84_na_liquid_g(temperature: float) -> float:
        tau = temperature / 1000.0
        polynomial = sum(
            coefficient * tau**power
            for power, coefficient in enumerate(liquid_coefficients)
        )
        return (
            -R_J_MOL_K * temperature * polynomial
            + R_J_MOL_K * lh84_liquid["dH298_R"] * 1000.0
        )

    expected = {
        "Na2O(g)": {
            "parent_scale_liquid_G": lambda temperature: abs(
                lh84_na_liquid_g(temperature) - janaf_na_g(temperature)
            ),
            "card_G_parent_scale_inconsistency": lambda temperature: abs(
                nasa_card_g("NG-0905", temperature)
                - lh84_anchored_card_g("Na2O(g)", "NG-0905", temperature)
            ),
        },
        "K2O(g)": {
            "card_G_parent_scale_inconsistency": lambda temperature: abs(
                nasa_card_g("NG-0760", temperature)
                - lh84_anchored_card_g("K2O(g)", "NG-0760", temperature)
            ),
        },
    }
    assert len(gas_rows) == 4
    assert {row["species_name"] for row in gas_rows} == set(expected)
    seen = set()
    for row in gas_rows:
        species = row["species_name"]
        interval = tuple(map(int, row["T_range_K"]))
        band_types = expected[species]
        bands = [
            row.get("uncertainty_band", {}),
            *row.get("additional_uncertainty_bands", []),
        ]
        row_knots = tuple(t for t in temperatures if interval[0] <= t <= interval[1])
        for band_type, calculate in band_types.items():
            band = next(item for item in bands if item.get("basis") == band_type)
            assert set(map(int, band["values_J_per_mol"])) == set(row_knots)
            for temperature in row_knots:
                assert band["values_J_per_mol"][temperature] == pytest.approx(
                    calculate(temperature), abs=0.001
                )
                seen.add((species, band_type, temperature))
    assert len(seen) == 15


def test_mixed_compilation_pairs_have_disclosure_or_named_exception() -> None:
    """Every channel pair is sourced from provenance and has an explicit outcome."""
    provenance = yaml.safe_load(PROVENANCE_PATH.read_text(encoding="utf-8"))
    gas_rows = [row for row in provenance["rows"] if row["table"] == "gas"]
    parent_rows = [
        row for row in provenance["rows"] if row["table"] == "condensate"
    ]
    oxygen_rows = [
        row for row in gas_rows if row["species_name"] == "O2(g)"
    ]

    known_unbanded_mixed_source_channels = {
        **dict.fromkeys(
            ("Sc", "ScO", "ScO2", "U", "UO", "UO3"),
            "NASA gas and parent liquid use the same adopted Gurvich compilation family.",
        ),
        **dict.fromkeys(
            ("Bi", "Bi2", "BiO", "Te", "Te2", "Se", "Se2", "SeO",
             "As", "As2", "As4", "Sb", "Sb2", "Sb4", "Ag"),
            "The residual is the 1 atm versus 1 bar standard-state shift only.",
        ),
        "Ga": "JANAF Ga(g) versus the NASA parent changes log10 K by at most 0.0023 dex.",
        "PbO2": "The all-NASA comparison changes log10 K by at most 0.0010 dex.",
        **dict.fromkeys(
            ("RbO", "GaO", "Ga2O", "Ge", "GeO", "In", "InO", "In2O",
             "Sn", "SnO2"),
            "NASA gas and parent agree to fit precision; NASA O2 is not vendored, so its thermal difference is unmeasured.",
        ),
        **dict.fromkeys(
            ("Mn", "Rb", "Cs", "CsO", "Cs2O", "Zn", "Cd"),
            "No single-compilation table is vendored for the gas and runtime parent.",
        ),
    }

    def compilation(row: dict) -> str:
        source = " ".join(
            str(row.get(key, ""))
            for key in (
                "source_path", "authority", "table_id", "source_table_id",
                "source",
            )
        ).lower()
        if "janaf" in source:
            return "JANAF"
        if "nasa" in source:
            return "NASA"
        if "tsiv" in source or "ivtan" in source:
            return "TSIV"
        if "pankratz" in source or "bulletin" in source or "p672" in source:
            return "Pankratz"
        if any(label in source for label in ("construction", "estimated", "lam1987")):
            return "construction"
        return "other"

    unbanded_mixed = set()
    for species, (parent, _n_gas, n_o2) in _SF04_REACTIONS.items():
        if not parent:
            continue
        rows = [row for row in gas_rows if row["species_name"] == f"{species}(g)"]
        parents = [
            row for row in parent_rows
            if row["species_name"] == f"{parent}(l)"
        ]
        if not rows or not parents:
            continue
        parent_compilations = {compilation(row) for row in parents}
        gas_compilations = {compilation(row) for row in rows}
        pair_compilations = gas_compilations | parent_compilations
        if n_o2 and oxygen_rows:
            pair_compilations |= {compilation(row) for row in oxygen_rows}
        if len(pair_compilations) <= 1:
            continue
        disclosed = all(
            any(
                band.get("kind") == "mixed_source_parent_scale_inconsistency"
                for band in [
                    row.get("uncertainty_band", {}),
                    *row.get("additional_uncertainty_bands", []),
                ]
            )
            for row in rows
        )
        if not disclosed:
            unbanded_mixed.add(species)

    assert unbanded_mixed == set(known_unbanded_mixed_source_channels)
    assert all(known_unbanded_mixed_source_channels.values())


def test_mixed_source_cluster_bands_recompute_from_parent_scales() -> None:
    """Cluster bands follow the runtime liquid parent at every row knot."""
    provenance = yaml.safe_load(PROVENANCE_PATH.read_text(encoding="utf-8"))
    comparisons = {
        row["species_name"]: row
        for row in provenance["source_comparisons"]["rows"]
    }
    gas_rows = [
        row for row in provenance["rows"]
        if row["table"] == "gas"
        and row["species_name"].startswith("(")
        and row.get("uncertainty_band", {}).get("kind")
        == "mixed_source_parent_scale_inconsistency"
    ]
    runtime_parents = pd.read_csv(GAS_DATA / "condensate.csv").set_index(
        "species_name", drop=False
    )
    knots = (1200, 1500, 1800, 2000, 2500, 3000)
    cluster_pattern = re.compile(r"^\((MoO3|WO3)\)([2-5])\(g\)$")
    assert len(gas_rows) == 10
    assert {row["species_name"] for row in gas_rows} == {
        "(MoO3)2(g)", "(MoO3)3(g)", "(MoO3)4(g)", "(MoO3)5(g)",
        "(WO3)5(g)",
    }
    seen = set()
    for row in gas_rows:
        match = cluster_pattern.fullmatch(row["species_name"])
        assert match is not None
        parent_formula, number = match.groups()
        parent = f"{parent_formula}(l)"
        n = int(number)
        identity = (row["species_name"], row["T_interval"])
        assert identity not in seen
        seen.add(identity)
        comparison = comparisons[parent]
        card = _nasa_record(comparison["nasa_record_id"])
        card_low = float(card["intervals"][0]["T_min_K"]["value"])
        card_high = float(card["intervals"][-1]["T_max_K"]["value"])
        row_low, row_high = map(float, row["T_range_K"])
        row_knots = {t for t in knots if row_low <= t <= row_high}
        undefined = {t for t in row_knots if not card_low <= t <= card_high}
        expected_knots = row_knots - undefined
        band = row["uncertainty_band"]
        assert band["basis"] == "card_G_parent_scale_inconsistency"
        assert band["per_formula"] == "cluster formula"
        assert band["factor_from_parent_formula"] == n
        assert set(map(int, band["values_J_per_mol"])) == expected_knots
        assert set(band.get("undefined_temperatures_K", [])) == undefined
        if row["species_name"] == "(WO3)5(g)" and row["T_interval"] == 1:
            assert band["pressure_shift_at_2000_K"] == "+1.34 dex"
        if undefined:
            assert "no card extrapolation is used" in band["undefined_reason"]
        for temperature in expected_knots:
            props = build_gas_tables._nasa7_properties(card, temperature)
            nasa_card_g = R_J_MOL_K * temperature * (
                props["h_rt"] - props["s_R"]
            )
            parent_row = _nearest_interval_row(
                runtime_parents, parent, temperature, allow_extrapolation=True
            )
            runtime_parent_g = _lamor_gibbs(temperature, parent_row)
            expected = n * abs(nasa_card_g - runtime_parent_g)
            assert band["values_J_per_mol"][temperature] == pytest.approx(
                expected, abs=0.002
            )
    assert len(seen) == 10


def _janaf_log10_kf(table_id: str, temperature: float) -> float:
    text_path = JANAF_DATA / f"{table_id}.txt"
    if text_path.is_file():
        for line in text_path.read_text(encoding="utf-8").splitlines()[2:]:
            columns = line.split("\t")
            if columns and float(columns[0]) == temperature:
                assert len(columns) >= 8 and columns[7].strip()
                return float(columns[7])
        raise AssertionError(f"{table_id} has no log Kf at {temperature} K")
    for row in _record(table_id)["table"]["values"]:
        if float(row["temperature"]["value"]) == temperature:
            value = row["log10_formation_equilibrium_constant"]["value"]
            assert value is not None
            return float(value)
    raise AssertionError(f"{table_id} has no JANAF row at {temperature} K")


def _janaf_log10_kf_nodes(table_id: str) -> set[float]:
    text_path = JANAF_DATA / f"{table_id}.txt"
    if text_path.is_file():
        nodes = set()
        for line in text_path.read_text(encoding="utf-8").splitlines()[2:]:
            columns = line.split("\t")
            if len(columns) >= 8 and columns[7].strip():
                nodes.add(float(columns[0]))
        return nodes
    return {
        float(row["temperature"]["value"])
        for row in _record(table_id)["table"]["values"]
        if row.get("log10_formation_equilibrium_constant", {}).get("value")
        is not None
    }


def test_fitted_ion_equilibria_match_janaf_log_kf_nodes() -> None:
    pack = load_gas_datapack()
    ion_table_ids = {
        species.removesuffix("(g)"): table_id
        for species, table_id, *_ in (
            *build_gas_tables.ION_GAS_TEXT_SOURCES,
            *build_gas_tables.ION_GAS_YAML_SOURCES,
            *build_gas_tables.TRACE_ION_GAS_TEXT_SOURCES,
            *build_gas_tables.TRACE_ION_GAS_YAML_SOURCES,
        )
    }
    reactions = [
        ("Na", "Na+", 1),
        ("K", "K+", 1),
        ("Ca", "Ca+", 1),
        ("Li", "Li+", 1),
        ("Rb", "Rb+", 1),
        ("Pb", "Pb+", 1),
        ("Ga", "Ga+", 1),
        ("B", "B+", 1),
        ("Mo", "Mo+", 1),
        ("W", "W+", 1),
        ("Cs", "Cs+", 1),
        ("Cu", "Cu-", -1),
        ("Mo", "Mo-", -1),
        ("W", "W-", -1),
        ("Na", "Na-", -1),
        ("K", "K-", -1),
        ("O", "O-", -1),
        ("Al", "Al-", -1),
        ("Fe", "Fe-", -1),
        ("Si", "Si-", -1),
        ("Ti", "Ti-", -1),
        ("O2", "O2-", -1),
        ("AlO", "AlO-", -1),
        ("AlO2", "AlO2-", -1),
        ("KO", "KO-", -1),
        ("NaO", "NaO-", -1),
        ("Cr", "Cr-", -1),
        ("V", "V-", -1),
        ("Nb", "Nb-", -1),
        ("Li", "Li-", -1),
        ("LiO", "LiO-", -1),
        ("Rb", "Rb-", -1),
        ("Pb", "Pb-", -1),
        ("Ga", "Ga-", -1),
        ("B", "B-", -1),
        ("BO2", "BO2-", -1),
        ("Cs", "Cs-", -1),
        ("Cu", "Cu-", -1),
    ]
    source_ids = set(ion_table_ids.values())
    source_ids.update(
        str(
            _nearest_interval_row(
                pack.gas_df,
                f"{neutral}(g)",
                1500.0,
                allow_extrapolation=False,
            )["Ref"]
        )
        for neutral, _charged, _charge in reactions
    )
    source_nodes = set.intersection(
        *(_janaf_log10_kf_nodes(table_id) for table_id in source_ids)
    )
    temperatures = [
        temperature
        for temperature in np.arange(1500.0, 3001.0, 100.0)
        if temperature in source_nodes
    ]
    assert len(temperatures) >= 10
    for temperature in temperatures:
        electron_row = _nearest_interval_row(
            pack.gas_df, "e-(g)", temperature, allow_extrapolation=False
        )
        electron_g = _janaf_gibbs(temperature, electron_row)
        electron_table = ion_table_ids["e-"]
        electron_log_kf = _janaf_log10_kf(electron_table, temperature)
        for neutral, charged, charge_sign in reactions:
            neutral_row = _nearest_interval_row(
                pack.gas_df,
                f"{neutral}(g)",
                temperature,
                allow_extrapolation=False,
            )
            charged_row = _nearest_interval_row(
                pack.gas_df,
                f"{charged}(g)",
                temperature,
                allow_extrapolation=False,
            )
            neutral_g = _janaf_gibbs(temperature, neutral_row)
            charged_g = _janaf_gibbs(temperature, charged_row)
            if charge_sign > 0:
                delta_g = charged_g + electron_g - neutral_g
                expected_log_k = (
                    _janaf_log10_kf(ion_table_ids[charged], temperature)
                    + electron_log_kf
                    - _janaf_log10_kf(str(neutral_row["Ref"]), temperature)
                )
            else:
                delta_g = charged_g - neutral_g - electron_g
                expected_log_k = (
                    _janaf_log10_kf(ion_table_ids[charged], temperature)
                    - _janaf_log10_kf(str(neutral_row["Ref"]), temperature)
                    - electron_log_kf
                )
            fitted_log_k = -delta_g / (
                R_J_MOL_K * temperature * math.log(10.0)
            )
            assert fitted_log_k == pytest.approx(
                expected_log_k, abs=0.002
            ), (neutral, charged, temperature)


def test_nasa_trace_ions_match_their_cards_and_neutral_ionisation_reactions() -> None:
    pack = load_gas_datapack()
    neutral_tables = {
        source[0].removesuffix("(g)"): source[1]
        for source in (
            *build_gas_tables.TRACE_JANAF_GAS_SOURCES,
            *build_gas_tables.TRACE_NASA_GAS_SOURCES,
        )
    }

    def source_gibbs(table_id: str, temperature: float) -> float:
        if table_id.startswith("NG-"):
            record = _nasa_record(table_id)
            reference_h = float(record["delta_f_H_298_15"]["value"])
            at_298 = build_gas_tables._nasa7_properties(record, 298.15)
            at_t = build_gas_tables._nasa7_properties(record, temperature)
            enthalpy = reference_h + R_J_MOL_K * (
                temperature * at_t["h_rt"] - 298.15 * at_298["h_rt"]
            )
            entropy = R_J_MOL_K * at_t["s_R"]
            return enthalpy - temperature * entropy
        rows = build_gas_tables._janaf_text_source_rows(JANAF_DATA, table_id)
        reference_h = rows[298.15]["formation_enthalpy"]
        node = rows[temperature]
        return (
            (reference_h + node["enthalpy_increment"]) * 1000.0
            - temperature * node["entropy"]
        )

    for ion_name, ion_table, neutral, *_ in build_gas_tables.TRACE_ION_GAS_NASA_SOURCES:
        ion = ion_name.removesuffix("(g)")
        neutral = neutral.removesuffix("E-1")
        neutral_table = neutral_tables[neutral]
        for temperature in range(1200, 3001, 100):
            charged_row = _nearest_interval_row(
                pack.gas_df, f"{ion}(g)", temperature, allow_extrapolation=False
            )
            neutral_row = _nearest_interval_row(
                pack.gas_df,
                f"{neutral}(g)",
                temperature,
                allow_extrapolation=False,
            )
            electron_row = _nearest_interval_row(
                pack.gas_df, "e-(g)", temperature, allow_extrapolation=False
            )
            fitted_delta_g = (
                _janaf_gibbs(temperature, charged_row)
                + _janaf_gibbs(temperature, electron_row)
                - _janaf_gibbs(temperature, neutral_row)
            )
            source_delta_g = (
                source_gibbs(ion_table, temperature)
                + source_gibbs("D-020", temperature)
                - source_gibbs(neutral_table, temperature)
            )
            # This difference sums the charged, electron, and neutral fit
            # residuals; their maximum over this grid is just over 2 J/mol.
            # Ag+'s 1200-1500 K fit residual reaches 10.418 J/mol, so its
            # source-specific reaction check uses a 10.5 J/mol bound.
            tolerance = 10.5 if ion == "Ag+" else 2.1
            assert abs(fitted_delta_g - source_delta_g) <= tolerance
def test_cu2o_liquid_source_comparison_uses_gibbs_energy() -> None:
    nasa = _nasa_record("NG-1844")
    nasa_gibbs = {}
    for temperature in (1800.0, 2000.0, 2500.0, 3000.0):
        properties = build_gas_tables._nasa7_properties(nasa, temperature)
        nasa_gibbs[temperature] = R_J_MOL_K * temperature * (
            properties["h_rt"] - properties["s_R"]
        )
    janaf_rows = build_gas_tables._janaf_text_source_rows(JANAF_DATA, "Cu-020")
    reference_enthalpy = janaf_rows[298.15]["formation_enthalpy"]
    janaf_gibbs = {
        temperature: (
            reference_enthalpy + janaf_rows[temperature]["enthalpy_increment"]
        )
        * 1000.0
        - temperature * janaf_rows[temperature]["entropy"]
        for temperature in (1800.0, 2000.0)
    }
    differences = {
        temperature: nasa_gibbs[temperature] - janaf_gibbs[temperature]
        for temperature in (1800.0, 2000.0)
    }
    assert differences[1800.0] == pytest.approx(-12.839699674514122, abs=0.01)
    assert differences[2000.0] == pytest.approx(-7.4065839719, abs=0.01)
    assert nasa_gibbs[2500.0] == pytest.approx(-682150.2608, abs=0.1)
    assert nasa_gibbs[3000.0] == pytest.approx(-841176.5626, abs=0.1)
    assert 2000.0 in janaf_rows
    assert 2500.0 not in janaf_rows
    assert 3000.0 not in janaf_rows
    provenance = yaml.safe_load(PROVENANCE_PATH.read_text(encoding="utf-8"))
    tail = next(
        row
        for row in provenance["rows"]
        if row["table"] == "condensate"
        and row["species_name"] == "Cu2O(l)"
        and row["T_range_K"] == [2000, 2500]
    )
    assert tail["authority"] == "janaf_fitted"
    assert tail["method"] == "janaf_anchored_nasa_tail_fit"
    assert tail["tail_source_path"] == "data-src/nasa-glenn/NG-1844.json"


def test_nasa_anchored_rows_match_direct_card_evaluation() -> None:
    _assert_nasa_anchored_rows_match_direct_card_evaluation()


def test_nasa_header_columns_and_molecular_weights_match_formulas() -> None:
    """Parse CEA's five fixed composition slots and verify every card mass."""
    import openimcc.gas as gas_module

    records = [
        _nasa_record(path.stem)
        for path in sorted(NASA_DATA.glob("NG-*.json"))
    ]
    assert _nasa_record("NG-0074")["delta_f_H_298_15"]["value"] == 284900.0
    assert _nasa_record("NG-0085")["delta_f_H_298_15"]["value"] == 1022093.730
    # NASA CEA's published mass convention uses Ge=72.64 and O=15.9994.
    # These source standard atomic weights make its GeO/GeO2 card masses
    # consistent to the requested 0.01 g/mol tolerance.
    atomic_masses = {
        **gas_module._ATOMIC_MASS_G_MOL,
        "Ge": 72.64,
        "O": 15.9994,
        "Zn": 65.39,
        "Mo": 95.94,
        "W": 183.84,
        "Ba": 137.327,
        "Sr": 87.62,
        "Zr": 91.224,
    }
    for record in records:
        formula = record["formula"]
        neutral_formula = (
            formula.removesuffix("+").removesuffix("-").removesuffix("E-1")
        )
        parts = gas_module._formula_atoms(neutral_formula)
        parts.pop("E", None)  # CEA's E-1 charge marker carries no atomic mass.
        formula_mass = sum(
            atomic_masses[element] * number for element, number in parts.items()
        )
        assert abs(record["molecular_weight"]["value"] - formula_mass) <= 0.01, (
            record["record_id"],
            formula,
            formula_mass,
            record["molecular_weight"]["value"],
        )

    new_record_ids = (
        "NG-1063", "NG-1074", "NG-11154", "NG-1122", "NG-1130",
        "NG-12403", "NG-12434", "NG-1266", "NG-12662", "NG-1274",
        "NG-1282", "NG-4940", "NG-4951", "NG-5066", "NG-5178",
        "NG-5186", "NG-5197", "NG-5331", "NG-5339", "NG-6005",
        "NG-6016", "NG-6131", "NG-6243",
        "NG-4060", "NG-4071", "NG-10887", "NG-10898", "NG-0074", "NG-0085", "NG-0086",
        "NG-0824", "NG-0825", "NG-0826", "NG-0827", "NG-0828", "NG-0829",
        "NG-0831", "NG-0832", "NG-0833", "NG-0834", "NG-1701",
        "NG-1245", "NG-1246", "NG-1247", "NG-1249", "NG-1251", "NG-1253",
        "NG-1255", "NG-1256", "NG-1257", "NG-1258", "NG-2013",
        "NG-0165", "NG-0166", "NG-0178", "NG-1168", "NG-1169",
        "NG-1181", "NG-1263", "NG-1264", "NG-1267", "NG-1269",
        "NG-1325", "NG-1917", "NG-2029",
    )
    manifest = yaml.safe_load(
        (NASA_DATA / "PROVENANCE.yaml").read_text(encoding="utf-8")
    )
    manifest_rows = {row["record_id"]: row for row in manifest["records"]}
    for table_id in new_record_ids:
        record = _nasa_record(table_id)
        header = record["source_text"]["header_line"]
        for index, composition in enumerate(record["composition"]):
            element_raw = header[10 + 8 * index:12 + 8 * index]
            count_raw = header[12 + 8 * index:18 + 8 * index]
            assert composition["element_as_published"] == element_raw
            assert composition["count_as_published"] == count_raw
        tail = header[50:].split()
        assert record["phase_flag"] == int(tail[0])
        assert record["phase_flag_as_published"] == header[50:52]
        assert record["molecular_weight"]["as_published"] == tail[1]
        assert record["delta_f_H_298_15"]["as_published"] == tail[2]
        locator = record["source_locator"]
        expected_name_lines = {
            "NG-0074": 76,
            "NG-0085": 87,
            "NG-0086": 98,
            "NG-4060": 4062,
            "NG-4071": 4073,
            "NG-10887": 10889,
            "NG-10898": 10900,
            "NG-0824": 7015, "NG-0825": 7026, "NG-0826": 7037,
            "NG-0827": 7048, "NG-0828": 7059, "NG-0829": 7067,
            "NG-0831": 7083, "NG-0832": 7091, "NG-0833": 7099,
            "NG-0834": 7107, "NG-1701": 13503,
            "NG-1245": 10743, "NG-1246": 10754, "NG-1247": 10765,
            "NG-1249": 10784, "NG-1251": 10803, "NG-1253": 10819,
            "NG-1255": 10835, "NG-1256": 10843, "NG-1257": 10851,
            "NG-1258": 10859, "NG-2013": 15372,
            "NG-0165": 1419, "NG-0166": 1430, "NG-0178": 1529,
            "NG-1168": 9983, "NG-1169": 9994, "NG-1181": 10111,
            "NG-1263": 10911, "NG-1264": 10922, "NG-1267": 10952,
            "NG-1269": 10974, "NG-1325": 11320, "NG-1917": 14790,
            "NG-2029": 15461,
        }
        assert locator["name_line"] == expected_name_lines.get(
            table_id, int(table_id.removeprefix("NG-")) + 1
        )
        assert locator["header_line"] == locator["name_line"] + 1
        source_path = NASA_DATA / f"{table_id}.json"
        assert manifest_rows[table_id]["source_sha256"] == hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest()
        assert manifest_rows[table_id]["source_locator"] == (
            f"thermo.inp lines {locator['name_line']}-{locator['end_line']}"
        )


# Ground-term fine-structure levels (statistical weight 2J+1, energy in cm^-1)
# from the NIST Atomic Spectra Database, for the neutral atom and its cation.
# Sn and Ge also list the low 1D2 level. Higher levels contribute below the
# test tolerance at 2500 K.
_SAHA_LEVELS = {
    "Sc": (
        6.56149,
        (
            (4, 0.0), (6, 168.34), (4, 11519.99), (6, 11557.69),
            (8, 11610.28), (10, 11677.38), (6, 14926.07), (8, 15041.92),
            (4, 15672.58), (6, 15756.57), (8, 15881.75), (10, 16026.62),
            (2, 16009.77), (4, 16021.82), (6, 16141.06), (8, 16210.85),
            (6, 16022.73), (4, 16096.90), (2, 18504.06), (4, 18515.69),
            (6, 18571.41), (2, 18711.02), (4, 18855.74),
        ),
        (
            (3, 0.0), (5, 67.72), (7, 177.76), (5, 2540.95),
            (5, 4802.87), (7, 4883.57), (9, 4987.79), (5, 10944.56),
            (1, 11736.36), (1, 12074.10), (3, 12101.50), (5, 12154.42),
            (9, 14261.32), (1, 25955.2), (5, 27443.71), (7, 27602.45),
            (9, 27841.35), (3, 27917.78), (5, 28021.29), (7, 28161.17),
            (1, 29736.27), (3, 29742.16), (5, 29823.93), (3, 32349.98),
        ),
    ),  # NIST Handbook Tables 5/6 and NIST ASD ionization-energy data.
    "Na": (5.13907696, ((2, 0.0),), ((1, 0.0),)),
    "K": (4.3406637, ((2, 0.0),), ((1, 0.0),)),
    "Ca": (6.1131553, ((1, 0.0),), ((2, 0.0),)),
    "Li": (5.3917148, ((2, 0.0),), ((1, 0.0),)),
    "Rb": (4.177128, ((2, 0.0),), ((1, 0.0),)),
    "Cs": (
        3.89390572743,
        ((2, 0.0), (2, 11178.2686), (4, 11732.3079), (4, 14499.2584), (6, 14596.8423)),
        ((1, 0.0),),
    ),  # NIST ASD: https://physics.nist.gov/PhysRefData/Handbook/Tables/cesiumtable5.htm
    "Pb": (
        7.4166798,
        ((1, 0.0), (3, 7819.263), (5, 10650.327)),
        ((2, 0.0), (4, 14081.074)),
    ),
    "Sn": (
        7.343917,
        ((1, 0.0), (3, 1691.806), (5, 3427.673), (5, 8612.955)),
        ((2, 0.0), (4, 4251.494)),
    ),
    "Ge": (
        7.89943,
        ((1, 0.0), (3, 557.134), (5, 1409.961), (5, 7125.299)),
        ((2, 0.0), (4, 1767.357)),
    ),
    "Ga": (5.999301, ((2, 0.0), (4, 826.19)), ((1, 0.0),)),
    "B": (8.29802, ((2, 0.0), (4, 15.287)), ((1, 0.0),)),
    # NIST Handbook ground levels and low-lying terms, tables 5 and 6.
    # Nickel IE is from NIST ASD; Co and Ni neutral/ion levels include the
    # complete listed terms below 17,000 cm^-1.
    "Ni": (
        7.639878,
        ((9, 0.0), (7, 1332.164), (5, 2216.550), (7, 204.787),
         (5, 879.816), (3, 1713.087), (5, 3409.937), (5, 13521.347),
         (1, 14728.840), (5, 15609.844), (3, 15734.001), (1, 16017.306)),
        ((6, 0.0), (4, 1506.94), (10, 8393.90), (8, 9330.04),
         (6, 10115.66), (4, 10663.89), (8, 13550.39), (6, 14995.57)),
    ),
    "Co": (
        7.88101,
        ((10, 0.0), (8, 816.000), (6, 1406.852), (4, 1809.313),
         (10, 3482.780), (8, 4142.631), (6, 4690.141), (4, 5075.788),
         (8, 7442.399), (6, 8460.783), (6, 13795.491), (4, 14036.241),
         (2, 14399.220), (6, 15184.013), (4, 15774.009), (2, 16195.610)),
        ((9, 0.0), (7, 950.324), (5, 1597.197), (11, 3350.494),
         (9, 4028.988), (7, 4560.789), (5, 4950.062), (3, 5204.698),
         (9, 9812.859), (7, 10708.330), (5, 11321.859)),
    ),
    # NIST Atomic Spectra Database Handbook tables: Zn I/II and Cd I/II have
    # non-degenerate ground levels; their first excited levels are above
    # 30,000 cm^-1. Bi I and II fine-structure levels below 17,100 cm^-1 are
    # included because they contribute at the 0.1% level at 2500 K.
    # https://physics.nist.gov/PhysRefData/Handbook/Tables/zinctable5.htm
    # https://physics.nist.gov/PhysRefData/Handbook/Tables/zinctable6.htm
    "Zn": (9.394199, ((1, 0.0),), ((2, 0.0),)),
    # https://physics.nist.gov/PhysRefData/Handbook/Tables/cadmiumtable5.htm
    # https://physics.nist.gov/PhysRefData/Handbook/Tables/cadmiumtable6.htm
    "Cd": (8.99382, ((1, 0.0),), ((2, 0.0),)),
    # https://physics.nist.gov/PhysRefData/Handbook/Tables/bismuthtable5.htm
    # https://physics.nist.gov/PhysRefData/Handbook/Tables/bismuthtable6.htm
    "Bi": (
        7.2855,
        ((4, 0.0), (4, 11419.039), (6, 15437.501), (2, 21660.914), (4, 33164.805)),
        ((1, 0.0), (3, 13325.401), (5, 17031.698)),
    ),
    # NIST Handbook Mo I/II and W I/II tables. Low levels and complete
    # fine-structure terms are included through the thermally relevant range.
    "Mo": (
        7.09243,
        ((7, 0.0), (5, 10768.332), (1, 10965.947), (3, 11142.784),
         (5, 11454.362), (7, 11858.499), (9, 12346.280)),
        ((6, 0.0), (2, 11783.36), (4, 12034.06), (6, 12417.28),
         (8, 12900.33), (10, 13460.70), (6, 15199.25), (8, 15330.56),
         (10, 15427.73), (12, 15446.97)),
    ),
    "W": (
        7.86403,
        ((1, 0.0), (3, 1670.29), (5, 3325.53), (7, 4830.00),
         (9, 6219.33), (7, 2951.29), (1, 9258.06), (3, 13307.10),
         (5, 19253.56)),
        ((2, 0.0), (4, 1518.829), (6, 3172.473), (8, 4716.278),
         (10, 6147.085), (6, 7420.261), (4, 8711.274), (6, 11301.024),
         (8, 13411.939), (10, 14857.160)),
    ),
    # NIST Handbook tables 5/6: Ba I/II and Sr I/II terms below 25,000 cm-1.
    "Ba": (
        5.211664,
        ((1, 0.0), (3, 9033.966), (5, 9215.501), (7, 9596.533),
         (5, 11395.350), (1, 12266.024), (3, 12636.623), (5, 13514.745),
         (3, 18060.261), (9, 22064.645), (7, 22947.423), (9, 23757.049),
         (5, 23074.387), (3, 24192.033), (5, 24531.513), (7, 24979.834)),
        ((2, 0.0), (4, 4873.852), (6, 5674.807), (2, 20261.561),
         (4, 21952.404)),
    ),
    "Sr": (
        5.6948672,
        ((1, 0.0), (1, 14317.520), (3, 14504.351), (5, 14898.563),
         (3, 18159.056), (5, 18218.795), (7, 18319.267), (5, 20149.700),
         (3, 21698.482)),
        ((2, 0.0), (4, 14555.90), (6, 14836.24), (2, 23715.19),
         (4, 24516.65)),
    ),
    # NIST Handbook tables 5/6: all listed Zr I/II terms below 20,000 cm-1;
    # the next Zr II manifold begins at 27,984 cm-1.
    "Zr": (
        6.63412,
        ((5, 0.0), (7, 570.41), (9, 1240.84), (5, 4186.11), (1, 4196.85),
         (3, 4376.28), (3, 4870.53), (5, 5023.41), (7, 5249.07),
         (9, 5540.54), (11, 5888.93), (5, 5101.68), (9, 8057.30),
         (11, 10885.36), (13, 11016.65), (7, 11258.38), (5, 11640.72),
         (7, 11956.33), (9, 12342.37), (7, 12503.44), (9, 12760.66),
         (11, 12772.78), (1, 13141.76), (3, 14123.01), (5, 14348.78),
         (7, 14697.03), (5, 14783.54), (7, 15201.26), (9, 15720.36),
         (11, 16316.96), (13, 16978.29), (5, 16296.51), (7, 16843.93),
         (9, 17556.26), (3, 16786.93), (5, 17059.61), (7, 17422.17),
         (9, 17833.73), (11, 18276.92), (1, 18976.36), (3, 19096.53),
         (5, 19323.84), (7, 19625.58), (9, 19833.78)),
        ((4, 0.0), (6, 314.673), (8, 763.428), (10, 1322.730),
         (4, 2572.372), (6, 2895.097), (8, 3299.732), (10, 3757.732),
         (4, 4248.165), (6, 4505.496), (2, 5724.150), (4, 6111.529),
         (6, 5752.868), (8, 6467.509)),
    ),
}


@pytest.mark.parametrize("element", tuple(_SAHA_LEVELS))
def test_every_m_plus_matches_saha_with_level_populations(element: str) -> None:
    """Check every packaged atomic cation against the Saha equation.

    For M = M+ + e-,
      Kp = (kB T / p0) * (2 Z+ / Z0) * (2 pi me kB T / h^2)^(3/2) * exp(-IE / kB T),
    with Z = sum_i g_i exp(-E_i / kB T) over the listed levels and the factor 2
    the electron spin weight. Units: kB T / p0 is m^3 and the translational
    term is m^-3, so Kp is dimensionless at p0 = 1 bar; level energies convert
    from cm^-1 with h c = 1.986445857e-23 J cm. Sanity: with single-level
    partition functions this is the ground-level Saha equation, which the
    alkali rows reproduce to better than 1%. Heavy p-block atoms need their
    fine-structure populations: a ground-term degeneracy (9 for 3P) would be
    wrong by a factor of 3 for Pb, whose 3P1 level lies 7819 cm^-1 up.
    The 2% check covers the omitted higher levels and fit residuals. Bi's
    independent thermochemical/spectroscopic source offset is pinned below;
    a source-consistency error such as the declined Cu+ and In+ tables is off
    by orders of magnitude.
    """
    ionisation_eV, neutral_levels, ion_levels = _SAHA_LEVELS[element]
    pack = load_gas_datapack()
    atomic_pairs = {
        neutral for neutral, ion in _GAS_IONIZATION_PAIRS if ion != "Cs2O+"
    }
    # Mn+ has no fine-structure level set in this spectroscopy-backed Saha
    # audit. Ag+ is checked against its NASA card and neutral-ionization
    # reaction, but no Ag level set is part of this audit.
    assert atomic_pairs - {"Mn", "Ag"} == set(_SAHA_LEVELS)
    k_b = 1.380649e-23
    h = 6.62607015e-34
    m_e = 9.1093837139e-31
    e_v = 1.602176634e-19
    h_c = 1.986445857e-23

    def partition(levels: tuple[tuple[int, float], ...], temperature: float) -> float:
        return sum(
            weight * math.exp(-h_c * energy / (k_b * temperature))
            for weight, energy in levels
        )

    temperatures = (1500.0, 2500.0, 3000.0) if element in {
        "Mo", "W", "Ba", "Sr", "Zr"
    } else (
        1500.0, 2500.0
    )
    for temperature in temperatures:
        ion_row = _nearest_interval_row(
            pack.gas_df, f"{element}+(g)", temperature, allow_extrapolation=False
        )
        neutral_row = _nearest_interval_row(
            pack.gas_df, f"{element}(g)", temperature, allow_extrapolation=False
        )
        electron_row = _nearest_interval_row(
            pack.gas_df, "e-(g)", temperature, allow_extrapolation=False
        )
        delta_g = (
            _janaf_gibbs(temperature, ion_row)
            + _janaf_gibbs(temperature, electron_row)
            - _janaf_gibbs(temperature, neutral_row)
        )
        fitted_k = math.exp(-delta_g / (R_J_MOL_K * temperature))
        saha_k = (
            k_b * temperature / 1.0e5
            * 2.0 * partition(ion_levels, temperature)
            / partition(neutral_levels, temperature)
            * (2.0 * math.pi * m_e * k_b * temperature / h**2) ** 1.5
            * math.exp(-ionisation_eV * e_v / (k_b * temperature))
        )
        if element == "Zn":
            # The source rail keeps Zn-006 (JANAF) over the CEA comparison
            # card. Check both the card's Saha consistency and that the
            # retained row preserves the independently measured G difference.
            nasa_record = _nasa_record("NG-10898")
            nasa_h298 = R_J_MOL_K * build_gas_tables.NASA_STANDARD_T_K * (
                build_gas_tables._nasa7_properties(
                    nasa_record, build_gas_tables.NASA_STANDARD_T_K
                )["h_rt"]
            )
            nasa_t = build_gas_tables._nasa7_properties(
                nasa_record, temperature
            )
            nasa_g = (
                float(nasa_record["delta_f_H_298_15"]["value"])
                + R_J_MOL_K * temperature * nasa_t["h_rt"]
                - nasa_h298
                - temperature * R_J_MOL_K * nasa_t["s_R"]
            )
            nasa_k = math.exp(
                -(
                    nasa_g
                    + _janaf_gibbs(temperature, electron_row)
                    - _janaf_gibbs(temperature, neutral_row)
                )
                / (R_J_MOL_K * temperature)
            )
            assert nasa_k / saha_k == pytest.approx(1.0, rel=0.02)
            assert fitted_k / saha_k == pytest.approx(
                nasa_k / saha_k
                * math.exp(
                    (nasa_g - _janaf_gibbs(temperature, ion_row))
                    / (R_J_MOL_K * temperature)
                ),
                rel=0.001,
            )
            continue
        if element == "Bi":
            # NBS Table 22's Bi+ enthalpy and Pankratz Table 80's Bi(g)
            # function differ by about 0.45 kJ/mol from the NIST ionization
            # energy after the electron-reference correction. Preserve that
            # source discrepancy explicitly instead of widening the 2% gate.
            expected_ratio = {
                1500.0: 0.9775126419,
                2500.0: 0.9924499829,
            }[temperature]
            assert fitted_k / saha_k == pytest.approx(
                expected_ratio, rel=5e-5
            ), element
        elif element in {"Mo", "W"}:
            # Pin the source offsets after the NIST low-level populations are
            # included; values are filled from the JANAF ion rows below.
            expected = {
                "Mo": {1500.0: 0.9479675449, 2500.0: 0.9683176023,
                       3000.0: 0.9727944998},
                "W": {1500.0: 0.9368243558, 2500.0: 1.8811176727,
                      3000.0: 2.1512200673},
            }[element][temperature]
            assert fitted_k / saha_k == pytest.approx(expected, rel=5e-5)
        elif element in {"Ba", "Sr", "Zr"}:
            if element == "Zr":
                # The nearly temperature-independent ~20 kJ/mol deficit is
                # an JANAF ionization-anchor offset, not missing partition
                # entropy. Preserve the measured ratio rather than weakening
                # the 2% spectroscopy gate for Ba and Sr.
                expected = {1500.0: 0.1872233437, 2500.0: 0.3803023410,
                            3000.0: 0.4543017938}[temperature]
                assert fitted_k / saha_k == pytest.approx(expected, rel=5e-5)
            else:
                assert fitted_k / saha_k == pytest.approx(1.0, rel=0.02)
        elif element in {"Ni", "Co"}:
            # Preserve the measured JANAF-to-NIST ionization comparison. Ni's
            # 11.759 kJ/mol enthalpy excess and the inferred 51.8 J/(mol K)
            # entropy contribution both matter to its large source ratio.
            # These are source offsets, not fit residuals or adjustments.
            expected = {
                "Ni": {1500.0: 165.8964143, 2500.0: 259.9036526},
                "Co": {1500.0: 1.139494036, 2500.0: 1.293825767},
            }[element][temperature]
            assert fitted_k / saha_k == pytest.approx(expected, rel=5e-5)
        else:
            assert fitted_k / saha_k == pytest.approx(1.0, rel=0.02), element


_ATTACHMENT_LEVELS = {
    # EAs are from NIST Chemistry WebBook SRD 69 records; Li, K, and O use
    # Andersen et al., JPCRD 28 (1999), Table 3. Neutral levels through
    # 15000 cm^-1 are from NIST ASD (https://physics.nist.gov/asd); the linked
    # Handbook level tables are the cited NIST level outputs. Bound anion
    # levels and splittings are from Andersen et al., Tables 3-4,
    # https://srd.nist.gov/jpcrdreprint/1.556047.pdf, with V- fine structure
    # updated from Fu et al., JCP 145, 164307 (2016). Energies are relative
    # to each species' ground level; tuples are (2J+1, cm^-1).
    "Al": (0.433816, ((2, 0.0), (4, 112.061)), ((1, 0.0), (3, 22.7), (5, 68.4), (5, 2611.0))),  # WebBook C7429905; https://physics.nist.gov/PhysRefData/Handbook/Tables/aluminumtable5.htm
    "B": (0.279743, ((2, 0.0), (4, 15.287)), ((1, 0.0), (3, 3.23), (5, 8.41))),  # WebBook C7440428; https://physics.nist.gov/PhysRefData/Handbook/Tables/borontable5.htm
    "Cr": (0.67583, ((7, 0.0), (5, 7593.16), (1, 7750.78), (3, 7810.82), (5, 7927.47), (7, 8095.21), (9, 8307.57)), ((6, 0.0),)),  # WebBook C7440473; https://physics.nist.gov/PhysRefData/Handbook/Tables/chromiumtable5.htm
    "Cs": (0.471630, ((2, 0.0), (2, 11178.2686), (4, 11732.3079), (4, 14499.2584), (6, 14596.8423)), ((1, 0.0),)),  # WebBook C7440462; https://physics.nist.gov/PhysRefData/Handbook/Tables/cesiumtable5.htm
    "Cu": (1.23578, ((2, 0.0), (6, 11202.565), (4, 13245.423)), ((1, 0.0),)),  # WebBook C7440508; https://physics.nist.gov/PhysRefData/Handbook/Tables/coppertable5.htm
    "Fe": (
        0.1510,
        ((9, 0.0), (7, 415.933), (5, 704.007), (3, 888.132), (1, 978.074), (11, 6928.268), (9, 7376.764), (7, 7728.059), (5, 7985.784), (3, 8154.713), (9, 11976.238), (7, 12560.933), (5, 12968.553)),
        ((10, 0.0), (8, 540.0), (6, 930.0), (4, 1200.0)),
    ),  # WebBook C7439896; https://physics.nist.gov/PhysRefData/Handbook/Tables/irontable5.htm
    "Ga": (0.430, ((2, 0.0), (4, 826.19)), ((1, 0.0), (3, 220.0), (5, 580.0))),  # WebBook C7440553; https://physics.nist.gov/PhysRefData/Handbook/Tables/galliumtable5.htm
    "K": (0.501459, ((2, 0.0), (2, 12985.186), (4, 13042.896)), ((1, 0.0),)),  # NIST JPCRD Table 3; https://physics.nist.gov/PhysRefData/Handbook/Tables/potassiumtable5.htm
    "Li": (0.618049, ((2, 0.0), (2, 14903.622), (4, 14903.957)), ((1, 0.0),)),  # NIST JPCRD Table 3; https://physics.nist.gov/PhysRefData/Handbook/Tables/lithiumtable5.htm
    "Na": (0.547926, ((2, 0.0),), ((1, 0.0),)),  # WebBook C7440235; https://physics.nist.gov/PhysRefData/Handbook/Tables/sodiumtable5.htm
    "Nb": (
        0.894,
        ((2, 0.0), (4, 154.19), (6, 391.99), (8, 695.25), (10, 1050.26), (4, 1142.79), (6, 1586.90), (8, 2154.11), (10, 2805.36), (2, 4998.17), (4, 5297.92), (6, 5965.45), (2, 8410.90), (4, 8705.32), (6, 9043.14), (8, 9497.52), (8, 8827.00), (10, 9328.88), (4, 9439.08), (6, 10237.51), (2, 10126.06), (4, 11318.09), (8, 10922.74), (10, 11044.08), (12, 11247.88), (14, 11524.65), (6, 11344.70), (6, 12018.25), (8, 12136.86), (10, 12357.70), (12, 13012.20), (10, 12102.12), (12, 12502.97), (4, 12288.25), (6, 12692.12), (8, 12982.38), (10, 13145.71), (6, 13404.77), (8, 13515.20), (2, 13629.15), (4, 14211.30), (6, 14899.26)),
        ((1, 0.0), (3, 110.0), (5, 310.0), (7, 560.0), (9, 870.0)),
    ),  # WebBook C7440031; anion levels from NIST JPCRD Table 4; https://physics.nist.gov/PhysRefData/Handbook/Tables/niobiumtable5.htm
    "O": (1.4611120, ((5, 0.0), (3, 158.265), (1, 226.977)), ((4, 0.0), (2, 177.10))),  # NIST JPCRD Tables 3-4
    "Pb": (0.3650, ((1, 0.0), (3, 7819.263), (5, 10650.327)), ((4, 0.0),)),  # WebBook C7439921; https://physics.nist.gov/PhysRefData/Handbook/Tables/leadtable5.htm
    "Rb": (0.485940, ((2, 0.0), (2, 12578.950), (4, 12816.545)), ((1, 0.0),)),  # WebBook C7440177; https://physics.nist.gov/PhysRefData/Handbook/Tables/rubidiumtable5.htm
    "Si": (
        1.389517,
        ((1, 0.0), (3, 77.112), (5, 223.157), (5, 6298.847)),
        ((4, 0.0), (4, 6954.78), (6, 6968.86), (2, 10973.31)),
    ),  # WebBook C7440213; bound states from NIST JPCRD Table 3; https://physics.nist.gov/PhysRefData/Handbook/Tables/silicontable5.htm
    "Ti": (0.0870, ((5, 0.0), (7, 170.134), (9, 386.875), (3, 6556.833), (5, 6598.764), (7, 6661.004), (9, 6742.755), (11, 6842.965), (5, 7255.354), (1, 8436.617), (3, 8492.421), (5, 8602.342), (5, 11531.759), (7, 11639.810), (9, 11776.811), (9, 12118.394)), ((4, 0.0), (6, 72.0), (8, 171.0), (10, 295.0))),  # WebBook C7440326; https://physics.nist.gov/PhysRefData/Handbook/Tables/titaniumtable5.htm
    "V": (
        0.526,
        ((4, 0.0), (6, 137.383), (8, 323.432), (10, 552.955), (2, 2112.282), (4, 2153.221), (6, 2220.156), (8, 2311.369), (10, 2424.809), (2, 8413.009), (4, 8476.234), (6, 8578.542), (8, 8715.747), (2, 9544.635), (4, 9637.039), (6, 9824.626), (8, 10892.520), (10, 11100.596), (4, 13801.551), (2, 13810.910), (4, 14514.756), (6, 14548.816), (8, 14909.958), (10, 14949.359)),
        ((1, 0.0), (3, 35.9), (5, 103.8), (7, 204.17), (9, 330.58)),
    ),  # WebBook C7440622; https://physics.nist.gov/PhysRefData/Handbook/Tables/vanadiumtable5.htm; V- FS from Fu et al. 2016
    "Mo": (0.74942, ((7, 0.0),), ((6, 0.0),)),  # https://webbook.nist.gov/cgi/cbook.cgi?ID=C7439987&Mask=267; Bilodeau et al. 1998
    "W": (0.816265, ((1, 0.0),), ((2, 0.0),)),  # https://webbook.nist.gov/cgi/cbook.cgi?ID=C7440337&Mask=267; Lindahl et al. 2010
    # Zr- 4F fine structure from Fu et al., J. Chem. Phys. 147, 064306 (2017), Table I.
    "Zr": (0.427, _SAHA_LEVELS["Zr"][1],
           ((4, 0.0), (6, 251.0), (8, 579.6), (10, 971.7))),
    # Ni- and Co- fine structure from Leopold and Lineberger, J. Chem. Phys.
    # 71 (1979), 4290-4297; neutral levels reuse the NIST Handbook arrays
    # audited above. Electron affinities use the NIST WebBook evaluations.
    "Ni": (1.15717, _SAHA_LEVELS["Ni"][1], ((6, 0.0), (4, 1470.0))),
    "Co": (0.66330, _SAHA_LEVELS["Co"][1],
           ((9, 0.0), (7, 920.9), (5, 1550.3))),
}


_ATOMIC_ANION_ELEMENTS = (
    "Al", "B", "Cr", "Cs", "Cu", "Fe", "Ga", "K", "Li", "Na", "Nb",
    "O", "Pb", "Rb", "Si", "Ti", "V", "Zn", "Ag", "Mo", "W", "Zr",
    "Ni", "Co",
)
_UNCHECKED_MOLECULAR_ANIONS = ("AlO", "AlO2", "BO", "BO2", "KO", "LiO", "NaO", "O2")


def test_gas_table_atomic_anion_inventory() -> None:
    """Keep every checked atomic row and unchecked molecular row enumerated."""
    packaged = {
        name
        for name in load_gas_datapack().gas_df.index.unique()
        if name.endswith("-(g)") and name != "e-(g)"
    }
    atomic = {f"{element}-(g)" for element in _ATOMIC_ANION_ELEMENTS}
    molecular = {f"{species}-(g)" for species in _UNCHECKED_MOLECULAR_ANIONS}
    assert packaged == atomic | molecular
    # JANAF supplies Zn-, but the neutral zinc atom has no bound anion term
    # listed for a ground-state attachment comparison. Ag- comes from NASA CEA,
    # whose card supplies no JANAF attachment-level entry.
    assert {f"{element}-(g)" for element in _ATTACHMENT_LEVELS} == atomic - {
        "Zn-(g)", "Ag-(g)"
    }


@pytest.mark.parametrize(
    "element",
    ("Al", "B", "Co", "Cr", "Cs", "Cu", "Fe", "Ga", "K", "Li", "Mo", "Na", "Nb", "Ni", "O", "Pb", "Rb", "Si", "Ti", "V", "W", "Zr"),
)
def test_atomic_anion_fits_match_electron_attachment(element: str) -> None:
    """Compare fitted attachment Kp with the NIST ground-state affinity.

    Al, B, Cr, Cs, Cu, Fe, K, Li, Na, Nb, O, Pb, Rb, Si, Ti, and V
    pass at 10%. Ga is pinned as a source-vintage known limit: its ratios
    are 0.369 at 1500 K and 0.568 at 2500 K, consistent with the JANAF
    source's implied EA near 0.30 eV rather than the current NIST 0.430 eV.

    For M + e- = M-,
    Kp=(p°/kBT)*(Z-/2Z0)*(h²/(2*pi*me*kBT))^(3/2)*exp(EA/kBT).
    Premise: attachment is the reverse of ionization of M-. Algebra therefore
    inverts the Saha translational factor and changes exp(-IE/kBT) to
    exp(+EA/kBT), retaining the electron spin factor 2 in the denominator.
    The units are m^-3 times m^3, so Kp is dimensionless at p°=1 bar.
    Energies in cm^-1 convert with hc=1.986445857e-23 J cm. Sanity: for
    one-level partitions this is exactly the inverse cation Saha expression;
    adding the NIST low-lying levels changes only the population factors.
    """
    affinity_eV, neutral_levels, anion_levels = _ATTACHMENT_LEVELS[element]
    pack = load_gas_datapack()
    attachment = f"{element}-(g)"
    k_b = 1.380649e-23
    h = 6.62607015e-34
    m_e = 9.1093837139e-31
    e_v = 1.602176634e-19
    h_c = 1.986445857e-23

    def partition(levels: tuple[tuple[int, float], ...], temperature: float) -> float:
        return sum(
            weight * math.exp(-h_c * energy / (k_b * temperature))
            for weight, energy in levels
        )

    ratios = []
    temperatures = (1500.0, 2500.0, 3000.0) if element == "Zr" else (
        1500.0, 2500.0
    )
    for temperature in temperatures:
        anion = _nearest_interval_row(
            pack.gas_df, attachment, temperature, allow_extrapolation=False
        )
        neutral = _nearest_interval_row(
            pack.gas_df, f"{element}(g)", temperature, allow_extrapolation=False
        )
        delta_g = (
            _janaf_gibbs(temperature, anion)
            - _janaf_gibbs(temperature, neutral)
            - _janaf_gibbs(
                temperature,
                _nearest_interval_row(
                    pack.gas_df, "e-(g)", temperature, allow_extrapolation=False
                ),
            )
        )
        fitted_k = math.exp(-delta_g / (R_J_MOL_K * temperature))
        saha_k = (
            1.0e5 / (k_b * temperature)
            * partition(anion_levels, temperature)
            / (2.0 * partition(neutral_levels, temperature))
            * (h**2 / (2.0 * math.pi * m_e * k_b * temperature)) ** 1.5
            * math.exp(affinity_eV * e_v / (k_b * temperature))
        )
        ratios.append(fitted_k / saha_k)
    if element == "Ga":
        # JANAF implies EA 0.301/0.308 eV; current NIST EA is 0.430 eV.
        assert ratios == pytest.approx((0.369, 0.568), rel=0.02), element
    elif element in {"Mo", "W"}:
        expected = {
            "Mo": (0.9737712359, 0.9782649482),
            "W": (1.7760941297, 1.1437305022),
        }[element]
        assert ratios == pytest.approx(expected, rel=5e-5), element
    elif element == "Zr":
        # The 0.427 eV NIST WebBook affinity and measured Zr- fine structure
        # are used. JANAF's fitted ratio changes with T, showing a partition-
        # entropy mismatch rather than a constant EA/formation-anchor offset.
        assert ratios == pytest.approx(
            (0.8971777200, 0.9632959954, 0.9720072362), rel=5e-5
        ), element
    elif element in {"Ni", "Co"}:
        # NIST WebBook affinities plus measured anion fine structure give
        # close attachment closure; pin their remaining source offsets.
        expected = {
            "Ni": (0.9909216236, 0.9945586334),
            "Co": (0.9838592042, 0.9902409117),
        }[element]
        assert ratios == pytest.approx(expected, rel=5e-5), element
    else:
        assert ratios == pytest.approx((1.0, 1.0), rel=0.10), element
    if element == "Cu":
        # Regression pin for the restored Cu- row, beside the shared physics
        # bound: JANAF Cu-007 implies EA 1.228 eV against NIST 1.23578 eV.
        assert ratios == pytest.approx(
            (0.9414909282, 0.9644856595), rel=5e-5
        ), element


def test_cr_channels_against_janaf_cells() -> None:
    """Cr channels against an independent 2200 K calculation from JANAF cells.

    The reference uses each table's own dfH(298), H-H(298) and S columns, the
    same thermodynamic route the fitted rows encode.  It deliberately does NOT
    use the printed dfG column: in JANAF 4th ed. the Cr-O tables (CrO(g),
    Cr2O3(l)) print dfG values that differ from their own dfH/H/S columns plus
    the Cr(ref) table Cr-001 by about -0.28 J/(mol K) x T per Cr atom (-1.12
    kJ/mol for Cr2O3(l) at 2000 K), while Cr(g) Cr-005 agrees with Cr-001 to
    within 1.5 J/mol.  Mixing the two conventions would put a spurious
    -0.015 dex offset on Cr(g) alone; the dH/H/S route is internally consistent.
    """
    T = 2200.0
    parent_activity = 1.0e-3
    fO2 = 1.0e-10
    gas_species = ("Cr", "CrO", "CrO2", "CrO3")
    pack = load_gas_datapack()
    pressures = evaluate_gas(
        {"Cr2O3": parent_activity},
        T,
        fO2,
        pack,
        gas_species=gas_species,
        allow_extrapolation=False,
    )
    g_o2 = _janaf_apparent_gibbs("O-029", T)
    g_parent = _janaf_apparent_gibbs("Cr-015", T)
    n_o2 = {"Cr": 1.5, "CrO": 0.5, "CrO2": -0.5, "CrO3": -1.5}

    for species in gas_species:
        # Cr2O3(l) = 2 X(g) + n_O2 O2(g); K = p_X^2 p_O2^n_O2 / a(Cr2O3).
        dG = (
            2.0 * _janaf_apparent_gibbs(GAS_TABLE_IDS[species], T)
            + n_o2[species] * g_o2
            - g_parent
        )
        expected = (
            math.exp(-dG / (R_J_MOL_K * T))
            * parent_activity
            / fO2**n_o2[species]
        ) ** 0.5
        assert math.isfinite(pressures[species]) and pressures[species] > 0.0
        # Gate: the Cr2O3(l) liquid-only fit residual is < 10 J/mol, shared
        # over two gas molecules; retain the independent 0.002 dex allowance.
        assert math.log10(pressures[species]) == pytest.approx(
            math.log10(expected), abs=0.002
        )


def test_generator_reproduces_packaged_tables_with_fit_tolerance(tmp_path: Path) -> None:
    generated = tmp_path / "gas-shomate.csv"
    condensate = tmp_path / "condensate.csv"
    packaged_condensate = GAS_DATA / "condensate.csv"
    packaged_gas = GAS_DATA / "gas-shomate.csv"

    def assert_table_matches(
        generated_path: Path,
        packaged_path: Path,
        fitted_columns: dict[str, tuple[str, ...]],
        evaluator=_shomate_g,
        ignore_row_order: bool = False,
    ) -> None:
        generated_table = pd.read_csv(generated_path, dtype=str, keep_default_na=False)
        packaged_table = pd.read_csv(packaged_path, dtype=str, keep_default_na=False)
        assert generated_table.columns.tolist() == packaged_table.columns.tolist()
        if ignore_row_order:
            sort_columns = ["species_name", "T_min", "T_max"]
            generated_table = generated_table.sort_values(sort_columns).reset_index(drop=True)
            packaged_table = packaged_table.sort_values(sort_columns).reset_index(drop=True)
        generated_species = generated_table["species_name"].tolist()
        packaged_species = packaged_table["species_name"].tolist()
        assert generated_species == packaged_species

        for row_index, species in enumerate(packaged_species):
            fit_columns = fitted_columns.get(species, ())
            assert set(fit_columns) <= set(packaged_table.columns)
            for column in packaged_table.columns:
                generated_value = generated_table.iloc[row_index][column]
                packaged_value = packaged_table.iloc[row_index][column]
                if column in fit_columns:
                    continue
                assert generated_value == packaged_value

            if fit_columns:
                generated_row = generated_table.iloc[row_index].to_dict()
                packaged_row = packaged_table.iloc[row_index].to_dict()
                _assert_fit_gibbs_matches(evaluator, packaged_row, generated_row)

    # Start from the packaged table with fitted rows removed, so the generator
    # must recreate them while passing every transcribed row through exactly.
    packaged_condensate_bytes = packaged_condensate.read_bytes()
    # Every condensate species the generator emits is fitted, including the
    # labelled constant-Cp continuation rows, so all of them are removed here
    # and compared below by evaluated G(T) rather than by coefficient text.
    fitted_condensate_species = frozenset(
        row["species_name"]
        for row in build_gas_tables.build_condensate_rows(JANAF_DATA)
    )
    fitted_prefixes = tuple(
        f"{species},".encode() for species in sorted(fitted_condensate_species)
    )
    stripped_condensate_bytes = b"".join(
        line
        for line in packaged_condensate_bytes.splitlines(keepends=True)
        if not line.startswith(fitted_prefixes)
    )
    condensate.write_bytes(stripped_condensate_bytes)
    command = [
        sys.executable,
        str(ROOT / "tools" / "build_gas_tables.py"),
        "--output",
        str(generated),
        "--condensate-output",
        str(condensate),
    ]
    first = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    first_gas_bytes = generated.read_bytes()
    first_condensate_bytes = condensate.read_bytes()
    # Rerun from the same stripped input: determinism means identical bytes
    # from identical inputs in this environment.
    condensate.write_bytes(stripped_condensate_bytes)
    second = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    assert first.stdout == second.stdout
    assert generated.read_bytes() == first_gas_bytes
    assert condensate.read_bytes() == first_condensate_bytes

    gas_table = pd.read_csv(packaged_gas, dtype=str, keep_default_na=False)
    gas_fit_columns = {
        species: tuple("ABCDEFG") for species in gas_table["species_name"]
    }
    condensate_fit_columns = {
        species: tuple(f"dG_{coefficient}" for coefficient in "ABCDE")
        for species in fitted_condensate_species
    }
    assert_table_matches(
        generated, packaged_gas, gas_fit_columns, ignore_row_order=True
    )
    assert_table_matches(
        condensate,
        packaged_condensate,
        condensate_fit_columns,
        evaluator=_condensate_gibbs,
        ignore_row_order=True,
    )

def test_default_major_condensate_rows_match_generator(tmp_path: Path) -> None:
    major_species = {"SiO2(l)", "Al2O3(l)", "MgO(l)", "CaO(l)"}
    generated = [
        row
        for row in build_gas_tables.build_condensate_rows(JANAF_DATA)
        if row["species_name"] in major_species
    ]
    default_path = GAS_DATA / "condensate.csv"
    with default_path.open(encoding="utf-8", newline="") as handle:
        packaged = list(csv.DictReader(handle))
    expected_rows = [row for row in packaged if row["species_name"] in major_species]
    assert len(expected_rows) == len(generated) == 15
    fit_coefficients = {f"dG_{coefficient}" for coefficient in "ABCDE"}
    for species in major_species:
        expected = sorted(
            (row for row in expected_rows if row["species_name"] == species),
            key=lambda row: float(row["T_min"]),
        )
        fitted = sorted(
            (row for row in generated if row["species_name"] == species),
            key=lambda row: float(row["T_min"]),
        )
        assert len(expected) == len(fitted)
        for packaged_row, generated_row in zip(expected, fitted):
            for column in build_gas_tables.CONDENSATE_COLUMNS:
                if column not in fit_coefficients:
                    assert packaged_row[column] == generated_row[column]
            _assert_fit_gibbs_matches(
                _condensate_gibbs, packaged_row, generated_row
            )
    research_payload = subprocess.run(
        [
            "git",
            "show",
            "5380e46:src/openimcc/data/packs/gas-janaf-parent-liquids-research/condensate.csv",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    research_rows = [
        row
        for row in csv.DictReader(io.StringIO(research_payload))
        if row["species_name"] in major_species
    ]
    continuation_suffix = build_gas_tables.SUPERCOOLED_LIQUID_REF_SUFFIX
    expected_unchanged_rows = [
        row for row in expected_rows
        if not row["Ref"].endswith(continuation_suffix)
    ]
    research_unchanged_rows = [
        row for row in research_rows
        if not row["Ref"].endswith(continuation_suffix)
    ]
    assert sorted(
        tuple(row[column] for column in build_gas_tables.CONDENSATE_COLUMNS)
        for row in expected_unchanged_rows
    ) == sorted(
        tuple(row[column] for column in build_gas_tables.CONDENSATE_COLUMNS)
        for row in research_unchanged_rows
    )
    crystal_rows = [row for row in packaged if row["species_name"] == "SiO2(cr)"]
    assert len(crystal_rows) == 1
    assert crystal_rows[0]["Ref"] == "LAM1987"

    regenerated_path = tmp_path / "condensate.csv"
    regenerated_path.write_bytes(default_path.read_bytes())
    build_gas_tables.merge_condensate_csv(
        build_gas_tables.build_condensate_rows(JANAF_DATA), regenerated_path
    )
    regenerated = pd.read_csv(regenerated_path, dtype=str, keep_default_na=False)
    packaged_table = pd.read_csv(default_path, dtype=str, keep_default_na=False)
    assert regenerated.columns.tolist() == packaged_table.columns.tolist()
    assert len(regenerated) == len(packaged_table)
    row_key = ["species_name", "T_min", "T_max"]
    regenerated = regenerated.set_index(row_key).sort_index()
    packaged_table = packaged_table.set_index(row_key).sort_index()
    assert regenerated.index.equals(packaged_table.index)
    fit_columns = {f"dG_{coefficient}" for coefficient in "ABCDE"}
    for key in packaged_table.index:
        regenerated_row = regenerated.loc[key].to_dict()
        packaged_row = packaged_table.loc[key].to_dict()
        for row in (regenerated_row, packaged_row):
            row["T_min"], row["T_max"] = key[1:]
        for column in packaged_table.columns:
            if column not in fit_columns:
                assert regenerated_row[column] == packaged_row[column]
        _assert_fit_gibbs_matches(
            _condensate_gibbs, packaged_row, regenerated_row
        )


def _shomate_g(row: dict[str, str], T: float) -> float:
    A, B, C, D, E, F, G, H = (float(row[key]) for key in "ABCDEFGH")
    t = T / 1000.0
    enthalpy = A * t + B * t**2 / 2 + C * t**3 / 3 + D * t**4 / 4 - E / t + F - H
    entropy = A * math.log(t) + B * t + C * t**2 / 2 + D * t**3 / 3 - E / (2 * t**2) + G
    return enthalpy * 1000.0 - T * entropy


def _assert_fit_gibbs_matches(evaluator, packaged: dict, generated: dict) -> None:
    t_min = float(packaged["T_min"])
    t_max = float(packaged["T_max"])
    temperatures = [t_min, *np.arange(t_min + 10.0, t_max, 10.0), t_max]
    # 1e-3/(R*T*ln 10) is 4e-8 dex at 1200 K (2e-8 near 2500 K), well below
    # the 1–10 J/mol source-node gates while allowing NumPy/BLAS fit drift.
    for temperature in temperatures:
        assert abs(
            evaluator(packaged, temperature) - evaluator(generated, temperature)
        ) <= 1.0e-3


def _condensate_gibbs(row: dict, temperature: float) -> float:
    columns = ("dG_A", "dG_B", "dG_C", "dG_D", "dG_E", "dH298_R")
    return _lamor_gibbs(
        temperature, pd.Series({column: float(row[column]) for column in columns})
    )


_CONDENSATE_EXPECTED = {
    "MgO(l)": (3100, 3500, -72.34, -0.804, 5.1067, -0.3615, 0.0, 0.0, "LAM1987"),
    "CaO(l)": (2900, 3800, -76.384, 0.924, 5.922, -0.7316, 0.0436, 0.0, "LAM1987"),
    "Al2O3(l)": (
        2327,
        3000,
        -201.54,
        232.345,
        -336.622,
        193.672,
        -48.1032,
        4.4461,
        "LAM1987",
    ),
    "SiO2(l)": (1996, 3000, -109.53, 2.12, 7.6492, -1.2588, 0.0998, 0.0, "LAM1987"),
    "SiO2(cr)": (1000, 1996, -109.53, 1.937, 8.59, -1.935, 0.225, 0.0, "LAM1987"),
    "Na2O(l)": (
        1500,
        3000,
        -44.8427056720216,
        5.80795016652333,
        15.2327690696853,
        -4.31343306861092,
        0.775422945866502,
        -0.0603564181113987,
        "Na-013",
    ),
    "K2O(l)": (1190, 3000, -43.58, 0.8, 18.889, -4.532, 0.467, 0.0, "LAM1984"),
    "FeO(l)": (1000, 5000, -30.01, 6.72, 6.588, -1.248, 0.150, -0.007697, "JANAF"),
    # Fitted by tools/build_gas_tables.py from JANAF O-044, not transcribed.
    "TiO2(l)": (
        1500,
        3000,
        -107.530100389706,
        6.93000608750468,
        6.19040408780147,
        -0.261882901980972,
        -0.134854424412715,
        0.0206704756752248,
        "O-044",
    ),
    # Fitted by tools/build_gas_tables.py from JANAF Cr-015's liquid branch;
    # transition markers are excluded rather than assigned a phase branch.
    "Cr2O3(l)": (
        1900,
        3000,
        -122.483081203024,
        12.1759221002617,
        11.862649354979,
        -1.6559818331697,
        0.117461993463336,
        -0.0004825006213904,
        "Cr-015",
    ),
    "V2O3(l)": (
        1700,
        3000,
        -131.464178771295,
        12.6885774332175,
        15.8920275807616,
        -3.31219814257183,
        0.45557724894072,
        -0.027915035621767,
        "O-063",
    ),
    "NbO2(l)": (
        1500,
        3000,
        -85.4982495755085,
        7.184739151057,
        9.85735655010328,
        -2.11399649945409,
        0.297926897960544,
        -0.0186305252415757,
        "Nb-013",
    ),
}


def test_sf04_condensate_rows_match_the_current_published_coefficients(
    sf04_gas_pack,
) -> None:
    table = sf04_gas_pack.oxide_df
    original_rows = set(table.index) - {"Li2O(l)", "Rb2O(l)", "PbO(l)"}
    assert original_rows == set(_CONDENSATE_EXPECTED)
    for species, expected in _CONDENSATE_EXPECTED.items():
        actual = table.loc[species]
        if isinstance(actual, pd.DataFrame):
            actual = actual.loc[actual["T_min"].astype(int) == expected[0]].iloc[0]
        assert (int(actual["T_min"]), int(actual["T_max"])) == expected[:2]
        for column, value in zip(
            ("dH298_R", "dG_A", "dG_B", "dG_C", "dG_D", "dG_E"),
            expected[2:8],
        ):
            assert actual[column] == value
        assert actual["Ref"] == expected[8]

    na_rows = table.loc["Na2O(l)"]
    na_low = na_rows.loc[na_rows["T_min"].astype(int) == 1200].iloc[0]
    assert (int(na_low["T_min"]), int(na_low["T_max"])) == (1200, 1500)
    assert tuple(
        na_low[column]
        for column in ("dH298_R", "dG_A", "dG_B", "dG_C", "dG_D", "dG_E")
    ) == (
        -44.8427056720216,
        5.37629004861128,
        15.9145037641667,
        -4.57747615825875,
        0.726170803003458,
        -0.0269424627020659,
    )
    assert na_low["Ref"] == "Na-013"


def test_v2o3_refit_changes_only_its_outputs_and_uses_liquid_boundary() -> None:
    from openimcc.gas import _oxide_row_for_T, evaluate_gas

    pack = load_gas_datapack(
        gas_path=GAS_DATA / "gas-shomate.csv",
        oxide_path=GAS_DATA / "condensate.csv",
    )
    v_rows = pack.oxide_df.loc["V2O3(l)"]
    high = v_rows.loc[v_rows["T_min"].astype(int) == 1700].iloc[0].copy()
    continuation = v_rows.loc[v_rows["T_min"].astype(int) == 1200].iloc[0].copy()
    for column, value in zip(
        ("T_min", "T_max", "dH298_R", "dG_A", "dG_B", "dG_C", "dG_D", "dG_E"),
        (
            1500,
            3000,
            -131.464178771295,
            12.7984408682205,
            15.7012635413964,
            -3.18957608585823,
            0.420979101959005,
            -0.0242973638300596,
        ),
    ):
        high[column] = value
    for column, value in zip(
        ("dG_A", "dG_B", "dG_C", "dG_D", "dG_E"),
        (
            15.4402114184811,
            9.2801679295331,
            2.67493200070515,
            -1.96286143257697,
            0.339268872704154,
        ),
    ):
        continuation[column] = value
    continuation["T_max"] = 1500
    old_rows = pd.DataFrame(
        [high, continuation], index=["V2O3(l)", "V2O3(l)"]
    )
    old_pack = replace(
        pack,
        oxide_df=pd.concat(
            [pack.oxide_df.drop(index="V2O3(l)"), old_rows]
        ),
    )

    assert _oxide_row_for_T(pack.oxide_df, "V2O3(l)", 1699.9)["T_min"] == 1200
    assert _oxide_row_for_T(pack.oxide_df, "V2O3(l)", 1700.0)["T_min"] == 1700
    activities = {
        str(name).removesuffix("(l)"): 1.0e-3
        for name in pack.oxide_df.index.unique()
    }
    v_channels = {"V", "VO", "VO2"}
    changed_v_temperatures = (1500.0, 1600.0, 1699.0, 1700.0, 2000.0, 3000.0)
    for temperature in range(1500, 3001):
        old = evaluate_gas(activities, temperature, 1.0e-8, old_pack)
        new = evaluate_gas(activities, temperature, 1.0e-8, pack)
        assert set(old) == set(new)
        unchanged = set(old) - v_channels
        assert all(old[name] == new[name] for name in unchanged)
        assert {
            name: old.domain_flags[name] for name in unchanged
        } == {
            name: new.domain_flags[name] for name in unchanged
        }
        if temperature in changed_v_temperatures:
            assert any(old[name] != new[name] for name in v_channels)


def test_lamoreaux_al2o3_condensate_matches_its_source_cells(sf04_gas_pack) -> None:
    """Pin the remaining corrected LH87 liquid row to its source cells."""
    rows = sf04_gas_pack.oxide_df
    al = rows.loc["Al2O3(l)"]
    assert tuple(
        float(al[key])
        for key in ("dH298_R", "dG_A", "dG_B", "dG_C", "dG_D", "dG_E")
    ) == (
        -201.54,
        232.345,
        -336.622,
        193.672,
        -48.1032,
        4.4461,
    )
    assert (int(al["T_min"]), int(al["T_max"])) == (2327, 3000)


def test_corrected_al2o3_liquid_matches_janaf_al100(sf04_gas_pack) -> None:
    """Corrected LH87 liquid G_app agrees with independent JANAF Al-100."""
    from openimcc.gas import _lamor_gibbs

    row = sf04_gas_pack.oxide_df.loc["Al2O3(l)"]
    source_rows = [
        source
        for source in _complete_rows("Al-100")
        if 2600.0 <= source["temperature"] <= 3000.0
    ]
    assert [source["temperature"] for source in source_rows] == [
        2600.0,
        2700.0,
        2800.0,
        2900.0,
        3000.0,
    ]
    for source in source_rows:
        assert abs(
            _lamor_gibbs(source["temperature"], row)
            - _source_g_app_from_row("Al-100", source)
        ) / 1000.0 < 0.5


def test_nb_low_condensate_interval_uses_liquid_rows_and_keeps_seam() -> None:
    from openimcc.gas import _lamor_gibbs, _oxide_row_for_T

    pack = load_gas_datapack()
    rows = pack.oxide_df.loc[["NbO2(l)"]]
    low = rows.loc[rows["T_min"].astype(float) == 1200.0].iloc[0]
    high = rows.loc[rows["T_min"].astype(float) == 1500.0].iloc[0]
    source_rows = [
        row
        for row in _complete_rows("Nb-013")
        if 1100.0 <= row["temperature"] <= 1500.0
        and row["temperature"] not in _ambiguous_temperatures("Nb-013")
    ]
    assert [row["temperature"] for row in source_rows] == [
        1100.0,
        1200.0,
        1300.0,
        1400.0,
        1500.0,
    ]
    assert all(row["heat_capacity"] == 94.14 for row in source_rows)
    residuals = [
        abs(_lamor_gibbs(row["temperature"], low) - _source_g_app_from_row("Nb-013", row))
        for row in source_rows
    ]
    provenance = yaml.safe_load(PROVENANCE_PATH.read_text(encoding="utf-8"))
    low_provenance = next(
        row
        for row in provenance["rows"]
        if row["table"] == "condensate"
        and row["species_name"] == "NbO2(l)"
        and row["T_range_K"] == [1200, 1500]
    )
    assert max(residuals) == pytest.approx(
        low_provenance["max_residual_J_per_mol"], abs=1.0e-6
    )
    assert max(residuals) < 1.0e-5
    for temperature in (1100.0, 1199.9):
        selected = _oxide_row_for_T(
            pack.oxide_df, "NbO2(l)", temperature, allow_extrapolation=True
        )
        assert int(selected["T_min"]) == 1200
        if temperature == 1100.0:
            source = next(row for row in source_rows if row["temperature"] == 1100.0)
            assert _lamor_gibbs(temperature, selected) == pytest.approx(
                _source_g_app_from_row("Nb-013", source), abs=1.0e-8
            )
    seam_log10 = (
        _lamor_gibbs(1500.0, low) - _lamor_gibbs(1500.0, high)
    ) / (R_J_MOL_K * 1500.0 * math.log(10.0))
    assert abs(seam_log10) < 0.001
    assert int(_oxide_row_for_T(pack.oxide_df, "NbO2(l)", 1499.9)["T_min"]) == 1200
    assert int(_oxide_row_for_T(pack.oxide_df, "NbO2(l)", 1500.0)["T_min"]) == 1500


def test_constant_cp_supercooled_rows_are_generated_and_provenanced(
    tmp_path: Path,
) -> None:
    default_generated = {
        (
            row["species_name"],
            int(float(row["T_min"])),
            int(float(row["T_max"])),
        ): row
        for row in build_gas_tables.build_condensate_rows(JANAF_DATA)
        if row["Ref"].endswith(build_gas_tables.SUPERCOOLED_LIQUID_REF_SUFFIX)
        and row["species_name"] != "SnO(l)"
    }
    datasets = (
        (
            default_generated,
            pd.read_csv(GAS_DATA / "condensate.csv"),
            yaml.safe_load(PROVENANCE_PATH.read_text(encoding="utf-8")),
                build_gas_tables.SUPERCOOLED_LIQUID_SOURCES,
            {
                "TiO2(l)", "Cr2O3(l)", "V2O3(l)", "SiO2(l)",
                "Al2O3(l)", "MgO(l)", "CaO(l)",
                "Ga2O3(l)", "GeO2(l)", "In2O3(l)", "Sc2O3(l)", "UO2(l)",
            },
        ),
    )
    default_copy = tmp_path / "default-condensate.csv"
    default_copy.write_bytes((GAS_DATA / "condensate.csv").read_bytes())
    build_gas_tables.merge_condensate_csv(
        build_gas_tables.build_condensate_rows(JANAF_DATA), default_copy
    )
    regenerated_table = pd.read_csv(default_copy, dtype=str, keep_default_na=False)
    packaged_table = pd.read_csv(
        GAS_DATA / "condensate.csv", dtype=str, keep_default_na=False
    )
    assert regenerated_table.columns.tolist() == packaged_table.columns.tolist()
    assert len(regenerated_table) == len(packaged_table)
    row_key = ["species_name", "T_min", "T_max"]
    regenerated_table = regenerated_table.set_index(row_key).sort_index()
    packaged_table = packaged_table.set_index(row_key).sort_index()
    assert regenerated_table.index.equals(packaged_table.index)
    fit_columns = {f"dG_{coefficient}" for coefficient in "ABCDE"}
    for key in packaged_table.index:
        regenerated_row = regenerated_table.loc[key].to_dict()
        packaged_row = packaged_table.loc[key].to_dict()
        for row in (regenerated_row, packaged_row):
            row["T_min"], row["T_max"] = key[1:]
        for column in packaged_table.columns:
            if column not in fit_columns:
                assert regenerated_row[column] == packaged_row[column]
        _assert_fit_gibbs_matches(
            _condensate_gibbs, packaged_row, regenerated_row
        )
    for generated, packaged, provenance, sources, expected_species in datasets:
        provenance_rows = {
            (row["species_name"], *row["T_range_K"]): row
            for row in provenance["rows"]
            if row["table"] == "condensate"
            and row.get("extrapolation") is True
            and row.get("authority") in {"janaf_fitted", "nasa_glenn_fitted"}
            and row["species_name"] not in {"SnO(l)", "MoO3(l)"}
        }
        assert {key[0] for key in generated} == expected_species
        assert set(provenance_rows) == set(generated)

    for generated, packaged, provenance, sources, _ in datasets:
        provenance_rows = {
            (row["species_name"], *row["T_range_K"]): row
            for row in provenance["rows"]
            if row["table"] == "condensate"
            and row.get("extrapolation") is True
            and row.get("authority") in {"janaf_fitted", "nasa_glenn_fitted"}
            and row["species_name"] not in {"SnO(l)", "MoO3(l)"}
        }
        for key, fitted in generated.items():
            species, T_min, T_max = key
            packaged_row = packaged.loc[
                (packaged["species_name"] == species)
                & (packaged["Ref"] == fitted["Ref"])
                & (packaged["T_min"].astype(float) == T_min)
                & (packaged["T_max"].astype(float) == T_max)
            ].iloc[0]
            fit_columns = {f"dG_{coefficient}" for coefficient in "ABCDE"}
            for column in build_gas_tables.CONDENSATE_COLUMNS:
                if column not in fit_columns:
                    assert str(packaged_row[column]) == fitted[column]
            _assert_fit_gibbs_matches(_condensate_gibbs, packaged_row, fitted)
            entry = provenance_rows[key]
            if entry["authority"] == "nasa_glenn_fitted":
                table_config = next(
                    item
                    for item in build_gas_tables.TRACE_NASA_CONDENSATE_SOURCES
                    if item[0] == species
                )
                source_t0 = table_config[6]
                source_path = NASA_DATA / f"{entry['table_id']}.json"
                source = _nasa_record(entry["table_id"])
                source_at_t0 = build_gas_tables._nasa7_properties(
                    source, source_t0
                )
                source_cp = R_J_MOL_K * source_at_t0["cp_R"]
                assert source["phase"] in {"liquid", "L"}
                assert entry["source_path"] == (
                    f"data-src/nasa-glenn/{entry['table_id']}.json"
                )
                assert entry["source_sha256"] == hashlib.sha256(
                    source_path.read_bytes()
                ).hexdigest()
                assert entry["method"] == (
                    "generated_constant_cp_extrapolation_fit"
                )
                assert entry["source_data"] is False
                assert entry["T_range_K"] == [T_min, T_max]
                assert entry["T0_K"] == source_t0
                assert entry["Cp_l_J_molK"] == pytest.approx(
                    source_cp, abs=1e-12
                )
                assert entry["max_residual_J_per_mol"] == pytest.approx(
                    float(fitted["_max_residual_J_per_mol"]), abs=1e-8
                )
                assert "NASA pure-liquid standard state at 1 atm" in entry["note"]
                continue
            source = _record(entry["table_id"])
            table_config = next(item for item in sources if item[0] == species)
            anchor = next(
                row
                for row in source["table"]["values"]
                if float(row["temperature"]["value"]) == table_config[-2]
            )
            anchor_row = next(
                row
                for row in _complete_rows(entry["table_id"])
                if row["temperature"] == table_config[-2]
            )
            liquid_nodes = [
                row
                for row in _complete_rows(entry["table_id"])
                if table_config[-2] <= row["temperature"] <= 3000.0
            ]
            expected_source_path = f"data-src/janaf/{entry['table_id']}.yaml"
            expected_source_sha256 = source["extraction"]["source_sha256"]
            cp = float(anchor["heat_capacity"]["value"])
            assert entry["source_path"] == expected_source_path
            assert entry["source_sha256"] == expected_source_sha256
            assert entry["method"] == "generated_constant_cp_extrapolation_fit"
            assert entry["authority"] == "janaf_fitted"
            assert entry["T_range_K"] == [T_min, T_max]
            assert entry["T0_K"] == table_config[-2]
            assert entry["Cp_l_J_molK"] == cp
            assert entry["max_residual_J_per_mol"] == pytest.approx(
                float(fitted["_max_residual_J_per_mol"]), abs=1e-8
            )
            if T_min == 1200:
                assert entry["uncertainty"]["delta_Cp_fraction"] == 0.1
                assert entry["uncertainty"]["scenario_only"] is True
                delta_cp = entry["uncertainty"]["delta_Cp_J_molK"]
                endpoints = (1200.0, float(table_config[-2]))
                delta_g = [
                    delta_cp
                    * (
                        (temperature - table_config[-2])
                        - temperature * math.log(temperature / table_config[-2])
                    )
                    for temperature in endpoints
                ]
                delta_log10_k = [
                    abs(value) / (R_J_MOL_K * temperature * math.log(10.0))
                    for value, temperature in zip(delta_g, endpoints)
                ]
                assert entry["uncertainty"]["max_abs_delta_G_J_per_mol"] == pytest.approx(
                    max(map(abs, delta_g)), abs=1e-9
                )
                assert entry["uncertainty"]["max_abs_delta_log10_K"] == pytest.approx(
                    max(delta_log10_k), abs=1e-12
                )

            for node in liquid_nodes:
                temperature = node["temperature"]
                assert node["heat_capacity"] == cp
                assert node["enthalpy_increment"] == pytest.approx(
                    anchor_row["enthalpy_increment"]
                    + cp * (temperature - table_config[-2]) / 1000.0,
                    abs=0.001,
                )
                assert node["entropy"] == pytest.approx(
                    anchor_row["entropy"]
                    + cp * math.log(temperature / table_config[-2]),
                    abs=0.001,
                )


def _assert_snol_constant_cp_continuation_uses_the_nasa_liquid_anchor() -> None:
    pack = load_gas_datapack()
    generated = next(
        row
        for row in build_gas_tables.build_condensate_rows(JANAF_DATA)
        if row["species_name"] == "SnO(l)"
        and row["Ref"].endswith(build_gas_tables.SUPERCOOLED_LIQUID_REF_SUFFIX)
    )
    packaged = pack.oxide_df.loc["SnO(l)"]
    packaged = packaged.loc[packaged["Ref"] == generated["Ref"]].iloc[0]
    record = _nasa_record("NG-1848")
    anchor_temperature = 1250.0
    properties = build_gas_tables._nasa7_properties(record, anchor_temperature)
    cp = R_J_MOL_K * properties["cp_R"]
    reference = float(record["delta_f_H_298_15"]["value"])
    anchor_h = (
        R_J_MOL_K * anchor_temperature * properties["h_rt"] - reference
    ) / 1000.0
    anchor_s = R_J_MOL_K * properties["s_R"]
    provenance = yaml.safe_load(PROVENANCE_PATH.read_text(encoding="utf-8"))
    row = next(
        item
        for item in provenance["rows"]
        if item["table"] == "condensate"
        and item["species_name"] == "SnO(l)"
        and item.get("extrapolation") is True
    )
    source_path = NASA_DATA / "NG-1848.json"
    assert row["table_id"] == "NG-1848"
    assert row["source_path"] == "data-src/nasa-glenn/NG-1848.json"
    assert row["source_sha256"] == hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert row["source_locator"] == "thermo.inp lines 14618-14622"
    assert row["authority"] == "nasa_glenn_fitted"
    assert row["method"] == "generated_constant_cp_extrapolation_fit"
    assert row["T0_K"] == anchor_temperature
    assert row["Cp_l_J_molK"] == pytest.approx(cp)
    assert row["T_range_K"] == [1200, 1250]
    residuals = []
    for temperature in (1200.0, 1250.0):
        enthalpy_increment = anchor_h + cp * (temperature - anchor_temperature) / 1000.0
        entropy = anchor_s + cp * math.log(temperature / anchor_temperature)
        source_g = reference + enthalpy_increment * 1000.0 - temperature * entropy
        residuals.append(abs(_lamor_gibbs(float(temperature), packaged) - source_g))
    assert max(residuals) < 10.0
    assert float(generated["_max_residual_J_per_mol"]) == pytest.approx(
        row["max_residual_J_per_mol"], abs=1e-6
    )


def test_every_constant_cp_continuation_preserves_cp_and_the_seam() -> None:
    """Continuation fits preserve their source Cp, H/S/G anchors, and seams.

    The Shomate Gibbs polynomial implies Cp/R = 2*B*tau + 6*C*tau^2 +
    12*D*tau^3 + 20*E*tau^4. Its entropy is R*(P + tau*P'). The fit fixes
    Gibbs and entropy at each generated interval's upper endpoint; comparison
    with the adjacent fit also bounds its published seam error.
    """
    from openimcc.gas import _lamor_gibbs

    pack = load_gas_datapack()
    generated_rows = [
        row
        for row in build_gas_tables.build_condensate_rows(JANAF_DATA)
        if row["Ref"].endswith(build_gas_tables.SUPERCOOLED_LIQUID_REF_SUFFIX)
    ]

    def entropy(row, temperature: float) -> float:
        tau = temperature / 1000.0
        coefficients = [float(row[f"dG_{key}"]) for key in "ABCDE"]
        polynomial = sum(value * tau**power for power, value in enumerate(coefficients))
        derivative = sum(
            power * coefficients[power] * tau ** (power - 1)
            for power in range(1, 5)
        )
        return R_J_MOL_K * (polynomial + tau * derivative)

    # The existing In2O3 row is the largest at 7.258 J/(mol K).
    maximum_cp_error_J_molK = 8.0

    for generated in generated_rows:
        species = generated["species_name"]
        rows = pack.oxide_df.loc[species]
        continuation = rows.loc[
            (rows["Ref"] == generated["Ref"])
            & (rows["T_min"].astype(float) == float(generated["T_min"]))
            & (rows["T_max"].astype(float) == float(generated["T_max"]))
        ].iloc[0]
        anchor_cp = float(generated["_Cp_l_J_molK"])
        temperatures = np.linspace(
            float(continuation["T_min"]), float(continuation["T_max"]), 1001
        )
        max_cp_error = 0.0
        for temperature in temperatures:
            tau = float(temperature) / 1000.0
            b, c, d, e = (
                float(continuation[f"dG_{key}"]) for key in "BCDE"
            )
            implied_cp = R_J_MOL_K * (
                2.0 * b * tau + 6.0 * c * tau**2
                + 12.0 * d * tau**3 + 20.0 * e * tau**4
            )
            max_cp_error = max(max_cp_error, abs(implied_cp - anchor_cp))
        assert max_cp_error <= maximum_cp_error_J_molK, (
            species,
            float(continuation["T_min"]),
            float(continuation["T_max"]),
            max_cp_error,
        )

        source_anchor_temperature = float(generated["_T0_K"])
        anchor_temperature = float(generated["T_max"])
        cp = float(generated["_Cp_l_J_molK"])
        anchor_g = (
            float(generated["_anchor_H_J_per_mol"])
            + cp * (anchor_temperature - source_anchor_temperature)
            - anchor_temperature
            * (
                float(generated["_anchor_S_J_molK"])
                + cp * math.log(anchor_temperature / source_anchor_temperature)
            )
        )
        anchor_s = float(generated["_anchor_S_J_molK"]) + cp * math.log(
            anchor_temperature / source_anchor_temperature
        )
        anchor_h = anchor_g + anchor_temperature * anchor_s
        fitted_g = _lamor_gibbs(anchor_temperature, continuation)
        fitted_s = entropy(continuation, anchor_temperature)
        fitted_h = fitted_g + anchor_temperature * fitted_s
        assert fitted_g == pytest.approx(anchor_g, abs=0.02), species
        # These continuation polynomials are accepted by the <10 J/mol
        # Gibbs-node gate; derivative entropy/enthalpy do not fit exactly.
        entropy_tolerance = (
            0.2 if species in {"Sc2O3(l)", "UO2(l)"} else 2e-5
        )
        assert fitted_s == pytest.approx(anchor_s, abs=entropy_tolerance), species
        h_tolerance = 0.03 + anchor_temperature * entropy_tolerance
        assert fitted_h == pytest.approx(anchor_h, abs=h_tolerance), species

        seam = float(continuation["T_max"])
        adjacent_rows = rows.loc[rows["T_min"].astype(float) == seam]
        if adjacent_rows.empty:
            assert seam == 3000.0
            continue
        adjacent = adjacent_rows.iloc[0]
        assert abs(_lamor_gibbs(seam, continuation) - _lamor_gibbs(seam, adjacent)) <= 30.0
        assert abs(entropy(continuation, seam) - entropy(adjacent, seam)) <= 1.3
    _assert_snol_constant_cp_continuation_uses_the_nasa_liquid_anchor()


def test_cu020_malformed_1600_row_is_excluded_from_the_fit() -> None:
    parsed = build_gas_tables._janaf_text_source_rows(JANAF_DATA, "Cu-020")
    assert 1600.0 not in parsed
    fit_nodes = [
        float(temperature)
        for temperature in np.arange(1500.0, 2000.1, 100.0)
    ]
    assert [temperature for temperature in fit_nodes if temperature in parsed] == [
        1500.0, 1700.0, 1800.0, 1900.0, 2000.0
    ]


def _assert_nasa_anchored_rows_match_direct_card_evaluation() -> None:
    """NASA-anchored rows match direct card G; all other NASA sources are listed."""
    from openimcc.gas import _lamor_gibbs

    pack = load_gas_datapack()
    provenance = yaml.safe_load(PROVENANCE_PATH.read_text(encoding="utf-8"))
    def row_key(record):
        return (
            record["table"], record["species_name"], tuple(record["T_range_K"])
        )

    # Independent formation/entropy anchors intentionally override the NASA
    # card's absolute G while retaining its thermal shape.
    NASA_ANCHORED_ROW_EXCLUSIONS = {
        ("gas", "Na2O(g)", (1500, 3000)): "LH84 provides the anchors; NASA supplies Cp(T).",
        ("gas", "Na2O(g)", (500, 1500)): "LH84 provides the anchors; NASA supplies Cp(T).",
        ("gas", "K2O(g)", (1500, 3000)): "LH84 provides the anchors; NASA supplies Cp(T).",
        ("gas", "K2O(g)", (500, 1500)): "LH84 provides the anchors; NASA supplies Cp(T).",
        ("condensate", "PbO(l)", (1500, 3000)): "JANAF anchors the NASA tail at 2500 K.",
        ("condensate", "Cu2O(l)", (2000, 2500)): "JANAF anchors the NASA tail at 2000 K.",
        ("condensate", "Cu2O(l)", (2500, 3000)): "JANAF anchors the NASA tail at 2000 K.",
        ("condensate", "Ga2O3(l)", (1200, 2080)): "Generated constant-Cp continuation below the NASA liquid card.",
        ("condensate", "GeO2(l)", (1200, 1388)): "Generated constant-Cp continuation below the NASA liquid card.",
        ("condensate", "In2O3(l)", (1200, 2186)): "Generated constant-Cp continuation below the NASA liquid card.",
        ("condensate", "SnO(l)", (1200, 1250)): "Generated constant-Cp continuation below the NASA liquid card.",
        ("condensate", "Sc2O3(l)", (1200, 1500)): "Generated constant-Cp continuation below the NASA liquid card.",
        ("condensate", "Sc2O3(l)", (1500, 2000)): "Generated constant-Cp continuation below the NASA liquid card.",
        ("condensate", "Sc2O3(l)", (2000, 2500)): "Generated constant-Cp continuation below the NASA liquid card.",
        ("condensate", "Sc2O3(l)", (2500, 2762)): "Generated constant-Cp continuation below the NASA liquid card.",
        ("condensate", "UO2(l)", (1200, 1500)): "Generated constant-Cp continuation below the NASA liquid card.",
        ("condensate", "UO2(l)", (1500, 2000)): "Generated constant-Cp continuation below the NASA liquid card.",
        ("condensate", "UO2(l)", (2000, 2500)): "Generated constant-Cp continuation below the NASA liquid card.",
        ("condensate", "UO2(l)", (2500, 3000)): "Generated constant-Cp continuation below the NASA liquid card.",
    }
    nasa_sourced = {
        row_key(record): record
        for record in provenance["rows"]
        if record.get("authority") == "nasa_glenn_fitted"
        or "tail_source_path" in record
        or str(record.get("source_path", "")).startswith("data-src/nasa-glenn/")
    }
    checked = set(nasa_sourced) - set(NASA_ANCHORED_ROW_EXCLUSIONS)
    assert set(nasa_sourced) == checked | set(NASA_ANCHORED_ROW_EXCLUSIONS)
    for key in checked:
        record = nasa_sourced[key]
        assert record.get("source_path", "").startswith("data-src/nasa-glenn/")
        assert record["authority"].startswith("nasa_glenn_fitted")
        assert record["method"] == "fitted"
        assert record["table_id"].startswith("NG-")

    for record in provenance["rows"]:
        if row_key(record) not in checked:
            continue
        card_path = NASA_DATA / f"{record['table_id']}.json"
        card = build_gas_tables._load_record(card_path)
        lower, upper = map(float, record["T_range_K"])
        temperatures = tuple(
            lower + fraction * (upper - lower) for fraction in (0.15, 0.5, 0.85)
        )
        if record["table"] == "gas":
            row = _gas_row_for_interval(
                pack, record["species_name"].removesuffix("(g)"),
                int(record["T_interval"]),
            )
            fitted_g = lambda T: _janaf_gibbs(T, row)
        else:
            candidates = pack.oxide_df.loc[record["species_name"]]
            row = candidates.loc[candidates["T_min"].astype(float) == lower].iloc[0]
            fitted_g = lambda T: _lamor_gibbs(T, row)

        allowed = float(record["max_residual_J_per_mol"]) + 10.0
        for temperature in temperatures:
            properties = build_gas_tables._nasa7_properties(card, temperature)
            if record["table_id"] in {
                "NG-0831", "NG-0832", "NG-0833", "NG-0834", "NG-1258"
            }:
                at_298 = build_gas_tables._nasa7_properties(card, 298.15)
                direct_g = (
                    float(card["delta_f_H_298_15"]["value"])
                    + R_J_MOL_K
                    * (temperature * properties["h_rt"] - 298.15 * at_298["h_rt"])
                    - temperature * R_J_MOL_K * properties["s_R"]
                )
            else:
                direct_g = R_J_MOL_K * temperature * (
                    properties["h_rt"] - properties["s_R"]
                )
            assert abs(fitted_g(temperature) - direct_g) <= allowed

    # The JANAF-anchored tails preserve NASA's H/S increments from their splice.
    for species, janaf_id, nasa_id, formula, splice in (
        ("PbO(l)", "O-007", "NG-1801", "PbO", 2500.0),
        ("Cu2O(l)", "Cu-020", "NG-1844", "Cu2O", 2000.0),
    ):
        source_rows, _reference = build_gas_tables._janaf_condensate_nasa_tail_rows(
            JANAF_DATA,
            NASA_DATA,
            janaf_id,
            nasa_id,
            nasa_formula=formula,
            anchor_temperature=splice,
        )
        card = _nasa_record(nasa_id)
        anchor = source_rows[splice]
        anchor_card = build_gas_tables._nasa7_properties(card, splice)
        for temperature, source in source_rows.items():
            if temperature < splice:
                continue
            card_values = build_gas_tables._nasa7_properties(card, temperature)
            expected_h_increment = R_J_MOL_K * (
                temperature * card_values["h_rt"]
                - splice * anchor_card["h_rt"]
            ) / 1000.0
            expected_s_increment = R_J_MOL_K * (
                card_values["s_R"] - anchor_card["s_R"]
            )
            assert source["enthalpy_increment"] - anchor["enthalpy_increment"] == pytest.approx(
                expected_h_increment, abs=1e-12
            )
            assert source["entropy"] - anchor["entropy"] == pytest.approx(
                expected_s_increment, abs=1e-12
            )


def _source_g_app_from_row(table_id: str, row: dict[str, float]) -> float:
    rows = _complete_rows(table_id)
    reference = next(
        row["formation_enthalpy"] for row in rows if row["temperature"] == 298.15
    )
    return (
        (reference + row["enthalpy_increment"]) * 1000.0
        - row["temperature"] * row["entropy"]
    )


def _reaction_source_series(
    species: str,
) -> tuple[int, float, list[float], list[float], list[float]]:
    parent, n_gas, n_o2 = _SF04_REACTIONS[species]
    species_id = GAS_TABLE_IDS[species]
    gas_rows = {row["temperature"]: row for row in _complete_rows(species_id)}
    oxygen_rows = {
        row["temperature"]: row for row in _complete_rows("O-029")
    }
    parent_id = PARENT_TABLE_IDS[parent] if parent else None
    parent_rows = (
        {row["temperature"]: row for row in _complete_rows(parent_id)}
        if parent_id
        else {}
    )
    common_temperatures = sorted(
        set(gas_rows)
        & set(oxygen_rows)
        & (set(parent_rows) if parent else set(gas_rows))
    )
    fit_t_min = max(
        build_gas_tables._FIT_T_MIN_BY_TABLE.get(
            species_id, build_gas_tables.FIT_T_MIN
        ),
        (
            build_gas_tables._FIT_T_MIN_BY_TABLE.get(
                parent_id, build_gas_tables.FIT_T_MIN
            )
            if parent_id
            else build_gas_tables.FIT_T_MIN
        ),
    )
    common_temperatures = [
        temperature
        for temperature in common_temperatures
        if fit_t_min
        <= temperature
        <= build_gas_tables._FIT_T_MAX_BY_TABLE.get(
            species_id, build_gas_tables.FIT_T_MAX
        )
        and temperature not in _ambiguous_temperatures(species_id)
        and temperature not in _ambiguous_temperatures("O-029")
        and (
            not parent_id
            or temperature not in _ambiguous_temperatures(parent_id)
        )
    ]
    independent_reactions = []
    source_parent_app = []
    for source_temperature in common_temperatures:
        gas_row = gas_rows[source_temperature]
        oxygen_row = oxygen_rows[source_temperature]
        parent_row = parent_rows[source_temperature] if parent_id else None
        if species.startswith("Cr"):
            # Cr's printed formation-Gibbs column carries a source-reference
            # offset from the apparent-Gibbs construction used by the fit.
            gas_g = _source_g_app_from_row(species_id, gas_row)
            oxygen_g = _source_g_app_from_row("O-029", oxygen_row)
            parent_g = (
                _source_g_app_from_row(parent_id, parent_row)
                if parent_row
                else 0.0
            )
            independent_reactions.append(n_gas * gas_g + n_o2 * oxygen_g - parent_g)
        else:
            independent_reactions.append(
                n_gas * gas_row["formation_gibbs_energy"] * 1000.0
                + n_o2 * oxygen_row["formation_gibbs_energy"] * 1000.0
                - (
                    parent_row["formation_gibbs_energy"] * 1000.0
                    if parent_row
                    else 0.0
                )
            )
        source_parent_app.append(
            _source_g_app_from_row(parent_id, parent_row) if parent_row else 0.0
        )
    return n_gas, n_o2, common_temperatures, independent_reactions, source_parent_app


def test_g2_reaction_convention_at_complete_janaf_nodes() -> None:
    """Check the gas convention only where every source table has a row.

    Premise: JANAF's formation-Gibbs column and the fitted apparent-Gibbs
    construction use the same elemental reference. Element-reference identity
    therefore cancels every balanced reaction's elemental baselines, leaving
    the fitted ``F - H°298`` offset, reference-state choice, and sign exposed.
    No interpolation is allowed here: both sides are evaluated at the same
    complete, non-ambiguous source node. Unit check: all values are J/mol.
    Sanity: a reference-state bias of 100 J/mol cannot hide behind a grid chord.
    """
    pack = load_gas_datapack()
    checked = 0
    maxima: dict[str, float] = {}
    for species in _SF04_REACTIONS:
        if species == "O2" or species in NASA_TABLE_IDS or species in {
            "Mn",
            "MnO",
            "Ni",
            "NiO",
            "Co",
            "CoO",
        } | build_gas_tables.TRACE_GAS_SPECIES:
            continue
        if (
            _SF04_REACTIONS[species][0]
            and _SF04_REACTIONS[species][0] not in PARENT_TABLE_IDS
        ):
            # No public evaluated P2O5(l) source is available; JANAF lists
            # P4O10(cr) only. S2(g) uses a gas-parent law covered by the
            # dedicated sulfur cell tests.
            continue
        n_gas, n_o2, temperatures, independent, source_parent_app = (
            _reaction_source_series(species)
        )
        row_max = 0.0
        for temperature, independent_reaction, independent_parent in zip(
            temperatures, independent, source_parent_app
        ):
            fitted_reaction = (
                n_gas
                * _janaf_gibbs(
                    temperature,
                    _nearest_interval_row(
                        pack.gas_df, f"{species}(g)", temperature
                    ),
                )
                + n_o2
                * _janaf_gibbs(
                    temperature,
                    _nearest_interval_row(pack.gas_df, "O2(g)", temperature),
                )
                - independent_parent
            )
            row_max = max(row_max, abs(fitted_reaction - independent_reaction))
            checked += 1
        maxima[species] = row_max

    # Liquid-only starts omit source rows before each selected branch; the
    # V2O3 start at 1700 K leaves 417 complete reaction nodes overall.
    assert checked == 417
    # The measured on-node maximum remains below 10 J/mol for every fitted gas
    # row; the separate condensate test records each parent-row fit residual.
    assert max(maxima.values()) <= 10.0


_G2_SOURCE_HOLES = (
    ("Na2O", "Na-013", 1500.0),
    ("FeO", "Fe-019", 1700.0),
    ("SiO2", "O-038", 1700.0),
    ("MgO", "Mg-009", 2100.001),
    ("CaO", "Ca-028", 2100.0),
    ("Al2O3", "Al-100", 2400.0),
    ("TiO2", "O-044", 2200.0),
    ("V2O3", "O-063", 1600.0),
    ("V2O3", "O-063", 2400.0),
    ("NbO2", "Nb-013", 2175.0),
    ("NbO2", "Nb-013", 2200.0),
)


def test_g2_reaction_convention_on_workbook_grid() -> None:
    """Use the workbook grid as coverage evidence, including interpolation.

    This is deliberately a looser ``< 320 J/mol`` gate, set from the measured
    306 J/mol maximum with a small margin: it includes the linear
    interpolation error across known source holes, not only convention error.
    The named holes are Na2O/Na-013 at 1500 K, FeO/Fe-019 at 1700 K,
    SiO2/O-038 at 1700 K, MgO/Mg-009 at 2100.001 K, CaO/Ca-028 at 2100 K, and
    Al2O3/Al-100 at 2400 K, TiO2/O-044 at 2200 K, V2O3/O-063 at 1600 and
    2400 K, and NbO2/Nb-013 at 2175 and 2200 K. JANAF liquid-branch starts
    also leave Al source comparisons below 2500 K, Si below 1800 K, Mg/Ca
    below 2200 K, and V below 1700 K. K2O's K-012 parent ends at 2000 K, so K,
    K2 and KO have no independent parent reference at 2125, 2250, 2375 or
    2500 K.
    """
    for _oxide, table_id, temperature in _G2_SOURCE_HOLES:
        assert temperature in _ambiguous_temperatures(table_id)

    pack = load_gas_datapack()
    checked = 0
    missing: set[tuple[str, float]] = set()
    maxima: dict[str, float] = {}
    for species in _SF04_REACTIONS:
        if species == "O2" or species in NASA_TABLE_IDS or species in {
            "Mn",
            "MnO",
            "Ni",
            "NiO",
            "Co",
            "CoO",
        } | build_gas_tables.TRACE_GAS_SPECIES:
            continue
        if (
            _SF04_REACTIONS[species][0]
            and _SF04_REACTIONS[species][0] not in PARENT_TABLE_IDS
        ):
            continue
        n_gas, n_o2, common_temperatures, independent_reactions, source_parent_app = (
            _reaction_source_series(species)
        )
        for temperature in IMCC_SF04_WORKBOOK_GRID_K:
            if (
                not common_temperatures
                or temperature < common_temperatures[0]
                or temperature > common_temperatures[-1]
            ):
                missing.add((species, temperature))
                continue
            # Premise: source reaction and parent apparent-G values are known
            # only at complete common nodes. Algebra: interpolate each series
            # to the workbook temperature. Unit check: the result remains
            # J/mol. Sanity: exact source nodes are returned unchanged.
            independent_reaction = float(
                np.interp(
                    temperature, common_temperatures, independent_reactions
                )
            )
            independent_parent = float(
                np.interp(temperature, common_temperatures, source_parent_app)
            )
            fitted_reaction = (
                n_gas
                * _janaf_gibbs(
                    temperature,
                    _nearest_interval_row(
                        pack.gas_df, f"{species}(g)", temperature
                    ),
                )
                + n_o2
                * _janaf_gibbs(
                    temperature,
                    _nearest_interval_row(pack.gas_df, "O2(g)", temperature),
                )
                - independent_parent
            )
            discrepancy = abs(fitted_reaction - independent_reaction)
            maxima[species] = max(maxima.get(species, 0.0), discrepancy)
            checked += 1

    assert missing == {
        (species, 1500.0)
        for species in ("Na", "Na2", "NaO")
    } | {
        (species, temperature)
        for species in ("V", "VO", "VO2")
        for temperature in (1500.0, 1625.0)
    } | {
        (species, temperature)
        for species in ("K", "K2", "KO")
        for temperature in (2125.0, 2250.0, 2375.0, 2500.0)
    } | {
        (species, temperature)
        for species in ("Cr", "CrO", "CrO2", "CrO3")
        for temperature in (1500.0, 1625.0, 1750.0, 1875.0)
    } | {
        (species, temperature)
        for species in ("SiO", "SiO2", "Si", "Si2", "Si3")
        for temperature in (1500.0, 1625.0, 1750.0)
    } | {
        (species, temperature)
        for species in ("Al", "AlO", "AlO2", "Al2", "Al2O", "Al2O2")
        for temperature in (
            1500.0,
            1625.0,
            1750.0,
            1875.0,
            1900.0,
            2000.0,
            2125.0,
            2250.0,
            2375.0,
        )
    } | {
        (species, temperature)
        for species in ("Mg", "MgO", "Ca", "CaO")
        for temperature in (
            1500.0,
            1625.0,
            1750.0,
            1875.0,
            1900.0,
            2000.0,
            2125.0,
        )
    }
    assert checked == 236
    existing = set(_SF04_REACTIONS) - {
        "O2",
        *NASA_TABLE_IDS,
        "V",
        "VO",
        "VO2",
        "Nb",
        "NbO",
        "NbO2",
        "Mn",
        "MnO",
        "Ni",
        "NiO",
        "Co",
        "CoO",
        "P",
        "P2",
        "P4",
        "PO",
        "PO2",
        "P4O6",
        "P4O10",
        "S",
        "S2",
        "S3",
        "S4",
        "S5",
        "S6",
        "S7",
        "S8",
        "SO",
        "SO2",
        "SO3",
        "SSO",
    } - build_gas_tables.TRACE_GAS_SPECIES
    assert max(maxima[species] for species in existing) < 300.0
    # The omitted O-063 and Nb-013 source nodes make the V/Nb interpolation
    # comparison slightly looser than the established channels.
    assert max(maxima[species] for species in set(maxima) - existing) < 320.0


# The legacy VapoRock tables are not shipped; comparisons against them run only
# when a checkout is selected with OPENIMCC_VAPOROCK_ROOT, and skip otherwise.
# tests/conftest.py moves that value to OPENIMCC_TEST_LEGACY_VAPOROCK_ROOT so it
# cannot redirect the packaged-table tests.
_LEGACY_ROOT_ENV = os.environ.get("OPENIMCC_TEST_LEGACY_VAPOROCK_ROOT")
VAPOROCK_ROOT = Path(_LEGACY_ROOT_ENV or "/nonexistent-vaporock")
VAPOROCK_GAS = VAPOROCK_ROOT / "src" / "vaporock" / "data" / "JANAF-vapor-data-full.csv"
VAPOROCK_OXIDE = VAPOROCK_ROOT / "data" / "condensate-thermo-data.csv"


@pytest.mark.skipif(
    not _LEGACY_ROOT_ENV,
    reason="legacy VapoRock checkout is not selected",
)
def test_legacy_k_row_differs_from_janaf_source_by_about_663_j_per_mol() -> None:
    legacy_root = VAPOROCK_ROOT
    legacy_gas = legacy_root / "src" / "vaporock" / "data" / "JANAF-vapor-data-full.csv"
    legacy_oxide = legacy_root / "data" / "condensate-thermo-data.csv"
    if not (legacy_gas.is_file() and legacy_oxide.is_file()):
        pytest.skip("selected VapoRock checkout is incomplete")

    legacy_pack = load_gas_datapack(gas_path=legacy_gas, oxide_path=legacy_oxide)
    source_row = next(
        row for row in _complete_rows("K-005") if row["temperature"] == 2000.0
    )
    janaf_source_g = _source_g_app_from_row("K-005", source_row)
    legacy_g = _janaf_gibbs(
        2000.0, _nearest_interval_row(legacy_pack.gas_df, "K(g)", 2000.0)
    )
    # The legacy row's +662.8 J/mol offset explains its 0.315609 log10(p_K)
    # and the obsolete 0.316 literal; the JANAF source-column reference rounds
    # to 0.333 and is the one used by the independent regression above.
    assert legacy_g - janaf_source_g == pytest.approx(662.8, abs=1.0)


@pytest.mark.skipif(
    not (VAPOROCK_GAS.is_file() and VAPOROCK_OXIDE.is_file()),
    reason="controller-supplied VapoRock checkout is unavailable for G3",
)
def test_g3_new_tables_are_finite_against_current_vaporock() -> None:
    new_pack = load_gas_datapack()
    old_pack = load_gas_datapack(gas_path=VAPOROCK_GAS, oxide_path=VAPOROCK_OXIDE)
    compositions = [
        {oxide: 1.0 for oxide in ("SiO2", "MgO", "FeO", "CaO", "Al2O3", "Na2O", "K2O")},
        {
            "SiO2": 0.71,
            "MgO": 0.01,
            "FeO": 0.02,
            "CaO": 0.10,
            "Al2O3": 0.08,
            "Na2O": 0.06,
            "K2O": 0.02,
        },
        {
            "SiO2": 0.45,
            "MgO": 0.15,
            "FeO": 0.08,
            "CaO": 0.12,
            "Al2O3": 0.12,
            "Na2O": 0.06,
            "K2O": 0.02,
        },
    ]
    # The legacy condensate table has no TiO2(l) row, so the Ti channels have
    # no legacy counterpart to compare against.
    legacy_species = tuple(
        species
        for species in IMCC_GAS_CHANNEL_SPECIES
        if _SF04_REACTIONS[species][0] != "TiO2"
        and species not in NASA_TABLE_IDS
    )
    assert "TiO2(l)" not in old_pack.oxide_df.index
    max_by_species = {species: 0.0 for species in legacy_species}
    for temperature in IMCC_SF04_WORKBOOK_GRID_K:
        for activities in compositions:
            for fugacity in (1.0, 1.0e-4, 1.0e-10):
                for species in legacy_species:
                    new = load_and_evaluate(
                        new_pack, activities, temperature, fugacity, species
                    )
                    old = load_and_evaluate(
                        old_pack, activities, temperature, fugacity, species
                    )
                    difference = math.log10(new / old)
                    assert math.isfinite(difference)
                    max_by_species[species] = max(
                        max_by_species[species], abs(difference)
                    )
    # CaO is the only >0.05 dex case: the old VapoRock CaO(g) fit starts at
    # 4500 K and this comparison intentionally evaluates its legacy extrapolation
    # down to 1500 K, while the new fit is sourced over the IMCC domain.
    assert {species for species, value in max_by_species.items() if value > 0.05} == {
        "CaO"
    }


@pytest.mark.skipif(
    not (VAPOROCK_GAS.is_file() and VAPOROCK_OXIDE.is_file()),
    reason="controller-supplied VapoRock checkout is unavailable",
)
def test_legacy_tables_default_call_returns_the_sf04_set_without_refusal() -> None:
    from openimcc.gas import ImccGasSpeciesNotFoundError, evaluate_gas

    old_pack = load_gas_datapack(gas_path=VAPOROCK_GAS, oxide_path=VAPOROCK_OXIDE)
    assert "TiO2(l)" not in old_pack.oxide_df.index
    sf04 = tuple(
        species
        for species in IMCC_GAS_CHANNEL_SPECIES
        if _SF04_REACTIONS[species][0] != "TiO2"
        and species not in NASA_TABLE_IDS
    )
    activities = {
        "SiO2": 0.45,
        "MgO": 0.15,
        "FeO": 0.08,
        "CaO": 0.12,
        "Al2O3": 0.12,
        "TiO2": 0.01,
        "Na2O": 0.06,
        "K2O": 0.02,
    }
    for temperature in IMCC_SF04_WORKBOOK_GRID_K:
        default = evaluate_gas(activities, temperature, 1.0e-8, old_pack)
        explicit = evaluate_gas(
            activities, temperature, 1.0e-8, old_pack, gas_species=sf04
        )
        assert tuple(default) == sf04
        assert dict(default) == dict(explicit)
    with pytest.raises(ImccGasSpeciesNotFoundError):
        evaluate_gas(activities, 2000.0, 1.0e-8, old_pack, gas_species=("TiO",))


def load_and_evaluate(pack, activities, temperature, fugacity, species):
    from openimcc.gas import evaluate_gas

    return evaluate_gas(
        activities,
        temperature,
        fugacity,
        pack,
        allow_extrapolation=True,
        gas_species=(species,),
    )[species]


def test_species_thermo_public_api_uses_runtime_rows_and_refuses_extrapolation(
    sf04_gas_pack,
) -> None:
    pack = sf04_gas_pack
    assert isinstance(species_thermo("O2", "g", 1500.0, pack), SpeciesThermo)
    low = species_thermo("O2", "g", 1499.999, pack)
    high = species_thermo("O2", "g", 1500.0, pack)
    assert low.T_interval == 2
    assert high.T_interval == 1
    assert high.source_row_id == "O-029"
    assert high.T_min == 1500.0
    assert high.G_J_mol == pytest.approx(
        high.H_app_kJ_mol * 1000.0 - 1500.0 * high.S_J_molK
    )
    with pytest.raises(FrozenInstanceError):
        high.T_min = 0.0

    assert species_thermo("O2", "g", 2000.0) == species_thermo(
        "O2", "g", 2000.0, pack
    )

    liquid = species_thermo("SiO2", "l", 2000.0, pack)
    crystal = species_thermo("SiO2", "cr", 1500.0, pack)
    default_crystal = species_thermo(
        "SiO2", "cr", 1500.0, load_gas_datapack()
    )
    janaf_liquid = species_thermo("FeO", "l", 2200.0, pack)
    assert liquid.source_row_id == "LAM1987"
    assert crystal.source_row_id == "LAM1987"
    assert default_crystal == crystal
    assert liquid.source_table_id == "LH87 Table 2"
    assert default_crystal.source_table_id == "LH87 Table 2"
    assert liquid.derivatives_fit_implied is True
    assert crystal.derivatives_fit_implied is True
    assert janaf_liquid.derivatives_fit_implied is True
    assert janaf_liquid.source_row_id == "JANAF"
    assert janaf_liquid.source_table_id == "Fe-019"
    assert species_thermo("TiO2", "l", 2000.0, pack).source_table_id == "O-044"
    assert species_thermo("O2", "g", 1500.0, pack).source_table_id == "O-029"
    assert species_thermo("Na2O", "g", 1500.0, pack).source_table_id == "NG-0905"
    assert high.derivatives_fit_implied is False
    assert liquid.T_interval is None
    assert crystal.T_max == 1996.0
    assert liquid.G_J_mol == pytest.approx(
        liquid.H_app_kJ_mol * 1000.0 - 2000.0 * liquid.S_J_molK
    )

    default_k = species_thermo("K2O", "l", 1500.0)
    published_k = species_thermo("K2O", "l", 1500.0, pack)
    assert (default_k.source_row_id, default_k.source_table_id) == (
        "K-012", "K-012"
    )
    assert (published_k.source_row_id, published_k.source_table_id) == (
        "LAM1984", None
    )

    with pytest.raises(ImccGasTemperatureOutsideDomainError):
        species_thermo("O2", "g", 499.0, pack)
    with pytest.raises(ImccGasTemperatureOutsideDomainError):
        species_thermo("SiO2", "l", 1199.0, pack)


def test_default_gas_channels_is_the_public_wrapper_for_private_selector() -> None:
    pack = load_gas_datapack()
    parents = ("Na2O", "TiO2")
    assert default_gas_channels(parents, pack) == _default_reactions(parents, pack)
    assert default_gas_channels(parents) == _default_reactions(parents, pack)
    assert default_gas_channels((), pack) == _default_reactions((), pack)


def test_public_species_thermo_matches_every_in_interval_janaf_gas_cell() -> None:
    """Gate every packaged JANAF gas row at each complete in-range JANAF node.

    H_app is dfH298 + H−H298, exactly the enthalpy represented by the stored
    Shomate row. Residual limits are rounded just above measured worst fits.
    """
    pack = load_gas_datapack()
    checked_rows = 0
    checked_nodes = 0
    overall_max = {key: 0.0 for key in GAS_PROPERTY_RESIDUAL_LIMITS}
    per_row_max = {}

    for species_name, row in pack.gas_df.iterrows():
        table_id = str(row["Ref"])
        if not (JANAF_DATA / f"{table_id}.yaml").is_file():
            # Na2O(g) and K2O(g) use NASA/LH84 source rows, not JANAF tables.
            continue
        species = species_name.removesuffix("(g)")
        interval = int(row["T_interval"])
        low, high = float(row["T_min"]), float(row["T_max"])
        source_rows = _complete_rows(table_id)
        reference_h = next(
            item["formation_enthalpy"]
            for item in source_rows
            if item["temperature"] == 298.15
        )
        ambiguous = _ambiguous_temperatures(table_id)
        row_data = pack.gas_df.loc[[species_name]]
        row_data = row_data.loc[row_data["T_interval"].astype(int) == interval]
        row_pack = replace(pack, gas_df=row_data)
        maxima = {key: 0.0 for key in GAS_PROPERTY_RESIDUAL_LIMITS}
        row_nodes = 0

        for source in source_rows:
            temperature = source["temperature"]
            if not low <= temperature <= high or temperature in ambiguous:
                continue
            actual = species_thermo(species, "g", temperature, row_pack)
            source_h_app = reference_h + source["enthalpy_increment"]
            source_g = source_h_app * 1000.0 - temperature * source["entropy"]
            residuals = {
                "Cp_J_molK": abs(actual.Cp_J_molK - source["heat_capacity"]),
                "S_J_molK": abs(actual.S_J_molK - source["entropy"]),
                "H_app_kJ_mol": abs(actual.H_app_kJ_mol - source_h_app),
                "G_kJ_mol": abs(actual.G_J_mol - source_g) / 1000.0,
            }
            assert actual.source_row_id == table_id
            assert actual.source_table_id == table_id
            assert actual.T_interval == interval
            for key, residual in residuals.items():
                maxima[key] = max(maxima[key], residual)
                overall_max[key] = max(overall_max[key], residual)
            row_nodes += 1

        assert row_nodes > 0, (species_name, interval)
        limits = dict(GAS_PROPERTY_RESIDUAL_LIMITS)
        if species_name == "Co+(g)" and interval == 1:
            # Co+ has a strongly curved Cp table; retain its single 1500-3000 K
            # interval and pin the measured source-node envelope explicitly.
            limits = {
                "Cp_J_molK": 0.75,
                "S_J_molK": 0.035,
                "H_app_kJ_mol": 0.084,
                "G_kJ_mol": 0.0053,
            }
        for key, residual in maxima.items():
            assert residual <= limits[key], (
                species_name,
                interval,
                key,
                residual,
                limits[key],
            )
        checked_rows += 1
        checked_nodes += row_nodes
        per_row_max[(species_name, interval)] = maxima

    # D-020 was already vendored, but its electron gas row is newly included in
    # the packaged fit and therefore joins the source-node residual audit.
    assert checked_rows == 185
    assert checked_nodes == 2268
    assert checked_nodes * 3 == 6804
    assert overall_max["Cp_J_molK"] == pytest.approx(0.7432227428, abs=1e-9)
    assert overall_max["S_J_molK"] == pytest.approx(0.0345525573, abs=1e-9)
    assert overall_max["H_app_kJ_mol"] == pytest.approx(0.0832967405, abs=1e-9)
    assert overall_max["G_kJ_mol"] == pytest.approx(0.0052861248, abs=1e-9)
    assert len(per_row_max) == checked_rows


def test_public_species_thermo_matches_every_janaf_condensate_interval() -> None:
    pack = load_gas_datapack()
    expected_nodes = {
        ("FeO(l)", 1000.0): 39,
        ("TiO2(l)", 1500.0): 15,
        ("Cr2O3(l)", 1900.0): 11,
        ("V2O3(l)", 1700.0): 13,
        ("NbO2(l)", 1200.0): 4,
        ("NbO2(l)", 1500.0): 15,
        ("Na2O(l)", 1200.0): 4,
        ("Na2O(l)", 1500.0): 16,
    }
    checked = {}
    measured_g_maxima = {}

    for species_name, table_id in JANAF_CONDENSATE_TABLE_IDS.items():
        source_rows = _complete_rows(table_id)
        if species_name == "Na2O(l)":
            source_rows = [
                row for row in source_rows if row["temperature"] != 1500.0
            ]
            source_rows.append({
                "temperature": 1500.0,
                "heat_capacity": 104.600,
                "entropy": 260.601,
                "enthalpy_increment": 125.714,
            })
        reference_h = next(
            item["formation_enthalpy"]
            for item in source_rows
            if item["temperature"] == 298.15
        )
        ambiguous = _ambiguous_temperatures(table_id)
        rows = pack.oxide_df.loc[[species_name]]
        for _, row in rows.iterrows():
            if str(row["Ref"]).endswith("-SC-CP"):
                continue
            low, high = float(row["T_min"]), float(row["T_max"])
            row_pack = replace(
                pack,
                oxide_df=rows.loc[rows["T_min"].astype(float) == low],
            )
            maximum_g_residual = 0.0
            count = 0
            for source in source_rows:
                temperature = source["temperature"]
                if not low <= temperature <= high or (
                    temperature in ambiguous
                    and not (species_name == "Na2O(l)" and temperature == 1500.0)
                ):
                    continue
                actual = species_thermo(species_name[:-3], "l", temperature, row_pack)
                source_h_app = reference_h + source["enthalpy_increment"]
                source_g = source_h_app * 1000.0 - temperature * source["entropy"]
                g_residual = abs(actual.G_J_mol - source_g) / 1000.0
                assert actual.source_row_id == str(row["Ref"])
                assert actual.source_table_id == table_id
                assert actual.derivatives_fit_implied
                maximum_g_residual = max(maximum_g_residual, g_residual)
                count += 1

            checked[(species_name, low)] = count
            assert count == expected_nodes[(species_name, low)]
            measured_g_maxima[(species_name, low)] = maximum_g_residual

    assert checked == expected_nodes
    assert measured_g_maxima == pytest.approx(
        JANAF_CONDENSATE_G_RESIDUALS, abs=1.0e-10
    )


def _lam_parent_source_deltas(species_name: str, datapack):
    exception = LAM_PARENT_SOURCE_EXCEPTIONS[species_name]
    row_pack = datapack.oxide_df.loc[[species_name]]
    row_pack = row_pack.loc[row_pack["Ref"] == "LAM1987"]
    assert len(row_pack) == 1
    row = row_pack.iloc[0]
    table_id = exception["table_id"]
    source_rows = _complete_rows(table_id)
    reference_h = next(
        item["formation_enthalpy"]
        for item in source_rows
        if item["temperature"] == 298.15
    )
    ambiguous = _ambiguous_temperatures(table_id)
    row_pack = replace(datapack, oxide_df=row_pack)
    deltas = {key: [] for key in exception["ranges"]}

    for source in source_rows:
        temperature = source["temperature"]
        if (
            temperature < float(row["T_min"])
            or temperature > float(row["T_max"])
            or temperature in ambiguous
        ):
            continue
        actual = species_thermo(species_name[:-3], "l", temperature, row_pack)
        source_h_app = reference_h + source["enthalpy_increment"]
        source_g = source_h_app * 1000.0 - temperature * source["entropy"]
        deltas["G_kJ_mol"].append((actual.G_J_mol - source_g) / 1000.0)
        assert actual.derivatives_fit_implied
        assert actual.source_row_id == str(row["Ref"])
    return deltas


def _assert_lam_parent_source_exception(species_name: str, datapack) -> None:
    exception = LAM_PARENT_SOURCE_EXCEPTIONS[species_name]
    assert exception["sources"]
    assert exception["reason"]
    deltas = _lam_parent_source_deltas(species_name, datapack)
    for property_name, (expected_min, expected_max, tolerance) in exception[
        "ranges"
    ].items():
        values = deltas[property_name]
        assert values
        assert min(values) == pytest.approx(expected_min, abs=tolerance), species_name
        assert max(values) == pytest.approx(expected_max, abs=tolerance), species_name


def test_lam_parent_liquids_pin_janaf_source_disagreements(sf04_gas_pack) -> None:
    pack = sf04_gas_pack
    expected_nodes = {
        "Al2O3(l)": 6,
        "SiO2(l)": 11,
        "MgO(l)": 4,
        "CaO(l)": 8,
    }
    assert set(LAM_PARENT_SOURCE_EXCEPTIONS) == set(expected_nodes)
    for species_name, count in expected_nodes.items():
        deltas = _lam_parent_source_deltas(species_name, pack)
        assert all(len(values) == count for values in deltas.values())
        _assert_lam_parent_source_exception(species_name, pack)


def test_janaf_gate_rejects_both_previously_incorrect_parent_liquid_rows(
    sf04_gas_pack,
) -> None:
    pack = sf04_gas_pack

    na_mutated = pack.oxide_df.copy(deep=True)
    na_rows = na_mutated.index == "Na2O(l)"
    na_mask = na_rows & (na_mutated["T_min"] == 1500.0)
    na_other_interval = na_rows & (na_mutated["T_min"] == 1200.0)
    na_mutated.loc[na_mask, "T_min"] = 825.0
    na_mutated.loc[na_other_interval, ["T_min", "T_max"]] = [2001.0, 3000.0]
    for key, value in zip(
        ("dG_A", "dG_B", "dG_C", "dG_D", "dG_E"),
        (4.82, 19.292, -5.267, 0.623, 0.0),
    ):
        na_mutated.loc[na_mask, key] = value
    old_na_copy = replace(pack, oxide_df=na_mutated)
    source_rows = _complete_rows("Na-013")
    reference_h = next(
        row["formation_enthalpy"]
        for row in source_rows
        if row["temperature"] == 298.15
    )
    source_2000 = next(row for row in source_rows if row["temperature"] == 2000.0)
    actual_2000 = species_thermo("Na2O", "l", 2000.0, old_na_copy)
    source_g_2000 = (
        reference_h + source_2000["enthalpy_increment"]
    ) * 1000.0 - 2000.0 * source_2000["entropy"]
    with pytest.raises(AssertionError):
        assert actual_2000.G_J_mol == pytest.approx(source_g_2000, abs=3.4)

    al_mutated = pack.oxide_df.copy(deep=True)
    al_mutated.loc[al_mutated.index == "Al2O3(l)", "dH298_R"] = -188.14
    with pytest.raises(AssertionError):
        _assert_lam_parent_source_exception(
            "Al2O3(l)", replace(pack, oxide_df=al_mutated)
        )
