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
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tools import build_gas_tables
from openimcc.gas import (
    _EXTERNAL_PACK_GAS_SPECIES,
    _GAS_PROVENANCE_AUTHORITY,
    _OXIDE_PROVENANCE_AUTHORITY,
    IMCC_GAS_CHANNEL_SPECIES,
    IMCC_SF04_WORKBOOK_GRID_K,
    R_J_MOL_K,
    _SF04_REACTIONS,
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
LH84_DATA = ROOT / "data-src" / "lh84"
PROVENANCE_PATH = GAS_DATA / "PROVENANCE.yaml"
_BASE_INTERVAL_1_SHA256 = "ea3a8db1419140329e36be433453fdc67969a7e3826d9af5a310d284a1526c0a"

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
}

NASA_TABLE_IDS = {"Na2O": "NG-0905", "K2O": "NG-0760"}

PARENT_TABLE_IDS = {
    "Na2O": "Na-013",
    "K2O": "K-012",  # JANAF has crystal K2O only; K2O(l) is LH84 secondary.
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
    gas_rows = {
        row["species_name"]: row
        for row in gas_records
        if row["T_range_K"][0] == 1500
    }
    low_gas_rows = {
        row["species_name"].removesuffix("(g)"): row
        for row in gas_records
        if row.get("T_interval") == 2
    }
    oxide_rows = {}
    for row in rows:
        if row["table"] != "condensate":
            continue
        current = oxide_rows.get(row["species_name"])
        if current is None or row["T_range_K"][1] > current["T_range_K"][1]:
            oxide_rows[row["species_name"]] = row
    public_gas_channels = set(IMCC_GAS_CHANNEL_SPECIES) - {
        "MnO",
        "NiO",
        "CoO",
    }
    assert set(gas_rows) == {f"{name}(g)" for name in public_gas_channels}
    assert set(low_gas_rows) == build_gas_tables.LOW_T_GAS_SPECIES | {"Na2O", "K2O"}
    assert len(oxide_rows) == 12

    for species, table_id in GAS_TABLE_IDS.items():
        source = _record(table_id)
        row = gas_rows[f"{species}(g)"]
        assert row["table_id"] == table_id
        assert row["source_path"] == f"data-src/janaf/{table_id}.yaml"
        assert (ROOT / row["source_path"]).is_file()
        assert row["source_sha256"] == source["extraction"]["source_sha256"]
        assert row["authority"] == "janaf_fitted"
        assert row["method"] == "fitted"
        expected_t_max = build_gas_tables._FIT_T_MAX_BY_TABLE.get(
            table_id, build_gas_tables.FIT_T_MAX
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
        assert row["T_range_K"] == [int(expected_t_min), 3000]
        assert source["source"]["doi"] == "10.18434/T42S31"

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
    assert k_row["authority"] == "secondary_transcription_unverified_primary"
    assert k_row["method"] == "transcribed_secondary_unverified_primary"
    assert k_row["source"]["primary_doi"] == "10.1063/1.555706"
    assert "VapoRock" in k_row["source"]["transcription"]


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


def test_runtime_provenance_mirror_matches_yaml() -> None:
    provenance = yaml.safe_load(PROVENANCE_PATH.read_text(encoding="utf-8"))
    expected_gas = {}
    expected_oxide = {}
    for row in provenance["rows"]:
        if row["table"] == "gas":
            expected_gas[row["species_name"].removesuffix("(g)")] = row["authority"]
        else:
            name = row["species_name"]
            expected_oxide[name.removesuffix("(l)")] = row["authority"]

    # This is the deliberate source-of-truth boundary: the runtime remains
    # free of a YAML dependency, while this test compares every gas and oxide
    # provenance row against the checked-in YAML.  SiO2(cr) is retained in the
    # mirror for parity even though the SF04 reaction set consumes SiO2(l).
    public_runtime_gas = {
        species: authority
        for species, authority in _GAS_PROVENANCE_AUTHORITY.items()
        if species not in _EXTERNAL_PACK_GAS_SPECIES
    }
    public_runtime_oxide = {
        species: authority
        for species, authority in _OXIDE_PROVENANCE_AUTHORITY.items()
        if species not in _EXTERNAL_PACK_GAS_SPECIES
    }
    assert public_runtime_gas == expected_gas
    assert public_runtime_oxide == expected_oxide


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
        if row["table"] == "gas" and row.get("T_interval") == 2
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
    } | {"Na2O(g)", "K2O(g)"}
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
        for coefficient in "ABCDEFG":
            assert packaged_low_rows[f"{species}(g)"][coefficient] == expected[
                coefficient
            ]

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
            float(recorded["max_residual_J_per_mol"]), abs=1.0e-9
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
    for line in lines[1:]:
        interval = next(csv.reader([line.decode("utf-8")]))[2]
        intervals.append(interval)
        if interval == "1":
            retained.append(line)

    assert intervals[: intervals.index("2")] == ["1"] * 43
    assert all(interval == "2" for interval in intervals[intervals.index("2") :])
    assert hashlib.sha256(b"".join(retained)).hexdigest() == _BASE_INTERVAL_1_SHA256


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
    pack = load_gas_datapack()
    gas_row = _gas_row_for_interval(pack, "Na2O", 1)
    oxide_row = pack.oxide_df.loc["Na2O(l)"]
    temperature = 2000.0
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
        row = pack.oxide_df.loc[[species]]
        row = row.loc[row["T_min"].astype(float) == fit_t_min].iloc[0]
        maximum_j = 0.0
        maximum_log10 = 0.0
        nodes = 0
        for source_row in _complete_rows(table_id):
            temperature = source_row["temperature"]
            if (
                not fit_t_min <= temperature <= 3000.0
                or temperature in _ambiguous_temperatures(table_id)
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
            "V2O3(l)": 14,
            "NbO2(l)": 15,
        }[species]
        assert nodes == expected_nodes
        max_log10_limit = {
            "TiO2(l)": 0.001,
            "Cr2O3(l)": 0.001,
            "V2O3(l)": 0.001,
            "NbO2(l)": 0.001,
        }[species]
        max_j_limit = {
            "TiO2(l)": 10.0,
            "Cr2O3(l)": 10.0,
            "V2O3(l)": 10.0,
            "NbO2(l)": 10.0,
        }[species]
        assert maximum_log10 <= max_log10_limit
        assert maximum_j < max_j_limit
        assert maximum_j == pytest.approx(
            recorded[(species, (int(fit_t_min), 3000))][
                "max_residual_J_per_mol"
            ],
            abs=1e-6,
        )
        assert maximum_log10 == pytest.approx(
            recorded[(species, (int(fit_t_min), 3000))]["max_residual_log10_K"],
            abs=1e-9,
        )


def test_no_fitted_row_consumes_a_parse_ambiguous_source_row() -> None:
    for table_id in (*GAS_TABLE_IDS.values(), *FITTED_CONDENSATE_TABLE_IDS.values()):
        fit_t_min = build_gas_tables._FIT_T_MIN_BY_TABLE.get(
            table_id, build_gas_tables.FIT_T_MIN
        )
        selected_temperatures = {
            row["temperature"]
            for row in _complete_rows(table_id)
            if (
                fit_t_min
                <= row["temperature"]
                <= build_gas_tables._FIT_T_MAX_BY_TABLE.get(
                    table_id, build_gas_tables.FIT_T_MAX
                )
                and row["temperature"] not in _ambiguous_temperatures(table_id)
            )
        }
        assert selected_temperatures.isdisjoint(_ambiguous_temperatures(table_id))


def _janaf_apparent_gibbs(table_id: str, T: float) -> float:
    """G_app = dfH(298) + [H(T) - H(298)] - T*S(T), in J/mol, from JANAF cells."""
    rows = _complete_rows(table_id)
    reference = next(row for row in rows if row["temperature"] == 298.15)
    row = next(row for row in rows if row["temperature"] == T)
    return (
        reference["formation_enthalpy"] + row["enthalpy_increment"]
    ) * 1000.0 - T * row["entropy"]


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

    # On this NumPy/LAPACK build, the largest relative difference across
    # generated gas A-G and condensate dG_A-E coefficients was 1.82e-11
    # (Fe(g).D). A 10x margin is 1.82e-10; use 2e-10 to allow minor BLAS
    # rounding changes while keeping all non-fit values exact.
    # NG-* rows come from smooth NASA-7 polynomials; their Shomate fit is
    # ill-conditioned in coefficient directions that barely change G(T), so
    # compare G(T) over the fit grid to 1e-6 J/mol instead of comparing bytes.
    fit_relative_tolerance = 2e-10

    def assert_table_matches(
        generated_path: Path,
        packaged_path: Path,
        fitted_columns: dict[str, tuple[str, ...]],
        gibbs_species: frozenset[str] = frozenset(),
    ) -> None:
        generated_table = pd.read_csv(generated_path, dtype=str, keep_default_na=False)
        packaged_table = pd.read_csv(packaged_path, dtype=str, keep_default_na=False)
        assert generated_table.columns.tolist() == packaged_table.columns.tolist()
        generated_species = generated_table["species_name"].tolist()
        packaged_species = packaged_table["species_name"].tolist()
        assert generated_species == packaged_species

        for row_index, species in enumerate(packaged_species):
            fit_columns = fitted_columns.get(species, ())
            assert set(fit_columns) <= set(packaged_table.columns)
            for column in packaged_table.columns:
                generated_value = generated_table.iloc[row_index][column]
                packaged_value = packaged_table.iloc[row_index][column]
                if species in gibbs_species and column in "ABCDEFGH":
                    continue
                if column in fit_columns:
                    generated_number = float(generated_value)
                    packaged_number = float(packaged_value)
                    if packaged_number == 0.0:
                        assert generated_number == packaged_number
                    else:
                        assert abs(generated_number - packaged_number) <= (
                            fit_relative_tolerance * abs(packaged_number)
                        )
                else:
                    assert generated_value == packaged_value

            if species in gibbs_species:
                generated_row = generated_table.iloc[row_index].to_dict()
                packaged_row = packaged_table.iloc[row_index].to_dict()
                for temperature in np.arange(
                    float(generated_row["T_min"]),
                    float(generated_row["T_max"]) + 1.0,
                    100.0,
                ):
                    assert abs(
                        _shomate_g(generated_row, temperature)
                        - _shomate_g(packaged_row, temperature)
                    ) < 1.0e-6

    # Start from the packaged table with fitted rows removed, so the generator
    # must recreate them while passing every transcribed row through exactly.
    packaged_condensate_bytes = packaged_condensate.read_bytes()
    fitted_prefixes = tuple(
        f"{species},".encode() for species in FITTED_CONDENSATE_TABLE_IDS
    )
    condensate.write_bytes(
        b"".join(
            line
            for line in packaged_condensate_bytes.splitlines(keepends=True)
            if not line.startswith(fitted_prefixes)
        )
    )
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
    second = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    assert first.stdout == second.stdout
    assert generated.read_bytes() == first_gas_bytes
    assert condensate.read_bytes() == first_condensate_bytes

    gas_table = pd.read_csv(packaged_gas, dtype=str, keep_default_na=False)
    gibbs_species = frozenset(
        gas_table.loc[gas_table["Ref"].str.startswith("NG-"), "species_name"]
    )
    gas_fit_columns = {
        species: tuple("ABCDEFG")
        for species in gas_table["species_name"]
        if species not in gibbs_species
    }
    condensate_fit_columns = {
        species: tuple(f"dG_{coefficient}" for coefficient in "ABCDE")
        for species in FITTED_CONDENSATE_TABLE_IDS
    }
    assert_table_matches(
        generated, packaged_gas, gas_fit_columns, gibbs_species=gibbs_species
    )
    assert_table_matches(condensate, packaged_condensate, condensate_fit_columns)

def test_research_condensate_rows_match_janaf_fits() -> None:
    pack_dir = (
        ROOT
        / "src"
        / "openimcc"
        / "data"
        / "packs"
        / "gas-janaf-parent-liquids-research"
    )
    generated = {
        row["species_name"]: row
        for row in build_gas_tables.build_research_condensate_rows(JANAF_DATA)
    }
    packaged = {
        row["species_name"]: row
        for row in csv.DictReader(
            io.StringIO((pack_dir / "condensate.csv").read_text(encoding="utf-8"))
        )
    }
    assert set(generated) == {"SiO2(l)", "Al2O3(l)", "MgO(l)", "CaO(l)"}
    assert set(generated) <= set(packaged)
    for species, fitted in generated.items():
        assert {
            column: fitted[column] for column in build_gas_tables.CONDENSATE_COLUMNS
        } == {
            column: packaged[species][column]
            for column in build_gas_tables.CONDENSATE_COLUMNS
        }
    provenance = yaml.safe_load((pack_dir / "PROVENANCE.yaml").read_text())
    assert provenance["pack_id"] == pack_dir.name
    assert provenance["pack_class"] == "research"
    records = {row["species_name"]: row for row in provenance["rows"]}
    assert set(records) == set(generated)
    for species, fitted in generated.items():
        source = build_gas_tables._load_record(
            JANAF_DATA / f"{fitted['Ref']}.yaml"
        )
        assert records[species]["authority"] == "janaf_fitted"
        assert records[species]["classification"] == "research"
        assert records[species]["table_id"] == fitted["Ref"]
        assert records[species]["source_sha256"] == source["extraction"]["source_sha256"]
        assert records[species]["user_agent"] == source["extraction"]["user_agent"]


def _shomate_g(row: dict[str, str], T: float) -> float:
    A, B, C, D, E, F, G, H = (float(row[key]) for key in "ABCDEFGH")
    t = T / 1000.0
    enthalpy = A * t + B * t**2 / 2 + C * t**3 / 3 + D * t**4 / 4 - E / t + F - H
    entropy = A * math.log(t) + B * t + C * t**2 / 2 + D * t**3 / 3 - E / (2 * t**2) + G
    return enthalpy * 1000.0 - T * entropy


_CONDENSATE_EXPECTED = {
    "MgO(l)": (3100, 3500, -72.34, -0.804, 5.1067, -0.3615, 0.0, 0.0, "LAM1987"),
    "CaO(l)": (2900, 3800, -76.384, 0.924, 5.922, -0.7316, 0.0436, 0.0, "LAM1987"),
    "Al2O3(l)": (
        2327,
        3000,
        -188.14,
        232.345,
        -336.622,
        193.672,
        -48.1032,
        4.4461,
        "LAM1987",
    ),
    "SiO2(l)": (1996, 3000, -109.53, 2.12, 7.6492, -1.2588, 0.0998, 0.0, "LAM1987"),
    "SiO2(cr)": (1000, 1996, -109.53, 1.937, 8.59, -1.935, 0.225, 0.0, "LAM1987"),
    "Na2O(l)": (825, 3000, -50.17, 4.82, 19.292, -5.267, 0.623, 0.0, "LAM1984"),
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
        1500,
        3000,
        -131.464178771295,
        12.7984408682205,
        15.7012635413964,
        -3.18957608585823,
        0.420979101959005,
        -0.0242973638300596,
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


def test_condensate_rows_match_the_current_published_coefficients() -> None:
    table = pd.read_csv(GAS_DATA / "condensate.csv").set_index("species_name")
    assert set(table.index) == set(_CONDENSATE_EXPECTED)
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
    seam_log10 = (
        _lamor_gibbs(1500.0, low) - _lamor_gibbs(1500.0, high)
    ) / (R_J_MOL_K * 1500.0 * math.log(10.0))
    assert abs(seam_log10) < 0.001
    assert int(_oxide_row_for_T(pack.oxide_df, "NbO2(l)", 1499.9)["T_min"]) == 1200
    assert int(_oxide_row_for_T(pack.oxide_df, "NbO2(l)", 1500.0)["T_min"]) == 1500


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
        }:
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

    # Liquid-only starts for Al2O3, Cr2O3, SiO2, MgO and CaO omit source rows
    # before their selected branches, leaving 420 complete reaction nodes.
    assert checked == 420
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
    also leave Al source comparisons below 2500 K, Si below 1800 K, and Mg/Ca
    below 2200 K. K2O's K-012 parent ends at 2000 K, so K, K2 and KO have no
    independent parent reference at 2125, 2250, 2375 or 2500 K.
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
        }:
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
    assert checked == 242
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
    }
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
