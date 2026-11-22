"""Opt-in species thermochemistry continuation follows the gas runtime path."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pytest
import yaml

from openimcc._crystal_interpolation import _linear_value_and_slope
import openimcc.crystal_species as crystal_species
import openimcc.janaf_crystals as janaf_crystals
from openimcc.gas import (
    ImccGasCrystalPolymorphError,
    ImccGasSpeciesNotFoundError,
    ImccGasTemperatureOutsideDomainError,
    SpeciesThermo,
    _janaf_gibbs,
    _lamor_gibbs,
    _nearest_interval_row,
    _oxide_row_for_T,
    evaluate_gas,
    load_gas_datapack,
    species_thermo,
)
from openimcc.crystal_species import (
    _barin_phases,
    _berman_phases,
    _cp_primitive,
    _berman_properties,
    _formula_element_counts,
    CrystalSpeciesTemperatureError,
    evaluate_crystal_species,
)
from openimcc.janaf_crystals import evaluate_janaf_crystal


def _value_hexes(value: SpeciesThermo) -> dict[str, str]:
    return {
        name: getattr(value, name).hex()
        for name in (
            "Cp_J_molK",
            "S_J_molK",
            "H_app_kJ_mol",
            "G_J_mol",
        )
    }


@lru_cache(maxsize=1)
def _crystal_dataset() -> dict:
    path = Path(__file__).parents[1] / "src/openimcc/data/diagnostics/crystal-phases.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _element_reference_sum_kJ_mol(formula: str, temperature_K: float) -> float | None:
    references = _crystal_dataset()["element_reference_g_app_nodes_kJ_mol"]
    atoms = {
        element: int(count or "1")
        for element, count in re.findall(r"([A-Z][a-z]?)([0-9]*)", formula)
    }
    total = 0.0
    node = str(int(temperature_K))
    for element, count in atoms.items():
        if element == "O":
            element, count = "O2", count / 2.0
        value = references.get(element, {}).get(node)
        if value is None:
            return None
        total += count * float(value)
    return total


_JANAF_ELEMENT_GAS_TABLES = {
    "Al": "Al-005",
    "B": "B-005",
    "Br": "Br-001",
    "Ca": "Ca-006",
    "C": "C-002",
    "Cl": "Cl-001",
    "Co": "Co-005",
    "Cr": "Cr-005",
    "Cu": "Cu-005",
    "F": "F-001",
    "Fe": "Fe-004",
    "Ga": "Ga-005",
    "H": "H-001",
    "I": "I-001",
    "K": "K-005",
    "Li": "Li-005",
    "Mn": "Mn-005",
    "Mg": "Mg-005",
    "Na": "Na-005",
    "Ni": "Ni-005",
    "Mo": "Mo-004",
    "O2": "O-029",
    "P": "P-001",
    "Si": "Si-005",
    "S": "S-006",
    "Ta": "Ta-005",
    "Ti": "Ti-006",
    "W": "W-005",
}


def _janaf_text_reference_row(
    raw_line: str,
) -> dict[str, float | bool | None] | None:
    cells = raw_line.strip().split("\t")
    try:
        temperature, entropy, enthalpy_increment = (
            float(cells[index]) for index in (0, 2, 4)
        )
    except (IndexError, ValueError):
        return None

    formation_enthalpy = formation_gibbs = log10_kf = None
    packed_formation_columns = False
    if len(cells) >= 7:
        try:
            formation_enthalpy = float(cells[5])
            formation_gibbs = float(cells[6])
        except ValueError:
            pass
    if len(cells) >= 8:
        try:
            log10_kf = float(cells[7])
        except ValueError:
            pass
    if len(cells) == 6:
        tail = cells[5].strip()
        if "<-->" in tail or "TRANSITION" in tail.upper():
            pass
        else:
            values = re.findall(
                r"(?<![A-Za-z])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?",
                tail,
            )
            if len(values) >= 3:
                # NIST's export packs three formation columns into one cell on
                # a few rows. Their delta-f H/G signs cannot be read reliably;
                # keep the printed log Kf available as an independent check.
                formation_enthalpy = float(values[0])
                formation_gibbs = float(values[1])
                log10_kf = float(values[2])
                packed_formation_columns = True
            elif not values:
                return None

    return {
        "T_K": temperature,
        "Hinc_kJ_mol": enthalpy_increment,
        "S_J_mol_K": entropy,
        "delta_fH_kJ_mol": formation_enthalpy,
        "delta_fG_kJ_mol": formation_gibbs,
        "log10_Kf": log10_kf,
        "packed_formation_columns": packed_formation_columns,
    }


@lru_cache(maxsize=None)
def _janaf_element_reference_rows(element: str) -> tuple[dict[str, float], ...]:
    table_id = _JANAF_ELEMENT_GAS_TABLES[element]
    root = Path(__file__).parents[1] / "data-src/janaf"
    yaml_path = root / f"{table_id}.yaml"
    if yaml_path.exists():
        table = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))["table"]
        rows = []
        for source_row in table["values"]:
            rows.append({
                "T_K": float(source_row["temperature"]["value"]),
                "Hinc_kJ_mol": source_row["enthalpy_increment"]["value"],
                "S_J_mol_K": source_row["entropy"]["value"],
                "delta_fH_kJ_mol": source_row["formation_enthalpy"]["value"],
                "delta_fG_kJ_mol": source_row["formation_gibbs_energy"]["value"],
            })
        rows.extend(
            row
            for ambiguity in table.get("parse_ambiguities", ())
            if (row := _janaf_text_reference_row(ambiguity["raw_line"])) is not None
        )
    else:
        text_path = root / f"{table_id}.txt"
        rows = [
            row
            for line in text_path.read_text(encoding="utf-8").splitlines()[2:]
            if (row := _janaf_text_reference_row(line)) is not None
        ]
    return tuple(sorted(rows, key=lambda row: row["T_K"]))


def _janaf_element_reference_g_app_kJ_mol(
    element: str, temperature_K: float
) -> float:
    rows = _janaf_element_reference_rows(element)
    anchor = next(
        row for row in rows
        if row["T_K"] == 298.15 and row["delta_fH_kJ_mol"] is not None
    )

    def value_at(column: str) -> float:
        points = {
            row["T_K"]: row[column]
            for row in rows
            if row[column] is not None
        }
        temperatures = sorted(points)
        if temperature_K in points:
            return float(points[temperature_K])
        upper = next(
            (index for index, value in enumerate(temperatures)
             if value > temperature_K),
            None,
        )
        if upper is None or upper == 0:
            raise ValueError(
                f"{element} JANAF reference has no {column} at "
                f"{temperature_K:g} K"
            )
        low_T, high_T = temperatures[upper - 1], temperatures[upper]
        low_value, high_value = points[low_T], points[high_T]
        return float(low_value) + (float(high_value) - float(low_value)) * (
            (temperature_K - low_T) / (high_T - low_T)
        )

    return (
        float(anchor["delta_fH_kJ_mol"])
        + value_at("Hinc_kJ_mol")
        - temperature_K * value_at("S_J_mol_K") / 1000.0
        - value_at("delta_fG_kJ_mol")
    )


def _janaf_element_reference_sum_kJ_mol(
    formula: str, temperature_K: float
) -> float:
    return math.fsum(
        (float(count) / 2.0 if element == "O" else float(count))
        * _janaf_element_reference_g_app_kJ_mol(
            "O2" if element == "O" else element,
            temperature_K,
        )
        for element, count in _formula_element_counts(formula).items()
    )


def test_species_thermo_in_range_values_remain_byte_identical() -> None:
    legacy = SpeciesThermo(1.0, 2.0, 3.0, 4.0, "row", None, None, 10.0, 20.0)
    assert legacy.extrapolated is False
    assert legacy.continuation is None
    assert legacy.authority is None
    assert legacy.phase_identity is None
    assert legacy.formation_enthalpy_298_kJ_mol is None
    assert legacy.formation_enthalpy_source_id is None

    pack = load_gas_datapack()
    gas = species_thermo("O2", "g", 2000.0, pack)
    liquid = species_thermo("SiO2", "l", 2000.0, pack)
    crystal = species_thermo("SiO2", "cr", 1500.0, pack)

    assert _value_hexes(gas) == {
        "Cp_J_molK": "0x1.2def84315b766p+5",
        "S_J_molK": "0x1.0cbf4b7f5a86ep+8",
        "H_app_kJ_mol": "0x1.d9652d0829742p+5",
        "G_J_mol": "-0x1.d31c05b57dbffp+18",
    }
    assert _value_hexes(liquid) == {
        "Cp_J_molK": "0x1.4574931af766bp+6",
        "S_J_molK": "0x1.5994ed1502996p+7",
        "H_app_kJ_mol": "-0x1.86aa470158b5ep+9",
        "G_J_mol": "-0x1.1320008d49734p+20",
    }
    assert _value_hexes(crystal) == {
        "Cp_J_molK": "0x1.240610b27dc5cp+6",
        "S_J_molK": "0x1.25a4de21ba0e6p+7",
        "H_app_kJ_mol": "-0x1.9ddd493cd318dp+9",
        "G_J_mol": "-0x1.ffb339de40090p+19",
    }
    assert gas.extrapolated is False
    assert gas.declared_interval_K == (1500.0, 3000.0)
    assert gas.continuation is None
    assert gas.authority
    assert liquid.extrapolated is False
    assert liquid.declared_interval_K == (1996.0, 3000.0)
    assert liquid.continuation is None
    assert liquid.authority
    assert crystal.extrapolated is False
    assert crystal.declared_interval_K == (1000.0, 1996.0)
    assert crystal.continuation is None
    assert crystal.authority

    with pytest.raises(FrozenInstanceError):
        gas.extrapolated = True


def test_existing_condensate_crystal_queries_remain_byte_identical() -> None:
    pack = load_gas_datapack()
    expected = {
        ("C", 1500.0): {
            "Cp_J_molK": "0x1.7e2bd59b3662dp+4",
            "S_J_molK": "0x1.0da84dd7860b3p+5",
            "H_app_kJ_mol": "0x1.73c8106d45f13p+4",
            "G_J_mol": "-0x1.aaf1b402b4672p+14",
        },
        ("Os", 3000.0): {
            "Cp_J_molK": "0x1.3b5e920517bc2p+5",
            "S_J_molK": "0x1.8d744eb6b7757p+6",
            "H_app_kJ_mol": "0x1.535d002836dc8p+6",
            "G_J_mol": "-0x1.a080f63a03f39p+17",
        },
    }
    for (formula, temperature), values in expected.items():
        assert _value_hexes(
            species_thermo(formula, "cr", temperature, pack)
        ) == values


def test_shipped_crystal_dataset_fallback_and_polymorph_selection() -> None:
    janaf = species_thermo("TiO2", "cr", 1500.0)
    assert janaf.source_table_id == "O-043"
    assert janaf.source_row_id == "O-043"
    assert janaf.phase_identity is not None
    assert janaf.phase_identity["dataset"] == "JANAF pure-oxide crystals"
    assert janaf.G_J_mol == evaluate_janaf_crystal("TiO2", 1500.0).G_J_mol

    barin = species_thermo("CaAl2O4", "cr", 1800.0)
    assert barin.source_table_id == "Barin-1995"
    assert barin.source_row_id == "CA/SOL"
    assert barin.Cp_J_molK == 193.101
    assert barin.phase_identity is not None
    assert barin.phase_identity["polymorph"] == "SOL"
    assert barin.phase_identity["declared_interval_K"] == (298.15, 1800.0)
    assert barin.phase_identity["source_solid_limit_kind"] == "decomposition"

    berman = species_thermo("CaSiO3", "cr", 1500.0)
    wollastonite = species_thermo(
        "CaSiO3", "cr", 1500.0, polymorph="wollastonite"
    )
    assert berman.source_table_id == "Berman-1988"
    assert berman.phase_identity is not None
    assert berman.phase_identity["polymorph"] == "pseudowollastonite"
    assert "lowest apparent G" in berman.phase_identity["selection_rule"]
    assert wollastonite.phase_identity is not None
    assert wollastonite.phase_identity["polymorph"] == "wollastonite"
    assert berman.G_J_mol < wollastonite.G_J_mol

    robie_hemingway = species_thermo("CaTiO3", "cr", 1500.0)
    assert robie_hemingway.source_table_id == "Robie-Hemingway-1995"
    assert robie_hemingway.phase_identity is not None
    assert robie_hemingway.phase_identity["polymorph"] == "perovskite"

    with pytest.raises(ImccGasCrystalPolymorphError) as refusal:
        species_thermo("CaSiO3", "cr", 1500.0, polymorph="quartz")
    assert "wollastonite" in str(refusal.value)
    assert "pseudowollastonite" in str(refusal.value)


def test_parenthesized_formula_matches_canonical_dataset_formula() -> None:
    diopside = species_thermo("CaMgSi2O6", "cr", 1500.0)
    grouped_formula = species_thermo("CaMg(SiO3)2", "cr", 1500.0)

    assert grouped_formula == diopside
    assert grouped_formula.source_row_id == "diopside"


def test_formula_parser_accepts_dotted_hydrates() -> None:
    expected = _formula_element_counts("MgSO4·7H2O")

    assert expected == {"Mg": 1, "S": 1, "O": 11, "H": 14}
    assert _formula_element_counts("MgSO4.7H2O") == expected
    assert _formula_element_counts("MgSO4*7H2O") == expected
    assert _formula_element_counts("Fe2.5O") == {
        "Fe": Fraction(5, 2), "O": 1
    }


def test_janaf_crystal_generator_check_matches_packaged_data() -> None:
    root = Path(__file__).parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "tools/generate_declared_assemblage_janaf.py"),
            "--check",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_janaf_liquid_fallback_uses_printed_segments_and_strict_limits() -> None:
    root = Path(__file__).parents[1]
    data = json.loads(
        (
            root / "src/openimcc/data/diagnostics/janaf-liquid-species.json"
        ).read_text(encoding="utf-8")
    )
    expected_limits = {
        "Mn": ("Mn-004", (1519.0, 3000.0)),
        "V2O5": ("O-085", (943.0, 3000.0)),
    }
    rows = {row["formula"]: row for row in data["rows"]}

    for formula, (table_id, interval) in expected_limits.items():
        assert tuple(rows[formula]["temperature_K"]) == interval
        assert rows[formula]["nodes"][-1]["T_K"] == interval[1]
        assert "LIQUID <--> IDEAL GAS" not in {
            node.get("label") for node in rows[formula]["nodes"]
        }
        result = species_thermo(formula, "l", interval[0])
        assert result.source_row_id == table_id
        assert result.source_table_id == table_id
        assert result.declared_interval_K == interval
        assert result.phase_identity["label"] == "LIQUID"
        assert (
            result.phase_identity["source_title"]
            == rows[formula]["source_title"]
        )
        assert result.authority == "janaf_transcribed"
        assert result.continuation is None
        assert (
            species_thermo(formula, "l", interval[1]).declared_interval_K
            == interval
        )
        with pytest.raises(ImccGasTemperatureOutsideDomainError):
            species_thermo(
                formula, "l", interval[0] - 0.01, allow_extrapolation=True
            )
        with pytest.raises(ImccGasTemperatureOutsideDomainError):
            species_thermo(
                formula, "l", interval[1] + 0.01, allow_extrapolation=True
            )

    aluminum = species_thermo("Al", "l", 1000.0)
    aluminum_source = rows["Al"]
    aluminum_nodes = aluminum_source["nodes"]
    h_increment, h_slope = _linear_value_and_slope(
        [
            (node["T_K"], node["enthalpy_increment_kJ_mol"])
            for node in aluminum_nodes
        ],
        1000.0,
    )
    entropy, _ = _linear_value_and_slope(
        [(node["T_K"], node["entropy_J_mol_K"]) for node in aluminum_nodes],
        1000.0,
    )
    enthalpy_app = aluminum_source["formation_enthalpy_298_kJ_mol"] + h_increment
    assert aluminum.Cp_J_molK == 1000.0 * h_slope
    assert aluminum.S_J_molK == entropy
    assert aluminum.H_app_kJ_mol == enthalpy_app
    assert aluminum.G_J_mol == 1000.0 * enthalpy_app - 1000.0 * entropy
    source = _oxide_row_for_T(
        load_gas_datapack().oxide_df, "SiO2(l)", 1500.0
    )
    assert aluminum.source_table_id == "Al-001"
    assert aluminum.authority == "janaf_transcribed"
    assert aluminum.phase_identity["label"] == "LIQUID"
    assert aluminum.phase_identity["printed_range_K"] == (933.45, 2790.812)
    assert aluminum.phase_identity["temperature_K"] == (933.45, 2790.812)
    assert species_thermo("SiO2", "l", 1500.0).source_row_id == str(source["Ref"])
    with pytest.raises(ImccGasTemperatureOutsideDomainError):
        species_thermo("SiO2", "l", 1000.0)


def test_janaf_liquid_nodes_match_printed_rows() -> None:
    root = Path(__file__).parents[1]
    data = json.loads(
        (
            root / "src/openimcc/data/diagnostics/janaf-liquid-species.json"
        ).read_text(encoding="utf-8")
    )

    assert len(data["rows"]) == 32
    new_table_ids = {
        "Na-003", "Ga-003", "Ca-003", "Li-003", "Cu-002", "Si-002",
        "Br-039", "Ni-002", "Fe-004", "Co-002", "I-025", "Ti-003",
        "Cl-094", "F-034", "Na-024", "H-063", "H-075", "Cl-162",
        "B-126", "F-152", "C-146", "H-086",
    }
    for row in data["rows"]:
        formula = row["formula"]
        formation_enthalpy = float(row["formation_enthalpy_298_kJ_mol"])
        verify_source = row["table_id"] in new_table_ids
        source_rows = {}
        recovered_temperatures = set()
        if verify_source:
            source_path = root / "data-src/janaf" / f"{row['table_id']}.txt"
            source_raw = source_path.read_bytes()
            source_lines = source_raw.decode("utf-8").splitlines()
            source_rows = {
                line_number: parsed
                for line_number, raw_line in enumerate(source_lines, start=1)
                if (parsed := _janaf_text_reference_row(raw_line)) is not None
            }
            assert row["source_sha256"] == hashlib.sha256(source_raw).hexdigest()
            assert row["source_path"] == str(source_path.relative_to(root))
            assert row["source_title"] == " | ".join(
                source_lines[0].split("\t")
            )
        assert row["temperature_K"][0] < row["temperature_K"][1]
        if verify_source:
            source_anchor = next(
                parsed for parsed in source_rows.values()
                if parsed["T_K"] == 298.15
            )
            expected_formation_enthalpy = source_anchor["delta_fH_kJ_mol"]
            if source_anchor["packed_formation_columns"]:
                next_printed_enthalpy = next(
                    float(parsed["delta_fH_kJ_mol"])
                    for parsed in source_rows.values()
                    if parsed["T_K"] > 298.15
                    and parsed["delta_fH_kJ_mol"] is not None
                    and not parsed["packed_formation_columns"]
                )
                expected_formation_enthalpy = math.copysign(
                    abs(float(expected_formation_enthalpy)),
                    next_printed_enthalpy,
                )
            assert formation_enthalpy == expected_formation_enthalpy
            recovered_temperatures = {
                float(source_rows[line_number]["T_K"])
                for line_number in source_rows
                if source_rows[line_number]["packed_formation_columns"]
            }
        for node in row["nodes"]:
            temperature = float(node["T_K"])
            result = species_thermo(formula, "l", temperature)
            if verify_source:
                source_line_number = int(node["source_line_number"])
                source_line = source_lines[source_line_number - 1]
                source_fields = source_line.split("\t")
                assert temperature == float(source_fields[0])
                assert float(node["entropy_J_mol_K"]) == float(
                    source_fields[2]
                )
                assert float(node["enthalpy_increment_kJ_mol"]) == float(
                    source_fields[4]
                )
            assert row["temperature_K"][0] <= temperature <= row[
                "temperature_K"
            ][1]
            enthalpy = formation_enthalpy + float(
                node["enthalpy_increment_kJ_mol"]
            )
            entropy = float(node["entropy_J_mol_K"])
            assert result.source_table_id == row["table_id"]
            assert result.H_app_kJ_mol == pytest.approx(enthalpy, abs=1e-12)
            assert result.S_J_molK == pytest.approx(entropy, abs=1e-12)
            assert result.G_J_mol == pytest.approx(
                1000.0 * enthalpy - temperature * entropy, abs=1e-9
            )
            if not verify_source:
                continue
            source_row = source_rows.get(source_line_number)
            if source_row is None or source_row["delta_fG_kJ_mol"] is None:
                continue
            absolute_g = enthalpy - temperature * entropy / 1000.0
            element_reference = _janaf_element_reference_sum_kJ_mol(
                formula, temperature
            )
            reconstructed_delta_f_g = absolute_g - element_reference
            printed_delta_f_g = float(source_row["delta_fG_kJ_mol"])
            if source_row["packed_formation_columns"]:
                assert abs(reconstructed_delta_f_g) == pytest.approx(
                    abs(printed_delta_f_g), abs=0.15
                )
            else:
                assert reconstructed_delta_f_g == pytest.approx(
                    printed_delta_f_g, abs=0.15
                )
            log10_kf = source_row["log10_Kf"]
            if log10_kf is not None:
                log_k_delta_f_g = (
                    -8.31446261815324 * temperature * math.log(10.0)
                    * float(log10_kf) / 1000.0
                )
                if source_row["packed_formation_columns"]:
                    assert abs(log_k_delta_f_g) == pytest.approx(
                        abs(reconstructed_delta_f_g), abs=0.25
                    )
                else:
                    assert log_k_delta_f_g == pytest.approx(
                        printed_delta_f_g, abs=0.25
                    )
        if verify_source:
            lower_K, upper_K = map(float, row["temperature_K"])
            served_temperatures = {float(node["T_K"]) for node in row["nodes"]}
            assert {
                temperature
                for temperature in recovered_temperatures
                if lower_K <= temperature <= upper_K
            } <= served_temperatures
            if row["table_id"] == "F-034":
                assert 1300.0 in recovered_temperatures
                assert 1300.0 in served_temperatures

    water = next(row for row in data["rows"] if row["formula"] == "H2O")
    assert water["opening_row"] is None
    assert water["cap_row"]["label"] == "LIQUID <--> REAL GAS"
    v2o5 = species_thermo("V2O5", "l", 1000.0)
    assert v2o5.H_app_kJ_mol == -1368.261
    assert v2o5.S_J_molK == 395.938
    assert v2o5.G_J_mol == pytest.approx(-1764199.0, abs=1e-9)


def test_janaf_liquid_source_order_and_duplicate_tables_refuse(monkeypatch) -> None:
    root = Path(__file__).parents[1]
    data = json.loads(
        (
            root / "src/openimcc/data/diagnostics/janaf-liquid-species.json"
        ).read_text(encoding="utf-8")
    )
    reference = next(row for row in data["rows"] if row["formula"] == "Al")
    dedicated = deepcopy(reference)
    dedicated.update(
        id="Al-liquid",
        table_id="Al-liquid",
        table_kind="l",
        source_title="Aluminum liquid table",
    )
    monkeypatch.setattr(
        janaf_crystals,
        "_load_liquid_species_rows",
        lambda: ((reference, dedicated), "test-resource-digest"),
    )

    result = species_thermo("Al", "l", 1000.0)
    assert result.source_table_id == "Al-liquid"

    second_dedicated = deepcopy(dedicated)
    second_dedicated["table_id"] = "Al-liquid-second"
    second_dedicated["id"] = "Al-liquid-second"
    monkeypatch.setattr(
        janaf_crystals,
        "_load_liquid_species_rows",
        lambda: ((reference, dedicated, second_dedicated), "test-resource-digest"),
    )
    with pytest.raises(
        ImccGasSpeciesNotFoundError,
        match="Al-liquid, Al-liquid-second",
    ):
        species_thermo("Al", "l", 1000.0)


def test_robie_hemingway_liquids_are_used_only_for_liquid_requests() -> None:
    root = Path(__file__).parents[1]
    data = json.loads(
        (
            root / "src/openimcc/data/diagnostics/rh-crystal-species.json"
        ).read_text(encoding="utf-8")
    )
    phases = [
        phase for phase in data["phases"]
        if phase.get("phase_state") == "liquid"
    ]
    assert len(phases) == 6

    for phase in phases:
        lower_K, upper_K = phase["valid_temperature_K"]
        result = species_thermo(
            phase["formula"], "l", lower_K, polymorph="ignored for liquid"
        )
        assert result.source_row_id == phase["id"]
        assert result.source_table_id == phase["source_table_id"]
        assert result.authority == "published_transcribed"
        assert result.declared_interval_K == (lower_K, upper_K)
        assert result.phase_identity["label"] == "LIQUID"
        assert result.phase_identity["printed_range"] == (
            phase["source"]["printed_phase_description"]
        )
        assert result.continuation is None

    tellurium = next(phase for phase in phases if phase["formula"] == "Te")
    with pytest.raises(ImccGasTemperatureOutsideDomainError):
        species_thermo(
            "Te", "l", tellurium["valid_temperature_K"][1] + 1,
            allow_extrapolation=True,
        )


def test_missing_parent_condensate_does_not_fall_back_to_janaf() -> None:
    pack = load_gas_datapack()
    stripped = replace(
        pack,
        oxide_df=pack.oxide_df.drop(index="SiO2(l)"),
    )

    with pytest.raises(ImccGasSpeciesNotFoundError, match=r"SiO2\(l\)"):
        species_thermo("SiO2", "l", 1800.0, datapack=stripped)


def test_rh_crystal_generator_check_matches_packaged_data() -> None:
    root = Path(__file__).parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "tools/rebuild_rh_crystal_phases.py"),
            "--check",
            "--species-only",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_elemental_reference_crystal_card_generator_check_matches_package() -> None:
    root = Path(__file__).parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "tools/generate_elemental_reference_crystal_cards.py"),
            "--check",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_vendored_elemental_reference_crystal_cards_and_ranges() -> None:
    cards = (
        ("Au", "BU-0054", "NASA-7 polynomial", (500.0, 1337.0)),
        ("Pd", "BU-3110", "NASA-7 polynomial", (298.15, 1827.0)),
        ("Pt", "BU-3120", "NASA-7 polynomial", (298.15, 2042.0)),
        ("Rh", "P672-Rh-cr", "tabulated Cp, S and H-H298", (298.15, 2237.0)),
        ("Ir", "BU-2611", "NASA-7 polynomial", (298.15, 2720.0)),
        ("Os", "BURCAT-OS-CR", "NASA-7 polynomial", (298.15, 3300.0)),
        ("Ru", "P672-Ru-cr", "tabulated Cp, S and H-H298", (298.15, 2607.0)),
    )
    for formula, card_id, functional_form, card_range in cards:
        result = species_thermo(formula, "cr", 1000.0, polymorph="reference")
        identity = result.phase_identity
        assert result.source_row_id == card_id
        assert result.source_table_id == card_id
        assert (result.T_min, result.T_max) == card_range
        assert identity is not None
        assert "elemental reference crystal" in identity["polymorph"]
        assert identity["source_card_id"] == card_id
        assert identity["functional_form"] == functional_form
        assert identity["selection_rule"] == (
            "selected by the vendored source card's elemental reference crystal"
        )


def test_elemental_reference_card_join_is_close_without_adjustment() -> None:
    # The condensate Phi fit implies Cp/H/S by differentiation; the seam gate
    # compares G directly because derivatives are fit-implied residuals.
    gibbs_gaps = {}
    for formula in ("Au", "Pd", "Pt", "Rh", "Ir", "Os", "Ru"):
        card = evaluate_crystal_species(formula, 1200.0, polymorph="reference")
        condensate = species_thermo(formula, "cr", 1200.0)
        assert card is not None
        gibbs_gaps[formula] = card.G_J_mol - condensate.G_J_mol
        assert abs(gibbs_gaps[formula]) <= 2.0

    assert abs(gibbs_gaps["Au"]) < 1.0e-6


def test_elemental_reference_card_preserves_condensate_at_1500_and_refuses_below_card() -> None:
    pd = species_thermo("Pd", "cr", 1500.0, polymorph="reference")
    assert _value_hexes(pd) == {
        "Cp_J_molK": "0x1.0c037ec008f96p+5",
        "S_J_molK": "0x1.4cb2790d14e4fp+6",
        "H_app_kJ_mol": "0x1.17a4823489c42p+5",
        "G_J_mol": "-0x1.5ece1bbe82569p+16",
    }
    with pytest.raises(ImccGasTemperatureOutsideDomainError) as refusal:
        species_thermo("Au", "cr", 499.0, polymorph="reference")
    assert refusal.value.declared_interval_K == (500.0, 1337.0)
    with pytest.raises(CrystalSpeciesTemperatureError) as refusal:
        evaluate_crystal_species("Au", 499.0, polymorph="reference")
    assert refusal.value.declared_interval_K == (500.0, 1337.0)


def test_elemental_reference_card_beats_condensate_continuation() -> None:
    direct = species_thermo("Au", "cr", 1100.0, polymorph="reference")
    continued = species_thermo(
        "Au", "cr", 1100.0, allow_extrapolation=True, polymorph="reference"
    )

    assert _value_hexes(continued) == _value_hexes(direct)
    assert continued.source_table_id == "BU-0054"
    assert not continued.extrapolated


def test_os_reference_card_covers_above_condensate_fit_and_reports_its_edge() -> None:
    result = species_thermo("Os", "cr", 3100.0, polymorph="reference")
    assert result.source_table_id == "BURCAT-OS-CR"
    assert result.T_min == 298.15
    assert result.T_max == 3300.0

    with pytest.raises(ImccGasTemperatureOutsideDomainError) as refusal:
        species_thermo("Os", "cr", 3301.0, polymorph="reference")
    assert refusal.value.declared_interval_K == (298.15, 3300.0)


def test_additional_janaf_crystals_keep_titles_transitions_and_caps() -> None:
    lead_oxide = species_thermo("PbO", "cr", 800.0)
    assert lead_oxide.source_table_id == "O-006"
    assert lead_oxide.formation_enthalpy_298_kJ_mol == pytest.approx(-218.062)
    assert lead_oxide.phase_identity is not None
    assert lead_oxide.phase_identity["source_title"].startswith(
        "Lead Oxide, Yellow"
    )
    assert lead_oxide.phase_identity["polymorph"] == "II"
    assert lead_oxide.phase_identity["temperature_segment_K"] == (762.0, 1159.0)
    selected_lead_oxide = species_thermo("PbO", "cr", 800.0, polymorph="II")
    assert selected_lead_oxide == lead_oxide
    selected_by_title = species_thermo(
        "PbO", "cr", 800.0, polymorph=lead_oxide.phase_identity["source_title"]
    )
    assert selected_by_title == lead_oxide
    assert lead_oxide.S_J_molK == 118.355
    assert lead_oxide.H_app_kJ_mol == pytest.approx(-192.449)
    assert lead_oxide.G_J_mol == pytest.approx(-287133.0)

    alpha_tio = species_thermo("TiO", "cr", 1000.0)
    beta_tio = species_thermo("TiO", "cr", 1500.0)
    assert alpha_tio.source_table_id == beta_tio.source_table_id == "O-019"
    assert alpha_tio.phase_identity is not None
    assert beta_tio.phase_identity is not None
    assert alpha_tio.phase_identity["polymorph"] == "ALPHA"
    assert beta_tio.phase_identity["polymorph"] == "BETA"
    recovered_tio = species_thermo("TiO", "cr", 1300.0)
    assert recovered_tio.S_J_molK == 114.192
    assert recovered_tio.H_app_kJ_mol == pytest.approx(-483.903)
    assert recovered_tio.G_J_mol == pytest.approx(-632352.6)

    forsterite = species_thermo("Mg2SiO4", "cr", 1500.0)
    enstatite = species_thermo("MgSiO3", "cr", 1500.0)
    assert forsterite.source_table_id == "Berman-1988"
    assert enstatite.source_table_id == "Berman-1988"
    named_forsterite = species_thermo(
        "Mg2SiO4", "cr", 1500.0, polymorph="forsterite"
    )
    named_enstatite = species_thermo(
        "MgSiO3", "cr", 1500.0, polymorph="enstatite"
    )
    assert named_forsterite.source_table_id == "Berman-1988"
    assert named_enstatite.source_table_id == "Berman-1988"
    assert named_forsterite.phase_identity["polymorph"] == "forsterite"
    assert named_enstatite.phase_identity["polymorph"] == "enstatite"
    assert enstatite.phase_identity["polymorph"] == "enstatite"
    assert enstatite.phase_identity is not None

    with pytest.raises(ImccGasTemperatureOutsideDomainError):
        species_thermo("PbO", "cr", 1160.0)

    janaf_path = Path(__file__).parents[1] / (
        "src/openimcc/data/diagnostics/janaf-crystal-species.json"
    )
    janaf = json.loads(janaf_path.read_text(encoding="utf-8"))
    ti3o5 = next(row for row in janaf["phases"] if row["formula"] == "Ti3O5")
    cap = ti3o5["crystal_transition_temperature_K"]
    assert ti3o5["transition_row_at_cap"] is None
    assert cap == ti3o5["source_temperature_K"][1]
    assert "last tabulated crystal node" in janaf["phase_restriction_rule"]
    with pytest.raises(ImccGasTemperatureOutsideDomainError):
        species_thermo("Ti3O5", "cr", cap + 1.0)
    lead_source = next(row for row in janaf["phases"] if row["formula"] == "PbO")
    assert all(node["T_K"] != 1200.0 for node in lead_source["nodes"])


def test_janaf_boron_and_tungsten_oxide_crystals_reconstruct_nodes() -> None:
    root = Path(__file__).parents[1]
    packaged = json.loads(
        (root / "src/openimcc/data/diagnostics/janaf-crystal-species.json")
        .read_text(encoding="utf-8")
    )
    expectations = {
        "B-095": ("B2O3", 723.0),
        "O-065": ("WO3", 1745.0),
    }
    element_reference_tolerances_kJ_mol = {"B2O3": 0.005, "WO3": 0.005}
    for table_id, (formula, cap_K) in expectations.items():
        source_path = root / "data-src/janaf" / f"{table_id}.yaml"
        source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        table = source["table"]
        phase = next(
            row for row in packaged["phases"]
            if row["crystal_table_id"] == table_id
        )
        assert phase["formula"] == formula
        assert phase["crystal_transition_temperature_K"] == cap_K
        assert phase["source_sha256"] == hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest()
        for row in table["values"]:
            temperature = float(row["temperature"]["value"])
            enthalpy_increment = row["enthalpy_increment"]["value"]
            entropy = row["entropy"]["value"]
            printed_delta_f_g = row["formation_gibbs_energy"]["value"]
            log_k = row["log10_formation_equilibrium_constant"]["value"]
            if (
                temperature < 298.15 or temperature > cap_K
                or enthalpy_increment is None or entropy is None
                or printed_delta_f_g is None or log_k is None
            ):
                continue
            selected = species_thermo(
                formula, "cr", temperature,
                polymorph=phase["source_title"],
            )
            assert selected.source_table_id == table_id
            absolute_g = (
                phase["formation_enthalpy_298_kJ_mol"]
                + float(enthalpy_increment)
                - temperature * float(entropy) / 1000.0
            )
            assert selected.G_J_mol / 1000.0 == pytest.approx(
                absolute_g, abs=0.002
            )
            log_k_delta_f_g = (
                -8.31446261815324 * temperature * math.log(10.0)
                * float(log_k) / 1000.0
            )
            assert log_k_delta_f_g == pytest.approx(
                float(printed_delta_f_g), abs=0.25
            )
            element_reference_sum = _janaf_element_reference_sum_kJ_mol(
                formula, temperature
            )
            assert absolute_g - element_reference_sum == pytest.approx(
                float(printed_delta_f_g),
                abs=element_reference_tolerances_kJ_mol[formula],
            )

    boron_oxide = species_thermo("B2O3", "cr", 723.0)
    assert boron_oxide.source_table_id == "B-095"
    with pytest.raises(ImccGasTemperatureOutsideDomainError):
        species_thermo("B2O3", "cr", 724.0)

    tungsten_low = species_thermo("WO3", "cr", 1000.0)
    tungsten_high = species_thermo("WO3", "cr", 1100.0)
    assert tungsten_low.source_table_id == tungsten_high.source_table_id == "O-065"
    assert tungsten_low.phase_identity["polymorph"] == "I"
    assert tungsten_high.phase_identity["polymorph"] == "II"
    assert species_thermo("WO3", "cr", 1745.0).T_max == 1745.0
    with pytest.raises(ImccGasTemperatureOutsideDomainError):
        species_thermo("WO3", "cr", 1746.0)


def test_indexed_janaf_crystal_species_reconstruct_source_rows() -> None:
    root = Path(__file__).parents[1]
    packaged = json.loads(
        (root / "src/openimcc/data/diagnostics/janaf-crystal-species.json")
        .read_text(encoding="utf-8")
    )
    expected_tables = {
        "Ti alpha": ("Ti", "Ti-002"),
        "Ti beta": ("Ti", "Ti-003"),
        "Ni": ("Ni", "Ni-002"),
        "Cr": ("Cr", "Cr-002"),
        "Ca alpha": ("Ca", "Ca-002"),
        "Ca beta": ("Ca", "Ca-003"),
        "Si": ("Si", "Si-002"),
        "Co": ("Co", "Co-002"),
        "Ta": ("Ta", "Ta-002"),
        "Cu": ("Cu", "Cu-002"),
        "K2SO4 alpha": ("K2SO4", "K-017"),
        "K2SO4 beta": ("K2SO4", "K-018"),
        "Na2SO4 delta": ("Na2SO4", "Na-019"),
        "Na2SO4 I": ("Na2SO4", "Na-020"),
        "Na2SO4 III": ("Na2SO4", "Na-021"),
        "Na2SO4 IV": ("Na2SO4", "Na-022"),
        "Na2SO4 V": ("Na2SO4", "Na-023"),
        "AlF3": ("AlF3", "Al-041"),
        "Ba": ("Ba", "Ba-002"),
        "Ba reference": ("Ba", "Ba-001"),
        "Mg": ("Mg", "Mg-002"),
        "Mg reference": ("Mg", "Mg-001"),
        "Sr alpha": ("Sr", "Sr-002"),
        "Sr beta": ("Sr", "Sr-003"),
        "Sr reference": ("Sr", "Sr-001"),
        "Al": ("Al", "Al-002"),
        "Al reference": ("Al", "Al-001"),
        "P white": ("P", "P-002"),
        "P red IV": ("P", "P-005"),
        "P red V": ("P", "P-006"),
        "P black": ("P", "P-007"),
        "P reference": ("P", "P-001"),
        "S orthorhombic": ("S", "S-002"),
        "S monoclinic": ("S", "S-003"),
        "S reference": ("S", "S-001"),
        "K": ("K", "K-002"),
        "K reference": ("K", "K-001"),
        "Zn": ("Zn", "Zn-002"),
        "Zn reference": ("Zn", "Zn-001"),
        "Pb": ("Pb", "Pb-002"),
        "Pb reference": ("Pb", "Pb-001"),
    }
    phases = {
        phase["id"]: phase
        for phase in packaged["phases"]
        if phase["id"] in expected_tables
    }
    assert set(phases) == set(expected_tables)
    reference_phase_ids = {
        formula: phase_id
        for phase_id, (formula, table_id) in expected_tables.items()
        if table_id.endswith("-001")
    }
    reference_gaps: dict[tuple[str, float], list[float]] = {}
    packed_formation_rows: dict[str, list[int]] = {}

    for phase_id, (formula, table_id) in expected_tables.items():
        phase = phases[phase_id]
        source_path = root / "data-src/janaf" / f"{table_id}.txt"
        raw = source_path.read_bytes()
        source_lines = raw.decode("utf-8").splitlines()
        assert phase["formula"] == formula
        assert phase["crystal_table_id"] == table_id
        assert phase["source_path"] == f"data-src/janaf/{table_id}.txt"
        assert phase["source_sha256"] == hashlib.sha256(raw).hexdigest()
        assert phase["source_title"] == " | ".join(
            source_lines[0].split("\t")
        )

        records = [*phase["nodes"], *phase["transition_rows"]]
        record_lines = {row["source_line_number"] for row in records}
        source_rows = {
            line_number: row
            for line_number, raw_line in enumerate(source_lines, start=1)
            if (row := _janaf_text_reference_row(raw_line)) is not None
        }
        packed_lines = sorted(
            line_number
            for line_number, row in source_rows.items()
            if row["packed_formation_columns"]
        )
        if packed_lines:
            packed_formation_rows[table_id] = packed_lines
        # Same-temperature rows with identical H/S can share one node.
        assert record_lines <= set(source_rows)
        for line_number in set(source_rows) - record_lines:
            source_row = source_rows[line_number]
            assert any(
                float(record["T_K"]) == float(source_row["T_K"])
                and float(record["enthalpy_increment_kJ_mol"])
                == float(source_row["Hinc_kJ_mol"])
                and float(record["entropy_J_mol_K"])
                == float(source_row["S_J_mol_K"])
                for record in records
            )
        for record in records:
            source_row = source_rows[record["source_line_number"]]
            assert record["T_K"] == source_row["T_K"]
            assert record["enthalpy_increment_kJ_mol"] == source_row[
                "Hinc_kJ_mol"
            ]
            assert record["entropy_J_mol_K"] == source_row["S_J_mol_K"]
            if "raw_line" in record:
                assert record["raw_line"] == source_lines[
                    record["source_line_number"] - 1
                ]

        anchor = source_rows[next(
            line_number
            for line_number, row in source_rows.items()
            if row["T_K"] == 298.15
        )]
        assert phase["formation_enthalpy_298_kJ_mol"] == anchor[
            "delta_fH_kJ_mol"
        ]
        if table_id.endswith("-001"):
            reference_gap = (
                float(phase["formation_enthalpy_298_kJ_mol"])
                + float(anchor["Hinc_kJ_mol"])
                - 298.15 * float(anchor["S_J_mol_K"]) / 1000.0
                - float(anchor["delta_fG_kJ_mol"])
            )
            assert reference_gap == pytest.approx(
                -298.15 * float(anchor["S_J_mol_K"]) / 1000.0,
                abs=0.001,
            )
        selected = evaluate_janaf_crystal(phase_id, 298.15)
        assert selected.provenance["table_id"] == table_id
        assert selected.provenance["source_sha256"] == phase["source_sha256"]
        selected_dataset = evaluate_crystal_species(
            formula, 298.15, polymorph=phase["source_title"]
        )
        assert selected_dataset is not None
        assert selected_dataset.source_table_id == table_id

        cap_K = float(phase["crystal_transition_temperature_K"])
        crystal_liquid_transitions = [
            row for row in phase["transition_rows"]
            if (
                "<-->" in row["label"]
                and "LIQUID" in row["label"].upper()
                and "IDEAL GAS" not in row["label"].upper()
            )
        ]
        if crystal_liquid_transitions:
            assert cap_K == min(
                float(row["T_K"]) for row in crystal_liquid_transitions
            )
            assert phase["transition_row_at_cap"] in crystal_liquid_transitions
        else:
            assert cap_K == float(phase["source_temperature_K"][1])
        evaluate_janaf_crystal(phase_id, cap_K)
        with pytest.raises(ValueError):
            evaluate_janaf_crystal(phase_id, cap_K + 0.1)

        for row in source_rows.values():
            temperature = float(row["T_K"])
            printed_delta_f_g = row["delta_fG_kJ_mol"]
            if temperature < 298.15 or temperature > cap_K:
                continue
            if printed_delta_f_g is None:
                continue
            absolute_g = (
                float(phase["formation_enthalpy_298_kJ_mol"])
                + float(row["Hinc_kJ_mol"])
                - temperature * float(row["S_J_mol_K"]) / 1000.0
            )
            reference_phase_id = reference_phase_ids.get(formula)
            if reference_phase_id is not None:
                reference_g = evaluate_janaf_crystal(
                    reference_phase_id,
                    temperature,
                    apply_phase_cap=False,
                ).G_J_mol / 1000.0
            else:
                reference_g = _janaf_element_reference_sum_kJ_mol(
                    formula, temperature
                )
            if reference_g is None:
                continue
            reconstructed_delta_f_g = absolute_g - reference_g
            if row["packed_formation_columns"]:
                # NIST packs the formation columns into one text cell, where
                # the signs are ambiguous. Compare the printed magnitudes.
                assert abs(reconstructed_delta_f_g) == pytest.approx(
                    abs(float(printed_delta_f_g)), abs=0.15
                )
            else:
                # Independent reference tables have different temperature
                # grids, so interpolation can differ by up to 0.11 kJ/mol.
                assert reconstructed_delta_f_g == pytest.approx(
                    float(printed_delta_f_g), abs=0.15
                )
            log10_kf = row["log10_Kf"]
            if log10_kf is not None:
                log_k_delta_f_g = (
                    -8.31446261815324 * temperature * math.log(10.0)
                    * float(log10_kf) / 1000.0
                )
                if row["packed_formation_columns"]:
                    assert abs(log_k_delta_f_g) == pytest.approx(
                        abs(reconstructed_delta_f_g), abs=0.25
                    )
                else:
                    assert log_k_delta_f_g == pytest.approx(
                        float(printed_delta_f_g), abs=0.25
                    )
            gap = (
                reference_g
                if row["packed_formation_columns"]
                else absolute_g - float(printed_delta_f_g)
            )
            reference_gaps.setdefault((formula, temperature), []).append(gap)

    for gaps in reference_gaps.values():
        if len(gaps) > 1:
            assert max(gaps) - min(gaps) <= 0.02

    assert packed_formation_rows == {
        "Al-041": [33],
        "Ca-002": [16],
        "Ca-003": [16, 21],
        "Co-002": [30],
        "Cr-002": [32],
        "Cu-002": [22],
        "K-017": [14],
        "K-018": [14, 20],
        "Na-020": [11, 18],
        "Na-022": [10, 12],
        "Na-023": [10],
        "Ni-002": [28],
        "Si-002": [25],
        "Ta-002": [41],
        "Ti-002": [20],
        "Ti-003": [20, 29],
        "Ba-002": [25],
        "Mg-002": [18],
        "Sr-002": [16],
        "Sr-003": [15, 18],
        "Al-002": [18],
        "P-002": [12],
        "S-002": [11],
        "K-002": [10],
        "Zn-002": [15],
        "Pb-002": [15],
    }
    for polymorph, expected_table in (("DELTA", "Na-019"), ("III", "Na-021")):
        selected_form = evaluate_crystal_species(
            "Na2SO4", 400.0, polymorph=polymorph
        )
        assert selected_form is not None
        assert selected_form.source_table_id == expected_table


@pytest.mark.parametrize(
    ("formula", "polymorph", "table_id", "title_fragment"),
    (
        ("Ba", "reference", "Ba-001", "Barium (Ba)"),
        ("Mg", "reference", "Mg-001", "Magnesium (Mg)"),
        ("Sr", "alpha", "Sr-002", "Strontium, Alpha (Sr)"),
        ("Sr", "beta", "Sr-003", "Strontium, Beta (Sr)"),
        ("Sr", "reference", "Sr-001", "Strontium (Sr)"),
        ("Al", "reference", "Al-001", "Aluminum (Al)"),
        ("P", "white", "P-002", "Phosphorus, White (P)"),
        ("P", "red IV", "P-005", "Phosphorus, Red, IV (P)"),
        ("P", "red V", "P-006", "Phosphorus, Red, V (P)"),
        ("P", "black", "P-007", "Phosphorus, Black (P)"),
        ("P", "reference", "P-001", "Phosphorus (P)"),
        ("S", "orthorhombic", "S-002", "Sulfur, Orthorhombic (S)"),
        ("S", "rhombic", "S-002", "Sulfur, Orthorhombic (S)"),
        ("S", "monoclinic", "S-003", "Sulfur, Monoclinic (S)"),
        ("S", "reference", "S-001", "Sulfur (S)"),
        ("K", "reference", "K-001", "Potassium (K)"),
        ("Zn", "reference", "Zn-001", "Zinc (Zn)"),
        ("Pb", "reference", "Pb-001", "Lead (Pb)"),
    ),
)
def test_janaf_element_crystal_short_names_select_the_printed_table(
    formula: str,
    polymorph: str,
    table_id: str,
    title_fragment: str,
) -> None:
    selected = evaluate_crystal_species(formula, 298.15, polymorph=polymorph)
    assert selected is not None
    assert selected.source_table_id == table_id
    assert title_fragment in selected.phase_identity["source_title"]


@pytest.mark.parametrize(
    ("formula", "temperature_K", "cap_K", "declared_interval", "windows"),
    (
        (
            "S", 400.0, 388.36, (298.15, 800.0),
            (
                "Sulfur, Orthorhombic (S) | S1(cr) [298.15-388.36 K]",
                "Sulfur, Monoclinic (S) | S1(cr) [298.15-388.36 K]",
                "Sulfur (S) | S1(ref) [298.15-388.36 K]",
            ),
        ),
        (
            "Sr", 1100.0, 1050.0, (298.15, 1500.0),
            (
                "Strontium, Alpha (Sr) | Sr1(cr) [298.15-1050 K]",
                "Strontium, Beta (Sr) | Sr1(cr) [298.15-1050 K]",
                "Strontium (Sr) | Sr1(ref) [298.15-1050 K]",
            ),
        ),
    ),
)
def test_unnamed_janaf_element_selection_stops_at_reference_melt(
    formula: str,
    temperature_K: float,
    cap_K: float,
    declared_interval: tuple[float, float],
    windows: tuple[str, ...],
) -> None:
    with pytest.raises(ImccGasTemperatureOutsideDomainError) as refusal:
        species_thermo(formula, "cr", temperature_K)

    message = str(refusal.value)
    assert refusal.value.declared_interval_K == declared_interval
    assert f"above its source solid limit of {cap_K:g} K" in str(refusal.value)
    assert "declared interval is" not in message
    assert "available windows:" in message
    for window in windows:
        assert window in message


def test_named_orthorhombic_sulfur_keeps_its_printed_metastable_rows() -> None:
    selected = species_thermo("S", "cr", 600.0, polymorph="orthorhombic")

    assert selected.source_table_id == "S-002"
    assert selected.phase_identity["polymorph"] == "ORTHORHOMBIC"


def test_unnamed_red_phosphorus_is_not_capped_at_white_phosphorus_melt() -> None:
    selected = species_thermo("P", "cr", 600.0)

    assert selected.source_table_id in {"P-005", "P-006", "P-007"}


@pytest.mark.parametrize(
    ("formula", "temperature_K", "windows"),
    (
        ("KNO3", 450.0, ("[298.15-400 K]", "[500-600 K]")),
        ("SnS", 850.0, ("[298.15-800 K]", "[900-1100 K]")),
    ),
)
def test_unnamed_polymorph_gaps_report_each_available_window(
    formula: str, temperature_K: float, windows: tuple[str, str],
) -> None:
    with pytest.raises(ImccGasTemperatureOutsideDomainError) as refusal:
        species_thermo(formula, "cr", temperature_K)

    message = str(refusal.value)
    assert "available windows:" in message
    assert "declared G(T) interval" not in message
    for window in windows:
        assert window in message


@pytest.mark.parametrize(
    ("formula", "temperature", "polymorph", "table_id", "answer_hexes"),
    (
        (
            "Ti", 1500.0, "BETA", "Ti-003",
            {
                "Cp_J_molK": "0x1.f7ffffffffffcp+4",
                "S_J_molK": "0x1.45374bc6a7efap+6",
                "H_app_kJ_mol": "0x1.4133333333333p+5",
                "G_J_mol": "-0x1.3f8e000000000p+16",
            },
        ),
        (
            "Ca", 900.0, "BETA", "Ca-003",
            {
                "Cp_J_molK": "0x1.1a00000000003p+5",
                "S_J_molK": "0x1.2def9db22d0e5p+6",
                "H_app_kJ_mol": "0x1.33fbe76c8b43ap+4",
                "G_J_mol": "-0x1.7c5d333333332p+15",
            },
        ),
        (
            "K2SO4", 900.0, "BETA", "K-018",
            {
                "Cp_J_molK": "0x1.737d05f417d01p+7",
                "S_J_molK": "0x1.6f87ae147ae14p+8",
                "H_app_kJ_mol": "-0x1.4b876c8b43958p+10",
                "G_J_mol": "-0x1.9483d00000000p+20",
            },
        ),
        (
            "Na2SO4", 400.0, "V", "Na-023",
            {
                "Cp_J_molK": "0x1.121eb851eb851p+7",
                "S_J_molK": "0x1.7b676c8b43958p+7",
                "H_app_kJ_mol": "-0x1.5777df3b645a2p+10",
                "G_J_mol": "-0x1.61f19cccccccdp+20",
            },
        ),
        (
            "Na2SO4", 458.0, "IV", "Na-022",
            {
                "Cp_J_molK": "0x1.2a8469ee5846bp+7",
                "S_J_molK": "0x1.a4ec8b4395810p+7",
                "H_app_kJ_mol": "-0x1.553d604189375p+10",
                "G_J_mol": "-0x1.64c66989374bcp+20",
            },
        ),
    ),
)
def test_janaf_dedicated_phase_names_select_tables_and_preserve_answers(
    formula: str,
    temperature: float,
    polymorph: str,
    table_id: str,
    answer_hexes: dict[str, str],
) -> None:
    unnamed = species_thermo(formula, "cr", temperature)
    named = species_thermo(
        formula, "cr", temperature, polymorph=polymorph
    )

    assert unnamed.source_table_id == table_id
    assert _value_hexes(unnamed) == answer_hexes
    assert named.source_table_id == table_id
    assert named.phase_identity is not None
    assert named.phase_identity["polymorph"] == polymorph


@pytest.mark.parametrize(
    ("formula", "temperature", "polymorph", "table_id"),
    (
        ("Ti", 1939.0, "BETA", "Ti-003"),
        ("Ca", 1115.0, "BETA", "Ca-003"),
        ("K2SO4", 1342.0, "BETA", "K-018"),
    ),
)
def test_dedicated_janaf_beta_selector_reaches_the_solid_limit(
    formula: str, temperature: float, polymorph: str, table_id: str,
) -> None:
    selected = species_thermo(
        formula, "cr", temperature, polymorph=polymorph
    )

    assert selected.source_table_id == table_id
    assert selected.phase_identity is not None
    assert selected.phase_identity["polymorph"] == "BETA"


@pytest.mark.parametrize(
    ("formula", "temperature", "polymorph", "table_id", "phase"),
    (
        ("Fe", 1300.0, "GAMMA", "Fe-004", "GAMMA"),
        ("Mn", 1200.0, "BETA", "Mn-004", "BETA"),
        ("WO3", 1100.0, "II", "O-065", "II"),
        ("Co", 1000.0, "BETA", "Co-002", "BETA"),
        ("AlF3", 1000.0, "BETA", "Al-041", "BETA"),
    ),
)
def test_multiphase_janaf_reference_selectors_keep_transition_segments(
    formula: str,
    temperature: float,
    polymorph: str,
    table_id: str,
    phase: str,
) -> None:
    unnamed = species_thermo(formula, "cr", temperature)
    named = species_thermo(
        formula, "cr", temperature, polymorph=polymorph
    )

    assert unnamed.source_table_id == named.source_table_id == table_id
    assert _value_hexes(named) == _value_hexes(unnamed)
    assert named.phase_identity is not None
    assert named.phase_identity["polymorph"] == phase


def test_janaf_internal_transition_uses_the_previous_phase_endpoint() -> None:
    source_path = Path(__file__).parents[1] / (
        "src/openimcc/data/diagnostics/janaf-crystal-species.json"
    )
    phase = next(
        row for row in json.loads(source_path.read_text(encoding="utf-8"))["phases"]
        if row["crystal_table_id"] == "O-065"
    )
    endpoint = next(
        row for row in phase["transition_rows"]
        if row["T_K"] == 1050.0 and row["label"] == "I <--> II"
    )
    transition = next(
        row for row in phase["transition_rows"]
        if row["T_K"] == 1050.0 and row["label"] == "TRANSITION"
    )
    lower = next(row for row in phase["nodes"] if row["T_K"] == 1000.0)
    upper = next(row for row in phase["nodes"] if row["T_K"] == 1100.0)

    def interpolate(left, right, temperature, key):
        fraction = (temperature - left["T_K"]) / (right["T_K"] - left["T_K"])
        return left[key] + (right[key] - left[key]) * fraction

    below_T = 1025.0
    below = evaluate_crystal_species("WO3", below_T)
    assert below is not None
    expected_below_hinc = interpolate(
        lower, endpoint, below_T, "enthalpy_increment_kJ_mol"
    )
    expected_below_entropy = interpolate(
        lower, endpoint, below_T, "entropy_J_mol_K"
    )
    assert below.H_app_kJ_mol == pytest.approx(
        phase["formation_enthalpy_298_kJ_mol"] + expected_below_hinc
    )
    assert below.S_J_molK == pytest.approx(expected_below_entropy)
    assert below.G_J_mol / 1000.0 == pytest.approx(
        below.H_app_kJ_mol - below_T * below.S_J_molK / 1000.0
    )

    old_transition_endpoint_hinc = interpolate(
        lower, transition, below_T, "enthalpy_increment_kJ_mol"
    )
    assert below.H_app_kJ_mol - (
        phase["formation_enthalpy_298_kJ_mol"] + old_transition_endpoint_hinc
    ) == pytest.approx(-0.7415)

    at_transition = evaluate_crystal_species("WO3", 1050.0)
    assert at_transition is not None
    assert at_transition.H_app_kJ_mol == pytest.approx(
        phase["formation_enthalpy_298_kJ_mol"]
        + transition["enthalpy_increment_kJ_mol"]
    )
    after_T = 1075.0
    after = evaluate_crystal_species("WO3", after_T)
    assert after is not None
    assert after.H_app_kJ_mol == pytest.approx(
        phase["formation_enthalpy_298_kJ_mol"]
        + interpolate(transition, upper, after_T, "enthalpy_increment_kJ_mol")
    )
    assert after.S_J_molK == pytest.approx(
        interpolate(transition, upper, after_T, "entropy_J_mol_K")
    )


def test_janaf_iron_reference_state_tracks_alpha_gamma_transition() -> None:
    alpha = species_thermo("Fe", "cr", 1000.0)
    gamma = species_thermo("Fe", "cr", 1300.0)

    assert alpha.source_table_id == "Fe-004"
    assert gamma.source_table_id == "Fe-004"
    assert alpha.formation_enthalpy_298_kJ_mol == 0.0
    assert gamma.formation_enthalpy_298_kJ_mol == 0.0
    assert alpha.phase_identity["polymorph"] == "ALPHA"
    assert gamma.phase_identity["polymorph"] == "GAMMA"
    assert alpha.phase_identity["temperature_segment_K"] == (298.15, 1184.0)
    assert gamma.phase_identity["temperature_segment_K"] == (1184.0, 1665.0)


def test_elemental_reference_polymorph_selectors() -> None:
    pack = load_gas_datapack()
    selections = (
        ("Au", 1300.0, "ReFeReNcE", "BU-0054", "gold elemental reference"),
        ("Pt", 1500.0, "ref", "BU-3120", "platinum elemental reference"),
        ("Fe", 1300.0, "Reference State", "Fe-004", "GAMMA"),
    )
    for formula, temperature, selector, table_id, expected_phase in selections:
        selected = species_thermo(
            formula, "cr", temperature, pack, polymorph=selector
        )
        default = species_thermo(formula, "cr", temperature, pack)
        assert selected.source_table_id == table_id
        assert _value_hexes(selected) == _value_hexes(default)
        assert selected.phase_identity is not None
        assert expected_phase in selected.phase_identity["polymorph"]
        assert "reference-state" in selected.phase_identity["selection_rule"]


@pytest.mark.parametrize(
    ("formula", "table_id"),
    (
        ("W", "W-004"),
        ("B", "B-004"),
        ("V", "V-004"),
        ("Mo", "Mo-004"),
        ("Nb", "Nb-004"),
    ),
)
def test_janaf_elemental_reference_selector_uses_zero_formation_enthalpy(
    formula: str, table_id: str
) -> None:
    selected = species_thermo(formula, "cr", 1000.0, polymorph="reference")
    default = species_thermo(formula, "cr", 1000.0)

    assert selected.source_table_id == table_id
    assert _value_hexes(selected) == _value_hexes(default)
    assert selected.phase_identity is not None
    assert "reference-state" in selected.phase_identity["selection_rule"]


def test_ambiguous_reference_selector_is_typed() -> None:
    with pytest.raises(
        ImccGasCrystalPolymorphError,
        match="reference selection is ambiguous",
    ):
        species_thermo("Al2SiO5", "cr", 1300.0, polymorph="reference")


@pytest.mark.parametrize(
    ("formula", "polymorph", "temperature", "table_id", "phase_fragment"),
    (
        ("MgO", "periclase", 1500.0, "Mg-008", "Mg-008"),
        ("TiO2", "rutile", 1500.0, "O-043", "O-043"),
        ("Ti3O5", "beta", 1000.0, "O-081", "Beta"),
        ("FeS", "troilite", 500.0, "Fe-023", "Troilite"),
    ),
)
def test_shipped_crystal_polymorph_aliases(
    formula: str,
    polymorph: str,
    temperature: float,
    table_id: str,
    phase_fragment: str,
) -> None:
    selected = species_thermo(
        formula, "cr", temperature, polymorph=polymorph
    )

    assert selected.source_table_id == table_id
    assert selected.phase_identity is not None
    assert phase_fragment in str(selected.phase_identity)
    assert "alias" in selected.phase_identity["selection_rule"]
    if formula == "FeS":
        assert selected.phase_identity["temperature_segment_K"] == (411.0, 598.0)
        assert selected.phase_identity["polymorph"] == "BETA"
        assert selected.phase_identity["selection_rule"] == (
            "selected as an alias of the source table title; transition rows "
            "identify the active segment at T"
        )


def test_distinct_crystal_phases_are_not_aliased() -> None:
    with pytest.raises(ImccGasCrystalPolymorphError):
        species_thermo("Ta2O5", "cr", 500.0, polymorph="beta")
    with pytest.raises(ImccGasCrystalPolymorphError):
        species_thermo("PbO", "cr", 500.0, polymorph="red")


def test_robie_hemingway_al2sio5_selection_and_printed_gibbs() -> None:
    expected = {300.0: "Kyanite", 1000.0: "Andalusite", 1800.0: "Sillimanite"}
    for temperature, polymorph in expected.items():
        result = species_thermo("Al2SiO5", "cr", temperature)
        assert result.phase_identity["polymorph"] == polymorph

    selected = species_thermo(
        "Al2SiO5", "cr", 1300.0, polymorph="kyanite"
    )
    assert selected.source_table_id == "USGS-B2131-p304"
    assert selected.phase_identity["polymorph"] == "Kyanite"

    table_path = Path(__file__).parents[1] / (
        "data-src/robie-hemingway-1995-usgs-b2131/crystal-tables.json"
    )
    source = json.loads(table_path.read_text(encoding="utf-8"))
    kyanite = next(phase for phase in source["phases"] if phase["id"] == "kyanite")
    printed_gibbs = next(
        row["formation_gibbs_kJ_mol"]
        for row in kyanite["rows"]
        if row["T_K"] == 1300.0
    )
    element_gibbs = _element_reference_sum_kJ_mol("Al2SiO5", 1300.0)
    assert element_gibbs is not None
    assert selected.G_J_mol / 1000.0 - element_gibbs == pytest.approx(
        printed_gibbs, abs=0.2
    )

    with pytest.raises(ImccGasTemperatureOutsideDomainError):
        species_thermo("AlO(OH)", "cr", 601.0)


def test_robie_hemingway_added_crystal_tables_reconstruct_printed_nodes() -> None:
    table_path = Path(__file__).parents[1] / (
        "data-src/robie-hemingway-1995-usgs-b2131/crystal-tables.json"
    )
    source = json.loads(table_path.read_text(encoding="utf-8"))
    phase_ids = {
        "tephroite", "alpha-spodumene", "beta-spodumene", "microcline",
        "sanidine", "albite", "analbite",
    }
    expected_served_upper_K = {
        "tephroite": 1500.0,
        "alpha-spodumene": 1200.0,
        "beta-spodumene": 1696.0,
        "microcline": 1400.0,
        "sanidine": 1400.0,
        "albite": 1400.0,
        "analbite": 1391.0,
    }
    element_reference_tolerances_kJ_mol = {
        "tephroite": 0.50,
        "alpha-spodumene": 0.10,
        "beta-spodumene": 0.10,
        "microcline": 0.15,
        "sanidine": 0.15,
        "albite": 0.20,
        "analbite": 0.15,
    }
    phases = [phase for phase in source["phases"] if phase["id"] in phase_ids]
    assert {phase["id"] for phase in phases} == phase_ids
    element_reference_gaps: dict[tuple[str, float], list[float]] = {}

    for phase in phases:
        first, last = phase["source_temperature_K"]
        valid_last = float(
            phase.get("valid_temperature_K", phase["source_temperature_K"])[1]
        )
        assert valid_last == expected_served_upper_K[phase["id"]]
        assert phase["rows"][0]["T_K"] == first
        assert phase["rows"][-1]["T_K"] == last
        for row in phase["rows"]:
            temperature = float(row["T_K"])
            absolute_g = (
                float(phase["rows"][0]["formation_enthalpy_kJ_mol"])
                + temperature * float(row["H_minus_H298_over_T_J_mol_K"]) / 1000.0
                - temperature * float(row["S_J_mol_K"]) / 1000.0
            )
            if temperature <= valid_last:
                selected = species_thermo(
                    phase["formula"], "cr", temperature,
                    polymorph=phase["name_as_published"],
                )
                assert selected.source_table_id == (
                    f"USGS-B2131-p{phase['printed_page']}"
                )
                assert selected.phase_identity["polymorph"] == (
                    phase["name_as_published"]
                )
                assert selected.G_J_mol / 1000.0 == pytest.approx(
                    absolute_g, abs=0.002
                )
            else:
                with pytest.raises(ImccGasTemperatureOutsideDomainError):
                    species_thermo(
                        phase["formula"], "cr", temperature,
                        polymorph=phase["name_as_published"],
                    )
            element_reference_sum = _janaf_element_reference_sum_kJ_mol(
                phase["formula"], temperature
            )
            reconstructed_delta_f_g = absolute_g - element_reference_sum
            assert reconstructed_delta_f_g == pytest.approx(
                float(row["formation_gibbs_kJ_mol"]),
                abs=element_reference_tolerances_kJ_mol[phase["id"]],
            )
            gap = absolute_g - float(row["formation_gibbs_kJ_mol"])
            element_reference_gaps.setdefault(
                (phase["formula"], temperature), []
            ).append(gap)

    # Polymorph tables for the same stoichiometry must share the same
    # elemental-reference contribution at their common temperature nodes.
    for gaps in element_reference_gaps.values():
        if len(gaps) > 1:
            assert max(gaps) - min(gaps) <= 0.15

    beta = species_thermo(
        "LiAlSi2O6", "cr", 1696.0, polymorph="β-SPODUMENE"
    )
    assert beta.T_max == 1696.0
    assert "Melting point is 1696 K." in beta.phase_identity["note"]
    with pytest.raises(ImccGasTemperatureOutsideDomainError):
        species_thermo("LiAlSi2O6", "cr", 1697.0, polymorph="β-SPODUMENE")
    with pytest.raises(ImccGasTemperatureOutsideDomainError) as refusal:
        species_thermo("LiAlSi2O6", "cr", 1697.0)
    assert refusal.value.declared_interval_K == (298.15, 1696.0)
    assert "(298.15, 1696) K" in str(refusal.value)

    analbite = species_thermo(
        "NaAlSi3O8", "cr", 1391.0, polymorph="ANALBITE"
    )
    assert analbite.T_max == 1391.0
    for temperature in (1392.0, 1395.0, 1400.0):
        with pytest.raises(ImccGasTemperatureOutsideDomainError) as refusal:
            species_thermo(
                "NaAlSi3O8", "cr", temperature, polymorph="ANALBITE"
            )
        assert refusal.value.declared_interval_K == (298.15, 1391.0)

    for temperature in (1395.0, 1400.0):
        albite = species_thermo("NaAlSi3O8", "cr", temperature)
        assert albite.source_table_id == "USGS-B2131-p373"
        assert albite.phase_identity["polymorph"] == "ALBITE"
        assert albite.T_max == 1400.0


def test_robie_hemingway_target_tables_reconstruct_printed_nodes() -> None:
    table_path = Path(__file__).parents[1] / (
        "data-src/robie-hemingway-1995-usgs-b2131/crystal-tables.json"
    )
    source = json.loads(table_path.read_text(encoding="utf-8"))
    phases = {
        phase["id"]: phase
        for phase in source["phases"]
        if phase["id"] in {
            "witherite", "aragonite", "calcite", "pyrite", "marcasite",
            "chalcocite", "chalcopyrite", "bornite", "grossular",
            "ca-al-pyroxene",
        }
    }
    assert set(phases) == {
        "witherite", "aragonite", "calcite", "pyrite", "marcasite",
        "chalcocite", "chalcopyrite", "bornite", "grossular",
        "ca-al-pyroxene",
    }
    packaged_path = Path(__file__).parents[1] / (
        "src/openimcc/data/diagnostics/rh-crystal-species.json"
    )
    packaged = {
        phase["id"]: phase
        for phase in json.loads(packaged_path.read_text(encoding="utf-8"))["phases"]
    }
    gas_constant_J_mol_K = 8.31446261815324
    element_reference_gaps: dict[tuple[str, float], list[float]] = {}

    for phase_id, phase in phases.items():
        runtime_phase = packaged[phase_id]
        assert runtime_phase["source"]["printed_page"] == phase["printed_page"]
        assert runtime_phase["source"]["printed_phase_description"] == (
            phase["printed_phase_description"]
        )
        assert runtime_phase["source_temperature_K"] == phase["source_temperature_K"]
        assert len(phase["rows"]) == len(runtime_phase["source_row_locators"])
        h298 = float(phase["rows"][0]["formation_enthalpy_kJ_mol"])

        for row in phase["rows"]:
            temperature = float(row["T_K"])
            label = (
                str(int(temperature))
                if temperature.is_integer()
                else str(temperature)
            )
            locator = row["locator"]
            assert locator["page"] == phase["printed_page"]
            assert locator["row"] == f"{label} K"
            assert runtime_phase["source_row_locators"][label] == locator

            selected = species_thermo(
                phase["formula"], "cr", temperature,
                polymorph=phase["name_as_published"],
            )
            assert selected.source_table_id == (
                f"USGS-B2131-p{phase['printed_page']}"
            )
            assert selected.phase_identity["polymorph"] == phase["name_as_published"]
            absolute_g = (
                h298
                + temperature
                * float(row["H_minus_H298_over_T_J_mol_K"])
                / 1000.0
                - temperature * float(row["S_J_mol_K"]) / 1000.0
            )
            assert selected.G_J_mol / 1000.0 == pytest.approx(absolute_g, abs=0.002)

            printed_g = float(row["formation_gibbs_kJ_mol"])
            reconstructed_g = (
                -gas_constant_J_mol_K
                * temperature
                * math.log(10.0)
                * float(row["log10_Kf"])
                / 1000.0
            )
            assert reconstructed_g == pytest.approx(printed_g, abs=0.25)
            element_reference_gaps.setdefault(
                (phase["formula"], temperature), []
            ).append(absolute_g - printed_g)

    # The two CaCO3 tables and two FeS2 tables share one elemental reference
    # contribution at their common nodes, within the printed table rounding.
    for gaps in element_reference_gaps.values():
        if len(gaps) > 1:
            assert max(gaps) - min(gaps) <= 0.25

    for formula, phase_id, source_id, limit in (
        ("Cu2S", "CHALCOCITE", "chalcocite", 1403.0),
        ("FeS2", "PYRITE", "pyrite", 1015.0),
    ):
        assert packaged[source_id]["source_solid_limit"]["temperature_K"] == limit
        last_row_temperature = float(phases[source_id]["rows"][-1]["T_K"])
        assert last_row_temperature <= limit
        with pytest.raises(
            ImccGasTemperatureOutsideDomainError,
        ):
            species_thermo(
                formula, "cr", limit + 1.0,
                polymorph=phase_id, allow_extrapolation=True,
            )


def test_rh_printed_solid_limit_caps_extended_candidate(monkeypatch) -> None:
    candidate = next(
        candidate
        for _, candidates in crystal_species._candidates_for_formula(
            "Cu2S", 1000.0
        )
        for candidate in candidates
        if candidate.phase["id"] == "chalcocite"
    )
    extended = replace(
        candidate,
        interval_K=(candidate.interval_K[0], 1410.0),
    )
    monkeypatch.setattr(
        crystal_species,
        "_candidates_for_formula",
        lambda _formula, _temperature_K: [
            ("Robie-Hemingway 1995 tables", [extended])
        ],
    )

    with pytest.raises(
        ImccGasTemperatureOutsideDomainError,
        match="above its source solid limit of 1403 K",
    ):
        species_thermo("Cu2S", "cr", 1404.0, polymorph="CHALCOCITE")


def test_robie_hemingway_perovskite_and_element_tables_reconstruct_nodes() -> None:
    root = Path(__file__).parents[1]
    source_path = root / (
        "data-src/robie-hemingway-1995-usgs-b2131/crystal-tables.json"
    )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    wanted_ids = {
        "perovskite-table", "silver-fcc", "silver-liquid", "thorium-alpha",
        "thorium-beta", "uranium-alpha", "uranium-beta", "uranium-gamma",
        "uranium-liquid", "arsenic-rhombohedral", "cerium-alpha",
        "cerium-beta", "cerium-liquid", "antimony-rhombohedral",
        "antimony-liquid", "tin-crystals", "tin-liquid",
        "tellurium-crystals", "tellurium-liquid",
    }
    phases = {
        phase["id"]: phase
        for phase in source["phases"]
        if phase["id"] in wanted_ids
    }
    assert set(phases) == wanted_ids
    expected_row_spans = {
        "perovskite-table": (17, 10025, 10041),
        "silver-fcc": (12, 2000, 2011),
        "silver-liquid": (7, 2012, 2018),
        "thorium-alpha": (16, 4039, 4054),
        "thorium-beta": (3, 4055, 4057),
        "uranium-alpha": (9, 4135, 4143),
        "uranium-beta": (3, 4144, 4146),
        "uranium-gamma": (6, 4147, 4152),
        "uranium-liquid": (5, 4153, 4157),
        "arsenic-rhombohedral": (8, 2092, 2099),
        "cerium-alpha": (9, 2590, 2598),
        "cerium-beta": (3, 2599, 2601),
        "cerium-liquid": (9, 2602, 2610),
        "antimony-rhombohedral": (9, 3756, 3764),
        "antimony-liquid": (10, 3765, 3774),
        "tin-crystals": (5, 3889, 3893),
        "tin-liquid": (14, 3894, 3907),
        "tellurium-crystals": (7, 3993, 3999),
        "tellurium-liquid": (7, 4000, 4006),
    }
    for phase_id, (count, first_line, last_line) in expected_row_spans.items():
        locators = [
            int(row["locator"]["pdf_text_line"])
            for row in phases[phase_id]["rows"]
        ]
        assert len(locators) == count
        assert locators == list(range(first_line, last_line + 1))

    packaged_path = root / "src/openimcc/data/diagnostics/rh-crystal-species.json"
    packaged = {
        phase["id"]: phase
        for phase in json.loads(packaged_path.read_text(encoding="utf-8"))["phases"]
    }
    gas_constant_J_mol_K = 8.31446261815324
    elemental_gaps: dict[tuple[str, float], list[float]] = {}
    perovskite_reference_residuals = []

    for phase_id, phase in phases.items():
        runtime_phase = packaged[phase_id]
        assert runtime_phase["source"]["printed_page"] == phase["printed_page"]
        assert runtime_phase["source"]["printed_phase_description"] == (
            phase["printed_phase_description"]
        )
        if phase.get("state") == "liquid":
            assert runtime_phase["phase_state"] == "liquid"
        assert len(phase["rows"]) == len(runtime_phase["source_row_locators"])
        if "formation_enthalpy_298_kJ_mol" in phase:
            h298 = float(phase["formation_enthalpy_298_kJ_mol"])
        else:
            h298 = float(next(
                row["formation_enthalpy_kJ_mol"]
                for row in phase["rows"]
                if float(row["T_K"]) == 298.15
            ))
        valid_lower, valid_upper = map(float, phase["valid_temperature_K"])

        for row in phase["rows"]:
            temperature = float(row["T_K"])
            label = (
                str(int(temperature))
                if temperature.is_integer()
                else str(temperature)
            )
            locator = row["locator"]
            assert locator["page"] == phase["printed_page"]
            assert locator["row"] == f"{label} K"
            assert runtime_phase["source_row_locators"][label] == locator

            absolute_g = (
                h298
                + temperature
                * float(row["H_minus_H298_over_T_J_mol_K"])
                / 1000.0
                - temperature * float(row["S_J_mol_K"]) / 1000.0
            )
            printed_g = float(row["formation_gibbs_kJ_mol"])
            reconstructed_g = (
                -gas_constant_J_mol_K
                * temperature
                * math.log(10.0)
                * float(row["log10_Kf"])
                / 1000.0
            )
            assert reconstructed_g == pytest.approx(printed_g, abs=0.25)
            formula_elements = _formula_element_counts(phase["formula"])
            is_element_reference = (
                formula_elements is not None
                and len(formula_elements) == 1
                and next(iter(formula_elements.values())) == 1
            )
            if temperature == 298.15 and is_element_reference:
                expected_element_gap = (
                    -temperature * float(row["S_J_mol_K"]) / 1000.0
                )
                assert absolute_g - printed_g == pytest.approx(
                    expected_element_gap, abs=1e-9
                )
            elemental_gaps.setdefault((phase["formula"], temperature), []).append(
                absolute_g - printed_g
            )

            if phase.get("state") == "liquid":
                continue
            if (
                phase.get("upper_temperature_inclusive") is False
                and temperature == valid_upper
            ):
                continue
            assert valid_lower <= temperature <= valid_upper
            selected = species_thermo(
                phase["formula"], "cr", temperature,
                polymorph=phase["name_as_published"],
            )
            assert selected.G_J_mol / 1000.0 == pytest.approx(
                absolute_g, abs=0.002
            )
            if phase_id == "perovskite-table":
                element_g = _element_reference_sum_kJ_mol(
                    phase["formula"], temperature
                )
                if element_g is not None:
                    perovskite_reference_residuals.append(
                        abs(absolute_g - element_g - printed_g)
                    )

    # Transition rows share an element-reference contribution on both sides.
    for formula, temperature in (
        ("Ag", 1234.9), ("Th", 1650.0), ("U", 942.0),
        ("U", 1049.0), ("U", 1408.0), ("Ce", 999.0),
        ("Ce", 1071.0), ("Sb", 904.0), ("Sn", 505.1),
        ("Te", 723.0),
    ):
        gaps = elemental_gaps[(formula, temperature)]
        assert len(gaps) == 2
        assert max(gaps) - min(gaps) <= 0.02

    for formula, temperature in (
        ("Sb", 1000.0), ("Sn", 600.0), ("Te", 800.0), ("Ce", 1100.0),
    ):
        with pytest.raises(ImccGasTemperatureOutsideDomainError):
            species_thermo(formula, "cr", temperature)

    # At common high-temperature nodes, the element-reference mismatch stays
    # within the previously established 2.5 kJ/mol R&H cross-check tolerance.
    assert max(perovskite_reference_residuals) == pytest.approx(2.3387, abs=1e-3)

    for phase_id, limit, page, pdf_text_line in (
        ("perovskite-table", 2188.0, 232, 10043),
        ("silver-fcc", 1234.9, 67, 2025),
        ("thorium-beta", 2023.0, 112, 4063),
        ("uranium-gamma", 1408.0, 114, 4166),
    ):
        source_limit = packaged[phase_id]["source_solid_limit"]
        assert source_limit["temperature_K"] == limit
        assert source_limit["locator"]["page"] == page
        assert source_limit["locator"]["pdf_text_line"] == pdf_text_line


def test_robie_hemingway_aliases_and_printed_phase_notes() -> None:
    kalsilite = species_thermo(
        "KAlSiO4", "cr", 1000.0, polymorph="kAlSiLiTe"
    )
    parenthesized_alias = species_thermo(
        "KAlSiO4", "cr", 1000.0, polymorph="(kAlSiLiTe)"
    )
    assert kalsilite.source_row_id == parenthesized_alias.source_row_id
    assert kalsilite.source_table_id == parenthesized_alias.source_table_id
    assert kalsilite.phase_identity["polymorph"] == "Kaliophillite (Kalsilite)"
    assert kalsilite.phase_identity["note"].startswith(
        "One row spans a transition: "
    )

    andalusite = species_thermo(
        "Al2SiO5", "cr", 1000.0, polymorph="Andalusite"
    )
    assert andalusite.phase_identity["note"] == (
        "Al₂SiO₅: Orthorhombic crystals 298.15 to 1800 K."
    )

    calcium_silicate = species_thermo(
        "Ca2SiO4", "cr", 500.0, polymorph="Larnite"
    )
    assert calcium_silicate.phase_identity["note"].startswith(
        "One row spans a transition: "
    )

    for polymorph, note in (
        (
            "PYRITE",
            "FeS₂: Cubic crystals 298.15 to 1000 K. "
            "Reference state for sulfur is orthorhombic sulfur.",
        ),
        (
            "MARCASITE",
            "FeS₂: Orthorhombic crystals 298.15 to 700 K. "
            "Reference state for sulfur is orthorhombic sulfur.",
        ),
    ):
        result = species_thermo("FeS2", "cr", 500.0, polymorph=polymorph)
        assert result.phase_identity["note"] == note

    chalcocite = species_thermo("Cu2S", "cr", 500.0, polymorph="CHALCOCITE")
    assert chalcocite.phase_identity["note"].startswith(
        "One row spans a transition: "
    )

    source_path = Path(__file__).parents[1] / (
        "src/openimcc/data/diagnostics/rh-crystal-species.json"
    )
    phases = json.loads(source_path.read_text(encoding="utf-8"))["phases"]
    source_notes = {
        phase["id"]: phase["source_formula_note"]
        for phase in phases
        if "source_formula_note" in phase
    }
    assert source_notes["larnite"] == (
        "The element-stoichiometry formula is printed as α-Ca₂SiO₄."
    )
    assert source_notes["calcio-olivine"] == (
        "The element-stoichiometry formula is printed as γ-Ca₂SiO₄."
    )
    assert source_notes["kalsilite"] == (
        "The element-stoichiometry formula is printed as KAlSiO₄."
    )
    margarite = next(phase for phase in phases if phase["id"] == "margarite")
    assert margarite["source_formula_as_published"] == (
        "CaAl2[Al2Si2]O10(OH)2"
    )
    assert source_notes["margarite"] == (
        "The element-stoichiometry formula is printed as "
        "CaAl2[Al2Si2]O10(OH)2."
    )
    assert "engine" not in " ".join(source_notes.values()).casefold()
    assert "formula weight is 258.160" in source_notes["kaolinite"]


@pytest.mark.parametrize(
    ("formula", "polymorph", "temperature_K", "phase_id"),
    (
        ("Th", "alpha", 1000.0, "thorium-alpha"),
        ("Th", "beta", 1650.0, "thorium-beta"),
        ("U", "alpha", 800.0, "uranium-alpha"),
        ("U", "beta", 942.0, "uranium-beta"),
        ("U", "gamma", 1049.0, "uranium-gamma"),
        ("Ag", "face-centered cubic", 500.0, "silver-fcc"),
    ),
)
def test_rh_ascii_polymorph_aliases_keep_phase_identity(
    formula: str, polymorph: str, temperature_K: float, phase_id: str,
) -> None:
    result = species_thermo(
        formula, "cr", temperature_K, polymorph=polymorph
    )

    assert result.source_row_id.endswith(f":{phase_id}")


@pytest.mark.parametrize(
    ("formula", "polymorph", "temperature_K", "interval_K"),
    (
        ("Th", "alpha", 1650.0, (298.15, 1650.0)),
        ("U", "alpha", 942.0, (298.15, 942.0)),
        ("U", "beta", 1049.0, (942.0, 1049.0)),
    ),
)
def test_named_rh_lower_phase_interval_is_half_open(
    formula: str,
    polymorph: str,
    temperature_K: float,
    interval_K: tuple[float, float],
) -> None:
    with pytest.raises(ImccGasTemperatureOutsideDomainError) as refusal:
        species_thermo(
            formula, "cr", temperature_K, polymorph=polymorph
        )

    lower_K, upper_K = interval_K
    assert refusal.value.declared_interval_K == interval_K
    assert f"[{lower_K:g}, {upper_K:g}) K" in str(refusal.value)


def test_named_janaf_segment_refusal_reports_segment_interval() -> None:
    with pytest.raises(ImccGasTemperatureOutsideDomainError) as refusal:
        species_thermo("PbO", "cr", 500.0, polymorph="II")

    assert refusal.value.declared_interval_K == (762.0, 1159.0)
    assert "(762, 1159) K" in str(refusal.value)
    assert "phase segment 'II'" in str(refusal.value)


def test_node_coverage_bounds_perovskite_interval_and_transition_note() -> None:
    table_perovskite = species_thermo("CaTiO3", "cr", 1400.0)
    assert table_perovskite.source_table_id == "USGS-B2131-p232"
    assert table_perovskite.T_min == 298.15
    assert table_perovskite.T_max == 1800.0

    perovskite = species_thermo("CaTiO3", "cr", 1600.0)
    assert perovskite.T_min == 1500.0
    assert perovskite.T_max == 1800.0
    assert perovskite.phase_identity is not None
    assert perovskite.phase_identity["note"] == (
        "One row spans a transition: Orthorhombic crystals 1530 K; "
        "cubic crystals 1530 to 1800 K."
    )
    for temperature, expected in {
        1500.0: {
            "Cp_J_molK": "0x1.3d051eb851eb8p+7",
            "S_J_molK": "0x1.241999999999ap+8",
            "H_app_kJ_mol": "-0x1.79023d70a3d70p+10",
            "G_J_mol": "-0x1.db248ffffffffp+20",
        },
        1600.0: {
            "Cp_J_molK": "0x1.3d051eb851eb8p+7",
            "S_J_molK": "0x1.2e75c28f5c28fp+8",
            "H_app_kJ_mol": "-0x1.750bc6a7ef9dbp+10",
            "G_J_mol": "-0x1.e273800000000p+20",
        },
        1700.0: {
            "Cp_J_molK": "0x1.0bfae147ae148p+7",
            "S_J_molK": "0x1.36947ae147ae1p+8",
            "H_app_kJ_mol": "-0x1.71b23d70a3d70p+10",
            "G_J_mol": "-0x1.e9ef2ffffffffp+20",
        },
        1800.0: {
            "Cp_J_molK": "0x1.0c9eb851eb856p+7",
            "S_J_molK": "0x1.3e47ae147ae14p+8",
            "H_app_kJ_mol": "-0x1.6e56a7ef9db22p+10",
            "G_J_mol": "-0x1.f19f1ffffffffp+20",
        },
    }.items():
        assert _value_hexes(species_thermo("CaTiO3", "cr", temperature)) == expected


@pytest.mark.parametrize(
    ("formula", "temperature_K", "polymorph", "expected_note"),
    (
        (
            "As", 600.0, None,
            "As: Rhombohedral crystals 298.15 to sublimation point 875 K. "
            "Ideal tetratomic gas 875 to 1800 K.",
        ),
        (
            "Sb", 500.0, None,
            "Sb: Rhombohedral crystals 298.15 to melting point 904 K. "
            "Liquid 904 to 1800 K.",
        ),
        (
            "Te", 500.0, None,
            "Te: Crystals 298.15 to melting point 723 K. "
            "Liquid 723 to boiling point 1261 K. "
            "Ideal diatomic gas 1261 to 1800 K.",
        ),
        (
            "Ce", 800.0, None,
            "Ce: α-crystals (face-centered cubic) 298.15 to 999 K.",
        ),
        (
            "KNO3", 350.0, None,
            "KNO₃: Orthorhombic crystals 298.15 to 401 K; "
            "rhombohedral crystals 401 to 610 K; liquid 610 to 700 K.",
        ),
        (
            "KNO3", 550.0, None,
            "KNO₃: Orthorhombic crystals 298.15 to 401 K; "
            "rhombohedral crystals 401 to 610 K; liquid 610 to 700 K.",
        ),
        (
            "SnS", 500.0, None,
            "SnS: Orthorhombic crystals 298.15 to 875 K; cubic crystals "
            "875 to melting point 1153 K. Liquid 1153 to 1300 K. "
            "Reference state for sulfur is orthorhombic sulfur.",
        ),
        (
            "SnS", 1000.0, None,
            "SnS: Orthorhombic crystals 298.15 to 875 K; cubic crystals "
            "875 to melting point 1153 K. Liquid 1153 to 1300 K. "
            "Reference state for sulfur is orthorhombic sulfur.",
        ),
        (
            "Sn", 500.0, None,
            "Sn: Crystals 298.15 to melting point 505.1 K. "
            "Liquid 501 to 1800 K.",
        ),
        (
            "FeS2", 500.0, "PYRITE",
            "FeS₂: Cubic crystals 298.15 to 1000 K. "
            "Reference state for sulfur is orthorhombic sulfur.",
        ),
        (
            "CaTiO3", 1600.0, None,
            "One row spans a transition: Orthorhombic crystals 1530 K; "
            "cubic crystals 1530 to 1800 K.",
        ),
        (
            "NaAlSi3O8", 1000.0, "ANALBITE",
            "NaAlSi3O8: Triclinic crystals 298.15 to 1400 K.",
        ),
    ),
)
def test_rh_phase_notes_only_report_printed_transitions(
    formula: str,
    temperature_K: float,
    polymorph: str | None,
    expected_note: str,
) -> None:
    result = species_thermo(
        formula, "cr", temperature_K, polymorph=polymorph
    )

    assert result.phase_identity["note"] == expected_note


def test_perovskite_refusal_names_the_printed_table_limit() -> None:
    with pytest.raises(ImccGasTemperatureOutsideDomainError) as refusal:
        species_thermo("CaTiO3", "cr", 2200.0)

    assert refusal.value.declared_interval_K == (298.15, 1800.0)
    assert "printed source table limit of 1800 K" in str(refusal.value)


def test_near_equal_polymorph_g_values_are_reported_as_a_tie() -> None:
    result = species_thermo("CaSiO3", "cr", 1400.083524)

    assert result.phase_identity is not None
    assert result.phase_identity["polymorph"] == "wollastonite"
    tie_note = result.phase_identity["tie_note"]
    assert "wollastonite" in tie_note
    assert "pseudowollastonite" in tie_note
    assert "1e-6 relative tolerance" in tie_note


def test_crystal_source_caps_and_declared_intervals_are_refused() -> None:
    with pytest.raises(
        ImccGasTemperatureOutsideDomainError,
        match="above its source solid limit of 1875 K",
    ):
        species_thermo(
            "CaAl2O4", "cr", 1876.0, allow_extrapolation=True
        )

    continued = species_thermo(
        "CaAl2O4", "cr", 1850.0, allow_extrapolation=True
    )
    assert continued.extrapolated is True
    assert continued.Cp_J_molK == 193.101
    assert continued.continuation is not None

    wollastonite = species_thermo("CaSiO3", "cr", 1200.0)
    assert wollastonite.phase_identity is not None
    assert wollastonite.phase_identity["polymorph"] == "wollastonite"
    assert wollastonite.declared_interval_K == (298.15, 1817.0)


def test_berman_public_service_starts_at_source_supported_temperature(monkeypatch) -> None:
    supported_rows = (
        ("Mg2SiO4", "forsterite", 2163.0),
        ("MgSiO3", "enstatite", 1830.0),
        ("CaMgSi2O6", "diopside", 1668.0),
        ("CaAl2Si2O8", "anorthite", 1826.0),
        ("MgAl2O4", "spinel", 2408.0),
        ("Ca2MgSi2O7", "akermanite", 1727.0),
        ("Ca2Al2SiO7", "gehlenite", 1823.0),
        ("Ca3MgSi2O8", "merwinite", 1840.0),
        ("CaMgSiO4", "monticellite", 1573.0),
        ("CaSiO3", "wollastonite", 1817.0),
        ("CaMg(SiO3)2", "diopside", 1668.0),
    )
    for formula, phase_id, upper_K in supported_rows:
        result = species_thermo(formula, "cr", 298.15)
        assert result.declared_interval_K == (298.15, upper_K)
        assert result.phase_identity is not None
        if formula == "MgAl2O4":
            # The higher-priority Barin row already covers its source anchor.
            assert result.source_table_id == "Barin-1995"
        else:
            assert result.source_table_id == "Berman-1988"
            assert result.phase_identity["polymorph"] == phase_id
            assert result.authority == "published_fitted"

    # Barin normally outranks the Berman spinel row. With higher-priority
    # sources removed, the Berman row itself keeps its 298.15 K service floor.
    monkeypatch.setattr(crystal_species, "_load_janaf_phases", lambda: ({}, ""))
    monkeypatch.setattr(
        crystal_species, "_load_janaf_crystal_species", lambda: ({}, "")
    )
    monkeypatch.setattr(crystal_species, "_barin_phases", lambda: ())
    berman_spinel = next(
        row for row in _berman_phases() if row["id"] == "spinel"
    )
    for temperature_K in (298.15, 1000.0):
        spinel = species_thermo("MgAl2O4", "cr", temperature_K)
        assert spinel.source_table_id == "Berman-1988"
        assert spinel.declared_interval_K == (298.15, 2408.0)
        assert spinel.authority == "published_fitted"
        assert (
            spinel.Cp_J_molK,
            spinel.S_J_molK,
            spinel.H_app_kJ_mol,
            spinel.G_J_mol,
        ) == _berman_properties(berman_spinel, temperature_K)

    with pytest.raises(ImccGasTemperatureOutsideDomainError) as refusal:
        species_thermo("Mg2SiO4", "cr", 298.14)
    assert refusal.value.declared_interval_K == (298.15, 2163.0)

    inside_condensate = species_thermo("SiO2", "cr", 1500.0)
    assert _value_hexes(inside_condensate) == {
        "Cp_J_molK": "0x1.240610b27dc5cp+6",
        "S_J_molK": "0x1.25a4de21ba0e6p+7",
        "H_app_kJ_mol": "-0x1.9ddd493cd318dp+9",
        "G_J_mol": "-0x1.ffb339de40090p+19",
    }

    previously_served = species_thermo("Mg2SiO4", "cr", 1500.0)
    assert _value_hexes(previously_served) == {
        "Cp_J_molK": "0x1.75dd4b3f35f87p+7",
        "S_J_molK": "0x1.5da710d34f166p+8",
        "H_app_kJ_mol": "-0x1.ed8950cb952a0p+10",
        "G_J_mol": "-0x1.310203a818f0cp+21",
    }


def test_berman_high_temperature_polymorph_floors(monkeypatch) -> None:
    monkeypatch.setattr(crystal_species, "_load_janaf_phases", lambda: ({}, ""))
    monkeypatch.setattr(
        crystal_species, "_load_janaf_crystal_species", lambda: ({}, "")
    )
    monkeypatch.setattr(crystal_species, "_barin_phases", lambda: ())

    for phase_id, floor_K in (
        ("quartz_beta", 848.0),
        ("cristobalite_beta", 535.0),
    ):
        with pytest.raises(CrystalSpeciesTemperatureError) as refusal:
            evaluate_crystal_species("SiO2", floor_K - 0.01, polymorph=phase_id)
        assert refusal.value.declared_interval_K[0] == floor_K
        result = evaluate_crystal_species("SiO2", floor_K, polymorph=phase_id)
        assert result is not None
        assert result.T_min == floor_K

    with pytest.raises(CrystalSpeciesTemperatureError) as refusal:
        evaluate_crystal_species("SiO2", 1322.99, polymorph="tridymite_high")
    assert refusal.value.declared_interval_K[0] == 1323.0
    tridymite = evaluate_crystal_species(
        "SiO2", 1323.0, polymorph="tridymite_high"
    )
    assert tridymite is not None
    assert tridymite.T_min == 1323.0

    pseudowollastonite = species_thermo(
        "CaSiO3", "cr", 1350.0, polymorph="pseudowollastonite"
    )
    assert pseudowollastonite.declared_interval_K == (1323.0, 1817.0)
    assert pseudowollastonite.phase_identity is not None
    assert pseudowollastonite.phase_identity["polymorph"] == "pseudowollastonite"
    assert _value_hexes(pseudowollastonite) == {
        "Cp_J_molK": "0x1.fbe4edf0ef261p+6",
        "S_J_molK": "0x1.fa6f7f5498e75p+7",
        "H_app_kJ_mol": "-0x1.785e39f457041p+10",
        "G_J_mol": "-0x1.c30144645dec9p+20",
    }

    below_transition = species_thermo("CaSiO3", "cr", 1397.99)
    assert below_transition.phase_identity is not None
    assert below_transition.phase_identity["polymorph"] == "wollastonite"


def test_berman_and_robie_hemingway_element_reference_anchors() -> None:
    phases = _berman_phases()
    forsterite = next(row for row in phases if row["id"] == "forsterite")
    _cp, entropy, enthalpy, _gibbs = _berman_properties(forsterite, 298.15)
    assert enthalpy * 1000.0 == forsterite["berman_parameters"]["HfPrTr_J_mol"]
    assert entropy == forsterite["berman_parameters"]["SPrTr_J_mol_K"]

    anorthite = next(row for row in phases if row["id"] == "anorthite")
    _cp, _entropy, _enthalpy, gibbs = _berman_properties(anorthite, 1800.0)
    element_g = _element_reference_sum_kJ_mol("CaAl2Si2O8", 1800.0)
    assert element_g is not None
    assert gibbs / 1000.0 - element_g == pytest.approx(
        anorthite["rh_crosscheck"]["G_crystal_1800_kJ_mol"], abs=2.5
    )

    wollastonite = next(row for row in phases if row["id"] == "wollastonite")
    for temperature in (1500.0, 1600.0, 1700.0, 1800.0):
        parent_gibbs = math.fsum(
            float(coefficient)
            * species_thermo(oxide, "l", temperature).G_J_mol
            for oxide, coefficient in wollastonite["oxide_stoichiometry"].items()
        )
        crystal_gibbs = _berman_properties(wollastonite, temperature)[3]
        source_fusion = float(
            wollastonite["fusion_gibbs_nodes_kJ_mol"][str(int(temperature))]
        )
        assert (parent_gibbs - crystal_gibbs) / 1000.0 == pytest.approx(
            source_fusion, abs=2.5
        )

    perovskite = next(
        row for row in _crystal_dataset()["phases"]
        if row["id"] == "perovskite"
    )
    residuals = []
    for temperature, source_g in perovskite[
        "source_crystal_gibbs_nodes_kJ_mol"
    ].items():
        t = float(temperature)
        result = species_thermo("CaTiO3", "cr", t, polymorph="perovskite")
        element_g = _element_reference_sum_kJ_mol("CaTiO3", t)
        assert element_g is not None
        residuals.append(
            abs(result.G_J_mol / 1000.0 - element_g - float(source_g))
        )
    assert max(residuals) == pytest.approx(2.3387, abs=1.0e-3)
    assert max(residuals) <= 2.5


def test_berman_lambda_transition_and_disorder_terms_are_retained() -> None:
    phases = _berman_phases()
    akermanite = next(row for row in phases if row["id"] == "akermanite")
    params = akermanite["berman_parameters"]
    assert params["Tlambda_K"] == 358.0
    coefficients = tuple(float(params.get(f"k{i}", 0.0)) for i in range(7))
    base_h, base_s = _cp_primitive(coefficients, 400.0, 298.15)
    _cp, entropy, enthalpy, _g = _berman_properties(akermanite, 400.0)
    assert enthalpy == pytest.approx(
        (float(params["HfPrTr_J_mol"]) + base_h + params["DtH_J_mol"]) / 1000.0
    )
    assert entropy == pytest.approx(
        params["SPrTr_J_mol_K"] + base_s
        + params["DtH_J_mol"] / params["Tlambda_K"]
    )
    public_akermanite = species_thermo("Ca2MgSi2O7", "cr", 400.0)
    assert (
        public_akermanite.Cp_J_molK,
        public_akermanite.S_J_molK,
        public_akermanite.H_app_kJ_mol,
        public_akermanite.G_J_mol,
    ) == _berman_properties(akermanite, 400.0)

    gehlenite = next(row for row in phases if row["id"] == "gehlenite")
    params = gehlenite["berman_parameters"]
    disorder = gehlenite["disorder_parameters"]
    assert (params["Tmin_K"], params["Tmax_K"]) == (698.0, 1600.0)
    coefficients = tuple(float(params.get(f"k{i}", 0.0)) for i in range(7))
    disorder_coefficients = (
        disorder["d0"], disorder["d1"], disorder["d2"], 0.0, 0.0,
        disorder["d3"], disorder["d4"],
    )
    for temperature_K in (1000.0, 1700.0):
        base_h, base_s = _cp_primitive(coefficients, temperature_K, 298.15)
        disorder_stop_K = min(
            max(temperature_K, params["Tmin_K"]), params["Tmax_K"]
        )
        disorder_h, disorder_s = _cp_primitive(
            disorder_coefficients, disorder_stop_K, params["Tmin_K"]
        )
        cp, entropy, enthalpy, _g = _berman_properties(gehlenite, temperature_K)
        assert enthalpy == pytest.approx(
            (float(params["HfPrTr_J_mol"]) + base_h + disorder_h) / 1000.0
        )
        assert entropy == pytest.approx(
            params["SPrTr_J_mol_K"] + base_s + disorder_s
        )
        lattice_cp = math.fsum((
            coefficients[0],
            coefficients[1] / math.sqrt(temperature_K),
            coefficients[2] / temperature_K**2,
            coefficients[3] / temperature_K**3,
            coefficients[4] / temperature_K,
            coefficients[5] * temperature_K,
            coefficients[6] * temperature_K**2,
        ))
        disorder_cp = 0.0
        if params["Tmin_K"] <= temperature_K <= params["Tmax_K"]:
            disorder_cp = math.fsum((
                disorder["d0"],
                disorder["d1"] / math.sqrt(temperature_K),
                disorder["d2"] / temperature_K**2,
                disorder["d3"] * temperature_K,
                disorder["d4"] * temperature_K**2,
            ))
        assert cp == pytest.approx(lattice_cp + disorder_cp)
        if temperature_K == 1000.0:
            public_gehlenite = species_thermo(
                "Ca2Al2SiO7", "cr", temperature_K
            )
            assert public_gehlenite.source_table_id == "Berman-1988"
            assert public_gehlenite.H_app_kJ_mol == enthalpy
            assert public_gehlenite.S_J_molK == entropy
            assert public_gehlenite.Cp_J_molK == pytest.approx(cp)
            assert public_gehlenite.G_J_mol == pytest.approx(_g)

    for phase_id in ("anorthite", "spinel"):
        phase = next(row for row in phases if row["id"] == phase_id)
        assert phase["disorder_parameters"] is None
        params = phase["berman_parameters"]
        coefficients = tuple(float(params.get(f"k{i}", 0.0)) for i in range(7))
        base_h, base_s = _cp_primitive(coefficients, 1000.0, 298.15)
        cp, entropy, enthalpy, _g = _berman_properties(phase, 1000.0)
        assert enthalpy == pytest.approx(
            (float(params["HfPrTr_J_mol"]) + base_h) / 1000.0
        )
        assert entropy == pytest.approx(params["SPrTr_J_mol_K"] + base_s)
        if phase_id == "anorthite":
            public_result = species_thermo(phase["formula"], "cr", 1000.0)
            assert public_result.source_table_id == "Berman-1988"
            assert public_result.Cp_J_molK == pytest.approx(cp)
            assert public_result.H_app_kJ_mol == enthalpy
            assert public_result.S_J_molK == entropy
            assert public_result.G_J_mol == pytest.approx(_g)


def test_barin_apparent_g_reconstructs_printed_delta_f_g_nodes() -> None:
    residuals = []
    for phase in _barin_phases():
        for polymorph in phase["polymorphs"]:
            for node in polymorph["source_series"]["delta_fG"]["nodes"]:
                temperature = float(node["T_K"])
                element_g = _element_reference_sum_kJ_mol(
                    phase["formula"], temperature
                )
                if element_g is None:
                    continue
                properties = evaluate_crystal_species(
                    phase["formula"], temperature,
                    polymorph=polymorph["label"],
                )
                assert properties is not None
                reconstructed = properties.G_J_mol / 1000.0 - element_g
                residuals.append(abs(reconstructed - float(node["value"])))

    # JANAF elemental references and Barin's printed formation values differ
    # by at most 19.6294 kJ/mol at the shared C12A7/1700 K node.
    assert max(residuals) == pytest.approx(19.6294, abs=1.0e-3)
    assert max(residuals) <= 20.0


def test_condensate_precedence_and_crystal_source_spread() -> None:
    spreads_J_mol = []
    for temperature in (1500.0, 1550.0, 1750.0):
        condensate = species_thermo("SiO2", "cr", temperature)
        dataset = evaluate_crystal_species("SiO2", temperature)
        assert dataset is not None
        assert condensate.source_table_id == "O-035"
        assert condensate.G_J_mol == species_thermo(
            "SiO2", "cr", temperature
        ).G_J_mol
        assert dataset.G_J_mol == evaluate_janaf_crystal(
            "SiO2", temperature
        ).G_J_mol
        spreads_J_mol.append(abs(condensate.G_J_mol - dataset.G_J_mol))
    assert max(spreads_J_mol) == pytest.approx(59.0223639, abs=1.0e-4)
    assert max(spreads_J_mol) <= 60.0


def test_readme_basalt_evaluate_gas_probe_is_unchanged() -> None:
    from openimcc import evaluate as evaluate_imcc

    basalt = {
        "SiO2": 51.85068,
        "MgO": 4.78527,
        "FeO": 13.77307,
        "CaO": 9.02862,
        "Al2O3": 14.80572,
        "TiO2": 1.73824,
        "Na2O": 3.23108,
        "K2O": 0.78732,
    }
    melt = evaluate_imcc(basalt, 1800.0, basis_type="wt")
    activities = {name: melt.activity(name) for name in melt.parent_oxides}
    assert {name: f"{value:.6g}" for name, value in activities.items()} == {
        "SiO2": "0.315634",
        "MgO": "0.00434375",
        "FeO": "0.0759",
        "CaO": "7.42238e-05",
        "Al2O3": "0.0128061",
        "TiO2": "0.00252514",
        "Na2O": "4.28163e-10",
        "K2O": "8.22663e-15",
    }
    gas = evaluate_gas(activities, 1800.0, 1.0e-10, load_gas_datapack())
    assert gas.unit == "bar"
    assert f"{gas['Na']:.6g}" == "0.000266815"
    assert f"{gas['K']:.6g}" == "6.99669e-06"
    assert gas.domain_flags["Mg"] == (
        "MgO(l) uses a labelled constant-Cp supercooled-liquid continuation "
        "from JANAF Mg-009; this is a hypothetical reference, not a prediction "
        "of a liquid condensate; band ±0.76 dex pO2"
    )


def test_species_thermo_extrapolation_reuses_gas_runtime_rows() -> None:
    pack = load_gas_datapack()
    with pytest.raises(ImccGasTemperatureOutsideDomainError):
        species_thermo("O2", "g", 499.0, pack)
    with pytest.raises(ImccGasTemperatureOutsideDomainError):
        species_thermo("SiO2", "l", 1199.0, pack)

    gas = species_thermo("O2", "g", 499.0, pack, allow_extrapolation=True)
    gas_row = _nearest_interval_row(
        pack.gas_df, "O2(g)", 499.0, allow_extrapolation=True
    )
    assert gas.G_J_mol == _janaf_gibbs(499.0, gas_row)
    assert gas.source_row_id == str(gas_row["Ref"])
    assert gas.extrapolated is True
    assert gas.declared_interval_K == (
        float(gas_row["T_min"]), float(gas_row["T_max"])
    )
    assert gas.continuation == "Shomate fit continued"
    assert gas.authority

    liquid = species_thermo(
        "SiO2", "l", 1199.0, pack, allow_extrapolation=True
    )
    liquid_row = _oxide_row_for_T(
        pack.oxide_df, "SiO2(l)", 1199.0, allow_extrapolation=True
    )
    assert liquid.G_J_mol == _lamor_gibbs(1199.0, liquid_row)
    assert liquid.source_row_id == str(liquid_row["Ref"])
    assert liquid.extrapolated is True
    assert liquid.declared_interval_K == (
        float(liquid_row["T_min"]), float(liquid_row["T_max"])
    )
    assert liquid.continuation is not None
    assert liquid.authority


def test_gas_formula_spelling_alias_uses_element_counts() -> None:
    hydroxyl = species_thermo("OH", "g", 2000.0)
    alias = species_thermo("HO", "g", 2000.0)

    assert _value_hexes(alias) == _value_hexes(hydroxyl)
    assert alias.source_row_id == hydroxyl.source_row_id
    assert alias.source_table_id == hydroxyl.source_table_id


def test_gas_formula_spelling_alias_refuses_duplicate_compositions() -> None:
    pack = load_gas_datapack()
    alternate_name = pack.gas_df.loc[pack.gas_df.index == "OH(g)"].copy()
    alternate_name.index = ["H1O1(g)"] * len(alternate_name)
    duplicate_pack = replace(
        pack, gas_df=pd.concat((pack.gas_df, alternate_name))
    )

    with pytest.raises(ImccGasSpeciesNotFoundError) as refusal:
        species_thermo("HO", "g", 2000.0, duplicate_pack)
    assert "OH(g)" in str(refusal.value)
    assert "H1O1(g)" in str(refusal.value)


def test_species_thermo_crystal_phase_identity_and_formation_anchors() -> None:
    pack = load_gas_datapack()
    high_cristobalite = species_thermo("SiO2", "cr", 1500.0, pack)
    assert high_cristobalite.phase_identity is not None
    assert high_cristobalite.phase_identity["kind"] == "single_polymorph"
    assert high_cristobalite.phase_identity["polymorph"] == "high cristobalite"
    assert high_cristobalite.phase_identity["transition_temperatures_K"][0][
        "temperature_K"
    ] == 1079
    assert "metastable below 1079 K" in high_cristobalite.phase_identity[
        "phase_notes"
    ]
    assert high_cristobalite.formation_enthalpy_298_kJ_mol == pytest.approx(
        -905.489
    )
    assert high_cristobalite.formation_enthalpy_source_id == "O-035"

    legacy_table = (
        Path(__file__).parents[1]
        / "src/openimcc/data/gas/condensate-feo-janaf-legacy.csv"
    )
    lam_pack = load_gas_datapack(oxide_path=legacy_table)
    lam_cristobalite = species_thermo("SiO2", "cr", 1500.0, lam_pack)
    assert lam_cristobalite.source_table_id == "LH87 Table 2"
    assert lam_cristobalite.phase_identity is not None
    assert lam_cristobalite.phase_identity["polymorph"] == (
        "cristobalite (low/high subtype not specified)"
    )
    assert lam_cristobalite.formation_enthalpy_298_kJ_mol == pytest.approx(
        -910.683
    )
    assert lam_cristobalite.phase_identity[
        "formation_enthalpy_reference_phase"
    ] == "quartz (SiO2(QU))"
    assert lam_cristobalite.formation_enthalpy_source_id == "LH87 Table 3"

    carbon = species_thermo("C", "cr", 1500.0, pack)
    assert carbon.phase_identity is not None
    assert carbon.phase_identity["polymorph"] == "C1(ref)"
    assert "graphite" not in carbon.phase_identity["phase_notes"]

    gas = species_thermo("H", "g", 2000.0, pack)
    assert gas.formation_enthalpy_298_kJ_mol == pytest.approx(217.999)
    assert gas.formation_enthalpy_source_id == "H-001"


def test_gas_formation_enthalpy_uses_source_anchor_across_segments() -> None:
    pack = load_gas_datapack()
    expected = (
        ("O2", 0.0, "O-029"),
        ("SiO", -100.416, "O-012"),
        ("Na", 107.3, "Na-005"),
    )
    for species, value, source_id in expected:
        low = species_thermo(species, "g", 1000.0, pack)
        high = species_thermo(species, "g", 2000.0, pack)
        assert low.formation_enthalpy_298_kJ_mol == value
        assert high.formation_enthalpy_298_kJ_mol == value
        assert low.formation_enthalpy_source_id == source_id
        assert high.formation_enthalpy_source_id == source_id


def test_gas_formation_enthalpy_uses_vendored_text_table_anchors() -> None:
    pack = load_gas_datapack()
    for species, value, source_id in (
        ("B", 560.0, "B-005"),
        ("Li", 159.3, "Li-005"),
    ):
        result = species_thermo(species, "g", 2000.0, pack)
        assert result.formation_enthalpy_298_kJ_mol == value
        assert result.formation_enthalpy_source_id == source_id


def test_species_thermo_uses_os_card_after_condensate_fit_ends() -> None:
    pack = load_gas_datapack()
    edge = species_thermo("Os", "cr", 3000.0, pack)
    card = species_thermo(
        "Os", "cr", 3100.0, pack, allow_extrapolation=True
    )

    assert edge.declared_interval_K == (1200.0, 3000.0)
    assert card.source_table_id == "BURCAT-OS-CR"
    assert card.extrapolated is False
    assert card.declared_interval_K == (298.15, 3300.0)
    assert card.continuation is None
    assert card == species_thermo("Os", "cr", 3100.0, pack)


def test_species_thermo_refuses_transition_and_melting_crossings() -> None:
    with pytest.raises(
        ImccGasTemperatureOutsideDomainError,
        match="outside its source-printed stability range.*transition at 1079 K",
    ):
        species_thermo(
            "SiO2", "cr", 999.0, allow_extrapolation=True
        )

    with pytest.raises(
        ImccGasTemperatureOutsideDomainError,
        match="above its source-printed melting point of 1996 K",
    ):
        species_thermo(
            "SiO2", "cr", 1997.0, allow_extrapolation=True
        )

    re = species_thermo("Re", "cr", 3452.0, allow_extrapolation=True)
    assert re.extrapolated is True
    assert re.phase_identity is not None
    assert re.phase_identity["melting_temperature_K"] == 3453
    with pytest.raises(
        ImccGasTemperatureOutsideDomainError,
        match="above its source-printed melting point of 3453 K",
    ):
        species_thermo("Re", "cr", 3453.0, allow_extrapolation=True)


def test_species_thermo_domain_error_reports_selected_interval() -> None:
    pack = load_gas_datapack()
    with pytest.raises(ImccGasTemperatureOutsideDomainError) as refusal:
        species_thermo("SiO2", "cr", 999.0, pack)

    assert str(refusal.value) == (
        "T=999.0 K outside declared G(T) interval for 'SiO2(cr)' "
        "[1000.0, 1996.0] K"
    )
    assert refusal.value.declared_interval_K == (1000.0, 1996.0)
    assert refusal.value.species == "SiO2(cr)"
    assert refusal.value.source_row_id == "O-035"

    without_row = ImccGasTemperatureOutsideDomainError("outside")
    assert without_row.declared_interval_K is None
    assert without_row.species is None
    assert without_row.source_row_id is None


@pytest.mark.parametrize(
    ("species", "phase", "temperature", "channel", "activities", "expected"),
    (
        (
            "SiO2", "l", 1199.0, "Si", {"SiO2": 1.0e-3},
            "SiO2(l) uses a tier-1 liquid_from_solid construction from "
            "Richet-Bottinga-Tequi-1984",
        ),
        (
            "Al2O3", "l", 999.0, "Al", {"Al2O3": 1.0e-3},
            "Al2O3(l) uses a labelled constant-Cp supercooled-liquid "
            "continuation from JANAF Al-100; this is a hypothetical reference, "
            "not a prediction of a liquid condensate",
        ),
        (
            "UO2", "l", 3001.0, "U", {"UO2": 1.0e-3},
            "UO2(l) uses a labelled constant-Cp supercooled-liquid "
            "continuation from NASA Glenn card NG-1977",
        ),
        (
            "ZnO", "l", 3001.0, "Zn", {"ZnO": 1.0e-3},
            "ZnO(l) uses a tier-4 liquid_from_solid construction from "
            "TSIV-1603; the central liquid uses the printed 2250 K node, and "
            "the quadratic crystal-Cp extension is its uncertainty bound",
        ),
        (
            "Re2O7", "g", 499.0, "Re2O7", {"Re": 1.0},
            "constructed Re2O7(g) from Smith 1952 measured vapor pressure "
            "and B677 condensed functions over 503.15-598.15 K; maximum "
            "log-pressure fit scatter propagated as ±1.3 kJ/mol at 499 K",
        ),
    ),
)
def test_continuation_label_matches_evaluate_gas_flag(
    packaged_gas_pack,
    species: str,
    phase: str,
    temperature: float,
    channel: str,
    activities: dict[str, float],
    expected: str,
) -> None:
    thermo = species_thermo(
        species,
        phase,
        temperature,
        packaged_gas_pack,
        allow_extrapolation=True,
    )
    result = evaluate_gas(
        activities,
        temperature,
        1.0e-8,
        packaged_gas_pack,
        gas_species=(channel,),
        allow_extrapolation=True,
    )

    assert thermo.continuation == expected
    runtime_flag = result.domain_flags[channel]
    assert runtime_flag is not None
    assert expected in runtime_flag
