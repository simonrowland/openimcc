"""Source closure, screening, and status checks for each supported element."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import pytest
import yaml

from openimcc import evaluate as evaluate_imcc
from openimcc.gas import (
    ELEMENT_STATUS,
    IMCC_GAS_CHANNEL_SPECIES,
    R_J_MOL_K,
    evaluate_gas,
    load_gas_datapack,
)


ROOT = Path(__file__).resolve().parents[1]
JANAF_DATA = ROOT / "data-src" / "janaf"
GAS_PROVENANCE = ROOT / "src" / "openimcc" / "data" / "gas" / "PROVENANCE.yaml"
README_BASALT = {
    "SiO2": 51.85068,
    "MgO": 4.78527,
    "FeO": 13.77307,
    "CaO": 9.02862,
    "Al2O3": 14.80572,
    "TiO2": 1.73824,
    "Na2O": 3.23108,
    "K2O": 0.78732,
}
FUGACITIES = (1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6, 1.0e-4)
TEMPERATURES_K = tuple(float(T) for T in range(1500, 3001, 100))
SCREEN_SPECIES = {
    "Cr": ("Cr", "CrO", "CrO2", "CrO3"),
    "V": ("V", "VO", "VO2"),
    "Nb": ("Nb", "NbO", "NbO2"),
}


def _record(table_id: str) -> dict:
    text = (JANAF_DATA / f"{table_id}.yaml").read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return yaml.safe_load(text)


def _neutral_oxide_sources() -> dict[str, tuple[str, str]]:
    """Return source-table candidates: table ID -> (element, neutral gas)."""
    result = {}
    for path in JANAF_DATA.glob("*.yaml"):
        record = _record(path.stem)
        entry = record["table"]["index_entry"]
        formula = entry.get("formula_normalised", entry.get("formula"))
        if entry.get("state") != "g" or entry.get("charge", 0) != 0:
            continue
        if "O" not in formula:
            continue
        for element in ELEMENT_STATUS:
            if element == "O":
                continue
            if re.search(rf"(?<![A-Za-z]){element}(?:[0-9]*)", formula):
                result[path.stem] = (element, formula)
                break
    return result


def test_janaf_neutral_oxide_sources_are_included_or_measured_screens() -> None:
    """Use data-src plus checked-in candidates; no JANAF manifest exists."""
    source_candidates = _neutral_oxide_sources()
    recorded_candidates = {
        table_id: (element, species)
        for element, status in ELEMENT_STATUS.items()
        for table_id, species in status["c2_candidates"]
    }
    assert source_candidates == recorded_candidates

    channels = set(IMCC_GAS_CHANNEL_SPECIES)
    for element, status in ELEMENT_STATUS.items():
        assert isinstance(status["criteria"]["C2"], bool)
        for _table_id, species in status["c2_candidates"]:
            if species in channels:
                continue
            screen = status.get("c2_screened_out", {}).get(species)
            assert screen is not None, (
                f"{element} {species} is neither included nor screened"
            )
            assert math.isfinite(screen["max_ratio"])
            assert screen["max_ratio"] < 1.0e-4
            assert screen["temperature_K"] in TEMPERATURES_K
            assert screen["fO2"] in FUGACITIES


def _caller_parent_activities() -> dict[str, float]:
    return {"Cr2O3": 1.0e-3, "V2O3": 1.0e-3, "NbO2": 1.0e-3}


def _screen_maxima() -> dict[str, dict[str, dict[str, float | str]]]:
    pack = load_gas_datapack()
    activities = _caller_parent_activities()
    maxima = {element: {} for element in SCREEN_SPECIES}
    for temperature in TEMPERATURES_K:
        for fugacity in FUGACITIES:
            for element, species in SCREEN_SPECIES.items():
                pressures = evaluate_gas(
                    activities,
                    temperature,
                    fugacity,
                    pack,
                    gas_species=species,
                    allow_extrapolation=True,
                )
                dominant = max(species, key=lambda name: pressures[name])
                for name in species:
                    ratio = pressures[name] / pressures[dominant]
                    current = maxima[element].get(name)
                    if current is None or ratio > current["max_ratio"]:
                        maxima[element][name] = {
                            "max_ratio": ratio,
                            "temperature_K": temperature,
                            "fO2": fugacity,
                            "dominant": dominant,
                        }
    return maxima


def test_cr_v_nb_screen_ratios_match_the_status_source() -> None:
    measured = _screen_maxima()
    for element, species_rows in measured.items():
        recorded = ELEMENT_STATUS[element]["c2_screen_maxima"]
        assert set(recorded) == set(species_rows)
        for species, result in species_rows.items():
            expected = recorded[species]
            assert expected["temperature_K"] == result["temperature_K"]
            assert expected["fO2"] == result["fO2"]
            assert expected["dominant"] == result["dominant"]
            assert expected["max_ratio"] == pytest.approx(
                result["max_ratio"], rel=1.0e-12
            )


def _source_value(table_id: str, field: str, temperature: float) -> float:
    for row in _record(table_id)["table"]["values"]:
        if row["temperature"]["value"] == temperature:
            value = row[field]["value"]
            assert value is not None, f"{table_id} has no {field} at {temperature} K"
            return float(value)
    raise AssertionError(f"{table_id} has no JANAF row at {temperature} K")


def _neutral_gibbs_value(table_id: str, temperature: float) -> float:
    rows = _record(table_id)["table"]["values"]
    points = [
        (float(row["temperature"]["value"]), float(row["formation_gibbs_energy"]["value"]))
        for row in rows
        if row["formation_gibbs_energy"]["value"] is not None
    ]
    for source_temperature, value in points:
        if source_temperature == temperature:
            return value
    below = [point for point in points if point[0] < temperature]
    above = [point for point in points if point[0] > temperature]
    # A few neutral JANAF tables omit an isolated point on the status grid.
    # Use adjacent tabulated values only to evaluate that diagnostic grid point.
    if below and above:
        first, second = below[-1], above[0]
    elif len(below) >= 2:
        first, second = below[-2:]
    elif len(above) >= 2:
        first, second = above[:2]
    else:
        raise AssertionError(f"{table_id} cannot bracket {temperature} K")
    return first[1] + (temperature - first[0]) * (second[1] - first[1]) / (second[0] - first[0])


def _text_table_value(table_id: str, column: int, temperature: float) -> float:
    lines = (JANAF_DATA / f"{table_id}.txt").read_text(encoding="utf-8").splitlines()
    for line in lines[2:]:
        columns = line.split("\t")
        if columns and float(columns[0]) == temperature:
            value = columns[column].strip() if column < len(columns) else ""
            assert value and value != "INFINITE", (
                f"{table_id} has no value at {temperature} K"
            )
            return float(value)
    raise AssertionError(f"{table_id} has no JANAF row at {temperature} K")


def _ion_bound_maxima() -> dict[str, dict[str, float | str]]:
    pack = load_gas_datapack()
    maxima = {}
    computed = {
        element: status["c3_ion_bound"]
        for element, status in ELEMENT_STATUS.items()
        if status.get("c3_ion_bound", {}).get("status") != "not computed"
        and "c3_ion_bound" in status
    }
    assert set(computed) == {"Si", "Mg", "Fe", "Ca", "Al", "Ti", "Cr", "V", "Nb", "Na", "K"}
    caller_parent_oxides = {"Cr": "Cr2O3", "V": "V2O3", "Nb": "NbO2"}
    activities_by_temperature = {}
    for temperature in TEMPERATURES_K:
        melt = evaluate_imcc(
            README_BASALT,
            temperature,
            basis_type="wt",
            allow_extrapolation=True,
        )
        activities = {name: melt.activity(name) for name in melt.parent_oxides}
        activities.update({oxide: 1.0e-3 for oxide in caller_parent_oxides.values()})
        activities_by_temperature[temperature] = activities
    neutral_pressures = {}
    for temperature in TEMPERATURES_K:
        for fugacity in FUGACITIES:
            neutral_pressures[(temperature, fugacity)] = evaluate_gas(
                activities_by_temperature[temperature],
                temperature,
                fugacity,
                pack,
                gas_species=tuple(computed),
                allow_extrapolation=True,
            )
    for element, source in computed.items():
        cation_table = source["source_tables"]["cation"]
        assert source["user_agent"] == "openimcc-janaf-vendor/1.0"
        assert set(source["source_tables"]) == {"cation", "neutral", "electron"}
        assert source["source_tables"]["cation"] == cation_table
        assert source["source_tables"]["electron"] == "D-020"
        for table_id, sha256 in source["upstream_sha256"].items():
            yaml_path = JANAF_DATA / f"{table_id}.yaml"
            if yaml_path.exists():
                assert _record(table_id)["extraction"]["source_sha256"] == sha256
            else:
                table_path = JANAF_DATA / f"{table_id}.txt"
                assert hashlib.sha256(table_path.read_bytes()).hexdigest() == sha256
        oxide = source["parent_oxide"]
        ion_constants = {}
        for temperature in TEMPERATURES_K:
            delta_g_kj = (
                (
                    _text_table_value(cation_table, 6, temperature)
                    if cation_table in {"Na-006", "K-006"}
                    else _source_value(cation_table, "formation_gibbs_energy", temperature)
                )
                + _text_table_value("D-020", 6, temperature)
                - _neutral_gibbs_value(source["source_tables"]["neutral"], temperature)
            )
            ion_constants[temperature] = math.exp(
                -delta_g_kj * 1000.0 / (R_J_MOL_K * temperature)
            )
        maximum = None
        ratio_at_3000 = None
        for temperature in TEMPERATURES_K:
            for fugacity in FUGACITIES:
                neutral = neutral_pressures[(temperature, fugacity)][element]
                k_ion = ion_constants[temperature]
                # For E(g) ⇌ E+(g) + e−(g), Kion = (p(E+)/p°)(p(e−)/p°)/(p(E)/p°).
                # With p° = 1 bar, the numerical pressure values obey
                # p(E+)·p(e−) = Kion·p(E). Charge balance with E as the only
                # electron source sets p(E+) = p(e−), hence p(E+) =
                # sqrt(Kion·p(E)) and p(E+)/p(E) = sqrt(Kion/p(E)). If other
                # sources add electrons, p(e−) > p(E+), so equilibrium gives
                # p(E+) < sqrt(Kion·p(E)): this is a conservative upper bound.
                bound_pressure = math.sqrt(k_ion * neutral)
                ratio = bound_pressure / neutral
                if temperature == 3000.0 and (
                    ratio_at_3000 is None or ratio > ratio_at_3000
                ):
                    ratio_at_3000 = ratio
                if maximum is None or ratio > maximum["max_ratio"]:
                    maximum = {
                        "max_ratio": ratio,
                        "temperature_K": temperature,
                        "fO2": fugacity,
                        "neutral_pressure_bar": neutral,
                        "bound_pressure_bar": bound_pressure,
                        "K_ion": k_ion,
                        "parent_oxide": oxide,
                    }
        assert maximum is not None
        maximum["ratio_at_3000K"] = ratio_at_3000
        maxima[element] = maximum
    return maxima


def test_thermal_ionisation_bounds_match_the_status_source() -> None:
    measured = _ion_bound_maxima()
    assert {path.stem for path in JANAF_DATA.glob("*.txt")} == {
        "Na-006", "K-006", "D-020"
    }
    assert set(measured) == {"Si", "Mg", "Fe", "Ca", "Al", "Ti", "Cr", "V", "Nb", "Na", "K"}
    for element, result in measured.items():
        recorded = ELEMENT_STATUS[element]["c3_ion_bound"]
        assert recorded["temperature_K"] == result["temperature_K"]
        assert recorded["fO2"] == result["fO2"]
        for field in (
            "max_ratio",
            "neutral_pressure_bar",
            "bound_pressure_bar",
            "K_ion",
        ):
            assert recorded[field] == pytest.approx(result[field], rel=1.0e-12)
        assert ELEMENT_STATUS[element]["criteria"]["C3"] is (
            result["max_ratio"] < 1.0e-4
        )
    assert ELEMENT_STATUS["Mg"]["criteria"]["C3"] is True
    assert ELEMENT_STATUS["Fe"]["status"] == "complete"
    assert ELEMENT_STATUS["Ti"]["status"] == "complete-except-ions"
    ratio_order = sorted(
        measured,
        key=lambda element: measured[element]["ratio_at_3000K"],
        reverse=True,
    )
    first_ionisation_energies = {}
    for element in measured:
        sources = ELEMENT_STATUS[element]["c3_ion_bound"]["source_tables"]
        cation = sources["cation"]
        cation_enthalpy = (
            _source_value(cation, "formation_enthalpy", 0.0)
            if (JANAF_DATA / f"{cation}.yaml").exists()
            else _text_table_value(cation, 5, 0.0)
        )
        neutral_enthalpy = _source_value(
            sources["neutral"], "formation_enthalpy", 0.0
        )
        first_ionisation_energies[element] = (
            cation_enthalpy - neutral_enthalpy
        ) / 96.4853321233
    ionisation_order = sorted(first_ionisation_energies, key=first_ionisation_energies.get)
    assert ionisation_order == [
        "K", "Na", "Al", "Ca", "V", "Cr", "Ti", "Nb", "Mg", "Fe", "Si"
    ]
    assert ratio_order == [
        "Nb", "K", "Ti", "Ca", "V", "Na", "Al", "Cr", "Si", "Mg", "Fe"
    ]
    assert ratio_order != ionisation_order


def test_cation_janaf_records_match_the_pinned_manifest_hashes() -> None:
    for element, table_id in (
        ("Si", "Si-006"), ("Mg", "Mg-006"), ("Fe", "Fe-009"),
        ("Ca", "Ca-007"), ("Al", "Al-006"), ("Ti", "Ti-007"),
        ("Cr", "Cr-006"), ("V", "V-006"), ("Nb", "Nb-006"),
        ("Mn", "Mn-006"), ("Ni", "Ni-006"), ("Co", "Co-006"),
    ):
        record = _record(table_id)
        extraction = record["extraction"]
        c3 = ELEMENT_STATUS[element]["c3_ion_bound"]
        assert record["table"]["table_id"] == table_id
        assert record["table"]["index_entry"]["state"] == "g"
        assert record["table"]["index_entry"]["charge"] == 1
        assert extraction["user_agent"] == "openimcc-janaf-vendor/1.0"
        assert "source_cache_path" not in extraction
        assert extraction["source_sha256"] == c3["upstream_sha256"][table_id]


def _status_is_complete(status: dict) -> str:
    criteria = status["criteria"]
    if all(criteria.values()):
        return "complete"
    if not criteria["C1"] and all(criteria[key] for key in ("C2", "C3", "C4")):
        return "gas-complete-melt-pending"
    if (
        criteria["C1"]
        and criteria["C2"]
        and not criteria["C3"]
        and criteria["C4"]
        and "c3_ion_bound" in status
    ):
        return "complete-except-ions"
    return "gas-partial"


def _one_bar_janaf_source(table_id: str) -> bool:
    standard_state = _record(table_id)["table"]["standard_state_as_published"]
    return "0.1 MPa" in standard_state


def _c4_rows_cover_domain(
    element: str, parent_by_element: dict[str, str], pack
) -> bool:
    provenance_rows = yaml.safe_load(GAS_PROVENANCE.read_text(encoding="utf-8"))["rows"]
    by_key = {
        (entry["table"], entry["species_name"]): entry
        for entry in provenance_rows
    }
    gas_rows = pack.gas_df[pack.gas_df["cation"] == element]
    if gas_rows.empty:
        return False
    for species_name, row in gas_rows.iterrows():
        if row["T_min"] > 1500 or row["T_max"] < 3000:
            return False
        source = by_key.get(("gas", species_name))
        if source is None or source["method"] != "fitted":
            return False
        if source["authority"] == "janaf_fitted":
            if not _one_bar_janaf_source(source["table_id"]):
                return False
        elif source["authority"] == "nasa_glenn_fitted":
            if "R ln 1.01325" not in source.get("note", ""):
                return False
        else:
            return False

    parent = parent_by_element.get(element)
    if parent is None or f"{parent}(l)" not in pack.oxide_df.index:
        return False
    liquid = pack.oxide_df.loc[f"{parent}(l)"]
    if liquid["T_min"] > 1500 or liquid["T_max"] < 3000:
        return False
    source = by_key.get(("condensate", f"{parent}(l)"))
    if source is None or not (
        source["T_range_K"][0] <= 1500 and source["T_range_K"][1] >= 3000
    ):
        return False
    if source["authority"] == "janaf_fitted":
        return source["method"] == "fitted" and _one_bar_janaf_source(
            source["table_id"]
        )
    return source["authority"] in {
        "janaf_transcribed",
        "lam1984_transcribed",
        "lam1987_transcribed",
        "secondary_transcription_unverified_primary",
    } and source["method"] in {
        "transcribed",
        "transcribed_secondary_unverified_primary",
    }


def test_status_criteria_and_melt_basis_are_consistent() -> None:
    expected_elements = {
        "O", "Si", "Mg", "Fe", "Ca", "Al", "Ti", "Na", "K",
        "Cr", "V", "Nb", "Mn", "Ni", "Co",
    }
    assert set(ELEMENT_STATUS) == expected_elements
    parent_by_element = {
        "Si": "SiO2",
        "Mg": "MgO",
        "Fe": "FeO",
        "Ca": "CaO",
        "Al": "Al2O3",
        "Ti": "TiO2",
        "Na": "Na2O",
        "K": "K2O",
    }
    liquid_by_element = {
        **parent_by_element,
        "Cr": "Cr2O3",
        "V": "V2O3",
        "Nb": "NbO2",
    }
    melt = evaluate_imcc(README_BASALT, 1800.0, basis_type="wt")
    parent_oxides = set(melt.parent_oxides)
    gas_pack = load_gas_datapack()
    measured_ions = _ion_bound_maxima()
    source_candidates = _neutral_oxide_sources()
    recorded_candidates = {
        table_id: (element, species)
        for element, status in ELEMENT_STATUS.items()
        for table_id, species in status["c2_candidates"]
    }
    assert source_candidates == recorded_candidates
    assert not any("+" in species for species in IMCC_GAS_CHANNEL_SPECIES)

    for element, status in ELEMENT_STATUS.items():
        assert set(status["criteria"]) == {"C1", "C2", "C3", "C4"}
        parent = parent_by_element.get(element)
        c1 = (
            parent is not None
            and parent in parent_oxides
            and math.isfinite(melt.activity(parent))
        )
        assert status["criteria"]["C1"] is c1
        c2 = all(
            species in IMCC_GAS_CHANNEL_SPECIES
            or (
                species in status.get("c2_screened_out", {})
                and status["c2_screened_out"][species]["max_ratio"] < 1.0e-4
            )
            for _table_id, species in status["c2_candidates"]
        )
        assert status["criteria"]["C2"] is c2
        ion_bound = measured_ions.get(element)
        c3 = ion_bound is not None and ion_bound["max_ratio"] < 1.0e-4
        if element in {"Mn", "Ni", "Co"}:
            assert status["c3_ion_bound"]["status"] == "not computed"
            assert status["c3_ion_bound"]["reason"] == (
                "not computed: p(E) needs a parent liquid row that is only available "
                "from an external private pack"
            )
        if element == "O":
            assert status["status"] == "input (fO2 pinned)"
            assert status["criteria"]["C1"] is False
            assert status["criteria"]["C3"] is False
            assert status["criteria"]["C4"] is False
        else:
            assert status["criteria"]["C3"] is c3
        c4 = _c4_rows_cover_domain(element, liquid_by_element, gas_pack)
        assert status["criteria"]["C4"] is c4
        assert status["validation"] in {"validated", "unvalidated"}
        assert status["status"] in {
            "input (fO2 pinned)",
            "complete",
            "complete-except-ions",
            "gas-complete-melt-pending",
            "gas-partial",
        }
        assert status["reason"] and "\n" not in status["reason"]
        if element != "O":
            assert status["status"] == _status_is_complete(status), element


def test_roadmap_status_table_matches_element_status() -> None:
    roadmap = (ROOT / "docs" / "ROADMAP.md").read_text(encoding="utf-8")
    rows = {}
    in_table = False
    for line in roadmap.splitlines():
        if line == "| Element | Status | C1 | C2 | C3 | C4 | Validation | Reason |":
            in_table = True
            continue
        if in_table and line.startswith("| ---"):
            continue
        if in_table and line.startswith("| "):
            values = [part.strip() for part in line.strip("|").split("|")]
            if len(values) == 8:
                rows[values[0]] = values[1:]
        elif in_table:
            break
    assert set(rows) == set(ELEMENT_STATUS)
    for element, status in ELEMENT_STATUS.items():
        values = rows[element]
        c3 = status.get("c3_ion_bound", {})
        criteria = [
            "n/a" if element == "O" and criterion in {"C1", "C3", "C4"}
            else "yes" if status["criteria"][criterion]
            else "no"
            for criterion in ("C1", "C2", "C3", "C4")
        ]
        if c3.get("status") == "not computed":
            criteria[2] = c3["reason"]
        assert values == [
            status["status"],
            *criteria,
            status["validation"],
            status["reason"],
        ]
