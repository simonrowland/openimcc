"""Source closure, screening, and status checks for each supported element."""

from __future__ import annotations

import hashlib
import json
import math
import re
from functools import cache
from pathlib import Path

import pytest
import yaml

from openimcc import evaluate as evaluate_imcc
from openimcc.gas import (
    ELEMENT_STATUS,
    IMCC_GAS_CHANNEL_SPECIES,
    R_J_MOL_K,
    ImccGasSpeciesNotFoundError,
    evaluate_gas,
    load_gas_datapack,
)
from tools import build_gas_tables


ROOT = Path(__file__).resolve().parents[1]
JANAF_DATA = ROOT / "data-src" / "janaf"
NASA_DATA = ROOT / "data-src" / "nasa-glenn"
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
TEMPERATURES_K = tuple(float(T) for T in range(1200, 3001, 100))
MODELED_ION_ELEMENTS = frozenset({"Ca", "Na", "K"})
SCREEN_SPECIES = {
    "Cr": ("Cr", "CrO", "CrO2", "CrO3"),
    "V": ("V", "VO", "VO2"),
    "Nb": ("Nb", "NbO", "NbO2"),
}


def _record(table_id: str, source_dir: Path = JANAF_DATA) -> dict:
    text = (source_dir / f"{table_id}.yaml").read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return yaml.safe_load(text)


def _neutral_oxide_sources(
    source_dir: Path = JANAF_DATA,
) -> dict[str, tuple[str, str]]:
    """Return source-table candidates: table ID -> (element, neutral gas)."""
    result = {}
    paths = sorted(source_dir.glob("*.yaml")) + sorted(source_dir.glob("*.txt"))
    for path in paths:
        if path.suffix == ".yaml":
            entry = _record(path.stem, source_dir)["table"]["index_entry"]
            formula = entry.get("formula_normalised", entry.get("formula"))
            formula = {
                "O6P4": "P4O6",
                "O10P4": "P4O10",
                "O1S2": "SSO",
            }.get(
                formula, formula
            )
            state = entry.get("state")
            charge = entry.get("charge", 0)
        else:
            fields = path.read_text(encoding="utf-8").splitlines()[0].split("\t")
            if len(fields) < 2:
                continue
            match = re.fullmatch(r"(?P<formula>.+)\((?P<state>[^()]*)\)", fields[1])
            if match is None:
                continue
            formula = match["formula"]
            state = match["state"]
            charge = 1 if re.search(r"\d*[+-]$", formula) else 0
        formula = {
            "Li1O1": "LiO",
            "Li2O1": "Li2O",
            "O1Pb1": "PbO",
            "B1O1": "BO",
            "B1O2": "BO2",
            "B2O1": "B2O",
            "Cs1O1": "CsO",
            "Cs2O1": "Cs2O",
            "Cu1O1": "CuO",
            "Sn1O1": "SnO",
            "Sn1O2": "SnO2",
        }.get(formula, formula)
        if state != "g" or charge != 0:
            continue
        if "O" not in formula:
            continue
        for element in sorted(ELEMENT_STATUS, key=len, reverse=True):
            if element == "O":
                continue
            if re.search(rf"(?<![A-Za-z]){element}(?:[0-9]*)", formula):
                result[path.stem] = (element, formula)
                break
    if source_dir == JANAF_DATA:
        for species_name, table_id, formula, cation, *_ in (
            build_gas_tables.TRACE_NASA_GAS_SOURCES
        ):
            if "O" in formula:
                result[table_id] = (cation, species_name.removesuffix("(g)"))
            if "O" not in species_name.removesuffix("(g)"):
                continue
            result[table_id] = (cation, species_name.removesuffix("(g)"))
    return result


def _evaluated_public_c2_species(
    source_candidates: dict[str, tuple[str, str]],
) -> set[str]:
    pack = load_gas_datapack()
    evaluated = set()
    for element, species in source_candidates.values():
        if element in {"P", "S"} and f"{species}(g)" in pack.gas_df.index:
            # Their C2 evidence is the public gas row; callers supply P2O5/S2
            # activities, and the P2O5 parent standard is tracked separately.
            evaluated.add(species)
            continue
        ion_bound = ELEMENT_STATUS[element].get("c3_ion_bound")
        if ion_bound is None:
            continue
        parent_oxide = ion_bound.get("parent_oxide")
        if parent_oxide is None:
            continue
        try:
            pressures = evaluate_gas(
                {parent_oxide: 1.0},
                2200.0,
                1.0e-10,
                pack,
                parent_oxides=(parent_oxide,),
                gas_species=(species,),
            )
        except ImccGasSpeciesNotFoundError:
            continue
        if species in pressures:
            evaluated.add(species)
    return evaluated


def _assert_c2_candidate_coverage(
    source_candidates: dict[str, tuple[str, str]],
    statuses: dict[str, dict],
    evaluated_species: set[str],
) -> None:
    recorded_candidates = {
        table_id: (element, species)
        for element, status in statuses.items()
        for table_id, species in status["c2_candidates"]
    }
    assert source_candidates == recorded_candidates
    for element, status in statuses.items():
        for _table_id, species in status["c2_candidates"]:
            if species in evaluated_species:
                continue
            screen = status.get("c2_screened_out", {}).get(species)
            assert screen is not None, (
                f"{element} {species} is neither included nor screened"
            )
            assert math.isfinite(screen["max_ratio"])
            assert screen["max_ratio"] < 1.0e-4
            assert screen["temperature_K"] in TEMPERATURES_K
            assert screen["fO2"] in FUGACITIES


def test_janaf_neutral_oxide_sources_are_included_or_measured_screens() -> None:
    """Use data-src plus checked-in candidates; no JANAF manifest exists."""
    source_candidates = _neutral_oxide_sources()
    assert all(
        isinstance(status["criteria"]["C2"], bool)
        for status in ELEMENT_STATUS.values()
    )
    _assert_c2_candidate_coverage(
        source_candidates,
        ELEMENT_STATUS,
        _evaluated_public_c2_species(source_candidates),
    )


def test_text_neutral_oxide_source_is_scanned(tmp_path: Path) -> None:
    (tmp_path / "SiO3.txt").write_text(
        "Silicon trioxide\tSiO3(g)\n", encoding="utf-8"
    )
    assert _neutral_oxide_sources(tmp_path) == {"SiO3": ("Si", "SiO3")}


def test_unscreened_fixture_candidate_is_rejected() -> None:
    fixture_dir = ROOT / "tests" / "fixtures" / "janaf-c2"
    source_candidates = _neutral_oxide_sources(fixture_dir)
    statuses = {
        "Mn": {"c2_candidates": (("MnO", "MnO"),), "c2_screened_out": {}}
    }
    assert source_candidates == {"MnO": ("Mn", "MnO")}
    assert "MnO" in IMCC_GAS_CHANNEL_SPECIES
    with pytest.raises(ImccGasSpeciesNotFoundError):
        evaluate_gas(
            {"MnO": 1.0},
            2200.0,
            1.0e-10,
            load_gas_datapack(),
            parent_oxides=("MnO",),
            gas_species=("MnO",),
        )
    with pytest.raises(AssertionError, match="MnO is neither included nor screened"):
        _assert_c2_candidate_coverage(
            source_candidates,
            statuses,
            _evaluated_public_c2_species(source_candidates),
        )


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


@cache
def _source_value(table_id: str, field: str, temperature: float) -> float:
    nasa_path = NASA_DATA / f"{table_id}.json"
    if nasa_path.is_file():
        record = json.loads(nasa_path.read_text(encoding="utf-8"))
        if field == "formation_enthalpy":
            return float(record["delta_f_H_298_15"]["value"]) / 1000.0
        if field == "formation_gibbs_energy":
            properties = build_gas_tables._nasa7_properties(record, temperature)
            return (
                R_J_MOL_K
                * temperature
                * (properties["h_rt"] - properties["s_R"])
                / 1000.0
            )
        raise AssertionError(f"unsupported NASA source field {field}")
    text_path = JANAF_DATA / f"{table_id}.txt"
    if text_path.is_file():
        columns = {
            "temperature": 0,
            "heat_capacity": 1,
            "entropy": 2,
            "enthalpy_increment": 4,
            "formation_enthalpy": 5,
            "formation_gibbs_energy": 6,
        }
        return _text_table_value(table_id, columns[field], temperature)
    for row in _record(table_id)["table"]["values"]:
        if row["temperature"]["value"] == temperature:
            value = row[field]["value"]
            assert value is not None, f"{table_id} has no {field} at {temperature} K"
            return float(value)
    raise AssertionError(f"{table_id} has no JANAF row at {temperature} K")


@cache
def _neutral_gibbs_value(table_id: str, temperature: float) -> float:
    if (NASA_DATA / f"{table_id}.json").is_file():
        return _source_value(table_id, "formation_gibbs_energy", temperature)
    if (JANAF_DATA / f"{table_id}.txt").is_file():
        reference = _source_value(table_id, "formation_enthalpy", 298.15)
        enthalpy_increment = _source_value(
            table_id, "enthalpy_increment", temperature
        )
        entropy = _source_value(table_id, "entropy", temperature)
        # Match the fitted runtime row's 298 K anchor construction, including
        # after JANAF's printed formation columns change at a reference-phase
        # transition.
        return reference + enthalpy_increment - temperature * entropy / 1000.0
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


@cache
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


def _gas_element_atom_count(species: str, element: str) -> int:
    return sum(
        int(count) if count else 1
        for symbol, count in re.findall(r"([A-Z][a-z]?)(\d*)", species)
        if symbol == element
    )


@cache
def _ion_bound_maxima() -> dict[str, dict[str, object]]:
    pack = load_gas_datapack()
    computed = {
        element: status["c3_ion_bound"]
        for element, status in ELEMENT_STATUS.items()
        if status.get("c3_ion_bound", {}).get("status") != "not computed"
        and "c3_ion_bound" in status
    }
    assert set(computed) == {
        "Si", "Mg", "Fe", "Ca", "Al", "Ti", "Cr", "V", "Nb", "Na", "K",
        "Li", "Rb", "Pb", "Ga", "Ge", "B", "In",
        "Li", "Rb", "Pb", "Cs", "Cu", "Sn",
    }
    caller_parent_oxides = {
        "Cr": "Cr2O3", "V": "V2O3", "Nb": "NbO2",
        "Li": "Li2O", "Rb": "Rb2O", "Pb": "PbO",
        "Cs": "Cs2O", "Cu": "Cu2O", "Sn": "SnO",
    }
    trace_elements = {"Li", "Rb", "Pb"}
    new_trace_elements = {"Ga", "Ge", "B", "In"}
    new_trace_parents = {
        "Ga": "Ga2O3", "Ge": "GeO2", "B": "B2O3", "In": "In2O3",
    }
    trace_elements = {"Li", "Rb", "Pb", "Cs", "Cu", "Sn"}
    existing_caller_parent_oxides = {
        element: oxide
        for element, oxide in caller_parent_oxides.items()
        if element not in trace_elements
    }
    activities_by_temperature = {}
    trace_activities_by_temperature = {}
    for temperature in TEMPERATURES_K:
        melt = evaluate_imcc(
            README_BASALT,
            temperature,
            basis_type="wt",
            allow_extrapolation=True,
        )
        activities = {name: melt.activity(name) for name in melt.parent_oxides}
        activities.update(
            {oxide: 1.0e-3 for oxide in existing_caller_parent_oxides.values()}
        )
        activities_by_temperature[temperature] = activities
        # Preserve the established Li/Rb/Pb joint screen. Measure each new
        # parent independently so adding one element cannot perturb another
        # element's status through the shared charge-balance closure.
        legacy_trace_activities = dict(activities)
        legacy_trace_activities.update(
            {caller_parent_oxides[element]: 1.0e-3 for element in ("Li", "Rb", "Pb")}
        )
        trace_activities_by_temperature[temperature] = {
            "legacy": legacy_trace_activities,
            **{
                element: {
                    **activities,
                    caller_parent_oxides[element]: 1.0e-3,
                }
                for element in ("Cs", "Cu", "Sn")
            },
        }

    unmodeled_constants = {}
    for element, source in computed.items():
        if element in MODELED_ION_ELEMENTS | new_trace_elements:
            continue
        cation_table = source["source_tables"]["cation"]
        constants = {}
        for temperature in TEMPERATURES_K:
            delta_g_kj = (
                (
                    _text_table_value(cation_table, 6, temperature)
                    if cation_table in {"Na-006", "K-006"}
                    else _source_value(
                        cation_table, "formation_gibbs_energy", temperature
                    )
                )
                + _text_table_value("D-020", 6, temperature)
                - _neutral_gibbs_value(
                    source["source_tables"]["neutral"], temperature
                )
            )
            constants[temperature] = math.exp(
                -delta_g_kj * 1000.0 / (R_J_MOL_K * temperature)
            )
        unmodeled_constants[element] = constants

    gas_pressures = {}
    trace_gas_pressures = {}
    for temperature in TEMPERATURES_K:
        for fugacity in FUGACITIES:
            gas_pressures[(temperature, fugacity)] = evaluate_gas(
                activities_by_temperature[temperature],
                temperature,
                fugacity,
                pack,
                allow_extrapolation=True,
                include_ions=True,
            )
            for trace_case, trace_activities in trace_activities_by_temperature[
                temperature
            ].items():
                trace_gas_pressures[(temperature, fugacity, trace_case)] = evaluate_gas(
                    trace_activities,
                    temperature,
                    fugacity,
                    pack,
                    allow_extrapolation=True,
                    include_ions=True,
                )

    maxima = {}
    maxima_by_point = {}
    negative_screen_species = ("O2", "AlO2", "KO", "Cr", "V", "Nb")
    attachment_term_max = {
        neutral: 0.0
        for neutral in negative_screen_species
    }
    negative_to_neutral_ratio_max = {
        neutral: 0.0 for neutral in negative_screen_species
    }
    attachment_term_location = {}
    negative_to_neutral_ratio_location = {}
    for temperature in TEMPERATURES_K:
        for fugacity in FUGACITIES:
            pressures = gas_pressures[(temperature, fugacity)]
            neutral_pressures = {
                species: pressure
                for species, pressure in pressures.items()
                if not species.endswith(("+", "-"))
            }
            electron_pressure = pressures["e-"]
            # Premise: A + e- = A- has K_A = exp[-(G(A-) - G(A) - G(e-))/RT],
            # and its charge-balance term is K_A*p(A) = p(A-)/p(e-).
            # This fitted-output screen covers all temperatures and fO2 nodes
            # used by C3. Since p(A-) = K_A*p(A)*p(e-), read both the
            # charge-balance term and pressure ratio directly from the closure.
            for neutral in negative_screen_species:
                anion_pressure = pressures[neutral + "-"]
                term = anion_pressure / electron_pressure
                ion_ratio = anion_pressure / pressures[neutral]
                if term > attachment_term_max[neutral]:
                    attachment_term_max[neutral] = term
                    attachment_term_location[neutral] = (temperature, fugacity)
                if ion_ratio > negative_to_neutral_ratio_max[neutral]:
                    negative_to_neutral_ratio_max[neutral] = ion_ratio
                    negative_to_neutral_ratio_location[neutral] = (
                        temperature,
                        fugacity,
                    )
            ion_constants = {}
            for element in computed:
                if element in trace_elements | new_trace_elements:
                    continue
                if element in MODELED_ION_ELEMENTS:
                    ion_constants[element] = (
                        pressures[element + "+"] * electron_pressure
                        / neutral_pressures[element]
                    )
                else:
                    ion_constants[element] = unmodeled_constants[element][temperature]
            source_terms = {
                element: ion_constants[element] * neutral_pressures[element]
                for element in ion_constants
            }
            maxima_by_point[(temperature, fugacity)] = {
                "electron_pressure_bar": electron_pressure,
                "electron_source_terms": source_terms,
            }

            # C3 compares channel shares only in the documented <=1-bar
            # neutral-pressure validity domain.
            if sum(neutral_pressures.values()) > 1.0:
                continue

            for element in computed:
                if element in trace_elements | new_trace_elements:
                    continue
                source = computed[element]
                neutral = neutral_pressures[element]
                k_ion = ion_constants[element]
                ion_pressure = (
                    pressures[element + "+"]
                    if element in MODELED_ION_ELEMENTS
                    else k_ion * neutral / electron_pressure
                    if electron_pressure > 0.0
                    else 0.0
                )
                element_total = sum(
                    _gas_element_atom_count(species, element) * pressure
                    for species, pressure in neutral_pressures.items()
                )
                ratio = ion_pressure / element_total if element_total else 0.0
                current = maxima.get(element)
                isolated_bound = math.sqrt(k_ion / neutral) if neutral > 0.0 else 0.0
                if current is None:
                    current = {
                        "max_ratio": -1.0,
                        "isolated_bound": 0.0,
                    }
                    maxima[element] = current
                current["isolated_bound"] = max(
                    current["isolated_bound"], isolated_bound
                )
                if ratio > current["max_ratio"]:
                    current.update(
                        {
                            "max_ratio": ratio,
                            "temperature_K": temperature,
                            "fO2": fugacity,
                            "neutral_pressure_bar": neutral,
                            "element_total_pressure_bar": element_total,
                            "joint_ion_pressure_bar": ion_pressure,
                            "electron_pressure_bar": electron_pressure,
                            "electron_source_terms": source_terms,
                            "K_ion": k_ion,
                            "parent_oxide": source["parent_oxide"],
                        }
                    )

            # Trace parents are screened in their own caller-supplied
            # 1e-3-activity case so their ions do not perturb the established
            # basalt/Cr/V/Nb C3 and negative-ion screens.
            for element in trace_elements:
                trace_case = "legacy" if element in {"Li", "Rb", "Pb"} else element
                trace_pressures = trace_gas_pressures[
                    (temperature, fugacity, trace_case)
                ]
                trace_neutrals = {
                    species: pressure
                    for species, pressure in trace_pressures.items()
                    if not species.endswith(("+", "-"))
                }
                trace_electron = trace_pressures["e-"]
                source = computed[element]
                neutral = trace_neutrals[element]
                k_ion = unmodeled_constants[element][temperature]
                ion_pressure = sum(
                    _gas_element_atom_count(species, element) * pressure
                    for species, pressure in trace_pressures.items()
                    if species.endswith("+")
                    and _gas_element_atom_count(species, element)
                )
                element_total = sum(
                    _gas_element_atom_count(species, element) * pressure
                    for species, pressure in trace_neutrals.items()
                )
                if sum(trace_neutrals.values()) > 1.0:
                    continue
                ratio = ion_pressure / element_total if element_total else 0.0
                current = maxima.get(element)
                isolated_bound = math.sqrt(k_ion / neutral) if neutral > 0.0 else 0.0
                if current is None:
                    current = {"max_ratio": -1.0, "isolated_bound": 0.0}
                    maxima[element] = current
                current["isolated_bound"] = max(
                    current["isolated_bound"], isolated_bound
                )
                if ratio > current["max_ratio"]:
                    current.update(
                        {
                            "max_ratio": ratio,
                            "temperature_K": temperature,
                            "fO2": fugacity,
                            "neutral_pressure_bar": neutral,
                            "element_total_pressure_bar": element_total,
                            "joint_ion_pressure_bar": ion_pressure,
                            "electron_pressure_bar": trace_electron,
                            "electron_source_terms": {},
                            "K_ion": k_ion,
                            "parent_oxide": source["parent_oxide"],
                        }
                    )

    new_trace_ions = {
        "Ga": ("Ga+", "Ga-"),
        "Ge": ("Ge+",),
        "B": ("B+", "B-", "BO-", "BO2-"),
        "In": ("In+",),
    }
    for element in new_trace_elements:
        current = {"max_ratio": -1.0, "isolated_bound": 0.0}
        for temperature in TEMPERATURES_K:
            activities = dict(activities_by_temperature[temperature])
            activities[new_trace_parents[element]] = 1.0e-3
            for fugacity in FUGACITIES:
                pressures = evaluate_gas(
                    activities,
                    temperature,
                    fugacity,
                    pack,
                    allow_extrapolation=True,
                    include_ions=True,
                )
                neutral_pressures = {
                    species: pressure
                    for species, pressure in pressures.items()
                    if not species.endswith(("+", "-"))
                }
                if sum(neutral_pressures.values()) > 1.0:
                    continue
                neutral = neutral_pressures.get(element, 0.0)
                element_total = sum(
                    _gas_element_atom_count(species, element) * pressure
                    for species, pressure in neutral_pressures.items()
                )
                ion_pressure = sum(
                    _gas_element_atom_count(ion.rstrip("+-"), element)
                    * pressures.get(ion, 0.0)
                    for ion in new_trace_ions[element]
                )
                electron_pressure = pressures["e-"]
                cation_pressure = pressures.get(element + "+", 0.0)
                k_ion = (
                    cation_pressure * electron_pressure / neutral
                    if neutral > 0.0
                    else 0.0
                )
                isolated_bound = (
                    math.sqrt(k_ion / neutral) if neutral > 0.0 else 0.0
                )
                current["isolated_bound"] = max(
                    current["isolated_bound"], isolated_bound
                )
                ratio = ion_pressure / element_total if element_total else 0.0
                if ratio > current["max_ratio"]:
                    current.update(
                        {
                            "max_ratio": ratio,
                            "temperature_K": temperature,
                            "fO2": fugacity,
                            "neutral_pressure_bar": neutral,
                            "element_total_pressure_bar": element_total,
                            "joint_ion_pressure_bar": ion_pressure,
                            "electron_pressure_bar": electron_pressure,
                            "electron_source_terms": {},
                            "K_ion": k_ion,
                            "parent_oxide": new_trace_parents[element],
                        }
                    )
        maxima[element] = current

    assert attachment_term_max["O2"] > 1.0e-4
    assert negative_to_neutral_ratio_max["AlO2"] > 1.0e-4
    assert negative_to_neutral_ratio_max["KO"] > 1.0e-4
    for neutral in ("Cr", "V", "Nb"):
        assert attachment_term_max[neutral] > 1.0e-4
    negative_screen = yaml.safe_load(
        GAS_PROVENANCE.read_text(encoding="utf-8")
    )["source_notes"]["negative_ion_screen"]
    for neutral in negative_screen_species:
        source_name = f"{neutral}-"
        recorded_term = negative_screen["maximum_attachment_terms"][source_name]
        assert attachment_term_max[neutral] == pytest.approx(
            recorded_term["value"], rel=1.0e-12
        )
        assert attachment_term_location[neutral] == (
            float(recorded_term["temperature_K"]),
            float(recorded_term["fO2_bar"]),
        )
        recorded_ratio = negative_screen[
            "maximum_ion_to_neutral_pressure_ratios"
        ][source_name]
        assert negative_to_neutral_ratio_max[neutral] == pytest.approx(
            recorded_ratio["value"], rel=1.0e-12
        )
        assert negative_to_neutral_ratio_location[neutral] == (
            float(recorded_ratio["temperature_K"]),
            float(recorded_ratio["fO2_bar"]),
        )
    source_terms_3000 = maxima_by_point[
        (3000.0, 1.0e-4)
    ]["electron_source_terms"]
    for maximum in maxima.values():
        maximum["electron_source_terms_3000K"] = source_terms_3000
    return maxima


def test_joint_thermal_ionisation_estimates_match_the_status_source() -> None:
    measured = _ion_bound_maxima()
    assert {path.stem for path in JANAF_DATA.glob("*.txt")} == {
        "Na-006", "K-006", "D-020",
        "Na-007", "K-007", "O-003", "Al-007", "Fe-010",
        "Si-007", "Ti-008", "Al-076", "Na-009",
        "O-031", "Al-078", "K-009", "Li-007", "Li-012", "Rb-007", "Pb-007",
        "Cs-007", "Cu-007",
        "Cs-005", "Cs-006", "Cs-017", "Cs-021",
        "Cu-005", "Cu-006", "Cu-016", "Cu-018", "Cu-020",
        "Cr-007", "V-007", "Nb-007",
        "Li-005", "Li-006", "Li-011", "Li-015", "Li-017", "Li-019",
        "O-007", "O-009", "Pb-005", "Pb-006", "Rb-005", "Rb-006",
        "B-005", "B-006", "B-007", "B-078", "B-079", "B-080",
        "B-093", "B-094", "B-096", "B-098", "Ga-005", "Ga-006", "Ga-007",
    }
    assert set(measured) == {
        "Si", "Mg", "Fe", "Ca", "Al", "Ti", "Cr", "V", "Nb", "Na", "K",
        "Li", "Rb", "Pb", "Cs", "Cu", "Sn", "Ga", "Ge", "B", "In",
    }
    for element, result in measured.items():
        recorded = ELEMENT_STATUS[element]["c3_ion_bound"]
        if element in {"Li", "Rb", "Pb", "Cs", "Cu", "Sn", "Ga", "Ge", "B", "In"}:
            assert recorded["parent_activity"] == 1.0e-3
        assert recorded["temperature_K"] == result["temperature_K"]
        assert recorded["fO2"] == result["fO2"]
        for field in (
            "max_ratio",
            "neutral_pressure_bar",
            "element_total_pressure_bar",
            "joint_ion_pressure_bar",
            "electron_pressure_bar",
            "K_ion",
        ):
            assert recorded[field] == pytest.approx(result[field], rel=1.0e-12)
        assert recorded["isolated_bound"] == pytest.approx(
            result["isolated_bound"], rel=1.0e-12
        )
        assert ELEMENT_STATUS[element]["criteria"]["C3"] is (
            result["max_ratio"] < 1.0e-4
        )
    source_terms_3000 = measured["Si"]["electron_source_terms_3000K"]
    assert set(
        sorted(source_terms_3000, key=source_terms_3000.get, reverse=True)[:2]
    ) == {"K", "Na"}
    assert source_terms_3000["K"] + source_terms_3000["Na"] > sum(
        source_terms_3000.values()
    ) / 2
    assert ELEMENT_STATUS["Mg"]["criteria"]["C3"] is True
    assert ELEMENT_STATUS["Fe"]["status"] == "complete"
    legacy_measured = {
        element: result
        for element, result in measured.items()
        if element not in {"Ga", "Ge", "B", "In"}
    }
    ratio_order = sorted(
        legacy_measured,
        key=lambda element: legacy_measured[element]["max_ratio"],
        reverse=True,
    )
    first_ionisation_energies = {}
    for element in legacy_measured:
        sources = ELEMENT_STATUS[element]["c3_ion_bound"]["source_tables"]
        cation = sources["cation"]
        cation_enthalpy = _source_value(cation, "formation_enthalpy", 0.0)
        neutral_enthalpy = _source_value(
            sources["neutral"], "formation_enthalpy", 0.0
        )
        first_ionisation_energies[element] = (
            cation_enthalpy - neutral_enthalpy
        ) / 96.4853321233
    ionisation_order = sorted(first_ionisation_energies, key=first_ionisation_energies.get)
    assert ionisation_order == [
        "Cs", "Rb", "K", "Na", "Li", "Al", "Ca", "V", "Cr", "Ti",
        "Nb", "Sn", "Pb", "Mg", "Cu", "Fe", "Si",
    ]
    assert ratio_order == [
        "K", "Na", "Cs", "Ca", "Rb", "Cu", "Al", "Cr", "Mg", "Li",
        "Sn", "Fe", "V", "Ti", "Nb", "Pb", "Si",
    ]
    assert ratio_order != ionisation_order


def test_cation_janaf_records_match_the_pinned_manifest_hashes() -> None:
    for element, table_id in (
        ("Si", "Si-006"), ("Mg", "Mg-006"), ("Fe", "Fe-009"),
        ("Ca", "Ca-007"), ("Al", "Al-006"), ("Ti", "Ti-007"),
        ("Cr", "Cr-006"), ("V", "V-006"), ("Nb", "Nb-006"),
        ("Mn", "Mn-006"), ("Ni", "Ni-006"), ("Co", "Co-006"),
        ("Li", "Li-006"), ("Rb", "Rb-006"), ("Pb", "Pb-006"),
        ("Ga", "Ga-006"), ("Ge", "NG-5197"), ("B", "B-006"),
        ("In", "NG-6016"),
        ("Cs", "Cs-006"), ("Cu", "Cu-006"),
    ):
        c3 = ELEMENT_STATUS[element]["c3_ion_bound"]
        text_path = JANAF_DATA / f"{table_id}.txt"
        if text_path.is_file():
            header = text_path.read_text(encoding="utf-8").splitlines()[0]
            formula = header.split("\t", 1)[1]
            assert formula == f"{element}1+(g)"
            digest = hashlib.sha256(text_path.read_bytes()).hexdigest()
            rows = yaml.safe_load(GAS_PROVENANCE.read_text(encoding="utf-8"))["rows"]
            provenance = next(
                row for row in rows if row.get("table_id") == table_id
            )
            assert provenance["source_sha256"] == digest
            assert provenance["user_agent"] == "neutral-source-vendor/1.0"
        elif (NASA_DATA / f"{table_id}.json").is_file():
            source_path = NASA_DATA / f"{table_id}.json"
            record = json.loads(source_path.read_text(encoding="utf-8"))
            assert record["record_id"] == table_id
            assert record["phase"] == "gas"
            assert record["formula"] == element
            assert record["name_as_published"] == f"{element}+"
            digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        else:
            record = _record(table_id)
            extraction = record["extraction"]
            assert record["table"]["table_id"] == table_id
            assert record["table"]["index_entry"]["state"] == "g"
            assert record["table"]["index_entry"]["charge"] == 1
            assert extraction["user_agent"] == "openimcc-janaf-vendor/1.0"
            assert "source_cache_path" not in extraction
            digest = extraction["source_sha256"]
        assert digest == c3["upstream_sha256"][table_id]
        if element in {"Ga", "Ge", "B", "In"}:
            for source_id, expected_sha in c3["upstream_sha256"].items():
                source_path = NASA_DATA / f"{source_id}.json"
                if not source_path.is_file():
                    source_path = JANAF_DATA / f"{source_id}.txt"
                assert source_path.is_file()
                assert hashlib.sha256(source_path.read_bytes()).hexdigest() == (
                    expected_sha
                )


def _status_is_complete(status: dict) -> str:
    criteria = status["criteria"]
    c3_not_computed = status.get("c3_ion_bound", {}).get("status") == "not computed"
    if all(criteria.values()):
        return "complete"
    if (
        not criteria["C1"]
        and criteria["C2"]
        and criteria["C4"]
        and (criteria["C3"] or c3_not_computed)
    ):
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
    yaml_path = JANAF_DATA / f"{table_id}.yaml"
    if yaml_path.is_file():
        standard_state = _record(table_id)["table"]["standard_state_as_published"]
        return "0.1 MPa" in standard_state
    text_path = JANAF_DATA / f"{table_id}.txt"
    if text_path.is_file():
        source_notes = yaml.safe_load(
            GAS_PROVENANCE.read_text(encoding="utf-8")
        )["source_notes"]
        return "0.1 MPa (1 bar)" in source_notes["janaf"]["gas_standard_state"]
    return False


def _c4_rows_cover_domain(
    element: str,
    parent_by_element: dict[str, str],
    pack,
    gas_parent_by_element: dict[str, str] | None = None,
) -> bool:
    domain_start = 1200
    domain_end = 3000
    provenance_rows = yaml.safe_load(GAS_PROVENANCE.read_text(encoding="utf-8"))["rows"]
    by_key = {
        (
            entry["table"],
            entry["species_name"],
            tuple(entry.get("T_range_K", ())),
        ): entry
        for entry in provenance_rows
    }
    gas_rows = pack.gas_df[
        (pack.gas_df["cation"] == element)
        & ~pack.gas_df.index.str.endswith(("+(g)", "-(g)"))
    ]
    if gas_rows.empty:
        return False
    for species_name in gas_rows.index.unique():
        species_rows = gas_rows.loc[gas_rows.index == species_name]
        coverage_end = domain_start
        for _, row in species_rows.sort_values("T_min").iterrows():
            row_start = max(domain_start, int(row["T_min"]))
            row_end = min(domain_end, int(row["T_max"]))
            if row_end < row_start:
                continue
            if row_start > coverage_end:
                return False
            source = by_key.get(
                (
                    "gas",
                    species_name,
                    (int(row["T_min"]), int(row["T_max"])),
                )
            )
            if source is None or source["method"] != "fitted":
                return False
            if source["authority"] == "janaf_fitted":
                if not _one_bar_janaf_source(source["table_id"]):
                    return False
            elif source["authority"] == "nasa_glenn_fitted":
                if "NASA/TP-2002-211556" not in source["source"].get(
                    "citation", ""
                ):
                    return False
            else:
                return False
            coverage_end = max(coverage_end, row_end)
        if coverage_end < domain_end:
            return False

    parent = parent_by_element.get(element)
    gas_parent = (gas_parent_by_element or {}).get(element)
    if parent is None:
        # Gas-parent rows share the gas coverage checks above; S2(g) is both
        # the caller reference and an exposed sulfur product channel.
        return gas_parent is not None and f"{gas_parent}(g)" in pack.gas_df.index
    if f"{parent}(l)" not in pack.oxide_df.index:
        return False
    liquid_rows = pack.oxide_df.loc[pack.oxide_df.index == f"{parent}(l)"]
    coverage_end = domain_start
    for _, liquid in liquid_rows.sort_values("T_min").iterrows():
        row_start = max(domain_start, int(liquid["T_min"]))
        row_end = min(domain_end, int(liquid["T_max"]))
        if row_end < row_start:
            continue
        if row_start > coverage_end:
            return False
        source = by_key.get(
            (
                "condensate",
                f"{parent}(l)",
                (int(liquid["T_min"]), int(liquid["T_max"])),
            )
        )
        if source is None:
            return False
        if source.get("extrapolation") is True:
            if source["source_data"] is not False or source["method"] != (
                "generated_constant_cp_extrapolation_fit"
            ):
                return False
            if source["authority"] == "nasa_glenn_fitted":
                if "NASA pure-liquid standard state at 1 atm" not in source.get(
                    "note", ""
                ):
                    return False
            elif source["authority"] != "janaf_fitted" or not _one_bar_janaf_source(
                source["table_id"]
            ):
                return False
            if source["authority"] == "janaf_fitted":
                if not _one_bar_janaf_source(source["table_id"]):
                    return False
            elif source["authority"] == "nasa_glenn_fitted":
                if (
                    "NASA/TP-2002-211556"
                    not in source["source"].get("citation", "")
                    or "NASA pure-liquid standard state at 1 atm"
                    not in source.get("note", "")
                ):
                    return False
            else:
                return False
        elif source["authority"] == "janaf_fitted":
            if source["method"] not in {
                "fitted",
                "janaf_anchored_nasa_tail_fit",
            } or not _one_bar_janaf_source(source["table_id"]):
                return False
            if source["method"] == "janaf_anchored_nasa_tail_fit":
                tail_sources = {
                    "PbO": "data-src/nasa-glenn/NG-1801.json",
                    "Cu2O": "data-src/nasa-glenn/NG-1844.json",
                }
                if source.get("tail_source_path") != tail_sources.get(parent):
                    return False
        elif source["authority"] == "nasa_glenn_fitted":
            if (
                source["method"] != "fitted"
                or "NASA pure-liquid standard state at 1 atm"
                not in source.get("note", "")
            ):
                return False
        elif source["authority"] not in {
            "janaf_transcribed",
            "lam1984_transcribed",
            "lam1987_transcribed",
            "secondary_transcription_unverified_primary",
        } or source["method"] not in {
            "transcribed",
            "transcribed_secondary_unverified_primary",
        }:
            return False
        coverage_end = max(coverage_end, row_end)
    return coverage_end >= domain_end


def test_status_criteria_and_melt_basis_are_consistent() -> None:
    expected_elements = {
        "O", "Si", "Mg", "Fe", "Ca", "Al", "Ti", "Na", "K",
        "Cr", "V", "Nb", "Li", "Rb", "Pb", "Ga", "Ge", "B", "In",
        "Cr", "V", "Nb", "Li", "Rb", "Pb", "Cs", "Cu", "Sn",
        "Mn", "Ni", "Co", "P", "S",
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
        "Ga": "Ga2O3",
        "Ge": "GeO2",
        "B": "B2O3",
        "In": "In2O3",
    }
    liquid_by_element = {
        **parent_by_element,
        "Cr": "Cr2O3",
        "V": "V2O3",
        "Nb": "NbO2",
        "Li": "Li2O",
        "Rb": "Rb2O",
        "Pb": "PbO",
        "Ga": "Ga2O3",
        "Ge": "GeO2",
        "B": "B2O3",
        "In": "In2O3",
        "Cs": "Cs2O",
        "Cu": "Cu2O",
        "Sn": "SnO",
    }
    gas_parent_by_element = {"S": "S2"}
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
        c4 = _c4_rows_cover_domain(
            element, liquid_by_element, gas_pack, gas_parent_by_element
        )
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
