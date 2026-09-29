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
        if state != "g" or charge != 0:
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


def _evaluated_public_c2_species(
    source_candidates: dict[str, tuple[str, str]],
) -> set[str]:
    pack = load_gas_datapack()
    evaluated = set()
    for element, species in source_candidates.values():
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


def _gas_element_atom_count(species: str, element: str) -> int:
    return sum(
        int(count) if count else 1
        for symbol, count in re.findall(r"([A-Z][a-z]?)(\d*)", species)
        if symbol == element
    )


@cache
def _ion_bound_maxima() -> dict[str, dict[str, object]]:
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
    gas_pressures = {}
    for temperature in TEMPERATURES_K:
        for fugacity in FUGACITIES:
            gas_pressures[(temperature, fugacity)] = evaluate_gas(
                activities_by_temperature[temperature],
                temperature,
                fugacity,
                pack,
                allow_extrapolation=True,
            )
    ion_constants_by_element = {}
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
        ion_constants_by_element[element] = ion_constants

    maxima_by_point = {}
    for temperature in TEMPERATURES_K:
        for fugacity in FUGACITIES:
            pressures = gas_pressures[(temperature, fugacity)]
            source_terms = {
                element: ion_constants_by_element[element][temperature]
                * pressures[element]
                for element in computed
            }
            # Premise: each listed E(g) is a neutral reservoir at its modeled
            # pressure, and every included E+ is singly charged.
            # Algebra: Saha gives p(E+) p(e-) = K_E p(E). Electroneutrality
            # gives p(e-) = sum_E p(E+), hence p(e-)^2 = sum_E K_E p(E) and
            # p(E+) = K_E p(E) / p(e-). The reported fraction divides this by
            # the total neutral-gas inventory, sum_channels n_E p(channel).
            # Units: pressures are reduced by p°=1 bar, so all pressures and
            # K_E p(E) terms in this calculation are dimensionless.
            # Melt-buffered reading treats the fixed neutral pressures as the
            # equilibrium. Relative to closed-parcel depletion, this model
            # overestimates p(e-), so p(E+) is high for donor Na/K but low for
            # trace Ca+. The neutral-model p_E,total stays fixed, making the
            # reported ratio slightly high for K and low for Ca; Ca remains
            # above 1e-4 and the largest passing value remains Cr at 9.7e-6.
            # Adding p(E+) to the neutrals-plus-ion denominator multiplies the
            # reported share by 1/(1+f), where f=p(E+)/p_E,total.
            # Sanity check: at 3000 K, K and Na dominate the summed source terms.
            electron_pressure = math.sqrt(sum(source_terms.values()))
            maxima_by_point[(temperature, fugacity)] = {
                "electron_pressure_bar": electron_pressure,
                "electron_source_terms": source_terms,
            }

    for element, source in computed.items():
        oxide = source["parent_oxide"]
        ion_constants = ion_constants_by_element[element]
        maximum = None
        isolated_bound = 0.0
        ratio_at_3000 = None
        for temperature in TEMPERATURES_K:
            for fugacity in FUGACITIES:
                pressures = gas_pressures[(temperature, fugacity)]
                neutral = pressures[element]
                k_ion = ion_constants[temperature]
                point = maxima_by_point[(temperature, fugacity)]
                ion_pressure = k_ion * neutral / point["electron_pressure_bar"]
                element_total = sum(
                    _gas_element_atom_count(species, element) * pressure
                    for species, pressure in pressures.items()
                )
                ratio = ion_pressure / element_total
                isolated_bound = max(isolated_bound, math.sqrt(k_ion / neutral))
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
                        "element_total_pressure_bar": element_total,
                        "joint_ion_pressure_bar": ion_pressure,
                        "electron_pressure_bar": point["electron_pressure_bar"],
                        "electron_source_terms": point["electron_source_terms"],
                        "K_ion": k_ion,
                        "parent_oxide": oxide,
                    }
        assert maximum is not None
        maximum["ratio_at_3000K"] = ratio_at_3000
        maximum["isolated_bound"] = isolated_bound
        maxima[element] = maximum

    terms_at_3000 = maxima_by_point[(3000.0, 1.0e-4)]["electron_source_terms"]
    for maximum in maxima.values():
        maximum["electron_source_terms_3000K"] = terms_at_3000
    return maxima


def test_joint_thermal_ionisation_estimates_match_the_status_source() -> None:
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
    assert ELEMENT_STATUS["Ti"]["status"] == "complete"
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
        "K", "Na", "Ca", "Al", "Cr", "V", "Ti", "Mg", "Fe", "Nb", "Si"
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
        (
            entry["table"],
            entry["species_name"],
            tuple(entry.get("T_range_K", ())),
        ): entry
        for entry in provenance_rows
    }
    gas_rows = pack.gas_df[pack.gas_df["cation"] == element]
    if gas_rows.empty:
        return False
    for species_name in gas_rows.index.unique():
        species_rows = gas_rows.loc[gas_rows.index == species_name]
        covering_rows = species_rows.loc[
            (species_rows["T_min"] <= 1500) & (species_rows["T_max"] >= 3000)
        ]
        if len(covering_rows) != 1:
            return False
        row = covering_rows.iloc[0]
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
    source = by_key.get(
        (
            "condensate",
            f"{parent}(l)",
            (int(liquid["T_min"]), int(liquid["T_max"])),
        )
    )
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
