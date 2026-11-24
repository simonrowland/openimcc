import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from openimcc.subsolidus import (
    R_J_MOL_K,
    STANDARD_PRESSURE_PA,
    evaluate_subsolidus_reactions,
)
from openimcc import subsolidus


@pytest.mark.parametrize("value", [True, np.bool_(True)])
def test_subsolidus_rejects_boolean_temperature(value: object) -> None:
    host = _host("FeS_troilite", "FeS", "troilite")
    with pytest.raises(ValueError, match="temperature"):
        _evaluate(host, value, {"S2": 1.0e-8})


@pytest.mark.parametrize("value", [True, np.bool_(True)])
def test_subsolidus_rejects_boolean_activity(value: object) -> None:
    host = _host("FeS_troilite", "FeS", "troilite", activity=value)
    result = _evaluate(host, 1200.0, {"S2": 1.0e-8})
    assert result["notices"][0]["detail"].startswith("Invalid activity input:")


def test_subsolidus_rejects_boolean_gas_pressure() -> None:
    host = _host("FeS_troilite", "FeS", "troilite")
    with pytest.raises(ValueError, match="partial pressure"):
        _evaluate(host, 800.0, {"S2": True})


def test_subsolidus_rejects_boolean_oxygen_fugacity() -> None:
    host = _host("FeS_troilite", "FeS", "troilite")
    with pytest.raises(ValueError, match="oxygen_fugacity_Pa"):
        evaluate_subsolidus_reactions(
            hosts=[host],
            temperature_K=800.0,
            gas_partial_pressures_Pa={"S2": 1.0e-8},
            condensed_phase_activities={},
            oxygen_fugacity_Pa=True,
        )


def test_subsolidus_refuses_boolean_remaining_inventory() -> None:
    host = _host("FeS_troilite", "FeS", "troilite", remaining_mol=True)
    result = _evaluate(host, 1200.0, {"S2": 1.0e-8})
    assert result["status"] == "unknown_host"
    assert result["notices"][0]["detail"].startswith("Invalid remaining_mol input:")


def _host(species_id, formula, phase, remaining_mol=2.0, activity=1.0):
    return {
        "species_id": species_id,
        "formula": formula,
        "phase": phase,
        "remaining_mol": remaining_mol,
        "activity": activity,
        "source_tag": "test-feed",
    }


def _evaluate(host, temperature, pressures, activities=None, oxygen=None):
    return evaluate_subsolidus_reactions(
        hosts=[host],
        temperature_K=temperature,
        gas_partial_pressures_Pa=pressures,
        condensed_phase_activities=activities or {},
        oxygen_fugacity_Pa=oxygen,
    )[0]


def _janaf_row(table_id: str, temperature_K: float) -> dict[str, float]:
    source_root = Path(__file__).resolve().parents[1] / "data-src" / "janaf"
    text_path = source_root / f"{table_id}.txt"
    if text_path.exists():
        lines = text_path.read_text().splitlines()
        headings = lines[1].split("\t")
        for line in lines[2:]:
            cells = line.split("\t")
            if len(cells) == 6:
                trailing_values = cells[-1].split()
                if len(trailing_values) == 3:
                    cells = cells[:-1] + trailing_values
            if len(cells) != len(headings):
                continue
            try:
                row = dict(zip(headings, cells))
                row_temperature = float(row["T(K)"])
            except (KeyError, ValueError):
                continue
            if row_temperature == temperature_K:
                return {
                    "temperature": row_temperature,
                    "entropy": float(row["S"]),
                    "enthalpy_increment": float(row["H-H(Tr)"]),
                    "formation_enthalpy": float(row["delta-f H"]),
                    "formation_gibbs_energy": float(row["delta-f G"]),
                }
        raise AssertionError(f"JANAF table {table_id} has no {temperature_K:g} K row")

    import yaml

    table = yaml.safe_load((source_root / f"{table_id}.yaml").read_text())
    rows = table.get("values", table.get("table", {}).get("values", ()))
    for row in rows:
        if float(row["temperature"]["value"]) == temperature_K:
            return {
                "temperature": temperature_K,
                "entropy": float(row["entropy"]["value"]),
                "enthalpy_increment": float(row["enthalpy_increment"]["value"]),
                "formation_enthalpy": float(row["formation_enthalpy"]["value"]),
                "formation_gibbs_energy": float(row["formation_gibbs_energy"]["value"]),
            }
    raise AssertionError(f"JANAF table {table_id} has no {temperature_K:g} K row")


def _janaf_absolute_gibbs_J(table_id: str, temperature_K: float) -> float:
    reference_enthalpy = _janaf_row(table_id, 298.15)["formation_enthalpy"]
    row = _janaf_row(table_id, temperature_K)
    return (
        (reference_enthalpy + row["enthalpy_increment"]) * 1000.0
        - temperature_K * row["entropy"]
    )


def _rh_absolute_gibbs_J(phase_id: str, temperature_K: float) -> float:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "data-src" / "robie-hemingway-1995-usgs-b2131" / "crystal-tables.json"
    )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    phase = next(row for row in source["phases"] if row["id"] == phase_id)
    row = next(row for row in phase["rows"] if row["T_K"] == temperature_K)
    enthalpy_298 = phase["rows"][0]["formation_enthalpy_kJ_mol"]
    return (
        enthalpy_298 * 1000.0
        + row["H_minus_H298_over_T_J_mol_K"] * temperature_K
        - temperature_K * row["S_J_mol_K"]
    )


def _raw_subsolidus_record(name: str) -> str:
    path = Path(subsolidus.__file__).with_name("data") / "subsolidus.json"
    text = path.read_text()
    species_start = text.index('"species"')
    record_start = text.index(f'"{name}":', species_start)
    object_start = text.index("{", record_start)
    _, object_end = json.JSONDecoder().raw_decode(text, object_start)
    return text[object_start:object_end]


def test_supported_reactions_are_atom_balanced_and_have_one_driver():
    formulas = {
        "FeS_troilite": {"Fe": 1, "S": 1},
        "Fe": {"Fe": 1},
        "S2": {"S": 2},
        "CaSO4": {"Ca": 1, "S": 1, "O": 4},
        "CaO": {"Ca": 1, "O": 1},
        "MgSO4": {"Mg": 1, "S": 1, "O": 4},
        "MgO": {"Mg": 1, "O": 1},
        "Fe2(SO4)3": {"Fe": 2, "S": 3, "O": 12},
        "FeSO4": {"Fe": 1, "S": 1, "O": 4},
        "Fe2O3": {"Fe": 2, "O": 3},
        "Mg(OH)2": {"Mg": 1, "O": 2, "H": 2},
        "Ca(OH)2": {"Ca": 1, "O": 2, "H": 2},
        "H2O": {"H": 2, "O": 1},
        "SO2": {"S": 1, "O": 2},
        "O2": {"O": 2},
        "calcite": {"Ca": 1, "C": 1, "O": 3},
        "magnesite": {"Mg": 1, "C": 1, "O": 3},
        "siderite": {"Fe": 1, "C": 1, "O": 3},
        "dolomite": {"Ca": 1, "Mg": 1, "C": 2, "O": 6},
        "CO2": {"C": 1, "O": 2},
    }
    cases = [
        (_host("FeS_troilite", "FeS", "troilite"), {"Fe": 0, "S": 0, "O": 0}, 800.0, {"S2": 1e-8}),
        (_host("CaSO4", "CaSO4", "anhydrite"), {"Ca": 0, "S": 0, "O": 0}, 800.0, {"SO2": 1e-8, "O2": 2e4}),
        (_host("MgSO4", "MgSO4", "crystalline anhydrous magnesium sulfate"), {"Mg": 0, "S": 0, "O": 0}, 800.0, {"SO2": 1e-8, "O2": 2e4}),
        (_host("Fe2(SO4)3", "Fe2(SO4)3", "crystal"), {"Fe": 0, "S": 0, "O": 0}, 800.0, {"SO2": 1e-8, "O2": 2e4}),
        (_host("FeSO4", "FeSO4", "crystal"), {"Fe": 0, "S": 0, "O": 0}, 800.0, {"SO2": 1e-8, "O2": 2e4}),
        (_host("Mg(OH)2", "Mg(OH)2", "brucite"), {"Mg": 0, "H": 0, "O": 0}, 800.0, {"H2O": 1e5}),
        (_host("Ca(OH)2", "Ca(OH)2", "portlandite"), {"Ca": 0, "H": 0, "O": 0}, 800.0, {"H2O": 1e5}),
        (_host("calcite", "CaCO3", "calcite"), {"Ca": 0, "C": 0, "O": 0}, 800.0, {"CO2": 1e5}),
        (_host("magnesite", "MgCO3", "magnesite"), {"Mg": 0, "C": 0, "O": 0}, 600.0, {"CO2": 1e5}),
        (_host("siderite", "FeCO3", "siderite"), {"Fe": 0, "C": 0, "O": 0}, 500.0, {"CO2": 1e5, "O2": 1e5}),
        (_host("dolomite", "CaMg(CO3)2", "dolomite"), {"Ca": 0, "Mg": 0, "C": 0, "O": 0}, 600.0, {"CO2": 1e5}),
    ]
    for host, expected, temperature, pressures in cases:
        result = _evaluate(host, temperature, pressures)
        assert result["status"] == "supported"
        assert "diffusion_limited_equilibrium_upper_bound" in {
            notice["code"] for notice in result["notices"]
        }
        terms = result["stoichiometry"]
        assert len(result["equilibrium_gas_partial_pressures_Pa"]) == 1
        assert set(result["equilibrium_gas_partial_pressures_Pa"]) == {result["driving_gas_species_id"]}
        assert result["driving_gas_species_id"] in pressures
        balance = {element: 0.0 for element in expected}
        for term in terms:
            for element, count in formulas[term["species_id"]].items():
                balance[element] += term["nu_mol_per_mol_rxn"] * count
        assert balance == pytest.approx(expected, rel=1.0e-13, abs=0.0)
        assert all(term["nu_mol_per_mol_rxn"] in {-1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 1.5, 3.0} for term in terms)
        assert result["extent_bound_mol_rxn"] <= host["remaining_mol"]


def test_siderite_requires_oxygen_fugacity_and_uses_hematite_path():
    host = _host("siderite", "FeCO3", "siderite")
    missing = _evaluate(host, 500.0, {"CO2": 1e5})
    assert missing["status"] == "missing_required_pressure"
    assert missing["equilibrium_gas_partial_pressures_Pa"] is None

    supported = _evaluate(host, 500.0, {"CO2": 1e5}, oxygen=1e5)
    assert supported["status"] == "supported"
    assert [
        (term["species_id"], term["nu_mol_per_mol_rxn"])
        for term in supported["stoichiometry"]
    ] == [("siderite", -1.0), ("Fe2O3", 0.5), ("CO2", 1.0), ("O2", -0.25)]


def test_siderite_has_no_one_bar_co2_crossing_in_its_source_range():
    host = _host("siderite", "FeCO3", "siderite")
    temperatures = [
        node[0] for node in subsolidus._data()["species"]["siderite"]["nodes"]
    ]
    pressures = {"CO2": STANDARD_PRESSURE_PA, "O2": STANDARD_PRESSURE_PA}
    equilibrium_pressures = [
        _evaluate(host, temperature, pressures)["equilibrium_gas_partial_pressures_Pa"]["CO2"]
        for temperature in temperatures
    ]
    assert all(pressure > STANDARD_PRESSURE_PA for pressure in equilibrium_pressures)


@pytest.mark.parametrize(
    ("host_id", "temperature_pair", "expected_crossing"),
    [
        ("calcite", (1100.0, 1200.0), 1161.5),
        ("magnesite", (600.0, 700.0), 681.8),
        ("dolomite", (600.0, 700.0), 693.9),
    ],
)
def test_carbonate_one_bar_co2_crossings(host_id, temperature_pair, expected_crossing):
    formula, phase = {
        "calcite": ("CaCO3", "calcite"),
        "magnesite": ("MgCO3", "magnesite"),
        "dolomite": ("CaMg(CO3)2", "dolomite"),
    }[host_id]
    host = _host(host_id, formula, phase)
    values = [
        _evaluate(host, temperature, {"CO2": STANDARD_PRESSURE_PA})[
            "delta_rG0_J_per_mol_rxn"
        ]
        for temperature in temperature_pair
    ]
    t0, t1 = temperature_pair
    g0, g1 = values
    crossing = t0 + (t1 - t0) * (-g0) / (g1 - g0)
    assert crossing == pytest.approx(expected_crossing, abs=0.2)
    at_crossing = _evaluate(host, crossing, {"CO2": STANDARD_PRESSURE_PA})
    assert at_crossing["equilibrium_gas_partial_pressures_Pa"]["CO2"] == pytest.approx(
        STANDARD_PRESSURE_PA, rel=1e-8
    )


@pytest.mark.parametrize(
    ("host", "temperature", "pressures", "expected_driver", "expected_pressure", "interval"),
    [
        (_host("Fe2(SO4)3", "Fe2(SO4)3", "crystal"), 800.0, {"SO2": 1e5, "O2": 21_000.0}, "SO2", 7.5, (500.0, 1500.0)),
        (_host("FeSO4", "FeSO4", "crystal"), 1000.0, {"SO2": 1e5, "O2": 21_000.0}, "SO2", 140_000.0, (500.0, 1500.0)),
        (_host("Mg(OH)2", "Mg(OH)2", "brucite"), 500.0, {"H2O": 1e5}, "H2O", 23_600.0, (298.15, 1000.0)),
        (_host("Ca(OH)2", "Ca(OH)2", "portlandite"), 700.0, {"H2O": 1e5}, "H2O", 12_900.0, (298.15, 1000.0)),
    ],
)
def test_new_hosts_return_their_ceiling_and_gas_driver(
    host, temperature, pressures, expected_driver, expected_pressure, interval,
):
    result = _evaluate(host, temperature, pressures)

    assert result["status"] == "supported"
    assert result["driving_gas_species_id"] == expected_driver
    assert result["joint_valid_temperature_K"] == interval
    assert result["equilibrium_gas_partial_pressures_Pa"] == pytest.approx(
        {expected_driver: expected_pressure}, rel=0.02
    )
    assert "diffusion_limited_equilibrium_upper_bound" in {
        notice["code"] for notice in result["notices"]
    }
    if host["species_id"] in {"Mg(OH)2", "Ca(OH)2"}:
        assert not any(term["species_id"] == "O2" for term in result["stoichiometry"])


@pytest.mark.parametrize(
    "host,oxide",
    [
        (_host("CaSO4", "CaSO4", "anhydrite"), "CaO"),
        (_host("MgSO4", "MgSO4", "crystalline anhydrous magnesium sulfate"), "MgO"),
    ],
)
def test_sulfate_reaction_gibbs_is_the_sourced_phase_sum(host, oxide):
    temperature = 800.0
    result = _evaluate(host, temperature, {"SO2": 1e5, "O2": 1e5})
    data = subsolidus._data()["species"]
    expected = (
        subsolidus._node_value(data[oxide], temperature)
        + subsolidus._node_value(data["SO2"], temperature)
        - subsolidus._node_value(data[host["species_id"]], temperature)
    )
    assert result["delta_rG0_J_per_mol_rxn"] == pytest.approx(expected)


@pytest.mark.parametrize(
    "host,pressures,oxygen",
    [
        (_host("FeS_troilite", "FeS", "troilite"), {"S2": 1e-8}, None),
        (_host("CaSO4", "CaSO4", "anhydrite"), {"SO2": 1e-8, "O2": 2e4}, None),
        (_host("MgSO4", "MgSO4", "crystalline anhydrous magnesium sulfate"), {"SO2": 1e-8, "O2": 2e4}, None),
    ],
)
def test_equilibrium_pressure_satisfies_gibbs_identity(host, pressures, oxygen):
    t = 800.0
    result = _evaluate(host, t, pressures, oxygen=oxygen)
    assert result["status"] == "supported"
    p_eq = result["equilibrium_gas_partial_pressures_Pa"][result["driving_gas_species_id"]]
    q = p_eq / STANDARD_PRESSURE_PA
    if host["species_id"] == "FeS_troilite":
        q = math.sqrt(q) / host["activity"]
    else:
        q *= math.sqrt(pressures["O2"] / STANDARD_PRESSURE_PA) / host["activity"]
    assert result["delta_rG0_J_per_mol_rxn"] + R_J_MOL_K * t * math.log(q) == pytest.approx(0.0, abs=1e-8)

    perturbed = dict(pressures)
    perturbed[result["driving_gas_species_id"]] = p_eq * (4.0 if host["species_id"] == "FeS_troilite" else 2.0)
    result_perturbed = _evaluate(host, t, perturbed, oxygen=oxygen)
    assert result_perturbed["extent_bound_mol_rxn"] == pytest.approx(
        0.0, rel=1.0e-13, abs=0.0
    )


def test_sulfates_require_one_consistent_oxygen_input():
    host = _host("CaSO4", "CaSO4", "anhydrite")
    rejected = _evaluate(host, 800.0, {"SO2": 100.0, "O2": 20_000.0}, oxygen=30_000.0)
    assert rejected["status"] == "missing_required_pressure"
    assert rejected["equilibrium_gas_partial_pressures_Pa"] is None
    assert rejected["extent_bound_mol_rxn"] is None
    accepted = _evaluate(host, 800.0, {"SO2": 100.0, "O2": 20_000.0}, oxygen=20_000.0)
    assert accepted["status"] == "supported"
    assert list(accepted["equilibrium_gas_partial_pressures_Pa"]) == ["SO2"]


@pytest.mark.parametrize(
    ("include_s2", "oxygen_conflict"),
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_troilite_checks_only_its_required_sulfur_pressure(
    include_s2: bool,
    oxygen_conflict: bool,
) -> None:
    pressures = {"O2": 20_000.0}
    if include_s2:
        pressures["S2"] = 1e-8

    result = _evaluate(
        _host("FeS_troilite", "FeS", "troilite"),
        800.0,
        pressures,
        oxygen=30_000.0 if oxygen_conflict else 20_000.0,
    )

    if include_s2:
        assert result["status"] == "supported"
        assert result["equilibrium_gas_partial_pressures_Pa"]["S2"] == pytest.approx(
            2.8375060515539325e-10
        )
    else:
        assert result["status"] == "missing_required_pressure"
        assert result["equilibrium_gas_partial_pressures_Pa"] is None
        assert result["notices"][0]["detail"] == (
            "Missing required pressure input: S2."
        )


def test_oxygen_dependency_is_derived_from_reaction_terms() -> None:
    reaction = subsolidus._REACTIONS["FeS_troilite"]
    terms = reaction[2]
    assert not subsolidus._reaction_requires_oxygen_fugacity(terms)
    assert subsolidus._reaction_requires_oxygen_fugacity(
        (*terms, ("O2", "gas", 0.5))
    )


@pytest.mark.parametrize("mutation", ["zero_o2", "rename_o", "rename_fo2"])
def test_sulfate_gas_term_mutations_refuse_with_typed_result(
    monkeypatch: pytest.MonkeyPatch, mutation: str,
) -> None:
    reaction_id, driver, terms, formula = subsolidus._REACTIONS["CaSO4"]
    changed_terms = []
    for species, phase, coefficient in terms:
        if species == "O2":
            species = {"rename_o": "O", "rename_fo2": "fO2"}.get(
                mutation, species
            )
            if mutation == "zero_o2":
                coefficient = 0.0
        changed_terms.append((species, phase, coefficient))
    monkeypatch.setitem(
        subsolidus._REACTIONS,
        "CaSO4",
        (reaction_id, driver, tuple(changed_terms), formula),
    )

    result = _evaluate(
        _host("CaSO4", "CaSO4", "anhydrite"),
        800.0,
        {"SO2": 100.0, "O2": 20_000.0},
    )
    assert result["status"] == "unknown_host"
    assert result["notices"][0]["code"] == "unknown_host"
    assert result["notices"][0]["detail"] == (
        "Reaction gas terms do not match the supported host algebra."
    )
    assert result["equilibrium_gas_partial_pressures_Pa"] is None


def test_reaction_with_unmodeled_oxygen_term_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reaction_id, driver, terms, formula = subsolidus._REACTIONS["FeS_troilite"]
    monkeypatch.setitem(
        subsolidus._REACTIONS,
        "FeS_troilite",
        (reaction_id, driver, (*terms, ("O2", "gas", 0.5)), formula),
    )
    result = _evaluate(
        _host("FeS_troilite", "FeS", "troilite"),
        800.0,
        {"S2": 1e-8, "O2": 20_000.0},
    )
    assert result["status"] == "unknown_host"
    assert result["notices"][0]["code"] == "unknown_host"


def test_higher_oxygen_fugacity_suppresses_sulfate_equilibrium_pressure():
    host = _host("MgSO4", "MgSO4", "crystalline anhydrous magnesium sulfate")
    low = _evaluate(host, 900.0, {"SO2": 100.0}, oxygen=10_000.0)
    high = _evaluate(host, 900.0, {"SO2": 100.0}, oxygen=40_000.0)
    assert low["status"] == high["status"] == "supported"
    assert low["equilibrium_gas_partial_pressures_Pa"]["SO2"] == pytest.approx(
        2.0 * high["equilibrium_gas_partial_pressures_Pa"]["SO2"]
    )


def test_required_pressures_and_unavailable_ranges_refuse_without_fake_numbers():
    missing = _evaluate(_host("FeS_troilite", "FeS", "troilite"), 800.0, {})
    assert missing["status"] == "missing_required_pressure"
    assert missing["equilibrium_gas_partial_pressures_Pa"] is None
    assert missing["extent_bound_mol_rxn"] is None

    sulfate = _evaluate(_host("MgSO4", "MgSO4", "crystalline anhydrous magnesium sulfate"), 450.0, {"SO2": 1e5, "O2": 1e5})
    assert sulfate["status"] == "temperature_out_of_range"
    assert sulfate["joint_valid_temperature_K"] == (500.0, 1400.0)
    assert sulfate["delta_rG0_J_per_mol_rxn"] is None
    assert sulfate["equilibrium_gas_partial_pressures_Pa"] is None
    assert sulfate["extent_bound_mol_rxn"] is None

    anhydrite = _evaluate(_host("CaSO4", "CaSO4", "anhydrite"), 1001.0, {"SO2": 1e5, "O2": 1e5})
    assert anhydrite["status"] == "temperature_out_of_range"
    assert anhydrite["joint_valid_temperature_K"] == (500.0, 1000.0)


@pytest.mark.parametrize(
    ("host", "temperature", "pressures", "interval"),
    [
        (_host("Fe2(SO4)3", "Fe2(SO4)3", "crystal"), 499.0, {"SO2": 1e5, "O2": 21_000.0}, (500.0, 1500.0)),
        (_host("FeSO4", "FeSO4", "crystal"), 1501.0, {"SO2": 1e5, "O2": 21_000.0}, (500.0, 1500.0)),
        (_host("Mg(OH)2", "Mg(OH)2", "brucite"), 298.14, {"H2O": 1e5}, (298.15, 1000.0)),
        (_host("Ca(OH)2", "Ca(OH)2", "portlandite"), 1000.1, {"H2O": 1e5}, (298.15, 1000.0)),
    ],
)
def test_new_hosts_keep_joint_source_range_refusals(host, temperature, pressures, interval):
    result = _evaluate(host, temperature, pressures)

    assert result["status"] == "temperature_out_of_range"
    assert result["joint_valid_temperature_K"] == interval
    assert result["delta_rG0_J_per_mol_rxn"] is None
    assert result["equilibrium_gas_partial_pressures_Pa"] is None
    assert result["extent_bound_mol_rxn"] is None


def test_unknown_host_and_wrong_phase_do_not_dispatch_by_formula():
    unknown = _evaluate(_host("S", "S", "element", 1.0), 800.0, {"S2": 1.0})
    assert unknown["status"] == "unknown_host"
    assert unknown["delta_rG0_J_per_mol_rxn"] is None
    wrong_phase = _evaluate(_host("MgSO4", "MgSO4", "epsomite"), 800.0, {"SO2": 1.0, "O2": 1.0})
    assert wrong_phase["status"] == "unknown_host"
    assert wrong_phase["extent_bound_mol_rxn"] is None


@pytest.mark.parametrize(
    ("species_id", "formula"),
    [
        ("gypsum", "CaSO4·2H2O"),
        ("bassanite", "CaSO4·0.5H2O"),
        ("goethite", "FeOOH"),
        ("jarosite", "KFe3(SO4)2(OH)6"),
        ("Na2SO4", "Na2SO4"),
        ("NaCl", "NaCl"),
        ("KCl", "KCl"),
        ("CaCl2", "CaCl2"),
        ("MgCl2", "MgCl2"),
        ("chlorapatite", "Ca5(PO4)3Cl"),
        ("fluorapatite", "Ca5(PO4)3F"),
        ("hydroxyapatite", "Ca5(PO4)3OH"),
        ("Mg(ClO4)2", "Mg(ClO4)2"),
    ],
)
def test_typed_gap_hosts_remain_unknown(species_id, formula):
    result = _evaluate(_host(species_id, formula, "crystal"), 800.0, {"SO2": 1e5, "O2": 21_000.0})

    assert result["status"] == "unknown_host"
    assert result["delta_rG0_J_per_mol_rxn"] is None
    assert result["equilibrium_gas_partial_pressures_Pa"] is None
    assert result["extent_bound_mol_rxn"] is None


@pytest.mark.parametrize(
    ("host_id", "temperatures"),
    [
        ("Fe2(SO4)3", (500.0, 1000.0, 1500.0)),
        ("FeSO4", (500.0, 1000.0, 1500.0)),
        ("Mg(OH)2", (298.15, 600.0, 1000.0)),
        ("Ca(OH)2", (298.15, 600.0, 1000.0)),
    ],
)
def test_janaf_reaction_formation_gibbs_matches_absolute_gibbs(
    host_id: str, temperatures: tuple[float, float, float],
) -> None:
    table_by_species = {
        "Fe2(SO4)3": "Fe-031",
        "FeSO4": "Fe-022",
        "Fe2O3": "Fe-030",
        "SO2": "O-034",
        "O2": "O-029",
        "Mg(OH)2": "H-057",
        "MgO": "Mg-008",
        "H2O": "H-064",
        "Ca(OH)2": "Ca-020",
        "CaO": "Ca-027",
    }
    data = subsolidus._data()["species"]
    terms = subsolidus._REACTIONS[host_id][2]

    for temperature in temperatures:
        formation_delta_g = sum(
            nu * subsolidus._node_value(data[species], temperature)
            for species, _phase, nu in terms
            if species != "O2"
        )
        absolute_delta_g = sum(
            nu * _janaf_absolute_gibbs_J(table_by_species[species], temperature)
            for species, _phase, nu in terms
        )
        assert formation_delta_g == pytest.approx(absolute_delta_g, abs=25.0), (
            f"{host_id} at {temperature:g} K: formation-from-elements "
            f"ΔrG={formation_delta_g:g} J/mol; absolute-G ΔrG="
            f"{absolute_delta_g:g} J/mol"
        )


@pytest.mark.parametrize(
    ("host_id", "temperatures", "source_tables"),
    [
        ("calcite", (298.15, 700.0, 1200.0), {"calcite": "rh:calcite", "CaO": "Ca-027", "CO2": "C-095"}),
        ("magnesite", (298.15, 700.0, 1000.0), {"magnesite": "rh:magnesite", "MgO": "Mg-008", "CO2": "C-095"}),
        ("siderite", (298.15, 400.0, 600.0), {"siderite": "rh:siderite", "Fe2O3": "Fe-030", "CO2": "C-095", "O2": "O-029"}),
        ("dolomite", (298.15, 700.0, 1100.0), {"dolomite": "rh:dolomite", "calcite": "rh:calcite", "MgO": "Mg-008", "CO2": "C-095"}),
    ],
)
def test_carbonate_runtime_gibbs_matches_absolute_route(
    host_id, temperatures, source_tables,
):
    # Reaction Gibbs energy rebuilt independently from absolute G = H - T*S of
    # each species (R&H rows and JANAF tables) must match the evaluator. A
    # fixed-reference formation sum would equal this identically for an
    # atom-balanced reaction (elemental terms cancel), so it cannot detect a
    # mixed reference; that guard is the pinned crossing in
    # test_carbonate_one_bar_co2_crossings (mixing each table's published
    # formation Gibbs energy moves calcite from 1161.5 K to about 1150.9 K).
    host_formula, host_phase = {
        "calcite": ("CaCO3", "calcite"),
        "magnesite": ("MgCO3", "magnesite"),
        "siderite": ("FeCO3", "siderite"),
        "dolomite": ("CaMg(CO3)2", "dolomite"),
    }[host_id]
    pressures = {"CO2": STANDARD_PRESSURE_PA}
    if host_id == "siderite":
        pressures["O2"] = STANDARD_PRESSURE_PA
    host = _host(host_id, host_formula, host_phase)
    terms = subsolidus._REACTIONS[host_id][2]

    for temperature in temperatures:
        absolute_delta_g = 0.0
        for species, _phase, nu in terms:
            table_id = source_tables[species]
            absolute_g = (
                _rh_absolute_gibbs_J(table_id.removeprefix("rh:"), temperature)
                if table_id.startswith("rh:")
                else _janaf_absolute_gibbs_J(table_id, temperature)
            )
            absolute_delta_g += nu * absolute_g

        result = _evaluate(host, temperature, pressures)
        assert result["status"] == "supported"
        assert result["delta_rG0_J_per_mol_rxn"] == pytest.approx(absolute_delta_g, abs=0.5)


def test_existing_subsolidus_records_remain_byte_identical() -> None:
    expected_sha256 = {
        "FeS_troilite": "d11368b1a6b27d23d2e35454d677cf039eeded99a96769fbdcabdcf1cc5202d6",
        "CaSO4": "3128d3f2b39d2f9068d87ee45bd864ff081c0ba36786940ea50f99152ce1f331",
        "MgSO4": "2a8b79cdf96609ce879945bc092dd6b5cbc608c8b978f4655f4c50f3dcdb76b5",
        "SO2": "ab98e0b0ee43e83eec1765d45a7342d722afd318ed828f4e434a1250302a2312",
        "Fe2(SO4)3": "8269228a5dcb14a8f8671db998022fd34e5c7dd38b391642ec1ea759bdd3e642",
        "FeSO4": "d5e83c4c8acbf948c6fa27a14dd10d085e9ae496bc95bbdd83001d1171c2293c",
        "Mg(OH)2": "20da55d4526823fb9786f0e027a13138079de43fecd35e067772b80136586a46",
        "Ca(OH)2": "b08e56445d49c79c3e015dc892efabeeb11a339b345d45b41adddd868aa53893",
        "CaO": "ae7951f61b6d9bab59021380b502065d4149db151677db2e633c19050dc990ea",
        "MgO": "bad77680143dc62fac3af25f092e6c27c5b07c27aa3d94f1039a93cae5404547",
        "Fe2O3": "01a11a267b5984924faf1c51260c8bebbf303eaff5c1e4ef1751a6907e27ce6d",
        "H2O": "4c038316243b07e79db32ed3a7b509f1ccd08425e232b77c62d5f662c05b741f",
    }

    for name, digest in expected_sha256.items():
        record = _raw_subsolidus_record(name).encode()
        assert hashlib.sha256(record).hexdigest() == digest


def test_literature_scale_checks_for_each_reaction():
    # Robie & Hemingway (1995), USGS Bulletin 2131, troilite table p. 156:
    # ΔfG°(FeS)=-111.4 kJ/mol at 800 K, with ideal S2 at 1 bar as reference.
    troilite = _evaluate(_host("FeS_troilite", "FeS", "troilite"), 800.0, {"S2": 1e-10})
    assert troilite["equilibrium_gas_partial_pressures_Pa"]["S2"] == pytest.approx(2.84e-10, rel=0.02)

    # Robie & Hemingway (1995), USGS Bulletin 2131, anhydrite table p. 285,
    # prints ΔfG°(CaSO4)=-1046.3 kJ/mol at 1000 K to 0.1 kJ precision. Newman
    # (1941), p. 191, discusses an earlier total sulfur-oxide/oxygen pressure
    # near 1225 °C, outside the host row range; it is not an SO2 pressure test.
    anhydrite = _evaluate(_host("CaSO4", "CaSO4", "anhydrite"), 1000.0, {"SO2": 1e5, "O2": 1e5})
    assert anhydrite["delta_rG0_J_per_mol_rxn"] == pytest.approx(226_898.0, abs=100.0)

    # Scheidema & Taskinen (2011), DOI 10.1021/ie102554f, report an HSC
    # Chemistry equilibrium temperature of 1080 °C, not a measured Kp. Pin
    # this check to the sourced Gibbs sum at 1353.15 K. For unit solid
    # activities and pO2=1 bar, the written reaction predicts pSO2,eq=Kp*1bar.
    magnesium = _evaluate(
        _host("MgSO4", "MgSO4", "crystalline anhydrous magnesium sulfate"),
        1353.15, {"SO2": 1e5, "O2": 1e5},
    )
    delta_g0 = magnesium["delta_rG0_J_per_mol_rxn"]
    assert delta_g0 == pytest.approx(13_631.0, abs=100.0)
    kp = math.exp(magnesium["log10_Kp"] * math.log(10.0))
    assert kp == pytest.approx(math.exp(-13_631.0 / (R_J_MOL_K * 1353.15)), rel=0.01)


def test_positive_subnormal_pressures_use_log_differences():
    troilite = _evaluate(
        _host("FeS_troilite", "FeS", "troilite"), 1000.0,
        {"S2": 5.0e-324}, {"Fe": 1.0},
    )
    assert troilite["status"] == "supported"

    anhydrite = _evaluate(
        _host("CaSO4", "CaSO4", "anhydrite"), 800.0,
        {"SO2": 1.0e5, "O2": 5.0e-324}, {"CaO": 1.0},
    )
    assert anhydrite["status"] == "supported"
    assert math.isfinite(
        anhydrite["equilibrium_gas_partial_pressures_Pa"]["SO2"]
    )


def test_underflowed_positive_activity_ratio_is_numeric_out_of_range():
    result = _evaluate(
        _host("FeS_troilite", "FeS", "troilite", activity=2.0),
        1000.0,
        {"S2": 1.0},
        {"Fe": 5.0e-324},
    )
    assert result["status"] == "numeric_out_of_range"
    assert result["notices"][0]["code"] == "numeric_out_of_range"


@pytest.mark.parametrize("oxygen_Pa", [1.0e20, 1.0e308])
def test_underflowed_activity_ratio_with_finite_pressure_returns(oxygen_Pa):
    # The host/product ratio underflows to 0, but the log-difference keeps the
    # equilibrium pressure finite, so the activity term must not take log(0).
    result = _evaluate(
        _host("CaSO4", "CaSO4", "anhydrite", activity=2.0), 800.0,
        {"SO2": 1.0, "O2": oxygen_Pa}, {"CaO": 5.0e-324},
    )
    assert result["status"] == "supported"


@pytest.mark.parametrize(
    ("target_log_pressure_eq", "expected_status"),
    [(-750.0, "supported"), (-800.0, "numeric_out_of_range")],
)
def test_troilite_pressure_scales_before_exponent_underflow(
    target_log_pressure_eq, expected_status
):
    temperature = 1000.0
    host_g = subsolidus._node_value(
        subsolidus._data()["species"]["FeS_troilite"], temperature
    )
    base_log_pressure_eq = 2.0 * host_g / (R_J_MOL_K * temperature)
    iron_activity = math.exp(
        (base_log_pressure_eq - target_log_pressure_eq) / 2.0
    )
    actual_log_pressure_eq = base_log_pressure_eq - 2.0 * math.log(iron_activity)
    assert actual_log_pressure_eq == pytest.approx(
        target_log_pressure_eq, abs=1e-12
    )

    result = _evaluate(
        _host("FeS_troilite", "FeS", "troilite", activity=1.0),
        temperature,
        {"S2": 1.0},
        {"Fe": iron_activity},
    )

    assert result["status"] == expected_status
    if expected_status == "supported":
        pressure = result["equilibrium_gas_partial_pressures_Pa"]["S2"]
        assert pressure == math.exp(
            actual_log_pressure_eq + math.log(STANDARD_PRESSURE_PA)
        )
        assert pressure > 0.0
    else:
        assert result["equilibrium_gas_partial_pressures_Pa"] is None
        assert result["notices"][0]["code"] == "numeric_out_of_range"


def test_unrepresentable_equilibrium_pressure_only_marks_its_host_unavailable():
    hosts = [
        _host("FeS_troilite", "FeS", "troilite"),
        _host("CaSO4", "CaSO4", "anhydrite"),
    ]
    results = evaluate_subsolidus_reactions(
        hosts=hosts,
        temperature_K=1000.0,
        gas_partial_pressures_Pa={"S2": 1.0, "SO2": 1.0e5, "O2": 1.0e5},
        condensed_phase_activities={"Fe": 1.0e-200, "CaO": 1.0},
    )
    unavailable, supported = results
    assert unavailable["status"] == "numeric_out_of_range"
    assert unavailable["equilibrium_gas_partial_pressures_Pa"] is None
    assert unavailable["extent_bound_mol_rxn"] is None
    assert unavailable["notices"][0]["code"] == "numeric_out_of_range"
    assert unavailable["notices"][0]["severity"] == "unavailable"
    assert supported["status"] == "supported"
