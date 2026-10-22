import math

import pytest

from openimcc.subsolidus import (
    R_J_MOL_K,
    STANDARD_PRESSURE_PA,
    evaluate_subsolidus_reactions,
)
from openimcc import subsolidus


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


def test_supported_reactions_are_atom_balanced_and_have_one_driver():
    cases = [
        (_host("FeS_troilite", "FeS", "troilite"), {"Fe": 0, "S": 0, "O": 0}),
        (_host("CaSO4", "CaSO4", "anhydrite"), {"Ca": 0, "S": 0, "O": 0}),
        (_host("MgSO4", "MgSO4", "crystalline anhydrous magnesium sulfate"), {"Mg": 0, "S": 0, "O": 0}),
    ]
    for host, expected in cases:
        pressures = {"S2": 1e-8} if host["species_id"] == "FeS_troilite" else {"SO2": 1e-8, "O2": 2e4}
        result = _evaluate(host, 800.0, pressures)
        assert result["status"] == "supported"
        assert "diffusion_limited_equilibrium_upper_bound" in {
            notice["code"] for notice in result["notices"]
        }
        terms = result["stoichiometry"]
        assert len(result["equilibrium_gas_partial_pressures_Pa"]) == 1
        assert set(result["equilibrium_gas_partial_pressures_Pa"]) == {result["driving_gas_species_id"]}
        balance = {element: 0.0 for element in expected}
        for term in terms:
            formula = {"FeS_troilite": {"Fe": 1, "S": 1}, "Fe": {"Fe": 1}, "S2": {"S": 2},
                       "CaSO4": {"Ca": 1, "S": 1, "O": 4}, "CaO": {"Ca": 1, "O": 1},
                       "MgSO4": {"Mg": 1, "S": 1, "O": 4}, "MgO": {"Mg": 1, "O": 1},
                       "SO2": {"S": 1, "O": 2}, "O2": {"O": 2}}[term["species_id"]]
            for element, count in formula.items():
                balance[element] += term["nu_mol_per_mol_rxn"] * count
        assert balance == expected
        assert all(term["nu_mol_per_mol_rxn"] in {-1.0, -0.5, 0.5, 1.0} for term in terms)
        assert result["extent_bound_mol_rxn"] <= host["remaining_mol"]


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
    assert result_perturbed["extent_bound_mol_rxn"] == 0.0


def test_sulfates_require_one_consistent_oxygen_input():
    host = _host("CaSO4", "CaSO4", "anhydrite")
    rejected = _evaluate(host, 800.0, {"SO2": 100.0, "O2": 20_000.0}, oxygen=30_000.0)
    assert rejected["status"] == "missing_required_pressure"
    assert rejected["equilibrium_gas_partial_pressures_Pa"] is None
    assert rejected["extent_bound_mol_rxn"] is None
    accepted = _evaluate(host, 800.0, {"SO2": 100.0, "O2": 20_000.0}, oxygen=20_000.0)
    assert accepted["status"] == "supported"
    assert list(accepted["equilibrium_gas_partial_pressures_Pa"]) == ["SO2"]


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


def test_unknown_host_and_wrong_phase_do_not_dispatch_by_formula():
    unknown = _evaluate(_host("S", "S", "element", 1.0), 800.0, {"S2": 1.0})
    assert unknown["status"] == "unknown_host"
    assert unknown["delta_rG0_J_per_mol_rxn"] is None
    wrong_phase = _evaluate(_host("MgSO4", "MgSO4", "epsomite"), 800.0, {"SO2": 1.0, "O2": 1.0})
    assert wrong_phase["status"] == "unknown_host"
    assert wrong_phase["extent_bound_mol_rxn"] is None


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
