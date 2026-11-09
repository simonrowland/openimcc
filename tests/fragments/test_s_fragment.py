import json
import math
import re

import pytest

from openimcc import evaluate, evaluate_sulfur, load_datapack
from openimcc._constants import MOLAR_MASS_G_MOL
from openimcc.activity_fragments import load_activity_fragment
from openimcc.gas import evaluate_gas, load_gas_datapack


# N-MORB from Smythe et al. (2017), Table 3, p. 800, DOI 10.2138/am-2017-5800CCBY.
_MORB = {
    "SiO2": 50.42,
    "TiO2": 1.53,
    "Al2O3": 15.13,
    "FeO": 9.81,
    "MgO": 7.76,
    "CaO": 11.35,
    "Na2O": 2.83,
    "K2O": 0.14,
}
_OXYGEN_ATOMS = {
    "SiO2": 2,
    "TiO2": 2,
    "Al2O3": 3,
    "FeO": 1,
    "MgO": 1,
    "CaO": 1,
    "Na2O": 1,
    "K2O": 1,
}


def _single_oxygen_fractions(composition_wt_percent):
    formula_moles = {
        oxide: amount / MOLAR_MASS_G_MOL[oxide]
        for oxide, amount in composition_wt_percent.items()
    }
    oxygen_total = math.fsum(
        formula_moles[oxide] * _OXYGEN_ATOMS[oxide]
        for oxide in formula_moles
    )
    return {
        oxide: formula_moles[oxide] * _OXYGEN_ATOMS[oxide] / oxygen_total
        for oxide in formula_moles
    }


def _printed_capacity_anchors(oxygen_fractions, temperature_K):
    log_cs2 = 0.225 + (
        25237 * oxygen_fractions.get("FeO", 0.0)
        + 5214 * oxygen_fractions.get("CaO", 0.0)
        + 12705 * oxygen_fractions.get("MnO", 0.0)
        + 19829 * oxygen_fractions.get("K2O", 0.0)
        - 1109 * oxygen_fractions.get("SiO2", 0.0)
        - 8879
    ) / temperature_K
    log_cs6 = -12.948 + (
        28649 * oxygen_fractions.get("Na2O", 0.0)
        + 15602 * oxygen_fractions.get("CaO", 0.0)
        + 9496 * oxygen_fractions.get("MgO", 0.0)
        + 16016 * oxygen_fractions.get("MnO", 0.0)
        + 4194 * oxygen_fractions.get("Al2O3", 0.0)
        + 29244
    ) / temperature_K
    return log_cs2, log_cs6


def test_fragment_rows_match_printed_capacity_equations_and_statistics():
    fragment = load_activity_fragment("S")
    sulfide = fragment["rows"]["sulfide_capacity_eq_7"]
    sulfate = fragment["rows"]["sulfate_capacity_eq_9"]

    assert sulfide["row"] == {
        "intercept": 0.225,
        "terms_K": {
            "FeO": 25237.0,
            "CaO": 5214.0,
            "MnO": 12705.0,
            "K2O": 19829.0,
            "SiO2": -1109.0,
        },
        "constant_K": -8879.0,
    }
    assert sulfide["coefficient_standard_errors"] == {
        "intercept": 0.190,
        "FeO": 784.0,
        "CaO": 838.0,
        "MnO": 4420.0,
        "K2O": 7139.0,
        "SiO2": 337.0,
        "constant_K": 408.0,
    }
    assert sulfide["fit_residual"]["value"] == 0.09
    assert sulfide["fit_residual"]["unit"] == "log10(wt% S)"
    assert sulfide["accuracy_band"] is None
    assert "Eq. 7, p. 6" in sulfide["provenance"]["source_location"]

    assert sulfate["row"] == {
        "intercept": -12.948,
        "terms_K": {
            "Na2O": 28649.0,
            "CaO": 15602.0,
            "MgO": 9496.0,
            "MnO": 16016.0,
            "Al2O3": 4194.0,
        },
        "constant_K": 29244.0,
    }
    assert sulfate["coefficient_standard_errors"] == {
        "intercept": 0.278,
        "Na2O": 4040.0,
        "CaO": 1645.0,
        "MgO": 1279.0,
        "MnO": 7128.0,
        "Al2O3": 1163.0,
        "constant_K": 493.0,
    }
    assert sulfate["fit_residual"]["value"] == 0.14
    assert sulfate["fit_residual"]["unit"] == "log10(wt% S)"
    assert sulfate["accuracy_band"] is None
    assert sulfate["provenance"]["underlying_data_doi"] == "10.1016/j.gca.2022.08.032"
    assert "Eq. 9, p. 7" in sulfate["provenance"]["source_location"]

    assert fragment["standards"]["S2(g)"]["reference_pressure_bar"] == 1.0
    assert fragment["standards"]["S2(g)"]["source_url"].endswith("S-012.html")
    assert "S2(l)" not in json.dumps(fragment["standards"])
    assert fragment["calibration"]["temperature_K"] == [1473.0, 1773.0]


def test_capacity_and_total_s_fugacity_follow_printed_equations():
    temperature_K = 1673.0
    log_fO2 = -8.97
    total_s_ppm = 500.0
    oxygen_fractions = _single_oxygen_fractions(_MORB)
    expected_cs2, expected_cs6 = _printed_capacity_anchors(
        oxygen_fractions, temperature_K,
    )
    result = evaluate_sulfur(
        _MORB, temperature_K, log10_fO2=log_fO2, total_s_ppm=total_s_ppm,
    )

    assert result.log10_sulfide_capacity_wt_percent == pytest.approx(expected_cs2)
    assert result.log10_sulfate_capacity_wt_percent == pytest.approx(expected_cs6)

    log_a2 = expected_cs2 - 0.5 * log_fO2
    log_a6 = expected_cs6 + 1.5 * log_fO2
    largest = max(log_a2, log_a6)
    log_sum = largest + math.log10(
        10 ** (log_a2 - largest) + 10 ** (log_a6 - largest)
    )
    expected_log_fS2 = 2 * (math.log10(total_s_ppm / 1.0e4) - log_sum)
    assert result.log10_fS2 == pytest.approx(expected_log_fS2)
    assert result.log10_fS2 == pytest.approx(-2.5721, abs=0.001)
    notice, = (
        item for item in result.notices
        if item.code == "imcc_sulfur_parent_domain_notice"
    )
    assert notice.model == "Boulliung-Wood-2023-Eq-7-and-Eq-9"
    assert notice.calibrated_interval == (1473.0, 1773.0)


def test_s_fragment_adds_finite_s_gases_and_preserves_non_s_pressures():
    temperature_K = 1773.0
    log_fO2 = -8.97
    fO2 = 10**log_fO2
    composition = {
        "SiO2": 45.5,
        "TiO2": 8.0,
        "Al2O3": 11.9,
        "FeO": 16.5,
        "MgO": 7.5,
        "CaO": 9.6,
        "Na2O": 0.8,
        "K2O": 0.2,
    }
    sulfur = evaluate_sulfur(
        composition, temperature_K, log10_fO2=log_fO2, total_s_ppm=630.0,
    )
    assert sulfur.log10_fS2 is not None
    melt = evaluate(composition, temperature_K, load_datapack())
    gas_pack = load_gas_datapack()
    activities = {name: melt.activity(name) for name in melt.parent_oxides}
    with_s2 = {**activities, "S2": 10**sulfur.log10_fS2}
    parents = (*melt.parent_oxides, "S2")

    baseline = evaluate_gas(
        activities, temperature_K, fO2, gas_pack, parent_oxides=parents,
    )
    candidate = evaluate_gas(
        with_s2, temperature_K, fO2, gas_pack, parent_oxides=parents,
    )
    contains_sulfur = lambda species: "S" in re.findall(r"[A-Z][a-z]?", species)
    sulfur_pressures = {
        species: pressure
        for species, pressure in candidate.items()
        if contains_sulfur(species)
    }
    assert sulfur_pressures
    assert all(math.isfinite(pressure) for pressure in sulfur_pressures.values())
    assert any(pressure > 0.0 for pressure in sulfur_pressures.values())

    baseline_non_s = {
        species: pressure for species, pressure in baseline.items()
        if not contains_sulfur(species)
    }
    candidate_non_s = {
        species: pressure for species, pressure in candidate.items()
        if not contains_sulfur(species)
    }
    assert candidate_non_s == baseline_non_s
