from __future__ import annotations

import math

import pytest

from openimcc import Cr3TraceActivity, cr3_trace_activity
from openimcc.gas import IMCC_PARENT_OXIDES, evaluate_gas, load_gas_datapack


_COMP_MOL_B1 = {"CaO": 1.0 / 56.0774, "SiO2": 1.0 / 60.0843}
_T_K = 1773.0
_FO2_BAR = 10.0**-11.5 * 1.01325
_HIGH_FO2_BAR = 10.0**-9.56 * 1.01325


def _composition_with_basicity(basicity: float) -> dict[str, float]:
    return {"CaO": basicity, "SiO2": 1.0}


def test_candidate_uses_dilute_half_formula_and_quadratic_parent_activity():
    low = cr3_trace_activity(_COMP_MOL_B1, _T_K, _FO2_BAR, 0.001)
    high = cr3_trace_activity(_COMP_MOL_B1, _T_K, _FO2_BAR, 0.002)

    assert isinstance(low, Cr3TraceActivity)
    assert low.validation_status == "unvalidated opt-in candidate"
    assert low.parent_activity == pytest.approx(low.half_formula_activity**2)
    assert high.parent_activity / low.parent_activity == pytest.approx(4.0)
    assert math.sqrt(low.parent_activity) / 0.001 == pytest.approx(
        low.gamma_infinity
    )
    assert low.gamma_infinity == pytest.approx(10.0**1.26499243521)
    assert high.gamma_infinity == low.gamma_infinity


def test_zero_inventory_is_exact_and_positive_loading_stays_in_measured_range():
    zero = cr3_trace_activity(_COMP_MOL_B1, _T_K, _FO2_BAR, 0.0)
    assert zero.parent_activity == 0.0
    assert zero.half_formula_activity == 0.0
    lower_edge = cr3_trace_activity(
        _composition_with_basicity(1.27), _T_K, _FO2_BAR, 0.0008, basis_type="wt"
    )
    assert lower_edge.parent_activity > 0.0
    with pytest.raises(ValueError, match="measured dilute-loading range"):
        cr3_trace_activity(
            _composition_with_basicity(1.27), _T_K, _FO2_BAR, 0.00079, basis_type="wt"
        )


def test_basicity_uses_sf04_mol_or_weight_input_and_refuses_other_matrices():
    by_moles = cr3_trace_activity(_COMP_MOL_B1, _T_K, _FO2_BAR, 0.001)
    by_weight = cr3_trace_activity(
        {"CaO": 50.0, "SiO2": 50.0},
        _T_K,
        _FO2_BAR,
        0.001,
        basis_type="wt",
    )
    assert by_moles.gamma_infinity == pytest.approx(by_weight.gamma_infinity)
    with pytest.raises(ValueError, match="outside measured Cr composition hull"):
        cr3_trace_activity(
            {**_COMP_MOL_B1, "MgO": 0.01}, _T_K, _FO2_BAR, 0.001
        )
    with pytest.raises(
        ValueError, match="basicity B=.*outside measured Cr composition hull"
    ):
        cr3_trace_activity(
            {"CaO": 2.0 / 56.0774, "SiO2": 1.0 / 60.0843},
            _T_K,
            _FO2_BAR,
            0.001,
        )


def test_basicity_slope_is_pinned_away_from_unity():
    result = cr3_trace_activity(
        _composition_with_basicity(0.52),
        _T_K,
        _FO2_BAR,
        0.003,
        basis_type="wt",
    )
    expected = 10.0 ** (1.26499243521 + 0.62567770680 * math.log10(0.52))
    assert result.gamma_infinity == pytest.approx(expected)
    assert result.gamma_infinity != pytest.approx(10.0**1.26499243521)


def test_all_basicity_hull_corners_are_accepted_and_edges_refuse_outside():
    corners = (
        (_FO2_BAR, 0.52),
        (_FO2_BAR, 1.27),
        (_HIGH_FO2_BAR, 0.71),
        (_HIGH_FO2_BAR, 1.25),
    )
    for fugacity, basicity in corners:
        result = cr3_trace_activity(
            _composition_with_basicity(basicity),
            _T_K,
            fugacity,
            0.003,
            basis_type="wt",
        )
        assert result.parent_activity > 0.0

    just_outside = (
        (_FO2_BAR, 0.52 - 1e-8),
        (_FO2_BAR, 1.27 + 1e-8),
        (_HIGH_FO2_BAR, 0.71 - 1e-8),
        (_HIGH_FO2_BAR, 1.25 + 1e-8),
    )
    for fugacity, basicity in just_outside:
        with pytest.raises(ValueError, match="outside measured Cr composition hull"):
            cr3_trace_activity(
                _composition_with_basicity(basicity),
                _T_K,
                fugacity,
                0.003,
                basis_type="wt",
            )


def test_ccrs5_at_high_fugacity_is_accepted_and_ccrs13_is_refused():
    ccrs5 = cr3_trace_activity(
        _composition_with_basicity(1.25),
        _T_K,
        _HIGH_FO2_BAR,
        0.0046,
        basis_type="wt",
    )
    assert ccrs5.parent_activity > 0.0

    with pytest.raises(ValueError, match="measured dilute-loading range"):
        cr3_trace_activity(
            _composition_with_basicity(0.93),
            _T_K,
            _HIGH_FO2_BAR,
            0.01,
            basis_type="wt",
        )


def test_upper_loading_bound_is_inclusive_and_the_next_value_is_refused():
    high_edge = cr3_trace_activity(
        _composition_with_basicity(1.21),
        _T_K,
        _FO2_BAR,
        0.0069,
        basis_type="wt",
    )
    assert high_edge.parent_activity > 0.0
    with pytest.raises(ValueError, match="measured dilute-loading range"):
        cr3_trace_activity(
            _composition_with_basicity(1.21),
            _T_K,
            _FO2_BAR,
            0.00691,
            basis_type="wt",
        )


def test_temperature_and_fugacity_outside_measured_hull_are_refused():
    with pytest.raises(ValueError, match="outside measured Cr range"):
        cr3_trace_activity(_COMP_MOL_B1, 1800.0, _FO2_BAR, 0.001)
    with pytest.raises(ValueError, match=r"log10\(fO2/atm\) is outside"):
        cr3_trace_activity(_COMP_MOL_B1, _T_K, 10.0**-12.0 * 1.01325, 0.001)


def test_existing_eight_parent_gas_outputs_and_cr2o3_pass_through():
    assert IMCC_PARENT_OXIDES == (
        "SiO2",
        "MgO",
        "FeO",
        "CaO",
        "Al2O3",
        "TiO2",
        "Na2O",
        "K2O",
    )
    pack = load_gas_datapack()
    activities = {
        "SiO2": 0.4,
        "CaO": 0.2,
        "FeO": 0.1,
        "MgO": 0.1,
        "Al2O3": 0.1,
        "TiO2": 0.05,
        "Na2O": 0.03,
        "K2O": 0.02,
    }
    species = ("Ca", "SiO", "Fe", "Mg", "AlO", "Ti", "Na", "K")

    # Hex literals pin the pre-change baseline bit-for-bit, one channel per parent.
    baseline = evaluate_gas(
        activities, 1800.0, 1.0e-8, pack, gas_species=species
    )
    assert tuple(float(baseline[name]).hex() for name in species) == (
        "0x1.c31f1c387bffbp-29",
        "0x1.fbfa0f749d3b0p-22",
        "0x1.4c8a33a9ff2f6p-20",
        "0x1.094053d32a5d2p-22",
        "0x1.2fb51a1637602p-41",
        "0x1.4c45c8294ecd4p-58",
        "0x1.672fdddc28782p-1",
        "0x1.b993b8a9db241p+1",
    )

    trace = cr3_trace_activity(_COMP_MOL_B1, _T_K, _FO2_BAR, 0.001)
    cr_gas = evaluate_gas(
        {"Cr2O3": trace.parent_activity},
        _T_K,
        _FO2_BAR,
        pack,
        parent_oxides=("Cr2O3",),
        gas_species=("Cr",),
    )
    assert cr_gas["Cr"] > 0.0
