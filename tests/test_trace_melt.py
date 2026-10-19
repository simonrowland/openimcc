from __future__ import annotations

import math

import pytest

from openimcc import Cr3TraceActivity, cr3_trace_activity
from openimcc.gas import IMCC_PARENT_OXIDES, evaluate_gas, load_gas_datapack


_COMP_MOL_B1 = {"CaO": 1.0 / 56.0774, "SiO2": 1.0 / 60.0843}
_T_K = 1773.0
_FO2_BAR = 10.0**-11.5 * 1.01325


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
    assert low.gamma_infinity == pytest.approx(10.0**1.27580279745)
    assert high.gamma_infinity == low.gamma_infinity


def test_zero_inventory_is_exact_and_positive_loading_stays_in_measured_range():
    zero = cr3_trace_activity(_COMP_MOL_B1, _T_K, _FO2_BAR, 0.0)
    assert zero.parent_activity == 0.0
    assert zero.half_formula_activity == 0.0
    with pytest.raises(ValueError, match="measured dilute-loading range"):
        cr3_trace_activity(_COMP_MOL_B1, _T_K, _FO2_BAR, 0.0004)


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
        "0x1.84713927983e3p-21",
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
