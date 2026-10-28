from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from openimcc import evaluate, evaluate_sulfur, load_datapack
from openimcc.gas import (
    evaluate_gas_oxygen_balance,
    load_gas_datapack,
    species_thermo,
)


# N-MORB from Smythe et al. (2017), Table 3, p. 800.
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
_OXIDES = {
    "SiO2": (60.0843, 1, 2, "Si"),
    "TiO2": (79.866, 1, 2, "Ti"),
    "Al2O3": (101.961, 2, 3, "Al"),
    "FeO": (71.844, 1, 1, "Fe"),
    "MnO": (70.9374, 1, 1, "Mn"),
    "MgO": (40.3044, 1, 1, "Mg"),
    "CaO": (56.0774, 1, 1, "Ca"),
    "Na2O": (61.979, 2, 1, "Na"),
    "K2O": (94.196, 2, 1, "K"),
}


def _fractions(composition: dict[str, float]):
    formula_moles = {name: wt / _OXIDES[name][0] for name, wt in composition.items()}
    oxygen_total = sum(n * _OXIDES[name][2] for name, n in formula_moles.items())
    cation_total = sum(n * _OXIDES[name][1] for name, n in formula_moles.items())
    oxygen = {
        name: n * _OXIDES[name][2] / oxygen_total
        for name, n in formula_moles.items()
    }
    cations: dict[str, float] = {}
    for name, n in formula_moles.items():
        symbol = _OXIDES[name][3]
        cations[symbol] = cations.get(symbol, 0.0) + n * _OXIDES[name][1] / cation_total
    return oxygen, cations, formula_moles


def _sulfur_liquid() -> dict[str, float]:
    return {"Fe": 0.35, "Ni": 0.10, "Cu": 0.05, "S": 0.46, "P": 0.02, "C": 0.02}


def _saturation_inventory(result) -> float:
    assert result.log10_fO2 is not None
    assert result.log10_sulfide_capacity_wt_percent is not None
    assert result.log10_sulfate_capacity_wt_percent is not None
    assert result.scss_smythe_2017_ppm is not None
    ratio = 10.0 ** (
        result.log10_sulfate_capacity_wt_percent
        - result.log10_sulfide_capacity_wt_percent
        + 2.0 * result.log10_fO2
    )
    return result.scss_smythe_2017_ppm * (1.0 + ratio)


def _assert_fugacity_reproduces_dissolved_species(result) -> None:
    assert result.log10_fO2 is not None
    assert result.log10_fS2 is not None
    assert result.log10_sulfide_capacity_wt_percent is not None
    assert result.log10_sulfate_capacity_wt_percent is not None
    expected_s2 = 10.0 ** (
        result.log10_sulfide_capacity_wt_percent
        + 0.5 * (result.log10_fS2 - result.log10_fO2)
        + 4.0
    )
    expected_s6 = 10.0 ** (
        result.log10_sulfate_capacity_wt_percent
        + 0.5 * result.log10_fS2
        + 1.5 * result.log10_fO2
        + 4.0
    )
    assert result.dissolved_s2_minus_ppm == pytest.approx(expected_s2)
    assert result.dissolved_s6_plus_ppm == pytest.approx(expected_s6)


def test_boulliung_wood_capacity_equations_match_printed_fits():
    T = 1673.0
    oxygen, cations, _ = _fractions(_MORB)
    result = evaluate_sulfur(_MORB, T, log10_fO2=-8.0, log10_fS2=-10.0)
    notice_codes = {notice.code for notice in result.notices}

    # Boulliung & Wood (2023), Eq. 7, journal p. 6; single-oxygen fractions,
    # wt% S. The fitted feed axes are listed in Table 1, journal p. 3.
    log_cs2 = 0.225 + (
        25237 * oxygen.get("FeO", 0.0)
        + 5214 * oxygen.get("CaO", 0.0)
        + 12705 * oxygen.get("MnO", 0.0)
        + 19829 * oxygen.get("K2O", 0.0)
        - 1109 * oxygen["SiO2"]
        - 8879
    ) / T
    # Boulliung & Wood (2022), Eq. 5, GCA 336, pp. 150-164;
    # single-cation fractions, wt% S.
    log_cs6 = -12.659 + (
        3692 * cations.get("Ca", 0.0)
        - 7592 * cations.get("Si", 0.0)
        - 13736 * cations.get("Ti", 0.0)
        + 3762 * cations.get("Al", 0.0)
        + 34483
    ) / T

    assert result.log10_sulfide_capacity_wt_percent == pytest.approx(log_cs2)
    assert result.log10_sulfate_capacity_wt_percent == pytest.approx(log_cs6)
    assert "metal-silicate-partition-not-modelled" in notice_codes
    assert "cos-h2s-not-in-gas-tables" in notice_codes


def test_smythe_ideal_scss_equation_and_sulfide_liquid_input():
    T, pressure = 1673.0, 1.0
    _oxygen, cations, _ = _fractions(_MORB)
    liquid = _sulfur_liquid()
    result = evaluate_sulfur(
        _MORB, T, log10_fO2=-8.0, log10_fS2=-10.0,
        total_pressure_bar=pressure, sulfide_liquid_composition=liquid,
    )
    # Smythe et al. (2017), Eq. 11 and Table 2, p. 799.
    x_fes = liquid["Fe"] / (liquid["Fe"] + liquid["Ni"] + liquid["Cu"])
    p_gpa = pressure * 1.0e-4
    ln_scss = (
        -14683.0 / T + 8.03 - 265.80 * p_gpa / T
        + (
            16430 * cations["Ti"] + 9295 * cations["Al"]
            + 13767 * cations["Mg"] + 19893 * cations["Ca"]
            - 7080 * cations["Fe"] + 14197 * cations["Na"]
            + 117827 * cations["Si"] * cations["Fe"]
        ) / T
        + math.log(x_fes) - math.log(cations["Fe"])
    )
    notices = {notice.code: notice for notice in result.notices}

    assert result.scss_smythe_2017_ppm == pytest.approx(math.exp(ln_scss))
    assert result.sulfide_liquid_x_fes == pytest.approx(x_fes)
    metal_rich = notices["metal-rich-sulfide-liquid-outside-smythe-calibration"]
    assert metal_rich.axis == "matte_S_to_stoichiometric_sulfide_ratio"
    assert metal_rich.calibrated_interval == (1.0, 1.0)
    assert metal_rich.distance_beyond_interval == pytest.approx(1.0 - 0.46 / 0.475)
    for element in ("P", "C"):
        component = next(
            notice for notice in result.notices
            if notice.code == "sulfide-liquid-component-outside-smythe-calibration"
            and notice.axis == f"matte_{element}_mole_fraction"
        )
        assert component.calibrated_interval == (0.0, 0.0)
        assert component.distance_beyond_interval == pytest.approx(0.02)
    assert "ideal-matte-activity" in notices


def test_sulfur_split_identity_and_mol_inventory_conversion():
    T, log_fo2, log_fs2 = 1673.0, -4.0, -30.0
    oxygen, cations, formula_moles = _fractions(_MORB)
    result = evaluate_sulfur(_MORB, T, log10_fO2=log_fo2, log10_fS2=log_fs2)
    log_cs2 = result.log10_sulfide_capacity_wt_percent
    log_cs6 = result.log10_sulfate_capacity_wt_percent
    expected_ratio = 10.0 ** (log_cs6 - log_cs2 + 2 * log_fo2)
    assert result.dissolved_s6_plus_ppm / result.dissolved_s2_minus_ppm == pytest.approx(
        expected_ratio
    )
    assert result.dissolved_total_s_ppm == pytest.approx(
        result.dissolved_s2_minus_ppm + result.dissolved_s6_plus_ppm
    )

    mean_molar_mass = sum(_MORB.values()) / sum(formula_moles.values())
    input_mol = 0.001
    by_mol = evaluate_sulfur(
        _MORB, T, log10_fO2=log_fo2,
        total_s_mol_per_mol_melt=input_mol,
    )
    expected_ppm = input_mol * 32.065 / mean_molar_mass * 1.0e6
    assert by_mol.log10_fS2 is not None
    assert by_mol.dissolved_total_s_ppm == pytest.approx(expected_ppm)
    assert by_mol.dissolved_total_s_mol_per_mol_melt == pytest.approx(input_mol)


def test_inventory_saturates_at_scss_and_reports_excess():
    base = evaluate_sulfur(_MORB, 1673.0, log10_fO2=-8.0, total_s_ppm=100.0)
    assert "oxygen-balance-unavailable" not in {
        notice.code for notice in base.notices
    }
    scss = base.scss_smythe_2017_ppm
    assert scss is not None
    low = evaluate_sulfur(_MORB, 1673.0, log10_fO2=-8.0, total_s_ppm=0.5 * scss)
    high = evaluate_sulfur(_MORB, 1673.0, log10_fO2=-8.0, total_s_ppm=1.5 * scss)
    assert low.saturation_state == "undersaturated"
    assert low.dissolved_total_s_ppm == pytest.approx(0.5 * scss)
    assert low.excess_s_ppm == 0.0
    assert high.saturation_state == "saturated"
    assert high.dissolved_s2_minus_ppm == pytest.approx(scss)
    assert high.dissolved_total_s_ppm == pytest.approx(
        high.dissolved_s2_minus_ppm + high.dissolved_s6_plus_ppm
    )
    assert high.dissolved_total_s_ppm == pytest.approx(_saturation_inventory(high))
    assert high.excess_s_ppm == pytest.approx(
        1.5 * scss - _saturation_inventory(high)
    )
    assert high.dissolved_total_s_ppm + high.excess_s_ppm == pytest.approx(1.5 * scss)
    _assert_fugacity_reproduces_dissolved_species(high)
    assert "sulfur-excess-reported-to-sulfide-liquid" in {
        notice.code for notice in high.notices
    }

    fixed_fugacity = evaluate_sulfur(
        _MORB, 1673.0, log10_fO2=-8.0, log10_fS2=0.0,
    )
    assert fixed_fugacity.saturation_state == "saturated"
    assert fixed_fugacity.excess_s_ppm > 0.0
    assert "matte-excess-is-capacity-overrun" in {
        notice.code for notice in fixed_fugacity.notices
    }


def test_lance_inventory_below_and_above_saturation_inventory():
    below = evaluate_sulfur(
        _MORB, 1673.0, log10_fO2=-0.7, total_s_ppm=60000.0,
    )
    below_codes = {notice.code for notice in below.notices}

    assert 60000.0 < _saturation_inventory(below)
    assert below.dissolved_s6_plus_ppm > below.dissolved_s2_minus_ppm
    assert below.dissolved_s2_minus_ppm < below.scss_smythe_2017_ppm
    assert below.saturation_state == "undersaturated"
    assert below.dissolved_total_s_ppm == pytest.approx(60000.0)
    assert below.excess_s_ppm == 0.0
    assert "sulfur-excess-reported-to-sulfide-liquid" not in below_codes
    assert "sulfate-saturation-and-so2-degassing-not-modelled" in below_codes
    assert "reduced-gas-fugacity-is-caller-input" not in below_codes
    assert below.dissolved_total_s_ppm + below.excess_s_ppm == pytest.approx(60000.0)
    _assert_fugacity_reproduces_dissolved_species(below)

    above_inventory = 2.0 * _saturation_inventory(below)
    above = evaluate_sulfur(
        _MORB, 1673.0, log10_fO2=-0.7, total_s_ppm=above_inventory,
    )
    above_codes = {notice.code for notice in above.notices}
    assert above.saturation_state == "saturated"
    assert above.dissolved_s2_minus_ppm == pytest.approx(above.scss_smythe_2017_ppm)
    assert above.dissolved_s6_plus_ppm > above.dissolved_s2_minus_ppm
    assert above.dissolved_total_s_ppm == pytest.approx(_saturation_inventory(above))
    assert above.excess_s_ppm == pytest.approx(above_inventory - above.dissolved_total_s_ppm)
    assert above.dissolved_total_s_ppm + above.excess_s_ppm == pytest.approx(above_inventory)
    assert "sulfur-excess-reported-to-sulfide-liquid" in above_codes
    assert "sulfate-saturation-and-so2-degassing-not-modelled" in above_codes
    _assert_fugacity_reproduces_dissolved_species(above)

    # The review case has S6+ > S2- while the unconstrained S2- inventory
    # exceeds SCSS. The full excess beyond S_max still goes to the sulfide tap.
    nmorb_inventory = 60000.0
    nmorb_max = evaluate_sulfur(
        _MORB, 1673.0, log10_fO2=-5.0, total_s_ppm=nmorb_inventory,
    )
    nmorb_codes = {notice.code for notice in nmorb_max.notices}
    assert nmorb_inventory > _saturation_inventory(nmorb_max)
    assert nmorb_max.saturation_state == "saturated"
    assert nmorb_max.dissolved_s2_minus_ppm == pytest.approx(
        nmorb_max.scss_smythe_2017_ppm
    )
    assert nmorb_max.dissolved_s6_plus_ppm > nmorb_max.dissolved_s2_minus_ppm
    assert nmorb_max.excess_s_ppm == pytest.approx(
        nmorb_inventory - _saturation_inventory(nmorb_max)
    )
    assert nmorb_max.dissolved_total_s_ppm + nmorb_max.excess_s_ppm == pytest.approx(
        nmorb_inventory
    )
    assert "sulfur-excess-reported-to-sulfide-liquid" in nmorb_codes
    assert "sulfate-saturation-and-so2-degassing-not-modelled" in nmorb_codes
    _assert_fugacity_reproduces_dissolved_species(nmorb_max)

    oxidizing_capacity_overrun = evaluate_sulfur(
        _MORB, 1673.0, log10_fO2=-0.7, log10_fS2=8.0,
    )
    overrun_codes = {notice.code for notice in oxidizing_capacity_overrun.notices}
    assert (
        oxidizing_capacity_overrun.dissolved_s6_plus_ppm
        > oxidizing_capacity_overrun.dissolved_s2_minus_ppm
    )
    assert oxidizing_capacity_overrun.excess_s_ppm > 0.0
    assert "sulfur-excess-reported-to-sulfide-liquid" in overrun_codes
    assert "sulfate-saturation-and-so2-degassing-not-modelled" in overrun_codes
    assert "sulfur-excess-unpartitioned-sulfate-saturation-not-modelled" not in overrun_codes


def test_intrinsic_oxygen_root_and_missing_or_conflicting_sulfur_inputs():
    no_inventory = evaluate_sulfur(_MORB, 1773.0)
    assert no_inventory.log10_fO2 is not None
    assert no_inventory.oxygen_source == "openimcc-intrinsic-oxygen-balance"
    assert no_inventory.dissolved_total_s_ppm is None
    assert no_inventory.log10_sulfide_capacity_wt_percent is None
    assert no_inventory.scss_smythe_2017_ppm is None
    assert "sulfur-input-missing" in {notice.code for notice in no_inventory.notices}

    fixed_s2 = evaluate_sulfur(_MORB, 1773.0, log10_fS2=-10.0)
    assert fixed_s2.log10_fO2 is not None
    assert fixed_s2.dissolved_total_s_ppm is not None

    inventory = evaluate_sulfur(_MORB, 1773.0, total_s_ppm=500.0)
    assert inventory.log10_fO2 is not None
    assert inventory.log10_fS2 is not None
    assert inventory.dissolved_total_s_ppm == pytest.approx(500.0)

    conflict = evaluate_sulfur(
        _MORB, 1773.0, log10_fO2=-8.0, log10_fS2=-10.0, total_s_ppm=500.0,
    )
    assert conflict.dissolved_total_s_ppm is None
    assert conflict.log10_sulfide_capacity_wt_percent is None
    assert conflict.scss_smythe_2017_ppm is None
    assert "sulfur-input-conflict" in {notice.code for notice in conflict.notices}

    unit_conflict = evaluate_sulfur(
        _MORB, 1773.0, log10_fO2=-8.0,
        total_s_ppm=500.0, total_s_mol_per_mol_melt=0.001,
    )
    assert unit_conflict.dissolved_total_s_ppm is None
    assert "sulfur-input-conflict" in {notice.code for notice in unit_conflict.notices}


def test_intrinsic_saturated_inventory_solves_oxygen_and_saturation_together():
    T = 1673.0
    base = evaluate_sulfur(_MORB, T, log10_fO2=-8.0, total_s_ppm=100.0)
    assert base.scss_smythe_2017_ppm is not None
    inventory = 1.5 * base.scss_smythe_2017_ppm
    result = evaluate_sulfur(_MORB, T, total_s_ppm=inventory)

    assert result.oxygen_source == "openimcc-intrinsic-oxygen-balance"
    assert result.log10_fO2 is not None
    assert result.log10_fS2 is not None
    assert result.saturation_state == "saturated"
    assert result.scss_smythe_2017_ppm is not None
    assert result.log10_sulfide_capacity_wt_percent is not None
    assert result.dissolved_s2_minus_ppm == pytest.approx(
        result.scss_smythe_2017_ppm
    )
    assert result.dissolved_total_s_ppm + result.excess_s_ppm == pytest.approx(
        inventory
    )
    expected_log_fs2 = result.log10_fO2 + 2.0 * (
        math.log10(result.scss_smythe_2017_ppm)
        - 4.0 - result.log10_sulfide_capacity_wt_percent
    )
    assert result.log10_fS2 == pytest.approx(expected_log_fs2, abs=1.0e-10)

    _oxygen, _cations, formula_moles = _fractions(_MORB)
    melt_result = evaluate(
        formula_moles,
        T,
        basis_type="mol",
        allow_extrapolation=True,
        allow_out_of_envelope=True,
    )
    activities = {
        name: melt_result.activity(name)
        for name in melt_result.parent_oxides
        if melt_result.activity(name) > 0.0
    }
    activities["S2"] = 10.0**result.log10_fS2
    p_o2, _gas_result, _diagnostics = evaluate_gas_oxygen_balance(
        activities,
        T,
        load_gas_datapack(),
        allow_extrapolation=True,
    )
    assert math.log10(p_o2) == pytest.approx(result.log10_fO2, abs=1.0e-7)


def test_fixed_oxygen_inventory_does_not_run_an_oxygen_root(monkeypatch):
    import openimcc.gas as gas_module

    def unavailable(*_args, **_kwargs):
        raise AssertionError("fixed fO2 must not run the gas oxygen-balance solver")

    monkeypatch.setattr(gas_module, "evaluate_gas_oxygen_balance", unavailable)
    result = evaluate_sulfur(
        _MORB, 1673.0, log10_fO2=-8.0, total_s_ppm=100.0,
    )

    assert result.dissolved_total_s_ppm == pytest.approx(100.0)
    assert "oxygen-balance-unavailable" not in {
        notice.code for notice in result.notices
    }
    assert "reduced-gas-fugacity-is-caller-input" not in {
        notice.code for notice in result.notices
    }


def test_comparison_models_reproduce_printed_parameters():
    T, log_fo2, log_fs2 = 1673.0, -8.0, -10.0
    _oxygen, cations, _ = _fractions(_MORB)
    result = evaluate_sulfur(
        _MORB, T, log10_fO2=log_fo2, log10_fS2=log_fs2,
        melt_pack=load_datapack(),
    )

    # O'Neill & Mavrogenes (2002) coefficients as printed in Smythe et al.
    # (2017), Eq. 12 / Table 2, p. 799; single-cation fractions at 1400 C.
    coeff = {
        "Si": -27561, "Ti": -11220, "Al": -18450, "Mg": -13970,
        "Ca": -7831, "Fe": -34274, "Na": -13247, "K": -29015,
    }
    ln_c2 = 9.087 + sum(coeff[key] * cations.get(key, 0.0) for key in coeff) / T
    ln_c2 += 116568 * cations["Si"] * cations["Fe"] / T
    assert result.comparison_band.sulfide_capacity_log10_ppm == pytest.approx(
        ln_c2 / math.log(10.0)
    )
    assert result.comparison_band.sulfide_s2_ppm == pytest.approx(
        math.exp(ln_c2 + 0.5 * math.log(10.0) * (log_fs2 - log_fo2))
    )

    # O'Neill & Mavrogenes (2022), sulfate-capacity fit in GCA 334,
    # pp. 368-382: printed ln capacity relation, ppm S6+.
    gas = load_gas_datapack()
    g_s2 = species_thermo("S2", "g", T, gas).G_J_mol
    g_o2 = species_thermo("O2", "g", T, gas).G_J_mol
    g_so3 = species_thermo("SO3", "g", T, gas).G_J_mol
    log_fso3 = 0.5 * (
        -(2 * g_so3 - g_s2 - 3 * g_o2) / (8.31446261815324 * T * math.log(10))
        + log_fs2 + 3 * log_fo2
    )
    ln_c6 = -8.02 + (
        21100 + 44000 * cations.get("Na", 0.0)
        + 18700 * cations.get("Mg", 0.0) + 4300 * cations.get("Al", 0.0)
        + 44200 * cations.get("K", 0.0) + 35600 * cations.get("Ca", 0.0)
        + 12600 * cations.get("Mn", 0.0) + 16500 * cations.get("Fe", 0.0)
    ) / T
    assert result.comparison_band.sulfate_s6_ppm == pytest.approx(
        math.exp(ln_c6 + math.log(10.0) * log_fso3)
    )
    _oxygen, _cations, formula_moles = _fractions(_MORB)
    melt_result = evaluate(
        formula_moles, T, load_datapack(), basis_type="mol",
        allow_extrapolation=True, allow_out_of_envelope=True,
    )
    # O'Neill (2021), Chapter 10 SCSS relation, with the FeO-FeS term from
    # Smythe et al. (2017), Eq. 12 / Table 2, p. 799.
    ln_delta_g_over_rt = 14695.0 / T - 9.656 + 1.02 * math.log(T)
    expected_oneill = math.exp(
        ln_delta_g_over_rt
        + math.log(10.0 ** (result.log10_sulfide_capacity_wt_percent + 4.0))
        + math.log(result.sulfide_liquid_x_fes)
        - math.log(melt_result.activity("FeO"))
        - 269.40 * (1.0e-4) / T
    )
    assert result.scss_oneill_2021_ppm == pytest.approx(expected_oneill)


def test_calibration_notices_report_axis_interval_and_distance():
    outside = {
        "SiO2": 80.0, "TiO2": 10.0, "Al2O3": 1.0, "FeO": 0.1,
        "MgO": 8.0, "CaO": 0.0, "Na2O": 0.0,
        "P2O5": 0.3, "NiO": 0.3, "Cr2O3": 0.3,
    }
    result = evaluate_sulfur(
        outside, 2500.0, log10_fO2=-20.0, log10_fS2=0.0,
        total_pressure_bar=1.0e-7,
    )
    notices = {(notice.code, notice.model, notice.axis): notice for notice in result.notices}
    expected = (
        ("capacity-temperature-outside-calibration", "temperature_K"),
        ("scss-temperature-outside-calibration", "temperature_K"),
        ("capacity-composition-outside-calibration", "SiO2_wt_percent"),
        ("capacity-composition-outside-calibration", "CaO_wt_percent"),
        ("capacity-composition-outside-calibration", "TiO2_wt_percent"),
        ("capacity-composition-outside-calibration", "Na2O_wt_percent"),
        ("scss-composition-outside-calibration", "FeO_wt_percent"),
        ("capacity-composition-component-outside-calibration", "P2O5_wt_percent"),
        ("capacity-composition-component-outside-calibration", "NiO_wt_percent"),
        ("capacity-composition-component-outside-calibration", "Cr2O3_wt_percent"),
        ("pressure-below-calibration", "total_pressure_bar"),
        ("capacity-prediction-above-calibrated-maximum", "capacity_predicted_S_ppm"),
    )
    for code, axis in expected:
        matches = [notice for (notice_code, _model, notice_axis), notice in notices.items()
                   if notice_code == code and notice_axis == axis]
        assert matches, (code, axis)
        assert matches[0].value is not None
        assert matches[0].calibrated_interval is not None
        assert matches[0].distance_beyond_interval is not None
    pressure_notice = next(n for n in result.notices if n.code == "pressure-below-calibration")
    assert pressure_notice.distance_beyond_interval == pytest.approx(7.0)
    assert result.notices[0].distance_beyond_interval == pytest.approx(727.0)
    higher_feo = {
        "SiO2": 45.0, "Al2O3": 10.0, "FeO": 30.0, "MgO": 5.0,
        "CaO": 5.0, "Na2O": 2.0, "K2O": 1.0, "TiO2": 1.0, "MnO": 1.0,
    }
    high_pressure = evaluate_sulfur(
        higher_feo, 1800.0, log10_fO2=-8.0, log10_fS2=-10.0,
        total_pressure_bar=300000.0,
    )
    assert any(
        n.code == "capacity-composition-outside-calibration" and n.axis == "FeO_wt_percent"
        for n in high_pressure.notices
    )
    assert "capacity-pressure-outside-calibration" in {
        n.code for n in high_pressure.notices
    }
    assert "scss-pressure-outside-calibration" in {
        n.code for n in high_pressure.notices
    }
    capacity_pressure = next(
        n for n in high_pressure.notices
        if n.code == "capacity-pressure-outside-calibration"
    )
    assert capacity_pressure.axis == "total_pressure_bar"
    assert capacity_pressure.calibrated_interval == (1.0, 1.0)
    assert capacity_pressure.distance_beyond_interval == pytest.approx(math.log10(300000.0))
    scss_pressure = next(
        n for n in high_pressure.notices
        if n.code == "scss-pressure-outside-calibration"
    )
    assert scss_pressure.calibrated_interval == (1.0, 240000.0)
    assert scss_pressure.distance_beyond_interval == pytest.approx(60000.0)
    assert any(
        n.code == "om2002-temperature-outside-calibration"
        for n in evaluate_sulfur(_MORB, 1773.0, log10_fO2=-8, log10_fS2=-10).notices
    )
    for temperature, distance in ((2073.0, 300.0), (2473.0, 700.0)):
        om22 = next(
            n for n in evaluate_sulfur(
                _MORB, temperature, log10_fO2=-8, log10_fS2=-10,
            ).notices
            if n.code == "om2022-temperature-outside-calibration"
        )
        assert om22.axis == "temperature_K"
        assert om22.calibrated_interval == (1473.0, 1773.0)
        assert om22.distance_beyond_interval == pytest.approx(distance)


def test_capacity_calibration_notice_survives_scss_cap():
    result = evaluate_sulfur(_MORB, 1673.0, log10_fO2=-8.0, log10_fS2=2.0)
    assert result.capacity_predicted_total_s_ppm > 3500.0
    assert result.dissolved_s2_minus_ppm == pytest.approx(result.scss_smythe_2017_ppm)
    assert result.dissolved_total_s_ppm == pytest.approx(
        result.dissolved_s2_minus_ppm + result.dissolved_s6_plus_ppm
    )
    assert result.dissolved_total_s_ppm < 3500.0
    notice = next(
        n for n in result.notices
        if n.code == "capacity-prediction-above-calibrated-maximum"
        and n.model == "Boulliung-Wood-2023-Eq-7"
    )
    assert notice.axis == "capacity_predicted_S_ppm"
    assert notice.calibrated_interval == (0.0, 3500.0)
    assert notice.distance_beyond_interval == pytest.approx(notice.value - 3500.0)


def test_default_openimcc_solve_without_sulfur_call_pins_v5():
    result = evaluate(
        _MORB, 1800.0, basis_type="wt", allow_extrapolation=True,
        allow_out_of_envelope=True,
    )
    # This is the D066 v5 default solve; the five reviewed association A
    # refits change its activity and bulk root independently of sulfur.
    assert result.activity("FeO") == pytest.approx(0.05700402442549344)
    assert result.D == pytest.approx(2.1787222992077693)


def test_excess_melt_disclosures_survive_sulfur_evaluation(tmp_path):
    source = Path(__file__).parents[1] / "src/openimcc/data/packs/imcc-sf04-d066-v5.json"
    pack_data = json.loads(source.read_text(encoding="utf-8"))
    pack_data["model_id"] = "IMCC-SF04-D066-EXCESS-CANDIDATE"
    pack_data["excess_gibbs"] = {
        "terms": [{
            "basis": "analytical_parent",
            "components": ["Na2O", "CaO"],
            "W_H_J_mol": -60_000.0,
            "W_S_J_mol_K": 0.0,
            "provenance": {"source": "sulfur disclosure regression"},
            "calibration_role": "sulfur melt-activity regression",
            "valid_composition": {"Na2O": [0.0, 1.0], "CaO": [0.0, 1.0]},
            "T_range_K": [1900.0, 2000.0],
            "fit_status": "empirical-fit",
        }],
    }
    candidate_path = tmp_path / "sulfur-excess-candidate.json"
    candidate_path.write_text(json.dumps(pack_data), encoding="utf-8")
    candidate = load_datapack(candidate_path)

    result = evaluate_sulfur(
        _MORB, 1673.0, log10_fO2=-8.0, log10_fS2=-10.0,
        melt_pack=candidate,
    )

    assert result.scss_oneill_2021_ppm == pytest.approx(5712.502825957212)
    candidate_notice = next(
        notice for notice in result.notices
        if notice.code == "excess-gibbs-candidate"
    )
    assert "Na2O–CaO (analytical_parent; empirical-fit)" in candidate_notice.message
    range_notice = next(
        notice for notice in result.notices
        if notice.code == "excess-gibbs-calibration-range"
    )
    assert "Na2O–CaO (analytical_parent)" in range_notice.message
    assert "T=1673 K below lower bound 1900 K" in range_notice.message
