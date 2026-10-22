import math

import pytest

from openimcc import (
    P2O5FeIIINotice,
    PGasPressureNotice,
    engine_binding_identity,
    evaluate,
    evaluate_p2o5_activity,
    load_datapack,
)
from openimcc.gas import evaluate_gas, load_gas_datapack, species_thermo


def test_suito_regression_and_explicit_atm_to_bar_standard_conversion() -> None:
    result = evaluate_p2o5_activity(
        {"CaO": 40, "MgO": 8, "FeO": 12, "SiO2": 39, "P2O5": 1}, 1873
    )
    # Direct Eq. 14 hand calculation on the listed oxide mole fractions.
    x_p = 1 / 100
    log_gamma = -1.02 * (23*.40 + 17*.08 + 8*.12 - 26*.01) - 22990/1873 + 9.490
    source_g = -1534480 + 506.26*1873
    pack_g = -1616372 + 539*1873 + 3.5*8.31446261815324*1873*math.log(1.01325)
    expected = log_gamma + math.log10(x_p) + (source_g-pack_g)/(8.31446261815324*1873*math.log(10))
    assert result.log10_activity == pytest.approx(expected)
    assert result.activity == pytest.approx(10**expected)
    assert result.activity == pytest.approx(result.gamma * result.x_p2o5)
    assert result.notice.code == "imcc_p2o5_activity_domain_notice"


@pytest.mark.parametrize(
    ("composition", "temperature", "axis"),
    [
        ({"P2O5": 1, "CaO": 99, "SiO2": 1, "MgO": 1}, 1700, "temperature_outside_1823_1923_K"),
        ({"P2O5": 1, "CaO": 99, "SiO2": 1, "MgO": 1, "TiO2": 1}, 1873, "TiO2_present_without_regression_term"),
        ({"P2O5": 1, "CaO": 99, "MgO": 1}, 1873, "SiO2_below_calibration_family"),
        ({"P2O5": 1, "CaO": 99, "SiO2": 1}, 1873, "MgO_absent"),
        ({"P2O5": 6, "CaO": 99, "SiO2": 1, "MgO": 1}, 1873, "P2O5_above_calibrated_range"),
    ],
)
def test_each_domain_axis_appears_in_stable_notice(composition, temperature, axis) -> None:
    result = evaluate_p2o5_activity(composition, temperature)
    assert axis in result.notice.violated_axes
    assert result.notice.regression == "Suito-Inoue-1984-Eq-14-no-Mn"
    assert result.notice.band_width_dex > 0


@pytest.mark.parametrize("oxygen", [1e-4, 1e-18])
def test_p_gas_mass_action_holds_at_lance_and_reduction_oxygen(oxygen: float) -> None:
    pack = load_gas_datapack()
    T, activity = 2000.0, 1e-18
    gas = evaluate_gas({"P2O5": activity}, T, oxygen, pack,
                       gas_species=("P2", "P4", "PO", "PO2", "P4O6", "P4O10"))
    p2o5_g = species_thermo("P2O5", "l", T, pack).G_J_mol
    oxygen_g = species_thermo("O2", "g", T, pack).G_J_mol
    for species, n_gas, n_o2 in (("P2", 1, 2.5), ("P4", .5, 2.5), ("PO", 2, 1.5),
                                 ("PO2", 2, .5), ("P4O6", .5, 1), ("P4O10", .5, 0)):
        gas_g = species_thermo(species, "g", T, pack).G_J_mol
        # Table stoichiometries are per one P2O5 parent; n_gas is the
        # possibly fractional gas coefficient, not the parent coefficient.
        delta_g = n_gas*gas_g + n_o2*oxygen_g - p2o5_g
        expected = (-delta_g/(8.31446261815324*T*math.log(10)) + math.log10(activity)
                    - n_o2*math.log10(oxygen))/n_gas
        assert math.log10(gas[species]) == pytest.approx(expected, abs=1e-9)


def test_carbon_reduction_p_gas_over_one_bar_has_typed_notice() -> None:
    pack = load_gas_datapack()
    gas = evaluate_gas(
        {"P2O5": 1e-18}, 2000.0, 1e-15, pack,
        gas_species=("P", "P2", "P4", "PO", "PO2", "P4O6", "P4O10"),
    )
    notice, = gas.notices
    assert isinstance(notice, PGasPressureNotice)
    assert notice.code == "p_gas_exceeds_one_bar_complete_volatilisation"
    assert "essentially completely" in notice.message
    assert "fixed-composition dilute equilibrium is not physical" in notice.message
    assert notice.total_pressure_bar > 1
    assert any(gas[name] > 1 for name in gas)
    assert all(gas.domain_flags[name] for name in gas)


def test_p2o5_parent_participates_in_existing_oxygen_balance() -> None:
    from openimcc.gas import evaluate_gas_oxygen_balance

    pressure, gases, _ = evaluate_gas_oxygen_balance(
        {"P2O5": 1e-15}, 2000.0, load_gas_datapack(),
        parent_oxides=("P2O5",),
    )
    assert 0 < pressure < 1
    assert gases["P2"] > 0 and gases["P4"] > 0


def test_default_pack_identity_and_p_free_evaluation_are_unchanged() -> None:
    pack = load_datapack()
    gas_pack = load_gas_datapack()
    composition = {"SiO2": 55, "MgO": 8, "FeO": 10, "CaO": 12,
                   "Al2O3": 10, "TiO2": 2, "Na2O": 2, "K2O": 1}
    before_identity = engine_binding_identity(pack, gas_pack)
    before = evaluate(composition, 1800.0, pack)
    evaluate_p2o5_activity({"CaO": 1, "P2O5": 0.01}, 1800.0)
    after = evaluate(composition, 1800.0, pack)
    assert engine_binding_identity(pack, gas_pack) == before_identity
    assert after.parent_oxides == before.parent_oxides
    assert after.parent_activity == pytest.approx(before.parent_activity, rel=0, abs=0)


def test_fe2o3_is_folded_into_feo_equivalent_with_feiii_notice() -> None:
    ferric = evaluate_p2o5_activity(
        {"CaO": 40, "MgO": 8, "Fe2O3": 6, "SiO2": 39, "P2O5": 1}, 1873
    )
    equivalent = evaluate_p2o5_activity(
        {"CaO": 40, "MgO": 8, "FeOeq": 12, "SiO2": 39, "P2O5": 1}, 1873
    )
    assert ferric.activity == pytest.approx(equivalent.activity)
    assert ferric.additional_notices == (P2O5FeIIINotice(),)
    assert ferric.additional_notices[0].code == "p2o5_feiii_phosphate_chemistry_not_modelled"
