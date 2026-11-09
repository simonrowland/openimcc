import math
import struct

import pytest

from openimcc import evaluate, load_datapack
from openimcc._constants import MOLAR_MASS_G_MOL, R_J_MOL_K
from openimcc.activity_fragments import load_activity_fragment
from openimcc.gas import evaluate_gas, load_gas_datapack
from openimcc.p2o5 import evaluate_p2o5_activity


def test_fragment_rows_match_printed_equations_and_standard_anchors() -> None:
    fragment = load_activity_fragment("P")
    central = fragment["rows"]["suito_inoue_1984_eq14_no_mn"]
    assert central["coefficients"] == {
        "bracket_multiplier": -1.02,
        "mole_fraction_terms": {"CaO": 23.0, "MgO": 17.0, "FeO": 8.0, "P2O5": -26.0},
        "inverse_temperature_k": -22990.0,
        "constant": 9.490,
    }
    assert central["flags"] == [
        "opt_in_empirical_activity_helper",
        "provisional_old_liquid_standard",
        "no_mn_term",
        "not_fitted_to_battery_rows",
    ]
    assert fragment["citations"]["suito-inoue-1984"]["doi"] == (
        "10.2355/isijinternational1966.24.40"
    )
    assert "Eq. 14" in fragment["citations"]["suito-inoue-1984"]["locator"]
    assert "provisional_estimate" in fragment["standards"][
        "turkdogan_2000_revised_liquid_1atm"
    ]["flags"]
    assert "provisional_revised_standard" in fragment["rows"][
        "turkdogan_2000_eq25"
    ]["flags"]
    for standard in fragment["standards"].values():
        citation = fragment["citations"][standard["citation"]]
        assert citation["doi"]
        assert citation["locator"]
    for row in fragment["rows"].values():
        assert row["citations"]
        for citation_id in row["citations"]:
            citation = fragment["citations"][citation_id]
            assert citation["doi"]
            assert citation["locator"]

    # Hand calculation from Suito and Inoue Eq. 14, p. 45, and the printed
    # standard functions in Turkdogan Eq. 4, p. 964, and Jung et al. Eq. 3.
    T = 1873.0
    composition = {"CaO": 40, "MgO": 8, "FeO": 12, "SiO2": 39, "P2O5": 1}
    x_p = 0.01
    source_log_gamma = (
        -1.02 * (23 * 0.40 + 17 * 0.08 + 8 * 0.12 - 26 * x_p)
        - 22990.0 / T
        + 9.490
    )
    source_g = -1534480.0 + 506.26 * T
    target_g = -1616372.0 + 539.0 * T + 3.5 * R_J_MOL_K * T * math.log(1.01325)
    source_to_target = (source_g - target_g) / (R_J_MOL_K * T * math.log(10.0))
    expected_log_activity = math.log10(x_p) + source_log_gamma + source_to_target
    result = evaluate_p2o5_activity(composition, T)
    assert central["derivation"]["sanity_point"]["log10_gamma_on_source_standard"] == pytest.approx(
        source_log_gamma, abs=1e-12
    )
    assert result.log10_activity == pytest.approx(expected_log_activity, abs=1e-12)
    assert result.gamma == pytest.approx(10 ** (source_log_gamma + source_to_target))
    assert result.activity == pytest.approx(10**expected_log_activity)

    # Eq. 13 uses the older standard; Eq. 25 uses Turkdogan's revised one.
    total_mass_g = sum(
        amount * MOLAR_MASS_G_MOL[oxide] for oxide, amount in composition.items()
    )
    bo_wt_pct = 100.0 * (
        40.0 * MOLAR_MASS_G_MOL["CaO"]
        + 0.33 * 8.0 * MOLAR_MASS_G_MOL["MgO"]
    ) / total_mass_g
    eq13_log_activity = (
        -11.85 - 0.1343 * bo_wt_pct + math.log10(x_p) + source_to_target
    )
    revised_g = -1655480.0 + 571.0 * T
    revised_shift = (revised_g - target_g) / (R_J_MOL_K * T * math.log(10.0))
    eq25_log_activity = -9.84 - 0.142 * bo_wt_pct + math.log10(x_p) + revised_shift
    expected_band = tuple(sorted((eq13_log_activity, eq25_log_activity)))
    assert result.band_log10_activity == pytest.approx(expected_band, abs=1e-12)
    assert result.band_width_dex == pytest.approx(expected_band[1] - expected_band[0])


@pytest.mark.parametrize(
    ("composition", "temperature", "axis"),
    [
        (
            {"CaO": 40, "MgO": 8, "FeO": 12, "SiO2": 39, "P2O5": 0.05},
            1873.15,
            "P2O5_below_0.2_wt_pct_calibrated_range",
        ),
        (
            {"CaO": 40, "MgO": 8, "FeO": 12, "SiO2": 39, "P2O5": 1},
            1873.15,
            "P2O5_in_1_4_wt_pct_calibration_gap",
        ),
        (
            {"CaO": 40, "MgO": 8, "FeO": 12, "SiO2": 39, "P2O5": 2},
            1800.0,
            "P2O5_4_5_wt_pct_outside_1873K_calibration",
        ),
        (
            {"CaO": 40, "MgO": 8, "FeO": 12, "SiO2": 39, "P2O5": 1, "MnO": 1},
            1873.15,
            "MnO_present_without_regression_term",
        ),
        (
            {"CaO": 40, "MgO": 8, "FeO": 12, "SiO2": 39, "P2O5": 1, "Al2O3": 1},
            1873.15,
            "Al2O3_present_without_regression_term",
        ),
    ],
)
def test_activity_notice_reports_uncovered_axes(composition, temperature, axis) -> None:
    result = evaluate_p2o5_activity(composition, temperature)
    assert axis in result.notice.violated_axes
    assert any("provisional" in limitation for limitation in result.notice.limitations)


def test_loaded_fragment_supplies_finite_p_gas_without_changing_other_channels() -> None:
    temperature = 1873.0
    composition_wt_pct = {
        "SiO2": 45.0,
        "MgO": 9.0,
        "FeO": 10.0,
        "CaO": 12.0,
        "Al2O3": 15.0,
        "TiO2": 2.0,
        "Na2O": 2.0,
        "K2O": 1.0,
    }
    melt = evaluate(
        composition_wt_pct,
        temperature,
        load_datapack(),
        basis_type="wt",
        allow_out_of_envelope=True,
    )
    base_activities = dict(zip(melt.parent_oxides, melt.parent_activity, strict=True))
    p_composition_moles = {
        oxide: amount / MOLAR_MASS_G_MOL[oxide]
        for oxide, amount in {**composition_wt_pct, "P2O5": 0.0675}.items()
    }
    p_activity = evaluate_p2o5_activity(p_composition_moles, temperature)

    gas_pack = load_gas_datapack()
    oxygen_fugacity = 1.0e-4
    p_species = ("P", "P2", "P4", "PO", "PO2", "P4O6", "P4O10")
    before = evaluate_gas(
        base_activities,
        temperature,
        oxygen_fugacity,
        gas_pack,
    )
    after = evaluate_gas(
        {**base_activities, "P2O5": p_activity.activity},
        temperature,
        oxygen_fugacity,
        gas_pack,
    )

    assert all(math.isfinite(after[name]) and after[name] > 0.0 for name in p_species)
    for name in (set(before) & set(after)) - set(p_species):
        assert struct.pack("!d", before[name]) == struct.pack("!d", after[name])
