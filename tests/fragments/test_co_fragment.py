import json
import math
from pathlib import Path

import pytest

from openimcc import evaluate, load_datapack
from openimcc._constants import MOLAR_MASS_G_MOL
from openimcc.co_activity import evaluate_co_activity
from openimcc.gas import evaluate_gas, load_gas_datapack


ROOT = Path(__file__).resolve().parents[2]
FRAGMENT = ROOT / "src/openimcc/data/fragments/co/rows.json"


def _read_fragment():
    return json.loads(FRAGMENT.read_text(encoding="utf-8"))


def test_activity_row_matches_printed_holzheid_anchor() -> None:
    fragment = _read_fragment()
    row = fragment["rows"][0]

    assert row["gamma_mean"] == 1.51
    assert row["gamma_reported_spread"] == 0.28
    assert row["experiment_count"] == 76
    assert row["domain"]["temperature_K"] == [1573.15, 1873.15]
    assert row["domain"]["log10_fO2_over_IW"] == [-3.0, 1.5]
    assert row["domain"]["FeO_wt_pct"] == [0.0, 12.0]
    assert row["domain"]["MgO_wt_pct"] == [4.0, 30.0]
    assert row["citation"]["doi"] == "10.1016/S0009-2541(97)00030-2"
    assert row["citation"]["pages"] == "21-38"

    # Independent hand calculation from the source's printed coefficient.
    expected_gamma = 1.51
    expected_spread = 0.28
    result = evaluate_co_activity({"SiO2": 0.999, "CoO": 0.001}, 1700.0)
    assert result.x_coo == pytest.approx(0.001)
    assert result.activity == pytest.approx(expected_gamma * 0.001)
    assert result.gamma_range == pytest.approx(
        (expected_gamma - expected_spread, expected_gamma + expected_spread)
    )
    assert result.activity_range == pytest.approx((0.00123, 0.00179))
    assert result.notice.code == "imcc_co_activity_domain_notice"


def test_coo_liquid_standard_matches_tsiv_printed_phase_pair() -> None:
    standard = _read_fragment()["standards"][0]
    printed = standard["printed_values"]
    crystal = printed["crystal_at_fusion"]
    liquid = printed["liquid_at_fusion"]

    assert standard["source_reference"] == "TSIV-1334"
    assert printed["fusion_temperature_K"] == 2090.0
    assert printed["fusion_temperature_uncertainty_K"] == 10.0
    assert printed["enthalpy_298_minus_H0_kJ_mol"] == 9.449
    assert printed["formation_enthalpy_298_kJ_mol"] == -238.0
    assert printed["entropy_298_J_mol_K"] == 52.83
    assert crystal["F_J_mol_K"] == liquid["F_J_mol_K"] == 108.388
    assert liquid["H_minus_H0_kJ_mol"] - crystal["H_minus_H0_kJ_mol"] == 40.0
    delta_s_fus = liquid["S_J_mol_K"] - crystal["S_J_mol_K"]
    assert delta_s_fus == pytest.approx(19.138)
    assert 40_000.0 / printed["fusion_temperature_K"] == pytest.approx(
        delta_s_fus, abs=0.002
    )
    assert liquid["Cp_J_mol_K"] == 67.0
    assert printed["fusion_enthalpy_source_status"] == "estimated"
    assert "fusion_enthalpy_is_estimated_by_source" in standard["flags"]
    assert "liquid_heat_capacity_is_estimated_by_source" in standard["flags"]


@pytest.mark.parametrize(
    ("temperature", "oxygen_offset", "composition", "expected_axis"),
    [
        (1500.0, 0.0, {"SiO2": 0.999, "CoO": 0.001}, "temperature_outside_1573.15_1873.15_K"),
        (1700.0, 2.0, {"SiO2": 0.999, "CoO": 0.001}, "oxygen_fugacity_outside_IW_minus_3_to_plus_1_5"),
        (1700.0, 0.0, {"SiO2": 0.749, "FeO": 0.25, "CoO": 0.001}, "FeO_outside_0_to_12_wt_pct"),
        (1700.0, 0.0, {"SiO2": 0.70, "MgO": 0.29, "CoO": 0.01}, "MgO_above_20_wt_pct_source_reports_gamma_increase"),
    ],
)
def test_activity_notice_flags_source_domain_limits(
    temperature, oxygen_offset, composition, expected_axis
) -> None:
    result = evaluate_co_activity(
        composition, temperature, log10_fO2_over_IW=oxygen_offset
    )
    assert expected_axis in result.notice.violated_axes


def test_co_fragment_with_default_melt_and_gas_preserves_non_co_outputs() -> None:
    # A basalt-like weight recipe, converted to oxide formula moles for both
    # the default melt solve and the independent dilute Co helper.
    weight_pct = {
        "SiO2": 49.0,
        "Al2O3": 17.0,
        "CaO": 12.0,
        "MgO": 9.0,
        "FeO": 9.0,
        "Na2O": 2.0,
        "K2O": 0.5,
        "TiO2": 1.0,
        "CoO": 0.001,
    }
    moles = {
        component: weight / MOLAR_MASS_G_MOL[component]
        for component, weight in weight_pct.items()
    }
    temperature = 1800.0
    oxide_moles = {name: value for name, value in moles.items() if name != "CoO"}
    melt = evaluate(oxide_moles, temperature)
    default_parents = load_datapack().kernel_datapack.parent_oxides
    major_activities = dict(zip(default_parents, melt.parent_activity))
    co = evaluate_co_activity(moles, temperature)

    gas_pack = load_gas_datapack()
    all_parents = (*default_parents, "CoO")
    baseline = evaluate_gas(
        major_activities,
        temperature,
        1.0e-9,
        gas_pack,
        parent_oxides=all_parents,
    )
    with_co = evaluate_gas(
        {**major_activities, "CoO": co.activity},
        temperature,
        1.0e-9,
        gas_pack,
        parent_oxides=all_parents,
    )

    co_pressures = {name: with_co[name] for name in ("Co", "CoO")}
    assert all(math.isfinite(value) and value > 0.0 for value in co_pressures.values())
    assert all(
        baseline[name] == with_co[name]
        for name in baseline
        if name not in {"Co", "CoO"}
    )


def test_co_activity_rejects_missing_or_invalid_coo_composition() -> None:
    with pytest.raises(ValueError, match="positive total moles and CoO"):
        evaluate_co_activity({"SiO2": 1.0}, 1700.0)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        evaluate_co_activity({"SiO2": 1.0, "CoO": -0.1}, 1700.0)
