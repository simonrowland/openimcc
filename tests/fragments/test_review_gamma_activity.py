import math
from pathlib import Path

import pytest

from openimcc._constants import MOLAR_MASS_G_MOL, R_J_MOL_K
from openimcc import evaluate, load_datapack
from openimcc.gas import (
    _janaf_gibbs,
    _lamor_gibbs,
    _nearest_interval_row,
    evaluate_gas,
    load_gas_datapack,
)
from openimcc.review_gamma_activity import evaluate_review_gamma_activity
from openimcc.trace_components import (
    TraceActivityEstimate,
    TraceActivityFlag,
    _resolve_trace_activity,
    trace_component_rule,
)


_CASES = (
    ("Sn", "SnO", 1573.0, 5.0, 6.2, 1.0, {"FeO": 1.0, "CaO": 1.0, "SiO2": 2.0}),
    ("Ga", "Ga2O3", 1700.0, 0.030, 0.042, 2.0, {
        "SiO2": 50.34, "Al2O3": 15.40, "MgO": 10.80, "CaO": 23.46,
    }),
    ("In", "In2O3", 1674.0, 0.005, 0.029, 2.0, {
        "SiO2": 50.34, "Al2O3": 15.40, "MgO": 10.80, "CaO": 23.46,
    }),
    ("Cu", "Cu2O", 1573.0, 9.0, 11.0, 2.0, {
        "FeO": 1.0, "CaO": 1.0, "Al2O3": 1.0, "MgO": 1.0, "SiO2": 2.0,
    }),
    ("Ge", "GeO2", 1573.0, 0.24, 1.50, 1.0, {
        "FeO": 1.0, "CaO": 1.0, "MgO": 1.0, "SiO2": 2.0,
    }),
    ("As", "As2O3", 1573.0, 0.03, 0.56, 2.0, {
        "FeO": 1.0, "CaO": 1.0, "SiO2": 2.0,
    }),
    ("Bi", "Bi2O3", 1573.0, 0.3, 0.5, 2.0, {
        "FeO": 1.0, "CaO": 1.0, "SiO2": 2.0,
    }),
)


def _composition_moles(parent: str, major: dict[str, float]) -> dict[str, float]:
    if parent in {"Ga2O3", "In2O3"}:
        element = parent[:-3]
        ppm = 429.0 if element == "Ga" else 0.05
        parent_mass = ppm / 10_000.0 * trace_component_rule(parent).molar_mass_g_mol / (
            2.0 * MOLAR_MASS_G_MOL[element]
        )
        parent_moles = parent_mass / trace_component_rule(parent).molar_mass_g_mol
        major_moles = {
            name: amount / MOLAR_MASS_G_MOL[name]
            for name, amount in major.items()
        }
    else:
        parent_moles = 0.001
        major_moles = dict(major)
    return {**major_moles, parent: parent_moles}


@pytest.mark.parametrize(
    ("element", "parent", "temperature", "gamma_low", "gamma_high", "power", "major"),
    _CASES,
)
def test_printed_gamma_ranges_and_activity_bands_are_independent_literals(
    element, parent, temperature, gamma_low, gamma_high, power, major
) -> None:
    composition = _composition_moles(parent, major)
    result = evaluate_review_gamma_activity(
        element, composition, temperature, parent_oxide=parent
    )

    parent_moles = composition[parent]
    source_units = 1.0 if power == 1.0 else 2.0
    total = math.fsum(composition.values())
    x_source = source_units * parent_moles / (
        total - parent_moles + source_units * parent_moles
    )
    gamma = {
        "Ga": 0.036,
        "In": 0.017,
    }.get(element, math.sqrt(gamma_low * gamma_high))
    assert result.gamma == pytest.approx(gamma)
    assert result.gamma_range == (gamma_low, gamma_high)
    assert result.source_mole_fraction == pytest.approx(x_source)
    assert result.activity == pytest.approx((gamma * x_source) ** power)
    assert result.activity_range == pytest.approx((
        (gamma_low * x_source) ** power,
        (gamma_high * x_source) ** power,
    ))


def test_sn_binary_rows_require_the_matching_binary_composition() -> None:
    fayalite = evaluate_review_gamma_activity(
        "Sn", {"FeO": 1.0, "SiO2": 1.0, "SnO": 0.001}, 1573.0,
        parent_oxide="SnO",
    )
    ferrite = evaluate_review_gamma_activity(
        "Sn", {"FeO": 1.0, "CaO": 1.0, "SnO": 0.001}, 1573.0,
        parent_oxide="SnO",
    )
    fcs = evaluate_review_gamma_activity(
        "Sn", {"FeO": 1.0, "CaO": 1.0, "SiO2": 2.0, "SnO": 0.001}, 1573.0,
        parent_oxide="SnO",
    )

    assert fayalite.row == "SnO-fayalite-binary-1573K"
    assert ferrite.row == "SnO-ferrite-binary-1573K"
    assert fcs.row == "SnO-FCS-1573K"
    assert fayalite.gamma_range == ferrite.gamma_range == (1.0, 2.0)
    assert fcs.gamma_range == (5.0, 6.2)


def test_temperature_selects_cu_row_and_family_only_appears_as_a_flag() -> None:
    cmass = {"CaO": 1.0, "MgO": 1.0, "Al2O3": 1.0, "SiO2": 2.0, "Cu2O": 0.001}
    unrelated = {"Na2O": 1.0, "Cu2O": 0.001}
    expected = evaluate_review_gamma_activity(
        "Cu", cmass, 1800.0, parent_oxide="Cu2O"
    )
    changed_family = evaluate_review_gamma_activity(
        "Cu", unrelated, 1800.0, parent_oxide="Cu2O"
    )

    assert expected.row == changed_family.row == "CuO0.5-1923K"
    assert expected.gamma_range == changed_family.gamma_range == (3.5, 3.5)
    assert "The melt-family label is a screen flag, not the temperature-row selector." in expected.reference


def test_gas_estimate_and_flag_expose_source_fraction_and_endpoint_band() -> None:
    composition = {"FeO": 1.0, "CaO": 1.0, "Al2O3": 1.0, "MgO": 1.0, "SiO2": 2.0, "Cu2O": 0.001}
    rule = trace_component_rule("Cu2O")
    assert rule is not None
    estimate, _, _ = _resolve_trace_activity(
        "Cu2O", 0.001, composition, 1573.0
    )
    assert isinstance(estimate, TraceActivityEstimate)
    assert estimate.activity_range is not None
    assert estimate.x_source is not None
    flag = TraceActivityFlag(estimate, context="gas")
    assert flag.activity_range == estimate.activity_range
    assert flag.x_source == estimate.x_source
    assert f"activity_range=[{estimate.activity_range[0]:.12g}," in str(flag)
    assert f"x_source={estimate.x_source:.12g}" in str(flag)


def test_ga_in_anchors_are_liquid_coefficients_without_a_second_fusion_step() -> None:
    ga = evaluate_review_gamma_activity(
        "Ga", _composition_moles("Ga2O3", _CASES[1][6]), 1700.0,
        parent_oxide="Ga2O3",
    )
    indium = evaluate_review_gamma_activity(
        "In", _composition_moles("In2O3", _CASES[2][6]), 1674.0,
        parent_oxide="In2O3",
    )
    assert ga.gamma == 0.036
    assert ga.gamma_range == (0.030, 0.042)
    assert indium.gamma == 0.017
    assert indium.gamma_range == (0.005, 0.029)
    assert "published x_source anchor 0.0172 at 429 ppm" in ga.reference
    assert "declared formula-unit MO1.5 basis" in ga.reference
    assert "published x_source anchor 6.9e-05 at 0.05 ppm" in indium.reference


def test_bischof_table8_s1_closure_literals_use_the_component_mole_fraction() -> None:
    # Table 8 source x and the printed S1 pressure ratios are independent source
    # literals; do not source any of these expected values from the helper rows.
    source_rows = (
        (1586.4, 0.0172, 0.036, 0.938, 6.9e-5, 0.017, 1.069),
        (1630.8, 0.0172, 0.036, 0.937, 6.9e-5, 0.017, 1.070),
        (1674.2, 0.0172, 0.036, 0.940, 6.9e-5, 0.017, 1.071),
        (1695.4, 0.0172, 0.036, 0.940, 6.9e-5, 0.017, 1.074),
    )
    expected_closures = (
        (0.938, 1.069),
        (0.937, 1.070),
        (0.940, 1.071),
        (0.940, 1.074),
    )
    major_wt = {"SiO2": 50.34, "Al2O3": 15.40, "MgO": 10.80, "CaO": 23.46}

    for source_row, expected in zip(source_rows, expected_closures, strict=True):
        temperature, ga_x, ga_gamma, ga_closure, in_x, in_gamma, in_closure = source_row
        activities = []
        for oxide, x, gamma, power in (
            ("Ga2O3", ga_x, ga_gamma, 2),
            ("In2O3", in_x, in_gamma, 2),
        ):
            major_moles = {
                name: amount / MOLAR_MASS_G_MOL[name]
                for name, amount in major_wt.items()
            }
            major_total = math.fsum(major_moles.values())
            parent_moles = x * major_total / (power * (1.0 - x))
            result = evaluate_review_gamma_activity(
                oxide.removesuffix("2O3"),
                {**major_moles, oxide: parent_moles},
                temperature,
                parent_oxide=oxide,
            )
            assert result.source_mole_fraction == pytest.approx(x, abs=1e-15)
            assert result.gamma == gamma
            activities.append(result.activity)
        assert activities == pytest.approx(
            ((ga_gamma * ga_x) ** 2, (in_gamma * in_x) ** 2), rel=1e-14
        )
        assert (ga_closure, in_closure) == expected


def test_as_parent_wording_and_inactive_ge_v_rows() -> None:
    arsenic = evaluate_review_gamma_activity(
        "As", {"FeO": 1.0, "CaO": 1.0, "SiO2": 2.0, "As2O3": 0.001},
        1573.0, parent_oxide="As2O3",
    )
    assert "pure liquid As2O3" not in arsenic.reference
    assert trace_component_rule("GeO") is None
    assert trace_component_rule("V2O3") is None


def test_bischof_parent_standard_candidate_matches_the_liquid_kems_constants() -> None:
    default_pack = load_gas_datapack()
    candidate_path = (
        Path(__file__).resolve().parents[2]
        / "src/openimcc/data/gas/condensate-ga-in-kems-candidate.csv"
    )
    candidate_pack = load_gas_datapack(oxide_path=candidate_path)
    source_constants = {
        "Ga": ("Ga2O3(l)", 112079.0, 40.3),
        "In": ("In2O3(l)", 90606.0, 33.1),
    }
    temperatures = (1586.4, 1630.8, 1674.2, 1695.4)

    for element, (oxide, slope, intercept) in source_constants.items():
        for temperature in temperatures:
            gas_row = _nearest_interval_row(
                default_pack.gas_df, f"{element}(g)", temperature, True
            )
            oxygen_row = _nearest_interval_row(
                default_pack.gas_df, "O2(g)", temperature, True
            )
            product_gibbs = (
                2.0 * _janaf_gibbs(temperature, gas_row)
                + 1.5 * _janaf_gibbs(temperature, oxygen_row)
            )
            ln_component_k_bar = -slope / temperature + intercept
            source_parent_gibbs = product_gibbs + (
                2.0 * R_J_MOL_K * temperature * ln_component_k_bar
            )
            before_row = _nearest_interval_row(
                default_pack.oxide_df, oxide, temperature, True
            )
            after_row = _nearest_interval_row(
                candidate_pack.oxide_df, oxide, temperature, True
            )
            before_gibbs = _lamor_gibbs(temperature, before_row)
            after_gibbs = _lamor_gibbs(temperature, after_row)
            before_pressure_ratio = math.exp(
                (before_gibbs - source_parent_gibbs)
                / (2.0 * R_J_MOL_K * temperature)
            )
            after_pressure_ratio = math.exp(
                (after_gibbs - source_parent_gibbs)
                / (2.0 * R_J_MOL_K * temperature)
            )

            assert before_pressure_ratio > 1.0
            assert after_pressure_ratio == pytest.approx(1.0, abs=2.0e-6)


def test_default_pack_carries_temperature_dependent_parent_standard_offsets() -> None:
    temperatures = (1586.4, 1630.8, 1674.2, 1695.4)
    expected = {
        "Ga2O3": (9.37, 6.52, 4.65, 3.96),
        "In2O3": (47.0, 37.1, 29.8, 26.8),
    }
    major = {"SiO2": 50.34, "Al2O3": 15.40, "MgO": 10.80, "CaO": 23.46}

    for oxide, factors in expected.items():
        composition = _composition_moles(oxide, major)
        for temperature, expected_factor in zip(temperatures, factors, strict=True):
            # The gas-path annotation is added by evaluate_gas; verify its
            # calculation independently from the helper's JSON and the pack rows.
            default_pack = load_gas_datapack()
            element = oxide.removesuffix("2O3")
            gas_row = _nearest_interval_row(default_pack.gas_df, f"{element}(g)", temperature, True)
            oxygen_row = _nearest_interval_row(default_pack.gas_df, "O2(g)", temperature, True)
            oxide_row = _nearest_interval_row(default_pack.oxide_df, f"{oxide}(l)", temperature, True)
            default_log_k = -(
                2.0 * _janaf_gibbs(temperature, gas_row)
                + 1.5 * _janaf_gibbs(temperature, oxygen_row)
                - _lamor_gibbs(temperature, oxide_row)
            ) / (R_J_MOL_K * temperature)
            slope, intercept = {
                "Ga2O3": (112079.0, 40.3),
                "In2O3": (90606.0, 33.1),
            }[oxide]
            factor = math.exp(
                0.5 * (default_log_k - 2.0 * (-slope / temperature + intercept))
            )
            assert factor == pytest.approx(expected_factor, rel=0.01)

    ga_result = evaluate_gas(
        {}, 1674.2, 1e-5, datapack=default_pack,
        composition_wt_pct={**major, "Ga2O3": 0.001},
        parent_oxides=("Ga2O3",), gas_species="Ga",
    )
    flag = ga_result.domain_flags["Ga"]
    assert isinstance(flag, TraceActivityFlag)
    assert flag.parent_liquid_standard_offset is not None
    assert flag.parent_liquid_standard_offset.code == "parent_liquid_standard_offset"
    assert flag.parent_liquid_standard_offset.factor == pytest.approx(4.65, rel=0.01)
    assert "Select the opt-in Ga/In KEMS condensate table." in str(flag)


def test_all_seven_trace_rows_preserve_nontrace_gas_channels_bit_for_bit() -> None:
    major_wt = {
        "SiO2": 49.5,
        "Al2O3": 17.0,
        "CaO": 12.0,
        "MgO": 9.0,
        "FeO": 9.0,
        "Na2O": 2.0,
        "K2O": 0.5,
        "TiO2": 1.0,
    }
    specifications = {
        "Sn": ("SnO", 1, 1),
        "Ga": ("Ga2O3", 2, 3),
        "In": ("In2O3", 2, 3),
        "Cu": ("Cu2O", 2, 1),
        "Ge": ("GeO2", 1, 2),
        "As": ("As2O3", 2, 3),
        "Bi": ("Bi2O3", 2, 3),
    }
    trace_wt = {}
    for element, (oxide, cations, oxygens) in specifications.items():
        ppm = {"In": 0.05}.get(element, 1.0)
        formula_mass = trace_component_rule(oxide).molar_mass_g_mol
        trace_wt[oxide] = (
            ppm / 10_000.0 * formula_mass
            / (cations * MOLAR_MASS_G_MOL[element])
        )

    temperature = 1800.0
    major_moles = {
        name: amount / MOLAR_MASS_G_MOL[name]
        for name, amount in major_wt.items()
    }
    melt = evaluate(major_moles, temperature, load_datapack())
    major_activities = dict(zip(melt.parent_oxides, melt.parent_activity, strict=True))
    gas_pack = load_gas_datapack()
    ordinary_species = ("SiO", "Mg")
    trace_species = (
        "Sn", "SnO", "SnO2", "Ga", "GaO", "Ga2O",
        "In", "InO", "In2O", "Cu", "CuO", "Cu2", "Ge", "GeO",
        "GeO2", "As", "As2", "As4", "As4O6", "Bi", "Bi2", "BiO",
    )
    baseline = evaluate_gas(
        major_activities, temperature, 1.0e-8, gas_pack,
        gas_species=ordinary_species,
    )
    augmented = evaluate_gas(
        major_activities, temperature, 1.0e-8, gas_pack,
        gas_species=(*ordinary_species, *trace_species),
        composition_wt_pct={**major_wt, **trace_wt},
    )

    assert all(
        augmented[name].hex() == baseline[name].hex()
        for name in ordinary_species
    )
    assert all(math.isfinite(augmented[name]) and augmented[name] > 0.0 for name in trace_species)
    for element in specifications:
        flags = [
            flag
            for name, flag in augmented.domain_flags.items()
            if name.startswith(element)
        ]
        assert flags
        assert all(flag.activity_range is not None for flag in flags)
        assert all(flag.x_source is not None for flag in flags)
