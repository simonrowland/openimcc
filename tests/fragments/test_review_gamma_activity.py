import hashlib
import json
import math
import pickle
from dataclasses import replace
from pathlib import Path

import pytest

from openimcc._constants import MOLAR_MASS_G_MOL, R_J_MOL_K
from openimcc import evaluate, load_datapack, supported_trace_oxides
from openimcc.activity_fragments import load_activity_fragment
from openimcc.gas import (
    _janaf_gibbs,
    _lamor_gibbs,
    _nearest_interval_row,
    evaluate_gas,
    load_gas_datapack,
)
from openimcc.co_activity import evaluate_co_activity
from openimcc.ni import evaluate_ni_activity
from openimcc.review_gamma_activity import evaluate_review_gamma_activity
import openimcc.review_gamma_activity as review_gamma_activity
from openimcc.trace_components import (
    ParentLiquidStandardOffset,
    TraceActivityEstimate,
    TraceActivityFlag,
    TraceMajorBiasFlag,
    _review_gamma_parent_oxides,
    _resolve_trace_activity,
    resolve_trace_activity,
    trace_component_rule,
    trace_fractions_from_wt,
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
    ("V", "V2O3", 1923.0, 0.66, 5.4, 2.0, {
        "SiO2": 2.0, "Al2O3": 1.0, "FeO": 1.0, "MgO": 1.0, "CaO": 1.0,
    }),
    ("Ag", "Ag2O", 1923.0, 0.033, 0.26, 2.0, {
        "SiO2": 2.0, "Al2O3": 1.0, "FeO": 1.0, "MgO": 1.0, "CaO": 1.0,
    }),
    ("Sb", "Sb2O3", 1923.0, 0.11, 8.2, 2.0, {
        "SiO2": 2.0, "Al2O3": 1.0, "FeO": 1.0, "MgO": 1.0, "CaO": 1.0,
    }),
    ("Tl", "Tl2O", 1923.0, 28.0, 1053.0, 2.0, {
        "SiO2": 2.0, "Al2O3": 1.0, "FeO": 1.0, "MgO": 1.0, "CaO": 1.0,
    }),
    ("Rb", "Rb2O", 1773.15, 0.000074, 0.00013, 2.0, {
        "SiO2": 2.0, "Al2O3": 1.0, "FeO": 1.0, "MgO": 1.0, "CaO": 1.0,
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
    if element in {"V", "Ag", "Sb", "Tl", "Rb"}:
        x_source = source_units * parent_moles / (7.0 + source_units * parent_moles)
    else:
        x_source = source_units * parent_moles / (
            total - parent_moles + source_units * parent_moles
        )
    gamma = {
        "Ga": 0.036,
        "In": 0.017,
        "V": 2.1,
        "Ag": 0.080,
        "Sb": 1.9,
        "Tl": 186.0,
        "Rb": 0.000074,
    }.get(element, math.sqrt(gamma_low * gamma_high))
    assert result.gamma == pytest.approx(gamma)
    assert result.gamma_range == (gamma_low, gamma_high)
    assert result.source_mole_fraction == pytest.approx(x_source)
    assert result.activity == pytest.approx((gamma * x_source) ** power)
    assert result.activity_range == pytest.approx((
        (gamma_low * x_source) ** power,
        (gamma_high * x_source) ** power,
    ))


@pytest.mark.parametrize(
    ("element", "parent", "temperature", "gamma"),
    (
        ("V", "V2O3", 1923.0, 2.1),
        ("Ag", "Ag2O", 1923.0, 0.080),
        ("Sb", "Sb2O3", 1923.0, 1.9),
        ("Tl", "Tl2O", 1923.0, 186.0),
        ("Rb", "Rb2O", 1773.15, 0.000074),
    ),
)
def test_one_cation_basis_conversion_uses_two_source_units_per_parent(
    element: str, parent: str, temperature: float, gamma: float
) -> None:
    result = evaluate_review_gamma_activity(
        element, {parent: 1.0, "SiO2": 1.0}, temperature,
        parent_oxide=parent,
    )

    assert result.gamma == gamma
    assert result.source_mole_fraction == pytest.approx(2.0 / 3.0)
    assert result.activity == pytest.approx((gamma * 2.0 / 3.0) ** 2)


@pytest.mark.parametrize(
    ("element", "parent", "temperature", "gamma"),
    (
        ("V", "V2O3", 1923.0, 2.1),
        ("Ag", "Ag2O", 1923.0, 0.080),
        ("Sb", "Sb2O3", 1923.0, 1.9),
        ("Tl", "Tl2O", 1923.0, 186.0),
        ("Rb", "Rb2O", 1773.15, 0.000074),
    ),
)
def test_norris_source_fraction_counts_all_melt_cations(
    element: str, parent: str, temperature: float, gamma: float
) -> None:
    result = evaluate_review_gamma_activity(
        element,
        {
            "SiO2": 2.0,
            "Al2O3": 1.0,
            "FeO": 1.0,
            "MgO": 1.0,
            "CaO": 1.0,
            parent: 0.001,
        },
        temperature,
        parent_oxide=parent,
    )

    assert result.source_mole_fraction == pytest.approx(0.002 / 7.002)
    assert result.activity == pytest.approx((gamma * 0.002 / 7.002) ** 2)
    old_formula_unit_activity = (gamma * 0.002 / 6.002) ** 2
    assert old_formula_unit_activity / result.activity == pytest.approx(
        (7.002 / 6.002) ** 2
    )


def test_sossi_rb_alpha_e_one_is_an_apparent_lower_bound() -> None:
    # Hertz–Knudsen for open evaporation gives J=alpha_e*gamma*x*p°/sqrt(2πMRT).
    # Solving at fixed flux gives gamma=J*sqrt(2πMRT)/(alpha_e*x*p°), so
    # alpha_e=1 is the lower-bound gamma when 0 < alpha_e <= 1.
    result = evaluate_review_gamma_activity(
        "Rb", {"Rb2O": 0.001, "SiO2": 1.0}, 1773.15,
        parent_oxide="Rb2O",
    )

    assert result.gamma == 0.000074
    assert result.gamma_range == (0.000074, 0.00013)
    assert "center-temperature 1773.15 K value" in result.reference
    assert any("lower-bound γ" in item for item in result.notice.limitations)
    assert any(
        "Sossi §4.6" in item and "fit precision only" in item
        for item in result.notice.limitations
    )
    assert any(
        "factor of 2–3" in item and "Zn 1.7–2.2" in item
        for item in result.notice.limitations
    )


@pytest.mark.parametrize(
    ("element", "parent", "temperature"),
    (
        ("V", "V2O3", 1800.0),
        ("Ag", "Ag2O", 1800.0),
        ("Sb", "Sb2O3", 1800.0),
        ("Tl", "Tl2O", 1800.0),
    ),
)
def test_norris_rows_flag_temperature_family_and_source_limits(
    element: str, parent: str, temperature: float
) -> None:
    result = evaluate_review_gamma_activity(
        element,
        {
            "SiO2": 50.0,
            "Al2O3": 15.0,
            "FeO": 10.0,
            "MgO": 10.0,
            "CaO": 15.0,
            "Na2O": 1.0,
            parent: 0.001,
        },
        temperature,
        parent_oxide=parent,
    )

    assert result.notice.code == "review_gamma_provisional"
    assert "temperature_extrapolated" in result.notice.violated_axes
    assert "melt_family_extrapolated" in result.notice.violated_axes
    assert any("accurate within an order of magnitude" in item for item in result.notice.limitations)
    assert any("inferred from metal–silicate partitioning (Kf + γ-in-Fe model); not a measured activity" == item for item in result.notice.limitations)
    assert any("single temperature 1923 K" in item for item in result.notice.limitations)
    if element == "V":
        assert any("valence was not measured per run" in item for item in result.notice.limitations)
    if element == "Ag":
        assert "P672-Ag2O-LFS" in result.reference
        assert any("Assal et al. (1997)" in item for item in result.notice.limitations)
    if element == "Sb":
        assert "P672-Sb2O3-LFS" in result.reference
        assert any("Barin (1973)" in item for item in result.notice.limitations)
    if element == "Tl":
        assert "P672-Tl2O-l-CP-EXT-1000" in result.reference
        assert any("upper estimate for Bi, Cd, and Tl" in item for item in result.notice.limitations)
        assert any("high direction" in item and "Eq. 4.1" in item for item in result.notice.limitations)


def test_family_screen_counts_declared_review_gamma_parents_and_nonoxides() -> None:
    solvent = {
        "SiO2": 0.594,
        "Al2O3": 0.107,
        "FeO": 0.113,
        "MgO": 0.104,
        "CaO": 0.072,
        "PbO": 1.0e-4,
    }
    for component in ("Ga2O3", "Cl"):
        composition = {**solvent, component: 1.0}
        result = evaluate_review_gamma_activity(
            "Pb", composition, 1923.0, parent_oxide="PbO"
        )

        if component == "Ga2O3":
            assert "melt_family_extrapolated" in result.notice.violated_axes
            assert (
                "melt-family distance 1.50249 oxide mole fraction"
                in result.reference
            )
        else:
            assert "melt_family_extrapolated" not in result.notice.violated_axes


def test_family_screen_counts_missing_oxides_and_excludes_nonoxides() -> None:
    ge = evaluate_review_gamma_activity(
        "Ge", {"SiO2": 1.0, "GeO2": 0.001}, 1573.0, parent_oxide="GeO2"
    )
    assert "melt_family_extrapolated" in ge.notice.violated_axes
    assert "melt-family distance 0 oxide mole fraction" in ge.reference
    assert "3 required family oxides absent" in ge.reference

    ce_with_sulfur = evaluate_review_gamma_activity(
        "Ce",
        {
            "CaO": 1.0,
            "MgO": 1.0,
            "Al2O3": 1.0,
            "SiO2": 1.0,
            "Ce2O3": 0.1,
            "S": 0.5,
        },
        1873.0,
        parent_oxide="Ce2O3",
        fO2_bar=4.76e-16 * 1.01325,
    )
    assert "melt_family_extrapolated" not in ce_with_sulfur.notice.violated_axes


def test_an_di_screen_renormalizes_declared_major_oxides_only() -> None:
    target_wt_pct = {
        "SiO2": 50.34,
        "Al2O3": 15.4,
        "MgO": 10.8,
        "CaO": 23.46,
    }
    target_moles = {
        component: weight / MOLAR_MASS_G_MOL[component]
        for component, weight in target_wt_pct.items()
    }
    reported_inputs = (
        {**target_moles, "Ga2O3": 0.02},
        {**target_moles, "Ga2O3": 1.0e-6, "Cl": 0.05},
    )

    for composition in reported_inputs:
        result = evaluate_review_gamma_activity(
            "Ga",
            composition,
            1700.0,
            parent_oxide="Ga2O3",
            fO2_bar=3.640629388810509e-08,
        )
        assert "melt_composition_differs_from_An_Di_target" not in (
            result.notice.violated_axes
        )
        assert "an-di-target-basis=major-oxides-only-v2" in result.reference

    off_target_major = evaluate_review_gamma_activity(
        "Ga",
        {**target_moles, "FeO": 0.05, "Ga2O3": 1.0e-6},
        1700.0,
        parent_oxide="Ga2O3",
        fO2_bar=3.640629388810509e-08,
    )
    assert "melt_composition_differs_from_An_Di_target" in (
        off_target_major.notice.violated_axes
    )


def test_an_di_failure_is_included_in_the_default_melt_trace_flag() -> None:
    composition = {
        "SiO2": 45.99,
        "Al2O3": 15.0,
        "FeO": 8.0,
        "CaO": 25.0,
        "MgO": 3.96,
        "NiO": 0.05,
        "Ga2O3": 2.0,
    }
    result = evaluate(composition, 1700.0, basis_type="wt")
    ga_flag = next(
        item
        for item in result.labels.notices
        if isinstance(item, TraceActivityFlag) and item.oxide == "Ga2O3"
    )

    assert "melt_composition_differs_from_An_Di_target" in str(ga_flag)


@pytest.mark.parametrize(
    ("trace_oxide", "window_flag"),
    (
        ("NiO", "MgO_wt_pct_outside_4_to_30"),
        ("CoO", "MgO_outside_4_to_30_wt_pct"),
    ),
)
def test_nico_flags_check_declared_weight_window_with_unsupported_oxide(
    trace_oxide: str, window_flag: str
) -> None:
    composition = {
        "SiO2": 45.99,
        "Al2O3": 15.0,
        "FeO": 8.0,
        "CaO": 25.0,
        "MgO": 3.96,
        trace_oxide: 0.05,
        "Ga2O3": 2.0,
    }
    result = evaluate(composition, 1700.0, basis_type="wt")
    flag = next(
        item
        for item in result.labels.notices
        if isinstance(item, TraceActivityFlag) and item.oxide == trace_oxide
    )

    assert flag.code == "depends_helper_activity"
    assert window_flag in str(flag)
    assert "unsupported components omitted from helper value: Ga2O3" in str(flag)


@pytest.mark.parametrize("trace_oxide", ("NiO", "CoO"))
def test_nico_resolver_accepts_feoeq_and_preserves_helper_activity(
    trace_oxide: str,
) -> None:
    composition = _activity_grid_composition(
        6.0, 10.0, trace_oxide, feo_component="FeOeq"
    )
    estimate, _, _ = _resolve_trace_activity(
        trace_oxide,
        composition[trace_oxide],
        composition,
        1700.0,
    )
    helper_composition = {
        name: amount
        for name, amount in composition.items()
        if trace_oxide == "NiO" or name != "FeOeq"
    }
    expected = (
        evaluate_ni_activity(helper_composition, 1700.0)
        if trace_oxide == "NiO"
        else evaluate_co_activity(helper_composition, 1700.0)
    )

    assert estimate.code == "depends_helper_activity"
    oxygen_flag = (
        "oxygen_fugacity_relative_to_IW_not_supplied"
        if trace_oxide == "NiO"
        else "oxygen_fugacity_domain_unchecked"
    )
    assert oxygen_flag in (estimate.reference or "")
    assert estimate.activity.hex() == expected.activity.hex()


@pytest.mark.parametrize("trace_oxide", ("NiO", "CoO"))
def test_nico_resolver_handles_zero_mole_composition(trace_oxide: str) -> None:
    estimate = resolve_trace_activity(trace_oxide, 0.0, {}, 1700.0)

    assert isinstance(estimate, TraceActivityEstimate)


def test_sossi_rb_row_flags_temperature_and_family() -> None:
    result = evaluate_review_gamma_activity(
        "Rb",
        {
            "SiO2": 50.0,
            "Al2O3": 15.0,
            "FeO": 10.0,
            "MgO": 10.0,
            "CaO": 15.0,
            "Na2O": 1.0,
            "Rb2O": 0.001,
        },
        1600.0,
        parent_oxide="Rb2O",
    )

    assert "temperature_extrapolated" in result.notice.violated_axes
    assert "melt_family_extrapolated" in result.notice.violated_axes
    assert result.notice.code == "review_gamma_provisional"


def test_new_review_gamma_oxides_are_listed_with_existing_trace_classes() -> None:
    supported = {item.oxide: item for item in supported_trace_oxides()}
    expected = {
        "V2O3": ("V", "basic_oxide"),
        "Ag2O": ("Ag", "basic_oxide"),
        "Sb2O3": ("Sb", "basic_oxide"),
        "Tl2O": ("Tl", "basic_oxide"),
        "Rb2O": ("Rb", "basic_oxide"),
    }
    for oxide, (element, trace_class) in expected.items():
        assert supported[oxide].trace_class == trace_class
        rule = trace_component_rule(oxide)
        assert rule is not None
        assert rule.review_gamma_element == element


@pytest.mark.parametrize(
    ("element", "parent", "amount_wt_pct", "temperature"),
    (
        ("V", "V2O3", 0.001, 1923.0),
        ("Ag", "Ag2O", 0.00001, 1923.0),
        ("Sb", "Sb2O3", 0.00001, 1923.0),
        ("Tl", "Tl2O", 0.00001, 1923.0),
        ("Rb", "Rb2O", 0.0005, 1773.15),
    ),
)
def test_new_review_gamma_rows_give_finite_regolith_gas_pressure_with_domain_flags(
    element: str, parent: str, amount_wt_pct: float, temperature: float
) -> None:
    composition = {
        "SiO2": 48.5,
        "TiO2": 2.0,
        "Al2O3": 15.0,
        "FeO": 12.0,
        "MgO": 9.0,
        "CaO": 11.5,
        "Na2O": 2.0,
        parent: amount_wt_pct,
    }
    melt = evaluate(
        composition, temperature, basis_type="wt", allow_extrapolation=True
    )
    activities = dict(zip(melt.parent_oxides, melt.parent_activity, strict=True))
    gas = evaluate_gas(
        activities,
        temperature,
        1.0e-8,
        load_gas_datapack(),
        gas_species=(element,),
        composition_wt_pct=composition,
    )

    flag = gas.domain_flags[element]
    assert math.isfinite(gas[element]) and gas[element] > 0.0
    assert isinstance(flag, TraceActivityFlag)
    assert flag.code == "review_gamma_provisional"
    assert "melt-family distance" in str(flag)
    if element == "Rb":
        assert "lower-bound γ" in str(flag)
    else:
        assert "inferred from metal–silicate partitioning" in str(flag)


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
    assert "gamma-range-printed-point-v1" in ga.reference
    assert "published x_source anchor 6.9e-05 at 0.05 ppm" in indium.reference
    assert "gamma-range-printed-point-v1" in indium.reference
    assert "an-di-target-basis=major-oxides-only-v2" in ga.reference


def test_bischof_table8_s1_closure_reproduces_source_constants() -> None:
    # Bischof et al. (2023), §3.2, p. 4, Eqs. 22–23 derive gamma from the
    # measured melt pressures and x, so this pressure ratio reduces to
    # K_engine/K_source (both per MO1.5; sqrt(K_engine/K_source) per M2O3)
    # rather than independently checking melt activity.
    # p scales with gamma (a=(gamma*x)^2, n_gas=2); printed p is three figures
    # (<=0.5%) and gamma four (<=0.2%), so rel=5e-3.
    candidate_path = (
        Path(__file__).resolve().parents[2]
        / "src/openimcc/data/gas/condensate-ga-in-kems-candidate.csv"
    )
    pack = load_gas_datapack(oxide_path=candidate_path)
    # Table 8 x(MO1.5). S1 1_low rows below are that row's gamma, p(M) and p(O2) in Pa.
    x_ga = 0.0172
    x_in = 6.9e-5
    source_rows = (
        (1586.4, 0.000112, 0.0632, 4.04e-05, 0.0527, 6.69e-05),
        (1630.8, 0.00027, 0.0285, 6.45e-05, 0.0377, 0.000117),
        (1674.2, 0.000681, 0.0248, 0.000166, 0.0301, 0.000197),
        (1695.4, 0.00106, 0.0246, 0.000273, 0.0254, 0.000234),
    )
    expected_closures = (
        (0.9166, 1.0449),
        (0.9159, 1.0461),
        (0.9190, 1.0463),
        (0.9187, 1.0494),
    )

    for source_row, expected in zip(source_rows, expected_closures, strict=True):
        temperature, p_o2_pa, ga_gamma, p_ga_pa, in_gamma, p_in_pa = source_row
        predicted = evaluate_gas(
            {
                "Ga2O3": (ga_gamma * x_ga) ** 2,
                "In2O3": (in_gamma * x_in) ** 2,
            },
            temperature,
            p_o2_pa / 1.0e5,
            pack,
            parent_oxides=("Ga2O3", "In2O3"),
            gas_species=("Ga", "In"),
        )
        closure = (
            predicted["Ga"] / (p_ga_pa / 1.0e5),
            predicted["In"] / (p_in_pa / 1.0e5),
        )
        assert closure == pytest.approx(expected, rel=5.0e-3)


def test_as_parent_wording_and_inactive_ge_row() -> None:
    arsenic = evaluate_review_gamma_activity(
        "As", {"FeO": 1.0, "CaO": 1.0, "SiO2": 2.0, "As2O3": 0.001},
        1573.0, parent_oxide="As2O3",
    )
    assert "pure liquid As2O3" not in arsenic.reference
    assert trace_component_rule("GeO") is None
    vanadium = trace_component_rule("V2O3")
    assert vanadium is not None
    assert vanadium.review_gamma_element == "V"


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


def test_default_pack_applies_kems_parent_activity_conversion() -> None:
    major = {"SiO2": 50.34, "Al2O3": 15.40, "MgO": 10.80, "CaO": 23.46}
    default_pack = load_gas_datapack()
    candidate_path = (
        Path(__file__).resolve().parents[2]
        / "src/openimcc/data/gas/condensate-ga-in-kems-candidate.csv"
    )
    candidate_pack = load_gas_datapack(oxide_path=candidate_path)
    source_constants = {
        "Ga2O3": ("Ga", 112079.0, 40.3),
        "In2O3": ("In", 90606.0, 33.1),
    }
    ratios_at_reported_temperatures = {
        "Ga2O3": (0.006116312906, 0.046192692774, 0.122394467160),
        "In2O3": (0.000302850168, 0.001125827170, 0.002126134242),
    }

    for oxide, (element, slope, intercept) in source_constants.items():
        for temperature, expected_ratio in zip(
            (1550.0, 1674.0, 1740.0),
            ratios_at_reported_temperatures[oxide],
            strict=True,
        ):
            gas_row = _nearest_interval_row(
                default_pack.gas_df, f"{element}(g)", temperature, True
            )
            oxygen_row = _nearest_interval_row(
                default_pack.gas_df, "O2(g)", temperature, True
            )
            oxide_row = _nearest_interval_row(
                default_pack.oxide_df, f"{oxide}(l)", temperature, True
            )
            nasa_parent_k_log = -(
                2.0 * _janaf_gibbs(temperature, gas_row)
                + 1.5 * _janaf_gibbs(temperature, oxygen_row)
                - _lamor_gibbs(temperature, oxide_row)
            ) / (R_J_MOL_K * temperature)
            bischof_parent_k_log = 2.0 * (-slope / temperature + intercept)
            expected_from_rows = math.exp(
                bischof_parent_k_log - nasa_parent_k_log
            )
            assert expected_from_rows == pytest.approx(expected_ratio, rel=2e-8)

            composition = {**major, oxide: 0.001}
            default_result = evaluate_gas(
                {}, temperature, 1e-5, datapack=default_pack,
                composition_wt_pct=composition,
                gas_species=element,
            )
            candidate_result = evaluate_gas(
                {}, temperature, 1e-5, datapack=candidate_pack,
                composition_wt_pct=composition,
                gas_species=element,
            )
            default_flag = default_result.domain_flags[element]
            candidate_flag = candidate_result.domain_flags[element]
            assert isinstance(default_flag, TraceActivityFlag)
            assert isinstance(candidate_flag, TraceActivityFlag)
            conversion = default_flag.parent_liquid_standard_offset
            assert conversion is not None
            assert conversion.code == "parent_liquid_standard_offset"
            assert conversion.factor == pytest.approx(expected_ratio, rel=2e-8)
            trace_moles, trace_fractions = trace_fractions_from_wt(composition)
            raw_estimate, _, _ = _resolve_trace_activity(
                oxide,
                trace_fractions[oxide],
                trace_moles,
                temperature,
                fO2_bar=1e-5,
            )
            assert raw_estimate.activity_range is not None
            assert default_flag.activity == pytest.approx(
                raw_estimate.activity * expected_ratio
            )
            assert default_flag.activity_range == pytest.approx(
                tuple(
                    endpoint * expected_ratio
                    for endpoint in raw_estimate.activity_range
                )
            )
            assert "Applied the Bischof KEMS-to-NASA parent-standard conversion" in str(
                default_flag
            )
            assert "out of domain: oxygen_fugacity_extrapolated" in str(
                default_flag
            )
            assert candidate_flag.parent_liquid_standard_offset is None
            # The two parent Gibbs functions are independently represented by
            # a 5-coefficient polynomial and Bischof's two-parameter ln K fit.
            # Their small residual is the only expected cross-path difference.
            assert default_result[element] > 0.0
            assert candidate_result[element] > 0.0
            # The 5e-6 cross-path pressure tolerance is for monatomic channels
            # (n_gas=2); n_gas=1 channels inherit roughly twice the fit residual.
            assert default_result[element] == pytest.approx(
                candidate_result[element], rel=5e-6
            )

    for oxide, (element, slope, intercept) in source_constants.items():
        native_card_start = {"Ga2O3": 2080.0, "In2O3": 2186.0}[oxide]
        for temperature in (1500.0, 1800.0, native_card_start + 20.0):
            gas_row = _nearest_interval_row(
                default_pack.gas_df, f"{element}(g)", temperature, True
            )
            oxygen_row = _nearest_interval_row(
                default_pack.gas_df, "O2(g)", temperature, True
            )
            oxide_row = _nearest_interval_row(
                default_pack.oxide_df, f"{oxide}(l)", temperature, True
            )
            nasa_parent_k_log = -(
                2.0 * _janaf_gibbs(temperature, gas_row)
                + 1.5 * _janaf_gibbs(temperature, oxygen_row)
                - _lamor_gibbs(temperature, oxide_row)
            ) / (R_J_MOL_K * temperature)
            bischof_parent_k_log = 2.0 * (-slope / temperature + intercept)
            ratio = math.exp(bischof_parent_k_log - nasa_parent_k_log)
            result = evaluate_gas(
                {}, temperature, 1e-5, datapack=default_pack,
                composition_wt_pct={**major, oxide: 0.001},
                gas_species=element,
            )
            flag = result.domain_flags[element]
            assert isinstance(flag, TraceActivityFlag)
            conversion = flag.parent_liquid_standard_offset
            assert conversion is not None
            assert conversion.factor == pytest.approx(ratio, rel=1e-8)
            assert "Applied the Bischof KEMS-to-NASA parent-standard conversion" in str(
                flag
            )
            assert "extrapolated outside the measured 1550–1740 K KEMS fit" in str(
                flag
            )

    for oxide, (element, _slope, _intercept) in source_constants.items():
        composition = {**major, oxide: 0.001}
        for temperature in (1549.0, 1740.000000000001):
            default_result = evaluate_gas(
                {}, temperature, 1e-5, datapack=default_pack,
                composition_wt_pct=composition, gas_species=element,
            )
            candidate_result = evaluate_gas(
                {}, temperature, 1e-5, datapack=candidate_pack,
                composition_wt_pct=composition, gas_species=element,
            )
            selected_row = _nearest_interval_row(
                candidate_pack.oxide_df, f"{oxide}(l)", temperature, True
            )
            assert str(selected_row["Ref"]) != "BISCHOF-KEMS-2023"
            default_flag = default_result.domain_flags[element]
            candidate_flag = candidate_result.domain_flags[element]
            assert isinstance(default_flag, TraceActivityFlag)
            assert isinstance(candidate_flag, TraceActivityFlag)
            assert default_flag.parent_liquid_standard_offset is not None
            assert candidate_flag.parent_liquid_standard_offset is not None
            assert candidate_flag.parent_liquid_standard_offset.factor == pytest.approx(
                default_flag.parent_liquid_standard_offset.factor, rel=2e-8
            )
            assert default_result[element] == pytest.approx(
                candidate_result[element], rel=5e-6
            )

        for temperature in (1550.0, 1740.0):
            default_result = evaluate_gas(
                {}, temperature, 1e-5, datapack=default_pack,
                composition_wt_pct=composition, gas_species=element,
            )
            candidate_result = evaluate_gas(
                {}, temperature, 1e-5, datapack=candidate_pack,
                composition_wt_pct=composition, gas_species=element,
            )
            selected_row = _nearest_interval_row(
                candidate_pack.oxide_df, f"{oxide}(l)", temperature, True
            )
            assert str(selected_row["Ref"]) == "BISCHOF-KEMS-2023"
            default_flag = default_result.domain_flags[element]
            candidate_flag = candidate_result.domain_flags[element]
            assert isinstance(default_flag, TraceActivityFlag)
            assert isinstance(candidate_flag, TraceActivityFlag)
            assert default_flag.parent_liquid_standard_offset is not None
            assert candidate_flag.parent_liquid_standard_offset is None
            assert default_result[element] == pytest.approx(
                candidate_result[element], rel=5e-6
            )

    hight_pack = load_gas_datapack(
        oxide_path=(
            Path(__file__).resolve().parents[2]
            / "src/openimcc/data/gas/condensate-hight-candidate-v2.csv"
        )
    )
    assert hight_pack.condensate_table_digest != default_pack.condensate_table_digest
    for oxide, (element, _slope, _intercept) in source_constants.items():
        composition = {**major, oxide: 0.001}
        default_result = evaluate_gas(
            {}, 1674.0, 1e-5, datapack=default_pack,
            composition_wt_pct=composition, gas_species=element,
        )
        hight_result = evaluate_gas(
            {}, 1674.0, 1e-5, datapack=hight_pack,
            composition_wt_pct=composition, gas_species=element,
        )
        default_flag = default_result.domain_flags[element]
        hight_flag = hight_result.domain_flags[element]
        assert isinstance(default_flag, TraceActivityFlag)
        assert isinstance(hight_flag, TraceActivityFlag)
        hight_row = _nearest_interval_row(
            hight_pack.oxide_df, f"{oxide}(l)", 1674.0, True
        )
        assert str(hight_row["Ref"]) != "BISCHOF-KEMS-2023"
        assert hight_flag.parent_liquid_standard_offset is not None
        assert hight_flag.parent_liquid_standard_offset.factor == pytest.approx(
            default_flag.parent_liquid_standard_offset.factor, rel=2e-8
        )
        assert default_result[element] == pytest.approx(
            hight_result[element], rel=5e-6
        )

    for oxide, (element, _slope, _intercept) in source_constants.items():
        composition = {**major, oxide: 0.001}
        trace_moles, trace_fractions = trace_fractions_from_wt(composition)
        raw_estimate, _, _ = _resolve_trace_activity(
            oxide,
            trace_fractions[oxide],
            trace_moles,
            1674.0,
            fO2_bar=1e-5,
        )
        explicit_result = evaluate_gas(
            {oxide: 0.5}, 1674.0, 1e-5, datapack=default_pack,
            composition_wt_pct=composition, gas_species=element,
        )
        caller_activity_result = evaluate_gas(
            {oxide: 0.5}, 1674.0, 1e-5, datapack=default_pack,
            gas_species=element,
        )
        explicit_flag = explicit_result.domain_flags[element]
        assert isinstance(explicit_flag, TraceActivityFlag)
        assert explicit_flag.parent_liquid_standard_offset is None
        assert "Applied the Bischof KEMS-to-NASA parent-standard conversion" not in str(
            explicit_flag
        )
        assert explicit_flag.activity == pytest.approx(raw_estimate.activity)
        assert explicit_flag.activity_range == pytest.approx(
            raw_estimate.activity_range
        )
        assert explicit_result[element] == pytest.approx(
            caller_activity_result[element], rel=1e-12
        )


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


def test_wood_wade_pb_point_carries_the_reported_uncertainty_status() -> None:
    major_wt_pct = {
        "SiO2": 59.4,
        "Al2O3": 10.7,
        "FeO": 11.3,
        "MgO": 10.4,
        "CaO": 7.2,
    }
    composition = {
        name: amount / MOLAR_MASS_G_MOL[name]
        for name, amount in major_wt_pct.items()
    }
    composition["PbO"] = 0.001
    result = evaluate_review_gamma_activity(
        "Pb", composition, 1923.0, parent_oxide="PbO"
    )

    printed_gamma = 0.22
    assert result.gamma == printed_gamma
    assert result.gamma_range == (printed_gamma, printed_gamma)
    cation_basis_total = math.fsum(
        amount * (2.0 if name == "Al2O3" else 1.0)
        for name, amount in composition.items() if name != "PbO"
    ) + composition["PbO"]
    assert result.source_mole_fraction == composition["PbO"] / cation_basis_total
    assert result.activity == printed_gamma * result.source_mole_fraction
    assert result.activity_range == (
        printed_gamma * result.source_mole_fraction,
        printed_gamma * result.source_mole_fraction,
    )
    assert result.row == "PbO-BW1102-1923K"
    assert "a(PbO)=(gamma*x_source)^1" in result.reference
    assert "pure liquid PbO at 1 bar" in result.reference
    assert "uncertainty not stated" in result.notice.limitations[0]
    assert "oxygen_fugacity_window_unreported" in result.notice.violated_axes
    assert "melt_family_extrapolated" not in result.notice.violated_axes


@pytest.mark.parametrize(
    ("element", "parent", "temperature", "composition", "gamma", "expected_hex"),
    [
        (
            "W", "WO3", 1923.0,
            {"SiO2": 3.0, "Al2O3": 0.5, "MgO": 2.0, "CaO": 1.0, "WO3": 0.001},
            math.sqrt(0.001 * 0.05), "0x1.0f1f3d2cd4b36p-20",
        ),
        (
            "Pb", "PbO", 1923.0,
            {"SiO2": 3.0, "Al2O3": 0.5, "FeO": 1.0, "MgO": 2.0, "CaO": 1.0, "PbO": 0.001},
            0.22, "0x1.cd50d6aa3de41p-16",
        ),
        (
            "Cu", "Cu2O", 1923.0,
            {"SiO2": 3.0, "Al2O3": 0.5, "MgO": 2.0, "CaO": 1.0, "Cu2O": 0.001},
            3.5, "0x1.0c4839ac5f19ep-20",
        ),
    ],
)
def test_wood_wade_rows_use_single_cation_basis_on_alumina_melts(
    element, parent, temperature, composition, gamma, expected_hex
) -> None:
    row = next(
        item for item in load_activity_fragment(element)["rows"]
        if item["parent_oxide"] == parent
        and item["source_citation"].startswith("Wood & Wade (2013)")
    )
    assert row["source_mole_fraction_kind"] == "one_cation_oxide"
    result = evaluate_review_gamma_activity(
        element, composition, temperature, parent_oxide=parent
    )

    source_units = float(row["source_units_per_parent"])
    numerator = source_units * composition[parent]
    formula_basis_total = math.fsum(
        amount * (source_units if name == parent else 1.0)
        for name, amount in composition.items()
    )
    cation_basis_total = math.fsum(
        amount * (2.0 if name == "Al2O3" else 1.0)
        for name, amount in composition.items() if name != parent
    ) + numerator
    parent_power = float(row["parent_power"])
    old_activity = (gamma * (numerator / formula_basis_total)) ** parent_power
    expected_activity = (gamma * (numerator / cation_basis_total)) ** parent_power

    assert result.gamma == gamma
    assert result.gamma_range == tuple(
        float(value) for value in row["gamma_law"]["gamma_endpoints"]
    )
    assert result.source_mole_fraction == numerator / cation_basis_total
    assert result.activity.hex() == expected_hex
    assert result.activity == expected_activity
    assert result.activity == pytest.approx(
        old_activity * (formula_basis_total / cation_basis_total) ** parent_power,
        rel=1e-15,
    )


@pytest.mark.parametrize(
    ("element", "parent", "temperature"),
    [("W", "WO3", 1923.0), ("Pb", "PbO", 1923.0), ("Cu", "Cu2O", 1923.0)],
)
def test_wood_wade_rows_are_unchanged_for_silica_only_melts(
    element, parent, temperature
) -> None:
    parent_moles = 0.001
    composition = {"SiO2": 1.0, parent: parent_moles}
    row = next(
        item for item in load_activity_fragment(element)["rows"]
        if item["parent_oxide"] == parent
        and item["source_citation"].startswith("Wood & Wade (2013)")
    )
    source_units = float(row["source_units_per_parent"])
    result = evaluate_review_gamma_activity(
        element, composition, temperature, parent_oxide=parent
    )
    old_formula_x = source_units * parent_moles / (
        1.0 + source_units * parent_moles
    )

    assert row["source_mole_fraction_kind"] == "one_cation_oxide"
    assert result.source_mole_fraction == old_formula_x
    assert result.activity == (result.gamma * old_formula_x) ** float(row["parent_power"])


def test_wood_wade_tungsten_contour_band_reaches_gas_flags() -> None:
    composition = {
        "CaO": 1.0,
        "MgO": 2.0,
        "Al2O3": 0.5,
        "SiO2": 3.0,
        "WO3": 0.001,
    }
    result = evaluate_review_gamma_activity(
        "W", composition, 1923.0, parent_oxide="WO3"
    )

    assert result.gamma_range == (0.001, 0.05)
    assert result.gamma == math.sqrt(0.001 * 0.05)
    assert result.activity_range == (
        0.001 * result.source_mole_fraction,
        0.05 * result.source_mole_fraction,
    )
    assert result.row == "WO3-Di-An-Fo-1923K"
    assert "assumes W⁶⁺ dominance" in " ".join(result.notice.limitations)
    assert "ΔIW −3.5" in " ".join(result.notice.limitations)
    assert "lower-fO₂ use requires W speciation" in " ".join(result.notice.limitations)
    assert "a(WO3)=(gamma*x_source)^1" in result.reference
    assert "pure liquid WO3 at 1 bar" in result.reference
    assert "contour-read (provisional)" in result.notice.limitations[0]
    assert "oxygen_fugacity_window_unreported" in result.notice.violated_axes
    assert "melt_family_extrapolated" not in result.notice.violated_axes

    default_pack = load_gas_datapack()
    for species in ("PbO(l)", "WO3(l)"):
        parent_row = _nearest_interval_row(
            default_pack.oxide_df, species, 1923.0, True
        )
        assert parent_row["state"] == "l"
        assert parent_row["T_min"] <= 1923.0 <= parent_row["T_max"]

    rule = trace_component_rule("WO3")
    assert rule is not None
    assert rule.gas_parent_oxide == "WO3"
    assert rule.review_gamma_element == "W"
    estimate, _, _ = _resolve_trace_activity(
        "WO3", composition["WO3"], composition, 1923.0, fO2_bar=1.0e-8
    )
    gas_flag = TraceActivityFlag(estimate, context="gas")
    assert gas_flag.activity_range == result.activity_range
    assert "contour-read" in str(gas_flag)

    pb_composition = {
        name: amount / MOLAR_MASS_G_MOL[name]
        for name, amount in {
            "SiO2": 59.4,
            "Al2O3": 10.7,
            "FeO": 11.3,
            "MgO": 10.4,
            "CaO": 7.2,
        }.items()
    }
    pb_composition["PbO"] = 0.001
    pb_rule = trace_component_rule("PbO")
    assert pb_rule is not None
    assert pb_rule.gas_parent_oxide == "PbO"
    assert pb_rule.review_gamma_element == "Pb"
    pb_estimate, _, _ = _resolve_trace_activity(
        "PbO", pb_composition["PbO"], pb_composition, 1923.0, fO2_bar=1.0e-8
    )
    pb_flag = TraceActivityFlag(pb_estimate, context="gas")
    assert pb_flag.activity_range == (
        pb_estimate.activity,
        pb_estimate.activity,
    )
    assert "uncertainty not stated" in str(pb_flag)


def _activity_grid_composition(
    feo_wt_pct: float,
    mgo_wt_pct: float,
    trace_oxide: str,
    trace_wt_pct: float = 0.01,
    feo_component: str = "FeO",
) -> dict[str, float]:
    major = {
        feo_component: feo_wt_pct / MOLAR_MASS_G_MOL["FeO"],
        "MgO": mgo_wt_pct / MOLAR_MASS_G_MOL["MgO"],
        "CaO": 15.0 / MOLAR_MASS_G_MOL["CaO"],
        "Al2O3": 10.0 / MOLAR_MASS_G_MOL["Al2O3"],
        "SiO2": (
            100.0 - feo_wt_pct - mgo_wt_pct - 15.0 - 10.0 - trace_wt_pct
        ) / MOLAR_MASS_G_MOL["SiO2"],
    }
    major[trace_oxide] = trace_wt_pct / MOLAR_MASS_G_MOL[trace_oxide]
    return major


def test_ni_and_co_match_b750046_float_hex_outputs_over_domain_grid() -> None:
    ni_compositions = [
        _activity_grid_composition(6.0, 10.0, "NiO"),
        _activity_grid_composition(0.0, 4.0, "NiO"),
        _activity_grid_composition(12.0, 4.0, "NiO"),
        _activity_grid_composition(12.000001, 4.0, "NiO"),
        _activity_grid_composition(0.0, 20.0, "NiO"),
        _activity_grid_composition(0.0, 30.0, "NiO"),
        _activity_grid_composition(0.0, 30.000001, "NiO"),
        _activity_grid_composition(6.0, 3.999999, "NiO"),
        _activity_grid_composition(6.0, 30.0, "NiO"),
        _activity_grid_composition(6.0, 30.000001, "NiO"),
        _activity_grid_composition(12.0, 30.0, "NiO"),
        _activity_grid_composition(6.0, 10.0, "NiO", feo_component="FeOeq"),
    ]
    co_compositions = [
        {"CoO" if name == "NiO" else name: amount for name, amount in composition.items()}
        for composition in ni_compositions[:-1]
    ]
    temperatures = (1573.149, 1573.15, 1700.0, 1873.15, 1873.151, 1923.0)
    delta_iw_values = (None, -3.0000000001, -3.0, -1.25, 1.5, 1.5000000001)

    def hex_value(value: float) -> str:
        return float(value).hex()

    ni_payload = []
    for composition in ni_compositions:
        for temperature in temperatures:
            for delta_iw in delta_iw_values:
                result = evaluate_ni_activity(composition, temperature, delta_iw)
                ni_payload.append([
                    hex_value(result.activity),
                    hex_value(result.gamma),
                    hex_value(result.x_nio),
                    [hex_value(value) for value in result.activity_spread],
                    result.notice.code,
                    list(result.notice.violated_axes),
                    list(result.notice.limitations),
                    str(result.notice),
                ])

    co_payload = []
    for composition in co_compositions:
        for temperature in temperatures:
            for delta_iw in delta_iw_values:
                result = evaluate_co_activity(
                    composition, temperature, log10_fO2_over_IW=delta_iw
                )
                co_payload.append([
                    hex_value(result.activity),
                    hex_value(result.log10_activity),
                    hex_value(result.gamma),
                    hex_value(result.x_coo),
                    [hex_value(value) for value in result.gamma_range],
                    [hex_value(value) for value in result.activity_range],
                    result.notice.code,
                    list(result.notice.violated_axes),
                    list(result.notice.limitations),
                    str(result.notice),
                ])

    def payload_hash(payload: list[list[object]]) -> str:
        encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode()
        return hashlib.sha256(encoded).hexdigest()

    # Baselines captured from b750046; each number is compared through float.hex().
    assert len(ni_payload) == 432
    assert len(co_payload) == 396
    assert payload_hash(ni_payload) == (
        "b160d9935af2aa0efe10f70ebc005be5cf9f326e4e006c2c32d0f9906fbb0701"
    )
    assert payload_hash(co_payload) == (
        "a49593cf750d325051433049c1f7ba11a4274c0da301f8749749c14603c689e4"
    )


_CE_ZHAO_TABLE_IV = (
    (301, 45.0, 50.0, 5.0, 3.29),
    (302, 41.0, 54.0, 5.0, 1.72),
    (303, 37.0, 58.0, 5.0, 0.86),
    (304, 35.0, 55.0, 10.0, 2.81),
    (305, 38.0, 52.0, 10.0, 3.64),
    (306, 35.0, 50.0, 15.0, 7.94),
    (307, 32.5, 52.5, 15.0, 5.03),
    (308, 32.5, 47.5, 20.0, 16.22),
    (309, 30.0, 50.0, 20.0, 8.97),
    (310, 27.5, 47.5, 25.0, 15.80),
    (311, 25.0, 50.0, 25.0, 9.02),
    (312, 23.75, 46.25, 30.0, 19.06),
    (313, 21.25, 48.75, 30.0, 9.03),
    (314, 18.75, 46.25, 35.0, 18.20),
)

_CE_CHEN_TABLE_2 = (
    (101, 22.5, 57.5, 5.0, 5.0, 10.0, 0.24),
    (102, 27.5, 52.5, 5.0, 5.0, 10.0, 0.42),
    (103, 32.5, 47.5, 5.0, 5.0, 10.0, 1.10),
    (104, 10.0, 60.0, 15.0, 5.0, 10.0, 0.18),
    (105, 15.0, 55.0, 15.0, 5.0, 10.0, 0.28),
    (106, 20.0, 50.0, 15.0, 5.0, 10.0, 0.78),
    (107, 25.0, 45.0, 15.0, 5.0, 10.0, 1.17),
    (108, 27.5, 42.5, 15.0, 5.0, 10.0, 2.10),
    (109, 5.0, 57.5, 22.5, 5.0, 10.0, 0.17),
    (110, 15.0, 47.5, 22.5, 5.0, 10.0, 0.96),
    (111, 25.0, 37.5, 22.5, 5.0, 10.0, 2.12),
    (112, 32.5, 30.0, 22.5, 5.0, 10.0, 16.75),
    (113, 5.0, 50.0, 30.0, 5.0, 10.0, 0.68),
    (114, 12.5, 42.5, 30.0, 5.0, 10.0, 1.41),
    (115, 20.0, 35.0, 30.0, 5.0, 10.0, 6.88),
    (116, 27.5, 27.5, 30.0, 5.0, 10.0, 27.28),
    (117, 7.5, 42.5, 35.0, 5.0, 10.0, 0.76),
)


def _ce_source_moles(weights: dict[str, float]) -> dict[str, float]:
    ce_rule = trace_component_rule("Ce2O3")
    assert ce_rule is not None
    masses = {**MOLAR_MASS_G_MOL, "Ce2O3": ce_rule.molar_mass_g_mol}
    return {name: amount / masses[name] for name, amount in weights.items()}


def _ce_parent_standard_delta_g(temperature: float) -> float:
    source_path = Path(__file__).resolve().parents[2] / "data-src/konings/ce.json"
    source = json.loads(source_path.read_text())["condensates"]["Ce2O3(l)"]
    phase_a = source["phases"][0]
    reference_temperature = 298.15
    enthalpy_increment = 0.0
    entropy_increment = 0.0
    for coefficient, power in zip(
        phase_a["coefficients"], phase_a["powers"], strict=True
    ):
        if power == -1:
            enthalpy_increment += coefficient * math.log(
                temperature / reference_temperature
            )
        else:
            enthalpy_increment += coefficient * (
                temperature ** (power + 1) - reference_temperature ** (power + 1)
            ) / (power + 1)
        if power == 0:
            entropy_increment += coefficient * math.log(
                temperature / reference_temperature
            )
        else:
            entropy_increment += coefficient * (
                temperature**power - reference_temperature**power
            ) / power
    g_stable_solid = (
        source["dfH298_kJ_mol"] * 1000.0
        + enthalpy_increment
        - temperature * (source["S298_J_mol_K"] + entropy_increment)
    )
    gas_pack = load_gas_datapack()
    liquid_row = _nearest_interval_row(gas_pack.oxide_df, "Ce2O3(l)", temperature, True)
    g_parent_liquid = _lamor_gibbs(temperature, liquid_row)
    return g_parent_liquid - g_stable_solid


def _ce_parent_standard_factor(temperature: float = 1873.0) -> float:
    delta_g = _ce_parent_standard_delta_g(temperature)
    if temperature == 1873.0:
        assert delta_g == pytest.approx(32_749.41, abs=0.02)
    return math.exp(-delta_g / (R_J_MOL_K * temperature))


def test_ce_equations_match_table_anchors_and_full_source_closure() -> None:
    assert _CE_ZHAO_TABLE_IV[0] == (301, 45.0, 50.0, 5.0, 3.29)
    assert _CE_ZHAO_TABLE_IV[4] == (305, 38.0, 52.0, 10.0, 3.64)
    assert _CE_ZHAO_TABLE_IV[-1] == (314, 18.75, 46.25, 35.0, 18.20)
    assert _CE_CHEN_TABLE_2[0] == (101, 22.5, 57.5, 5.0, 5.0, 10.0, 0.24)
    assert _CE_CHEN_TABLE_2[3] == (104, 10.0, 60.0, 15.0, 5.0, 10.0, 0.18)
    assert _CE_CHEN_TABLE_2[-1] == (117, 7.5, 42.5, 35.0, 5.0, 10.0, 0.76)

    parent_standard_factor = _ce_parent_standard_factor()
    zhao_residuals = []
    for sample, cao, sio2, ce2o3, activity_x1e7 in _CE_ZHAO_TABLE_IV:
        composition = _ce_source_moles(
            {"CaO": cao, "SiO2": sio2, "Ce2O3": ce2o3, "Al2O3": 0.0, "MgO": 0.0}
        )
        total = math.fsum(composition.values())
        x_cao = composition["CaO"] / total
        x_sio2 = composition["SiO2"] / total
        x_ce = composition["Ce2O3"] / total
        ln_gamma = (
            4.5
            - 14.5 * x_cao**2
            - 27.7 * x_sio2**2
            - 17.5 * x_cao * x_sio2
        )
        observed_gamma = activity_x1e7 * 1e-7 / x_ce
        predicted_gamma = math.exp(ln_gamma)
        result = evaluate_review_gamma_activity(
            "Ce", composition, 1873.0, parent_oxide="Ce2O3",
            fO2_bar=4.76e-16 * 1.01325,
        )
        assert result.row == "Ce2O3-Zhao-2019-CaO-SiO2-1873K"
        assert result.gamma / parent_standard_factor == pytest.approx(predicted_gamma, rel=1e-12)
        zhao_residuals.append(math.log(predicted_gamma / observed_gamma))

    chen_residuals = []
    for sample, cao, sio2, al2o3, mgo, ce2o3, activity_x1e7 in _CE_CHEN_TABLE_2:
        composition = _ce_source_moles({
            "CaO": cao, "SiO2": sio2, "Al2O3": al2o3, "MgO": mgo,
            "Ce2O3": ce2o3,
        })
        total = math.fsum(composition.values())
        x_ce = composition["Ce2O3"] / total
        ln_gamma = 3.46 * ((cao + 0.72 * al2o3) / sio2) - 14.93
        observed_gamma = activity_x1e7 * 1e-7 / x_ce
        predicted_gamma = math.exp(ln_gamma)
        result = evaluate_review_gamma_activity(
            "Ce", composition, 1873.0, parent_oxide="Ce2O3",
            fO2_bar=4.76e-16 * 1.01325,
        )
        assert result.row == "Ce2O3-Chen-2020-CMAS-1873K"
        assert result.gamma / parent_standard_factor == pytest.approx(predicted_gamma, rel=1e-12)
        chen_residuals.append(math.log(predicted_gamma / observed_gamma))

    zhao_rms = math.sqrt(math.fsum(value * value for value in zhao_residuals) / 14)
    chen_rms = math.sqrt(math.fsum(value * value for value in chen_residuals) / 17)
    assert zhao_rms == pytest.approx(0.0751, abs=1e-4)
    assert chen_rms == pytest.approx(0.2919, abs=1e-4)


def test_ce_fusion_direction_band_and_oxygen_screen() -> None:
    composition = _ce_source_moles({
        "CaO": 22.5, "SiO2": 57.5, "Al2O3": 5.0, "MgO": 5.0,
        "Ce2O3": 10.0,
    })
    result = evaluate_review_gamma_activity(
        "Ce", composition, 1873.0, parent_oxide="Ce2O3",
        fO2_bar=4.76e-16 * 1.01325,
    )
    fusion_factor = _ce_parent_standard_factor()
    transition_variance = math.fsum((
        (8_000.0 * (1.0 - 1873.0 / 2392.0)) ** 2,
        (5_000.0 * (1.0 - 1873.0 / 2406.0)) ** 2,
        (10_000.0 * (1.0 - 1873.0 / 2512.0)) ** 2,
    ))
    ln_band = math.hypot(0.2919, math.sqrt(transition_variance) / (R_J_MOL_K * 1873.0))

    assert result.row == "Ce2O3-Chen-2020-CMAS-1873K"
    assert result.gamma < result.gamma / fusion_factor
    assert result.gamma / (result.gamma / fusion_factor) == pytest.approx(
        fusion_factor, rel=1e-12
    )
    assert fusion_factor == pytest.approx(0.122094, rel=1e-5)
    assert result.gamma_range == pytest.approx((
        result.gamma * math.exp(-ln_band),
        result.gamma * math.exp(ln_band),
    ))
    assert any("transition enthalpy uncertainties" in item for item in result.notice.limitations)
    assert not any(
        "source used graphite and pure CO" in axis
        for axis in result.notice.violated_axes
    )

    no_alumina = _ce_source_moles({
        "CaO": 45.0, "SiO2": 50.0, "Al2O3": 0.0, "MgO": 0.0,
        "Ce2O3": 5.0,
    })
    zhao = evaluate_review_gamma_activity(
        "Ce", no_alumina, 1873.0, parent_oxide="Ce2O3",
        fO2_bar=4.76e-16 * 1.01325 * 10.0,
    )
    assert zhao.row == "Ce2O3-Zhao-2019-CaO-SiO2-1873K"
    assert any(
        "source used graphite and pure CO" in axis
        for axis in zhao.notice.violated_axes
    )

    trace_alumina = _ce_source_moles({
        "CaO": 45.0, "SiO2": 50.0, "Al2O3": 1e-8, "MgO": 0.0,
        "Ce2O3": 5.0,
    })
    chen = evaluate_review_gamma_activity(
        "Ce", trace_alumina, 1873.0, parent_oxide="Ce2O3",
    )
    assert chen.row == "Ce2O3-Chen-2020-CMAS-1873K"
    assert any("Al2O3 is outside" in axis for axis in chen.notice.violated_axes)


def test_ce_standard_conversion_tracks_temperature_and_keeps_band_width() -> None:
    composition = _ce_source_moles({
        "CaO": 22.5, "SiO2": 57.5, "Al2O3": 5.0, "MgO": 5.0,
        "Ce2O3": 10.0,
    })
    solid_gamma = math.exp(3.46 * ((22.5 + 0.72 * 5.0) / 57.5) - 14.93)
    expected_delta_g = {
        1700.0: 42_684.311,
        1800.0: 36_922.417,
        1873.0: 32_749.411,
        2000.0: 25_562.620,
        2100.0: 19_974.126,
    }
    for temperature, expected in expected_delta_g.items():
        result = evaluate_review_gamma_activity(
            "Ce", composition, temperature, parent_oxide="Ce2O3"
        )
        assert "ce2o3-liquid-standard-gibbs-v3" in result.reference
        delta_g = -R_J_MOL_K * temperature * math.log(result.gamma / solid_gamma)
        assert delta_g == pytest.approx(_ce_parent_standard_delta_g(temperature), abs=1.0)
        assert delta_g == pytest.approx(expected, abs=1.0)

        transition_variance = math.fsum((
            (8_000.0 * (1.0 - temperature / 2392.0)) ** 2,
            (5_000.0 * (1.0 - temperature / 2406.0)) ** 2,
            (10_000.0 * (1.0 - temperature / 2512.0)) ** 2,
        ))
        ln_band = math.hypot(
            0.2919, math.sqrt(transition_variance) / (R_J_MOL_K * temperature)
        )
        assert result.gamma_range == pytest.approx((
            result.gamma * math.exp(-ln_band),
            result.gamma * math.exp(ln_band),
        ))

        if temperature == 1873.0:
            assert result.activity == pytest.approx(
                float.fromhex("0x1.023f9a65705dep-28"), rel=1e-9
            )


def test_selector_no_match_returns_a_depends_flagged_nearest_row(monkeypatch) -> None:
    def selector_row(name: str, component: str) -> dict[str, object]:
        return {
            "row": name,
            "parent_oxide": "SnO",
            "gamma_law": {
                "kind": "range",
                "gamma_endpoints": [1.0, 1.0],
                "gamma_point": 1.0,
                "temperature_K": [1800.0, 1800.0],
            },
            "source_units_per_parent": 1.0,
            "parent_power": 1.0,
            "source_mole_fraction_kind": "simple_oxide",
            "temperature_window_K": [1800.0, 1800.0],
            "composition_screen": {
                "kind": "supported_component_set",
                "components": ["SiO2", "SnO"],
            },
            "select_when_component_positive": component,
            "component": "SnO(l)",
            "component_basis": "liquid SnO formula-unit mole fraction",
            "parent_standard_state": "pure liquid SnO, 1 bar",
            "source_citation": "Synthetic selector case",
            "source_locator": "test rows",
            "log10_fO2_source": None,
            "limitations": [],
        }

    rows = [
        selector_row("SnO-select-Al2O3", "Al2O3"),
        selector_row("SnO-select-MgO", "MgO"),
    ]
    monkeypatch.setattr(
        review_gamma_activity,
        "load_activity_fragment",
        lambda _element: {"rows": rows},
    )

    result = evaluate_review_gamma_activity(
        "Sn", {"SiO2": 1.0, "SnO": 0.001}, 1800.0, parent_oxide="SnO"
    )

    assert result.row == "SnO-select-Al2O3"
    assert "review_gamma_selector_no_match" in result.notice.violated_axes
    assert "DEPENDS: review_gamma_selector_no_match" in result.reference


def test_la_binary_dilute_limit_recomputes_table_2_and_domain_flags() -> None:
    # At x_La2O3 -> 0, ln(gamma_inf) = (L0-L1+L2-L3)/(R*T).
    # Li et al. Table 2 prints L_j = a_j + b_j*T for La2O3-SiO2.
    parameters = (
        (-352587.83, 78.86),
        (-119125.74, 22.64),
        (513748.94, -211.27),
        (435926.54, -166.77),
    )
    a_sum = math.fsum((-1.0) ** j * a for j, (a, _b) in enumerate(parameters))
    b_sum = math.fsum((-1.0) ** j * b for j, (_a, b) in enumerate(parameters))
    expected_A = a_sum / (8.314 * math.log(10.0))
    expected_B = b_sum / (8.314 * math.log(10.0))
    assert expected_A == pytest.approx(-8130.0768, abs=5e-5)
    assert expected_B == pytest.approx(0.6122121, abs=5e-8)

    row = load_activity_fragment("La")["rows"][0]
    assert row["gamma_law"] == {
        "kind": "log10_linear",
        "A_K": pytest.approx(expected_A, abs=5e-5),
        "B": pytest.approx(expected_B, abs=5e-8),
    }
    assert "La2O3 formula-unit mole fraction" in row["component_basis"]
    assert "one La2O3 oxide formula unit per component unit" in row["component_basis"]
    assert row["source_units_per_parent"] == row["parent_power"] == 1
    assert row["temperature_window_K"] == [1773.0, 1873.0]
    assert row["composition_screen"] == {
        "kind": "family_oxide_set",
        "oxides": ["SiO2"],
    }
    assert "melt_family_extrapolated" in row["flag_labels"]

    composition = {"SiO2": 1.0, "La2O3": 1e-8}
    for temperature in (1773.0, 1873.0):
        result = evaluate_review_gamma_activity(
            "La", composition, temperature, parent_oxide="La2O3"
        )
        assert result.gamma == pytest.approx(10.0 ** (expected_A / temperature + expected_B))
        assert result.gamma_range == (result.gamma, result.gamma)
        assert "temperature_extrapolated" not in result.notice.violated_axes
        assert "melt_family_extrapolated" not in result.notice.violated_axes
    alumina_bearing = evaluate_review_gamma_activity(
        "La", {"SiO2": 1.0, "Al2O3": 0.2, "La2O3": 1e-8},
        1873.0, parent_oxide="La2O3",
    )
    assert any(
        "outside the assessed SiO2-La2O3 binary edge" in item
        for item in alumina_bearing.notice.violated_axes
    )
    reference_value = evaluate_review_gamma_activity(
        "La", composition, 1873.0, parent_oxide="La2O3"
    ).gamma
    assert reference_value == pytest.approx(1.8687e-4, rel=1e-4)

    outside_window = evaluate_review_gamma_activity(
        "La", composition, 1874.0, parent_oxide="La2O3"
    )
    assert "temperature_extrapolated" in outside_window.notice.violated_axes
    basic_melt = evaluate_review_gamma_activity(
        "La",
        {"SiO2": 1.0, "CaO": 0.5, "MgO": 0.2, "FeO": 0.3, "La2O3": 1e-8},
        1873.0,
        parent_oxide="La2O3",
    )
    assert any(
        "outside the assessed SiO2-La2O3 binary edge" in item
        for item in basic_melt.notice.violated_axes
    )
    assert any("No CaO/MgO/FeO interaction is modeled" in item for item in basic_melt.notice.limitations)
    assert any("bias direction for those basic-melt extrapolations is unknown" in item for item in basic_melt.notice.limitations)
    assert any("bias direction depends on the solvent ratio" in item for item in basic_melt.notice.limitations)


def test_la_default_gas_channel_uses_assessed_trace_row_and_flags() -> None:
    supported = {item.oxide for item in supported_trace_oxides()}
    assert "La2O3" in supported

    result = evaluate_gas(
        {},
        1873.0,
        1.0e-8,
        load_gas_datapack(),
        gas_species=("La",),
        composition_wt_pct={
            "SiO2": 49.5,
            "Al2O3": 15.0,
            "CaO": 18.0,
            "MgO": 9.0,
            "FeO": 8.5,
            "La2O3": 0.01,
        },
    )
    assert math.isfinite(result["La"]) and result["La"] > 0.0
    flag = result.domain_flags["La"]
    assert isinstance(flag, TraceActivityFlag)
    assert flag.activity_range is not None
    assert flag.activity_range[0] == flag.activity_range[1]
    assert "review_gamma_provisional" in str(flag)
    assert "assessed phase-diagram fit; not a measured activity" in str(flag)
    assert "uncertainty not stated" in str(flag)
    assert "No CaO/MgO/FeO interaction is modeled" in str(flag)
    assert "melt-family distance" in str(flag)


def test_ce_default_gas_channel_uses_the_new_trace_row() -> None:
    supported = {item.oxide for item in supported_trace_oxides()}
    assert "Ce2O3" in supported

    composition = {
        "SiO2": 47.0,
        "Al2O3": 15.0,
        "CaO": 18.0,
        "MgO": 9.0,
        "FeO": 11.0,
        "Ce2O3": 0.01,
    }
    result = evaluate_gas(
        {},
        1873.0,
        4.76e-16 * 1.01325,
        load_gas_datapack(),
        gas_species=("Ce", "CeO"),
        composition_wt_pct=composition,
    )
    assert math.isfinite(result["Ce"]) and result["Ce"] > 0.0
    assert math.isfinite(result["CeO"]) and result["CeO"] > 0.0

    gas_flag = result.domain_flags["Ce"]
    assert isinstance(gas_flag, TraceActivityFlag)

    dilute_composition = _ce_source_moles({
        **{name: composition[name] for name in ("SiO2", "Al2O3", "CaO", "MgO", "FeO")},
        "Ce2O3": composition["Ce2O3"],
    })
    dilute_activity = evaluate_review_gamma_activity(
        "Ce", dilute_composition, 1873.0, parent_oxide="Ce2O3",
        fO2_bar=4.76e-16 * 1.01325,
    )
    assert any(
        "Ce2O3 differs from the 10 wt%" in axis
        for axis in dilute_activity.notice.violated_axes
    )
    estimate = gas_flag.estimates[0]
    assert estimate.violated_axes == dilute_activity.notice.violated_axes
    assert estimate.limitations == dilute_activity.notice.limitations
    assert any(axis in str(gas_flag) for axis in estimate.violated_axes)
    assert all(item in str(gas_flag) for item in estimate.limitations)

    melt = evaluate(composition, 1873.0, basis_type="wt")
    melt_flag = next(
        notice for notice in melt.labels.notices
        if isinstance(notice, TraceActivityFlag) and notice.oxide == "Ce2O3"
    )
    assert any(axis in str(melt_flag) for axis in estimate.violated_axes)
    assert any(item in str(melt_flag) for item in estimate.limitations)


def test_trace_activity_flag_pickle_preserves_constructor_fields() -> None:
    estimate = TraceActivityEstimate(
        "Ce2O3", 0.25, 0.01, "review_gamma_provisional",
        gamma=2.5,
        reference="source",
        activity_range=(0.2, 0.3),
        x_source=0.005,
        parent_liquid_standard_offset=ParentLiquidStandardOffset(
            "parent_liquid_standard_offset", 1.25, "convert to selected standard"
        ),
    )
    major_flag = TraceMajorBiasFlag(
        "MgO", 0.2, 0.004, False,
        prior_flags=("out of domain: gas row interval",),
    )
    flag = TraceActivityFlag(
        estimate,
        context="gas",
        prior_flags=("gas channel used extrapolated row",),
        major_flags=(major_flag,),
    )

    restored = pickle.loads(pickle.dumps(flag))

    assert str(restored) == str(flag)
    assert vars(restored) == vars(flag)
    assert vars(restored.major_flags[0]) == vars(major_flag)


def test_review_gamma_payloads_pin_changed_and_unchanged_rows() -> None:
    elements = ("Sn", "Ga", "In", "Cu", "Ge", "As", "Bi", "Pb", "W", "Ni", "Co", "Ce")
    payload = []
    non_ce_payload = []
    ce_payload = []
    unchanged_payload = []
    changed_wood_wade_rows = {
        ("Cu", "CuO0.5-1923K"),
        ("Pb", "PbO-BW1102-1923K"),
        ("W", "WO3-Di-An-Fo-1923K"),
    }
    for element in elements:
        for row in load_activity_fragment(element)["rows"]:
            screen = row["composition_screen"]
            if screen["kind"] == "family_oxide_set":
                composition = {name: 1.0 for name in screen["oxides"]}
            elif screen["kind"] == "an_di_target":
                composition = {
                    name: float(weight) / MOLAR_MASS_G_MOL[name]
                    for name, weight in screen["major_wt_pct"].items()
                }
            elif screen["kind"] == "supported_component_set":
                composition = {name: 1.0 for name in screen["components"]}
            else:
                raise AssertionError(screen["kind"])

            parent = row["parent_oxide"]
            composition[parent] = 0.001
            low, high = map(float, row["temperature_window_K"])
            result = evaluate_review_gamma_activity(
                element,
                composition,
                (low + high) / 2.0,
                parent_oxide=parent,
            )
            row_payload = [
                element,
                row["row"],
                float(result.activity).hex(),
                float(result.source_activity).hex(),
                float(result.gamma).hex(),
                [float(value).hex() for value in result.gamma_range],
                float(result.source_mole_fraction).hex(),
                [float(value).hex() for value in result.activity_range],
                result.parent_oxide,
                result.row,
                list(result.notice.violated_axes),
                list(result.notice.limitations),
                str(result.notice),
                result.reference,
            ]
            payload.append(row_payload)
            (ce_payload if element == "Ce" else non_ce_payload).append(row_payload)
            if (element, row["row"]) not in changed_wood_wade_rows:
                unchanged_payload.append(row_payload)

    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode()
    assert len(payload) == 16
    # The full and non-Ce hashes move because the W, Pb, and Cu Wood & Wade
    # activities and source-x references changed. Separate hashes pin the
    # unaffected rows and the unchanged Ce payload.
    assert hashlib.sha256(encoded).hexdigest() == (
        "73b21a31c6ae377d53c304ff4f53e0c785ffc31e5c8db7c08b012d726093546f"
    )

    def payload_hash(rows: list[list[object]]) -> str:
        return hashlib.sha256(
            json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode()
        ).hexdigest()

    assert payload_hash(non_ce_payload) == (
        "5aed62db57c28b8becaa92438b1e9e94a14f9975327fb8ec75997d8a1be154d9"
    )
    assert payload_hash(unchanged_payload) == (
        "e736ce1b850b2f72c1aacffdbb96cfb3c5bca48dbce694596279e2c05b84a1e9"
    )
    assert payload_hash(ce_payload) == (
        "cf1226c0a8fc05611178ac34c4196f784ca909447044cfd8638922b238bff3dc"
    )


def test_review_gamma_parent_registry_preserves_legacy_screening(monkeypatch) -> None:
    legacy_parents = frozenset({
        "SnO", "Ga2O3", "In2O3", "Cu2O", "GeO2", "As2O3", "Bi2O3",
        "Ce2O3", "NiO", "CoO", "PbO", "WO3", "La2O3",
        "V2O3", "Ag2O", "Sb2O3", "Tl2O", "Rb2O",
    })
    assert review_gamma_activity._TRACE_PARENTS == legacy_parents

    new_rule = replace(trace_component_rule("PbO"), gas_parent_oxide="Xx2O3")
    assert "Xx2O3" in _review_gamma_parent_oxides({"new": new_rule})

    elements = ("Sn", "Ga", "In", "Cu", "Ge", "As", "Bi", "Pb", "W", "Ce", "La", "Ni", "Co")

    def grid_payload() -> list[list[object]]:
        payload = []
        for element in elements:
            for row in load_activity_fragment(element)["rows"]:
                screen = row["composition_screen"]
                if screen["kind"] == "family_oxide_set":
                    base = {name: 1.0 for name in screen["oxides"]}
                    variants = (base, {**base, "FeO": 0.5}, {**base, "Al2O3": 0.25})
                elif screen["kind"] == "an_di_target":
                    base = {
                        name: float(weight) / MOLAR_MASS_G_MOL[name]
                        for name, weight in screen["major_wt_pct"].items()
                    }
                    variants = (base, {**base, "MgO": 0.3}, {**base, "CaO": base.get("CaO", 0.0) * 1.1})
                elif screen["kind"] == "supported_component_set":
                    base = {name: 1.0 for name in screen["components"]}
                    variants = (base, {**base, "FeO": 0.5}, {**base, "SiO2": 0.25})
                else:
                    raise AssertionError(screen["kind"])

                parent = row["parent_oxide"]
                low, high = map(float, row["temperature_window_K"])
                for index, composition in enumerate(variants):
                    amounts = {**composition, parent: 0.001 * (index + 1)}
                    result = evaluate_review_gamma_activity(
                        element, amounts, (low + high) / 2.0, parent_oxide=parent
                    )
                    payload.append([
                        element, row["row"], index,
                        float(result.activity).hex(), float(result.source_activity).hex(),
                        float(result.gamma).hex(),
                        [float(value).hex() for value in result.gamma_range],
                        float(result.source_mole_fraction).hex(),
                        [float(value).hex() for value in result.activity_range],
                        result.notice.violated_axes, result.notice.limitations,
                    ])
        return payload

    registry_payload = grid_payload()
    monkeypatch.setattr(review_gamma_activity, "_TRACE_PARENTS", legacy_parents)
    legacy_payload = grid_payload()
    assert registry_payload == legacy_payload
