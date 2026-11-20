from __future__ import annotations

from dataclasses import FrozenInstanceError
import math
import re

import numpy as np
import pytest

from openimcc import (
    ImccTraceLimitExceededError,
    ImccTraceMajorBiasError,
    TraceActivityEstimate,
    TraceActivityFlag,
    element_to_parent_oxide,
    evaluate,
    load_datapack,
    supported_trace_oxides,
)
from openimcc.activity_fragments import load_activity_fragment
from openimcc._constants import MOLAR_MASS_G_MOL
from openimcc.gas import (
    TraceGasChannelRefusal,
    evaluate_gas,
)
from openimcc.gas_data import IMCC_PARENT_OXIDES
from openimcc.kernel import (
    ImccComponentOutsideDomainError,
    ImccCompositionIncompleteError,
)
from openimcc.redox import evaluate_redox
from openimcc.redox_pack import RedoxInputError, load_redox_pack
from openimcc.profiles import ProfileRedoxUnavailableError, load_profile
from openimcc.trace_envelope_calibration import (
    AMPHOTERIC_ENVELOPE_K,
    BASIC_MONOXIDE_ENVELOPE_K,
    CALIBRATION_ROWS,
    ENVELOPE_DELTA_IW_GRID,
    ENVELOPE_T_GRID_K,
    _iw_log10_fO2_bar,
)
import openimcc.trace_components as trace_components
from openimcc.trace_components import (
    TraceComponentsOutsideMeltModelNotice,
    TraceMajorBiasFlag,
    trace_component_rule,
    trace_fractions_from_wt,
)


_MAJOR_MOLES = {oxide: 1.0 for oxide in IMCC_PARENT_OXIDES}
_MARE_BASALT_WT = {
    "SiO2": 46.234,
    "TiO2": 1.587,
    "Al2O3": 14.214,
    "FeO": 16.353,
    "MgO": 9.463,
    "CaO": 11.637,
    "Na2O": 0.182,
}
_ALKALI_RICH_WT = {
    "SiO2": 50.0,
    "Al2O3": 12.0,
    "CaO": 6.0,
    "FeO": 5.0,
    "MgO": 5.0,
    "TiO2": 2.0,
    "Na2O": 12.0,
    "K2O": 8.0,
}


def _weight_composition(trace_fractions: dict[str, float]) -> dict[str, float]:
    trace_total = math.fsum(trace_fractions.values())
    assert 0.0 <= trace_total < 1.0
    major_scale = 1.0 / math.fsum(_MAJOR_MOLES.values())
    composition = {
        oxide: amount * major_scale * MOLAR_MASS_G_MOL[oxide]
        for oxide, amount in _MAJOR_MOLES.items()
    }
    for oxide, fraction in trace_fractions.items():
        rule = trace_component_rule(oxide)
        assert rule is not None
        composition[oxide] = (
            fraction / (1.0 - trace_total)
            * rule.molar_mass_g_mol
        )
    return composition


def test_trace_fractions_reject_negative_major_before_normalizing() -> None:
    with pytest.raises(ImccCompositionIncompleteError, match="negative"):
        trace_fractions_from_wt({"SiO2": -10.0, "NiO": 1.0})


@pytest.mark.parametrize("value", [True, np.bool_(True)])
def test_trace_fractions_reject_boolean_major_weight(value: object) -> None:
    with pytest.raises(ImccCompositionIncompleteError, match="SiO2"):
        trace_fractions_from_wt({"SiO2": value, "NiO": 1.0})


def _major_activities(result) -> dict[str, float]:
    return dict(zip(result.parent_oxides, result.parent_activity, strict=True))


def test_moo3_review_gamma_uses_public_evaluate_and_table7_domain() -> None:
    fragment = load_activity_fragment("Mo")
    assert fragment["rows"][0]["parent_standard_state"].startswith(
        "pure liquid MoO3"
    )

    source_composition = {
        "CaO": 19.1,
        "MgO": 8.5,
        "Al2O3": 12.1,
        "SiO2": 40.7,
        "TiO2": 19.6,
        "MoO3": 0.001,
    }
    result = evaluate(
        source_composition, 1673.15, basis_type="wt", allow_extrapolation=True
    )
    notice = next(
        item for item in result.labels.notices
        if isinstance(item, TraceComponentsOutsideMeltModelNotice)
    )
    estimate = notice.activities["MoO3"]
    assert estimate.gamma == pytest.approx(0.327281, abs=2.0e-5)
    assert estimate.x_source is not None
    assert estimate.activity_range is not None
    gamma_band = tuple(value / estimate.x_source for value in estimate.activity_range)
    assert gamma_band[0] <= 0.325 <= gamma_band[1]
    assert abs(estimate.gamma - 0.325) <= 0.036
    assert estimate.reference is not None
    assert "pure liquid MoO3" in estimate.reference

    outside_domain = {
        **source_composition,
        "TiO2": 18.6,
        "Na2O": 1.0,
    }
    outside_result = evaluate(
        outside_domain, 1800.0, basis_type="wt", allow_extrapolation=True
    )
    outside_notice = next(
        item for item in outside_result.labels.notices
        if isinstance(item, TraceComponentsOutsideMeltModelNotice)
    )
    outside_estimate = outside_notice.activities["MoO3"]
    assert any(
        "1400 °C (1673.15 K) only" in warning
        for warning in outside_estimate.violated_axes
    )
    assert any(
        "Na2O is absent" in warning
        for warning in outside_estimate.violated_axes
    )


def test_shared_oxide_counts_match_previous_logic_for_default_components() -> None:
    components = set(load_datapack().parent_oxides)
    redox_pack = load_redox_pack()
    components.update(str(parent["parent_oxide"]) for parent in redox_pack.parents)
    components.update(str(name) for name in redox_pack.content["thermo_species"])
    components.update(trace_components._TRACE_COMPONENT_RULES)

    for component in sorted(components):
        model_match = re.fullmatch(r"([A-Z][a-z]?)(\d*)O(\d*)", component)
        old_model_counts = (
            None
            if model_match is None
            else (
                int(model_match.group(2) or 1),
                int(model_match.group(3) or 1),
            )
        )
        assert (
            trace_components._single_cation_oxide_counts(component)
            == old_model_counts
        )

        gamma_match = re.fullmatch(r"([A-Z][a-z]?)(\d*)O\d*", component)
        old_gamma_cations = (
            int(gamma_match.group(2) or 1) if gamma_match is not None else 1
        )
        new_counts = trace_components._single_cation_oxide_counts(component)
        new_gamma_cations = new_counts[0] if new_counts is not None else 1
        assert new_gamma_cations == old_gamma_cations


def _mole_fraction_composition(
    base_weights: dict[str, float], trace_oxide: str, x_trace: float
) -> dict[str, float]:
    base_moles = {
        oxide: amount / MOLAR_MASS_G_MOL[oxide]
        for oxide, amount in base_weights.items()
    }
    base_total = math.fsum(base_moles.values())
    return {
        **{
            oxide: amount / base_total * (1.0 - x_trace)
            for oxide, amount in base_moles.items()
        },
        trace_oxide: x_trace,
    }


def test_supported_trace_listing_matches_registry_and_shared_limits() -> None:
    supported = supported_trace_oxides()
    assert isinstance(supported, tuple)
    assert tuple(item.oxide for item in supported) == tuple(
        sorted(trace_components._TRACE_COMPONENT_RULES)
    )

    source_kinds = {
        "redox_associates": "redox_associates",
        "activity_fragment": "helper",
        "review_gamma": "review_gamma",
        "ideal": "ideal_flagged",
    }
    for item in supported:
        rule = trace_components._TRACE_COMPONENT_RULES[item.oxide]
        assert item.trace_class == rule.trace_class
        assert item.full_component is rule.full_component
        assert item.activity_source == source_kinds[rule.activity_model]
        assert item.max_mole_fraction == (
            None
            if rule.full_component
            else trace_components._TRACE_COMPONENT_MAX_MOLE_FRACTION
        )
        assert item.max_total_mole_fraction == (
            None
            if rule.full_component
            else trace_components._TRACE_COMPONENT_MAX_TOTAL_MOLE_FRACTION
        )
        assert (
            item.gas_channel_blocked_above_mole_fraction
            == rule.gas_channel_blocked_above_mole_fraction
        )
        if rule.gas_channel_blocked_above_mole_fraction is not None:
            assert (
                item.gas_channel_blocked_above_mole_fraction
                == trace_components._TRACE_GAS_CHANNEL_BLOCKING_MOLE_FRACTION
            )

    with pytest.raises(FrozenInstanceError):
        supported[0].oxide = "changed"


def test_supported_trace_listing_is_deterministic() -> None:
    first = supported_trace_oxides()
    second = supported_trace_oxides()
    assert first == second
    assert tuple(item.oxide for item in first) == tuple(
        sorted(item.oxide for item in second)
    )


def test_supported_trace_components_are_accepted_at_dilute_loading() -> None:
    for item in supported_trace_oxides():
        composition = _mole_fraction_composition(
            _MARE_BASALT_WT, item.oxide, 0.001
        )
        try:
            evaluate(
                composition, 1800.0, basis_type="mol", allow_extrapolation=True
            )
        except Exception as error:
            pytest.fail(f"{item.oxide} was refused at dilute loading: {error}")


def test_mno_is_returned_as_a_full_component_when_supplied() -> None:
    result = evaluate(
        _mole_fraction_composition(_MARE_BASALT_WT, "MnO", 0.001),
        1800.0,
        basis_type="mol",
        allow_extrapolation=True,
    )
    assert "MnO" in result.parent_oxides
    mno_activity = result.parent_activity[result.parent_oxides.index("MnO")]
    assert math.isfinite(float(mno_activity))


def test_mno_redox_uses_profile_selected_melt_and_redox_packs() -> None:
    selected_profile = load_profile("openimcc")
    selected_melt = selected_profile.melt_pack
    composition = {oxide: 1.0 for oxide in selected_melt.parent_oxides}
    composition["MnO"] = 0.01
    result = evaluate(
        composition,
        1800.0,
        profile=selected_profile,
        basis_type="mol",
        allow_extrapolation=True,
        allow_out_of_envelope=True,
    )

    assert result.receipt.data_digests["melt"] == selected_melt.binding_digest
    assert result.receipt.data_digests["redox"] == (
        selected_profile.redox_pack.canonical_digest
    )
    assert "MnO" in result.parent_oxides
    assert {"MnSiO3(l)", "Mn2SiO4(l)", "MnAl2O4(l)"} <= set(
        result.species_names
    )


def test_mno_redox_refuses_incompatible_selected_melt_pack() -> None:
    selected_melt = load_datapack(
        "src/openimcc/data/packs/imcc-sf04-v1.0.2.json"
    )
    composition = {oxide: 1.0 for oxide in selected_melt.parent_oxides}
    composition["MnO"] = 0.01

    with pytest.raises(RedoxInputError, match="D066 v4 liquid-oxide standards"):
        evaluate(
            composition,
            1800.0,
            pack=selected_melt,
            basis_type="mol",
            allow_extrapolation=True,
            allow_out_of_envelope=True,
        )


def test_mno_redox_refuses_profile_without_redox_pack() -> None:
    with pytest.raises(ProfileRedoxUnavailableError):
        evaluate(
            {
                "SiO2": 50.0,
                "Al2O3": 15.0,
                "MgO": 10.0,
                "CaO": 12.0,
                "FeO": 8.0,
                "Na2O": 2.0,
                "K2O": 1.0,
                "TiO2": 1.0,
                "MnO": 0.4,
            },
            1800.0,
            profile=load_profile("published"),
            basis_type="wt",
            allow_extrapolation=True,
            allow_out_of_envelope=True,
        )


def test_gas_mno_requires_selected_profile_redox_pack() -> None:
    with pytest.raises(ProfileRedoxUnavailableError):
        evaluate_gas(
            {oxide: 1.0 for oxide in IMCC_PARENT_OXIDES},
            1800.0,
            1.0e-8,
            profile=load_profile("published"),
            composition_wt_pct={
                "SiO2": 50.0,
                "Al2O3": 15.0,
                "MgO": 10.0,
                "CaO": 12.0,
                "FeO": 8.0,
                "Na2O": 2.0,
                "K2O": 1.0,
                "TiO2": 1.0,
                "MnO": 0.4,
            },
        )


_OXIDE_ATOMS = {
    "SiO2": {"Si": 1, "O": 2},
    "MgO": {"Mg": 1, "O": 1},
    "FeO": {"Fe": 1, "O": 1},
    "CaO": {"Ca": 1, "O": 1},
    "Al2O3": {"Al": 2, "O": 3},
    "TiO2": {"Ti": 1, "O": 2},
    "Na2O": {"Na": 2, "O": 1},
    "K2O": {"K": 2, "O": 1},
    "MnO": {"Mn": 1, "O": 1},
    "Cr2O3": {"Cr": 2, "O": 3},
}


def _redox_inventory(oxide_moles: dict[str, float]) -> dict[str, float]:
    inventory: dict[str, float] = {}
    for oxide, amount in oxide_moles.items():
        for element, count in _OXIDE_ATOMS[oxide].items():
            inventory[element] = inventory.get(element, 0.0) + amount * count
    return inventory


def test_trace_thresholds_are_inclusive_and_keep_the_existing_refusal() -> None:
    at_individual_limit = _weight_composition({"NiO": 0.01})
    accepted = evaluate(
        at_individual_limit, 1800.0, basis_type="wt", allow_extrapolation=True
    )
    notice = next(
        item for item in accepted.labels.notices
        if isinstance(item, TraceComponentsOutsideMeltModelNotice)
    )
    assert notice.components["NiO"] == pytest.approx(0.01, abs=1e-15)

    at_total_limit = _weight_composition({"NiO": 0.01, "CoO": 0.01})
    accepted = evaluate(
        at_total_limit, 1800.0, basis_type="wt", allow_extrapolation=True
    )
    notice = next(
        item for item in accepted.labels.notices
        if isinstance(item, TraceComponentsOutsideMeltModelNotice)
    )
    assert notice.total == pytest.approx(0.02, abs=1e-15)

    above_individual_limit = _weight_composition({"NiO": 0.0101})
    with pytest.raises(ImccComponentOutsideDomainError) as individual_error:
        evaluate(
            above_individual_limit,
            1800.0,
            basis_type="wt",
            allow_extrapolation=True,
        )
    assert str(individual_error.value).startswith(
        "component(s) outside IMCC-SF04 domain: ['NiO'];"
    )

    above_total_limit = _weight_composition(
        {"NiO": 0.007, "CoO": 0.007, "SrO": 0.007}
    )
    with pytest.raises(ImccComponentOutsideDomainError) as total_error:
        evaluate(
            above_total_limit,
            1800.0,
            basis_type="wt",
            allow_extrapolation=True,
        )
    assert str(total_error.value).startswith(
        "component(s) outside IMCC-SF04 domain: ['CoO', 'NiO', 'SrO']"
    )

    with pytest.raises(ImccCompositionIncompleteError, match="negative values"):
        evaluate(
            {**_weight_composition({}), "NiO": -0.1},
            1800.0,
            basis_type="wt",
            allow_extrapolation=True,
        )


@pytest.mark.parametrize(
    ("traces", "components"),
    [
        ({"NiO": 0.0101, "MnO": 0.05}, ("NiO",)),
        ({"V2O3": 0.0101}, ("V2O3",)),
        ({"Ag2O": 0.0101}, ("Ag2O",)),
        ({"Sb2O3": 0.0101}, ("Sb2O3",)),
        ({"Tl2O": 0.0101}, ("Tl2O",)),
        ({"Rb2O": 0.0101}, ("Rb2O",)),
        ({"NiO": 0.007, "CoO": 0.007, "SrO": 0.007},
         ("CoO", "NiO", "SrO")),
    ],
)
def test_trace_limit_error_has_read_only_structured_details(
    traces: dict[str, float], components: tuple[str, ...]
) -> None:
    with pytest.raises(ImccTraceLimitExceededError) as refused:
        evaluate(
            _weight_composition(traces),
            1800.0,
            basis_type="wt",
            allow_extrapolation=True,
        )

    error = refused.value
    assert isinstance(error, ImccComponentOutsideDomainError)
    assert error.code == "imcc_trace_limit_exceeded"
    assert error.components == components
    assert error.fractions == pytest.approx(
        {name: traces[name] for name in components}, abs=1.0e-15
    )
    assert error.per_component_limit == (
        trace_components._TRACE_COMPONENT_MAX_MOLE_FRACTION
    )
    assert error.total_limit == (
        trace_components._TRACE_COMPONENT_MAX_TOTAL_MOLE_FRACTION
    )
    assert error.full_component_exempt == (("MnO",) if "MnO" in traces else ())
    assert str(error).startswith(
        f"component(s) outside IMCC-SF04 domain: {list(components)}"
    )
    with pytest.raises(AttributeError):
        error.components = ()
    with pytest.raises(TypeError):
        error.fractions[components[0]] = 0.0


def test_unknown_component_refusal_is_not_a_trace_limit_error() -> None:
    with pytest.raises(ImccComponentOutsideDomainError) as unknown:
        evaluate({"SiO2": 1.0, "Unknown": 1.0}, 1800.0)
    assert not isinstance(unknown.value, ImccTraceLimitExceededError)


def test_per_major_bounds_flag_rows_and_trace_target_returns_estimate() -> None:
    composition = _weight_composition({"P2O5": 0.002})
    result = evaluate(
        composition, 1800.0, basis_type="wt", allow_extrapolation=True
    )
    notice = next(
        item for item in result.labels.notices
        if isinstance(item, TraceComponentsOutsideMeltModelNotice)
    )
    assert notice.code == "trace_components_outside_melt_model"
    assert notice.total == pytest.approx(0.002, abs=1e-15)
    flags = [
        item for item in result.labels.notices
        if isinstance(item, TraceMajorBiasFlag)
    ]
    assert flags
    silica_flag = next(item for item in flags if item.major_oxide == "SiO2")
    assert notice.bias_bound_dex["SiO2"] == pytest.approx(
        math.log10(1.0 + 4.0 * 0.002 / silica_flag.x_free)
    )
    refused = next(item for item in flags if not item.scoreable)
    with pytest.raises(ImccTraceMajorBiasError, match="exceeds 0.01 dex") as error:
        result.activity(refused.major_oxide)
    assert isinstance(error.value, ImccComponentOutsideDomainError)
    assert error.value.code == refused.code
    assert error.value.major_oxide == refused.major_oxide
    with pytest.raises(AttributeError):
        error.value.major_oxide = "MgO"
    assert result.activity("P2O5") == pytest.approx(notice.activities["P2O5"].activity)


def test_unverified_monoxide_envelope_uses_strict_bound_and_caps_major_rows() -> None:
    dilute = evaluate(
        _mole_fraction_composition(_MARE_BASALT_WT, "CoO", 0.001),
        1800.0,
        basis_type="mol",
        allow_extrapolation=True,
    )
    dilute_flags = [
        item for item in dilute.labels.notices
        if isinstance(item, TraceMajorBiasFlag)
    ]
    sodium = next(item for item in dilute_flags if item.major_oxide == "Na2O")
    assert sodium.bias_bound_dex == pytest.approx(
        math.log10(1.0 + 0.002 / sodium.x_free)
    )
    assert sodium.envelope_flags == ()
    assert sodium.mn_calibrated_envelope is None
    assert "DEPENDS" in str(sodium)
    assert "calibration melt and oxygen conditions are not evidenced" in str(sodium)
    assert "mn_calibrated_envelope_v3" in str(sodium)
    assert sodium.limiting_cap_dex == 0.01
    assert not sodium.scoreable
    with pytest.raises(ImccTraceMajorBiasError):
        dilute.activity("Na2O")

    concentrated = evaluate(
        _mole_fraction_composition(_MARE_BASALT_WT, "CoO", 0.01),
        1800.0,
        basis_type="mol",
        allow_extrapolation=True,
    )
    concentrated_flags = [
        item for item in concentrated.labels.notices
        if isinstance(item, TraceMajorBiasFlag)
    ]
    assert concentrated_flags
    assert all(not item.scoreable for item in concentrated_flags)
    assert all(item.bias_bound_dex > 0.01 for item in concentrated_flags)
    with pytest.raises(ImccComponentOutsideDomainError, match="exceeds 0.01 dex"):
        concentrated.activity("Na2O")

    amphoteric = evaluate(
        _mole_fraction_composition(_MARE_BASALT_WT, "Cr2O3", 0.001),
        1800.0,
        basis_type="mol",
        allow_extrapolation=True,
    )
    chromium = next(
        item for item in amphoteric.labels.notices
        if isinstance(item, TraceMajorBiasFlag) and item.major_oxide == "Na2O"
    )
    weighted_cr_trace = (
        1.0 + trace_component_rule("Cr2O3").nu
    ) * 0.001
    assert chromium.bias_bound_dex == pytest.approx(
        math.log10(1.0 + weighted_cr_trace / chromium.x_free)
    )
    assert chromium.cr_calibrated_envelope is None
    assert "cr_calibrated_envelope_v3" in str(chromium)
    assert "DEPENDS" in str(chromium)


@pytest.mark.parametrize("temperature_K", [1399.0, 1400.0, 2200.0, 2201.0])
def test_trace_envelope_requires_evidenced_melt_and_oxygen_conditions(
    temperature_K: float,
) -> None:
    result = evaluate(
        _mole_fraction_composition(_MARE_BASALT_WT, "CoO", 0.001),
        temperature_K,
        basis_type="mol",
        allow_extrapolation=True,
    )
    sodium = next(
        item for item in result.labels.notices
        if isinstance(item, TraceMajorBiasFlag) and item.major_oxide == "Na2O"
    )
    assert sodium.envelope_flags == ()
    assert sodium.limiting_cap_dex == 0.01
    assert "mn_calibrated_envelope_v3" in str(sodium)
    assert "DEPENDS" in str(sodium)
    if temperature_K < 1400.0 or temperature_K > 2200.0:
        assert f"temperature outside 1400–2200 K" in str(sodium)


def test_gas_trace_bias_uses_melt_envelope_flags(packaged_gas_pack) -> None:
    def evaluate_composition(x_coo: float) -> tuple[dict[str, float], dict[str, float]]:
        mole_composition = _mole_fraction_composition(
            {name: 1.0 for name in IMCC_PARENT_OXIDES}, "CoO", x_coo
        )
        weight_composition = {
            oxide: amount * MOLAR_MASS_G_MOL[oxide]
            for oxide, amount in mole_composition.items()
        }
        melt = evaluate(
            weight_composition,
            1800.0,
            basis_type="wt",
            allow_extrapolation=True,
        )
        return _major_activities(melt), weight_composition

    dilute_activities, dilute_composition = evaluate_composition(0.001)
    with pytest.raises(ImccTraceMajorBiasError) as sodium_refusal:
        evaluate_gas(
            dilute_activities,
            1800.0,
            1.0e-8,
            packaged_gas_pack,
            gas_species="Na",
            composition_wt_pct=dilute_composition,
        )
    assert sodium_refusal.value.major_oxide == "Na2O"

    dilute_gas = evaluate_gas(
        dilute_activities,
        1800.0,
        1.0e-8,
        packaged_gas_pack,
        gas_species="Mg",
        composition_wt_pct=dilute_composition,
    )
    assert "Mg" in dilute_gas
    magnesium_flag = dilute_gas.domain_flags["Mg"]
    assert isinstance(magnesium_flag, TraceMajorBiasFlag)
    assert magnesium_flag.mn_calibrated_envelope is None
    assert "DEPENDS" in str(magnesium_flag)
    assert magnesium_flag.scoreable
    assert any(
        "constant-Cp supercooled-liquid continuation" in flag
        for flag in magnesium_flag.prior_flags
    )
    assert any("±0.76 dex pO2" in flag for flag in magnesium_flag.prior_flags)

    concentrated_activities, concentrated_composition = evaluate_composition(0.01)
    with pytest.raises(ImccTraceMajorBiasError) as refusal:
        evaluate_gas(
            concentrated_activities,
            1800.0,
            1.0e-8,
            packaged_gas_pack,
            gas_species=("Na", "SiO"),
            composition_wt_pct=concentrated_composition,
        )
    assert isinstance(refusal.value, ImccComponentOutsideDomainError)
    assert refusal.value.code == TraceMajorBiasFlag.code
    assert refusal.value.major_oxide == "Na2O"


def test_ga_gas_conversion_identity_and_kems_fit_domain(packaged_gas_pack) -> None:
    temperature = 1700.0
    an_di = {
        "SiO2": 50.34,
        "Al2O3": 15.4,
        "MgO": 10.8,
        "CaO": 23.46,
        "Ga2O3": 0.05,
    }

    def gas_at(composition: dict[str, float], delta_iw: float, T: float = temperature):
        f_o2 = 10.0 ** (_iw_log10_fO2_bar(T) + delta_iw)
        return evaluate_gas(
            {}, T, f_o2, packaged_gas_pack,
            gas_species="Ga", composition_wt_pct=composition,
        )

    inside = gas_at(an_di, 2.0)
    assert inside["Ga"].hex() == "0x1.1b42c3b4c2a01p-35"
    inside_flag = inside.domain_flags["Ga"]
    assert inside_flag.code == "nasa_standard_converted_activity_v1"
    assert inside_flag.reference is not None
    assert "NASA standard" in inside_flag.reference
    assert "pure liquid Ga2O3 on the Bischof KEMS parent standard" not in (
        inside_flag.reference
    )
    assert inside_flag.parent_liquid_standard_offset is not None
    assert "nasa_standard_activity_ratio_v1" in (
        inside_flag.parent_liquid_standard_offset.recommendation
    )
    assert "inside the measured 1550–1740 K KEMS fit" in str(inside_flag)

    oxygen_outside = gas_at(an_di, -6.0)
    oxygen_flag = oxygen_outside.domain_flags["Ga"]
    assert "out of domain: oxygen_fugacity_extrapolated" in str(oxygen_flag)
    assert "inside the measured 1550–1740 K KEMS fit" not in str(oxygen_flag)

    composition_outside = gas_at(
        {"SiO2": 90.0, "Al2O3": 5.0, "MgO": 2.0, "CaO": 2.0, "Ga2O3": 0.05},
        2.0,
    )
    assert "out of domain: melt_composition_differs_from_An_Di_target" in str(
        composition_outside.domain_flags["Ga"]
    )

    temperature_outside = gas_at(an_di, 2.0, T=1741.0)
    assert "out of domain: temperature_extrapolated" in str(
        temperature_outside.domain_flags["Ga"]
    )

    indium_composition = {
        "SiO2": 50.34,
        "Al2O3": 15.4,
        "MgO": 10.8,
        "CaO": 23.46,
        "In2O3": 0.05,
    }
    indium_temperature = 1674.0
    indium_f_o2 = 10.0 ** (
        _iw_log10_fO2_bar(indium_temperature) + 2.0
    )
    indium = evaluate_gas(
        {}, indium_temperature, indium_f_o2, packaged_gas_pack,
        gas_species="In", composition_wt_pct=indium_composition,
    )
    indium_flag = indium.domain_flags["In"]
    assert indium_flag.code == "nasa_standard_converted_activity_v1"
    assert indium_flag.reference is not None
    assert "NASA standard" in indium_flag.reference
    assert "inside the measured 1550–1740 K KEMS fit" in str(indium_flag)


def test_ga_nasa_conversion_overflow_is_a_typed_trace_refusal(
    monkeypatch: pytest.MonkeyPatch,
    packaged_gas_pack,
) -> None:
    from types import SimpleNamespace

    import openimcc.model as model_module

    monkeypatch.setattr(
        model_module,
        "evaluate",
        lambda *_args, **_kwargs: SimpleNamespace(
            labels=SimpleNamespace(notices=())
        ),
    )
    with pytest.raises(ImccComponentOutsideDomainError) as refused:
        evaluate_gas(
            {},
            1.0,
            1.0e-8,
            packaged_gas_pack,
            gas_species=("Ga", "GaO"),
            composition_wt_pct={
                "SiO2": 50.34,
                "Al2O3": 15.4,
                "MgO": 10.8,
                "CaO": 23.46,
                "Ga2O3": 0.05,
            },
            allow_extrapolation=True,
        )

    assert "trace_gas_channel_refused" in str(refused.value)
    assert "GaO1.5-liquid-KEMS-1700K" in str(refused.value)
    assert "temperature_K" in str(refused.value)


@pytest.mark.parametrize(
    ("trace", "species", "row_id"),
    [
        ("P2O5", "P2", "suito_inoue_1984_eq14_no_mn"),
        ("S", "S2", "sulfide_capacity_eq_7"),
    ],
)
def test_unrepresentable_trace_law_refuses_only_its_gas_channel(
    trace: str,
    species: str,
    row_id: str,
    monkeypatch: pytest.MonkeyPatch,
    packaged_gas_pack,
) -> None:
    from types import SimpleNamespace

    import openimcc.model as model_module

    monkeypatch.setattr(
        model_module,
        "evaluate",
        lambda *_args, **_kwargs: SimpleNamespace(
            labels=SimpleNamespace(notices=())
        ),
    )
    composition = {
        "SiO2": 50.34,
        "Al2O3": 15.4,
        "MgO": 10.8,
        "CaO": 23.46,
        trace: 0.05,
    }
    with pytest.raises(ImccComponentOutsideDomainError) as refused:
        evaluate_gas(
            {},
            5.0e-324,
            1.0e-8,
            packaged_gas_pack,
            gas_species=(species,),
            composition_wt_pct=composition,
            allow_extrapolation=True,
        )

    assert "trace_gas_channel_refused" in str(refused.value)
    assert row_id in str(refused.value)
    assert "temperature_K" in str(refused.value)


def test_activity_ladder_uses_source_rows_helpers_and_the_shared_registration(
    packaged_gas_pack,
) -> None:
    assert trace_component_rule("MnO").full_component
    assert trace_component_rule("P2O5").activity_fragment == "P"
    assert trace_component_rule("NiO").activity_fragment == "Ni"
    assert trace_component_rule("CoO").activity_fragment == "Co"
    assert trace_component_rule("S").activity_fragment == "S"
    assert trace_component_rule("NiO").nu_reason
    assert trace_component_rule("Cr2O3").gamma_species == "CrO1.5"
    assert trace_component_rule("Cr2O3").gamma == 31.1
    assert trace_component_rule("P2O5").trace_class == "acidic"
    assert trace_component_rule("Cr2O3").trace_class == "amphoteric"
    assert trace_component_rule("NiO").trace_class == "basic_monoxide"
    assert trace_component_rule("CoO").trace_class == "basic_monoxide"
    assert trace_component_rule("ZnO").trace_class == "basic_monoxide"
    assert all(
        rule.trace_class_reason
        for rule in trace_components._TRACE_COMPONENT_RULES.values()
    )
    assert all(
        trace_component_rule(oxide).activity_model == "redox_associates"
        for oxide in ("SrO", "Li2O", "ZnO", "Yb2O3", "MnO", "Cr2O3")
    )
    moo3_rule = trace_component_rule("MoO3")
    assert moo3_rule is not None
    assert moo3_rule.activity_model == "review_gamma"
    assert moo3_rule.review_gamma_element == "Mo"
    assert any(item.oxide == "MoO3" for item in supported_trace_oxides())

    p2o5_composition = {
        "CaO": 40.0,
        "MgO": 8.0,
        "FeO": 12.0,
        "SiO2": 39.0,
        "P2O5": 1.0,
    }
    p2o5_result = evaluate(
        p2o5_composition, 1873.0, basis_type="wt", allow_extrapolation=True
    )
    assert p2o5_result.labels.notices
    p2o5_notice = next(
        item for item in p2o5_result.labels.notices
        if isinstance(item, TraceComponentsOutsideMeltModelNotice)
    )
    assert p2o5_notice.activities["P2O5"].code == "depends_helper_activity"
    assert "MgO_saturation" in (
        p2o5_notice.activities["P2O5"].dependency_axes
    )

    co_composition = _weight_composition({"CoO": 0.002})
    co_result = evaluate(
        co_composition, 1800.0, basis_type="wt", allow_extrapolation=True
    )
    co_notice = next(
        item for item in co_result.labels.notices
        if isinstance(item, TraceComponentsOutsideMeltModelNotice)
    )
    assert co_notice.activities["CoO"].code == "out_of_domain_helper_activity"

    sulfur_moles = {
        "SiO2": 0.50,
        "TiO2": 0.015,
        "Al2O3": 0.15,
        "FeO": 0.10,
        "MgO": 0.08,
        "CaO": 0.11,
        "Na2O": 0.03,
        "K2O": 0.001,
        "S": 0.001,
    }
    sulfur = trace_components.resolve_trace_activity(
        "S", 0.001, sulfur_moles, 1673.0, fO2_bar=10.0**-8.97
    )
    assert sulfur.code == "out_of_domain_helper_activity"
    assert "capacity-composition-outside-calibration" in (sulfur.reference or "")
    assert "total_pressure_bar" in sulfur.dependency_axes
    assert "total_pressure_bar" in str(TraceActivityFlag(sulfur))

    supplied_pressure, _, _ = trace_components._resolve_trace_activity(
        "S",
        0.001,
        {"SiO2": 1.0, "S": 0.001},
        1673.0,
        fO2_bar=1.0e-8,
        total_pressure_bar=1.0,
    )
    assert "total_pressure_bar" not in supplied_pressure.dependency_axes

    sulfur_composition = _weight_composition({"S": 0.001})
    sulfur_melt = evaluate(
        sulfur_composition, 1673.0, basis_type="wt", allow_extrapolation=True
    )
    sulfur_gas = evaluate_gas(
        _major_activities(sulfur_melt),
        1673.0,
        1.0e-8,
        packaged_gas_pack,
        gas_species=("S",),
        composition_wt_pct=sulfur_composition,
    )
    sulfur_flag = sulfur_gas.domain_flags["S"]
    assert isinstance(sulfur_flag, TraceActivityFlag)
    assert "total_pressure_bar" in sulfur_flag.dependency_axes

    cr_composition = _weight_composition({"Cr2O3": 0.001})
    cr_result = evaluate(
        cr_composition, 1800.0, basis_type="wt", allow_extrapolation=True
    )
    cr_notice = next(
        item for item in cr_result.labels.notices
        if isinstance(item, TraceComponentsOutsideMeltModelNotice)
    )
    cr_estimate = cr_notice.activities["Cr2O3"]
    assert cr_estimate.code == "out_of_domain_associate_activity"
    assert cr_estimate.reference is not None
    assert "required redox rows extrapolated" in cr_estimate.reference
    assert "temperature_K" in cr_estimate.violated_axes
    cr_activity = cr_result.trace_activities()["Cr2O3"]
    assert cr_activity.activity == cr_estimate.activity
    assert cr_activity.code == cr_estimate.code

    ce_rule = trace_component_rule("Ce2O3")
    assert ce_rule is not None
    ce_sio2_moles = (
        MOLAR_MASS_G_MOL["CaO"] + 0.72 * MOLAR_MASS_G_MOL["Al2O3"]
    ) / (
        MOLAR_MASS_G_MOL["SiO2"] * ((17508.97 + 14.93) / 3.46)
    )
    ce_composition = {
        "CaO": 1.0,
        "MgO": 1.0,
        "Al2O3": 1.0,
        "FeO": 1.0,
        "SiO2": ce_sio2_moles,
        "Ce2O3": 1.0e-8,
    }
    ce_masses = {**MOLAR_MASS_G_MOL, "Ce2O3": ce_rule.molar_mass_g_mol}
    ce_composition_wt = {
        oxide: amount * ce_masses[oxide]
        for oxide, amount in ce_composition.items()
    }
    ce_result = evaluate(ce_composition_wt, 1873.0, basis_type="wt")
    ce_activity = ce_result.trace_activities()["Ce2O3"]
    assert ce_activity.activity is None
    assert ce_activity.code == "trace_refused_review_gamma_activity"

    joint_trace_result = evaluate(
        {
            **_MARE_BASALT_WT,
            "MnO": 0.206,
            "P2O5": 0.1,
            "Cr2O3": 0.1,
        },
        2100.0,
        basis_type="wt",
        allow_extrapolation=True,
    )
    assert "MnO" in joint_trace_result.parent_oxides
    assert "P2O5" not in joint_trace_result.parent_oxides
    assert "Cr2O3" not in joint_trace_result.parent_oxides
    flags_by_oxide = {
        item.oxide: item
        for item in joint_trace_result.labels.notices
        if isinstance(item, TraceActivityFlag)
    }
    trace_activities = joint_trace_result.trace_activities()
    for oxide in ("P2O5", "Cr2O3"):
        assert isinstance(trace_activities[oxide], TraceActivityEstimate)
        assert trace_activities[oxide] is flags_by_oxide[oxide].estimates[0]
        assert trace_activities[oxide].activity == flags_by_oxide[oxide].activity
    with pytest.raises(TypeError):
        trace_activities["P2O5"] = trace_activities["Cr2O3"]

    mn_composition = _weight_composition({"MnO": 0.002})
    mn_result = evaluate(
        mn_composition, 1800.0, basis_type="wt", allow_extrapolation=True
    )
    assert "MnO" in mn_result.parent_oxides
    assert not any(
        isinstance(item, TraceActivityFlag) and item.oxide == "MnO"
        for item in mn_result.labels.notices
    )
    assert not any(
        isinstance(item, TraceComponentsOutsideMeltModelNotice)
        for item in mn_result.labels.notices
    )
    assert math.isfinite(mn_result.activity("MnO"))

    full_mn_result = evaluate(
        _weight_composition({"MnO": 0.05}),
        1800.0,
        basis_type="wt",
        allow_extrapolation=True,
    )
    assert math.isfinite(full_mn_result.activity("MnO"))
    assert not any(
        isinstance(item, TraceComponentsOutsideMeltModelNotice)
        for item in full_mn_result.labels.notices
    )


def test_trace_helper_validity_uses_declared_checks_and_keeps_activity_values() -> None:
    ni_outside_window = trace_components.resolve_trace_activity(
        "NiO",
        0.001,
        {"SiO2": 1.0, "FeO": 0.4, "NiO": 0.001},
        1700.0,
        fO2_bar=1.0e-8,
    )
    assert ni_outside_window.code == "out_of_domain_helper_activity"
    assert ni_outside_window.activity > 0.0
    assert ni_outside_window.limitations
    assert "FeO_wt_pct_outside_0_to_12" in (ni_outside_window.reference or "")
    assert "MgO_wt_pct_outside_4_to_30" in (ni_outside_window.reference or "")

    ni_unknown_oxygen = trace_components.resolve_trace_activity(
        "NiO",
        0.001,
        {
            "SiO2": 50.0,
            "MgO": 10.0,
            "FeO": 6.0,
            "CaO": 20.0,
            "Al2O3": 14.0,
            "NiO": 0.05,
        },
        1700.0,
    )
    assert ni_unknown_oxygen.code == "depends_helper_activity"
    assert ni_unknown_oxygen.activity > 0.0
    assert ni_unknown_oxygen.violated_axes == ()
    assert "oxygen_fugacity_relative_to_IW_not_supplied" in (
        ni_unknown_oxygen.dependency_axes
    )
    assert "oxygen_fugacity_relative_to_IW_not_supplied" in (
        ni_unknown_oxygen.reference or ""
    )
    assert "dilute_NiO_loading_limit" not in ni_unknown_oxygen.dependency_axes
    assert ni_unknown_oxygen.limitations

    co_unknown_oxygen = trace_components.resolve_trace_activity(
        "CoO",
        0.001,
        {
            "SiO2": 50.0,
            "MgO": 10.0,
            "FeO": 6.0,
            "CaO": 20.0,
            "Al2O3": 14.0,
            "CoO": 0.05,
        },
        1700.0,
    )
    assert co_unknown_oxygen.code == "depends_helper_activity"
    assert co_unknown_oxygen.violated_axes == ()
    assert "oxygen_fugacity_domain_unchecked" in (
        co_unknown_oxygen.dependency_axes
    )
    assert "oxygen_fugacity_domain_unchecked" in (
        co_unknown_oxygen.reference or ""
    )

    co_outside_window = trace_components.resolve_trace_activity(
        "CoO",
        0.001,
        _mole_fraction_composition(_MARE_BASALT_WT, "CoO", 0.001),
        1700.0,
        fO2_bar=1.0e-8,
    )
    assert co_outside_window.code == "out_of_domain_helper_activity"
    assert co_outside_window.activity > 0.0
    assert "FeO_outside_0_to_12_wt_pct" in (co_outside_window.reference or "")
    assert co_outside_window.limitations


@pytest.mark.parametrize("trace_oxide", ("NiO", "CoO"))
def test_passing_ni_co_helper_inputs_remain_source_grounded(
    trace_oxide: str,
) -> None:
    weights = {
        "SiO2": 50.0,
        "MgO": 10.0,
        "FeO": 6.0,
        "CaO": 20.0,
        "Al2O3": 14.0,
        trace_oxide: 0.001,
    }
    composition = {
        oxide: amount / MOLAR_MASS_G_MOL[oxide]
        for oxide, amount in weights.items()
    }
    x_trace = composition[trace_oxide] / math.fsum(composition.values())
    fO2_bar = 10.0 ** (_iw_log10_fO2_bar(1700.0) - 1.0)

    estimate = trace_components.resolve_trace_activity(
        trace_oxide,
        x_trace,
        composition,
        1700.0,
        fO2_bar=fO2_bar,
    )

    assert estimate.code == "source_grounded_helper_activity"
    assert estimate.violated_axes == ()
    assert estimate.dependency_axes == ()
    assert estimate.limitations


@pytest.mark.parametrize(
    "trace_oxide,gas_species", (("NiO", "Ni"), ("CoO", "Co"))
)
def test_passing_ni_co_gas_channels_add_no_validity_flags(
    trace_oxide: str,
    gas_species: str,
    packaged_gas_pack,
) -> None:
    composition = {
        "SiO2": 50.0,
        "MgO": 10.0,
        "FeO": 6.0,
        "CaO": 20.0,
        "Al2O3": 14.0,
        trace_oxide: 0.001,
    }
    melt = evaluate(
        composition, 1700.0, basis_type="wt", allow_extrapolation=True
    )
    fO2_bar = 10.0 ** (_iw_log10_fO2_bar(1700.0) - 1.0)

    gas = evaluate_gas(
        _major_activities(melt),
        1700.0,
        fO2_bar,
        packaged_gas_pack,
        gas_species=(gas_species,),
        composition_wt_pct=composition,
    )
    flag = gas.domain_flags[gas_species]

    assert isinstance(flag, TraceActivityFlag)
    assert flag.code == "source_grounded_helper_activity"
    assert flag.estimates[0].violated_axes == ()
    assert flag.dependency_axes == ()


def test_unmodelled_helper_components_are_dependencies() -> None:
    composition = {
        "SiO2": 50.0,
        "MgO": 10.0,
        "FeO": 6.0,
        "CaO": 20.0,
        "Al2O3": 14.0,
        "Fe2O3": 0.001,
        "NiO": 0.001,
    }
    x_trace = composition["NiO"] / math.fsum(composition.values())

    estimate = trace_components.resolve_trace_activity(
        "NiO", x_trace, composition, 1700.0, fO2_bar=1.0e-8
    )

    assert estimate.code == "depends_helper_activity"
    assert "unsupported_component_Fe2O3" in estimate.dependency_axes
    assert "unsupported_component_Fe2O3" not in estimate.violated_axes


def test_redox_validity_uses_typed_axes_not_row_flag_text() -> None:
    composition = {
        "SiO2": 0.5,
        "MgO": 0.15,
        "CaO": 0.2,
        "Al2O3": 0.15,
        "Cr2O3": 0.0001,
    }
    estimate = trace_components.resolve_trace_activity(
        "Cr2O3", 0.0001, composition, 1700.0, fO2_bar=1.0e-8
    )

    assert estimate.code == "depends_associate_activity"
    assert estimate.violated_axes == ()
    assert "redox_association" in estimate.dependency_axes
    assert "redox_chemistry" not in estimate.dependency_axes
    assert "oxygen_buffering" not in estimate.dependency_axes
    assert "redox_calibration" not in estimate.violated_axes
    assert "CrO(l)-effective:" in (estimate.reference or "")

    oxidizing = trace_components.resolve_trace_activity(
        "Cr2O3", 0.0001, composition, 1700.0, fO2_bar=1.0e-4
    )
    assert "redox_chemistry" in oxidizing.dependency_axes


def test_fixed_valence_trace_ignores_global_redox_coverage_notes() -> None:
    composition = {
        "SiO2": 0.5,
        "MgO": 0.15,
        "CaO": 0.2,
        "Al2O3": 0.15,
        "Li2O": 0.0001,
    }
    estimate = trace_components.resolve_trace_activity(
        "Li2O", 0.0001, composition, 1700.0, fO2_bar=1.0e-8
    )

    assert "redox_chemistry" not in estimate.dependency_axes
    assert "oxygen_buffering" not in estimate.dependency_axes


def test_zinc_activity_notice_is_a_failed_calibration_axis(
    packaged_gas_pack,
) -> None:
    composition = _weight_composition({"ZnO": 0.001})
    melt = evaluate(
        composition, 1700.0, basis_type="wt", allow_extrapolation=True
    )
    notice = next(
        item for item in melt.labels.notices
        if isinstance(item, TraceComponentsOutsideMeltModelNotice)
    )
    estimate = notice.activities["ZnO"]
    assert estimate.code == "out_of_domain_associate_activity"
    assert "ZnO_activity_calibration" in estimate.violated_axes

    gas = evaluate_gas(
        _major_activities(melt),
        1700.0,
        1.0e-8,
        packaged_gas_pack,
        gas_species=("ZnO",),
        composition_wt_pct=composition,
    )
    flag = gas.domain_flags["ZnO"]
    assert isinstance(flag, TraceActivityFlag)
    assert flag.code == estimate.code
    assert "ZnO_activity_calibration" in flag.estimates[0].violated_axes


def test_sulfur_trace_uses_only_returned_capacity_and_pressure_checks() -> None:
    sulfur_moles = {
        oxide: amount / MOLAR_MASS_G_MOL[oxide]
        for oxide, amount in _MARE_BASALT_WT.items()
    }
    total_major_mass = math.fsum(_MARE_BASALT_WT.values())
    sulfur_moles["S"] = 1.0e-5 * total_major_mass / MOLAR_MASS_G_MOL["S"]
    x_s = sulfur_moles["S"] / math.fsum(sulfur_moles.values())

    assumed_pressure, _, _ = trace_components._resolve_trace_activity(
        "S", x_s, sulfur_moles, 1700.0, fO2_bar=1.0e-8
    )
    assert assumed_pressure.code == "depends_helper_activity"
    assert assumed_pressure.violated_axes == ()
    assert assumed_pressure.dependency_axes == ("total_pressure_bar",)
    assert "om2002-temperature-outside-calibration" not in (
        assumed_pressure.reference or ""
    )

    low_pressure, _, _ = trace_components._resolve_trace_activity(
        "S", x_s, sulfur_moles, 1673.0, fO2_bar=1.0e-8,
        total_pressure_bar=0.1,
    )
    assert low_pressure.code == "out_of_domain_helper_activity"
    assert low_pressure.violated_axes == ("total_pressure_bar",)


def test_sulfur_pressure_notice_names_each_calibrated_model() -> None:
    composition = {**_MARE_BASALT_WT, "S": 0.001}
    composition_moles, fractions = trace_fractions_from_wt(composition)
    estimate, _, _ = trace_components._resolve_trace_activity(
        "S",
        fractions["S"],
        composition_moles,
        1673.0,
        fO2_bar=1.0e-8,
        total_pressure_bar=0.1,
    )

    assert estimate.code == "out_of_domain_helper_activity"
    assert estimate.violated_axes == ("total_pressure_bar",)
    assert estimate.reference is not None
    assert estimate.reference.count("pressure-below-calibration") == 2
    assert estimate.reference.count("Boulliung-Wood-2023-capacities") == 1
    assert estimate.reference.count("Smythe-2017-SCSS") == 1


def test_manganese_is_full_component_without_redox_bookkeeping() -> None:
    result = evaluate(
        _weight_composition({"MnO": 0.002}),
        1800.0,
        basis_type="wt",
        allow_extrapolation=True,
    )
    assert "MnO" in result.parent_oxides
    assert {"MnSiO3(l)", "Mn2SiO4(l)", "MnAl2O4(l)"} <= set(
        result.species_names
    )
    assert not any(flag.startswith("RED:") for flag in result.labels.flags)
    assert not any(
        isinstance(item, TraceActivityFlag) and item.oxide == "MnO"
        for item in result.labels.notices
    )


def test_mixed_redox_traces_render_shared_messages_once() -> None:
    composition = {**_MARE_BASALT_WT, "MnO": 0.001, "Li2O": 0.001}
    result = evaluate(
        composition, 1800.0, basis_type="wt", allow_extrapolation=True
    )
    rendered = "\n".join((
        *result.labels.flags,
        *(str(item) for item in result.labels.notices),
    ))
    assert rendered.count("RED: unmodelled_redox_couples") == 1

    composition_moles, fractions = trace_fractions_from_wt(composition)
    flags = {
        item.oxide: item
        for item in result.labels.notices
        if isinstance(item, TraceActivityFlag)
    }
    assert "MnO" in result.parent_oxides
    assert "MnO" not in flags
    for oxide in ("Li2O",):
        expected = trace_components.resolve_trace_activity(
            oxide,
            fractions[oxide],
            composition_moles,
            1800.0,
        )
        estimate = flags[oxide].estimates[0]
        assert estimate.code == expected.code
        assert estimate.violated_axes == expected.violated_axes
        assert estimate.dependency_axes == expected.dependency_axes


@pytest.mark.parametrize("trace_oxide", ("NiO", "CoO"))
@pytest.mark.parametrize("fO2_bar", (0.0, -1.0, math.nan))
def test_nico_invalid_oxygen_keeps_helper_activity_and_fails_the_oxygen_check(
    trace_oxide: str, fO2_bar: float
) -> None:
    composition = {
        "SiO2": 50.0,
        "MgO": 10.0,
        "FeO": 6.0,
        "CaO": 20.0,
        "Al2O3": 14.0,
        trace_oxide: 0.05,
    }
    x_trace = composition[trace_oxide] / math.fsum(composition.values())

    if trace_oxide == "NiO":
        from openimcc.ni import evaluate_ni_activity

        expected = evaluate_ni_activity(composition, 1700.0).activity
    else:
        from openimcc.co_activity import evaluate_co_activity

        expected = evaluate_co_activity(composition, 1700.0).activity

    estimate = trace_components.resolve_trace_activity(
        trace_oxide,
        x_trace,
        composition,
        1700.0,
        fO2_bar=fO2_bar,
    )
    flag = trace_components.TraceActivityFlag(estimate)

    assert estimate.activity.hex() == expected.hex()
    assert flag.code == "out_of_domain_helper_activity"
    assert "oxygen_fugacity_not_finite_or_positive" in str(flag)
    assert "ideal_mixing_assumption" not in str(flag)


def test_trace_associate_validity_carries_redox_extrapolations() -> None:
    estimate = trace_components.resolve_trace_activity(
        "ZnO",
        0.001,
        {"SiO2": 0.5, "MgO": 0.2, "FeO": 0.1, "CaO": 0.2, "ZnO": 0.001},
        1400.0,
        fO2_bar=1.0e-8,
    )
    assert estimate.code == "out_of_domain_associate_activity"
    assert estimate.activity > 0.0
    assert "redox_row_temperature_extrapolation" in (estimate.reference or "")
    assert "FeO(l)" in (estimate.reference or "")
    assert "temperature_K" in estimate.violated_axes
    assert "redox_calibration" not in estimate.violated_axes
    assert "ZnO_activity_calibration" in estimate.violated_axes
    assert "redox_chemistry" not in estimate.dependency_axes


def test_sulfur_trace_validity_carries_capacity_calibration_notices() -> None:
    estimate = trace_components.resolve_trace_activity(
        "S", 0.001, {"SiO2": 1.0, "S": 0.001}, 1673.0, fO2_bar=1.0e-8
    )
    assert estimate.code == "out_of_domain_helper_activity"
    assert estimate.activity > 0.0
    assert "capacity-composition-outside-calibration" in (
        estimate.reference or ""
    )


def test_p2o5_trace_validity_includes_ferric_chemistry_notice() -> None:
    estimate = trace_components.resolve_trace_activity(
        "P2O5",
        0.002,
        {
            "CaO": 0.4,
            "MgO": 0.08,
            "FeO": 0.12,
            "SiO2": 0.39,
            "P2O5": 0.002,
            "Fe2O3": 0.0001,
        },
        1873.0,
    )
    assert estimate.code == "out_of_domain_helper_activity"
    assert estimate.activity > 0.0
    assert "p2o5_feiii_phosphate_chemistry_not_modelled" in (
        estimate.reference or ""
    )
    assert estimate.limitations
    assert "MgO_saturation" in estimate.dependency_axes


def test_redox_composition_screen_flags_any_unsupported_cation() -> None:
    inventory = {
        "Si": 0.3,
        "Mg": 0.1,
        "Fe": 0.1,
        "Ca": 0.15,
        "Al": 0.1,
        "Ti": 0.01,
        "Na": 0.01,
        "K": 0.001,
        "Cr": 0.002,
        "O": 1.05,
        "P": 0.003,
    }
    result = evaluate_redox(
        inventory,
        1673.0,
        "imposed",
        lambda_imposed=math.log(1.0e-8),
    )
    assert any("unsupported major cations=P" in flag for flag in result.flags)


@pytest.mark.parametrize("oxide", ["SrO", "Li2O", "ZnO", "Yb2O3"])
def test_redox_v5_associates_are_first_rung_trace_sources(oxide: str) -> None:
    result = evaluate(
        _weight_composition({oxide: 0.001}),
        1800.0,
        basis_type="wt",
        allow_extrapolation=True,
    )
    notice = next(
        item for item in result.labels.notices
        if isinstance(item, TraceComponentsOutsideMeltModelNotice)
    )
    estimate = notice.activities[oxide]
    expected_code = (
        "out_of_domain_associate_activity"
        if oxide == "ZnO"
        else "depends_associate_activity"
        if oxide == "Yb2O3"
        else "source_grounded_associate_activity"
    )
    assert estimate.code == expected_code
    assert math.isfinite(estimate.activity) and estimate.activity > 0.0
    if oxide == "ZnO":
        assert "ZnO_activity_calibration" in estimate.violated_axes


@pytest.mark.parametrize("x_mno", [0.005, 0.05])
def test_evaluate_uses_full_manganese_associate_solve_for_basalt(
    x_mno: float,
) -> None:
    mare_wt = {
        "SiO2": 46.234,
        "TiO2": 1.587,
        "Al2O3": 14.214,
        "FeO": 16.353,
        "MgO": 9.463,
        "CaO": 11.637,
        "Na2O": 0.182,
    }
    base_moles = {
        oxide: amount / MOLAR_MASS_G_MOL[oxide]
        for oxide, amount in mare_wt.items()
    }
    base_total = math.fsum(base_moles.values())
    composition = {
        oxide: amount / base_total * (1.0 - x_mno)
        for oxide, amount in base_moles.items()
    }
    composition["MnO"] = x_mno

    result = evaluate(
        composition,
        1800.0,
        basis_type="mol",
        allow_extrapolation=True,
    )
    from openimcc.redox_pack import load_redox_pack

    mno_lineage = load_redox_pack().content["thermo_species"]["MnO(l)"][
        "provenance"
    ]["source"]
    assert result.labels.coverage["MnO"] == mno_lineage

    assert "MnO" in result.parent_oxides
    assert not any(flag.startswith("RED:") for flag in result.labels.flags)
    assert not any(
        isinstance(item, TraceActivityFlag) and item.oxide == "MnO"
        for item in result.labels.notices
    )
    assert not math.isclose(result.activity("MnO"), x_mno, rel_tol=1e-3)
    assert not math.isclose(
        result.activity("MnO"), 1.9 * x_mno, rel_tol=1e-3
    )

    assert math.fsum(result.parent_mol) == pytest.approx(result.basis)
    assert math.fsum(result.parent_x) == pytest.approx(1.0)
    assert math.fsum(result.species_x) == pytest.approx(1.0)
    for parent, free_fraction in zip(
        result.parent_oxides, result.parent_x_star, strict=True
    ):
        species_index = result.species_names.index(parent)
        assert free_fraction == pytest.approx(result.species_x[species_index])

    mn_species = {
        "MnO", "MnSiO3(l)", "Mn2SiO4(l)", "MnAl2O4(l)"
    }
    assert mn_species <= set(result.species_names)
    assert all(
        result.species_x[result.species_names.index(name)] > 0.0
        for name in mn_species
    )

    melt_pack = load_datapack().kernel_datapack
    complex_names = result.species_names[len(result.parent_oxides):]
    stoichiometry = {
        name: math.fsum(float(value) for value in melt_pack.nu[:, index])
        for index, name in enumerate(melt_pack.reactions)
    }
    for row in load_redox_pack().rows:
        if row.get("role") == "complex":
            stoichiometry[str(row["complex"])] = math.fsum(
                float(value) for value in row["nu"].values()
            )
    recomputed_D = 1.0 + math.fsum(
        (stoichiometry[name] - 1.0) * float(result.complex_x[index])
        for index, name in enumerate(complex_names)
    )
    assert result.D == pytest.approx(recomputed_D, abs=1e-12)
    assert result.convergence.iterations > 0
    assert not any(
        isinstance(item, (TraceComponentsOutsideMeltModelNotice, TraceMajorBiasFlag))
        for item in result.labels.notices
    )


@pytest.mark.parametrize("temperature_K", (1800.0, 2100.0))
def test_manganese_full_component_limit_is_continuous(
    temperature_K: float,
) -> None:
    composition_wt = {**_MARE_BASALT_WT, "K2O": 0.05}
    without_mn = evaluate(
        composition_wt,
        temperature_K,
        basis_type="wt",
        allow_extrapolation=True,
    )
    loadings_wt_pct = (1.0e-5, 1.0e-6, 1.0e-7)
    mole_fractions: list[float] = []
    shifts_by_parent = {parent: [] for parent in without_mn.parent_oxides}
    for loading_wt_pct in loadings_wt_pct:
        composition = {**composition_wt, "MnO": loading_wt_pct}
        with_mn = evaluate(
            composition,
            temperature_K,
            basis_type="wt",
            allow_extrapolation=True,
        )
        _, trace_fractions = trace_fractions_from_wt(composition)
        mole_fractions.append(trace_fractions["MnO"])
        for parent in without_mn.parent_oxides:
            shifts_by_parent[parent].append(math.log10(
                with_mn.activity(parent) / without_mn.activity(parent)
            ))

    for parent, shifts in shifts_by_parent.items():
        slopes = [
            shift / mole_fraction
            for shift, mole_fraction in zip(shifts, mole_fractions, strict=True)
        ]
        ratios = [
            abs(shifts[index] / shifts[index + 1])
            for index in range(len(shifts) - 1)
        ]
        x_mean = math.fsum(mole_fractions) / len(mole_fractions)
        y_mean = math.fsum(shifts) / len(shifts)
        fitted_slope = math.fsum(
            (x_value - x_mean) * (shift - y_mean)
            for x_value, shift in zip(mole_fractions, shifts, strict=True)
        ) / math.fsum((x_value - x_mean) ** 2 for x_value in mole_fractions)
        intercept = y_mean - fitted_slope * x_mean

        assert all(
            math.isclose(ratio, 10.0, rel_tol=0.05)
            for ratio in ratios
        ), (parent, temperature_K, ratios, slopes)
        assert abs(intercept) <= 1.0e-11, (
            parent,
            temperature_K,
            intercept,
            slopes,
        )
        assert abs(shifts[-1]) <= 1.0e-7, (
            parent,
            temperature_K,
            shifts[-1],
            slopes,
        )


def test_manganese_association_competes_with_silica_and_raises_sodium() -> None:
    without_mn = evaluate(
        _MARE_BASALT_WT,
        2100.0,
        basis_type="wt",
        allow_extrapolation=True,
    )
    with_mn = evaluate(
        {**_MARE_BASALT_WT, "MnO": 0.206},
        2100.0,
        basis_type="wt",
        allow_extrapolation=True,
    )

    na2o_shift_dex = math.log10(
        with_mn.activity("Na2O") / without_mn.activity("Na2O")
    )
    sio2_shift_dex = math.log10(
        with_mn.activity("SiO2") / without_mn.activity("SiO2")
    )
    assert na2o_shift_dex > 0.0
    assert sio2_shift_dex < 0.0


def test_fe_free_manganese_associate_path() -> None:
    composition = {
        "SiO2": 0.45,
        "MgO": 0.2,
        "CaO": 0.15,
        "Al2O3": 0.1,
        "MnO": 0.1,
    }
    result = evaluate(
        composition,
        1800.0,
        basis_type="mol",
        allow_extrapolation=True,
    )
    assert "MnO" in result.parent_oxides
    assert {"MnSiO3(l)", "Mn2SiO4(l)", "MnAl2O4(l)"} <= set(
        result.species_names
    )
    assert not any(
        isinstance(item, TraceActivityFlag) and item.oxide == "MnO"
        for item in result.labels.notices
    )


def test_manganese_full_component_does_not_use_trace_helper_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse_associates(*args, **kwargs):
        raise ImccComponentOutsideDomainError("associate solve refused")

    monkeypatch.setattr(
        trace_components, "_redox_trace_result", refuse_associates
    )
    result = evaluate(
        _weight_composition({"MnO": 0.005}),
        1800.0,
        basis_type="wt",
        allow_extrapolation=True,
    )
    assert "MnO" in result.parent_oxides
    assert "MnSiO3(l)" in result.species_names
    assert not any(
        isinstance(item, TraceActivityFlag) and item.oxide == "MnO"
        for item in result.labels.notices
    )


def test_redox_sweeps_recompute_stored_class_envelopes() -> None:
    def redox_result(
        oxide_moles: dict[str, float], temperature_K: float, log_fO2: float
    ):
        return evaluate_redox(
            _redox_inventory(oxide_moles),
            temperature_K,
            "imposed",
            lambda_imposed=log_fO2 * math.log(10.0),
        )

    stored = {
        trace_class: (calibrant, melt, temperature, log_fO2, x_trace,
                      major, shift, contribution)
        for trace_class, calibrant, melt, temperature, log_fO2, x_trace,
        major, shift, contribution in CALIBRATION_ROWS
    }
    by_class: dict[str, list[tuple[float, float, str, float]]] = {
        "basic_monoxide": [],
        "amphoteric": [],
    }
    recomputed: dict[
        tuple[str, str, str, float, float, float],
        tuple[str, float, float],
    ] = {}
    for trace_class, calibrant, melts in (
        ("basic_monoxide", "MnO", (
            ("mare basalt", _MARE_BASALT_WT),
            ("alkali-rich", _ALKALI_RICH_WT),
        )),
        ("amphoteric", "Cr2O3", (
            ("mare basalt", _MARE_BASALT_WT),
            ("alkali-rich", _ALKALI_RICH_WT),
        )),
    ):
        nu = trace_component_rule(calibrant).nu
        for melt_name, weights in melts:
            base_moles = {
                oxide: amount / MOLAR_MASS_G_MOL[oxide]
                for oxide, amount in weights.items()
            }
            total = math.fsum(base_moles.values())
            base_fractions = {
                oxide: amount / total for oxide, amount in base_moles.items()
            }
            for temperature_K in ENVELOPE_T_GRID_K:
                for delta_iw in ENVELOPE_DELTA_IW_GRID:
                    log_fO2 = _iw_log10_fO2_bar(temperature_K) + delta_iw
                    baseline = redox_result(
                        base_fractions, temperature_K, log_fO2
                    )
                    winners: list[tuple[float, str, float, float]] = []
                    for x_trace in (0.001, 0.01):
                        with_trace = {
                            oxide: amount * (1.0 - x_trace)
                            for oxide, amount in base_fractions.items()
                        }
                        with_trace[calibrant] = x_trace
                        result = redox_result(
                            with_trace, temperature_K, log_fO2
                        )
                        for major, activity in result.parent_activity.items():
                            if major == calibrant or major not in baseline.parent_activity:
                                continue
                            assert activity > 0.0 and baseline.parent_activity[major] > 0.0
                            shift = abs(
                                math.log10(activity)
                                - math.log10(baseline.parent_activity[major])
                            )
                            contribution = shift / ((1.0 + nu) * x_trace)
                            winners.append((contribution, major, shift, x_trace))
                    k, major, shift, x_trace = max(winners)
                    key = (
                        trace_class, calibrant, melt_name, temperature_K,
                        log_fO2, x_trace,
                    )
                    recomputed[key] = (major, shift, k)
                    by_class[trace_class].append((k, shift, melt_name, x_trace))

    winning_rows = {}
    for trace_class in by_class:
        class_rows = [
            (value[2], key[5], key, value)
            for key, value in recomputed.items()
            if key[0] == trace_class
        ]
        contribution, _x, key, (major, shift, _contribution) = max(class_rows)
        winning_rows[trace_class] = (
            key[1], key[2], key[3], key[4], key[5], major, shift, contribution
        )
    assert winning_rows.keys() == stored.keys()
    for trace_class, actual in winning_rows.items():
        expected = stored[trace_class]
        assert actual[:6] == expected[:6]
        assert actual[6] == pytest.approx(expected[6], rel=1e-7, abs=1e-10)
        assert actual[7] == pytest.approx(expected[7], rel=1e-7, abs=1e-10)

    k_basic = max(item[0] for item in by_class["basic_monoxide"])
    k_amphoteric = max(item[0] for item in by_class["amphoteric"])
    assert k_basic == pytest.approx(BASIC_MONOXIDE_ENVELOPE_K, rel=1e-12)
    assert k_amphoteric == pytest.approx(AMPHOTERIC_ENVELOPE_K, rel=1e-12)
    assert all(math.isfinite(item[0]) for item in by_class["amphoteric"])
    assert k_amphoteric > 1.8

    def point_max_k(
        trace_class: str, melt: str, temperature_K: float, delta_iw: float
    ) -> float:
        log_fO2 = _iw_log10_fO2_bar(temperature_K) + delta_iw
        return max(
            value[2]
            for key, value in recomputed.items()
            if key[0] == trace_class
            and key[2] == melt
            and key[3] == temperature_K
            and key[4] == log_fO2
        )

    mn_minus_2 = point_max_k("basic_monoxide", "mare basalt", 1400.0, -2.0)
    mn_minus_4 = point_max_k("basic_monoxide", "mare basalt", 1400.0, -4.0)
    mn_minus_6 = point_max_k("basic_monoxide", "mare basalt", 1400.0, -6.0)
    assert mn_minus_4 < mn_minus_2
    assert mn_minus_6 < mn_minus_4
    cr_minus_4 = point_max_k("amphoteric", "mare basalt", 1400.0, -4.0)
    cr_minus_6 = point_max_k("amphoteric", "mare basalt", 1400.0, -6.0)
    assert cr_minus_6 < cr_minus_4


@pytest.mark.parametrize(
    ("bad_silica", "trace_composition", "gas_species"),
    (
        (-50.0, {"Ga2O3": 0.001}, "Ga"),
        (math.nan, {"Ga2O3": 0.001}, "Ga"),
        (-50.0, {"Cl": 0.05}, "Na"),
    ),
)
def test_gas_refuses_declared_bad_major_before_trace_projection(
    bad_silica: float,
    trace_composition: dict[str, float],
    gas_species: str,
    packaged_gas_pack,
) -> None:
    with pytest.raises(ImccCompositionIncompleteError):
        evaluate_gas(
            {},
            1674.0,
            1.0e-5,
            packaged_gas_pack,
            gas_species=gas_species,
            composition_wt_pct={"SiO2": bad_silica, **trace_composition},
        )


def test_gas_an_di_flag_uses_majors_only_for_ga_trace_parent(
    packaged_gas_pack,
) -> None:
    composition = {
        "SiO2": 50.34,
        "Al2O3": 15.4,
        "MgO": 10.8,
        "CaO": 23.46,
        "Ga2O3": 2.5,
    }
    assert 50.34 - 50.34 / 102.5 * 100.0 == pytest.approx(1.2278, abs=1e-3)

    result = evaluate_gas(
        {},
        1674.0,
        1.0e-5,
        packaged_gas_pack,
        gas_species="Ga",
        composition_wt_pct=composition,
    )
    flag = result.domain_flags["Ga"]

    assert isinstance(flag, TraceActivityFlag)
    assert "melt_composition_differs_from_An_Di_target" not in str(flag)
    assert "an-di-target-basis=major-oxides-only-v2" in str(flag)


@pytest.mark.parametrize("oxide", ["Cl", "F", "S", "SO3"])
def test_halogen_and_sulfur_channels_refuse_only_above_point_one_mol_percent(
    oxide: str, packaged_gas_pack
) -> None:
    activity_map = {major: 0.1 for major in IMCC_PARENT_OXIDES}
    activity_map["S2"] = 0.1
    at_limit = _weight_composition({oxide: 0.001})
    allowed = evaluate_gas(
        activity_map,
        1800.0,
        1.0e-8,
        packaged_gas_pack,
        gas_species=("S",),
        composition_wt_pct=at_limit,
    )
    assert "S" in allowed

    above_limit = _weight_composition({oxide: 0.0011})
    with pytest.raises(ImccComponentOutsideDomainError, match="0.1 mol%"):
        evaluate_gas(
            activity_map,
            1800.0,
            1.0e-8,
            packaged_gas_pack,
            gas_species=("S",),
            composition_wt_pct=above_limit,
        )

    refused = evaluate_gas(
        activity_map,
        1800.0,
        1.0e-8,
        packaged_gas_pack,
        composition_wt_pct=above_limit,
    )
    assert isinstance(refused.omitted_channels["S"], TraceGasChannelRefusal)
    assert oxide in refused.omitted_channels["S"].triggers
    assert any(
        isinstance(item, TraceGasChannelRefusal)
        and item.species == "Na"
        and oxide in item.triggers
        for item in refused.notices
    )


def test_gas_trace_activity_flag_uses_the_same_ladder(packaged_gas_pack) -> None:
    composition = _weight_composition({"NiO": 0.002})
    melt_result = evaluate(
        composition, 1800.0, basis_type="wt", allow_extrapolation=True
    )
    gas = evaluate_gas(
        _major_activities(melt_result),
        1800.0,
        1.0e-8,
        packaged_gas_pack,
        composition_wt_pct=composition,
    )
    assert "Ni" in gas
    assert isinstance(gas.domain_flags["Ni"], TraceActivityFlag)
    assert gas.domain_flags["Ni"].code == "out_of_domain_helper_activity"
    assert gas.domain_flags["Ni"].estimates[0].oxide == "NiO"

    p2o5_composition = {
        "CaO": 40.0,
        "MgO": 8.0,
        "FeO": 12.0,
        "SiO2": 39.0,
        "P2O5": 1.0,
    }
    p2o5_result = evaluate(
        p2o5_composition, 1873.0, basis_type="wt", allow_extrapolation=True
    )
    p2o5_gas = evaluate_gas(
        _major_activities(p2o5_result),
        1873.0,
        1.0e-8,
        packaged_gas_pack,
        gas_species=("P2",),
        composition_wt_pct=p2o5_composition,
    )
    assert isinstance(p2o5_gas.domain_flags["P2"], TraceActivityFlag)
    assert p2o5_gas.domain_flags["P2"].code == "depends_helper_activity"
    assert "MgO_saturation" in p2o5_gas.domain_flags["P2"].dependency_axes
    assert "MgO_saturation" in str(p2o5_gas.domain_flags["P2"])


def test_gas_trace_parents_use_gas_standards_or_refuse(packaged_gas_pack) -> None:
    base = {
        "SiO2": 50.0,
        "MgO": 15.0,
        "FeO": 10.0,
        "CaO": 12.0,
        "Al2O3": 12.0,
    }
    for component, amount, species in (
        ("SO3", 0.1, "S"),
        ("Cl", 0.05, "Cl"),
        ("F", 0.05, "F"),
    ):
        result = evaluate_gas(
            {},
            1800.0,
            1.0e-8,
            packaged_gas_pack,
            composition_wt_pct={**base, component: amount},
        )
        refusal = result.omitted_channels[species]
        assert isinstance(refusal, TraceGasChannelRefusal)
        assert component in refusal.triggers
        assert "fugacity model is available" in refusal.reason
        parent_standard = {"SO3": "S2(g)", "Cl": "Cl2(g)", "F": "F2(g)"}
        assert parent_standard[component] in refusal.reason

    sulfur = evaluate_gas(
        {},
        1800.0,
        1.0e-8,
        packaged_gas_pack,
        gas_species=("S",),
        composition_wt_pct={**base, "S": 0.01},
    )
    sulfur_flag = sulfur.domain_flags["S"]
    assert isinstance(sulfur_flag, TraceActivityFlag)
    assert sulfur_flag.estimates[0].oxide == "S"
    assert sulfur_flag.code != "ideal_mixing_assumption"

    for element, oxide, species, cations_per_oxide in (
        ("Li", "Li2O", "Li", 2),
        ("Zn", "ZnO", "Zn", 1),
    ):
        element_mass = trace_component_rule(element).molar_mass_g_mol
        oxide_mass = trace_component_rule(oxide).molar_mass_g_mol
        element_amount = 0.05
        oxide_amount = 0.05
        converted_oxide, oxide_moles, converted_cations = element_to_parent_oxide(
            element, element_amount / element_mass
        )
        assert converted_oxide == oxide
        assert converted_cations == cations_per_oxide
        combined = evaluate_gas(
            {},
            1800.0,
            1.0e-8,
            packaged_gas_pack,
            composition_wt_pct={
                **base,
                element: element_amount,
                oxide: oxide_amount,
            },
        )
        oxide_equivalent = evaluate_gas(
            {},
            1800.0,
            1.0e-8,
            packaged_gas_pack,
            composition_wt_pct={
                **base,
                oxide: oxide_amount
                + element_amount * oxide_mass
                / (cations_per_oxide * element_mass),
            },
        )
        element_only = evaluate_gas(
            {},
            1800.0,
            1.0e-8,
            packaged_gas_pack,
            composition_wt_pct={**base, element: element_amount},
            gas_species=(species,),
        )
        baseline_pressure = {
            "Li": 1.42369622857686e-06,
            "Zn": 0.022852842899383257,
        }[element]
        assert element_only[species] == baseline_pressure
        assert oxide_moles * oxide_mass == pytest.approx(
            element_amount * oxide_mass / (cations_per_oxide * element_mass)
        )
        flag = combined.domain_flags[species]
        assert isinstance(flag, TraceActivityFlag)
        assert flag.estimates[0].oxide == oxide
        assert flag.code != "ideal_mixing_assumption"
        assert combined[species] == pytest.approx(
            oxide_equivalent[species], rel=1.0e-12
        )


@pytest.mark.parametrize(
    ("element", "amount_mol", "expected"),
    (
        ("Li", 6.0, ("Li2O", 3.0, 2)),
        ("Zn", 6.0, ("ZnO", 6.0, 1)),
        ("Yb", 6.0, ("Yb2O3", 3.0, 2)),
    ),
)
def test_element_to_parent_oxide_returns_formula_units_and_stoichiometry(
    element: str, amount_mol: float, expected: tuple[str, float, int]
) -> None:
    assert element_to_parent_oxide(element, amount_mol) == expected


def test_element_to_parent_oxide_rejects_unknown_element() -> None:
    with pytest.raises(ImccComponentOutsideDomainError, match="parent oxide"):
        element_to_parent_oxide("Xx", 1.0)


def test_sulfur_fragment_keeps_s2_when_so3_is_also_present(packaged_gas_pack) -> None:
    composition = {
        "SiO2": 50.0,
        "MgO": 15.0,
        "FeO": 10.0,
        "CaO": 12.0,
        "Al2O3": 12.0,
        "S": 0.01,
        "SO3": 0.1,
    }
    result = evaluate_gas(
        {},
        1800.0,
        1.0e-8,
        packaged_gas_pack,
        composition_wt_pct=composition,
    )

    assert set(result.omitted_channels) == {
        "Al",
        "Al2",
        "Al2O",
        "Al2O2",
        "AlO",
        "AlO2",
        "AlSiO",
        "Ca",
        "CaO",
        "Fe",
        "FeO",
        "Mg",
        "MgO",
    }
    assert "S" in result
    assert result["S"] > 0.0
    sulfur_flag = result.domain_flags["S"]
    assert isinstance(sulfur_flag, TraceActivityFlag)
    assert any(estimate.oxide == "S" for estimate in sulfur_flag.estimates)


def test_cli_labels_sulfur_activity_gas_standard(capsys) -> None:
    from openimcc.cli import main

    assert main([
        "solve",
        "--temperature", "1800",
        "--basis-type", "wt",
        "--oxide", "SiO2=50",
        "--oxide", "MgO=15",
        "--oxide", "FeO=10",
        "--oxide", "CaO=12",
        "--oxide", "Al2O3=12",
        "--oxide", "S=0.01",
    ]) == 0
    output = capsys.readouterr().out
    assert "S2 gas-standard activity=f(S2)/p° on JANAF 1-bar S2(g)=" in output


def test_review_gamma_gas_parents_refuse_unconverted_reference_standards(
    packaged_gas_pack,
) -> None:
    base = {
        "SiO2": 50.0,
        "MgO": 15.0,
        "FeO": 10.0,
        "CaO": 12.0,
        "Al2O3": 12.0,
    }
    source_markers = {
        "Ag2O": "Assal et al. (1997)",
        "Sb2O3": "Barin (1973)",
        "Tl2O": "Lamoreaux et al. (1987)",
    }
    for oxide, source in source_markers.items():
        rule = trace_component_rule(oxide)
        assert rule is not None
        composition = {
            **base,
            oxide: 0.001 * rule.molar_mass_g_mol,
        }
        melt = evaluate(
            composition, 1923.0, basis_type="wt", allow_extrapolation=True
        )
        activity_flag = next(
            item for item in melt.labels.notices
            if isinstance(item, TraceActivityFlag) and item.oxide == oxide
        )
        assert source in str(activity_flag)

        gas = evaluate_gas(
            _major_activities(melt),
            1923.0,
            1.0e-8,
            packaged_gas_pack,
            composition_wt_pct=composition,
        )
        element = rule.review_gamma_element
        assert element is not None
        refusal = gas.omitted_channels[element]
        assert isinstance(refusal, TraceGasChannelRefusal)
        assert oxide in refusal.triggers
        assert "no shipped standard conversion is available" in refusal.reason


def test_sp_extension_p2o5_proxy_is_refused_by_gas(packaged_gas_pack) -> None:
    composition = {
        "SiO2": 50.0,
        "MgO": 15.0,
        "FeO": 10.0,
        "CaO": 12.0,
        "Al2O3": 12.0,
        "P2O5": 0.1,
    }
    extension_pack = load_datapack(
        "src/openimcc/data/packs/imcc-sf04-d066-ext-v1.json"
    )
    extension_melt = evaluate(
        composition,
        1800.0,
        extension_pack,
        basis_type="wt",
        enable_sp_extension=True,
        allow_extrapolation=True,
        allow_out_of_envelope=True,
    )
    assert "P2O5" in extension_melt.parent_oxides
    with pytest.raises(
        ImccComponentOutsideDomainError,
        match="half-P4O10\\(cr\\)",
    ):
        evaluate_gas(
            _major_activities(extension_melt),
            1800.0,
            1.0e-8,
            packaged_gas_pack,
            gas_species=("P2",),
            composition_wt_pct=composition,
        )

    default_melt = evaluate(
        composition,
        1800.0,
        basis_type="wt",
        allow_extrapolation=True,
        allow_out_of_envelope=True,
    )
    default_gas = evaluate_gas(
        _major_activities(default_melt),
        1800.0,
        1.0e-8,
        packaged_gas_pack,
        gas_species=("P2",),
        composition_wt_pct=composition,
    )
    assert "P2" in default_gas


def test_li_bearing_cr_domain_warning_reaches_melt_and_gas_flags(
    packaged_gas_pack,
) -> None:
    composition = _weight_composition({"Cr2O3": 0.001, "Li2O": 0.001})
    melt = evaluate(
        composition, 1750.0, basis_type="wt", allow_extrapolation=True
    )
    melt_flag = next(
        item for item in melt.labels.notices
        if isinstance(item, TraceActivityFlag) and item.oxide == "Cr2O3"
    )
    assert melt_flag.code == "out_of_domain_associate_activity"
    assert "major_cation_composition" in melt_flag.estimates[0].violated_axes
    assert "unsupported major cations=Li" in str(melt_flag)
    assert "temperature_K" not in melt_flag.estimates[0].violated_axes

    gas = evaluate_gas(
        _major_activities(melt),
        1750.0,
        1.0e-8,
        packaged_gas_pack,
        gas_species=("Cr",),
        composition_wt_pct=composition,
    )
    gas_flag = gas.domain_flags["Cr"]
    assert isinstance(gas_flag, TraceActivityFlag)
    assert gas_flag.code == "out_of_domain_associate_activity"
    assert "major_cation_composition" in gas_flag.estimates[0].violated_axes
    assert "unsupported major cations=Li" in str(gas_flag)


def test_trace_level_mgo_p2o5_stays_dependent_through_gas_evaluation(
    packaged_gas_pack,
) -> None:
    composition = {
        "CaO": 40.0,
        "SiO2": 49.5,
        "FeO": 10.0,
        "MgO": 1.0e-10,
        "P2O5": 0.5,
    }
    melt = evaluate(
        composition, 1873.0, basis_type="wt", allow_extrapolation=True
    )
    melt_flag = next(
        item for item in melt.labels.notices
        if isinstance(item, TraceActivityFlag) and item.oxide == "P2O5"
    )
    estimate = melt_flag.estimates[0]
    assert estimate.code == "depends_helper_activity"
    assert estimate.violated_axes == ()
    assert "MgO_saturation" in estimate.dependency_axes
    assert any("MgO-saturated" in item for item in estimate.limitations)

    gas = evaluate_gas(
        _major_activities(melt),
        1873.0,
        1.0e-8,
        packaged_gas_pack,
        gas_species=("P2",),
        composition_wt_pct=composition,
    )
    gas_flag = gas.domain_flags["P2"]
    assert isinstance(gas_flag, TraceActivityFlag)
    assert gas_flag.code == "depends_helper_activity"
    assert "MgO_saturation" in gas_flag.dependency_axes
    assert "MgO_saturation" in str(gas_flag)
    assert any(
        "MgO-saturated" in item
        for item in gas_flag.estimates[0].limitations
    )

    absent_mgo = trace_components.resolve_trace_activity(
        "P2O5",
        0.002,
        {"CaO": 0.4, "FeO": 0.1, "SiO2": 0.5, "P2O5": 0.002},
        1873.0,
    )
    assert absent_mgo.code == "out_of_domain_helper_activity"
    assert "MgO_absent" in absent_mgo.violated_axes


def test_in_domain_results_remain_bit_identical_without_trace_components(
    packaged_gas_pack,
) -> None:
    composition = {
        "SiO2": 55.0,
        "MgO": 8.0,
        "FeO": 10.0,
        "CaO": 12.0,
        "Al2O3": 10.0,
        "TiO2": 2.0,
        "Na2O": 2.0,
        "K2O": 1.0,
    }
    plain = evaluate(composition, 1800.0, allow_extrapolation=True)
    expected_in_domain_bits = {
        (
            (
                "0x1.6916c5c0f1f08p-2",
                "0x1.020381240eac4p-8",
                "0x1.e8629fd54c604p-5",
                "0x1.183b5277600abp-14",
                "0x1.1287168dc05edp-6",
                "0x1.f24159dff3440p-9",
                "0x1.15240ec317686p-34",
                "0x1.c61d62943dcf2p-47",
            ),
            (
                "0x1.c14e3a40f644fp-22",
                "0x1.0f8757b206258p-15",
                "0x1.6f93fc11a42c0p-19",
                "0x1.4e2c09cfaadd5p-27",
            ),
        ),
        (
            (
                "0x1.6916c5c0f1f06p-2",
                "0x1.020381240eac8p-8",
                "0x1.e8629fd54c604p-5",
                "0x1.183b5277600b4p-14",
                "0x1.1287168dc05e9p-6",
                "0x1.f24159dff3440p-9",
                "0x1.15240ec317686p-34",
                "0x1.c61d62943dd2bp-47",
            ),
            (
                "0x1.c14e3a40f644cp-22",
                "0x1.0f8757b206258p-15",
                "0x1.6f93fc11a42d7p-19",
                "0x1.4e2c09cfaaddap-27",
            ),
        ),
    }
    # Exact outputs vary across the two numerical runtimes used for verification;
    # each tuple was checked against the unchanged base implementation.
    actual_parent_hex = tuple(value.hex() for value in plain.parent_activity)
    with_zero_trace = evaluate(
        {**composition, "NiO": 0.0}, 1800.0, allow_extrapolation=True
    )
    assert np.array_equal(plain.parent_activity, with_zero_trace.parent_activity)
    assert plain.labels == with_zero_trace.labels
    gas = evaluate_gas(
        _major_activities(plain),
        1800.0,
        1.0e-8,
        packaged_gas_pack,
        gas_species=("SiO", "Na", "K", "Mg"),
    )
    actual_gas_hex = tuple(gas[name].hex() for name in ("SiO", "Na", "K", "Mg"))
    assert (actual_parent_hex, actual_gas_hex) in expected_in_domain_bits


def test_mole_fraction_helper_uses_registered_formula_masses() -> None:
    composition = _weight_composition({"P2O5": 0.005, "NiO": 0.002})
    moles, fractions = trace_fractions_from_wt(composition)
    assert fractions["P2O5"] == pytest.approx(0.005, abs=1e-15)
    assert fractions["NiO"] == pytest.approx(0.002, abs=1e-15)
    assert moles["P2O5"] == pytest.approx(
        composition["P2O5"] / MOLAR_MASS_G_MOL["P2O5"]
    )
