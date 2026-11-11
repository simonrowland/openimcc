from __future__ import annotations

from dataclasses import FrozenInstanceError
import math

import numpy as np
import pytest

from openimcc import (
    ImccTraceLimitExceededError,
    ImccTraceMajorBiasError,
    evaluate,
    load_datapack,
    supported_trace_oxides,
)
from openimcc._constants import MOLAR_MASS_G_MOL
from openimcc.gas import (
    TraceActivityFlag,
    TraceGasChannelRefusal,
    evaluate_gas,
)
from openimcc.gas_data import IMCC_PARENT_OXIDES
from openimcc.kernel import (
    ImccComponentOutsideDomainError,
    ImccCompositionIncompleteError,
)
from openimcc.redox import evaluate_redox
from openimcc.redox_pack import load_redox_pack
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


def _major_activities(result) -> dict[str, float]:
    return dict(zip(result.parent_oxides, result.parent_activity, strict=True))


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


def test_basic_monoxide_envelope_scores_dilute_cobalt_and_caps_major_rows() -> None:
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
    assert sodium.scoreable
    assert sodium.bias_bound_dex <= 0.02
    assert dict(sodium.mn_calibrated_envelope) == {
        "k": BASIC_MONOXIDE_ENVELOPE_K,
        "calibrant": "MnO",
        "class": "basic_monoxide",
        "temperature_range_K": (1400.0, 2200.0),
        "log10_fO2_range_bar": trace_components.ENVELOPE_LOG10_FO2_RANGE_BAR,
        "delta_IW_grid": (-6.0, -4.0, -2.0, 0.0, 2.0, 4.0),
    }
    assert "mn_calibrated_envelope" in str(sodium)
    assert math.isfinite(dilute.activity("Na2O"))

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
    assert all(item.bias_bound_dex > 0.02 for item in concentrated_flags)
    with pytest.raises(ImccComponentOutsideDomainError, match="exceeds 0.02 dex"):
        concentrated.activity("Na2O")


@pytest.mark.parametrize(
    ("temperature_K", "uses_envelope"),
    [(1399.0, False), (1400.0, True), (2200.0, True), (2201.0, False)],
)
def test_trace_envelope_temperature_gate_is_inclusive(
    temperature_K: float, uses_envelope: bool
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
    if uses_envelope:
        assert sodium.envelope_flags
        assert sodium.limiting_cap_dex == 0.02
        assert sodium.scoreable
    else:
        assert sodium.envelope_flags == ()
        assert sodium.limiting_cap_dex == 0.01
        assert "temperature outside 1400–2200 K" in str(sodium)
        assert not sodium.scoreable


def test_gas_trace_bias_uses_melt_envelope_flags(packaged_gas_pack) -> None:
    def evaluate_composition(x_coo: float) -> tuple[dict[str, float], dict[str, float]]:
        mole_composition = _mole_fraction_composition(
            _MARE_BASALT_WT, "CoO", x_coo
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
    dilute_gas = evaluate_gas(
        dilute_activities,
        1800.0,
        1.0e-8,
        packaged_gas_pack,
        gas_species=("Na", "SiO", "Mg"),
        composition_wt_pct=dilute_composition,
    )
    assert "Na" in dilute_gas
    sodium_flag = dilute_gas.domain_flags["Na"]
    assert isinstance(sodium_flag, TraceMajorBiasFlag)
    assert sodium_flag.mn_calibrated_envelope is not None
    assert sodium_flag.scoreable
    magnesium_flag = dilute_gas.domain_flags["Mg"]
    assert isinstance(magnesium_flag, TraceMajorBiasFlag)
    assert magnesium_flag.mn_calibrated_envelope is not None
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


def test_activity_ladder_uses_source_rows_helpers_and_the_shared_registration() -> None:
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
    assert p2o5_notice.activities["P2O5"].code == "source_grounded_helper_activity"

    co_composition = _weight_composition({"CoO": 0.002})
    co_result = evaluate(
        co_composition, 1800.0, basis_type="wt", allow_extrapolation=True
    )
    co_notice = next(
        item for item in co_result.labels.notices
        if isinstance(item, TraceComponentsOutsideMeltModelNotice)
    )
    assert co_notice.activities["CoO"].code == "source_grounded_helper_activity"

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
    assert sulfur.code == "source_grounded_helper_activity"

    cr_composition = _weight_composition({"Cr2O3": 0.001})
    cr_result = evaluate(
        cr_composition, 1800.0, basis_type="wt", allow_extrapolation=True
    )
    cr_notice = next(
        item for item in cr_result.labels.notices
        if isinstance(item, TraceComponentsOutsideMeltModelNotice)
    )
    cr_estimate = cr_notice.activities["Cr2O3"]
    assert cr_estimate.code == "source_grounded_associate_activity"

    mn_composition = _weight_composition({"MnO": 0.002})
    mn_result = evaluate(
        mn_composition, 1800.0, basis_type="wt", allow_extrapolation=True
    )
    mn_flag = next(
        item for item in mn_result.labels.notices
        if isinstance(item, TraceActivityFlag) and item.oxide == "MnO"
    )
    assert mn_flag.code == "source_grounded_associate_activity"
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
    assert estimate.code == "source_grounded_associate_activity"
    assert math.isfinite(estimate.activity) and estimate.activity > 0.0


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

    expected = evaluate_redox(
        _redox_inventory(composition), 1800.0, "closed"
    )
    result = evaluate(
        composition,
        1800.0,
        basis_type="mol",
        allow_extrapolation=True,
    )

    for major, activity in expected.parent_activity.items():
        if major in result.parent_oxides:
            assert result.activity(major) == pytest.approx(activity, rel=1e-12)
    assert result.activity("MnO") == pytest.approx(
        expected.parent_activity["MnO"], rel=1e-12
    )
    assert "RED: unmodelled_redox_couples" in expected.flags
    expected_mn_thermochemistry_flags = {
        flag for flag in expected.flags
        if flag.startswith("Mn")
    }
    assert expected_mn_thermochemistry_flags
    assert expected_mn_thermochemistry_flags <= set(result.labels.flags)
    assert "RED: unmodelled_redox_couples" in result.labels.flags
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
    assert result.convergence.solver_path == expected.solver_path
    assert result.convergence.iterations > 0
    assert not any(
        isinstance(item, (TraceComponentsOutsideMeltModelNotice, TraceMajorBiasFlag))
        for item in result.labels.notices
    )


def test_fe_free_manganese_associate_path() -> None:
    composition = {
        "SiO2": 0.45,
        "MgO": 0.2,
        "CaO": 0.15,
        "Al2O3": 0.1,
        "MnO": 0.1,
    }
    expected = evaluate_redox(
        _redox_inventory(composition), 1800.0, "closed"
    )

    result = evaluate(
        composition,
        1800.0,
        basis_type="mol",
        allow_extrapolation=True,
    )
    assert result.activity("MnO") == pytest.approx(
        expected.parent_activity["MnO"], rel=1e-12
    )


def test_manganese_associate_refusal_uses_published_gamma_rung(
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
    flag = next(
        item for item in result.labels.notices
        if isinstance(item, TraceActivityFlag) and item.oxide == "MnO"
    )
    assert flag.code == "published_gamma_x_assumption"
    assert flag.gamma == 1.9
    assert flag.activity == pytest.approx(1.9 * flag.x)


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
    assert gas.domain_flags["Ni"].code == "source_grounded_helper_activity"
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
    assert p2o5_gas.domain_flags["P2"].code == "source_grounded_helper_activity"


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
