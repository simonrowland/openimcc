"""Trace-component rules shared by melt and gas evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
import math
import re
from types import MappingProxyType
from typing import Any, Literal

from openimcc._constants import MOLAR_MASS_G_MOL
from openimcc.kernel import ImccRefusal
from openimcc.trace_envelope_calibration import (
    CR_CALIBRATED_ENVELOPE_ID,
    ENVELOPE_DELTA_IW_GRID,
    ENVELOPE_LOG10_FO2_RANGE_BAR,
    ENVELOPE_T_RANGE_K,
    MN_CALIBRATED_ENVELOPE_ID,
    _iw_log10_fO2_bar,
)


_SOSSI_FEGLY_2018 = (
    "Sossi & Fegley (2018), Rev. Mineral. Geochem. 84, Table 2, pp. 409–410, "
    "doi:10.2138/rmg.2018.84.11"
)

# Caller composition limits, distinct from the per-major bias caps below.
_TRACE_COMPONENT_MAX_MOLE_FRACTION = 0.01
_TRACE_COMPONENT_MAX_TOTAL_MOLE_FRACTION = 0.02
_TRACE_GAS_CHANNEL_BLOCKING_MOLE_FRACTION = 0.001
_SINGLE_CATION_OXIDE = re.compile(r"([A-Z][a-z]?)(\d*)O(\d*)")


def _single_cation_oxide_counts(component: str) -> tuple[int, int] | None:
    """Return cation and oxygen counts for an ``M_aO_b`` formula."""
    match = _SINGLE_CATION_OXIDE.fullmatch(component)
    if match is None:
        return None
    return int(match.group(2) or 1), int(match.group(3) or 1)


@dataclass(frozen=True)
class TraceComponentRule:
    molar_mass_g_mol: float
    nu: int
    nu_reason: str
    activity_model: Literal[
        "redox_associates", "activity_fragment", "review_gamma", "ideal"
    ]
    trace_class: Literal[
        "basic_monoxide", "basic_oxide", "acidic", "amphoteric", "non_oxide"
    ]
    trace_class_reason: str
    gas_parent_oxide: str | None = None
    gamma: float | None = None
    gamma_species: str | None = None
    gamma_reference: str | None = None
    full_component: bool = False
    gas_channel_blocked_above_mole_fraction: float | None = None
    activity_fragment: Literal["Ni", "Co", "P", "S"] | None = None
    uses_review_gamma_evaluator: bool = False
    review_gamma_element: Literal[
        "Sn", "Ga", "In", "Cu", "Ge", "As", "Bi", "Pb", "W", "Ce", "La",
        "V", "Ag", "Sb", "Tl", "Rb",
    ] | None = None


_O_MASS = MOLAR_MASS_G_MOL["O"]
_TRACE_COMPONENT_RULES = MappingProxyType(
    {
        # ν is the maximum major-oxide binding stoichiometry per trace formula
        # unit. Keep every component's value and physical assignment together
        # here so a newly integrated source row has one registration point.
        "MnO": TraceComponentRule(
            MOLAR_MASS_G_MOL["MnO"], 1,
            "one monoxide unit can bind at most one major-oxide unit",
            "redox_associates", gas_parent_oxide="MnO", gamma=1.9, gamma_species="MnO",
            trace_class="basic_monoxide",
            trace_class_reason="MnO is a basic divalent monoxide and the modelled class calibrant",
            gamma_reference=_SOSSI_FEGLY_2018, full_component=True,
        ),
        "P2O5": TraceComponentRule(
            MOLAR_MASS_G_MOL["P2O5"], 3,
            "three major-oxide units per phosphate formula unit is the specified maximum",
            "activity_fragment", trace_class="acidic",
            trace_class_reason="P(V) oxide is an acidic network-forming oxide",
            activity_fragment="P", gas_parent_oxide="P2O5",
        ),
        "Cr2O3": TraceComponentRule(
            MOLAR_MASS_G_MOL["Cr2O3"], 1,
            "one major-oxide unit per chromium-oxide formula unit is the specified maximum",
            "redox_associates", trace_class="amphoteric",
            trace_class_reason="Cr(III) oxide is amphoteric and has a separate redox calibrant",
            gas_parent_oxide="Cr2O3", gamma=31.1, gamma_species="CrO1.5",
            gamma_reference=_SOSSI_FEGLY_2018,
        ),
        "NiO": TraceComponentRule(
            MOLAR_MASS_G_MOL["NiO"], 1,
            "one monoxide unit can bind at most one major-oxide unit", "activity_fragment",
            trace_class="basic_monoxide",
            trace_class_reason="NiO is a basic divalent monoxide",
            activity_fragment="Ni",
            uses_review_gamma_evaluator=True,
            gas_parent_oxide="NiO",
        ),
        "CoO": TraceComponentRule(
            MOLAR_MASS_G_MOL["CoO"], 1,
            "one monoxide unit can bind at most one major-oxide unit", "activity_fragment",
            trace_class="basic_monoxide",
            trace_class_reason="CoO is a basic divalent monoxide",
            activity_fragment="Co",
            uses_review_gamma_evaluator=True,
            gas_parent_oxide="CoO",
        ),
        "PbO": TraceComponentRule(
            MOLAR_MASS_G_MOL["Pb"] + _O_MASS, 1,
            "one monoxide unit can bind at most one major-oxide unit",
            "review_gamma",
            trace_class="basic_monoxide",
            trace_class_reason="PbO is a basic divalent monoxide with a liquid-standard activity estimate",
            gas_parent_oxide="PbO",
            review_gamma_element="Pb",
        ),
        "WO3": TraceComponentRule(
            MOLAR_MASS_G_MOL["W"] + 3.0 * _O_MASS, 1,
            "one oxide formula unit can bind at most one major-oxide unit",
            "review_gamma",
            trace_class="acidic",
            trace_class_reason="WO3 is a hexavalent trace oxide with a liquid-standard activity estimate",
            gas_parent_oxide="WO3",
            review_gamma_element="W",
        ),
        "SrO": TraceComponentRule(
            MOLAR_MASS_G_MOL["Sr"] + _O_MASS, 1,
            "one monoxide unit can bind at most one major-oxide unit", "redox_associates",
            trace_class="basic_monoxide",
            trace_class_reason="SrO is a basic alkaline-earth monoxide",
            gas_parent_oxide="SrO",
        ),
        "SO3": TraceComponentRule(
            MOLAR_MASS_G_MOL["S"] + 3.0 * _O_MASS, 1,
            "one sulfate formula unit can bind at most one major-oxide unit", "ideal",
            trace_class="acidic",
            trace_class_reason="SO3 is an acidic sulfur(VI) oxide",
            gas_parent_oxide="S2",
            gas_channel_blocked_above_mole_fraction=(
                _TRACE_GAS_CHANNEL_BLOCKING_MOLE_FRACTION
            ),
        ),
        "S": TraceComponentRule(
            MOLAR_MASS_G_MOL["S"], 1,
            "one sulfur component unit can bind at most one major-oxide unit", "activity_fragment",
            trace_class="non_oxide",
            trace_class_reason="S is an elemental trace component, not an oxide",
            activity_fragment="S",
            gas_parent_oxide="S2",
            gas_channel_blocked_above_mole_fraction=(
                _TRACE_GAS_CHANNEL_BLOCKING_MOLE_FRACTION
            ),
        ),
        "Li2O": TraceComponentRule(
            2.0 * MOLAR_MASS_G_MOL["Li"] + _O_MASS, 1,
            "one alkali-oxide unit can bind at most one major-oxide unit", "redox_associates",
            trace_class="basic_oxide",
            trace_class_reason="Li2O is a basic alkali oxide but not a monoxide formula",
            gas_parent_oxide="Li2O",
        ),
        "Li": TraceComponentRule(
            MOLAR_MASS_G_MOL["Li"], 1,
            "one lithium component unit can bind at most one major-oxide unit", "ideal",
            trace_class="non_oxide",
            trace_class_reason="Li is an elemental trace component, not an oxide",
            gas_parent_oxide="Li2O",
        ),
        "ZnO": TraceComponentRule(
            MOLAR_MASS_G_MOL["Zn"] + _O_MASS, 1,
            "one monoxide unit can bind at most one major-oxide unit", "redox_associates",
            trace_class="basic_monoxide",
            trace_class_reason="ZnO is a basic divalent monoxide",
            gas_parent_oxide="ZnO",
        ),
        "Zn": TraceComponentRule(
            MOLAR_MASS_G_MOL["Zn"], 1,
            "one zinc component unit can bind at most one major-oxide unit", "ideal",
            trace_class="non_oxide",
            trace_class_reason="Zn is an elemental trace component, not an oxide",
            gas_parent_oxide="ZnO",
        ),
        "Cl": TraceComponentRule(
            MOLAR_MASS_G_MOL["Cl"], 1,
            "one halogen component unit can bind at most one major-oxide unit", "ideal",
            trace_class="non_oxide",
            trace_class_reason="Cl is an elemental trace component, not an oxide",
            gas_parent_oxide="Cl2",
            gas_channel_blocked_above_mole_fraction=(
                _TRACE_GAS_CHANNEL_BLOCKING_MOLE_FRACTION
            ),
        ),
        "F": TraceComponentRule(
            MOLAR_MASS_G_MOL["F"], 1,
            "one halogen component unit can bind at most one major-oxide unit", "ideal",
            trace_class="non_oxide",
            trace_class_reason="F is an elemental trace component, not an oxide",
            gas_parent_oxide="F2",
            gas_channel_blocked_above_mole_fraction=(
                _TRACE_GAS_CHANNEL_BLOCKING_MOLE_FRACTION
            ),
        ),
        "Yb2O3": TraceComponentRule(
            2.0 * MOLAR_MASS_G_MOL["Yb"] + 3.0 * _O_MASS, 1,
            "one rare-earth oxide unit can bind at most one major-oxide unit", "redox_associates",
            trace_class="basic_oxide",
            trace_class_reason="Yb2O3 is a basic rare-earth oxide but not a monoxide formula",
            gas_parent_oxide="Yb2O3",
        ),
        "SnO": TraceComponentRule(
            MOLAR_MASS_G_MOL["Sn"] + _O_MASS,
            1,
            "one tin monoxide unit can bind at most one major-oxide unit",
            "review_gamma",
            trace_class="basic_oxide",
            trace_class_reason="SnO is an amphoteric trace oxide in the silicate slag source",
            gas_parent_oxide="SnO",
            review_gamma_element="Sn",
        ),
        "Ga2O3": TraceComponentRule(
            2.0 * MOLAR_MASS_G_MOL["Ga"] + 3.0 * _O_MASS,
            1,
            "one gallium sesquioxide unit can bind at most one major-oxide unit",
            "review_gamma",
            trace_class="basic_oxide",
            trace_class_reason="Ga2O3 is a trivalent trace oxide with a direct liquid-component activity estimate",
            gas_parent_oxide="Ga2O3",
            review_gamma_element="Ga",
        ),
        "In2O3": TraceComponentRule(
            2.0 * MOLAR_MASS_G_MOL["In"] + 3.0 * _O_MASS,
            1,
            "one indium sesquioxide unit can bind at most one major-oxide unit",
            "review_gamma",
            trace_class="basic_oxide",
            trace_class_reason="In2O3 is a trivalent trace oxide with a direct liquid-component activity estimate",
            gas_parent_oxide="In2O3",
            review_gamma_element="In",
        ),
        "Cu2O": TraceComponentRule(
            2.0 * MOLAR_MASS_G_MOL["Cu"] + _O_MASS,
            1,
            "one cuprous oxide unit can bind at most one major-oxide unit",
            "review_gamma",
            trace_class="basic_oxide",
            trace_class_reason="Cu2O is a cuprous trace oxide with a published melt activity estimate",
            gas_parent_oxide="Cu2O",
            review_gamma_element="Cu",
        ),
        "GeO2": TraceComponentRule(
            MOLAR_MASS_G_MOL["Ge"] + 2.0 * _O_MASS,
            1,
            "one germanium dioxide unit can bind at most one major-oxide unit",
            "review_gamma",
            trace_class="acidic",
            trace_class_reason="GeO2 is an acidic trace oxide with a published melt activity estimate",
            gas_parent_oxide="GeO2",
            review_gamma_element="Ge",
        ),
        "As2O3": TraceComponentRule(
            2.0 * MOLAR_MASS_G_MOL["As"] + 3.0 * _O_MASS,
            1,
            "one arsenic sesquioxide unit can bind at most one major-oxide unit",
            "review_gamma",
            trace_class="acidic",
            trace_class_reason="As2O3 is an acidic trace oxide with a published liquid-from-solid parent estimate",
            gas_parent_oxide="As2O3",
            review_gamma_element="As",
        ),
        "Bi2O3": TraceComponentRule(
            2.0 * MOLAR_MASS_G_MOL["Bi"] + 3.0 * _O_MASS,
            1,
            "one bismuth sesquioxide unit can bind at most one major-oxide unit",
            "review_gamma",
            trace_class="basic_oxide",
            trace_class_reason="Bi2O3 is a trivalent trace oxide with a published melt activity estimate",
            gas_parent_oxide="Bi2O3",
            review_gamma_element="Bi",
        ),
        "Ce2O3": TraceComponentRule(
            2.0 * MOLAR_MASS_G_MOL["Ce"] + 3.0 * _O_MASS,
            1,
            "one Ce2O3 formula unit is one source component unit",
            "review_gamma",
            trace_class="basic_oxide",
            trace_class_reason="Ce2O3 is the trivalent rare-earth oxide measured in the reducing slag sources",
            gas_parent_oxide="Ce2O3",
            review_gamma_element="Ce",
        ),
        "La2O3": TraceComponentRule(
            2.0 * MOLAR_MASS_G_MOL["La"] + 3.0 * _O_MASS,
            1,
            "one La2O3 formula unit is one source component unit",
            "review_gamma",
            trace_class="basic_oxide",
            trace_class_reason="La2O3 is a trivalent rare-earth oxide with an assessed silicate-melt activity limit",
            gas_parent_oxide="La2O3",
            review_gamma_element="La",
        ),
        "V2O3": TraceComponentRule(
            2.0 * MOLAR_MASS_G_MOL["V"] + 3.0 * _O_MASS,
            1,
            "one sesquioxide unit can bind at most one major-oxide unit",
            "review_gamma",
            trace_class="basic_oxide",
            trace_class_reason="V(III) sesquioxide is a basic trivalent oxide",
            gas_parent_oxide="V2O3",
            review_gamma_element="V",
        ),
        "Ag2O": TraceComponentRule(
            2.0 * MOLAR_MASS_G_MOL["Ag"] + _O_MASS,
            1,
            "one monovalent oxide formula unit can bind at most one major-oxide unit",
            "review_gamma",
            trace_class="basic_oxide",
            trace_class_reason="Ag(I) oxide is a basic oxide with a liquid-standard activity estimate",
            gas_parent_oxide="Ag2O",
            review_gamma_element="Ag",
        ),
        "Sb2O3": TraceComponentRule(
            2.0 * MOLAR_MASS_G_MOL["Sb"] + 3.0 * _O_MASS,
            1,
            "one sesquioxide unit can bind at most one major-oxide unit",
            "review_gamma",
            trace_class="basic_oxide",
            trace_class_reason="Sb(III) sesquioxide is amphoteric",
            gas_parent_oxide="Sb2O3",
            review_gamma_element="Sb",
        ),
        "Tl2O": TraceComponentRule(
            2.0 * MOLAR_MASS_G_MOL["Tl"] + _O_MASS,
            1,
            "one monovalent oxide formula unit can bind at most one major-oxide unit",
            "review_gamma",
            trace_class="basic_oxide",
            trace_class_reason="Tl(I) oxide is a basic oxide with a liquid-standard activity estimate",
            gas_parent_oxide="Tl2O",
            review_gamma_element="Tl",
        ),
        "Rb2O": TraceComponentRule(
            2.0 * MOLAR_MASS_G_MOL["Rb"] + _O_MASS,
            1,
            "one alkali-oxide unit can bind at most one major-oxide unit",
            "review_gamma",
            trace_class="basic_oxide",
            trace_class_reason="Rb(I) oxide is a basic alkali oxide with a liquid-standard activity estimate",
            gas_parent_oxide="Rb2O",
            review_gamma_element="Rb",
        ),
    }
)


def _review_gamma_parent_oxides(rules: Mapping[str, TraceComponentRule]) -> frozenset[str]:
    return frozenset(
        rule.gas_parent_oxide
        for rule in rules.values()
        if rule.gas_parent_oxide is not None
        and (rule.activity_model == "review_gamma" or rule.uses_review_gamma_evaluator)
    )

_TRACE_STRICT_BOUND_CAP_DEX = 0.01


@dataclass(frozen=True)
class ParentLiquidStandardOffset:
    """Activity ratio converting a measured parent to the selected liquid standard."""

    code: Literal["parent_liquid_standard_offset"]
    factor: float
    recommendation: str


@dataclass(frozen=True)
class TraceActivityEstimate:
    oxide: str
    activity: float
    x: float
    code: str
    gamma: float | None = None
    reference: str | None = None
    activity_range: tuple[float, float] | None = None
    x_source: float | None = None
    parent_liquid_standard_offset: ParentLiquidStandardOffset | None = None
    violated_axes: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()


class TraceActivityFlag(str):
    """String-compatible typed flag identifying the activity-ladder rung."""

    code: str
    oxide: str
    activity: float
    x: float
    gamma: float | None
    reference: str | None
    activity_range: tuple[float, float] | None
    x_source: float | None
    parent_liquid_standard_offset: ParentLiquidStandardOffset | None
    estimates: tuple[TraceActivityEstimate, ...]
    major_flags: tuple["TraceMajorBiasFlag", ...]
    context: str
    prior_flags: tuple[str, ...]

    def __new__(
        cls,
        estimate: TraceActivityEstimate | tuple[TraceActivityEstimate, ...],
        *,
        context: str = "melt",
        prior_flags: tuple[str, ...] = (),
        major_flags: tuple["TraceMajorBiasFlag", ...] = (),
    ):
        estimates = (estimate,) if isinstance(estimate, TraceActivityEstimate) else estimate
        estimate_text = []
        for item in estimates:
            message = (
                f"{item.code}: {item.oxide} activity={item.activity:.12g} "
                f"x={item.x:.12g} context={context}"
            )
            if item.gamma is not None:
                message += f" gamma={item.gamma:.12g}"
            if item.activity_range is not None:
                message += (
                    f" activity_range=[{item.activity_range[0]:.12g},"
                    f" {item.activity_range[1]:.12g}]"
                )
            if item.x_source is not None:
                message += f" x_source={item.x_source:.12g}"
            if item.parent_liquid_standard_offset is not None:
                offset = item.parent_liquid_standard_offset
                message += (
                    f" {offset.code}={offset.factor:.12g}; "
                    f"{offset.recommendation}"
                )
            if item.violated_axes:
                message += "; violated axes: " + "; ".join(item.violated_axes)
            reference = item.reference
            if item.limitations:
                limitation_text = " ".join(item.limitations)
                embedded_limitations = f"; limitation: {limitation_text}"
                if reference and embedded_limitations in reference:
                    reference = reference.replace(embedded_limitations, "", 1)
                message += f"; limitations: {limitation_text}"
            if reference:
                message += f"; {reference}"
            estimate_text.append(message)
        prior_flags = tuple(prior_flags)
        major_flags = tuple(major_flags)
        major_text = tuple(str(flag) for flag in major_flags)
        message = "; ".join((*prior_flags, *major_text, *estimate_text))
        flag = super().__new__(cls, message)
        first = estimates[0]
        flag.code = first.code if len(estimates) == 1 else "trace_activity_ladder"
        flag.oxide = first.oxide
        flag.activity = first.activity
        flag.x = first.x
        flag.gamma = first.gamma
        flag.reference = first.reference
        flag.activity_range = first.activity_range
        flag.x_source = first.x_source
        flag.parent_liquid_standard_offset = first.parent_liquid_standard_offset
        flag.estimates = tuple(estimates)
        flag.major_flags = major_flags
        flag.context = context
        flag.prior_flags = prior_flags
        return flag

    def __reduce__(self):
        return (
            partial(
                type(self),
                self.estimates,
                context=self.context,
                prior_flags=self.prior_flags,
                major_flags=self.major_flags,
            ),
            (),
        )


class TraceMajorBiasFlag(str):
    """Typed per-major bound and row-scoring decision."""

    code = "trace_major_activity_bound"

    def __new__(
        cls,
        major_oxide: str,
        x_free: float,
        bias_bound_dex: float,
        scoreable: bool,
        *,
        limiting_bound_dex: float | None = None,
        limiting_cap_dex: float = _TRACE_STRICT_BOUND_CAP_DEX,
        prior_flags: tuple[str, ...] = (),
        additional_major_flags: tuple["TraceMajorBiasFlag", ...] = (),
        envelope_flags: tuple[tuple[str, float, str, str], ...] = (),
    ):
        disposition = "scoreable" if scoreable else "refused"
        envelope_domains = {
            MN_CALIBRATED_ENVELOPE_ID: {
                "temperature_range_K": ENVELOPE_T_RANGE_K,
                "log10_fO2_range_bar": ENVELOPE_LOG10_FO2_RANGE_BAR,
                "delta_IW_grid": ENVELOPE_DELTA_IW_GRID,
            },
            CR_CALIBRATED_ENVELOPE_ID: {
                "temperature_range_K": ENVELOPE_T_RANGE_K,
                "log10_fO2_range_bar": ENVELOPE_LOG10_FO2_RANGE_BAR,
                "delta_IW_grid": ENVELOPE_DELTA_IW_GRID,
            },
        }
        envelope_text = tuple(
            f'{name} {{k: {k:.12g}, calibrant: "{calibrant}", '
            f'class: "{trace_class}", T_K: '
            f'{envelope_domains[name]["temperature_range_K"]}, log10_fO2/bar: '
            f'{envelope_domains[name]["log10_fO2_range_bar"]}}}'
            for name, k, calibrant, trace_class in envelope_flags
        )
        message = "; ".join((
            f"trace_major_activity_bound: {major_oxide} x_free={x_free:.12g} "
            f"bias_bound_dex={bias_bound_dex:.12g} row={disposition}",
            *envelope_text,
            *prior_flags,
        ))
        flag = super().__new__(cls, message)
        flag.major_oxide = major_oxide
        flag.x_free = x_free
        flag.bias_bound_dex = bias_bound_dex
        flag.scoreable = scoreable
        flag.limiting_bound_dex = (
            bias_bound_dex if limiting_bound_dex is None else limiting_bound_dex
        )
        flag.limiting_cap_dex = limiting_cap_dex
        flag.prior_flags = tuple(prior_flags)
        flag.additional_major_flags = tuple(additional_major_flags)
        flag.envelope_flags = tuple(envelope_flags)
        flag.envelope_domains = MappingProxyType({
            name: MappingProxyType(envelope_domains[name])
            for name, *_rest in envelope_flags
        })
        flag.mn_calibrated_envelope = next(
            (
                MappingProxyType({
                    "k": k, "calibrant": calibrant, "class": trace_class,
                    **envelope_domains[name],
                })
                for name, k, calibrant, trace_class in envelope_flags
                if name == MN_CALIBRATED_ENVELOPE_ID
            ),
            None,
        )
        flag.cr_calibrated_envelope = next(
            (
                MappingProxyType({
                    "k": k, "calibrant": calibrant, "class": trace_class,
                    **envelope_domains[name],
                })
                for name, k, calibrant, trace_class in envelope_flags
                if name == CR_CALIBRATED_ENVELOPE_ID
            ),
            None,
        )
        return flag

    def __reduce__(self):
        return _restore_trace_major_bias_flag, (
            self.major_oxide,
            self.x_free,
            self.bias_bound_dex,
            self.scoreable,
            self.limiting_bound_dex,
            self.limiting_cap_dex,
            self.prior_flags,
            self.additional_major_flags,
            self.envelope_flags,
        )


def _restore_trace_major_bias_flag(
    major_oxide: str,
    x_free: float,
    bias_bound_dex: float,
    scoreable: bool,
    limiting_bound_dex: float,
    limiting_cap_dex: float,
    prior_flags: tuple[str, ...],
    additional_major_flags: tuple[TraceMajorBiasFlag, ...],
    envelope_flags: tuple[tuple[str, float, str, str], ...],
) -> TraceMajorBiasFlag:
    return TraceMajorBiasFlag(
        major_oxide,
        x_free,
        bias_bound_dex,
        scoreable,
        limiting_bound_dex=limiting_bound_dex,
        limiting_cap_dex=limiting_cap_dex,
        prior_flags=prior_flags,
        additional_major_flags=additional_major_flags,
        envelope_flags=envelope_flags,
    )


class TraceComponentsOutsideMeltModelNotice(str):
    """Typed notice containing trace composition, activities, and major bounds."""

    code = "trace_components_outside_melt_model"
    family = code
    score = "unscored"

    def __new__(
        cls,
        components: Mapping[str, float],
        total: float,
        major_bias_bounds: Mapping[str, float],
        activities: Mapping[str, TraceActivityEstimate],
    ):
        component_text = ", ".join(
            f"{oxide}: {amount:.12g}" for oxide, amount in components.items()
        )
        bounds_text = ", ".join(
            f"{oxide}: {bound:.12g}" for oxide, bound in major_bias_bounds.items()
        )
        message = (
            f"trace_components_outside_melt_model {{{component_text}, total: {total:.12g}, "
            f"bias_bound_dex: {{{bounds_text}}}}}"
        )
        notice = super().__new__(cls, message)
        notice.components = MappingProxyType(dict(components))
        notice.total = total
        notice.major_bias_bounds = MappingProxyType(dict(major_bias_bounds))
        notice.bias_bound_dex = notice.major_bias_bounds
        notice.activities = MappingProxyType(dict(activities))
        return notice

    def __reduce__(self):
        return type(self), (
            dict(self.components), self.total, dict(self.major_bias_bounds),
            dict(self.activities),
        )


class TraceGasChannelRefusal(str):
    """Typed notice for a gas channel excluded by the trace policy."""

    code = "trace_gas_channel_refused"

    def __new__(
        cls,
        species: str,
        reason: str,
        *,
        triggers: tuple[str, ...] = (),
        major_flags: tuple[TraceMajorBiasFlag, ...] = (),
    ):
        message = f"{cls.code}: {species}: {reason}"
        refusal = super().__new__(cls, message)
        refusal.species = species
        refusal.reason = reason
        refusal.triggers = tuple(triggers)
        refusal.major_flags = tuple(major_flags)
        return refusal

    def __reduce__(self):
        return type(self), (self.species, self.reason), {
            "triggers": self.triggers,
            "major_flags": self.major_flags,
        }


def trace_component_rule(oxide: str) -> TraceComponentRule | None:
    return _TRACE_COMPONENT_RULES.get(oxide)


@dataclass(frozen=True)
class SupportedTraceOxide:
    """Registered trace component and its source and concentration limits."""

    oxide: str
    trace_class: Literal[
        "basic_monoxide", "basic_oxide", "acidic", "amphoteric", "non_oxide"
    ]
    full_component: bool
    activity_source: Literal[
        "redox_associates", "helper", "review_gamma", "ideal_flagged"
    ]
    max_mole_fraction: float | None
    max_total_mole_fraction: float | None
    gas_channel_blocked_above_mole_fraction: float | None


def supported_trace_oxides() -> tuple[SupportedTraceOxide, ...]:
    """List registered trace components in deterministic name order.

    Entries include registered elemental components such as S, Cl, and F;
    ``trace_class`` distinguishes those from oxide formulas.
    """
    from openimcc.activity_fragments import load_activity_fragment

    supported: list[SupportedTraceOxide] = []
    for oxide, rule in sorted(_TRACE_COMPONENT_RULES.items()):
        if rule.activity_fragment is not None:
            load_activity_fragment(rule.activity_fragment)
        if rule.review_gamma_element is not None:
            load_activity_fragment(rule.review_gamma_element)

        if rule.activity_model == "redox_associates":
            activity_source = "redox_associates"
        elif rule.activity_model == "activity_fragment":
            activity_source = "helper"
        elif rule.activity_model == "review_gamma":
            activity_source = "review_gamma"
        elif rule.activity_model == "ideal":
            activity_source = "ideal_flagged"
        else:
            raise ValueError(f"unknown trace activity source {rule.activity_model!r}")

        supported.append(
            SupportedTraceOxide(
                oxide=oxide,
                trace_class=rule.trace_class,
                full_component=rule.full_component,
                activity_source=activity_source,
                max_mole_fraction=(
                    None
                    if rule.full_component
                    else _TRACE_COMPONENT_MAX_MOLE_FRACTION
                ),
                max_total_mole_fraction=(
                    None
                    if rule.full_component
                    else _TRACE_COMPONENT_MAX_TOTAL_MOLE_FRACTION
                ),
                gas_channel_blocked_above_mole_fraction=(
                    rule.gas_channel_blocked_above_mole_fraction
                ),
            )
        )
    return tuple(supported)


def _activity_fragment_registered(element: str) -> bool:
    """Return whether the trace activity table registers this helper fragment."""
    return any(
        rule.activity_fragment == element or rule.review_gamma_element == element
        for rule in _TRACE_COMPONENT_RULES.values()
    )


def trace_fractions_from_wt(
    composition_wt_pct: Mapping[str, float],
) -> tuple[Mapping[str, float], Mapping[str, float]]:
    """Convert a supplied weight composition to formula moles and trace x."""
    moles: dict[str, float] = {}
    for oxide, raw_amount in composition_wt_pct.items():
        rule = _TRACE_COMPONENT_RULES.get(oxide)
        molar_mass = (
            rule.molar_mass_g_mol
            if rule is not None
            else MOLAR_MASS_G_MOL.get(oxide)
        )
        if molar_mass is None:
            continue
        amount = float(raw_amount)
        if not math.isfinite(amount) or amount < 0.0:
            if rule is not None:
                raise ValueError(
                    f"composition weight for {oxide} must be finite and nonnegative"
                )
            continue
        moles[oxide] = amount / molar_mass
    total = math.fsum(moles.values())
    if total <= 0.0:
        return MappingProxyType(moles), MappingProxyType({})
    fractions = {
        oxide: amount / total
        for oxide, amount in moles.items()
        if oxide in _TRACE_COMPONENT_RULES and amount > 0.0
    }
    return MappingProxyType(moles), MappingProxyType(fractions)


def resolve_trace_activity(
    oxide: str,
    x: float,
    composition_moles: Mapping[str, float],
    T_K: float,
    *,
    fO2_bar: float | None = None,
) -> TraceActivityEstimate:
    estimate, _redox_result, _redox_state = _resolve_trace_activity(
        oxide, x, composition_moles, T_K, fO2_bar=fO2_bar
    )
    return estimate


def _resolve_trace_activity(
    oxide: str,
    x: float,
    composition_moles: Mapping[str, float],
    T_K: float,
    *,
    fO2_bar: float | None = None,
) -> tuple[TraceActivityEstimate, Any | None, Any | None]:
    rule = _TRACE_COMPONENT_RULES[oxide]
    redox_result: Any | None = None
    redox_state: Any | None = None
    if rule.activity_model == "redox_associates":
        try:
            redox_result, redox_state = _redox_trace_result(
                oxide, composition_moles, T_K, fO2_bar=fO2_bar
            )
        except ImccRefusal:
            if rule.gamma is None:
                raise
        else:
            extrapolated_rows = tuple(redox_result.extrapolated_redox_rows)
            reference = None
            code = "source_grounded_associate_activity"
            if extrapolated_rows:
                code = "depends_associate_activity"
                reference = (
                    "required redox rows extrapolated: "
                    + ", ".join(extrapolated_rows)
                    + "; redox flags: "
                    + "; ".join(redox_result.flags)
                )
            return (
                TraceActivityEstimate(
                    oxide,
                    float(redox_result.parent_activity[oxide]),
                    x,
                    code,
                    reference=reference,
                ),
                redox_result,
                redox_state,
            )
    if rule.activity_model == "activity_fragment" and rule.activity_fragment is not None:
        helper_result = _activity_fragment_result(
            rule.activity_fragment, composition_moles, T_K, fO2_bar=fO2_bar
        )
        if helper_result is not None:
            activity, source_is_valid, reference = helper_result
            return (
                TraceActivityEstimate(
                    oxide,
                    activity,
                    x,
                    (
                        "source_grounded_helper_activity"
                        if source_is_valid
                        else "depends_helper_activity"
                    ),
                    reference=reference,
                ),
                None,
                None,
            )
    if rule.activity_model == "review_gamma" and rule.review_gamma_element is not None:
        from openimcc.review_gamma_activity import evaluate_review_gamma_activity

        result = evaluate_review_gamma_activity(
            rule.review_gamma_element,
            composition_moles,
            T_K,
            parent_oxide=rule.gas_parent_oxide or oxide,
            fO2_bar=fO2_bar,
        )
        return (
            TraceActivityEstimate(
                oxide,
                result.activity,
                x,
                result.notice.code,
                result.gamma,
                result.reference,
                result.activity_range,
                result.source_mole_fraction,
                violated_axes=result.notice.violated_axes,
                limitations=result.notice.limitations,
            ),
            None,
            None,
        )
    if rule.gamma is not None:
        return (
            TraceActivityEstimate(
                oxide, rule.gamma * x, x, "published_gamma_x_assumption",
                rule.gamma, rule.gamma_reference,
            ),
            None,
            None,
        )
    return TraceActivityEstimate(oxide, x, x, "ideal_mixing_assumption"), None, None


def _activity_fragment_result(
    element: str,
    composition_moles: Mapping[str, float],
    T_K: float,
    *,
    fO2_bar: float | None,
) -> tuple[float, bool, str | None] | None:
    invalid_oxygen = False

    def relative_oxygen_offset() -> float | None:
        nonlocal invalid_oxygen
        if fO2_bar is None:
            return None
        oxygen = float(fO2_bar)
        if not math.isfinite(oxygen) or oxygen <= 0.0:
            invalid_oxygen = True
            return None
        return math.log10(oxygen) - _iw_log10_fO2_bar(T_K)

    def declared_composition_reference(
        supported_components: frozenset[str],
        violated_axes: tuple[str, ...] = (),
    ) -> tuple[bool, str | None]:
        from openimcc.activity_fragments import load_activity_fragment
        from openimcc.review_gamma_activity import (
            _canonicalize_composition_moles,
            _composition_window_violations,
            _select_row,
        )

        row = _select_row(
            load_activity_fragment(element)["rows"], T_K, composition_moles
        )
        declared_window_flags = _composition_window_violations(
            row, _canonicalize_composition_moles(composition_moles)
        )
        source_checks = tuple(dict.fromkeys(
            (*declared_window_flags, *violated_axes)
        ))
        details = []
        unsupported = [
            name for name in composition_moles if name not in supported_components
        ]
        if unsupported:
            details.append(
                "unsupported components omitted from helper value: "
                + ", ".join(unsupported)
            )
        if source_checks:
            details.append(
                "declared source checks: " + ", ".join(source_checks)
            )
        if not details:
            return True, None
        return (
            not unsupported and not source_checks,
            "declared composition checks: " + "; ".join(details),
        )

    if element == "Ni":
        from openimcc.ni import _SUPPORTED_COMPONENTS, evaluate_ni_activity

        helper_composition = {
            name: amount
            for name, amount in composition_moles.items()
            if name in _SUPPORTED_COMPONENTS
        }
        try:
            result = evaluate_ni_activity(
                helper_composition, T_K, relative_oxygen_offset()
            )
        except ValueError:
            return None
        violated_axes = result.notice.violated_axes
        if invalid_oxygen:
            violated_axes = (
                *violated_axes,
                "oxygen_fugacity_not_finite_or_positive",
            )
        source_is_valid, reference = declared_composition_reference(
            _SUPPORTED_COMPONENTS, violated_axes
        )
        return (
            result.activity,
            source_is_valid,
            reference,
        )
    if element == "Co":
        from openimcc.co_activity import _OXIDE_COMPONENTS, evaluate_co_activity

        helper_composition = {
            name: amount
            for name, amount in composition_moles.items()
            if name in _OXIDE_COMPONENTS
        }
        try:
            result = evaluate_co_activity(
                helper_composition,
                T_K,
                log10_fO2_over_IW=relative_oxygen_offset(),
            )
        except ValueError:
            return None
        violated_axes = result.notice.violated_axes
        if invalid_oxygen:
            violated_axes = (
                *violated_axes,
                "oxygen_fugacity_not_finite_or_positive",
            )
        source_is_valid, reference = declared_composition_reference(
            _OXIDE_COMPONENTS, violated_axes
        )
        return (
            result.activity,
            source_is_valid,
            reference,
        )
    if element == "P":
        from openimcc.p2o5 import _SUPPORTED_COMPONENTS, evaluate_p2o5_activity

        helper_composition = {
            name: amount
            for name, amount in composition_moles.items()
            if name in _SUPPORTED_COMPONENTS
        }
        try:
            result = evaluate_p2o5_activity(helper_composition, T_K)
        except ValueError:
            return None
        violated_axes = tuple(result.notice.violated_axes) + tuple(
            notice.code for notice in result.additional_notices
        )
        reference = (
            "declared chemistry checks: " + "; ".join(violated_axes)
            if violated_axes
            else None
        )
        return result.activity, not violated_axes, reference
    if element == "S":
        from openimcc.sulfur import _OXIDES, evaluate_sulfur

        sulfur_moles = composition_moles.get("S", 0.0)
        sulfur_composition = {
            name: amount * MOLAR_MASS_G_MOL[name]
            for name, amount in composition_moles.items()
            if name in _OXIDES and amount > 0.0
        }
        total_mass = math.fsum(sulfur_composition.values())
        if sulfur_moles <= 0.0 or total_mass <= 0.0:
            return None
        log_fO2 = None if fO2_bar is None else math.log10(fO2_bar)
        try:
            result = evaluate_sulfur(
                sulfur_composition,
                T_K,
                log10_fO2=log_fO2,
                total_s_ppm=(
                    sulfur_moles * MOLAR_MASS_G_MOL["S"] / total_mass * 1.0e6
                ),
            )
        except (ValueError, ImccRefusal, ArithmeticError):
            return None
        if result.log10_fS2 is None:
            return None
        calibration_notices = tuple(
            notice
            for notice in result.notices
            if notice.code.startswith("capacity-")
            and "outside-calibration" in notice.code
        )
        reference = (
            "sulfur capacity calibration notices: "
            + "; ".join(
                f"{notice.code} ({notice.axis})" for notice in calibration_notices
            )
            if calibration_notices
            else None
        )
        return (
            10.0**result.log10_fS2,
            not calibration_notices,
            reference,
        )
    return None


def _redox_trace_result(
    oxide: str,
    composition_moles: Mapping[str, float],
    T_K: float,
    *,
    fO2_bar: float | None,
) -> tuple[Any, Any]:
    from openimcc.model import load_datapack
    from openimcc.redox import _evaluate_redox_with_packs
    from openimcc.redox_pack import load_redox_pack

    redox_pack = load_redox_pack()
    formulas = {
        str(parent["parent_oxide"]).removesuffix("(l)"):
        dict(parent["parent_formula_atoms"])
        for parent in redox_pack.parents
    }
    inventory: dict[str, float] = {}
    for parent, amount in composition_moles.items():
        atoms = formulas.get(parent)
        if atoms is None:
            continue
        for element, count in atoms.items():
            inventory[element] = inventory.get(element, 0.0) + amount * float(count)
    if fO2_bar is None:
        mode = "closed"
        lambda_imposed = None
    else:
        if not math.isfinite(fO2_bar) or fO2_bar <= 0.0:
            raise ValueError("fO2_bar must be finite and positive")
        mode = "imposed"
        lambda_imposed = math.log(fO2_bar)
    result, state = _evaluate_redox_with_packs(
        inventory,
        T_K,
        mode,
        lambda_imposed=lambda_imposed,
        melt_pack=load_datapack(),
        loaded_redox_pack=redox_pack,
    )
    if oxide not in result.parent_activity:
        raise ValueError(f"redox result has no modelled {oxide} activity")
    return result, state
