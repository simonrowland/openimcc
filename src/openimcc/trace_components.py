"""Trace-component rules shared by melt and gas evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Literal

from openimcc._constants import MOLAR_MASS_G_MOL
from openimcc.kernel import ImccRefusal
from openimcc.trace_envelope_calibration import (
    AMPHOTERIC_ENVELOPE_K,
    BASIC_MONOXIDE_ENVELOPE_K,
)


_SOSSI_FEGLY_2018 = (
    "Sossi & Fegley (2018), Rev. Mineral. Geochem. 84, Table 2, pp. 409–410, "
    "doi:10.2138/rmg.2018.84.11"
)


@dataclass(frozen=True)
class TraceComponentRule:
    molar_mass_g_mol: float
    nu: int
    nu_reason: str
    activity_model: Literal["redox_associates", "activity_fragment", "ideal"]
    trace_class: Literal[
        "basic_monoxide", "basic_oxide", "acidic", "amphoteric", "non_oxide"
    ]
    trace_class_reason: str
    gas_parent_oxide: str | None = None
    gamma: float | None = None
    gamma_species: str | None = None
    gamma_reference: str | None = None
    full_component: bool = False
    activity_fragment: Literal["Ni", "Co", "P", "S"] | None = None


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
            gas_parent_oxide="NiO",
        ),
        "CoO": TraceComponentRule(
            MOLAR_MASS_G_MOL["CoO"], 1,
            "one monoxide unit can bind at most one major-oxide unit", "activity_fragment",
            trace_class="basic_monoxide",
            trace_class_reason="CoO is a basic divalent monoxide",
            activity_fragment="Co",
            gas_parent_oxide="CoO",
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
        ),
        "S": TraceComponentRule(
            MOLAR_MASS_G_MOL["S"], 1,
            "one sulfur component unit can bind at most one major-oxide unit", "activity_fragment",
            trace_class="non_oxide",
            trace_class_reason="S is an elemental trace component, not an oxide",
            activity_fragment="S",
            gas_parent_oxide="S2",
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
        ),
        "F": TraceComponentRule(
            MOLAR_MASS_G_MOL["F"], 1,
            "one halogen component unit can bind at most one major-oxide unit", "ideal",
            trace_class="non_oxide",
            trace_class_reason="F is an elemental trace component, not an oxide",
            gas_parent_oxide="F2",
        ),
        "Yb2O3": TraceComponentRule(
            2.0 * MOLAR_MASS_G_MOL["Yb"] + 3.0 * _O_MASS, 1,
            "one rare-earth oxide unit can bind at most one major-oxide unit", "redox_associates",
            trace_class="basic_oxide",
            trace_class_reason="Yb2O3 is a basic rare-earth oxide but not a monoxide formula",
            gas_parent_oxide="Yb2O3",
        ),
    }
)

_TRACE_ENVELOPE_CAP_DEX = 0.02
_TRACE_STRICT_BOUND_CAP_DEX = 0.01


@dataclass(frozen=True)
class TraceActivityEstimate:
    oxide: str
    activity: float
    x: float
    code: str
    gamma: float | None = None
    reference: str | None = None


class TraceActivityFlag(str):
    """String-compatible typed flag identifying the activity-ladder rung."""

    code: str
    oxide: str
    activity: float
    x: float
    gamma: float | None
    reference: str | None
    estimates: tuple[TraceActivityEstimate, ...]
    major_flags: tuple["TraceMajorBiasFlag", ...]

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
            if item.reference:
                message += f"; {item.reference}"
            estimate_text.append(message)
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
        flag.estimates = tuple(estimates)
        flag.major_flags = tuple(major_flags)
        return flag

    def __reduce__(self):
        return type(self), (self.estimates,), {
            "major_flags": self.major_flags,
        }


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
        envelope_text = tuple(
            f'{name} {{k: {k:.12g}, calibrant: "{calibrant}", class: "{trace_class}"}}'
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
        flag.mn_calibrated_envelope = next(
            (
                MappingProxyType({
                    "k": k, "calibrant": calibrant, "class": trace_class,
                })
                for name, k, calibrant, trace_class in envelope_flags
                if name == "mn_calibrated_envelope"
            ),
            None,
        )
        flag.cr_calibrated_envelope = next(
            (
                MappingProxyType({
                    "k": k, "calibrant": calibrant, "class": trace_class,
                })
                for name, k, calibrant, trace_class in envelope_flags
                if name == "cr_calibrated_envelope"
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


def _activity_fragment_registered(element: str) -> bool:
    """Return whether the trace activity table registers this helper fragment."""
    return any(
        rule.activity_fragment == element
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
            return (
                TraceActivityEstimate(
                    oxide,
                    float(redox_result.parent_activity[oxide]),
                    x,
                    "source_grounded_associate_activity",
                ),
                redox_result,
                redox_state,
            )
    if rule.activity_model == "activity_fragment" and rule.activity_fragment is not None:
        helper_result = _activity_fragment_result(
            rule.activity_fragment, composition_moles, T_K, fO2_bar=fO2_bar
        )
        if helper_result is not None:
            activity, source_is_valid = helper_result
            if source_is_valid:
                return (
                    TraceActivityEstimate(
                        oxide, activity, x, "source_grounded_helper_activity"
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
) -> tuple[float, bool] | None:
    if element == "Ni":
        from openimcc.ni import _SUPPORTED_COMPONENTS, evaluate_ni_activity

        helper_composition = {
            name: amount
            for name, amount in composition_moles.items()
            if name in _SUPPORTED_COMPONENTS
        }
        try:
            result = evaluate_ni_activity(helper_composition, T_K)
        except ValueError:
            return None
        return result.activity, True
    if element == "Co":
        from openimcc.co_activity import _OXIDE_COMPONENTS, evaluate_co_activity

        helper_composition = {
            name: amount
            for name, amount in composition_moles.items()
            if name in _OXIDE_COMPONENTS
        }
        try:
            result = evaluate_co_activity(helper_composition, T_K)
        except ValueError:
            return None
        return result.activity, True
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
        return result.activity, not result.notice.violated_axes
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
        return 10.0**result.log10_fS2, True
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
