"""Evaluate the shared, source-railed review-gamma activity rows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache, partial
import math

from openimcc._constants import MOLAR_MASS_G_MOL, R_J_MOL_K
from openimcc._numeric_guards import (
    _checked_finite as _numeric_checked_finite,
    _checked_fsum as _numeric_checked_fsum,
    _checked_positive_finite as _numeric_checked_positive_finite,
    _checked_pow10 as _numeric_checked_pow10,
)
from openimcc.activity_fragments import load_activity_fragment
from openimcc.kernel import ImccComponentOutsideDomainError
from openimcc.trace_envelope_calibration import _iw_log10_fO2_bar
from openimcc.trace_components import (
    _TRACE_COMPONENT_RULES,
    _review_gamma_parent_oxides,
    _single_cation_oxide_counts,
)

_LOG10_BAR_PER_ATM = math.log10(1.01325)
_TRACE_PARENTS = _review_gamma_parent_oxides(_TRACE_COMPONENT_RULES)
_NON_OXIDE_COMPONENTS = frozenset({"S", "Cl", "F", "Br", "I", "C", "H", "N"})


class ReviewGammaActivityRefusal(ImccComponentOutsideDomainError):
    """Typed refusal that keeps the selected source row and failed axes."""

    def __init__(
        self,
        row_id: str,
        violated_axes: tuple[str, ...],
        reason: str,
        *,
        code: str,
    ) -> None:
        self.code = code
        self.row_id = row_id
        self.violated_axes = violated_axes
        self.reason = reason
        super().__init__(
            f"{code}: review-gamma row {row_id} refused; "
            f"violated axes: {', '.join(violated_axes)}; {reason}"
        )


def _oxide_cation_units(component: str) -> int:
    """Return the cations in a single-cation oxide formula unit."""
    oxide_counts = _single_cation_oxide_counts(component)
    return oxide_counts[0] if oxide_counts is not None else 1


@lru_cache(maxsize=1)
def _packaged_ce2o3_liquid_rows():
    """Return the shipped Ce2O3(l) Gibbs rows used for standard conversion."""
    from openimcc.gas import _PACKAGED_CONDENSATE_NAME, _read_packaged_table

    table, _path = _read_packaged_table(_PACKAGED_CONDENSATE_NAME)
    return table.loc[["Ce2O3(l)"]]


@dataclass(frozen=True)
class ReviewGammaNotice:
    code: str
    violated_axes: tuple[str, ...]
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class ReviewGammaActivityResult:
    activity: float
    source_activity: float
    gamma: float
    gamma_range: tuple[float, float]
    source_mole_fraction: float
    activity_range: tuple[float, float]
    parent_oxide: str
    row: str
    notice: ReviewGammaNotice
    reference: str


def evaluate_review_gamma_activity(
    element: str,
    composition_moles: Mapping[str, float],
    T_K: float,
    *,
    parent_oxide: str,
    fO2_bar: float | None = None,
) -> ReviewGammaActivityResult:
    """Return ``a(parent) = (gamma * x_source)**k`` and its endpoint band.

    Bare ranges use the geometric centre; a source-printed central value is
    used when provided. Printed endpoints propagate through the activity
    conversion. Temperature picks the nearest source row. Declared composition
    screens and windows are reported as notice flags; composition-dependent
    law inputs are checked against their declared source ranges. An-Di targets
    compare declared major oxides after renormalizing those majors alone.
    """
    return _evaluate_review_gamma_activity(
        element,
        composition_moles,
        T_K,
        parent_oxide=parent_oxide,
        fO2_bar=fO2_bar,
    )


def _evaluate_review_gamma_activity(
    element: str,
    composition_moles: Mapping[str, float],
    T_K: float,
    *,
    parent_oxide: str,
    fO2_bar: float | None = None,
    log10_fO2_relative_to_IW: float | None = None,
) -> ReviewGammaActivityResult:
    temperature = float(T_K)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")

    if fO2_bar is not None and log10_fO2_relative_to_IW is not None:
        raise ValueError("supply either fO2_bar or log10_fO2_relative_to_IW")
    validated_amounts: dict[str, float] = {}
    input_components = tuple(composition_moles)
    for component, raw_amount in composition_moles.items():
        amount = float(raw_amount)
        if not math.isfinite(amount) or amount < 0.0:
            raise ValueError(f"moles for {component} must be finite and nonnegative")
        validated_amounts[component] = amount
    amounts = _canonicalize_composition_moles(validated_amounts)

    if log10_fO2_relative_to_IW is not None:
        oxygen_offset = float(log10_fO2_relative_to_IW)
        if not math.isfinite(oxygen_offset):
            raise ValueError("log10_fO2_relative_to_IW must be finite")
        log_fO2_bar = None
    elif fO2_bar is None:
        oxygen_offset = None
        log_fO2_bar = None
    else:
        oxygen_pressure = float(fO2_bar)
        if not math.isfinite(oxygen_pressure) or oxygen_pressure <= 0.0:
            raise ValueError("fO2_bar must be finite and positive when supplied")
        log_fO2_bar = math.log10(oxygen_pressure)
        oxygen_offset = None

    rows = load_activity_fragment(element)["rows"]
    candidates = [row for row in rows if row.get("parent_oxide") == parent_oxide]
    if not candidates:
        raise ValueError(
            f"no review-gamma row converts to gas parent {parent_oxide!r} for {element!r}"
        )
    row, selector_no_match = _select_row_with_status(
        candidates, temperature, amounts
    )
    screen = row["composition_screen"]
    row_id = str(row["row"])

    if screen["kind"] == "supported_component_set":
        supported = frozenset(str(name) for name in screen["components"])
        for component in input_components:
            if component not in supported:
                raise ValueError(f"unsupported oxide component {component!r}")

    parent_moles = amounts.get(parent_oxide, 0.0)
    total_moles = _checked_fsum(amounts.values(), row_id, "composition_total")
    if parent_moles <= 0.0 or total_moles <= 0.0:
        raise ValueError(
            f"composition must contain positive total moles and {parent_oxide}"
        )

    source_units = float(row["source_units_per_parent"])
    parent_power = float(row["parent_power"])
    if not math.isfinite(source_units) or source_units <= 0.0:
        raise ValueError("source_units_per_parent must be finite and positive")
    if not math.isfinite(parent_power) or parent_power <= 0.0:
        raise ValueError("parent_power must be finite and positive")
    if row.get("source_mole_fraction_kind") == "simple_oxide":
        source_x = parent_moles / total_moles
    elif row.get("source_mole_fraction_kind") == "one_cation_oxide":
        # Premise: Norris and Sossi define x_source on one-cation oxide units;
        # every solvent M_aO_b formula contributes a cation count of a, while
        # the M2O_n parent contributes its declared source_units (two here).
        # Algebra: x_source = source_units*n_parent /
        # sum_i(cations_i*n_i), replacing the parent's formula count by its
        # source-unit count. The ratio is dimensionless (mol cations/mol
        # cations); a(parent)=(gamma*x_source)^k with k=2.
        # Sanity: V2O3=0.001, SiO2=2, Al2O3=FeO=MgO=CaO=1 gives
        # x_source=0.002/7.002 and, versus the old 0.002/6.002 denominator,
        # an old/new activity ratio (7.002/6.002)^2 ~= 1.36.
        solvent_basis_total = _checked_fsum(
            (
                amount * _oxide_cation_units(component)
                for component, amount in amounts.items()
                if component != parent_oxide
            ),
            row_id,
            "source_basis_total",
        )
        source_basis_total = _checked_finite(
            solvent_basis_total + source_units * parent_moles,
            row_id,
            "source_basis_total",
        )
        source_x = source_units * parent_moles / source_basis_total
    else:
        # Bischof's Table 8 uses MO1.5 component units with its current
        # parent-only expansion: 1 mol M2O3 + 1 mol SiO2 gives x=2/3 and,
        # for gamma=1, a(M2O3)=4/9.
        source_basis_total = _checked_finite(
            total_moles - parent_moles + source_units * parent_moles,
            row_id,
            "source_basis_total",
        )
        source_x = source_units * parent_moles / source_basis_total

    gamma, gamma_range, ln_gamma = _gamma_for_row(
        row, temperature, amounts
    )
    conversion = row.get("fusion_conversion")
    if conversion is not None:
        transitions = conversion["transitions"]
        source_solid = conversion["source_solid_gibbs"]
        reference_temperature = float(source_solid["reference_temperature_K"])
        enthalpy_increment = 0.0
        entropy_increment = 0.0
        for coefficient, power in zip(
            source_solid["heat_capacity_coefficients"],
            source_solid["heat_capacity_powers"],
            strict=True,
        ):
            coefficient = float(coefficient)
            power = int(power)
            try:
                if power == -1:
                    enthalpy_increment += coefficient * math.log(
                        temperature / reference_temperature
                    )
                else:
                    enthalpy_increment += coefficient * (
                        temperature ** (power + 1)
                        - reference_temperature ** (power + 1)
                    ) / (power + 1)
                if power == 0:
                    entropy_increment += coefficient * math.log(
                        temperature / reference_temperature
                    )
                else:
                    entropy_increment += coefficient * (
                        temperature**power - reference_temperature**power
                    ) / power
            except (ArithmeticError, ValueError) as exc:
                raise _numeric_refusal(
                    row_id,
                    "gamma_standard_conversion",
                    f"Ce2O3 fusion conversion failed: {exc}",
                ) from exc
            _checked_finite(
                enthalpy_increment, row_id, "gamma_standard_conversion_enthalpy"
            )
            _checked_finite(
                entropy_increment, row_id, "gamma_standard_conversion_entropy"
            )
        g_stable_solid = (
            float(source_solid["delta_Hf_298_J_mol"])
            + enthalpy_increment
            - temperature
            * (float(source_solid["S_298_J_mol_K"]) + entropy_increment)
        )
        _checked_finite(g_stable_solid, row_id, "gamma_standard_conversion_gibbs")

        from openimcc.gas import _lamor_gibbs, _nearest_interval_row

        liquid_row = _nearest_interval_row(
            _packaged_ce2o3_liquid_rows(), "Ce2O3(l)", temperature, True
        )
        try:
            g_parent_liquid = _lamor_gibbs(temperature, liquid_row)
        except (ArithmeticError, ValueError) as exc:
            raise _numeric_refusal(
                row_id,
                "gamma_standard_conversion_gibbs",
                f"Ce2O3 liquid Gibbs conversion failed: {exc}",
            ) from exc
        _checked_finite(g_parent_liquid, row_id, "gamma_standard_conversion_gibbs")
        delta_g = g_parent_liquid - g_stable_solid
        _checked_finite(delta_g, row_id, "gamma_standard_conversion_gibbs")
        transition_variance = 0.0
        for transition in transitions:
            transition_temperature = float(transition["T_K"])
            delta_h_uncertainty = float(transition["delta_H_uncertainty_J_mol"])
            try:
                transition_variance += (
                    delta_h_uncertainty
                    * (1.0 - temperature / transition_temperature)
                ) ** 2
            except (ArithmeticError, ValueError) as exc:
                raise _numeric_refusal(
                    row_id,
                    "gamma_standard_conversion_uncertainty",
                    f"Ce2O3 transition uncertainty failed: {exc}",
                ) from exc
            _checked_finite(
                transition_variance,
                row_id,
                "gamma_standard_conversion_uncertainty",
            )
        # Premise: this source activity a_A is on the pure stable A-phase
        # standard, while the gas parent activity a_l is on pure Ce2O3(l).
        # At equilibrium μ°_A + RT ln(a_A) = μ°_l + RT ln(a_l), hence
        # a_l = a_A exp(-(G°_l(T)-G°_A(T))/(RT)). G_A uses the shipped Konings
        # et al. (2014) A-phase Cp function; G_l uses the shipped Ce2O3(l) row.
        # ΔG/(R T) has units (J/mol)/(J mol^-1 K^-1 × K) = 1.
        # Sanity: the residual beyond the transition sum is 1533, 843 and
        # 199 J/mol at 1700, 1873 and 2100 K; total ΔG is 42684, 32749 and
        # 19974 J/mol at those temperatures. The band keeps the transition
        # enthalpy uncertainties; their derivative is σ_H(1-T/T_tr).
        try:
            conversion_log_factor = -delta_g / (R_J_MOL_K * temperature)
        except (ArithmeticError, ValueError) as exc:
            raise _numeric_refusal(
                row_id,
                "gamma_standard_conversion",
                f"Ce2O3 standard conversion failed: {exc}",
            ) from exc
        _checked_finite(conversion_log_factor, row_id, "ln_gamma_standard_conversion")
        factor = _checked_exp(
            conversion_log_factor, row_id, "gamma_standard_conversion"
        )
        gamma *= factor
        gamma = _checked_positive_finite(gamma, row_id, "gamma")
        ln_gamma += conversion_log_factor
        _checked_finite(ln_gamma, row_id, "ln_gamma")
        closure_rms = float(row["gamma_law"]["ln_gamma_rms_closure"])
        transition_log_uncertainty = math.sqrt(transition_variance) / (
            R_J_MOL_K * temperature
        )
        _checked_finite(
            transition_log_uncertainty,
            row_id,
            "gamma_standard_conversion_uncertainty",
        )
        log_half_width = math.hypot(closure_rms, transition_log_uncertainty)
        _checked_finite(log_half_width, row_id, "ln_gamma_range")
        gamma_range = (
            gamma * _checked_exp(-log_half_width, row_id, "gamma_range"),
            gamma * _checked_exp(log_half_width, row_id, "gamma_range"),
        )
        _check_gamma_range(gamma_range, row_id)

    _checked_positive_finite(source_x, row_id, "source_mole_fraction")
    log_source_x = math.log(source_x)
    ln_source_activity = ln_gamma + log_source_x
    _checked_finite(ln_source_activity, row_id, "ln_source_activity")
    ln_activity = ln_source_activity * parent_power
    _checked_finite(ln_activity, row_id, "ln_activity")
    ln_activity_range = tuple(
        (math.log(gamma_endpoint) + log_source_x) * parent_power
        for gamma_endpoint in gamma_range
    )
    for endpoint in ln_activity_range:
        _checked_finite(endpoint, row_id, "ln_activity_range")

    source_activity = _checked_positive_finite(
        gamma * source_x, row_id, "source_activity"
    )
    try:
        activity = source_activity**parent_power
        activity_range = tuple(
            (gamma_endpoint * source_x) ** parent_power
            for gamma_endpoint in gamma_range
        )
    except (OverflowError, ValueError) as exc:
        raise _numeric_refusal(
            row_id, "activity_overflow", f"activity conversion failed: {exc}"
        ) from exc
    activity = _checked_positive_finite(activity, row_id, "activity")
    _check_activity_range(activity_range, row_id)

    violated: list[str] = []
    distances: list[str] = []
    t_low, t_high = map(float, row["temperature_window_K"])
    t_distance = max(t_low - temperature, 0.0, temperature - t_high)
    if t_distance > 0.0:
        violated.append("temperature_extrapolated")
        distances.append(f"temperature distance {t_distance:.6g} K")

    if screen["kind"] == "family_oxide_set":
        family_distance, missing_family_members = _family_score(
            screen, amounts, parent_oxide=parent_oxide
        )
        if family_distance > 0.0 or missing_family_members > 0:
            violated.append("melt_family_extrapolated")
            distances.append(
                f"melt-family distance {family_distance:.6g} oxide mole fraction "
                "(source-family-distance-v2)"
            )
            if missing_family_members:
                distances.append(
                    f"{missing_family_members} required family oxides absent"
                )
    elif screen["kind"] == "an_di_target":
        if not _matches_an_di_target(screen, amounts):
            violated.append("melt_composition_differs_from_An_Di_target")
    elif screen["kind"] != "supported_component_set":
        raise ValueError(f"unknown composition screen {screen['kind']!r}")

    if selector_no_match:
        violated.append("review_gamma_selector_no_match")

    oxygen_window = row.get("log10_fO2_source")
    if oxygen_window is None:
        violated.append("oxygen_fugacity_window_unreported")
    elif oxygen_offset is None and log_fO2_bar is None:
        violated.append("oxygen_fugacity_window_unchecked")
    else:
        if oxygen_offset is not None:
            if oxygen_window["unit"] != "delta_iw":
                raise ValueError(
                    "log10_fO2_relative_to_IW requires a delta_iw source window"
                )
            oxygen_low, oxygen_high = map(float, oxygen_window["range"])
            oxygen_value = oxygen_offset
        elif oxygen_window["unit"] == "delta_iw":
            oxygen_low, oxygen_high = map(float, oxygen_window["range"])
            oxygen_value = log_fO2_bar - _iw_log10_fO2_bar(temperature)
        else:
            oxygen_low, oxygen_high = map(float, oxygen_window["range"])
            if oxygen_window["unit"] == "atm":
                oxygen_low += _LOG10_BAR_PER_ATM
                oxygen_high += _LOG10_BAR_PER_ATM
            oxygen_value = log_fO2_bar
        oxygen_distance = max(
            oxygen_low - oxygen_value, 0.0, oxygen_value - oxygen_high
        )
        if oxygen_distance > 0.0:
            violated.append("oxygen_fugacity_extrapolated")
            distances.append(f"oxygen-fugacity distance {oxygen_distance:.6g} dex")

    violated.extend(_composition_window_violations(row, amounts))

    flag_labels = row.get("flag_labels", {})
    violated = [str(flag_labels.get(flag, flag)) for flag in violated]

    reference = (
        f"{row['source_citation']}; {row['source_locator']}; "
        f"{row['component']} gamma endpoints {gamma_range[0]:g}–{gamma_range[1]:g}; "
        f"point {gamma:g}; {row['component_basis']}; "
        f"a({parent_oxide})=(gamma*x_source)^{parent_power:g}; "
        f"{row['parent_standard_state']}"
    )
    if row["gamma_law"]["kind"] == "range":
        reference += "; nearest-temperature source row"
        if "gamma_point" in row["gamma_law"]:
            reference += "; gamma-range-printed-point-v1"
    if screen["kind"] == "an_di_target":
        reference += "; an-di-target-basis=major-oxides-only-v2"
    if conversion is not None:
        reference += "; ce2o3-liquid-standard-gibbs-v3"
    selection_note = row.get("selection_note")
    if selection_note:
        reference += f"; {selection_note}"
    source_x_anchor = row.get("source_x_anchor")
    if source_x_anchor is not None:
        reference += (
            f"; published x_source anchor {float(source_x_anchor['x_source']):g}"
            f" at {float(source_x_anchor['concentration_ppm']):g} ppm"
        )
    limitations = tuple(str(item) for item in row["limitations"])
    if limitations:
        reference += "; limitation: " + " ".join(limitations)
    if distances:
        reference += "; extrapolated: " + ", ".join(distances)
    if "melt_composition_differs_from_An_Di_target" in violated:
        reference += "; out of domain: melt_composition_differs_from_An_Di_target"
    if "oxygen_fugacity_window_unchecked" in violated:
        reference += "; oxygen-fugacity window unchecked"
    if "oxygen_fugacity_window_unreported" in violated:
        reference += "; oxygen-fugacity window not reported"
    if selector_no_match:
        reference += (
            "; DEPENDS: review_gamma_selector_no_match; nearest row shown "
            "with this flag"
        )

    return ReviewGammaActivityResult(
        activity=activity,
        source_activity=source_activity,
        gamma=gamma,
        gamma_range=gamma_range,
        source_mole_fraction=source_x,
        activity_range=activity_range,
        parent_oxide=parent_oxide,
        row=str(row["row"]),
        notice=ReviewGammaNotice(
            code=str(row.get("notice_code", "review_gamma_provisional")),
            violated_axes=tuple(violated),
            limitations=limitations,
        ),
        reference=reference,
    )


def _select_row(
    candidates: list[Mapping[str, object]],
    temperature: float,
    amounts: Mapping[str, float],
) -> Mapping[str, object]:
    row, _selector_no_match = _select_row_with_status(
        candidates, temperature, amounts
    )
    return row


def _select_row_with_status(
    candidates: list[Mapping[str, object]],
    temperature: float,
    amounts: Mapping[str, float],
) -> tuple[Mapping[str, object], bool]:
    def temperature_distance(row: Mapping[str, object]) -> float:
        source_temperature = row["gamma_law"].get(
            "temperature_K", row["temperature_window_K"]
        )
        low, high = map(float, source_temperature)
        return max(low - temperature, 0.0, temperature - high)

    nearest_distance = min(temperature_distance(row) for row in candidates)
    nearest = [
        row for row in candidates if temperature_distance(row) == nearest_distance
    ]
    selector_rows = [
        row
        for row in nearest
        if row.get("select_when_component_zero") is not None
        or row.get("select_when_component_positive") is not None
    ]
    component_selection = []
    for row in nearest:
        zero_component = row.get("select_when_component_zero")
        positive_component = row.get("select_when_component_positive")
        if zero_component is not None and amounts.get(str(zero_component), 0.0) != 0.0:
            continue
        if positive_component is not None and amounts.get(str(positive_component), 0.0) <= 0.0:
            continue
        if zero_component is not None or positive_component is not None:
            component_selection.append(row)
    if component_selection:
        if len(component_selection) != 1:
            raise ValueError("review-gamma component selectors must identify one row")
        return component_selection[0], False
    if selector_rows:
        return nearest[0], True
    if len(nearest) > 1 and any(
        row["composition_screen"].get("select_tie_by_family") for row in nearest
    ):
        return min(
            nearest,
            key=lambda row: _family_score(
                row["composition_screen"],
                amounts,
                parent_oxide=str(row["parent_oxide"]),
            ),
        ), False
    return nearest[0], False


def _family_score(
    screen: Mapping[str, object],
    amounts: Mapping[str, float],
    *,
    parent_oxide: str,
) -> tuple[float, int]:
    family = frozenset(str(name) for name in screen["oxides"])
    # Source rows give solvent-oxide families. Keep other declared trace oxides
    # in the solvent total, but leave non-oxides to their own trace screens.
    oxide_total = math.fsum(
        amount
        for name, amount in amounts.items()
        if name not in _NON_OXIDE_COMPONENTS
    )
    outside = math.fsum(
        amount
        for name, amount in amounts.items()
        if name != parent_oxide
        and name not in family
        and name not in _NON_OXIDE_COMPONENTS
    )
    missing_family_members = sum(amounts.get(name, 0.0) <= 0.0 for name in family)
    distance = outside / oxide_total if oxide_total > 0.0 else 1.0
    if screen.get("require_exact_family") and (
        missing_family_members > 0 or outside > 0.0
    ):
        distance += 1.0
    return distance, missing_family_members


def _matches_an_di_target(
    screen: Mapping[str, object], amounts: Mapping[str, float]
) -> bool:
    target = {str(name): float(value) for name, value in screen["major_wt_pct"].items()}
    for component, amount in amounts.items():
        if amount <= 0.0 or component in target:
            continue
        if component not in _TRACE_PARENTS and component not in _NON_OXIDE_COMPONENTS:
            return False
    masses = {
        component: amounts.get(component, 0.0) * MOLAR_MASS_G_MOL[component]
        for component in target
    }
    # The source target is printed on a major-oxide basis. Normalize only its
    # declared target majors; trace parents and non-oxides are screened apart.
    total_mass = math.fsum(masses.values())
    if total_mass <= 0.0:
        return False
    tolerance = float(screen["tolerance_wt_pct"])
    return all(
        abs(masses[component] / total_mass * 100.0 - expected) <= tolerance
        for component, expected in target.items()
    )


def _composition_window_violations(
    row: Mapping[str, object], amounts: Mapping[str, float]
) -> tuple[str, ...]:
    composition_windows = row.get("composition_windows_wt_pct", {})
    if not composition_windows:
        return ()
    total_mass = math.fsum(
        amount * _component_molar_mass(component)
        for component, amount in amounts.items()
    )
    if total_mass <= 0.0:
        return ()
    violations: list[str] = []
    for component, window in composition_windows.items():
        weight = (
            amounts.get(component, 0.0)
            * _component_molar_mass(component)
            / total_mass
            * 100.0
        )
        low, high = map(float, window["range"])
        if weight < low or weight > high:
            violations.append(str(window["outside_flag"]))
        else:
            above = window.get("above")
            if above is not None and weight > float(above["value"]):
                violations.append(str(above["flag"]))
    return tuple(violations)


def _canonicalize_composition_moles(
    composition_moles: Mapping[str, float],
) -> dict[str, float]:
    amounts: dict[str, float] = {}
    for component, amount in composition_moles.items():
        canonical = "FeO" if component == "FeOeq" else component
        amounts[canonical] = amounts.get(canonical, 0.0) + amount
    return amounts


def _numeric_refusal(
    row_id: str, axis: str, reason: str
) -> ReviewGammaActivityRefusal:
    return ReviewGammaActivityRefusal(
        row_id,
        (axis,),
        reason,
        code="review_gamma_numeric_failure",
    )


def _numeric_guard_refusal(
    row_id: str,
    axis: str,
    failure: str,
    cause: BaseException | None,
    _context: str | None,
) -> ReviewGammaActivityRefusal:
    if failure == "sum_overflow":
        failed_axis = f"{axis}_overflow"
        reason = f"{axis} could not be summed: {cause}"
    elif failure == "nonfinite":
        failed_axis = f"{axis}_nonfinite"
        reason = f"{axis} is not finite"
    elif failure == "not_positive_finite":
        failed_axis = f"{axis}_nonfinite"
        reason = f"{axis} is not a finite positive value"
    else:
        failed_axis = f"{axis}_overflow"
        reason = f"{axis} cannot be represented: {cause}"
    return _numeric_refusal(row_id, failed_axis, reason)


_checked_fsum = partial(_numeric_checked_fsum, refusal=_numeric_guard_refusal)
_checked_finite = partial(_numeric_checked_finite, refusal=_numeric_guard_refusal)
_checked_positive_finite = partial(
    _numeric_checked_positive_finite, refusal=_numeric_guard_refusal,
)


def _checked_exp(log_value: float, row_id: str, axis: str) -> float:
    _checked_finite(log_value, row_id, f"ln_{axis}")
    try:
        value = math.exp(log_value)
    except (OverflowError, ValueError) as exc:
        raise _numeric_refusal(
            row_id, f"{axis}_overflow", f"{axis} cannot be represented: {exc}"
        ) from exc
    return _checked_positive_finite(value, row_id, axis)


_checked_pow10 = partial(_numeric_checked_pow10, refusal=_numeric_guard_refusal)


def _check_gamma_range(
    gamma_range: tuple[float, float], row_id: str
) -> None:
    for value in gamma_range:
        _checked_positive_finite(value, row_id, "gamma_range")


def _check_activity_range(
    activity_range: tuple[float, float], row_id: str
) -> None:
    for value in activity_range:
        _checked_positive_finite(value, row_id, "activity_range")


def _composition_law_input_axes(law: Mapping[str, object]) -> tuple[str, ...]:
    axes: list[str] = []
    for term in law["terms"]:
        name = str(term["name"])
        if name == "wt_CaO_plus_0_72_Al2O3_over_SiO2":
            axes.append(name)
        elif name.startswith("x_"):
            if "_times_x_" in name:
                axes.extend(name.split("_times_"))
            elif name.endswith("_squared"):
                axes.append(name[:-8])
    return tuple(dict.fromkeys(axes))


def _check_composition_law_input_ranges(
    row: Mapping[str, object],
    law: Mapping[str, object],
    input_basis: str,
    input_values: Mapping[str, float],
) -> None:
    row_id = str(row["row"])
    windows = row.get("composition_windows_wt_pct", {})
    for axis in _composition_law_input_axes(law):
        value = input_values.get(axis)
        if value is None:
            raise _numeric_refusal(
                row_id, "gamma_law_input", f"the law input {axis} is unavailable"
            )

        if axis == "wt_CaO_plus_0_72_Al2O3_over_SiO2":
            try:
                cao_low, cao_high = map(float, windows["CaO"]["range"])
                alumina_low, alumina_high = map(float, windows["Al2O3"]["range"])
                silica_low, silica_high = map(float, windows["SiO2"]["range"])
            except (KeyError, TypeError, ValueError) as exc:
                raise _numeric_refusal(
                    row_id,
                    "gamma_law_source_range",
                    f"the declared source windows do not bound {axis}: {exc}",
                ) from exc
            if silica_low <= 0.0:
                raise _numeric_refusal(
                    row_id,
                    "gamma_law_source_range",
                    f"the declared SiO2 source window cannot bound {axis}",
                )
            # The row declares marginal Table 2 windows for CaO, Al2O3, and
            # SiO2. The ratio increases with the first two and decreases with
            # SiO2, so their extrema give a conservative source-data envelope.
            low = (cao_low + 0.72 * alumina_low) / silica_high
            high = (cao_high + 0.72 * alumina_high) / silica_low
        elif input_basis in ("oxide_mole_fraction", "cmas_component_mole_fraction"):
            component = axis.removeprefix("x_")
            if input_basis == "oxide_mole_fraction":
                components = tuple(str(name) for name in windows)
                units_per_formula = {name: 1.0 for name in components}
            else:
                components = ("CaO", "MgO", "Al2O3", "SiO2")
                units_per_formula = {
                    name: float(_oxide_cation_units(name)) for name in components
                }
            if component not in components:
                raise _numeric_refusal(
                    row_id,
                    "gamma_law_source_range",
                    f"the declared source windows do not include {component}",
                )

            mole_ranges: dict[str, tuple[float, float]] = {}
            try:
                for name in components:
                    wt_low, wt_high = map(float, windows[name]["range"])
                    factor = units_per_formula[name] / _component_molar_mass(name)
                    mole_ranges[name] = (wt_low * factor, wt_high * factor)
            except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
                raise _numeric_refusal(
                    row_id,
                    "gamma_law_source_range",
                    f"the declared source windows do not bound {axis}: {exc}",
                ) from exc

            target_low, target_high = mole_ranges[component]
            other_low = math.fsum(
                bounds[0] for name, bounds in mole_ranges.items() if name != component
            )
            other_high = math.fsum(
                bounds[1] for name, bounds in mole_ranges.items() if name != component
            )
            low_denominator = target_low + other_high
            high_denominator = target_high + other_low
            if low_denominator <= 0.0 or high_denominator <= 0.0:
                raise _numeric_refusal(
                    row_id,
                    "gamma_law_source_range",
                    f"the declared source windows cannot bound {axis}",
                )
            low = target_low / low_denominator
            high = target_high / high_denominator
        else:
            raise _numeric_refusal(
                row_id,
                "gamma_law_source_range",
                f"the law input {axis} has no supported source-window basis",
            )

        if not math.isfinite(low) or not math.isfinite(high):
            raise _numeric_refusal(
                row_id,
                "gamma_law_source_range",
                f"the declared source windows produce an invalid range for {axis}",
            )
        if value < low or value > high:
            violated_axis = f"gamma_law_input_{axis}_outside_source_range"
            raise ReviewGammaActivityRefusal(
                row_id,
                (violated_axis,),
                f"{axis}={value:.12g} is outside the source-window range "
                f"[{low:.12g}, {high:.12g}]",
                code="review_gamma_out_of_domain",
            )


def _gamma_for_row(
    row: Mapping[str, object],
    temperature: float,
    amounts: Mapping[str, float],
) -> tuple[float, tuple[float, float], float]:
    row_id = str(row["row"])
    law = row["gamma_law"]
    if law["kind"] == "range":
        try:
            low, high = map(float, law["gamma_endpoints"])
        except (OverflowError, TypeError, ValueError) as exc:
            raise _numeric_refusal(
                row_id, "gamma_range", f"gamma endpoints are invalid: {exc}"
            ) from exc
        if low <= 0.0 or high < low:
            raise _numeric_refusal(
                row_id,
                "gamma_range",
                "gamma endpoints must be positive and ordered",
            )
        try:
            point = float(
                law["gamma_point"]
                if "gamma_point" in law
                else math.sqrt(low * high)
            )
        except (ArithmeticError, TypeError, ValueError) as exc:
            raise _numeric_refusal(
                row_id, "gamma_point", f"gamma point could not be formed: {exc}"
            ) from exc
        if not low <= point <= high:
            raise _numeric_refusal(
                row_id,
                "gamma_point",
                "gamma point must lie within its printed endpoints",
            )
        _checked_positive_finite(point, row_id, "gamma")
        gamma_range = (low, high)
        _check_gamma_range(gamma_range, row_id)
        return point, gamma_range, math.log(point)
    if law["kind"] == "log10_linear":
        try:
            log10_gamma = float(law["A_K"]) / temperature + float(law["B"])
        except (ArithmeticError, ValueError) as exc:
            raise _numeric_refusal(
                row_id, "ln_gamma_overflow", f"log gamma could not be formed: {exc}"
            ) from exc
        _checked_finite(log10_gamma, row_id, "log10_gamma")
        gamma = _checked_pow10(log10_gamma, row_id, "gamma")
        return gamma, (gamma, gamma), math.log(gamma)
    if law["kind"] == "composition_terms":
        input_basis = str(law["input_basis"])
        if input_basis == "oxide_mole_fraction":
            total_moles = _checked_fsum(amounts.values(), row_id, "composition_total")
            x_cao = amounts.get("CaO", 0.0) / total_moles
            x_sio2 = amounts.get("SiO2", 0.0) / total_moles
            law_input_values = {"x_CaO": x_cao, "x_SiO2": x_sio2}
            term_values = {
                "constant": 1.0,
                "x_CaO_squared": x_cao * x_cao,
                "x_SiO2_squared": x_sio2 * x_sio2,
                "x_CaO_times_x_SiO2": x_cao * x_sio2,
            }
        elif input_basis == "cmas_component_mole_fraction":
            # Premise: O'Neill and Eggins Table 8 fits γ in a CMAS component
            # basis; Al2O3 contributes two AlO1.5 units, and TiO2 in the one
            # quinary Table 7 melt is excluded from the CMAS normalization.
            # Algebra: X_i = n_i / Σ(n_j * c_j), where c_j counts oxide
            # cations; the fraction is dimensionless. For AD+TiO2, this gives
            # γ(MoO3) = 0.3273, matching the Table 7 value 0.325(36).
            cmas_moles = _checked_fsum(
                (
                    amount * _oxide_cation_units(component)
                    for component, amount in amounts.items()
                    if component in {"CaO", "MgO", "Al2O3", "SiO2"}
                ),
                row_id,
                "cmas_basis_total",
            )
            x_cao = amounts.get("CaO", 0.0) / cmas_moles if cmas_moles else 0.0
            x_mgo = amounts.get("MgO", 0.0) / cmas_moles if cmas_moles else 0.0
            law_input_values = {"x_CaO": x_cao, "x_MgO": x_mgo}
            term_values = {
                "constant": 1.0,
                "x_CaO_squared": x_cao * x_cao,
                "x_MgO_squared": x_mgo * x_mgo,
                "x_CaO_times_x_MgO": x_cao * x_mgo,
            }
        elif input_basis == "weight_percent":
            cao_mass = amounts.get("CaO", 0.0) * MOLAR_MASS_G_MOL["CaO"]
            sio2_mass = amounts.get("SiO2", 0.0) * MOLAR_MASS_G_MOL["SiO2"]
            al2o3_mass = amounts.get("Al2O3", 0.0) * MOLAR_MASS_G_MOL["Al2O3"]
            if sio2_mass <= 0.0:
                raise ReviewGammaActivityRefusal(
                    row_id,
                    ("SiO2_required_by_gamma_law",),
                    "the selected weight-percent law requires positive SiO2",
                    code="review_gamma_out_of_domain",
                )
            term_values = {
                "constant": 1.0,
                "wt_CaO_plus_0_72_Al2O3_over_SiO2": (
                    (cao_mass + 0.72 * al2o3_mass) / sio2_mass
                ),
            }
            law_input_values = {
                "wt_CaO_plus_0_72_Al2O3_over_SiO2": term_values[
                    "wt_CaO_plus_0_72_Al2O3_over_SiO2"
                ]
            }
        else:
            raise ValueError(f"unknown composition_terms basis {input_basis!r}")

        _check_composition_law_input_ranges(
            row, law, input_basis, law_input_values
        )

        try:
            ln_gamma = math.fsum(
                float(term["coefficient"]) * term_values[str(term["name"])]
                for term in law["terms"]
            )
        except (OverflowError, ValueError) as exc:
            raise _numeric_refusal(
                row_id, "ln_gamma_overflow", f"ln gamma could not be summed: {exc}"
            ) from exc
        _checked_finite(ln_gamma, row_id, "ln_gamma")
        gamma = _checked_exp(ln_gamma, row_id, "gamma")
        ln_gamma_rms = float(law["ln_gamma_rms_closure"])
        _checked_finite(ln_gamma_rms, row_id, "ln_gamma_rms")
        gamma_range = (
            gamma * _checked_exp(-ln_gamma_rms, row_id, "gamma_range"),
            gamma * _checked_exp(ln_gamma_rms, row_id, "gamma_range"),
        )
        _check_gamma_range(gamma_range, row_id)
        return gamma, gamma_range, ln_gamma
    raise ValueError(f"unknown gamma law {law['kind']!r}")


def _component_molar_mass(component: str) -> float:
    mass = MOLAR_MASS_G_MOL.get(component)
    if mass is not None:
        return mass
    from openimcc.trace_components import trace_component_rule

    rule = trace_component_rule(component)
    if rule is None:
        raise ValueError(f"no molar mass is registered for {component!r}")
    return rule.molar_mass_g_mol
