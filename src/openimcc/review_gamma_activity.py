"""Evaluate the shared, source-railed review-gamma activity rows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

from openimcc._constants import MOLAR_MASS_G_MOL, R_J_MOL_K
from openimcc.activity_fragments import load_activity_fragment
from openimcc.trace_envelope_calibration import _iw_log10_fO2_bar

_LOG10_BAR_PER_ATM = math.log10(1.01325)
_TRACE_PARENTS = frozenset(
    {
        "SnO", "Ga2O3", "In2O3", "Cu2O", "GeO2", "As2O3", "Bi2O3",
        "Ce2O3",
        "NiO", "CoO", "PbO", "WO3",
    }
)
_NON_OXIDE_COMPONENTS = frozenset({"S", "Cl", "F", "Br", "I", "C", "H", "N"})


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
    conversion. Temperature picks the nearest source row. A family screen
    selects a row only when explicitly requested; otherwise it supplies a
    domain flag and does not change the selected coefficient.
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
    amounts: dict[str, float] = {}
    input_components = tuple(composition_moles)
    for component, raw_amount in composition_moles.items():
        amount = float(raw_amount)
        if not math.isfinite(amount) or amount < 0.0:
            raise ValueError(f"moles for {component} must be finite and nonnegative")
        canonical = "FeO" if component == "FeOeq" else component
        amounts[canonical] = amounts.get(canonical, 0.0) + amount

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
    row = _select_row(candidates, temperature, amounts)
    screen = row["composition_screen"]
    if screen["kind"] == "supported_component_set":
        supported = frozenset(str(name) for name in screen["components"])
        for component in input_components:
            if component not in supported:
                raise ValueError(f"unsupported oxide component {component!r}")

    parent_moles = amounts.get(parent_oxide, 0.0)
    total_moles = math.fsum(amounts.values())
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
    else:
        source_basis_total = total_moles - parent_moles + source_units * parent_moles
        source_x = source_units * parent_moles / source_basis_total

    gamma, gamma_range = _gamma_for_row(row, temperature, amounts)
    conversion = row.get("fusion_conversion")
    if conversion is not None:
        transitions = conversion["transitions"]
        delta_g = float(conversion.get("delta_G_heat_capacity_correction_J_mol", 0.0))
        transition_variance = 0.0
        for transition in transitions:
            transition_temperature = float(transition["T_K"])
            delta_h = float(transition["delta_H_J_mol"])
            delta_g += delta_h * (1.0 - temperature / transition_temperature)
            delta_h_uncertainty = float(transition["delta_H_uncertainty_J_mol"])
            transition_variance += (
                delta_h_uncertainty * (1.0 - temperature / transition_temperature)
            ) ** 2
        # At phase equilibrium, μ°_solid + RT ln(a_s) = μ°_parent + RT ln(a_l),
        # so ln(a_l) = ln(a_s) - [G°_parent(T)-G°_stable-solid(T)]/(RT).
        # The transition sum uses ΔCp≈0; the source's integrated phase heat
        # capacities correct its central ΔG to the shipped liquid-parent Gibbs
        # function. The exponent is dimensionless and positive ΔG makes the
        # parent-liquid activity smaller than the stable-solid activity.
        factor = math.exp(-delta_g / (R_J_MOL_K * temperature))
        gamma *= factor
        closure_rms = float(row["gamma_law"].get("ln_gamma_rms_closure", 0.0))
        transition_log_uncertainty = math.sqrt(transition_variance) / (
            R_J_MOL_K * temperature
        )
        log_half_width = math.hypot(closure_rms, transition_log_uncertainty)
        gamma_range = (
            gamma * math.exp(-log_half_width),
            gamma * math.exp(log_half_width),
        )

    source_activity = gamma * source_x
    activity = source_activity**parent_power
    activity_range = (
        (gamma_range[0] * source_x) ** parent_power,
        (gamma_range[1] * source_x) ** parent_power,
    )

    violated: list[str] = []
    distances: list[str] = []
    t_low, t_high = map(float, row["temperature_window_K"])
    t_distance = max(t_low - temperature, 0.0, temperature - t_high)
    if t_distance > 0.0:
        violated.append("temperature_extrapolated")
        distances.append(f"temperature distance {t_distance:.6g} K")

    if screen["kind"] == "family_oxide_set":
        family_distance, _family_size = _family_score(screen, amounts)
        if family_distance > 0.0:
            violated.append("melt_family_extrapolated")
            distances.append(f"melt-family distance {family_distance:.6g} mole fraction")
    elif screen["kind"] == "an_di_target":
        if not _matches_an_di_target(screen, amounts):
            violated.append("melt_composition_differs_from_An_Di_target")
    elif screen["kind"] == "supported_component_set":
        pass
    else:
        raise ValueError(f"unknown composition screen {screen['kind']!r}")

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

    composition_windows = row.get("composition_windows_wt_pct", {})
    if composition_windows:
        total_mass = math.fsum(
            amount * _component_molar_mass(component)
            for component, amount in amounts.items()
        )
        weight_pct = {
            component: amounts.get(component, 0.0)
            * _component_molar_mass(component)
            / total_mass
            * 100.0
            for component in composition_windows
        }
        for component, window in composition_windows.items():
            weight = weight_pct[component]
            low, high = map(float, window["range"])
            if weight < low or weight > high:
                violated.append(str(window["outside_flag"]))
            else:
                above = window.get("above")
                if above is not None and weight > float(above["value"]):
                    violated.append(str(above["flag"]))

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
    if "oxygen_fugacity_window_unchecked" in violated:
        reference += "; oxygen-fugacity window unchecked"
    if "oxygen_fugacity_window_unreported" in violated:
        reference += "; oxygen-fugacity window not reported"

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
        return component_selection[0]
    if len(nearest) > 1 and any(
        row["composition_screen"].get("select_tie_by_family") for row in nearest
    ):
        return min(
            nearest,
            key=lambda row: _family_score(row["composition_screen"], amounts),
        )
    return nearest[0]


def _family_score(
    screen: Mapping[str, object], amounts: Mapping[str, float]
) -> tuple[float, int]:
    family = frozenset(str(name) for name in screen["oxides"])
    major_total = math.fsum(
        amount
        for name, amount in amounts.items()
        if name not in _TRACE_PARENTS and name not in _NON_OXIDE_COMPONENTS
    )
    outside = math.fsum(
        amount
        for name, amount in amounts.items()
        if name not in _TRACE_PARENTS
        and name not in _NON_OXIDE_COMPONENTS
        and name not in family
    )
    distance = outside / major_total if major_total > 0.0 else 1.0
    if screen.get("require_exact_family"):
        present_family = sum(amounts.get(name, 0.0) > 0.0 for name in family)
        if present_family != len(family) or outside > 0.0:
            distance += 1.0
    return distance, len(family)


def _matches_an_di_target(
    screen: Mapping[str, object], amounts: Mapping[str, float]
) -> bool:
    target = {str(name): float(value) for name, value in screen["major_wt_pct"].items()}
    for component, amount in amounts.items():
        if amount <= 0.0 or component in target:
            continue
        if component in _TRACE_PARENTS or component in _NON_OXIDE_COMPONENTS:
            continue
        return False
    masses = {
        component: amounts.get(component, 0.0) * MOLAR_MASS_G_MOL[component]
        for component in target
    }
    total_mass = math.fsum(masses.values())
    if total_mass <= 0.0:
        return False
    tolerance = float(screen["tolerance_wt_pct"])
    return all(
        abs(masses[component] / total_mass * 100.0 - expected) <= tolerance
        for component, expected in target.items()
    )


def _gamma_for_row(
    row: Mapping[str, object],
    temperature: float,
    amounts: Mapping[str, float],
) -> tuple[float, tuple[float, float]]:
    law = row["gamma_law"]
    if law["kind"] == "range":
        low, high = map(float, law["gamma_endpoints"])
        if low <= 0.0 or high < low:
            raise ValueError("gamma endpoints must be positive and ordered")
        point = float(law.get("gamma_point", math.sqrt(low * high)))
        if not low <= point <= high:
            raise ValueError("gamma point must lie within its printed endpoints")
        return point, (low, high)
    if law["kind"] == "log10_linear":
        gamma = 10.0 ** (float(law["A_K"]) / temperature + float(law["B"]))
        return gamma, (gamma, gamma)
    if law["kind"] == "composition_terms":
        input_basis = str(law["input_basis"])
        if input_basis == "oxide_mole_fraction":
            total_moles = math.fsum(amounts.values())
            x_cao = amounts.get("CaO", 0.0) / total_moles
            x_sio2 = amounts.get("SiO2", 0.0) / total_moles
            term_values = {
                "constant": 1.0,
                "x_CaO_squared": x_cao * x_cao,
                "x_SiO2_squared": x_sio2 * x_sio2,
                "x_CaO_times_x_SiO2": x_cao * x_sio2,
            }
        elif input_basis == "weight_percent":
            cao_mass = amounts.get("CaO", 0.0) * MOLAR_MASS_G_MOL["CaO"]
            sio2_mass = amounts.get("SiO2", 0.0) * MOLAR_MASS_G_MOL["SiO2"]
            al2o3_mass = amounts.get("Al2O3", 0.0) * MOLAR_MASS_G_MOL["Al2O3"]
            if sio2_mass <= 0.0:
                raise ValueError("weight-percent composition_terms require SiO2")
            term_values = {
                "constant": 1.0,
                "wt_CaO_plus_0_72_Al2O3_over_SiO2": (
                    (cao_mass + 0.72 * al2o3_mass) / sio2_mass
                ),
            }
        else:
            raise ValueError(f"unknown composition_terms basis {input_basis!r}")

        ln_gamma = math.fsum(
            float(term["coefficient"]) * term_values[str(term["name"])]
            for term in law["terms"]
        )
        gamma = math.exp(ln_gamma)
        ln_gamma_rms = float(law.get("ln_gamma_rms_closure", 0.0))
        return gamma, (
            gamma * math.exp(-ln_gamma_rms),
            gamma * math.exp(ln_gamma_rms),
        )
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
