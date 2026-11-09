"""Opt-in dilute-solute P2O5 activity estimate for silicate melts.

The regression estimates the P2O5 tracer activity only; it leaves other
parent activities unchanged to first order. Results always carry a domain
notice, including when the composition lies inside the reported calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from openimcc._constants import MOLAR_MASS_G_MOL, R_J_MOL_K
from openimcc.activity_fragments import load_activity_fragment

_LN10 = math.log(10.0)
_SUPPORTED_COMPONENTS = frozenset(
    {
        "SiO2", "CaO", "MgO", "FeO", "FeOeq", "TiO2", "Al2O3",
        "Na2O", "K2O", "P2O5", "MnO", "Fe2O3",
    }
)


@dataclass(frozen=True)
class P2O5ActivityNotice:
    code: str
    regression: str
    violated_axes: tuple[str, ...]
    band_width_dex: float
    limitations: tuple[str, ...] = (
        "Fe-P alloy partitioning under carbon reduction is not modelled; metallic iron belongs to the simulator metal account.",
        "Fe(III) phosphates are not modelled.",
        "The absolute activity conversion uses the provisional older Turkdogan-Pearson liquid standard.",
        "The regression requires an MgO-saturated steelmaking-slag family; MgO saturation is not inferred from composition.",
    )


@dataclass(frozen=True)
class P2O5ActivityResult:
    activity: float
    log10_activity: float
    gamma: float
    x_p2o5: float
    band_log10_activity: tuple[float, float]
    band_width_dex: float
    notice: P2O5ActivityNotice
    additional_notices: tuple["P2O5FeIIINotice", ...] = ()


@dataclass(frozen=True)
class P2O5FeIIINotice:
    code: str = "p2o5_feiii_phosphate_chemistry_not_modelled"
    message: str = "Fe(III) phosphate chemistry is not modelled."


def evaluate_p2o5_activity(composition_moles: Mapping[str, float], T_K: float) -> P2O5ActivityResult:
    """Evaluate Suito-Inoue Eq. 14 and return the Turkdogan 13/25 band.

    Composition values are oxide formula-unit mole amounts. Eq. 14 reports
    gamma on the Turkdogan-Pearson hypothetical-liquid P2O5 standard (1 atm).
    Convert its reaction standard to the gas pack's Jung-Hudon 1-bar standard:
    P2(g)+5/2 O2(g)->P2O5(l) has 3.5 gas moles, so ΔG shifts by
    ``3.5 RT ln(1.01325)``. The activity log shift is (G_source-G_pack)/RTln10.
    The pressure correction at 1873 K is about 0.02 dex, a useful sign check.
    """
    T = float(T_K)
    if not math.isfinite(T) or T <= 0:
        raise ValueError("temperature must be finite and positive")
    fragment = load_activity_fragment("P")
    rows = fragment["rows"]
    standards = fragment["standards"]
    central_row = rows["suito_inoue_1984_eq14_no_mn"]
    central_coefficients = central_row["coefficients"]
    amounts: dict[str, float] = {}
    for key, amount in composition_moles.items():
        if key not in _SUPPORTED_COMPONENTS:
            raise ValueError(f"unsupported oxide component {key!r}")
        value = float(amount)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"moles for {key} must be finite and nonnegative")
        canonical = "FeO" if key in {"FeOeq", "Fe2O3"} else key
        # Fe2O3 contributes two Fe atoms, represented as two FeO-equivalent
        # formula moles in this regression's FetO term and composition basis.
        amounts[canonical] = amounts.get(canonical, 0.0) + value * (2 if key == "Fe2O3" else 1)
    total = math.fsum(amounts.values())
    if total <= 0 or amounts.get("P2O5", 0.0) <= 0:
        raise ValueError("composition must contain positive total moles and P2O5")
    x = {name: value / total for name, value in amounts.items()}
    x_p = x["P2O5"]
    log10_x_p = (
        math.log10(x_p)
        if x_p > 0.0
        else math.log10(amounts["P2O5"]) - math.log10(total)
    )
    wt = {name: value * MOLAR_MASS_G_MOL[name] for name, value in amounts.items()}
    total_mass = math.fsum(wt.values())
    wt = {name: value / total_mass * 100.0 for name, value in wt.items()}
    term = math.fsum(
        float(coefficient) * x.get(name, 0.0)
        for name, coefficient in central_coefficients["mole_fraction_terms"].items()
    )
    log_gamma = (
        float(central_coefficients["bracket_multiplier"]) * term
        + float(central_coefficients["inverse_temperature_k"]) / T
        + float(central_coefficients["constant"])
    )
    source_standard = standards[central_row["source_standard"]]
    source_delta_g = source_standard["delta_g_standard_j_mol"]
    source_g = float(source_delta_g["intercept"]) + float(source_delta_g["slope_j_mol_k"]) * T
    target_standard = standards["jung_hudon_liquid_1bar"]
    target_delta_g = target_standard["delta_g_standard_j_mol_before_pressure_correction"]
    pack_g = float(target_delta_g["intercept"]) + float(target_delta_g["slope_j_mol_k"]) * T
    pack_g -= (
        float(target_standard["delta_n_gas"])
        * R_J_MOL_K
        * T
        * math.log(float(target_standard["pressure_ratio_atm_per_bar"]))
    )
    old_standard_shift = (source_g - pack_g) / (R_J_MOL_K*T*_LN10)
    central = log_gamma + log10_x_p + old_standard_shift
    # Turkdogan Eq. 13 and Eq. 25 use different published liquid standards;
    # retain each row's standard when forming the comparison spread.
    eq13 = rows["turkdogan_2000_eq13"]
    eq13_coefficients = eq13["coefficients"]
    eq25 = rows["turkdogan_2000_eq25"]
    eq25_coefficients = eq25["coefficients"]
    bo = math.fsum(
        wt.get(name, 0.0) * float(coefficient)
        for name, coefficient in eq13_coefficients["BO_weight_pct_terms"].items()
    )
    turk13 = (
        float(eq13_coefficients["intercept"])
        + float(eq13_coefficients["slope_per_wt_pct_BO"]) * bo
        + log10_x_p
        + old_standard_shift
    )
    revised_standard = standards[eq25["source_standard"]]
    revised_delta_g = revised_standard["delta_g_standard_j_mol"]
    revised_source_g = float(revised_delta_g["intercept"]) + float(revised_delta_g["slope_j_mol_k"]) * T
    revised_standard_shift = (revised_source_g - pack_g) / (R_J_MOL_K*T*_LN10)
    turk25 = (
        float(eq25_coefficients["intercept"])
        + float(eq25_coefficients["slope_per_wt_pct_BO"]) * bo
        + log10_x_p
        + revised_standard_shift
    )
    band = (min(turk13, turk25), max(turk13, turk25))
    width = band[1] - band[0]
    axes: list[str] = []
    calibration_windows = central_row["calibration_windows"]
    low_p_window, high_p_window = calibration_windows
    low_p_min, low_p_max = (float(value) for value in low_p_window["P2O5_wt_pct"])
    high_p_min, high_p_max = (float(value) for value in high_p_window["P2O5_wt_pct"])
    low_t_min, low_t_max = (float(value) for value in low_p_window["temperature_K"])
    high_t = float(high_p_window["temperature_K"][0])
    p_wt_pct = wt.get("P2O5", 0.0)
    if T < low_t_min or T > low_t_max:
        axes.append("temperature_outside_1823_1923_K")
    if wt.get("TiO2", 0.0) > 0:
        axes.append("TiO2_present_without_regression_term")
    if wt.get("SiO2", 0.0) < 1.0:
        axes.append("SiO2_below_calibration_family")
    if wt.get("MgO", 0.0) <= 0:
        axes.append("MgO_absent")
    if p_wt_pct > high_p_max:
        axes.append("P2O5_above_calibrated_range")
    elif p_wt_pct < low_p_min:
        axes.append("P2O5_below_0.2_wt_pct_calibrated_range")
    elif low_p_max < p_wt_pct < high_p_min:
        axes.append("P2O5_in_1_4_wt_pct_calibration_gap")
    elif high_p_min <= p_wt_pct <= high_p_max and not math.isclose(T, high_t, abs_tol=0.5):
        axes.append("P2O5_4_5_wt_pct_outside_1873K_calibration")
    for oxide in ("Al2O3", "Na2O", "K2O", "MnO"):
        if wt.get(oxide, 0.0) > 0.0:
            axes.append(f"{oxide}_present_without_regression_term")
    notice = P2O5ActivityNotice(
        "imcc_p2o5_activity_domain_notice", "Suito-Inoue-1984-Eq-14-no-Mn",
        tuple(axes), width,
    )
    additional_notices = (
        (P2O5FeIIINotice(),) if composition_moles.get("Fe2O3", 0) > 0 else ()
    )
    return P2O5ActivityResult(10.0**central, central, 10.0**(central-log10_x_p),
                              x_p, band, width, notice, additional_notices)
