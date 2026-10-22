"""Opt-in dilute-solute P2O5 activity estimate for silicate melts.

The regression estimates the P2O5 tracer activity only; it leaves other
parent activities unchanged to first order. Results always carry a domain
notice, including when the composition lies inside the reported calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

R_J_MOL_K = 8.31446261815324
_LN10 = math.log(10.0)
_MW = {
    "SiO2": 60.0843, "CaO": 56.0774, "MgO": 40.3044,
    "FeO": 71.844, "FeOeq": 71.844, "TiO2": 79.866,
    "Al2O3": 101.961, "Na2O": 61.979, "K2O": 94.196,
    "P2O5": 141.94452, "MnO": 70.9374, "Fe2O3": 159.6882,
}


@dataclass(frozen=True)
class P2O5ActivityNotice:
    code: str
    regression: str
    violated_axes: tuple[str, ...]
    band_width_dex: float
    limitations: tuple[str, ...] = (
        "Fe-P alloy partitioning under carbon reduction is not modelled; metallic iron belongs to the simulator metal account.",
        "Fe(III) phosphates are not modelled.",
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
    amounts: dict[str, float] = {}
    for key, amount in composition_moles.items():
        if key not in _MW:
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
    wt = {name: value * _MW[name] for name, value in amounts.items()}
    total_mass = math.fsum(wt.values())
    wt = {name: value / total_mass * 100.0 for name, value in wt.items()}
    term = 23*x.get("CaO", 0) + 17*x.get("MgO", 0) + 8*x.get("FeO", 0) - 26*x_p
    log_gamma = -1.02*term - 22990.0/T + 9.490
    source_g = -1534480.0 + 506.26*T
    pack_g = -1616372.0 + 539.0*T + 3.5*R_J_MOL_K*T*math.log(1.01325)
    shift = (source_g - pack_g) / (R_J_MOL_K*T*_LN10)
    central = log_gamma + math.log10(x_p) + shift
    # Turkdogan Eq. 13/25 are separate published fits; their envelope is a
    # reported spread and never contributes to the central Suito value.
    bo = wt.get("CaO", 0.0) + 0.33*wt.get("MgO", 0.0)
    turk13 = (-11.85 - 0.1343*bo) + math.log10(x_p)
    turk25 = (-9.84 - 0.142*bo) + math.log10(x_p)
    revised_source_g = -1655480.0 + 571.0*T
    revised_shift = (revised_source_g - pack_g) / (R_J_MOL_K*T*_LN10)
    turk13 += revised_shift
    turk25 += revised_shift
    band = (min(turk13, turk25), max(turk13, turk25))
    width = band[1] - band[0]
    axes: list[str] = []
    if T < 1823 or T > 1923:
        axes.append("temperature_outside_1823_1923_K")
    if wt.get("TiO2", 0.0) > 0:
        axes.append("TiO2_present_without_regression_term")
    if wt.get("SiO2", 0.0) < 1.0:
        axes.append("SiO2_below_calibration_family")
    if wt.get("MgO", 0.0) <= 0:
        axes.append("MgO_absent")
    if wt.get("P2O5", 0.0) > 5.0:
        axes.append("P2O5_above_calibrated_range")
    notice = P2O5ActivityNotice(
        "imcc_p2o5_activity_domain_notice", "Suito-Inoue-1984-Eq-14-no-Mn",
        tuple(axes), width,
    )
    additional_notices = (
        (P2O5FeIIINotice(),) if composition_moles.get("Fe2O3", 0) > 0 else ()
    )
    return P2O5ActivityResult(10.0**central, central, 10.0**(central-math.log10(x_p)),
                              x_p, band, width, notice, additional_notices)
