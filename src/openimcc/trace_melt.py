"""Opt-in trace-melt activity candidates outside the certified SF04 solve."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from openimcc.model import _EXPECTED_PARENT_OXIDES, _OXIDE_MOLAR_MASS_G_MOL
from openimcc.scalar_boundary import is_declared_real_scalar


_LOG10_GAMMA_INTERCEPT = 1.27580279745
_LOG10_GAMMA_LOG10_BASICITY_SLOPE = 0.62054327952
_MIN_T_K = 1771.15
_MAX_T_K = 1775.15
_MIN_LOG10_FO2_ATM = -11.50
_MAX_LOG10_FO2_ATM = -9.56
_MIN_BASICITY_AT_LOW_FO2 = 0.52
_MAX_BASICITY_AT_LOW_FO2 = 1.27
_MIN_BASICITY_AT_HIGH_FO2 = 0.71
_MAX_BASICITY_AT_HIGH_FO2 = 0.93
_MIN_X_CRO15 = 0.0005
_MAX_X_CRO15 = 0.01
_BAR_PER_ATM = 1.01325


@dataclass(frozen=True, slots=True)
class Cr3TraceActivity:
    """Cr(III) activity result, explicitly marked as an unvalidated candidate."""

    parent_activity: float
    half_formula_activity: float
    gamma_infinity: float
    validation_status: str = "unvalidated opt-in candidate"


def _finite_real(name: str, value: object) -> float:
    if not is_declared_real_scalar(value):
        raise ValueError(f"{name} must be a finite real scalar")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite real scalar")
    return converted


def _basicity(
    composition: Mapping[str, float], basis_type: Literal["mol", "wt"]
) -> float:
    unknown = set(composition) - set(_EXPECTED_PARENT_OXIDES)
    if unknown:
        raise ValueError(f"unknown SF04 oxide(s): {sorted(unknown)}")

    amounts = {
        name: _finite_real(name, composition.get(name, 0.0))
        for name in _EXPECTED_PARENT_OXIDES
    }
    negative = [name for name, amount in amounts.items() if amount < 0.0]
    if negative:
        raise ValueError(f"composition amounts must be >= 0: {negative}")
    unsupported = [
        name
        for name, amount in amounts.items()
        if name not in ("CaO", "SiO2") and amount > 0.0
    ]
    if unsupported:
        raise ValueError(
            "outside measured Cr composition hull; positive non-CaO/SiO2 "
            f"components: {unsupported}"
        )

    if basis_type == "mol":
        weights = {
            name: amounts[name] * _OXIDE_MOLAR_MASS_G_MOL[name]
            for name in ("CaO", "SiO2")
        }
    elif basis_type == "wt":
        weights = {name: amounts[name] for name in ("CaO", "SiO2")}
    else:
        raise ValueError("basis_type must be 'mol' or 'wt'")

    if weights["CaO"] <= 0.0 or weights["SiO2"] <= 0.0:
        raise ValueError("composition must contain positive CaO and SiO2")
    return weights["CaO"] / weights["SiO2"]


def cr3_trace_activity(
    sf04_composition: Mapping[str, float],
    T_K: float,
    fO2_bar: float,
    x_cro15: float,
    *,
    basis_type: Literal["mol", "wt"] = "mol",
) -> Cr3TraceActivity:
    """Return a candidate Cr2O3(l) activity for a dilute Cr(III) inventory.

    ``sf04_composition`` contains the eight SF04 major oxides and may omit
    zero-valued components. ``x_cro15`` is the CrO1.5 component mole fraction
    (one Cr(III) per half-formula) on the source-table basis. The fitted
    Pretorius–Muan model is only supported on its CaO–SiO2 composition, T,
    oxygen-fugacity, and dilute-loading hull. Other compositions or conditions
    raise ``ValueError``. This remains an **unvalidated opt-in candidate**.

    Pass ``result.parent_activity`` to ``evaluate_gas`` under the ``Cr2O3``
    parent with ``parent_oxides=(*IMCC_PARENT_OXIDES, "Cr2O3")``.
    """
    temperature = _finite_real("T_K", T_K)
    fugacity = _finite_real("fO2_bar", fO2_bar)
    loading = _finite_real("x_cro15", x_cro15)
    if temperature <= 0.0:
        raise ValueError("T_K must be positive")
    if fugacity <= 0.0:
        raise ValueError("fO2_bar must be positive")
    if loading < 0.0:
        raise ValueError("x_cro15 must be >= 0")
    if basis_type not in ("mol", "wt"):
        raise ValueError("basis_type must be 'mol' or 'wt'")

    basicity = _basicity(sf04_composition, basis_type)
    log10_fugacity_atm = math.log10(fugacity / _BAR_PER_ATM)
    if not _MIN_T_K <= temperature <= _MAX_T_K:
        raise ValueError(
            f"T_K={temperature} is outside measured Cr range "
            f"[{_MIN_T_K}, {_MAX_T_K}] K"
        )
    if not _MIN_LOG10_FO2_ATM <= log10_fugacity_atm <= _MAX_LOG10_FO2_ATM:
        raise ValueError(
            "log10(fO2/atm) is outside measured Cr range "
            f"[{_MIN_LOG10_FO2_ATM}, {_MAX_LOG10_FO2_ATM}]"
        )

    fugacity_fraction = (
        (log10_fugacity_atm - _MIN_LOG10_FO2_ATM)
        / (_MAX_LOG10_FO2_ATM - _MIN_LOG10_FO2_ATM)
    )
    basicity_min = _MIN_BASICITY_AT_LOW_FO2 + fugacity_fraction * (
        _MIN_BASICITY_AT_HIGH_FO2 - _MIN_BASICITY_AT_LOW_FO2
    )
    basicity_max = _MAX_BASICITY_AT_LOW_FO2 + fugacity_fraction * (
        _MAX_BASICITY_AT_HIGH_FO2 - _MAX_BASICITY_AT_LOW_FO2
    )
    if not basicity_min <= basicity <= basicity_max:
        raise ValueError(
            f"basicity B={basicity} is outside measured Cr composition hull "
            f"[{basicity_min}, {basicity_max}] at log10(fO2/atm)="
            f"{log10_fugacity_atm}"
        )
    if loading > 0.0 and not _MIN_X_CRO15 <= loading <= _MAX_X_CRO15:
        raise ValueError(
            f"x_cro15={loading} is outside measured dilute-loading range "
            f"[{_MIN_X_CRO15}, {_MAX_X_CRO15}]"
        )

    log10_gamma = (
        _LOG10_GAMMA_INTERCEPT
        + _LOG10_GAMMA_LOG10_BASICITY_SLOPE * math.log10(basicity)
    )
    gamma_infinity = 10.0**log10_gamma
    half_formula_activity = gamma_infinity * loading

    # A CrO1.5 half-formula contains one Cr(III); the Cr2O3(l) parent contains
    # two. Thus mu(Cr2O3)=2*mu(CrO1.5), sqrt(a_parent)=a_half=gamma_inf*X_CrO1.5
    # on Pretorius–Muan's matching liquid standard, and a_parent is quadratic
    # in Cr(III) inventory. Equivalently sqrt(a_parent)/X_CrO1.5=gamma_inf.
    parent_activity = half_formula_activity * half_formula_activity
    return Cr3TraceActivity(
        parent_activity=parent_activity,
        half_formula_activity=half_formula_activity,
        gamma_infinity=gamma_infinity,
    )
