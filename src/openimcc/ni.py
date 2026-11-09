"""Opt-in NiO dilute-solute activity from the Ni fragment."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from openimcc._constants import MOLAR_MASS_G_MOL
from openimcc.activity_fragments import load_activity_fragment

_SUPPORTED_COMPONENTS = frozenset(
    {
        "SiO2", "CaO", "MgO", "FeO", "FeOeq", "Al2O3",
        "TiO2", "Na2O", "K2O", "MnO", "Cr2O3", "P2O5", "NiO",
    }
)
_NI_FRAGMENT = load_activity_fragment("Ni")
_NI_ROW = _NI_FRAGMENT["rows"][0]


@dataclass(frozen=True)
class NiActivityNotice:
    code: str
    violated_axes: tuple[str, ...]
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class NiActivityResult:
    activity: float
    gamma: float
    x_nio: float
    activity_spread: tuple[float, float]
    notice: NiActivityNotice


def evaluate_ni_activity(
    composition_moles: Mapping[str, float],
    T_K: float,
    log10_fO2_relative_to_IW: float | None = None,
) -> NiActivityResult:
    """Calculate the opt-in dilute NiO activity from the Ni fragment.

    Composition entries are formula-unit mole amounts.  The composition is
    only used to calculate the simple oxide mole fraction and check the
    published FeO/MgO bounds; this helper does not change the melt solve.
    """
    row = _NI_ROW
    temperature = float(T_K)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    fO2_iw = None
    if log10_fO2_relative_to_IW is not None:
        fO2_iw = float(log10_fO2_relative_to_IW)
        if not math.isfinite(fO2_iw):
            raise ValueError("log10 fO2 relative to IW must be finite")

    amounts: dict[str, float] = {}
    for component, amount in composition_moles.items():
        if component not in _SUPPORTED_COMPONENTS:
            raise ValueError(f"unsupported oxide component {component!r}")
        value = float(amount)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"moles for {component} must be finite and nonnegative")
        canonical = "FeO" if component == "FeOeq" else component
        amounts[canonical] = amounts.get(canonical, 0.0) + value

    total = math.fsum(amounts.values())
    if total <= 0.0 or amounts.get("NiO", 0.0) <= 0.0:
        raise ValueError("composition must contain positive total moles and NiO")
    x_nio = amounts["NiO"] / total
    total_mass = math.fsum(
        amount * MOLAR_MASS_G_MOL[component]
        for component, amount in amounts.items()
    )
    feo_wt = amounts.get("FeO", 0.0) * MOLAR_MASS_G_MOL["FeO"] / total_mass * 100.0
    mgo_wt = amounts.get("MgO", 0.0) * MOLAR_MASS_G_MOL["MgO"] / total_mass * 100.0

    gamma = float(row["gamma_NiO_l"])
    spread = float(row["reported_spread"]["value"])
    activity = gamma * x_nio
    activity_spread = ((gamma - spread) * x_nio, (gamma + spread) * x_nio)
    t_min, t_max = map(float, row["T_domain_K"])
    fo_min, fo_max = map(float, row["fO2_relative_to_IW"])
    feo_min, feo_max = map(float, row["composition_domain"]["FeO_wt_pct"])
    mgo_min, mgo_max = map(float, row["composition_domain"]["MgO_wt_pct"])
    axes: list[str] = []
    if not t_min <= temperature <= t_max:
        axes.append("temperature_outside_1573_1873_K")
    if fO2_iw is None:
        axes.append("oxygen_fugacity_relative_to_IW_not_supplied")
    elif not fo_min <= fO2_iw <= fo_max:
        axes.append("oxygen_fugacity_outside_IW_minus_3_to_plus_1_5")
    if not feo_min <= feo_wt <= feo_max:
        axes.append("FeO_wt_pct_outside_0_to_12")
    if not mgo_min <= mgo_wt <= mgo_max:
        axes.append("MgO_wt_pct_outside_4_to_30")

    notice = NiActivityNotice(
        code=str(_NI_FRAGMENT["flags"][0]["code"]),
        violated_axes=tuple(axes),
        limitations=tuple(str(flag) for flag in row["range_flags"]),
    )
    return NiActivityResult(activity, gamma, x_nio, activity_spread, notice)
