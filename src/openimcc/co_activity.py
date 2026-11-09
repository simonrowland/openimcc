"""Opt-in trace CoO activity estimate from silicate-melt experiments.

The measured mean is composition-pooled and always carries a domain notice.
It supplements the existing pure-liquid CoO standard used by the gas layer;
it does not alter the default melt model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from openimcc._constants import MOLAR_MASS_G_MOL
from openimcc.activity_fragments import load_activity_fragment

_FRAGMENT = load_activity_fragment("Co")
_ACTIVITY_ROW = _FRAGMENT["rows"][0]
_OXIDE_COMPONENTS = frozenset(
    {"SiO2", "MgO", "FeO", "CaO", "Al2O3", "TiO2", "Na2O", "K2O", "CoO"}
)


@dataclass(frozen=True)
class CoActivityNotice:
    code: str
    violated_axes: tuple[str, ...]
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class CoActivityResult:
    activity: float
    log10_activity: float
    gamma: float
    x_coo: float
    gamma_range: tuple[float, float]
    activity_range: tuple[float, float]
    notice: CoActivityNotice


def evaluate_co_activity(
    composition_moles: Mapping[str, float],
    T_K: float,
    *,
    log10_fO2_over_IW: float | None = None,
) -> CoActivityResult:
    """Return the dilute CoO activity ``a = gamma * x_CoO``.

    ``composition_moles`` contains oxide formula-unit moles on the supported
    nine-oxide basis. The 1.51 ± 0.28 coefficient is the reported pooled mean
    for liquid-standard CoO (76 experiments); the spread is kept as reported,
    not recast as a confidence interval. The source reports it as independent
    of temperature and oxygen fugacity over 1573.15–1873.15 K and IW−3 to
    IW+1.5. Composition ranges are checked where the paper reports them.

    Premise: at dilute concentration, the CoO activity is its simple oxide
    mole fraction times the measured liquid-standard coefficient.
    Algebra: ``a_CoO = gamma_CoO * n_CoO / sum(n_oxide)``. Units: both factors
    are dimensionless. Sanity: at x_CoO=0.001, a_CoO=0.00151 with the reported
    range 0.00123–0.00179. The pooled coefficient remains provisional for any
    particular basalt composition and the dilute limit has no published
    numeric upper bound in this record.
    """
    temperature = float(T_K)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")

    amounts: dict[str, float] = {}
    for component, raw_amount in composition_moles.items():
        if component not in _OXIDE_COMPONENTS:
            raise ValueError(f"unsupported oxide component {component!r}")
        amount = float(raw_amount)
        if not math.isfinite(amount) or amount < 0.0:
            raise ValueError(f"moles for {component} must be finite and nonnegative")
        amounts[component] = amount

    total_moles = math.fsum(amounts.values())
    coo_moles = amounts.get("CoO", 0.0)
    if total_moles <= 0.0 or coo_moles <= 0.0:
        raise ValueError("composition must contain positive total moles and CoO")
    x_coo = coo_moles / total_moles

    total_mass = math.fsum(
        amount * MOLAR_MASS_G_MOL[component]
        for component, amount in amounts.items()
    )
    weight_pct = {
        component: amount * MOLAR_MASS_G_MOL[component] / total_mass * 100.0
        for component, amount in amounts.items()
    }

    domain = _ACTIVITY_ROW["domain"]
    violated: list[str] = []
    t_min, t_max = domain["temperature_K"]
    if temperature < t_min or temperature > t_max:
        violated.append("temperature_outside_1573.15_1873.15_K")
    if log10_fO2_over_IW is None:
        violated.append("oxygen_fugacity_domain_unchecked")
    else:
        oxygen_offset = float(log10_fO2_over_IW)
        if not math.isfinite(oxygen_offset):
            raise ValueError("log10_fO2_over_IW must be finite when supplied")
        o_min, o_max = domain["log10_fO2_over_IW"]
        if oxygen_offset < o_min or oxygen_offset > o_max:
            violated.append("oxygen_fugacity_outside_IW_minus_3_to_plus_1_5")

    feo_min, feo_max = domain["FeO_wt_pct"]
    feo_wt_pct = weight_pct.get("FeO", 0.0)
    if feo_wt_pct < feo_min or feo_wt_pct > feo_max:
        violated.append("FeO_outside_0_to_12_wt_pct")
    mgo_min, mgo_max = domain["MgO_wt_pct"]
    mgo_wt_pct = weight_pct.get("MgO", 0.0)
    if mgo_wt_pct < mgo_min or mgo_wt_pct > mgo_max:
        violated.append("MgO_outside_4_to_30_wt_pct")
    elif mgo_wt_pct > domain["MgO_no_reported_increase_through_wt_pct"]:
        violated.append("MgO_above_20_wt_pct_source_reports_gamma_increase")

    gamma = float(_ACTIVITY_ROW["gamma_mean"])
    spread = float(_ACTIVITY_ROW["gamma_reported_spread"])
    gamma_range = (gamma - spread, gamma + spread)
    activity_range = (gamma_range[0] * x_coo, gamma_range[1] * x_coo)
    notice = CoActivityNotice(
        code="imcc_co_activity_domain_notice",
        violated_axes=tuple(violated),
        limitations=(
            "The pooled coefficient is provisional for a specific melt composition.",
            "The dilute-concentration upper limit is not numerically specified by the source.",
            "Other major-oxide composition effects are not parameterized.",
        ),
    )
    return CoActivityResult(
        activity=gamma * x_coo,
        log10_activity=math.log10(gamma) + math.log10(x_coo),
        gamma=gamma,
        x_coo=x_coo,
        gamma_range=gamma_range,
        activity_range=activity_range,
        notice=notice,
    )
