"""Opt-in trace CoO activity estimate from silicate-melt experiments.

The measured mean is composition-pooled and always carries a domain notice.
It supplements the existing pure-liquid CoO standard used by the gas layer;
it does not alter the default melt model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from openimcc.activity_fragments import load_activity_fragment
from openimcc.review_gamma_activity import _evaluate_review_gamma_activity

_FRAGMENT = load_activity_fragment("Co")
_ACTIVITY_ROW = _FRAGMENT["rows"][0]
_OXIDE_COMPONENTS = frozenset(
    _ACTIVITY_ROW["composition_screen"]["components"]
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
    result = _evaluate_review_gamma_activity(
        "Co",
        composition_moles,
        T_K,
        parent_oxide="CoO",
        log10_fO2_relative_to_IW=log10_fO2_over_IW,
    )
    x_coo = result.source_mole_fraction
    return CoActivityResult(
        activity=result.source_activity,
        log10_activity=math.log10(result.gamma) + math.log10(x_coo),
        gamma=result.gamma,
        x_coo=x_coo,
        gamma_range=result.gamma_range,
        activity_range=result.activity_range,
        notice=CoActivityNotice(
            code=result.notice.code,
            violated_axes=result.notice.violated_axes,
            limitations=result.notice.limitations,
        ),
    )
