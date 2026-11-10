"""Opt-in NiO dilute-solute activity from the Ni fragment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from openimcc.activity_fragments import load_activity_fragment
from openimcc.review_gamma_activity import _evaluate_review_gamma_activity

_NI_FRAGMENT = load_activity_fragment("Ni")
_NI_ROW = _NI_FRAGMENT["rows"][0]
_SUPPORTED_COMPONENTS = frozenset(
    _NI_ROW["composition_screen"]["components"]
)


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
    result = _evaluate_review_gamma_activity(
        "Ni",
        composition_moles,
        T_K,
        parent_oxide="NiO",
        log10_fO2_relative_to_IW=log10_fO2_relative_to_IW,
    )
    return NiActivityResult(
        activity=result.source_activity,
        gamma=result.gamma,
        x_nio=result.source_mole_fraction,
        activity_spread=result.activity_range,
        notice=NiActivityNotice(
            code=result.notice.code,
            violated_axes=result.notice.violated_axes,
            limitations=result.notice.limitations,
        ),
    )
