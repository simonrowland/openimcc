"""Shared loading for opt-in element activity fragments."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from copy import deepcopy
from functools import lru_cache
from importlib.resources import files
from typing import Any

from openimcc.scalar_boundary import is_declared_real_scalar


@lru_cache(maxsize=4)
def _cached_activity_fragment(element: str) -> dict[str, Any]:
    from openimcc.trace_components import _activity_fragment_registered

    if not _activity_fragment_registered(element):
        raise ValueError(f"no activity fragment is registered for {element!r}")
    resource = files("openimcc.data").joinpath(
        "fragments", element.lower(), "rows.json"
    )
    fragment = json.loads(resource.read_text(encoding="utf-8"))
    if fragment.get("element") != element:
        raise ValueError(f"activity fragment data does not match {element!r}")
    rows = fragment.get("rows", ())
    if isinstance(rows, Mapping):
        rows = rows.values()
    for row in rows:
        law = row.get("gamma_law", {})
        if law.get("kind") != "composition_terms":
            continue
        closure_rms = law.get("ln_gamma_rms_closure")
        if (
            not is_declared_real_scalar(closure_rms)
            or not math.isfinite(float(closure_rms))
            or float(closure_rms) < 0.0
        ):
            raise ValueError(
                "composition_terms ln_gamma_rms_closure must be finite and nonnegative"
            )
    return fragment


def load_activity_fragment(element: str) -> dict[str, Any]:
    """Return an independent copy of cached fragment data."""
    return deepcopy(_cached_activity_fragment(element))
