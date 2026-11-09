"""Shared loading for opt-in element activity fragments."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from typing import Any


@lru_cache(maxsize=4)
def load_activity_fragment(element: str) -> dict[str, Any]:
    """Load fragment data registered by the trace-component activity table."""
    from openimcc.trace_components import _activity_fragment_registered

    if not _activity_fragment_registered(element):
        raise ValueError(f"no activity fragment is registered for {element!r}")
    resource = files("openimcc.data").joinpath(
        "fragments", element.lower(), "rows.json"
    )
    fragment = json.loads(resource.read_text(encoding="utf-8"))
    if fragment.get("element") != element:
        raise ValueError(f"activity fragment data does not match {element!r}")
    return fragment
