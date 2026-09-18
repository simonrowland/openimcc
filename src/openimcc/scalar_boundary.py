"""Shared admission check for caller-declared numeric scalars."""

from __future__ import annotations

import numbers
from decimal import Decimal
from typing import Any


def is_declared_real_scalar(
    value: Any,
    *,
    allow_numeric_str: bool = False,
) -> bool:
    """Return whether ``value`` declares a real number, never a boolean."""

    if isinstance(value, bool):
        return False
    if isinstance(value, (numbers.Real, Decimal)):
        return True
    if not allow_numeric_str or not isinstance(value, str):
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


__all__ = ("is_declared_real_scalar",)
