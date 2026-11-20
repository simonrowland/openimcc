"""Shared checked numeric operations with caller-owned typed refusals."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Literal

_NumericFailure = Literal[
    "sum_overflow",
    "nonfinite",
    "not_positive_finite",
    "power_overflow",
]
_RefusalFactory = Callable[
    [str, str, _NumericFailure, BaseException | None, str | None], Exception
]


def _checked_fsum(
    values: Iterable[float],
    row_id: str,
    axis: str,
    *,
    refusal: _RefusalFactory,
    context: str | None = None,
) -> float:
    try:
        value = math.fsum(values)
    except (OverflowError, ValueError) as exc:
        raise refusal(row_id, axis, "sum_overflow", exc, context) from exc
    return _checked_finite(
        value, row_id, axis, refusal=refusal, context=context,
    )


def _checked_finite(
    value: float,
    row_id: str,
    axis: str,
    *,
    refusal: _RefusalFactory,
    context: str | None = None,
) -> float:
    if not math.isfinite(value):
        raise refusal(row_id, axis, "nonfinite", None, context)
    return value


def _checked_positive_finite(
    value: float,
    row_id: str,
    axis: str,
    *,
    refusal: _RefusalFactory,
    context: str | None = None,
) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise refusal(row_id, axis, "not_positive_finite", None, context)
    return value


def _checked_pow10(
    log10_value: float,
    row_id: str,
    axis: str,
    *,
    refusal: _RefusalFactory,
    context: str | None = None,
    check_log_finite: bool = True,
    check_result: bool = True,
) -> float:
    if check_log_finite:
        _checked_finite(
            log10_value,
            row_id,
            f"log10_{axis}",
            refusal=refusal,
            context=context,
        )
    try:
        value = 10.0**log10_value
    except (OverflowError, ValueError) as exc:
        raise refusal(row_id, axis, "power_overflow", exc, context) from exc
    if not check_result:
        return value
    return _checked_positive_finite(
        value, row_id, axis, refusal=refusal, context=context,
    )
