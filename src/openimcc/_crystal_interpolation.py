"""Shared interpolation for printed crystal temperature nodes."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterable


def _linear_value_and_slope(
    nodes: Iterable[tuple[float, float]],
    temperature_K: float,
    *,
    slope_required: bool = True,
) -> tuple[float, float]:
    """Return the last-row-wins linear value and its selected secant slope."""
    by_temperature = {
        float(temperature): float(value)
        for temperature, value in nodes
    }
    points = sorted(by_temperature.items())
    temperatures = [point[0] for point in points]
    index = bisect_left(temperatures, temperature_K)
    if index < len(points) and points[index][0] == temperature_K:
        value = points[index][1]
        if index > 0:
            left, right = points[index - 1], points[index]
        elif len(points) > 1:
            left, right = points[index], points[index + 1]
        elif slope_required:
            raise ValueError("a node series needs at least two points for its slope")
        else:
            return value, 0.0
    else:
        if index == 0 or index == len(points):
            raise ValueError(f"no source node interval at {temperature_K:g} K")
        left, right = points[index - 1], points[index]
        fraction = (temperature_K - left[0]) / (right[0] - left[0])
        value = left[1] + (right[1] - left[1]) * fraction
    slope = (right[1] - left[1]) / (right[0] - left[0])
    return value, slope
