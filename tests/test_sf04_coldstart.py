"""Public SF04 cold starts for strongly depleted trace-parent melts."""

from __future__ import annotations

import math

import pytest

from openimcc import evaluate


CASES = (
    (
        {
            "SiO2": 0.011138886562497141,
            "MgO": 2288.8042997487846,
            "FeO": 87.38220230346212,
            "CaO": 2014.8644234379233,
            "Al2O3": 1360.0091131723866,
            "TiO2": 192.82496476933426,
            "Na2O": 5.3731722769972105e-09,
            "K2O": 8.844980280923895,
        },
        2445.65,
        3.4859611072197934e-09,
    ),
    (
        {
            "SiO2": 196.99908214824137,
            "MgO": 2291.7546219084725,
            "FeO": 175.85697827570334,
            "CaO": 2014.8648153917845,
            "Al2O3": 1360.008634090145,
            "TiO2": 192.88986671222364,
            "Na2O": 5.3662996336774405e-09,
            "K2O": 8.96629272065383,
        },
        2400.65,
        6.914222520149932e-05,
    ),
    (
        {
            "SiO2": 196.99908214823887,
            "MgO": 2291.754621908471,
            "FeO": 175.8569782757039,
            "CaO": 2014.8648153917845,
            "Al2O3": 1360.008634090145,
            "TiO2": 192.88986671222364,
            "Na2O": 5.366760660756533e-09,
            "K2O": 8.966292720650552,
        },
        2400.65,
        6.914222520149845e-05,
    ),
)


@pytest.mark.parametrize(
    ("composition", "temperature", "expected_silica_activity"), CASES
)
def test_depleted_melt_cold_start_converges(
    composition: dict[str, float],
    temperature: float,
    expected_silica_activity: float,
) -> None:
    result = evaluate(composition, temperature)

    assert result.convergence.residual_inf <= 1.0e-12
    assert math.isclose(
        result.parent_activity[result.parent_oxides.index("SiO2")],
        expected_silica_activity,
        rel_tol=2.0e-12,
    )


def test_evaluate_accepts_a_validated_warm_start() -> None:
    composition, temperature, _expected = CASES[0]
    cold = evaluate(composition, temperature)
    active = [name for name in cold.parent_oxides if composition.get(name, 0.0) > 0.0]
    y_init = [
        math.log(cold.parent_x_star[cold.parent_oxides.index(name)])
        for name in active
    ]

    warm = evaluate(composition, temperature + 1.0, y_init=y_init)

    assert warm.convergence.residual_inf <= 1.0e-12
    assert all(math.isfinite(value) for value in warm.parent_activity)

    with pytest.raises(ValueError, match="y_init"):
        evaluate(composition, temperature, y_init=[0.0])
