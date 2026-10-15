"""Public SF04 cold starts for strongly depleted trace-parent melts."""

from __future__ import annotations

import math

import pytest

from openimcc import ImccInvalidWarmStartError, evaluate
from openimcc.kernel import ImccNonconvergenceError, ImccRefusal
from openimcc import model


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
    assert result.convergence.solver_path == "composition-continuation"
    assert result.convergence.continuation_stages > 0
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

    invalid_seeds = (
        [0.0],
        [float("nan")] * len(y_init),
        [-201.0] * len(y_init),
        [101.0] * len(y_init),
        "not-a-vector",
        [[0.0], [0.0, 1.0]],
    )
    for seed in invalid_seeds:
        with pytest.raises(ImccInvalidWarmStartError, match="y_init"):
            evaluate(composition, temperature, y_init=seed)
    assert issubclass(ImccInvalidWarmStartError, ImccRefusal)


def test_non_silica_trace_parent_uses_composition_continuation() -> None:
    composition = {
        "SiO2": 197.0,
        "MgO": 1.0e-12
        * (197.0 + 175.85 + 2014.86 + 1360.0 + 192.88 + 5.3 + 8.96),
        "FeO": 175.85,
        "CaO": 2014.86,
        "Al2O3": 1360.0,
        "TiO2": 192.88,
        "Na2O": 5.3,
        "K2O": 8.96,
    }

    result = evaluate(
        composition,
        2400.65,
        allow_extrapolation=True,
        allow_out_of_envelope=True,
    )

    assert result.convergence.solver_path == "composition-continuation"
    assert result.convergence.continuation_stages > 0
    assert result.convergence.residual_inf <= 1.0e-12


def test_failed_composition_continuation_preserves_original_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    diagnostics = {
        "iterations": 17,
        "residual_inf": 0.25,
        "residual_l2": 0.4,
        "final_y": [0.0] * 8,
        "solver_path": "failed",
        "continuation_stages": 2,
        "scipy_message": "fixture failure",
    }

    def refuse(*_args: object, **_kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise ImccNonconvergenceError("fixture refuses", diagnostics)

    monkeypatch.setattr(model, "solve_imcc_sf04", refuse)
    composition, temperature, _expected = CASES[0]
    with pytest.raises(ImccNonconvergenceError, match="fixture refuses") as caught:
        evaluate(composition, temperature)

    assert calls == 8
    assert caught.value.diagnostics == diagnostics


def test_none_warm_start_is_bit_identical_to_omitted_default() -> None:
    composition = {
        "SiO2": 51.85068,
        "MgO": 4.78527,
        "FeO": 13.77307,
        "CaO": 9.02862,
        "Al2O3": 14.80572,
        "TiO2": 1.73824,
        "Na2O": 3.23108,
        "K2O": 0.78732,
    }
    omitted = evaluate(composition, 1800.0)
    explicit = evaluate(composition, 1800.0, y_init=None)

    for name in (
        "parent_mol",
        "parent_x",
        "parent_x_star",
        "parent_activity",
        "parent_gamma",
        "complex_x",
        "species_x",
    ):
        assert getattr(omitted, name).tobytes() == getattr(explicit, name).tobytes()
    assert omitted.temperature_K.hex() == explicit.temperature_K.hex()
    assert omitted.basis.hex() == explicit.basis.hex()
    assert omitted.D.hex() == explicit.D.hex()
    assert omitted.parent_oxides == explicit.parent_oxides
    assert omitted.species_names == explicit.species_names
    assert omitted.convergence == explicit.convergence
    assert omitted.labels == explicit.labels
    assert omitted.extrapolated == explicit.extrapolated
