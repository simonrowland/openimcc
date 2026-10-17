"""Public SF04 cold starts for strongly depleted trace-parent melts."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from openimcc import ImccInvalidWarmStartError, evaluate, load_datapack
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


def _cai_mole_vector(ti_fraction: float) -> np.ndarray:
    oxides = (
        "SiO2", "MgO", "FeO", "CaO", "Al2O3", "TiO2", "Na2O", "K2O"
    )
    weight = {
        "SiO2": 29.1,
        "MgO": 10.2,
        "FeO": 0.6,
        "CaO": 28.8,
        "Al2O3": 29.6,
        "TiO2": 1.3,
        "Na2O": 0.18,
        "K2O": 0.1,
    }
    mole = {
        name: weight[name] / model._OXIDE_MOLAR_MASS_G_MOL[name]
        for name in oxides
    }
    vector = np.asarray([mole[name] for name in oxides])
    vector /= vector.sum()
    ti_index = oxides.index("TiO2")
    total_other = float(vector.sum() - vector[ti_index])
    vector[ti_index] = ti_fraction
    for name in oxides:
        if name != "TiO2":
            vector[oxides.index(name)] *= (1.0 - ti_fraction) / total_other
    return vector


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


def test_lunar_trace_state_uses_relative_log_x_last_resort() -> None:
    composition = {
        "SiO2": 1489.5026789257954,
        "MgO": 2292.1310472268683,
        "FeO": 0.044418319238703666,
        "CaO": 2014.8649752615986,
        "Al2O3": 1360.0091164074408,
        "TiO2": 192.89908941162028,
        "Na2O": 1.9086093358233833e-07,
        "K2O": 8.337651502659869,
    }

    result = evaluate(
        composition,
        2385.65,
        basis_type="mol",
        enable_sp_extension=False,
        allow_extrapolation=False,
        allow_out_of_envelope=False,
    )

    assert result.convergence.solver_path == "relative-log-x"
    assert result.convergence.residual_inf <= 1.0e-12
    assert result.parent_activity[result.parent_oxides.index("SiO2")] == pytest.approx(
        0.0059419, rel=2.0e-5
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


def test_silica_walk_recovers_above_floor_feo_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oxides = (
        "SiO2", "MgO", "FeO", "CaO", "Al2O3", "TiO2", "Na2O", "K2O"
    )
    feo_fraction = 1.2e-3
    composition = {
        oxide: (feo_fraction if oxide == "FeO" else (1.0 - feo_fraction) / 7.0)
        for oxide in oxides
    }

    calls: list[np.ndarray] = []
    solve = model._solve_imcc_sf04_with_seed

    def record(parent_mol: np.ndarray, *args: object, **kwargs: object) -> object:
        calls.append(parent_mol.copy())
        return solve(parent_mol, *args, **kwargs)

    monkeypatch.setattr(model, "_solve_imcc_sf04_with_seed", record)
    result = evaluate(
        composition,
        2288.0,
        basis_type="mol",
        allow_extrapolation=True,
        allow_out_of_envelope=True,
    )

    assert result.convergence.solver_path == "composition-continuation"
    assert result.convergence.continuation_stages == 6
    assert result.convergence.residual_inf <= 1.0e-12
    assert len(calls) == 7
    assert np.array_equal(calls[0], calls[-1])
    assert calls[1][0] == pytest.approx(calls[0][0] * 10.0)
    assert result.parent_mol[oxides.index("FeO")] / result.basis == pytest.approx(
        feo_fraction
    )


def test_silica_walk_recovers_depleted_cai_tio2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector = _cai_mole_vector(1.0e-5)
    calls: list[np.ndarray] = []
    solve = model._solve_imcc_sf04_with_seed

    def record(parent_mol: np.ndarray, *args: object, **kwargs: object) -> object:
        calls.append(parent_mol.copy())
        return solve(parent_mol, *args, **kwargs)

    monkeypatch.setattr(model, "_solve_imcc_sf04_with_seed", record)
    result = evaluate(
        vector,
        1710.0,
        basis_type="mol",
        allow_extrapolation=True,
        allow_out_of_envelope=True,
    )

    assert result.convergence.solver_path == "composition-continuation"
    assert result.convergence.continuation_stages == 13
    assert result.convergence.residual_inf <= 1.0e-12
    assert any(call[0] >= vector[0] * 10.0 for call in calls)
    assert np.array_equal(calls[-1], vector)
    assert result.parent_mol[5] / result.basis == pytest.approx(1.0e-5)


def test_silica_walk_recovers_cai_trace_missed_by_staged_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector = _cai_mole_vector(1.414213562373095e-7)
    calls: list[np.ndarray] = []
    solve = model._solve_imcc_sf04_with_seed

    def record(parent_mol: np.ndarray, *args: object, **kwargs: object) -> object:
        calls.append(parent_mol.copy())
        return solve(parent_mol, *args, **kwargs)

    monkeypatch.setattr(model, "_solve_imcc_sf04_with_seed", record)
    result = evaluate(
        vector,
        1710.0,
        basis_type="mol",
        allow_extrapolation=True,
        allow_out_of_envelope=True,
    )

    assert result.convergence.solver_path == "composition-continuation"
    assert result.convergence.continuation_stages == 13
    assert result.convergence.residual_inf <= 1.0e-12
    assert any(call[0] >= vector[0] * 10.0 for call in calls)
    assert np.array_equal(calls[-1], vector)


def test_last_converged_trace_seed_recovers_depleted_feo() -> None:
    composition = {name: 1.0 / 8.0 for name in (
        "SiO2", "MgO", "FeO", "CaO", "Al2O3", "TiO2", "Na2O", "K2O"
    )}
    composition["FeO"] = 1.0e-12
    remaining = (1.0 - 1.0e-12) / 7.0
    for name in composition:
        if name != "FeO":
            composition[name] = remaining

    result = evaluate(
        composition,
        1710.0,
        basis_type="mol",
        allow_extrapolation=True,
        allow_out_of_envelope=True,
    )

    assert result.convergence.solver_path == "composition-continuation"
    assert result.convergence.continuation_stages > 7
    assert result.convergence.residual_inf <= 1.0e-12


def test_trace_anchor_recovers_depleted_feo_at_2288k() -> None:
    feo_fraction = float(np.geomspace(2.0e-2, 1.0e-12, 9)[7])
    composition = {
        name: (
            feo_fraction
            if name == "FeO"
            else (1.0 - feo_fraction) / 7.0
        )
        for name in ("SiO2", "MgO", "FeO", "CaO", "Al2O3", "TiO2", "Na2O", "K2O")
    }

    result = evaluate(
        composition,
        2288.0,
        basis_type="mol",
        allow_extrapolation=True,
        allow_out_of_envelope=True,
    )

    assert result.convergence.solver_path == "composition-continuation"
    assert result.convergence.continuation_stages > 7
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
    original_error = ImccNonconvergenceError("fixture refuses", diagnostics)

    def refuse(*_args: object, **_kwargs: object) -> None:
        nonlocal calls
        calls += 1
        raise original_error

    monkeypatch.setattr(model, "_solve_imcc_sf04_with_seed", refuse)
    composition, temperature, _expected = CASES[0]
    with pytest.raises(ImccNonconvergenceError, match="fixture refuses") as caught:
        evaluate(composition, temperature)

    assert calls == 15
    assert caught.value is original_error
    assert caught.value.diagnostics == diagnostics


@pytest.mark.parametrize(
    ("composition", "trace_name", "trace_fraction"),
    (
        (_cai_mole_vector(1.0e-10), "TiO2", 1.0e-10),
        (_cai_mole_vector(1.0e-12), "TiO2", 1.0e-12),
        (np.full(8, 1.0 / 8.0), "FeO", 1.0e-12),
    ),
)
def test_temperature_continuation_recovers_low_temperature_depletions(
    composition: np.ndarray,
    trace_name: str,
    trace_fraction: float,
) -> None:
    if trace_name == "FeO":
        composition = composition.copy()
        composition[2] = trace_fraction
        composition[np.arange(8) != 2] *= (1.0 - trace_fraction) / (
            7.0 / 8.0
        )

    result = evaluate(
        composition,
        1500.0,
        basis_type="mol",
        allow_extrapolation=True,
        allow_out_of_envelope=True,
    )

    assert result.convergence.solver_path == "temperature-continuation"
    assert result.convergence.continuation_stages > 1
    assert result.convergence.residual_inf <= 1.0e-12
    assert np.array_equal(result.parent_mol, composition)
    trace_amount = result.parent_mol[result.parent_oxides.index(trace_name)]
    assert trace_amount / result.basis == pytest.approx(trace_fraction)


def test_temperature_continuation_refusal_call_count_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    calls_by_temperature: dict[float, int] = {}
    composition = _cai_mole_vector(1.0e-12)
    diagnostics = {
        "iterations": 0,
        "residual_inf": 1.0,
        "residual_l2": 1.0,
        "final_y": [0.0] * 8,
        "solver_path": "failed",
        "continuation_stages": 0,
        "scipy_message": "fixture failure",
    }
    warm_result = evaluate(
        composition,
        1710.0,
        basis_type="mol",
        allow_extrapolation=True,
        allow_out_of_envelope=True,
    )

    def refuse(*args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        temperature = float(args[1])
        calls_by_temperature[temperature] = (
            calls_by_temperature.get(temperature, 0) + 1
        )
        if temperature == 1710.0 and calls_by_temperature[temperature] == 14:
            # The final silica stage at the warm temperature returns its
            # independently computed root, allowing the target solve to run.
            return (
                warm_result,
                np.log(warm_result.parent_x_star[composition > 0.0]),
            )
        raise ImccNonconvergenceError("fixture refuses", diagnostics)

    monkeypatch.setattr(model, "_solve_imcc_sf04_with_seed", refuse)
    with pytest.raises(ImccNonconvergenceError) as caught:
        evaluate(
            composition,
            1500.0,
            basis_type="mol",
            allow_extrapolation=True,
            allow_out_of_envelope=True,
        )

    # Both temperatures execute direct solve + seven trace stages + six
    # silica stages; the final warm-started target solve makes call 29, and
    # the relative log-x retry at the target makes call 30. This exact bound
    # fails if the last-resort stage is removed.
    assert calls == 30
    assert calls_by_temperature == {1500.0: 16, 1710.0: 14}
    assert caught.value.diagnostics == diagnostics


@pytest.mark.parametrize(
    "pack_file",
    ("imcc-sf04-d066-ext-v1.json", "imcc-sf04-ext-v4.json"),
)
def test_extension_temperature_refusal_preserves_original_diagnostics(
    pack_file: str,
) -> None:
    pack = load_datapack(
        Path(__file__).parents[1]
        / "src"
        / "openimcc"
        / "data"
        / "packs"
        / pack_file
    )
    core_names = (
        "SiO2", "MgO", "FeO", "CaO", "Al2O3", "TiO2", "Na2O", "K2O"
    )
    core_vector = _cai_mole_vector(1.0e-12)
    composition = np.array(
        [
            core_vector[core_names.index(name)] if name in core_names else 0.0
            for name in pack.parent_oxides
        ]
    )
    options = {
        "pack": pack,
        "basis_type": "mol",
        "enable_sp_extension": True,
        "allow_extrapolation": True,
        "allow_out_of_envelope": True,
    }

    with pytest.raises(ImccNonconvergenceError) as original:
        model.solve_imcc_sf04(
            composition,
            1500.0,
            pack.kernel_datapack,
            basis=float(composition.sum()),
            allow_extrapolation=True,
            max_iter=1,
        )
    with pytest.raises(ImccNonconvergenceError) as caught:
        evaluate(composition, 1500.0, max_iter=1, **options)

    assert caught.value.diagnostics == original.value.diagnostics


@pytest.mark.parametrize(
    "pack_file",
    ("imcc-sf04-d066-ext-v1.json", "imcc-sf04-ext-v4.json"),
)
def test_extension_temperature_max_iter_zero_preserves_did_not_run(
    pack_file: str,
) -> None:
    pack = load_datapack(
        Path(__file__).parents[1]
        / "src"
        / "openimcc"
        / "data"
        / "packs"
        / pack_file
    )
    composition = np.full(len(pack.parent_oxides), 1.0 / 8.0)
    composition[8:] = 0.0
    options = {
        "pack": pack,
        "basis_type": "mol",
        "enable_sp_extension": True,
        "allow_extrapolation": True,
        "allow_out_of_envelope": True,
        "max_iter": 0,
    }

    with pytest.raises(ImccNonconvergenceError) as original:
        model.solve_imcc_sf04(
            composition,
            1500.0,
            pack.kernel_datapack,
            basis=float(composition.sum()),
            allow_extrapolation=True,
            max_iter=0,
        )
    with pytest.raises(ImccNonconvergenceError, match="did not run") as caught:
        evaluate(composition, 1500.0, **options)

    assert caught.value.diagnostics == original.value.diagnostics


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
