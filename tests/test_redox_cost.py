from __future__ import annotations

import gc
import math
import statistics
import time
from typing import Literal

import pytest

from openimcc import evaluate, evaluate_redox
from openimcc.redox_pack import load_redox_pack


def _element_inventory(
    composition: dict[str, float], T_K: float, basis_type: str, *, metal_bearing: bool
) -> dict[str, float]:
    legacy = evaluate(
        composition,
        T_K,
        basis_type=basis_type,
        allow_out_of_envelope=True,
    )
    records = {
        str(parent["parent_oxide"]).removesuffix("(l)"): parent
        for parent in load_redox_pack().parents
    }
    elements: dict[str, float] = {}
    oxygen = 0.0
    for oxide, amount in zip(legacy.parent_oxides, legacy.parent_mol):
        atoms = records[oxide]["parent_formula_atoms"]
        for element, count in atoms.items():
            if element == "O":
                oxygen += float(amount) * float(count)
            elif amount:
                elements[element] = elements.get(element, 0.0) + float(amount) * float(count)
    if metal_bearing:
        oxygen -= 0.25 * elements["Fe"]
    elements["O"] = oxygen
    return elements


# Measured after the cold-start fix: median process CPU and max-minus-min
# spread over seven cold calls per fixture, with no warm start. The median and
# spread both contribute to a ceiling that rejects a return to fallback costs.
_COLD_COST_BASELINES = {
    "basalt": (0.009259, 0.027440),
    "basalt_with_traces": (0.009416, 0.001370),
    "closed_natural_melt": (0.022004, 0.003787),
}

_COLD_BASALT_INVENTORY = {
    "Si": 0.855,
    "Ti": 0.0238,
    "Al": 0.275,
    "Fe": 0.1576,
    "Mg": 0.172,
    "Ca": 0.187,
    "Na": 0.0739,
    "K": 0.0153,
    "O": 2.66455,
}
_COLD_TRACE_BASALT_INVENTORY = {
    **_COLD_BASALT_INVENTORY,
    "Mn": 0.0035,
    "P": 0.0028,
    "Cr": 0.00026,
    "V": 0.0008,
}
# Derivation: convert measured oxide weights to moles with their molar masses,
# then sum elemental atom counts; O is stored as moles of oxygen atoms.
_CLOSED_NATURAL_MELT_INVENTORY = {
    "Si": 0.855797604365866,
    "Ti": 0.025041945258307666,
    "Al": 0.27696782995116775,
    "Fe": 0.15520533538813916,
    "Cr": 0.0002631745162852391,
    "Mn": 0.0031013259578163282,
    "Mg": 0.17839243358045276,
    "Ca": 0.20329045212509855,
    "Na": 0.08551297296337947,
    "K": 0.004458788058940932,
    "P": 0.0033816033731493647,
    "V": 0.0010675121362785994,
    "O": 2.7862705529228937,
}


@pytest.mark.parametrize(
    ("name", "inventory", "temperature", "mode", "baseline"),
    [
        (
            "basalt",
            _COLD_BASALT_INVENTORY,
            1734.15,
            "imposed",
            _COLD_COST_BASELINES["basalt"],
        ),
        (
            "basalt_with_traces",
            _COLD_TRACE_BASALT_INVENTORY,
            1734.15,
            "imposed",
            _COLD_COST_BASELINES["basalt_with_traces"],
        ),
        (
            "closed_natural_melt",
            _CLOSED_NATURAL_MELT_INVENTORY,
            1572.15,
            "closed",
            _COLD_COST_BASELINES["closed_natural_melt"],
        ),
    ],
)
def test_cold_redox_solver_cost_against_recorded_baseline(
    name: str,
    inventory: dict[str, float],
    temperature: float,
    mode: Literal["closed", "imposed"],
    baseline: tuple[float, float],
) -> None:
    repetitions = 7
    baseline_median, baseline_spread = baseline
    cpu_samples: list[float] = []
    wall_samples: list[float] = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repetitions):
            cpu_started = time.process_time()
            wall_started = time.perf_counter()
            result = evaluate_redox(
                inventory,
                temperature,
                mode,
                lambda_imposed=(
                    -7.17 * math.log(10) if mode == "imposed" else None
                ),
            )
            wall_samples.append(time.perf_counter() - wall_started)
            cpu_samples.append(time.process_time() - cpu_started)
            assert result.solver_path == "newton"
            assert result.fallback_fired is False
            if name == "closed_natural_melt":
                assert abs(result.oxygen_residual_mol) <= 1.0e-10
    finally:
        if gc_was_enabled:
            gc.enable()

    median = statistics.median(cpu_samples)
    spread = max(cpu_samples) - min(cpu_samples)
    ceiling = min(
        1.3 * baseline_median + 2.0 * (baseline_spread + spread),
        1.9 * baseline_median,
    )
    print(
        f"{name}: cold CPU median={median * 1000:.3f} ms "
        f"spread={spread * 1000:.3f} ms; baseline="
        f"{baseline_median * 1000:.3f} ms spread="
        f"{baseline_spread * 1000:.3f} ms; ceiling={ceiling * 1000:.3f} ms; "
        f"max wall={max(wall_samples) * 1000:.3f} ms"
    )
    assert median <= ceiling


@pytest.mark.parametrize(
    (
        "name",
        "inventory",
        "temperature",
        "mode",
        "lambda_imposed",
        "inner_call",
    ),
    [
        (
            "imposed liquid",
            _COLD_BASALT_INVENTORY,
            1734.15,
            "imposed",
            -7.17 * math.log(10),
            0,
        ),
        (
            "closed liquid",
            _CLOSED_NATURAL_MELT_INVENTORY,
            1572.15,
            "closed",
            None,
            0,
        ),
        (
            "closed buffered endpoint",
            {"Fe": 1.0, "O": 1.0},
            2200.0,
            "closed",
            None,
            11,
        ),
        (
            "buffered imposed initial liquid",
            _COLD_BASALT_INVENTORY,
            1734.15,
            "imposed",
            -20.0 * math.log(10),
            0,
        ),
        (
            "buffered imposed reduced seed",
            _COLD_BASALT_INVENTORY,
            1734.15,
            "imposed",
            -20.0 * math.log(10),
            1,
        ),
        (
            "buffered imposed bisection",
            _COLD_BASALT_INVENTORY,
            1734.15,
            "imposed",
            -20.0 * math.log(10),
            20,
        ),
        (
            "saturation lower endpoint probe",
            {"Fe": 1.0, "O": 1.0},
            2200.0,
            "imposed",
            -15.127740855106792,
            2,
        ),
        (
            "saturation upper endpoint probe",
            {"Fe": 1.0, "O": 1.0},
            2200.0,
            "imposed",
            -15.127740855106792,
            4,
        ),
        (
            "invariant extent bisection",
            {"Fe": 1.0, "O": 1.0},
            2200.0,
            "imposed",
            -15.127740855106792,
            52,
        ),
    ],
)
def test_fallback_flag_covers_forced_inner_scipy_calls(
    name: str,
    inventory: dict[str, float],
    temperature: float,
    mode: Literal["closed", "imposed"],
    lambda_imposed: float | None,
    inner_call: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np
    import openimcc.redox as redox_module

    solve_active_warm = redox_module.solve_active_warm
    seen_calls = 0
    forced_calls: list[int] = []

    def fail_linear_solve(*args, **kwargs):
        raise np.linalg.LinAlgError("forced redox fallback")

    def force_selected_inner_call(*args, **kwargs):
        nonlocal seen_calls
        call_index = seen_calls
        seen_calls += 1
        if call_index != inner_call:
            return solve_active_warm(*args, **kwargs)
        with monkeypatch.context() as selected_call:
            selected_call.setattr(np.linalg, "solve", fail_linear_solve)
            result = solve_active_warm(*args, **kwargs)
        assert result[-1] is True, (
            f"{name}: selected inner solve did not use SciPy"
        )
        forced_calls.append(call_index)
        return result

    monkeypatch.setattr(redox_module, "solve_active_warm", force_selected_inner_call)
    result = evaluate_redox(
        inventory,
        temperature,
        mode,
        lambda_imposed=lambda_imposed,
    )

    assert seen_calls > inner_call, f"{name}: selected inner call was not reached"
    assert forced_calls == [inner_call]
    assert result.fallback_fired is True
    if name == "invariant extent bisection":
        assert result.metal_buffered is True


@pytest.mark.parametrize(
    ("name", "composition", "temperature", "basis_type", "metal_bearing"),
    [
        (
            "metal_free_common_melt",
            {
                "SiO2": 0.50,
                "MgO": 0.15,
                "FeO": 0.15,
                "CaO": 0.10,
                "Al2O3": 0.08,
                "TiO2": 0.01,
                "Na2O": 0.008,
                "K2O": 0.002,
            },
            3000.0,
            "mol",
            False,
        ),
        (
            "metal_bearing_oprl2n",
            {
                "SiO2": 46.2,
                "TiO2": 5.5,
                "Al2O3": 12.9,
                "FeO": 12.9,
                "MgO": 2.7,
                "CaO": 3.2,
                "Na2O": 3.0,
            },
            2200.0,
            "wt",
            True,
        ),
    ],
)
def test_closed_solver_cost_against_local_legacy_baseline(
    name: str,
    composition: dict[str, float],
    temperature: float,
    basis_type: str,
    metal_bearing: bool,
) -> None:
    repetitions = 7
    inventory = _element_inventory(
        composition, temperature, basis_type, metal_bearing=metal_bearing
    )
    if not metal_bearing:
        inventory["O"] = evaluate_redox(
            inventory, temperature, "imposed", lambda_imposed=0.0
        ).oxygen_mol + 1.0e-3
    first = evaluate_redox(inventory, temperature, "closed")
    assert first.metal_buffered is metal_bearing
    warm_start = first.warm_start

    legacy_cpu: list[float] = []
    redox_cpu: list[float] = []
    redox_wall: list[float] = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repetitions):
            started = time.process_time()
            evaluate(
                composition,
                temperature,
                basis_type=basis_type,
                allow_out_of_envelope=True,
            )
            legacy_cpu.append(time.process_time() - started)

            cpu_started = time.process_time()
            wall_started = time.perf_counter()
            result = evaluate_redox(
                inventory,
                temperature,
                "closed",
                warm_start=warm_start,
            )
            redox_wall.append(time.perf_counter() - wall_started)
            redox_cpu.append(time.process_time() - cpu_started)
            assert result.metal_buffered is metal_bearing
    finally:
        if gc_was_enabled:
            gc.enable()

    legacy_median = statistics.median(legacy_cpu)
    redox_median = statistics.median(redox_cpu)
    legacy_spread = max(legacy_cpu) - min(legacy_cpu)
    redox_spread = max(redox_cpu) - min(redox_cpu)
    ratio = redox_median / legacy_median
    max_wall = max(redox_wall)
    wall_cpu_ratio = max_wall / redox_median
    print(
        f"{name}: legacy CPU median={legacy_median * 1000:.3f} ms "
        f"spread={legacy_spread * 1000:.3f} ms; closed CPU median="
        f"{redox_median * 1000:.3f} ms spread={redox_spread * 1000:.3f} ms; "
        f"ratio={ratio:.3f}; closed max wall={max_wall * 1000:.3f} ms "
        f"max wall/median CPU={wall_cpu_ratio:.3f}"
    )
    if metal_bearing:
        assert ratio <= 1.3
    else:
        # The metal-free case adds an outer oxygen root. Keep its 1.3x median
        # target, with two observed combined ranges as a load-variation margin:
        # each range is the measured max-minus-min over these seven samples.
        # The allowance is capped at 1.9x legacy so a genuine 2x regression
        # (every closed call doing double work) still fails however noisy the
        # run: ceiling = min(1.3*L + 2*(range_L + range_R), 1.9*L).
        combined_spread = legacy_spread + redox_spread
        ceiling = min(
            1.3 * legacy_median + 2.0 * combined_spread, 1.9 * legacy_median
        )
        assert redox_median <= ceiling
