from __future__ import annotations

import gc
import math
import os
import platform
import statistics
import subprocess
import sys
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


# Cold-cost baselines keyed by CPU model. Each fixture stores the five
# independent-run medians of redox CPU per Newton iteration divided by a fixed
# legacy-solver reference call's CPU. See the gate for the spread allowance.
# The key is the public chip name (on Darwin, `sysctl -n machdep.cpu.brand_string`),
# never a host name: the same chip shares a ceiling, and a different chip does not.
# Recorded after the cold-start fix on the development laptop's chip.
_DEV_CPU_CLASS = "Apple M5 Max"
_COLD_REFERENCE_COMPOSITION = {
    "SiO2": 0.50,
    "MgO": 0.15,
    "FeO": 0.15,
    "CaO": 0.10,
    "Al2O3": 0.08,
    "TiO2": 0.01,
    "Na2O": 0.008,
    "K2O": 0.002,
}
_COLD_REFERENCE_TEMPERATURE = 1734.15
_COLD_COST_BASELINES: dict[str, dict[str, tuple[float, ...]]] = {
    # Values are five independent run medians. Each run is the median of five
    # trial medians, each from seven interleaved solve/reference pairs. Ratios
    # are redox CPU per Newton iteration divided by reference-solver CPU.
    _DEV_CPU_CLASS: {
        "basalt": (0.00378906, 0.00373555, 0.00372911, 0.00368432, 0.00366756),
        "basalt_with_traces": (
            0.00481097,
            0.00479506,
            0.00472397,
            0.00490507,
            0.00490697,
        ),
        "closed_natural_melt": (
            0.00553817,
            0.00566824,
            0.00566409,
            0.00572298,
            0.00576032,
        ),
    },
    "Apple M3 Ultra": {
        "basalt": (0.00384046, 0.00368501, 0.00375076, 0.00372141, 0.00375380),
        "basalt_with_traces": (
            0.00483423,
            0.00483398,
            0.00414773,
            0.00448994,
            0.00448712,
        ),
        "closed_natural_melt": (
            0.00531161,
            0.00539129,
            0.00608211,
            0.00542010,
            0.00583425,
        ),
    },
}


def _cpu_class() -> str:
    """Public CPU model used as the cold-cost machine class."""
    if sys.platform == "darwin":
        try:
            brand = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            brand = ""
        if brand:
            return brand
    return platform.machine() or "unknown"


def _cold_baseline(cpu_class: str, name: str) -> tuple[float, float] | None:
    """Return the median and range of five recorded ratios for this chip.

    An unknown chip or fixture returns None. Callers must not substitute
    another class's numbers: that would silently apply the wrong ceiling.
    """
    run_values = _COLD_COST_BASELINES.get(cpu_class, {}).get(name)
    if run_values is None or len(run_values) != 5:
        return None
    return statistics.median(run_values), max(run_values) - min(run_values)


def _missing_cold_baseline_message(
    cpu_class: str,
    name: str,
    ratio_median: float,
    ratio_spread: float,
    trial_medians: list[float],
    reference_cpu_median: float,
    wall_max: float,
    wall_cpu_ratio: float,
    newton_iterations: int,
    solver_residual_evaluations: int,
    load_start: float,
    load_end: float,
) -> str:
    return (
        f"{name}: unrecorded paired baseline for {cpu_class!r}; "
        f"normalized ratio median={ratio_median:.8f}, spread={ratio_spread:.8f}, "
        f"five trial medians={trial_medians}; "
        f"reference CPU={reference_cpu_median * 1000:.3f} ms; "
        f"max wall={wall_max * 1000:.3f} ms, wall/CPU={wall_cpu_ratio:.3f}; "
        f"Newton iterations={newton_iterations}; "
        f"solver residual evaluations={solver_residual_evaluations}; "
        f"loadavg={load_start:.2f}/{load_end:.2f}. "
        "Add five independent run medians to _COLD_COST_BASELINES for this chip."
    )


def _cold_cost_ceiling(
    baseline_median: float, baseline_spread: float, measured_spread: float
) -> float:
    return min(
        baseline_median + 2.0 * (baseline_spread + measured_spread),
        1.9 * baseline_median,
    )


def test_cold_cost_baseline_lookup_does_not_cross_cpu_classes() -> None:
    assert all(
        len(values) == 5
        for entries in _COLD_COST_BASELINES.values()
        for values in entries.values()
    )
    values = _COLD_COST_BASELINES[_DEV_CPU_CLASS]["basalt"]
    assert _cold_baseline(_DEV_CPU_CLASS, "basalt") == (
        statistics.median(values),
        max(values) - min(values),
    )
    assert _cold_baseline("not-a-recorded-chip", "basalt") is None
    assert _cold_baseline(_DEV_CPU_CLASS, "missing-fixture") is None


def test_cold_cost_budget_caps_at_a_1_9_ratio() -> None:
    assert _cold_cost_ceiling(0.001, 0.0, 0.0) == 0.001
    noisy_ceiling = _cold_cost_ceiling(0.001, 0.01, 0.01)
    assert noisy_ceiling == 0.0019
    assert 2 * 0.001 > noisy_ceiling

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
    ("name", "inventory", "temperature", "mode"),
    [
        (
            "basalt",
            _COLD_BASALT_INVENTORY,
            1734.15,
            "imposed",
        ),
        (
            "basalt_with_traces",
            _COLD_TRACE_BASALT_INVENTORY,
            1734.15,
            "imposed",
        ),
        (
            "closed_natural_melt",
            _CLOSED_NATURAL_MELT_INVENTORY,
            1572.15,
            "closed",
        ),
    ],
)
def test_cold_redox_solver_cost_against_recorded_baseline(
    name: str,
    inventory: dict[str, float],
    temperature: float,
    mode: Literal["closed", "imposed"],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare cold CPU per Newton iteration to a paired legacy solve.

    Each call is paired back-to-back with the same fixed legacy least-squares
    workload in the same process. Five trial medians, each from seven pairs,
    are compared with five independent baseline-run medians. The allowance is
    twice the sum of baseline and measured ranges, capped at 1.9x; therefore
    even the smallest guaranteed detected slowdown is 1.9x. Wall/CPU and load
    are reported as tripwires, not pass criteria.
    """
    trials = 5
    repetitions_per_trial = 7
    trial_medians: list[float] = []
    wall_samples: list[float] = []
    solve_cpu_samples: list[float] = []
    reference_cpu_samples: list[float] = []
    newton_iterations = 0
    solver_residual_evaluations = 0
    import openimcc.redox as redox_module
    import openimcc.kernel as kernel_module

    original_solve_active_warm = redox_module.solve_active_warm

    def count_newton_iterations(*args, **kwargs):
        nonlocal newton_iterations
        solution = original_solve_active_warm(*args, **kwargs)
        newton_iterations += int(solution[3])
        return solution

    monkeypatch.setattr(redox_module, "solve_active_warm", count_newton_iterations)

    original_active_residual = kernel_module._active_residual

    def count_solver_residual_evaluations(*args, **kwargs):
        nonlocal solver_residual_evaluations
        solver_residual_evaluations += 1
        return original_active_residual(*args, **kwargs)

    monkeypatch.setattr(
        kernel_module, "_active_residual", count_solver_residual_evaluations
    )
    load_start = os.getloadavg()[0]

    def measure_solve() -> tuple[float, int]:
        nonlocal newton_iterations, solver_residual_evaluations
        iterations_before = newton_iterations
        residual_evaluations_before = solver_residual_evaluations
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
        elapsed = time.process_time() - cpu_started
        iterations = newton_iterations - iterations_before
        call_residual_evaluations = (
            solver_residual_evaluations - residual_evaluations_before
        )
        assert iterations > 0, (
            f"{name}: evaluate_redox call performed no Newton iterations"
        )
        assert call_residual_evaluations > 0, (
            f"{name}: evaluate_redox call performed no Newton residual evaluations"
        )
        assert result.solver_path == "newton"
        assert result.fallback_fired is False
        if name == "closed_natural_melt":
            assert abs(result.oxygen_residual_mol) <= 1.0e-10
        return elapsed, iterations

    def measure_reference() -> float:
        started = time.process_time()
        evaluate(
            _COLD_REFERENCE_COMPOSITION,
            _COLD_REFERENCE_TEMPERATURE,
            basis_type="mol",
            allow_out_of_envelope=True,
        )
        return time.process_time() - started

    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for trial_index in range(trials):
            trial_ratios: list[float] = []
            for pair_index in range(repetitions_per_trial):
                # Alternate order to avoid consistently favoring one solver
                # as the process heats or the scheduler changes core class.
                if (trial_index * repetitions_per_trial + pair_index) % 2:
                    reference_cpu = measure_reference()
                    solve_cpu, iterations = measure_solve()
                else:
                    solve_cpu, iterations = measure_solve()
                    reference_cpu = measure_reference()
                solve_cpu_samples.append(solve_cpu)
                reference_cpu_samples.append(reference_cpu)
                trial_ratios.append(solve_cpu / (reference_cpu * iterations))
            trial_medians.append(statistics.median(trial_ratios))
    finally:
        if gc_was_enabled:
            gc.enable()

    ratio_median = statistics.median(trial_medians)
    ratio_spread = max(trial_medians) - min(trial_medians)
    load_end = os.getloadavg()[0]
    wall_max = max(wall_samples)
    solve_cpu_median = statistics.median(solve_cpu_samples)
    wall_cpu_ratio = wall_max / solve_cpu_median
    reference_cpu_median = statistics.median(reference_cpu_samples)
    cpu_class = _cpu_class()
    recorded = _cold_baseline(cpu_class, name)
    if recorded is None:
        message = _missing_cold_baseline_message(
            cpu_class,
            name,
            ratio_median,
            ratio_spread,
            trial_medians,
            reference_cpu_median,
            wall_max,
            wall_cpu_ratio,
            newton_iterations,
            solver_residual_evaluations,
            load_start,
            load_end,
        )
        print(message)
        pytest.skip(message)
    baseline_median, baseline_spread = recorded
    ceiling = _cold_cost_ceiling(baseline_median, baseline_spread, ratio_spread)
    baseline_values = _COLD_COST_BASELINES[cpu_class][name]
    print(
        f"{name}: paired redox CPU/reference CPU/Newton iteration median="
        f"{ratio_median:.8f} spread={ratio_spread:.8f}; "
        f"baseline median={baseline_median:.8f} spread={baseline_spread:.8f} "
        f"run ratios={baseline_values}; ceiling={ceiling:.8f}; "
        f"reference CPU median={reference_cpu_median * 1000:.3f} ms; "
        f"redox CPU median={solve_cpu_median * 1000:.3f} ms; "
        f"max wall={wall_max * 1000:.3f} ms; "
        f"max wall/median CPU={wall_cpu_ratio:.3f}; "
        f"Newton iterations={newton_iterations}; "
        f"solver residual evaluations={solver_residual_evaluations}; "
        f"loadavg start/end={load_start:.2f}/{load_end:.2f}"
    )
    assert ratio_median <= ceiling, (
        f"{name}: paired ratio median {ratio_median:.8f} exceeds "
        f"{ceiling:.8f} ceiling; spread={ratio_spread:.8f}; "
        f"max wall={wall_max * 1000:.3f} ms; "
        f"max wall/median CPU={wall_cpu_ratio:.3f}; "
        f"reference CPU median={reference_cpu_median * 1000:.3f} ms; "
        f"Newton iterations={newton_iterations}; "
        f"solver residual evaluations={solver_residual_evaluations}; "
        f"loadavg start/end={load_start:.2f}/{load_end:.2f}"
    )


# Fe saturation pins follow the direct JANAF Fe refit; the previous Fe
# reference rows missed their JANAF source by up to 79 J/mol (a fusion-anchor
# offset on Fe(l); a single polynomial across the Fe(cr) delta phase).
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
            -14.73700130009624,
            2,
        ),
        (
            "saturation upper endpoint probe",
            {"Fe": 1.0, "O": 1.0},
            2200.0,
            "imposed",
            -14.73700130009624,
            4,
        ),
        (
            "invariant extent bisection",
            {"Fe": 1.0, "O": 1.0},
            2200.0,
            "imposed",
            -14.73700130009624,
            51,
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


def test_fallback_flag_collects_kernel_bit_during_invariant_extent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import openimcc.redox as redox_module

    solve_active_warm = redox_module.solve_active_warm
    selected_call = 15
    seen_calls = 0
    marked_calls: list[int] = []

    def mark_kernel_fallback(*args, **kwargs):
        nonlocal seen_calls
        call_index = seen_calls
        seen_calls += 1
        result = solve_active_warm(*args, **kwargs)
        if call_index == selected_call:
            marked_calls.append(call_index)
            return (*result[:-1], True)
        return result

    monkeypatch.setattr(redox_module, "solve_active_warm", mark_kernel_fallback)
    result = evaluate_redox(
        {"Fe": 1.0, "O": 1.0},
        2200.0,
        "imposed",
        lambda_imposed=-14.73700130009624,
    )

    assert seen_calls > selected_call
    assert marked_calls == [selected_call]
    assert result.fallback_fired is True


@pytest.mark.parametrize(
    ("name", "composition", "temperature", "basis_type", "metal_bearing"),
    [
        (
            "metal_free_common_melt",
            _COLD_REFERENCE_COMPOSITION,
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
    """Keep closed CPU cost within 1.3x of legacy on the same case.

    This is a relative budget against the legacy solver, not a regression
    budget against prior closed-solver runtime. The two-scaled-MAD allowance
    is capped at 0.15, so the largest ceiling is 1.45 and any ratio of 1.46
    or greater is guaranteed to fail regardless of co-tenant spread.
    """
    repetitions = 9
    # At the measured 5–13 ms per-call cost, k=20 makes each timed batch
    # roughly 0.1–0.25 CPU seconds, well above timer granularity and short
    # scheduler interruptions.
    calls_per_sample = 20
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

    pair_ratios: list[float] = []
    legacy_cpu: list[float] = []
    redox_cpu: list[float] = []
    redox_wall: list[float] = []
    pair_wall: list[float] = []
    load_start = os.getloadavg()[0]
    wall_started = time.perf_counter()

    def measure_legacy_batch() -> float:
        started = time.process_time()
        for _ in range(calls_per_sample):
            evaluate(
                composition,
                temperature,
                basis_type=basis_type,
                allow_out_of_envelope=True,
            )
        return time.process_time() - started

    def measure_closed_batch() -> tuple[float, float]:
        cpu_started = time.process_time()
        wall_started = time.perf_counter()
        for _ in range(calls_per_sample):
            result = evaluate_redox(
                inventory,
                temperature,
                "closed",
                warm_start=warm_start,
            )
            assert result.metal_buffered is metal_bearing
        return time.process_time() - cpu_started, time.perf_counter() - wall_started

    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for pair_index in range(repetitions):
            pair_started = time.perf_counter()
            if pair_index % 2 == 0:
                legacy_elapsed = measure_legacy_batch()
                closed_elapsed, closed_wall = measure_closed_batch()
            else:
                closed_elapsed, closed_wall = measure_closed_batch()
                legacy_elapsed = measure_legacy_batch()
            redox_wall.append(closed_wall)
            legacy_cpu.append(legacy_elapsed)
            redox_cpu.append(closed_elapsed)
            pair_ratios.append(closed_elapsed / legacy_elapsed)
            pair_wall.append(time.perf_counter() - pair_started)
    finally:
        if gc_was_enabled:
            gc.enable()

    load_end = os.getloadavg()[0]
    total_wall = time.perf_counter() - wall_started
    ratio = statistics.median(pair_ratios)
    mad = statistics.median(abs(value - ratio) for value in pair_ratios)
    robust_spread = 1.4826 * mad
    raw_noise_margin = 2.0 * robust_spread
    noise_margin = min(raw_noise_margin, 0.15)
    legacy_median = statistics.median(legacy_cpu) / calls_per_sample
    redox_median = statistics.median(redox_cpu) / calls_per_sample
    max_wall = max(redox_wall)
    wall_cpu_ratio = max_wall / (redox_median * calls_per_sample)
    print(
        f"{name}: {repetitions} paired samples x k={calls_per_sample}; "
        f"median CPU ratio={ratio:.3f}; scaled MAD={robust_spread:.3f}; "
        f"raw/capped noise margin={raw_noise_margin:.3f}/{noise_margin:.3f}; "
        f"legacy CPU median={legacy_median * 1000:.3f} ms; "
        f"closed CPU median={redox_median * 1000:.3f} ms; "
        f"total wall={total_wall:.3f} s; max paired wall={max(pair_wall) * 1000:.3f} ms; "
        f"max closed wall={max_wall * 1000:.3f} ms; "
        f"max wall/median CPU={wall_cpu_ratio:.3f}; "
        f"load average start/end={load_start:.2f}/{load_end:.2f}"
    )
    # The capped allowance admits noisy paired samples without allowing a
    # ratio of 1.46 or more to pass.
    ceiling = 1.3 + noise_margin
    assert ratio <= ceiling
