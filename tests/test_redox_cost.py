from __future__ import annotations

import gc
import statistics
import time

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
    assert ratio <= 1.3
