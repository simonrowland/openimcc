from __future__ import annotations

import json
from pathlib import Path
import statistics
import time

from openimcc import evaluate, evaluate_redox
from openimcc.redox_pack import load_redox_pack


BASELINE_PATH = Path(__file__).with_name("redox_cost_baseline.json")


def _element_inventory(composition: dict[str, float], T_K: float, basis_type: str) -> dict[str, float]:
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
    for oxide, amount in zip(legacy.parent_oxides, legacy.parent_mol):
        for element, count in records[oxide]["parent_formula_atoms"].items():
            if element != "O" and amount:
                elements[element] = elements.get(element, 0.0) + float(amount) * float(count)
    return elements


def test_warm_numpy_path_stays_within_recorded_cpu_spread() -> None:
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    fixtures = {
        "common_melt_3000k": (
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
            "direct",
        ),
        "oprl2n_2200k": (
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
            "continuation",
        ),
    }
    assert baseline["repetitions"] >= 5

    for name, (composition, temperature, basis_type, expected_legacy_path) in fixtures.items():
        legacy = evaluate(
            composition,
            temperature,
            basis_type=basis_type,
            allow_out_of_envelope=True,
        )
        assert legacy.convergence.solver_path == expected_legacy_path
        inventory = _element_inventory(composition, temperature, basis_type)
        cold = evaluate_redox(
            inventory, temperature, "imposed", lambda_imposed=0.0
        )
        assert cold.fallback_fired is baseline["fixtures"][name]["new_numpy_warm"][
            "cold_seed_fallback_fired"
        ]

        warm_start = cold.warm_start
        cpu_seconds: list[float] = []
        wall_seconds: list[float] = []
        fallback_count = 0
        for _ in range(int(baseline["repetitions"])):
            cpu_start = time.process_time()
            wall_start = time.perf_counter()
            result = evaluate_redox(
                inventory,
                temperature,
                "imposed",
                lambda_imposed=0.0,
                warm_start=warm_start,
            )
            wall_seconds.append(time.perf_counter() - wall_start)
            cpu_seconds.append(time.process_time() - cpu_start)
            fallback_count += int(result.fallback_fired)
            assert result.solver_path == baseline["fixtures"][name]["new_numpy_warm"][
                "solver_path"
            ]

        measured_median_ms = statistics.median(cpu_seconds) * 1000.0
        recorded = baseline["fixtures"][name]["new_numpy_warm"]
        assert measured_median_ms <= recorded["median_cpu_ms"] + recorded[
            "spread_cpu_ms"
        ]
        assert max(wall_seconds) * 1000.0 <= baseline["wall_tripwire_ms"]
        assert fallback_count / len(cpu_seconds) <= recorded["fallback_rate"]
