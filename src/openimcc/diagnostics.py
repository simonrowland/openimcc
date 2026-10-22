"""Reported derived-property checks that never select data or coefficients.

Data choice remains based on source fidelity. These comparisons describe what a
loaded pack implies and must not feed coefficient fitting.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from openimcc.model import ImccLoadedDatapack, evaluate, load_datapack

R = 8.31446261815324
_PARENT_CP = {
    # Printed constant liquid branches of the JANAF parent standards used here:
    # SiO2 85.772 from 1800 K, Na2O 104.600 at liquid nodes, MgO 66.944
    # from 2200 K, and CaO 62.760 from 2200 K; continued as supercooled liquids.
    "SiO2": 85.772, "Na2O": 104.600, "MgO": 66.944, "CaO": 62.760,
}

def implied_heat_capacity(
    pack: ImccLoadedDatapack,
    composition: Mapping[str, float],
    temperatures: Sequence[float],
    *,
    step_K: float = 1.0,
) -> list[dict[str, Any]]:
    """Return parent, configurational and total Cp, with h and h/2 checks.

    This is a report only and never a rule for choosing data or coefficients.
    Derivation: with log10 K=A+B/T, ΔrH=−R ln(10)B and ΔrCp=0. For
    equilibrium species amounts n_j=complex_x[j]/D per mole analytical oxide,
    Cp_conf=Σ ΔrH_j dn_j/dT. Central differences supply dn/dT. ΔrH is J/mol
    reaction and dn/dT is mol reaction/mol analytical oxide/K, so Cp_conf is
    J/mol analytical oxide/K. With no associations the sum is empty and Cp_conf=0.
    """
    if not math.isfinite(step_K) or step_K <= 0:
        raise ValueError("step_K must be a finite positive number")
    normalized = {name: float(value) for name, value in composition.items()}
    total = sum(normalized.values())
    if total <= 0 or any(value < 0 for value in normalized.values()):
        raise ValueError("composition values must be non-negative with positive sum")
    missing_cp = sorted(set(normalized) - set(_PARENT_CP))
    if missing_cp:
        raise ValueError(f"no liquid parent-standard Cp recorded for {missing_cp}")
    parent_cp = sum(
        value / total * _PARENT_CP[name] for name, value in normalized.items()
    )
    kernel_pack = pack.kernel_datapack
    reaction_enthalpies = -R * math.log(10.0) * np.asarray(kernel_pack.B)

    def evaluate_state(temp: float) -> Any:
        return evaluate(
            normalized, temp, pack, basis_type="mol", allow_extrapolation=True,
            allow_out_of_envelope=True,
        )

    def amounts(temp: float) -> np.ndarray:
        result = evaluate_state(temp)
        return np.asarray(result.complex_x) / result.D

    output = []
    for temp in temperatures:
        t = float(temp)
        if not math.isfinite(t) or t <= step_K:
            raise ValueError("temperatures must be finite and greater than step_K")

        def calculate(h: float) -> tuple[float, list[float]]:
            derivative = (amounts(t + h) - amounts(t - h)) / (2.0 * h)
            contributions = reaction_enthalpies * derivative
            return float(np.sum(contributions)), contributions.tolist()

        cp_h, contributions_h = calculate(step_K)
        cp_half, contributions_half = calculate(step_K / 2.0)
        state = evaluate_state(t)
        flags = list(state.labels.flags)
        notices = list(state.labels.notices)
        output.append({
            "composition": dict(normalized), "temperature_K": t,
            "cp_parent_J_mol_K": parent_cp,
            "cp_conf_J_mol_K": cp_h,
            "cp_total_J_mol_K": parent_cp + cp_h,
            "cp_conf_half_step_J_mol_K": cp_half,
            "step_K": step_K,
            "half_step_K": step_K / 2.0,
            "step_check_delta_J_mol_K": cp_half - cp_h,
            "extrapolated": bool(state.extrapolated),
            "paper_demonstrated_window": any(
                "paper-demonstrated-window" in flag for flag in flags
            ),
            "flags": flags,
            "notices": notices,
            "association_cp_J_mol_K": dict(zip(kernel_pack.reactions, contributions_h)),
            "association_cp_half_step_J_mol_K": dict(zip(kernel_pack.reactions, contributions_half)),
        })
    return output


def score_against_calorimetry(
    results: Sequence[Mapping[str, Any]], dataset: Mapping[str, Any]
) -> dict[str, Any]:
    """Score only absolute total, composition curvature, T-flatness and controls."""
    entries = dataset["compositions"]
    def composition_key(composition: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
        total = sum(float(value) for value in composition.values())
        return tuple(sorted((name, round(float(value) / total, 12))
                            for name, value in composition.items() if value))

    entries_by_composition = {
        composition_key(entry["composition_mol"]): entry["id"]
        for entry in entries if "composition_mol" in entry
    }
    rows_by_id: dict[str, list[Mapping[str, Any]]] = {}
    for row in results:
        row_id = row.get("dataset_id")
        if row_id is None and "composition" in row:
            row_id = entries_by_composition.get(composition_key(row["composition"]))
        if row_id is not None:
            rows_by_id.setdefault(str(row_id), []).append(row)
    tolerance = float(dataset["envelope"]["relative"])
    absolute = []
    for entry in entries:
        cp_measured = entry.get("measured_cp_J_mol_K")
        if cp_measured is None:
            continue
        for row in rows_by_id.get(entry["id"], []):
            error = float(row["cp_total_J_mol_K"]) - float(cp_measured)
            band = abs(float(cp_measured)) * tolerance
            absolute.append({"composition": entry["id"], "temperature_K": row["temperature_K"],
                             "residual_J_mol_K": error, "envelope_J_mol_K": band,
                             "verdict": "(a) PASS" if abs(error) <= band else "(a) OUTSIDE ENVELOPE"})

    curvature = []
    temperatures = sorted({float(row["temperature_K"]) for row in results})
    usable = [entry for entry in entries if entry.get("measured_cp_J_mol_K") is not None]
    for temp in temperatures:
        points = [next((r for r in rows_by_id.get(e["id"], [])
                        if float(r["temperature_K"]) == temp), None) for e in usable]
        points = [(float(e["x_coordinate"]), float(r["cp_conf_J_mol_K"]))
                  for e, r in zip(usable, points) if r is not None]
        if len(points) < 2:
            continue
        x, y = np.asarray(points).T
        residuals = y - np.polyval(np.polyfit(x, y, 1), x)
        for entry, (_, residual) in zip([e for e in usable if any(
                float(r["temperature_K"]) == temp for r in rows_by_id.get(e["id"], []))],
                zip(x, residuals)):
            band = float(entry["measured_cp_J_mol_K"]) * tolerance
            curvature.append({"composition": entry["id"], "temperature_K": temp,
                              "residual_J_mol_K": float(residual), "envelope_J_mol_K": band,
                              "verdict": "(b) PASS" if abs(residual) <= band else "(b) CURVATURE OUTSIDE ENVELOPE"})

    t_flatness = []
    if dataset.get("temperature_independent"):
        lo, hi = map(float, dataset["temperature_interval_K"])
        for entry in usable:
            matched = [r for r in rows_by_id.get(entry["id"], [])
                       if lo <= float(r["temperature_K"]) <= hi]
            if len(matched) < 2:
                continue
            values = [float(r["cp_conf_J_mol_K"]) for r in matched]
            mean = float(np.mean(values))
            deviation = max(abs(value - mean) for value in values)
            band = float(entry["measured_cp_J_mol_K"]) * tolerance
            t_flatness.append({"composition": entry["id"], "max_deviation_J_mol_K": deviation,
                               "envelope_J_mol_K": band,
                               "verdict": "(c) PASS" if deviation <= band else "(c) T-DEPENDENCE OUTSIDE ENVELOPE"})

    controls = []
    for entry in entries:
        if not entry.get("pure_end_member"):
            continue
        for row in rows_by_id.get(entry["id"], []):
            measured = entry.get("measured_cp_J_mol_K")
            if measured is not None:
                controls.append({"composition": entry["id"], "temperature_K": row["temperature_K"],
                                "standard_state_offset_J_mol_K": float(row["cp_total_J_mol_K"]) - float(measured),
                                "verdict": "(d) STANDARD-STATE OFFSET"})
    return {"absolute_total": absolute, "curvature": curvature,
            "temperature_flatness": t_flatness, "pure_end_member_controls": controls}


def load_calorimetry_dataset() -> dict[str, Any]:
    path = resources.files("openimcc").joinpath("data/diagnostics/liquid-cp-calorimetry.json")
    return json.loads(path.read_text(encoding="utf-8"))["datasets"][0]


def run_cp_implied(pack_path: str | Path | None = None) -> tuple[ImccLoadedDatapack, list[dict[str, Any]], dict[str, Any]]:
    """Evaluate the packaged sodium calorimetry points and produce scored rows."""
    if pack_path is None:
        pack_path = resources.files("openimcc").joinpath("data/packs/imcc-sf04-d066-v3.json")
    pack = load_datapack(str(pack_path))
    dataset = load_calorimetry_dataset()
    temperatures = (900., 1323., 1500., 1700., 1800.)
    results: list[dict[str, Any]] = []
    for entry in dataset["compositions"]:
        values = implied_heat_capacity(pack, entry["composition_mol"], temperatures)
        for row in values:
            results.append({"dataset_id": entry["id"], **row})
    return pack, results, score_against_calorimetry(results, dataset)
