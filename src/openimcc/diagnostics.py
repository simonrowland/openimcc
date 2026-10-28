"""Reported derived-property checks that never select data or coefficients.

Data choice remains based on source fidelity. These comparisons describe what a
loaded pack implies and must not feed coefficient fitting.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from openimcc.model import ImccLoadedDatapack, evaluate, load_datapack

R = 8.31446261815324
_PARENT_CP_BY_PACK = {
    "252f8705a3d04799ab9131836785992ffbedb6e01c98780c849498f4836804f2": {
        # Prior D066 default: JANAF parent liquid branches.
        "SiO2": 85.772, "Na2O": 104.600, "MgO": 66.944, "CaO": 62.760,
    },
    "0879f5a8e94dfd844ed5cfcd900677b496a89e717a85f320b3e0b374ebcdc3b0": {
        # Promoted measured silicate-liquid Cp for SiO2 and Na2O.
        "SiO2": 81.37, "Na2O": 100.6, "MgO": 66.944, "CaO": 62.760,
    },
    "13a185563870ec1a9c20d51bd25fb58df08c9bbbbdf7b2212b0f43ead64738cf": {
        # D066 v5 keeps the measured silicate-liquid Cp standards.
        "SiO2": 81.37, "Na2O": 100.6, "MgO": 66.944, "CaO": 62.760,
    },
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
    binding_digest = getattr(pack, "binding_digest", None)
    parent_cp_by_name = (
        _PARENT_CP_BY_PACK["252f8705a3d04799ab9131836785992ffbedb6e01c98780c849498f4836804f2"]
        if binding_digest is None
        else _PARENT_CP_BY_PACK.get(binding_digest)
    )
    if parent_cp_by_name is None:
        raise ValueError("no liquid parent-standard Cp recorded for this pack identity")
    missing_cp = sorted(set(normalized) - set(parent_cp_by_name))
    if missing_cp:
        raise ValueError(f"no liquid parent-standard Cp recorded for {missing_cp}")
    parent_cp = sum(
        value / total * parent_cp_by_name[name] for name, value in normalized.items()
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
        pack_path = resources.files("openimcc").joinpath("data/packs/imcc-sf04-d066-v5.json")
    pack = load_datapack(str(pack_path))
    dataset = load_calorimetry_dataset()
    temperatures = (900., 1323., 1500., 1700., 1800.)
    results: list[dict[str, Any]] = []
    for entry in dataset["compositions"]:
        values = implied_heat_capacity(pack, entry["composition_mol"], temperatures)
        for row in values:
            results.append({"dataset_id": entry["id"], **row})
    return pack, results, score_against_calorimetry(results, dataset)


def load_mixing_enthalpy_dataset() -> dict[str, Any]:
    path = resources.files("openimcc").joinpath(
        "data/diagnostics/liquid-mixing-enthalpies.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _association_enthalpy(
    pack: ImccLoadedDatapack, composition: Mapping[str, float], temperature_K: float
) -> dict[str, Any]:
    """Return total and association enthalpies with reaction contributions in kJ.

    With log10 K=A+B/T and ΔrCp=0, the van't Hoff relation gives
    ΔrH=−R ln(10) B J/mol complex. Solver ``complex_x`` is a species mole
    fraction; dividing by D gives moles of complex per analytical parent mole.
    Multiplication by total analytical parent moles therefore gives an
    extensive association enthalpy. Active candidate results add Hex/1000 to
    that total; the two contributions remain separately named. Both scale
    linearly with analytical amounts.
    """
    result = evaluate(
        composition, temperature_K, pack, basis_type="mol",
        allow_extrapolation=True, allow_out_of_envelope=True,
    )
    kernel_pack = pack.kernel_datapack
    enthalpies_kJ = -R * math.log(10.0) * np.asarray(kernel_pack.B) / 1000.0
    extents = np.asarray(result.complex_x, dtype=float) / float(result.D)
    scale = sum(float(value) for value in composition.values())
    contributions = extents * enthalpies_kJ * scale
    association_enthalpy_kJ = float(np.sum(contributions))
    excess_interaction_enthalpy_kJ = float(result.excess_enthalpy_J) / 1000.0
    total_enthalpy_kJ = (
        association_enthalpy_kJ + excess_interaction_enthalpy_kJ
        if any(term.active for term in pack.excess_gibbs_terms)
        else association_enthalpy_kJ
    )
    return {
        "total_kJ": total_enthalpy_kJ,
        "per_parent_mol_kJ": total_enthalpy_kJ / scale,
        "association_enthalpy_kJ": association_enthalpy_kJ,
        "association_per_parent_mol_kJ": association_enthalpy_kJ / scale,
        "excess_interaction_enthalpy_kJ": excess_interaction_enthalpy_kJ,
        "breakdown": [
            {"association": name, "extent_mol_per_mol_parent": float(extent),
             "delta_rH_kJ_per_mol_complex": float(dh),
             "contribution_kJ": float(contribution)}
            for name, extent, dh, contribution in zip(
                kernel_pack.reactions, extents, enthalpies_kJ, contributions
            )
        ],
        "evaluation": result,
    }


def _fusion_enthalpy(dataset: Mapping[str, Any], oxide: str, temperature_K: float) -> float:
    nodes = [(float(t), float(h)) for t, h in dataset["fusion_nodes_kJ_mol"][oxide]]
    if not nodes[0][0] <= temperature_K <= nodes[-1][0]:
        raise ValueError(f"no fusion-standard coverage for {oxide} at {temperature_K:g} K")
    for (t0, h0), (t1, h1) in zip(nodes, nodes[1:]):
        if t0 <= temperature_K <= t1:
            return h0 + (h1 - h0) * (temperature_K - t0) / (t1 - t0)
    return nodes[-1][1]


def mixing_enthalpy(
    pack: ImccLoadedDatapack,
    composition: Mapping[str, float],
    temperature_K: float,
    *,
    endmembers: Mapping[str, Mapping[str, float]] | None = None,
    endmember_fractions: Mapping[str, float] | None = None,
    oxide_standard_enthalpies_kJ_mol: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Evaluate enthalpy; optionally subtract liquid endmember endpoints.

    The model's association enthalpy is Σ n_j ΔrH_j. For a mixture of pure
    Σ n_j ΔrH_j. When active, candidate results also include the analytical-
    parent interaction enthalpy. For a mixture of pure silicate endmembers,
    excess enthalpy is H(mix)−Σ X_e H(e) at the same T. Liquid-minus-crystal
    oxide standard terms are linear in parent amounts, so they cancel exactly
    in this endpoint subtraction; they remain explicit for the dissolution
    conversion below. Units are kJ per supplied analytical amount. Scaling
    every component leaves intensive values unchanged while scaling extensive
    enthalpy homogeneously. This does not identify association entropy or
    liquid-standard Gibbs-energy errors.
    """
    normalized = {str(name): float(value) for name, value in composition.items()}
    if (not normalized or any(not math.isfinite(v) or v < 0 for v in normalized.values())
            or sum(normalized.values()) <= 0 or not math.isfinite(temperature_K)):
        raise ValueError("composition must be finite, non-negative, and have positive sum")
    mixture = _association_enthalpy(pack, normalized, float(temperature_K))
    standard = {name: float(value) for name, value in (oxide_standard_enthalpies_kJ_mol or {}).items()}
    mixture_standard = sum(normalized.get(name, 0.0) * value for name, value in standard.items())
    output: dict[str, Any] = {
        "composition_mol": normalized, "temperature_K": float(temperature_K),
        "association_enthalpy_kJ": mixture["association_enthalpy_kJ"],
        "association_enthalpy_kJ_per_mol_parent": mixture["association_per_parent_mol_kJ"],
        "association_breakdown": mixture["breakdown"],
        "extrapolated": bool(mixture["evaluation"].extrapolated),
        "composition_envelope": mixture["evaluation"].labels.envelope_status,
        "flags": list(mixture["evaluation"].labels.flags),
        "notices": list(mixture["evaluation"].labels.notices),
        "liquid_standard_enthalpy_kJ": mixture_standard,
    }
    if any(term.active for term in pack.excess_gibbs_terms):
        output["excess_interaction_enthalpy_kJ"] = mixture[
            "excess_interaction_enthalpy_kJ"
        ]
        output["total_enthalpy_kJ"] = mixture["total_kJ"]
    if endmembers is not None:
        mixture_names = set(normalized)
        if any(set(formula) - mixture_names for formula in endmembers.values()):
            raise ValueError("endmember components must be present in mixture composition")
        fractions = ({name: float(value) for name, value in endmember_fractions.items()}
                     if endmember_fractions is not None else {})
        denom = sum(fractions.values())
        if set(fractions) != set(endmembers) or denom <= 0:
            raise ValueError("endmember proportions must have positive sum")
        fractions = {name: value / denom for name, value in fractions.items()}
        endpoints = {}
        expected_composition: dict[str, float] = {}
        for name, fraction in fractions.items():
            formula = {oxide: float(value) for oxide, value in endmembers[name].items()}
            formula_total = sum(formula.values())
            if formula_total <= 0:
                raise ValueError(f"endmember {name} has no parent components")
            for oxide, value in formula.items():
                expected_composition[oxide] = expected_composition.get(oxide, 0.0) + fraction * value
            endpoint = _association_enthalpy(pack, formula, float(temperature_K))
            endpoint_standard = sum(formula.get(oxide, 0.0) * value for oxide, value in standard.items())
            endpoints[name] = {
                "fraction": fraction,
                "association_enthalpy_kJ_per_formula": endpoint[
                    "association_per_parent_mol_kJ"
                ] * formula_total,
                "standard_enthalpy_kJ_per_formula": endpoint_standard,
                "association_breakdown": endpoint["breakdown"],
            }
            if any(term.active for term in pack.excess_gibbs_terms):
                endpoints[name]["excess_interaction_enthalpy_kJ_per_formula"] = (
                    endpoint["excess_interaction_enthalpy_kJ"]
                )
                endpoints[name]["total_enthalpy_kJ_per_formula"] = endpoint[
                    "total_kJ"
                ]
        actual_total = sum(normalized.values())
        expected_total = sum(expected_composition.values())
        total_formula_moles = actual_total / expected_total
        if any(not math.isclose(normalized.get(oxide, 0.0), total_formula_moles * value,
                                rel_tol=1e-8, abs_tol=1e-10)
               for oxide, value in expected_composition.items()) or any(
                   oxide not in expected_composition and value > 1e-10
                   for oxide, value in normalized.items()
               ):
            raise ValueError("mixture composition does not match the supplied endmember fractions")
        baseline = sum(
            item["fraction"] * (item.get("total_enthalpy_kJ_per_formula",
                                          item["association_enthalpy_kJ_per_formula"])
                                + item["standard_enthalpy_kJ_per_formula"])
            for item in endpoints.values()
        )
        output["endmember_fractions"] = fractions
        output["endmember_formula_moles"] = total_formula_moles
        output["endmember_endpoints"] = endpoints
        output["excess_enthalpy_kJ"] = mixture["total_kJ"] + mixture_standard - baseline * total_formula_moles
        output["excess_enthalpy_kJ_per_mol_endmember"] = output["excess_enthalpy_kJ"] / total_formula_moles
    return output


def run_mixing_enthalpy(
    pack_path: str | Path | None = None,
) -> tuple[ImccLoadedDatapack, list[dict[str, Any]]]:
    """Run reported liquid excess and oxide-dissolution enthalpy comparisons."""
    if pack_path is None:
        pack_path = resources.files("openimcc").joinpath(
            "data/packs/imcc-sf04-d066-v5.json"
        )
    pack = load_datapack(str(pack_path))
    has_active_excess = any(term.active for term in pack.excess_gibbs_terms)
    datasets = load_mixing_enthalpy_dataset()["datasets"]
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        if dataset["type"] == "endmember-liquid-excess-enthalpy":
            for observation in dataset["observations"]:
                weights = observation["weight_percent"]
                amounts = {name: float(weight) / float(dataset["endmembers"][name]["formula_mass_g_mol"])
                           for name, weight in weights.items()}
                total = sum(amounts.values())
                formula_fractions = {name: amount / total for name, amount in amounts.items()}
                parent: dict[str, float] = {}
                for name, fraction in formula_fractions.items():
                    for oxide, coeff in dataset["endmembers"][name]["composition_mol"].items():
                        parent[oxide] = parent.get(oxide, 0.0) + fraction * float(coeff)
                rows.append({
                    "dataset_id": dataset["id"], "observation_id": observation["id"],
                    "measurement_type": dataset["type"], "temperature_K": observation["temperature_K"],
                    "measured_kJ_mol": observation["measured_kJ_mol"],
                    "uncertainty_kJ_mol": observation["uncertainty_kJ_mol"],
                    "composition": parent, "endmember_formula_fractions": formula_fractions,
                    **mixing_enthalpy(
                        pack, parent, observation["temperature_K"],
                        endmembers={name: dataset["endmembers"][name]["composition_mol"]
                                    for name in formula_fractions},
                        endmember_fractions=formula_fractions,
                    ),
                })
                rows[-1]["residual_kJ_mol"] = rows[-1]["excess_enthalpy_kJ_per_mol_endmember"] - rows[-1]["measured_kJ_mol"]
                rows[-1]["standardized_residual"] = rows[-1]["residual_kJ_mol"] / rows[-1]["uncertainty_kJ_mol"]
                rows[-1]["flags"].extend(observation.get("flags", []))
        elif dataset["type"] == "oxide-dissolution-partial-enthalpy":
            composition = dataset["composition_mol_fraction"]
            temp = float(dataset["temperature_K"])
            fusion = {oxide: _fusion_enthalpy(dataset, oxide, temp) for oxide in composition}
            for obs in dataset["observations"]:
                oxide = obs["id"]
                measured = float(obs["measured_solution_kJ_mol_oxide"]) - fusion[oxide]
                h = 1.0e-3
                plus = dict(composition); minus = dict(composition)
                plus[oxide] += h; minus[oxide] -= h
                plus_state = _association_enthalpy(pack, plus, temp)
                minus_state = _association_enthalpy(pack, minus, temp)
                if has_active_excess:
                    hp = plus_state["total_kJ"]
                    hm = minus_state["total_kJ"]
                else:
                    # Keep the legacy arithmetic order byte-identical when
                    # the candidate pack has no active interaction.
                    hp = plus_state["per_parent_mol_kJ"] * sum(plus.values())
                    hm = minus_state["per_parent_mol_kJ"] * sum(minus.values())
                partial = (hp - hm) / (2.0 * h)
                base = _association_enthalpy(pack, composition, temp)
                association_partial = (
                    plus_state["association_enthalpy_kJ"]
                    - minus_state["association_enthalpy_kJ"]
                ) / (2.0 * h)
                row = {
                    "dataset_id": dataset["id"], "observation_id": oxide,
                    "measurement_type": dataset["type"], "temperature_K": temp,
                    "measured_solution_kJ_mol": obs["measured_solution_kJ_mol_oxide"],
                    "uncertainty_kJ_mol": obs["uncertainty_kJ_mol_oxide"],
                    "fusion_enthalpy_kJ_mol": fusion[oxide],
                    "measured_mixing_partial_kJ_mol": measured,
                    "model_mixing_partial_kJ_mol": partial,
                    "residual_kJ_mol": partial - measured,
                    "standardized_residual": (partial - measured) / obs["uncertainty_kJ_mol_oxide"],
                    "composition": dict(composition), "association_enthalpy_kJ_per_mol_parent": base["association_per_parent_mol_kJ"],
                    "association_breakdown": base["breakdown"], "extrapolated": bool(base["evaluation"].extrapolated),
                    "composition_envelope": base["evaluation"].labels.envelope_status,
                    "flags": list(base["evaluation"].labels.flags) + list(obs.get("flags", [])),
                }
                if has_active_excess:
                    row.update({
                        "association_mixing_partial_kJ_mol": association_partial,
                        "excess_interaction_mixing_partial_kJ_mol": (
                            partial - association_partial
                        ),
                        "excess_interaction_enthalpy_kJ_per_mol_parent": (
                            base["excess_interaction_enthalpy_kJ"]
                            / sum(composition.values())
                        ),
                        "total_enthalpy_kJ_per_mol_parent": base["per_parent_mol_kJ"],
                    })
                rows.append(row)
    return pack, rows


_OXIDE_MOLAR_MASS = {
    "SiO2": 60.0843, "MgO": 40.3044, "CaO": 56.0774, "Al2O3": 101.9613,
}


class CrystalStatus(str, Enum):
    AVAILABLE = "available"
    OUT_OF_RANGE = "out-of-range"
    NO_SUPPORTED_CROSSING = "no-supported-crossing"
    NO_SUPPORTED_COMPOSITION = "no-supported-composition"


@dataclass(frozen=True)
class CrystalNotice:
    code: str
    dropped_oxides_wt_percent: Mapping[str, float]
    pack_used: str


@dataclass(frozen=True)
class SaturationIndex:
    phase: str
    formula: str
    parent_stoichiometry: Mapping[str, float]
    source: Mapping[str, Any]
    temperature_range_K: tuple[float, float]
    temperature_K: float
    SI: float | None
    status: CrystalStatus
    solid_solution_bound: bool
    notices: tuple[CrystalNotice, ...] = ()


@dataclass(frozen=True)
class LiquidusBracket:
    status: CrystalStatus
    T_low_K: float
    T_high_K: float
    liquidus_K: float | None
    controlling_phase: str | None
    missing_feed_relevant_phases: tuple[str, ...]
    notices: tuple[CrystalNotice, ...] = ()


@dataclass(frozen=True)
class LiquidOnlyNotice:
    code: str = "crystals-may-coexist"
    message: str = "crystals may coexist; liquid-only model"


def _crystal_thermo_context(condensate: str | Path | None):
    from openimcc.gas import (
        default_condensate_database_path, default_gas_database_path,
        load_gas_datapack,
    )
    oxide_path = Path(condensate) if condensate else default_condensate_database_path()
    return (
        load_gas_datapack(default_gas_database_path(), oxide_path),
        load_gas_datapack(default_gas_database_path(), default_condensate_database_path()),
    )


def _prepare_crystal_composition(composition, pack):
    auto_pack = pack is None
    if pack is None:
        pack = load_datapack()
    values = {str(name): float(value) for name, value in composition.items()}
    if any(not math.isfinite(value) or value < 0.0 for value in values.values()):
        raise ValueError("composition values must be finite and non-negative")
    use_sp_extension = False
    if auto_pack and values.get("P2O5", 0.0) > 0.0:
        extension_path = resources.files("openimcc").joinpath(
            "data/packs/imcc-sf04-d066-ext-v1.json"
        )
        pack = load_datapack(str(extension_path))
        use_sp_extension = True
    elif "P2O5" in pack.extension_parents and values.get("P2O5", 0.0) > 0.0:
        use_sp_extension = True

    supported = set(pack.parent_oxides)
    if use_sp_extension:
        supported.update(pack.extension_parents)
    total = sum(values.values())
    dropped = {
        name: 100.0 * value / total
        for name, value in values.items()
        if value > 0.0 and name not in supported
    } if total > 0.0 else {}
    retained = {name: value for name, value in values.items() if name in supported}
    identity = pack.model_id
    if pack.binding_digest:
        identity = f"{identity}@{pack.binding_digest}"
    notices = ((CrystalNotice("crystal_si_dropped_parents", dropped, identity),)
               if dropped else ())
    return pack, retained, use_sp_extension, notices


def _crystal_si(phase, temp, composition, pack, gas_pack, reference_pack,
                enable_sp_extension=False):
    from openimcc.gas import species_thermo
    bounds = phase["valid_temperature_K"]
    if not float(bounds[0]) <= temp <= float(bounds[1]):
        raise ValueError(f"{phase['id']} crystal function is valid only in {bounds} K")
    dg_kj = _interpolate_crystal_fusion_nodes(phase, temp)
    if gas_pack.condensate_table_digest != reference_pack.condensate_table_digest:
        dg_kj += sum(float(nu) * (
            species_thermo(oxide, "l", temp, gas_pack).G_J_mol
            - species_thermo(oxide, "l", temp, reference_pack).G_J_mol
        ) / 1000.0 for oxide, nu in phase["oxide_stoichiometry"].items())
    weights = composition or {
        oxide: float(nu) * _OXIDE_MOLAR_MASS[oxide]
        for oxide, nu in phase["oxide_stoichiometry"].items()
    }
    melt = evaluate(weights, temp, pack, basis_type="wt", allow_extrapolation=True,
                    allow_out_of_envelope=True,
                    enable_sp_extension=enable_sp_extension)
    dg = dg_kj * 1000.0
    activities = [melt.activity(oxide) for oxide in phase["oxide_stoichiometry"]]
    if any(activity <= 0.0 for activity in activities):
        return {"temperature_K": temp, "SI": -math.inf, "residual_kJ_mol": -math.inf,
                "delta_log10_vapor_product": -math.inf, "deltaG_fusion_J_mol": dg,
                "log10_activity_product": -math.inf}
    logq = sum(float(nu) * math.log10(melt.activity(oxide))
               for oxide, nu in phase["oxide_stoichiometry"].items())
    si = logq + dg / (R * math.log(10.0) * temp)
    residual_kj = (dg + logq * R * math.log(10.0) * temp) / 1000.0
    residual_si = residual_kj * 1000.0 / (R * math.log(10.0) * temp)
    if not math.isclose(si, residual_si, rel_tol=0.0, abs_tol=1e-12):
        raise ArithmeticError(f"SI/residual unit mismatch for {phase['id']} at {temp:g} K")
    return {"temperature_K": temp, "SI": si, "residual_kJ_mol": residual_kj,
            "delta_log10_vapor_product": residual_si, "deltaG_fusion_J_mol": dg,
            "log10_activity_product": logq}


def saturation_indices(composition, T_K, pack=None, *, condensate=None):
    """Return sourced, range-checked crystal saturation indices for a melt."""
    pack, composition, enable_sp_extension, notices = _prepare_crystal_composition(
        composition, pack
    )
    gas_pack, reference_pack = _crystal_thermo_context(condensate)
    has_supported_input = any(value > 0.0 for value in composition.values())
    rows = []
    for phase in load_crystal_dataset()["phases"]:
        lo, hi = map(float, phase["valid_temperature_K"])
        source_nodes = sorted(float(node) for node in phase["fusion_gibbs_nodes_kJ_mol"])
        lo, hi = max(lo, source_nodes[0]), min(hi, source_nodes[-1])
        in_range = lo <= float(T_K) <= hi
        parents_supported = set(phase["oxide_stoichiometry"]).issubset(
            set(pack.parent_oxides) | (set(pack.extension_parents) if enable_sp_extension else set())
        )
        state = (_crystal_si(phase, float(T_K), composition, pack, gas_pack,
                             reference_pack, enable_sp_extension)
                 if in_range and has_supported_input and parents_supported else None)
        if not has_supported_input or not parents_supported:
            status = CrystalStatus.NO_SUPPORTED_COMPOSITION
        elif in_range:
            status = CrystalStatus.AVAILABLE
        else:
            status = CrystalStatus.OUT_OF_RANGE
        rows.append(SaturationIndex(
            phase["id"], phase["formula"], dict(phase["oxide_stoichiometry"]),
            dict(phase["source"]), (lo, hi), float(T_K),
            None if state is None else state["SI"],
            status,
            phase.get("solid_solution_bound", phase["id"] in {
                "forsterite", "enstatite", "diopside", "anorthite",
                "spinel", "akermanite", "gehlenite",
            }),
            notices,
        ))
    return rows


def liquidus_bracket(composition, T_low, T_high, pack=None, *, condensate=None):
    """Bracket the highest supported SI zero, positive on its cooling side."""
    from scipy.optimize import brentq
    if not math.isfinite(T_low) or not math.isfinite(T_high) or T_low >= T_high:
        raise ValueError("require finite T_low < T_high")
    pack, composition, enable_sp_extension, notices = _prepare_crystal_composition(
        composition, pack
    )
    gas_pack, reference_pack = _crystal_thermo_context(condensate)
    missing = {
        "fayalite": ("FeO", "SiO2"), "ferrosilite": ("FeO", "SiO2"),
        "hedenbergite": ("CaO", "FeO", "SiO2"), "hercynite": ("FeO", "Al2O3"),
        "ilmenite": ("FeO", "TiO2"), "ulvospinel": ("FeO", "TiO2"),
        "magnetite": ("FeO",), "perovskite": ("CaO", "TiO2"),
        "armalcolite": ("FeO", "TiO2"),
    }
    available_phase_ids = {row["id"] for row in load_crystal_dataset()["phases"]}
    feed_relevant_missing = tuple(
        phase for phase, parents in missing.items()
        if phase not in available_phase_ids
        and all(float(composition.get(parent, 0.0)) > 0.0 for parent in parents)
    )
    if not any(value > 0.0 for value in composition.values()):
        return LiquidusBracket(CrystalStatus.NO_SUPPORTED_CROSSING, float(T_low),
                               float(T_high), None, None, feed_relevant_missing, notices)
    roots = []
    supported_parents = set(pack.parent_oxides) | (
        set(pack.extension_parents) if enable_sp_extension else set()
    )
    for phase in load_crystal_dataset()["phases"]:
        if not set(phase["oxide_stoichiometry"]).issubset(supported_parents):
            continue
        lo, hi = map(float, phase["valid_temperature_K"])
        source_nodes = sorted(float(node) for node in phase["fusion_gibbs_nodes_kJ_mol"])
        lo, hi = max(lo, source_nodes[0]), min(hi, source_nodes[-1])
        a, b = max(float(T_low), lo), min(float(T_high), hi)
        if a >= b:
            continue
        fn = lambda t: _crystal_si(phase, t, composition, pack, gas_pack, reference_pack,
                                   enable_sp_extension)["SI"]
        nodes = sorted({a, b, *(float(t) for t in phase["source_node_temperatures_K"] if a < float(t) < b)})
        vals = [fn(t) for t in nodes]
        for left, right, yl, yr in zip(nodes, nodes[1:], vals, vals[1:]):
            if yl * yr <= 0:
                roots.append((float(brentq(fn, left, right)), phase["id"], yl > 0))
    cooling_crossings = [row for row in roots if row[2]]
    if cooling_crossings:
        root, phase, _ = max(cooling_crossings)
        return LiquidusBracket(CrystalStatus.AVAILABLE, float(T_low), float(T_high), root,
                               phase, feed_relevant_missing, notices)
    return LiquidusBracket(CrystalStatus.NO_SUPPORTED_CROSSING, float(T_low), float(T_high),
                           None, None, feed_relevant_missing, notices)


def liquid_only_notice(temperature_K, bracket):
    """Return a typed coexistence notice below a known or unsupported liquidus."""
    if (bracket.status is CrystalStatus.NO_SUPPORTED_CROSSING
            or (bracket.liquidus_K is not None and temperature_K < bracket.liquidus_K)):
        return LiquidOnlyNotice()
    return None


def load_crystal_dataset() -> dict[str, Any]:
    path = resources.files("openimcc").joinpath("data/diagnostics/crystal-phases.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _interpolate_crystal_fusion_nodes(phase: Mapping[str, Any], temperature_K: float) -> float:
    nodes = sorted((float(t), float(g)) for t, g in phase["fusion_gibbs_nodes_kJ_mol"].items())
    if temperature_K < nodes[0][0] or temperature_K > nodes[-1][0]:
        raise ValueError(
            f"{phase['id']} fusion function has no source node at {temperature_K:g} K"
        )
    for temp, value in nodes:
        if temp == temperature_K:
            return value
    lo, hi = next((a, b) for a, b in zip(nodes, nodes[1:])
                  if a[0] <= temperature_K <= b[0])
    return lo[1] + (hi[1] - lo[1]) * (temperature_K - lo[0]) / (hi[0] - lo[0])


def _closure_miss(
    residual_kJ_mol: float, temperature_K: float, phase: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Compare the parent-based residual with mineral fusion Gibbs energy.

    Without measured fusion ΔCp, approximate ΔG_fus(T) as
    ΔH_fus(Tm)(1−T/Tm). This is kJ per mole of mineral formula units; dividing
    the miss in J/mol by RT ln(10) gives a dimensionless log10 product.
    """
    fusion = phase.get("independent_fusion")
    if fusion is None:
        return None
    tm = float(fusion["melting_temperature_K"])
    hfus = float(fusion["enthalpy_kJ_mol"])
    delta_g = hfus * (1.0 - temperature_K / tm)
    miss = residual_kJ_mol - delta_g
    return {
        "independent_deltaG_fusion_kJ_mol": delta_g,
        "independent_fusion_method": fusion.get("method", "linear-enthalpy"),
        "independent_fusion_flag": fusion.get(
            "flag", "linear ΔHfus(1 − T/Tm); no measured fusion ΔCp"
        ),
        "closure_miss_kJ_mol": miss,
        "closure_miss_vapor_product": miss * 1000.0 / (R * temperature_K * math.log(10.0)),
    }


def crystal_closure(
    pack: ImccLoadedDatapack,
    condensate_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Report sourced crystal saturation and melting-point closure values.

    For a crystal φ, the balanced chain φ(cr) -> crystalline parent oxides ->
    the melt pack's pure-liquid parent standards cancels Berman's elemental
    reference terms on the first leg. SI = Σν log10(a_i) + ΔG°/(RT ln 10).
    ΔG° is J/mol and the residual is RT ln(10) SI, converted to kJ/mol.
    At a pure oxide's JANAF fusion anchor, a=1 and G(liquid)-G(crystal)=0,
    so SI=0. The residual-derived SI is computed independently as a unit check.
    """
    dataset = load_crystal_dataset()
    gas_pack, reference_pack = _crystal_thermo_context(condensate_path)
    digest = gas_pack.condensate_table_digest

    def phase_state(
        phase: Mapping[str, Any], temp: float,
        composition: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        return _crystal_si(phase, temp, composition, pack, gas_pack, reference_pack)

    results = []
    for phase in dataset["phases"]:
        if phase.get("closure_diagnostic") is False:
            continue
        tm = float(phase["melting_point"]["temperature_K"])
        at_tm = phase_state(phase, tm)
        grid = sorted(set(float(t) for t in phase["source_node_temperatures_K"] if t < tm))
        grid += [tm]
        values = [phase_state(phase, t) for t in grid]
        root = None
        for lo, hi, left, right in zip(grid, grid[1:], values, values[1:]):
            if left["SI"] == 0:
                root = lo; break
            if left["SI"] * right["SI"] <= 0:
                from scipy.optimize import brentq
                root = float(brentq(lambda t: phase_state(phase, t)["SI"], lo, hi))
                break
        in_range = next((v for v in reversed(values[:-1]) if v["temperature_K"] <= 1800), None)
        if in_range is not None:
            in_range = {
                **in_range,
                **(_closure_miss(
                    in_range["residual_kJ_mol"], in_range["temperature_K"], phase
                ) or {}),
            }
        at_tm = {**at_tm, **(_closure_miss(at_tm["residual_kJ_mol"], tm, phase) or {})}
        results.append({"phase": phase["id"], "formula": phase["formula"],
            "melting_point_K": tm, "congruent": phase["melting_point"]["congruent"],
            "source": phase["source"], "SI_at_melting_point": at_tm["SI"],
            "residual_at_melting_point_kJ_mol": at_tm["residual_kJ_mol"],
            "delta_log10_vapor_product_at_melting_point": at_tm["delta_log10_vapor_product"],
            "root_K": root, "in_range_node": in_range,
            "independent_deltaG_fusion_kJ_mol": at_tm.get("independent_deltaG_fusion_kJ_mol"),
            "independent_fusion_method": at_tm.get("independent_fusion_method"),
            "independent_fusion_flag": at_tm.get("independent_fusion_flag"),
            "closure_miss_kJ_mol": at_tm.get("closure_miss_kJ_mol"),
            "closure_miss_vapor_product": at_tm.get("closure_miss_vapor_product"),
            "SI_positive_below_root": (
                phase_state(phase, max(grid[0], root - 1.0))["SI"] > 0
                if root is not None else None
            ),
            "source_temperature_range_K": phase["valid_temperature_K"]})
    eutectic_composition = {oxide: 0.0 for oxide in _OXIDE_MOLAR_MASS}
    phase_by_id = {phase["id"]: phase for phase in dataset["phases"]}
    for fraction, phase_id in ((0.42, "anorthite"), (0.58, "diopside")):
        sto = phase_by_id[phase_id]["oxide_stoichiometry"]
        phase_mass = sum(float(nu) * _OXIDE_MOLAR_MASS[oxide] for oxide, nu in sto.items())
        for oxide, nu in sto.items():
            eutectic_composition[oxide] += fraction * float(nu) * _OXIDE_MOLAR_MASS[oxide] / phase_mass * 100.0
    eutectic = {phase_id: phase_state(phase_by_id[phase_id], 1547.0, eutectic_composition)
                for phase_id in ("anorthite", "diopside")}
    controls = []
    for control in dataset["pure_oxide_controls"]:
        temp = float(control["melting_point"]["temperature_K"])
        oxide = control["id"]
        melt = evaluate({oxide: _OXIDE_MOLAR_MASS[oxide]}, temp, pack,
                        basis_type="wt", allow_extrapolation=True,
                        allow_out_of_envelope=True)
        fusion_g = float(control["fusion_gibbs_at_melting_kJ_mol"]) * 1000.0
        si = math.log10(melt.activity(oxide)) + fusion_g / (R * math.log(10.0) * temp)
        controls.append({"phase": oxide, "temperature_K": temp, "SI": si,
                         "activity": melt.activity(oxide), "source": control["source"]})
    return results, {"condensate_table_digest": digest, "pure_oxide_controls": controls,
                     "An42Di58_1547K": eutectic}
