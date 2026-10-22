"""Equilibrium thermodynamics for selected subsolidus host reactions.

This module returns thermodynamic ceilings only. It has no evaporation-rate,
area, transport, or equipment model.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Literal, Mapping, Sequence, TypedDict

R_J_MOL_K = 8.31446261815324
STANDARD_PRESSURE_PA = 100_000.0
OXYGEN_INPUT_REL_TOL = 1e-9
OXYGEN_INPUT_ABS_TOL_PA = 1e-6
SpeciesId = str
GasSpeciesId = str

# Reaction construction (one mole as written): sulfate ΔrG° is JANAF crystal
# oxide ΔfG° plus the O-034 tabulated SO2 ΔfG° (with O2 as an elemental
# standard, ΔfG°=0), minus the Robie-Hemingway sulfate ΔfG°. This keeps both
# sulfur references on their published elemental standard states. Troilite
# uses the source's Fe + 1/2 S2 reference directly, so
# ΔrG° = -ΔfG°FeS. All terms are J/mol and Kp = exp(-ΔrG°/RT), dimensionless.
# At 800 K the troilite source node ΔfG°=-111.4 kJ/mol gives pS2,eq about
# 2.84e-10 Pa for unit solid activities, a useful scale/sign check.


class HostInventory(TypedDict):
    species_id: str
    formula: str
    phase: str
    remaining_mol: float
    activity: float | None
    source_tag: str


class StoichTerm(TypedDict):
    species_id: str
    phase: str
    nu_mol_per_mol_rxn: float
    activity: float | None


class SourceRef(TypedDict):
    source_id: str
    row_or_table: str
    standard_state: str
    valid_temperature_K: tuple[float, float]
    doi: str | None


class TypedNotice(TypedDict):
    code: Literal[
        "diffusion_limited_equilibrium_upper_bound",
        "thermochemistry_missing",
        "temperature_out_of_range",
        "unknown_host",
        "missing_required_pressure",
        "first_melt_handoff",
    ]
    severity: Literal["info", "warning", "unavailable"]
    source_id: str | None
    detail: str


ReactionStatus = Literal[
    "supported",
    "thermochemistry_missing",
    "temperature_out_of_range",
    "unknown_host",
    "missing_required_pressure",
    "first_melt_handoff",
]


class ReactionEquilibrium(TypedDict):
    reaction_id: str
    status: ReactionStatus
    stoichiometry: Sequence[StoichTerm]
    driving_gas_species_id: str | None
    delta_rG0_J_per_mol_rxn: float | None
    log10_Kp: float | None
    equilibrium_gas_partial_pressures_Pa: Mapping[str, float] | None
    extent_bound_mol_rxn: float | None
    joint_valid_temperature_K: tuple[float, float] | None
    sources: Sequence[SourceRef]
    notices: Sequence[TypedNotice]


@lru_cache(maxsize=1)
def _data() -> dict:
    return json.loads((Path(__file__).parent / "data" / "subsolidus.json").read_text())


def _node_value(record: Mapping, temperature_K: float) -> float:
    """Linearly interpolate only between adjacent published Gibbs nodes."""
    low, high = record["valid_temperature_K"]
    if temperature_K < low or temperature_K > high:
        raise ValueError(f"outside [{low}, {high}] K")
    nodes = record["nodes"]
    if not nodes or temperature_K < nodes[0][0] or temperature_K > nodes[-1][0]:
        raise ValueError("temperature is outside tabulated Gibbs nodes")
    for (t0, g0), (t1, g1) in zip(nodes, nodes[1:]):
        if t0 <= temperature_K <= t1:
            return g0 + (g1 - g0) * (temperature_K - t0) / (t1 - t0)
    return float(nodes[-1][1])


def _notice(code: str, severity: str, detail: str, source_id: str | None = None) -> TypedNotice:
    return {"code": code, "severity": severity, "source_id": source_id, "detail": detail}


def _source(record: Mapping) -> SourceRef:
    return {
        "source_id": record["source_id"],
        "row_or_table": record["row_or_table"],
        "standard_state": record["standard_state"],
        "valid_temperature_K": tuple(record["valid_temperature_K"]),
        "doi": record["doi"],
    }


def _base_result(
    reaction_id: str,
    status: ReactionStatus,
    stoichiometry: Sequence[StoichTerm],
    driver: str | None,
    source_refs: Sequence[SourceRef],
    notices: Sequence[TypedNotice],
    interval: tuple[float, float] | None,
    *,
    delta_g0: float | None = None,
    log_k: float | None = None,
    pressure: Mapping[str, float] | None = None,
    extent: float | None = None,
) -> ReactionEquilibrium:
    return {
        "reaction_id": reaction_id,
        "status": status,
        "stoichiometry": tuple(stoichiometry),
        "driving_gas_species_id": driver,
        "delta_rG0_J_per_mol_rxn": delta_g0,
        "log10_Kp": log_k,
        "equilibrium_gas_partial_pressures_Pa": pressure,
        "extent_bound_mol_rxn": extent,
        "joint_valid_temperature_K": interval,
        "sources": tuple(source_refs),
        "notices": tuple(notices),
    }


_REACTIONS = {
    "FeS_troilite": (
        "troilite_decomposition",
        "S2",
        (("FeS_troilite", "troilite", -1.0), ("Fe", "metal", 1.0), ("S2", "gas", 0.5)),
        "FeS",
    ),
    "CaSO4": (
        "anhydrite_decomposition",
        "SO2",
        (("CaSO4", "anhydrite", -1.0), ("CaO", "crystal", 1.0), ("SO2", "gas", 1.0), ("O2", "gas", 0.5)),
        "CaSO4",
    ),
    "MgSO4": (
        "magnesium_sulfate_decomposition",
        "SO2",
        (("MgSO4", "crystalline anhydrous magnesium sulfate", -1.0), ("MgO", "crystal", 1.0), ("SO2", "gas", 1.0), ("O2", "gas", 0.5)),
        "MgSO4",
    ),
}


def _activity(value: float | None, name: str) -> float:
    result = 1.0 if value is None else float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"activity for {name} must be finite and positive")
    return result


def _host_activity(host: HostInventory, activities: Mapping[str, float], species_id: str) -> float:
    supplied = host.get("activity")
    return _activity(supplied if supplied is not None else activities.get(species_id), species_id)


def evaluate_subsolidus_reactions(
    *,
    hosts: Sequence[HostInventory],
    temperature_K: float,
    gas_partial_pressures_Pa: Mapping[GasSpeciesId, float],
    condensed_phase_activities: Mapping[SpeciesId, float],
    oxygen_fugacity_Pa: float | None = None,
) -> Sequence[ReactionEquilibrium]:
    """Evaluate troilite and sulfate reaction equilibrium at ``temperature_K``.

    Pressures are Pa, activities are dimensionless, Gibbs energy is J/mol of
    reaction as written, and extent is mol of reaction. Gas standard pressure
    is 100000 Pa. Unknown hosts and unsupported ranges return typed results;
    no thermodynamic value is extrapolated beyond its source interval.
    """
    if not math.isfinite(temperature_K) or temperature_K <= 0:
        raise ValueError("temperature_K must be finite and positive")
    gas_pressures = dict(gas_partial_pressures_Pa)
    for key, value in gas_pressures.items():
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"partial pressure for {key} must be finite and positive")

    gas_o2 = gas_pressures.get("O2")
    if oxygen_fugacity_Pa is not None:
        if not math.isfinite(oxygen_fugacity_Pa) or oxygen_fugacity_Pa <= 0:
            raise ValueError("oxygen_fugacity_Pa must be finite and positive")
        if gas_o2 is not None and not math.isclose(
            oxygen_fugacity_Pa, gas_o2,
            rel_tol=OXYGEN_INPUT_REL_TOL, abs_tol=OXYGEN_INPUT_ABS_TOL_PA,
        ):
            oxygen_conflict = True
        else:
            oxygen_conflict = False
            gas_o2 = oxygen_fugacity_Pa
    else:
        oxygen_conflict = False

    data = _data()["species"]
    results: list[ReactionEquilibrium] = []
    from .gas import ImccGasSpeciesNotFoundError, ImccGasTemperatureOutsideDomainError

    for host in hosts:
        host_id = host.get("species_id", "")
        if host_id not in _REACTIONS:
            results.append(_base_result(
                "unknown_host", "unknown_host", (), None, (),
                (_notice("unknown_host", "unavailable", f"Unrecognized exact host {host_id!r}."),), None,
            ))
            continue
        reaction_id, driver, terms_raw, formula = _REACTIONS[host_id]
        expected = data[host_id]
        stoich: tuple[StoichTerm, ...] = tuple({
            "species_id": species, "phase": phase, "nu_mol_per_mol_rxn": nu,
            "activity": None,
        } for species, phase, nu in terms_raw)
        if host.get("formula") != formula or host.get("phase") != expected["phase"]:
            results.append(_base_result(
                reaction_id, "unknown_host", stoich, driver, (_source(expected),),
                (_notice("unknown_host", "unavailable", "Host formula or phase does not match the supported endmember.", expected["source_id"]),),
                tuple(expected["valid_temperature_K"]),
            ))
            continue

        sources = [_source(expected)]
        interval_low, interval_high = expected["valid_temperature_K"]
        datapack = None
        if host_id == "FeS_troilite":
            iron = data["Fe"]
            sources.append(_source(iron))
            interval_low = max(interval_low, iron["valid_temperature_K"][0])
            interval_high = min(interval_high, iron["valid_temperature_K"][1])
        else:
            oxide_id = "CaO" if host_id == "CaSO4" else "MgO"
            sources.extend((_source(data[oxide_id]), _source(data["SO2"])))
            interval_low = max(
                interval_low,
                data[oxide_id]["valid_temperature_K"][0],
                data["SO2"]["valid_temperature_K"][0],
            )
            interval_high = min(
                interval_high,
                data[oxide_id]["valid_temperature_K"][1],
                data["SO2"]["valid_temperature_K"][1],
            )
            try:
                from .gas import load_gas_datapack, species_thermo
                datapack = load_gas_datapack()
                for name in ("SO2", "O2"):
                    table_rows = datapack.gas_df.loc[f"{name}(g)"]
                    if not hasattr(table_rows, "columns"):
                        table_rows = table_rows.to_frame().T
                    lows = [float(v) for v in table_rows["T_min"]]
                    highs = [float(v) for v in table_rows["T_max"]]
                    interval_low = max(interval_low, min(lows))
                    interval_high = min(interval_high, max(highs))
                if interval_low <= temperature_K <= interval_high:
                    active_lows = []
                    active_highs = []
                    for name in ("SO2", "O2"):
                        row = species_thermo(name, "g", temperature_K, datapack)
                        active_lows.append(row.T_min)
                        active_highs.append(row.T_max)
                        if row.source_row_id not in {s["source_id"] for s in sources}:
                            sources.append({
                                "source_id": row.source_row_id,
                                "row_or_table": f"JANAF gas {row.source_row_id}",
                                "standard_state": "ideal gas, 1 bar; JANAF standard Gibbs function",
                                "valid_temperature_K": (row.T_min, row.T_max),
                                "doi": "10.18434/T42S31",
                            })
                    interval_low = max(interval_low, *active_lows)
                    interval_high = min(interval_high, *active_highs)
            except ImccGasSpeciesNotFoundError as exc:
                results.append(_base_result(
                    reaction_id, "thermochemistry_missing", stoich, driver, sources,
                    (_notice("thermochemistry_missing", "unavailable", str(exc), expected["source_id"]),),
                    None,
                ))
                continue

        joint_interval = (float(interval_low), float(interval_high))
        if not interval_low <= temperature_K <= interval_high:
            detail = f"T={temperature_K:g} K is outside the joint source interval {joint_interval}."
            results.append(_base_result(
                reaction_id, "temperature_out_of_range", stoich, driver, sources,
                (_notice("temperature_out_of_range", "unavailable", detail, expected["source_id"]),),
                joint_interval,
            ))
            continue

        try:
            host_g = _node_value(expected, temperature_K)
            if host_id == "FeS_troilite":
                delta_g0 = -host_g
                activity_host = _host_activity(host, condensed_phase_activities, host_id)
                activity_product = _activity(condensed_phase_activities.get("Fe"), "Fe")
                log_q_no_driver = math.log(activity_product / activity_host)
                log_pressure_eq = -2.0 * delta_g0 / (R_J_MOL_K * temperature_K) - 2.0 * log_q_no_driver
                ln_kp = -delta_g0 / (R_J_MOL_K * temperature_K)
            else:
                oxide_id = "CaO" if host_id == "CaSO4" else "MgO"
                oxide_g = _node_value(data[oxide_id], temperature_K)
                gas_so2_g = _node_value(data["SO2"], temperature_K)
                delta_g0 = oxide_g + gas_so2_g - host_g
                activity_host = _host_activity(host, condensed_phase_activities, host_id)
                activity_product = _activity(condensed_phase_activities.get(oxide_id), oxide_id)
                ln_kp = -delta_g0 / (R_J_MOL_K * temperature_K)
                log_pressure_eq = ln_kp + math.log(activity_host / activity_product)
            activity_by_species = {
                host_id: activity_host,
                "Fe" if host_id == "FeS_troilite" else ("CaO" if host_id == "CaSO4" else "MgO"): activity_product,
            }
            stoich = tuple({
                "species_id": species,
                "phase": phase,
                "nu_mol_per_mol_rxn": nu,
                "activity": activity_by_species.get(species),
            } for species, phase, nu in terms_raw)
        except (ValueError, ImccGasTemperatureOutsideDomainError) as exc:
            results.append(_base_result(
                reaction_id, "temperature_out_of_range", stoich, driver, sources,
                (_notice("temperature_out_of_range", "unavailable", str(exc), expected["source_id"]),), joint_interval,
            ))
            continue

        log10_k = ln_kp / math.log(10.0)
        try:
            remaining = float(host["remaining_mol"])
            if not math.isfinite(remaining) or remaining < 0:
                raise ValueError("remaining_mol must be finite and nonnegative")
        except (KeyError, TypeError, ValueError) as exc:
            results.append(_base_result(
                reaction_id, "unknown_host", stoich, driver, sources,
                (_notice("unknown_host", "unavailable", str(exc), expected["source_id"]),), joint_interval,
            ))
            continue

        if oxygen_conflict:
            results.append(_base_result(
                reaction_id, "missing_required_pressure", stoich, driver, sources,
                (_notice("missing_required_pressure", "unavailable", "oxygen_fugacity_Pa disagrees with the O2 partial pressure."),),
                joint_interval, delta_g0=delta_g0, log_k=log10_k,
            ))
            continue

        missing = []
        if driver not in gas_pressures:
            missing.append(driver)
        if host_id != "FeS_troilite" and gas_o2 is None:
            missing.append("O2 fugacity")
        if missing:
            results.append(_base_result(
                reaction_id, "missing_required_pressure", stoich, driver, sources,
                (_notice("missing_required_pressure", "unavailable", f"Missing required pressure input: {', '.join(missing)}."),),
                joint_interval, delta_g0=delta_g0, log_k=log10_k,
            ))
            continue

        if host_id != "FeS_troilite":
            log_pressure_eq -= 0.5 * math.log(gas_o2 / STANDARD_PRESSURE_PA)
        equilibrium_pressure = STANDARD_PRESSURE_PA * math.exp(log_pressure_eq)

        activities = math.log(activity_product / activity_host)
        if host_id == "FeS_troilite":
            log_q = activities + 0.5 * math.log(gas_pressures[driver] / STANDARD_PRESSURE_PA)
        else:
            log_q = (
                activities
                + math.log(gas_pressures[driver] / STANDARD_PRESSURE_PA)
                + 0.5 * math.log(gas_o2 / STANDARD_PRESSURE_PA)
            )
        delta_g = delta_g0 + R_J_MOL_K * temperature_K * log_q
        extent = remaining if delta_g < 0.0 else 0.0
        notices = [_notice(
            "diffusion_limited_equilibrium_upper_bound", "info",
            "Equilibrium extent is a thermodynamic ceiling; no solid-release rate law is modeled.",
        )]
        # Gibbs identity: the reaction function uses formation rows from R&H and JANAF.
        results.append(_base_result(
            reaction_id, "supported", stoich, driver, sources, notices, joint_interval,
            delta_g0=delta_g0, log_k=log10_k,
            pressure={driver: equilibrium_pressure}, extent=extent,
        ))
    return tuple(results)
