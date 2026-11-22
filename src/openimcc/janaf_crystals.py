"""Package-backed JANAF pure-crystal apparent Gibbs standards."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

from openimcc._crystal_interpolation import _linear_value_and_slope


_RESOURCE = "data/diagnostics/janaf-pure-oxide-crystals.json"
_SPECIES_RESOURCE = "data/diagnostics/janaf-crystal-species.json"
_LIQUID_RESOURCE = "data/diagnostics/janaf-pure-oxide-liquid-pairs.json"
_ANCHOR_TEMPERATURE_K = 298.15


@dataclass(frozen=True)
class _JanafOxidePair:
    oxide: str
    crystal_table: str
    liquid_table: str | None
    transition_K: float
    construction: str | None = None
    construction_uncertainty_input_K: tuple[float, float] | None = None


_JANAF_OXIDE_PAIRS = MappingProxyType(
    {
        pair.oxide: pair
        for pair in (
            _JanafOxidePair("SiO2", "O-035", "O-038", 1996.0),
            _JanafOxidePair("MgO", "Mg-008", "Mg-009", 3105.0),
            _JanafOxidePair("CaO", "Ca-027", "Ca-028", 3200.0),
            _JanafOxidePair("Al2O3", "Al-096", "Al-100", 2327.0),
            _JanafOxidePair("FeO", "Fe-018", "Fe-019", 1650.0),
            _JanafOxidePair("Na2O", "Na-012", "Na-013", 1405.2),
            _JanafOxidePair("TiO2", "O-043", "O-044", 2130.0),
            _JanafOxidePair(
                "K2O",
                "K-012",
                None,
                1013.0,
                "constructed liquid standard; selected Tfus=1013 K, "
                "source band 919–1190 K",
                (919.0, 1190.0),
            ),
        )
    }
)


@dataclass(frozen=True)
class JanafCrystalGibbs:
    """One crystal's apparent Gibbs value and its thermochemical provenance."""

    G_J_mol: float
    domain_K: tuple[float, float]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


@dataclass(frozen=True)
class JanafApparentGibbs:
    """One JANAF source row's apparent Gibbs value and provenance."""

    G_J_mol: float
    domain_K: tuple[float, float]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


class JanafCrystalTemperatureError(ValueError):
    """Requested temperature lies outside a JANAF crystal's allowed domain."""

    def __init__(self, phase_id: str, temperature_K: float, domain_K: tuple[float, float]):
        self.phase_id = phase_id
        self.temperature_K = temperature_K
        self.domain_K = domain_K
        super().__init__(
            f"JANAF crystal {phase_id} has no Gibbs value at {temperature_K:g} K; "
            f"domain is [{domain_K[0]:g}, {domain_K[1]:g}] K"
        )


class JanafRowTemperatureError(ValueError):
    """Requested temperature lies outside a packaged JANAF source row."""

    def __init__(self, table_id: str, temperature_K: float, domain_K: tuple[float, float]):
        self.table_id = table_id
        self.temperature_K = temperature_K
        self.domain_K = domain_K
        super().__init__(
            f"JANAF row {table_id} has no Gibbs value at {temperature_K:g} K; "
            f"domain is [{domain_K[0]:g}, {domain_K[1]:g}] K"
        )


@lru_cache(maxsize=1)
def _load_phases() -> tuple[dict[str, Any], str]:
    raw = resources.files("openimcc").joinpath(_RESOURCE).read_bytes()
    data = json.loads(raw.decode("utf-8"))
    return {row["id"]: row for row in data["phases"]}, hashlib.sha256(raw).hexdigest()


@lru_cache(maxsize=1)
def _load_crystal_species() -> tuple[dict[str, Any], str]:
    raw = resources.files("openimcc").joinpath(_SPECIES_RESOURCE).read_bytes()
    data = json.loads(raw.decode("utf-8"))
    return {row["id"]: row for row in data["phases"]}, hashlib.sha256(raw).hexdigest()


@lru_cache(maxsize=1)
def _load_liquid_rows() -> tuple[dict[str, Any], str]:
    raw = resources.files("openimcc").joinpath(_LIQUID_RESOURCE).read_bytes()
    data = json.loads(raw.decode("utf-8"))
    return {row["table_id"]: row for row in data["rows"]}, hashlib.sha256(raw).hexdigest()


def _apparent_gibbs(
    formation_enthalpy_298_kj_mol: float,
    nodes: Sequence[tuple[float, float, float]],
    temperature: float,
    *,
    interpolation: Literal["nodes", "numpy"] = "nodes",
) -> float:
    # Keep each caller's established interpolation arithmetic for exact floats.
    def interpolate(column: int) -> float:
        if interpolation == "numpy":
            import numpy as np

            return float(
                np.interp(
                    temperature,
                    [node[0] for node in nodes],
                    [node[column] for node in nodes],
                )
            )
        value, _ = _linear_value_and_slope(
            ((float(node[0]), float(node[column])) for node in nodes),
            temperature,
            slope_required=False,
        )
        return value

    h_increment = interpolate(1)
    entropy = interpolate(2)
    # G_app(T) = Hf(298.15) + [H(T)-H(298.15)] - T*S(T).
    # Enthalpy is stored in kJ/mol and entropy in J/(mol K); multiplying the
    # enthalpy sum by 1000 gives J/mol, matching T*S.
    return (
        (formation_enthalpy_298_kj_mol + h_increment) * 1000.0
        - temperature * entropy
    )


def _node_values(
    nodes: Sequence[Mapping[str, Any]],
) -> list[tuple[float, float, float]]:
    return [
        (
            float(node["T_K"]),
            float(node["enthalpy_increment_kJ_mol"]),
            float(node["entropy_J_mol_K"]),
        )
        for node in nodes
    ]


def _janaf_nodes_for_temperature(
    row: Mapping[str, Any],
    temperature_K: float,
    *,
    include_cap: bool = True,
) -> list[Mapping[str, Any]]:
    """Select each equal-temperature JANAF transition row for this segment.

    The named ``A <--> B`` row closes the segment below a transition; the
    following ``TRANSITION`` row opens the next segment at that same T. At a
    crystal-liquid cap, retain the dedicated cap row at the inclusive edge.
    """
    nodes = [*row["nodes"], *row.get("transition_rows", [])]
    cap_row = row.get("transition_row_at_cap") if include_cap else None
    if cap_row is not None:
        nodes.append(cap_row)

    by_temperature: dict[float, list[Mapping[str, Any]]] = {}
    for node in nodes:
        by_temperature.setdefault(float(node["T_K"]), []).append(node)

    selected = []
    cap_temperature = float(cap_row["T_K"]) if cap_row is not None else None
    for node_temperature in sorted(by_temperature):
        group = by_temperature[node_temperature]
        if (
            cap_row is not None
            and node_temperature == cap_temperature
            and temperature_K == cap_temperature
        ):
            node = cap_row
        else:
            transition = next(
                (
                    candidate
                    for candidate in reversed(group)
                    if str(candidate.get("label", "")).strip().upper()
                    == "TRANSITION"
                ),
                None,
            )
            phase_endpoint = next(
                (
                    candidate
                    for candidate in group
                    if "<-->" in str(candidate.get("label", ""))
                ),
                None,
            )
            if transition is not None and phase_endpoint is not None:
                node = (
                    transition
                    if temperature_K >= node_temperature
                    else phase_endpoint
                )
            else:
                node = group[-1]
        selected.append(node)
    return selected


def evaluate_janaf_crystal(
    phase_id: str, temperature_K: float, *, apply_phase_cap: bool = True
) -> JanafCrystalGibbs:
    """Return apparent crystal G (J/mol), its valid domain, and provenance.

    The packaged source subset is generated from vendored JANAF tables. At a
    crystal-liquid cap, use the printed transition-row enthalpy and entropy
    rather than an interpolation through adjacent bulk-table nodes. The stable
    phase cap applies by default; set ``apply_phase_cap=False`` to evaluate the
    reconstructed crystal function across its full tabulated temperature span.
    """
    rows, resource_digest = _load_phases()
    if phase_id not in rows:
        rows, resource_digest = _load_crystal_species()
    if phase_id not in rows:
        raise KeyError(f"no packaged JANAF crystal function for {phase_id!r}")
    row = rows[phase_id]
    temperature = float(temperature_K)
    if not math.isfinite(temperature):
        raise ValueError("temperature_K must be finite")
    source_lo, source_hi = map(float, row["source_temperature_K"])
    tabulated_domain = (
        max(_ANCHOR_TEMPERATURE_K, source_lo),
        source_hi,
    )
    phase_domain = (
        tabulated_domain[0],
        min(tabulated_domain[1], float(row["crystal_transition_temperature_K"])),
    )
    domain = phase_domain if apply_phase_cap else tabulated_domain
    if not domain[0] <= temperature <= domain[1]:
        raise JanafCrystalTemperatureError(phase_id, temperature, domain)

    transition = row.get("transition_row_at_cap")
    use_transition = transition is not None and temperature == float(transition["T_K"])
    # Include the printed transition point as the final interpolation node
    # below the cap. Otherwise the last ordinary 100 K node is extrapolated
    # toward the transition, causing a discontinuity immediately below Tm.
    source = (
        [transition]
        if use_transition
        else _janaf_nodes_for_temperature(
            row, temperature, include_cap=False
        )
    )
    gibbs = _apparent_gibbs(
        float(row["formation_enthalpy_298_kJ_mol"]),
        _node_values(source),
        temperature,
    )
    provenance = {
        "table_id": row["crystal_table_id"],
        "source_sha256": row["source_sha256"],
        "resource_sha256": resource_digest,
        "source_authority": row["source_authority"],
        "gibbs_basis": "JANAF apparent Gibbs anchored at elemental enthalpies at 298.15 K",
        "transition_row_used": use_transition,
        "transition_source_line_number": (
            transition["source_line_number"] if use_transition else None
        ),
    }
    if "source_title" in row:
        provenance["source_title"] = row["source_title"]
    return JanafCrystalGibbs(gibbs, domain, provenance)


def evaluate_janaf_apparent_gibbs(
    table_id: str, temperature_K: float
) -> JanafApparentGibbs:
    """Evaluate a packaged JANAF liquid control row on its apparent-G basis."""
    rows, resource_digest = _load_liquid_rows()
    if table_id not in rows:
        raise KeyError(f"no packaged JANAF liquid control row for {table_id!r}")
    row = rows[table_id]
    temperature = float(temperature_K)
    if not math.isfinite(temperature):
        raise ValueError("temperature_K must be finite")
    source_lo, source_hi = map(float, row["source_temperature_K"])
    domain = (max(_ANCHOR_TEMPERATURE_K, source_lo), source_hi)
    if not domain[0] <= temperature <= domain[1]:
        raise JanafRowTemperatureError(table_id, temperature, domain)

    transitions = row["transition_rows"]
    transition = next(
        (node for node in reversed(transitions) if float(node["T_K"]) == temperature),
        None,
    )
    source_nodes = (
        [transition]
        if transition is not None
        else _janaf_nodes_for_temperature(
            row, temperature, include_cap=False
        )
    )
    gibbs = _apparent_gibbs(
        float(row["formation_enthalpy_298_kJ_mol"]),
        _node_values(source_nodes),
        temperature,
    )
    provenance = {
        "table_id": table_id,
        "source_sha256": row["source_sha256"],
        "resource_sha256": resource_digest,
        "source_authority": row["source_authority"],
        "gibbs_basis": "JANAF apparent Gibbs anchored at elemental enthalpies at 298.15 K",
        "transition_row_used": transition is not None,
        "transition_source_line_number": (
            transition["source_line_number"] if transition is not None else None
        ),
    }
    return JanafApparentGibbs(gibbs, domain, provenance)
