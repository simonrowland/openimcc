"""Package-backed JANAF pure-crystal apparent Gibbs standards."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from types import MappingProxyType
from typing import Any, Mapping


_RESOURCE = "data/diagnostics/janaf-pure-oxide-crystals.json"
_ANCHOR_TEMPERATURE_K = 298.15


@dataclass(frozen=True)
class JanafCrystalGibbs:
    """One crystal's apparent Gibbs value and its thermochemical provenance."""

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


@lru_cache(maxsize=1)
def _load_phases() -> tuple[dict[str, Any], str]:
    raw = resources.files("openimcc").joinpath(_RESOURCE).read_bytes()
    data = json.loads(raw.decode("utf-8"))
    return {row["id"]: row for row in data["phases"]}, hashlib.sha256(raw).hexdigest()


def _interpolate_nodes(nodes: list[dict[str, Any]], key: str, temperature_K: float) -> float:
    # Collapse equal-temperature source rows in their stable input order; a
    # printed final row wins at the exact node, matching the data order.
    values = {float(node["T_K"]): float(node[key]) for node in nodes}
    points = sorted(values.items())
    if temperature_K in values:
        return values[temperature_K]
    for (lo_T, lo_value), (hi_T, hi_value) in zip(points, points[1:]):
        if lo_T < temperature_K < hi_T:
            return lo_value + (hi_value - lo_value) * (
                (temperature_K - lo_T) / (hi_T - lo_T)
            )
    raise ValueError(f"no source node interval at {temperature_K:g} K")


def evaluate_janaf_crystal(phase_id: str, temperature_K: float) -> JanafCrystalGibbs:
    """Return apparent crystal G (J/mol), its valid domain, and provenance.

    The packaged source subset is generated from vendored JANAF tables. At a
    crystal-liquid cap, use the printed transition-row enthalpy and entropy
    rather than an interpolation through adjacent bulk-table nodes.
    """
    rows, resource_digest = _load_phases()
    if phase_id not in rows:
        raise KeyError(f"no packaged JANAF crystal function for {phase_id!r}")
    row = rows[phase_id]
    temperature = float(temperature_K)
    if not math.isfinite(temperature):
        raise ValueError("temperature_K must be finite")
    source_lo, source_hi = map(float, row["source_temperature_K"])
    domain = (
        max(_ANCHOR_TEMPERATURE_K, source_lo),
        min(source_hi, float(row["crystal_transition_temperature_K"])),
    )
    if not domain[0] <= temperature <= domain[1]:
        raise JanafCrystalTemperatureError(phase_id, temperature, domain)

    transition = row.get("transition_row_at_cap")
    use_transition = transition is not None and temperature == domain[1]
    source = [transition] if use_transition else row["nodes"]
    h_increment = _interpolate_nodes(
        source, "enthalpy_increment_kJ_mol", temperature
    )
    entropy = _interpolate_nodes(source, "entropy_J_mol_K", temperature)
    # G_app(T) = Hf(298.15) + [H(T)-H(298.15)] - T*S(T).
    # Enthalpy is stored in kJ/mol and entropy in J/(mol K); multiplying the
    # enthalpy sum by 1000 gives J/mol, matching T*S.
    gibbs = (
        float(row["formation_enthalpy_298_kJ_mol"]) + h_increment
    ) * 1000.0 - temperature * entropy
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
    return JanafCrystalGibbs(gibbs, domain, provenance)
