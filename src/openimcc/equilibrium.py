"""Nested ideal-melt and stoichiometric-crystal phase equilibrium.

The liquid remains delegated to openimcc.model.evaluate; this module solves
the outer parent-balance/crystal problem and independently checks its result.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from scipy.optimize import LinearConstraint, brentq, linprog, lsq_linear, minimize
from scipy.special import logsumexp

from openimcc import diagnostics
from openimcc._constants import MOLAR_MASS_G_MOL, R_J_MOL_K
from openimcc.declared_assemblage import (
    EquilibriumRefusal,
    PhaseEquilibriumLabels,
    _evaluate_gas_attachment,
    _least_authority,
    _load_json,
    _phase_authority,
)
from openimcc.gas_data import _OXIDE_PROVENANCE_AUTHORITY
from openimcc.janaf_crystals import (
    JanafCrystalTemperatureError,
    evaluate_janaf_crystal,
)
from openimcc.kernel import ImccRefusal
from openimcc.model import ImccLoadedDatapack, evaluate, load_datapack
from openimcc.receipt import CalculationReceipt, build_calculation_receipt


_PURE_RESOURCE = "data/diagnostics/janaf-pure-oxide-crystals.json"
_CRYSTAL_RESOURCE = "data/diagnostics/crystal-phases.json"
_BARIN_RESOURCE = "data/diagnostics/barin-aluminate-crystals.json"
_PRESSURE_BAR = 1.0
_LIQUID_DOMAIN_K = (1700.0, 3000.0)
phase_state_contract_version = 1

_MISSING_PHASE_COMPONENTS = {
    "CA6": frozenset(("CaO", "Al2O3")),
    "beta-alumina": frozenset(("Na2O", "Al2O3")),
}

# Each named tolerance applies to a different physical/numerical certificate.
_DEFAULT_BALANCE_TOL = 1.0e-10
_INNER_BALANCE_TOL = 1.0e-12
_DEFAULT_AFFINITY_TOL_LN = 1.0e-8
_DUAL_SLACK_TOL_RT = 1.0e-10
_LOGQ_COEXISTENCE_TOL = 1.0e-8
_FD_CORRECTION_TOL_LN = 2.0e-9
_INNER_ACTIVITY_ERROR_TOL_LN = 2.0e-9
_INNER_CORRECTION_TOL_LN = 2.0e-9
_DUAL_FACE_REL_TOL = 1.0e-9
_FD_STEP_SCALED = 1.0e-4
_FRACTION_TO_BOUNDARY = 0.995
_MAX_OUTER_ITERATIONS = 80
_MAX_ACTIVE_RESTARTS = 40


@dataclass(frozen=True)
class CrystalCandidate:
    """A stoichiometric crystal Gibbs function at the requested temperature.

    Gibbs energy is J per formula unit; stoichiometric coefficients are mol
    parent oxide per mol formula unit. Records may describe synthetic systems.
    """

    phase_id: str
    stoichiometry: Mapping[str, float]
    G_crystal_J_mol: float
    formula: str | None = None
    source_authority: str = "external_datapack"
    temperature_range_K: tuple[float, float] | None = None
    source: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = {str(key): float(value) for key, value in self.stoichiometry.items()}
        if not values or any(not math.isfinite(v) or v <= 0.0 for v in values.values()):
            raise ValueError("crystal stoichiometry must contain positive finite amounts")
        if not math.isfinite(float(self.G_crystal_J_mol)):
            raise ValueError("crystal Gibbs energy must be finite")
        object.__setattr__(self, "stoichiometry", MappingProxyType(values))
        object.__setattr__(self, "source", MappingProxyType(dict(self.source)))


@dataclass(frozen=True)
class PhaseEquilibriumResult:
    temperature_K: float
    pressure_bar: float
    bulk_parent_mol: Mapping[str, float]
    liquid_parent_mol: Mapping[str, float] | None
    liquid_mass_fraction: float | None
    solid_mol: Mapping[str, float] | None
    phase_amount_ranges_mol: Mapping[str, tuple[float, float]]
    phase_amounts_unique: bool
    phase_presence: Mapping[str, Literal["present", "absent", "unresolved", "infeasible"]]
    saturated_candidates: tuple[str, ...]
    identifiable_activities: Mapping[str, float]
    activity_ranges_ln: Mapping[str, tuple[float, float]]
    phase_affinities_ln: Mapping[str, float]
    phase_affinity_ranges_ln: Mapping[str, tuple[float, float]]
    phase_saturation_indices: tuple[diagnostics.SaturationIndex, ...]
    logQ: float | None
    logQ_interval: tuple[float, float] | None
    melt: Any | None
    gas: Any | None
    mass_balance_relative_inf: float | None
    labels: PhaseEquilibriumLabels
    receipt: CalculationReceipt
    refusal: EquilibriumRefusal | None
    active_set_history: tuple[tuple[str, ...], ...] = ()
    kkt_recheck: Mapping[str, float] = field(default_factory=dict)
    crystal_only_minimum: bool = False
    fO2: float | None = None
    gas_depends: Mapping[str, str] = field(default_factory=dict)
    vapour_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the result plus the versioned simulator phase-state contract."""
        result = _json_compatible(self)
        if self.gas is None:
            vapour = None
            vapour_domain_flags: Mapping[str, Any] = {}
            vapour_authority: Mapping[str, Any] = {}
            vapour_omitted: Mapping[str, Any] = {}
        else:
            vapour = {
                "partial_pressures_bar": dict(self.gas),
                "unit": self.gas.unit,
            }
            vapour_domain_flags = self.gas.domain_flags
            vapour_authority = self.gas.provenance_class
            vapour_omitted = self.gas.omitted_channels
        vapour_out_of_domain_axes = (
            ("gas_species.temperature_K",)
            if self.vapour_error else
            tuple(
                f"gas_species.{name}.temperature_K"
                for name, detail in self.gas.domain_flags.items()
                if detail is not None
            ) if self.gas is not None else ()
        )
        if self.fO2 is None:
            vapour_status = "NOT_REQUESTED"
        elif vapour_out_of_domain_axes:
            vapour_status = "OUT_OF_DOMAIN"
        elif self.labels.validity == "OUT_OF_DOMAIN":
            vapour_status = "OUT_OF_DOMAIN"
        elif self.gas_depends or self.labels.validity == "DEPENDS":
            vapour_status = "DEPENDS"
        else:
            vapour_status = "AVAILABLE"
        liquid_fraction_certified = (
            self.liquid_mass_fraction is not None
            and not self.crystal_only_minimum
            and self.kkt_recheck.get("independent_kkt_passed") == 1.0
        )
        request = self.receipt.request
        contract_candidates = request.get("candidates")
        if (
            contract_candidates is None
            and "restricted_candidate_set" in self.labels.flags
        ):
            contract_candidates = request.get("candidate_phase_ids")
        result["phase_state_contract"] = {
            "name": "openimcc.phase-state",
            "version": phase_state_contract_version,
            "input": {
                "bulk_basis": "parent_oxide_mol",
                "bulk_parent_mol": result["bulk_parent_mol"],
                "temperature_K": self.temperature_K,
                "pressure_bar": self.pressure_bar,
                "candidates": _json_compatible(contract_candidates),
                "fO2": self.fO2,
            },
            "output": {
                "phase_presence": result["phase_presence"],
                "phase_amount_ranges_mol": result["phase_amount_ranges_mol"],
                "phase_amounts_unique": self.phase_amounts_unique,
                "liquid_mass_fraction": {
                    "value": self.liquid_mass_fraction,
                    "certification_status": (
                        "CERTIFIED" if liquid_fraction_certified else "UNRESOLVED"
                    ),
                },
                "identifiable_activities": result["identifiable_activities"],
                "labels": result["labels"],
                "refusal": result["refusal"],
                "receipt": result["receipt"],
                "vapour": {
                    "status": vapour_status,
                    "interpretation": (
                        "Fixed-activity equilibrium vapour potential over the stated "
                        "bulk at T; no oxygen closure, vapour mass balance, rate, "
                        "extent, or depletion is implied."
                    ),
                    "fO2": self.fO2,
                    "unit": "bar",
                    "partial_pressures_bar": (
                        vapour["partial_pressures_bar"] if vapour is not None else None
                    ),
                    "depends": result["gas_depends"],
                    "domain_flags": _json_compatible(vapour_domain_flags),
                    "out_of_domain_axes": vapour_out_of_domain_axes,
                    "provenance_class": _json_compatible(vapour_authority),
                    "omitted_channels": _json_compatible(vapour_omitted),
                    "error": self.vapour_error,
                },
            },
        }
        return result

    def __post_init__(self) -> None:
        for name in (
            "bulk_parent_mol", "phase_amount_ranges_mol", "phase_presence",
            "identifiable_activities", "activity_ranges_ln",
            "phase_affinities_ln", "phase_affinity_ranges_ln", "kkt_recheck",
            "gas_depends",
        ):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        for name in ("liquid_parent_mol", "solid_mol"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, MappingProxyType(dict(value)))


@dataclass(frozen=True)
class PhaseBoundaryCrossing:
    """A scanned and refined interval containing one phase boundary."""

    kind: Literal["solidus", "liquidus"]
    bracket_K: tuple[float, float]
    root_temperature_K: float
    root_residual: float
    lower_endpoint_status: str
    upper_endpoint_status: str
    lower_phase_set: tuple[str, ...]
    upper_phase_set: tuple[str, ...]
    residuals: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        return _json_compatible(self)


@dataclass(frozen=True)
class PhaseBoundaryResult:
    """All certified boundary brackets found in the requested temperature scan."""

    pressure_bar: float
    solidus: tuple[PhaseBoundaryCrossing, ...]
    liquidus: tuple[PhaseBoundaryCrossing, ...]
    solidus_status: Literal["AVAILABLE", "NO_CROSSING", "OUT_OF_DOMAIN", "DEPENDS"]
    liquidus_status: Literal["AVAILABLE", "NO_CROSSING", "OUT_OF_DOMAIN", "DEPENDS"]
    labels: PhaseEquilibriumLabels
    receipt: CalculationReceipt
    refusal: EquilibriumRefusal | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_compatible(self)


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.ndarray):
        return [_json_compatible(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_compatible(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_compatible(item) for item in sorted(value, key=str)]
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _json_compatible(getattr(value, item.name))
            for item in fields(value)
        }
    raise TypeError(f"cannot convert {type(value).__name__} to a JSON value")


@dataclass(frozen=True)
class _CandidateState:
    candidate: CrystalCandidate
    authority: str
    domain: tuple[float, float] | None
    available: bool
    reason: str | None = None


@dataclass(frozen=True)
class _LPState:
    amounts: np.ndarray
    chemical_potentials: np.ndarray
    objective_J: float
    dual_slacks_J_mol: np.ndarray
    amount_ranges: tuple[tuple[float, float], ...]
    amount_ranges_certified: bool


def _loaded_pack(pack: Any | None) -> Any:
    return load_datapack() if pack is None else pack


def _has_active_excess(pack: Any) -> bool:
    return any(term.active for term in getattr(pack, "excess_gibbs_terms", ()))


def _source_solid_limit_K(row: Mapping[str, Any]) -> float | None:
    limit = row.get("source_solid_limit")
    if isinstance(limit, Mapping):
        value = limit.get("temperature_K")
    else:
        melting = row.get("melting_point")
        value = melting.get("temperature_K") if isinstance(melting, Mapping) else None
    return None if value is None else float(value)


def _source_solid_limits_K(
    candidates: Sequence[str | CrystalCandidate] | None,
) -> dict[str, float]:
    selected_ids = (
        None if candidates is None else
        {item for item in candidates if isinstance(item, str)}
    )
    limits: dict[str, float] = {}
    if selected_ids is None or selected_ids:
        for resource in (_CRYSTAL_RESOURCE, _BARIN_RESOURCE):
            data, _digest = _load_json(resource)
            for row in data["phases"]:
                phase_id = str(row["id"])
                if selected_ids is not None and phase_id not in selected_ids:
                    continue
                if len(row["oxide_stoichiometry"]) <= 1:
                    continue
                limit = _source_solid_limit_K(row)
                if limit is not None:
                    limits[phase_id] = limit
    if candidates is not None:
        for item in candidates:
            if not isinstance(item, CrystalCandidate) or len(item.stoichiometry) <= 1:
                continue
            limit = _source_solid_limit_K(item.source)
            if limit is not None:
                limits[item.phase_id] = limit
    return limits


def _melt_liquid_standards(
    parents: Sequence[str], temperature_K: float, gas_pack: Any,
) -> np.ndarray:
    from openimcc import gas

    return np.asarray([
        gas.species_thermo(parent, "l", temperature_K, gas_pack).G_J_mol
        for parent in parents
    ], dtype=float)


def _redox_candidate(candidate: CrystalCandidate) -> bool:
    identity = f"{candidate.phase_id} {candidate.formula or ''}".lower().replace(" ", "")
    if any(token in identity for token in ("fe2o3", "hematite", "magnetite", "ferric", "fe3o4")):
        return True
    return any(name.lower().replace(" ", "") == "fe2o3" for name in candidate.stoichiometry)


def _evaluate_fusion_crystal_row(
    phase_id: str,
    row: Mapping[str, Any],
    temperature_K: float,
    gas_pack: Any,
) -> tuple[_CandidateState, bool]:
    """Build one Berman or Barin fusion-node candidate and flag missing data."""
    from openimcc import gas

    source_solid_limit = _source_solid_limit_K(row)
    nodes = sorted(float(value) for value in row["fusion_gibbs_nodes_kJ_mol"])
    domain = (
        max(float(row["valid_temperature_K"][0]), nodes[0]),
        min(float(row["valid_temperature_K"][1]), nodes[-1]),
    )
    authority = _phase_authority(row)

    def unavailable_state(reason: str) -> _CandidateState:
        placeholder = CrystalCandidate(
            phase_id, row["oxide_stoichiometry"], 0.0, row.get("formula"),
            authority, domain, row.get("source", {}),
        )
        return _CandidateState(placeholder, authority, domain, False, reason)

    if (
        len(row["oxide_stoichiometry"]) > 1
        and source_solid_limit is not None
        and temperature_K > source_solid_limit
    ):
        return unavailable_state("above_source_melting_point"), False
    if not domain[0] <= temperature_K <= domain[1]:
        return unavailable_state("temperature_out_of_range"), True
    try:
        liquid_gibbs = {
            oxide: gas.species_thermo(oxide, "l", temperature_K, gas_pack).G_J_mol
            for oxide in row["oxide_stoichiometry"]
        }
        fusion_kJ = diagnostics._interpolate_crystal_fusion_nodes(row, temperature_K)
    except (gas.ImccGasTemperatureOutsideDomainError, ValueError):
        return unavailable_state("standard_out_of_range"), True
    g_crystal = math.fsum(
        float(nu) * liquid_gibbs[oxide]
        for oxide, nu in row["oxide_stoichiometry"].items()
    ) - fusion_kJ * 1000.0
    candidate = CrystalCandidate(
        phase_id, row["oxide_stoichiometry"], g_crystal,
        row.get("formula"), authority, domain, row.get("source", {}),
    )
    return _CandidateState(candidate, authority, domain, True), False


def _exclude_zero_bulk_infeasible_candidates(
    all_states: Sequence[_CandidateState],
    unavailable: Sequence[str],
    normalized_bulk: Mapping[str, float],
    parent_names: Sequence[str],
) -> tuple[tuple[_CandidateState, ...], tuple[str, ...]]:
    zero_bulk_parents = set(parent_names) - set(normalized_bulk)
    infeasible = {
        state.candidate.phase_id
        for state in all_states
        if set(state.candidate.stoichiometry) & zero_bulk_parents
    }
    if not infeasible:
        return tuple(all_states), tuple(unavailable)
    filtered_states = tuple(
        replace(state, available=False, reason="infeasible")
        if state.candidate.phase_id in infeasible else state
        for state in all_states
    )
    filtered_unavailable = tuple(
        phase for phase in unavailable if phase not in infeasible
    )
    return filtered_states, filtered_unavailable


def _default_candidate_states(
    temperature_K: float,
    gas_pack: Any | None = None,
) -> tuple[tuple[_CandidateState, ...], str, str, Any, tuple[str, ...]]:
    from openimcc import gas

    pure, pure_digest = _load_json(_PURE_RESOURCE)
    crystals, crystal_digest = _load_json(_CRYSTAL_RESOURCE)
    barin, _barin_digest = _load_json(_BARIN_RESOURCE)
    pure_by_id = {row["id"]: row for row in pure["phases"]}
    crystal_by_id = {row["id"]: row for row in crystals["phases"]}
    barin_by_id = {row["id"]: row for row in barin["phases"]}
    if gas_pack is None:
        gas_pack, _ = diagnostics._crystal_thermo_context(None)
    states: list[_CandidateState] = []
    unavailable: list[str] = []

    # For crystal fusion nodes, the phase standard is G_cr=ΣνG_l−ΔG_fus.
    # ν is mol oxide/mol formula and the stored fusion node is kJ/mol formula,
    # so multiplying the latter by 1000 makes both terms J/mol formula.
    for phase_id, row in sorted(crystal_by_id.items()):
        state, missing = _evaluate_fusion_crystal_row(
            phase_id, row, temperature_K, gas_pack,
        )
        states.append(state)
        if missing:
            unavailable.append(phase_id)

    # Barin Chapter 4.4 defines standard states at 1 bar and Chapter 4.5 sets
    # elemental reference phases at 1 bar (p. I-36), matching the solver; no
    # 1 atm-to-1 bar oxygen-reference correction is needed. Barin gives crystal
    # formation Gibbs energies from the elements, while
    # the solver needs G_crystal on the selected parent-liquid reference. The
    # balanced Hess cycle is ΔG_fus = Σν_i ΔfG°_JANAF(parent_i,l) − ΔfG°_Barin(cr),
    # so G_crystal = Σν_i G_parent_i,l − ΔG_fus. The source functions are
    # kJ/mol formula unit; convert ΔG_fus by 1000 before combining with the
    # J/mol parent standards. Sanity: at 1000 K, CA has Barin ΔfG=−1931.566
    # kJ/mol and JANAF CaO(l)+Al2O3(l) gives −1799.692 kJ/mol, hence
    # ΔG_fus=+131.874 kJ/mol, matching the stored node.
    for phase_id, row in sorted(barin_by_id.items()):
        state, missing = _evaluate_fusion_crystal_row(
            phase_id, row, temperature_K, gas_pack,
        )
        states.append(state)
        if missing:
            unavailable.append(phase_id)

    # JANAF pure-oxide G values use the single packaged reconstruction and its
    # stable crystal cap, preserving the transition-row convention.
    for phase_id, row in sorted(pure_by_id.items()):
        try:
            crystal = evaluate_janaf_crystal(phase_id, temperature_K, apply_phase_cap=True)
        except JanafCrystalTemperatureError as exc:
            authority = str(row.get("source_authority", "external_datapack"))
            oxide = next(iter(row["oxide_stoichiometry"]))
            try:
                uncapped = evaluate_janaf_crystal(
                    phase_id, temperature_K, apply_phase_cap=False
                )
                liquid_g = gas.species_thermo(oxide, "l", temperature_K, gas_pack).G_J_mol
            except (JanafCrystalTemperatureError, gas.ImccGasTemperatureOutsideDomainError, ValueError):
                unavailable.append(phase_id)
                absent_by_construction = False
            else:
                # For a pure oxide, affinity=(Gl−Gcr)/(RT)+ln(a). Since a≤1,
                # Gl−Gcr<0 proves affinity<0 for every feasible melt. Compare
                # against the selected liquid standard (not the JANAF pair);
                # at 1800 K K2O/Na2O are negative while FeO/DG46-MASSON65 is
                # positive, a sanity check that FeO remains unproven.
                absent_by_construction = liquid_g - uncapped.G_J_mol < 0.0
                if not absent_by_construction:
                    unavailable.append(phase_id)
            placeholder = CrystalCandidate(
                phase_id, row["oxide_stoichiometry"], 0.0, phase_id,
                authority, exc.domain_K, {},
            )
            states.append(_CandidateState(
                placeholder, authority, exc.domain_K, False,
                "absent_by_construction" if absent_by_construction else "temperature_out_of_range",
            ))
            continue
        candidate = CrystalCandidate(
            phase_id, row["oxide_stoichiometry"], crystal.G_J_mol, phase_id,
            str(crystal.provenance["source_authority"]), crystal.domain_K,
            crystal.provenance,
        )
        states.append(_CandidateState(candidate, candidate.source_authority, crystal.domain_K, True))
    return tuple(states), crystal_digest, pure_digest, gas_pack, tuple(sorted(set(unavailable)))


def _resolve_candidates(
    candidates: Sequence[str | CrystalCandidate] | None,
    temperature_K: float,
    gas_pack: Any | None = None,
) -> tuple[tuple[_CandidateState, ...], str, str, Any, tuple[str, ...], bool]:
    defaults, crystal_digest, pure_digest, gas_pack, unavailable = (
        _default_candidate_states(temperature_K, gas_pack)
    )
    if candidates is None:
        return defaults, crystal_digest, pure_digest, gas_pack, unavailable, False
    by_id = {state.candidate.phase_id: state for state in defaults}
    selected: list[_CandidateState] = []
    for item in candidates:
        if isinstance(item, CrystalCandidate):
            available = (
                item.temperature_range_K is None
                or item.temperature_range_K[0] <= temperature_K <= item.temperature_range_K[1]
            )
            selected.append(_CandidateState(
                item, item.source_authority, item.temperature_range_K, available,
                None if available else "temperature_out_of_range",
            ))
        elif isinstance(item, str) and item in by_id:
            selected.append(by_id[item])
        else:
            raise KeyError(f"unknown crystal candidate {item!r}")
    ids = [state.candidate.phase_id for state in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate crystal phase IDs must be unique")
    return (
        tuple(selected), crystal_digest, pure_digest, gas_pack,
        tuple(
            state.candidate.phase_id for state in selected
            if not state.available and state.reason != "above_source_melting_point"
        ),
        True,
    )


def _parse_bulk(
    bulk: Mapping[str, float] | Sequence[float],
    parents: Sequence[str],
) -> tuple[dict[str, float], tuple[str, ...]]:
    if isinstance(bulk, Mapping):
        values = {str(name): float(value) for name, value in bulk.items()}
        unknown = set(values) - set(parents)
        if unknown:
            raise KeyError(f"unsupported parent names: {sorted(unknown)}")
        complete = {name: values.get(name, 0.0) for name in parents}
    else:
        raw = np.asarray(bulk, dtype=object)
        if raw.ndim != 1 or raw.size != len(parents):
            raise ValueError(f"bulk sequence must have {len(parents)} parent entries")
        complete = {name: float(value) for name, value in zip(parents, raw, strict=True)}
    if any(not math.isfinite(value) or value < 0.0 for value in complete.values()):
        raise ValueError("bulk amounts must be finite and non-negative")
    if not any(value > 0.0 for value in complete.values()):
        raise ValueError("bulk must contain at least one positive parent amount")
    active = tuple(name for name in parents if complete[name] > 0.0)
    return {name: complete[name] for name in active}, active


def _candidate_matrix(
    states: Sequence[_CandidateState],
    parents: Sequence[str],
    known_parents: Sequence[str],
    bulk: Mapping[str, float],
) -> tuple[tuple[_CandidateState, ...], np.ndarray, np.ndarray, tuple[str, ...]]:
    active_states: list[_CandidateState] = []
    columns: list[list[float]] = []
    excluded: list[str] = []
    parent_set = set(parents)
    known_parent_set = set(known_parents)
    for state in states:
        candidate = state.candidate
        if _redox_candidate(candidate):
            raise ValueError(f"redox-sensitive crystal {candidate.phase_id!r} is unsupported")
        if not state.available:
            continue
        if set(candidate.stoichiometry) - known_parent_set:
            excluded.append(candidate.phase_id)
            continue
        if set(candidate.stoichiometry) - parent_set:
            # A known parent with exact-zero bulk cannot enter a feasible
            # line-compound amount; its phase is incompatible, not missing data.
            continue
        if any(bulk.get(name, 0.0) == 0.0 for name in candidate.stoichiometry):
            # Exact-zero components make these line-compound columns infeasible.
            continue
        active_states.append(state)
        columns.append([float(candidate.stoichiometry.get(name, 0.0)) for name in parents])
    matrix = np.asarray(columns, dtype=np.float64).T if columns else np.zeros((len(parents), 0))
    gibbs = np.asarray([state.candidate.G_crystal_J_mol for state in active_states], dtype=np.float64)
    return tuple(active_states), matrix, gibbs, tuple(excluded)


def _logk(pack: Any, temperature_K: float) -> np.ndarray:
    kernel = pack.kernel_datapack if isinstance(pack, ImccLoadedDatapack) else pack
    # The pack stores log10 K=A+B/T; multiplication by ln(10) gives natural log.
    return math.log(10.0) * (np.asarray(kernel.A, dtype=float) + np.asarray(kernel.B, dtype=float) / temperature_K)


def _logq_from_potentials(
    mu: np.ndarray,
    liquid_gibbs: np.ndarray,
    temperature_K: float,
    pack: Any,
    parent_indices: Sequence[int],
) -> tuple[float, np.ndarray]:
    # For an ideal associate melt, each free parent contributes a_i and each
    # complex contributes K_j∏a_i^νij. Taking log turns their sum into a stable
    # log-sum-exp; differentiating gives the softmax-weighted ν/(RT), in
    # inverse J/mol. With one parent and no complexes, this reduces to log(a).
    kernel = pack.kernel_datapack if isinstance(pack, ImccLoadedDatapack) else pack
    active = np.asarray(parent_indices, dtype=int)
    nu_full = np.asarray(kernel.nu, dtype=float)
    nu = nu_full[active, :]
    inactive = np.asarray([i for i in range(kernel.n_parents) if i not in set(active)], dtype=int)
    log_activity = (mu - liquid_gibbs) / (R_J_MOL_K * temperature_K)
    terms = list(map(float, log_activity))
    included_nu: list[np.ndarray] = []
    logk = _logk(pack, temperature_K)
    for j in range(nu.shape[1]):
        # A complex requiring an exact-zero component has exactly zero weight.
        if inactive.size and np.any(nu_full[inactive, j] > 0.0):
            continue
        terms.append(float(logk[j] + nu[:, j] @ log_activity))
        included_nu.append(nu[:, j])
    logits = np.asarray(terms, dtype=float)
    value = float(logsumexp(logits))
    weights = np.exp(logits - value)
    gradient = weights[:len(active)] / (R_J_MOL_K * temperature_K)
    for index, stoichiometry in enumerate(included_nu, start=len(active)):
        gradient += weights[index] * stoichiometry / (R_J_MOL_K * temperature_K)
    return value, gradient


def _logq_roundoff_bound(
    log_activity: np.ndarray,
    temperature_K: float,
    pack: Any,
    parent_indices: Sequence[int],
    standard_scales: Sequence[float] = (),
) -> float:
    kernel = pack.kernel_datapack if isinstance(pack, ImccLoadedDatapack) else pack
    active = np.asarray(parent_indices, dtype=int)
    nu_full = np.asarray(kernel.nu, dtype=float)
    nu = nu_full[active, :]
    inactive = [i for i in range(kernel.n_parents) if i not in set(active)]
    logk = _logk(pack, temperature_K)
    # Rebuild Q's log-sum-exp terms. Including its normalized weights bounds
    # exp/sum/log rounding even when logQ is exactly zero; standard_scales
    # carries μ/(RT) and G°/(RT) so cancellation in (μ−G°)/(RT) is counted.
    terms = list(map(float, log_activity))
    included_logk: list[float] = []
    included_logk_inputs: list[float] = []
    products: list[float] = []
    for j in range(nu.shape[1]):
        if any(nu_full[i, j] > 0.0 for i in inactive):
            continue
        products.extend(map(float, nu[:, j] * log_activity))
        included_logk.append(float(logk[j]))
        included_logk_inputs.extend((
            float(kernel.A[j]),
            float(kernel.B[j] / temperature_K),
            float(logk[j] / math.log(10.0)),
        ))
        terms.append(float(logk[j] + nu[:, j] @ log_activity))
    values = np.asarray(terms, dtype=float)
    logq = float(logsumexp(values))
    weights = np.exp(values - logq)
    return _roundoff_bound(
        [*map(float, values), *map(float, weights), *included_logk,
         *included_logk_inputs, *products, *map(float, standard_scales)],
        24 * max(1, len(values) + len(standard_scales) + len(products)),
    )


def _roundoff_bound(values: Sequence[float], operations: int) -> float:
    # γ_m Σ|x| bounds floating-point accumulation error; no unit-sized floor is
    # inserted, so a rounded zero never becomes an unsupported certificate.
    eps = np.finfo(np.float64).eps
    meps = operations * eps
    if meps >= 1.0:
        return math.inf
    return (meps / (1.0 - meps)) * math.fsum(abs(float(value)) for value in values)


def _lp_solid_minimum(
    A: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    balance_tol: float = _DEFAULT_BALANCE_TOL,
) -> _LPState | None:
    # The primal is min gᵀn subject to An=b,n≥0; its dual is max bᵀμ with
    # Aᵀμ≤g. g and μ are J/mol formula and J/mol parent respectively, so both
    # objectives are J for the supplied molar bulk and slack has units J/mol.
    if A.shape[1] == 0:
        return None
    # x_j=n_j/s_j keeps every feasible extent O(1); dividing each parent row
    # by b_i then makes the balance equations independent of bulk scale.
    scales = _scaled_phase_amounts(A, b)
    row_scales = b.copy()
    scaled_A = A * scales[None, :] / row_scales[:, None]
    scaled_cost = g * scales
    cost_scale = float(np.max(np.abs(scaled_cost), initial=0.0))
    if cost_scale > 0.0:
        scaled_cost = scaled_cost / cost_scale
    solved = linprog(
        scaled_cost, A_eq=scaled_A, b_eq=np.ones_like(b),
        bounds=(0.0, None), method="highs",
    )
    if not solved.success:
        return None
    amounts = np.asarray(solved.x, dtype=float) * scales
    marginals = np.asarray(solved.eqlin.marginals, dtype=float)
    mu = marginals * cost_scale / row_scales if cost_scale > 0.0 else np.zeros_like(b)
    objective = float(g @ amounts)
    slack = g - A.T @ mu
    amount_ranges: list[tuple[float, float]] = []
    amount_ranges_certified = True
    # Add normalized gᵀn=Gmin to the scaled balance rows. Each equality row is
    # normalized by its largest coefficient/RHS so HiGHS sees comparable units.
    objective_coefficients = scaled_cost.copy()
    objective_rhs = objective / cost_scale if cost_scale > 0.0 else 0.0
    objective_row_scale = max(
        float(np.max(np.abs(objective_coefficients), initial=0.0)),
        abs(objective_rhs),
    )
    if objective_row_scale > 0.0:
        objective_coefficients = objective_coefficients / objective_row_scale
        objective_rhs /= objective_row_scale
    face_A = np.vstack((scaled_A, objective_coefficients))
    face_b = np.concatenate((np.ones_like(b), [objective_rhs]))
    for j in range(A.shape[1]):
        cost = np.zeros(A.shape[1]); cost[j] = 1.0
        low = linprog(cost, A_eq=face_A, b_eq=face_b, bounds=(0.0, None), method="highs")
        high = linprog(-cost, A_eq=face_A, b_eq=face_b, bounds=(0.0, None), method="highs")
        if not low.success or not high.success:
            amount_ranges.append((0.0, math.inf))
            amount_ranges_certified = False
            continue
        endpoints = (np.asarray(low.x) * scales, np.asarray(high.x) * scales)
        endpoint_moles = (float(low.fun) * scales[j], float(-high.fun) * scales[j])
        valid = True
        for endpoint in endpoints:
            inventory = A @ endpoint
            rel = np.abs(inventory - b) / b
            objective_error = abs(float(g @ endpoint) - objective)
            objective_bound = _roundoff_bound(
                [*map(float, g * endpoint), objective], 8 * (len(g) + len(b))
            )
            if np.max(rel, initial=0.0) > balance_tol or objective_error > objective_bound:
                valid = False
        lo, hi = endpoint_moles
        if not valid or not math.isfinite(lo + hi) or lo < -balance_tol * scales[j]:
            amount_ranges.append((0.0, math.inf))
            amount_ranges_certified = False
        else:
            amount_ranges.append((max(0.0, lo), max(0.0, hi)))
    return _LPState(
        amounts, mu, objective, slack, tuple(amount_ranges), amount_ranges_certified
    )


def _dual_potential_ranges(
    A: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    objective: float,
    temperature_K: float,
) -> tuple[tuple[tuple[float, float], ...], bool]:
    """Bound each chemical potential independently on the dual optimal face.

    Dividing μ and g by RT makes the inequalities dimensionless. The face is
    Aᵀx≤g/(RT), bᵀx=Gmin/(RT), x=μ/(RT); coordinate LPs then give genuine
    independent extrema instead of an arbitrary LP dual solution. Endpoint
    feasibility is checked again in the original units before certification.
    """
    rt = R_J_MOL_K * temperature_K
    scaled_g = g / rt
    bulk_scale = float(np.max(np.abs(b), initial=0.0))
    if not math.isfinite(bulk_scale) or bulk_scale <= 0.0:
        return tuple((-math.inf, math.inf) for _ in range(A.shape[0])), False
    # Normalize both sides of the dual-face equality by the same bulk scale.
    # This keeps HiGHS' feasibility test independent of the user's mole units;
    # the original-unit endpoint check below remains the certificate.
    normalized_bulk = b / bulk_scale
    scaled_objective = objective / (rt * bulk_scale)
    equality = normalized_bulk[None, :]
    equality_rhs = np.asarray([scaled_objective], dtype=float)
    inequality = A.T
    result: list[tuple[float, float]] = []
    certified = True

    for parent_index in range(A.shape[0]):
        cost = np.zeros(A.shape[0], dtype=float)
        cost[parent_index] = 1.0
        endpoints: list[float] = []
        endpoint_valid = True
        for sign in (1.0, -1.0):
            solved = linprog(
                sign * cost,
                A_ub=inequality,
                b_ub=scaled_g,
                A_eq=equality,
                b_eq=equality_rhs,
                bounds=[(None, None)] * A.shape[0],
                method="highs",
                options={"dual_feasibility_tolerance": 1.0e-10,
                         "primal_feasibility_tolerance": 1.0e-10},
            )
            if solved.status == 3:
                endpoints.append(-math.inf if sign > 0.0 else math.inf)
                continue
            if not solved.success:
                endpoint_valid = False
                break
            x = np.asarray(solved.x, dtype=float)
            mu = x * rt
            slacks = g - A.T @ mu
            face_error = abs(float(b @ mu) - objective)
            face_scale = math.fsum(abs(float(value)) for value in b * mu) + abs(objective)
            face_bound = (
                _DUAL_FACE_REL_TOL * face_scale
                + _roundoff_bound([*map(float, b * mu), objective], 8 * (len(b) + 1))
            )
            endpoint_valid = endpoint_valid and (
                np.all(np.isfinite(mu))
                and np.min(slacks, initial=0.0) >= -_DUAL_SLACK_TOL_RT * rt
                and face_error <= face_bound
            )
            if not endpoint_valid:
                break
            endpoints.append(float(mu[parent_index]))
        if not endpoint_valid or len(endpoints) != 2:
            result.append((-math.inf, math.inf))
            certified = False
        else:
            result.append((min(endpoints), max(endpoints)))
    return tuple(result), certified


def _minimize_logq_on_dual_face(
    A: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    objective: float,
    start: np.ndarray,
    liquid_gibbs: np.ndarray,
    temperature_K: float,
    pack: Any,
    parent_indices: Sequence[int],
) -> tuple[np.ndarray, float, tuple[float, float], bool, Mapping[str, float]]:
    # Dimensionless potentials x=μ/(RT) make the objective and inequalities
    # well-scaled. Eliminate bᵀx=Gmin/(RT) with x=x0+Ny, N spanning b⊥;
    # this removes the face equality and any redundant fixed inequalities.
    rt = R_J_MOL_K * temperature_K
    face_rhs = objective / rt
    x0 = b * (face_rhs / float(b @ b))
    _u, _singular, vh = np.linalg.svd(b[None, :], full_matrices=True)
    basis = vh[1:, :].T
    x_start = np.asarray(start, dtype=float) / rt
    y_start = basis.T @ (x_start - x0)
    projected = A.T @ basis
    upper_face = (g - A.T @ (rt * x0)) / rt
    row_norms = np.linalg.norm(projected, axis=1)
    variable_rows = row_norms > 100.0 * np.finfo(float).eps
    if np.any((~variable_rows) & (upper_face < -_DUAL_SLACK_TOL_RT)):
        raise ArithmeticError("dual optimal face has an infeasible fixed inequality")
    C = projected[variable_rows]
    d = upper_face[variable_rows]

    cached_point: tuple[float, ...] | None = None
    cached_value_gradient: tuple[float, np.ndarray] | None = None

    def value_gradient(y: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal cached_point, cached_value_gradient
        point = tuple(map(float, y))
        if point != cached_point:
            x = x0 + basis @ y
            value, gradient_mu = _logq_from_potentials(
                rt * x, liquid_gibbs, temperature_K, pack, parent_indices
            )
            cached_point = point
            cached_value_gradient = value, basis.T @ (rt * gradient_mu)
        assert cached_value_gradient is not None
        return cached_value_gradient

    if basis.shape[1] == 0:
        y = np.zeros(0)
        result_success = True
        optimizer_message = "dual face is a point"
    else:
        constraints = () if C.shape[0] == 0 else (
            LinearConstraint(C, np.full(C.shape[0], -np.inf), d),
        )
        result = minimize(
            value_gradient, y_start,
            jac=True, constraints=constraints,
            method="SLSQP", options={"ftol": 1.0e-13, "maxiter": 1000},
        )
        y = np.asarray(result.x, dtype=float)
        result_success = bool(result.success)
        optimizer_message = str(result.message)

    x = x0 + basis @ y
    mu = rt * x
    logq, gradient_mu = _logq_from_potentials(
        mu, liquid_gibbs, temperature_K, pack, parent_indices
    )
    gradient_y = basis.T @ (rt * gradient_mu)
    slack = g - A.T @ mu
    face_residual = abs(float(b @ mu) - objective)
    active = np.flatnonzero(slack <= _DUAL_SLACK_TOL_RT * rt)
    active_normals = basis.T @ A[:, active]
    if active.size:
        multipliers = lsq_linear(
            active_normals, -gradient_y,
            bounds=(np.zeros(active.size), np.full(active.size, np.inf)),
            tol=1.0e-14, lsmr_tol="auto", max_iter=1000,
        ).x
        stationarity_vector = gradient_y + active_normals @ multipliers
        complementarity = float(np.max(
            np.abs(multipliers * slack[active] / rt), initial=0.0
        ))
    else:
        stationarity_vector = gradient_y
        complementarity = 0.0
    stationarity_scaled = (
        float(np.linalg.norm(stationarity_vector, ord=np.inf))
        if stationarity_vector.size else 0.0
    )
    dual_violation = float(np.max(np.maximum(-slack, 0.0), initial=0.0) / rt)
    face_error = _roundoff_bound([*map(float, b * mu), objective], 8 * (len(b) + 1))
    log_activity = (mu - liquid_gibbs) / (R_J_MOL_K * temperature_K)
    arithmetic_error = _logq_roundoff_bound(
        log_activity,
        temperature_K,
        pack,
        parent_indices,
        [*map(float, mu / (R_J_MOL_K * temperature_K)),
         *map(float, liquid_gibbs / (R_J_MOL_K * temperature_K))],
    )
    # The objective interval is the computed value at this supporting point;
    # a point with Q≤1 proves existence, regardless of optimizer display digits.
    interval = (logq - arithmetic_error, logq + arithmetic_error)
    # logQ is convex and the dual-face constraints are linear, so an
    # independently feasible point satisfying the KKT residual budgets is a
    # global minimum even when SLSQP's status flag rejects a degenerate vertex.
    minimum_certified = (
        stationarity_scaled <= _DEFAULT_AFFINITY_TOL_LN
        and complementarity <= _DEFAULT_AFFINITY_TOL_LN
        and dual_violation <= _DUAL_SLACK_TOL_RT
        and face_residual <= face_error
    )
    residuals = {
        "logQ_optimizer_success": float(result_success),
        "logQ_minimum_certified": float(minimum_certified),
        "logQ_optimizer_message": optimizer_message,
        "dual_slack_min_J_mol": float(np.min(slack, initial=0.0)),
        "dual_face_objective_error_J": face_residual,
        "dual_face_roundoff_bound_J": face_error,
        "logQ_stationarity_scaled": stationarity_scaled,
        "logQ_complementarity": complementarity,
        "logQ_roundoff_bound": arithmetic_error,
    }
    certified = minimum_certified and interval[1] <= 0.0
    return mu, logq, interval, certified, residuals


def _subtraction_error(A: np.ndarray, n: np.ndarray, b: np.ndarray) -> np.ndarray:
    # l=b−An uses one dot product and subtraction per component. γ_m bounds
    # cancellation error using the magnitudes of both subtracted quantities.
    eps = np.finfo(np.float64).eps
    operations = 2 * A.shape[1] + 3
    meps = operations * eps
    gamma = meps / (1.0 - meps)
    return gamma * (np.abs(b) + np.abs(A) @ np.abs(n))


def _safe_liquid(
    A: np.ndarray,
    n: np.ndarray,
    b: np.ndarray,
    tol: float,
) -> tuple[np.ndarray | None, np.ndarray]:
    if (
        np.any(~np.isfinite(n))
        or np.any(n < 0.0)
        or np.any(A @ n > b)
    ):
        return None, np.full(b.shape, math.inf)
    liquid = b - A @ n
    error = _subtraction_error(A, n, b)
    if np.any(liquid - error <= 0.0):
        return None, error
    if np.any(error / liquid > tol):
        return None, error
    return liquid, error


def _phase_affinity_from_melt(
    candidate: CrystalCandidate,
    parent_index: Mapping[str, int],
    liquid_gibbs: np.ndarray,
    log_activities: np.ndarray,
    temperature_K: float,
) -> float:
    indices = [parent_index[name] for name in candidate.stoichiometry]
    nu = np.asarray([candidate.stoichiometry[name] for name in candidate.stoichiometry])
    # ΔrG=Gcr−Σν(Gl+RT ln a); the design's positive precipitation affinity is
    # −ΔrG/RT. This is dimensionless, and SI in dex is affinity/ln(10).
    fusion = math.fsum(
        candidate.stoichiometry[name] * liquid_gibbs[index]
        for name, index in zip(candidate.stoichiometry, indices, strict=True)
    ) - candidate.G_crystal_J_mol
    return float(nu @ log_activities[indices] + fusion / (R_J_MOL_K * temperature_K))


def _independent_thermochemical_standards(
    states: Sequence[_CandidateState],
    parents: Sequence[str],
    temperature_K: float,
    gas_pack: Any | None = None,
) -> tuple[np.ndarray, Mapping[str, float]]:
    """Re-resolve packaged functions for the independent KKT verifier.

    Synthetic CrystalCandidate values have no package row and retain their
    caller-supplied G. Packaged JANAF and fusion-node standards are rebuilt
    from their source rows, independently of the solver's candidate objects.
    """
    from openimcc import gas

    if gas_pack is None:
        gas_pack, _ = diagnostics._crystal_thermo_context(None)
    liquid = _melt_liquid_standards(parents, temperature_K, gas_pack)
    pure, _ = _load_json(_PURE_RESOURCE)
    crystals, _ = _load_json(_CRYSTAL_RESOURCE)
    barin, _ = _load_json(_BARIN_RESOURCE)
    pure_ids = {row["id"] for row in pure["phases"]}
    crystal_rows = {row["id"]: row for row in crystals["phases"]}
    barin_rows = {row["id"]: row for row in barin["phases"]}
    crystal_gibbs: dict[str, float] = {}
    for state in states:
        phase = state.candidate.phase_id
        if not state.available:
            continue
        if phase in pure_ids:
            crystal_gibbs[phase] = evaluate_janaf_crystal(
                phase, temperature_K, apply_phase_cap=True
            ).G_J_mol
        elif phase in crystal_rows:
            row = crystal_rows[phase]
            parent_liquid_g = {
                oxide: gas.species_thermo(oxide, "l", temperature_K, gas_pack).G_J_mol
                for oxide in row["oxide_stoichiometry"]
            }
            fusion = diagnostics._interpolate_crystal_fusion_nodes(row, temperature_K)
            crystal_gibbs[phase] = math.fsum(
                float(nu) * parent_liquid_g[oxide]
                for oxide, nu in row["oxide_stoichiometry"].items()
            ) - fusion * 1000.0
        elif phase in barin_rows:
            row = barin_rows[phase]
            parent_liquid_g = {
                oxide: gas.species_thermo(oxide, "l", temperature_K, gas_pack).G_J_mol
                for oxide in row["oxide_stoichiometry"]
            }
            fusion = diagnostics._interpolate_crystal_fusion_nodes(row, temperature_K)
            crystal_gibbs[phase] = math.fsum(
                float(nu) * parent_liquid_g[oxide]
                for oxide, nu in row["oxide_stoichiometry"].items()
            ) - fusion * 1000.0
        else:
            crystal_gibbs[phase] = float(state.candidate.G_crystal_J_mol)
    return liquid, MappingProxyType(crystal_gibbs)


def _positive_melt(
    liquid: np.ndarray,
    parents: Sequence[str],
    temperature_K: float,
    pack: Any,
    inner_tol: float,
    max_iter: int,
    y_init: Sequence[float] | None = None,
) -> Any:
    # The outer solver calls this only after the cancellation bound proves
    # every supplied liquid component positive; no floors or clipping occur.
    composition = {name: float(value) for name, value in zip(parents, liquid, strict=True)}
    return evaluate(
        composition,
        temperature_K,
        pack,
        basis_type="mol",
        allow_extrapolation=False,
        allow_out_of_envelope=True,
        tol=inner_tol,
        max_iter=max_iter,
        y_init=y_init,
    )


def _inner_activity_uncertainty(
    liquid: np.ndarray,
    parents: Sequence[str],
    temperature_K: float,
    pack: Any,
    melt: Any,
    inner_tol: float,
    max_iter: int,
) -> tuple[float, float]:
    """Propagate inner balance residual and arithmetic error into ln(a).

    The melt balance Jacobian is represented by finite differences of the
    converged map from log analytical parent moles to log activities. Its
    infinity norm maps a normalized balance perturbation into an activity
    perturbation. This is independent of the outer crystal affinity/extent
    Jacobian.
    """
    step = 1.0e-5
    jacobian = np.zeros((len(parents), len(parents)), dtype=float)
    try:
        nominal_logs = _melt_logs(melt, parents)
        for column in range(len(parents)):
            plus, minus = liquid.copy(), liquid.copy()
            plus[column] *= math.exp(step)
            minus[column] *= math.exp(-step)
            plus_melt = _positive_melt(
                plus, parents, temperature_K, pack, inner_tol, max_iter
            )
            minus_melt = _positive_melt(
                minus, parents, temperature_K, pack, inner_tol, max_iter
            )
            plus_logs = _melt_logs(plus_melt, parents)
            minus_logs = _melt_logs(minus_melt, parents)
            jacobian[:, column] = (plus_logs - minus_logs) / (2.0 * step)
    except (ImccRefusal, ArithmeticError, FloatingPointError, ValueError, OverflowError):
        return math.inf, math.inf
    residual = float(getattr(melt.convergence, "residual_inf", math.inf))
    total = math.fsum(map(float, liquid))
    fractions = (
        np.asarray(liquid, dtype=float) / total
        if total > 0.0 else np.zeros_like(liquid)
    )
    minimum_fraction = float(np.min(fractions))
    D = float(getattr(melt, "D", math.nan))
    if (
        not math.isfinite(residual)
        or not math.isfinite(total)
        or total <= 0.0
        or minimum_fraction <= 0.0
        or not math.isfinite(D)
        or D <= 0.0
        or np.any(~np.isfinite(jacobian))
    ):
        return math.inf, math.inf

    # Premise: the kernel's normalized parent-balance residual is
    # f_i = a_i + Σ_j ν_ij g_j − D x_i, where x_i=n_i/Σn and D is
    # dimensionless. Algebra: an input shift δx=f/D would cancel f; because
    # normalized inputs must sum to one, project it to the simplex tangent as
    # δx_i=(f_i−x_i Σf)/D. In log-mole coordinates q_i=δx_i/x_i, so
    # ||q||∞ ≤ ||f||∞/D · (1/min(x)+number_of_parents). Every term is
    # dimensionless. For x=(1e−6,1−1e−6), D=1, and f=(1e−10,−1e−10),
    # this allows a 1e−4 trace-component log-activity shift; multiplying all
    # input moles by one factor leaves x, D, and the bound unchanged.
    # Normalize the arithmetic bound in moles before adding it to f.
    balance_roundoff = _roundoff_bound(
        [*map(float, liquid), *map(float, melt.parent_mol)],
        8 * (len(liquid) + 1),
    ) / total
    balance_input_error = (
        (residual + balance_roundoff) / D
        * (1.0 / minimum_fraction + len(parents))
    )
    jacobian_norm = float(np.linalg.norm(jacobian, ord=np.inf))
    activity_error = jacobian_norm * balance_input_error
    # Include arithmetic in the final log-activity evaluation itself.
    activity_error += _roundoff_bound([*map(float, nominal_logs)], 8 * len(nominal_logs))
    return activity_error, jacobian_norm


def _melt_logs(melt: Any, parents: Sequence[str]) -> np.ndarray:
    values = np.asarray([melt.activity(name) for name in parents], dtype=float)
    if np.any(values < 0.0) or np.any(~np.isfinite(values)):
        raise ArithmeticError("inner melt returned an invalid parent activity")
    if np.any(values == 0.0):
        raise ArithmeticError("positive-parent activity underflowed to zero")
    logs = np.full(values.shape, -math.inf, dtype=float)
    positive = values > 0.0
    logs[positive] = np.log(values[positive])
    return logs


def _scaled_phase_amounts(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    # max_i b_i/νij is the formula-mole extent that would consume the limiting
    # parent if phase j were the only solid; each scale therefore has units mol.
    scales = np.empty(A.shape[1], dtype=float)
    for j in range(A.shape[1]):
        positive = A[:, j] > 0.0
        scales[j] = float(np.max(b[positive] / A[positive, j]))
    return scales


def _fraction_to_boundary(
    n: np.ndarray,
    direction: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
) -> float:
    # For n(α)=n+αd, nonnegativity bounds α by −n_j/d_j where d_j<0, while
    # An(α)≤b bounds it by (b−An)_i/(Ad)_i where (Ad)_i>0. These ratios are
    # dimensionless. The 0.995 factor stays inside a finite boundary.
    bounds = [1.0]
    falling = direction < 0.0
    if np.any(falling):
        bounds.extend((-n[falling] / direction[falling]).tolist())
    parent_step = A @ direction
    rising = parent_step > 0.0
    if np.any(rising):
        bounds.extend(((b - A @ n)[rising] / parent_step[rising]).tolist())
    limit = max(0.0, min(bounds))
    return limit * _FRACTION_TO_BOUNDARY if limit < 1.0 else 1.0


def _active_residual(
    n_active: np.ndarray,
    active: Sequence[int],
    full_n: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    states: Sequence[_CandidateState],
    parents: Sequence[str],
    liquid_gibbs: np.ndarray,
    temperature_K: float,
    pack: Any,
    inner_tol: float,
    max_iter: int,
    balance_tol: float,
    y_init: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, Any] | None:
    trial = full_n.copy()
    if active:
        trial[np.asarray(active, dtype=int)] = n_active
    liquid, _error = _safe_liquid(A, trial, b, balance_tol)
    if liquid is None:
        return None
    melt = _positive_melt(
        liquid, parents, temperature_K, pack, inner_tol, max_iter, y_init
    )
    try:
        logs = _melt_logs(melt, parents)
    except ArithmeticError:
        return None
    index = {name: i for i, name in enumerate(parents)}
    residual = np.asarray([
        _phase_affinity_from_melt(
            states[j].candidate, index, liquid_gibbs, logs, temperature_K
        )
        for j in active
    ], dtype=float)
    return residual, trial, melt


def _solve_active_equalities(
    n: np.ndarray,
    active: list[int],
    A: np.ndarray,
    b: np.ndarray,
    states: Sequence[_CandidateState],
    parents: Sequence[str],
    liquid_gibbs: np.ndarray,
    temperature_K: float,
    pack: Any,
    affinity_tol_ln: float,
    balance_tol: float,
    inner_tol: float,
    max_iter: int,
    y_init: Sequence[float] | None = None,
) -> tuple[np.ndarray, bool, str, dict[str, float]]:
    if not active:
        return n, True, "converged", {}
    scales = _scaled_phase_amounts(A, b)
    diagnostics_map: dict[str, float] = {}
    for outer_iteration in range(_MAX_OUTER_ITERATIONS):
        current = _active_residual(
            n[active], active, n, A, b, states, parents, liquid_gibbs,
            temperature_K, pack, inner_tol, max_iter, balance_tol, y_init,
        )
        if current is None:
            return n, False, "liquid_boundary", diagnostics_map
        residual, _trial, _melt = current
        norm = float(np.linalg.norm(residual, ord=np.inf))
        diagnostics_map["active_affinity_inf"] = norm
        diagnostics_map["active_iterations"] = float(outer_iteration + 1)
        # Finite differences are scaled in formula moles. At a boundary use a
        # feasible one-sided probe; every probe remains in P={n≥0, An≤b}.
        jacobian = np.zeros((len(active), len(active)), dtype=float)
        base = n[active].copy()

        def derivative(column: int, step: float, probe_tol: float) -> np.ndarray | None:
            plus = base.copy(); plus[column] += step
            plus_result = _active_residual(
                plus, active, n, A, b, states, parents, liquid_gibbs,
                temperature_K, pack, probe_tol, max_iter, balance_tol, y_init,
            )
            minus_result = None
            if base[column] >= step:
                minus = base.copy(); minus[column] -= step
                minus_result = _active_residual(
                    minus, active, n, A, b, states, parents, liquid_gibbs,
                    temperature_K, pack, probe_tol, max_iter, balance_tol, y_init,
                )
            if plus_result is not None and minus_result is not None:
                return (plus_result[0] - minus_result[0]) / (2.0 * step)
            if plus_result is not None:
                return (plus_result[0] - residual) / step
            if minus_result is not None:
                return (residual - minus_result[0]) / step
            return None

        for column, phase_index in enumerate(active):
            step = _FD_STEP_SCALED * max(1.0, abs(base[column] / scales[phase_index])) * scales[phase_index]
            full_derivative = derivative(column, step, inner_tol)
            if full_derivative is None:
                return n, False, "no_feasible_jacobian", diagnostics_map
            jacobian[:, column] = full_derivative

        # Compare predicted Newton corrections from h and h/2 in scaled solid
        # coordinates, and compare nominal/tighter inner chemical potentials.
        scaled_full_jacobian = jacobian * scales[np.asarray(active)][None, :]
        scaled_jacobian = scaled_full_jacobian
        lower = -n[np.asarray(active)] / scales[np.asarray(active)]
        # SVD separates the solvable residual row space from extent nullspace.
        # Extent columns are divided by formula-mole scales, so relative
        # singular values are dimensionless. A sqrt(tol) cutoff treats a
        # near-invariant direction as unresolved before Newton amplifies finite-
        # difference noise; the later residual check still enforces tol.
        # The machine threshold protects exact zero matrices. Solving
        # ΣVᵀdx=−Uᵀr minimizes only the consistent row-space residual.
        # Bounds retain feasible extents, while the nullspace records the
        # invariant's lever-rule freedom instead of rejecting a singular J.
        left, singular, right = np.linalg.svd(scaled_jacobian, full_matrices=True)
        largest_singular = float(singular[0]) if singular.size else 0.0
        rank_threshold = (
            max(
                np.finfo(float).eps * max(scaled_jacobian.shape),
                math.sqrt(affinity_tol_ln),
            ) * largest_singular
        )
        rank = int(np.count_nonzero(singular > rank_threshold))
        nullspace = right[rank:, :]
        diagnostics_map["active_jacobian_rank"] = float(rank)
        diagnostics_map["active_jacobian_nullity"] = float(nullspace.shape[0])
        row_fit = (
            left[:, :rank] @ (left[:, :rank].T @ residual)
            if rank else np.zeros_like(residual)
        )
        inconsistent_component = float(np.linalg.norm(residual - row_fit, ord=np.inf))
        diagnostics_map["rank_inconsistent_affinity_inf_ln"] = inconsistent_component
        if rank == 0:
            if norm <= affinity_tol_ln:
                return n, True, "converged_rank_zero", diagnostics_map
            return n, False, "singular_inconsistent_affinity", diagnostics_map
        if norm <= affinity_tol_ln:
            relative_rank_cutoff = rank_threshold / largest_singular if largest_singular else 1.0
            half_jacobian = np.zeros_like(jacobian)
            for column, phase_index in enumerate(active):
                step = _FD_STEP_SCALED * max(
                    1.0, abs(base[column] / scales[phase_index])
                ) * scales[phase_index]
                half_derivative = derivative(column, step / 2.0, inner_tol)
                if half_derivative is None:
                    return n, False, "error_budget_unresolved", diagnostics_map
                half_jacobian[:, column] = half_derivative
            scaled_half_jacobian = half_jacobian * scales[np.asarray(active)][None, :]
            full_correction = np.linalg.lstsq(
                scaled_full_jacobian, -residual, rcond=relative_rank_cutoff
            )[0]
            half_correction = np.linalg.lstsq(
                scaled_half_jacobian, -residual, rcond=relative_rank_cutoff
            )[0]
            fd_correction_error = float(np.linalg.norm(
                scaled_half_jacobian @ (full_correction - half_correction), ord=np.inf
            ))
            _half_left, half_singular, _half_right = np.linalg.svd(
                scaled_half_jacobian, full_matrices=True
            )
            half_rank = int(np.count_nonzero(half_singular > rank_threshold))
            if half_rank == 0 or half_rank != rank:
                return n, False, "error_budget_unresolved", diagnostics_map
            diagnostics_map["fd_correction_error_ln"] = fd_correction_error
            try:
                tighter = _active_residual(
                    n[active], active, n, A, b, states, parents, liquid_gibbs,
                    temperature_K, pack, inner_tol * 0.1, max_iter, balance_tol,
                    y_init,
                )
            except (ImccRefusal, ArithmeticError, FloatingPointError):
                tighter = None
            if tighter is None:
                return n, False, "error_budget_unresolved", diagnostics_map
            tighter_residual, _tighter_trial, tighter_melt = tighter
            normal_logs = _melt_logs(_melt, parents)
            tighter_logs = _melt_logs(tighter_melt, parents)
            inner_activity_comparison = float(
                np.max(np.abs(normal_logs - tighter_logs), initial=0.0)
            )
            liquid_vector, _liquid_roundoff = _safe_liquid(A, _trial, b, balance_tol)
            if liquid_vector is None:
                return n, False, "error_budget_unresolved", diagnostics_map
            inner_activity_error, inner_balance_jacobian_norm = _inner_activity_uncertainty(
                liquid_vector, parents, temperature_K, pack, _melt, inner_tol, max_iter
            )
            inner_coordinate_correction = np.linalg.lstsq(
                scaled_half_jacobian,
                -(tighter_residual - residual),
                rcond=relative_rank_cutoff,
            )[0]
            # Map the scaled extent correction back through J to report its
            # predicted effect on ln affinities, the units used by the 2e-9 budget.
            inner_correction_error = float(np.linalg.norm(
                scaled_half_jacobian @ inner_coordinate_correction, ord=np.inf
            ))
            diagnostics_map["inner_activity_error_ln"] = inner_activity_error
            diagnostics_map["inner_activity_comparison_ln"] = inner_activity_comparison
            diagnostics_map["inner_balance_jacobian_norm"] = inner_balance_jacobian_norm
            diagnostics_map["inner_correction_error_ln"] = inner_correction_error
            if (
                not math.isfinite(fd_correction_error)
                or not math.isfinite(inner_activity_error)
                or not math.isfinite(inner_activity_comparison)
                or not math.isfinite(inner_correction_error)
                or fd_correction_error > _FD_CORRECTION_TOL_LN
                or inner_activity_error > _INNER_ACTIVITY_ERROR_TOL_LN
                or inner_activity_comparison > _INNER_ACTIVITY_ERROR_TOL_LN
                or inner_correction_error > _INNER_CORRECTION_TOL_LN
            ):
                return n, False, "error_budget_unresolved", diagnostics_map
            return n, True, "converged", diagnostics_map
        step_result = lsq_linear(
            singular[:rank, None] * right[:rank, :],
            -(left[:, :rank].T @ residual),
            bounds=(lower, np.full(len(active), np.inf)),
            tol=1.0e-14,
            lsmr_tol="auto",
            max_iter=500,
        )
        direction = np.zeros_like(n)
        direction[np.asarray(active)] = step_result.x * scales[np.asarray(active)]
        scaled_norm = float(np.linalg.norm(step_result.x, ord=np.inf))
        if scaled_norm > 1.0:
            direction *= 1.0 / scaled_norm
        alpha_bound = _fraction_to_boundary(n, direction, A, b)
        if alpha_bound <= np.finfo(float).eps:
            return n, False, "no_feasible_descent", diagnostics_map

        accepted = False
        alpha = alpha_bound
        for _line_search in range(30):
            candidate_n = n + alpha * direction
            trial = _active_residual(
                candidate_n[active], active, candidate_n, A, b, states, parents,
                liquid_gibbs, temperature_K, pack, inner_tol, max_iter,
                balance_tol, y_init,
            )
            if trial is not None:
                next_norm = float(np.linalg.norm(trial[0], ord=np.inf))
                if next_norm < norm or next_norm <= affinity_tol_ln:
                    n = candidate_n
                    accepted = True
                    break
            alpha *= 0.5
        if not accepted:
            # A row-space Newton step can be flat in a singular but consistent
            # system. Accept only if its computed correction is already below
            # the declared affinity budget; otherwise the active set is not solved.
            diagnostics_map["linearized_residual_inf"] = float(
                np.linalg.norm(scaled_jacobian @ step_result.x + residual, ord=np.inf)
            )
            reason = (
                "singular_inconsistent_affinity"
                if rank < len(active)
                and inconsistent_component > affinity_tol_ln
                else "inconsistent_or_stalled"
            )
            return n, False, reason, diagnostics_map
    return n, False, "iteration_limit", diagnostics_map


def _polymorph_filter(
    states: Sequence[_CandidateState],
    temperature_K: float,
    affinity_tol_ln: float,
) -> tuple[tuple[_CandidateState, ...], Mapping[str, tuple[str, ...]], Mapping[str, str]]:
    # Equal stoichiometry has identical composition dependence, so only the
    # lowest G° can be stable. A ΔG band of RT·tol treats numerically unresolved
    # crossings as aliases instead of choosing an arbitrary polymorph split.
    grouped: dict[tuple[float, ...], list[_CandidateState]] = {}
    for state in states:
        if not state.available:
            continue
        key = tuple(sorted(
            (name, float(amount))
            for name, amount in state.candidate.stoichiometry.items()
        ))
        grouped.setdefault(key, []).append(state)
    selected: list[_CandidateState] = []
    aliases: dict[str, tuple[str, ...]] = {}
    rejected: dict[str, str] = {}
    for group in grouped.values():
        ordered = sorted(group, key=lambda item: item.candidate.phase_id)
        lowest = min(item.candidate.G_crystal_J_mol for item in ordered)
        band = affinity_tol_ln * R_J_MOL_K * temperature_K
        stable = [item for item in ordered if item.candidate.G_crystal_J_mol - lowest <= band]
        # Select the exact minimum for the LP; phase ID breaks only an exact
        # Gibbs tie, while the tolerance band marks unresolved aliases.
        minima = [item for item in stable if item.candidate.G_crystal_J_mol == lowest]
        representative = min(minima, key=lambda item: item.candidate.phase_id)
        selected.append(representative)
        if len(stable) > 1:
            aliases[representative.candidate.phase_id] = tuple(
                item.candidate.phase_id for item in stable
            )
        for item in ordered:
            if item not in stable:
                rejected[item.candidate.phase_id] = representative.candidate.phase_id
    return tuple(sorted(selected, key=lambda item: item.candidate.phase_id)), aliases, rejected


def _phase_extent_ranges(
    A: np.ndarray,
    n: np.ndarray,
    indices: Sequence[int],
    residual_inventory: np.ndarray,
    scales: np.ndarray,
    tol: float,
    *,
    bulk_vector: np.ndarray | None = None,
    liquid_vector: np.ndarray | None = None,
) -> tuple[dict[int, tuple[float, float]], bool, bool, tuple[float, float]]:
    # At an invariant, scaling the liquid inventory at fixed parent proportions
    # leaves every activity unchanged: ideal-melt activities depend on normalized
    # composition, not total moles. Add that liquid-scale direction to the
    # balance face so LP endpoints recover lever-rule ranges instead of freezing
    # the arbitrary liquid amount from the Newton iterate.
    result: dict[int, tuple[float, float]] = {}
    certified = True
    if bulk_vector is None or liquid_vector is None:
        if not indices:
            return result, True, True, (0.0, 0.0)
        bulk_vector = residual_inventory.copy()
        liquid_vector = np.zeros_like(bulk_vector)
    liquid_total = float(np.sum(liquid_vector))
    if liquid_total <= 0.0:
        return {index: (0.0, math.inf) for index in indices}, False, False, (0.0, math.inf)
    liquid_proportions = liquid_vector / liquid_total
    bulk_scale = float(np.sum(bulk_vector))
    if not math.isfinite(bulk_scale) or bulk_scale <= 0.0:
        return {index: (0.0, math.inf) for index in indices}, False, False, (0.0, math.inf)
    selected_A = A[:, np.asarray(indices, dtype=int)]
    phase_scales = scales[np.asarray(indices, dtype=int)]
    variable_scales = np.concatenate((phase_scales, [bulk_scale]))
    face_matrix = np.column_stack((selected_A, liquid_proportions))
    scaled_A = face_matrix * variable_scales[None, :] / bulk_vector[:, None]
    # HiGHS defaults to a 1e-7 feasibility tolerance, which can accept an
    # endpoint outside this solver's tighter balance budget near liquidus.
    # Its supported floor is 1e-10; every returned endpoint is still checked
    # independently against the caller's balance tolerance in parent moles.
    lp_tolerance = max(1.0e-10, min(1.0e-4, tol))
    lp_options = {
        "primal_feasibility_tolerance": lp_tolerance,
        "dual_feasibility_tolerance": lp_tolerance,
    }
    for local, global_index in enumerate(indices):
        cost = np.zeros(len(indices) + 1); cost[local] = 1.0
        low = linprog(
            cost, A_eq=scaled_A, b_eq=np.ones_like(bulk_vector),
            bounds=(0.0, None), method="highs", options=lp_options,
        )
        high = linprog(
            -cost, A_eq=scaled_A, b_eq=np.ones_like(bulk_vector),
            bounds=(0.0, None), method="highs", options=lp_options,
        )
        if low.success and high.success:
            endpoints = (np.asarray(low.x) * variable_scales,
                         np.asarray(high.x) * variable_scales)
            amounts = (float(low.x[local]) * phase_scales[local],
                       float(high.x[local]) * phase_scales[local])
            valid = True
            for endpoint in endpoints:
                inventory = face_matrix @ endpoint
                relative_error = np.abs(inventory - bulk_vector) / bulk_vector
                if np.max(relative_error, initial=0.0) > tol:
                    valid = False
            lo, hi = amounts
            result[global_index] = (
                (max(0.0, lo), max(0.0, hi)) if valid else (0.0, math.inf)
            )
            certified = certified and valid
        else:
            result[global_index] = (0.0, math.inf)
            certified = False
    liquid_cost = np.zeros(len(indices) + 1, dtype=float)
    liquid_cost[-1] = 1.0
    liquid_low = linprog(
        liquid_cost, A_eq=scaled_A, b_eq=np.ones_like(bulk_vector),
        bounds=(0.0, None), method="highs", options=lp_options,
    )
    liquid_high = linprog(
        -liquid_cost, A_eq=scaled_A, b_eq=np.ones_like(bulk_vector),
        bounds=(0.0, None), method="highs", options=lp_options,
    )
    if liquid_low.success and liquid_high.success:
        liquid_range = (
            max(0.0, float(liquid_low.x[-1]) * bulk_scale),
            max(0.0, float(liquid_high.x[-1]) * bulk_scale),
        )
        for endpoint in (np.asarray(liquid_low.x) * variable_scales,
                         np.asarray(liquid_high.x) * variable_scales):
            relative_error = np.abs(face_matrix @ endpoint - bulk_vector) / bulk_vector
            certified = certified and np.max(relative_error, initial=0.0) <= tol
    else:
        liquid_range = (0.0, math.inf)
        certified = False
    unique = certified and all(
        high - low <= tol * scales[index] + _roundoff_bound((low, high), 8)
        for index, (low, high) in result.items()
    )
    return result, unique, certified, liquid_range


def _make_saturation_rows(
    states: Sequence[_CandidateState],
    affinities: Mapping[str, float],
    temperature_K: float,
) -> tuple[diagnostics.SaturationIndex, ...]:
    # SI is log10(Q_phase), while the affinity map is dimensionless ln(Q_phase).
    rows = []
    for state in states:
        candidate = state.candidate
        has_affinity = state.available and candidate.phase_id in affinities
        lo, hi = state.domain or (temperature_K, temperature_K)
        rows.append(diagnostics.SaturationIndex(
            phase=candidate.phase_id,
            formula=candidate.formula or candidate.phase_id,
            parent_stoichiometry=candidate.stoichiometry,
            source=candidate.source,
            temperature_range_K=(float(lo), float(hi)),
            temperature_K=temperature_K,
            SI=(float(affinities[candidate.phase_id] / math.log(10.0)) if has_affinity else None),
            status=(
                diagnostics.CrystalStatus.AVAILABLE
                if state.available else diagnostics.CrystalStatus.OUT_OF_RANGE
            ),
            solid_solution_bound=False,
        ))
    return tuple(rows)


def _result_receipt(
    *,
    bulk: Mapping[str, float],
    temperature_K: float,
    pack: Any,
    crystal_digest: str,
    pure_digest: str,
    candidates: Sequence[_CandidateState],
    solver: Mapping[str, Any],
    gas_pack: Any | None = None,
) -> CalculationReceipt:
    kernel = pack.kernel_datapack if isinstance(pack, ImccLoadedDatapack) else pack
    _barin_data, barin_digest = _load_json(_BARIN_RESOURCE)
    return build_calculation_receipt(
        request={
            "bulk_parent_mol": dict(bulk),
            "temperature_K": temperature_K,
            "pressure_bar": _PRESSURE_BAR,
            "candidate_phase_ids": tuple(state.candidate.phase_id for state in candidates),
        },
        data_digests={
            "melt_binding": getattr(pack, "binding_digest", getattr(kernel, "binding_digest", None)),
            "crystal_phases": crystal_digest,
            "barin_aluminate_crystals": barin_digest,
            "janaf_pure_oxide_crystals": pure_digest,
            "parent_liquid_standards": (
                gas_pack.condensate_table_digest if gas_pack is not None else
                diagnostics._crystal_thermo_context(None)[0].condensate_table_digest
            ),
        },
        solver={
            "method": "nested_feasible_active_set_with_solid_lp",
            "bulk_balance_relative_tolerance": _DEFAULT_BALANCE_TOL,
            "inner_balance_relative_tolerance": _INNER_BALANCE_TOL,
            "outer_affinity_tolerance_ln": _DEFAULT_AFFINITY_TOL_LN,
            "finite_difference_correction_tolerance_ln": _FD_CORRECTION_TOL_LN,
            "inner_activity_error_tolerance_ln": _INNER_ACTIVITY_ERROR_TOL_LN,
            "inner_correction_tolerance_ln": _INNER_CORRECTION_TOL_LN,
            "dual_slack_tolerance_over_RT": _DUAL_SLACK_TOL_RT,
            "dual_face_relative_tolerance": _DUAL_FACE_REL_TOL,
            "activity_identifiability_tolerance_ln": _INNER_ACTIVITY_ERROR_TOL_LN,
            "pressure_bar": _PRESSURE_BAR,
            "crystals_above_source_melting_point": tuple(sorted(
                state.candidate.phase_id for state in candidates
                if state.reason == "above_source_melting_point"
            )),
            **dict(solver),
        },
    )


def _labels(
    pack: Any,
    states: Sequence[_CandidateState],
    missing: Sequence[str],
    explicit_candidates: bool,
    flags: Sequence[str],
    notices: Sequence[str],
    validity: Literal["IN", "OUT_OF_DOMAIN", "DEPENDS"],
    axes: Sequence[str] = (),
    parent_oxides: Sequence[str] = (),
) -> PhaseEquilibriumLabels:
    kernel = pack.kernel_datapack if isinstance(pack, ImccLoadedDatapack) else pack
    authorities = [
        state.authority for state in states
        if state.available or state.reason == "above_source_melting_point"
    ]
    above_source_melting = tuple(sorted(
        state.candidate.phase_id for state in states
        if state.reason == "above_source_melting_point"
    ))
    authorities.extend(
        _OXIDE_PROVENANCE_AUTHORITY.get(parent, "external_datapack")
        for parent in parent_oxides
    )
    authorities.append(str(getattr(kernel, "evidence_class", "external_datapack")))
    if isinstance(pack, ImccLoadedDatapack):
        authorities.extend(
            str(term.provenance.get("source_authority", "external_datapack"))
            if isinstance(term.provenance, Mapping) else "external_datapack"
            for term in getattr(pack, "excess_gibbs_terms", ())
        )
    return PhaseEquilibriumLabels(
        identity={
            "melt_model": str(getattr(kernel, "model_id", "unresolved")),
            "melt_binding_digest": str(getattr(pack, "binding_digest", getattr(kernel, "binding_digest", "unresolved"))),
            "crystal_phases": "openimcc-crystal-phases-v1",
            "barin_aluminate_crystals": "openimcc-barin-aluminate-crystals-v1",
            "janaf_pure_oxide_crystals": "openimcc-janaf-apparent-crystals-v2",
            "standard_pressure_bar": "1.0",
        },
        coverage={
            state.candidate.phase_id: (
                f"{state.domain[0]:g}-{state.domain[1]:g} K"
                if state.available and state.domain else state.reason or "unavailable"
            )
            for state in states
        } | {
            phase: "missing_phase_data"
            for phase in missing
            if phase not in {state.candidate.phase_id for state in states}
        },
        validity=validity,
        out_of_domain_axes=tuple(axes),
        evidence_class=_least_authority(authorities),
        flags=tuple(dict.fromkeys(flags)),
        notices=tuple(dict.fromkeys((
            *notices,
            *(
                ("crystals_above_source_melting_point:" + ",".join(above_source_melting),)
                if above_source_melting else ()
            ),
            *(("restricted_candidate_set",) if explicit_candidates else ()),
            *(("missing_phase_data",) if missing else ()),
        ))),
    )


def _mass_fraction(
    liquid: Mapping[str, float] | None,
    bulk: Mapping[str, float],
) -> float | None:
    if liquid is None:
        return None
    # Weight each oxide-parent mole by g/mol; common formula-unit scaling cancels
    # in the liquid-mass / total-mass ratio.
    total = math.fsum(amount * MOLAR_MASS_G_MOL[name] for name, amount in bulk.items())
    if total <= 0.0:
        return None
    return math.fsum(
        amount * MOLAR_MASS_G_MOL[name] for name, amount in liquid.items()
    ) / total


def _empty_result(
    *,
    bulk: Mapping[str, float],
    temperature_K: float,
    pack: Any,
    states: Sequence[_CandidateState],
    crystal_digest: str,
    pure_digest: str,
    explicit_candidates: bool,
    refusal: EquilibriumRefusal,
    validity: Literal["IN", "OUT_OF_DOMAIN", "DEPENDS"] = "DEPENDS",
    flags: Sequence[str] = (),
    notices: Sequence[str] = (),
    axes: Sequence[str] = (),
    solver: Mapping[str, Any] | None = None,
    crystal_only_minimum: bool = False,
    solid_ranges: Mapping[str, tuple[float, float]] | None = None,
) -> PhaseEquilibriumResult:
    labels = _labels(
        pack, states, (), explicit_candidates, (*flags, refusal.code), notices,
        validity, axes, tuple(bulk),
    )
    receipt = _result_receipt(
        bulk=bulk, temperature_K=temperature_K, pack=pack,
        crystal_digest=crystal_digest, pure_digest=pure_digest,
        candidates=states, solver=solver or {},
    )
    phase_ids = tuple(state.candidate.phase_id for state in states)
    result_ranges = dict(solid_ranges or {})
    for state in states:
        if state.reason in ("absent_by_construction", "above_source_melting_point"):
            result_ranges[state.candidate.phase_id] = (0.0, 0.0)
    return PhaseEquilibriumResult(
        temperature_K=temperature_K,
        pressure_bar=_PRESSURE_BAR,
        bulk_parent_mol=bulk,
        liquid_parent_mol=None,
        liquid_mass_fraction=None,
        solid_mol=None,
        phase_amount_ranges_mol=(
            result_ranges or {phase: (0.0, 0.0) for phase in phase_ids}
        ),
        phase_amounts_unique=False,
        phase_presence={
            state.candidate.phase_id: (
                "infeasible" if state.reason == "infeasible" else
                "absent" if state.reason in (
                    "absent_by_construction", "above_source_melting_point"
                ) else
                "unresolved"
            )
            for state in states
        },
        saturated_candidates=(),
        identifiable_activities={},
        activity_ranges_ln={},
        phase_affinities_ln={},
        phase_affinity_ranges_ln={},
        phase_saturation_indices=_make_saturation_rows(states, {}, temperature_K),
        logQ=None,
        logQ_interval=None,
        melt=None,
        gas=None,
        mass_balance_relative_inf=None,
        labels=labels,
        receipt=receipt,
        refusal=refusal,
        active_set_history=tuple((solver or {}).get("active_set_history", ())),
        crystal_only_minimum=crystal_only_minimum,
    )


def _independent_kkt_recheck(
    *,
    bulk_vector: np.ndarray,
    liquid_vector: np.ndarray | None,
    solid_vector: np.ndarray,
    solid_states: Sequence[_CandidateState],
    A: np.ndarray,
    states: Sequence[_CandidateState],
    parents: Sequence[str],
    temperature_K: float,
    pack: Any,
    liquid_gibbs: np.ndarray,
    inner_tol: float,
    balance_tol: float,
    affinity_tol_ln: float,
    absent_tol_ln: float,
    dual_mu: np.ndarray | None = None,
    gas_pack: Any | None = None,
    y_init: Sequence[float] | None = None,
) -> tuple[bool, dict[str, float], Any | None, dict[str, float]]:
    # This verifier deliberately reconstructs balances and affinities from the
    # returned vectors and a fresh melt solve; it consumes none of the outer
    # solver's residuals or Jacobians.
    verified_liquid_gibbs, verified_crystal_gibbs = (
        _independent_thermochemical_standards(states, parents, temperature_K, gas_pack)
    )
    balance = bulk_vector - A @ solid_vector
    melt = None
    inner_balance_relative = 0.0
    inner_logq = math.nan
    inner_mass_action = 0.0
    affinities: dict[str, float] = {}
    if liquid_vector is not None:
        balance -= liquid_vector
        if np.any(liquid_vector <= 0.0):
            return False, {"balance_relative_inf": math.inf}, None, affinities
        melt = _positive_melt(
            liquid_vector, parents, temperature_K, pack, inner_tol, 200, y_init
        )
        activities = np.asarray([melt.activity(name) for name in parents], dtype=float)
        if np.any(activities <= 0.0):
            return False, {"inner_activity_underflow": math.inf}, melt, affinities
        logs = np.full(activities.shape, -math.inf)
        positive = activities > 0.0
        logs[positive] = np.log(activities[positive])
        index = {name: i for i, name in enumerate(parents)}
        # Independent scalar SI reconstruction from ΔGfus=ΣνGl−Gcr.
        for state in states:
            if not state.available or set(state.candidate.stoichiometry) - set(parents):
                continue
            c = state.candidate
            crystal_g = verified_crystal_gibbs[c.phase_id]
            dg_fus = math.fsum(
                c.stoichiometry[name] * verified_liquid_gibbs[index[name]]
                for name in c.stoichiometry
            ) - crystal_g
            affinity = math.fsum(
                c.stoichiometry[name] * logs[index[name]]
                for name in c.stoichiometry
            ) + dg_fus / (R_J_MOL_K * temperature_K)
            affinities[c.phase_id] = float(affinity)

        kernel = pack.kernel_datapack if isinstance(pack, ImccLoadedDatapack) else pack
        logk = _logk(pack, temperature_K)
        nu = np.asarray(kernel.nu, dtype=float)
        active_indices = [tuple(kernel.parent_oxides).index(name) for name in parents]
        parent_logs_full = np.asarray([
            math.log(melt.activity(name)) if melt.activity(name) > 0.0 else -math.inf
            for name in kernel.parent_oxides
        ])
        active_parent_logs = parent_logs_full[np.asarray(active_indices, dtype=int)]
        species_logits = list(active_parent_logs)
        for j in range(nu.shape[1]):
            required = np.flatnonzero(nu[:, j] > 0.0)
            if any(not math.isfinite(parent_logs_full[k]) for k in required):
                continue
            species_logits.append(
                float(logk[j] + nu[active_indices, j] @ active_parent_logs)
            )
        inner_logq = float(logsumexp(np.asarray(species_logits)))
        mass_action_errors = []
        complex_x = np.asarray(melt.complex_x, dtype=float)
        for j, fraction in enumerate(complex_x):
            required = np.flatnonzero(nu[:, j] > 0.0)
            if all(math.isfinite(parent_logs_full[k]) for k in required):
                if not math.isfinite(float(fraction)) or fraction <= 0.0:
                    mass_action_errors.append(math.inf)
                else:
                    mass_action_errors.append(
                        math.log(float(fraction))
                        - float(logk[j] + nu[active_indices, j] @ active_parent_logs)
                    )
        species_x = np.asarray(melt.species_x, dtype=float)
        inner_species_sum_error = (
            abs(math.fsum(map(float, species_x)) - 1.0)
            if np.all(np.isfinite(species_x)) and np.all(species_x >= 0.0)
            else math.inf
        )
        # D is analytical parent moles / total species moles; multiplying the
        # normalized species vector by basis/D returns the independent species
        # inventory used to verify the inner parent-balance certificate.
        species_moles = float(melt.basis) / float(melt.D) * species_x
        stoich = np.column_stack((np.eye(len(kernel.parent_oxides)), nu))
        inner_inventory = stoich @ species_moles
        requested = np.asarray(melt.parent_mol, dtype=float)
        positive_inner = requested > 0.0
        inner_balance_relative = float(np.max(
            np.abs(inner_inventory[positive_inner] - requested[positive_inner])
            / requested[positive_inner],
            initial=0.0,
        ))
        inner_mass_action = float(np.max(np.abs(mass_action_errors), initial=0.0))
    elif dual_mu is not None:
        inner_species_sum_error = 0.0
        parent_index = {name: i for i, name in enumerate(parents)}
        for state in states:
            candidate = state.candidate
            if not state.available or set(candidate.stoichiometry) - set(parent_index):
                continue
            # Reconstruct the dual affinity independently as Aᵀμ−g_cr over RT.
            affinities[candidate.phase_id] = (
                math.fsum(
                    candidate.stoichiometry[name] * dual_mu[parent_index[name]]
                    for name in candidate.stoichiometry
                ) - verified_crystal_gibbs[candidate.phase_id]
            ) / (R_J_MOL_K * temperature_K)
        kernel = pack.kernel_datapack if isinstance(pack, ImccLoadedDatapack) else pack
        dual_logs = (dual_mu - verified_liquid_gibbs) / (R_J_MOL_K * temperature_K)
        inner_terms = list(map(float, dual_logs))
        nu_full = np.asarray(kernel.nu, dtype=float)
        active_indices = [tuple(kernel.parent_oxides).index(name) for name in parents]
        inactive = [i for i in range(kernel.n_parents) if i not in set(active_indices)]
        logk = _logk(pack, temperature_K)
        for j in range(nu_full.shape[1]):
            if any(nu_full[i, j] > 0.0 for i in inactive):
                continue
            inner_terms.append(float(logk[j] + nu_full[active_indices, j] @ dual_logs))
        inner_logq = float(logsumexp(np.asarray(inner_terms)))
        q_error = _logq_roundoff_bound(
            dual_logs,
            temperature_K,
            pack,
            active_indices,
            [*map(float, dual_mu / (R_J_MOL_K * temperature_K)),
             *map(float, verified_liquid_gibbs / (R_J_MOL_K * temperature_K))],
        )
        inner_certificate = inner_logq + q_error <= 0.0
    else:
        inner_mass_action = 0.0
        inner_certificate = True
        inner_species_sum_error = 0.0

    positive_bulk = bulk_vector > 0.0
    balance_relative = float(np.max(
        np.abs(balance[positive_bulk]) / bulk_vector[positive_bulk], initial=0.0
    ))
    phase_eq = []
    phase_ineq = []
    amounts_by_phase = {
        state.candidate.phase_id: float(solid_vector[index])
        for index, state in enumerate(solid_states)
    }
    for state in states:
        if not state.available or state.candidate.phase_id not in affinities:
            continue
        amount = amounts_by_phase.get(state.candidate.phase_id, 0.0)
        if amount > 0.0:
            phase_eq.append(abs(affinities[state.candidate.phase_id]))
        else:
            phase_ineq.append(max(affinities[state.candidate.phase_id], 0.0))
    max_present = float(np.max(phase_eq, initial=0.0))
    max_absent = float(np.max(phase_ineq, initial=0.0))
    if liquid_vector is not None:
        inner_certificate = (
            abs(inner_logq) <= _LOGQ_COEXISTENCE_TOL
            and inner_balance_relative <= _INNER_BALANCE_TOL
            and inner_mass_action <= _INNER_BALANCE_TOL
            and inner_species_sum_error <= _INNER_BALANCE_TOL
        )
    residuals = {
        "balance_relative_inf": balance_relative,
        "inner_balance_relative_inf": inner_balance_relative,
        "present_affinity_inf_ln": max_present,
        "absent_affinity_violation_inf_ln": max_absent,
        "inner_mass_action_inf_ln": inner_mass_action,
        "inner_species_sum_error": inner_species_sum_error,
        "inner_logQ": inner_logq,
    }
    passed = (
        balance_relative <= balance_tol
        and max_present <= affinity_tol_ln
        and max_absent <= absent_tol_ln
        and inner_certificate
    )
    return passed, residuals, melt, affinities


def _kernel_parent_names(pack: Any) -> Sequence[str]:
    kernel = pack.kernel_datapack if isinstance(pack, ImccLoadedDatapack) else pack
    return kernel.parent_oxides


def _positive_branch(
    *,
    bulk: Mapping[str, float],
    parents: Sequence[str],
    states: Sequence[_CandidateState],
    A: np.ndarray,
    g: np.ndarray,
    liquid_gibbs: np.ndarray,
    temperature_K: float,
    pack: Any,
    gas_pack: Any,
    candidates_all: Sequence[_CandidateState],
    aliases: Mapping[str, tuple[str, ...]],
    rejected_polymorphs: Mapping[str, str],
    crystal_digest: str,
    pure_digest: str,
    explicit_candidates: bool,
    missing: Sequence[str],
    balance_tol: float,
    affinity_tol_ln: float,
    inner_tol: float,
    max_iter: int,
    warm_start: PhaseEquilibriumResult | None = None,
) -> PhaseEquilibriumResult:
    b = np.asarray([bulk[name] for name in parents], dtype=float)
    n = np.zeros(A.shape[1], dtype=float)
    scales = _scaled_phase_amounts(A, b) if A.shape[1] else np.zeros(0)
    phase_ids = [state.candidate.phase_id for state in states]
    phase_index = {phase: i for i, phase in enumerate(phase_ids)}
    warm_y: np.ndarray | None = None
    if warm_start is not None and warm_start.labels.validity != "OUT_OF_DOMAIN":
        previous_melt = warm_start.melt
        if previous_melt is not None:
            previous_x = np.asarray(getattr(previous_melt, "parent_x_star", ()), dtype=float)
            kernel_names = tuple(_kernel_parent_names(pack))
            if previous_x.size == len(kernel_names):
                selected_x = np.asarray([previous_x[kernel_names.index(name)] for name in parents])
                if np.all(np.isfinite(selected_x)) and np.all(selected_x > 0.0):
                    warm_y = np.log(selected_x)
        if not warm_start.crystal_only_minimum:
            candidate_amounts: Mapping[str, float] = warm_start.solid_mol or {}
            if not candidate_amounts:
                candidate_amounts = warm_start.receipt.solver.get(
                    "crystal_only_candidate_amounts_mol", {}
                )
            warm_ids = {
                phase for phase, presence in warm_start.phase_presence.items()
                if presence == "present"
            }
            for phase in warm_start.saturated_candidates:
                if warm_start.phase_amount_ranges_mol.get(phase, (0.0, 0.0))[1] > 0.0:
                    warm_ids.add(phase)
            for phase in sorted(warm_ids):
                index = phase_index.get(phase)
                if index is None:
                    continue
                amount = float(candidate_amounts.get(phase, 0.0))
                if amount <= 0.0:
                    low, high = warm_start.phase_amount_ranges_mol.get(phase, (0.0, 0.0))
                    amount = (low + high) / 2.0 if math.isfinite(high) else low
                if math.isfinite(amount) and amount > 0.0:
                    n[index] = amount
            consumed = A @ n
            ratios = [float(b[i] / consumed[i]) for i in range(len(b)) if consumed[i] > 0.0]
            if np.any(n > 0.0):
                n *= min(0.9, 0.9 * min(ratios, default=1.0))
            active = sorted(
                (index for index, amount in enumerate(n) if amount > 0.0),
                key=lambda index: phase_ids[index],
            )
        else:
            active = []
    else:
        active = []
    history: list[tuple[str, ...]] = []
    history_seen: set[tuple[str, ...]] = set()
    state_history_seen: set[tuple[tuple[str, ...], tuple[float, ...]]] = set()
    residual_sign_history: list[tuple[tuple[str, int], ...]] = []
    solver_error_history: list[Mapping[str, float]] = []
    cycle_detected = False
    tie_breaks: list[str] = []
    polymorph_tie_breaks = tuple(
        f"{','.join(members)}->{representative}"
        for representative, members in sorted(aliases.items())
    )
    polymorph_tie_breaks += tuple(
        f"{phase}->{winner}"
        for phase, winner in sorted(rejected_polymorphs.items())
    )
    pending_add: int | None = None

    def history_solver(**extra: Any) -> dict[str, Any]:
        return {
            "active_set_history": tuple(history),
            "active_set_residual_signs": tuple(residual_sign_history),
            "outer_error_budget_history": tuple(solver_error_history),
            "polymorph_tie_breaks": polymorph_tie_breaks,
            "deterministic_tie_breaks": tuple(tie_breaks),
            **extra,
        }

    # Parent columns requiring an exact-zero bulk component were removed before
    # this function, so every column below is feasible at a sufficiently small
    # positive extent.
    for _restart in range(_MAX_ACTIVE_RESTARTS):
        active_key = tuple(sorted(phase_ids[index] for index in active))
        scaled_state = tuple(map(float, np.round(n / scales, decimals=14)))
        state_key = (active_key, scaled_state)
        history.append(active_key)
        repeated_state = state_key in state_history_seen
        repeated_transition = (
            len(history) > 1
            and history[-2] != active_key
            and active_key in history_seen
        )
        cycle_detected = cycle_detected or repeated_state or repeated_transition
        state_history_seen.add(state_key)
        history_seen.add(active_key)
        current_liquid, subtraction_error = _safe_liquid(A, n, b, balance_tol)
        if current_liquid is None:
            return _empty_result(
                bulk=bulk, temperature_K=temperature_K, pack=pack,
                states=candidates_all, crystal_digest=crystal_digest,
                pure_digest=pure_digest, explicit_candidates=explicit_candidates,
                refusal=EquilibriumRefusal(
                    "liquid_fraction_unresolved",
                    "Liquid subtraction is not safely positive under its computed error bound.",
                    ("liquid_parent_mol",),
                    {f"liquid_error.{name}": float(subtraction_error[i]) for i, name in enumerate(parents)},
                ),
                validity="DEPENDS",
                solver=history_solver(),
            )
        try:
            melt = _positive_melt(
                current_liquid, parents, temperature_K, pack, inner_tol, max_iter,
                warm_y,
            )
            logs = _melt_logs(melt, parents)
        except (ImccRefusal, ArithmeticError) as exc:
            return _empty_result(
                bulk=bulk, temperature_K=temperature_K, pack=pack,
                states=candidates_all, crystal_digest=crystal_digest,
                pure_digest=pure_digest, explicit_candidates=explicit_candidates,
                refusal=EquilibriumRefusal(
                    "inner_melt_failed" if isinstance(exc, ImccRefusal) else "kkt_residual",
                    str(exc), ("liquid_parent_mol",),
                ),
                solver=history_solver(),
            )
        by_parent = {name: i for i, name in enumerate(parents)}
        affinities: dict[str, float] = {}
        for state in candidates_all:
            candidate = state.candidate
            if not state.available or set(candidate.stoichiometry) - set(parents):
                continue
            if any(logs[by_parent[name]] == -math.inf for name in candidate.stoichiometry):
                affinities[candidate.phase_id] = -math.inf
                continue
            affinities[candidate.phase_id] = _phase_affinity_from_melt(
                candidate, by_parent, liquid_gibbs, logs, temperature_K
            )
        residual_sign_history.append(tuple(
            (phase, 1 if value > 0.0 else -1 if value < 0.0 else 0)
            for phase, value in sorted(affinities.items())
        ))

        # Remove a phase whose extent has resolved to zero. Stable phase IDs
        # make all same-affinity choices repeatable and are recorded in receipt.
        zero_active = [
            index for index in active
            if n[index] <= balance_tol * scales[index]
            and affinities.get(phase_ids[index], math.inf) <= affinity_tol_ln
        ]
        if zero_active:
            for index in sorted(zero_active, key=lambda item: phase_ids[item]):
                n[index] = 0.0
                active.remove(index)
                if pending_add == index:
                    pending_add = None
            tie_breaks.append("drop_resolved_zero:" + ",".join(phase_ids[i] for i in sorted(zero_active)))
            continue

        undersaturated_active = [
            index for index in active
            if affinities.get(phase_ids[index], math.inf) < -affinity_tol_ln
        ]
        if undersaturated_active:
            drop = min(
                undersaturated_active,
                key=lambda index: (affinities[phase_ids[index]], phase_ids[index]),
            )
            n[drop] = 0.0
            active.remove(drop)
            if pending_add == drop:
                pending_add = None
            tie_breaks.append("drop_undersaturated:" + phase_ids[drop])
            continue

        active_bad = [
            index for index in active
            if abs(affinities.get(phase_ids[index], math.inf)) > affinity_tol_ln
        ]
        if active_bad:
            if repeated_state:
                solved_n, converged, reason, _solver_residuals = n.copy(), False, "repeated_state", {}
            else:
                solved_n, converged, reason, _solver_residuals = _solve_active_equalities(
                    n.copy(), active.copy(), A, b, states, parents, liquid_gibbs,
                    temperature_K, pack, affinity_tol_ln, balance_tol, inner_tol,
                    max_iter, warm_y,
                )
            if _solver_residuals:
                solver_error_history.append(dict(_solver_residuals))
            if converged:
                n = solved_n
                pending_add = None
                continue
            # Replacement order follows resolved formula moles; phase IDs break
            # ties, then two-phase removals sort by total moles and sorted IDs.
            trial_absent = [
                i for i, state in enumerate(states)
                if i not in active and affinities.get(state.candidate.phase_id, -math.inf) > affinity_tol_ln
            ]
            if (
                pending_add is not None
                and pending_add in active
                and affinities.get(phase_ids[pending_add], -math.inf) > affinity_tol_ln
            ):
                trial_absent.append(pending_add)
            if not trial_absent:
                return _empty_result(
                    bulk=bulk, temperature_K=temperature_K, pack=pack,
                    states=candidates_all, crystal_digest=crystal_digest,
                    pure_digest=pure_digest, explicit_candidates=explicit_candidates,
                    refusal=EquilibriumRefusal(
                        "active_set_cycle" if cycle_detected else
                        "singular_inconsistent_affinity"
                        if reason == "singular_inconsistent_affinity" else
                        "resolution_unresolved" if reason == "error_budget_unresolved" else
                        "active_set_incompatibility",
                        "Active crystal equalities could not be solved within the affinity tolerance.",
                        tuple(f"phase.{phase_ids[i]}" for i in active),
                        {"active_affinity_inf_ln": max(
                            (abs(affinities.get(phase_ids[i], math.inf)) for i in active),
                            default=math.inf,
                        ), **_solver_residuals},
                    ),
                    solver=history_solver(failure_reason=reason),
                )
            ordered_trial_absent = sorted(
                trial_absent,
                key=lambda i: (
                    i == pending_add,
                    -affinities[phase_ids[i]],
                    phase_ids[i],
                ),
            )
            replacement_sets: list[tuple[int, ...]] = []
            ordered_remove = sorted(
                active,
                key=lambda i: (n[i], phase_ids[i]),
            )
            replacement_sets.extend((i,) for i in ordered_remove)
            replacement_sets.extend(sorted(
                itertools.combinations(ordered_remove, 2),
                key=lambda pair: (
                    math.fsum(float(n[i]) for i in pair),
                    tuple(sorted(phase_ids[i] for i in pair)),
                ),
            ))
            replacement_found = False
            trial_error_budget: dict[str, float] = (
                dict(_solver_residuals)
                if reason == "error_budget_unresolved" else {}
            )
            replacement_trials = 0
            for most in ordered_trial_absent:
                for remove in replacement_sets:
                    trial_active = [i for i in active if i not in remove]
                    if most not in trial_active:
                        trial_active.append(most)
                    trial_n = n.copy()
                    for i in remove:
                        trial_n[i] = 0.0
                    trial_n[most] = 0.0
                    trial_key = tuple(sorted(phase_ids[i] for i in trial_active))
                    if trial_key in history_seen:
                        continue
                    replacement_trials += 1
                    trial_result = _solve_active_equalities(
                        trial_n, trial_active, A, b, states, parents, liquid_gibbs,
                        temperature_K, pack, affinity_tol_ln, balance_tol,
                        inner_tol, max_iter,
                    )
                    trial_n, trial_ok, _reason, _diag = trial_result
                    if not trial_ok:
                        if _reason == "error_budget_unresolved":
                            trial_error_budget.update(_diag)
                        continue
                    trial_liquid, _err = _safe_liquid(A, trial_n, b, balance_tol)
                    if trial_liquid is None:
                        continue
                    try:
                        trial_melt = _positive_melt(
                            trial_liquid, parents, temperature_K, pack, inner_tol, max_iter
                        )
                        trial_logs = _melt_logs(trial_melt, parents)
                    except (ImccRefusal, ArithmeticError):
                        continue
                    trial_aff = {}
                    for state in candidates_all:
                        candidate = state.candidate
                        if state.available and set(candidate.stoichiometry).issubset(by_parent):
                            trial_aff[candidate.phase_id] = _phase_affinity_from_melt(
                                candidate, by_parent, liquid_gibbs, trial_logs, temperature_K
                            )
                    all_equalities = all(
                        abs(trial_aff.get(phase_ids[i], math.inf)) <= affinity_tol_ln
                        for i in trial_active
                    )
                    trial_phase_names = {phase_ids[i] for i in trial_active}
                    all_inequalities = all(
                        trial_aff.get(state.candidate.phase_id, -math.inf) <= affinity_tol_ln
                        for state in candidates_all
                        if state.available and state.candidate.phase_id not in trial_phase_names
                        and set(state.candidate.stoichiometry).issubset(by_parent)
                    )
                    if all_equalities and all_inequalities:
                        n, active = trial_n, trial_active
                        pending_add = None
                        tie_breaks.append(
                            "replacement:" + ",".join(phase_ids[i] for i in remove)
                            + "->" + phase_ids[most]
                        )
                        replacement_found = True
                        break
                if replacement_found:
                    break
            if replacement_found:
                continue
            return _empty_result(
                bulk=bulk, temperature_K=temperature_K, pack=pack,
                states=candidates_all, crystal_digest=crystal_digest,
                pure_digest=pure_digest, explicit_candidates=explicit_candidates,
                refusal=EquilibriumRefusal(
                    "active_set_cycle" if cycle_detected else
                    "singular_inconsistent_affinity"
                    if reason == "singular_inconsistent_affinity" else
                    "resolution_unresolved" if trial_error_budget else
                    "active_set_incompatibility",
                    "No deterministic one- or two-phase replacement set passed all crystal inequalities.",
                    tuple(f"phase.{phase_ids[i]}" for i in active),
                    {"replacement_trials": float(replacement_trials), **trial_error_budget},
                ),
                solver=history_solver(failure_reason=reason),
            )

        absent = [
            i for i, state in enumerate(states)
            if i not in active
            and affinities.get(state.candidate.phase_id, -math.inf) > affinity_tol_ln
        ]
        if absent:
            maximum = max(affinities[phase_ids[i]] for i in absent)
            tied = [
                i for i in absent
                if maximum - affinities[phase_ids[i]] <= affinity_tol_ln
            ]
            chosen = min(tied, key=lambda i: phase_ids[i])
            if len(tied) > 1:
                tie_breaks.append(
                    "supersaturation_tie:" + ",".join(sorted(phase_ids[i] for i in tied))
                    + "->" + phase_ids[chosen]
                )
            active.append(chosen)
            active.sort(key=lambda i: phase_ids[i])
            pending_add = chosen
            continue

        # Recheck from fresh data, independently reconstructing every active
        # equality, absent inequality, bulk balance and kernel inner certificate.
        passed, recheck, checked_melt, checked_affinities = _independent_kkt_recheck(
            bulk_vector=b, liquid_vector=current_liquid, solid_vector=n,
            solid_states=states, A=A,
            states=candidates_all, parents=parents, temperature_K=temperature_K,
            pack=pack, liquid_gibbs=liquid_gibbs, inner_tol=inner_tol,
            balance_tol=balance_tol, affinity_tol_ln=affinity_tol_ln,
            absent_tol_ln=affinity_tol_ln,
            gas_pack=gas_pack,
            y_init=warm_y,
        )
        if not passed:
            return _empty_result(
                bulk=bulk, temperature_K=temperature_K, pack=pack,
                states=candidates_all, crystal_digest=crystal_digest,
                pure_digest=pure_digest, explicit_candidates=explicit_candidates,
                refusal=EquilibriumRefusal(
                    "kkt_residual",
                    "Independent KKT recheck exceeded a named numerical tolerance.",
                    residuals=recheck,
                ),
                solver=history_solver(**recheck),
            )
        recheck["independent_kkt_passed"] = 1.0
        residual_inventory = b - current_liquid
        saturated_indices = [
            index for index, state in enumerate(states)
            if abs(checked_affinities.get(state.candidate.phase_id, math.inf))
            <= affinity_tol_ln
        ]
        ranges, unique, ranges_certified, liquid_scale_range = _phase_extent_ranges(
            A, n, saturated_indices, residual_inventory, scales, balance_tol,
            bulk_vector=b, liquid_vector=current_liquid,
        )
        state_amounts: dict[str, float] = {}
        state_ranges: dict[str, tuple[float, float]] = {}
        presence: dict[str, Literal["present", "absent", "unresolved", "infeasible"]] = {}
        for index, state in enumerate(states):
            phase = state.candidate.phase_id
            lo, hi = ranges.get(index, (0.0, 0.0))
            state_ranges[phase] = (lo, hi)
            threshold = balance_tol * scales[index]
            presence[phase] = (
                "present" if lo > threshold else
                "absent" if hi <= threshold else "unresolved"
            )
            state_amounts[phase] = float(n[index])
        missing_set = set(missing)
        for state in candidates_all:
            phase = state.candidate.phase_id
            if state.reason in ("absent_by_construction", "above_source_melting_point"):
                state_ranges[phase] = (0.0, 0.0)
                state_amounts[phase] = 0.0
                presence[phase] = "absent"
                continue
            if state.reason == "infeasible":
                state_ranges.setdefault(phase, (0.0, 0.0))
                state_amounts.setdefault(phase, 0.0)
                presence.setdefault(phase, "infeasible")
                continue
            unknown = not state.available or phase in missing_set
            state_ranges.setdefault(
                phase, (0.0, math.inf) if unknown else (0.0, 0.0)
            )
            state_amounts.setdefault(phase, 0.0)
            presence.setdefault(
                phase,
                "unresolved" if unknown else "absent",
            )
        for phase, representative in rejected_polymorphs.items():
            state_ranges[phase] = (0.0, 0.0)
            state_amounts[phase] = 0.0
            presence[phase] = "absent"
        for representative, members in aliases.items():
            if representative in phase_index:
                total_range = state_ranges.get(representative, (0.0, 0.0))
                total_amount = n[phase_index[representative]]
                for member in members:
                    state_ranges[member] = (0.0, total_range[1])
                    state_amounts[member] = total_amount if member == representative else 0.0
                    presence[member] = (
                        "unresolved"
                        if total_range[1] > balance_tol * scales[phase_index[representative]]
                        else "absent"
                    )
                unique = False
        positive = {name: float(value) for name, value in zip(parents, current_liquid, strict=True)}
        phase_affinities = {
            phase: float(value) for phase, value in checked_affinities.items()
        }
        phase_ranges = {
            phase: (value, value) for phase, value in phase_affinities.items()
        }
        positive_log_activity = np.log(np.asarray([
            checked_melt.activity(name) for name in parents
        ], dtype=float))
        positive_logq_error = _logq_roundoff_bound(
            positive_log_activity,
            temperature_K,
            pack,
            [tuple(_kernel_parent_names(pack)).index(name) for name in parents],
        )
        saturated = tuple(sorted(
            phase for phase, value in phase_affinities.items()
            if abs(value) <= affinity_tol_ln
        ))
        for rejected, winner in rejected_polymorphs.items():
            # Its computed affinity is already available and its higher G
            # makes it less supersaturated than the stable polymorph.
            if abs(phase_affinities.get(rejected, math.inf)) <= affinity_tol_ln:
                saturated = tuple(sorted((*saturated, rejected)))
        authorities_missing = bool(missing) or explicit_candidates
        refusal = (
            EquilibriumRefusal(
                "resolution_unresolved",
                "A phase amount range solve did not resolve within numerical precision.",
                ("phase_amount_ranges",),
                {"phase_amount_ranges_certified": 0.0},
            )
            if not ranges_certified else
            EquilibriumRefusal(
                "missing_phase_data",
                "The candidate set is restricted or has crystal functions outside their source range.",
                tuple(f"crystal.{phase}" for phase in missing),
            )
            if authorities_missing else None
        )
        validity: Literal["IN", "OUT_OF_DOMAIN", "DEPENDS"] = (
            "DEPENDS" if authorities_missing or not ranges_certified else "IN"
        )
        labels = _labels(
            pack, candidates_all, missing, explicit_candidates,
            ("restricted_candidate_set",) if explicit_candidates else (),
            tie_breaks,
            validity,
            (
                tuple(f"crystal.{phase}.temperature_K" for phase in missing)
                if ranges_certified else ("phase_amount_ranges",)
            ),
            parents,
        )
        receipt = _result_receipt(
            bulk=bulk, temperature_K=temperature_K, pack=pack,
            crystal_digest=crystal_digest, pure_digest=pure_digest,
            candidates=candidates_all,
            solver={
                "active_set_history": history,
                "active_set_residual_signs": residual_sign_history,
                "outer_error_budget_history": solver_error_history,
                "missing_candidate_phase_ids": tuple(missing),
                "polymorph_tie_breaks": polymorph_tie_breaks,
                "deterministic_tie_breaks": tie_breaks,
                "outer_iterations": len(history),
                "phase_amount_ranges_certified": ranges_certified,
                "liquid_parent_scale_range_mol": liquid_scale_range,
                "kkt_recheck": recheck,
            },
            gas_pack=gas_pack,
        )
        return PhaseEquilibriumResult(
            temperature_K=temperature_K,
            pressure_bar=_PRESSURE_BAR,
            bulk_parent_mol=bulk,
            liquid_parent_mol=(
                positive
                if liquid_scale_range[1] - liquid_scale_range[0]
                <= balance_tol * float(np.sum(b))
                + _roundoff_bound(liquid_scale_range, 8)
                else None
            ),
            liquid_mass_fraction=(
                _mass_fraction(positive, bulk)
                if liquid_scale_range[1] - liquid_scale_range[0]
                <= balance_tol * float(np.sum(b))
                + _roundoff_bound(liquid_scale_range, 8)
                else None
            ),
            solid_mol=state_amounts if unique else None,
            phase_amount_ranges_mol=state_ranges,
            phase_amounts_unique=unique,
            phase_presence=presence,
            saturated_candidates=saturated,
            identifiable_activities={
                name: checked_melt.activity(name) for name in parents
            },
            activity_ranges_ln={
                name: (math.log(checked_melt.activity(name)), math.log(checked_melt.activity(name)))
                for name in parents if checked_melt.activity(name) > 0.0
            },
            phase_affinities_ln=phase_affinities,
            phase_affinity_ranges_ln=phase_ranges,
            phase_saturation_indices=_make_saturation_rows(
                candidates_all, phase_affinities, temperature_K
            ),
            logQ=recheck["inner_logQ"],
            logQ_interval=(
                recheck["inner_logQ"] - positive_logq_error,
                recheck["inner_logQ"] + positive_logq_error,
            ),
            melt=checked_melt,
            gas=None,
            mass_balance_relative_inf=recheck["balance_relative_inf"],
            labels=labels,
            receipt=receipt,
            refusal=refusal,
            active_set_history=tuple(history),
            kkt_recheck=recheck,
        )

    return _empty_result(
        bulk=bulk, temperature_K=temperature_K, pack=pack,
        states=candidates_all, crystal_digest=crystal_digest,
        pure_digest=pure_digest, explicit_candidates=explicit_candidates,
        refusal=EquilibriumRefusal(
            "active_set_cycle" if cycle_detected else "active_set_incompatibility",
            "The deterministic active-set transition allowance was exhausted.",
            tuple(f"phase.{phase}" for phase in history[-1]) if history else (),
            {"active_set_restarts": float(_MAX_ACTIVE_RESTARTS)},
        ),
        solver=history_solver(),
    )


def _solid_result(
    *,
    bulk: Mapping[str, float],
    parents: Sequence[str],
    states: Sequence[_CandidateState],
    all_states: Sequence[_CandidateState],
    A: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    lp: _LPState,
    mu_q: np.ndarray | None,
    logq: float | None,
    logq_interval: tuple[float, float] | None,
    q_certified: bool,
    liquid_gibbs: np.ndarray,
    temperature_K: float,
    pack: Any,
    gas_pack: Any,
    crystal_digest: str,
    pure_digest: str,
    explicit_candidates: bool,
    missing: Sequence[str],
    aliases: Mapping[str, tuple[str, ...]],
    rejected: Mapping[str, str],
    balance_tol: float,
    affinity_tol_ln: float,
    inner_tol: float,
    out_of_domain: bool = False,
    optimizer_residuals: Mapping[str, float] | None = None,
) -> PhaseEquilibriumResult:
    active_parent_index = {name: i for i, name in enumerate(parents)}
    if mu_q is None:
        mu_q = lp.chemical_potentials
    q_interval = logq_interval
    q_value = logq
    if not out_of_domain:
        q_value, _grad = _logq_from_potentials(
            mu_q, liquid_gibbs, temperature_K, pack,
            [tuple(_kernel_parent_names(pack)).index(name) for name in parents],
        )
        if q_interval is None:
            log_activity = (mu_q - liquid_gibbs) / (R_J_MOL_K * temperature_K)
            indices = [tuple(_kernel_parent_names(pack)).index(name) for name in parents]
            err = _logq_roundoff_bound(
                log_activity,
                temperature_K,
                pack,
                indices,
                [*map(float, mu_q / (R_J_MOL_K * temperature_K)),
                 *map(float, liquid_gibbs / (R_J_MOL_K * temperature_K))],
            )
            q_interval = (q_value - err, q_value + err)

    solid_ranges: dict[str, tuple[float, float]] = {}
    phase_presence: dict[str, Literal["present", "absent", "unresolved", "infeasible"]] = {}
    amounts: dict[str, float] = {}
    scales = _scaled_phase_amounts(A, b)
    amount_unique = lp.amount_ranges_certified
    dual_ranges_J, dual_ranges_certified = _dual_potential_ranges(
        A, g, b, lp.objective_J, temperature_K
    )
    phase_to_index = {
        state.candidate.phase_id: index for index, state in enumerate(states)
    }
    for index, state in enumerate(states):
        lo, hi = lp.amount_ranges[index]
        phase = state.candidate.phase_id
        threshold = balance_tol * scales[index]
        solid_ranges[phase] = (lo, hi)
        amounts[phase] = float(lp.amounts[index])
        phase_presence[phase] = (
            "present" if lo > threshold else
            "absent" if hi <= threshold else
            "unresolved"
        )
        amount_unique = amount_unique and hi - lo <= threshold + _roundoff_bound((lo, hi), 8)
    for phase in rejected:
        amounts[phase] = 0.0
        solid_ranges[phase] = (0.0, 0.0)
        phase_presence[phase] = "absent"
    missing_set = set(missing)
    proven_absent = {
        state.candidate.phase_id for state in all_states
        if state.reason in ("absent_by_construction", "above_source_melting_point")
    }
    for state in all_states:
        phase = state.candidate.phase_id
        if state.reason in ("absent_by_construction", "above_source_melting_point"):
            solid_ranges[phase] = (0.0, 0.0)
            amounts[phase] = 0.0
            phase_presence[phase] = "absent"
            continue
        if state.reason == "infeasible":
            solid_ranges[phase] = (0.0, 0.0)
            amounts[phase] = 0.0
            phase_presence[phase] = "infeasible"
            continue
        unknown = not state.available or phase in missing_set
        solid_ranges.setdefault(
            phase, (0.0, math.inf) if unknown else (0.0, 0.0)
        )
        amounts.setdefault(phase, 0.0)
        phase_presence.setdefault(
            phase, "unresolved" if unknown else "absent"
        )
    for representative, members in aliases.items():
        index = phase_to_index[representative]
        lo, hi = lp.amount_ranges[index]
        total = float(lp.amounts[index])
        for phase in members:
            solid_ranges[phase] = (0.0, hi)
            amounts[phase] = total if phase == representative else 0.0
            phase_presence[phase] = (
                "unresolved" if hi > balance_tol * scales[index] else "absent"
            )
        amount_unique = False

    activity_ranges: dict[str, tuple[float, float]] = {}
    activities: dict[str, float] = {}
    rt = R_J_MOL_K * temperature_K
    for parent_index, parent in enumerate(parents):
        low_mu, high_mu = dual_ranges_J[parent_index]
        low_log = (low_mu - liquid_gibbs[parent_index]) / rt
        high_log = (high_mu - liquid_gibbs[parent_index]) / rt
        activity_ranges[parent] = (float(low_log), float(high_log))
        if (
            dual_ranges_certified
            and math.isfinite(low_log)
            and math.isfinite(high_log)
            and high_log - low_log <= _INNER_ACTIVITY_ERROR_TOL_LN
        ):
            value = low_log + (high_log - low_log) / 2.0
            try:
                activity = math.exp(value)
            except OverflowError:
                continue
            if math.isfinite(activity) and activity > 0.0:
                activities[parent] = activity

    affinities: dict[str, float] = {}
    affinity_ranges: dict[str, tuple[float, float]] = {}
    saturated: list[str] = []
    for state in all_states:
        c = state.candidate
        if not state.available or set(c.stoichiometry) - set(active_parent_index):
            continue
        coefficients = np.asarray(
            [c.stoichiometry.get(name, 0.0) for name in parents], dtype=float
        )
        point = (float(coefficients @ mu_q) - c.G_crystal_J_mol) / (
            R_J_MOL_K * temperature_K
        )
        affinities[c.phase_id] = point
        if len(activities) == len(parents):
            affinity_ranges[c.phase_id] = (point, point)
        if abs(point) <= affinity_tol_ln:
            saturated.append(c.phase_id)

    original_dual_slack = g - A.T @ lp.chemical_potentials
    gap = abs(float(g @ lp.amounts - b @ lp.chemical_potentials))
    gap_bound = _roundoff_bound(
        [*map(float, g * lp.amounts), *map(float, b * lp.chemical_potentials)],
        8 * (len(g) + len(b)),
    )
    passed, recheck, _melt, checked_affinities = _independent_kkt_recheck(
        bulk_vector=b,
        liquid_vector=None,
        solid_vector=lp.amounts,
        solid_states=states,
        A=A,
        states=all_states,
        parents=parents,
        temperature_K=temperature_K,
        pack=pack,
        liquid_gibbs=liquid_gibbs,
        inner_tol=inner_tol,
        balance_tol=balance_tol,
        affinity_tol_ln=affinity_tol_ln,
        absent_tol_ln=affinity_tol_ln,
        dual_mu=None if out_of_domain else mu_q,
        gas_pack=gas_pack,
    )
    recheck.update({
        "solid_lp_primal_dual_gap_J": gap,
        "solid_lp_primal_dual_gap_bound_J": gap_bound,
        "solid_lp_min_dual_slack_J_mol": float(np.min(lp.dual_slacks_J_mol, initial=0.0)),
        "solid_amount_ranges_certified": float(lp.amount_ranges_certified),
        "logQ_upper": (q_interval[1] if q_interval else math.nan),
    })
    if optimizer_residuals:
        recheck.update(optimizer_residuals)
    dual_feasible = bool(np.all(
        original_dual_slack >= -_DUAL_SLACK_TOL_RT * R_J_MOL_K * temperature_K
    ))
    equilibrium_state_certified = (
        not out_of_domain and q_certified and passed and gap <= gap_bound
        and dual_feasible and lp.amount_ranges_certified and dual_ranges_certified
    )
    if equilibrium_state_certified:
        recheck["independent_kkt_passed"] = 1.0
    if out_of_domain:
        refusal = EquilibriumRefusal(
            "liquid_absence_not_certified",
            "Only the crystal-only LP minimum is returned outside the liquid-model temperature domain.",
            ("temperature_K",),
            recheck,
        )
        validity: Literal["IN", "OUT_OF_DOMAIN", "DEPENDS"] = "OUT_OF_DOMAIN"
        axes = ("temperature_K",)
    elif (
        not q_certified
        and bool((optimizer_residuals or {}).get("logQ_minimum_certified", 0.0))
        and q_interval is not None
        and q_interval[0] <= 0.0 < q_interval[1]
    ):
        refusal = EquilibriumRefusal(
            "incipient_liquid_unresolved",
            "The certified logQ interval straddles zero on the dual optimal face.",
            ("liquid_absence",),
            recheck,
        )
        validity = "DEPENDS"
        axes = ()
    elif not q_certified:
        refusal = EquilibriumRefusal(
            "liquid_absence_not_certified",
            "The dual optimal-face liquid-absence certificate did not resolve logQ≤0.",
            ("liquid_absence",),
            recheck,
        )
        validity = "DEPENDS"
        axes = ()
    elif (
        not passed
        or gap > gap_bound
        or not dual_feasible
    ):
        refusal = EquilibriumRefusal(
            "kkt_residual",
            "The independent crystal-only KKT/LP recheck exceeded its numerical resolution.",
            residuals=recheck,
        )
        validity = "DEPENDS"
        axes = ()
    elif not lp.amount_ranges_certified:
        refusal = EquilibriumRefusal(
            "resolution_unresolved",
            "A crystal phase amount range solve did not resolve within numerical precision.",
            ("phase_amount_ranges",),
            recheck,
        )
        validity = "DEPENDS"
        axes = ("phase_amount_ranges",)
    elif not dual_ranges_certified:
        refusal = EquilibriumRefusal(
            "resolution_unresolved",
            "A dual optimal-face potential range solve did not resolve within numerical precision.",
            ("activity_ranges_ln",),
            recheck,
        )
        validity = "DEPENDS"
        axes = ("activity_ranges_ln",)
    elif missing or explicit_candidates:
        refusal = EquilibriumRefusal(
            "missing_phase_data",
            "The candidate set is restricted or contains crystal functions outside their source range.",
            tuple(f"crystal.{phase}" for phase in missing),
        )
        validity = "DEPENDS"
        axes = tuple(f"crystal.{phase}.temperature_K" for phase in missing)
    else:
        refusal = None
        validity = "IN"
        axes = ()

    labels = _labels(
        pack,
        all_states,
        missing,
        explicit_candidates,
        (("crystal_only_minimum",) if not equilibrium_state_certified else ()),
        (),
        validity,
        axes,
        parents,
    )
    receipt = _result_receipt(
        bulk=bulk,
        temperature_K=temperature_K,
        pack=pack,
        crystal_digest=crystal_digest,
        pure_digest=pure_digest,
        candidates=all_states,
        solver={
            "solid_lp_objective_J": lp.objective_J,
            "crystal_only_candidate_amounts_mol": dict(zip(
                (state.candidate.phase_id for state in states),
                map(float, lp.amounts), strict=True,
            )),
            "crystal_only_candidate_ranges_mol": dict(zip(
                (state.candidate.phase_id for state in states),
                lp.amount_ranges, strict=True,
            )),
            "missing_candidate_phase_ids": tuple(missing),
            "liquid_absence_certified": bool(q_certified),
            "dual_potentials_J_mol": dict(zip(parents, map(float, mu_q), strict=True)),
            "dual_optimal_face_logQ": q_value,
            "logQ_interval": q_interval,
            "polymorph_tie_breaks": tuple(
                f"{','.join(members)}->{representative}"
                for representative, members in sorted(aliases.items())
            ) + tuple(
                f"{phase}->{winner}"
                for phase, winner in sorted(rejected.items())
            ),
            "solid_lp_primal_dual_gap_J": gap,
            "solid_lp_primal_dual_gap_bound_J": gap_bound,
            "phase_amount_ranges_certified": lp.amount_ranges_certified,
            "dual_potential_ranges_J_mol": dict(zip(
                parents, dual_ranges_J, strict=True,
            )),
            "dual_potential_ranges_certified": dual_ranges_certified,
            "kkt_recheck": recheck,
        },
        gas_pack=gas_pack,
    )
    return PhaseEquilibriumResult(
        temperature_K=temperature_K,
        pressure_bar=_PRESSURE_BAR,
        bulk_parent_mol=bulk,
        liquid_parent_mol=None,
        liquid_mass_fraction=0.0 if equilibrium_state_certified else None,
        solid_mol=amounts if amount_unique and equilibrium_state_certified else None,
        phase_amount_ranges_mol=(
            solid_ranges if equilibrium_state_certified else
            {
                phase: (0.0, 0.0) if phase in proven_absent else (0.0, math.inf)
                for phase in phase_presence
            }
        ),
        phase_amounts_unique=bool(amount_unique and equilibrium_state_certified),
        phase_presence=(
            phase_presence if equilibrium_state_certified else
            {
                phase: (
                    "infeasible" if presence == "infeasible" else
                    "absent" if phase in proven_absent else
                    "unresolved"
                )
                for phase, presence in phase_presence.items()
            }
        ),
        saturated_candidates=(tuple(sorted(saturated)) if equilibrium_state_certified else ()),
        identifiable_activities=activities if equilibrium_state_certified else {},
        activity_ranges_ln=activity_ranges if equilibrium_state_certified else {},
        phase_affinities_ln=affinities if equilibrium_state_certified else {},
        phase_affinity_ranges_ln=affinity_ranges if equilibrium_state_certified else {},
        phase_saturation_indices=_make_saturation_rows(
            all_states, affinities if equilibrium_state_certified else {}, temperature_K
        ),
        logQ=q_value,
        logQ_interval=q_interval,
        melt=None,
        gas=None,
        mass_balance_relative_inf=(
            recheck.get("balance_relative_inf") if equilibrium_state_certified else None
        ),
        labels=labels,
        receipt=receipt,
        refusal=refusal,
        kkt_recheck=recheck,
        crystal_only_minimum=not equilibrium_state_certified,
    )


def _equilibrate_one(
    bulk: Mapping[str, float] | Sequence[float],
    T_K: float,
    *,
    candidates: Sequence[str | CrystalCandidate] | None = None,
    pack: Any | None = None,
    tol: float = _DEFAULT_BALANCE_TOL,
    affinity_tol_ln: float = _DEFAULT_AFFINITY_TOL_LN,
    inner_tol: float = _INNER_BALANCE_TOL,
    max_iter: int = 100,
    warm_start: PhaseEquilibriumResult | None = None,
) -> PhaseEquilibriumResult:
    """Equilibrate oxide-parent moles with supported stoichiometric crystals.

    The returned state is certified only for the ideal associate melt and the
    supplied candidate crystal set. Pressure is fixed at 1 bar. Liquid claims
    are in-domain only from 1700 to 3000 K.
    """
    selected_pack = _loaded_pack(pack)
    kernel = selected_pack.kernel_datapack if isinstance(selected_pack, ImccLoadedDatapack) else selected_pack
    temperature = float(T_K)
    candidate_temperature = temperature if math.isfinite(temperature) and temperature > 0.0 else 2000.0
    from openimcc import diagnostics as diagnostic_module
    gas_pack, _ = diagnostic_module._crystal_thermo_context(None)
    try:
        all_states, crystal_digest, pure_digest, gas_pack, unavailable, explicit = (
            _resolve_candidates(candidates, candidate_temperature, gas_pack)
        )
    except Exception as exc:
        # Unknown crystal IDs and malformed candidate input are typed as
        # missing phase data at this API boundary.
        fallback_states, crystal_digest, pure_digest, gas_pack, _unavailable, _explicit = (
            _resolve_candidates(None, candidate_temperature, gas_pack)
        )
        empty_bulk: dict[str, float] = {}
        return _empty_result(
            bulk=empty_bulk,
            temperature_K=temperature,
            pack=selected_pack,
            states=fallback_states,
            crystal_digest=crystal_digest,
            pure_digest=pure_digest,
            explicit_candidates=candidates is not None,
            refusal=EquilibriumRefusal(
                "missing_phase_data", str(exc), ("crystal_candidates",)
            ),
        )

    parent_names = tuple(str(name) for name in kernel.parent_oxides)
    if not math.isfinite(temperature) or temperature <= 0.0:
        return _empty_result(
            bulk={},
            temperature_K=temperature,
            pack=selected_pack,
            states=all_states,
            crystal_digest=crystal_digest,
            pure_digest=pure_digest,
            explicit_candidates=explicit,
            refusal=EquilibriumRefusal(
                "nonfinite_input" if not math.isfinite(temperature) else "invalid_input",
                "Temperature must be finite and positive.",
                ("temperature_K",),
            ),
        )
    try:
        normalized_bulk, parents = _parse_bulk(bulk, parent_names)
    except (TypeError, ValueError, KeyError, OverflowError) as exc:
        supplied = bulk.values() if isinstance(bulk, Mapping) else bulk
        try:
            contains_nonfinite = any(
                not math.isfinite(float(value)) for value in supplied
            )
        except (TypeError, ValueError, OverflowError):
            contains_nonfinite = False
        code = (
            "nonfinite_input" if contains_nonfinite else
            "unsupported_component" if isinstance(exc, KeyError) else
            "invalid_input"
        )
        return _empty_result(
            bulk={},
            temperature_K=temperature,
            pack=selected_pack,
            states=all_states,
            crystal_digest=crystal_digest,
            pure_digest=pure_digest,
            explicit_candidates=explicit,
            refusal=EquilibriumRefusal(
                code,
                str(exc),
                ("bulk_parent_mol",),
            ),
        )
    all_states, unavailable = _exclude_zero_bulk_infeasible_candidates(
        all_states, unavailable, normalized_bulk, parent_names,
    )
    bulk_vector = np.asarray([normalized_bulk[name] for name in parents], dtype=float)
    if (
        not math.isfinite(float(tol)) or float(tol) <= 0.0
        or not math.isfinite(float(affinity_tol_ln)) or float(affinity_tol_ln) <= 0.0
        or not math.isfinite(float(inner_tol)) or float(inner_tol) <= 0.0
        or max_iter < 1
    ):
        return _empty_result(
            bulk=normalized_bulk,
            temperature_K=temperature,
            pack=selected_pack,
            states=all_states,
            crystal_digest=crystal_digest,
            pure_digest=pure_digest,
            explicit_candidates=explicit,
            refusal=EquilibriumRefusal(
                "invalid_input", "Solver tolerances and max_iter must be positive and finite.",
                ("solver_settings",),
            ),
        )
    if _has_active_excess(selected_pack):
        return _empty_result(
            bulk=normalized_bulk,
            temperature_K=temperature,
            pack=selected_pack,
            states=all_states,
            crystal_digest=crystal_digest,
            pure_digest=pure_digest,
            explicit_candidates=explicit,
            refusal=EquilibriumRefusal(
                "unsupported_nonconvex_model",
                "Active excess-Gibbs terms are outside the ideal-pack convexity proof.",
                ("melt_model",),
            ),
        )
    redox = tuple(
        state.candidate.phase_id for state in all_states
        if _redox_candidate(state.candidate)
    )
    if redox:
        return _empty_result(
            bulk=normalized_bulk,
            temperature_K=temperature,
            pack=selected_pack,
            states=all_states,
            crystal_digest=crystal_digest,
            pure_digest=pure_digest,
            explicit_candidates=explicit,
            refusal=EquilibriumRefusal(
                "unsupported_redox_phase",
                "Ferric/redox-sensitive crystal phases are not supported.",
                tuple(f"crystal.{phase}.redox" for phase in redox),
            ),
        )
    try:
        states, aliases, rejected_polymorphs = _polymorph_filter(
            all_states, temperature, float(affinity_tol_ln)
        )
        states, A, g, unsupported = _candidate_matrix(
            states, parents, parent_names, normalized_bulk
        )
    except ValueError as exc:
        return _empty_result(
            bulk=normalized_bulk,
            temperature_K=temperature,
            pack=selected_pack,
            states=all_states,
            crystal_digest=crystal_digest,
            pure_digest=pure_digest,
            explicit_candidates=explicit,
            refusal=EquilibriumRefusal(
                "unsupported_redox_phase", str(exc), ("crystal_candidates",)
            ),
        )
    missing_set = set(unavailable) | set(unsupported)
    if not explicit:
        present_components = set(normalized_bulk)
        missing_set.update(
            phase for phase, components in _MISSING_PHASE_COMPONENTS.items()
            if components <= present_components
        )
    missing = tuple(sorted(missing_set))

    from openimcc import gas as gas_module
    try:
        liquid_gibbs = _melt_liquid_standards(parents, temperature, gas_pack)
    except (
        gas_module.ImccGasTemperatureOutsideDomainError,
        gas_module.ImccGasSpeciesNotFoundError,
        KeyError,
        ValueError,
    ) as exc:
        return _empty_result(
            bulk=normalized_bulk,
            temperature_K=temperature,
            pack=selected_pack,
            states=all_states,
            crystal_digest=crystal_digest,
            pure_digest=pure_digest,
            explicit_candidates=explicit,
            refusal=EquilibriumRefusal(
                "missing_phase_data", f"Parent-liquid Gibbs standard unavailable: {exc}",
                ("parent_liquid.temperature_K",),
            ),
            validity="OUT_OF_DOMAIN",
            axes=("temperature_K",),
        )
    b = np.asarray([normalized_bulk[name] for name in parents], dtype=float)
    lp = _lp_solid_minimum(A, g, b, float(tol))
    out_of_domain = not (_LIQUID_DOMAIN_K[0] <= temperature <= _LIQUID_DOMAIN_K[1])
    lp_diagnostics: dict[str, float] = {}
    lp_valid = False
    if lp is not None:
        primal = A @ lp.amounts - b
        relative = float(np.max(np.abs(primal) / b, initial=0.0))
        gap = abs(float(g @ lp.amounts - b @ lp.chemical_potentials))
        gap_bound = _roundoff_bound(
            [*map(float, g * lp.amounts), *map(float, b * lp.chemical_potentials)],
            8 * (len(g) + len(b)),
        )
        all_slacks = []
        for state in all_states:
            candidate = state.candidate
            if not state.available or set(candidate.stoichiometry) - set(parents):
                continue
            column = np.asarray(
                [candidate.stoichiometry.get(name, 0.0) for name in parents],
                dtype=float,
            )
            all_slacks.append(candidate.G_crystal_J_mol - float(column @ lp.chemical_potentials))
        minimum_slack = float(min(all_slacks, default=math.inf))
        lp_valid = (
            relative <= float(tol)
            and gap <= gap_bound
            and minimum_slack >= -_DUAL_SLACK_TOL_RT * R_J_MOL_K * temperature
        )
        lp_diagnostics = {
            "solid_lp_primal_balance_relative_inf": relative,
            "solid_lp_primal_dual_gap_J": gap,
            "solid_lp_primal_dual_gap_bound_J": gap_bound,
            "solid_lp_min_all_candidate_dual_slack_J_mol": minimum_slack,
        }

    if out_of_domain:
        if lp is None or not lp_valid:
            return _empty_result(
                bulk=normalized_bulk,
                temperature_K=temperature,
                pack=selected_pack,
                states=all_states,
                crystal_digest=crystal_digest,
                pure_digest=pure_digest,
                explicit_candidates=explicit,
                refusal=EquilibriumRefusal(
                    "infeasible_bulk" if lp is None else "kkt_residual",
                    "A crystal-only LP minimum could not be independently verified.",
                    ("crystal_balance",),
                    lp_diagnostics,
                ),
                validity="OUT_OF_DOMAIN",
                axes=("temperature_K",),
            )
        return _solid_result(
            bulk=normalized_bulk,
            parents=parents,
            states=states,
            all_states=all_states,
            A=A,
            g=g,
            b=b,
            lp=lp,
            mu_q=None,
            logq=None,
            logq_interval=None,
            q_certified=False,
            liquid_gibbs=liquid_gibbs,
            temperature_K=temperature,
            pack=selected_pack,
            gas_pack=gas_pack,
            crystal_digest=crystal_digest,
            pure_digest=pure_digest,
            explicit_candidates=explicit,
            missing=missing,
            aliases=aliases,
            rejected=rejected_polymorphs,
            balance_tol=float(tol),
            affinity_tol_ln=float(affinity_tol_ln),
            inner_tol=float(inner_tol),
            out_of_domain=True,
            optimizer_residuals=lp_diagnostics,
        )

    solid_diagnostic: PhaseEquilibriumResult | None = None
    if lp is not None and lp_valid:
        try:
            mu_q, logq, logq_interval, q_certified, q_diagnostics = (
                _minimize_logq_on_dual_face(
                    A, g, b, lp.objective_J, lp.chemical_potentials,
                    liquid_gibbs, temperature, selected_pack,
                    [tuple(kernel.parent_oxides).index(name) for name in parents],
                )
            )
        except (ValueError, ArithmeticError, FloatingPointError) as exc:
            return _empty_result(
                bulk=normalized_bulk,
                temperature_K=temperature,
                pack=selected_pack,
                states=all_states,
                crystal_digest=crystal_digest,
                pure_digest=pure_digest,
                explicit_candidates=explicit,
                refusal=EquilibriumRefusal(
                    "liquid_absence_not_certified",
                    f"The dual optimal-face logQ solve failed: {exc}",
                    ("liquid_absence",),
                    lp_diagnostics,
                ),
                solver=lp_diagnostics,
            )
        minimum_certified = bool(q_diagnostics.get("logQ_minimum_certified", 0.0))
        interval_straddles = (
            logq_interval is not None
            and logq_interval[0] <= 0.0 < logq_interval[1]
        )
        if q_certified or (minimum_certified and interval_straddles) or not minimum_certified:
            solid_diagnostic = _solid_result(
                bulk=normalized_bulk,
                parents=parents,
                states=states,
                all_states=all_states,
                A=A,
                g=g,
                b=b,
                lp=lp,
                mu_q=mu_q,
                logq=logq,
                logq_interval=logq_interval,
                q_certified=q_certified,
                liquid_gibbs=liquid_gibbs,
                temperature_K=temperature,
                pack=selected_pack,
                gas_pack=gas_pack,
                crystal_digest=crystal_digest,
                pure_digest=pure_digest,
                explicit_candidates=explicit,
                missing=missing,
                aliases=aliases,
                rejected=rejected_polymorphs,
                balance_tol=float(tol),
                affinity_tol_ln=float(affinity_tol_ln),
                inner_tol=float(inner_tol),
                optimizer_residuals={**lp_diagnostics, **dict(q_diagnostics)},
            )
            if q_certified:
                return solid_diagnostic

    result = _positive_branch(
        bulk=normalized_bulk,
        parents=parents,
        states=states,
        A=A,
        g=g,
        liquid_gibbs=liquid_gibbs,
        temperature_K=temperature,
        pack=selected_pack,
        gas_pack=gas_pack,
        candidates_all=all_states,
        aliases=aliases,
        rejected_polymorphs=rejected_polymorphs,
        crystal_digest=crystal_digest,
        pure_digest=pure_digest,
        explicit_candidates=explicit,
        missing=missing,
        balance_tol=float(tol),
        affinity_tol_ln=float(affinity_tol_ln),
        inner_tol=float(inner_tol),
        max_iter=int(max_iter),
        warm_start=warm_start,
    )
    if result.kkt_recheck.get("independent_kkt_passed") == 1.0:
        return result
    if solid_diagnostic is not None:
        return solid_diagnostic
    return result


def _attach_phase_state_vapour(
    result: PhaseEquilibriumResult,
    *,
    candidates: Sequence[str | CrystalCandidate] | None,
    fO2: float | None,
) -> PhaseEquilibriumResult:
    candidate_ids = (
        None if candidates is None else tuple(
            candidate.phase_id if isinstance(candidate, CrystalCandidate)
            else str(candidate)
            for candidate in candidates
        )
    )
    request = dict(result.receipt.request)
    request.update({"candidates": candidate_ids, "fO2": fO2})
    gas_result = None
    gas_depends: dict[str, str] = {}
    vapour_error = None
    gas_pack = None
    refusal = result.refusal
    solver = dict(result.receipt.solver)
    data_digests = dict(result.receipt.data_digests)

    if fO2 is not None:
        from openimcc import gas as gas_module

        gas_pack, _reference_pack = diagnostics._crystal_thermo_context(None)
        request["fO2"] = float(fO2)
        try:
            gas_result, gas_depends = _evaluate_gas_attachment(
                result.identifiable_activities,
                result.temperature_K,
                float(fO2),
                gas_pack,
                dependency_components=tuple(result.bulk_parent_mol),
                parent_oxides=tuple(result.identifiable_activities),
            )
        except gas_module.ImccGasTemperatureOutsideDomainError as exc:
            vapour_error = str(exc)

        data_digests.update({
            "gas": gas_pack.gas_table_digest,
            "condensate": gas_pack.condensate_table_digest,
        })
        solver["vapour_attachment"] = "fixed_activity_equilibrium_potential"
        solver["vapour_channels_depends"] = dict(gas_depends)
        if vapour_error is not None:
            gas_axes = ("gas_species.temperature_K",)
        else:
            gas_axes = tuple(
                f"gas_species.{name}.temperature_K"
                for name, detail in gas_result.domain_flags.items()
                if detail is not None
            )
        if refusal is None and vapour_error is not None:
            refusal = EquilibriumRefusal(
                "gas_temperature_out_of_range",
                vapour_error,
                gas_axes,
            )
        elif refusal is None and gas_depends:
            refusal = EquilibriumRefusal(
                "gas_activity_not_identifiable",
                "Some vapour channels need parent activities not identified by the phase state.",
                tuple(f"gas_species.{name}" for name in gas_depends),
            )

    receipt = build_calculation_receipt(
        request=request,
        data_digests=data_digests,
        solver=solver,
    )
    return replace(
        result,
        gas=gas_result,
        fO2=None if fO2 is None else float(fO2),
        gas_depends=gas_depends,
        vapour_error=vapour_error,
        refusal=refusal,
        receipt=receipt,
    )


def equilibrate(
    bulk: Mapping[str, float] | Sequence[float],
    T_K: float,
    *,
    candidates: Sequence[str | CrystalCandidate] | None = None,
    pack: Any | None = None,
    fO2: float | None = None,
    tol: float = _DEFAULT_BALANCE_TOL,
    affinity_tol_ln: float = _DEFAULT_AFFINITY_TOL_LN,
    inner_tol: float = _INNER_BALANCE_TOL,
    max_iter: int = 100,
) -> PhaseEquilibriumResult:
    """Equilibrate a parent-oxide bulk, with optional fixed-fO2 vapour output.

    Vapour is a fixed-activity equilibrium potential over the stated bulk at T;
    it does not close oxygen or vapour mass balance, or imply rate, extent, or
    depletion.
    """
    result = _equilibrate_one(
        bulk, T_K, candidates=candidates, pack=pack, tol=tol,
        affinity_tol_ln=affinity_tol_ln, inner_tol=inner_tol,
        max_iter=max_iter,
    )
    return _attach_phase_state_vapour(result, candidates=candidates, fO2=fO2)


def equilibrate_sweep(
    bulk: Mapping[str, float] | Sequence[float],
    temperatures_K: Sequence[float],
    *,
    candidates: Sequence[str | CrystalCandidate] | None = None,
    pack: Any | None = None,
    tol: float = _DEFAULT_BALANCE_TOL,
    affinity_tol_ln: float = _DEFAULT_AFFINITY_TOL_LN,
    inner_tol: float = _INNER_BALANCE_TOL,
    max_iter: int = 100,
) -> tuple[PhaseEquilibriumResult, ...]:
    """Solve temperatures in order, seeding each point from the previous result.

    Stable phase IDs and the previous kernel parent root seed the next point.
    ``equilibrate``'s domain and independent KKT checks still run at every
    temperature; the seed affects convergence only, never the certificate.
    """
    results: list[PhaseEquilibriumResult] = []
    previous: PhaseEquilibriumResult | None = None
    for temperature in temperatures_K:
        current = _equilibrate_one(
            bulk, temperature, candidates=candidates, pack=pack, tol=tol,
            affinity_tol_ln=affinity_tol_ln, inner_tol=inner_tol,
            max_iter=max_iter, warm_start=previous,
        )
        results.append(current)
        previous = current
    return tuple(results)


@dataclass(frozen=True)
class _BoundaryContext:
    bulk: Mapping[str, float]
    parents: tuple[str, ...]
    states: tuple[_CandidateState, ...]
    all_states: tuple[_CandidateState, ...]
    matrix: np.ndarray
    crystal_gibbs: np.ndarray
    liquid_gibbs: np.ndarray
    pack: Any
    gas_pack: Any
    missing: tuple[str, ...]


def _boundary_context(
    bulk: Mapping[str, float] | Sequence[float],
    temperature_K: float,
    candidates: Sequence[str | CrystalCandidate] | None,
    pack: Any | None,
) -> _BoundaryContext:
    if not _LIQUID_DOMAIN_K[0] <= temperature_K <= _LIQUID_DOMAIN_K[1]:
        raise ValueError("liquid-dependent boundaries are outside the 1700–3000 K domain")
    selected_pack = _loaded_pack(pack)
    if _has_active_excess(selected_pack):
        raise ValueError("active excess-Gibbs terms are outside the ideal-pack boundary model")
    kernel = (
        selected_pack.kernel_datapack
        if isinstance(selected_pack, ImccLoadedDatapack) else selected_pack
    )
    parent_names = tuple(kernel.parent_oxides)
    normalized_bulk, parents = _parse_bulk(bulk, parent_names)
    gas_pack, _ = diagnostics._crystal_thermo_context(None)
    all_states, _crystal_digest, _pure_digest, gas_pack, unavailable, _explicit = (
        _resolve_candidates(candidates, temperature_K, gas_pack)
    )
    all_states, unavailable = _exclude_zero_bulk_infeasible_candidates(
        all_states, unavailable, normalized_bulk, parent_names,
    )
    redox = [state.candidate.phase_id for state in all_states if _redox_candidate(state.candidate)]
    if redox:
        raise ValueError("redox-sensitive crystal phases are unsupported")
    states, _aliases, _rejected = _polymorph_filter(
        all_states, temperature_K, _DEFAULT_AFFINITY_TOL_LN
    )
    states, matrix, crystal_gibbs, unsupported = _candidate_matrix(
        states, parents, parent_names, normalized_bulk
    )
    liquid_gibbs = _melt_liquid_standards(parents, temperature_K, gas_pack)
    return _BoundaryContext(
        normalized_bulk, parents, states, all_states, matrix, crystal_gibbs,
        liquid_gibbs, selected_pack, gas_pack,
        tuple(sorted(set(unavailable) | set(unsupported))),
    )


def _solidus_boundary_residual(
    bulk: Mapping[str, float] | Sequence[float],
    temperature_K: float,
    candidates: Sequence[str | CrystalCandidate] | None,
    pack: Any | None,
    tol: float,
    inner_tol: float,
) -> float:
    """Return min-face log(Q), whose zero is the solidus certificate boundary."""
    context = _boundary_context(bulk, temperature_K, candidates, pack)
    if context.missing:
        raise ValueError("missing crystal functions prevent a solidus certificate")
    b = np.asarray([context.bulk[name] for name in context.parents], dtype=float)
    lp = _lp_solid_minimum(context.matrix, context.crystal_gibbs, b, tol)
    if lp is None:
        raise ValueError("no feasible fully-solid candidate face exists")
    parent_indices = [tuple(_kernel_parent_names(context.pack)).index(name) for name in context.parents]
    _mu, logq, _interval, _certified, diagnostics_map = _minimize_logq_on_dual_face(
        context.matrix, context.crystal_gibbs, b, lp.objective_J,
        lp.chemical_potentials, context.liquid_gibbs, temperature_K,
        context.pack, parent_indices,
    )
    if diagnostics_map.get("logQ_minimum_certified", 0.0) != 1.0:
        raise ValueError("the dual-face log(Q) minimum is unresolved")
    return float(logq)


def _liquidus_boundary_residual(
    bulk: Mapping[str, float] | Sequence[float],
    temperature_K: float,
    candidates: Sequence[str | CrystalCandidate] | None,
    pack: Any | None,
    inner_tol: float,
    max_iter: int,
    y_init: Sequence[float] | None = None,
) -> float:
    """Return the largest crystal affinity in the hypothetical all-liquid state."""
    context = _boundary_context(bulk, temperature_K, candidates, pack)
    if context.missing:
        raise ValueError("missing crystal functions prevent a liquidus certificate")
    liquid = np.asarray([context.bulk[name] for name in context.parents], dtype=float)
    melt = _positive_melt(
        liquid, context.parents, temperature_K, context.pack,
        inner_tol, max_iter, y_init,
    )
    logs = _melt_logs(melt, context.parents)
    by_parent = {name: index for index, name in enumerate(context.parents)}
    affinities = [
        _phase_affinity_from_melt(
            state.candidate, by_parent, context.liquid_gibbs, logs, temperature_K
        )
        for state in context.all_states
        if state.available
        and not (set(state.candidate.stoichiometry) - set(context.parents))
    ]
    if not affinities:
        return -math.inf
    return float(max(affinities))


def _boundary_endpoint_status(result: PhaseEquilibriumResult) -> str:
    if result.refusal is not None:
        return f"{result.labels.validity}:{result.refusal.code}"
    return result.labels.validity


def _boundary_is_certified(result: PhaseEquilibriumResult) -> bool:
    return (
        result.labels.validity == "IN"
        and result.refusal is None
        and result.kkt_recheck.get("independent_kkt_passed") == 1.0
    )


def _boundary_has_positive_liquid(
    result: PhaseEquilibriumResult,
    balance_tol: float,
) -> bool:
    if result.liquid_parent_mol is not None:
        return math.fsum(result.liquid_parent_mol.values()) > 0.0
    liquid_range = result.receipt.solver.get("liquid_parent_scale_range_mol")
    if not isinstance(liquid_range, (tuple, list)) or len(liquid_range) != 2:
        return False
    lower = float(liquid_range[0])
    bulk_scale = math.fsum(result.bulk_parent_mol.values())
    return lower > balance_tol * bulk_scale + _roundoff_bound(
        (lower, bulk_scale), 8
    )


def _source_cap_transition_phases(
    left_state: PhaseEquilibriumResult,
    right_state: PhaseEquilibriumResult,
    balance_tol: float,
    cap_endpoint_phases: Sequence[str] = (),
) -> tuple[tuple[str, ...], bool, bool]:
    """Return cap phases, plus phase-set and liquid-presence change flags."""
    left_capped = set(
        left_state.receipt.solver.get("crystals_above_source_melting_point", ())
    )
    right_capped = set(
        right_state.receipt.solver.get("crystals_above_source_melting_point", ())
    )
    cap_phases = (left_capped ^ right_capped) | set(cap_endpoint_phases)
    if not cap_phases:
        return (), False, False
    left_phase_set = {
        phase for phase, presence in left_state.phase_presence.items()
        if presence == "present"
    }
    right_phase_set = {
        phase for phase, presence in right_state.phase_presence.items()
        if presence == "present"
    }
    phase_set_changed = left_phase_set != right_phase_set
    liquid_presence_changed = (
        _boundary_has_positive_liquid(left_state, balance_tol)
        != _boundary_has_positive_liquid(right_state, balance_tol)
    )
    return (
        tuple(sorted(cap_phases)) if phase_set_changed or liquid_presence_changed else (),
        phase_set_changed,
        liquid_presence_changed,
    )


def phase_boundary_brackets(
    bulk: Mapping[str, float] | Sequence[float],
    T_low_K: float,
    T_high_K: float,
    pack: Any | None = None,
    *,
    candidates: Sequence[str | CrystalCandidate] | None = None,
    scan_points: int = 21,
    tol: float = _DEFAULT_BALANCE_TOL,
    affinity_tol_ln: float = _DEFAULT_AFFINITY_TOL_LN,
    inner_tol: float = _INNER_BALANCE_TOL,
    max_iter: int = 100,
) -> PhaseBoundaryResult:
    """Scan every sampled phase-set change, then refine valid roots with Brent.

    Samples outside 1700–3000 K remain OUT_OF_DOMAIN. A bracket is never
    bridged across a refused or unresolved equilibrium point, and each Brent
    evaluation reruns equilibrium/domain checks before using its residual.
    Crossings in either transition direction are retained in temperature order.
    """
    low = float(T_low_K)
    high = float(T_high_K)
    if (
        not math.isfinite(low) or not math.isfinite(high) or low >= high
        or scan_points < 2
    ):
        raise ValueError("temperature bounds must be finite and increasing; scan_points must be at least two")
    source_limits = _source_solid_limits_K(candidates)
    source_caps = {
        phase: cap for phase, cap in source_limits.items()
        if low <= cap <= high
    }
    scan_temperatures = set(map(float, np.linspace(low, high, int(scan_points))))
    for cap in source_caps.values():
        just_below = float(np.nextafter(cap, -math.inf))
        if just_below >= low:
            scan_temperatures.add(just_below)
        scan_temperatures.add(cap)
        just_above = float(np.nextafter(cap, math.inf))
        if just_above <= high:
            scan_temperatures.add(just_above)
    temperatures = tuple(sorted(scan_temperatures))
    results = equilibrate_sweep(
        bulk, temperatures, candidates=candidates, pack=pack, tol=tol,
        affinity_tol_ln=affinity_tol_ln, inner_tol=inner_tol, max_iter=max_iter,
    )
    out_of_domain_seen = any(
        result.labels.validity == "OUT_OF_DOMAIN"
        and "temperature_K" in result.labels.out_of_domain_axes
        for result in results
    )
    depends_seen = any(
        result.labels.validity == "DEPENDS" or result.refusal is not None
        for result in results
        if result.labels.validity != "OUT_OF_DOMAIN"
    )
    residual_cache: dict[tuple[str, float], float] = {}
    result_cache = {temperature: result for temperature, result in zip(temperatures, results, strict=True)}

    def residual(kind: Literal["solidus", "liquidus"], temperature: float) -> float:
        key = (kind, float(temperature))
        if key in residual_cache:
            return residual_cache[key]
        state = result_cache.get(float(temperature))
        if state is None:
            state = equilibrate(
                bulk, temperature, candidates=candidates, pack=pack, tol=tol,
                affinity_tol_ln=affinity_tol_ln, inner_tol=inner_tol,
                max_iter=max_iter,
            )
            result_cache[float(temperature)] = state
        if state.labels.validity == "OUT_OF_DOMAIN":
            raise ValueError("temperature_K is outside the liquid-equilibrium domain")
        if state.refusal is not None:
            raise ValueError(
                f"boundary point at {temperature:g} K refused: {state.refusal.code}"
            )
        if not _boundary_is_certified(state):
            raise ValueError("boundary point lacks an independent KKT certificate")
        if kind == "solidus":
            value = _solidus_boundary_residual(
                bulk, temperature, candidates, pack, tol, inner_tol
            )
        else:
            y_init = None
            if state.melt is not None:
                x = np.asarray(getattr(state.melt, "parent_x_star", ()), dtype=float)
                names = tuple(_kernel_parent_names(_loaded_pack(pack)))
                active_names = tuple(name for name in names if name in state.bulk_parent_mol)
                if x.size == len(names):
                    selected_x = np.asarray([x[names.index(name)] for name in active_names])
                    if np.all(np.isfinite(selected_x)) and np.all(selected_x > 0.0):
                        y_init = np.log(selected_x)
            value = _liquidus_boundary_residual(
                bulk, temperature, candidates, pack, inner_tol, max_iter, y_init
            )
        residual_cache[key] = value
        if (
            state.refusal is not None
            and abs(value) > affinity_tol_ln
        ):
            raise ValueError("an unresolved equilibrium point is not the boundary root")
        return value

    def refine(
        kind: Literal["solidus", "liquidus"],
        left: float,
        right: float,
        left_state: PhaseEquilibriumResult,
        right_state: PhaseEquilibriumResult,
    ) -> PhaseBoundaryCrossing:
        f_left = residual(kind, left)
        f_right = residual(kind, right)
        if not (math.isfinite(f_left) and math.isfinite(f_right)):
            raise ValueError("boundary residual is not finite at a scan endpoint")
        if abs(f_left) <= affinity_tol_ln:
            root = left
        elif abs(f_right) <= affinity_tol_ln:
            root = right
        elif f_left * f_right > 0.0:
            raise ValueError("scan phase change does not bracket the thermodynamic root")
        else:
            root = float(brentq(
                lambda value: residual(kind, float(value)),
                left, right, xtol=1.0e-8, rtol=4.0 * np.finfo(float).eps,
                maxiter=100,
            ))
        root_residual = residual(kind, root)
        if abs(root_residual) > affinity_tol_ln:
            raise ValueError(
                "refined boundary root residual exceeds the coexistence tolerance"
            )
        if abs(root_residual) <= 8.0 * np.finfo(float).eps:
            final_left = final_right = root
            final_residuals = (root_residual, root_residual)
        else:
            radius = min(0.005, (right - left) / 2.0)
            final_left = final_right = root
            final_residuals = (root_residual, root_residual)
            while radius <= 0.05 + np.finfo(float).eps:
                candidate_left = max(left, root - radius)
                candidate_right = min(right, root + radius)
                candidate_f_left = residual(kind, candidate_left)
                candidate_f_right = residual(kind, candidate_right)
                if candidate_f_left * candidate_f_right <= 0.0:
                    final_left, final_right = candidate_left, candidate_right
                    final_residuals = (candidate_f_left, candidate_f_right)
                    break
                radius *= 2.0
            if final_right - final_left > 0.1 + 1.0e-12:
                raise ValueError("refined boundary bracket exceeds 0.1 K")
        lower_phase_set = tuple(sorted(
            phase for phase, presence in left_state.phase_presence.items()
            if presence == "present"
        ))
        upper_phase_set = tuple(sorted(
            phase for phase, presence in right_state.phase_presence.items()
            if presence == "present"
        ))
        return PhaseBoundaryCrossing(
            kind, (float(final_left), float(final_right)), root, root_residual,
            _boundary_endpoint_status(result_cache[float(final_left)]),
            _boundary_endpoint_status(result_cache[float(final_right)]),
            lower_phase_set, upper_phase_set, final_residuals,
        )

    solidus_crossings: list[PhaseBoundaryCrossing] = []
    liquidus_crossings: list[PhaseBoundaryCrossing] = []
    refinement_failed = False
    refinement_errors: list[str] = []
    source_cap_notices: list[str] = []
    source_cap_solidus_affected = False
    source_cap_liquidus_affected = False
    for index in range(len(results) - 1):
        left_state, right_state = results[index], results[index + 1]
        cap_endpoint_phases = tuple(
            phase for phase, cap in source_caps.items()
            if cap == temperatures[index + 1]
        )
        (
            source_cap_phases,
            phase_set_changed,
            liquid_presence_changed,
        ) = _source_cap_transition_phases(
            left_state, right_state, tol, cap_endpoint_phases,
        )
        if source_cap_phases:
            cap_detail = ",".join(
                f"{phase}@{source_caps.get(phase, temperatures[index + 1]):g}K"
                for phase in source_cap_phases
            )
            source_cap_notices.append(f"source_cap:{cap_detail}")
            if phase_set_changed or liquid_presence_changed:
                source_cap_liquidus_affected = True
            if liquid_presence_changed:
                source_cap_solidus_affected = True
            refinement_errors.append(
                f"source_cap:{temperatures[index]:g}-{temperatures[index + 1]:g}:"
                "phase-set or liquid-presence change at a source cap"
            )
            continue
        if not (_boundary_is_certified(left_state) and _boundary_is_certified(right_state)):
            continue
        left_liquid = _boundary_has_positive_liquid(left_state, tol)
        right_liquid = _boundary_has_positive_liquid(right_state, tol)
        if left_liquid != right_liquid:
            try:
                solidus_crossings.append(refine(
                    "solidus", temperatures[index], temperatures[index + 1],
                    left_state, right_state,
                ))
            except (ValueError, ArithmeticError, ImccRefusal) as exc:
                refinement_failed = True
                refinement_errors.append(
                    f"solidus:{temperatures[index]:g}-{temperatures[index + 1]:g}:{exc}"
                )
        left_crystals_absent = all(
            value in ("absent", "infeasible")
            for value in left_state.phase_presence.values()
        )
        right_crystals_absent = all(
            value in ("absent", "infeasible")
            for value in right_state.phase_presence.values()
        )
        if (
            left_crystals_absent != right_crystals_absent
            and (left_crystals_absent and left_liquid
                 or right_crystals_absent and right_liquid)
        ):
            try:
                liquidus_crossings.append(refine(
                    "liquidus", temperatures[index], temperatures[index + 1],
                    left_state, right_state,
                ))
            except (ValueError, ArithmeticError, ImccRefusal) as exc:
                refinement_failed = True
                refinement_errors.append(
                    f"liquidus:{temperatures[index]:g}-{temperatures[index + 1]:g}:{exc}"
                )

    solidus_status: Literal["AVAILABLE", "NO_CROSSING", "OUT_OF_DOMAIN", "DEPENDS"] = (
        "DEPENDS" if source_cap_solidus_affected else
        "AVAILABLE" if solidus_crossings else
        "DEPENDS" if depends_seen or refinement_failed else
        "OUT_OF_DOMAIN" if out_of_domain_seen else "NO_CROSSING"
    )
    liquidus_status: Literal["AVAILABLE", "NO_CROSSING", "OUT_OF_DOMAIN", "DEPENDS"] = (
        "DEPENDS" if source_cap_liquidus_affected else
        "AVAILABLE" if liquidus_crossings else
        "DEPENDS" if depends_seen or refinement_failed else
        "OUT_OF_DOMAIN" if out_of_domain_seen else "NO_CROSSING"
    )
    base_labels = results[0].labels
    source_cap_affected = source_cap_solidus_affected or source_cap_liquidus_affected
    validity: Literal["IN", "OUT_OF_DOMAIN", "DEPENDS"] = (
        "DEPENDS" if depends_seen or refinement_failed or source_cap_affected else
        "OUT_OF_DOMAIN" if out_of_domain_seen else "IN"
    )
    axes = ("temperature_K",) if out_of_domain_seen else ()
    labels = replace(
        base_labels,
        validity=validity,
        out_of_domain_axes=axes,
        notices=tuple(dict.fromkeys((
            *base_labels.notices,
            *source_cap_notices,
            "temperature_scan_refined_by_phase_set_changes",
            *(('liquid_dependent_results_below_1700_K_out_of_domain',) if low < _LIQUID_DOMAIN_K[0] else ()),
        ))),
    )
    refusal = None
    if depends_seen or refinement_failed or source_cap_affected:
        refusal = EquilibriumRefusal(
            "boundary_refinement_unresolved",
            "A required scan point or phase-boundary refinement was refused or unresolved.",
            tuple(axis for axis, present in (
                ("solidus", solidus_status == "DEPENDS"),
                ("liquidus", liquidus_status == "DEPENDS"),
            ) if present),
            {"refinement_failed": float(refinement_failed)},
        )
    elif out_of_domain_seen:
        refusal = EquilibriumRefusal(
            "out_of_domain",
            "Liquid-dependent boundaries outside 1700–3000 K are OUT_OF_DOMAIN on temperature_K.",
            ("temperature_K",),
        )
    first_receipt = results[0].receipt
    receipt = build_calculation_receipt(
        request={
            "bulk_parent_mol": dict(results[0].bulk_parent_mol),
            "temperature_range_K": (low, high),
            "pressure_bar": _PRESSURE_BAR,
        },
        data_digests=first_receipt.data_digests,
        solver={
            "method": "phase_set_scan_with_brent_refinement",
            "scan_points": int(scan_points),
            "solidus_brackets_K": tuple(item.bracket_K for item in solidus_crossings),
            "liquidus_brackets_K": tuple(item.bracket_K for item in liquidus_crossings),
            "solidus_status": solidus_status,
            "liquidus_status": liquidus_status,
            "scan_endpoint_statuses": tuple(
                _boundary_endpoint_status(result) for result in results
            ),
            "scan_liquid_parent_scale_ranges_mol": tuple(
                result.receipt.solver.get("liquid_parent_scale_range_mol")
                for result in results
            ),
            "scan_phase_sets": tuple(
                tuple(sorted(
                    phase for phase, presence in result.phase_presence.items()
                    if presence == "present"
                ))
                for result in results
            ),
            "boundary_refinement_errors": tuple(refinement_errors),
            "pressure_bar": _PRESSURE_BAR,
        },
    )
    return PhaseBoundaryResult(
        _PRESSURE_BAR, tuple(solidus_crossings), tuple(liquidus_crossings),
        solidus_status, liquidus_status, labels, receipt, refusal,
    )
