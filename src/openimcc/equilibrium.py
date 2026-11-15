"""Nested ideal-melt and stoichiometric-crystal phase equilibrium.

The liquid remains delegated to openimcc.model.evaluate; this module solves
the outer parent-balance/crystal problem and independently checks its result.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from scipy.optimize import LinearConstraint, linprog, lsq_linear, minimize
from scipy.special import logsumexp

from openimcc import diagnostics
from openimcc._constants import MOLAR_MASS_G_MOL, R_J_MOL_K
from openimcc.declared_assemblage import (
    EquilibriumRefusal,
    PhaseEquilibriumLabels,
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
_PRESSURE_BAR = 1.0
_LIQUID_DOMAIN_K = (1700.0, 3000.0)

# Each named tolerance applies to a different physical/numerical certificate.
_DEFAULT_BALANCE_TOL = 1.0e-10
_INNER_BALANCE_TOL = 1.0e-12
_DEFAULT_AFFINITY_TOL_LN = 1.0e-8
_DUAL_SLACK_TOL_RT = 1.0e-10
_LOGQ_COEXISTENCE_TOL = 1.0e-8
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
    phase_presence: Mapping[str, Literal["present", "absent", "unresolved"]]
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

    def __post_init__(self) -> None:
        for name in (
            "bulk_parent_mol", "phase_amount_ranges_mol", "phase_presence",
            "identifiable_activities", "activity_ranges_ln",
            "phase_affinities_ln", "phase_affinity_ranges_ln", "kkt_recheck",
        ):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        for name in ("liquid_parent_mol", "solid_mol"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, MappingProxyType(dict(value)))


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


def _default_candidate_states(
    temperature_K: float,
) -> tuple[tuple[_CandidateState, ...], str, str, Any, tuple[str, ...]]:
    from openimcc import gas

    pure, pure_digest = _load_json(_PURE_RESOURCE)
    crystals, crystal_digest = _load_json(_CRYSTAL_RESOURCE)
    pure_by_id = {row["id"]: row for row in pure["phases"]}
    crystal_by_id = {row["id"]: row for row in crystals["phases"]}
    gas_pack, _ = diagnostics._crystal_thermo_context(None)
    states: list[_CandidateState] = []
    unavailable: list[str] = []

    # For crystal fusion nodes, the phase standard is G_cr=ΣνG_l−ΔG_fus.
    # ν is mol oxide/mol formula and the stored fusion node is kJ/mol formula,
    # so multiplying the latter by 1000 makes both terms J/mol formula.
    for phase_id, row in sorted(crystal_by_id.items()):
        nodes = sorted(float(value) for value in row["fusion_gibbs_nodes_kJ_mol"])
        domain = (
            max(float(row["valid_temperature_K"][0]), nodes[0]),
            min(float(row["valid_temperature_K"][1]), nodes[-1]),
        )
        authority = _phase_authority(row)
        if not domain[0] <= temperature_K <= domain[1]:
            unavailable.append(phase_id)
            placeholder = CrystalCandidate(
                phase_id, row["oxide_stoichiometry"], 0.0, row.get("formula"),
                authority, domain, row.get("source", {}),
            )
            states.append(_CandidateState(placeholder, authority, domain, False, "temperature_out_of_range"))
            continue
        try:
            liquid_gibbs = {
                oxide: gas.species_thermo(oxide, "l", temperature_K, gas_pack).G_J_mol
                for oxide in row["oxide_stoichiometry"]
            }
            fusion_kJ = diagnostics._interpolate_crystal_fusion_nodes(row, temperature_K)
        except (gas.ImccGasTemperatureOutsideDomainError, ValueError):
            unavailable.append(phase_id)
            placeholder = CrystalCandidate(
                phase_id, row["oxide_stoichiometry"], 0.0, row.get("formula"),
                authority, domain, row.get("source", {}),
            )
            states.append(_CandidateState(placeholder, authority, domain, False, "standard_out_of_range"))
            continue
        g_crystal = math.fsum(
            float(nu) * liquid_gibbs[oxide]
            for oxide, nu in row["oxide_stoichiometry"].items()
        ) - fusion_kJ * 1000.0
        candidate = CrystalCandidate(
            phase_id, row["oxide_stoichiometry"], g_crystal,
            row.get("formula"), authority, domain, row.get("source", {}),
        )
        states.append(_CandidateState(candidate, authority, domain, True))

    # JANAF pure-oxide G values use the single packaged reconstruction and its
    # stable crystal cap, preserving the transition-row convention.
    for phase_id, row in sorted(pure_by_id.items()):
        try:
            crystal = evaluate_janaf_crystal(phase_id, temperature_K, apply_phase_cap=True)
        except JanafCrystalTemperatureError as exc:
            unavailable.append(phase_id)
            authority = str(row.get("source_authority", "external_datapack"))
            placeholder = CrystalCandidate(
                phase_id, row["oxide_stoichiometry"], 0.0, phase_id,
                authority, exc.domain_K, {},
            )
            states.append(_CandidateState(placeholder, authority, exc.domain_K, False, "temperature_out_of_range"))
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
) -> tuple[tuple[_CandidateState, ...], str, str, Any, tuple[str, ...], bool]:
    defaults, crystal_digest, pure_digest, gas_pack, unavailable = (
        _default_candidate_states(temperature_K)
    )
    if candidates is None:
        return defaults, crystal_digest, pure_digest, gas_pack, unavailable, True
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
        tuple(state.candidate.phase_id for state in selected if not state.available), False,
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


def _lp_solid_minimum(A: np.ndarray, g: np.ndarray, b: np.ndarray) -> _LPState | None:
    # The primal is min gᵀn subject to An=b,n≥0; its dual is max bᵀμ with
    # Aᵀμ≤g. g and μ are J/mol formula and J/mol parent respectively, so both
    # objectives are J for the supplied molar bulk and slack has units J/mol.
    if A.shape[1] == 0:
        return None
    solved = linprog(g, A_eq=A, b_eq=b, bounds=(0.0, None), method="highs")
    if not solved.success:
        return None
    amounts = np.asarray(solved.x, dtype=float)
    mu = np.asarray(solved.eqlin.marginals, dtype=float)
    objective = float(g @ amounts)
    slack = g - A.T @ mu
    amount_ranges: list[tuple[float, float]] = []
    amount_ranges_certified = True
    for j in range(A.shape[1]):
        cost = np.zeros(A.shape[1]); cost[j] = 1.0
        # Add gᵀn=Gmin so each LP endpoint ranges only over primal minima.
        eq = np.vstack((A, g)); rhs = np.concatenate((b, [objective]))
        low = linprog(cost, A_eq=eq, b_eq=rhs, bounds=(0.0, None), method="highs")
        high = linprog(-cost, A_eq=eq, b_eq=rhs, bounds=(0.0, None), method="highs")
        if low.success and high.success:
            amount_ranges.append((float(low.fun), float(-high.fun)))
        else:
            # A failed range solve cannot establish phase presence or uniqueness.
            amount_ranges.append((0.0, math.inf))
            amount_ranges_certified = False
    return _LPState(
        amounts, mu, objective, slack, tuple(amount_ranges), amount_ranges_certified
    )


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
    # The dual face is Aᵀμ≤g, bᵀμ=Gmin. At its minimizer, KKT stationarity is
    # ∇logQ+λb+A_activeη=0 with λ free and η≥0 for the crystal inequalities;
    # the scaled residual is multiplied by RT to report dimensionless logQ.
    constraints = (
        LinearConstraint(A.T, np.full(A.shape[1], -np.inf), g),
        LinearConstraint(b[None, :], [objective], [objective]),
    )
    result = minimize(
        lambda mu: _logq_from_potentials(mu, liquid_gibbs, temperature_K, pack, parent_indices)[0],
        np.asarray(start, dtype=float),
        jac=lambda mu: _logq_from_potentials(mu, liquid_gibbs, temperature_K, pack, parent_indices)[1],
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1.0e-13, "maxiter": 1000},
    )
    mu = np.asarray(result.x, dtype=float)
    logq, gradient = _logq_from_potentials(mu, liquid_gibbs, temperature_K, pack, parent_indices)
    slack = g - A.T @ mu
    face_residual = abs(float(b @ mu) - objective)
    active = np.flatnonzero(slack <= 1.0e-9 * R_J_MOL_K * temperature_K)
    normals = np.column_stack((b, A[:, active]))
    lower = np.concatenate(([-np.inf], np.zeros(len(active))))
    upper = np.full(1 + len(active), np.inf)
    stationarity = lsq_linear(
        normals, -gradient, bounds=(lower, upper), tol=1.0e-14,
        lsmr_tol="auto", max_iter=1000,
    )
    stationarity_scaled = float(
        np.linalg.norm(normals @ stationarity.x + gradient, ord=np.inf)
        * R_J_MOL_K * temperature_K
    )
    # For slack s=g−Aᵀμ≥0 and multiplier η≥0, KKT requires ηs=0. The
    # multiplier has inverse-energy units, so this product is dimensionless.
    complementarity = float(np.max(
        np.abs(stationarity.x[1:] * slack[active]), initial=0.0
    ))
    dual_violation = float(np.max(np.maximum(-slack, 0.0), initial=0.0))
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
    minimum_certified = (
        bool(result.success)
        and stationarity_scaled <= _DEFAULT_AFFINITY_TOL_LN
        and complementarity <= _DEFAULT_AFFINITY_TOL_LN
        and dual_violation <= _DUAL_SLACK_TOL_RT * R_J_MOL_K * temperature_K
        and face_residual <= face_error
    )
    residuals = {
        "logQ_optimizer_success": float(bool(result.success)),
        "logQ_minimum_certified": float(minimum_certified),
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


def _positive_melt(
    liquid: np.ndarray,
    parents: Sequence[str],
    temperature_K: float,
    pack: Any,
    inner_tol: float,
    max_iter: int,
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
    )


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
) -> tuple[np.ndarray, np.ndarray, Any] | None:
    trial = full_n.copy()
    if active:
        trial[np.asarray(active, dtype=int)] = n_active
    liquid, _error = _safe_liquid(A, trial, b, balance_tol)
    if liquid is None:
        return None
    melt = _positive_melt(liquid, parents, temperature_K, pack, inner_tol, max_iter)
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
) -> tuple[np.ndarray, bool, str, dict[str, float]]:
    if not active:
        return n, True, "converged", {}
    scales = _scaled_phase_amounts(A, b)
    diagnostics_map: dict[str, float] = {}
    for outer_iteration in range(_MAX_OUTER_ITERATIONS):
        current = _active_residual(
            n[active], active, n, A, b, states, parents, liquid_gibbs,
            temperature_K, pack, inner_tol, max_iter, balance_tol,
        )
        if current is None:
            return n, False, "liquid_boundary", diagnostics_map
        residual, _trial, _melt = current
        norm = float(np.linalg.norm(residual, ord=np.inf))
        diagnostics_map["active_affinity_inf"] = norm
        diagnostics_map["active_iterations"] = float(outer_iteration + 1)
        if norm <= affinity_tol_ln:
            return n, True, "converged", diagnostics_map

        # Finite differences are scaled in formula moles. At a boundary use a
        # feasible one-sided probe; every probe remains in P={n≥0, An≤b}.
        jacobian = np.zeros((len(active), len(active)), dtype=float)
        base = n[active].copy()
        for column, phase_index in enumerate(active):
            step = _FD_STEP_SCALED * max(1.0, abs(base[column] / scales[phase_index])) * scales[phase_index]
            plus = base.copy(); plus[column] += step
            plus_result = _active_residual(
                plus, active, n, A, b, states, parents, liquid_gibbs,
                temperature_K, pack, inner_tol, max_iter, balance_tol,
            )
            minus_result = None
            if base[column] >= step:
                minus = base.copy(); minus[column] -= step
                minus_result = _active_residual(
                    minus, active, n, A, b, states, parents, liquid_gibbs,
                    temperature_K, pack, inner_tol, max_iter, balance_tol,
                )
            if plus_result is not None and minus_result is not None:
                jacobian[:, column] = (
                    plus_result[0] - minus_result[0]
                ) / (2.0 * step)
            elif plus_result is not None:
                jacobian[:, column] = (plus_result[0] - residual) / step
            elif minus_result is not None:
                jacobian[:, column] = (residual - minus_result[0]) / step
            else:
                return n, False, "no_feasible_jacobian", diagnostics_map

        scaled_jacobian = jacobian * scales[np.asarray(active)][None, :]
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
        if rank == 0:
            return n, False, "no_active_jacobian", diagnostics_map
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
                balance_tol,
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
            return n, False, "inconsistent_or_stalled", diagnostics_map
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
) -> tuple[dict[int, tuple[float, float]], bool, bool]:
    # Active crystal equalities share one μ. Thus every feasible combination
    # satisfying A_active n=b−l has the same solid Gibbs energy, and LP endpoint
    # solves give the true amount range without selecting an arbitrary split.
    result: dict[int, tuple[float, float]] = {}
    certified = True
    if not indices:
        return result, True, True
    selected_A = A[:, np.asarray(indices, dtype=int)]
    for local, global_index in enumerate(indices):
        cost = np.zeros(len(indices)); cost[local] = 1.0
        low = linprog(
            cost, A_eq=selected_A, b_eq=residual_inventory,
            bounds=(0.0, None), method="highs",
        )
        high = linprog(
            -cost, A_eq=selected_A, b_eq=residual_inventory,
            bounds=(0.0, None), method="highs",
        )
        if low.success and high.success:
            result[global_index] = (float(low.fun), float(-high.fun))
        else:
            result[global_index] = (0.0, math.inf)
            certified = False
    unique = certified and all(
        high - low <= tol * scales[index] + _roundoff_bound((low, high), 8)
        for index, (low, high) in result.items()
    )
    return result, unique, certified


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
) -> CalculationReceipt:
    kernel = pack.kernel_datapack if isinstance(pack, ImccLoadedDatapack) else pack
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
            "janaf_pure_oxide_crystals": pure_digest,
            "parent_liquid_standards": diagnostics._crystal_thermo_context(None)[0].condensate_table_digest,
        },
        solver={
            "method": "nested_feasible_active_set_with_solid_lp",
            "bulk_balance_relative_tolerance": _DEFAULT_BALANCE_TOL,
            "inner_balance_relative_tolerance": _INNER_BALANCE_TOL,
            "outer_affinity_tolerance_ln": _DEFAULT_AFFINITY_TOL_LN,
            "dual_slack_tolerance_over_RT": _DUAL_SLACK_TOL_RT,
            "pressure_bar": _PRESSURE_BAR,
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
        state.authority for state in states if state.available
    ]
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
            "janaf_pure_oxide_crystals": "openimcc-janaf-apparent-crystals-v2",
            "standard_pressure_bar": "1.0",
        },
        coverage={
            state.candidate.phase_id: (
                f"{state.domain[0]:g}-{state.domain[1]:g} K"
                if state.available and state.domain else state.reason or "unavailable"
            )
            for state in states
        },
        validity=validity,
        out_of_domain_axes=tuple(axes),
        evidence_class=_least_authority(authorities),
        flags=tuple(dict.fromkeys(flags)),
        notices=tuple(dict.fromkeys((
            *notices,
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
    return PhaseEquilibriumResult(
        temperature_K=temperature_K,
        pressure_bar=_PRESSURE_BAR,
        bulk_parent_mol=bulk,
        liquid_parent_mol=None,
        liquid_mass_fraction=None,
        solid_mol=None,
        phase_amount_ranges_mol=solid_ranges or {phase: (0.0, 0.0) for phase in phase_ids},
        phase_amounts_unique=False,
        phase_presence={phase: "unresolved" for phase in phase_ids},
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
) -> tuple[bool, dict[str, float], Any | None, dict[str, float]]:
    # This verifier deliberately reconstructs balances and affinities from the
    # returned vectors and a fresh melt solve; it consumes none of the outer
    # solver's residuals or Jacobians.
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
            liquid_vector, parents, temperature_K, pack, inner_tol, 200
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
            dg_fus = math.fsum(
                c.stoichiometry[name] * liquid_gibbs[index[name]]
                for name in c.stoichiometry
            ) - c.G_crystal_J_mol
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
                ) - candidate.G_crystal_J_mol
            ) / (R_J_MOL_K * temperature_K)
        kernel = pack.kernel_datapack if isinstance(pack, ImccLoadedDatapack) else pack
        dual_logs = (dual_mu - liquid_gibbs) / (R_J_MOL_K * temperature_K)
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
             *map(float, liquid_gibbs / (R_J_MOL_K * temperature_K))],
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
) -> PhaseEquilibriumResult:
    b = np.asarray([bulk[name] for name in parents], dtype=float)
    n = np.zeros(A.shape[1], dtype=float)
    scales = _scaled_phase_amounts(A, b) if A.shape[1] else np.zeros(0)
    phase_ids = [state.candidate.phase_id for state in states]
    phase_index = {phase: i for i, phase in enumerate(phase_ids)}
    history: list[tuple[str, ...]] = []
    history_seen: set[tuple[str, ...]] = set()
    state_history_seen: set[tuple[tuple[str, ...], tuple[float, ...]]] = set()
    residual_sign_history: list[tuple[tuple[str, int], ...]] = []
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
            "polymorph_tie_breaks": polymorph_tie_breaks,
            "deterministic_tie_breaks": tuple(tie_breaks),
            **extra,
        }

    # Parent columns requiring an exact-zero bulk component were removed before
    # this function, so every column below is feasible at a sufficiently small
    # positive extent.
    active: list[int] = []
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
                current_liquid, parents, temperature_K, pack, inner_tol, max_iter
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

        active_bad = [
            index for index in active
            if abs(affinities.get(phase_ids[index], math.inf)) > affinity_tol_ln
        ]
        if active_bad:
            if repeated_state:
                solved_n, converged, reason = n.copy(), False, "repeated_state"
            else:
                solved_n, converged, reason, _solver_residuals = _solve_active_equalities(
                    n.copy(), active.copy(), A, b, states, parents, liquid_gibbs,
                    temperature_K, pack, affinity_tol_ln, balance_tol, inner_tol,
                    max_iter,
                )
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
                    "active_set_cycle" if cycle_detected else "active_set_incompatibility",
                        "Active crystal equalities could not be solved within the affinity tolerance.",
                        tuple(f"phase.{phase_ids[i]}" for i in active),
                        {"active_affinity_inf_ln": max(
                            (abs(affinities.get(phase_ids[i], math.inf)) for i in active),
                            default=math.inf,
                        )},
                    ),
                    solver=history_solver(failure_reason=reason),
                )
            most = sorted(
                trial_absent,
                key=lambda i: (-affinities[phase_ids[i]], phase_ids[i]),
            )[0]
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
                trial_result = _solve_active_equalities(
                    trial_n, trial_active, A, b, states, parents, liquid_gibbs,
                    temperature_K, pack, affinity_tol_ln, balance_tol,
                    inner_tol, max_iter,
                )
                trial_n, trial_ok, _reason, _diag = trial_result
                if not trial_ok:
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
                continue
            return _empty_result(
                bulk=bulk, temperature_K=temperature_K, pack=pack,
                states=candidates_all, crystal_digest=crystal_digest,
                pure_digest=pure_digest, explicit_candidates=explicit_candidates,
                refusal=EquilibriumRefusal(
                    "active_set_cycle" if cycle_detected else "active_set_incompatibility",
                    "No deterministic one- or two-phase replacement set passed all crystal inequalities.",
                    tuple(f"phase.{phase_ids[i]}" for i in active),
                    {"replacement_trials": float(len(replacement_sets))},
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
        residual_inventory = b - current_liquid
        ranges, unique, ranges_certified = _phase_extent_ranges(
            A, n, active, residual_inventory, scales, balance_tol
        )
        state_amounts: dict[str, float] = {}
        state_ranges: dict[str, tuple[float, float]] = {}
        presence: dict[str, Literal["present", "absent", "unresolved"]] = {}
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
                "polymorph_tie_breaks": polymorph_tie_breaks,
                "deterministic_tie_breaks": tie_breaks,
                "outer_iterations": len(history),
                "phase_amount_ranges_certified": ranges_certified,
                "kkt_recheck": recheck,
            },
        )
        return PhaseEquilibriumResult(
            temperature_K=temperature_K,
            pressure_bar=_PRESSURE_BAR,
            bulk_parent_mol=bulk,
            liquid_parent_mol=positive,
            liquid_mass_fraction=_mass_fraction(positive, bulk),
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
    phase_presence: dict[str, Literal["present", "absent", "unresolved"]] = {}
    amounts: dict[str, float] = {}
    scales = _scaled_phase_amounts(A, b)
    amount_unique = lp.amount_ranges_certified
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
    for state in all_states:
        phase = state.candidate.phase_id
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
    present_indices = np.flatnonzero(lp.amounts > balance_tol * scales)
    if present_indices.size:
        # Complementary slackness makes every positive-amount crystal equality
        # tight on the full dual optimal face. If their stoichiometric columns
        # span the parent space, those equalities pin μ and therefore each a_i;
        # this avoids computing the dual-face min/max ranges reserved for a
        # separate boundary analysis. μ and G are J/mol, so division by RT is
        # dimensionless; a pure one-parent solid gives its standard activity.
        present_matrix = A[:, present_indices]
        if np.linalg.matrix_rank(present_matrix) == len(parents):
            log_activities = (mu_q - liquid_gibbs) / (R_J_MOL_K * temperature_K)
            for parent, log_activity in zip(parents, log_activities, strict=True):
                value = float(log_activity)
                activities[parent] = math.exp(value)
                activity_ranges[parent] = (value, value)

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
        or np.any(original_dual_slack < -_DUAL_SLACK_TOL_RT * R_J_MOL_K * temperature_K)
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
        (("crystal_only_minimum",) if out_of_domain else ()),
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
            "kkt_recheck": recheck,
        },
    )
    return PhaseEquilibriumResult(
        temperature_K=temperature_K,
        pressure_bar=_PRESSURE_BAR,
        bulk_parent_mol=bulk,
        liquid_parent_mol=None,
        liquid_mass_fraction=0.0,
        solid_mol=amounts if amount_unique else None,
        phase_amount_ranges_mol=solid_ranges,
        phase_amounts_unique=amount_unique,
        phase_presence=phase_presence,
        saturated_candidates=tuple(sorted(saturated)),
        identifiable_activities=activities,
        activity_ranges_ln=activity_ranges,
        phase_affinities_ln=affinities,
        phase_affinity_ranges_ln=affinity_ranges,
        phase_saturation_indices=_make_saturation_rows(all_states, affinities, temperature_K),
        logQ=q_value,
        logQ_interval=q_interval,
        melt=None,
        gas=None,
        mass_balance_relative_inf=recheck.get("balance_relative_inf"),
        labels=labels,
        receipt=receipt,
        refusal=refusal,
        kkt_recheck=recheck,
        crystal_only_minimum=out_of_domain,
    )


def equilibrate(
    bulk: Mapping[str, float] | Sequence[float],
    T_K: float,
    *,
    candidates: Sequence[str | CrystalCandidate] | None = None,
    pack: Any | None = None,
    tol: float = _DEFAULT_BALANCE_TOL,
    affinity_tol_ln: float = _DEFAULT_AFFINITY_TOL_LN,
    inner_tol: float = _INNER_BALANCE_TOL,
    max_iter: int = 100,
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
    try:
        all_states, crystal_digest, pure_digest, gas_pack, unavailable, explicit = (
            _resolve_candidates(candidates, candidate_temperature)
        )
    except Exception as exc:
        # Unknown crystal IDs and malformed candidate input are typed as
        # missing phase data at this API boundary.
        fallback_states, crystal_digest, pure_digest, gas_pack, _unavailable, _explicit = (
            _resolve_candidates(None, candidate_temperature)
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
    missing = tuple(sorted(set(unavailable) | set(unsupported)))

    if gas_pack is None:
        from openimcc import diagnostics as diagnostic_module
        gas_pack, _ = diagnostic_module._crystal_thermo_context(None)
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
    lp = _lp_solid_minimum(A, g, b)
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
            return _solid_result(
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

    result = _positive_branch(
        bulk=normalized_bulk,
        parents=parents,
        states=states,
        A=A,
        g=g,
        liquid_gibbs=liquid_gibbs,
        temperature_K=temperature,
        pack=selected_pack,
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
    )
    return result
