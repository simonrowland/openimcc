"""IMCC-SF04 kernel: 8-parent ideal associated-solution melt-activity solver.

This is the independent diagnostic shadow engine described in
``docs/imcc-sf04-spec.md``. It solves the 8 coupled nonlinear parent-oxide
balances for 46 species (8 unbound parents + 38 complexes) using mass-action
complexing in an ideal associated-solution framework.

Derivation comments (premise -> algebra -> unit check -> sanity case) are
required around the residual construction and the mole-fraction bookkeeping.
The gamma_i = x_i*/x_i subtlety (different denominators) is explicitly noted.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import Decimal
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares


LOG10 = math.log(10.0)
_CONTINUATION_LOGK_STEP = 2.0 * LOG10
_CONTINUATION_INITIAL_COMPLEX_X = 1.0e-4
_UNTRUSTED_IDENTITY_TOKEN = "internal-analytical"
_PUBLISHED_MODEL_ID = "IMCC-SF04"
_PUBLISHED_EXTENSION_MODEL_ID = "IMCC-SF04-EXT"
_PUBLISHED_COVERAGE = "A-published-imcc"
_PUBLISHED_PARENT_OXIDES = (
    "SiO2",
    "MgO",
    "FeO",
    "CaO",
    "Al2O3",
    "TiO2",
    "Na2O",
    "K2O",
)
_PUBLISHED_CORE_ROWS = 38
# Complete canonical v1.0.2 published FC87/SF04 datapack, including every
# field of every row and the top-level identity/version. Changing this digest
# is deliberate published-datapack re-versioning, never a silent row edit.
_PUBLISHED_DATAPACK_SHA256 = (
    "f2b479cd54e3c82704a5863fcc06836f72045375d9a8c7f8d2fad19e98f75d05"
)


# --------------------------------------------------------------------------- #
# Typed refusals
# --------------------------------------------------------------------------- #


class ImccRefusal(Exception):
    """Base for IMCC-SF04 domain/refusal errors."""


class ImccTOutsideDatapackDomainError(ImccRefusal):
    code = "imcc_T_outside_datapack_domain"


class ImccCompositionIncompleteError(ImccRefusal):
    code = "imcc_composition_incomplete"


class ImccFerricInputUnsupportedError(ImccRefusal):
    code = "imcc_ferric_input_unsupported"


class ImccComponentOutsideDomainError(ImccRefusal):
    code = "imcc_component_outside_domain"


class ImccNonconvergenceError(ImccRefusal):
    code = "imcc_nonconvergence"

    def __init__(self, message: str, diagnostics: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


# --------------------------------------------------------------------------- #
# Data pack
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _ImccDatapackIdentity:
    model_id: str
    evidence_class: str
    coverage: Mapping[str, str]
    proven: bool


def _untrusted_datapack_identity() -> _ImccDatapackIdentity:
    return _ImccDatapackIdentity(
        model_id=_UNTRUSTED_IDENTITY_TOKEN,
        evidence_class=_UNTRUSTED_IDENTITY_TOKEN,
        coverage=MappingProxyType({}),
        proven=False,
    )


def _immutable_array_copy(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values)
    return np.frombuffer(
        contiguous.tobytes(),
        dtype=contiguous.dtype,
    ).reshape(contiguous.shape)


@dataclass(frozen=True)
class ImccDatapack:
    """
    Synthetic/real data pack for the IMCC-SF04 mass-action complexing model.

    Fields
    ------
    reactions:
        Complex names, length = n_complexes.
    nu:
        Stoichiometric matrix, shape (n_parents, n_complexes).
        ``nu[i, j]`` = moles of unbound parent ``i`` in one mole of complex ``j``.
    A, B:
        log10 equilibrium constants, ``log10 K_j(T) = A_j + B_j / T_K``.
    domains:
        Per-complex demonstrated temperature domain as ``[(T_low, T_high), ...]``.
    version:
        Review-gated data pack version string.
    parent_oxides:
        Ordered parent oxide names, length = n_parents.
    """

    reactions: Sequence[str]
    nu: np.ndarray
    A: np.ndarray
    B: np.ndarray
    domains: Sequence[tuple[float, float]]
    version: str
    parent_oxides: Sequence[str] = field(
        default_factory=lambda: (
            "SiO2",
            "MgO",
            "FeO",
            "CaO",
            "Al2O3",
            "TiO2",
            "Na2O",
            "K2O",
        )
    )
    _identity: _ImccDatapackIdentity = field(
        default_factory=_untrusted_datapack_identity,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        # Convert to arrays for shape validation and arithmetic.
        nu = np.asarray(self.nu, dtype=float)
        A = np.asarray(self.A, dtype=float)
        B = np.asarray(self.B, dtype=float)
        if nu.ndim != 2:
            raise ValueError("nu must be a 2-D matrix")
        n_parents, n_complexes = nu.shape
        if A.shape != (n_complexes,):
            raise ValueError(
                f"A shape {A.shape} does not match n_complexes={n_complexes}"
            )
        if B.shape != (n_complexes,):
            raise ValueError(
                f"B shape {B.shape} does not match n_complexes={n_complexes}"
            )
        if len(self.domains) != n_complexes:
            raise ValueError("domains length must match n_complexes")
        if len(self.reactions) != n_complexes:
            raise ValueError("reactions length must match n_complexes")
        if len(self.parent_oxides) != n_parents:
            raise ValueError("parent_oxides length must match nu rows")
        # Fractional stoichiometries are allowed; negative coefficients are not.
        if np.any(nu < 0.0):
            raise ValueError("nu must be non-negative")
        # An all-zero column has no required parents, so the later
        # ``np.any((nu > 0) & inactive_parent)`` test would mark it active
        # vacuously and mass action would emit x_j = K_j (empty product = 1).
        # That is a composition-independent mole fraction, not a complex.
        # Sanity: one-parent nu=[[0]], log10(K)=-1 previously returned
        # complex_x=[0.1], D=0.9 with status="converged".
        if np.any(np.all(nu == 0.0, axis=0)):
            raise ValueError(
                "each complex column of nu must contain at least one "
                "positive coefficient"
            )
        if not np.all(np.isfinite(A)) or not np.all(np.isfinite(B)):
            raise ValueError("A and B must be finite")
        domains_out = []
        for idx, window in enumerate(self.domains):
            try:
                low_raw, high_raw = window
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"domains[{idx}] must be a (T_low, T_high) pair"
                ) from exc
            low = float(low_raw)
            high = float(high_raw)
            # T < nan and T > nan are both False, so a NaN/NaN window was
            # treated as in-domain and extrapolated stayed False. Same silent
            # pass for (-inf, inf). Refuse non-finite endpoints here; inverted
            # finite windows still fail-loud later at the T gate.
            if not (math.isfinite(low) and math.isfinite(high)):
                raise ValueError(
                    f"domains[{idx}] endpoints must be finite Kelvin values, "
                    f"got ({low_raw!r}, {high_raw!r})"
                )
            domains_out.append((low, high))
        object.__setattr__(self, "reactions", tuple(self.reactions))
        object.__setattr__(self, "domains", tuple(domains_out))
        object.__setattr__(self, "parent_oxides", tuple(self.parent_oxides))
        object.__setattr__(self, "nu", nu)
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "B", B)

    @property
    def n_parents(self) -> int:
        return self.nu.shape[0]

    @property
    def n_complexes(self) -> int:
        return self.nu.shape[1]

    @property
    def n_species(self) -> int:
        return self.n_parents + self.n_complexes

    @property
    def model_id(self) -> str:
        return self._identity.model_id

    @property
    def evidence_class(self) -> str:
        return self._identity.evidence_class

    @property
    def coverage(self) -> Mapping[str, str]:
        return self._identity.coverage

    @property
    def identity_is_proven(self) -> bool:
        return self._identity.proven


def _canonical_published_serialization(value: Any) -> bytes:
    """Serialize JSON data with sorted keys and value-normalized numbers."""

    def encode(item: Any) -> str:
        if item is None:
            return "null"
        if item is True:
            return "true"
        if item is False:
            return "false"
        if isinstance(item, (int, float)):
            number = Decimal(str(item))
            if not number.is_finite():
                raise ValueError("canonical manifest numbers must be finite")
            if number == 0:
                return "0"
            token = format(number, "f")
            if "." in token:
                token = token.rstrip("0").rstrip(".")
            return token
        if isinstance(item, str):
            return json.dumps(item, ensure_ascii=True, separators=(",", ":"))
        if isinstance(item, (list, tuple)):
            return "[" + ",".join(encode(member) for member in item) + "]"
        if isinstance(item, Mapping):
            if not all(isinstance(key, str) for key in item):
                raise TypeError("canonical manifest object keys must be strings")
            members = (
                json.dumps(key, ensure_ascii=True, separators=(",", ":"))
                + ":"
                + encode(item[key])
                for key in sorted(item)
            )
            return "{" + ",".join(members) + "}"
        raise TypeError(f"unsupported canonical manifest value {type(item).__name__}")

    return encode(value).encode("utf-8")


def _published_datapack_manifest_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_published_serialization(value)).hexdigest()


def _datapack_with_identity(
    datapack: ImccDatapack,
    *,
    model_id: str,
    coverage: str | Mapping[str, str],
    published_manifest_sha256: str | None = None,
) -> ImccDatapack:
    species_names = tuple(datapack.parent_oxides) + tuple(datapack.reactions)
    if len(set(species_names)) != len(species_names):
        raise ValueError("datapack parent and reaction names must be globally unique")
    coverage_by_species = (
        {name: coverage for name in species_names}
        if isinstance(coverage, str)
        else dict(coverage)
    )
    missing = sorted(set(species_names) - set(coverage_by_species))
    extra = sorted(set(coverage_by_species) - set(species_names))
    if missing or extra:
        raise ValueError(
            "datapack coverage must exactly match species; "
            f"missing={missing}, extra={extra}"
        )
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("datapack model_id must be a non-empty string")
    if not all(
        isinstance(label, str) and label for label in coverage_by_species.values()
    ):
        raise ValueError("datapack coverage labels must be non-empty strings")
    published_species = {
        name
        for name, label in coverage_by_species.items()
        if label == _PUBLISHED_COVERAGE
    }
    claims_published = model_id == _PUBLISHED_MODEL_ID or bool(published_species)
    if claims_published:
        if model_id not in {_PUBLISHED_MODEL_ID, _PUBLISHED_EXTENSION_MODEL_ID}:
            raise ValueError(
                "published IMCC coverage requires a recognized IMCC model identity"
            )
        if published_manifest_sha256 != _PUBLISHED_DATAPACK_SHA256:
            raise ValueError(
                "published IMCC identity requires the frozen complete canonical "
                "datapack hash"
            )
        expected_published_species = set(_PUBLISHED_PARENT_OXIDES) | set(
            datapack.reactions[:_PUBLISHED_CORE_ROWS]
        )
        if published_species != expected_published_species:
            raise ValueError(
                "A-published-imcc coverage must exactly match the frozen core"
            )
        if model_id == _PUBLISHED_MODEL_ID and (
            datapack.n_parents != len(_PUBLISHED_PARENT_OXIDES)
            or datapack.n_complexes != _PUBLISHED_CORE_ROWS
        ):
            raise ValueError(
                "plain IMCC-SF04 identity cannot include extension rows or parents"
            )

    labelled = replace(
        datapack,
        nu=_immutable_array_copy(datapack.nu),
        A=_immutable_array_copy(datapack.A),
        B=_immutable_array_copy(datapack.B),
    )
    object.__setattr__(
        labelled,
        "_identity",
        _ImccDatapackIdentity(
            model_id=model_id,
            evidence_class=_UNTRUSTED_IDENTITY_TOKEN,
            coverage=MappingProxyType(coverage_by_species),
            proven=True,
        ),
    )
    return labelled


def label_research_datapack(
    datapack: ImccDatapack,
    *,
    model_id: str,
    coverage: str | Mapping[str, str],
) -> ImccDatapack:
    """Return a non-published raw pack with explicit research provenance.

    This function refuses ``model_id == IMCC-SF04`` and any coverage
    value equal to ``A-published-imcc``. It does not hash arrays or read a
    manifest. Published labels are attached by ``_label_loaded_datapack``
    when the caller supplies ``published_manifest_sha256`` equal to
    ``_PUBLISHED_DATAPACK_SHA256`` and the species/coverage/shape checks
    in ``_datapack_with_identity`` pass; adapter ``load_datapack()`` hashes
    the JSON manifest before calling that helper. Tests in this repository
    also import the private helper directly.
    """
    coverage_values = (
        (coverage,) if isinstance(coverage, str) else tuple(coverage.values())
    )
    if model_id == _PUBLISHED_MODEL_ID or _PUBLISHED_COVERAGE in coverage_values:
        raise ValueError(
            "research datapacks cannot claim published IMCC identity or coverage"
        )
    return _datapack_with_identity(
        datapack,
        model_id=model_id,
        coverage=coverage,
    )


def _label_loaded_datapack(
    datapack: ImccDatapack,
    *,
    model_id: str,
    coverage: Mapping[str, str],
    published_manifest_sha256: str | None = None,
) -> ImccDatapack:
    """Attach caller-supplied identity to an in-memory datapack.

    When the caller claims published IMCC identity or coverage,
    ``_datapack_with_identity`` checks that the supplied
    ``published_manifest_sha256`` string equals ``_PUBLISHED_DATAPACK_SHA256``
    and that species, coverage, and shape constraints match. This helper
    does not hash the in-memory ``nu``/``A``/``B`` arrays or re-read a
    manifest file; a matching digest string is sufficient to pass the
    published-identity gate on the paths in this function. Adapter
    ``load_datapack()`` hashes the JSON manifest and then calls this
    helper.
    """
    return _datapack_with_identity(
        datapack,
        model_id=model_id,
        coverage=coverage,
        published_manifest_sha256=published_manifest_sha256,
    )


# --------------------------------------------------------------------------- #
# Result / diagnostics
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ImccConvergence:
    iterations: int
    residual_inf: float
    residual_l2: float
    total_displacement: float
    status: str = "converged"


@dataclass(frozen=True)
class ImccLabels:
    model_id: str = _UNTRUSTED_IDENTITY_TOKEN
    datapack_version: str = ""
    evidence_class: str = _UNTRUSTED_IDENTITY_TOKEN
    coverage: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ImccResult:
    temperature_K: float
    parent_oxides: tuple[str, ...]
    parent_mol: np.ndarray
    basis: float
    parent_x: np.ndarray
    parent_x_star: np.ndarray
    parent_activity: np.ndarray
    parent_gamma: np.ndarray
    complex_x: np.ndarray
    species_x: np.ndarray
    species_names: tuple[str, ...]
    D: float
    convergence: ImccConvergence
    labels: ImccLabels
    extrapolated: bool = False


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #


def _active_residual(
    y: np.ndarray,
    x_target: np.ndarray,
    nu: np.ndarray,
    lnK: np.ndarray,
    S: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Compute the parent-balance residual and its Jacobian for the active
    subset of parents and complexes.

    Derivation
    ----------
    Premise: 46 species mix ideally. Let ``x_i*`` be the mole fraction of
    unbound parent ``i`` on the total-species basis; ``x_j`` the mole fraction
    of complex ``j``. Mass action:

        x_j = K_j(T) * prod_i (x_i*)^{nu_ij},       log10 K_j = A_j + B_j / T.

    Taking natural logs,

        ln K_j = ln(10) * (A_j + B_j / T).

    Let ``n_i`` be the analytical (total) moles of parent oxide ``i`` and
    ``N_total`` the total moles of the 46 species. Then

        n_i = N_total * x_i* + N_total * sum_j nu_ij x_j.            (1)

    Define the total-species denominator

        D = 1 + sum_j (S_j - 1) x_j,   where   S_j = sum_i nu_ij.    (2)

    Summing (1) over ``i`` gives ``sum_i n_i = N_total * D``, so the analytical
    mole fraction ``x_i = n_i / sum_k n_k`` satisfies

        x_i = (x_i* + sum_j nu_ij x_j) / D.                         (3)

    Residual in log-space variables ``y_i = ln x_i*``:

        f_i(y) = x_i* + sum_j nu_ij x_j(y) - x_i * D(y) = 0.        (4)

    Unit check: all terms are dimensionless mole fractions; ``D`` is
    dimensionless.

    Sanity case: ``K_j -> 0`` for all ``j`` => ``x_j -> 0``, ``D -> 1``, and
    ``f_i = x_i* - x_i``, so ``x_i* = x_i`` and ``gamma_i = 1``.

    Jacobian
    --------
    Differentiating (4) and using ``g_j = exp(lnK_j + sum_i nu_ij y_i) = x_j``:

        J_ik = delta_ik x_i*
               + sum_j g_j * [nu_ij - x_i(S_j - 1)] * nu_kj.

    Here ``x_i`` is the analytical (total) parent fraction, not the unbound
    fraction ``x_i*``. This is the expression implemented below.

    Returns
    -------
    f: residual vector (n_active,)
    g: complex mole fractions (n_active_complexes,)
    D: total-species denominator
    J: Jacobian (n_active, n_active)
    """
    x_star = np.exp(y)
    # Mass-action complex mole fractions on the total-species basis.
    g = np.exp(lnK + nu.T @ y)
    D = 1.0 + np.sum((S - 1.0) * g)
    # Parent-balance residual (analytical vs total-species bookkeeping).
    f = x_star + nu @ g - x_target * D
    # Jacobian.
    M = nu - np.outer(x_target, S - 1.0)  # M[i, j] = nu_ij - x_i*(S_j - 1)
    J = np.diag(x_star) + M @ (g[:, None] * nu.T)
    return f, g, D, J


def _solve_active(
    x_target: np.ndarray,
    nu: np.ndarray,
    lnK: np.ndarray,
    S: np.ndarray,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, float, int, float, float, float]:
    """
    Solve the reduced log-space parent-balance system.

    Uses ``scipy.optimize.least_squares`` with the analytical Jacobian and
    explicit bounds in log-space variables ``y_i = ln x_i*``. This enforces
    strict positivity of the unbound parent fractions and is stable for
    strongly associated systems.

    Solver robustness
    -----------------
    The ideal-fraction start ``x_i* = x_i`` (i.e. ``y = ln x``) is the correct
    solution in the no-complexing limit (K -> 0).  Direct-start stalls on this
    path are diagnosed from the retained fields: final ``y``, residual norms,
    displacement, SciPy ``nfev``, and SciPy's termination message.  Those
    fields do not measure residual-surface flatness, do not retain the
    Jacobian or gradient, and do not distinguish an active bound from a
    basin.  The same residual-above-tol + y-near-bound + xtol pattern also
    occurs when the physical root lies outside the imposed ``[-200, 100]``
    box.  Sanity case established here: one-parent ``nu=0.5``,
    ``log10(K)=50`` has asymptotic root ``y = -2 ln(K) ≈ -230.26``, below
    the lower bound; the solver parked at ``y ≈ -200`` with SciPy xtol,
    ``residual_inf ≈ 3.72e6``, 200 evaluations, and ``|J| ≈ 1.86e6`` at
    that point (not a flat residual).  The rung-3 workbook regression
    (31/70 melt solves refusing) showed residual floors of 3e-3 to 1.4e-1
    with final y values parked near the bound; that pattern alone does not
    identify the cause.

    If the direct solve stalls, association-strength continuation supplies a
    physically connected start.  Multiplying every equilibrium constant by
    ``lambda`` changes the mass-action term to

        x_j(lambda) = lambda K_j prod_i (x_i*)**nu_ij.

    At ``lambda -> 0``, complexes vanish and ``y = ln(x)`` is the exact
    solution.  We choose the first log(lambda) so every complex evaluated at
    that ideal solution is at most 1e-4, then increase log(lambda) by no more
    than two decades per stage until lambda = 1.  This follows the same
    equilibrium branch instead of guessing composition-specific starts.  All
    residuals are dimensionless; lambda is dimensionless.  If continuation
    cannot reach the requested tolerance inside the original evaluation
    budget, the typed ``ImccNonconvergenceError`` remains the outcome.
    """
    n = len(x_target)
    if n == 0:
        raise ValueError("active parent set is empty")

    if max_iter <= 0:
        raise ImccNonconvergenceError(
            "IMCC-SF04 parent-balance solve did not run (max_iter <= 0)",
            {"iterations": 0, "residual_inf": float("inf"), "residual_l2": float("inf")},
        )

    # Ideal-fraction start (no-complexing limit).  Floor tiny parents so the
    # logarithm stays finite.
    y0 = np.log(np.maximum(x_target, 1.0e-12))

    # Trust-region reflective least-squares with explicit bounds keeps the
    # log-space variables away from the exponential overflow cliff while still
    # allowing very small unbound fractions (lower bound -200 -> x* ~ 1e-87).
    lb = np.full(n, -200.0)
    ub = np.full(n, 100.0)
    max_nfev = max(10, max_iter * (n + 1))

    def _attempt(
        y_init: np.ndarray,
        trial_lnK: np.ndarray,
        nfev_budget: int,
    ) -> tuple[np.ndarray, np.ndarray, float, int, float, float, float, str] | None:
        """One least-squares attempt; returns None if residuals are non-finite."""
        if nfev_budget <= 0:
            return None

        def fun(y: np.ndarray) -> np.ndarray:
            f, _g, _D, _J = _active_residual(y, x_target, nu, trial_lnK, S)
            return f

        def jac(y: np.ndarray) -> np.ndarray:
            _f, _g, _D, J = _active_residual(y, x_target, nu, trial_lnK, S)
            return J

        try:
            sol = least_squares(
                fun,
                y_init,
                jac=jac,
                bounds=(lb, ub),
                method="trf",
                ftol=tol,
                xtol=tol,
                gtol=tol,
                max_nfev=nfev_budget,
            )
        except ValueError:
            # Non-finite residuals during this attempt (e.g. extreme K values).
            return None
        y = sol.x
        f, g, D, _J = _active_residual(y, x_target, nu, trial_lnK, S)
        residual_inf = float(np.linalg.norm(f, ord=np.inf))
        residual_l2 = float(np.linalg.norm(f, ord=2))
        total_displacement = float(np.linalg.norm(y - y0))
        return (
            y,
            g,
            D,
            int(sol.nfev),
            residual_inf,
            residual_l2,
            total_displacement,
            str(sol.message),
        )

    # The direct solve remains the fast path. The final message is retained only
    # for a typed refusal, so converged returns slice it off.
    first = _attempt(y0, lnK, max_nfev)
    if first is None:
        raise ImccNonconvergenceError(
            "IMCC-SF04 parent-balance solve produced non-finite residuals",
            {
                "iterations": 0,
                "residual_inf": float("inf"),
                "residual_l2": float("inf"),
                "final_y": y0.tolist(),
                "total_displacement": 0.0,
                "continuation": [],
            },
        )

    if first[4] <= tol:
        return first[:7]

    total_nfev = first[3]
    continuation = [
        {
            "log_lambda": 0.0,
            "iterations": first[3],
            "residual_inf": first[4],
        }
    ]
    max_log_complex_at_ideal = float(np.max(lnK + nu.T @ y0))
    start_log_lambda = min(
        0.0,
        math.log(_CONTINUATION_INITIAL_COMPLEX_X) - max_log_complex_at_ideal,
    )
    stage_count = max(
        1,
        math.ceil(-start_log_lambda / _CONTINUATION_LOGK_STEP),
    )
    log_lambdas = np.linspace(start_log_lambda, 0.0, stage_count + 1)

    best = first
    y_stage = y0
    for log_lambda in log_lambdas:
        remaining_nfev = max_nfev - total_nfev
        attempt = _attempt(y_stage, lnK + log_lambda, remaining_nfev)
        if attempt is None:
            break
        total_nfev += attempt[3]
        continuation.append(
            {
                "log_lambda": float(log_lambda),
                "iterations": attempt[3],
                "residual_inf": attempt[4],
            }
        )
        if log_lambda == 0.0 and attempt[4] < best[4]:
            best = attempt
        if attempt[4] > tol:
            break
        y_stage = attempt[0]
        if log_lambda == 0.0:
            return (
                attempt[0],
                attempt[1],
                attempt[2],
                total_nfev,
                attempt[4],
                attempt[5],
                attempt[6],
            )

    # Continuation did not reach lambda=1 at tolerance. Diagnostics remain for
    # the full-strength system; intermediate homotopy residuals are a trajectory,
    # not substitutes for the requested physical solution.
    (
        y_best,
        g_best,
        D_best,
        nfev_best,
        res_inf_best,
        res_l2_best,
        disp_best,
        msg_best,
    ) = best
    diagnostics = {
        "iterations": total_nfev,
        "residual_inf": res_inf_best,
        "residual_l2": res_l2_best,
        "final_y": y_best.tolist(),
        "total_displacement": disp_best,
        "scipy_message": msg_best,
        "continuation": continuation,
    }
    raise ImccNonconvergenceError(
        "IMCC-SF04 parent-balance solve did not converge",
        diagnostics,
    )


def solve_imcc_sf04(
    parent_mol: Sequence[float],
    T_K: float,
    datapack: ImccDatapack,
    *,
    allow_extrapolation: bool = False,
    basis: float | None = None,
    extra_mol: Mapping[str, float] | None = None,
    tol: float = 1.0e-12,
    max_iter: int = 100,
) -> ImccResult:
    """
    Solve the IMCC-SF04 ideal-associated-solution equilibrium.

    Parameters
    ----------
    parent_mol:
        Mol vector over the datapack's parent oxides.
    T_K:
        Temperature in Kelvin.
    datapack:
        Mass-action data pack.
    allow_extrapolation:
        If ``True``, evaluate rows outside their declared T domains and mark
        the result as extrapolated; if ``False``, refuse.
    basis:
        Declared normalization basis. If ``None``, ``basis = sum(parent_mol)``.
    extra_mol:
        Optional mapping of additional components. Positive ``Fe2O3`` triggers
        ``ImccFerricInputUnsupportedError``; other positive components trigger
        ``ImccComponentOutsideDomainError``.
    tol:
        Positive finite infinity-norm residual convergence tolerance. SciPy
        ``least_squares`` receives this value as ``ftol``, ``xtol``, and
        ``gtol``. Non-finite or non-positive ``tol`` is refused before the
        solve: ``residual_inf <= inf`` is true for every finite residual, so
        ``tol=inf`` previously labelled an unmoved initial guess as
        ``status="converged"``.
    max_iter:
        Positive scale used inside ``_solve_active`` to build the SciPy
        ``max_nfev`` budget as ``max(10, max_iter * (n_active + 1))``. The
        solver is trust-region reflective least-squares, not Newton. The
        returned ``ImccConvergence.iterations`` field is SciPy ``nfev``,
        not a Newton step count. Because of the floor of ten,
        ``max_iter=1`` does not impose a one-evaluation ceiling on the
        paths in ``_solve_active``.

    Returns
    -------
    ImccResult
        Activities, gamma coefficients, full speciation, convergence
        diagnostics, and identity/trust labels.

    Raises
    ------
    ImccTOutsideDatapackDomainError
        ``T_K`` outside a data-pack row domain.
    ImccCompositionIncompleteError
        Missing, negative, or basis-mismatched input.
    ImccFerricInputUnsupportedError
        ``Fe2O3`` present in ``extra_mol``.
    ImccComponentOutsideDomainError
        Unexpected components in ``extra_mol`` or parent vector length mismatch.
    ImccNonconvergenceError
        Parent-balance residual did not meet ``tol`` inside the function-
        evaluation budget, the solve produced non-finite residuals, or
        ``max_iter <= 0`` so the solve did not run.
    ValueError
        ``tol`` is not a positive finite number, or ``max_iter`` is not
        finite.
    """
    # --- Input validation ----------------------------------------------------
    parent_mol = np.asarray(parent_mol, dtype=float)
    if parent_mol.ndim != 1:
        raise ImccCompositionIncompleteError(
            "parent_mol must be a 1-D array or vector"
        )
    if parent_mol.shape[0] != datapack.n_parents:
        if parent_mol.shape[0] > datapack.n_parents:
            raise ImccComponentOutsideDomainError(
                f"parent vector length {parent_mol.shape[0]} exceeds the "
                f"IMCC-SF04 parent basis of {datapack.n_parents} oxides"
            )
        raise ImccCompositionIncompleteError(
            f"parent vector length {parent_mol.shape[0]} does not match the "
            f"declared {datapack.n_parents} oxide basis"
        )

    if not np.all(np.isfinite(parent_mol)):
        raise ImccCompositionIncompleteError(
            "parent mol vector contains non-finite values"
        )

    if np.any(parent_mol < 0.0):
        raise ImccCompositionIncompleteError(
            "parent mol vector contains negative values"
        )

    total = float(parent_mol.sum())
    if total <= 0.0:
        raise ImccCompositionIncompleteError("total parent moles are zero")

    if basis is None:
        basis = total
    else:
        basis = float(basis)
        if basis <= 0.0:
            raise ImccCompositionIncompleteError("declared basis must be positive")
        if abs(total - basis) > 1.0e-6 * basis:
            raise ImccCompositionIncompleteError(
                f"parent mol sum {total:.12g} does not match declared basis "
                f"{basis:.12g} within 1e-6 relative"
            )

    T = float(T_K)
    if not math.isfinite(T) or T <= 0.0:
        raise ImccTOutsideDatapackDomainError(
            f"temperature {T_K} is not a positive finite Kelvin value"
        )

    # Extra-component screening (Tier-B/C and ferric refusal).
    if extra_mol:
        for species, mol in extra_mol.items():
            value = float(mol)
            if value == 0.0:
                continue
            if value < 0.0:
                raise ImccCompositionIncompleteError(
                    f"extra component {species} has negative moles {value}; "
                    "extra components must be non-negative"
                )
            if species == "Fe2O3":
                raise ImccFerricInputUnsupportedError(
                    "Fe2O3 input is unsupported; convert to FeO under the "
                    "caller's redox model before calling IMCC-SF04"
                )
            raise ImccComponentOutsideDomainError(
                f"component {species} is outside the IMCC-SF04 parent basis"
            )

    if not math.isfinite(float(tol)) or float(tol) <= 0.0:
        raise ValueError(
            f"tol must be a positive finite residual tolerance, got {tol!r}"
        )
    try:
        max_iter_f = float(max_iter)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"max_iter must be a finite number, got {max_iter!r}"
        ) from exc
    if not math.isfinite(max_iter_f):
        raise ValueError(
            f"max_iter must be a finite number, got {max_iter!r}"
        )

    # --- Analytical parent fractions on the declared basis -------------------
    # x_i = parent_mol_i / basis. This equals n_i / sum_k n_k only when
    # basis is exactly the parent-mol sum. The acceptance slack at
    # |sum(parent_mol) - basis| <= 1e-6 * basis is not renormalized, so
    # accepted x need not sum to 1. Unit check: moles / moles, dimensionless.
    # Sanity: parent_mol=[0.50000025, 0.50000025], basis=1.0 is inside the
    # slack (5e-7 < 1e-6) and yields sum(x)=1.0000005.
    x = parent_mol / basis

    # --- Active parent/complex subset ----------------------------------------
    # A parent with zero analytical moles cannot supply a complex.
    active_parent = x > 0.0
    # A complex is inactive on this path if any parent with nu_ij > 0 is
    # inactive. ImccDatapack refuses all-zero columns, so every complex has
    # at least one required parent; a missing required parent inactivates
    # the complex. Parents with nu_ij == 0 are not required.
    inactive_complex = np.any(
        (datapack.nu > 0.0) & (~active_parent[:, None]), axis=0
    )
    active_complex = ~inactive_complex

    # Temperature-domain check (per-row, fail-loud default). Only active
    # complexes (those that consume at least one present parent) have their
    # declared T domains honored on this path. Domain endpoints are finite
    # by ImccDatapack construction.
    extrapolated = False
    active_indices = np.flatnonzero(active_complex)
    for idx in active_indices:
        low, high = datapack.domains[idx]
        if T < low or T > high:
            if not allow_extrapolation:
                raise ImccTOutsideDatapackDomainError(
                    f"temperature {T_K} K is outside the declared domain "
                    f"[{low}, {high}] K for complex {datapack.reactions[idx]}"
                )
            extrapolated = True

    # --- Complex equilibrium constants at T (active subset only) ------------
    log10_K_active = datapack.A[active_complex] + datapack.B[active_complex] / T
    lnK_active = LOG10 * log10_K_active

    x_active = x[active_parent]
    nu_active = datapack.nu[active_parent, :][:, active_complex]
    S_active = nu_active.sum(axis=0)

    if x_active.size == 0:
        # Should not happen because total > 0.
        raise ImccCompositionIncompleteError("no positive parent oxides")

    # Solve the reduced log-space system.
    y_active, g_active, D, iterations, res_inf, res_l2, final_step = _solve_active(
        x_active, nu_active, lnK_active, S_active, tol, max_iter
    )

    # Reconstruct the full speciation vector (8 parents + 38 complexes).
    n_parents = datapack.n_parents
    n_complexes = datapack.n_complexes
    parent_x_star = np.zeros(n_parents, dtype=float)
    parent_x_star[active_parent] = np.exp(y_active)
    complex_x = np.zeros(n_complexes, dtype=float)
    complex_x[active_complex] = g_active

    # Activity = unbound parent mole fraction on the total-species basis.
    parent_activity = parent_x_star.copy()

    # gamma_i = x_i* / x_i. Different bases: x_i* is unbound parent on the
    # total-species (N_total) basis; x_i is analytical parent on the
    # declared-basis vector above (n_i / basis).
    #
    # From (2): D = 1 + sum_j (S_j - 1) x_j, and sum_i n_i = N_total * D
    # when x is n / sum(n). Then D = n_analytical / N_total.
    # Unit check: S_j and x_j are dimensionless, so D is dimensionless.
    # Sign of (D - 1) follows (S_j - 1) for each contributing complex:
    #   S_j > 1 (association, e.g. B2): D > 1, N_total shrinks vs n_analytical
    #   S_j = 1: that complex does not change D
    #   S_j < 1 (fractional stoichiometry, allowed by this datapack): D < 1,
    #            N_total grows vs n_analytical
    # For a parent with nu_ij = 0 for every active j, (3) gives gamma_i = D,
    # so gamma_i > 1 on that inert-parent path requires D > 1 (net S > 1).
    # Sanity: B2 (S=2) has gamma_A > 1; one-parent S=0.5, K=1 gives
    # D ≈ 0.691 < 1 and gamma ≈ 0.382.
    with np.errstate(divide="ignore", invalid="ignore"):
        parent_gamma = np.where(x > 0.0, parent_x_star / x, 0.0)

    species_x = np.concatenate([parent_x_star, complex_x])
    species_names = tuple(datapack.parent_oxides) + tuple(datapack.reactions)

    convergence = ImccConvergence(
        iterations=iterations,
        residual_inf=res_inf,
        residual_l2=res_l2,
        total_displacement=final_step,
        status="converged",
    )

    labels = ImccLabels(
        model_id=datapack.model_id,
        datapack_version=datapack.version,
        coverage={
            name: datapack.coverage.get(name, _UNTRUSTED_IDENTITY_TOKEN)
            for name in species_names
        },
        evidence_class=datapack.evidence_class,
    )

    return ImccResult(
        temperature_K=T,
        parent_oxides=tuple(datapack.parent_oxides),
        parent_mol=parent_mol,
        basis=basis,
        parent_x=x,
        parent_x_star=parent_x_star,
        parent_activity=parent_activity,
        parent_gamma=parent_gamma,
        complex_x=complex_x,
        species_x=species_x,
        species_names=species_names,
        D=D,
        convergence=convergence,
        labels=labels,
        extrapolated=extrapolated,
    )
