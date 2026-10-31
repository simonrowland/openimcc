"""IMCC-SF04 kernel: 8-parent ideal associated-solution melt-activity solver.

This is the independent diagnostic shadow engine described in
``docs/spec.md``. It solves the 8 coupled nonlinear parent-oxide
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
import importlib
import json
import math
from numbers import Integral
import warnings
from types import MappingProxyType
from typing import Any, Mapping, NoReturn, Sequence

import numpy as np
from scipy.optimize import least_squares

from openimcc.receipt import CalculationReceipt


LOG10 = math.log(10.0)
_R_J_MOL_K = 8.31446261815324
_CONTINUATION_LOGK_STEP = 2.0 * LOG10
_CONTINUATION_INITIAL_COMPLEX_X = 1.0e-4
_UNTRUSTED_IDENTITY_TOKEN = "internal-analytical"
_PUBLISHED_MODEL_ID = "IMCC-SF04"
_PUBLISHED_EXTENSION_MODEL_ID = "IMCC-SF04-EXT"
_EXCESS_GIBBS_MODEL_ID = "IMCC-SF04-D066-EXCESS-CANDIDATE"
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
# Integrity proof for the normalized published core, not a per-pack identity.
# Complete canonical v1.0.2 published FC87/SF04 datapack, including every
# field of every row and the top-level identity/version. Changing this digest
# is deliberate published-datapack re-versioning, never a silent row edit.
_PUBLISHED_DATAPACK_SHA256 = (
    "ae979b68295f6e653fb1c37d6b381f06cb0be8d4c5b421e673420a6e4596357c"
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


class ImccSpeciesNotFoundError(ImccComponentOutsideDomainError):
    """Raised when a result lookup names no parent oxide."""

    code = "imcc_species_not_found"


class ImccDataframeUnavailableError(ImccRefusal):
    """Raised when the optional pandas dependency is not installed."""

    code = "imcc_dataframe_unavailable"


class ImccInvalidWarmStartError(ImccRefusal):
    """Raised when a caller-provided solver seed is malformed or out of bounds."""

    code = "imcc_invalid_warm_start"


class _ImccExcessGibbsKernelRequiresAdapterError(ImccRefusal):
    """Raised when the ideal kernel entry point receives a corrected candidate."""

    code = "imcc_excess_gibbs_requires_loaded_adapter"


class ImccSpeciesPairUnstableError(ImccRefusal):
    """Raised when a species-pair candidate root is locally unstable."""

    code = "imcc_species_pair_unstable"


class ImccSpeciesPairMultipleRootsError(ImccRefusal):
    """Raised when deterministic starts find distinct species-pair roots."""

    code = "imcc_species_pair_multiple_roots"


def _warm_start_vector(y_init: Any) -> np.ndarray:
    try:
        return np.array(y_init, dtype=np.float64)
    except (TypeError, ValueError):
        raise ImccInvalidWarmStartError(
            "y_init must be a finite vector matching active parents"
        ) from None


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
    binding_digest: str | None
    proven: bool


def _untrusted_datapack_identity() -> _ImccDatapackIdentity:
    return _ImccDatapackIdentity(
        model_id=_UNTRUSTED_IDENTITY_TOKEN,
        evidence_class=_UNTRUSTED_IDENTITY_TOKEN,
        coverage=MappingProxyType({}),
        binding_digest=None,
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
    paper_domains:
        Optional per-complex paper-demonstrated temperature windows. ``None``
        means the row carries no paper window and cannot receive that flag.
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
    parent_oxides: Sequence[str] = _PUBLISHED_PARENT_OXIDES
    paper_domains: Sequence[tuple[float, float] | None] | None = None
    _identity: _ImccDatapackIdentity = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __init__(
        self,
        reactions: Sequence[str],
        nu: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        domains: Sequence[tuple[float, float]],
        version: str,
        parent_oxides: Sequence[str] = _PUBLISHED_PARENT_OXIDES,
        paper_domains: Sequence[tuple[float, float] | None] | None = None,
    ) -> None:
        if hasattr(self, "_identity"):
            raise TypeError("ImccDatapack cannot be initialized more than once")
        object.__setattr__(self, "reactions", reactions)
        object.__setattr__(self, "nu", nu)
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "B", B)
        object.__setattr__(self, "domains", domains)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "parent_oxides", parent_oxides)
        object.__setattr__(self, "paper_domains", paper_domains)
        object.__setattr__(self, "_identity", _untrusted_datapack_identity())
        self.__post_init__()

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
        if not np.all(np.isfinite(nu)):
            raise ValueError("nu must be finite")
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
        if self.paper_domains is None:
            paper_domains_out = [None] * n_complexes
        else:
            if len(self.paper_domains) != n_complexes:
                raise ValueError("paper_domains length must match n_complexes")
            paper_domains_out = []
            for idx, window in enumerate(self.paper_domains):
                if window is None:
                    paper_domains_out.append(None)
                    continue
                try:
                    low_raw, high_raw = window
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"paper_domains[{idx}] must be a (T_low, T_high) pair"
                    ) from exc
                low = float(low_raw)
                high = float(high_raw)
                if not (math.isfinite(low) and math.isfinite(high)):
                    raise ValueError(
                        f"paper_domains[{idx}] endpoints must be finite Kelvin values"
                    )
                paper_domains_out.append((low, high))
        object.__setattr__(self, "paper_domains", tuple(paper_domains_out))
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

    @property
    def binding_digest(self) -> str | None:
        """Canonical identity of this explicitly labelled datapack, if any."""
        return self._identity.binding_digest


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


def _kernel_datapack_binding_payload(
    datapack: ImccDatapack,
    *,
    model_id: str,
    coverage: Mapping[str, str],
) -> dict[str, Any]:
    # ImccDatapack stores coefficients as float64 arrays. tolist() walks them
    # in index order and yields Python binary64 floats; their shortest
    # round-trip decimal representation, consumed by the canonical encoder's
    # Decimal(str(value)) rule, preserves each nonzero coefficient across
    # platforms. The encoder normalizes both zero signs to `0`, so packs that
    # differ only by a zero's sign have the same binding digest.
    return {
        "model_id": model_id,
        "version": datapack.version,
        "parent_oxides": list(datapack.parent_oxides),
        "complex_names": list(datapack.reactions),
        "nu": datapack.nu.tolist(),
        "A": datapack.A.tolist(),
        "B": datapack.B.tolist(),
        "domains": [list(window) for window in datapack.domains],
        "paper_domains": [
            None if window is None else list(window)
            for window in datapack.paper_domains
        ],
        "coverage": dict(coverage),
    }


def _datapack_with_identity(
    datapack: ImccDatapack,
    *,
    model_id: str,
    coverage: str | Mapping[str, str],
    published_manifest_sha256: str | None = None,
    binding_digest: str | None = None,
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
    if binding_digest is None:
        binding_digest = _published_datapack_manifest_hash(
            _kernel_datapack_binding_payload(
                datapack,
                model_id=model_id,
                coverage=coverage_by_species,
            )
        )
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
            binding_digest=binding_digest,
            proven=(
                claims_published
                and published_manifest_sha256 == _PUBLISHED_DATAPACK_SHA256
            ),
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
    value equal to ``A-published-imcc``. It hashes the labelled kernel content
    to produce a binding digest; it does not read a manifest. Published labels
    are attached by ``_label_loaded_datapack``
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
    binding_digest: str | None = None,
) -> ImccDatapack:
    """Attach caller-supplied identity to an in-memory datapack.

    When the caller claims published IMCC identity or coverage,
    ``_datapack_with_identity`` checks that the supplied
    ``published_manifest_sha256`` string equals ``_PUBLISHED_DATAPACK_SHA256``
    and that species, coverage, and shape constraints match. A matching
    ``published_manifest_sha256`` is sufficient to pass the published-core
    integrity gate on the paths in this function. The separate
    ``binding_digest`` comes from the complete JSON manifest on the adapter
    load path; if omitted, the kernel content payload is hashed here.
    """
    return _datapack_with_identity(
        datapack,
        model_id=model_id,
        coverage=coverage,
        published_manifest_sha256=published_manifest_sha256,
        binding_digest=binding_digest,
    )


# --------------------------------------------------------------------------- #
# Result / diagnostics
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ImccConvergence:
    """Convergence diagnostics for an accepted parent-balance root.

    ``residual_inf`` and ``residual_l2`` are always absolute norms of the
    unscaled parent-balance residual. A relative-tolerance refinement may
    control ``|f_i / X_i|`` internally, but the public record reports ``|f|``
    for both refined and unrefined roots.
    """

    iterations: int
    residual_inf: float
    residual_l2: float
    total_displacement: float
    status: str = "converged"
    # Appended after ``status`` with defaults so existing positional and
    # keyword constructions keep their meaning: "direct" / 0 is exactly what a
    # converged solve reported before continuation was budgeted separately.
    solver_path: str = "direct"
    continuation_stages: int = 0


@dataclass(frozen=True)
class ImccLabels:
    model_id: str = _UNTRUSTED_IDENTITY_TOKEN
    datapack_version: str = ""
    evidence_class: str = _UNTRUSTED_IDENTITY_TOKEN
    coverage: Mapping[str, str] = field(default_factory=dict)
    flags: tuple[str, ...] = ()


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
    # Extensive excess contributions use joules for the caller's supplied
    # analytical amounts. The ideal kernel leaves both at exactly zero.
    excess_gibbs_J: float = 0.0
    excess_enthalpy_J: float = 0.0
    receipt: CalculationReceipt | None = None

    def _parent_index(self, name: str) -> int:
        try:
            return self.parent_oxides.index(name)
        except ValueError:
            raise ImccSpeciesNotFoundError(
                f"unknown parent oxide {name!r}; available names: "
                f"{', '.join(self.parent_oxides)}"
            ) from None

    def activity(self, name: str) -> float:
        """Return one parent-oxide activity by name."""
        return float(self.parent_activity[self._parent_index(name)])

    def gamma(self, name: str) -> float:
        """Return one parent-oxide activity coefficient by name."""
        return float(self.parent_gamma[self._parent_index(name)])

    def to_dataframe(self) -> Any:
        """Return parent-oxide activities and coefficients as a pandas frame."""
        try:
            pd = importlib.import_module("pandas")
        except ImportError as exc:
            raise ImccDataframeUnavailableError(
                "to_dataframe() requires optional pandas; install openimcc[gas]"
            ) from exc
        return pd.DataFrame(
            {
                "species": self.parent_oxides,
                "x": self.parent_x,
                "x_star": self.parent_x_star,
                "activity": self.parent_activity,
                "gamma": self.parent_gamma,
            }
        ).set_index("species")


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
    """NumPy compatibility wrapper for the shared array residual."""
    return active_residual_jacobian(y, x_target, nu, lnK, S, np)


def _species_pair_residual_jacobian(
    z: np.ndarray,
    x_target: np.ndarray,
    nu: np.ndarray,
    lnK: np.ndarray,
    S: np.ndarray,
    interaction: np.ndarray,
    temperature_K: float,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Return balances and coupled mass action for a species-pair candidate.

    Premise: ``Gex/N = gex = 1/2 x.T W x`` on the normalized total-species
    composition ``x``. Differentiation of ``N*gex`` gives
    ``mu_ex = W*x - gex`` for every species, including spectators. The
    reaction residual is ``ln(xj)-nu.T@ln(xparent)-ln(K)+dmu/(R*T)``.
    Parent balances retain the kernel's existing ``D`` normalization. Every
    energy coefficient and potential is in J/mol, so ``dmu/(R*T)`` is
    dimensionless. With ``W=0`` this branch reduces algebraically to ideal
    mass action, but callers omit the branch for a zero matrix to preserve the
    established solver bit-for-bit.

    The Jacobian uses ``d(mu_ex)/dx = W - 1 (W*x).T`` at an iterate; at a
    normalized root its restriction to ``sum(dx)=0`` is the symmetric
    constrained Gibbs Hessian.
    """
    n_parent = x_target.size
    n_complex = lnK.size
    parent_log = z[:n_parent]
    complex_log = z[n_parent:]
    parent_x = np.exp(parent_log)
    complex_x = np.exp(complex_log)
    species_x = np.concatenate((parent_x, complex_x))

    wx = interaction @ species_x
    gex = 0.5 * float(species_x @ wx)
    mu_ex = wx - gex
    reaction_operator = np.hstack((-nu.T, np.eye(n_complex)))
    reaction_mu = reaction_operator @ mu_ex
    D = 1.0 + float(np.sum((S - 1.0) * complex_x))

    balance = parent_x + nu @ complex_x - x_target * D
    mass_action = (
        complex_log - nu.T @ parent_log - lnK
        + reaction_mu / (_R_J_MOL_K * temperature_K)
    )
    residual = np.concatenate((balance, mass_action))

    balance_parent_jac = np.diag(parent_x)
    balance_complex_jac = (
        nu - x_target[:, None] * (S - 1.0)[None, :]
    ) * complex_x[None, :]
    mu_jacobian = interaction - np.ones((species_x.size, 1)) * wx[None, :]
    mu_log_jacobian = mu_jacobian * species_x[None, :]
    reaction_jacobian = np.hstack((-nu.T, np.eye(n_complex)))
    reaction_jacobian = reaction_jacobian + (
        reaction_operator @ mu_log_jacobian
    ) / (_R_J_MOL_K * temperature_K)
    jacobian = np.vstack((
        np.hstack((balance_parent_jac, balance_complex_jac)),
        reaction_jacobian,
    ))
    return residual, np.concatenate((parent_x, complex_x)), D, jacobian


def _solve_species_pair_active(
    x_target: np.ndarray,
    nu: np.ndarray,
    lnK: np.ndarray,
    S: np.ndarray,
    interaction: np.ndarray,
    temperature_K: float,
    tol: float,
    max_iter: int,
    *,
    y_init: np.ndarray | None = None,
    relative_log_x_recovery: bool = False,
) -> tuple[np.ndarray, np.ndarray, float, int, float, float, float, str, int]:
    """Solve a species-pair candidate, compare starts, and certify stability."""
    if max_iter <= 0:
        raise ImccNonconvergenceError(
            "species-pair solve did not run (max_iter <= 0)",
            {"iterations": 0, "residual_inf": float("inf"), "residual_l2": float("inf")},
        )

    n_parent = x_target.size
    n_complex = lnK.size
    n_species = n_parent + n_complex
    if interaction.shape != (n_species, n_species):
        raise ValueError("species-pair interaction matrix has the wrong shape")

    if y_init is not None:
        warm_y = _warm_start_vector(y_init)
        if warm_y.shape != x_target.shape or not np.all(np.isfinite(warm_y)):
            raise ImccInvalidWarmStartError(
                "y_init must be a finite vector matching active parents"
            )
        if np.any(warm_y < -200.0) or np.any(warm_y > 100.0):
            raise ImccInvalidWarmStartError(
                "y_init values must lie inside the solver bounds [-200, 100]"
            )
    else:
        warm_y = None

    if n_complex:
        try:
            ideal = _solve_active(
                x_target, nu, lnK, S, tol, max_iter, y_init=warm_y
            )
            ideal_y, ideal_g = ideal[0], ideal[1]
            ideal_z = np.concatenate((ideal_y, np.log(ideal_g)))
        except (ImccNonconvergenceError, ValueError):
            ideal_y = np.log(np.maximum(x_target, 1.0e-12))
            ideal_z = np.concatenate((ideal_y, lnK + nu.T @ ideal_y))
    else:
        ideal_y = np.log(x_target)
        ideal_z = ideal_y.copy()

    seeds = [ideal_z]
    if warm_y is not None:
        warm_z = (
            np.concatenate((warm_y, lnK + nu.T @ warm_y))
            if n_complex else warm_y.copy()
        )
        seeds.append(warm_z)

    # Move each species participating in a nonzero pair in both directions.
    # These deterministic starts test basin agreement without introducing a
    # random seed or claiming that finite starts prove global uniqueness.
    interacting = np.flatnonzero(np.any(interaction != 0.0, axis=0))
    for species_index in interacting:
        for displacement in (-2.0, 2.0):
            seed = ideal_z.copy()
            seed[species_index] += displacement
            seeds.append(seed)

    unique_seeds: list[np.ndarray] = []
    for seed in seeds:
        if not any(np.array_equal(seed, existing) for existing in unique_seeds):
            unique_seeds.append(seed)

    lower = np.full(n_species, -200.0)
    upper = np.full(n_species, 100.0)
    max_nfev = max(10, max_iter * (n_species + 1))
    roots: list[tuple[np.ndarray, np.ndarray, float, int, float, float, float]] = []
    residual_records: list[dict[str, float | int]] = []
    best_z: np.ndarray | None = None
    best_residual = float("inf")
    for seed_index, seed in enumerate(unique_seeds):
        if not np.all(np.isfinite(seed)) or np.any(seed < lower) or np.any(seed > upper):
            continue

        def run(
            start: np.ndarray, relative: bool = False, solve_tol: float = tol
        ) -> Any:
            def residual_jac(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                residual, _species, D, jacobian = _species_pair_residual_jacobian(
                    value, x_target, nu, lnK, S, interaction, temperature_K
                )
                if relative:
                    # Scale inventory equations by feed abundance so trace
                    # parents receive the same refinement priority as majors.
                    residual = residual.copy()
                    jacobian = jacobian.copy()
                    residual[:n_parent] /= x_target
                    jacobian[:n_parent] /= x_target[:, None]
                return residual, jacobian

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                return least_squares(
                    lambda value: residual_jac(value)[0], start,
                    jac=lambda value: residual_jac(value)[1],
                    bounds=(lower, upper), method="trf",
                    ftol=solve_tol, xtol=solve_tol, gtol=solve_tol,
                    max_nfev=max_nfev,
                )

        try:
            solved = run(seed)
        except (ValueError, FloatingPointError):
            continue

        residual, species_x, D, _jacobian = _species_pair_residual_jacobian(
            solved.x, x_target, nu, lnK, S, interaction, temperature_K
        )
        residual_inf = float(np.linalg.norm(residual, ord=np.inf))
        residual_l2 = float(np.linalg.norm(residual, ord=2))
        relative_balance = residual[:n_parent] / (D * x_target)
        relative_inf = float(np.linalg.norm(relative_balance, ord=np.inf))
        residual_records.append({
            "seed": seed_index,
            "iterations": int(solved.nfev),
            "residual_inf": residual_inf,
        })
        if residual_inf < best_residual:
            best_residual = residual_inf
            best_z = solved.x.copy()
        if (
            math.isfinite(D) and D > 0.0
            and residual_inf <= tol and relative_inf <= tol
            and np.all(np.isfinite(species_x))
        ):
            roots.append((
                solved.x.copy(), species_x, D, int(solved.nfev),
                residual_inf, residual_l2, float(np.linalg.norm(solved.x - ideal_z)),
            ))

    if relative_log_x_recovery and not roots:
        # Retry from the best absolute iterate with parent-scaled balances.
        # Least-squares then resolves trace inventories that are negligible in
        # the unscaled objective, while the acceptance checks below retain the
        # original absolute and relative requirements.
        start = best_z if best_z is not None else ideal_z
        try:
            _f0, _x0, start_D, _J0 = _species_pair_residual_jacobian(
                start, x_target, nu, lnK, S, interaction, temperature_K
            )
            relative_tol = tol * min(start_D, 1.0) if start_D > 0.0 else tol
            solved = run(start, relative=True, solve_tol=relative_tol)
            residual, species_x, D, _jacobian = _species_pair_residual_jacobian(
                solved.x, x_target, nu, lnK, S, interaction, temperature_K
            )
            residual_inf = float(np.linalg.norm(residual, ord=np.inf))
            residual_l2 = float(np.linalg.norm(residual, ord=2))
            relative_inf = float(np.linalg.norm(
                residual[:n_parent] / (D * x_target), ord=np.inf
            )) if math.isfinite(D) and D > 0.0 else float("inf")
            if (
                math.isfinite(D) and D > 0.0
                and residual_inf <= tol and relative_inf <= tol
                and np.all(np.isfinite(species_x))
            ):
                roots.append((
                    solved.x.copy(), species_x, D, int(solved.nfev),
                    residual_inf, residual_l2,
                    float(np.linalg.norm(solved.x - ideal_z)),
                ))
            if residual_inf < best_residual:
                best_residual = residual_inf
                best_z = solved.x.copy()
        except (ValueError, FloatingPointError):
            pass

    if not roots:
        best = min(residual_records, key=lambda item: float(item["residual_inf"])) if residual_records else {}
        raise ImccNonconvergenceError(
            "species-pair mass-action and parent-balance solve did not converge",
            {
                "iterations": sum(int(item["iterations"]) for item in residual_records),
                "residual_inf": float(best.get("residual_inf", float("inf"))),
                "starts": residual_records,
                "final_y": best_z[:n_parent].tolist() if best_z is not None else ideal_y.tolist(),
            },
        )

    reference_x = roots[0][1]
    distinct = [
        root for root in roots[1:]
        if float(np.max(np.abs(root[1] - reference_x))) > max(100.0 * tol, 1.0e-8)
    ]
    if distinct:
        raise ImccSpeciesPairMultipleRootsError(
            "deterministic species-pair starts converged to distinct roots; "
            "the single-liquid result is refused",
        )

    # Local stability is tested on feasible composition changes only. The
    # active conservation matrix is C=[I, nu], so Z=[-nu; I] is an exact
    # basis for null(C), the speciation changes at fixed analytical feed.
    root = roots[0]
    root_z, species_x, D, iterations, residual_inf, residual_l2, displacement = root
    if n_complex:
        nullspace = np.vstack((-nu, np.eye(n_complex)))
        wx = interaction @ species_x
        gex = 0.5 * float(species_x @ wx)
        ones = np.ones(n_species)
        ideal_projected = _R_J_MOL_K * temperature_K * (
            nullspace.T @ (nullspace / species_x[:, None])
            - np.outer(np.sum(nullspace, axis=0), np.sum(nullspace, axis=0))
        )
        excess_hessian = (
            interaction
            - np.outer(ones, wx)
            - np.outer(wx, ones)
            + 2.0 * gex * np.ones((n_species, n_species))
        )
        projected = ideal_projected + nullspace.T @ excess_hessian @ nullspace
        # Equilibrate the reaction coordinates before the eigenvalue check.
        # Trace species can make the ideal Hessian span many orders of
        # magnitude; diagonal congruence preserves inertia while preventing
        # roundoff in a well-conditioned positive direction from appearing
        # as a negative eigenvalue next to a very large one.
        diagonal = np.diag(projected)
        if np.any(diagonal <= 0.0) or not np.all(np.isfinite(diagonal)):
            min_eigenvalue = float(np.min(diagonal))
        else:
            scale = 1.0 / np.sqrt(diagonal)
            equilibrated = projected * scale[:, None] * scale[None, :]
            min_eigenvalue = float(np.linalg.eigvalsh(equilibrated)[0])
        stability_tolerance = (
            100.0 * np.finfo(float).eps * max(1, n_species)
        )
        if (
            not math.isfinite(min_eigenvalue)
            or min_eigenvalue < -stability_tolerance
        ):
            raise ImccSpeciesPairUnstableError(
                "species-pair root fails the projected local Gibbs stability "
                "check (minimum equilibrated eigenvalue "
                f"{min_eigenvalue:.12g})",
            )

    y_active = root_z[:n_parent]
    complex_x = species_x[n_parent:]
    return (
        y_active,
        complex_x,
        D,
        iterations,
        residual_inf,
        residual_l2,
        displacement,
        "species-pair-multistart",
        0,
    )


def _association_continuation_log_lambdas(
    nu: np.ndarray,
    lnK: np.ndarray,
    ideal_y: np.ndarray,
    max_logk_step: float = _CONTINUATION_LOGK_STEP,
) -> np.ndarray:
    # Premise: g_j = K_j exp(nu_j·y). Algebra: choose ln(lambda) so every
    # ideal-seed complex is <= 1e-4, then raise ln(lambda) by at most the
    # configured step per stage. Units check: lambda is dimensionless; sanity:
    # lambda -> 0 removes association and makes y=ln(x) the exact root.
    max_log_complex_at_ideal = float(np.max(lnK + nu.T @ ideal_y))
    start_log_lambda = min(
        0.0,
        math.log(_CONTINUATION_INITIAL_COMPLEX_X) - max_log_complex_at_ideal,
    )
    stage_count = max(
        1,
        math.ceil(-start_log_lambda / max_logk_step),
    )
    return np.linspace(start_log_lambda, 0.0, stage_count + 1)


def active_residual_jacobian(
    y: Any,
    x_target: Any,
    nu: Any,
    lnK: Any,
    S: Any,
    xp: Any = np,
) -> tuple[Any, Any, Any, Any]:
    """
    Pure array-namespace implementation of the active parent residual/Jacobian.

    ``xp`` is a Python Array API namespace (NumPy by default); supplying the
    namespace keeps the equations usable by optional array backends without
    importing or requiring one here.

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
    x_star = xp.exp(y)
    # Mass-action complex mole fractions on the total-species basis.
    g = xp.exp(lnK + nu.T @ y)
    D = 1.0 + xp.sum((S - 1.0) * g)
    # Parent-balance residual (analytical vs total-species bookkeeping).
    f = x_star + nu @ g - x_target * D
    # Jacobian.
    M = nu - x_target[:, None] * (S[None, :] - 1.0)
    J = xp.eye(x_star.shape[0], dtype=x_star.dtype) * x_star[:, None]
    J = J + M @ (g[:, None] * nu.T)
    return f, g, D, J


def _solve_active(
    x_target: np.ndarray,
    nu: np.ndarray,
    lnK: np.ndarray,
    S: np.ndarray,
    tol: float,
    max_iter: int,
    *,
    y_init: np.ndarray | None = None,
    relative_tolerance: bool = False,
) -> tuple[np.ndarray, np.ndarray, float, int, float, float, float, str, int]:
    """
    Solve the reduced log-space parent-balance system.

    Uses ``scipy.optimize.least_squares`` with the analytical Jacobian and
    explicit bounds in log-space variables ``y_i = ln x_i*``. This enforces
    strict positivity of the unbound parent fractions and is stable for
    strongly associated systems.

    Solver robustness
    -----------------
    ``tol`` is applied to one of two explicitly distinct norms:
    ``absolute_balance_inf = ||f||_inf`` when ``relative_tolerance`` is false,
    and ``parent_scaled_balance_inf = ||f / x_target||_inf`` when it is true.
    The root certificate is separate and checks per-parent relative inventory
    error ``f_i / (D * x_target_i)``.

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
    than two decades per stage until lambda = 1. If that path stalls above
    tolerance, it retries with one-decade stages. This follows the same
    equilibrium branch instead of guessing composition-specific starts.  All
    residuals are dimensionless; lambda is dimensionless.  Direct and
    continuation each receive the original evaluation allowance.  If
    continuation cannot reach tolerance inside its allowance, the typed
    ``ImccNonconvergenceError`` remains the outcome.
    """
    n = len(x_target)
    if n == 0:
        raise ValueError("active parent set is empty")

    if max_iter <= 0:
        raise ImccNonconvergenceError(
            "IMCC-SF04 parent-balance solve did not run (max_iter <= 0)",
            {"iterations": 0, "residual_inf": float("inf"), "residual_l2": float("inf")},
        )

    # The relative-tolerance path receives only positive active parents and
    # must seed them at their actual scale. Keep the legacy evaluate() seed
    # unchanged so its established outputs remain bit-identical.
    y0 = (
        np.log(x_target)
        if relative_tolerance
        else np.log(np.maximum(x_target, 1.0e-12))
    )
    direct_y0 = y0 if y_init is None else _warm_start_vector(y_init)
    if direct_y0.shape != y0.shape or not np.all(np.isfinite(direct_y0)):
        raise ImccInvalidWarmStartError(
            "y_init must be a finite vector matching active parents"
        )

    # Trust-region reflective least-squares with explicit bounds keeps the
    # log-space variables away from the exponential overflow cliff while still
    # allowing very small unbound fractions (lower bound -200 -> x* ~ 1e-87).
    lb = np.full(n, -200.0)
    ub = np.full(n, 100.0)
    if np.any(direct_y0 < lb) or np.any(direct_y0 > ub):
        raise ImccInvalidWarmStartError(
            "y_init values must lie inside the solver bounds [-200, 100]"
        )
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
            return f / x_target if relative_tolerance else f

        def jac(y: np.ndarray) -> np.ndarray:
            _f, _g, _D, J = _active_residual(y, x_target, nu, trial_lnK, S)
            return J / x_target[:, None] if relative_tolerance else J

        try:
            # SciPy can warn while exploring deliberately ill-conditioned
            # extrapolated systems. The residual and convergence checks below
            # decide whether the result is usable or becomes a typed refusal.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
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
        residual = f / x_target if relative_tolerance else f
        residual_inf = float(np.linalg.norm(residual, ord=np.inf))
        residual_l2 = float(np.linalg.norm(residual, ord=2))
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
    first = _attempt(direct_y0, lnK, max_nfev)
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
                "solver_path": "failed",
                "continuation_stages": 0,
            },
        )

    if first[4] > tol and not relative_tolerance:
        # A trace-parent root can be numerically close in absolute residual
        # while still failing the relative inventory certificate. If SciPy
        # stops on gtol just above the requested absolute tolerance, polish
        # that direct iterate with the existing analytic Jacobian. Keep the
        # result only when both the solver residual and inventory certificate
        # pass; otherwise retain the original result and use continuation.
        polish_y = first[0].copy()
        polish_steps = 0
        for _ in range(12):
            polish_f, polish_g, polish_D, polish_J = _active_residual(
                polish_y, x_target, nu, lnK, S
            )
            polish_residual = polish_f
            polish_residual_inf = float(np.linalg.norm(polish_residual, ord=np.inf))
            if (
                math.isfinite(polish_D)
                and polish_D > 0.0
                and polish_residual_inf <= tol
            ):
                relative_error = polish_f / (polish_D * x_target)
                if (
                    np.all(np.isfinite(relative_error))
                    and np.all(np.abs(relative_error) <= tol)
                ):
                    polish_l2 = float(np.linalg.norm(polish_residual, ord=2))
                    first = (
                        polish_y,
                        polish_g,
                        polish_D,
                        first[3] + polish_steps,
                        polish_residual_inf,
                        polish_l2,
                        float(np.linalg.norm(polish_y - y0)),
                        f"{first[7]}; analytic Newton polish",
                    )
                    break
            try:
                correction = np.linalg.solve(polish_J, polish_f)
            except np.linalg.LinAlgError:
                break
            candidate_y = polish_y - correction
            if (
                not np.all(np.isfinite(candidate_y))
                or np.any(candidate_y < lb)
                or np.any(candidate_y > ub)
            ):
                break
            polish_y = candidate_y
            polish_steps += 1

    if first[4] <= tol:
        return (*first[:7], "direct", 0)

    total_nfev = first[3]

    def _run_continuation(
        log_lambdas: np.ndarray,
    ) -> tuple[
        tuple[np.ndarray, np.ndarray, float, int, float, float, float, str] | None,
        tuple[np.ndarray, np.ndarray, float, int, float, float, float, str],
        int,
        int,
        list[dict[str, float | int]],
    ]:
        continuation_nfev = 0
        stages = 0
        records = [
            {
                "log_lambda": 0.0,
                "iterations": first[3],
                "residual_inf": first[4],
            }
        ]
        best = first
        y_stage = y0
        for log_lambda in log_lambdas:
            # Keep the direct fast path's full allowance so every previously
            # converged direct result stays bit-identical. A stalled direct solve
            # gets no more than that same allowance for continuation.
            remaining_nfev = max_nfev - continuation_nfev
            attempt = _attempt(y_stage, lnK + log_lambda, remaining_nfev)
            if attempt is None:
                break
            stages += 1
            continuation_nfev += attempt[3]
            records.append(
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
                return attempt, best, continuation_nfev, stages, records
        return None, best, continuation_nfev, stages, records

    log_lambdas = _association_continuation_log_lambdas(nu, lnK, y0)
    continued, best, continuation_nfev, continuation_stages, continuation = (
        _run_continuation(log_lambdas)
    )
    total_nfev += continuation_nfev
    if continued is None and log_lambdas[-1] == 0.0:
        # Keep ordinary roots bit-identical on the established coarse path. A
        # stalled full-strength endpoint gets one retry on a finer connected
        # path, which can resolve trace-parent roots that the coarse rung misses.
        fine_lambdas = _association_continuation_log_lambdas(
            nu, lnK, y0, _CONTINUATION_LOGK_STEP / 2.0
        )
        continued, fine_best, retry_nfev, continuation_stages, continuation = (
            _run_continuation(fine_lambdas)
        )
        total_nfev += retry_nfev
        if fine_best[4] < best[4]:
            best = fine_best

    if continued is not None:
        return (
            continued[0],
            continued[1],
            continued[2],
            total_nfev,
            continued[4],
            continued[5],
            continued[6],
            "continuation",
            continuation_stages,
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
        "solver_path": "failed",
        "continuation_stages": continuation_stages,
    }
    raise ImccNonconvergenceError(
        "IMCC-SF04 parent-balance solve did not converge "
        f"(residual_inf={res_inf_best:.6g}; {msg_best})",
        diagnostics,
    )


def _certify_active_root(
    x_target: np.ndarray,
    nu: np.ndarray,
    lnK: np.ndarray,
    S: np.ndarray,
    tol: float,
    max_iter: int,
    solution: tuple[
        np.ndarray, np.ndarray, float, int, float, float, float, str, int
    ],
    *,
    parent_names: Sequence[str] | None = None,
    species_pair_interaction: np.ndarray | None = None,
    temperature_K: float | None = None,
) -> tuple[
    tuple[np.ndarray, np.ndarray, float, int, float, float, float, str, int], bool
]:
    """Certify every active parent balance, refining an accepted root once."""
    y, g, D, *_ = solution
    names = (
        tuple(str(name) for name in parent_names)
        if parent_names is not None
        else tuple(f"active parent {index}" for index in range(len(x_target)))
    )
    if len(names) != len(x_target):
        raise ValueError("parent_names must match the active parent vector")

    # The model balance is X_i = (a_i + sum_j nu_ij*g_j) / D, with
    # a_i=exp(y_i), nu_ij>=0, g_j>=0, and D>0. Therefore the exact free-parent
    # bound is a_i <= D*X_i. Also, the relative inventory error is exactly
    # f_i/(D*X_i), where f_i is the existing parent-balance residual. Use the
    # public dimensionless solver tolerance as tol_rel; checking every positive
    # parent avoids an arbitrary trace cutoff. An absolute |f_i|<=tol only
    # implies relative error <=tol/(D*X_i), so it can allow order-one error
    # when a trace X_i is on the scale of tol. At tol=1e-12, tol_rel is about
    # 4.5e3 float64 epsilons, above summation roundoff for this fixed system.
    tol_rel = float(tol)

    def measure(
        candidate_y: np.ndarray,
        candidate_g: np.ndarray,
        candidate_D: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            activity = np.exp(candidate_y)
            bound_rhs = candidate_D * x_target
            relative_error = (
                activity + nu @ candidate_g
            ) / bound_rhs - 1.0
        return relative_error, activity, bound_rhs, activity <= bound_rhs

    def certified(
        relative_error: np.ndarray, bound_ok: np.ndarray, candidate_D: float
    ) -> bool:
        return bool(
            math.isfinite(candidate_D)
            and candidate_D > 0.0
            and np.all(np.isfinite(relative_error))
            and np.all(np.abs(relative_error) <= tol_rel)
            and np.all(bound_ok)
        )

    def refuse(
        relative_error: np.ndarray,
        activity: np.ndarray,
        bound_rhs: np.ndarray,
        bound_ok: np.ndarray,
        *,
        cause: ImccNonconvergenceError | None = None,
    ) -> NoReturn:
        bad = np.flatnonzero(
            (~np.isfinite(relative_error))
            | (np.abs(relative_error) > tol_rel)
            | (~bound_ok)
        )
        if bad.size == 0:
            bad = np.arange(len(relative_error))
        finite_error = np.where(
            np.isfinite(relative_error), np.abs(relative_error), np.inf
        )
        index = int(bad[np.argmax(finite_error[bad])])
        error = float(relative_error[index])
        bound_violation = not bool(bound_ok[index])
        message = (
            "IMCC-SF04 parent-balance certification failed for parent "
            f"{names[index]}: relative inventory error={error:+.6g} "
            f"(tol_rel={tol_rel:.6g})"
        )
        if bound_violation:
            message += (
                f"; activity {activity[index]:.6g} exceeds D*X_i="
                f"{bound_rhs[index]:.6g}"
            )
        diagnostics: dict[str, Any] = {
            "parent": names[index],
            "relative_inventory_error": error,
            "tol_rel": tol_rel,
            "activity_bound_violated": bound_violation,
            "activity": float(activity[index]),
            "activity_bound": float(bound_rhs[index]),
            "solver_path": solution[7],
        }
        if cause is not None:
            message += f"; relative-tolerance refinement failed: {cause}"
            diagnostics["refinement_diagnostics"] = cause.diagnostics
        raise ImccNonconvergenceError(message, diagnostics) from None

    relative_error, activity, bound_rhs, bound_ok = measure(y, g, D)
    if certified(relative_error, bound_ok, D):
        return solution, False

    if not math.isfinite(D) or D <= 0.0:
        refuse(relative_error, activity, bound_rhs, bound_ok)

    if species_pair_interaction is not None:
        if temperature_K is None:
            raise ValueError("temperature_K is required for species-pair certification")
        refined = _solve_species_pair_active(
            x_target,
            nu,
            lnK,
            S,
            species_pair_interaction,
            temperature_K,
            tol,
            max_iter,
            y_init=y,
            relative_log_x_recovery=True,
        )
        refined_error, refined_activity, refined_bound_rhs, refined_bound_ok = (
            measure(refined[0], refined[1], refined[2])
        )
        if not certified(refined_error, refined_bound_ok, refined[2]):
            refuse(refined_error, refined_activity, refined_bound_rhs, refined_bound_ok)
        return (
            (
                *refined[:3],
                solution[3] + refined[3],
                refined[4],
                refined[5],
                refined[6],
                refined[7],
                solution[8] + refined[8],
            ),
            True,
        )

    # _solve_active(relative_tolerance=True) scales f_i by X_i. Since the exact
    # inventory error divides once more by D, scale that solver tolerance by
    # min(D, 1): this targets tol_rel in the exact inventory ratio without
    # loosening the caller's original tolerance. D is recomputed below and the
    # exact certificate is checked again after refinement.
    relative_solve_tol = tol_rel * min(D, 1.0)
    try:
        refined = _solve_active(
            x_target,
            nu,
            lnK,
            S,
            relative_solve_tol,
            max_iter,
            y_init=y,
            relative_tolerance=True,
        )
    except ImccNonconvergenceError as exc:
        refuse(relative_error, activity, bound_rhs, bound_ok, cause=exc)

    refined_y, refined_g, refined_D = refined[:3]
    refined_error, refined_activity, refined_bound_rhs, refined_bound_ok = measure(
        refined_y, refined_g, refined_D
    )
    # For a parent with no complex inventory the exact bound is equality, but
    # exp(log(D*X_i)) can round above D*X_i by several ulps. Only when the
    # relative balance already passes, move a violating y_i to the greatest
    # float64 value whose exp is below the bound, then recompute mass action and
    # D. Recheck both conditions after each adjustment; a real residual failure
    # cannot be hidden by this equality-rounding correction.
    if np.all(np.isfinite(refined_error)) and np.all(
        np.abs(refined_error) <= tol_rel
    ):
        while np.any(~refined_bound_ok):
            violating = np.flatnonzero(~refined_bound_ok)
            lower_y = refined_y.copy()
            target_activity = np.nextafter(refined_bound_rhs[violating], 0.0)
            if np.any(target_activity <= 0.0):
                break
            with np.errstate(divide="ignore", invalid="ignore"):
                log_target_activity = np.log(target_activity)
            lower_y[violating] = np.minimum(
                lower_y[violating], log_target_activity
            )
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                above_target = np.exp(lower_y[violating]) > target_activity
            while np.any(above_target):
                indices = violating[above_target]
                lower_y[indices] = np.nextafter(lower_y[indices], -np.inf)
                with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                    above_target = np.exp(lower_y[violating]) > target_activity
            if np.any(lower_y < -200.0) or np.array_equal(lower_y, refined_y):
                break
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                _f, refined_g, refined_D, _J = _active_residual(
                    lower_y, x_target, nu, lnK, S
                )
            refined_y = lower_y
            refined_error, refined_activity, refined_bound_rhs, refined_bound_ok = (
                measure(refined_y, refined_g, refined_D)
            )
            if not (
                np.all(np.isfinite(refined_error))
                and np.all(np.abs(refined_error) <= tol_rel)
            ):
                break
    if not certified(refined_error, refined_bound_ok, refined_D):
        refuse(
            refined_error,
            refined_activity,
            refined_bound_rhs,
            refined_bound_ok,
        )

    refined_solution = (
        refined_y,
        refined_g,
        refined_D,
        solution[3] + refined[3],
        refined[4],
        refined[5],
        refined[6],
        solution[7],
        solution[8] + refined[8],
    )
    return refined_solution, True


def solve_active_warm(
    x_target: np.ndarray,
    nu: np.ndarray,
    lnK: np.ndarray,
    S: np.ndarray,
    tol: float,
    max_iter: int,
    *,
    y_init: np.ndarray | None = None,
    parent_names: Sequence[str] | None = None,
) -> tuple[
    np.ndarray, np.ndarray, float, int, float, float, float, str, int, bool
]:
    """Solve using ``parent_scaled_balance_inf = ||f / x_target||_inf``.

    The public tolerance is checked against this norm in the Newton and SciPy
    paths. The returned root then passes the separate per-parent relative
    inventory certificate ``f_i / (D * x_target_i)``.
    """
    # Active parents are positive by construction; retaining their real scale
    # avoids a 1e-12 floor for ppm and lower feed fractions.
    ideal_y = np.log(x_target)
    if y_init is None:
        y = ideal_y.copy()
    else:
        y = _warm_start_vector(y_init)
        if y.shape != ideal_y.shape or not np.all(np.isfinite(y)):
            raise ImccInvalidWarmStartError(
                "y_init must be a finite vector matching active parents"
            )
    y = np.clip(y, -200.0, 100.0)

    def _newton_attempt(
        trial_lnK: np.ndarray,
        y_init: np.ndarray,
        *,
        scale_rows: bool,
    ) -> tuple[
        tuple[np.ndarray, np.ndarray, float, int, float, float] | None,
        np.ndarray,
        int,
    ]:
        """Try one damped-Newton solve, optionally scaling balance rows."""
        y_attempt = np.clip(
            np.asarray(y_init, dtype=np.float64).copy(), -200.0, 100.0
        )
        for iteration in range(max_iter + 1):
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                f, g, D, J = _active_residual(
                    y_attempt, x_target, nu, trial_lnK, S
                )
                scaled = f / x_target
            if not (
                np.all(np.isfinite(scaled))
                and np.all(np.isfinite(g))
                and math.isfinite(float(D))
                and np.all(np.isfinite(J))
            ):
                return None, y_attempt, iteration
            residual_inf = float(np.linalg.norm(scaled, ord=np.inf))
            residual_l2 = float(np.linalg.norm(scaled, ord=2))
            if residual_inf <= tol:
                return (
                    y_attempt,
                    g,
                    float(D),
                    iteration,
                    residual_inf,
                    residual_l2,
                ), y_attempt, iteration
            if iteration == max_iter:
                return None, y_attempt, iteration
            try:
                if scale_rows:
                    step = np.linalg.solve(J / x_target[:, None], -scaled)
                else:
                    step = np.linalg.solve(J, -f)
            except np.linalg.LinAlgError:
                return None, y_attempt, iteration
            if not np.all(np.isfinite(step)):
                return None, y_attempt, iteration

            accepted = False
            alpha = 1.0
            for _ in range(24):
                candidate = y_attempt + alpha * step
                if np.all(candidate >= -200.0) and np.all(candidate <= 100.0):
                    with np.errstate(
                        over="ignore", invalid="ignore", under="ignore"
                    ):
                        trial_f, _trial_g, _trial_D, _trial_J = _active_residual(
                            candidate, x_target, nu, trial_lnK, S
                        )
                        trial_norm = float(
                            np.linalg.norm(trial_f / x_target, ord=np.inf)
                        )
                    if math.isfinite(trial_norm) and trial_norm < residual_inf:
                        y_attempt = candidate
                        accepted = True
                        break
                alpha *= 0.5
            if not accepted:
                return None, y_attempt, iteration
        return None, y_attempt, max_iter

    first, _, direct_iterations = _newton_attempt(lnK, y, scale_rows=False)
    if first is not None:
        y, g, D, newton_iterations, residual_inf, residual_l2 = first
        certified, _refined = _certify_active_root(
            x_target,
            nu,
            lnK,
            S,
            tol,
            max_iter,
            (
                y,
                g,
                D,
                newton_iterations,
                residual_inf,
                residual_l2,
                float(np.linalg.norm(y - ideal_y)),
                "newton",
                0,
            ),
            parent_names=parent_names,
        )
        return (
            *certified,
            False,
        )

    # Follow the association branch inside Newton before using SciPy. Scaling
    # each balance by its analytical amount leaves the root unchanged and puts
    # each parent row on the same relative scale in the linear solve.
    total_iterations = direct_iterations
    continuation_stages = 0
    y_stage = ideal_y.copy()
    for log_lambda in _association_continuation_log_lambdas(nu, lnK, ideal_y):
        stage, y_stage_next, stage_iterations = _newton_attempt(
            lnK + log_lambda,
            y_stage,
            scale_rows=True,
        )
        total_iterations += stage_iterations
        if stage is None:
            break
        continuation_stages += 1
        y_stage = y_stage_next
        if log_lambda == 0.0:
            y, g, D, _stage_iterations, residual_inf, residual_l2 = stage
            certified, _refined = _certify_active_root(
                x_target,
                nu,
                lnK,
                S,
                tol,
                max_iter,
                (
                    y,
                    g,
                    D,
                    total_iterations,
                    residual_inf,
                    residual_l2,
                    float(np.linalg.norm(y - ideal_y)),
                    "newton",
                    continuation_stages,
                ),
                parent_names=parent_names,
            )
            return (
                *certified,
                False,
            )

    # SciPy remains the last resort and receives the latest connected Newton
    # iterate, with each balance scaled to its analytical amount.
    fallback = _solve_active(
        x_target,
        nu,
        lnK,
        S,
        tol,
        max_iter,
        y_init=y_stage,
        relative_tolerance=True,
    )
    certified, _refined = _certify_active_root(
        x_target,
        nu,
        lnK,
        S,
        tol,
        max_iter,
        fallback,
        parent_names=parent_names,
    )
    return (*certified, True)


def _solve_imcc_sf04_with_seed(
    parent_mol: Sequence[float],
    T_K: float,
    datapack: ImccDatapack,
    *,
    allow_extrapolation: bool = False,
    basis: float | None = None,
    extra_mol: Mapping[str, float] | None = None,
    tol: float = 1.0e-12,
    max_iter: int = 100,
    y_init: np.ndarray | None = None,
    relative_recovery: bool = False,
    relative_log_x_recovery: bool = False,
    species_pair_W: np.ndarray | None = None,
) -> tuple[ImccResult, np.ndarray]:
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
    y_init:
        Optional natural-log free-parent fractions aligned with the active
        parents in datapack order. The vector must be finite, match the active
        parent count, and lie inside the solver bounds.
    relative_recovery:
        Internal continuation option to retry an absolute-tolerance stall
        with parent-scaled residuals. Public solves leave this disabled.
    relative_log_x_recovery:
        Internal last-resort option to solve once with parent-scaled residuals
        from the ideal-fraction start. The public adapter enables this only
        after its existing recovery stages fail.
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
    tuple[ImccResult, np.ndarray]
        The certified result and the pre-refinement log activities to use only
        as a continuation start. The second item is not a returned solution.

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

    # Finite entries can still overflow when summed; the next finiteness check
    # turns that expected condition into the typed refusal callers receive.
    with np.errstate(over="ignore"):
        total = float(parent_mol.sum())
    if not math.isfinite(total):
        raise ImccCompositionIncompleteError("parent mole total is not finite")
    if total <= 0.0:
        raise ImccCompositionIncompleteError("total parent moles are zero")

    if basis is None:
        basis = total
    else:
        basis = float(basis)
        # `basis <= 0.0` alone is not a guard: under IEEE 754 every ordered
        # comparison with nan is False, and inf <= 0 is False, so both pass.
        # The division `parent_mol / basis` below then yields all zeros (inf)
        # or all nan, and the call used to refuse much later as "no positive
        # parent oxides" -- a true refusal carrying a false reason, since the
        # composition was fine and the basis was the invalid input.
        if not math.isfinite(basis) or basis <= 0.0:
            raise ImccCompositionIncompleteError(
                f"declared basis must be a positive finite number, got {basis}"
            )
        if abs(total - basis) > 1.0e-6 * basis:
            raise ImccCompositionIncompleteError(
                f"parent mol sum {total:.12g} does not match declared basis "
                f"{basis:.12g} within 1e-6 relative"
            )

    try:
        T = float(T_K)
    except OverflowError as exc:
        raise ImccTOutsideDatapackDomainError(
            "temperature is too large to represent as a finite Kelvin value"
        ) from exc
    if not math.isfinite(T) or T <= 0.0:
        raise ImccTOutsideDatapackDomainError(
            f"temperature {T_K} is not a positive finite Kelvin value"
        )

    # Extra-component screening (Tier-B/C and ferric refusal).
    if extra_mol:
        for species, mol in extra_mol.items():
            value = float(mol)
            # Checked before the zero-skip and the sign test, for the same
            # IEEE 754 reason as the basis guard: nan and inf pass `< 0.0`.
            if not math.isfinite(value):
                raise ImccCompositionIncompleteError(
                    f"extra component {species} has non-finite moles {value}"
                )
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
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(
            f"max_iter must be a finite number, got {max_iter!r}"
        ) from exc
    if not math.isfinite(max_iter_f):
        raise ValueError(
            f"max_iter must be a finite number, got {max_iter!r}"
        )
    if isinstance(max_iter, bool) or not isinstance(max_iter, Integral):
        raise ValueError(
            f"max_iter must be an integer count, got {max_iter!r}"
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
    paper_window_rows: list[str] = []
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
        paper_domain = datapack.paper_domains[idx]
        if (
            paper_domain is not None
            and low <= T <= high
            and (T < paper_domain[0] or T > paper_domain[1])
        ):
            paper_window_rows.append(datapack.reactions[idx])

    flags = ()
    if paper_window_rows:
        flags = (
            "paper-demonstrated-window: "
            f"T={T:g} K is outside the paper-demonstrated domain for rows: "
            + ", ".join(paper_window_rows),
        )

    # --- Complex equilibrium constants at T (active subset only) ------------
    log10_K_active = datapack.A[active_complex] + datapack.B[active_complex] / T
    lnK_active = LOG10 * log10_K_active

    x_active = x[active_parent]
    nu_active = datapack.nu[active_parent, :][:, active_complex]
    S_active = nu_active.sum(axis=0)

    if x_active.size == 0:
        # Should not happen because total > 0.
        raise ImccCompositionIncompleteError("no positive parent oxides")

    if species_pair_W is not None:
        species_indices = np.concatenate((
            np.flatnonzero(active_parent),
            datapack.n_parents + np.flatnonzero(active_complex),
        ))
        active_interaction = species_pair_W[np.ix_(species_indices, species_indices)]
        if active_interaction.shape != (species_indices.size, species_indices.size):
            raise ValueError("species-pair interaction matrix has the wrong shape")
    else:
        active_interaction = None

    if active_interaction is not None and np.any(active_interaction != 0.0):
        solution = _solve_species_pair_active(
            x_active,
            nu_active,
            lnK_active,
            S_active,
            active_interaction,
            T,
            float(tol),
            int(max_iter),
            y_init=y_init,
            relative_log_x_recovery=(
                relative_recovery or relative_log_x_recovery
            ),
        )
        continuation_y = solution[0]
        solution, refined = _certify_active_root(
            x_active,
            nu_active,
            lnK_active,
            S_active,
            tol,
            max_iter,
            solution,
            parent_names=tuple(
                datapack.parent_oxides[index]
                for index in np.flatnonzero(active_parent)
            ),
            species_pair_interaction=active_interaction,
            temperature_K=T,
        )
    else:
        # Solve the established reduced log-space system. Keeping this block
        # unchanged preserves ideal and analytical_parent candidate outputs.
        solve_y_init = y_init
        solve_tol = tol
        relative_tolerance = False
        if relative_log_x_recovery:
            # log(x) is the exact no-complexing solution and keeps this bounded
            # final retry to one start. Cap the scaled tolerance at 1e-14, the
            # precision needed to resolve the trace-parent inventories, without
            # relaxing a stricter caller tolerance.
            solve_y_init = np.log(x_active)
            solve_tol = min(float(tol), 1.0e-14)
            relative_tolerance = True
        try:
            solution = _solve_active(
                x_active,
                nu_active,
                lnK_active,
                S_active,
                solve_tol,
                max_iter,
                y_init=solve_y_init,
                relative_tolerance=relative_tolerance,
            )
        except ImccNonconvergenceError as absolute_error:
            if relative_log_x_recovery or not relative_recovery:
                raise
            failed_y = np.asarray(
                absolute_error.diagnostics.get("final_y", ()), dtype=float
            )
            if (
                failed_y.ndim != 1
                or failed_y.size != x_active.size
                or not np.all(np.isfinite(failed_y))
            ):
                raise
            _failed_f, _failed_g, failed_D, _failed_J = _active_residual(
                failed_y, x_active, nu_active, lnK_active, S_active
            )
            if not math.isfinite(failed_D) or failed_D <= 0.0:
                raise
            # The relative solver measures f_i/X_i, while the certificate measures
            # f_i/(D*X_i). Scaling by min(D, 1) targets tol_rel without loosening
            # the public tolerance when the current iterate has D > 1.
            relative_solve_tol = float(tol) * min(failed_D, 1.0)
            try:
                solution = _solve_active(
                    x_active,
                    nu_active,
                    lnK_active,
                    S_active,
                    relative_solve_tol,
                    max_iter,
                    y_init=failed_y,
                    relative_tolerance=True,
                )
            except ImccNonconvergenceError:
                raise absolute_error from None
        if relative_log_x_recovery:
            solution = (*solution[:7], "relative-log-x", solution[8])
        # Keep the original accepted iterate available as a continuation start.
        # The result below is still certified before it leaves this function.
        continuation_y = solution[0]
        solution, refined = _certify_active_root(
            x_active,
            nu_active,
            lnK_active,
            S_active,
            tol,
            max_iter,
            solution,
            parent_names=tuple(
                datapack.parent_oxides[index]
                for index in np.flatnonzero(active_parent)
            ),
        )
        if relative_log_x_recovery:
            solution = (*solution[:7], "relative-log-x", solution[8])
    (
        y_active,
        g_active,
        D,
        iterations,
        res_inf,
        res_l2,
        final_step,
        solver_path,
        continuation_stages,
    ) = solution
    if (
        active_interaction is None or not np.any(active_interaction != 0.0)
    ) and (relative_recovery or relative_log_x_recovery or refined):
        # The absolute path already records these norms. A relative solve or
        # refinement records |f/X|, so recompute |f| only for those uncommon
        # roots and keep the default passing path at the certificate's O(P*C)
        # cost.
        final_residual, _final_g, _final_D, _final_J = _active_residual(
            y_active, x_active, nu_active, lnK_active, S_active
        )
        res_inf = float(np.linalg.norm(final_residual, ord=np.inf))
        res_l2 = float(np.linalg.norm(final_residual, ord=2))

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
        solver_path=solver_path,
        continuation_stages=continuation_stages,
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
        flags=flags,
    )

    result = ImccResult(
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
    return result, continuation_y


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
    y_init: np.ndarray | None = None,
) -> ImccResult:
    """Return the certified result of the IMCC-SF04 parent-balance solve."""
    if datapack.model_id == _EXCESS_GIBBS_MODEL_ID:
        raise _ImccExcessGibbsKernelRequiresAdapterError(
            "the public kernel solver is ideal-only for an excess-Gibbs candidate; "
            "pass the loaded wrapper returned by openimcc.model.load_datapack() "
            "to openimcc.model.evaluate()"
        )
    result, _continuation_y = _solve_imcc_sf04_with_seed(
        parent_mol,
        T_K,
        datapack,
        allow_extrapolation=allow_extrapolation,
        basis=basis,
        extra_mol=extra_mol,
        tol=tol,
        max_iter=max_iter,
        y_init=y_init,
    )
    return result
