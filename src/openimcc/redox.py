"""Closed and imposed Fe redox equilibrium on the IMCC liquid kernel."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import math
from numbers import Real
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from scipy.optimize import linprog

from openimcc.kernel import (
    ImccRefusal,
    active_residual_jacobian,
    solve_active_warm,
)
from openimcc.model import (
    ImccComponentOutsideDomainError,
    ImccMalformedDatapackError,
    ImccTOutsideDatapackDomainError,
    load_datapack,
)
from openimcc.redox_pack import RedoxPack, load_redox_pack, ln_k_prime


_DEFAULT_DATAPACK_PATH = (
    Path(__file__).parent / "data" / "packs" / "imcc-sf04-d066-v1.json"
)
_DEFAULT_REDOX_PACK_PATH = (
    Path(__file__).parent / "data" / "packs" / "openimcc-redox-v1.json"
)
_ORIGINAL_LOAD_DATAPACK = load_datapack
_ORIGINAL_LOAD_REDOX_PACK = load_redox_pack
_DEFAULT_PACK_CACHE_LOCK = RLock()
_DEFAULT_DATAPACK_CACHE: tuple[tuple[str, int, int], Any] | None = None
_DEFAULT_REDOX_PACK_CACHE: tuple[tuple[str, int, int], RedoxPack] | None = None


def _pack_file_identity(path: Path) -> tuple[str, int, int]:
    # The cache key is (resolved path, mtime_ns, size): any ordinary rewrite of
    # a pack file changes mtime_ns, so the pack is re-read and re-validated.
    # A deliberate same-size rewrite that also restores mtime_ns is not
    # detected; hashing the file on every call would cost what the cache saves.
    try:
        resolved = path.resolve(strict=True)
        stat = resolved.stat()
    except OSError as exc:
        # Keep a missing or unreadable default pack inside the refusal
        # hierarchy, as load_datapack does for an explicit path.
        raise ImccMalformedDatapackError(
            f"default datapack {path.name} is not readable: {exc}"
        ) from exc
    return str(resolved), stat.st_mtime_ns, stat.st_size


def _load_default_datapack() -> Any:
    """Reuse the immutable default melt pack until its file identity changes."""
    if load_datapack is not _ORIGINAL_LOAD_DATAPACK:
        return load_datapack()
    identity = _pack_file_identity(_DEFAULT_DATAPACK_PATH)
    global _DEFAULT_DATAPACK_CACHE
    with _DEFAULT_PACK_CACHE_LOCK:
        if _DEFAULT_DATAPACK_CACHE is None or _DEFAULT_DATAPACK_CACHE[0] != identity:
            _DEFAULT_DATAPACK_CACHE = (
                identity,
                load_datapack(_DEFAULT_DATAPACK_PATH),
            )
        return _DEFAULT_DATAPACK_CACHE[1]


def _load_default_redox_pack() -> RedoxPack:
    """Reuse the recursively read-only redox pack until its file changes."""
    if load_redox_pack is not _ORIGINAL_LOAD_REDOX_PACK:
        return load_redox_pack()
    identity = _pack_file_identity(_DEFAULT_REDOX_PACK_PATH)
    global _DEFAULT_REDOX_PACK_CACHE
    with _DEFAULT_PACK_CACHE_LOCK:
        if _DEFAULT_REDOX_PACK_CACHE is None or _DEFAULT_REDOX_PACK_CACHE[0] != identity:
            _DEFAULT_REDOX_PACK_CACHE = (
                identity,
                load_redox_pack(_DEFAULT_REDOX_PACK_PATH),
            )
        return _DEFAULT_REDOX_PACK_CACHE[1]


class RedoxInputError(ImccRefusal):
    """The supplied inventory or mode cannot define a redox state."""

    code = "redox_invalid_input"


class RedoxNoLiveCoupleError(ImccRefusal):
    """No finite oxygen-fugacity equality exists for the supplied feed."""

    code = "redox_no_live_couple"


class RedoxEndpointError(ImccRefusal):
    """A typed absence or one-sided thermodynamic endpoint."""

    code = "redox_endpoint"

    def __init__(
        self,
        message: str,
        *,
        endpoint: str,
        flags: Sequence[str],
        lambda_bound: float | None = None,
        oxygen_residual_mol: float | None = None,
    ) -> None:
        super().__init__(message)
        self.endpoint = endpoint
        self.flags = tuple(flags)
        self.lambda_bound = lambda_bound
        self.oxygen_residual_mol = oxygen_residual_mol


class RedoxNumericalError(ImccRefusal):
    """A derivative or safeguarded root could not produce a finite state."""

    code = "redox_numerical_failure"


@dataclass(frozen=True)
class RedoxResult:
    """A complete redox point with speciation, Fe ledger, and row-domain flags.

    ``extrapolated_redox_rows`` lists active redox rows evaluated outside their
    evidence intervals. ``active_row_band_intersection_K`` is the intersection
    of the active base and redox row evidence intervals.
    """

    temperature_K: float
    mode: Literal["closed", "imposed"]
    lambda_ln_f_o2: float
    lambda_sat: float
    parent_moles: Mapping[str, float]
    parent_activity: Mapping[str, float]
    species_moles: Mapping[str, float]
    oxygen_inventory_mol: float | None
    oxygen_mol: float
    oxygen_residual_mol: float | None
    oxygen_uptake_mol: float
    oxygen_capacity_mol_per_ln_f_o2: float
    metal_moles: float
    metal_buffered: bool
    endpoint_status: str
    flags: tuple[str, ...]
    last_resort_ideal_parents: tuple[str, ...]
    extrapolated_redox_rows: tuple[str, ...]
    active_row_band_intersection_K: tuple[float, float]
    solver_path: str
    fallback_fired: bool
    warm_start: tuple[float, ...]

    @property
    def speciation(self) -> Mapping[str, float]:
        """Alias for species amounts, in moles of formula units."""
        return self.species_moles

    @property
    def partition(self) -> Mapping[str, float]:
        """Return the Fe metal and dissolved-Fe ledger."""
        return MappingProxyType(
            {
                "Fe_dissolved_mol": self.parent_moles["FeO"],
                "Fe_metal_mol": self.metal_moles,
            }
        )


@dataclass(frozen=True)
class _LiquidState:
    parent_moles: np.ndarray
    basis: float
    active_parent: np.ndarray
    active_complex: np.ndarray
    complex_names: tuple[str, ...]
    complex_oxygen_counts: np.ndarray
    complex_oxygen_deltas: np.ndarray
    complex_lambda_shift: np.ndarray
    x_target: np.ndarray
    nu: np.ndarray
    lnK: np.ndarray
    S: np.ndarray
    y: np.ndarray
    g: np.ndarray
    D: float
    J: np.ndarray
    solver_path: str
    fallback_fired: bool
    iterations: int


@dataclass(frozen=True)
class _Response:
    oxygen_mol: float
    oxygen_capacity: float
    oxygen_derivative_m: float
    lambda_sat_derivative_m: float
    y_lambda: np.ndarray
    y_m: np.ndarray


def _parent_records(redox_pack: RedoxPack) -> dict[str, Mapping[str, Any]]:
    return {str(parent["element"]): parent for parent in redox_pack.parents}


def _redox_parent_names(redox_pack: RedoxPack) -> tuple[str, ...]:
    return tuple(
        str(parent["parent_oxide"]).removesuffix("(l)")
        for parent in redox_pack.parents
    )


def _redox_parent_stoichiometry(
    base_datapack: Any, redox_pack: RedoxPack
) -> np.ndarray:
    parent_names = _redox_parent_names(redox_pack)
    parent_index = {name: index for index, name in enumerate(parent_names)}
    nu = np.zeros(
        (len(parent_names), base_datapack.n_complexes), dtype=np.float64
    )
    for base_index, name in enumerate(base_datapack.parent_oxides):
        nu[parent_index[name], :] = base_datapack.nu[base_index, :]
    return nu


def _redox_complex_rows(redox_pack: RedoxPack) -> tuple[Mapping[str, Any], ...]:
    parent_names = set(_redox_parent_names(redox_pack))
    return tuple(
        row
        for row in redox_pack.rows
        if str(row["complex"]).removesuffix("(l)") not in parent_names
    )


def _redox_complex_stoichiometry(
    parent_names: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    parent_index = {name: index for index, name in enumerate(parent_names)}
    nu = np.zeros((len(parent_names), len(rows)), dtype=np.float64)
    for column, row in enumerate(rows):
        for parent, amount in row["nu"].items():
            nu[parent_index[str(parent).removesuffix("(l)")], column] = float(amount)
    return nu


def _parent_oxygen_counts(
    parent_names: Sequence[str], redox_pack: RedoxPack
) -> np.ndarray:
    by_oxide = {
        str(parent["parent_oxide"]).removesuffix("(l)"): parent
        for parent in redox_pack.parents
    }
    return np.asarray(
        [
            float(by_oxide[name]["parent_formula_atoms"].get("O", 0.0))
            for name in parent_names
        ],
        dtype=np.float64,
    )


def _complex_oxygen_data(
    base_datapack: Any,
    redox_pack: RedoxPack,
    parent_oxygen: np.ndarray,
    redox_rows: Sequence[Mapping[str, Any]],
    redox_nu: np.ndarray,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    base_oxygen = parent_oxygen @ _redox_parent_stoichiometry(
        base_datapack, redox_pack
    )
    thermo_species = redox_pack.content["thermo_species"]
    product_oxygen = np.asarray(
        [
            float(thermo_species[str(row["complex"])]["formula_atoms"].get("O", 0.0))
            for row in redox_rows
        ],
        dtype=np.float64,
    )
    consumed_oxygen = parent_oxygen @ redox_nu
    # Premise: each balanced pseudo-complex product contains the consumed
    # parent oxygen plus the row's external O2. Algebra: Delta-O = O_product
    # - sum(nu_i*O_parent_i) = -2*nu_O2 = +2*q. Units are mol O atoms per mol
    # product formula unit. Sanity: FeO + 1/4 O2 -> FeO1.5 gives +0.5 O.
    oxygen_delta = product_oxygen - consumed_oxygen
    oxygen_counts = np.concatenate((base_oxygen, product_oxygen))
    deltas = np.concatenate(
        (np.zeros(base_datapack.n_complexes, dtype=np.float64), oxygen_delta)
    )
    shifts = np.concatenate(
        (
            np.zeros(base_datapack.n_complexes, dtype=np.float64),
            np.asarray(
                [
                    -float(row["external_oxygen_stoich_product_positive"])
                    for row in redox_rows
                ],
                dtype=np.float64,
            ),
        )
    )
    names = tuple(base_datapack.reactions) + tuple(
        str(row["complex"]) for row in redox_rows
    )
    return names, oxygen_counts, deltas, shifts


def _oxygen_inventory_targets(
    parent_moles: np.ndarray,
    oxygen_parent_counts: np.ndarray,
    fe_index: int,
    oxygen_inventory: float,
    redox_pack: RedoxPack,
) -> tuple[float, float, bool]:
    parent_names = _redox_parent_names(redox_pack)
    redox_rows = _redox_complex_rows(redox_pack)
    redox_nu = _redox_complex_stoichiometry(parent_names, redox_rows)
    thermo_species = redox_pack.content["thermo_species"]
    product_oxygen = np.asarray(
        [
            float(thermo_species[str(row["complex"])]["formula_atoms"].get("O", 0.0))
            for row in redox_rows
        ],
        dtype=np.float64,
    )
    row_oxygen_delta = product_oxygen - oxygen_parent_counts @ redox_nu
    shared_positive_rows = bool(
        np.any(
            np.count_nonzero(
                (redox_nu > 0.0) & (row_oxygen_delta[np.newaxis, :] > 0.0),
                axis=1,
            )
            > 1
        )
    )

    # Premise: a row's extent is limited by each consumed parent. Algebra:
    # n_max=min_i(n_i/nu_ij), so its endpoint shift is Delta-O_j*n_max.
    # Units are mol O atoms. Sanity: FeO1.5 adds 0.5 O/Fe; TiO removes 1 O/Ti.
    # Premise: rows sharing a parent compete for the same feed. Algebra: the
    # feasible row extents satisfy nu @ n <= parent_moles and n >= 0; min/max
    # of Delta-O @ n give the endpoint shifts. Units are mol O atoms. Sanity:
    # TiO3 and TiO4 cannot each consume the full amount of one TiO2 parent.
    # A missing required parent forces that row's extent to zero, so it cannot
    # create a shared-feed constraint for this inventory.
    active_redox_rows = np.all(
        (redox_nu <= 0.0) | (parent_moles[:, np.newaxis] > 0.0), axis=0
    )
    shared_parent = bool(
        np.any(
            np.count_nonzero(redox_nu[:, active_redox_rows] > 0.0, axis=1) > 1
        )
    )
    if shared_parent:
        constraints = {
            "A_ub": redox_nu,
            "b_ub": parent_moles,
            "bounds": (0.0, None),
            "method": "highs",
        }
        minimum = linprog(row_oxygen_delta, **constraints)
        maximum = linprog(-row_oxygen_delta, **constraints)
        if not minimum.success or not maximum.success:
            raise RedoxNumericalError(
                "could not determine shared-parent oxygen inventory limits"
            )
        minimum_extents = minimum.x
        maximum_extents = maximum.x
    else:
        # With disjoint active rows, each objective is separable by row. Its
        # optimum is zero or the limiting parent ratio, selected by Delta-O.
        minimum_extents = np.zeros(len(redox_rows), dtype=np.float64)
        maximum_extents = np.zeros(len(redox_rows), dtype=np.float64)
        for column, delta in enumerate(row_oxygen_delta):
            if delta == 0.0:
                continue
            consumed = np.flatnonzero(redox_nu[:, column] > 0.0)
            if consumed.size == 0:
                continue
            max_extent = min(
                float(parent_moles[index]) / float(redox_nu[index, column])
                for index in consumed
            )
            if delta < 0.0:
                minimum_extents[column] = max_extent
            else:
                maximum_extents[column] = max_extent

    exact_parent_oxygen = sum(
        (
            Fraction(float(parent_moles[index]))
            * Fraction(float(oxygen_parent_counts[index]))
            for index in range(len(parent_moles))
        ),
        Fraction(),
    )
    exact_minimum_change = sum(
        (
            Fraction(float(delta)) * Fraction(float(extent))
            for delta, extent in zip(row_oxygen_delta, minimum_extents)
        ),
        Fraction(),
    )
    exact_maximum_change = sum(
        (
            Fraction(float(delta)) * Fraction(float(extent))
            for delta, extent in zip(row_oxygen_delta, maximum_extents)
        ),
        Fraction(),
    )
    exact_fe_consumption = sum(
        (
            Fraction(float(redox_nu[fe_index, column]))
            * Fraction(float(extent))
            for column, extent in enumerate(minimum_extents)
        ),
        Fraction(),
    )
    # Premise: minimizing extents can lock Fe in a product. Algebra:
    # O_min=O_parent-O_Fe*(n_Fe-nu_Fe@n_min)+DeltaO@n_min. Units are mol O.
    # Sanity: FeO+TiO2->FeTiO+O2 leaves 0.75 FeO and has O_min=0.25.
    exact_minimum = (
        exact_parent_oxygen
        - Fraction(float(parent_moles[fe_index]))
        * Fraction(float(oxygen_parent_counts[fe_index]))
        + Fraction(float(oxygen_parent_counts[fe_index]))
        * exact_fe_consumption
        + exact_minimum_change
    )
    exact_maximum = exact_parent_oxygen + exact_maximum_change
    exact_inventory = Fraction(float(oxygen_inventory))
    return (
        float(exact_inventory - exact_minimum),
        float(exact_maximum - exact_inventory),
        shared_positive_rows,
    )


def _inventory_parent_moles(
    inventory: Mapping[str, float],
    parent_names: Sequence[str],
    redox_pack: RedoxPack,
    *,
    require_oxygen: bool,
) -> tuple[np.ndarray, float | None]:
    if not isinstance(inventory, Mapping):
        raise RedoxInputError("inventory must map element symbols to mol atoms")
    records = _parent_records(redox_pack)
    allowed = set(records) | {"O"}
    unknown = set(inventory) - allowed
    if unknown:
        raise ImccComponentOutsideDomainError(
            f"inventory names unregistered element(s): {sorted(unknown)}"
        )
    element_moles: dict[str, float] = {}
    for element, raw in inventory.items():
        if isinstance(raw, bool) or not isinstance(raw, Real):
            raise RedoxInputError(f"inventory amount for {element!r} must be real")
        amount = float(raw)
        if not math.isfinite(amount) or amount < 0.0:
            raise RedoxInputError(
                f"inventory amount for {element!r} must be finite and non-negative"
            )
        element_moles[element] = amount
    if require_oxygen and "O" not in element_moles:
        raise RedoxInputError("closed mode requires total oxygen in mol O atoms")
    oxygen = element_moles.get("O")

    parent_index = {name: index for index, name in enumerate(parent_names)}
    parent_moles = np.zeros(len(parent_names), dtype=np.float64)
    for element, amount in element_moles.items():
        if element == "O" or amount == 0.0:
            continue
        record = records[element]
        parent_name = str(record["parent_oxide"]).removesuffix("(l)")
        if parent_name not in parent_index:
            raise ImccComponentOutsideDomainError(
                f"{element} has no populated liquid parent in this redox core"
            )
        count = float(record["parent_formula_atoms"].get(element, 0.0))
        if count <= 0.0:
            raise RedoxInputError(f"parent formula for {element} has no {element}")
        parent_moles[parent_index[parent_name]] += amount / count

    na = element_moles.get("Na", 0.0)
    potassium = element_moles.get("K", 0.0)
    fe_index = parent_index["FeO"]
    if (na > 0.0) != (potassium > 0.0) and parent_moles[fe_index] == 0.0:
        raise RedoxEndpointError(
            "the supplied alkali inventory does not contain both Na and K",
            endpoint="alkali_couple_incomplete",
            flags=("alkali_couple_incomplete",),
        )
    if not np.any(parent_moles > 0.0):
        raise RedoxNoLiveCoupleError("inventory has no supported liquid parents")
    if parent_moles[fe_index] <= 0.0:
        raise RedoxNoLiveCoupleError(
            "a finite Fe redox state requires a positive total Fe inventory"
        )
    return parent_moles, oxygen


def _active_inner(
    parent_moles: np.ndarray,
    temperature_K: float,
    lambda_value: float,
    base_datapack: Any,
    redox_pack: RedoxPack,
    *,
    exclude_fe_redox: bool = False,
    y_init: np.ndarray | None = None,
    tol: float,
    max_iter: int,
) -> _LiquidState:
    parent_names = _redox_parent_names(redox_pack)
    redox_rows = _redox_complex_rows(redox_pack)
    if exclude_fe_redox:
        fe_parent_names = {
            str(parent["parent_oxide"]).removesuffix("(l)")
            for parent in redox_pack.parents
            if str(parent["element"]) == "Fe"
        }
        redox_rows = tuple(
            row
            for row in redox_rows
            if not any(
                float(amount) > 0.0
                and str(parent).removesuffix("(l)") in fe_parent_names
                for parent, amount in row["nu"].items()
            )
        )
    redox_nu = _redox_complex_stoichiometry(parent_names, redox_rows)
    nu_full = np.column_stack(
        (_redox_parent_stoichiometry(base_datapack, redox_pack), redox_nu)
    )
    lnK_base = np.log(10.0) * (
        base_datapack.A + base_datapack.B / temperature_K
    )
    redox_lnK: list[float] = []
    active_parent_names = {
        parent_names[index]
        for index in np.flatnonzero(parent_moles > 0.0)
    }
    for row in redox_rows:
        if any(
            float(amount) > 0.0
            and str(parent).removesuffix("(l)") not in active_parent_names
            for parent, amount in row["nu"].items()
        ):
            # An inactive association has zero extent and its K is discarded
            # below; do not reject its absent reactants because of its domain.
            redox_lnK.append(0.0)
            continue
        try:
            redox_lnK.append(
                ln_k_prime(
                    redox_pack,
                    row,
                    temperature_K,
                    ln_f_o2=lambda_value,
                    allow_extrapolation=True,
                )
            )
        except ValueError as exc:
            raise ImccTOutsideDatapackDomainError(
                f"temperature {temperature_K:g} K cannot be evaluated for "
                f"{row['complex']} because species G(T) evaluation failed: {exc}"
            ) from exc
    lnK_full = np.concatenate((lnK_base, np.asarray(redox_lnK, dtype=np.float64)))
    parent_oxygen = _parent_oxygen_counts(parent_names, redox_pack)
    complex_names, complex_oxygen, complex_delta, complex_shift = (
        _complex_oxygen_data(
            base_datapack,
            redox_pack,
            parent_oxygen,
            redox_rows,
            redox_nu,
        )
    )

    basis = float(np.sum(parent_moles))
    active_parent = parent_moles > 0.0
    inactive_complex = np.any(
        (nu_full > 0.0) & (~active_parent[:, None]), axis=0
    )
    active_complex = ~inactive_complex
    active_parent_moles = parent_moles[active_parent]
    x_target = active_parent_moles / basis
    underflowed = x_target == 0.0
    if np.any(underflowed):
        # The feed amount remains active even when its normalized fraction is
        # below float64 range. Form its scale as log(n)-log(basis) for the
        # typed refusal; never pass a synthetic zero to the log-space kernel.
        active_indices = np.flatnonzero(active_parent)
        active_index = int(active_indices[int(np.flatnonzero(underflowed)[0])])
        log_fraction = math.log(float(parent_moles[active_index])) - math.log(basis)
        parent_name = parent_names[active_index]
        raise RedoxNumericalError(
            f"positive active parent {parent_name} has an unrepresentable float64 "
            f"mole fraction (ln fraction={log_fraction:g}); it remains in the "
            "active inventory but is outside the normalized solve range"
        )
    nu = nu_full[active_parent, :][:, active_complex]
    lnK = lnK_full[active_complex]
    S = np.sum(nu, axis=0)
    (
        y,
        g,
        D,
        iterations,
        _residual_inf,
        _residual_l2,
        _displacement,
        solver_path,
        _continuation_stages,
        fallback_fired,
    ) = solve_active_warm(
        x_target,
        nu,
        lnK,
        S,
        tol,
        max_iter,
        y_init=y_init,
    )
    _f, _g, _D, J = active_residual_jacobian(y, x_target, nu, lnK, S, np)
    return _LiquidState(
        parent_moles=parent_moles,
        basis=basis,
        active_parent=np.flatnonzero(active_parent),
        active_complex=np.flatnonzero(active_complex),
        complex_names=complex_names,
        complex_oxygen_counts=complex_oxygen,
        complex_oxygen_deltas=complex_delta,
        complex_lambda_shift=complex_shift,
        x_target=x_target,
        nu=nu,
        lnK=lnK,
        S=S,
        y=y,
        g=g,
        D=float(D),
        J=J,
        solver_path=solver_path,
        fallback_fired=fallback_fired,
        iterations=iterations,
    )


def _state_species(
    state: _LiquidState,
    base_datapack: Any,
    redox_pack: RedoxPack,
) -> tuple[dict[str, float], dict[str, float], float]:
    parent_names = _redox_parent_names(redox_pack)
    oxygen_parent = _parent_oxygen_counts(parent_names, redox_pack)

    total_species_moles = state.basis / state.D
    parent_amount = np.zeros(len(parent_names), dtype=np.float64)
    parent_amount[state.active_parent] = total_species_moles * np.exp(state.y)
    complex_fraction = np.zeros(len(state.complex_names), dtype=np.float64)
    complex_fraction[state.active_complex] = state.g
    complex_amount = total_species_moles * complex_fraction
    oxygen_moles = float(
        np.dot(parent_amount, oxygen_parent)
        + np.dot(complex_amount, state.complex_oxygen_counts)
    )
    parents = {
        name: float(value) for name, value in zip(parent_names, state.parent_moles)
    }
    species = {
        name: float(value)
        for name, value in zip(
            parent_names + state.complex_names,
            np.concatenate((parent_amount, complex_amount)),
        )
    }
    return parents, species, oxygen_moles


def _oxygen_residual(
    state: _LiquidState,
    species_moles: Mapping[str, float],
    redox_pack: RedoxPack,
    oxygen_inventory: float,
) -> float:
    parent_names = _redox_parent_names(redox_pack)
    oxygen_parent = _parent_oxygen_counts(parent_names, redox_pack)
    return math.fsum(
        [
            float(species_moles.get(name, 0.0)) * float(oxygen_parent[index])
            for index, name in enumerate(parent_names)
        ]
        + [
            float(species_moles.get(name, 0.0))
            * float(state.complex_oxygen_counts[index])
            for index, name in enumerate(state.complex_names)
        ]
        + [-oxygen_inventory]
    )


def _closed_oxygen_deficit_residual(
    state: _LiquidState,
    species_moles: Mapping[str, float],
    redox_pack: RedoxPack,
    target_deficit: float,
) -> float:
    parent_names = _redox_parent_names(redox_pack)
    oxygen_parent = _parent_oxygen_counts(parent_names, redox_pack)
    redox_rows = _redox_complex_rows(redox_pack)
    redox_nu = _redox_complex_stoichiometry(parent_names, redox_rows)
    thermo_species = redox_pack.content["thermo_species"]
    product_oxygen = np.asarray(
        [
            float(thermo_species[str(row["complex"])]["formula_atoms"].get("O", 0.0))
            for row in redox_rows
        ],
        dtype=np.float64,
    )
    oxygen_delta = product_oxygen - oxygen_parent @ redox_nu
    missing_oxygen = []
    for column, delta in enumerate(oxygen_delta):
        consumed = np.flatnonzero(redox_nu[:, column] > 0.0)
        if consumed.size == 0 or any(
            index not in state.active_parent for index in consumed
        ):
            continue
        row_name = str(redox_rows[column]["complex"])
        row_index = state.complex_names.index(row_name)
        local_row = np.flatnonzero(state.active_complex == row_index)
        amount = float(species_moles.get(row_name, 0.0))
        if delta > 0.0:
            other_usage: list[list[float]] = [
                [float(species_moles.get(parent_names[index], 0.0))]
                for index in state.active_parent
            ]
            for local_complex, full_complex in enumerate(state.active_complex):
                if local_row.size and full_complex == row_index:
                    continue
                complex_amount = float(
                    species_moles.get(state.complex_names[full_complex], 0.0)
                )
                for local_parent, parent_index in enumerate(state.active_parent):
                    coefficient = float(state.nu[local_parent, local_complex])
                    if coefficient > 0.0:
                        other_usage[local_parent].append(
                            coefficient * complex_amount
                        )
            available_extents = []
            for parent_index in consumed:
                local_parent = int(
                    np.flatnonzero(state.active_parent == parent_index)[0]
                )
                available_extents.append(
                    math.fsum(other_usage[local_parent])
                    / float(redox_nu[parent_index, column])
                )
            missing_oxygen.append(float(delta) * min(available_extents))
        elif delta < 0.0:
            missing_oxygen.append(-float(delta) * amount)
    # Premise: at parent balance, the unfilled extent of a positive-Delta row
    # is carried by other species consuming its parents; negative-Delta rows
    # are absent at the oxidised endpoint. Algebra: inventory-minus-O is the
    # missing Delta-O extent minus the upper-endpoint deficit. Units are mol O.
    # Sanity: FeO1.5 reduces to 0.5*FeO_equivalents - (1.5*Fe - O_inventory).
    return math.fsum((*missing_oxygen, -target_deficit))


def _response(
    state: _LiquidState,
    base_datapack: Any,
    redox_pack: RedoxPack,
) -> _Response:
    parent_names = _redox_parent_names(redox_pack)
    fe_index = parent_names.index("FeO")
    active_fe = int(np.flatnonzero(state.active_parent == fe_index)[0])
    shift = state.complex_lambda_shift[state.active_complex]
    oxygen_delta = state.complex_oxygen_deltas[state.active_complex]
    M = state.nu - np.outer(state.x_target, state.S - 1.0)
    # Premise: the converged balances satisfy F(y, x, lnK) = 0. For a
    # parameter p, J dy/dp = -F_p; the row-scaled Jacobian is the one used by
    # Newton. Each pseudo-complex has q=-nu_O2, so F_lambda=M@(g*q).
    # Units: y and lambda are dimensionless. Sanity: for FeO + 1/4 O2 =
    # FeO1.5, q=1/4 and Delta-O=+0.5; the original one-row response follows.
    f_lambda = M @ (state.g * shift)
    try:
        y_lambda = np.linalg.solve(state.J, -f_lambda)
    except np.linalg.LinAlgError as exc:
        raise RedoxNumericalError("singular IMCC Jacobian in dO/dlambda") from exc
    dg_lambda = state.g * (state.nu.T @ y_lambda + shift)
    dD_lambda = float(np.dot(state.S - 1.0, dg_lambda))
    d_complex_lambda = state.basis / state.D * (
        dg_lambda - state.g * dD_lambda / state.D
    )
    capacity = math.fsum(
        float(delta) * float(amount_prime)
        for delta, amount_prime in zip(oxygen_delta, d_complex_lambda)
    )

    parent_prime = np.zeros(len(parent_names), dtype=np.float64)
    parent_prime[fe_index] = -1.0
    basis_prime = -1.0
    parent_x = state.parent_moles[state.active_parent] / state.basis
    # From x=n/B and dB/dm=-1, dx/dm=(dn/dm+x)/B. This retains the
    # composition response when Fe leaves the liquid for the ingot.
    x_prime = (parent_prime[state.active_parent] - parent_x * basis_prime) / state.basis
    try:
        y_m = np.linalg.solve(state.J, state.D * x_prime)
    except np.linalg.LinAlgError as exc:
        raise RedoxNumericalError("singular IMCC Jacobian in dO/dm") from exc
    dg_m = state.g * (state.nu.T @ y_m)
    dD_m = float(np.dot(state.S - 1.0, dg_m))
    d_complex_m = state.basis / state.D * (
        dg_m
        + state.g * (basis_prime / state.basis - dD_m / state.D)
    )
    oxygen_parent = _parent_oxygen_counts(parent_names, redox_pack)
    oxygen_derivative_m = -float(oxygen_parent[fe_index]) + math.fsum(
        float(delta) * float(amount_prime)
        for delta, amount_prime in zip(oxygen_delta, d_complex_m)
    )

    # Fe + 1/2 O2 = FeO has ln K = ln(a_FeO) - lambda/2 at a_Fe = 1.
    # Therefore lambda_sat = 2(ln(a_FeO) - ln K); lambda is dimensionless,
    # and the derivative below is per mole Fe removed from the liquid.
    lambda_fe_derivative_m = 2.0 * y_m[active_fe]
    denominator = 1.0 - 2.0 * y_lambda[active_fe]
    if abs(denominator) <= np.finfo(np.float64).eps:
        raise RedoxNumericalError("pure-Fe saturation derivative is singular")
    lambda_prime_sat = lambda_fe_derivative_m / denominator
    oxygen_derivative_m += capacity * lambda_prime_sat
    _parents, _species, oxygen_mol = _state_species(state, base_datapack, redox_pack)
    return _Response(
        oxygen_mol=oxygen_mol,
        oxygen_capacity=float(capacity),
        oxygen_derivative_m=float(oxygen_derivative_m),
        lambda_sat_derivative_m=float(lambda_prime_sat),
        y_lambda=y_lambda,
        y_m=y_m,
    )


def _ln_k_fe(redox_pack: RedoxPack, temperature_K: float) -> float:
    try:
        return ln_k_prime(
            redox_pack,
            redox_pack.rows[0],
            temperature_K,
            allow_extrapolation=True,
        )
    except ValueError as exc:
        raise ImccTOutsideDatapackDomainError(
            f"temperature {temperature_K:g} K cannot be evaluated for the Fe "
            f"metal reference row because species G(T) evaluation failed: {exc}"
        ) from exc


def _saturation_state(
    parent_moles: np.ndarray,
    temperature_K: float,
    base_datapack: Any,
    redox_pack: RedoxPack,
    ln_k_fe: float,
    *,
    initial_lambda: float | None,
    y_init: np.ndarray | None,
    tol: float,
    max_iter: int,
) -> tuple[float, _LiquidState, int]:
    fe_index = _redox_parent_names(redox_pack).index("FeO")
    active_parent = parent_moles > 0.0
    active_parent_indices = np.flatnonzero(active_parent)
    parent_total = float(np.sum(parent_moles))
    reduced_seed = np.log(parent_moles[active_parent] / parent_total)
    try:
        # The full-state Fe activity includes ferric Fe; reset that coordinate
        # to its reduced-state ideal fraction while keeping the other warm starts.
        if y_init is not None and y_init.shape == reduced_seed.shape:
            reduced_seed = y_init.copy()
        reduced_fe_seed = int(
            np.flatnonzero(active_parent_indices == fe_index)[0]
        )
        reduced_seed[reduced_fe_seed] = math.log(
            float(parent_moles[fe_index]) / parent_total
        )
        reduced = _active_inner(
            parent_moles,
            temperature_K,
            0.0,
            base_datapack,
            redox_pack,
            exclude_fe_redox=True,
            y_init=reduced_seed,
            tol=tol,
            max_iter=max_iter,
        )
        reduced_fe = int(np.flatnonzero(reduced.active_parent == fe_index)[0])
    except (IndexError, ValueError) as exc:
        raise RedoxNumericalError(
            "ferric-free Fe saturation solve failed"
        ) from exc
    lambda_reduced = 2.0 * (float(reduced.y[reduced_fe]) - ln_k_fe)
    lower = lambda_reduced
    lower_state = _active_inner(
        parent_moles,
        temperature_K,
        lower,
        base_datapack,
        redox_pack,
        y_init=reduced.y,
        tol=tol,
        max_iter=max_iter,
    )
    lower_fe = int(np.flatnonzero(lower_state.active_parent == fe_index)[0])
    lower_s = lower - 2.0 * (float(lower_state.y[lower_fe]) - ln_k_fe)
    lower_fallbacks = int(reduced.fallback_fired) + int(lower_state.fallback_fired)
    step = 1.0
    for _ in range(64):
        if lower_s < 0.0:
            break
        lower -= step
        step *= 2.0
        lower_state = _active_inner(
            parent_moles,
            temperature_K,
            lower,
            base_datapack,
            redox_pack,
            y_init=lower_state.y,
            tol=tol,
            max_iter=max_iter,
        )
        lower_fe = int(np.flatnonzero(lower_state.active_parent == fe_index)[0])
        lower_s = lower - 2.0 * (float(lower_state.y[lower_fe]) - ln_k_fe)
        lower_fallbacks += int(lower_state.fallback_fired)
    if lower_s >= 0.0:
        raise RedoxNumericalError("could not bracket the reduced Fe saturation limit")

    # Since a_FeO <= 1, lambda <= -2 ln K for pure Fe + 1/2 O2 = FeO.
    upper = -2.0 * ln_k_fe
    upper_state = _active_inner(
        parent_moles,
        temperature_K,
        upper,
        base_datapack,
        redox_pack,
        y_init=lower_state.y,
        tol=tol,
        max_iter=max_iter,
    )
    upper_fe = int(np.flatnonzero(upper_state.active_parent == fe_index)[0])
    upper_s = upper - 2.0 * (float(upper_state.y[upper_fe]) - ln_k_fe)
    upper_fallbacks = lower_fallbacks + int(upper_state.fallback_fired)
    if upper_s < 0.0:
        raise RedoxNumericalError("FeO activity violated its thermodynamic upper bound")

    lam = (
        min(max(initial_lambda, lower), upper)
        if initial_lambda is not None
        else (lower + upper) / 2.0
    )
    state_seed = y_init if y_init is not None else lower_state.y
    for _ in range(80):
        state = _active_inner(
            parent_moles,
            temperature_K,
            lam,
            base_datapack,
            redox_pack,
            y_init=state_seed,
            tol=tol,
            max_iter=max_iter,
        )
        upper_fallbacks += int(state.fallback_fired)
        fe_local = int(np.flatnonzero(state.active_parent == fe_index)[0])
        saturation_residual = lam - 2.0 * (float(state.y[fe_local]) - ln_k_fe)
        if abs(saturation_residual) <= max(1.0e-12, tol):
            return lam, state, upper_fallbacks
        if saturation_residual < 0.0:
            lower = lam
        else:
            upper = lam
        response = _response(state, base_datapack, redox_pack)
        saturation_derivative = 1.0 - 2.0 * response.y_lambda[fe_local]
        candidate = lam - saturation_residual / saturation_derivative
        if not lower < candidate < upper:
            candidate = (lower + upper) / 2.0
        state_seed = state.y
        lam = candidate
        if upper - lower <= max(1.0e-12, tol):
            final_lam = (lower + upper) / 2.0
            final_state = _active_inner(
                parent_moles,
                temperature_K,
                final_lam,
                base_datapack,
                redox_pack,
                y_init=state_seed,
                tol=tol,
                max_iter=max_iter,
            )
            return final_lam, final_state, upper_fallbacks + int(final_state.fallback_fired)
    raise RedoxNumericalError("pure-Fe saturation solve did not converge")


def _last_resort_ideal_parents(
    base_datapack: Any,
    state: _LiquidState,
    redox_pack: RedoxPack,
) -> tuple[str, ...]:
    base_parents = set(base_datapack.parent_oxides)
    sourced_parents = {
        str(row["complex"]).removesuffix("(l)")
        for row in redox_pack.rows
        if row.get("role") == "parent_reference"
    }
    return tuple(
        str(parent["element"])
        for index, parent in enumerate(redox_pack.parents)
        if index in state.active_parent
        and str(parent["parent_oxide"]).removesuffix("(l)") not in base_parents
        and str(parent["parent_oxide"]).removesuffix("(l)") not in sourced_parents
    )


def _make_flags(
    temperature_K: float,
    base_datapack: Any,
    state: _LiquidState,
    redox_pack: RedoxPack,
    capacity: float,
    capacity_tolerance: float,
    active_redox_rows: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    flags: list[str] = ["liquid_only_model"]
    for index in state.active_complex:
        if index >= len(base_datapack.domains):
            continue
        low, high = base_datapack.domains[index]
        if temperature_K < low or temperature_K > high:
            flags.append(
                f"temperature extrapolation for {base_datapack.reactions[index]}"
            )
    unassociated = _last_resort_ideal_parents(
        base_datapack, state, redox_pack
    )
    if unassociated:
        flags.append(
            "RED: unassociated liquid parents without published IMCC rows "
            "(last_resort_ideal_parents): "
            + ", ".join(unassociated)
        )
    # Only rows that took part in the solve report their provenance range
    # flags; an inactive row (its parent absent) says nothing about this melt.
    for row in active_redox_rows:
        row_name = str(row["complex"])
        for flag in row["range_flags"]:
            flags.append(f"{row_name}: {flag}")
    extrapolated_redox_rows: list[str] = []
    for row in active_redox_rows:
        row_name = str(row["complex"])
        low, high = map(float, row["T_domain_K"])
        if temperature_K < low or temperature_K > high:
            extrapolated_redox_rows.append(row_name)
            flags.append(
                "RED: redox_row_temperature_extrapolation "
                f"row={row_name} temperature_K={temperature_K:.17g}"
            )
    if capacity <= capacity_tolerance:
        flags.append("redox_poorly_buffered")
    return tuple(dict.fromkeys(flags)), tuple(extrapolated_redox_rows)


def _assemble_result(
    *,
    mode: Literal["closed", "imposed"],
    temperature_K: float,
    lambda_value: float,
    lambda_sat: float,
    state: _LiquidState,
    metal_moles: float,
    oxygen_inventory: float | None,
    base_datapack: Any,
    redox_pack: RedoxPack,
    oxygen_parent_counts: np.ndarray,
    tolerance: float,
    fallback_fired: bool = False,
) -> RedoxResult:
    parent_values, species_values, oxygen_mol = _state_species(
        state, base_datapack, redox_pack
    )
    response = _response(state, base_datapack, redox_pack)
    oxygen_capacity = response.oxygen_capacity
    if metal_moles > 0.0:
        # Along the buffered branch, m is the independent coordinate: C is
        # (dO/dm)/(d(lambda_sat)/dm). A zero lambda slope is an invariant
        # segment, so oxygen changes at fixed lambda and its capacity is +inf.
        slope = response.lambda_sat_derivative_m
        if abs(slope) <= np.finfo(np.float64).eps:
            oxygen_capacity = (
                math.inf
                if abs(response.oxygen_derivative_m) > np.finfo(np.float64).eps
                else 0.0
            )
        else:
            oxygen_capacity = response.oxygen_derivative_m / slope
    parent_names = _redox_parent_names(redox_pack)
    fe_index = parent_names.index("FeO")
    last_resort_ideal_parents = _last_resort_ideal_parents(
        base_datapack, state, redox_pack
    )
    non_fe_oxygen = math.fsum(
        float(state.parent_moles[index]) * float(oxygen_parent_counts[index])
        for index in range(len(parent_names))
        if index != fe_index
    )
    baseline_oxygen = math.fsum(
        (
            non_fe_oxygen,
            (float(state.parent_moles[fe_index]) + metal_moles)
            * float(oxygen_parent_counts[fe_index]),
        )
    )
    residual = None
    if oxygen_inventory is not None:
        species_oxygen_terms = [
            float(species_values[name]) * float(oxygen_parent_counts[index])
            for index, name in enumerate(parent_names)
        ]
        species_oxygen_terms.extend(
            float(species_values[name]) * float(state.complex_oxygen_counts[index])
            for index, name in enumerate(state.complex_names)
        )
        residual = math.fsum((*species_oxygen_terms, -oxygen_inventory))
    oxygen_scale = oxygen_mol if oxygen_inventory is None else oxygen_inventory
    active_parent_names = {parent_names[index] for index in state.active_parent}
    active_redox_rows = []
    for row in redox_pack.rows:
        if row.get("role") == "parent_reference":
            product = str(row["complex"]).removesuffix("(l)")
            is_active = product in active_parent_names
        else:
            is_active = all(
                float(amount) <= 0.0
                or str(parent).removesuffix("(l)") in active_parent_names
                for parent, amount in row.get("nu", {}).items()
            )
        if is_active:
            active_redox_rows.append(row)
    flags, extrapolated_redox_rows = _make_flags(
        temperature_K,
        base_datapack,
        state,
        redox_pack,
        oxygen_capacity,
        tolerance * max(abs(oxygen_scale), 1.0),
        active_redox_rows,
    )
    active_log_activity = {
        parent_names[index]: float(value)
        for index, value in zip(state.active_parent, state.y)
    }
    # These are declared evidence windows, not a method-imposed solve limit.
    active_lows = [
        float(base_datapack.domains[index][0])
        for index in state.active_complex
        if index < len(base_datapack.domains)
    ]
    active_highs = [
        float(base_datapack.domains[index][1])
        for index in state.active_complex
        if index < len(base_datapack.domains)
    ]
    active_lows.extend(float(row["T_domain_K"][0]) for row in active_redox_rows)
    active_highs.extend(float(row["T_domain_K"][1]) for row in active_redox_rows)
    band_intersection = (
        max(active_lows, default=temperature_K),
        min(active_highs, default=temperature_K),
    )
    return RedoxResult(
        temperature_K=temperature_K,
        mode=mode,
        lambda_ln_f_o2=float(lambda_value),
        lambda_sat=float(lambda_sat),
        parent_moles=MappingProxyType(parent_values),
        parent_activity=MappingProxyType(
            {
                parent_names[index]: float(math.exp(value))
                for index, value in zip(state.active_parent, state.y)
            }
        ),
        species_moles=MappingProxyType(species_values),
        oxygen_inventory_mol=oxygen_inventory,
        oxygen_mol=float(oxygen_mol),
        oxygen_residual_mol=None if residual is None else float(residual),
        oxygen_uptake_mol=float(oxygen_mol - baseline_oxygen),
        oxygen_capacity_mol_per_ln_f_o2=float(oxygen_capacity),
        metal_moles=float(metal_moles),
        metal_buffered=metal_moles > 0.0,
        endpoint_status="point",
        flags=flags,
        last_resort_ideal_parents=last_resort_ideal_parents,
        extrapolated_redox_rows=extrapolated_redox_rows,
        active_row_band_intersection_K=band_intersection,
        solver_path=state.solver_path,
        fallback_fired=state.fallback_fired or fallback_fired,
        warm_start=tuple(active_log_activity.values()),
    )


def evaluate_redox(
    inventory: Mapping[str, float],
    T_K: float,
    mode: Literal["closed", "imposed"],
    lambda_imposed: float | None = None,
    warm_start: Sequence[float] | None = None,
    *,
    tol: float = 1.0e-12,
    max_iter: int = 100,
) -> RedoxResult:
    """Solve an imposed- or closed-oxygen Fe liquid equilibrium.

    ``inventory`` contains elemental moles, with oxygen counted as mol O
    atoms. Closed mode fixes all of them. Imposed mode fixes the non-oxygen
    element totals and ``lambda_imposed``, then predicts oxygen and metal.
    Input oxygen is ignored except on a genuinely invariant buffered segment,
    where it selects the metal extent. Elsewhere metal extent satisfies Fe
    saturation at the imposed lambda.
    ``lambda_imposed`` is ``ln(fO2 / 1 bar)``. Pressure is not an independent
    input once oxygen fugacity has been supplied through lambda.
    """
    try:
        temperature = float(T_K)
        tolerance = float(tol)
        iterations = int(max_iter)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RedoxInputError("temperature, tolerance, and iteration limit must be numeric") from exc
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise RedoxInputError("temperature must be finite and positive Kelvin")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise RedoxInputError("tol must be positive and finite")
    if iterations <= 0:
        raise RedoxInputError("max_iter must be positive")
    if mode not in {"closed", "imposed"}:
        raise RedoxInputError("mode must be 'closed' or 'imposed'")
    if mode == "imposed":
        if lambda_imposed is None:
            raise RedoxInputError("imposed mode requires lambda_imposed")
        try:
            lambda_value = float(lambda_imposed)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RedoxInputError("lambda_imposed must be finite") from exc
        if not math.isfinite(lambda_value):
            raise RedoxInputError("lambda_imposed must be finite")
    elif lambda_imposed is not None:
        raise RedoxInputError("closed mode determines lambda and rejects lambda_imposed")

    loaded = _load_default_datapack()
    base_datapack = loaded.kernel_datapack
    redox_pack = _load_default_redox_pack()
    parent_names = _redox_parent_names(redox_pack)
    fe_index = parent_names.index("FeO")
    oxygen_parent_counts = _parent_oxygen_counts(parent_names, redox_pack)
    parent_moles, oxygen_inventory = _inventory_parent_moles(
        inventory,
        parent_names,
        redox_pack,
        require_oxygen=mode == "closed",
    )
    if warm_start is None:
        y_seed = None
    else:
        try:
            y_seed = np.asarray(warm_start, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RedoxInputError("warm_start must contain finite numeric log activities") from exc
        if y_seed.ndim != 1:
            raise RedoxInputError("warm_start must be a one-dimensional log-activity vector")
        if not np.all(np.isfinite(y_seed)):
            raise RedoxInputError("warm_start must contain finite numeric log activities")
        if y_seed.shape != (int(np.count_nonzero(parent_moles)),):
            raise RedoxInputError(
                "warm_start length must match the positive parent set"
            )
    ln_k_fe = _ln_k_fe(redox_pack, temperature)
    total_fe = float(parent_moles[fe_index])

    if mode == "imposed":
        state = _active_inner(
            parent_moles,
            temperature,
            lambda_value,
            base_datapack,
            redox_pack,
            y_init=y_seed,
            tol=tolerance,
            max_iter=iterations,
        )
        lambda_sat, sat_state, saturation_fallbacks = _saturation_state(
            parent_moles,
            temperature,
            base_datapack,
            redox_pack,
            ln_k_fe,
            initial_lambda=None,
            y_init=state.y,
            tol=tolerance,
            max_iter=iterations,
        )
        sat_response = _response(sat_state, base_datapack, redox_pack)
        invariant_segment = (
            abs(sat_response.lambda_sat_derivative_m)
            <= np.finfo(np.float64).eps
        )
        lambda_tolerance = max(
            10.0 * tolerance, 8.0 * math.ulp(lambda_sat)
        )
        below_invariant_bound = (
            invariant_segment and lambda_value < lambda_sat - lambda_tolerance
        )
        at_invariant_bound = (
            invariant_segment
            and abs(lambda_value - lambda_sat) <= lambda_tolerance
        )
        if below_invariant_bound:
            raise RedoxEndpointError(
                "imposed oxygen fugacity is below the buffered saturation range",
                endpoint="lambda_below_dissolved_fe_saturation_range",
                flags=("lambda_below_dissolved_fe_saturation_range",),
                lambda_bound=lambda_sat,
            )
        buffered = lambda_value < lambda_sat - (
            lambda_tolerance
            if invariant_segment
            else max(1.0e-10, tolerance)
        )
        if at_invariant_bound and oxygen_inventory is None:
            raise RedoxEndpointError(
                "imposed lambda is on a buffered invariant segment; oxygen inventory is required to select metal extent",
                endpoint="imposed_invariant_requires_oxygen",
                flags=("imposed_invariant_requires_oxygen",),
                lambda_bound=lambda_sat,
            )
        if (
            invariant_segment
            and oxygen_inventory is not None
            and at_invariant_bound
        ):
            # On a buffered invariant segment lambda does not identify m;
            # the supplied oxygen inventory selects its Fe-metal extent.
            metal_low = 0.0
            metal_high = float(np.nextafter(total_fe, 0.0))

            def imposed_state_at_m(metal_amount: float) -> _LiquidState:
                amounts = parent_moles.copy()
                amounts[fe_index] = total_fe - metal_amount
                return _active_inner(
                    amounts,
                    temperature,
                    lambda_value,
                    base_datapack,
                    redox_pack,
                    y_init=state.y,
                    tol=tolerance,
                    max_iter=iterations,
                )

            low_state = imposed_state_at_m(metal_low)
            high_state = imposed_state_at_m(metal_high)
            low_species = _state_species(low_state, base_datapack, redox_pack)[1]
            high_species = _state_species(high_state, base_datapack, redox_pack)[1]
            low_residual = _oxygen_residual(
                low_state, low_species, redox_pack, oxygen_inventory
            )
            high_residual = _oxygen_residual(
                high_state, high_species, redox_pack, oxygen_inventory
            )
            if low_residual * high_residual > 0.0:
                raise RedoxEndpointError(
                    "imposed oxygen inventory is outside the buffered extent range",
                    endpoint="imposed_inventory_inconsistent",
                    flags=("imposed_inventory_inconsistent",),
                    oxygen_residual_mol=min(abs(low_residual), abs(high_residual)),
                )
            if low_residual == 0.0:
                metal, buffered_state = metal_low, low_state
            elif high_residual == 0.0:
                metal, buffered_state = metal_high, high_state
            else:
                buffered_state = low_state
                metal = 0.0
                for _ in range(64):
                    metal = metal_low + (metal_high - metal_low) / 2.0
                    if not metal_low < metal < metal_high:
                        low_error = abs(low_residual)
                        high_error = abs(high_residual)
                        if low_error <= high_error:
                            metal, buffered_state = metal_low, low_state
                        else:
                            metal, buffered_state = metal_high, high_state
                        break
                    buffered_state = imposed_state_at_m(metal)
                    buffered_species = _state_species(
                        buffered_state, base_datapack, redox_pack
                    )[1]
                    residual = _oxygen_residual(
                        buffered_state, buffered_species, redox_pack,
                        oxygen_inventory,
                    )
                    oxygen_extent_tolerance = max(
                        tolerance * abs(oxygen_inventory), math.ulp(metal)
                    )
                    if abs(residual) <= oxygen_extent_tolerance:
                        break
                    if residual > 0.0:
                        metal_low = metal
                        low_residual, low_state = residual, buffered_state
                    else:
                        metal_high = metal
                        high_residual, high_state = residual, buffered_state
                else:
                    raise RedoxNumericalError("imposed oxygen extent did not converge")
            buffered_parents = parent_moles.copy()
            buffered_parents[fe_index] = total_fe - metal
            buffered_lambda, buffered_state, extent_fallbacks = _saturation_state(
                buffered_parents,
                temperature,
                base_datapack,
                redox_pack,
                ln_k_fe,
                initial_lambda=lambda_value,
                y_init=buffered_state.y,
                tol=tolerance,
                max_iter=iterations,
            )
            if abs(buffered_lambda - lambda_value) > max(1.0e-9, 10.0 * tolerance):
                if lambda_value >= lambda_sat - max(1.0e-10, tolerance):
                    return _assemble_result(
                        mode="imposed",
                        temperature_K=temperature,
                        lambda_value=lambda_value,
                        lambda_sat=lambda_sat,
                        state=state,
                        metal_moles=0.0,
                        oxygen_inventory=oxygen_inventory,
                        base_datapack=base_datapack,
                        redox_pack=redox_pack,
                        oxygen_parent_counts=oxygen_parent_counts,
                        tolerance=tolerance,
                        fallback_fired=(
                            sat_state.fallback_fired or saturation_fallbacks > 0
                        ),
                    )
                raise RedoxEndpointError(
                    "imposed oxygen inventory is inconsistent with Fe saturation",
                    endpoint="imposed_inventory_inconsistent",
                    flags=("imposed_inventory_inconsistent",),
                    lambda_bound=buffered_lambda,
                )
            return _assemble_result(
                mode="imposed",
                temperature_K=temperature,
                lambda_value=lambda_value,
                lambda_sat=buffered_lambda,
                state=buffered_state,
                metal_moles=metal,
                oxygen_inventory=oxygen_inventory,
                base_datapack=base_datapack,
                redox_pack=redox_pack,
                oxygen_parent_counts=oxygen_parent_counts,
                tolerance=tolerance,
                fallback_fired=(
                    saturation_fallbacks > 0
                    or extent_fallbacks > 0
                    or buffered_state.fallback_fired
                ),
            )
        if buffered:
            metal_low = 0.0
            metal_high = float(np.nextafter(total_fe, 0.0))
            high_parents = parent_moles.copy()
            high_parents[fe_index] = total_fe - metal_high
            high_lambda, high_state, high_fallbacks = _saturation_state(
                high_parents,
                temperature,
                base_datapack,
                redox_pack,
                ln_k_fe,
                initial_lambda=lambda_value,
                y_init=sat_state.y,
                tol=tolerance,
                max_iter=iterations,
            )
            if high_lambda > lambda_value + max(1.0e-9, 10.0 * tolerance):
                raise RedoxEndpointError(
                    "imposed oxygen fugacity is below the buffered saturation range",
                    endpoint="lambda_below_dissolved_fe_saturation_range",
                    flags=("lambda_below_dissolved_fe_saturation_range",),
                    lambda_bound=high_lambda,
                )
            buffered_state = high_state
            metal = metal_high
            fallback_fired = sat_state.fallback_fired or high_fallbacks > 0
            for _ in range(64):
                metal = (metal_low + metal_high) / 2.0
                candidate_parents = parent_moles.copy()
                candidate_parents[fe_index] = total_fe - metal
                candidate_lambda, candidate_state, candidate_fallbacks = _saturation_state(
                    candidate_parents,
                    temperature,
                    base_datapack,
                    redox_pack,
                    ln_k_fe,
                    initial_lambda=lambda_value,
                    y_init=buffered_state.y,
                    tol=tolerance,
                    max_iter=iterations,
                )
                buffered_state = candidate_state
                fallback_fired = fallback_fired or candidate_fallbacks > 0
                residual = candidate_lambda - lambda_value
                if abs(residual) <= max(1.0e-10, tolerance):
                    break
                if residual > 0.0:
                    metal_low = metal
                else:
                    metal_high = metal
            else:
                raise RedoxNumericalError("imposed saturation extent did not converge")
            return _assemble_result(
                mode="imposed",
                temperature_K=temperature,
                lambda_value=lambda_value,
                lambda_sat=candidate_lambda,
                state=buffered_state,
                metal_moles=metal,
                oxygen_inventory=oxygen_inventory,
                base_datapack=base_datapack,
                redox_pack=redox_pack,
                oxygen_parent_counts=oxygen_parent_counts,
                tolerance=tolerance,
                fallback_fired=fallback_fired,
            )
        if lambda_value < lambda_sat - max(1.0e-10, tolerance):
            raise RedoxEndpointError(
                "imposed oxygen fugacity is below the pure-Fe saturation bound, "
                "and no oxygen inventory fixes the metal extent",
                endpoint="lambda_below_dissolved_fe_saturation_range",
                flags=("lambda_below_dissolved_fe_saturation_range",),
                lambda_bound=lambda_sat,
            )
        return _assemble_result(
            mode="imposed",
            temperature_K=temperature,
            lambda_value=lambda_value,
            lambda_sat=lambda_sat,
            state=state,
            metal_moles=0.0,
            oxygen_inventory=oxygen_inventory,
            base_datapack=base_datapack,
            redox_pack=redox_pack,
            oxygen_parent_counts=oxygen_parent_counts,
            tolerance=tolerance,
            fallback_fired=(
                sat_state.fallback_fired or saturation_fallbacks > 0
            ),
        )

    assert oxygen_inventory is not None
    target_surplus, target_deficit, shared_positive_rows = _oxygen_inventory_targets(
        parent_moles,
        oxygen_parent_counts,
        fe_index,
        oxygen_inventory,
        redox_pack,
    )
    if target_surplus < 0.0:
        raise RedoxEndpointError(
            "oxygen inventory is below the fully reduced liquid limit",
            endpoint="reductant_exceeds_reducible_oxygen",
            flags=("reductant_exceeds_reducible_oxygen",),
            oxygen_residual_mol=-target_surplus,
        )
    if target_surplus == 0.0:
        raise RedoxNoLiveCoupleError(
            "the oxygen inventory is at the fully reduced limit; no finite Fe redox couple remains"
        )

    # Each endpoint starts from parent formula oxygen. Shared-parent rows use
    # the joint extent constraints above; disjoint rows can each reach their
    # individual limiting extent. The reduced endpoint removes only unreacted
    # Fe parent oxygen after those minimizing extents.
    if target_deficit <= 0.0:
        excess = -target_deficit
        raise RedoxEndpointError(
            "oxygen inventory is at or above the fully oxidised liquid limit",
            endpoint="fully_oxidised_inventory_limit",
            flags=("fully_oxidised_inventory_limit",),
            oxygen_residual_mol=excess,
        )
    endpoint_distance = min(
        target_surplus,
        target_deficit,
    )
    oxygen_tolerance = tolerance * endpoint_distance
    # Premise: the reported float64 ledger cannot resolve below its rounding
    # scale. Algebra: accept max(solver tolerance, 16 ULPs of O), in mol O.
    # Sanity: this floor matters only when the endpoint distance is itself
    # below float64 resolution; the ordinary closed-root bar is unchanged.
    ledger_tolerance = max(
        oxygen_tolerance,
        16.0 * math.ulp(max(abs(oxygen_inventory), 1.0)),
    )
    root_tolerance = (
        min(tolerance, 1.0e-14)
        if endpoint_distance <= 1.0e-8 * max(abs(oxygen_inventory), 1.0)
        else tolerance
    )

    oxidised_state = _active_inner(
        parent_moles,
        temperature,
        0.0,
        base_datapack,
        redox_pack,
        y_init=y_seed,
        tol=tolerance,
        max_iter=iterations,
    )
    oxidised_species = _state_species(
        oxidised_state, base_datapack, redox_pack
    )[1]
    if shared_positive_rows:
        # Independent row deficits double-count capacity when same-sign rows
        # compete for a feed parent; use the reported speciated oxygen ledger.
        oxidised_residual = -_oxygen_residual(
            oxidised_state, oxidised_species, redox_pack, oxygen_inventory
        )
    else:
        oxidised_residual = _closed_oxygen_deficit_residual(
            oxidised_state, oxidised_species, redox_pack, target_deficit
        )
    oxidised_acceptance_tolerance = (
        ledger_tolerance if shared_positive_rows else oxygen_tolerance
    )
    if abs(oxidised_residual) <= oxidised_acceptance_tolerance:
        oxidised_ledger_residual = _oxygen_residual(
            oxidised_state, oxidised_species, redox_pack, oxygen_inventory
        )
        if abs(oxidised_ledger_residual) <= ledger_tolerance:
            lambda_sat, sat_state, saturation_fallbacks = _saturation_state(
                parent_moles,
                temperature,
                base_datapack,
                redox_pack,
                ln_k_fe,
                initial_lambda=None,
                y_init=oxidised_state.y,
                tol=tolerance,
                max_iter=iterations,
            )
            return _assemble_result(
                mode="closed",
                temperature_K=temperature,
                lambda_value=0.0,
                lambda_sat=lambda_sat,
                state=oxidised_state,
                metal_moles=0.0,
                oxygen_inventory=oxygen_inventory,
                base_datapack=base_datapack,
                redox_pack=redox_pack,
                oxygen_parent_counts=oxygen_parent_counts,
                tolerance=tolerance,
                fallback_fired=(
                    oxidised_state.fallback_fired
                    or sat_state.fallback_fired
                    or saturation_fallbacks > 0
                ),
            )

    lambda_sat_zero, sat_state, saturation_fallbacks = _saturation_state(
        parent_moles,
        temperature,
        base_datapack,
        redox_pack,
        ln_k_fe,
        initial_lambda=None,
        y_init=oxidised_state.y,
        tol=tolerance,
        max_iter=iterations,
    )
    sat_species = _state_species(sat_state, base_datapack, redox_pack)[1]
    sat_residual = _oxygen_residual(
        sat_state, sat_species, redox_pack, oxygen_inventory
    )
    if sat_residual <= 0.0:
        fallback_fired = (
            oxidised_state.fallback_fired
            or sat_state.fallback_fired
            or saturation_fallbacks > 0
        )
        lower = lambda_sat_zero
        upper = 0.0
        upper_state = oxidised_state
        upper_residual = oxidised_residual
        step = 1.0
        for _ in range(64):
            if upper_residual <= 0.0:
                break
            upper += step
            step *= 2.0
            upper_state = _active_inner(
                parent_moles,
                temperature,
                upper,
                base_datapack,
                redox_pack,
                y_init=upper_state.y,
                tol=root_tolerance,
                max_iter=iterations,
            )
            fallback_fired = fallback_fired or upper_state.fallback_fired
            upper_species = _state_species(
                upper_state, base_datapack, redox_pack
            )[1]
            if shared_positive_rows:
                upper_residual = -_oxygen_residual(
                    upper_state, upper_species, redox_pack, oxygen_inventory
                )
            else:
                upper_residual = _closed_oxygen_deficit_residual(
                    upper_state, upper_species, redox_pack, target_deficit
                )
        if upper_residual > 0.0:
            raise RedoxNumericalError("could not bracket the fully oxidised inventory limit")
        lower_residual = -sat_residual
        lam = lower + (upper - lower) * (
            lower_residual / (lower_residual - upper_residual)
        )
        state_seed = sat_state.y
        for inner_solves in range(1, 81):
            state = _active_inner(
                parent_moles,
                temperature,
                lam,
                base_datapack,
                redox_pack,
                y_init=state_seed,
                tol=root_tolerance,
                max_iter=iterations,
            )
            fallback_fired = fallback_fired or state.fallback_fired
            state_species = _state_species(state, base_datapack, redox_pack)[1]
            if shared_positive_rows:
                residual = -_oxygen_residual(
                    state, state_species, redox_pack, oxygen_inventory
                )
            else:
                residual = _closed_oxygen_deficit_residual(
                    state, state_species, redox_pack, target_deficit
                )
            residual_tolerance = (
                ledger_tolerance if shared_positive_rows else oxygen_tolerance
            )
            if abs(residual) <= residual_tolerance:
                ledger_residual = _oxygen_residual(
                    state, state_species, redox_pack, oxygen_inventory
                )
                if abs(ledger_residual) > ledger_tolerance:
                    state = _active_inner(
                        parent_moles,
                        temperature,
                        lam,
                        base_datapack,
                        redox_pack,
                        tol=min(root_tolerance, tolerance * 0.01),
                        max_iter=iterations,
                    )
                    fallback_fired = fallback_fired or state.fallback_fired
                    state_species = _state_species(
                        state, base_datapack, redox_pack
                    )[1]
                    ledger_residual = _oxygen_residual(
                        state, state_species, redox_pack, oxygen_inventory
                    )
                if abs(ledger_residual) <= ledger_tolerance:
                    return _assemble_result(
                        mode="closed",
                        temperature_K=temperature,
                        lambda_value=lam,
                        lambda_sat=lambda_sat_zero,
                        state=state,
                        metal_moles=0.0,
                        oxygen_inventory=oxygen_inventory,
                        base_datapack=base_datapack,
                        redox_pack=redox_pack,
                        oxygen_parent_counts=oxygen_parent_counts,
                        tolerance=tolerance,
                        fallback_fired=fallback_fired,
                    )
                response = _response(state, base_datapack, redox_pack)
                candidate = (
                    lam - ledger_residual / response.oxygen_capacity
                    if response.oxygen_capacity > 0.0
                    and math.isfinite(response.oxygen_capacity)
                    else math.nan
                )
                if not lower < candidate < upper:
                    candidate = (lower + upper) / 2.0
                state_seed = state.y
                lam = candidate
                continue
            response = _response(state, base_datapack, redox_pack)
            if residual > 0.0:
                lower = lam
                lower_residual = residual
            else:
                upper = lam
                upper_residual = residual
            candidate = (
                lam + residual / response.oxygen_capacity
                if response.oxygen_capacity > 0.0
                and math.isfinite(response.oxygen_capacity)
                else math.nan
            )
            if not lower < candidate < upper:
                candidate = (lower + upper) / 2.0
            state_seed = state.y
            lam = candidate
        raise RedoxNumericalError("closed oxygen root did not converge")

    # Saturated branch: solve O(lambda_sat(m), m) = O_inventory.  At fixed m,
    # implicit differentiation gives dy/dlambda and dy/dm from the same inner
    # Jacobian; the saturation equation then supplies d(lambda_sat)/dm.
    metal_low = 0.0
    residual_low = sat_residual
    metal_high = float(np.nextafter(total_fe, 0.0))
    parent_moles[fe_index] = total_fe - metal_high
    (
        _lambda_sat_high,
        high_state,
        high_saturation_fallbacks,
    ) = _saturation_state(
        parent_moles,
        temperature,
        base_datapack,
        redox_pack,
        ln_k_fe,
        initial_lambda=lambda_sat_zero,
        y_init=sat_state.y,
        tol=tolerance,
        max_iter=iterations,
    )
    high_species = _state_species(high_state, base_datapack, redox_pack)[1]
    residual_high = _oxygen_residual(
        high_state, high_species, redox_pack, oxygen_inventory
    )
    parent_moles[fe_index] = total_fe
    all_metal_resolution = max(
        oxygen_tolerance,
        math.ulp(max(abs(oxygen_inventory), 1.0)),
    )
    # Premise: a positive endpoint gap below one oxygen ULP is not resolvable
    # at the last representable metal amount. Algebra: compare the speciated
    # gap with max(outer tolerance, ULP(O)); units are mol O. Sanity: a
    # subnormal Fe-only oxygen feed remains a representable buffered result.
    if residual_high >= all_metal_resolution:
        raise RedoxEndpointError(
            "oxygen inventory is below the all-metal endpoint",
            endpoint="reductant_exceeds_reducible_oxygen",
            flags=("reductant_exceeds_reducible_oxygen",),
            oxygen_residual_mol=residual_high,
        )
    # The stoichiometric surplus sets only the first trial. Endpoint
    # classification and subsequent bracket updates use speciated residuals;
    # retaining this seed preserves the Fe-only closed-solve pins.
    metal = total_fe * residual_low / (residual_low + target_surplus)
    metal = min(max(metal, metal_low), metal_high)
    lam = lambda_sat_zero
    state_seed = sat_state.y
    fallback_fired = (
        oxidised_state.fallback_fired
        or sat_state.fallback_fired
        or saturation_fallbacks > 0
        or high_state.fallback_fired
        or high_saturation_fallbacks > 0
    )
    for outer_iteration in range(1, 81):
        parent_moles[fe_index] = total_fe - metal
        lam, state, saturation_fallbacks = _saturation_state(
            parent_moles,
            temperature,
            base_datapack,
            redox_pack,
            ln_k_fe,
            initial_lambda=lam,
            y_init=state_seed,
            tol=tolerance,
            max_iter=iterations,
        )
        fallback_fired = (
            fallback_fired or state.fallback_fired or saturation_fallbacks > 0
        )
        response = _response(state, base_datapack, redox_pack)
        state_species = _state_species(state, base_datapack, redox_pack)[1]
        residual = _oxygen_residual(
            state, state_species, redox_pack, oxygen_inventory
        )
        derivative = response.oxygen_derivative_m
        # Premise: both metal amount and the reported oxygen ledger are
        # float64. Algebra: one metal ULP moves oxygen by dO/dm*ulp(m), while
        # the ledger itself resolves only to an oxygen ULP. Units are mol O;
        # this floor matters only when the requested root is less representable.
        representable_tolerance = max(
            abs(derivative) * math.ulp(metal),
            math.ulp(max(abs(oxygen_inventory), 1.0)),
        )
        if abs(residual) <= max(oxygen_tolerance, representable_tolerance):
            return _assemble_result(
                mode="closed",
                temperature_K=temperature,
                lambda_value=lam,
                lambda_sat=lam,
                state=state,
                metal_moles=metal,
                oxygen_inventory=oxygen_inventory,
                base_datapack=base_datapack,
                redox_pack=redox_pack,
                oxygen_parent_counts=oxygen_parent_counts,
                tolerance=tolerance,
                fallback_fired=fallback_fired,
            )
        if residual > 0.0:
            metal_low = metal
            residual_low = residual
        else:
            metal_high = metal
            residual_high = residual
        candidate = metal - residual / derivative if derivative != 0.0 else math.nan
        if not (
            math.isfinite(candidate)
            and metal_low < candidate < metal_high
            and 0.0 < candidate < total_fe
        ):
            candidate = (metal_low + metal_high) / 2.0
        metal = candidate
        state_seed = state.y
    raise RedoxNumericalError("pure-Fe metal amount root did not converge")
