"""Closed and imposed Fe redox equilibrium on the IMCC liquid kernel."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Real
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from openimcc.kernel import (
    ImccRefusal,
    active_residual_jacobian,
    solve_active_warm,
)
from openimcc.model import (
    ImccComponentOutsideDomainError,
    ImccTOutsideDatapackDomainError,
    load_datapack,
)
from openimcc.redox_pack import RedoxPack, load_redox_pack, ln_k_prime


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
    """A complete redox point with liquid speciation and the Fe ledger."""

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
    include_ferric: bool = True,
    y_init: np.ndarray | None = None,
    tol: float,
    max_iter: int,
) -> _LiquidState:
    parent_names = _redox_parent_names(redox_pack)
    fe_index = parent_names.index("FeO")
    ferric_nu = np.zeros(len(parent_names), dtype=np.float64)
    ferric_nu[fe_index] = 1.0
    nu_full = np.column_stack(
        (_redox_parent_stoichiometry(base_datapack, redox_pack), ferric_nu)
    )
    lnK_base = np.log(10.0) * (
        base_datapack.A + base_datapack.B / temperature_K
    )
    ferric_row = redox_pack.rows[1]
    try:
        lnK_ferric = ln_k_prime(
            redox_pack, ferric_row, temperature_K, ln_f_o2=lambda_value
        )
    except ValueError as exc:
        raise ImccTOutsideDatapackDomainError(
            f"temperature {temperature_K:g} K is outside the FeO1.5 redox row domain"
        ) from exc
    lnK_full = np.concatenate((lnK_base, np.asarray([lnK_ferric])))
    if not include_ferric:
        nu_full = nu_full[:, :-1]
        lnK_full = lnK_full[:-1]

    basis = float(np.sum(parent_moles))
    active_parent = parent_moles > 0.0
    inactive_complex = np.any(
        (nu_full > 0.0) & (~active_parent[:, None]), axis=0
    )
    active_complex = ~inactive_complex
    x_target = parent_moles[active_parent] / basis
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
    complex_names = tuple(base_datapack.reactions) + ("FeO1.5(l)",)
    fe_index = parent_names.index("FeO")
    all_nu = np.column_stack(
        (
            _redox_parent_stoichiometry(base_datapack, redox_pack),
            np.eye(len(parent_names), dtype=np.float64)[fe_index],
        )
    )
    oxygen_parent = _parent_oxygen_counts(parent_names, redox_pack)
    ferric_delta = -2.0 * float(
        redox_pack.rows[1]["external_oxygen_stoich_product_positive"]
    )
    oxygen_complex = oxygen_parent @ all_nu
    oxygen_complex[-1] += ferric_delta

    total_species_moles = state.basis / state.D
    parent_amount = np.zeros(len(parent_names), dtype=np.float64)
    parent_amount[state.active_parent] = total_species_moles * np.exp(state.y)
    complex_fraction = np.zeros(len(complex_names), dtype=np.float64)
    complex_fraction[state.active_complex] = state.g
    complex_amount = total_species_moles * complex_fraction
    oxygen_moles = float(
        np.dot(parent_amount, oxygen_parent)
        + np.dot(complex_amount, oxygen_complex)
    )
    parents = {
        name: float(value) for name, value in zip(parent_names, state.parent_moles)
    }
    species = {
        name: float(value)
        for name, value in zip(parent_names + complex_names, np.concatenate((parent_amount, complex_amount)))
    }
    return parents, species, oxygen_moles


def _response(
    state: _LiquidState,
    base_datapack: Any,
    redox_pack: RedoxPack,
) -> _Response:
    parent_names = _redox_parent_names(redox_pack)
    fe_index = parent_names.index("FeO")
    ferric_full_index = base_datapack.n_complexes
    active_fe = int(np.flatnonzero(state.active_parent == fe_index)[0])
    ferric_hits = np.flatnonzero(state.active_complex == ferric_full_index)
    if ferric_hits.size == 0:
        raise RedoxNumericalError("ferric FeO1.5 is inactive in the liquid solve")
    ferric_local = int(ferric_hits[0])
    ferric_delta = -2.0 * float(
        redox_pack.rows[1]["external_oxygen_stoich_product_positive"]
    )
    shift = np.zeros(len(state.active_complex), dtype=np.float64)
    shift[ferric_local] = -float(
        redox_pack.rows[1]["external_oxygen_stoich_product_positive"]
    )
    M = state.nu - np.outer(state.x_target, state.S - 1.0)
    # Premise: the converged balances satisfy F(y, x, lnK) = 0. For a
    # parameter p, J dy/dp = -F_p; the row-scaled Jacobian is the one used by
    # Newton. The ferric lnK shift is q*d(lambda), q=-nu_O2=1/4, hence
    # F_lambda=M@(g*q). Units: y and lambda are dimensionless. Sanity: when
    # ferric Fe tends to zero, dO/dlambda tends to zero with its amount.
    f_lambda = M @ (state.g * shift)
    try:
        y_lambda = np.linalg.solve(state.J, -f_lambda)
    except np.linalg.LinAlgError as exc:
        raise RedoxNumericalError("singular IMCC Jacobian in dO/dlambda") from exc
    dg_lambda = state.g * (state.nu.T @ y_lambda + shift)
    dD_lambda = float(np.dot(state.S - 1.0, dg_lambda))
    d_ferric_lambda = state.basis / state.D * (
        dg_lambda[ferric_local]
        - state.g[ferric_local] * dD_lambda / state.D
    )
    capacity = ferric_delta * d_ferric_lambda

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
    d_ferric_m = state.basis / state.D * (
        state.g[ferric_local] * (basis_prime / state.basis - dD_m / state.D)
        + dg_m[ferric_local]
    )
    oxygen_parent = _parent_oxygen_counts(parent_names, redox_pack)
    oxygen_derivative_m = (
        -oxygen_parent[fe_index] + ferric_delta * d_ferric_m
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
        return ln_k_prime(redox_pack, redox_pack.rows[0], temperature_K)
    except ValueError as exc:
        raise ImccTOutsideDatapackDomainError(
            f"temperature {temperature_K:g} K is outside the Fe metal reference row domain"
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
    reduced = _active_inner(
        parent_moles,
        temperature_K,
        0.0,
        base_datapack,
        redox_pack,
        include_ferric=False,
        y_init=y_init,
        tol=tol,
        max_iter=max_iter,
    )
    reduced_fe = int(np.flatnonzero(reduced.active_parent == fe_index)[0])
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


def _make_flags(
    temperature_K: float,
    base_datapack: Any,
    state: _LiquidState,
    redox_pack: RedoxPack,
    capacity: float,
    capacity_tolerance: float,
) -> tuple[str, ...]:
    flags: list[str] = ["liquid_only_model"]
    for index in state.active_complex:
        if index >= len(base_datapack.domains):
            continue
        low, high = base_datapack.domains[index]
        if temperature_K < low or temperature_K > high:
            flags.append(
                f"temperature extrapolation for {base_datapack.reactions[index]}"
            )
    base_parents = set(base_datapack.parent_oxides)
    unassociated = [
        str(parent["element"])
        for index, parent in enumerate(redox_pack.parents)
        if index in state.active_parent
        and str(parent["parent_oxide"]).removesuffix("(l)") not in base_parents
    ]
    if unassociated:
        flags.append(
            "unassociated liquid parents without published IMCC rows: "
            + ", ".join(unassociated)
        )
    for row in redox_pack.rows:
        row_name = str(row["complex"])
        for flag in row["range_flags"]:
            flags.append(f"{row_name}: {flag}")
        low, high = map(float, row["T_domain_K"])
        if temperature_K < low or temperature_K > high:
            flags.append(f"temperature extrapolation for {row_name}")
    if capacity <= capacity_tolerance:
        flags.append("redox_poorly_buffered")
    return tuple(dict.fromkeys(flags))


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
    baseline_oxygen = float(
        np.dot(state.parent_moles, oxygen_parent_counts)
        + metal_moles * oxygen_parent_counts[fe_index]
    )
    residual = (
        None if oxygen_inventory is None else oxygen_mol - oxygen_inventory
    )
    oxygen_scale = oxygen_mol if oxygen_inventory is None else oxygen_inventory
    flags = _make_flags(
        temperature_K,
        base_datapack,
        state,
        redox_pack,
        oxygen_capacity,
        tolerance * max(abs(oxygen_scale), 1.0),
    )
    active_log_activity = {
        parent_names[index]: float(value)
        for index, value in zip(state.active_parent, state.y)
    }
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
    active_lows.extend(float(row["T_domain_K"][0]) for row in redox_pack.rows)
    active_highs.extend(float(row["T_domain_K"][1]) for row in redox_pack.rows)
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

    loaded = load_datapack()
    base_datapack = loaded.kernel_datapack
    redox_pack = load_redox_pack()
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
    non_fe_oxygen = float(
        np.dot(parent_moles, oxygen_parent_counts)
        - total_fe * oxygen_parent_counts[fe_index]
    )

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
        buffered = lambda_value < lambda_sat - max(1.0e-10, tolerance)
        if buffered and invariant_segment and oxygen_inventory is None:
            raise RedoxEndpointError(
                "imposed lambda is on a buffered invariant segment; oxygen inventory is required to select metal extent",
                endpoint="imposed_invariant_requires_oxygen",
                flags=("imposed_invariant_requires_oxygen",),
                lambda_bound=lambda_sat,
            )
        if (
            invariant_segment
            and oxygen_inventory is not None
            and lambda_value <= lambda_sat + max(1.0e-10, tolerance)
        ):
            # On a buffered invariant segment lambda does not identify m;
            # the supplied oxygen inventory selects its Fe-metal extent.
            metal_low = 0.0
            metal_high = total_fe * (1.0 - 1.0e-10)

            def imposed_state_at_m(metal_amount: float) -> tuple[_LiquidState, float]:
                amounts = parent_moles.copy()
                amounts[fe_index] = total_fe - metal_amount
                candidate_state = _active_inner(
                    amounts,
                    temperature,
                    lambda_value,
                    base_datapack,
                    redox_pack,
                    y_init=state.y,
                    tol=tolerance,
                    max_iter=iterations,
                )
                return candidate_state, _state_species(
                    candidate_state, base_datapack, redox_pack
                )[2]

            low_state, low_oxygen = imposed_state_at_m(metal_low)
            high_state, high_oxygen = imposed_state_at_m(metal_high)
            low_residual = low_oxygen - oxygen_inventory
            high_residual = high_oxygen - oxygen_inventory
            if low_residual * high_residual > 0.0:
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
                    "imposed oxygen inventory is outside the buffered extent range",
                    endpoint="imposed_inventory_inconsistent",
                    flags=("imposed_inventory_inconsistent",),
                    oxygen_residual_mol=min(abs(low_residual), abs(high_residual)),
                )
            buffered_state = low_state
            metal = 0.0
            for _ in range(64):
                metal = (metal_low + metal_high) / 2.0
                buffered_state, buffered_oxygen = imposed_state_at_m(metal)
                residual = buffered_oxygen - oxygen_inventory
                if abs(residual) <= tolerance * max(1.0, abs(oxygen_inventory)):
                    break
                if residual > 0.0:
                    metal_low = metal
                else:
                    metal_high = metal
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
            metal_high = total_fe * (1.0 - 1.0e-10)
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
    if oxygen_inventory < non_fe_oxygen:
        raise RedoxEndpointError(
            "oxygen inventory is below the fixed non-Fe oxide oxygen",
            endpoint="reductant_exceeds_reducible_oxygen",
            flags=("reductant_exceeds_reducible_oxygen",),
            oxygen_residual_mol=non_fe_oxygen - oxygen_inventory,
        )
    if oxygen_inventory == non_fe_oxygen:
        raise RedoxNoLiveCoupleError(
            "all Fe is metallic at the oxygen inventory; no finite Fe redox couple remains"
        )

    # All non-Fe parents retain their oxygen; fully oxidising each Fe parent
    # raises its one-O FeO basis to FeO1.5, adding 0.5 O per Fe. This is the
    # inventory limit (mol O atoms), independent of a gas reference pressure.
    fully_oxidised_oxygen = non_fe_oxygen + 1.5 * total_fe
    if oxygen_inventory >= fully_oxidised_oxygen:
        excess = fully_oxidised_oxygen - oxygen_inventory
        raise RedoxEndpointError(
            "oxygen inventory is at or above the fully oxidised FeO1.5 limit",
            endpoint="fully_oxidised_inventory_limit",
            flags=("fully_oxidised_inventory_limit",),
            oxygen_residual_mol=excess,
        )
    endpoint_distance = min(
        oxygen_inventory - non_fe_oxygen,
        fully_oxidised_oxygen - oxygen_inventory,
    )
    oxygen_tolerance = min(
        tolerance * max(abs(oxygen_inventory), 1.0),
        0.01 * endpoint_distance,
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
    oxidised_oxygen = _state_species(
        oxidised_state, base_datapack, redox_pack
    )[2]
    oxidised_residual = oxidised_oxygen - oxygen_inventory
    if abs(oxidised_residual) <= oxygen_tolerance:
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
    sat_oxygen = _state_species(sat_state, base_datapack, redox_pack)[2]
    sat_residual = sat_oxygen - oxygen_inventory
    if sat_residual <= oxygen_tolerance:
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
            if upper_residual >= 0.0:
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
            upper_residual = (
                _state_species(upper_state, base_datapack, redox_pack)[2]
                - oxygen_inventory
            )
        if upper_residual < 0.0:
            raise RedoxNumericalError("could not bracket the fully oxidised inventory limit")
        lower_residual = sat_residual
        lam = lower + (upper - lower) * (
            -lower_residual / (upper_residual - lower_residual)
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
            oxygen_mol = _state_species(state, base_datapack, redox_pack)[2]
            residual = oxygen_mol - oxygen_inventory
            if abs(residual) <= oxygen_tolerance:
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
            if residual < 0.0:
                lower = lam
                lower_residual = residual
            else:
                upper = lam
                upper_residual = residual
            candidate = (
                lam - residual / response.oxygen_capacity
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
    residual_high = non_fe_oxygen - oxygen_inventory
    if residual_high >= 0.0:
        raise RedoxEndpointError(
            "oxygen inventory is below the all-metal endpoint",
            endpoint="reductant_exceeds_reducible_oxygen",
            flags=("reductant_exceeds_reducible_oxygen",),
            oxygen_residual_mol=residual_high,
        )
    metal = total_fe * residual_low / (residual_low - residual_high)
    metal = min(max(metal, total_fe * 1.0e-12), total_fe * (1.0 - 1.0e-12))
    lam = lambda_sat_zero
    state_seed = sat_state.y
    fallback_fired = (
        oxidised_state.fallback_fired
        or sat_state.fallback_fired
        or saturation_fallbacks > 0
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
        residual = response.oxygen_mol - oxygen_inventory
        if abs(residual) <= oxygen_tolerance:
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
        derivative = response.oxygen_derivative_m
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
