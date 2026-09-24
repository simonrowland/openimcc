"""IMCC-SF04 thin gas mass-action layer (chunk 5a).

Given melt parent activities from the IMCC-SF04 kernel/adapter plus gas-species
G(T) rows from the JANAF tables, compute equilibrium partial pressures for the
SF04 vaporization reaction set.

This module is intentionally thin: it performs no fO2 modeling (fO2 is pinned by
the caller), no melt equilibrium solve (activities are inputs), and does not
silently upgrade data authority.  It is a diagnostic shadow, consistent with
the IMCC-SF04 spec r2.1.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np

from openimcc.kernel import ImccRefusal

if TYPE_CHECKING:
    import pandas as pd


# --------------------------------------------------------------------------- #
# Physical constants and reference states
# --------------------------------------------------------------------------- #

R_J_MOL_K = 8.314462618
"""Molar gas constant, J / (mol K)."""

BAR = 1.0
"""Numerical value of the JANAF gas standard pressure p° = 1 bar.

``evaluate_gas`` stores each gas pressure as the dimensionless ratio
``p_i / p°``. With p° = 1 bar that ratio equals the pressure in bar, so
the mass-action algebra uses the stored floats directly and does not
divide by this constant.
"""


# --------------------------------------------------------------------------- #
# Typed refusal
# --------------------------------------------------------------------------- #


class ImccGasSpeciesNotFoundError(ImccRefusal):
    """Raised when a requested gas species or oxide has no G(T) row at T."""

    code = "imcc_gas_species_not_found"


class ImccGasTemperatureOutsideDomainError(ImccRefusal):
    """Raised when T falls outside the declared G(T) interval for a species."""

    code = "imcc_gas_T_outside_domain"


# --------------------------------------------------------------------------- #
# Packaged data and explicit VapoRock override
# --------------------------------------------------------------------------- #


_VAPOROCK_ENV_VAR = "OPENIMCC_VAPOROCK_ROOT"
_GAS_TABLE_RELPATH = Path("src") / "vaporock" / "data" / "JANAF-vapor-data-full.csv"
_CONDENSATE_TABLE_RELPATH = Path("data") / "condensate-thermo-data.csv"
_PACKAGED_DATA_PACKAGE = "openimcc.data.gas"
_PACKAGED_GAS_NAME = "gas-shomate.csv"
_PACKAGED_CONDENSATE_NAME = "condensate.csv"


class ImccGasDataUnavailableError(ImccRefusal):
    """The gas thermodynamic tables have not been located."""

    code = "imcc_gas_data_unavailable"


def _vaporock_root() -> Path:
    """Root of the explicitly selected VapoRock checkout.

    The normal source is the package's own tables.  This path exists only for a
    caller who deliberately sets ``OPENIMCC_VAPOROCK_ROOT``.
    """
    override = os.environ.get(_VAPOROCK_ENV_VAR)
    if not override:
        raise ImccGasDataUnavailableError(
            f"{_VAPOROCK_ENV_VAR} is not set; packaged gas tables are the "
            "default, and this helper resolves only an explicit VapoRock override"
        )
    root = Path(override).expanduser()
    if not (root / _GAS_TABLE_RELPATH).is_file():
        raise ImccGasDataUnavailableError(
            f"{_VAPOROCK_ENV_VAR}={override!r} does not contain "
            f"{_GAS_TABLE_RELPATH}"
        )
    return root.resolve()


def _packaged_resource(name: str):
    resource = resources.files(_PACKAGED_DATA_PACKAGE).joinpath(name)
    if not resource.is_file():
        raise ImccGasDataUnavailableError(
            f"packaged gas resource {name!r} is missing from "
            f"{_PACKAGED_DATA_PACKAGE}"
        )
    return resource


def _packaged_database_path(name: str) -> Path:
    """Filesystem path used for diagnostics when the package is unpacked."""
    return Path(__file__).resolve().parent / "data" / "gas" / name


def default_gas_database_path() -> Path:
    """Path to the packaged gas table, or to an explicit VapoRock override."""
    if os.environ.get(_VAPOROCK_ENV_VAR):
        return _vaporock_root() / _GAS_TABLE_RELPATH
    _packaged_resource(_PACKAGED_GAS_NAME)
    return _packaged_database_path(_PACKAGED_GAS_NAME)


def default_condensate_database_path() -> Path:
    """Path to the packaged condensate table, or to an explicit override."""
    if os.environ.get(_VAPOROCK_ENV_VAR):
        return _vaporock_root() / _CONDENSATE_TABLE_RELPATH
    _packaged_resource(_PACKAGED_CONDENSATE_NAME)
    return _packaged_database_path(_PACKAGED_CONDENSATE_NAME)


# --------------------------------------------------------------------------- #
# SF04 vaporization reaction set
# --------------------------------------------------------------------------- #

# Map retained gas species to the vaporization reaction
#     parent_oxide(l) -> n_gas * gas(g) + n_O2 * O2(g).
#
# Derivation: for a parent E_a O_b and target gas E_c O_d, element balance gives
# n_gas = a/c and n_O2 = (b - n_gas*d)/2.  A negative n_O2 puts O2 on the
# reactant side.  O(g) is the oxide-free dissociation 1/2 O2(g) -> O(g), encoded
# with an empty parent and n_O2 = -1/2.  All G degrees use elements in their
# 298 K standard states and ideal gas at 1 bar, so the elemental references
# cancel in Delta G degrees.  Unit check: stoichiometric coefficients multiply
# J/mol values, hence -Delta G degrees/(R*T) is dimensionless.  Sanity checks:
# Al2O3 -> 2 AlO + 1/2 O2 balances Al2O3, while
# Na2O + 1/2 O2 -> 2 NaO balances Na2O2.
_SF04_REACTIONS: dict[str, tuple[str, int, float]] = {
    "Na": ("Na2O", 2, 0.5),
    "K": ("K2O", 2, 0.5),
    "SiO": ("SiO2", 1, 0.5),
    "Fe": ("FeO", 1, 0.5),
    "FeO": ("FeO", 1, 0.0),
    "Mg": ("MgO", 1, 0.5),
    "MgO": ("MgO", 1, 0.0),
    "SiO2": ("SiO2", 1, 0.0),
    "O": ("", 1, -0.5),
    "AlO": ("Al2O3", 2, 0.5),
    "AlO2": ("Al2O3", 2, -0.5),
    "Al2O": ("Al2O3", 1, 1.0),
    "Al2O2": ("Al2O3", 1, 0.5),
    "Na2": ("Na2O", 1, 0.5),
    "NaO": ("Na2O", 2, -0.5),
    "K2": ("K2O", 1, 0.5),
    "KO": ("K2O", 2, -0.5),
    "Si": ("SiO2", 1, 1.0),
    "Al": ("Al2O3", 2, 1.5),
    "CaO": ("CaO", 1, 0.0),
    "Ca": ("CaO", 1, 0.5),
    "O2": ("", 1, 0.0),  # special: p_O2 = fO2
}

IMCC_GAS_CHANNEL_SPECIES = tuple(_SF04_REACTIONS)

# These authority labels mirror the row-level classes in PROVENANCE.yaml.  A
# reaction is only as authoritative as its least-authoritative input row, so a
# potassium channel inherits the K2O(l) secondary-transcription flag while the
# other channels retain their JANAF/Lamoreaux classes.
_GAS_PROVENANCE_AUTHORITY = {
    species: "janaf_fitted" for species in IMCC_GAS_CHANNEL_SPECIES
}
_OXIDE_PROVENANCE_AUTHORITY = {
    "MgO": "lam1987_transcribed",
    "CaO": "lam1987_transcribed",
    "Al2O3": "lam1987_transcribed",
    "SiO2": "lam1987_transcribed",
    "SiO2(cr)": "lam1987_transcribed",
    "Na2O": "lam1984_transcribed",
    "K2O": "secondary_transcription_unverified_primary",
    "FeO": "janaf_transcribed",
}
_PROVENANCE_AUTHORITY_RANK = {
    "secondary_transcription_unverified_primary": 0,
    "lam1984_transcribed": 1,
    "lam1987_transcribed": 1,
    "janaf_transcribed": 2,
    "janaf_fitted": 3,
}


def gas_species_provenance(species: str) -> dict[str, str | None]:
    """Return the row authorities feeding one retained gas channel.

    ``evaluate_gas`` intentionally keeps its historical ``dict[str, float]``
    return type.  Callers that need the data-quality label query it separately;
    the overall authority is the least-authoritative row in that reaction.
    """
    if species not in _SF04_REACTIONS:
        raise ImccGasSpeciesNotFoundError(
            f"unknown retained gas species {species!r}"
        )
    oxide, _, _ = _SF04_REACTIONS[species]
    gas_authority = _GAS_PROVENANCE_AUTHORITY[species]
    oxide_authority = _OXIDE_PROVENANCE_AUTHORITY.get(oxide)
    authorities = [gas_authority]
    if oxide_authority is not None:
        authorities.append(oxide_authority)
    authority = min(authorities, key=_PROVENANCE_AUTHORITY_RANK.__getitem__)
    return {
        "species_name": species,
        "authority": authority,
        "gas_authority": gas_authority,
        "condensate_authority": oxide_authority,
    }

IMCC_SF04_WORKBOOK_GRID_K = (
    1500.0,
    1625.0,
    1750.0,
    1875.0,
    1900.0,
    2000.0,
    2125.0,
    2250.0,
    2375.0,
    2500.0,
)

# EXTRAPOLATED-INFORMATIONAL labels for every channel whose gas or parent-oxide
# table does not cover the full Schaefer-2004 workbook grid above.  The fitted
# gas rows cover 1500--3000 K; these labels therefore describe parent-oxide
# gaps only.  The evaluator still refuses these temperatures unless
# allow_extrapolation=True; labels disclose the reason and never widen the
# executable domain.
IMCC_GAS_WORKBOOK_EXTRAPOLATION_LABELS: dict[str, str] = {
    "SiO": "SiO2(l) [1996, 3000] K misses workbook T < 1996 K",
    "Mg": "MgO(l) [3100, 3500] K lies above the whole workbook grid",
    "MgO": "MgO(l) [3100, 3500] K lies above the whole workbook grid",
    "SiO2": "SiO2(l) [1996, 3000] K misses workbook T < 1996 K",
    "AlO": "Al2O3(l) [2327, 3000] K misses workbook T < 2327 K",
    "AlO2": "Al2O3(l) [2327, 3000] K misses workbook T < 2327 K",
    "Al2O": "Al2O3(l) [2327, 3000] K misses workbook T < 2327 K",
    "Al2O2": "Al2O3(l) [2327, 3000] K misses workbook T < 2327 K",
    "Si": "SiO2(l) [1996, 3000] K misses workbook T < 1996 K",
    "Al": "Al2O3(l) [2327, 3000] K misses workbook T < 2327 K",
    "CaO": "CaO(l) [2900, 3800] K lies above the whole workbook grid",
    "Ca": "CaO(l) [2900, 3800] K lies above the whole workbook grid",
}

IMCC_GAS_WORKBOOK_IN_DOMAIN_SPECIES = (
    "Na",
    "K",
    "Fe",
    "FeO",
    "O",
    "Na2",
    "NaO",
    "K2",
    "KO",
    "O2",
)

# Closure ledger for species outside the retained channel set.  The first seven
# entries have no gas G(T) row in the vendored JANAF source set.  The titanium
# gas rows exist, but no TiO2(l) parent row exists, so those reactions remain
# incomplete.
IMCC_GAS_NO_JANAF_ROWS: dict[str, str] = {
    "Na2O": "needs a source-rated Na2O(g) standard-Gibbs row",
    "K2O": "needs a source-rated K2O(g) standard-Gibbs row",
    "Na+": (
        "needs Na+(g) and electron standard-Gibbs rows plus a disclosed "
        "ionization/electroneutrality convention"
    ),
    "K+": (
        "needs K+(g) and electron standard-Gibbs rows plus a disclosed "
        "ionization/electroneutrality convention"
    ),
    "e-": "needs an electron standard state and a coupled charge-balance model",
    "Zn": (
        "needs Zn(g) and ZnO(l) standard-Gibbs rows plus ZnO in the melt-oxide basis"
    ),
    "ZnO": (
        "needs ZnO(g) and ZnO(l) standard-Gibbs rows plus ZnO in the melt-oxide basis"
    ),
}

IMCC_GAS_INCOMPLETE_PARENT_SPECIES: dict[str, str] = {
    "Ti": "Ti(g) exists; needs a source-rated TiO2(l) standard-Gibbs row",
    "TiO": "TiO(g) exists; needs a source-rated TiO2(l) standard-Gibbs row",
    "TiO2": "TiO2(g) exists; needs a source-rated TiO2(l) standard-Gibbs row",
}

IMCC_GAS_UNAVAILABLE_SPECIES = {
    **IMCC_GAS_NO_JANAF_ROWS,
    **IMCC_GAS_INCOMPLETE_PARENT_SPECIES,
}

IMCC_PARENT_OXIDES = (
    "SiO2",
    "MgO",
    "FeO",
    "CaO",
    "Al2O3",
    "TiO2",
    "Na2O",
    "K2O",
)


# --------------------------------------------------------------------------- #
# Data pack
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ImccGasDatapack:
    """Loaded JANAF + condensate thermodynamic tables for the gas layer."""

    gas_df: pd.DataFrame
    oxide_df: pd.DataFrame
    gas_path: Path
    oxide_path: Path


def load_gas_datapack(
    gas_path: str | Path | None = None,
    oxide_path: str | Path | None = None,
) -> ImccGasDatapack:
    """Load the JANAF gas and condensate-oxide thermodynamic tables.

    With no arguments the packaged tables are opened through
    :mod:`importlib.resources`.  Setting ``OPENIMCC_VAPOROCK_ROOT`` explicitly
    selects the matching VapoRock files instead.  Explicit ``gas_path`` and
    ``oxide_path`` arguments always win for callers building a diagnostic pack.
    Both files are read-only.
    """
    import pandas as pd

    # Resolved lazily: an unconfigured install must fail with a typed refusal
    # at CALL time, not at import time, or `import openimcc.gas` breaks for
    # everyone who only wanted the species constants.
    if gas_path is not None:
        gas_source = Path(gas_path)
        gas_display_path = gas_source
        gas_df = pd.read_csv(gas_source)
    elif os.environ.get(_VAPOROCK_ENV_VAR):
        gas_source = _vaporock_root() / _GAS_TABLE_RELPATH
        gas_display_path = gas_source
        gas_df = pd.read_csv(gas_source)
    else:
        gas_resource = _packaged_resource(_PACKAGED_GAS_NAME)
        gas_display_path = _packaged_database_path(_PACKAGED_GAS_NAME)
        with gas_resource.open("rb") as handle:
            gas_df = pd.read_csv(handle)
    if "species_name" not in gas_df.columns:
        raise ImccGasSpeciesNotFoundError(
            f"JANAF gas database {gas_display_path} missing 'species_name' column"
        )
    gas_df = gas_df.set_index("species_name")

    if oxide_path is not None:
        oxide_source = Path(oxide_path)
        oxide_display_path = oxide_source
        oxide_df = pd.read_csv(oxide_source)
    elif os.environ.get(_VAPOROCK_ENV_VAR):
        oxide_source = _vaporock_root() / _CONDENSATE_TABLE_RELPATH
        if not oxide_source.is_file():
            raise ImccGasDataUnavailableError(
                f"{_VAPOROCK_ENV_VAR}={os.environ[_VAPOROCK_ENV_VAR]!r} does not contain "
                f"{_CONDENSATE_TABLE_RELPATH}"
            )
        oxide_display_path = oxide_source
        oxide_df = pd.read_csv(oxide_source)
    else:
        oxide_resource = _packaged_resource(_PACKAGED_CONDENSATE_NAME)
        oxide_display_path = _packaged_database_path(_PACKAGED_CONDENSATE_NAME)
        with oxide_resource.open("rb") as handle:
            oxide_df = pd.read_csv(handle)
    if "species_name" not in oxide_df.columns:
        raise ImccGasSpeciesNotFoundError(
            f"condensate database {oxide_display_path} missing 'species_name' column"
        )
    oxide_df = oxide_df.set_index("species_name")

    return ImccGasDatapack(
        gas_df=gas_df,
        oxide_df=oxide_df,
        gas_path=gas_display_path,
        oxide_path=oxide_display_path,
    )


# --------------------------------------------------------------------------- #
# G(T) evaluation
# --------------------------------------------------------------------------- #


def _janaf_gibbs(T: float, row: pd.Series) -> float:
    """Return G°(T) in J/mol for a JANAF gas species from Shomate coefficients.

    Derivation
    ----------
    Premise: the packaged and explicit-override gas CSVs store NIST Shomate
    coefficients in columns A–H with ``t = T/1000`` (T in K). This function
    uses A–G only, matching the historical ``_janaf_dH`` / ``_janaf_S`` /
    ``_janaf_G`` implementation. It is not the NASA Glenn seven-coefficient
    family (NASA/TP-2002-211556), which
    writes ``Cp°/R``, ``H°/(RT)``, and ``S°/R`` as polynomials in T (K) with
    integration constants ``a1…a7``.

    NIST WebBook Shomate enthalpy includes a ``− H`` term so that
    ``H°(T) − H°_298.15`` is ~0 at 298.15 K on a segment that covers 298.15 K.
    The source transcription omits that ``− H`` term because the F offset
    already reproduces the LAMOR/JANAF scale. This function follows that
    transcription:

        dH(t) = A t + B/2 t^2 + C/3 t^3 + D/4 t^4 − E/t + F        (kJ/mol)
        S(t)  = A ln(t) + B t + C/2 t^2 + D/3 t^3 − E/(2 t^2) + G  (J/mol/K)
        G°(T) = dH(t) * 1000 − T * S(t)                            (J/mol)

    Column H is present on the CSV row and is unused here. Reference state
    on the gas table is ideal gas at 1 bar, elements in their 298 K
    standard states, so those baselines cancel in a reaction ΔG° assembled
    from these rows.

    Unit check: dH in kJ/mol * 1000 → J/mol; T * S (K * J/mol/K) → J/mol.
    Sanity (in-domain): the packaged O2(g) row at T = 2000 K gives
    G° ≈ −478320.38 J/mol. Sign is negative; −T S dominates
    (S ≈ 268.75 J/mol/K, T S ≈ 537.5 kJ/mol). 298.15 K is below the packaged
    fit's T_min = 1500 K, so this function does not claim a 298.15 K JANAF match
    from these coefficients.
    """
    t = T / 1000.0
    dH = (
        row["A"] * t
        + row["B"] / 2.0 * t**2
        + row["C"] / 3.0 * t**3
        + row["D"] / 4.0 * t**4
        - row["E"] / t
        + row["F"]
    )
    S = (
        row["A"] * np.log(t)
        + row["B"] * t
        + row["C"] / 2.0 * t**2
        + row["D"] / 3.0 * t**3
        - row["E"] / 2.0 / t**2
        + row["G"]
    )
    return dH * 1000.0 - T * S


def _lamor_gibbs(T: float, row: pd.Series) -> float:
    """Return G°(T) in J/mol for a LAMOR condensate species.

    Derivation
    ----------
    The packaged condensate table (LAMOR / JANAF fits) stores a 5-coefficient
    polynomial fit plus ΔH°298/R.  With τ = T/1000 K:

        poly(τ) = dG_A + dG_B τ + dG_C τ^2 + dG_D τ^3 + dG_E τ^4
        G°(T) = -R T poly(τ) + R ΔH°298 * 1000

    The polynomial term captures the temperature dependence of ΔG°/R/T; the
    dH298 term anchors the absolute scale.  Result is the standard Gibbs energy
    of the condensed oxide in J/mol, sharing the same elemental reference as the
    JANAF gas table, so reaction ΔG° is reference-state-consistent.

    Unit check: R*T*dimensionless -> J/mol; R*ΔH°298*1000 -> J/mol.
    Sanity case: for a species with ΔH°298=0 and zero polynomial, G°(T)=0 at all T
    (the reference itself).
    """
    A = row["dG_A"]
    B = row["dG_B"]
    C = row["dG_C"]
    D = row["dG_D"]
    E = row["dG_E"]
    dH298 = row["dH298_R"]

    G_coefs = np.array([A, B, C, D, E])
    G_scale = np.array([1.0, 1e3, 1e6, 1e9, 1e12])
    G_poly_coefs = (G_coefs / G_scale)[::-1]
    return -R_J_MOL_K * T * np.polyval(G_poly_coefs, T) + R_J_MOL_K * dH298 * 1000.0


def _nearest_interval_row(
    df: pd.DataFrame, species: str, T: float, allow_extrapolation: bool = False
) -> pd.Series:
    """Select one thermodynamic interval for ``species`` at temperature ``T``.

    Selection inside this function: among rows with ``T_min <= T``, take the
    one with the largest ``T_min``. If every ``T_min`` is ``> T``, take
    ``rows.iloc[0]`` (first row of that species in the loaded frame). Then,
    unless ``allow_extrapolation=True``, raise
    ``ImccGasTemperatureOutsideDomainError`` when ``T`` is outside that
    selected row's ``[T_min, T_max]``. That default refusal is the V2
    refusal-semantics contract in the IMCC-SF04 spec §2.

    This is not legacy VapoRock ``_calc_gibbs_species_JANAF_singleT`` interval
    parity. That path, after replacing the lowest ``T_min`` with 0 and the
    highest ``T_max`` with 1e8, masks with ``(T > T_min) & (T <= T_max)``.
    At a shared breakpoint ``T = T_max(i) = T_min(i+1)`` that mask selects
    the lower interval; this function selects the upper interval. On the
    current ``JANAF-vapor-data-full.csv`` those shared breakpoints include
    AlO 2000 K, AlO2 1000 K, K 1800 K, Mg 2200 K, O2 2000 K, and SiO 1100 K.
    """
    rows = df.loc[df.index == species]
    if rows.empty:
        raise ImccGasSpeciesNotFoundError(
            f"no JANAF G(T) row for gas species {species!r}"
        )
    t_mins = rows["T_min"].astype(float).to_numpy()
    # Largest T_min that is <= T; fallback to rows.iloc[0] if T is below every
    # T_min in the loaded frame.
    valid = t_mins <= T
    if np.any(valid):
        idx = int(np.argmax(t_mins * valid))  # argmax of masked mins gives largest <= T
    else:
        idx = 0
    selected = rows.iloc[idx]
    if not allow_extrapolation and (T < selected["T_min"] or T > selected["T_max"]):
        raise ImccGasTemperatureOutsideDomainError(
            f"T={T} K outside declared G(T) interval for {species!r} "
            f"[{selected['T_min']}, {selected['T_max']}] K"
        )
    return selected


def _oxide_row_for_T(
    df: pd.DataFrame, oxide: str, T: float, allow_extrapolation: bool = False
) -> pd.Series:
    """Select the condensate interval closest to T, refusing extrapolation by default."""
    import pandas as pd

    if oxide not in df.index:
        raise ImccGasSpeciesNotFoundError(
            f"no condensate G(T) row for oxide {oxide!r}"
        )
    rows = df.loc[[oxide]] if df.index.name is None or not isinstance(df.loc[oxide], pd.DataFrame) else df.loc[oxide]
    # The condensate table has one row per oxide; wrap it if a single row.
    if isinstance(rows, pd.Series):
        selected = rows
    else:
        t_mins = rows["T_min"].astype(float).to_numpy()
        valid = t_mins <= T
        idx = int(np.argmax(t_mins * valid)) if np.any(valid) else 0
        selected = rows.iloc[idx]
    if not allow_extrapolation and (T < selected["T_min"] or T > selected["T_max"]):
        raise ImccGasTemperatureOutsideDomainError(
            f"T={T} K outside declared G(T) interval for {oxide!r} "
            f"[{selected['T_min']}, {selected['T_max']}] K"
        )
    return selected


# --------------------------------------------------------------------------- #
# Mass-action: Kp derivation and partial-pressure evaluation
# --------------------------------------------------------------------------- #


def evaluate_gas(
    activities: Mapping[str, float] | Sequence[float] | np.ndarray,
    T_K: float,
    fO2: float,
    datapack: ImccGasDatapack,
    parent_oxides: Sequence[str] | None = None,
    allow_extrapolation: bool = False,
    gas_species: Sequence[str] | None = None,
) -> dict[str, float]:
    """Compute equilibrium partial pressures for the SF04 retained gas set.

    Parameters
    ----------
    activities:
        Parent-oxide activities.  Either a dict keyed by parent-oxide name, or a
        vector aligned with ``parent_oxides`` (default IMCC order).  Activities
        are relative to the pure liquid oxide standard state.
    T_K:
        Temperature in Kelvin. Must be finite and positive on this path.
    fO2:
        Oxygen fugacity, stored as the numerical value of p_O2 / p° with
        p° = 1 bar (pinned by the caller; no internal fO2 model is applied).
        Must be finite and positive on this path.
    datapack:
        Loaded JANAF + condensate thermodynamic tables.
    parent_oxides:
        Ordered parent-oxide names.  Defaults to the IMCC-SF04 8-oxide basis.
    allow_extrapolation:
        If False (default), raise ``ImccGasTemperatureOutsideDomainError``
        when a finite T falls outside the declared G(T) interval of the
        interval selected by ``_nearest_interval_row`` for any species
        consumed by the reaction set.  If True, evaluate at that finite T
        even when it lies outside the selected row (legacy run_gate.py
        behavior).  Non-finite or non-positive T and fO2 raise
        ``ValueError`` before this flag is consulted.
    gas_species:
        Optional retained-species subset.  The default evaluates every
        available channel.  A subset permits channel-specific diagnostics
        and preserves the same typed domain refusal semantics.

    Returns
    -------
    dict[str, float]
        Partial pressure of each retained gas species, reported in bar
        because p° = 1 bar makes p_i / p° numerically equal to p_i / bar
        (see Derivation).

    Derivation
    ----------
    Premise: for each retained gas species we write a single vaporization
    reaction with the parent oxide in the melt as the reactant and O2 as an
    explicit product when stoichiometry requires it:

        oxide(l)  =  n_gas * gas(g)  +  n_O2 * O2(g)                (1)

    The standard Gibbs free energy change for (1) is assembled from the JANAF
    gas-species G°(T) rows and the condensate oxide G°(T) rows:

        ΔG°(T) = n_gas * G°(gas, T) + n_O2 * G°(O2, T) - G°(oxide, T)   (2)

    Both tables are authored against the same elemental reference (elements
    in their 298 K standard states), so that baseline cancels in (2).  The
    standard equilibrium constant is

        K° = exp(-ΔG°(T) / (R T))                                    (3)

    with R in J/(mol K), ΔG° in J/mol, T in K; K° is dimensionless.
    Gas standard states are ideal gas at p° = 1 bar, so IUPAC K° uses
    dimensionless activities p_i / p° (IUPAC Recommendations 1994, eq. 49),
    not pressures with units of bar:

        K° = ((p_gas / p°)^n_gas * (p_O2 / p°)^n_O2) / a_oxide       (4)

    This function stores each gas pressure as the float ``p̃_i = p_i / p°``.
    With p° = 1 bar, ``p̃_i`` equals the numerical value of p_i in bar, and
    (4) is implemented as

        K° = (p̃_gas^n_gas * p̃_O2^n_O2) / a_oxide                    (4')

    without dividing by the ``BAR`` constant.  Solving for p̃_gas at the
    caller-pinned p̃_O2 = fO2:

        p̃_gas = (K° * a_oxide / fO2^n_O2)^(1 / n_gas)               (5)

    The returned dict reports those p̃_gas values as bar.  For the special
    retained species O2, p̃_O2 = fO2 by definition.  For n_O2 = 0, (5)
    reduces to p̃_gas = K° * a_oxide.

    Unit check: ΔG° / (R T) is dimensionless, so K° is dimensionless;
    p̃_i is dimensionless; the bar label on the return value is the p° = 1
    bar identification, not a leftover unit on K°.
    Sanity on this path: ΔG° → +∞ gives K° → 0 and p̃_gas → 0; a_oxide → 0
    gives p̃_gas → 0. For n_O2 > 0, decreasing a finite positive fO2 raises
    p̃_gas as fO2^(-n_O2/n_gas). fO2 = 0 is refused (non-positive), so
    that limit is not a returned result.
    """
    if parent_oxides is None:
        parent_oxides = IMCC_PARENT_OXIDES

    if gas_species is None:
        reactions = tuple(_SF04_REACTIONS.items())
    else:
        requested = (
            (gas_species,) if isinstance(gas_species, str) else tuple(gas_species)
        )
        missing = [name for name in requested if name not in _SF04_REACTIONS]
        if missing:
            raise ImccGasSpeciesNotFoundError(
                f"no IMCC-SF04 gas channel for species {missing!r}"
            )
        reactions = tuple((name, _SF04_REACTIONS[name]) for name in requested)

    T = float(T_K)
    p_O2 = float(fO2)
    if not math.isfinite(T) or T <= 0.0:
        raise ValueError(f"temperature must be finite and positive, got {T_K}")
    if not math.isfinite(p_O2) or p_O2 <= 0.0:
        raise ValueError(f"fO2 must be finite and positive, got {fO2}")

    if isinstance(activities, Mapping):
        act = {name: float(activities.get(name, 0.0)) for name in parent_oxides}
    else:
        arr = np.asarray(activities, dtype=float)
        if arr.shape[0] != len(parent_oxides):
            raise ValueError(
                f"activities vector length {arr.shape[0]} does not match "
                f"{len(parent_oxides)} parent oxides"
            )
        act = {name: float(arr[i]) for i, name in enumerate(parent_oxides)}

    for _gas_name, (oxide, _n_gas, _n_O2) in reactions:
        if not oxide:
            continue
        a_used = act[oxide]
        if not math.isfinite(a_used) or a_used < 0.0:
            raise ValueError(
                f"activity of {oxide!r} must be finite and >= 0, got {a_used}"
            )

    # G°(O2, T) is needed for every O2-producing reaction.
    O2_row = _nearest_interval_row(
        datapack.gas_df, "O2(g)", T, allow_extrapolation=allow_extrapolation
    )
    G_O2 = _janaf_gibbs(T, O2_row)

    result: dict[str, float] = {}
    for gas_name, (oxide, n_gas, n_O2) in reactions:
        if gas_name == "O2":
            result[gas_name] = p_O2
            continue

        gas_species = f"{gas_name}(g)"
        gas_row = _nearest_interval_row(
            datapack.gas_df, gas_species, T, allow_extrapolation=allow_extrapolation
        )
        G_gas = _janaf_gibbs(T, gas_row)

        if oxide:
            oxide_name = f"{oxide}(l)"
            oxide_row = _oxide_row_for_T(
                datapack.oxide_df,
                oxide_name,
                T,
                allow_extrapolation=allow_extrapolation,
            )
            G_oxide = _lamor_gibbs(T, oxide_row)
            a_oxide = act[oxide]
        else:
            G_oxide = 0.0
            a_oxide = 1.0

        # Reaction (1): oxide(l) -> n_gas * gas(g) + n_O2 * O2(g)
        dG = n_gas * G_gas + n_O2 * G_O2 - G_oxide
        Kp = np.exp(-dG / (R_J_MOL_K * T))

        p_gas = (Kp * a_oxide / (p_O2**n_O2)) ** (1.0 / n_gas)
        result[gas_name] = float(p_gas)

    return result
