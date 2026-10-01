"""IMCC-SF04 thin gas mass-action layer.

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
import re
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Callable, Literal, Mapping, NamedTuple, Sequence

import numpy as np
from scipy.optimize import brentq

from openimcc.kernel import ImccRefusal

if TYPE_CHECKING:
    import pandas as pd


# --------------------------------------------------------------------------- #
# Physical constants and reference states
# --------------------------------------------------------------------------- #

R_J_MOL_K = 8.314462618
_SUPERCOOLED_EXTENSION_REF_SUFFIX = "-SC-CP"
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


class ImccGasInvalidFugacityError(ImccRefusal):
    """Raised when the caller's bar-relative oxygen fugacity is invalid."""

    code = "imcc_gas_invalid_fO2"


class ImccGasOxygenBalanceError(ImccRefusal):
    """Raised when the oxygen-balance root is not monotone or bracketed."""

    code = "imcc_gas_oxygen_balance_failed"


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


@dataclass(frozen=True)
class ImccGasResult(MappingABC[str, float]):
    """Immutable gas pressures and channel metadata, including default omissions."""

    _values: Mapping[str, float]
    unit: str
    domain_flags: Mapping[str, str | None]
    provenance_class: Mapping[str, str]
    omitted_channels: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_values", MappingProxyType(dict(self._values)))
        object.__setattr__(
            self, "domain_flags", MappingProxyType(dict(self.domain_flags))
        )
        object.__setattr__(
            self,
            "provenance_class",
            MappingProxyType(dict(self.provenance_class)),
        )
        object.__setattr__(
            self,
            "omitted_channels",
            MappingProxyType(dict(self.omitted_channels)),
        )

    def __getitem__(self, species: str) -> float:
        return self._values[species]

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


@dataclass(frozen=True)
class SpeciesThermo:
    """Thermodynamic properties returned by :func:`species_thermo`.

    ``source_row_id`` is the table row's Ref value; ``source_table_id`` names
    the source table when provenance records one. ``derivatives_fit_implied``
    is true for every condensate row because its Cp, S, and H_app are
    derivatives of a fitted Gibbs-energy polynomial. JANAF-fitted condensates
    fit Phi (the Gibbs polynomial) only. Gas Shomate rows fit Cp, H, and S,
    so their derivatives are not fit-implied.
    """

    Cp_J_molK: float
    S_J_molK: float
    H_app_kJ_mol: float
    G_J_mol: float
    source_row_id: str
    source_table_id: str | None
    T_interval: int | None
    T_min: float
    T_max: float
    derivatives_fit_implied: bool = False


def _pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImccGasDataUnavailableError(
            f'install "openimcc[gas]" (pandas import failed: {exc})'
        ) from exc
    return pd


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

# Map retained gas species to a vaporization reaction. Condensed-parent rows use
#     parent_oxide(l) -> n_gas * gas(g) + n_O2 * O2(g).
# Sulfur rows use n_S2*S2(g) + n_O2*O2(g) -> gas(g), as documented below.
#
# Derivation: for a parent E_a O_b and target gas E_c O_d, element balance gives
# n_gas = a/c and n_O2 = (b - n_gas*d)/2.  A negative n_O2 puts O2 on the
# reactant side.  O(g) is the oxide-free dissociation 1/2 O2(g) -> O(g), encoded
# with an empty parent and n_O2 = -1/2.  All G degrees use elements in their
# 298 K standard states and ideal gas at 1 bar, so the elemental references
# cancel in Delta G degrees.  Unit check: stoichiometric coefficients multiply
# J/mol values, hence -Delta G degrees/(R*T) is dimensionless.  Sanity checks:
# Al2O3 -> 2 AlO + 1/2 O2 balances Al2O3, while
# Na2O + 1/2 O2 -> 2 NaO balances Na2O2.  For the TiO2 parent (a=1, b=2):
# Ti (c=1, d=0) gives n_gas=1, n_O2=(2-0)/2=1; TiO gives n_gas=1,
# n_O2=(2-1)/2=1/2; TiO2 gives n_gas=1, n_O2=0.
# For the retained association channels, the same balance gives Al2 from Al2O3
# as n_gas=2/2=1, n_O2=(3-1*0)/2=3/2.  The Si2 and Si3 equations are written
# per one SiO2 formula unit: Si2 uses n_gas=1/2 and n_O2=1, so multiplying by
# 2 gives 2 SiO2 = Si2 + 2 O2; Si3 uses n_gas=1/3 and n_O2=1, so multiplying
# by 3 gives 3 SiO2 = Si3 + 3 O2.  Unit check: these are dimensionless
# stoichiometric coefficients.
# For the caller-supplied Cr2O3 parent (a=2, b=3), Cr, CrO, CrO2 and CrO3 use
# n_gas=2 and n_O2=3/2, 1/2, -1/2 and -3/2 (Cr2O3(l) = 2 Cr(g) + 3/2 O2). The parent activity remains an input
# because Cr2O3 is outside the eight-parent melt basis; the same balance and
# mass-action equation apply without an internal phase-selection rule.
# For the caller-supplied V2O3 parent (a=2, b=3), V, VO and VO2 use
# n_gas=2 and n_O2=3/2, 1/2 and -1/2. For the caller-supplied NbO2 parent
# (a=1, b=2), Nb, NbO and NbO2 use n_gas=1 and n_O2=1, 1/2 and 0.
# For caller-supplied MnO, NiO and CoO parents (a=1, b=1), the atomic and
# monoxide channels use n_gas=1 and n_O2=1/2 and 0. These entries are data-free
# wiring: the public pack has the gas rows but no parent rows, while an external
# pack can provide both sides of each reaction.
# For caller-supplied P2O5(l) (a=2, b=5), n_gas=2/n_P and
# n_O2=(5-n_gas*n_O)/2: P, P2, P4, PO, PO2, P4O6 and P4O10 use
# (2, 5/2), (1, 5/2), (1/2, 5/2), (2, 3/2), (2, 1/2),
# (1/2, 1) and (1/2, 0), respectively. Each tuple balances one P2O5(l).
# Sulfur tuples use the caller's gas parent directly: n_S2 S2(g) + n_O2 O2(g)
# = species(g), with n_S2=n_S/2 and n_O2=n_O/2. Thus S, S2, S3, S4, S5,
# S6, S7, S8, SO, SO2, SO3 and SSO use (1/2, 0), (1, 0), (3/2, 0),
# (2, 0), (5/2, 0), (3, 0), (7/2, 0), (4, 0), (1/2, 1/2),
# (1/2, 1), (1/2, 3/2) and (1, 1/2), respectively. Each equation balances
# sulfur and oxygen; positive n_O2 is consumed on the reactant side.
_SF04_REACTIONS: dict[str, tuple[str, float, float]] = {
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
    # Appended after O2 so every earlier channel keeps its position.
    "Ti": ("TiO2", 1, 1.0),
    "TiO": ("TiO2", 1, 0.5),
    "TiO2": ("TiO2", 1, 0.0),
    "Al2": ("Al2O3", 1, 1.5),
    "Si2": ("SiO2", 0.5, 1.0),
    "Si3": ("SiO2", 1 / 3, 1.0),
    "Cr": ("Cr2O3", 2, 1.5),
    "CrO": ("Cr2O3", 2, 0.5),
    "CrO2": ("Cr2O3", 2, -0.5),
    "CrO3": ("Cr2O3", 2, -1.5),
    "V": ("V2O3", 2, 1.5),
    "VO": ("V2O3", 2, 0.5),
    "VO2": ("V2O3", 2, -0.5),
    "Nb": ("NbO2", 1, 1.0),
    "NbO": ("NbO2", 1, 0.5),
    "NbO2": ("NbO2", 1, 0.0),
    # Appended after the existing channels so every pre-existing mapping and
    # output position stays unchanged.
    "Na2O": ("Na2O", 1, 0.0),
    "K2O": ("K2O", 1, 0.0),
    "Mn": ("MnO", 1, 0.5),
    "MnO": ("MnO", 1, 0.0),
    "Ni": ("NiO", 1, 0.5),
    "NiO": ("NiO", 1, 0.0),
    "Co": ("CoO", 1, 0.5),
    "CoO": ("CoO", 1, 0.0),
    # Appended after every existing channel to preserve legacy output order.
    "P": ("P2O5", 2, 2.5),
    "P2": ("P2O5", 1, 2.5),
    "P4": ("P2O5", 0.5, 2.5),
    "PO": ("P2O5", 2, 1.5),
    "PO2": ("P2O5", 2, 0.5),
    "P4O6": ("P2O5", 0.5, 1.0),
    "P4O10": ("P2O5", 0.5, 0.0),
    "S": ("S2", 0.5, 0.0),
    "S2": ("S2", 1, 0.0),
    "S3": ("S2", 1.5, 0.0),
    "S4": ("S2", 2.0, 0.0),
    "S5": ("S2", 2.5, 0.0),
    "S6": ("S2", 3.0, 0.0),
    "S7": ("S2", 3.5, 0.0),
    "S8": ("S2", 4.0, 0.0),
    "SO": ("S2", 0.5, 0.5),
    "SO2": ("S2", 0.5, 1.0),
    "SO3": ("S2", 0.5, 1.5),
    "SSO": ("S2", 1.0, 0.5),
}

# Parent activity keys omit phase suffixes. A gas-phase parent's G° is read
# from the JANAF gas table; all other parent keys keep the liquid lookup.
_GAS_PHASE_PARENT_ROWS = {"S2": "S2(g)"}

IMCC_GAS_CHANNEL_SPECIES = tuple(_SF04_REACTIONS)

# Channels that a default evaluate_gas call includes only when the active
# datapack carries their gas row and their parent's standard-state row. The legacy
# tables selected by OPENIMCC_VAPOROCK_ROOT have Ti gas rows but no TiO2(l)
# row, while the public tables have Mn/Ni/Co gas rows but no MnO(l)/NiO(l)/CoO(l)
# rows. Optional channels therefore stay absent from a default call until an
# active pack supplies both sides. Original SF04 channels stay strict: a table
# missing one of their rows is broken and the default call keeps refusing on it
# rather than silently omitting it.
# The Na2O and K2O channels use the optional rule for alternate tables that
# predate their gas rows.
_DATAPACK_OPTIONAL_CHANNELS = frozenset(
    {
        "Na2O",
        "K2O",
        "Ti",
        "TiO",
        "TiO2",
        "Al2",
        "Si2",
        "Si3",
        "Cr",
        "CrO",
        "CrO2",
        "CrO3",
        "V",
        "VO",
        "VO2",
        "Nb",
        "NbO",
        "NbO2",
        "Mn",
        "MnO",
        "Ni",
        "NiO",
        "Co",
        "CoO",
        "P",
        "P2",
        "P4",
        "PO",
        "PO2",
        "P4O6",
        "P4O10",
        "S",
        "S2",
        "S3",
        "S4",
        "S5",
        "S6",
        "S7",
        "S8",
        "SO",
        "SO2",
        "SO3",
        "SSO",
    }
)


_REACTION_PARENT_OXIDES = frozenset(
    reaction[0] for reaction in _SF04_REACTIONS.values() if reaction[0]
)


def _default_reactions(
    available_parents: Sequence[str], datapack: ImccGasDatapack
) -> tuple[tuple[tuple[str, tuple[str, float, float]], ...], Mapping[str, str]]:
    """Return selected channels and optional channels skipped for missing rows.

    Unified availability rule: a channel is included only when its parent
    activity is supplied (O and O2 have no parent); an optional channel also
    requires both its gas row and parent standard-state row in the
    active datapack. Gas-only source rows stay outside the runtime channel set
    until this same rule can be satisfied. Missing optional rows are returned
    by channel name; channels whose parent activity was not supplied are not
    reported. Channel order is the ``_SF04_REACTIONS`` order.
    """
    parents = set(available_parents)
    gas_rows = set(datapack.gas_df.index)
    oxide_rows = set(datapack.oxide_df.index)
    selected = []
    omitted: dict[str, str] = {}
    for name, reaction in _SF04_REACTIONS.items():
        oxide = reaction[0]
        if oxide and oxide not in parents:
            continue
        if name in _DATAPACK_OPTIONAL_CHANNELS:
            missing_rows = []
            gas_row = f"{name}(g)"
            gas_parent_row = _GAS_PHASE_PARENT_ROWS.get(oxide)
            oxide_row = gas_parent_row or f"{oxide}(l)"
            if gas_row not in gas_rows:
                missing_rows.append(gas_row)
            parent_rows = gas_rows if gas_parent_row else oxide_rows
            if oxide_row != gas_row and oxide_row not in parent_rows:
                missing_rows.append(oxide_row)
            if missing_rows:
                omitted[name] = (
                    "missing from active table: " + ", ".join(missing_rows)
                )
                continue
        selected.append((name, reaction))
    return tuple(selected), omitted


def default_gas_channels(
    parent_oxides: Sequence[str], datapack: ImccGasDatapack | None = None
) -> tuple[tuple[tuple[str, tuple[str, float, float]], ...], Mapping[str, str]]:
    """Return default gas channels and optional channels omitted for missing rows.

    The returned channels retain the reaction order and shape used by
    :func:`evaluate_gas`; ``omitted`` maps optional channel names to the
    missing rows that prevented their inclusion. When ``datapack`` is omitted,
    the packaged tables (or the explicit VapoRock override) are loaded.
    """
    active_pack = load_gas_datapack() if datapack is None else datapack
    return _default_reactions(tuple(parent_oxides), active_pack)

# These authority labels mirror the row-level classes in PROVENANCE.yaml. A
# reaction is only as authoritative as its least-authoritative input row, so a
# potassium channel inherits the K2O(l) secondary-transcription flag while the
# other retained channels retain their JANAF/Lamoreaux classes. The Na2O/K2O
# gas rows are NASA/Gurvich fits. The Mn/Ni/Co atomic rows are public JANAF
# fits; the monoxide rows are external-pack-only and carry no public row
# provenance.
_EXTERNAL_PACK_GAS_SPECIES = frozenset({"MnO", "NiO", "CoO"})
_GAS_PROVENANCE_AUTHORITY = {
    species: (
        "external_datapack"
        if species in _EXTERNAL_PACK_GAS_SPECIES
        else "janaf_fitted"
    )
    for species in (*IMCC_GAS_CHANNEL_SPECIES, "Mn", "Ni", "Co")
}
_GAS_PROVENANCE_AUTHORITY.update(
    {"Na2O": "nasa_glenn_fitted", "K2O": "nasa_glenn_fitted"}
)
_ION_GAS_SPECIES = (
    "Na+", "K+", "Ca+", "e-",
    "Na-", "K-", "O-", "Al-", "Fe-", "Si-", "Ti-", "O2-", "AlO-", "AlO2-", "KO-", "NaO-",
    "Cr-", "V-", "Nb-",
)
_GAS_PROVENANCE_AUTHORITY.update(
    {species: "janaf_fitted_ionisation" for species in _ION_GAS_SPECIES}
)
_OXIDE_PROVENANCE_AUTHORITY = {
    "MnO": "external_datapack",
    "NiO": "external_datapack",
    "CoO": "external_datapack",
    "P2O5": "external_datapack",
    "MgO": "lam1987_transcribed",
    "CaO": "lam1987_transcribed",
    "Al2O3": "lam1987_transcribed",
    "SiO2": "lam1987_transcribed",
    "SiO2(cr)": "lam1987_transcribed",
    "Na2O": "janaf_fitted",
    "K2O": "secondary_transcription_unverified_primary",
    "FeO": "janaf_transcribed",
    "TiO2": "janaf_fitted",
    "Cr2O3": "janaf_fitted",
    "V2O3": "janaf_fitted",
    "NbO2": "janaf_fitted",
}
# Mirrors the source table identifiers recorded for packaged rows in
# data/gas/PROVENANCE.yaml. Rows without a table identifier there return None.
_OXIDE_SOURCE_TABLE_IDS = {
    "Na2O": "Na-013",
    "K2O": None,
    "MgO": "LH87 Table 2",
    "CaO": "LH87 Table 2",
    "Al2O3": "LH87 Table 2 / Table 3",
    "SiO2": "LH87 Table 2",
    "SiO2(cr)": "LH87 Table 2",
    "FeO": "Fe-019",
    "TiO2": "O-044",
    "Cr2O3": "Cr-015",
    "V2O3": "O-063",
    "NbO2": "Nb-013",
}
_PROVENANCE_AUTHORITY_RANK = {
    "secondary_transcription_unverified_primary": 0,
    "lam1984_transcribed": 1,
    "lam1987_transcribed": 1,
    "janaf_transcribed": 2,
    "janaf_fitted": 3,
    "janaf_fitted_ionisation": 3,
    "nasa_glenn_fitted": 3,
    "external_datapack": 2,
}


def gas_species_provenance(species: str) -> dict[str, str | None]:
    """Return the row authorities feeding one retained gas channel.

    The overall authority is the least-authoritative row in that reaction.
    """
    if species in _ION_GAS_SPECIES:
        return {
            "species_name": species,
            "authority": "janaf_fitted_ionisation",
            "gas_authority": "janaf_fitted_ionisation",
            "condensate_authority": None,
        }
    if species not in _SF04_REACTIONS:
        raise ImccGasSpeciesNotFoundError(
            f"unknown retained gas species {species!r}"
        )
    oxide, _, _ = _SF04_REACTIONS[species]
    gas_authority = _GAS_PROVENANCE_AUTHORITY[species]
    oxide_authority = (
        None
        if oxide in _GAS_PHASE_PARENT_ROWS
        else _OXIDE_PROVENANCE_AUTHORITY.get(oxide)
    )
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

# EXTRAPOLATED-INFORMATIONAL labels for default channels whose gas or parent
# table misses part of the workbook grid. Major-oxide channels retain their
# Lamoreaux rows and therefore keep their established out-of-interval flags.
IMCC_GAS_WORKBOOK_EXTRAPOLATION_LABELS: dict[str, str] = {
    "SiO": "SiO2(l) [1996, 3000] K misses workbook T < 1996 K",
    "Mg": "MgO(l) [3100, 3500] K lies above the whole workbook grid",
    "MgO": "MgO(l) [3100, 3500] K lies above the whole workbook grid",
    "SiO2": "SiO2(l) [1996, 3000] K misses workbook T < 1996 K",
    "AlO": "Al2O3(l) [2327, 3000] K misses workbook T < 2327 K",
    "AlO2": "Al2O3(l) [2327, 3000] K misses workbook T < 2327 K",
    "Al2O": "Al2O3(l) [2327, 3000] K misses workbook T < 2327 K",
    "Al2O2": "Al2O3(l) [2327, 3000] K misses workbook T < 2327 K",
    "Al2": "Al2O3(l) [2327, 3000] K misses workbook T < 2327 K",
    "Si": "SiO2(l) [1996, 3000] K misses workbook T < 1996 K",
    "Si2": "SiO2(l) [1996, 3000] K misses workbook T < 1996 K",
    "Si3": "SiO2(l) [1996, 3000] K misses workbook T < 1996 K",
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
    "Na2O",
    "K2O",
    "O2",
    "Cr",
    "CrO",
    "CrO2",
    "CrO3",
    "Ti",
    "TiO",
    "TiO2",
    "V",
    "VO",
    "VO2",
    "Nb",
    "NbO",
    "NbO2",
    "Mn",
    "MnO",
    "Ni",
    "NiO",
    "Co",
    "CoO",
    "S",
    "S2",
    "S3",
    "S4",
    "S5",
    "S6",
    "S7",
    "S8",
    "SO",
    "SO2",
    "SO3",
    "SSO",
)

# Closure ledger for species outside the retained channel set. These entries
# still lack a gas G(T) row in the vendored source set or require a model
# convention not present in the layer. Na2O/K2O left this ledger when their
# NASA/Gurvich gas rows joined the package. Mn/Ni/Co now have retained,
# optional channels; the public-only load skips them because it has no parent
# rows, while an external pack can activate them.
IMCC_GAS_NO_JANAF_ROWS: dict[str, str] = {
    "Zn": (
        "needs Zn(g) and ZnO(l) standard-Gibbs rows plus ZnO in the melt-oxide basis"
    ),
    "ZnO": (
        "needs ZnO(g) and ZnO(l) standard-Gibbs rows plus ZnO in the melt-oxide basis"
    ),
}

_GAS_IONIZATION_PAIRS = (("Na", "Na+"), ("K", "K+"), ("Ca", "Ca+"))
_GAS_ELECTRON_ATTACHMENTS = (
    ("Na", "Na-"),
    ("K", "K-"),
    ("O", "O-"),
    ("Al", "Al-"),
    ("Fe", "Fe-"),
    ("Si", "Si-"),
    ("Ti", "Ti-"),
    ("O2", "O2-"),
    ("AlO", "AlO-"),
    ("AlO2", "AlO2-"),
    ("KO", "KO-"),
    ("NaO", "NaO-"),
    ("Cr", "Cr-"),
    ("V", "V-"),
    ("Nb", "Nb-"),
)
_ION_PROVENANCE_CLASS = "janaf_fitted_ionisation"

IMCC_GAS_INCOMPLETE_PARENT_SPECIES: dict[str, str] = {}

# These retained channels are wired but cannot run from the public pack: only
# the atomic gas rows are public, while the monoxide gas and parent rows arrive
# through an external datapack.
IMCC_GAS_PUBLIC_ONLY_PARENT_SPECIES: dict[str, str] = {
    "Mn": (
        "public pack has Mn(g) but no MnO(g) or MnO(l); "
        "external pack supplies both"
    ),
    "Ni": (
        "public pack has Ni(g) but no NiO(g) or NiO(l); "
        "external pack supplies both"
    ),
    "Co": (
        "public pack has Co(g) but no CoO(g) or CoO(l); "
        "external pack supplies both"
    ),
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

# Research-pack note: each JANAF parent row passes its declared-range C4 check
# through 3000 K (Al 2500 K, Si 1800 K, Mg/Ca 2200 K). The default TiO2, Cr2O3,
# and V2O3 parents have labelled continuations below their fitted rows.
ELEMENT_STATUS: dict[str, dict[str, object]] = {
    "O": {
        "status": "input (fO2 pinned)",
        "criteria": {"C1": False, "C2": True, "C3": False, "C4": False},
        "validation": "unvalidated",
        "reason": "O is input (fO2 pinned), so C1, C3 and C4 do not apply.",
        "c2_candidates": (),
    },
    "P": {
        "status": "gas-partial",
        "criteria": {"C1": False, "C2": True, "C3": False, "C4": False},
        "validation": "unvalidated",
        "reason": (
            "JANAF P gas rows cover 500-3000 K; no public evaluated P2O5(l) "
            "G(T) function was found and JANAF lists P4O10(cr) only. Channels "
            "need an external, source-rated P2O5(l) standard state and "
            "caller-supplied a(P2O5); OpenIMCC provides no "
            "pyrolysis-temperature melt "
            "activity model."
        ),
        "c3_ion_bound": {
            "status": "not computed",
            "reason": "not computed: ionized phosphorus channels are outside this neutral gas-only change",
        },
        "c2_candidates": (
            ("O-004", "PO"), ("O-032", "PO2"),
            ("O-087", "P4O6"), ("O-095", "P4O10"),
        ),
    },
    "S": {
        "status": "gas-complete-melt-pending",
        "criteria": {"C1": False, "C2": True, "C3": False, "C4": True},
        "validation": "unvalidated",
        "reason": (
            "JANAF sulfur gas rows and the S2(g) parent cover 500-3000 K; "
            "callers supply a(S2)=f(S2)/p° on the 1-bar gas reference. "
            "OpenIMCC provides no pyrolysis-temperature sulfur melt "
            "activity model."
        ),
        "c3_ion_bound": {
            "status": "not computed",
            "reason": "not computed: ionized sulfur channels are outside this neutral gas-only change",
        },
        "c2_candidates": (
            ("O-010", "SO"), ("O-034", "SO2"),
            ("O-058", "SO3"), ("O-011", "SSO"),
        ),
    },
    "Si": {
        "status": "gas-partial",
        "criteria": {"C1": True, "C2": True, "C3": True, "C4": False},
        "validation": "validated",
        "reason": "SiO2 liquid row starts at 1996 K; joint C3 closure-screen maximum 5.545e-14 at 2800 K and fO2=1e-4 within the <=1-bar neutral-pressure domain; unmodeled positive molecular ions and thermal electrons from walls or other sources remain outside the estimate.",
        "c2_candidates": (("O-012", "SiO"), ("O-040", "SiO2")),
        "c3_ion_bound": {
            "max_ratio": 5.544921857331724e-14,
            "isolated_bound": 714.576268991371,
            "temperature_K": 2800.0,
            "fO2": 1.0e-04,
            "neutral_pressure_bar": 3.659448750056158e-06,
            "element_total_pressure_bar": 0.29379142444188944,
            "joint_ion_pressure_bar": 1.6290504908844544e-14,
            "electron_pressure_bar": 8.434764281794561e-05,
            "K_ion": 3.754843374576903e-13,
            "parent_oxide": "SiO2",
            "source_tables": {"cation": "Si-006", "neutral": "Si-005", "electron": "D-020"},
            # No JANAF manifest entry exists for D-020; this is the SHA-256 of
            # the committed data-src/janaf/D-020.txt record.
            "upstream_sha256": {
                "Si-006": "6d81642518890fb4c67840b3bb103582ad037e800e312cd2fb5cf227cb7ffd58",
                "D-020": "c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd",
            },
            "user_agent": "openimcc-janaf-vendor/1.0",
        },
    },
    "Mg": {
        "status": "gas-partial",
        "criteria": {"C1": True, "C2": True, "C3": True, "C4": False},
        "validation": "unvalidated",
        "reason": "MgO liquid row starts at 3100 K; joint C3 closure-screen maximum 1.117e-07 at 2800 K and fO2=1e-4 within the <=1-bar neutral-pressure domain; unmodeled positive molecular ions and thermal electrons from walls or other sources remain outside the estimate.",
        "c2_candidates": (("Mg-011", "MgO"),),
        "c3_ion_bound": {
            "max_ratio": 1.1170685493345971e-07,
            "isolated_bound": 7.348653080974237e-05,
            "temperature_K": 2800.0,
            "fO2": 1.0e-04,
            "neutral_pressure_bar": 0.001769224939073943,
            "element_total_pressure_bar": 0.0017940263652892997,
            "joint_ion_pressure_bar": 2.004050429341738e-10,
            "electron_pressure_bar": 8.434764281794561e-05,
            "K_ion": 9.55429273406838e-12,
            "parent_oxide": "MgO",
            "source_tables": {"cation": "Mg-006", "neutral": "Mg-005", "electron": "D-020"},
            "upstream_sha256": {
                "Mg-006": "1525bdbc6926d9fab8c79ee002d1968c4d5597ca446049149886d79119787c3e",
                "D-020": "c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd",
            },
            "user_agent": "openimcc-janaf-vendor/1.0",
        },
    },
    "Fe": {
        "status": "complete",
        "criteria": {"C1": True, "C2": True, "C3": True, "C4": True},
        "validation": "unvalidated",
        "reason": "FeO and neutral gas rows cover the domain; joint C3 closure-screen maximum 3.342e-08 at 2800 K and fO2=1e-4 within the <=1-bar neutral-pressure domain; unmodeled positive molecular ions and thermal electrons from walls or other sources remain outside the estimate.",
        "c2_candidates": (("Fe-021", "FeO"),),
        "c3_ion_bound": {
            "max_ratio": 3.342322955453911e-08,
            "isolated_bound": 9.17341452911598e-06,
            "temperature_K": 2800.0,
            "fO2": 1.0e-04,
            "neutral_pressure_bar": 0.034690158619771164,
            "element_total_pressure_bar": 0.035921399382343955,
            "joint_ion_pressure_bar": 1.2006091774763614e-09,
            "electron_pressure_bar": 8.434764281794561e-05,
            "K_ion": 2.9192300668238232e-12,
            "parent_oxide": "FeO",
            "source_tables": {"cation": "Fe-009", "neutral": "Fe-008", "electron": "D-020"},
            "upstream_sha256": {
                "Fe-009": "d4979c41394748f0ffea48163a8a792e5e4f763989426f6cd4ba198ee3e532ab",
                "D-020": "c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd",
            },
            "user_agent": "openimcc-janaf-vendor/1.0",
        },
    },
    "Ca": {
        "status": "gas-partial",
        "criteria": {"C1": True, "C2": True, "C3": True, "C4": False},
        "validation": "unvalidated",
        "reason": "CaO liquid row starts at 2900 K; fitted Ca+ channel maximum p(Ca+)/neutral Ca gas 6.310e-05 at 2800 K and fO2=1e-4 within the <=1-bar neutral-pressure domain, below the 1e-4 C3 threshold.",
        "c2_candidates": (("Ca-030", "CaO"),),
        "c3_ion_bound": {
            "max_ratio": 6.310119343347233e-05,
            "isolated_bound": 70.1162365014019,
            "temperature_K": 2800.0,
            "fO2": 1.0e-04,
            "neutral_pressure_bar": 1.6709964790838675e-06,
            "element_total_pressure_bar": 1.724639262456349e-06,
            "joint_ion_pressure_bar": 1.0882679570321912e-10,
            "electron_pressure_bar": 8.434764281794561e-05,
            "K_ion": 5.493299242634703e-09,
            "parent_oxide": "CaO",
            "source_tables": {"cation": "Ca-007", "neutral": "Ca-006", "electron": "D-020"},
            "upstream_sha256": {
                "Ca-007": "ddd7d08df4aaf34beeaead89af83e7281f850eb453e859483f16520d61e98724",
                "D-020": "c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd",
            },
            "user_agent": "openimcc-janaf-vendor/1.0",
        },
    },
    "Al": {
        "status": "gas-partial",
        "criteria": {"C1": True, "C2": True, "C3": True, "C4": False},
        "validation": "unvalidated",
        "reason": "Al2O3 liquid row starts at 2327 K; joint C3 closure-screen maximum 4.135e-06 at 2800 K and fO2=1e-4 within the <=1-bar neutral-pressure domain; unmodeled positive molecular ions and thermal electrons from walls or other sources remain outside the estimate.",
        "c2_candidates": (
            ("Al-074", "AlO"),
            ("Al-077", "AlO2"),
            ("Al-092", "Al2O"),
            ("Al-094", "Al2O2"),
        ),
        "c3_ion_bound": {
            "max_ratio": 4.135469655015644e-06,
            "isolated_bound": 9021443.39824646,
            "temperature_K": 2800.0,
            "fO2": 1.0e-04,
            "neutral_pressure_bar": 4.787348402255381e-06,
            "element_total_pressure_bar": 1.3589244539678473e-05,
            "joint_ion_pressure_bar": 5.619790842842736e-11,
            "electron_pressure_bar": 8.434764281794561e-05,
            "K_ion": 9.90143333834541e-10,
            "parent_oxide": "Al2O3",
            "source_tables": {"cation": "Al-006", "neutral": "Al-005", "electron": "D-020"},
            "upstream_sha256": {
                "Al-006": "7e4e45bc108863c4baed5ca1bffe6b931a78a050af86d50a2fcbbe7b9251c6dd",
                "D-020": "c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd",
            },
            "user_agent": "openimcc-janaf-vendor/1.0",
        },
    },
    "Ti": {
        "status": "gas-partial",
        "criteria": {"C1": True, "C2": True, "C3": True, "C4": False},
        "validation": "unvalidated",
        "reason": "A labelled constant-Cp TiO2(l) continuation covers 1200-1500 K below the glass branch; Ti(g), TiO(g), and TiO2(g) still start at 1500 K, leaving a lower C4 gap; joint C3 closure-screen maximum 2.468e-09 at 2500 K and fO2=1e-8 within the <=1-bar neutral-pressure domain; unmodeled positive molecular ions and thermal electrons from walls or other sources remain outside the estimate.",
        "c2_candidates": (("O-022", "TiO"), ("O-046", "TiO2")),
        "c3_ion_bound": {
            "max_ratio": 2.468021079468733e-09,
            "isolated_bound": 143063233.9706761,
            "temperature_K": 2500.0,
            "fO2": 1.0e-08,
            "neutral_pressure_bar": 3.718576242708182e-08,
            "element_total_pressure_bar": 5.027603220057954e-06,
            "joint_ion_pressure_bar": 1.2408230726307911e-14,
            "electron_pressure_bar": 2.342325071257254e-05,
            "K_ion": 7.815924166451596e-12,
            "parent_oxide": "TiO2",
            "source_tables": {"cation": "Ti-007", "neutral": "Ti-006", "electron": "D-020"},
            "upstream_sha256": {
                "Ti-007": "ea066f88e882e8f102f5499c5bc6b58ffdeb8a0d9f50295afe60af0236eb5c1e",
                "D-020": "c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd",
            },
            "user_agent": "openimcc-janaf-vendor/1.0",
        },
    },
    "Na": {
        "status": "complete-except-ions",
        "criteria": {"C1": True, "C2": True, "C3": False, "C4": True},
        "validation": "unvalidated",
        "reason": "Na2O(g) intervals cover 500-3000 K and JANAF Na-013 Na2O(l) intervals cover 1200-3000 K, including its supercooled-liquid branch; fitted Na+ channel maximum p(Na+)/neutral Na gas 9.261e-04 at 2800 K and fO2=1e-4 remains above 1e-4; ions are opt-in and omitted from default results.",
        "c2_candidates": (("Na-008", "NaO"),),
        "c3_ion_bound": {
            "max_ratio": 0.0009261148973454778,
            "isolated_bound": 0.0009246793240051798,
            "temperature_K": 2800.0,
            "fO2": 1.0e-04,
            "neutral_pressure_bar": 0.09147641574868166,
            "element_total_pressure_bar": 0.09159309879517875,
            "joint_ion_pressure_bar": 8.482573328825118e-05,
            "electron_pressure_bar": 8.434764281794561e-05,
            "K_ion": 7.82152491941164e-08,
            "parent_oxide": "Na2O",
            "source_tables": {
                "cation": "Na-006",
                "neutral": "Na-005",
                "electron": "D-020",
            },
            "upstream_sha256": {
                "Na-006": (
                    "1324311c226aaef128378768ce75d1f45c15949219359b3388efda2822db1159"
                ),
                "D-020": (
                    "c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd"
                ),
            },
            "user_agent": "openimcc-janaf-vendor/1.0",
        },
    },
    "K": {
        "status": "complete-except-ions",
        "criteria": {"C1": True, "C2": True, "C3": False, "C4": True},
        "validation": "validated",
        "reason": "K2O(g) intervals cover 500-3000 K and K2O(l) covers the C4 domain; fitted K+ channel maximum p(K+)/neutral K gas 4.325e-02 at 1200 K and fO2=1e-4 remains above 1e-4; ions are opt-in and omitted from default results.",
        "c2_candidates": (("K-008", "KO"),),
        "c3_ion_bound": {
            "max_ratio": 0.04324736798988459,
            "isolated_bound": 0.18968918064603676,
            "temperature_K": 1200.0,
            "fO2": 1.0e-04,
            "neutral_pressure_bar": 1.7487620205734108e-15,
            "element_total_pressure_bar": 1.7543022029391172e-15,
            "joint_ion_pressure_bar": 7.58689529359732e-17,
            "electron_pressure_bar": 2.257363747459153e-16,
            "K_ion": 9.793432262395198e-18,
            "parent_oxide": "K2O",
            "source_tables": {
                "cation": "K-006",
                "neutral": "K-005",
                "electron": "D-020",
            },
            "upstream_sha256": {
                "K-006": (
                    "27cd2083096e8faf6281a319dfb74e27c90ac472a965fb9366cdc43242e963b4"
                ),
                "D-020": (
                    "c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd"
                ),
            },
            "user_agent": "openimcc-janaf-vendor/1.0",
        },
    },
    "Cr": {
        "status": "gas-partial",
        "criteria": {"C1": False, "C2": True, "C3": True, "C4": False},
        "validation": "unvalidated",
        "reason": "A labelled constant-Cp Cr2O3(l) continuation covers the parent interval below 1900 K; Cr2O3 activity is caller-supplied and Cr(g) ends at 2900 K, leaving the upper C4 gap; joint C3 closure-screen maximum 1.064e-06 at 2800 K and fO2=1e-4 within the <=1-bar neutral-pressure domain; unmodeled positive molecular ions and thermal electrons from walls or other sources remain outside the estimate.",
        "c2_candidates": (("Cr-010", "CrO"), ("Cr-011", "CrO2"), ("Cr-012", "CrO3")),
        "c3_ion_bound": {
            "max_ratio": 1.0635080893815698e-06,
            "isolated_bound": 0.0675715822588812,
            "temperature_K": 2800.0,
            "fO2": 1.0e-04,
            "neutral_pressure_bar": 0.0023340760987797158,
            "element_total_pressure_bar": 0.0038602274477255102,
            "joint_ion_pressure_bar": 4.105383117508851e-09,
            "electron_pressure_bar": 8.434764281794561e-05,
            "K_ion": 1.4835822576971673e-10,
            "parent_oxide": "Cr2O3",
            "source_tables": {"cation": "Cr-006", "neutral": "Cr-005", "electron": "D-020"},
            "upstream_sha256": {
                "Cr-006": "3767ef70fb010220ba12abb9131d157617d5466b8aac2c384734f2acbada8fa3",
                "D-020": "c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd",
            },
            "user_agent": "openimcc-janaf-vendor/1.0",
        },
        "c2_screen_maxima": {
            "Cr": {
                "max_ratio": 1.0, "temperature_K": 1400.0,
                "fO2": 1.0e-12, "dominant": "Cr",
            },
            "CrO": {
                "max_ratio": 1.0, "temperature_K": 2100.0,
                "fO2": 1.0e-6, "dominant": "CrO",
            },
            "CrO2": {
                "max_ratio": 1.0, "temperature_K": 1200.0,
                "fO2": 1.0e-12, "dominant": "CrO2",
            },
            "CrO3": {
                "max_ratio": 1.0, "temperature_K": 1200.0,
                "fO2": 1.0e-8, "dominant": "CrO3",
            },
        },
    },
    "V": {
        "status": "gas-partial",
        "criteria": {"C1": False, "C2": True, "C3": True, "C4": False},
        "validation": "unvalidated",
        "reason": "A labelled constant-Cp V2O3(l) continuation covers 1200-1500 K below the glass branch; V(g), VO(g), and VO2(g) still start at 1500 K, leaving a lower C4 gap; activity is caller-supplied; joint C3 closure-screen maximum 1.335e-08 at 2500 K and fO2=1e-8 within the <=1-bar neutral-pressure domain; unmodeled positive molecular ions and thermal electrons from walls or other sources remain outside the estimate.",
        "c2_candidates": (("O-026", "VO"), ("O-076", "VO2")),
        "c3_ion_bound": {
            "max_ratio": 1.3348601433686416e-08,
            "isolated_bound": 244.34080490083159,
            "temperature_K": 2500.0,
            "fO2": 1.0e-08,
            "neutral_pressure_bar": 7.700127466601133e-06,
            "element_total_pressure_bar": 0.00012134647130830025,
            "joint_ion_pressure_bar": 1.6198056808787642e-12,
            "electron_pressure_bar": 2.3423250712572525e-05,
            "K_ion": 4.9273359088456654e-12,
            "parent_oxide": "V2O3",
            "source_tables": {"cation": "V-006", "neutral": "V-005", "electron": "D-020"},
            "upstream_sha256": {
                "V-006": "c37e63246365b3335b5c2efdabf501741249c50357b6514a1a94e96f14ca2e1c",
                "D-020": "c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd",
            },
            "user_agent": "openimcc-janaf-vendor/1.0",
        },
        "c2_screen_maxima": {
            "V": {
                "max_ratio": 1.0, "temperature_K": 2200.0,
                "fO2": 1.0e-12, "dominant": "V",
            },
            "VO": {
                "max_ratio": 1.0, "temperature_K": 2000.0,
                "fO2": 1.0e-12, "dominant": "VO",
            },
            "VO2": {
                "max_ratio": 1.0, "temperature_K": 1200.0,
                "fO2": 1.0e-12, "dominant": "VO2",
            },
        },
    },
    "Nb": {
        "status": "gas-partial",
        "criteria": {"C1": False, "C2": True, "C3": True, "C4": False},
        "validation": "unvalidated",
        "reason": "Nb-013 is liquid from 1000 K and its condensate intervals cover 1200-3000 K; Nb(g), NbO(g), and NbO2(g) start at 1500 K, leaving a gas C4 gap; activity is caller-supplied; joint C3 closure-screen maximum 1.356e-12 at 2500 K and fO2=1e-8 within the <=1-bar neutral-pressure domain; unmodeled positive molecular ions and thermal electrons from walls or other sources remain outside the estimate.",
        "c2_candidates": (("Nb-011", "NbO"), ("Nb-015", "NbO2")),
        "c3_ion_bound": {
            "max_ratio": 1.3558676547446381e-12,
            "isolated_bound": 3202840578.16235,
            "temperature_K": 2500.0,
            "fO2": 1.0e-08,
            "neutral_pressure_bar": 2.6011334891258776e-11,
            "element_total_pressure_bar": 1.722830561352621e-06,
            "joint_ion_pressure_bar": 2.3359302327435667e-18,
            "electron_pressure_bar": 2.3423250712572525e-05,
            "K_ion": 2.1035090939149642e-12,
            "parent_oxide": "NbO2",
            "source_tables": {"cation": "Nb-006", "neutral": "Nb-005", "electron": "D-020"},
            "upstream_sha256": {
                "Nb-006": "afc5e0d57e2be188eaae3d111ac71adb7df586121c51a0a1c1d67206d9473380",
                "D-020": "c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd",
            },
            "user_agent": "openimcc-janaf-vendor/1.0",
        },
        "c2_screen_maxima": {
            "Nb": {"max_ratio": 0.9601449643935895, "temperature_K": 3000.0, "fO2": 1.0e-12, "dominant": "NbO"},
            "NbO": {"max_ratio": 1.0, "temperature_K": 2200.0, "fO2": 1.0e-12, "dominant": "NbO"},
            "NbO2": {"max_ratio": 1.0, "temperature_K": 1200.0, "fO2": 1.0e-12, "dominant": "NbO2"},
        },
    },
    "Mn": {
        "status": "gas-partial",
        "criteria": {"C1": False, "C2": True, "C3": False, "C4": False},
        "validation": "unvalidated",
        "reason": "Only Mn(g) has a public row; the MnO gas and MnO liquid parents are absent.",
        "c2_candidates": (),
        "c3_ion_bound": {
            "status": "not computed",
            "reason": "not computed: p(E) needs a parent liquid row that is only available from an external private pack",
            "source_tables": {"cation": "Mn-006", "neutral": "Mn-005", "electron": "D-020"},
            "upstream_sha256": {
                "Mn-006": "e90269050c0e94dae563a80fcb120b812af5382fada2c54aa6b951264fa56542",
                "D-020": "c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd",
            },
            "user_agent": "openimcc-janaf-vendor/1.0",
        },
    },
    "Ni": {
        "status": "gas-partial",
        "criteria": {"C1": False, "C2": True, "C3": False, "C4": False},
        "validation": "unvalidated",
        "reason": "Only Ni(g) has a public row; the NiO gas and NiO liquid parents are absent.",
        "c2_candidates": (),
        "c3_ion_bound": {
            "status": "not computed",
            "reason": "not computed: p(E) needs a parent liquid row that is only available from an external private pack",
            "source_tables": {"cation": "Ni-006", "neutral": "Ni-005", "electron": "D-020"},
            "upstream_sha256": {
                "Ni-006": "1719b0562921ca8e79bc0250e34812eb331440e00864853629c025eedf92fd16",
                "D-020": "c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd",
            },
            "user_agent": "openimcc-janaf-vendor/1.0",
        },
    },
    "Co": {
        "status": "gas-partial",
        "criteria": {"C1": False, "C2": True, "C3": False, "C4": False},
        "validation": "unvalidated",
        "reason": "Only Co(g) has a public row; the CoO gas and CoO liquid parents are absent.",
        "c2_candidates": (),
        "c3_ion_bound": {
            "status": "not computed",
            "reason": "not computed: p(E) needs a parent liquid row that is only available from an external private pack",
            "source_tables": {"cation": "Co-006", "neutral": "Co-005", "electron": "D-020"},
            "upstream_sha256": {
                "Co-006": "28ba36fd244128e314eb794af7bf4aadcd3de6e87bfde3bc36ee43e6b033ee20",
                "D-020": "c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd",
            },
            "user_agent": "openimcc-janaf-vendor/1.0",
        },
    },
}


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
    pd = _pandas()

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


def _outside_interval_flag(
    row_name: str, T: float, row: pd.Series, *, label: str = "G(T)"
) -> str | None:
    low = float(row["T_min"])
    high = float(row["T_max"])
    if T < low or T > high:
        return (
            f"T={T} K outside declared {label} interval for {row_name!r} "
            f"[{low:g}, {high:g}] K"
        )
    return None


def _nearest_interval_row(
    df: pd.DataFrame, species: str, T: float, allow_extrapolation: bool = False
) -> pd.Series:
    """Select one thermodynamic interval for ``species`` at temperature ``T``.

    Selection inside this function: among rows with ``T_min <= T``, take the
    one with the largest ``T_min``. If every ``T_min`` is ``> T``, take the
    row with the smallest ``T_min``: the interval nearest to ``T``, because
    low-temperature intervals are appended after the rows they extend. Ties on
    ``T_min`` resolve to the first such row in file order. Then,
    unless ``allow_extrapolation=True``, raise
    ``ImccGasTemperatureOutsideDomainError`` when ``T`` is outside that
    selected row's ``[T_min, T_max]``. That strict-mode refusal is the V2
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
    # Largest T_min that is <= T; below every T_min, the lowest interval.
    # Masking with -inf rather than multiplying by the boolean mask keeps a
    # valid row whose T_min is 0 from losing to an invalid row (0 * False == 0).
    valid = t_mins <= T
    if np.any(valid):
        idx = int(np.argmax(np.where(valid, t_mins, -np.inf)))
    else:
        idx = int(np.argmin(t_mins))
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
    if oxide not in df.index:
        raise ImccGasSpeciesNotFoundError(
            f"no condensate G(T) row for oxide {oxide!r}"
        )
    return _nearest_interval_row(df, oxide, T, allow_extrapolation)


def _add_ion_channels(
    values: dict[str, float],
    domain_flags: dict[str, str | None],
    provenance_class: dict[str, str],
    datapack: ImccGasDatapack,
    T: float,
    allow_extrapolation: bool,
) -> None:
    """Append fitted charge channels and close electroneutrality.

    JANAF gives all gas species, ions and electrons the same 1-bar ideal-gas
    standard state. Ion formation functions use JANAF's elemental reference
    states. D-020 treats e-(g) as a monatomic ideal gas and tabulates
    H°(T)-H°(0). Use the printed ion rows with that electron row; do not apply
    the +6.197 kJ/mol conversion some ion-table notes specify for the alternate
    convention that excludes the electron. For M(g) = M+(g) + e-(g),
    K_M = p(M+) p(e-)/p(M), so
    p(M+) = K_M p(M)/p(e-). For A(g) + e-(g) = A-(g),
    K_A = p(A-)/(p(A) p(e-)), so p(A-) = K_A p(A) p(e-).
    Every p here is the dimensionless pressure p/p° (numerically bar because
    p° = 1 bar). Unit check: both K values and each product are dimensionless.
    Electroneutrality gives p(e-) + Σ K_A p(A) p(e-) = Σ K_M p(M)/p(e-),
    hence p(e-) = sqrt(Σ K_M p(M) / (1 + Σ K_A p(A))). With no negative ions
    this reduces to the positive-only square-root closure. The computed
    residual is algebraically zero up to floating-point rounding.
    """
    row_cache: dict[str, tuple[float, pd.Series, str | None]] = {}

    def gibbs(species: str) -> tuple[float, pd.Series, str | None]:
        row_name = f"{species}(g)"
        if row_name not in row_cache:
            row = _nearest_interval_row(
                datapack.gas_df, row_name, T, allow_extrapolation
            )
            row_cache[row_name] = (
                _janaf_gibbs(T, row),
                row,
                _outside_interval_flag(row_name, T, row),
            )
        return row_cache[row_name]

    G_electron, electron_row, electron_flag = gibbs("e-")
    positive: list[tuple[str, str, float, tuple[str | None, ...]]] = []
    negative: list[tuple[str, str, float, tuple[str | None, ...]]] = []
    positive_sum = 0.0
    negative_sum = 0.0

    for neutral, ion in _GAS_IONIZATION_PAIRS:
        if neutral not in values:
            continue
        G_neutral, _neutral_row, neutral_flag = gibbs(neutral)
        G_ion, ion_row, ion_flag = gibbs(ion)
        # Premise: the fitted apparent-Gibbs rows share JANAF's elemental
        # reference, so the baseline cancels for M = M+ + e-. Algebra:
        # K_M = exp[-(G(M+) + G(e-) - G(M))/(R*T)]. Unit check: ΔG and R*T
        # are J/mol. Saha check for Na at 1500 K:
        # Kp = 2*(2*pi*m_e*k_B*T/h**2)**(3/2)*(k_B*T/p°)
        #      *(U_Na+/U_Na)*exp(-chi/(k_B*T)).
        # With U_Na+/U_Na = 1/2 for the ground states and chi = 5.13907696 eV
        # (NIST Atomic Spectra Database), Kp = 1.57265e-16. The fitted value
        # is 1.58211e-16 (JANAF log Kf = -15.801); its 0.60% offset is in
        # JANAF's tabulated thermochemical equilibrium, not our fit residual.
        # The k_B*T/p° factor converts Saha number density to dimensionless
        # pressure. Excited-state partition functions are not the explanation:
        # their correction at 1500 K is below this difference.
        dG = G_ion + G_electron - G_neutral
        equilibrium = math.exp(-dG / (R_J_MOL_K * T))
        positive_sum += equilibrium * values[neutral]
        positive.append(
            (
                neutral,
                ion,
                equilibrium,
                (neutral_flag, ion_flag, electron_flag, domain_flags.get(neutral)),
            )
        )

    for neutral, ion in _GAS_ELECTRON_ATTACHMENTS:
        if neutral not in values:
            continue
        G_neutral, _neutral_row, neutral_flag = gibbs(neutral)
        G_ion, ion_row, ion_flag = gibbs(ion)
        # Premise: A + e- = A- uses the same ideal-gas standard state.
        # Algebra: K_A = exp[-(G(A-) - G(A) - G(e-))/(R*T)].
        dG = G_ion - G_neutral - G_electron
        equilibrium = math.exp(-dG / (R_J_MOL_K * T))
        negative_sum += equilibrium * values[neutral]
        negative.append(
            (
                neutral,
                ion,
                equilibrium,
                (neutral_flag, ion_flag, electron_flag, domain_flags.get(neutral)),
            )
        )

    electron_pressure = (
        math.sqrt(positive_sum / (1.0 + negative_sum))
        if positive_sum > 0.0
        else 0.0
    )
    values["e-"] = electron_pressure
    domain_flags["e-"] = electron_flag
    provenance_class["e-"] = _ION_PROVENANCE_CLASS

    for neutral, ion, equilibrium, flags in positive:
        values[ion] = (
            equilibrium * values[neutral] / electron_pressure
            if electron_pressure > 0.0
            else 0.0
        )
        domain_flags[ion] = "; ".join(flag for flag in flags if flag) or None
        provenance_class[ion] = _ION_PROVENANCE_CLASS

    for neutral, ion, equilibrium, flags in negative:
        values[ion] = equilibrium * values[neutral] * electron_pressure
        domain_flags[ion] = "; ".join(flag for flag in flags if flag) or None
        provenance_class[ion] = _ION_PROVENANCE_CLASS


def species_thermo(
    species: str,
    phase: Literal["g", "l", "cr"],
    T_K: float,
    datapack: ImccGasDatapack | None = None,
) -> SpeciesThermo:
    """Return row-level Cp, S, apparent enthalpy, and runtime apparent G.

    ``species`` is the bare formula (for example ``"FeO"``); ``phase`` is
    ``"g"``, ``"l"`` or ``"cr"``. The active row follows the runtime interval
    selector and temperatures outside its closed interval are refused.

    For gas rows, let ``t = T/1000``. Shomate gives
    ``Cp = A + B*t + C*t² + D*t³ + E/t²`` (J mol⁻¹ K⁻¹), with enthalpy
    primitive ``I_H = A*t + B*t²/2 + C*t³/3 + D*t⁴/4 − E/t`` (kJ mol⁻¹)
    and ``S = A*ln(t) + B*t + C*t²/2 + D*t³/3 − E/(2*t²) + G``
    (J mol⁻¹ K⁻¹). The packaged convention has Shomate H=0 and folds
    ``dfH298`` into F, so ``H_app = I_H + F − H`` includes the formation
    enthalpy anchor. ``G_J_mol`` is the existing runtime
    ``1000*H_app − T*S``; the factor converts H_app from kJ/mol to J/mol.
    A caller needing only H(T)−H(298.15) must subtract the source row's
    ``dfH298`` from ``H_app``; that anchor is in the JANAF source record and
    provenance, not separately encoded in the runtime row.

    For condensate rows, ``phi(t) = dG_A + dG_B*t + dG_C*t² + dG_D*t³
    + dG_E*t⁴ = −(G − H298)/(R*T)``. Gibbs–Helmholtz at fixed H298 gives
    ``H−H298 = R*T²*dphi/dT`` and ``S = R*phi + R*T*dphi/dT``; differentiating
    enthalpy gives Cp. The returned apparent enthalpy is
    ``H_app = R*1000*dH298_R + R*T²*dphi/dT`` in J/mol, converted to kJ/mol
    for this field; equivalently its anchor term is ``R*dH298_R`` kJ/mol.
    Here H298 is the source's 298.15 K enthalpy anchored to its stable phase,
    also for a liquid row. In ``t`` form the H−H298, S, and Cp polynomial terms
    are ``R*t²*(B+2*C*t+3*D*t²+4*E*t³)`` kJ/mol,
    ``R*(A+2*B*t+3*C*t²+4*D*t³+5*E*t⁴)`` J/(mol K), and
    ``R*(2*B*t+6*C*t²+12*D*t³+20*E*t⁴)`` J/(mol K). The factors of 1000
    convert the kK anchor and Shomate enthalpy units. A zero polynomial has
    zero Cp, S and H−H298 while G equals H298, a sign/unit sanity check.
    The returned ``derivatives_fit_implied`` flag is true for every condensate
    row: their Cp, S and H_app are implied by differentiating the fitted G
    polynomial. JANAF-fitted condensates fit Phi only. Gas Shomate fits target
    Cp, H and S directly, so their flag is false. For condensate rows the
    source-comparison gate therefore checks G_app only; derivative residuals
    are fit-implied information, not source-quantity gates.
    The regression gate checks FeO(l) against JANAF Fe-019 and Na2O(l) against
    Na-013. Al2O3(l) is a LAM-source exception compared with Al-100.
    """
    if phase not in {"g", "l", "cr"}:
        raise ValueError("phase must be 'g', 'l', or 'cr'")
    if not math.isfinite(T_K) or T_K <= 0.0:
        raise ValueError(f"temperature must be finite and positive, got {T_K}")

    active_pack = load_gas_datapack() if datapack is None else datapack
    row_name = f"{species}({phase})"
    if phase == "g":
        row = _nearest_interval_row(active_pack.gas_df, row_name, T_K)
        t = T_K / 1000.0
        A, B, C, D, E = (float(row[key]) for key in "ABCDE")
        enthalpy_primitive = (
            A * t + B * t**2 / 2.0 + C * t**3 / 3.0
            + D * t**4 / 4.0 - E / t
        )
        cp = A + B * t + C * t**2 + D * t**3 + E / t**2
        entropy = (
            A * math.log(t) + B * t + C * t**2 / 2.0
            + D * t**3 / 3.0 - E / (2.0 * t**2) + float(row["G"])
        )
        h_app = enthalpy_primitive + float(row["F"]) - float(row["H"])
        apparent_g = _janaf_gibbs(T_K, row)
        interval = int(row["T_interval"])
    else:
        row = _oxide_row_for_T(active_pack.oxide_df, row_name, T_K)
        t = T_K / 1000.0
        A, B, C, D, E = (
            float(row[key]) for key in ("dG_A", "dG_B", "dG_C", "dG_D", "dG_E")
        )
        h_increment = R_J_MOL_K * t**2 * (
            B + 2.0 * C * t + 3.0 * D * t**2 + 4.0 * E * t**3
        )
        h_app = R_J_MOL_K * float(row["dH298_R"]) + h_increment
        entropy = R_J_MOL_K * (
            A + 2.0 * B * t + 3.0 * C * t**2 + 4.0 * D * t**3 + 5.0 * E * t**4
        )
        cp = R_J_MOL_K * (
            2.0 * B * t + 6.0 * C * t**2 + 12.0 * D * t**3 + 20.0 * E * t**4
        )
        apparent_g = _lamor_gibbs(T_K, row)
        interval = None

    return SpeciesThermo(
        Cp_J_molK=float(cp),
        S_J_molK=float(entropy),
        H_app_kJ_mol=float(h_app),
        G_J_mol=float(apparent_g),
        source_row_id=str(row["Ref"]),
        source_table_id=(
            str(row["Ref"])
            if phase == "g" and str(row["Ref"])
            else (
                str(row["Ref"]).removesuffix(_SUPERCOOLED_EXTENSION_REF_SUFFIX)
                if phase != "g"
                and str(row["Ref"]).endswith(_SUPERCOOLED_EXTENSION_REF_SUFFIX)
                else _OXIDE_SOURCE_TABLE_IDS.get(species)
            )
        ),
        T_interval=interval,
        T_min=float(row["T_min"]),
        T_max=float(row["T_max"]),
        derivatives_fit_implied=(
            phase != "g"
        ),
    )


# --------------------------------------------------------------------------- #
# Mass-action: Kp derivation and partial-pressure evaluation
# --------------------------------------------------------------------------- #


def evaluate_gas(
    activities: Mapping[str, float] | Sequence[float] | np.ndarray,
    T_K: float,
    fO2: float,
    datapack: ImccGasDatapack,
    parent_oxides: Sequence[str] | None = None,
    allow_extrapolation: bool = True,
    gas_species: Sequence[str] | str | None = None,
    include_ions: bool = False,
) -> ImccGasResult:
    """Compute equilibrium partial pressures for the SF04 retained gas set.

    Parameters
    ----------
    activities:
        Parent activities. Either a dict keyed by parent name, or a vector
        aligned with ``parent_oxides`` (default IMCC order). Condensed parents
        use their pure-liquid standard state; sulfur channels use caller-supplied
        ``S2`` fugacity relative to JANAF's 1-bar gas standard.
    T_K:
        Temperature in Kelvin. Must be finite and positive on this path.
    fO2:
        Oxygen fugacity, stored as the numerical value of p_O2 / p° with
        p° = 1 bar (pinned by the caller; no internal fO2 model is applied).
        Must be finite and positive on this path.
    datapack:
        Loaded JANAF + condensate thermodynamic tables.
    parent_oxides:
        Ordered parent names. Defaults to the IMCC-SF04 8-oxide basis; callers
        may add ``P2O5`` or ``S2`` when supplying those parent activities.
    allow_extrapolation:
        If True (default), evaluate finite temperatures outside the selected
        row and return a per-species ``domain_flags`` entry.  If False, refuse
        with ``ImccGasTemperatureOutsideDomainError`` instead.  Non-finite or
        non-positive T raises ``ValueError`` before this flag is consulted.
    gas_species:
        Optional retained-species subset.  The default evaluates every
        channel whose parent activity is supplied, subject to the shared
        datapack-row availability rule in ``_default_reactions``.  A named
        channel that cannot be served raises ``ImccGasSpeciesNotFoundError``;
        a subset otherwise permits channel-specific diagnostics with the same
        typed domain refusals.
    include_ions:
        If True, solve charge balance using the default neutral gas set, then
        append available fitted cations and every JANAF-supported anion whose
        neutral channel is present. The default keeps the established
        neutral-only mapping. Ion names in ``gas_species`` require this option.

    Returns
    -------
    ImccGasResult
        A read-only mapping of partial pressures in bar.  ``unit`` is
        ``"bar"``; ``domain_flags`` names extrapolated source rows; and
        ``provenance_class`` carries the least-authoritative class returned by
        ``gas_species_provenance`` for each channel.

    Derivation
    ----------
    Premise: for each retained gas species we write a single vaporization
    reaction with the parent oxide in the melt as the reactant and O2 as an
    explicit product when stoichiometry requires it:

        oxide(l)  =  n_gas * gas(g)  +  n_O2 * O2(g)                (1)

    For condensed parents, the standard Gibbs free energy change for (1) is
    assembled from JANAF gas product rows and the condensate parent G°(T) row:

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

    Sulfur uses the gas parent directly, with tuple coefficients n_S2,n_O2:

        n_S2 * S2(g) + n_O2 * O2(g) = species(g)

    Here ΔG° = G°(species) - n_S2*G°(S2) - n_O2*G°(O2), and
    p̃_gas = K° * a(S2)^n_S2 * fO2^n_O2. The caller supplies
    a(S2)=f(S2)/p° on the JANAF 1-bar gas reference.

    The returned mapping reports those p̃_gas values as bar.  For the special
    retained species O2, p̃_O2 = fO2 by definition.  For n_O2 = 0, (5)
    reduces to p̃_gas = K° * a_oxide.

    Unit check: ΔG° / (R T) is dimensionless, so K° is dimensionless;
    p̃_i is dimensionless; the bar label on the return value is the p° = 1
    bar identification, not a leftover unit on K°.
    Sanity on this path: ΔG° → +∞ gives K° → 0 and p̃_gas → 0; a_oxide → 0
    gives p̃_gas → 0. For condensed-parent channels, n_O2 > 0 makes decreasing
    a finite positive fO2 raise p̃_gas as fO2^(-n_O2/n_gas). Sulfur channels
    consume O2, so p̃_gas scales as fO2^n_O2 and decreases as fO2 is lowered
    when n_O2 > 0. fO2 = 0 is refused (non-positive), so that limit is not returned.
    """
    if parent_oxides is None:
        parent_oxides = IMCC_PARENT_OXIDES

    omitted_channels: Mapping[str, str] = {}
    requested_channels: tuple[str, ...] | None = None
    requested_ions: set[str] = set()
    if gas_species is None:
        # Resolved after the inputs are validated: the default set depends on
        # parent_oxides and on which rows the datapack carries.
        reactions = None
    else:
        requested_channels = (
            (gas_species,) if isinstance(gas_species, str) else tuple(gas_species)
        )
        ion_names = {
            ion
            for _neutral, ion in (
                *_GAS_IONIZATION_PAIRS,
                *_GAS_ELECTRON_ATTACHMENTS,
            )
        } | {"e-"}
        requested_ions = set(requested_channels) & ion_names
        if requested_ions and not include_ions:
            raise ImccGasSpeciesNotFoundError(
                "ion channels require include_ions=True; pass include_ions=True "
                f"to request {sorted(requested_ions)!r}"
            )
        neutral_requests = tuple(
            name for name in requested_channels if name not in requested_ions
        )
        missing = [name for name in neutral_requests if name not in _SF04_REACTIONS]
        if missing:
            raise ImccGasSpeciesNotFoundError(
                f"no IMCC-SF04 gas channel for species {missing!r}"
            )
        reactions = tuple(
            (name, _SF04_REACTIONS[name]) for name in neutral_requests
        )

    T = float(T_K)
    try:
        p_O2 = float(fO2)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ImccGasInvalidFugacityError(
            "fO2 must be finite and positive p_O2/p° (bar-relative, not "
            f"log10 fO2); got {fO2}"
        ) from exc
    if not math.isfinite(T) or T <= 0.0:
        raise ValueError(f"temperature must be finite and positive, got {T_K}")
    if not math.isfinite(p_O2) or p_O2 <= 0.0:
        raise ImccGasInvalidFugacityError(
            "fO2 must be finite and positive p_O2/p° (bar-relative, not "
            f"log10 fO2); got {fO2}"
        )

    if isinstance(activities, Mapping):
        act = {name: float(activities.get(name, 0.0)) for name in parent_oxides}
        act.update(
            {
                name: float(activities[name])
                for name in activities
                if (
                    name in _REACTION_PARENT_OXIDES
                    and name not in IMCC_PARENT_OXIDES
                    and name not in act
                )
            }
        )
    else:
        arr = np.asarray(activities, dtype=float)
        if arr.shape[0] != len(parent_oxides):
            raise ValueError(
                f"activities vector length {arr.shape[0]} does not match "
                f"{len(parent_oxides)} parent oxides"
            )
        act = {name: float(arr[i]) for i, name in enumerate(parent_oxides)}

    if reactions is None or (include_ions and requested_channels is not None):
        reactions, omitted_channels = _default_reactions(tuple(act), datapack)

    for gas_name, (oxide, _n_gas, _n_O2) in reactions:
        if not oxide:
            continue
        if oxide not in act:
            raise ImccGasSpeciesNotFoundError(
                f"gas channel {gas_name!r} needs parent activity {oxide!r}, "
                "which is not in parent_oxides"
            )
        a_used = act[oxide]
        if not math.isfinite(a_used) or a_used < 0.0:
            raise ValueError(
                f"activity of {oxide!r} must be finite and >= 0, got {a_used}"
            )

    # G°(O2, T) is needed only for reactions that produce or consume O2.
    needs_O2_row = any(n_O2 != 0.0 for _, (_, _, n_O2) in reactions)
    O2_row = None
    G_O2 = 0.0
    if needs_O2_row:
        O2_row = _nearest_interval_row(
            datapack.gas_df, "O2(g)", T, allow_extrapolation=allow_extrapolation
        )
        G_O2 = _janaf_gibbs(T, O2_row)

    result: dict[str, float] = {}
    domain_flags: dict[str, str | None] = {}
    provenance_class: dict[str, str] = {}
    for gas_name, (oxide, n_gas, n_O2) in reactions:
        flags: list[str] = []
        if gas_name == "O2":
            # O2 is the caller-pinned fO2 value, so this channel evaluates no
            # Gibbs row and must never inherit the O2(g) row's domain flag.
            result[gas_name] = p_O2
            domain_flags[gas_name] = None
            provenance_class[gas_name] = str(
                gas_species_provenance(gas_name)["authority"]
            )
            continue

        gas_species = f"{gas_name}(g)"
        gas_row = _nearest_interval_row(
            datapack.gas_df, gas_species, T, allow_extrapolation=allow_extrapolation
        )
        G_gas = _janaf_gibbs(T, gas_row)
        flag = _outside_interval_flag(gas_species, T, gas_row)
        if flag is not None:
            flags.append(flag)

        if O2_row is not None and n_O2 != 0.0:
            flag = _outside_interval_flag("O2(g)", T, O2_row)
            if flag is not None:
                flags.append(flag)

        if oxide:
            oxide_row = None
            gas_parent_row = _GAS_PHASE_PARENT_ROWS.get(oxide)
            if gas_parent_row:
                if gas_parent_row == gas_species:
                    parent_row = gas_row
                else:
                    parent_row = _nearest_interval_row(
                        datapack.gas_df,
                        gas_parent_row,
                        T,
                        allow_extrapolation=allow_extrapolation,
                    )
                G_oxide = _janaf_gibbs(T, parent_row)
                flag = (
                    None
                    if gas_parent_row == gas_species
                    else _outside_interval_flag(gas_parent_row, T, parent_row)
                )
            else:
                oxide_name = f"{oxide}(l)"
                oxide_row = _oxide_row_for_T(
                    datapack.oxide_df,
                    oxide_name,
                    T,
                    allow_extrapolation=allow_extrapolation,
                )
                G_oxide = _lamor_gibbs(T, oxide_row)
                flag = _outside_interval_flag(oxide_name, T, oxide_row)
            a_oxide = act[oxide]
            if flag is not None:
                flags.append(flag)
            if (
                oxide_row is not None
                and str(oxide_row["Ref"]).endswith(
                    _SUPERCOOLED_EXTENSION_REF_SUFFIX
                )
            ):
                source_table = str(oxide_row["Ref"]).removesuffix(
                    _SUPERCOOLED_EXTENSION_REF_SUFFIX
                )
                flags.append(
                    f"{oxide_name} uses a labelled constant-Cp supercooled-liquid "
                    f"continuation from JANAF {source_table}"
                )
        else:
            G_oxide = 0.0
            a_oxide = 1.0

        if oxide in _GAS_PHASE_PARENT_ROWS:
            # Sulfur tuple: n_S2 S2(g) + n_O2 O2(g) = gas(g).
            # Kp = p_gas / (a_S2**n_S2 * fO2**n_O2), with ΔG° formed from
            # the same balanced reaction; a_S2 is caller-supplied f(S2)/p°.
            dG = G_gas - n_gas * G_oxide - n_O2 * G_O2
            Kp = np.exp(-dG / (R_J_MOL_K * T))
            p_gas = Kp * (a_oxide**n_gas) * (p_O2**n_O2)
        else:
            # Condensed-parent reaction (1): oxide(l) = n_gas*gas(g) +
            # n_O2*O2(g). Keep its established arithmetic path unchanged.
            dG = n_gas * G_gas + n_O2 * G_O2 - G_oxide
            Kp = np.exp(-dG / (R_J_MOL_K * T))
            p_gas = (Kp * a_oxide / (p_O2**n_O2)) ** (1.0 / n_gas)
        result[gas_name] = float(p_gas)
        domain_flags[gas_name] = "; ".join(flags) or None
        provenance_class[gas_name] = str(
            gas_species_provenance(gas_name)["authority"]
        )

    if include_ions:
        _add_ion_channels(
            result,
            domain_flags,
            provenance_class,
            datapack,
            T,
            allow_extrapolation,
        )
        closure_neutral_pressure = sum(
            value
            for name, value in result.items()
            if not name.endswith(("+", "-"))
        )

        # Charge balance belongs to the full default gas. Retain requested
        # neutrals and their charge channels only after that closure is solved.
        neutral_names = {
            name for name in result if not name.endswith(("+", "-"))
        }
        neutral_ions = {
            ion
            for neutral, ion in (
                *_GAS_IONIZATION_PAIRS,
                *_GAS_ELECTRON_ATTACHMENTS,
            )
            if neutral in neutral_names
        }
        if requested_channels is not None:
            requested_neutrals = set(requested_channels) - requested_ions
            unavailable_neutrals = requested_neutrals - result.keys()
            if unavailable_neutrals:
                raise ImccGasSpeciesNotFoundError(
                    f"requested gas channels are unavailable for this gas: "
                    f"{sorted(unavailable_neutrals)!r}"
                )
            requested_ions |= neutral_ions & {
                ion
                for neutral, ion in (
                    *_GAS_IONIZATION_PAIRS,
                    *_GAS_ELECTRON_ATTACHMENTS,
                )
                if neutral in requested_neutrals
            }
            unavailable_ions = requested_ions - result.keys()
            if unavailable_ions:
                raise ImccGasSpeciesNotFoundError(
                    f"requested ion channels are unavailable for this gas: "
                    f"{sorted(unavailable_ions)!r}"
                )
            keep = requested_neutrals | requested_ions | {"e-"}
            result = {name: value for name, value in result.items() if name in keep}
            domain_flags = {
                name: value for name, value in domain_flags.items() if name in keep
            }
            provenance_class = {
                name: value for name, value in provenance_class.items() if name in keep
            }

        if closure_neutral_pressure > 1.0:
            notice = (
                f"summed neutral pressure {closure_neutral_pressure:.6g} bar "
                "exceeds 1 bar; "
                "ideal-gas ion closure is not credible at this pressure"
            )
            domain_flags = {
                name: "; ".join(filter(None, (flag, notice)))
                for name, flag in domain_flags.items()
                if name.endswith(("+", "-"))
            }

    return ImccGasResult(
        result,
        unit="bar",
        domain_flags=domain_flags,
        provenance_class=provenance_class,
        omitted_channels=omitted_channels,
    )


_ATOMIC_MASS_G_MOL = {
    "O": 15.999, "Na": 22.989769, "K": 39.0983, "Si": 28.085,
    "Fe": 55.845, "Mg": 24.305, "Al": 26.9815385, "Ca": 40.078,
    "Ti": 47.867, "Cr": 51.9961, "V": 50.9415, "Nb": 92.90637,
    "Mn": 54.938044, "Ni": 58.6934, "Co": 58.933194,
    "P": 30.973761998, "S": 32.06,
}
_FORMULA_PART = re.compile(r"([A-Z][a-z]?)(\d*)")
# log10(pO2/bar) search bracket. The upper edge (1 bar) is a physical ceiling,
# not a numerical convenience: the effusion law J ∝ p/√(M T) holds only in
# molecular flow, where the mean free path λ ≫ orifice diameter d.
# λ = k_B T / (√2 π σ² p); with σ ≈ 3.6e-10 m at 2000 K, λ ≈ 0.5 µm × (1 bar / p).
# Knudsen orifices are ~0.1–1 mm, so molecular flow (λ/d ≳ 1) already requires a
# total pressure ≲ 1e-3 bar. A root above 1 bar therefore lies far outside the
# model's validity and is refused rather than returned. The lower edge (1e-30 bar)
# is below any physically resolvable oxygen pressure.
_OXYGEN_BALANCE_BRACKET = (-30.0, 0.0)


class OxygenBalanceSpeciesMetadata(NamedTuple):
    """Formula and pO2 scaling data for one oxygen-balance gas species."""

    molar_mass: float
    oxygen_atoms: float
    parent_oxygen_demand: float
    pO2_exponent: float


def _formula_atoms(formula: str) -> dict[str, int]:
    """Count atoms in a retained gas or parent-oxide formula."""
    parts = _FORMULA_PART.findall(formula)
    if not parts or "".join(element + count for element, count in parts) != formula:
        raise ImccGasOxygenBalanceError(
            f"cannot parse retained formula {formula!r} for oxygen balance"
        )
    atoms: dict[str, int] = {}
    for element, count in parts:
        atoms[element] = atoms.get(element, 0) + int(count or 1)
    return atoms


def oxygen_balance_species_metadata(
    species_parent_oxides: Mapping[str, str | None],
    *,
    pO2_exponents: Mapping[str, float] | None = None,
) -> dict[str, OxygenBalanceSpeciesMetadata]:
    """Return formula, parent-demand, and pO2 exponent metadata by species.

    Parent oxygen demand is the parent oxide's O/metal ratio multiplied by
    the gas species' metal-atom count. For M_x O_y from parent M_a O_b,
    the pO2 exponent is (y - x*b/a) / 2. Parentless oxygen species use
    y / 2. Supply pO2_exponents to override or define custom tables.
    """
    overrides = pO2_exponents or {}
    unknown_overrides = overrides.keys() - species_parent_oxides.keys()
    if unknown_overrides:
        raise ImccGasOxygenBalanceError(
            "pO2 exponent override names are absent from species metadata: "
            f"{sorted(unknown_overrides)}"
        )
    metadata: dict[str, OxygenBalanceSpeciesMetadata] = {}
    for species, parent in species_parent_oxides.items():
        atoms = _formula_atoms(species)
        mass = sum(_ATOMIC_MASS_G_MOL[element] * number for element, number in atoms.items())
        oxygen_atoms = atoms.get("O", 0)
        metal_atoms = sum(number for element, number in atoms.items() if element != "O")
        if parent:
            parent_atoms = _formula_atoms(parent)
            parent_metals = sum(number for element, number in parent_atoms.items() if element != "O")
            if parent_metals <= 0:
                raise ImccGasOxygenBalanceError(
                    f"parent oxide {parent!r} has no metal atoms for oxygen balance"
                )
            parent_ratio = parent_atoms.get("O", 0) / parent_metals
            parent_oxygen = parent_ratio * metal_atoms
            exponent = (oxygen_atoms - parent_ratio * metal_atoms) / 2.0
        else:
            parent_oxygen = 0.0
            exponent = oxygen_atoms / 2.0
        if species in overrides:
            try:
                exponent = float(overrides[species])
            except (TypeError, ValueError, OverflowError) as exc:
                raise ImccGasOxygenBalanceError(
                    f"invalid pO2 exponent for {species!r}"
                ) from exc
        if not math.isfinite(exponent):
            raise ImccGasOxygenBalanceError(
                f"invalid pO2 exponent for {species!r}: {exponent}"
            )
        metadata[species] = OxygenBalanceSpeciesMetadata(
            mass, oxygen_atoms, parent_oxygen, exponent
        )
    return metadata


def oxygen_balance_from_pressure_model(
    pressure_model: Callable[[float], Mapping[str, float]],
    species: Mapping[str, OxygenBalanceSpeciesMetadata],
    *,
    bracket: tuple[float, float] = _OXYGEN_BALANCE_BRACKET,
) -> tuple[float, Mapping[str, float], dict[str, object]]:
    """Solve the generic Knudsen oxygen-balance root for a pressure model.

    Each pressure must follow its declared power law in pO2. If every signed
    coefficient has the same sign as its exponent, F is monotone; a strictly
    increasing term makes its sign-changing root unique.
    """
    for name, data in species.items():
        weight = (
            data.oxygen_atoms - data.parent_oxygen_demand
        ) / math.sqrt(data.molar_mass)
        if weight * data.pO2_exponent < 0.0:
            raise ImccGasOxygenBalanceError(
                f"oxygen-balance flux is non-monotone for retained species {name!r} "
                f"(weight={weight}, pO2 exponent={data.pO2_exponent})"
            )
    if not any(
        (data.oxygen_atoms - data.parent_oxygen_demand) * data.pO2_exponent > 0.0
        for data in species.values()
    ):
        raise ImccGasOxygenBalanceError(
            "oxygen-balance flux is not strictly monotone for declared species"
        )

    def at(logp: float) -> tuple[float, Mapping[str, float], float, float]:
        pressures = pressure_model(logp)
        if not isinstance(pressures, MappingABC):
            raise ImccGasOxygenBalanceError(
                f"pressure model must return a mapping at log10(pO2/bar)={logp}"
            )
        validated: dict[str, float] = {}
        for name in species:
            try:
                pressure = float(pressures[name])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ImccGasOxygenBalanceError(
                    f"pressure model is missing or has invalid pressure for "
                    f"species {name!r} at log10(pO2/bar)={logp}"
                ) from exc
            if not math.isfinite(pressure) or pressure < 0.0:
                raise ImccGasOxygenBalanceError(
                    f"pressure model returned non-finite or negative pressure for "
                    f"species {name!r} at log10(pO2/bar)={logp}: {pressure!r}"
                )
            validated[name] = pressure
        oxygen_flux = math.fsum(
            data.oxygen_atoms * validated[name] / math.sqrt(data.molar_mass)
            for name, data in species.items()
        )
        parent_flux = math.fsum(
            data.parent_oxygen_demand * validated[name] / math.sqrt(data.molar_mass)
            for name, data in species.items()
        )
        if not (math.isfinite(oxygen_flux) and math.isfinite(parent_flux)):
            raise ImccGasOxygenBalanceError(
                f"oxygen-balance flux is non-finite at log10(pO2/bar)={logp}"
            )
        return oxygen_flux - parent_flux, pressures, oxygen_flux, parent_flux

    low, high = bracket
    if not (math.isfinite(low) and math.isfinite(high) and low < high):
        raise ImccGasOxygenBalanceError(f"invalid oxygen-balance bracket {bracket}")
    if high > 0.0:
        raise ImccGasOxygenBalanceError(
            "oxygen-balance bracket exceeds pO2 = 1 bar, outside the Knudsen "
            "molecular-flow regime where the effusion law holds"
        )
    f_low, f_high = at(low)[0], at(high)[0]
    if high == 0.0 and f_high < 0.0:
        raise ImccGasOxygenBalanceError(
            "oxygen-balance root lies above pO2 = 1 bar, outside the Knudsen "
            "molecular-flow regime where the effusion law holds; no value returned "
            f"(F(1 bar)={f_high})"
        )
    if not (f_low < 0.0 < f_high):
        raise ImccGasOxygenBalanceError(
            f"oxygen-balance root is not bracketed on {bracket}: "
            f"F(low)={f_low}, F(high)={f_high}"
        )
    try:
        root, solver = brentq(
            lambda value: at(value)[0], low, high, xtol=1e-12,
            full_output=True, disp=False,
        )
    except (ValueError, RuntimeError) as exc:
        raise ImccGasOxygenBalanceError(f"oxygen-balance solve failed: {exc}") from exc
    if not solver.converged:
        raise ImccGasOxygenBalanceError("oxygen-balance solve did not converge")

    residual, pressures, oxygen_flux, parent_flux = at(root)
    slope_low = at(root - 0.1)[1]
    slope_high = at(root + 0.1)[1]
    for name, data in species.items():
        if float(pressures[name]) > 1e-300:
            try:
                if float(slope_low[name]) <= 0 or float(slope_high[name]) <= 0:
                    raise ValueError("non-positive pressure around root")
                observed_exponent = (
                    math.log10(float(slope_high[name]))
                    - math.log10(float(slope_low[name]))
                ) / 0.2
            except (KeyError, ValueError, TypeError) as exc:
                raise ImccGasOxygenBalanceError(
                    f"pressure model is not a power law in pO2 with the declared "
                    f"exponent for species {name!r}"
                ) from exc
            # Power-law log ratios incur about 1e-12 dex of float round-off;
            # 1e-6 leaves a generous margin while detecting real curvature.
            if abs(observed_exponent - data.pO2_exponent) > 1e-6:
                raise ImccGasOxygenBalanceError(
                    f"pressure model is not a power law in pO2 with the declared "
                    f"exponent for species {name!r} (declared="
                    f"{data.pO2_exponent}, observed={observed_exponent})"
                )
    oxygen_carriers = sorted(
        (
            (
                name,
                data.oxygen_atoms * float(pressures[name]) / math.sqrt(data.molar_mass),
            )
            for name, data in species.items()
            if data.oxygen_atoms > 0
        ),
        key=lambda item: item[1],
        reverse=True,
    )[:5]
    metal_carriers = sorted(
        (
            (
                name,
                data.parent_oxygen_demand
                * float(pressures[name])
                / math.sqrt(data.molar_mass),
            )
            for name, data in species.items()
            if data.parent_oxygen_demand > 0
        ),
        key=lambda item: item[1],
        reverse=True,
    )[:5]
    diagnostics: dict[str, object] = {
        "mode": "oxygen_balance_effusion",
        "residual": abs(residual) / max(oxygen_flux, parent_flux, 1e-300),
        "bracket": bracket,
        "iterations": solver.iterations,
        "dominant_O_carriers": oxygen_carriers,
        "dominant_metal_carriers": metal_carriers,
    }
    return 10.0**root, pressures, diagnostics


def evaluate_gas_oxygen_balance(
    activities: Mapping[str, float] | Sequence[float] | np.ndarray,
    T_K: float,
    datapack: ImccGasDatapack,
    parent_oxides: Sequence[str] | None = None,
    allow_extrapolation: bool = True,
) -> tuple[float, ImccGasResult, dict[str, object]]:
    """Solve pO2 that balances oxygen in the Knudsen effusion flux.

    Species flux is ``J_i = p_i A W / sqrt(2 pi M_i R T)``. The common
    aperture, Clausing, and temperature factors cancel, leaving
    ``F = sum_i (nO_i - (nuO/nuM)_parent nM_i) p_i / sqrt(M_i) = 0``.
    Existing equilibria give ``p_i ∝ pO2**k_i``. For condensed-parent
    reactions, the signed oxygen coefficient is ``-2 nO2/n_gas`` and
    ``k_i = -nO2/n_gas``. For the S2(g)-parent reactions, they are ``2 nO2``
    and ``k_i = nO2``. In both cases their product is nonnegative, so
    ``dF/dlog10(pO2) = ln(10) sum_i C_i k_i pO2**k_i >= 0``. Parentless O and
    O2 terms have positive coefficients and exponents 1/2 and 1. F is strictly
    increasing when an oxygen-bearing channel exists, so a sign-changing
    bracket contains one root. Brent's ``xtol=1e-12`` dex is well below
    source-table precision. Flux terms use pressures in bar and molar masses
    in g/mol; the common scale cancels. The upper bracket is the 1 bar
    molecular-flow ceiling.

    Returns ``(pO2_bar, partial_pressures, diagnostics)``. Partial pressures
    match the neutral-only :func:`evaluate_gas` result; diagnostics contain relative residual, the
    log10(bar) bracket, iteration count, and the five strongest O and parent
    oxygen carriers.
    """
    if parent_oxides is None:
        parent_oxides = IMCC_PARENT_OXIDES
    T = float(T_K)
    if not math.isfinite(T) or T <= 0.0:
        raise ValueError(f"temperature must be finite and positive, got {T_K}")

    channels, _omitted_channels = _default_reactions(parent_oxides, datapack)
    species_data = oxygen_balance_species_metadata({
        species: parent for species, (parent, _n_gas, _n_O2) in channels
    })
    for species, (parent, n_gas, n_O2) in channels:
        data = species_data[species]
        if parent in _GAS_PHASE_PARENT_ROWS:
            exponent = n_O2
        else:
            exponent = (
                -n_O2 / n_gas
                if parent
                else (1.0 if species == "O2" else 0.5)
            )
        if not math.isclose(data.pO2_exponent, exponent, abs_tol=1e-12):
            raise ImccGasOxygenBalanceError(
                f"gas reaction metadata for {species!r} violates oxygen balance"
            )

    def pressure_model(logp: float) -> ImccGasResult:
        return evaluate_gas(
            activities, T, 10.0**logp, datapack,
            parent_oxides=parent_oxides,
            allow_extrapolation=allow_extrapolation,
        )

    return oxygen_balance_from_pressure_model(pressure_model, species_data)
