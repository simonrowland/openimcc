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

import csv
import hashlib
import json
import math
import os
import re
import warnings
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from functools import lru_cache
from importlib import resources
from numbers import Number
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Callable, Literal, Mapping, NamedTuple, Sequence

import numpy as np
from scipy.optimize import brentq

from openimcc.kernel import ImccRefusal, _canonical_published_serialization

if TYPE_CHECKING:
    import pandas as pd


# --------------------------------------------------------------------------- #
# Physical constants and reference states
# --------------------------------------------------------------------------- #

R_J_MOL_K = 8.314462618
"""Molar gas constant, J / (mol K)."""

_SUPERCOOLED_EXTENSION_REF_SUFFIX = "-SC-CP"
_SOURCE_CP_EXTENSION_RE = re.compile(r"-CP-EXT-(?P<limit>[0-9]+(?:\.[0-9]+)?)$")
_LIQUID_FROM_SOLID_REF_RE = re.compile(
    r"^LFS\[(?P<source>[^;\]]+);Tf=(?P<temperature>[0-9]+(?:\.[0-9]+)?)\]$"
)
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


class ImccGasDuplicateIntervalError(ImccRefusal):
    """Raised when one table has competing rows with the same interval start."""

    code = "imcc_gas_duplicate_interval"


class ImccGasInvalidIntervalError(ImccRefusal):
    """Raised when a table interval has invalid or non-finite bounds."""

    code = "imcc_gas_invalid_interval"


class ImccGasTemperatureOutsideDomainError(ImccRefusal):
    """Raised when T falls outside the declared G(T) interval for a species."""

    code = "imcc_gas_T_outside_domain"


class ImccGasInvalidFugacityError(ImccRefusal):
    """Raised when the caller's bar-relative oxygen fugacity is invalid."""

    code = "imcc_gas_invalid_fO2"


class ImccGasOxygenBalanceError(ImccRefusal):
    """Raised when the oxygen-balance root is not monotone or bracketed."""

    code = "imcc_gas_oxygen_balance_failed"


class ImccGasIngestionWarning(UserWarning):
    """A CSV load discarded one or more columns with empty headers."""


# --------------------------------------------------------------------------- #
# Packaged data and explicit VapoRock override
# --------------------------------------------------------------------------- #


_VAPOROCK_ENV_VAR = "OPENIMCC_VAPOROCK_ROOT"
_GAS_TABLE_RELPATH = Path("src") / "vaporock" / "data" / "JANAF-vapor-data-full.csv"
_CONDENSATE_TABLE_RELPATH = Path("data") / "condensate-thermo-data.csv"
_PACKAGED_DATA_PACKAGE = "openimcc.data.gas"
_PACKAGED_GAS_NAME = "gas-shomate.csv"
_PACKAGED_CONDENSATE_NAME = "condensate.csv"
_PACKAGED_SOURCE_LIQUID_MARKERS_NAME = "source-liquid-markers.json"
_GAS_TABLE_SCHEMA = frozenset(
    {
        "species_name", "state", "T_interval", "cation", "cat_num", "oxy_num",
        "T_min", "T_max", "A", "B", "C", "D", "E", "F", "G", "H", "Ref",
    }
)
_CONDENSATE_TABLE_SCHEMA = frozenset(
    {
        "species_name", "state", "cation", "cat_num", "oxy_num", "T_min", "T_max",
        "dH298_R", "dG_A", "dG_B", "dG_C", "dG_D", "dG_E", "Ref",
    }
)
_GAS_NUMERIC_COLUMNS = (
    "T_interval", "cat_num", "oxy_num", "T_min", "T_max", *tuple("ABCDEFGH")
)
_CONDENSATE_NUMERIC_COLUMNS = (
    "cat_num", "oxy_num", "T_min", "T_max", "dH298_R", "dG_A", "dG_B", "dG_C",
    "dG_D", "dG_E",
)
_GAS_TEXT_COLUMNS = ("state", "cation", "Ref")
_CONDENSATE_TEXT_COLUMNS = ("state", "cation", "Ref")


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


def _drop_unnamed_columns(
    table: pd.DataFrame, header_source: object, display_path: Path
) -> tuple[pd.DataFrame, str | None]:
    pd = _pandas()
    with header_source.open("r", encoding="utf-8-sig", newline="") as handle:
        raw_header = next(csv.reader(handle))
    unnamed_columns = [
        table.columns[position]
        for position, header in enumerate(raw_header)
        if header == ""
    ]
    if not unnamed_columns:
        return table, None

    non_empty_values = []
    for row_index, (species, row) in enumerate(table.iterrows(), start=1):
        species_name = row.get("species_name", species)
        if isinstance(species_name, str):
            species_name = species_name.removesuffix("(l)")
        reference = row.get("Ref", "")
        for column in unnamed_columns:
            value = row[column]
            if not pd.isna(value):
                non_empty_values.append(
                    f"{column} at row {row_index} "
                    f"({species_name}, {reference}) = {value}"
                )
    message = f"dropped unnamed columns {unnamed_columns!r} from {display_path}"
    if non_empty_values:
        message += "; non-empty: " + "; ".join(non_empty_values)
    return table.drop(columns=unnamed_columns), message


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
# monoxide channels use n_gas=1 and n_O2=1/2 and 0. MnO(l) and its gas products
# use the public tables, including the directly fitted TSIV liquid parents.
# For caller-supplied P2O5(l) (a=2, b=5), n_gas=2/n_P and
# n_O2=(5-n_gas*n_O)/2: P, P2, P4, PO, PO2, P4O6 and P4O10 use
# (2, 5/2), (1, 5/2), (1/2, 5/2), (2, 3/2), (2, 1/2),
# (1/2, 1) and (1/2, 0), respectively. Each tuple balances one P2O5(l).
# PS uses P2O5(l) + S2(g) = 2 PS(g) + 5/2 O2(g), so its parent tuple is
# (2, 5/2) and it consumes one mole of the separate S2 gas parent.
# Sulfur tuples use the caller's gas parent directly: n_S2 S2(g) + n_O2 O2(g)
# = species(g), with n_S2=n_S/2 and n_O2=n_O/2. Thus S, S2, S3, S4, S5,
# S6, S7, S8, SO, SO2, SO3 and SSO use (1/2, 0), (1, 0), (3/2, 0),
# (2, 0), (5/2, 0), (3, 0), (7/2, 0), (4, 0), (1/2, 1/2),
# (1/2, 1), (1/2, 3/2) and (1, 1/2), respectively. Each equation balances
# sulfur and oxygen; positive n_O2 is consumed on the reactant side.
# Premise: one mole of E_aO_b(l) is the reactant. For
# E_aO_b(l) -> n_g E_cO_d(g) + n_O2 O2(g), element balance gives n_g=a/c;
# oxygen balance gives n_O2=(b-n_g*d)/2. Unit check: these are dimensionless
# stoichiometric coefficients, and O2 contributes two O atoms per mole.
# This yields Li2O(l) -> 2Li(g)+1/2O2, 2LiO(g)-1/2O2, Li2O(g), and
# Li2O2(g)-1/2O2; Rb2O(l) -> 2Rb(g)+1/2O2, 2RbO(g)-1/2O2, and Rb2O(g);
# PbO(l) -> Pb(g)+1/2O2, PbO(g), and PbO2(g)-1/2O2. Sanity: the balance
# test checks every trace atom count against these tuples.
# PbO(l) is the caller-supplied Pb(II) parent because it has a source-rated
# liquid standard state; PbO2 is retained as a gas product, with no evaluated
# PbO2(l) parent row in the source set used here.
# For these parent oxides, E_aO_b(l) balances with n_g=a/c and
# n_O2=(b-n_g*d)/2 for E_cO_d(g). Cs2O(l) and Cu2O(l) therefore give (2,1/2)
# for E, (2,-1/2) for EO, and (1,0) for E2 or E2O; SnO(l) gives (1,1/2),
# (1,0), and (1,-1/2) for Sn, SnO, and SnO2. Coefficients are dimensionless;
# the O2 term contributes two O atoms. Sanity: the tuples balance parent atoms.
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
    "Ta": ("Ta2O5", 2, 2.5),
    "TaO": ("Ta2O5", 2, 1.5),
    "TaO2": ("Ta2O5", 2, 0.5),
    "Be": ("BeO", 1, 0.5),
    "BeO": ("BeO", 1, 0.0),
    "Be2O": ("BeO", 0.5, 0.25),
    "(BeO)2": ("BeO", 0.5, 0.0),
    "(BeO)3": ("BeO", 1 / 3, 0.0),
    "(BeO)4": ("BeO", 0.25, 0.0),
    "Be2": ("BeO", 0.5, 0.5),
    "Tl": ("Tl2O", 2, 0.5),
    "Tl2O": ("Tl2O", 1, 0.0),
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
    "PS": ("P2O5", 2, 2.5),
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
    "Li": ("Li2O", 2, 0.5),
    "LiO": ("Li2O", 2, -0.5),
    "Li2O": ("Li2O", 1, 0.0),
    "Li2O2": ("Li2O", 1, -0.5),
    "Rb": ("Rb2O", 2, 0.5),
    "RbO": ("Rb2O", 2, -0.5),
    "Rb2O": ("Rb2O", 1, 0.0),
    "Pb": ("PbO", 1, 0.5),
    "PbO": ("PbO", 1, 0.0),
    "PbO2": ("PbO", 1, -0.5),
    # For a source liquid E_aO_b, n_gas=a/c and n_O2=(b-n_gas*d)/2.
    # This gives Ga2O3 -> 2Ga + 3/2O2, GeO2 -> GeO2, B2O3 -> 2BO2-1/2O2,
    # and In2O3 -> In2O + O2; the test checks every element balance.
    "Ga": ("Ga2O3", 2, 1.5),
    "GaO": ("Ga2O3", 2, 0.5),
    "Ga2O": ("Ga2O3", 1, 1.0),
    "Ge": ("GeO2", 1, 1.0),
    "GeO": ("GeO2", 1, 0.5),
    "GeO2": ("GeO2", 1, 0.0),
    "B": ("B2O3", 2, 1.5),
    "BO": ("B2O3", 2, 0.5),
    "BO2": ("B2O3", 2, -0.5),
    "B2O": ("B2O3", 1, 1.0),
    "B2O2": ("B2O3", 1, 0.5),
    "B2O3": ("B2O3", 1, 0.0),
    "In": ("In2O3", 2, 1.5),
    "InO": ("In2O3", 2, 0.5),
    "In2O": ("In2O3", 1, 1.0),
    "Cs": ("Cs2O", 2, 0.5),
    "CsO": ("Cs2O", 2, -0.5),
    "Cs2O": ("Cs2O", 1, 0.0),
    "Cu": ("Cu2O", 2, 0.5),
    "CuO": ("Cu2O", 2, -0.5),
    "Cu2": ("Cu2O", 1, 0.5),
    "Sn": ("SnO", 1, 0.5),
    "SnO": ("SnO", 1, 0.0),
    "SnO2": ("SnO", 1, -0.5),
    # E_aO_b(l) -> E_cO_d(g) + n_O2 O2(g), with n_gas=a/c and
    # n_O2=(b-n_gas*d)/2. ZnO and CdO parents give (1, 1/2) for E and (1, 0)
    # for EO. Bi2O3 gives (2, 3/2), (1, 3/2), and (2, 1/2) for Bi, Bi2, BiO.
    # These are dimensionless; the oxygen atoms balance explicitly.
    "Zn": ("ZnO", 1, 0.5),
    "ZnO": ("ZnO", 1, 0.0),
    "Cd": ("CdO", 1, 0.5),
    "CdO": ("CdO", 1, 0.0),
    "Bi": ("Bi2O3", 2, 1.5),
    "Bi2": ("Bi2O3", 1, 1.5),
    "BiO": ("Bi2O3", 2, 0.5),
    # Premise: TeO2(l) supplies one Te atom and two O atoms. Algebra: its
    # Te/Te2/TeO2 gas coefficients are 1, 1/2, and 1; the O2 coefficients
    # are (2 - n_gas*oxy_num)/2 = 1, 1, and 0. These are dimensionless; the
    # reaction-balance test checks Te and O atom counts independently.
    "Te": ("TeO2", 1, 1.0),
    "Te2": ("TeO2", 0.5, 1.0),
    "TeO2": ("TeO2", 1, 0.0),
    # SeO2(l) supplies one Se atom and two O atoms, so Se and Se2 release
    # one O2 each while SeO2(g) releases none.
    "Se": ("SeO2", 1, 1.0),
    "Se2": ("SeO2", 0.5, 1.0),
    "SeO": ("SeO2", 1, 0.5),
    "SeO2": ("SeO2", 1, 0.0),
    # As2O3(l) supplies two As and three O atoms; the selected gas rows balance
    # at n_gas=(2,1,1/2,1/2) and n_O2=(3/2,3/2,3/2,0).
    "As": ("As2O3", 2, 1.5),
    "As2": ("As2O3", 1, 1.5),
    "As4": ("As2O3", 0.5, 1.5),
    "As4O6": ("As2O3", 0.5, 0.0),
    # Sb2O3(l) is a tier-4 hypothetical activity standard constructed from
    # its crystal; atom and oxygen balances give (2,3/2), (1,3/2),
    # (1/2,3/2), and (1/2,0) for Sb, Sb2, Sb4, and Sb4O6 respectively.
    "Sb": ("Sb2O3", 2, 1.5),
    "Sb2": ("Sb2O3", 1, 1.5),
    "Sb4": ("Sb2O3", 0.5, 1.5),
    "Sb4O6": ("Sb2O3", 0.5, 0.0),
    # The source reports decomposition near 500 K, not a bulk liquid row;
    # Ag2O(l) is only the flagged caller-activity construction convention.
    "Ag": ("Ag2O", 2, 0.5),
    # For MoO3(l), balance gases per oxide parent unit: n_gas=1/cation_count
    # and n_O2=(3-n_gas*oxygen_count)/2. The dimensionless stoichiometry
    # closes Mo and O independently for monomers and oxide clusters.
    "Mo": ("MoO3", 1, 1.5),
    "MoO": ("MoO3", 1, 1.0),
    "MoO2": ("MoO3", 1, 0.5),
    "MoO3": ("MoO3", 1, 0.0),
    "(MoO3)2": ("MoO3", 0.5, 0.0),
    "(MoO3)3": ("MoO3", 1 / 3, 0.0),
    "(MoO3)4": ("MoO3", 0.25, 0.0),
    "(MoO3)5": ("MoO3", 0.2, 0.0),
    # WO3(l) supplies one W and three O atoms. The W3O8 row therefore needs
    # 1/3 mol gas and 1/6 mol O2 per WO3 parent unit; each (WO3)n is a
    # stoichiometric multiple of the parent and releases no O2.
    "W": ("WO3", 1, 1.5),
    "WO": ("WO3", 1, 1.0),
    "WO2": ("WO3", 1, 0.5),
    "WO3": ("WO3", 1, 0.0),
    "(WO3)2": ("WO3", 0.5, 0.0),
    "(WO3)3": ("WO3", 1 / 3, 0.0),
    "(WO3)4": ("WO3", 0.25, 0.0),
    "(WO3)5": ("WO3", 0.2, 0.0),
    "W3O8": ("WO3", 1 / 3, 1 / 6),
    # JANAF BaO, SrO and ZrO2 liquid parents. Premise: balance one mole of
    # parent oxide; n_gas=a/c and n_O2=(b-n_gas*d)/2. The dimensionless
    # tuples close metal and oxygen atoms independently.
    "Ba": ("BaO", 1, 0.5),
    "BaO": ("BaO", 1, 0.0),
    "Sr": ("SrO", 1, 0.5),
    "SrO": ("SrO", 1, 0.0),
    "Zr": ("ZrO2", 1, 1.0),
    "ZrO": ("ZrO2", 1, 0.5),
    "ZrO2": ("ZrO2", 1, 0.0),
    # NASA/Gurvich single-family channels. For Sc2O3, (n_g,n_O2) is
    # (2, 3/2), (2, 1/2), and (2, -1/2) for Sc, ScO, ScO2. For UO2 it is
    # (1, 1), (1, 1/2), (1, 0), and (1, -1/2) for U, UO, UO2, UO3.
    # Negative n_O2 means oxygen is a reactant; UO2(l)+1/2 O2(g)=UO3(g),
    # so the UO3 pressure rises with fO2.
    "Sc": ("Sc2O3", 2, 1.5),
    "ScO": ("Sc2O3", 2, 0.5),
    "ScO2": ("Sc2O3", 2, -0.5),
    "U": ("UO2", 1, 1.0),
    "UO": ("UO2", 1, 0.5),
    "UO2": ("UO2", 1, 0.0),
    "UO3": ("UO2", 1, -0.5),
    # ThO2(l) provides one Th and two O atoms: Th, ThO, and ThO2 balance
    # with 1, 1/2, and 0 O2 respectively. Coefficients are per parent mole.
    "Th": ("ThO2", 1, 1.0),
    "ThO": ("ThO2", 1, 0.5),
    "ThO2": ("ThO2", 1, 0.0),
    # HfO2(l) is the caller-supplied parent: Hf(g) balances with one O2.
    "Hf": ("HfO2", 1, 1.0),
    # Y2O3(l) -> 2Y(g)+3/2O2(g), and -> 2YO(g)+1/2O2(g).
    "Y": ("Y2O3", 2, 1.5),
    "YO": ("Y2O3", 2, 0.5),
    # La2O3(l) -> 2La(g)+3/2O2 and -> 2LaO(g)+1/2O2; atom balance
    # fixes the per-La2O3 gas coefficient at 2 and the O2 coefficients above.
    "La": ("La2O3", 2, 1.5),
    "LaO": ("La2O3", 2, 0.5),
    # Cerium's monoxide shares the sesquioxide parent; CeO2(g) uses the
    # source-paired CeO2(l) parent and therefore has no oxygen term.
    "Ce": ("Ce2O3", 2, 1.5),
    "CeO": ("Ce2O3", 2, 0.5),
    "CeO2": ("CeO2", 1, 0.0),
    "Pr": ("Pr2O3", 2, 1.5),
    "PrO": ("Pr2O3", 2, 0.5),
    "Nd": ("Nd2O3", 2, 1.5),
    "NdO": ("Nd2O3", 2, 0.5),
    # Premise: one mole of Sm2O3 supplies 2 Sm and 3 O atoms. Algebra: 2 Sm
    # leaves 3/2 O2; 2 SmO leaves 2 O atoms, or 1/2 O2. Unit check: reaction
    # coefficients are mol/mol parent. Sanity: both sides have 2 Sm and 3 O.
    "Sm": ("Sm2O3", 2, 1.5),
    "SmO": ("Sm2O3", 2, 0.5),
    # Premise: Eu2O3 and Gd2O3 each supply 2 metal and 3 oxygen atoms. The
    # 2 E(g) and 2 EO(g) products leave 3/2 and 1/2 O2 respectively; these
    # dimensionless per-formula coefficients close both atom balances.
    "Eu": ("Eu2O3", 2, 1.5),
    "EuO": ("Eu2O3", 2, 0.5),
    "Gd": ("Gd2O3", 2, 1.5),
    "GdO": ("Gd2O3", 2, 0.5),
    # Premise: E2O3 supplies two E and three O atoms. The monoxide leaves
    # one O atom per parent for 1/2 O2; the element leaves 3/2 O2. Coefficients
    # are mol/mol parent and independently balance both atom counts.
    "Tb": ("Tb2O3", 2, 1.5),
    "TbO": ("Tb2O3", 2, 0.5),
    "Dy": ("Dy2O3", 2, 1.5),
    "DyO": ("Dy2O3", 2, 0.5),
    "Ho": ("Ho2O3", 2, 1.5),
    "HoO": ("Ho2O3", 2, 0.5),
    "Er": ("Er2O3", 2, 1.5),
    "ErO": ("Er2O3", 2, 0.5),
    "Tm": ("Tm2O3", 2, 1.5),
    "TmO": ("Tm2O3", 2, 0.5),
    "Yb": ("Yb2O3", 2, 1.5),
    "YbO": ("Yb2O3", 2, 0.5),
    "Lu": ("Lu2O3", 2, 1.5),
    "LuO": ("Lu2O3", 2, 0.5),
    # Appended halogen families use X2(g) as their gas-parent activity.
    "Br2": ("Br2", 1, 0.0),
    "Br": ("Br2", 0.5, 0.0),
    "I2": ("I2", 1, 0.0),
    "I": ("I2", 0.5, 0.0),
    "NaBr": ("Br2", 0.5, 0.25),
    "(NaBr)2": ("Br2", 1.0, 0.5),
    "KBr": ("Br2", 0.5, 0.25),
    "(KBr)2": ("Br2", 1.0, 0.5),
    "NaI": ("I2", 0.5, 0.25),
    "(NaI)2": ("I2", 1.0, 0.5),
    "KI": ("I2", 0.5, 0.25),
    "(KI)2": ("I2", 1.0, 0.5),
    "Cl2": ("Cl2", 1, 0.0),
    "Cl": ("Cl2", 0.5, 0.0),
    "NaCl": ("Cl2", 0.5, 0.25),
    "(NaCl)2": ("Cl2", 1.0, 0.5),
    "KCl": ("Cl2", 0.5, 0.25),
    "(KCl)2": ("Cl2", 1.0, 0.5),
    "F2": ("F2", 1, 0.0),
    "F": ("F2", 0.5, 0.0),
    "NaF": ("F2", 0.5, 0.25),
    "(NaF)2": ("F2", 1.0, 0.5),
    "KF": ("F2", 0.5, 0.25),
    "(KF)2": ("F2", 1.0, 0.5),
    "HCl": ("H2", 0.5, 0.0),
    "HF": ("H2", 0.5, 0.0),
    "HBr": ("H2", 0.5, 0.0),
    "HI": ("H2", 0.5, 0.0),
    "H2": ("H2", 1, 0.0),
    "H": ("H2", 0.5, 0.0),
    "OH": ("H2", 0.5, 0.5),
    "H2O": ("H2", 1, 0.5),
    "CO": ("C", 1, -0.5),
    "CO2": ("C", 1, -1.0),
    "N2": ("N2", 1, 0.0),
    "N": ("N2", 0.5, 0.0),
    "NO": ("N2", 0.5, 0.5),
    "NO2": ("N2", 0.5, 1.0),
    "Hg": ("Hg", 1.0, 0.0),
    "Au": ("Au", 1.0, 0.0),
    "Pd": ("Pd", 1.0, 0.0),
    "Pt": ("Pt", 1.0, 0.0),
    "Ru": ("Ru", 1.0, 0.0),
    "Rh": ("Rh", 1.0, 0.0),
    "Ir": ("Ir", 1.0, 0.0),
    "PtO2": ("Pt", 1.0, -1.0),
}

# Append oxide + halogen reactions after every pre-existing channel. For MO,
# MO + X2 = MX2 + 1/2 O2 and MO + 1/2 X2 = MX + 1/2 O2; dimer rows scale
# both reactants by two. For M2O, the oxide and O2 coefficients are halved.
_HALIDE_REACTIONS: dict[str, tuple[str, float, float]] = {
    "BaBr": ("Br2", 0.5, 0.0),
    "BaBr2": ("Br2", 1, 0.0),
    "BaCl": ("Cl2", 0.5, 0.0),
    "BaCl2": ("Cl2", 1, 0.0),
    "BaF": ("F2", 0.5, 0.0),
    "BaF2": ("F2", 1, 0.0),
    "BaI": ("I2", 0.5, 0.0),
    "BaI2": ("I2", 1, 0.0),
    "CaBr": ("Br2", 0.5, 0.0),
    "CaBr2": ("Br2", 1, 0.0),
    "CaCl": ("Cl2", 0.5, 0.0),
    "CaCl2": ("Cl2", 1, 0.0),
    "CaF": ("F2", 0.5, 0.0),
    "CaF2": ("F2", 1, 0.0),
    "CaI": ("I2", 0.5, 0.0),
    "CaI2": ("I2", 1, 0.0),
    "MgF": ("F2", 0.5, 0.0),
    "MgF2": ("F2", 1, 0.0),
    "(MgF2)2": ("F2", 2, 0.0),
    "MgCl": ("Cl2", 0.5, 0.0),
    "MgCl2": ("Cl2", 1, 0.0),
    "(MgCl2)2": ("Cl2", 2, 0.0),
    "MgBr": ("Br2", 0.5, 0.0),
    "MgBr2": ("Br2", 1, 0.0),
    "(MgBr2)2": ("Br2", 2, 0.0),
    "MgI": ("I2", 0.5, 0.0),
    "MgI2": ("I2", 1, 0.0),
    "SrBr": ("Br2", 0.5, 0.0),
    "SrBr2": ("Br2", 1, 0.0),
    "SrCl": ("Cl2", 0.5, 0.0),
    "SrCl2": ("Cl2", 1, 0.0),
    "SrF": ("F2", 0.5, 0.0),
    "SrF2": ("F2", 1, 0.0),
    "SrI": ("I2", 0.5, 0.0),
    "SrI2": ("I2", 1, 0.0),
    "LiF": ("F2", 0.5, 0.0),
    "(LiF)2": ("F2", 1, 0.0),
    "LiCl": ("Cl2", 0.5, 0.0),
    "(LiCl)2": ("Cl2", 1, 0.0),
    "LiBr": ("Br2", 0.5, 0.0),
    "(LiBr)2": ("Br2", 1, 0.0),
    "LiI": ("I2", 0.5, 0.0),
    "(LiI)2": ("I2", 1, 0.0),
    "RbF": ("F2", 0.5, 0.0),
    "(RbF)2": ("F2", 1, 0.0),
    "RbCl": ("Cl2", 0.5, 0.0),
    "(RbCl)2": ("Cl2", 1, 0.0),
    "RbBr": ("Br2", 0.5, 0.0),
    "(RbBr)2": ("Br2", 1, 0.0),
    "RbI": ("I2", 0.5, 0.0),
    "(RbI)2": ("I2", 1, 0.0),
    "CsF": ("F2", 0.5, 0.0),
    "(CsF)2": ("F2", 1, 0.0),
    "CsCl": ("Cl2", 0.5, 0.0),
    "(CsCl)2": ("Cl2", 1, 0.0),
    "CsBr": ("Br2", 0.5, 0.0),
    "(CsBr)2": ("Br2", 1, 0.0),
    "CsI": ("I2", 0.5, 0.0),
    "(CsI)2": ("I2", 1, 0.0),
}
_SF04_REACTIONS.update(_HALIDE_REACTIONS)

# Parent activity keys omit phase suffixes. A gas-phase parent's G° is read
# from the JANAF gas table; condensed parent rows use their resolved phase.
_GAS_PHASE_PARENT_ROWS = {
    "S2": "S2(g)", "Br2": "Br2(g)", "I2": "I2(g)",
    "Cl2": "Cl2(g)", "F2": "F2(g)", "H2": "H2(g)", "N2": "N2(g)",
}
_CONDENSED_PARENT_ROWS = {"C": "C(cr)", "Hg": "Hg(l)"}
# Parents whose stable condensed reference changes phase inside the runtime
# domain. The tuple is (transition K, crystal row, liquid row); below the
# marker the crystal is selected, and at/above it the liquid is selected.
# Keep the phase rule beside the condensed-parent registry so new metal
# parents use the same selection path as existing condensed references.
_CONDENSED_PARENT_PHASE_ROWS: dict[str, tuple[float, str, str]] = {
    "Au": (1337.0, "Au(cr)", "Au(l)"),
    "Pd": (1827.0, "Pd(cr)", "Pd(l)"),
    "Pt": (2042.0, "Pt(cr)", "Pt(l)"),
    "Ru": (2607.0, "Ru(cr)", "Ru(l)"),
    "Rh": (2237.0, "Rh(cr)", "Rh(l)"),
    "Ir": (2720.0, "Ir(cr)", "Ir(l)"),
}
_ALKALI_HALIDE_REACTION_OXIDES = {
    "NaBr": ("Na2O", 0.5, 0.25),
    "(NaBr)2": ("Na2O", 1.0, 0.5),
    "KBr": ("K2O", 0.5, 0.25),
    "(KBr)2": ("K2O", 1.0, 0.5),
    "NaI": ("Na2O", 0.5, 0.25),
    "(NaI)2": ("Na2O", 1.0, 0.5),
    "KI": ("K2O", 0.5, 0.25),
    "(KI)2": ("K2O", 1.0, 0.5),
    "NaCl": ("Na2O", 0.5, 0.25),
    "(NaCl)2": ("Na2O", 1.0, 0.5),
    "KCl": ("K2O", 0.5, 0.25),
    "(KCl)2": ("K2O", 1.0, 0.5),
    "NaF": ("Na2O", 0.5, 0.25),
    "(NaF)2": ("Na2O", 1.0, 0.5),
    "KF": ("K2O", 0.5, 0.25),
    "(KF)2": ("K2O", 1.0, 0.5),
    # Each oxide parent in this table contains one O atom, so explicit oxygen
    # demand equals n_oxide. MO supplies one O per M2+; M2O supplies one O
    # per two M+ ions. With one product gas, the fugacity
    # exponent is -n_O2, giving matching signs and a monotone root.
    "BaBr": ("BaO", 1, 0.5),
    "BaBr2": ("BaO", 1, 0.5),
    "BaCl": ("BaO", 1, 0.5),
    "BaCl2": ("BaO", 1, 0.5),
    "BaF": ("BaO", 1, 0.5),
    "BaF2": ("BaO", 1, 0.5),
    "BaI": ("BaO", 1, 0.5),
    "BaI2": ("BaO", 1, 0.5),
    "CaBr": ("CaO", 1, 0.5),
    "CaBr2": ("CaO", 1, 0.5),
    "CaCl": ("CaO", 1, 0.5),
    "CaCl2": ("CaO", 1, 0.5),
    "CaF": ("CaO", 1, 0.5),
    "CaF2": ("CaO", 1, 0.5),
    "CaI": ("CaO", 1, 0.5),
    "CaI2": ("CaO", 1, 0.5),
    "MgF": ("MgO", 1, 0.5),
    "MgF2": ("MgO", 1, 0.5),
    "(MgF2)2": ("MgO", 2, 1),
    "MgCl": ("MgO", 1, 0.5),
    "MgCl2": ("MgO", 1, 0.5),
    "(MgCl2)2": ("MgO", 2, 1),
    "MgBr": ("MgO", 1, 0.5),
    "MgBr2": ("MgO", 1, 0.5),
    "(MgBr2)2": ("MgO", 2, 1),
    "MgI": ("MgO", 1, 0.5),
    "MgI2": ("MgO", 1, 0.5),
    "SrBr": ("SrO", 1, 0.5),
    "SrBr2": ("SrO", 1, 0.5),
    "SrCl": ("SrO", 1, 0.5),
    "SrCl2": ("SrO", 1, 0.5),
    "SrF": ("SrO", 1, 0.5),
    "SrF2": ("SrO", 1, 0.5),
    "SrI": ("SrO", 1, 0.5),
    "SrI2": ("SrO", 1, 0.5),
    "LiF": ("Li2O", 0.5, 0.25),
    "(LiF)2": ("Li2O", 1, 0.5),
    "LiCl": ("Li2O", 0.5, 0.25),
    "(LiCl)2": ("Li2O", 1, 0.5),
    "LiBr": ("Li2O", 0.5, 0.25),
    "(LiBr)2": ("Li2O", 1, 0.5),
    "LiI": ("Li2O", 0.5, 0.25),
    "(LiI)2": ("Li2O", 1, 0.5),
    "RbF": ("Rb2O", 0.5, 0.25),
    "(RbF)2": ("Rb2O", 1, 0.5),
    "RbCl": ("Rb2O", 0.5, 0.25),
    "(RbCl)2": ("Rb2O", 1, 0.5),
    "RbBr": ("Rb2O", 0.5, 0.25),
    "(RbBr)2": ("Rb2O", 1, 0.5),
    "RbI": ("Rb2O", 0.5, 0.25),
    "(RbI)2": ("Rb2O", 1, 0.5),
    "CsF": ("Cs2O", 0.5, 0.25),
    "(CsF)2": ("Cs2O", 1, 0.5),
    "CsCl": ("Cs2O", 0.5, 0.25),
    "(CsCl)2": ("Cs2O", 1, 0.5),
    "CsBr": ("Cs2O", 0.5, 0.25),
    "(CsBr)2": ("Cs2O", 1, 0.5),
    "CsI": ("Cs2O", 0.5, 0.25),
    "(CsI)2": ("Cs2O", 1, 0.5),
}
_HYDROGEN_HALIDE_PARENTS = {
    "HCl": "Cl2", "HF": "F2", "HBr": "Br2", "HI": "I2",
}
_MIXED_GAS_REACTION_SIDE_PARENTS = {"PS": ("S2", 1.0)}


def _parent_standard_rows(parent: str) -> tuple[str, ...]:
    """Return all condensed standard rows registered for an activity key."""
    phase_rows = _CONDENSED_PARENT_PHASE_ROWS.get(parent)
    if phase_rows is not None:
        _transition, crystal_row, liquid_row = phase_rows
        return crystal_row, liquid_row
    return (
        _GAS_PHASE_PARENT_ROWS.get(parent)
        or _CONDENSED_PARENT_ROWS.get(parent)
        or f"{parent}(l)",
    )


def _parent_standard_row(parent: str, T_K: float | None = None) -> str:
    """Resolve an activity key to its elemental-reference standard row."""
    phase_rows = _CONDENSED_PARENT_PHASE_ROWS.get(parent)
    if phase_rows is not None and T_K is not None:
        transition, crystal_row, liquid_row = phase_rows
        return crystal_row if T_K < transition else liquid_row
    return (
        _GAS_PHASE_PARENT_ROWS.get(parent)
        or _CONDENSED_PARENT_ROWS.get(parent)
        or f"{parent}(l)"
    )


def _nonoxide_parent_keys() -> frozenset[str]:
    """Return registered gas-phase and condensed non-oxide parent keys."""
    return frozenset(
        _GAS_PHASE_PARENT_ROWS
        | _CONDENSED_PARENT_ROWS
        | _CONDENSED_PARENT_PHASE_ROWS
    )


def _side_parent(species: str) -> str | None:
    halide = _HYDROGEN_HALIDE_PARENTS.get(species)
    if halide is not None:
        return halide
    alkali = _ALKALI_HALIDE_REACTION_OXIDES.get(species)
    if alkali is not None:
        return alkali[0]
    mixed = _MIXED_GAS_REACTION_SIDE_PARENTS.get(species)
    return mixed[0] if mixed is not None else None

IMCC_GAS_CHANNEL_SPECIES = tuple(_SF04_REACTIONS)

# Channels that a default evaluate_gas call includes only when the active
# datapack carries their gas row and their parent's standard-state row. The legacy
# tables selected by OPENIMCC_VAPOROCK_ROOT have Ti gas rows but no TiO2(l)
# row. Optional channels
# stay absent from a default call until an active pack supplies the parent
# activity and both table rows. Original SF04 channels stay strict: a table
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
        "Mo", "MoO", "MoO2", "MoO3",
        "(MoO3)2", "(MoO3)3", "(MoO3)4", "(MoO3)5",
        "W", "WO", "WO2", "WO3",
        "(WO3)2", "(WO3)3", "(WO3)4", "(WO3)5", "W3O8",
        "Ba", "BaO", "Sr", "SrO", "Zr", "ZrO", "ZrO2",
        "V",
        "VO",
        *_HALIDE_REACTIONS,
        "VO2",
        "Nb",
        "NbO",
        "NbO2",
        "Ta",
        "TaO",
        "TaO2",
        "Be",
        "BeO",
        "Be2O",
        "(BeO)2",
        "(BeO)3",
        "(BeO)4",
        "Be2",
        "Tl",
        "Tl2O",
        "Mn",
        "MnO",
        "Ni",
        "NiO",
        "Co",
        "CoO",
        "Zn",
        "ZnO",
        "Cd",
        "CdO",
        "Bi",
        "Bi2",
        "BiO",
        "Te",
        "Te2",
        "TeO2",
        "Se",
        "Se2",
        "SeO",
        "SeO2",
        "As",
        "As2",
        "As4",
        "As4O6",
        "Sb",
        "Sb2",
        "Sb4",
        "Sb4O6",
        "Ag",
        "P",
        "P2",
        "P4",
        "PO",
        "PO2",
        "P4O6",
        "P4O10",
        "PS",
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
        "Li",
        "LiO",
        "Li2O",
        "Li2O2",
        "Rb",
        "RbO",
        "Rb2O",
        "Pb",
        "PbO",
        "PbO2",
        "Ga",
        "GaO",
        "Ga2O",
        "Ge",
        "GeO",
        "GeO2",
        "B",
        "BO",
        "BO2",
        "B2O",
        "B2O2",
        "B2O3",
        "In",
        "InO",
        "In2O",
        "Cs",
        "CsO",
        "Cs2O",
        "Cu",
        "CuO",
        "Cu2",
        "Sn",
        "SnO",
        "SnO2",
        "Sc", "ScO", "ScO2",
        "U", "UO", "UO2", "UO3",
        "Th", "ThO", "ThO2",
        "Hf",
        "Y", "YO",
        "La", "LaO", "Ce", "CeO", "CeO2", "Pr", "PrO", "Nd", "NdO",
        "Sm", "SmO", "Eu", "EuO", "Gd", "GdO",
        "Tb", "TbO", "Dy", "DyO", "Ho", "HoO", "Er", "ErO",
        "Tm", "TmO", "Yb", "YbO", "Lu", "LuO",
        "Br2", "Br", "I2", "I", "NaBr", "(NaBr)2", "KBr", "(KBr)2",
        "NaI", "(NaI)2", "KI", "(KI)2",
        "Cl2", "Cl", "NaCl", "(NaCl)2", "KCl", "(KCl)2",
        "F2", "F", "NaF", "(NaF)2", "KF", "(KF)2",
        "BaBr",
        "BaBr2",
        "BaCl",
        "BaCl2",
        "BaF",
        "BaF2",
        "BaI",
        "BaI2",
        "CaBr",
        "CaBr2",
        "CaCl",
        "CaCl2",
        "CaF",
        "CaF2",
        "CaI",
        "CaI2",
        "MgF",
        "MgF2",
        "(MgF2)2",
        "MgCl",
        "MgCl2",
        "MgBr",
        "MgBr2",
        "(MgBr2)2",
        "MgI",
        "MgI2",
        "SrBr",
        "SrBr2",
        "SrCl",
        "SrCl2",
        "SrF",
        "SrF2",
        "SrI",
        "SrI2",
        "LiF",
        "(LiF)2",
        "LiCl",
        "(LiCl)2",
        "LiBr",
        "(LiBr)2",
        "LiI",
        "(LiI)2",
        "RbF",
        "(RbF)2",
        "RbCl",
        "(RbCl)2",
        "RbBr",
        "(RbBr)2",
        "RbI",
        "(RbI)2",
        "CsF",
        "(CsF)2",
        "CsCl",
        "(CsCl)2",
        "CsBr",
        "(CsBr)2",
        "CsI",
        "(CsI)2",
        "HCl", "HF", "HBr", "HI", "H2", "H", "OH", "H2O",
        "CO", "CO2", "N2", "N", "NO", "NO2",
        "Hg",
        "Au",
        "Pd",
        "Pt",
        "Ru",
        "Rh",
        "Ir",
        "PtO2",
    }
)


_REACTION_PARENT_OXIDES = frozenset(
    reaction[0] for reaction in _SF04_REACTIONS.values() if reaction[0]
)


def _default_reactions(
    available_parents: Sequence[str],
    datapack: ImccGasDatapack,
    T_K: float | None = None,
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
        side_key = _side_parent(name)
        if side_key and side_key not in parents:
            continue
        if name in _DATAPACK_OPTIONAL_CHANNELS:
            missing_rows = []
            gas_row = f"{name}(g)"
            gas_parent_row = _GAS_PHASE_PARENT_ROWS.get(oxide)
            if gas_row not in gas_rows:
                missing_rows.append(gas_row)
            parent_rows = gas_rows if gas_parent_row else oxide_rows
            oxide_rows_required = (
                _parent_standard_rows(oxide)
                if T_K is None and oxide not in _GAS_PHASE_PARENT_ROWS
                else (_parent_standard_row(oxide, T_K),)
            )
            for oxide_row in oxide_rows_required:
                if oxide_row != gas_row and oxide_row not in parent_rows:
                    missing_rows.append(oxide_row)
            if side_key:
                side_gas_row = _GAS_PHASE_PARENT_ROWS.get(side_key)
                side_row = _parent_standard_row(side_key, T_K)
                side_rows = gas_rows if side_gas_row else oxide_rows
                if side_row not in side_rows:
                    missing_rows.append(side_row)
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
# potassium channel inherits the K2O(l) estimated-systematic class while the
# other retained channels retain their source row classes. The Na2O/K2O gas
# rows are NASA/Gurvich fits; MnO and CoO(g) use published IVTAN tables.
_GAS_PROVENANCE_AUTHORITY = {
    species: "janaf_fitted"
    for species in (*IMCC_GAS_CHANNEL_SPECIES, "Mn", "Ni", "Co")
    if species not in {"NiO", "CoO"}
}
_GAS_PROVENANCE_AUTHORITY["Hg"] = "published_fitted"
_GAS_PROVENANCE_AUTHORITY["Au"] = "published_fitted"
_GAS_PROVENANCE_AUTHORITY["Pd"] = "published_fitted"
_GAS_PROVENANCE_AUTHORITY.update(
    {species: "published_fitted" for species in ("Pt", "PtO2", "Ru", "Rh", "Ir")}
)
_GAS_PROVENANCE_AUTHORITY.update(
    {
        "(MoO3)2": "nasa_glenn_fitted",
        "(MoO3)3": "nasa_glenn_fitted",
        "(MoO3)4": "nasa_glenn_fitted",
        "(MoO3)5": "nasa_glenn_fitted",
        "(WO3)2": "janaf_fitted",
        "(WO3)3": "janaf_fitted",
        "(WO3)4": "janaf_fitted",
        "(WO3)5": "nasa_glenn_fitted",
        "Na2O": "nasa_glenn_fitted",
        "K2O": "nasa_glenn_fitted",
        "NaI": "nasa_glenn_fitted",
        "(NaI)2": "nasa_glenn_fitted",
        "RbF": "nasa_glenn_fitted",
        "(RbF)2": "nasa_glenn_fitted",
        "RbCl": "nasa_glenn_fitted",
        "(RbCl)2": "nasa_glenn_fitted",
        "RbBr": "nasa_glenn_fitted",
        "(RbBr)2": "nasa_glenn_fitted",
        "RbI": "nasa_glenn_fitted",
        "(RbI)2": "nasa_glenn_fitted",
        "CsBr": "nasa_glenn_fitted",
        "(CsBr)2": "nasa_glenn_fitted",
        "CsI": "nasa_glenn_fitted",
        "(CsI)2": "nasa_glenn_fitted",
        "RbO": "nasa_glenn_fitted",
        "Rb2O": "nasa_glenn_fitted",
        "Be": "nasa_glenn_fitted",
        "BeO": "nasa_glenn_fitted",
        "Be2O": "nasa_glenn_fitted",
        "(BeO)2": "nasa_glenn_fitted",
        "(BeO)3": "nasa_glenn_fitted",
        "(BeO)4": "nasa_glenn_fitted",
        "Be2": "nasa_glenn_fitted",
        "Tl": "published_fitted",
        "Tl2O": "published_fitted",
        "PbO2": "nasa_glenn_fitted",
        "BO-": "nasa_glenn_fitted",
        "GaO": "nasa_glenn_fitted",
        "Ga2O": "nasa_glenn_fitted",
        "Ge": "nasa_glenn_fitted",
        "GeO": "nasa_glenn_fitted",
        "GeO2": "nasa_glenn_fitted",
        "In": "nasa_glenn_fitted",
        "InO": "nasa_glenn_fitted",
        "In2O": "nasa_glenn_fitted",
        "Sn": "nasa_glenn_fitted",
        "SnO": "nasa_glenn_fitted",
        "SnO2": "nasa_glenn_fitted",
        "Cd": "nasa_glenn_fitted",
        "ZnO": "published_fitted",
        "CdO": "published_fitted",
        "Bi": "published_fitted",
        "Bi2": "published_fitted",
        "BiO": "published_fitted",
        "Te": "published_fitted",
        "Te2": "published_fitted",
        "TeO2": "published_fitted",
        "Se": "published_fitted",
        "Se2": "published_fitted",
        "SeO": "published_fitted",
        "SeO2": "published_fitted",
        "As": "published_fitted",
        "As2": "published_fitted",
        "As4": "published_fitted",
        "As4O6": "published_fitted",
        "Sb": "published_fitted",
        "Sb2": "published_fitted",
        "Sb4": "published_fitted",
        "Sb4O6": "published_fitted",
        "Ag": "nasa_glenn_fitted",
        "MnO": "published_fitted",
        "NiO": "nasa_glenn_fitted",
        "CoO": "published_fitted",
        "Sc": "nasa_glenn_fitted",
        "ScO": "nasa_glenn_fitted",
        "ScO2": "nasa_glenn_fitted",
        "U": "nasa_glenn_fitted",
        "UO": "nasa_glenn_fitted",
        "UO2": "nasa_glenn_fitted",
        "UO3": "nasa_glenn_fitted",
        "Th": "published_fitted",
        "ThO": "published_fitted",
        "ThO2": "published_fitted",
        "Hf": "janaf_fitted",
        "Y": "published_fitted",
        "YO": "published_fitted",
    }
)
_GAS_PROVENANCE_AUTHORITY.update(
    {
        species: "published_fitted"
        for species in (
            "La", "LaO", "Ce", "CeO", "CeO2", "Pr", "PrO", "Nd", "NdO",
            "Sm", "SmO", "Eu", "EuO", "Gd", "GdO",
            "Tb", "TbO", "Dy", "DyO", "Ho", "HoO", "Er", "ErO",
            "Tm", "TmO", "Yb", "YbO", "Lu", "LuO",
        )
    }
)
_ION_GAS_SPECIES = (
    "Na+", "K+", "Ca+", "e-",
    "Na-", "K-", "O-", "Al-", "Fe-", "Si-", "Ti-", "O2-", "AlO-", "AlO2-", "KO-", "NaO-",
    "Cr-", "V-", "Nb-", "Li-", "LiO-", "Rb-", "Pb-",
    "Li+", "Rb+", "Pb+", "Hf+", "Hf-", "Th+",
    "Ga+", "Ge+", "B+", "Ga-", "B-", "BO-", "BO2-",
    "Cs-", "Cu-", "Cs+", "Sn+", "Cs2O+",
    "Zn+", "Zn-", "Cd+", "Bi+",
    "Mn+", "Ni+", "Ni-", "Co+", "Co-", "Ag+", "Ag-", "Mo+", "Mo-", "W+", "W-",
    "Ba+", "Sr+", "Zr+", "Zr-", "Sc+",
    "Hg+", "Au+", "Au-", "Pd+", "Pt+", "Pt-",
)
_GAS_PROVENANCE_AUTHORITY.update(
    {
        species: (
            "nasa_glenn_fitted"
            if species in {"Sn+", "Cs2O+"}
            else "janaf_fitted_ionisation"
        )
        for species in _ION_GAS_SPECIES
    }
)
_GAS_PROVENANCE_AUTHORITY.update(
    {
        species: "published_fitted_ionisation"
        for species in ("Th+", "Au+", "Au-", "Pd+", "Pt+", "Pt-")
    }
)
_NASA_ION_GAS_SPECIES = frozenset({"Ge+", "BO-"})
_GAS_PROVENANCE_AUTHORITY.update(
    {
        **{
            species: "nasa_glenn_fitted_ionisation"
            for species in (*_NASA_ION_GAS_SPECIES, "Cd+")
        },
        "Bi+": "nist_partition_fitted_ionisation",
        "Ag+": "nasa_glenn_fitted_ionisation",
        "Ag-": "nasa_glenn_fitted_ionisation",
        "Sc+": "nasa_glenn_fitted_ionisation",
    }
)
_OXIDE_PROVENANCE_AUTHORITY = {
    "C(cr)": "janaf_fitted",
    "Hg": "published_fitted",
    "Au": "published_fitted",
    "Pd": "published_fitted",
    "Pt(cr)": "published_fitted",
    "Pt": "published_fitted",
    "Ru(cr)": "published_fitted",
    "Ru": "published_fitted",
    "Rh(cr)": "published_fitted",
    "Rh": "published_fitted",
    "Ir(cr)": "published_fitted",
    "Ir": "published_fitted",
    "MnO": "liquid_from_solid_estimate",
    "NiO": "published_fitted",
    "CoO": "published_fitted",
    "P2O5": "published_fitted",
    "MgO": "janaf_fitted",
    "CaO": "janaf_fitted",
    "Al2O3": "janaf_fitted",
    "SiO2": "janaf_fitted",
    "SiO2(cr)": "lam1987_transcribed",
    "Na2O": "janaf_fitted",
    "K2O": "estimated_systematic",
    "FeO": "janaf_transcribed",
    "TiO2": "janaf_fitted",
    "Cr2O3": "janaf_fitted",
    "V2O3": "janaf_fitted",
    "NbO2": "janaf_fitted",
    "Ta2O5": "janaf_fitted",
    "BeO": "nasa_glenn_fitted",
    "Tl2O": "published_fitted",
    "Li2O": "janaf_fitted",
    "Rb2O": "nasa_glenn_fitted",
    "PbO": "janaf_fitted",
    "BaO": "janaf_fitted",
    "SrO": "janaf_fitted",
    "ZrO2": "janaf_fitted",
    "B2O3": "janaf_fitted",
    "Ga2O3": "nasa_glenn_fitted",
    "GeO2": "nasa_glenn_fitted",
    "In2O3": "nasa_glenn_fitted",
    "Cs2O": "nasa_glenn_fitted",
    "Cu2O": "janaf_fitted",
    "SnO": "nasa_glenn_fitted",
    "ZnO": "liquid_from_solid_estimate",
    "CdO": "liquid_from_solid_estimate",
    "Bi2O3": "published_fitted",
    "TeO2": "published_fitted",
    "SeO2": "liquid_from_solid_estimate",
    "As2O3": "liquid_from_solid_estimate",
    "Sb2O3": "liquid_from_solid_estimate",
    "Ag2O": "liquid_from_solid_estimate",
    "MoO3": "janaf_fitted",
    "WO3": "janaf_fitted",
    "Sc2O3": "nasa_glenn_fitted",
    "UO2": "nasa_glenn_fitted",
    "ThO2": "liquid_from_solid_estimate",
    "HfO2": "liquid_from_solid_estimate",
    "Y2O3": "liquid_from_solid_estimate",
    "La2O3": "published_fitted",
    "Ce2O3": "published_fitted",
    "CeO2": "published_fitted",
    "Pr2O3": "published_fitted",
    "Nd2O3": "published_fitted",
    "Sm2O3": "published_fitted",
    "Eu2O3": "published_fitted",
    "Gd2O3": "published_fitted",
    "Tb2O3": "published_fitted",
    "Dy2O3": "published_fitted",
    "Ho2O3": "published_fitted",
    "Er2O3": "published_fitted",
    "Tm2O3": "published_fitted",
    "Yb2O3": "published_fitted",
    "Lu2O3": "published_fitted",
}
# Mirrors the source table identifiers recorded for packaged rows in
# data/gas/PROVENANCE.yaml. K2O is resolved from the active row because its
# table is K-012 by default and LAM1984 in sf04-published.
_OXIDE_SOURCE_TABLE_IDS = {
    "C": "C-002",
    "Hg": "BU-2586",
    "Au": "BU-0055",
    "Pd": "BU-3111",
    "Pt": "BU-3121",
    "Ru": "P672-Ru-l",
    "Rh": "P672-Rh-l",
    "Ir": "BU-2612",
    "P2O5": "Jung-Hudon-2012",
    "Na2O": "Na-013",
    "MgO": "Mg-009",
    "CaO": "Ca-028",
    "Al2O3": "Al-100",
    "SiO2": "O-038",
    "FeO": "Fe-019",
    "TiO2": "O-044",
    "Cr2O3": "Cr-015",
    "V2O3": "O-063",
    "NbO2": "Nb-013",
    "Ta2O5": "O-078",
    "BeO": "NG-BeO-L-1352",
    "Tl2O": "P672-Tl2O-l",
    "Li2O": "Li-015",
    "MoO3": "Mo-015",
    "WO3": "O-066",
    "Rb2O": "NG-1841",
    "PbO": "O-007",
    "B2O3": "B-096",
    "Ga2O3": "NG-12403",
    "GeO2": "NG-12434",
    "In2O3": "NG-12662",
    "Cs2O": "NG-1842",
    "Cu2O": "Cu-020",
    "SnO": "NG-1848",
    "ZnO": "TSIV-1603",
    "NiO": "TSIV-2026",
    "CoO": "TSIV-1334",
    "CdO": "P672-T102",
    "Bi2O3": "P672-T82",
    "TeO2": "P672-TeO2-l",
    "SeO2": "P672-SeO2",
    "Sc2O3": "NG-1857",
    "UO2": "NG-1977",
    "As2O3": "P672-As2O3",
    "Sb2O3": "P672-Sb2O3-c",
    "Ag2O": "P672-Ag2O-c",
    "ThO2": "P672-ThO2-c",
    "HfO2": "P672-HfO2-c",
    "Y2O3": "P672-Y2O3-c",
    "MnO": "MnO-liquid-construction",
    "La2O3": "Konings-Table88-Tf2577",
    "Ce2O3": "Konings-Table88-Tf2512",
    "CeO2": "Konings-Table88-Tf3083",
    "Pr2O3": "Konings-Table88-Tf2583",
    "Nd2O3": "Konings-Table88-Tf2577",
    "Sm2O3": "Konings-Table88-Tf2613",
    "Eu2O3": "Konings-Table88-Tf2622",
    "Gd2O3": "Konings-Table88-Tf2693",
    "Tb2O3": "Konings-Table88-Tf2682",
    "Dy2O3": "Konings-Table88-Tf2680",
    "Ho2O3": "Konings-Table88-Tf2686",
    "Er2O3": "Konings-Table88-Tf2690",
    "Tm2O3": "Konings-Table88-Tf2682",
    "Yb2O3": "Konings-Table88-Tf2707",
    "Lu2O3": "Konings-Table88-Tf2762",
}
_LAM1987_SOURCE_TABLE_IDS = {
    "MgO": "LH87 Table 2",
    "CaO": "LH87 Table 2",
    "Al2O3": "LH87 Table 2 / Table 3",
    "SiO2": "LH87 Table 2",
}
_PROVENANCE_AUTHORITY_RANK = {
    "secondary_transcription_unverified_primary": 0,
    "estimated_systematic": 1,
    "lam1984_transcribed": 1,
    "lam1987_transcribed": 1,
    "janaf_transcribed": 2,
    "janaf_fitted": 3,
    "janaf_fitted_ionisation": 3,
    "nasa_glenn_fitted": 3,
    "nasa_glenn_fitted_ionisation": 3,
    "external_datapack": 2,
    "published_fitted": 2,
    "published_fitted_ionisation": 2,
    "liquid_from_solid_estimate": 1,
    "nist_partition_fitted_ionisation": 3,
}


def gas_species_provenance(species: str) -> dict[str, str | None]:
    """Return the row authorities feeding one retained gas channel.

    The overall authority is the least-authoritative row in that reaction.
    """
    if species in _ION_GAS_SPECIES:
        authority = (
            "nasa_glenn_fitted_ionisation"
            if species in _NASA_ION_GAS_SPECIES
            else "janaf_fitted_ionisation"
        )
        authority = _GAS_PROVENANCE_AUTHORITY[species]
        return {
            "species_name": species,
            "authority": authority,
            "gas_authority": authority,
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
    condensate_authority = oxide_authority
    authorities = [gas_authority]
    if oxide_authority is not None:
        authorities.append(oxide_authority)
    side_reactant = _ALKALI_HALIDE_REACTION_OXIDES.get(species)
    if side_reactant is not None:
        condensate_authority = _OXIDE_PROVENANCE_AUTHORITY[side_reactant[0]]
        authorities.append(condensate_authority)
    mixed_side = _MIXED_GAS_REACTION_SIDE_PARENTS.get(species)
    if mixed_side is not None:
        authorities.append(_GAS_PROVENANCE_AUTHORITY[mixed_side[0]])
    authority = min(authorities, key=_PROVENANCE_AUTHORITY_RANK.__getitem__)
    return {
        "species_name": species,
        "authority": authority,
        "gas_authority": gas_authority,
        "condensate_authority": condensate_authority,
    }


def _result_provenance_authority(
    species: str, oxide_row: pd.Series | None
) -> str:
    """Use the active parent row when a loaded pack changes its authority."""
    provenance = gas_species_provenance(species)
    if oxide_row is not None:
        row_authority = {
            "LAM1984": "lam1984_transcribed",
            "LAM1987": "lam1987_transcribed",
        }.get(str(oxide_row["Ref"]))
        if row_authority is not None:
            return min(
                (str(provenance["gas_authority"]), row_authority),
                key=_PROVENANCE_AUTHORITY_RANK.__getitem__,
            )
    return str(provenance["authority"])

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

# No default channel misses the workbook temperature grid. Results using the
# generated supercooled-liquid rows still carry their row-specific notice.
IMCC_GAS_WORKBOOK_EXTRAPOLATION_LABELS: dict[str, str] = {}

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
    "Co",
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
    "SiO",
    "Mg",
    "MgO",
    "SiO2",
    "AlO",
    "AlO2",
    "Al2O",
    "Al2O2",
    "Al2",
    "Si",
    "Si2",
    "Si3",
    "Al",
    "CaO",
    "Ca",
)

# Closure ledger for species outside the retained channel set. These entries
# still lack a gas G(T) row in the vendored source set or require a model
# convention not present in the layer. Na2O/K2O left this ledger when their
# NASA/Gurvich gas rows joined the package. Mn/Ni/Co have retained optional
# channels activated by caller-supplied parent activities.
IMCC_GAS_NO_JANAF_ROWS: dict[str, str] = {
    "TeO": "needs a source; none was located in Bulletin 672, JANAF, NASA Glenn/CEA, NIST WebBook, or Gurvich/IVTAN",
    "AsO": "needs a source; none was located in Bulletin 672, JANAF, NASA Glenn/CEA, NIST WebBook, or Gurvich/IVTAN",
}

_GAS_IONIZATION_PAIRS = (
    ("Na", "Na+"),
    ("K", "K+"),
    ("Ca", "Ca+"),
    ("Li", "Li+"),
    ("Rb", "Rb+"),
    ("Pb", "Pb+"),
    ("Ga", "Ga+"),
    ("Ge", "Ge+"),
    ("B", "B+"),
    ("Cs", "Cs+"),
    ("Sn", "Sn+"),
    ("Cs2O", "Cs2O+"),
    ("Zn", "Zn+"),
    ("Cd", "Cd+"),
    ("Bi", "Bi+"),
    ("Mn", "Mn+"),
    ("Ni", "Ni+"),
    ("Co", "Co+"),
    ("Ag", "Ag+"),
    ("Mo", "Mo+"),
    ("W", "W+"),
    ("Ba", "Ba+"),
    ("Sr", "Sr+"),
    ("Zr", "Zr+"),
    ("Sc", "Sc+"),
    ("Hf", "Hf+"),
    ("Th", "Th+"),
    ("Hg", "Hg+"),
    ("Au", "Au+"),
    ("Pd", "Pd+"),
    ("Pt", "Pt+"),
)
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
    ("Li", "Li-"),
    ("LiO", "LiO-"),
    ("Rb", "Rb-"),
    ("Pb", "Pb-"),
    ("Ga", "Ga-"),
    ("B", "B-"),
    ("BO", "BO-"),
    ("BO2", "BO2-"),
    ("Cs", "Cs-"),
    ("Cu", "Cu-"),
    ("Zn", "Zn-"),
    ("Ag", "Ag-"),
    ("Mo", "Mo-"),
    ("W", "W-"),
    ("Zr", "Zr-"),
    ("Ni", "Ni-"),
    ("Co", "Co-"),
    ("Hf", "Hf-"),
    ("Au", "Au-"),
    ("Pt", "Pt-"),
)
_ION_PROVENANCE_CLASS = "janaf_fitted_ionisation"

IMCC_GAS_INCOMPLETE_PARENT_SPECIES: dict[str, str] = {}

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

# The default major parents cover 1200-3000 K using labelled continuations
# below their fits. Added low-temperature gas intervals close the lower C4 gaps
# for Si, Al, Ti, V and Nb; Mg and Ca also have full C4 coverage.
ELEMENT_STATUS: dict[str, dict[str, object]] = {   'O': {   'status': 'input (fO2 pinned)',
             'criteria': {'C1': False, 'C2': True, 'C3': False, 'C4': False},
             'validation': 'unvalidated',
             'reason': 'O is input (fO2 pinned), so C1, C3 and C4 do not apply.',
             'c2_candidates': ()},
    'P': {   'status': 'gas-complete-melt-pending',
             'criteria': {'C1': False, 'C2': True, 'C3': False, 'C4': True},
             'validation': 'unvalidated',
             'reason': 'Jung and Hudon (2012) supply the assessed P2O5(l) reaction function, fitted '
                       'here to JANAF\'s 1-bar P2(g)/O2(g) references over 1200-3000 K. JANAF P, P2, '
                       'P4, PO, PO2, P4O6, P4O10, and PS(g) channels are available; PS also uses '
                       'caller-supplied a(S2). C1 stays no because phosphorus melt activity is not '
                       'computed. sp_extension P2O5 melt activity uses half-P4O10(cr), a different '
                       'reference from this liquid standard.',
             'c3_ion_bound': {   'status': 'not computed',
                                 'reason': 'not computed: ionized phosphorus channels are outside '
                                           'this neutral gas-only change'},
             'c2_candidates': (   ('O-004', 'PO'),
                                  ('O-032', 'PO2'),
                                  ('O-087', 'P4O6'),
                                  ('O-095', 'P4O10'))},
    'S': {   'status': 'gas-complete-melt-pending',
             'criteria': {'C1': False, 'C2': True, 'C3': False, 'C4': True},
             'validation': 'unvalidated',
             'reason': 'JANAF sulfur gas rows and the S2(g) parent cover 500-3000 K; callers '
                       'supply a(S2)=f(S2)/p° on the 1-bar gas reference. OpenIMCC provides no '
                       'pyrolysis-temperature sulfur melt activity model.',
             'c3_ion_bound': {   'status': 'not computed',
                                 'reason': 'not computed: ionized sulfur channels are outside this '
                                           'neutral gas-only change'},
             'c2_candidates': (   ('O-010', 'SO'),
                                  ('O-034', 'SO2'),
                                  ('O-058', 'SO3'),
                                  ('O-011', 'SSO'))},
    'Si': {   'status': 'complete',
              'criteria': {'C1': True, 'C2': True, 'C3': True, 'C4': True},
              'validation': 'validated',
              'reason': 'The JANAF SiO2(l) parent covers 1200-3000 K with a labelled continuation '
                        'below 1800 K; Si2(g) and Si3(g) now cover the 1200-1500 K interval; joint '
                        'C3 closure-screen maximum 5.729e-14 at 2800 K and fO2=1e-4 '
                        'within the <=1-bar neutral-pressure domain; unmodeled positive molecular '
                        'ions and thermal electrons from walls or other sources remain outside the '
                        'estimate.',
              'c2_candidates': (('O-012', 'SiO'), ('O-040', 'SiO2')),
              'c3_ion_bound': {   'max_ratio': 5.751521582586683e-14,
                                  'isolated_bound': 635.773963217761,
                                  'temperature_K': 2800.0,
                                  'fO2': 0.0001,
                                  'neutral_pressure_bar': 4.084360439777764e-06,
                                  'element_total_pressure_bar': 0.3279045980715964,
                                  'joint_ion_pressure_bar': 1.885950372838198e-14,
                                  'electron_pressure_bar': 8.131780113388637e-05,
                                  'K_ion': 3.754843374576903e-13,
                                  'parent_oxide': 'SiO2',
                                  'source_tables': {   'cation': 'Si-006',
                                                       'neutral': 'Si-005',
                                                       'electron': 'D-020'},
                                  'upstream_sha256': {   'Si-006': '6d81642518890fb4c67840b3bb103582ad037e800e312cd2fb5cf227cb7ffd58',
                                                         'D-020': 'c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd'},
                                  'user_agent': 'openimcc-janaf-vendor/1.0'}},
    'Mg': {   'status': 'complete',
              'criteria': {'C1': True, 'C2': True, 'C3': True, 'C4': True},
              'validation': 'unvalidated',
              'reason': 'The JANAF MgO(l) row and Mg gas rows cover 1200-3000 K, including a '
                        'labelled parent continuation below 2200 K; joint C3 closure-screen '
                        'maximum 1.154e-07 at 2800 K and fO2=1e-4 within the <=1-bar '
                        'neutral-pressure domain; unmodeled positive molecular ions and thermal '
                        'electrons from walls or other sources remain outside the estimate.',
              'c2_candidates': (('Mg-011', 'MgO'),),
              'c3_ion_bound': {   'max_ratio': 1.1586897049429865e-07,
                                  'isolated_bound': 7.252406196525388e-05,
                                  'temperature_K': 2800.0,
                                  'fO2': 0.0001,
                                  'neutral_pressure_bar': 0.0018164954020862998,
                                  'element_total_pressure_bar': 0.0018419594771682164,
                                  'joint_ion_pressure_bar': 2.1342594831169783e-10,
                                  'electron_pressure_bar': 8.131780113388637e-05,
                                  'K_ion': 9.55429273406838e-12,
                                  'parent_oxide': 'MgO',
                                  'source_tables': {   'cation': 'Mg-006',
                                                       'neutral': 'Mg-005',
                                                       'electron': 'D-020'},
                                  'upstream_sha256': {   'Mg-006': '1525bdbc6926d9fab8c79ee002d1968c4d5597ca446049149886d79119787c3e',
                                                         'D-020': 'c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd'},
                                  'user_agent': 'openimcc-janaf-vendor/1.0'}},
    'Fe': {   'status': 'complete',
              'criteria': {'C1': True, 'C2': True, 'C3': True, 'C4': True},
              'validation': 'unvalidated',
              'reason': 'FeO and neutral gas rows cover the domain; joint C3 closure-screen '
                        'maximum 3.453e-08 at 2800 K and fO2=1e-4 within the <=1-bar '
                        'neutral-pressure domain; unmodeled positive molecular ions and thermal '
                        'electrons from walls or other sources remain outside the estimate.',
              'c2_candidates': (('Fe-021', 'FeO'),),
              'c3_ion_bound': {   'max_ratio': 3.466855459663526e-08,
                                  'isolated_bound': 9.173414529115984e-06,
                                  'temperature_K': 2800.0,
                                  'fO2': 0.0001,
                                  'neutral_pressure_bar': 0.03469015861977115,
                                  'element_total_pressure_bar': 0.03592139938234394,
                                  'joint_ion_pressure_bar': 1.245342995674331e-09,
                                  'electron_pressure_bar': 8.131780113388637e-05,
                                  'K_ion': 2.9192300668238232e-12,
                                  'parent_oxide': 'FeO',
                                  'source_tables': {   'cation': 'Fe-009',
                                                       'neutral': 'Fe-008',
                                                       'electron': 'D-020'},
                                  'upstream_sha256': {   'Fe-009': 'd4979c41394748f0ffea48163a8a792e5e4f763989426f6cd4ba198ee3e532ab',
                                                         'D-020': 'c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd'},
                                  'user_agent': 'openimcc-janaf-vendor/1.0'}},
    'Ca': {   'status': 'complete',
              'criteria': {'C1': True, 'C2': True, 'C3': True, 'C4': True},
              'validation': 'unvalidated',
              'reason': 'The JANAF CaO(l) row and Ca gas rows cover 1200-3000 K, including a '
                        'labelled parent continuation below 2200 K; fitted Ca+ channel maximum '
                        'p(Ca+)/neutral Ca gas 6.519e-05 at 2800 K and fO2=1e-4 within the <=1-bar '
                        'neutral-pressure domain, below the 1e-4 C3 threshold.',
              'c2_candidates': (('Ca-030', 'CaO'),),
              'c3_ion_bound': {   'max_ratio': 6.545229766296126e-05,
                                  'isolated_bound': 12.158555162056784,
                                  'temperature_K': 2800.0,
                                  'fO2': 0.0001,
                                  'neutral_pressure_bar': 2.264780230212865e-06,
                                  'element_total_pressure_bar': 2.3374848210341417e-06,
                                  'joint_ion_pressure_bar': 1.5299375228898038e-10,
                                  'electron_pressure_bar': 8.131780113388637e-05,
                                  'K_ion': 5.493299242634703e-09,
                                  'parent_oxide': 'CaO',
                                  'source_tables': {   'cation': 'Ca-007',
                                                       'neutral': 'Ca-006',
                                                       'electron': 'D-020'},
                                  'upstream_sha256': {   'Ca-007': 'ddd7d08df4aaf34beeaead89af83e7281f850eb453e859483f16520d61e98724',
                                                         'D-020': 'c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd'},
                                  'user_agent': 'openimcc-janaf-vendor/1.0'}},
    'Al': {   'status': 'complete',
              'criteria': {'C1': True, 'C2': True, 'C3': True, 'C4': True},
              'validation': 'unvalidated',
              'reason': 'The JANAF Al2O3(l) parent covers 1200-3000 K with a labelled continuation '
                        'below 2500 K; Al2(g) now covers the 1200-1500 K interval; joint C3 '
                        'closure-screen maximum 4.273e-06 at 2800 K and fO2=1e-4 within the '
                        '<=1-bar neutral-pressure domain; unmodeled positive molecular ions and '
                        'thermal electrons from walls or other sources remain outside the '
                        'estimate.',
              'c2_candidates': (   ('Al-074', 'AlO'),
                                   ('Al-077', 'AlO2'),
                                   ('Al-092', 'Al2O'),
                                   ('Al-094', 'Al2O2')),
              'c3_ion_bound': {   'max_ratio': 4.28942087279601e-06,
                                  'isolated_bound': 30254.271715108043,
                                  'temperature_K': 2800.0,
                                  'fO2': 0.0001,
                                  'neutral_pressure_bar': 4.796751744263063e-06,
                                  'element_total_pressure_bar': 1.3616359923229827e-05,
                                  'joint_ion_pressure_bar': 5.84062984662051e-11,
                                  'electron_pressure_bar': 8.131780113388637e-05,
                                  'K_ion': 9.90143333834541e-10,
                                  'parent_oxide': 'Al2O3',
                                  'source_tables': {   'cation': 'Al-006',
                                                       'neutral': 'Al-005',
                                                       'electron': 'D-020'},
                                  'upstream_sha256': {   'Al-006': '7e4e45bc108863c4baed5ca1bffe6b931a78a050af86d50a2fcbbe7b9251c6dd',
                                                         'D-020': 'c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd'},
                                  'user_agent': 'openimcc-janaf-vendor/1.0'}},
    'Ti': {   'status': 'complete',
              'criteria': {'C1': True, 'C2': True, 'C3': True, 'C4': True},
              'validation': 'unvalidated',
              'reason': 'A labelled constant-Cp TiO2(l) continuation covers 1200-1500 K below the '
                        'glass branch; Ti(g), TiO(g), and TiO2(g) now cover the lower gas interval; '
                        'joint C3 closure-screen maximum 1.159e-09 at 2300 K and '
                        'fO2=1e-10 within the <=1-bar neutral-pressure domain; unmodeled positive '
                        'molecular ions and thermal electrons from walls or other sources remain '
                        'outside the estimate.',
              'c2_candidates': (('O-022', 'TiO'), ('O-046', 'TiO2')),
              'c3_ion_bound': {   'max_ratio': 1.166768940375097e-09,
                                  'isolated_bound': 143072179.55520213,
                                  'temperature_K': 2300.0,
                                  'fO2': 1e-10,
                                  'neutral_pressure_bar': 1.2935270323764889e-08,
                                  'element_total_pressure_bar': 8.992765922553097e-07,
                                  'joint_ion_pressure_bar': 1.0492479966498558e-15,
                                  'electron_pressure_bar': 4.935841167923758e-06,
                                  'K_ion': 4.0037210878469924e-13,
                                  'parent_oxide': 'TiO2',
                                  'source_tables': {   'cation': 'Ti-007',
                                                       'neutral': 'Ti-006',
                                                       'electron': 'D-020'},
                                  'upstream_sha256': {   'Ti-007': 'ea066f88e882e8f102f5499c5bc6b58ffdeb8a0d9f50295afe60af0236eb5c1e',
                                                         'D-020': 'c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd'},
                                  'user_agent': 'openimcc-janaf-vendor/1.0'}},
    'Na': {   'status': 'complete-except-ions',
              'criteria': {'C1': True, 'C2': True, 'C3': False, 'C4': True},
              'validation': 'unvalidated',
              'reason': 'Na2O(g) intervals cover 500-3000 K and JANAF Na-013 Na2O(l) intervals '
                        'cover 1200-3000 K, including its supercooled-liquid branch; fitted Na+ '
                        'channel maximum p(Na+)/neutral Na gas 9.568e-04 at 2800 K and fO2=1e-4 '
                        'remains above 1e-4; ions are opt-in and omitted from default results.',
              'c2_candidates': (('Na-008', 'NaO'),),
              'c3_ion_bound': {   'max_ratio': 0.000960621259803381,
                                  'isolated_bound': 0.0009246793240051793,
                                  'temperature_K': 2800.0,
                                  'fO2': 0.0001,
                                  'neutral_pressure_bar': 0.09147641574868175,
                                  'element_total_pressure_bar': 0.09159309879517884,
                                  'joint_ion_pressure_bar': 8.798627795392024e-05,
                                  'electron_pressure_bar': 8.131780113388637e-05,
                                  'K_ion': 7.82152491941164e-08,
                                  'parent_oxide': 'Na2O',
                                  'source_tables': {   'cation': 'Na-006',
                                                       'neutral': 'Na-005',
                                                       'electron': 'D-020'},
                                  'upstream_sha256': {   'Na-006': '1324311c226aaef128378768ce75d1f45c15949219359b3388efda2822db1159',
                                                         'D-020': 'c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd'},
                                  'user_agent': 'openimcc-janaf-vendor/1.0'}},
    'K': {   'status': 'complete-except-ions',
             'criteria': {'C1': True, 'C2': True, 'C3': False, 'C4': True},
             'validation': 'validated',
             'reason': 'K2O(g) intervals cover 500-3000 K and K2O(l) covers the C4 domain; fitted '
                       'K+ channel maximum p(K+)/neutral K gas 4.789e-02 at 1200 K and fO2=1e-4 '
                       'remains above 1e-4; ions are opt-in and omitted from default results.',
             'c2_candidates': (('K-008', 'KO'),),
             'c3_ion_bound': {   'max_ratio': 0.047888815560848744,
                                 'isolated_bound': 0.257637968186603,
                                 'temperature_K': 1200.0,
                                 'fO2': 0.0001,
                                 'neutral_pressure_bar': 7.886394285176307e-16,
                                 'element_total_pressure_bar': 7.911378852563667e-16,
                                 'joint_ion_pressure_bar': 3.788665627024206e-17,
                                 'electron_pressure_bar': 2.0385770566688572e-16,
                                 'K_ion': 9.7934322623952e-18,
                                 'parent_oxide': 'K2O',
                                 'source_tables': {   'cation': 'K-006',
                                                      'neutral': 'K-005',
                                                      'electron': 'D-020'},
                                 'upstream_sha256': {   'K-006': '27cd2083096e8faf6281a319dfb74e27c90ac472a965fb9366cdc43242e963b4',
                                                        'D-020': 'c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd'},
                                 'user_agent': 'openimcc-janaf-vendor/1.0'}},
    'Cr': {   'status': 'gas-complete-melt-pending',
              'criteria': {'C1': False, 'C2': True, 'C3': True, 'C4': True},
              'validation': 'unvalidated',
              'reason': 'Cr(g) and its neutral oxide channels cover 1200-3000 K, and a labelled '
                        'constant-Cp Cr2O3(l) continuation covers the parent interval; Cr2O3 '
                        'activity is caller-supplied, so melt closure remains pending; joint C3 '
                        'closure-screen maximum 1.099e-06 at 2800 K and fO2=1e-4 within the '
                        '<=1-bar neutral-pressure domain; unmodeled positive molecular ions and '
                        'thermal electrons from walls or other sources remain outside the estimate.',
              'c2_candidates': (('Cr-010', 'CrO'), ('Cr-011', 'CrO2'), ('Cr-012', 'CrO3')),
              'c3_ion_bound': {   'max_ratio': 1.1031336214989125e-06,
                                  'isolated_bound': 0.06758135170885075,
                                  'temperature_K': 2800.0,
                                  'fO2': 0.0001,
                                  'neutral_pressure_bar': 0.0023340760987797158,
                                  'element_total_pressure_bar': 0.0038602274477255102,
                                  'joint_ion_pressure_bar': 4.258346684218946e-09,
                                  'electron_pressure_bar': 8.131780113388637e-05,
                                  'K_ion': 1.4835822576971673e-10,
                                  'parent_oxide': 'Cr2O3',
                                  'source_tables': {   'cation': 'Cr-006',
                                                       'neutral': 'Cr-005',
                                                       'electron': 'D-020'},
                                  'upstream_sha256': {   'Cr-006': '3767ef70fb010220ba12abb9131d157617d5466b8aac2c384734f2acbada8fa3',
                                                         'D-020': 'c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd'},
                                  'user_agent': 'openimcc-janaf-vendor/1.0'},
              'c2_screen_maxima': {   'Cr': {   'max_ratio': 1.0,
                                                'temperature_K': 1400.0,
                                                'fO2': 1e-12,
                                                'dominant': 'Cr'},
                                      'CrO': {   'max_ratio': 1.0,
                                                 'temperature_K': 2100.0,
                                                 'fO2': 1e-06,
                                                 'dominant': 'CrO'},
                                      'CrO2': {   'max_ratio': 1.0,
                                                  'temperature_K': 1200.0,
                                                  'fO2': 1e-12,
                                                  'dominant': 'CrO2'},
                                      'CrO3': {   'max_ratio': 1.0,
                                                  'temperature_K': 1200.0,
                                                  'fO2': 1e-08,
                                                  'dominant': 'CrO3'}}},
    'V': {   'status': 'gas-complete-melt-pending',
             'criteria': {'C1': False, 'C2': True, 'C3': True, 'C4': True},
             'validation': 'unvalidated',
             'reason': 'A labelled constant-Cp V2O3(l) continuation covers 1200-1700 K; V(g), '
                       'VO(g), and VO2(g) now cover the lower gas interval; activity '
                       'is caller-supplied; joint C3 closure-screen maximum 8.022e-09 at 2300 K '
                       'and fO2=1e-10 within the <=1-bar neutral-pressure domain; unmodeled '
                       'positive molecular ions and thermal electrons from walls or other sources '
                       'remain outside the estimate.',
             'c2_candidates': (('O-026', 'VO'), ('O-076', 'VO2')),
              'c3_ion_bound': {   'max_ratio': 8.07650296357727e-09,
                                 'isolated_bound': 244.34360766894284,
                                 'temperature_K': 2300.0,
                                 'fO2': 1e-10,
                                 'neutral_pressure_bar': 3.435228834140807e-06,
                                 'element_total_pressure_bar': 2.271257370331679e-05,
                                  'joint_ion_pressure_bar': 1.8343816882530522e-13,
                                  'electron_pressure_bar': 4.935841167923758e-06,
                                 'K_ion': 2.635695347157119e-13,
                                 'parent_oxide': 'V2O3',
                                 'source_tables': {   'cation': 'V-006',
                                                      'neutral': 'V-005',
                                                      'electron': 'D-020'},
                                 'upstream_sha256': {   'V-006': 'c37e63246365b3335b5c2efdabf501741249c50357b6514a1a94e96f14ca2e1c',
                                                        'D-020': 'c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd'},
                                 'user_agent': 'openimcc-janaf-vendor/1.0'},
             'c2_screen_maxima': {   'V': {   'max_ratio': 1.0,
                                              'temperature_K': 2200.0,
                                              'fO2': 1e-12,
                                              'dominant': 'V'},
                                     'VO': {   'max_ratio': 1.0,
                                               'temperature_K': 2000.0,
                                               'fO2': 1e-12,
                                               'dominant': 'VO'},
                                     'VO2': {   'max_ratio': 1.0,
                                                'temperature_K': 1200.0,
                                                'fO2': 1e-12,
                                                'dominant': 'VO2'}}},
    'Nb': {   'status': 'gas-complete-melt-pending',
              'criteria': {'C1': False, 'C2': True, 'C3': True, 'C4': True},
              'validation': 'unvalidated',
              'reason': 'Nb-013 is liquid from 1000 K and its condensate intervals cover 1200-3000 '
                        'K; Nb(g), NbO(g), and NbO2(g) now cover the lower gas interval; '
                        'activity is caller-supplied; joint C3 closure-screen maximum 5.436e-13 at '
                        '2300 K and fO2=1e-10 within the <=1-bar neutral-pressure domain; unmodeled '
                        'positive molecular ions and thermal electrons from walls or other sources '
                        'remain outside the estimate.',
              'c2_candidates': (('Nb-011', 'NbO'), ('Nb-015', 'NbO2')),
              'c3_ion_bound': {   'max_ratio': 5.472683479442313e-13,
                                  'isolated_bound': 3202840673.576661,
                                  'temperature_K': 2300.0,
                                  'fO2': 1e-10,
                                  'neutral_pressure_bar': 7.762835489882393e-12,
                                  'element_total_pressure_bar': 3.0118668665341213e-07,
                                  'joint_ion_pressure_bar': 1.6482994042760972e-19,
                                  'electron_pressure_bar': 4.935841167923758e-06,
                                  'K_ion': 1.0480376748024353e-13,
                                  'parent_oxide': 'NbO2',
                                  'source_tables': {   'cation': 'Nb-006',
                                                       'neutral': 'Nb-005',
                                                       'electron': 'D-020'},
                                  'upstream_sha256': {   'Nb-006': 'afc5e0d57e2be188eaae3d111ac71adb7df586121c51a0a1c1d67206d9473380',
                                                         'D-020': 'c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd'},
                                  'user_agent': 'openimcc-janaf-vendor/1.0'},
              'c2_screen_maxima': {   'Nb': {   'max_ratio': 0.9601449643935895,
                                                'temperature_K': 3000.0,
                                                'fO2': 1e-12,
                                                'dominant': 'NbO'},
                                      'NbO': {   'max_ratio': 1.0,
                                                 'temperature_K': 2200.0,
                                                 'fO2': 1e-12,
                                                 'dominant': 'NbO'},
                                      'NbO2': {   'max_ratio': 1.0,
                                                  'temperature_K': 1200.0,
                                                  'fO2': 1e-12,
                                                  'dominant': 'NbO2'}}},
    'Li': {   'status': 'gas-complete-melt-pending',
              'criteria': {'C1': False, 'C2': True, 'C3': True, 'C4': True},
              'validation': 'unvalidated',
              'reason': 'Li2O(l) and neutral Li gas rows cover 1200-3000 K using the printed JANAF '
                        'liquid-branch cells; a(Li2O) is caller-supplied; the joint C3 screen used '
                        'a(Li2O)=1e-3 and reached 4.335e-08 at 2000 K and fO2=1e-4 within the '
                        '<=1-bar neutral-pressure domain; unmodeled positive molecular ions and '
                        'thermal electrons from walls or other sources remain outside the '
                        'estimate.',
              'c2_candidates': (('Li-011', 'LiO'), ('Li-017', 'Li2O'), ('Li-019', 'Li2O2')),
              'c3_ion_bound': {   'max_ratio': 4.335233951479634e-08,
                                  'isolated_bound': 1.3108025093318018e-06,
                                  'temperature_K': 2000.0,
                                  'fO2': 0.0001,
                                  'neutral_pressure_bar': 8.799540794291536e-05,
                                  'element_total_pressure_bar': 9.797437981283206e-05,
                                  'joint_ion_pressure_bar': 4.247418577397504e-12,
                                  'electron_pressure_bar': 3.200752704465291e-05,
                                  'K_ion': 9.070786665087108e-17,
                                  'parent_oxide': 'Li2O',
                                  'parent_activity': 0.001,
                                  'source_tables': {   'cation': 'Li-006',
                                                       'neutral': 'Li-005',
                                                       'electron': 'D-020'},
                                  'upstream_sha256': {   'Li-006': 'aa03b9472bee061c1d19234bd345ccecd8d9c46e6266f036ed3791b6984f955c',
                                                         'D-020': 'c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd'},
                                  'user_agent': 'neutral-source-vendor/1.0'}},
    'Rb': {   'status': 'gas-complete-melt-pending',
              'criteria': {'C1': False, 'C2': True, 'C3': True, 'C4': True},
              'validation': 'unvalidated',
              'reason': 'Rb2O(l) and neutral Rb gas rows cover 1200-3000 K; a(Rb2O) is '
                        'caller-supplied; the joint C3 screen used a(Rb2O)=1e-3 and reached '
                        '5.472e-05 at 2000 K and fO2=1e-4 within the <=1-bar neutral-pressure '
                        'domain; unmodeled positive molecular ions and thermal electrons from '
                        'walls or other sources remain outside the estimate.',
              'c2_candidates': (('NG-1329', 'RbO'), ('NG-1352', 'Rb2O')),
              'c3_ion_bound': {   'max_ratio': 5.4723553937520966e-05,
                                  'isolated_bound': 5.984478396735006e-09,
                                  'temperature_K': 2000.0,
                                  'fO2': 0.0001,
                                  'neutral_pressure_bar': 0.6038309372784079,
                                  'element_total_pressure_bar': 0.6122875501491727,
                                  'joint_ion_pressure_bar': 3.3506550775860826e-05,
                                  'electron_pressure_bar': 3.200752704465291e-05,
                                  'K_ion': 2.1625590126102705e-17,
                                  'parent_oxide': 'Rb2O',
                                  'parent_activity': 0.001,
                                  'source_tables': {   'cation': 'Rb-006',
                                                       'neutral': 'Rb-005',
                                                       'electron': 'D-020'},
                                  'upstream_sha256': {   'Rb-006': '8a106cf972c8c553591083d28d9371b71e6d12451d69afcc6e7a2e565b31cf0f',
                                                         'D-020': 'c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd'},
                                  'user_agent': 'neutral-source-vendor/1.0'}},
    'Pb': {   'status': 'gas-complete-melt-pending',
              'criteria': {'C1': False, 'C2': True, 'C3': True, 'C4': True},
              'validation': 'unvalidated',
              'reason': 'PbO(l) and neutral Pb gas rows cover 1200-3000 K; a(PbO) is '
                        'caller-supplied; PbO(l) is the Pb(II) parent because no evaluated PbO2(l) '
                        'parent row was found; the joint C3 screen used a(PbO)=1e-3 and reached '
                        '1.301e-12 at 2000 K and fO2=1e-4 within the <=1-bar neutral-pressure '
                        'domain; unmodeled positive molecular ions and thermal electrons from '
                        'walls or other sources remain outside the estimate.',
              'c2_candidates': (('O-009', 'PbO'), ('NG-1276', 'PbO2')),
              'c3_ion_bound': {   'max_ratio': 1.3005719979281524e-12,
                                  'isolated_bound': 2.1364268975758982e-10,
                                  'temperature_K': 2000.0,
                                  'fO2': 0.0001,
                                  'neutral_pressure_bar': 0.005873842806589901,
                                  'element_total_pressure_bar': 0.006796768656190416,
                                  'joint_ion_pressure_bar': 8.839686990637013e-15,
                                  'electron_pressure_bar': 3.200752704465291e-05,
                                  'K_ion': 2.6810097545132173e-22,
                                  'parent_oxide': 'PbO',
                                  'parent_activity': 0.001,
                                  'source_tables': {   'cation': 'Pb-006',
                                                       'neutral': 'Pb-005',
                                                       'electron': 'D-020'},
                                  'upstream_sha256': {   'Pb-006': '74f62dbb997ac98d8822dab2082d7076f483e0d0b411d221a431907e518043f9',
                                                         'D-020': 'c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd'},
                                  'user_agent': 'neutral-source-vendor/1.0'}},
    'Ga': {   'status': 'gas-complete-melt-pending',
              'criteria': {'C1': False, 'C2': True, 'C3': True, 'C4': True},
              'validation': 'unvalidated',
              'reason': 'Ga2O3(l) uses its NASA liquid card from 2080 K with a labelled '
                        'constant-Cp continuation to 1200 K; GaO(g) and Ga2O(g) are included; '
                        'a(Ga2O3) is caller-supplied; the joint C3 screen used a(Ga2O3)=1e-3 and '
                        'reached 5.443e-07 at 2600 K and fO2=1e-4 within the <=1-bar '
                        'neutral-pressure domain.',
              'c2_candidates': (('NG-5066', 'GaO'), ('NG-5178', 'Ga2O')),
              'c3_ion_bound': {   'max_ratio': 5.46482569998207e-07,
                                  'isolated_bound': 0.0019556139830335627,
                                  'temperature_K': 2600.0,
                                  'fO2': 0.0001,
                                  'neutral_pressure_bar': 0.06901684913530377,
                                  'element_total_pressure_bar': 0.9111811172801805,
                                  'joint_ion_pressure_bar': 4.979445987051107e-07,
                                  'electron_pressure_bar': 1.7729618917254428e-05,
                                  'K_ion': 1.191739772748273e-10,
                                  'parent_oxide': 'Ga2O3',
                                  'parent_activity': 0.001,
                                  'source_tables': {   'cation': 'Ga-006',
                                                       'neutral': 'Ga-005',
                                                       'electron': 'D-020',
                                                       'anions': ['Ga-007']},
                                  'upstream_sha256': {   'Ga-006': 'c340f61a42cc79773a6dede7a6acd92aa91c2ce2383f97716ab24c69ffcac9b6',
                                                         'Ga-005': '460c1c6f62d2f90ebd727c631597c891f7ffe4aa442998947ea4c45eaf3c9b51',
                                                         'D-020': 'c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd',
                                                         'Ga-007': 'bf28d3d3e0813faba6c4bd4e75d2dad96ca60d50d890c31f237170cf1273facd'},
                                  'user_agent': 'neutral-source-vendor/1.0'}},
    'Ge': {   'status': 'gas-complete-melt-pending',
              'criteria': {'C1': False, 'C2': True, 'C3': True, 'C4': True},
              'validation': 'unvalidated',
              'reason': 'GeO2(l) uses its NASA liquid card from 1388 K with a labelled constant-Cp '
                        'continuation to 1200 K; GeO(g) and GeO2(g) are included; a(GeO2) is '
                        'caller-supplied; the joint C3 screen used a(GeO2)=1e-3 and reached '
                        '3.495e-14 at 2300 K and fO2=1e-4 within the <=1-bar neutral-pressure '
                        'domain.',
              'c2_candidates': (('NG-5331', 'GeO'), ('NG-5339', 'GeO2')),
              'c3_ion_bound': {   'max_ratio': 3.5185744725742474e-14,
                                  'isolated_bound': 3.1175640352052395e-05,
                                  'temperature_K': 2300.0,
                                  'fO2': 0.0001,
                                  'neutral_pressure_bar': 6.299975812190224e-05,
                                  'element_total_pressure_bar': 0.9117974757246918,
                                  'joint_ion_pressure_bar': 3.208227322242538e-14,
                                  'electron_pressure_bar': 1.0331357586337778e-06,
                                  'K_ion': 5.26118586363642e-16,
                                  'parent_oxide': 'GeO2',
                                  'parent_activity': 0.001,
                                  'source_tables': {   'cation': 'NG-5197',
                                                       'neutral': 'NG-5186',
                                                       'electron': 'D-020'},
                                  'upstream_sha256': {   'NG-5197': 'a93d69f7eb19303878437467b22d46a986ac80723b79d1bd041aa9e312b1f24f',
                                                         'NG-5186': '8999ac1a39bbf761a895d5b383dbb3d6145af8f07ca06d5fe574e3d78a2130eb',
                                                         'D-020': 'c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd'},
                                  'user_agent': 'neutral-source-vendor/1.0'}},
    'B': {   'status': 'gas-partial',
             'criteria': {'C1': False, 'C2': True, 'C3': False, 'C4': True},
             'validation': 'unvalidated',
             'reason': 'JANAF B2O3(l) is fitted from its printed liquid branch after the 723 K '
                       'marker; BO(g), BO2(g), B2O(g), B2O2(g), and B2O3(g) are included; B2(g) '
                       'from JANAF B-083 has a conservative screened share of 3.836e-06; a(B2O3) '
                       'is caller-supplied, so C1 remains pending; B+, B-, BO-, and BO2- are '
                       'included on the opt-in ion path; their joint screen reaches 1.247e-02 '
                       'at 2800 K and fO2=1e-4, above the 1e-4 C3 threshold.',
             'c2_candidates': (   ('B-078', 'BO'),
                                  ('B-079', 'BO2'),
                                  ('B-093', 'B2O'),
                                  ('B-094', 'B2O2'),
                                  ('B-098', 'B2O3')),
             'c3_ion_bound': {   'max_ratio': 0.01246506196756266,
                                 'isolated_bound': 12.62461597168318,
                                 'temperature_K': 2800.0,
                                 'fO2': 0.0001,
                                 'neutral_pressure_bar': 4.93089563126449e-08,
                                 'element_total_pressure_bar': 0.05744488671684352,
                                 'joint_ion_pressure_bar': 0.0007160540726450716,
                                 'electron_pressure_bar': 9.936031679334023e-06,
                                 'K_ion': 5.372149210451025e-14,
                                 'parent_oxide': 'B2O3',
                                 'parent_activity': 0.001,
                                 'source_tables': {   'cation': 'B-006',
                                                      'neutral': 'B-005',
                                                      'electron': 'D-020',
                                                      'anions': ['B-007', 'NG-1074', 'B-080']},
                                 'upstream_sha256': {   'B-006': '9bcc9635f7886aab1b35db7a3e2cebcf626c498107f1939409937093f28ed031',
                                                        'B-005': 'a8bfb752b6ffc96b8a139cfd63c10b96d12afaaa5f8b14d03fca57c25410e154',
                                                        'D-020': 'c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd',
                                                        'B-007': 'aaaf2d1ba2c953bd90ce7ae5547403a1be873c53d69d4f12cd9afbee356a808c',
                                                        'NG-1074': '373f73ffc5c9a5b5778e608d3037f71a8866f02fe7b40789afa24a56cebe6b94',
                                                        'B-080': 'fa44f68b4415ca45cb3a8cfd62febcb04ff0c2ca8bd92ddffc4b7b1d90d05947'},
                                 'user_agent': 'neutral-source-vendor/1.0'}},
    'In': {   'status': 'gas-partial',
              'criteria': {'C1': False, 'C2': True, 'C3': False, 'C4': True},
              'validation': 'unvalidated',
              'reason': 'In2O3(l) uses its NASA liquid card from 2186 K with a labelled '
                        'constant-Cp continuation to 1200 K; InO(g) and In2O(g) are included; '
                        'a(In2O3) is caller-supplied. In+ is declined because its NASA card gives '
                        'Kion/Saha ratios of 1.510e28 at 1500 K and 9.143e16 at 2500 K against '
                        "NIST's 5.78636 eV ionisation energy.",
              'c2_candidates': (('NG-6131', 'InO'), ('NG-6243', 'In2O')),
              'c3_ion_bound': {   'status': 'declined',
                                  'reason': "The NASA In+ card's dfH298 is 6996.425 J/mol versus "
                                            '240700 J/mol for neutral In. Its fitted Kion is '
                                            '5.293e9 at 1500 K and 6.870e6 at 2500 K, while the '
                                            'NIST ground-term Saha values are 3.505e-19 and '
                                            '7.514e-11, respectively; the source card is retained '
                                            'without adjustment and In+ is not emitted.',
                                  'parent_oxide': 'In2O3',
                                  'parent_activity': 0.001,
                                  'ionisation_energy_eV': 5.78636,
                                  'saha_checks': (   {   'temperature_K': 1500.0,
                                                         'fitted_K_ion': 5293385888.0,
                                                         'saha_K_ion': 3.505390251e-19,
                                                         'ratio': 1.510070351e+28},
                                                     {   'temperature_K': 2500.0,
                                                         'fitted_K_ion': 6870174.263,
                                                         'saha_K_ion': 7.514218545e-11,
                                                         'ratio': 9.142899187e+16}),
                                  'source_tables': {   'cation': 'NG-6016',
                                                       'neutral': 'NG-6005',
                                                       'electron': 'D-020'},
                                  'upstream_sha256': {   'NG-6016': '2ca1ace033413d51c8f2bcb88d3c05c37bf39d107b46442b671e5896ca237784',
                                                         'NG-6005': 'a826485a5b317bc65da051086db82c8b15798ceb1ce68bf65319b2abdc08312a',
                                                         'D-020': 'c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd'},
                                  'user_agent': 'neutral-source-vendor/1.0'}},
    'Zn': {   'status': 'gas-complete-melt-pending',
              'criteria': {'C1': False, 'C2': True, 'C3': True, 'C4': True},
              'validation': 'unvalidated',
              'reason': 'Zn(g), ZnO(g), ZnO(l), and opt-in Zn ions cover 1200-3000 K; ZnO(l) is a flagged tier-4 construction from TSIV Table 1603; a(ZnO) is caller-supplied; the joint C3 screen used a(ZnO)=1e-3 and reached 1.477e-07 at 2600 K and fO2=1e-4 within the <=1-bar neutral-pressure domain.',
              'c2_candidates': (('IVTAN-264', 'ZnO'),),
              'c3_ion_bound': {   'max_ratio': 1.476911645013106e-07,
                                  'isolated_bound': 2.338972206610675e-08,
                                  'temperature_K': 2600.0,
                                  'fO2': 0.0001,
                                  'neutral_pressure_bar': 0.6745524435643065,
                                  'element_total_pressure_bar': 0.6746587563673206,
                                  'joint_ion_pressure_bar': 9.964113736889559e-08,
                                  'electron_pressure_bar': 1.7475121363727175e-05,
                                  'K_ion': 3.690335426012708e-16,
                                  'parent_oxide': 'ZnO',
                                  'parent_activity': 0.001,
                                  'source_tables': {   'cation': 'Zn-006',
                                                       'anion': 'Zn-007',
                                                       'neutral': 'Zn-005',
                                                       'electron': 'D-020'},
                                  'upstream_sha256': {   'Zn-006': '83272492a0be9e530ee03bfb5d0f88dcf59436b8d4862711cd29039cc1a2c9d6',
                                                         'Zn-007': '01eb728c6cc7adefbca5e04eca489ab2901be44a20b86303951497e4799c8ddf',
                                                         'Zn-005': 'b67dd110d66081622c01e120d71e6d2f181f50c22b157d8f3a6aa6f9732cd060',
                                                         'D-020': 'c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd'},
                                  'user_agent': 'neutral-source-vendor/1.0'}},
    'Cd': {   'status': 'gas-complete-melt-pending',
              'criteria': {'C1': False, 'C2': True, 'C3': True, 'C4': True},
              'validation': 'unvalidated',
              'reason': 'Cd(g), estimated CdO(g), CdO(l), and opt-in Cd+ cover 1200-3000 K; CdO(l) is a flagged tier-4 construction from Pankratz Bulletin 672 Table 102; a(CdO) is caller-supplied; the joint C3 screen used a(CdO)=1e-3 and reached 2.015e-13 at 2000 K and fO2=1e-4 within the <=1-bar neutral-pressure domain.',
              'c2_candidates': (('P696-T191', 'CdO'),),
              'c3_ion_bound': {   'max_ratio': 2.015279219813648e-13,
                                  'isolated_bound': 8.845662795030468e-11,
                                  'temperature_K': 2000.0,
                                  'fO2': 0.0001,
                                  'neutral_pressure_bar': 0.6622948642070532,
                                  'element_total_pressure_bar': 0.6623235403766004,
                                  'joint_ion_pressure_bar': 1.3347668677143683e-13,
                                  'electron_pressure_bar': 2.5713317716962474e-08,
                                  'K_ion': 5.1821758558714285e-21,
                                  'parent_oxide': 'CdO',
                                  'parent_activity': 0.001,
                                  'source_tables': {   'cation': 'NG-4071',
                                                       'neutral': 'NG-4060',
                                                       'electron': 'D-020'},
                                  'upstream_sha256': {   'NG-4071': 'a08a8a7386eb929042f3ce341771856ed22394892471756f0d146cb6c17837ec',
                                                         'NG-4060': '957b120d215dfa3fd2be09efb1d8940cc6a230c71738b06f2b45b9502c8f9535',
                                                         'D-020': 'c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd'},
                                  'user_agent': 'neutral-source-vendor/1.0'}},
    'Bi': {   'status': 'gas-complete-melt-pending',
              'criteria': {'C1': False, 'C2': True, 'C3': True, 'C4': True},
              'validation': 'unvalidated',
              'reason': 'Bi(g), Bi2(g), BiO(g), Bi2O3(l), and opt-in Bi+ cover 1200-3000 K; the measured Bi2O3(l) branch is selected as the Bi(III) parent; a(Bi2O3) is caller-supplied; the joint C3 screen used a(Bi2O3)=1e-3 and reached 4.195e-11 at 1800 K and fO2=1e-4 within the <=1-bar neutral-pressure domain.',
              'c2_candidates': (('P696-T115', 'BiO'),),
              'c3_ion_bound': {   'max_ratio': 4.1945095902479937e-11,
                                  'isolated_bound': 5.538611779245599e-10,
                                  'temperature_K': 1800.0,
                                  'fO2': 0.0001,
                                  'neutral_pressure_bar': 0.29343632468173547,
                                  'element_total_pressure_bar': 0.5603688775751227,
                                  'joint_ion_pressure_bar': 2.350472631065356e-11,
                                  'electron_pressure_bar': 1.1237621498095004e-09,
                                  'K_ion': 9.001517381391894e-20,
                                  'parent_oxide': 'Bi2O3',
                                  'parent_activity': 0.001,
                                  'source_tables': {   'cation': 'NBS-T22-ASD-BiII',
                                                       'neutral': 'P672-T80',
                                                       'electron': 'D-020'},
                                  'upstream_sha256': {   'NBS-T22-ASD-BiII': '1a6b0de572c176af13c402d531277781122dc837006092960bb6925eb3f56792',
                                                         'P672-T80': '368475919ca53f62ad8e4579ddaf7c6832cc38a1ca75e46f891c7b569d181ded',
                                                         'D-020': 'c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd'},
                                  'user_agent': 'neutral-source-vendor/1.0'}},
    'Mn': {   'status': 'gas-complete-melt-pending',
              'criteria': {'C1': False, 'C2': True, 'C3': True, 'C4': True},
              'validation': 'unvalidated',
              'reason': 'Mn(g), MnO(g), MnO(l), and opt-in Mn+ are available; MnO(l) is a tier-4 '
                        'liquid_from_solid construction fitted over 1200-3000 K and supercooled '
                        'below measured Tm 2120 ± 20 K; a(MnO) is caller-supplied; the joint C3 '
                        'screen used a(MnO)=1e-3 and reached 1.621e-07 at 2800 K and fO2=1e-4, '
                        'within the fitted interval and the <=1-bar neutral-pressure domain.',
              'c2_candidates': (('TSIV-1436', 'MnO'),),
              'c3_ion_bound': {   'max_ratio': 1.6207006631146375e-07,
                                  'isolated_bound': 0.0001771497679124323,
                                  'temperature_K': 2800.0,
                                  'fO2': 0.0001,
                                  'neutral_pressure_bar': 0.0004260542904211233,
                                  'element_total_pressure_bar': 0.0004322375111120103,
                                  'joint_ion_pressure_bar': 7.005276208822556e-11,
                                  'electron_pressure_bar': 8.131783320211644e-05,
                                  'K_ion': 1.337045289981064e-11,
                                  'parent_oxide': 'MnO',
                                  'parent_activity': 0.001,
                                  'source_tables': {   'cation': 'Mn-006',
                                                       'neutral': 'Mn-005',
                                                       'electron': 'D-020'},
                                  'upstream_sha256': {   'Mn-006': 'e90269050c0e94dae563a80fcb120b812af5382fada2c54aa6b951264fa56542',
                                                         'Mn-005': '6e2b4c4f4fee21fa925c34e7aac461d5a8d709f06159bd70c626c621596faa19',
                                                         'D-020': 'c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd'},
                                  'user_agent': 'openimcc-janaf-vendor/1.0'}},
    'Cs': {   'status': 'gas-partial',
              'criteria': {'C1': False, 'C2': True, 'C3': False, 'C4': True},
              'validation': 'unvalidated',
              'reason': 'Cs2O(l) and neutral Cs gas rows cover 1200-3000 K; Cs2(g) from JANAF '
                        'Cs-018 has a conservative screened share of 4.866e-23; a(Cs2O) is '
                        'caller-supplied, so C1 remains pending; Cs+, Cs2O+, and Cs- are included '
                        'on the opt-in ion path; their joint screen reaches 1.389e-04 at 2000 K '
                        'and fO2=1e-4 within the <=1-bar neutral-pressure domain, above the 1e-4 '
                        'limit.',
              'c2_candidates': (('Cs-017', 'CsO'), ('Cs-021', 'Cs2O')),
              'c3_ion_bound': {   'max_ratio': 0.00013890184686647745,
                                  'isolated_bound': 8.94135201604957e-09,
                                  'temperature_K': 2000.0,
                                  'fO2': 0.0001,
                                  'neutral_pressure_bar': 0.5532219297945025,
                                  'element_total_pressure_bar': 0.5552861930691437,
                                  'joint_ion_pressure_bar': 7.713027775675943e-05,
                                  'electron_pressure_bar': 7.057232466395302e-05,
                                  'K_ion': 4.422886285229813e-17,
                                  'parent_oxide': 'Cs2O',
                                  'parent_activity': 0.001,
                                  'source_tables': {   'cation': 'Cs-006',
                                                       'neutral': 'Cs-005',
                                                       'molecular_cation': 'NG-1843',
                                                       'anions': ['Cs-007'],
                                                       'electron': 'D-020'},
                                  'upstream_sha256': {   'Cs-006': 'b2705d424836ca9fd585303e923b54ad1fca54d81f4db8c4e37c7a198595da3c',
                                                         'Cs-005': '5872ade6306f453d12dc6380c58a693117f5020ba1bcaad6f3c016a5211412a5',
                                                         'NG-1843': 'b6bc19299621afada6708b0c109c4215c91ab7521b8ec5c6b48250e24e133948',
                                                         'Cs-007': '6c574457a4addcc5518d4c42cb0cdfdac61da9c998afd7b52c4b934f8c7afd4a',
                                                         'D-020': 'c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd'},
                                  'user_agent': 'neutral-source-vendor/1.0'}},
    'Cu': {   'status': 'gas-complete-melt-pending',
              'criteria': {'C1': False, 'C2': True, 'C3': True, 'C4': True},
              'validation': 'unvalidated',
              'reason': 'Cu2O(l) is the Cu(I) parent because JANAF provides its liquid row; '
                        'neutral Cu gas rows and the parent cover 1200-3000 K; a(Cu2O) is '
                        'caller-supplied; Cu- is available on the opt-in ion path; its corrected '
                        'attachment-Saha ratios against NIST EA 1.23578 eV are 0.941424 at '
                        '1500 K and 0.958988 at 2500 K; Cu+ is declined because JANAF Cu-006 '
                        'gives fitted/Saha ratios 1666.15 and 1656.61.',
              'c2_candidates': (('Cu-016', 'CuO'),),
              'c3_ion_bound': {   'max_ratio': 2.3213908675545964e-05,
                                  'isolated_bound': 0.0,
                                  'temperature_K': 2800.0,
                                  'fO2': 0.0001,
                                  'neutral_pressure_bar': 0.23790232420841811,
                                  'element_total_pressure_bar': 0.24164133355057577,
                                  'joint_ion_pressure_bar': 5.609439849280207e-06,
                                  'electron_pressure_bar': 8.131780113388637e-05,
                                  'K_ion': 0.0,
                                  'parent_oxide': 'Cu2O',
                                  'parent_activity': 0.001,
                                  'source_tables': {   'cation': 'Cu-006',
                                                       'anion': 'Cu-007',
                                                       'neutral': 'Cu-005',
                                                       'electron': 'D-020'},
                                  'upstream_sha256': {   'Cu-006': '7699877b974780f89a88b8328d50996e4ab227c0f61c567daabe185fa8a8f9bf',
                                                         'Cu-007': '2e45a80821a53dcd1f581910e0d2e109c3fb7be0f6026498e45a1460755b2cd6',
                                                         'Cu-005': 'f7a575e44a41d314cc6532ca84b063e05201f7f7076e77d1e2dd80ad817e817d',
                                                         'D-020': 'c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd'},
                                  'user_agent': 'neutral-source-vendor/1.0'}},
    'Te': {   'status': 'gas-partial',
              'criteria': {'C1': False, 'C2': True, 'C3': False, 'C4': True},
              'validation': 'unvalidated',
              'reason': 'Te(g), Te2(g), TeO2(g), and the TeO2(l) parent cover 1200-3000 K; TeO2(l) is selected because Bulletin 672 prints a measured liquid branch; a(TeO2) is caller-supplied. This oxide-parent convention does not model telluride dissolution in reduced melts.',
              'c2_candidates': (('P672-TeO2-g', 'TeO2'),),
              'c3_ion_bound': {   'status': 'declined',
                                  'reason': 'No source for Te+(g) was located in the checked JANAF, NASA Glenn/CEA, NIST WebBook, or Gurvich/IVTAN routes.'}},
    'Se': {   'status': 'gas-partial',
              'criteria': {'C1': False, 'C2': True, 'C3': False, 'C4': True},
              'validation': 'unvalidated',
              'reason': 'Se(g), Se2(g), SeO(g), SeO2(g), and the SeO2(l) parent cover 1200-3000 K; SeO2(l) is the selected caller parent because no standard liquid function was found and the oxide state is the sourced parent convention; a(SeO2) is caller-supplied. This oxide-parent convention does not model selenide dissolution in reduced melts.',
              'c2_candidates': (('P672-SeO-g', 'SeO'), ('P672-SeO2-g', 'SeO2')),
              'c3_ion_bound': {   'status': 'declined',
                                  'reason': 'No source for Se+(g) was located in the checked JANAF, NASA Glenn/CEA, NIST WebBook, or Gurvich/IVTAN routes.'}},
    'As': {   'status': 'gas-partial',
              'criteria': {'C1': False, 'C2': True, 'C3': False, 'C4': True},
              'validation': 'unvalidated',
              'reason': 'As(g), As2(g), As4(g), As4O6(g), and a tier-4 As2O3(l) parent cover 1200-3000 K; As(g), As2(g), and As4(g) are constant-Cp continuation above 1200 K, and As4O6(g) is continuation above 1000 K. As2O3(l) is constructed from the Pankratz Table 53 crystal function and the repository tier-3 liquid_from_solid estimates because no standard liquid function was located; a(As2O3) is caller-supplied.',
              'c2_candidates': (('P672-As4O6-g', 'As4O6'),),
              'c3_ion_bound': {   'status': 'declined',
                                  'reason': 'No source for As+(g) was located in the checked JANAF, NASA Glenn/CEA, NIST WebBook, or Gurvich/IVTAN routes.'}},
    'Sn': {   'status': 'gas-complete-melt-pending',
              'criteria': {'C1': False, 'C2': True, 'C3': True, 'C4': True},
              'validation': 'unvalidated',
              'reason': 'SnO(l) is selected over the later-starting SnO2(l) card as the tin '
                        'parent; neutral Sn gas rows and the parent cover 1200-3000 K; a(SnO) is '
                        'caller-supplied; the C3 screen at a(SnO)=1e-3 reached 3.515e-08 at 2800 K '
                        'and fO2=1e-4 within the <=1-bar neutral-pressure domain.',
              'c2_candidates': (('NG-1847', 'SnO'), ('NG-1849', 'SnO2')),
              'c3_ion_bound': {   'max_ratio': 3.5287042788621135e-08,
                                  'isolated_bound': 1.9366616224859834e-06,
                                  'temperature_K': 2800.0,
                                  'fO2': 0.0001,
                                  'neutral_pressure_bar': 0.009527622799418886,
                                  'element_total_pressure_bar': 0.04282233357243156,
                                  'joint_ion_pressure_bar': 1.5110735170789996e-09,
                                  'electron_pressure_bar': 8.13184928646377e-05,
                                  'K_ion': 3.573485696034799e-14,
                                  'parent_oxide': 'SnO',
                                  'parent_activity': 0.001,
                                  'source_tables': {   'cation': 'NG-1846',
                                                       'neutral': 'NG-1845',
                                                       'electron': 'D-020'},
                                  'upstream_sha256': {   'NG-1846': '56d079d5c12c4c0f120a6eb2ca1637d6df683417a12676ecbfaf2b6792cca312',
                                                         'NG-1845': '676fcb71f4498c9c8c0c1e7d82c1022d1f25091ce078a2517825feb4026f11a2',
                                                         'D-020': 'c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd'},
                                  'user_agent': 'neutral-source-vendor/1.0'}}}


# Sb is gas-partial: the neutral channels and a tier-4 hypothetical parent
# cover the requested domain, while no sourced Sb+ card was located.
ELEMENT_STATUS["Sb"] = {
    "status": "gas-partial",
    "criteria": {"C1": False, "C2": True, "C3": False, "C4": True},
    "validation": "unvalidated",
    "reason": "Sb(g), Sb2(g), Sb4(g), Sb4O6(g), and a tier-4 Sb2O3(l) construction cover 1200-3000 K; the source Sb4O6(g) nodes end at 1000 K and the other source nodes end at 2000 K. Bulletin 672 reports a 928 K Sb2O3 melting marker but insufficient fusion-enthalpy data and no liquid function; a(Sb2O3) is caller-supplied.",
    "c2_candidates": (("P672-Sb4O6-g", "Sb4O6"),),
    "c3_ion_bound": {
        "status": "declined",
        "reason": "No JANAF, NASA Glenn/CEA, or Bulletin 672 source for Sb+(g) was located.",
    },
}

ELEMENT_STATUS["Ag"] = {
    "status": "gas-complete-melt-pending",
    "criteria": {"C1": False, "C2": True, "C3": True, "C4": True},
    "validation": "unvalidated",
    "reason": "Ag(g) and a tier-4 hypothetical Ag2O(l) caller-activity standard cover 1200-3000 K; NASA Glenn supplies opt-in Ag+(g) and Ag-(g). Ag2O decomposes near 500 K and has no measured liquid function; the Ag2O-parent convention does not represent reduced melts where silver dissolves largely as metal.",
    "c2_candidates": (),
    "c3_ion_bound": {
        "max_ratio": 8.677635624166437e-09,
        "isolated_bound": 3.462457407497183e-09,
        "temperature_K": 2200.0,
        "fO2": 1.0e-4,
        "neutral_pressure_bar": 0.6590716895454442,
        "element_total_pressure_bar": 0.6590716895454442,
        "joint_ion_pressure_bar": 5.719183972079108e-09,
        "electron_pressure_bar": 1.9336054153397106e-07,
        "K_ion": 7.901354303958975e-18,
        "parent_oxide": "Ag2O",
        "parent_activity": 0.001,
        "source_tables": {"cation": "NG-0085", "neutral": "NG-0074", "electron": "D-020"},
        "upstream_sha256": {
            "NG-0085": "d16a68a0d76ed157b085a32683f5b54f5e973e224e1618d6b29eb9e3186dde43",
            "NG-0074": "9c755e46e9429dbffb432ac877457f0b907a032916642e7ba45fb4822fc24aad",
            "D-020": "c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd",
        },
        "user_agent": "neutral-source-vendor/1.0",
    },
}

ELEMENT_STATUS["Mo"] = {
    "status": "gas-complete-melt-pending",
    "criteria": {"C1": False, "C2": True, "C3": True, "C4": True},
    "validation": "unvalidated",
    "reason": "MoO3(l), JANAF Mo monomers and ions, and NASA Glenn Mo oxide clusters cover 1200-3000 K; MoO3(l) uses its JANAF liquid branch through 2000 K and a labelled constant-Cp continuation above it; a(MoO3) is caller-supplied; the C3 screen at a(MoO3)=1e-3 reached 1.192e-09 at 2600 K and fO2=1e-6 within the <=1-bar neutral-pressure domain.",
    "c2_candidates": (
        ("Mo-008", "MoO"), ("Mo-010", "MoO2"), ("Mo-017", "MoO3"),
        ("NG-0831", "(MoO3)2"), ("NG-0832", "(MoO3)3"),
        ("NG-0833", "(MoO3)4"), ("NG-0834", "(MoO3)5"),
    ),
    "c3_ion_bound": {
        "status": "computed",
        "max_ratio": 1.1918231274644968e-09,
        "isolated_bound": 38906.85178134413,
        "temperature_K": 2600.0,
        "fO2": 1.0e-6,
        "neutral_pressure_bar": 9.751648553137538e-05,
        "element_total_pressure_bar": 0.009347350884885657,
        "joint_ion_pressure_bar": 1.1140388965132455e-11,
        "electron_pressure_bar": 2.983390605958947e-05,
        "K_ion": 3.4082432969578687e-12,
        "parent_oxide": "MoO3",
        "parent_activity": 1.0e-3,
        "source_tables": {"neutral": "Mo-005", "cation": "Mo-006", "electron": "D-020"},
        "upstream_sha256": {
            "Mo-005": "831d1538d9b43dc01390e595aac3e97854290b2b714ad27c048529c06aa5041b",
            "Mo-006": "0ceb49007573c24e17516379ec29a8e69af92004db4210ba1e12db7f1715621f",
            "D-020": "c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd",
        },
        "user_agent": "openimcc-janaf-vendor/1.0",
    },
}

ELEMENT_STATUS["W"] = {
    "status": "gas-complete-melt-pending",
    "criteria": {"C1": False, "C2": True, "C3": True, "C4": True},
    "validation": "unvalidated",
    "reason": "WO3(l), JANAF W monomers, oxide clusters and ions, and NASA Glenn (WO3)5(g) cover 1200-3000 K; O-066 prints the liquid values below its 1745 K marker as supercooled source data; a(WO3) is caller-supplied; the C3 screen at a(WO3)=1e-3 reached 6.594e-13 at 2300 K and fO2=1e-10 within the <=1-bar neutral-pressure domain.",
    "c2_candidates": (
        ("O-027", "WO"), ("O-048", "WO2"), ("O-068", "WO3"),
        ("O-088", "(WO3)2"), ("O-092", "W3O8"), ("O-093", "(WO3)3"),
        ("O-096", "(WO3)4"), ("NG-1258", "(WO3)5"),
    ),
    "c3_ion_bound": {
        "status": "computed",
        "max_ratio": 6.594318508976297e-13,
        "isolated_bound": 665186539.0955995,
        "temperature_K": 2300.0,
        "fO2": 1.0e-10,
        "neutral_pressure_bar": 1.3347845296417244e-07,
        "element_total_pressure_bar": 8.318312853938228e-05,
        "joint_ion_pressure_bar": 5.48536044161803e-17,
        "electron_pressure_bar": 4.935840956984356e-06,
        "K_ion": 2.0283218318286082e-15,
        "parent_oxide": "WO3",
        "parent_activity": 1.0e-3,
        "source_tables": {"neutral": "W-005", "cation": "W-006", "electron": "D-020"},
        "upstream_sha256": {
            "W-005": "fdf2523b89b49953a092eb84a71fd967244fe4a1124be77daa227c1f9fa8fa0e",
            "W-006": "f16518d3b08afc18b91a7da69c615429ebbeb690eabf451d4556929a1ced90b6",
            "D-020": "c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd",
        },
        "user_agent": "openimcc-janaf-vendor/1.0",
    },
}

for _element, _status in {
    "Ba": {
        "max_ratio": 0.00017523336722220458,
        "isolated_bound": 5.214371321920937,
        "temperature_K": 2300.0,
        "neutral_pressure_bar": 9.078475370636664e-06,
        "element_total_pressure_bar": 1.5180568934889218e-05,
        "joint_ion_pressure_bar": 2.6601422108094335e-09,
        "electron_pressure_bar": 4.936788834303332e-06,
        "K_ion": 1.4465601136572801e-09,
        "parent_oxide": "BaO",
        "cation": "Ba-006",
        "neutral": "Ba-005",
        "cation_sha": "f803f83975a93a065d02f8c5d60e602beb36dd375be8260f10cb8a0234e4d662",
        "neutral_sha": "a276bdbc4dbc6022da2aab69e8c82761369806d9baa1e082637f6dd842af7548",
        "c2": (("Ba-037", "BaO"),),
        "fO2": 1e-10,
    },
    "Sr": {
        "max_ratio": 0.0003099985241896078,
        "isolated_bound": 0.44153951926914414,
        "temperature_K": 2800.0,
        "neutral_pressure_bar": 7.7835861525105e-06,
        "element_total_pressure_bar": 9.56492050832456e-06,
        "joint_ion_pressure_bar": 2.965111241571527e-09,
        "electron_pressure_bar": 8.131915849141929e-05,
        "K_ion": 3.0978053852500063e-08,
        "parent_oxide": "SrO",
        "cation": "Sr-007",
        "neutral": "Sr-006",
        "cation_sha": "c409276dff239cf08f53e5c4e0eaabfbcbfbe74a28b4ee7ad60b077fca36fecf",
        "neutral_sha": "3129e5c71c8c76df78513225c90e1696c6db122f02c436047b82b03b24325f27",
        "c2": (("O-016", "SrO"),),
        "fO2": 1e-4,
    },
    "Zr": {
        "max_ratio": 1.187656337254675e-11,
        "isolated_bound": 28884129582391.32,
        "temperature_K": 2300.0,
        "neutral_pressure_bar": 3.104224957021332e-15,
        "element_total_pressure_bar": 9.12726922141502e-11,
        "joint_ion_pressure_bar": 1.0840059132643091e-21,
        "electron_pressure_bar": 4.935841167923753e-06,
        "K_ion": 2.4353276293503586e-13,
        "parent_oxide": "ZrO2",
        "cation": "Zr-007",
        "neutral": "Zr-006",
        "cation_sha": "8540f2d728b5d72715407f7ce84120eb9a6cb8c14addaebf267a768743e79f78",
        "neutral_sha": "29e24aea81a83cfeb75ebf91580fe0c14c50459a3b448270990c547805d9e32b",
        "c2": (("O-028", "ZrO"), ("O-055", "ZrO2")),
        "fO2": 1e-10,
        "anion": "Zr-008",
        "anion_sha": "da6964825dd3ca69ba3d6dfc1aa3a9c72ed794641640a72d058be090dff5d032",
    },
}.items():
    ELEMENT_STATUS[_element] = {
        "status": (
            "gas-complete-melt-pending"
            if _status["max_ratio"] < 1e-4
            else "gas-partial"
        ),
        "criteria": {"C1": False, "C2": True, "C3": _status["max_ratio"] < 1e-4, "C4": True},
        "validation": "unvalidated",
        "reason": (
            f'{_element}(g), its JANAF oxide gas channels, and the printed '
            f'{_status["parent_oxide"]}(l) liquid branch cover 1200-3000 K; '
            f'{_status["parent_oxide"]} is the selected parent because JANAF prints its liquid function; '
            f'caller supplies a({_status["parent_oxide"]}); the C3 screen at '
            f'a({_status["parent_oxide"]})=1e-3 reached {_status["max_ratio"]:.3e} at '
            f'{_status["temperature_K"]:.0f} K and fO2={_status["fO2"]:.0e} '
            'within the <=1-bar neutral-pressure domain'
            + (
                "; above the 1e-4 C3 threshold."
                if _status["max_ratio"] >= 1.0e-4
                else "; below the 1e-4 C3 threshold."
            )
            + (" ZrO(g) JANAF is retained despite its documented NASA G_app difference." if _element == "Zr" else "")
        ),
        "c2_candidates": _status["c2"],
        "c3_ion_bound": {
            "status": "computed",
            "max_ratio": _status["max_ratio"],
            "isolated_bound": _status["isolated_bound"],
            "temperature_K": _status["temperature_K"],
            "fO2": _status["fO2"],
            "neutral_pressure_bar": _status["neutral_pressure_bar"],
            "element_total_pressure_bar": _status["element_total_pressure_bar"],
            "joint_ion_pressure_bar": _status["joint_ion_pressure_bar"],
            "electron_pressure_bar": _status["electron_pressure_bar"],
            "K_ion": _status["K_ion"],
            "parent_oxide": _status["parent_oxide"],
            "parent_activity": 1e-3,
            "source_tables": {
                "neutral": _status["neutral"], "cation": _status["cation"],
                "electron": "D-020",
                **({"anion": _status["anion"]} if "anion" in _status else {}),
            },
            "upstream_sha256": {
                _status["neutral"]: _status["neutral_sha"],
                _status["cation"]: _status["cation_sha"],
                "D-020": "c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd",
                **({_status["anion"]: _status["anion_sha"]} if "anion" in _status else {}),
            },
            "user_agent": "neutral-source-vendor/1.0",
        },
    }
ELEMENT_STATUS["Ta"] = {
    "status": "gas-complete-melt-pending",
    "criteria": {"C1": False, "C2": True, "C3": False, "C4": True},
    "validation": "unvalidated",
    "reason": (
        "JANAF Ta-005, O-017, O-041, and O-078 provide Ta(g), TaO(g), TaO2(g), "
        "and the Ta2O5(l) parent on the same elemental reference states over "
        "1200-3000 K. Ta2O5 is the selected oxide parent and a(Ta2O5) is "
        "caller-supplied; fO2 is also an input. No ion channel or melt activity "
        "model is included."
    ),
    "c2_candidates": (("O-017", "TaO"), ("O-041", "TaO2")),
    "c3_ion_bound": {
        "status": "not computed",
        "reason": "not computed: ionized tantalum channels are outside this neutral gas change",
        "source_tables": {"selected": "JANAF"},
        "parent_activity": None,
    },
}
ELEMENT_STATUS["Be"] = {
    "status": "gas-complete-melt-pending",
    "criteria": {"C1": False, "C2": True, "C3": False, "C4": True},
    "validation": "unvalidated",
    "reason": (
        "NASA Glenn cards NG-0185, NG-0200, NG-0205, NG-0208, NG-0210, NG-0211, "
        "and NG-0212 provide Be(g), BeO(g), Be2(g), Be2O(g), (BeO)2(g), "
        "(BeO)3(g), and (BeO)4(g); NG-1352 provides BeO(l). The liquid card's "
        "2851 K source limit is extended downward by the labelled constant-Cp "
        "continuation. BeO is the selected oxide parent and a(BeO) is caller-supplied; "
        "fO2 is also an input. No ion channel or melt activity model is included."
    ),
    "c2_candidates": (
        ("NG-0200", "BeO"), ("NG-0208", "Be2O"),
        ("NG-0210", "(BeO)2"), ("NG-0211", "(BeO)3"),
        ("NG-0212", "(BeO)4"),
    ),
    "c3_ion_bound": {
        "status": "not computed",
        "reason": "not computed: ionized beryllium channels are outside this neutral gas change",
        "source_tables": {"selected": "NASA Glenn"},
        "parent_activity": None,
    },
}
ELEMENT_STATUS["Tl"] = {
    "status": "gas-complete-melt-pending",
    "criteria": {"C1": False, "C2": True, "C3": False, "C4": True},
    "validation": "unvalidated",
    "reason": (
        "Pankratz Bulletin 672 printed pp. 439-441 provide Tl(g), Tl2O(g), and "
        "the Tl2O(l) parent on elemental 298.15 K formation enthalpy anchors. "
        "The parent table prints its liquid branch from its 852 K melting point "
        "through 1000 K; its labelled constant-Cp continuation supplies the "
        "1200-3000 K runtime range. Tl2O is the selected parent and a(Tl2O) is "
        "caller-supplied; fO2 is also an input. No ion channel or melt activity "
        "model is included."
    ),
    "c2_candidates": (("P672-Tl2O-g", "Tl2O"),),
    "c3_ion_bound": {
        "status": "not computed",
        "reason": "not computed: ionized thallium channels are outside this neutral gas change",
        "source_tables": {"selected": "Pankratz Bulletin 672"},
        "parent_activity": None,
    },
}

for _element, _parent, _cation, _neutral, _anion, _hashes in (
    (
        "Ni", "NiO", "Ni-006", "Ni-005", "Ni-007",
        {
            "Ni-005": "1b86d14a543e9268d056860462f4604b8e987bfe66d78c17ea84ce7a85414cbf",
            "Ni-006": "1719b0562921ca8e79bc0250e34812eb331440e00864853629c025eedf92fd16",
            "Ni-007": "21d9c31d3edbc49a6440e18369022a0e30bdd83209c38056862d93ddc84dc82e",
        },
    ),
    (
        "Co", "CoO", "Co-006", "Co-005", "Co-007",
        {
            "Co-005": "10e4ec800f8f0015f6f7b5033449ffd21e60f9563b273738b128f0594afbfc2a",
            "Co-006": "28ba36fd244128e314eb794af7bf4aadcd3de6e87bfde3bc36ee43e6b033ee20",
            "Co-007": "9a13c2f52f861bb34cd4cd256b6954153f758eb2be5eb51cc12fcf55f0301b4a",
        },
    ),
):
    _tsiv_table = "2026" if _element == "Ni" else "1334"
    _melt_temperature = 2228 if _element == "Ni" else 2090
    _max_ratio = (
        1.6031810894215375e-05
        if _element == "Ni"
        else 3.0268368928067777e-06
    )
    ELEMENT_STATUS[_element] = {
        "status": "gas-complete-melt-pending",
        "criteria": {
            "C1": False,
            "C2": True,
            "C3": True,
            "C4": True,
        },
        "validation": "unvalidated",
        "reason": (
            f"{_element}(g), {_parent}(g), {_parent}(l), and opt-in "
            f"{_element}+/{_element}− cover "
            + (
                "1200-3000 K; NiO(g) uses the tier-2 NASA Glenn NG-0926 card "
                "selected over tier-3 Bulletin 672, with their measured G_app "
                "difference carried as a band; "
                if _element == "Ni"
                else "1200-3000 K; CoO(g) uses TSIV Volume IV Table 1335 "
                "(accuracy class 4-F); "
            )
            + f"{_parent}(l) uses the estimated tier-3 liquid branch "
            f"from TSIV Volume IV Table {_tsiv_table}, with a labelled "
            f"continuation below its {_melt_temperature} K source limit; a({_parent}) is "
            f"caller-supplied; the joint C3 screen used a({_parent})=1e-3 and "
            f"reached {_max_ratio:.3e} at 2800 K and fO2=1e-4 within the "
            "<=1-bar neutral-pressure domain, below the 1e-4 C3 threshold."
        ),
        "c2_candidates": (
            (("NG-0926", "NiO"),) if _element == "Ni"
            else (("TSIV-1335", "CoO"),)
        ),
        "c3_ion_bound": {
            "status": "computed",
            "max_ratio": 1.6031810894215375e-05 if _element == "Ni" else 3.0268368928067777e-06,
            "isolated_bound": 0.00016848973710661597 if _element == "Ni" else 3.3102018564057046e-05,
            "temperature_K": 2800.0,
            "fO2": 1.0e-4,
            "neutral_pressure_bar": 0.014971609461365306 if _element == "Ni" else 0.0021375904714789826,
            "element_total_pressure_bar": 0.01500165599221954 if _element == "Ni" else 0.0021594687581678014,
            "joint_ion_pressure_bar": 2.405037119673366e-07 if _element == "Ni" else 6.5363597060852885e-09,
            "electron_pressure_bar": 8.12793917760601e-05 if _element == "Ni" else 8.131486539410398e-05,
            "K_ion": 4.250258995716842e-10 if _element == "Ni" else 2.3422511491170064e-12,
            "parent_oxide": _parent,
            "parent_activity": 1.0e-3,
            "source_tables": {
                "neutral": _neutral,
                "cation": _cation,
                "anion": _anion,
                "electron": "D-020",
            },
            "upstream_sha256": {
                **_hashes,
                "D-020": "c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd",
            },
            "user_agent": "neutral-source-vendor/1.0",
        },
    }

ELEMENT_STATUS["Sc"] = {
    "status": "gas-complete-melt-pending",
    "criteria": {"C1": False, "C2": True, "C3": True, "C4": True},
    "validation": "unvalidated",
    "reason": "NASA Glenn Sc(g), ScO(g), ScO2(g), Sc+(g), and Sc2O3(l) cover 1200-3000 K; Sc2O3(l) uses its liquid card from 2762 K with a labelled constant-Cp continuation below; Sc2O3 is selected because it is the sourced liquid parent; a(Sc2O3) is caller-supplied; the joint C3 screen at a(Sc2O3)=1e-3 reached 1.929e-09 at 2300 K and fO2=1e-10 within the <=1-bar neutral-pressure domain, below the 1e-4 threshold.",
    "c2_candidates": (
        ("NG-1090", "ScO"), ("NG-1092", "ScO2"),
    ),
    "c3_ion_bound": {
        "status": "computed",
        "parent_oxide": "Sc2O3",
        "parent_activity": 1.0e-3,
        "source_tables": {"neutral": "NG-1087", "cation": "NG-1088", "electron": "D-020"},
        "max_ratio": 1.928791710917186e-09,
        "isolated_bound": 27406969.099476915,
        "temperature_K": 2300.0,
        "fO2": 1e-10,
        "neutral_pressure_bar": 2.1043533268879577e-10,
        "element_total_pressure_bar": 2.6727435360753636e-08,
        "joint_ion_pressure_bar": 5.15516557778965e-17,
        "electron_pressure_bar": 4.935841167942117e-06,
        "K_ion": 1.2091637920919846e-12,
        "electron_source_terms": {},
        "electron_source_terms_3000K": {},
        "upstream_sha256": {
            "NG-1087": "e2bf9fcbdcd866dfbc3305a6d05a756100f0b36d41b63559e4687fe542b6ffcf",
            "NG-1088": "9c6bde0ea24a3a0b5c33ecb64263c6172ac8df9cef175aa9106e7a04f0672181",
            "D-020": "c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd",
        },
        "user_agent": "neutral-source-vendor/1.0",
    },
}
ELEMENT_STATUS["U"] = {
    "status": "gas-partial",
    "criteria": {"C1": False, "C2": True, "C3": False, "C4": True},
    "validation": "unvalidated",
    "reason": "NASA Glenn U(g), UO(g), UO2(g), UO3(g), and UO2(l) cover 1200-3000 K; the UO2(l) row is a labelled supercooled continuation from its 3123 K liquid card throughout this runtime range; UO2 is selected because it is the sourced liquid parent; a(UO2) is caller-supplied; UO3 formation consumes 1/2 O2. No evaluated U+ thermo function was located in JANAF, NASA Glenn/CEA, Pankratz Bulletins 672/677/696, TSIV, NIST WebBook, OECD/NEA, or Burcat.",
    "c2_candidates": (
        ("NG-1224", "UO"), ("NG-1230", "UO2"), ("NG-1235", "UO3"),
    ),
    "c3_ion_bound": {
        "status": "declined",
        "reason": "No evaluated U+ thermo function was located in the checked JANAF, NASA Glenn/CEA, Pankratz Bulletins 672/677/696, TSIV, NIST WebBook, OECD/NEA, or Burcat sources.",
    },
}
ELEMENT_STATUS["Th"] = {
    "status": "gas-complete-melt-pending",
    "criteria": {"C1": False, "C2": True, "C3": True, "C4": True},
    "validation": "unvalidated",
    "reason": "Pankratz Bulletin 672 Th(g), ThO(g), ThO2(g), and ThO2(cr) cover source nodes through 2500 K; the ThO2(l) activity standard is a tier-4 liquid_from_solid construction from the sourced crystal using OECD/NEA fusion inputs, with crystal Cp continued above the Pankratz limit; ThO2 is selected as the sole sourced oxide parent; a(ThO2) is caller-supplied. Burcat's Gurvich-based Th+ card is adopted as an opt-in tier-3 ion; its card composition TH 1 E -1 and HF0/HF298 consistency resolve the conflicting header CAS label as singly ionized Th. The NIST ASD Saha audit is outside the 2% gate (ratios 4.496 at 1500 K and 2.558 at 2500 K); C3 remains available with a maximum joint ion fraction 1.653e-15 at 2300 K and fO2=1e-10.",
    "c2_candidates": (("P672-ThO-g", "ThO"), ("P672-ThO2-g", "ThO2")),
    "c3_ion_bound": {
        "status": "available",
        "reason": "Burcat's Gurvich-based Th+ card is an opt-in tier-3 ion. On the selected 597.2 kJ/mol Pankratz Th(g) neutral anchor, the selected Pankratz-neutral/Burcat-cation ionization energy is 12.486 kJ/mol below the NIST ASD value. That contributes +12.486 kJ/mol to RT ln(K_pair/K_Saha); the remaining thermal-function and level-population contribution is +6.261 kJ/mol at 1500 K and +7.037 kJ/mol at 2500 K, giving ratios 4.496 and 2.558. Separately, the Pankratz neutral anchor is 4.8 kJ/mol below Burcat's 602.0 kJ/mol neutral anchor. C3 uses that adopted pair.",
        "source_tables": {"cation": "Burcat-Th-plus", "neutral": "P672-Th-g"},
        "parent_activity": 1.0e-3,
        "max_ratio": 1.6534253288851723e-15,
        "temperature_K": 2300.0,
        "fO2": 1e-10,
        "neutral_pressure_bar": 2.1723092581101416e-18,
        "element_total_pressure_bar": 2.2710124684330474e-09,
        "joint_ion_pressure_bar": 3.7549495375212384e-24,
        "electron_pressure_bar": 4.935841167923754e-06,
        "K_ion": 8.531858178838494e-12,
        "isolated_bound": 8.368401213697542e17,
        "parent_oxide": "ThO2",
    },
}
ELEMENT_STATUS["Hf"] = {
    "status": "gas-complete-melt-pending",
    "criteria": {"C1": False, "C2": True, "C3": True, "C4": True},
    "validation": "unvalidated",
    "reason": "JANAF Hf(g), Hf+, and Hf- are available; Pankratz Bulletin 672 supplies the HfO2(cr) elemental formation function through 1800 K. HfO2(l) is a tier-4 liquid_from_solid caller standard built from that crystal, with constant-Cp extension above the source limit; HfO2 is the selected parent because no lower oxide liquid is sourced. The source-scale comparison uses Pankratz Hf(g) over its shared range. The joint C3 screen at a(HfO2)=1e-3 reached 1.140e-06 at 2800 K and fO2=1e-4 within the <=1-bar neutral-pressure domain. NIST ASD Saha and NIST WebBook attachment audits are separate and disclose the JANAF source offsets. No HfO(g) or HfO2(g) function was found in JANAF, NASA Glenn/CEA, Pankratz Bulletins 672/677/696, TSIV, NIST WebBook, OECD/NEA, or Burcat.",
    "c2_candidates": (("Hf-006", "Hf"),),
    "c3_ion_bound": {
        "status": "available",
        "reason": "JANAF Hf-007 and Hf-008 provide Hf+ and Hf- thermo functions. The Hf-007 minus Hf-006 formation difference is 660.555 kJ/mol versus the NIST ASD 6.825070 eV ionization energy (658.519 kJ/mol), a 2.036 kJ/mol 298 K offset; NIST ASD Saha ratios are 1.236 at 1500 K and 0.794 at 2500 K, including the temperature-dependent function/population offset. The NIST WebBook Hf atom page has no EA entry, so Hf- attachment uses Tang et al.'s primary 0.1780(6) eV measurement; ratios are 0.168 and 0.292. These source offsets are separate from the joint C3 ion-bound screen.",
        "source_tables": {"cation": "Hf-007", "anion": "Hf-008", "neutral": "Hf-006"},
        "upstream_sha256": {
            "Hf-006": "54dc600c8f019494179e8e322a47a9d98eaba19c3b0f64278a4eea4c683c8e1b",
            "Hf-007": "1f36fd116532d2ff548ffd70a31953a024f5a38be6ebd7df6b349c7458fc8359",
            "Hf-008": "c5670f809d516a57712f479edd14279f5e47820391f51e1dcab3f3a6b6013eb2",
            "D-020": "c9be269f34eb1a7ffd2c599a8540c44ba002cd5602db4efdab94bb27bd8e1dfd",
        },
        "user_agent": "neutral-source-vendor/1.0",
        "parent_activity": 0.001,
        "max_ratio": 1.1395960271219746e-06,
        "temperature_K": 2800.0,
        "fO2": 1e-4,
        "neutral_pressure_bar": 6.911595189680933e-16,
        "element_total_pressure_bar": 6.911595189680933e-16,
        "joint_ion_pressure_bar": 7.876426419235743e-22,
        "electron_pressure_bar": 8.131780113388633e-05,
        "K_ion": 8.144468639954743e-11,
        "isolated_bound": 334980761349517.25,
        "parent_oxide": "HfO2",
    },
}
ELEMENT_STATUS["La"] = {
    "status": "gas-partial",
    "criteria": {"C1": False, "C2": True, "C3": False, "C4": True},
    "validation": "unvalidated",
    "reason": "Konings Part 1 Table 42 and Part 2 Tables 88–89 provide La(g), LaO(g), and the La2O3(l) phase chain through 3000 K. The table's constant-Cp liquid branch is supercooled below 2577 K, and La2O3 is the selected parent because it is the sourced liquid oxide; a(La2O3) is caller-supplied. No ion thermal function was selected for this batch, so C3 is declined.",
    "c2_candidates": (("Konings-Table89-LaO", "LaO"),),
    "c3_ion_bound": {
        "status": "declined",
        "reason": "No La+ or La− thermal function is present in the selected Konings tables, NIST-JANAF index, NASA Glenn thermo.inp corpus, or local Pankratz Bulletins 672/677/696 records. The NIST Chemistry WebBook La+ page contains reaction thermochemistry, not a temperature-dependent ideal-gas function. Burcat was searched by species name; no La+ or La− card was located.",
        "source_tables": {"konings": "Part 1 Table 42; Part 2 Tables 88–89", "janaf": "name index", "nasa": "thermo.inp inventory", "pankratz": "Bulletin 672 p. 213; Bulletins 677 and 696 inventory", "nist_webbook": "La+ species page", "burcat": "species search"},
        "parent_oxide": "La2O3",
        "parent_activity": 1.0e-3,
    },
}
# Y2O3 fusion entropy is 119 kJ/mol ÷ 2712 K = 43.879 J/(mol K). Use the
# complete melting-region step: Pankratz's separate 1.297 kJ/mol alpha-beta
# jump is already in the crystal baseline, but is not part of Kapush's step.
ELEMENT_STATUS["Y"] = {
    "status": "gas-partial",
    "criteria": {"C1": False, "C2": True, "C3": False, "C4": True},
    "validation": "unvalidated",
    "reason": "Pankratz Bulletin 672 printed p. 474 supplies Y(g), and printed p. 475 supplies the Y2O3(cr) elemental formation table; Pedley and Marshall (1983) supplies the selected YO(g) assessment and thermal functions. Y2O3(l) is the required tier-4 liquid_from_solid construction from the Pankratz crystal, using the measured 2712 +/- 12 K melting point. Kapush et al.'s 119 +/- 10 kJ/mol cubic-crystal-to-liquid step is used directly; the distinct 0.310 kcal/mol alpha-beta transition at 1330 K remains in the crystal function and is not subtracted from the melting-region step. Values above the Pankratz crystal range are flagged; activities are caller-supplied. No Y+ temperature-dependent gas function was found in the checked JANAF, NASA Glenn, Pankratz 672/677/696, TSIV, NIST WebBook, OECD/NEA, or Burcat routes, so C3 is declined.",
    "c2_candidates": (("PM83-YO", "YO"),),
    "c3_ion_bound": {
        "status": "declined",
        "reason": "No Y+(g) temperature-dependent ideal-gas function was located in JANAF, NASA Glenn, Pankratz Bulletins 672/677/696, TSIV, NIST WebBook, OECD/NEA, or Burcat.",
        "source_tables": {"routes_checked": "JANAF; NASA Glenn/CEA; Pankratz 672/677/696; TSIV/IVTAN; NIST WebBook; OECD/NEA; Burcat"},
        "parent_oxide": "Y2O3",
        "parent_activity": 1.0e-3,
    },
}
for _element, _parent, _products, _fusion, _note in (
    ("Ce", "Ce2O3", "Ce(g), CeO(g), CeO2(g)", 2512, "CeO2(l) additionally supplies the same-state parent for CeO2(g) below its 3083 K fusion marker."),
    ("Pr", "Pr2O3", "Pr(g), PrO(g)", 2583, "PrO2 and PrO1.833 are crystal-only and decompose; neither supplies a liquid parent."),
    ("Nd", "Nd2O3", "Nd(g), NdO(g)", 2577, ""),
    ("Sm", "Sm2O3", "Sm(g), SmO(g)", 2613, "The complete B-form 298.15 K anchor is used because the C-form entropy is not tabulated."),
    ("Eu", "Eu2O3", "Eu(g), EuO(g)", 2622, "EuO(cr) and Eu3O4(cr) are crystal-only; the Eu2O3 parent convention does not represent Eu2+ melt speciation."),
    ("Gd", "Gd2O3", "Gd(g), GdO(g)", 2693, "The C-form Cp range is cut at the 1473 K C→B transition; the B form has no separate 298.15 K anchor."),
    ("Tb", "Tb2O3", "Tb(g), TbO(g)", 2682, "The C→B→H liquid chain has no A phase; TbO2(cr) decomposes and is not offered as a parent."),
    ("Dy", "Dy2O3", "Dy(g), DyO(g)", 2680, "The C→B→H liquid chain has no A phase; DyO(g) retains its ±20 kJ/mol formation uncertainty."),
    ("Ho", "Ho2O3", "Ho(g), HoO(g)", 2686, "The C→B→H liquid chain has no A phase."),
    ("Er", "Er2O3", "Er(g), ErO(g)", 2690, "The C→H liquid chain has no B phase."),
    ("Tm", "Tm2O3", "Tm(g), TmO(g)", 2682, "The C→H liquid chain has no B phase; 1680 K is a solid transition, not fusion."),
    ("Yb", "Yb2O3", "Yb(g), YbO(g)", 2707, "The C→H liquid chain has no B phase; 1365 K is a solid transition, not fusion. Reduced melts may contain Yb2+; this channel uses caller-supplied Yb2O3 activity without a speciation model."),
    ("Lu", "Lu2O3", "Lu(g), LuO(g)", 2762, "The selected source chain is crystal→liquid with no high polymorph."),
):
    _ion_routes = "JANAF, NASA Glenn, Pankratz Bulletins 672/677/696, NIST WebBook, and Burcat"
    _liquid_note = (
        f" The {_parent} liquid branch is supercooled below {_fusion} K."
        if _element != "Ce" else
        " The Ce2O3 and CeO2 liquid branches are supercooled below 2512 K and 3083 K."
    )
    ELEMENT_STATUS[_element] = {
        "status": "gas-partial",
        "criteria": {"C1": False, "C2": True, "C3": False, "C4": True},
        "validation": "unvalidated",
        "reason": (
            f"Konings Part 1 Table 42 and Part 2 Tables 88–89 provide {_products} "
            f"and the {_parent} liquid parent through 3000 K.{_liquid_note} "
            f"{_note + ' ' if _note else ''}Activities are caller-supplied. "
            f"No ion thermal function was selected after checking {_ion_routes}."
        ),
        "c2_candidates": tuple(
            (f"Konings-Table89-{species}", species)
            for species in (("CeO", "CeO2") if _element == "Ce" else
                            (f"{_element}O",))
        ),
        "c3_ion_bound": {
            "status": "declined",
            "reason": f"No {_element}+ or {_element}− temperature-dependent ideal-gas thermal function is available in the checked sources.",
            "source_tables": {"konings": "Part 1 Table 42; Part 2 Tables 88–89", "routes_checked": _ion_routes},
            "parent_oxide": _parent,
            "parent_activity": 1.0e-3,
        },
    }

# Halogen gas-only channels use the gas-parent activity path; no oxide parent
# or melt model is provided for elemental Br, I, Cl, or F.
for _element, _parent, _species in (
    ("Br", "Br2", "JANAF Br2(g), Br(g), Na/K/Li and alkaline-earth bromides; NASA Glenn Rb/Cs bromides"),
    ("I", "I2", "JANAF I2(g), I(g), KI, LiI, and alkaline-earth iodides; NASA Glenn NaI and Rb/Cs iodides"),
    ("Cl", "Cl2", "JANAF Cl2(g), Cl(g), Na/K/Li/Cs and alkaline-earth chlorides; NASA Glenn Rb chlorides"),
    ("F", "F2", "JANAF F2(g), F(g), Na/K/Li/Cs and alkaline-earth fluorides; NASA Glenn Rb fluorides"),
):
    ELEMENT_STATUS[_element] = {
        "status": "gas-complete-melt-pending",
        "criteria": {"C1": False, "C2": True, "C3": False, "C4": True},
        "validation": "unvalidated",
        "reason": (
            f"{_species} cover 1200-3000 K; callers supply "
            f"a({_parent})=f({_parent})/p° on the 1-bar gas reference. "
            "No ion channel or halogen dissolution model is included."
        ),
        "c2_candidates": (),
        "c3_ion_bound": {
            "status": "not computed",
            "reason": (
                "not computed: ionized bromine channels are outside this neutral "
                "gas-only change"
                if _element == "Br"
                else "not computed: ionized iodine channels are outside this neutral "
                "gas-only change"
                if _element == "I"
                else f"not computed: ionized {'chlorine' if _element == 'Cl' else 'fluorine'} channels are outside this neutral gas-only change"
            ),
            "source_tables": {
                "selected": (
                    {
                        "Br": "JANAF; NASA Glenn for Rb/Cs bromides",
                        "I": "JANAF; NASA Glenn for NaI and Rb/Cs iodides",
                        "Cl": "JANAF; NASA Glenn for Rb chlorides",
                        "F": "JANAF; NASA Glenn for Rb fluorides",
                    }[_element]
                )
            },
            "parent_activity": None,
        },
    }
for _element, _parent, _species, _reference, _c2_candidates in (
    (
        "H", "H2", "JANAF H-001, H-038, and H-064 gas rows",
        "JANAF H-050 H2(g) reference",
        (("H-038", "OH"), ("H-064", "H2O")),
    ),
    (
        "C", "C", "JANAF C-093 and C-095 gas rows",
        "JANAF C-002 graphite reference",
        (("C-093", "CO"), ("C-095", "CO2")),
    ),
    (
        "N", "N2", "JANAF N-002, N-005, and N-007 gas rows",
        "JANAF N-023 N2(g) reference",
        (("N-005", "NO"), ("N-007", "NO2")),
    ),
):
    ELEMENT_STATUS[_element] = {
        "status": "gas-complete-melt-pending",
        "criteria": {"C1": False, "C2": True, "C3": False, "C4": True},
        "validation": "unvalidated",
        "reason": (
            f"{_species} and {_reference} cover 1200-3000 K; callers supply "
            "a(H2)=f(H2)/p° on the 1-bar gas standard state. HCl, HF, HBr, "
            "and HI additionally require both H2 and X2. No dissolved-hydrogen "
            "budget or ion channel is modeled."
            if _element == "H"
            else f"{_species} use the {_reference} over 1200-3000 K; callers "
            "supply a(C) relative to graphite. No dissolved-carbon budget or "
            "activity model is provided."
            if _element == "C"
            else f"{_species} and {_reference} cover 1200-3000 K; callers "
            "supply a(N2)=f(N2)/p° on the 1-bar gas standard state. No "
            "dissolved-nitrogen budget or ion channel is modeled."
        ),
        "c2_candidates": _c2_candidates,
        "c3_ion_bound": {
            "status": "not computed",
            "reason": "not computed: ionized channels are outside this neutral gas change",
            "source_tables": {"selected": "JANAF"},
            "parent_activity": None,
        },
    }
ELEMENT_STATUS["Hg"] = {
    "status": "gas-complete-melt-pending",
    "criteria": {"C1": False, "C2": True, "C3": True, "C4": True},
    "validation": "unvalidated",
    "reason": (
        "Burcat BU-2587 Hg(g) and BU-2586 Hg(l) provide one-compilation "
        "metal-vapour coverage; callers supply a(Hg) relative to liquid Hg. "
        "The liquid row is continued above its 629.839 K source limit and "
        "flagged. JANAF Hg-004 supplies opt-in Hg+(g); its NIST ASD Saha "
        "ratios are 196.284 at 1500 K and 196.161 at 2500 K, a source entropy "
        "offset retained without tuning. At a(Hg)=1e-3, the C3 screen reaches "
        "2.020e-17 at 1700 K and fO2=1e-4, below the 1e-4 threshold."
    ),
    "c2_candidates": (),
    "c3_ion_bound": {
        "status": "computed",
        "parent_oxide": "Hg",
        "source_tables": {
            "neutral": "BU-2587", "cation": "Hg-004", "electron": "D-020",
        },
        "parent_activity": 1.0e-3,
        "max_ratio": 2.0204928231203823e-17,
        "isolated_bound": 6.245214561301318e-14,
        "temperature_K": 1700.0,
        "fO2": 1.0e-4,
        "neutral_pressure_bar": 0.9121537423279414,
        "element_total_pressure_bar": 0.9121537423279414,
        "joint_ion_pressure_bar": 1.843000089956004e-17,
        "electron_pressure_bar": 1.7607814709150057e-10,
        "K_ion": 3.557646325067119e-27,
    },
    "source_tables": {"gas": "BU-2587", "parent": "BU-2586"},
    "parent_activity": "Hg",
}
for _element, _gas_table, _parent_table, _phase_marker, _saha_note, _c3 in (
    (
        "Au", "BU-0056", "BU-0054/BU-0055", 1337,
        "NIST ASD Saha ratios are 0.527 at 1500 K and 0.681 at 2500 K.",
        {
            "source_tables": {
                "neutral": "BU-0056", "cation": "Burcat-Au-plus",
                "anion": "Burcat-Au-minus", "electron": "D-020",
            },
            "max_ratio": 0.0014494135223238737,
            "isolated_bound": 2.7799594603323795e-06,
            "temperature_K": 2800.0,
            "fO2": 1.0e-4,
            "neutral_pressure_bar": 0.0003076462576336098,
            "element_total_pressure_bar": 0.0003076462576336098,
            "joint_ion_pressure_bar": 4.4590664590648826e-07,
            "electron_pressure_bar": 8.111393356443537e-05,
            "K_ion": 2.377543994364913e-15,
        },
    ),
    (
        "Pd", "BU-3112", "BU-3110/BU-3111", 1827,
        "NIST ASD Saha ratios are 0.991986 at 1500 K and 0.975696 at 2500 K; "
        "the 2.43% deficit at 2500 K is retained as a source offset.",
        {
            "source_tables": {
                "neutral": "BU-3112", "cation": "Burcat-Pd-plus",
                "electron": "D-020",
            },
            "max_ratio": 1.6072867979839367e-08,
            "isolated_bound": 9.761839905919717e-05,
            "temperature_K": 2800.0,
            "fO2": 1.0e-4,
            "neutral_pressure_bar": 0.0001371562642352788,
            "element_total_pressure_bar": 0.0001371562642352788,
            "joint_ion_pressure_bar": 2.2044945276616e-12,
            "electron_pressure_bar": 8.131780214304317e-05,
            "K_ion": 1.307010298255831e-12,
        },
    ),
    (
        "Pt", "BU-3122/BU-3126", "BU-3120/BU-3121", 2042,
        "NIST ASD Saha ratios are 1.314 at 1500 K and 1.180 at 2500 K.",
        {
            "source_tables": {
                "neutral": "BU-3122", "cation": "Burcat-Pt-plus",
                "anion": "Burcat-Pt-minus", "electron": "D-020",
            },
            "max_ratio": 0.00024347537024246028,
            "isolated_bound": 0.00011298413832918825,
            "temperature_K": 2800.0,
            "fO2": 1.0e-4,
            "neutral_pressure_bar": 7.95216101229406e-07,
            "element_total_pressure_bar": 7.952688760338734e-07,
            "joint_ion_pressure_bar": 1.9362838403465257e-10,
            "electron_pressure_bar": 8.131771249628793e-05,
            "K_ion": 1.015126395560782e-14,
        },
    ),
):
    _c3_passes = _c3["max_ratio"] < 1.0e-4
    ELEMENT_STATUS[_element] = {
        "status": "gas-complete-melt-pending" if _c3_passes else "gas-partial",
        "criteria": {"C1": False, "C2": True, "C3": _c3_passes, "C4": True},
        "validation": "unvalidated",
        "reason": (
            f"Burcat {_gas_table} metal-gas and {_parent_table} stable-metal "
            f"rows provide one-compilation coverage over 1200-3000 K; callers "
            f"supply a({_element}) relative to the stable metal phase, with a "
            f"crystal-to-liquid fit step at {_phase_marker} K. Opt-in {_element}+ "
            + (
                "is sourced; no Pd- source was found. "
                if _element == "Pd"
                else f"and {_element}- source rows are included. "
            )
            + f"{_saha_note} "
            f"At a({_element})=1e-3, the C3 screen reaches "
            f"{_c3['max_ratio']:.3e} at {_c3['temperature_K']:.0f} K and "
            f"fO2={_c3['fO2']:.0e} within the <=1-bar neutral-pressure domain, "
            + ("below" if _c3_passes else "above")
            + " the 1e-4 threshold."
        ),
        "c2_candidates": (),
        "c3_ion_bound": {
            "status": "computed",
            "parent_oxide": _element,
            "parent_activity": 1.0e-3,
            **_c3,
        },
        "source_tables": {"gas": _gas_table, "parent": _parent_table},
        "parent_activity": _element,
    }
for _element, _parent in (
    ("Ca", "CaO"), ("Mg", "MgO"), ("Sr", "SrO"), ("Ba", "BaO"),
    ("Li", "Li2O"), ("Rb", "Rb2O"), ("Cs", "Cs2O"),
):
    ELEMENT_STATUS[_element]["halide_gas_channels"] = {
        "oxide_parent": _parent,
        "species": tuple(sorted(
            species for species in _HALIDE_REACTIONS
            if species.startswith(_element) or species.startswith(f"({_element}")
        )),
        "halogen_activity_required": True,
    }

ELEMENT_STATUS["Ru"] = {
    "status": "gas-partial",
    "criteria": {"C1": False, "C2": True, "C3": False, "C4": True},
    "validation": "unvalidated",
    "reason": (
        "Pankratz Bulletin 672 printed pp. 332-333 supplies paired Ru(g) and "
        "stable-metal phase functions over 1200-3000 K; callers supply a(Ru) "
        "relative to the stable metal phase, switching from crystal to liquid "
        "at 2607 K. No evaluated Ru+ or Ru- gas function was found in the "
        "checked JANAF, NASA Glenn/CEA, Pankratz 672/677/696, TSIV, NIST "
        "WebBook, OECD/NEA, or Burcat sources; C3 is declined."
    ),
    "c2_candidates": (),
    "c3_ion_bound": {
        "status": "declined",
        "reason": "No evaluated Ru+ or Ru- thermo function was found in the checked JANAF, NASA Glenn/CEA, Pankratz 672/677/696, TSIV, NIST WebBook, OECD/NEA, or Burcat sources.",
    },
    "source_tables": {"gas": "P672-Ru-g", "parent": "P672-Ru-cr/P672-Ru-l"},
    "parent_activity": "Ru",
}
ELEMENT_STATUS["Rh"] = {
    "status": "gas-partial",
    "criteria": {"C1": False, "C2": True, "C3": False, "C4": True},
    "validation": "unvalidated",
    "reason": (
        "Pankratz Bulletin 672 printed pp. 330-331 supplies paired Rh(g) and "
        "stable-metal functions; Rh(g) uses a flagged held-Cp continuation "
        "above its 2000 K source limit. Callers supply a(Rh) relative to "
        "Rh(cr)/Rh(l), switching at 2237 K. No evaluated Rh+ or Rh- gas "
        "function was found in the checked JANAF, NASA Glenn/CEA, Pankratz "
        "672/677/696, TSIV, NIST WebBook, OECD/NEA, or Burcat sources; C3 is declined."
    ),
    "c2_candidates": (),
    "c3_ion_bound": {
        "status": "declined",
        "reason": "No evaluated Rh+ or Rh- thermo function was found in the checked JANAF, NASA Glenn/CEA, Pankratz 672/677/696, TSIV, NIST WebBook, OECD/NEA, or Burcat sources.",
    },
    "source_tables": {"gas": "P672-Rh-g", "parent": "P672-Rh-cr/P672-Rh-l"},
    "parent_activity": "Rh",
}
ELEMENT_STATUS["Ir"] = {
    "status": "gas-partial",
    "criteria": {"C1": False, "C2": True, "C3": False, "C4": True},
    "validation": "unvalidated",
    "reason": (
        "Burcat cards BU-2613, BU-2611, and BU-2612 provide paired Ir(g) and "
        "stable iridium phase functions on one elemental reference scale. "
        "Callers supply a(Ir) relative to Ir(cr)/Ir(l), switching at 2720 K. "
        "No evaluated Ir+ or Ir- gas function was found in the checked JANAF, "
        "NASA Glenn/CEA, Pankratz 672/677/696, TSIV, NIST WebBook, OECD/NEA, "
        "or Burcat sources; C3 is declined."
    ),
    "c2_candidates": (),
    "c3_ion_bound": {
        "status": "declined",
        "reason": "No evaluated Ir+ or Ir- thermo function was found in the checked JANAF, NASA Glenn/CEA, Pankratz 672/677/696, TSIV, NIST WebBook, OECD/NEA, or Burcat sources.",
    },
    "source_tables": {"gas": "BU-2613", "parent": "BU-2611/BU-2612"},
    "parent_activity": "Ir",
}
# --------------------------------------------------------------------------- #
# Data pack
# --------------------------------------------------------------------------- #


def _gas_table_identity_cell(value: object) -> object:
    """Map parsed pandas cells to values accepted by the canonical encoder."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return None
        return {
            "non_finite_float": "positive_infinity" if value > 0.0 else "negative_infinity"
        }
    return value


def _gas_table_digest(table_name: str, table: "pd.DataFrame") -> str:
    """Hash every parsed cell, independent of CSV column and row order."""
    rows = [
        {
            column: (
                None
                if column == "cation" and value == ""
                else _gas_table_identity_cell(value)
            )
            for column, value in row.items()
        }
        for row in table.reset_index().to_dict(orient="records")
    ]
    rows.sort(key=_canonical_published_serialization)
    payload = {
        "table": table_name,
        "columns": sorted((table.index.name, *table.columns)),
        "rows": rows,
    }
    return hashlib.sha256(_canonical_published_serialization(payload)).hexdigest()


def _canonical_text_value(
    value: object, pd: object, *, allow_empty: bool = False
) -> str:
    """Convert text-schema cells to stable Python strings."""
    if pd.isna(value):
        if allow_empty:
            return ""
        raise ValueError("missing")
    if isinstance(value, (bool, np.bool_)):
        result = str(value)
    elif isinstance(value, (int, np.integer)):
        result = str(int(value))
    elif isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("non-finite")
        result = str(int(number)) if math.isfinite(number) and number.is_integer() else str(number)
    elif isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("non-finite")
        result = str(value)
    else:
        result = str(value)
    if not result.strip() and not allow_empty:
        raise ValueError("empty")
    return result


def _canonical_numeric_column(
    table: "pd.DataFrame",
    column: str,
    table_name: str,
    table_path: Path,
    species_labels: list[str],
    pd: object,
) -> np.ndarray:
    """Convert one complete numeric column to exact, finite binary64 values."""
    values = np.empty(len(table), dtype=np.float64)
    for position, value in enumerate(table[column].array):
        species = species_labels[position]
        invalid_prefix = (
            f"{table_name} database {table_path} has non-finite or non-numeric "
            f"{column} for {species!r}: {value!r}"
        )
        if pd.isna(value) or isinstance(value, (bool, np.bool_)):
            raise ImccGasInvalidIntervalError(invalid_prefix)
        if not isinstance(value, (Number, Decimal, Fraction)):
            if not isinstance(value, (str, np.str_)):
                raise ImccGasInvalidIntervalError(invalid_prefix)
            try:
                decimal_value = Decimal(str(value).strip())
            except InvalidOperation:
                raise ImccGasInvalidIntervalError(invalid_prefix) from None
            if not decimal_value.is_finite():
                raise ImccGasInvalidIntervalError(invalid_prefix)
            try:
                number = float(decimal_value)
            except (OverflowError, ValueError):
                raise ImccGasInvalidIntervalError(invalid_prefix) from None
            if (
                not math.isfinite(number)
                or decimal_value != Decimal.from_float(number)
            ):
                raise ImccGasInvalidIntervalError(
                    f"{table_name} database {table_path} cannot exactly convert "
                    f"{column} for {species!r}: {value!r}"
                )
            values[position] = number
            continue
        if isinstance(value, (int, np.integer)) and abs(int(value)) > 2**53:
            raise ImccGasInvalidIntervalError(
                f"{table_name} database {table_path} cannot exactly convert "
                f"{column} for {species!r}: {value!r}"
            )
        try:
            number = float(value)
        except (OverflowError, TypeError, ValueError):
            raise ImccGasInvalidIntervalError(invalid_prefix) from None
        if not math.isfinite(number):
            raise ImccGasInvalidIntervalError(invalid_prefix)
        try:
            exact = bool(value == number)
        except (TypeError, ValueError):
            exact = False
        if not exact:
            raise ImccGasInvalidIntervalError(
                f"{table_name} database {table_path} cannot exactly convert "
                f"{column} for {species!r}: {value!r}"
            )
        values[position] = number
    return values


def _canonicalize_gas_table(
    table: "pd.DataFrame",
    table_name: str,
    table_path: Path,
    schema: frozenset[str],
    numeric_columns: tuple[str, ...],
    text_columns: tuple[str, ...],
) -> "pd.DataFrame":
    """Validate the closed table schema and canonicalize every stored value."""
    pd = _pandas()
    if not isinstance(table, pd.DataFrame):
        raise ImccGasInvalidIntervalError(
            f"{table_name} database {table_path} requires a pandas DataFrame"
        )
    if isinstance(table.index, pd.MultiIndex):
        raise ImccGasInvalidIntervalError(
            f"{table_name} database {table_path} requires a simple species index"
        )
    expected_columns = schema - {"species_name"}
    actual_columns = list(table.columns)
    if (
        len(actual_columns) != len(set(actual_columns))
        or set(actual_columns) != expected_columns
    ):
        missing = sorted(expected_columns - set(actual_columns))
        extra = sorted(set(actual_columns) - expected_columns)
        raise ImccGasInvalidIntervalError(
            f"{table_name} database {table_path} has an invalid schema "
            f"(missing={missing}, extra={extra})"
        )

    species_labels = []
    for value in table.index:
        try:
            label = _canonical_text_value(value, pd)
        except ValueError:
            raise ImccGasInvalidIntervalError(
                f"{table_name} database {table_path} has a missing, empty, or "
                "whitespace-only species label"
            ) from None
        species_labels.append(label)

    canonical = table.copy(deep=True)
    canonical.index = pd.Index(species_labels, name="species_name", dtype=object)
    for column in numeric_columns:
        canonical[column] = _canonical_numeric_column(
            table, column, table_name, table_path, species_labels, pd
        )
    for column in text_columns:
        text_values = []
        for value in table[column].array:
            try:
                text_values.append(
                    _canonical_text_value(value, pd, allow_empty=column == "cation")
                )
            except ValueError:
                raise ImccGasInvalidIntervalError(
                    f"{table_name} database {table_path} has a missing, empty, or "
                    f"whitespace-only {column} label"
                ) from None
        canonical[column] = pd.Series(
            text_values, index=canonical.index, dtype=object
        )

    t_mins = canonical["T_min"].to_numpy(copy=False)
    t_maxs = canonical["T_max"].to_numpy(copy=False)
    invalid_bounds = np.flatnonzero(t_mins >= t_maxs)
    if invalid_bounds.size:
        position = int(invalid_bounds[0])
        species = species_labels[position]
        raise ImccGasInvalidIntervalError(
            f"{table_name} database {table_path} requires T_min < T_max "
            f"for {species!r}: [{t_mins[position]}, {t_maxs[position]}] K"
        )

    starts = set()
    for species, start in zip(species_labels, t_mins, strict=True):
        key = species, float(start)
        if key in starts:
            raise ImccGasDuplicateIntervalError(
                f"{table_name} database {table_path} has duplicate interval start "
                f"for {species!r} at T_min={start!r} K"
            )
        starts.add(key)
    return canonical


@dataclass(frozen=True)
class ImccGasDatapack:
    """Loaded JANAF + condensate tables and their parsed-content identities.

    ``gas_path`` and ``oxide_path`` are diagnostic source locations; content
    identity is exposed by the table digest properties. Both frames are
    canonicalized once at construction. In-place mutation of a stored frame is
    outside the contract; use :func:`dataclasses.replace` to create a validated
    pack with changed tables.
    """

    gas_df: pd.DataFrame
    oxide_df: pd.DataFrame
    gas_path: Path
    oxide_path: Path
    _gas_digest: str = field(init=False, repr=False, compare=False)
    _condensate_digest: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        gas_df = _canonicalize_gas_table(
            self.gas_df,
            "JANAF gas",
            self.gas_path,
            _GAS_TABLE_SCHEMA,
            _GAS_NUMERIC_COLUMNS,
            _GAS_TEXT_COLUMNS,
        )
        oxide_df = _canonicalize_gas_table(
            self.oxide_df,
            "condensate",
            self.oxide_path,
            _CONDENSATE_TABLE_SCHEMA,
            _CONDENSATE_NUMERIC_COLUMNS,
            _CONDENSATE_TEXT_COLUMNS,
        )
        object.__setattr__(self, "_gas_digest", _gas_table_digest("gas", gas_df))
        object.__setattr__(
            self, "_condensate_digest", _gas_table_digest("condensate", oxide_df)
        )
        object.__setattr__(self, "gas_df", gas_df)
        object.__setattr__(self, "oxide_df", oxide_df)

    @property
    def gas_table_digest(self) -> str:
        return self._gas_digest

    @property
    def condensate_table_digest(self) -> str:
        return self._condensate_digest


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
    notice_messages = []
    if gas_path is not None:
        gas_source = Path(gas_path)
        gas_header_source = gas_source
        gas_display_path = gas_source
        gas_df = pd.read_csv(gas_source)
    elif os.environ.get(_VAPOROCK_ENV_VAR):
        gas_source = _vaporock_root() / _GAS_TABLE_RELPATH
        gas_header_source = gas_source
        gas_display_path = gas_source
        gas_df = pd.read_csv(gas_source)
    else:
        gas_resource = _packaged_resource(_PACKAGED_GAS_NAME)
        gas_header_source = gas_resource
        gas_display_path = _packaged_database_path(_PACKAGED_GAS_NAME)
        with gas_resource.open("rb") as handle:
            gas_df = pd.read_csv(handle)
    if "species_name" not in gas_df.columns:
        raise ImccGasSpeciesNotFoundError(
            f"JANAF gas database {gas_display_path} missing 'species_name' column"
        )
    gas_df, gas_notice = _drop_unnamed_columns(
        gas_df, gas_header_source, gas_display_path
    )
    if gas_notice:
        notice_messages.append(gas_notice)
    gas_df = gas_df.set_index("species_name")

    if oxide_path is not None:
        oxide_source = Path(oxide_path)
        oxide_header_source = oxide_source
        oxide_display_path = oxide_source
        oxide_df = pd.read_csv(oxide_source)
    elif os.environ.get(_VAPOROCK_ENV_VAR):
        oxide_source = _vaporock_root() / _CONDENSATE_TABLE_RELPATH
        if not oxide_source.is_file():
            raise ImccGasDataUnavailableError(
                f"{_VAPOROCK_ENV_VAR}={os.environ[_VAPOROCK_ENV_VAR]!r} does not contain "
                f"{_CONDENSATE_TABLE_RELPATH}"
            )
        oxide_header_source = oxide_source
        oxide_display_path = oxide_source
        oxide_df = pd.read_csv(oxide_source)
    else:
        oxide_resource = _packaged_resource(_PACKAGED_CONDENSATE_NAME)
        oxide_header_source = oxide_resource
        oxide_display_path = _packaged_database_path(_PACKAGED_CONDENSATE_NAME)
        with oxide_resource.open("rb") as handle:
            oxide_df = pd.read_csv(handle)
    oxide_df, oxide_notice = _drop_unnamed_columns(
        oxide_df, oxide_header_source, oxide_display_path
    )
    if oxide_notice:
        notice_messages.append(oxide_notice)
    if notice_messages:
        warnings.warn(
            ImccGasIngestionWarning("; ".join(notice_messages)), stacklevel=2
        )
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


def _source_model_flag(
    row_name: str, T: float, row: pd.Series, *, phase: str
) -> str | None:
    """Name a selected row's coverage construction or upper-tail continuation."""
    source_ref = str(row.get("Ref", ""))
    declared_continuation = re.fullmatch(
        r"(.+)-CONT-(\d+(?:\.\d+)?)", source_ref
    )
    if declared_continuation is not None and T >= float(declared_continuation.group(2)):
        continuation_limit = float(declared_continuation.group(2))
        if phase == "l" and "-METAL-CONT-" in source_ref:
            return (
                f"{row_name} uses a flagged liquid-metal continuation above the "
                f"{continuation_limit:g} K source limit"
            )
        return (
            f"{row_name} uses a flagged {phase}-phase continuation above the "
            f"{continuation_limit:g} K source limit"
        )
    konings_liquid = re.fullmatch(r"Konings-Table88-Tf(\d+(?:\.\d+)?)", source_ref)
    if phase == "l" and konings_liquid is not None and T < float(konings_liquid[1]):
        return (
            f"{row_name} uses the printed constant-Cp liquid branch as a "
            f"supercooled continuation below {float(konings_liquid[1]):g} K"
        )
    supercooled = re.fullmatch(r"(TSIV-\d+)-SC-(\d+(?:\.\d+)?)", source_ref)
    if phase == "l" and supercooled is not None:
        return (
            f"{row_name} uses the TSIV liquid function as a supercooled branch "
            f"below {float(supercooled.group(2)):g} K"
        )
    extension = _SOURCE_CP_EXTENSION_RE.search(source_ref)
    source_table = _source_table_identifier(source_ref)
    if phase == "l" and _liquid_from_solid_reference(source_ref) is not None:
        if extension is None:
            return (
                f"{row_name} uses a tier-4 liquid_from_solid construction "
                f"from {source_table}"
            )
        limit = float(extension["limit"])
        if source_table == "TSIV-1603":
            return (
                f"{row_name} uses a tier-4 liquid_from_solid construction from "
                f"{source_table}; the central liquid uses the printed 2250 K "
                "node, and the quadratic crystal-Cp extension is its uncertainty bound"
            )
        return (
            f"{row_name} uses a tier-4 liquid_from_solid construction from "
            f"{source_table}; its crystal Cp was extended above {limit:g} K "
            "to the estimated fusion temperature"
        )
    source_marker = _source_supercooled_liquid_markers().get(row_name)
    if (
        phase == "l"
        and source_marker is not None
        and source_ref in {source_marker["table_id"], "JANAF"}
        and not source_ref.endswith(_SUPERCOOLED_EXTENSION_REF_SUFFIX)
        # Below the packaged row's own floor the source branch is not
        # tabulated; the outside-interval flag already describes that case.
        and T >= float(row["T_min"])
        and T < source_marker["fusion_temperature_K"]
    ):
        glass_join = source_marker["glass_join_temperature_K"]
        if glass_join is not None and T < glass_join:
            return (
                f"{row_name} uses source-tabulated glass cells below the "
                f"{glass_join:g} K glass/liquid join ({source_marker['table_id']})"
            )
        return (
            f"{row_name} uses a source-tabulated supercooled liquid branch "
            f"below the {source_marker['fusion_temperature_K']:g} K fusion "
            f"marker ({source_marker['table_id']})"
        )
    if extension is not None and T > float(extension["limit"]):
        limit = float(extension["limit"])
        return (
            f"{row_name} uses a labelled constant-Cp continuation above "
            f"{limit:g} K from {source_table}"
        )
    return None


@lru_cache(maxsize=1)
def _source_supercooled_liquid_markers() -> dict[str, dict[str, object]]:
    resource = _packaged_resource(_PACKAGED_SOURCE_LIQUID_MARKERS_NAME)
    metadata = json.loads(resource.read_text(encoding="utf-8"))
    if metadata.get("schema") != "openimcc.source-liquid-markers.v1":
        raise ValueError("unsupported source-liquid marker metadata schema")
    return metadata["rows"]


def _derived_source_supercooled_liquid_markers(
    datapack: ImccGasDatapack,
) -> dict[str, dict[str, object]]:
    """Select packaged source liquids with runtime rows below fusion."""
    derived = {}
    for species, marker in _source_supercooled_liquid_markers().items():
        if species not in datapack.oxide_df.index:
            continue
        rows = datapack.oxide_df.loc[species]
        if getattr(rows, "ndim", 1) == 1:
            species_rows = [rows]
        else:
            species_rows = [row for _, row in rows.iterrows()]
        for row in species_rows:
            source_ref = str(row["Ref"])
            if source_ref not in {marker["table_id"], "JANAF"}:
                continue
            if _liquid_from_solid_reference(source_ref) is not None:
                continue
            if source_ref.endswith(_SUPERCOOLED_EXTENSION_REF_SUFFIX):
                continue
            if float(row["T_min"]) < float(marker["fusion_temperature_K"]):
                derived[species] = marker
                break
    return derived


def _liquid_from_solid_reference(
    source_ref: str,
) -> tuple[str, float | None] | None:
    """Read explicit liquid-from-solid metadata, retaining legacy table refs."""
    base_ref = _SOURCE_CP_EXTENSION_RE.sub("", source_ref)
    match = _LIQUID_FROM_SOLID_REF_RE.fullmatch(base_ref)
    if match is not None:
        return match["source"], float(match["temperature"])
    if source_ref == "K-012":
        return "K-012", None
    if "-LFS" in source_ref:
        return _source_table_identifier(source_ref), None
    return None


def _supercooled_liquid_from_solid_flag(
    row_name: str, T: float, row: pd.Series
) -> str | None:
    """Flag a constructed liquid below its measured or selected fusion point."""
    construction = _liquid_from_solid_reference(str(row.get("Ref", "")))
    if construction is None or construction[1] is None or T >= construction[1]:
        return None
    return (
        f"{row_name} uses a supercooled liquid_from_solid branch below "
        f"the {construction[1]:g} K fusion temperature"
    )


def _source_table_identifier(source_ref: str) -> str:
    """Remove runtime construction suffixes while retaining the source ID."""
    base_ref = _SOURCE_CP_EXTENSION_RE.sub("", source_ref)
    match = _LIQUID_FROM_SOLID_REF_RE.fullmatch(base_ref)
    if match is not None:
        return match["source"]
    base = _SOURCE_CP_EXTENSION_RE.sub("", source_ref)
    return base.removesuffix("-LFS").removesuffix(_SUPERCOOLED_EXTENSION_REF_SUFFIX)


def _nearest_interval_row(
    df: pd.DataFrame, species: str, T: float, allow_extrapolation: bool = False
) -> pd.Series:
    """Select one thermodynamic interval for ``species`` at temperature ``T``.

    Selection inside this function: among rows with ``T_min <= T``, take the
    one with the largest ``T_min``. If every ``T_min`` is ``> T``, take the
    row with the smallest ``T_min``: the interval nearest to ``T``, because
    low-temperature intervals are appended after the rows they extend. Tied
    ``T_min`` values are refused. Then,
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
    t_mins = rows["T_min"].to_numpy(copy=False)
    t_maxs = rows["T_max"].to_numpy(copy=False)
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

    JANAF and NASA CEA gas cards use the same 1-bar ideal-gas standard state.
    JANAF ion formation functions use JANAF's elemental reference states;
    NASA ion cards retain their published functions. D-020 treats e-(g) as a
    monatomic ideal gas and tabulates
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
                "; ".join(
                    flag
                    for flag in (
                        _outside_interval_flag(row_name, T, row),
                        _source_model_flag(row_name, T, row, phase="g"),
                    )
                    if flag
                ) or None,
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
                str(ion_row.get("Ref", "")),
            )
        )

    for neutral, ion in _GAS_ELECTRON_ATTACHMENTS:
        if neutral not in values:
            continue
        G_neutral, _neutral_row, neutral_flag = gibbs(neutral)
        G_ion, ion_row, ion_flag = gibbs(ion)
        # Premise: A + e- = A- uses the same ideal-gas standard state.
        # Algebra: K_A = exp[-(G(A-) - G(A) - G(e-))/(R*T)]. Its partition
        # form is (p°/kBT)*(Z-/2Z0)*(h²/(2*pi*me*kBT))^(3/2)*exp(EA/kBT):
        # p°/kBT is m^-3 and the electron translational factor m^3. It is the
        # inverse cation Saha form, so detailed balance gives K_attach*K_ion=1.
        dG = G_ion - G_neutral - G_electron
        equilibrium = math.exp(-dG / (R_J_MOL_K * T))
        negative_sum += equilibrium * values[neutral]
        negative.append(
            (
                neutral,
                ion,
                equilibrium,
                (neutral_flag, ion_flag, electron_flag, domain_flags.get(neutral)),
                str(ion_row.get("Ref", "")),
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

    for neutral, ion, equilibrium, flags, source_id in positive:
        values[ion] = (
            equilibrium * values[neutral] / electron_pressure
            if electron_pressure > 0.0
            else 0.0
        )
        domain_flags[ion] = "; ".join(flag for flag in flags if flag) or None
        provenance_class[ion] = (
            "nasa_glenn_fitted_ionisation"
            if source_id.startswith("NG-")
            else _ION_PROVENANCE_CLASS
        )
        provenance_class[ion] = _GAS_PROVENANCE_AUTHORITY[ion]

    for neutral, ion, equilibrium, flags, source_id in negative:
        values[ion] = equilibrium * values[neutral] * electron_pressure
        domain_flags[ion] = "; ".join(flag for flag in flags if flag) or None
        provenance_class[ion] = (
            "nasa_glenn_fitted_ionisation"
            if source_id.startswith("NG-")
            else _ION_PROVENANCE_CLASS
        )
        provenance_class[ion] = _GAS_PROVENANCE_AUTHORITY[ion]


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

    source_ref = str(row["Ref"])
    if phase == "g" and source_ref:
        source_table_id = _source_table_identifier(source_ref)
    elif phase != "g" and source_ref == "LAM1987":
        source_table_id = _LAM1987_SOURCE_TABLE_IDS.get(species)
    elif phase != "g" and source_ref.endswith(_SUPERCOOLED_EXTENSION_REF_SUFFIX):
        source_table_id = source_ref.removesuffix(_SUPERCOOLED_EXTENSION_REF_SUFFIX)
    elif phase != "g" and source_ref == "K-012":
        source_table_id = source_ref
    elif phase != "g" and source_ref.startswith("BU-"):
        source_table_id = source_ref.split("-METAL-", 1)[0]
    elif phase != "g" and source_ref == "LAM1984":
        source_table_id = None
    elif phase != "g" and ("-LFS" in source_ref or _SOURCE_CP_EXTENSION_RE.search(source_ref)):
        source_table_id = _source_table_identifier(source_ref)
    else:
        source_table_id = _OXIDE_SOURCE_TABLE_IDS.get(species)

    return SpeciesThermo(
        Cp_J_molK=float(cp),
        S_J_molK=float(entropy),
        H_app_kJ_mol=float(h_app),
        G_J_mol=float(apparent_g),
        source_row_id=str(row["Ref"]),
        source_table_id=source_table_id,
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
        aligned with ``parent_oxides`` (default IMCC order). Oxide parents use
        their liquid standard state; Hg/Au/Pd activities use the stable pure
        metal phase at T, with Au/Pd crystal-to-liquid steps retained. Hg's
        liquid function above 629.839 K is a flagged continuation. Sulfur uses
        caller-supplied
        ``S2`` fugacity relative to JANAF's 1-bar gas standard. Halogen channels
        similarly accept ``Br2``, ``I2``, ``Cl2``, and ``F2`` fugacity ratios;
        their halide products use the supported metal-oxide liquid activities.
        ``H2`` and ``N2`` accept ``f(parent)/p°``; ``C`` is
        relative to graphite. HCl, HF, HBr, and HI require both H2 and X2;
        PS requires both ``P2O5`` and ``S2`` activities.
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
        may add ``P2O5``, ``S2``, ``Br2``, ``I2``, ``Cl2``, ``F2``, ``H2``,
        ``C``, ``N2``, ``Hg``, ``Au``, or ``Pd`` when supplying those parent
        activities.
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

        Halide rows use ``n_oxide*oxide(l) + n_X2*X2(g) =
        gas(g) + n_O2*O2(g)``. Their activity power is the reaction
        coefficient, and their oxygen-fugacity power is ``-n_O2``.

    The returned mapping reports those p̃_gas values as bar.  For the special
    retained species O2, p̃_O2 = fO2 by definition. For n_O2 = 0 the oxygen
    term vanishes, while p̃_gas = (K° * a_oxide)^(1/n_gas) still applies.

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
        reactions, omitted_channels = _default_reactions(tuple(act), datapack, T)

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
        side_key = _side_parent(gas_name)
        if side_key:
            if side_key not in act:
                raise ImccGasSpeciesNotFoundError(
                    f"gas channel {gas_name!r} needs parent activity "
                    f"{side_key!r}, which is not in parent_oxides"
                )
            side_activity = act[side_key]
            if not math.isfinite(side_activity) or side_activity < 0.0:
                raise ValueError(
                    f"activity of {side_key!r} must be finite and >= 0, "
                    f"got {side_activity}"
                )

    # G°(O2, T) is needed only for reactions that produce or consume O2.
    needs_O2_row = any(n_O2 != 0.0 for _, (_, _, n_O2) in reactions) or any(
        species in _HALIDE_REACTIONS
        and species in _ALKALI_HALIDE_REACTION_OXIDES
        and _ALKALI_HALIDE_REACTION_OXIDES[species][2] != 0.0
        for species, _reaction in reactions
    )
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
            provenance_class[gas_name] = _result_provenance_authority(
                gas_name, None
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
        flag = _source_model_flag(gas_species, T, gas_row, phase="g")
        if flag is not None:
            flags.append(flag)

        if O2_row is not None and (
            n_O2 != 0.0
            or gas_name in _HALIDE_REACTIONS
        ):
            flag = _outside_interval_flag("O2(g)", T, O2_row)
            if flag is not None:
                flags.append(flag)

        oxide_row = None
        if oxide:
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
                oxide_name = _parent_standard_row(oxide, T)
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
                source_name = "NASA Glenn/CEA" if source_table.startswith("NG-") else "JANAF"
                source_name = (
                    "NASA Glenn card"
                    if source_table.startswith("NG-")
                    else "JANAF"
                )
                flags.append(
                    f"{oxide_name} uses a labelled constant-Cp supercooled-liquid "
                    f"continuation from {source_name} {source_table}"
                )
            elif oxide_row is not None:
                parent_phase = "s" if str(oxide_row["state"]) == "s" else "l"
                flag = _source_model_flag(
                    oxide_name, T, oxide_row, phase=parent_phase
                )
                if flag is not None:
                    flags.append(flag)
                if parent_phase == "l":
                    flag = _supercooled_liquid_from_solid_flag(
                        oxide_name, T, oxide_row
                    )
                    if flag is not None:
                        flags.append(flag)
        else:
            G_oxide = 0.0
            a_oxide = 1.0

        hydrogen_halide_parent = _HYDROGEN_HALIDE_PARENTS.get(gas_name)
        mixed_side_parent = _MIXED_GAS_REACTION_SIDE_PARENTS.get(gas_name)
        if hydrogen_halide_parent:
            # 1/2 H2(g) + 1/2 X2(g) = HX(g); both activities use their
            # JANAF 1-bar gas references, and no oxygen enters the reaction.
            side_row_name = _parent_standard_row(hydrogen_halide_parent)
            side_row = _nearest_interval_row(
                datapack.gas_df,
                side_row_name,
                T,
                allow_extrapolation=allow_extrapolation,
            )
            G_side = _janaf_gibbs(T, side_row)
            flag = _outside_interval_flag(side_row_name, T, side_row)
            if flag is not None:
                flags.append(flag)
            dG = G_gas - n_gas * G_oxide - 0.5 * G_side
            Kp = np.exp(-dG / (R_J_MOL_K * T))
            p_gas = Kp * a_oxide**n_gas * act[hydrogen_halide_parent]**0.5
        else:
            side_reactant = _ALKALI_HALIDE_REACTION_OXIDES.get(gas_name)
        if not hydrogen_halide_parent and side_reactant:
            # Halide rows balance an oxide liquid plus X2(g). The side table
            # gives n_oxide and n_O2 explicitly; with product coefficient one,
            # Kp puts product-side oxygen in the denominator, so the pO2
            # exponent is -n_O2. This also leaves the X2 exponent in the main
            # reaction tuple, preserving the single gas-parent rule.
            side_oxide, n_oxide, side_n_O2 = side_reactant
            side_name = f"{side_oxide}(l)"
            side_row = _oxide_row_for_T(
                datapack.oxide_df,
                side_name,
                T,
                allow_extrapolation=allow_extrapolation,
            )
            G_side_oxide = _lamor_gibbs(T, side_row)
            flag = _outside_interval_flag(side_name, T, side_row)
            if flag is not None:
                flags.append(flag)
            flag = _source_model_flag(side_name, T, side_row, phase="l")
            if flag is not None:
                flags.append(flag)
            # The side table carries complete oxygen stoichiometry for new
            # halides whose reaction tuple has a zero O2 slot. Existing Na/K
            # rows keep using their tuple coefficient and unchanged arithmetic.
            side_G_O2 = (
                G_O2 if n_O2 != 0.0 or gas_name in _HALIDE_REACTIONS else 0.0
            )
            dG = G_gas + side_n_O2 * side_G_O2 - n_oxide * G_side_oxide - n_gas * G_oxide
            Kp = np.exp(-dG / (R_J_MOL_K * T))
            p_gas = (
                Kp
                * act[side_oxide] ** n_oxide
                * a_oxide**n_gas
                / p_O2**side_n_O2
            )
        elif mixed_side_parent:
            side_parent, n_side = mixed_side_parent
            side_name = _parent_standard_row(side_parent)
            side_row = _nearest_interval_row(
                datapack.gas_df,
                side_name,
                T,
                allow_extrapolation=allow_extrapolation,
            )
            G_side = _janaf_gibbs(T, side_row)
            flag = _outside_interval_flag(side_name, T, side_row)
            if flag is not None:
                flags.append(flag)
            # P2O5(l) + S2(g) = 2 PS(g) + 5/2 O2(g). The tuple coefficients
            # are per mole of P2O5; solve mass action for the two PS products.
            dG = n_gas * G_gas + n_O2 * G_O2 - G_oxide - n_side * G_side
            Kp = np.exp(-dG / (R_J_MOL_K * T))
            p_gas = (
                Kp
                * a_oxide
                * act[side_parent] ** n_side
                / p_O2**n_O2
            ) ** (1.0 / n_gas)
        elif not hydrogen_halide_parent and oxide in _GAS_PHASE_PARENT_ROWS:
            # Sulfur tuple: n_S2 S2(g) + n_O2 O2(g) = gas(g).
            # Kp = p_gas / (a_S2**n_S2 * fO2**n_O2), with ΔG° formed from
            # the same balanced reaction; a_S2 is caller-supplied f(S2)/p°.
            dG = G_gas - n_gas * G_oxide - n_O2 * G_O2
            Kp = np.exp(-dG / (R_J_MOL_K * T))
            p_gas = Kp * (a_oxide**n_gas) * (p_O2**n_O2)
        elif not hydrogen_halide_parent:
            # Condensed-parent reaction (1): oxide(l) = n_gas*gas(g) +
            # n_O2*O2(g). Keep its established arithmetic path unchanged.
            dG = n_gas * G_gas + n_O2 * G_O2 - G_oxide
            Kp = np.exp(-dG / (R_J_MOL_K * T))
            p_gas = (Kp * a_oxide / (p_O2**n_O2)) ** (1.0 / n_gas)
        result[gas_name] = float(p_gas)
        domain_flags[gas_name] = "; ".join(flags) or None
        provenance_class[gas_name] = _result_provenance_authority(
            gas_name, oxide_row
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
    "Ta": 180.94788, "Be": 9.0121831, "Tl": 204.38,
    "Mn": 54.938044, "Ni": 58.6934, "Co": 58.933194,
    "Sb": 121.760, "Ag": 107.8682,
    "P": 30.973761998, "S": 32.06,
    "Li": 6.94, "Rb": 85.4678, "Pb": 207.2,
    "B": 10.81, "Ga": 69.723, "Ge": 72.630, "In": 114.818,
    "Cs": 132.90545196, "Cu": 63.546, "Sn": 118.710,
    "Zn": 65.38, "Cd": 112.411, "Bi": 208.9804,
    "As": 74.921595, "Se": 78.971, "Te": 127.60,
    "Mo": 95.95, "W": 183.84,
    "Ba": 137.327, "Sr": 87.62, "Zr": 91.224,
    "Sc": 44.955908, "Y": 88.90584, "U": 238.02891, "Th": 232.0377, "Hf": 178.49,
    "La": 138.90547, "Ce": 140.116, "Pr": 140.90766, "Nd": 144.242,
    "Sm": 150.36, "Eu": 151.964, "Gd": 157.25,
    "Tb": 158.92535, "Dy": 162.500, "Ho": 164.93033,
    "Er": 167.259, "Tm": 168.93422, "Yb": 173.045, "Lu": 174.9668,
    "Br": 79.904, "I": 126.90447, "Cl": 35.45, "F": 18.998403163,
    "Hg": 200.592, "Au": 196.96657, "Pd": 106.42,
    "Pt": 195.084,
    "Ru": 101.07,
    "Rh": 102.9055,
    "Ir": 192.217,
    "H": 1.008, "C": 12.011, "N": 14.007,
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
    group = re.fullmatch(r"\(([^()]+)\)(\d+)", formula)
    if group:
        subformula, multiplier = group.groups()
        return {
            element: number * int(multiplier)
            for element, number in _formula_atoms(subformula).items()
        }
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
    the count of parent-element atoms in the gas species. Other elements in
    mixed products are side reactants and do not carry the oxide's oxygen
    demand. For M_x O_y from parent M_a O_b, the pO2 exponent is
    (y - x*b/a) / 2. Parentless oxygen species use y / 2. Supply
    pO2_exponents to override or define custom tables.
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
        if parent:
            parent_atoms = _formula_atoms(parent)
            parent_elements = set(parent_atoms) - {"O"}
            parent_metals = sum(parent_atoms[element] for element in parent_elements)
            if parent_metals <= 0:
                raise ImccGasOxygenBalanceError(
                    f"parent oxide {parent!r} has no metal atoms for oxygen balance"
                )
            parent_ratio = parent_atoms.get("O", 0) / parent_metals
            parent_element_atoms = sum(
                atoms.get(element, 0) for element in parent_elements
            )
            parent_oxygen = parent_ratio * parent_element_atoms
            exponent = (oxygen_atoms - parent_oxygen) / 2.0
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
    imposed_parents: Sequence[str] | None = None,
) -> tuple[float, ImccGasResult, dict[str, object]]:
    """Solve pO2 that balances oxygen in the Knudsen effusion flux.

    Species flux is ``J_i = p_i A W / sqrt(2 pi M_i R T)``. The common
    aperture, Clausing, and temperature factors cancel, leaving
    ``F = sum_i (nO_i - (nuO/nuM)_parent nM_i) p_i / sqrt(M_i) = 0``.
    Existing equilibria give ``p_i ∝ pO2**k_i``. For condensed-parent
    reactions, the signed oxygen coefficient is ``-2 nO2/n_gas`` and
    ``k_i = -nO2/n_gas``. For the S2(g)-parent reactions, they are ``2 nO2``
    and ``k_i = nO2``. Halides use an explicit oxide oxygen demand; each
    parent oxide here contains one O atom, so demand is ``n_oxide`` and the
    product-side exponent is ``-n_O2 = -n_oxide/2``. Their signed oxygen coefficient and pressure
    exponent have matching signs, so
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

    ``imposed_parents`` names external gas reservoirs such as ``("N2",)`` or
    ``("C",)`` for a nitrogen or CO2 sweep. Their species remain in the
    returned pressure map but are excluded from the melt-derived oxygen
    balance. Names whose activity was not supplied are ignored. The default is
    no imposed reservoirs; a bare string is rejected because it is not a
    sequence of parent keys.

    Every supplied non-oxide gas parent and condensed parent contributes its
    oxygen-carrying species to the melt-derived oxygen demand unless named in
    ``imposed_parents``. Metal vapours carry no oxygen. Parent oxides omitted
    from an explicit ``parent_oxides`` sequence remain excluded.
    """
    if parent_oxides is None:
        parent_oxides = IMCC_PARENT_OXIDES
    if isinstance(imposed_parents, str):
        raise TypeError("imposed_parents must be a sequence of parent-key strings")
    imposed = frozenset(imposed_parents or ())
    nonoxide_parents = _nonoxide_parent_keys()
    unknown_imposed = imposed - nonoxide_parents
    if unknown_imposed:
        raise ValueError(
            "imposed_parents must name gas-parent activity keys; unknown: "
            f"{sorted(unknown_imposed)}"
        )
    T = float(T_K)
    if not math.isfinite(T) or T <= 0.0:
        raise ValueError(f"temperature must be finite and positive, got {T_K}")

    available_parents = list(parent_oxides)
    if isinstance(activities, Mapping):
        available_parents.extend(
            parent
            for parent in activities
            if parent in nonoxide_parents and parent not in available_parents
        )
    channels, _omitted_channels = _default_reactions(
        tuple(available_parents), datapack, T
    )
    balance_channels = tuple(
        (species, reaction)
        for species, reaction in channels
        # An imposed reservoir only removes a balance channel when its activity
        # was supplied and the channel was therefore constructed above.
        if reaction[0] not in imposed and _side_parent(species) not in imposed
    )
    species_data = oxygen_balance_species_metadata({
        species: parent for species, (parent, _n_gas, _n_O2) in balance_channels
    })
    for species, (parent, n_gas, n_O2) in balance_channels:
        data = species_data[species]
        side_reactant = _ALKALI_HALIDE_REACTION_OXIDES.get(species)
        if side_reactant:
            _oxide, n_oxide, n_O2 = side_reactant
            # Every side-table oxide here has one O atom per formula. Thus
            # demand is n_oxide O atoms: n_O2 = demand/2 and the product-gas
            # fugacity exponent is -n_O2. This covers MO (one O per M2+),
            # M2O (half O per M+), and dimers by the same scaled balance.
            data = data._replace(
                parent_oxygen_demand=n_oxide,
                pO2_exponent=-n_O2,
            )
            species_data[species] = data
            exponent = -n_O2
        elif parent in _GAS_PHASE_PARENT_ROWS:
            data = data._replace(parent_oxygen_demand=0.0)
            species_data[species] = data
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
