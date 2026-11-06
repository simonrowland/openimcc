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
from contextvars import ContextVar
import hashlib
import json
import math
import os
import re
import warnings
from collections.abc import Mapping as MappingABC
from dataclasses import InitVar, dataclass, field, replace
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from functools import lru_cache
from importlib import resources
from numbers import Number
from pathlib import Path
from types import MappingProxyType
from typing import Any, TYPE_CHECKING, Callable, Iterable, Literal, Mapping, NamedTuple, Sequence

import numpy as np
from scipy.optimize import brentq

from openimcc._constants import (
    ATOMIC_MASS_G_MOL as _ATOMIC_MASS_G_MOL,
    R_PUBLISHED_GAS_J_MOL_K as R_J_MOL_K,
)
from openimcc.kernel import ImccRefusal, _canonical_published_serialization
from openimcc.liquid_reference import (
    CertificationStatus,
    LiquidReferenceAuthority,
    liquid_reference_authority,
    liquid_reference_records,
)
from openimcc.receipt import CalculationReceipt, build_calculation_receipt
from openimcc.scalar_boundary import is_declared_real_scalar
from openimcc.gas_data import (
    ELEMENT_STATUS,
    IMCC_GAS_CHANNEL_SPECIES,
    IMCC_GAS_INCOMPLETE_PARENT_SPECIES,
    IMCC_GAS_NO_JANAF_ROWS,
    IMCC_GAS_UNAVAILABLE_SPECIES,
    IMCC_GAS_WORKBOOK_EXTRAPOLATION_LABELS,
    IMCC_GAS_WORKBOOK_IN_DOMAIN_SPECIES,
    IMCC_PARENT_OXIDES,
    IMCC_SF04_WORKBOOK_GRID_K,
    _ALKALI_HALIDE_REACTION_OXIDES,
    _CONDENSED_PARENT_PHASE_ROWS,
    _CONDENSED_PARENT_ROWS,
    _CONSTRUCTED_SAHA_C3_RESULTS,
    _CONSTRUCTED_SAHA_PARENT_SPECIES,
    _DATAPACK_OPTIONAL_CHANNELS,
    _GAS_ELECTRON_ATTACHMENTS,
    _GAS_IONIZATION_PAIRS,
    _GAS_PHASE_PARENT_ROWS,
    _GAS_PRESSURE_COLUMNS,
    _GAS_PROVENANCE_AUTHORITY,
    _HALIDE_REACTIONS,
    _HYDROGEN_HALIDE_PARENTS,
    _ION_PROVENANCE_CLASS,
    _ION_GAS_SPECIES,
    _LAM1987_SOURCE_TABLE_IDS,
    _LAM_PARENT_REF_AUTHORITIES,
    _MIXED_GAS_REACTION_SIDE_PARENTS,
    _NASA_ION_GAS_SPECIES,
    _OXIDE_PRESSURE_COLUMNS,
    _OXIDE_PROVENANCE_AUTHORITY,
    _OXIDE_SOURCE_TABLE_IDS,
    _PROVENANCE_AUTHORITY_RANK,
    _REACTION_PARENT_OXIDES,
    _SF04_REACTIONS,
    _anion,
    _c2_candidates,
    _c3,
    _c3_passes,
    _cation,
    _element,
    _fusion,
    _gas_table,
    _hashes,
    _ion_routes,
    _liquid_note,
    _max_ratio,
    _melt_temperature,
    _neutral,
    _note,
    _parent,
    _parent_table,
    _phase_marker,
    _products,
    _reference,
    _saha_note,
    _species,
    _status,
    _tsiv_table,
    _apply_constructed_saha_status,
)

if TYPE_CHECKING:
    import pandas as pd
    from openimcc.profiles import Profile


# --------------------------------------------------------------------------- #
# Physical constants and reference states
# --------------------------------------------------------------------------- #

# R_J_MOL_K retains the shared published-gas value for pinned table outputs.

_SUPERCOOLED_EXTENSION_REF_SUFFIX = "-SC-CP"
_SUPPRESS_TRIAL_RECEIPTS: ContextVar[bool] = ContextVar(
    "openimcc_suppress_trial_receipts", default=False
)


def _consume_trial_receipt_suppression() -> bool:
    """Consume one trial marker; the solver's finally block restores its token."""
    suppress = _SUPPRESS_TRIAL_RECEIPTS.get()
    if suppress:
        _SUPPRESS_TRIAL_RECEIPTS.set(False)
    return suppress


_SOURCE_CP_EXTENSION_RE = re.compile(r"-CP-EXT-(?P<limit>[0-9]+(?:\.[0-9]+)?)$")
_HIGH_TEMPERATURE_CANDIDATE_NOTICE_RE = re.compile(
    r"^(?P<source>.+-HT-CANDIDATE)\|tier=(?P<tier>[^;|]+);"
    r"dg=(?P<dg>[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?);"
    r"band=(?P<band>[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?);"
    r"limit=(?P<limit>[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)"
    r"(?:;basis=(?P<basis>[^;|]+))?$"
)
_HIGH_TEMPERATURE_LEGACY_NOTICE_RE = re.compile(
    r"^(?P<source>[A-Za-z0-9()._-]+)-HT-CANDIDATE$"
)
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


class ImccGasValidityMetadataError(ImccRefusal):
    """Raised when typed source-validity facts are incomplete or inconsistent."""

    code = "imcc_gas_validity_metadata_missing"


@dataclass(frozen=True)
class _HighTemperatureCandidateNotice:
    tier: str
    dg_kj_mol: float
    band_dex: float
    limit_K: float
    basis: str | None


@dataclass(frozen=True)
class _HighTemperatureLegacyNotice:
    source: str
    kind: Literal["extension", "bridge", "constant_cp"]


_HighTemperatureNotice = _HighTemperatureCandidateNotice | _HighTemperatureLegacyNotice


def _parse_high_temperature_candidate_notice(
    table_name: str, table_path: Path, species: str, source_ref: str
) -> _HighTemperatureNotice:
    match = _HIGH_TEMPERATURE_CANDIDATE_NOTICE_RE.fullmatch(source_ref)
    if match is not None:
        tier = match.group("tier")
        basis = match.group("basis")
        if tier in {"corroborated_calculation", "rough_extrapolated"} and basis in {
            None, "unavailable", "internal", "paired_sources"
        }:
            dg = float(match.group("dg"))
            band = float(match.group("band"))
            limit = float(match.group("limit"))
            if (
                all(math.isfinite(value) for value in (dg, band, limit))
                and dg >= 0.0
                and band >= 0.0
                and limit >= 3000.0
            ):
                return _HighTemperatureCandidateNotice(tier, dg, band, limit, basis)

    legacy_match = _HIGH_TEMPERATURE_LEGACY_NOTICE_RE.fullmatch(source_ref)
    if legacy_match is not None:
        source = legacy_match.group("source")
        kind = "extension"
        if source.endswith("-BRIDGE"):
            source = source.removesuffix("-BRIDGE")
            kind = "bridge"
        source = source.removeprefix("CEA-")
        if source == "FeO-constant-Cp":
            kind = "constant_cp"
        if source:
            return _HighTemperatureLegacyNotice(source, kind)

    raise ImccGasValidityMetadataError(
        f"{table_name} high-temperature candidate table {table_path.name} "
        f"has malformed notice metadata for {species!r}"
    )


def _high_temperature_candidate_notice_records(
    table: "pd.DataFrame", table_name: str, table_path: Path, *, phase: str
) -> dict[tuple[str, str, float], _HighTemperatureNotice]:
    # Candidate markers identify rows even when a table was filtered down to a
    # standalone interval. Also track continuations at the 3000 K source seam
    # after the marker is reworded, while leaving standalone published rows out.
    source_window_species = {
        str(species)
        for species, row in table.iterrows()
        if float(row["T_min"]) < 3000.0 and float(row["T_max"]) == 3000.0
    }
    candidate_species = source_window_species | {
        str(species)
        for species, row in table.iterrows()
        if float(row["T_min"]) == 3000.0
    }
    records = {}
    for species, row in table.iterrows():
        t_min = float(row["T_min"])
        source_ref = str(row["Ref"])
        has_candidate_marker = "HT-CANDIDATE" in source_ref
        is_boundary_continuation = (
            t_min >= 3000.0 and str(species) in candidate_species
        )
        if not has_candidate_marker and not is_boundary_continuation:
            continue
        key = phase, str(species), t_min
        records[key] = _parse_high_temperature_candidate_notice(
            table_name, table_path, str(species), source_ref
        )
    return records


class ImccGasTemperatureOutsideDomainError(ImccRefusal):
    """Raised when T falls outside the declared G(T) interval for a species."""

    code = "imcc_gas_T_outside_domain"


class ImccGasInvalidFugacityError(ImccRefusal):
    """Raised when the caller's bar-relative oxygen fugacity is invalid."""

    code = "imcc_gas_invalid_fO2"


class ImccGasUnknownActivityError(ImccRefusal):
    """Raised when an activity map names a parent this layer does not recognise."""

    code = "imcc_gas_unknown_activity"


class ImccGasUnrecognizedSourceError(ImccRefusal):
    """Raised when a condensate reference is not a packaged table id."""

    code = "imcc_gas_unrecognized_source"


class ImccGasEquilibriumRangeError(ImccRefusal):
    """Raised when an equilibrium exponential is not a finite float."""

    code = "imcc_gas_equilibrium_out_of_range"


class ImccGasOxygenBalanceError(ImccRefusal):
    """Raised when an oxygen-balance root cannot be certified or refined."""

    code = "imcc_gas_oxygen_balance_failed"


class ImccGasSaturationError(ImccRefusal):
    """Raised when a fixed-composition saturation root is not unique or bracketed."""

    code = "imcc_gas_saturation_failed"


class ImccGasIngestionWarning(UserWarning):
    """A CSV load discarded one or more columns with empty headers."""


class ImccGasSourceOverrideWarning(UserWarning):
    """``OPENIMCC_VAPOROCK_ROOT`` replaced a packaged gas table."""


# --------------------------------------------------------------------------- #
# Packaged data and explicit VapoRock override
# --------------------------------------------------------------------------- #


_VAPOROCK_ENV_VAR = "OPENIMCC_VAPOROCK_ROOT"
_GAS_TABLE_RELPATH = Path("src") / "vaporock" / "data" / "JANAF-vapor-data-full.csv"
_CONDENSATE_TABLE_RELPATH = Path("data") / "condensate-thermo-data.csv"
_PACKAGED_DATA_PACKAGE = "openimcc.data.gas"
_PACKAGED_GAS_NAME = "gas-shomate.csv"
_PACKAGED_CONDENSATE_NAME = "condensate.csv"
_SOURCE_VALIDITY_TEXT_COLUMNS = (
    "validity_kind", "continuation_kind", "cp_extension", "construction_method",
    "table_id", "source_label", "source_liquid_marker", "supercooled_continuation",
)
_SOURCE_VALIDITY_NUMERIC_COLUMNS = (
    "source_fit_min_K", "source_fit_max_K", "source_limit_K",
    "fusion_temperature_K", "supercooled_segment_min_K",
    "supercooled_segment_max_K", "construction_tier", "glass_join_temperature_K",
)
_GAS_TABLE_SCHEMA = frozenset(
    {
        "species_name", "state", "T_interval", "cation", "cat_num", "oxy_num",
        "T_min", "T_max", "A", "B", "C", "D", "E", "F", "G", "H", "Ref",
        *_SOURCE_VALIDITY_TEXT_COLUMNS, *_SOURCE_VALIDITY_NUMERIC_COLUMNS,
    }
)
_CONDENSATE_TABLE_SCHEMA = frozenset(
    {
        "species_name", "state", "cation", "cat_num", "oxy_num", "T_min", "T_max",
        "dH298_R", "dG_A", "dG_B", "dG_C", "dG_D", "dG_E", "Ref",
        *_SOURCE_VALIDITY_TEXT_COLUMNS, *_SOURCE_VALIDITY_NUMERIC_COLUMNS,
    }
)
_GAS_NUMERIC_COLUMNS = (
    "T_interval", "cat_num", "oxy_num", "T_min", "T_max", *tuple("ABCDEFGH"),
    *_SOURCE_VALIDITY_NUMERIC_COLUMNS,
)
_CONDENSATE_NUMERIC_COLUMNS = (
    "cat_num", "oxy_num", "T_min", "T_max", "dH298_R", "dG_A", "dG_B", "dG_C",
    "dG_D", "dG_E", *_SOURCE_VALIDITY_NUMERIC_COLUMNS,
)
_GAS_TEXT_COLUMNS = ("state", "cation", "Ref", *_SOURCE_VALIDITY_TEXT_COLUMNS)
_CONDENSATE_TEXT_COLUMNS = ("state", "cation", "Ref", *_SOURCE_VALIDITY_TEXT_COLUMNS)


class ImccGasDataUnavailableError(ImccRefusal):
    """The gas thermodynamic tables have not been located."""

    code = "imcc_gas_data_unavailable"


@dataclass(frozen=True)
class PGasPressureNotice:
    """Typed warning that dilute fixed-composition P-gas pressures exceed 1 bar."""

    code: str
    message: str
    species: tuple[str, ...]
    total_pressure_bar: float


@dataclass(frozen=True)
class LiquidReferenceNotice:
    """Typed evidence notice for a calculated, unmeasured liquid reference."""

    code: str
    message: str
    status: CertificationStatus


def _gas_result_notices(
    values: Mapping[str, float],
    domain_flags: Mapping[str, str | None],
    liquid_reference_status: Mapping[str, CertificationStatus],
) -> tuple[PGasPressureNotice | LiquidReferenceNotice, ...]:
    p_species = tuple(
        name for name in ("P", "P2", "P4", "PO", "PO2", "P4O6", "P4O10", "PS")
        if name in values
    )
    code = "p_gas_exceeds_one_bar_complete_volatilisation"
    affected = tuple(
        name for name in p_species if code in (domain_flags.get(name) or "")
    )
    notices = []
    if affected:
        total = math.fsum(values[name] for name in p_species)
        notices.append(PGasPressureNotice(
            code,
            "P would leave essentially completely; the fixed-composition dilute "
            "equilibrium is not physical above 1 bar.",
            affected,
            total,
        ))
    seen_reference_species = set()
    for status in liquid_reference_status.values():
        if status.species in seen_reference_species:
            continue
        seen_reference_species.add(status.species)
        if status.authority is LiquidReferenceAuthority.CORROBORATED_CALCULATION:
            notices.append(LiquidReferenceNotice(
                "liquid-reference-corroborated-calculation",
                "Liquid reference is corroborated by independent calculation, not measured.",
                status,
            ))
        elif (
            status.species == "K2O(l)"
            and status.authority is LiquidReferenceAuthority.EXTRAPOLATED
            and status.certified_T_K == (1765, 2473)
            and status.band_value is not None
        ):
            notices.append(LiquidReferenceNotice(
                "liquid-reference-source-construction-envelope",
                "K2O liquid reference remains extrapolated; compilation bridge "
                "from NASA Glenn/Gurvich and LH84. Reason: source/construction "
                "spread is asymmetric, envelope_dex=-0.15/+0.33 log10 pK; "
                "band_value=0.33 dex is the symmetric outward bound, not a "
                "confidence interval or validated accuracy bound.",
                status,
            ))
        if status.in_composition_scope is False:
            evidence = status.evidence_ref[0]
            key, (low, high) = next(iter(status.composition_scope.items()))
            notices.append(LiquidReferenceNotice(
                "liquid-reference-outside-evidence-composition-scope",
                f"{status.species} composition is outside the evidence range "
                f"{key} {low:g}–{high:g}; {evidence.author} ({evidence.year}): "
                f"{evidence.basis}.",
                status,
            ))
    return tuple(notices)


@dataclass(frozen=True)
class ImccGasResult(MappingABC[str, float]):
    """Immutable gas pressures and channel metadata, including default omissions."""

    _values: Mapping[str, float]
    unit: str
    domain_flags: Mapping[str, str | None]
    provenance_class: Mapping[str, str]
    omitted_channels: Mapping[str, str] = field(default_factory=dict)
    liquid_reference_status: Mapping[str, CertificationStatus] = field(default_factory=dict)
    _profile_identity: InitVar[tuple[str | None, str, str] | None] = None
    receipt: CalculationReceipt | None = field(
        default=None, compare=False, hash=False
    )

    def __post_init__(
        self, _profile_identity: tuple[str | None, str, str] | None
    ) -> None:
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
        object.__setattr__(
            self, "liquid_reference_status",
            MappingProxyType(dict(self.liquid_reference_status)),
        )
        if _profile_identity is not None:
            object.__setattr__(self, "_loaded_profile_identity", _profile_identity)

    @property
    def profile(self) -> str | None:
        identity = getattr(self, "_loaded_profile_identity", None)
        return identity[0] if identity is not None else None

    @property
    def gas_table_digest(self) -> str | None:
        identity = getattr(self, "_loaded_profile_identity", None)
        return identity[1] if identity is not None else None

    @property
    def condensate_table_digest(self) -> str | None:
        identity = getattr(self, "_loaded_profile_identity", None)
        return identity[2] if identity is not None else None

    def __getitem__(self, species: str) -> float:
        return self._values[species]

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    @property
    def notices(self) -> tuple[PGasPressureNotice | LiquidReferenceNotice, ...]:
        """Typed notices from per-channel domain flags and liquid-reference status."""
        return _gas_result_notices(
            self._values, self.domain_flags, self.liquid_reference_status
        )


@dataclass(frozen=True)
class ImccGasSaturationResult:
    """Fixed-composition equilibrium vapor saturated over the supplied melt."""

    pO2_bar: float
    species_pressures_bar: Mapping[str, float]
    channel_pressures_bar: Mapping[str, float]
    P_sat_bar: float
    mole_fractions: Mapping[str, float]
    mode: str
    diagnostics: Mapping[str, object]
    domain_flags: Mapping[str, str | None]
    _profile_identity: InitVar[tuple[str | None, str, str] | None] = None
    receipt: CalculationReceipt | None = field(
        default=None, compare=False, hash=False
    )
    liquid_reference_status: Mapping[str, CertificationStatus] = field(default_factory=dict)

    def __post_init__(
        self, _profile_identity: tuple[str | None, str, str] | None
    ) -> None:
        for name in ("species_pressures_bar", "channel_pressures_bar", "mole_fractions", "diagnostics", "domain_flags"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        object.__setattr__(
            self, "liquid_reference_status",
            MappingProxyType(dict(self.liquid_reference_status)),
        )
        if _profile_identity is not None:
            object.__setattr__(self, "_loaded_profile_identity", _profile_identity)

    @property
    def profile(self) -> str | None:
        identity = getattr(self, "_loaded_profile_identity", None)
        return identity[0] if identity is not None else None

    @property
    def gas_table_digest(self) -> str | None:
        identity = getattr(self, "_loaded_profile_identity", None)
        return identity[1] if identity is not None else None

    @property
    def condensate_table_digest(self) -> str | None:
        identity = getattr(self, "_loaded_profile_identity", None)
        return identity[2] if identity is not None else None

    @property
    def notices(self) -> tuple[PGasPressureNotice | LiquidReferenceNotice, ...]:
        """Typed notices for the saturated vapor's pressures and liquid references."""
        return _gas_result_notices(
            self.species_pressures_bar, self.domain_flags, self.liquid_reference_status
        )


@dataclass(frozen=True)
class GasChannelMetadata:
    """Stoichiometric and charge metadata for one selected gas channel."""

    species: str
    oxygen_atoms: float
    parent_oxygen_demand: float
    charge: int
    available: bool
    neutral_pO2_exponent: float | None


@dataclass(frozen=True)
class WeightedOxygenBalanceResult:
    """Root and signed/applied flux evidence for a frozen transport snapshot."""

    pO2_bar: float
    pressures_bar: Mapping[str, float]
    signed_fluxes_mol_m2_s: Mapping[str, float]
    residual_mol_m2_s: float
    absolute_weighted_transfer_scale_mol_m2_s: float
    equilibrium_weighted_transfer_scale_mol_m2_s: float
    bulk_weighted_transfer_scale_mol_m2_s: float
    gross_weighted_transfer_scale_mol_m2_s: float
    forward_weighted_scale_mol_m2_s: float
    backward_weighted_scale_mol_m2_s: float
    applied_residual_mol_m2_s: float
    applied_flux_closes: bool
    applied_flux_closure_state: str
    root_bracket_log10_pO2: tuple[float, float]
    root_count: int
    root_count_evidence: str
    monotonicity: str
    model_evaluations: int
    # Caller-named plane at which bulk_pressures_bar apply (for example a
    # well-mixed headspace or a boundary-layer edge). Echoed, never interpreted.
    bulk_plane: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "pressures_bar", MappingProxyType(dict(self.pressures_bar)))
        object.__setattr__(self, "signed_fluxes_mol_m2_s", MappingProxyType(dict(self.signed_fluxes_mol_m2_s)))


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




# The simulator imports this private helper as a cross-repo API; pin its name
# and call signature in tests before changing it.
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
        row_authority = _LAM_PARENT_REF_AUTHORITIES.get(str(oxide_row["Ref"]))
        if row_authority is not None:
            _, _, _, _, lam_parent_rows = _packaged_pressure_signatures()
            if (
                _pressure_row_signature(oxide_row, _OXIDE_PRESSURE_COLUMNS)
                in lam_parent_rows
            ):
                return min(
                    (str(provenance["gas_authority"]), row_authority),
                    key=_PROVENANCE_AUTHORITY_RANK.__getitem__,
                )
    return str(provenance["authority"])




def _active_ion_pairs(
    datapack: ImccGasDatapack,
    pairs: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    """Return ion channels backed by rows in this gas table."""
    gas_rows = datapack.gas_df.index
    return tuple(
        (neutral, ion)
        for neutral, ion in pairs
        if f"{ion}(g)" in gas_rows
    )



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
    validity_columns = (*_SOURCE_VALIDITY_TEXT_COLUMNS, *_SOURCE_VALIDITY_NUMERIC_COLUMNS)
    missing_validity = sorted(set(validity_columns) - set(actual_columns))
    if missing_validity:
        raise ImccGasValidityMetadataError(
            f"{table_name} database {table_path} lacks typed source-validity columns "
            f"{missing_validity}"
        )
    for column in validity_columns:
        values = table[column].array
        if any(pd.isna(value) for value in values):
            raise ImccGasValidityMetadataError(
                f"{table_name} database {table_path} has missing typed source-validity "
                f"values in {column}"
            )
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


def _gas_table_is_locked(frame: "pd.DataFrame") -> bool:
    state = getattr(frame, "__dict__", None)
    return bool(state and state.get("_locked"))


def _immutable_gas_frame_type():
    """DataFrame subclass that refuses structural edits once locked.

    Built on first use so importing this module does not import pandas.
    """
    existing = getattr(_immutable_gas_frame_type, "cls", None)
    if existing is not None:
        return existing
    pd = _pandas()
    index_property = pd.DataFrame.index
    columns_property = pd.DataFrame.columns

    class _ImmutableGasFrame(pd.DataFrame):
        # Slices and copies stay this type but do not inherit the lock: the
        # flag lives on the stored instance only.
        @property
        def _constructor(self):
            return _ImmutableGasFrame

        def __setattr__(self, key, value):
            if key == "_locked":
                if _gas_table_is_locked(self):
                    raise TypeError("immutable gas table")
                object.__setattr__(self, key, value)
                return
            # An owned-block write replaces nothing; a column or index
            # replacement installs a new manager. Both have to fail closed.
            if key == "_mgr" and _gas_table_is_locked(self):
                raise TypeError("immutable gas table")
            return super().__setattr__(key, value)

        @property
        def index(self):
            return index_property.__get__(self, type(self))

        @index.setter
        def index(self, value):
            if _gas_table_is_locked(self):
                raise TypeError("immutable gas table")
            index_property.__set__(self, value)

        @property
        def columns(self):
            return columns_property.__get__(self, type(self))

        @columns.setter
        def columns(self, value):
            if _gas_table_is_locked(self):
                raise TypeError("immutable gas table")
            columns_property.__set__(self, value)

        def __setitem__(self, key, value):
            if _gas_table_is_locked(self):
                raise TypeError("immutable gas table")
            return super().__setitem__(key, value)

        def __delitem__(self, key):
            if _gas_table_is_locked(self):
                raise TypeError("immutable gas table")
            return super().__delitem__(key)

        def insert(self, *args, **kwargs):
            if _gas_table_is_locked(self):
                raise TypeError("immutable gas table")
            return super().insert(*args, **kwargs)

        def drop(self, *args, **kwargs):
            if kwargs.get("inplace") and _gas_table_is_locked(self):
                raise TypeError("immutable gas table")
            return super().drop(*args, **kwargs)

        def rename(self, *args, **kwargs):
            if kwargs.get("inplace") and _gas_table_is_locked(self):
                raise TypeError("immutable gas table")
            return super().rename(*args, **kwargs)

    _ImmutableGasFrame.__module__ = __name__
    _ImmutableGasFrame.__qualname__ = "_ImmutableGasFrame"
    _immutable_gas_frame_type.cls = _ImmutableGasFrame
    return _ImmutableGasFrame


def _freeze_gas_table_blocks(table: "pd.DataFrame") -> None:
    """Mark stored numpy blocks read-only.

    Copy-on-write updates an owned block in place and does not replace the
    manager, so refusing ``_mgr`` assignment leaves ``loc`` cell writes live.
    """
    for block in table._mgr.blocks:
        values = block.values
        if not isinstance(values, np.ndarray):
            raise TypeError(
                "immutable gas table cannot freeze "
                f"{type(values).__name__} values"
            )
        values.flags.writeable = False


def _lock_gas_table(table: "pd.DataFrame") -> "pd.DataFrame":
    """Return a deep copy that cannot diverge from the digest of its cells."""
    frame = _immutable_gas_frame_type()(table).copy(deep=True)
    _freeze_gas_table_blocks(frame)
    frame._locked = True
    return frame


@dataclass(frozen=True)
class ImccGasDatapack:
    """Loaded JANAF + condensate tables and their parsed-content identities.

    ``gas_path`` and ``oxide_path`` are diagnostic source locations; content
    identity is exposed by the table digest properties. Both frames are
    canonicalized once at construction and then locked. In-place edits raise,
    so a stored digest cannot describe different coefficients from the ones
    evaluations read. Use :func:`dataclasses.replace` to create a validated
    pack with changed tables.
    """

    gas_df: pd.DataFrame
    oxide_df: pd.DataFrame
    gas_path: Path
    oxide_path: Path
    _gas_digest: str = field(init=False, repr=False, compare=False)
    _condensate_digest: str = field(init=False, repr=False, compare=False)
    _high_temperature_candidate_notices: Mapping[
        tuple[str, str, float], _HighTemperatureNotice
    ] = field(init=False, repr=False, compare=False)

    def __init__(
        self,
        gas_df: pd.DataFrame,
        oxide_df: pd.DataFrame,
        gas_path: Path,
        oxide_path: Path,
    ) -> None:
        # A second call would write the new tables before validation. If that
        # validation fails, the stored frame no longer matches its digest.
        if hasattr(self, "_gas_digest"):
            raise TypeError("ImccGasDatapack cannot be initialized more than once")
        object.__setattr__(self, "_gas_digest", "")
        object.__setattr__(self, "_condensate_digest", "")
        object.__setattr__(self, "gas_df", gas_df)
        object.__setattr__(self, "oxide_df", oxide_df)
        object.__setattr__(self, "gas_path", gas_path)
        object.__setattr__(self, "oxide_path", oxide_path)
        self.__post_init__()

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
        for species, row in gas_df.iterrows():
            _source_validity_for_row(str(species), row, phase="g")
        for species, row in oxide_df.iterrows():
            _source_validity_for_row(str(species), row, phase="l")
        candidate_notices = {
            **_high_temperature_candidate_notice_records(
                gas_df, "JANAF gas", self.gas_path, phase="g"
            ),
            **_high_temperature_candidate_notice_records(
                oxide_df, "condensate", self.oxide_path, phase="l"
            ),
        }
        gas_df = _lock_gas_table(gas_df)
        oxide_df = _lock_gas_table(oxide_df)
        object.__setattr__(
            self,
            "_high_temperature_candidate_notices",
            MappingProxyType(candidate_notices),
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
    :mod:`importlib.resources`.  Setting ``OPENIMCC_VAPOROCK_ROOT`` selects
    the matching VapoRock file for any path that was not passed explicitly,
    and warns once with :class:`ImccGasSourceOverrideWarning` because those
    files replace the packaged tables.  Passing both ``gas_path`` and
    ``oxide_path`` uses those files and does not warn.  Both files are
    read-only.
    """
    pd = _pandas()

    # Resolved lazily: an unconfigured install must fail with a typed refusal
    # at CALL time, not at import time, or `import openimcc.gas` breaks for
    # everyone who only wanted the species constants.  The override notice is
    # emitted before either file is opened so a missing checkout is not silent.
    override = os.environ.get(_VAPOROCK_ENV_VAR)
    if override and (gas_path is None or oxide_path is None):
        warnings.warn(
            ImccGasSourceOverrideWarning(
                f"{_VAPOROCK_ENV_VAR}={override} replaces the packaged gas tables"
            ),
            stacklevel=2,
        )
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


def _select_gas_datapack(
    datapack: ImccGasDatapack | None, profile: str | Profile | None
) -> tuple[ImccGasDatapack, str | None]:
    if profile is None:
        if datapack is None:
            raise TypeError("datapack is required unless profile is specified")
        return datapack, None
    from openimcc.profiles import ProfileConflictError, _resolve_profile

    if datapack is not None:
        raise ProfileConflictError(
            "profile cannot be combined with an explicit gas datapack"
        )
    selected = _resolve_profile(profile)
    return selected.gas_datapack, selected.name


def _gas_result_identity(
    datapack: ImccGasDatapack, profile_name: str | None
) -> tuple[str | None, str, str] | None:
    gas_digest = getattr(datapack, "gas_table_digest", None)
    condensate_digest = getattr(datapack, "condensate_table_digest", None)
    if not isinstance(gas_digest, str) or not isinstance(condensate_digest, str):
        return None
    return profile_name, gas_digest, condensate_digest


# --------------------------------------------------------------------------- #
# G(T) evaluation
# --------------------------------------------------------------------------- #


def _thermochemistry_value(value: float, row_name: str, temperature_K: float) -> float:
    try:
        finite = math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        finite = False
    if not finite:
        raise ImccGasEquilibriumRangeError(
            f"thermochemistry for {row_name} is outside the finite range at T={temperature_K} K"
        )
    return value


def _janaf_gibbs(T: float, row: Mapping[str, Any] | pd.Series) -> float:
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
    row_name = str(row.get("species_name") or "thermochemistry row")
    try:
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
        gibbs = dH * 1000.0 - T * S
    except (ArithmeticError, ValueError) as exc:
        raise ImccGasEquilibriumRangeError(
            f"thermochemistry for {row_name} is outside the finite range at T={T} K; "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return _thermochemistry_value(gibbs, row_name, T)


def _lamor_gibbs(T: float, row: Mapping[str, Any] | pd.Series) -> float:
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
    row_name = str(row.get("species_name") or "condensate")

    try:
        G_coefs = np.array([A, B, C, D, E])
        G_scale = np.array([1.0, 1e3, 1e6, 1e9, 1e12])
        G_poly_coefs = (G_coefs / G_scale)[::-1]
        gibbs = -R_J_MOL_K * T * np.polyval(G_poly_coefs, T) + R_J_MOL_K * dH298 * 1000.0
    except (ArithmeticError, ValueError) as exc:
        raise ImccGasEquilibriumRangeError(
            f"thermochemistry for {row_name} is outside the finite range at T={T} K"
        ) from exc
    return _thermochemistry_value(gibbs, row_name, T)


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


def _high_temperature_candidate_notice_flag(
    notice: _HighTemperatureNotice, T: float
) -> str:
    if isinstance(notice, _HighTemperatureLegacyNotice):
        if notice.kind == "bridge":
            return (
                "rough high-temperature extension above 3000 K: Cp continued "
                f"from the reference row to its CEA bridge point, then from NASA CEA {notice.source}; "
                "anchored at 3000 K; not validated"
            )
        if notice.kind == "constant_cp":
            return (
                "rough high-temperature extension above 3000 K: FeO(l) Cp held "
                "constant from the current reference row; anchored at 3000 K; "
                "not validated"
            )
        return (
            "rough high-temperature extension above 3000 K: Cp shape from NASA "
            f"CEA {notice.source} anchored at 3000 K; not validated"
        )
    tier = notice.tier
    dg = notice.dg_kj_mol
    band = notice.band_dex
    limit = notice.limit_K
    if notice.basis == "unavailable":
        return (
            "high-temperature extension, rough: no paired-source "
            "comparison is available above 3000 K"
        )
    if notice.basis == "internal":
        return (
            "high-temperature extension, rough: internal diagnostic "
            f"maximum spread is {dg:g} kJ/mol (maximum pressure band "
            f"±{band:g} dex) over the reported comparison range; no "
            "independent source comparison is available"
        )
    if tier == "corroborated_calculation" and T <= limit:
        return (
            "high-temperature extension, corroborated by calculation "
            f"over 3000-{limit:g} K: JANAF and Gurvich Cp shapes agree "
            f"within at most {dg:g} kJ/mol (at most ±{band:g} dex) "
            f"over that range; anchored at 3000 K"
        )
    if tier == "corroborated_calculation":
        return (
            f"high-temperature extension, rough beyond {limit:g} K: "
            f"maximum corroborated spread over 3000-{limit:g} K was "
            f"{dg:g} kJ/mol (maximum pressure band ±{band:g} dex)"
        )
    return (
        f"high-temperature extension, rough above {limit:g} K: sources "
        f"diverge by up to {dg:g} kJ/mol (maximum pressure band "
        f"±{band:g} dex) over the reported comparison range"
    )


def _source_model_flag(
    row_name: str,
    T: float,
    row: pd.Series,
    *,
    phase: str,
    candidate_notices: Mapping[
        tuple[str, str, float], _HighTemperatureNotice
    ] | None = None,
) -> str | None:
    """Name a selected row's coverage construction or upper-tail continuation."""
    source_ref = str(row.get("Ref", ""))
    if T > 3000.0:
        notice_key = (phase, row_name, float(row["T_min"]))
        notice = (
            candidate_notices.get(notice_key)
            if candidate_notices is not None
            else None
        )
        if notice is not None:
            return _high_temperature_candidate_notice_flag(notice, T)
    validity = _source_validity_for_row(row_name, row, phase=phase)
    if row_name == "FeO(l)" and phase == "l":
        fit_range, fit_row = _selected_redox_feo_fit()
        # Ref is provenance; the coefficients identify whether this fit applies.
        if all(float(row[column]) == value for column, value in fit_row):
            status = liquid_reference_authority(row_name, T)
            if status.authority is LiquidReferenceAuthority.EXTRAPOLATED:
                low, high = status.certified_T_K or fit_range
                return (
                    f"{row_name} uses the D&G/Masson correction extrapolated "
                    f"outside its {low:g}-{high:g} K fit range "
                    f"(Ref={source_ref})"
                )
    if validity is None:
        return None
    method = str(validity["method"])
    fit_range = validity.get("source_fit_range_K") or validity["T_range_K"]
    if method == "constructed_from_vapor_pressure_and_condensed_function":
        # Premise: Smith et al. (1952), pp. 4964-4966, measured Re2O7 vapor
        # pressures; the fitted maximum absolute residual is 0.131480 dex.
        # From ΔG = -RT ln p, |δG| = R T ln(10) |δlog10 p|. R is J mol^-1 K^-1
        # and T is K, so /1000 converts J mol^-1 to kJ mol^-1; at 550 K this
        # is about 1.4 kJ mol^-1.
        uncertainty_kj = (
            R_J_MOL_K * math.log(10.0) * T * 0.131479893726766 / 1000.0
        )
        measured_range = (
            f"{float(fit_range[0]):g}-{float(fit_range[1]):g} K"
        )
        return (
            "constructed Re2O7(g) from Smith 1952 measured vapor pressure and "
            f"B677 condensed functions over {measured_range}; maximum log-pressure "
            "fit scatter propagated as "
            f"±{uncertainty_kj:.1f} kJ/mol at {T:g} K"
        )
    if method == "constructed_from_equilibrium_and_condensed_function":
        # Premise: Battles et al. (1968), Eq. (10), gives log10(p_atm)=A-B/T
        # with coefficient uncertainties 0.145 dex and 143 K. Propagating
        # |δlog10 p| <= 0.145 + 143/T through ΔG = -RT ln p gives
        # |δG| = R ln(10) (0.145 T + 143 K). With R in J mol^-1 K^-1 and T
        # in K the result is J mol^-1; at 1000 K it is about 5.5 kJ mol^-1.
        uncertainty_kj = R_J_MOL_K * math.log(10.0) * (0.145 * T + 143.0) / 1000.0
        return (
            "constructed Re2O7(g) from Battles vapor-pressure equilibrium measured "
            "at 961-1087 K; B677 Re2O7(l) Cp held above its 700 K limit for the "
            "condensed-function reconstruction; propagated regression coefficient "
            f"band ±{uncertainty_kj:.1f} kJ/mol at {T:g} K"
        )
    if method == "constructed_from_equilibrium_and_condensed_function_with_held_cp_continuation":
        # Use the same Battles et al. (1968), Eq. (10), coefficient bound as
        # above; the held-Cp continuation carries that Gibbs-energy bound.
        uncertainty_kj = R_J_MOL_K * math.log(10.0) * (0.145 * T + 143.0) / 1000.0
        return (
            "constructed Re2O7(g) from Battles vapor-pressure equilibrium measured "
            "at 961-1087 K; held-Cp runtime continuation from 1087 K to 3000 K; "
            "B677 Re2O7(l) Cp held above its 700 K limit for the condensed-function "
            "reconstruction; propagated "
            f"vapor-pressure regression bound ±{uncertainty_kj:.1f} kJ/mol at {T:g} K"
        )
    continuation_limit = validity.get("source_limit_K")
    if (
        validity.get("continuation_kind") != "none"
        and continuation_limit is not None
        and T >= float(continuation_limit)
    ):
        continuation_limit = float(continuation_limit)
        if validity["continuation_kind"] == "liquid_metal":
            return (
                f"{row_name} uses a flagged liquid-metal continuation above the "
                f"{continuation_limit:g} K source limit"
            )
        return (
            f"{row_name} uses a flagged {phase}-phase continuation above the "
            f"{continuation_limit:g} K source limit"
        )
    fusion_temperature = validity.get("fusion_temperature_K")
    if (
        phase == "l"
        and method == "integrated_phase_chain_and_supercooled_liquid"
        and fusion_temperature is not None
        and T < float(fusion_temperature)
    ):
        return (
            f"{row_name} uses the printed constant-Cp liquid branch as a "
            f"supercooled continuation below {float(fusion_temperature):g} K"
        )
    supercooled_segment = validity.get("supercooled_segment_K")
    if (
        phase == "l"
        and method == "fitted"
        and supercooled_segment is not None
    ):
        return (
            f"{row_name} uses the TSIV liquid function as a supercooled branch "
            f"below {float(supercooled_segment[1]):g} K"
        )
    source_table = str(validity.get("display_source_table") or validity.get("table_id") or "")
    if phase == "l" and validity.get("liquid_from_solid"):
        tier = validity.get("tier")
        if tier is None:
            raise ValueError(f"{row_name} liquid_from_solid validity has no estimate tier")
        if not validity.get("cp_extension"):
            if row_name == "K2O(l)" and 1033.0 <= T <= 1093.0:
                return (
                    f"{row_name} uses a tier-{tier} liquid_from_solid construction "
                    f"from {source_table}; a competing 1190 ± 50 K fusion estimate "
                    "leaves the stable phase uncertain in this temperature band"
                )
            return (
                f"{row_name} uses a tier-{tier} liquid_from_solid construction "
                f"from {source_table}"
            )
        limit = float(validity["source_limit_K"])
        if source_table == "TSIV-1603":
            return (
                f"{row_name} uses a tier-{tier} liquid_from_solid construction from "
                f"{source_table}; the central liquid uses the printed 2250 K "
                "node, and the quadratic crystal-Cp extension is its uncertainty bound"
            )
        return (
            f"{row_name} uses a tier-{tier} liquid_from_solid construction from "
            f"{source_table}; its crystal Cp was extended above {limit:g} K "
            "to the estimated fusion temperature"
        )
    source_marker = validity.get("source_marker")
    if (
        phase == "l"
        and source_marker is not None
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
            + (
                "; the stable deposit in the 1033–1093 K process band is crystalline"
                if row_name == "Na2O(l)" and 1033.0 <= T <= 1093.0
                else ""
            )
        )
    if validity.get("cp_extension") and continuation_limit is not None and T > float(continuation_limit):
        limit = float(continuation_limit)
        return (
            f"{row_name} uses a labelled constant-Cp continuation above "
            f"{limit:g} K from {source_table}"
        )
    return None


def _liquid_reference_flags(
    species: str,
    T: float,
    source_flags: list[str],
    *,
    selected_row: pd.Series,
    channel: str,
    composition_wt_pct: Mapping[str, float] | None,
    statuses: dict[str, CertificationStatus],
) -> list[str]:
    """Apply authority only when it was assessed for the selected source row."""
    composition = None
    if composition_wt_pct is not None:
        composition = {
            f"{oxide}_wt_pct": value
            for oxide, value in composition_wt_pct.items()
        }
    validity = _source_validity_for_row(species, selected_row, phase="l")
    selected_source_id = (
        str(validity["table_id"])
        if validity is not None and validity.get("table_id")
        else str(selected_row.get("Ref", ""))
    )
    status = liquid_reference_authority(
        species,
        T,
        composition,
        selected_source_id=selected_source_id,
    )
    if status is None:
        return source_flags
    statuses[channel] = status
    scope_unresolved = bool(status.composition_scope) and not status.scope_applied
    if (
        status.authority in {
            LiquidReferenceAuthority.CERTIFIED,
            LiquidReferenceAuthority.CORROBORATED_CALCULATION,
        }
        and not scope_unresolved
    ):
        return []
    band_text = (
        f"; band ±{status.band_value:g} {status.band_unit}"
        if status.band_value is not None else ""
    )
    if source_flags:
        return [flag + band_text for flag in source_flags]
    reason = status.reason
    prefix = f"{species} liquid reference is extrapolated"
    if scope_unresolved:
        reason = "composition scope was not evaluated"
        prefix = f"{species} liquid-reference scope is unresolved"
    return [
        f"{prefix}: {reason}{band_text}"
    ]


def _source_validity_for_row(
    row_name: str, row: pd.Series, *, phase: str
) -> dict[str, Any] | None:
    required = (*_SOURCE_VALIDITY_TEXT_COLUMNS, *_SOURCE_VALIDITY_NUMERIC_COLUMNS)
    if any(column not in row.index for column in required):
        raise ImccGasValidityMetadataError(
            f"{row_name} {phase}-phase row lacks typed source-validity columns"
        )
    kind = str(row["validity_kind"])
    boolean_values = {
        column: str(row[column]).casefold()
        for column in ("cp_extension", "source_liquid_marker", "supercooled_continuation")
    }
    if kind not in {"none", "coverage"} or any(
        value not in {"true", "false"} for value in boolean_values.values()
    ):
        raise ImccGasValidityMetadataError(
            f"{row_name} {phase}-phase row has invalid typed source-validity values"
        )
    try:
        values = {column: float(row[column]) for column in _SOURCE_VALIDITY_NUMERIC_COLUMNS}
    except (TypeError, ValueError) as exc:
        raise ImccGasValidityMetadataError(
            f"{row_name} {phase}-phase row has incomplete typed source-validity values"
        ) from exc
    if not all(math.isfinite(value) for value in values.values()):
        raise ImccGasValidityMetadataError(
            f"{row_name} {phase}-phase row has non-finite typed source-validity values"
        )
    method = str(row["construction_method"])
    table_id = str(row["table_id"])
    source_label = str(row["source_label"])
    continuation_kind = str(row["continuation_kind"])
    has_typed_facts = (
        method != "none"
        or table_id != "none"
        or source_label != "none"
        or continuation_kind != "none"
        or any(value == "true" for value in boolean_values.values())
        or any(value != 0.0 for value in values.values())
    )
    expected_kind = "coverage" if has_typed_facts else "none"
    if kind != expected_kind:
        raise ImccGasValidityMetadataError(
            f"{row_name} {phase}-phase validity_kind {kind!r} contradicts "
            f"its typed source-validity facts (expected {expected_kind!r})"
        )
    if kind == "none":
        return None
    if (
        method in {"", "none"}
        or table_id in {"", "none"}
        or source_label in {"", "none"}
        or continuation_kind not in {"none", "phase", "liquid_metal"}
        or values["source_fit_min_K"] > values["source_fit_max_K"]
        or not values["construction_tier"].is_integer()
    ):
        raise ImccGasValidityMetadataError(
            f"{row_name} {phase}-phase row has incomplete typed source-validity facts"
        )
    if boolean_values["source_liquid_marker"] == "true" and not values["fusion_temperature_K"]:
        raise ImccGasValidityMetadataError(
            f"{row_name} {phase}-phase source marker lacks its fusion temperature"
        )
    if (
        (continuation_kind != "none" or boolean_values["cp_extension"] == "true")
        and not values["source_limit_K"]
    ):
        raise ImccGasValidityMetadataError(
            f"{row_name} {phase}-phase continuation lacks its source limit"
        )
    if method in {"liquid_from_solid", "liquid_from_solid_construction"} and not values["construction_tier"]:
        raise ImccGasValidityMetadataError(
            f"{row_name} liquid_from_solid validity lacks its estimate tier"
        )
    return {
        "method": method,
        "source_fit_range_K": [values["source_fit_min_K"], values["source_fit_max_K"]],
        "source_limit_K": values["source_limit_K"] or None,
        "continuation_kind": continuation_kind,
        "fusion_temperature_K": values["fusion_temperature_K"] or None,
        "supercooled_segment_K": (
            [values["supercooled_segment_min_K"], values["supercooled_segment_max_K"]]
            if values["supercooled_segment_max_K"] else None
        ),
        "cp_extension": boolean_values["cp_extension"] == "true",
        "liquid_from_solid": method in {
            "liquid_from_solid", "liquid_from_solid_construction"
        },
        "tier": int(values["construction_tier"]) if values["construction_tier"] else None,
        "table_id": table_id,
        "display_source_table": source_label,
        "source_marker": (
            {
                "table_id": table_id,
                "fusion_temperature_K": values["fusion_temperature_K"],
                "glass_join_temperature_K": values["glass_join_temperature_K"] or None,
            }
            if boolean_values["source_liquid_marker"] == "true" else None
        ),
        "supercooled_continuation": boolean_values["supercooled_continuation"] == "true",
    }


@lru_cache(maxsize=1)
def _selected_redox_feo_fit() -> tuple[
    tuple[float, float], tuple[tuple[str, float], ...]
]:
    """Return the selected FeO(l) redox row and its measured fit interval."""
    from openimcc.redox_pack import load_redox_pack

    record = load_redox_pack().content["thermo_species"]["FeO(l)"]
    fit_range = tuple(
        float(value) for value in record["fit_residual"]["temperature_range"]
    )
    fit_columns = ("dH298_R", "dG_A", "dG_B", "dG_C", "dG_D", "dG_E")
    fit_row = tuple((key, float(record["row"][key])) for key in fit_columns)
    if len(fit_range) != 2:
        raise ValueError("selected redox FeO(l) fit metadata is incomplete")
    return (fit_range[0], fit_range[1]), fit_row


def _derived_source_supercooled_liquid_markers(
    datapack: ImccGasDatapack,
) -> dict[str, dict[str, object]]:
    """Summarize source-liquid marker facts already stored on condensate rows."""
    derived = {}
    for species, rows in datapack.oxide_df.groupby(level=0, sort=False):
        species_rows = [rows] if getattr(rows, "ndim", 1) == 1 else [
            row for _, row in rows.iterrows()
        ]
        for row in species_rows:
            validity = _source_validity_for_row(str(species), row, phase="l")
            marker = validity.get("source_marker") if validity else None
            if marker is not None and float(row["T_min"]) < marker["fusion_temperature_K"]:
                derived[str(species)] = marker
                break
    return derived


def _supercooled_liquid_from_solid_flag(
    row_name: str, T: float, row: pd.Series
) -> str | None:
    """Flag a constructed liquid below its measured or selected fusion point."""
    validity = _source_validity_for_row(row_name, row, phase="l")
    fusion_temperature = validity.get("fusion_temperature_K") if validity else None
    if (
        not validity
        or not validity.get("liquid_from_solid")
        or fusion_temperature is None
        or T >= float(fusion_temperature)
    ):
        return None
    return (
        f"{row_name} uses a supercooled liquid_from_solid branch below "
        f"the {float(fusion_temperature):g} K fusion temperature"
    )


def _source_table_identifier(source_ref: str) -> str:
    """Remove runtime construction suffixes while retaining the source ID."""
    base_ref = _SOURCE_CP_EXTENSION_RE.sub("", source_ref)
    match = _LIQUID_FROM_SOLID_REF_RE.fullmatch(base_ref)
    if match is not None:
        return match["source"]
    base = _SOURCE_CP_EXTENSION_RE.sub("", source_ref)
    return base.removesuffix("-LFS").removesuffix(_SUPERCOOLED_EXTENSION_REF_SUFFIX)


# Continuation tails stored on a packaged Ref. They are not a new table.
_REF_CONTINUATION_RE = re.compile(
    r"-(?:CONT|LOW|SC)-[0-9]+(?:\.[0-9]+)?$"
)


def _ref_continues_table(source_ref: str, table_id: str) -> bool:
    base = _REF_CONTINUATION_RE.sub("", source_ref)
    return base == table_id and base != source_ref


@lru_cache(maxsize=1)
def _packaged_condensate_ref_pairs() -> frozenset[tuple[str, str]]:
    """Shipped ``(species_name, Ref)`` pairs. Not the environment override."""
    resource = _packaged_resource(_PACKAGED_CONDENSATE_NAME)
    with resource.open("r", encoding="utf-8-sig", newline="") as handle:
        return frozenset(
            (row["species_name"], row["Ref"]) for row in csv.DictReader(handle)
        )


def _other_phase_carries_mapped_table(
    oxide_df: pd.DataFrame, species: str, phase: str, mapped: str
) -> bool:
    for other in ("l", "cr"):
        if other == phase:
            continue
        name = f"{species}({other})"
        if name not in oxide_df.index:
            continue
        refs = oxide_df.loc[name, "Ref"]
        values = [refs] if isinstance(refs, str) else list(refs)
        for value in values:
            text = str(value)
            if text == mapped or _ref_continues_table(text, mapped):
                return True
    return False


def _condensate_source_table_id(
    species: str, phase: str, source_ref: str, oxide_df: pd.DataFrame
) -> str | None:
    """Cite a condensate row the earlier Ref arms did not name.

    ``_OXIDE_SOURCE_TABLE_IDS`` is one id per species. A Ref that is that id,
    a continuation of it, or a packaged alias (``JANAF``, ``LFS[...]``) keeps
    the map. A packaged Ref that belongs to the other phase is cited as
    itself. Any other Ref on a mapped species is refused, so a caller-supplied
    row cannot inherit the packaged table id.
    """
    # The opt-in high-cristobalite table carries its JANAF phase reference here.
    if species == "SiO2" and phase == "cr" and source_ref == "O-035":
        return source_ref
    mapped = _OXIDE_SOURCE_TABLE_IDS.get(species)
    if source_ref == "JANAF" and species == "FeO":
        return "Fe-019"
    if source_ref == "JANAF":
        return mapped
    if source_ref.startswith("LFS["):
        return mapped or _source_table_identifier(source_ref)
    if mapped is not None and (
        source_ref == mapped or _ref_continues_table(source_ref, mapped)
    ):
        return mapped
    row_name = f"{species}({phase})"
    packaged = (row_name, source_ref) in _packaged_condensate_ref_pairs()
    if not packaged and mapped is not None:
        raise ImccGasUnrecognizedSourceError(
            f"unrecognised condensate reference {source_ref!r} for {row_name}"
        )
    if mapped is None:
        return source_ref or None
    if _other_phase_carries_mapped_table(oxide_df, species, phase, mapped):
        return source_ref
    return mapped


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


def _equilibrium_range_message(exponent: float) -> str:
    try:
        shown: object = float(exponent)
    except (TypeError, ValueError, OverflowError):
        shown = exponent
    return f"equilibrium constant is not a finite float for exp({shown!r})"


def _finite_equilibrium_exp(exponent: float) -> np.floating:
    """Return ``exp(exponent)`` for one dimensionless equilibrium argument.

    Call sites already use ``np.exp``. Keeping that function leaves in-range
    values bit-identical. Overflow and invalid results become a typed refusal.
    Underflow to 0 is the K→0 limit and is kept.
    """
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        try:
            value = np.exp(exponent)
        except FloatingPointError as exc:
            raise ImccGasEquilibriumRangeError(
                _equilibrium_range_message(exponent)
            ) from exc
    if not math.isfinite(float(value)):
        raise ImccGasEquilibriumRangeError(_equilibrium_range_message(exponent))
    return value


def _log_mass_action_power(value: float, exponent: float) -> float:
    if exponent == 0.0:
        return 0.0
    if value == 0.0:
        return -math.inf if exponent > 0.0 else math.inf
    return exponent * math.log(float(value))


def _finite_mass_action_pressure(
    direct_calculation: Callable[[], float],
    log_pressure: float,
    species: str,
) -> float:
    direct_error: Exception | None = None
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        try:
            pressure = float(direct_calculation())
        except (OverflowError, ZeroDivisionError) as exc:
            direct_error = exc
        else:
            if math.isfinite(pressure) and pressure > 0.0:
                return pressure
    try:
        pressure = math.exp(log_pressure)
    except (OverflowError, ValueError) as exc:
        cause = direct_error if direct_error is not None else exc
        raise ImccGasEquilibriumRangeError(
            f"mass-action pressure for {species} is outside the finite float range"
        ) from cause
    if not math.isfinite(pressure):
        raise ImccGasEquilibriumRangeError(
            f"mass-action pressure for {species} is outside the finite float range"
        ) from direct_error
    return pressure


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
                        _source_model_flag(
                            row_name,
                            T,
                            row,
                            phase="g",
                            candidate_notices=datapack._high_temperature_candidate_notices,
                        ),
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

    for neutral, ion in _active_ion_pairs(datapack, _GAS_IONIZATION_PAIRS):
        if neutral not in values:
            continue
        G_neutral, neutral_row, neutral_flag = gibbs(neutral)
        G_ion, ion_row, ion_flag = gibbs(ion)
        # Premise: each ideal gas uses p°=1 bar and
        # Z_i°=(kBT/p°)*(2*pi*m_i*kBT/h²)^(3/2)*Q_i*exp(-E_i/kBT).
        # For M -> M+ + e-, Kp=p(M+)p(e-)/(p(M)p°)=Z_+°*Z_e°/Z_M°;
        # neglecting the tiny M+/M mass correction gives
        # Kp=(kBT/p°)*2*(2*pi*m_e*kBT/h²)^(3/2)*(Q_M+/Q_M)*exp(-IE/kBT).
        # The 2 is electron spin. Equivalently,
        # G(M+)+G(e-)-G(M)=IE+RT*ln(Q_M/Q_M+)+G_e,physical, where
        # Z_e°=(kBT/p°)*2*(2*pi*m_e*kBT/h²)^(3/2),
        # G_e,physical=-RT*ln(Z_e°), H_e,physical=5/2 RT, and
        # S_e,physical=R*(ln(Z_e°)+5/2). D-020's shipped electron row uses
        # H_e(298.15 K)=0, so G_e,JANAF=G_e,physical-5/2*R*298.15 K.
        # Constructed NIST-ASD cations add +5/2*R*298.15 K to G(M+); the
        # existing D-020 row stays unchanged. Their closure reproduces the
        # dimensionless 1-bar Saha Kp, including its T^(5/2) translation.
        # Unit check: Kp and ΔG/(RT) are dimensionless. Sanity: the Na+ pilot
        # differs from JANAF by at most 128.537 J/mol and 0.3005% in ion share
        # over 1200-3000 K.
        dG = G_ion + G_electron - G_neutral
        equilibrium = float(_finite_equilibrium_exp(-dG / (R_J_MOL_K * T)))
        positive_sum += equilibrium * values[neutral]
        positive.append(
            (
                neutral,
                ion,
                equilibrium,
                (neutral_flag, ion_flag, electron_flag, domain_flags.get(neutral)),
                str(ion_row.get("Ref", "")),
                _gas_rows_match_packaged(
                    datapack, (neutral_row, ion_row, electron_row)
                ),
            )
        )

    for neutral, ion in _active_ion_pairs(datapack, _GAS_ELECTRON_ATTACHMENTS):
        if neutral not in values:
            continue
        G_neutral, neutral_row, neutral_flag = gibbs(neutral)
        G_ion, ion_row, ion_flag = gibbs(ion)
        # Premise: A + e- = A- uses the same ideal-gas standard state.
        # Algebra: K_A = exp[-(G(A-) - G(A) - G(e-))/(R*T)]. Its partition
        # form is (p°/kBT)*(Z-/2Z0)*(h²/(2*pi*me*kBT))^(3/2)*exp(EA/kBT):
        # p°/kBT is m^-3 and the electron translational factor m^3. It is the
        # inverse cation Saha form, so detailed balance gives K_attach*K_ion=1.
        dG = G_ion - G_neutral - G_electron
        equilibrium = float(_finite_equilibrium_exp(-dG / (R_J_MOL_K * T)))
        negative_sum += equilibrium * values[neutral]
        negative.append(
            (
                neutral,
                ion,
                equilibrium,
                (neutral_flag, ion_flag, electron_flag, domain_flags.get(neutral)),
                str(ion_row.get("Ref", "")),
                _gas_rows_match_packaged(
                    datapack, (neutral_row, ion_row, electron_row)
                ),
            )
        )

    electron_pressure = (
        math.sqrt(positive_sum / (1.0 + negative_sum))
        if positive_sum > 0.0
        else 0.0
    )
    values["e-"] = electron_pressure
    electron_flags = []
    for dependency in (*positive, *negative):
        for composite_flag in dependency[3]:
            if composite_flag:
                for flag in composite_flag.split("; "):
                    if flag and flag not in electron_flags:
                        electron_flags.append(flag)
    domain_flags["e-"] = "; ".join(electron_flags) or None
    provenance_class["e-"] = _ION_PROVENANCE_CLASS
    if not _gas_rows_match_packaged(datapack, (electron_row,)):
        provenance_class["e-"] = _EXTERNAL_DATAPACK_CLASS

    for neutral, ion, equilibrium, flags, source_id, rows_match in positive:
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
        if not rows_match:
            provenance_class[ion] = _EXTERNAL_DATAPACK_CLASS

    for neutral, ion, equilibrium, flags, source_id, rows_match in negative:
        values[ion] = equilibrium * values[neutral] * electron_pressure
        domain_flags[ion] = "; ".join(flag for flag in flags if flag) or None
        provenance_class[ion] = (
            "nasa_glenn_fitted_ionisation"
            if source_id.startswith("NG-")
            else _ION_PROVENANCE_CLASS
        )
        provenance_class[ion] = _GAS_PROVENANCE_AUTHORITY[ion]
        if not rows_match:
            provenance_class[ion] = _EXTERNAL_DATAPACK_CLASS


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
    The default FeO(l) row uses the D&G/Masson primary-measurement reduction;
    JANAF Fe-019 remains available through its explicit legacy condensate
    table. Na2O(l) is checked against Na-013. Al2O3(l) is a LAM-source
    exception compared with Al-100.
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
        try:
            t4 = t**4
        except OverflowError as exc:
            raise ImccGasEquilibriumRangeError(
                f"thermochemistry for {row_name} is outside the finite range at T={T_K} K"
            ) from exc
        A, B, C, D, E = (float(row[key]) for key in "ABCDE")
        enthalpy_primitive = (
            A * t + B * t**2 / 2.0 + C * t**3 / 3.0
            + D * t4 / 4.0 - E / t
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
        try:
            t4 = t**4
        except OverflowError as exc:
            raise ImccGasEquilibriumRangeError(
                f"thermochemistry for {row_name} is outside the finite range at T={T_K} K"
            ) from exc
        h_increment = R_J_MOL_K * t**2 * (
            B + 2.0 * C * t + 3.0 * D * t**2 + 4.0 * E * t**3
        )
        h_app = R_J_MOL_K * float(row["dH298_R"]) + h_increment
        entropy = R_J_MOL_K * (
            A + 2.0 * B * t + 3.0 * C * t**2 + 4.0 * D * t**3 + 5.0 * E * t4
        )
        cp = R_J_MOL_K * (
            2.0 * B * t + 6.0 * C * t**2 + 12.0 * D * t**3 + 20.0 * E * t4
        )
        apparent_g = _lamor_gibbs(T_K, row)
        interval = None

    for quantity, value in (
        ("Cp", cp), ("entropy", entropy), ("enthalpy", h_app),
        ("Gibbs energy", apparent_g),
    ):
        _thermochemistry_value(value, f"{row_name} {quantity}", T_K)

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
    elif phase != "g" and source_ref.endswith("-LFS"):
        # The liquid is constructed from the crystal table. Stripping "-LFS"
        # leaves a name that is not that table; the species map has the id.
        mapped = _OXIDE_SOURCE_TABLE_IDS.get(species)
        source_table_id = (
            mapped if mapped is not None else _source_table_identifier(source_ref)
        )
    elif phase != "g" and (
        "-LFS" in source_ref or _SOURCE_CP_EXTENSION_RE.search(source_ref)
    ):
        source_table_id = _source_table_identifier(source_ref)
    elif phase != "g":
        source_table_id = _condensate_source_table_id(
            species, phase, source_ref, active_pack.oxide_df
        )
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



_EXTERNAL_DATAPACK_CLASS = "external_datapack"


def _pressure_row_signature(
    row: pd.Series, columns: tuple[str, ...]
) -> tuple[object, ...]:
    """Species, interval start, Gibbs coefficients, and Ref.

    ``T_max`` and the gas ``H`` column do not enter G, so a change there is
    not a different pressure.
    """
    return (
        str(row.name),
        float(row["T_min"]),
        *(float(row[column]) for column in columns),
        str(row["Ref"]),
    )


def _frame_pressure_signatures(
    frame: pd.DataFrame, columns: tuple[str, ...]
) -> frozenset[tuple[object, ...]]:
    names = frame.index.to_numpy()
    starts = frame["T_min"].to_numpy()
    refs = frame["Ref"].to_numpy()
    numeric = tuple(frame[column].to_numpy() for column in columns)
    return frozenset(
        (
            str(names[position]),
            float(starts[position]),
            *(float(values[position]) for values in numeric),
            str(refs[position]),
        )
        for position in range(len(frame))
    )


def _read_packaged_table(name: str) -> tuple[pd.DataFrame, Path]:
    pd = _pandas()
    resource = _packaged_resource(name)
    display_path = _packaged_database_path(name)
    with resource.open("rb") as handle:
        frame = pd.read_csv(handle)
    frame, _notice = _drop_unnamed_columns(frame, resource, display_path)
    return frame.set_index("species_name"), display_path


@lru_cache(maxsize=1)
def _packaged_pressure_signatures() -> tuple[
    str,
    str,
    frozenset[tuple[object, ...]],
    frozenset[tuple[object, ...]],
    frozenset[tuple[object, ...]],
]:
    """Digests and Gibbs-cell signatures of the packaged tables.

    Read through the importlib resources, not ``load_gas_datapack``. An
    environment override must not become the reference the digests name. The
    final signature set contains the shipped LAM parent source rows.
    """
    pd = _pandas()
    gas_df, gas_path = _read_packaged_table(_PACKAGED_GAS_NAME)
    oxide_df, oxide_path = _read_packaged_table(_PACKAGED_CONDENSATE_NAME)
    pack = ImccGasDatapack(
        gas_df=gas_df,
        oxide_df=oxide_df,
        gas_path=gas_path,
        oxide_path=oxide_path,
    )
    lam_parent_resource = resources.files("openimcc").joinpath(
        "data/packs/sf04-published/condensate.csv"
    )
    with lam_parent_resource.open("rb") as handle:
        lam_parent_df = pd.read_csv(handle).set_index("species_name")
    lam_parent_rows = lam_parent_df.loc[
        lam_parent_df["Ref"].isin(_LAM_PARENT_REF_AUTHORITIES)
    ]
    return (
        pack.gas_table_digest,
        pack.condensate_table_digest,
        _frame_pressure_signatures(pack.gas_df, _GAS_PRESSURE_COLUMNS),
        _frame_pressure_signatures(pack.oxide_df, _OXIDE_PRESSURE_COLUMNS),
        _frame_pressure_signatures(lam_parent_rows, _OXIDE_PRESSURE_COLUMNS),
    )


def _channel_provenance(
    species: str,
    oxide_row: pd.Series | None,
    datapack: ImccGasDatapack,
    rows: list[tuple[str, pd.Series]],
) -> str:
    """Label a channel from the rows that enter its pressure.

    The static species map is the packaged authority. It stays for the
    packaged tables themselves, and for a pack whose used Gibbs cells are
    those packaged cells. Dropping an unused interval therefore does not
    relabel the channels that remain. A LAM parent keeps its transcribed
    class only when its coefficients match the shipped source row; a gas row
    beside it still has to match.
    """
    declared = _result_provenance_authority(species, oxide_row)
    gas_digest, oxide_digest, gas_rows, oxide_rows, lam_parent_rows = (
        _packaged_pressure_signatures()
    )
    if (
        datapack.gas_table_digest == gas_digest
        and datapack.condensate_table_digest == oxide_digest
    ):
        return declared
    for kind, row in rows:
        if kind == "gas":
            if (
                _pressure_row_signature(row, _GAS_PRESSURE_COLUMNS)
                not in gas_rows
            ):
                return _EXTERNAL_DATAPACK_CLASS
            continue
        row_signature = _pressure_row_signature(row, _OXIDE_PRESSURE_COLUMNS)
        if row_signature not in oxide_rows and row_signature not in lam_parent_rows:
            return _EXTERNAL_DATAPACK_CLASS
    return declared


def _gas_rows_match_packaged(
    datapack: ImccGasDatapack, rows: tuple[pd.Series, ...]
) -> bool:
    """True when ion Gibbs rows are the packaged gas rows."""
    gas_digest, oxide_digest, gas_rows, _oxide_rows, _lam_parent_rows = (
        _packaged_pressure_signatures()
    )
    if (
        datapack.gas_table_digest == gas_digest
        and datapack.condensate_table_digest == oxide_digest
    ):
        return True
    return all(
        _pressure_row_signature(row, _GAS_PRESSURE_COLUMNS) in gas_rows
        for row in rows
    )


def evaluate_gas(
    activities: Mapping[str, float] | Sequence[float] | np.ndarray,
    T_K: float,
    fO2: float,
    datapack: ImccGasDatapack | None = None,
    parent_oxides: Sequence[str] | None = None,
    allow_extrapolation: bool = True,
    gas_species: Sequence[str] | str | None = None,
    include_ions: bool = False,
    *,
    composition_wt_pct: Mapping[str, float] | None = None,
    profile: str | Profile | None = None,
) -> ImccGasResult:
    """Compute equilibrium partial pressures for the SF04 retained gas set.

    Parameters
    ----------
    activities:
        Parent activities. Either a dict keyed by parent name, or a vector
        aligned with ``parent_oxides`` (default IMCC order). A dict key that
        is not a recognised parent or gas species raises
        ``ImccGasUnknownActivityError``. A recognised species that is not a
        parent is ignored. With a mapping, a recognised parent key outside an
        explicitly supplied ``parent_oxides`` raises ``ValueError``; add the
        key to ``parent_oxides`` to evaluate it. If ``parent_oxides`` is
        omitted, recognized parents in the mapping are included as before. A
        selected parent omitted from the dict has zero activity.
        Oxide parents use
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
    composition_wt_pct:
        Optional melt composition keyed by oxide, in weight percent. This is
        used to resolve composition-scoped liquid-reference authority such as
        the K2O(l) measured-pressure window.

    Returns
    -------
    ImccGasResult
        A read-only mapping of partial pressures in bar.  ``unit`` is
        ``"bar"``; ``domain_flags`` names extrapolated source rows; and
        ``provenance_class`` carries the least-authoritative class returned by
        ``gas_species_provenance`` when the rows that enter the pressure are
        the packaged rows. A Gibbs cell that is not that packaged cell is
        ``external_datapack``, except a LAM1984 or LAM1987 parent row whose
        Gibbs coefficients match the shipped transcription.

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
    suppress_receipt = _consume_trial_receipt_suppression()
    requested_parent_oxides = parent_oxides
    datapack, profile_name = _select_gas_datapack(datapack, profile)
    parent_oxides_explicit = parent_oxides is not None
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
        ion_names = set(_ION_GAS_SPECIES) | {"e-"}
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

    if not is_declared_real_scalar(T_K, allow_numeric_str=True):
        raise ValueError(
            f"temperature must be a real number, got {T_K!r}"
        )
    try:
        T = float(T_K)
    except OverflowError as exc:
        raise ValueError("temperature must be finite and positive") from exc
    if not is_declared_real_scalar(fO2, allow_numeric_str=True):
        raise ImccGasInvalidFugacityError(
            "fO2 must be finite and positive p_O2/p° (bar-relative, not "
            f"log10 fO2); got {fO2}"
        )
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

    def activity_as_float(name: str, value: Any) -> float:
        if not is_declared_real_scalar(value, allow_numeric_str=True):
            raise ValueError(
                f"activity for {name!r} must be a real number"
            )
        try:
            return float(value)
        except OverflowError as exc:
            raise ValueError(
                f"activity for {name!r} is too large to represent as a float"
            ) from exc

    if isinstance(activities, Mapping):
        # An omitted recognised parent stays zero. A key that is neither a
        # parent nor a gas species is a misspelling, not an omission. The
        # species S is not the parent S2, so that key must not be refused.
        unknown_activities = sorted(
            name
            for name in activities
            if name not in _REACTION_PARENT_OXIDES
            and name not in _SF04_REACTIONS
        )
        if unknown_activities:
            raise ImccGasUnknownActivityError(
                "activity key(s) not a recognised gas parent or species: "
                f"{unknown_activities}"
            )
        if parent_oxides_explicit:
            extra_parents = sorted(
                name
                for name in activities
                if name in _REACTION_PARENT_OXIDES
                and name not in parent_oxides
            )
            if extra_parents:
                raise ValueError(
                    "activity mapping contains recognised parent key(s) "
                    f"{extra_parents} outside parent_oxides; add them to "
                    "parent_oxides"
                )
        act = {
            name: activity_as_float(name, activities.get(name, 0.0))
            for name in parent_oxides
        }
        if not parent_oxides_explicit:
            act.update(
                {
                    name: activity_as_float(name, activities[name])
                    for name in activities
                    if (
                        name in _REACTION_PARENT_OXIDES
                        and name not in act
                    )
                }
            )
    else:
        raw_arr = np.asarray(activities, dtype=object)
        if raw_arr.shape[0] != len(parent_oxides):
            raise ValueError(
                f"activities vector length {raw_arr.shape[0]} does not match "
                f"{len(parent_oxides)} parent oxides"
            )
        arr = np.asarray(
            [
                activity_as_float(name, value)
                for name, value in zip(parent_oxides, raw_arr, strict=True)
            ],
            dtype=float,
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
    liquid_statuses: dict[str, CertificationStatus] = {}
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
        flag = _source_model_flag(
            gas_species,
            T,
            gas_row,
            phase="g",
            candidate_notices=datapack._high_temperature_candidate_notices,
        )
        if flag is not None:
            flags.append(flag)
        # Rows whose G enters this channel. O2 is appended only when its
        # coefficient in ΔG is not zero.
        pressure_rows: list[tuple[str, pd.Series]] = [("gas", gas_row)]

        if O2_row is not None and (
            n_O2 != 0.0
            or gas_name in _HALIDE_REACTIONS
        ):
            flag = _outside_interval_flag("O2(g)", T, O2_row)
            if flag is not None:
                flags.append(flag)
            flag = _source_model_flag(
                "O2(g)",
                T,
                O2_row,
                phase="g",
                candidate_notices=datapack._high_temperature_candidate_notices,
            )
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
                pressure_rows.append(("gas", parent_row))
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
                pressure_rows.append(("oxide", oxide_row))
                flag = _outside_interval_flag(oxide_name, T, oxide_row)
            a_oxide = act[oxide]
            parent_present = a_oxide != 0.0
            if parent_present and flag is not None:
                flags.append(flag)
            oxide_flag_start = len(flags)
            validity = (
                _source_validity_for_row(
                    oxide_name, oxide_row, phase=("s" if str(oxide_row["state"]) == "s" else "l")
                )
                if oxide_row is not None else None
            )
            if (
                parent_present
                and oxide_row is not None
                and validity
                and validity.get("supercooled_continuation")
            ):
                source_table = str(validity.get("table_id") or "")
                source_name = "NASA Glenn/CEA" if source_table.startswith("NG-") else "JANAF"
                source_name = (
                    "NASA Glenn card"
                    if source_table.startswith("NG-")
                    else "JANAF"
                )
                flags.append(
                    f"{oxide_name} uses a labelled constant-Cp supercooled-liquid "
                    f"continuation from {source_name} {source_table}"
                    + (
                        "; this is a hypothetical reference, not a prediction of a "
                        "liquid condensate"
                        if oxide_name in {"CaO(l)", "MgO(l)", "SiO2(l)", "Al2O3(l)"}
                        else ""
                    )
                )
            elif parent_present and oxide_row is not None:
                parent_phase = "s" if str(oxide_row["state"]) == "s" else "l"
                flag = _source_model_flag(
                    oxide_name,
                    T,
                    oxide_row,
                    phase=parent_phase,
                    candidate_notices=datapack._high_temperature_candidate_notices,
                )
                if flag is not None:
                    flags.append(flag)
                if parent_phase == "l":
                    flag = _supercooled_liquid_from_solid_flag(
                        oxide_name, T, oxide_row
                    )
                    if flag is not None:
                        flags.append(flag)
            if parent_present and oxide_row is not None and oxide_name in {
                record.species for record in liquid_reference_records()
            }:
                flags[oxide_flag_start:] = _liquid_reference_flags(
                    oxide_name,
                    T,
                    flags[oxide_flag_start:],
                    selected_row=oxide_row,
                    channel=gas_name,
                    composition_wt_pct=composition_wt_pct,
                    statuses=liquid_statuses,
                )
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
            pressure_rows.append(("gas", side_row))
            flag = _outside_interval_flag(side_row_name, T, side_row)
            if flag is not None:
                flags.append(flag)
            dG = G_gas - n_gas * G_oxide - 0.5 * G_side
            Kp = _finite_equilibrium_exp(-dG / (R_J_MOL_K * T))
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
            pressure_rows.append(("oxide", side_row))
            if side_n_O2 != 0.0 and O2_row is not None and (
                n_O2 != 0.0 or gas_name in _HALIDE_REACTIONS
            ):
                pressure_rows.append(("gas", O2_row))
            if act[side_oxide] != 0.0:
                flag = _outside_interval_flag(side_name, T, side_row)
                if flag is not None:
                    flags.append(flag)
                flag = _source_model_flag(
                    side_name,
                    T,
                    side_row,
                    phase="l",
                    candidate_notices=datapack._high_temperature_candidate_notices,
                )
                source_flags = [flag] if flag is not None else []
                if side_name in {record.species for record in liquid_reference_records()}:
                    source_flags = _liquid_reference_flags(
                        side_name,
                        T,
                        source_flags,
                        selected_row=side_row,
                        channel=gas_name,
                        composition_wt_pct=composition_wt_pct,
                        statuses=liquid_statuses,
                    )
                flags.extend(source_flags)
            # The side table carries complete oxygen stoichiometry for new
            # halides whose reaction tuple has a zero O2 slot. Existing Na/K
            # rows keep using their tuple coefficient and unchanged arithmetic.
            side_G_O2 = (
                G_O2 if n_O2 != 0.0 or gas_name in _HALIDE_REACTIONS else 0.0
            )
            dG = G_gas + side_n_O2 * side_G_O2 - n_oxide * G_side_oxide - n_gas * G_oxide
            Kp = _finite_equilibrium_exp(-dG / (R_J_MOL_K * T))
            log_pressure = (
                _log_mass_action_power(float(Kp), 1.0)
                + _log_mass_action_power(act[side_oxide], n_oxide)
                + _log_mass_action_power(a_oxide, n_gas)
                - _log_mass_action_power(p_O2, side_n_O2)
            )
            p_gas = _finite_mass_action_pressure(
                lambda: (
                    Kp
                    * act[side_oxide] ** n_oxide
                    * a_oxide**n_gas
                    / p_O2**side_n_O2
                ),
                log_pressure,
                gas_name,
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
            pressure_rows.append(("gas", side_row))
            if n_O2 != 0.0 and O2_row is not None:
                pressure_rows.append(("gas", O2_row))
            flag = _outside_interval_flag(side_name, T, side_row)
            if flag is not None:
                flags.append(flag)
            # P2O5(l) + S2(g) = 2 PS(g) + 5/2 O2(g). The tuple coefficients
            # are per mole of P2O5; solve mass action for the two PS products.
            dG = n_gas * G_gas + n_O2 * G_O2 - G_oxide - n_side * G_side
            Kp = _finite_equilibrium_exp(-dG / (R_J_MOL_K * T))
            log_pressure = (
                _log_mass_action_power(float(Kp), 1.0)
                + _log_mass_action_power(a_oxide, 1.0)
                + _log_mass_action_power(act[side_parent], n_side)
                - _log_mass_action_power(p_O2, n_O2)
            ) / n_gas
            p_gas = _finite_mass_action_pressure(
                lambda: (
                    Kp
                    * a_oxide
                    * act[side_parent] ** n_side
                    / p_O2**n_O2
                ) ** (1.0 / n_gas),
                log_pressure,
                gas_name,
            )
        elif not hydrogen_halide_parent and oxide in _GAS_PHASE_PARENT_ROWS:
            # Sulfur tuple: n_S2 S2(g) + n_O2 O2(g) = gas(g).
            # Kp = p_gas / (a_S2**n_S2 * fO2**n_O2), with ΔG° formed from
            # the same balanced reaction; a_S2 is caller-supplied f(S2)/p°.
            if n_O2 != 0.0 and O2_row is not None:
                pressure_rows.append(("gas", O2_row))
            dG = G_gas - n_gas * G_oxide - n_O2 * G_O2
            Kp = _finite_equilibrium_exp(-dG / (R_J_MOL_K * T))
            log_pressure = (
                _log_mass_action_power(float(Kp), 1.0)
                + _log_mass_action_power(a_oxide, n_gas)
                + _log_mass_action_power(p_O2, n_O2)
            )
            p_gas = _finite_mass_action_pressure(
                lambda: Kp * (a_oxide**n_gas) * (p_O2**n_O2),
                log_pressure,
                gas_name,
            )
        elif not hydrogen_halide_parent:
            # Condensed-parent reaction (1): oxide(l) = n_gas*gas(g) +
            # n_O2*O2(g). Keep its established arithmetic path unchanged.
            if n_O2 != 0.0 and O2_row is not None:
                pressure_rows.append(("gas", O2_row))
            dG = n_gas * G_gas + n_O2 * G_O2 - G_oxide
            Kp = _finite_equilibrium_exp(-dG / (R_J_MOL_K * T))
            log_pressure = (
                _log_mass_action_power(float(Kp), 1.0)
                + _log_mass_action_power(a_oxide, 1.0)
                - _log_mass_action_power(p_O2, n_O2)
            ) / n_gas
            p_gas = _finite_mass_action_pressure(
                lambda: (Kp * a_oxide / (p_O2**n_O2)) ** (1.0 / n_gas),
                log_pressure,
                gas_name,
            )
        result[gas_name] = float(p_gas)
        domain_flags[gas_name] = "; ".join(flags) or None
        provenance_class[gas_name] = _channel_provenance(
            gas_name, oxide_row, datapack, pressure_rows
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
                *_active_ion_pairs(datapack, _GAS_IONIZATION_PAIRS),
                *_active_ion_pairs(datapack, _GAS_ELECTRON_ATTACHMENTS),
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
                    *_active_ion_pairs(datapack, _GAS_IONIZATION_PAIRS),
                    *_active_ion_pairs(datapack, _GAS_ELECTRON_ATTACHMENTS),
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
            closure_notice = (
                f"summed neutral pressure {closure_neutral_pressure:.6g} bar "
                "exceeds 1 bar; "
                "ideal-gas ion closure is not credible at this pressure"
            )
            domain_flags = {
                name: "; ".join(filter(None, (flag, closure_notice)))
                for name, flag in domain_flags.items()
                if name.endswith(("+", "-"))
            }

    p_species = tuple(
        name for name in ("P", "P2", "P4", "PO", "PO2", "P4O6", "P4O10", "PS")
        if name in result
    )
    p_total = math.fsum(result[name] for name in p_species)
    exceeded = tuple(name for name in p_species if result[name] > 1.0)
    if exceeded or p_total > 1.0:
        notice = PGasPressureNotice(
            "p_gas_exceeds_one_bar_complete_volatilisation",
            "P would leave essentially completely; the fixed-composition dilute "
            "equilibrium is not physical above 1 bar.",
            exceeded or p_species,
            p_total,
        )
        flag = f"{notice.code}: {notice.message}"
        for species in p_species:
            domain_flags[species] = "; ".join(filter(None, (domain_flags.get(species), flag)))

    identity = _gas_result_identity(datapack, profile_name)
    return ImccGasResult(
        result,
        unit="bar",
        domain_flags=domain_flags,
        provenance_class=provenance_class,
        omitted_channels=omitted_channels,
        liquid_reference_status=liquid_statuses,
        _profile_identity=identity,
        receipt=(
            None
            if suppress_receipt
            else build_calculation_receipt(
                request={
                    "activities": activities,
                    "T_K": T_K,
                    "fO2": fO2,
                    "profile": profile_name,
                    "options": {
                        "parent_oxides": requested_parent_oxides,
                        "allow_extrapolation": allow_extrapolation,
                        "gas_species": gas_species,
                        "include_ions": include_ions,
                        "composition_wt_pct": composition_wt_pct,
                    },
                },
                data_digests=(
                    {"gas": identity[1], "condensate": identity[2]}
                    if identity is not None else {}
                ),
                solver={},
            )
        ),
    )



_FORMULA_PART = re.compile(r"([A-Z][a-z]?)(\d*)")
# log10(pO2/bar) search bracket. The upper edge (1 bar) is a physical ceiling,
# not a numerical convenience: the effusion law J ∝ p/√(M T) holds only in
# molecular flow, where the mean free path λ ≫ orifice diameter d.
# λ = k_B T / (√2 π σ² p); with σ ≈ 3.6e-10 m at 2000 K, λ ≈ 0.5 µm × (1 bar / p).
# Knudsen orifices are ~0.1–1 mm, so molecular flow (λ/d ≳ 1) already requires a
# total pressure ≲ 1e-3 bar. That total-pressure line is
# ``_KNUDSEN_TOTAL_PRESSURE_BAR``. A root above 1 bar therefore lies far outside
# the model's validity and is refused rather than returned. The lower edge
# (1e-30 bar) is below any physically resolvable oxygen pressure.
_KNUDSEN_TOTAL_PRESSURE_BAR = 1e-3
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


def gas_channel_metadata(
    activities: Mapping[str, float] | Sequence[float] | np.ndarray,
    temperature_K: float,
    datapack: ImccGasDatapack,
    *,
    parent_oxides: Sequence[str] | None = None,
    include_ions: bool = False,
) -> Mapping[str, GasChannelMetadata]:
    """Expose selected reaction-channel stoichiometry and table availability.

    Neutral weights reuse :func:`oxygen_balance_species_metadata` and the
    retained reaction tables, including the side-oxide and gas-parent rules
    used by the legacy oxygen balance. Charged channels use full neutral
    formula attribution; their neutral pO2 exponent is not asserted. Parent
    selection, activity vector shape and values, and temperature validation
    follow :func:`evaluate_gas`.
    """
    if not is_declared_real_scalar(temperature_K, allow_numeric_str=True):
        raise ValueError(f"temperature must be a real number, got {temperature_K!r}")
    try:
        temperature = float(temperature_K)
    except OverflowError as exc:
        raise ValueError("temperature must be finite and positive") from exc
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"temperature must be finite and positive, got {temperature_K}")

    explicit_parents = parent_oxides is not None
    parents = list(IMCC_PARENT_OXIDES if parent_oxides is None else parent_oxides)
    if isinstance(activities, Mapping):
        unknown = sorted(
            name for name in activities
            if name not in _REACTION_PARENT_OXIDES and name not in _SF04_REACTIONS
        )
        if unknown:
            raise ImccGasUnknownActivityError(
                "activity key(s) not a recognised gas parent or species: "
                f"{unknown}"
            )
        if explicit_parents:
            extra = sorted(
                name for name in activities
                if name in _REACTION_PARENT_OXIDES and name not in parents
            )
            if extra:
                raise ValueError(
                    "activity mapping contains recognised parent key(s) "
                    f"{extra} outside parent_oxides; add them to parent_oxides"
                )
        else:
            parents.extend(
                name for name in activities
                if name in _REACTION_PARENT_OXIDES and name not in parents
            )
    def activity_as_float(name: str, value: Any) -> float:
        if not is_declared_real_scalar(value, allow_numeric_str=True):
            raise ValueError(f"activity for {name!r} must be a real number")
        try:
            return float(value)
        except OverflowError as exc:
            raise ValueError(
                f"activity for {name!r} is too large to represent as a float"
            ) from exc

    if isinstance(activities, Mapping):
        activity_values = {
            name: activity_as_float(name, activities.get(name, 0.0))
            for name in parents
        }
        if not explicit_parents:
            activity_values.update(
                (name, activity_as_float(name, activities[name]))
                for name in activities
                if name in _REACTION_PARENT_OXIDES and name not in activity_values
            )
    else:
        raw = np.asarray(activities, dtype=object)
        if raw.shape[0] != len(parents):
            raise ValueError(
                f"activities vector length {raw.shape[0]} does not match "
                f"{len(parents)} parent oxides"
            )
        activity_values = {
            name: activity_as_float(name, value)
            for name, value in zip(parents, raw, strict=True)
        }

    channels, omitted = _default_reactions(parents, datapack, temperature)
    selected_by_name = dict(channels)
    selected_by_name.update(
        (name, _SF04_REACTIONS[name]) for name in omitted
    )
    for species, (parent, _n_gas, _n_o2) in channels:
        if parent:
            value = activity_values.get(parent)
            if value is None:
                raise ImccGasSpeciesNotFoundError(
                    f"gas channel {species!r} needs parent activity {parent!r}, "
                    "which is not in parent_oxides"
                )
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"activity of {parent!r} must be finite and >= 0, got {value}"
                )
        side_parent = _side_parent(species)
        if side_parent:
            value = activity_values.get(side_parent)
            if value is None:
                raise ImccGasSpeciesNotFoundError(
                    f"gas channel {species!r} needs parent activity "
                    f"{side_parent!r}, which is not in parent_oxides"
                )
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"activity of {side_parent!r} must be finite and >= 0, got {value}"
                )
    neutral_parents = {name: reaction[0] for name, reaction in selected_by_name.items()}
    base = oxygen_balance_species_metadata(neutral_parents)
    result: dict[str, GasChannelMetadata] = {}
    for name, (parent, n_gas, n_o2) in selected_by_name.items():
        data = base[name]
        exponent: float | None = data.pO2_exponent
        side = _ALKALI_HALIDE_REACTION_OXIDES.get(name)
        if side:
            data = data._replace(
                parent_oxygen_demand=side[1], pO2_exponent=-side[2]
            )
            exponent = -side[2]
        elif parent in _GAS_PHASE_PARENT_ROWS:
            data = data._replace(parent_oxygen_demand=0.0, pO2_exponent=n_o2)
            exponent = n_o2
        elif name in _HALIDE_REACTIONS or name in _HYDROGEN_HALIDE_PARENTS:
            data = data._replace(parent_oxygen_demand=0.0)
        result[name] = GasChannelMetadata(
            name, data.oxygen_atoms, data.parent_oxygen_demand, 0,
            name not in omitted, exponent,
        )

    if include_ions:
        ion_pairs = (*_GAS_IONIZATION_PAIRS, *_GAS_ELECTRON_ATTACHMENTS)
        ion_rows = set(datapack.gas_df.index)
        for neutral, ion in ion_pairs:
            if neutral not in result:
                continue
            neutral_data = result[neutral]
            formula = neutral
            atoms = _formula_atoms(formula)
            result[ion] = GasChannelMetadata(
                ion,
                float(atoms.get("O", 0)),
                neutral_data.parent_oxygen_demand,
                1 if ion.endswith("+") else -1,
                f"{ion}(g)" in ion_rows and neutral_data.available,
                None,
            )
        result["e-"] = GasChannelMetadata(
            "e-", 0.0, 0.0, -1,
            "e-(g)" in ion_rows, None,
        )
    return MappingProxyType(result)


def _checked_oxygen_fsum(values: Iterable[float], label: str) -> float:
    terms = tuple(values)
    if any(not math.isfinite(value) for value in terms):
        cause = OverflowError(f"{label} contains a non-finite term")
        raise ImccGasOxygenBalanceError(str(cause)) from cause
    try:
        total = math.fsum(terms)
    except (OverflowError, ValueError) as exc:
        raise ImccGasOxygenBalanceError(
            f"{label} aggregation overflow: {exc}"
        ) from exc
    if not math.isfinite(total):
        cause = OverflowError(f"{label} sum is non-finite")
        raise ImccGasOxygenBalanceError(str(cause)) from cause
    return total


def weighted_oxygen_balance_from_pressure_model(
    pressure_model: Callable[[float], Mapping[str, float]],
    oxygen_weights: Mapping[str, float],
    *,
    conductance_mol_m2_s_Pa: Mapping[str, float],
    bulk_pressures_bar: Mapping[str, float],
    log10_pO2_bracket: tuple[float, float],
    bulk_plane: str | None = None,
) -> WeightedOxygenBalanceResult:
    """Solve a frozen, engine-neutral signed net-flux oxygen balance.

    ``pressure_model`` maps log10(pO2/bar) to equilibrium pressures in bar.
    Conductances are mol m^-2 s^-1 Pa^-1; bar-to-Pa conversion occurs here
    exactly once. The scan only provides sampled root-count evidence for
    arbitrary callbacks, not a global uniqueness proof. The returned signed
    root is re-evaluated and refused unless its residual is at most 1e-12
    times the absolute weighted-transfer scale. The result exposes that scale
    and its forward/backward components. Applied-flux closure uses the same
    relative scale with no absolute floor; zero transfer is reported
    explicitly and never certified as applied closure. ``bulk_plane`` names
    the plane the caller's bulk pressures describe; it is echoed in the result
    and not interpreted.
    """
    if bulk_plane is not None and (not isinstance(bulk_plane, str) or not bulk_plane.strip()):
        raise ImccGasOxygenBalanceError("bulk_plane must be a nonempty string or None")
    names = tuple(oxygen_weights)
    if not names or set(names) != set(conductance_mol_m2_s_Pa) or set(names) != set(bulk_pressures_bar):
        raise ImccGasOxygenBalanceError(
            "weights, conductances, and bulk pressures must have the same nonempty species keys"
        )
    weights: dict[str, float] = {}
    conductance: dict[str, float] = {}
    bulk: dict[str, float] = {}
    for name in names:
        try:
            weights[name] = float(oxygen_weights[name])
            conductance[name] = float(conductance_mol_m2_s_Pa[name])
            bulk[name] = float(bulk_pressures_bar[name])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ImccGasOxygenBalanceError(f"invalid frozen input for {name!r}") from exc
        if not math.isfinite(weights[name]):
            raise ImccGasOxygenBalanceError(f"non-finite oxygen weight for {name!r}")
        if not math.isfinite(conductance[name]) or conductance[name] < 0.0:
            raise ImccGasOxygenBalanceError(f"conductance for {name!r} must be finite and nonnegative")
        if not math.isfinite(bulk[name]) or bulk[name] < 0.0:
            raise ImccGasOxygenBalanceError(f"bulk pressure for {name!r} must be finite and nonnegative")
    try:
        low, high = map(float, log10_pO2_bracket)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ImccGasOxygenBalanceError("invalid weighted oxygen-balance bracket") from exc
    if not (math.isfinite(low) and math.isfinite(high) and low < high):
        raise ImccGasOxygenBalanceError(f"invalid weighted oxygen-balance bracket {log10_pO2_bracket}")

    evaluations = 0

    def weighted_pressure_magnitude(name: str, pressure: float) -> float:
        magnitude = abs(weights[name]) * conductance[name] * pressure * 1e5
        if not math.isfinite(magnitude):
            cause = OverflowError(f"weighted pressure-transfer term overflow for {name!r}")
            raise ImccGasOxygenBalanceError(str(cause)) from cause
        if weights[name] != 0.0 and conductance[name] != 0.0 and pressure != 0.0 and magnitude == 0.0:
            raise ImccGasOxygenBalanceError(
                f"weighted pressure-transfer term underflow for {name!r}"
            )
        return magnitude

    def evaluate(logp: float) -> tuple[float, dict[str, float], dict[str, float]]:
        nonlocal evaluations
        evaluations += 1
        try:
            raw = pressure_model(logp)
        except Exception as exc:
            raise ImccGasOxygenBalanceError(
                f"pressure model refused log10(pO2/bar)={logp}: {exc}"
            ) from exc
        if not isinstance(raw, MappingABC) or set(raw) != set(names):
            raise ImccGasOxygenBalanceError(
                f"pressure model must return exactly the declared species at log10(pO2/bar)={logp}"
            )
        pressures: dict[str, float] = {}
        fluxes: dict[str, float] = {}
        for name in names:
            try:
                pressure = float(raw[name])
            except (TypeError, ValueError, OverflowError) as exc:
                raise ImccGasOxygenBalanceError(f"invalid pressure for {name!r}") from exc
            if not math.isfinite(pressure) or pressure < 0.0:
                raise ImccGasOxygenBalanceError(f"pressure for {name!r} must be finite and nonnegative")
            pressures[name] = pressure
            delta = pressure - bulk[name]
            fluxes[name] = conductance[name] * delta * 1e5
            if not math.isfinite(fluxes[name]):
                raise ImccGasOxygenBalanceError(f"non-finite flux for {name!r}")
            if conductance[name] != 0.0 and delta != 0.0 and fluxes[name] == 0.0:
                raise ImccGasOxygenBalanceError(f"flux underflow for {name!r}")
        terms = [weights[name] * fluxes[name] for name in names]
        if any(
            weights[name] != 0.0 and fluxes[name] != 0.0 and term == 0.0
            for name, term in zip(names, terms, strict=True)
        ):
            raise ImccGasOxygenBalanceError("weighted oxygen-flux term underflow")
        if any(not math.isfinite(term) for term in terms):
            cause = OverflowError("weighted oxygen-flux term is non-finite")
            raise ImccGasOxygenBalanceError(str(cause)) from cause
        residual = _checked_oxygen_fsum(terms, "weighted oxygen-flux residual")
        return residual, pressures, fluxes

    nodes = np.linspace(low, high, 81)
    samples = [evaluate(float(node))[0] for node in nodes]
    if all(value == 0.0 for value in samples):
        raise ImccGasOxygenBalanceError("weighted oxygen-flux residual is identically zero; pO2 is underdetermined")
    increasing = all(right >= left for left, right in zip(samples, samples[1:]))
    decreasing = all(right <= left for left, right in zip(samples, samples[1:]))
    monotonicity = "increasing" if increasing and not decreasing else "decreasing" if decreasing and not increasing else "constant" if increasing else "nonmonotone"
    roots: list[tuple[float, float]] = []
    for index, value in enumerate(samples):
        if value == 0.0:
            roots.append((float(nodes[index]), float(nodes[index])))
    for index, (left, right) in enumerate(zip(samples, samples[1:])):
        if (left < 0.0 < right) or (right < 0.0 < left):
            roots.append((float(nodes[index]), float(nodes[index + 1])))
    if len(roots) != 1:
        raise ImccGasOxygenBalanceError(
            f"weighted oxygen-flux residual has {len(roots)} sampled roots in {log10_pO2_bracket}"
        )
    root_bracket = roots[0]
    if root_bracket[0] == root_bracket[1]:
        root = root_bracket[0]
    else:
        try:
            root = float(brentq(
                lambda value: evaluate(value)[0], *root_bracket,
                xtol=5e-324,
            ))
        except (ValueError, RuntimeError) as exc:
            raise ImccGasOxygenBalanceError(f"weighted oxygen-flux solve failed: {exc}") from exc
    residual, pressures, fluxes = evaluate(root)
    weighted_terms = [weights[name] * fluxes[name] for name in names]
    if any(not math.isfinite(term) for term in weighted_terms):
        cause = OverflowError("returned weighted oxygen-flux term is non-finite")
        raise ImccGasOxygenBalanceError(str(cause)) from cause
    forward_scale = _checked_oxygen_fsum(
        (max(term, 0.0) for term in weighted_terms),
        "forward weighted-flux scale",
    )
    backward_scale = _checked_oxygen_fsum(
        (max(-term, 0.0) for term in weighted_terms),
        "backward weighted-flux scale",
    )
    scale = _checked_oxygen_fsum(
        (abs(term) for term in weighted_terms),
        "absolute weighted-transfer scale",
    )
    equilibrium_scale = _checked_oxygen_fsum(
        (weighted_pressure_magnitude(name, pressures[name]) for name in names),
        "equilibrium weighted pressure-transfer scale",
    )
    bulk_scale = _checked_oxygen_fsum(
        (weighted_pressure_magnitude(name, bulk[name]) for name in names),
        "bulk weighted pressure-transfer scale",
    )
    gross_scale = _checked_oxygen_fsum(
        (equilibrium_scale, bulk_scale), "gross weighted pressure-transfer scale",
    )
    if scale == 0.0 and residual != 0.0:
        raise ImccGasOxygenBalanceError("zero weighted transfer has a nonzero residual")
    if scale > 0.0 and abs(residual) / scale > 1e-12:
        raise ImccGasOxygenBalanceError(
            "returned weighted oxygen-flux root failed signed closure "
            f"(residual={residual}, scale={scale})"
        )
    applied_terms = [
        weights[name] * max(fluxes[name], 0.0) for name in names
    ]
    if any(not math.isfinite(term) for term in applied_terms):
        cause = OverflowError("applied weighted-flux term is non-finite")
        raise ImccGasOxygenBalanceError(str(cause)) from cause
    applied = _checked_oxygen_fsum(
        applied_terms, "applied weighted oxygen-flux residual"
    )
    applied_closes = scale > 0.0 and abs(applied) / scale <= 1e-12
    closure_state = (
        "zero_transfer" if scale == 0.0
        else "closed" if applied_closes else "not_closed"
    )
    try:
        p_o2 = 10.0**root
    except OverflowError as exc:
        raise ImccGasOxygenBalanceError("root pO2 is outside the finite pressure range") from exc
    if not math.isfinite(p_o2) or p_o2 <= 0.0:
        raise ImccGasOxygenBalanceError("root pO2 is outside the finite pressure range")
    return WeightedOxygenBalanceResult(
        p_o2, pressures, fluxes, residual, scale, equilibrium_scale,
        bulk_scale, gross_scale, forward_scale,
        backward_scale, applied, applied_closes, closure_state,
        root_bracket, 1, "sampled; finite scan does not prove global uniqueness",
        monotonicity, evaluations, bulk_plane,
    )


def oxygen_balance_from_pressure_model(
    pressure_model: Callable[[float], Mapping[str, float]],
    species: Mapping[str, OxygenBalanceSpeciesMetadata],
    *,
    bracket: tuple[float, float] = _OXYGEN_BALANCE_BRACKET,
) -> tuple[float, Mapping[str, float], dict[str, object]]:
    """Solve the generic Knudsen oxygen-balance root for a pressure model.

    The declared exponents establish the expected monotonic direction of each
    flux term. The supplied model is evaluated directly, so it may bend between
    pressure nodes as long as its oxygen-balance residual has a bracketed root.
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

    model_evaluations = 0

    def at(logp: float) -> tuple[float, Mapping[str, float], float, float]:
        nonlocal model_evaluations
        model_evaluations += 1
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
        oxygen_flux = _checked_oxygen_fsum(
            (
                data.oxygen_atoms * validated[name] / math.sqrt(data.molar_mass)
                for name, data in species.items()
            ),
            "oxygen-balance oxygen flux",
        )
        parent_flux = _checked_oxygen_fsum(
            (
                data.parent_oxygen_demand * validated[name] / math.sqrt(data.molar_mass)
                for name, data in species.items()
            ),
            "oxygen-balance parent flux",
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
    root_xtol = 1e-12
    try:
        root, solver = brentq(
            lambda value: at(value)[0], low, high, xtol=root_xtol,
            full_output=True, disp=False,
        )
    except (ValueError, RuntimeError) as exc:
        raise ImccGasOxygenBalanceError(f"oxygen-balance solve failed: {exc}") from exc
    if not solver.converged:
        raise ImccGasOxygenBalanceError("oxygen-balance solve did not converge")

    residual, pressures, oxygen_flux, parent_flux = at(root)
    residual_scale = max(oxygen_flux, parent_flux, 1e-300)
    relative_residual = abs(residual) / residual_scale
    max_exponent = max(
        (abs(data.pO2_exponent) for data in species.values()), default=1.0
    )
    # At a power-law root, |dF/dlog10(pO2)| / max(fluxes) is bounded by
    # 2*ln(10)*max(|k|): positive and negative flux sums are equal, and
    # |nO-demand| <= nO+demand. Thus this residual tolerance follows the
    # 1e-12 dex Brent xtol while allowing the solver's default relative tol.
    solver_logp_tol = root_xtol + 4.0 * math.ulp(1.0) * max(abs(low), abs(high))
    residual_tolerance = max(
        32.0 * math.ulp(1.0),
        2.0 * math.log(10.0) * max(1.0, max_exponent) * solver_logp_tol,
    )
    root_path = "direct"
    refinement_evaluations = 0
    iterations = solver.iterations
    if relative_residual > residual_tolerance:
        if not (f_low < 0.0 < f_high):
            raise ImccGasOxygenBalanceError(
                "oxygen-balance residual exceeds tolerance and no sign change "
                f"certifies refinement (residual={residual}, "
                f"F(low)={f_low}, F(high)={f_high})"
            )
        initial_model_evaluations = model_evaluations
        try:
            root, refinement_solver = brentq(
                lambda value: at(value)[0], low, high,
                xtol=root_xtol / 1000.0,
                full_output=True, disp=False,
            )
        except (ValueError, RuntimeError) as exc:
            raise ImccGasOxygenBalanceError(
                "oxygen-balance refinement failed for true residual "
                f"{residual}: {exc}"
            ) from exc
        if not refinement_solver.converged:
            raise ImccGasOxygenBalanceError(
                "oxygen-balance refinement did not converge for true residual "
                f"{residual}"
            )
        iterations += refinement_solver.iterations
        refinement_evaluations = model_evaluations - initial_model_evaluations
        residual, pressures, oxygen_flux, parent_flux = at(root)
        residual_scale = max(oxygen_flux, parent_flux, 1e-300)
        relative_residual = abs(residual) / residual_scale
        if relative_residual > residual_tolerance:
            raise ImccGasOxygenBalanceError(
                "oxygen-balance refinement did not meet residual tolerance "
                f"{residual_tolerance} (true residual={residual}, "
                f"relative residual={relative_residual})"
            )
        root_path = "refined"
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
    total_pressure_bar = math.fsum(
        pressure
        for pressure in map(float, pressures.values())
        if math.isfinite(pressure)
    )
    diagnostics: dict[str, object] = {
        "mode": "oxygen_balance_effusion",
        "residual": relative_residual,
        "bracket": bracket,
        "iterations": iterations,
        "dominant_O_carriers": oxygen_carriers,
        "dominant_metal_carriers": metal_carriers,
        "total_pressure_bar": total_pressure_bar,
        "knudsen_regime": total_pressure_bar <= _KNUDSEN_TOTAL_PRESSURE_BAR,
        "root_solver": {
            "path": root_path,
            "model_evaluations": model_evaluations,
            "refinement_evaluations": refinement_evaluations,
            "residual_tolerance": residual_tolerance,
        },
    }
    return 10.0**root, pressures, diagnostics


def evaluate_gas_oxygen_balance(
    activities: Mapping[str, float] | Sequence[float] | np.ndarray,
    T_K: float,
    datapack: ImccGasDatapack | None = None,
    parent_oxides: Sequence[str] | None = None,
    allow_extrapolation: bool = True,
    imposed_parents: Sequence[str] | None = None,
    include_ions: bool = False,
    *,
    profile: str | Profile | None = None,
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

    With ``include_ions=True``, charge closure is recalculated at each trial
    pressure. Metal ions retain their neutral parent's oxygen demand per metal
    atom. The ion's effective pO2 exponent is taken from that parent for the
    generic solver's monotonicity check; charge closure makes the actual ion
    pressure composition-dependent, so the generic monotonicity proof does
    not strictly cover the ion contribution. Before solving, a fixed 64-interval
    scan checks the allowed pO2 bracket and refuses when it finds multiple
    simple sign-changing intervals. A finite scan cannot rule out a tangent
    root between samples.
    Positive ions are counted as ambipolar ion/electron pairs effusing at the
    ion mass. Counting free electrons at their own mass would apply the
    ``1/sqrt(m_e)`` factor to a beam that cannot escape with net charge; this
    paired approximation enforces quasi-neutral effusion. Negative ions are
    excluded from the balance: the model's attachment channels do not define
    their compensating positive beam partner, so assigning an independent
    mass-weighted anion flux would violate the same ambipolar constraint.

    Returns ``(pO2_bar, partial_pressures, diagnostics)``. Partial pressures
    match :func:`evaluate_gas` with the selected ion setting; diagnostics
    contain relative residual, the
    log10(bar) bracket, iteration count, the five strongest O and parent
    oxygen carriers, the summed finite partial pressures, and whether that
    total is at or below the 1e-3 bar molecular-flow line. The returned root
    is unchanged when the total is above that line.

    ``imposed_parents`` names external gas reservoirs such as ``("N2",)`` or
    ``("C",)`` for a nitrogen or CO2 sweep. Their species remain in the
    returned pressure map but are excluded from the melt-derived oxygen
    balance. Names whose activity was not supplied are ignored. The default is
    no imposed reservoirs; a bare string is rejected because it is not a
    sequence of parent keys.

    Every supplied non-oxide gas parent and condensed parent contributes its
    oxygen-carrying species to the melt-derived oxygen demand unless named in
    ``imposed_parents``. Metal vapours carry no oxygen. With ``parent_oxides``
    omitted, recognised oxide parents in a mapping (for example ``Cr2O3``)
    enter the balance as they enter the returned pressures. Parent oxides
    omitted from an explicit ``parent_oxides`` sequence remain excluded.
    """
    requested_parent_oxides = parent_oxides
    requested_imposed_parents = (
        imposed_parents
        if imposed_parents is None or isinstance(imposed_parents, str)
        else tuple(imposed_parents)
    )
    datapack, profile_name = _select_gas_datapack(datapack, profile)
    parent_oxides_explicit = parent_oxides is not None
    if parent_oxides is None:
        parent_oxides = IMCC_PARENT_OXIDES
    if isinstance(imposed_parents, str):
        raise TypeError("imposed_parents must be a sequence of parent-key strings")
    imposed = frozenset(requested_imposed_parents or ())
    nonoxide_parents = _nonoxide_parent_keys()
    unknown_imposed = imposed - nonoxide_parents
    if unknown_imposed:
        raise ValueError(
            "imposed_parents must name gas-parent activity keys; unknown: "
            f"{sorted(unknown_imposed)}"
        )
    try:
        T = float(T_K)
    except OverflowError as exc:
        raise ValueError("temperature must be finite and positive") from exc
    if not math.isfinite(T) or T <= 0.0:
        raise ValueError(f"temperature must be finite and positive, got {T_K}")

    available_parents = list(parent_oxides)
    if isinstance(activities, Mapping):
        # Without an explicit list, evaluate_gas auto-includes every
        # recognised parent in the mapping, oxides included. The balance must
        # count the same channels the pressure callback returns, or the root
        # ignores oxygen carried by those gases. An explicit list keeps
        # excluding unlisted oxides; non-oxide parents are added either way.
        available_parents.extend(
            parent
            for parent in activities
            if (
                parent in nonoxide_parents
                or (
                    not parent_oxides_explicit
                    and parent in _REACTION_PARENT_OXIDES
                )
            )
            and parent not in available_parents
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

    ion_parents: dict[str, str] = {}
    if include_ions:
        # Ions inherit each neutral channel's parent oxygen demand per atom.
        # Their exponent uses the parent's declared exponent as an effective
        # monotonicity proxy; the true charge-closure response is composition
        # dependent (see limitation in this function's docstring).
        for neutral, ion in _active_ion_pairs(datapack, _GAS_IONIZATION_PAIRS):
            if neutral not in species_data:
                continue
            neutral_data = species_data[neutral]
            neutral_atoms = _formula_atoms(neutral)
            neutral_element_count = sum(
                count for element, count in neutral_atoms.items() if element != "O"
            )
            if neutral_element_count <= 0:
                continue
            ion_data = OxygenBalanceSpeciesMetadata(
                # Electron mass is negligible at the precision of the gas
                # atomic masses and the ambipolar pair is assigned ion mass.
                neutral_data.molar_mass,
                0.0,
                neutral_data.parent_oxygen_demand / neutral_element_count,
                neutral_data.pO2_exponent,
            )
            # This weight/exponent product must have the same nonnegative
            # sign as the neutral contribution for the generic solver.
            weight = (
                ion_data.oxygen_atoms - ion_data.parent_oxygen_demand
            ) / math.sqrt(ion_data.molar_mass)
            if weight * ion_data.pO2_exponent < 0.0:
                raise ImccGasOxygenBalanceError(
                    f"ion oxygen-balance flux is non-monotone for {ion!r}"
                )
            species_data[ion] = ion_data
            ion_parents[ion] = neutral

    def pressure_model(logp: float) -> ImccGasResult:
        pressure_activities = activities
        if isinstance(activities, Mapping) and parent_oxides_explicit:
            # Drop only recognised parents left out of the explicit selection,
            # so they stay out of the pressures as they stay out of the
            # balance channels. Every other key still reaches evaluate_gas,
            # which keeps refusing unknown keys.
            included_parents = set(available_parents)
            pressure_activities = {
                name: value
                for name, value in activities.items()
                if not (
                    name in _REACTION_PARENT_OXIDES
                    and name not in included_parents
                )
            }
        token = _SUPPRESS_TRIAL_RECEIPTS.set(True)
        try:
            return evaluate_gas(
                pressure_activities, T, 10.0**logp, datapack,
                parent_oxides=(
                    tuple(available_parents) if parent_oxides_explicit else None
                ),
                allow_extrapolation=allow_extrapolation,
                include_ions=include_ions,
            )
        finally:
            _SUPPRESS_TRIAL_RECEIPTS.reset(token)

    if include_ions:
        # Charge closure makes ion pressures composition-dependent, so sample
        # the actual residual before passing it to the generic Brent contract.
        # Sixty-four equal log-pressure intervals are deterministic and detect
        # multiple simple roots when each root changes sign between samples;
        # no finite grid can exclude a tangent root between adjacent samples.
        low, high = _OXYGEN_BALANCE_BRACKET
        grid_size = 64
        log_pressures = np.linspace(low, high, grid_size + 1)
        residuals: list[float] = []
        for log_pressure in log_pressures:
            trial_pressures = pressure_model(float(log_pressure))
            residuals.append(math.fsum(
                (data.oxygen_atoms - data.parent_oxygen_demand)
                * float(trial_pressures[name]) / math.sqrt(data.molar_mass)
                for name, data in species_data.items()
            ))
        intervals: list[tuple[float, float]] = []
        previous_nonzero: int | None = None
        for index, residual in enumerate(residuals):
            if residual == 0.0:
                continue
            if (
                previous_nonzero is not None
                and residuals[previous_nonzero] * residual < 0.0
            ):
                intervals.append((
                    float(log_pressures[previous_nonzero]),
                    float(log_pressures[index]),
                ))
            previous_nonzero = index
        if len(intervals) > 1:
            rendered = ", ".join(
                f"[{left:g}, {right:g}]" for left, right in intervals
            )
            raise ImccGasOxygenBalanceError(
                "ion oxygen-balance residual has multiple sign-changing intervals "
                f"in log10(pO2/bar): {rendered}"
            )
    p_o2, pressures, diagnostics = oxygen_balance_from_pressure_model(
        pressure_model, species_data
    )
    identity = _gas_result_identity(datapack, profile_name)
    if include_ions:
        diagnostics["ions_included"] = True
        diagnostics["ion_fraction_by_element"] = {
            ion.removesuffix("+"): (
                float(pressures[ion]) / (
                    float(pressures[ion]) + float(pressures[neutral])
                )
                if float(pressures[ion]) + float(pressures[neutral]) > 0.0
                else 0.0
            )
            for ion, neutral in ion_parents.items()
            if ion.endswith("+") and ion in pressures and neutral in pressures
        }
    return p_o2, replace(
        pressures,
        _profile_identity=identity,
        receipt=build_calculation_receipt(
            request={
                "activities": activities,
                "T_K": T_K,
                "profile": profile_name,
                "options": {
                    "parent_oxides": requested_parent_oxides,
                    "allow_extrapolation": allow_extrapolation,
                    "imposed_parents": requested_imposed_parents,
                    "include_ions": include_ions,
                },
            },
            data_digests=(
                {"gas": identity[1], "condensate": identity[2]}
                if identity is not None else {}
            ),
            solver={
                "log10_pO2_bracket": _OXYGEN_BALANCE_BRACKET,
                "root_xtol": 1e-12,
                "residual_tolerance": diagnostics["root_solver"]["residual_tolerance"],
                **(
                    {"refinement_xtol": 1e-15}
                    if diagnostics["root_solver"]["path"] == "refined"
                    else {}
                ),
                **({"ion_scan_intervals": 64} if include_ions else {}),
            },
        ),
    ), diagnostics


def evaluate_gas_saturation(
    parent_activities: Mapping[str, float] | Sequence[float] | np.ndarray,
    temperature_K: float,
    datapack: ImccGasDatapack | None = None,
    *,
    parent_oxides: Sequence[str] | None = None,
    include_ions: bool = False,
    allow_extrapolation: bool = False,
    log10_pO2_bracket: tuple[float, float] = (-100.0, 100.0),
    profile: str | Profile | None = None,
) -> ImccGasSaturationResult:
    """Solve saturated vapor over a fixed melt composition and temperature.

    Activities are held fixed at this temperature. Oxygen closure is the
    pressure weighted stoichiometric balance; this is an equilibrium vapor
    model, not the Knudsen flux closure.
    """
    suppress_receipt = _consume_trial_receipt_suppression()
    requested_parent_oxides = parent_oxides
    datapack, profile_name = _select_gas_datapack(datapack, profile)
    if parent_oxides is None:
        if isinstance(parent_activities, Mapping):
            parents = tuple(k for k in parent_activities if k in _REACTION_PARENT_OXIDES)
        else:
            parents = IMCC_PARENT_OXIDES
    else:
        parents = tuple(parent_oxides)
    reactions, omitted = _default_reactions(parents, datapack, float(temperature_K))
    # For E_aO_b(l) -> n_g E_cO_d(g) + n_O2 O2(g), each mole of
    # channel pressure represents gas molecules while the corresponding
    # parent oxygen is b/n_g per gas molecule. Thus its oxygen excess is
    # (n_g*d-b)/n_g = -2*n_O2/n_g atoms per molecule. Unit check: multiplying
    # by bar gives oxygen-atom pressure. For pure MO -> M + 1/2 O2, if the
    # atomic-O dissociation channel is absent, the metal's -1 coefficient
    # cancels O2's +2 coefficient at pM=2 pO2; with O(g), its +1 term also
    # enters, giving pM=pO+2 pO2.
    weights: dict[str, float] = {"O2": 2.0}
    exponents: dict[str, float] = {"O2": 1.0}
    for species, (parent, n_gas, n_O2) in reactions:
        if species == "O2":
            continue
        if parent in _GAS_PHASE_PARENT_ROWS:
            # Gas-parent channels (for example sulfur oxidation) borrow no
            # oxygen from the melt, so their vapor oxygen is all net oxygen.
            parent_o = float(_formula_atoms(species).get("O", 0))
        elif species in _ALKALI_HALIDE_REACTION_OXIDES:
            # The side table expresses oxygen demand per product molecule.
            parent_o = -_ALKALI_HALIDE_REACTION_OXIDES[species][1]
        elif species in _HALIDE_REACTIONS or species in _HYDROGEN_HALIDE_PARENTS:
            parent_o = 0.0
        elif parent:
            parent_o = -2.0 * n_O2 / n_gas
        else:
            parent_o = float(_formula_atoms(species).get("O", 0))
        weights[species] = parent_o
        if species in _ALKALI_HALIDE_REACTION_OXIDES:
            exponent = -_ALKALI_HALIDE_REACTION_OXIDES[species][2]
        elif species in _HYDROGEN_HALIDE_PARENTS:
            exponent = 0.0
        elif parent in _GAS_PHASE_PARENT_ROWS:
            exponent = n_O2
        elif parent:
            exponent = -n_O2 / n_gas
        else:
            exponent = _formula_atoms(species).get("O", 0) / 2.0
        exponents[species] = exponent
    if include_ions:
        for neutral, ion in (
            *_active_ion_pairs(datapack, _GAS_IONIZATION_PAIRS),
            *_active_ion_pairs(datapack, _GAS_ELECTRON_ATTACHMENTS),
        ):
            weights[ion] = weights.get(neutral, 0.0)
            exponents[ion] = exponents.get(neutral, 0.0)

    lo, hi = map(float, log10_pO2_bracket)
    if not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi:
        raise ImccGasSaturationError("invalid saturation pO2 bracket")
    try:
        endpoint_pressures = (10.0**lo, 10.0**hi)
    except OverflowError as exc:
        raise ImccGasSaturationError(
            "saturation pO2 bracket is outside the finite pressure range"
        ) from exc
    if not all(math.isfinite(value) and value > 0.0 for value in endpoint_pressures):
        raise ImccGasSaturationError(
            "saturation pO2 bracket is outside the finite pressure range"
        )

    def state(logp: float):
        token = _SUPPRESS_TRIAL_RECEIPTS.set(True)
        try:
            return evaluate_gas(
                parent_activities, temperature_K, 10.0 ** logp, datapack,
                parent_oxides=parents, allow_extrapolation=allow_extrapolation,
                include_ions=include_ions,
            )
        finally:
            _SUPPRESS_TRIAL_RECEIPTS.reset(token)

    def residual(logp: float) -> float:
        pressures = state(logp)
        return math.fsum(weights.get(name, 0.0) * pressure for name, pressure in pressures.items())

    # The sign criterion proves monotonicity for independent mass action
    # channels. Coupled charge closure may invalidate that proof, so scan it.
    sign_terms = [weights[name] * exponents.get(name, 0.0) for name in weights]
    monotone = (
        not include_ions
        and all(term >= -1e-15 for term in sign_terms)
        and any(term > 1e-15 for term in sign_terms)
    )
    if monotone:
        f_lo, f_hi = residual(lo), residual(hi)
        brackets = [(lo, hi)] if f_lo * f_hi < 0 else ([(lo, lo)] if f_lo == 0 else [(hi, hi)] if f_hi == 0 else [])
    else:
        grid = np.linspace(lo, hi, 81)
        vals = [residual(float(x)) for x in grid]
        brackets = [(float(grid[i]), float(grid[i])) for i in range(len(grid)) if vals[i] == 0]
        brackets.extend(
            (float(grid[i]), float(grid[i + 1]))
            for i in range(len(grid) - 1)
            if vals[i] * vals[i + 1] < 0
        )
    if len(brackets) != 1:
        raise ImccGasSaturationError(
            f"oxygen-stoichiometry residual has {len(brackets)} roots in pO2 bracket; refusing ambiguous saturation"
        )
    bracket = brackets[0]
    if bracket[0] == bracket[1]:
        root = bracket[0]
    else:
        root = float(brentq(residual, *bracket, xtol=1e-12))
    result = state(root)
    identity = _gas_result_identity(datapack, profile_name)
    channels = dict(result)
    p_total = math.fsum(channels.values())
    species_pressures = dict(channels)
    # Current retained reaction rows use one channel per species; preserve the
    # channel map explicitly so future repeated products can be accounted for.
    return ImccGasSaturationResult(
        10.0 ** root, species_pressures, channels, p_total,
        {name: p / p_total for name, p in channels.items()} if p_total else {},
        "fixed_composition_saturation",
        {"log10_pO2_bracket": bracket, "oxygen_residual": residual(root),
         "monotonicity_sign_criterion": monotone,
         "root_uniqueness": "proved" if monotone else "sampled",
         "include_ions": include_ions,
         "omitted_channels": omitted},
        result.domain_flags,
        _profile_identity=identity,
        receipt=(
            None
            if suppress_receipt
            else build_calculation_receipt(
                request={
                    "parent_activities": parent_activities,
                    "temperature_K": temperature_K,
                    "profile": profile_name,
                    "options": {
                        "parent_oxides": requested_parent_oxides,
                        "include_ions": include_ions,
                        "allow_extrapolation": allow_extrapolation,
                        "log10_pO2_bracket": log10_pO2_bracket,
                    },
                },
                data_digests={
                    "gas": identity[1] if identity is not None else None,
                    "condensate": identity[2] if identity is not None else None,
                },
                solver={
                    "log10_pO2_bracket": bracket,
                    "brentq_xtol": 1e-12,
                    **({"root_scan_points": 81} if not monotone else {}),
                },
            )
        ),
        liquid_reference_status=getattr(result, "liquid_reference_status", {}),
    )


def gas_boiling_temperature(
    parent_activities: Mapping[str, float] | Sequence[float] | np.ndarray,
    datapack: ImccGasDatapack | None = None,
    ambient_pressure_bar: float | None = None,
    T_bracket: tuple[float, float] | None = None,
    *,
    parent_oxides: Sequence[str] | None = None,
    include_ions: bool = False,
    allow_extrapolation: bool = False,
    profile: str | Profile | None = None,
) -> tuple[float, ImccGasSaturationResult]:
    """Find boiling T with caller activities held fixed throughout the bracket."""
    datapack, profile_name = _select_gas_datapack(datapack, profile)
    if ambient_pressure_bar is None or T_bracket is None:
        raise TypeError("ambient_pressure_bar and T_bracket are required")
    if not math.isfinite(ambient_pressure_bar) or ambient_pressure_bar <= 0:
        raise ValueError("ambient_pressure_bar must be finite and positive")
    low, high = map(float, T_bracket)
    if not math.isfinite(low) or not math.isfinite(high) or not (0 < low < high):
        raise ImccGasSaturationError("invalid boiling temperature bracket")
    cache: dict[float, ImccGasSaturationResult] = {}

    def saturation_state(T: float) -> ImccGasSaturationResult:
        token = _SUPPRESS_TRIAL_RECEIPTS.set(True)
        try:
            return evaluate_gas_saturation(
                parent_activities, T, datapack, parent_oxides=parent_oxides,
                include_ions=include_ions, allow_extrapolation=allow_extrapolation,
            )
        finally:
            _SUPPRESS_TRIAL_RECEIPTS.reset(token)

    def residual(T: float) -> float:
        state = saturation_state(T)
        cache[T] = state
        return state.P_sat_bar - ambient_pressure_bar
    f_low, f_high = residual(low), residual(high)
    if f_low * f_high > 0:
        raise ImccGasSaturationError("boiling temperature is not bracketed by T_bracket")
    temperature = low if f_low == 0 else high if f_high == 0 else float(brentq(residual, low, high, xtol=1e-8))
    state = cache.get(temperature) or saturation_state(temperature)
    diagnostics = dict(state.diagnostics)
    diagnostics.update({
        "ambient_pressure_bar": ambient_pressure_bar,
        "temperature_bracket_K": (low, high),
        "boiling_residual_bar": state.P_sat_bar - ambient_pressure_bar,
    })
    identity = _gas_result_identity(datapack, profile_name)
    return temperature, ImccGasSaturationResult(
        state.pO2_bar, state.species_pressures_bar, state.channel_pressures_bar,
        state.P_sat_bar, state.mole_fractions, state.mode, diagnostics,
        state.domain_flags,
        _profile_identity=identity,
        receipt=build_calculation_receipt(
            request={
                "parent_activities": parent_activities,
                "ambient_pressure_bar": ambient_pressure_bar,
                "T_bracket": T_bracket,
                "profile": profile_name,
                "options": {
                    "parent_oxides": parent_oxides,
                    "include_ions": include_ions,
                    "allow_extrapolation": allow_extrapolation,
                },
            },
            data_digests={
                "gas": identity[1] if identity is not None else None,
                "condensate": identity[2] if identity is not None else None,
            },
            solver={
                "temperature_bracket_K": (low, high),
                "temperature_brentq_xtol": 1e-8,
                "log10_pO2_bracket": (-100.0, 100.0),
                "pO2_brentq_xtol": 1e-12,
            },
        ),
        liquid_reference_status=getattr(state, "liquid_reference_status", {}),
    )
