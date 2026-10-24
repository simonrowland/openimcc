"""Typed authority records for liquid-oxide reference standards."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping


class LiquidReferenceAuthority(str, Enum):
    CERTIFIED = "certified"
    CORROBORATED_CALCULATION = "corroborated_calculation"
    EXTRAPOLATED = "extrapolated"


@dataclass(frozen=True)
class EvidenceReference:
    author: str
    year: int
    basis: str
    composition_scope: Mapping[str, tuple[float, float]] | None = None

    def __post_init__(self) -> None:
        if self.composition_scope is not None:
            object.__setattr__(self, "composition_scope", MappingProxyType(dict(self.composition_scope)))


@dataclass(frozen=True)
class LiquidReferenceRecord:
    species: str
    authority: LiquidReferenceAuthority
    certified_T_K: tuple[float, float] | None
    band_value: float | None
    band_unit: str | None
    evidence_ref: tuple[EvidenceReference, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.authority, LiquidReferenceAuthority):
            raise ValueError(f"unknown liquid-reference authority: {self.authority!r}")
        if (self.band_value is None) != (self.band_unit is None):
            raise ValueError("band value and unit must both be present or both absent")
        if self.certified_T_K is not None:
            low, high = self.certified_T_K
            if low < 0 or high < low:
                raise ValueError("invalid certified temperature window")
        if self.band_value is not None and self.band_value < 0:
            raise ValueError("uncertainty band must be non-negative")


@dataclass(frozen=True)
class CertificationStatus:
    species: str
    authority: LiquidReferenceAuthority
    certified_T_K: tuple[float, float] | None
    band_value: float | None
    band_unit: str | None
    evidence_ref: tuple[EvidenceReference, ...]
    composition_scope: Mapping[str, tuple[float, float]] | None
    scope_applied: bool
    reason: str

    def __post_init__(self) -> None:
        if self.composition_scope is not None:
            object.__setattr__(self, "composition_scope", MappingProxyType(dict(self.composition_scope)))


def _e(author: str, year: int, basis: str, scope=None) -> EvidenceReference:
    return EvidenceReference(author, year, basis, scope)


_RECORDS = (
    LiquidReferenceRecord("SiO2(l)", LiquidReferenceAuthority.CERTIFIED, (900, 1996), None, None,
        (_e("Richet, Bottinga & Tequi", 1984, "measured liquid Cp 81.37 J/mol/K by drop calorimetry, 900-1800 K; measured fusion anchor Tm 1996 K, Delta Hfus 9.58 kJ/mol; DOI 10.1111/j.1151-2916.1984.tb19155.x"),)),
    LiquidReferenceRecord("K2O(l)", LiquidReferenceAuthority.CERTIFIED, (1259, 1765), 0.13, "dex pO2",
        (_e("Plante", 1979, "measured potassium pressures; residual about -0.13 dex at 25-40 wt% K2O; about -0.38 dex at 12-20 wt% K2O", {"K2O_wt_pct": (25, 40)}),)),
    LiquidReferenceRecord("CaO(l)", LiquidReferenceAuthority.CERTIFIED, (1323, 2473), 0.07, "dex pO2",
        (_e("Alvares, Deffrennes, Pisch & Jakse", 2020, "molecular-dynamics calculation of melting and fusion enthalpy; corroborated by independent calculation, not measured; within 2.6 kJ/mol; no measured fusion datum exists; DOI 10.1063/1.5141841"), _e("Menescardi, Ceresoli & Belmonte", 2024, "molecular-dynamics calculation of melting and fusion enthalpy; corroborated by independent calculation, not measured; within 2.6 kJ/mol; no measured fusion datum exists; DOI 10.1021/acs.jpcc.4c04752"))),
    LiquidReferenceRecord("MgO(l)", LiquidReferenceAuthority.EXTRAPOLATED, None, 0.3, "dex pO2",
        (_e("JANAF and calculated values", 2024, "assumed fusion entropy; calculated fusion values scatter by about 10-19 kJ/mol"),)),
    LiquidReferenceRecord("FeO(l)", LiquidReferenceAuthority.CERTIFIED, (1649.15, 1766.15), None, None,
        (_e("Darken & Gurry via Masson", 1965, "measured FeO liquid activity range; DOI 10.1021/ja01209a030 and 10.1098/rspa.1965.0176"), _e("Hashimoto", 1983, "run 17c3-2 at 1973 K: free-evaporation residue retains 27.92 wt% FeO versus about 14 wt% predicted; kinetics included"))),
    LiquidReferenceRecord("Al2O3(l)", LiquidReferenceAuthority.CERTIFIED, (2500, 3000), None, None,
        (_e("JANAF Al-100", 1998, "measured liquid enthalpy fit starts at 2500 K; source liquid marker at 2327 K"),)),
    LiquidReferenceRecord("TiO2(l)", LiquidReferenceAuthority.CERTIFIED, (2130, 3000), None, None,
        (_e("JANAF O-044", 1998, "source-supported liquid function through 3000 K; supercooled below the 2130 K fusion marker"),)),
    LiquidReferenceRecord("Na2O(l)", LiquidReferenceAuthority.CERTIFIED, (1405.2, 1800), None, None,
        (_e("Richet, Bottinga & Tequi", 1984, "measured liquid Cp through 1800 K on the 1405.2 K fusion anchor; Cp extrapolation above 1800 K; DOI 10.1111/j.1151-2916.1984.tb19155.x"),)),
)
_BY_SPECIES = MappingProxyType({record.species: record for record in _RECORDS})


def liquid_reference_records() -> tuple[LiquidReferenceRecord, ...]:
    """Return the immutable liquid-reference authority records."""
    return _RECORDS


def liquid_reference_authority(
    species: str, T_K: float, composition: Mapping[str, float] | None = None
) -> CertificationStatus:
    """Resolve authority at T; certified-window endpoints are inclusive."""
    record = _BY_SPECIES.get(species)
    if record is None:
        raise KeyError(f"no liquid-reference authority record for {species!r}")
    authority = record.authority
    reason = "source authority applies"
    scope = record.evidence_ref[0].composition_scope if record.evidence_ref else None
    in_window = record.certified_T_K is not None and record.certified_T_K[0] <= T_K <= record.certified_T_K[1]
    scope_applied = False
    if authority is not LiquidReferenceAuthority.EXTRAPOLATED and not in_window:
        authority, reason = LiquidReferenceAuthority.EXTRAPOLATED, "outside certified temperature window"
    elif scope and composition is not None:
        scope_applied = True
        key, (low, high) = next(iter(scope.items()))
        value = composition.get(key)
        if value is None or not low <= value <= high:
            authority, reason = LiquidReferenceAuthority.EXTRAPOLATED, "outside evidence composition scope"
    elif scope:
        reason = "temperature window applies; composition scope not evaluated"
    return CertificationStatus(record.species, authority, record.certified_T_K,
        record.band_value, record.band_unit, record.evidence_ref, scope,
        scope_applied, reason)
