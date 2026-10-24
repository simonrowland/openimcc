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
class AuthorityWindow:
    authority: LiquidReferenceAuthority
    T_K: tuple[float, float]
    band_value: float | None
    band_unit: str | None
    evidence_ref: tuple[EvidenceReference, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.authority, LiquidReferenceAuthority):
            raise ValueError(f"unknown liquid-reference authority: {self.authority!r}")
        low, high = self.T_K
        if low < 0 or high < low:
            raise ValueError("invalid authority temperature window")
        if (self.band_value is None) != (self.band_unit is None):
            raise ValueError("band value and unit must both be present or both absent")
        if self.band_value is not None and self.band_value < 0:
            raise ValueError("uncertainty band must be non-negative")


@dataclass(frozen=True)
class LiquidReferenceRecord:
    species: str
    authority: LiquidReferenceAuthority
    certified_T_K: tuple[float, float] | None
    band_value: float | None
    band_unit: str | None
    evidence_ref: tuple[EvidenceReference, ...]
    additional_windows: tuple[AuthorityWindow, ...] = ()

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
        windows = sorted(
            ([self.certified_T_K] if self.certified_T_K is not None else [])
            + [window.T_K for window in self.additional_windows]
        )
        for previous, current in zip(windows, windows[1:]):
            if current[0] < previous[1]:
                raise ValueError("authority temperature windows overlap")


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
    in_composition_scope: bool | None = None

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
    # Premise: anchor each measured Cp continuation to shipped G at Tm=2327 K,
    # then compare G(T)=H-T*S with the shipped branch. For constant dCp,
    # dG=dCp[(T-Tm)-T ln(T/Tm)]; for Paradis' dCp=a+b(T-Tm),
    # dH=a(T-Tm)+b(T-Tm)^2/2 and dS=a ln(T/Tm)+b[(T-Tm)-Tm ln(T/Tm)].
    # Cp is J mol^-1 K^-1, b is J mol^-1 K^-2, and dG is J mol^-1
    # (divide by 1000 for kJ mol^-1); both forms give dG=0 at Tm.
    LiquidReferenceRecord("Al2O3(l)", LiquidReferenceAuthority.CERTIFIED, (2120, 3000), 0.007, "dex log10 pressure",
        (_e("JANAF Al-100", 1998, "measured liquid enthalpy fit starts at 2500 K; source liquid marker at 2327 K"),
         _e("Paradis, Ishikawa, Saita & Yoda", 2004, "electrostatic-levitation liquid and supercooled-liquid Cp, 2120-2450 K; measured fusion enthalpy 109.0 kJ/mol; DOI 10.1143/JJAP.43.1496"),
         _e("Crochetet et al.", 2026, "aerodynamic-levitation drop calorimetry; liquid Cp 168.6 +/- 9.6 J/mol/K over 2327-2873 K; DOI 10.1016/j.oceram.2026.101029")),
        (AuthorityWindow(LiquidReferenceAuthority.CORROBORATED_CALCULATION, (1801, 2120), 0.05,
            "dex log10 pressure", (
                _e("Paradis, Ishikawa, Saita & Yoda", 2004, "continuation of two independent measured supercooled-liquid heat-capacity series agrees with the reference within 2.6 kJ/mol; corroborated by independent calculation, not measured below 2120 K; DOI 10.1143/JJAP.43.1496"),
                _e("Crochetet et al.", 2026, "continuation of two independent measured supercooled-liquid heat-capacity series agrees with the reference within 2.6 kJ/mol; corroborated by independent calculation, not measured below 2120 K; DOI 10.1016/j.oceram.2026.101029"),
            )),)),
    LiquidReferenceRecord("TiO2(l)", LiquidReferenceAuthority.EXTRAPOLATED, None, None, None,
        (_e("JANAF O-044", 1998, "estimated end-to-end: assumed fusion entropy 7.5 cal/mol/K by analogy with V2O4 and estimated liquid Cp 8 cal per g-atom; no measured pure-liquid data"),)),
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
    band_value, band_unit, evidence_ref = record.band_value, record.band_unit, record.evidence_ref
    reason = "source authority applies"
    matched = next((window for window in record.additional_windows
                    if window.T_K[0] <= T_K <= window.T_K[1]
                    and not (record.certified_T_K is not None
                             and record.certified_T_K[0] <= T_K <= record.certified_T_K[1])), None)
    active_window = matched
    if active_window is not None:
        authority, band_value, band_unit, evidence_ref = (
            active_window.authority, active_window.band_value,
            active_window.band_unit, active_window.evidence_ref,
        )
    active_T_K = active_window.T_K if active_window else record.certified_T_K
    scope = evidence_ref[0].composition_scope if evidence_ref else None
    in_window = active_T_K is not None and active_T_K[0] <= T_K <= active_T_K[1]
    scope_applied = False
    in_composition_scope = None
    if authority is not LiquidReferenceAuthority.EXTRAPOLATED and not in_window:
        # A window's band applies only inside that window.
        authority, reason = LiquidReferenceAuthority.EXTRAPOLATED, "outside certified temperature window"
        band_value = band_unit = None
    elif scope and composition is not None:
        scope_applied = True
        key, (low, high) = next(iter(scope.items()))
        value = composition.get(key)
        in_composition_scope = value is not None and low <= value <= high
        if not in_composition_scope:
            reason = "outside evidence composition scope; scored with notice"
    elif scope:
        reason = "temperature window applies; composition scope not evaluated"
    return CertificationStatus(record.species, authority, active_T_K,
        band_value, band_unit, evidence_ref, scope,
        scope_applied, reason, in_composition_scope)
