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
    # Window bands multiply the largest Gibbs departure (kJ/mol) by the
    # worst Si3 pressure sensitivity (0.0688 dex/kJ): 0.627 * 0.0688 =
    # 0.0431 dex, rounded to 0.05 for 1200-1500 K; 0.748 * 0.0688 =
    # 0.0515 dex, rounded to 0.06 for 1800-3000 K. The lower extension uses
    # the 2.6 kJ/mol stress cap: 2.6 * 0.0688 = 0.179 dex, rounded to 0.18;
    # this is a stress bound, not a confidence interval. pO2 sensitivity is
    # smaller than the Si3 worst case. The measured amorphous-silica enthalpy
    # window begins conservatively above the 1480 K glass transition.
    LiquidReferenceRecord("SiO2(l)", LiquidReferenceAuthority.CERTIFIED, (1500, 1800), None, None,
        (_e("Richet, Bottinga, Denielou, Petitet & Tequi", 1982, "drop-calorimetry relative enthalpy of amorphous silica on the liquid side above its glass transition; agrees with the reference within 0.001 kJ/mol from 1500-1800 K; DOI 10.1016/0016-7037(82)90383-0"),
         _e("Richet, Bottinga & Tequi", 1984, "measured SiO2 partial-molar Cp of 81.37 J/mol/K in sodium-silicate liquids by drop calorimetry from 900-1800 K; this is not pure-liquid SiO2 Cp; DOI 10.1111/j.1151-2916.1984.tb19155.x")),
        (AuthorityWindow(LiquidReferenceAuthority.CORROBORATED_CALCULATION, (1200, 1500), 0.05,
            "dex log10 pressure", (
                _e("Connolly", 2016, "metastable-liquid thermodynamic function within 0.627 kJ/mol of the reference from 1200-1500 K; calculated corroboration below the glass transition, not measurement; DOI 10.1002/2016JE005059"),
            )),
         AuthorityWindow(LiquidReferenceAuthority.CORROBORATED_CALCULATION, (1800, 3000), 0.06,
            "dex log10 pressure", (
                _e("Connolly", 2016, "metastable-liquid thermodynamic function; calculated corroboration only, not measurement; DOI 10.1002/2016JE005059"),
                _e("Gurvich (1991), as transcribed in NASA Glenn CEA", 2002, "liquid function agrees with the reference within 0.748 kJ/mol from 1800-3000 K; NASA/TP-2002-211556"),
            )),
         AuthorityWindow(LiquidReferenceAuthority.EXTRAPOLATED, (900, 1200), 0.18,
            "dex log10 pressure", (
                _e("JANAF O-038 and Connolly", 2016, "below the packaged row and outside Connolly's temperature range; ±0.18 dex is the 2.6 kJ/mol stress cap propagated at 0.0688 dex/kJ, not a confidence interval"),
            )))),
    LiquidReferenceRecord("K2O(l)", LiquidReferenceAuthority.CERTIFIED, (1259, 1765), 0.13, "dex pO2",
        (_e("Plante", 1979, "measured potassium pressures; residual about -0.13 dex at 25-40 wt% K2O; about -0.38 dex at 12-20 wt% K2O", {"K2O_wt_pct": (25, 40)}),)),
    LiquidReferenceRecord("CaO(l)", LiquidReferenceAuthority.CERTIFIED, (1323, 2473), 0.07, "dex pO2",
        (_e("Alvares, Deffrennes, Pisch & Jakse", 2020, "molecular-dynamics calculation of melting and fusion enthalpy; corroborated by independent calculation, not measured; within 2.6 kJ/mol; no measured fusion datum exists; DOI 10.1063/1.5141841"), _e("Menescardi, Ceresoli & Belmonte", 2024, "molecular-dynamics calculation of melting and fusion enthalpy; corroborated by independent calculation, not measured; within 2.6 kJ/mol; no measured fusion datum exists; DOI 10.1021/acs.jpcc.4c04752"))),
    # JANAF Mg-009 prints ±16.7 kJ/mol for fusion enthalpy and Tm=3105 K.
    # At 1323 K, δG=δΔH(1−T/Tm)=±9.58 kJ/mol (±9584 J/mol); for
    # MgO(l)=Mg(g)+½O2, Δlog10 pO2=2δG/(RT ln 10)=±0.757 dex, using
    # R=8.314462618 J mol⁻¹ K⁻¹. Store the outward-rounded cold-edge value
    # across the reviewed window; it is not a confidence interval, and no
    # band applies beyond that window.
    LiquidReferenceRecord("MgO(l)", LiquidReferenceAuthority.EXTRAPOLATED, None, None, None,
        (_e("JANAF Mg-009", 1998, "assumes fusion entropy; no primary pure-liquid fusion calorimetry or liquid-Cp series was found. DFT, molecular-dynamics, binary-liquidus and CALPHAD fusion values are listed conflicts and do not set a band"),),
        (AuthorityWindow(LiquidReferenceAuthority.EXTRAPOLATED, (1323, 2473), 0.76,
            "dex pO2", (
                _e("JANAF Mg-009", 1998, "printed ±16.7 kJ/mol fusion-enthalpy uncertainty; δG=δΔH(1−T/Tm) with Tm=3105 K gives a ±0.76 dex pO2 cold-edge band on 1323–2473 K, not a confidence interval"),
            )),)),
    LiquidReferenceRecord("FeO(l)", LiquidReferenceAuthority.CERTIFIED, (1649.15, 1766.15), None, None,
        (_e("Darken & Gurry via Masson", 1965, "measured FeO liquid activity range; DOI 10.1021/ja01209a030 and 10.1098/rspa.1965.0176"), _e("Hashimoto", 1983, "run 17c3-2 at 1973 K: free-evaporation residue retains 27.92 wt% FeO versus about 14 wt% predicted; kinetics included"))),
    # Premise: anchor each measured Cp continuation to shipped G at Tm=2327 K,
    # then compare G(T)=H-T*S with the shipped branch. For constant dCp,
    # dG=dCp[(T-Tm)-T ln(T/Tm)]; for Paradis' dCp=a+b(T-Tm),
    # dH=a(T-Tm)+b(T-Tm)^2/2 and dS=a ln(T/Tm)+b[(T-Tm)-Tm ln(T/Tm)].
    # Cp is J mol^-1 K^-1, b is J mol^-1 K^-2, and dG is J mol^-1
    # (divide by 1000 for kJ mol^-1); both forms give dG=0 at Tm.
    # Bands: max |dG| over each window times the worst row sensitivity,
    # 0.016616 dex per kJ/mol (Al2 pressure). Certified 2327-3000 K: the
    # Crochetet Cp continuation reaches 2.126 kJ/mol at 3000 K, and
    # 2.126 * 0.016616 = 0.0353 dex, rounded up to 0.036. Corroborated
    # 1801-2327 K: Paradis is the widest, at 2.550 kJ/mol at 1801 K, and
    # 2.550 * 0.016616 = 0.0424 dex, rounded up to 0.05. The Paradis 2.6 kJ/mol
    # root is at 1796.2 K; 1801 K is a conservative inclusive edge. Hallstedt's
    # independent liquid is +1.45 kJ/mol at 1800 K, inside both bounds.
    # Low-window envelope: 9.84 kJ/mol * 0.0166 dex/kJ (Al2) = 0.164 dex,
    # rounded up to 0.17; the corresponding worst pO2 band is 0.109 dex.
    # Above Crochetet's measured Cp endpoint, its continuation differs by at
    # most 2.13 kJ/mol through 3000 K; 2.13 * 0.016616 = 0.0354 dex,
    # rounded up to 0.036 for calculated corroboration.
    LiquidReferenceRecord("Al2O3(l)", LiquidReferenceAuthority.CERTIFIED, (2327, 2873), 0.036, "dex log10 pressure",
        (_e("JANAF Al-100/Al-101", 1998, "liquid from the 2327 K melting point; liquid enthalpy fit from 2500 K"),
         _e("Crochetet et al.", 2026, "aerodynamic-levitation drop calorimetry measured liquid Cp 168.6 +/- 9.6 J/mol/K over 2327-2873 K; DOI 10.1016/j.oceram.2026.101029")),
        (AuthorityWindow(LiquidReferenceAuthority.CORROBORATED_CALCULATION, (1801, 2327), 0.05,
            "dex log10 pressure", (
                _e("Hallstedt", 1992, "independent CALPHAD liquid Al2O3 Gibbs energy that does not reuse the reference liquid Cp; within 1.5 kJ/mol of the reference at 1800-2327 K; corroborated by independent calculation, not measured; DOI 10.1111/j.1151-2916.1992.tb04216.x"),
                _e("Paradis, Ishikawa, Saita & Yoda", 2004, "electrostatic levitation; liquid Cp estimated over 2120-2450 K assuming a constant hemispherical emissivity of 0.8, with a calculated fusion enthalpy of 109.0 kJ/mol; continued below 2120 K it stays within 2.6 kJ/mol of the reference down to 1801 K; DOI 10.1143/JJAP.43.1496"),
                _e("Crochetet et al.", 2026, "liquid Cp measured above the melting point, continued below it, stays within 1.6 kJ/mol of the reference down to 1801 K; DOI 10.1016/j.oceram.2026.101029"),
            )),
         AuthorityWindow(LiquidReferenceAuthority.CORROBORATED_CALCULATION, (2873, 3000), 0.036,
            "dex log10 pressure", (
                _e("JANAF Al-100/Al-101", 1998, "assessed liquid fit continued above the measured heat-capacity range"),
                _e("Crochetet et al.", 2026, "measured liquid Cp continued above 2873 K; calculated continuation differs from the reference by at most 2.13 kJ/mol through 3000 K; DOI 10.1016/j.oceram.2026.101029"),
            )),
         AuthorityWindow(LiquidReferenceAuthority.EXTRAPOLATED, (1323, 1801), 0.17,
            "dex log10 pressure", (
                _e("Hallstedt", 1992, "CALPHAD liquid continued below its fitted range; the low-temperature continuations spread by up to 9.84 kJ/mol at 1323 K, so this band is a model envelope, not corroboration; DOI 10.1111/j.1151-2916.1992.tb04216.x"),
                _e("Paradis, Ishikawa, Saita & Yoda", 2004, "measured-range Cp continued far below 2120 K; part of the model envelope; DOI 10.1143/JJAP.43.1496"),
                _e("Crochetet et al.", 2026, "measured-range Cp continued far below 2327 K; part of the model envelope; DOI 10.1016/j.oceram.2026.101029"),
                _e("Gurvich via NASA Glenn CEA", 2002, "liquid card from 2327 K continued downward; part of the model envelope; NASA/TP-2002-211556"),
            )),)),
    LiquidReferenceRecord("TiO2(l)", LiquidReferenceAuthority.EXTRAPOLATED, None, None, None,
        (_e("JANAF O-044", 1998, "estimated end-to-end: assumed fusion entropy 7.5 cal/mol/K by analogy with V2O4 and estimated liquid Cp 8 cal per g-atom; no measured pure-liquid data"),),
        (AuthorityWindow(LiquidReferenceAuthority.EXTRAPOLATED, (2185, 2473), 0.11,
            "dex log10 pressure", (
                _e("Gurvich via NASA Glenn CEA", 2002, "versus JANAF O-044, the one-sided positive residual is +4.28 to +4.44 kJ/mol (+0.094 to +0.102 dex log10 pressure) on 2185–2473 K; part of a cross-assessment envelope, not a measurement or confidence interval; NASA/TP-2002-211556"),
                _e("Li & Ishigaki (Barin-derived)", 2002, "liquid Gibbs line differs from JANAF O-044 by +2.13 kJ/mol at 2185 K to −0.98 kJ/mol at 2473 K; included in the cross-assessment envelope, not a measurement; DOI 10.1016/S0022-0248(02)01438-0"),
            )),)),
    # With no active Na2O sensitivity rows, use the stoichiometric stress
    # bound at 1300 K: q=1000/(R*T*ln(10))=0.04018 dex/kJ, so pO2 is at
    # most 2q*2.6=0.2089 dex; round outward to 0.21 (not a confidence band).
    LiquidReferenceRecord("Na2O(l)", LiquidReferenceAuthority.CORROBORATED_CALCULATION, (1405.2, 2473), 0.21, "dex log10 pressure",
        (_e("Richet, Bottinga & Tequi", 1984, "Na2O partial-molar Cp of 100.6 J/mol/K derived from sodium-silicate liquids, not pure-liquid Cp; measured only through 1800 K; continued from the fusion anchor over 1405.2-1800 K, it agrees with the reference within 0.21 kJ/mol; the 1800-2473 K corroboration rests on the Gurvich function, not on this measurement; calculated corroboration; DOI 10.1111/j.1151-2916.1984.tb19155.x"),
         _e("Gurvich via NASA Glenn CEA", 2002, "assessed liquid Cp continuation agrees with the reference within 1.52 kJ/mol through 2473 K; calculated corroboration, not a measurement; NASA/TP-2002-211556"),
         _e("Chase", 1998, "JANAF Na-013 assessed liquid function and 1405.2 K fusion anchor; the table is not a measured pure-liquid Cp interval; DOI 10.18434/T42S31"),
         _e("Smith et al.", 2017, "reports measured pure-Na2O fusion enthalpy 47.7 +/- 2.5 kJ/mol and melting point 1405.2 +/- 4 K; DOI 10.1016/j.jct.2017.04.003")),
        (AuthorityWindow(LiquidReferenceAuthority.EXTRAPOLATED, (1300, 1405.2), 0.21,
            "dex log10 pressure", (
                _e("JANAF Na-013", 1998, "supercooled liquid continuation below the measured 1405.2 K melting point; 0.21 dex is the stoichiometric 2.6 kJ/mol stress bound, not a confidence interval; DOI 10.18434/T42S31"),
                _e("Smith et al.", 2017, "reports the pure-sample 1405.2 K melting point and 47.7 +/- 2.5 kJ/mol fusion enthalpy; no pure-liquid Cp interval is measured; DOI 10.1016/j.jct.2017.04.003"),
            )),)),
)
_BY_SPECIES = MappingProxyType({record.species: record for record in _RECORDS})


def liquid_reference_records() -> tuple[LiquidReferenceRecord, ...]:
    """Return the immutable liquid-reference authority records."""
    return _RECORDS


def _contains(bounds: tuple[float, float] | None, T_K: float) -> bool:
    return bounds is not None and bounds[0] <= T_K <= bounds[1]


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
    active_window = None
    if not _contains(record.certified_T_K, T_K):
        active_window = next(
            (window for window in record.additional_windows if _contains(window.T_K, T_K)),
            None,
        )
    if active_window is not None:
        authority, band_value, band_unit, evidence_ref = (
            active_window.authority, active_window.band_value,
            active_window.band_unit, active_window.evidence_ref,
        )
    active_T_K = active_window.T_K if active_window else record.certified_T_K
    scope = evidence_ref[0].composition_scope if evidence_ref else None
    in_window = _contains(active_T_K, T_K)
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
