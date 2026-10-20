# Liquid functions from solid references

`tools/liquid_from_solid.py` builds an apparent liquid Gibbs function from a
crystal JANAF table and tiered inputs. It anchors enthalpy at fusion with
`H_cr(T_fus) + T_fus ΔS_fus`, anchors entropy with `S_cr(T_fus) + ΔS_fus`,
then integrates the selected constant liquid heat capacity. It evaluates
`G_app = ΔfH°298 + (H − H298) − T S`; values below fusion are labelled
`supercooled_extrapolation`. Condensate-form rows use the existing fitter and
runtime Gibbs evaluator.

## Input tiers and provenance

`data-src/liquid-from-solid-inputs.json` stores each selected value, tier,
source, references, units, and spread. Alternatives are data rows rather than
species-specific branches in the construction code.

Each input row also uses `fit_residual`, `accuracy_band`, `band_basis`, and
`independent_check`. Input spreads remain source provenance; a composed liquid
Gibbs band belongs to the consumer row. Band basis kinds are
`source_uncertainty`, `measured_construction`, `independent_measurement`,
`fit_only`, and `none`. Values state their own units (the construction records
use J mol⁻¹ or kJ mol⁻¹ as declared), and unavailable checks use
`independent_check.status: not_found` with a reason. See
[`provenance-schema.md`](provenance-schema.md) for field shapes and units.

| Input | Tier 1 | Tier 2 | Tier 3 |
| --- | --- | --- | --- |
| Fusion temperature | JANAF crystal-to-liquid marker or assessed value | Cited alternative assessment with uncertainty | Extrapolated liquidus for incongruently melting or decomposing solids; band is the range of cited hypothetical constructions |
| Fusion entropy | JANAF liquid-minus-crystal entropy or a selected direct primary measurement | Mean of measured/assessed members in a declared oxide or homologue family | Leave-one-out Richards-type mean per formula atom, including oxygen; red fallback |
| Liquid heat capacity | JANAF liquid Cp at the marker | Stoichiometric sum of cited oxide partial-molar liquid Cp values | Crystal Cp at fusion carried over; red fallback |

## Tier-4 ZnO and CdO constructions

ZnO and CdO have no stable 1-bar liquid table. Their optional gas channels use
caller-supplied parent activities; OpenIMCC supplies no activity model or
activity coefficients. ZnO(l) is built from TSIV Volume V, Table 1603, which
prints crystal nodes through 2250 K and estimates fusion at 2250 ± 30 K. A
quadratic Cp(T) fit to the final five crystal nodes is integrated from 2250 K
to the 2280 K upper bound. The largest fit residual is 0.000335 J/(mol K);
holding that residual across 30 K and propagating `ΔH − TΔS` gives a 0.00472
J/mol crystal-extension band over 1200–3000 K. The TSIV Phi uncertainty and
fusion-input spreads are added to the per-row band. The runtime ref flags this
crystal extension and the tier-4 liquid construction.

CdO(l) uses the Pankratz Bulletin 672 Table 102 crystal Cp expression, printed
through 1500 K, and extends that expression to the hypothetical fusion range.
The continuation band compares that integrated expression with holding the
last printed Cp constant; the liquid row also carries the measured
rocksalt-monoxide fusion-entropy and liquid-Cp family ranges. The estimated
CdO liquidus is not a claim that CdO has a stable 1-bar melt. Both the crystal
Cp extension and liquid construction are flagged in the runtime result.

Family tiers record their training members and citations. The K₂O entropy tier
uses the mean of Na₂O 33.95 J/(mol K) and the Lamoreaux–Hildenbrand K₂O value
33.53 J/(mol K), giving 33.74 J/(mol K); its half-range spread is 0.21 J/(mol
K). Where the cited partial-molar table does not cover every oxide component,
the tier-2 Cp value is unavailable rather than inferred.

K₂O's selected fusion temperature is 1013 K from the Gurvich lineage in the
NASA Glenn CEA K₂O(l) interval (McBride, Zehe, and Gordon, NASA TP-2002-211556;
Gurvich et al., *Thermodynamic Properties of Individual Substances*, vol. IV,
pt. 1, p. 377 and pt. 2, p. 406); it is also listed in the TKV database
(Gurvich lineage, with no further bibliography). The alkali M₂O trend—Li₂O
1711 K (Ortman and Larsen), Na₂O 1405.2 K (NIST-JANAF/DTA), Rb₂O 778 K and
Cs₂O 763–768 K (Gurvich)—supports this value over Lamoreaux and Hildenbrand's
1190 ± 50 K estimate (1984, Table
1, p. 153). Natola and Touzain measured 919 ± 5 K in Pyrex (1970, *Canadian
Journal of Chemistry* 48, 1955–1958); silica contamination makes this a
one-sided lower bound. The selected K₂O temperature band is therefore
[919, 1190] K, with 919 K as the lower bound. Its tier-3 formula entropy is
27.497678907 J/(mol K), equal to three atoms times the mean fusion entropy of
9.165892969 J/(mol K) per atom across all 48 pairs in the declared census.
K₂O is not in that census, so it
is excluded by construction; this is not a leave-one-out estimate for K₂O. It
is red-flagged because the rule is known to perform poorly.

The Tier 3 temperature rung records a hypothetical fusion temperature for an
incongruently melting or decomposing solid. Its band uses the minimum and
maximum of cited constructions and propagates both endpoints through the
crystal interpolation. Hematite is a literature example to check: Robie,
Hemingway, and Fisher (1978), USGS Bulletin 1452, p. 165, gives a nominal
metastable 1895 K; JANAF reports decomposition at 1735 K; Sugawara and Akaogi
(2004, DOI 10.1016/S0016-7037(03)00202-3) report 133.4 ± 10.5 kJ/mol fusion at
1895 K. Those sources were not verified first-hand here, so the hematite
temperatures are recorded as a reference example, not a sourced band. A
synthetic Fe₂O₃ construction test verifies endpoint propagation.

The partial-molar Cp values come from [Navrotsky (1995)](https://doi.org/10.2138/rmg.1995.32.5),
*Energetics of Silicate Melts*, Table 3, p. 130, which cites [Stebbins,
Carmichael and Moret (1984)](https://doi.org/10.1007/BF00381840) and [Lange
and Navrotsky (1992)](https://doi.org/10.1007/BF00310746). The original 1984
and 1992 tables were not available for first-hand page-level verification, so
every transcribed component is marked `secondary_transcription_unverified_primary`; the values
are not described as first-hand checked. The source values and reported
uncertainties are retained per component. For K₂O the tier-2 Cp is
`(97.0 + 98.5) / 2 = 97.75 J/(mol K)` and uses the larger reported uncertainty,
5.5 J/(mol K).

## JANAF pair set

The census includes the 48 selected oxygen-bearing simple oxides, silicates,
aluminates, and borates with both crystal and liquid tables and a usable
transition marker. It excludes tables for compounds containing H, C, N, F,
Cl, P, or S and polyborates with more than four B atoms per formula. The full
selected list is below; the family column names the declared entropy systematic
and “—” means no family estimate is available.

| Formula | Crystal | Liquid | Entropy family |
| --- | --- | --- | --- |
| Al₂O₃ | Al-096 | Al-100 | corundum sesquioxide |
| B₂O₃ | B-095 | B-096 | — |
| BaO | Ba-034 | Ba-035 | monoxide homologues |
| CaO | Ca-027 | Ca-028 | monoxide homologues |
| Cr₂O₃ | Cr-014 | Cr-015 | corundum sesquioxide |
| Cu₂O | Cu-019 | Cu-020 | — |
| FeO | Fe-018 | Fe-019 | monoxide homologues |
| K₂B₄O₇ | B-115 | B-116 | alkali tetraborates |
| K₂SiO₃ | K-014 | K-015 | alkali metasilicates |
| KBO₂ | B-064 | B-065 | alkali metaborates |
| Li₂B₄O₇ | B-118 | B-119 | alkali tetraborates |
| Li₂O | Li-014 | Li-015 | alkali M₂O |
| Li₂Si₂O₅ | Li-031 | Li-032 | alkali disilicates |
| Li₂SiO₃ | Li-020 | Li-021 | alkali metasilicates |
| Li₂TiO₃ | Li-023 | Li-024 | — |
| LiAlO₂ | Al-068 | Al-069 | — |
| LiBO₂ | B-068 | B-069 | alkali metaborates |
| Mg₂SiO₄ | Mg-028 | Mg-029 | — |
| Mg₂TiO₄ | Mg-031 | Mg-032 | — |
| MgAl₂O₄ | Al-089 | Al-090 | — |
| MgO | Mg-008 | Mg-009 | monoxide homologues |
| MgSiO₃ | Mg-012 | Mg-013 | — |
| MgTi₂O₅ | Mg-022 | Mg-023 | — |
| MgTiO₃ | Mg-015 | Mg-016 | — |
| MoO₃ | Mo-014 | Mo-015 | group-6 trioxides |
| Na₂B₄O₇ | B-122 | B-123 | alkali tetraborates |
| Na₂O | Na-012 | Na-013 | alkali M₂O |
| Na₂Si₂O₅ | Na-028 | Na-029 | alkali disilicates |
| Na₂SiO₃ | Na-016 | Na-017 | alkali metasilicates |
| NaBO₂ | B-074 | B-075 | alkali metaborates |
| Nb₂O₅ | Nb-016 | Nb-017 | metal pentoxides |
| NbO | Nb-008 | Nb-009 | — |
| NbO₂ | Nb-012 | Nb-013 | rutile-type dioxides |
| PbO | O-006 | O-007 | — |
| SiO₂ | O-035 | O-038 | — |
| SrO | O-013 | O-014 | monoxide homologues |
| Ta₂O₅ | O-077 | O-078 | metal pentoxides |
| Ti₂O₃ | O-059 | O-060 | corundum sesquioxide |
| Ti₃O₅ | O-081 | O-082 | — |
| Ti₄O₇ | O-089 | O-090 | — |
| TiO | O-019 | O-020 | monoxide homologues |
| TiO₂ | O-043 | O-044 | rutile-type dioxides |
| V₂O₃ | O-062 | O-063 | corundum sesquioxide |
| V₂O₄ | O-073 | O-074 | rutile-type dioxides |
| V₂O₅ | O-084 | O-085 | metal pentoxides |
| VO | O-023 | O-024 | monoxide homologues |
| WO₃ | O-065 | O-066 | group-6 trioxides |
| ZrO₂ | O-052 | O-053 | — |

The 96 records retain their JANAF source SHA-256 and neutral extractor name
`openimcc-janaf-vendor/1.0`; copied records have their local cache-path field
removed. Eighteen records were already present and 78 were copied from the
supplied corpus, including canonical replacements for six earlier records.
The source hashes were checked against the corpus records. The Mg-012 and
Na-016 text files now match their recorded SHA-256 after removing an extra
terminal newline. The redundant text copies for K-014, K-015, Mg-013, and
Na-017 are omitted because their bytes do not match their recorded hashes;
their YAML records remain authoritative.

## Measured construction bands

Each tier was evaluated against the corresponding JANAF liquid table at
`T_fus`, `T_fus + 300 K`, and `T_fus + 800 K`. Tier 2 and tier 3 estimates
exclude the tested pair when building their systematic. The Tier 1
fusion-temperature, fusion-entropy, and liquid-Cp bands share the same
measured-value control, so they use the row being validated. Three liquid
tables end before `T_fus + 800 K`; they are omitted only at that offset. Each
input's row band is its measured tier band plus that row's propagated input
spread. Family entropy spread is half the training-member range; additive Cp
spread conservatively sums the cited component uncertainties.

Cells below give maximum / RMS absolute error in kJ/mol. Sample counts are for
`T_fus / +300 / +800 K`.

| Input tier | N | T_fus | +300 K | +800 K |
| --- | ---: | ---: | ---: | ---: |
| Fusion temperature, entropy, and Cp, tier 1 control | 48 / 48 / 45 | 1.155 / 0.167 | 16.996 / 2.461 | 45.375 / 6.799 |
| Fusion entropy, tier 2 family | 32 / 32 / 31 | 0.002 / 0.001 | 7.357 / 2.947 | 19.757 / 8.141 |
| Fusion entropy, tier 3 per formula atom | 48 / 48 / 45 | 1.155 / 0.167 | 15.998 / 5.127 | 42.231 / 13.440 |
| Liquid Cp, tier 2 additive | 15 / 15 / 15 | 1.155 / 0.298 | 16.255 / 4.281 | 40.417 / 11.564 |
| Liquid Cp, tier 3 crystal carry-over | 48 / 48 / 45 | 1.155 / 0.167 | 16.778 / 2.910 | 45.198 / 12.799 |

The paired maximum / RMS errors in dex per metal atom are:

| Input tier | T_fus | +300 K | +800 K |
| --- | ---: | ---: | ---: |
| Fusion temperature, entropy, and Cp, tier 1 control | 0.035569 / 0.005134 | 0.109275 / 0.016426 | 0.246271 / 0.037023 |
| Fusion entropy, tier 2 family | 0.000035 / 0.000014 | 0.133452 / 0.040429 | 0.293746 / 0.087499 |
| Fusion entropy, tier 3 per formula atom | 0.035569 / 0.005134 | 0.227979 / 0.063576 | 0.445472 / 0.123293 |
| Liquid Cp, tier 2 additive | 0.035569 / 0.009184 | 0.104513 / 0.028828 | 0.219363 / 0.067897 |
| Liquid Cp, tier 3 crystal carry-over | 0.035569 / 0.005134 | 0.107876 / 0.020488 | 0.238365 / 0.073963 |

The independent census reports maximum errors of 0.77 / 0.13 / 0.27 kJ/mol
for integrated temperature-dependent `ΔCp`, 0.77 / 0.32 / 6.2 kJ/mol for
constant `ΔCp(T_fus)`, and 0.77 / 6.7 / 37.6 kJ/mol for `ΔCp = 0`. The
integrated-`ΔCp` figures use temperature-dependent liquid/solid heat-capacity
differences, which are not one of these constant-Cp tiers. The `ΔCp = 0`
cross-check is close at +300 K (6.710 versus 6.7) and differs at +800 K
(45.282 versus 37.6): that cross-check integrates the changing crystal Cp
curve, while the tier-3 fallback holds crystal Cp at its fusion value. The
measured-Cp tier also holds an absolute liquid Cp constant, while constant
`ΔCp(T_fus)` retains the changing crystal Cp. The remaining differences also
reflect table selection and interpolation conventions; the
independent census did not include enough method detail to reproduce its
exact pair-by-pair calculation. Its Richards-rule maxima, 24.9 / 58.7 kJ/mol
at +300 / +800 K, are above the formula-atom LOO results here (15.998 /
42.231); both use 48 formulas, but their exact phase-selection and
normalization conventions were not supplied for a stricter comparison.

The Tier 1 control now includes Shornikov's primary MgAl₂O₄ fusion
measurement in place of JANAF Al-090's analogy estimate. Its larger
disagreement with the JANAF liquid table raises the measured +300/+800 K
Tier 1 errors; the fusion-point control remains 1.155 kJ/mol.

The largest fusion-point residual is 1.155 kJ/mol for SiO₂ (O-035/O-038).
Its crystal and liquid `ΔfH°298` values differ by 2.828 kJ/mol, and the
JANAF marker enthalpy increment does not equal the construction's
`T_fus ΔS_fus`; the Gibbs residual therefore does not reduce to the anchor
difference alone.

## K₂O construction check

The tier-2 K₂O construction uses `T_fus = 1013 K`, `ΔS_fus = 33.74 J/(mol K)`,
and `Cp_l = 97.75 J/(mol K)` with crystal table K-012. Its pinned Gibbs values
are −555.309542, −582.045759, −637.722336, −787.694272, −950.015529, and
−1122.178019 kJ/mol at 1200, 1300, 1500, 2000, 2500, and 3000 K. The separate
synthetic mechanism test uses `Cp_l = 104.6 J/(mol K)` and reproduces
−582.301144 kJ/mol at 1300 K and −790.252486 kJ/mol at 2000 K. The 104.6 value
is an explicit synthetic test fixture. The
K₂O row's Tier 3 fallback is the K-012 crystal Cp at fusion, 114.03681
J/(mol K). Its selected fusion temperature, bounds, alternative assessments,
and entropy references are recorded with that row.

The Na₂SiO₃ pair is also exercised through `liquid_from_solid` and the existing
condensate fitter/evaluator as the crystalline-compound example.

## Below-fusion band

For a constructed liquid evaluated below its fusion temperature, constant
`ΔCp = Cp_liquid - Cp_crystal(T_fus)` gives the exact fusion difference

```text
ΔG_fus(T) = ΔH_fus(1 - T/T_fus)
             - ΔCp[(T_fus - T) - T ln(T_fus/T)]
ΔH_fus = T_fus ΔS_fus
```

The band is the measured leave-one-out maximum at `T_fus`, plus the absolute
fusion-entropy, heat-capacity-difference, and fusion-temperature sensitivities
times their selected input spreads. The 48-pair validation census gives a
maximum absolute leave-one-out residual of 54.1349 J/(mol K) for the Tier 3
per-formula-atom fusion-entropy estimator (RMS 17.3520 J/(mol K)). For Tier 3
crystal-Cp carry-over, the maximum absolute residual in `ΔCp` is 170.200
J/(mol K) (RMS 45.2682 J/(mol K)). Those census maxima remain the fallback
spreads where no closer measured homologues apply. The measured Tier 3
leave-one-out Gibbs maximum at `T_fus` is 1.154923 kJ/mol across the same 48
pairs. At `T = T_fus`, `ΔG_fus` and all propagated terms vanish, so the band
equals that measured leave-one-out maximum.

For MnO, the selected entropy and `ΔCp` spreads are the maximum absolute
leave-one-out residuals over the seven measured rock-salt monoxides BaO, CaO,
FeO, MgO, SrO, TiO, and VO: 7.8596 J/(mol K) and 14.0011 J/(mol K). The
48-pair Gibbs anchor at `T_fus` remains 1.154923 kJ/mol. The construction uses
the homologue maxima, not the census-wide maxima or RMS values.

Above fusion, an association species uses the Tier 3 leave-one-out entropy and
heat-capacity Gibbs columns linearly interpolated in `T - T_fus` over that
species' evaluation domain. The same residuals set the construction input
spreads, so they are not added a second time. The independent fusion-temperature
interval, source thermochemistry uncertainty, and condensate-fit residual are
added once to the species band. Reaction bands then sum each participating
species band once, with its stoichiometric coefficient.
