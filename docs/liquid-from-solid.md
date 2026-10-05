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

| Input | Tier 1 | Tier 2 | Tier 3 |
| --- | --- | --- | --- |
| Fusion temperature | JANAF crystal-to-liquid marker or an assessed value | — | — |
| Fusion entropy | Liquid-minus-crystal entropy at the marker | Mean of measured/assessed members in a declared oxide or homologue family | Leave-one-out Richards-type mean per formula atom, including oxygen; red fallback |
| Liquid heat capacity | JANAF liquid Cp at the marker | Stoichiometric sum of cited oxide partial-molar liquid Cp values | Crystal Cp at fusion carried over; red fallback |

Family tiers record their training members and citations. The K₂O entropy tier
uses the mean of Na₂O 33.95 J/(mol K) and the Lamoreaux–Hildenbrand K₂O value
33.53 J/(mol K), giving 33.74 J/(mol K); its half-range spread is 0.21 J/(mol
K). Where the cited partial-molar table does not cover every oxide component,
the tier-2 Cp value is unavailable rather than inferred.

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
The source hashes were checked against the corpus
records. A fresh byte-for-byte download/hash spot-check against the live NIST
text endpoint could not be completed in this network-restricted environment.

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
| Fusion temperature, entropy, and Cp, tier 1 control | 48 / 48 / 45 | 1.155 / 0.167 | 1.165 / 0.194 | 4.240 / 0.687 |
| Fusion entropy, tier 2 family | 32 / 32 / 31 | 0.002 / 0.001 | 7.357 / 2.946 | 19.756 / 8.139 |
| Fusion entropy, tier 3 per formula atom | 48 / 48 / 45 | 1.155 / 0.167 | 16.463 / 5.276 | 43.473 / 13.854 |
| Liquid Cp, tier 2 additive | 15 / 15 / 15 | 1.155 / 0.298 | 1.674 / 0.864 | 10.777 / 5.144 |
| Liquid Cp, tier 3 crystal carry-over | 48 / 48 / 45 | 1.155 / 0.167 | 6.710 / 1.614 | 45.282 / 11.007 |

The paired maximum / RMS errors in dex per metal atom are:

| Input tier | T_fus | +300 K | +800 K |
| --- | ---: | ---: | ---: |
| Fusion temperature, entropy, and Cp, tier 1 control | 0.035569 / 0.005134 | 0.030486 / 0.004586 | 0.024324 / 0.004785 |
| Fusion entropy, tier 2 family | 0.000047 / 0.000017 | 0.133430 / 0.040422 | 0.293724 / 0.087490 |
| Fusion entropy, tier 3 per formula atom | 0.035569 / 0.005134 | 0.232041 / 0.064309 | 0.454136 / 0.124734 |
| Liquid Cp, tier 2 additive | 0.035569 / 0.009184 | 0.027550 / 0.010219 | 0.066705 / 0.038084 |
| Liquid Cp, tier 3 crystal carry-over | 0.035569 / 0.005134 | 0.042530 / 0.013328 | 0.208794 / 0.064923 |

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
at +300 / +800 K, are above the formula-atom LOO results here (16.463 /
43.473); both use 48 formulas, but their exact phase-selection and
normalization conventions were not supplied for a stricter comparison.

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
requested mechanism test uses `Cp_l = 104.6 J/(mol K)` and reproduces
−582.301144 kJ/mol at 1300 K and −790.252486 kJ/mol at 2000 K. The 104.6 value
is an explicit test override from the requested controller calculation. The
K₂O row's Tier 3 fallback is the K-012 crystal Cp at fusion, 114.03681
J/(mol K). Its assessed fusion temperature and entropy references are recorded
with that row.

The Na₂SiO₃ pair is also exercised through `liquid_from_solid` and the existing
condensate fitter/evaluator as the crystalline-compound example.
