# Liquid functions from solid references

`tools/liquid_from_solid.py` builds an apparent liquid Gibbs function from a
crystal JANAF table and a row of tiered inputs. It anchors enthalpy to the
crystal value at fusion, adds `T_fus * ΔS_fus`, then integrates liquid heat
capacity. Entropy is anchored by the crystal entropy and `ΔS_fus`; the
resulting apparent Gibbs energy uses the crystal `ΔfH°298` reference. Values
below fusion are labelled `supercooled_extrapolation`.

The input table is `data-src/liquid-from-solid-inputs.json`. Each input records
its selected value and tier, source, spread, units, and alternatives in its
ladder. Each alternative records a tier and source; the tier-2 Cp entry lists
the oxide components and citations, and red fallback entries are flagged.
Fusion entropy tiers are measured values, a stated structural-family mean,
and a red Richards-type per-atom fallback. Liquid heat-capacity tiers are
measured values, additive partial-molar oxide values, and a red crystal-Cp
carry-over. Fusion temperatures use an assessed or measured tier-1 value;
incongruent melting requires an explicitly sourced hypothetical liquidus and
spread.

Each JANAF table record carries its source URL and table id, extraction method,
SHA-256 of the vendored text, extraction tool name, review status, and any
parse ambiguities. The sodium metasilicate pair is captured from the indexed
NIST text tables. The potassium crystal and magnesium liquid tables were
normalized from indexed NIST HTML because their original text downloads were
unavailable; their extraction records say so and hash the normalized files.

## Accessible validation bands

The available leave-one-out validation set has six pairs: Na₂SiO₃, K₂SiO₃,
MgSiO₃, Al₂O₃, CaO, and SiO₂. The alkali-family entropy row holds out one of
the two alkali metasilicates; per-atom entropy predicts from the other pair
rows. Tier-2 Cp uses the cited partial-molar component values; tier-3 carries
crystal Cp at fusion. Cells are maximum / RMS absolute error in kJ/mol at
`T_fus`, `T_fus + 300 K`, and `T_fus + 800 K`.

| Input tier | Pairs | At fusion | +300 K | +800 K |
| --- | ---: | ---: | ---: | ---: |
| Fusion entropy, tier 2 family | 2 | 0.001551 / 0.001361 | 0.778006 / 0.658539 | 1.832280 / 1.735373 |
| Fusion entropy, tier 3 per atom | 6 | 1.154923 / 0.471497 | 8.071990 / 4.144969 | 19.581036 / 10.424638 |
| Liquid Cp, tier 2 additive | 6 | 1.154923 / 0.471497 | 1.052749 / 0.613449 | 4.913977 / 2.769430 |
| Liquid Cp, tier 3 crystal carry-over | 6 | 1.154923 / 0.471497 | 0.922182 / 0.581916 | 6.578776 / 3.183846 |

Dex per metal atom, reported as maximum / RMS on the same sample:

| Input tier | At fusion | +300 K | +800 K |
| --- | ---: | ---: | ---: |
| Fusion entropy, tier 2 family | 0.000022 / 0.000018 | 0.008150 / 0.007055 | 0.014756 / 0.014322 |
| Fusion entropy, tier 3 per atom | 0.035569 / 0.014521 | 0.211237 / 0.091497 | 0.409771 / 0.180945 |
| Liquid Cp, tier 2 additive | 0.035569 / 0.014521 | 0.027550 / 0.012452 | 0.048429 / 0.028353 |
| Liquid Cp, tier 3 crystal carry-over | 0.035569 / 0.014521 | 0.022441 / 0.010385 | 0.054946 / 0.028462 |

These bands are not the requested full JANAF census. The largest fusion-point
residual (1.155 kJ/mol) comes from the SiO₂ pair, whose crystal and liquid
JANAF entries use 298 K formation-enthalpy anchors that differ by 2.828
kJ/mol. The six-pair sample also differs from the independent 48-pair
cross-check in both composition coverage and estimator: here tier-2 Cp uses
fixed literature partial molar values and tier-3 uses crystal Cp at fusion.
Those differences account for the present disagreement with the cross-check;
the full census remains necessary before these sample bands can be treated as
general bands. The K₂O row's selected band adds its tier-2 entropy and Cp
measured bands to its propagated input spread.

The accessible validation pairs and JANAF table ids are:

| Pair | Crystal | Liquid | Provenance |
| --- | --- | --- | --- |
| Na₂SiO₃ | Na-016 | Na-017 | Newly vendored from indexed NIST text |
| K₂SiO₃ | K-014 | K-015 | Newly vendored; crystal normalized from indexed HTML |
| MgSiO₃ | Mg-012 | Mg-013 | Newly vendored; liquid normalized from indexed HTML |
| Al₂O₃ | Al-096 | Al-100 | Existing repository records |
| CaO | Ca-027 | Ca-028 | Existing repository records |
| SiO₂ | O-035 | O-038 | Existing repository records |

The K₂O tier-2 liquid Cp is 97.75 J/(mol K), the arithmetic mean of the cited
97.0 ± 5.1 and 98.5 ± 5.5 values; tier 3 carries the Na₂O(l) JANAF value of
104.6 J/(mol K). The original articles' full tables were not accessible for
first-hand page-level verification. The available Springer records confirm
that the 1984 paper uses an additive partial-molar model and the 1992 paper
fits liquid data with a no-excess-Cp model, but not their tabulated component
values. The values are also listed in Navrotsky (1995), Table 3, p. 130,
DOI 10.2138/rmg.1995.32.5; this is an accessible supporting compilation, not
first-hand verification of those original tables. The JSON provenance
preserves this limitation, so these numbers are provisional until the source
tables can be checked. Both K₂O estimates are
below the K-012 crystal Cp at 1013 K; diagnostics report the negative fusion
ΔCp and say this may mean the crystal's high-temperature Cp runs high or the
liquid estimate is low.

The tier-2 K₂O central construction gives `G_app` of −555.309542,
−582.045759, −637.722336, −787.694272, −950.015529, and −1122.178019 kJ/mol
at 1200, 1300, 1500, 2000, 2500, and 3000 K, respectively. The separate
104.6 J/(mol K) tier-3 mechanism check gives −582.301144 kJ/mol at 1300 K and
−790.252486 kJ/mol at 2000 K, reproducing the requested −582.30 and −790.25
values. The tier-1 K₂O fusion temperature of 1013 K is assessed by the alkali
oxide trend in Lamoreaux and Hildenbrand (1984), DOI 10.1063/1.555706; that
source describes it as an estimate rather than a direct measurement.
