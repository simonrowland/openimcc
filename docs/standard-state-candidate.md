# Standard state and Hg continuation candidate

This is an opt-in pair: load `gas-shomate-standard-state-candidate.csv` and `condensate-standard-state-candidate.csv` together with `load_gas_datapack(gas_path=..., oxide_path=...)`. The default tables and packs are unchanged.

## Construction and conversion

Bulletin 672's [Methods, Conventions, and Symbols](https://stacks.cdc.gov/view/cdc/219421/cdc_219421_DS1.pdf) defines gases as hypothetical ideal gases at unit fugacity and gives pressure in atmospheres. For an ideal-gas species, changing the standard pressure from 1 atm to 1 bar gives `S°(1 bar) = S°(1 atm) + R ln(1.01325)`. Since `G = H - TS`, this lowers the species Gibbs energy by `T R ln(1.01325)` while leaving H and Cp fixed. For a reaction whose converted gas standards have net coefficient `Δνg`, `Δlog10 K = Δνg log10(1.01325) = Δνg × 0.005717`. Here `R ln(1.01325) = 0.109443 J mol⁻¹ K⁻¹`; dividing `RT ln(10)` leaves a dimensionless log value. Te(l)→Te(g), 2Te(l)→Te₂(g), TeO₂(l)→TeO₂(g), and SeO₂(l)→SeO₂(g) each have `Δνg=+1`, so the sign is positive: candidate vapor pressure is 1.01325 times the 1 atm value, or +0.005717 dex.

The gas candidate starts from the shipped `gas-shomate.csv`, and the oxide candidate starts from `condensate.csv`. Only the Shomate entropy constant G changes, by +0.109443159414 J mol⁻¹ K⁻¹, in the documented 1 atm gas rows. This changes 54 gas rows (27 species, two intervals each) and the Hg(l) thermochemical row. Following the existing opt-in condensate candidate convention, the NiO(l) and CoO(l) supercooled row labels use base TSIV table IDs in `Ref`, with endpoints retained in `source_label`; their thermochemical coefficients are unchanged. A test explicitly subtracts the entropy shift to recover each 1 atm source row.

| Candidate file | SHA-256 |
| --- | --- |
| `gas-shomate-standard-state-candidate.csv` | `d16227a5cc6740cfba3262b7e27e94baf962c0301910ad7062cff01c433716f5` |
| `condensate-standard-state-candidate.csv` | `92fb8701bffb2e2e10fbeaaf47d5083dab696cf151c55c1335aa1a019ac3e10b` |

## Every changed gas row

Each listed row changes `log10 p` by +0.005717 dex at temperatures inside that row interval. A dash means the temperature is outside that row interval. Endpoints are shown as part of the row interval.

| Species | Source row | Interval | 1500 K | 2000 K | 2473 K |
| --- | --- | ---: | ---: | ---: | ---: |
| As(g) | `P672-As-g-CP-EXT-1200` | 1500–3000 K (interval 1) | +0.005717 | +0.005717 | +0.005717 |
| As(g) | `P672-As-g-CP-EXT-1200` | 1200–1500 K (interval 2) | +0.005717 | — | — |
| As2(g) | `P672-As2-g-CP-EXT-1200` | 1500–3000 K (interval 1) | +0.005717 | +0.005717 | +0.005717 |
| As2(g) | `P672-As2-g-CP-EXT-1200` | 1200–1500 K (interval 2) | +0.005717 | — | — |
| As4(g) | `P672-As4-g-CP-EXT-1200` | 1500–3000 K (interval 1) | +0.005717 | +0.005717 | +0.005717 |
| As4(g) | `P672-As4-g-CP-EXT-1200` | 1200–1500 K (interval 2) | +0.005717 | — | — |
| As4O6(g) | `P672-As4O6-g-CP-EXT-1000` | 1500–3000 K (interval 1) | +0.005717 | +0.005717 | +0.005717 |
| As4O6(g) | `P672-As4O6-g-CP-EXT-1000` | 1200–1500 K (interval 2) | +0.005717 | — | — |
| Bi(g) | `P672-T80-CP-EXT-1800` | 1500–3000 K (interval 1) | +0.005717 | +0.005717 | +0.005717 |
| Bi(g) | `P672-T80` | 1200–1500 K (interval 2) | +0.005717 | — | — |
| Bi2(g) | `P672-T81-CP-EXT-1800` | 1500–3000 K (interval 1) | +0.005717 | +0.005717 | +0.005717 |
| Bi2(g) | `P672-T81` | 1200–1500 K (interval 2) | +0.005717 | — | — |
| CoO(g) | `TSIV-1335` | 1500–3000 K (interval 1) | +0.005717 | +0.005717 | +0.005717 |
| CoO(g) | `TSIV-1335` | 1200–1500 K (interval 2) | +0.005717 | — | — |
| MnO(g) | `TSIV-1436` | 1500–3000 K (interval 1) | +0.005717 | +0.005717 | +0.005717 |
| MnO(g) | `TSIV-1436` | 1200–1500 K (interval 2) | +0.005717 | — | — |
| Rh(g) | `P672-Rh-g-CONT-2000` | 2000–3000 K (interval 1) | — | +0.005717 | +0.005717 |
| Rh(g) | `P672-Rh-g` | 1200–2000 K (interval 2) | +0.005717 | +0.005717 | — |
| Ru(g) | `P672-Ru-g` | 1500–3000 K (interval 1) | +0.005717 | +0.005717 | +0.005717 |
| Ru(g) | `P672-Ru-g` | 1200–1500 K (interval 2) | +0.005717 | — | — |
| Sb(g) | `P672-Sb-g-CP-EXT-2000` | 2000–3000 K (interval 1) | — | +0.005717 | +0.005717 |
| Sb(g) | `P672-Sb-g` | 1200–2000 K (interval 2) | +0.005717 | +0.005717 | — |
| Sb2(g) | `P672-Sb2-g-CP-EXT-2000` | 2000–3000 K (interval 1) | — | +0.005717 | +0.005717 |
| Sb2(g) | `P672-Sb2-g` | 1200–2000 K (interval 2) | +0.005717 | +0.005717 | — |
| Sb4(g) | `P672-Sb4-g-CP-EXT-2000` | 2000–3000 K (interval 1) | — | +0.005717 | +0.005717 |
| Sb4(g) | `P672-Sb4-g` | 1200–2000 K (interval 2) | +0.005717 | +0.005717 | — |
| Sb4O6(g) | `P672-Sb4O6-g-CP-EXT-1000` | 1500–3000 K (interval 1) | +0.005717 | +0.005717 | +0.005717 |
| Sb4O6(g) | `P672-Sb4O6-g-CP-EXT-1000` | 1200–1500 K (interval 2) | +0.005717 | — | — |
| Se(g) | `P672-Se-g-CP-EXT-2000` | 2000–3000 K (interval 1) | — | +0.005717 | +0.005717 |
| Se(g) | `P672-Se-g` | 1200–2000 K (interval 2) | +0.005717 | +0.005717 | — |
| Se2(g) | `P672-Se2-g-CP-EXT-2000` | 2000–3000 K (interval 1) | — | +0.005717 | +0.005717 |
| Se2(g) | `P672-Se2-g` | 1200–2000 K (interval 2) | +0.005717 | +0.005717 | — |
| SeO(g) | `P672-SeO-g-CP-EXT-2000` | 2000–3000 K (interval 1) | — | +0.005717 | +0.005717 |
| SeO(g) | `P672-SeO-g` | 1200–2000 K (interval 2) | +0.005717 | +0.005717 | — |
| SeO2(g) | `P672-SeO2-g-CP-EXT-1500` | 1500–3000 K (interval 1) | +0.005717 | +0.005717 | +0.005717 |
| SeO2(g) | `P672-SeO2-g` | 1200–1500 K (interval 2) | +0.005717 | — | — |
| Te(g) | `P672-Te-g-CP-EXT-2000` | 2000–3000 K (interval 1) | — | +0.005717 | +0.005717 |
| Te(g) | `P672-Te-g` | 1200–2000 K (interval 2) | +0.005717 | +0.005717 | — |
| Te2(g) | `P672-Te2-g-CP-EXT-2000` | 2000–3000 K (interval 1) | — | +0.005717 | +0.005717 |
| Te2(g) | `P672-Te2-g` | 1200–2000 K (interval 2) | +0.005717 | +0.005717 | — |
| TeO2(g) | `P672-TeO2-g-CP-EXT-1500` | 1500–3000 K (interval 1) | +0.005717 | +0.005717 | +0.005717 |
| TeO2(g) | `P672-TeO2-g` | 1200–1500 K (interval 2) | +0.005717 | — | — |
| Th(g) | `P672-Th-g-CP-EXT-2500` | 1500–3000 K (interval 1) | +0.005717 | +0.005717 | +0.005717 |
| Th(g) | `P672-Th-g` | 1200–1500 K (interval 2) | +0.005717 | — | — |
| ThO(g) | `P672-ThO-g-CP-EXT-2500` | 1500–3000 K (interval 1) | +0.005717 | +0.005717 | +0.005717 |
| ThO(g) | `P672-ThO-g` | 1200–1500 K (interval 2) | +0.005717 | — | — |
| ThO2(g) | `P672-ThO2-g-CP-EXT-2500` | 1500–3000 K (interval 1) | +0.005717 | +0.005717 | +0.005717 |
| ThO2(g) | `P672-ThO2-g` | 1200–1500 K (interval 2) | +0.005717 | — | — |
| Tl(g) | `P672-Tl-g-CP-EXT-2000` | 2000–3000 K (interval 1) | — | +0.005717 | +0.005717 |
| Tl(g) | `P672-Tl-g` | 1200–2000 K (interval 2) | +0.005717 | +0.005717 | — |
| Tl2O(g) | `P672-Tl2O-g-CP-EXT-1000` | 1500–3000 K (interval 1) | +0.005717 | +0.005717 | +0.005717 |
| Tl2O(g) | `P672-Tl2O-g-CP-EXT-1000` | 1200–1500 K (interval 2) | +0.005717 | — | — |
| Y(g) | `P672-Y-g-CP-EXT-2500` | 1500–3000 K (interval 1) | +0.005717 | +0.005717 | +0.005717 |
| Y(g) | `P672-Y-g` | 1200–1500 K (interval 2) | +0.005717 | — | — |

## Changed condensate rows

| Species and interval | Candidate change | Δlog10 p at 1500 K | 2000 K | 2473 K |
| --- | --- | ---: | ---: | ---: |
| Hg(l), 1200–3000 K | Fit of paired BU-2587 H/S/Cp branch; continuation flag retained. | −2.744105 | −3.204276 | −3.505621 |
| NiO(l), 1200–2228 K | `Ref` is `TSIV-2026`; endpoint 2228 K is in `source_label`. Thermochemical values unchanged. | 0 | 0 | 0 |
| CoO(l), 1200–2090 K | `Ref` is `TSIV-1334`; endpoint 2090 K is in `source_label`. Thermochemical values unchanged. | 0 | 0 | 0 |

## Hg(l) continuation

BU-2586 covers Hg(l) only through 629.839 K and states that above 630 K Hg is an ideal monatomic gas. Its paired BU-2587 card is that branch; its Gibbs energy matches BU-2586 at the source limit. The candidate fits BU-2587 H/S/Cp over 1200–3000 K as the flagged Hg(l) reference continuation. Its Cp is nonnegative across the range: the test evaluates quartic Cp at both endpoints and every real internal extremum; sampled values are 21.059 J mol⁻¹ K⁻¹ at 1200 K, 20.669 at 1800 K, 20.860 at 2200 K, 20.992 at 2473 K, and 18.602 at 3000 K. The shipped default row has Cp(3000 K)=−166.9 J mol⁻¹ K⁻¹.

| T (K) | Default Hg partial pressure (bar) | Candidate (bar) | Δlog10 p |
| ---: | ---: | ---: | ---: |
| 1200 | 187.295292 | 0.999859 | −2.272588 |
| 1800 | 1121.248393 | 0.999889 | −3.049750 |
| 2200 | 2175.942042 | 1.000065 | −3.337619 |
| 2473 | 3203.730399 | 1.000080 | −3.505621 |

## Latent source-pressure audit

| Source row(s) | Local source evidence | Candidate action | Pressure effect or bound |
| --- | --- | --- | --- |
| Bulletin 672 gas rows, including Te, Te₂, TeO₂, SeO₂ | `pankratz-bulletin-672.json`; `data-src/published-thermo/PROVENANCE.yaml` records Bulletin 672 Methods, Conventions, and Symbols: ideal gases at unit fugacity, pressure in atm. | Shift 54 rows by the 1 atm to 1 bar entropy term. | +0.005717 dex for each one-gas-product vaporization reaction (`Δνg=+1`). |
| TSIV-1436 MnO(g), TSIV-1335 CoO(g) | Their table JSON and pages do not repeat p°. The TSIV publisher's general [thermal-constants conventions](https://www.chem.msu.ru/cgi-bin/tkv.pl?show=params.html) state a 1 atm standard and define gas standards at fugacity 1 atm. | Shift both intervals for each row. | +0.005717 dex (`Δνg=+1`). |
| IVTAN-264 ZnO(g) | `ivtan-table-264.json` and its provenance give no standard pressure. | Unchanged. | ±0.005717 dex row-level ambiguity. |
| PM83-YO; P696-T115 BiO(g); P696-T191 CdO(g) | Their source JSON/provenance records do not state the gas standard pressure. | Unchanged. | ±0.005717 dex row-level ambiguity for each. |
| P677-Re-g; Kelley-584-SbO | Their source JSON/provenance records do not state the gas standard pressure. | Unchanged. | ±0.005717 dex row-level ambiguity for each. |
| NUREG-4308-TeO | `nureg-cr-4308-teo.json` identifies TeO(g), Te(g) Gibbs values and the reaction Te(g)+½O₂(g)=TeO(g), but the underlying reference-288 table/page and p° are not identified in source provenance. | Unchanged. | ±0.005717 dex row-level ambiguity. If all standards in that reaction were 1 atm, `Δνg=−½` gives −0.0028585 dex in log K; that assumption is not applied. |

All remaining rows from the same fitter are either in the documented Bulletin 672 class or are explicitly retained as ambiguous above. Candidate construction and the assertions are in `tools/build_gas_tables.py` and `tests/test_gas_tables.py`.
