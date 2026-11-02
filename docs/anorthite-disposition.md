# Anorthite association: Hess thermochemistry and CAS activities

The association reaction in the melt model is

```text
CaO(l) + Al₂O₃(l) + 2 SiO₂(l) ⇌ CaAl₂Si₂O₈(l)
log₁₀ K = A + B/T
```

The two recorded routes differ only in the constant term. The older v4 row is
the FC87 Table 3 R12 value. The default v5 row refits `A` against mineral-
composition liquid formation Gibbs energies from an independent Hess cycle,
holding the FC87 `B` fixed. The cycle combines crystal formation functions,
measured anorthite fusion thermochemistry, and liquid heat capacity; no
activity or vapor data were fit targets. The source records note assumed
JANAF CaO(l) and MgO(l) fusion terms with ±16.7 kJ mol⁻¹ per oxide. That is an
input uncertainty, not a row-specific uncertainty band. See the [v4 row](../src/openimcc/data/packs/imcc-sf04-d066-v4.json) and [v5 row](../src/openimcc/data/packs/imcc-sf04-d066-v5.json).

| Route | `A` | `B` (K) | Recorded derivation |
| --- | ---: | ---: | --- |
| FC87, stored in v4 | 2.691357 | 5209.541032 | FC87 Table 3 R12; Appendix 1 uses anorthite fusion/liquid Cp and solid data ([Fegley & Cameron, 1987](https://doi.org/10.1016/0012-821X(87)90196-8)). |
| Hess refit, default v5 | 1.749738 | 5209.541032 | Fit to formula-composition liquid formation Gibbs energies at 1600, 1800, and 2000 K from the crystal, fusion, and liquid-Cp chain; crystal inputs follow Berman (1988) and Robie & Hemingway (1995), with anorthite fusion thermochemistry from Richet & Bottinga (1984). |

The Hess reaction energy is
`ΔrG° = G°[CaAl₂Si₂O₈(l)] − G°[CaO(l)] − G°[Al₂O₃(l)] − 2G°[SiO₂(l)]`,
with `log₁₀K = −ΔrG°/(RT ln 10)`. At the four requested temperatures, the
FC87 row makes association 8.74 times stronger than the Hess-refit row. The
Gibbs-energy difference grows with temperature because `B` is shared and the
`A` offset is constant. This is a comparison of the two anorthite rows,
independent of the other pack changes.

| T (K) | log₁₀ K, FC87 row | log₁₀ K, Hess row | FC87 − Hess (dex) | ΔrG°(FC87) − ΔrG°(Hess) (kJ mol⁻¹) |
| ---: | ---: | ---: | ---: | ---: |
| 1700 | 5.7558 | 4.8142 | 0.9416 | −30.65 |
| 1873 | 5.4727 | 4.5311 | 0.9416 | −33.76 |
| 2000 | 5.2961 | 4.3545 | 0.9416 | −36.05 |
| 2473 | 4.7979 | 3.8563 | 0.9416 | −44.58 |

## Effect in CAS melts

I isolated the anorthite row by copying v5 to a scratch candidate pack and
replacing only the `A` and `B` coefficients for `CaAl₂Si₂O₈` with the v4
values. All primary comparisons below run from that FC87-row substitution in
otherwise-v5 to the default v5 Hess row. The secondary full-pack column shows
v4-pack to v5-pack changes and includes the effects of other changed
associations, including CaSiO₃; it does not isolate anorthite.

Baird's Glasgow thesis reports six CaO–Al₂O₃–SiO₂ melts with about 20 wt%
Al₂O₃ at 1500 °C (1773.15 K). The activities below use three compositions
across that silica range and are model outputs on the liquid parent-oxide
standard. Baird's reported silica activity uses the cristobalite reference.
At this temperature the liquid-reference conversion is
`log₁₀(a_liq) = log₁₀(a_cristobalite) − 0.028852`, or
`a_liq = 0.9357 a_cristobalite`. The numerical transcriptions from Table XII
used below could not be verified against the scan; data-based residuals and
pressure scalings are therefore provisional.

Both comparison temperatures, 1773 K here and 1905 K below, are within the
packs' adopted 1700–3000 K exercise domain but below FC87's 2500–3500 K
paper-demonstrated span. These are model-continuation comparisons, not FC87
measurements at those temperatures.

| Baird composition (wt% CaO / SiO₂ / Al₂O₃) | a(SiO₂), FC87 row in v5 → v5 Hess row (Δdex) | a(CaO), same direction (Δdex) | a(Al₂O₃), same direction (Δdex) | Full-pack v5−v4 Δdex (Si / Ca / Al; secondary) |
| --- | ---: | ---: | ---: | ---: |
| CAS1 (46.7 / 33.5 / 20.25) | 0.03991 → 0.04404 (+0.043) | 0.004418 → 0.004087 (−0.034) | 0.002751 → 0.002966 (+0.033) | +0.368 / −0.147 / +0.008 |
| CAS3 (38.8 / 42.0 / 19.5) | 0.17137 → 0.24266 (+0.151) | 0.001145 → 0.000765 (−0.175) | 0.005838 → 0.010494 (+0.255) | +0.237 / −0.045 / −0.003 |
| CAS6 (11.0 / 69.5 / 19.9) | 0.78079 → 0.77997 (−0.0005) | 0.0000196 → 0.0000418 (+0.330) | 0.02829 → 0.04479 (+0.200) | +0.003 / +0.466 / +0.111 |

Across all six Baird compositions, the isolated CaO response is
non-monotonic. The isolated silica shift is small at CAS6 and largest at
CAS3; full-pack shifts therefore cannot be assigned to the anorthite row
alone.

| Oxide activity | Isolated v5−swap range (dex) | Full-pack v5−v4 range (dex; secondary) |
| --- | ---: | ---: |
| a(SiO₂) | −0.0005 to +0.151 | +0.003 to +0.368 |
| a(CaO) | −0.194 to +0.330 | −0.147 to +0.466 |
| a(Al₂O₃) | +0.033 to +0.480 | −0.059 to +0.222 |

For `SiO₂(l) ⇌ SiO(g) + ½ O₂(g)`, p(SiO) is proportional to a(SiO₂) at a
fixed oxygen fugacity. The following values scale the transcribed Baird
inferred pressure by `a_model/a_Baird,liq` at each composition. This is a
matched-buffer model comparison, not a separately calculated gas
equilibrium. Baird measured total CO+SiO pressure and inferred p(SiO) from
the SiC+CO equilibrium.

| Composition | Transcribed p(SiO) (atm) | FC87 row in v5 (atm) | v5 Hess row (atm) | Full-v4-pack value (atm; secondary) |
| --- | ---: | ---: | ---: | ---: |
| CAS1 | 0.005 | 0.00496 | 0.00547 | 0.00234 |
| CAS3 | 0.010 | 0.00954 | 0.01351 | 0.00782 |
| CAS6 | 0.022 | 0.02196 | 0.02194 | 0.02180 |

On the six transcribed Baird silica-activity rows, after converting to the
liquid reference and combining the CAS5 replicate activities by their
geometric mean, the residual `log₁₀(a_model/a_measured)` gives these scores:

| Anorthite row scored | Isolated substitution in v5: MAE (RMSE), dex | Full pack: MAE (RMSE), dex (secondary) |
| --- | ---: | ---: |
| FC87 anorthite row | 0.057 (0.077) | 0.124 (0.164), v4 pack |
| Hess-refit anorthite row | 0.080 (0.116) | 0.080 (0.116), v5 pack |

The isolated FC87-row substitution is closer at CAS1, CAS2, CAS3, and CAS6;
v5 is closer at CAS4 and CAS5. On this unverified transcription, the Baird
activity comparison modestly favors the older anorthite row. The full-pack
score points the other way because it combines several row changes. The
pressure scaling expresses the same silica-activity comparison at Baird's
buffer and is not an independent score. The primary record is [Baird, *The
Measurement of Silica Activities in Liquid Slags*, Glasgow PhD thesis (1956),
Table XII](https://theses.gla.ac.uk/73568/1/10656328.pdf); the university
record catalogues it as a 1957 thesis ([record](https://theses.gla.ac.uk/73568/)).

## Activity and vapor evidence

Stolyarova, Shornikov & Shultz (1996), “High-temperature mass spectrometric
study of the thermodynamic properties of the CaO–Al₂O₃–SiO₂ system,” *High
Temperature and Materials Science* 36, 15–35, is available as audited numerical
evidence. A Russian article by Shornikov, Stolyarova & Shultz (1996), *Tekhnika
i tekhnologiya silikatov* 3(1–2), 8–22, has a similar title; its relationship
to the English paper has not been verified. Table 3 supplies 137 activity cells
on 30 CAS compositions, Table 2 supplies 30 measured SiO pressures paired
with atomic-oxygen pressures, and Figure 2 adds 80 digitized activity points.
Table 3 and Table 2 compositions are normalized from the printed
`(x_A/x_C, x_S)` coordinates; Figure 2 coordinates are digitized from the
authors' graph. The comparisons below use the isolated row swap and liquid-
standard activities.

The paper gives its reference currents “over the individual oxides” without
naming their phases. These scores therefore assume crystalline pure-oxide
references at 1933 K, converted to pure-liquid standards with the model's own
references; this is an inference from the pure-oxide samples, not a printed
standard state. The conversion does not explain the CaO misfit below: it
subtracts 0.840 dex from the measured log activity and reduces the mean
residual from −1.653 to −0.812 dex, so omitting it would make the misfit worse.

### Isolated anorthite-row comparison

The default D066 v5 Hess row is compared with an otherwise identical v5 pack
using only the v4 FC87 value for the `CaAl₂Si₂O₈` A coefficient. All activity
MAE/RMSE values below are in dex on the pure-liquid parent-oxide standard;
the SiO pressure scores are log-pressure residuals. Positive improvements
are reductions in absolute error for the FC87 row.

| Independent evidence | T and composition overlap | Hess MAE / RMSE | FC87 MAE / RMSE | Comparison |
| --- | --- | ---: | ---: | --- |
| Stolyarova, Shornikov & Shultz (1996), Table 3 and author Figure 2 activities (n=217) | 1933 K; CAS x(CaO)=0.240–0.781, x(Al₂O₃)=0.055–0.450, x(SiO₂)=0.094–0.523; exact composition in Table 3 | 0.5167 / 0.7051 | 0.4731 / 0.6415 | FC87 MAE / RMSE lower by 0.0436 / 0.0636 dex |
| Stolyarova, Shornikov & Shultz (1996), Table 2 SiO pressures (n=30) | 1933 K; CAS x(CaO)=0.250–0.769, x(Al₂O₃)=0.055–0.450, x(SiO₂)=0.10–0.50; exact point present | 0.7255 / 0.8466 | 0.5845 / 0.7356 | FC87 lower by 0.1410 / 0.1110 dex; conditional on the source pO-inferred oxygen buffer and from the same laboratory as the activity data |
| Allibert et al. (1981), Table II activities (n=16) | 2060 K; binary CaO–Al₂O₃, x(CaO)=0.352–0.800, x(SiO₂)=0 | 0.4974 / 0.6835 | 0.4974 / 0.6835 | Tie; no silica, so the anorthite association is inactive |
| Cho & Suito (1994), Table 5 activities (n=20) | 1873 K; four derived CAS compositions from Cho & Suito's Table 2 mass-percent analyses (ACS2, ACS4, CAS5, CAS8), x(CaO)=0.305–0.727, x(Al₂O₃)=0.180–0.399, x(SiO₂)=0.093–0.309; no exact anorthite composition | 0.6791 / 0.7983 | 0.6227 / 0.7547 | FC87 lower by 0.0564 / 0.0437 dex |

At the exact Table 3 anorthite composition,
`(x(CaO), x(Al₂O₃), x(SiO₂)) = (0.25, 0.25, 0.50)` at 1933 K, its five
activity cells favor Hess: MAE/RMSE 0.3873/0.5437 dex, versus 0.4637/0.5742
for FC87. The matched Table 2 SiO point instead favors FC87, with absolute
residuals 0.2342 versus 0.4059 dex. The aggregate activity MAE differences
are 0.0436 dex for Stolyarova and 0.0564 dex for Cho–Suito, small beside the
roughly 0.5-dex overall activity miss: Stolyarova MAEs are 0.47–0.52 dex,
Allibert is 0.497 dex, and Cho–Suito is 0.62–0.68 dex. The SiO result is conditional: the source gives pO, used to infer the oxygen
buffer. It is neither an independent test of oxygen fugacity nor an
independent-laboratory confirmation.

The claim that the Table 3 CaO residual is an outlier is supported under
either row. Across its 30 paired compositions, the CaO absolute residual
exceeds both other oxide residuals at 25/30 compositions for Hess and 25/30
for FC87; it exceeds Al₂O₃ at 26/30 for Hess and 25/30 for FC87, and exceeds
SiO₂ at 25/30 for both. The median paired excess over the mean absolute
Al₂O₃/SiO₂ residual is 0.776 dex for Hess and 0.582 dex for FC87. These are
paired comparisons within one source, not independent laboratory
replications.

For one laboratory's Table 3 measurements across 30 unique compositions at
1933 K, a residual regression on NBO/T with `x(Al₂O₃)` controlled gives
significant, oxide-specific slopes:
CaO +0.78 (95% CI +0.60 to +0.96), SiO₂ −1.01 (−1.29 to −0.73), and Al₂O₃
−0.38 (−0.62 to −0.15) dex per NBO/T. At fixed T and P the Gibbs–Duhem
relation, `Σᵢ xᵢ d ln aᵢ = 0`, couples the composition trends of all three
components ([IUPAC definition](https://goldbook.iupac.org/terms/view/15329/html)),
so these slopes are compatible with coupled composition-dependent errors in the
solution model, but this regression does not establish that: the signs of two
fitted slopes are not required by Gibbs–Duhem, and regression coefficients are
not local activity differentials. This is a diagnostic, not a correction, and
not a Gibbs–Duhem integration test.

Stolyarova, Lopatin & Stolyar (2003) measured silica activities by
high-temperature mass spectrometry at 1905 ± 10 K along the x(SiO₂)=0.5 and
0.6 joins, with CaO:Al₂O₃ mole ratios from 0.25 to 4. Their abstract reports
that activity remains constant along each join. Using the isolated row swap,
the model's calculated activity span falls from 0.308 to 0.241 dex at
x(SiO₂)=0.5 and from 0.153 to 0.121 dex at x(SiO₂)=0.6 when the FC87 row is
replaced by the Hess row. The full-v4-pack spans are shown only as secondary
context. This qualitative shape check favors the flatter v5 Hess-row trend,
but the accessible abstract does not provide absolute activities for a
level-based residual.

| x(SiO₂) | Isolated span: FC87 row in v5 → v5 Hess row (dex) | Full v4 pack span (dex; secondary) |
| ---: | ---: | ---: |
| 0.5 | 0.308 → 0.241 | 0.374 |
| 0.6 | 0.153 → 0.121 | 0.174 |

The paper is “Determination of the SiO₂ Activity in Melts of the
CaO–Al₂O₃–SiO₂ System by High-Temperature Mass Spectrometry,” *Glass Physics
and Chemistry* 29, 451–455 (2003),
[doi:10.1023/A:1026382828677](https://doi.org/10.1023/A:1026382828677).

Baird provides a second-laboratory CAS cross-check, but not a direct
measurement of p(SiO): the measured quantity is total CO+SiO pressure, and
the SiO portion is inferred using a SiC equilibrium. The Stolyarova 1996
SiO pressures are from the same laboratory as its activity measurements and
are conditional on an oxygen buffer inferred from pO. I found no independent
second-laboratory CAS study that directly measures p(SiO) over matched
compositions and temperatures.

## Calorimetric evidence above 1800 K

* Weill, Stebbins, Hon & Carmichael (1980) measured high-temperature liquid
  and glass anorthite enthalpies by diphenyl-ether drop calorimetry. Their
  article reports liquid enthalpy at 1830 K of 130.4 ± 1.2 kcal mol⁻¹,
  fusion enthalpy 32.4 ± 2.1 kcal mol⁻¹ (135.6 ± 8.8 kJ mol⁻¹; the paper
  states the enthalpy limits are two standard deviations), and average
  supercooled-liquid Cp of 102 ± 2 cal mol⁻¹ K⁻¹ from Tg≈1086 K to 1830 K.
  A contemporaneous AGU abstract reports direct liquid enthalpy measurements
  from 1832–1882 K and prints 427 ± 4 J mol⁻¹ K⁻¹ for the average liquid Cp;
  the peer-reviewed article's printed Cp uncertainty converts to ±8 J
  mol⁻¹ K⁻¹, so the two printed precision statements are not identical.
  “The enthalpy of fusion of anorthite,” *Contributions to Mineralogy and
  Petrology* 74, 95–102, [DOI](https://doi.org/10.1007/BF00375493);
  [AGU abstract](https://cmosarchives.ca/Congress_P_A/abstracts_AGU_CMOS1980.pdf).
* Richet & Bottinga (1984) report drop-calorimetry measurements from 900 to
  1850 K for amorphous anorthite and derive glass/liquid Cp and fusion
  thermochemistry. This is an independent high-temperature record, but it
  only extends slightly above the 1830 K melting point. “Anorthite, andesine,
  wollastonite, diopside, cordierite and pyrope: thermodynamics of melting,
  glass transitions, and properties of the amorphous phases,” *Earth and
  Planetary Science Letters* 67, 415–432, [DOI](https://doi.org/10.1016/0012-821X(84)90179-1).
* Stebbins, Weill, Carmichael & Moret (1982) measured enthalpies and Cp of
  glasses and stable liquids along the albite–anorthite join by drop and
  differential scanning calorimetry. Their An100 liquid Cp is derived from
  interpolation of three intermediate compositions and is described as a
  better estimate than one based on pure-substance measurements alone. The
  publisher abstract does not give a numerical upper temperature, so it does
  not establish independent pure-anorthite coverage to 2000 K. “High
  temperature heat contents and heat capacities of liquids and glasses in
  the system NaAlSi₃O₈–CaAl₂Si₂O₈,” *Contributions to Mineralogy and
  Petrology* 80, 276–284, [DOI](https://doi.org/10.1007/BF00371357).
* Kosa et al. (1992) measured relative enthalpies in the CaSiO₃–anorthite–
  gehlenite system by high-temperature drop calorimetry combined with
  solution calorimetry, and derive mixing/partial-molar enthalpies over
  1600–1950 K. These are useful CAS constraints, but are mixture-relative
  quantities rather than an independent pure-anorthite liquid Gibbs
  function. “Enthalpic analysis of melts in the CaO·SiO₂–CaO·Al₂O₃·2SiO₂–
  2CaO·Al₂O₃·SiO₂ system,” *Geochimica et Cosmochimica Acta* 56, 2643–2655,
  [DOI](https://doi.org/10.1016/0016-7037(92)90350-r).
* Lange & Navrotsky (1992) measured heat capacities of Fe₂O₃-bearing silicate
  liquids, not the pure CaAl₂Si₂O₈ liquid. Richet & Fiquet (1991) studied
  high-temperature mineral heat capacity and premelting, not a pure-anorthite
  liquid Cp series above 1800 K. These works do not provide the missing
  high-temperature pure-An arbitration. Lange & Navrotsky, “Heat capacities
  of Fe₂O₃-bearing silicate liquids,” *Contributions to Mineralogy and
  Petrology* 110, 311–320 (1992), [DOI](https://doi.org/10.1007/BF00310746);
  Richet & Fiquet, “High temperature heat capacity and premelting of minerals
  in the system MgO–CaO–Al₂O₃–SiO₂,” *Journal of Geophysical Research* 96(B1),
  445–456 (1991), [DOI](https://doi.org/10.1029/90JB02172).

I found no independent pure-anorthite liquid enthalpy or Cp series reaching
2000 K or higher. The available pure-An calorimetry anchors the near-liquidus
region and, in Weill et al., reaches 1882 K; Kosa et al. extend relevant
mixture enthalpies to 1950 K. A same-standard pure-An liquid formation Gibbs
energy or direct enthalpy/Cp measurement near 2000 K and above would be a
material new discriminator.

## Why the default remains v5

The source rail is consistency, then primary measurement, then domain match,
then estimates. The default keeps the v5 Hess-refit anorthite row because it
closes the documented formula-molar liquid formation Gibbs energy on the
crystal, fusion, liquid-Cp, and parent-liquid standard-state chain. Activity
data are an independent sensitivity check, not a coefficient-selection
rule: the aggregate Stolyarova and Cho–Suito activity MAEs favor FC87 by
0.0436 and 0.0564 dex, while Allibert ties and the exact-composition Table 3
activity cells favor Hess. The matched SiO point favors FC87. This row
difference is small beside the roughly 0.5-dex activity miss. The Baird
Table XII transcription could not be verified against the scan and remains
provisional; its provisional activity score favors FC87 (0.057 dex MAE
versus 0.080 for Hess), while the 2003 constant-activity result favors v5's
flatter trend. These comparisons show composition-dependent sensitivity,
not a universal correction. The FC87 row remains a genuine published
thermochemical route; its association is 30.6–44.6 kJ mol⁻¹ more favorable
over the requested range than the Hess-refit row. No row-specific confidence
band is available, and the parent liquid-reference inputs retain their
stated uncertainty.

Revisit the default if a primary, same-standard liquid-anorthite formation
Gibbs measurement or higher-temperature enthalpy/Cp series moves the Hess
target toward FC87. Independently audited CAS activities and directly
measured SiO pressures over matched compositions and temperatures, with
reference-state and uncertainty details, would also justify reassessing the
source-rail conclusion; activity-score preference alone does not select a
coefficient.

## Melt notice

The row-sensitivity notice is implemented as `anorthite-row-sensitivity`.
Its solved-speciation gate covers basalts, CMAS, and MgO-free CAS whenever at
least 0.002 (0.2%) of analytical Al₂O₃ is held in CaAl₂Si₂O₈. It
fires on most Ca–Al–Si melts, like the K–Al over-sequestration notice fires on
most K-bearing basalts. The existing `multicomponent-activity-pattern` gate
requires SiO₂, CaO, MgO, and Al₂O₃ each to exceed 1 mol% and therefore does not
cover the MgO-free CAS compositions in these measurements. Its Kume 2000
summary does not include the single-laboratory 1933 K Stolyarova Table 3
slopes or the isolated anorthite-row comparison.

Each entry below is log₁₀(FC87-swap/default) at log₁₀ fO₂ = −8; these are
model-output shifts, not observed measurements:

| Composition | Δlog₁₀ a(Na₂O), 1800 / 2100 K | Δlog₁₀ a(K₂O), 1800 / 2100 K | Δlog₁₀ p(Na), 1800 / 2100 K | Δlog₁₀ p(K), 1800 / 2100 K |
| --- | ---: | ---: | ---: | ---: |
| README basalt | +0.374 / +0.418 | +0.354 / +0.401 | +0.187 / +0.209 | +0.177 / +0.200 |
| Lunar mare basalt | +0.316 / +0.336 | +0.294 / +0.318 | +0.158 / +0.168 | +0.147 / +0.159 |
| Regolith simulant | +0.271 / +0.302 | +0.266 / +0.301 | +0.135 / +0.151 | +0.133 / +0.151 |
| Highland anorthosite | +0.999 / +1.108 | — | +0.499 / +0.554 | — |

The mare basalt uses 45 SiO₂, 3 TiO₂, 10 Al₂O₃, 19 FeO, 10 MgO, 11 CaO,
0.4 Na₂O, and 0.1 K₂O wt%. The regolith simulant uses 47 SiO₂, 15 Al₂O₃,
13 FeO, 9 MgO, 10 CaO, 2 TiO₂, 2.5 Na₂O, and 0.8 K₂O wt%; the highland
anorthosite uses 45 SiO₂, 32 Al₂O₃, 18 CaO, 3 MgO, 1.5 FeO, and 0.5 Na₂O
wt%. The README basalt composition is listed in the example above.

At 1800 K in the README basalt, solved Al₂O₃ held in anorthite rises from
5.0% to 22.8%; the Na–Al share of analytical Na₂O falls from 99.23% to
98.46%, and the K–Al share of K₂O falls from 99.998% to 99.996%.
The released alkali raises its parent activity. At fixed oxygen fugacity the
gas equilibrium gives p(M) ∝ a(M₂O)^(1/2), matching the half-sized log-pressure
shifts in the table.

No alkali-bearing measurement in the evidence summarized here yet
discriminates the two anorthite rows. The known low-K comparison is the
separate `k-al-over-sequestration` notice: the FC87 row swap raises predicted
p(K) in Ca-bearing melts, but this remains a sensitivity result, not evidence
for choosing that row. Hastie's Beaver's Bend illite composition lists no
CaO ([Hastie et al. 1981, NBSIR 81-2279, sec. 8.2](https://nvlpubs.nist.gov/nistpubs/Legacy/IR/nbsir81-2279.pdf)), so that sample cannot test the Ca-bearing effect.

The 0.2% gate is a permissive sampled onset, chosen so that no sampled melt
with a maximum parent log-activity shift ≥0.1 dex was missed. On the earlier
1,468-point, non-monotonic alkali-free Stolyarova CAS grid, the lowest such
bound-Al fraction was about 0.2%; the runtime uses the round 0.002 threshold.
This does not make the threshold a physical crossover or an error bound.

In the reproduced boundary-focused FC87-vs-v5 row-swap sweep, CaO was
0.1/0.2/0.5/1/1.5/2 wt%, Al₂O₃ was 0.1/0.2/0.5/1/2/3 wt%, and compositions
were sampled both as CAS and with 0.45 wt% Na₂O plus 0.144 wt% K₂O, balance
SiO₂, at 1700/2000/2400 K (216 conditions). It yielded TP=152, FP=10, FN=0,
TN=54: 10 of 162 firings (6.2%) had shifts below 0.1 dex, and all 152 sampled
shifts at or above 0.1 dex fired. Those false alerts occurred at 0.1–0.2 wt%
CaO and 0.1–1 wt% Al₂O₃, including the 0.1 CaO / 1 Al₂O₃ / 0.45 Na₂O /
0.144 K₂O wt% example. A broader 80-condition sweep used CaO 5/10/15/20 wt%, Al₂O₃
5/10/15/20/30 wt%, two alkali-bearing basaltic-like and highland-like
backgrounds, and 1800/2100 K; all 80 firings shifted by at least 0.1 dex.
Across both sweeps the counts were TP=232, FP=10, FN=0, TN=54. Rounding from
the previous extra-precision threshold changed no firing in these sweeps.

This remains a permissive sampled gate: firing means the row may matter, not
that it does, and the gate is not a physical crossover or an error bound. The
notice fires on most Ca–Al–Si melts, like the K–Al over-sequestration notice
fires on most K-bearing basalts. No per-melt leverage solve is done because it
would add a second speciation solve, a cost a disclosure does not justify. The
provisional Baird transcription and qualitative 2003 shape comparison remain
caveats. The exact notice text is recorded in `docs/melt-notices.md`. This is
a flag-only disclosure; pack coefficients and calculated activities are
unchanged.
