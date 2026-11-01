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
with `log₁₀K = −ΔrG°/(RT ln 10)`. At the four requested temperatures, v4
therefore makes association 8.74 times stronger than v5. The Gibbs-energy
difference grows with temperature because `B` is shared and the `A` offset is
constant.

| T (K) | log₁₀ K, v4 | log₁₀ K, v5 | v4 − v5 (dex) | ΔrG°(v4) − ΔrG°(v5) (kJ mol⁻¹) |
| ---: | ---: | ---: | ---: | ---: |
| 1700 | 5.7558 | 4.8142 | 0.9416 | −30.65 |
| 1873 | 5.4727 | 4.5311 | 0.9416 | −33.76 |
| 2000 | 5.2961 | 4.3545 | 0.9416 | −36.05 |
| 2473 | 4.7979 | 3.8563 | 0.9416 | −44.58 |

## Effect in CAS melts

Baird's Glasgow thesis reports six CaO–Al₂O₃–SiO₂ melts with about 20 wt%
Al₂O₃ at 1500 °C (1773.15 K). The table below shows three compositions across
that silica range. Activities are model outputs on the liquid parent-oxide
standard; each pair is v4 → v5 and each parenthetical number is
`log₁₀(a_v5/a_v4)` in dex. Baird's reported silica activity uses the
cristobalite reference. At this temperature the liquid-reference conversion
is `log₁₀(a_liq) = log₁₀(a_cristobalite) − 0.028852`, or
`a_liq = 0.9357 a_cristobalite`.

Both comparison temperatures, 1773 K here and 1905 K below, are within the
packs' adopted 1700–3000 K exercise domain but below FC87's 2500–3500 K
paper-demonstrated span. These are model-continuation comparisons, not FC87
measurements at those temperatures.

| Baird composition (wt% CaO / SiO₂ / Al₂O₃) | a(SiO₂), v4 → v5 | a(CaO), v4 → v5 | a(Al₂O₃), v4 → v5 |
| --- | ---: | ---: | ---: |
| CAS1 (46.7 / 33.5 / 20.25) | 0.01886 → 0.04404 (+0.368) | 0.005738 → 0.004087 (−0.147) | 0.002913 → 0.002966 (+0.008) |
| CAS3 (38.8 / 42.0 / 19.5) | 0.14054 → 0.24266 (+0.237) | 0.000848 → 0.000765 (−0.045) | 0.010564 → 0.010494 (−0.003) |
| CAS6 (11.0 / 69.5 / 19.9) | 0.77507 → 0.77997 (+0.003) | 0.0000143 → 0.0000418 (+0.466) | 0.03465 → 0.04479 (+0.111) |

Across all six Baird compositions, the v5−v4 changes span +0.003 to +0.368
dex for a(SiO₂), −0.147 to +0.466 dex for a(CaO), and −0.059 to +0.222 dex
for a(Al₂O₃). The CaO change is non-monotonic and the large silica-rich
relative shift occurs where both predicted CaO activities are very small.

For `SiO₂(l) ⇌ SiO(g) + ½ O₂(g)`, p(SiO) is proportional to a(SiO₂) at a
fixed oxygen fugacity. The table below scales Baird's inferred pressure by
`a_model/a_Baird,liq` at each composition; it is a matched-buffer model
comparison, not a separately calculated gas equilibrium. Baird measured the
total CO+SiO pressure, then inferred p(SiO) from the SiC+CO equilibrium.

| Composition | Baird inferred p(SiO) (atm) | Scaled v4 p(SiO) (atm) | Scaled v5 p(SiO) (atm) |
| --- | ---: | ---: | ---: |
| CAS1 | 0.005 | 0.00234 | 0.00547 |
| CAS3 | 0.010 | 0.00782 | 0.01351 |
| CAS6 | 0.022 | 0.02180 | 0.02194 |

On all six Baird silica-activity rows, after converting to the liquid reference
and combining the CAS5 replicate activities by their geometric mean, the
residual `log₁₀(a_model/a_measured)` has mean absolute residual (MAE) 0.124 dex
(RMSE 0.164) for v4 and 0.080 dex (RMSE 0.116) for v5. V4 is closer at CAS2
and CAS3; v5 is closer at CAS1, CAS4, CAS5, and CAS6. The aggregate difference
is modest, but this second-laboratory activity check favors v5 rather than v4.
The pressure table expresses the same silica-activity comparison at Baird's
buffer and is not an independent second score. The primary record is [Baird, *The Measurement of
Silica Activities in Liquid Slags*, Glasgow PhD thesis (1956), Table XII](https://theses.gla.ac.uk/73568/1/10656328.pdf); the university record catalogues it as a 1957 thesis ([record](https://theses.gla.ac.uk/73568/)).

## Activity and vapor evidence

The reported CAS SiO evolution and activities in Shornikov, Stolyarova &
Shultz (1996), “Mass spectrometric study of thermodynamic properties of
melts of the CaO–Al₂O₃–SiO₂ system,” *Tekhnika i tekhnologiya silikatov* 3
(1–2), 8–22 (English translation listed as “High-temperature mass
spectrometric study of the thermodynamic properties of the
CaO–Al₂O₃–SiO₂ system,” *High Temperature and Materials Science* 36(1),
15–35), are reported to favor the stronger, FC87-based v4 association. The
reported CaO table is an outlier relative to that silica/SiO trend and is not
used here as independent support for a(CaO). I could not obtain a primary
full-text copy of the 1996 numerical tables, so I assign no exact activity or
pressure residual to this result and do not use it as a quantitative score.
The 2003 article cites the translated 1996 paper in its references
([publisher record](https://link.springer.com/article/10.1023/A:1026382828677)).

Stolyarova, Lopatin & Stolyar (2003) measured silica activities by
high-temperature mass spectrometry at 1905 ± 10 K along the x(SiO₂)=0.5 and
0.6 joins, with CaO:Al₂O₃ mole ratios from 0.25 to 4. Their abstract reports
that activity remains constant along each join. On those same compositions,
the model's calculated activity span is 0.374 dex (v4) versus 0.241 dex (v5)
at x(SiO₂)=0.5, and 0.174 dex (v4) versus 0.121 dex (v5) at x(SiO₂)=0.6.
This qualitative shape check therefore favors v5's flatter trend, but the
accessible abstract does not provide absolute activities for a level-based
residual, and this is the same research group as the 1996 study. The paper is
“Determination of the SiO₂ Activity in Melts of the CaO–Al₂O₃–SiO₂ System by
High-Temperature Mass Spectrometry,” *Glass Physics and Chemistry* 29,
451–455 (2003), [doi:10.1023/A:1026382828677](https://doi.org/10.1023/A:1026382828677).

Baird supplies a second-laboratory CAS cross-check, but not a direct
measurement of p(SiO): the measured quantity is total CO+SiO pressure, and
the SiO portion is inferred using a SiC equilibrium. I found no independent
second-laboratory CAS study that directly measures p(SiO) over matched
compositions and temperatures. Thus the available cross-lab comparison is
useful but does not arbitrate the 1996/v4 report by itself.

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
then estimates. V5 remains the default because its A-only refit makes the
anorthite liquid formula Gibbs energy close on the documented crystal,
fusion, liquid-Cp, and parent-liquid standard-state chain. V4 is a genuine
published thermochemical route, not a casual estimate, but its older FC87
value is 30.6–44.6 kJ mol⁻¹ more favorable over the requested range than the
later Hess reconstruction. Activity evidence alone does not warrant replacing
the Hess-consistent row with the competing coefficient. The pressure/activity
evidence is also split: the unquantified 1996 report points to v4, while
Baird's cross-laboratory table and the 2003 constant-silica pattern modestly
favor v5. No row-specific confidence band is available, and the parent
liquid-reference inputs retain their stated uncertainty.

The choice should be revisited if a primary, same-standard liquid-An
formation Gibbs measurement or higher-temperature enthalpy/Cp series moves
the Hess target toward v4, or if a second independent laboratory publishes
reproducible CAS activities and directly measured SiO pressures over matched
composition/temperature points with reference-state and uncertainty details.
The 1996 primary tables should also be recovered and reduced on the same
standards before assigning them a numerical score.

## Melt notice

A future CAS activity-sensitivity notice is warranted because the association
choice changes predicted silica activity and therefore p(SiO) by up to 0.37
dex (a factor of 2.33) in Baird's measured composition range. The existing
near-liquidus notice covers a nearly stoichiometric anorthite melt and does
not describe this composition-dependent source split. A notice should state
that v5 is retained on Hess consistency, that the 1996 report and later
cross-checks do not agree on the preferred association, and that no universal
correction is justified. Its comparison should remain unscored until the 1996
tables and a direct second-lab CAS p(SiO) dataset are available. This note
does not change pack coefficients or implement a notice.
