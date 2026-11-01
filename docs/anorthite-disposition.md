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

Shornikov, Stolyarova & Shultz (1996), “Mass spectrometric study of
thermodynamic properties of melts of the CaO–Al₂O₃–SiO₂ system,” *Tekhnika i
tekhnologiya silikatov* 3 (1–2), 8–22 (English translation listed as
“High-temperature mass-spectrometric study of the thermodynamic properties
of the CaO–Al₂O₃–SiO₂ system,” *High Temperature and Materials Science*
36(1), 15–35), remains a lead for future data recovery. I could not obtain
the primary numerical tables in a form that exposes their values and
conditions for a traceable comparison. The evidence direction is therefore
unverified; the data cannot establish which association value it favors or
whether its CaO results agree with the other oxide results. The translated
paper is cited by the 2003 article
([publisher record](https://link.springer.com/article/10.1023/A:1026382828677)).

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
the SiO portion is inferred using a SiC equilibrium. I found no independent
second-laboratory CAS study that directly measures p(SiO) over matched
compositions and temperatures. The 1996 numerical tables remain a future
retrieval and reduction lead, with no direction assigned here.

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
evidence does not enter that fit. The Baird Table XII transcription, which
could not be verified against the scan, gives a lower MAE for the isolated
FC87-row substitution (0.057 dex versus 0.080 dex for v5), while the 2003
constant-activity trend favors v5's flatter result. These activity checks
leave an empirical conflict, with one comparison provisional and the other
qualitative. The FC87 row remains a genuine published thermochemical route;
its association is 30.6–44.6 kJ mol⁻¹ more favorable over the requested
range than the Hess-refit row. No row-specific confidence band is available,
and the parent liquid-reference inputs retain their stated uncertainty.

The choice should be revisited if a primary, same-standard liquid-An
formation Gibbs measurement or higher-temperature enthalpy/Cp series moves
the Hess target toward FC87, or if scan-checked primary CAS activity data and
independent, directly measured SiO pressures become available over matched
composition/temperature points with reference-state and uncertainty details.

## Melt notice

A short CAS activity-sensitivity notice is warranted because changing only
the anorthite row moves predicted a(SiO₂) and p(SiO) by up to 0.151 dex
(factor 1.42) across the six Baird compositions, with a(Al₂O₃) changing by
up to 0.480 dex. The existing near-liquidus notice covers a nearly
stoichiometric anorthite melt and does not describe this composition-dependent
sensitivity. The CAS notice should state that v5 is retained on thermochemical
consistency, that the Baird transcription is unverified against the scan,
and that the 2003 shape check favors v5's flatter trend while the provisional
Baird activity score favors the FC87 row. It should report model sensitivity
without implying a universal correction. This note does not change pack
coefficients or implement a notice.
