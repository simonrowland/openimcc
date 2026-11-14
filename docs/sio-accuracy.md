# Observed SiO(g) pressure accuracy on liquid charges

This summary covers scored SiO evolution observations in the external
validation battery. It describes the rows and sources tested; it is not a
claim about every silicate melt.

## 1700 to below 1950 K

For the 45 liquid or source-reported molten-charge rows, the observed
empirical P10–P90 band of `log10(predicted/observed)` is **−0.283 to
+0.566 dex**. The median is **+0.188 dex** and the RMS is **0.584 dex**.
Positive residuals mean predicted pressure is higher than observed. This is a
summary band for these sources, not a confidence interval and not a guarantee
for future compositions.

Across those 45 rows, the full observed range is **−2.253 to +1.161 dex**.
The low end is concentrated in high-basicity CaO–Al₂O₃–SiO₂ (CAS)
compositions. Family summaries are:

| Family | n | Median (dex) | RMS (dex) |
| --- | ---: | ---: | ---: |
| CAS liquid charges, phase inferred ([Stolyarova et al., 1996](https://www.researchgate.net/publication/255261329_High-temperature_Mass_Spectrometric_Study_of_the_Thermodynamic_Properties_of_the_CaO-Al2O3-SiO2_System)) | 29 | +0.337 | 0.703 |
| CaO–SiO₂, clear liquid ([Stolyarova et al., 1991](https://doi.org/10.1149/1.2085485)) | 8 | +0.0585 | 0.270 |
| CaO–SiO₂, liquidus-borderline† ([Stolyarova et al., 1991](https://doi.org/10.1149/1.2085485)) | 6 | −0.263 | 0.233 |
| Wollastonite liquid ([KEMS study](https://ies.unitech-mo.ru/files/upload/publications/15845/0cfdd75246e16e601a54061e5c0d205f.pdf)) | 5 | −0.282 | 0.281 |
| Lunar highland analogue, phase inferred ([Dhofar 280 study](https://www.researchgate.net/publication/276834102_Origin_of_Native_Silicon_and_Iron_Silicides_in_the_Dhofar_280_Lunar_Meteorite)) | 3 | −0.092 | 0.092 |

† The six borderline CaO–SiO₂ rows are shown separately and are not part of
the 45-row liquid aggregate or its envelope. The source's competing liquidus
bounds straddle these compositions. The CAS paper reports measurements on
melts but does not give a liquidus for every scored composition; the lunar
highland phase assignment is also inferred for the analogue mixture.

## Rows with solid or uncertain phase

Solid-bearing or mixed MgO–Al₂O₃–SiO₂ charges and crystal-tagged hatrurite rows
show much larger negative residuals. The 14 mixed MAS rows have median −1.800
dex and RMS 2.084; the five hatrurite rows have median −1.007 dex and RMS
1.007. For the exact-matched MAS rows, other engines show the same pattern
(median −1.887 dex, RMS 1.983), while the openimcc-specific difference has
median +0.032 dex and RMS 0.225. This common-mode comparison points to a
phase/source effect in those rows, not a measure of homogeneous-liquid
accuracy. The MAS phase assessment follows the source's solid and mixed-phase
descriptions and mullite boundary data ([Aksay & Pask](https://doi.org/10.1126/science.183.4120.69)); hatrurite rows retain their source crystal tags.

## 1950–2200 K

The 33 liquid rows in this higher-temperature interval have median residual
**−0.755 dex** and RMS **0.886 dex**. Twenty-five CaO–TiO₂ slag rows at
2100 K drive the negative result; their cause is unresolved, and they have no
exact multi-engine comparison in the scored set. See the
[`cao-tio2-activity-bias` notice in `melt-notices.md`](melt-notices.md).

## Bounded source terms

Choosing NASA CEA rather than JANAF O-012 for SiO(g) lowers calculated
`log10(pSiO)` by about 0.045–0.051 dex from 1700 to 2000 K
([JANAF O-012](https://janaf.nist.gov/tables/O-012.html), [NASA CEA coefficients](https://raw.githubusercontent.com/nasa/cea/main/data/thermo.inp)).

The supercooled SiO₂ liquid-standard conversion contributes at most about
0.039 dex in the target band, based on high-cristobalite JANAF O-035 and the
fusion data used for the liquid standard ([JANAF O-035](https://janaf.nist.gov/tables/O-035.html), [RBT84](https://doi.org/10.1111/j.1151-2916.1984.tb19155.x)).

Independent CaO–SiO₂ activity measurements by Kume et al. (2000) inform the
activity-side comparison, but are not SiO-pressure rows in these summaries
([DOI 10.2355/isijinternational.40.561](https://doi.org/10.2355/isijinternational.40.561)).

## Method

Residuals are `log10(predicted/observed SiO pressure)`, so positive values
indicate over-prediction. The score uses openimcc commit `9c327c2` with the
D066 v4 melt pack. Rows are classified from reported experimental phase,
liquidus, or source phase tags; solid-bearing or mixed charges and
crystal-tagged rows are kept outside the homogeneous-liquid group. Some
per-row assignments, including CAS and the lunar analogue, are inferred from
the sources' reported experiments rather than composition-specific phase
boundaries.
