# Schaefer & Fegley (2004) published reference

This is an independent, partial reference set transcribed or digitized only
from Schaefer, L. & Fegley, B. (2004), “A thermodynamic model of high
temperature lava vaporization on Io”, *Icarus* 169, 216–241,
doi:10.1016/j.icarus.2003.08.023. This directory's own arrangement, provenance
text and derived tables are covered by the package's CC-BY-4.0 data licence.
Published coefficients and values digitized from published figures are cited
facts, not a grant of rights to the publisher's tables or figures. NIST-JANAF
records are NIST public data, not CC-BY-4.0.

It does not replace `tests/fixtures/imcc_sf04_magma_workbook.csv`. That fixture
is a separately licensed, model-output comparison and remains an optional local
regression. This directory contains only values recoverable from the published
paper: five Table 5 compositions, thirteen Table 9 flux cells converted with
Eq. 11, and a cautious subset of Fig. 10.

## Files and provenance

`compositions.csv` contains the five model compositions in Table 5 (p. 224),
with the printed precision retained as strings. `table9_anchors.csv` preserves
the printed 1900 K flux in `flux_cm2_s` and stores the independent Eq. 11
pressure in a separate column. `fig10_digitized.csv` contains only the seven
requested temperatures and includes both the digitized mole fraction and the
Table 7 pressure-fit derivation. `PROVENANCE.yaml` records source locators,
method classes, and hashes.

Method-class counts:

| method class | count | location |
| --- | ---: | --- |
| `transcribed` | 5 | Table 5 composition rows |
| `transcribed` | 13 | Table 9 flux cells |
| `derived_eq11` | 13 | Table 9 pressure cells |
| `digitized_figure` | 350 | Fig. 10 points |
| `derived_table7` | 350 | Fig. 10 pressure cells |

## Fig. 10 feasibility and digitization

Step 1 result: **raster**. On PDF page 12 (printed p. 227),
`pdfimages -list -f 12 -l 12` reports one 3654×5148 1-bit CCITT stencil and
no vector curve paths. The build script renders that page with
`pdftoppm -r 600 -gray` to a temporary PGM, applies the recorded panel-axis
calibration, and writes only numbers; no figure image is shipped.

There are 72 labelled panel/species curves and 504 candidate grid cells. The
CSV includes 350 points. The 154 excluded points are listed below: 75 original
floor/label/ambiguity exclusions plus 79 points rejected by the curve gate.
Omitted curves are never replaced by a guessed value.

| rock | curve and excluded temperatures (K) | reason |
| --- | --- | --- |
| tholeiite | Mg: 1750; TiO2: 1750, 1875, 1900, 2000, 2125; MgO: 1750, 1875, 1900, 2000, 2125, 2375 | clipped at the −5 floor or merged with the floor/adjacent trace |
| alkali basalt | SiO2: 1750; Mg: 1750; TiO2: 1750–2125; Na2: 1750–2250; MgO: 1750–2125 | clipped at the −5 floor |
| komatiite | SiO: 1750 | ambiguous at a crossing |
| komatiite | SiO2: 1875; MgO: 1750–1900 | clipped at the −5 floor |
| dunite | SiO2: 1750; MgO: 1750; K+: 1750–2000; KO: 1750–2125; Na2: 1750–2250 | clipped at the −5 floor |
| Allende B1 CAI | Mg: 1750; SiO2: 1750, 2125; MgO: 1750–2125, 2250, 2375; TiO2: 1750–2250 | clipped at the −5 floor; the two isolated lower traces at 2125/2250 were not unambiguous |
| Allende B1 CAI | SiO2: 1900 | overlap with another trace |
| alkali basalt | Fe: 1900; NaO: 1875; K: 2125; Na+: 2125; Mg: 2125 | two strokes within the local measurement window; no unambiguous assignment |
| dunite | MgO: 2000 | two strokes within the local measurement window; no unambiguous assignment |

The curve gate added in this fix excluded these 79 points from the previously
included 429. Each `rock/species/T` entry is one excluded point; the reason in
the first column applies to every entry on that row.

| reason | newly excluded rock/species/T |
| --- | --- |
| two strokes at the calibrated column | `aba/Fe/2125`, `aba/Fe/2375`, `cai/NaO/1900`, `cai/NaO/2250`, `dun/SiO/1750`, `tho/TiO2/2375` |
| vertical thickness 8 px; expected 3–7 px | `aba/Mg/1900`, `aba/NaO/1750`, `dun/MgO/2250`, `dun/Na+/2375`, `kom/K/1875`, `kom/Mg/2000`, `kom/MgO/2000`, `kom/Na+/1875`, `kom/SiO2/2000`, `tho/K/2000`, `tho/Na+/1875`, `tho/Na+/2000`, `tho/e-/1750` |
| vertical thickness 9 px; expected 3–7 px | `aba/Na+/1750`, `aba/e-/1875`, `cai/Fe/1750`, `cai/Na+/2000`, `dun/Fe/1750`, `kom/Na+/2375`, `kom/SiO2/2375` |
| vertical thickness 10 px; expected 3–7 px | `dun/e-/1750` |
| vertical thickness 11 px; expected 3–7 px | `cai/Fe/1900`, `cai/Mg/1900`, `cai/Na+/1900` |
| vertical thickness 12 px; expected 3–7 px | `kom/Mg/2250` |
| vertical thickness 13 px; expected 3–7 px | `aba/Na+/1900`, `cai/K/1875` |
| vertical thickness 29 px; expected 3–7 px | `kom/Fe/2125` |
| no thin path at horizontal offset −1 | `aba/FeO/2375`, `dun/NaO/1750`, `kom/Na+/1750` |
| no thin path at horizontal offset +1 | `cai/SiO/2000`, `dun/Fe/1900`, `dun/Mg/1900`, `dun/NaO/1900` |
| no thin path at horizontal offset −2 | `aba/Na2/2375`, `aba/TiO2/2250`, `dun/FeO/1875`, `kom/K/1900`, `kom/NaO/1900`, `tho/Na+/1900`, `tho/e-/1900` |
| no thin path at horizontal offset −3 | `aba/TiO2/2375`, `aba/e-/2000`, `cai/SiO/2250`, `cai/e-/2000`, `dun/Na+/1750`, `kom/K/2000` |
| no thin path at horizontal offset −4 | `kom/K/2250`, `kom/NaO/2250` |
| no thin path at horizontal offset −5 | `cai/Mg/1875`, `cai/Na+/1875`, `dun/MgO/2375`, `tho/O/2375` |
| no thin path at horizontal offset −6 | `cai/Mg/2375`, `kom/Fe/2375`, `kom/SiO/2375` |
| no thin path at horizontal offset −7 | `aba/e-/1900` |
| no thin path at horizontal offset −8 | `cai/FeO/1750` |
| no thin path at horizontal offset +4 | `kom/Na+/2000`, `kom/e-/2000` |
| no thin path at horizontal offset +5 | `aba/MgO/2375`, `cai/SiO2/1875`, `dun/K+/2125`, `dun/SiO2/2125` |
| no thin path at horizontal offset +7 | `aba/Na+/1875`, `cai/Fe/2250`, `cai/FeO/2250`, `cai/K/2250`, `cai/Na+/2250`, `cai/SiO2/2250`, `cai/e-/2250`, `tho/TiO2/2250` |

Each included value is measured from the raster: at the calibrated x column,
the build searches x±2 pixels for a dark contiguous stroke whose centroid is
within ±0.05 dex of the stored seed. Clean curves are 4–5 dark pixels thick
at 600 dpi, so the accepted thickness band is 3–7 pixels (±2 pixels of
tolerance). The selected stroke must then continue as one 3–7-pixel run at
every column from x−8 through x+8; adjacent centroids may move by at most 2
pixels, and the full path must stay within 2 pixels of its fitted local slope.
N=8 gives 17 consecutive columns, long enough to outlast the compact labels
and glyph fragments around these seeds. The emitted value remains the centroid
of the accepted stroke across x±2 pixels. A missing, thick, discontinuous, or
multiply-stroked window fails the build; the six local ambiguities above are
explicitly omitted. Most included points have ±0.05 dex uncertainty. Six
points where a label or crossing obscures the local centreline use ±0.10 dex:
tholeiite NaO at 1900 K; komatiite NaO and Na+ at 2000 K; dunite Fe at 1900 K
and SiO at 2000 K; and CAI Na+ at 2000 K. Tholeiite Na, O2, and O at 1900 K
are clean strokes and use ±0.05 dex.

The 13 review-flagged seeds were re-traced as follows. The “before” column is
the old stored seed; the “after” column is the measured centroid written to
the CSV when the point passes the curve gate. Rows marked excluded remain
omitted because the stricter gate found thick or discontinuous ink.

| rock / species / T (K) | before | after |
| --- | ---: | ---: |
| tholeiite / Mg / 1875 | −4.968 | −4.534 |
| tholeiite / SiO2 / 2125 | −3.424 | −3.642 |
| komatiite / MgO / 2000 | −4.940 | excluded: vertical thickness 8 px; expected 3–7 px |
| alkali basalt / NaO / 1875 | −3.228 | excluded: two strokes |
| tholeiite / TiO2 / 2250 | −4.607 | excluded: no thin path at horizontal offset +7 |
| komatiite / SiO2 / 2375 | −3.828 | excluded: vertical thickness 9 px; expected 3–7 px |
| tholeiite / NaO / 1900 | −3.091 | −2.978 |
| komatiite / e− / 2000 | −4.571 | excluded: no thin path at horizontal offset +4 |
| dunite / Fe / 1900 | −3.409 | excluded: no thin path at horizontal offset +1 |
| dunite / Mg / 1875 | −3.462 | −3.391 |
| dunite / MgO / 2000 | −4.924 | excluded: two strokes |
| dunite / Na2 / 2375 | −4.859 | −4.743 |
| Allende B1 CAI / SiO2 / 2000 | −4.686 | excluded: no thin path at horizontal offset +1 |

## Printed-anchor closure and self-checks

For Eq. 11, the 13 converted Table 9 pressures sum to
`7.72825e-5 bar`. SF04 Table 7 prints `7.72e-5 bar` at 1900 K, so the closure
is +0.107%. Against the rounded A+B/T fit (`7.89625e-5 bar`), the fractional
gap is `sum/fit - 1 = -0.0213`, while `log10(sum/fit) = -0.0093 dex`. The
build-script sanity check is the Na cell:
`7.67e18 cm^-2 s^-1` gives `6.08409e-5 bar`.

At 1900 K the included Fig. 10 mole fractions sum to:

| rock | sum of included plotted species |
| --- | ---: |
| tholeiite | 0.810682 |
| alkali basalt | 0.803796 |
| komatiite | 0.809342 |
| dunite | 0.801864 |
| Allende B1 CAI | 0.805968 |

The approximately 0.80–0.81 band is a digitization check, not a claim that
the selected labels form a normalized gas inventory. Fig. 10 draws Na and O2
at nearly the same heights on all five panels, not only the tholeiite panel.
Those two major curves therefore contribute nearly the same amount in every
rock, which is why every rock's included plotted sum is about 0.80. The
remaining shortfall comes from a Fig. 10/Table 9 disagreement for tholeiite at
1900 K. Fig. 10 places Na at about −0.197 and O2 at about −0.805, while Table 9
implies about −0.103 and −0.718, respectively: roughly −0.094 and −0.087 dex.
Replacing only those two figure values with the Table 9 mole fractions closes
the tholeiite sum to `0.9985` using the printed Table 7 total.

For tholeiite at 1900 K, the figure-versus-Table-9 log-mole-fraction deltas
(figure minus Eq. 11 pressure divided by the printed Table 7 total pressure)
are:

| species | delta (dex) | figure uncertainty (dex) |
| --- | ---: | ---: |
| Na | −0.094 | 0.05 |
| O2 | −0.087 | 0.05 |
| e- | −0.031 | 0.05 |
| O | −0.068 | 0.05 |
| Fe | −0.053 | 0.05 |
| FeO | −0.047 | 0.05 |
| SiO | −0.057 | 0.05 |
| NaO | 0.103 | 0.10 |
| K | −0.022 | 0.05 |
| Na+ | −0.019 | 0.05 |
| SiO2 | 0.004 | 0.05 |
| Mg | −0.021 | 0.05 |

K+ is below the Fig. 10a plot floor and is not a figure point.

## Engine comparison

`tests/test_sf04_published_reference.py` runs the public IMCC `evaluate` path
on each Table 5 composition with `basis_type="wt"`, then calls `evaluate_gas`
at the reference temperature and the reference O2 partial pressure. Because
the adapter deliberately refuses Fe2O3, the test converts the printed Fe2O3
to its FeO-equivalent mass (`2 M(FeO) / M(Fe2O3)`) before passing the complete
eight-oxide map to `evaluate`; the wt-to-mol conversion itself remains the
engine's path.

The run produced 313 comparable residuals and 50 explicit “not comparable”
rows: e-, Na+, K+, and TiO2 have no openimcc gas channel for the relevant
points. The updated comparable counts by method class are:

| method class | reference rows | comparable rows |
| --- | ---: | ---: |
| `derived_eq11` | 13 | 10 |
| `digitized_figure` | 350 | 303 |
| `derived_table7` | 350 | 303 |

The all-comparable residual summary is min −2.211553, median 0.012866, p95
0.792812, max 1.399972, and RMSE 0.623590 dex.

The residual gate is per species, not one 3.0-dex gate. Each gate is the
measured maximum absolute residual from this run, rounded upward to 0.001 dex,
plus a fixed 0.3-dex margin. The baseline is fixed in the test so a regression
cannot raise its own threshold; these are failure gates, not fit targets.

| species | n | measured abs max (dex) | gate (dex) |
| --- | ---: | ---: | ---: |
| Fe | 26 | 1.399972 | 1.700 |
| FeO | 32 | 0.830320 | 1.131 |
| K | 28 | 0.992137 | 1.293 |
| KO | 2 | 0.261598 | 0.562 |
| Mg | 25 | 0.958989 | 1.259 |
| MgO | 8 | 1.241290 | 1.542 |
| Na | 36 | 1.416392 | 1.717 |
| Na2 | 1 | 2.211553 | 2.512 |
| NaO | 28 | 1.826513 | 2.127 |
| O | 35 | 0.027536 | 0.328 |
| O2 | 36 | 0.000000 | 0.300 |
| SiO | 31 | 1.075267 | 1.376 |
| SiO2 | 25 | 1.056831 | 1.357 |

Large residuals are findings in this comparison, not adjusted values. In
particular, openimcc's Na, NaO, and Na2 are below the SF04 reference at every
comparable point; Na has median residual −1.08 dex and is −1.42 dex at the
printed Table 9 anchor. O, SiO, and FeO are close at that anchor (about the
0.02-dex scale; this centroid run reports −0.058, −0.047, and −0.037 dex), so
the sodium result is not a pin or unit error. It is a model-to-model
difference to diagnose later; no model tuning is part of this reference set.
The test prints the same per-species, per-rock, and per-method report so a
channel-specific review can distinguish a printed anchor issue from a
digitization issue.

## What remains unavailable

This public reference does not provide Bishop Tuff or Type B CAI compositions;
the paper prints Allende Type B1 CAI instead. It also does not provide the
unplotted Al/Ca/Ti/K2/Zn species requested by the old workbook grid, nor the
1500, 1625, or 2500 K cells. Those omissions are intentional and require a
separately licensed computation or newly published numerical data.

### Why Na⁺ and e⁻ are mostly excluded, and which comparisons are thin

Na⁺ and e⁻ are held nearly equal by charge balance, so their Fig. 10 curves run
on top of each other and read as one merged stroke; the curve-shape gate refuses
those columns rather than assigning the merged ink to either species. Neither
species has an openimcc gas channel, so these exclusions do not affect the
engine comparison. After the gate, three compared species rest on very few
points and should not be read as a characterisation: Na2 (1 residual), KO (2)
and MgO (8).
