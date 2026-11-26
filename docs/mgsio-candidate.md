# MgO–SiO₂ minimal-associate candidate

`mgsio-candidate` is an opt-in activity-fit profile. Its D066-derived pack
retains all 38 row records and the existing eight parent liquid standards,
marks MgSiO₃ inactive with an explicit reason, and fits Mg₂SiO₄ A and B. The
default profile and `imcc-sf04-d066-v4.json` are unchanged. The fit is
provisional and uses measured activities; it is not an independent
thermochemical determination.

The fit used 46 exposed training activity rows from 1873–1973 K, weighting
a(MgO) and a(SiO₂) residuals jointly. Coincident Kume and Morita silica
observations each received half weight. The fit used the declared bounds
−250 to +100 kJ/mol for ΔH° and ±100 J/(mol·K) for ΔS°. The fitted constants
are stored in `tools/build_mgsio_candidate_pack.py`, so its `--check` does not
need the private residual data.

## Training fit

Residuals are signed `log10(predicted/measured)` dex; each pair is mean / RMS.

| Source | Oxide | n | D066 mean / RMS | Candidate mean / RMS |
|---|---:|---:|---:|---:|
| Arutyunyan 2005 | MgO | 6 | +0.0245 / 0.0265 | +0.0402 / 0.0445 |
| Arutyunyan 2005 | SiO₂ | 6 | −0.3766 / 0.3767 | −0.1755 / 0.1764 |
| Kambayashi & Kato 1984 | MgO | 8 | +0.0107 / 0.0192 | −0.0008 / 0.0642 |
| Kambayashi & Kato 1984 | SiO₂ | 8 | −0.3535 / 0.3538 | −0.1154 / 0.1338 |
| Kambayashi 1983 | MgO | 6 | +0.0206 / 0.0425 | +0.0172 / 0.0205 |
| Kambayashi 1983 | SiO₂ | 6 | −0.3943 / 0.3962 | −0.1745 / 0.1749 |
| Kume 2000 | MgO | 2 | −0.1933 / 0.2072 | −0.1732 / 0.1767 |
| Kume 2000 | SiO₂ | 2 | −0.4117 / 0.4160 | −0.2133 / 0.2146 |
| Morita et al. 2000 | SiO₂ | 2 | −0.4117 / 0.4160 | −0.2133 / 0.2146 |

Across both oxides, D066 joint RMS is 0.279390 unweighted and 0.265585
weighted. The candidate gives 0.132483 unweighted and 0.124698 weighted. These
printed values improve on the existing associate set refit with A and B free,
where MgSiO₃ reached its weakening bound.

## Held-out and family checks

Zaitsev 2006 was held out and reported only in aggregate: n=12, median absolute
residual 0.202337 → 0.109047 dex, signed median −0.177989 → −0.067925 dex. No
row-level Zaitsev values were used to choose the candidate.

| Family | Oxide | n | Δlog₁₀a | Residual mean, before → after | RMS, before → after |
|---|---:|---:|---:|---:|---:|
| CMAS | CaO | 15 | −0.0390 | −0.1633 → −0.2023 | 0.9005 → 0.8764 |
| CMAS | MgO | 23 | +0.0692 | −0.3740 → −0.3048 | 0.4995 → 0.4274 |
| CAS | Al₂O₃ | 116 | 0.0000 | −0.0305 → −0.0305 | 0.4813 → 0.4813 |
| CAS | CaO | 112 | 0.0000 | −0.2475 → −0.2475 | 0.7590 → 0.7590 |
| CAS | SiO₂ | 150 | 0.0000 | −0.1677 → −0.1677 | 0.4755 → 0.4755 |
| Kume CMS | CaO | 25 | −0.1482 | +0.3829 → +0.2347 | 0.5058 → 0.4749 |
| Kume CMS | MgO | 28 | +0.0539 | −0.2182 → −0.1644 | 0.3633 → 0.2927 |
| Kume CMS | SiO₂ | 25 | +0.1097 | −0.5916 → −0.4820 | 0.6315 → 0.5289 |
| Na-silicate, in-domain | Na₂O | 10 | 0.0000 | +0.1149 → +0.1149 | 0.2811 → 0.2811 |
| Na-silicate, in-domain | SiO₂ | 6 | 0.0000 | −0.0046 → −0.0046 | 0.0154 → 0.0154 |

For the Na-silicate family, 16 of 335 rows were in-domain; 284 were outside
the temperature domain and 35 were outside the composition envelope. There
were no in-domain MgO rows. The family diagnostics retain archive rows marked
`score_eligible=false` for sensitivity and are not default benchmark scores.

## Liquidus crossings

The candidate preserves every requested crossing. Temperatures are in kelvin;
the phase column is the returned saturation crossing.

| Composition | D066 | Candidate |
|---|---:|---:|
| Forsterite | 1801.2, forsterite | 1858.4, forsterite |
| Enstatite | 1538.8, forsterite | 1701.5, forsterite |
| Arutyunyan xSiO₂ = 0.488 | 1557.3, forsterite | 1712.6, forsterite |
| Arutyunyan xSiO₂ = 0.563 | 1450.6, forsterite | 1614.8, enstatite |

## Mg₂SiO₄ fit thermochemistry

The stored fit is A = −0.470703 and B = +6619.71 K. For
`log10 K = A + B/T`, matching `ln K = ΔS°/R − ΔH°/(RT)` gives
`ΔS° = R ln(10) A = −9.011 J/(mol·K)` and
`ΔH° = −R ln(10) B = −126.733 kJ/mol`. Both parameters are within the fit
bounds and close to D066 v4 (approximately −141.2 kJ/mol and
−17.4 J/(mol·K)). The literature formation reactions use solid oxide
standards and are not numerically identical to this liquid-standard
association fit. Kambayashi reports ΔG° = −56.8 ± 0.6 kJ/mol at 1873 K for
Mg₂SiO₄(s) formation from solid MgO and SiO₂; a later calorimetric study gives
−60.68 ± 2.20 kJ/mol for forsterite formation from 2MgO(s) + SiO₂(quartz) at
298 K. They support the exothermic formation direction, not an exact fit
value. [Kambayashi (1983)](https://doi.org/10.1016/0021-9614(83)90136-2),
[Kojitani et al. (2015)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2015JB012211).

The minimal associate choice is consistent with Pelton's binary MgO–SiO₂
model, which represents ordering with Mg–Si pairs and an orthosilicate-like
configuration near Mg₂SiO₄ composition; it does not independently validate
these fitted constants. [Pelton (2004)](https://pyro.co.za/MoltenSlags2004/607-Pelton.pdf).

Training sources: [Arutyunyan et al. (2005)](https://repo.library.stonybrook.edu/xmlui/bitstream/handle/11401/70084/DOPv50i8final.pdf?sequence=2),
[Kambayashi & Kato (1984)](https://doi.org/10.1016/0021-9614(84)90085-5),
[Kambayashi (1983)](https://doi.org/10.1016/0021-9614(83)90136-2),
[Kume et al. (2000)](https://doi.org/10.2355/isijinternational.40.561), and
[Morita, Kume & Sano (2000)](https://doi.org/10.2355/isijinternational.40.554).
