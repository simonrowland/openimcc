# Mg-silicate activity candidate

`load_profile("silicate-candidate")` selects the opt-in
`imcc-sf04-d066-silicate-candidate-v2.json` pack. It retains all 38 D066 v4
associate rows and changes only the A intercepts of Mg₂SiO₄(l) and MgSiO₃(l).
The default and published profiles are unchanged.

## Scope and construction

The candidate targets low SiO₂ activities in SiO₂-poor MgO-rich melts. Its
fit uses 118 exposed liquid-activity observations from Arutyunyan (2005),
Kambayashi and Kato (1984), Kambayashi (1983), and Kume (2000): 59 SiO₂
objective rows and 59 paired MgO guard rows at 1873–1973 K. The objective is
the unweighted sum of squared dex residuals,
`Σ[log₁₀(a_model/a_measured)]²`. Each MgO guard residual is constrained to
change by no more than 0.05 dex from D066 v4. Both association entropies remain
within ±50 J mol⁻¹ K⁻¹. The Ca-silicate coefficients remain at D066 v4.

Selected FIT activities use pure liquid oxide standards. For any
solid-referenced value, equality of chemical potentials gives
`G°ₗ + RT ln(aₗ) = G°cr + RT ln(a_cr)`, hence
`aₗ = a_cr exp((G°cr − G°ₗ)/(RT))`. The conversion uses openimcc's own liquid
and crystal Gibbs functions. The figure-only Shornikov and Archakov point is
assumed to use solid periclase.

For `log₁₀ K = A + B/T`, the implied quantities are
`ΔH = −R ln(10) B` and `ΔS = R ln(10) A`.

| Associate | A v4 → candidate | B v4 → candidate (K) | ΔH v4 → candidate (kJ/mol) | ΔS v4 → candidate (J/mol K) |
|---|---:|---:|---:|---:|
| Mg₂SiO₄(l) | −0.909321 → −0.864853 | 7375.770516 → 7375.770516 | −141.207 → −141.207 | −17.409 → −16.557 |
| MgSiO₃(l) | +0.450679 → +0.350669 | 2270.770516 → 2270.770516 | −43.473 → −43.473 | +8.628 → +6.713 |
| Ca₂SiO₄(l) | +0.660679 → +0.660679 | 8357.770516 → 8357.770516 | −160.007 → −160.007 | +12.649 → +12.649 |
| CaSiO₃(l) | +0.570679 → +0.570679 | 5509.770516 → 5509.770516 | −105.483 → −105.483 | +10.926 → +10.926 |

The two fitted A shifts are constant over the 1700–3000 K pack domain, so
`log₁₀ K(T)` remains continuous there. The narrow 100 K FIT interval does not
identify separate B slopes, so both remain at D066 v4.

## FIT and held-out results

| FIT group | n | Median |dex| v4 → candidate | Signed median v4 → candidate | RMS v4 → candidate |
|---|---:|---:|---:|---:|---:|
| Selected a(MgO) guard rows | 59 | 0.118675 → 0.129192 | — | — |
| Selected a(SiO₂) objective rows | 59 | 0.386245 → 0.345002 | — | — |
| Full scored FIT-source rows | 511 | 0.335645 → 0.323623 | −0.112789 → −0.107364 | 0.573489 → 0.568910 |

The SiO₂-poor Mg-melt correction is limited: Arutyunyan's median signed
SiO₂ residual moves from −0.377 to −0.337 dex, correcting about 10% of the
bias. Across Arutyunyan and the two Kambayashi sources, MgO residuals are already
near zero while SiO₂ is low. A shifts large enough to correct all of the SiO₂
bias over-move a(MgO), so this candidate keeps a conservative MgO guard and
accepts the remaining silica bias.

| Source and activity | n | Median |dex| v4 → candidate | Signed median v4 → candidate |
|---|---:|---:|---:|---:|
| Arutyunyan (2005), a(MgO) | 6 | 0.026041 → 0.011146 | +0.026041 → +0.011146 |
| Arutyunyan (2005), a(SiO₂) | 6 | 0.377295 → 0.337257 | −0.377295 → −0.337257 |
| Kambayashi and Kato (1984), a(MgO) | 16 | 0.031616 → 0.042905 | +0.019683 → +0.001797 |
| Kambayashi and Kato (1984), a(SiO₂) | 16 | 0.360503 → 0.318952 | −0.360503 → −0.318952 |
| Kambayashi (1983), a(MgO) | 10 | 0.030903 → 0.025144 | +0.010890 → −0.007156 |
| Kambayashi (1983), a(SiO₂) | 10 | 0.381100 → 0.338231 | −0.381100 → −0.338231 |
| Kume et al. (2000), a(MgO) | 45 | 0.267930 → 0.270913 | −0.257470 → −0.252964 |
| Kume et al. (2000), a(SiO₂) | 86 | 0.309014 → 0.300418 | −0.298601 → −0.294567 |

Zaitsev (2006) and Shornikov and Archakov (2001) are held out from fitting and
parameter selection. Zaitsev's 12 liquid-activity rows are at 1873 K. The
figure-only Shornikov and Archakov point is at 2050 K, with
`x(MgO) = x(Al₂O₃) = 0.45`, `x(SiO₂) = 0.10`, and measured solid-reference
`a(MgO) = 0.34`.

| Held-out set | n | Median |dex| v4 → candidate | Signed median v4 → candidate | RMS v4 → candidate |
|---|---:|---:|---:|---:|---:|
| Zaitsev (2006) | 12 | 0.202337 → 0.175859 | −0.177989 → −0.164696 | 0.267039 → 0.239002 |
| Shornikov and Archakov (2001), Fig. 2a | 1 | 0.325517 → 0.327483 | −0.325517 → −0.327483 | 0.325517 → 0.327483 |
| Combined | 13 | 0.325517 → 0.327483 | −0.325517 → −0.327483 | 0.271984 → 0.246937 |

In the held-out 1700–1950 K binary-silicate cell, Charles (1967) and
KEMS-MS2000-044 are unchanged, while all 12 Zaitsev rows improve.

| Source in held-out binary cell | n | Median |dex| v4 → candidate | Signed median v4 → candidate | RMS v4 → candidate |
|---|---:|---:|---:|---:|---:|
| Charles (1967) | 2 | 0.015413 → 0.015413 | −0.015413 → −0.015413 | 0.015421 → 0.015421 |
| KEMS-MS2000-044 | 2 | 0.073312 → 0.073312 | +0.050942 → +0.050942 | 0.089273 → 0.089273 |
| Zaitsev (2006) | 12 | 0.202337 → 0.175859 | −0.177989 → −0.164696 | 0.267039 → 0.239002 |

## Residue-path and scored checks

Markova (1984) residual-melt compositions are diagnostics, not fit targets. The
Mg/Si vapour flux ratio sums each Mg- and Si-bearing gas-species flux,
proportional to `p/√M` and weighted by its atom count. Positive
`log₁₀[(J_Mg/J_Si)/(x_Mg/x_Si)]` favors MgO loss.

| T (K) | γ(SiO₂) v4 → candidate | J_Mg/J_Si v4 → candidate | Flux preference v4 → candidate (dex) |
|---:|---:|---:|---:|
| 1748 | 0.207473 → 0.240182 | 2.070 → 1.630 | +0.223 → +0.119 |
| 1765 | 0.206043 → 0.239304 | 2.036 → 1.595 | +0.211 → +0.105 |
| 1905 | 0.175987 → 0.206309 | 2.110 → 1.635 | +0.211 → +0.101 |
| 1999 | 0.083446 → 0.096141 | 6.775 → 5.392 | +0.656 → +0.556 |
| 2069 | 0.024269 → 0.025206 | 47.36 → 43.98 | +1.403 → +1.371 |
| 2156 | 0.003378 → 0.003258 | 1338 → 1382 | +2.418 → +2.432 |
| 2233 | 0.0004802 → 0.0004798 | 49,819 → 49,859 | +3.379 → +3.379 |

The high-temperature Markova sign flip remains: the model still predicts MgO
loss above 2050 K. No bead-composition extract was available to recompute the
Mendybaev (2013) direction.

On the full 84,107-row scored set, no melt-activity cell worsens by more than
0.05 dex. The held-out leaderboard's process-weighted median improves from
0.035 to 0.020 dex over two validated cells; validated share is 5.1%. The
combined held-out RMS improves from 0.272 to 0.247, and all 12 Zaitsev rows
improve. The full scored FIT-source median |dex| improves from 0.336 to 0.324.

Default replay rows that could not be reproduced from the available inputs
(111 rows) are excluded from paired comparisons. On faithful rows, the maximum
default replay difference is 1.5e−13 dex.
