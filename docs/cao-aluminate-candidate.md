# CaO–Al2O3 aluminate candidate

`load_profile("cao-aluminate-candidate")` selects the opt-in
`imcc-sf04-d066-cao-aluminate-candidate-v1.json` pack. It keeps the D066 v4
parent basis and 38-row layout. The default profile and published packs are
unchanged.

The D066 rows in this family are CaAl2O4 (row 8), CaAl4O7 (row 9),
Ca12Al14O33 (row 10), and CaAl12O19 (row 22). There is no Ca3Al2O6 associate
row in D066 v4. Only rows 9 and 22 are refitted. All 14 exposed Allibert
parent-activity observations are at 2060 K. The current shared `A` shift is
fit against those activities and 345 diagnosis-exposed Stolyarova 1996 rows
across melt and gas rails. Each source has equal weight in the objective, so
the larger Stolyarova row count does not dominate. Stolyarova is a fit source
for this candidate; independent Cho-Suito rows remain excluded. Both inherited
`B` values stay fixed because the sources are close in temperature and do not
justify a separate slope. No crystal Gibbs value is fitted.

The previous one-eighth factor was a hand-selected transfer constraint, not an
objective minimum. It is replaced by minimizing
`J(f) = 0.5 × (MSE_Allibert(f) + MSE_Stolyarova(f))` for the signed
`log10(predicted/measured)` residual in dex, with `f` bounded to `[0, 1]` and
`ΔA = f × ΔA_Allibert`. Each MSE is averaged within its source. A 0.05 grid was
refined by golden-section search near the minimum. The minimum is
`f = 0.889630787`, `ΔA = −1.125561024`, and `J = 0.438645513 dex²`.

| f | J (dex²) | f | J (dex²) | f | J (dex²) |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.484156296 | 0.35 | 0.449631026 | 0.70 | 0.439533246 |
| 0.05 | 0.477014709 | 0.40 | 0.447122126 | 0.75 | 0.439102799 |
| 0.10 | 0.470698291 | 0.45 | 0.445053006 | 0.80 | 0.438825035 |
| 0.15 | 0.465159533 | 0.50 | 0.443369795 | 0.85 | 0.438679041 |
| 0.20 | 0.460341821 | 0.55 | 0.442020912 | 0.90 | 0.438647714 |
| 0.25 | 0.456185926 | 0.60 | 0.440958921 | 0.95 | 0.438717121 |
| 0.30 | 0.452634142 | 0.65 | 0.440141771 | 1.00 | 0.438875943 |

The fit used 14 Allibert and 345 Stolyarova numeric dex rows. At the refined
minimum, the source RMS residuals are 0.470376 dex and 0.809961 dex,
respectively. The coefficients keep the positive Allibert correction while
reducing the combined source-balanced objective versus both `f=0` and the
previous `f=0.125` value. The diagnostic table below describes the prior
unconstrained candidate; the final coefficients are scored under
"Full replay of the final coefficients".

## Coefficients and liquid activities

For `log10 K = A+B/T`, the association reaction has
`ΔG° = −R T ln(10)(A+B/T)`, so `ΔH° = −R ln(10)B` and
`ΔS° = +R ln(10)A`. The table reports the implied values, not direct
calorimetric measurements.

| Associate | A, D066 | A, objective-fit candidate | B (K), both | ΔH° (kJ/mol), both | ΔS°, D066 (J/mol K) | ΔS°, candidate (J/mol K) |
|---|---:|---:|---:|---:|---:|---:|
| CaAl4O7 | −0.590000 | −1.715561 | 9713 | −185.95 | −11.30 | −32.84 |
| CaAl12O19 | −3.790000 | −4.915561 | 22612 | −432.90 | −72.56 | −94.11 |

The fitted residual is `log10(predicted/measured)`. The Allibert component of
the objective fit has RMS 0.470376 dex; the candidate retains a strong binary
activity correction. The prior candidate comparison found 56 Stolyarova melt
rows with absolute-residual worsening above 0.05 dex (median 0.158, maximum
0.362).

## Full replay of the final coefficients

The final coefficients were replayed against the full scored residual set
(1,462 rows; the default replay reproduces the stored predictions to 8e-15
dex). There were no candidate-only failures. The 70 rows that fail are refused
by the default profile as well (45 trace-component bias, 25 ferric input).

| Cell or source | Role | n | Median absolute residual, D066 → candidate (dex) |
|---|---|---:|---:|
| Allibert 1981, Ca-Al melt 1950–2200 K | fit | 14 | 0.258 → 0.123 |
| Stolyarova 1996, a(Al2O3) | fit | 108 | 0.280 → 0.264 |
| Stolyarova 1996, a(SiO2) | fit | 52 | 0.197 → 0.213 |
| Stolyarova 1995, Ca-Al vapour 1700–1950 K | independent | 35 | 0.243 → 0.238 |
| Cho and Suito 1994, CAS melt 1700–1950 K | independent | 20 | 0.599 → 0.634 |
| CAS melt 1700–1950 K, all sources | mixed | 384 | 0.288 → 0.289 |
| CAS vapour 1700–1950 K, all sources | mixed | 92 | 1.187 → 1.136 |

77 rows worsen by more than 0.05 dex; 55 of them are Stolyarova 1996 melt
activities (median worsening 0.151 dex). The two independent sources do not
support the fit: one is flat, and one worsens by 0.035 dex. The candidate
therefore stays opt-in. Promotion needs independent CaO–Al2O3(–SiO2) activity
data.

## Prior transfer diagnosis

The prior full replay found 64 Stolyarova rows whose absolute residual worsened
by more than 0.05 dex: 56 melt activities, six Ca-vapour rows, and two SiO
evolution rows. These summaries are only for those flagged rows, not all
Stolyarova observations.

| Quantity | n | Signed median before → after | Median absolute residual before → after |
|---|---:|---:|---:|
| a(Al2O3) | 40 | +0.315 → +0.415 | 0.315 → 0.415 |
| a(CaO) | 11 | −1.129 → −1.271 | 1.129 → 1.271 |
| a(SiO2) | 5 | −0.135 → −0.197 | 0.173 → 0.246 |
| SiO evolution | 2 | +0.372 → +0.541 | 0.433 → 0.541 |
| Ca vapour | 6 | −1.021 → −1.140 | 1.021 → 1.140 |

| Composition slice | n | Signed median before → after | Median absolute residual before → after |
|---|---:|---:|---:|
| x(SiO2) < 0.20 | 7 | −0.118 → −0.183 | 0.118 → 0.188 |
| 0.20 ≤ x(SiO2) < 0.40 | 46 | +0.211 → +0.363 | 0.346 → 0.608 |
| 0.40 ≤ x(SiO2) < 0.60 | 11 | +0.147 → +0.301 | 0.191 → 0.301 |
| 1.00 ≤ x(CaO)/x(Al2O3) < 2.00 | 49 | −0.006 → +0.218 | 0.296 → 0.596 |
| x(CaO)/x(Al2O3) ≥ 2.00 | 15 | +0.314 → +0.403 | 0.346 → 0.439 |

The strongest cluster is x(SiO2)=0.20–0.60 with CaO/Al2O3 between 1 and 2.
Alumina residuals shift positive while CaO and silica residuals shift negative;
SiO pressure shifts positive and Ca vapour shifts negative. Since the
candidate lowers the two association constants, this pattern is consistent
with over-destabilizing the aluminate associates in silica-bearing melts.
This diagnosis preceded the source-balanced objective fit above. The
independent Cho-Suito source had six rows above the 0.05 dex worsening
threshold in the prior run; it is excluded from fitting and appears only as an
aggregate in the full replay results above.

## Invariant constraints and typed gaps

Affinity is computed from the phase Gibbs function and the liquid parent
activities. The implementation's dimensionless affinity is a natural-log
quantity; affinity in dex is that value divided by `ln(10)`. The earlier
invariant report called natural-log values dex, overstating the CA gap by a
factor of `ln(10)`. Crystal functions are used only through their source
limits: CA to 1875 K, CA2 to 2035 K, C3A to 1808 K, and C12A7 to 1700 K. The
measured CA and CA2 peritectics sit 0.15–3.15 K above the corresponding
limits; the C3A peritectics sit 5–6 K above its limit. CA6 is absent. The
table gives affinities at the last source-valid crystal temperature and at
the measured liquid composition; it does not extend crystal G beyond those
limits.

The measured zero-affinity points are diagnosis constraints, not coefficient
fit targets. Their crystal inputs are not source-complete at the measured
conditions: CA and CA2 exceed Barin's printed limits, CA6 is absent from the
CA2 peritectic, C3A is 5–6 K beyond its limit, and C12A7 is 28 K beyond its
last Gibbs value. Forcing zero at those points would require extending or
inventing crystal G and would make the Allibert liquid fit absorb that
unsupported choice.

| Phase constraint | Measured T (K), x(Al2O3) | Affinity at G cap, D066 (ln; dex) | Previous unrestricted candidate (ln; dex) | Computed invariant T and x |
|---|---:|---:|---:|---|
| CA = CA2 + L, Jerebtsov–Mikhailov | 1877.15, 0.484 | −1.381; −0.600 | −1.214; −0.528 | No source-valid zero through the 1875 K cap; measured T is 2.15 K beyond cap |
| CA = CA2 + L, Nurse et al. | 1877.15, 0.493 | −1.379; −0.599 | −1.210; −0.525 | No source-valid zero through the 1875 K cap; measured T is 2.15 K beyond cap |
| CA2 = CA6 + L, Jerebtsov–Mikhailov | 2035.15, 0.604 | −1.457; −0.633 | −0.215; −0.094 | No source-valid zero through the 2035 K cap; CA6 is absent |
| CA2 = CA6 + L, Nurse et al. | 2038.15, 0.667 | −1.305; −0.567 | −0.160; −0.069 | No source-valid zero through the 2035 K cap; measured T is 3.15 K beyond cap and CA6 is absent |
| C3A = CaO + L, Nurse | 1814.15, 0.291 | +1.787; +0.776 | +1.787; +0.776 | No invariant; C3A remains supersaturated while CaO is also stable |
| C3A = CaO + L, Jerebtsov–Mikhailov | 1813.15, 0.364 | +0.559; +0.243 | +0.557; +0.242 | No invariant; C3A remains supersaturated while CaO is also stable |
| C12A7 = L, Rankin–Wright | 1728.15, 0.368 | Not available at measured T; at 1700 K: −7.69; −3.340 | Not available at measured T; at 1700 K: −7.69; −3.340 | None; C12A7 Gibbs data stop at 1700 K |

The table's candidate values describe the unrestricted predecessor. Affinity
values for the final objective-fit coefficients have not been recalculated.
The crystal evidence still leaves typed gaps: CA and CA2 are at
or just beyond their source limits, CA6 is missing, C3A is beyond its limit,
and C12A7 ends 28 K below its measured point. No crystal-G values are extended
to close those gaps. No CA, CA2, or C3A stable mixed invariant is computed by
this pack.

At the pure Al2O3 end the unchanged computed transition is 2326.97 K at
x=1. No mixed invariant temperature or composition is produced by this pack.
