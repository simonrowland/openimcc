# Sulfur gas-parent activity fragment

This opt-in fragment supplies the existing gas evaluator with the missing S2
parent activity when total sulfur and oxygen fugacity are known. It follows
the existing `evaluate_sulfur` capacity/speciation helper because the engine
already has the measured sulfide and sulfate capacity equations. A crystal,
fusion, and liquid-heat-capacity construction for S2(l) is not available here,
so this fragment does not invent a liquid standard or associate Gibbs rows.

The published gas reference is S2(g) at 1 bar. The JSON records the JANAF S2
and O2 standards and the two published capacity rows. The internal runtime
loader `activity_fragments.load_activity_fragment("S")` reads these rows; it
does not alter a default melt or gas table.

## Source rows

The sulfide capacity is Boulliung and Wood (2023), Eq. 7, p. 6, Contributions
to Mineralogy and Petrology 178:56, [DOI 10.1007/s00410-023-02033-9](https://doi.org/10.1007/s00410-023-02033-9).
It combines measured sulfide capacities for 13 melts at 1473–1773 K with
O'Neill and Mavrogenes measurements at 1673 K. The reported regression standard
error is 0.09 log10(wt% S), with R2 = 0.96. The composition ranges are listed
in Table 1, p. 3.

The sulfate capacity is Boulliung and Wood (2023), Eq. 9, p. 7, using sulfate
measurements from their 2022 study, [DOI 10.1016/j.gca.2022.08.032](https://doi.org/10.1016/j.gca.2022.08.032).
The 2023 paper refits these data on a single-oxygen basis alongside Eq. 7.
Its regression standard error is 0.14 log10(wt% S), with R2 = 0.988. Original
sulfate calibration compositions are in Table 1, p. 151, of the 2022 paper.
Neither equation was fit to battery rows.

NIST-JANAF S2(g) table S-012 and O2 table O-029 state a standard pressure of
0.1 MPa, equivalent to 1 bar. The database DOI is
[10.18434/T42S31](https://doi.org/10.18434/T42S31); the tables are
[S-012](https://janaf.nist.gov/tables/S-012.html) and
[O-029](https://janaf.nist.gov/tables/O-029.html).

## Closure and units

The printed dissolution equations are:

```text
log10[S2-] = log10 C_S2- + 0.5(log10 fS2 - log10 fO2)
log10[S6+] = log10 C_S6+ + 0.5 log10 fS2 + 1.5 log10 fO2
```

At fixed `fO2`, both species contain the same `0.5 log10 fS2` term. Summing
their concentrations and solving gives:

```text
log10 fS2 = 2 (log10 S_total,wt%
  - log10(10^(log10 C_S2- - 0.5 log10 fO2)
         + 10^(log10 C_S6+ + 1.5 log10 fO2)))
a(S2) = fS2 / 1 bar
```

The oxide fractions and fugacity ratios are dimensionless. Each capacity and
the sulfur inventory are on the source's wt% S scale; `T` is kelvin and every
temperature-dependent coefficient has kelvin units. The returned `a(S2)` is
dimensionless on the NIST-JANAF 1-bar gas standard. The fragment does not
convert capacity alone into activity: callers provide total S and `fO2`.

As a sanity value, N-MORB from Smythe et al. (2017), Table 3, p. 800,
[DOI 10.2138/am-2017-5800CCBY](https://doi.org/10.2138/am-2017-5800CCBY), at
1673 K, `log10 fO2 = -8.97`, and 500 ppm total S gives approximately
`log10 fS2 = -2.5721`, or `a(S2) = 0.002678`. This is calculated from the
printed Eq. 7 and Eq. 9 coefficients.

## Scope and flags

Both central fits are for dry silicate melts at 1 atm and 1473–1773 K. Their
calibration composition intervals are surfaced as typed notices. Eq. 7 covers
SiO2 39.9–61.6 wt%, FeO 0–16.9 wt%, CaO 4.1–22.3 wt%, TiO2 0–8.34 wt%, and
Na2O 0.13–7.77 wt%. Eq. 9 covers SiO2 39.9–73.2 wt%, FeO 0–16.9 wt%, CaO
1.0–22.3 wt%, TiO2 0–8.34 wt%, Na2O 0–7.77 wt%, K2O 0–8.6 wt%, MgO
0.3–17.4 wt%, MnO 0–2.45 wt%, and Al2O3 10.9–22.9 wt%.

Every capacity-derived gas-parent value carries
`imcc_sulfur_parent_domain_notice`. Temperature, pressure, composition, and
components outside the reported fits also receive the existing capacity
notices. The published regression errors are recorded as fit errors, not
prediction intervals; no provisional coefficient is added inside the stated
domain.

## Known limits and smoke result

The closure assumes only dissolved sulfide and sulfate, at the source's
atmospheric-pressure conditions. It omits water effects, metal-silicate sulfur
partitioning, sulfate saturation, and changing feed speciation. Pressure
outside the calibration and compositions outside the recorded axes are flagged
as extrapolations. The gas table has no COS or H2S rows; this fragment does not
add them. The battery coverage contains no S-bearing rows, so this change
unlocks zero scored rows and has no predicted-versus-printed battery pairs to
report. A lunar-mare-like major-oxide smoke composition at 1773 K and 630 ppm S
produced finite S-bearing gas pressures; every non-S gas pressure was exactly
unchanged from the default evaluation.
