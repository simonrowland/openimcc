# Cobalt melt activity fragment

This opt-in fragment supplies a trace CoO activity estimate to the existing
gas layer. It uses the measured dilute activity coefficient from Holzheid,
Palme, and Chakraborty (1997),
[doi:10.1016/S0009-2541(97)00030-2](https://doi.org/10.1016/S0009-2541(97)00030-2),
*Chemical Geology* 139(1–4), 21–38. Their liquid-standard pooled value is
γCoO = 1.51 ± 0.28 over 76 experiments. The paper reports Co as divalent and
reports no temperature or oxygen-fugacity dependence from 1300 to 1600 °C
(1573.15–1873.15 K) and IW+1.5 to IW−3. It reports FeO contents of 0–12 wt%
and MgO contents of 4–30 wt%; above 20 wt% MgO, γCoO increases slightly.

For dilute CoO, the helper evaluates

```text
a_CoO = gamma_CoO * x_CoO
```

where `x_CoO` is the simple oxide formula-unit mole fraction. The estimate is
dimensionless and uses the same pure-liquid CoO standard state as the existing
gas condensate data. The reported ±0.28 is carried as a range on gamma and
activity; it is not described as a confidence interval. Every result includes
`imcc_co_activity_domain_notice`, because one pooled coefficient is not a
composition-specific model. Temperature, reported FeO/MgO ranges, and an
optional caller-supplied log10(fO2/IW) are checked; uncertain axes are flagged.
The dilute limit has no numeric upper bound in the cited source, and major
oxide effects are not parameterized. The result is therefore provisional for
specific basalt or regolith compositions.

## CoO pure-liquid reference

Gurvich, Veyts, and Medvedev, *Thermodynamic Properties of Individual
Substances*, Volume IV, Table 1334, supplies CoO crystal and liquid functions.
At the printed 2090 K fusion point, both phases have F = S−(H−H(0))/T =
108.388 J/(mol K). The crystal/liquid H−H(0) values are 116.405 and 156.405
kJ/mol, giving
ΔHfus = 40.000 kJ/mol; the corresponding entropies are 164.085 and 183.223
J/(mol K), giving ΔSfus = 19.138 J/(mol K). The ratio 40,000/2090 gives
19.139 J/(mol K), consistent with the printed entropy jump. The source labels
ΔHfus and liquid Cp as estimates; Cp(l) is 67 J/(mol K). The liquid Gibbs
branch follows from

```text
H_l(T) = H_cr(Tm) + ΔHfus + Cp_l (T − Tm)
S_l(T) = S_cr(Tm) + ΔSfus + Cp_l ln(T/Tm)
G_l(T) = H_l(T) − T S_l(T)
```

At Tm, equal printed F values and ΔHfus ≈ Tm ΔSfus recover phase equality.
The existing CoO(l) gas-table row already contains the source-derived liquid
standard and its supercooled continuation; this fragment does not regenerate
or edit that table.

## Scope and checks

The helper is opt-in and does not modify the default eight-oxide melt solve.
Import `evaluate_co_activity` from `openimcc.co_activity`, then pass its CoO
activity alongside the default melt parent activities to `evaluate_gas` and
include `CoO` in `parent_oxides`. Its implementation and source anchors are in
`co_activity.py` and `data/fragments/co/rows.json`. The fragment adds no
default-pack overlay hook. A default-melt/default-gas integration test checks
that Co and CoO pressures are finite and positive and every non-Co pressure
remains bit-identical.

The source domain covers 1573–1873 K within the 1323–2473 K process interval.
The supplied battery coverage census contains no Co-bearing observation rows,
so this fragment unlocks zero battery species/source rows and there are no
battery predicted-versus-printed pairs to report. It is not calibrated to any
battery output.
