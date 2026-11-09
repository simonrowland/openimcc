# Dilute P2O5 activity extension

`evaluate_p2o5_activity(composition_moles, T_K)` is an opt-in activity
estimate. It returns `a(P2O5) = gamma(P2O5) x(P2O5)` on the
Jung-Hudon P2O5(l) standard used by the gas layer. Other melt-parent
activities are left unchanged. Pass the returned activity as the `P2O5` entry
to `evaluate_gas` or `evaluate_gas_oxygen_balance`.

The central row is Suito and Inoue (1984), Eq. 14, without the separate MnO
term:

```text
log10(gamma) = -1.02(23 xCaO + 17 xMgO + 8 xFetO - 26 xP2O5)
               - 22990/T + 9.490
```

The source activity coefficient uses the earlier Turkdogan-Pearson
hypothetical-liquid standard. That standard is an approximate estimate, so
the absolute activity carries a provisional-standard limitation. The
fragment converts it to the gas table's Jung-Hudon 1 bar standard with
`log10(a_target) = log10(a_source) + (G_source - G_target)/(R T ln(10))`.
The conversion includes the 1 atm to 1 bar correction for the 3.5 gaseous
reactant moles; it does not apply a liquid pressure correction.

The source window has two composition ranges: 0.2-1 wt% P2O5 from the prior
low-P dataset at 1823-1923 K, and 4-5 wt% P2O5 at 1873 K. The 1-4 wt% gap,
P2O5 below 0.2 wt% or above 5 wt%, other temperatures, MnO, unsupported
matrix oxides, missing MgO, and very low SiO2 are reported in the typed
`imcc_p2o5_activity_domain_notice`. The no-Mn row does not use the paper's
separate `13 +/- 9` MnO coefficient. An MgO input does not establish that the
melt is MgO-saturated. Outside these domains the helper returns an
extrapolated estimate with the notice; it does not refuse the calculation.

The Turkdogan (2000) Eq. 13 and Eq. 25 regressions form a comparison spread,
not an uncertainty interval and not an input to the central estimate. Each is
converted from its own published liquid standard before comparison. The
source equations, derivation, notices, and limitations are recorded in
[`fragments/p.md`](fragments/p.md) and
[`rows.json`](../src/openimcc/data/fragments/p/rows.json).

The activity is a dilute tracer approximation and does not model Fe-P alloy
partitioning during reduction or Fe(III) phosphate chemistry. The existing
gas rows for P, P2, P4, PO, PO2, P4O6, and P4O10 use the supplied P2O5
activity and oxygen fugacity. With an explicit `parent_oxides` list, include
`P2O5`; mapping-only calls recognize the supplied key. These are equilibrium
pressures, not finite-rate boil-off. A typed
`p_gas_exceeds_one_bar_complete_volatilisation` notice marks results where
the fixed-composition equilibrium is not physical.

References: Suito and Inoue, *Transactions of the Iron and Steel Institute of
Japan* 24 (1984), 40-46, [doi:10.2355/isijinternational1966.24.40](https://doi.org/10.2355/isijinternational1966.24.40);
Suito, Inoue, and Takada, *Transactions of the Iron and Steel Institute of
Japan* 21 (1981), 250-259,
[doi:10.2355/isijinternational1966.21.250](https://doi.org/10.2355/isijinternational1966.21.250);
Turkdogan, *ISIJ International* 40 (2000), 964-970,
[doi:10.2355/isijinternational.40.964](https://doi.org/10.2355/isijinternational.40.964);
Jung and Hudon, *Journal of the American Ceramic Society* 95 (2012),
3665-3672, [doi:10.1111/j.1551-2916.2012.05382.x](https://doi.org/10.1111/j.1551-2916.2012.05382.x).
