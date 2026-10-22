# Dilute P2O5 activity extension

`evaluate_p2o5_activity(composition_moles, T_K)` is an opt-in dilute-solute
estimate. It returns `a(P2O5) = gamma(P2O5) x(P2O5)` on the hypothetical pure
liquid P2O5 standard used by the gas layer. The calculation leaves other parent
activities unchanged to first order; it does not renormalize their activities.
Pass the returned activity as the `P2O5` entry to `evaluate_gas` or
`evaluate_gas_oxygen_balance`.

The central equation is Suito & Inoue (1984), Eq. 14, no-Mn form:

```text
log10(gamma) = -1.02(23 xCaO + 17 xMgO + 8 xFeO - 26 xP2O5)
               - 22990/T + 9.490
```

The selection criterion is written before score comparison: use measured
activity data, an explicit temperature term, and composition terms covering
the main oxides in the melt. This selects Suito & Inoue Eq. 14. The source
liquid standard is converted from the Turkdogan-Pearson 1 atm reaction basis
to the gas pack's Jung-Hudon 1 bar basis. For
`P2(g) + 5/2 O2(g) -> P2O5(l)`, the 3.5 gaseous reactant moles give
`delta G = 3.5 RT ln(1.01325)`; the activity log shift is
`(G_source - G_pack)/(RT ln(10))`.

The Turkdogan (2000) Eq. 13 and Eq. 25 values are two other regressions,
reported as a comparison spread rather than an uncertainty interval around
Suito. Suito can lie outside both: for the in-domain 1873 K hand case it is
1.206 dex above the higher Turkdogan value. They are not blended into the
central activity. Every result
contains a typed `P2O5ActivityNotice` naming the regression, violated domain
axes, and band width in dex, even when no listed axis is violated. The
regression is extrapolated with notice when outside its basis; it does not
refuse the composition. Suito's calibration is MgO-saturated CaO-FeO-SiO2
slags, with low-P measurements across 1823-1923 K and high-P points around
4-5 wt% P2O5 at 1873 K. There is no TiO2 term. Near-zero SiO2, absent MgO,
temperatures outside that interval, TiO2 presence, and P2O5 above 5 wt% are
reported in the notice. The MgO flag reports only that MgO is absent; it does
not test whether an MgO-bearing composition is MgO-saturated or otherwise in
the calibration family.

The existing gas species P, P2, P4, PO, PO2, P4O6 and P4O10 use the same
liquid-standard activity and caller oxygen condition. The oxygen-balance
solver includes a supplied P2O5 parent and its gas products in its existing
oxygen accounting. When a caller supplies an explicit `parent_oxides` list,
include `P2O5` in that list; mapping-only calls recognize it automatically.
These are equilibrium pressures, not a finite-rate boil-off model. If any
P-gas pressure or their total exceeds 1 bar, the gas result carries a typed
`p_gas_exceeds_one_bar_complete_volatilisation` notice: phosphorus would leave
essentially completely, and the fixed-composition dilute equilibrium is not
physical in that regime.

This extension does not model phosphorus partitioning into metallic iron as
Fe-P alloy during carbon reduction. Carbothermic processes can transfer much
of the phosphorus to metal, which belongs to the simulator's metal account.
Fe(III) phosphate chemistry is also outside the model. Fe2O3 input is folded
into the FeO-equivalent FetO term on an Fe-atom basis and carries a typed
notice of this limitation. The activity regression is a dilute tracer
approximation, not a validated phosphorus-bearing melt solution.

References: Suito & Inoue, *ISIJ International* 24 (1984), Eq. 14,
[doi:10.2355/isijinternational1966.24.40](https://doi.org/10.2355/isijinternational1966.24.40);
Turkdogan, *ISIJ International* 40 (2000), Eqs. 13 and 25,
[doi:10.2355/isijinternational.40.964](https://doi.org/10.2355/isijinternational.40.964).
