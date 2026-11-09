# Nickel activity fragment

This opt-in fragment supplies a dilute NiO activity for `evaluate_gas`. It
does not change the eight-oxide melt solve or any published data table. The
caller evaluates the major-oxide melt as usual, then adds the helper's
`NiO` activity to the activity mapping passed to `evaluate_gas`.

## Source and derivation

Holzheid, Palme, and Chakraborty measured NiO, CoO, and FeO activities in
metal-saturated silicate melts. For NiO they report
`gamma_NiO(l) = 2.70 +/- 0.52` from 77 experiments, using simple oxide mole
fractions and pure liquid oxide standard states. Their reported interval is
1300-1600 °C and oxygen fugacity IW-3 to IW+1.5; the reported FeO and MgO
composition intervals are 0-12 wt% and 4-30 wt%, respectively. Citation:
[Holzheid et al. (1997), Chemical Geology 139, 21-38,
doi:10.1016/S0009-2541(97)00030-2](https://doi.org/10.1016/S0009-2541(97)00030-2).

The algebra is `a_NiO = gamma_NiO(l) * x_NiO`. Both factors and the activity
are dimensionless. As a sanity check, `x_NiO = 0.001` gives
`a_NiO = 0.00270` at the reported central coefficient. The `+/- 0.52` value
is carried as the paper's reported spread; it is not relabelled as a standard
error or confidence interval.

The pure-liquid NiO standard is already present in the default gas condensate
table. It cites Gurvich et al., *Thermodynamic Properties of Individual
Substances*, Volume IV, Table 2026, which reports the crystal and liquid
functions, a fusion enthalpy of 42 kJ mol-1 at 2228 K, and liquid heat
capacity of 67 J mol-1 K-1. The helper
uses that same liquid standard and does not reconstruct or replace its Gibbs
row. Its activity-calibration temperatures are below the NiO melting point,
so this liquid standard is supercooled over that whole interval. The public table is available at
[TSIV NiO Table 2026](https://www.chem.msu.ru/rus/tsiv/Ni/print-NiO_c.html).

## Scope and limits

- The function is opt-in and returns `imcc_ni_activity_domain_notice` on every
  result. It flags temperature, supplied oxygen-fugacity offset, FeO, or MgO
  outside the reported intervals. If the oxygen-fugacity offset is omitted,
  the result is also flagged.
- The calibration is from metal-saturated silicate melts. The paper's abstract
  does not provide a complete multicomponent composition envelope or a
  numerical upper mole-fraction limit for dilute NiO. Transfer to regolith-like
  composition is therefore provisional even when the listed axes are inside
  their measured intervals.
- The abstract does not state a numeric NiO mole-fraction ceiling. The helper
  applies the linear dilute relation and carries a provisional notice; callers
  must keep NiO in the dilute regime. It does not model nickel metal
  partitioning. A battery-level Ni vapour prediction also depends on the
  supplied oxygen fugacity and the existing gas thermochemistry.
- No simulator battery rows in the supplied census carry Ni species, so this
  coefficient has no Ni vapour observation pair available for validation.
