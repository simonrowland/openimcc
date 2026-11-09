# Zinc melt fragment

The Zn association row is included in the immutable redox v5 default. This
source fragment remains the assembly input for a later pack version.

This fragment adds `ZnO(l)` as a registered redox parent, a liquid-zinc
reference, and one `Zn2SiO4(l)` associate. It follows the redox-pack extension
pattern used for manganese. It does not add a general ZnO activity model: the
available vapor measurements are for ZnO–phosphate melts at high ZnO mole
fractions, not dilute lunar or basaltic melts. The associate parameters below
come from independent compound thermochemistry and were not adjusted to those
measurements or to battery rows.

## Standards and derivations

### Zn(l)

`Zn(l)` uses the direct liquid-zinc H and S nodes in NIST-JANAF table Zn-003,
from 1200 to 2000 K. The 298.15 K reference enthalpy is the printed
`+6.519 kJ mol−1`. The condensate-phi row fits

`G(T) = (Hf,298 + Hinc) × 1000 − T S`.

At 1300 K, the printed values are `Hinc=30.684 kJ mol−1` and
`S=94.914 J mol−1 K−1`, giving `G=−86185.2 J mol−1`. The row fit's maximum
source residual is 0.575 J mol−1. The liquid standard is above JANAF's
1180.173 K one-bar fugacity marker; it is used as a condensed reference state,
not as a vapor-pressure prediction.

Source: Chase (1998), *NIST-JANAF Thermochemical Tables*, Fourth Edition,
table Zn-003, liquid Zn rows 1200–2000 K and the 298.15 K reference value;
[DOI 10.18434/T42S31](https://doi.org/10.18434/T42S31).

### ZnO(l)

The fragment copies the existing default `ZnO(l)` coefficients unchanged.
Their source construction uses the TSIV ZnO crystal Gibbs function, a fusion
step, and an estimated liquid heat capacity. TSIV gives
`ΔfH°298(ZnO,cr)=−350.5±1.0 kJ mol−1`, `Tfus=2250±30 K`, and liquid
`Cp=67 J mol−1 K−1` (estimated). The paired Table 1603 enthalpy entries at
2250 K are 112.162 kJ mol−1 for the crystal and 182.162 kJ mol−1 for the
liquid, so their difference is 70.000 kJ mol−1, consistent with TSIV's
`ΔHfus=70±10 kJ mol−1` estimate. This does not turn either liquid property
into a direct measurement.

Source: TSIV, ZnO(cr,l), table 1603 and accompanying source discussion,
[Table 1603](https://www.chem.msu.ru/rus/tsiv/Zn/print-ZnO_c.html).

### Zn2SiO4(l)

The solid 298.15 K formation enthalpy is re-anchored to the active ZnO and
quartz standards using King’s primary calorimetry:

`ΔH°rxn = Hf(Zn2SiO4) − 2 Hf(ZnO) − Hf(SiO2)`

`Hf(Zn2SiO4) = 2(−350.500) + (−910.857) + (−6990 × 4.184 / 1000)`

`= −1641.10316 kJ mol−1`.

King reports `2ZnO(cr)+SiO2(quartz)→Zn2SiO4(cr)` as
`−6990±140 cal mol−1` (Table III, p. 657). The quartz value
`−910.857 kJ mol−1` is from JANAF O-037 at 298.15 K. Barin's compiled
willemite enthalpy, transcribed as `−1641.5671 kJ mol−1` in Jeon et al.,
differs by 0.464 kJ mol−1. This is within the King reaction uncertainty plus
the ZnO anchor uncertainty; the fragment keeps the primary reaction re-anchored
to the current oxide standards.

The crystal entropy is `131.378 J mol−1 K−1`. The Barin crystal heat-capacity
equation (as transcribed in Jeon et al., Table 2, p. 47006) is

`Cp = 180.07936 + 0.013589632 T − 2528556.71904 T−2`
`     − 685.68429632 T−0.5 + 156170176.512 T−3`

in J mol−1 K−1. It is integrated from 298.15 K to the melting point. Zaitseva
et al. report the `α→β` transition at 1500°C with a 0.13 kJ mol−1 endotherm,
then melting at 1512°C with a 106.44 kJ mol−1 endotherm (Table 1 and Fig. 4,
p. 3). The transition enthalpy and entropy are included before the fusion
step.

There is no direct willemite-liquid Cp row. The provisional constant liquid
value is the oxide-additive estimate
`2(67) + 81.39 = 215.39 J mol−1 K−1`, using TSIV's estimated ZnO liquid Cp
and the measured silica-liquid Cp source of Richet et al. at the fusion anchor.
The liquid branch continues as

`H_l(T) = H_cr(Tfus) + ΔHfus + Cp_l (T−Tfus)`

`S_l(T) = S_cr(Tfus) + ΔHfus/Tfus + Cp_l ln(T/Tfus)`.

The condensate-phi fit over nine 100 K source nodes from 1200–2000 K has a
maximum numerical Gibbs residual of 3.25 J mol−1. That fit residual is not an
accuracy bound. No numeric accuracy interval is claimed because the liquid Cp
estimate has no measured uncertainty.

Sources:

- King (1951), JACS 73(2), 656–658, Table III, p. 657,
  [DOI 10.1021/ja01146a046](https://doi.org/10.1021/ja01146a046).
- Chase (1998), JANAF O-037, quartz formation enthalpy at 298.15 K,
  [DOI 10.18434/T42S31](https://doi.org/10.18434/T42S31).
- Barin (1989), *Thermochemical Data of Pure Substances*, as transcribed by
  Jeon, Michallik, and Lindberg (2025), Table 2, p. 47006,
  [DOI 10.1016/j.ceramint.2025.07.405](https://doi.org/10.1016/j.ceramint.2025.07.405).
- Zaitseva et al. (2019), *Russian Journal of Inorganic Chemistry* 64(1),
  1–6, Table 1 and Fig. 4, p. 3,
  [DOI 10.1134/S0036023619010224](https://doi.org/10.1134/S0036023619010224).
- Richet, Bottinga, and Tequi (1984), liquid silicate heat-capacity source,
  *Journal of the American Ceramic Society* 67, C-6–C-8,
  [DOI 10.1111/j.1151-2916.1984.tb19155.x](https://doi.org/10.1111/j.1151-2916.1984.tb19155.x).

## Scope and flags

- Reaction rows cover 1200–2000 K. The willemite-derived association is a
  provisional single-complex approximation; it has no general activity
  calibration for phosphate-poor basalt or regolith melts.
- When `Zn2SiO4(l)` participates, `evaluate_redox` reports the row's
  `imcc_zn_activity_domain_notice` range flag. The ZnO–phosphate activity data
  are not a valid calibration window for the regolith composition in this
  task, so no dilute phosphate coefficient is applied.
- `ZnO(l)` reuses the existing immutable condensate row. The default pack
  includes the association, while the condensate table remains unchanged.
- A user-supplied trace Zn inventory must still be accompanied by an oxygen
  condition and composition. The gas engine only computes Zn-bearing vapor
  after the melt caller supplies the resulting ZnO activity.

## Battery smoke

The census has one Zn source group in the ZnO–P2O5 vaporization study:

| Source | Species | Records | In the 1323–2473 K window | Unlocked |
| --- | --- | ---: | ---: | ---: |
| ZnO–P2O5 vaporization, Table 2 | Zn | 5 | 0 | 0 |

All five records are at 1300 K, below the stated process window, and the joined
coverage has no complete source composition or oxygen-fugacity input. No
predicted-versus-printed pairs are reported: a prediction would require
inferring missing source inputs, which the census explicitly leaves unknown.
These observations were not used to fit the fragment.
