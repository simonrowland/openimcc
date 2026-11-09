# Yb silicate association fragment

The default redox v5 pack includes these Yb-bearing silicate associations. The
fragment registers Yb2O3(l) as the Yb(III) activity parent and adds two
associates:

| Reaction | Role |
| --- | --- |
| Yb2O3(l) + SiO2(l) = Yb2SiO5(l) | Monosilicate association |
| Yb2O3(l) + 2 SiO2(l) = Yb2Si2O7(l) | Disilicate association |

The rows use the existing `condensate_phi` species evaluator and redox row
schema. The source fragment remains the assembly input for a later pack
version. `imcc_yb_associate_pack_notice` marks the unvalidated pack identity
and source-domain scope.

## Thermochemical anchors

The `selected_dfH298_kJ_mol` and `selected_S298_J_mol_K` fields in
`rows.json` are crystalline 298.15 K anchors. The corresponding liquid
Gibbs functions are constructed by integrating the crystal heat capacity to
the melting point, applying the fusion entropy, and continuing with the
selected liquid heat capacity. The row reaction Gibbs energies therefore use
the same liquid parent standards as the existing redox model.

| Value | Source and construction |
| --- | --- |
| Yb2O3 crystal ΔfH°298 = −1814.5 ± 6.0 kJ/mol, S°298 = 133.1 ± 0.3 J/(mol K), C→H at 2687 K with 27 ± 5 kJ/mol, H→liquid at 2707 K with 84 ± 8 kJ/mol, liquid Cp = 146 J/(mol K) | Konings et al., *J. Phys. Chem. Ref. Data* 43, 013101 (2014), Table 88, pp. 013101-83–84, [DOI 10.1063/1.4825256](https://doi.org/10.1063/1.4825256). The 27 + 84 = 111 kJ/mol phase-chain increment is checked against the independent drop-and-catch result, 102 ± 10 kJ/mol, in Fyhrie et al., *J. Chem. Thermodyn.* 132 (2019), Table 2, [DOI 10.1016/j.jct.2019.01.008](https://doi.org/10.1016/j.jct.2019.01.008). Konings et al. identify the H and liquid heat capacities as estimates. |
| SiO2 crystal anchor = −905.489 kJ/mol | NIST-JANAF, SiO2 high-cristobalite table O-035, 298.15 K row, [table O-035](https://janaf.nist.gov/tables/O-035.html). This is the crystalline silica reference used by the existing redox SiO2(l) standard. |
| Yb2SiO5 formation from elements = −2774.75 ± 8.21 kJ/mol; from Yb2O3 + SiO2 = −49.45 ± 8.35 kJ/mol | Liang et al., *J. Mater. Res.* 14 (1999), pp. 1181–1185, [DOI 10.1557/JMR.1999.0158](https://doi.org/10.1557/JMR.1999.0158). The crystalline Hess anchor uses the oxide value and the matching crystal references: −1814.5 − 905.489 − 49.45 = −2769.439 kJ/mol. It differs from the elemental value by 5.31 kJ/mol, within the printed uncertainties. The corrected Costa and Jacobson thermochemical cycle independently reports an elemental value of −2774 ± 11 kJ/mol in corrected Table 2, [corrigendum DOI 10.1016/j.jeurceramsoc.2018.03.007](https://doi.org/10.1016/j.jeurceramsoc.2018.03.007). |
| Yb2Si2O7 formation from oxide crystals = −158.87 ± 19.88 kJ/mol | Costa et al., *J. Am. Ceram. Soc.* 105 (2022), Table 3, [DOI 10.1111/jace.18715](https://doi.org/10.1111/jace.18715). At 42 mol% SiO2, the printed solution enthalpies give 2.41 + 2(−12.53) − 136.22 = −158.87 kJ/mol. The uncertainty is the root-sum-square of the printed 13.44, 6.40, and 7.13 kJ/mol uncertainties, with the silica term counted twice. With the same crystal references the product anchor is −1814.5 + 2(−905.489) − 158.87 = −3784.348 kJ/mol. |
| Yb2Si2O7 fusion enthalpy = 151 ± 8 kJ/mol | Costa et al., *J. Am. Ceram. Soc.* 102 (2019), Table 4, [DOI 10.1111/jace.16113](https://doi.org/10.1111/jace.16113); reprinted in Costa et al. (2022), Table 4. |
| Yb2SiO5 fusion temperature ≈ 1950 °C; Yb2Si2O7 ≈ 1850 °C | Toropov and Bondar, *Bulletin of the Academy of Sciences of the USSR, Division of Chemical Science* 10 (1961), pp. 1278–1285; phase diagram reproduced in Bakan et al. (2017), Fig. 5. These are approximate phase-diagram liquidus anchors, not calorimetric melting measurements. |
| Yb2SiO5 heat capacity | Al Nasiri et al., *J. Am. Ceram. Soc.* 99 (2016), pp. 589–596, [DOI 10.1111/jace.13982](https://doi.org/10.1111/jace.13982), report specific-heat measurements for Yb2SiO5 among five monosilicates. The fragment uses provisional 3nR crystal Cp for its eight atoms per formula and scales the Yb2O3 liquid Cp by 8/5. |
| Yb2Si2O7 heat capacity | Bodenschatz, NASA/TM-20230004256 (2023), Fig. 6, pp. 8–10, [NASA report](https://ntrs.nasa.gov/citations/20230004256), presents DFT and ensemble-model Cp estimates and describes the work as a candidate-material screening model. The fragment uses provisional 3nR crystal Cp for eleven atoms per formula and scales the Yb2O3 liquid Cp by 11/5. |
| Crystalline S°298 anchors: Yb2SiO5 = 174.5 and Yb2Si2O7 = 216 J/(mol K) | Paksoy et al., *Corrosion Science* 207 (2022), 110555, Table A1, [DOI 10.1016/j.corsci.2022.110555](https://doi.org/10.1016/j.corsci.2022.110555). These secondary tabulations are used provisionally for the crystal anchors. |

For Yb2Si2O7, the fusion entropy is 151000/2123.15 = 71.1207 J/(mol K).
For Yb2SiO5, no direct fusion enthalpy was found; the provisional estimate is
the disilicate fusion entropy scaled by formula atom count, then multiplied by
the approximate monosilicate melting temperature: (151000/2123.15)(8/11) ×
2223.15/1000 = 114.991 kJ/mol. Both silicate liquid Cp values are provisional
formula-atom scalings from the assessed Yb2O3 value. The source anchor values
are encoded and independently checked in `tests/fragments/test_yb_fragment.py`.

## Scope and limits

- The modeled channel is caller-supplied Yb2O3 activity with Yb(III) only.
  Yb(II), metal formation, phosphate association, and other rare-earth
  associates are not represented.
- Both pure silicate liquid standards are extrapolated below their approximate
  melting points over much of the 1323–2473 K process interval. The
  monosilicate fusion enthalpy, both silicate crystal Cp functions, both
  liquid Cp functions, and the silicate S°298 anchors are provisional.
- Costa and Jacobson's silica activities are reported for the
  Yb2O3 + Yb2SiO5 and Yb2SiO5 + Yb2Si2O7 two-phase fields. The corrected 2018
  corrigendum is [DOI 10.1016/j.jeurceramsoc.2018.03.007](https://doi.org/10.1016/j.jeurceramsoc.2018.03.007), to the 2015 study
  ([DOI 10.1016/j.jeurceramsoc.2015.07.019](https://doi.org/10.1016/j.jeurceramsoc.2015.07.019)). Those solid coexistence data are not a dilute liquid activity coefficient and were not used to fit either association row.
- Costa et al.'s CMAS calorimetry supplies an independent crystalline Hess
  target at a stated melt composition. It does not validate a universal or
  composition-independent Yb activity model.
- The species evidence domain is 1200–3000 K; the reaction rows declare
  1323–2473 K. The notice flag remains attached until the fragment's pack
  identity and source domain receive broader validation.
