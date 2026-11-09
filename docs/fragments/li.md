# Lithium melt-activity fragment

This fragment supplies the Li₂O parent standard and two lithium-silicate association rows packaged in the default redox v5 pack. The gas table already contains Li(g), LiO(g), Li₂O(g), and Li₂O₂(g), so it adds the missing melt-side Li₂O activity without changing those gas rows.

The fragment rows are included in the immutable redox v5 default. The source data remains the assembly input for a later pack version; no overlay loader is used.

## Rows and source construction

All enthalpies are kJ mol⁻¹, entropies and heat capacities are J mol⁻¹ K⁻¹, Gibbs energies are J mol⁻¹, and temperatures are K. Reaction rows use the existing product-positive Hess convention and the current SiO₂(l) reference.

| Standard | Source anchors and construction | Status |
|---|---|---|
| Li(g) | Existing JANAF Li-005 Shomate intervals, copied unchanged. At 2000 K, `159.300 + 35.386 − 2000×0.178349 = −162.012 kJ mol⁻¹`. | Source assessed; no separate accuracy band. |
| Li₂O(l) | Retains the exact existing Li-015 liquid fit. At 1843 K, `ΔHfus = [−553.247 + 144.742] − [−598.730 + 131.649] = 58.576`; `TmΔSfus = 1843×(213.153−181.370)/1000 = 58.576069`. The printed liquid heat capacity is 100.416. The paired crystal and liquid values close within their printed precision. | Source assessed; supercooled below 1843 K. |
| Li₂SiO₃(l) | de Abreu and Fabrichnaya Table 5 gives `ΔfH°298 = −1646.1` by solution calorimetry and `S°298 = 79.8` by calorimetry. The Li-020 crystal transition row supplies `H−H298 = 169.896` and `S = 298.915` at 1474 K. Téqui et al. measured `ΔHfus = 71.3 ± 0.6` at 1474 K; the Li-021 liquid transition row supplies `Cp = 167.360`. At fusion the selected crystal Gibbs anchor is about −1916.075 kJ mol⁻¹. | Provisional mixed-source branch: JANAF Li-021 implies about 28.033 kJ mol⁻¹ for fusion, 43.267 kJ mol⁻¹ below the direct calorimetry value. The direct fusion value is retained and the old JANAF liquid Cp is flagged. |
| Li₄SiO₄(l) | de Abreu and Fabrichnaya Table 5 gives `ΔfH°298 = −2328.9` by KEMS and an assessed `S°298 = 126`. Kleykamp's measured enthalpy relation is extrapolated from 1300 K to the 1531 K transition. Its derivative gives `Cp = 73.694 + 0.206420T + 4163115/T²`; integration gives a 1531 K crystal enthalpy increment of 334.843 and entropy of 523.582. Kleykamp reports `ΔHfus = 53`. | Provisional: the 126 entropy is assessed, the enthalpy curve is extrapolated 231 K, and liquid Cp is inferred by Hess closure to the liquid KEMS reaction. The phase relation is reported differently across sources. |

The Li₄SiO₄ effective liquid heat capacity is 233.911. It is solved from the printed liquid reaction enthalpy `Li₄SiO₄(l) → 2Li(g) + ½O₂(g) + Li₂SiO₃(l)`, `ΔH°298 = 946.34 ± 0.73`, using JANAF Li(g) and the selected Li₂SiO₃ liquid enthalpy. This is a construction input, not independent validation. With the selected crystalline formation enthalpies, the separate solid KEMS reaction predicts 1001.40 kJ mol⁻¹; Ihle et al. report 960.70 ± 2.38 for Type A and 957.26 ± 2.69 for Type B. That source mismatch is unresolved and remains visible in the fragment flags.

The redox rows are:

- `2 Li(g) + ½ O₂(g) = Li₂O(l)`
- `Li₂O(l) + SiO₂(l) = Li₂SiO₃(l)`
- `2 Li₂O(l) + SiO₂(l) = Li₄SiO₄(l)`

For a constructed liquid from a crystal anchor, the row uses `G_l(Tm) = G_cr(Tm)` and `ΔSfus = ΔHfus/Tm`. Away from fusion, the liquid enthalpy and entropy continue with the selected constant liquid Cp; the fitter stores the Gibbs polynomial in the existing condensate-φ form. No activity measurements were used to select or tune these values.

## Scope and flags

The reaction rows are limited to 1323–2473 K. The thermo evaluator intervals cover 1200–3000 K so the existing redox evaluator can select its rows normally; that wider evaluator interval is not the fragment's evidence domain. Pure liquid standards below their transition temperatures are supercooled continuations. Li₄SiO₄'s transition and congruent/incongruent phase behavior differ across published phase-equilibrium sources, so its pure-liquid row is an association standard and does not assert congruent melting.

The fragment carries `imcc_li_associate_pack_notice` and flags its unvalidated pack identity, mixed-source standards, source domain, and provisional Li₄SiO₄ phase relation. There is no statistical accuracy band for the provisional associates. The reported polynomial residuals are fit residuals only, not source uncertainty or activity accuracy.

Known limits:

- No Li-bearing melt activity measurement validates either association row.
- Li₂SiO₃ combines direct fusion calorimetry with the older JANAF liquid heat capacity because the available branches disagree.
- Li₄SiO₄ liquid Cp is inferred from one KEMS reaction. The independent solid KEMS comparison does not close.
- No lithium phosphate or other Li-bearing associate is included.
- The output supplies standard-state melt thermochemistry. It does not predict Li₂O activity from bulk composition without the existing redox solver.

## Battery and smoke results

The Li row in the coverage ladder identifies zero non-scored battery rows in the process window. The census and coverage-gap CSV also contain no Li matches. Counts by gas species and source group are therefore:

| Gas species | Battery rows | Source groups |
|---|---:|---:|
| Li | 0 | 0 |
| LiO | 0 | 0 |
| Li₂O | 0 | 0 |
| Li₂O₂ | 0 | 0 |
| **Total** | **0** | **0** |

There are no battery predicted-versus-printed Li pairs to score. The following are source checks only, not battery predictions:

- Li₂O JANAF fusion: 58.576 kJ mol⁻¹ from the printed enthalpies versus 58.576069 kJ mol⁻¹ from `TmΔSfus`.
- Li₂SiO₃ fusion: selected direct value 71.3 ± 0.6 kJ mol⁻¹ versus 28.033 kJ mol⁻¹ implied by the older JANAF branches.
- Li₄SiO₄ solid KEMS reaction: 1001.40 kJ mol⁻¹ from the selected formation enthalpies versus 960.70 ± 2.38 (Type A) and 957.26 ± 2.69 (Type B) printed.
- The liquid Li₄SiO₄ KEMS reaction closes at 946.34 kJ mol⁻¹ because it sets the inferred liquid Cp; it is not a holdout check.

The end-to-end smoke uses a regolith-like oxide feed with 0.01 mol Li atoms at 2000 K. The overlay gives `a(Li₂O) = 3.0307×10⁻⁶` and finite pressures: Li `1.3821×10⁻⁶` bar, LiO `3.3889×10⁻⁷` bar, Li₂O `1.2435×10⁻⁸` bar, and Li₂O₂ `3.0841×10⁻¹⁰` bar. Adding the Li₂O activity leaves all 30 default non-Li gas outputs bit-identical for the same activities and oxygen fugacity. On a feed with no Li, the default and overlaid redox solvers also return bit-identical oxygen fugacity and non-Li parent activities.

## References

- Chase, NIST-JANAF Thermochemical Tables, 4th ed., Monograph 9, Tables Li-005, Li-014, and Li-015 ([JANAF Li-005](https://janaf.nist.gov/tables/Li-005.html), [Li-014](https://janaf.nist.gov/tables/Li-014.html), [Li-015](https://janaf.nist.gov/tables/Li-015.html); DOI [10.18434/T42S31](https://doi.org/10.18434/T42S31)).
- Chase, NIST-JANAF Tables Li-020 and Li-021, Li₂SiO₃ crystal and liquid transition rows ([Li-020](https://janaf.nist.gov/tables/Li-020.html), [Li-021](https://janaf.nist.gov/tables/Li-021.html); DOI [10.18434/T42S31](https://doi.org/10.18434/T42S31)).
- de Abreu and Fabrichnaya, “Critical Experiments and Thermodynamic Modeling of the Li₂O-SiO₂ System,” *Solids* 5 (2024), Table 5, p. 315, and Section 2/Table 6 for phase-relation variants ([DOI 10.3390/solids5020020](https://doi.org/10.3390/solids5020020)).
- Téqui, Grinspan, and Richet, “Thermodynamic Properties of Alkali Silicates: Heat Capacity of Li₂SiO₃ and Lithium-Bearing Melts,” *J. Am. Ceram. Soc.* 75 (1992), pp. 2601–2604 ([DOI 10.1111/j.1151-2916.1992.tb05621.x](https://doi.org/10.1111/j.1151-2916.1992.tb05621.x)).
- Kleykamp, “Enthalpy, heat capacity, second-order transitions and enthalpy of fusion of Li₄SiO₄ by high-temperature calorimetry,” *Thermochim. Acta* 287 (1996), pp. 191–201 ([DOI 10.1016/S0040-6031(96)02996-6](https://doi.org/10.1016/S0040-6031(96)02996-6)).
- Penzhorn, Ihle, and Schuster, “The thermochemistry of lithium silicates in view of their use as breeder materials,” *Fusion Eng. Des.* 8 (1989), pp. 393–397 ([DOI 10.1016/S0920-3796(89)80138-3](https://doi.org/10.1016/S0920-3796(89)80138-3)); KEMS reaction values also appear in KfK/Euratom Semi-Annual Report KfK 4677 (November 1989), section BS NN-D-8, Table 9, p. 60 ([report](https://publikationen.bibliothek.kit.edu/270028373/18331037)).
