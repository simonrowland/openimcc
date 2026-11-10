# Review-gamma trace activities

These seven trace activities use one row format and one evaluator. The point value for a `range` law is the geometric centre; `activity_range` is calculated from both printed gamma endpoints after applying the row's parent power. `x_source` is the source-component mole fraction, with the parent formula expanded into the declared number of component units. Temperature selects the nearest source row. A composition screen adds a domain flag; Sn's explicit binary-slag rows are selected only when their binary screen is met.

| Element | Gamma law and value | Component basis | Parent power | Window and composition screen |
| --- | --- | --- | ---: | --- |
| Sn, FCS silicate | range 5.0–6.2 at 1573 K; geometric centre 5.5678 | SnO(l) formula-unit mole fraction | 1 | 1573 K; FeO-CaO-SiO2 |
| Sn, fayalite binary | range 1.0–2.0 at 1573 K | SnO(l) formula-unit mole fraction | 1 | 1573 K; exact FeO-SiO2 binary |
| Sn, ferrite binary | range 1.0–2.0 at 1573 K | SnO(l) formula-unit mole fraction | 1 | 1573 K; exact FeO-CaO binary |
| Ga | range 0.030–0.042 at 1700 K; geometric centre 0.03550 | liquid GaO1.5 component mole fraction, two units per Ga2O3 parent | 2 | 1550–1740 K, ΔIW +1.5 to +2.5; An-Di target |
| In | range 0.005–0.029 at 1674 K; geometric centre 0.01204 | liquid InO1.5 component mole fraction, two units per In2O3 parent | 2 | 1550–1740 K, ΔIW +1.5 to +2.5; An-Di target |
| Cu, 1573 K | range 9.0–11.0; geometric centre 9.9499 | liquid CuO0.5 component mole fraction | 2 | 1573 K; (F)CMAS family flag |
| Cu, 1923 K | range 3.5–3.5 | liquid CuO0.5 component mole fraction | 2 | 1500–2100 K; CMAS family flag |
| GeO2 | range 0.24–1.50 at 1573 K; geometric centre 0.6000 | liquid GeO2 formula-unit mole fraction | 1 | 1573 K; FeO-CaO-MgO-SiO2 |
| As | range 0.03–0.56 at 1573 K; geometric centre 0.1296 | liquid AsO1.5 component mole fraction | 2 | 1573 K; FeO-CaO-SiO2 |
| Bi | range 0.3–0.5 at 1573 K; geometric centre 0.3873 | liquid BiO1.5 component mole fraction | 2 | 1573 K; FeO-CaO-SiO2 |

Sn values use Sossi & Fegley (2018), Table 2, which cites Takeda & Yazawa (1989) for the FCS band; the binary range is from Takeda et al. (1990). The standard state is liquid SnO, matching the NG-1848 parent. The NASA/Glenn card cites Gurvich (1991) and covers 1250–2000 K, including 1573 K; only the table's 1200–1250 K segment is a constant-Cp continuation. I found no same-temperature, independently parameterized pure-liquid SnO Gibbs function to quantify a standard offset, so the available evidence does not show a Ga/In-like continuation mismatch at the Sn measurement temperature.

I also checked the SnO(l) function compiled by Lamoreaux, Hildenbrand & Brewer (1987), *Journal of Physical and Chemical Reference Data* 16, DOI 10.1063/1.555799. It covers the same 1250–2000 K interval but is a selected compilation of high-temperature oxide data, not an independently parameterized comparator; I found no separate source-grounded SnO(l) value at 1573 K from which to calculate an offset.

Ga and In use the direct liquid-standard KEMS results in Bischof et al. (2023), Section 4.4.4: γ(GaO1.5)=0.036±0.006 at 1700 K and γ(InO1.5)=0.017±0.012 at 1674 K. Both rows leave the fusion block empty because these coefficients are already for liquid MO1.5. Table 9's linear coefficients are not used as melt activities. Bischof derives x from LA-ICP-MS measurements and calls it the MO1.5 component mole fraction; the helper uses that declared component basis by expanding each M2O3 parent into two MO1.5 units. Table 8 reports x(GaO1.5)=0.0172 for 429 ppm Ga, about 47 times the ordinary component mole fraction reconstructed from that concentration and the An-Di bulk. Its note says the x from the 695 ppm 2_low run is also used for the 429 ppm 1_low polytherm run. It reports x(InO1.5)=6.9×10⁻⁵ for the 0.0065–0.05 ppm 1_low surface range; at 0.05 ppm this is about 2650 times the ordinary bulk-derived component fraction. Both published anchors and the In discrepancy are recorded in the rows and surfaced in gas flags. Runtime `x_source` is calculated from caller composition on the declared component basis. The source anchors and bulk conversions do not reconcile, so pressure ratios at those analytical concentrations do not validate composition transfer.

## Ga/In parent standard

Bischof's pure-oxide equilibrium constants are pressure-derived KEMS results for Ga2O3(s) and In2O3(s), reported in Bischof et al. (2023), *Calphad* 80, 102507, DOI 10.1016/j.calphad.2022.102507, Section 3.2, Eqs. (18)–(21), Figures 7–8, and the article's supplementary figure data. The measured K uses p°=1 bar. Its third-law reaction enthalpies separately use FactSage Gibbs-energy functions (Section 3.2, Eqs. (27)–(30)): Fig. 9 identifies the Ga reference data as Chaplygin (1975), Hultgren et al. (1973), and Medvedev et al. (1981); Fig. 10 identifies the In set as Barin (1995), Chaplygin (1975), Hultgren et al. (1973), and Medvedev et al. (1981). Those database-derived third-law values are not the measured K used here. The 2023 GCA paper converts the empirical solid-oxide KEMS constants to liquid MO1.5 using the fusion correction in Eq. (18), then reports the liquid-standard fits in Eqs. (20) and (21), with p°=1 atm. The opt-in table converts those liquid K fits to the engine's 1 bar standard and reconstructs parent Gibbs functions with the shipped JANAF gas functions. This measured KEMS basis is better founded in the 1550–1740 K experimental window than the engine's NASA-Glenn constant-Cp supercooled-liquid continuation.

The opt-in `condensate-ga-in-kems-candidate.csv` fits the parent liquid Gibbs functions from those liquid component constants. For component KEMS fit `ln K = -A/T + B`, the table uses

```text
ln K(M2O3,l), 1 bar = 2 * (-A/T + B + 1.75 * ln(1.01325))
G°(M2O3,l) = 2 G°(M,g) + 1.5 G°(O2,g) + R T ln K(M2O3,l)
```

The 1.75 factor converts the component reaction's 1-atm gas standard to the engine's 1-bar standard. The parent equation is fitted over 1550–1740 K only. Rows outside that interval retain the prior table functions. The CSV source-label comment records the conversion. The default condensate table and selected defaults are unchanged. The candidate table's generated Gibbs coefficients are a fit of the published KEMS constants, not a replacement for the published table or a default selection.

At 1586.4, 1630.8, 1674.2 and 1695.4 K, the default parent standard alone gives pressure factors `sqrt(K_default/K_Bischof)` of 9.16, 6.37, 4.54 and 3.87 for Ga, and 45.91, 36.27, 29.10 and 26.21 for In. The opt-in table reduces those pure-parent factors to 1.000000 within about 2×10⁻⁶. These are parent-standard factors with the same gamma and source fraction on both sides.

The earlier review's full-pressure comparisons for its Table 9 activity were 3.29, 5.57, 5.00 and 4.47 for Ga at 429 ppm; and 24.1, 27.6, 28.6 and 31.1 for In at 0.05 ppm. Replacing that activity with the current runtime composition-derived `x_source`, the direct KEMS gamma point, and the opt-in parent standard gives the rebased diagnostics below. The after point values use the geometric-centre gamma; brackets propagate the printed endpoint gamma range through `a=(γx)^2`.

| T (K) | Ga before `p/p(S1)` | Ga after `p/p(S1)` | In before `p/p(S1)` | In after `p/p(S1)` |
| ---: | ---: | ---: | ---: | ---: |
| 1586.4 | 3.29 | 0.000353 [0.000252, 0.000494] | 24.1 | 1.62×10⁻⁸ [2.79×10⁻⁹, 9.38×10⁻⁸] |
| 1630.8 | 5.57 | 0.000710 [0.000507, 0.000994] | 27.6 | 2.19×10⁻⁸ [3.77×10⁻⁹, 1.27×10⁻⁷] |
| 1674.2 | 5.00 | 0.000750 [0.000536, 0.00105] | 28.6 | 2.65×10⁻⁸ [4.57×10⁻⁹, 1.54×10⁻⁷] |
| 1695.4 | 4.47 | 0.000725 [0.000518, 0.00101] | 31.1 | 3.10×10⁻⁸ [5.35×10⁻⁹, 1.80×10⁻⁷] |

These after values are derived from the review's rounded before ratios, the printed gamma endpoints, the current formula-unit/component conversions, and the candidate's pure-parent correction; they are not a rerun against raw S1 measurements. The full `kems-137` simulator extract was not available in this workspace. The reported x discrepancies make the low after ratios a warning about composition transfer, not evidence that the corrected KEMS model should predict those pressures. The pure-parent correction itself is independently checked against the printed K fits.

Cu, GeO2, As and Bi ranges are from the cited primary melt studies as summarized in Sossi & Fegley (2018), Table 2. The Cu family text is a flag, not the temperature selector. The As parent remains described as the engine's tier-4 liquid-from-solid estimate, not as independently measured pure liquid As2O3. GeO stays inactive and V remains withheld. Ni and Co continue through their existing helpers.
