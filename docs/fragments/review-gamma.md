# Review-gamma trace activities

The source-railed trace activity rows use one format and evaluator. A printed bare `range` uses its geometric centre; when a source prints a central value and uncertainty, the point is that printed centre; `activity_range` is calculated from both printed gamma endpoints after applying the row's parent power. `x_source` is the source-component mole fraction, with the parent formula expanded into the declared number of component units. Temperature selects the nearest source row. A composition screen adds a domain flag; Sn's explicit binary-slag rows are selected only when their binary screen is met.

| Element | Gamma law and value | Component basis | Parent power | Window and composition screen |
| --- | --- | --- | ---: | --- |
| Sn, FCS silicate | range 5.0–6.2 at 1573 K; geometric centre 5.5678 | SnO(l) formula-unit mole fraction | 1 | 1573 K; FeO-CaO-SiO2 |
| Sn, fayalite binary | range 1.0–2.0 at 1573 K | SnO(l) formula-unit mole fraction | 1 | 1573 K; exact FeO-SiO2 binary |
| Sn, ferrite binary | range 1.0–2.0 at 1573 K | SnO(l) formula-unit mole fraction | 1 | 1573 K; exact FeO-CaO binary |
| Ga | range 0.030–0.042 at 1700 K; printed point 0.036 | liquid GaO1.5 formula-unit mole fraction; two units per Ga2O3 parent | 2 | 1550–1740 K, ΔIW +1.5 to +2.5; An-Di target |
| In | range 0.005–0.029 at 1674 K; printed point 0.017 | liquid InO1.5 formula-unit mole fraction; two units per In2O3 parent | 2 | 1550–1740 K, ΔIW +1.5 to +2.5; An-Di target |
| Cu, 1573 K | range 9.0–11.0; geometric centre 9.9499 | liquid CuO0.5 component mole fraction | 2 | 1573 K; (F)CMAS family flag |
| Cu, 1923 K | range 3.5–3.5 | liquid CuO0.5 component mole fraction | 2 | 1500–2100 K; CMAS family flag |
| GeO2 | range 0.24–1.50 at 1573 K; geometric centre 0.6000 | liquid GeO2 formula-unit mole fraction | 1 | 1573 K; FeO-CaO-MgO-SiO2 |
| As | range 0.03–0.56 at 1573 K; geometric centre 0.1296 | liquid AsO1.5 component mole fraction | 2 | 1573 K; FeO-CaO-SiO2 |
| Bi | range 0.3–0.5 at 1573 K; geometric centre 0.3873 | liquid BiO1.5 component mole fraction | 2 | 1573 K; FeO-CaO-SiO2 |
| Ni | range 2.18–3.22 at 1573.15–1873.15 K; printed point 2.70 | liquid NiO simple-oxide mole fraction | 1 | ΔIW −3 to +1.5; FeO 0–12 wt%; MgO 4–30 wt% |
| Co | range 1.23–1.79 at 1573.15–1873.15 K; printed point 1.51 | liquid CoO simple-oxide mole fraction | 1 | ΔIW −3 to +1.5; FeO 0–12 wt%; MgO 4–30 wt%, with a source note above 20 wt% |
| Pb | point 0.22 at 1923 K; uncertainty not stated | liquid PbO simple-oxide mole fraction | 1 | BW1102 five-oxide family; fO2 unreported |
| W | range 0.001–0.05 at 1923 K; geometric centre 0.007071 | liquid WO3 simple-oxide mole fraction | 1 | Di-An-Fo; fO2 window unreported; contour-read provisional |
| Ce, Al/Mg-bearing | Chen Eq. (3): `ln γ = 3.46*((wt% CaO + 0.72 wt% Al2O3)/wt% SiO2) - 14.93`; closure RMS 0.2919 | Ce2O3 formula-unit mole fraction | 1 | 1873 K; CMAS-like Table 2 composition screen; C/CO oxygen potential |
| Ce, Al2O3 = 0 | Zhao Eq. [6]: `ln γ = 4.5 - 14.5*x_CaO² - 27.7*x_SiO2² - 17.5*x_CaO*x_SiO2`; closure RMS 0.0751 | Ce2O3 formula-unit mole fraction | 1 | 1873 K; Al-free CaO-SiO2-Ce2O3 Table IV screen; C/CO oxygen potential |

Both Ce activity equations use the pure solid Ce2O3 source standard, while the gas parent is pure liquid Ce2O3(l). The evaluator converts between them with `exp(-ΔG_fus/(RT))`, where `ΔG_fus = ΔH_fus*(1 - T/Tm)`. At 1873 K, `Tm=2512 K` and the estimated `ΔH_fus=85 kJ/mol` give a central factor of about 0.249, so the liquid-standard activity is smaller below Tm. The 85±10 kJ/mol value is estimated from La2O3 analogue entropies, not directly measured for Ce2O3. Each gamma band combines its equation's ln-gamma closure RMS with the propagated fusion-enthalpy endpoints in log space. Chen is selected by default for Al-bearing compositions and was measured at 5 wt% MgO and 10 wt% Ce2O3; Zhao is selected when Al2O3 is zero. Both equations are single-temperature screens for the source families at 1873 K and `pO2=4.76×10⁻¹⁶ atm` under graphite/pure-CO C/CO conditions; the evaluator flags temperature, oxygen-potential, and composition-range departures.

Ni and Co now use this shared evaluator while retaining their source-specific domain flags and result types. The simple-oxide fraction path preserves the original `n/sum(n)` arithmetic; Co retains its reported MgO increase note above 20 wt%.

Wood & Wade (2013), pp. 916–918, supplies the W and Pb rows. The Pb value is for the analyzed BW1102 silica-saturated melt; no uncertainty for its activity coefficient is stated, so its band is the point 0.22. The W endpoints are inferred from Fig. 4 contour labels and remain marked `contour-read`; the full 0.001–0.05 range is surfaced as its band. The default liquid PbO and WO3 parent rows cover 1923 K and use matching pure-liquid component standards, with one parent formula unit per source component.

Wood & Wade's inferred NiO range of 1.5–3 at 1923 K overlaps the existing Holzheid Ni point of 2.70 and reported spread of 0.52. The Holzheid temperature window ends at 1873.15 K, so this is a value-level cross-check outside that window, not same-domain validation; it is not an additional Ni row.

Sn values use Sossi & Fegley (2018), Table 2, which cites Takeda & Yazawa (1989) for the FCS band; the binary range is from Takeda et al. (1990). The standard state is liquid SnO, matching the NG-1848 parent. The NASA/Glenn card cites Gurvich (1991) and covers 1250–2000 K, including 1573 K; only the table's 1200–1250 K segment is a constant-Cp continuation. I found no same-temperature, independently parameterized pure-liquid SnO Gibbs function to quantify a standard offset, so the available evidence does not show a Ga/In-like continuation mismatch at the Sn measurement temperature.

I also checked the SnO(l) function compiled by Lamoreaux, Hildenbrand & Brewer (1987), *Journal of Physical and Chemical Reference Data* 16, DOI 10.1063/1.555799. It covers the same 1250–2000 K interval but is a selected compilation of high-temperature oxide data, not an independently parameterized comparator; I found no separate source-grounded SnO(l) value at 1573 K from which to calculate an offset.

Ga and In use the direct liquid-standard KEMS results in Bischof et al. (2023), Section 4.4.4: γ(GaO1.5)=0.036±0.006 at 1700 K and γ(InO1.5)=0.017±0.012 at 1674 K. The point is the printed central value, with the printed endpoints retained as a band. Both rows leave the fusion block empty because these coefficients are already for liquid MO1.5. Table 9's linear coefficients are not used as melt activities. Bischof's Table 8 `x` values come from LA-ICP-MS and are the MO1.5 formula-unit mole fraction: x(GaO1.5)=0.0172 for 429 ppm Ga (using the 695 ppm 2_low value for 429 ppm in 1_low) and x(InO1.5)=6.9×10⁻⁵ for the 0.0065–0.05 ppm 1_low surface range. These Table 8 fractions are not the analytical-ppm column converted to an ordinary bulk fraction, and that column is not used to validate the Table 8 basis. For openimcc callers, runtime `x_source` is computed from their melt composition on the same formula-unit MO1.5 basis by expanding each M2O3 parent into two MO1.5 units.

## Ga/In parent standard

Bischof's pure-oxide equilibrium constants are pressure-derived KEMS results for Ga2O3(s) and In2O3(s), reported in Bischof et al. (2023), *Calphad* 80, 102507, DOI 10.1016/j.calphad.2022.102507, Section 3.2, Eqs. (18)–(21), Figures 7–8, and the article's supplementary figure data. Calphad Eq. 19 sets the pressure standard to p0=10^5 Pa. Its third-law reaction enthalpies separately use FactSage Gibbs-energy functions (Section 3.2, Eqs. (27)–(30)): Fig. 9 identifies the Ga reference data as Chaplygin (1975), Hultgren et al. (1973), and Medvedev et al. (1981); Fig. 10 identifies the In set as Barin (1995), Chaplygin (1975), Hultgren et al. (1973), and Medvedev et al. (1981). Those database-derived third-law values are not the measured K used here. The 2023 GCA paper converts the empirical solid-oxide KEMS constants to liquid MO1.5 using the fusion correction in Eq. (18), then reports the liquid-standard fits in Eqs. (20) and (21). These pressure-derived K fits are used at p0=10^5 Pa. The opt-in table reconstructs parent Gibbs functions with the shipped JANAF gas functions. This measured KEMS basis is better founded in the 1550–1740 K experimental window than the engine's NASA-Glenn constant-Cp supercooled-liquid continuation.

The opt-in `condensate-ga-in-kems-candidate.csv` fits the parent liquid Gibbs functions from those liquid component constants. For component KEMS fit `ln K = -A/T + B`, the table uses

```text
ln K(M2O3,l), p0=1e5 Pa = 2 * (-A/T + B)
G°(M2O3,l) = 2 G°(M,g) + 1.5 G°(O2,g) + R T ln K(M2O3,l)
```

The parent equation is fitted over 1550–1740 K only. Rows outside that interval retain functions copied from `condensate-default-v1.csv`. The CSV source-label comment records the conversion. The default condensate table and selected defaults are unchanged. The candidate table's generated Gibbs coefficients are a fit of the published KEMS constants, not a replacement for the published table or a default selection.

When the default condensate table is active, each Ga/In gas flag carries the numeric `parent_liquid_standard_offset = sqrt(K_default/K_Bischof,parent)` and recommends the opt-in KEMS table. The activity is not rescaled. At 1586.4, 1630.8, 1674.2 and 1695.4 K, the factors are:

| T (K) | Ga offset | In offset |
| ---: | ---: | ---: |
| 1586.4 | 9.37 | 47.0 |
| 1630.8 | 6.52 | 37.1 |
| 1674.2 | 4.65 | 29.8 |
| 1695.4 | 3.96 | 26.8 |

## Closure against Bischof Table S1

The earlier Table 9 activity with analytical ppm gave the prior `p/p(S1)` ratios below. The closure uses Table 8 `x` (Ga 0.0172; In 6.9×10⁻⁵), each S1 row's own γ (Ga 0.0632, 0.0285, 0.0248, 0.0246; In 0.0527, 0.0377, 0.0301, 0.0254), activity `(γ x)^2`, and the opt-in parent. The earlier and closure columns use different source compositions, so read the latter as the Table 8/S1 closure check:

| T (K) | Ga prior | Ga Table 8 closure | In prior | In Table 8 closure |
| ---: | ---: | ---: | ---: | ---: |
| 1586.4 | 3.29 | 0.9166 | 24.1 | 1.0449 |
| 1630.8 | 5.57 | 0.9159 | 27.6 | 1.0461 |
| 1674.2 | 5.00 | 0.9190 | 28.6 | 1.0463 |
| 1695.4 | 4.47 | 0.9187 | 31.1 | 1.0494 |

With the single printed central γ (Ga 0.036; In 0.017) the same comparison is Ga 0.52/1.16/1.33/1.34 and In 0.34/0.47/0.59/0.70. The central value does not capture the per-row γ spread, which is why the endpoint band is surfaced to callers.

These closures use Table 8's measured component fractions; the analytical-ppm column is not substituted for those `x` values. Runtime callers instead supply a melt composition, from which openimcc computes the same formula-unit `x_source` basis.

Cu, GeO2, As and Bi ranges are from the cited primary melt studies as summarized in Sossi & Fegley (2018), Table 2. The Cu family text is a flag, not the temperature selector. The As parent remains described as the engine's tier-4 liquid-from-solid estimate, not as independently measured pure liquid As2O3. GeO stays inactive and V remains withheld.
