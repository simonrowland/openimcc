# Review-gamma trace activities

These seven trace activities use one row format and one evaluator. A printed bare `range` uses its geometric centre; when a source prints a central value and uncertainty, the point is that printed centre; `activity_range` is calculated from both printed gamma endpoints after applying the row's parent power. `x_source` is the source-component mole fraction, with the parent formula expanded into the declared number of component units. Temperature selects the nearest source row. A composition screen adds a domain flag; Sn's explicit binary-slag rows are selected only when their binary screen is met.

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

The earlier Table 9 activity with analytical ppm gave these `p/p(S1)` ratios. The closure uses Table 8 `x` (Ga 0.0172; In 6.9×10⁻⁵), printed central γ (Ga 0.036; In 0.017), and the opt-in parent. The earlier and closure columns use different source compositions, so read the latter as the Table 8/S1 closure check:

| T (K) | Ga prior | Ga Table 8 closure | In prior | In Table 8 closure |
| ---: | ---: | ---: | ---: | ---: |
| 1586.4 | 3.29 | 0.938 | 24.1 | 1.069 |
| 1630.8 | 5.57 | 0.937 | 27.6 | 1.070 |
| 1674.2 | 5.00 | 0.940 | 28.6 | 1.071 |
| 1695.4 | 4.47 | 0.940 | 31.1 | 1.074 |

These closures use Table 8's measured component fractions; the analytical-ppm column is not substituted for those `x` values. Runtime callers instead supply a melt composition, from which openimcc computes the same formula-unit `x_source` basis.

Cu, GeO2, As and Bi ranges are from the cited primary melt studies as summarized in Sossi & Fegley (2018), Table 2. The Cu family text is a flag, not the temperature selector. The As parent remains described as the engine's tier-4 liquid-from-solid estimate, not as independently measured pure liquid As2O3. GeO stays inactive and V remains withheld. Ni and Co continue through their existing helpers.
