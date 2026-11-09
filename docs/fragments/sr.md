# Strontium melt fragment

`src/openimcc/data/fragments/sr/rows.json` supplies the Sr rows packaged in the default redox v5 pack. It adds Sr(l) and SrO(l) thermochemical standards and redox rows for SrO(l), SrSiO3(l), Sr2SiO4(l), and SrAl2O4(l). The existing gas model already has Sr-bearing vapor species; callers can supply the resulting SrO parent activity when evaluating those channels.

The thermochemistry is deliberately marked provisional. The rows are included in the immutable default pack; the source fragment remains the assembly input for a later pack version.

## Source values and construction

| Species | Source anchors | Construction and checks |
| --- | --- | --- |
| Sr(l) | NIST-JANAF Table Sr-004: ΔfH°298 = 0.911 kJ/mol, S°298 = 50.901 J/(mol K), Cp = 39.463 J/(mol K); at 1800 K, Hinc = 59.267 kJ/mol and S = 121.853 J/(mol K). | Direct liquid standard. H(T) = Hf298 + Cp(T−298.15)/1000; S(T) = S298 + Cp ln(T/298.15); G = H−TS. It reproduces the printed 1800 K anchor. The liquid branch above its 1685.492 K marker is a metastable reference continuation. [JANAF Sr-004](https://janaf.nist.gov/tables/Sr-004.html), DOI [10.18434/T42S31](https://doi.org/10.18434/T42S31). |
| SrO(l) | NIST-JANAF Tables O-013/O-014: crystal/liquid marker 2938 K; at 2900 K, crystal Hinc/S = 155.921 kJ/mol / 183.013 J/(mol K), liquid Hinc/S = 156.579 kJ/mol / 208.677 J/(mol K); liquid Cp = 66.944 J/(mol K). | The existing O-014 condensate row is copied unchanged. JANAF crystal/liquid functions give ΔSfus = 25.63386 J/(mol K), then ΔHfus = TmΔSfus = 75.312 kJ/mol. Units are J/mol for TΔS and kJ/mol after division by 1000. The independent Handbook value is 75.00 kJ/mol (0.31 kJ/mol lower); equality of crystal and liquid G at 2938 K is the sanity check. [JANAF O-013](https://janaf.nist.gov/tables/O-013.html) and [O-014](https://janaf.nist.gov/tables/O-014.html), DOI [10.18434/T42S31](https://doi.org/10.18434/T42S31); Lide, *Handbook of Inorganic Compounds*, 2nd ed., entry 3076. |
| SrSiO3(l) | Measured ΔfH°298 = −1635.3 ± 1.8 kJ/mol; crystal Hinc/S/Cp at 1000 K = 79.976 kJ/mol / 229.082 J/(mol K) / 129.226 J/(mol K); congruent melt 1580 ± 15 °C = 1853.15 K. | Above the measured 1000 K Cp range, crystal Cp is held at its 1000 K value through Tm. Estimated ΔSfus = 5 × 8.297530569 = 41.488 J/(mol K), using the 17-pair JANAF class-C mean per formula atom. Estimated liquid Cp = Cp(SrO(l)) + Cp(SiO2(l)) = 66.944 + 85.772 = 152.716 J/(mol K). At Tm, Hinc(s) = 190.225 kJ/mol, S(s) = 308.800 J/(mol K), ΔHfus = 76.883 kJ/mol, and G(l) = G(s) = −2017.327 kJ/mol. [Formation enthalpy](https://doi.org/10.1016/S0021-9614(05)80042-4), pp. 139–143; [crystal properties](https://doi.org/10.1016/0022-3697(92)90193-H), Tables 4–5, pp. 805–806; [melting point](https://doi.org/10.1111/j.1151-2916.1972.tb13448.x), pp. 585–588; JANAF Tables O-014/O-038, DOI 10.18434/T42S31. |
| Sr2SiO4(l) | Measured ΔfH°298 = −2305.7 ± 2.2 kJ/mol; crystal Hinc/S/Cp at 1000 K = 118.607 kJ/mol / 353.753 J/(mol K) / 194.694 J/(mol K); congruent melt 2325 ± 15 °C = 2598.15 K. | Use the same high-temperature crystal-Cp continuation and class-C fusion-entropy estimate: ΔSfus = 7 × 8.297530569 = 58.083 J/(mol K). Estimated liquid Cp = 2Cp(SrO(l)) + Cp(SiO2(l)) = 219.660 J/(mol K). At Tm, Hinc(s) = 429.757 kJ/mol, S(s) = 539.647 J/(mol K), ΔHfus = 150.908 kJ/mol, and G(l) = G(s) = −3278.026 kJ/mol. Formation enthalpy, crystal properties, melting point, and JANAF table citations are the same sources as for SrSiO3(l). |
| SrAl2O4(l) | Brisi and Abbattista's formation-from-oxides target is −79.2 ± 2.9 kJ/mol, transcribed in Ye et al. (2007), Tables 1 and 3; JANAF gives SrO(cr) ΔfH°298 = −592.036 and Al2O3(cr) = −1675.692 kJ/mol. Ye et al. select Tm = 2233 K; Table 4 also reports competing 2063, 2238, and 2288 K markers. | Hess cycle: ΔfH°298 = −592.036 − 1675.692 − 79.2 = −2346.928 kJ/mol. Use Ye's high-temperature crystal relation G(s) = G(SrO,cr) + G(Al2O3,cr) − 79.2 kJ/mol through 2233 K. Estimate ΔSfus = 7 × 8.297530569 = 58.083 J/(mol K) and liquid Cp = Cp(SrO(l)) + Cp(Al2O3(l)) = 66.944 + 192.464 = 259.408 J/(mol K). At Tm, G(l) = G(s) = −3002.667 kJ/mol and ΔHfus = 129.699 kJ/mol. The selected melting marker is assessment-based and provisional. [Ye et al.](https://doi.org/10.1007/s11669-007-9086-x), Tables 1 and 3–4, pp. 364–365; JANAF Tables O-013/Al-096/O-014/Al-100, DOI 10.18434/T42S31. |

For the three associates, the liquid state is constructed from the crystal at Tm using H_l(T) = H_s(Tm) + TmΔSfus + Cp,l(T−Tm), S_l(T) = S_s(Tm) + ΔSfus + Cp,l ln(T/Tm), and G_l = H_l−TS_l. Enthalpies are converted from J/mol to kJ/mol where shown. The row coefficients encode these G(T) functions over 1200–3000 K; their maximum encoding residuals are below 0.31 J/mol. The source thermochemistry and the fusion/liquid-Cp estimates are kept separate in the derivation records.

The fusion-entropy estimate is the arithmetic mean of 17 measured class-C JANAF crystal/liquid pairs: 8.297530569 J/(mol K) per formula atom. Its leave-one-out maximum residual is 46.401 J/(mol K), with 17.845 J/(mol K) RMS. This broad scatter is why all three associate rows are flagged provisional. The activity coefficients of the SrO–Al2O3, SrO–La2O3, and SrO–BaO–Al2O3–SiO2 experiments were not used to fit any row.

## Battery coverage and smoke

The coverage inventory contains 212 in-window rows whose compositions include SrO and whose current refusal is missing melt-parent coverage:

| Source | Species counts | Total |
| --- | --- | ---: |
| Stolyarova, Vorozhtcov, and Lopatin SrO–Al2O3 supplement (2023) | Al 40; Al2O 37; AlO 37; O 43; Sr 43 | 200 |
| Shilov SrO–La2O3 KEMS source (2026) | Sr 12 | 12 |

The Sr-only fragment supplies an SrO activity path for the 200 SrO–Al2O3 rows, with provisional compound thermochemistry. They remain unvalidated: the source rows do not map an oxygen fugacity, and no source activity data were fitted. The 12 SrO–La2O3 rows still require a La2O3 melt model, so this fragment alone does not cover them. The broader ladder estimate of 6751 potential non-scored rows is a projection, not a count of rows in this coverage inventory.

As a report-only pressure check, SrO(g) rows from Lopatin, Shugurov, and Tyurnina (2022), supplemental Table S2, were compared against the independent model. For each printed SrO(s) reservoir, activity was computed as exp((Gcrystal−Gliquid)/(RT)) from JANAF O-013 and the fragment's SrO(l), then evaluated through the existing gas channel. Printed Pa were converted to bar. No battery value was used in fitting.

| T (K) | Printed (bar) | Predicted (bar) | Predicted / printed |
| ---: | ---: | ---: | ---: |
| 1951 | 2.60 × 10⁻⁸ | 5.84 × 10⁻⁸ | 2.25 |
| 2059 | 1.02 × 10⁻⁷ | 3.44 × 10⁻⁷ | 3.37 |
| 2093 | 1.14 × 10⁻⁷ | 5.77 × 10⁻⁷ | 5.06 |
| 2190 | 1.121 × 10⁻⁶ | 2.32 × 10⁻⁶ | 2.07 |

The mismatch is material and supports the provisional notice; these points are a smoke check, not a validation result. [Lopatin et al. (2022), DOI 10.1002/rcm.9298, Table S2](https://doi.org/10.1002/rcm.9298).

## Scope and limits

- The default pack includes four Sr redox rows and the Sr/SrO standards. Gas and condensate tables are unchanged.
- The three associates use independent formation/fusion thermochemistry, not fitted activity coefficients. High-temperature crystal Cp continuation, class-C fusion entropy, and additive liquid Cp are estimates.
- The SrAl2O4 melting marker has competing reported values; the selected 2233 K assessment is flagged.
- The current fragment has no Sr-bearing phosphate associates, no La2O3 component, and no validated activity model for concentrated SrO mixtures.
- Source-domain activity validation remains open. Battery KEMS results are report-only, and no general-use claim is made.
