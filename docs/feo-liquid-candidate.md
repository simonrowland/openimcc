# FeO(l) liquid re-anchor candidate

**Decision: keep opt-in.** The D&G-derived candidate improves the held-out Kress–Carmichael and Matzen scores and the Sossi fitted relation, but worsens Aithala. The source data used to anchor FeO carry no formal pointwise activity uncertainty, the Distin stoichiometric correction is method-dependent, and the validation is mixed. The candidate remains separate from every default pack.

## Source transcriptions and uncertainty

[Darken & Gurry Table I transcription](feo-liquid-dg-table-i.csv) records all 76 liquid-oxide measurements, the printed starting material and equilibration time, the imposed gas condition, the reported ratio r = (Fe²⁺ + Fe³⁺)/Fe²⁺, and the derived O/Fe value. Gas conditions are recorded as the source states them; for mixed CO₂/H₂ runs these are initial mixture ratios, not measured equilibrium gas ratios. Cells that the text extraction misread (times printed as fractions, and several oxygen-group temperatures and times) were re-read from the journal page image of p. 800 and are flagged in the CSV; none of them feeds the Gibbs-energy anchor.

[Darken & Gurry Tables IIIA, IIIB, and VI transcription](feo-liquid-dg-anchor-data.csv) preserves the iron-saturated compositions, CO₂/CO ratios, and printed FeO activity coordinates used for the re-anchor.

Darken & Gurry report no pointwise uncertainty for r or O/Fe. Their methods describe the steam delivery calibration as agreeing within 1%; this is not a stated per-run uncertainty on the equilibrium gas ratio or oxide analysis. They say the low-oxygen results are more precise and the most oxidized results are less precise because gas impurity and quench oxidation become more important.

[Distin et al. Table I transcription](feo-liquid-distin-activities.csv) contains all 24 FeO–SiO₂ activity measurements, final and initial SiO₂, FeO and Fe₂O₃ analyses, metal oxygen, equilibration time, reported activity, and the derived stoichiometric-standard activity. Distin et al. give no pointwise uncertainty for the activities or metal oxygen values. They report weighing tolerances of ±0.015 g for slag and ±0.09 g for iron; those tolerances are not a complete activity uncertainty.

The source papers are [Darken & Gurry (1946)](https://doi.org/10.1021/ja01209a030), [Masson (1965)](https://doi.org/10.1098/rspa.1965.0176), and [Distin, Whiteway & Masson (1971)](https://doi.org/10.1179/cmq.1971.10.2.73).

## Standard-state reduction

Darken & Gurry report the liquid oxide composition as

$$
r = (Fe^{2+} + Fe^{3+})/Fe^{2+}.
$$

With Fe²⁺ and Fe³⁺ as the only iron valences,

$$
O/Fe = 1/r + 1.5(r-1)/r = 1.5 - 0.5/r.
$$

The Temkin iron-sublattice count used by Masson is

$$
N_{Fe^{2+}} = 3 - 2(O/Fe).
$$

At the Darken & Gurry Table VI reference composition O/Fe = 1.07, this gives N_Fe²⁺⁰ = 0.86. Masson’s stoichiometric-standard transformation is

$$
a_{FeO}(stoich) = N_{Fe^{2+}}^0 \cdot a_{FeO}(nonstoich).
$$

Table VI prints −log₁₀ a_FeO; the sign is reversed to recover log₁₀ a_FeO(nonstoich), then log₁₀(0.86) is added. Distin’s pure-oxide, iron-saturated N_Fe²⁺⁰ is not tabulated. I estimate it for each temperature by calculating N_Fe²⁺ = n_FeO/(n_FeO + 2n_Fe₂O₃) from the first three low-silica analyses and linearly extrapolating against final SiO₂ wt% to zero. The inferred factors are 0.978696 at 1785 °C, 0.986138 at 1880 °C, and 0.996788 at 1960 °C. The first-three-analysis N values span 0.978478–0.981258, 0.986940–0.993659, and 0.993568–0.997943 respectively. Those ranges describe method sensitivity in the selected points; they are not confidence intervals.

Distin’s activities are multiplied by the inferred temperature-specific factor in the CSV. This reproduces Masson’s transformation but makes the extrapolation explicit. Gibbs–Duhem is not needed for this conversion: D&G’s own Table VI already integrates the gas-equilibrium data to activities relative to O/Fe = 1.07.

## D&G Fe-saturation anchor and coherent row transformation

The candidate anchors the standard FeO Gibbs energy to D&G’s iron-saturated boundary:

| T (K) | Table IIIA/B pCO₂/pCO | interpolated r | O/Fe | log₁₀ a(nonstoich) | log₁₀ a(stoich) | target ΔG° (kJ/mol) | default ΔG° | correction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1649.15 | 0.268 | 1.085500 | 1.039383 | +0.021911 | −0.043590 | −156.5506 | −166.2052 | +9.6546 |
| 1709.15 | 0.233 | 1.060789 | 1.028653 | +0.027633 | −0.037869 | −154.2487 | −163.3895 | +9.1408 |
| 1766.15 | 0.199 | 1.055941 | 1.026489 | +0.026387 | −0.039115 | −152.3560 | −160.7255 | +8.3696 |

For CO + ½O₂ = CO₂, log₁₀ fO₂ = 2(log₁₀(pCO₂/pCO) − log₁₀K), using NIST-JANAF CO, CO₂ and O₂ functions. For Fe + ½O₂ = FeO(l),

$$
\Delta G^\circ = RT \ln(10) \left(\tfrac{1}{2}\log_{10} fO_2 - \log_{10} a_{FeO}(stoich)\right).
$$

The saturated compositions are interpolated from D&G Table IIIA, with the 1376 °C point extrapolated from the two nearest smooth values; the anomalous 1392 °C pair remains in the raw source record but is excluded from that interpolation. Table VI is interpolated in composition and in temperature, with the 1376 °C activity a small extrapolation below its first 1400 °C block. The resulting three-point correction is fit as

$$
\delta G_{FeO}(T) = 27{,}778.965828 - 10.961554232\,T\quad J\ mol^{-1}.
$$

The candidate-minus-target residuals at the three anchors are +0.047, −0.097, and +0.050 kJ/mol. This is a fit residual, not an uncertainty estimate. JANAF gas data are from the [NIST CO](https://janaf.nist.gov/tables/C-093.html), [CO₂](https://janaf.nist.gov/tables/C-095.html), and [O₂](https://janaf.nist.gov/tables/O-029.html) tables.

The correction is anchored from 1649 to 1766 K and is linearly extrapolated above 1766 K across most of the candidate melt-pack domain, which extends to 3000 K.

The same δG_FeO is propagated through every FeO-dependent standard state. In the redox rows, FeO(l) shifts by +δG; the FeO1.5 oxidation reaction shifts by −δG because FeO is its reactant while the FeO1.5 product standard remains unchanged. The gas FeO condensate row receives the same parent-standard shift. For each melt association with ν_FeO FeO parents, keeping the complex standard chemical potential fixed requires

$$
\Delta\log_{10}K = \nu_{FeO}\,\delta G_{FeO}/(RT\ln 10).
$$

That produces ΔA = −0.572561660·ν_FeO and ΔB = 1450.995948·ν_FeO K for log₁₀K = A + B/T. All three Fe-bearing melt associations are transformed:

| Complex | νFeO | Default A | Candidate A | Default B (K) | Candidate B (K) |
|---|---:|---:|---:|---:|---:|
| FeTiO₃ | 1 | −0.510000 | −1.082562 | 3569.000 | 5019.996 |
| Fe₂SiO₄ | 2 | −0.630000 | −1.775123 | 3103.000 | 6004.992 |
| FeAl₂O₄ | 1 | −1.760000 | −2.332562 | 5692.000 | 7142.996 |

The candidate bundle is opt-in in the existing pack formats:

- src/openimcc/data/packs/openimcc-redox-feo-liquid-candidate.json
- src/openimcc/data/packs/imcc-sf04-d066-feo-liquid-candidate.json
- src/openimcc/data/gas/condensate-feo-liquid-candidate.csv

Select the redox and melt JSON paths together in evaluate_redox(..., redox_pack_path=..., melt_pack_path=...); pass the candidate condensate file to the gas loader for gas calculations. The packaged defaults were not edited. The candidate redox canonical digest is 00bb1056567715f2a81c53339b0c5d3296cb0f250ac79b7db62ec2b8cac0cb7b; the melt binding digest is 8c96383e5c1464dbec4e748ff7b528151028f5b2b4d731c5f9861d677c8cff39.

## Held-out comparison

[Residuals for each scored row](feo-liquid-heldout-residuals.csv) use prediction minus observation for ferric Fe / total dissolved Fe. MAE and RMSE are absolute fraction units, unweighted because the source uncertainty models are incomplete or not comparable. Candidate construction used D&G and the standard-state conversion above, not these validation data. The default Kress–Carmichael calibration has seen related calibration data, so that set is a useful held-out check for this candidate but is not independent of the default.

The held-out sources are [Kress & Carmichael (1991)](https://doi.org/10.1007/BF00307328), [Matzen et al. (2022)](https://doi.org/10.2138/am-2021-7682), [Aithala et al. (2026)](https://doi.org/10.7185/geochemlet.2617), and [Sossi et al. (2020)](https://doi.org/10.1126/sciadv.abd1387).

| Held-out set | n | Default MAE | Default RMSE | Candidate MAE | Candidate RMSE | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| Kress–Carmichael | 32 | 0.07446 | 0.08478 | 0.05229 | 0.06183 | Improved; 32 additional runs below 1700 K were outside the melt-pack domain. |
| Matzen, conventional Mössbauer assignment | 11 | 0.09230 | 0.09899 | 0.07963 | 0.08770 | Improved; each reported result has ±0.03 uncertainty. |
| Matzen, Berry et al. (2018) assignment | 11 | 0.02387 | 0.02819 | 0.01278 | 0.01570 | Improved; each reported result has ±0.03 uncertainty. |
| Aithala | 8 | 0.04237 | 0.05201 | 0.07258 | 0.07868 | Worse; converted source standard deviations are about 0.015–0.031. |
| Sossi Eq. 4 regression grid | 4 | 0.01616 | 0.02150 | 0.00272 | 0.00372 | Improved against the reported fitted line, not raw run residuals. |

Kress entries use the source’s Table 1 iron analyses, converting Fe₂O₃/(FeO+Fe₂O₃) to ferric-iron fraction; unsupported minor oxides are omitted. The 32 retained points are at or above the 1700 K melt model floor. Matzen uses the reported Syn-MB1 as-weighed composition, with iron entered as FeO; unsupported trace oxides and measured alkali uptake are omitted. The conventional and Berry et al. (2018) Mössbauer assignments are scored separately; the run below 1700 K and two records without ferric ratios are excluded. Aithala uses the WDS compositions and Fe³⁺/Fe²⁺ data in its main and supplementary tables, omits unsupported P₂O₅, and excludes the below-liquidus / below-domain runs.

For Sossi et al. (2020), the locally available paper reports log₁₀(Fe³⁺/Fe²⁺) = (0.252 ± 0.015)ΔIW − 1.530 ± 0.053 at 2173 K, but the supplementary per-run table is not present in the corpus. The four residual rows evaluate that published regression at ΔIW −1.5, 0, +3, and +6.4; they are derived curve values, not four measured runs, and have no pointwise σ. The paper’s separate mantle-derived anchor Fe³⁺/ΣFe = 0.037 ± 0.005 at ΔIW +0.5 gives default residual −0.00510 and candidate residual −0.00011. The fO₂ conversion uses log₁₀ fO₂(IW) = −6.47 at 2173 K as reported in the [comparison literature](https://doi.org/10.3847/psj/ac67a9). Sossi’s two most reducing levitation glasses also lost 7% and 9% of their initial iron, a further limit on comparing those points to this closed-composition melt model.

The held-out scores use the redox and melt candidates together; the gas file is checked as part of the bundle but does not enter ferric-fraction scoring. All predictions were computed through the normal pack loaders and evaluate_redox. Tests cover omitted/None equivalence, candidate selection and digest reporting, missing and mismatched packs, plus the pinned default redox outputs.
