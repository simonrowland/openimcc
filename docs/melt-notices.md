# Melt notices

Melt notices report evidence limits or measured comparisons. They do not alter
the solved activities. Where a notice includes a measured comparison, it names
or pins the datapack identity for which that evidence was scored; other packs
inside identity-scoped gates receive an unscored notice.

| Code | Trigger | Evidence and meaning |
|---|---|---|
| Hastie KEMS bias notice (no stable code) | Any nonzero Na₂O or K₂O parent | Hastie (1981), KEMS case 4; the measured K pressure residual applies to scored pack identities. Other packs say that the comparison has not been evaluated. |
| `low-temperature-uncertain` | Active Na₂Si₂O₅ association below 1700 K on the scored pack | Piacente & Matoušek (1973), MgO-cell p(Na), is about 0.54 dex low at 1278–1530 K; Richet, Bottinga & Téqui (1984) sodium-silicate calorimetric curvature misses its envelope by 1.5–3× at 1323–1700 K. Other packs use `low-temperature-uncertain-unscored`. |
| `cao-sio2-activity-bias` | Effectively binary CaO–SiO₂, sub-basis x(CaO) 0.42–0.70, 1873–1933 K | Kume (2000) Table 1 and Omori & Sanbongi (1961) support low silica activity. The default refit shifts predicted log10(a(SiO2)) by 0.045–0.410 dex on x(CaO) [0.42, 0.56); on [0.56, 0.70] the shift is non-monotonic, spanning 0.076–0.479 dex and peaking near x(CaO)=0.614. These are v5−v4 changes; prior residual magnitudes are not inherited as new-pack scores. Stolyarova (1991) Tables Ib/IIb give the stronger near-Ca₂SiO₄ comparison with the published closure caveat. Other packs use `cao-sio2-activity-bias-unscored`. |
| `multicomponent-activity-pattern` | Each of SiO₂, CaO, MgO, and Al₂O₃ exceeds 1 mol% of complete parent moles; 1700–3000 K | On default D066 v5, Kume et al. (2000; DOI [10.2355/isijinternational.40.561](https://doi.org/10.2355/isijinternational.40.561)) synthetic CAS/CMS/CMAS slags measured at 1823–1873 K give SiO₂ residuals (log₁₀(predicted/measured), after solid-to-liquid standard conversion) that decrease with NBO/T in CAS (Table 2: slope −0.238 dex/NBO/T, n=47) and CMS (Table 3: −0.362, n=27); this direction is measured on CAS/CMS, not on four-oxide CMAS. The CAS+CMS+CMAS slice has RMSEs of 0.47 dex for SiO₂ (n=72), 0.59 for CaO (n=87), and 0.50 for MgO (n=43). Trigger-matching four-oxide CMAS (Table 4) has no SiO₂ rows; its CaO and MgO RMSEs are 0.81 and 0.57 dex (n=15 each). CaO and MgO residuals become more positive with NBO/T. Al-bearing association, temperature, and source method are confounded; NBO/T is descriptive and no cause is established. Effectively binary Kume CaO–SiO₂ slopes are near zero. A regression of binary CaO–SiO₂ KEMS activities from Stolyarova et al. (1991, 1933 K) gives a CaO slope of about +0.2 to +0.3 dex/NBO/T; their silica table conflicts with the paper's own closure, and this binary result does not establish the multicomponent pattern. Kume's evidence is limited to 1823–1873 K, not the full notice domain. Other identities use `multicomponent-activity-pattern-unscored` under the same gate. |
| `na2o-activity-bias` | Positive Na₂O and either molar Na₂O/SiO₂ ≥ 1 or x(CaO+MgO) ≥ 0.30, with the latter measured on the complete parent-oxide mole basis | Held-out Mathieu (2009; PhD thesis, INPL Nancy, [docnum.univ-lorraine.fr](https://docnum.univ-lorraine.fr/public/INPL/2009_MATHIEU_R.pdf)) CMAS activity residual tiers at 1673 K: x(CaO+MgO) 0.30–<0.40, n=12, median +0.77 dex (range +0.37 to +1.20); x(CaO+MgO) ≥0.40 with Na₂O/SiO₂ <1, n=17, median +1.53 dex (range +0.81 to +2.67); Na₂O/SiO₂ ≥1, n=6, median +4.29 dex (range +4.03 to +4.46). The bins are disjoint; the Na/Si tier takes priority where the conditions overlap. The 0.30 threshold is the lowest observed basicity band with a consistently positive residual; below it the three Mathieu rows are mixed (−0.26 to +1.37 dex). Pak et al. (1989; DOI [10.2355/isijinternational.29.318](https://doi.org/10.2355/isijinternational.29.318)) report +2.76 and +1.74 dex at Na₂O/SiO₂ ratios 1.5 and 1, and +0.49 and +0.67 dex at Ca-bearing points with x(CaO) 0.32 and 0.42, while higher-silica Ca-free binary points are slightly negative. The Mathieu measurement method is described by Mathieu et al. (2008; DOI [10.1016/j.jnoncrysol.2008.07.004](https://doi.org/10.1016/j.jnoncrysol.2008.07.004)). Evidence is at 1673 K only; temperature dependence is untested. On the default D066 v5 pack, the predicted a(Na₂O) and Na(g) pressures are high in these regions, and the cause remains unresolved. These numerical claims are pinned to its model ID and complete binding digest. Other identities, including excess-Gibbs candidates, use `na2o-activity-bias-unscored` under the same composition gate, stating that the evidence was measured on the default pack without inheriting its numerical claims. |
| `silicate-vapour-product-low-near-liquidus` | Normalize the candidate crystal's oxide-mole sub-basis to one. Require the sum of absolute differences from its stoichiometric fractions to be ≤0.02 (with 1e-12 comparison tolerance), the sub-basis to contain at least 98% of total parent moles, and every other parent to be below 1% of total parent moles. Temperature is within inclusive ±50 K of the measured melting point: diopside 1668 K, akermanite 1727 K, gehlenite 1823 K, anorthite 1826 K, wollastonite 1817 K. | SI at the nominal melting points, using crystal data extended to those temperatures, is respectively −0.53, +0.11, −0.51, +0.044 and +0.0028. Negative values are low combined vapour-product residuals and positive values are high; the split remains unresolved. Diopside at 1668 K is an incongruent-melting onset and diagnostic, not a congruent coexistence. Crystal functions follow Berman (1988), Tables 2, 3a–3b and 5 (gehlenite disorder); melting anchors follow Robie & Hemingway (1995), and the diopside onset is also reported by Lange et al. (1991). The split between activities and liquid standards is unresolved. Other packs use `silicate-vapour-product-low-near-liquidus-unscored`. |
| `temperature-domain-extension` | An active association row is evaluated below its declared temperature domain | The model uses ΔrCp = 0 continuation; this is calculation authority, not validation, and no uncertainty band is available. |

Liquid gas-parent results expose `liquid_reference_status`, keyed by gas
channel, only for parent oxides present in the melt; reference flags follow the
same rule. Each status has a typed authority, inclusive `certified_T_K` endpoints, band and
structured `evidence_ref`. `liquid_reference_records()` returns the immutable
records and `liquid_reference_authority(species, T_K, composition=None)` resolves
one status. Inside K₂O's inclusive 1259–1765 K window the reference stays
certified; without a composition the returned status leaves the 25–40 wt% K₂O
evidence range unevaluated. Outside a
record's temperature window, status is `extrapolated`. A supplied composition
outside K₂O's 25–40 wt% evidence range keeps the certified temperature-window
authority and adds a notice: Plante (1979) residuals were about −0.13 dex at
25–40 wt% and about −0.38 dex at 12–20 wt%. The composition range describes
where the mixture model was checked; it does not exclude out-of-range results.
`evaluate_gas` accepts an optional `composition_wt_pct` mapping keyed by oxide
to resolve composition-scoped records; calls without it leave the scope
unresolved and retain the flag.
SiO₂(l)'s certified range narrowed to the measured 1500–1800 K interval.
From 1200–1500 K and 1800–3000 K it is corroborated by calculation, with
respective ±0.05 and ±0.06 dex log₁₀-pressure bands and a notice; those process
temperatures still score. From 900–1200 K it remains extrapolated with a
±0.18 dex stress band, not a confidence interval. Below 900 K and above 3000 K
it remains extrapolated without a band. The pure-silica measurement is the
drop-calorimetry relative enthalpy of amorphous silica above its glass
transition reported by Richet, Bottinga, Deniélou, Petitet & Téqui (1982; DOI
10.1016/0016-7037(82)90383-0). Richet, Bottinga & Téqui (1984) measured a
81.37 J mol⁻¹ K⁻¹ partial-molar SiO₂ heat capacity in sodium-silicate liquids;
that value is not pure-liquid SiO₂ heat capacity.

CaO(l) is certified from 1323–2473 K; its record retains the evidence that two
independent calculations corroborate the fusion enthalpy, though no measured
fusion datum exists. It emits no notice inside that window and carries a
±0.07 dex pO₂ band there. Al₂O₃(l) has an extrapolated 1323–1801 K window
with a ±0.17 dex log₁₀-pressure model-envelope band; its low-temperature source
curves are extrapolated and do not validate that envelope. Above 1801 K and
below 2327 K, where the liquid is supercooled, authority is corroborated
calculation with a ±0.05 dex band: an independent CALPHAD liquid and continued measured
heat-capacity series stay within 2.6 kJ mol⁻¹, but nothing is measured
calorimetrically there. A window's band applies only inside that window;
extrapolated statuses outside every window carry no band.
Al₂O₃(l) is certified from its 2327 K
melting point through 2873 K, the end of Crochetet et al.'s measured liquid
heat-capacity range, with a ±0.036 dex log₁₀-pressure band. From 2873–3000 K,
the JANAF liquid fit and Crochetet heat-capacity continuation agree within
2.13 kJ mol⁻¹; this is calculated corroboration with the same band, not
measured coverage. Na₂O(l) has no certified interval because pure-liquid heat
capacity has not been measured. Its measured 1405.2 K melting point and
47.7 ± 2.5 kJ mol⁻¹ fusion enthalpy anchor the reference; calculated
continuations corroborate 1405.2–2473 K, so process rows in that interval
still score with a notice and ±0.21 dex stress band. The supercooled
1300–1405.2 K range remains extrapolated with the same stress band. These are
calculated continuations, not pure-liquid Cp measurements.
MgO(l) is extrapolated at every temperature. Its ±0.76 dex pO₂ band applies
only on 1323–2473 K and is JANAF Mg-009's printed fusion-enthalpy uncertainty,
±16.7 kJ/mol. With Tm = 3105 K, δG = δΔH(1−T/Tm) = 9.58 kJ/mol at 1323 K,
and Δlog₁₀ pO₂ = 2δG/(RT ln 10) for MgO(l) = Mg(g) + ½O₂. The same
uncertainty is 0.41 dex at 1800 K and 0.14 dex at 2473 K; the stored value is
the cold edge of the window. It is not a confidence interval. Assumed liquid
Cp and the unresolved melting-point dispute are outside it. DFT,
molecular-dynamics, binary-liquidus and CALPHAD fusion values are listed
conflicts and do not set the band. The extrapolation flag therefore shows the
band only within this window.

TiO₂(l) is extrapolated at every temperature; the JANAF O-044 and O-044-SC-CP
continuation flags both remain. Its extrapolated 2185–2473 K window carries
±0.11 dex log₁₀ pressure, the outward-rounded cross-assessment envelope from
NASA/Gurvich versus JANAF O-044 and a Barin-derived line. The NASA/Gurvich
residual is +4.28 to +4.44 kJ mol⁻¹ (+0.094 to +0.102 dex pO₂); the
Barin-derived line differs by +2.13 to −0.98 kJ mol⁻¹. For TiO₂,
Δlog₁₀ pO₂ = ΔG/(RT ln 10). This is not a measurement or a confidence
interval. JANAF O-044 estimates liquid Cp and fusion entropy, and the NASA
offset is one-sided. There is no band outside this window. These records
describe reference authority only and do not change calculated gas pressures.
Condensate G(T) interval flags remain in place even when the liquid reference
itself is certified.

For SiO₂(l), Connolly's (2016; DOI 10.1002/2016JE005059) metastable-liquid
function corroborates the 1200–1500 K interval below the glass transition.
Above 1800 K, Connolly's branches and Gurvich (1991), as transcribed in NASA
Glenn CEA (NASA/TP-2002-211556), corroborate the reference by calculation;
no pure-liquid measurement covers that range. K₂O(l) uses Plante (1979) measured K
pressures. FeO(l) uses Darken & Gurry (1946), as reported by Masson (1965), and
Hashimoto (1983) residue run 17c3-2 outside the window. Al₂O₃(l) follows
JANAF Al-100/Al-101 from the melting point, with drop-calorimetry liquid heat
capacity from Crochetet et al. (2026; DOI 10.1016/j.oceram.2026.101029) over
2327–2873 K; from 2873–3000 K the JANAF fit and continued Crochetet Cp agree
within 2.13 kJ mol⁻¹ by calculation. Below the melting point, Hallstedt's (1992; DOI
10.1111/j.1151-2916.1992.tb04216.x) CALPHAD liquid is an independent
calculation within 1.5 kJ mol⁻¹ of the reference. The liquid heat capacity of
Paradis, Ishikawa, Saita & Yoda (2004; DOI 10.1143/JJAP.43.1496) is an estimate
over 2120–2450 K that assumes a constant emissivity. Continued downward, it and
the Crochetet series stay within 2.6 kJ mol⁻¹ down to 1801 K. TiO₂(l) has no measured pure-liquid
data: JANAF O-044 assumes its fusion entropy (7.5 cal/mol/K, by analogy with
V₂O₄) and estimates its liquid heat capacity. Na₂O(l) has a measured fusion
anchor but no measured pure-liquid Cp interval; its JANAF function is
corroborated by the sodium-silicate partial-molar Cp of Richet, Bottinga & Téqui
(1984; DOI 10.1111/j.1151-2916.1984.tb19155.x) and the Gurvich (1982) liquid
function transcribed in NASA Glenn CEA (NASA/TP-2002-211556). Smith et al.
(2017; DOI 10.1016/j.jct.2017.04.003) reports the 47.7 ± 2.5 kJ mol⁻¹
fusion enthalpy. These liquid Cp continuations are calculations, not
pure-liquid measurements. CaO(l)'s two independent molecular-dynamics
calculations (2020, 2024) reproduce fusion enthalpy within 2.6 kJ/mol; neither
is measured fusion evidence. MgO(l)'s assumed fusion entropy and calculated
fusion values scatter by about 10–19 kJ/mol.

The silicate composition distance is evaluated only after renormalizing the
candidate crystal's oxide sub-basis. The temperature window is inclusive.
Notices do not establish a liquidus or identify the first stable crystal.
