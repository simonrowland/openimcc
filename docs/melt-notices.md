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
calorimetrically there. A window's band applies only inside that window, so
extrapolated statuses outside every window carry no band unless the record is
itself extrapolated, as MgO(l) is. Al₂O₃(l) is certified from
its 2327 K melting point to 3000 K with a ±0.036 dex log₁₀-pressure band, the
largest departure of the measured liquid heat capacity from the reference.
Na₂O(l)'s measured Cp window ends at 1800 K.
TiO₂(l) is extrapolated at every temperature and has no certified window; the
JANAF O-044 flag and the O-044-SC-CP continuation flag both remain. MgO(l) remains extrapolated and carries its
±0.3 dex pO₂ band. These records describe reference authority only and do not
change calculated gas pressures. Condensate G(T) interval flags remain in
place even when the liquid reference itself is certified.

For SiO₂(l), Connolly's (2016; DOI 10.1002/2016JE005059) metastable-liquid
function corroborates the 1200–1500 K interval below the glass transition.
Above 1800 K, Connolly's branches and Gurvich (1991), as transcribed in NASA
Glenn CEA (NASA/TP-2002-211556), corroborate the reference by calculation;
no pure-liquid measurement covers that range. K₂O(l) uses Plante (1979) measured K
pressures. FeO(l) uses Darken & Gurry (1946), as reported by Masson (1965), and
Hashimoto (1983) residue run 17c3-2 outside the window. Al₂O₃(l) follows
JANAF Al-100/Al-101 from the melting point, with drop-calorimetry liquid heat
capacity from Crochetet et al. (2026; DOI 10.1016/j.oceram.2026.101029) over
2327–2873 K. Below the melting point, Hallstedt's (1992; DOI
10.1111/j.1151-2916.1992.tb04216.x) CALPHAD liquid is an independent
calculation within 1.5 kJ mol⁻¹ of the reference. The liquid heat capacity of
Paradis, Ishikawa, Saita & Yoda (2004; DOI 10.1143/JJAP.43.1496) is an estimate
over 2120–2450 K that assumes a constant emissivity. Continued downward, it and
the Crochetet series stay within 2.6 kJ mol⁻¹ down to 1801 K. TiO₂(l) has no measured pure-liquid
data: JANAF O-044 assumes its fusion entropy (7.5 cal/mol/K, by analogy with
V₂O₄) and estimates its liquid heat capacity. Na₂O(l) follows its source-supported
liquid-from-solid construction. CaO(l)'s two independent molecular-dynamics
calculations (2020, 2024) reproduce fusion enthalpy within 2.6 kJ/mol; neither
is measured fusion evidence. MgO(l)'s assumed fusion entropy and calculated
fusion values scatter by about 10–19 kJ/mol.

The silicate composition distance is evaluated only after renormalizing the
candidate crystal's oxide sub-basis. The temperature window is inclusive.
Notices do not establish a liquidus or identify the first stable crystal.
