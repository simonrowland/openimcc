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
channel, with a typed authority, inclusive `certified_T_K` endpoints, band and
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
CaO(l) is certified from 1323–2473 K; its record retains the evidence that two
independent calculations corroborate the fusion enthalpy, though no measured
fusion datum exists. It emits no notice inside that window and carries its
±0.07 dex pO₂ band on extrapolated flags outside it. Al₂O₃(l)'s measured fit
window begins at 2500 K, and Na₂O(l)'s measured Cp window ends at 1800 K.
TiO₂(l) is extrapolated at every temperature and has no certified window; the
JANAF O-044 flag and the O-044-SC-CP continuation flag both remain. MgO(l) remains extrapolated and carries its
±0.3 dex pO₂ band. These records describe reference authority only and do not
change calculated gas pressures. Condensate G(T) interval flags remain in
place even when the liquid reference itself is certified.

The SiO₂(l) window uses Richet, Bottinga & Téqui (1984), measured liquid heat
capacity and the measured fusion anchor. K₂O(l) uses Plante (1979) measured K
pressures. FeO(l) uses Darken & Gurry (1946), as reported by Masson (1965), and
Hashimoto (1983) residue run 17c3-2 outside the window. Al₂O₃(l) follows the
JANAF liquid fit to measured liquid enthalpy. TiO₂(l) has no measured pure-liquid
data: JANAF O-044 assumes its fusion entropy (7.5 cal/mol/K, by analogy with
V₂O₄) and estimates its liquid heat capacity. Na₂O(l) follows its source-supported
liquid-from-solid construction. CaO(l)'s two independent molecular-dynamics
calculations (2020, 2024) reproduce fusion enthalpy within 2.6 kJ/mol; neither
is measured fusion evidence. MgO(l)'s assumed fusion entropy and calculated
fusion values scatter by about 10–19 kJ/mol.

The silicate composition distance is evaluated only after renormalizing the
candidate crystal's oxide sub-basis. The temperature window is inclusive.
Notices do not establish a liquidus or identify the first stable crystal.
