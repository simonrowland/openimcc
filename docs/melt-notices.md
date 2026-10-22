# Melt notices

Melt notices report evidence limits or measured comparisons. They do not alter
the solved activities. Where a notice includes a measured comparison, it names
or pins the datapack identity for which that evidence was scored; other packs
inside identity-scoped gates receive an unscored notice.

| Code | Trigger | Evidence and meaning |
|---|---|---|
| Hastie KEMS bias notice (no stable code) | Any nonzero Na₂O or K₂O parent | Hastie (1981), KEMS case 4; the measured K pressure residual applies to scored pack identities. Other packs say that the comparison has not been evaluated. |
| `low-temperature-uncertain` | Active Na₂Si₂O₅ association below 1700 K on the scored pack | Piacente & Matoušek (1973), MgO-cell p(Na), is about 0.54 dex low at 1278–1530 K; Richet, Bottinga & Téqui (1984) sodium-silicate calorimetric curvature misses its envelope by 1.5–3× at 1323–1700 K. Other packs use `low-temperature-uncertain-unscored`. |
| `cao-sio2-activity-bias` | Effectively binary CaO–SiO₂, sub-basis x(CaO) 0.42–0.70, 1873–1933 K | Kume (2000) Table 1 and Omori & Sanbongi (1961) support the moderate silica-activity deficit; Stolyarova (1991) Tables Ib/IIb give the stronger near-Ca₂SiO₄ comparison with the published closure caveat. Other packs use `cao-sio2-activity-bias-unscored`. |
| `silicate-vapour-product-low-near-liquidus` | Normalize the candidate crystal's oxide-mole sub-basis to one. Require the sum of absolute differences from its stoichiometric fractions to be ≤0.02 (with 1e-12 comparison tolerance), the sub-basis to contain at least 98% of total parent moles, and every other parent to be below 1% of total parent moles. Temperature is within inclusive ±50 K of the measured melting point: diopside 1668 K, akermanite 1727 K, gehlenite 1823 K, anorthite 1826 K, wollastonite 1817 K. | SI at the nominal melting points, using crystal data extended to those temperatures, is respectively −0.77, −0.49, −0.61, −0.40 and −0.21; these are the corresponding low combined vapour-product residuals in log₁₀ units. Diopside at 1668 K is an incongruent-melting onset and diagnostic, not a congruent coexistence. Crystal functions follow Berman (1988), Tables 2, 3a–3b and 5 (gehlenite disorder); melting anchors follow Robie & Hemingway (1995), and the diopside onset is also reported by Lange et al. (1991). The split between activities and liquid standards is unresolved. Other packs use `silicate-vapour-product-low-near-liquidus-unscored`. |
| `temperature-domain-extension` | An active association row is evaluated below its declared temperature domain | The model uses ΔrCp = 0 continuation; this is calculation authority, not validation, and no uncertainty band is available. |

The silicate composition distance is evaluated only after renormalizing the
candidate crystal's oxide sub-basis. The temperature window is inclusive.
Notices do not establish a liquidus or identify the first stable crystal.
