# K₂Si₂O₅ association choice and potassium predictions

The association reaction in the melt model is

```text
K₂O(l) + 2 SiO₂(l) ⇌ K₂Si₂O₅(l)
log₁₀ K = A + B/T
```

Two published thermochemical routes give materially different values. The
default `imcc-sf04-d066-v5` pack uses the SF04-derived row, re-referenced to
that pack's liquid parent standards:

| Route | `A` | `B` (K) | Source |
| --- | ---: | ---: | --- |
| SF04 vapor-revised, as stored in v5 | 0.411357 | 14568.541 | Schaefer & Fegley (2004), Table 2, [DOI 10.1016/j.icarus.2003.08.023](https://doi.org/10.1016/j.icarus.2003.08.023); that table cites Hastie & Bonnell (1985), *High Temperature Science* 19, 275–306 (no DOI located). The printed SF04 fit is 0.35 + 14685/T; v5 re-references it to its current parent standards. |
| FC87 calorimetric route | −0.73 | 18466 | Fegley & Cameron (1987), Table 3 R32 and Appendix 1, [DOI 10.1016/0012-821X(87)90196-8](https://doi.org/10.1016/0012-821X(87)90196-8), derived from Beyer, Ferrante, Brown & Daut (1980), *Thermodynamic Properties of Potassium Metasilicate and Disilicate*, U.S. Bureau of Mines RI 8410 ([report scan](https://stacks.cdc.gov/view/cdc/237250/cdc_237250_DS1.pdf); no DOI located). |

The SF04 fit is less favorable to association than FC87. In a binary melt at
36 wt% K₂O and 64 wt% SiO₂, replacing only the K₂Si₂O₅ fit with the FC87
values changes the model's predicted activities as follows. The pressure shift
uses the congruent Knudsen-effusion relation for
K₂O(l) ⇌ 2 K(g) + ½ O₂(g),
`Δlog₁₀ P(K) ≈ 0.4 Δlog₁₀ a(K₂O)`; it is a model comparison, not a
measurement or uncertainty band.

| T (K) | log₁₀ a(K₂O), SF04 | log₁₀ a(K₂O), FC87 | FC87 − SF04, log₁₀ a(K₂O) | FC87 − SF04, log₁₀ P(K) |
| ---: | ---: | ---: | ---: | ---: |
| 1700 | −8.395 | −9.646 | −1.251 dex | −0.500 dex |
| 1800 | −7.976 | −9.046 | −1.070 dex | −0.428 dex |
| 2000 | −7.258 | −8.035 | −0.776 dex | −0.311 dex |

Thus, at the same composition the calorimetric route predicts substantially
lower K₂O activity and K(g) pressure. At the disilicate composition the
thermochemical comparison points the other way: the SF04-derived formula
Gibbs energy is about 20–31 kJ mol⁻¹ above the rebuilt Hess-cycle targets
from 1600–2000 K, while substituting the FC87 values brings it to within
about 3.5–4.8 kJ mol⁻¹. The formula basis is
`G = RT[ln a(K₂O) + 2 ln a(SiO₂)]`.

## Why v5 keeps SF04

The source-selection order is standard-state consistency, primary measurement,
domain match, then estimates. Here the consistency checks disagree: the Hess
cycle favors the Beyer/FC87 calorimetric route, while the K-vapor comparisons
favor the weaker SF04 association. Both routes are measurement-based; the
Hess target and FC87 value share the Beyer calorimetry, so that closure is not
an independent K-activity measurement. For K-pressure predictions, the direct
K-vapor measurements are the primary observable. Plante's in-domain points
overlap 1700–1800 K, inside the v5 row's 1700–3000 K exercised range; the
Beyer liquid heat-content measurements end at 1453.8 K, so their application
to those vapor conditions is extrapolative. No lower-tier estimate is used
to break the conflict. The default therefore retains the SF04-derived row
while the evidence remains split; this preserves the in-domain vapor
comparison and does not resolve the Hess mismatch.

Plante (1979), “Vapor pressure measurements of potassium over K₂O–SiO₂
solutions by a Knudsen effusion, mass spectrometric method,” measured K(g)
pressures in the binary melt ([NBS Special Publication 561, vol. 1, DOI
10.6028/NBS.SP.561v1](https://doi.org/10.6028/NBS.SP.561v1)). On the default
pack, the in-domain residual medians `log₁₀(Pmodel/Pmeasured)` are −0.319 dex
for 12–<25 wt% K₂O (12 points) and −0.179 dex for 25–<40 wt% K₂O (9
points). The FC87 substitution lowers those pressures further. A new
independent liquid K₂Si₂O₅ formation-G measurement, reduced on the same K₂O(l)
and SiO₂(l) parent standards, would distinguish the two routes.

## What to expect

Default-pack K pressures remain low against K-vapor measurements, and the
size of the discrepancy varies with composition and source. The separate
Hastie (1981) KEMS case-4 comparison carried by the melt notice is lower by
1.17–1.22 dex for the default pack; it is not the same composition slice as
the in-domain Plante comparison. Treat K predictions as source-sensitive
outputs, keep their temperature and composition with any reported value, and
do not apply a universal correction from either comparison. The default pack
and its numerical behavior are unchanged by this note.
