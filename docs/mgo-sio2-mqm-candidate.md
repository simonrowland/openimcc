# Experimental MgO–SiO₂ MQM candidate

`openimcc.evaluate_mgo_sio2_mqm(composition_moles, T_K)` exposes an opt-in
binary liquid calculation based on Wu et al. (1993), *ISIJ International*
33(1), 26–35, [doi:10.2355/isijinternational.33.26](https://doi.org/10.2355/isijinternational.33.26).
The pair reaction and entropy equations are Eqs. (2)–(5), p. 27; the
MgO–SiO₂ parameters are Eqs. (7)–(8), p. 30; pure oxide liquid functions are
Table 1, p. 28. It is experimental, leaves `evaluate()` and all defaults
unchanged, and is not a datapack or profile.

## Use and domain

Pass MgO and SiO₂ formula-unit mole amounts, for example:

```python
from openimcc import evaluate_mgo_sio2_mqm

result = evaluate_mgo_sio2_mqm({"MgO": 0.5, "SiO2": 0.5}, 1873.15)
```

Any other positive oxide raises `ImccComponentOutsideDomainError`; zero-valued
other entries are ignored. Empty and invalid compositions are refused. The
supported temperature interval is 1200–3000 K, the overlap between Wu's liquid
reference ranges (Table 1, p. 28) and the packaged OpenIMCC MgO(l)/SiO₂(l)
reference functions. Alternate gas packs must provide usable references over
the requested temperature. Temperatures outside that overlap raise
`ImccTOutsideDatapackDomainError`. Evaluating the OpenIMCC reference functions
uses `species_thermo`, so this candidate call requires the optional `gas`
extra. If `OPENIMCC_VAPOROCK_ROOT` selects an alternate gas pack, its MgO(l)/
SiO₂(l) references are used; the result records the loaded condensate table's
canonical SHA-256 in `standard_state`.

The result is a frozen `MgOSiO2MQMResult` with activities, mole fraction,
temperature, the ordered pair fractions `(X_MgMg, X_MgSi, X_SiSi)`, stability,
the estimated two-liquid bounds when present, source citation, standard-state
label, and candidate identity `wu-1993-mgo-sio2-mqm-openimcc-liquid-standards-v1`.
`stability` is `homogeneous` or `inside_two_liquid_field` for the sampled
liquid Gibbs-energy lower hull; the returned bounds are mole fractions of
SiO₂. Bounds use a 0.001 composition grid and are approximate.

This candidate refuses multicomponent melts. CaO, FeO, Al₂O₃, Na₂O, and other
regolith or basalt components are not represented or silently dropped. A
thermodynamic embedding is future work; this binary result does not predict
activities in a multicomponent melt. It also contains no solid phases, so it
does not calculate a complete liquidus or monotectic invariant.

## Derivation and standard conversion

For one mole of oxide formula units with SiO₂ mole fraction `x`, Wu's p. 27
equivalent fraction is
`Y_Si = 2 x / ((1−x) + 2x)`. The coordination numbers are twice the
coordination-equivalent factors, `z_Mg=1.37744375` and `z_Si=2.7548875`, so
`N_pair = (z_Mg(1−x)+z_Si x)/2`. Let `q` be the normalized number of unlike
pairs: `X_MgSi=2q`, `X_MgMg=Y_Mg−q`, and `X_SiSi=Y_Si−q`. Wu Eq. (2), p. 27,
then gives
`q²/((Y_Mg−q)(Y_Si−q)) = exp(−Δg/(RT))`. The evaluator solves this scalar
equation by log-space bisection over its feasible interval.

The minimized molar mixing Gibbs energy is the ideal oxide term plus the
pair configurational entropy from Eq. (5), p. 27, plus
`N_pair q Δg`. Wu Eqs. (3)–(4), p. 27, give that energy term because
`X_MgSi=2q`, `ΔH=X_MgSi ω/2`, and `ΔS_nonconf=X_MgSi η/2`. Here `Δg=ω−ηT`
uses Eqs. (7)–(8), p. 30, with energy in J/mol and entropy in J/(mol K).
For molar `g=G_mix/n`, the component derivatives are
`ln a_Si=(g+(1−x)dg/dx)/(RT)` and
`ln a_Mg=(g−x dg/dx)/(RT)`. Both numerator and `RT` are J/mol, making the
log activities dimensionless. At `x=0.5` and 2073.15 K, the calculated
Wu-standard `RT ln γ` values are −41602.0 J/mol for MgO and +4315.4 J/mol for
SiO₂, close to the printed values in Wu Table 2, p. 32.

Wu Table 1, p. 28, gives `G°_Wu(T)` relative to the elements. The loaded
OpenIMCC `MgO(l)` and `SiO₂(l)` functions provide `G°_open(T)` on the same
elemental basis. These are the packaged standards unless
`OPENIMCC_VAPOROCK_ROOT` selects an alternate gas pack. From `μ=G°+RT ln a`,
the conversion is
`ln(a_open)=ln(a_Wu)+(G°_Wu−G°_open)/(RT)`. The Gibbs difference is J/mol and
`RT` is J/mol. At 1873.15 K, the resulting log₁₀ activity shifts are
−0.000389 dex for SiO₂ and +0.032007 dex for MgO. These values also verify the
conversion sign and units. The shift preserves Wu's chemical potential, so the
pure-end activities equal `exp((G°_Wu−G°_open)/(RT))`; with the packaged
standards, pure MgO at 1873.15 K has `a(MgO)≈1.076`. Below 2200 K the packaged
MgO(l) row is the supercooled continuation `Mg-009-SC-CP`. Arutyunyan's
β-cristobalite silica activities are separately converted with
`ln(a_open)=ln(a_β)+(G°_β−G°_liquid,open)/(RT)` using the loaded OpenIMCC
high-cristobalite and liquid-silica references.

## Validation

The focused automated gates are in `tests/test_mqm_mgo_sio2.py`.
Arutyunyan et al. (2005) is the sole genuinely independent activity check;
Wu Table 2 and Kambayashi & Kato are fit or reproducibility evidence.

| Gate | Result |
| --- | --- |
| Wu's printed binary values | At `x_SiO₂=0.5`, all six MgO/SiO₂ `RT ln γ` values at 1800, 2000, and 2200 °C reproduce Wu Table 2, p. 32, within 5 J/mol after standard conversion. The largest absolute difference is 3.0 J/mol. The 5 J/mol tolerance allows integer table rounding and numerical differentiation. |
| Arutyunyan et al. (2005), sole independent activity check | Table 2, p. 394, reports 1873 K β-cristobalite-standard values. The `x=0.486` point is starred for melt + Mg₂SiO₄; the paper's liquidus values are `x=0.488` and `0.563`. The OpenIMCC conversion is −0.015727 dex. Candidate signed SiO₂ residuals from low to high `x_SiO₂` are `+0.0032, +0.0090, +0.0070, +0.0084, +0.0079, +0.0066 dex`; each is inside the predeclared ±0.015 dex gate. The tolerance covers printed activity rounding and the source Gibbs-function fit residual. The corresponding unchanged default SiO₂ residuals are `−0.3901, −0.3810, −0.3795, −0.3734, −0.3670, −0.3636 dex`, matching the reported approximately −0.38 dex bias. `a(MgO)` is relative to crystalline MgO and is not scored. |
| Chen & Presnall (1975), peritectic consistency check | Chen remeasured the Bowen & Andersen peritectic that Wu fitted using Table 1 solid enthalpy adjustments (−3206 J/mol for Mg₂SiO₄ and +453 J/mol for MgSiO₃); Chen is absent from Wu's references. Chen reports agreement within experimental uncertainty: Bowen's 1557 °C is 1562 °C on IPTS-68, and the liquid is 60.6 wt% SiO₂ versus Bowen's 60.9 wt%. The model output at 1830 K and `x=0.503` has both saturation indices within `1e-4` dex. At Chen's `x=0.50781` and 1830 K they are −0.0127 dex for forsterite and −0.0002 dex for protoenstatite, and the liquid is homogeneous. This is a consistency check, not an independent test. |
| Stability | The returned activities are derivatives of the single-phase Gibbs curve, including inside the estimated two-liquid gap: at 1960 K and `x=0.7`, `a(SiO₂)≈1.115`, versus about `0.985` at the binodal. At 1873.15 K the silica-rich two-liquid bounds are `x_SiO₂=0.567–0.994`; all six Arutyunyan points are outside that liquid–liquid field. At 1960 K the bounds are `0.596–0.988`. At 2214 K a narrow field remains (`0.835–0.846`); the sampled gap is present at 2214.10 K (`0.839–0.842`) and absent at 2214.16 K. Hudon et al. (2004), Table 7, doi:10.1029/2003JB002659, report the 1960 K bounds `0.596` and `0.988`, the 2123 K bounds `0.684` and `0.955`, and a 2214 K consolute at `x=0.843`. Separately, Belmonte et al. (2017) calculate a 2244 K consolute at about `x=0.84`. The gap was inside Wu's optimization; these are consistency checks against the assessed diagram. This liquid-only code does not solve a monotectic. The bound tolerance is the 0.001 `x` grid spacing. |
| Kambayashi & Kato (1984), not independent | Wu fitted these activities, so this is not held-out evidence. At 1973.15 K, four representative Table 2 rows have signed candidate residuals `+0.0203, +0.0855, −0.0118, −0.0096 dex` after the inferred β-cristobalite-to-liquid conversion. The source labels pure SiO₂(s); the high-β-cristobalite assignment is inferred. |
| Default output identity | The focused test records the 1800 K binary default `parent_activity`, `complex_x`, and `D` IEEE-754 bit sequences, calls the candidate, then verifies the default bit sequences are exactly unchanged. `model.py` and default pack data are untouched; the full suite is required alongside this check. |
| Multicomponent gate | Not applicable: the API refuses every other positive oxide. |

Kambayashi source: Kambayashi & Kato (1984), Table 2, journal p. 246,
[doi:10.1016/0021-9614(84)90085-5](https://doi.org/10.1016/0021-9614(84)90085-5).
Arutyunyan source: [Table 2 and discussion](https://repo.library.stonybrook.edu/xmlui/bitstream/handle/11401/70084/DOPv50i8final.pdf?sequence=2).
Chen & Presnall source: [1975 paper](https://deanpresnall.org/files/75Chen-Pres.FoQto25kb.pdf).

## Limitations

The liquid mixing function follows the binary Wu assessment and uses the
current OpenIMCC default pure-liquid reference functions for activity
conversion. It does not represent ternary interactions, redox, crystals,
pressure dependence, or multicomponent regolith melts. The two-liquid bounds
are a sampled liquid-only convex hull, not a replacement for a full phase
diagram assessment.
