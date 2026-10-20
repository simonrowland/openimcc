# Redox inventory basis

An inventory is a map from element symbols to amounts in moles of atoms. Oxide
formula-unit amounts are converted through explicit atom counts. Oxygen is
counted as its own element total; it is not hidden in an oxide coordinate.
Temperatures use kelvin and standard Gibbs energies use joules per mole of
reaction. Redox fit residuals and accuracy bands carry their units directly:
species values use kJ mol⁻¹ species and reaction values use kJ mol⁻¹ reaction.
`fit_residual` stores source-fit quality only; `accuracy_band` stores source
uncertainty or a measured construction envelope. `band_basis.kind` is one of
`source_uncertainty`, `measured_construction`, `independent_measurement`,
`fit_only`, or `none`. `independent_check` records the check status, equilibrium,
source, conditions, measured/predicted values, miss, miss-to-band ratio, and
reason. A `not_found` check explains what was not located and keeps unknown
values null. The shared field contract is in
[`provenance-schema.md`](provenance-schema.md). Define
`λ = ln(fO2 / 1 bar)`. The oxygen chemical-potential shift is
`Δμ_O = RTλ/2` per mole of O atoms because the gas standard is O2. Reaction
rows store product-positive external O2 stoichiometry `ν_O2`; their fugacity
term is `-ν_O2 λ`. Thus `FeO + 1/4 O2 = FeO1.5` shifts `ln K` by `λ/4`.

`FeO1.5` represents one half formula unit of `Fe2O3`. Rebase a reaction
written per mole of `FeO1.5` to one mole of `Fe2O3` by doubling every
stoichiometric coefficient, including its oxygen coefficient and reaction
Gibbs energy. Complex-component activities use the Raoult standard,
`a_i = x_i*`, where `x_i*` is its complex mole fraction. When converting a
Henry-standard activity to the Raoult standard, multiply by the infinite
dilution Raoult activity coefficient:
`a_i^R = (H_i/f_i^*) a_i^H = γ_i^R(∞) a_i^H`.

For example, one mole of Fe2O3 maps to Fe = 2 mol atoms and O = 3 mol atoms.
Two moles of FeO plus one half mole of O2 map to the same element inventory.
The two formula splits are equivalent because the map is linear in atom counts.
The Fe2O3 input is never folded to two FeO formula units without separately
accounting for the oxygen difference.

`openimcc.redox_basis.oxide_inventory_to_elements()` is the pure mapping
function. It takes formula-unit mole amounts and a formula-to-atom-count
mapping, then returns mol atoms by element. It performs no formula parsing,
redox solve, or oxygen-fugacity adjustment.

`evaluate_redox(..., mode="closed")` fixes all supplied element totals,
including oxygen. In `mode="imposed"`, the solver fixes temperature, oxygen
fugacity through `lambda_imposed`, and all non-oxygen element totals, then
predicts both oxygen content and metal amount. Any oxygen value in the input
inventory is ignored. On the buffered branch, metal amount is chosen by the
Fe saturation condition `a_Fe = 1` at the imposed lambda. If changing metal
amount leaves that saturation lambda invariant, oxygen is needed to select an
extent; without it, the solver raises a typed endpoint error.

`RedoxResult.element_moles_by_oxidation_state` reports each present liquid
element as moles of atoms by integer oxidation state. MELT-pack complexes
expand through parent stoichiometry, and redox-pack species use their valence
metadata. Fe metal is included separately at state 0. `ferric_fe_mol` and
`ferrous_fe_mol` alias Fe states 3 and 2; form Fe3+/sum-Fe from the valence
totals, using states 2 and 3 for dissolved-only Fe and including state 0 for
total inventory Fe.

`unmodelled_redox_couples` records present elements whose known melt valences
are not represented by rows in the selected redox pack. Its records contain
`element`, `modelled_states`, and `unmodelled_states`, without estimated
shares. `redox_pack_digest` and `melt_pack_binding_digest` identify the pack
objects used by the solve. Result flag strings remain descriptive, not a
stable API.

The keyword-only `redox_pack_path` argument selects a validated redox pack;
omitting it uses the packaged default. Relative paths resolve against the
process working directory. Missing or unreadable files raise the typed
`ImccMalformedDatapackError` refusal with a message that omits the full path.
Content digest mismatches continue to raise the loader's `ValueError`.
The keyword-only `melt_pack_path` argument selects a validated melt
association pack for the same solve. It uses `load_datapack()` and its normal
identity checks; `None` preserves the packaged default. Missing, malformed,
unpublished, or identity-mismatched packs raise
`ImccMalformedDatapackError`. The selected pack's canonical binding identity
is returned as `melt_pack_binding_digest`.

## Chromium candidate pack

`src/openimcc/data/packs/openimcc-redox-cr-candidate-v3.json` is opt-in and is
not the default. Select it with
`evaluate_redox(inventory, T_K, mode, redox_pack_path="src/openimcc/data/packs/openimcc-redox-cr-candidate-v3.json")`.
It adds `1/2 Cr2O3 = CrO + 1/4 O2` using O’Neill & Berry (2021)'s 1-bar
composition and temperature relation. The Cr K is conditioned on normalized
single-cation oxide fractions. Since the Fe and Cr rows are solved together,
the Cr K iterates with the Fe2+O and Fe3+O1.5 fractions to a fixed point.
`chromous_cr_mol`, `chromic_cr_mol`, and `cr_oxygen_capacity_mol` report Cr(II)
and Cr(III) atom totals and the 0.5 mol O per mol Cr(II) capacity convention.

The candidate lists Berry & O’Neill (2004), Berry et al. (2006), Berry,
O’Neill & Foran (2021), and O’Neill & Berry (2021) as anchors. Schreiber &
Haskin (1976), Roeder & Reynolds (1991), Hanson & Jones (1998), and Apollo
14072 estimates are listed separately as independent checks. Reported bands
remain about 0.02 Cr2+ fraction model scatter and ±0.015 XANES measurement
uncertainty. The row flags its 1-bar scope, the 1673 K isothermal calibration,
the 1773 K variable-temperature limit, and compositions outside the demonstrated
CMAS and basaltic range; pressure is not modelled. It does not define a
composition-independent CrO Gibbs function.

## Titanium candidate packs

Two additional packs are opt-in. Select the Ti-only row with
`evaluate_redox(inventory, T_K, mode, redox_pack_path="src/openimcc/data/packs/openimcc-redox-ti-candidate-v3.json")`,
or select the combined Cr+Ti candidate with
`evaluate_redox(inventory, T_K, mode, redox_pack_path="src/openimcc/data/packs/openimcc-redox-cr-ti-candidate-v3.json")`.
Neither is the packaged default. The Ti-only pack starts from the default
redox pack; the combined pack carries both empirical rows.

The Ti row uses `TiO2 = TiO1.5 + 1/4 O2`, so `nu_O2=+1/4`. Berry et al.
(2024), equation 7, supplies `log10 K' = -1.84 + 3.44 Lambda + 6262.5/T`.
Berry et al. (2024, GCA 366, pp. 210-220, Table 1 p. 212; [article](https://doi.org/10.1016/j.gca.2023.11.014)) define optical basicity by oxygen-weighted oxide mole fractions,
`Lambda = sum(x_i * n_i * Lambda_i) / sum(x_i * n_i)` (supplementary text;
equivalently use oxide formula moles `m_i` in place of normalized `x_i`). Its
oxide coefficients are 0.48 (SiO2), 0.61 (Al2O3), 0.78 (MgO), 1.00 (CaO),
1.10 (TiO2), 1.15 (Na2O), and 1.00 (FeO). The TiO2 value is Berry's cited
Duffy (1989) value ([paper](https://doi.org/10.1111/j.1151-2916.1989.tb06022.x)); it is distinct from the commonly reused Duffy-Ingram/Mills
TiO2 value 0.61. The evaluator also includes K2O at 1.40 from Duffy's common
oxide/fluoride scale (1989, Table 1, p. 36, printed as 1.4;
[paper](https://doi.org/10.1016/0022-3093(89)90438-9)). Berry's measured Table 1 glasses support the 1.10 recipe:
recalculation from their reported oxide analyses gives the following pack
Lambda values, versus the published values:

| Berry glass | Published Lambda (Table 1, p. 212) | Pack Lambda |
|---|---:|---:|
| AD | 0.6094 | 0.611046 |
| AD+Fo | 0.6127 | 0.614028 |
| AD+Wo | 0.6345 | 0.635226 |
| AD+En | 0.5986 | 0.599595 |
| AD+Qz | 0.5633 | 0.564323 |

The small differences reflect recomputation from the rounded oxide analyses;
the pack uses the paper's definition and coefficient set, so Ti predictions
change by 0.000 dex at these anchors and at CST4. The pack-computed Berry
anchor envelope is 0.5643-0.6353; its flag interval `[0.5633,0.6353]` also
retains the published rounded endpoints. For CST4 (60.30 wt% SiO2, 31.80 wt%
CaO, 7.33 wt% TiO2), the pack gives Lambda 0.628191 (0.62819 rounded), both
before and after this audit. At 1843 K and 1e-9 atm it predicts Ti3+/sumTi
0.0327447, unchanged from the candidate equation.

Dissolved Fe(II) and Fe(III) both enter as bulk FeO at coefficient 1.00, so
the Lambda contribution does not depend on their split; Fe metal is excluded.
The shared fixed point supplies both dissolved Fe amounts. Berry's supplement
does not list K2O or Cr2O3: K2O is added from the Duffy scale cited above,
while Cr2O3 is omitted because a value on this same basis has not been verified
in a primary table. A positive Cr inventory therefore adds a candidate flag
that the melt is outside this optical-basicity basis. MnO and P2O5 remain
excluded as minor oxides. With product-positive oxygen stoichiometry, the row gives
`log10(Ti3+/Ti4+) = -log10 K' - 1/4 log10(fO2/1 bar)`.

Berry's five measured glasses anchor the position. Their measured `log K'`
values span 3.852-4.096 at 1400 °C, a 0.244 dex composition envelope; the
stored 7.8 kJ/mol value is that envelope converted at 1673.15 K and is not a
fit residual. Berry's FMQ-parallel `6262.5/T` slope is carried as an
assumption. Held-out checks remain separate: Tranell's CST4 titration is
0.090 Ti3+/sum Ti at 1843 K and 1e-9 atm, while this Berry equation predicts
about 0.033 there for CST4's Berry-recipe Lambda near 0.628. It is a comparison,
not a fit. Tranell reports 152 ± 25 kJ/mol (7940/T in her correlation; the
1997 free pO2 slope is -0.21 ± 0.01). Schreiber's 1977/1978 EPR/titration
series implies steeper isotherms, about 15000/T, and reports 1-7% Ti3+ for
lunar conditions. No alternate slope is fitted. Johnston (1965) alkali-silicate
isotherms are also held out.

The Berry measured glasses span published Lambda 0.5633-0.6345 (the pack
recomputes 0.5643-0.6353 from their reported oxide analyses), about 1.03 wt%
TiO2, and 1673 K. The evaluator flags active Ti rows outside the inclusive
Lambda envelope `[0.5633,0.6353]`,
above 1.03 wt% TiO2, or outside 1673-1674 K. The TiO2 threshold is the anchor
content because the available comparison cannot locate error onset between
about 1 and 7 wt%; the cutoff does not claim a validated onset. In held-out
Tranell titrations (73 scored points) span Lambda 0.6631-0.9079 and 7.3-52.35
wt% TiO2; all 73 lie outside the Berry Lambda envelope. CST4 (Lambda 0.628, 7.33 wt% TiO2) is the one Tranell slag inside the Lambda envelope; its titrations are listed under CST4a/CST4b without composition records and are not among the 73 scored points. A downstream reconstruction from the parent CST4 composition under-predicts it by about 0.44 dex, so the miss is not explained by Lambda extrapolation alone; CST4 is still outside in TiO2 content and temperature.  The candidate
under-predicts Ti3+/Ti4+ by a median 0.5246 dex (maximum absolute residual
1.3300 dex), unchanged because the pack recipe already matches Berry's
definition. Apollo
14072-like low-Ti basalt (about 2.6 wt% TiO2, Lambda about 0.63) is inside
the Lambda envelope but above the 1.03 wt% anchor TiO2 content, while high-Ti
mare basalt (TiO2 > 6 wt% in the Neal & Taylor 1992 classification, GCA 56,
2177-2211) is well outside. These
are extrapolation flags; evaluation continues.

The ideal Ti3+/Ti4+ mixing relation and FMQ-parallel temperature slope are
assumptions of the candidate, not established by the held-out comparison.
Tranell's larger temperature coefficient and flatter free-fO2 slope do not
explain the observed under-prediction; the available data cannot separate
composition transfer from high-concentration activity effects.

The Fe-free calibration is a different observable from quenched Fe-bearing
glasses, where Ti3+/Fe3+ exchange can occur on quench. The row flags
temperatures outside the 1273-1903 K measured comparison span and optical
basicity outside the five-glass flag envelope `[0.5633,0.6353]`; it evaluates extrapolated
points rather than refusing them. `titanous_ti_mol`, `titanic_ti_mol`, and
`ti_oxygen_capacity_mol` report the Ti(III)/Ti(IV) atom totals and the 0.5 mol
O per mol Ti(III) reduction capacity. `oxygen_capacity_mol_per_ln_f_o2`
includes the composition fixed point's total derivative.

## Threading

These solves use small dense linear algebra. Multi-threaded BLAS gives no
benefit and, under machine load, can cost orders of magnitude in wall time;
CPU time far above wall time is a sign of oversubscription. Callers can limit
BLAS threads with `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, and `VECLIB_MAXIMUM_THREADS=1` set before NumPy is
imported, or with one `threadpoolctl.threadpool_limits(limits=1)` context
around a whole run. Entering that context costs about 0.7 ms, so do not enter
it per solve; it does not control Apple Accelerate. The library does not
change thread settings.

The thread count can also change the last bits of a result. In measurements
with NumPy 1.26 on OpenBLAS, `numpy.linalg.solve` returned identical bytes at
every thread count for systems of up to nine unknowns; from ten unknowns the
one-thread result differed from the multi-thread results by about 1e-16 to
5e-15 relative, and larger systems differed between thread counts. A redox
solve with twelve parent oxides used 8 to 260 times the CPU of the one-thread
run when threads were left unlimited on loaded machines. Fix the thread count
when results must be reproducible to the last bit across processes or hosts.

## Frozen preregistered predictions — 2026-10-08

These predictions were recorded before any closed-mode redox result exists.
They are fixed expectations for later validation and are not fit targets.

For the FeO + 1/4 O2 = FeO1.5 row, a shift `delta G°` in the FeO1.5 standard
chemical potential changes the oxygen-fugacity crossover by
`delta lambda = 4 delta G° / (R T)`. Since FeO1.5 is one half Fe2O3(l), a shift
in the per-mole Fe2O3 Gibbs function contributes half as much to `delta G°`.
For scale, a 5 kJ/mol shift in the FeO1.5 standard state corresponds to about
0.62 log10 units of fO2 at 1673 K.

Without ferrite complexes and with no alkali dependence, the predicted Fe-only ferric
slope is about 1/4. Kress reports 0.196. A poor held-out match is a missing-
species signal; the slope is never tuned to the Kress value.

Erratum (2026-10-09): the preregistration premise was corrected to “without
ferrite complexes” before any closed-mode result existed; the original
preregistration date is unchanged.
