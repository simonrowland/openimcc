# Redox inventory basis

An inventory is a map from element symbols to amounts in moles of atoms. Oxide
formula-unit amounts are converted through explicit atom counts. Oxygen is
counted as its own element total; it is not hidden in an oxide coordinate.
Temperatures use kelvin, standard Gibbs energies use joules per mole of
reaction, and uncertainty bands use kilojoules per mole of reaction. Define
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

Erratum: the preregistration premise was corrected to “without ferrite
complexes” before any closed-mode result existed.
