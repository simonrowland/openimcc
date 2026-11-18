# Melt and crystal equilibrium

openimcc.equilibrium.equilibrate() computes a fixed-pressure Gibbs
equilibrium for parent-oxide moles, an ideal associate melt, and a declared
set of stoichiometric line-compound crystals.

    from openimcc.equilibrium import equilibrate

    result = equilibrate(
        {"CaO": 1.0, "Al2O3": 1.0, "SiO2": 2.0},
        1900.0,
    )

The default crystal candidates include the phases in the diagnostics crystal
dataset, the JANAF pure-oxide crystals, and six aluminate crystals from Barin
(1995): CaAl2O4 (CA), CaAl4O7 (CA2), Ca3Al2O6 (C3A), Ca12Al14O33 (C12A7),
NaAlO2, and MgAl2O4. Their printed solid Gibbs-energy nodes are used only
within these ranges: CA 298.15–1800 K, CA2 298.15–2000 K, C3A
298.15–1800 K, C12A7 298.15–1700 K, NaAlO2 298.15–1900 K, and MgAl2O4
298.15–2408 K. NaAlO2 keeps the SOL-A to SOL-B transition at 740 K. Fusion
nodes use a common absolute-G Hess basis: each species is evaluated from its
formation enthalpy at 298.15 K, enthalpy increment, and entropy, with
ΔG_fus = ΣνG_abs,JANAF(parent liquid) − G_abs,Barin(crystal). The JANAF
parents use the packaged liquid-pair tables; no temperature-dependent element
functions enter this cycle. This basis is recorded as
`openimcc-barin-aluminate-crystals-v3`; the earlier v2 identity retains its
element-referenced meaning. Barin's compound-liquid rows are not candidates.
Source locators are Barin (1995), pp. 442–445 (calcium aluminates), p. 1132
(NaAlO2), and p. 1015 (MgAl2O4).

The R&H crystal rows use the `openimcc-crystal-phases-v2` identity: each source
crystal's absolute Gibbs energy is reconstructed from its printed 298.15 K
formation enthalpy, enthalpy increment, and entropy, then differenced against
the existing parent-liquid standards on the same absolute-G basis. Berman rows
and JANAF pure-oxide controls are unchanged; the published v1 identity retains
its former element-referenced meaning. R&H melting limits are recorded for
fayalite (1490 K) and perovskite (2188 K). Ilmenite's table ends at 1600 K
without a printed melting point, so it is missing above its final source node.

Between their final printed nodes and source limits, CA, CA2, and C3A continue
with constant heat capacity from the last Barin Cp node, anchored at the last
printed fusion node and adding JANAF liquid-parent absolute-G increments less
the Barin-crystal absolute-G increment. Since the v3 anchor is already on that
common basis, no element-reference residual is carried into the short gap.
These crystals remain available through their printed decomposition points.
Each calculation using a continued segment records its phase and temperature
range in a notice and receipt. The continuation stops at the source limit.

Barin defines its standard state and element reference phases at 1 bar
(Chapter 4, p. I-36), the same reference pressure used by the solver. It prints
decomposition points for CA at 1875 K, CA2 at 2035 K, and C3A at 1808 K
(pp. 442–443 and 445), and a SOL-to-LIQ melting row for MgAl2O4 at 2408 K
(p. 1015). Above these temperatures those crystals are excluded because their
source says the solid is no longer stable. Berman multioxide crystals are
treated the same way using their recorded melting point, including an
incongruent melting onset. These exclusions appear in the
`crystals_above_source_melting_point` notice and receipt field; they do not
count as missing data. Barin prints no melting or decomposition point for
C12A7, whose table ends at 1700 K, so it remains missing above that range.
NaAlO2's printed SOL-A to SOL-B change at 740 K is a solid transition, not a
melting point (p. 1132).

CA6 (CaAl12O19) and beta-alumina are not in the candidate data. A
`missing_phase_data` refusal means the candidate set is empty, a required
crystal has no shipped Gibbs function, or a crystal function is outside its
source range. At 1800 K, for example, the default CAS case lists C12A7 as
missing because its table ends at 1700 K, and CA6 because it has no shipped
Gibbs function. When the bulk contains the parent oxides needed to form either
missing phase, the solver keeps the phase-set result labelled `DEPENDS` on
missing phase data, even if its numerical equilibrium checks pass. A table
that ends without a printed melting or decomposition point also remains
missing above its last Gibbs node. `IN` means the state is certified among the
candidates available at the requested temperature; crystals excluded above
their source-printed melting or decomposition point are listed in the
`crystals_above_source_melting_point` notice. Pass candidates= to restrict the
set or to supply CrystalCandidate values for a synthetic system. Restricted
sets are explicitly labelled and cannot support an unrestricted stability
claim.

For MgAl2O4, the sources remain separate on the shared 1323–2408 K range.
The existing Berman fusion function is 5.677–9.326 kJ/mol above the Barin
Hess-cycle fusion values over the combined source-node grid. Barin's crystal
formation Gibbs values are 9.779–21.962 kJ/mol above JANAF Al-089 on that
range; subtracting the same JANAF liquid-oxide formation values leaves this
formation-energy difference unchanged. No average is formed.

## Scope

- The melt pack must be ideal: any active excess-Gibbs term returns the typed
  unsupported_nonconvex_model refusal.
- Crystals are stoichiometric line compounds. Solid solutions and ferric or
  other redox-sensitive phases are unsupported.
- Pressure is fixed at 1 bar.
- Liquid-dependent claims are in-domain from 1700 K through 3000 K. Outside
  that interval the result may contain a labelled crystal-only minimum, with
  OUT_OF_DOMAIN on temperature_K; it makes no liquid stability claim.
- Exact-zero bulk parents are removed from the solve. No amount or activity
  floors are added.
- Crystals requiring any exact-zero bulk parent are labelled `infeasible` and
  cannot affect missing-data validity.

## Result and certificates

PhaseEquilibriumResult separates positive liquid amounts, solid amounts and
their feasible ranges, phase presence, saturated candidates, activities,
phase affinities, labels, notices, a refusal, and a calculation receipt.
kkt_recheck contains residuals recomputed independently at the returned
state. A positive-liquid state is accepted only after the parent balances,
each present crystal equality, every available absent-crystal inequality, and
the inner melt normalization, mass-action, and parent-inventory checks pass.
An unavailable crystal above its source-printed melting point is absent by
source rule, not because its calculated affinity shows thermodynamic undersaturation.

If an otherwise direct inner-melt root is accepted after analytic Newton
polishing, its `solver_path` is `"direct-newton-polish"`; this path has zero
continuation stages. A composition or temperature recovery continues to use
its corresponding continuation path.

An inner melt arithmetic error is reported as the typed refusal code
`arithmetic_failure`. The `kkt_residual` code retains its distinct meaning: the
independent KKT recheck failed. Other typed inner-melt refusals use
`inner_melt_failed`.

`labels.evidence_class` reports the least-authority source token. Its
`unresolved` value is reserved for a source token genuinely absent from the
authority rank table and for early missing-phase or redox refusals where the
source set is incomplete. All source-authority tokens shipped with the
default candidate and parent data are ranked, so ordinary default results do
not emit `unresolved` because of a missing rank. This does not change the
meaning of any phase-state field; the phase-state contract remains version 1.

When the bulk can be represented by crystals alone, the solver first minimizes
the crystal Gibbs energy as a linear program. Its dual chemical potentials
must satisfy every available crystal inequality and agree with the primal
objective. The liquid-absence test then minimizes

    Q = sum(a_i) + sum(K_j * product(a_i ** nu_ij))

over the complete dual optimal face. The sum uses the melt pack's full free
parent and associate species list. A numerically bounded log(Q) <= 0 at a
feasible supporting dual certifies that adding an absent liquid cannot lower
Gibbs energy within the declared model and candidate set. If the reported
logQ_interval straddles zero, it is not a fully-solid certificate. The solver
checks the positive-liquid branch; if neither branch resolves coexistence, it
returns `incipient_liquid_unresolved`. Rounded Q == 1 is not a certificate.

The LP dual inequalities alone are insufficient: above a pure oxide's
transition, a crystal-only LP can remain feasible while the liquid is more
stable. That is why the optimal-face Q check is mandatory. A failed or
unresolved certificate triggers the positive-liquid solve. A passing,
independently rechecked liquid state is returned; the crystal-only diagnostic
is retained only when neither branch establishes an equilibrium state.

Pure parent crystals above their stable cap are marked absent by construction
only if the selected liquid standard and uncapped crystal standard establish
`Gl−Gcr < 0` at the requested temperature. The phase affinity is
`(Gl−Gcr)/(RT)+ln(a)`; because feasible ideal-melt activities satisfy
`ln(a)≤0`, that strict negative reference difference proves negative affinity.
At 1800 K this gives K2O −20.14 kJ/mol and Na2O −13.37 kJ/mol. It does not
prove FeO absent: against the selected DG46-MASSON65 liquid standard,
`Gl−Gcr=+5.85 kJ/mol`, so FeO remains unavailable outside its crystal domain.
This sign check is specific to the selected standard and temperature; a JANAF
liquid/crystal pair cannot substitute for it. Multioxide compounds without
their own Gibbs functions remain unavailable.

When the crystal-only LP is solved but liquid absence is unresolved, its
candidate amounts and optimal-face ranges appear in the calculation receipt.
Equilibrium solid amounts, phase presence, amount uniqueness and liquid mass
fraction remain unresolved until the absent-liquid certificate passes.

phase_amounts_unique is reported separately from activity identifiability.
At polymorph crossings or other non-unique extents, amount ranges are exposed
and the solver does not report an arbitrary split as a measured phase amount.
On the fully solid branch, each parent chemical potential is bounded
independently over the dual optimal face. `activity_ranges_ln` reports those
log-activity bounds after subtracting the selected liquid standard and
dividing by RT. An activity appears in `identifiable_activities` only when
both bounds are finite and their width is at most the 2e-9 ln-activity error
budget. Primal amount ranges use independent endpoint solves; uniqueness is
tested against the bulk-relative amount tolerance and each phase's bulk scale.
Activity and amount uniqueness are independent, so either can be unique while
the other is not.

At rank-deficient invariant points the active affinity Jacobian is decomposed
with SVD. Newton steps solve only the consistent row space; the nullspace is
retained for lever-rule amount ranges. The range solve includes a nonnegative
scale for the liquid at its certified parent proportions, since ideal-melt
activities do not change when every liquid amount is scaled together. If that
scale is not unique, `liquid_parent_mol` and `liquid_mass_fraction` are left
unset. Rank deficiency by itself is accepted; incompatible equalities beyond
the affinity tolerance return `singular_inconsistent_affinity`.

## Temperature sweeps and boundaries

`equilibrate_sweep()` accepts temperatures in the order supplied and carries
the previous stable phases, extents, and kernel parent root as initial guesses.
Every point still runs its own domain checks and independent KKT recheck. A
seed does not contribute to the certificate.

`phase_boundary_brackets()` scans the requested interval for every change in
the certified phase set, then refines each eligible solidus or liquidus with
Brent's method in either transition direction. A solidus is the first heating
state with certified positive liquid; a liquidus is the first state where
every feasible crystal is certified absent and all checks pass. An unresolved
amount range does not establish absence. The result retains every crossing, its final temperature
bracket, endpoint status and phase sets. A refusal or unresolved point breaks
the scan; no root is interpolated across it. Liquid-dependent boundaries below
1700 K or above 3000 K are `OUT_OF_DOMAIN` on `temperature_K`.
Across a source-printed melting cap, if the certified phase set or liquid
presence changes, the scan records a `source_cap` notice naming the crystal
and cap and returns `DEPENDS` for the affected boundary type (both types when
liquid presence changes), without reporting a model crossing at the cap. Model
crossings away from source caps remain eligible for `AVAILABLE` when their
residual passes the coexistence tolerance.

    from openimcc.equilibrium import equilibrate_sweep, phase_boundary_brackets

    sweep = equilibrate_sweep(
        {"CaO": 1.0, "SiO2": 1.0}, [1800.0, 1900.0, 2000.0],
    )
    boundaries = phase_boundary_brackets(
        {"CaO": 1.0, "SiO2": 1.0}, 1750.0, 2200.0,
    )

The sweep returns one `PhaseEquilibriumResult` per input temperature. Boundary
results separate `solidus_status` and `liquidus_status` (`AVAILABLE`,
`NO_CROSSING`, `OUT_OF_DOMAIN`, or `DEPENDS`) and include zero or more refined
crossings for each boundary type.

## Fixed-activity vapour

Pass `fO2` to `equilibrate()` to attach `evaluate_gas()` over the result's
identifiable parent activities:

```python
result = equilibrate(
    {"MgO": 1.0, "SiO2": 1.0}, 2000.0, fO2=1.0e-8,
)
pressures_bar = result.gas
```

`fO2` is the caller-stated oxygen fugacity ratio `pO2 / 1 bar`. The attachment
uses the same channel filtering as `evaluate_declared_assemblage()`. It passes
only activities in `identifiable_activities`; a gas row that needs an
unidentified activity appears in `gas_depends` and is not evaluated. The
returned pressures are a fixed-activity equilibrium vapour potential over the
stated bulk at `T`. They do not close oxygen or vapour mass balance and make
no rate, extent, or depletion claim. `labels.validity` and
`labels.out_of_domain_axes` describe the condensed equilibrium only. Vapour
status and `vapour.out_of_domain_axes` describe the gas overlay; vapour status
cannot be better than either the condensed equilibrium or gas rows.

## Simulator phase-state contract

`PhaseEquilibriumResult.to_dict()` retains its result fields and adds a
`phase_state_contract` object. Its version is exported as
`openimcc.equilibrium.phase_state_contract_version` (currently `1`):

```text
phase_state_contract = {
  "name": "openimcc.phase-state",
  "version": 1,
  "input": {
    "bulk_basis": "parent_oxide_mol",
    "bulk_parent_mol": {oxide: mol, ...},
    "temperature_K": number,
    "pressure_bar": 1.0,
    "candidates": null | [phase_id, ...],
    "fO2": null | pO2_over_1_bar
  },
  "output": {
    "phase_presence": {phase_id: status, ...},
    "phase_amount_ranges_mol": {phase_id: [minimum, maximum], ...},
    "phase_amounts_unique": boolean,
    "liquid_mass_fraction": {
      "value": null | number,
      "certification_status": "CERTIFIED" | "UNRESOLVED"
    },
    "identifiable_activities": {oxide: activity, ...},
    "labels": PhaseEquilibriumLabels,
    "refusal": null | EquilibriumRefusal,
    "receipt": CalculationReceipt,
    "vapour": {
      "status": "NOT_REQUESTED" | "AVAILABLE" | "DEPENDS" | "OUT_OF_DOMAIN",
      "interpretation": string,
      "fO2": null | number,
      "unit": "bar",
      "partial_pressures_bar": null | {species: pressure, ...},
      "depends": {species: reason, ...},
      "domain_flags": {species: flag, ...},
      "out_of_domain_axes": [axis, ...],
      "provenance_class": {species: authority, ...},
      "omitted_channels": {species: reason, ...},
      "error": null | string
    }
  }
}
```

The input `candidates` is `null` for the default candidate set. A certified
liquid mass fraction is resolved within the returned equilibrium certificate;
candidate coverage limits remain visible in `labels` and `refusal`. Gas
channels that depend on an unidentified activity are listed in `vapour.depends`
and have no pressure value. An external simulator can adopt this dictionary
interface and decide how to consume a multiphase state; this package cannot
change the external engine itself.

Consumers should check both `name` and `version` before reading the contract,
and should reject an unsupported version rather than guessing field meanings.
Within a version, producers may add fields while preserving existing field
names, types, units, meanings, and status values; consumers should ignore
unknown fields. Removing or renaming a field, changing a field's type or unit,
changing its meaning or an existing status value, or making a previously
optional field required needs a version bump. A new version may add fields
while retaining prior semantics. In version 1, `labels.validity` describes the
condensed equilibrium only; vapour validity and its out-of-domain axes are
reported on the vapour block. Consumers should branch on the version before
interpreting fields whose meanings differ.

See the API dataclasses in openimcc.equilibrium for the complete field
definitions and the typed refusal codes. `PhaseEquilibriumResult.to_dict()`
exports nested records, arrays, and read-only mappings as JSON-compatible
values.
