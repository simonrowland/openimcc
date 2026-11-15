# Melt and crystal equilibrium

openimcc.equilibrium.equilibrate() computes a fixed-pressure Gibbs
equilibrium for parent-oxide moles, an ideal associate melt, and a declared
set of stoichiometric line-compound crystals.

    from openimcc.equilibrium import equilibrate

    result = equilibrate(
        {"CaO": 1.0, "Al2O3": 1.0, "SiO2": 2.0},
        1900.0,
    )

The default crystal candidates are the phase functions in the diagnostics
crystal dataset plus the reconstructed JANAF pure-oxide crystals. Pass
candidates= to restrict the set or to supply CrystalCandidate values for a
synthetic system. Restricted sets are explicitly labelled and cannot support
an unrestricted stability claim.

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
each present crystal equality, every absent-crystal inequality, and the inner
melt normalization, mass-action, and parent-inventory checks pass.

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
Brent's method. A solidus is the first heating state with certified positive
liquid; a liquidus is the first state where every feasible crystal is
certified absent and all checks pass. An unresolved amount range does not
establish absence. The result retains every crossing, its final temperature
bracket, endpoint status and phase sets. A refusal or unresolved point breaks
the scan; no root is interpolated across it. Liquid-dependent boundaries below
1700 K or above 3000 K are `OUT_OF_DOMAIN` on `temperature_K`.

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

See the API dataclasses in openimcc.equilibrium for the complete field
definitions and the typed refusal codes. `PhaseEquilibriumResult.to_dict()`
exports nested records, arrays, and read-only mappings as JSON-compatible
values.
