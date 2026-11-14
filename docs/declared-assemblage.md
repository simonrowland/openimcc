# Vapour over stated crystal phases

`evaluate_declared_assemblage(phases, T_K, ...)` solves the chemical-potential
equations for exactly the crystal phases supplied by the caller. It reports
each uniquely determined oxide chemical potential and activity relative to
the active parent-liquid standard. It can pass those activities to
`evaluate_gas()` or, with `oxygen_balance=True`, to the gas layer's oxygen
balance solver. Gas pressures use a fixed 1 bar standard state.

The supplied phases are treated as coexisting. This calculation does not
minimize over phase amounts, choose a phase assemblage, prove that the supplied
phases are stable, or test stability against an absent phase or liquid. Its
result is a vapour potential over the stated solids, not a phase-equilibrium
answer.

```python
from openimcc import evaluate_declared_assemblage

result = evaluate_declared_assemblage(
    ("MgO", "spinel"), 1800.0, fO2=1e-8
)
result.oxide_activities
result.gas
```

For each declared phase, the evaluator uses `ν · μ = G°(cr)` and solves the
combined linear system. An oxide activity is returned only when that component
of `μ` is fixed by the equations. `non_identifiable_components` lists the
remaining components; gas channels that need them appear in `gas_depends` and
are not evaluated using a guessed activity. An inconsistent overdetermined
set returns an `inconsistent_phase_set` refusal with its per-phase residuals.

The supported crystal functions include the sourced phases in the diagnostics
crystal dataset and pure-oxide JANAF crystals: SiO₂, MgO, CaO, Al₂O₃, FeO,
Na₂O, TiO₂, and K₂O. The JANAF source intervals are enforced without crystal
extrapolation. K₂O uses the existing estimated liquid-from-solid standard and
retains that authority in the result. Fe³⁺-bearing phases, including magnetite
and hematite, are refused because this API does not balance iron redox.

The result's `declared_phases`, `oxide_activities`, and
`non_identifiable_components` are separate. `labels.notices` records that the
assemblage was assumed and not minimized; `labels.evidence_class` gives the
least authoritative participating source. The optional oxygen-balance result
only balances gas effusion fluxes at fixed condensed activities. It does not
close oxygen mass balance through the solids or feed gas back into the
assemblage.
