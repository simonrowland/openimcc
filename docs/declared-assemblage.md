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
Na₂O, TiO₂, and K₂O. Pure JANAF crystal Gibbs energies are reconstructed from
the vendored enthalpy anchor, enthalpy increment, and entropy columns on the
same apparent-Gibbs basis used by the runtime liquid standard. The generated
subset records each source table digest; regenerate it with
`python tools/generate_declared_assemblage_janaf.py`. The former fusion-node
copy is not used. In particular, the ambiguous O-038 1700 K fusion datum is
discarded; pure crystal Gibbs values do not depend on that liquid row.

For each pure JANAF phase, the source span below is the span of finite JANAF
enthalpy-increment and entropy rows. Evaluation begins at 298.15 K because the
apparent-Gibbs enthalpy anchor is defined there, and ends at the crystal-to-
liquid transition listed below, inclusive. The active parent-liquid standard
must also cover the requested temperature; the effective interval is the
intersection of these domains and that liquid standard's coverage.

| Crystal | JANAF source | Tabulated source span (K) | Crystal-to-liquid cap (K) | Construction uncertainty input |
| --- | --- | ---: | ---: | --- |
| SiO₂ | O-035 | 298.15–3000 | 1996 | — |
| MgO | Mg-008 | 0–4000 | 3105 | — |
| CaO | Ca-027 | 0–3600 | 3200 | — |
| Al₂O₃ | Al-096 | 0–3000 | 2327 | — |
| FeO | Fe-018 | 298.15–2000 | 1650 | — |
| Na₂O | Na-012 | 0–2000 | 1405.2 | — |
| TiO₂ | O-043 | 0–2500 | 2130 | — |
| K₂O | K-012 | 298.15–2000 | 1013 (selected) | Fusion-temperature input 919–1190 K |

For K₂O, 919–1190 K describes uncertainty in the selected fusion-temperature
input, not an evaluation-temperature domain. The selected 1013 K transition
sets the inclusive crystal cap; the liquid standard further narrows the usable
range. K₂O uses the existing estimated liquid-from-solid standard and retains
that authority in the result. Fe³⁺-bearing phases, including magnetite and
hematite, are refused because this API does not balance iron redox. Mullite
currently has no sourced Gibbs function in the repository, so it returns a
typed `missing_phase_data` refusal and requested parent oxides remain
non-identifiable. A single sourced compound such as forsterite likewise does
not identify either of its two parent activities. The FC87 mullite-derived
entry in the melt packs is a liquid association fit, not a solid crystal Gibbs
function, so it does not supply the missing crystal equation.

The result's `declared_phases`, `oxide_activities`, and
`non_identifiable_components` are separate. `labels.notices` records that the
assemblage was assumed and not minimized; `labels.evidence_class` gives the
least authoritative participating source. The optional oxygen-balance result
only balances gas effusion fluxes at fixed condensed activities. It does not
close oxygen mass balance through the solids or feed gas back into the
assemblage.
