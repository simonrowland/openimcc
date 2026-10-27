# Opt-in gas saturation and boiling temperature

`evaluate_gas_saturation` computes the equilibrium vapor over caller-supplied
melt activities at a fixed temperature. It closes oxygen with the
pressure-weighted stoichiometric balance

`Σᵣ (nO_gas,ᵣ − dO_parent,ᵣ) pᵣ = 0`.

The result includes oxygen pressure, channel and species pressures, total
saturation pressure, mole fractions, domain flags, and root diagnostics. Set
`include_ions=True` to use the gas evaluator's charge-neutrality closure at each
trial oxygen pressure. The electron pressure is included in total pressure and
mole fractions. This equilibrium closure does not model ion transport or
finite vaporization of the melt.

`gas_boiling_temperature` solves `P_sat(T) = P_amb` over a temperature bracket.
It holds the supplied activities fixed across that bracket. Callers that need
activities to evolve with temperature must recompute them for the melt state
and call the saturation evaluator at each temperature. An unbracketed
temperature raises `ImccGasSaturationError`; unsupported thermochemistry also
raises its typed refusal unless extrapolation is explicitly enabled.

Both functions are opt-in. The existing `evaluate_gas_oxygen_balance` remains
the Knudsen effusion oxygen-flux closure, with its molecular-flow pressure
limit. Saturation results can exceed 1 bar; they report the ideal-gas
calculation and its domain flags without treating pressure as a vaporization
path or a finite-inventory state.

Pressures in the hundreds or thousands of bar under this ideal-gas,
dilute-ionisation model are not credible physical predictions. They are
closure snapshots at the supplied fixed activities; the domain flags identify
the model's extrapolation and pressure limits. Ion fractions are only as
complete as the ion rows present in the selected table. Ions without rows are
omitted from charge closure, so their fractions are not physically predicted
zeros.

When `root_uniqueness` is `"sampled"`, the solver scans a deterministic finite
grid and refuses if it finds more than one sign-changing interval. That scan
cannot exclude a narrow pair of roots or a tangent root between sample points.
The neutral path reports `"proved"` only when its pressure-exponent sign
criterion establishes monotonicity.
