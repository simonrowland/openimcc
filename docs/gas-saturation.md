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

## Engine-neutral weighted net-flux root

`weighted_oxygen_balance_from_pressure_model` is the shared mathematical
primitive for a simulator that compares melt engines. Each engine supplies
equilibrium channel pressures in bar as a function of `log10(pO2/bar)` and
its own fixed surface snapshot. The simulator supplies one explicit oxygen
weight per channel, frozen nonnegative conductances in
`mol m^-2 s^-1 Pa^-1`, frozen bulk partial pressures in bar, and a bounded
log-pO2 interval, and may name the plane its bulk pressures describe with
`bulk_plane` (for example a well-mixed headspace); the label is echoed in the
result and never interpreted. The function solves

`Σᵢ wᵢ kᵢ (pᵢ,eq − pᵢ,bulk) = 0`,

converting bar to Pa once when it returns signed molar fluxes. It contains no
transport preset or engine-specific policy. A pressures-only engine must
additionally supply canonical species identity, oxygen atoms, parent-oxide
oxygen demand (or the chosen explicit weight), reservoir inclusion policy,
charged-species availability and any shared charge-aware transport policy.
Name the equilibrium and bulk pressure planes, surface temperature, and flux
area basis; report the callback's supported range, phase changes, validity and
omitted species. The root callback must hold temperature, melt state,
conductances and bulk pressures fixed over each solve.

The result reports the signed residual and the absolute net-transfer scale
`Σᵢ |wᵢ Jᵢ|`, split by the sign of weighted net flux into forward and backward
net scales. It separately reports the equilibrium and bulk pressure-transfer
magnitudes
`1e5·Σᵢ |wᵢ| kᵢ pᵢ,eq` and `1e5·Σᵢ |wᵢ| kᵢ pᵢ,bulk`; their sum is the gross
transfer scale `S_gross`. These gross magnitudes expose cancellation between
equilibrium and bulk pressure terms, including at exact zero net transfer.
A returned root is accepted only when its recomputed signed residual is at
most `1e-12` times the absolute net-transfer scale. Zero transfer is a
distinct state, and a numerically underflowed flux or weighted term is
refused. The result also reports the residual after negative fluxes are
clipped to zero. `applied_flux_closes` uses the relative applied residual
against the absolute net-transfer scale, without an absolute tolerance
floor; uniform positive scaling of all conductances therefore cannot change
that decision. Gross scales are evidence only and do not loosen this applied
source gate. Zero transfer is never certified as applied closure. A signed
root whose applied residual fails this check is not oxygen-closed after
clipping; the simulator must not certify it as a closed applied source. Root
count and monotonicity are sampled evidence for arbitrary callbacks, not a
global uniqueness proof.
`gas_channel_metadata` exposes openimcc's selected reaction metadata and
missing-table status using the same reaction tables and
`oxygen_balance_species_metadata`. The standalone saturation and molecular
effusion closures remain convenience policies; a cross-engine simulator uses
the weighted root with the same conductance and control policy for every
engine.
