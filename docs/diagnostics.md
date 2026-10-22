# Derived-property diagnostics

`openimcc diagnose cp-implied` reports the heat capacity implied by a loaded
association pack. It writes `cp-implied.csv` with per-composition, per-temperature
parent, configurational, and total Cp, per-association terms, and the h versus
h/2 finite-difference check. `cp-implied.md` is a short receipt keyed by model
id and binding digest. Supply `--pack PATH` to report another pack or `--out DIR`
to choose the output directory.

For each association, `log10 K = A + B/T` implies
`ΔrH = −R ln(10) B` and `ΔrCp = 0`. The diagnostic takes central differences of
`n_j = complex_x[j]/D`, moles of complex per mole of analytical oxide, and
calculates `Cp_conf = Σ ΔrH_j dn_j/dT`. The parent-standard Cp is added to obtain
total model Cp. This comparison uses the pack-cited JANAF liquid constants:
SiO₂ 85.772 (liquid branch from 1800 K) and Na₂O 104.600 J mol⁻¹ K⁻¹. The
JANAF constants are continued as supercooled liquid standards where needed. Values are reported on the analytical-oxide basis in
J mol⁻¹ K⁻¹.

## How to read the scores

The bundled Richet, Bottinga & Téqui (1984) Na₂O–SiO₂ dataset reports partial
molar Cp values of 100.6 J mol⁻¹ K⁻¹ for Na₂O and 81.37 J mol⁻¹ K⁻¹ for SiO₂,
with an additive and temperature-independent envelope of ±0.5% over 900–1800 K.
The comparison uses these values to derive measured mixture Cp. Data and source
records live in `src/openimcc/data/diagnostics/liquid-cp-calorimetry.json`.

The scorer reports four distinct checks:

- **(a) Absolute total:** parent-standard Cp plus `Cp_conf` against measured Cp
  and the source envelope.
- **(b) Composition curvature:** the residual of `Cp_conf` from its best straight
  line across the measured compositions at each temperature.
- **(c) Temperature flatness:** the largest deviation of `Cp_conf` from its mean
  across the source's measured temperature interval.
- **(d) Pure end-member control:** the total-Cp offset at a pure end member,
  reported as a standard-state offset and kept separate from curvature and
  temperature dependence.

The trap is fitting a baseline to the same liquids and then testing
whether the model's excess is near zero. A fitted pair of partial molar heat
capacities can absorb any excess that is linear in composition and constant in
temperature. Testing the raw `Cp_conf ≈ 0` would therefore penalize an effect
that the source's own fitted baseline already includes. This report instead
scores the part that baseline cannot absorb: curvature, temperature variation,
and the absolute total including parent standards. It never uses `Cp_conf ≈ 0`
as a pass criterion.

Each CSV verdict is labelled (a), (b), (c), or (d). The pure-silica standard-state
offset is reported as its own (d) result. SiO₂ remains among the measured
compositions in the (b) line fit, and its row can also have a curvature residual.
Curvature residuals are compared with the source's stated relative envelope
applied to the measured mixture Cp. The h/2 column should closely match the h
estimate; a large difference flags finite-difference sensitivity. Scores are
reported only and are never used to choose coefficients. Dataset choice follows
source fidelity.

The next diagnostics in this family are implied mixing enthalpy,
`H_mix = Σ n_j ΔrH_j`, against solution/drop calorimetry, and implied partial
molar enthalpy, `d ln a/d(1/T)`, against EMF temperature coefficients. They are
not implemented here.

## Sources without numeric scores

The data file also records Richet & Bottinga (1985), Stebbins, Carmichael & Moret
(1984), and Lange & Navrotsky (1992) as unscored entries with the reason each is
not scored. No composition-matched liquid Cp excess with a usable numeric
uncertainty was available for those records when this dataset was prepared.
