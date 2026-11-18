# Molecular dynamics validation decks

These LAMMPS decks are independent checks on the liquid heat-capacity
continuations used by OpenIMCC. They do not calibrate, fit, or anchor the
thermochemical model.

## Contents

- `inputs/in.point.lmp` and `inputs/in.prep.lmp`: production and soft push-off
  LAMMPS inputs. `material` selects CaO, MgO, TiO2, Al2O3, Cr2O3, or SiO2.
- `scripts/make_cell.py`: neutral random periodic starting cells.
- `scripts/run.sh`: sequential temperature runs with relaxation and
  crystallization screening. Set `LAMMPS`, `OUT`, and `MATERIALS` first.
- `inputs/potential.md`: parameter values, conversions, citations, and limits.
- `analysis/analyze.py`: enthalpy and density with contiguous-block standard
  errors; it checks completed sample files, production timesteps, target
  temperature (within 5%), and actual formula-unit counts. It also fits a
  quadratic heat capacity and reports an algebraic consistency check.
  NumPy 1.x and 2.x are supported.

**Status: syntax-checked, not executed.** `lmp` was unavailable while these
deck fixes were made. The pilot analysis uses one cell seed and one Langevin
stream per point; its block standard errors describe within-stream sampling and
do not estimate between-stream uncertainty. The heat-capacity algebraic
consistency check only confirms that the fitted polynomial and derivative
agree; it is not an independent validation of the fit.

## Measurements and outcomes

The decks sample enthalpy, oxygen mean-square displacement, density, and a
crystallization-order diagnostic over temperature. A quadratic fit to sampled
enthalpy estimates heat capacity. The pilot found:

- **SiO2:** calibration failed; the simulated liquid Cp was about 20–25% higher
  than the calorimetric liquid Cp.
- **Al2O3:** calibration passed its comparison gate.
- **CaO:** simulated liquid density was too low for the density gate.
- **TiO2 and MgO:** exploratory checks; no independent validation result was
  established.
- **Cr2O3:** unconverged. Enthalpy continued to drift and oxygen diffusion
  remained sluggish, so the apparent Cp is not usable.

The Guillot and Sator (2007) potential is parameterized for natural silicate
melts, not pure refractory endmembers. The Cr2O3 rigid-ion set is reported by
San Miguel et al. (1998), who took its short-range parameters from Lewis and
Catlow (1985, via San Miguel Table I). Transfers to bulk pure-oxide liquids
remain exploratory.

## Run

Install LAMMPS with `KSPACE` and `OPENMP`, plus Python with NumPy. Generate a
cell with `python3 scripts/make_cell.py al2o3 1500 cell.data --density 2.93`,
then configure the driver environment and run `scripts/run.sh`.
Outputs are written only to the selected `OUT` directory. Analyze a completed
manifest with `python3 analysis/analyze.py --manifest OUT/progress.json
--output OUT/analysis.json`.

No trajectories or other large binary outputs are included here.
