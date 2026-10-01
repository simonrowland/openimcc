# Changelog

## Unreleased

- Added fitted optional JANAF phosphorus and sulfur gas rows. Sulfur channels
  accept caller-supplied S2 fugacity. No public evaluated P2O5(l) G(T) function
  was found; JANAF lists P4O10(cr) only, so P channels need an external,
  source-rated P2O5(l) standard state. Default outputs stay fixed.
- Exported `species_thermo` for row-level Cp, S, apparent enthalpy and Gibbs
  energy, plus `default_gas_channels` for stable access to channel selection.
  Added JANAF in-interval fidelity gates, pinned the LAM/JANAF parent-liquid
  source differences, and proved the Na2O(l) and Al2O3(l) prior rows fail them.
- Corrected the default Na2O(l) row to LH84 Table 2's Na2O liquid coefficients
  and 1405 K lower bound; the prior coefficients came from NaO2(l). Corrected
  Al2O3(l)'s anchor to the LH87 solid 298 K value, removing the fusion
  enthalpy that had been counted twice. Updated gas comparison results and
  recorded the Na2O(l)-versus-JANAF source difference.
- Documented a known limit: complex rows below their demonstrated range, and their consistency with JANAF.
- Extended element completeness screening to 1200–3000 K, recomputed C2/C3
  maxima, and recorded C4 gaps in gas and parent-liquid coverage below 1500 K.
- Added 500–1500 K interval 2 rows for Na2O(g) and K2O(g), using LH84
  formation/entropy anchors with piecewise NASA Glenn heat-capacity functions;
  added a 1200–1500 K NbO2(l) interval while preserving its existing row.
- Recorded the JANAF glass-branch limits for TiO2(l) and V2O3(l); their C4
  gaps remain because their liquid branches begin at 1400 K and 1600 K.
- Below every declared gas interval, extrapolate from the lowest interval rather than the first row in file order.
- Made Na2O and K2O gas channels optional when alternate tables lack their rows, and report each skipped channel.
- Added a second JANAF-fitted Shomate interval for the battery gas species over
  500–1500 K. It uses at least ten complete 100 K JANAF nodes per species,
  preserves the existing interval 1 rows, and selects interval 1 at 1500 K.
- Added opt-in research pack `gas-janaf-parent-liquids-research` with JANAF-fitted
  SiO2(l), Al2O3(l), MgO(l) and CaO(l) rows; the SF04/Lamoreaux default
  coefficient contract keeps it out of default loads and outputs.
- Added JANAF-fitted Al2, Si2, and Si3 gas channels after a magnitude screen at
  melt temperature; their parent-reaction stoichiometry and source
  residuals are covered by the gas-layer tests and provenance ledger.
- Added JANAF-fitted Cr, CrO, CrO2, and CrO3 gas channels with a fitted Cr2O3(l)
  parent. Cr2O3 is caller-supplied because it is outside the IMCC melt basis;
  the parent uses its liquid branch from 1900 K, with lower temperatures
  flagged or refused; transition-marked Cr-015 nodes are omitted rather than
  assigned a phase branch. Cr(g) is fitted over the complete Cr-005 rows from
  1500-2900 K; JANAF's 3000 K row is parse-ambiguous after the element
  reference switches at the 2952 K boiling point.
- Added JANAF-fitted V, VO, VO2, Nb, NbO and NbO2 gas channels with fitted
  caller-supplied V2O3(l) and NbO2(l) parents. Transition-marked liquid nodes
  are omitted rather than assigned a phase branch; candidate liquid records
  remain vendored for source and transition coverage evidence.
- Wired data-free Mn/Ni/Co and monoxide channels; they activate only when an
  external pack provides their gas and parent rows, while the public pack has
  no MnO(l), NiO(l) or CoO(l) parent rows.
- Added Na2O(g) and K2O(g) channels from the NIST evaluation using LH84
  formation anchors (entropy converted from LH84's 1 atm standard state to
  1 bar) and NASA/Gurvich heat-capacity shapes; the roughly
  15 kJ/mol LH84-vs-NASA/Gurvich disagreement remains an open certification
  item.
- Added JANAF-derived gas tables with tracked provenance and fit checks.
- Scored the bench `partial_pressure` observable through the analytical gas
  layer, including the bar-to-Pa conversion.
- Added non-finite input guards with typed refusals.
- Raised kernel and model test coverage to 99%.
- Documented the datapack format and loader checks.
- Added the independent Schaefer & Fegley (2004) published reference and
  comparison tests.
- Narrowed the data licence to what the project can license.
  Documented the Schaefer & Fegley (2004) DOI erratum: the wrong DOI
  sits in the hash-verified SF04 core that both shipped packs carry, so
  that core is left unchanged. Re-issued the ext-v4 research pack as
  `1.0.2-ext-sp-2` with its private working paths removed; its core and
  rows are unchanged. Retired the ext-v1 to ext-v3 packs, which could not
  be loaded, and added a test that loads every shipped pack.
- Results are flagged instead of refused when they fall outside what the
  model covers: the species-coverage edge on the acid-sink ratio, a
  temperature outside the source paper's demonstrated window, and a
  notice on Na- and K-bearing results about the known low alkali bias.
- The bench reports by slice (dataset, observable, standard state)
  instead of one RMSE across incompatible standard states, and names
  every extrapolation or envelope override. The text report still prints
  the arithmetic total, labelled as not an accuracy figure.
- `evaluate_gas` returns `ImccGasResult`: partial pressures in bar with
  per-species domain flags and provenance classes. It predicts and flags
  outside the tables' temperature bands by default;
  `allow_extrapolation=False` refuses instead.
- Pinned the accuracy contract against the published SF04 reference as
  signed median residuals, and made the test suite fail on any warning.
- Added a quickstart, and a test that runs every documented Python and
  shell command and checks its output. The pip install commands are
  text-checked against the package extras, not run.
- Stated the Na species-set sensitivity jointly: removing every
  Na-bearing complex overshoots the SF04 Table 9 Na miss, so the
  single-family shifts do not add.
- Added CONTRIBUTING, with the provenance rule for proposing a datapack
  row.
- Added the Ti(g), TiO(g) and TiO2(g) gas channels of the TiO2 parent.
  Their Shomate rows are fitted from NIST-JANAF Ti-006, O-022 and O-046
  over 1500-3000 K the same way as the existing rows (max residual
  0.88-1.20 J/mol). The missing piece was the parent, not only the gas
  side: a TiO2(l) condensate row is now fitted from the JANAF liquid table
  O-044 over the same interval (max residual 6.9 J/mol, 1.6e-4 dex; the
  1500-2130 K part of that interval is JANAF's supercooled liquid). The
  generator in `tools/build_gas_tables.py` builds both and leaves every
  transcribed condensate row untouched. The Ti channels appear in a
  default `evaluate_gas` call when they are available: TiO2 is among
  `parent_oxides` and the active tables carry the Ti gas rows and the
  TiO2(l) row. Under `OPENIMCC_VAPOROCK_ROOT` (no TiO2(l) row) or with a
  `parent_oxides` list without TiO2, a default call returns the same 22
  channels as before; naming a Ti channel there refuses with
  `ImccGasSpeciesNotFoundError`. `IMCC_GAS_UNAVAILABLE_SPECIES` no
  longer lists Ti, TiO or TiO2. Every pre-existing channel's pressures,
  flags and provenance are unchanged bit for bit. The one TiO2 point in
  the Schaefer & Fegley (2004) reference (Allende B1 CAI, 2375 K) is now
  comparable, at +0.382 dex.
- A `parent_oxides` list that omits a parent no longer raises `KeyError`
  in `evaluate_gas`: a default call skips that parent's channels, and an
  explicitly requested one refuses with `ImccGasSpeciesNotFoundError`.

### Changes that can break callers

- Omitting the pack now loads the packaged SF04 pack: `--pack` is
  optional on the CLI, and `load_datapack()` and `evaluate(..., pack=None)`
  default to it. A command that used to exit with a usage error now
  solves.
- `result.labels.trust` is removed. Labels gain `flags`, `notices` and
  `acid_sink_ratio`, and the `trust` key is gone from `--json` output.
- `evaluate_gas` returns `ImccGasResult`, a read-only `Mapping`, not a
  `dict`. Indexing still works; item assignment, `isinstance(x, dict)`
  and `json.dumps(result)` do not, so use `dict(result)`. A non-positive
  fO2 raises `ImccGasInvalidFugacityError`, an `ImccRefusal`, where it
  used to raise `ValueError`.
- The solve transcript's `basis = ...` line is now `basis type = ...`
  followed by the mole total, labelled `(from wt%)` for wt% input.
- The X(Me2O) envelope admits values up to 0.5 × (1 + 1e-5), so a
  printed X = 0.5 composition converted from wt% is no longer refused.
- The bench JSON no longer carries `rmse`, `flagged_rmse` or
  `median_abs_residual` across slices, and its `per_population` and
  `per_species` groupings are replaced by `per_slice` and
  `binary_slices`.
- ext-v1 to ext-v3 are removed. The ext-v4 file is re-issued as
  `1.0.2-ext-sp-2`, so a pin of its previous file digest no longer
  matches; the SF04 core it carries is unchanged.
