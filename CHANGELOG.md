# Changelog

## Unreleased

- Added optional Ga, Ge, B and In gas channels with caller-supplied Ga2O3,
  GeO2, B2O3 and In2O3 parent activities. JANAF remains selected where it has
  tables, including the directly fitted B2O3 liquid branch; NASA CEA supplies
  the remaining gas and liquid functions. Labelled constant-Cp continuations
  extend the Ga2O3, GeO2 and In2O3 parents to 1200 K. Added Ga+, Ge+ and B+,
  Ga−, B−, BO− and BO2− to opt-in charge closure. Source Gibbs comparisons,
  row hashes, locators and fit residuals are recorded in the provenance
  ledger. Direct NASA parent-liquid fits pass the 10 J/mol source-node gate;
  continuation fit residuals are recorded separately.
  Activities remain caller-supplied, and default gas outputs are unchanged.
  The source-based ion screen passes for Ga and Ge; B exceeds the C3 threshold.
  The published NASA In+ card gives an anomalously low formation enthalpy and
  fails C3 unchanged; its source function is retained without adjustment.
- Added optional Cs, Cu and Sn gas channels with caller-supplied Cs2O(l),
  Cu2O(l) and SnO(l) activities, source-fitted 1200–3000 K rows, and opt-in
  Cs+, Cu+, Sn+, Cs2O+, Cs− and Cu− channels. JANAF liquid cells are used for
  Cu2O(l) through 2000 K, with a NASA H/S-increment tail; NASA liquid cards
  supply Cs2O(l) and SnO(l), with a labelled SnO constant-Cp continuation from
  1250 K to 1200 K. SnO is selected over the later-starting SnO2(l) card.
  Cs exceeds the recorded C3 ion-share limit at the caller activity screen;
  Cu and Sn remain below it. Default neutral results and existing coefficient
  rows are unchanged.
  The published NASA In+ card is declined after its fitted Kion/Saha ratios
  are 1.510e28 at 1500 K and 9.143e16 at 2500 K. The source card is retained
  without adjustment, and neutral In channels remain available.
- Switched default SiO2(l), Al2O3(l), MgO(l), and CaO(l) parent rows to the
  generated JANAF fits and labelled constant-Cp liquid continuations through
  1200 K. The former LAM1987 default rows, including SiO2(cr), are preserved
  in the opt-in `sf04-published` pack. Published-SF04 reference tests now use
  that pack. Retired `gas-janaf-parent-liquids-research` because it would
  duplicate the new default. The melt activities and FC87/SF04 complex log K
  coefficients are unchanged; the gas reaction code computes K from the
  active parent Gibbs row. The unchanged melt constants have not been jointly
  assessed against this particular JANAF parent dataset.
- Measured default SF04 comparison values and per-metal vapor-pressure shifts
  are recorded below under “Changes that can break callers.” They are report
  values and do not tighten the existing test gates.

- The default SiO2(l) parent now covers 1200–3000 K, but Si2(g) and Si3(g)
  still begin at 1500 K; Al2(g) also begins at 1500 K. Si and Al therefore
  remain C4 gas-partial. Mg and Ca now have C4 coverage and move to `complete`.
  SiO2(cr) has no gas-reaction consumer, but the public `species_thermo` solid
  phase query uses the row, so it remains in both default and compatibility
  tables. Gas reactions continue to use liquid SiO2. JANAF has K2O(cr) but no
  K2O(l) table, so the secondary LAM1984 K2O(l) row is unchanged.
- Added optional Li, Rb and Pb gas channels with caller-supplied Li2O(l),
  Rb2O(l) and PbO(l) activities. NIST-JANAF rows are preferred, with NASA
  Glenn cards used for Rb oxides and the PbO high-temperature tail. Li2O(l)
  is fitted on the printed JANAF liquid-branch cells from 700 K and served
  from 1200 K. Added Li+, Rb+, Pb+, Li-, LiO-, Rb- and Pb- to opt-in ion
  closure. Activities remain caller-supplied; openimcc has no
  trace-element activity model. Source disagreements, fits and coverage screens
  are recorded in provenance and the roadmap.
- Vendored the NIST-JANAF crystal tables Al-096 (corundum), Ca-027 (CaO) and
  O-035 (high cristobalite) as reference solids for converting oxide
  activities between solid and liquid standard states. They are not runtime
  inputs; a test checks the fusion Gibbs energies they give against the
  vendored liquid tables.
- Added generated, labelled constant-Cp supercooled-liquid intervals for the
  default JANAF-fitted TiO2(l), Cr2O3(l), and V2O3(l) rows. Added matching
  intervals for the JANAF-fitted MgO(l), CaO(l), Al2O3(l), and SiO2(l) rows
  initially in the opt-in research pack. The latest entry above moves those
  four parent rows into the default. Continuation use is visible in gas domain
  flags; the README and provenance record residuals, illustrative Cp
  sensitivity, and interval seams.
- Refit the default V2O3(l) high interval from the 1700–2300 K and 2500–3000 K
  liquid nodes only and extend its generated constant-Cp continuation through
  1700 K. The high interval is selected at the shared boundary; V gas results
  change over 1500–<1700 K and at fit-residual scale from 1700 K onward. The
  then-current research pack used these same generated V2O3(l) rows, so its V
  results changed too.
- Added opt-in JANAF-fitted Na+, K+, Ca+, electron and negative-ion gas
  channels, including O2−, AlO2−, KO−, Cr−, V− and Nb−, with melt-buffered
  electroneutrality. Neutral defaults remain unchanged. The completeness screen
  now uses fitted channels for Na, K and Ca; Na and K move to
  `complete-except-ions`, while Ca was then gas-partial because its parent row
  started at 2900 K. The former research pack included the default gas table
  and supported ion evaluation before it was retired above.
- Replaced the corrected LH84 Na2O(l) parent with the JANAF Na-013 fit. Added its supercooled-liquid interval, moved Na into the
  JANAF-fitted condensate and C4 gates, and re-pinned Na-sensitive SF04,
  quickstart, species-set, and oxygen-balance results. Recovered the intact
  1500 K JANAF thermal cells for both Na2O(l) fits and re-pinned Na-bearing
  outputs to the resulting node-anchored fit.
- Added fitted optional JANAF phosphorus and sulfur gas rows. Sulfur channels
  accept caller-supplied S2 fugacity. No public evaluated P2O5(l) G(T) function
  was found; JANAF lists P4O10(cr) only, so P channels need an external,
  source-rated P2O5(l) standard state. Default outputs stay fixed.
- Exported `species_thermo` for row-level Cp, S, apparent enthalpy and Gibbs
  energy, plus `default_gas_channels` for stable access to channel selection.
  Added JANAF in-interval fidelity gates, pinned the remaining LAM/JANAF
  parent-liquid source differences, and proved the prior Na2O(l) and Al2O3(l)
  rows fail their source gates.
- The interim Na2O(l) correction replaced a prior NaO2(l) row copy with LH84;
  the default now uses JANAF Na-013. Corrected
  Al2O3(l)'s anchor to the LH87 solid 298 K value, removing the fusion
  enthalpy that had been counted twice. Updated gas comparison results.
- Documented a known limit: complex rows below their demonstrated range, and their consistency with JANAF.
- Extended element completeness screening to 1200–3000 K, recomputed C2/C3
  maxima, and recorded C4 gaps in gas and parent-liquid coverage below 1500 K.
- Added 500–1500 K interval 2 rows for Na2O(g) and K2O(g), using LH84
  formation/entropy anchors with piecewise NASA Glenn heat-capacity functions;
  added a 1200–1500 K NbO2(l) interval while preserving its existing row.
- Recorded the JANAF glass-branch limits for TiO2(l) and V2O3(l). Their parent
  liquid intervals now include labelled constant-Cp continuations; Ti and V
  retain C4 gaps because their gas rows begin at 1500 K.
- Below every declared gas interval, extrapolate from the lowest interval rather than the first row in file order.
- Made Na2O and K2O gas channels optional when alternate tables lack their rows, and report each skipped channel.
- Added a second JANAF-fitted Shomate interval for the battery gas species over
  500–1500 K. It uses at least ten complete 100 K JANAF nodes per species,
  preserves the existing interval 1 rows, and selects interval 1 at 1500 K.
- Added opt-in research pack `gas-janaf-parent-liquids-research` with JANAF-fitted
  SiO2(l), Al2O3(l), MgO(l) and CaO(l) rows. At that time the LAM1987 rows
  remained the default; the first entry above records the later default switch
  and retirement of this pack.
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

- Default parent changes raise vapor pressures relative to the previous
  LAM1987 rows by these measured amounts. Values are log10 pressure shifts per
  metal atom; Si covers Si, SiO, SiO2, Si2, Si3; Al covers Al, AlO, AlO2, Al2O,
  Al2O2, Al2; Ca covers Ca/CaO; Mg covers Mg/MgO. A multi-metal molecule's
  total pressure shift is its per-atom value multiplied by its metal-atom
  count. These are comparison values, not regression gates.

  | T (K) | Si | Al | Ca | Mg |
  | ---: | ---: | ---: | ---: | ---: |
  | 1400 | +0.097374 | +2.772603 | +1.115757 | +0.635302 |
  | 1600 | +0.090826 | +1.403291 | +0.825530 | +0.422368 |
  | 1933 | +0.079586 | +0.312060 | +0.504033 | +0.199772 |
  | 2200 | +0.070464 | +0.045960 | +0.338890 | +0.099171 |
  | 2600 | +0.055522 | +0.000113 | +0.183019 | +0.025963 |

- Parent Gibbs differences and the remeasured default SF04 comparison are in
  the README and roadmap; published-reference tests gate only `sf04-published`.
- With `allow_extrapolation=False`, Cr, CrO, CrO2 and CrO3 now return at
  1500-1900 K using the labelled Cr2O3(l) continuation instead of raising
  `ImccGasTemperatureOutsideDomainError`. Ti and V strict calls below 1500 K
  still raise because their gas rows start there. With the default
  `allow_extrapolation=True`, temperatures below 1200 K now follow the new
  parent polynomials; at 1100 K, ΔG changes by -466 J/mol for TiO2, -1216
  J/mol for Cr2O3, and -393 J/mol for V2O3.
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
