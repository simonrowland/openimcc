# Changelog

## Unreleased

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
