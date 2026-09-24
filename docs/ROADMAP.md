# Roadmap

What is not done yet, and what was decided along the way. Everything here was
measured against the code rather than estimated.

## Known gaps

### 1. The gas layer is self-contained (done; one provenance gap remains)

The packaged gas tables now load by default through `importlib.resources`.
Twenty-two Shomate rows are fitted deterministically from vendored NIST-JANAF
records, with the exact source hash and fit residual recorded per row. The
maximum fit residual is **1.9231 J/mol** (**3.7465e-5 log10 K**), well below the
0.01 log10 K gate. The eight condensate rows reproduce the current published
coefficient values exactly. `OPENIMCC_VAPOROCK_ROOT` remains an explicit
comparison override.

Known gap: the shipped K2O(l) coefficients are a secondary transcription from
the VapoRock condensate CSV, cited to Lamoreaux & Hildenbrand 1984, and are
flagged `secondary_transcription_unverified_primary`. The K, K2 and KO channel
authority therefore carries that flag. Once the LH84 primary coefficient table
is available, re-certify the row and remove the flag.

### 2. The bench's `partial_pressure` observable is wired (done)

The runner now passes each point's parent-oxide activities, temperature and
independent fO2 pin to `openimcc.gas.evaluate_gas`, converts the returned bar
value to the bench's Pa units, and carries the gas channel's provenance class
into each result row. The datapack loads once per run. The tracked set with
`imcc-sf04-v1.0.2` produces these six Hastie 1981 KEMS outcomes:

| id | T (K) | fO2 (bar) | measured (Pa) | predicted (Pa) | log10 residual | domain flag | provenance class |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `hastie_k_1917_186` | 1917.186331 | 5.694375450e-6 | 2.820005790 | 0.361935171 | −0.891619212 | — | `secondary_transcription_unverified_primary` |
| `hastie_k_1917_282` | 1917.281980 | 5.703622101e-6 | 3.210036717 | 0.362238799 | −0.947515035 | — | `secondary_transcription_unverified_primary` |
| `hastie_k_1955_825` | 1955.824724 | 1.082207471e-5 | 4.530018754 | 0.504424868 | −0.953303511 | — | `secondary_transcription_unverified_primary` |
| `hastie_sio_1907_796` | 1907.795668 | 4.851884054e-6 | 0.082727519 | 0.049889861 | −0.219637706 | `SiO2(l) [1996, 3000] K` | `lam1987_transcribed` |
| `hastie_sio_1909_739` | 1909.739189 | 5.016024124e-6 | 0.116737396 | 0.051569589 | −0.354816329 | `SiO2(l) [1996, 3000] K` | `lam1987_transcribed` |
| `hastie_sio_1948_149` | 1948.149056 | 9.545997137e-6 | 0.206709293 | 0.097839724 | −0.324844781 | `SiO2(l) [1996, 3000] K` | `lam1987_transcribed` |

The three SiO points are predicted with the bench's per-point extrapolation
opt-in and carry the SiO2(l) certified band in `domain_flag`. The fresh full-set
summary is 304 predictions, 301 headline-scored, 3 flagged, 94 out-of-domain
and 4 refused; headline RMSE **0.883362 dex**, flagged RMSE **0.305319 dex**.
The 396 non-gas rows retain their pre-rewire predictions, residuals, statuses
and reasons exactly.

Known gap: **K partial pressure is ~0.9 dex low on Hastie 1981 case 4; the
reviewer's decomposition attributes −0.845 dex to the IMCC a_K2O (free-parent
activity, gamma ~ 4e-13) against VapoRock on the same K2O(l) row, not to the
K2O(l) transcription (a JANAF crystal swap moves it −0.010 dex).**

Short decomposition:

| source | dex |
| --- | ---: |
| IMCC a_K2O versus VapoRock at the same T, pin and K2O(l) row | −0.845 |
| remaining VapoRock-versus-measurement residual | −0.047 |

Upstream this called the simulator's analytical vapour stack, which reads a
large catalogue through two further subsystems. **That is not worth porting.**
`openimcc.gas.evaluate_gas()` already covers every species the bench needs —
all six of SiO, Ca, Al, Na, Mg, K are in `IMCC_GAS_CHANNEL_SPECIES` — and takes
parent-oxide activities, T and fO2, which is exactly what the bench holds. So
this is a rewire, not a transplant.

> ★ **Unit evidence for the completed rewire.** `evaluate_gas` returns **bar**;
> every `measured` value on a `partial_pressure` point is in **Pa**. The runner
> asserts `units: Pa` and applies the single named `1 bar = 1e5 Pa` conversion.
> A regression test compares a direct gas result against the Pa prediction, so
> dropping or doubling the factor fails instead of silently shifting the log
> residual by five orders of magnitude.

### 3. Test coverage (done)

Measured on the current tree: **93%** overall (1445 statements). `kernel.py`
99%, `model.py` 99%, `gas.py` 90%, `cli.py` 86%, `bench.py` 85%. The ≥85% target
on kernel and model is met. Line coverage shows the physics executes, not that
a test notices when it changes; the published-reference and conformance tests
are what pin the numbers.

### 4. The MAGMA workbook regression cannot be distributed

`tests/test_rung3_fixture.py` compares against
`tests/fixtures/imcc_sf04_magma_workbook.csv`, a 327 KB derivative of
`Schaefer2004-MAGMA-valid.xlsx` from the VapoRock data directory. Same AGPL
problem as item 1, plus the workbook is MAGMA output whose own redistribution
terms are unestablished. The test skips when the file is absent.

An independent, public partial reference now ships at
`benchmarks/references/schaefer-fegley-2004/`: 5 Table 5 composition rows
(`transcribed`), 13 Table 9 flux rows (`transcribed`) with 13 Eq. 11 pressures
(`derived_eq11`), and 350 Fig. 10 points (`digitized_figure`) with 350 Table 7
pressure derivations (`derived_table7`). It is covered by
`tests/test_sf04_published_reference.py` and does **not** replace the workbook
fixture. Bishop Tuff, Type B CAI, the Al/Ca/Ti/K2/Zn channels, and 1500, 1625,
and 2500 K remain unavailable from the published paper.

The published-reference comparison also records a model difference for sodium:
openimcc's Na, NaO, and Na2 are low against SF04 at every comparable point. Na
has median residual −1.08 dex and residual −1.42 dex at the printed Table 9
anchor. O, SiO, and FeO match the anchor on the roughly 0.02-dex scale, so the
sodium result is not a pin or unit error. It is left for later diagnosis; the
reference work does not tune the model.

### 5. Provenance vocabulary is not uniform across packs

`v1.0.2` and `ext-v4` distinguish `authority_by_publication` (28),
`janaf_direct_liquid` (9) and `partial_non_janaf_direct_liquid` (1).
`ext-v1`, `ext-v2` and `ext-v3` collapse 30 rows into a coarser
`published-imcc`. Less detail, not worse provenance — but a reader cannot tell
which of those rows are FC87-table-derived and which are JANAF-derived.

An audit of all five packs found **zero reverse-engineered rows**.

### 6. Documentation

- `docs/datapack-format.md` documents the pack schema, what the loader and
  kernel refuse, and in what order (done).
- `docs/spec.md` — the model spec has not been ported.
- `CONTRIBUTING.md`, in particular how to propose a **datapack row**: the
  unusual contribution type here, and the one needing a provenance rule.
- `CHANGELOG.md`.

### 7. Identity

`CITATION.cff` has no DOI and no ORCID. Zenodo wiring would give a citable DOI.
No PyPI publish workflow yet.

### 8. Validation decks

`docs/validation/md-decks/` is empty.

## Running the checks

```bash
scripts/ci-local.sh            # all gates, across every interpreter uv can supply
scripts/ci-local.sh --quick    # current interpreter only
```

Three gates, and two of them are things `pytest` alone cannot catch:

- **datapack drift** — a pack edited under an unchanged version string moves
  every activity the engine reports;
- **test matrix** — across 3.11 through 3.14;
- **core-only install** — `pip install openimcc` must need numpy and scipy
  alone. A developer machine has the extras installed, so one convenience
  import in a core module breaks users while every local test still passes.
  The gate refuses to report success if pandas or pyyaml turn out to be
  present, because then it would be proving nothing.

`.github/workflows/ci.yml` runs the same set. The two are mirrors and must be
changed together.

## Decisions taken

- **Apache-2.0 for code, CC-BY-4.0 for data.** Split because the halves have
  different origins; `NOTICE` carries the attribution chain.
- **`adapter.py` became `model.py`.** "Adapter" only ever meant "adapter to the
  host simulator's backend ABC"; standalone it is the public API.
- **The `MeltBackend` glue did not travel.** It is host policy and stays
  upstream, importing this package rather than the reverse.
- **Gas is exported lazily** through a module `__getattr__`. Eager import would
  put pandas on the core path and make the dependency tiering — and the CI job
  guarding it — fiction.
- **`ImccRefusal` is exported as the catchable base**, so `except ImccRefusal`
  works and the exit-2 contract is usable from Python, not just a shell.
- **Two console scripts, not one.** `openimcc-bench` is separate because
  `bench.py` carries the only pyyaml dependency.

## Traps fixed during extraction, recorded so they are not reintroduced

- **`Path(__file__).resolve().parents[3]`** anchored pack paths on a repository
  root. Correct in the source tree it was written for; from `src/openimcc/` it
  lands on the repo's parent, and from site-packages inside the Python
  installation. Neither raises — it yields a plausible path holding no data. Now
  `importlib.resources`.
- **`gas.py` searched the filesystem** for a VapoRock checkout (a sibling
  directory, a vendored copy, two paths under `$HOME`). That binds results to
  whatever happens to sit next to the checkout, so two users can get different
  numbers with no diagnostic. Now configuration only.
- **The bench resolved relative paths against a derived repo root.** Installed,
  that produced a path inside site-packages' parent instead of reporting the
  file missing. Now resolves against the CWD.
