# Roadmap

What is not done yet, and what was decided along the way. Everything here was
measured against the code rather than estimated.

## Known gaps

### 1. The gas layer is not self-contained (licence)

`openimcc.gas` works, but its JANAF tables are not bundled: they come from
[VapoRock](https://gitlab.com/ENKI-portal/vaporock), which is **AGPL-3.0**,
while this package is Apache-2.0. Redistributing its data files here is not
clearly permitted — the underlying values are NIST-JANAF public domain, but
VapoRock's compilation of them is covered by its licence.

Today: set `OPENIMCC_VAPOROCK_ROOT`, or pass `gas_path=` / `oxide_path=`
explicitly. Unset, `load_gas_datapack()` raises a typed
`ImccGasDataUnavailableError` and the gas tests skip.

**Fix:** regenerate the tables from NIST-JANAF directly under our own
provenance. Roughly 6 KB covering the ~22 species in
`IMCC_GAS_CHANNEL_SPECIES`. This removes the AGPL contact entirely and makes
`pip install "openimcc[gas]"` work out of the box.

### 2. The bench's `partial_pressure` observable is not wired

It returns a typed refusal, so 6 of 402 points (3 SiO, 3 K) do not score and K
scores 0 of 3.

Upstream this called the simulator's analytical vapour stack, which reads a
large catalogue through two further subsystems. **That is not worth porting.**
`openimcc.gas.evaluate_gas()` already covers every species the bench needs —
all six of SiO, Ca, Al, Na, Mg, K are in `IMCC_GAS_CHANNEL_SPECIES` — and takes
parent-oxide activities, T and fO2, which is exactly what the bench holds. So
this is a rewire, not a transplant.

> ★ **Whoever does the rewire: assert the units first.** `evaluate_gas` returns
> **bar**; every `measured` value on a `partial_pressure` point is in **Pa**.
> That is exactly 5 dex. Measured on one CMAS state at 1873 K, raw
> `log10(gas/simulator)` was −4.58 (SiO), −4.99 (Ca), −5.32 (Al), −5.75 (Mg);
> corrected for the 1e5, +0.42 / +0.01 / −0.32 / −0.75, i.e. agreement to about
> a dex. Miss the conversion and every gas residual shifts five orders of
> magnitude while the log-residual table still looks entirely plausible.

### 3. Test coverage

Measured on the pre-extraction package: **67.9%** overall (1354 statements).
`kernel.py` 78.9%, `model.py` 81.6%, `cli.py` 60.8%, `bench.py` 54.2%,
`gas.py` 50.6%. `kernel.py` is the physics and matters most — 21% of the solver
is unexercised, with untested runs around lines 661–666, 715–723, 730–736 and
984–989. Target ≥85% on kernel and model before 0.1.0 final.

### 4. The MAGMA workbook regression cannot be distributed

`tests/test_rung3_fixture.py` compares against
`tests/fixtures/imcc_sf04_magma_workbook.csv`, a 327 KB derivative of
`Schaefer2004-MAGMA-valid.xlsx` from the VapoRock data directory. Same AGPL
problem as item 1, plus the workbook is MAGMA output whose own redistribution
terms are unestablished. The test skips when the file is absent. Replacing it
with an independently sourced reference set would restore the regression for
everyone.

### 5. Provenance vocabulary is not uniform across packs

`v1.0.2` and `ext-v4` distinguish `authority_by_publication` (28),
`janaf_direct_liquid` (9) and `partial_non_janaf_direct_liquid` (1).
`ext-v1`, `ext-v2` and `ext-v3` collapse 30 rows into a coarser
`published-imcc`. Less detail, not worse provenance — but a reader cannot tell
which of those rows are FC87-table-derived and which are JANAF-derived.

An audit of all five packs found **zero reverse-engineered rows**.

### 6. Documentation

- **`docs/datapack-format.md` does not exist.** The pack schema is documented
  nowhere, so a third party cannot author or audit a pack — which undercuts the
  auditability this package claims. Highest-value doc gap.
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
