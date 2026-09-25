# openimcc

**Ideal mixing of complex components (IMCC) for silicate melts** — activities,
activity coefficients, and degree of association, from an oxide composition and
a temperature.

> **Status: pre-release (0.1.0.dev0).** The solver and the benchmark harness are
> working and tested. Datapack provenance is documented but the release gate is
> not closed; see [Datapacks and provenance](#datapacks-and-provenance).

There is, as far as we can establish, no other open-source IMCC implementation.
Melt thermochemistry of this kind is otherwise the province of commercial
software — FactSage, Thermo-Calc, MTDATA, HSC Chemistry. This package exists to
fill that gap with something auditable: every coefficient carries a provenance
class, every benchmark point carries its convention, and refusals are reported
as data rather than quietly dropped.

## What it computes

IMCC treats a silicate melt as an **ideal solution of complex components**. The
non-ideality you observe in the oxide mixture is not fitted with interaction
parameters — it emerges entirely from *speciation*: oxides associate into
complexes (CaSiO3, MgSiO3, NaAlO2, …) and what is left unbound sets the
activity.

```
46 melt species = 8 unbound parent oxides + 38 complexes
x_j = K_j(T) · Π_i (x_i*)^ν_ij
log10 K_j = A_j + B_j / T          (T in K)
```

Parent oxides: `SiO2 MgO FeO CaO Al2O3 TiO2 Na2O K2O`.

Given a composition and T, `openimcc` solves for the free-oxide fractions `x*`,
and returns per-parent **activity** and **activity coefficient γ**, the full
species vector, and the **degree of association D** (total moles in, moles of
species out — D > 1 means an associated solution, D = 1 would be ideal oxide
mixing).

## What it is *not*

**This is not a phase-equilibrium code, and it is not a MELTS.** alphaMELTS,
MELTS and MAGEMin perform multiphase Gibbs minimisation: they return a phase
assemblage, a liquidus, fractionation paths. `openimcc` does homogeneous
speciation **inside a single liquid**. There is no phase assemblage to return
and no solid solution model. If you want to know what crystallises, you want one
of those; if you want the activity of Na2O in a melt you already know is liquid,
you want this.

It pairs with them rather than competing: activities from here feed a vapour
model, a volatility calculation, or an evaporation-flux term.

## Install

```bash
pip install openimcc                 # core solver
pip install "openimcc[bench]"        # + the empirical benchmark runner
pip install "openimcc[all]"          # + the vapour-species layer
```

Core is `numpy` + `scipy` only. The vapour layer (`openimcc.gas`) is the sole
consumer of `pandas` and nothing on the core path imports it, so the base
install stays small.

### Gas layer

`openimcc.gas` computes equilibrium partial pressures for the SF04 gas set.
The default tables ship in `openimcc.data.gas` and are loaded through
`importlib.resources`, so a fresh `pip install "openimcc[gas]"` works without a
neighbouring checkout. The gas Shomate rows are deterministic fits to vendored
NIST-JANAF 4th-edition records; the condensate rows retain their source-attributed
Lamoreaux/Hildenbrand and JANAF coefficients. Row-level source hashes, methods,
temperature ranges and fit residuals are in `PROVENANCE.yaml`.

For comparison with an existing VapoRock installation, opt in explicitly:

```bash
export OPENIMCC_VAPOROCK_ROOT=/path/to/VapoRock
```

When set, that variable overrides both packaged tables. The packaged source
records used to fit the gas rows are retained in `data-src/janaf/` and included
in source distributions, not the runtime wheel. NIST SRD 13 is public data,
and the publication attributions are recorded in `NOTICE`.

The benchmark runner's `partial_pressure` observable is wired to
`openimcc.gas`: it passes parent-oxide activities on the pure-liquid standard
state, converts bar to Pa, and retains out-of-domain predictions with a domain
flag. Missing or invalid inputs remain typed refusals.

## Use

```bash
# What is in a datapack?
openimcc describe --pack packs/imcc-sf04-v1.0.2.json

# Solve one composition (wt%), human-readable or JSON
openimcc solve --pack packs/imcc-sf04-v1.0.2.json --temperature 1800 \
    --basis-type wt \
    --oxide SiO2=45.4 --oxide MgO=8.1  --oxide FeO=10.9 --oxide CaO=11.4 \
    --oxide Al2O3=14.2 --oxide TiO2=3.2 --oxide Na2O=0.4 --oxide K2O=0.1
```

```python
from openimcc import load_datapack, evaluate

pack = load_datapack("packs/imcc-sf04-v1.0.2.json")
r = evaluate({"SiO2": 0.45, "MgO": 0.10, "CaO": 0.15,
              "Al2O3": 0.15, "FeO": 0.15}, 1800.0, pack)

r.D                 # degree of association, e.g. 1.9837
r.parent_activity   # a_i on the parent-oxide formula-unit basis
r.parent_gamma      # γ_i = a_i / x_i
r.labels.acid_sink_ratio  # x*(SiO2) / x(SiO2), the continuous edge diagnostic
```

### Exit codes are part of the contract

| code | meaning |
|-----:|---------|
| `0` | solved |
| `1` | usage error — you called it wrong |
| `2` | **typed refusal** — the model declined, with a machine-readable code |

A refusal is a *result*, not a crash. Out-of-domain temperature, a composition
outside the validated envelope, and a malformed pack each produce a distinct
code on exit 2. Pass `--allow-extrapolation` or `--allow-out-of-envelope` to
turn a refusal into an answer, and the answer is **flagged** — never silently
extrapolated.

## The empirical bench

`openimcc bench` runs a tracked set of published measurements against a datapack
and prints per-point residuals. This is the part we would most like other people
to attack.

```bash
openimcc-bench benchmarks/sets/basalt-bench-set-v1.yaml packs/imcc-sf04-v1.0.2.json
openimcc-bench ... --json                 # machine-readable BenchReport
openimcc-bench ... --species SiO,Ca       # filter
openimcc-bench ... --populations kume2000_slag_si_alloy
```

(`openimcc-bench` is a separate command from `openimcc` because it is the only
part that needs `pyyaml`; installing the core does not pull it in.)

### What is in the set

`basalt-bench-set-v1` — schema `melt-activity-bench.v1`, **402 points over 129
compositions**, T = 1173–2173 K, six independent populations:

| population | n | T (K) | technique |
|---|--:|---|---|
| `kume2000_slag_si_alloy` | 292 | 1823–1873 | CMAS slag / Si-alloy equilibration |
| `yamaguchi1983_emf_na2o_sio2` | 70 | 1173–1673 | EMF, Na2O–SiO2 binary |
| `tsaplin2000_kems_na2o_sio2` | 24 | 1173–1673 | Knudsen-cell mass spectrometry |
| `hastie1981_kems` | 6 | 1908–1956 | Knudsen-cell mass spectrometry |
| `richter_type_b_cai_gamma` | 6 | 1873–2173 | Type B CAI-like CMAS |
| `richter_type_b_cai_digitized_flux` | 4 | 1873–2173 | digitised evaporation flux |

Observables are `activity` (386), `partial_pressure` (6),
`activity_coefficient` (6) and `evaporation_flux` (4). Every point carries an
explicit `convention` string stating the basis its number is on, because a
residual computed across mismatched bases is wrong *and looks plausible*.

### Results, `imcc-sf04-v1.0.2`

Residual is **log10(predicted / measured)**, i.e. dex.

The runner reports each dataset × observable × standard-state slice. `n` counts
rows, signed median and mean are over residuals, and RMSE includes flagged
predictions in that slice. Refusals have no residual.

| dataset | observable | standard state | n | signed median | mean | RMSE |
|---|---|---|--:|--:|--:|--:|
| `hastie1981_kems` | `partial_pressure` | pure-liquid parent | 6 | −0.623 | −0.615 | 0.693 |
| `kume2000_slag_si_alloy` | `activity` | pure-solid | 292 | −0.526 | −0.481 | 0.890 |
| `richter_type_b_cai_digitized_flux` | `evaporation_flux` | unspecified | 4 | — | — | — |
| `richter_type_b_cai_gamma` | `activity_coefficient` | pure-liquid | 3 | +0.050 | +0.053 | 0.110 |
| `richter_type_b_cai_gamma` | `activity_coefficient` | CMAS basis | 3 | +0.543 | +0.554 | 0.577 |

Kume's 292 rows carry the explicit warning: **unconverted solid standard
state; not an accuracy figure for liquid activities**. No solid-to-liquid
conversion is applied; `condensate.csv` cannot provide the CaO and MgO
conversions.

#### Sodium binary: separate predict-and-flag block

The 54 Na2O rows and 40 SiO2 rows in Yamaguchi and Tsaplin state pure-liquid
parent-oxide activities in their own conventions. They are scored in this
separate block, never averaged into the table above or into another slice.
Flags remain visible: 94 predictions are out of the declared model domain,
split into **74 temperature-only** and **20 both-temperature-and-envelope**
flags; 12 Na2O rows also carry the source's per-composition extrapolation flag.

| dataset | parent activity | standard state | nominal X(Na2O) | n | signed median | mean | RMSE |
|---|---|---|---|--:|--:|--:|--:|
| `tsaplin2000_kems_na2o_sio2` | Na2O | pure-liquid | X ≤ 0.5 | 7 | +0.062 | −0.024 | 0.171 |
| `tsaplin2000_kems_na2o_sio2` | Na2O | pure-liquid | X > 0.5 | 5 | +4.196 | +4.216 | 4.230 |
| `tsaplin2000_kems_na2o_sio2` | SiO2 | pure-liquid | X ≤ 0.5 | 7 | −0.164 | −0.156 | 0.162 |
| `tsaplin2000_kems_na2o_sio2` | SiO2 | pure-liquid | X > 0.5 | 5 | −4.179 | −4.172 | 4.207 |
| `yamaguchi1983_emf_na2o_sio2` | Na2O | pure-liquid | X ≤ 0.5 | 36 | −0.316 | +0.030 | 0.975 |
| `yamaguchi1983_emf_na2o_sio2` | Na2O | pure-liquid | X > 0.5 | 6 | +3.801 | +3.877 | 3.940 |
| `yamaguchi1983_emf_na2o_sio2` | SiO2 | pure-liquid | X ≤ 0.5 | 24 | +0.016 | −0.344 | 0.909 |
| `yamaguchi1983_emf_na2o_sio2` | SiO2 | pure-liquid | X > 0.5 | 4 | −3.519 | −3.547 | 3.577 |

The nominal X = 0.5 Yamaguchi rows affected by the wt%-conversion fencepost
are `yamaguchi1983_a_sio2_liquid_x0500_1373`,
`yamaguchi1983_a_sio2_liquid_x0500_1473`,
`yamaguchi1983_a_sio2_liquid_x0500_1573`,
`yamaguchi1983_a_sio2_liquid_x0500_1673`,
`yamaguchi1983_a_na2o_x0500_1173`,
`yamaguchi1983_a_na2o_x0500_1273`,
`yamaguchi1983_a_na2o_x0500_1373`,
`yamaguchi1983_a_na2o_x0500_1473`,
`yamaguchi1983_a_na2o_x0500_1573`, and
`yamaguchi1983_a_na2o_x0500_1673`. Their converted X is 0.500003947168;
the model tolerance fix moves these rows from the envelope count to the
temperature-only count without changing their nominal X ≤ 0.5 sub-slice.

The old **0.883 dex** may be reproduced only as an **arithmetic total across
incompatible slices**, not as an accuracy claim: it is the 301 unflagged,
non-binary residuals (signed median −0.521 dex, mean −0.470 dex). The four
digitised flux rows remain typed refusals for *"OCR scatter digitization and no
independent experimental fO2 pin"*. No coefficient was tuned.

## Datapacks and provenance

Every row carries a `provenance_class`. For `imcc-sf04-v1.0.2` (38 rows):

| class | n | source |
|---|--:|---|
| `authority_by_publication` | 28 | Fegley & Cameron 1987, Table 3 |
| `janaf_direct_liquid` | 9 | NIST-JANAF (public domain); double as consistency checks |
| `partial_non_janaf_direct_liquid` | 1 | KAlO2, via Glushko/Gurvich as cited by FC87 Appendix 1 |

Extension packs add `extension-compound-thermo` rows (ours) and carry a
`screens` block recording candidates that were **considered and deliberately not
activated** — so a reader can see what was rejected, not just what was kept.

The shipped `imcc-sf04-ext-v4.json` S/P extension declares
`certification: denied`; it is screening-only and requires explicit opt-in.

`tools/packmanifest.py` maintains a SHA-256 per pack. A consumer pins
*(package version, manifest hash)*: a pack edited in place under an unchanged
version string would silently move every activity the engine reports, and the
manifest is what makes that fail loudly at CI time instead of quietly months
later.

### Sources

- **FC87** — Fegley B., Cameron A.G.W., *A vaporization model for iron/silicate
  fractionation in the Mercury protoplanet*, Earth Planet. Sci. Lett. **82**,
  207–222 (1987). [doi:10.1016/0012-821X(87)90196-8](https://doi.org/10.1016/0012-821X(87)90196-8)
- **SF04** — Schaefer L., Fegley B., *A thermodynamic model of high temperature
  lava vaporization on Io*, Icarus **169**, 216–241 (2004).
  [doi:10.1016/j.icarus.2003.08.023](https://doi.org/10.1016/j.icarus.2003.08.023)

The packs `imcc-sf04-v1.0.2.json` and `imcc-sf04-ext-v4.json` contain
`10.1016/j.icarus.2003.11.023` in their `sources.SF04.citation`. That DOI is wrong and must
not be cited; the correct DOI for Schaefer & Fegley (2004), Icarus 169, 216-241 is
`10.1016/j.icarus.2003.08.023`. The citation sits inside the published SF04 core, which both
packs carry byte-identically and the loader verifies by hash, so the core is deliberately left
unchanged: correcting it would change the published digest and
make every pin of the current identity unloadable.

## Author

Simon Rowland <simon@simonrowland.com>

## Licence

Code is Apache-2.0 (`LICENSE`). The project's own arrangement, provenance text
and derived tables are CC-BY-4.0 (`LICENSE-DATA`). NIST-JANAF records are NIST
public data. Published coefficients and digitized values are cited facts, not a
grant. Attribution is in `NOTICE`.
