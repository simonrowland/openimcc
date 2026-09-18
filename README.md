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

### Gas layer: not yet self-contained

`openimcc.gas` computes equilibrium partial pressures for the SF04 gas set, and
the code is here — but **its JANAF tables are not bundled**. They live in
[VapoRock](https://gitlab.com/ENKI-portal/vaporock), which is AGPL-3.0, and this
package is Apache-2.0, so redistributing them here is not clearly permitted.

Point it at a checkout to use it:

```bash
export OPENIMCC_VAPOROCK_ROOT=/path/to/VapoRock
```

Without that, `load_gas_datapack()` raises a typed
`ImccGasDataUnavailableError` naming the variable, and the gas-dependent tests
skip rather than fail. The underlying values are NIST-JANAF (public domain), so
the intended fix is to regenerate our own tables — roughly 6 KB for the ~22
species in `IMCC_GAS_CHANNEL_SPECIES` — after which this step goes away.

The benchmark runner's `partial_pressure` observable is a separate gap: it is
awaiting a rewire onto `openimcc.gas` and currently returns a typed refusal.

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

```
402 points   298 scored   94 out-of-domain   10 refused
RMSE 0.883 dex   median |residual| 0.577 dex
```

| species | n | scored | RMSE (dex) | median abs (dex) |
|---|--:|--:|--:|--:|
| SiO | 132 | 89 | **0.452** | 0.319 |
| Ca | 99 | 99 | 0.834 | 0.615 |
| Mg | 52 | 48 | 1.064 | 0.981 |
| Al | 62 | 62 | **1.211** | 0.568 |
| K | 3 | **0** | — | — |
| Na | 54 | **0** | — | — |

### Read this before quoting the number

**98 of 402 points do not score, and that is the most informative part of the
output.** Dropping them would improve nothing and would flatter the RMSE:

- **All 94 out-of-domain points are Na2O–SiO2 binaries.** The pack declares a
  domain of 1700–3000 K for every row; Yamaguchi 1983 and Tsaplin 2000 sit at
  1173–1673 K, entirely below the floor. So `Na` scores **0 of 54** — we hold
  Na-silicate activity data and the published coefficients do not reach it. That
  is a real coverage gap, not a rounding error.
- **Of the 10 refused, 4** are the digitised flux set: *"OCR scatter
  digitization and no independent experimental fO2 pin"* — a data-quality
  refusal, not an engine failure. **The other 6** are every `partial_pressure`
  point (3 SiO, 3 K): that observable is not wired in this package yet, so K
  scores 0 of 3. See [Gas layer](#gas-layer-not-yet-self-contained).

**And the scored points are themselves an extrapolation.** 34 of the 38 rows
carry a paper-demonstrated domain of **2500–3500 K** (Fegley & Cameron computed
vaporization at 2500, 3000 and 3500 K). The whole bench sits 600–1300 K *below*
that. So `RMSE 0.879 dex` is not "the model is off by a factor of 7.6" so much
as "these coefficients, used several hundred kelvin below where their authors
demonstrated them, hold to about half a dex in the median." Al is the worst
(RMSE 1.211 with median 0.568 — heavy tails, a few points far out), SiO the
best.

We would rather publish that honestly than tune it.

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
  [doi:10.1016/j.icarus.2003.11.023](https://doi.org/10.1016/j.icarus.2003.11.023)

## Author

Simon Rowland <simon@simonrowland.com>

## Licence

Code Apache-2.0 (`LICENSE`); datapacks, benchmark sets and validation decks
CC-BY-4.0 (`LICENSE-DATA`); attribution in `NOTICE`.
