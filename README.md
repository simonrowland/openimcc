# openimcc

**Ideal mixing of complex components (IMCC) for silicate melts** — activities,
activity coefficients, and degree of association, from an oxide composition and
a temperature.

> **Status: pre-release (0.1.0.dev0).** The solver and the benchmark harness are
> working and tested. Datapack provenance is documented but the release gate is
> not closed; see [Datapacks and provenance](#datapacks-and-provenance).

There is, as far as we can establish, no other open-source IMCC implementation.
Open melt-thermodynamics tools do exist: the MELTS family (via
ThermoEngine/alphaMELTS), VapoRock, and LavAtmos use different thermodynamic
models; commercial CALPHAD suites — FactSage, Thermo-Calc, MTDATA, and HSC
Chemistry — are another route. This package exists to fill that gap with
something auditable: every coefficient carries a provenance class, every
benchmark point carries its convention, and refusals are reported as data
rather than quietly dropped.

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

Python >= 3.11 is required. Until the first release is published, install from
a source checkout:

```bash
git clone https://github.com/simonrowland/openimcc
cd openimcc
python -m pip install -e ".[gas]"
# To run the empirical bench, also install:
python -m pip install -e ".[bench]"
```

The core install is `openimcc`; the `[gas]` extra adds the vapour layer used by
this quickstart. The optional `[bench]` extra adds the empirical benchmark
runner, and `[all]` installs both extras. The PyPI install becomes available at
release; until then, use the source-checkout commands above.

### Gas layer

`openimcc.gas` computes equilibrium partial pressures for the SF04 gas set.
The default tables ship in `openimcc.data.gas` and are loaded through
`importlib.resources`, so a release `pip install "openimcc[gas]"` works without a
neighbouring checkout. The gas Shomate rows are deterministic fits to vendored
NIST-JANAF 4th-edition records; the condensate rows retain their source-attributed
Lamoreaux/Hildenbrand and JANAF coefficients. Row-level source hashes, methods,
temperature ranges and fit residuals are in `PROVENANCE.yaml`.

For comparison with an existing VapoRock installation, set
`OPENIMCC_VAPOROCK_ROOT` explicitly; that variable overrides both packaged
tables. The packaged source records used to fit the gas rows are retained in
`data-src/janaf/` and included in source distributions, not the runtime wheel.
NIST SRD 13 is public data, and the publication attributions are recorded in
`NOTICE`.

## Use

### 30-minute quickstart

The shipped IMCC-SF04 pack is the default; no pack path is needed. This example
uses a basalt in weight percent at 1800 K. Parent-oxide activities are relative
to pure-liquid oxide standard states; for this model, `melt.activity(name)` is
the unbound `x*` fraction for that parent oxide, not a pressure.

```bash
openimcc describe
openimcc solve --temperature 1800 --basis-type wt --oxide SiO2=51.85068 --oxide MgO=4.78527 --oxide FeO=13.77307 --oxide CaO=9.02862 --oxide Al2O3=14.80572 --oxide TiO2=1.73824 --oxide Na2O=3.23108 --oxide K2O=0.78732
```

```python
from openimcc import evaluate, load_gas_datapack, evaluate_gas

basalt = {"SiO2": 51.85068, "MgO": 4.78527, "FeO": 13.77307, "CaO": 9.02862,
          "Al2O3": 14.80572, "TiO2": 1.73824, "Na2O": 3.23108, "K2O": 0.78732}
melt = evaluate(basalt, 1800.0, basis_type="wt")
activities = {name: melt.activity(name) for name in melt.parent_oxides}
gas = evaluate_gas(activities, 1800.0, 1e-10, load_gas_datapack())

print("activities:", {name: f"{value:.6g}" for name, value in activities.items()})
print("melt flags:", melt.labels.flags)
print("melt notices:", melt.labels.notices)
print("gas:", gas.unit, {"Na": f"{gas['Na']:.6g}", "K": f"{gas['K']:.6g}"})
print("Mg domain flag:", gas.domain_flags["Mg"])
```

Output:

```text
activities: {'SiO2': '0.329959', 'MgO': '0.00468755', 'FeO': '0.106157', 'CaO': '5.58726e-05', 'Al2O3': '0.0165703', 'TiO2': '0.00299186', 'Na2O': '2.47359e-10', 'K2O': '2.29657e-19'}
melt flags: ('paper-demonstrated-window: T=1800 K is outside the paper-demonstrated domain for rows: Mg2SiO4, MgSiO3, MgAl2O4, MgTiO3, MgTi2O5, Mg2TiO4, Al6Si2O13, CaAl2O4, CaAl4O7, Ca12Al14O33, CaSiO3, CaAl2Si2O8, CaMgSi2O6, Ca2MgSi2O7, Ca2Al2SiO7, CaTiO3, Ca2SiO4, CaTiSiO5, FeTiO3, Fe2SiO4, FeAl2O4, CaAl12O19, Mg2Al4Si5O18, Na2SiO3, Na2Si2O5, NaAlSiO4, NaAlSi3O8, NaAlO2, Na2TiO3, NaAlSi2O6, KAlSiO4, KAlSi3O8, KAlO2, KAlSi2O6',)
melt notices: ('Na and K activities from IMCC-SF04 are biased low against published anchors (SF04 Table 9 Na −1.4 dex; Hastie 1981 K −0.9 dex); see https://github.com/simonrowland/openimcc',)
gas: bar {'Na': '1.0369e-05', 'K': '7.67598e-08'}
Mg domain flag: T=1800.0 K outside declared G(T) interval for 'MgO(l)' [3100, 3500] K
```

The `paper-demonstrated-window` flag records that some complex rows are outside
their paper-demonstrated temperature range. The notice is a known low Na/K
activity bias against the cited anchors; it is part of the result, not a reason
to hide those activities. The gas `Mg` flag records extrapolation below the
declared MgO(l) thermodynamic row, so the pressure remains a prediction with a
visible limitation.

Here `fO2 = 1e-10` is bar-relative (`pO2 / 1 bar`), not `log10(fO2)` and not a
buffer offset such as ΔIW.

`evaluate_gas` predicts and flags out-of-domain temperatures by default; pass
`allow_extrapolation=False` to refuse them.

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

From a **source checkout** (the bench set is not included in the wheel),
`openimcc-bench` runs a tracked set of published measurements against a datapack
and prints per-point residuals. This is the part we would most like other people
to attack.

```bash
openimcc-bench benchmarks/sets/basalt-bench-set-v1.yaml src/openimcc/data/packs/imcc-sf04-v1.0.2.json
openimcc-bench benchmarks/sets/basalt-bench-set-v1.yaml src/openimcc/data/packs/imcc-sf04-v1.0.2.json --json
openimcc-bench benchmarks/sets/basalt-bench-set-v1.yaml src/openimcc/data/packs/imcc-sf04-v1.0.2.json --species SiO,Ca
openimcc-bench benchmarks/sets/basalt-bench-set-v1.yaml src/openimcc/data/packs/imcc-sf04-v1.0.2.json --populations kume2000_slag_si_alloy
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

The 402-point headline is a censored, mixed-standard-state number, not an
accuracy figure. The slices below keep their conventions visible; they are not
pooled into one claim.

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

### Published SF04 pressure comparison

The independent Schaefer & Fegley (2004) comparison is split into transcribed
Table 9 anchors and digitized Fig. 10 points. Residuals are
`log10(predicted / measured)`, in dex. O2 is the reference fO2 pin and is
excluded from agreement claims because the O2 channel returns that pin by
definition.

#### Table 9 anchors (transcribed)

| species | n | signed median | abs max |
|---|--:|--:|--:|
| Na | 1 | −1.416 | 1.416 |
| NaO | 1 | −1.615 | 1.615 |
| O | 1 | −0.007 | 0.007 |
| SiO | 1 | −0.009 | 0.009 |
| FeO | 1 | +0.009 | 0.009 |

The printed Table 9 Na anchor is −1.42 dex. Na, NaO and Na2 are low at every
comparable point: Na and NaO are about 1.1 dex low in aggregate, with medians
of −1.082 dex (n = 36) and −1.098 dex (n = 28); Na2 is −2.212 dex at its one
digitized point. O, SiO and FeO each match their own Table 9 anchor to about
0.01 dex (−0.007, −0.009 and +0.009), so this is not a unit or fO2-pin error. The Fig. 10 medians for SiO and FeO sit
about +0.2 dex above the transcribed anchor; that is a figure-versus-table
difference in the paper's digitized data, not a reconciled result.

#### Fig. 10 points (digitized)

| species | n | signed median | abs max |
|---|--:|--:|--:|
| Na | 35 | −1.079 | 1.364 |
| NaO | 27 | −1.096 | 1.827 |
| Na2 | 1 | −2.212 | 2.212 |
| O | 34 | +0.014 | 0.028 |
| SiO | 30 | +0.175 | 1.075 |
| FeO | 31 | +0.223 | 0.830 |

As a separate pressure-vs-pressure check, Plante 1979 KEMS K pressures agree
to median +0.09 dex (RMSE 0.20). The low-temperature drift is about +0.27 dex
at about 1250 K (n = 6), falling to about +0.02 dex at 1750 K; the K channel
uses the flagged K2O(l) secondary transcription. This is independent evidence
for the gas pressure path, not a replacement for the SF04 activity comparison.

#### Known limit: potassium in Ca- and Al-bearing melts

That Plante agreement does not cover basaltic melts. The Plante melts are
K2O–SiO2 binaries, and in melts with CaO and Al2O3 almost all of the K sits
in one SF04 complex, `KCaAlSi2O7` (SF04 Table 2, `log10 K = 4.30 + 17037/T`,
cited there to Hastie & Bonnell 1985). Bonnell & Hastie (1990, High Temp.
Sci. 26, 313–334) describe it as a model liquid with no known pure solid
(p. 316), introduced as an empirical correction to K-pressure predictions
across almost three decades of CaO/K2O in dolomitic limestones (p. 327).
Outside that calibration domain it can over-bind K.

A physical bound shows the size of the effect. Zhang et al. (2021, ACS Earth
Space Chem.) measured the product of the evaporation coefficient (their `γ`)
and the activity coefficient, `α·Γ(KO0.5)`, for a synthetic N-MORB-like
basalt: `6.9e-8` at 1473 K
and `1.11e-6` at 1673 K. Because `α ≤ 1`, `Γ(KO0.5)` cannot be lower than
those values. On the same composition openimcc gives `3.5e-9` and `2.4e-8`,
which is 1.3 and 1.7 dex below the floor. Both temperatures are below the
pack domain, so these are extrapolated evaluations. Na passes the same test,
with an implied `α` of about 0.05–0.10.

Treat K activities in Ca- and Al-bearing melts as a known low bias. The pack
is unchanged: the row is SF04's published value. The activity coefficients
printed in Zhang et al.'s Table 4 are MELTS model inputs, not measurements,
and are not used here.

### Independently pinned behaviour

The independently pinned contract is the analytic binary and limits in
`tests/test_kernel.py`, pure-silica D = 1 in conformance, atom balance, and
the JANAF gas-fit reproduction. Conformance goldens are a drift alarm at
RTOL 1e-9, not an external reference. None of these pins is a coefficient
retuning claim.

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
