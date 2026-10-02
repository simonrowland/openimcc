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

`openimcc.gas` computes equilibrium partial pressures for the SF04 gas set
plus the titanium channels, screened Al/Si association channels, Cr channels,
and caller-supplied V/Nb/P/S/Li/Rb/Pb channels:
Na, Na2, NaO, K, K2, KO, Si, SiO,
SiO2, Fe, FeO, Mg, MgO, Al, AlO, AlO2, Al2O, Al2O2, Ca, CaO, Ti, TiO, TiO2,
Al2, Si2, Si3, Cr, CrO, CrO2, CrO3, V, VO, VO2, Nb, NbO, NbO2, P, P2, P4,
PO, PO2, P4O6, P4O10, S, S2, S3, S4, S5, S6, S7, S8, SO, SO2, SO3, SSO,
Li, LiO, Li2O, Li2O2, Rb, RbO, Rb2O, Pb, PbO, PbO2, Na2O, K2O, O and O2.
The Mn/Ni/Co channels activate only when an external pack provides their gas
and MnO(l)/NiO(l)/CoO(l) parent rows; the public pack carries only the atomic
gas rows.
The sulfur channels activate only when the caller supplies `S2` fugacity as
`a(S2) = f(S2)/p°` on JANAF's 1-bar `S2(g)` reference. OpenIMCC does not
provide a sulfur melt-side activity model for pyrolysis temperatures.

The P gas tables are fitted, but no public evaluated `P2O5(l)` G(T) function
was found; JANAF lists `P4O10(cr)` only. P channels therefore remain unavailable
until an external pack supplies a source-rated `P2O5(l)` standard state and the
caller supplies `a(P2O5)` on that reference. OpenIMCC does not provide a
melt-side P2O5 activity model for pyrolysis temperatures.

Li, Rb, and Pb channels activate when the caller supplies `Li2O`, `Rb2O`, or
`PbO` activity in the `evaluate_gas` mapping. Activities are caller-supplied;
openimcc has no trace-element activity model. The Li2O(l) and PbO(l) parents
use NIST-JANAF rows; Rb2O(l) uses a NASA Glenn CEA coefficient card, whose
published pure-liquid reference is 1 atm. PbO(l) is the Pb(II) parent because
it has a source-rated liquid row; PbO2 remains a gas channel because no
evaluated PbO2(l) parent row was found. NASA CEA gas cards use the same 1 bar
standard pressure as JANAF. The printed NASA liquid-card formation enthalpies
are crystal anchors because NG-1643 and NG-1801 start at 1726 K and 1160 K.
The source comparison therefore uses liquid Gibbs energy, computed as NASA
polynomial minus JANAF `dfH298 + (H-H298) - T·S`:

| T (K) | Li2O ΔG (kJ/mol) | PbO ΔG (kJ/mol) |
|---:|---:|---:|
| 1800 | −4.098 | −0.002 |
| 2000 | −3.462 | −0.016 |
| 2500 | −2.171 | −0.048 |
| 3000 | −1.242 | −0.079 |

Li2O also differs in liquid Cp (103.999 versus 100.416 J/(mol·K)) and entropy
(ΔS −3.37 to −1.54 J/(mol·K)); JANAF remains selected. PbO extrapolated NASA
liquid H(298.15 K) is −202.139 kJ/mol versus JANAF −202.249 kJ/mol, while ΔH
over the overlap is about 0.11 kJ/mol and ΔS about 0.063 J/(mol·K). JANAF
remains selected. O-007 ends at 2500 K, so the required PbO tail anchors
ΔH = ΔS = ΔG = 0 there; against continuing JANAF Cp = 65 J/(mol·K), its
ΔG at 3000 K is 0.017 J/mol.

Published composition-specific activity-coefficient work is relevant context:
[Borisov (2009)](https://doi.org/10.1134/S0869591109060058) reports alkali
oxide behavior in silicate melts, with Li estimates extrapolated from the
experiments, and [Wood and Wade (2013)](https://doi.org/10.1007/s00410-013-0896-z)
infer PbO activity from metal-silicate partitioning. Those studies' coefficients
are not runtime inputs.

The default tables ship in `openimcc.data.gas` and are loaded through
`importlib.resources`, so a release `pip install "openimcc[gas]"` works without a
neighbouring checkout. The gas Shomate rows are deterministic fits to vendored
NIST-JANAF 4th-edition records and NASA Glenn CEA coefficient cards; the
condensate rows retain their source-attributed Lamoreaux/Hildenbrand, JANAF,
and NASA coefficients. TiO2(l), Cr2O3(l), V2O3(l), NbO2(l), Na2O(l),
Li2O(l), and PbO(l) are fitted from JANAF liquid tables by the same generator;
Rb2O(l) uses its NASA CEA liquid card. The default table adds explicitly
labelled constant-Cp continuations for TiO2(l), Cr2O3(l), and V2O3(l),
down to 1200 K. Li2O(l) uses JANAF's complete liquid-branch cells from
700–3000 K directly; its runtime fit intervals begin at 1200 K. The JANAF-fitted major-oxide
parents SiO2(l), Al2O3(l), MgO(l), and CaO(l) remain in the opt-in
`gas-janaf-parent-liquids-research` pack; that pack has matching generated
continuation intervals down to 1200 K. These rows are generated
extrapolations, not source data. When selected they add a constant-Cp
supercooled-liquid notice to each affected gas channel's `domain_flags`.
Row-level source hashes, methods, temperature ranges and fit residuals are in
the corresponding provenance files.

Thermal ions are opt-in: `evaluate_gas(..., include_ions=True)` solves charge
balance on the default neutral gas set, then appends `Na+`, `K+`, `Ca+`, `e-`,
`Li+`, `Rb+`, `Pb+`, and every available JANAF-supported negative ion whose
neutral is present. The default call remains neutral-only and returns the same
mapping as before. Ion entries use provenance class
`janaf_fitted_ionisation`; their source rows cover 1200–3000 K and their
domain flags include the neutral and charge-species source rows.
`gas_species` requests use the full default neutral set for the charge closure,
then retain the requested neutral and charge channels. Requesting an ion name
requires `include_ions=True`. `evaluate_gas_oxygen_balance` remains neutral-only.

JANAF gives neutral gases, ions and electrons the same ideal-gas standard
state, `p° = 0.1 MPa = 1 bar`. Its electron table uses a monatomic ideal-gas
reference and tabulates `H°(T)-H°(0)`; ionic formation functions use JANAF's
elemental reference states. The fitted rows use the printed ion tables with the
electron row; they do not apply the +6.197 kJ/mol adjustment some ion-table
notes specify when converting to a convention that excludes the electron. For
`M(g) = M+(g) + e-(g)`,
`K_M = exp[-(G(M+) + G(e-) - G(M))/(R T)]`. At 1500 K, the fitted Na
equilibrium constant is 1.58211e-16 and matches JANAF log Kf = -15.801.
Ground-state Saha, with g(Na+)/g(Na) = 1/2 and chi = 5.13907696 eV, gives
1.57265e-16. The 0.60% difference comes from the JANAF tabulated
thermochemistry, not our fit residual; Na's first excited level at 2.10 eV
contributes less than 3e-7 to its partition function at 1500 K, using the
[NIST Atomic Spectra Database](https://physics.nist.gov/PhysRefData/Handbook/Tables/sodiumtable5_a.htm).
The derivation and unit check are in `openimcc.gas`; the JANAF convention is
described in the
[NIST-JANAF introduction](https://janaf.nist.gov/pdf/JANAF-FourthEd-1998-1Vol1-Intro.pdf).

The charge closure keeps the melt-buffered neutral pressures fixed. For each
available JANAF-supported negative ion `A-` whose neutral is present,
`K_A = p(A-)/(p(A) p(e-))`; electroneutrality
then gives `p(e-) = sqrt(sum(K_M p(M)) / (1 + sum(K_A p(A))))`. The included
negative species include Na−, K−, O−, O2−, Al−, AlO−, AlO2−, Fe−, Si−,
Ti−, KO−, NaO−, Cr−, V− and Nb−. On the C3 screen (README basalt plus
caller-supplied Cr2O3, V2O3 and NbO2 activities of 1e-3; 1200–3000 K,
fO2 = 1e-12–1e-4), O2− has a maximum attachment term `K_A p(O2)` of
4.14e-4 at 1200 K and fO2 = 1e-4; AlO2− reaches p(AlO2−)/p(AlO2) = 6.46,
and KO− reaches p(KO−)/p(KO) = 1.48e-3. Cr−, V− and Nb− have maximum
charge-balance terms `K_A p(A)` of 8.53e2, 0.385 and 8.91e-4, respectively.
All these species are included in the fitted charge balance regardless of
their individual attachment-term size. The KO− source is
[JANAF table K-009](https://janaf.nist.gov/tables/K-009.html); its supplied
thermochemistry is used as published.

For the README basalt at fO2 = 1e-10, the fitted atomic-ion ratios and electron
pressure are:

| T (K) | p(Na+)/p(Na) | p(K+)/p(K) | p(Ca+)/p(Ca) | p(e−) (bar) |
|---:|---:|---:|---:|---:|
| 1500 | 1.26401e-5 | 6.04931e-3 | 2.68168e-8 | 1.25166e-11 |
| 2000 | 4.60159e-5 | 4.70024e-3 | 6.42409e-7 | 1.46320e-7 |

Rows at 2500 K and 3000 K are omitted because their summed neutral pressures
are 7.65 bar and 2.50e6 bar, above the 1-bar ideal-gas validity ceiling.
Opt-in ion results above that ceiling carry a domain flag; they are returned
for inspection, but the ideal-gas charge closure is not credible there.

For comparison with an existing VapoRock installation, set
`OPENIMCC_VAPOROCK_ROOT` explicitly; that variable overrides both packaged
tables. The Ti, Cr, V and Nb channels appear only when available: VapoRock's
condensate table has no TiO2(l), Cr2O3(l), V2O3(l) or NbO2(l) rows, so under the
override a default call returns the other channels. Caller-supplied Cr2O3,
V2O3 or NbO2 activities opt into their corresponding channels.
The Cr(g) fit uses complete JANAF Cr-005 rows through 2900 K; evaluation above
that declared endpoint is flagged or refused according to the caller's
extrapolation setting. The fitted Cr2O3(l) source interval starts at 1900 K;
the labelled constant-Cp continuation covers 1200–1900 K. Cr channels carry its
continuation notice in that interval, and Cr(g) still limits complete C4
coverage to 2900 K.
The packaged source records used to fit the gas rows are retained in
`data-src/janaf/` and included in source distributions, not the runtime wheel.
NIST SRD 13 is public data, and the publication attributions are recorded in
`NOTICE`.

The Na2O(g) and K2O(g) rows are a provisional extension. Their temperature
dependence uses the 1000–6000 K NASA Glenn/Gurvich cards; their formation
enthalpy and entropy anchors use Table 4 of [Lamoreaux and Hildenbrand
(1984)](https://srd.nist.gov/JPCRD/jpcrd241.pdf). The NASA and LH84 formation
enthalpies differ by about 15 kJ/mol, so `PROVENANCE.yaml` records the
`lh84_evaluated_with_nasa_cp` certification item and marks both rows as pending
KEMS certification.

For the README basalt above, the new-channel magnitude is:

| T (K) | fO2 | p(Na2O)/p(Na) | p(K2O)/p(K) |
|---:|---:|---:|---:|
| 1800 | 1e-10 | 3.10041e-11 | 2.70081e-13 |
| 1800 | 1e-6 | 3.10041e-10 | 2.70081e-12 |
| 2200 | 1e-10 | 4.41418e-11 | 2.01182e-13 |
| 2200 | 1e-6 | 4.41418e-10 | 2.01182e-12 |
| 2600 | 1e-10 | 5.31024e-11 | 1.53256e-13 |
| 2600 | 1e-6 | 5.31024e-10 | 1.53256e-12 |

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
melt notices: ('K predictions from IMCC-SF04 remain low against Hastie 1981 KEMS pressures (case 4: −0.89 dex); see https://github.com/simonrowland/openimcc',)
gas: bar {'Na': '0.000201444', 'K': '7.67598e-08'}
Mg domain flag: T=1800.0 K outside declared G(T) interval for 'MgO(l)' [3100, 3500] K
```

The `paper-demonstrated-window` flag records that some complex rows are outside
their paper-demonstrated temperature range. The notice records the known low K
prediction against Hastie 1981 KEMS pressures; it is part of the result, not a
reason to hide those activities. The gas `Mg` flag records extrapolation beyond
the declared default MgO(l) interval, so callers can see that the prediction uses
the existing Lamoreaux function outside its fitted domain.

With the basalt's activities evaluated at each temperature (extrapolated and
flagged outside the melt pack's domain) and `fO2 = 1e-10`, the sodium gas
pressures are:

| T (K) | p(Na) (bar) | p(NaO) (bar) | p(Na2) (bar) | p(Na2O) (bar) |
|---:|---:|---:|---:|---:|
| 1800 | 2.01453e-4 | 3.79925e-10 | 3.36639e-10 | 1.22956e-13 |
| 2200 | 2.77842e-2 | 3.91481e-8 | 2.43405e-6 | 1.23175e-10 |
| 2600 | 7.94594e-1 | 9.12685e-7 | 9.86681e-4 | 1.31905e-8 |

Here `fO2 = 1e-10` is bar-relative (`pO2 / 1 bar`), not `log10(fO2)` and not a
buffer offset such as ΔIW.

`evaluate_gas` predicts and flags out-of-domain temperatures by default; pass
`allow_extrapolation=False` to refuse them.

### Oxygen-balance (Knudsen) effusion

`evaluate_gas_oxygen_balance(activities, T, datapack)` solves the oxygen
pressure that balances oxygen carried out by each gas species against oxygen
released from its parent oxide. For Knudsen effusion,
`J_i = p_i A W / sqrt(2 pi M_i R T)`; aperture area `A`, Clausing factor `W`,
and temperature are common factors and cancel. The solved condition is

```
sum_i nO_i p_i / sqrt(M_i) =
sum_i (nuO/nuM)_parent nM_i p_i / sqrt(M_i)
```

The function returns pO2 in bar, the same partial-pressure mapping as
`evaluate_gas`, and residual/bracket/carrier diagnostics. Result mode is
`oxygen_balance_effusion`. For a K2O–SiO2 melt where K and O2 dominate, this
reduces to `pO2/pK = 0.25 sqrt(M_O2/M_K) = 0.2262`, the Plante 1979 anchor.

This balance applies to inert cells such as Pt or Ir. It does not apply to
reactive W or Mo cells: their effusing oxides contribute oxygen-bearing
fluxes that this model does not include.

#### Reusing the balance with another gas table

`oxygen_balance_from_pressure_model(pressure_model, species)` accepts a
callable from
`log10(pO2 / bar)` to species partial pressures and metadata mapping each gas
formula to molar mass, oxygen atoms, parent oxygen demand, and its pO2 exponent.
For a gas MxOy formed from parent oxide MaOb, the exponent is
`(y - x*b/a) / 2`; parentless oxygen species use half their oxygen atom count.
Build metadata with `oxygen_balance_species_metadata({"K": "K2O", "O2": None})`.
Pass `pO2_exponents={"custom gas": exponent}` to override or declare exponents
for a custom table. Each declared pressure must follow that power law in pO2.
A caller can supply pressures from another gas table without routing them through
`evaluate_gas`. The solver enforces the 1 bar molecular-flow ceiling and requires
a monotone, bracketed oxygen flux. The inert-cell caveat above still applies.

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

## API

The `[gas]` extra exports per-row thermochemistry and the default channel
selection:

```python
from openimcc import default_gas_channels, species_thermo

props = species_thermo("FeO", "g", 2000.0)
# props.Cp_J_molK, props.S_J_molK, props.H_app_kJ_mol, props.G_J_mol
# props.source_row_id, props.source_table_id, props.T_interval, props.T_min, props.T_max

channels, omitted = default_gas_channels(("SiO2", "MgO"))
```

`species_thermo(species, phase, T_K, datapack=None)` selects the same row the
runtime selector uses and raises `ImccGasTemperatureOutsideDomainError` outside
that row's declared interval. `species` is the bare formula; `phase` is `"g"`,
`"l"`, or `"cr"`. The frozen `SpeciesThermo` result reports Cp and S in
J/(mol K), apparent enthalpy in kJ/mol, and apparent Gibbs energy in J/mol.
`source_row_id` preserves the row's `Ref` value; `source_table_id` gives the
source table identifier recorded in `PROVENANCE.yaml`. `derivatives_fit_implied`
is true for every condensate row because each row's Cp, S, and H_app follow by
differentiating its fitted Gibbs-energy polynomial. This includes JANAF-fitted
condensates, whose fits target Phi (the Gibbs polynomial) only. It is false for
gas rows, whose Shomate fits target source Cp, H, and S directly. Condensate
source gates therefore compare G_app only. Cp, S, and H_app residuals against
JANAF are fit-implied information, not source-quantity gates. The measured
maximum absolute derivative residuals are reported here for the JANAF-backed
condensates:

| Condensate interval | Cp (J/mol K) | S (J/mol K) | H_app (kJ/mol) | G_app (kJ/mol) |
|---|---:|---:|---:|---:|
| FeO(l), 1000–5000 K | 6.111 | 1.536 | 5.393 | 2.273854 |
| TiO2(l), 1500–3000 K | 5.960 | 0.210 | 0.608 | 0.006878 |
| Cr2O3(l), 1900–3000 K | 1.315 | 0.036 | 0.105 | 0.001551 |
| V2O3(l), 1700–3000 K | 0.626 | 0.026 | 0.044 | 0.001300 |
| NbO2(l), 1500–3000 K | 0.911 | 0.047 | 0.070 | 0.001405 |
| Na2O(l), 1200–1500 K | 0.408 | 0.008 | 0.011 | 0.000484 |
| Na2O(l), 1500–3000 K | 5.401 | 0.167 | 0.497 | 0.005218 |

NbO2(l)'s separate 1200–1500 K interval has its own measured maximum G_app
residual, 3.49e-13 kJ/mol; it is not grouped with the 1500–3000 K interval.
Na2O(l) has separate 1200–1500 K and 1500–3000 K runtime intervals. Both fits
include the recovered 1500 K thermal cells; the low fit uses six nodes from
1000–1500 K, and the high fit uses 16 nodes from 1500–3000 K. The 1500 K
source line has ambiguous formation columns, which remain unused. Maximum
G_app residuals across the four low-interval runtime nodes and 16 high-interval
nodes are 0.000484 and 0.005218 kJ/mol, respectively. The low value now covers
four nodes (including the recovered endpoint), rather than the prior five-node
interpolant's 4.66e-13 kJ/mol at three in-range nodes.

The low parent rows are polynomial fits to generated constant-Cp
continuations, not source measurements. Their maximum fit residuals are
measured against the generated continuation; liquid Cp is fixed at T0. JANAF
uses constant liquid Cp for these branches, matching its analytic continuation
for any printed supercooled-liquid nodes. Where a glass or lower-solid branch is
printed, the extension deliberately continues the liquid. A ±10% Cp change is
an illustrative sensitivity scenario, not a statistical uncertainty; ΔG and
log10(K) are derived from `ΔG = ΔCp[(T−T0)−T ln(T/T0)]` and peak at 1200 K.

| Pack | Parent interval (K) | T0 (K) | Cp_l (J/mol K) | Max fit residual (J/mol) | ±10% Cp at 1200 K (J/mol; dex) |
|---|---|---:|---:|---:|---:|
| Research | MgO(l), 1200–2200 | 2200 | 66.944 | 5.627 | 1825; 0.0794 |
| Research | CaO(l), 1200–2200 | 2200 | 62.760 | 2.931 | 1711; 0.0745 |
| Research | Al2O3(l), 1200–2500 | 2500 | 192.464 | 36.583 | 8069; 0.3512 |
| Research | SiO2(l), 1200–1800 | 1800 | 85.772 | 0.800 | 973; 0.0424 |
| Default | TiO2(l), 1200–1500 | 1500 | 100.416 | 1.007 | 324; 0.0141 |
| Default | Cr2O3(l), 1200–1900 | 1900 | 156.900 | 2.626 | 2331; 0.1015 |
| Default | V2O3(l), 1200–1700 | 1700 | 156.900 | 0.457 | 1287; 0.0560 |

For O-044 TiO2(l), the 1400 K source row contains glass-side thermal cells
(Cp = 76.944 J/mol K) followed by the GLASS ↔ LIQUID marker. The first
complete, unambiguous liquid node is 1500 K, Cp = 100.416 J/mol K, and is the
continuation anchor. Genuine supercooled liquid nodes are complete from
1500–2100 K and 2300–3000 K; the 2200 K row is parse-ambiguous. The fit checks
the constant-Cp continuation against those genuine liquid-branch nodes.

For O-063 V2O3(l), the generated continuation anchors at the 1700 K liquid
node (Cp = 156.900 J/mol K); its enthalpy and entropy reproduce the liquid
branch at the 1600 K transition. The liquid-only fit uses 13 complete nodes
from 1700–2300 K and 2500–3000 K, omitting the glass-side 1500 K node, both
transition-marked 1600 K lines, the 2340 K II ↔ LIQUID marker, and the
parse-ambiguous 2400 K grid point. Both omitted lines sit on the Cp = 156.900
J/(mol K) branch. The continuation covers 1200–1700 K, with the high row
selected exactly at 1700 K. Strict gas calls below 1500 K still refuse because
the V gas rows begin there.

The continuation-minus-high-row Gibbs seams are −0.0031 kJ/mol (research MgO),
−0.0016 (research CaO), −0.0295 (research Al2O3), −0.0005 (research SiO2),
−0.0043 (default TiO2), −0.0010 (default Cr2O3), and −0.0005 (default V2O3).
The major-oxide rows and default results remain the Lamoreaux functions; the
JANAF major continuations are available only through the opt-in research pack.

In default data, the three labelled rows close the TiO2, Cr2O3, and V2O3
parent-liquid gaps. Ti, V and Cr still have gas-interval gaps, so their full
standard-state C4 checks remain incomplete. Major-oxide default rows keep their
existing out-of-interval behavior and domain flags.

For gas rows, `G_J_mol = 1000*H_app_kJ_mol - T*S_J_molK`; the factor of
1000 converts enthalpy to J/mol. For LAM condensates, the following measured
model-minus-JANAF G residual ranges are retained as documented exceptions.

| LAM row | Compared sources | Measured G_app residual range (kJ/mol) | Tolerance (kJ/mol) | Reason |
|---|---|---:|---:|---|
| Al2O3(l) | LH87 Tables 2/3; JANAF Al-100 | −0.130343 to 0.101552 | 0.002 | Both anchors refer to stable solid at 298 K; sources do not reconcile liquid Gibbs-energy fits. |
| SiO2(l) | LH87 Table 2; JANAF O-038 | −2.972323 to −2.302783 | 0.002 | Sources report different liquid Gibbs thermochemistry without a reconciliation. |
| MgO(l) | LH87 Table 2; JANAF Mg-009 | −0.998335 to −0.356769 (3100–3500 K) | 0.002 | Complete JANAF liquid nodes inside the packaged interval are included; sources do not reconcile their G fits. |
| CaO(l) | LH87 Table 2; JANAF Ca-028 | −6.185 to −1.210793 (2900–3800 K) | 0.003 | Complete JANAF liquid nodes inside the packaged interval are included; sources do not reconcile their G fits. |

The default Na2O(l) row is the JANAF Na-013 liquid fit, replacing the corrected
LH84 Table 2 row. The printed LH84 row minus the new JANAF Na-013 fit is
+13.44, +25.57, +29.64, and +17.66 kJ/mol at 1600, 2000, 2400, and 3000 K.
Na-013 tabulates liquid Cp = 104.600 J/(mol K). Its 1405.2 K
ALPHA ↔ LIQUID marker is excluded from fitting. At 1500 K, only the intact
thermal cells are recovered from the parse-ambiguous line; its formation
columns remain unused. The supercooled-liquid fit uses 1000–1500 K nodes and
the high fit uses 1500–3000 K nodes.

For gas rows, `H_app_kJ_mol` includes the formation-enthalpy anchor folded into
Shomate F (the stored Shomate H is zero); callers needing H−H298 must subtract
the source record's `dfH298`. Condensate H_app includes the stable-phase
298.15 K anchor and the analytic temperature-dependent increment.

`default_gas_channels(parent_oxides, datapack=None)` returns the ordered
`(channels, omitted)` pair used by the gas evaluator. Both functions use the
packaged tables by default and accept a loaded `ImccGasDatapack` for explicit
table selection.

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
| Na | 1 | −0.151 | 0.151 |
| NaO | 1 | −0.349 | 0.349 |
| O | 1 | −0.007 | 0.007 |
| SiO | 1 | −0.009 | 0.009 |
| FeO | 1 | +0.009 | 0.009 |

With the JANAF Na-013 parent, the Table 9 Na and NaO residual medians are
−0.151 and −0.349 dex. Across comparable Fig. 10 points, Na and NaO have
medians of +0.158 dex (n = 35) and +0.174 dex (n = 27); Na2 is +0.132 dex at
its one digitized point. Na2 has no Table 9 anchor. O, SiO and FeO each match
their own Table 9 anchor to about 0.01 dex (−0.007, −0.009 and +0.009), so this
is not a unit or fO2-pin error. The Fig. 10 medians for SiO and FeO sit about
+0.2 dex above the transcribed anchor; that is a figure-versus-table
difference in the paper's digitized data, not a reconciled result.

#### Fig. 10 points (digitized)

| species | n | signed median | abs max |
|---|--:|--:|--:|
| Na | 35 | +0.158 | 0.342 |
| NaO | 27 | +0.174 | 0.782 |
| Na2 | 1 | +0.132 | 0.132 |
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

#### Known limit: alkali silicate melts at and beyond the metasilicate

The most basic alkali silicate complexes in `imcc-sf04-v1.0.2` are the
metasilicates `Na2SiO3` and `K2SiO3`. There is no orthosilicate (`Na4SiO4`,
`K4SiO4`) or pyrosilicate (`Na6Si2O7`, `K6Si2O7`). The +4 dex X > 0.5 rows in
the sodium table above follow from that.

In a Me2O–SiO2 binary with `x = X(Me2O) > 0.5`, the metasilicate takes all
`1 − x` mol of SiO2 and the same amount of Me2O. That leaves `2x − 1` mol of
free Me2O among `x` mol of associate species. The Raoultian activity becomes

    a(Me2O) → (2x − 1) / x

That is the limit for a fully formed complex, and it depends only on
stoichiometry, not on temperature or any coefficient. The model sits on it:
with `allow_out_of_envelope=True` and `allow_extrapolation=True` it returns
0.181818–0.181819 at x = 0.55 and 0.333333–0.333334 at x = 0.60, for both
Na2O and K2O at 1373.15 K and 1573.15 K. So a value beyond the metasilicate is a
mass-balance artefact, not a thermodynamic prediction. That is why the
validated envelope stops at X(Me2O) = 0.5.

The same missing complexes probably also degrade the model at X = 0.5, which
is still inside the envelope. There, free Me2O comes only from dissociation
of the metasilicate, and no more basic complex takes up the excess. The model
reads high by about 2 dex for Na2O against Tsukihashi & Sano (1985, Tetsu to
Hagane 71, 815; 1373–1573 K) and by about 3.5 dex for K2O against Tsaplin et
al. (2000). At X ≤ 0.45 both agree within 0.5 dex.

The `species-coverage-edge` flag marks this regime. It fires when the free
SiO2 fraction falls below 1.91e-3 of nominal, i.e. when the ladder has used
up its acidic sink. In the binaries it first appears at X = 0.493–0.499
(K2O: 0.493 at 1373 K, 0.497 at 1873 K; Na2O: 0.498 and 0.499), and it does
not fire for lunar basalts. Treat any Me2O activity carrying that
flag as a structural limit of the complex set, not as a measured
disagreement. The fix is to add the orthosilicate and pyrosilicate
complexes. Nothing is retuned.

#### Known limit: complex rows below their demonstrated range

The complex rows are evaluated from 1700 K, but several are Fegley & Cameron
(1987) `A + B/T` fits that the paper demonstrated only over 2500–3500 K.
Below 2500 K they are extrapolated. Against JANAF (4th ed.) reaction Gibbs
energies (the complex minus its parent liquids), the packaged rows differ at
1700–2500 K by up to 6.4 kJ/mol for MgAl2O4 (at 1700 K, on the glass segments
of the JANAF liquid tables), 1.1 kJ/mol for Mg2SiO4, 0.7 kJ/mol for MgTi2O5,
and 0.3–0.4 kJ/mol for MgTiO3, Mg2TiO4 and MgSiO3 (0.27 kJ/mol). On the liquid
branch at 2600–3000 K, MgAl2O4 differs by +5.0 to +6.0 kJ/mol (0.10–0.11 dex).

The alkali-silicate rows are assessed or empirical fits, so they cannot be
extended below 1700 K from crystal data, and
evaluations below 1700 K are extrapolations (`evaluate(...,
allow_extrapolation=True)` flags them). The silica parent adds a further
caveat: SF04 places the SiO2 transition at 1996 K, while JANAF adopts about
1696 K and tabulates no supercooled liquid branch below 1800 K.

The pack is unchanged: every row is its source's published value.

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
