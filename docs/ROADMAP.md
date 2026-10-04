# Roadmap

What is not done yet, and what was decided along the way. Everything here was
measured against the code rather than estimated.

## Completed

## Element completeness

On the README basalt over 1200–3000 K and fO2 = 1e-12, 1e-10, 1e-8, 1e-6,
1e-4, `complete` requires an internally computed parent-oxide activity (C1),
every source-listed neutral E–O gas included or screened below a maximum pressure
ratio of 1e-4 (C2), charged species modeled when their source-screened share
exceeds 1e-4 (C3), and 1-bar gas rows with source-defined parent-liquid rows
covering the domain (C4). A NASA condensed-phase parent retains its source
card's pure-liquid standard state; the caller activity is relative to that
standard. Validation is recorded separately: it is `validated` only when a
scoreable observation exists.

Default TiO2(l), Cr2O3(l), V2O3(l), SiO2(l), Al2O3(l), MgO(l), and CaO(l)
have labelled constant-Cp continuation rows that count as parent coverage
while preserving runtime domain flags.

The C3 ion-share grid includes only nodes where summed neutral partial pressure
is at most 1 bar, the validity ceiling used by the gas layer. Of 95 nodes, 69
meet that bound. `evaluate_gas(..., include_ions=True)` solves charge balance
on the full default neutral set, adds fitted Na+, K+, Ca+ and e−, and includes
every available JANAF-supported anion whose neutral is present. Caller-supplied
trace-parent activities enable Li+, Rb+, Pb+, Cs+, Cu+, Sn+, Cs2O+, Ga+, Ge+,
B+, Ga−, B−, BO− and BO2−. BO− uses a NASA CEA card because JANAF has no BO−
table. In+ remains in the source ledger but is declined after its fitted
Kion/Saha ratios (1.510e28 at 1500 K and 9.143e16 at 2500 K) fail the NIST
5.78636 eV ionisation-energy check. The default result remains neutral-only.
Na and K remain `complete-except-ions` because their bounded shares exceed
1e-4 while ions are opt-in; Ca, Li, Rb, Pb, Ga and Ge remain below the C3
threshold on their recorded activity screens. B exceeds the threshold when
its cation and available anions are combined; Cs is also above the threshold.
Positive molecular ions and nonthermal electrons remain outside this
calculation. Mn, Ni and Co are not computed because their parent-liquid rows
are available only from an external pack. P channels also need an external
source-rated P2O5(l) row.

| Element | Status | C1 | C2 | C3 | C4 | Validation | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| O | input (fO2 pinned) | n/a | yes | n/a | n/a | unvalidated | O is input (fO2 pinned), so C1, C3 and C4 do not apply. |
| P | gas-partial | no | yes | not computed: ionized phosphorus channels are outside this neutral gas-only change | no | unvalidated | JANAF P gas rows cover 500-3000 K; no public evaluated P2O5(l) G(T) function was found and JANAF lists P4O10(cr) only. Channels need an external, source-rated P2O5(l) standard state and caller-supplied a(P2O5); OpenIMCC provides no pyrolysis-temperature melt activity model. |
| S | gas-complete-melt-pending | no | yes | not computed: ionized sulfur channels are outside this neutral gas-only change | yes | unvalidated | JANAF sulfur gas rows and the S2(g) parent cover 500-3000 K; callers supply a(S2)=f(S2)/p° on the 1-bar gas reference. OpenIMCC provides no pyrolysis-temperature sulfur melt activity model. |
| Si | gas-partial | yes | yes | yes | no | validated | The JANAF SiO2(l) parent covers 1200-3000 K with a labelled continuation below 1800 K, but Si2(g) and Si3(g) start at 1500 K, leaving a gas C4 gap; joint C3 closure-screen maximum 5.729e-14 at 2800 K and fO2=1e-4 within the <=1-bar neutral-pressure domain; unmodeled positive molecular ions and thermal electrons from walls or other sources remain outside the estimate. |
| Mg | complete | yes | yes | yes | yes | unvalidated | The JANAF MgO(l) row and Mg gas rows cover 1200-3000 K, including a labelled parent continuation below 2200 K; joint C3 closure-screen maximum 1.154e-07 at 2800 K and fO2=1e-4 within the <=1-bar neutral-pressure domain; unmodeled positive molecular ions and thermal electrons from walls or other sources remain outside the estimate. |
| Fe | complete | yes | yes | yes | yes | unvalidated | FeO and neutral gas rows cover the domain; joint C3 closure-screen maximum 3.453e-08 at 2800 K and fO2=1e-4 within the <=1-bar neutral-pressure domain; unmodeled positive molecular ions and thermal electrons from walls or other sources remain outside the estimate. |
| Ca | complete | yes | yes | yes | yes | unvalidated | The JANAF CaO(l) row and Ca gas rows cover 1200-3000 K, including a labelled parent continuation below 2200 K; fitted Ca+ channel maximum p(Ca+)/neutral Ca gas 6.519e-05 at 2800 K and fO2=1e-4 within the <=1-bar neutral-pressure domain, below the 1e-4 C3 threshold. |
| Al | gas-partial | yes | yes | yes | no | unvalidated | The JANAF Al2O3(l) parent covers 1200-3000 K with a labelled continuation below 2500 K, but Al2(g) starts at 1500 K, leaving a gas C4 gap; joint C3 closure-screen maximum 4.273e-06 at 2800 K and fO2=1e-4 within the <=1-bar neutral-pressure domain; unmodeled positive molecular ions and thermal electrons from walls or other sources remain outside the estimate. |
| Ti | gas-partial | yes | yes | yes | no | unvalidated | A labelled constant-Cp TiO2(l) continuation covers 1200-1500 K below the glass branch; Ti(g), TiO(g), and TiO2(g) still start at 1500 K, leaving a lower C4 gap; joint C3 closure-screen maximum 1.159e-09 at 2300 K and fO2=1e-10 within the <=1-bar neutral-pressure domain; unmodeled positive molecular ions and thermal electrons from walls or other sources remain outside the estimate. |
| Na | complete-except-ions | yes | yes | no | yes | unvalidated | Na2O(g) intervals cover 500-3000 K and JANAF Na-013 Na2O(l) intervals cover 1200-3000 K, including its supercooled-liquid branch; fitted Na+ channel maximum p(Na+)/neutral Na gas 9.568e-04 at 2800 K and fO2=1e-4 remains above 1e-4; ions are opt-in and omitted from default results. |
| K | complete-except-ions | yes | yes | no | yes | validated | K2O(g) intervals cover 500-3000 K and K2O(l) covers the C4 domain; fitted K+ channel maximum p(K+)/neutral K gas 4.325e-02 at 1200 K and fO2=1e-4 remains above 1e-4; ions are opt-in and omitted from default results. |
| Cr | gas-partial | no | yes | yes | no | unvalidated | A labelled constant-Cp Cr2O3(l) continuation covers the parent interval below 1900 K; Cr2O3 activity is caller-supplied and Cr(g) ends at 2900 K, leaving the upper C4 gap; joint C3 closure-screen maximum 1.099e-06 at 2800 K and fO2=1e-4 within the <=1-bar neutral-pressure domain; unmodeled positive molecular ions and thermal electrons from walls or other sources remain outside the estimate. |
| V | gas-partial | no | yes | yes | no | unvalidated | A labelled constant-Cp V2O3(l) continuation covers 1200-1700 K; V(g), VO(g), and VO2(g) still start at 1500 K, leaving a lower C4 gap; activity is caller-supplied; joint C3 closure-screen maximum 8.022e-09 at 2300 K and fO2=1e-10 within the <=1-bar neutral-pressure domain; unmodeled positive molecular ions and thermal electrons from walls or other sources remain outside the estimate. |
| Nb | gas-partial | no | yes | yes | no | unvalidated | Nb-013 is liquid from 1000 K and its condensate intervals cover 1200-3000 K; Nb(g), NbO(g), and NbO2(g) start at 1500 K, leaving a gas C4 gap; activity is caller-supplied; joint C3 closure-screen maximum 5.436e-13 at 2300 K and fO2=1e-10 within the <=1-bar neutral-pressure domain; unmodeled positive molecular ions and thermal electrons from walls or other sources remain outside the estimate. |
| Li | gas-complete-melt-pending | no | yes | yes | yes | unvalidated | Li2O(l) and neutral Li gas rows cover 1200-3000 K using the printed JANAF liquid-branch cells; a(Li2O) is caller-supplied; the joint C3 screen used a(Li2O)=1e-3 and reached 4.335e-08 at 2000 K and fO2=1e-4 within the <=1-bar neutral-pressure domain; unmodeled positive molecular ions and thermal electrons from walls or other sources remain outside the estimate. |
| Rb | gas-complete-melt-pending | no | yes | yes | yes | unvalidated | Rb2O(l) and neutral Rb gas rows cover 1200-3000 K; a(Rb2O) is caller-supplied; the joint C3 screen used a(Rb2O)=1e-3 and reached 5.472e-05 at 2000 K and fO2=1e-4 within the <=1-bar neutral-pressure domain; unmodeled positive molecular ions and thermal electrons from walls or other sources remain outside the estimate. |
| Pb | gas-complete-melt-pending | no | yes | yes | yes | unvalidated | PbO(l) and neutral Pb gas rows cover 1200-3000 K; a(PbO) is caller-supplied; PbO(l) is the Pb(II) parent because no evaluated PbO2(l) parent row was found; the joint C3 screen used a(PbO)=1e-3 and reached 1.301e-12 at 2000 K and fO2=1e-4 within the <=1-bar neutral-pressure domain; unmodeled positive molecular ions and thermal electrons from walls or other sources remain outside the estimate. |
| Ga | gas-complete-melt-pending | no | yes | yes | yes | unvalidated | Ga2O3(l) uses its NASA liquid card from 2080 K with a labelled constant-Cp continuation to 1200 K; GaO(g) and Ga2O(g) are included; a(Ga2O3) is caller-supplied; the joint C3 screen used a(Ga2O3)=1e-3 and reached 5.443e-07 at 2600 K and fO2=1e-4 within the <=1-bar neutral-pressure domain. |
| Ge | gas-complete-melt-pending | no | yes | yes | yes | unvalidated | GeO2(l) uses its NASA liquid card from 1388 K with a labelled constant-Cp continuation to 1200 K; GeO(g) and GeO2(g) are included; a(GeO2) is caller-supplied; the joint C3 screen used a(GeO2)=1e-3 and reached 3.495e-14 at 2300 K and fO2=1e-4 within the <=1-bar neutral-pressure domain. |
| B | gas-partial | no | yes | no | yes | unvalidated | JANAF B2O3(l) is fitted from its printed liquid branch after the 723 K marker; BO(g), BO2(g), B2O(g), B2O2(g), and B2O3(g) are included; a(B2O3) is caller-supplied; the joint B-ion screen used a(B2O3)=1e-3 and reached 1.252e-02 at 2800 K and fO2=1e-4, above the 1e-4 C3 threshold. |
| Cs | gas-partial | no | yes | no | yes | unvalidated | Cs2O(l) and neutral Cs gas rows cover 1200-3000 K; a(Cs2O) is caller-supplied; the C3 screen at a(Cs2O)=1e-3 reached 1.342e-04 at 2000 K and fO2=1e-4 within the <=1-bar neutral-pressure domain, above the 1e-4 limit. |
| Cu | gas-complete-melt-pending | no | yes | yes | yes | unvalidated | Cu2O(l) is the Cu(I) parent because JANAF provides its liquid row; neutral Cu gas rows and the parent cover 1200-3000 K; a(Cu2O) is caller-supplied; Cu- is available on the opt-in ion path; its corrected attachment-Saha ratios against NIST EA 1.23578 eV are 0.941424 at 1500 K and 0.958988 at 2500 K; Cu+ is declined because JANAF Cu-006 gives fitted/Saha ratios 1666.15 and 1656.61. |
| Sn | gas-complete-melt-pending | no | yes | yes | yes | unvalidated | SnO(l) is selected over the later-starting SnO2(l) card as the tin parent; neutral Sn gas rows and the parent cover 1200-3000 K; a(SnO) is caller-supplied; the C3 screen at a(SnO)=1e-3 reached 3.515e-08 at 2800 K and fO2=1e-4 within the <=1-bar neutral-pressure domain. |
| In | gas-partial | no | yes | declined | yes | unvalidated | In2O3(l) uses its NASA liquid card from 2186 K with a labelled constant-Cp continuation to 1200 K; InO(g) and In2O(g) are included; a(In2O3) is caller-supplied. In+ is declined because its NASA card gives Kion/Saha ratios of 1.510e28 at 1500 K and 9.143e16 at 2500 K against NIST's 5.78636 eV ionisation energy. |
| Mn | gas-partial | no | yes | not computed: p(E) needs a parent liquid row that is only available from an external private pack | no | unvalidated | Only Mn(g) has a public row; the MnO gas and MnO liquid parents are absent. |
| Ni | gas-partial | no | yes | not computed: p(E) needs a parent liquid row that is only available from an external private pack | no | unvalidated | Only Ni(g) has a public row; the NiO gas and NiO liquid parents are absent. |
| Co | gas-partial | no | yes | not computed: p(E) needs a parent liquid row that is only available from an external private pack | no | unvalidated | Only Co(g) has a public row; the CoO gas and CoO liquid parents are absent. |

The C3 maximum share is the fitted or estimated pressure in the modeled
charge states divided by total neutral gas pressure carrying that element,
evaluated only at nodes with summed neutral pressure <=1 bar. For B the
numerator includes B+, B−, BO− and BO2−. For the newly modeled elements, the
prior positive-only estimate and bounded fitted closure screen give:

| Element | Prior max share (T, fO2) | Fitted max share (T, fO2) |
| --- | --- | --- |
| Na | 1.169e-3 (3000 K, 1e-4) | 9.568e-4 (2800 K, 1e-4) |
| K | 3.979e-2 (1300 K, 1e-4) | 4.325e-2 (1200 K, 1e-4) |
| Ca | 1.052e-4 (3000 K, 1e-4) | 6.519e-5 (2800 K, 1e-4) |
| Li | — | 4.335e-8 (2000 K, 1e-4) |
| Rb | — | 5.472e-5 (2000 K, 1e-4) |
| Pb | — | 1.301e-12 (2000 K, 1e-4) |
| Ga | — | 5.443e-7 (2600 K, 1e-4) |
| Ge | — | 3.495e-14 (2300 K, 1e-4) |
| B | — | 1.252e-2 (2800 K, 1e-4) |
| Cs | — | 1.342e-4 (2000 K, 1e-4) |
| Cu | — | 2.331e-5 (2800 K, 1e-4) |
| Sn | — | 3.515e-8 (2800 K, 1e-4) |

The P/S channels are optional and stay out of default gas results unless their
caller-supplied parent activity is present. Sulfur uses the JANAF `S2(g)` gas
standard. Phosphorus uses `P2O5(l)`, but no public evaluated liquid G(T) function
was found (JANAF lists `P4O10(cr)` only); an external pack must supply a
source-rated `P2O5(l)` standard state before callers can use `a(P2O5)`.

The default uses the JANAF major-oxide fits. The opt-in `sf04-published` pack
preserves the previous LAM1987 parent rows for comparison reproduction. The
former research pack is retired because it duplicated the new default. Sulfur
has C4=yes via its S2(g) parent. The 1200–3000 K default C4 criterion requires continuous
one-bar gas-table and parent-liquid coverage for oxide-parent channels. Na2O(g)
and K2O(g) cover the gas domain with NASA Glenn interval 2. The Na2O(l) row is
a JANAF Na-013 fit, replacing the corrected LH84 row. Its low runtime interval
covers 1200–1500 K from five complete 1000–1400 K supercooled-liquid nodes; its
high interval covers 1500–3000 K from complete 1600–3000 K nodes. The 1405.2 K
ALPHA ↔ LIQUID marker and parse-ambiguous 1500 K row are omitted from fitting.
Na therefore has C4=yes at 1200 K. The melt kernel computes the Na2O activity
from its unchanged published melt pack and does not read the gas-side
condensate row.
NbO2(l), TiO2(l), Cr2O3(l), V2O3(l), SiO2(l), Al2O3(l), MgO(l), and CaO(l)
have generated low-temperature continuation intervals. The major-oxide
continuations extend default parent coverage; Si and Al retain gas C4 gaps
because their Si2/Si3 and Al2 rows start at 1500 K.

### Low-temperature gas interval (done)

The 22 JANAF-fit battery gas species and Na₂O(g)/K₂O(g) have interval 2 from
500–1500 K. The NASA rows use piecewise 300–1000 K and 1000–6000 K cards, with
H and S continuity checked at 1000 K. Existing 1500–3000 K interval 1 rows
remain byte-identical to base `52db3a9`; the runtime selects interval 1 at the
shared 1500 K node. The low-temperature KEMS impact calculation at 1300 K and
1400 K is domain-refused because the packaged K₂O–SiO₂ melt reactions begin at
1700 K.

The default SiO2(l) fit starts at 1800 K; below it, the labelled 1200–1800 K
constant-Cp liquid continuation is selected and reported in gas result flags.
JANAF O-038 has no source liquid branch below 1800 K; its lower source rows are
not used as liquid data. SiO2(cr) remains in the default and `sf04-published`
tables because the public `species_thermo(..., phase="cr")` query reads that
row. No gas reaction consumes it: the SF04 silica parent is SiO2(l), matching
the liquid melt-activity standard state.

### Major-oxide parent basis (done)

The gas reaction table stores stoichiometry only; `evaluate_gas` calculates
each reaction Gibbs energy and pressure from the active gas and parent rows.
Oxygen balance is computed from those pressures and reaction stoichiometry.
The FC87/SF04 complex-species log K coefficients are in the separate melt
datapack and are not derived from these gas parent rows. No other default row,
fitted gas constant, or oxygen-balance constant tied to the old LAM parents
was found, so nothing was re-fit.

The table reports `G_JANAF - G_LAM` for each replaced liquid row. With other
gas Gibbs terms and fO2 fixed, the oxide-to-vapor reaction gives
`x log10(p_M) = constant + G_parent/(R T ln(10))`; subtracting the old-parent
equation from the new gives a per-metal-atom pressure shift of
`(G_JANAF - G_LAM)/(x R T ln(10))`. Here `x` is the oxide's metal count
(`SiO2`, `MgO`, `CaO`: 1; `Al2O3`: 2). The units are J/mol divided by J/mol,
so the result is dex; a positive parent Gibbs difference must raise vapor
pressure, matching every value below. No constant is fitted.

| T (K) | SiO2(l) | Al2O3(l) | MgO(l) | CaO(l) |
| ---: | ---: | ---: | ---: | ---: |
| 1400 | +2609.883 | +148626.283 | +17027.777 | +29905.243 |
| 1600 | +2782.161 | +85970.108 | +12937.805 | +25287.310 |
| 1933 | +2945.207 | +23096.714 | +7392.937 | +18652.675 |
| 2200 | +2967.851 | +3871.566 | +4176.937 | +14273.533 |
| 2600 | +2763.704 | +11.241 | +1292.342 | +9110.007 |

All Gibbs differences are J/mol. Every value is positive, so the new parent
rows raise the corresponding equilibrium vapor pressure. These are direct
row comparisons; published-SF04 pressure pins remain on `sf04-published`.

The fitted negative-ion rows include Na−, K−, O−, O2−, Al−, AlO2−, Fe−, Si−,
Ti−, KO−, AlO−, NaO−, Cr−, V−, Nb−, Cs− and Cu−. Every available listed anion whose
neutral channel is present enters charge balance, regardless of its individual
attachment term. Across the README-basalt temperature and fO2 grid,
O2− reaches `K_A p(O2) = 4.14e-4`, AlO2− reaches a 6.46 pressure ratio to its
neutral, and KO− reaches a 1.48e-3 pressure ratio to KO; each is included in
the charge closure. JANAF table K-009 supplies KO− thermochemistry; its
formation enthalpy is not marked estimated in that source table.

### 1. The gas layer is self-contained (done)

The packaged gas tables now load by default through `importlib.resources`.
Forty-one Shomate rows are fitted deterministically from vendored NIST-JANAF
records, with the exact source hash and fit residual recorded per row. The
gas-row maximum fit residual is **1.9231 J/mol** (**3.7465e-5 log10 K**), well below the
0.01 log10 K gate. The eight transcribed condensate rows reproduce the current
published coefficient values exactly; TiO2(l), Cr2O3(l), V2O3(l) and NbO2(l)
are fitted from JANAF by the same generator, with maximum residuals of 6.88,
1.55, 1.30 and 1.41 J/mol respectively (TiO2(l), Cr2O3(l), V2O3(l) and
NbO2(l)); the Cr2O3(l) residual is 3.52e-5 log10 K. The caller-supplied Cr/V/Nb
parents remain outside the eight-oxide melt basis.
Mn(g), Ni(g) and Co(g) are also vendored and fitted, but remain explicitly
incomplete-parent rows until their liquid parent records are available.
`OPENIMCC_VAPOROCK_ROOT` remains an explicit comparison override; its legacy
condensate table has none of these fitted parent rows, so a default call under
it omits the Ti, Cr, V and Nb channels.

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
| `hastie_sio_1907_796` | 1907.795668 | 4.851884054e-6 | 0.082727519 | 0.060040280 | −0.139207291 | — | `janaf_fitted` |
| `hastie_sio_1909_739` | 1909.739189 | 5.016024124e-6 | 0.116737396 | 0.062052444 | −0.274451105 | — | `janaf_fitted` |
| `hastie_sio_1948_149` | 1948.149056 | 9.545997137e-6 | 0.206709293 | 0.117379741 | −0.245766853 | — | `janaf_fitted` |

The three SiO points are inside the default JANAF SiO2(l) fit interval, so they
have no gas domain flag. The current full-set report has 402 rows, 398
predictions, 94 flagged, 94 out-of-domain and 4 refused. Its arithmetic total
across incompatible slices uses 304 residuals (median −0.5094 dex, mean
−0.4678 dex, RMSE **0.8793 dex**, flagged RMSE **2.002 dex**); it is not an
accuracy figure. `per_slice` and `binary_slices` are the machine-readable
standard-state-separated aggregates. The 396 non-gas rows retain their
pre-rewire predictions, residuals, statuses and reasons exactly.

The gas wiring does not settle the alkali species-family residuals. The
published Table 9 K/Na and Hastie K counterfactuals below are the relevant
diagnosis; the low Hastie K result must not be attributed to the gas-layer
rewire or silently corrected by changing K2O(l).

Upstream this called the simulator's analytical vapour stack, which reads a
large catalogue through two further subsystems. **That is not worth porting.**
`openimcc.gas.evaluate_gas()` already covers every species the bench needs —
the six SF04 oxide families plus Ti, screened Al/Si dimers, CrO, and the
caller-supplied V/Nb channels are in `IMCC_GAS_CHANNEL_SPECIES` — and takes
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

### 4. The independent SF04 reference is tracked (done; workbook remains local)

`tests/test_rung3_fixture.py` may compare against
`tests/fixtures/imcc_sf04_magma_workbook.csv`, a 327 KB derivative of
`Schaefer2004-MAGMA-valid.xlsx` from the VapoRock data directory. The source
material has the same redistribution problem as the gas-source material, and
the workbook is MAGMA output whose own redistribution terms are unestablished;
it is not a public reference and the test skips when the file is absent.

An independent, public partial reference now ships at
`benchmarks/references/schaefer-fegley-2004/`: 5 Table 5 composition rows
(`transcribed`), 13 Table 9 flux rows (`transcribed`) with 13 Eq. 11 pressures
(`derived_eq11`), and 350 Fig. 10 points (`digitized_figure`) with 350 Table 7
pressure derivations (`derived_table7`). It is covered by
`tests/test_sf04_published_reference.py` against `sf04-published` for the
published-parent reproduction checks and does **not** replace the workbook
fixture. Bishop Tuff, Type B CAI, the Al/Ca/Ti/K2/Zn channels, and 1500, 1625,
and 2500 K remain unavailable from the published paper.

The default JANAF-parent comparison was remeasured against the same anchors and
digitized points. These are reported figures, not default acceptance gates;
the pack above owns the published-parent regression pins. Residuals are
`log10(predicted / measured)` in dex, with O2 excluded as the fO2 pin.

| Source | Species | n | Signed median | Abs max |
| --- | --- | ---: | ---: | ---: |
| Table 9 | Mg | 1 | +0.115 | 0.115 |
| Table 9 | SiO | 1 | +0.072 | 0.072 |
| Table 9 | SiO2 | 1 | +0.060 | 0.060 |
| Fig. 10 | Mg | 24 | +0.192 | 0.867 |
| Fig. 10 | MgO | 8 | +0.063 | 1.012 |
| Fig. 10 | SiO | 30 | +0.244 | 1.139 |
| Fig. 10 | SiO2 | 24 | +0.184 | 1.121 |

At the published Table 9 tholeiite anchor, K is **+0.14 dex** with
`KCaAlSi2O7`; removing that complex moves K to **+2.31 dex**. It must stay:
the **−2.3 dex K cliff** in the old MAGMA workbook baseline is not the paper's
Table 9 result. Switching the corrected LH84 Na2O(l) row to JANAF Na-013 moves
the same anchor's Na residual from **+0.168 dex** to **−0.151 dex**. Removing
the four `nu(Na2O) = 0.5` complexes moves that residual to about
**+0.737 dex**; removing
all seven Na-bearing complexes moves it to **+3.794 dex**. The sum of the
single-family shifts predicts about **+0.746 dex**, still well below the joint
deletion result. The shifts do not add because the complexes compete for one
Na inventory. The published-reference and species-set tests record these
diagnostics; they do not tune the model.

The K errors have opposite signs in different complex families. In Hastie case
4, K is **−0.89 dex** and removing the `nu(K2O) = 0.5` K-aluminosilicates
overshoots to **+1.45 dex**. In the Tsaplin K2O–SiO2 binary, the `nu = 1`
silicates under-bind K and the measured activity residual is high (median
**+0.69 dex** in the 11-row evidence summary; the tracked transcript's 10
explicit rows have median **+0.61 dex**). The same K2O–SiO2 binary is reproduced
more closely through pressures in Plante 1979 (NBS SP561): the P_K residual has
median **+0.09 dex**, RMSE **0.20 dex**, and 98/162 points fall inside the
authors' ±0.15-dex temperature error. This is a pressure comparison, not an
activity comparison; the roughly **0.5-dex disagreement** between the two rails
stays as data and is not reconciled. The sodium and potassium ladders both end
at the metasilicate: there is no Na4SiO4 or K4SiO4 sink above that edge. At the
edge, the model's sodium residuals expose that coverage collapse: Tsaplin is
**+4.20 dex** (X ≥ 0.523, median) and Tsukihashi & Sano is **+2.08 dex**
(X = 0.50, mean). Below it, Tsaplin's Na residual is about **+0.06 dex** while
Tsukihashi & Sano's is about **−0.39 dex**;
the roughly **0.4-dex disagreement stays as data**.

The continuous edge metric is `acid_sink_ratio = x*(SiO2) / x(SiO2)`. Its
predict-and-flag cut is **1.910055e-3**, the geometric mean of the largest
1473 K edge-shoulder ratio below the validated floor that must be flagged,
**1.864382e-3** (K2O-SiO2, X = 0.494, +1.40 dex against the linear anchor
bridge), and the nearest strict-path validated composition at
**1.956846e-3** (`kume2000_s145`, 1823 K). A 0.001 scan from X = 0.450 to
0.500 with both overrides found the adjacent K points X = 0.495 and 0.496
and Na X = 0.498 below the floor as well; 1373/1573 K sanity scans remain
below the selected cut.

This is an irreducible limitation: above the validated floor, the alkali
shoulder **cannot** be separated from a validated composition by this ratio
alone. K2O-SiO2 at X = 0.493 has ratio **2.183642e-3** and misses by +1.33
dex; at X = 0.48 it is **6.588448e-3** and +0.87 dex; at X = 0.457 it is
**1.587897e-2** and +0.52 dex. No edge flag does not mean accurate. A narrow band also remains between the cut and the validated floor
(K2O-SiO2 X = 0.49372 at 1473 K: ratio 1.953518e-3, +1.38 dex, unflagged); closing it would
mean a zero-margin cut on a single bench point, so it is documented rather than closed. Use
`acid_sink_ratio` where it is informative—near the lower edge as an
acid-sink-exhaustion indicator—and consult `acid_sink_ratio` with the
activity/residual diagnostics above the validated floor.

The equimolar eight-oxide control is flagged at ratio **8.988617e-4** from
the solved partition: 72.5% of bound silica is in K–Ca–Al complexes, so its
family is **K–Ca–Al silicate**, not generic mixed basic oxide.

The non-tuning fix routes are an **inactive primary re-derivation** of the
`nu = 0.5` rows as a separate coefficient set, and **MD free energies**. The
published rows remain unchanged until either route supplies independent
evidence.

### 5. Documentation (done)

`docs/datapack-format.md`, `CONTRIBUTING.md`, and `CHANGELOG.md` document the
shipped format, contribution path, and extracted history.

## Known gaps

### 6. Primary provenance for K2O(l)

The shipped K2O(l) coefficients are a secondary transcription from a
condensate CSV, cited to Lamoreaux & Hildenbrand 1984, and are flagged
`secondary_transcription_unverified_primary`. The K, K2 and KO channel authority
therefore carries that flag. Once the LH84 primary coefficient table is
available, re-certify the row and remove the flag.

The NIST-JANAF potassium index lists K2O(cr) but no K2O(l) table
([K tables](https://janaf.nist.gov/tables/K-index.html)); the vendored K-012
record is the crystalline phase. A future liquid-row switch would need a
public liquid Gibbs-energy function or other source-rated thermochemistry,
then a fitted row with source provenance and updated K/K2/KO comparisons.
K2O(cr) cannot be substituted because the gas reaction and melt activity use
the liquid standard state. This switch leaves the existing LH84 row untouched.

### 7. Provenance vocabulary is shared across the shipped core

`v1.0.2` and `ext-v4` distinguish `authority_by_publication` (28),
`janaf_direct_liquid` (9) and `partial_non_janaf_direct_liquid` (1). The
Both packs use those same three classes for their 38 core rows. `ext-v4` adds
five `extension-compound-thermo` rows in its separate S/P extension vocabulary;
those are not a different level of detail for the published core. The ext-v1,
ext-v2 and ext-v3 records are retired and no longer shipped; see
`docs/datapack-format.md` for the retirement record. A reader should consult
each row's provenance class to distinguish FC87-table-derived from JANAF-derived
rows.

An audit of the two shipped packs found **zero reverse-engineered rows**.

### 8. The model spec is not ported

`docs/spec.md` — the model spec — has not been ported.

### 9. Identity

`CITATION.cff` has no DOI and no ORCID. Zenodo wiring would give a citable DOI.
No PyPI publish workflow yet.

### 10. Validation decks

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

- **Apache-2.0 for code; CC-BY-4.0 for the project's own arrangement, provenance
  text, and derived tables.** NIST-JANAF records are public data, and published
  coefficients and digitized values are cited facts, not a grant of rights to
  publishers' tables or figures. Split because the halves have different origins;
  `NOTICE` carries the attribution chain.
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
