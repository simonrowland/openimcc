# Changelog

## Unreleased

- Added optional La(g) and LaO(g) channels with caller-supplied La2O3(l)
  activity. Konings Part 1 Table 42 and Part 2 Tables 88–89 supply the gas and
  phase-chain functions; the printed constant-Cp liquid branch is flagged as
  supercooled below its 2577 K fusion marker.
- Added optional Hf(g) with caller-supplied HfO2(l) activity and opt-in Hf+
  and Hf- JANAF rows. Pankratz Bulletin 672 p. 188 supplies HfO2(cr), extended
  by the labelled constant-Cp rule and used in the tier-4 liquid construction.
  Hong et al. (2018) supplies the measured HfO2 fusion enthalpy and melting point.
  The Hf gas/parent source-scale difference is banded; no HfO(g) or HfO2(g)
  table was found in the checked compilations.
- Added optional Th, ThO, and ThO2 channels with caller-supplied ThO2(l)
  activity. Pankratz Bulletin 672 supplies the gas rows and ThO2 crystal;
  the liquid standard is a flagged tier-4 construction using the reviewed
  fusion inputs and continued crystal heat capacity. The opt-in Th+ row uses
  the Burcat card based on Gurvich (1992); its NIST ASD Saha source offset is
  recorded and banded in provenance.
- Added optional Sc, ScO, ScO2, U, UO, UO2, and UO3 channels with caller-supplied
  Sc2O3(l) and UO2(l) activities, plus opt-in Sc+. NASA Glenn cards supply both
  gas and liquid functions. Sc2O3(l) is fitted directly from its liquid card
  above 2762 K and uses labelled constant-Cp continuation below it; UO2(l) uses
  the labelled continuation from its 3123 K liquid card throughout 1200–3000 K.
  The Sc2O3 liquid branch differs from Bulletin 672 by its apparent-G difference,
  including both formation anchors; the 13.792 kJ/mol fusion-enthalpy gap (140.792 kJ/mol
  from Bulletin 672 versus 126.999 kJ/mol from the NASA cards) is recorded separately,
  not as a Gibbs band. The NASA liquid-versus-crystal apparent-G separation is recorded
  as a comparison. U(g)'s apparent-G gap
  against Bulletin 672 is disclosed from both source records. No evaluated U+ thermo
  function was located.
- Added computed source-gap disclosures for the Na2O(g) parent and gas anchors
  and the K2O(g) gas anchor. The compilation-pair check now uses each runtime
  liquid parent and records the remaining named exceptions with reasons.
- Added optional Ba, Sr, and Zr gas channels with caller-supplied BaO(l),
  SrO(l), and ZrO2(l) activities, plus opt-in Ba+, Sr+, Zr+, and JANAF Zr-.
  JANAF supplies the gas and liquid branches; the three liquid fits reproduce
  their printed nodes within the 10 J/mol gate. Ba-035's glass cells below
  1600.010 K and printed supercooled-liquid cells from 1700 through 2200 K are
  retained. NASA comparisons are
  recorded on G_app, including the ZrO(g) disagreement; JANAF remains selected.
- Added optional Mo, MoO, MoO2, MoO3, and (MoO3)2–5 gas channels with
  caller-supplied MoO3(l) activity, plus opt-in Mo+ and Mo−. JANAF supplies the
  monomers, ions, and parent liquid through 2000 K; NASA Glenn supplies the
  oxide clusters JANAF does not list. The MoO3(l) continuation above 2000 K is
  labelled and is not source data.
- Added optional W, WO, WO2, WO3, (WO3)2–5, and W3O8 gas channels with
  caller-supplied WO3(l) activity, plus opt-in W+ and W−. JANAF supplies the
  measured gas and liquid rows except (WO3)5, which uses NASA Glenn. O-066's
  printed supercooled liquid cells below its 1745 K marker are retained as
  source data.
- Disclosed the NASA-cluster/JANAF-parent card-G scale gaps in provenance
  bands per cluster formula. NG-1701 and NG-2013 liquid comparisons report the
  fitter's card-G value separately from the header-anchored comparison; the
  JANAF MoO3(l) and WO3(l) parents remain selected. The MoO3(l) held-Cp
  continuation now carries the existing continuation envelope.
- Added Mo+ and W+ to the NIST-level Saha audit, and Mo− and W− to the
  NIST-affinity attachment audit. Measured JANAF source offsets are pinned in
  tests and recorded in provenance.
- Added optional Ag(g) with caller-supplied Ag2O(l) activity and opt-in Ag+ and Ag-.
  NASA Glenn supplies the atomic and cation cards; Bulletin 672's independent
  Ag(g) row is compared by apparent Gibbs energy through 2000 K. Ag2O(l) is a
  flagged tier-4 construction from the crystal table, which marks decomposition
  near 500 K and provides no liquid function. Its construction band extends the
  +800 K-calibrated tier-3 slope about 1500–2500 K past that decomposition
  region. The oxide-parent convention does
  not represent silver dissolved as metal in reduced melts; Ag2(g) and AgO(g)
  had no source table in the checked records.
- Added optional Sb, Sb2, Sb4, and Sb4O6 gas channels with caller-supplied
  Sb2O3(l) activity. Bulletin 672 supplies the gas and crystal data; the
  hypothetical liquid parent is a tier-4 construction because its fusion
  enthalpy and liquid function are unavailable. Constant-Cp continuation above
  each gas row's source limit is flagged and banded. SbO(g) and Sb+(g) were not
  located in the checked sources.
- Added optional As, As2, As4, and As4O6 gas channels with caller-supplied
  As2O3(l) activity. Pankratz Bulletin 672 supplies the gas channels and the
  As2O3 crystal function. As2O3(l) is a flagged tier-4 construction using the
  repository's tier-3 liquid_from_solid estimate and leave-one-out band.
  As(g), As2(g), and As4(g) continue by constant Cp above 1200 K; As4O6(g)
  continues above 1000 K. No AsO(g) or As+(g) source was located in Bulletin
  672, JANAF, NASA Glenn/CEA, NIST WebBook, or Gurvich/IVTAN.
- Added optional Se, Se2, SeO, and SeO2 gas channels with caller-supplied SeO2(l)
  activity. Pankratz Bulletin 672 supplies the gas rows. SeO2(l) is a flagged
  tier-4 construction using the OECD/NEA reviewed fusion inputs and the
  documented crystal-Cp carry-over estimate; its crystal Cp equation extends
  above the printed 602 K range. The Se2 entropy transcription was checked
  against the published Cp equation. SeO(g) uses the Bulletin 672 gas table;
  Se+(g) had no source in Bulletin 672, JANAF, NASA Glenn/CEA, NIST WebBook,
  or Gurvich/IVTAN. SeO2(g)'s 1000–1500 K nodes are transcribed from the
  printed reaction table.
- Added optional Te, Te2, and TeO2 gas channels with caller-supplied TeO2(l)
  activity. Pankratz Bulletin 672 supplies the selected rows; the TeO2(l)
  liquid branch is fitted through its printed nodes, with a flagged, banded
  constant-Cp continuation above them. TeO(g) and Te+(g) had no source in the
  checked Bulletin 672, JANAF, NASA Glenn/CEA, NIST WebBook, or Gurvich/IVTAN
  routes.
- Refit the elemental iron reference functions directly to the JANAF phase
  tables. The earlier Fe(l) row carried a 76–79 J/mol fusion-anchor offset,
  and the earlier single-polynomial Fe(cr) row missed its delta-phase source
  nodes by −63 to +77 J/mol; the direct fits remove the anchor correction
  and retain a 2.345 J/mol Fe(cr)/Fe(l)
  seam, within the 10 J/mol fit gate. Fe(cr) now separates alpha, gamma, and
  delta at 1184 and 1665 K, with an additional 900 K alpha fit split for the
  source-node residual gate. The recorded Fe reference bands are fit/seam
  residuals, not thermochemical accuracy estimates.

  Closed Fe-only inventory `{Fe: 1, O: 1}` changes from the prior pack are:

  | T (K) | Δλ | Δmetal (mol) | ΔFeO (mol) | ΔFeO1.5 (mol) |
  |---:|---:|---:|---:|---:|
  | 1700 | -0.008884 | -1.9345e-5 | +5.8036e-5 | -3.8690e-5 |
  | 2200 | +0.008404 | +2.6463e-5 | -7.9388e-5 | +5.2925e-5 |
  | 3000 | +0.006080 | +3.6994e-5 | -1.1098e-4 | +7.3987e-5 |

  At imposed `λ = -4`, the Fe-only species and metal outputs are unchanged at
  these temperatures. The FeO(l) reaction row has Δln K′ values +0.004482,
  -0.004258, and -0.003122 at 1700, 2200, and 3000 K; its largest absolute
  change is 0.004482 at 1700 K.

  For the basalt cation inventory used by the redox regression fixture, the
  closed oxygen inventory is 0.050 mol below the parent-oxide oxygen total.
  The imposed comparison fixes λ to the prior-pack closed result at each
  temperature. Changes are:

  | T (K) | Mode | Δλ | Δmetal (mol) | ΔFeO (mol) | ΔFeO1.5 (mol) | ΔO (mol) |
  |---:|---|---:|---:|---:|---:|---:|
  | 1700 | closed | -0.008952 | -4.7777e-7 | +1.4333e-6 | -9.5555e-7 | — |
  | 1700 | imposed | 0 | -0.0011766 | +0.0011743 | +2.2570e-6 | +0.0011777 |
  | 2200 | closed | +0.008486 | +1.2012e-6 | -3.6035e-6 | +2.4023e-6 | — |
  | 2200 | imposed | 0 | +0.0012060 | -0.0012000 | -5.9622e-6 | -0.0012090 |

  Across the default redox test fixtures, the largest absolute change in
  log10 fO2 is 0.012012 at 3000 K in a closed manganese-bearing case. Its
  reaction evidence domain is unchanged, so the result remains flagged as an
  extrapolation. All reaction evidence domains are unchanged.

- Added the Mn evaporation channel with caller-supplied MnO activity. Mn(g) and
  Mn+(g) use JANAF; MnO(g) uses IVTAN Table 1436. MnO(l) is generated from the
  same `liquid_from_solid` construction and input record as the redox pack,
  using the TSIV Table 1435 crystal function and measured 2120 ± 20 K melting
  temperature. Its 1673–2000 K supercooled runtime band carries the tier-3
  fusion-input uncertainty; the fitted Gibbs residual is 0.186 J/mol. Mn+
  remains opt-in, and existing coefficients and default gas results are
  unchanged.

- Added optional Zn, Cd and Bi gas channels with caller-supplied ZnO, CdO and
  Bi2O3 activities. JANAF Zn(g), Zn+(g) and Zn-(g) remain selected; source
  G_app comparisons are recorded for the NASA Zn and Zn+ cards. The JANAF
  catalogue has no Cd table, so NASA Glenn supplies Cd(g) and Cd+. Published
  tables supply ZnO(g), Bi(g), Bi2(g), BiO(g) and estimated CdO(g). ZnO(l) and
  CdO(l) are flagged tier-4 `liquid_from_solid` constructions with fitted
  crystal-Cp extensions and measured input bands; Bi2O3(l) uses its printed
  Pankratz liquid branch and a flagged constant-Cp upper continuation. Added
  opt-in Zn+, Zn-, Cd+ and Bi+ channels. All new condensate fits pass the
  10 J/mol source-node gate, the computed C3 ion shares remain below 1e-4, and
  default gas results and existing coefficient rows are unchanged. Source
  hashes, locators, G_app differences, bands and residuals are in the
  provenance ledger.

- Refit the seven JANAF constant-Cp liquid continuations with Cp-constrained
  polynomials. Cr2O3 and SiO2 use two continuation segments; Al2O3, MgO and CaO
  use three. Across all segments, the largest Gibbs residual is 1.618 J/mol,
  the largest Cp deviation is 1.043 J/(mol K), and the largest seam mismatch is
  4.322 J/mol in G and 0.194 J/(mol K) in S. The high-temperature source fits
  and all other condensate rows are unchanged.
  The maximum old-to-new parent Gibbs differences and their largest
  per-metal-atom pressure equivalents are:

  | Parent | Max absolute ΔG (J/mol) | Max absolute Δlog10 p per metal atom |
  |---|---:|---:|
  | TiO2(l) | 1.102 | 0.0000464 |
  | Cr2O3(l) | 2.757 | 0.0000551 |
  | V2O3(l) | 1.376 | 0.0000266 |
  | SiO2(l) | 0.868 | 0.0000361 |
  | Al2O3(l) | 36.748 | 0.0006082 |
  | MgO(l) | 5.670 | 0.0002029 |
  | CaO(l) | 2.949 | 0.0001094 |

  At fixed caller activities and oxygen fugacity, each gas channel using a
  parent has `Δlog10(p) / n_metal = (G_new − G_old) / (n_parent_metal R T ln 10)`.
  Values below are signed `log10(p_new/p_old)` per metal atom; they apply to
  Ti/TiO/TiO2, Cr/CrO/CrO2/CrO3, V/VO/VO2, Si/SiO/SiO2/Si2/Si3,
  Al/AlO/AlO2/Al2O/Al2O2/Al2, Mg/MgO, and Ca/CaO, grouped by parent. A zero
  means the continuation is no longer selected at that temperature.

  | Parent | 1200 K | 1400 K | 1600 K | 1933 K |
  |---|---:|---:|---:|---:|
  | TiO2(l) | −0.00000083 | −0.00000302 | 0 | 0 |
  | Cr2O3(l) | −0.00001039 | −0.00002865 | +0.00002679 | 0 |
  | V2O3(l) | −0.00001083 | +0.00002319 | −0.00001298 | 0 |
  | SiO2(l) | −0.00000421 | −0.00002088 | +0.00001794 | 0 |
  | Al2O3(l) | −0.00040938 | +0.00031001 | −0.00038286 | +0.00026990 |
  | MgO(l) | −0.00008941 | +0.00000598 | −0.00009668 | +0.00005784 |
  | CaO(l) | −0.00004768 | +0.00000201 | −0.00005056 | +0.00002944 |

  Default oxygen-balance numeric and result digests were re-pinned for cases
  0–31, 33, 36–37, 39–42 and 45–49 because those cases select a changed parent
  continuation. Cases 32, 34–35, 38 and 43–44 retain their pins because they
  select unchanged high-temperature rows. The published-pack digests are
  unchanged. The derived AlO2−/AlO pressure-ratio pin at 2100 K and
  fO2 = 1e-12 was updated from 5.051482184349738 to 5.051477362451509 as the
  Al2O3 continuation changes the default parent Gibbs energy. The derived C3
  status pins that moved are:

  | Status value | Old | New |
  |---|---:|---:|
  | Si isolated bound | 635.770885325284 | 635.773963217761 |
  | Ca isolated bound | 12.157887783887084 | 12.158555162056784 |
  | Al isolated bound | 30240.015809492263 | 30254.271715108043 |
  | Ti isolated bound | 143063233.9706761 | 143063370.34002876 |
  | Cr isolated bound | 0.0675715822588812 | 0.0675723903092475 |
  | V isolated bound | 244.3406917318666 | 244.34373945658453 |
  | Ti, V, Nb electron pressure | 4.969104777497582e-06 | 4.96912101530768e-06 |

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
  Cs+, Sn+, Cs2O+, Cs− and Cu− channels. JANAF liquid cells are used for
  Cu2O(l) through 2000 K, with a NASA H/S-increment tail; NASA liquid cards
  supply Cs2O(l) and SnO(l), with a labelled SnO constant-Cp continuation from
  1250 K to 1200 K. SnO is selected over the later-starting SnO2(l) card.
  Cs exceeds the recorded C3 ion-share limit at the caller activity screen;
  Cu and Sn remain below it. Default neutral results and existing coefficient
  rows are unchanged.
  The published NASA In+ card is declined after its fitted Kion/Saha ratios
  are 1.510e28 at 1500 K and 9.143e16 at 2500 K. The source card is retained
  without adjustment, and neutral In channels remain available.
  Cu− is available on the opt-in ion path: corrected attachment-Saha ratios
  against [NIST's 1.23578 eV copper affinity](https://webbook.nist.gov/cgi/cbook.cgi?ID=C7440508&Mask=1020) are 0.941424 and 0.958988 at
  1500 and 2500 K, consistent with the older Cu-007 evaluation within 10%.
  Cu+ remains declined because the JANAF Cu-006 entropy gives fitted/Saha
  ratios of 1666.15 and 1656.61 at those temperatures.
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
  K2O(l) table; the default K2O(l) row is constructed from the K-012 crystal,
  while `sf04-published` retains LAM1984.
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
- Wired the Ni and Co channels to public TSIV liquid parents. NiO(l) and CoO(l)
  use their estimated phase-paired functions from Volume IV Tables 2026 and
  1334, with labelled liquid-function continuations below their 2228 K and
  2090 K source limits. Added optional NiO(g) from the tier-2 NASA Glenn card
  with a measured Bulletin 672 disagreement band, and CoO(g) from TSIV Table
  1335 (accuracy class 4-F). Opt-in Ni+/Ni− and Co+/Co− use JANAF; parent
  activities are supplied by callers.
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

- Changed the default K2O(l) reference from LAM1984 to a construction from the
  JANAF K-012 crystal, using `liquid_from_solid` inputs in
  `data-src/liquid-from-solid-inputs.json`: fusion temperature 1013 K (band
  919–1190 K), fusion entropy 33.74 ± 0.21 J/(mol K), and liquid heat capacity
  97.75 ± 5.5 J/(mol K). The combined selected-tier Gibbs band samples are
  7.420, 14.626, and 35.823 kJ/mol at 1013, 1313, and 1813 K. The apparent
  Gibbs energies are:

  | T (K) | G_app (kJ/mol) |
  | ---: | ---: |
  | 1200 | −555.309542 |
  | 1500 | −637.722336 |
  | 2000 | −787.694272 |
  | 3000 | −1122.178019 |

  Relative to LAM1984 this lowers K pressure by about 0.35 dex per K atom at
  1200–1500 K and 0.26 dex at 3000 K. The `sf04-published` pack retains its
  LAM1984 row. Potassium complex constants are being re-derived against this
  reference in a separate change; coupled gas and oxygen balances use the new
  parent Gibbs energy with the current constants. Default gas-binding and
  oxygen-balance hashes and golden values were re-pinned for cases containing
  K2O; SiO results are unchanged. The three Hastie K pressure residuals are
  now −1.20 to −1.26 dex.
- Python 3.12 is now the minimum supported version. Python 3.11 is no longer
  tested.
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
