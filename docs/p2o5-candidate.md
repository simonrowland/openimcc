# P2O5 full-component candidate

`load_profile("p2o5-candidate")` selects the opt-in
`imcc-sf04-d066-p2o5-candidate-v1.json` melt pack. It appends P2O5 to the
D066 v4 parent list, keeps the 38 D066 rows unchanged, and adds Mg3(PO4)2 and
Ca3(PO4)2. Its declared solve domain is 1700–2300 K. The default profile,
`DEFAULT-COMBINATION`, and published packs remain unchanged.

The parent is Jung and Hudon's P2O5(l) standard at 1 bar, reused from the
existing gas and condensate tables. Its source range is 1200–3000 K. The
liquid parent standard itself is not reconstructed from JANAF O-094. Jung and
Hudon publish no numeric uncertainty for this standard, which remains a typed
gap.

## Phosphate rows

The source transcriptions are in `data-src/p2o5-candidate/thermochemistry.json`.
Mg3(PO4)2 uses NIST-JANAF tables Mg-038 (crystal), Mg-039 (liquid), and Mg-040
(transition). The Mg-039 liquid is printed through the fit interval and the
1621 K fusion marker is tabulated. The candidate re-references the printed
liquid to D066 MgO(l) and Jung–Hudon P2O5(l), then fits `log10 K = A + B/T`
over 1001 evenly spaced points from 1700 to 2300 K.

The Mg row uses JANAF formation Gibbs functions throughout. Jung–Hudon reports
`P2(g) + 5/2 O2(g) → P2O5(l)`, so
`ΔGf(P2O5,l) = ΔGr°(Jung–Hudon, 1 bar) + ΔGf(P2,g)`; JANAF has
`ΔGf(O2,g) = 0`. The inferred 1 atm equation is converted to 1 bar with
`Δνgas = −3.5`. The Mg elemental reference changes from liquid to gas near
1363 K, below this fit interval. Mg3(PO4)2, MgO, and P2O5 all use JANAF
temperature-dependent elemental references, so the three Mg terms cancel at
each temperature. Formation-route and absolute-entropy checks agree at
1700, 2000, and 2300 K.

The earlier Ca fit continued the R&H formation Gibbs function with
`ΔCp_form = 41 J mol⁻¹ K⁻¹` and subtracted JANAF CaO(l) formation Gibbs. That
mixes elemental references above calcium's boiling point: JANAF CaO changes
from Ca(l) to Ca(g) near 1757 K, while the continued R&H formation function
does not carry the same switch. Three unmatched calcium references caused a
spurious reaction entropy and flattened the temperature dependence.

The Ca row now uses absolute Gibbs functions. At the printed 1600 K Whitlockite
anchor, add `3G(Ca,l) + G(P2,g) + 4G(O2,g)` to the R&H formation Gibbs value,
then set `H = G + T S` using its printed absolute entropy. Continue absolute
H and S above 1600 K with the printed crystal `Cp = 330.54 J mol⁻¹ K⁻¹`:
`H(T) = H0 + Cp(T−T0)` and `S(T) = S0 + Cp ln(T/T0)`. The CaO(l) and existing
Jung–Hudon P2O5(l) parents use their absolute `H−TS` functions. At 2083.15 K,
the corrected Ca liquid association has `ΔH ≈ −644.0 kJ/mol` and
`ΔS ≈ +55.6 J mol⁻¹ K⁻¹`; the Mg liquid row gives `ΔH ≈ −414.6 kJ/mol`.

For `3 MO(l) + P2O5(l) = M3(PO4)2(l)`, both routes balance the same three metal,
two phosphorus, and eight oxygen atoms. In the Mg formation route,
`ΔG°rxn = ΔG°f(product) − 3ΔG°f(MO) − ΔG°f(P2O5)` cancels the common JANAF
element references. For Ca, the 1600 K R&H product is first converted to
absolute Gibbs energy and the elemental references are selected at each
temperature; the reaction's elemental terms then cancel without crossing
reference conventions. Gibbs energies are in J/mol; dividing by `R T ln 10`
produces dimensionless `log10 K`. The independent 298.15 K crystal enthalpy
check is
`−3745.098 − 3(−601.241) − (−3009.936/2) = −436.407 kJ/mol`, consistent with
the rounded approximately −435 kJ/mol value. JANAF publishes no numeric
uncertainty for these entries; the source-precision cycle and the pack's
Mg `A+B/T` enthalpy path are both asserted in tests.

The corrected Mg3(PO4)2 refit has a maximum absolute residual of 0.070911 dex
and RMS residual of 0.030373 dex. Its maximum exceeds the 0.01 dex reporting
threshold and is flagged in the pack. The Ca3(PO4)2 fit residuals are below
2×10⁻¹⁴ dex.

| Associate | A | B (K) | log₁₀K at 1700 / 2000 / 2300 K | max residual (dex) | RMS residual (dex) |
|---|---:|---:|---:|---:|---:|
| Mg3(PO4)2 | 1.696026 | 21657.641 | 14.436 / 12.525 / 11.112 | 0.070911 | 0.030373 |
| Ca3(PO4)2 | 2.902103 | 33640.648 | 22.691 / 19.722 / 17.528 | <2×10⁻¹⁴ | <8×10⁻¹⁵ |

Ca3(PO4)2 uses the Whitlockite rows transcribed from Robie and Hemingway,
*USGS Bulletin 2131*, p. 300, Table e1 ([public-domain report](https://pubs.usgs.gov/bul/2131/report.pdf)).
The rhombohedral table covers 298.15–1373 K; the monoclinic rows cover
1373–1600 K. Welch and Gutt's primary CaO–P2O5 phase-diagram study reports the
congruent melting marker at 1810 °C, or 2083.15 K
([primary paper](https://doi.org/10.1039/JR9610004442)).

Above 1600 K, the absolute Whitlockite crystal Cp is continued at its printed
1600 K value, 330.54 J mol⁻¹ K⁻¹, after its formation-G anchor is converted to
absolute G with the JANAF elemental references at 1600 K. This handles the
later Ca(l)-to-Ca(g) reference change without extending a formation function
across it. Fusion entropy uses the class-C estimate
`13 × 8.297530569 = 107.8679 J mol⁻¹ K⁻¹`. The liquid heat capacity at the
fusion marker is the additive CaO(l) and P2O5(l) parent value,
320.4027 J mol⁻¹ K⁻¹. These are provisional assumptions. The fitted liquid
association also extends below the melting marker as a supercooled-liquid
branch. The Ca3(PO4)2 maximum residual is below 2×10⁻¹⁴ dex and RMS residual
is below 8×10⁻¹⁵ dex.

Both rows have a 1700–2300 K candidate fit domain and retain their source
windows and flags independently:

| Associate | Source or demonstrated window | Fit domain | Construction flags |
|---|---:|---:|---|
| Mg3(PO4)2 | 1621–3000 K liquid | 1700–2300 K | Printed JANAF liquid; measured 1621 K fusion |
| Ca3(PO4)2 | 298.15–1600 K crystal | 1700–2300 K | Constant-Cp continuation above 1600 K; provisional class-C fusion entropy; provisional additive liquid Cp; supercooled branch below 2083.15 K |

## KREEP check and P-gas gate

The diagnostic composition is a representative KREEP-style analysis in wt%:
SiO2 50.8, MgO 9.34, FeO 10.4, CaO 9.6, Al2O3 15.0, TiO2 2.06, Na2O 0.77,
K2O 0.61, MnO 0.16, and P2O5 1.5. It gives
`x(P2O5) = 0.006614`. With the corrected Ca reaction, `a(P2O5)` rises by
about 2.32 orders of magnitude from 1700 to 2300 K. It remains 6.0–8.4 orders
below the mole fraction, as expected for acidic P2O5 in this basic, Ca-rich
silicate melt.

| T (K) | `a(P2O5)` | `log10(a/x)` | Exposed Suito helper `a(P2O5)` |
|---:|---:|---:|---:|
| 1700 | 2.8802×10⁻¹¹ | −8.361 | 9.5633×10⁻¹² |
| 2000 | 6.8340×10⁻¹⁰ | −6.986 | 4.2826×10⁻¹⁰ |
| 2300 | 6.0260×10⁻⁹ | −6.040 | 7.1136×10⁻⁹ |

The existing Suito and Inoue (1984) Eq. 14 helper is comparison-only; neither
associate was fitted to it ([paper](https://doi.org/10.2355/isijinternational1966.24.40)).
At 1.5 wt% P2O5, the comparison lies in the helper's 1–4 wt% calibration gap,
and all three temperatures are outside its 1823–1923 K main temperature
window. No held-out phosphate-capacity data from the identified set are
transcribed here, so that check is **awaiting data**.

The activity screen alone does not validate P-bearing volatility. Only Mg and
Ca phosphates are represented, and no held-out phosphate-capacity or volatility
set has scored this candidate. The pack therefore sets
`_p2o5_channel_status = "associates_incomplete"`: P, P2, P4, PO, PO2, P4O6, and
P4O10 are omitted from default gas results with reason code
`p2o5_associates_incomplete`. Explicit requests raise the typed
`ImccP2O5AssociatesIncompleteError`.

For the requested fixed-composition diagnostic, the following formal
mass-action pressures use the candidate's P2O5 activity and gas standard
states. They are not returned by the runtime profile while the gate is active.
At 2000 K, P2 alone exceeds 1 bar; at 2300 K, P and PO also exceed 1 bar and
P2 reaches about 1.8×10⁸ bar. The fixed-composition gas contract treats
pressures above 1 bar as complete volatilisation, so these values do not
support a physical volatility prediction and reinforce keeping all P-bearing
channels gated.

| T (K) | P (bar) | P2 (bar) | PO (bar) | PO2 (bar) |
|---:|---:|---:|---:|---:|
| 1700 | 9.037×10⁻⁹ | 9.319×10⁻⁸ | 1.030×10⁻⁵ | 1.577×10⁻⁵ |
| 2000 | 3.267×10⁻³ | 6.223×10¹ | 9.692×10⁻² | 6.885×10⁻³ |
| 2300 | 3.884×10¹ | 1.761×10⁸ | 7.721×10¹ | 5.683×10⁻¹ |

Before this candidate, P2O5 was an unmodelled trace: at 1.5 wt% its bias bound
made every parent activity refuse, and 32 of 41 default gas channels were
omitted, including SiO, Na, K, Fe, Mg, Al, Ca, Ti, and Mn. With this profile,
all parent activities return, and the default gas path serves the 34
non-phosphorus channels, including those majors. The seven P-bearing channels
remain omitted until held-out validation.

## MnO graft

The MnO association graft accepts a candidate only when the first 38 rows and
the common-parent liquid standards match D066 v4 by content digest. This
allows both Sr and P2O5 candidate profiles to solve MnO-bearing compositions.
When the added parent is zero, the candidate's MnO activity matches the D066
v4 result.

## Provisional assumptions and typed gaps

The provisional Ca3(PO4)2 set is the constant-Cp crystal continuation above
1600 K, class-C fusion entropy, additive parent liquid Cp, and the
supercooled-liquid branch below 2083.15 K. The Mg3(PO4)2 liquid and fusion
inputs are printed JANAF values. The Jung–Hudon parent has no published
numeric uncertainty; no liquid volume is available for a pressure correction.

The candidate deliberately stops at these two associates. Typed gaps are:

- Ca2P2O7;
- AlPO4 melting data;
- Na3PO4 and NaPO3 thermochemistry;
- potassium phosphate thermochemistry;
- Fe3(PO4)2 thermochemistry;
- apatite, which additionally requires F, Cl, or OH;
- numeric uncertainty and a liquid volume for the Jung–Hudon P2O5(l) standard;
- held-out phosphate capacity and volatility comparisons.
