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

The initial fit mixed JANAF formation Gibbs energies for the phosphates with
enthalpy-anchored absolute Gibbs energies for the oxide parents and Jung–Hudon
P2O5(l), producing impossible negative `log10 K` values near −62 at 2000 K.
The correction puts every term on one JANAF elemental reference. Jung–Hudon
reports `P2(g) + 5/2 O2(g) → P2O5(l)`, so
`ΔGf(P2O5,l) = ΔGr°(Jung–Hudon, 1 bar) + ΔGf(P2,g)`; JANAF has
`ΔGf(O2,g) = 0`. The inferred 1 atm equation is converted to 1 bar with
`Δνgas = −3.5`.

For `3 MO(l) + P2O5(l) = M3(PO4)2(l)`, the product contains the same three
metal, two phosphorus, and eight oxygen atoms as the parents. Therefore
`ΔG°rxn = ΔG°f(product) − 3ΔG°f(MO) − ΔG°f(P2O5)` cancels all elemental
reference terms, including phosphorus. Each Gibbs energy is in J/mol;
dividing by `R T ln 10` produces dimensionless `log10 K`. The independent
298.15 K crystal enthalpy check is
`−3745.098 − 3(−601.241) − (−3009.936/2) = −436.407 kJ/mol`, consistent with
the review's rounded approximately −435 kJ/mol value. JANAF publishes no
numeric uncertainty for these entries; the source-precision cycle is asserted
in a test.

The corrected Mg3(PO4)2 refit has a maximum absolute residual of 0.070911 dex
and RMS residual of 0.030373 dex. Its maximum exceeds the 0.01 dex reporting
threshold and is flagged in the pack. The Ca3(PO4)2 fit residuals are below
2×10⁻¹⁴ dex.

| Associate | A | B (K) | log₁₀K at 1700 / 2000 / 2300 K | max residual (dex) | RMS residual (dex) |
|---|---:|---:|---:|---:|---:|
| Mg3(PO4)2 | 1.696026 | 21657.641 | 14.436 / 12.525 / 11.112 | 0.070911 | 0.030373 |
| Ca3(PO4)2 | 15.585333 | 11143.826 | 22.141 / 21.157 / 20.430 | <2×10⁻¹⁴ | <8×10⁻¹⁵ |

Ca3(PO4)2 uses the Whitlockite rows transcribed from Robie and Hemingway,
*USGS Bulletin 2131*, p. 300, Table e1 ([public-domain report](https://pubs.usgs.gov/bul/2131/report.pdf)).
The rhombohedral table covers 298.15–1373 K; the monoclinic rows cover
1373–1600 K. Welch and Gutt's primary CaO–P2O5 phase-diagram study reports the
congruent melting marker at 1810 °C, or 2083.15 K
([primary paper](https://doi.org/10.1039/JR9610004442)).

Above 1600 K, the formation heat-capacity difference is continued at
41 J mol⁻¹ K⁻¹, from the 4.1 kJ/mol formation-enthalpy change between the
1500 and 1600 K printed rows. Fusion entropy uses the class-C estimate
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
`x(P2O5) = 0.006614`. The corrected two-associate result gives an activity
7.9–8.8 orders of magnitude below the mole fraction, as expected for acidic
P2O5 in this basic, Ca-rich silicate melt. This passes the physical plausibility
screen, which would refuse P channels if `a(P2O5)` came within 3 dex of
`x(P2O5)`.

| T (K) | `a(P2O5)` | `log10(a/x)` | Exposed Suito helper `a(P2O5)` |
|---:|---:|---:|---:|
| 1700 | 8.1748×10⁻¹¹ | −7.908 | 9.5633×10⁻¹² |
| 2000 | 3.1334×10⁻¹¹ | −8.324 | 4.2826×10⁻¹⁰ |
| 2300 | 9.7488×10⁻¹² | −8.831 | 7.1136×10⁻⁹ |

The existing Suito and Inoue (1984) Eq. 14 helper is comparison-only; neither
associate was fitted to it ([paper](https://doi.org/10.2355/isijinternational1966.24.40)).
At 1.5 wt% P2O5, the comparison lies in the helper's 1–4 wt% calibration gap,
and all three temperatures are outside its 1823–1923 K main temperature
window. No held-out phosphate-capacity data from the identified set are
transcribed here, so that check is **awaiting data**.

The pressures below are mass-action results from the candidate activity at
`fO2 = 1×10⁻¹⁰ bar` and are returned by this profile. Above 1 bar, the gas
contract marks the fixed-composition result as complete volatilisation, so
these high-temperature values are not composition-preserving volatility
predictions.

| T (K) | P (bar) | PO (bar) | PO2 (bar) | P2 (bar) |
|---:|---:|---:|---:|---:|
| 1700 | 1.5225×10⁻⁸ | 1.7351×10⁻⁵ | 2.6573×10⁻⁵ | 2.6451×10⁻⁷ |
| 2000 | 6.9954×10⁻⁴ | 0.020753 | 0.0014742 | 2.8534 |
| 2300 | 1.5622 | 3.1055 | 0.022859 | 284846 |

The corrected activity passes the requested channel-plausibility gate, so
P-bearing channels ship with the associate construction flags, including the
provisional Ca3(PO4)2 terms. The pressure table retains the fixed-composition
complete-volatilisation limitation above 1 bar.

Before this candidate, P2O5 was an unmodelled trace: at 1.5 wt% its bias bound
made every parent activity refuse, and 32 of 41 default gas channels were
omitted, including SiO, Na, K, Fe, Mg, Al, Ca, Ti, and Mn. With this profile,
the melt solver returns all parent activities, including P2O5, and the default
gas path returns all 41 channels, including those majors and the P-bearing
channels.

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
