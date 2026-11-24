# P2O5 full-component candidate

`load_profile("p2o5-candidate")` selects the opt-in
`imcc-sf04-d066-p2o5-candidate-v1.json` melt pack. It appends P2O5 to the
D066 v4 parent list, keeps the 38 D066 rows unchanged, and adds Mg3(PO4)2 and
Ca3(PO4)2. Its declared solve domain is 1700–2300 K. The default profile,
`DEFAULT-COMBINATION`, and published packs remain unchanged.

The parent is Jung and Hudon's P2O5(l) standard at 1 bar, reused from the
existing gas and condensate tables. Its source range is 1200–3000 K. The
candidate does not reconstruct the parent from JANAF O-094. Jung and Hudon
publish no numeric uncertainty for this standard, which remains a typed gap.

## Phosphate rows

The source transcriptions are in `data-src/p2o5-candidate/thermochemistry.json`.
Mg3(PO4)2 uses NIST-JANAF tables Mg-038 (crystal), Mg-039 (liquid), and Mg-040
(transition). The Mg-039 liquid is printed through the fit interval and the
1621 K fusion marker is tabulated. The candidate re-references the printed
liquid to D066 MgO(l) and Jung–Hudon P2O5(l), then fits `log10 K = A + B/T`
over 1001 evenly spaced points from 1700 to 2300 K.

For both associates, the formation-from-oxides reaction is
`3 MO(l) + P2O5(l) = M3(PO4)2(l)`. The product and P2O5 parent each contain two
phosphorus atoms. With a shared elemental reference,
`ΔG°rxn = ΔG°f(product) − 3ΔG°f(MO) − ΔG°f(P2O5)` cancels the phosphorus
reference term exactly. Each Gibbs energy is in J/mol; dividing by `R T ln 10`
produces dimensionless `log10 K`. This is recorded in the builder comment and
pack provenance.

The Mg3(PO4)2 refit has a maximum absolute residual of 0.048582 dex and RMS
residual of 0.019274 dex. The maximum exceeds the 0.01 dex reporting threshold
and is flagged in the pack. It is the residual from representing the printed
liquid Gibbs curve with `A + B/T`.

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
branch. The Ca3(PO4)2 maximum and RMS fit residuals are both below
4×10⁻¹⁴ dex.

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
`x(P2O5) = 0.006614`. The two-associate result gives an activity slightly
above the mole fraction across the full candidate interval, whereas basic
silicate melts should suppress the activity of this acidic oxide many orders
below its mole fraction.

| T (K) | `a(P2O5)` | `log10(a/x)` | Exposed Suito helper `a(P2O5)` |
|---:|---:|---:|---:|
| 1700 | 1.2543×10⁻² | +0.278 | 9.5633×10⁻¹² |
| 2000 | 1.2863×10⁻² | +0.289 | 4.2826×10⁻¹⁰ |
| 2300 | 1.3165×10⁻² | +0.299 | 7.1136×10⁻⁹ |

The existing Suito and Inoue (1984) Eq. 14 helper is only a comparison, with
no fitting ([paper](https://doi.org/10.2355/isijinternational1966.24.40)).
The 1.5 wt% point falls in its 1–4 wt% calibration gap; 1700 and 2300 K also
lie outside its 1823–1923 K main temperature window. The exposed held-out
phosphate-melt comparison is **awaiting data**.

The pressure values below are diagnostic mass-action results from the same
candidate activity before the channel gate was applied. They are not returned
by this profile:

| T (K) | P (bar) | PO (bar) | PO2 (bar) | P2 (bar) |
|---:|---:|---:|---:|---:|
| 1700 | 1.8859×10⁻⁴ | 0.21493 | 0.32915 | 40.583 |
| 2000 | 14.174 | 420.50 | 29.870 | 1.1714×10⁹ |
| 2300 | 5.7410×10⁴ | 1.1412×10⁵ | 840.03 | 3.8468×10¹⁴ |

These pressures are not plausible volatility predictions; P2 already exceeds
1 bar at 1700 K, where the gas contract treats a fixed composition as
completely volatilized. The profile therefore refuses its P-bearing gas
channels with the typed reason `p2o5_associates_incomplete`. A default gas
request still returns the major channels, including SiO and Na. An explicit
P-channel request raises `ImccComponentOutsideDomainError` with that code.

Before this candidate, P2O5 was an unmodelled trace: at 1.5 wt% its bias bound
made every parent activity refuse, and 32 of 41 default gas channels were
omitted, including SiO, Na, K, Fe, Mg, Al, Ca, Ti, and Mn. With this profile,
the melt solver returns the full parent activities and the default gas path
returns 34 channels; the P-bearing channels remain gated for the evidence
reason above.

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
