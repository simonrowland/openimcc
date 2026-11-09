# P2O5 activity fragment

This fragment supplies an opt-in P2O5 melt activity for the existing P gas
channels. It uses `evaluate_p2o5_activity` rather than Mn's redox associate
rows because the gas table already has a P2O5(l) parent standard and the
published dilute activity helper is the smaller extension. It adds no
associates and changes no shipped pack or default table. The machine-readable
rows and standard-state records are in
[`rows.json`](../../src/openimcc/data/fragments/p/rows.json).

## Central activity row

The central relation is the no-Mn Eq. 14 of Suito and Inoue (1984), on their
Turkdogan-Pearson hypothetical-liquid P2O5 standard:

```text
log10(gamma_P2O5) = -1.02(23*x_CaO + 17*x_MgO + 8*x_FetO - 26*x_P2O5)
                   - 22990/T + 9.490
```

The composition terms use formula-unit mole fractions. `Fe2O3` inputs are
folded to two FeO-equivalent formula moles per formula unit and carry the
existing Fe(III)-chemistry notice. The fit's standard reaction is
`P2(g) + 5/2 O2(g) = P2O5(l)`. For a source standard `s` and the gas-layer
standard `g`, chemical-potential equality gives:

```text
log10(a_g) = log10(x_P2O5) + log10(gamma_s)
             + (G_s - G_g)/(R*T*ln(10))
```

The source standard is `G_s = -1534480 + 506.26*T J mol-1`, the earlier
approximate estimate reproduced as Turkdogan (2000) Eq. 4, p. 964. It is
explicitly flagged provisional. The gas table's existing Jung-Hudon standard
is `G = -1616372 + 539.0*T J mol-1`, reproduced in Jung et al. (2013) Eq. 3.
The reaction consumes 3.5 mol gas per mol P2O5. Converting the source's
inferred 1 atm gas standard to the table's 1 bar standard adds
`-delta_n_gas*R*T*ln(1.01325)` to the target Gibbs function. Units reduce to
dimensionless logs because Gibbs energy and `R*T` are both J mol-1.

At 1873 K and `(x_CaO, x_MgO, x_FetO, x_SiO2, x_P2O5) =
(0.40, 0.08, 0.12, 0.39, 0.01)`, the bracket is 11.26, `log10(gamma)` on
the source standard is about -14.27, and the converted activity is about
`2e-16`. This is a hand calculation from the printed equation and standard
functions; the fragment test recomputes the same anchor without using the
loader's coefficients as its expected value.

## Domain and notices

Eq. 14 combines prior low-P results below 1 wt% with new 4-5 wt% results at
1600 °C. Turkdogan (2000) Table 1, p. 965, lists the prior low-P composition
interval as 0.2-1.0 wt%. The encoded windows are therefore 0.2-1.0 wt% at
1823-1923 K and 4-5 wt% at 1873 K. The 1-4 wt% interval is not covered.
Suito and Inoue report a separate MnO coefficient of `13 +/- 9`; this
fragment intentionally uses the no-Mn row and flags any MnO input.

Every result carries `imcc_p2o5_activity_domain_notice`. It records
temperature, P2O5 range, missing MgO, very low SiO2, and unsupported
composition axes (TiO2, Al2O3, Na2O, K2O, and MnO). The notice is present even
inside the listed numeric windows because MgO saturation and full membership
in the source slag family cannot be established from composition alone.
Outside a numeric window the helper returns the extrapolated value with its
axes; it does not claim validation there. The absolute activity also carries
the provisional old-standard limitation. The Turkdogan Eq. 13 and Eq. 25
fits are comparison values only, not an error band. Eq. 13 is converted from
its older standard; Eq. 25 is converted from the revised
`-1655480 + 571.0*T J mol-1` standard in Turkdogan Eq. 23, p. 966.

## Report-only check against measured P activity

Kambayashi, Awaka, and Kato (1985), pp. 1917-1918, report Henry activity
coefficients for FetO-P2O5 and convert them to the earlier P2O5(l) standard.
The following independent check uses a FeO-only limiting matrix in Eq. 14,
then puts both values on the Jung-Hudon 1 bar standard. These conditions are
outside the helper's temperature and slag-family windows; they are shown for
diagnosis only and were not used to choose or adjust coefficients.

| Temperature | Eq. 14 gamma, Jung-Hudon 1 bar | Printed gamma, old liquid standard | Printed gamma converted to Jung-Hudon 1 bar |
| ---: | ---: | ---: | ---: |
| 1643.15 K | 1.6e-12 | (3.4 +/- 1.2)e-13 | 2.5e-12 |
| 1653.15 K | 1.9e-12 | (4.0 +/- 1.3)e-13 | 2.9e-12 |
| 1663.15 K | 2.2e-12 | (4.7 +/- 1.7)e-13 | 3.3e-12 |

The source's P2, PO, and PO2 pressure table does not provide an independent
oxygen fugacity in the battery census or coverage rows. Gas-pressure
predictions against those printed pressures would therefore require an
assumed or inferred condition; this fragment does not report them as
predicted-versus-printed gas pairs. The melt helper only supplies the parent
activity. The caller must still provide oxygen fugacity to the existing gas
evaluator.

## Known limits

- This is an opt-in dilute activity estimate, not a full phosphate melt pack.
- Most lunar regolith proxies and the Mars proxy are below 0.2 wt% P2O5; the
  result is flagged below the published low-P range.
- The source is an MgO-saturated steelmaking slag family, not a broad lunar or
  Martian regolith calibration.
- Mn-bearing inputs are flagged; the separately reported Mn coefficient is
  too uncertain to add in this minimal row.
- Fe-P alloy partitioning during carbon reduction and Fe(III) phosphate
  chemistry are not modeled.
- The Jung-Hudon liquid function has no published numeric uncertainty in the
  source record. No liquid pressure correction is available without a molar
  volume.

## Sources

- Suito and Inoue (1984), *Transactions of the Iron and Steel Institute of
  Japan* 24(1), 40-46, Eq. 14 and the MnO discussion, pp. 45-46.
  [doi:10.2355/isijinternational1966.24.40](https://doi.org/10.2355/isijinternational1966.24.40)
- Suito, Inoue, and Takada (1981), *Transactions of the Iron and Steel
  Institute of Japan* 21(4), 250-259, low-P data referenced by the 1984
  equation. [doi:10.2355/isijinternational1966.21.250](https://doi.org/10.2355/isijinternational1966.21.250)
- Turkdogan (2000), *ISIJ International* 40(10), 964-970, Eqs. 4, 13, 23,
  and 25, Table 1. [doi:10.2355/isijinternational.40.964](https://doi.org/10.2355/isijinternational.40.964)
- Jung and Hudon (2012), *Journal of the American Ceramic Society* 95(11),
  3665-3672; equation reproduced in Jung et al. (2013), Eq. 3.
  [doi:10.1111/j.1551-2916.2012.05382.x](https://doi.org/10.1111/j.1551-2916.2012.05382.x)
- Kambayashi, Awaka, and Kato (1985), *Tetsu-to-Hagané* 71(16), 1911-1918,
  Tables 6-7 and pp. 1917-1918.
  [doi:10.2355/tetsutohagane1955.71.16_1911](https://doi.org/10.2355/tetsutohagane1955.71.16_1911)
