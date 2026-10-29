# Sulfur capacity and matte saturation

`evaluate_sulfur` is an opt-in estimate for sulfur dissolved in one silicate
melt and its saturation against an Fe–Ni–Cu sulfide liquid. It does not change
the default melt solve or the gas tables. The central solubility models are
measured capacity fits; typed notices report their temperature, pressure,
composition, and dissolved-sulfur limits without refusing an estimate.

When melt activities are available, sulfur results also carry the selected
pack's excess-Gibbs candidate and calibration-range notices, plus
composition-bias notices for Hastie KEMS alkali, Na2O, binary CaO–SiO2, and
multicomponent CMAS evidence. Scored bias notices use their family code;
unscored notices use a distinct `<family>-unscored` code, including alkali
notices whose melt text has no code prefix. Crystal-vapour closure,
Fe-Hess-candidate, temperature-domain, and low-temperature notices remain on
the melt result.

```python
from openimcc import evaluate_sulfur

result = evaluate_sulfur(
    {"SiO2": 47.2, "Al2O3": 10.9, "FeO": 18.7, "MgO": 9.2,
     "CaO": 7.9, "Na2O": 3.8, "K2O": 0.5, "TiO2": 1.0},
    1773.0,
    log10_fO2=-8.0,
    total_s_ppm=1000.0,
    sulfide_liquid_composition={"Fe": 0.5, "Ni": 0.1, "Cu": 0.05, "S": 0.35},
)
```

The two log fugacities are absolute base-10 values relative to 1 bar. Supply
exactly one of `log10_fS2`, `total_s_ppm`, or
`total_s_mol_per_mol_melt`. Total sulfur is on a sulfur-free oxide-melt mass
basis; mol/mol is moles of S atoms per mole of oxide formula units. If neither
input is supplied, or multiple inputs conflict, the result carries a typed
notice and returns no sulfur model numbers. If `log10_fO2`
is omitted, the existing openimcc gas oxygen-balance root is used and the
result says `openimcc-intrinsic-oxygen-balance`. That path needs the `[gas]`
extra; if it cannot solve, the result carries a typed notice. No C/CO recipe is
parsed. A caller can pass the absolute `log10_fO2` produced by the redox pack;
this module does not parse buffer offsets or reduction recipes. For an inventory
input and omitted `log10_fS2`, the function solves for sulfur fugacity from the
capacities at the resolved oxygen fugacity.

The optional `sulfide_liquid_composition` is a mole-fraction mapping for Fe,
Ni, Cu, S, and optionally P and C; values must sum to one. Smythe's ideal
matte term is approximated as `a_FeS = X_FeS = Fe/(Fe + Ni + Cu)`. Omitting the
mapping assumes stoichiometric pure FeS and produces a notice. Ni/Cu additions
produce an ideal-matte-activity notice; low S relative to Fe–Ni–Cu sulfide,
and P or C in the liquid, are marked outside the FeS–NiS–CuS0.5 calibration.
The caller supplies the composition of one Fe–Ni–S–P–C sulfide liquid; any
sulfide-saturated sulfur excess is assigned to that liquid. Metal–silicate
partitioning and the resulting liquid composition are not calculated.

## Capacity equations and species split

Both central capacities follow the Fincham–Richardson dissolution reactions.
Concentrations in the equations below are wt% S; fugacities are dimensionless
numerical values in bar relative to the 1-bar standard state.

Boulliung and Wood (2023), Eq. 7, is the central sulfide capacity:

```text
log10 C_S2- = 0.225 + (25237 X_FeO + 5214 X_CaO + 12705 X_MnO
                        + 19829 X_K2O - 1109 X_Si0.5O - 8879) / T
log10 [S2-] = log10 C_S2- + 0.5 (log10 fS2 - log10 fO2)
```

Its X values are oxide fractions on a single-oxygen basis (`X_Si0.5O` is the
silica contribution on that basis), and T is kelvin. The fit combines 13
natural melts with O'Neill and Mavrogenes (2002) sulfide-capacity data at
1673 K. Experiments were at 1 atm and 1473–1773 K, with sulfide stabilized
below FMQ−1. The fitted melt set spans basanite to andesite–dacite transition
compositions and includes iron-free CMAS melts; reported axes include SiO2
39.9–61.6 wt%, FeO 0–16.9 wt%, CaO 4.1–22.3 wt%, TiO2 0–8.34 wt%, and Na2O
0.13–7.77 wt%. Sulfur measurements reach about 3500 ppm. The reported fit
standard error is 0.09 log unit (R² = 0.96). This is an empirical capacity,
not an activity model for dissolved sulfur.

Boulliung and Wood (2022), Eq. 5, is the central sulfate capacity:

```text
log10 C_S6+ = -12.659
  + (3692 X_Ca - 7592 X_Si - 13736 X_Ti + 3762 X_Al + 34483) / T
log10 [S6+] = log10 C_S6+ + 0.5 log10 fS2 + 1.5 log10 fO2
```

The X values are single-cation fractions (`Al` denotes the AlO1.5 component).
The paper measured 14 melts at 1 atm and 1473–1773 K in oxidizing air/SO2
mixtures, spanning basanite through Fe/Ti-rich lunar basalt, andesite, dacite,
and rhyolite. Reported axes include SiO2 39.9–73.2 wt%, FeO 0–16.9 wt%, CaO
1.0–22.3 wt%, TiO2 0–8.34 wt%, and Na2O 0–7.77 wt%. The fit standard error is
0.154 log unit (R² = 0.985); the highest measured sulfate concentration is
about 4090 ppm. Major-element analytical uncertainty dominates the model's
composition uncertainty.

The species fraction follows directly from the ratio of the two dissolution
equations:

```text
S6+ / S2- = (C_S6+ / C_S2-) fO2^2
```

The two species are calculated separately and then summed. This gives ppm S
and mol S atoms per mole of oxide formula units. For a supplied total-S
inventory, the code calculates the ratio `r` above and maximum dissolved
inventory `S_max = SCSS (1 + r)`. At or below `S_max`, it inverts the capacity
sum for `fS2`. Above `S_max`, it sets `fS2` to the value where S2- equals
SCSS, giving S6+ = `r SCSS`; the inventory above `S_max` is assigned to the
caller-supplied sulfide liquid. The returned fugacity therefore reproduces
both dissolved species. When dissolved S6+ exceeds S2-, a separate notice
states that sulfate saturation (including anhydrite) and SO2 degassing are
not modelled; SCAS is not calculated. With a fixed `log10_fS2`, SCSS caps
S2- and any predicted S2- capacity overrun is assigned to the caller's
sulfide liquid, while the supplied fugacity remains the capacity condition.
This fixed-fugacity excess is not a bulk-inventory balance.

### Capacity comparisons

O'Neill and Mavrogenes (2002) is reported as a separate sulfide comparison at
1673 K only. It uses their 1400 °C cation-fraction fit and the coefficients
reproduced in Smythe et al. (2017), Eq. 12/Table 2, p. 799:

| Component | Si | Ti | Al | Mg | Ca | Fe | Na | K | H |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A (K) | −27561 | −11220 | −18450 | −13970 | −7831 | −34274 | −13247 | −29015 | −17495 |

The fit is `ln C_S = 9.087 + (Σ A_i X_i + 116568 X_Si X_Fe)/T`, with
concentration in ppm S. The source combined 19 CaO–MgO–Al2O3–SiO2 compositions
with TiO2 and/or FeO variants and 150 Na/K variants at 1673 K. It is a
comparison point, not blended into either central capacity. It has no
temperature dependence adequate for the process window; the code returns a
typed temperature notice away from 1673 K.

O'Neill and Mavrogenes (2022) is a separate sulfate comparison. It uses
`ln C_S6+ = -8.02 + (21100 + 44000 X_Na + 18700 X_Mg + 4300 X_Al + 44200 X_K
+ 35600 X_Ca + 12600 X_Mn + 16500 X_Fe2+)/T`, where capacity is ppm S6+ and
the standard-state fugacity is `fSO3` in bar. Its 232 experiments span
1200–1500 °C and simple CMAS through natural melts under oxidizing SO2–O2 and
SO2–CO2 conditions. `fSO3` is derived from the existing S2, O2, and SO3 gas
thermochemistry. `fe3_fraction` optionally supplies the Fe3+/ΣFe split; if it
is absent, all iron is assumed ferrous and a notice is returned. The source
identifies major-element analytical uncertainty as the dominant uncertainty.
This prediction remains separate from the two central capacities.
Its 1473–1773 K calibration interval is reported as a typed temperature
notice: predictions at 2073 and 2473 K are 300 and 700 K above that interval.

O'Neill (2021) supplies an optional thermodynamic SCSS cross-check only when a
caller supplies `melt_pack`. The melt pack contributes its own `a(FeO)`; the
matte term uses `a(FeS) = X_FeS`. The cross-check combines the central
Boulliung–Wood sulfide capacity with the 1-bar FeO–FeS Gibbs term tabulated by
Smythe et al. (2017), Eq. 12. It is not a second independently fitted capacity
or a replacement for central Smythe SCSS.

## Sulfide saturation

Smythe, Wood, and Kiseeva (2017), Eq. 11 and Table 2, p. 799, is central. In
the ideal sulfide-liquid form used here:

```text
ln [S]SCSS = -14683/T + 8.03 - 265.80 P(GPa)/T
  + (16430 X_Ti + 9295 X_Al + 13767 X_Mg + 19893 X_Ca - 7080 X_Fe
     + 14197 X_Na + 10189 X_H + 117827 X_Si X_Fe) / T
  + ln X_FeS - ln a_FeO
```

The silicate X values use the single-cation basis; the ideal regression sets
`a_FeO = X_Fe`. Pressure is converted from bar to GPa. The input FeS mole
fraction is `Fe/(Fe+Ni+Cu)` for an FeS–NiS–CuS0.5 liquid. The regression used
392 observations; across 398 compared values, 351 predictions were within
0.667–1.5 times the observed S. Its experimental envelope is 1150–2160 °C
(about 1423–2433 K), 1 bar–24 GPa, and silicate FeO 0.3–40.1 wt%. The ideal
fit has R² = 0.977. Table 2 reports coefficient standard errors; the
0.667–1.5 interval is an empirical fit envelope rather than a one-sigma
prediction interval. The ideal solution approximation works best above X_FeS
about 0.5; deviations grow at lower Fe/(Fe+Ni+Cu). P and C in the supplied
tap liquid are not represented by this calibration and are reported as typed
notices.

After the sulfide/sulfate split, dissolved S2- is compared to SCSS. Below SCSS
the melt is `undersaturated` with respect to sulfide; at or above SCSS it is
`saturated`. For an inventory input, `excess_s_ppm` is the inventory above
`S_max`, and is assigned to the caller-supplied Fe–Ni–S–P–C liquid. For a
fixed fugacity input, it is the capacity-predicted S2- above SCSS. This sulfide
excess is assigned to the same liquid regardless of whether dissolved S6+
exceeds dissolved S2-. In that sulfate-dominant case, a separate notice marks
that sulfate saturation and SO2 degassing are not modelled. SCSS does not
constrain dissolved S6+.

## Transition models and metal partitioning

Jugo et al. (2010) provide an empirical basaltic-glass relation in buffer
coordinates, `S6+/ΣS = 1 / (1 + 10^(2.1 - 2 ΔQFM))`. Their XANES measurements
place the transition over a narrow oxygen-fugacity interval around FMQ to
FMQ+2. Nash, Smythe, and Wood (2019) instead describe sulfur/iron redox
exchange:

```text
log10(S6+/S2-) = 8 log10(Fe3+/Fe2+)
                 + 8.7436e6/T^2 - 27703/T + 20.273
```

The Nash equation is reported over 1000–2000 K, anchored by measurements at
1300 °C and 1 atm and compared with hydrous basalt/andesite data at 1050 °C
and 200 MPa. It requires Fe3+/Fe2+ and therefore an Fe-redox model or supplied
measurement. Neither the Jugo buffer curve nor the Nash Fe-redox relation is
used for the central result: the capacity ratio above is composition-aware
and takes caller-supplied absolute `log10_fO2`, or the intrinsic openimcc gas
oxygen root.

Metal–silicate sulfur partitioning is not implemented. The measured quantity
is generally `D_S^(metal/silicate) = [S]_metal/[S]_silicate`; published fits
relate its logarithm to temperature, pressure, oxygen fugacity, silicate FeO,
and the Fe-rich alloy's S, Si, C, O, and Ni contents. One representative
multi-anvil study covers 1800–2400 °C, 2–23 GPa, and about IW−3.5 to IW−1.5.
That pressure range is 4.3–5.4 decades above a 1-bar process, before extending
to vacuum; the lower part of the requested 1573–2473 K process window is also
below its temperature range. The reducing IW−2 case overlaps its redox range,
but the lance case and pressure do not. The API therefore emits
`metal-silicate-partition-not-modelled`. The caller's one Fe–Ni–S–P–C liquid
receives the modeled sulfide-saturation excess; no metal/silicate partition
coefficient or resulting liquid composition is inferred.

## Process-window limits and notices

The capacity equations are 1-atm fits. At total pressures 0.02–1e-7 bar, their
pressure condition is 1.70–7 decades below calibration; the Smythe model also
has no data below 1 bar. The equations contain no capacity pressure term, so a
low-pressure notice marks extrapolation rather than applying a pressure
correction. Temperatures 1573 and 1773 K lie within the capacity fits; 2073 K
is 300 K above their upper limit, and 2473 K is 700 K above. Smythe covers
1573–2073 K; 2473 K is about 40 K above its upper limit. The requested
compositions have the following calculated distances beyond capacity-fit
composition axes. “Above” and “below” refer to the nearest reported interval
endpoint. SiO2, TiO2, and Na2O are inside their fitted intervals for all seven
rows. All seven FeO values are inside Smythe's 0.3–40.1 wt% interval.

| Feed | Capacity axes outside the interval | Components absent from the capacity fits |
|---|---|---|
| CI carbonaceous chondrite | FeO 3.747 wt% above; CaO 0.771 wt% below Boulliung–Wood 2023 | NiO 2.316 wt% |
| CM carbonaceous chondrite | FeO 4.985 wt% above; CaO 0.976 wt% below Boulliung–Wood 2023 | NiO 2.237 wt% |
| Mars basalt | FeO 1.756 wt% above | P2O5 0.881 wt% |
| Mars sulfate-rich | FeO 1.679 wt% above | P2O5 0.902 wt% |
| Mars phyllosilicate clay | None on reported axes | None |
| Mars perchlorate-rich | FeO 2.338 wt% above | None |
| Lunar mare low-Ti | FeO 0.084 wt% above | P2O5 0.103 wt%; Cr2O3 0.360 wt% |

CI and CM CaO fall below the sulfide-capacity lower limit; those values are
inside the sulfate-capacity lower limit. Oxides absent from the fitted
equations (including P2O5, Cr2O3, NiO, and CoO) are called out. Every
low-pressure run remains below Smythe's 1-bar pressure minimum.

Notices are `SulfurNotice` records. Domain notices contain the model, axis,
input value, calibrated interval, and distance outside it. The API flags
capacity, SCSS, and comparison-model temperature, pressure, and composition
axes; capacity components absent from the calibration; unconstrained
capacity-predicted dissolved sulfur above the largest reported measurements
(even when the separate S2- SCSS cap lowers the returned dissolved amount);
omitted oxides in a supplied melt pack; a default or non-ideal matte
assumption; and unavailable couplings, including the unmodelled sulfate
saturation and SO2 degassing paths. Domain notices never turn into refusals.
Invalid input shapes and nonfinite values still raise `ValueError`.

## Gas coupling

The existing gas tables already contain S, S2, S3–S8, SO, SO2, SO3, and SSO.
They do not contain COS or H2S. The sulfur extension makes no gas-table
changes. Its reduced-gas coupling is the caller's absolute `log10_fS2`; when
the caller supplies a total sulfur inventory instead, the capacity inversion
solves that fugacity. It does not infer a carbon-reduction recipe. The
metal–silicate partitioning remains excluded as described above.

## PySulfSat: implementation facts

The [PySulfSat repository](https://github.com/PennyWieser/PySulfSat), its
[available-models list](https://github.com/PennyWieser/PySulfSat/blob/main/docs/Available_Models.pdf),
and current source expose SCSS calculations from Smythe et al. (2017), Li and
Zhang (2022), O'Neill (2021), Fortin et al. (2015), Liu et al. (2021), and
Blanchard et al. (2021). Its SCAS source names Li and Ripley (2009), Baker and
Moretti (2011), Masotta and Keppler (2015), Chowdhury and Dasgupta (2019), and
Zajacz and Tsay (2019). Speciation and total-S correction methods include
Jugo (2009, 2010), Nash et al. (2019), Kleinsasser et al. (2022), O'Neill and
Mavrogenes (2022), Boulliung and Wood (2022, 2023), and Gorojovsky and Wood
(2026), with atmospheric and pressure-corrected variants for the latter.

PySulfSat has no single calibration envelope. It ships model-specific
calibration datasets; the actual model limits remain those of each source
paper. Examples relevant to this extension are Smythe et al. (2017),
1423–2433 K and 1 bar–24 GPa with silicate FeO 0.3–40.1 wt%; Li and Zhang
(2022), 785–1600 °C and 0.15–3 GPa with about 1–13 wt% H2O; Boulliung and Wood
(2022, 2023), 1473–1773 K at 1 atm; and O'Neill and Mavrogenes (2022),
1200–1500 °C. The PySulfSat routines do not enforce one shared range, so these
limits must be checked against the selected source model; the corresponding
paper ranges are described above where used here.

Its `setup.py` says Python >=3.9 and lists pandas, NumPy, python-ternary,
matplotlib, scikit-learn, SciPy, statsmodels, and openpyxl as dependencies.
Current documentation examples report version 1.0.15. At the 2026-10-22
repository snapshot, GitHub showed a main branch, 150 commits, seven open
issues, and no open pull requests; the project does not state a maintenance
guarantee in its README.

The extension has its own paper-equation tests and does not install or depend
on PySulfSat.

## Sources

- [Boulliung & Wood (2023), *Sulfur oxidation state and solubility in silicate melts*](https://doi.org/10.1007/s00410-023-02033-9)
- [Boulliung & Wood (2022), *SO2 solubility and degassing behavior in silicate melts*](https://doi.org/10.1016/j.gca.2022.08.032)
- [O'Neill & Mavrogenes (2002), *The sulfide capacity and the sulfur content at sulfide saturation of silicate melts at 1400 °C and 1 bar*](https://doi.org/10.1093/petrology/43.6.1049)
- [O'Neill & Mavrogenes (2022), *The sulfate capacities of silicate melts*, GCA 334, 368–382](https://researchportalplus.anu.edu.au/en/publications/the-sulfate-capacities-of-silicate-melts/)
- [Smythe, Wood & Kiseeva (2017), *The S content of silicate melts at sulfide saturation*](https://doi.org/10.2138/am-2017-5800CCBY)
- [O'Neill (2021), *The sulfur solubility minimum and maximum in silicate melt*](https://doi.org/10.1002/9781119473206.ch10)
- [Jugo et al. (2010), sulfur K-edge XANES in basaltic glasses](https://doi.org/10.1016/j.gca.2010.07.022)
- [Nash, Smythe & Wood (2019), *Compositional and temperature effects on sulfur speciation and solubility*](https://doi.org/10.1016/j.epsl.2018.12.006)
- [Bouhifd et al. (2014), *Metal–silicate partitioning of sulphur*](https://doi.org/10.1016/j.epsl.2014.01.021)
- [PySulfSat license](https://github.com/PennyWieser/PySulfSat/blob/main/LICENSE.txt), [package setup](https://github.com/PennyWieser/PySulfSat/blob/main/setup.py), and [model source](https://github.com/PennyWieser/PySulfSat/tree/main/src/PySulfSat)
