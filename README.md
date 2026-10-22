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

The reported derived-property checks are documented in [Diagnostics](docs/diagnostics.md).

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

Python >= 3.12 is required. Until the first release is published, install from
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
and caller-supplied Ta/Be/Tl/V/Nb/P/S/Li/Rb/Pb/Cs/Cu/Sn/Ga/Ge/B/In/Te/Se/As/Sb/Ag/Ba/Sr/Zr/Ni/Co/Sc/U/Th/Hf/La/Ce/Pr/Nd/Sm/Eu/Gd/Tb/Dy/Ho/Er/Tm/Yb/Lu/Hg/Au/Pd/Pt/Ru/Rh/Ir channels:
Na, Na2, NaO, K, K2, KO, Si, SiO,
SiO2, Fe, FeO, Mg, MgO, Al, AlO, AlO2, Al2O, Al2O2, Ca, CaO, Ti, TiO, TiO2,
Al2, Si2, Si3, Cr, CrO, CrO2, CrO3, V, VO, VO2, Nb, NbO, NbO2, P, P2, P4,
PO, PO2, P4O6, P4O10, S, S2, S3, S4, S5, S6, S7, S8, SO, SO2, SO3, SSO,
Ga, GaO, Ga2O,
Ge, GeO, GeO2, B, BO, BO2, B2O, B2O2, B2O3, In, InO, In2O,
Li, LiO, Li2O, Li2O2, Rb, RbO, Rb2O, Pb, PbO, PbO2, Cs, CsO, Cs2O, Cu,
CuO, Cu2, Sn, SnO, SnO2, Te, Te2, TeO, TeO2, Se, Se2, SeO, SeO2, As, As2, As4,
As4O6, Sb, Sb2, Sb4, SbO, Sb4O6, Ag, Ba, BaO, Sr, SrO, Zr, ZrO, ZrO2, NiO, CoO,
Sc, ScO, ScO2, U, UO,
UO2, UO3, Th, ThO, ThO2, Hf, Hg, Au, Pd, La, LaO, Ce, CeO, CeO2, Pr, PrO, Nd, NdO,
Sm, SmO, Eu, EuO, Gd, GdO, Tb, TbO, Dy, DyO, Ho, HoO, Er, ErO, Tm, TmO,
Yb, YbO, Lu, LuO, Pt, PtO2, Ru, Rh, Ir,
Ta, TaO, TaO2, Be, BeO, Be2, Be2O, (BeO)2, (BeO)3, (BeO)4, Tl, Tl2O,
Na2O, K2O, Ta2O5, BeO, Tl2O, O and O2.
Ta channels use Ta2O5, Be channels use BeO (including Be2(g)), and Tl
channels use Tl2O as their caller-supplied parent activities.
La(g) and LaO(g) use the caller-supplied La2O3(l) activity; the selected
La2O3 liquid branch comes from Konings et al. Table 88, with its printed
constant-Cp liquid continuation flagged as supercooled below 2577 K.
The Ce, Pr, and Nd element and monoxide channels use caller-supplied sesquioxide
liquid activities. CeO2(g) uses caller-supplied CeO2(l), selected from the same
Konings Tables 88–89 by its matching oxidation state. Its liquid branch is
flagged as supercooled below 3083 K. PrO2 and PrO1.833 are crystal rows that
decompose and provide no liquid parent. Parent activities are supplied by the
caller; the gas layer applies no activity model or activity coefficients.
Sm(g) and SmO(g) use caller-supplied Sm2O3(l), selected because Konings Table
88 provides their liquid functions. Sm uses the complete B-form anchor because
the C form has no printed S298. Eu uses Eu2O3(l) as its parent; EuO(cr) and
Eu3O4(cr) have no liquid rows, and this oxide-parent convention does not
represent Eu2+ melt speciation. Gd uses the C branch only through its 1473 K
transition to B; the B branch has no independent 298.15 K anchor. Tb, Dy,
and Ho use their C→B→H→liquid chains; Er, Tm, and Yb use C→H→liquid; Lu
uses its crystal-to-liquid chain. Skipped polymorphs are absent. The 1590 K
Dy2O3, 1680 K Tm2O3, and 1365 K Yb2O3 markers are solid transitions, not
fusion. TbO2 decomposes and is not a parent. Reduced melts can contain Yb2+;
the caller-supplied Yb2O3 activity does not model that speciation. Activities
are caller-supplied; no activity model or activity coefficients are provided.
The Mn channels use public Mn(g), MnO(g) and opt-in Mn+(g) rows with a
caller-supplied MnO(l) activity. Ni(g), NiO(g), Co(g), and CoO(g) use
caller-supplied NiO(l) and CoO(l) activities. The liquid parents use the
estimated phase-paired branches from TSIV Volume IV Tables 2026 and 1334
(tier 3), with labelled liquid-function continuations below their 2228 K and
2090 K source limits. NiO(g) uses the tier-2 NASA Glenn card; its disagreement
with Bulletin 672 is carried as a measured source band. CoO(g) uses TSIV
Volume IV Table 1335 (accuracy class 4-F). Ni+/Ni− and Co+/Co− are available
only when ions are requested. No activity model is supplied.
The sulfur channels activate only when the caller supplies `S2` fugacity as
`a(S2) = f(S2)/p°` on JANAF's 1-bar `S2(g)` reference. OpenIMCC does not
provide a sulfur melt-side activity model for pyrolysis temperatures.

Hydrogen channels use `a(H2) = f(H2)/p°` on JANAF's 1-bar `H2(g)` reference;
nitrogen channels use the same convention for `N2`. Carbon uses `a(C)` relative
to JANAF graphite `C(cr)`, with pure graphite at unit activity. These caller
quantities are pressure ratios or activities, not melt-oxide activities.
Convert a specified water fugacity outside the package with
`a(H2) = f(H2O)/(K fO2^1/2)`, and convert a specified carbon-dioxide fugacity
with `a(C) = f(CO2)/(K fO2)`. The gas layer holds `a(H2)` or `a(C)` fixed as
`fO2` changes; it does not solve a hydrogen or carbon inventory. The H, C, and
N families are omitted unless their parent key is supplied. HCl, HF, HBr, and
HI use JANAF gas rows and require both `H2` and their corresponding diatomic
halogen key; they have zero oxygen demand.

Hg(g) uses the caller-supplied metal activity `a(Hg)` relative to liquid Hg;
the liquid row is continued and flagged above its 629.839 K source limit. The
Au(g) and Pd(g) channels use `a(Au)` and `a(Pd)` relative to the stable metal
phase at each temperature, switching from crystal to liquid at 1337 K and
1827 K respectively. These metal-gas channels have no oxygen-fugacity power.
Pt(g) uses `a(Pt)` on the same stable-metal basis, switching at 2042 K. The
PtO2(g) channel uses that activity with an `fO2` exponent of +1, so it
contributes oxygen demand and moves the oxygen-balance root. Its Burcat card's
continuation above 2000 K is flagged.
Ru(g) uses `a(Ru)` relative to the Pankratz stable-metal functions, switching
from crystal to liquid at 2607 K; this monatomic gas channel has no
oxygen-fugacity power.
Rh(g) uses `a(Rh)` relative to the Pankratz stable-metal functions, switching
from crystal to liquid at 2237 K. Rh(g) uses a flagged continuation above its
2000 K source limit; its liquid-parent continuation above 2500 K is also
flagged. This monatomic gas channel has no oxygen-fugacity power.
Ir(g) uses `a(Ir)` relative to the Burcat stable-metal functions, switching
from crystal to liquid at 2720 K. This monatomic gas channel has no
oxygen-fugacity power.
Os(g) uses `a(Os)` relative to the Burcat Os(cr) stable-metal function over
1200-3000 K. The source liquid branch begins at 3300 K, above the runtime
interval; this monatomic gas channel has no oxygen-fugacity power.
Re(g) uses `a(Re)` relative to the Pankratz Bulletin 677 Re(cr) function over
1200-2000 K, with a flagged held-Cp continuation to 3000 K. The Re(cr)
reference function also covers 298-1200 K so the measured low-temperature
Re2O7(g) intervals use the B677 parent function in range. Re2O7(g) is a
flagged construction over the Smith measured windows at 503-598 K and 613-633 K,
the Battles Re-O equilibrium at 961-1087 K, and the B677 Re2O7 condensed
functions. The Smith pressure-fit scatter is propagated over its measured
windows; the Battles regression uncertainty is carried through the labelled
held-Cp continuation from 1087 K to 3000 K. The B677 Re2O7(l) Cp is held above
its 700 K source limit when reconstructing the high-temperature sublimation
Gibbs function. ReO3(g) is declined because the available evidence is only a
`<3-5%` abundance bound, without a temperature-dependent partial-pressure or
equilibrium function.
OsO4(g) and OsO3(g) are declined because Grimley et al. detected the gases but
could not obtain a temperature-dependent equilibrium relation, and the 1991
study reports reaction/formation enthalpies rather than the temperature-
dependent Cp(T), S(T), or G(T) functions needed for rows.

With `include_ions=True`, the gas result also includes Au+, Au-, Hg+, Pd+,
Pt+, Pt-, and e-. No sourced Ru+, Rh+, or Ir+ functions were found. Ion
channels are opt-in; default gas results remain neutral-only.
For a specified metal-gas fugacity, convert outside the package with
`a(M) = f(M)/(K p°)`, where `K = exp(-(G°(M(g)) - G°(selected metal))/RT)`
and `p° = 1 bar`. The caller activity stays fixed as `fO2` changes.

For example, at 1800 K with `fO2 = 1e-8 bar`, the JANAF-fitted CO2 equilibrium
constant gives `a(C) = 3.15011592e-6` for a 10 mbar CO2 sweep through
`a(C) = f(CO2)/(K fO2)`. This calculation pins `fO2`; the oxygen-balance
solver is not used to set it.

The halogen channels use the same gas-parent input convention: supply
`a(Br2)`, `a(I2)`, `a(Cl2)`, and/or `a(F2)` as `f(X2)/p°`, with `p° = 1 bar`.
These dimensionless fugacity ratios activate each available elemental pair
(Br2/Br, I2/I, Cl2/Cl, F2/F), alkaline-earth halides using caller-supplied
`a(CaO)`, `a(MgO)`, `a(SrO)`, or `a(BaO)`, and Li/Rb/Cs halides using the
existing `a(Li2O)`, `a(Rb2O)`, or `a(Cs2O)` liquid parents. Alkali-halide
pressures use the oxide activity and diatomic halogen activity. No
halogen family is returned when its diatomic gas activity is omitted. Elemental
rows use JANAF gas functions on the 1-bar standard state. Halide product rows
use the cited JANAF or NASA Glenn compilation; mixed-compilation channels carry
a computed parent-scale inconsistency band. NaI and (NaI)2 use NASA Glenn gas
functions paired with JANAF Na2O(l), I2(g), and O2(g) partners.

Cl− and F− dissolution, and Br− dissolution or capacity models remain outside
this gas-parent implementation. Li/Rb/Cs channels are available only where
the existing liquid oxide parent is present; no new oxide parents are added.
The excluded H/C/N species are HO2, H2O2, C(g), carbonates, N2O, and NO3.

The P gas channels now include `P`, `P2`, `P4`, `PO`, `PO2`, `P4O6`, `P4O10`,
and JANAF `PS`. Their parent is a fitted pure-liquid `P2O5(l)` reference from
the Jung-Hudon assessment over 1200-3000 K. Callers supply `a(P2O5)` on this
liquid standard; `PS` also needs caller-supplied `a(S2)` on the JANAF 1-bar gas
standard. OpenIMCC still does not calculate phosphorus activity in a melt.

The `sp_extension` melt packs keep their existing `P2O5` rows, whose activity
reference is half `P4O10(cr)`. That crystal-based melt reference is distinct
from the new pure-liquid gas parent; convert chemical potentials before using
an extension-pack activity as `a(P2O5)` for gas calculations.

Li, Rb, Pb, Cs, Cu, Sn, Ga, Ge, B, In, Te, Se, As, Sb, Ag, Mo, W, Ba, Sr,
Zr, Sc, U, Th, and Hf channels activate when the caller
supplies their parent activity (`Li2O`, `Rb2O`, `PbO`, `Cs2O`, `Cu2O`, `SnO`,
`Ga2O3`, `GeO2`, `B2O3`, `In2O3`, `TeO2`, `SeO2`, `As2O3`, `Sb2O3`, `Ag2O`,
`MoO3`, `WO3`, `BaO`, `SrO`, `ZrO2`, `Sc2O3`, `UO2`, `ThO2`, or `HfO2`) in the
`evaluate_gas` mapping. Activities are caller-supplied; openimcc has no
trace-element activity model and applies no activity coefficients. As(g), As2(g),
and As4(g) are constant-Cp continuation above 1200 K; As4O6(g) is continuation
above 1000 K. SbO(g) uses [Kelley Bulletin 584 Table 30](https://digital.library.unt.edu/ark:/67531/metadc12739/m1/31/)
heat/entropy increments and the [NBS Table 21](https://srd.nist.gov/JPCRD/jpcrdS2Vol11.pdf)
formation-enthalpy anchor; its S298 is inferred from the [Sohn et al. Table V
reaction](https://doi.org/10.1007/s11663-004-0006-8) bridged through [NUREG
Table 12](https://www.osti.gov/servlets/purl/5150148) Sb(g)-Sb(l) values. The
alternate NUREG SbO curve differs by 132.0-149.4 kJ/mol and is recorded as a
non-statistical source-disagreement band; its reference 289 is unidentified.
SbO has a labelled constant-Cp continuation above 2000 K. TeO2(l) is
the Te(IV) parent because Bulletin 672 prints its measured liquid branch; its
constant-Cp continuation above the last printed node is flagged and banded.
In2O3(l) uses its NASA Glenn liquid card from 2186 K with a labelled
constant-Cp continuation down to 1200 K. The As2O3(l), Sb2O3(l), and SeO2(l)
parents retain their flagged tier-4 liquid-from-solid constructions.
MoO3(l) uses the JANAF liquid branch from its 1074 K crystal/liquid marker
through 2000 K, then a labelled constant-Cp continuation. WO3(l) uses the
JANAF O-066 liquid cells through 3000 K, including its printed supercooled
cells below the 1745 K II/liquid marker. These parent activities are supplied
by the caller on their 1-bar liquid standard states.
Sc(g), ScO(g), ScO2(g), and U(g), UO(g), UO2(g), UO3(g) use NASA Glenn
functions with caller-supplied Sc2O3(l) and UO2(l) activities; Sc+(g) is
available only with ions requested. Sc2O3(l) uses its liquid card from 2762 K
with a labelled constant-Cp continuation below. UO2(l) uses a labelled
continuation from its 3123 K liquid card throughout the 1200–3000 K runtime
domain; Bulletin 672 prints its melting marker at 3115 K. Sc2O3(l) and UO2(l)
are the only sourced Sc and U oxide liquid parents in the checked compilations;
no alternate oxide liquid parent is offered. The Sc2O3 liquid branch differs
from Bulletin 672 and is banded in provenance. UO3 formation consumes 1/2 O2.
No evaluated U+ thermo function was located. The Th, ThO, and ThO2 gas rows use
Pankratz Bulletin 672, printed pp. 423, 424, and 426; ThO2(cr) on p. 425 is the
selected oxide parent because no other thorium oxide liquid row was located.
ThO2(l) is a hypothetical supercooled tier-4 liquid_from_solid standard over
1200–3000 K, using the reviewed 3650 K fusion inputs and constant-last-Cp
crystal continuation above 2500 K. The opt-in Th+ row uses the Burcat card
based on Gurvich's 1992 thermal table. Its composition TH 1, E -1 and HF0/HF298
values identify singly ionized Th; the header CAS label “Th+4” conflicts with
those card fields. The Pankratz neutral/Burcat cation pair combines sources.
On the selected 597.2 kJ/mol neutral anchor, its ionization energy is 12.486
kJ/mol below NIST ASD. This contributes +12.486 kJ/mol to RT ln(K_pair/K_Saha);
the remaining thermal-function and level-population terms contribute +6.261
kJ/mol at 1500 K and +7.037 kJ/mol at 2500 K, producing ratios 4.496 and
2.558. Separately, the Pankratz neutral anchor is 4.8 kJ/mol below Burcat's
602.0 kJ/mol neutral anchor.
Activities are caller-supplied and no activity coefficients are applied. The
Hf(g) row uses JANAF Hf-006; opt-in Hf+ and Hf- use JANAF Hf-007 and Hf-008.
Pankratz Bulletin 672 p. 188 supplies HfO2(cr) through 1800 K. HfO2(l) is a
tier-4 construction using the measured Hong et al. (2018) fusion enthalpy
61 +/- 10 kJ/mol and 3073.15 K melting point, with a labelled constant-Cp
crystal continuation above 1800 K. Hf+ and Hf- are checked against NIST ASD
levels, ionization energy, and electron affinity; their measured source ratios
are recorded in provenance. The Hf gas and oxide parent use different compilations, so the
apparent-G comparison is recorded as a source-scale band. No HfO(g) or
HfO2(g) functions were found in the checked JANAF, NASA, Pankratz, TSIV,
NIST WebBook, OECD/NEA, or Burcat sources. BaO(l), SrO(l), and ZrO2(l) use their
printed JANAF liquid branches. JANAF
marks crystal/liquid transitions at 2286 K, 2938 K, and 2950 K respectively.
Ba-035's glass cells below 1600.010 K and its printed supercooled-liquid cells
from 1700 through 2200 K are retained as source data.
BaO and SrO are selected over other oxidation states because those are the
available JANAF liquid parent rows. ZrO(g)'s JANAF G_app is retained despite
its disagreement with the NASA Glenn card; the temperature differences are
recorded in gas provenance.
The oxide-parent convention does not represent telluride dissolution in
reduced melts. SeO2(l) is a flagged tier-4 construction from the Pankratz
crystal function and the OECD/NEA reviewed fusion inputs; its liquid heat
capacity is the documented crystal-Cp carry-over estimate. Selenium can
dissolve as selenide in reduced melts, which this oxide-parent convention does
not represent. SeO(g) is included from the Pankratz gas table. A second Se+
search found NIST WebBook ionization/appearance data only; JANAF has no
selenium element table, and no evaluated Se+(g) thermal function was located in
NASA Glenn/CEA, Pankratz Bulletins 672/677/696, TSIV/IVTAN, OECD/NEA, or Burcat.
TeO(g) uses the
1200-3000 K Gibbs nodes in NUREG/CR-4308 Table 30, with the later measured
TeO formation enthalpy as its 298 K anchor; provenance records the
55.657 kJ/mol anchor adjustment and the remaining reference-288 citation gap.
Te+(g) is declined because the Burcat/Gurvich candidate's fitted Kion/Saha
ratios are 1.228 at 1200 K and 1.107 at 3000 K against NIST's 9.009808 eV
ionization energy, outside the 2% gate. The NBS tables provide an AsO(g)
298.15 K anchor but no temperature-dependent function; none was located in
JANAF, NASA Glenn/CEA, Pankratz 672/677/696, TSIV/IVTAN, NIST WebBook, OECD/NEA,
or Burcat. As+(g), Sb+(g), and U+(g) have no evaluated thermochemical function in JANAF, NASA
Glenn/CEA, Pankratz Bulletins 672/677/696, TSIV/IVTAN, NIST WebBook, OECD/NEA,
or Burcat. As2O3(l) is a flagged tier-4 construction based on the Pankratz
Table 53 crystal function and the repository's leave-one-out liquid_from_solid
error band. The source has no liquid standard-state function. Sb2O3(l) is likewise
a tier-4 caller-supplied liquid standard constructed from the Pankratz crystal
row: the 928 K marker has no fusion enthalpy, and Bulletin 672 publishes no
liquid function. Sb(g), Sb2(g), and Sb4(g) continue above their source nodes;
Sb4O6(g) continues above 1000 K. Sb+ remains declined because no evaluated
function was located in JANAF, NASA Glenn/CEA, Pankratz 672/677/696, TSIV/IVTAN,
NIST WebBook, OECD/NEA, or Burcat. Li2O(l),
Cu2O(l), and PbO(l) use JANAF liquid rows. Rb2O(l), Cs2O(l), SnO(l),
Ga2O3(l), GeO2(l), and In2O3(l) use NASA Glenn CEA liquid cards. Ag(g) and
opt-in Ag+(g) and Ag-(g) use NASA Glenn cards because JANAF has no silver
tables; the Pankratz Ag(g) row is retained for a G_app comparison over its
1800–2000 K source range. Ag2O(l) is a flagged tier-4 construction from the
Pankratz crystal table, which marks decomposition near 500 K and has no liquid
function. Ag2(g) and AgO(g) were not located in the checked NASA Appendix D or
Bulletin 672 tables. The Ag2O-parent convention requires caller-supplied
activity and does not represent silver dissolved as metal in reduced melts.
The Ag2O(l) construction extends the +800 K-calibrated tier-3 band slope about
1500–2500 K past the source's decomposition region near 500 K.
Constant-Cp continuations cover SnO(l) from 1200–1250 K and the Ga2O3, GeO2,
and In2O3 parents down to 1200 K. SnO is selected over the later-starting
SnO2(l) card. Cu2O(l) is the Cu(I) parent because JANAF provides its liquid
row. PbO(l) is the Pb(II) parent; no evaluated PbO2(l) row was found. No
CuO(l), Cu2O(g), or Sn2O2(g) source was found. NASA CEA gas cards use the same
1 bar standard pressure as JANAF, while NASA liquid cards retain their
published pure-liquid 1 atm reference. The printed liquid-card formation
enthalpies are crystal anchors for NG-1643, NG-1801, NG-1842, NG-1844,
NG-1848, NG-1850, NG-12403, NG-12434, NG-12662, NG-1701 and NG-2013; liquid
functions start above the crystal reference state. NASA's [coefficient report](https://ntrs.nasa.gov/api/citations/20020085330/downloads/20020085330.pdf)
defines the NASA-9 H/(R·T) polynomial; including b1 gives the assigned
element-reference enthalpy directly, so
G_app(T)=H_card(T)-T·S_card(T). A liquid-card value extrapolated to 298.15 K
does not replace the assigned formation enthalpy. The JANAF-style expression
dfH298+[H(T)-H(298.15 K)]-T·S(T) agrees only when H_card(298.15 K) equals
dfH298.

The same header-anchor distinction applies to the MoO3(l) card NG-1701 and
WO3(l) card NG-2013. Their liquid-fitter card-G differences from JANAF are
+0.970 and −10.240 kJ/mol at 2000 K; the header-anchored differences are
−22.893 and −39.920 kJ/mol. The liquid-card comparison ledger records both
definitions, and the JANAF parent rows remain selected. The NASA-only clusters
(MoO3)2–5(g) are paired with JANAF MoO3(l), and (WO3)5(g) with JANAF WO3(l).
Their provenance bands report n times the absolute parent card-G scale gap in
J per mole of cluster formula, at each defined comparison temperature; at 2000 K this
is 1.940–4.850 kJ/mol for the Mo clusters and 51.199 kJ/mol for the W pentamer,
a +1.34 dex pressure shift. The WO3(l) card is undefined below 1747 K, so the
W pentamer band is explicitly undefined at 1200 and 1500 K rather than
extrapolating that card.

Where the JANAF and NASA Cu2O(l) functions overlap, compare
`G_app = dfH298 + (H-H298) - T·S`, using the crystal anchor for the NASA card.
The values below are NASA minus JANAF; JANAF remains selected through its last
printed liquid node at 2000 K. The NASA H/S tail is anchored to that JANAF node.

| T (K) | Li2O ΔG (kJ/mol) | PbO ΔG (kJ/mol) | Cu2O ΔG (kJ/mol) |
|---:|---:|---:|---:|
| 1800 | −4.098 | −0.002 | −0.013 |
| 2000 | −3.462 | −0.016 | −0.007 |
| 2500 | −2.171 | −0.048 | — |
| 3000 | −1.242 | −0.079 | — |

Li2O also differs in liquid Cp (103.999 versus 100.416 J/(mol·K)) and entropy
(ΔS −3.37 to −1.54 J/(mol·K)); JANAF remains selected. PbO extrapolated NASA
liquid H(298.15 K) is −202.139 kJ/mol versus JANAF −202.249 kJ/mol, while ΔH
over the overlap is about 0.11 kJ/mol and ΔS about 0.063 J/(mol·K). JANAF
remains selected. O-007 ends at 2500 K, so the required PbO tail anchors
ΔH = ΔS = ΔG = 0 there; against continuing JANAF Cp = 65 J/(mol·K), its
ΔG at 3000 K is 0.017 J/mol. Cu-020 ends at 2000 K; the Cu2O tail anchors its
NASA H/S increments there and does not apply a pressure correction because no
liquid molar volume is included.

Ga, Ge, B and In channels use caller-supplied Ga2O3, GeO2, B2O3 and
In2O3 activities. No activity model or activity-coefficient values are
provided. These are the available source-rated liquid parent rows for their
gas reactions; lower-oxidation-state gases remain channels balanced against
O2. JANAF has no liquid parent row for Ga2O3, GeO2 or In2O3, so their NASA CEA
liquid cards are used. Their first liquid-card nodes (2080, 1388 and 2186 K)
are not interpreted as melting markers. These caller activities are relative
to the cards' published pure-liquid standard states at 1 atm. Labelled
constant-Cp continuations extend those rows to 1200 K; the direct card fits use
G_app=H_card(T)-T·S_card(T). The B2O3(l) parent uses
JANAF table B-096: its
text marks crystal-to-liquid at 723 K, and the first complete liquid thermal
node is 800 K. Its printed liquid cells are fitted directly. No other parent
oxidation state is offered for these four elements because no other evaluated
liquid row was identified in the checked sources.

Where JANAF and NASA cards both provide a function, the selection uses JANAF.
The comparison below is NASA G_app(T) minus JANAF G_app(T), in J/mol, at
1800, 2000, 2500 and 3000 K, calculated from each source's assigned enthalpy
and entropy. The full source Gibbs values and identifiers
are in the provenance ledger. Gas standard states are 1 bar; for B2O3(l),
JANAF is 1 bar and the NASA CEA card is its published 1 atm liquid reference,
with no pressure correction.

| Species | JANAF / NASA | ΔG at 1800 / 2000 / 2500 / 3000 K (J/mol) |
| --- | --- | --- |
| Ga(g) | Ga-005 / NG-4940 | +31.811 / +29.799 / +27.863 / +26.697 |
| Ga+(g) | Ga-006 / NG-4951 | +37.241 / +35.958 / +36.702 / +33.773 |
| BO(g) | B-078 / NG-1063 | +20410.049 / +20410.596 / +20404.021 / +20395.307 |
| BO2(g) | B-079 / NG-1122 | −25139.606 / −25204.912 / −25391.518 / −25609.504 |
| BO2−(g) | B-080 / NG-1130 | −20522.718 / −20683.813 / −21109.513 / −21558.183 |
| B2O(g) | B-093 / NG-1266 | +84633.634 / +82057.491 / +75195.657 / +67851.428 |
| B2O2(g) | B-094 / NG-1274 | −16495.698 / −18238.082 / −22601.945 / −26969.962 |
| B2O3(g) | B-098 / NG-1282 | −1180.877 / −1265.604 / −1467.793 / −1660.838 |
| B2O3(l) | B-096 / NG-11154 | −4968.607 / −5130.645 / −5379.176 / −5410.716 |

The new parent-liquid fit residual maxima are 5.362 and 2.356 J/mol for the
B2O3 runtime intervals 1200–1500 K (fit to 800–1500 K source nodes) and
1500–3000 K; 6.058 and 1.162 J/mol for the Ga2O3 continuation and direct
interval; 25.145, 4.530 and 0.051 J/mol for the GeO2 continuation and two
direct intervals; and 10.793 and 0.591 J/mol for the In2O3 continuation and
direct interval. The lower branches are generated continuations, not source
data; direct NASA-card fit residuals remain below 10 J/mol.

Opt-in trace ions add Ga+, Ge+, B+, Ag+, Ga−, B−, BO− and BO2−; BO− uses a
NASA CEA card because no JANAF BO− table is available. The source-based C3
screen passes for Ga and Ge. B's combined cation and anion share reaches
1.252e-2 at 2800 K and exceeds the 1e-4 criterion. The NASA In+ card is
retained as a source but declined: its dfH298 is 6996.425 J/mol versus
240700 J/mol for neutral In, and its fitted Kion is 5.293e9 at 1500 K and
6.870e6 at 2500 K. [NIST's 5.78636 eV ionisation energy](https://physics.nist.gov/PhysRefData/Handbook/Tables/indiumtable1.htm)
gives ground-term Saha
values 3.505e-19 and 7.514e-11; the fitted-to-Saha ratios are 1.510e28 and
9.143e16. No source coefficient was adjusted, and neutral In channels remain
available.

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
Li2O(l), PbO(l), Cu2O(l), and B2O3(l) are fitted from JANAF liquid tables by
the same generator. Rb2O(l), Cs2O(l), SnO(l), Ga2O3(l), GeO2(l), and In2O3(l)
use NASA CEA liquid cards. The default table adds explicitly labelled
constant-Cp continuations for TiO2(l), Cr2O3(l), V2O3(l), SiO2(l), Al2O3(l),
MgO(l), CaO(l), Ga2O3(l), GeO2(l), In2O3(l), and SnO(l), down to 1200 K.
Li2O(l) uses JANAF's complete liquid-branch cells from 700–3000 K directly;
its runtime fit intervals begin at 1200 K. The Cu2O(l) fit uses JANAF through
2000 K and a NASA H/S-increment tail above it. These continuations are
generated extrapolations, not source data. The seven older continuations now
fit Cp directly while constraining Gibbs energy and entropy at each segment's
upper endpoint. Their maximum Cp error is 1.043 J/(mol K), and their maximum
Gibbs residual is 1.618 J/mol. The four other continuations retain their
Cp-constrained fits; maximum Cp errors are 5.01 (Ga2O3), 0.069 (GeO2), 7.26
(In2O3), and 0.001 J/(mol K) (SnO). The generic continuation test applies one
8.0 J/(mol K) ceiling to every row; the largest measured error is 7.258
J/(mol K) for In2O3. When selected, continuations add a distinct
supercooled-liquid notice to each affected gas channel's `domain_flags`.
Row-level source hashes, methods, temperature ranges, fit residuals, Cp errors,
and Gibbs/entropy seams are in `src/openimcc/data/gas/PROVENANCE.yaml`.

Optional Zn, Cd and Bi channels cover 1200–3000 K with caller-supplied ZnO,
CdO and Bi2O3 activities. Zn(g) uses JANAF; the NASA Zn atom and Zn+ cards are
recorded as G_app comparisons, with JANAF retained. The JANAF catalogue has no
Cd table, so Cd(g) and Cd+ use NASA Glenn cards. ZnO(g), Bi(g), Bi2(g), BiO(g)
and CdO(g) use their cited published tables; CdO(g)'s source is explicitly an
estimate. ZnO(l) and CdO(l) are flagged tier-4 `liquid_from_solid`
constructions. Their fitted crystal-Cp extensions, construction input ranges
and measured bands are recorded per row. Bi2O3(l) uses the printed Pankratz
Table 82 liquid branch, followed by a flagged constant-Cp continuation above
its last liquid node. No activity model or activity coefficients are supplied.

The optional Mn channel uses caller-supplied MnO activity with Mn(g), MnO(g)
and opt-in Mn+(g). MnO(g) is fitted to IVTAN Table 1436 (tier 3); the table
prints accuracy class 4-E and formation enthalpies of 155.620 kJ/mol at
298.15 K and 156.096 kJ/mol at 0 K. The cited page does not quantify a numeric
uncertainty band.
MnO(l) covers 1673–2000 K and reuses the redox pack's tier-4
`liquid_from_solid` construction and input record: the TSIV Table 1435 crystal
function, measured Tm = 2120 ± 20 K, and tier-3 fusion-entropy and liquid-Cp
estimates. The parent is supercooled throughout this runtime interval. Its
shared construction band is 5.754 kJ/mol at 1673 K and 2.520 kJ/mol at 2000 K;
the fitted-row Gibbs residual is 0.186 J/mol. The 10 J/mol on-node fit gate
applies, and the row provenance records the input band and supercooled flag.
No activity model or activity coefficients are supplied.

Melt activities use the published FC87/SF04 complex-formation constants
relative to pure liquid oxide parents. The default gas conversion uses JANAF
4th-edition parent Gibbs functions and labelled liquid continuations; the melt
constants have not been re-derived or jointly assessed against this particular
parent dataset. FC87 itself used JANAF parent thermochemistry (Appendix 1,
p. 218), but historical estimates and later source revisions can leave a
thermodynamic consistency limitation. This default is not an exact reproduction
of published SF04 gas pressures. Use `sf04-published` to reproduce the prior
packaged gas behavior. Continuation flags describe the parent rows and do not
certify the historical melt constants below their demonstrated ranges.

The old LAM1987 parent rows are available in the opt-in `sf04-published` pack
for reproduction of the prior SF04 comparison. Load its `condensate.csv` with
the shared default `gas-shomate.csv`. The former
`gas-janaf-parent-liquids-research` pack is retired because its rows now match
the default; no alias is needed. `SiO2(cr)` stays in both condensate tables
because the public `species_thermo(..., phase="cr")` query reads it. No gas
reaction consumes the crystal row: SF04 uses `SiO2(l)`, matching the liquid
standard state of the melt activities.

K2O(l) uses a flagged liquid-from-solid estimate constructed from the JANAF
K-012 crystal because the [NIST-JANAF potassium index](https://janaf.nist.gov/tables/K-index.html)
lists K2O(cr), not K2O(l). The runtime flag identifies the estimate; the
provenance records its fusion inputs and uncertainty band.

The gas reaction definitions contain stoichiometry, not stored equilibrium
constants: `evaluate_gas` recalculates each K from the active parent Gibbs
row. Oxygen balance is computed from the evaluated gas pressures. The
FC87/SF04 complex log K values live in the separate melt pack and were not
derived from these parent rows. The parent Gibbs differences and resulting
pressure shifts are recorded in the roadmap and changelog; no constant was
re-fitted.

Thermal ions are opt-in: `evaluate_gas(..., include_ions=True)` solves charge
balance on the default neutral gas set, then appends `Na+`, `K+`, `Ca+`, `e-`,
`Li+`, `Rb+`, `Pb+`, `Cs+`, `Sn+`, `Cs2O+`, `Ga+`, `Ge+`, `B+`,
`Zn+`, `Cd+`, `Bi+`, `Mn+`, `Mo+`, and `W+`,
`La+`, `Ce+`, `Pr+`, `Nd+`, `Sm+`, `Eu+`, `Gd+`, `Tb+`, `Dy+`, `Ho+`,
`Er+`, `Tm+`, `Yb+`, `Lu+`, `Y+`, `U+`, `Ru+`, `Rh+`, and `Ir+`,
`Ga-`, `B-`, `BO-`, `BO2-`, `Cu-`, `Mo-`, `W-`, and every available JANAF-supported negative ion
whose neutral is present. In+ is retained in the source ledger but declined
after its NASA card failed the NIST Saha check. JANAF W-006/W-005 imply a 0 K
ionization enthalpy of 770.396 kJ/mol versus NIST IE 7.86403 eV = 758.764
kJ/mol (11.632 kJ/mol, 0.121 eV higher), while their ionization entropy is
about 12 J/(mol K) above the NIST ASD partition functions; that growing
entropy term carries the fitted/Saha ratio from 0.937 at 1500 K to 2.15 at
3000 K, and the full ASD level list changes it by <0.004. Cu+ is declined because its
JANAF Cu-006 entropy yields fitted/Saha ratios 1666.15 and 1656.61 at 1500 and
2500 K; the anomalous entropy is present in the published Cu-006 table. Cu-
is available on the opt-in ion path. For Cu + e- = Cu-,
Kp=(p°/kBT)*(Z-/2Z0)*(h²/(2πme kBT))^(3/2)*exp(EA/kBT). The first factor
has units m^-3 and the electron translational factor m^3, so Kp is
dimensionless. This is the inverse of the cation Saha form in `openimcc.gas`;
detailed balance gives K_attach(Cu)*K_ion(Cu-)=1. With NIST WebBook
[EA(Cu)=1.23578 eV](https://webbook.nist.gov/cgi/cbook.cgi?ID=C7440508&Mask=1020)
and Cu:Cu− ground weights 2:1, fitted/attachment ratios are 0.941424 and
0.958988 at 1500 and 2500 K. Cu-007 implies EA about 1.227 eV (1.22798 and
1.22676 eV from those temperatures); the ~0.009 eV vintage difference is
consistent with the JANAF evaluation. A 0.01 eV affinity shift changes K by
factors 1.08 and 1.05 at those temperatures, supporting the test's 10%
tolerance. Both Cu source records remain in the ledger unchanged. BO- uses
its NASA CEA card because JANAF has no BO- table. The default call remains neutral-only and
returns the same
mapping as before. Ion entries use provenance class
`janaf_fitted_ionisation` for JANAF rows and `nasa_glenn_fitted` for NASA
cards; their source rows cover 1200–3000 K and their domain flags include the
neutral and charge-species source rows. The Cs− and Cu− rows are emitted.
No JANAF Sn− row was found.

The listed La+ through Lu+ rows except Pm, along with Y+, U+, Ru+, Rh+, and
Ir+, are `FLAGGED` and `DERIVED`: their ion functions combine the neutral gas
row, NIST ASD first ionization energy, and neutral and cation electronic
partition functions from listed levels below 25,000 cm⁻¹. Pm+ is omitted
because the project has no Pm(g) neutral thermal function or oxide-parent
model for its C3 screen.
The level cutoff keeps the listed low terms while the source record flags that
the bound-level sets are not asserted complete. These channels are opt-in and
their per-element C3 share screens are in `docs/ROADMAP.md`.
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
Ti−, KO−, NaO−, Cr−, V−, Nb−, Cs− and Cu−. On the C3 screen (README basalt plus
caller-supplied Cr2O3, V2O3 and NbO2 activities of 1e-3; 1200–3000 K,
fO2 = 1e-12–1e-4), O2− has a maximum attachment term `K_A p(O2)` of
4.14e-4 at 1200 K and fO2 = 1e-4; AlO2− reaches p(AlO2−)/p(AlO2) = 6.46,
and KO− reaches p(KO−)/p(KO) = 1.48e-3. Cr−, V− and Nb− have maximum
charge-balance terms `K_A p(A)` of 8.53e2, 0.385 and 8.91e-4, respectively.
All these species are included in the fitted charge balance regardless of
their individual attachment-term size. Ga− is a known source-vintage limit:
its fitted/reference attachment ratios are 0.369 and 0.568 (low by factors
2.71 and 1.76 at 1500 and 2500 K); all ion channels are opt-in. The KO− source is
[JANAF table K-009](https://janaf.nist.gov/tables/K-009.html); its supplied
thermochemistry is used as published.

For the Cs2O, Cu2O and SnO parents at caller activity 1e-3, the largest
charge-balanced positive-ion shares are 1.342e-4 for Cs at 2000 K, 3.257e-5
for Cu at 2800 K, and 3.402e-8 for Sn at 2800 K (all at fO2=1e-4 and within
the 1-bar neutral-pressure domain). The Cs value is above the 1e-4 C3 screen
limit; the Cu and Sn values are below it. The Cs2O+ source row participates in
the Cs charge balance.

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
The Cr(g) fit uses the complete Cr-005 nodes from 1500 to 2900 K and is declared
through 3000 K. Cr-005's 3000 K line has intact thermal cells (Cp, S, H−H298)
although its formation columns change reference at the 2952 K boiling point;
the unchanged coefficients reproduce G_app there to 0.054 J/mol, so the
declared endpoint is source-validated without refitting and results below
2900 K are unchanged. The fitted Cr2O3(l) source interval starts at 1900 K;
the labelled constant-Cp continuation covers 1200–1900 K. Cr channels carry its
continuation notice in that interval; Cr(g) no longer limits C4 coverage.
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
At 2000 K, the NASA-card gas-anchor gaps are 0.154 dex for Na2O(g) and
0.526 dex for K2O(g), with a further 0.668 dex Na2O(g) parent-scale gap to the
LH84 Table 2 liquid.

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

The packaged IMCC-SF04 melt model is the default; no pack path is needed. This example
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

For time-stepped evaluations, pass `y_init` as the natural logs of the prior
result's `parent_x_star` values for the positive-mole parents, in datapack
order. If a depleted parent needs a composition homotopy to converge,
`result.convergence.solver_path` is `"composition-continuation"` and
`continuation_stages` reports the number of stages used.
For low-temperature refusals recovered from a warm root at a higher
temperature, `solver_path` is `"temperature-continuation"`; its stage count
includes the warm solve's recovery stages and the final solve at the requested
temperature.

Output:

```text
activities: {'SiO2': '0.315285', 'MgO': '0.00433676', 'FeO': '0.075856', 'CaO': '7.40178e-05', 'Al2O3': '0.012809', 'TiO2': '0.00252758', 'Na2O': '4.25573e-10', 'K2O': '8.13513e-15'}
melt flags: ('paper-demonstrated-window: T=1800 K is outside the paper-demonstrated domain for rows: Mg2SiO4, MgSiO3, MgAl2O4, MgTiO3, MgTi2O5, Mg2TiO4, Al6Si2O13, CaAl2O4, CaAl4O7, Ca12Al14O33, CaSiO3, CaAl2Si2O8, CaMgSi2O6, Ca2MgSi2O7, Ca2Al2SiO7, CaTiO3, Ca2SiO4, CaTiSiO5, FeTiO3, Fe2SiO4, FeAl2O4, CaAl12O19, Mg2Al4Si5O18, Na2SiO3, Na2Si2O5, NaAlSiO4, NaAlSi3O8, NaAlO2, Na2TiO3, NaAlSi2O6, KAlSiO4, KAlSi3O8, KAlO2, KAlSi2O6',)
melt notices: ('K predictions from IMCC-SF04-D066 remain low against Hastie 1981 KEMS pressures (case 4: −1.18 to −1.24 dex); see https://github.com/simonrowland/openimcc',)
gas: bar {'Na': '0.000264227', 'K': '6.95767e-06'}
Mg domain flag: MgO(l) uses a labelled constant-Cp supercooled-liquid continuation from JANAF Mg-009
```

The `paper-demonstrated-window` flag records that some complex rows are outside
their paper-demonstrated temperature range. The notice reports the Hastie 1981
KEMS case-4 bias measured for the loaded melt pack; a pack without a measured
row says it has not been evaluated. It is part of the result, not a reason to
hide those activities. The gas `Mg` flag reports use of the labelled
supercooled-liquid continuation below the default JANAF MgO(l) fit start at
2200 K.

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

For mapping inputs, an explicitly supplied `parent_oxides` list is authoritative:
if the activity mapping also contains a recognized parent outside that list,
`evaluate_gas` raises `ValueError` and names the parent to add. When
`parent_oxides` is omitted, recognized mapping parents continue to be included
automatically.

Gas result flags identify held-Cp `-SC-CP` continuations and source-tabulated
liquid branches below their recorded fusion markers, and source-tabulated glass
cells below their recorded glass/liquid joins. The source-liquid flags cover
FeO (1650 K), TiO2 (2130 K), Na2O (1405.2 K), WO3 (1745 K), BaO (2286 K), SrO
(2938 K), ZrO2 (2950 K), Li2O (1843 K), Cu2O (1516.7 K), NbO2 (2175 K), Cr2O3
(2603 K), V2O3 (2340 K), MgO (3105 K), and CaO (3200 K). The Na2O flag applies
only to its 1200–1500 K fitted interval; its 1500–3000 K interval starts above
the fusion marker. Gas tables begin at 1200 K; no held-Cp segment is defined
below that temperature.

The melt `evaluate` API also reports a `temperature-domain-extension` notice
when an allowed evaluation falls below an active association row's declared
temperature domain. It identifies the continued ΔrCp = 0 assumption, the
declared domains of the active rows below their lower bounds, and the
unavailable uncertainty band; it does not alter the calculated activities.

It also reports a `cao-sio2-activity-bias` notice for an effectively binary
CaO–SiO2 melt, and that notice does not change activities. The measured
moderate and strong text applies only to the scored default D066 pack
identity. Any other pack inside the same composition and temperature gate
carries `cao-sio2-activity-bias-unscored`, which says the bias was measured
on that default pack and has not been re-scored. After
weight-to-mole conversion, CaO+SiO2 must be at least 0.98 of the parent
moles and every other parent must be below 0.01. On that sub-basis,
x(CaO) = n(CaO)/(n(CaO)+n(SiO2)). The notice fires only for T in
1873–1933 K, including both ends. Moderate severity is x(CaO) in
[0.42, 0.56): Kume 2000 and Omori–Sanbongi 1961 show a(SiO2) about
0.45–0.5 dex low. Strong severity is x(CaO) in [0.56, 0.70], including
both ends: the composition is near Ca2SiO4, the pack has no Ca3SiO5
associate, and the Stolyarova 1991 Table IIb scale (about 0.6–1.0 dex
low) conflicts with Table Ib. At x(CaO)=0.62 the pressure-implied residual
is still 0.25–0.29 dex low. The notice does not fire below x(CaO)=0.42,
above 0.70, below 1873 K, or above 1933 K. That includes x(CaO)=0.30,
lunar basalt compositions, and the unvalidated 1973 K map. A parent at
exactly 0.01 of parent moles, or a CaO+SiO2 share below 0.98, is outside
the binary gate.

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

Use `imposed_parents=("C",)` or `imposed_parents=("N2",)` when that family is
an external sweep reservoir. Its species remain in the pressure mapping but
are excluded from the melt-derived oxygen balance. The default imposes no
reservoirs; the option also accepts other gas-parent keys such as `S2`, `H2`,
`Br2`, `I2`, `Cl2`, and `F2`.
An imposed key without a supplied activity is ignored. Pass parent keys as a
sequence; a bare string raises `TypeError`.

Every supplied gas-parent activity and condensed parent `C` contributes its
oxygen-carrying species to the melt-derived demand unless that parent is named
in `imposed_parents`. This applies whether the parent is supplied in the
activity dictionary alone or also listed in `parent_oxides`; the latter selects
oxide parents and does not control gas-parent inclusion. When `parent_oxides`
is omitted, recognised oxide parents in the activity dictionary (for example
`Cr2O3` or `ZnO`) enter both the returned pressures and the balance; an
explicit `parent_oxides` list still decides which oxide parents are included.

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
for a custom table. The pressure model may be curved between nodes, including a
monotone piecewise fit or a table interpolated linearly in pressure. The solver
evaluates the supplied model's true oxygen-balance residual and refines a root
when its relative residual exceeds the tolerance derived from Brent's `1e-12`
dex root tolerance. A caller can supply pressures from another gas table without
routing them through `evaluate_gas`. The solver enforces the 1 bar molecular-flow
ceiling and requires a sign-changing bracket; a curved model's root is unique
when its true oxygen flux is monotone. Diagnostics report the root path and
pressure-model evaluation count. The inert-cell caveat above still applies.

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

For thermodynamic ceilings on troilite, anhydrite, and crystalline anhydrous
magnesium sulfate decomposition, see
[`evaluate_subsolidus_reactions`](docs/subsolidus-reactions.md).

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

| Parent | Continuation segment (K) | T0 (K) | Cp_l (J/mol K) | Max absolute ΔG (J/mol) | Max absolute ΔCp (J/mol K) | Seam ΔG / ΔS (J/mol; J/mol K) |
|---|---:|---:|---:|---:|---:|---:|
| TiO2(l) | 1200–1500 | 1500 | 100.416 | 0.221 | 0.303 | −4.322 / −0.194 |
| Cr2O3(l) | 1200–1500 | 1900 | 156.900 | 0.326 | 0.447 | +0.046 / −0.000157 |
| Cr2O3(l) | 1500–1900 | 1900 | 156.900 | 0.334 | 0.337 | −0.386 / −0.031735 |
| V2O3(l) | 1200–1700 | 1700 | 156.900 | 1.618 | 0.946 | −0.448 / −0.025636 |
| SiO2(l) | 1200–1500 | 1800 | 85.772 | 0.174 | 0.238 | +0.001 / −0.000010 |
| SiO2(l) | 1500–1800 | 1800 | 85.772 | 0.051 | 0.085 | −0.352 / −0.026378 |
| Al2O3(l) | 1200–1500 | 2500 | 192.464 | 0.546 | 0.745 | +0.347 / −0.001188 |
| Al2O3(l) | 1500–2000 | 2500 | 192.464 | 1.530 | 1.043 | +0.043 / −0.000102 |
| Al2O3(l) | 2000–2500 | 2500 | 192.464 | 0.314 | 0.273 | −0.025 / +0.019091 |
| MgO(l) | 1200–1500 | 2200 | 66.944 | 0.194 | 0.265 | +0.002 / −0.000012 |
| MgO(l) | 1500–1800 | 2200 | 66.944 | 0.061 | 0.102 | +0.007 / −0.000021 |
| MgO(l) | 1800–2200 | 2200 | 66.944 | 0.080 | 0.093 | +0.022 / +0.021617 |
| CaO(l) | 1200–1500 | 2200 | 62.760 | 0.113 | 0.155 | +0.001 / −0.000006 |
| CaO(l) | 1500–1800 | 2200 | 62.760 | 0.032 | 0.054 | +0.003 / −0.000009 |
| CaO(l) | 1800–2200 | 2200 | 62.760 | 0.035 | 0.042 | +0.063 / −0.001708 |

Each seam is the lower-temperature segment minus the adjacent higher-temperature
row at the shared endpoint. The ±10% Cp values below cover the full parent
continuation and remain an illustrative scenario, not a statistical uncertainty.

| Parent continuation (K) | ±10% Cp at 1200 K (J/mol; dex) |
|---|---:|
| MgO(l), 1200–2200 | 1825; 0.0794 |
| CaO(l), 1200–2200 | 1711; 0.0745 |
| Al2O3(l), 1200–2500 | 8069; 0.3512 |
| SiO2(l), 1200–1800 | 973; 0.0424 |
| TiO2(l), 1200–1500 | 324; 0.0141 |
| Cr2O3(l), 1200–1900 | 2331; 0.1015 |
| V2O3(l), 1200–1700 | 1287; 0.0560 |

The analytic-fit residuals above compare each fit with its generated
continuation. A separate source comparison measures the continuation against
the printed JANAF lower-branch node at the same temperature. Across each full
continuation interval, the maximum absolute Gibbs differences are 648.65
J/mol for SiO2(l), 526.06 J/mol for Al2O3(l), 2780.78 J/mol for MgO(l), and
1205.41 J/mol for CaO(l). Over 1400–2200 K, those maxima are 154.67, 26.80,
1570.26, and 651.67 J/mol, respectively. At 1400 K the largest per-metal
pressure discrepancy is 0.0586 dex for Mg; the ≤0.06 dex comparison applies
from 1400 K, not from 1200 K. The printed lower branches include glass or solid
behavior that the liquid continuation deliberately does not follow.

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

The continuation-minus-high-row Gibbs seams are −0.0031 kJ/mol (default MgO),
−0.0016 (default CaO), −0.0295 (default Al2O3), −0.0005 (default SiO2),
−0.0043 (default TiO2), −0.0010 (default Cr2O3), and −0.0005 (default V2O3).
The default major-oxide rows use the JANAF fits above those continuation
intervals. The historical LAM1987 rows remain available through
`sf04-published`.

The TiO2, Cr2O3, and V2O3 continuations close their parent-liquid gaps; Ti, V
and Cr still have gas-interval gaps, so their full standard-state C4 checks
remain incomplete. The major-oxide parents now cover 1200–3000 K, and each
selected supercooled-liquid continuation is identified in the result flags.

For gas rows, `G_J_mol = 1000*H_app_kJ_mol - T*S_J_molK`; the factor of
1000 converts enthalpy to J/mol. For the LAM condensates in `sf04-published`,
the following measured model-minus-JANAF G residual ranges are comparison
context.

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

### Redox results

`evaluate_redox()` returns `RedoxResult.element_moles_by_oxidation_state`, a
mapping from element symbols to oxidation-state/mole-atom mappings. It counts
all liquid parent and complex species; Fe-bearing MELT complexes inherit their
Fe(II) count from parent stoichiometry, while redox-pack species use declared
valence metadata. Fe metal is included at state `0` when present.
`ferric_fe_mol` and `ferrous_fe_mol` alias the Fe entries at states `3` and `2`.

Build Fe3+/sum-Fe from this mapping. For the full Fe inventory, divide state
`3` by the sum of Fe state amounts; for dissolved-only Fe, use states `2` and
`3` in the denominator and leave metal state `0` out. The
`unmodelled_redox_couples` tuple reports present elements whose other known
states have no redox rows in the selected pack. Each record has `element`,
`modelled_states`, and `unmodelled_states`; the records do not estimate shares.
`redox_pack_digest` and `melt_pack_binding_digest` identify the loaded pack
objects. Result flag strings are descriptive, not a stable API.

The keyword-only `redox_pack_path` argument selects a validated redox
component. Relative paths resolve against the process working directory;
omitting the argument uses the packaged default.

The opt-in Cr2+/Cr3+ candidate and its evidence domain are documented in
[`docs/redox-basis.md`](docs/redox-basis.md). Select it explicitly with:

```python
from importlib.resources import as_file, files
from openimcc import evaluate_redox

inventory = {"Fe": 1.0, "Cr": 0.01, "O": 1.0}
candidate = files("openimcc.data").joinpath(
    "packs", "openimcc-redox-cr-candidate-v2.json"
)
with as_file(candidate) as candidate_path:
    result = evaluate_redox(
        inventory,
        1673.0,
        "closed",
        redox_pack_path=candidate_path,
    )
```

Leaving `redox_pack_path` unset continues to use the packaged default
`openimcc-redox-v2.json`.

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
| `hastie1981_kems` | `partial_pressure` | pure-liquid parent | 6 | −0.737 | −0.729 | 0.890 |
| `kume2000_slag_si_alloy` | `activity` | pure-solid | 292 | −0.526 | −0.481 | 0.890 |
| `richter_type_b_cai_digitized_flux` | `evaporation_flux` | unspecified | 4 | — | — | — |
| `richter_type_b_cai_gamma` | `activity_coefficient` | pure-liquid | 3 | +0.050 | +0.053 | 0.110 |
| `richter_type_b_cai_gamma` | `activity_coefficient` | CMAS basis | 3 | +0.543 | +0.554 | 0.577 |

The Hastie 1981 KEMS slice includes three K and three SiO pressure points.
The K residuals are now −1.20 to −1.26 dex with the default K2O(l)
reference; the SiO results are unchanged.

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

### Default JANAF comparison with SF04 measurements (reported)

The following figures remeasure the same independent anchors and digitized
points using the default JANAF major-oxide parents. They are descriptive
comparison values, not acceptance gates. Residuals are
`log10(predicted / measured)`, in dex; O2 is excluded because it supplies the
reference fO2 pin.

| source | species | n | signed median | abs max |
|---|---|--:|--:|--:|
| Table 9 | Mg | 1 | +0.115 | 0.115 |
| Table 9 | SiO | 1 | +0.072 | 0.072 |
| Table 9 | SiO2 | 1 | +0.060 | 0.060 |
| Fig. 10 | Mg | 24 | +0.192 | 0.867 |
| Fig. 10 | MgO | 8 | +0.063 | 1.012 |
| Fig. 10 | SiO | 30 | +0.244 | 1.139 |
| Fig. 10 | SiO2 | 24 | +0.184 | 1.121 |

The unchanged rows retain their existing comparisons. The separate
`sf04-published` pack below preserves the former LAM parents and remains the
target for bit-identical SF04 reproduction tests.

### Published SF04 pressure comparison (`sf04-published`)

The opt-in `sf04-published` pack preserves the prior LAM1987 liquid parents and
the independent Schaefer & Fegley (2004) comparison is split into transcribed
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
uses the JANAF K-012 crystal with its liquid-from-solid construction. This is
independent evidence for the gas pressure path, not a replacement for the
SF04 activity comparison.

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

The default K2O(l) reference changes from the LAM1984 transcription to a
liquid-from-solid construction based on JANAF K-012. At fixed melt activities,
K pressure falls by about 0.35 dex per K atom at 1200–1500 K and 0.26 dex at
3000 K. The SF04 `sf04-published` pack retains LAM1984. Potassium complex
constants are being re-derived against the new reference in a separate change;
until then the current constants leave K predictions 1.20–1.26 dex below the
three Hastie 1981 KEMS pressures. Treat K activities in Ca- and Al-bearing
melts as a known low bias. The activity coefficients
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

The complex rows are evaluated from 1700 K, but several Fegley & Cameron
(1987) `A + B/T` fits were demonstrated only over 2500–3500 K. Below 2500 K
they are extrapolated. Against JANAF (4th ed.) reaction Gibbs energies (the
complex minus its parent liquids), the frozen SF04 reproduction pack
`imcc-sf04-v1.0.2.json` differs at 1700–2500 K by up to 6.4 kJ/mol for
MgAl2O4 (at 1700 K, on the glass segments of the JANAF liquid tables),
1.1 kJ/mol for Mg2SiO4, 0.7 kJ/mol for MgTi2O5, and 0.3–0.4 kJ/mol for
MgTiO3, Mg2TiO4 and MgSiO3 (0.27 kJ/mol). On the liquid branch at
2600–3000 K, its MgAl2O4 row differs by +5.0 to +6.0 kJ/mol (0.10–0.11 dex).
The current default D066 v2 pack combines Shornikov's primary MgAl2O4
fusion measurement and the D&G/Masson FeO(l) re-anchor; the frozen published
pack and D066 v1 remain available with their historical rows. See
[`docs/liquid-from-solid.md`](docs/liquid-from-solid.md) for the source choice
and derivation.

The alkali-silicate rows are assessed or empirical fits, so they cannot be
extended below 1700 K from crystal data, and
evaluations below 1700 K are extrapolations (`evaluate(...,
allow_extrapolation=True)` flags them). The silica parent adds a further
caveat: the published comparison pack preserves SF04's SiO2 transition at
1996 K. The default JANAF O-038 branch changes at about 1696 K and has no
source liquid nodes below 1800 K; its 1200–1800 K parent row is therefore an
explicit constant-Cp continuation, which every affected gas result flags.

The frozen published reproduction pack remains unchanged: every row is its
source's published value. D066 v2 is the single new default with the MgAl2O4
and FeO(l) source corrections and their provenance recorded in the pack.

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
