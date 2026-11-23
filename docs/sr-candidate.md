# SrO full-component candidate

`load_profile("sr-candidate")` selects the opt-in 3000 K melt pack
`imcc-sf04-d066-sr-candidate-v2.json`. Its model identity is
`IMCC-SF04-D066-SR-CANDIDATE`, its version is `1.0.0-d066-sr-candidate-v2`,
and its parent basis is the D066 eight oxides followed by SrO. The 38 D066 core
rows are unchanged; seven Sr associations bring the candidate to 45 source
rows (44 active in the solve because the existing out-of-domain KCaAlSi₂O₇
row remains inactive). `load_profile("sr-candidate-v1")` keeps the immutable
v1 pack selectable with its original 2400 K domain.

Both named profiles use the existing default gas and condensate tables and the
existing redox v6 pack. `evaluate_gas(..., profile="sr-candidate")` takes its
parent basis from the profile. The `openimcc` default profile and its SrO trace
limit are unchanged. The manifest's `DEFAULT-COMBINATION` still selects
`imcc-sf04-d066-v4.json`.

## Construction and source windows

The v1 candidate `log10 K = A + B/T` rows were refit by least squares to 1001
evenly spaced pointwise values from 1700 through 2400 K. V2 uses the same
method, sources, and `[1, 1/T]` basis through 3000 K. The absolute
liquid-associate G was re-referenced to the candidate's parent liquid
standards. The candidate SrO liquid standard is NIST-JANAF O-014; the crystal
input is O-013. SrO is a gas reaction parent in the existing gas data and is
added to the melt basis only by this candidate identity. The O-014 reference
and the redox-v6 SrAl₂O₄, SrSiO₃, and Sr₂SiO₄ liquid references declare ranges
through 3000 K. The four new aluminate continuations use the provisional
class-C fusion construction and retain their narrower source windows in
`T_domain_paper_demonstrated_K`; evaluation reports those windows with the
`paper-demonstrated-window` flag.

The 1700–2400 K table compares the original and wider-window coefficients by
their maximum absolute difference in `log10 K`. The final column is each v2
row's maximum absolute residual over its full 1700–3000 K fit window.

| Complex | max absolute Δlog10 K, 1700–2400 K (dex) | v2 max fit residual, 1700–3000 K (dex) |
|---|---:|---:|
| SrAl₂O₄ | 0.000694 | 0.002190 |
| SrAl₄O₇ | 0.000492 | 0.001927 |
| SrAl₁₂O₁₉ | 0.000649 | 0.002273 |
| Sr₃Al₂O₆ | 0.001054 | 0.005023 |
| Sr₄Al₂O₇ | 0.002283 | 0.007956 |
| SrSiO₃ | 0.005453 | 0.009044 |
| Sr₂SiO₄ | 0.006132 | 0.011184 |

Sr₂SiO₄ slightly exceeds the builder's 0.01 dex residual flag threshold. This
is a fit-form residual from representing the source Gibbs curve with `A+B/T`,
not a least-squares solver instability; the value and flag are recorded in the
pack metadata.

| Complex | Source crystal window represented by `T_domain_paper_demonstrated_K` | Fusion/decomposition marker (K) | Class-C `ΔS_fus` (J mol⁻¹ K⁻¹) |
|---|---:|---:|---:|
| SrAl₂O₄ | 932–2233 | 2233 | 58.0827 |
| SrAl₄O₇ | 1000–1300 | 2053 | 99.5704 |
| SrAl₁₂O₁₉ | 1000–1300 | 2063 | 265.5210 |
| Sr₃Al₂O₆ | 298.15 formation enthalpy only | 2009 | 91.2728 |
| α-Sr₄Al₂O₇ | 298.15 formation enthalpy only | 2073 | 107.8679 |
| SrSiO₃ | 298.15–1000 | 1853.15 | 41.4877 |
| Sr₂SiO₄ | 298.15–1000 | 2598.15 | 58.0827 |

The fusion estimate is `8.297530569 J mol⁻¹ formula atom⁻¹ K⁻¹ × total
formula atoms` for every row. It is provisional in every case. For the four
new aluminates, the liquid heat capacity is the sum of the parent liquid heat
capacities at the selected transition marker. The existing redox v6 rows for
SrAl₂O₄, SrSiO₃, and Sr₂SiO₄ retain their source crystal and liquid
constructions and are re-referenced here.

Jacob and Shreyas (2018, [doi:10.1007/s10853-017-1634-0](https://doi.org/10.1007/s10853-017-1634-0))
print formation-from-crystalline-oxide relations `ΔG = A + B·T` for SrAl₄O₇
and SrAl₁₂O₁₉. Recalculation at each printed temperature reproduces those
published equations as a transcription check. Because the tabulated values
are evaluated from the same printed linear equations, this is not an
independent reading or reconstruction from the underlying EMF measurements:

| T (K) | ΔGform SrAl₄O₇ (J/mol) | ΔGform SrAl₁₂O₁₉ (J/mol) |
|---:|---:|---:|
| 1000 | −105563.0 | −109130.0 |
| 1100 | −108100.6 | −111704.4 |
| 1200 | −110638.2 | −114278.8 |
| 1300 | −113175.8 | −116853.2 |

Adding each printed formation Gibbs energy to the JANAF O-013 SrO and Al-096
Al₂O₃ crystal Gibbs energies reproduces the absolute crystal Gibbs energies
used by the builder:

| T (K) | Gcr SrAl₄O₇ (kJ/mol) | Gcr SrAl₁₂O₁₉ (kJ/mol) |
|---:|---:|---:|
| 1000 | −4334.243 | −11449.558 |
| 1100 | −4386.043 | −11575.907 |
| 1200 | −4440.664 | −11709.683 |
| 1300 | −4497.908 | −11850.363 |

For Sr₃Al₂O₆ and α-Sr₄Al₂O₇, the allowed Brisi–Abbattista values transcribed
as printed in Ye et al. (2007, Tables 1 and 3) are respectively
`ΔHform,ox(298.15 K) = −97.7 ± 4.2 kJ/mol` and
`−81.4 ± 5.4 kJ/mol`. Those sources do not print formation entropy or heat
capacity. Following the approved Neumann–Kopp estimate, the candidate uses
`ΔGform,ox(T) = ΔHform,ox(298.15 K)`, or `ΔSform,ox = ΔCpform,ox = 0`.
In algebraic form, `Hcomplex(T) = ΣνHoxide(T) + ΔHform,ox`,
`Scomplex(T) = ΣνSoxide(T)`, and therefore
`Gcomplex(T) = ΣνGoxide(T) + ΔHform,ox`. The printed kJ/mol enthalpy is
converted to J/mol before combining with JANAF quantities. This entropy and
heat-capacity premise is provisional and separate from the provisional
class-C fusion term. No liquid interaction parameters from Ye et al. were used.

The measured Jacob–Shreyas formation entropies give a conservative bound of
`max(25.376/3, 25.744/7) = 8.4586667 J mol⁻¹ oxide K⁻¹`. Scaling by the
four and five oxide formula units bounds the missing formation entropy by
33.8347 J mol⁻¹ K⁻¹ for Sr₃Al₂O₆ and 42.2933 J mol⁻¹ K⁻¹ for Sr₄Al₂O₇.
The resulting uncertainty sensitivity is:

| T (K) | Sr₃Al₂O₆ ±ΔG (kJ/mol) | Sr₃Al₂O₆ ±log₁₀K (dex) | Sr₄Al₂O₇ ±ΔG (kJ/mol) | Sr₄Al₂O₇ ±log₁₀K (dex) |
|---:|---:|---:|---:|---:|
| 1800 | 60.902 | 1.7673 | 76.128 | 2.2091 |
| 2000 | 67.669 | 1.7673 | 84.587 | 2.2091 |
| 2200 | 74.436 | 1.7673 | 93.045 | 2.2091 |
| 2400 | 81.203 | 1.7673 | 101.504 | 2.2091 |

These are sensitivity bounds for the missing formation entropy under the
zero-ΔCp assumption; they do not bound a missing formation heat-capacity
function. For the provisional ΔSₒₓ = 0 Brisi rows, the 1800 K sensitivity
bounds are ±1.77 dex for Sr₃Al₂O₆ and ±2.21 dex for Sr₄Al₂O₇. SrO-rich
predictions depending on these complexes are highly uncertain. The 298.15 K
checks reproduce the two printed enthalpies exactly.

The other source crystal checks use the redox v6 Sr fragment. Its SrAl₂O₄
source construction gives `Gcr(2233 K) = −3002.667 kJ/mol`; its SrSiO₃ and
Sr₂SiO₄ crystal records at 1000 K give −1784.406 and −2540.846 kJ/mol,
respectively, from the printed formation enthalpy, enthalpy increment, and
entropy. These source functions, rather than any scored KEMS results, feed
the candidate refits.

The candidate SrO liquid standard reproduces JANAF O-014's absolute liquid G
(`Hf,298 + Hinc − T·S`) with these fit-minus-table gaps:

| T (K) | G gap (J/mol) |
|---:|---:|
| 1700 | +0.059 |
| 1800 | −0.063 |
| 2000 | −0.030 |
| 2200 | −0.334 |
| 2400 | −0.182 |

The maximum absolute gap is 0.334361 J/mol.

## Phase-diagram sanity checks

The experimental SrO–Al₂O₃ points reported by Ganits et al. (1979), as
reproduced in Ye et al. (2007, [doi:10.1007/s11669-007-9086-x](https://doi.org/10.1007/s11669-007-9086-x)),
were used only as a consistency check. The gaps below are
`Gcr(complex) − Σνμoxide(liquid)`, with
`μoxide = G°oxide(liquid) + RT ln(aoxide)`. They were not used to fit a row.

| Point | T (K) | x(Al₂O₃) | SrAl₂O₄ gap (kJ/mol) | SrAl₄O₇ gap (kJ/mol) |
|---|---:|---:|---:|---:|
| Eutectic | 1933 | 0.584 | +0.049 | −1.626 |
| SrAl₂O₄ congruent point | 2063 | 0.500 | +7.714 | — |

The 7.714 kJ/mol congruent-point gap is retained as a diagnostic; the
transition markers and liquid associates remain provisional.

## Melt and gas examples

At 2000 K the candidate returns the following parent activities:

| Composition (mole fractions) | a(SrO) | a(Al₂O₃) |
|---|---:|---:|
| SrO 0.25, Al₂O₃ 0.75 | 1.32130e−4 | 0.308284 |
| SrO 0.50, Al₂O₃ 0.50 | 0.0591594 | 0.0119361 |
| SrO 0.75, Al₂O₃ 0.25 | 0.197756 | 0.00115226 |
| SrO 0.30, Al₂O₃ 0.30, SiO₂ 0.40 | 1.17303e−4 | 0.162817 |

With `fO2 = 1e-10` bar-relative, `evaluate_gas()` returns `p(Sr)` (bar):

| T (K) | Composition | p(Sr) (bar) |
|---:|---|---:|
| 1700 | pure SrO | 1.25388e−7 |
| 1700 | SrO/Al₂O₃ 0.5/0.5 | 3.76692e−9 |
| 1700 | SrO/Al₂O₃/SiO₂ 0.3/0.3/0.4 | 4.50882e−12 |
| 2000 | pure SrO | 1.28977e−4 |
| 2000 | SrO/Al₂O₃ 0.5/0.5 | 7.63021e−6 |
| 2000 | SrO/Al₂O₃/SiO₂ 0.3/0.3/0.4 | 1.51295e−8 |
| 2400 | pure SrO | 8.27720e−2 |
| 2400 | SrO/Al₂O₃ 0.5/0.5 | 8.97331e−3 |
| 2400 | SrO/Al₂O₃/SiO₂ 0.3/0.3/0.4 | 3.43382e−5 |

No Stolyarova, Vorozhtcov, Lopatin, Shugurov, Tyurnina, or Balabanova KEMS
row was compared or used to select these parameters.

## V1 served values

The profile reports these immutable pack and engine identities:

| Served value | Value |
|---|---|
| Profile | `sr-candidate-v1` |
| Melt file | `imcc-sf04-d066-sr-candidate-v1.json` |
| Melt SHA-256 | `b597601f0cf04c7235ab6964c37d4600e1172b17fb33c47a698c086b929d0a99` |
| Melt binding digest | `c3eb1090b56cba8b6e07c0dd4d261efcb8f1801f10e1153ac523bf5a0324f4ef` |
| Engine binding digest | `07e9374a4ea0b4cdda201a87bbf18164fe6f7c3e4465225be3aac6c47023f14d` |
| Gas table | `gas-shomate-default-v2.csv`, `f049df6efc810f1821565acfe6a2966464153ecf4cb0bbdff9cdbbb8f5cb2543` |
| Condensate table | `condensate-default-v1.csv`, `493c67d5ba23b75767cc3ce4bd6e571bd2400162b3affdbbc581a7b0ae75362e` |
| Redox pack | `openimcc-redox-v6.json`, canonical digest `4beed3886a70ddda484fd2da46c1da74c8a0ea23dd8e6125cca4ecdadaf7cdbc` |

## V2 served values

| Served value | Value |
|---|---|
| Profile | `sr-candidate` |
| Melt file | `imcc-sf04-d066-sr-candidate-v2.json` |
| Melt SHA-256 | `4cb29b0f7fc06ccd93743a2b60ebc1bec1b0bea89623d12dfef7bc3b4551335d` |
| Melt binding digest | `5c3baee0e827e93464860f059552a52d11680227c38f2c315fb45d9f6c1bdb17` |
| Declared Sr candidate row domain | 1700–3000 K |
| Gas table | `gas-shomate-default-v2.csv` |
| Condensate table | `condensate-default-v1.csv` |
| Redox pack | `openimcc-redox-v6.json`, canonical digest `4beed3886a70ddda484fd2da46c1da74c8a0ea23dd8e6125cca4ecdadaf7cdbc` |

The openimcc default remains selected by `DEFAULT-COMBINATION` and keeps its
existing outputs and trace handling.
