# GitHub repository metadata

## Description (the field under the repo name, ~350 char limit)

> Open implementation of the ideal mixing of complex components (IMCC) model for
> silicate melt speciation — parent-oxide activities, activity coefficients and
> degree of association, with per-row datapack provenance and an empirical
> benchmark harness that reports refusals as data.

## Topics

```
geochemistry  thermodynamics  silicate-melt  melt-speciation
activity-coefficients  planetary-science  volatility  isru
computational-chemistry  scientific-computing  python  imcc
```

## Website

Leave unset until there are docs; a dead link is worse than none.

## Longer blurb (for a release note or an announcement)

IMCC treats a silicate melt as an ideal solution of complex components: the  
non-ideality of the oxide mixture is not fitted with interaction parameters, it
emerges from speciation. Oxides associate into complexes and what stays unbound
sets the activity.

This is, as far as we can establish, the only open-source IMCC
implementation; the open vapour tools in this space (VapoRock, LavAtmos) take
their melt activities from MELTS instead. It ships auditable datapacks
where every coefficient carries a provenance class. Its 402-point benchmark is
a censored, mixed-standard-state number, not an accuracy figure: 304
predictions are produced — 301 headline-scored and 3 flagged — while 94 fall
outside the declared temperature domain and 4 are refused.

It is not a phase-equilibrium code. It does homogeneous speciation inside one
liquid, and pairs with MELTS-family codes rather than competing with them.

## Accuracy contract

### Schaefer & Fegley (2004), Table 9 anchors

| species | n | signed median residual (dex) |
|---|--:|--:|
| Na | 1 | −1.416 |
| NaO | 1 | −1.615 |
| O | 1 | −0.007 |
| SiO | 1 | −0.009 |
| FeO | 1 | +0.009 |

### Schaefer & Fegley (2004), Fig. 10 digitized points

| species | n | signed median residual (dex) |
|---|--:|--:|
| Na | 35 | −1.079 |
| NaO | 27 | −1.096 |
| Na2 | 1 | −2.212 |
| O | 34 | +0.014 |
| SiO | 30 | +0.175 |
| FeO | 31 | +0.223 |

Na, NaO and Na2 are low at every comparable point; Na and NaO are about 1.1
dex low in aggregate, with medians of −1.082 dex (n = 36) and −1.098 dex
(n = 28), while Na2 is −2.212 dex at its one digitized point. The
printed Table 9 Na anchor is −1.42 dex. O, SiO and FeO each match their own
Table 9 anchor to about 0.01 dex (O −0.007, SiO −0.009, FeO +0.009), so this
is not a unit or fO2-pin error. The Fig. 10 medians
for SiO and FeO sit about +0.2 dex above the transcribed anchor; that is a
figure-versus-table difference in the paper's digitized data, not a reconciled
result. O2 is the fO2 pin identity, not agreement evidence.

Plante 1979 KEMS K pressures agree to median +0.09 dex (RMSE 0.20). The
low-temperature drift is about +0.27 dex at about 1250 K (n = 6), falling to
about +0.02 dex at 1750 K; the K channel uses the flagged K2O(l) secondary
transcription. Independently pinned behaviour is the analytic binary and
limits in `tests/test_kernel.py`,
pure-silica D = 1 in conformance, atom balance, and the JANAF gas-fit
reproduction. Conformance goldens are a drift alarm at RTOL 1e-9, not an
external reference.
