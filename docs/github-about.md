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

This is, as far as we can establish, the only open implementation; melt
thermochemistry of this kind is otherwise the province of commercial software
(FactSage, Thermo-Calc, MTDATA, HSC Chemistry). It ships auditable datapacks
where every coefficient carries a provenance class, and a benchmark harness run
against 402 published measurements that reports what it *cannot* answer
alongside what it can: 298 points score, 94 fall outside the declared
temperature domain, and 10 are refused — 4 on data quality, 6 because the
partial-pressure observable is not wired yet. The Na₂O–SiO₂ binaries are
entirely out of domain, so sodium scores zero of 54 — a real coverage gap we
would rather publish than hide.

It is not a phase-equilibrium code. It does homogeneous speciation inside one
liquid, and pairs with MELTS-family codes rather than competing with them.
