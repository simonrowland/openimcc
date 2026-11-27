# Opt-in Wu MgO–SiO₂ MQM candidate

Select this fixed-parameter candidate with
`evaluate(composition, T_K, profile="mqm-candidate")`. It is opt-in; the
default D066 pack and its outputs are unchanged. The fixed binary activities
come from Wu, Eriksson, Pelton & Blander (1993), *ISIJ International* 33,
26–35, Eqs. (7)–(8), p. 30,
[doi:10.2355/isijinternational.33.26](https://doi.org/10.2355/isijinternational.33.26).
They are transcribed as published and are not fitted to the multicomponent
results.

## Gibbs function

The candidate starts from default D066 with `Mg2SiO4` and `MgSiO3` inactive,
then adds

~~~text
G = G_D066(with Mg2SiO4 and MgSiO3 removed) + N_f * g_E(y)
y = n*(SiO2) / (n*(MgO) + n*(SiO2))
g_E = g_Wu(y,T) - R*T*((1-y)*ln(1-y) + y*ln(y))
mu*_Mg,E = g_E - y * dg_E/dy
mu*_Si,E = g_E + (1-y) * dg_E/dy
~~~

`g_Wu` is the existing binary pair-approximation mixing function, including
its ideal free-oxide entropy. D066 already supplies that ideal entropy, so the
embedded Gibbs function adds only the excess part. There is no Wu standard-
state sum in `G`: the binary join is reported in Wu standards before
conversion, and a melt with no MgO is exactly the default D066 result,
including `a(SiO2)`. Only `evaluate_mgo_sio2_mqm()` converts its returned binary
activities to OpenIMCC liquid standards.

The free-pool subsystem uses only MgO and SiO₂. It is not Wu's Toop extension,
and other oxides do not dilute `y`. Retained D066 associates—including
CaMgSi2O6, Ca2MgSi2O7, and Mg2Al4Si5O18—keep their coefficients and respond to
the corrected free-parent potentials. They were calibrated alongside the two
removed binary silicates. Nothing is refit here; the full rescore will tell
whether CMAS regresses.

The loaded solve uses deterministic multistart and the species-pair path's
projected Gibbs-Hessian stability check. On the binary join it raises
`ImccMQMCandidateNonconvexError` inside or within one binodal grid step of
Wu's silica-rich two-liquid gap. In a multicomponent melt, a free-pool value in
that interval alone does not establish instability because retained associates
can stabilize the coupled speciation; the typed refusal follows when the
projected Hessian is unstable there, or no root can be found there.
`evaluate_redox()` carries the same Gibbs block through its liquid solve.
`equilibrate()` returns the typed `unsupported_nonconvex_model` refusal only
when both free MgO and SiO₂ are present, because the single-liquid crystal
optimizer does not carry the block through. Mg-free states run normally, where
the candidate Gibbs term is inactive.

## Exposed-source diagnostics

Under the same self-buffered oxygen-balance flux calculation, Hashimoto's
Table 5 starting composition changes the predicted Fe/Si preference from
`+1.19` to `+0.50` dex. The previously reported Markova panel VI diagnostics
at the evolved residue compositions are Mg-first at `+1.18` and `+2.51` dex.
Replaying the gas calculation with this candidate's OpenIMCC-standard
activities gives `+1.16` and `+2.49` dex; both calculations remain Mg-first,
so the Markova sign is not fixed by this model. These are model diagnostics,
not fitted binary activities or independent validation of the gas calculation.

## Full residual replay

The candidate was replayed against the full scored residual set on the same
rows as the default (melt activity 1,023, vapour 1,749, SiO evolution 164; the
default replay reproduces the stored predictions to 1.5e-13 dex). Median
|log10(model/measured)| per melt-activity cell, default → candidate:

| Cell | n | Median abs residual (dex) | Signed median (dex) |
|---|---:|---:|---:|
| binary silicate, 1700–1950 K | 102 | 0.371 → 0.268 | −0.334 → −0.029 |
| binary silicate, 1950–2200 K | 27 | 0.358 → 0.036 | −0.014 → +0.003 |
| ternary silicate, 1700–1950 K | 112 | 0.411 → 0.356 | −0.313 → −0.262 |
| Na-silicate, 1323–1700 K | 253 | 0.161 → 0.201 | −0.001 → −0.074 |
| CMAS, 1700–1950 K | 38 | 0.473 → 0.473 | −0.278 → −0.413 |
| CAS, 1700–1950 K | 364 | unchanged | unchanged |

The MgO–SiO₂ binary bias is removed. The Na-silicate 1323–1700 K cell and
the CMAS signed median move the wrong way, consistent with retained
Mg-bearing associates that were calibrated alongside the removed binary
silicates. The candidate also refuses 47 rows that the default scores:
32 `ImccSpeciesPairMultipleRootsError`, 11 `ImccMQMCandidateNonconvexError`,
and 4 `ImccNonconvergenceError`. It therefore stays opt-in.

The opt-in profile uses the same packaged gas, condensate, and redox packs as
the default profile. The new Gibbs parameters and inactive-row reasons are
part of the candidate pack's binding digest and ancestry record.
