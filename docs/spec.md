# OpenIMCC model specification

This document describes the shipped model and its public runtime contracts.
It summarizes the equations implemented by the melt kernel, the separate
redox and gas layers, and the identity checks enforced when data are loaded.

## 1. Model boundary

The base melt model is the published SF04 ideal-mixing-of-complex-components
(IMCC) association model. Its analytical parent basis is the ordered set
\(\mathrm{SiO_2, MgO, FeO, CaO, Al_2O_3, TiO_2, Na_2O, K_2O}\). Inputs may be
nonnegative mole amounts or mass amounts with an explicit `mol` or `wt`
declaration. Mass inputs are converted to moles before the kernel is called.
The declared basis must match the supplied total within a relative tolerance
of \(10^{-6}\); the adapter does not silently renormalize an incomplete
composition. All iron is supplied on the FeO-equivalent basis. Ferric oxide
input is refused so that any Fe(III)/Fe(II) conversion remains an explicit
caller choice.

The published association set has 38 complex rows over these eight parents.
The SF04 table contributes two additions and revised fits for two FC87 rows;
the other rows carry FC87 fits in the published-identity core. The shipped
D066 pack translates the A and B fit coefficients for KAlSiO4, KAlSi3O8,
KAlO2, and KAlSi2O6, and retains KCaAlSi2O7 as inactive. The loader fixes the
expected row set and order for a published core. Only active rows whose
required parents are present participate in a solve. The separate S/P
extension adds the S and P2O5 parent basis and its compound rows only when an
extension pack is loaded and its explicit opt-in is enabled; it does not
change the base model.

## 2. Association equilibrium and parent balances

Let \(b\) be the declared analytical basis in moles and let
\(X_i=n_i/b\) be the analytical amount of parent oxide \(i\) on that basis.
When the basis is omitted, \(b=\sum_i n_i\), so \(X_i\) are mole fractions.
An explicitly supplied basis may differ from the parent-mole sum by at most
\(10^{-6}\) relative and is kept as supplied, without renormalizing \(X\).
Let \(a_i=x_i^*\) be the mole fraction of
free parent \(i\) on the total equilibrium-species basis. Let \(g_j\) be the
mole fraction of complex \(j\) on that same species basis. The stoichiometric
coefficient \(\nu_{ij}\) is the number of moles of parent \(i\) incorporated
per mole of complex \(j\), and \(S_j=\sum_i\nu_{ij}\).

The ideal associated-solution mass-action relation is

\[
g_j=K_j(T)\prod_i a_i^{\nu_{ij}},\qquad
\log_{10}K_j(T)=A_j+\frac{B_j}{T}.
\]

Temperature \(T\) is in kelvin. \(K_j\), \(a_i\), and \(g_j\) are
dimensionless; \(A_j\) is dimensionless and \(B_j\) is in kelvin. The
kernel evaluates \(\ln K_j=\ln(10)(A_j+B_j/T)\). Fractional \(\nu\) values
are allowed and are parsed exactly from the datapack before conversion to
floating-point solver arrays.

For analytical amounts \(n_i\), total species amount \(N\), and species
fractions \(a_i,g_j\), conservation of each parent is

\[
n_i=N\left(a_i+\sum_j\nu_{ij}g_j\right).
\]

When \(b=\sum_i n_i\), summing over parents gives the dimensionless
conversion factor

\[
D=\frac{\sum_i n_i}{N}
 =1+\sum_j(S_j-1)g_j.
\]

The second equality follows because one mole of complex \(j\) replaces
\(S_j\) parent-oxide formula units with one species particle. For a matching
basis, \(D=b/N\). The kernel uses the displayed conversion-factor identity
and declared-basis fractions \(X_i=n_i/b\) in the solved system. A permitted
near-match in the declared basis is kept as supplied; the resulting small
normalization discrepancy is not silently corrected by renormalizing \(X\).

\[
D X_i=a_i+\sum_j\nu_{ij}g_j,\qquad
f_i=a_i+\sum_j\nu_{ij}g_j-DX_i=0.
\]

Every term is dimensionless. This form also supplies a per-parent numerical
certificate: for each positive input parent, the solver checks the relative
inventory residual against its tolerance and enforces the exact bound

\[
0<a_i\le D X_i.
\]

The bound follows directly from nonnegative complex fractions and
stoichiometries. A result that fails the refined balance certificate is
reported as nonconvergence, not returned as a partial solution.

The activity coefficient is

\[
\gamma_i=\frac{a_i}{X_i}.
\]

Here the numerator uses the total-species basis while the denominator is the
analytical parent basis. They are not the same normalization; consequently
\(\gamma_i\le1\) is not a general theorem. In the no-association limit
\(K_j\to0\), all \(g_j\to0\), \(D\to1\), and \(a_i=X_i\), so
\(\gamma_i\to1\).

## 3. Numerical solve and cold starts

The unknowns are \(y_i=\ln a_i\), which keep free-parent fractions positive.
For a trial \(y\), the kernel evaluates

\[
g_j=\exp\!\left(\ln K_j+\sum_i\nu_{ij}y_i\right),\quad
D=1+\sum_j(S_j-1)g_j,
\]

then solves the coupled residual vector \(f_i=0\) with its analytic
Jacobian. The primary solver is bounded in log space. It checks convergence
using the largest absolute parent residual; a certification pass verifies
each parent’s relative inventory error and may refine the root.

The standard melt cold start uses bounded trust-region least squares in
log-activity space. With absolute-tolerance residuals, its initial vector is
\(y_i=\ln(\max(X_i,10^{-12}))\); the relative-tolerance path uses
\(y_i=\ln X_i\). Cold starts follow the zero-association solution. At
\(\lambda=0\), set \(K_j^{(\lambda)}=\lambda K_j\); then every complex
vanishes and \(a_i=X_i\) is an exact root. The first nonzero continuation
stage is chosen so no ideal-seed complex exceeds \(10^{-4}\). The solver
raises \(\log_{10}\lambda\) toward zero in steps no larger than two decades,
reusing each converged root as the next seed until the full-strength system
\(\lambda=1\) is reached. Damped Newton followed by a relative-tolerance
least-squares fallback belongs to the separate redox inner solve.

If the standard melt evaluation raises nonconvergence, the composition
adapter tries these recovery steps in order:

1. If the single smallest positive parent is below \(10^{-3}\) of the input
   mole total, walk that parent through seven geometric stages from the
   \(10^{-3}\) fraction down to its requested amount. A failed stage can pass
   its finite last iterate to the next stage; it is not necessarily a
   converged root. If a stage fails after at least one earlier stage
   converged, try an anchor at the smaller of the requested fraction and
   \(10^{-11}\), then retry the exact input when the anchor was an
   intermediate composition. At or below \(10^{-11}\), the anchor is already
   the exact input and is retried directly. A walk with no converged stage
   skips this anchor.
2. If the trace walk did not recover and the input has positive SiO2, reset
   the seed and try silica multipliers \(10,8,6,4,2,1\). Failed stages can
   pass a finite last iterate forward. Factor 1 restores the input
   composition.
3. If those steps fail, form the shared temperature interval from active-row
   domains. When its start at 10 K above the highest lower edge lies inside
   the interval and the target temperature is below that start, evaluate at
   the start using the original caller seed, then retry the target from that
   result with parent-scaled residuals. Targets below the interval reach this
   step only if extrapolation was already allowed; otherwise the earlier
   domain check refuses before nonconvergence recovery begins.
4. As a final retry, discard warm starts and solve once from \(\ln X_i\) with
   parent-scaled residuals and tolerance capped at \(10^{-14}\) (or a stricter
   caller tolerance).

These stages provide seeds only; the requested composition and temperature
are restored for the returned solve, and the same per-parent certificate
decides acceptance. Failure to reach and certify the full-strength root
raises a typed nonconvergence error with diagnostics.

Declared per-row temperature intervals are enforced by default. Explicit
extrapolation can evaluate a row outside its interval and labels that result.
Rows lacking a required input parent are inactive for that composition.

## 4. Redox layer

Iron redox is a separate opt-in calculation over the melt kernel. It supports
closed oxygen inventory and imposed oxygen fugacity \(f_{\mathrm O_2}\)
modes. A redox reaction row provides balanced standard-state species terms
and an external oxygen coefficient \(\nu_{\mathrm O_2}\), positive when
oxygen is a product. Its standard reaction Gibbs energy is

\[
\Delta_rG^\circ(T)=\sum_s\nu_sG_s^\circ(T),\qquad
\ln K^\circ=-\frac{\Delta_rG^\circ}{RT},
\]

where \(G_s^\circ\) and \(\Delta_rG^\circ\) are in J mol\(^{-1}\),
\(R\) is in J mol\(^{-1}\) K\(^{-1}\), and stoichiometric coefficients
are positive for products. This displayed reaction function sums its
standard-state terms. For a complex row that consumes another complex, the
solver instead forms the parent-basis standard Gibbs difference

\[
\Delta G_{\rm parent}=G^\circ_{\rm product}
-\sum_i\nu_iG^\circ_i-\lambda_{\rm shift}G^\circ_{\rm O_2},\qquad
\lambda_{\rm shift}=\frac{O_{\rm product}-\sum_i\nu_iO_i}{2},
\]

and uses \(\ln K=-\Delta G_{\rm parent}/(RT)
+\lambda_{\rm shift}\ln(f_{\mathrm O_2}/p^\circ)\). For ideal gas oxygen
with standard pressure \(p^\circ=1\) bar,
\(\lambda=\ln(f_{\mathrm O_2}/p^\circ)\) is dimensionless. In imposed
mode, \(\lambda_{\rm imposed}=\ln(f_{\mathrm O_2}/1\,{\rm bar})\). An
imposed value below the dissolved-iron saturation range is refused with a
typed endpoint; on an invariant buffered segment the supplied oxygen
inventory selects the metal extent.

In closed mode, total oxygen is supplied in moles of O atoms; the solver
finds the redox state that satisfies that inventory together with the
parent-oxide balance. The standard-state redox datapack has its own schema,
temperature evidence intervals, identity digest, and row-level provenance.
Candidate packs or candidate-only composition-conditioned terms remain
opt-in and are not silently promoted into the default redox model.

The base melt API itself does not infer ferric iron or convert Fe2O3 to FeO.
The caller must choose and declare whether to use the separate redox
calculation; this preserves the base model’s FeO-equivalent input contract.

## 5. Gas layer and oxygen-balance contract

The gas layer evaluates retained vaporization channels from the supplied melt
parent activities. Gas standard states are ideal gases at \(p^\circ=1\) bar;
oxide parents use their liquid standard-state rows. The condensed references
for Au, Pd, Pt, Ru, Rh, and Ir use crystal rows below their pack transition
temperatures (1337, 1827, 2042, 2607, 2237, and 2720 K, respectively) and
liquid rows at or above them. C uses C(cr), Hg uses Hg(l), and Os and Re
use Os(cr) and Re(cr) throughout, since both melt above the 3000 K domain
ceiling. Gas-phase
parents use their gas rows. For
the channel

\[
\mathrm{oxide(l)}=m\,\mathrm{gas(g)}+q\,\mathrm O_2\mathrm{(g)},
\]

\[
\Delta_rG^\circ=mG^\circ_{\rm gas}+qG^\circ_{\rm O_2}-G^\circ_{\rm oxide},
\quad K^\circ=\exp\left(-\frac{\Delta_rG^\circ}{RT}\right),
\]

and the dimensionless gas pressure is

\[
\frac{p_{\rm gas}}{p^\circ}
 =\left(\frac{K^\circ a_{\rm oxide}}
 {(f_{\rm O_2}/p^\circ)^q}\right)^{1/m}.
\]

With the one-bar standard state, the numerical gas pressure is in bar.
Reaction-specific stoichiometries also cover gas-phase parent standards,
dimers, halides, and non-oxide gas parents. The default channel set includes
a channel only when its parent activity was supplied (oxygen channels have no
parent); missing-parent channels are skipped. Optional channels whose gas or
parent reference rows are unavailable are omitted with a reason. An explicit
request for a species absent from the reaction table is refused. Oxygen
fugacity must be finite and positive on every `evaluate_gas` call, even when
the selected channel has zero oxygen stoichiometry. Temperature extrapolation
is allowed by default and flagged; setting `allow_extrapolation=False`
refuses an out-of-interval row.

Ion channels are off by default. With `include_ions=True`, charge balance
uses the available fitted neutral/ion pairs and adds electrons and ions. In
dimensionless pressure units \(p/p^\circ\), its electron pressure is

\[
p(e^-)=\sqrt{\frac{\sum_M K_Mp(M)}{1+\sum_AK_Ap(A)}}.
\]

The oxygen-balance evaluator uses the neutral-only gas calculation, so ions
are not included in its flux function. A summed neutral pressure above 1 bar
is returned; when ions are enabled it adds a credibility notice to ion
domain flags. The 1-bar ceiling applies to the oxygen-balance \(p_{\rm O_2}\)
bracket, not to the sum of neutral pressures.

The oxygen-balance solver chooses \(p_{\mathrm O_2}\) so the net oxygen flux
relative to the parent-oxide oxygen demand vanishes. In a Knudsen effusion
source, each species flux scales as \(p_i/\sqrt{M_iT}\). Common aperture and
temperature factors cancel, leaving

\[
F=\sum_i
\frac{\left(n_{{\rm O},i}-d_i\right)p_i}{\sqrt{M_i}}=0,
\]

where \(n_{{\rm O},i}\) is oxygen atoms per gas molecule, \(d_i\) is the
oxygen demand assigned to the corresponding parent, and \(M_i\) is molar
mass in g mol\(^{-1}\). Thus each weighted term has units proportional to
bar\((\mathrm{g\,mol^{-1}})^{-1/2}\); the shared flux scale is omitted.
For analytic gas tables, each retained pressure follows its declared power law
\(p_i\propto p_{\mathrm O_2}^{k_i}\); the signed oxygen weight and exponent
must agree on the expected flux direction. A caller-supplied pressure model may
curve between nodes, including a monotone piecewise fit or interpolation linear
in pressure. Brent evaluates that model's true residual over the bracket. A
sign change certifies at least one root; the root is unique when the true oxygen
flux is monotone. The initial root tolerance is \(10^{-12}\) dex. The solver
derives a relative flux-residual tolerance from that value and the declared
exponents, then uses a tighter bracketed solve when the true residual exceeds
it. The bracket does not extend above \(p_{\mathrm O_2}=1\) bar. Invalid
brackets, a root above 1 bar, a missing sign change, a Brent failure, invalid
pressure values, or a residual that remains above tolerance produce typed
refusals. The `root_solver` diagnostic reports the direct or refined path and
model-evaluation counts.

Explicitly imposed external gas reservoirs may be reported in the pressure
map but are excluded from melt-derived oxygen accounting. Species parents
omitted by the caller are likewise outside that balance. The returned
diagnostics include the relative flux residual and the dominant oxygen and
parent carriers.

## 6. Datapack identity and provenance

The melt datapack fixes the ordered parent basis, the expected 38-row
association set, each row’s stoichiometry, \(A\), \(B\), temperature domain,
and provenance. Published-identity packs are canonicalized and checked
against a frozen SHA-256 core digest before use. A mismatch raises at that
identity gate before band-field validation. The shipped D066 default pack
first passes its own row-set gate and then the band-field checks; it is not
represented as the unchanged published identity. Extension rows are
separately gated and carry their extension identity and certification status.

The public melt result labels contain `identity`, `coverage`,
`envelope_status`, `flags`, `notices`, `acid_sink_ratio`, and `inactive_rows`;
there is no result `trust` field. Coverage is a lineage class, such as
`A-published-imcc`, `D-translated-d066`, `D-reference-unknown`,
`D-e25-kcaalsi2o7`, `D-carried-published`, or `EXT-SP`, rather than a report
of row or screen availability. The kernel's internal evidence class is
`internal-analytical` and is not copied into the adapter labels. Extension
certification status is checked as a load-time pack field. The redox pack
validates its own canonical content digest; gas rows retain their
source-table provenance. Neither inherits the melt-pack identity.

## 7. Uncertainty-band schema

For thermochemical records that use the common band schema, four fields
separate fit quality, evidence about accuracy, the basis of a band, and an
independent check:

| Field | Meaning |
| --- | --- |
| `fit_residual` | Residual against the named fit target, with value, unit, target, method, temperature range, and in-sample or held-out designation. It measures fit quality only. |
| `accuracy_band` | Source uncertainty or a measured construction envelope, with value, unit, temperature range, and coverage. It is not copied from a fit residual. |
| `band_basis` | One of `source_uncertainty`, `measured_construction`, `independent_measurement`, `fit_only`, or `none`, with an explanation and provenance. |
| `independent_check` | Status (`tested`, `lead`, `not_found`, or `not_applicable`), equilibrium, primary source, conditions, measured and predicted values, miss, miss-to-band ratio, and reason. Unknown values remain null. |

The loader requires every melt complex row to have
`band_basis.kind: none` and `accuracy_band: null`; these are enforced schema
rules. A fit residual is never treated as thermodynamic accuracy. See
[Datapack format](datapack-format.md) for the complete field and loader
contract.

## 8. Implementation references

The equations above are implemented in `src/openimcc/kernel.py` and
`src/openimcc/model.py`. Redox thermochemistry and equilibrium are implemented
in `src/openimcc/redox_pack.py` and `src/openimcc/redox.py`. Gas reactions and
oxygen balance are implemented in `src/openimcc/gas.py`. The machine-readable
melt-pack contract is documented in [Datapack format](datapack-format.md).
