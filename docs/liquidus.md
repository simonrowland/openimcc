# Crystal saturation and liquidus brackets

`saturation_indices(composition, T_K, pack=None, *, condensate=None)` reports
one `SaturationIndex` per sourced crystal phase. `SI` is `None` with status
`OUT_OF_RANGE` outside the phase's tabulated temperature range; no crystal
function is extrapolated. A negative SI is undersaturated and a positive SI
means the phase is stable relative to the melt. Phases representing olivine,
pyroxene, plagioclase, spinel, and melilite end members are labelled
`solid_solution_bound=True`; their indices are end-member bounds, not a
non-ideal solid-solution model.

Both result types expose `notices`, empty when all supplied parents are
supported. If a parent is unsupported, the calculation proceeds with the
supported parents and adds a `CrystalNotice` with code
`crystal_si_dropped_parents`, input wt% for each dropped oxide, and the
`model_id@binding_digest` used. With the default pack, P2O5 selects the
packaged D066 extension; caller-supplied packs are respected. Phase rows that
need unsupported parents have `SI=None` and status `NO_SUPPORTED_COMPOSITION`.

`liquidus_bracket(composition, T_low, T_high, pack=None, *, condensate=None)`
returns the highest supported zero crossing with SI positive toward the
cooling side. Its `status` is `NO_SUPPORTED_CROSSING` if no available phase
crosses in the bracket. The result lists feed-relevant Fe/Ti phases that do
not yet have sourced data. A result with a supported crossing also carries
this list because those phases may change the controlling phase.

The dataset includes Robie and Hemingway (1995) formation-Gibbs nodes for
fayalite (p. 332), hercynite (p. 233), ilmenite (p. 235), ulvöspinel (p. 236),
and perovskite (p. 232). Fayalite and hercynite cover 1300–1400 K; ilmenite
covers 1300–1600 K; ulvöspinel covers 1300–1800 K; perovskite covers
1500–1800 K. The fayalite rows at 1300 and 1400 K and hercynite rows at 1300
and 1400 K are the printed crystal entries. The 1500–1800 K fayalite entries
are liquid and are excluded. Fayalite's incongruent melting point is 1490 K
and perovskite's is 2188 K. The cited tables do not give melting points for
hercynite, ilmenite, or ulvöspinel.

Each R&H fusion node uses one element reference at T for both terms:
`sum(nu * (G_app(parent liquid) - sum(element count * G_app(element)))) -`
`G_f(crystal)`. Parent liquid Gibbs energies come from the active packaged
condensate standards, and JANAF reference-state element Gibbs energies come
from Chase (1998), 4th edition, at the same temperature. This retains the
packaged FeO(l) standard as supplied and avoids mixing it with a separate
wüstite crystal standard. The source element values and table pages are
included in the crystal dataset for audit. The crystal nodes at 1300 K for
fayalite and 1400 K for hercynite were included because those adjacent printed
rows lie within the same parent-liquid segments as the previously included
nodes. As a pure-oxide conversion check, packaged SiO2(l) at 1400 K gives
formation Gibbs −659.823 kJ/mol after subtracting the JANAF Si and O2
references; the printed JANAF liquid value on p. 1754 is −659.670 kJ/mol, a
−0.153 kJ/mol fit difference. The packaged SiO2 liquid-minus-crystal Gibbs
difference is +2.902 kJ/mol at 1995.9 K, just below the JANAF congruent melt
at 1996 K.
Fayalite, hercynite, ilmenite, and ulvöspinel are labelled solid-solution
bounds. The packaged melt model supports their FeO/TiO2 parents. Ferrosilite's
tabulated high-temperature data end at 800 K, and the checked corpus has no
usable high-temperature hedenbergite Gibbs nodes or armalcolite data.
Magnetite is Fe3O4 while melt iron input is FeO-equivalent: interpreting its
index requires an iron-redox balance and must not be treated as a direct FeO
stoichiometric phase.

After rebuilding the Fe/Ti fusion nodes on the JANAF elemental-reference
chain, the cleaned-melt brackets were recomputed over 1323–2100 K. The lunar
low-Ti and Mars feeds remain `NO_SUPPORTED_CROSSING`; CI remains controlled by
forsterite at 1533.6888743 K. The supported statuses did not change. At 1500 K,
corrected SI values on the lunar feed are −2.681 for ilmenite, −3.243 for
ulvöspinel, and −2.676 for perovskite; on the Mars feed they are −2.630,
−2.969, and −2.629, respectively.

Pure anorthite and forsterite tests pin the current roots
at 1691.4 K and 1799.1 K against measured melting points of 1826 K and 2163 K;
these are reported standard-state misses and are not tuned away.

`liquid_only_notice(T_K, bracket)` returns a typed `LiquidOnlyNotice` below a
supported bracketed liquidus, or when there is no supported crossing. It does
not modify `evaluate()` or its labels; callers can attach the notice to their
own liquid-only model result.
