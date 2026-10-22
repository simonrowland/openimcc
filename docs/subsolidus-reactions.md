# Subsolidus host reactions

`openimcc.subsolidus.evaluate_subsolidus_reactions` evaluates one mole
normalized decomposition reactions for exact feed-declared hosts:

- Troilite: `FeS_troilite → Fe + ½ S2`
- Anhydrite: `CaSO4 → CaO + SO2 + ½ O2`
- Crystalline anhydrous magnesium sulfate: `MgSO4 → MgO + SO2 + ½ O2`

Pass temperature in K, gas partial pressures and oxygen fugacity in Pa,
condensed-phase activities as dimensionless values, and host inventories in
moles. Gibbs energy is returned in J/mol reaction, `log10_Kp` is dimensionless
with a 1 bar gas standard, equilibrium pressure is in Pa, and the forward
extent bound is in moles of the reaction as written. Every solid coproduct
shares that extent.

Supported results include the signed balanced stoichiometry, the single
Hertz–Knudsen driving gas, its equilibrium pressure, source references, the
joint source temperature interval, and a
`diffusion_limited_equilibrium_upper_bound` notice. The extent is the remaining
host inventory when the supplied state favors the forward reaction, or zero
when it does not. It is not a predicted release amount or rate.

`temperature_out_of_range`, `thermochemistry_missing`, `unknown_host`, and
`missing_required_pressure` are typed refusals. Refused values are `None`, not
zero. The sulfate reactions need both SO2 pressure and one oxygen potential.
Provide that potential through `oxygen_fugacity_Pa` or the O2 pressure entry;
if both are present they must agree within the module's declared tolerance.
The output pressure map contains only S2 for troilite or SO2 for a sulfate.
The sulfate O2 coproduct remains in stoichiometry and is never an independent
release channel.

The R&H mineral formation-Gibbs nodes, JANAF crystal nodes, and the JANAF
SO2 formation-Gibbs nodes are linearly interpolated between adjacent published
temperatures. Existing SO2 and O2 gas rows supply the sulfate domain checks
and source references; the SO2 formation function is evaluated from its JANAF
formation-Gibbs table so its sulfur elemental reference is preserved. The
troilite host row already uses S2 at 1 bar as its sulfur reference. No source
row is extrapolated. The sulfate gas rows begin at 500 K, so sulfate
calls below 500 K are unavailable. Anhydrite is unavailable above its 1000 K
source limit; crystalline anhydrous MgSO4 is covered to 1400 K. Troilite's
solid source table is used through 1400 K; its reported transition temperatures
are preserved in the data record.

This API does not calculate flux, area, accommodation, pump or sweep limits,
mass outputs, timestep release, or solid-release kinetics. Equilibrium is only
a thermodynamic ceiling. It does not model gas-solid back-oxidation. The
caller owns first-melt handoff and must retain unsupported host inventory.

## Sources

- Robie and Hemingway, *Thermodynamic Properties of Minerals and Related
  Substances at 298.15 K and 1 Bar Pressure and at Higher Temperatures*, USGS
  Bulletin 2131 (1995), troilite p. 156, anhydrite p. 285, magnesium sulfate
  p. 292. [USGS publication](https://pubs.usgs.gov/bul/2131/report.pdf)
- Chase, *NIST-JANAF Thermochemical Tables*, 4th ed. (1998), Ca-027,
  Mg-008, O-034 and O-029. The elemental Fe(s) standard has ΔfG° = 0 by
  definition. DOI: [10.18434/T42S31](https://doi.org/10.18434/T42S31).
- Madeleine N. Scheidema and Pekka Taskinen, “Decomposition Thermodynamics
  of Magnesium Sulfate,” *Industrial & Engineering Chemistry Research* 50
  (2011), 9550–9556, DOI [10.1021/ie102554f](https://doi.org/10.1021/ie102554f).
  Their HSC Chemistry calculation reports an equilibrium temperature of
  1080 °C for MgSO4 under the paper’s stated conditions; it does not report a
  measured Kp.
- Newman, “Behavior of Calcium Sulfate at High Temperatures,”
  *Journal of Research of the National Bureau of Standards* 27 (1941), 191.
  [Paper](https://nvlpubs.nist.gov/nistpubs/jres/27/jresv27n2p191_A1b.pdf).
  It discusses earlier total sulfur-oxide/oxygen pressures near 1225 °C, above
  the anhydrite row’s 1000 K limit; that total pressure is not an SO2 partial
  pressure check for this API.

The mineral-table entries have no DOI. Their printed Gibbs nodes and standard
states are identified in the packaged data and returned source references.
