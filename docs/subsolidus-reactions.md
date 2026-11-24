# Subsolidus host reactions

`openimcc.subsolidus.evaluate_subsolidus_reactions` evaluates one mole
normalized decomposition reactions for exact feed-declared hosts:

- Troilite: `FeS_troilite → Fe + ½ S2`
- Anhydrite: `CaSO4 → CaO + SO2 + ½ O2`
- Crystalline anhydrous magnesium sulfate: `MgSO4 → MgO + SO2 + ½ O2`
- Ferric sulfate: `Fe2(SO4)3 → Fe2O3 + 3 SO2 + 3/2 O2`
- Ferrous sulfate: `FeSO4 → 1/2 Fe2O3 + SO2 + 1/4 O2`
- Brucite: `Mg(OH)2 → MgO + H2O`
- Portlandite: `Ca(OH)2 → CaO + H2O`

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
The output pressure map contains only S2 for troilite, SO2 for a sulfate, or
H2O for brucite and portlandite. The sulfate O2 coproduct remains in
stoichiometry and is never an independent release channel. Ferric sulfate has
three moles of SO2 per reaction mole, and its returned SO2 ceiling solves that
cubic pressure term.

`numeric_out_of_range` marks one host whose equilibrium pressure is not
representable as a finite positive float. That host's pressure and extent are
`None`, and evaluation continues for later hosts in the batch.

The R&H mineral formation-Gibbs nodes, JANAF crystal nodes, and JANAF gas
formation-Gibbs nodes are linearly interpolated between adjacent published
temperatures. The new iron sulfate and hydroxide reactions use formation-Gibbs
values for every term from JANAF's own elemental reference, including hematite
and water. The sulfate reactions use JANAF's SO2 formation function and the
JANAF gas rows for temperature-domain checks. No source row is extrapolated.
Sulfates using SO2 are unavailable below 500 K; anhydrite is covered through
1000 K, MgSO4 through 1400 K, and the new iron sulfates through 1500 K. Brucite
and portlandite are covered from 298.15 through 1000 K. Troilite's solid source
table is used through 1400 K; its reported transition temperatures are
preserved in the data record.

These IDs remain `unknown_host`: gypsum and bassanite lack a usable ΔfG(T)
grid; goethite has one OCR-disputed 298.15 K row; jarosite has no table in the
cited compilations; Na2SO4 has no useful oxide-decomposition ceiling before
melting; NaCl, KCl, and CaCl2 ceilings are melting followed by chloride
vaporization. MgCl2's active air reaction consumes oxygen and has a different
stoichiometry.
Chlorapatite is absent from the tables, while fluorapatite and hydroxyapatite
have formation grids without a closed decomposition assemblage or printed
melting limit.

This API does not calculate flux, area, accommodation, pump or sweep limits,
mass outputs, timestep release, or solid-release kinetics. Equilibrium is only
a thermodynamic ceiling. It does not model gas-solid back-oxidation. The
caller owns first-melt handoff and must retain unsupported host inventory.

## Sources

- Robie and Hemingway, *Thermodynamic Properties of Minerals and Related
  Substances at 298.15 K and 1 Bar Pressure and at Higher Temperatures*, USGS
  Bulletin 2131 (1995), troilite p. 156, anhydrite p. 285, magnesium sulfate
  p. 292. [USGS publication](https://pubs.usgs.gov/bul/2131/report.pdf)
- Chase, *NIST-JANAF Thermochemical Tables*, 4th ed. (1998), Ca-020, Ca-027,
  Fe-022, Fe-030, Fe-031, H-057, H-064, Mg-008, O-034, and O-029. The
  elemental oxygen standard has ΔfG° = 0 by definition. DOI:
  [10.18434/T42S31](https://doi.org/10.18434/T42S31).
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
