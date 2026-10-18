# Potential parameters and source limits

LAMMPS uses `units real`, Buckingham `buck/coul/long`, PPPM 1e-5, a 10 Å
short-range cutoff, and 1 fs timestep. Its Buckingham A and C values are in
kcal/mol and kcal mol-1 Å-6; rho is in Å.

## CaO, MgO, TiO2, Al2O3

Guillot and Sator, “A computer simulation study of natural silicate melts.
Part I: Low pressure properties,” *Geochimica et Cosmochimica Acta* 71 (2007),
1249–1265, Table 2, p. 1251, DOI: 10.1016/j.gca.2006.11.015. Charges and
Buckingham parameters are transcribed from the table. The printed A and C
values are kJ/mol; divide them by 4.184 to obtain LAMMPS real units.

| Oxide | q(cation), q(O) | cation–O A, rho, C | O–O A, rho, C |
|---|---|---|---|
| CaO | +0.945, -0.945 | 3589789.459847, 0.178, 974.533939 | 208071.223709, 0.265, 1962.277725 |
| MgO | +0.945, -0.945 | 752989.340344, 0.178, 629.115679 | same |
| TiO2 | +1.890, -0.945 | 1155950.047801, 0.178, 1067.655354 | same |
| Al2O3 | +1.4175, -0.945 | 658112.882409, 0.172, 797.385277 | same |
| SiO2 | +1.890, -0.945 | 1160089.746654, 0.161, 1067.655354 | 208071.223709, 0.265, 1962.277725 |

The paper validates natural silicate melts, not pure endmembers. These
endmember calculations are extrapolations and are capped at 3000 K.

## Cr2O3

San Miguel et al., “Molecular-dynamics simulations of premelting processes in
Cr2O3,” *Physical Review B* 58 (1998), 6057–6062, Table I, pp. 6057–6058,
DOI: 10.1103/PhysRevB.58.6057. The rigid-ion short-range parameters are
reported there from Ref. 12, Lewis and Catlow, *Journal of Physics C* 18
(1985), 1149. Formal charges are Cr +3 and O -2.

| Pair | A (kcal/mol) | rho (Å) | C (kcal mol-1 Å-6) |
|---|---:|---:|---:|
| Cr–Cr | 0 | 1 | 0 |
| Cr–O | 39989.296287 | 0.301 | 0 |
| O–O | 524957.232836 | 0.149 | 642.928078 |

The source studies a Cr2O3 surface, not bulk liquid Cp or density. Bulk-liquid
transfer is an extrapolation; the pilot did not converge sufficiently for a
usable Cp estimate.
