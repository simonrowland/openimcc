#!/usr/bin/env python3
"""Build a neutral random periodic starting cell for a supported oxide."""
import argparse
import math
import numpy as np

AVOGADRO = 6.02214076e23
SPECIES = {
    "cao": ("Ca", 1, 1, 40.078, 0.945), "mgo": ("Mg", 1, 1, 24.305, 0.945),
    "tio2": ("Ti", 1, 2, 47.867, 1.890), "cr2o3": ("Cr", 2, 3, 51.9961, 3.0),
    "al2o3": ("Al", 2, 3, 26.9815385, 1.4175), "sio2": ("Si", 1, 2, 28.0855, 1.890),
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("material", choices=SPECIES)
    parser.add_argument("atoms", type=int)
    parser.add_argument("output")
    parser.add_argument("--density", type=float, required=True)
    parser.add_argument("--seed", type=int, default=1216)
    args = parser.parse_args()
    cation, nc, no, mc, qc = SPECIES[args.material]
    nformula = max(1, round(args.atoms / (nc + no)))
    names = [cation] * (nc * nformula) + ["O"] * (no * nformula)
    count = len(names)
    # Convert the input mass and density to cubic angstroms for the periodic box.
    volume = nformula * (nc * mc + no * 15.999) / AVOGADRO / args.density * 1e24
    nx = max(1, round(count ** (1 / 3))); ny = nx; nz = math.ceil(count / (nx * ny))
    spacing = (volume / (nx * ny * nz)) ** (1 / 3)
    box = np.array([nx, ny, nz], dtype=float) * spacing
    rng = np.random.default_rng(args.seed)
    xyz = np.array([(i + .5, j + .5, k + .5) for i in range(nx)
                    for j in range(ny) for k in range(nz)], dtype=float)[:count] * spacing
    xyz += rng.uniform(-.08, .08, size=(count, 3))
    order = rng.permutation(count); xyz = xyz[order]; names = [names[i] for i in order]
    with open(args.output, "w") as out:
        out.write(f"Neutral {args.material} cell\n\n{count} atoms\n2 atom types\n\n")
        for axis, length in zip("xyz", box): out.write(f"0 {length:.8f} {axis}lo {axis}hi\n")
        out.write(f"\nMasses\n\n1 {mc:.8f}\n2 15.999\n\nAtoms # charge\n\n")
        qo = -qc * nc / no
        for idx, (element, pos) in enumerate(zip(names, xyz), 1):
            atom_type, charge = (2, qo) if element == "O" else (1, qc)
            out.write(f"{idx} {atom_type} {charge:.6f} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}\n")

if __name__ == "__main__": main()
