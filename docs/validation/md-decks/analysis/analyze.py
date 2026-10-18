#!/usr/bin/env python3
"""Summarize sampled enthalpy and density for oxide temperature points."""
import argparse
import json
from pathlib import Path
import numpy as np

OXIDES = {"cao": (2, 56.077), "mgo": (2, 40.304), "tio2": (3, 79.865),
          "al2o3": (5, 101.96), "cr2o3": (5, 151.99), "sio2": (3, 60.0843)}

def integrate(y, x):
    # NumPy 2 removed trapz; retain compatibility with supported NumPy 1.x.
    trapezoid = getattr(np, "trapezoid", None)
    return trapezoid(y, x) if trapezoid is not None else np.trapz(y, x)

def heat_capacity(temperatures, enthalpies):
    """Return dH/dT from a quadratic H(T) fit, in J mol-1 K-1."""
    coefficients = np.polynomial.polynomial.polyfit(temperatures, enthalpies, 2)
    cp = (coefficients[1] + 2 * coefficients[2] * temperatures) * 1000
    # Integrate the fitted Cp over the sampled range as a consistency check.
    recovered_delta_h = integrate(cp / 1000, temperatures)
    fitted_delta_h = float(np.polynomial.polynomial.polyval(temperatures[-1], coefficients)
                           - np.polynomial.polynomial.polyval(temperatures[0], coefficients))
    return cp, float(recovered_delta_h - fitted_delta_h)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    points = []
    for row in manifest.get("points", []):
        if row.get("status") != "complete" or not row.get("samples"):
            continue
        material = row["material"]
        if material not in OXIDES:
            continue
        path = Path(row["samples"])
        if not path.exists():
            continue
        samples = np.loadtxt(path, comments="#", ndmin=2)
        if samples.shape[1] < 5:
            continue
        atom_count, formula_mass = OXIDES[material]
        formula_units = int(row.get("formula_units", row.get("atoms", 1500) // atom_count))
        enthalpy_kj = float(np.mean(samples[:, 1]) * 4.184 / formula_units)
        density = formula_units * formula_mass / 6.02214076e23 / float(np.mean(samples[:, 4])) * 1e24
        points.append({"material": material, "temp_K": int(row["temp_K"]),
                       "enthalpy_kJ_mol_formula": enthalpy_kj, "density_g_cm3": density})
    points.sort(key=lambda point: (point["material"], point["temp_K"]))
    fits = {}
    for material in OXIDES:
        selected = [point for point in points if point["material"] == material]
        if len(selected) < 3:
            fits[material] = {"status": "fewer_than_three_points"}
            continue
        temperatures = np.array([point["temp_K"] for point in selected], dtype=float)
        enthalpies = np.array([point["enthalpy_kJ_mol_formula"] for point in selected], dtype=float)
        cp, integration_residual = heat_capacity(temperatures, enthalpies)
        fits[material] = {"status": "fit", "temperature_K": temperatures.tolist(),
                          "Cp_J_mol_formula_K": cp.tolist(),
                          "integration_check_kJ_mol": integration_residual}
    Path(args.output).write_text(json.dumps({"points": points, "heat_capacity_fits": fits}, indent=2) + "\n")

if __name__ == "__main__":
    main()
