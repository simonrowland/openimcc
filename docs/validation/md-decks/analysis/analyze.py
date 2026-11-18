#!/usr/bin/env python3
"""Summarize completed oxide runs and their within-stream sampling error."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

OXIDES = {"cao": (2, 56.077), "mgo": (2, 40.304), "tio2": (3, 79.865),
          "al2o3": (5, 101.96), "cr2o3": (5, 151.99), "sio2": (3, 60.0843)}
SAMPLE_COLUMNS = 7  # step, enthalpy, oxygen MSD, PE, volume, temperature, Q6
TEMPERATURE_RELATIVE_TOLERANCE = 0.05


def integrate(y, x):
    # NumPy 2 removed trapz; retain compatibility with supported NumPy 1.x.
    trapezoid = getattr(np, "trapezoid", None)
    return trapezoid(y, x) if trapezoid is not None else np.trapz(y, x)


def heat_capacity(temperatures, enthalpies):
    """Return dH/dT from a quadratic H(T) fit and its algebraic check."""
    coefficients = np.polynomial.polynomial.polyfit(temperatures, enthalpies, 2)
    cp = (coefficients[1] + 2 * coefficients[2] * temperatures) * 1000
    # A polynomial's derivative integrates back to its endpoint difference;
    # this check is algebraic consistency, not a goodness-of-fit diagnostic.
    recovered_delta_h = integrate(cp / 1000, temperatures)
    fitted_delta_h = float(np.polynomial.polynomial.polyval(temperatures[-1], coefficients)
                           - np.polynomial.polynomial.polyval(temperatures[0], coefficients))
    return cp, float(recovered_delta_h - fitted_delta_h)


def _block_means(values, block_count=10):
    values = np.asarray(values, dtype=float)
    blocks = np.array_split(values, min(block_count, len(values)))
    return np.asarray([np.mean(block) for block in blocks if len(block)], dtype=float)


def _standard_error(block_means):
    if len(block_means) < 2:
        return None
    return float(np.std(block_means, ddof=1) / math.sqrt(len(block_means)))


def _integer_field(row, name):
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"completed point requires a positive integer {name}")
    return value


def _completed_point(row, manifest_path):
    material = row.get("material")
    if material not in OXIDES:
        raise ValueError(f"completed point has unsupported material {material!r}")
    temp_K = _integer_field(row, "temp_K")
    sample_value = row.get("samples")
    if not isinstance(sample_value, str) or not sample_value:
        raise ValueError(f"completed {material} {temp_K} K point has no sample path")
    samples_path = Path(sample_value)
    if not samples_path.is_absolute():
        samples_path = manifest_path.parent / samples_path
    if samples_path.name != f"{material}-{temp_K}.samples":
        raise ValueError(f"completed point sample path does not match {material} {temp_K} K")
    if not samples_path.is_file():
        raise ValueError(f"completed point sample file is missing: {samples_path}")

    try:
        samples = np.loadtxt(samples_path, comments="#", ndmin=2)
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read completed sample file {samples_path}: {error}") from error
    if samples.shape[1] != SAMPLE_COLUMNS or samples.shape[0] == 0:
        raise ValueError(
            f"completed sample file {samples_path} must contain {SAMPLE_COLUMNS} columns and rows"
        )
    if not np.isfinite(samples).all() or np.any(samples[:, 4] <= 0.0):
        raise ValueError(f"completed sample file has non-finite values or invalid volume: {samples_path}")

    production_steps = _integer_field(row, "production_steps")
    sample_interval = _integer_field(row, "sample_interval")
    expected_count = production_steps // sample_interval
    expected_steps = np.arange(1, expected_count + 1, dtype=float) * sample_interval
    if expected_count < 2 or not np.array_equal(samples[:, 0], expected_steps):
        raise ValueError(f"completed sample file has incomplete timesteps: {samples_path}")

    sample_temperature = float(np.mean(samples[:, 5]))
    if abs(sample_temperature - temp_K) > TEMPERATURE_RELATIVE_TOLERANCE * temp_K:
        raise ValueError(
            f"completed sample temperature {sample_temperature:.1f} K does not match "
            f"the {temp_K} K manifest point within 5%"
        )

    atom_count, formula_mass = OXIDES[material]
    actual_atoms = _integer_field(row, "actual_atoms")
    formula_units = _integer_field(row, "formula_units")
    if actual_atoms != formula_units * atom_count:
        raise ValueError(
            f"completed {material} point records {actual_atoms} atoms but "
            f"{formula_units} formula units require {formula_units * atom_count}"
        )

    enthalpy_samples = samples[:, 1] * 4.184 / formula_units
    enthalpy_blocks = _block_means(enthalpy_samples)
    density_constant = formula_units * formula_mass / 6.02214076e23 * 1e24
    density_blocks = density_constant / _block_means(samples[:, 4])
    point = {
        "material": material,
        "temp_K": temp_K,
        "actual_atoms": actual_atoms,
        "formula_units": formula_units,
        "mean_sample_temperature_K": sample_temperature,
        "enthalpy_kJ_mol_formula": float(np.mean(enthalpy_samples)),
        "enthalpy_standard_error_kJ_mol_formula": _standard_error(enthalpy_blocks),
        "density_g_cm3": float(density_constant / np.mean(samples[:, 4])),
        "density_standard_error_g_cm3": _standard_error(density_blocks),
        "sampling_blocks": len(enthalpy_blocks),
    }
    return point


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    points = []
    for row in manifest.get("points", []):
        if row.get("status") == "complete":
            points.append(_completed_point(row, manifest_path))
    points.sort(key=lambda point: (point["material"], point["temp_K"]))
    fits = {}
    for material in OXIDES:
        selected = [point for point in points if point["material"] == material]
        if len(selected) < 3:
            fits[material] = {"status": "fewer_than_three_points"}
            continue
        temperatures = np.array([point["temp_K"] for point in selected], dtype=float)
        enthalpies = np.array([point["enthalpy_kJ_mol_formula"] for point in selected], dtype=float)
        cp, algebraic_check = heat_capacity(temperatures, enthalpies)
        fits[material] = {
            "status": "fit",
            "temperature_K": temperatures.tolist(),
            "Cp_J_mol_formula_K": cp.tolist(),
            "algebraic_consistency_check_kJ_mol": algebraic_check,
        }
    sampling = dict(manifest.get("sampling", {}))
    sampling.update({
        "uncertainty_method": "standard error across up to 10 contiguous block means",
        "seed_count_per_point": 1,
        "langevin_streams_per_point": 1,
        "between_stream_uncertainty_estimated": False,
    })
    Path(args.output).write_text(json.dumps({
        "points": points,
        "heat_capacity_fits": fits,
        "sampling": sampling,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
