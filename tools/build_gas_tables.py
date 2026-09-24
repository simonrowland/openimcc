#!/usr/bin/env python3
"""Build the packaged JANAF Shomate table from the vendored records.

The vendored ``*.yaml`` records are the NIST-JANAF rows copied from the
regolith compilation.  Some records are JSON with a ``.yaml`` suffix and some
are YAML, so the loader accepts both documented representations.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


FIT_T_MIN = 1500.0
FIT_T_MAX = 3000.0
RUNTIME_COLUMNS = (
    "species_name",
    "state",
    "T_interval",
    "cation",
    "cat_num",
    "oxy_num",
    "T_min",
    "T_max",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "Ref",
)

# The table IDs are the verified rows named in the task brief.  The explicit
# metadata keeps the runtime schema stable and avoids guessing element counts
# from a formula parser when the source title contains a phase or alias.
GAS_SOURCES = (
    ("Na(g)", "Na-005", "Na", 1, 0),
    ("K(g)", "K-005", "K", 1, 0),
    ("SiO(g)", "O-012", "Si", 1, 1),
    ("Fe(g)", "Fe-008", "Fe", 1, 0),
    ("FeO(g)", "Fe-021", "Fe", 1, 1),
    ("Mg(g)", "Mg-005", "Mg", 1, 0),
    ("MgO(g)", "Mg-011", "Mg", 1, 1),
    ("SiO2(g)", "O-040", "Si", 1, 2),
    ("O(g)", "O-001", "", 0, 1),
    ("AlO(g)", "Al-074", "Al", 1, 1),
    ("AlO2(g)", "Al-077", "Al", 1, 2),
    ("Al2O(g)", "Al-092", "Al", 2, 1),
    ("Al2O2(g)", "Al-094", "Al", 2, 2),
    ("Na2(g)", "Na-011", "Na", 2, 0),
    ("NaO(g)", "Na-008", "Na", 1, 1),
    ("K2(g)", "K-011", "K", 2, 0),
    ("KO(g)", "K-008", "K", 1, 1),
    ("Si(g)", "Si-005", "Si", 1, 0),
    ("Al(g)", "Al-005", "Al", 1, 0),
    ("CaO(g)", "Ca-030", "Ca", 1, 1),
    ("Ca(g)", "Ca-006", "Ca", 1, 0),
    # O2 is the JANAF reference state, not a ``g`` row.  O-029 is the
    # explicitly labelled O2(ref) record in the verified corpus.
    ("O2(g)", "O-029", "", 0, 2),
)

REQUIRED_FIELDS = (
    "temperature",
    "heat_capacity",
    "entropy",
    "enthalpy_increment",
    "formation_enthalpy",
    "formation_gibbs_energy",
)


def _load_record(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - dev extra supplies it
            raise RuntimeError(
                f"{path} is YAML, so PyYAML is required to build the gas table"
            ) from exc
        record = yaml.safe_load(text)
        if not isinstance(record, dict):
            raise ValueError(f"{path} did not contain a mapping")
        return record


def _value(row: dict[str, Any], field: str) -> float | None:
    value = row[field]
    if not isinstance(value, dict):
        raise ValueError(f"source field {field!r} is not a value record")
    return value.get("value")


def _usable_rows(table: dict[str, Any], table_id: str) -> list[dict[str, float]]:
    ambiguous_temperatures: set[float] = set()
    for ambiguity in table.get("parse_ambiguities", []):
        raw_line = str(ambiguity.get("raw_line", "")).strip()
        token = raw_line.split("\t", 1)[0]
        try:
            ambiguous_temperatures.add(float(token))
        except ValueError:
            # A malformed nonnumeric header cannot identify a tabulated row.
            continue
    rows: list[dict[str, float]] = []
    for raw in table["values"]:
        temperature = _value(raw, "temperature")
        if temperature is None or not FIT_T_MIN <= temperature <= FIT_T_MAX:
            continue
        values = {field: _value(raw, field) for field in REQUIRED_FIELDS}
        # Premise: an ambiguous or refused source row has at least one missing
        # parsed value.  Algebra: only complete rows may enter least squares;
        # no missing value is reconstructed from a neighbour.  Unit check:
        # these fields retain NIST's K, J/(mol K), kJ/mol, and kJ/mol units.
        # Sanity: every selected gas record has at least 15 complete rows in
        # the required 1500--3000 K interval.
        if temperature in ambiguous_temperatures:
            raise ValueError(
                f"{table_id} has a parse-ambiguous row at {temperature} K"
            )
        if any(value is None for value in values.values()):
            raise ValueError(
                f"{table_id} has an incomplete/ambiguous row at {temperature} K"
            )
        rows.append({field: float(value) for field, value in values.items()})
    rows.sort(key=lambda row: row["temperature"])
    if len(rows) < 15 or rows[0]["temperature"] != FIT_T_MIN:
        raise ValueError(
            f"{table_id} does not cover the required fit interval: "
            f"{len(rows)} rows from {rows[0]['temperature'] if rows else None} K"
        )
    return rows


def _fit_row(source_dir: Path, species_name: str, table_id: str, cation: str, cat_num: int, oxy_num: int) -> dict[str, str]:
    record = _load_record(source_dir / f"{table_id}.yaml")
    table = record["table"]
    if table["table_id"] != table_id:
        raise ValueError(f"{table_id}: record table_id does not match filename")
    rows = _usable_rows(table, table_id)

    # Premise: NIST Shomate Cp uses t = T/1000 and
    # Cp = A + B*t + C*t² + D*t³ + E/t².  Algebra: solve the linear least
    # squares system X*[A,B,C,D,E] = Cp.  Unit check: every matrix column is
    # dimensionless, so the coefficients retain Cp's J/(mol K) unit.  Sanity:
    # the fitted Cp is smooth across the 1500--3000 K runtime interval.
    temperatures = np.array([row["temperature"] for row in rows])
    t = temperatures / 1000.0
    design = np.column_stack((np.ones_like(t), t, t**2, t**3, t**-2))
    coefficients = np.linalg.lstsq(
        design,
        np.array([row["heat_capacity"] for row in rows]),
        rcond=None,
    )[0]
    A, B, C, D, E = coefficients

    # Premise: integrating the fitted Cp gives the Shomate enthalpy primitive
    # I_H(t) = A*t + B*t²/2 + C*t³/3 + D*t⁴/4 - E/t.  Algebra: the omitted
    # constant F must satisfy F = dfH(298) + (H-H(298)) - I_H(t); the entropy
    # constant satisfies G = S - I_S(t), with
    # I_S(t) = A*ln(t) + B*t + C*t²/2 + D*t³/3 - E/(2*t²).  We take the
    # arithmetic mean of each constant's tabulated targets over the fit rows,
    # which is the least-squares intercept after A--E are fixed.  Unit check:
    # F is kJ/mol and G is J/(mol K).  Sanity: evaluating the resulting row
    # reproduces each source G_app row to the recorded residual below.
    i_h = A * t + B * t**2 / 2.0 + C * t**3 / 3.0 + D * t**4 / 4.0 - E / t
    i_s = A * np.log(t) + B * t + C * t**2 / 2.0 + D * t**3 / 3.0 - E / (2.0 * t**2)
    all_rows = table["values"]
    reference = next(
        _value(row, "formation_enthalpy")
        for row in all_rows
        if _value(row, "temperature") == 298.15
    )
    if reference is None:
        raise ValueError(f"{table_id} has no complete 298.15 K formation enthalpy")
    F = float(
        np.mean(
            reference
            + np.array([row["enthalpy_increment"] for row in rows])
            - i_h
        )
    )
    G = float(np.mean(np.array([row["entropy"] for row in rows]) - i_s))

    # G_app is in J/mol: convert the kJ/mol enthalpy expression by 1000 before
    # subtracting T*S.  The residual is compared against the same algebra using
    # the tabulated H-H(298) and absolute S columns, so it is independent of
    # the evaluator implementation.
    model_g = (
        (i_h + F) * 1000.0
        - temperatures * (i_s + G)
    )
    source_g = (
        (reference + np.array([row["enthalpy_increment"] for row in rows])) * 1000.0
        - temperatures * np.array([row["entropy"] for row in rows])
    )
    residual_j = float(np.max(np.abs(model_g - source_g)))
    residual_log10 = float(
        np.max(np.abs((model_g - source_g) / (8.314462618 * temperatures * np.log(10.0))))
    )

    def number(value: float) -> str:
        return format(float(value), ".15g")

    return {
        "species_name": species_name,
        "state": "g",
        "T_interval": "1",
        "cation": cation,
        "cat_num": str(cat_num),
        "oxy_num": str(oxy_num),
        "T_min": str(int(FIT_T_MIN)),
        "T_max": str(int(FIT_T_MAX)),
        "A": number(A),
        "B": number(B),
        "C": number(C),
        "D": number(D),
        "E": number(E),
        "F": number(F),
        "G": number(G),
        "H": "0",
        "Ref": table_id,
        "_max_residual_J_per_mol": number(residual_j),
        "_max_residual_log10_K": number(residual_log10),
    }


def build_rows(source_dir: Path) -> list[dict[str, str]]:
    return [_fit_row(source_dir, *source) for source in GAS_SOURCES]


def write_csv(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUNTIME_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({column: row[column] for column in RUNTIME_COLUMNS} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    repository = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=repository / "data-src/janaf",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository / "src/openimcc/data/gas/gas-shomate.csv",
    )
    args = parser.parse_args()
    rows = build_rows(args.source_dir)
    write_csv(rows, args.output)
    for row in rows:
        print(
            f"{row['species_name']}: {row['_max_residual_J_per_mol']} J/mol, "
            f"{row['_max_residual_log10_K']} log10 K"
        )


if __name__ == "__main__":
    main()
