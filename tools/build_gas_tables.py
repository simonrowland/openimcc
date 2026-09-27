#!/usr/bin/env python3
"""Build the packaged JANAF gas table and JANAF-fitted condensate rows.

The vendored ``*.yaml`` records are NIST-JANAF tables harvested from the
official text files, with their upstream hashes.  Some records are JSON with a ``.yaml`` suffix and some
are YAML, so the loader accepts both documented representations.

The gas table is written whole.  ``condensate.csv`` also carries transcribed
Lamoreaux/JANAF rows that this tool does not own, so only the rows named in
``CONDENSATE_SOURCES`` are replaced (or appended) and every other line is
kept byte-for-byte.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


FIT_T_MIN = 1500.0
FIT_T_MAX = 3000.0
JANAF_GRID_STEP_K = 100.0
# Same value as openimcc.gas.R_J_MOL_K; kept local so the tool does not import
# the package it builds data for.
R_J_MOL_K = 8.314462618
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
    # Titanium channels of the TiO2(l) parent.  Each record's own metadata
    # names the gas phase: Ti-006 "Ti1(g)", O-022 "O1Ti1(g)", O-046
    # "O2Ti1(g)".  They are appended after O2 so every earlier row keeps its
    # position and bytes.
    ("Ti(g)", "Ti-006", "Ti", 1, 0),
    ("TiO(g)", "O-022", "Ti", 1, 1),
    ("TiO2(g)", "O-046", "Ti", 1, 2),
    ("Al2(g)", "Al-080", "Al", 2, 0),
    ("Si2(g)", "Si-008", "Si", 2, 0),
    ("Si3(g)", "Si-009", "Si", 3, 0),
)

CONDENSATE_COLUMNS = (
    "species_name",
    "state",
    "cation",
    "cat_num",
    "oxy_num",
    "T_min",
    "T_max",
    "dH298_R",
    "dG_A",
    "dG_B",
    "dG_C",
    "dG_D",
    "dG_E",
    "Ref",
)

# Parent-oxide liquids fitted here rather than transcribed.  O-044 is the
# JANAF "O2Ti1(l)" record; over 1500--3000 K it is on its liquid branch
# (glass transition 1400 K, Cp = 100.416 J/(mol K) throughout), supercooled
# below the 2130 K melting point exactly as the JANAF liquid table states.
CONDENSATE_SOURCES = (("TiO2(l)", "O-044", "Ti", 1, 2),)

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
        # Sanity: every selected record has complete rows spanning the declared
        # 1500--3000 K interval; an omitted normal-grid point is allowed only
        # when the source explicitly records that row as parse-ambiguous.
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
    if (
        len(rows) < 15
        or rows[0]["temperature"] != FIT_T_MIN
        or rows[-1]["temperature"] != FIT_T_MAX
    ):
        raise ValueError(
            f"{table_id} does not cover the required fit interval: "
            f"{len(rows)} rows from {rows[0]['temperature'] if rows else None} "
            f"to {rows[-1]['temperature'] if rows else None} K"
        )
    for previous, current in zip(rows, rows[1:]):
        previous_temperature = previous["temperature"]
        current_temperature = current["temperature"]
        gap = current_temperature - previous_temperature
        if gap <= 0.0:
            # A repeated temperature (a phase-transition node printed twice)
            # would count twice toward the row minimum and weight that node
            # double in the least squares.
            raise ValueError(
                f"{table_id} repeats the temperature {current_temperature} K"
            )
        if gap <= JANAF_GRID_STEP_K:
            continue
        intervals = round(gap / JANAF_GRID_STEP_K)
        if not math.isclose(gap, intervals * JANAF_GRID_STEP_K):
            raise ValueError(
                f"{table_id} has an off-grid gap from "
                f"{previous_temperature} K to {current_temperature} K"
            )
        if intervals > 2:
            # The parse-ambiguity exemption covers one skipped grid point.
            # Adjacent ambiguous rows would open a gap of 200 K or more, which
            # is a coverage hole rather than one unreadable row.
            raise ValueError(
                f"{table_id} has a gap of {gap} K from {previous_temperature} K "
                f"to {current_temperature} K; at most one skipped grid point is "
                "allowed"
            )
        missing_temperatures = {
            previous_temperature + JANAF_GRID_STEP_K * index
            for index in range(1, intervals)
        }
        if not missing_temperatures <= ambiguous_temperatures:
            raise ValueError(
                f"{table_id} has an unaccounted gap from "
                f"{previous_temperature} K to {current_temperature} K"
            )
    return rows


def _validate_record_identity(
    record: dict[str, Any],
    species_name: str,
    table_id: str,
    expected_phase: str,
) -> None:
    table = record["table"]
    index_entry = table.get("index_entry", {})
    expected_formula, separator, state_suffix = species_name.rpartition("(")
    expected_state = state_suffix.removesuffix(")") if separator else ""
    formula = index_entry.get("formula")
    if formula != expected_formula:
        raise ValueError(
            f"{table_id}: record formula {formula!r} does not match "
            f"species {species_name!r}"
        )
    phase = index_entry.get("state")
    if isinstance(phase, str) and "," in phase:
        raise ValueError(
            f"{table_id}: mixed-phase record {phase!r} cannot emit "
            f"{species_name!r}"
        )
    # JANAF labels the O2 reference record ``ref`` even though the runtime
    # emits that standard gaseous reference row as O2(g).
    source_phase = "g" if formula == "O2" and phase == "ref" else phase
    if source_phase != expected_phase or expected_state != expected_phase:
        raise ValueError(
            f"{table_id}: record phase {phase!r} does not match emitted "
            f"phase {expected_phase!r} for {species_name!r}"
        )


def _fit_row(source_dir: Path, species_name: str, table_id: str, cation: str, cat_num: int, oxy_num: int) -> dict[str, str]:
    record = _load_record(source_dir / f"{table_id}.yaml")
    table = record["table"]
    if table["table_id"] != table_id:
        raise ValueError(f"{table_id}: record table_id does not match filename")
    _validate_record_identity(record, species_name, table_id, "g")
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


def _fit_condensate_row(
    source_dir: Path,
    species_name: str,
    table_id: str,
    cation: str,
    cat_num: int,
    oxy_num: int,
) -> dict[str, str]:
    """Fit one parent-oxide liquid to the runtime's condensate row form.

    Premise: ``openimcc.gas._lamor_gibbs`` evaluates a condensate row as
    ``G(T) = 1000*R*dH298_R - R*T*P(tau)`` with ``tau = T/1000`` and
    ``P(tau) = dG_A + dG_B*tau + dG_C*tau**2 + dG_D*tau**3 + dG_E*tau**4``.
    The JANAF apparent Gibbs energy used for every gas row is
    ``G_app = dfH(298) + (H - H(298)) - T*S = dfH(298) - T*Phi`` with the
    Gibbs-energy function ``Phi = S - (H - H(298))/T``.

    Algebra: matching the two forms term by term gives
    ``dH298_R = dfH(298)/R`` (dfH in kJ/mol, so 1000*R*dH298_R is J/mol) and
    ``P(tau) = Phi/R``.  Phi is taken from the tabulated S and H - H(298)
    columns, the same columns the gas residual uses, and ``P`` is the linear
    least-squares fit of ``Phi/R`` on ``[1, tau, tau**2, tau**3, tau**4]``.
    Because ``(G_fit - G_app)/(R*T*ln 10) = -(P - Phi/R)/ln 10``, this is
    the least-squares fit of the log10 K error with every fit row weighted
    equally.

    Unit check: Phi is J/(mol K) and R is J/(mol K), so P is dimensionless;
    dfH(298)/R is kJ/mol / (J/(mol K)) = 1000 K, matching the ``* 1000`` in
    the runtime.  Sanity: evaluating the row with this algebra reproduces
    every complete source G_app row to the residual recorded below, and the
    same tabulated Cp = 100.416 J/(mol K) makes Phi smooth, so a quartic is
    ample over 1500--3000 K.
    """
    record = _load_record(source_dir / f"{table_id}.yaml")
    table = record["table"]
    if table["table_id"] != table_id:
        raise ValueError(f"{table_id}: record table_id does not match filename")
    _validate_record_identity(record, species_name, table_id, "l")
    rows = _usable_rows(table, table_id)
    reference = next(
        _value(row, "formation_enthalpy")
        for row in table["values"]
        if _value(row, "temperature") == 298.15
    )
    if reference is None:
        raise ValueError(f"{table_id} has no complete 298.15 K formation enthalpy")

    temperatures = np.array([row["temperature"] for row in rows])
    tau = temperatures / 1000.0
    enthalpy_increment = np.array([row["enthalpy_increment"] for row in rows])
    entropy = np.array([row["entropy"] for row in rows])
    phi = entropy - enthalpy_increment * 1000.0 / temperatures
    design = np.column_stack((np.ones_like(tau), tau, tau**2, tau**3, tau**4))
    coefficients = np.linalg.lstsq(design, phi / R_J_MOL_K, rcond=None)[0]
    dH298_R = float(reference) / R_J_MOL_K

    model_g = 1000.0 * R_J_MOL_K * dH298_R - R_J_MOL_K * temperatures * (
        design @ coefficients
    )
    source_g = (reference + enthalpy_increment) * 1000.0 - temperatures * entropy
    residual_j = float(np.max(np.abs(model_g - source_g)))
    residual_log10 = float(
        np.max(np.abs((model_g - source_g) / (R_J_MOL_K * temperatures * np.log(10.0))))
    )

    def number(value: float) -> str:
        return format(float(value), ".15g")

    A, B, C, D, E = coefficients
    return {
        "species_name": species_name,
        "state": "l",
        "cation": cation,
        "cat_num": str(cat_num),
        "oxy_num": str(oxy_num),
        "T_min": str(int(FIT_T_MIN)),
        "T_max": str(int(FIT_T_MAX)),
        "dH298_R": number(dH298_R),
        "dG_A": number(A),
        "dG_B": number(B),
        "dG_C": number(C),
        "dG_D": number(D),
        "dG_E": number(E),
        "Ref": table_id,
        "_max_residual_J_per_mol": number(residual_j),
        "_max_residual_log10_K": number(residual_log10),
    }


def build_rows(source_dir: Path) -> list[dict[str, str]]:
    return [_fit_row(source_dir, *source) for source in GAS_SOURCES]


def build_condensate_rows(source_dir: Path) -> list[dict[str, str]]:
    return [_fit_condensate_row(source_dir, *source) for source in CONDENSATE_SOURCES]


def write_csv(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUNTIME_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({column: row[column] for column in RUNTIME_COLUMNS} for row in rows)


def merge_condensate_csv(rows: list[dict[str, str]], output: Path) -> None:
    """Replace or append the fitted rows; keep every other line unchanged."""
    lines = output.read_text(encoding="utf-8").splitlines()
    if not lines or tuple(lines[0].split(",")) != CONDENSATE_COLUMNS:
        raise ValueError(f"{output} does not have the condensate header")
    for row in rows:
        buffer = io.StringIO()
        csv.writer(buffer, lineterminator="\n").writerow(
            [row[column] for column in CONDENSATE_COLUMNS]
        )
        line = buffer.getvalue().rstrip("\n")
        matches = [
            index
            for index, existing in enumerate(lines)
            if existing.split(",", 1)[0] == row["species_name"]
        ]
        if len(matches) > 1:
            raise ValueError(f"{output} has duplicate {row['species_name']} rows")
        if matches:
            lines[matches[0]] = line
        else:
            lines.append(line)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    repository = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=repository / "data-src/janaf",
    )
    packaged_gas = repository / "src/openimcc/data/gas/gas-shomate.csv"
    parser.add_argument("--output", type=Path, default=packaged_gas)
    parser.add_argument(
        "--condensate-output",
        type=Path,
        default=None,
        help=(
            "existing condensate CSV to update in place (fitted rows only); "
            "defaults to the packaged file only when --output is also the "
            "packaged default, so a scratch --output never edits the package"
        ),
    )
    args = parser.parse_args()
    if args.condensate_output is None and args.output == packaged_gas:
        args.condensate_output = repository / "src/openimcc/data/gas/condensate.csv"
    rows = build_rows(args.source_dir)
    write_csv(rows, args.output)
    condensate_rows = build_condensate_rows(args.source_dir)
    if args.condensate_output is not None:
        merge_condensate_csv(condensate_rows, args.condensate_output)
    for row in [*rows, *condensate_rows]:
        print(
            f"{row['species_name']}: {row['_max_residual_J_per_mol']} J/mol, "
            f"{row['_max_residual_log10_K']} log10 K"
        )


if __name__ == "__main__":
    main()
