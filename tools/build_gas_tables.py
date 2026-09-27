#!/usr/bin/env python3
"""Build the packaged JANAF/NASA gas table and fitted condensate rows.

The vendored ``*.yaml`` records are NIST-JANAF tables harvested from the
official text files, with their upstream hashes.  NASA Glenn JSON records are
handled by the separate NASA-7 path below; some JANAF records are JSON with a
``.yaml`` suffix and some are YAML, so the loader accepts both representations.

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
NASA_GRID_STEP_K = 100.0
NASA_STANDARD_T_K = 298.15
# Same value as openimcc.gas.R_J_MOL_K; kept local so the tool does not import
# the package it builds data for.
R_J_MOL_K = 8.314462618
# ln(1.01325): converts a 1 atm standard-state entropy (S/R) to 1 bar.
LH84_ATM_TO_BAR_S_R = math.log(1.01325)
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
    ("Cr(g)", "Cr-005", "Cr", 1, 0),
    ("CrO(g)", "Cr-010", "Cr", 1, 1),
    ("CrO2(g)", "Cr-011", "Cr", 1, 2),
    ("CrO3(g)", "Cr-012", "Cr", 1, 3),
    ("V(g)", "V-005", "V", 1, 0),
    ("VO(g)", "O-026", "V", 1, 1),
    ("VO2(g)", "O-076", "V", 1, 2),
    ("Nb(g)", "Nb-005", "Nb", 1, 0),
    ("NbO(g)", "Nb-011", "Nb", 1, 1),
    ("NbO2(g)", "Nb-015", "Nb", 1, 2),
    ("Mn(g)", "Mn-005", "Mn", 1, 0),
    ("Ni(g)", "Ni-005", "Ni", 1, 0),
    ("Co(g)", "Co-005", "Co", 1, 0),
)

# The NASA cards supply the Cp(T) shape and internally consistent H/RT and S/R
# functions.  Their 298 K formation enthalpies are deliberately *not* used as
# the runtime row anchors: Lamoreaux--Hildenbrand Table 4 is the selected
# primary evaluation for those two anchors.  The final two fields identify the
# elemental reference rows used while converting the card's species G function
# to, and back from, the apparent-Gibbs convention used by the JANAF fitter.
NASA_GAS_SOURCES = (
    ("Na2O(g)", "NG-0905", "Na2O", "Na", 2, 1, "Na-005"),
    ("K2O(g)", "NG-0760", "K2O", "K", 2, 1, "K-005"),
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
CONDENSATE_SOURCES = (
    ("TiO2(l)", "O-044", "Ti", 1, 2),
    ("Cr2O3(l)", "Cr-015", "Cr", 2, 3),
    ("V2O3(l)", "O-063", "V", 2, 3),
    ("NbO2(l)", "Nb-013", "Nb", 1, 2),
)

REQUIRED_FIELDS = (
    "temperature",
    "heat_capacity",
    "entropy",
    "enthalpy_increment",
    "formation_enthalpy",
    "formation_gibbs_energy",
)

_TRANSITION_SOURCE_TABLES = frozenset({"Cr-015", "O-063", "Nb-013"})

# Sources whose liquid fit starts after FIT_T_MIN.  Cr-015 has glass Cp values
# at 1500--1700 K, while its liquid branch is Cp = 156.9 J/(mol K) from
# 1900 K through the fit range; start at the first liquid grid node.
_FIT_T_MIN_BY_TABLE = {"Cr-015": 1900.0}

# Sources whose tabulated rows stop short of FIT_T_MAX.  Cr(g), Cr-005: Cr
# boils at 2952 K, where JANAF switches the element reference to the gas and
# prints the 3000 K formation columns as "0. 0. 0.".  The harvest records that
# line only as parse-ambiguous, so no complete 3000 K row exists.  Fit to the
# last complete grid row and declare that as the row's T_max; the runtime then
# flags or refuses T above it instead of silently extrapolating.
_FIT_T_MAX_BY_TABLE = {"Cr-005": 2900.0}


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
    fit_t_min = _FIT_T_MIN_BY_TABLE.get(table_id, FIT_T_MIN)
    fit_t_max = _FIT_T_MAX_BY_TABLE.get(table_id, FIT_T_MAX)
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
        if temperature is None or not fit_t_min <= temperature <= fit_t_max:
            continue
        values = {field: _value(raw, field) for field in REQUIRED_FIELDS}
        # Premise: an ambiguous or refused source row has at least one missing
        # parsed value.  Algebra: only complete rows may enter least squares;
        # no missing value is reconstructed from a neighbour.  Unit check:
        # these fields retain NIST's K, J/(mol K), kJ/mol, and kJ/mol units.
        # Sanity: every selected record has complete rows spanning the declared
        # fit interval; an omitted normal-grid point is allowed only
        # when the source explicitly records that row as parse-ambiguous.
        if (
            temperature in ambiguous_temperatures
            and table_id not in _TRANSITION_SOURCE_TABLES
        ):
            raise ValueError(
                f"{table_id} has a parse-ambiguous row at {temperature} K"
            )
        if temperature in ambiguous_temperatures:
            # These liquid tables mark glass/liquid or crystal/liquid
            # transitions as non-data rows inside the fit interval. Omit those
            # markers; never choose a branch or reconstruct a value.
            continue
        if any(value is None for value in values.values()):
            raise ValueError(
                f"{table_id} has an incomplete/ambiguous row at {temperature} K"
            )
        rows.append({field: float(value) for field, value in values.items()})
    rows.sort(key=lambda row: row["temperature"])
    minimum_rows = 15
    if table_id == "O-063":
        minimum_rows = 14
    elif table_id == "Cr-015":
        # 1900--3000 K has 12 grid nodes; the omitted 2700 K row leaves 11.
        minimum_rows = 11
    if (
        len(rows) < minimum_rows
        or rows[0]["temperature"] != fit_t_min
        or rows[-1]["temperature"] != fit_t_max
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


def _nasa7_properties(record: dict[str, Any], temperature: float) -> dict[str, float]:
    """Evaluate one NASA Glenn nine-constant interval.

    NASA's seven heat-capacity coefficients use the exponents ``-2`` through
    ``4`` in kelvin.  The two integration constants then give

    ``Cp/R = Σ ai*T**ei``
    ``H/(R*T) = -a1/T² + a2*ln(T)/T + a3 + a4*T/2 + ... + b1/T``
    ``S/R = -a1/(2*T²) - a2/T + a3*ln(T) + a4*T + ... + b2``.

    ``H/(R*T)`` and ``S/R`` are dimensionless; multiplying the first by T
    gives H/R in kelvin and ``R*T*(H/(R*T) - S/R)`` gives G in J/mol.
    """
    interval = next(
        (
            candidate
            for candidate in record["intervals"]
            if candidate["T_min_K"]["value"] <= temperature <= candidate["T_max_K"]["value"]
        ),
        None,
    )
    if interval is None:
        # The cards start at 300 K while the thermodynamic reference is
        # 298.15 K.  NASA's polynomial is smooth over that 1.85 K gap, so use
        # the first interval for the reference evaluation; all fitted points
        # remain inside the declared 1000--6000 K interval.
        if temperature < record["intervals"][0]["T_min_K"]["value"]:
            interval = record["intervals"][0]
        else:
            raise ValueError(
                f"{record['record_id']} has no NASA-7 interval covering {temperature} K"
            )
    coefficients = [float(item["value"]) for item in interval["a_coefficients"]]
    exponents = [float(item["value"]) for item in interval["exponents"][:7]]
    if len(coefficients) != 7 or exponents != [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]:
        raise ValueError(f"{record['record_id']} is not a NASA-7 record")
    a1, a2, a3, a4, a5, a6, a7 = coefficients
    cp_R = sum(coefficient * temperature**exponent for coefficient, exponent in zip(coefficients, exponents))
    h_rt = (
        -a1 / temperature**2
        + a2 * math.log(temperature) / temperature
        + a3
        + a4 * temperature / 2.0
        + a5 * temperature**2 / 3.0
        + a6 * temperature**3 / 4.0
        + a7 * temperature**4 / 5.0
        + float(interval["b1"]["value"]) / temperature
    )
    s_R = (
        -a1 / (2.0 * temperature**2)
        - a2 / temperature
        + a3 * math.log(temperature)
        + a4 * temperature
        + a5 * temperature**2 / 2.0
        + a6 * temperature**3 / 3.0
        + a7 * temperature**4 / 4.0
        + float(interval["b2"]["value"])
    )
    return {"cp_R": cp_R, "h_rt": h_rt, "s_R": s_R}


def _janaf_element_functions(
    records: dict[str, dict[str, Any]], table_id: str, temperature: float
) -> tuple[float, float]:
    """Return one vendored JANAF element's apparent H/R and S/R functions.

    The Na/K gas records and the O2 reference record are the same element
    functions used by the existing gas rows.  Returning H/R and S/R rather
    than only G makes the standard-state conversion explicit in the NASA
    derivation below.
    """
    table = records[table_id]["table"]
    reference_row = next(
        row
        for row in table["values"]
        if _value(row, "temperature") == NASA_STANDARD_T_K
    )
    source_row = next(
        (
            row
            for row in table["values"]
            if _value(row, "temperature") == temperature
        ),
        None,
    )
    if source_row is None:
        raise ValueError(f"{table_id} has no JANAF row at {temperature} K")
    formation_enthalpy = _value(reference_row, "formation_enthalpy")
    enthalpy_increment = _value(source_row, "enthalpy_increment")
    entropy = _value(source_row, "entropy")
    if formation_enthalpy is None or enthalpy_increment is None or entropy is None:
        raise ValueError(f"{table_id} has incomplete reference data at {temperature} K")
    h_over_R = (formation_enthalpy + enthalpy_increment) * 1000.0 / R_J_MOL_K
    return h_over_R, entropy / R_J_MOL_K


def _fit_shomate_values(
    rows: list[dict[str, float]], reference: float
) -> dict[str, float]:
    """Fit the shared runtime Shomate form to apparent-Gibbs source rows."""
    # Premise: NIST Shomate Cp uses t = T/1000 and
    # Cp = A + B*t + C*t² + D*t³ + E/t².  Algebra: solve the linear least
    # squares system X*[A,B,C,D,E] = Cp.  Unit check: every matrix column is
    # dimensionless, so the coefficients retain Cp's J/(mol K) unit.  Sanity:
    # the fitted Cp is smooth across the declared runtime fit interval.
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
    F = float(
        np.mean(
            reference
            + np.array([row["enthalpy_increment"] for row in rows])
            - i_h
        )
    )
    G = float(np.mean(np.array([row["entropy"] for row in rows]) - i_s))

    # G_app is in J/mol: convert the kJ/mol enthalpy expression by 1000 before
    # subtracting T*S.  The residual is compared with the source apparent-G
    # values, not with a second evaluation of the fitted row.
    model_g = (i_h + F) * 1000.0 - temperatures * (i_s + G)
    source_g = (
        reference + np.array([row["enthalpy_increment"] for row in rows])
    ) * 1000.0 - temperatures * np.array([row["entropy"] for row in rows])
    residual_j = float(np.max(np.abs(model_g - source_g)))
    residual_log10 = float(
        np.max(
            np.abs(
                (model_g - source_g)
                / (R_J_MOL_K * temperatures * np.log(10.0))
            )
        )
    )
    return {
        "A": float(A),
        "B": float(B),
        "C": float(C),
        "D": float(D),
        "E": float(E),
        "F": F,
        "G": G,
        "max_residual_J_per_mol": residual_j,
        "max_residual_log10_K": residual_log10,
    }


def _nasa_source_rows(
    nasa_record: dict[str, Any],
    lh84: dict[str, Any],
    element_records: dict[str, dict[str, Any]],
    cation_table_id: str,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    """Make JANAF-shaped apparent-G rows from one NASA card.

    NASA-7 -> G/RT: evaluate the card's ``H/(RT)`` and ``S/R`` functions and
    form ``G_card = R*T*(H/(RT) - S/R)``.  For the reference conversion, first
    subtract the element functions ``2*G(Na/K) + 1/2*G(O2)`` from that species
    G to expose the card's formation-Gibbs degree, then add the same element
    baseline back.  This is the openimcc apparent-Gibbs convention: the
    runtime gas rows retain the species entropy and the formation enthalpy
    reference, while balanced reactions cancel the elemental baselines.  The
    element functions are taken from the vendored JANAF Na/K and O2 records,
    so no NASA reference-state convention leaks into the existing rows.

    The primary LH84 cells replace the NASA card's formation reference.  Let
    ``h0_lh/R = dfH_lh/R - (H298-H0)_lh/R``.  The card supplies only the
    temperature-dependent shape, so

        H_app(T)/R = h0_lh/R + (H298-H0)_lh/R
                     + [H_card(T)-H_card(298.15)]/R
        S_app(T)/R = S_lh/R + [S_card(T)-S_card(298.15)]/R.

    The two LH84 Hinc terms cancel algebraically in the first line, but are
    kept in the calculation to make the 0-K-to-298-K reference explicit.  The
    result has H in J/mol after multiplication by R and G in J/mol after
    subtracting T*S.  At 2000 K the hand check is exactly
    ``R*2000*(h_rt - s_R)`` plus the LH84 formation/entropy anchor shifts;
    the fitted Shomate row is compared against that anchored card evaluation
    below.  The unshifted ``G_card`` is retained only as a spot check, so the
    NASA card's disagreeing ``dfH298`` never enters the runtime anchor.
    """
    dfh_over_R = float(lh84["dfH_over_R_kK"]["value"]) * 1000.0
    # LH84 tabulates S° at a 1 atm standard state (LH84 p. 153; its O2(g)
    # S/R = 24.66 matches JANAF's 1 atm value, not the 1 bar 24.674).
    # openimcc rows are 1 bar.  For an ideal gas S(p) = S(p°) - R ln(p/p°),
    # so S°(1 bar) = S°(1 atm) + R ln(1.01325 bar / 1 bar):
    # S/R gains ln(1.01325) = 0.0131630, i.e. +0.109443 J/(mol K).
    # Units: dimensionless S/R.  Sanity: G at 2000 K drops by
    # 2000 * 0.109443 = 218.9 J/mol and each M2O(g) pressure rises by the
    # factor 1.01325, exactly the atm/bar ratio, as a pure unit change must.
    # H is pressure independent for an ideal gas, so dfH needs no change.
    s298_R = float(lh84["S_over_R"]["value"]) + LH84_ATM_TO_BAR_S_R
    hinc_over_R = float(lh84["Hinc_over_R_kK"]["value"]) * 1000.0
    h0_over_R = dfh_over_R - hinc_over_R
    card_298 = _nasa7_properties(nasa_record, NASA_STANDARD_T_K)
    card_h298_over_R = NASA_STANDARD_T_K * card_298["h_rt"]
    card_s298_R = card_298["s_R"]
    rows: list[dict[str, float]] = []
    card_g_2000 = None
    target_g_2000 = None
    for temperature in np.arange(
        FIT_T_MIN, FIT_T_MAX + NASA_GRID_STEP_K / 2.0, NASA_GRID_STEP_K
    ):
        temperature = float(temperature)
        card = _nasa7_properties(nasa_record, temperature)
        cation_h_over_R, cation_s_R = _janaf_element_functions(
            element_records, cation_table_id, temperature
        )
        oxygen_h_over_R, oxygen_s_R = _janaf_element_functions(
            element_records, "O-029", temperature
        )
        element_h_over_R = 2.0 * cation_h_over_R + 0.5 * oxygen_h_over_R
        element_s_R = 2.0 * cation_s_R + 0.5 * oxygen_s_R
        card_g = R_J_MOL_K * temperature * (card["h_rt"] - card["s_R"])
        element_g = R_J_MOL_K * (
            element_h_over_R - temperature * element_s_R
        )
        card_formation_g = card_g - element_g
        # Add the same JANAF elemental baseline back. This explicit
        # subtraction/addition proves that the fitted target remains the
        # apparent G convention used by the existing rows, rather than the
        # card's formation-Gibbs function or a second baseline-shifted value.
        card_apparent_g = card_formation_g + element_g
        h_over_R = h0_over_R + hinc_over_R + (
            temperature * card["h_rt"] - card_h298_over_R
        )
        entropy = R_J_MOL_K * (s298_R + card["s_R"] - card_s298_R)
        target_g = R_J_MOL_K * (h_over_R - temperature * (entropy / R_J_MOL_K))
        rows.append(
            {
                "temperature": temperature,
                "heat_capacity": R_J_MOL_K * card["cp_R"],
                "entropy": entropy,
                "enthalpy_increment": (h_over_R - dfh_over_R) * R_J_MOL_K / 1000.0,
                "formation_enthalpy": dfh_over_R * R_J_MOL_K / 1000.0,
                "formation_gibbs_energy": target_g / 1000.0,
                "_card_apparent_gibbs": card_apparent_g,
            }
        )
        if temperature == 2000.0:
            card_g_2000 = card_apparent_g
            target_g_2000 = target_g
    if card_g_2000 is None or target_g_2000 is None:
        raise ValueError(f"{nasa_record['record_id']} did not produce a 2000 K check")
    return rows, {
        "card_gibbs_2000_J_per_mol": card_g_2000,
        "anchored_card_gibbs_2000_J_per_mol": target_g_2000,
    }


def _fit_row(source_dir: Path, species_name: str, table_id: str, cation: str, cat_num: int, oxy_num: int) -> dict[str, str]:
    record = _load_record(source_dir / f"{table_id}.yaml")
    table = record["table"]
    if table["table_id"] != table_id:
        raise ValueError(f"{table_id}: record table_id does not match filename")
    _validate_record_identity(record, species_name, table_id, "g")
    rows = _usable_rows(table, table_id)

    all_rows = table["values"]
    reference = next(
        _value(row, "formation_enthalpy")
        for row in all_rows
        if _value(row, "temperature") == 298.15
    )
    if reference is None:
        raise ValueError(f"{table_id} has no complete 298.15 K formation enthalpy")
    fit = _fit_shomate_values(rows, float(reference))

    def number(value: float) -> str:
        return format(float(value), ".15g")

    return {
        "species_name": species_name,
        "state": "g",
        "T_interval": "1",
        "cation": cation,
        "cat_num": str(cat_num),
        "oxy_num": str(oxy_num),
        "T_min": str(int(_FIT_T_MIN_BY_TABLE.get(table_id, FIT_T_MIN))),
        "T_max": str(int(_FIT_T_MAX_BY_TABLE.get(table_id, FIT_T_MAX))),
        "A": number(fit["A"]),
        "B": number(fit["B"]),
        "C": number(fit["C"]),
        "D": number(fit["D"]),
        "E": number(fit["E"]),
        "F": number(fit["F"]),
        "G": number(fit["G"]),
        "H": "0",
        "Ref": table_id,
        "_max_residual_J_per_mol": number(fit["max_residual_J_per_mol"]),
        "_max_residual_log10_K": number(fit["max_residual_log10_K"]),
    }


def _fit_nasa_row(
    nasa_source_dir: Path,
    lh84_source_dir: Path,
    element_source_dir: Path,
    species_name: str,
    table_id: str,
    formula: str,
    cation: str,
    cat_num: int,
    oxy_num: int,
    cation_table_id: str,
) -> dict[str, str]:
    nasa_record = _load_record(nasa_source_dir / f"{table_id}.json")
    if nasa_record.get("record_id") != table_id:
        raise ValueError(f"{table_id}: NASA record record_id does not match filename")
    if nasa_record.get("formula") != formula or nasa_record.get("phase") != "gas":
        raise ValueError(f"{table_id}: NASA record identity does not match {species_name}")
    lh84_record = _load_record(lh84_source_dir / "lh84.yaml")
    lh84_rows = {
        row["species_name"]: row for row in lh84_record["rows"]
    }
    if species_name not in lh84_rows:
        raise ValueError(f"LH84 record has no {species_name} row")
    element_records = {
        source_id: _load_record(element_source_dir / f"{source_id}.yaml")
        for source_id in (cation_table_id, "O-029")
    }
    rows, checks = _nasa_source_rows(
        nasa_record,
        lh84_rows[species_name],
        element_records,
        cation_table_id,
    )
    reference = float(lh84_rows[species_name]["dfH_over_R_kK"]["value"]) * R_J_MOL_K
    fit = _fit_shomate_values(rows, reference)

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
        "A": number(fit["A"]),
        "B": number(fit["B"]),
        "C": number(fit["C"]),
        "D": number(fit["D"]),
        "E": number(fit["E"]),
        "F": number(fit["F"]),
        "G": number(fit["G"]),
        "H": "0",
        "Ref": table_id,
        "_max_residual_J_per_mol": number(fit["max_residual_J_per_mol"]),
        "_max_residual_log10_K": number(fit["max_residual_log10_K"]),
        "_card_gibbs_2000_J_per_mol": number(checks["card_gibbs_2000_J_per_mol"]),
        "_anchored_card_gibbs_2000_J_per_mol": number(
            checks["anchored_card_gibbs_2000_J_per_mol"]
        ),
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
    every complete source G_app row in the selected interval to the residual
    recorded below; each source's declared fit interval supplies its own
    liquid-branch coverage.
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
        "T_min": str(int(_FIT_T_MIN_BY_TABLE.get(table_id, FIT_T_MIN))),
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


def build_rows(
    source_dir: Path,
    nasa_source_dir: Path | None = None,
    lh84_source_dir: Path | None = None,
) -> list[dict[str, str]]:
    repository = Path(__file__).resolve().parents[1]
    nasa_source_dir = nasa_source_dir or repository / "data-src/nasa-glenn"
    lh84_source_dir = lh84_source_dir or repository / "data-src/lh84"
    rows = [_fit_row(source_dir, *source) for source in GAS_SOURCES]
    rows.extend(
        _fit_nasa_row(
            nasa_source_dir,
            lh84_source_dir,
            source_dir,
            *source,
        )
        for source in NASA_GAS_SOURCES
    )
    return rows


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
    parser.add_argument(
        "--nasa-source-dir",
        type=Path,
        default=repository / "data-src/nasa-glenn",
    )
    parser.add_argument(
        "--lh84-source-dir",
        type=Path,
        default=repository / "data-src/lh84",
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
    rows = build_rows(args.source_dir, args.nasa_source_dir, args.lh84_source_dir)
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
