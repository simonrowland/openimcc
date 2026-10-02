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
LOW_FIT_T_MIN = 500.0
LOW_FIT_T_MAX = 1500.0
JANAF_GRID_STEP_K = 100.0
NASA_GRID_STEP_K = 100.0
NASA_STANDARD_T_K = 298.15
# The 500–1500 K JANAF interval has eleven inclusive grid nodes. That leaves at
# least ten complete nodes when a single parse-ambiguous row is skipped: five
# Shomate Cp coefficients plus a five-node margin. It covers the Plante (1259 K)
# and TS1985 (~1373 K) lower bench limits while retaining the shared 1500 K node.
# G_app = dfH298 + [H-H298] - T*S uses the gas species' own JANAF row; elemental
# reference-state transitions (including K boiling at 1032 K) do not enter it.
# The K(g) test checks the fit against K-005 on both sides of 1032 K.
SHOMATE_CP_PARAMETER_COUNT = 5
LOW_FIT_NODE_MARGIN = 5
LOW_FIT_MINIMUM_ROWS = SHOMATE_CP_PARAMETER_COUNT + LOW_FIT_NODE_MARGIN
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
    # Public phosphorus and sulfur rows from JANAF 4th ed.; these are gas-only
    # inputs, with caller-supplied P2O5(l) and S2(g) parent standards.
    ("P(g)", "P-008", "P", 1, 0),
    ("P2(g)", "P-012", "P", 2, 0),
    ("P4(g)", "P-013", "P", 4, 0),
    ("PO(g)", "O-004", "P", 1, 1),
    ("PO2(g)", "O-032", "P", 1, 2),
    ("P4O6(g)", "O-087", "P", 4, 6),
    ("P4O10(g)", "O-095", "P", 4, 10),
    ("S(g)", "S-006", "S", 1, 0),
    ("S2(g)", "S-012", "S", 2, 0),
    ("S3(g)", "S-016", "S", 3, 0),
    ("S4(g)", "S-017", "S", 4, 0),
    ("S5(g)", "S-018", "S", 5, 0),
    ("S6(g)", "S-019", "S", 6, 0),
    ("S7(g)", "S-020", "S", 7, 0),
    ("S8(g)", "S-021", "S", 8, 0),
    ("SO(g)", "O-010", "S", 1, 1),
    ("SO2(g)", "O-034", "S", 1, 2),
    ("SO3(g)", "O-058", "S", 1, 3),
    ("SSO(g)", "O-011", "S", 2, 1),
)

# First public caller-supplied trace parents. JANAF text rows take priority;
# NASA CEA cards supply only rows JANAF does not list, plus PbO(l)'s missing
# high-temperature tail.
TRACE_JANAF_GAS_SOURCES = (
    ("Li(g)", "Li-005", "Li", 1, 0),
    ("LiO(g)", "Li-011", "Li", 1, 1),
    ("Li2O(g)", "Li-017", "Li", 2, 1),
    ("Li2O2(g)", "Li-019", "Li", 2, 2),
    ("Rb(g)", "Rb-005", "Rb", 1, 0),
    ("Pb(g)", "Pb-005", "Pb", 1, 0),
    ("PbO(g)", "O-009", "Pb", 1, 1),
)
TRACE_NASA_GAS_SOURCES = (
    ("RbO(g)", "NG-1329", "RbO", "Rb", 1, 1),
    ("Rb2O(g)", "NG-1352", "Rb2O", "Rb", 2, 1),
    ("PbO2(g)", "NG-1276", "PbO2", "Pb", 1, 2),
)
TRACE_GAS_SPECIES = frozenset(
    {source[0].removesuffix("(g)") for source in TRACE_JANAF_GAS_SOURCES}
    | {source[0].removesuffix("(g)") for source in TRACE_NASA_GAS_SOURCES}
)

# Charge-balance rows are appended after every neutral row so the established
# neutral coefficient rows and their ordering remain unchanged. The text
# tables are the NIST downloads; Ca+ already has a normalized JANAF record.
ION_GAS_TEXT_SOURCES = (
    ("Na+(g)", "Na-006", "Na", 1, 0),
    ("K+(g)", "K-006", "K", 1, 0),
    ("e-(g)", "D-020", "", 0, 0),
    ("Na-(g)", "Na-007", "Na", 1, 0),
    ("K-(g)", "K-007", "K", 1, 0),
    ("O-(g)", "O-003", "", 0, 1),
    ("Al-(g)", "Al-007", "Al", 1, 0),
    ("Fe-(g)", "Fe-010", "Fe", 1, 0),
    ("Si-(g)", "Si-007", "Si", 1, 0),
    ("Ti-(g)", "Ti-008", "Ti", 1, 0),
    ("O2-(g)", "O-031", "", 0, 2),
    ("AlO-(g)", "Al-076", "Al", 1, 1),
    ("AlO2-(g)", "Al-078", "Al", 1, 2),
    ("KO-(g)", "K-009", "K", 1, 1),
    ("NaO-(g)", "Na-009", "Na", 1, 1),
    ("Cr-(g)", "Cr-007", "Cr", 1, 0),
    ("V-(g)", "V-007", "V", 1, 0),
    ("Nb-(g)", "Nb-007", "Nb", 1, 0),
    ("Li-(g)", "Li-007", "Li", 1, 0),
    ("LiO-(g)", "Li-012", "Li", 1, 1),
    ("Rb-(g)", "Rb-007", "Rb", 1, 0),
    ("Pb-(g)", "Pb-007", "Pb", 1, 0),
)
# Positive monatomic ion rows are opt-in and are admitted by the same C3
# source screen as the existing cations.
TRACE_ION_GAS_TEXT_SOURCES = (
    ("Li+(g)", "Li-006", "Li", 1, 0),
    ("Rb+(g)", "Rb-006", "Rb", 1, 0),
    ("Pb+(g)", "Pb-006", "Pb", 1, 0),
)
ION_GAS_YAML_SOURCES = (("Ca+(g)", "Ca-007", "Ca", 1, 0),)

LOW_T_GAS_SPECIES = frozenset(
    {
        "K",
        "K2",
        "KO",
        "Na",
        "NaO",
        "Na2",
        "SiO",
        "Si",
        "SiO2",
        "O",
        "O2",
        "Fe",
        "FeO",
        "Mg",
        "MgO",
        "Ca",
        "CaO",
        "Al",
        "AlO",
        "AlO2",
        "Al2O",
        "Al2O2",
        "P",
        "P2",
        "P4",
        "PO",
        "PO2",
        "P4O6",
        "P4O10",
        "S",
        "S2",
        "S3",
        "S4",
        "S5",
        "S6",
        "S7",
        "S8",
        "SO",
        "SO2",
        "SO3",
        "SSO",
    }
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

# Parent-oxide liquids fitted here rather than transcribed. O-044 is the JANAF
# "O2Ti1(l)" record. Its 1400 K row prints glass-side thermal cells followed
# by the GLASS <--> LIQUID marker; the first complete, unambiguous liquid node
# is 1500 K with Cp = 100.416 J/(mol K). Complete supercooled liquid nodes run
# from 1500-2100 K and 2300-3000 K (2200 K is parse-ambiguous). The continuation
# anchor is therefore a genuine liquid node, supercooled below 2130 K.
# Na-013 tabulates Cp = 104.600 J/(mol K) at every liquid node. The 1405.2 K
# ALPHA <--> LIQUID marker identifies the branch; complete supercooled-liquid
# nodes are 1000--1400 K; the 1500 K thermal cells are recovered from its
# formation-ambiguous row, and complete post-marker nodes are 1600--3000 K.
CONDENSATE_SOURCES = (
    ("TiO2(l)", "O-044", "Ti", 1, 2),
    ("Cr2O3(l)", "Cr-015", "Cr", 2, 3),
    ("V2O3(l)", "O-063", "V", 2, 3),
    ("NbO2(l)", "Nb-013", "Nb", 1, 2),
    ("Na2O(l)", "Na-013", "Na", 2, 1),
    ("SiO2(l)", "O-038", "Si", 1, 2),
    ("Al2O3(l)", "Al-100", "Al", 2, 3),
    ("MgO(l)", "Mg-009", "Mg", 1, 1),
    ("CaO(l)", "Ca-028", "Ca", 1, 1),
)

# O-063 V2O3(l): the 1500 K Cp cell is glass-side; both 1600 K lines are
# transition-marked and omitted. The 13 complete nodes from 1700-2300 K and
# 2500-3000 K define the high fit. The parse-ambiguous 2400 K grid point and
# 2340 K II <--> LIQUID marker are omitted; both sit on the Cp = 156.900
# J/(mol K) branch.

# Major-oxide parent liquids are fitted from these JANAF records.
# SiO2(l), O-038: the 1696 K II <--> LIQUID marker follows glass Cp values
# (74.475 at 1500 K, 80.040 at 1600 K); 85.772 J/(mol K) is constant from
# the first complete post-marker node at 1800 K through 3000 K.  The 1700 K
# row is parse-ambiguous, so the declared interval starts at 1800 K.
# Al2O3(l), Al-100: the GLASS <--> LIQUID marker is at 1350 K, but the later
# 2327 K ALPHA <--> LIQUID marker separates the alpha-solid branch from liquid.
# Cp is 192.464 J/(mol K) on both sides, so Cp alone cannot assign phase. The
# 2400 K row is parse-ambiguous; fit starts at the first complete post-marker
# liquid node, 2500 K.
# MgO(l), Mg-009: glass Cp rises from 53.693 at 1500 K to 56.019 at 2100 K;
# after the 2100.001 K GLASS <--> LIQUID marker, liquid Cp is 66.944 from the
# first complete node at 2200 K through 3000 K. The crystal/liquid marker is
# at 3105 K, outside this fit.
# CaO(l), Ca-028: glass Cp rises from 56.275 at 1500 K to 58.894 at 2100 K;
# after the 2100 K GLASS <--> LIQUID marker, liquid Cp is 62.760 from 2200 K
# through 3000 K. The crystal/liquid marker is at 3200 K, outside this fit.
# Constant-Cp supercooled-liquid continuations for default JANAF-fitted rows.
# T0 is the first complete liquid node; T_max meets the existing row's T_min.
SUPERCOOLED_LIQUID_SOURCES = (
    ("TiO2(l)", "O-044", "Ti", 1, 2, 1500.0, 1500.0),
    ("Cr2O3(l)", "Cr-015", "Cr", 2, 3, 1900.0, 1900.0),
    ("V2O3(l)", "O-063", "V", 2, 3, 1700.0, 1700.0),
    ("SiO2(l)", "O-038", "Si", 1, 2, 1800.0, 1800.0),
    ("Al2O3(l)", "Al-100", "Al", 2, 3, 2500.0, 2500.0),
    ("MgO(l)", "Mg-009", "Mg", 1, 1, 2200.0, 2200.0),
    ("CaO(l)", "Ca-028", "Ca", 1, 1, 2200.0, 2200.0),
)
ALL_SUPERCOOLED_LIQUID_SPECIES = {
    source[0] for source in SUPERCOOLED_LIQUID_SOURCES
}
SUPERCOOLED_LIQUID_REF_SUFFIX = "-SC-CP"
# These default rows now come only from their generated JANAF fits. Remove every
# prior source interval, including disjoint LAM ranges above the new fit domain.
REPLACED_DEFAULT_CONDENSATE_SPECIES = {
    "SiO2(l)", "Al2O3(l)", "MgO(l)", "CaO(l)"
}

REQUIRED_FIELDS = (
    "temperature",
    "heat_capacity",
    "entropy",
    "enthalpy_increment",
    "formation_enthalpy",
    "formation_gibbs_energy",
)

_TRANSITION_SOURCE_TABLES = frozenset({"Cr-015", "O-063", "Nb-013", "Na-013"})

# Sources whose liquid fit starts after FIT_T_MIN. These are the first complete
# grid nodes on the liquid branch; phase-marker rows are never fitted.
_FIT_T_MIN_BY_TABLE = {
    "Cr-015": 1900.0,
    "O-063": 1700.0,
    "O-038": 1800.0,
    "Al-100": 2500.0,
    "Mg-009": 2200.0,
    "Ca-028": 2200.0,
    "Na-013": 1500.0,
}

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


def _usable_rows(
    table: dict[str, Any],
    table_id: str,
    *,
    fit_t_min: float | None = None,
    fit_t_max: float | None = None,
    minimum_rows: int | None = None,
) -> list[dict[str, float]]:
    fit_t_min = (
        _FIT_T_MIN_BY_TABLE.get(table_id, FIT_T_MIN)
        if fit_t_min is None
        else fit_t_min
    )
    fit_t_max = (
        _FIT_T_MAX_BY_TABLE.get(table_id, FIT_T_MAX)
        if fit_t_max is None
        else fit_t_max
    )
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
    if table_id == "Na-013" and fit_t_min <= 1500.0 <= fit_t_max:
        for ambiguity in table.get("parse_ambiguities", []):
            cells = str(ambiguity.get("raw_line", "")).strip().split("\t")
            try:
                temperature = float(cells[0])
            except (IndexError, ValueError):
                continue
            if temperature == 1500.0:
                # This exact row is ambiguous only in its formation cells;
                # parse the thermal fields and use dfH(298) from the complete
                # 298.15 K row.
                rows.append(
                    {
                        "temperature": temperature,
                        "heat_capacity": float(cells[1]),
                        "entropy": float(cells[2]),
                        "enthalpy_increment": float(cells[4]),
                    }
                )
    rows.sort(key=lambda row: row["temperature"])
    if minimum_rows is None:
        minimum_rows = 13
        if table_id == "Cr-015":
            # 1900--3000 K has 12 grid nodes; the omitted 2700 K row leaves 11.
            minimum_rows = 11
        elif table_id == "O-038":
            # The liquid branch starts at 1800 K and has 13 complete grid nodes.
            minimum_rows = 13
        elif table_id == "Al-100":
            # The alpha/liquid marker is at 2327 K; 2500--3000 K has six nodes.
            minimum_rows = 6
        elif table_id in {"Mg-009", "Ca-028"}:
            # Each liquid branch starts at 2200 K and has 9 complete grid nodes.
            minimum_rows = 9
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
    # JANAF prints some oxygen-first canonical formulas; preserve established
    # runtime names for the phosphorus oxides and sulfur oxide.
    formula = {"O6P4": "P4O6", "O10P4": "P4O10", "O1S2": "SSO"}.get(
        formula, formula
    )
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


def _nasa7_properties(
    record: dict[str, Any], temperature: float, *, interval_index: int | None = None
) -> dict[str, float]:
    """Evaluate one NASA Glenn nine-constant interval.

    NASA's seven heat-capacity coefficients use the exponents ``-2`` through
    ``4`` in kelvin.  The two integration constants then give

    ``Cp/R = Σ ai*T**ei``
    ``H/(R*T) = -a1/T² + a2*ln(T)/T + a3 + a4*T/2 + ... + b1/T``
    ``S/R = -a1/(2*T²) - a2/T + a3*ln(T) + a4*T + ... + b2``.

    ``H/(R*T)`` and ``S/R`` are dimensionless; multiplying the first by T
    gives H/R in kelvin and ``R*T*(H/(R*T) - S/R)`` gives G in J/mol.
    """
    if interval_index is None:
        interval = next(
            (
                candidate
                for candidate in record["intervals"]
                if candidate["T_min_K"]["value"]
                <= temperature
                <= candidate["T_max_K"]["value"]
            ),
            None,
        )
    else:
        try:
            interval = record["intervals"][interval_index]
        except IndexError as exc:
            raise ValueError(
                f"{record['record_id']} has no NASA-7 interval {interval_index}"
            ) from exc
        if not interval["T_min_K"]["value"] <= temperature <= interval["T_max_K"]["value"]:
            raise ValueError(
                f"{record['record_id']} interval {interval_index} does not cover "
                f"{temperature} K"
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
    *,
    fit_t_min: float = FIT_T_MIN,
    fit_t_max: float = FIT_T_MAX,
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
    breakpoint_checks = []
    for interval_index, interval in enumerate(nasa_record["intervals"][1:], start=1):
        breakpoint = float(interval["T_min_K"]["value"])
        if not fit_t_min <= breakpoint <= fit_t_max:
            continue
        left = _nasa7_properties(
            nasa_record, breakpoint, interval_index=interval_index - 1
        )
        right = _nasa7_properties(
            nasa_record, breakpoint, interval_index=interval_index
        )
        delta_h = R_J_MOL_K * breakpoint * (right["h_rt"] - left["h_rt"])
        delta_s = R_J_MOL_K * (right["s_R"] - left["s_R"])
        if abs(delta_h) >= 1.0e-3 or abs(delta_s) >= 1.0e-6:
            raise ValueError(
                f"{nasa_record['record_id']} NASA-7 intervals are discontinuous "
                f"at {breakpoint} K: dH={delta_h} J/mol, dS={delta_s} J/(mol K)"
            )
        breakpoint_checks.append(
            {
                "temperature_K": breakpoint,
                "delta_H_J_per_mol": delta_h,
                "delta_S_J_per_mol_K": delta_s,
            }
        )
    rows: list[dict[str, float]] = []
    check_temperature = 2000.0 if fit_t_min <= 2000.0 <= fit_t_max else fit_t_max
    card_g_check = None
    target_g_check = None
    for temperature in np.arange(
        fit_t_min, fit_t_max + NASA_GRID_STEP_K / 2.0, NASA_GRID_STEP_K
    ):
        temperature = float(temperature)
        card = _nasa7_properties(nasa_record, temperature)
        card_g = R_J_MOL_K * temperature * (card["h_rt"] - card["s_R"])
        try:
            cation_h_over_R, cation_s_R = _janaf_element_functions(
                element_records, cation_table_id, temperature
            )
            oxygen_h_over_R, oxygen_s_R = _janaf_element_functions(
                element_records, "O-029", temperature
            )
        except ValueError as exc:
            if "has no JANAF row at" not in str(exc):
                raise
            # The cation tables omit some nodes at reference-state transitions
            # (Na at 1200 K, K at 1100 K). In G_card - G_elements + G_elements,
            # the elemental baseline cancels exactly, so use G_card directly at
            # those holes instead of refusing an otherwise complete NASA node.
            card_apparent_g = card_g
        else:
            element_h_over_R = 2.0 * cation_h_over_R + 0.5 * oxygen_h_over_R
            element_s_R = 2.0 * cation_s_R + 0.5 * oxygen_s_R
            element_g = R_J_MOL_K * (
                element_h_over_R - temperature * element_s_R
            )
            card_formation_g = card_g - element_g
            # Add the same JANAF elemental baseline back. This explicit
            # subtraction/addition proves that the fitted target remains the
            # apparent G convention rather than a second baseline-shifted value.
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
        if temperature == check_temperature:
            card_g_check = card_apparent_g
            target_g_check = target_g
    if card_g_check is None or target_g_check is None:
        raise ValueError(
            f"{nasa_record['record_id']} did not produce a "
            f"{check_temperature:g} K check"
        )
    return rows, {
        "check_temperature_K": check_temperature,
        "card_gibbs_check_J_per_mol": card_g_check,
        "anchored_card_gibbs_check_J_per_mol": target_g_check,
        "breakpoint_checks": breakpoint_checks,
    }


def _nasa_card_thermal_rows(
    record: dict[str, Any], temperatures: list[float]
) -> list[dict[str, float]]:
    """Return source-grid Cp, H-H(298), and S cells from one CEA card.

    Premise: NASA Glenn publishes ΔfH°(298.15 K) plus a NASA-7 H/RT and S/R
    function. Algebra: Hinc(T)=R*[T*(H/RT)(T)-Tref*(H/RT)(Tref)], then
    G_app(T)=ΔfH°(298.15 K)+Hinc(T)-T*S(T). Unit check: R*T and ΔfH are
    J/mol and S is J/(mol K). Sanity: Hinc(298.15 K)=0, so the function
    reproduces the published formation-enthalpy anchor there.

    NASA/TP-2002-211556 uses the
    thermodynamically stable elemental reference at 298.15 K; Li, Rb, and Pb
    therefore need no elemental reference-phase conversion from JANAF. Gas
    cards use 1 bar, matching JANAF. Condensed cards use a pure-liquid
    standard at 1 atm, as specified on report page 2; the caller-supplied
    parent activity is relative to that source-row standard state.
    """
    reference_h = float(record["delta_f_H_298_15"]["value"])
    h298 = _nasa7_properties(record, NASA_STANDARD_T_K)
    h298_j = R_J_MOL_K * NASA_STANDARD_T_K * h298["h_rt"]
    if abs(h298_j - reference_h) > 1.0:
        raise ValueError(
            f"{record['record_id']} H(298.15) differs from its formation anchor "
            f"by {h298_j - reference_h:g} J/mol"
        )
    rows = []
    for temperature in temperatures:
        properties = _nasa7_properties(record, temperature)
        rows.append(
            {
                "temperature": temperature,
                "heat_capacity": R_J_MOL_K * properties["cp_R"],
                "entropy": R_J_MOL_K * properties["s_R"],
                "enthalpy_increment": (
                    R_J_MOL_K
                    * (temperature * properties["h_rt"] - NASA_STANDARD_T_K * h298["h_rt"])
                    / 1000.0
                ),
                "formation_enthalpy": reference_h / 1000.0,
            }
        )
    return rows


def _fit_nasa_card_gas_row(
    nasa_source_dir: Path,
    species_name: str,
    table_id: str,
    formula: str,
    cation: str,
    cat_num: int,
    oxy_num: int,
    *,
    t_interval: int = 1,
    fit_t_min: float = FIT_T_MIN,
    fit_t_max: float = FIT_T_MAX,
    runtime_t_min: float | None = None,
    runtime_t_max: float | None = None,
) -> dict[str, str]:
    """Fit a published NASA-CEA gas card without a second formation anchor."""
    record = _load_record(nasa_source_dir / f"{table_id}.json")
    if record.get("record_id") != table_id:
        raise ValueError(f"{table_id}: NASA record record_id does not match filename")
    if record.get("formula") != formula or record.get("phase") != "gas":
        raise ValueError(f"{table_id}: NASA record identity does not match {species_name}")
    temperatures = [float(T) for T in np.arange(fit_t_min, fit_t_max + 50.0, 100.0)]
    rows = _nasa_card_thermal_rows(record, temperatures)
    reference = float(record["delta_f_H_298_15"]["value"]) / 1000.0
    fit = _fit_shomate_values(rows, reference)

    def number(value: float) -> str:
        return format(float(value), ".15g")

    return {
        "species_name": species_name,
        "state": "g",
        "T_interval": str(t_interval),
        "cation": cation,
        "cat_num": str(cat_num),
        "oxy_num": str(oxy_num),
        "T_min": number(fit_t_min if runtime_t_min is None else runtime_t_min),
        "T_max": number(fit_t_max if runtime_t_max is None else runtime_t_max),
        **{key: number(fit[key]) for key in "ABCDEFG"},
        "H": "0",
        "Ref": table_id,
        "_max_residual_J_per_mol": number(fit["max_residual_J_per_mol"]),
        "_max_residual_log10_K": number(fit["max_residual_log10_K"]),
    }


def _fit_row(
    source_dir: Path,
    species_name: str,
    table_id: str,
    cation: str,
    cat_num: int,
    oxy_num: int,
    *,
    t_interval: int = 1,
    fit_t_min: float | None = None,
    fit_t_max: float | None = None,
    minimum_rows: int | None = None,
) -> dict[str, str]:
    record = _load_record(source_dir / f"{table_id}.yaml")
    table = record["table"]
    if table["table_id"] != table_id:
        raise ValueError(f"{table_id}: record table_id does not match filename")
    _validate_record_identity(record, species_name, table_id, "g")
    rows = _usable_rows(
        table,
        table_id,
        fit_t_min=fit_t_min,
        fit_t_max=fit_t_max,
        minimum_rows=minimum_rows,
    )

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
        "T_interval": str(t_interval),
        "cation": cation,
        "cat_num": str(cat_num),
        "oxy_num": str(oxy_num),
        "T_min": str(
            int(
                _FIT_T_MIN_BY_TABLE.get(table_id, FIT_T_MIN)
                if fit_t_min is None
                else fit_t_min
            )
        ),
        "T_max": str(
            int(
                _FIT_T_MAX_BY_TABLE.get(table_id, FIT_T_MAX)
                if fit_t_max is None
                else fit_t_max
            )
        ),
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


def _janaf_text_source_rows(
    source_dir: Path, table_id: str
) -> dict[float, dict[str, float]]:
    """Read the thermal JANAF cells needed by gas and condensate fits.

    A source row is useful when Cp, S, and H-H(298) are intact, even if its
    printed formation columns are blank after an elemental reference-state
    transition. The 298.15 K row separately supplies the formation-enthalpy
    anchor. Rows with malformed thermal cells are skipped; no field is
    reconstructed from neighboring temperatures.
    """
    path = source_dir / f"{table_id}.txt"
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3 or lines[1].split("\t")[0] != "T(K)":
        raise ValueError(f"{table_id}: expected a NIST tab-delimited table")
    parsed: dict[float, dict[str, float]] = {}
    for line in lines[2:]:
        cells = line.split("\t")
        if len(cells) < 5:
            continue
        try:
            temperature, cp, entropy, _minus_g_over_t, h_increment = (
                float(value) for value in cells[:5]
            )
        except ValueError:
            continue
        row = {
            "temperature": temperature,
            "heat_capacity": cp,
            "entropy": entropy,
            "enthalpy_increment": h_increment,
        }
        for field, index in (("formation_enthalpy", 5), ("formation_gibbs_energy", 6)):
            try:
                row[field] = float(cells[index])
            except (IndexError, ValueError):
                pass
        parsed[temperature] = row
    return parsed


def _fit_janaf_text_row(
    source_dir: Path,
    species_name: str,
    table_id: str,
    cation: str,
    cat_num: int,
    oxy_num: int,
    *,
    t_interval: int = 1,
    fit_t_min: float = 1200.0,
    fit_t_max: float = 3000.0,
    runtime_t_min: float | None = None,
    runtime_t_max: float | None = None,
) -> dict[str, str]:
    """Fit NIST gas cells to the Shomate form used by the runtime.

    JANAF's ideal-gas standard state is 0.1 MPa (1 bar). The fitted target is
    G_app = Hf(298.15) + [H-H(298.15)] - T*S, using the source's own 298 K
    elemental reference convention. Hf is kJ/mol, H-H(298) is kJ/mol, and S
    is J/(mol K); _fit_shomate_values converts enthalpy to J/mol before
    subtracting T*S. If the low interval is fitted over 500-1500 K but exposed
    only from 1200 K, that lower declaration is a runtime domain boundary,
    not an extrapolated fit.
    """
    parsed = _janaf_text_source_rows(source_dir, table_id)
    reference = parsed.get(298.15, {}).get("formation_enthalpy")
    if reference is None:
        raise ValueError(f"{table_id}: no complete 298.15 K formation enthalpy")
    required_nodes = np.arange(fit_t_min, fit_t_max + 50.0, 100.0)
    missing = [float(T) for T in required_nodes if float(T) not in parsed]
    if missing:
        raise ValueError(f"{table_id}: missing 100 K fit nodes {missing!r}")
    rows = [parsed[float(T)] for T in required_nodes]
    fit = _fit_shomate_values(rows, float(reference))

    def number(value: float) -> str:
        return format(float(value), ".15g")

    return {
        "species_name": species_name,
        "state": "g",
        "T_interval": str(t_interval),
        "cation": cation,
        "cat_num": str(cat_num),
        "oxy_num": str(oxy_num),
        "T_min": number(fit_t_min if runtime_t_min is None else runtime_t_min),
        "T_max": number(fit_t_max if runtime_t_max is None else runtime_t_max),
        **{key: number(fit[key]) for key in "ABCDEFG"},
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
    *,
    t_interval: int = 1,
    fit_t_min: float = FIT_T_MIN,
    fit_t_max: float = FIT_T_MAX,
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
        fit_t_min=fit_t_min,
        fit_t_max=fit_t_max,
    )
    reference = float(lh84_rows[species_name]["dfH_over_R_kK"]["value"]) * R_J_MOL_K
    fit = _fit_shomate_values(rows, reference)

    def number(value: float) -> str:
        return format(float(value), ".15g")

    return {
        "species_name": species_name,
        "state": "g",
        "T_interval": str(t_interval),
        "cation": cation,
        "cat_num": str(cat_num),
        "oxy_num": str(oxy_num),
        "T_min": str(int(fit_t_min)),
        "T_max": str(int(fit_t_max)),
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
        "_card_gibbs_check_J_per_mol": number(
            checks["card_gibbs_check_J_per_mol"]
        ),
        "_anchored_card_gibbs_check_J_per_mol": number(
            checks["anchored_card_gibbs_check_J_per_mol"]
        ),
        "_check_temperature_K": number(checks["check_temperature_K"]),
        "_breakpoint_checks": json.dumps(checks["breakpoint_checks"], sort_keys=True),
    }


def _fit_condensate_values(
    species_name: str,
    table_id: str,
    cation: str,
    cat_num: int,
    oxy_num: int,
    source_rows: dict[float, dict[str, float]],
    reference_kJ_mol: float,
    *,
    fit_t_min: float,
    fit_t_max: float,
    runtime_t_min: float,
    runtime_t_max: float,
    ref: str | None = None,
) -> dict[str, str]:
    """Fit source Cp/S/H increments to the existing condensate row form.

    Premise: the runtime stores G_app = 1000*R*dH298_R - R*T*P(tau), where
    tau=T/1000 and P is a quartic. Algebra: source cells give
    G_app=dH298*1000 + Hinc*1000 - T*S, so P(tau) fits
    (S-Hinc*1000/T)/R. Unit check: the fitted polynomial is dimensionless,
    dH298_R is kelvin, and both Gibbs expressions are J/mol. Sanity: the
    independently reconstructed source G values set the recorded fit
    residual in J/mol and log10(K).
    """
    temperatures = [
        float(T)
        for T in np.arange(fit_t_min, fit_t_max + 50.0, JANAF_GRID_STEP_K)
    ]
    missing = [T for T in temperatures if T not in source_rows]
    if missing:
        raise ValueError(f"{table_id}: missing condensate source nodes {missing!r}")
    rows = [source_rows[T] for T in temperatures]
    tau = np.asarray(temperatures, dtype=float) / 1000.0
    enthalpy_increment = np.asarray([row["enthalpy_increment"] for row in rows])
    entropy = np.asarray([row["entropy"] for row in rows])
    phi = entropy - enthalpy_increment * 1000.0 / np.asarray(temperatures)
    design = np.column_stack((np.ones_like(tau), tau, tau**2, tau**3, tau**4))
    coefficients = np.linalg.lstsq(design, phi / R_J_MOL_K, rcond=None)[0]
    dH298_R = reference_kJ_mol / R_J_MOL_K
    model_g = 1000.0 * R_J_MOL_K * dH298_R - R_J_MOL_K * np.asarray(temperatures) * (
        design @ coefficients
    )
    source_g = (
        reference_kJ_mol + enthalpy_increment
    ) * 1000.0 - np.asarray(temperatures) * entropy
    residual_j = float(np.max(np.abs(model_g - source_g)))
    residual_log10 = float(
        np.max(
            np.abs(
                (model_g - source_g)
                / (R_J_MOL_K * np.asarray(temperatures) * np.log(10.0))
            )
        )
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
        "T_min": number(runtime_t_min),
        "T_max": number(runtime_t_max),
        "dH298_R": number(dH298_R),
        "dG_A": number(A),
        "dG_B": number(B),
        "dG_C": number(C),
        "dG_D": number(D),
        "dG_E": number(E),
        "Ref": table_id if ref is None else ref,
        "_max_residual_J_per_mol": number(residual_j),
        "_max_residual_log10_K": number(residual_log10),
    }


def _fit_janaf_text_condensate_row(
    source_dir: Path,
    species_name: str,
    table_id: str,
    cation: str,
    cat_num: int,
    oxy_num: int,
    *,
    fit_t_min: float,
    fit_t_max: float,
    runtime_t_min: float,
    runtime_t_max: float,
) -> dict[str, str]:
    parsed = _janaf_text_source_rows(source_dir, table_id)
    reference = parsed.get(298.15, {}).get("formation_enthalpy")
    if reference is None:
        raise ValueError(f"{table_id}: no complete 298.15 K formation enthalpy")
    return _fit_condensate_values(
        species_name,
        table_id,
        cation,
        cat_num,
        oxy_num,
        parsed,
        float(reference),
        fit_t_min=fit_t_min,
        fit_t_max=fit_t_max,
        runtime_t_min=runtime_t_min,
        runtime_t_max=runtime_t_max,
    )


def _fit_nasa_card_condensate_row(
    nasa_source_dir: Path,
    species_name: str,
    table_id: str,
    formula: str,
    cation: str,
    cat_num: int,
    oxy_num: int,
    *,
    fit_t_min: float = 1200.0,
    fit_t_max: float = 3000.0,
    runtime_t_min: float | None = None,
    runtime_t_max: float | None = None,
) -> dict[str, str]:
    record = _load_record(nasa_source_dir / f"{table_id}.json")
    if (
        record.get("record_id") != table_id
        or record.get("formula") != formula
        or record.get("phase") != "liquid"
    ):
        raise ValueError(f"{table_id}: NASA record identity does not match {species_name}")
    temperatures = [
        float(T)
        for T in np.arange(fit_t_min, fit_t_max + 50.0, NASA_GRID_STEP_K)
    ]
    # This CEA liquid card starts above its 298 K condensed reference state.
    # Preserve its integrated H(T) and S(T) functions directly instead of
    # extrapolating the liquid polynomial to 298 K and treating that value as
    # the formation anchor. Writing Hinc(T) = [H_card(T)-Hf_card(298)]/1000
    # makes Hf+Hinc-T*S exactly the card's G(T) at every fitted liquid node.
    # Unit check: Hinc is converted from J/mol to kJ/mol for the source-row
    # fitter; Hf and T*S use the same kJ/mol scale there. Sanity: the equation
    # simplifies to R*T*h_rt - T*R*s_R, the card's G(T), at each source node.
    reference_j = float(record["delta_f_H_298_15"]["value"])
    source_rows = {
        temperature: {
            "temperature": temperature,
            "heat_capacity": R_J_MOL_K * properties["cp_R"],
            "entropy": R_J_MOL_K * properties["s_R"],
            "enthalpy_increment": (
                R_J_MOL_K * temperature * properties["h_rt"] - reference_j
            )
            / 1000.0,
        }
        for temperature in temperatures
        for properties in [_nasa7_properties(record, temperature)]
    }
    return _fit_condensate_values(
        species_name,
        table_id,
        cation,
        cat_num,
        oxy_num,
        source_rows,
        reference_j / 1000.0,
        fit_t_min=fit_t_min,
        fit_t_max=fit_t_max,
        runtime_t_min=fit_t_min if runtime_t_min is None else runtime_t_min,
        runtime_t_max=fit_t_max if runtime_t_max is None else runtime_t_max,
    )


def _fit_janaf_condensate_with_nasa_tail(
    janaf_source_dir: Path,
    nasa_source_dir: Path,
    species_name: str,
    janaf_table_id: str,
    nasa_table_id: str,
    cation: str,
    cat_num: int,
    oxy_num: int,
    *,
    anchor_temperature: float,
    fit_t_min: float = 1200.0,
    fit_t_max: float = 3000.0,
    runtime_t_min: float | None = None,
    runtime_t_max: float | None = None,
) -> dict[str, str]:
    """Keep JANAF's anchor and cells, using the NASA card only above its tail.

    Premise: JANAF O-007 supplies complete PbO(l) thermal cells through
    2500 K, while the published CEA liquid card continues past 3000 K.
    Algebra: anchor the CEA H(T)-H(T0) and S(T)-S(T0) differences at JANAF's
    T0 cells, then use
    G_app=Hf_JANAF(298)+Hinc-T*S. Unit check: the enthalpy difference is
    converted from J/mol to kJ/mol before adding to JANAF's increment; entropy
    remains J/(mol K). At 2500 K the splice has zero ΔH, ΔS, and ΔG; by 3000 K
    its ΔG is 0.017 J/mol relative to continuing JANAF Cp=65 J/(mol K). NASA's
    condensed standard pressure is 1 atm while the
    JANAF anchor is 1 bar; the source-row activity convention is documented
    and no pressure correction is claimed without a sourced liquid molar
    volume. Sanity: Cp, Hinc, and S remain continuous at T0.
    """
    janaf_rows = _janaf_text_source_rows(janaf_source_dir, janaf_table_id)
    reference = janaf_rows.get(298.15, {}).get("formation_enthalpy")
    anchor = janaf_rows.get(anchor_temperature)
    if reference is None or anchor is None:
        raise ValueError(f"{janaf_table_id}: missing JANAF anchor data")
    record = _load_record(nasa_source_dir / f"{nasa_table_id}.json")
    if record.get("phase") != "liquid" or record.get("formula") != "PbO":
        raise ValueError(f"{nasa_table_id}: expected a PbO liquid NASA card")
    nasa_anchor = _nasa7_properties(record, anchor_temperature)
    for temperature in np.arange(anchor_temperature + 100.0, fit_t_max + 50.0, 100.0):
        temperature = float(temperature)
        properties = _nasa7_properties(record, temperature)
        janaf_rows[temperature] = {
            "temperature": temperature,
            "heat_capacity": R_J_MOL_K * properties["cp_R"],
            "enthalpy_increment": anchor["enthalpy_increment"]
            + R_J_MOL_K
            * (
                temperature * properties["h_rt"]
                - anchor_temperature * nasa_anchor["h_rt"]
            )
            / 1000.0,
            "entropy": anchor["entropy"]
            + R_J_MOL_K * (properties["s_R"] - nasa_anchor["s_R"]),
        }
    return _fit_condensate_values(
        species_name,
        janaf_table_id,
        cation,
        cat_num,
        oxy_num,
        janaf_rows,
        float(reference),
        fit_t_min=fit_t_min,
        fit_t_max=fit_t_max,
        runtime_t_min=fit_t_min if runtime_t_min is None else runtime_t_min,
        runtime_t_max=fit_t_max if runtime_t_max is None else runtime_t_max,
    )


def _fit_condensate_row(
    source_dir: Path,
    species_name: str,
    table_id: str,
    cation: str,
    cat_num: int,
    oxy_num: int,
    *,
    fit_t_min: float | None = None,
    fit_t_max: float | None = None,
    minimum_rows: int | None = None,
    runtime_t_min: float | None = None,
    runtime_t_max: float | None = None,
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
    rows = _usable_rows(
        table,
        table_id,
        fit_t_min=fit_t_min,
        fit_t_max=fit_t_max,
        minimum_rows=minimum_rows,
    )
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
        "T_min": str(
            int(
                _FIT_T_MIN_BY_TABLE.get(table_id, FIT_T_MIN)
                if runtime_t_min is None
                else runtime_t_min
            )
        ),
        "T_max": str(int(FIT_T_MAX if runtime_t_max is None else runtime_t_max)),
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


def _fit_supercooled_liquid_row(
    source_dir: Path,
    species_name: str,
    table_id: str,
    cation: str,
    cat_num: int,
    oxy_num: int,
    T0: float,
    runtime_t_max: float,
) -> dict[str, str]:
    """Fit the generated constant-Cp continuation to the condensate form.

    Premise: JANAF gives the first complete liquid-node enthalpy increment,
    entropy, and liquid Cp at T0.  For the labelled continuation,
    H(T)=H(T0)+Cp*(T-T0) and S(T)=S(T0)+Cp*ln(T/T0), then
    G_app=dfH298+[H(T)-H298]-T*S(T).  The fitted polynomial represents this
    generated function; it is not JANAF source data below the liquid branch.

    Unit check: Cp*(T-T0) is J/mol and is divided by 1000 for JANAF's
    kJ/mol enthalpy increment; Cp*ln(T/T0) is J/(mol K); both terms in G
    are J/mol.  The condensate fit matches Phi/R with Phi=S-(H-H298)/T.
    Sanity: the analytic continuation has the source H, S, G, and -dG/dT
    exactly at T0.  At source nodes below a glass/liquid marker, differences
    from JANAF's printed glass rows are expected because this deliberately
    continues the liquid rather than the glass.  Any genuine supercooled
    liquid nodes in the JANAF table are checked by the caller's regression
    tests.
    """
    yaml_path = source_dir / f"{table_id}.yaml"
    if yaml_path.is_file():
        record = _load_record(yaml_path)
        table = record["table"]
        if table["table_id"] != table_id:
            raise ValueError(f"{table_id}: record table_id does not match filename")
        _validate_record_identity(record, species_name, table_id, "l")
        source_row = next(
            (row for row in table["values"] if _value(row, "temperature") == T0),
            None,
        )
        if source_row is None:
            raise ValueError(f"{table_id}: no complete liquid anchor at {T0:g} K")
        ambiguous_temperatures = set()
        for ambiguity in table.get("parse_ambiguities", []):
            try:
                ambiguous_temperatures.add(
                    float(ambiguity["raw_line"].split("\t", 1)[0])
                )
            except ValueError:
                continue
        if T0 in ambiguous_temperatures:
            raise ValueError(
                f"{table_id}: liquid anchor at {T0:g} K is parse-ambiguous"
            )
        cp = float(_value(source_row, "heat_capacity"))
        entropy_0 = float(_value(source_row, "entropy"))
        enthalpy_0 = float(_value(source_row, "enthalpy_increment"))
        reference = next(
            _value(row, "formation_enthalpy")
            for row in table["values"]
            if _value(row, "temperature") == 298.15
        )
    else:
        parsed = _janaf_text_source_rows(source_dir, table_id)
        source_row = parsed.get(T0)
        if source_row is None:
            raise ValueError(f"{table_id}: no complete liquid anchor at {T0:g} K")
        cp = source_row["heat_capacity"]
        entropy_0 = source_row["entropy"]
        enthalpy_0 = source_row["enthalpy_increment"]
        reference = parsed.get(298.15, {}).get("formation_enthalpy")
    if reference is None:
        raise ValueError(f"{table_id} has no complete 298.15 K formation enthalpy")

    # Fit on JANAF's 100 K grid plus both interval endpoints.  Evaluate the
    # resulting polynomial every kelvin below to report the actual maximum.
    fit_temperatures = sorted(
        {1200.0, runtime_t_max}
        | set(float(T) for T in range(1200, int(runtime_t_max) + 1, 100))
    )

    def phi(temperature: float) -> float:
        enthalpy_increment = enthalpy_0 + cp * (temperature - T0) / 1000.0
        entropy = entropy_0 + cp * math.log(temperature / T0)
        return entropy - enthalpy_increment * 1000.0 / temperature

    tau = np.asarray(fit_temperatures, dtype=float) / 1000.0
    design = np.column_stack((np.ones_like(tau), tau, tau**2, tau**3, tau**4))
    coefficients = np.linalg.lstsq(
        design, np.asarray([phi(T) for T in fit_temperatures]) / R_J_MOL_K,
        rcond=None,
    )[0]
    sample_temperatures = np.linspace(1200.0, runtime_t_max, int(runtime_t_max - 1200) + 1)
    sample_tau = sample_temperatures / 1000.0
    sample_design = np.column_stack(
        (np.ones_like(sample_tau), sample_tau, sample_tau**2, sample_tau**3, sample_tau**4)
    )
    g_fit = 1000.0 * reference - R_J_MOL_K * sample_temperatures * (
        sample_design @ coefficients
    )
    g_generated = np.asarray([
        (reference + enthalpy_0 + cp * (T - T0) / 1000.0) * 1000.0
        - T * (entropy_0 + cp * math.log(T / T0))
        for T in sample_temperatures
    ])
    residual_j = float(np.max(np.abs(g_fit - g_generated)))
    residual_log10 = float(
        np.max(np.abs((g_fit - g_generated) /
                      (R_J_MOL_K * sample_temperatures * math.log(10.0))))
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
        "T_min": "1200",
        "T_max": number(runtime_t_max),
        "dH298_R": number(float(reference) / R_J_MOL_K),
        "dG_A": number(A),
        "dG_B": number(B),
        "dG_C": number(C),
        "dG_D": number(D),
        "dG_E": number(E),
        "Ref": table_id + SUPERCOOLED_LIQUID_REF_SUFFIX,
        "_max_residual_J_per_mol": number(residual_j),
        "_max_residual_log10_K": number(residual_log10),
        "_T0_K": number(T0),
        "_Cp_l_J_molK": number(cp),
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
    rows.extend(
        _fit_row(
            source_dir,
            *source,
            t_interval=2,
            fit_t_min=LOW_FIT_T_MIN,
            fit_t_max=LOW_FIT_T_MAX,
            minimum_rows=LOW_FIT_MINIMUM_ROWS,
        )
        for source in GAS_SOURCES
        if source[0].removesuffix("(g)") in LOW_T_GAS_SPECIES
    )
    rows.extend(
        _fit_nasa_row(
            nasa_source_dir,
            lh84_source_dir,
            source_dir,
            *source,
            t_interval=2,
            fit_t_min=LOW_FIT_T_MIN,
            fit_t_max=LOW_FIT_T_MAX,
        )
        for source in NASA_GAS_SOURCES
    )
    rows.extend(
        _fit_row(
            source_dir,
            *source,
            fit_t_min=1200.0,
            fit_t_max=3000.0,
            minimum_rows=19,
        )
        for source in ION_GAS_YAML_SOURCES
    )
    rows.extend(
        _fit_janaf_text_row(source_dir, *source)
        for source in ION_GAS_TEXT_SOURCES
    )
    # These optional trace rows are appended after the established gas and ion
    # rows so each existing coefficient row remains byte-for-byte unchanged.
    rows.extend(
        _fit_janaf_text_row(
            source_dir,
            *source,
            t_interval=1,
            fit_t_min=FIT_T_MIN,
            fit_t_max=FIT_T_MAX,
        )
        for source in TRACE_JANAF_GAS_SOURCES
    )
    rows.extend(
        _fit_nasa_card_gas_row(
            nasa_source_dir,
            *source,
            t_interval=1,
            fit_t_min=FIT_T_MIN,
            fit_t_max=FIT_T_MAX,
        )
        for source in TRACE_NASA_GAS_SOURCES
    )
    rows.extend(
        _fit_janaf_text_row(
            source_dir,
            *source,
            t_interval=2,
            fit_t_min=LOW_FIT_T_MIN,
            fit_t_max=LOW_FIT_T_MAX,
            runtime_t_min=1200.0,
        )
        for source in TRACE_JANAF_GAS_SOURCES
    )
    rows.extend(
        _fit_nasa_card_gas_row(
            nasa_source_dir,
            *source,
            t_interval=2,
            fit_t_min=LOW_FIT_T_MIN,
            fit_t_max=LOW_FIT_T_MAX,
            runtime_t_min=1200.0,
        )
        for source in TRACE_NASA_GAS_SOURCES
    )
    rows.extend(
        _fit_janaf_text_row(source_dir, *source)
        for source in TRACE_ION_GAS_TEXT_SOURCES
    )
    return rows


def build_condensate_rows(
    source_dir: Path, nasa_source_dir: Path | None = None
) -> list[dict[str, str]]:
    nasa_source_dir = nasa_source_dir or (
        Path(__file__).resolve().parents[1] / "data-src/nasa-glenn"
    )
    rows = []
    for source in CONDENSATE_SOURCES:
        if source[1] == "Na-013":
            rows.append(
                _fit_condensate_row(
                    source_dir,
                    *source,
                    runtime_t_min=1500.0,
                )
            )
        else:
            rows.append(_fit_condensate_row(source_dir, *source))
    # Na-013 is a liquid table below its 1405.2 K ALPHA <--> LIQUID marker.
    # Both fits share the recovered 1500 K thermal row so the runtime seam is
    # constrained by the declared endpoint on each side.
    rows.append(
        _fit_condensate_row(
            source_dir,
            "Na2O(l)",
            "Na-013",
            "Na",
            2,
            1,
            fit_t_min=1000.0,
            fit_t_max=1500.0,
            minimum_rows=6,
            runtime_t_min=1200.0,
            runtime_t_max=1500.0,
        )
    )
    rows.append(
        _fit_condensate_row(
            source_dir,
            "NbO2(l)",
            "Nb-013",
            "Nb",
            1,
            2,
            fit_t_min=1100.0,
            fit_t_max=1500.0,
            minimum_rows=5,
            runtime_t_min=1200.0,
            runtime_t_max=1500.0,
        )
    )
    rows.extend(
        _fit_supercooled_liquid_row(source_dir, *source)
        for source in SUPERCOOLED_LIQUID_SOURCES
    )
    # JANAF's Li2O liquid cells begin at 700 K, before the 1843 K melting
    # marker. Fit the printed liquid H/S/Cp nodes directly on each interval.
    for fit_t_min, fit_t_max, runtime_t_min in (
        (700.0, 1400.0, 1200.0),
        (1400.0, 2000.0, 1400.0),
        (2000.0, 3000.0, 2000.0),
    ):
        rows.append(
            _fit_janaf_text_condensate_row(
                source_dir,
                "Li2O(l)",
                "Li-015",
                "Li",
                2,
                1,
                fit_t_min=fit_t_min,
                fit_t_max=fit_t_max,
                runtime_t_min=runtime_t_min,
                runtime_t_max=fit_t_max,
            )
        )
    # NASA's constant-Cp liquid card has ln(T) entropy curvature. The 2000 K
    # breakpoint keeps all three Rb2O fits below the 10 J/mol source-residual gate.
    for fit_t_min, fit_t_max in (
        (1200.0, 1500.0),
        (1500.0, 2000.0),
        (2000.0, 3000.0),
    ):
        rows.append(
            _fit_nasa_card_condensate_row(
                nasa_source_dir,
                "Rb2O(l)",
                "NG-1841",
                "Rb2O",
                "Rb",
                2,
                1,
                fit_t_min=fit_t_min,
                fit_t_max=fit_t_max,
            )
        )
    for fit_t_min, fit_t_max in ((1200.0, 1500.0), (1500.0, 3000.0)):
        rows.append(
            _fit_janaf_condensate_with_nasa_tail(
                source_dir,
                nasa_source_dir,
                "PbO(l)",
                "O-007",
                "NG-1801",
                "Pb",
                1,
                1,
                anchor_temperature=2500.0,
                fit_t_min=fit_t_min,
                fit_t_max=fit_t_max,
            )
        )
    return rows


def write_csv(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUNTIME_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({column: row[column] for column in RUNTIME_COLUMNS} for row in rows)


def merge_condensate_csv(rows: list[dict[str, str]], output: Path) -> None:
    """Replace generated intervals only; retain unrelated source intervals."""
    lines = output.read_text(encoding="utf-8").splitlines()
    if not lines or tuple(lines[0].split(",")) != CONDENSATE_COLUMNS:
        raise ValueError(f"{output} does not have the condensate header")
    grouped_rows: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped_rows.setdefault(row["species_name"], []).append(row)
    generated_continuations = {
        row["species_name"]
        for row in rows
        if row["Ref"].endswith(SUPERCOOLED_LIQUID_REF_SUFFIX)
    }
    lines = [
        lines[0],
        *(
            line for line in lines[1:]
            if (fields := next(csv.reader([line])))[0]
            not in REPLACED_DEFAULT_CONDENSATE_SPECIES
            and not (
                fields[0] in ALL_SUPERCOOLED_LIQUID_SPECIES
                and fields[-1].endswith(SUPERCOOLED_LIQUID_REF_SUFFIX)
                and fields[0] not in generated_continuations
            )
        ),
    ]
    for species_name, species_rows in grouped_rows.items():
        serialized_rows = []
        for row in species_rows:
            buffer = io.StringIO()
            csv.writer(buffer, lineterminator="\n").writerow(
                [row[column] for column in CONDENSATE_COLUMNS]
            )
            serialized_rows.append(buffer.getvalue().rstrip("\n"))
        generated_starts = {str(row["T_min"]) for row in species_rows}
        generated_intervals = [
            (float(row["T_min"]), float(row["T_max"])) for row in species_rows
        ]
        generated_refs: dict[str, int] = {}
        for row in species_rows:
            generated_refs[row["Ref"]] = generated_refs.get(row["Ref"], 0) + 1
        matching_intervals = []
        for index, existing in enumerate(lines):
            fields = next(csv.reader([existing]))
            if fields[0] != species_name:
                continue
            replaced_interval = fields[5] in generated_starts
            existing_interval = (float(fields[5]), float(fields[6]))
            # Generated fits supersede overlapping stale intervals even when
            # their reference or lower bound changed.
            replaced_interval |= any(
                max(existing_interval[0], generated_min)
                < min(existing_interval[1], generated_max)
                for generated_min, generated_max in generated_intervals
            )
            # A singleton generated reference owns one complete interval. Let
            # it replace a stale interval whose T_min changed, as for O-063.
            replaced_interval |= generated_refs.get(fields[-1]) == 1
            if replaced_interval:
                matching_intervals.append(index)
        if matching_intervals:
            insert_at = matching_intervals[0]
            for index in reversed(matching_intervals):
                del lines[index]
            lines[insert_at:insert_at] = serialized_rows
        else:
            lines.extend(serialized_rows)
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
    condensate_rows = build_condensate_rows(args.source_dir, args.nasa_source_dir)
    if args.condensate_output is not None:
        merge_condensate_csv(condensate_rows, args.condensate_output)
    for row in [*rows, *condensate_rows]:
        print(
            f"{row['species_name']}: {row['_max_residual_J_per_mol']} J/mol, "
            f"{row['_max_residual_log10_K']} log10 K"
        )


if __name__ == "__main__":
    main()
