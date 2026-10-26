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
import hashlib
import io
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linprog

# Keep package imports available when this file is invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
PUBLISHED_DATA = Path(__file__).resolve().parents[1] / "data-src/published-thermo"


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
KONINGS_SOURCE_DIR = Path(__file__).resolve().parents[1] / "data-src/konings"
KONINGS_BATCH_ELEMENTS = ("La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu")
CONSTRUCTED_SAHA_ION_ELEMENTS = (
    "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho",
    "Er", "Tm", "Yb", "Lu", "Y", "U", "Ru", "Rh", "Ir",
)
CONSTRUCTED_SAHA_ION_SOURCES = tuple(
    (f"{element}+(g)", element) for element in CONSTRUCTED_SAHA_ION_ELEMENTS
)
KONINGS_GAS_SPECIES = {
    "La": (("La(g)", 0), ("LaO(g)", 1)),
    "Ce": (("Ce(g)", 0), ("CeO(g)", 1), ("CeO2(g)", 2)),
    "Pr": (("Pr(g)", 0), ("PrO(g)", 1)),
    "Nd": (("Nd(g)", 0), ("NdO(g)", 1)),
    "Sm": (("Sm(g)", 0), ("SmO(g)", 1)),
    "Eu": (("Eu(g)", 0), ("EuO(g)", 1)),
    "Gd": (("Gd(g)", 0), ("GdO(g)", 1)),
    "Tb": (("Tb(g)", 0), ("TbO(g)", 1)),
    "Dy": (("Dy(g)", 0), ("DyO(g)", 1)),
    "Ho": (("Ho(g)", 0), ("HoO(g)", 1)),
    "Er": (("Er(g)", 0), ("ErO(g)", 1)),
    "Tm": (("Tm(g)", 0), ("TmO(g)", 1)),
    "Yb": (("Yb(g)", 0), ("YbO(g)", 1)),
    "Lu": (("Lu(g)", 0), ("LuO(g)", 1)),
}
# Ho(g)'s printed first and second Cp intervals meet at 1000 K. The 1000–1500
# fit still exceeds the on-node gate, so fit the requested low runtime interval
# directly; the 1500–3000 interval remains the established high fit.
KONINGS_LOW_FIT_START_K = {"Ho(g)": 1200.0}
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
    "validity_kind", "continuation_kind", "cp_extension", "construction_method",
    "table_id", "source_label", "source_liquid_marker", "supercooled_continuation",
    "source_fit_min_K", "source_fit_max_K", "source_limit_K",
    "fusion_temperature_K", "supercooled_segment_min_K",
    "supercooled_segment_max_K", "construction_tier", "glass_join_temperature_K",
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
    # Public phosphorus and sulfur gas rows from JANAF 4th ed.; their parent
    # standards are the assessed P2O5(l) row and caller-supplied S2(g).
    ("P(g)", "P-008", "P", 1, 0),
    ("P2(g)", "P-012", "P", 2, 0),
    ("P4(g)", "P-013", "P", 4, 0),
    ("PO(g)", "O-004", "P", 1, 1),
    ("PO2(g)", "O-032", "P", 1, 2),
    ("P4O6(g)", "O-087", "P", 4, 6),
    ("P4O10(g)", "O-095", "P", 4, 10),
    # Appended within the P/S block so all existing source rows retain order.
    ("PS(g)", "P-011", "P", 1, 0),
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
    ("Hf(g)", "Hf-006", "Hf", 1, 0),
    ("Li(g)", "Li-005", "Li", 1, 0),
    ("LiO(g)", "Li-011", "Li", 1, 1),
    ("Li2O(g)", "Li-017", "Li", 2, 1),
    ("Li2O2(g)", "Li-019", "Li", 2, 2),
    ("Rb(g)", "Rb-005", "Rb", 1, 0),
    ("Pb(g)", "Pb-005", "Pb", 1, 0),
    ("PbO(g)", "O-009", "Pb", 1, 1),
    ("Ga(g)", "Ga-005", "Ga", 1, 0),
    ("B(g)", "B-005", "B", 1, 0),
    ("BO(g)", "B-078", "B", 1, 1),
    ("BO2(g)", "B-079", "B", 1, 2),
    ("B2O(g)", "B-093", "B", 2, 1),
    ("B2O2(g)", "B-094", "B", 2, 2),
    ("B2O3(g)", "B-098", "B", 2, 3),
    ("Cs(g)", "Cs-005", "Cs", 1, 0),
    ("CsO(g)", "Cs-017", "Cs", 1, 1),
    ("Cs2O(g)", "Cs-021", "Cs", 2, 1),
    ("Cu(g)", "Cu-005", "Cu", 1, 0),
    ("CuO(g)", "Cu-016", "Cu", 1, 1),
    ("Cu2(g)", "Cu-018", "Cu", 2, 0),
    ("Zn(g)", "Zn-005", "Zn", 1, 0),
)
# Mo/W JANAF tables are vendored as normalized YAML records. Keep them on the
# same fit path and two-interval domain as the other optional trace channels.
TRACE_JANAF_YAML_GAS_SOURCES = (
    ("H2(g)", "H-050", "H", 2, 0),
    ("H(g)", "H-001", "H", 1, 0),
    ("OH(g)", "H-038", "H", 1, 1),
    ("H2O(g)", "H-064", "H", 2, 1),
    ("CO(g)", "C-093", "C", 1, 1),
    ("CO2(g)", "C-095", "C", 1, 2),
    ("N2(g)", "N-023", "N", 2, 0),
    ("N(g)", "N-002", "N", 1, 0),
    ("NO(g)", "N-005", "N", 1, 1),
    ("NO2(g)", "N-007", "N", 1, 2),
    ("HCl(g)", "Cl-026", "Cl", 1, 1),
    ("HF(g)", "F-005", "F", 1, 1),
    ("HBr(g)", "Br-010", "Br", 1, 1),
    ("HI(g)", "H-005", "H", 1, 1),
    ("Mo(g)", "Mo-005", "Mo", 1, 0),
    ("MoO(g)", "Mo-008", "Mo", 1, 1),
    ("MoO2(g)", "Mo-010", "Mo", 1, 2),
    ("MoO3(g)", "Mo-017", "Mo", 1, 3),
    ("W(g)", "W-005", "W", 1, 0),
    ("WO(g)", "O-027", "W", 1, 1),
    ("WO2(g)", "O-048", "W", 1, 2),
    ("WO3(g)", "O-068", "W", 1, 3),
    ("(WO3)2(g)", "O-088", "W", 2, 6),
    ("(WO3)3(g)", "O-093", "W", 3, 9),
    ("(WO3)4(g)", "O-096", "W", 4, 12),
    ("W3O8(g)", "O-092", "W", 3, 8),
    ("Ba(g)", "Ba-005", "Ba", 1, 0),
    ("BaO(g)", "Ba-037", "Ba", 1, 1),
    ("Sr(g)", "Sr-006", "Sr", 1, 0),
    ("SrO(g)", "O-016", "Sr", 1, 1),
    ("Zr(g)", "Zr-006", "Zr", 1, 0),
    ("ZrO(g)", "O-028", "Zr", 1, 1),
    ("ZrO2(g)", "O-055", "Zr", 1, 2),
    ("Ta(g)", "Ta-005", "Ta", 1, 0),
    ("TaO(g)", "O-017", "Ta", 1, 1),
    ("TaO2(g)", "O-041", "Ta", 1, 2),
    ("Br2(g)", "Br-040", "Br", 2, 0),
    ("Br(g)", "Br-001", "Br", 1, 0),
    ("I2(g)", "I-027", "I", 2, 0),
    ("I(g)", "I-001", "I", 1, 0),
    ("NaBr(g)", "Br-030", "Na", 1, 0),
    ("(NaBr)2(g)", "Br-064", "Na", 2, 0),
    ("KBr(g)", "Br-018", "K", 1, 0),
    ("(KBr)2(g)", "Br-055", "K", 2, 0),
    ("KI(g)", "I-007", "K", 1, 0),
    ("(KI)2(g)", "I-028", "K", 2, 0),
    # Alkaline-earth and supported alkali halides appended for the oxide + X2
    # reaction family. JANAF gas-phase rows take priority; NASA cards below
    # supply only species without a matching JANAF gas table.
    ("BaBr(g)", "Ba-007", "Ba", 1, 0),
    ("BaBr2(g)", "Ba-011", "Ba", 1, 0),
    ("BaCl(g)", "Ba-012", "Ba", 1, 0),
    ("BaCl2(g)", "Ba-016", "Ba", 1, 0),
    ("BaF(g)", "Ba-017", "Ba", 1, 0),
    ("BaF2(g)", "Ba-022", "Ba", 1, 0),
    ("BaI(g)", "Ba-029", "Ba", 1, 0),
    ("BaI2(g)", "Ba-033", "Ba", 1, 0),
    ("CaBr(g)", "Br-004", "Ca", 1, 0),
    ("CaBr2(g)", "Br-044", "Ca", 1, 0),
    ("CaCl(g)", "Ca-008", "Ca", 1, 0),
    ("CaCl2(g)", "Ca-012", "Ca", 1, 0),
    ("CaF(g)", "Ca-013", "Ca", 1, 0),
    ("CaF2(g)", "Ca-017", "Ca", 1, 0),
    ("CaI(g)", "Ca-022", "Ca", 1, 0),
    ("CaI2(g)", "Ca-026", "Ca", 1, 0),
    ("MgF(g)", "F-026", "Mg", 1, 0),
    ("MgF2(g)", "F-075", "Mg", 1, 0),
    ("(MgF2)2(g)", "F-129", "Mg", 2, 0),
    ("MgCl(g)", "Cl-049", "Mg", 1, 0),
    ("MgCl2(g)", "Cl-096", "Mg", 1, 0),
    ("(MgCl2)2(g)", "Cl-150", "Mg", 2, 0),
    ("MgBr(g)", "Br-023", "Mg", 1, 0),
    ("MgBr2(g)", "Br-060", "Mg", 1, 0),
    ("(MgBr2)2(g)", "Br-092", "Mg", 2, 0),
    ("MgI(g)", "I-012", "Mg", 1, 0),
    ("MgI2(g)", "I-033", "Mg", 1, 0),
    ("SrBr(g)", "Br-034", "Sr", 1, 0),
    ("SrBr2(g)", "Br-073", "Sr", 1, 0),
    ("SrCl(g)", "Cl-069", "Sr", 1, 0),
    ("SrCl2(g)", "Cl-122", "Sr", 1, 0),
    ("SrF(g)", "F-049", "Sr", 1, 0),
    ("SrF2(g)", "F-105", "Sr", 1, 0),
    ("SrI(g)", "I-020", "Sr", 1, 0),
    ("SrI2(g)", "I-044", "Sr", 1, 0),
    ("LiF(g)", "F-024", "Li", 1, 0),
    ("(LiF)2(g)", "F-071", "Li", 2, 0),
    ("LiCl(g)", "Cl-044", "Li", 1, 0),
    ("(LiCl)2(g)", "Cl-092", "Li", 2, 0),
    ("LiBr(g)", "Br-022", "Li", 1, 0),
    ("(LiBr)2(g)", "Br-056", "Li", 2, 0),
    ("LiI(g)", "I-011", "Li", 1, 0),
    ("(LiI)2(g)", "I-029", "Li", 2, 0),
    ("CsF(g)", "Cs-011", "Cs", 1, 0),
    ("(CsF)2(g)", "Cs-019", "Cs", 2, 0),
    ("CsCl(g)", "Cl-008", "Cs", 1, 0),
    ("(CsCl)2(g)", "Cl-078", "Cs", 2, 0),
    ("Cl2(g)", "Cl-073", "Cl", 2, 0),
    ("Cl(g)", "Cl-001", "Cl", 1, 0),
    ("NaCl(g)", "Cl-056", "Na", 1, 0),
    ("(NaCl)2(g)", "Cl-098", "Na", 2, 0),
    ("KCl(g)", "Cl-039", "K", 1, 0),
    ("(KCl)2(g)", "Cl-091", "K", 2, 0),
    ("F2(g)", "F-054", "F", 2, 0),
    ("F(g)", "F-001", "F", 1, 0),
    ("NaF(g)", "F-036", "Na", 1, 0),
    ("(NaF)2(g)", "F-082", "Na", 2, 0),
    ("KF(g)", "F-020", "K", 1, 0),
    ("(KF)2(g)", "F-069", "K", 2, 0),
)
HALOGEN_JANAF_TABLES = frozenset(
    {
        "Br-040", "Br-001", "I-027", "I-001", "Br-030", "Br-064",
        "Br-018", "Br-055", "I-007", "I-028", "Cl-073", "Cl-001",
        "Cl-056", "Cl-098", "Cl-039", "Cl-091", "F-054", "F-001",
        "F-036", "F-082", "F-020", "F-069",
        "Cl-026", "F-005", "Br-010", "H-005",
        "Ba-007", "Ba-011", "Ba-012", "Ba-016", "Ba-017", "Ba-022",
        "Ba-029", "Ba-033", "Br-004", "Br-044", "Ca-022", "Ca-026",
        "Br-023", "Br-060", "Br-092", "I-012", "I-033", "Br-034",
        "Br-073", "Cl-069", "Cl-122", "F-049", "F-105", "I-020",
        "I-044", "F-024", "F-071", "Cl-044", "Cl-092", "Br-022",
        "Br-056", "I-011", "I-029", "Cs-011", "Cs-019", "Cl-008",
    "Cl-078", "Ca-013", "Ca-017", "Ca-008", "Ca-012", "F-026",
        "F-075", "F-129", "Cl-049", "Cl-096", "Cl-150",
    }
)
TRACE_NASA_GAS_SOURCES = (
    ("NiO(g)", "NG-0926", "NiO", "Ni", 1, 1),
    ("Ag(g)", "NG-0074", "Ag", "Ag", 1, 0),
    ("RbO(g)", "NG-1329", "RbO", "Rb", 1, 1),
    ("Rb2O(g)", "NG-1352", "Rb2O", "Rb", 2, 1),
    ("Be(g)", "NG-0185", "Be", "Be", 1, 0),
    ("BeO(g)", "NG-0200", "BeO", "Be", 1, 1),
    ("Be2O(g)", "NG-0208", "Be2O", "Be", 2, 1),
    ("(BeO)2(g)", "NG-0210", "Be2O2", "Be", 2, 2),
    ("(BeO)3(g)", "NG-0211", "Be3O3", "Be", 3, 3),
    ("(BeO)4(g)", "NG-0212", "Be4O4", "Be", 4, 4),
    ("Be2(g)", "NG-0205", "Be2", "Be", 2, 0),
    ("PbO2(g)", "NG-1276", "PbO2", "Pb", 1, 2),
    ("GaO(g)", "NG-5066", "GaO", "Ga", 1, 1),
    ("Ga2O(g)", "NG-5178", "Ga2O", "Ga", 2, 1),
    ("Ge(g)", "NG-5186", "Ge", "Ge", 1, 0),
    ("GeO(g)", "NG-5331", "GeO", "Ge", 1, 1),
    ("GeO2(g)", "NG-5339", "GeO2", "Ge", 1, 2),
    ("In(g)", "NG-6005", "In", "In", 1, 0),
    ("InO(g)", "NG-6131", "InO", "In", 1, 1),
    ("In2O(g)", "NG-6243", "In2O", "In", 2, 1),
    ("Sn(g)", "NG-1845", "Sn", "Sn", 1, 0),
    ("SnO(g)", "NG-1847", "SnO", "Sn", 1, 1),
    ("SnO2(g)", "NG-1849", "SnO2", "Sn", 1, 2),
    ("Cd(g)", "NG-4060", "Cd", "Cd", 1, 0),
    ("(MoO3)2(g)", "NG-0831", "Mo2O6", "Mo", 2, 6),
    ("(MoO3)3(g)", "NG-0832", "Mo3O9", "Mo", 3, 9),
    ("(MoO3)4(g)", "NG-0833", "Mo4O12", "Mo", 4, 12),
    ("(MoO3)5(g)", "NG-0834", "Mo5O15", "Mo", 5, 15),
    ("(WO3)5(g)", "NG-1258", "W5O15", "W", 5, 15),
    # Sc and U have no JANAF tables in the surveyed corpus. NASA Glenn's
    # Gurvich-adopted cards supply both gas and parent-liquid functions.
    ("Sc(g)", "NG-1087", "Sc", "Sc", 1, 0),
    ("ScO(g)", "NG-1090", "ScO", "Sc", 1, 1),
    ("ScO2(g)", "NG-1092", "ScO2", "Sc", 1, 2),
    ("U(g)", "NG-1206", "U", "U", 1, 0),
    ("UO(g)", "NG-1224", "UO", "U", 1, 1),
    ("UO2(g)", "NG-1230", "UO2", "U", 1, 2),
    ("UO3(g)", "NG-1235", "UO3", "U", 1, 3),
    ("NaI(g)", "NG-0893", "NaI", "Na", 1, 0),
    ("(NaI)2(g)", "NG-0904", "Na2I2", "Na", 2, 0),
    ("RbF(g)", "NG-1020", "RbF", "Rb", 1, 0),
    ("(RbF)2(g)", "NG-1032", "Rb2F2", "Rb", 2, 0),
    ("RbCl(g)", "NG-1019", "RbCl", "Rb", 1, 0),
    ("(RbCl)2(g)", "NG-1031", "Rb2Cl2", "Rb", 2, 0),
    ("RbBr(g)", "NG-1018", "RbBr", "Rb", 1, 0),
    ("(RbBr)2(g)", "NG-1030", "Rb2Br2", "Rb", 2, 0),
    ("RbI(g)", "NG-1022", "RbI", "Rb", 1, 0),
    ("(RbI)2(g)", "NG-1033", "Rb2I2", "Rb", 2, 0),
    ("CsBr(g)", "NG-0511", "CsBr", "Cs", 1, 0),
    ("(CsBr)2(g)", "NG-0524", "Cs2Br2", "Cs", 2, 0),
    ("CsI(g)", "NG-0515", "CsI", "Cs", 1, 0),
    ("(CsI)2(g)", "NG-0528", "Cs2I2", "Cs", 2, 0),
)
HALOGEN_NASA_TABLES = frozenset({
    "NG-0893", "NG-0904", "NG-1020",
    "NG-1032", "NG-1019", "NG-1031", "NG-1018", "NG-1030", "NG-1022",
    "NG-1033", "NG-0511", "NG-0524", "NG-0515", "NG-0528",
})
TRACE_PUBLISHED_GAS_SOURCES = (
    ("Ru(g)", "P672-Ru-g", "Ru", "Ru", 1, 0, 3000.0),
    ("Rh(g)", "P672-Rh-g", "Rh", "Rh", 1, 0, 2000.0),
    ("Re(g)", "P677-Re-g", "Re", "Re", 1, 0, 2000.0),
    ("Y(g)", "P672-Y-g", "Y", "Y", 1, 0, 2500.0),
    ("YO(g)", "PM83-YO", "YO", "Y", 1, 1, 1500.0),
    ("ZnO(g)", "IVTAN-264", "ZnO", "Zn", 1, 1, 3000.0),
    ("Bi(g)", "P672-T80", "Bi", "Bi", 1, 0, 1800.0),
    ("Bi2(g)", "P672-T81", "Bi2", "Bi", 2, 0, 1800.0),
    ("BiO(g)", "P696-T115", "BiO", "Bi", 1, 1, 1200.0),
    ("CdO(g)", "P696-T191", "CdO", "Cd", 1, 1, 2000.0),
    ("Te(g)", "P672-Te-g", "Te", "Te", 1, 0, 2000.0),
    ("Te2(g)", "P672-Te2-g", "Te2", "Te", 2, 0, 2000.0),
    ("TeO2(g)", "P672-TeO2-g", "TeO2", "Te", 1, 2, 1500.0),
    ("Se(g)", "P672-Se-g", "Se", "Se", 1, 0, 2000.0),
    ("Se2(g)", "P672-Se2-g", "Se2", "Se", 2, 0, 2000.0),
    ("SeO(g)", "P672-SeO-g", "SeO", "Se", 1, 1, 2000.0),
    ("SeO2(g)", "P672-SeO2-g", "SeO2", "Se", 1, 2, 1500.0),
    ("As(g)", "P672-As-g", "As", "As", 1, 0, 1200.0),
    ("As2(g)", "P672-As2-g", "As2", "As", 2, 0, 1200.0),
    ("As4(g)", "P672-As4-g", "As4", "As", 4, 0, 1200.0),
    ("As4O6(g)", "P672-As4O6-g", "As4O6", "As", 4, 6, 1000.0),
    ("Sb(g)", "P672-Sb-g", "Sb", "Sb", 1, 0, 2000.0),
    ("Sb2(g)", "P672-Sb2-g", "Sb2", "Sb", 2, 0, 2000.0),
    ("Sb4(g)", "P672-Sb4-g", "Sb4", "Sb", 4, 0, 2000.0),
    ("Sb4O6(g)", "P672-Sb4O6-g", "Sb4O6", "Sb", 4, 6, 1000.0),
    ("Th(g)", "P672-Th-g", "Th", "Th", 1, 0, 2500.0),
    ("ThO(g)", "P672-ThO-g", "ThO", "Th", 1, 1, 2500.0),
    ("ThO2(g)", "P672-ThO2-g", "ThO2", "Th", 1, 2, 2500.0),
    ("Tl(g)", "P672-Tl-g", "Tl", "Tl", 1, 0, 2000.0),
    ("Tl2O(g)", "P672-Tl2O-g", "Tl2O", "Tl", 2, 1, 1000.0),
    ("MnO(g)", "TSIV-1436", "MnO", "Mn", 1, 1, 6000.0),
    ("CoO(g)", "TSIV-1335", "CoO", "Co", 1, 1, 3000.0),
    ("TeO(g)", "NUREG-4308-TeO", "TeO", "Te", 1, 1, 3000.0),
    ("SbO(g)", "Kelley-584-SbO", "SbO", "Sb", 1, 1, 2000.0),
)
TRACE_GAS_SPECIES = frozenset(
    {source[0].removesuffix("(g)") for source in TRACE_JANAF_GAS_SOURCES}
    | {source[0].removesuffix("(g)") for source in TRACE_JANAF_YAML_GAS_SOURCES}
    | {source[0].removesuffix("(g)") for source in TRACE_NASA_GAS_SOURCES}
    | {source[0].removesuffix("(g)") for source in TRACE_PUBLISHED_GAS_SOURCES}
)
HALOGEN_GAS_SPECIES = frozenset(
    source[0].removesuffix("(g)")
    for source in (
        *TRACE_JANAF_GAS_SOURCES,
        *TRACE_JANAF_YAML_GAS_SOURCES,
        *TRACE_NASA_GAS_SOURCES,
    )
    if source[1] in HALOGEN_JANAF_TABLES | HALOGEN_NASA_TABLES
)
TRACE_NASA_COMPARISON_SOURCES = (("NG-10887", "Zn"), ("NG-10898", "Zn+"), ("NG-0074", "Ag"))

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
    ("Hf+(g)", "Hf-007", "Hf", 1, 0),
    ("Hf-(g)", "Hf-008", "Hf", 1, 0),
    ("Li+(g)", "Li-006", "Li", 1, 0),
    ("Rb+(g)", "Rb-006", "Rb", 1, 0),
    ("Pb+(g)", "Pb-006", "Pb", 1, 0),
    ("Ga+(g)", "Ga-006", "Ga", 1, 0),
    ("B+(g)", "B-006", "B", 1, 0),
    ("Ga-(g)", "Ga-007", "Ga", 1, 0),
    ("B-(g)", "B-007", "B", 1, 0),
    ("BO2-(g)", "B-080", "B", 1, 2),
    ("Cs+(g)", "Cs-006", "Cs", 1, 0),
    ("Cs-(g)", "Cs-007", "Cs", 1, 0),
    ("Cu+(g)", "Cu-006", "Cu", 1, 0),
    ("Cu-(g)", "Cu-007", "Cu", 1, 0),
    ("Zn+(g)", "Zn-006", "Zn", 1, 0),
    ("Zn-(g)", "Zn-007", "Zn", 1, 0),
    ("Hg+(g)", "Hg-004", "Hg", 1, 0),
)
DECLINED_ION_GAS_SPECIES = frozenset({"Cu+(g)"})
TRACE_ION_GAS_NASA_SOURCES = (
    ("Ag+(g)", "NG-0085", "Ag", "Ag", 1, 0),
    ("Ag-(g)", "NG-0086", "Ag", "Ag", 1, 0),
    ("BO-(g)", "NG-1074", "BO", "B", 1, 1),
    ("Ge+(g)", "NG-5197", "Ge", "Ge", 1, 0),
    ("Cd+(g)", "NG-4071", "Cd", "Cd", 1, 0),
    ("Sc+(g)", "NG-1088", "ScE-1", "Sc", 1, 0),
)
TRACE_ION_GAS_BURCAT_SOURCES = (
    ("Th+(g)", "Burcat-Th-plus", "Th+", "Th", 1, 0),
    ("Au+(g)", "Burcat-Au-plus", "Au+", "Au", 1, 0),
    ("Au-(g)", "Burcat-Au-minus", "Au-", "Au", 1, 0),
    ("Pd+(g)", "Burcat-Pd-plus", "Pd+", "Pd", 1, 0),
    ("Pt+(g)", "Burcat-Pt-plus", "Pt+", "Pt", 1, 0),
    ("Pt-(g)", "Burcat-Pt-minus", "Pt-", "Pt", 1, 0),
)
BURCAT_METAL_GAS_SOURCES = (
    ("Hg(g)", "burcat-hg.json", "Hg(g)", "Hg", 1, 0),
    ("Au(g)", "burcat-au.json", "Au(g)", "Au", 1, 0),
    ("Pd(g)", "burcat-pd.json", "Pd(g)", "Pd", 1, 0),
    ("Pt(g)", "burcat-pt.json", "Pt(g)", "Pt", 1, 0),
    ("Ir(g)", "burcat-ir.json", "Ir(g)", "Ir", 1, 0),
    ("PtO2(g)", "burcat-pt.json", "PtO2(g)", "Pt", 1, 2),
    ("Os(g)", "burcat-os.json", "Os(g)", "Os", 1, 0),
)
BURCAT_METAL_PARENT_SOURCES = (
    ("Hg(l)", "burcat-hg.json", "Hg(l)", "Hg", 1, 0, "l", 1200.0, 3000.0, "continuation"),
    ("Au(cr)", "burcat-au.json", "Au(cr)", "Au", 1, 0, "cr", 1200.0, 1337.0, "source"),
    ("Au(l)", "burcat-au.json", "Au(l)", "Au", 1, 0, "l", 1337.0, 3000.0, "source"),
    ("Pd(cr)", "burcat-pd.json", "Pd(cr)", "Pd", 1, 0, "cr", 1200.0, 1827.0, "source"),
    ("Pd(l)", "burcat-pd.json", "Pd(l)", "Pd", 1, 0, "l", 1827.0, 3000.0, "source"),
    ("Pt(cr)", "burcat-pt.json", "Pt(cr)", "Pt", 1, 0, "cr", 1200.0, 2042.0, "source"),
    ("Pt(l)", "burcat-pt.json", "Pt(l)", "Pt", 1, 0, "l", 2042.0, 3000.0, "source"),
    ("Ir(cr)", "burcat-ir.json", "Ir(cr)", "Ir", 1, 0, "cr", 1200.0, 2720.0, "source"),
    ("Ir(l)", "burcat-ir.json", "Ir(l)", "Ir", 1, 0, "l", 2720.0, 3000.0, "source"),
    ("Os(cr)", "burcat-os.json", "Os(cr)", "Os", 1, 0, "cr", 1200.0, 3000.0, "source"),
)
PANKRATZ_METAL_PARENT_SOURCES = (
    ("Ru(cr)", "P672-Ru-cr", "Ru", 1, 0, "cr", 1200.0, 2607.0, "source"),
    ("Ru(l)", "P672-Ru-l", "Ru", 1, 0, "l", 2607.0, 3000.0, "source"),
    ("Rh(cr)", "P672-Rh-cr", "Rh", 1, 0, "cr", 1200.0, 2237.0, "source"),
    ("Rh(l)", "P672-Rh-l", "Rh", 1, 0, "l", 2237.0, 2500.0, "source"),
    ("Rh(l)", "P672-Rh-l", "Rh", 1, 0, "l", 2500.0, 3000.0, "continuation"),
    ("Re(cr)", "P677-Re-cr", "Re", 1, 0, "cr", 298.0, 1200.0, "source-low"),
    ("Re(cr)", "P677-Re-cr", "Re", 1, 0, "cr", 1200.0, 2000.0, "source"),
    ("Re(cr)", "P677-Re-cr", "Re", 1, 0, "cr", 2000.0, 3000.0, "continuation"),
)
TRACE_ION_PUBLISHED_SOURCES = (("Bi+(g)", "NBS-T22-ASD-BiII", "Bi", 1, 0),)
TRACE_ION_NASA_SOURCES = (
    ("Sn+(g)", "NG-1846", "Sn+", "Sn", 1, 0),
    ("Cs2O+(g)", "NG-1843", "Cs2O+", "Cs", 2, 1),
)
ION_GAS_YAML_SOURCES = (("Ca+(g)", "Ca-007", "Ca", 1, 0),)
TRACE_ION_GAS_YAML_SOURCES = (
    ("Mn+(g)", "Mn-006", "Mn", 1, 0),
    ("Ni+(g)", "Ni-006", "Ni", 1, 0),
    ("Ni-(g)", "Ni-007", "Ni", 1, 0),
    ("Co+(g)", "Co-006", "Co", 1, 0),
    ("Co-(g)", "Co-007", "Co", 1, 0),
    ("Mo+(g)", "Mo-006", "Mo", 1, 0),
    ("Mo-(g)", "Mo-007", "Mo", 1, 0),
    ("W+(g)", "W-006", "W", 1, 0),
    ("W-(g)", "W-007", "W", 1, 0),
    ("Ba+(g)", "Ba-006", "Ba", 1, 0),
    ("Sr+(g)", "Sr-007", "Sr", 1, 0),
    ("Zr+(g)", "Zr-007", "Zr", 1, 0),
    ("Zr-(g)", "Zr-008", "Zr", 1, 0),
)

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
        "Si2",
        "Si3",
        "Al2",
        "Co",
        "Cr",
        "CrO",
        "CrO2",
        "CrO3",
        "Mn",
        "Nb",
        "NbO",
        "NbO2",
        "Ni",
        "Ti",
        "TiO",
        "TiO2",
        "V",
        "VO",
        "VO2",
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
        "PS",
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

# Opt-in rough extension inputs. These rows are generated separately from the
# packaged tables, and are never included in the default build. The CEA cards
# supply Cp(T); each candidate interval is integrated from the existing row's
# H and S at 3000 K so the standard state remains continuous at the seam.
HIGH_TEMP_GAS_CARDS = (
    ("SiO(g)", "SiO", "Si", 1, 1), ("O(g)", "O", "", 0, 1),
    ("O2(g)", "O2", "", 0, 2), ("Mg(g)", "Mg", "Mg", 1, 0),
    ("Fe(g)", "Fe", "Fe", 1, 0), ("Na(g)", "Na", "Na", 1, 0),
    ("K(g)", "K", "K", 1, 0), ("Ca(g)", "Ca", "Ca", 1, 0),
    ("Al(g)", "AL", "Al", 1, 0), ("AlO(g)", "AlO", "Al", 1, 1),
    ("Ti(g)", "Ti", "Ti", 1, 0), ("TiO(g)", "TiO", "Ti", 1, 1),
    ("SiO2(g)", "SiO2", "Si", 1, 2), ("MgO(g)", "MgO", "Mg", 1, 1),
    ("FeO(g)", "FeO", "Fe", 1, 1), ("CaO(g)", "CaO", "Ca", 1, 1),
    ("AlO2(g)", "ALO2", "Al", 1, 2), ("TiO2(g)", "TiO2", "Ti", 1, 2),
    ("NaO(g)", "NaO", "Na", 1, 1), ("KO(g)", "KO", "K", 1, 1),
    ("Na2O(g)", "Na2O", "Na", 2, 1), ("K2O(g)", "K2O", "K", 2, 1),
    ("Al2O(g)", "AL2O", "Al", 2, 1), ("Al2O2(g)", "AL2O2", "Al", 2, 2),
    ("Al2(g)", "AL2", "Al", 2, 0), ("Na2(g)", "Na2", "Na", 2, 0),
    ("K2(g)", "K2", "K", 2, 0), ("Si(g)", "Si", "Si", 1, 0),
    ("Si2(g)", "Si2", "Si", 2, 0), ("Si3(g)", "Si3", "Si", 3, 0),
)
HIGH_TEMP_LIQUID_CARDS = (
    ("SiO2(l)", "SiO2(L)", "Si", 1, 2, 1996.0),
    ("Al2O3(l)", "AL2O3(L)", "Al", 2, 3, 2327.0),
    ("CaO(l)", "CaO(L)", "Ca", 1, 1, 3172.0),
    ("MgO(l)", "MgO(L)", "Mg", 1, 1, 3100.0),
    ("TiO2(l)", "TiO2(L)", "Ti", 1, 2, 2185.0),
    ("Na2O(l)", "Na2O(L)", "Na", 2, 1, 1405.0),
    ("K2O(l)", "K2O(L)", "K", 2, 1, 1013.0),
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
    "validity_kind", "continuation_kind", "cp_extension", "construction_method",
    "table_id", "source_label", "source_liquid_marker", "supercooled_continuation",
    "source_fit_min_K", "source_fit_max_K", "source_limit_K",
    "fusion_temperature_K", "supercooled_segment_min_K",
    "supercooled_segment_max_K", "construction_tier", "glass_join_temperature_K",
)

# Reference metal rows are generated for the independent redox data component.
# They do not enter condensate.csv or the gas evaluator's oxide-parent set.
REDOX_METAL_STANDARD_SOURCES = (
    ("Fe(cr)", "Fe-007", "Fe", 1, 0, "cr", 500.0, 1800.0, 1700.0, 1809.0),
    ("Fe(l)", "Fe-006", "Fe", 1, 0, "l", 1700.0, 3000.0, 1700.0, 3000.0),
    ("Si(l)", "Si-003", "Si", 1, 0, "l", 1700.0, 3000.0, 1700.0, 3000.0),
    ("Cr(l)", "Cr-003", "Cr", 1, 0, "l", 1700.0, 3000.0, 1700.0, 3000.0),
    ("Ni(l)", "Ni-003", "Ni", 1, 0, "l", 1700.0, 3000.0, 1700.0, 3000.0),
    ("Co(l)", "Co-003", "Co", 1, 0, "l", 1700.0, 3000.0, 1700.0, 3000.0),
    ("Mn(l)", "Mn-003", "Mn", 1, 0, "l", 1600.0, 2200.0, 1600.0, 2200.0),
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

# B2O3(l) is fitted from NIST's printed liquid branch, beginning at the first
# complete post-marker node. The other three requested liquids have no JANAF
# table and use the published NASA CEA liquid cards.
TRACE_CONDENSATE_SOURCES = (("B2O3(l)", "B-096", "B", 2, 3),)
TRACE_CONDENSED_SOURCES = (("C(cr)", "C-002", "C", 1, 0),)
TRACE_NASA_CONDENSATE_SOURCES = (
    ("BeO(l)", "NG-BeO-L-1352", "BeO", "Be", 1, 1, 2851.0,
     ((1200.0, 1500.0), (1500.0, 2000.0), (2000.0, 2500.0), (2500.0, 2851.0)),
     ((2851.0, 3000.0),)),
    ("Ga2O3(l)", "NG-12403", "Ga2O3", "Ga", 2, 3, 2080.0, ((1200.0, 2080.0),), ((2080.0, 3000.0),)),
    (
        "GeO2(l)", "NG-12434", "GeO2", "Ge", 1, 2, 1388.0,
        ((1200.0, 1388.0),), ((1388.0, 2400.0), (2400.0, 3000.0)),
    ),
    ("In2O3(l)", "NG-12662", "In2O3", "In", 2, 3, 2186.0, ((1200.0, 2186.0),), ((2186.0, 3000.0),)),
    # NASA liquid cards use their crystal formation enthalpy as the shared
    # 298 K anchor. The liquid H/S polynomial is fitted directly above its
    # boundary; the existing constant-Cp continuation extends it below.
    ("Sc2O3(l)", "NG-1857", "Sc2O3", "Sc", 2, 3, 2762.0,
     ((1200.0, 1500.0), (1500.0, 2000.0), (2000.0, 2500.0), (2500.0, 2762.0)),
     ((2762.0, 3000.0),)),
    # The UO2 liquid card begins above the runtime ceiling. The empty direct
    # interval list keeps the entire 1200-3000 K row on the labelled
    # supercooled continuation from the card's 3123 K liquid anchor.
    ("UO2(l)", "NG-1977", "UO2", "U", 1, 2, 3123.0,
     ((1200.0, 1500.0), (1500.0, 2000.0), (2000.0, 2500.0), (2500.0, 3000.0)), ()),
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
# T0 is the first complete liquid node; the last segment meets the high row.
SUPERCOOLED_LIQUID_SOURCES = (
    (
        "TiO2(l)", "O-044", "Ti", 1, 2, 1500.0,
        ((1200.0, 1500.0),),
    ),
    (
        "Cr2O3(l)", "Cr-015", "Cr", 2, 3, 1900.0,
        ((1200.0, 1500.0), (1500.0, 1900.0)),
    ),
    (
        "V2O3(l)", "O-063", "V", 2, 3, 1700.0,
        ((1200.0, 1700.0),),
    ),
    (
        "SiO2(l)", "O-038", "Si", 1, 2, 1800.0,
        ((1200.0, 1500.0), (1500.0, 1800.0), (1000.0, 1200.0)),
    ),
    (
        "Al2O3(l)", "Al-100", "Al", 2, 3, 2500.0,
        ((1200.0, 1500.0), (1500.0, 2000.0), (2000.0, 2500.0), (1000.0, 1200.0)),
    ),
    (
        "MgO(l)", "Mg-009", "Mg", 1, 1, 2200.0,
        ((1200.0, 1500.0), (1500.0, 1800.0), (1800.0, 2200.0), (1000.0, 1200.0)),
    ),
    (
        "CaO(l)", "Ca-028", "Ca", 1, 1, 2200.0,
        ((1200.0, 1500.0), (1500.0, 1800.0), (1800.0, 2200.0), (1000.0, 1200.0)),
    ),
)
NASA_SUPERCOOLED_LIQUID_SOURCES = (
    ("SnO(l)", "NG-1848", "Sn", 1, 1, 1250.0, 1250.0),
)
ALL_SUPERCOOLED_LIQUID_SPECIES = {
    source[0] for source in SUPERCOOLED_LIQUID_SOURCES
} | {source[0] for source in NASA_SUPERCOOLED_LIQUID_SOURCES} | {
    source[0] for source in TRACE_NASA_CONDENSATE_SOURCES
}
SUPERCOOLED_LIQUID_REF_SUFFIX = "-SC-CP"
# These default rows now come only from their generated JANAF fits. Remove every
# prior source interval, including disjoint LAM ranges above the new fit domain.
REPLACED_DEFAULT_CONDENSATE_SPECIES = {
    "SiO2(l)", "Al2O3(l)", "MgO(l)", "CaO(l)",
    "K2O(l)",
    "ZnO(l)", "CdO(l)", "Bi2O3(l)",
    "TeO2(l)",
}
# Shipped packs name these liquids as a standard-state reference. A coefficient
# change would force a new pack version, so their packaged rows stay byte for
# byte and any discontinuous join is reported open.
# K2O(l): FC87 new_reference of every D066 pack.
# SiO2(l), Na2O(l): the liquids the default v4 packs were re-referenced to.
# FeO(l): D&G/Masson standard of the FeO-liquid candidate and the v3 re-anchor.
# Al2O3(l), MnO(l): copied into the shipped redox packs.
PACK_REFERENCE_LIQUIDS = frozenset({
    "K2O(l)", "SiO2(l)", "Na2O(l)", "FeO(l)", "Al2O3(l)", "MnO(l)",
})
# FC87 Table 1 K2O(l), the reference every shipped D066 pack is translated
# against, is 2*G(K, g) + G(O, g) plus a closed-form term. Refitting these
# rows changes that basis, so the packaged lines stay byte for byte.
PACK_REFERENCE_GAS = frozenset({"K(g)", "O(g)"})

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

# Sources whose tabulated rows stop short of FIT_T_MAX. Cr(g), Cr-005: Cr boils
# at 2952 K, where JANAF switches the element reference to the gas and prints
# the 3000 K formation columns as "0. 0. 0.". The thermal cells remain intact
# (Cp 30.788 J/(mol K), S 225.796 J/(mol K), H-H298 64.417 kJ/mol), but fitting
# them would change the existing Cr coefficient row and outputs above 1500 K.
# Keep the fit capped at its last complete row, 2900 K, to preserve bit identity.
_FIT_T_MAX_BY_TABLE = {"Cr-005": 2900.0}

# Cr-005's intact 3000 K thermal cells validate the unchanged fit's endpoint,
# although its parse-ambiguous formation cells are excluded from fitting.
_DECLARED_T_MAX_BY_TABLE = {"Cr-005": 3000.0}


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


def _nasa_liquid_record(
    nasa_source_dir: Path, species_name: str, table_id: str
) -> dict[str, Any]:
    return _load_record(nasa_source_dir / f"{table_id}.json")


def _value(row: dict[str, Any], field: str) -> float | None:
    value = row[field]
    if not isinstance(value, dict):
        raise ValueError(f"source field {field!r} is not a value record")
    return value.get("value")


def _recover_thermal_transition_rows(
    table: dict[str, Any],
) -> list[dict[str, float]]:
    """Recover thermal cells when only a JANAF formation column is malformed."""
    recovered = []
    for ambiguity in table.get("parse_ambiguities", []):
        cells = str(ambiguity.get("raw_line", "")).strip().split("\t")
        if (
            len(cells) < 6
            or "<-->" in cells[-1]
            or cells[5].strip() == "TRANSITION"
        ):
            continue
        try:
            temperature, cp, entropy, _, enthalpy_increment = map(
                float, cells[:5]
            )
        except ValueError:
            continue
        malformed_formation = False
        for cell in cells[5:7]:
            try:
                float(cell)
            except ValueError:
                malformed_formation = True
                break
        if not malformed_formation:
            continue
        recovered.append(
            {
                "temperature": temperature,
                "heat_capacity": cp,
                "entropy": entropy,
                "enthalpy_increment": enthalpy_increment,
            }
        )
    return recovered


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
        rows.extend(
            row
            for row in _recover_thermal_transition_rows(table)
            if row["temperature"] == 1500.0
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
    formula = {
        "O6P4": "P4O6",
        "O10P4": "P4O10",
        "O1S2": "SSO",
        "O6W2": "(WO3)2",
        "O9W3": "(WO3)3",
        "O12W4": "(WO3)4",
        "O8W3": "W3O8",
        "Br2Na2": "(NaBr)2",
        "Br2K2": "(KBr)2",
        "I2K2": "(KI)2",
        "Cl2Na2": "(NaCl)2",
        "Cl2K2": "(KCl)2",
        "F2Na2": "(NaF)2",
        "F2K2": "(KF)2",
        "Br4Mg2": "(MgBr2)2",
        "F4Mg2": "(MgF2)2",
        "Cl4Mg2": "(MgCl2)2",
        "F2Li2": "(LiF)2",
        "Cl2Li2": "(LiCl)2",
        "Br2Li2": "(LiBr)2",
        "I2Li2": "(LiI)2",
        "F2Cs2": "(CsF)2",
        "Cl2Cs2": "(CsCl)2",
        "Cs2F2": "(CsF)2",
    }.get(
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
    source_phase = (
        "g" if formula in {"O2", "Cl2", "F2", "H2", "N2"} and phase == "ref"
        else "cr" if formula == "C" and phase == "ref"
        else phase
    )
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


def _burcat7_properties(coefficients: list[float], temperature: float) -> dict[str, float]:
    """Evaluate one published Burcat seven-coefficient polynomial interval."""
    a1, a2, a3, a4, a5, a6, a7 = map(float, coefficients)
    cp_R = a1 + a2 * temperature + a3 * temperature**2 + a4 * temperature**3 + a5 * temperature**4
    h_over_R = (
        a1 * temperature + a2 * temperature**2 / 2.0
        + a3 * temperature**3 / 3.0 + a4 * temperature**4 / 4.0
        + a5 * temperature**5 / 5.0 + a6
    )
    s_over_R = (
        a1 * math.log(temperature) + a2 * temperature
        + a3 * temperature**2 / 2.0 + a4 * temperature**3 / 3.0
        + a5 * temperature**4 / 4.0 + a7
    )
    return {"cp_R": cp_R, "h_J_mol": R_J_MOL_K * h_over_R, "s_J_molK": R_J_MOL_K * s_over_R}


def _burcat_source_record(filename: str, species_name: str) -> dict[str, Any]:
    record = json.loads((PUBLISHED_DATA / filename).read_text(encoding="utf-8"))
    try:
        return record["records"][species_name]
    except KeyError as exc:
        raise ValueError(f"{filename}: missing Burcat source record {species_name}") from exc


def _fit_burcat_metal_gas_row(
    species_name: str, filename: str, record_key: str, cation: str,
    cat_num: int, oxy_num: int, *, interval: int, fit_t_min: float, fit_t_max: float,
) -> dict[str, str]:
    source = _burcat_source_record(filename, record_key)
    coefficients_key = "high_coefficients" if fit_t_min >= 1000.0 else "low_coefficients"
    coefficients = source[coefficients_key]
    reference = float(source["delta_f_H_298_15_kJ_mol"])
    h298 = _burcat7_properties(source["low_coefficients"], NASA_STANDARD_T_K)["h_J_mol"]
    if abs(h298 - reference * 1000.0) > 5.0:
        raise ValueError(
            f"{source['record_id']}: H(298.15 K) misses its printed formation anchor"
        )
    temperatures = np.arange(fit_t_min, fit_t_max + JANAF_GRID_STEP_K / 2.0, JANAF_GRID_STEP_K)
    rows = []
    for temperature in temperatures:
        properties = _burcat7_properties(coefficients, float(temperature))
        rows.append({
            "temperature": float(temperature),
            "heat_capacity": R_J_MOL_K * properties["cp_R"],
            "entropy": properties["s_J_molK"],
            "enthalpy_increment": (properties["h_J_mol"] - h298) / 1000.0,
        })
    fit = _fit_shomate_values(rows, reference)
    number = lambda value: format(float(value), ".15g")
    return {
        "species_name": species_name, "state": "g", "T_interval": str(interval),
        "cation": cation, "cat_num": str(cat_num), "oxy_num": str(oxy_num),
        "T_min": number(fit_t_min), "T_max": number(fit_t_max),
        **{key: number(fit[key]) for key in "ABCDEFG"}, "H": "0",
        "Ref": source["record_id"],
        "_max_residual_J_per_mol": number(fit["max_residual_J_per_mol"]),
        "_max_residual_log10_K": number(fit["max_residual_log10_K"]),
        **_shomate_source_fields(fit),
    }


def _fit_burcat_metal_parent(
    species_name: str, filename: str, record_key: str, cation: str,
    cat_num: int, oxy_num: int, phase: str, fit_t_min: float, fit_t_max: float,
    source_kind: str,
) -> dict[str, str]:
    source = _burcat_source_record(filename, record_key)
    # Burcat condensed cards can carry separate low/high blocks just like gas
    # cards. A fit wholly above 1000 K must use the high block valid there.
    coefficients = source.get("high_coefficients") if fit_t_min >= 1000.0 else None
    coefficients = coefficients or source["coefficients"]
    reference = float(source.get("delta_f_H_298_15_kJ_mol", 0.0))
    temperatures = sorted({
        *(float(T) for T in np.arange(
            fit_t_min, fit_t_max + JANAF_GRID_STEP_K / 2.0, JANAF_GRID_STEP_K
        ) if T <= fit_t_max),
        fit_t_max,
    })
    source_rows = {}
    for temperature in temperatures:
        value = _burcat7_properties(coefficients, float(temperature))
        source_rows[float(temperature)] = {
            "temperature": float(temperature),
            "heat_capacity": R_J_MOL_K * value["cp_R"],
            "entropy": value["s_J_molK"],
            # Premise: the selected elemental block stores absolute G already.
            # Algebra: G=H-T*S, so Hinc(T) must be H(T), with no block-local
            # subtraction H(T)-H(298.15); that subtraction shifts G by the
            # block's extrapolated H(298.15). Units: J/mol -> kJ/mol for the
            # shared fitter. Sanity: independent absolute K remains continuous
            # at each crystal/liquid marker.
            "enthalpy_increment": value["h_J_mol"] / 1000.0,
        }
    row = _fit_condensate_values(
        species_name, source["record_id"], cation, cat_num, oxy_num,
        source_rows, reference, fit_t_min=fit_t_min, fit_t_max=fit_t_max,
        runtime_t_min=fit_t_min, runtime_t_max=fit_t_max,
        ref=(f"{source['record_id']}-METAL-CONT-{source['source_limit_K']}"
             if source_kind == "continuation" else source["record_id"]),
        fit_temperatures=temperatures,
    )
    row["state"] = "s" if phase == "cr" else "l"
    return row


def _fit_pankratz_metal_parent(
    species_name: str, table_id: str, cation: str, cat_num: int, oxy_num: int,
    phase: str, fit_t_min: float, fit_t_max: float, source_kind: str,
) -> dict[str, str]:
    """Fit one elemental Pankratz metal phase on its printed 298 K scale."""
    record = _published_table_record(PUBLISHED_DATA, table_id)
    source_rows, reference, source_limit = _published_gas_source_rows(
        PUBLISHED_DATA, table_id
    )
    if source_kind == "continuation":
        source_rows = _constant_cp_rows_through(source_rows, fit_t_max)
    temperatures = sorted(
        temperature for temperature in source_rows
        if fit_t_min <= temperature <= fit_t_max
    )
    if not temperatures or temperatures[0] != fit_t_min or temperatures[-1] != fit_t_max:
        raise ValueError(f"{table_id}: parent fit endpoints are not source nodes")
    row = _fit_condensate_values(
        species_name, table_id, cation, cat_num, oxy_num,
        source_rows, reference,
        fit_t_min=fit_t_min, fit_t_max=fit_t_max,
        runtime_t_min=fit_t_min, runtime_t_max=fit_t_max,
        ref=(
            f"{table_id}-LOW-{fit_t_max:g}"
            if source_kind == "source-low"
            else f"{table_id}-CONT-{source_limit:g}"
            if source_kind == "continuation"
            else table_id
        ),
        fit_temperatures=temperatures,
    )
    row["state"] = "s" if phase == "cr" else "l"
    return row


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
        "_source_rows": [
            {
                "temperature": float(row["temperature"]),
                "heat_capacity": float(row["heat_capacity"]),
                "entropy": float(row["entropy"]),
                "enthalpy_increment": float(row["enthalpy_increment"]),
            }
            for row in rows
        ],
        "_reference_kJ": float(reference),
    }


def _shomate_source_fields(fit: dict[str, Any]) -> dict[str, Any]:
    return {
        "_source_rows": fit["_source_rows"],
        "_reference_kJ": fit["_reference_kJ"],
    }


def _shomate_primitives(
    coefficients: Sequence[float], temperature: float
) -> tuple[float, float]:
    """Return the Shomate enthalpy and entropy primitives at one temperature.

    Premise: Cp = A + B*t + C*t² + D*t³ + E/t² with t = T/1000. Algebra:
    I_H = A*t + B*t²/2 + C*t³/3 + D*t⁴/4 - E/t and
    I_S = A*ln(t) + B*t + C*t²/2 + D*t³/3 - E/(2*t²). The runtime enthalpy
    in kJ/mol is I_H + F - H, and the entropy in J/(mol K) is I_S + G.
    """
    A, B, C, D, E = coefficients
    t = temperature / 1000.0
    enthalpy = A * t + B * t**2 / 2.0 + C * t**3 / 3.0 + D * t**4 / 4.0 - E / t
    entropy = (
        A * math.log(t)
        + B * t
        + C * t**2 / 2.0
        + D * t**3 / 3.0
        - E / (2.0 * t**2)
    )
    return enthalpy, entropy


# Temperatures at which a seam refit must stay inside the row's old residual.
_SEAM_AUDIT_TEMPERATURES_K = (1323.0, 1700.0, 2200.0, 2473.0)


def _seam_sample_temperatures(t_min: float, t_max: float) -> list[float]:
    temps = [float(t_min), float(t_max)]
    step = 25.0
    value = math.ceil(t_min / step) * step
    while value < t_max - 1e-9:
        temps.append(float(value))
        value += step
    for temperature in _SEAM_AUDIT_TEMPERATURES_K:
        if t_min - 1e-9 <= temperature <= t_max + 1e-9:
            temps.append(float(temperature))
    unique = sorted(set(round(temperature, 6) for temperature in temps))
    return unique


def _stated_gibbs_budget(row: dict[str, Any], temperature: float) -> float:
    """Largest |ΔG| this row may take at one temperature.

    The cap is the unconstrained fit residual in J/mol, tightened by the
    stated log10(K) residual when that field was recorded.
    """
    raw = row.get("_max_residual_J_per_mol")
    if raw is None:
        raise ValueError(f"{row['species_name']}: seam row has no fit residual")
    budget = float(raw)
    log_raw = row.get("_max_residual_log10_K")
    if log_raw not in (None, ""):
        budget = min(
            budget,
            float(log_raw) * R_J_MOL_K * temperature * math.log(10.0),
        )
    return max(budget, 0.0)


def _refreshed_residual_within_previous(refreshed: float, previous: float) -> bool:
    """A seam shift may use the old residual, not grow past it by more than that.

    Rows that already publish a residual above 10 J/mol keep that allowance.
    A row that started at or below 10 stays there: the existing fit gates
    reject a larger published residual.
    """
    allowance = previous + max(previous, 1e-8) + 1e-5
    if previous <= 10.0:
        allowance = min(allowance, 10.0 + 1e-8)
    return refreshed <= allowance


def _condensate_fit_node_temperatures(row: dict[str, Any]) -> list[float] | None:
    """Temperatures that set a condensate row's stored fit residual.

    The runtime interval can start above the fit. Those colder nodes are part
    of the residual the tables publish, so a seam shift has to keep them.
    """
    sources = row.get("_source_rows")
    if not isinstance(sources, dict):
        return None
    explicit = row.get("_fit_temperatures")
    if explicit:
        candidates = [float(temperature) for temperature in explicit]
    elif "_fit_t_min" in row and "_fit_t_max" in row:
        candidates = [
            float(temperature)
            for temperature in np.arange(
                float(row["_fit_t_min"]),
                float(row["_fit_t_max"]) + 50.0,
                JANAF_GRID_STEP_K,
            )
        ]
    else:
        return None
    found = []
    for temperature in candidates:
        if temperature in sources:
            found.append(temperature)
            continue
        if any(abs(float(key) - temperature) <= 1e-6 for key in sources):
            found.append(temperature)
    return found


def _seam_constraint_temperatures(segment: dict[str, Any]) -> list[float]:
    """Sample points where the seam shift must stay inside the stated residual.

    Gas source nodes are a list, and the stored residual scores every node,
    including nodes below the runtime lower bound. Condensate residuals are
    scored on the fit nodes, which can also sit below that bound.
    """
    t_min = float(segment["T_min"])
    t_max = float(segment["T_max"])
    temperatures = set(_seam_sample_temperatures(t_min, t_max))
    sources = segment.get("_source_rows")
    if isinstance(sources, dict):
        for temperature in _condensate_fit_node_temperatures(segment) or []:
            temperatures.add(float(temperature))
    elif sources:
        for node in sources:
            temperatures.add(float(node["temperature"]))
    return sorted(temperatures)


def _refresh_shomate_residual(row: dict[str, Any]) -> None:
    coefficients = [float(row[key]) for key in "ABCDEFG"]
    reference = float(row["_reference_kJ"])
    column_h = float(row["H"])
    max_j = 0.0
    max_log = 0.0
    for node in row["_source_rows"]:
        temperature = float(node["temperature"])
        primitive_h, primitive_s = _shomate_primitives(coefficients[:5], temperature)
        model_g = (primitive_h + coefficients[5] - column_h) * 1000.0 - temperature * (
            primitive_s + coefficients[6]
        )
        source_g = (
            reference + float(node["enthalpy_increment"])
        ) * 1000.0 - temperature * float(node["entropy"])
        error = abs(model_g - source_g)
        max_j = max(max_j, error)
        max_log = max(max_log, error / (R_J_MOL_K * temperature * math.log(10.0)))
    row["_max_residual_J_per_mol"] = format(max_j, ".15g")
    row["_max_residual_log10_K"] = format(max_log, ".15g")


def _stored_shomate_hs(row: dict[str, Any], temperature: float) -> tuple[float, float]:
    coefficients = [float(row[key]) for key in "ABCDEFGH"]
    primitive_h, primitive_s = _shomate_primitives(coefficients[:5], temperature)
    return primitive_h + coefficients[5] - coefficients[7], primitive_s + coefficients[6]


def _shomate_join_open(left: dict[str, Any], right: dict[str, Any], temperature: float) -> bool:
    left_h, left_s = _stored_shomate_hs(left, temperature)
    right_h, right_s = _stored_shomate_hs(right, temperature)
    gap_g = abs((right_h - left_h) * 1000.0 - temperature * (right_s - left_s))
    return gap_g > 1e-3 or abs(right_s - left_s) > 1e-5


def _gas_coefficient_bases(temperature: float) -> tuple[np.ndarray, np.ndarray]:
    """Jacobian of enthalpy (J/mol) and entropy with respect to A..G."""
    t = temperature / 1000.0
    d_h = 1000.0 * np.array([
        t, t**2 / 2.0, t**3 / 3.0, t**4 / 4.0, -1.0 / t, 1.0, 0.0,
    ])
    d_s = np.array([
        math.log(t), t, t**2 / 2.0, t**3 / 3.0, -1.0 / (2.0 * t**2), 0.0, 1.0,
    ])
    return d_h, d_s


def _minimum_residual_shift(
    ordered: list[dict[str, Any]],
    joins: list[tuple[int, int, float]],
    *,
    ncoef: int,
    gibbs_basis,
    entropy_basis,
    frozen,
    extra=None,
    join_extra=None,
) -> np.ndarray | None:
    """Return per-segment coefficient deltas that close joins inside each residual.

    ``gibbs_basis(T)`` and ``entropy_basis(T)`` map a temperature to the
    Jacobian of G (J/mol) and S. A frozen segment is held at a zero delta.
    The result is None when no shift keeps every sample inside its budget.
    """
    n = len(ordered)
    width = n * ncoef + 1
    slack = width - 1
    equalities = []
    equality_values = []
    for left_index, right_index, temperature in joins:
        left_g, left_s = gibbs_basis(ordered[left_index], temperature, value=True)
        right_g, right_s = gibbs_basis(ordered[right_index], temperature, value=True)
        g_row = entropy_basis(temperature, kind="g")
        s_row = entropy_basis(temperature, kind="s")
        row = np.zeros(width)
        row[left_index * ncoef:(left_index + 1) * ncoef] = g_row
        row[right_index * ncoef:(right_index + 1) * ncoef] -= g_row
        equalities.append(row)
        equality_values.append(right_g - left_g)
        row = np.zeros(width)
        row[left_index * ncoef:(left_index + 1) * ncoef] = s_row
        row[right_index * ncoef:(right_index + 1) * ncoef] -= s_row
        equalities.append(row)
        equality_values.append(right_s - left_s)
    if join_extra is not None:
        for left_index, right_index, temperature in joins:
            more_eq, more_values = join_extra(
                left_index, right_index, temperature, width, ncoef
            )
            equalities.extend(more_eq)
            equality_values.extend(more_values)
    for index, segment in enumerate(ordered):
        if not frozen(segment):
            continue
        for coefficient in range(ncoef):
            row = np.zeros(width)
            row[index * ncoef + coefficient] = 1.0
            equalities.append(row)
            equality_values.append(0.0)
    inequalities = []
    inequality_values = []
    for index, segment in enumerate(ordered):
        if frozen(segment):
            continue
        for temperature in _seam_constraint_temperatures(segment):
            budget = _stated_gibbs_budget(segment, temperature)
            g_row = entropy_basis(temperature, kind="g")
            # |ΔG| <= t * budget. A zero budget freezes G at this sample.
            if budget <= 1e-12:
                row = np.zeros(width)
                row[index * ncoef:(index + 1) * ncoef] = g_row
                inequalities.append(row.copy())
                inequality_values.append(0.0)
                inequalities.append(-row)
                inequality_values.append(0.0)
                continue
            row = np.zeros(width)
            row[index * ncoef:(index + 1) * ncoef] = g_row
            row[slack] = -budget
            inequalities.append(row.copy())
            inequality_values.append(0.0)
            row = np.zeros(width)
            row[index * ncoef:(index + 1) * ncoef] = -g_row
            row[slack] = -budget
            inequalities.append(row)
            inequality_values.append(0.0)
    if extra is not None:
        for index, segment in enumerate(ordered):
            more_eq, more_eq_values, more_ub, more_ub_values = extra(
                index, segment, width, ncoef
            )
            equalities.extend(more_eq)
            equality_values.extend(more_eq_values)
            inequalities.extend(more_ub)
            inequality_values.extend(more_ub_values)
    if not equalities or not inequalities:
        return None
    objective = np.zeros(width)
    objective[slack] = 1.0
    bounds = [(None, None)] * (width - 1) + [(0.0, None)]
    solved = linprog(
        objective,
        A_ub=np.asarray(inequalities, dtype=float),
        b_ub=np.asarray(inequality_values, dtype=float),
        A_eq=np.asarray(equalities, dtype=float),
        b_eq=np.asarray(equality_values, dtype=float),
        bounds=bounds,
        method="highs",
    )
    if not solved.success or float(solved.x[slack]) > 1.0 + 1e-6:
        return None
    return solved.x[:-1].reshape(n, ncoef)


def _gas_window_delta(
    donor: dict[str, Any],
    frozen: dict[str, Any],
    join: float,
    far: float,
) -> np.ndarray | None:
    """Minimum-norm A–G change matching the neighbour at the join and the donor at the far end."""

    def basis(temperature: float) -> tuple[np.ndarray, np.ndarray]:
        d_h, d_s = _gas_coefficient_bases(temperature)
        return d_h - temperature * d_s, d_s

    join_g_row, join_s_row = basis(join)
    far_g_row, far_s_row = basis(far)
    frozen_h, frozen_s = _stored_shomate_hs(frozen, join)
    donor_h, donor_s = _stored_shomate_hs(donor, join)
    frozen_g = frozen_h * 1000.0 - join * frozen_s
    donor_g = donor_h * 1000.0 - join * donor_s
    design = np.vstack((join_g_row, join_s_row, far_g_row, far_s_row))
    target = np.array([frozen_g - donor_g, frozen_s - donor_s, 0.0, 0.0])
    kkt = np.block([
        [np.eye(7), design.T],
        [design, np.zeros((4, 4))],
    ])
    try:
        solved = np.linalg.solve(kkt, np.concatenate((np.zeros(7), target)))
    except np.linalg.LinAlgError:
        return None
    return solved[:7]


def _gas_window_move(delta: np.ndarray, low: float, high: float) -> float:
    worst = 0.0
    for temperature in np.linspace(low, high, 41):
        temperature = float(temperature)
        d_h, d_s = _gas_coefficient_bases(temperature)
        change = float((d_h - temperature * d_s) @ delta)
        worst = max(worst, abs(change))
    return worst


def _clip_shomate_sources(row: dict[str, Any]) -> None:
    t_min = float(row["T_min"])
    t_max = float(row["T_max"])
    kept = [
        node
        for node in row["_source_rows"]
        if t_min - 1e-6 <= float(node["temperature"]) <= t_max + 1e-6
    ]
    if not kept:
        return
    row["_source_rows"] = kept
    _refresh_shomate_residual(row)


def _insert_gas_transition(
    rows: list[dict[str, Any]],
    members: list[dict[str, Any]],
    left: dict[str, Any],
    right: dict[str, Any],
) -> None:
    """Split a short Shomate window off the side that can absorb the join."""
    join = float(left["T_max"])
    candidates = []
    rejected = []
    for donor, frozen, high_side in ((right, left, True), (left, right, False)):
        t_min = float(donor["T_min"])
        t_max = float(donor["T_max"])
        budget = _stated_gibbs_budget(donor, join)
        best = None
        for width in (20.0, 30.0, 40.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0):
            far = join + width if high_side else join - width
            if not (t_min + 5.0 < far < t_max - 5.0):
                continue
            low, high = sorted((join, far))
            if any(
                low + 1e-6 < audit < high - 1e-6
                for audit in _SEAM_AUDIT_TEMPERATURES_K
            ):
                continue
            delta = _gas_window_delta(donor, frozen, join, far)
            if delta is None:
                continue
            move = _gas_window_move(delta, low, high)
            fits = move <= budget + 1e-6
            if not fits:
                rejected.append((move, budget))
                continue
            record = (0, 0 if high_side else 1, move, far, delta, high_side)
            if best is None or record[:3] < best[:3]:
                best = record
        if best is not None:
            candidates.append((best, donor))
    if not candidates:
        if rejected:
            move, budget = min(rejected, key=lambda item: (item[0] - item[1], item[0]))
            raise ValueError(
                f"{left['species_name']} {left['Ref']}: gas transition at {join:g} K "
                f"requires Gibbs move {move:g} J/mol, exceeding donor budget "
                f"{budget:g} J/mol"
            )
        raise ValueError(
            f"{left['species_name']} {left['Ref']}: no gas transition at {join:g} K"
        )
    candidates.sort(key=lambda item: item[0][:3])
    (_rank, _side, _move, far, delta, high_side), donor = candidates[0]
    created = donor.copy()
    created["_source_rows"] = [dict(node) for node in donor["_source_rows"]]
    used_intervals = [
        int(row["T_interval"])
        for row in rows
        if row["species_name"] == donor["species_name"] and row["state"] == donor["state"]
    ]
    created["T_interval"] = str(max(used_intervals) + 1)
    donor_index = rows.index(donor)
    if high_side:
        created["T_min"] = format(join, ".15g")
        created["T_max"] = format(far, ".15g")
        donor["T_min"] = format(far, ".15g")
        rows.insert(donor_index, created)
    else:
        created["T_min"] = format(far, ".15g")
        created["T_max"] = format(join, ".15g")
        donor["T_max"] = format(far, ".15g")
        rows.insert(donor_index + 1, created)
    for name, change in zip("ABCDEFG", delta):
        created[name] = format(float(donor[name]) + float(change), ".15g")
    created["_seam_constrained"] = "true"
    donor["_seam_constrained"] = "true"
    _clip_shomate_sources(created)
    _clip_shomate_sources(donor)
    members.append(created)


def _pin_gas_same_ref_seams(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Make adjacent same-Ref gas segments share G and S at the join.

    Premise: each segment already carries its unconstrained Shomate fit.
    Algebra: a linear program moves A–G by the smallest factor of that fit's
    residual that makes enthalpy and entropy agree at every shared boundary.
    H = G + T*S then agrees as well. Segments that already agree, and any
    same-Ref chain with no stored fit, are left unchanged.
    """
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (row["species_name"], row["state"], row["Ref"]), []
        ).append(row)

    def gas_values(row: dict[str, Any], temperature: float, *, value: bool):
        enthalpy, entropy = _stored_shomate_hs(row, temperature)
        return enthalpy * 1000.0 - temperature * entropy, entropy

    def gas_basis(temperature: float, *, kind: str) -> np.ndarray:
        d_h, d_s = _gas_coefficient_bases(temperature)
        if kind == "s":
            return d_s
        return d_h - temperature * d_s

    for key, group in groups.items():
        if len(group) < 2:
            continue
        # Packaged K(g) and O(g) are restored after the write. Closing their
        # joins here would describe rows that are not shipped.
        if key[0] in PACK_REFERENCE_GAS:
            continue
        if not any("_source_rows" in row for row in group):
            continue
        if any("_source_rows" not in row for row in group):
            raise ValueError(f"{key}: same-Ref gas rows lack source nodes")
        ordered = sorted(group, key=lambda row: float(row["T_min"]))
        joins = []
        for index, (left, right) in enumerate(zip(ordered, ordered[1:])):
            if abs(float(left["T_max"]) - float(right["T_min"])) > 1e-4:
                continue
            temperature = float(left["T_max"])
            if _shomate_join_open(left, right, temperature):
                joins.append((index, index + 1, temperature))
        if not joins:
            continue
        def low_temperature_cp_match(left_index, right_index, temperature, width, ncoef):
            # The low-temperature gas rows are checked to share Cp at 1200 K.
            if abs(temperature - 1200.0) > 1e-6:
                return [], []
            if ordered[left_index]["species_name"] not in {
                "HCl(g)", "HF(g)", "NaCl(g)", "(NaCl)2(g)", "KCl(g)", "(KCl)2(g)",
                "NaF(g)", "(NaF)2(g)", "KF(g)", "(KF)2(g)", "Hg(g)",
            }:
                return [], []
            t = temperature / 1000.0
            jacobian = np.array([1.0, t, t**2, t**3, 1.0 / t**2, 0.0, 0.0])

            def heat_capacity(row: dict[str, Any]) -> float:
                return float(jacobian[:5] @ np.array([float(row[name]) for name in "ABCDE"]))

            row = np.zeros(width)
            row[left_index * ncoef:(left_index + 1) * ncoef] = jacobian
            row[right_index * ncoef:(right_index + 1) * ncoef] = -jacobian
            return [row], [
                heat_capacity(ordered[right_index]) - heat_capacity(ordered[left_index])
            ]

        deltas = _minimum_residual_shift(
            ordered,
            joins,
            ncoef=7,
            gibbs_basis=gas_values,
            entropy_basis=gas_basis,
            frozen=lambda _row: False,
            join_extra=low_temperature_cp_match,
        )
        accepted = deltas is not None
        if accepted:
            originals = [
                (
                    {name: segment[name] for name in "ABCDEFG"},
                    segment["_max_residual_J_per_mol"],
                    segment.get("_max_residual_log10_K"),
                    segment.get("_seam_constrained"),
                )
                for segment in ordered
            ]
            for segment, delta in zip(ordered, deltas):
                if float(np.max(np.abs(delta))) == 0.0:
                    continue
                for name, change in zip("ABCDEFG", delta):
                    segment[name] = format(float(segment[name]) + float(change), ".15g")
                _refresh_shomate_residual(segment)
                segment["_seam_constrained"] = "true"
            for segment, (coeffs, old_j, _old_log, _old_flag) in zip(ordered, originals):
                if _refreshed_residual_within_previous(
                    float(segment["_max_residual_J_per_mol"]), float(old_j)
                ):
                    continue
                accepted = False
                break
            if not accepted:
                for segment, (coeffs, old_j, old_log, old_flag) in zip(ordered, originals):
                    for name, value in coeffs.items():
                        segment[name] = value
                    segment["_max_residual_J_per_mol"] = old_j
                    if old_log is None:
                        segment.pop("_max_residual_log10_K", None)
                    else:
                        segment["_max_residual_log10_K"] = old_log
                    if old_flag is None:
                        segment.pop("_seam_constrained", None)
                    else:
                        segment["_seam_constrained"] = old_flag
        if not accepted:
            members = list(group)
            closed = False
            for _attempt in range(8):
                current = sorted(members, key=lambda item: float(item["T_min"]))
                pair = None
                for left, right in zip(current, current[1:]):
                    if abs(float(left["T_max"]) - float(right["T_min"])) > 1e-4:
                        continue
                    if _shomate_join_open(left, right, float(left["T_max"])):
                        pair = (left, right)
                        break
                if pair is None:
                    closed = True
                    break
                _insert_gas_transition(rows, members, pair[0], pair[1])
            if not closed:
                raise ValueError(
                    f"{key[0]} {key[2]}: gas seam cannot close inside the fit residual"
                )
            continue
        for left_index, right_index, temperature in joins:
            if _shomate_join_open(
                ordered[left_index], ordered[right_index], temperature
            ):
                raise ValueError(
                    f"{key[0]} {key[2]}: gas seam at {temperature:g} K stayed open"
                )
    return rows


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
    # CEA prints rounded coefficients; their 298 K polynomial evaluation can
    # miss the separately printed Hf anchor by a few joules after cancellation.
    # Hinc below is still zero at 298 K by construction. Mo/W oxide polymer
    # coefficients miss their separate Hf anchors by up to 20.24 J/mol
    # (NG-1258). The Burcat Pt- card has a 39.78 J/mol printed-anchor residual;
    # retain that card's published anchor and coefficients without widening
    # the consistency check for other cards.
    # NASA-7 coefficients are printed to finite precision. Retain the exact
    # published formation anchor and subtract the polynomial's own H(298),
    # while allowing the small coefficient-rounding residual found in cards.
    anchor_residual_limit = (
        41.0 if record["record_id"] == "Burcat-Pt-minus" else 21.0
    )
    if abs(h298_j - reference_h) > anchor_residual_limit:
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
        **_shomate_source_fields(fit),
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
                _DECLARED_T_MAX_BY_TABLE.get(
                    table_id, _FIT_T_MAX_BY_TABLE.get(table_id, FIT_T_MAX)
                )
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
        **_shomate_source_fields(fit),
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
        if table_id == "Cu-020" and temperature == 1600.0:
            # This source line concatenates three formation columns into one
            # malformed cell; exclude the entire ambiguous row from every fit.
            continue
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
        **_shomate_source_fields(fit),
    }


def _konings_integral(coefficients: Sequence[float], powers: Sequence[float],
                      lower_k: float, upper_k: float) -> tuple[float, float]:
    """Integrate a printed Konings Cp polynomial to enthalpy and entropy.

    Premise: the tables define Cp as a sum c_i*T**p_i. Algebra: integrate
    Cp dT for H and Cp/T dT for S; powers −1 in Cp require logarithms.
    Unit check: coefficients use the printed J/mol/K basis, so H is J/mol
    and S is J/mol/K. Sanity: for constant Cp, the result is Cp*ΔT and
    Cp*ln(T2/T1), respectively.
    """
    h = s = 0.0
    for coefficient, power in zip(coefficients, powers, strict=True):
        coefficient, power = float(coefficient), int(power)
        if power == -1:
            h += coefficient * math.log(upper_k / lower_k)
        else:
            h += coefficient * (upper_k ** (power + 1) - lower_k ** (power + 1)) / (power + 1)
        if power == 0:
            s += coefficient * math.log(upper_k / lower_k)
        else:
            s += coefficient * (upper_k**power - lower_k**power) / power
    return h, s


def _konings_source_rows(record: dict[str, Any], species: str,
                         temperatures: Sequence[float]) -> tuple[list[dict[str, float]], float]:
    """Convert one published polynomial record to the existing node shape."""
    item = record["gases"][species]
    segments = item["segments"]
    reference_t = NASA_STANDARD_T_K
    reference_h = float(item["dfH298_kJ_mol"])
    reference_s = float(item["S298_J_mol_K"])
    rows = []
    for temperature in temperatures:
        T = float(temperature)
        h = s = 0.0
        for lower, upper, coefficients, powers in segments:
            lo = max(reference_t, float(lower))
            hi = min(T, float(upper))
            if hi > lo:
                dh, ds = _konings_integral(coefficients, powers, lo, hi)
                h += dh
                s += ds
        active = next((seg for seg in segments if float(seg[0]) <= T <= float(seg[1])), segments[-1])
        cp = sum(float(c) * T ** int(p) for c, p in zip(active[2], active[3], strict=True))
        rows.append({"temperature": T, "heat_capacity": cp,
                     "entropy": reference_s + s,
                     "enthalpy_increment": h / 1000.0})
    return rows, reference_h


def _fit_konings_gas_row(record: dict[str, Any], species: str, cation: str,
                         cat_num: int, oxy_num: int, *, interval: int,
                         fit_t_min: float, fit_t_max: float,
                         runtime_t_min: float | None = None) -> dict[str, str]:
    nodes = np.arange(fit_t_min, fit_t_max + 50.0, 100.0)
    source_rows, reference = _konings_source_rows(record, species, nodes)
    fit = _fit_shomate_values(source_rows, reference)
    number = lambda value: format(float(value), ".15g")
    return {"species_name": species, "state": "g", "T_interval": str(interval),
            "cation": cation, "cat_num": str(cat_num), "oxy_num": str(oxy_num),
            "T_min": number(fit_t_min if runtime_t_min is None else runtime_t_min),
            "T_max": number(fit_t_max), **{key: number(fit[key]) for key in "ABCDEFG"},
            "H": "0", "Ref": f"Konings-{species}",
            "_max_residual_J_per_mol": number(fit["max_residual_J_per_mol"]),
            "_max_residual_log10_K": number(fit["max_residual_log10_K"]),
            **_shomate_source_fields(fit)}


def _konings_condensate_rows(record: dict[str, Any], species: str, cation: str,
                             cat_num: int, oxy_num: int) -> list[dict[str, str]]:
    item = record["condensates"][species]
    Tf = float(item["fusion_temperature_K"])
    phases = item["phases"]
    # Integrate each printed solid phase and its transition increment to Tf,
    # then extend the printed liquid's constant Cp below Tf as a supercooled
    # branch. The liquid anchor is the selected elemental dfH298 plus the
    # accumulated source H/S; no supercooling energy enters the uncertainty.
    T = NASA_STANDARD_T_K
    H = 0.0
    S = float(item["S298_J_mol_K"])
    current = phases[0]
    for phase in phases:
        if phase["name"] == "liquid":
            break
        start, end = map(float, phase["range_K"])
        if phase is not phases[0]:
            step = phase["transition_from_previous"]
            transition = float(step["temperature_K"])
            if transition > T:
                continuation_after = current.get("continuation_after_K")
                if continuation_after is not None:
                    continuation_start = max(T, float(continuation_after))
                    if transition > continuation_start:
                        dh, ds = _konings_integral(
                            current["coefficients"], current["powers"],
                            continuation_start, transition,
                        )
                        H += dh
                        S += ds
                T = transition
            H += float(step["enthalpy_kJ_mol"]) * 1000.0
            S += float(step["enthalpy_kJ_mol"]) * 1000.0 / transition
        current = phase
        if end > T:
            dh, ds = _konings_integral(phase["coefficients"], phase["powers"], T, end)
            H += dh; S += ds; T = end
    liquid = phases[-1]
    cp_liquid = float(item["liquid_Cp_J_mol_K"])
    # Fusion enthalpy is printed separately; include it exactly once.
    fusion = liquid["transition_from_previous"]
    H += float(fusion["enthalpy_kJ_mol"]) * 1000.0
    S += float(fusion["enthalpy_kJ_mol"]) * 1000.0 / Tf
    Hf = H
    Sf = S

    def source_rows(nodes: Sequence[float]) -> dict[float, dict[str, float]]:
        result = {}
        for value in nodes:
            t = float(value)
            result[t] = {"temperature": t, "heat_capacity": cp_liquid,
                         "entropy": Sf + cp_liquid * math.log(t / Tf),
                         "enthalpy_increment": (Hf + cp_liquid * (t - Tf)) / 1000.0}
        return result

    rows = []
    pending = [(1200.0, min(Tf, 3000.0))]
    if Tf < 3000.0:
        pending.append((Tf, 3000.0))
    while pending:
        lo, hi = pending.pop(0)
        nodes = [lo, *np.arange(lo + 100.0, hi, 100.0).tolist(), hi]
        fit = _fit_condensate_values(
            species, f"Konings-{species}", cation,
            cat_num, oxy_num, source_rows(nodes),
            float(item["dfH298_kJ_mol"]),
            fit_t_min=lo, fit_t_max=hi,
            runtime_t_min=lo, runtime_t_max=hi,
            fit_temperatures=nodes,
        )
        residual = float(fit["_max_residual_J_per_mol"])
        if residual > 10.0:
            midpoint = round(((lo + hi) / 2.0) / 100.0) * 100.0
            if midpoint <= lo or midpoint >= hi:
                midpoint = (lo + hi) / 2.0
            if midpoint <= lo or midpoint >= hi:
                raise ValueError(
                    f"{species}: source-node residual {residual:g} J/mol "
                    "exceeds 10 at the minimum interval width"
                )
            pending[0:0] = [(lo, midpoint), (midpoint, hi)]
            continue
        fit["Ref"] = f"Konings-Table88-Tf{Tf:g}"
        rows.append(fit)
    return sorted(rows, key=lambda row: float(row["T_min"]))


def _konings_condensate_uncertainty_j_mol(item: dict[str, Any], temperature: float) -> float:
    """Propagate printed La2O3 formation/step/fusion uncertainties to G(T).

    Premise: every phase step adds ΔH at Ttrs and ΔS=ΔH/Ttrs; algebra gives
    |∂G/∂ΔH|=|1−T/Ttrs|. For a transition-temperature uncertainty, differentiate
    the step plus adjacent Cp branches: dG/dTtrs = Cp(before)-Cp(after) +
    T*ΔH/Ttrs². Fusion-T uses the same derivative for the solid-to-liquid
    anchor and constant liquid Cp. Unit check: each term is J/mol/K. Sanity:
    at T=Ttrs, the ΔH sensitivity vanishes, while the temperature sensitivity
    retains the Cp discontinuity and transition-entropy contribution.
    The band sums absolute contributions and excludes supercooling energy;
    an unreported transition-temperature uncertainty contributes no term.
    """
    T = float(temperature)
    phases = item["phases"]
    bound = float(item["dfH_unc_kJ_mol"]) * 1000.0 + T * float(item["S_unc_J_mol_K"])
    for index, phase in enumerate(phases[1:]):
        step = phase["transition_from_previous"]
        transition = float(step["temperature_K"])
        dh = float(step["enthalpy_kJ_mol"]) * 1000.0
        bound += float(step["enthalpy_unc_kJ_mol"]) * 1000.0 * abs(1.0 - T / transition)
        before = phases[index]
        after = phase
        def cp_at(branch: dict[str, Any], at: float) -> float:
            return sum(float(c) * at ** int(p) for c, p in zip(
                branch["coefficients"], branch["powers"], strict=True
            ))
        dGdT = cp_at(before, transition) - cp_at(after, transition) + dh * T / transition**2
        if step.get("temperature_unc_K") is not None:
            bound += float(step["temperature_unc_K"]) * abs(dGdT)
    return bound


def _published_table_record(source_dir: Path, table_id: str) -> dict[str, Any]:
    """Read one transcribed record from its named published table bundle."""
    if table_id == "PM83-YO":
        return _load_record(source_dir / "pedley-marshall-1983-yo.json")
    if table_id in {"P672-HfO2-c", "P672-Hf-g"}:
        filename = "pankratz-hf.json"
    elif table_id.startswith("P672-Ru-"):
        filename = "pankratz-ru.json"
    elif table_id.startswith("P672-Rh-"):
        filename = "pankratz-rh.json"
    elif table_id.startswith("P677-"):
        filename = "pankratz-bulletin-677-re.json"
    elif table_id.startswith("P672-"):
        filename = "pankratz-bulletin-672.json"
    elif table_id.startswith("P696-"):
        filename = "pankratz-bulletin-696.json"
    elif table_id == "IVTAN-264":
        filename = "ivtan-table-264.json"
    elif table_id == "TSIV-1603":
        filename = "tsiv-table-1603.json"
    elif table_id == "TSIV-1436":
        filename = "tsiv-table-1436.json"
    elif table_id == "TSIV-1335":
        filename = "tsiv-table-1335.json"
    else:
        raise ValueError(f"{table_id}: unsupported published source record")
    record = _load_record(source_dir / filename)
    if table_id in {"IVTAN-264", "TSIV-1603", "TSIV-1436", "TSIV-1335"}:
        return record
    try:
        return record["tables"][table_id]
    except KeyError as error:
        raise ValueError(f"{filename}: no table record {table_id}") from error


def _published_gas_source_rows(
    source_dir: Path, table_id: str, *, breakpoint_side: str = "single"
) -> tuple[dict[float, dict[str, float]], float, float]:
    """Convert one published gas table's cells to the common fit units.

    Premise: the selected Pankratz tables print Cp and S in cal/(mol K), H-H298
    in kcal/mol, and Hf298 in kcal/mol; IVTAN Tables 264 and 1436 print Cp and
    S in J/(mol K), H-H0 in kJ/mol, and separately give H(298)-H(0). Algebra:
    multiply Pankratz thermal values by 4.184 and subtract IVTAN's H(298)-H(0)
    from each H-H0 cell. Unit check: the returned rows use J/(mol K), kJ/mol,
    and kJ/mol for Cp, S, and Hinc, respectively. Sanity: an IVTAN 298.15 K
    node becomes H-H(298)=0 and retains its printed S and Cp.
    """
    record = _published_table_record(source_dir, table_id)
    rows: dict[float, dict[str, float]] = {}
    if table_id == "IVTAN-264":
        h298_minus_h0 = float(record["enthalpy_298_minus_H0_kj_mol"])
        for temperature, cp, entropy, enthalpy, side in record["nodes"]:
            if side not in ("single", breakpoint_side):
                continue
            rows[float(temperature)] = {
                "temperature": float(temperature),
                "heat_capacity": float(cp),
                "entropy": float(entropy),
                "enthalpy_increment": float(enthalpy) - h298_minus_h0,
            }
        reference = float(record["formation_enthalpy_298_kj_mol"])
        source_limit = 3000.0
    elif table_id in {"TSIV-1436", "TSIV-1335"}:
        h298_minus_h0 = float(record["enthalpy_298_minus_H0_kj_mol"])
        for temperature, cp, entropy, enthalpy in record["nodes"]:
            rows[float(temperature)] = {
                "temperature": float(temperature),
                "heat_capacity": float(cp),
                "entropy": float(entropy),
                "enthalpy_increment": float(enthalpy) - h298_minus_h0,
            }
        reference = float(record["formation_enthalpy_298_kj_mol"])
        source_limit = max(rows)
    else:
        for temperature, cp, entropy, enthalpy in record["nodes"]:
            rows[float(temperature)] = {
                "temperature": float(temperature),
                "heat_capacity": float(cp) * 4.184,
                "entropy": float(entropy) * 4.184,
                "enthalpy_increment": float(enthalpy) * 4.184,
            }
        reference = float(record["formation_enthalpy_298_kcal_mol"]) * 4.184
        source_limit = float(record.get("source_limit_K", max(rows)))
    return rows, reference, source_limit


def _constant_cp_rows_through(
    rows: dict[float, dict[str, float]], end_temperature: float
) -> dict[float, dict[str, float]]:
    """Continue printed Cp, Hinc and S from the last published thermal node.

    Premise: the coverage-only continuation holds the last printed Cp
    constant. Algebra: Hinc(T)=Hinc(T0)+Cp*(T-T0)/1000 and
    S(T)=S(T0)+Cp*ln(T/T0). Unit check: the enthalpy increment is kJ/mol and
    entropy is J/(mol K). Sanity: both generated functions meet the last
    printed H/S node exactly; they are an estimate, not source data.
    """
    result = dict(rows)
    anchor_temperature = max(result)
    anchor = result[anchor_temperature]
    cp = anchor["heat_capacity"]
    previous_temperature = anchor_temperature
    previous_h = anchor["enthalpy_increment"]
    previous_s = anchor["entropy"]
    first_grid_temperature = (
        math.floor(anchor_temperature / JANAF_GRID_STEP_K) + 1
    ) * JANAF_GRID_STEP_K
    for temperature in np.arange(
        first_grid_temperature,
        end_temperature + JANAF_GRID_STEP_K / 2.0,
        JANAF_GRID_STEP_K,
    ):
        temperature = float(temperature)
        previous_h += cp * (temperature - previous_temperature) / 1000.0
        previous_s += cp * math.log(temperature / previous_temperature)
        result[temperature] = {
            "temperature": temperature,
            "heat_capacity": cp,
            "enthalpy_increment": previous_h,
            "entropy": previous_s,
        }
        previous_temperature = temperature
    return result


def _fit_published_gas_row(
    source_dir: Path,
    species_name: str,
    table_id: str,
    formula: str,
    cation: str,
    cat_num: int,
    oxy_num: int,
    *,
    t_interval: int,
    fit_t_min: float,
    fit_t_max: float,
    runtime_t_min: float,
    runtime_t_max: float | None = None,
    breakpoint_side: str = "single",
) -> dict[str, str]:
    record = _published_table_record(source_dir, table_id)
    source_formula = record.get("formula")
    source_species = record.get("species")
    if source_formula is not None and source_formula != formula:
        raise ValueError(f"{table_id}: expected published formula {formula}")
    if source_species is not None and source_species != species_name:
        raise ValueError(f"{table_id}: expected published species {species_name}")
    rows, reference, source_limit = _published_gas_source_rows(
        source_dir, table_id, breakpoint_side=breakpoint_side
    )
    rows = _constant_cp_rows_through(rows, fit_t_max)
    fit_step = 200.0 if table_id == "P677-Re-g" else JANAF_GRID_STEP_K
    temperatures = [
        float(T)
        for T in np.arange(fit_t_min, fit_t_max + fit_step / 2.0, fit_step)
    ]
    missing = [T for T in temperatures if T not in rows]
    if missing:
        raise ValueError(f"{table_id}: missing published gas fit nodes {missing!r}")
    fit = _fit_shomate_values([rows[T] for T in temperatures], reference)

    def number(value: float) -> str:
        return format(float(value), ".15g")

    ref = table_id
    if source_limit < fit_t_max:
        ref += f"-CP-EXT-{number(source_limit)}"
    return {
        "species_name": species_name,
        "state": "g",
        "T_interval": str(t_interval),
        "cation": cation,
        "cat_num": str(cat_num),
        "oxy_num": str(oxy_num),
        "T_min": number(runtime_t_min),
        "T_max": number(fit_t_max if runtime_t_max is None else runtime_t_max),
        **{key: number(fit[key]) for key in "ABCDEFG"},
        "H": "0",
        "Ref": ref,
        "_max_residual_J_per_mol": number(fit["max_residual_J_per_mol"]),
        "_max_residual_log10_K": number(fit["max_residual_log10_K"]),
        **_shomate_source_fields(fit),
    }


def _generated_gas_gibbs(
    gas_rows: list[dict[str, str]],
    species: str,
    temperature: float,
    source_id: str,
) -> float:
    candidates = [
        row
        for row in gas_rows
        if row["species_name"] == f"{species}(g)"
        and float(row["T_min"]) <= temperature <= float(row["T_max"])
    ]
    if not candidates:
        raise ValueError(
            f"{source_id}: no generated {species}(g) reference at "
            f"{temperature:g} K"
        )
    row = min(candidates, key=lambda candidate: int(candidate["T_interval"]))
    A, B, C, D, E, F, G = (float(row[key]) for key in "ABCDEFG")
    value = temperature / 1000.0
    enthalpy = (
        A * value
        + B * value**2 / 2.0
        + C * value**3 / 3.0
        + D * value**4 / 4.0
        - E / value
        + F
    )
    entropy = (
        A * math.log(value)
        + B * value
        + C * value**2 / 2.0
        + D * value**3 / 3.0
        - E / (2.0 * value**2)
        + G
    )
    return enthalpy * 1000.0 - temperature * entropy


def _fit_nureg_teo_rows(
    source_dir: Path, gas_rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Fit TeO to NUREG's Gibbs nodes with its later measured enthalpy anchor."""
    record = _load_record(source_dir / "nureg-cr-4308-teo.json")
    if record.get("table_id") != "NUREG-4308-TeO":
        raise ValueError("NUREG-4308-TeO: source record identity mismatch")
    nodes = record["nodes"]
    temperatures = np.asarray([float(node[0]) for node in nodes])
    anchor = record["anchor"]
    anchor_enthalpy = float(anchor["formation_enthalpy_kj_mol"])
    t = temperatures / 1000.0
    t0 = 298.15 / 1000.0

    def source_basis(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = np.column_stack(
            (values, values**2 / 2.0, values**3 / 3.0, values**4 / 4.0, -1.0 / values)
        )
        s = np.column_stack(
            (
                np.log(values),
                values,
                values**2 / 2.0,
                values**3 / 3.0,
                -1.0 / (2.0 * values**2),
            )
        )
        return h, s

    # Premise: Table 30 reports ΔfG for TeO(g) and Te(g) against the same
    # elements, so their difference is ΔrG for Te(g)+1/2 O2(g)=TeO(g).
    # Algebra: add that reaction ΔG to the generated Te(g) and O2(g) G_app
    # rows. The NUREG temperature curve implies Hf298=17.043 kJ/mol; the later
    # primary measurement supplies 72.7 kJ/mol, so its +55.657 kJ/mol update
    # shifts G by a constant while retaining the NUREG thermal dependence.
    # Unit check: Table 30 calories use 4.184 J/cal; the Shomate enthalpy basis
    # is kJ/mol and its entropy basis is J/(mol K). Sanity: the corrected fit
    # follows all 19 converted nodes within its recorded residual.
    source_gibbs = np.asarray(
        [
            _generated_gas_gibbs(gas_rows, "Te", float(node[0]), "NUREG-4308-TeO")
            + 0.5
            * _generated_gas_gibbs(gas_rows, "O2", float(node[0]), "NUREG-4308-TeO")
            + (float(node[1]) - float(node[2])) * 4.184
            for node in nodes
        ]
    )
    h_basis, s_basis = source_basis(t)
    h0, s0 = source_basis(np.asarray([t0]))
    h0 = h0[0]
    s0 = s0[0]
    design = np.column_stack(
        (
            1000.0 * (h_basis - h0)
            - temperatures[:, None] * (s_basis - s0),
            np.full(len(temperatures), 1000.0),
            -temperatures,
        )
    )
    fitted = np.linalg.lstsq(design, source_gibbs, rcond=None)[0]
    source_enthalpy = float(fitted[5])
    source_entropy = float(fitted[6])
    enthalpy_shift = anchor_enthalpy - source_enthalpy
    corrected_gibbs = source_gibbs + enthalpy_shift * 1000.0
    residual = float(
        np.max(np.abs(design @ fitted + enthalpy_shift * 1000.0 - corrected_gibbs))
    )
    residual_log10 = float(
        np.max(
            np.abs(
                (design @ fitted + enthalpy_shift * 1000.0 - corrected_gibbs)
                / (R_J_MOL_K * temperatures * np.log(10.0))
            )
        )
    )
    A, B, C, D, E = (float(value) for value in fitted[:5])
    F = anchor_enthalpy - float(h0 @ fitted[:5])
    G = source_entropy - float(s0 @ fitted[:5])

    def number(value: float) -> str:
        return format(float(value), ".15g")

    coefficients = {key: number(value) for key, value in zip("ABCDEFG", (A, B, C, D, E, F, G))}
    rows = []
    for interval, runtime_min, runtime_max in (
        (1, 1500.0, 3000.0),
        (2, 1200.0, 1500.0),
    ):
        rows.append(
            {
                "species_name": "TeO(g)",
                "state": "g",
                "T_interval": str(interval),
                "cation": "Te",
                "cat_num": "1",
                "oxy_num": "1",
                "T_min": number(runtime_min),
                "T_max": number(runtime_max),
                **coefficients,
                "H": "0",
                "Ref": "NUREG-4308-TeO",
                "_max_residual_J_per_mol": number(residual),
                "_max_residual_log10_K": number(residual_log10),
            }
        )
    if residual > 10.0:
        raise ValueError(
            f"TeO(g): NUREG fit residual {residual:g} J/mol exceeds 10"
        )
    return rows


def _fit_kelley_sbo_rows(
    source_dir: Path, gas_rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Fit Kelley SbO increments, anchored and bridged to selected Sb sources."""
    record = _load_record(source_dir / "kelley-1960-sbo.json")
    if record.get("table_id") != "Kelley-584-SbO":
        raise ValueError("Kelley-584-SbO: source record identity mismatch")

    thermal = record["thermal_source"]
    nodes = np.asarray(thermal["nodes"], dtype=float)
    temperatures = nodes[:, 0]
    h_nodes = nodes[:, 1] * 4.184
    s_nodes = nodes[:, 2] * 4.184
    t = temperatures / 1000.0
    t298 = 298.15 / 1000.0

    def source_basis(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = np.column_stack(
            (values, values**2 / 2.0, values**3 / 3.0, values**4 / 4.0, -1.0 / values)
        )
        s = np.column_stack(
            (
                np.log(values),
                values,
                values**2 / 2.0,
                values**3 / 3.0,
                -1.0 / (2.0 * values**2),
            )
        )
        return h, s

    h_basis, s_basis = source_basis(t)
    h298_basis, s298_basis = source_basis(np.asarray([t298]))
    h298_basis = h298_basis[0]
    s298_basis = s298_basis[0]

    # Premise: Kelley Table 30 supplies independent H-H298 and S-S298 nodes,
    # while NBS Table 21 supplies ΔfH°298 and Sohn et al. Table V supplies a
    # selected liquid-Sb oxidation function. Algebra: fit A-E jointly to
    # ΔH/T and ΔS (both J mol^-1 K^-1), then infer S298 where the fitted
    # thermal increments close to the Sohn reaction bridged through NUREG's
    # Sb(g)-Sb(l) Gibbs difference. Units are cal*4.184 -> J and Shomate H is
    # converted from kJ/mol before division by T. Sanity: Cp remains positive;
    # the NUREG SbO alternative is retained as a separate disagreement band.
    design_h = 1000.0 * (h_basis - h298_basis) / temperatures[:, None]
    design_s = s_basis - s298_basis
    coefficients = np.linalg.lstsq(
        np.vstack((design_h, design_s)),
        np.concatenate((h_nodes / temperatures, s_nodes)),
        rcond=None,
    )[0]
    h_residuals = 1000.0 * (h_basis - h298_basis) @ coefficients - h_nodes
    s_residuals = (s_basis - s298_basis) @ coefficients - s_nodes

    anchor = record["formation_enthalpy_anchor"]
    formation_enthalpy = float(anchor["formation_enthalpy_kj_mol"])
    reaction = record["reaction_source"]["gibbs_function"]
    reaction_constant = float(reaction["constant_J_mol"])
    reaction_slope = float(reaction["slope_J_mol_K"])
    comparison_nodes = {
        int(node[0]): tuple(float(value) for value in node[1:])
        for node in record["comparison_source"]["nodes"]
    }

    def source_gibbs(temperature: float) -> float:
        # Sohn's function starts from liquid Sb. NUREG gives G(Sb(g))-G(Sb(l)),
        # so subtract that vaporization Gibbs value to express the reaction on
        # the generated Sb(g)/O2(g) basis.
        sb_g, sb_l, _ = comparison_nodes[int(temperature)]
        vaporization_gibbs = (sb_g - sb_l) * 4.184
        gas_basis_reaction_gibbs = (
            reaction_constant + reaction_slope * temperature - vaporization_gibbs
        )
        return (
            _generated_gas_gibbs(gas_rows, "Sb", temperature, "Kelley-584-SbO")
            + 0.5
            * _generated_gas_gibbs(gas_rows, "O2", temperature, "Kelley-584-SbO")
            + gas_basis_reaction_gibbs
        )

    source_temperatures = np.asarray((1200.0, 1400.0, 1600.0, 1800.0, 2000.0))
    node_by_temperature = {
        int(node[0]): (float(node[1]) * 4.184, float(node[2]) * 4.184)
        for node in nodes
    }
    entropy_anchors = []
    for temperature in source_temperatures:
        h_increment, s_increment = node_by_temperature[int(temperature)]
        entropy_anchors.append(
            (
                formation_enthalpy * 1000.0
                + h_increment
                - source_gibbs(float(temperature))
            )
            / temperature
            - s_increment
        )
    formation_entropy = float(np.mean(entropy_anchors))

    A, B, C, D, E = (float(value) for value in coefficients)
    F = formation_enthalpy - float(h298_basis @ coefficients)
    G = formation_entropy - float(s298_basis @ coefficients)

    def shomate_cp(temperature: float) -> float:
        value = temperature / 1000.0
        return A + B * value + C * value**2 + D * value**3 + E / value**2

    if min(shomate_cp(float(temperature)) for temperature in temperatures) <= 0.0:
        raise ValueError("Kelley-584-SbO: fitted heat capacity is not positive")

    def number(value: float) -> str:
        return format(float(value), ".15g")

    low_coefficients = {key: number(value) for key, value in zip("ABCDEFG", (A, B, C, D, E, F, G))}
    low_row = {
        "species_name": "SbO(g)",
        "state": "g",
        "T_interval": "2",
        "cation": "Sb",
        "cat_num": "1",
        "oxy_num": "1",
        "T_min": "1200",
        "T_max": "2000",
        **low_coefficients,
        "H": "0",
        "Ref": "Kelley-584-SbO",
    }

    boundary_t = 2.0
    boundary_h = (
        A * boundary_t
        + B * boundary_t**2 / 2.0
        + C * boundary_t**3 / 3.0
        + D * boundary_t**4 / 4.0
        - E / boundary_t
        + F
    )
    boundary_s = (
        A * math.log(boundary_t)
        + B * boundary_t
        + C * boundary_t**2 / 2.0
        + D * boundary_t**3 / 3.0
        - E / (2.0 * boundary_t**2)
        + G
    )
    held_cp = shomate_cp(2000.0)
    high_coefficients = {
        "A": number(held_cp),
        "B": "0",
        "C": "0",
        "D": "0",
        "E": "0",
        "F": number(boundary_h - held_cp * boundary_t),
        "G": number(boundary_s - held_cp * math.log(boundary_t)),
    }
    high_row = {
        "species_name": "SbO(g)",
        "state": "g",
        "T_interval": "1",
        "cation": "Sb",
        "cat_num": "1",
        "oxy_num": "1",
        "T_min": "2000",
        "T_max": "3000",
        **high_coefficients,
        "H": "0",
        "Ref": "Kelley-584-SbO-CP-EXT-2000",
    }

    source_residuals = [
        _generated_gas_gibbs([low_row, high_row], "SbO", float(temperature), "Kelley-584-SbO")
        - source_gibbs(float(temperature))
        for temperature in range(1200, 2001, 100)
    ]
    max_source_residual = max(abs(value) for value in source_residuals)
    max_source_log10_residual = max(
        abs(value / (R_J_MOL_K * temperature * math.log(10.0)))
        for temperature, value in zip(range(1200, 2001, 100), source_residuals)
    )
    seam_residual = (
        _generated_gas_gibbs([high_row], "SbO", 2000.0, "Kelley-584-SbO")
        - _generated_gas_gibbs([low_row], "SbO", 2000.0, "Kelley-584-SbO")
    )
    for row in (low_row, high_row):
        row["_max_enthalpy_increment_residual_J_per_mol"] = number(
            float(np.max(np.abs(h_residuals)))
        )
        row["_max_entropy_increment_residual_J_per_mol_K"] = number(
            float(np.max(np.abs(s_residuals)))
        )
        row["_inferred_formation_entropy_J_per_mol_K"] = number(formation_entropy)
    low_row["_max_residual_J_per_mol"] = number(max_source_residual)
    low_row["_max_residual_log10_K"] = number(max_source_log10_residual)
    high_row["_max_residual_J_per_mol"] = number(abs(seam_residual))
    high_row["_max_residual_log10_K"] = number(
        abs(seam_residual / (R_J_MOL_K * 2000.0 * math.log(10.0)))
    )
    return [high_row, low_row]
def _fit_re2o7_constructed_gas_rows(
    source_dir: Path,
) -> list[dict[str, str]]:
    """Construct Re2O7(g) from measured vapor pressures and B677 functions."""
    source = _load_record(source_dir / "pankratz-bulletin-677-re.json")
    tables = source["tables"]
    pressure = source["vapor_pressure_sources"]["Battles_Gundersen_Edwards_1968"]
    equilibrium = pressure["runtime_construction_equilibrium"]
    A = float(equilibrium["intercept"])
    B = float(equilibrium["inverse_temperature_coefficient_K"])

    def source_state(table_id: str, temperature: float) -> tuple[float, float, float, float]:
        """Evaluate B677 Cp/Hinc/S with piecewise-linear Cp integration."""
        record = tables[table_id]
        nodes = [
            (
                float(row[0]), float(row[1]) * 4.184,
                float(row[2]) * 4.184, float(row[3]) * 4.184,
            )
            for row in record["nodes"]
        ]
        # Table 1251's liquid branch begins with its printed fusion jump at
        # 600 K; anchor on its last liquid node not above T and hold the
        # printed liquid Cp after the 700 K source limit.
        if table_id == "P677-Re2O7-l":
            anchor = max(node for node in nodes if node[0] <= temperature)
            cp = anchor[1]
            delta_t = temperature - anchor[0]
            h_increment = anchor[3] + cp * delta_t / 1000.0
            entropy = anchor[2] + cp * math.log(temperature / anchor[0])
            return cp, h_increment, entropy, float(
                record["formation_enthalpy_298_kcal_mol"]
            ) * 4.184

        def cp_at(at: float) -> float:
            if at <= nodes[0][0]:
                return nodes[0][1]
            for left, right in zip(nodes, nodes[1:]):
                if at <= right[0]:
                    fraction = (at - left[0]) / (right[0] - left[0])
                    return left[1] + fraction * (right[1] - left[1])
            return nodes[-1][1]

        anchor = nodes[0]
        h_increment = anchor[3]
        entropy = anchor[2]
        points = [anchor[0]]
        points.extend(node[0] for node in nodes[1:] if node[0] < temperature)
        points.append(temperature)
        for low, high in zip(points, points[1:]):
            cp_low = cp_at(low)
            cp_high = cp_at(high)
            slope = (cp_high - cp_low) / (high - low)
            delta_t = high - low
            log_ratio = math.log(high / low)
            h_increment += (cp_low * delta_t + slope * delta_t**2 / 2.0) / 1000.0
            entropy += cp_low * log_ratio + slope * (delta_t - low * log_ratio)
        return (
            cp_at(temperature), h_increment, entropy,
            float(record["formation_enthalpy_298_kcal_mol"]) * 4.184,
        )

    re_o2_ref = float(tables["P677-ReO2-c"]["formation_enthalpy_298_kcal_mol"]) * 4.184
    re_ref = float(tables["P677-Re-cr"]["formation_enthalpy_298_kcal_mol"]) * 4.184
    gas_reference = 3.5 * re_o2_ref - 1.5 * re_ref + R_J_MOL_K * math.log(10.0) * B / 1000.0
    pressure_entropy = (
        R_J_MOL_K * math.log(10.0) * A
        + R_J_MOL_K * math.log(1.01325)
    )

    def measured_state(temperature: float) -> dict[str, float]:
        cp_o2, h_o2, s_o2, _ = source_state("P677-ReO2-c", temperature)
        cp_re, h_re, s_re, _ = source_state("P677-Re-cr", temperature)
        cp = 3.5 * cp_o2 - 1.5 * cp_re
        h_increment = 3.5 * h_o2 - 1.5 * h_re
        entropy = 3.5 * s_o2 - 1.5 * s_re + pressure_entropy
        return {
            "temperature": temperature,
            "heat_capacity": cp,
            "enthalpy_increment": h_increment,
            "entropy": entropy,
        }

    # Fit each Smith phase's measurements as log10(p_bar)=a-b/T. Then
    # dG_sub(T)=-RT ln(p_bar)=R ln(10)(b-aT), with ΔH_sub=R ln(10)b and
    # ΔS_sub=R ln(10)a. Adding these to B677 Re2O7(c) below 600 K or Re2O7(l)
    # above the transition gives G_f(g)=G_f(cond)+dG_sub. For Battles,
    # log10(p_atm)=A-B/T describes 3.5 ReO2(s)->1.5 Re(s)+Re2O7(g). Convert
    # p_atm to the 1-bar standard with p_bar=1.01325*p_atm, then
    # ΔG_rxn=-RT ln(p_bar) and G_f(g)=ΔG_rxn+3.5G_f(ReO2)-1.5G_f(Re).
    # Subtracting B677 Re2O7(l) Gibbs gives dG_sub; adding it back reconstructs
    # G_f(g). These are flagged constructions, not a published Re2O7(g) Cp table.
    smith = source["vapor_pressure_sources"]["Smith_Line_Bell_1952"]
    smith_points = smith["measured_points_temperature_C_pressure_mmHg"]
    smith_temperatures = np.array([float(row[0]) + 273.15 for row in smith_points])
    smith_pressures_bar = np.array(
        [float(row[1]) * 1.01325 / 760.0 for row in smith_points]
    )
    smith_log_pressure = np.log10(smith_pressures_bar)
    smith_phase_fits = []
    for interval, table_id, source_ref, mask in (
        (
            1,
            "P677-Re2O7-c-high",
            "P677-Re2O7-g-SMITH-EST-598",
            smith_temperatures < 600.0,
        ),
        (
            2,
            "P677-Re2O7-l",
            "P677-Re2O7-g-SMITH-EST-633",
            smith_temperatures >= 600.0,
        ),
    ):
        indices = np.flatnonzero(mask)
        phase_temperatures = smith_temperatures[indices]
        design = np.column_stack((np.ones_like(phase_temperatures), 1.0 / phase_temperatures))
        phase_log_pressure = smith_log_pressure[indices]
        a, slope = np.linalg.lstsq(design, phase_log_pressure, rcond=None)[0]
        b = -float(slope)
        residual = phase_log_pressure - (float(a) - b / phase_temperatures)
        condensed_reference = (
            float(tables[table_id]["formation_enthalpy_298_kcal_mol"]) * 4.184
        )
        delta_h = R_J_MOL_K * math.log(10.0) * b
        smith_phase_fits.append(
            {
                "interval": interval,
                "table_id": table_id,
                "source_ref": source_ref,
                "indices": indices,
                "a": float(a),
                "b": b,
                "delta_h": delta_h,
                "delta_s": R_J_MOL_K * math.log(10.0) * float(a),
                "gas_reference": condensed_reference + delta_h / 1000.0,
                "max_log_pressure_residual": float(np.max(np.abs(residual))),
            }
        )
    smith_max_log_pressure_residual = max(
        fit["max_log_pressure_residual"] for fit in smith_phase_fits
    )

    # Verify dG_sub=-RT ln(p_bar) and G_f(g)=G_f(cond)+dG_sub at every
    # measured Smith point. Residuals from the phase-specific fits set the
    # pressure-scatter band; the solid and liquid fits retain their own H/S.
    for fit in smith_phase_fits:
        for index in fit["indices"]:
            temperature = float(smith_temperatures[index])
            pressure_bar = float(smith_pressures_bar[index])
            condensed = source_state(fit["table_id"], temperature)
            condensed_g = (
                (condensed[3] + condensed[1]) * 1000.0
                - temperature * condensed[2]
            )
            gas_from_pressure = condensed_g - R_J_MOL_K * temperature * math.log(
                pressure_bar
            )
            gas_from_fit = condensed_g + R_J_MOL_K * math.log(10.0) * (
                fit["b"] - fit["a"] * temperature
            )
            residual = gas_from_pressure - gas_from_fit
            expected_residual = -R_J_MOL_K * math.log(10.0) * temperature * (
                smith_log_pressure[index]
                - (fit["a"] - fit["b"] / temperature)
            )
            if not math.isclose(residual, expected_residual, rel_tol=0.0, abs_tol=1e-8):
                raise ValueError("Re2O7(g): Smith pressure-to-Gibbs conversion failed")

    def fit_row(
        interval: int,
        low: float,
        high: float,
        source_rows: list[dict[str, float]],
        reference: float,
        source_ref: str,
    ) -> dict[str, str]:
        fit = _fit_shomate_values(source_rows, reference)

        def number(value: float) -> str:
            return format(float(value), ".15g")

        return {
            "species_name": "Re2O7(g)",
            "state": "g",
            "T_interval": str(interval),
            "cation": "Re",
            "cat_num": "2",
            "oxy_num": "7",
            "T_min": number(low),
            "T_max": number(high),
            **{key: number(fit[key]) for key in "ABCDEFG"},
            "H": "0",
            "Ref": source_ref,
            "_max_residual_J_per_mol": number(fit["max_residual_J_per_mol"]),
            "_max_residual_log10_K": number(fit["max_residual_log10_K"]),
            **_shomate_source_fields(fit),
        }

    def smith_row(phase_fit: dict[str, Any]) -> dict[str, str]:
        interval = int(phase_fit["interval"])
        table_id = str(phase_fit["table_id"])
        source_ref = str(phase_fit["source_ref"])
        indices = phase_fit["indices"]
        low = float(smith_temperatures[indices[0]])
        high = float(smith_temperatures[indices[-1]])
        grid = sorted({low, high, *np.arange(math.ceil(low), high, 1.0).tolist()})
        source_rows = []
        for temperature in grid:
            cp, h_increment, entropy, _ = source_state(table_id, temperature)
            source_rows.append(
                {
                    "temperature": temperature,
                    "heat_capacity": cp,
                    "enthalpy_increment": h_increment,
                    "entropy": entropy + float(phase_fit["delta_s"]),
                }
            )
        row = fit_row(
            interval,
            # Runtime bounds are exact integer K in this table. Fit to the
            # exact Celsius-to-Kelvin source endpoints, but round the exposed
            # interval inward so it never extends past measured pressure data.
            math.ceil(low),
            math.floor(high),
            source_rows,
            float(phase_fit["gas_reference"]),
            source_ref,
        )
        row["_max_log10_pressure_residual"] = format(
            smith_max_log_pressure_residual, ".15g"
        )
        return row

    smith_solid_row = smith_row(smith_phase_fits[0])
    smith_liquid_row = smith_row(smith_phase_fits[1])

    # The regression-derived branch and the condensed-function reconstruction
    # are checked at both measured Battles endpoints below. The resulting row
    # is a constructed estimate, not an independently published Re2O7(g) table.
    # The measured regression is valid only at 961-1087 K. Above 1087 K,
    # continue H/S with the derived gas Cp fixed at its 1087 K value.
    measured_high = measured_state(1087.0)
    for temperature in (961.0, 1087.0):
        state = measured_state(temperature)
        gas_g = (
            (gas_reference + state["enthalpy_increment"]) * 1000.0
            - temperature * state["entropy"]
        )
        pressure_bar = 1.01325 * 10.0 ** (A - B / temperature)
        delta_g = -R_J_MOL_K * temperature * math.log(pressure_bar)
        re_o2 = source_state("P677-ReO2-c", temperature)
        re = source_state("P677-Re-cr", temperature)
        equilibrium_g = (
            delta_g
            + 3.5 * ((re_o2[3] + re_o2[1]) * 1000.0 - temperature * re_o2[2])
            - 1.5 * ((re[3] + re[1]) * 1000.0 - temperature * re[2])
        )
        condensed = source_state("P677-Re2O7-l", temperature)
        condensed_g = (condensed[3] + condensed[1]) * 1000.0 - temperature * condensed[2]
        d_g_sub = equilibrium_g - condensed_g
        reconstructed_g = condensed_g + d_g_sub
        if not math.isclose(gas_g, equilibrium_g, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("Re2O7(g): vapor-pressure and H/S constructions disagree")
        if not math.isclose(reconstructed_g, equilibrium_g, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("Re2O7(g): condensed + sublimation Gibbs check failed")

    measured_rows = [
        measured_state(float(temperature))
        for temperature in np.arange(961.0, 1087.0 + 0.5, 1.0)
    ]
    battles_row = fit_row(
        3,
        961.0,
        1087.0,
        measured_rows,
        gas_reference,
        "P677-Re2O7-g-BATTLES-EST-961-1087",
    )

    # Battles' measured interval ends at 1087 K. Continue its derived gas Cp
    # at that endpoint, keeping H/S continuous and labelling the range.
    cp_continuation = measured_high["heat_capacity"]
    continuation_rows = []
    continuation_temperatures = sorted(
        {1087.0, 3000.0, *np.arange(1100.0, 3000.0 + 50.0, 100.0).tolist()}
    )
    for temperature in continuation_temperatures:
        temperature = float(temperature)
        continuation_rows.append({
            "temperature": temperature,
            "heat_capacity": cp_continuation,
            "enthalpy_increment": measured_high["enthalpy_increment"]
            + cp_continuation * (temperature - 1087.0) / 1000.0,
            "entropy": measured_high["entropy"]
            + cp_continuation * math.log(temperature / 1087.0),
        })
    continuation_row = fit_row(
        4, 1087.0, 3000.0, continuation_rows, gas_reference,
        "P677-Re2O7-g-EST-CONT-1087",
    )
    smith_solid_row["_max_log10_pressure_residual"] = format(
        smith_max_log_pressure_residual, ".15g"
    )
    smith_liquid_row["_max_log10_pressure_residual"] = format(
        smith_max_log_pressure_residual, ".15g"
    )
    return [smith_solid_row, smith_liquid_row, battles_row, continuation_row]


def _bi_plus_thermal_rows(
    temperatures: list[float], published_source_dir: Path
) -> tuple[list[dict[str, float]], float]:
    """Build Bi+ ideal-gas H/S nodes from its sourced anchor and ASD levels."""
    # Premise: NBS Table 22 supplies ΔfH°(Bi+, 298.15 K)=916.7 kJ/mol;
    # NIST ASD supplies Bi II term energies and J values. The ion is treated
    # as a monatomic ideal gas at 1 bar. Algebra: Qel=Σ(2J+1)e^(-hcν/kT),
    # Uel=R*<hcν/k>, Cp=5R/2+dUel/dT, S=Strans+R(ln Qel+Uel/RT), and Hinc
    # is integrated relative to 298.15 K. Unit check: hcν is converted from
    # cm^-1 to J per particle before multiplication by Avogadro's constant.
    # Sanity: the ground level has J=0, and excited levels make a small,
    # positive correction to H and S across the 1200-3000 K fit interval.
    R = R_J_MOL_K
    k_b = 1.380649e-23
    h = 6.62607015e-34
    c = 299792458.0
    avogadro = 6.02214076e23
    pressure = 100000.0
    record = _load_record(published_source_dir / "bismuth-ion-sources.json")
    mass = 208.9804e-3 / avogadro
    levels = tuple(
        (int(weight), float(energy))
        for weight, energy in record["levels_cm_minus_1"]
    )

    def electronic(temperature: float) -> tuple[float, float]:
        energies = [h * c * 100.0 * wavenumber for _, wavenumber in levels]
        weights = [
            degeneracy * math.exp(-energy / (k_b * temperature))
            for (degeneracy, _), energy in zip(levels, energies)
        ]
        partition = sum(weights)
        mean_energy_mol = avogadro * sum(
            weight * energy for weight, energy in zip(weights, energies)
        ) / partition
        return math.log(partition), mean_energy_mol

    def entropy(temperature: float) -> float:
        log_q, u_el = electronic(temperature)
        q_trans = (2.0 * math.pi * mass * k_b * temperature / h**2) ** 1.5
        q_trans *= k_b * temperature / pressure
        return R * (math.log(q_trans) + 2.5 + log_q) + u_el / temperature

    def enthalpy_increment(temperature: float) -> float:
        _log_q_t, u_t = electronic(temperature)
        _log_q_298, u_298 = electronic(NASA_STANDARD_T_K)
        return 2.5 * R * (temperature - NASA_STANDARD_T_K) + u_t - u_298

    rows = []
    for temperature in temperatures:
        rows.append(
            {
                "temperature": temperature,
                "heat_capacity": (
                    enthalpy_increment(temperature + 0.01)
                    - enthalpy_increment(temperature - 0.01)
                )
                / 0.02,
                "entropy": entropy(temperature),
                "enthalpy_increment": enthalpy_increment(temperature) / 1000.0,
            }
        )
    return rows, float(record["cation_formation_enthalpy_298_kj_mol"])


def _fit_bi_plus_row(
    published_source_dir: Path,
    *, t_interval: int, fit_t_min: float, fit_t_max: float, runtime_t_min: float
) -> dict[str, str]:
    temperatures = [
        float(T)
        for T in np.arange(fit_t_min, fit_t_max + JANAF_GRID_STEP_K / 2.0, JANAF_GRID_STEP_K)
    ]
    rows, reference = _bi_plus_thermal_rows(temperatures, published_source_dir)
    fit = _fit_shomate_values(rows, reference)

    def number(value: float) -> str:
        return format(float(value), ".15g")

    return {
        "species_name": "Bi+(g)",
        "state": "g",
        "T_interval": str(t_interval),
        "cation": "Bi",
        "cat_num": "1",
        "oxy_num": "0",
        "T_min": number(runtime_t_min),
        "T_max": number(fit_t_max),
        **{key: number(fit[key]) for key in "ABCDEFG"},
        "H": "0",
        "Ref": "NBS-T22-ASD-BiII",
        "_max_residual_J_per_mol": number(fit["max_residual_J_per_mol"]),
        "_max_residual_log10_K": number(fit["max_residual_log10_K"]),
        **_shomate_source_fields(fit),
    }


def _neutral_row_for_saha(
    rows: list[dict[str, Any]], element: str, temperature: float
) -> dict[str, Any]:
    """Pick the neutral interval the way the runtime selector does.

    The largest T_min that is still <= T wins, which is the upper row at a
    shared breakpoint. The previously packaged table is not a source: the
    ion has to follow the neutral fitted in this same build, including
    before that neutral's seam shift is applied.
    """
    name = f"{element}(g)"
    candidates = [
        row
        for row in rows
        if row["species_name"] == name and row.get("state") == "g"
    ]
    if not candidates:
        raise ValueError(f"{name}: Saha ion has no neutral row in this build")
    valid = [
        row for row in candidates if float(row["T_min"]) <= temperature + 1e-12
    ]
    selected = (
        max(valid, key=lambda row: float(row["T_min"]))
        if valid
        else min(candidates, key=lambda row: float(row["T_min"]))
    )
    if temperature < float(selected["T_min"]) - 1e-9 or temperature > float(selected["T_max"]) + 1e-9:
        raise ValueError(
            f"{name}: T={temperature:g} K is outside "
            f"[{selected['T_min']}, {selected['T_max']}]"
        )
    return selected


def _constructed_saha_ion_thermal_rows(
    element: str,
    temperatures: list[float],
    published_source_dir: Path,
    neutral_rows: list[dict[str, Any]],
) -> list[dict[str, float]]:
    """Derive flagged M+ H/S/Cp from the neutral function and ASD levels.

    At p°=1 bar, the ideal-gas standard partition factors give
    Kp(M -> M+ + e-) = (kBT/p°)*2*(2*pi*me*kBT/h²)^(3/2)
    *(Q_M+/Q_M)*exp(-IE/kBT), neglecting the tiny M+/M mass correction.
    Thus G(M+)+G(e-)-G(M) = IE + RT*ln(Q_M/Q_M+) + G_e,physical, where
    Z_e°=(kBT/p°)*2*(2*pi*me*kBT/h²)^(3/2), G_e,physical=-RT*ln(Z_e°),
    H_e,physical=5/2 RT, and S_e,physical=R*(ln(Z_e°)+5/2). The factor 2
    is electron spin. D-020's shipped electron row sets H_e(298.15 K)=0,
    so G_e,JANAF=G_e,physical-5/2*R*T0. Add the matching constant
    +5/2*R*T0 to G(M+) and retain the existing D-020 row; then the JANAF
    closure reproduces the dimensionless 1-bar Saha Kp exactly. Differentiating
    RT*ln(Q_M/Q_M+) gives S+ = S0-R*ln(Q_M/Q_M+)-(U0-U+)/T and
    H+ = H0+IE+5/2*R*T0+(U+-U0); Cp adds Cv,+-Cv,0. Energies are molar
    J and level inputs are 2J+1 with cm^-1 converted by N_A*h*c*100.
    """
    record = _load_record(
        published_source_dir / "constructed-ion-sources.json"
    )
    source = record["elements"][element]
    ie_j_mol = (
        float(source["ionization_energy"]["value_eV"])
        * 1.602176634e-19
        * 6.02214076e23
    )
    neutral_levels = tuple(
        (int(weight), float(energy))
        for weight, energy in source["neutral"][
            "levels_2Jplus1_energy_cm_minus_1"
        ]
    )
    ion_levels = tuple(
        (int(weight), float(energy))
        for weight, energy in source["cation"][
            "levels_2Jplus1_energy_cm_minus_1"
        ]
    )
    if not neutral_levels or not ion_levels:
        raise ValueError(f"{element} ASD partitions require a ground level")

    r = R_J_MOL_K
    t0 = NASA_STANDARD_T_K
    electron_reference_shift = 2.5 * r * t0
    h_c_mol_cm = 6.02214076e23 * 6.62607015e-34 * 299792458.0 * 100.0

    def electronic(
        levels: tuple[tuple[int, float], ...], temperature: float
    ) -> tuple[float, float, float]:
        energies = [h_c_mol_cm * level for _weight, level in levels]
        weighted = [
            degeneracy * math.exp(-energy / (r * temperature))
            for (degeneracy, _level), energy in zip(levels, energies)
        ]
        partition = math.fsum(weighted)
        probabilities = [weight / partition for weight in weighted]
        mean_energy = math.fsum(
            probability * energy
            for probability, energy in zip(probabilities, energies)
        )
        variance = math.fsum(
            probability * (energy - mean_energy) ** 2
            for probability, energy in zip(probabilities, energies)
        )
        return (
            math.log(partition),
            mean_energy,
            variance / (r * temperature**2),
        )

    thermal = []
    for temperature in temperatures:
        neutral = _neutral_row_for_saha(neutral_rows, element, temperature)
        enthalpy, entropy = _stored_shomate_hs(neutral, temperature)
        gibbs = enthalpy * 1000.0 - temperature * entropy
        t = temperature / 1000.0
        heat_capacity = (
            float(neutral["A"])
            + float(neutral["B"]) * t
            + float(neutral["C"]) * t**2
            + float(neutral["D"]) * t**3
            + float(neutral["E"]) / t**2
        )
        ln_q0, u0, cv0 = electronic(neutral_levels, temperature)
        ln_q_plus, u_plus, cv_plus = electronic(ion_levels, temperature)
        ln_q_ratio = ln_q0 - ln_q_plus
        g_plus = (
            gibbs
            + ie_j_mol
            + electron_reference_shift
            + r * temperature * ln_q_ratio
        )
        s_plus = (
            entropy
            - r * ln_q_ratio
            - (u0 - u_plus) / temperature
        )
        h_plus = g_plus + temperature * s_plus
        thermal.append(
            {
                "temperature": temperature,
                "heat_capacity": heat_capacity + cv_plus - cv0,
                "entropy": s_plus,
                "enthalpy_increment": h_plus / 1000.0,
            }
        )
    return thermal


def _fit_constructed_saha_ion_row(
    element: str,
    published_source_dir: Path,
    neutral_rows: list[dict[str, Any]],
    *,
    t_interval: int,
    fit_t_min: float,
    fit_t_max: float,
    runtime_t_min: float,
) -> dict[str, str]:
    temperatures = [
        float(T)
        for T in np.arange(
            fit_t_min,
            fit_t_max + JANAF_GRID_STEP_K / 2.0,
            JANAF_GRID_STEP_K,
        )
    ]
    thermal_rows = _constructed_saha_ion_thermal_rows(
        element, temperatures, published_source_dir, neutral_rows
    )
    fit = _fit_shomate_values(thermal_rows, 0.0)

    def number(value: float) -> str:
        return format(float(value), ".15g")

    return {
        "species_name": f"{element}+(g)",
        "state": "g",
        "T_interval": str(t_interval),
        "cation": element,
        "cat_num": "1",
        "oxy_num": "0",
        "T_min": number(runtime_t_min),
        "T_max": number(fit_t_max),
        **{key: number(fit[key]) for key in "ABCDEFG"},
        "H": "0",
        "Ref": f"FLAGGED-DERIVED-NIST-ASD-{element}",
        "_max_residual_J_per_mol": number(fit["max_residual_J_per_mol"]),
        "_max_residual_log10_K": number(fit["max_residual_log10_K"]),
        **_shomate_source_fields(fit),
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
        **_shomate_source_fields(fit),
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
    allow_missing_nodes: bool = False,
    fit_temperatures: Sequence[float] | None = None,
    upper_g_cp: tuple[float, float] | None = None,
    endpoint_g_s: Sequence[tuple[float, float, float]] | None = None,
    endpoint_cp: tuple[float, float] | None = None,
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
    temperatures = (
        [
            float(T)
            for T in np.arange(fit_t_min, fit_t_max + 50.0, JANAF_GRID_STEP_K)
        ]
        if fit_temperatures is None
        else sorted(float(T) for T in fit_temperatures)
    )
    if fit_temperatures is not None and (
        len(temperatures) < 2
        or len(set(temperatures)) != len(temperatures)
        or temperatures[0] != fit_t_min
        or temperatures[-1] != fit_t_max
    ):
        raise ValueError(f"{table_id}: explicit fit nodes must span the fit interval")
    missing = [T for T in temperatures if T not in source_rows]
    if missing:
        if not allow_missing_nodes:
            raise ValueError(f"{table_id}: missing condensate source nodes {missing!r}")
        temperatures = [T for T in temperatures if T in source_rows]
    minimum_nodes = min(
        5, round((fit_t_max - fit_t_min) / JANAF_GRID_STEP_K) + 1
    )
    if allow_missing_nodes and len(temperatures) < minimum_nodes:
        raise ValueError(
            f"{table_id}: condensate fit needs at least {minimum_nodes} source nodes"
        )
    rows = [source_rows[T] for T in temperatures]
    tau = np.asarray(temperatures, dtype=float) / 1000.0
    enthalpy_increment = np.asarray([row["enthalpy_increment"] for row in rows])
    entropy = np.asarray([row["entropy"] for row in rows])
    phi = entropy - enthalpy_increment * 1000.0 / np.asarray(temperatures)
    # Transition intervals can contain fewer than five published nodes. Fit
    # only as many polynomial terms as those independent source nodes support;
    # regular grid fits keep the historical quartic path byte-for-byte.
    # A G/S seam uses the full quartic even on a short node list: the anchors
    # supply the missing rank, and an unconstrained short fit is unchanged.
    degree = 4 if fit_temperatures is None else min(4, len(temperatures) - 1)
    if endpoint_g_s is not None and degree != 4:
        degree = 4
    fit_design = np.column_stack([tau**power for power in range(degree + 1)])
    if endpoint_g_s is not None:
        if upper_g_cp is not None:
            raise ValueError(
                f"{table_id}: G/S endpoint fit requires an unconstrained quartic"
            )
        constraints = []
        constraint_values = []
        for temperature, anchor_g, anchor_s in endpoint_g_s:
            anchor_tau = temperature / 1000.0
            powers = [anchor_tau**power for power in range(5)]
            constraints.extend((
                powers,
                [powers[0], 2.0 * powers[1], 3.0 * powers[2],
                 4.0 * powers[3], 5.0 * powers[4]],
            ))
            # G=-RT*P and S/R=P+tau*P'; coefficients are dimensionless.
            constraint_values.extend((
                (1000.0 * reference_kJ_mol - anchor_g)
                / (R_J_MOL_K * temperature),
                anchor_s / R_J_MOL_K,
            ))
        if endpoint_cp is not None:
            temperature, anchor_cp = endpoint_cp
            anchor_tau = temperature / 1000.0
            constraints.append(
                [0.0, 2.0 * anchor_tau, 6.0 * anchor_tau**2,
                 12.0 * anchor_tau**3, 20.0 * anchor_tau**4]
            )
            constraint_values.append(anchor_cp / R_J_MOL_K)
        constraints_array = np.asarray(constraints)
        normal = fit_design.T @ fit_design
        target = fit_design.T @ (phi / R_J_MOL_K)
        kkt = np.block([
            [normal, constraints_array.T],
            [constraints_array, np.zeros((len(constraint_values), len(constraint_values)))],
        ])
        fitted_coefficients = np.linalg.solve(
            kkt, np.concatenate((target, np.asarray(constraint_values)))
        )[:5]
    elif upper_g_cp is None:
        fitted_coefficients = np.linalg.lstsq(
            fit_design, phi / R_J_MOL_K, rcond=None
        )[0]
    else:
        if degree != 4:
            raise ValueError(
                f"{table_id}: endpoint-constrained condensate fit needs a quartic"
            )
        # The next runtime interval owns the seam. Constrain this fit to its
        # G and Cp there while least-squares fitting the source Phi nodes.
        # G fixes P(tau); Cp/R=2*tau*P'+tau^2*P'' supplies the derivative
        # constraint. G uses J/mol, Cp uses J/(mol K), and P is dimensionless.
        anchor_g, anchor_cp = upper_g_cp
        anchor_tau = runtime_t_max / 1000.0
        constraints = np.asarray(
            [
                [anchor_tau**power for power in range(5)],
                [0.0, 2.0 * anchor_tau, 6.0 * anchor_tau**2,
                 12.0 * anchor_tau**3, 20.0 * anchor_tau**4],
            ]
        )
        constraint_values = np.asarray(
            [
                (1000.0 * reference_kJ_mol - anchor_g)
                / (R_J_MOL_K * runtime_t_max),
                anchor_cp / R_J_MOL_K,
            ]
        )
        normal = fit_design.T @ fit_design
        target = fit_design.T @ (phi / R_J_MOL_K)
        kkt = np.block(
            [
                [normal, constraints.T],
                [constraints, np.zeros((2, 2))],
            ]
        )
        fitted_coefficients = np.linalg.solve(
            kkt, np.concatenate((target, constraint_values))
        )[:5]
    coefficients = np.zeros(5, dtype=float)
    coefficients[: degree + 1] = fitted_coefficients
    design = np.column_stack((np.ones_like(tau), tau, tau**2, tau**3, tau**4))
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
        "_source_rows": {
            float(temperature): {
                "enthalpy_increment": float(source["enthalpy_increment"]),
                "entropy": float(source["entropy"]),
            }
            for temperature, source in source_rows.items()
        },
        "_reference_kJ": float(reference_kJ_mol),
        "_fit_t_min": float(fit_t_min),
        "_fit_t_max": float(fit_t_max),
        "_fit_temperatures": (
            None
            if fit_temperatures is None
            else [float(temperature) for temperature in fit_temperatures]
        ),
        "_allow_missing_nodes": allow_missing_nodes,
        "_fit_table_id": table_id,
    }


def _condensate_seam_frozen(row: dict[str, Any]) -> bool:
    """Rows whose 1200 K state is copied by the low-oxide interval stay put."""
    if row.get("_seam_frozen") == "true":
        return True
    if row["species_name"] not in {"SiO2(l)", "Al2O3(l)", "MgO(l)", "CaO(l)"}:
        return False
    bounds = (float(row["T_min"]), float(row["T_max"]))
    return bounds in {(1000.0, 1200.0), (1200.0, 1500.0)}


# Same Cp allowance the packaged continuation test already uses. A seam
# window has to stay inside this band of the donor heat capacity. Negative
# Cp is outside the band whenever the donor Cp is positive.
DONOR_CP_BAND_J_MOL_K = 8.0
_OPEN_CONDENSATE_JOINS: list[str] = []
_CLOSED_CONDENSATE_JOINS: list[str] = []
_OPEN_GAS_JOINS: list[str] = []
_LAST_WINDOW_MISS = ""


def _continuation_polynomial_holds(row: dict[str, Any]) -> bool:
    """Constant-Cp continuation rows keep that Cp and the upper G/S anchor.

    The packaged continuation test allows 8 J/(mol K) of Cp error and 0.02 J/mol
    of Gibbs error at the row's own upper temperature. Entropy there is 2e-5
    J/(mol K), except the two unconstrained continuations at 0.2.
    """
    if row.get("_Cp_l_J_molK") is None:
        return True
    heat_capacity = float(row["_Cp_l_J_molK"])
    t_min = float(row["T_min"])
    t_max = float(row["T_max"])
    for temperature in np.linspace(t_min, t_max, 201):
        if abs(_condensate_row_g_cp(row, float(temperature))[1] - heat_capacity) > DONOR_CP_BAND_J_MOL_K:
            return False
    anchor_t = float(row["_T0_K"])
    anchor_h = float(row["_anchor_H_J_per_mol"])
    anchor_s = float(row["_anchor_S_J_molK"])
    analytic_s = anchor_s + heat_capacity * math.log(t_max / anchor_t)
    analytic_g = (
        anchor_h + heat_capacity * (t_max - anchor_t) - t_max * analytic_s
    )
    gibbs, entropy = _condensate_row_g_s(row, t_max)
    entropy_tolerance = (
        0.2 if row["species_name"] in {"Sc2O3(l)", "UO2(l)"} else 2.0e-5
    )
    return (
        abs(gibbs - analytic_g) <= 0.02
        and abs(entropy - analytic_s) <= entropy_tolerance
    )


def _condensate_join_open(
    left: dict[str, Any], right: dict[str, Any], temperature: float
) -> bool:
    left_g, left_s = _condensate_row_g_s(left, temperature)
    right_g, right_s = _condensate_row_g_s(right, temperature)
    return abs(right_g - left_g) > 1e-3 or abs(right_s - left_s) > 1e-5


def _refresh_condensate_residual(row: dict[str, Any], *, score_fit: bool = False) -> None:
    """Rewrite the stored residual from the row's current coefficients.

    ``score_fit`` scores the original fit nodes. A split window keeps the
    default and scores only its own runtime interval.
    """
    coefficients = [float(row[f"dG_{name}"]) for name in "ABCDE"]
    reference_enthalpy = float(row["dH298_R"])
    t_min = float(row["T_min"])
    t_max = float(row["T_max"])
    fit_nodes = _condensate_fit_node_temperatures(row) if score_fit else None

    def model_g(temperature: float) -> float:
        tau = temperature / 1000.0
        polynomial = sum(
            coefficients[power] * tau**power for power in range(5)
        )
        return (
            1000.0 * R_J_MOL_K * reference_enthalpy
            - R_J_MOL_K * temperature * polynomial
        )

    samples: list[tuple[float, float]] = []
    source_rows = row.get("_source_rows")
    reference = row.get("_reference_kJ")
    if source_rows and reference is not None:
        for key, source in source_rows.items():
            temperature = float(key)
            if fit_nodes is not None:
                if not any(abs(temperature - node) <= 1e-6 for node in fit_nodes):
                    continue
            elif temperature < t_min - 1e-6 or temperature > t_max + 1e-6:
                continue
            gibbs = (
                float(reference) + float(source["enthalpy_increment"])
            ) * 1000.0 - temperature * float(source["entropy"])
            samples.append((temperature, gibbs))
    elif row.get("_Cp_l_J_molK") is not None:
        heat_capacity = float(row["_Cp_l_J_molK"])
        anchor_t = float(row["_T0_K"])
        anchor_h = float(row["_anchor_H_J_per_mol"])
        anchor_s = float(row["_anchor_S_J_molK"])
        grid = np.linspace(t_min, t_max, max(2, int(round(t_max - t_min)) + 1))
        for temperature in grid:
            temperature = float(temperature)
            gibbs = anchor_h + heat_capacity * (temperature - anchor_t) - temperature * (
                anchor_s + heat_capacity * math.log(temperature / anchor_t)
            )
            samples.append((temperature, gibbs))
    if not samples:
        return
    max_j = 0.0
    max_log = 0.0
    for temperature, gibbs in samples:
        error = abs(model_g(temperature) - gibbs)
        max_j = max(max_j, error)
        max_log = max(max_log, error / (R_J_MOL_K * temperature * math.log(10.0)))
    row["_max_residual_J_per_mol"] = format(max_j, ".15g")
    row["_max_residual_log10_K"] = format(max_log, ".15g")


def _transition_quartic(
    donor: dict[str, Any],
    start_g: float,
    start_s: float,
    start_t: float,
    end_g: float,
    end_s: float,
    end_t: float,
) -> np.ndarray | None:
    """Cubic Hermite in a local coordinate, stored as a tau quartic.

    The local coordinate runs from 0 to 1 across the window, so a 20 K slice
    does not go through a Vandermonde in T/1000. The tau^4 coefficient is 0.
    """
    reference_enthalpy = float(donor["dH298_R"])

    def slope_state(gibbs: float, entropy: float, temperature: float) -> tuple[float, float]:
        tau = temperature / 1000.0
        value = (
            1000.0 * R_J_MOL_K * reference_enthalpy - gibbs
        ) / (R_J_MOL_K * temperature)
        derivative = (entropy / R_J_MOL_K - value) / tau
        return value, derivative

    start_value, start_slope = slope_state(start_g, start_s, start_t)
    end_value, end_slope = slope_state(end_g, end_s, end_t)
    width = (end_t - start_t) / 1000.0
    if abs(width) < 1e-12:
        return None
    start_moment = start_slope * width
    end_moment = end_slope * width
    local = (
        start_value,
        start_moment,
        -3.0 * start_value - 2.0 * start_moment + 3.0 * end_value - end_moment,
        2.0 * start_value + start_moment - 2.0 * end_value + end_moment,
    )
    alpha = 1.0 / width
    beta = -(start_t / 1000.0) * alpha
    accumulated = np.zeros(4)
    power = np.array([1.0])
    step = np.array([beta, alpha])
    for coefficient in local:
        accumulated[: len(power)] += coefficient * power
        power = np.convolve(power, step)
    return np.array([accumulated[0], accumulated[1], accumulated[2], accumulated[3], 0.0])


def _transition_widths(limit: float, *, continuation: bool) -> list[float]:
    """Widths from the shortest bridge up to the remainder of the donor segment."""
    shortest = 8.0 if continuation else 20.0
    if limit < shortest - 1e-9:
        return []
    if continuation:
        widths = [float(width) for width in np.arange(shortest, limit + 1e-9, 2.0)]
        if widths and widths[-1] < limit - 1e-6:
            widths.append(float(limit))
        return widths
    widths: list[float] = []
    width = shortest
    while width <= limit + 1e-9:
        widths.append(width)
        if width < 100.0:
            width += 10.0
        elif width < 300.0:
            width += 25.0
        else:
            width += 50.0
    if widths[-1] < limit - 1.0:
        widths.append(float(limit))
    return widths


def _transition_window_acceptable(
    donor: dict[str, Any],
    coefficients: np.ndarray,
    low: float,
    high: float,
) -> tuple[bool, float, float, float, bool]:
    """Whether the window stays inside the donor residual and Cp band.

    |ΔG| is measured against the donor polynomial on a 201-point grid and
    compared with that segment's stated residual at the same temperature.
    Cp must stay within ``DONOR_CP_BAND_J_MOL_K`` of the donor Cp. A positive
    donor Cp does not accept a negative window Cp.

    Returns acceptance, the worst |ΔG|, the Gibbs excess over the residual,
    the Cp excess over the band, and whether Cp went negative.
    """
    trial = donor.copy()
    for name, value in zip("ABCDE", coefficients):
        trial[f"dG_{name}"] = format(float(value), ".15g")
    worst = 0.0
    gibbs_excess = 0.0
    cp_excess = 0.0
    negative_cp = False
    for temperature in np.linspace(low, high, 201):
        temperature = float(temperature)
        donor_g, donor_cp = _condensate_row_g_cp(donor, temperature)
        window_g, window_cp = _condensate_row_g_cp(trial, temperature)
        error = abs(window_g - donor_g)
        worst = max(worst, error)
        gibbs_excess = max(
            gibbs_excess, error - _stated_gibbs_budget(donor, temperature)
        )
        cp_excess = max(cp_excess, abs(window_cp - donor_cp) - DONOR_CP_BAND_J_MOL_K)
        if donor_cp > 0.0 and window_cp <= 0.0:
            negative_cp = True
    acceptable = (
        gibbs_excess <= 1e-6 and cp_excess <= 1e-8 and not negative_cp
    )
    return acceptable, worst, gibbs_excess, cp_excess, negative_cp


def _open_condensate_join_report(
    left: dict[str, Any], right: dict[str, Any], temperature: float
) -> str:
    left_g, left_s = _condensate_row_g_s(left, temperature)
    right_g, right_s = _condensate_row_g_s(right, temperature)
    return (
        f"{left['species_name']} {left['Ref']} at {temperature:g} K: "
        f"|dG|={abs(right_g - left_g):.6g} J/mol "
        f"|dS|={abs(right_s - left_s):.6g} J/mol/K; "
        f"left residual {_stated_gibbs_budget(left, temperature):.6g}, "
        f"right residual {_stated_gibbs_budget(right, temperature):.6g}; "
        "no window kept G inside the donor residual and Cp inside the donor band"
        + (f"; widest tried: {_LAST_WINDOW_MISS}" if _LAST_WINDOW_MISS else "")
    )


def _insert_condensate_transition(
    rows: list[dict[str, Any]],
    members: list[dict[str, Any]],
    left: dict[str, Any],
    right: dict[str, Any],
) -> bool:
    """Split off a Hermite window that stays inside the donor residual and Cp band.

    The window matches the untouched segment at the join and the donor
    polynomial at the interior endpoint. Widths grow until both checks hold.
    A window that cannot is not inserted. Returns whether a window was added.
    """
    global _LAST_WINDOW_MISS
    join = float(left["T_max"])
    candidates = []
    # Widest rejected attempt on each side, so an open join records how far
    # the search went and which budget it still missed.
    widest_miss: dict[bool, tuple[float, str]] = {}
    for donor, frozen, high_side in ((right, left, True), (left, right, False)):
        side_name = "high" if high_side else "low"
        if _condensate_seam_frozen(donor):
            widest_miss[high_side] = (
                -1.0, f"{side_name} side frozen"
            )
            continue
        t_min = float(donor["T_min"])
        t_max = float(donor["T_max"])
        continuation = donor.get("_Cp_l_J_molK") is not None
        room = (t_max - join) if high_side else (join - t_min)
        best = None
        widths = _transition_widths(room - 5.0, continuation=continuation)
        if not widths:
            widest_miss[high_side] = (-1.0, f"{side_name} side has no room")
        for width in widths:
            far = join + width if high_side else join - width
            # Leave at least 5 K of the donor segment on the far side.
            if not (t_min + 5.0 - 1e-6 <= far <= t_max - 5.0 + 1e-6):
                continue
            low, high = sorted((join, far))
            if high_side:
                join_g, join_s = _condensate_row_g_s(frozen, join)
                far_g, far_s = _condensate_row_g_s(donor, far)
                coefficients = _transition_quartic(
                    donor, join_g, join_s, join, far_g, far_s, far
                )
            else:
                far_g, far_s = _condensate_row_g_s(donor, far)
                join_g, join_s = _condensate_row_g_s(frozen, join)
                coefficients = _transition_quartic(
                    donor, far_g, far_s, far, join_g, join_s, join
                )
            if coefficients is None:
                continue
            if continuation:
                trial = donor.copy()
                if high_side:
                    trial["T_min"] = format(join, ".15g")
                    trial["T_max"] = format(far, ".15g")
                else:
                    trial["T_min"] = format(far, ".15g")
                    trial["T_max"] = format(join, ".15g")
                    remainder = donor.copy()
                    remainder["T_max"] = format(far, ".15g")
                    if not _continuation_polynomial_holds(remainder):
                        widest_miss[high_side] = (
                            width,
                            f"{side_name} side {width:g} K: low-side remainder leaves the continuation anchor",
                        )
                        continue
                for name, value in zip("ABCDE", coefficients):
                    trial[f"dG_{name}"] = format(float(value), ".15g")
                if not _continuation_polynomial_holds(trial):
                    widest_miss[high_side] = (
                        width,
                        f"{side_name} side {width:g} K: continuation anchor does not hold",
                    )
                    continue
            acceptable, move, gibbs_excess, cp_excess, negative_cp = (
                _transition_window_acceptable(donor, coefficients, low, high)
            )
            if not acceptable:
                note = (
                    f"{side_name} side {width:g} K: "
                    f"Gibbs excess {max(gibbs_excess, 0.0):.6g} J/mol, "
                    f"Cp excess {max(cp_excess, 0.0):.6g} J/mol/K"
                )
                if negative_cp:
                    note += ", negative Cp"
                widest_miss[high_side] = (width, note)
                continue
            # Shortest window that holds, then the smaller Gibbs move, then
            # the upper segment so its T_max stays the packaged identity key.
            record = (width, move, 0 if high_side else 1, far, coefficients, high_side)
            if best is None or record[:3] < best[:3]:
                best = record
        if best is not None:
            candidates.append((best, donor))
    if not candidates:
        _LAST_WINDOW_MISS = "; ".join(
            widest_miss[side][1] for side in (True, False) if side in widest_miss
        )
        return False
    candidates.sort(key=lambda item: item[0][:3])
    (_width, _move, _side, far, coefficients, high_side), donor = candidates[0]
    _LAST_WINDOW_MISS = ""
    created = donor.copy()
    donor_index = rows.index(donor)
    if high_side:
        created["T_min"] = format(join, ".15g")
        created["T_max"] = format(far, ".15g")
        donor["T_min"] = format(far, ".15g")
        rows.insert(donor_index, created)
    else:
        created["T_min"] = format(far, ".15g")
        created["T_max"] = format(join, ".15g")
        donor["T_max"] = format(far, ".15g")
        rows.insert(donor_index + 1, created)
    for name, value in zip("ABCDE", coefficients):
        created[f"dG_{name}"] = format(float(value), ".15g")
    created["_seam_constrained"] = "true"
    donor["_seam_constrained"] = "true"
    _refresh_condensate_residual(created)
    _refresh_condensate_residual(donor)
    members.append(created)
    low, high = sorted((join, float(far)))
    side_name = "high" if high_side else "low"
    _CLOSED_CONDENSATE_JOINS.append(
        f"{donor['species_name']} {donor['Ref']} at {join:g} K: "
        f"closed by {side_name}-side Hermite window {low:g}-{high:g} K "
        f"(width {_width:g} K, max |dG| {_move:.6g} J/mol)"
    )
    return True


def _pin_condensate_same_ref_seams(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Make adjacent same-Ref condensate segments share G and S.

    Premise: independently fitted segments of one construction disagree at
    the shared boundary by more than the runtime tolerance. Algebra: move the
    quartic by the smallest factor of each segment's unconstrained residual
    that closes G and S. When that factor would exceed 1, insert a Hermite
    window only if its Gibbs energy stays inside the donor residual and its
    heat capacity stays inside the donor band. Otherwise the join stays open
    and is listed in ``_OPEN_CONDENSATE_JOINS``. H = G + T*S agrees on every
    window that is inserted. A second call sees closed joins and does not
    move them again.
    """
    _OPEN_CONDENSATE_JOINS.clear()
    _CLOSED_CONDENSATE_JOINS.clear()
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if "_max_residual_J_per_mol" not in row:
            continue
        groups.setdefault(
            (row["species_name"], row["state"], row["Ref"]), []
        ).append(row)

    def condensate_values(row: dict[str, Any], temperature: float, *, value: bool):
        del value
        return _condensate_row_g_s(row, temperature)

    def condensate_basis(temperature: float, *, kind: str) -> np.ndarray:
        tau = temperature / 1000.0
        if kind == "s":
            return R_J_MOL_K * np.array([(power + 1) * tau**power for power in range(5)])
        return -R_J_MOL_K * temperature * np.array([tau**power for power in range(5)])

    def continuation_bounds(index: int, segment: dict[str, Any], width: int, ncoef: int):
        """Keep a continuation row's upper anchor and its constant-Cp band."""
        if segment.get("_Cp_l_J_molK") is None:
            return [], [], [], []
        equalities = []
        equality_values = []
        inequalities = []
        inequality_values = []
        t_max = float(segment["T_max"])
        for kind in ("g", "s"):
            row = np.zeros(width)
            row[index * ncoef:(index + 1) * ncoef] = condensate_basis(t_max, kind=kind)
            equalities.append(row)
            equality_values.append(0.0)
        anchor = float(segment["_Cp_l_J_molK"])
        for temperature in _seam_sample_temperatures(
            float(segment["T_min"]), t_max
        ):
            current = _condensate_row_g_cp(segment, temperature)[1]
            tau = temperature / 1000.0
            jacobian = R_J_MOL_K * np.array(
                [0.0, 2.0 * tau, 6.0 * tau**2, 12.0 * tau**3, 20.0 * tau**4]
            )
            upper = np.zeros(width)
            upper[index * ncoef:(index + 1) * ncoef] = jacobian
            inequalities.append(upper.copy())
            inequality_values.append(8.0 - (current - anchor))
            lower = np.zeros(width)
            lower[index * ncoef:(index + 1) * ncoef] = -jacobian
            inequalities.append(lower)
            inequality_values.append(8.0 + (current - anchor))
        return equalities, equality_values, inequalities, inequality_values

    for group in groups.values():
        if len(group) < 2:
            continue
        # Generated fits of a pack-reference liquid are discarded at merge.
        # Recording a closure here would describe rows that are not written.
        if group[0]["species_name"] in PACK_REFERENCE_LIQUIDS:
            continue
        ordered = sorted(group, key=lambda item: float(item["T_min"]))
        joins = []
        for index, (left, right) in enumerate(zip(ordered, ordered[1:])):
            if abs(float(left["T_max"]) - float(right["T_min"])) > 1e-4:
                continue
            temperature = float(left["T_max"])
            if _condensate_join_open(left, right, temperature):
                joins.append((index, index + 1, temperature))
        if not joins:
            continue
        deltas = _minimum_residual_shift(
            ordered,
            joins,
            ncoef=5,
            gibbs_basis=condensate_values,
            entropy_basis=condensate_basis,
            frozen=_condensate_seam_frozen,
            extra=continuation_bounds,
        )
        if deltas is not None:
            trials = []
            applicable = True
            for segment, delta in zip(ordered, deltas):
                trial = segment.copy()
                if float(np.max(np.abs(delta))) != 0.0:
                    for name, change in zip("ABCDE", delta):
                        trial[f"dG_{name}"] = format(
                            float(segment[f"dG_{name}"]) + float(change), ".15g"
                        )
                if not _continuation_polynomial_holds(trial):
                    applicable = False
                    break
                if float(np.max(np.abs(delta))) != 0.0:
                    _refresh_condensate_residual(trial, score_fit=True)
                    if not _refreshed_residual_within_previous(
                        float(trial["_max_residual_J_per_mol"]),
                        float(segment["_max_residual_J_per_mol"]),
                    ):
                        applicable = False
                        break
                trials.append((segment, trial, delta))
            if applicable:
                for segment, trial, delta in trials:
                    if float(np.max(np.abs(delta))) == 0.0:
                        continue
                    for name in "ABCDE":
                        segment[f"dG_{name}"] = trial[f"dG_{name}"]
                    _refresh_condensate_residual(segment, score_fit=True)
                    segment["_seam_constrained"] = "true"
                species_name = ordered[0]["species_name"]
                reference = ordered[0]["Ref"]
                for _left_index, _right_index, temperature in joins:
                    _CLOSED_CONDENSATE_JOINS.append(
                        f"{species_name} {reference} at {temperature:g} K: "
                        "closed by linear program inside the previous residual"
                    )
                continue
        members = list(group)
        blocked: set[tuple[int, int]] = set()
        for _attempt in range(24):
            current = sorted(members, key=lambda item: float(item["T_min"]))
            pair = None
            pair_key = None
            for left, right in zip(current, current[1:]):
                if abs(float(left["T_max"]) - float(right["T_min"])) > 1e-4:
                    continue
                key = (id(left), id(right))
                if key in blocked:
                    continue
                temperature = float(left["T_max"])
                if _condensate_join_open(left, right, temperature):
                    pair = (left, right, temperature)
                    pair_key = key
                    break
            if pair is None:
                break
            if not _insert_condensate_transition(rows, members, pair[0], pair[1]):
                blocked.add(pair_key)
                _OPEN_CONDENSATE_JOINS.append(
                    _open_condensate_join_report(pair[0], pair[1], pair[2])
                )
        else:
            raise ValueError(
                f"{group[0]['species_name']} {group[0]['Ref']}: seam windows did not close"
            )
    return rows


def build_redox_metal_standard_rows(
    source_tables: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Fit JANAF pure-metal reference rows with the existing condensate form.

    Premise: each source table carries ``[T, Cp, S, H-H(298)]`` nodes and its
    own 298.15 K formation-enthalpy anchor. Algebra: delegate to
    ``_fit_condensate_values`` so these standards use the same φ polynomial
    and residual calculation as existing condensate rows. Unit check: the
    source H column is kJ/mol and S is J/(mol K), as required by that fitter.
    Sanity: each returned row records the maximum source-node residual.
    """
    fitted: dict[str, dict[str, Any]] = {}
    for (
        species_name,
        table_id,
        cation,
        cat_num,
        oxy_num,
        phase,
        fit_t_min,
        fit_t_max,
        runtime_t_min,
        runtime_t_max,
    ) in REDOX_METAL_STANDARD_SOURCES:
        table = source_tables[table_id]
        if species_name == "Fe(cr)":
            # JANAF Fe-007 reports stable crystal allotropes, with separate
            # alpha/gamma and gamma/delta values at 1184 and 1665 K. Fit each
            # phase from its own H-H(298), S, Cp nodes so no polynomial
            # smooths through either transition. The 1800-1809 K tail is the
            # existing short flagged continuation to the solid-liquid marker.
            phase_nodes = table.get("transition_nodes", [])
            source_nodes = {
                float(node[0]): {
                    "temperature": float(node[0]),
                    "heat_capacity": float(node[1]),
                    "entropy": float(node[2]),
                    "enthalpy_increment": float(node[3]),
                }
                for node in table["nodes"]
            }
            intervals = []
            # One alpha fit has a 19.316 J/mol source-node residual, so split
            # that phase at an existing 900 K node; each fitted segment then
            # stays under the same 10 J/mol gate without an anchor correction.
            for phase, fit_low, fit_high, run_low, run_high in (
                ("alpha", 500.0, 900.0, 500.0, 900.0),
                ("alpha", 900.0, 1184.0, 900.0, 1184.0),
                ("gamma", 1184.0, 1665.0, 1184.0, 1665.0),
                ("delta", 1665.0, 1800.0, 1665.0, 1809.0),
            ):
                branch_nodes = {
                    temperature: row
                    for temperature, row in source_nodes.items()
                    if fit_low <= temperature <= fit_high
                }
                for endpoint in (fit_low, fit_high):
                    phase_node = next(
                        (
                            node
                            for node in phase_nodes
                            if node.get("phase") == phase
                            and float(node["temperature_K"]) == endpoint
                        ),
                        None,
                    )
                    if phase_node is not None:
                        branch_nodes[endpoint] = {
                            "temperature": endpoint,
                            "heat_capacity": float(phase_node["heat_capacity"]),
                            "entropy": float(phase_node["entropy"]),
                            "enthalpy_increment": float(
                                phase_node["enthalpy_increment"]
                            ),
                        }
                temperatures = sorted(branch_nodes)
                row = _fit_condensate_values(
                    species_name,
                    table_id,
                    cation,
                    cat_num,
                    oxy_num,
                    branch_nodes,
                    float(table["formation_enthalpy_298_kj_mol"]),
                    fit_t_min=fit_low,
                    fit_t_max=fit_high,
                    runtime_t_min=run_low,
                    runtime_t_max=run_high,
                    ref=table_id,
                    fit_temperatures=temperatures,
                )
                row["state"] = phase
                intervals.append({
                    "T_domain_K": [run_low, run_high],
                    "row": row,
                })
            fitted[species_name] = {
                "intervals": intervals,
                "_max_residual_J_per_mol": format(
                    max(
                        float(interval["row"]["_max_residual_J_per_mol"])
                        for interval in intervals
                    ),
                    ".15g",
                ),
            }
            continue
        if species_name == "Mn(l)":
            # Preserve the original 1600-2200 K row, and fit the same JANAF
            # liquid reference on its newly represented lower and upper nodes.
            source_rows = {
                float(node[0]): {
                    "temperature": float(node[0]),
                    "heat_capacity": float(node[1]),
                    "entropy": float(node[2]),
                    "enthalpy_increment": float(node[3]),
                }
                for node in table["nodes"]
            }
            intervals = []
            for fit_low, fit_high in (
                (1200.0, 1600.0), (1600.0, 2200.0), (2200.0, 3000.0),
            ):
                row = _fit_condensate_values(
                    species_name,
                    table_id,
                    cation,
                    cat_num,
                    oxy_num,
                    source_rows,
                    float(table["formation_enthalpy_298_kj_mol"]),
                    fit_t_min=fit_low,
                    fit_t_max=fit_high,
                    runtime_t_min=fit_low,
                    runtime_t_max=fit_high,
                    ref=table_id,
                )
                row["state"] = phase
                intervals.append({"T_domain_K": [fit_low, fit_high], "row": row})
            fitted[species_name] = {
                "intervals": intervals,
                "_max_residual_J_per_mol": format(
                    max(
                        float(interval["row"]["_max_residual_J_per_mol"])
                        for interval in intervals
                    ),
                    ".15g",
                ),
            }
            continue
        source_rows = {
            float(node[0]): {
                "temperature": float(node[0]),
                "heat_capacity": float(node[1]),
                "entropy": float(node[2]),
                "enthalpy_increment": float(node[3]),
            }
            for node in table["nodes"]
        }
        row = _fit_condensate_values(
            species_name,
            table_id,
            cation,
            cat_num,
            oxy_num,
            source_rows,
            float(table["formation_enthalpy_298_kj_mol"]),
            fit_t_min=fit_t_min,
            fit_t_max=fit_t_max,
            runtime_t_min=runtime_t_min,
            runtime_t_max=runtime_t_max,
            ref=table_id,
        )
        row["state"] = phase
        fitted[species_name] = row
    return fitted


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
    record = _nasa_liquid_record(nasa_source_dir, species_name, table_id)
    if (
        record.get("record_id") != table_id
        or record.get("formula") != formula
        or record.get("phase") not in {"liquid", "L"}
    ):
        raise ValueError(f"{table_id}: NASA record identity does not match {species_name}")
    temperatures = sorted(
        {float(fit_t_min), float(fit_t_max)}
        | {
            float(T)
            for T in np.arange(fit_t_min, fit_t_max + 50.0, NASA_GRID_STEP_K)
        }
    )
    # The NASA-9 H(T)/(R*T) polynomial, including b1, gives the card's
    # assigned element-reference enthalpy H_card(T) directly. Thus
    # G_app(T)=H_card(T)-T*S_card(T); equivalently choose the source-row
    # increment H_card(T)-dfH298 so dfH298+increment reconstructs H_card(T).
    # Unit check: both enthalpies are J/mol before the increment is converted
    # to kJ/mol for the fitter. Sanity: evaluating source G from these rows
    # must match an independent evaluation of the published NASA card.
    reference_j = float(record["delta_f_H_298_15"]["value"])
    source_rows = {
        temperature: {
            "temperature": temperature,
            "heat_capacity": R_J_MOL_K * properties["cp_R"],
            "entropy": R_J_MOL_K * properties["s_R"],
            "enthalpy_increment": (
                R_J_MOL_K * temperature * properties["h_rt"] - reference_j
            ) / 1000.0,
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


def _janaf_condensate_nasa_tail_rows(
    janaf_source_dir: Path,
    nasa_source_dir: Path,
    janaf_table_id: str,
    nasa_table_id: str,
    *,
    nasa_formula: str = "PbO",
    anchor_temperature: float,
    fit_t_max: float = 3000.0,
) -> tuple[dict[float, dict[str, float]], float]:
    """Build the splice-anchored JANAF/NASA H/S rows used by the fit.

    Premise: the JANAF liquid table supplies complete thermal cells through
    ``anchor_temperature``, while the published CEA liquid card continues
    through ``fit_t_max``.
    Algebra: anchor the CEA H(T)-H(T0) and S(T)-S(T0) differences at the JANAF
    T0 cell, then use G_app=Hf_JANAF(298)+Hinc-T*S. Unit check: the enthalpy
    difference is converted from J/mol to kJ/mol before adding to JANAF's
    increment; entropy remains J/(mol K). NASA's condensed standard pressure
    is 1 atm while the JANAF anchor is 1 bar; no pressure correction is
    claimed without a sourced liquid molar volume. Sanity: Hinc, S, and G
    match the JANAF cell at T0 and are continuous there.
    """
    janaf_rows = _janaf_text_source_rows(janaf_source_dir, janaf_table_id)
    reference = janaf_rows.get(298.15, {}).get("formation_enthalpy")
    anchor = janaf_rows.get(anchor_temperature)
    if reference is None or anchor is None:
        raise ValueError(f"{janaf_table_id}: missing JANAF anchor data")
    record = _load_record(nasa_source_dir / f"{nasa_table_id}.json")
    if record.get("phase") != "liquid" or record.get("formula") != nasa_formula:
        raise ValueError(
            f"{nasa_table_id}: expected a {nasa_formula} liquid NASA card"
        )
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
    return janaf_rows, float(reference)


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
    nasa_formula: str = "PbO",
    anchor_temperature: float,
    fit_t_min: float = 1200.0,
    fit_t_max: float = 3000.0,
    runtime_t_min: float | None = None,
    runtime_t_max: float | None = None,
) -> dict[str, str]:
    janaf_rows, reference = _janaf_condensate_nasa_tail_rows(
        janaf_source_dir,
        nasa_source_dir,
        janaf_table_id,
        nasa_table_id,
        nasa_formula=nasa_formula,
        anchor_temperature=anchor_temperature,
        fit_t_max=fit_t_max,
    )
    return _fit_condensate_values(
        species_name,
        janaf_table_id,
        cation,
        cat_num,
        oxy_num,
        janaf_rows,
        reference,
        fit_t_min=fit_t_min,
        fit_t_max=fit_t_max,
        runtime_t_min=fit_t_min if runtime_t_min is None else runtime_t_min,
        runtime_t_max=fit_t_max if runtime_t_max is None else runtime_t_max,
        allow_missing_nodes=janaf_table_id == "Cu-020",
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
    phase: str = "l",
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
    _validate_record_identity(record, species_name, table_id, phase)
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
        "state": "s" if phase == "cr" else "l",
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
        "_source_rows": {
            float(row["temperature"]): {
                "enthalpy_increment": float(row["enthalpy_increment"]),
                "entropy": float(row["entropy"]),
            }
            for row in rows
        },
        "_reference_kJ": float(reference),
        "_fit_t_min": float(min(temperatures)),
        "_fit_t_max": float(max(temperatures)),
        "_fit_temperatures": sorted(float(temperature) for temperature in temperatures),
        "_allow_missing_nodes": False,
        "_fit_table_id": table_id,
    }


def _fit_janaf_parent_liquid(
    source_dir: Path,
    species_name: str,
    table_id: str,
    cation: str,
    cat_num: int,
    oxy_num: int,
    intervals: tuple[tuple[float, float, tuple[float, ...] | None, bool], ...],
) -> list[dict[str, str]]:
    """Fit printed liquid nodes and any explicitly requested Cp continuation.

    Each interval declares its source nodes; only intervals flagged as
    continuation receive generated constant-Cp nodes. The independent fit
    residual remains a check against whichever nodes generated that interval.
    """
    record = _load_record(source_dir / f"{table_id}.yaml")
    table = record["table"]
    _validate_record_identity(record, species_name, table_id, "l")
    source_rows = {
        float(_value(row, "temperature")): {
            "temperature": float(_value(row, "temperature")),
            "heat_capacity": float(_value(row, "heat_capacity")),
            "entropy": float(_value(row, "entropy")),
            "enthalpy_increment": float(_value(row, "enthalpy_increment")),
        }
        for row in table["values"]
        if _value(row, "temperature") is not None
        and all(
            _value(row, field) is not None
            for field in ("heat_capacity", "entropy", "enthalpy_increment")
        )
    }
    if table_id in {"Ba-035", "O-014", "O-053"}:
        # The official tables print thermal cells for the 2300 K BaO row and
        # the 3000 K SrO/ZrO2 rows, but concatenate formation columns into a
        # malformed final field. Preserve those printed thermal nodes using
        # the existing recovery rule; the 298 K formation anchor stays pinned
        # to its separately parsed JANAF cell.
        source_rows.update(
            {
                float(row["temperature"]): row
                for row in _recover_thermal_transition_rows(table)
            }
        )
    reference = next(
        _value(row, "formation_enthalpy")
        for row in table["values"]
        if _value(row, "temperature") == NASA_STANDARD_T_K
    )
    if reference is None:
        raise ValueError(f"{table_id}: no complete 298.15 K formation enthalpy")
    source_limit = max(source_rows)
    fitted_rows = []
    for fit_min, fit_max, explicit_nodes, continuation in intervals:
        nodes = (
            _constant_cp_rows_through(source_rows, fit_max)
            if continuation
            else source_rows
        )
        selected = {
            temperature: row
            for temperature, row in nodes.items()
            if fit_min <= temperature <= fit_max
        }
        row = _fit_condensate_values(
            species_name,
            table_id,
            cation,
            cat_num,
            oxy_num,
            selected,
            float(reference),
            fit_t_min=fit_min,
            fit_t_max=fit_max,
            runtime_t_min=fit_min,
            runtime_t_max=fit_max,
            ref=(f"{table_id}-CP-EXT-{source_limit:g}" if continuation else table_id),
            fit_temperatures=explicit_nodes,
        )
        if float(row["_max_residual_J_per_mol"]) > 10.0:
            raise ValueError(
                f"{species_name}: condensate fit residual "
                f"{row['_max_residual_J_per_mol']} J/mol exceeds 10"
            )
        fitted_rows.append(row)
    return fitted_rows


def _shared_interval_bounds(
    intervals: Sequence[tuple[float, float]],
) -> tuple[float, ...]:
    lows = {float(low) for low, _high in intervals}
    highs = {float(high) for _low, high in intervals}
    return tuple(sorted(lows & highs))


def _fit_supercooled_liquid_row(
    source_dir: Path,
    species_name: str,
    table_id: str,
    cation: str,
    cat_num: int,
    oxy_num: int,
    T0: float,
    runtime_t_max: float,
    *,
    runtime_t_min: float = 1200.0,
    upper_g_cp: tuple[float, float] | None = None,
    upper_g_s: tuple[float, float] | None = None,
    upper_cp: float | None = None,
    nasa_source_dir: Path | None = None,
    shared_seams: Sequence[float] = (),
) -> dict[str, str]:
    """Fit the generated constant-Cp continuation to the condensate form.

    Premise: a JANAF row or NASA card supplies the liquid enthalpy increment,
    entropy, and Cp at T0; NASA Hinc uses its crystal dfH298 anchor plus
    H(T0)-H(298). For the labelled continuation,
    H(T)=H(T0)+Cp*(T-T0) and S(T)=S(T0)+Cp*ln(T/T0), then
    G_app=dfH298+[H(T)-H298]-T*S(T).  The fitted polynomial represents this
    generated function; it is not source data below the liquid branch.

    Unit check: Cp*(T-T0) is J/mol and is divided by 1000 for JANAF's
    kJ/mol enthalpy increment; Cp*ln(T/T0) is J/(mol K); both terms in G
    are J/mol.  The condensate fit matches Phi/R with Phi=S-(H-H298)/T.
    Sanity: the analytic continuation has the source H, S, G, and -dG/dT
    exactly at T0.  At source nodes below a glass/liquid marker, differences
    from the source's printed glass rows are expected because this deliberately
    continues the liquid rather than the glass.  Any genuine supercooled
    liquid nodes in the JANAF table are checked by the caller's regression
    tests.
    Premise: a JANAF row supplies the first complete liquid H/S/Cp node; for a
    NASA card, T0 is its first supported liquid node and Hinc is H_card(T0)
    minus the card's dfH298. NASA's H(T)/(R*T) polynomial already includes
    b1 and returns the assigned element-reference H_card(T), so the card's
    apparent Gibbs energy is H_card(T)-T*S_card(T), without extrapolating a
    liquid H(298) anchor. Algebra below T0 is
    H(T)=H(T0)+Cp*(T-T0), S(T)=S(T0)+Cp*ln(T/T0), and
    G_app=dfH298+Hinc-T*S. Unit check: Cp*(T-T0) is J/mol and becomes kJ/mol
    in the source-row fitter; Cp*ln(T/T0) is J/(mol K); G remains J/mol.
    Sanity: the continuation matches the supported liquid's H, S, G, and
    -dG/dT exactly at T0; it makes no claim about a melting point below that
    source node.
    """
    yaml_path = source_dir / f"{table_id}.yaml"
    if nasa_source_dir is not None:
        record = _nasa_liquid_record(nasa_source_dir, species_name, table_id)
        formula = species_name.removesuffix("(l)")
        if (
            record.get("record_id") != table_id
            or record.get("formula") != formula
            or record.get("phase") not in {"liquid", "L"}
        ):
            raise ValueError(f"{table_id}: NASA liquid record does not match {formula}")
        properties = _nasa7_properties(record, T0)
        reference_j = float(record["delta_f_H_298_15"]["value"])
        cp = R_J_MOL_K * properties["cp_R"]
        entropy_0 = R_J_MOL_K * properties["s_R"]
        enthalpy_0 = (
            R_J_MOL_K * T0 * properties["h_rt"] - reference_j
        ) / 1000.0
        reference = reference_j / 1000.0
    elif yaml_path.is_file():
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
    elif (source_dir / f"{table_id}.txt").is_file():
        parsed = _janaf_text_source_rows(source_dir, table_id)
        source_row = parsed.get(T0)
        if source_row is None:
            raise ValueError(f"{table_id}: no complete liquid anchor at {T0:g} K")
        cp = source_row["heat_capacity"]
        entropy_0 = source_row["entropy"]
        enthalpy_0 = source_row["enthalpy_increment"]
        reference = parsed.get(298.15, {}).get("formation_enthalpy")
    else:
        nasa_source_dir = nasa_source_dir or (
            Path(__file__).resolve().parents[1] / "data-src/nasa-glenn"
        )
        record = _nasa_liquid_record(nasa_source_dir, species_name, table_id)
        if (
            record.get("record_id") != table_id
            or record.get("formula") != species_name.removesuffix("(l)")
            or record.get("phase") != "liquid"
        ):
            raise ValueError(f"{table_id}: expected a matching NASA liquid card")
        properties = _nasa7_properties(record, T0)
        cp = R_J_MOL_K * properties["cp_R"]
        entropy_0 = R_J_MOL_K * properties["s_R"]
        reference = float(record["delta_f_H_298_15"]["value"]) / 1000.0
        # Store Hinc relative to dfH298: dfH298+Hinc=H_card(T0), the
        # element-reference enthalpy returned by the NASA-9 H/RT polynomial.
        enthalpy_0 = (
            R_J_MOL_K * T0 * properties["h_rt"] / 1000.0 - reference
        )
    if reference is None:
        raise ValueError(f"{table_id} has no complete 298.15 K formation enthalpy")

    use_cp_constrained_fit = species_name in {
        "BeO(l)",
        "TiO2(l)",
        "Cr2O3(l)",
        "V2O3(l)",
        "SiO2(l)",
        "Al2O3(l)",
        "MgO(l)",
        "CaO(l)",
        "SnO(l)",
        "GeO2(l)",
        "Ga2O3(l)",
        "In2O3(l)",
    }
    if use_cp_constrained_fit:
        # Premise: each runtime interval follows the same analytic constant-Cp
        # continuation, and the shared upper-temperature seam must reproduce
        # its G and S. Algebra: set P(tau)=phi/R and P+tau*P'=S/R there while
        # least-squares fitting Cp/R = 2*tau*P' + tau**2*P''. Unit check: P,
        # Cp/R and both constraints are dimensionless. Sanity: G and S meet
        # the analytic continuation at the interval's upper endpoint.
        fit_temperatures = np.linspace(
            runtime_t_min,
            runtime_t_max,
            max(5, int(runtime_t_max - runtime_t_min) + 1),
        ).tolist()
    else:
        # Keep the established fit nodes for other continuation rows.
        fit_temperatures = sorted(
            {runtime_t_min, runtime_t_max}
            | set(
                float(T)
                for T in range(int(runtime_t_min), int(runtime_t_max) + 1, 100)
            )
        )

    def phi(temperature: float) -> float:
        enthalpy_increment = enthalpy_0 + cp * (temperature - T0) / 1000.0
        entropy = entropy_0 + cp * math.log(temperature / T0)
        return entropy - enthalpy_increment * 1000.0 / temperature

    def analytic_g_s(temperature: float) -> tuple[float, float]:
        enthalpy_increment = enthalpy_0 + cp * (temperature - T0) / 1000.0
        entropy = entropy_0 + cp * math.log(temperature / T0)
        gibbs = (float(reference) + enthalpy_increment) * 1000.0 - temperature * entropy
        return gibbs, entropy

    shared_here = [
        float(temperature)
        for temperature in shared_seams
        if runtime_t_min - 1e-6 <= float(temperature) <= runtime_t_max + 1e-6
    ]
    # The 1000-1200 K oxide row copies the 1200 K row, so that row keeps its
    # existing upper pin and does not take a second constraint at 1200 K.
    skip_lower_copy = (
        species_name in {"SiO2(l)", "Al2O3(l)", "MgO(l)", "CaO(l)"}
        and abs(runtime_t_min - 1200.0) <= 1e-6
    )
    if use_cp_constrained_fit:
        lower_seams = [
            temperature
            for temperature in shared_here
            if abs(temperature - runtime_t_max) > 1e-6
            and not (skip_lower_copy and abs(temperature - runtime_t_min) <= 1e-6)
        ]
    else:
        # Unconstrained continuations keep the packaged polynomial. The
        # same-Ref pass closes their joins without a second least-squares fit.
        lower_seams = []

    tau = np.asarray(fit_temperatures, dtype=float) / 1000.0
    design = np.column_stack((np.ones_like(tau), tau, tau**2, tau**3, tau**4))
    if use_cp_constrained_fit:
        heat_capacity_design = np.column_stack(
            (2.0 * tau, 6.0 * tau**2, 12.0 * tau**3, 20.0 * tau**4)
        )
        anchor_temperature = runtime_t_max
        anchor_tau = anchor_temperature / 1000.0
        if upper_g_cp is None:
            anchor_g, anchor_entropy = (
                upper_g_s
                if upper_g_s is not None
                else (
                    (reference + enthalpy_0 + cp * (anchor_temperature - T0) / 1000.0)
                    * 1000.0
                    - anchor_temperature
                    * (entropy_0 + cp * math.log(anchor_temperature / T0)),
                    entropy_0 + cp * math.log(anchor_temperature / T0),
                )
            )
            anchor_design = np.asarray(
                [
                    [1.0, anchor_tau, anchor_tau**2, anchor_tau**3, anchor_tau**4],
                    [
                        1.0,
                        2.0 * anchor_tau,
                        3.0 * anchor_tau**2,
                        4.0 * anchor_tau**3,
                        5.0 * anchor_tau**4,
                    ],
                ]
            )
            anchor_phi = (1000.0 * reference - anchor_g) / (
                R_J_MOL_K * anchor_temperature
            )
            constraints = [anchor_design[1, 1:] - anchor_design[0, 1:]]
            constraint_values = [anchor_entropy / R_J_MOL_K - anchor_phi]
            if upper_cp is not None:
                constraints.append(heat_capacity_design[-1])
                constraint_values.append(upper_cp / R_J_MOL_K)
            for seam_temperature in lower_seams:
                seam_tau = seam_temperature / 1000.0
                seam_g, seam_s = analytic_g_s(seam_temperature)
                seam_phi = (1000.0 * float(reference) - seam_g) / (
                    R_J_MOL_K * seam_temperature
                )
                constraints.append([
                    seam_tau - anchor_tau,
                    seam_tau**2 - anchor_tau**2,
                    seam_tau**3 - anchor_tau**3,
                    seam_tau**4 - anchor_tau**4,
                ])
                constraint_values.append(seam_phi - anchor_phi)
                constraints.append([
                    2.0 * seam_tau - anchor_tau,
                    3.0 * seam_tau**2 - anchor_tau**2,
                    4.0 * seam_tau**3 - anchor_tau**3,
                    5.0 * seam_tau**4 - anchor_tau**4,
                ])
                constraint_values.append(seam_s / R_J_MOL_K - anchor_phi)
            constraints = np.asarray(constraints)
            # G=-RT*P gives S/R=P+tau*P'. The constraint matches Gibbs
            # energy and entropy at this interval's upper seam.
            normal = heat_capacity_design.T @ heat_capacity_design
            target = heat_capacity_design.T @ np.full(len(tau), cp / R_J_MOL_K)
            kkt = np.block(
                [
                    [normal, constraints.T],
                    [constraints, np.zeros((len(constraint_values), len(constraint_values)))],
                ]
            )
            coefficients_tail = np.linalg.solve(
                kkt, np.concatenate((target, np.asarray(constraint_values)))
            )[:4]
            coefficients = np.concatenate(
                ([anchor_phi - anchor_tau * coefficients_tail[0]
                  - anchor_tau**2 * coefficients_tail[1]
                  - anchor_tau**3 * coefficients_tail[2]
                  - anchor_tau**4 * coefficients_tail[3]], coefficients_tail)
            )
        else:
            # The adjacent packaged interval owns this seam. Minimize the
            # source G residual directly, subject to exact endpoint G and Cp;
            # the two constraints are P(tau) and Cp/R=2*tau*P'+tau^2*P''.
            # G is J/mol, Cp is J/(mol K), and P is dimensionless.
            anchor_g, anchor_cp = upper_g_cp
            source_g = np.asarray([
                (reference + enthalpy_0 + cp * (temperature - T0) / 1000.0)
                * 1000.0
                - temperature * (entropy_0 + cp * math.log(temperature / T0))
                for temperature in fit_temperatures
            ])
            gibbs_design = -R_J_MOL_K * np.asarray(fit_temperatures)[:, None] * design
            constraints = np.asarray(
                [
                    [anchor_tau**power for power in range(5)],
                    [0.0, 2.0 * anchor_tau, 6.0 * anchor_tau**2,
                     12.0 * anchor_tau**3, 20.0 * anchor_tau**4],
                ]
            )
            constraint_values = np.asarray(
                [
                    (1000.0 * reference - anchor_g)
                    / (R_J_MOL_K * anchor_temperature),
                    anchor_cp / R_J_MOL_K,
                ]
            )
            normal = gibbs_design.T @ gibbs_design
            target = gibbs_design.T @ (source_g - 1000.0 * reference)
            kkt = np.block(
                [
                    [normal, constraints.T],
                    [constraints, np.zeros((len(constraint_values), len(constraint_values)))],
                ]
            )
            coefficients = np.linalg.solve(
                kkt, np.concatenate((target, constraint_values))
            )[:5]
    else:
        coefficients = np.linalg.lstsq(
            design, np.asarray([phi(T) for T in fit_temperatures]) / R_J_MOL_K,
            rcond=None,
        )[0]
    sample_temperatures = np.linspace(
        runtime_t_min,
        runtime_t_max,
        max(2, int(runtime_t_max - runtime_t_min) + 1),
    )
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
    implied_cp = R_J_MOL_K * (
        2.0 * coefficients[1] * sample_tau
        + 6.0 * coefficients[2] * sample_tau**2
        + 12.0 * coefficients[3] * sample_tau**3
        + 20.0 * coefficients[4] * sample_tau**4
    )
    max_abs_cp_deviation = float(np.max(np.abs(implied_cp - cp)))
    anchor_h = (reference + enthalpy_0) * 1000.0
    anchor_g = anchor_h - T0 * entropy_0

    def number(value: float) -> str:
        return format(float(value), ".15g")

    A, B, C, D, E = coefficients
    # Keep the established runtime Ref while the vendor record ID selects
    # BeO's distinct NASA file instead of the Rb2O card.
    runtime_source_id = (
        "NG-1352" if table_id == "NG-BeO-L-1352" else table_id
    )
    return {
        "species_name": species_name,
        "state": "l",
        "cation": cation,
        "cat_num": str(cat_num),
        "oxy_num": str(oxy_num),
        "T_min": number(runtime_t_min),
        "T_max": number(runtime_t_max),
        "dH298_R": number(float(reference) / R_J_MOL_K),
        "dG_A": number(A),
        "dG_B": number(B),
        "dG_C": number(C),
        "dG_D": number(D),
        "dG_E": number(E),
        "Ref": runtime_source_id + SUPERCOOLED_LIQUID_REF_SUFFIX,
        "_max_residual_J_per_mol": number(residual_j),
        "_max_residual_log10_K": number(residual_log10),
        "_T0_K": number(T0),
        "_Cp_l_J_molK": number(cp),
        "_max_abs_Cp_deviation_J_molK": number(max_abs_cp_deviation),
        "_anchor_H_J_per_mol": number(anchor_h),
        "_anchor_S_J_molK": number(entropy_0),
        "_anchor_G_J_per_mol": number(anchor_g),
        **(
            {"_seam_constrained": "true"}
            if lower_seams
            else {}
        ),
    }


def build_rows(
    source_dir: Path,
    nasa_source_dir: Path | None = None,
    lh84_source_dir: Path | None = None,
) -> list[dict[str, str]]:
    repository = Path(__file__).resolve().parents[1]
    nasa_source_dir = nasa_source_dir or repository / "data-src/nasa-glenn"
    lh84_source_dir = lh84_source_dir or repository / "data-src/lh84"
    published_source_dir = repository / "data-src/published-thermo"
    konings_records = {
        element: _load_record(KONINGS_SOURCE_DIR / f"{element.lower()}.json")
        for element in KONINGS_BATCH_ELEMENTS
        if (KONINGS_SOURCE_DIR / f"{element.lower()}.json").is_file()
    }
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
    # Konings Part 1/2 are polynomial records rather than tabulated nodes;
    # integrate their Cp functions to the shared 100 K node grid before using
    # the established Shomate fitter. The low fit is exposed only from 1200 K.
    for interval, lo, hi, runtime_lo in (
        (1, 1500.0, 3000.0, None), (2, 500.0, 1500.0, 1200.0)
    ):
        for element, species_rows in KONINGS_GAS_SPECIES.items():
            if element not in konings_records:
                continue
            rows.extend(
                _fit_konings_gas_row(
                    konings_records[element], species, element, 1, oxy_num,
                    interval=interval,
                    fit_t_min=(
                        KONINGS_LOW_FIT_START_K.get(species, lo)
                        if interval == 2 else lo
                    ),
                    fit_t_max=hi,
                    runtime_t_min=runtime_lo,
                )
                for species, oxy_num in species_rows
            )
    rows.extend(
        _fit_row(
            source_dir,
            *source,
            t_interval=1,
            fit_t_min=FIT_T_MIN,
            fit_t_max=FIT_T_MAX,
        )
        for source in TRACE_JANAF_YAML_GAS_SOURCES
        if not source[0].startswith(("Mo", "W"))
        and source[1] not in HALOGEN_JANAF_TABLES
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
        if source[1] not in HALOGEN_NASA_TABLES
    )
    for source in TRACE_PUBLISHED_GAS_SOURCES:
        if source[1] in {"NUREG-4308-TeO", "Kelley-584-SbO"}:
            continue
        split_temperature = (
            min(max(source[6], 1500.0), 2000.0)
            if source[1].startswith(
                ("P672-Te", "P672-Se", "P672-As", "P672-Sb", "P672-Tl")
            )
            else (2000.0 if source[1] in {"P672-Rh-g", "P677-Re-g"} else 1500.0)
        )
        high_fit_start = math.floor(split_temperature / 100.0) * 100.0
        fitted = _fit_published_gas_row(
                published_source_dir,
                *source[:6],
                t_interval=1,
                fit_t_min=high_fit_start,
                fit_t_max=FIT_T_MAX,
                runtime_t_min=split_temperature,
                runtime_t_max=FIT_T_MAX,
                breakpoint_side="high",
            )
        if source[1] in {"P672-Rh-g", "P677-Re-g"}:
            fitted["Ref"] = f"{source[1]}-CONT-{source[6]:g}"
        rows.append(fitted)
    rows.extend(
        _fit_row(
            source_dir,
            *source,
            t_interval=1,
            fit_t_min=FIT_T_MIN,
            fit_t_max=FIT_T_MAX,
        )
        for source in TRACE_JANAF_YAML_GAS_SOURCES
        if source[0].startswith(("Mo", "W"))
    )
    rows.extend(
        _fit_janaf_text_row(
            source_dir,
            *source,
            t_interval=2,
            fit_t_min=LOW_FIT_T_MIN,
            fit_t_max=LOW_FIT_T_MAX,
            runtime_t_min=(
                1000.0 if source[0] in {"Zn(g)", "Pb(g)", "PbO(g)"}
                else 1200.0
            ),
        )
        for source in TRACE_JANAF_GAS_SOURCES
    )
    for source in TRACE_JANAF_YAML_GAS_SOURCES:
        if source[0].startswith(("Mo", "W")) or source[1] in HALOGEN_JANAF_TABLES:
            continue
        low_row = _fit_row(
            source_dir,
            *source,
            t_interval=2,
            fit_t_min=LOW_FIT_T_MIN,
            fit_t_max=LOW_FIT_T_MAX,
            minimum_rows=11,
        )
        low_row["T_min"] = "1000" if source[0] == "H2O(g)" else "1200"
        rows.append(low_row)
    rows.extend(
        _fit_nasa_card_gas_row(
            nasa_source_dir,
            *source,
            t_interval=2,
            fit_t_min=LOW_FIT_T_MIN,
            fit_t_max=LOW_FIT_T_MAX,
            runtime_t_min=(
                1000.0 if source[0] in {"Cd(g)", "PbO2(g)"}
                else 1200.0
            ),
        )
        for source in TRACE_NASA_GAS_SOURCES
        if source[1] not in HALOGEN_NASA_TABLES
    )
    for source in TRACE_PUBLISHED_GAS_SOURCES:
        if source[1] in {"NUREG-4308-TeO", "Kelley-584-SbO"}:
            continue
        split_temperature = (
            min(max(source[6], 1500.0), 2000.0)
            if source[1].startswith(
                ("P672-Te", "P672-Se", "P672-As", "P672-Sb", "P672-Tl")
            )
            else (2000.0 if source[1] in {"P672-Rh-g", "P677-Re-g"} else 1500.0)
        )
        fitted = _fit_published_gas_row(
                published_source_dir,
                *source[:6],
                t_interval=2,
                fit_t_min=1200.0,
                fit_t_max=split_temperature,
                runtime_t_min=1200.0,
                runtime_t_max=split_temperature,
                breakpoint_side="low",
            )
        rows.append(fitted)
    rows.extend(_fit_re2o7_constructed_gas_rows(published_source_dir))
    for source in TRACE_JANAF_YAML_GAS_SOURCES:
        if not source[0].startswith(("Mo", "W")):
            continue
        low_row = _fit_row(
            source_dir,
            *source,
            t_interval=2,
            fit_t_min=LOW_FIT_T_MIN,
            fit_t_max=LOW_FIT_T_MAX,
            minimum_rows=11,
        )
        low_row["T_min"] = "1200"
        rows.append(low_row)
    rows.extend(
        _fit_janaf_text_row(source_dir, *source)
        for source in TRACE_ION_GAS_TEXT_SOURCES
        if source[0] not in DECLINED_ION_GAS_SPECIES
        and source[1] not in {"Zn-006", "Zn-007"}
    )
    rows.extend(
        _fit_nasa_card_gas_row(
            nasa_source_dir,
            *source,
            t_interval=1,
            fit_t_min=FIT_T_MIN,
            fit_t_max=FIT_T_MAX,
        )
        for source in TRACE_ION_GAS_NASA_SOURCES + TRACE_ION_NASA_SOURCES
    )
    rows.extend(
        _fit_nasa_card_gas_row(
            published_source_dir,
            *source,
            t_interval=1,
            fit_t_min=1200.0,
            fit_t_max=3000.0,
            runtime_t_min=1200.0,
        )
        for source in TRACE_ION_GAS_BURCAT_SOURCES
    )
    rows.extend(
        _fit_janaf_text_row(
            source_dir,
            *source,
            t_interval=1,
            fit_t_min=FIT_T_MIN,
            fit_t_max=FIT_T_MAX,
        )
        for source in TRACE_ION_GAS_TEXT_SOURCES
        if source[1] in {"Zn-006", "Zn-007"}
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
        for source in TRACE_ION_GAS_TEXT_SOURCES
        if source[1] in {"Zn-006", "Zn-007"}
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
        for source in TRACE_ION_GAS_NASA_SOURCES + TRACE_ION_NASA_SOURCES
    )
    rows.extend(
        _fit_row(
            source_dir,
            *source,
            t_interval=1,
            fit_t_min=FIT_T_MIN,
            fit_t_max=FIT_T_MAX,
        )
        for source in TRACE_ION_GAS_YAML_SOURCES
    )
    for source in TRACE_ION_GAS_YAML_SOURCES:
        low_row = _fit_row(
            source_dir,
            *source,
            t_interval=2,
            fit_t_min=1000.0,
            fit_t_max=1500.0,
            minimum_rows=6,
        )
        low_row["T_min"] = "1200"
        rows.append(low_row)
    rows.extend(
        _fit_bi_plus_row(
            published_source_dir,
            t_interval=interval,
            fit_t_min=fit_t_min,
            fit_t_max=fit_t_max,
            runtime_t_min=runtime_t_min,
        )
        for interval, fit_t_min, fit_t_max, runtime_t_min in (
            (1, FIT_T_MIN, FIT_T_MAX, FIT_T_MIN),
            (2, 1200.0, 1500.0, 1200.0),
        )
    )
    rows.extend(
        _fit_row(
            source_dir,
            *source,
            fit_t_min=1200.0,
            fit_t_max=3000.0,
            minimum_rows=19,
        )
        for source in TRACE_JANAF_YAML_GAS_SOURCES
        if source[1] in HALOGEN_JANAF_TABLES
    )
    # These JANAF halogen rows have source nodes from 298 K onward. Fit a
    # distinct 1000-1200 K low interval from the 1000-1500 K source cells;
    # the existing 1200-3000 K row remains byte-identical above the seam.
    for source in TRACE_JANAF_YAML_GAS_SOURCES:
        if source[0] not in {
            "HCl(g)", "HF(g)", "NaCl(g)", "(NaCl)2(g)", "KCl(g)",
            "(KCl)2(g)", "NaF(g)", "(NaF)2(g)", "KF(g)", "(KF)2(g)",
        }:
            continue
        low_row = _fit_row(
            source_dir,
            *source,
            t_interval=2,
            fit_t_min=1000.0,
            fit_t_max=1500.0,
            minimum_rows=6,
        )
        low_row["T_max"] = "1200"
        rows.append(low_row)
    rows.extend(
        _fit_nasa_card_gas_row(
            nasa_source_dir,
            *source,
            fit_t_min=1200.0,
            fit_t_max=3000.0,
        )
        for source in TRACE_NASA_GAS_SOURCES
        if source[1] in HALOGEN_NASA_TABLES
    )
    for species_name, filename, record_key, cation, cat_num, oxy_num in BURCAT_METAL_GAS_SOURCES:
        intervals = (
            ((1, 1500.0, 1900.0), (2, 1200.0, 1500.0), (3, 1900.0, 3000.0))
            if species_name == "PtO2(g)"
            else (
                ((1, 1500.0, 3000.0), (2, 1200.0, 1500.0), (3, 1000.0, 1500.0))
                if species_name == "Hg(g)"
                else ((1, 1500.0, 3000.0), (2, 1200.0, 1500.0))
            )
        )
        for interval, fit_min, fit_max in intervals:
            row = _fit_burcat_metal_gas_row(
                species_name, filename, record_key, cation, cat_num, oxy_num,
                interval=interval, fit_t_min=fit_min, fit_t_max=fit_max,
            )
            if species_name == "Hg(g)" and interval == 3:
                row["T_max"] = "1200"
            if species_name == "PtO2(g)" and interval == 3:
                row["Ref"] = f"{row['Ref']}-CONT-2000"
            rows.append(row)
    rows.extend(
        _fit_constructed_saha_ion_row(
            element,
            published_source_dir,
            rows,
            t_interval=interval,
            fit_t_min=fit_t_min,
            fit_t_max=fit_t_max,
            runtime_t_min=runtime_t_min,
        )
        for element in CONSTRUCTED_SAHA_ION_ELEMENTS
        for interval, fit_t_min, fit_t_max, runtime_t_min in (
            (1, 1500.0, 3000.0, 1500.0),
            (2, 1200.0, 1500.0, 1200.0),
        )
    )
    rows.extend(_fit_nureg_teo_rows(published_source_dir, rows))
    rows.extend(_fit_kelley_sbo_rows(published_source_dir, rows))
    return _pin_gas_same_ref_seams(rows)


def _cd_o_crystal_table(record: dict[str, Any], *, extrapolate_cp: bool) -> dict[str, Any]:
    """Build the CdO crystal table, extending its published Cp equation if asked.

    Premise: Pankratz Table 102 gives a CdO crystal Cp equation through 1500 K
    and printed H/S/Cp there. For the required hypothetical fusion range, the
    same equation may be extended, with a flag. Algebra: integrate
    Cp=a+bT-c/T² from 1500 K; Hinc adds aΔT+bΔ(T²)/2+c(1/T-1/T0), and S adds
    a ln(T/T0)+bΔT+c(1/T²-1/T0²)/2. Unit check: the source equation is in
    cal/(mol K), so both integrals are converted with 4.184; Hinc is then
    divided by 1000 to kJ/mol. Sanity: at 1500 K the generated anchor is the
    printed table node; each extension node is the analytic equation value.
    """
    equation = record["cp_equation_cal_mol_K"]
    a = float(equation["a"])
    b = float(equation["b_per_K"])
    c = float(equation["c_K2"])
    t0 = float(record["last_printed_node"][0])
    cp0_cal = float(record["last_printed_node"][1])
    s0_cal = float(record["last_printed_node"][2])
    h0_kcal = float(record["last_printed_node"][3])
    tfus = 1974.1
    values = [
        {
            "temperature": 298.15,
            "heat_capacity": (a + b * 298.15 - c / 298.15**2) * 4.184,
            "entropy": float(record["entropy_298_cal_mol_K"]) * 4.184,
            "enthalpy_increment": 0.0,
            "formation_enthalpy": float(record["formation_enthalpy_298_kcal_mol"])
            * 4.184,
        },
        {
            "temperature": t0,
            "heat_capacity": cp0_cal * 4.184,
            "entropy": s0_cal * 4.184,
            "enthalpy_increment": h0_kcal * 4.184,
        },
    ]
    extension_temperatures = sorted(
        {T for T in (1700.0, *range(1600, 2000, 100), tfus) if T > t0}
    )
    for temperature in extension_temperatures:
        delta_t = temperature - t0
        if extrapolate_cp:
            cp_cal = a + b * temperature - c / temperature**2
            delta_h_cal = (
                a * delta_t
                + 0.5 * b * (temperature**2 - t0**2)
                + c * (1.0 / temperature - 1.0 / t0)
            )
            delta_s_cal = (
                a * math.log(temperature / t0)
                + b * delta_t
                + 0.5 * c * (1.0 / temperature**2 - 1.0 / t0**2)
            )
        else:
            cp_cal = cp0_cal
            delta_h_cal = cp_cal * delta_t
            delta_s_cal = cp_cal * math.log(temperature / t0)
        values.append(
            {
                "temperature": float(temperature),
                "heat_capacity": cp_cal * 4.184,
                "entropy": (s0_cal + delta_s_cal) * 4.184,
                "enthalpy_increment": (h0_kcal + delta_h_cal / 1000.0) * 4.184,
            }
        )
    return {"table": {"values": values}}


def _pankratz_crystal_table(
    record: dict[str, Any], *, extension_limit_k: float, extrapolate_cp: bool = True
) -> dict[str, Any]:
    """Convert a Pankratz crystal record and extend its Cp equation to fusion.

    Premise: the printed crystal nodes anchor H, S, and Cp, while the source's
    own Cp equation continues through its last declared range and may be
    continued to the hypothetical fusion marker under the task's rule.
    Algebra: integrate Cp=a+bT-c/T² from the final node: dH=aΔT+bΔ(T²)/2+
    c(1/T-1/T0), dS=a ln(T/T0)+bΔT+c(1/T²-1/T0²)/2. Unit check: the
    source equation is cal/(mol K); convert H and Cp by 4.184 and H to kJ.
    Sanity: the extension begins at the source's final H/S/Cp node.
    """
    source_limit = record.get("source_limit_K")
    source_nodes = record["nodes"]
    if source_limit is not None:
        source_nodes = [
            node for node in source_nodes if float(node[0]) <= float(source_limit)
        ]
    values = [
        {
            "temperature": float(t),
            "heat_capacity": float(cp) * 4.184,
            "entropy": float(entropy) * 4.184,
            "enthalpy_increment": float(h) * 4.184,
            **(
                {"formation_enthalpy": float(record["formation_enthalpy_298_kcal_mol"]) * 4.184}
                if float(t) == 298.15
                else {}
            ),
        }
        for t, cp, entropy, h in source_nodes
    ]
    t0, cp0, s0, h0 = values[-1]["temperature"], None, None, None
    cp0 = float(source_nodes[-1][1])
    s0 = float(source_nodes[-1][2])
    h0 = float(source_nodes[-1][3])
    equation = record["cp_equation_cal_mol_K"]
    a, b, c = float(equation["a"]), float(equation["b_per_K"]), float(equation["c_K2"])
    t = float(extension_limit_k)
    dt = t - t0
    if extrapolate_cp:
        cp = a + b * t - c / t**2
        dh = a * dt + 0.5 * b * (t**2 - t0**2) + c * (1.0 / t - 1.0 / t0)
        ds = a * math.log(t / t0) + b * dt + 0.5 * c * (1.0 / t**2 - 1.0 / t0**2)
    else:
        cp = cp0
        dh = cp * dt
        ds = cp * math.log(t / t0)
    if t > t0:
        values.append(
            {
                "temperature": t,
                "heat_capacity": cp * 4.184,
                "entropy": (s0 + ds) * 4.184,
                "enthalpy_increment": (h0 + dh / 1000.0) * 4.184,
            }
        )
    return {"table": {"values": values}}


def _tsiv_crystal_table(record: dict[str, Any]) -> dict[str, Any]:
    """Convert TSIV ZnO cells and extend fitted Cp to the upper fusion bound.

    Premise: TSIV prints ZnO crystal nodes through 2250 K and estimates the
    transition as 2250 +/- 30 K. Algebra: a quadratic Cp(T) fit to the final
    five crystal nodes is integrated from the printed endpoint to 2280 K,
    with H=integral(Cp dT) and S=integral(Cp/T dT). Unit check: Cp is
    J/(mol K), so H is converted to kJ/mol while S remains J/(mol K).
    Sanity: the extension starts at the final printed H/S/Cp node and the
    quadratic reproduces all five source Cp values within their rounding.
    """
    h298_minus_h0 = float(record["enthalpy_298_minus_H0_kj_mol"])
    values = []
    for temperature, cp, entropy, enthalpy_h0 in record["crystal_nodes"]:
        row = {
            "temperature": float(temperature),
            "heat_capacity": float(cp),
            "entropy": float(entropy),
            "enthalpy_increment": float(enthalpy_h0) - h298_minus_h0,
        }
        if float(temperature) == 298.15:
            row["formation_enthalpy"] = float(
                record["formation_enthalpy_298_kj_mol"]
            )
        values.append(row)
    source_nodes = np.asarray(record["crystal_nodes"][-5:], dtype=float)
    fit = np.polyfit(source_nodes[:, 0], source_nodes[:, 1], deg=2)
    extension_limit = float(record["fusion_temperature_K"]["value"])
    extension_limit += float(record["fusion_temperature_K"]["spread"])
    t0 = float(values[-1]["temperature"])
    h0 = float(values[-1]["enthalpy_increment"])
    s0 = float(values[-1]["entropy"])
    c2, b, a = (float(coefficient) for coefficient in fit)
    delta_h = (
        a * (extension_limit - t0)
        + 0.5 * b * (extension_limit**2 - t0**2)
        + (c2 / 3.0) * (extension_limit**3 - t0**3)
    )
    delta_s = (
        a * math.log(extension_limit / t0)
        + b * (extension_limit - t0)
        + 0.5 * c2 * (extension_limit**2 - t0**2)
    )
    values.append(
        {
            "temperature": extension_limit,
            "heat_capacity": float(np.polyval(fit, extension_limit)),
            "entropy": s0 + delta_s,
            "enthalpy_increment": h0 + delta_h / 1000.0,
        }
    )
    return {"table": {"values": values}}


def _fit_liquid_from_solid_parent(
    published_source_dir: Path,
    species: str,
    config: dict[str, Any],
) -> tuple[list[dict[str, str]], Any, Any | None]:
    """Fit one required tier-4 parent from its published crystal and inputs."""
    if __package__:
        from .liquid_from_solid import liquid_from_solid
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from tools.liquid_from_solid import liquid_from_solid

    if species == "ZnO":
        record = _published_table_record(published_source_dir, "TSIV-1603")
        crystal = _tsiv_crystal_table(record)
        alternate = None
    elif species == "CdO":
        record = _published_table_record(published_source_dir, "P672-T102")
        crystal = _cd_o_crystal_table(record, extrapolate_cp=True)
        alternate = _cd_o_crystal_table(record, extrapolate_cp=False)
    elif species == "MnO":
        crystals = _load_record(
            published_source_dir.parent / "crystal-thermochemistry.json"
        )["tables"]
        crystal = crystals[config["crystal_table_id"]]
        construction = liquid_from_solid(species, crystal, config["inputs"])
        fit = config["inputs"]["condensate_fit"]
        # The 1600-2200 K interval keeps the measured legacy node set. The
        # same-Ref seam pass then pins every MnO interval, including this one,
        # to the construction's G and S at each shared endpoint.
        intervals = ((1200.0, 1600.0), (1600.0, 2200.0), (2200.0, 3000.0))
        rows = []
        for fit_t_min, fit_t_max in intervals:
            if (fit_t_min, fit_t_max) == (
                float(fit["fit_t_min"]),
                float(fit["fit_t_max"]),
            ):
                row = construction.fit_condensate_row()
                row["T_min"] = str(int(fit_t_min))
                row["T_max"] = str(int(fit_t_max))
            else:
                temperatures = [
                    float(t)
                    for t in np.arange(
                        fit_t_min,
                        fit_t_max + JANAF_GRID_STEP_K / 2.0,
                        JANAF_GRID_STEP_K,
                    )
                ]
                row = _fit_condensate_values(
                    f"{species}(l)",
                    str(fit["table_id"]),
                    str(fit["cation"]),
                    int(fit["cat_num"]),
                    int(fit["oxy_num"]),
                    construction.source_rows(temperatures),
                    construction.dfh298_kj_mol,
                    fit_t_min=fit_t_min,
                    fit_t_max=fit_t_max,
                    runtime_t_min=fit_t_min,
                    runtime_t_max=fit_t_max,
                    ref=str(fit["ref"]),
                )
            residual = float(row["_max_residual_J_per_mol"])
            if residual > 10.0:
                raise ValueError(
                    f"MnO(l): condensate fit residual {residual:g} J/mol exceeds 10"
                )
            row["Ref"] = (
                f"LFS[{config['crystal_table_id']};Tf="
                f"{float(config['inputs']['fusion_temperature']['value']):g}]"
            )
            rows.append(row)
        return rows, construction, None
    elif species in {"HfO2", "ThO2"}:
        record = _published_table_record(published_source_dir, config["crystal_table_id"])
        extension_limit = max(
            float(value)
            for value in config["inputs"]["fusion_temperature"].get(
                "bounds_k", [config["inputs"]["fusion_temperature"]["value"]]
            )
        )
        # Premise: the HfO2/ThO2 melting markers are above 3000 K, so the
        # complete 1200-3000 K parent is a hypothetical supercooled liquid.
        # Algebra: liquid_from_solid
        # evaluates G_l(T)=G_c(Tfus)+(Tfus-T)*DeltaSfus+Cp_l*((T-Tfus)-T*ln(T/Tfus))
        # with its formation anchor restored; the crystal H/S at fusion comes
        # from the selected Pankratz oxide table. Unit check: each term is J/mol before conversion
        # to the fitted kJ/mol basis. Sanity: the fusion correction cancels at
        # Tfus, and no supercooling energy is used as an uncertainty band.
        # Pankratz prints the crystal equation only through the source limit. Continue
        # the crystal by its final Cp node; retain the printed-equation
        # extrapolation as the alternate used to calculate the continuation
        # band. The solid equation is J/mol-K after 4.184 conversion.
        crystal = _pankratz_crystal_table(
            record, extension_limit_k=extension_limit, extrapolate_cp=False
        )
        alternate = _pankratz_crystal_table(
            record, extension_limit_k=extension_limit, extrapolate_cp=True
        )
    elif species in {"SeO2", "As2O3", "Sb2O3", "Ag2O", "Y2O3"}:
        record = _published_table_record(published_source_dir, config["crystal_table_id"])
        crystal = _pankratz_crystal_table(
            record,
            extension_limit_k=max(
                float(value)
                for value in config["inputs"]["fusion_temperature"].get(
                    "bounds_k",
                    [config["inputs"]["fusion_temperature"]["value"]],
                )
            ),
        )
        alternate = None
    else:
        raise ValueError(f"{species}: no source-supported liquid-from-solid parent")
    inputs = {**config["inputs"], "condensate_fit": config["condensate_fit"]}
    construction = liquid_from_solid(species, crystal, inputs)
    if species in {"SeO2", "As2O3", "Sb2O3"}:
        alternate_crystal = _pankratz_crystal_table(
            record,
            extension_limit_k=max(
                float(value)
                for value in config["inputs"]["fusion_temperature"].get(
                    "bounds_k",
                    [config["inputs"]["fusion_temperature"]["value"]],
                )
            ),
            extrapolate_cp=False,
        )
        alternate = liquid_from_solid(species, alternate_crystal, inputs)
    intervals = {
        "ZnO": ((1200.0, 1700.0), (1700.0, 3000.0)),
        "CdO": ((1200.0, 1600.0), (1600.0, 3000.0)),
        "SeO2": ((1200.0, 1600.0), (1600.0, 3000.0)),
        "As2O3": ((1200.0, 1600.0), (1600.0, 3000.0)),
        # The tier-4 Cp carry-over is linear in G, but a single 1600-3000 K
        # Shomate interval misses the 10 J/mol source-node gate. These three
        # additional intervals fit the same construction, each below 10 J/mol.
        "Sb2O3": (
            (1200.0, 1600.0),
            (1600.0, 2000.0),
            (2000.0, 2400.0),
            (2400.0, 3000.0),
        ),
        "Ag2O": ((1200.0, 1600.0), (1600.0, 2000.0), (2000.0, 2400.0), (2400.0, 3000.0)),
        "ThO2": ((1200.0, 1600.0), (1600.0, 2000.0), (2000.0, 2400.0), (2400.0, 3000.0)),
        "HfO2": ((1200.0, 1600.0), (1600.0, 2000.0), (2000.0, 2400.0), (2400.0, 3000.0)),
        "Y2O3": ((1200.0, 1600.0), (1600.0, 2000.0), (2000.0, 2400.0), (2400.0, 3000.0)),
    }[species]
    fit = inputs["condensate_fit"]
    rows = []
    for fit_t_min, fit_t_max in intervals:
        temperatures = list(
            np.arange(fit_t_min, fit_t_max + JANAF_GRID_STEP_K / 2.0, JANAF_GRID_STEP_K)
        )
        rows.append(
            _fit_condensate_values(
                f"{species}(l)",
                str(fit["table_id"]),
                str(fit["cation"]),
                int(fit["cat_num"]),
                int(fit["oxy_num"]),
                construction.source_rows(temperatures),
                construction.dfh298_kj_mol,
                fit_t_min=fit_t_min,
                fit_t_max=fit_t_max,
                runtime_t_min=fit_t_min,
                runtime_t_max=fit_t_max,
                ref=str(fit["ref"]),
            )
        )
    return rows, construction, alternate


def _fit_published_condensate_row(
    table_id: str,
    species_name: str,
    cation: str,
    cat_num: int,
    oxy_num: int,
    published_source_dir: Path,
) -> list[dict[str, str]]:
    """Fit a published liquid branch plus a labelled constant-Cp upper tail."""
    record = _published_table_record(published_source_dir, table_id)
    if record.get("species") != species_name:
        raise ValueError(f"{table_id}: source liquid does not match {species_name}")
    rows = {
        float(temperature): {
            "temperature": float(temperature),
            "heat_capacity": float(cp) * 4.184,
            "entropy": float(entropy) * 4.184,
            "enthalpy_increment": float(enthalpy) * 4.184,
        }
        for temperature, cp, entropy, enthalpy in record["nodes"]
    }
    source_limit = max(rows)
    rows = _constant_cp_rows_through(rows, 3000.0)
    result = []
    intervals = (
        ((1200.0, source_limit), (source_limit, 3000.0))
        if table_id == "P672-TeO2-l"
        else ((1200.0, 1600.0), (1600.0, 3000.0))
    )
    for fit_t_min, fit_t_max in intervals:
        # The split keeps the on-node G residual below 10 J/mol over the
        # published liquid nodes and the labelled constant-Cp continuation.
        fit_rows = {
            temperature: value
            for temperature, value in rows.items()
            if fit_t_min <= temperature <= fit_t_max
        }
        result.append(
            _fit_condensate_values(
                species_name,
                table_id,
                cation,
                cat_num,
                oxy_num,
                fit_rows,
                float(record["formation_enthalpy_298_kcal_mol"]) * 4.184,
                fit_t_min=fit_t_min,
                fit_t_max=fit_t_max,
                runtime_t_min=fit_t_min,
                runtime_t_max=fit_t_max,
                ref=(
                    table_id
                    if table_id == "P672-TeO2-l" and fit_t_min < source_limit
                    else f"{table_id}-CP-EXT-{source_limit:g}"
                ),
            )
        )
    return result


def _tsiv_liquid_source_rows(record: dict[str, Any]) -> dict[float, dict[str, float]]:
    """Evaluate the printed TSIV constant-Cp liquid branch at source nodes.

    Premise: TSIV prints F°=-(G-H(0))/T and the liquid H-H(0) function in its
    table; the compact coefficients below are transcribed from that branch.
    Algebra: S=F+(H-H(0))/T, while H-H(298) is H-H(0) less the 298 K solid
    anchor. Units are J/(mol K) for F/S and kJ/mol for Hinc. Sanity: at Tm
    the liquid enthalpy is the solid value plus the printed fusion enthalpy.
    Below Tm this function is a labelled supercooled-liquid continuation.
    """
    c0, cp, inverse = (float(value) for value in record["liquid_F_coefficients"])
    h298_h0 = float(record["enthalpy_298_minus_H0_kj_mol"])
    dfh = float(record["formation_enthalpy_298_kj_mol"]["value"])
    fusion_temperature = float(record["fusion_temperature_K"]["value"])
    temperatures = sorted(
        {*map(float, range(1200, 3001, 100)), fusion_temperature}
    )
    rows = {}
    for temperature in temperatures:
        x = temperature * 1.0e-4
        F = c0 + cp * math.log(x) + inverse / x
        enthalpy_h0 = cp * temperature - inverse * 1.0e4
        rows[temperature] = {
            "temperature": temperature,
            "heat_capacity": cp,
            "entropy": F + enthalpy_h0 / temperature,
            "enthalpy_increment": enthalpy_h0 / 1000.0 - h298_h0,
            "formation_enthalpy": dfh,
        }
    return rows


def _fit_tsiv_liquid_parent(
    source_dir: Path, species: str, table_id: str, cation: str
) -> list[dict[str, str]]:
    """Fit a tier-3 TSIV liquid and its below-melt supercooled branch."""
    record = _load_record(source_dir / f"tsiv-table-{table_id.split('-')[1]}.json")
    if record.get("record_id") != table_id:
        raise ValueError(f"{table_id}: source record identity mismatch")
    source_rows = _tsiv_liquid_source_rows(record)
    fusion_temperature = float(record["fusion_temperature_K"]["value"])
    intervals = (
        (1200.0, fusion_temperature, False),
        (fusion_temperature, 3000.0, True),
    )
    fitted = []
    for fit_min, fit_max, source_data in intervals:
        fit_nodes = sorted(
            temperature
            for temperature in source_rows
            if fit_min <= temperature <= fit_max
        )
        row = _fit_condensate_values(
            f"{species}(l)",
            table_id,
            cation,
            1,
            1,
            source_rows,
            float(record["formation_enthalpy_298_kj_mol"]["value"]),
            fit_t_min=fit_min,
            fit_t_max=fit_max,
            runtime_t_min=fit_min,
            runtime_t_max=fit_max,
            ref=(table_id if source_data else f"{table_id}-SC-{fusion_temperature:g}"),
            fit_temperatures=fit_nodes,
        )
        if float(row["_max_residual_J_per_mol"]) > 10.0:
            raise ValueError(
                f"{species}(l): TSIV fit residual "
                f"{row['_max_residual_J_per_mol']} J/mol exceeds 10"
            )
        fitted.append(row)
    return fitted


def build_condensate_rows(
    source_dir: Path,
    nasa_source_dir: Path | None = None,
    published_source_dir: Path | None = None,
) -> list[dict[str, str]]:
    repository = Path(__file__).resolve().parents[1]
    nasa_source_dir = nasa_source_dir or repository / "data-src/nasa-glenn"
    published_source_dir = published_source_dir or repository / "data-src/published-thermo"
    rows = []
    for element in KONINGS_BATCH_ELEMENTS:
        source_path = KONINGS_SOURCE_DIR / f"{element.lower()}.json"
        if not source_path.is_file():
            continue
        konings_record = _load_record(
            source_path
        )
        for species, item in konings_record["condensates"].items():
            formula = item["formula"]
            rows.extend(_konings_condensate_rows(
                konings_record, species, element, int(formula[element]),
                int(formula["O"]),
            ))
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
    # FeO(l) keeps the existing D&G/Masson candidate coefficients. This is
    # the selected primary-measurement standard-state reduction, not a refit
    # of the JANAF Fe-019 construction.
    rows.append({
        "species_name": "FeO(l)", "state": "l", "cation": "Fe",
        "cat_num": "1", "oxy_num": "1", "T_min": "1000", "T_max": "5000",
        "dH298_R": "-26.668958359186785",
        "dG_A": "8.03837194242113", "dG_B": "6.588",
        "dG_C": "-1.248", "dG_D": "0.150", "dG_E": "-0.007697",
        "Ref": "DG46-MASSON65",
    })
    for source in TRACE_CONDENSED_SOURCES:
        rows.append(
            _fit_condensate_row(
                source_dir,
                *source,
                fit_t_min=1200.0,
                fit_t_max=3000.0,
                runtime_t_min=1200.0,
                runtime_t_max=3000.0,
                phase="cr",
            )
        )
    # Mo-015 prints the liquid branch through 2000 K. Continue its last
    # printed Cp above that endpoint with the labelled, source-limited rule.
    rows.extend(
        _fit_janaf_parent_liquid(
            source_dir,
            "MoO3(l)",
            "Mo-015",
            "Mo",
            1,
            3,
            (
                (1200.0, 1600.0, None, False),
                (1600.0, 2000.0, None, False),
                (2000.0, 2500.0, None, True),
                (2500.0, 3000.0, None, True),
            ),
        )
    )
    # O-066 prints WO3 liquid cells below its 1745 K II <--> LIQUID marker;
    # these supercooled values are source data. Its liquid branch extends to
    # 3000 K, so all four intervals use printed nodes without continuation.
    rows.extend(
        _fit_janaf_parent_liquid(
            source_dir,
            "WO3(l)",
            "O-066",
            "W",
            1,
            3,
            (
                (1200.0, 1500.0, (1200, 1300, 1400, 1500), False),
                (1500.0, 2000.0, (1500, 1600, 1700, 1900, 2000), False),
                (2000.0, 2500.0, None, False),
                (2500.0, 3000.0, None, False),
            ),
        )
    )
    # These oxide parents use their JANAF liquid formation tables directly.
    # The printed branches include metastable liquid values below each melt
    # marker; they remain source data and are not replaced by a construction.
    rows.extend(
        _fit_janaf_parent_liquid(
            source_dir,
            "BaO(l)",
            "Ba-035",
            "Ba",
            1,
            1,
            (
                (1200.0, 1500.0, None, False),
                (1500.0, 2000.0, None, False),
                # BaO's 2100 K thermal row is valid; unlike Ta2O5, use all parsed nodes.
                (2000.0, 2500.0, None, False),
                (2500.0, 3000.0, None, False),
            ),
        )
    )
    rows.extend(
        _fit_janaf_parent_liquid(
            source_dir,
            "SrO(l)",
            "O-014",
            "Sr",
            1,
            1,
            (
                (1200.0, 1500.0, None, False),
                (1500.0, 2000.0, None, False),
                (2000.0, 2500.0, None, False),
                (2500.0, 3000.0, None, False),
            ),
        )
    )
    rows.extend(
        _fit_janaf_parent_liquid(
            source_dir,
            "ZrO2(l)",
            "O-053",
            "Zr",
            1,
            2,
            (
                (1200.0, 1500.0, None, False),
                (1500.0, 2000.0, None, False),
                (2000.0, 2500.0, None, False),
                (2500.0, 3000.0, None, False),
            ),
        )
    )
    rows.extend(
        _fit_janaf_parent_liquid(
            source_dir,
            "Ta2O5(l)",
            "O-078",
            "Ta",
            2,
            5,
            (
                (1200.0, 1500.0, None, False),
                (1500.0, 2000.0, None, False),
                (2000.0, 2500.0, (2000, 2200, 2300, 2400, 2500), False),
                (2500.0, 3000.0, None, False),
            ),
        )
    )
    rows.extend(_fit_k2o_from_solid(source_dir))
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
            runtime_t_min=1000.0,
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
    for (
        species_name,
        table_id,
        cation,
        cat_num,
        oxy_num,
        T0,
        intervals,
    ) in SUPERCOOLED_LIQUID_SOURCES:
        for runtime_t_min, runtime_t_max in intervals:
            upper_g_cp = None
            upper_g_s = None
            upper_cp = None
            if species_name in {"SiO2(l)", "Al2O3(l)", "MgO(l)", "CaO(l)"} and (
                runtime_t_min == 1000.0 and runtime_t_max == 1200.0
            ):
                seam_row = next(
                    row for row in rows
                    if row["species_name"] == species_name
                    and float(row["T_min"]) == 1200.0
                    and float(row["T_max"]) == 1500.0
                )
                upper_g_s = _condensate_row_g_s(seam_row, 1200.0)
                upper_cp = _condensate_row_g_cp(seam_row, 1200.0)[1]
            rows.append(
                _fit_supercooled_liquid_row(
                    source_dir,
                    species_name,
                    table_id,
                    cation,
                    cat_num,
                    oxy_num,
                    T0,
                    runtime_t_max,
                    runtime_t_min=runtime_t_min,
                    upper_g_cp=upper_g_cp,
                    upper_g_s=upper_g_s,
                    upper_cp=upper_cp,
                    shared_seams=_shared_interval_bounds(intervals),
                )
            )
    rows.extend(
        _fit_supercooled_liquid_row(
            source_dir, *source, nasa_source_dir=nasa_source_dir
        )
        for source in NASA_SUPERCOOLED_LIQUID_SOURCES
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
    # B-096 prints a crystal/liquid marker at 723 K and complete liquid
    # thermal cells from 800 K. Its constant-Cp branch needs two direct fits
    # to keep both intervals below the 10 J/mol node residual limit.
    for fit_t_min, fit_t_max, runtime_t_min in (
        (800.0, 1500.0, 1200.0),
        (1500.0, 3000.0, 1500.0),
    ):
        for species_name, table_id, cation, cat_num, oxy_num in TRACE_CONDENSATE_SOURCES:
            rows.append(
                _fit_janaf_text_condensate_row(
                    source_dir,
                    species_name,
                    table_id,
                    cation,
                    cat_num,
                    oxy_num,
                    fit_t_min=fit_t_min,
                    fit_t_max=fit_t_max,
                    runtime_t_min=runtime_t_min,
                    runtime_t_max=fit_t_max,
                )
            )
    for (
        species_name,
        table_id,
        formula,
        cation,
        cat_num,
        oxy_num,
        source_t_min,
        continuation_intervals,
        fit_intervals,
    ) in TRACE_NASA_CONDENSATE_SOURCES:
        for runtime_t_min, runtime_t_max in continuation_intervals:
            rows.append(
                _fit_supercooled_liquid_row(
                    source_dir,
                    species_name,
                    table_id,
                    cation,
                    cat_num,
                    oxy_num,
                    source_t_min,
                    runtime_t_max,
                    runtime_t_min=runtime_t_min,
                    nasa_source_dir=nasa_source_dir,
                    shared_seams=_shared_interval_bounds(continuation_intervals),
                )
            )
        for fit_t_min, fit_t_max in fit_intervals:
            if species_name == "BeO(l)":
                # The liquid card prints a constant-Cp function. Reuse its
                # exact T0 continuation over the card-supported interval so
                # the narrow 2851-3000 K fit is constrained by the function,
                # not just two endpoint samples.
                row = _fit_supercooled_liquid_row(
                    source_dir,
                    species_name,
                    table_id,
                    cation,
                    cat_num,
                    oxy_num,
                    source_t_min,
                    fit_t_max,
                    runtime_t_min=fit_t_min,
                    nasa_source_dir=nasa_source_dir,
                )
                if fit_t_min >= source_t_min:
                    row["Ref"] = (
                        "NG-1352" if table_id == "NG-BeO-L-1352" else table_id
                    )
                rows.append(row)
            else:
                rows.append(
                    _fit_nasa_card_condensate_row(
                        nasa_source_dir,
                        species_name,
                        table_id,
                        formula,
                        cation,
                        cat_num,
                        oxy_num,
                        fit_t_min=fit_t_min,
                        fit_t_max=fit_t_max,
                    )
                )
    # Cs2O(l) has no JANAF liquid table; retain the NASA pure-liquid card.
    for fit_t_min, fit_t_max in (
        (1200.0, 1500.0),
        (1500.0, 2000.0),
        (2000.0, 3000.0),
    ):
        rows.append(
            _fit_nasa_card_condensate_row(
                nasa_source_dir,
                "Cs2O(l)",
                "NG-1842",
                "Cs2O",
                "Cs",
                2,
                1,
                fit_t_min=fit_t_min,
                fit_t_max=fit_t_max,
            )
        )
    # Cu2O(l) is selected from JANAF through its final 2000 K liquid cell;
    # NASA supplies only the H/S-increment tail above that node.
    for fit_t_min, fit_t_max in (
        (1200.0, 1500.0),
        (1500.0, 2000.0),
        (2000.0, 2500.0),
        (2500.0, 3000.0),
    ):
        rows.append(
            _fit_janaf_condensate_with_nasa_tail(
                source_dir,
                nasa_source_dir,
                "Cu2O(l)",
                "Cu-020",
                "NG-1844",
                "Cu",
                2,
                1,
                nasa_formula="Cu2O",
                anchor_temperature=2000.0,
                fit_t_min=fit_t_min,
                fit_t_max=fit_t_max,
            )
        )
    # SnO(l) is the offered parent; its 1250 K NASA liquid start requires the
    # labelled constant-Cp continuation only between 1200 and that source row.
    for fit_t_min, fit_t_max in (
        (1250.0, 2000.0),
        (2000.0, 3000.0),
    ):
        rows.append(
            _fit_nasa_card_condensate_row(
                nasa_source_dir,
                "SnO(l)",
                "NG-1848",
                "SnO",
                "Sn",
                1,
                1,
                fit_t_min=fit_t_min,
                fit_t_max=fit_t_max,
            )
        )
    liquid_inputs = _load_record(repository / "data-src/liquid-from-solid-inputs.json")
    for species in ("ZnO", "CdO", "MnO", "SeO2", "As2O3", "Sb2O3", "Ag2O", "ThO2", "HfO2", "Y2O3"):
        fitted_rows, _construction, _alternate = _fit_liquid_from_solid_parent(
            published_source_dir, species, liquid_inputs["rows"][species]
        )
        rows.extend(fitted_rows)
    rows.extend(
        _fit_published_condensate_row(
            "P672-T82",
            "Bi2O3(l)",
            "Bi",
            2,
            3,
            published_source_dir,
        )
    )
    rows.extend(
        _fit_published_condensate_row(
            "P672-TeO2-l",
            "TeO2(l)",
            "Te",
            1,
            2,
            published_source_dir,
        )
    )
    rows.extend(
        _fit_published_condensate_row(
            "P672-Tl2O-l",
            "Tl2O(l)",
            "Tl",
            2,
            1,
            published_source_dir,
        )
    )
    # The TSIV liquid branch is selected directly for NiO and CoO. Its
    # below-melt portion is the same printed constant-Cp function continued
    # as supercooled liquid; source coverage begins at the source melt marker.
    rows.extend(
        _fit_tsiv_liquid_parent(published_source_dir, "NiO", "TSIV-2026", "Ni")
    )
    rows.extend(
        _fit_tsiv_liquid_parent(published_source_dir, "CoO", "TSIV-1334", "Co")
    )
    rows.extend(
        _fit_burcat_metal_parent(
            species_name, filename, record_key, cation, cat_num, oxy_num,
            phase, fit_min, fit_max, _source_kind,
        )
        for (
            species_name, filename, record_key, cation, cat_num, oxy_num,
            phase, fit_min, fit_max, _source_kind,
        ) in BURCAT_METAL_PARENT_SOURCES
    )
    rows.extend(
        _fit_pankratz_metal_parent(
            species_name, table_id, cation, cat_num, oxy_num,
            phase, fit_min, fit_max, source_kind,
        )
        for (
            species_name, table_id, cation, cat_num, oxy_num,
            phase, fit_min, fit_max, source_kind,
        ) in PANKRATZ_METAL_PARENT_SOURCES
    )
    # Jung and Hudon assess the reaction P2(g) + 5/2 O2(g) = P2O5(l) as
    # ΔG° = -1616372 + 539.0*T J/mol. The accessible equation does not state
    # p°; use the metallurgical 1-atm convention and convert its gas standards
    # to JANAF's 1 bar. Here Δν_g = 0 - (1 + 5/2) = -3.5, so ΔG°_1bar is
    # ΔG°_1atm - Δν_g*R*T*ln(1.01325), i.e. its slope increases by
    # 3.5*R*ln(1.01325) and ΔS°_rxn decreases by the same amount.
    # As a phase-normalization check, P4O10 = 2 P2O5 in formula units. With
    # identical elemental references and the same pure-liquid standard,
    # formation Gibbs energies are additive: G°f(P4O10(l)) =
    # 2*G°f(P2O5(l)). NASA's P4O10(l) function must therefore be divided by
    # two. This relation does not apply to JANAF P4O10(cr) or to the separate
    # sp_extension activity basis, which uses half P4O10(cr), not P4O10(l).
    p2o5_source = _load_record(
        published_source_dir / "jung-hudon-p2o5.json"
    )
    p2o5_reaction = p2o5_source["delta_g_standard_j_mol"]
    reaction_enthalpy_kj_mol = float(p2o5_reaction["intercept"]) / 1000.0
    reaction_slope_1bar = float(p2o5_reaction["slope"]) + (
        3.5 * R_J_MOL_K * math.log(1.01325)
    )
    reaction_entropy_1bar = -reaction_slope_1bar
    janaf_thermal_rows: dict[str, dict[float, dict[str, float]]] = {}
    for table_id in ("P-012", "O-029"):
        table = _load_record(source_dir / f"{table_id}.yaml")["table"]
        table_rows = {}
        for raw in table["values"]:
            temperature = _value(raw, "temperature")
            if temperature is None:
                continue
            fields = {
                name: _value(raw, name)
                for name in (
                    "heat_capacity",
                    "entropy",
                    "enthalpy_increment",
                )
            }
            if all(value is not None for value in fields.values()):
                table_rows[float(temperature)] = {
                    name: float(value) for name, value in fields.items()
                }
                formation_enthalpy = _value(raw, "formation_enthalpy")
                if formation_enthalpy is not None:
                    table_rows[float(temperature)]["formation_enthalpy"] = float(
                        formation_enthalpy
                    )
        for recovered in _recover_thermal_transition_rows(table):
            table_rows[recovered["temperature"]] = {
                name: recovered[name]
                for name in ("heat_capacity", "entropy", "enthalpy_increment")
            }
        janaf_thermal_rows[table_id] = table_rows
    anchor_hf = (
        janaf_thermal_rows["P-012"][298.15]["formation_enthalpy"]
        + 2.5 * janaf_thermal_rows["O-029"][298.15]["formation_enthalpy"]
        + reaction_enthalpy_kj_mol
    )
    p2o5_rows = {}
    for temperature in np.arange(
        1200.0, 3000.0 + JANAF_GRID_STEP_K, JANAF_GRID_STEP_K
    ):
        T = float(temperature)
        p2 = janaf_thermal_rows["P-012"][T]
        o2 = janaf_thermal_rows["O-029"][T]
        enthalpy_increment = (
            p2["enthalpy_increment"] + 2.5 * o2["enthalpy_increment"]
        )
        entropy = (
            p2["entropy"] + 2.5 * o2["entropy"] + reaction_entropy_1bar
        )
        p2o5_rows[T] = {
            "temperature": T,
            "heat_capacity": p2["heat_capacity"] + 2.5 * o2["heat_capacity"],
            "entropy": entropy,
            "enthalpy_increment": enthalpy_increment,
            "formation_enthalpy": anchor_hf + enthalpy_increment,
            "formation_gibbs_energy": anchor_hf + enthalpy_increment - T * entropy / 1000.0,
        }
    for fit_min, fit_max, fit_nodes in (
        (1200.0, 1500.0, tuple(np.arange(1200.0, 1500.0 + 100.0, 100.0))),
        (1500.0, 3000.0, tuple(np.arange(1500.0, 3000.0 + 100.0, 100.0))),
    ):
        row = _fit_condensate_values(
            "P2O5(l)",
            "Jung-Hudon-2012",
            "P",
            2,
            5,
            p2o5_rows,
            anchor_hf,
            fit_t_min=fit_min,
            fit_t_max=fit_max,
            runtime_t_min=fit_min,
            runtime_t_max=fit_max,
            ref="Jung-Hudon-2012",
            fit_temperatures=fit_nodes,
        )
        if float(row["_max_residual_J_per_mol"]) > 10.0:
            raise ValueError(
                f"P2O5(l): Jung-Hudon fit residual "
                f"{row['_max_residual_J_per_mol']} J/mol exceeds 10"
            )
        rows.append(row)
    return _pin_condensate_same_ref_seams(rows)


def build_source_liquid_markers(
    source_dir: Path,
    provenance_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Extract phase-transition markers for source-backed condensate liquids."""
    repository = Path(__file__).resolve().parents[1]
    provenance_path = provenance_path or (
        repository / "src/openimcc/data/gas/PROVENANCE.yaml"
    )
    provenance = _load_record(provenance_path)
    markers: dict[str, dict[str, Any]] = {}
    source_records = {}
    for row in provenance.get("rows", []):
        if row.get("table") != "condensate":
            continue
        source_path = row.get("source_path")
        source = row.get("source") or {}
        table_id = row.get("table_id") or source.get("table_id")
        if table_id and (
            (source_path or "").startswith("data-src/janaf/")
            or row.get("authority") == "janaf_transcribed"
        ):
            source_records.setdefault(row["species_name"], table_id)
        legacy_table_id = source.get("legacy_table_id")
        if legacy_table_id:
            source_records.setdefault(row["species_name"], legacy_table_id)

    for species_name, table_id in sorted(source_records.items()):
        source_file = next(
            (source_dir / f"{table_id}{suffix}" for suffix in (".yaml", ".json")
             if (source_dir / f"{table_id}{suffix}").is_file()),
            None,
        )
        if source_file is None:
            continue
        record = _load_record(source_file)
        table = record.get("table", record)
        joins: dict[str, tuple[float, int]] = {}
        for ambiguity in table.get("parse_ambiguities", []):
            raw_line = str(ambiguity.get("raw_line", ""))
            cells = raw_line.split("\t")
            if not cells:
                continue
            label = cells[-1].strip().upper()
            if "<--> LIQUID" not in label:
                continue
            try:
                temperature = float(cells[0])
            except ValueError:
                continue
            kind = "glass_join" if label.startswith("GLASS") else "fusion"
            joins[kind] = (temperature, int(ambiguity["line_number"]))
        if "fusion" not in joins:
            continue
        fusion, fusion_line = joins["fusion"]
        glass = joins.get("glass_join")
        markers[species_name] = {
            "table_id": table_id,
            "fusion_temperature_K": fusion,
            "glass_join_temperature_K": glass[0] if glass else None,
        }
    return markers


def _flat_temperature_pair(span: Any) -> tuple[float, float] | None:
    if (
        isinstance(span, (list, tuple))
        and len(span) == 2
        and all(
            isinstance(item, (int, float)) and not isinstance(item, bool)
            for item in span
        )
    ):
        return float(span[0]), float(span[1])
    return None


def _source_fit_range(record: dict[str, Any]) -> Any:
    """Fit interval recorded for a runtime row.

    Seam windows omit source_fit_range_K and keep the parent fit's
    temperature_range. Explicit ranges are returned unchanged so integer
    bounds keep the text already stored on the tables.
    """
    explicit = record.get("source_fit_range_K")
    if explicit:
        return explicit
    residual = record.get("fit_residual")
    if not isinstance(residual, dict):
        return None
    span = _flat_temperature_pair(residual.get("temperature_range"))
    if span is None:
        return None
    low, high = (float(value) for value in record["T_range_K"])
    if span[0] <= low and span[1] >= high:
        return [span[0], span[1]]
    return None


_VALIDITY_FACT_FIELDS = (
    "table_id", "source_table_id", "source_fit_range_K", "source_limit_K",
    "extrapolation_source_limit_K", "fusion_temperature_K", "tier",
    "flags", "method", "source_path", "source_data",
)


def _records_covering_interval(
    records: list[dict[str, Any]],
    low: float,
    high: float,
    *,
    same_fit_span: bool = False,
) -> list[dict[str, Any]]:
    """Records that contain [low, high], or one tile of same-fact pieces."""
    direct = [
        record for record in records
        if float(record["T_range_K"][0]) <= low
        and float(record["T_range_K"][1]) >= high
    ]
    if direct:
        return direct
    overlapping = sorted(
        (
            record for record in records
            if float(record["T_range_K"][0]) < high
            and float(record["T_range_K"][1]) > low
        ),
        key=lambda record: float(record["T_range_K"][0]),
    )
    cursor = low
    for candidate in overlapping:
        candidate_low, candidate_high = (
            float(value) for value in candidate["T_range_K"]
        )
        if candidate_low > cursor:
            break
        cursor = max(cursor, candidate_high)
    signatures = {
        json.dumps(
            {key: record.get(key) for key in _VALIDITY_FACT_FIELDS},
            sort_keys=True, separators=(",", ":"),
        )
        for record in overlapping
    }
    spans_match = True
    if same_fit_span:
        spans = {
            json.dumps(
                record["fit_residual"].get("temperature_range")
                if isinstance(record.get("fit_residual"), dict) else None,
                sort_keys=True,
            )
            for record in overlapping
        }
        spans_match = len(spans) == 1
    if overlapping and cursor >= high and len(signatures) == 1 and spans_match:
        combined = dict(overlapping[0])
        combined["T_range_K"] = [low, high]
        return [combined]
    return []


def _validity_identifier_matches(record: dict[str, Any], ref: str) -> bool:
    return any(
        identifier and (ref == identifier or ref.startswith(identifier))
        for identifier in (
            str(record.get("table_id", "")),
            str(record.get("source_table_id", "")),
        )
    )


def _validity_display_source_table(runtime_ref: str) -> str:
    """Transfer the legacy display identifier into typed row metadata."""
    base = re.sub(r"-CP-EXT-[0-9]+(?:\.[0-9]+)?$", "", runtime_ref)
    bracketed = re.fullmatch(r"LFS\[([^;\]]+)(?:;Tf=[^\]]+)?\]", base)
    if bracketed is not None:
        return bracketed[1]
    return base.removesuffix("-LFS").removesuffix("-SC-CP")


def build_source_liquid_validity_rows(
    gas_csv: Path,
    condensate_csv: Path,
    provenance_path: Path | None = None,
    markers: dict[str, dict[str, Any]] | None = None,
    additional_condensate_csvs: tuple[Path, ...] = (),
) -> dict[str, list[dict[str, Any] | None]]:
    """Build typed validity facts in runtime CSV row order."""
    repository = Path(__file__).resolve().parents[1]
    provenance_path = provenance_path or (
        repository / "src/openimcc/data/gas/PROVENANCE.yaml"
    )
    provenance = _load_record(provenance_path)
    markers = markers or {}
    result: dict[str, list[dict[str, Any] | None]] = {}
    for table_name, path in (
        ("gas", gas_csv),
        ("condensate", condensate_csv),
        *(
            ("condensate", path)
            for path in additional_condensate_csvs
        ),
    ):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            runtime_rows = list(csv.DictReader(handle))
        records = [
            row for row in provenance.get("rows", [])
            if row.get("table") == table_name
        ]
        row_validity: list[dict[str, Any] | None] = []
        for runtime in runtime_rows:
            species = runtime["species_name"]
            ref = runtime["Ref"]
            low, high = float(runtime["T_min"]), float(runtime["T_max"])
            species_records = [
                record for record in records
                if record.get("species_name") == species
                and record.get("T_range_K")
            ]
            interval_records = [
                record for record in species_records
                if float(record["T_range_K"][0]) <= low
                and float(record["T_range_K"][1]) >= high
            ]
            needs_typed_validity = any(
                tag in ref
                for tag in (
                    "P677-Re2O7-g-", "K-012", "-CONT-", "Konings-Table88-Tf",
                    "-CP-EXT-", "LFS", "-SC-CP",
                )
            ) or ("TSIV-" in ref and "-SC-" in ref)
            candidates = _records_covering_interval(
                [
                    record for record in species_records
                    if record.get("runtime_ref") == ref
                ],
                low,
                high,
            )
            if not candidates:
                # An unsplit legacy row tiles the parent fit across seam windows.
                candidates = _records_covering_interval(
                    [
                        record for record in species_records
                        if _validity_identifier_matches(record, ref)
                    ],
                    low,
                    high,
                    same_fit_span=True,
                )
            if not candidates and needs_typed_validity:
                candidates = interval_records
            record = None
            if not candidates:
                marker = markers.get(species)
                if marker is not None and ref in {marker["table_id"], "JANAF"}:
                    record = {
                        "T_range_K": [low, high], "source_fit_range_K": [low, high],
                        "method": "fitted", "table_id": marker["table_id"],
                        "source_path": "data-src/janaf/", "source_data": True,
                        "flags": [],
                    }
                else:
                    candidates = []
            if record is None and not candidates:
                if needs_typed_validity:
                    raise ValueError(
                        f"{table_name} row {species} [{low:g}, {high:g}] {ref!r} "
                        "has no matching provenance validity record"
                    )
                row_validity.append(None)
                continue
            if record is None:
                record = min(
                    candidates,
                    key=lambda item: float(item["T_range_K"][1]) - float(item["T_range_K"][0]),
                )
            source_limit = record.get("extrapolation_source_limit_K")
            if source_limit is None:
                source_limit = record.get("source_limit_K")
            fusion_temperature = record.get("fusion_temperature_K")
            if fusion_temperature is None:
                fusion_temperature = (
                    record.get("inputs", {})
                    .get("fusion_temperature", {})
                    .get("value_K")
                )
            if fusion_temperature is None:
                match = re.search(r"(?:;Tf=|-Tf)([0-9]+(?:\.[0-9]+)?)", ref)
                if match:
                    fusion_temperature = float(match.group(1))
            method = str(record.get("method", ""))
            table_id = record.get("table_id") or record.get("source_table_id")
            flags = [str(flag) for flag in record.get("flags", [])]
            continuation_kind = None
            if "-CONT-" in ref:
                continuation_kind = (
                    "liquid_metal" if "-METAL-CONT-" in ref else "phase"
                )
            supercooled = (
                "supercooled" in method.lower()
                or any("supercooled" in flag.lower() for flag in flags)
            )
            liquid_from_solid = method in {
                "liquid_from_solid", "liquid_from_solid_construction"
            }
            tier = record.get("tier")
            if tier is None and method == "liquid_from_solid":
                # Preserve the legacy K-012 construction's existing tier-4 flag.
                tier = 4
            cp_extension = "-CP-EXT-" in ref
            supercooled_continuation = bool(
                record.get("supercooled_continuation")
                or ref.endswith(SUPERCOOLED_LIQUID_REF_SUFFIX)
            )
            if supercooled and fusion_temperature is None and str(record.get("table_id", "")).startswith("TSIV-"):
                fusion_temperature = high
            marker = markers.get(species)
            source_marker = None
            if (
                marker is not None
                and table_id == marker.get("table_id")
                and str(record.get("source_path", "")).startswith("data-src/janaf/")
                and record.get("source_data", True)
            ):
                source_marker = marker
            source_tagged = any(
                tag in ref
                for tag in (
                    "P677-Re2O7-g-", "K-012", "-CONT-", "Konings-Table88-Tf",
                    "-CP-EXT-", "LFS", "-SC-CP",
                )
            ) or ("TSIV-" in ref and "-SC-" in ref)
            validity_required = bool(
                continuation_kind
                or cp_extension
                or supercooled
                or supercooled_continuation
                or liquid_from_solid
                or method.startswith("constructed_from_")
                or source_marker is not None
            )
            if source_tagged and not validity_required:
                raise ValueError(
                    f"{table_name} row {species} [{low:g}, {high:g}] {ref!r} "
                    "has a coverage tag but no typed validity facts"
                )
            if (continuation_kind or cp_extension) and source_limit is None:
                raise ValueError(
                    f"{table_name} row {species} [{low:g}, {high:g}] {ref!r} "
                    "has no typed continuation source limit"
                )
            if liquid_from_solid and tier is None:
                raise ValueError(
                    f"{table_name} row {species} [{low:g}, {high:g}] {ref!r} "
                    "has no typed liquid_from_solid estimate tier"
                )
            if (
                source_marker is not None
                and source_marker.get("fusion_temperature_K") is None
            ):
                raise ValueError(
                    f"{table_name} row {species} [{low:g}, {high:g}] {ref!r} "
                    "has source marker metadata without a fusion temperature"
                )
            validity = {
                "T_range_K": [float(value) for value in record["T_range_K"]],
                "source_fit_range_K": _source_fit_range(record),
                "source_limit_K": source_limit,
                "fusion_temperature_K": fusion_temperature,
                "supercooled_segment_K": (
                    [low, float(fusion_temperature)]
                    if supercooled and fusion_temperature is not None else None
                ),
                "continuation_kind": continuation_kind,
                "cp_extension": cp_extension,
                "supercooled_continuation": supercooled_continuation,
                "liquid_from_solid": liquid_from_solid,
                "method": method,
                "tier": tier,
                "table_id": table_id,
                "display_source_table": _validity_display_source_table(ref),
                "source_marker": source_marker,
            }
            if not (
                continuation_kind
                or cp_extension
                or supercooled
                or validity["supercooled_continuation"]
                or liquid_from_solid
                or method.startswith("constructed_from_")
                or source_marker is not None
            ):
                row_validity.append(None)
                continue
            row_validity.append(validity)
        result[f"{table_name}:{path.resolve()}"] = row_validity
    return result


def write_source_validity_columns(
    table_path: Path,
    validity_rows: dict[str, list[dict[str, Any] | None]],
    *,
    phase: str,
) -> None:
    """Write generated validity facts directly onto their runtime CSV rows."""
    row_validity = validity_rows[f"{phase}:{table_path.resolve()}"]
    with table_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    expected = set(SOURCE_VALIDITY_COLUMNS)
    if not expected.issubset(fieldnames):
        raise ValueError(f"{table_path} lacks source-validity columns")
    if len(row_validity) != len(rows):
        raise ValueError(f"{table_path} validity row count does not match runtime rows")
    for row, validity in zip(rows, row_validity):
        fields = _empty_source_validity_fields()
        if validity is not None:
            source_fit = validity.get("source_fit_range_K") or validity["T_range_K"]
            segment = validity.get("supercooled_segment_K") or (0.0, 0.0)
            marker = validity.get("source_marker") or {}
            fields.update({
                "validity_kind": "coverage",
                "continuation_kind": validity.get("continuation_kind") or "none",
                "cp_extension": "true" if validity.get("cp_extension") else "false",
                "construction_method": validity.get("method") or "none",
                "table_id": validity.get("table_id") or "none",
                "source_label": validity.get("display_source_table") or validity.get("table_id") or "none",
                "source_liquid_marker": "true" if marker else "false",
                "supercooled_continuation": (
                    "true" if validity.get("supercooled_continuation") else "false"
                ),
                "source_fit_min_K": str(source_fit[0]),
                "source_fit_max_K": str(source_fit[1]),
                "source_limit_K": str(validity.get("source_limit_K") or 0),
                "fusion_temperature_K": str(
                    validity.get("fusion_temperature_K")
                    or marker.get("fusion_temperature_K")
                    or 0
                ),
                "supercooled_segment_min_K": str(segment[0]),
                "supercooled_segment_max_K": str(segment[1]),
                "construction_tier": str(validity.get("tier") or 0),
                "glass_join_temperature_K": str(marker.get("glass_join_temperature_K") or 0),
            })
        row.update(fields)
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _fit_k2o_from_solid(source_dir: Path) -> list[dict[str, str]]:
    """Fit K2O(l) from its crystal row and the recorded liquid input ladder."""
    from tools.liquid_from_solid import liquid_from_solid

    repository = Path(__file__).resolve().parents[1]
    inputs_path = repository / "data-src/liquid-from-solid-inputs.json"
    inputs = json.loads(inputs_path.read_text(encoding="utf-8"))["rows"]["K2O"]
    crystal = _load_record(source_dir / f"{inputs['crystal_table_id']}.yaml")
    construction = liquid_from_solid("K2O", crystal, inputs["inputs"])
    fit = inputs["inputs"]["condensate_fit"]
    domain_min = float(fit["fit_t_min"])
    domain_max = float(fit["fit_t_max"])
    # Premise: the existing condensate row is a quartic in T/1000. Algebra:
    # fitting the full 1200–3000 K construction leaves a 76.5 J/mol node
    # residual, while 1200–2000 and 2000–3000 K each stay below 10 J/mol.
    # Unit check: the fitter records residuals in J/mol; sanity: both fits
    # include their shared 2000 K seam and cover the full runtime interval.
    intervals = ((domain_min, 2000.0), (2000.0, domain_max))
    result = []
    for fit_min, fit_max in intervals:
        nodes = [
            float(temperature)
            for temperature in np.arange(
                fit_min, fit_max + JANAF_GRID_STEP_K, JANAF_GRID_STEP_K
            )
        ]
        row = _fit_condensate_values(
            "K2O(l)",
            str(fit["table_id"]),
            str(fit["cation"]),
            int(fit["cat_num"]),
            int(fit["oxy_num"]),
            construction.source_rows(nodes),
            construction.dfh298_kj_mol,
            fit_t_min=fit_min,
            fit_t_max=fit_max,
            runtime_t_min=fit_min,
            runtime_t_max=fit_max,
            ref=str(fit["ref"]),
        )
        residual = float(row["_max_residual_J_per_mol"])
        if residual > 10.0:
            raise ValueError(
                f"K2O(l): condensate fit residual {residual:g} J/mol exceeds 10"
            )
        result.append(row)
    # The 1000-1200 K row copies this segment at 1200 K. Leave the segment
    # fixed so that copy stays the shared state; the same-Ref pass closes
    # 2000 K on the upper segment.
    result[0]["_seam_frozen"] = "true"

    # Fit the lower interval against the constructed continuation at 1 K spacing.
    lower_nodes = np.arange(1000.0, 1200.0 + 1.0, 1.0)
    lower_source = construction.source_rows(lower_nodes.tolist())
    upper_g, upper_s = _condensate_row_g_s(result[0], 1200.0)
    upper_cp = _condensate_row_g_cp(result[0], 1200.0)[1]
    fusion_state = construction.source_rows([1013.0])[1013.0]
    fusion_s = fusion_state["entropy"]
    row = _fit_condensate_values(
        "K2O(l)",
        str(fit["table_id"]),
        str(fit["cation"]),
        int(fit["cat_num"]),
        int(fit["oxy_num"]),
        lower_source,
        construction.dfh298_kj_mol,
        fit_t_min=1000.0,
        fit_t_max=1200.0,
        runtime_t_min=1000.0,
        runtime_t_max=1200.0,
        ref="LFS[K-012;Tf=1013]",
        fit_temperatures=lower_nodes,
        endpoint_g_s=(
            (1013.0, construction.at(1013.0).gibbs_kj_mol * 1000.0, fusion_s),
            (1200.0, upper_g, upper_s),
        ),
        endpoint_cp=(1200.0, upper_cp),
    )
    residual = float(row["_max_residual_J_per_mol"])
    if residual > 10.0:
        raise ValueError(
            f"K2O(l): 1000-1200 K source-fit residual {residual:g} J/mol exceeds 10"
        )
    result.append(row)
    return result


def _condensate_row_g_cp(
    row: dict[str, str], temperature: float
) -> tuple[float, float]:
    """Evaluate one condensate Shomate row's Gibbs energy and heat capacity."""
    tau = temperature / 1000.0
    coefficients = [float(row[f"dG_{name}"]) for name in "ABCDE"]
    A, B, C, D, E = coefficients
    polynomial = A + B * tau + C * tau**2 + D * tau**3 + E * tau**4
    cp = R_J_MOL_K * (
        2.0 * B * tau + 6.0 * C * tau**2
        + 12.0 * D * tau**3 + 20.0 * E * tau**4
    )
    gibbs = (
        1000.0 * R_J_MOL_K * float(row["dH298_R"])
        - R_J_MOL_K * temperature * polynomial
    )
    return gibbs, cp


def _condensate_row_g_s(
    row: dict[str, str], temperature: float
) -> tuple[float, float]:
    """Evaluate stored condensate Gibbs energy and entropy (−dG/dT)."""
    tau = temperature / 1000.0
    coefficients = [float(row[f"dG_{name}"]) for name in "ABCDE"]
    polynomial = sum(value * tau**power for power, value in enumerate(coefficients))
    derivative = sum(
        power * coefficients[power] * tau ** (power - 1)
        for power in range(1, 5)
    )
    gibbs = 1000.0 * float(row["dH298_R"]) * R_J_MOL_K - R_J_MOL_K * temperature * polynomial
    entropy = R_J_MOL_K * (polynomial + tau * derivative)
    return gibbs, entropy


def write_csv(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    defaults = _empty_source_validity_fields()
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUNTIME_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            {
                column: row.get(column, defaults.get(column, ""))
                for column in RUNTIME_COLUMNS
            }
            for row in rows
        )


def _parse_cea_thermo_cards(path: Path) -> dict[str, dict[str, Any]]:
    """Read the legacy fixed-width NASA CEA thermo.inp card layout.

    Each card has a name/description, an elemental-reference line, then blocks
    of a temperature-range line and two coefficient lines. The 7 NASA
    coefficients and b1/b2 integration constants are enough to evaluate Cp,
    H, and S. This parser is deliberately scoped to the cards used below.
    """
    number = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][+-]?\d+)?")
    lines = path.read_text(encoding="ascii").splitlines()

    def values(line: str) -> list[float]:
        return [float(token.replace("D", "E").replace("d", "e")) for token in number.findall(line)]

    cards: dict[str, dict[str, Any]] = {}
    wanted = {item[1].upper() for item in (*HIGH_TEMP_GAS_CARDS, *HIGH_TEMP_LIQUID_CARDS)}
    for index, line in enumerate(lines):
        name = line.strip().split(maxsplit=1)[0].upper() if line.strip() else ""
        if name not in wanted:
            continue
        if index + 2 >= len(lines):
            raise ValueError(f"truncated CEA card {name}")
        description = lines[index].strip()
        reference_line = lines[index + 1]
        reference_values = values(reference_line)
        if len(reference_values) < 2:
            raise ValueError(f"invalid CEA reference line for {name}")
        intervals = []
        cursor = index + 2
        while cursor + 2 < len(lines):
            range_line = lines[cursor]
            if not range_line[:1].isspace():
                break
            bounds = values(range_line)
            first = values(lines[cursor + 1])
            second = values(lines[cursor + 2])
            if len(bounds) < 2 or len(first) != 5 or len(second) < 4:
                break
            intervals.append({
                "T_min": bounds[0], "T_max": bounds[1],
                "a": first + second[:2], "b1": second[-2], "b2": second[-1],
            })
            cursor += 3
        if not intervals:
            raise ValueError(f"no NASA intervals found for CEA card {name}")
        if max(item["T_max"] for item in intervals) < 6000.0:
            raise ValueError(f"CEA card {name} does not cover 6000 K")
        cards[name] = {
            "description": description,
            "delta_f_H_298_15": reference_values[-1],
            "intervals": intervals,
        }
    missing = wanted - cards.keys()
    if missing:
        raise ValueError(f"CEA thermo file is missing cards: {sorted(missing)}")
    return cards


def _cea_properties(card: dict[str, Any], temperature: float) -> dict[str, float]:
    """Evaluate the source NASA-7 polynomial at ``temperature`` in K."""
    interval = next(
        (item for item in card["intervals"] if item["T_min"] <= temperature <= item["T_max"]),
        card["intervals"][-1] if temperature > card["intervals"][-1]["T_max"] else card["intervals"][0],
    )
    a1, a2, a3, a4, a5, a6, a7 = interval["a"]
    cp_R = a1 / temperature**2 + a2 / temperature + a3 + a4 * temperature + a5 * temperature**2 + a6 * temperature**3 + a7 * temperature**4
    h_rt = -a1 / temperature**2 + a2 * math.log(temperature) / temperature + a3 + a4 * temperature / 2 + a5 * temperature**2 / 3 + a6 * temperature**3 / 4 + a7 * temperature**4 / 5 + interval["b1"] / temperature
    s_R = -a1 / (2 * temperature**2) - a2 / temperature + a3 * math.log(temperature) + a4 * temperature + a5 * temperature**2 / 2 + a6 * temperature**3 / 3 + a7 * temperature**4 / 4 + interval["b2"]
    return {"cp": R_J_MOL_K * cp_R, "H": R_J_MOL_K * temperature * h_rt, "S": R_J_MOL_K * s_R}


def _gas_row_h_s(row: dict[str, Any], temperature: float) -> tuple[float, float]:
    t = temperature / 1000.0
    A, B, C, D, E, F, G = (float(row[key]) for key in "ABCDEFG")
    h = A * t + B * t**2 / 2 + C * t**3 / 3 + D * t**4 / 4 - E / t + F
    s = A * math.log(t) + B * t + C * t**2 / 2 + D * t**3 / 3 - E / (2 * t**2) + G
    return h * 1000.0, s


def _gas_candidate_row(base: dict[str, Any], card: dict[str, Any], card_name: str) -> tuple[dict[str, str], dict[float, tuple[float, float, float]]]:
    temperatures = np.arange(3000.0, 6000.1, 100.0)
    cp_rows = [_cea_properties(card, float(T))["cp"] for T in temperatures]
    t = temperatures / 1000.0
    design = np.column_stack((np.ones_like(t), t, t**2, t**3, t**-2))
    A, B, C, D, E = np.linalg.lstsq(design, np.asarray(cp_rows), rcond=None)[0]
    h0, s0 = _gas_row_h_s(base, 3000.0)
    tau = 3.0
    ih = A*tau + B*tau**2/2 + C*tau**3/3 + D*tau**4/4 - E/tau
    ins = A*math.log(tau) + B*tau + C*tau**2/2 + D*tau**3/3 - E/(2*tau**2)
    row = {key: str(base[key]) for key in RUNTIME_COLUMNS if key in base}
    row.update({"T_interval": "3", "T_min": "3000", "T_max": "6000", "A": f"{A:.15g}", "B": f"{B:.15g}", "C": f"{C:.15g}", "D": f"{D:.15g}", "E": f"{E:.15g}", "F": f"{h0/1000-ih:.15g}", "G": f"{s0-ins:.15g}", "Ref": f"{card_name}-HT-CANDIDATE"})
    offsets = {}
    for T in (3000.0, 4000.0, 6000.0):
        source = _cea_properties(card, T)
        # Compare the actual emitted Shomate row, not the unrounded Cp fit, to
        # the source card so the report reflects runtime values.
        h, s = _gas_row_h_s(row, T)
        offsets[T] = (abs(h-source["H"]), abs(s-source["S"]), abs((h-T*s)-(source["H"]-T*source["S"])))
    return row, offsets


def _liquid_cp(row: dict[str, Any], temperature: float) -> float:
    tau = temperature / 1000.0
    A, B, C, D, E = (float(row[f"dG_{key}"]) for key in "ABCDE")
    return R_J_MOL_K * (2*B*tau + 6*C*tau**2 + 12*D*tau**3 + 20*E*tau**4)


def _liquid_source_rows(
    card: dict[str, Any], base_row: dict[str, Any],
    start: float, end: float, *, use_base_cp: bool = False,
) -> dict[float, dict[str, float]]:
    """Integrate Cp from an anchored liquid seam into fitter source cells."""
    g0, s0 = _condensate_row_g_s(base_row, start)
    h0 = g0 + start*s0
    ref_j = float(base_row["dH298_R"]) * R_J_MOL_K
    nodes = sorted({start, end, *np.arange(start, end + 1.0, 100.0).tolist()})
    grid = np.arange(start, end + 1.0, 1.0)
    if grid[-1] != end:
        grid = np.append(grid, end)
    cp_values = np.asarray([
        _liquid_cp(base_row, float(T)) if use_base_cp else _cea_properties(card, float(T))["cp"]
        for T in grid
    ])
    h_grid = np.concatenate(([h0], h0 + np.cumsum((cp_values[:-1]+cp_values[1:]) * np.diff(grid) / 2.0)))
    s_grid = np.concatenate(([s0], s0 + np.cumsum((cp_values[:-1]/grid[:-1]+cp_values[1:]/grid[1:]) * np.diff(grid) / 2.0)))
    result = {}
    for T in nodes:
        index = int(round(T-start))
        H, S = float(h_grid[index]), float(s_grid[index])
        cp = _liquid_cp(base_row, T) if use_base_cp else _cea_properties(card, T)["cp"]
        result[float(T)] = {"temperature": float(T), "heat_capacity": cp,
                            "entropy": S, "enthalpy_increment": (H-ref_j)/1000.0}
    return result


def _make_liquid_candidate_row(
    species: str, card_name: str, cation: str, cat_num: int, oxy_num: int,
    base_row: dict[str, Any], card: dict[str, Any], start: float, end: float,
) -> dict[str, str]:
    source_rows = _liquid_source_rows(card, base_row, start, end)
    g0, s0 = _condensate_row_g_s(base_row, start)
    reference_kj = float(base_row["dH298_R"]) * R_J_MOL_K / 1000.0
    row = _fit_condensate_values(
        species, f"{card_name}-HT-CANDIDATE", cation, cat_num, oxy_num,
        source_rows, reference_kj, fit_t_min=start, fit_t_max=end,
        runtime_t_min=start, runtime_t_max=end,
        fit_temperatures=sorted(source_rows), endpoint_g_s=[(start, g0, s0)],
    )
    row.update(_empty_source_validity_fields())
    row.update({"state": "l", "Ref": f"{card_name}-HT-CANDIDATE"})
    return row


def build_high_temperature_candidates(
    cea_path: Path, gas_input: Path, oxide_input: Path,
    gas_output: Path, oxide_output: Path,
) -> dict[str, dict[float, tuple[float, float, float]]]:
    """Write opt-in 3000–6000 K tables without touching packaged defaults.

    At each 3000 K seam (and the CaO/MgO bridge point), take G and S from the
    prior row and integrate Cp(T) to obtain H. The source card contributes only
    Cp shape above the anchor. This makes G continuous, makes S continuous,
    and therefore makes H=G+TS continuous. CEA absolute H/S anchors remain
    visible through the printed offset report instead of shifting the default.
    """
    cards = _parse_cea_thermo_cards(cea_path)
    gas_rows = list(csv.DictReader(gas_input.open(encoding="utf-8", newline="")))
    new_gas = []
    offsets: dict[str, dict[float, tuple[float, float, float]]] = {}
    for species, card_name, cation, cat_num, oxy_num in HIGH_TEMP_GAS_CARDS:
        matches = [row for row in gas_rows if row["species_name"] == species and float(row["T_max"]) == 3000.0]
        if not matches:
            raise ValueError(f"no 3000 K packaged gas interval for {species}")
        base = dict(matches[-1])
        for key in "ABCDEFG":
            base[key] = float(base[key])
        card = cards[card_name.upper()]
        row, offset = _gas_candidate_row(base, card, card_name)
        row.update({"species_name": species, "state": "g", "cation": cation,
                    "cat_num": str(cat_num), "oxy_num": str(oxy_num),
                    "Ref": f"CEA-{card_name}-HT-CANDIDATE"})
        new_gas.append(row)
        offsets[species] = offset
    gas_output.parent.mkdir(parents=True, exist_ok=True)
    with gas_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUNTIME_COLUMNS, lineterminator="\n")
        writer.writeheader(); writer.writerows(gas_rows + new_gas)

    oxide_rows = list(csv.DictReader(oxide_input.open(encoding="utf-8", newline="")))
    new_oxide = []
    for species, card_name, cation, cat_num, oxy_num, _ in HIGH_TEMP_LIQUID_CARDS:
        base_matches = [row for row in oxide_rows if row["species_name"] == species and float(row["T_max"]) == 3000.0]
        if not base_matches:
            raise ValueError(f"no 3000 K packaged liquid interval for {species}")
        base = base_matches[-1]
        bridge = 3172.0 if species == "CaO(l)" else 3100.0 if species == "MgO(l)" else 3000.0
        if bridge > 3000.0:
            seam = {**base, "T_min": "3000", "T_max": str(bridge),
                    "Ref": f"CEA-{card_name}-BRIDGE-HT-CANDIDATE"}
            new_oxide.append(seam)
        row = _make_liquid_candidate_row(
            species, card_name, cation, cat_num, oxy_num, base,
            cards[card_name.upper()], bridge, 6000.0,
        )
        new_oxide.append(row)
        liquid_offsets = {}
        for temperature in (3000.0, 4000.0, 6000.0):
            used_rows = [
                row for row in new_oxide if row["species_name"] == species
                and float(row["T_min"]) <= temperature <= float(row["T_max"])
            ]
            used = max(used_rows, key=lambda row: float(row["T_min"]))
            g, s = _condensate_row_g_s(used, temperature)
            h = g + temperature * s
            raw = _cea_properties(cards[card_name.upper()], temperature)
            liquid_offsets[temperature] = (
                abs(h-raw["H"]), abs(s-raw["S"]),
                abs(g-(raw["H"]-temperature*raw["S"])),
            )
        offsets[species] = liquid_offsets
    feo_matches = [row for row in oxide_rows if row["species_name"] == "FeO(l)" and float(row["T_min"]) <= 3000 <= float(row["T_max"])]
    if not feo_matches:
        raise ValueError("no default FeO(l) interval covers 3000 K")
    feo_base = feo_matches[-1]
    # FeO has no CEA liquid card. Hold the current default row's Cp at 3000 K.
    feo_sources = _liquid_source_rows({}, feo_base, 3000, 6000, use_base_cp=True)
    feo = _fit_condensate_values(
        "FeO(l)", "FeO-constant-Cp-HT-CANDIDATE", "Fe", 1, 1,
        feo_sources, float(feo_base["dH298_R"])*R_J_MOL_K/1000,
        fit_t_min=3000, fit_t_max=6000, runtime_t_min=3000, runtime_t_max=6000,
        fit_temperatures=sorted(feo_sources), endpoint_g_s=[(3000, *_condensate_row_g_s(feo_base, 3000))],
    )
    feo.update(_empty_source_validity_fields())
    feo.update({"state": "l", "Ref": "FeO-constant-Cp-HT-CANDIDATE"})
    new_oxide.append(feo)
    offsets["FeO(l)"] = {T: None for T in (3000.0, 4000.0, 6000.0)}
    oxide_output.parent.mkdir(parents=True, exist_ok=True)
    with oxide_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONDENSATE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in CONDENSATE_COLUMNS} for row in oxide_rows + new_oxide)
    return offsets


SOURCE_VALIDITY_COLUMNS = (
    "validity_kind", "continuation_kind", "cp_extension", "construction_method",
    "table_id", "source_label", "source_liquid_marker", "supercooled_continuation",
    "source_fit_min_K", "source_fit_max_K", "source_limit_K",
    "fusion_temperature_K", "supercooled_segment_min_K",
    "supercooled_segment_max_K", "construction_tier", "glass_join_temperature_K",
)


def _empty_source_validity_fields() -> dict[str, str]:
    return {
        "validity_kind": "none", "continuation_kind": "none", "cp_extension": "false",
        "construction_method": "none", "table_id": "none", "source_label": "none",
        "source_liquid_marker": "false", "supercooled_continuation": "false",
        "source_fit_min_K": "0",
        "source_fit_max_K": "0", "source_limit_K": "0",
        "fusion_temperature_K": "0", "supercooled_segment_min_K": "0",
        "supercooled_segment_max_K": "0", "construction_tier": "0",
        "glass_join_temperature_K": "0",
    }


def _pack_reference_gas_key(fields: list[str]) -> tuple[str, str, str, str, str, str]:
    """Identity of one packaged gas row, independent of float text."""
    return (
        fields[0],
        fields[1],
        fields[2],
        format(float(fields[6]), ".15g"),
        format(float(fields[7]), ".15g"),
        fields[16],
    )


def _pack_reference_gas_lines(path: Path) -> dict[tuple[str, str, str, str, str, str], str]:
    """Exact CSV lines of gas rows the FC87 pack translation evaluates."""
    if not path.is_file():
        raise ValueError(f"{path}: packaged gas table is missing")
    saved: dict[tuple[str, str, str, str, str, str], str] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        if not line:
            continue
        fields = next(csv.reader([line]))
        if fields[0] not in PACK_REFERENCE_GAS:
            continue
        key = _pack_reference_gas_key(fields)
        if key in saved:
            raise ValueError(f"{path}: duplicate pack-reference gas row {key}")
        saved[key] = line
    missing = PACK_REFERENCE_GAS - {key[0] for key in saved}
    if missing:
        raise ValueError(f"{path}: pack-reference gas rows missing: {sorted(missing)}")
    return saved


def _restore_pack_reference_gas_lines(
    path: Path, saved: dict[tuple[str, str, str, str, str, str], str]
) -> None:
    """Put the packaged K(g) and O(g) lines back after the table write."""
    lines = path.read_text(encoding="utf-8").splitlines()
    restored = [lines[0]]
    seen: set[tuple[str, str, str, str, str, str]] = set()
    for line in lines[1:]:
        if not line:
            continue
        fields = next(csv.reader([line]))
        if fields[0] in PACK_REFERENCE_GAS:
            key = _pack_reference_gas_key(fields)
            original = saved.get(key)
            if original is None:
                raise ValueError(f"{path}: pack-reference gas row {key} has no snapshot")
            restored.append(original)
            seen.add(key)
        else:
            restored.append(line)
    missing = set(saved) - seen
    if missing:
        raise ValueError(f"{path}: pack-reference gas rows dropped: {sorted(missing)}")
    path.write_text("\n".join(restored) + "\n", encoding="utf-8")


def _record_pack_reference_gas_open_joins(path: Path) -> None:
    """Append discontinuous joins of the frozen FC87 gas basis."""
    _OPEN_GAS_JOINS.clear()
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        if row["species_name"] not in PACK_REFERENCE_GAS:
            continue
        groups.setdefault(
            (row["species_name"], row["state"], row["Ref"]), []
        ).append(row)
    for members in groups.values():
        members.sort(key=lambda item: float(item["T_min"]))
        for left, right in zip(members, members[1:]):
            if abs(float(left["T_max"]) - float(right["T_min"])) > 1e-4:
                continue
            temperature = float(left["T_max"])
            if not _shomate_join_open(left, right, temperature):
                continue
            left_h, left_s = _stored_shomate_hs(left, temperature)
            right_h, right_s = _stored_shomate_hs(right, temperature)
            gap_g = abs((right_h - left_h) * 1000.0 - temperature * (right_s - left_s))
            _OPEN_GAS_JOINS.append(
                f"{left['species_name']} {left['Ref']} at {temperature:g} K: "
                f"|dG|={gap_g:.6g} J/mol "
                f"|dS|={abs(right_s - left_s):.6g} J/mol/K; "
                "left open because a shipped pack's FC87 translation uses these rows"
            )


def _pack_reference_row_key(fields: list[str]) -> tuple[str, str, str, str, str]:
    """Identity of one packaged liquid row, independent of float text."""
    return (
        fields[0],
        fields[1],
        format(float(fields[5]), ".15g"),
        format(float(fields[6]), ".15g"),
        fields[13],
    )


def _pack_reference_liquid_lines(path: Path) -> dict[tuple[str, str, str, str, str], str]:
    """Exact CSV lines of liquids a shipped pack uses as a reference."""
    saved: dict[tuple[str, str, str, str, str], str] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        if not line:
            continue
        fields = next(csv.reader([line]))
        if fields[0] not in PACK_REFERENCE_LIQUIDS:
            continue
        key = _pack_reference_row_key(fields)
        if key in saved:
            raise ValueError(f"{path}: duplicate pack-reference row {key}")
        saved[key] = line
    return saved


def _restore_pack_reference_liquid_lines(
    path: Path, saved: dict[tuple[str, str, str, str, str], str]
) -> None:
    """Put the packaged reference-liquid lines back after a validity rewrite."""
    if not saved:
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    restored = [lines[0]]
    seen: set[tuple[str, str, str, str, str]] = set()
    for line in lines[1:]:
        if not line:
            continue
        fields = next(csv.reader([line]))
        if fields[0] in PACK_REFERENCE_LIQUIDS:
            key = _pack_reference_row_key(fields)
            original = saved.get(key)
            if original is None:
                raise ValueError(f"{path}: pack-reference row {key} has no snapshot")
            restored.append(original)
            seen.add(key)
        else:
            restored.append(line)
    missing = set(saved) - seen
    if missing:
        raise ValueError(f"{path}: pack-reference rows dropped: {sorted(missing)}")
    path.write_text("\n".join(restored) + "\n", encoding="utf-8")


def _record_pack_reference_open_joins(path: Path) -> None:
    """Append discontinuous joins of the frozen reference liquids.

    Seam size is the packaged |ΔG| and |ΔS|. Those rows are not refit, so the
    report does not quote a donor residual.
    """
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        if row["species_name"] not in PACK_REFERENCE_LIQUIDS:
            continue
        groups.setdefault(
            (row["species_name"], row["state"], row["Ref"]), []
        ).append(row)
    for members in groups.values():
        members.sort(key=lambda item: float(item["T_min"]))
        for left, right in zip(members, members[1:]):
            if abs(float(left["T_max"]) - float(right["T_min"])) > 1e-4:
                continue
            temperature = float(left["T_max"])
            if not _condensate_join_open(left, right, temperature):
                continue
            left_g, left_s = _condensate_row_g_s(left, temperature)
            right_g, right_s = _condensate_row_g_s(right, temperature)
            _OPEN_CONDENSATE_JOINS.append(
                f"{left['species_name']} {left['Ref']} at {temperature:g} K: "
                f"|dG|={abs(right_g - left_g):.6g} J/mol "
                f"|dS|={abs(right_s - left_s):.6g} J/mol/K; "
                "left open because a shipped pack uses this liquid as a reference"
            )


def merge_condensate_csv(rows: list[dict[str, str]], output: Path) -> None:
    """Replace generated intervals only; retain unrelated source intervals."""
    lines = output.read_text(encoding="utf-8").splitlines()
    if not lines or tuple(lines[0].split(",")) != CONDENSATE_COLUMNS:
        raise ValueError(f"{output} does not have the condensate header")
    coefficient_columns = ("dH298_R", *(f"dG_{name}" for name in "ABCDE"))
    existing_coefficients = {}
    for line in lines[1:]:
        existing = next(csv.reader([line]))
        key = (
            existing[0], existing[1], existing[2], existing[3], existing[4],
            float(existing[5]), float(existing[6]), existing[13],
        )
        existing_coefficients[key] = dict(
            zip(coefficient_columns, (existing[7], *existing[8:13]))
        )
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
            not in (REPLACED_DEFAULT_CONDENSATE_SPECIES - PACK_REFERENCE_LIQUIDS)
            and not (
                fields[0] in ALL_SUPERCOOLED_LIQUID_SPECIES
                and fields[13].endswith(SUPERCOOLED_LIQUID_REF_SUFFIX)
                and fields[0] not in generated_continuations
            )
        ),
    ]
    for species_name, species_rows in grouped_rows.items():
        if species_name in PACK_REFERENCE_LIQUIDS:
            continue
        serialized_rows = []
        for row in species_rows:
            row = row.copy()
            key = (
                row["species_name"], row["state"], row["cation"],
                row["cat_num"], row["oxy_num"], float(row["T_min"]),
                float(row["T_max"]), row["Ref"],
            )
            refit_low_oxide = (
                row["species_name"] in {
                    "K2O(l)", "SiO2(l)", "Al2O3(l)", "MgO(l)", "CaO(l)"
                }
                and float(row["T_min"]) == 1000.0
                and float(row["T_max"]) == 1200.0
            )
            if not refit_low_oxide and row.get("_seam_constrained") != "true":
                row.update(existing_coefficients.get(key, {}))
            buffer = io.StringIO()
            csv.writer(buffer, lineterminator="\n").writerow(
                [{**_empty_source_validity_fields(), **row}.get(column, "") for column in CONDENSATE_COLUMNS]
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
            replaced_interval |= generated_refs.get(fields[13]) == 1
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
    parser.add_argument(
        "--high-temperature-thermo",
        type=Path,
        help="build opt-in rough 3000–6000 K candidate tables from NASA CEA thermo.inp",
    )
    parser.add_argument(
        "--high-temperature-gas-output", type=Path,
        default=repository / "src/openimcc/data/gas/gas-shomate-hight-candidate.csv",
    )
    parser.add_argument(
        "--high-temperature-oxide-output", type=Path,
        default=repository / "src/openimcc/data/gas/condensate-hight-candidate.csv",
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
    if args.high_temperature_thermo is not None:
        report = build_high_temperature_candidates(
            args.high_temperature_thermo,
            repository / "src/openimcc/data/gas/gas-shomate.csv",
            repository / "src/openimcc/data/gas/condensate.csv",
            args.high_temperature_gas_output,
            args.high_temperature_oxide_output,
        )
        for species, temperatures in report.items():
            for temperature, values in temperatures.items():
                if values is None:
                    print(f"{species}\t{temperature:g}\tno source card; FeO Cp is held constant")
                    continue
                dh, ds, dg = values
                print(f"{species}\t{temperature:g}\t|dH|={dh:.6f} J/mol\t|dS|={ds:.6f} J/mol/K\t|dG|={dg:.6f} J/mol")
        return
    if args.condensate_output is None and args.output == packaged_gas:
        args.condensate_output = repository / "src/openimcc/data/gas/condensate.csv"
    rows = build_rows(args.source_dir, args.nasa_source_dir, args.lh84_source_dir)
    # A scratch --output must still copy the packaged K(g) and O(g) lines.
    # Updating the packaged file in place snapshots those lines before write.
    gas_reference_source = (
        args.output if args.output.resolve() == packaged_gas.resolve() else packaged_gas
    )
    pack_reference_gas_lines = _pack_reference_gas_lines(gas_reference_source)
    write_csv(rows, args.output)
    condensate_rows = build_condensate_rows(args.source_dir, args.nasa_source_dir)
    pack_reference_lines: dict[tuple[str, str, str, str, str], str] = {}
    if args.condensate_output is not None:
        pack_reference_lines = _pack_reference_liquid_lines(args.condensate_output)
        merge_condensate_csv(condensate_rows, args.condensate_output)
    if args.condensate_output is not None:
        markers = build_source_liquid_markers(args.source_dir)
        additional_condensate_csvs: tuple[Path, ...] = ()
        if args.output == packaged_gas:
            legacy_condensate = repository / "src/openimcc/data/gas/condensate-feo-janaf-legacy.csv"
            published_condensate = repository / "src/openimcc/data/packs/sf04-published/condensate.csv"
            additional_condensate_csvs = (legacy_condensate, published_condensate)
        additional_snapshots = {
            path: path.read_bytes() for path in additional_condensate_csvs
        }
        validity_rows = build_source_liquid_validity_rows(
            args.output, args.condensate_output, markers=markers,
            additional_condensate_csvs=additional_condensate_csvs,
        )
        write_source_validity_columns(args.output, validity_rows, phase="gas")
        write_source_validity_columns(args.condensate_output, validity_rows, phase="condensate")
        for path in additional_condensate_csvs:
            write_source_validity_columns(path, validity_rows, phase="condensate")
        for path, snapshot in additional_snapshots.items():
            if path.read_bytes() != snapshot:
                path.write_bytes(snapshot)
        _restore_pack_reference_liquid_lines(
            args.condensate_output, pack_reference_lines
        )
        _record_pack_reference_open_joins(args.condensate_output)
    _restore_pack_reference_gas_lines(args.output, pack_reference_gas_lines)
    _record_pack_reference_gas_open_joins(args.output)
    if _CLOSED_CONDENSATE_JOINS:
        print(
            f"{len(_CLOSED_CONDENSATE_JOINS)} condensate join(s) closed:",
            file=sys.stderr,
        )
        for line in _CLOSED_CONDENSATE_JOINS:
            print(line, file=sys.stderr)
    if _OPEN_CONDENSATE_JOINS:
        print(
            f"{len(_OPEN_CONDENSATE_JOINS)} condensate join(s) left open:",
            file=sys.stderr,
        )
        for line in _OPEN_CONDENSATE_JOINS:
            print(line, file=sys.stderr)
    if _OPEN_GAS_JOINS:
        print(
            f"{len(_OPEN_GAS_JOINS)} gas join(s) left open:",
            file=sys.stderr,
        )
        for line in _OPEN_GAS_JOINS:
            print(line, file=sys.stderr)
    if _OPEN_CONDENSATE_JOINS or _OPEN_GAS_JOINS:
        raise SystemExit(1)
    for row in [*rows, *condensate_rows]:
        if "_max_residual_J_per_mol" not in row:
            continue
        print(
            f"{row['species_name']}: {row['_max_residual_J_per_mol']} J/mol, "
            f"{row['_max_residual_log10_K']} log10 K"
        )


if __name__ == "__main__":
    main()
