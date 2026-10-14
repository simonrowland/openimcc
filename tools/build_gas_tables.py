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
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Keep package imports available when this file is invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


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
KONINGS_BATCH_ELEMENTS = ("La", "Ce", "Pr", "Nd")
KONINGS_GAS_SPECIES = {
    "La": (("La(g)", 0), ("LaO(g)", 1)),
    "Ce": (("Ce(g)", 0), ("CeO(g)", 1), ("CeO2(g)", 2)),
    "Pr": (("Pr(g)", 0), ("PrO(g)", 1)),
    "Nd": (("Nd(g)", 0), ("NdO(g)", 1)),
}
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
)
TRACE_NASA_GAS_SOURCES = (
    ("NiO(g)", "NG-0926", "NiO", "Ni", 1, 1),
    ("Ag(g)", "NG-0074", "Ag", "Ag", 1, 0),
    ("RbO(g)", "NG-1329", "RbO", "Rb", 1, 1),
    ("Rb2O(g)", "NG-1352", "Rb2O", "Rb", 2, 1),
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
)
TRACE_PUBLISHED_GAS_SOURCES = (
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
    ("MnO(g)", "TSIV-1436", "MnO", "Mn", 1, 1, 6000.0),
    ("CoO(g)", "TSIV-1335", "CoO", "Co", 1, 1, 3000.0),
)
TRACE_GAS_SPECIES = frozenset(
    {source[0].removesuffix("(g)") for source in TRACE_JANAF_GAS_SOURCES}
    | {source[0].removesuffix("(g)") for source in TRACE_JANAF_YAML_GAS_SOURCES}
    | {source[0].removesuffix("(g)") for source in TRACE_NASA_GAS_SOURCES}
    | {source[0].removesuffix("(g)") for source in TRACE_PUBLISHED_GAS_SOURCES}
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
TRACE_NASA_CONDENSATE_SOURCES = (
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
        ((1200.0, 1500.0), (1500.0, 1800.0)),
    ),
    (
        "Al2O3(l)", "Al-100", "Al", 2, 3, 2500.0,
        ((1200.0, 1500.0), (1500.0, 2000.0), (2000.0, 2500.0)),
    ),
    (
        "MgO(l)", "Mg-009", "Mg", 1, 1, 2200.0,
        ((1200.0, 1500.0), (1500.0, 1800.0), (1800.0, 2200.0)),
    ),
    (
        "CaO(l)", "Ca-028", "Ca", 1, 1, 2200.0,
        ((1200.0, 1500.0), (1500.0, 1800.0), (1800.0, 2200.0)),
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
    # CEA prints rounded coefficients; their 298 K polynomial evaluation can
    # miss the separately printed Hf anchor by a few joules after cancellation.
    # Hinc below is still zero at 298 K by construction. Mo/W oxide polymer
    # coefficients miss their separate Hf anchors by up to 20.24 J/mol
    # (NG-1258), so allow 21 J for rounding but reject material mismatches.
    # NASA-7 coefficients are printed to finite precision. Retain the exact
    # published formation anchor and subtract the polynomial's own H(298),
    # while allowing the small coefficient-rounding residual found in cards.
    if abs(h298_j - reference_h) > 21.0:
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
            "_max_residual_log10_K": number(fit["max_residual_log10_K"])}


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
                dh, ds = _konings_integral(current["coefficients"], current["powers"], T, transition)
                H += dh; S += ds; T = transition
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
        fit = _fit_condensate_values(species, f"Konings-{species}", cation,
                                     cat_num, oxy_num, source_rows(nodes),
                                     float(item["dfH298_kJ_mol"]),
                                     fit_t_min=lo, fit_t_max=hi,
                                     runtime_t_min=lo, runtime_t_max=hi,
                                     fit_temperatures=nodes)
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
    The band sums absolute contributions and excludes supercooling energy.
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
        bound += float(step["temperature_unc_K"]) * abs(dGdT)
    return bound


def _published_table_record(source_dir: Path, table_id: str) -> dict[str, Any]:
    """Read one transcribed record from its named published table bundle."""
    if table_id in {"P672-HfO2-c", "P672-Hf-g"}:
        filename = "pankratz-hf.json"
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
        source_limit = max(rows)
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
    temperatures = [
        float(T)
        for T in np.arange(fit_t_min, fit_t_max + JANAF_GRID_STEP_K / 2.0, JANAF_GRID_STEP_K)
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
    }


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
    allow_missing_nodes: bool = False,
    fit_temperatures: Sequence[float] | None = None,
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
    degree = 4 if fit_temperatures is None else min(4, len(temperatures) - 1)
    fit_design = np.column_stack([tau**power for power in range(degree + 1)])
    fitted_coefficients = np.linalg.lstsq(
        fit_design, phi / R_J_MOL_K, rcond=None
    )[0]
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
    }


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
    record = _load_record(nasa_source_dir / f"{table_id}.json")
    if (
        record.get("record_id") != table_id
        or record.get("formula") != formula
        or record.get("phase") not in {"liquid", "L"}
    ):
        raise ValueError(f"{table_id}: NASA record identity does not match {species_name}")
    temperatures = [
        float(T)
        for T in np.arange(fit_t_min, fit_t_max + 50.0, NASA_GRID_STEP_K)
    ]
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
    nasa_source_dir: Path | None = None,
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
        record = _load_record(nasa_source_dir / f"{table_id}.json")
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
        record = _load_record(nasa_source_dir / f"{table_id}.json")
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

    tau = np.asarray(fit_temperatures, dtype=float) / 1000.0
    design = np.column_stack((np.ones_like(tau), tau, tau**2, tau**3, tau**4))
    if use_cp_constrained_fit:
        heat_capacity_design = np.column_stack(
            (2.0 * tau, 6.0 * tau**2, 12.0 * tau**3, 20.0 * tau**4)
        )
        anchor_temperature = runtime_t_max
        anchor_tau = anchor_temperature / 1000.0
        anchor_entropy = entropy_0 + cp * math.log(anchor_temperature / T0)
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
        anchor_phi = phi(anchor_temperature) / R_J_MOL_K
        constraints = anchor_design[1, 1:] - anchor_design[0, 1:]
        constraint_value = anchor_entropy / R_J_MOL_K - anchor_phi
        # G=-RT*P gives S/R=P+tau*P' and Cp/R=2*tau*P'+tau^2*P''.
        # The constraints match Gibbs energy and entropy at this interval seam.
        normal = heat_capacity_design.T @ heat_capacity_design
        target = heat_capacity_design.T @ np.full(len(tau), cp / R_J_MOL_K)
        kkt = np.block(
            [
                [normal, constraints[:, None]],
                [constraints[None, :], np.zeros((1, 1))],
            ]
        )
        coefficients_tail = np.linalg.solve(
            kkt, np.concatenate((target, [constraint_value]))
        )[:4]
        coefficients = np.concatenate(
            ([anchor_phi - anchor_tau * coefficients_tail[0]
              - anchor_tau**2 * coefficients_tail[1]
              - anchor_tau**3 * coefficients_tail[2]
              - anchor_tau**4 * coefficients_tail[3]], coefficients_tail)
        )
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
        "Ref": table_id + SUPERCOOLED_LIQUID_REF_SUFFIX,
        "_max_residual_J_per_mol": number(residual_j),
        "_max_residual_log10_K": number(residual_log10),
        "_T0_K": number(T0),
        "_Cp_l_J_molK": number(cp),
        "_max_abs_Cp_deviation_J_molK": number(max_abs_cp_deviation),
        "_anchor_H_J_per_mol": number(anchor_h),
        "_anchor_S_J_molK": number(entropy_0),
        "_anchor_G_J_per_mol": number(anchor_g),
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
                    interval=interval, fit_t_min=lo, fit_t_max=hi,
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
    for source in TRACE_PUBLISHED_GAS_SOURCES:
        split_temperature = (
            min(max(source[6], 1500.0), 2000.0)
            if source[1].startswith(("P672-Te", "P672-Se", "P672-As", "P672-Sb"))
            else 1500.0
        )
        high_fit_start = math.floor(split_temperature / 100.0) * 100.0
        rows.append(
            _fit_published_gas_row(
                published_source_dir,
                *source[:6],
                t_interval=1,
                fit_t_min=high_fit_start,
                fit_t_max=FIT_T_MAX,
                runtime_t_min=split_temperature,
                runtime_t_max=FIT_T_MAX,
                breakpoint_side="high",
            )
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
        if source[0].startswith(("Mo", "W"))
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
    for source in TRACE_JANAF_YAML_GAS_SOURCES:
        if source[0].startswith(("Mo", "W")):
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
    for source in TRACE_PUBLISHED_GAS_SOURCES:
        split_temperature = (
            min(max(source[6], 1500.0), 2000.0)
            if source[1].startswith(("P672-Te", "P672-Se", "P672-As", "P672-Sb"))
            else 1500.0
        )
        rows.append(
            _fit_published_gas_row(
                published_source_dir,
                *source[:6],
                t_interval=2,
                fit_t_min=1200.0,
                fit_t_max=split_temperature,
                runtime_t_min=1200.0,
                runtime_t_max=split_temperature,
                breakpoint_side="low",
            )
        )
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
    return rows


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
        for t, cp, entropy, h in record["nodes"]
    ]
    t0, cp0, s0, h0 = values[-1]["temperature"], None, None, None
    cp0 = float(record["nodes"][-1][1])
    s0 = float(record["nodes"][-1][2])
    h0 = float(record["nodes"][-1][3])
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
        # Keep the measured legacy fit exactly on its original 1600-2200 K
        # interval. Outside it, fit the same liquid_from_solid construction
        # on each 100 K source node; every interval remains below the 10 J/mol
        # residual gate, and the shared endpoint checks continuity.
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
    elif species in {"SeO2", "As2O3", "Sb2O3", "Ag2O"}:
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
                )
            )
        for fit_t_min, fit_t_max in fit_intervals:
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
    for species in ("ZnO", "CdO", "MnO", "SeO2", "As2O3", "Sb2O3", "Ag2O", "ThO2", "HfO2"):
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
    # The TSIV liquid branch is selected directly for NiO and CoO. Its
    # below-melt portion is the same printed constant-Cp function continued
    # as supercooled liquid; source coverage begins at the source melt marker.
    rows.extend(
        _fit_tsiv_liquid_parent(published_source_dir, "NiO", "TSIV-2026", "Ni")
    )
    rows.extend(
        _fit_tsiv_liquid_parent(published_source_dir, "CoO", "TSIV-1334", "Co")
    )
    return rows


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
    return result


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
