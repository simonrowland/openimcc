#!/usr/bin/env python3
"""Build the independent Schaefer & Fegley (2004) reference CSVs.

The only numerical inputs in this file are transcriptions from SF04 Table 5
and Table 9, plus seed values for traces in the 600 dpi raster of Fig. 10.
Each seed selects one local ink stroke; the builder measures that stroke's
centroid at the requested x column and regenerates the CSV value from the
recorded axis calibration every time this script runs.

The script does not read the old workbook fixture or any VapoRock checkout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np


# The paper PDF is not distributed. Pass it with --pdf, or set OPENIMCC_SF04_PDF.
PAPER_DEFAULT = Path(os.environ["OPENIMCC_SF04_PDF"]) if os.environ.get("OPENIMCC_SF04_PDF") else None
FIGURE_PAGE = 12  # printed SF04 p. 227; the article starts on PDF page 1.
GRID_TEMPERATURES = (1750, 1875, 1900, 2000, 2125, 2250, 2375)
TRACE_TOLERANCE_DEX = 0.05
TRACE_X_HALF_WIDTH_PX = 2
INK_THRESHOLD = 128
# Clean, unambiguous curves are 4–5 dark pixels thick in the 600 dpi raster.
# The two-pixel band admits antialiasing and small rasterization changes while
# rejecting the much thicker label/crossing blobs.
TRACE_LINE_WIDTH_MIN_PX = 3
TRACE_LINE_WIDTH_MAX_PX = 7
# Eight columns on each side span more than the compact labels and glyph
# fragments near a seed.  A curve may move by at most two pixels per column;
# its complete 17-column path must also stay within a two-pixel straight-line
# residual, which allows the observed gentle curve slope.
TRACE_CONTINUITY_HALF_WIDTH_PX = 8
TRACE_CONTINUITY_MAX_STEP_PX = 2
TRACE_CONTINUITY_MAX_RESIDUAL_PX = 2


# Table 5, p. 224.  Keep printed precision as strings.  The letter in
# ``other_note`` is the printed footnote marker, not a guessed oxide value.
TABLE5_ROWS = (
    {
        "rock": "tho",
        "rock_name": "Tholeiite",
        "SiO2_wt_pct": "50.71",
        "MgO_wt_pct": "4.68",
        "Al2O3_wt_pct": "14.48",
        "TiO2_wt_pct": "1.70",
        "Fe2O3_wt_pct": "4.89",
        "FeO_wt_pct": "9.07",
        "CaO_wt_pct": "8.83",
        "Na2O_wt_pct": "3.16",
        "K2O_wt_pct": "0.77",
        "Other_wt_pct": "1.62",
        "other_note": "a",
        "Total_wt_pct": "99.91",
        "Tliq_K": "1433",
        "printed_reference": "[1]",
    },
    {
        "rock": "aba",
        "rock_name": "Alkali basalt",
        "SiO2_wt_pct": "44.80",
        "MgO_wt_pct": "11.07",
        "Al2O3_wt_pct": "13.86",
        "TiO2_wt_pct": "1.96",
        "Fe2O3_wt_pct": "2.91",
        "FeO_wt_pct": "9.63",
        "CaO_wt_pct": "10.16",
        "Na2O_wt_pct": "3.19",
        "K2O_wt_pct": "1.09",
        "Other_wt_pct": "1.45",
        "other_note": "b",
        "Total_wt_pct": "100.12",
        "Tliq_K": "1504",
        "printed_reference": "[2]",
    },
    {
        "rock": "kom",
        "rock_name": "Barberton-type komatiite",
        "SiO2_wt_pct": "47.10",
        "MgO_wt_pct": "29.60",
        "Al2O3_wt_pct": "4.04",
        "TiO2_wt_pct": "0.24",
        "Fe2O3_wt_pct": "12.80",
        "FeO_wt_pct": "0",
        "CaO_wt_pct": "5.44",
        "Na2O_wt_pct": "0.46",
        "K2O_wt_pct": "0.09",
        "Other_wt_pct": "0.27",
        "other_note": "c",
        "Total_wt_pct": "100.04",
        "Tliq_K": "1838",
        "printed_reference": "[3]",
    },
    {
        "rock": "dun",
        "rock_name": "Dunite",
        "SiO2_wt_pct": "40.20",
        "MgO_wt_pct": "43.20",
        "Al2O3_wt_pct": "0.80",
        "TiO2_wt_pct": "0.20",
        "Fe2O3_wt_pct": "1.90",
        "FeO_wt_pct": "11.90",
        "CaO_wt_pct": "0.80",
        "Na2O_wt_pct": "0.30",
        "K2O_wt_pct": "0.10",
        "Other_wt_pct": "0.70",
        "other_note": "d",
        "Total_wt_pct": "100.10",
        "Tliq_K": "1954",
        "printed_reference": "[4]",
    },
    {
        "rock": "cai",
        "rock_name": "Allende type B1 CAI",
        "SiO2_wt_pct": "29.10",
        "MgO_wt_pct": "10.20",
        "Al2O3_wt_pct": "29.60",
        "TiO2_wt_pct": "1.30",
        "Fe2O3_wt_pct": "0",
        "FeO_wt_pct": "0.60",
        "CaO_wt_pct": "28.80",
        "Na2O_wt_pct": "0.18",
        "K2O_wt_pct": "0.10",
        "Other_wt_pct": "0.10",
        "other_note": "e",
        "Total_wt_pct": "100.01",
        "Tliq_K": "1823",
        "printed_reference": "[5]",
    },
)


# Table 9, p. 234.  The parenthesized exponent notation is preserved exactly
# as printed in ``flux_cm2_s``.  The pressure is calculated below, never
# transcribed from another source.
TABLE9_FLUXES = (
    ("Na", "7.67(18)"),
    ("O2", "1.58(18)"),
    ("e-", "2.42(17)"),
    ("O", "1.76(17)"),
    ("Fe", "1.29(16)"),
    ("FeO", "6.45(15)"),
    ("SiO", "1.19(16)"),
    ("NaO", "6.20(15)"),
    ("K", "1.22(15)"),
    ("Na+", "1.15(15)"),
    ("SiO2", "9.26(14)"),
    ("Mg", "3.49(14)"),
    ("K+", "2.37(13)"),
)


# Fig. 10, p. 227.  Each tuple is (left, right, top, bottom) in the 600 dpi
# PGM rendered by pdftoppm.  The plot axes are calibrated to x=1700..2400 K
# and log10(x_mole)=0..-5.  Borders are specified by their centre rows/cols.
PANEL_CALIBRATION = {
    "tho": (805.5, 2149.0, 679.5, 2021.5),
    "aba": (2856.5, 4200.0, 679.5, 2021.5),
    "kom": (805.5, 2149.0, 2482.0, 3823.5),
    "dun": (2856.5, 4200.0, 2514.5, 3862.5),
    "cai": (1821.5, 3166.0, 4276.5, 5620.5),
}


# Recorded log10 seeds, in the order GRID_TEMPERATURES. A seed selects the
# intended local stroke; the emitted value is measured from that stroke's
# centroid. None means that the trace was clipped by the -5 plot floor or was
# not unambiguous at the grid temperature. Values are intentionally a partial,
# defensible digitization; the paper says species below 10^-5 are not shown.
FIGURE_LOG10_TRACES = {
    "tho": {
        "Na": (-0.197, -0.197, -0.197, -0.201, -0.212, -0.231, -0.255),
        "O2": (-0.797, -0.805, -0.805, -0.808, -0.816, -0.820, -0.827),
        "O": (-2.107, -1.924, -1.889, -1.760, -1.617, -1.492, -1.395),
        "SiO": (-3.102, -2.925, -2.828, -2.459, -2.299, -2.113, -1.962),
        "Fe": (-3.102, -2.794, -2.737, -2.528, -2.299, -2.472, -2.351),
        "FeO": (-3.450, -3.361, -2.977, -2.811, -2.629, -2.774, -2.5795),
        "NaO": (-3.268, -3.023, -2.978, -3.139, -3.174, -2.865, -2.6562),
        "SiO2": (-4.383, -4.532, -3.7973, -3.530, -3.642, -3.5568, -3.327),
        "K": (-3.886, -3.821, -3.7973, -3.8127, -3.718, -3.683, -3.400),
        "e-": (-4.195, -3.933, -3.925, -3.7502, -3.642, -3.5032, -3.659),
        "Na+": (-4.383, -3.987, -3.947, -3.8127, -3.828, -3.718, -3.660),
        "Mg": (None, -4.534, -4.454, -4.156, -3.828, -3.5032, -3.327),
        "TiO2": (None, None, None, None, None, -4.477, -4.449),
        "MgO": (None, None, None, None, None, -4.7220, None),
    },
    "aba": {
        "Na": (-0.197, -0.197, -0.197, -0.196, -0.199, -0.209, -0.216),
        "O2": (-0.792, -0.795, -0.799, -0.805, -0.812, -0.820, -0.827),
        "O": (-2.304, -2.113, -2.077, -1.937, -1.785, -1.652, -1.541),
        "SiO": (-4.314, -3.865, -3.657, -3.184, -2.984, -2.420, -2.079),
        "Fe": (-3.407, -3.3632, -3.316, -3.089, -2.824, -2.5636, -2.437),
        "FeO": (-3.6515, -3.4280, -3.3677, -3.260, -2.984, -2.813, -2.871),
        "NaO": (-3.210, -3.378, -3.1138, -2.949, -2.757, -2.6399, -2.6657),
        "K": (-3.940, -3.865, -3.851, -3.798, -3.765, -3.539, -3.526),
        "e-": (-4.314, -4.024, -3.990, -3.981, -3.8570, -3.431, -3.631),
        "Na+": (-4.396, -4.1968, -4.124, -4.149, -3.808, -3.6326, -3.663),
        "SiO2": (None, -4.575, -4.542, -4.218, -3.8570, -3.539, -3.141),
        "Mg": (None, -4.627, -4.642, -4.149, -3.765, -3.7059, -3.526),
        "TiO2": (None, None, None, None, None, -4.767, -4.786),
        "Na2": (None, None, None, None, None, None, -4.663),
        "MgO": (None, None, None, None, None, -4.713, -4.193),
    },
    "kom": {
        "Na": (-0.196, -0.196, -0.196, -0.199, -0.211, -0.227, -0.248),
        "O2": (-0.796, -0.805, -0.807, -0.814, -0.820, -0.826, -0.835),
        "O": (-2.091, -1.901, -1.865, -1.733, -1.586, -1.457, -1.353),
        "SiO": (None, -3.386, -3.341, -3.231, -2.870, -2.562, -2.2813),
        "Fe": (-3.053, -2.997, -2.954, -2.704, -2.505, -2.452, -2.2813),
        "FeO": (-3.235, -3.187, -3.088, -3.168, -2.976, -2.814, -2.672),
        "Mg": (-3.627, -3.651, -3.561, -3.7538, -3.438, -3.338, -2.814),
        "NaO": (-3.725, -3.876, -3.925, -3.231, -3.608, -3.4000, -3.353),
        "e-": (-4.120, -4.219, -4.132, -4.670, -4.243, -3.686, -3.645),
        "Na+": (-4.180, -3.964, -3.861, -4.674, -3.738, -3.822, -3.709),
        "K": (-3.960, -3.964, -3.925, -3.8235, -3.608, -3.4000, -3.466),
        "SiO2": (-4.676, None, -4.132, -4.728, -3.774, -3.822, -3.703),
        "MgO": (None, None, None, -4.730, -4.243, -3.686, -3.645),
    },
    "dun": {
        "Na": (-0.195, -0.200, -0.200, -0.200, -0.204, -0.208, -0.215),
        "O2": (-0.792, -0.796, -0.796, -0.799, -0.809, -0.818, -0.827),
        "O": (-2.342, -2.146, -2.111, -1.968, -1.810, -1.673, -1.556),
        "K": (-2.795, -2.702, -2.684, -2.617, -2.537, -2.468, -2.3762),
        "NaO": (-3.501, -3.392, -3.338, -3.181, -3.000, -2.825, -2.680),
        "FeO": (-3.368, -3.079, -3.086, -2.919, -2.864, -2.8867, -2.680),
        "Fe": (-3.561, -3.1363, -3.340, -3.114, -3.242, -3.622, -3.489),
        "Mg": (-3.704, -3.391, -3.338, -3.181, -2.990, -2.825, -2.680),
        "e-": (-4.223, -4.050, -3.963, -3.948, -3.774, -3.622, -3.489),
        "Na+": (-4.284, -4.143, -4.2202, -4.075, -3.891, -3.730, -3.6196),
        "SiO": (-4.525, -4.382, -4.2892, -4.523, -3.774, -3.622, -3.489),
        "SiO2": (None, -4.681, -4.649, -4.705, -4.379, -4.822, -4.605),
        "MgO": (None, -4.280, -4.2892, -4.876, -3.935, -3.935, -3.542),
        "K+": (None, None, None, None, -4.379, -4.251, -4.138),
        "KO": (None, None, None, None, None, -4.822, -4.742),
        "Na2": (None, None, None, None, None, None, -4.743),
    },
    "cai": {
        "Na": (-0.197, -0.197, -0.197, -0.201, -0.205, -0.216, -0.227),
        "O2": (-0.796, -0.807, -0.807, -0.817, -0.831, -0.843, -0.863),
        "O": (-2.085, -1.888, -1.851, -1.708, -1.549, -1.410, -1.289),
        "SiO": (-3.631, -3.397, -3.354, -3.1686, -2.714, -2.210, -1.903),
        "Mg": (None, -3.951, -3.897, -3.744, -3.341, -2.865, -2.041),
        "NaO": (-3.631, -3.6414, -3.579, -3.629, -3.564, -3.560, -2.855),
        "FeO": (-4.1300, -4.221, -4.169, -4.310, -4.249, -3.7656, -3.718),
        "Fe": (-4.1907, -3.7062, -3.897, -3.969, -3.748, -3.8381, -3.382),
        "e-": (-4.245, -4.074, -4.053, -4.542, -3.8718, -3.7656, -3.445),
        "Na+": (-4.310, -3.951, -3.897, -4.615, -3.9233, -3.8381, -3.718),
        "K": (-4.498, -4.626, -4.650, -4.745, -4.249, -3.8381, -3.718),
        "SiO2": (None, -4.736, None, -4.745, None, -3.7656, -3.718),
        "MgO": (None, None, None, None, None, None, None),
        "TiO2": (None, None, None, None, None, None, -4.725),
    },
}


# A handful of points lie under a curve label or at a crossing.  They are
# retained only when the line can be followed from both sides; the larger
# uncertainty records that judgement.  All other included points use 0.05
# dex, the target resolution stated in the brief.
HIGH_UNCERTAINTY = {
    ("tho", "NaO", 1900),
    ("kom", "NaO", 2000),
    ("kom", "Na+", 2000),
    ("dun", "Fe", 1900),
    ("dun", "SiO", 2000),
    ("cai", "Na+", 2000),
}


# These seeds are retained in the source as the before-values for the review
# record, but the local raster cannot identify one curve with the stated
# thickness and continuity checks.  They are omitted rather than assigned to a
# species by proximity alone.  Existing ``None`` entries in
# FIGURE_LOG10_TRACES are the original floor/label exclusions documented in
# the README.  The reason strings keep every post-check exclusion auditable.
TRACE_EXCLUDED: dict[tuple[str, str, int], str] = {
    ("aba", "Fe", 1900): "two strokes in the local measurement window",
    ("aba", "NaO", 1875): "two strokes in the local measurement window",
    ("aba", "K", 2125): "two strokes in the local measurement window",
    ("aba", "Na+", 2125): "two strokes in the local measurement window",
    ("aba", "Mg", 2125): "two strokes in the local measurement window",
    ("dun", "MgO", 2000): "two strokes in the local measurement window",
    ("tho", "TiO2", 2375): "two strokes at the calibrated column",
    ("aba", "Fe", 2125): "two strokes at the calibrated column",
    ("aba", "Fe", 2375): "two strokes at the calibrated column",
    ("dun", "SiO", 1750): "two strokes at the calibrated column",
    ("cai", "NaO", 1900): "two strokes at the calibrated column",
    ("cai", "NaO", 2250): "two strokes at the calibrated column",
    ("tho", "K", 2000): "vertical thickness 8 px; expected 3–7 px",
    ("tho", "e-", 1750): "vertical thickness 8 px; expected 3–7 px",
    ("tho", "Na+", 1875): "vertical thickness 8 px; expected 3–7 px",
    ("tho", "Na+", 2000): "vertical thickness 8 px; expected 3–7 px",
    ("aba", "NaO", 1750): "vertical thickness 8 px; expected 3–7 px",
    ("aba", "e-", 1875): "vertical thickness 9 px; expected 3–7 px",
    ("aba", "Na+", 1750): "vertical thickness 9 px; expected 3–7 px",
    ("aba", "Na+", 1900): "vertical thickness 13 px; expected 3–7 px",
    ("aba", "Mg", 1900): "vertical thickness 8 px; expected 3–7 px",
    ("kom", "Fe", 2125): "vertical thickness 29 px; expected 3–7 px",
    ("kom", "Mg", 2000): "vertical thickness 8 px; expected 3–7 px",
    ("kom", "Mg", 2250): "vertical thickness 12 px; expected 3–7 px",
    ("kom", "Na+", 1875): "vertical thickness 8 px; expected 3–7 px",
    ("kom", "Na+", 2375): "vertical thickness 9 px; expected 3–7 px",
    ("kom", "K", 1875): "vertical thickness 8 px; expected 3–7 px",
    ("kom", "SiO2", 2000): "vertical thickness 8 px; expected 3–7 px",
    ("kom", "SiO2", 2375): "vertical thickness 9 px; expected 3–7 px",
    ("kom", "MgO", 2000): "vertical thickness 8 px; expected 3–7 px",
    ("dun", "Fe", 1750): "vertical thickness 9 px; expected 3–7 px",
    ("dun", "e-", 1750): "vertical thickness 10 px; expected 3–7 px",
    ("dun", "Na+", 2375): "vertical thickness 8 px; expected 3–7 px",
    ("dun", "MgO", 2250): "vertical thickness 8 px; expected 3–7 px",
    ("cai", "Mg", 1900): "vertical thickness 11 px; expected 3–7 px",
    ("cai", "Fe", 1750): "vertical thickness 9 px; expected 3–7 px",
    ("cai", "Fe", 1900): "vertical thickness 11 px; expected 3–7 px",
    ("cai", "Na+", 1900): "vertical thickness 11 px; expected 3–7 px",
    ("cai", "Na+", 2000): "vertical thickness 9 px; expected 3–7 px",
    ("cai", "K", 1875): "vertical thickness 13 px; expected 3–7 px",
    ("tho", "O", 2375): "no thin path at horizontal offset −5",
    ("tho", "e-", 1900): "no thin path at horizontal offset −2",
    ("tho", "Na+", 1900): "no thin path at horizontal offset −2",
    ("tho", "TiO2", 2250): "no thin path at horizontal offset +7",
    ("aba", "FeO", 2375): "no thin path at horizontal offset −1",
    ("aba", "e-", 1900): "no thin path at horizontal offset −7",
    ("aba", "e-", 2000): "no thin path at horizontal offset −3",
    ("aba", "Na+", 1875): "no thin path at horizontal offset +7",
    ("aba", "TiO2", 2250): "no thin path at horizontal offset −2",
    ("aba", "TiO2", 2375): "no thin path at horizontal offset −3",
    ("aba", "Na2", 2375): "no thin path at horizontal offset −2",
    ("aba", "MgO", 2375): "no thin path at horizontal offset +5",
    ("kom", "SiO", 2375): "no thin path at horizontal offset −6",
    ("kom", "Fe", 2375): "no thin path at horizontal offset −6",
    ("kom", "NaO", 1900): "no thin path at horizontal offset −2",
    ("kom", "NaO", 2250): "no thin path at horizontal offset −4",
    ("kom", "e-", 2000): "no thin path at horizontal offset +4",
    ("kom", "Na+", 1750): "no thin path at horizontal offset −1",
    ("kom", "Na+", 2000): "no thin path at horizontal offset +4",
    ("kom", "K", 1900): "no thin path at horizontal offset −2",
    ("kom", "K", 2000): "no thin path at horizontal offset −3",
    ("kom", "K", 2250): "no thin path at horizontal offset −4",
    ("dun", "NaO", 1750): "no thin path at horizontal offset −1",
    ("dun", "NaO", 1900): "no thin path at horizontal offset +1",
    ("dun", "FeO", 1875): "no thin path at horizontal offset −2",
    ("dun", "Fe", 1900): "no thin path at horizontal offset +1",
    ("dun", "Mg", 1900): "no thin path at horizontal offset +1",
    ("dun", "Na+", 1750): "no thin path at horizontal offset −3",
    ("dun", "SiO2", 2125): "no thin path at horizontal offset +5",
    ("dun", "MgO", 2375): "no thin path at horizontal offset −5",
    ("dun", "K+", 2125): "no thin path at horizontal offset +5",
    ("cai", "SiO", 2000): "no thin path at horizontal offset +1",
    ("cai", "SiO", 2250): "no thin path at horizontal offset −3",
    ("cai", "Mg", 1875): "no thin path at horizontal offset −5",
    ("cai", "Mg", 2375): "no thin path at horizontal offset −6",
    ("cai", "FeO", 1750): "no thin path at horizontal offset −8",
    ("cai", "FeO", 2250): "no thin path at horizontal offset +7",
    ("cai", "Fe", 2250): "no thin path at horizontal offset +7",
    ("cai", "e-", 2000): "no thin path at horizontal offset −3",
    ("cai", "e-", 2250): "no thin path at horizontal offset +7",
    ("cai", "Na+", 1875): "no thin path at horizontal offset −5",
    ("cai", "Na+", 2250): "no thin path at horizontal offset +7",
    ("cai", "K", 2250): "no thin path at horizontal offset +7",
    ("cai", "SiO2", 1875): "no thin path at horizontal offset +5",
    ("cai", "SiO2", 2250): "no thin path at horizontal offset +7",
}


# Molar masses used by the molecular-flux form of SF04 Eq. 11.  They are the
# formula masses needed to turn a molecular count flux into a molecule mass;
# no workbook or gas-table value enters this conversion.
MOLECULAR_MASS_G_MOL = {
    "Na": 22.98976928,
    "O2": 31.9988,
    "e-": 0.000548579909,
    "O": 15.9994,
    "Fe": 55.845,
    "FeO": 71.844,
    "SiO": 44.0849,
    "NaO": 38.9892,
    "K": 39.0983,
    "Na+": 22.98976928,
    "SiO2": 60.0843,
    "Mg": 24.305,
    "K+": 39.0983,
}
AVOGADRO = 6.02214076e23
BOLTZMANN_J_K = 1.380649e-23
TABLE9_T_K = 1900.0


def _parse_printed_flux(value: str) -> float:
    mantissa, exponent = value.split("(")
    return float(mantissa) * 10.0 ** int(exponent.rstrip(")"))


def _eq11_pressure_bar(flux_cm2_s: str, species: str) -> float:
    """Return Eq. 11 pressure in bar from the printed molecular flux.

    Hertz--Knudsen premise: with alpha_s=1, a molecular flux J obeys

        J = P / sqrt(2*pi*m*k*T).

    Therefore P = J*sqrt(2*pi*m*k*T).  The molecule mass m is the formula
    molar mass M divided by Avogadro's number and converted from g/mol to
    kg/mol.  The printed J is in cm^-2 s^-1, so J*1e4 is m^-2 s^-1; the
    resulting pressure is Pa and is divided by 1e5 to report bar.  Unit check:
    (m^-2 s^-1)*sqrt(kg*J) = kg m^-1 s^-2 = Pa, then Pa/1e5 = bar.

    Sanity value: the printed Na flux 7.67(18) at 1900 K gives
    6.08409e-5 bar.  Summing all 13 converted cells gives 7.72825e-5 bar,
    against Table 7's printed 7.72e-5 bar for tholeiite.
    """
    molecule_mass_kg = MOLECULAR_MASS_G_MOL[species] / 1000.0 / AVOGADRO
    flux_m2_s = _parse_printed_flux(flux_cm2_s) * 1.0e4
    pressure_pa = flux_m2_s * math.sqrt(
        2.0 * math.pi * molecule_mass_kg * BOLTZMANN_J_K * TABLE9_T_K
    )
    return pressure_pa / 1.0e5


def _read_pgm(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        if handle.readline().strip() != b"P5":
            raise ValueError(f"{path} is not a binary PGM")
        line = handle.readline()
        while line.startswith(b"#"):
            line = handle.readline()
        width, height = (int(part) for part in line.split())
        max_value = int(handle.readline())
        if max_value != 255:
            raise ValueError(f"{path} has unsupported max value {max_value}")
        pixels = np.fromfile(handle, dtype=np.uint8, count=width * height)
    if pixels.size != width * height:
        raise ValueError(f"{path} is truncated")
    return pixels.reshape(height, width)


def _render_page(pdf: Path, work_dir: Path) -> np.ndarray:
    stem = work_dir / "sf04-page-227"
    subprocess.run(
        [
            "pdftoppm",
            "-f",
            str(FIGURE_PAGE),
            "-l",
            str(FIGURE_PAGE),
            "-r",
            "600",
            "-gray",
            "-singlefile",
            str(pdf),
            str(stem),
        ],
        check=True,
    )
    return _read_pgm(stem.with_suffix(".pgm"))


def _contiguous_runs(rows: np.ndarray) -> list[list[int]]:
    runs: list[list[int]] = []
    for row in rows.tolist():
        if not runs or row > runs[-1][-1] + 1:
            runs.append([int(row)])
        else:
            runs[-1].append(int(row))
    return runs


def _dark_runs_at_column(
    pixels: np.ndarray, column: int, row_start: int, row_stop: int
) -> list[list[int]]:
    dark_rows = np.flatnonzero(
        pixels[row_start : row_stop + 1, column] < INK_THRESHOLD
    ) + row_start
    return _contiguous_runs(dark_rows)


def _follow_curve(
    pixels: np.ndarray, column: int, center_run: list[int]
) -> tuple[dict[int, float] | None, str | None]:
    """Follow a thin stroke on both sides of the calibrated column."""
    center_row = (center_run[0] + center_run[-1]) / 2.0
    path = {0: center_row}
    for direction in (-1, 1):
        previous_row = center_row
        for offset in range(
            direction,
            direction * (TRACE_CONTINUITY_HALF_WIDTH_PX + 1),
            direction,
        ):
            search_half_width = (
                TRACE_CONTINUITY_MAX_STEP_PX + TRACE_LINE_WIDTH_MAX_PX
            )
            row_start = max(
                0, math.floor(previous_row - search_half_width)
            )
            row_stop = min(
                pixels.shape[0] - 1,
                math.ceil(previous_row + search_half_width),
            )
            thin_runs = []
            for run in _dark_runs_at_column(
                pixels, column + offset, row_start, row_stop
            ):
                run_center = (run[0] + run[-1]) / 2.0
                if (
                    TRACE_LINE_WIDTH_MIN_PX <= len(run) <= TRACE_LINE_WIDTH_MAX_PX
                    and abs(run_center - previous_row)
                    <= TRACE_CONTINUITY_MAX_STEP_PX
                ):
                    thin_runs.append(run)
            if not thin_runs:
                return None, (
                    "no thin stroke at horizontal offset "
                    f"{offset:+d} (expected within "
                    f"±{TRACE_CONTINUITY_MAX_STEP_PX} px of the curve)"
                )
            if len(thin_runs) != 1:
                return None, (
                    f"{len(thin_runs)} thin strokes at horizontal offset "
                    f"{offset:+d}; curve assignment is ambiguous"
                )
            run = thin_runs[0]
            previous_row = (run[0] + run[-1]) / 2.0
            path[offset] = previous_row

    offsets = np.arange(
        -TRACE_CONTINUITY_HALF_WIDTH_PX,
        TRACE_CONTINUITY_HALF_WIDTH_PX + 1,
        dtype=float,
    )
    rows = np.array([path[int(offset)] for offset in offsets])
    fit = np.polyfit(offsets, rows, 1)
    residual = float(np.max(np.abs(rows - np.polyval(fit, offsets))))
    if residual > TRACE_CONTINUITY_MAX_RESIDUAL_PX:
        return None, (
            f"curve path bends {residual:.1f} px from its local straight-line "
            f"slope (limit {TRACE_CONTINUITY_MAX_RESIDUAL_PX} px)"
        )
    return path, None


def _trace_pixel_rows(pixels: np.ndarray) -> dict[tuple[str, str, int], float]:
    """Measure one unambiguous curve stroke at each recorded trace seed.

    At the calibrated x column, the seed selects a stroke inside the stated
    ±0.05 dex y window.  The stroke must be 3–7 dark pixels thick and follow
    one thin path across eight columns on both sides.  The returned row is the
    centroid of all dark pixels in that accepted stroke across x±2 px.  A
    missing, thick, discontinuous, or multiply-stroked window is a build
    error; the corresponding seed must be corrected or omitted instead of
    silently accepting a nearby frame or species.
    """
    if pixels.shape != (6617, 4934):
        raise ValueError(f"SF04 p. 227 raster changed shape: {pixels.shape}")
    traced: dict[tuple[str, str, int], float] = {}
    for rock, curves in FIGURE_LOG10_TRACES.items():
        x0, x1, y0, y1 = PANEL_CALIBRATION[rock]
        for species, values in curves.items():
            if len(values) != len(GRID_TEMPERATURES):
                raise ValueError(f"{rock}/{species} trace has wrong grid length")
            for T_K, log10_x in zip(GRID_TEMPERATURES, values):
                key = (rock, species, T_K)
                if log10_x is None or key in TRACE_EXCLUDED:
                    continue
                x = round(x0 + (T_K - 1700) / 700.0 * (x1 - x0))
                seed_row = y0 - (float(log10_x) / 5.0) * (y1 - y0)
                if not (y0 <= seed_row <= y1):
                    raise ValueError(f"{rock}/{species}/{T_K} leaves plot")

                row_delta = (TRACE_TOLERANCE_DEX + 0.02) / 5.0 * (y1 - y0)
                row_start = max(0, math.ceil(seed_row - row_delta))
                row_stop = min(pixels.shape[0] - 1, math.floor(seed_row + row_delta))
                x_start = max(0, x - TRACE_X_HALF_WIDTH_PX)
                x_stop = min(pixels.shape[1] - 1, x + TRACE_X_HALF_WIDTH_PX)
                window = pixels[row_start : row_stop + 1, x_start : x_stop + 1]
                dark_rows = np.flatnonzero(np.any(window < INK_THRESHOLD, axis=1)) + row_start
                runs = _contiguous_runs(dark_rows)
                candidates = []
                for run in runs:
                    stroke = pixels[run[0] : run[-1] + 1, x_start : x_stop + 1]
                    dark_pixels = np.argwhere(stroke < INK_THRESHOLD)
                    centroid_row = run[0] + float(dark_pixels[:, 0].mean())
                    centroid_log10_x = -5.0 * (centroid_row - y0) / (y1 - y0)
                    if abs(centroid_log10_x - float(log10_x)) <= TRACE_TOLERANCE_DEX:
                        candidates.append((run, centroid_row))
                if not candidates:
                    raise ValueError(
                        f"{rock}/{species}/{T_K} has no raster stroke within "
                        f"±{TRACE_TOLERANCE_DEX:.2f} dex of seed"
                    )
                if len(candidates) != 1:
                    raise ValueError(
                        f"{rock}/{species}/{T_K} has {len(candidates)} raster strokes "
                        f"within ±{TRACE_TOLERANCE_DEX:.2f} dex of seed"
                    )
                candidate_run, candidate_centroid = candidates[0]
                center_runs = [
                    center_run
                    for center_run in _dark_runs_at_column(
                        pixels, x, 0, pixels.shape[0] - 1
                    )
                    if center_run[-1] >= candidate_run[0]
                    and center_run[0] <= candidate_run[-1]
                ]
                if not center_runs:
                    raise ValueError(
                        f"{rock}/{species}/{T_K} has no raster stroke at "
                        "the calibrated column"
                    )
                if len(center_runs) != 1:
                    raise ValueError(
                        f"{rock}/{species}/{T_K} has {len(center_runs)} raster "
                        "strokes at the calibrated column; curve assignment "
                        "is ambiguous"
                    )
                center_run = center_runs[0]
                line_width = len(center_run)
                if not (
                    TRACE_LINE_WIDTH_MIN_PX
                    <= line_width
                    <= TRACE_LINE_WIDTH_MAX_PX
                ):
                    raise ValueError(
                        f"{rock}/{species}/{T_K} has vertical ink thickness "
                        f"{line_width} px at the calibrated column; expected "
                        f"{TRACE_LINE_WIDTH_MIN_PX}–{TRACE_LINE_WIDTH_MAX_PX} px "
                        "for a curve"
                    )
                _path, failure = _follow_curve(pixels, x, center_run)
                if failure is not None:
                    raise ValueError(
                        f"{rock}/{species}/{T_K} is not a continuous curve: "
                        f"{failure}"
                    )
                traced[key] = candidate_centroid
    return traced


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build(output_dir: Path, pdf: Path) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sf04-reference-") as work_dir:
        pixels = _render_page(pdf, Path(work_dir))
    trace_rows = _trace_pixel_rows(pixels)

    composition_fields = [
        "rock",
        "rock_name",
        "SiO2_wt_pct",
        "MgO_wt_pct",
        "Al2O3_wt_pct",
        "TiO2_wt_pct",
        "Fe2O3_wt_pct",
        "FeO_wt_pct",
        "CaO_wt_pct",
        "Na2O_wt_pct",
        "K2O_wt_pct",
        "Other_wt_pct",
        "other_note",
        "Total_wt_pct",
        "Tliq_K",
        "printed_reference",
        "method_class",
        "source_locator",
    ]
    composition_rows = [
        {
            **row,
            "method_class": "transcribed",
            "source_locator": "Schaefer & Fegley (2004), Table 5, p. 224",
        }
        for row in TABLE5_ROWS
    ]
    _write_csv(output_dir / "compositions.csv", composition_fields, composition_rows)

    table9_fields = [
        "rock",
        "species",
        "T_K",
        "flux_cm2_s",
        "flux_method_class",
        "partial_pressure_bar",
        "pressure_method_class",
        "source_locator",
        "pressure_source_locator",
    ]
    table9_rows = []
    for species, flux in TABLE9_FLUXES:
        table9_rows.append(
            {
                "rock": "tho",
                "species": species,
                "T_K": "1900",
                "flux_cm2_s": flux,
                "flux_method_class": "transcribed",
                "partial_pressure_bar": f"{_eq11_pressure_bar(flux, species):.12g}",
                "pressure_method_class": "derived_eq11",
                "source_locator": "Schaefer & Fegley (2004), Table 9, p. 234",
                "pressure_source_locator": "Schaefer & Fegley (2004), Eq. 11, pp. 233-234",
            }
        )
    _write_csv(output_dir / "table9_anchors.csv", table9_fields, table9_rows)

    figure_fields = [
        "rock",
        "species",
        "T_K",
        "log10_x",
        "log10_x_uncertainty_dex",
        "method_class",
        "log10_p_bar",
        "pressure_method_class",
        "source_locator",
        "pressure_source_locator",
    ]
    total_pressure_fit = {
        "tho": (4.719, -16761),
        "aba": (4.716, -16037),
        "kom": (4.480, -16404),
        "dun": (4.618, -15724),
        "cai": (3.890, -15373),
    }
    figure_rows = []
    for rock, curves in FIGURE_LOG10_TRACES.items():
        A, B = total_pressure_fit[rock]
        _x0, _x1, y0, y1 = PANEL_CALIBRATION[rock]
        for species, values in curves.items():
            for T_K, log10_x in zip(GRID_TEMPERATURES, values):
                if log10_x is None or (rock, species, T_K) in TRACE_EXCLUDED:
                    continue
                # Emit the value recovered from the calibrated raster row,
                # rather than the decimal transcription used to seed the
                # trace.  This keeps the PDF/calibration path authoritative.
                row = trace_rows[(rock, species, T_K)]
                log10_x = -5.0 * (row - y0) / (y1 - y0)
                log10_p = float(log10_x) + A + B / T_K
                uncertainty = 0.10 if (rock, species, T_K) in HIGH_UNCERTAINTY else 0.05
                figure_rows.append(
                    {
                        "rock": rock,
                        "species": species,
                        "T_K": str(T_K),
                        "log10_x": f"{float(log10_x):.3f}",
                        "log10_x_uncertainty_dex": f"{uncertainty:.2f}",
                        "method_class": "digitized_figure",
                        "log10_p_bar": f"{log10_p:.6f}",
                        "pressure_method_class": "derived_table7",
                        "source_locator": (
                            "Schaefer & Fegley (2004), Fig. 10, "
                            f"panel {chr(ord('a') + ('tho', 'aba', 'kom', 'dun', 'cai').index(rock))}, p. 227"
                        ),
                        "pressure_source_locator": "Schaefer & Fegley (2004), Table 7, p. 226",
                    }
                )
    _write_csv(output_dir / "fig10_digitized.csv", figure_fields, figure_rows)
    return {
        "compositions": len(composition_rows),
        "table9_anchors": len(table9_rows),
        "fig10_digitized": len(figure_rows),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, default=PAPER_DEFAULT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/references/schaefer-fegley-2004"),
    )
    args = parser.parse_args()
    if args.pdf is None:
        parser.error("pass --pdf PATH or set OPENIMCC_SF04_PDF (the paper is not distributed)")
    if not args.pdf.is_file():
        parser.error(f"paper PDF not found: {args.pdf}")
    counts = build(args.output_dir, args.pdf)
    for name, count in counts.items():
        print(f"{name}: {count}")
    for name in ("compositions.csv", "table9_anchors.csv", "fig10_digitized.csv"):
        print(f"sha256 {name}: {_sha256(args.output_dir / name)}")


if __name__ == "__main__":
    main()
