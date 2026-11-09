"""Measured redox-sweep rows supporting class-scoped trace envelopes.

Columns are (class, calibrant, melt, temperature K, log10(fO2/bar),
trace mole fraction, major oxide,
absolute log10 activity shift in dex, k contribution in dex per weighted
trace mole fraction). The oxygen grid is ΔIW = -2, 0, +2, +4 at each
temperature, converted to absolute log10(fO2/bar) with the 1-bar IW expression
log10(fO2) = 6.57 - 27215/T. Each doped result uses its modelled calibrant
associates and is compared with the same base melt with that oxide dropped and
the base renormalised. For both calibrants nu=1, so each contribution is shift
/ (2*x), in dex per unit mole fraction; multiplying k by 2*x returns dex.
The winning Mn row gives 5.2233 * 0.02 = 0.1045 dex, matching its stored
shift. The table stores the winning row from the complete sweep for each class;
the recompute test checks every grid point and verifies those maxima.
"""

ENVELOPE_T_RANGE_K = (1400.0, 2200.0)
ENVELOPE_DELTA_IW_GRID = (-2.0, 0.0, 2.0, 4.0)
ENVELOPE_T_GRID_K = (1400.0, 1600.0, 1800.0, 2000.0, 2200.0)


def _iw_log10_fO2_bar(temperature_K: float) -> float:
    return 6.57 - 27215.0 / temperature_K


ENVELOPE_LOG10_FO2_RANGE_BAR = (
    min(
        _iw_log10_fO2_bar(temperature_K) + delta_iw
        for temperature_K in ENVELOPE_T_GRID_K
        for delta_iw in ENVELOPE_DELTA_IW_GRID
    ),
    max(
        _iw_log10_fO2_bar(temperature_K) + delta_iw
        for temperature_K in ENVELOPE_T_GRID_K
        for delta_iw in ENVELOPE_DELTA_IW_GRID
    ),
)

# Each row is (class, calibrant, melt, T_K, log10_fO2_bar, x_trace, major,
# shift_dex, k_contribution).
CALIBRATION_ROWS = (
    # class, calibrant, melt, T_K, log10(fO2/bar), x, major, shift, k
    (
        "basic_monoxide", "MnO", "mare basalt", 1400.0,
        -14.869285714285713, 0.01, "Na2O", 0.10446689634778394,
        5.223344817389197,
    ),
    (
        "amphoteric", "Cr2O3", "mare basalt", 1400.0,
        -14.869285714285713, 0.001, "Na2O", 0.006149588488552737,
        3.0747942442763687,
    ),
)

BASIC_MONOXIDE_ENVELOPE_K = max(
    row[8] for row in CALIBRATION_ROWS if row[0] == "basic_monoxide"
)
AMPHOTERIC_ENVELOPE_K = max(
    row[8] for row in CALIBRATION_ROWS if row[0] == "amphoteric"
)
