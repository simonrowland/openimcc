"""Structural guards for the frozen external IMCC rung-3 workbook fixture."""

from __future__ import annotations

import csv
import math
import re
from collections import Counter
from pathlib import Path

import pytest


FIXTURE = Path("tests/fixtures/imcc_sf04_magma_workbook.csv")

# This fixture is NOT distributed with openimcc, deliberately.
#
# It is derived from Schaefer2004-MAGMA-valid.xlsx in the VapoRock data
# directory. VapoRock is AGPL-3.0 and this package is Apache-2.0, so
# redistributing a 327 KB derivative of its data here is not clearly
# permitted -- and the underlying workbook is MAGMA output, whose own
# redistribution terms we have not established.
#
# The regression is still valuable, so the test runs for anyone who has the
# file locally and skips for everyone else rather than failing. Replacing it
# with an independently sourced reference set is tracked in
# docs/PRE-EXTRACTION-CHECKLIST.md.
pytestmark = pytest.mark.skipif(
    not FIXTURE.is_file(),
    reason=(
        "MAGMA workbook fixture is not distributed (VapoRock/AGPL derivative); "
        f"place it at {FIXTURE} to run this regression"
    ),
)
SHEETS = {"tho", "aba", "kom", "dun", "bit", "cai", "cab"}
TEMPERATURES = {1500.0, 1625.0, 1750.0, 1875.0, 1900.0, 2000.0, 2125.0, 2250.0, 2375.0, 2500.0}
SPECIES = {
    "O", "O2", "Mg", "MgO", "Si", "SiO", "SiO2", "Fe", "FeO",
    "Al", "AlO", "AlO2", "Al2O", "Al2O2", "Ca", "CaO", "Na",
    "Na2", "NaO", "Na2O", "Na+", "K", "K2", "KO", "K2O", "K+",
    "Ti", "TiO", "TiO2", "e-", "Zn", "ZnO",
}
COMPOSITION_FIELDS = {
    "Material", "Source", "dIW", "SiO2", "MgO", "Al2O3", "TiO2",
    "Fe2O3", "FeO", "CaO", "Na2O", "K2O", "MnO", "H2O", "P2O5",
    "Cr2O3", "NiO",
}


def _load() -> tuple[list[str], list[dict[str, str]]]:
    comments = []
    with FIXTURE.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            comments.append(line.rstrip())
    with FIXTURE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(line for line in handle if not line.startswith("#")))
    return comments, rows


def test_imcc_rung3_fixture_provenance_is_frozen() -> None:
    comments, _ = _load()
    joined = "\n".join(comments)
    assert "Schaefer2004-MAGMA-valid.xlsx" in joined
    assert "Schaefer2004_comp_MAGMA_dIW.csv" in joined
    assert "sheets: tho,aba,kom,dun,bit,cai,cab" in joined
    assert "excluded_sheet: 1900rel2Na (derived 1900 K" in joined
    assert re.search(r"source_workbook_sha256: [0-9a-f]{64}", joined)
    assert re.search(r"source_compositions_sha256: [0-9a-f]{64}", joined)
    assert re.search(r"extraction_date_utc: \d{4}-\d{2}-\d{2}", joined)


def test_imcc_rung3_fixture_grid_and_values_are_complete() -> None:
    _, rows = _load()
    assert len(rows) == 7 * 32 * 10
    assert {row["composition_sheet"] for row in rows} == SHEETS
    assert {row["species"] for row in rows} == SPECIES
    assert {float(row["T_K"]) for row in rows} == TEMPERATURES
    assert "1900rel2Na" not in {row["composition_sheet"] for row in rows}
    assert len({(row["composition_sheet"], row["species"], row["T_K"]) for row in rows}) == len(rows)
    assert Counter(row["composition_sheet"] for row in rows) == Counter({sheet: 320 for sheet in SHEETS})
    assert COMPOSITION_FIELDS <= set(rows[0])

    for row in rows:
        pressure = float(row["workbook_pressure_bar"])
        if pressure > 0.0:
            assert math.isclose(
                float(row["log10p_bar"]),
                math.log10(pressure),
                rel_tol=0.0,
                abs_tol=2.0e-15,
            )
        else:
            assert row["log10p_bar"] == ""


def test_imcc_rung3_fixture_composition_is_constant_per_sheet() -> None:
    _, rows = _load()
    for sheet in SHEETS:
        sheet_rows = [row for row in rows if row["composition_sheet"] == sheet]
        for field in COMPOSITION_FIELDS:
            assert len({row[field] for row in sheet_rows}) == 1
