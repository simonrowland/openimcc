from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "src/openimcc/data/packs/imcc-sf04-d066-p2o5-candidate-v1.json"
SOURCES = ROOT / "data-src/p2o5-candidate/thermochemistry.json"


@pytest.mark.parametrize("row_index", (38, 39))
def test_phosphate_associate_equilibrium_constants_are_favourable_and_bounded(
    row_index: int,
) -> None:
    pack = json.loads(PACK.read_text(encoding="utf-8"))
    row = pack["rows"][row_index]

    assert all(
        row["A"] + row["B"] / temperature > 0.0
        for temperature in (1700, 2000, 2300)
    )
    assert abs(row["A"] * 8.31446261815324 * math.log(10.0)) < 300.0


def test_magnesium_phosphate_crystal_formation_enthalpy_hess_check() -> None:
    source = json.loads(SOURCES.read_text(encoding="utf-8"))
    mg_rows = source["janaf_pages"]["Mg-038"]["values"]
    mg_hf = next(row[5] for row in mg_rows if row[0] == 298.15)

    mgo = yaml.safe_load(
        (ROOT / "data-src/janaf/Mg-008.yaml").read_text(encoding="utf-8")
    )["table"]
    mgo_hf = next(
        row["formation_enthalpy"]["value"]
        for row in mgo["values"]
        if row["temperature"]["value"] == 298.15
    )
    p2o5_crystal = source["janaf_reference_checks"]["p2o5_crystal_298"]
    p2o5_hf = (
        p2o5_crystal["formation_enthalpy_kJ_mol"]
        / p2o5_crystal["p2o5_formula_units_per_table_formula"]
    )
    delta_h = mg_hf - 3.0 * mgo_hf - p2o5_hf

    # JANAF prints no numeric uncertainties for these entries; this locks the
    # printed-value cycle (±0.01 kJ/mol) and its rounded approximately -435 kJ/mol check.
    assert delta_h == pytest.approx(-436.407, abs=0.01)
    assert abs(delta_h - (-435.0)) < 2.0
