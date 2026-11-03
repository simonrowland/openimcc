from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import median

from openimcc import evaluate_redox


ROOT = Path(__file__).resolve().parents[1]
PACKS = ROOT / "src/openimcc/data/packs"
FE_HESS = PACKS / "imcc-sf04-d066-feo-hess-candidate.json"
BANYA = ROOT / "docs/feo-liquid-banya80-fetO-sio2.csv"
_MOLAR_MASS = {"FeO": 71.844, "Fe2O3": 159.687, "SiO2": 60.0843}


def _iron_saturated_inventory(row: dict[str, str]) -> dict[str, float]:
    oxide_moles = {
        oxide: float(row[f"{oxide}_wt_percent"]) / molar_mass
        for oxide, molar_mass in _MOLAR_MASS.items()
    }
    return {
        "Fe": oxide_moles["FeO"] + 2 * oxide_moles["Fe2O3"] + 1.0,
        "O": oxide_moles["FeO"] + 3 * oxide_moles["Fe2O3"] + 2 * oxide_moles["SiO2"],
        "Si": oxide_moles["SiO2"],
    }


def test_banya80_sio2_median_activity_residual_improves_with_hess_pack() -> None:
    with BANYA.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12

    medians = []
    for melt_pack_path in (None, FE_HESS):
        residuals = []
        for row in rows:
            result = evaluate_redox(
                _iron_saturated_inventory(row),
                float(row["temperature_C"]) + 273.15,
                "closed",
                melt_pack_path=melt_pack_path,
            )
            assert result.metal_buffered
            residuals.append(
                abs(
                    math.log10(
                        result.parent_activity["FeO"]
                        / float(row["a_FetO_reported"])
                    )
                )
            )
        medians.append(median(residuals))

    default_median, hess_median = medians
    assert 0.10 < default_median < 0.16
    assert 0.015 < hess_median < 0.06
    assert hess_median + 0.04 < default_median
