from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from openimcc import load_datapack
from openimcc.gas import R_J_MOL_K
from openimcc.redox import _active_inner, _redox_parent_names
from openimcc.redox_pack import load_redox_pack
from tools.liquid_from_solid import liquid_from_solid


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "tests/data/redox-mn-heldout.json"
REPORT_PATH = ROOT / "tests/data/redox-mn-heldout-residuals.json"
INPUTS_PATH = ROOT / "data-src/liquid-from-solid-inputs.json"
CRYSTALS_PATH = ROOT / "data-src/crystal-thermochemistry.json"
FORMULA_WEIGHT_G_MOL = {
    "MnO": 54.938044 + 15.999,
    "FeO": 55.845 + 15.999,
    "SiO2": 28.085 + 2.0 * 15.999,
    "CaO": 40.078 + 15.999,
    "MgO": 24.305 + 15.999,
    "Al2O3": 2.0 * 26.9815385 + 3.0 * 15.999,
}


def _heldout_residuals() -> list[list[float | int | str]]:
    dataset = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    inputs = json.loads(INPUTS_PATH.read_text(encoding="utf-8"))
    crystals = json.loads(CRYSTALS_PATH.read_text(encoding="utf-8"))["tables"]
    row = inputs["rows"]["MnO"]
    construction = liquid_from_solid(
        "MnO", crystals[row["crystal_table_id"]], row["inputs"]
    )
    fitted = construction.fit_condensate_row()
    loo_band = float(
        inputs["tier_error_bands"]["fusion_entropy_tier3"]["max"][0]
    )

    base = load_datapack().kernel_datapack
    pack = load_redox_pack()
    parent_names = _redox_parent_names(pack)
    manganese_parent = parent_names.index("MnO")
    report: list[list[float | int | str]] = []

    for study, point, temperature, mass_percent, printed_activity in dataset[
        "points"
    ]:
        source = dataset["sources"][study]
        oxides = source["oxide_order"]
        assert len(mass_percent) == len(oxides)
        parent_moles = np.zeros(len(parent_names), dtype=np.float64)
        for oxide, mass in zip(oxides, mass_percent):
            parent_name = str(oxide).removesuffix("(l)")
            # Premise: each charge is reported in mass percent on an arbitrary
            # 100 g basis. Algebra: n_i = mass_i / M_i. Units are mol formula
            # units; the solver normalizes parent moles, so the common 100 g
            # factor cancels. Sanity: every listed oxide maps to one parent.
            parent_moles[parent_names.index(parent_name)] += (
                float(mass) / FORMULA_WEIGHT_G_MOL[oxide]
            )

        state = _active_inner(
            parent_moles,
            float(temperature),
            0.0,
            base,
            pack,
            exclude_fe_redox=True,
            tol=1e-12,
            max_iter=100,
        )
        local_mn = int(np.flatnonzero(state.active_parent == manganese_parent)[0])
        ln_predicted = float(state.y[local_mn])
        activity_scale = 1e-3 if study == "chen_2022" else 1.0
        measured_solid_activity = float(printed_activity) * activity_scale
        delta_g_fusion = construction.fusion_gibbs_kj_mol(float(temperature))
        # Premise: mu_s° + RT ln(a_s) = mu_l° + RT ln(a_l), with
        # DeltaG_fus = G_l° - G_s°. Rearranging gives the measured liquid
        # reference activity below. Units convert kJ/mol to J/mol; at T_m the
        # fusion correction is zero and the two reported activities coincide.
        ln_converted = math.log(measured_solid_activity) - (
            delta_g_fusion * 1000.0 / (R_J_MOL_K * float(temperature))
        )
        conversion_band_kj = construction.below_fusion_band_kj_mol(
            float(temperature), loo_band
        ) + float(fitted["_max_residual_J_per_mol"]) / 1000.0
        band_ln = conversion_band_kj * 1000.0 / (
            R_J_MOL_K * float(temperature)
        )
        residual_ln = ln_predicted - ln_converted
        report.append(
            [
                study,
                int(point),
                int(temperature),
                math.exp(ln_predicted),
                math.exp(ln_converted),
                residual_ln,
                conversion_band_kj,
                band_ln,
                abs(residual_ln) / band_ln,
            ]
        )

    return report


def test_manganese_heldout_activities_are_converted_and_reported():
    report = _heldout_residuals()
    recorded = json.loads(REPORT_PATH.read_text(encoding="utf-8"))["points"]
    assert len(report) == 55
    assert len(recorded) == len(report)
    for row, saved in zip(report, recorded):
        assert row[0] in {"chen_2021", "chen_2022"}
        assert all(math.isfinite(float(value)) for value in row[3:])
        assert float(row[3]) > 0.0
        assert float(row[4]) > 0.0
        assert float(row[6]) > 0.0
        assert row[:3] == saved[:3]
        for value, expected in zip(row[3:], saved[3:]):
            # This checks that the report reproduces the calculations. It does
            # not impose any residual, activity, or within-band fit threshold.
            assert float(value) == pytest.approx(float(expected), rel=1e-10)
    print(json.dumps(report, separators=(",", ":")))
