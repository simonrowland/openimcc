from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from openimcc import evaluate, evaluate_redox, load_datapack


ROOT = Path(__file__).resolve().parents[1]
PACKS = ROOT / "src/openimcc/data/packs"
FE_HESS = PACKS / "imcc-sf04-d066-feo-hess-candidate.json"
V5 = PACKS / "imcc-sf04-d066-v5.json"
DISTIN = ROOT / "docs/feo-liquid-distin-activities.csv"
_FE_HESS_NOTICE = (
    "feo-hess-candidate: opt-in Fe2SiO4 and FeTiO3 A-only Hess refits; "
    "open validation items are listed in docs/feo-liquid-candidate.md"
)
_ATOMIC_MASS = {"Fe": 55.845, "O": 15.999, "Si": 28.085}


def _rows(path: Path) -> dict[str, dict]:
    return {
        row["complex"]: row
        for row in json.loads(path.read_text(encoding="utf-8"))["rows"]
    }


def _without_coefficients_and_provenance(row: dict) -> dict:
    rest = dict(row)
    for key in (
        "A",
        "source",
        "derivation",
        "source_refs",
    ):
        rest.pop(key, None)
    return rest


def _distin_inventory(row: dict[str, str]) -> dict[str, float]:
    n_feo = float(row["FeO_wt_percent"]) / (_ATOMIC_MASS["Fe"] + _ATOMIC_MASS["O"])
    n_fe2o3 = float(row["Fe2O3_wt_percent"]) / (
        2 * _ATOMIC_MASS["Fe"] + 3 * _ATOMIC_MASS["O"]
    )
    n_sio2 = float(row["final_SiO2_wt_percent"]) / (
        _ATOMIC_MASS["Si"] + 2 * _ATOMIC_MASS["O"]
    )
    return {
        "Fe": n_feo + 2 * n_fe2o3 + 1.0,
        "O": n_feo + 3 * n_fe2o3 + 2 * n_sio2,
        "Si": n_sio2,
    }


def test_fe_hess_loads_and_changes_only_the_two_a_only_refits() -> None:
    default = load_datapack()
    candidate = load_datapack(FE_HESS)
    base_rows = _rows(V5)
    candidate_rows = _rows(FE_HESS)

    assert default.version == "1.0.7-d066-v5"
    assert default.kernel_datapack.binding_digest == (
        "13a185563870ec1a9c20d51bd25fb58df08c9bbbbdf7b2212b0f43ead64738cf"
    )
    assert candidate.version == "1.0.9-d066-feo-hess-candidate"
    assert candidate.kernel_datapack.binding_digest != (
        default.kernel_datapack.binding_digest
    )

    expected = {
        "Fe2SiO4": -9.744444811651487,
        "FeTiO3": -2.165573644482091,
    }
    for name, row in base_rows.items():
        actual = candidate_rows[name]
        if name in expected:
            assert actual["A"] == expected[name]
            assert actual["B"] == row["B"]
            if name == "Fe2SiO4":
                assert "standard_state_refit" not in actual
                base_row = dict(row)
                base_row.pop("standard_state_refit")
            else:
                base_row = row
            assert _without_coefficients_and_provenance(actual) == (
                _without_coefficients_and_provenance(base_row)
            )
        else:
            assert actual == row


def test_fe_hess_notice_is_identity_scoped() -> None:
    composition = {"FeO": 2.0, "SiO2": 1.0}
    default = evaluate(composition, 1800.0, pack=load_datapack(), basis_type="mol")
    candidate = evaluate(
        composition,
        1800.0,
        pack=load_datapack(FE_HESS),
        basis_type="mol",
        allow_out_of_envelope=True,
    )

    assert not any(
        notice.startswith("feo-hess-candidate:") for notice in default.labels.notices
    )
    assert [
        notice
        for notice in candidate.labels.notices
        if notice.startswith("feo-hess-candidate:")
    ] == [_FE_HESS_NOTICE]


def test_full_composition_metal_saturated_distin_row_moves_closer() -> None:
    with DISTIN.open(newline="", encoding="utf-8") as stream:
        row = next(
            item
            for item in csv.DictReader(stream)
            if item["temperature_C"] == "1960"
            and item["final_SiO2_wt_percent"] == "48.1"
        )
    inventory = _distin_inventory(row)
    temperature_K = float(row["temperature_C"]) + 273.15
    measured = float(row["a_FeO_reported"])

    errors = []
    for melt_pack_path in (None, FE_HESS):
        result = evaluate_redox(
            inventory,
            temperature_K,
            "closed",
            melt_pack_path=melt_pack_path,
        )
        assert result.metal_buffered
        errors.append(abs(math.log10(result.parent_activity["FeO"] / measured)))

    assert errors[1] < errors[0]



def test_fe_hess_distin_all_rows_median_and_mean() -> None:
    with DISTIN.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 24

    residuals_by_pack = []
    for melt_pack_path in (None, FE_HESS):
        residuals = []
        for row in rows:
            result = evaluate_redox(
                _distin_inventory(row),
                float(row["temperature_C"]) + 273.15,
                "closed",
                melt_pack_path=melt_pack_path,
            )
            assert result.metal_buffered
            residuals.append(
                math.log10(
                    result.parent_activity["FeO"] / float(row["a_FeO_reported"])
                )
            )
        residuals_by_pack.append(residuals)

    expected = ((0.148833, -0.153287), (0.018347, -0.009295))
    for residuals, (median_abs, mean_signed) in zip(residuals_by_pack, expected):
        assert np.median(np.abs(residuals)) == pytest.approx(median_abs, abs=1e-5)
        assert np.mean(residuals) == pytest.approx(mean_signed, abs=1e-5)
