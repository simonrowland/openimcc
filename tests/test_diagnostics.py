from __future__ import annotations

import math
import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from openimcc import diagnostics
from openimcc.cli import main
from openimcc.diagnostics import implied_heat_capacity, run_cp_implied, score_against_calorimetry


def _score_fixture(curve):
    comps = []
    rows = []
    for i, x in enumerate((0.0, 0.3, 0.7, 1.0)):
        name = f"c{i}"
        comps.append({"id": name, "x_coordinate": x, "measured_cp_J_mol_K": 100.0,
                      "pure_end_member": x == 1.0})
        for temp in (900.0, 1200.0):
            excess = 2.0 + 3.0 * x + (curve * x * (1.0 - x))
            rows.append({"dataset_id": name, "temperature_K": temp,
                         "cp_conf_J_mol_K": excess,
                         "cp_total_J_mol_K": 100.0 + excess})
    dataset = {"compositions": comps, "envelope": {"relative": 0.005},
               "temperature_independent": True, "temperature_interval_K": [900, 1200]}
    return rows, dataset


def test_linear_excess_is_absorbed_and_quadratic_is_detected():
    rows, dataset = _score_fixture(0.0)
    linear = score_against_calorimetry(rows, dataset)
    assert max(abs(row["residual_J_mol_K"]) for row in linear["curvature"]) < 1e-12
    assert max(row["max_deviation_J_mol_K"] for row in linear["temperature_flatness"]) == 0
    assert all(row["verdict"].startswith("(d)") for row in linear["pure_end_member_controls"])

    rows, dataset = _score_fixture(8.0)
    quadratic = score_against_calorimetry(rows, dataset)
    assert max(abs(row["residual_J_mol_K"]) for row in quadratic["curvature"]) > 0.1


def test_one_association_finite_difference_matches_closed_form(monkeypatch):
    a, b = 0.3, 1200.0
    pack = SimpleNamespace(kernel_datapack=SimpleNamespace(B=np.asarray([b]), reactions=("AB",)))

    def fake_evaluate(composition, temperature, pack, **kwargs):
        k = 10.0 ** (a + b / temperature)
        z = math.sqrt(1.0 + k) - 1.0
        extent = z / (1.0 + z)
        return SimpleNamespace(
            complex_x=np.asarray([extent / (2.0 - extent)]),
            D=2.0 / (2.0 - extent), extrapolated=False,
            labels=SimpleNamespace(flags=(), notices=()),
        )

    monkeypatch.setattr(diagnostics, "evaluate", fake_evaluate)
    temp = 1400.0
    row = implied_heat_capacity(pack, {"SiO2": 1.0, "Na2O": 1.0}, [temp])[0]
    k = 10.0 ** (a + b / temp)
    root = math.sqrt(1.0 + k)
    z = root - 1.0
    dy_dk = 1.0 / (2.0 * root * (1.0 + z) ** 2)
    dk_dt = k * math.log(10.0) * (-b / temp**2)
    expected = (-diagnostics.R * math.log(10.0) * b) * 0.5 * dy_dk * dk_dt
    assert row["cp_conf_J_mol_K"] == pytest.approx(expected, rel=2e-7)
    assert row["step_check_delta_J_mol_K"] == pytest.approx(0.0, abs=3e-7)


def test_no_association_has_zero_configurational_cp(monkeypatch):
    pack = SimpleNamespace(kernel_datapack=SimpleNamespace(B=np.asarray([]), reactions=()))
    monkeypatch.setattr(diagnostics, "evaluate", lambda *a, **kw: SimpleNamespace(
        complex_x=np.asarray([]), D=1.0, extrapolated=False,
        labels=SimpleNamespace(flags=(), notices=())))
    row = implied_heat_capacity(pack, {"SiO2": 1.0}, [1200.0])[0]
    assert row["cp_conf_J_mol_K"] == 0.0
    assert row["cp_total_J_mol_K"] == row["cp_parent_J_mol_K"]


def test_step_size_converges_and_default_pack_regression():
    pack, rows, scores = run_cp_implied()
    assert max(abs(row["step_check_delta_J_mol_K"]) for row in rows) < 1e-5
    expected = {
        "Na2Si2O5": (1.337, 1.062, 0.729),
        "Na2SiO3": (-1.167, -0.975, -0.701),
    }
    for composition, values in expected.items():
        actual = [next(item["residual_J_mol_K"] for item in scores["curvature"]
                       if item["composition"] == composition and item["temperature_K"] == temp)
                  for temp in (1323.0, 1500.0, 1700.0)]
        assert actual == pytest.approx(values, abs=0.001)
    assert pack.model_id == "IMCC-SF04-D066"


def test_pure_end_member_has_zero_curvature():
    rows, dataset = _score_fixture(0.0)
    scores = score_against_calorimetry(rows, dataset)
    pure = [row for row in scores["curvature"] if row["composition"] == "c3"]
    assert all(abs(row["residual_J_mol_K"]) < 1e-12 for row in pure)


def test_cli_csv_contains_flatness_and_evaluation_metadata(tmp_path):
    assert main(["diagnose", "cp-implied", "--out", str(tmp_path)]) == 0
    with (tmp_path / "cp-implied.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert rows
    assert all(row["temperature_flatness_verdict"].startswith("(c)") for row in rows)
    assert all(row["temperature_flatness_max_deviation_J_mol_K"] for row in rows)
    assert all(row["extrapolated"] in {"True", "False"}
               and row["paper_demonstrated_window"] in {"True", "False"}
               for row in rows)
    silica = next(row for row in rows
                  if row["dataset_id"] == "SiO2" and row["temperature_K"] == "1500.0")
    assert silica["pure_control_verdict"] == "(d) STANDARD-STATE OFFSET"
    assert silica["curvature_verdict"].startswith("(b)")
    sodium_1323 = next(row for row in rows
                       if row["dataset_id"] == "Na2Si2O5" and row["temperature_K"] == "1323.0")
    assert sodium_1323["extrapolated"] == "True"
    notices = json.loads(sodium_1323["notices"])
    assert any(notice.startswith("temperature-domain-extension:") for notice in notices)
    assert any(notice.startswith("low-temperature-uncertain:") for notice in notices)
    sodium_1700 = next(row for row in rows
                       if row["dataset_id"] == "Na2Si2O5" and row["temperature_K"] == "1700.0")
    assert sodium_1700["extrapolated"] == "False"
    assert sodium_1700["paper_demonstrated_window"] == "True"
