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


def test_crystal_closure_units_controls_and_default_regression():
    from openimcc.diagnostics import crystal_closure, load_crystal_dataset
    from openimcc.model import load_datapack

    pack = load_datapack()
    rows, details = crystal_closure(pack)
    by_phase = {row["phase"]: row for row in rows}
    for row in rows:
        tm = row["melting_point_K"]
        assert row["SI_at_melting_point"] == pytest.approx(
            row["residual_at_melting_point_kJ_mol"] * 1000
            / (8.31446261815324 * tm * math.log(10)), abs=1e-12,
        )
        assert row["delta_log10_vapor_product_at_melting_point"] == pytest.approx(
            row["SI_at_melting_point"], abs=1e-12,
        )
        node = row["in_range_node"]
        assert node["SI"] == pytest.approx(
            node["residual_kJ_mol"] * 1000 / (8.31446261815324 * node["temperature_K"] * math.log(10)),
            abs=1e-12,
        )
        assert node["delta_log10_vapor_product"] == pytest.approx(node["SI"], abs=1e-12)
    assert by_phase["forsterite"]["root_K"] == pytest.approx(1799, abs=1)
    assert by_phase["forsterite"]["SI_positive_below_root"] is True
    assert by_phase["anorthite"]["in_range_node"]["residual_kJ_mol"] == pytest.approx(-11.79, abs=0.02)
    assert all(control["activity"] == pytest.approx(1.0) for control in details["pure_oxide_controls"])
    assert all(control["SI"] == pytest.approx(0.0, abs=1e-12)
               for control in details["pure_oxide_controls"])
    eutectic = details["An42Di58_1547K"]
    assert eutectic["anorthite"]["SI"] == pytest.approx(-0.203, abs=0.001)
    assert eutectic["diopside"]["SI"] == pytest.approx(-0.744, abs=0.001)
    assert details["condensate_table_digest"]
    data = load_crystal_dataset()
    parameters = {phase["id"]: phase["berman_parameters"]
                  for phase in data["phases"] if phase.get("berman_parameters") is not None}
    for phase_parameters in parameters.values():
        assert phase_parameters["HfPrTr_J_mol"] != 0
        assert phase_parameters["SPrTr_J_mol_K"] != 0
        assert phase_parameters["k0"] != 0
    assert parameters["pseudowollastonite"]["HfPrTr_J_mol"] == -1627427.0
    assert parameters["pseudowollastonite"]["k0"] == 141.16
    differences = [p["rh_crosscheck"]["crystal_to_oxide_leg_difference_kJ_mol"]
                   for p in data["phases"] if p["rh_crosscheck"].get("crystal_to_oxide_leg_difference_kJ_mol") is not None]
    assert differences == pytest.approx([0.621, -0.675, -0.45], abs=0.01)


def test_crystal_fusion_function_refuses_out_of_source_range():
    from openimcc.diagnostics import _interpolate_crystal_fusion_nodes, load_crystal_dataset

    phase = {"id": "test", "fusion_gibbs_nodes_kJ_mol": {"1500": 10, "1800": 4}}
    assert _interpolate_crystal_fusion_nodes(phase, 1650) == pytest.approx(7)
    with pytest.raises(ValueError, match="no source node"):
        _interpolate_crystal_fusion_nodes(phase, 1800.1)
    for sourced_phase in load_crystal_dataset()["phases"]:
        upper = max(map(float, sourced_phase["fusion_gibbs_nodes_kJ_mol"]))
        _interpolate_crystal_fusion_nodes(sourced_phase, upper)
        with pytest.raises(ValueError, match="no source node"):
            _interpolate_crystal_fusion_nodes(sourced_phase, upper + 0.01)


def test_crystal_closure_cli_writes_pack_and_condensate_identities(tmp_path):
    from openimcc.model import load_datapack

    assert main(["diagnose", "crystal-closure", "--out", str(tmp_path)]) == 0
    with (tmp_path / "crystal-closure.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 14
    assert "delta_log10_vapor_product_at_melting_point" in rows[0]
    receipt = (tmp_path / "crystal-closure.md").read_text(encoding="utf-8")
    pack = load_datapack()
    assert pack.model_id in receipt
    assert pack.binding_digest in receipt
    assert "Condensate table digest:" in receipt
