from __future__ import annotations

import math
import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from openimcc import diagnostics
from openimcc.cli import main
from openimcc.diagnostics import (
    _association_enthalpy, _fusion_enthalpy, implied_heat_capacity,
    load_mixing_enthalpy_dataset, mixing_enthalpy, run_cp_implied,
    run_mixing_enthalpy, score_against_calorimetry,
)


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
        "Na2Si2O5": (1.242, 0.979, 0.662),
        "Na2SiO3": (-1.090, -0.901, -0.638),
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
    assert by_phase["forsterite"]["root_K"] == pytest.approx(1801, abs=1)
    assert by_phase["forsterite"]["SI_positive_below_root"] is True
    assert by_phase["anorthite"]["in_range_node"]["residual_kJ_mol"] == pytest.approx(-11.61, abs=0.02)
    assert all(control["activity"] == pytest.approx(1.0) for control in details["pure_oxide_controls"])
    assert all(control["SI"] == pytest.approx(0.0, abs=1e-12)
               for control in details["pure_oxide_controls"])
    eutectic = details["An42Di58_1547K"]
    assert eutectic["anorthite"]["SI"] == pytest.approx(-0.177, abs=0.001)
    assert eutectic["diopside"]["SI"] == pytest.approx(-0.715, abs=0.001)
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


def test_public_saturation_and_liquidus_api():
    from openimcc import (
        CrystalNotice, CrystalStatus, liquid_only_notice, liquidus_bracket,
        saturation_indices,
    )
    from openimcc.model import load_datapack

    pack = load_datapack()
    rows = saturation_indices({"SiO2": 100.0}, 1500.0, pack)
    silica = next(row for row in rows if row.phase == "quartz_beta")
    assert silica.status is CrystalStatus.AVAILABLE
    assert silica.parent_stoichiometry == {"SiO2": 1}
    assert silica.source
    assert silica.notices == ()
    assert isinstance(CrystalNotice("test", {}, pack.model_id), CrystalNotice)
    unavailable = next(row for row in saturation_indices(
        {"SiO2": 100.0}, 1200.0, pack
    ) if row.phase == "anorthite")
    assert unavailable.status is CrystalStatus.OUT_OF_RANGE
    assert unavailable.SI is None
    result = liquidus_bracket({"SiO2": 100.0}, 1323.0, 2000.0, pack)
    assert result.status is CrystalStatus.AVAILABLE
    assert result.controlling_phase == "cristobalite_beta"
    assert result.notices == ()
    assert liquid_only_notice(result.liquidus_K - 1.0, result) is not None
    assert liquid_only_notice(result.liquidus_K + 1.0, result) is None


def test_anorthite_and_forsterite_liquidus_misses_are_reported():
    from openimcc import liquidus_bracket
    from openimcc.model import load_datapack

    pack = load_datapack()
    anorthite = liquidus_bracket(
        {"CaO": 20.15, "Al2O3": 36.65, "SiO2": 43.20}, 1323.0, 1900.0, pack
    )
    forsterite = liquidus_bracket(
        {"MgO": 57.30, "SiO2": 42.70}, 1323.0, 2100.0, pack
    )
    assert anorthite.liquidus_K == pytest.approx(1697.9, abs=0.1)
    assert forsterite.liquidus_K == pytest.approx(1801.2, abs=0.1)
    # Current SI roots miss the measured melting points; they are not fitted here.
    assert anorthite.liquidus_K - 1826.0 == pytest.approx(-128.1, abs=0.1)
    assert forsterite.liquidus_K - 2163.0 == pytest.approx(-361.8, abs=1.0)


def test_cleaned_feed_liquidus_regression_and_dropped_parent_notices():
    from openimcc import CrystalStatus, liquidus_bracket

    feeds = {
        "lunar_low_ti": {
            "SiO2": 45.76, "TiO2": 1.54, "Al2O3": 13.88, "FeO": 16.97,
            "MgO": 9.26, "CaO": 11.31, "Na2O": 0.41, "K2O": 0.10,
            "Cr2O3": 0.36, "MnO": 0.21, "P2O5": 0.10,
        },
        "mars_basalt": {
            "SiO2": 47.16, "TiO2": 1.04, "Al2O3": 10.88, "FeO": 18.66,
            "MgO": 9.24, "CaO": 7.85, "Na2O": 3.77, "K2O": 0.52,
            "P2O5": 0.88,
        },
        "ci": {
            "SiO2": 41.70, "Al2O3": 2.90, "FeO": 20.65, "MgO": 28.32,
            "CaO": 3.33, "Na2O": 0.80, "NiO": 2.32,
        },
    }
    results = {name: liquidus_bracket(values, 1323.0, 2100.0)
               for name, values in feeds.items()}
    assert results["lunar_low_ti"].status is CrystalStatus.AVAILABLE
    assert results["lunar_low_ti"].liquidus_K == pytest.approx(1334.686172857904, abs=1e-6)
    assert results["lunar_low_ti"].controlling_phase == "anorthite"
    assert (results["mars_basalt"].status, results["mars_basalt"].liquidus_K,
            results["mars_basalt"].controlling_phase) == (
                CrystalStatus.NO_SUPPORTED_CROSSING, None, None)
    # Recomputed after the Fe/Ti nodes were rebuilt on the JANAF elemental
    # reference chain; the supported feed statuses remain unchanged.
    assert results["ci"].status is CrystalStatus.AVAILABLE
    assert results["ci"].liquidus_K == pytest.approx(1542.9466545748282, abs=1e-6)
    assert results["ci"].controlling_phase == "forsterite"
    dropped = results["lunar_low_ti"].notices[0]
    assert dropped.code == "crystal_si_dropped_parents"
    assert dropped.dropped_oxides_wt_percent == pytest.approx(
        {"Cr2O3": 0.36036036, "MnO": 0.21021021})
    assert results["mars_basalt"].notices == ()
    ci_dropped = results["ci"].notices[0]
    assert ci_dropped.dropped_oxides_wt_percent == pytest.approx({"NiO": 2.31953609})
    assert ci_dropped.pack_used.startswith("IMCC-SF04-D066@")


def test_fe_ti_fusion_nodes_use_the_janaf_elemental_reference_chain():
    from openimcc.diagnostics import load_crystal_dataset
    from openimcc.gas import load_gas_datapack, species_thermo

    dataset = load_crystal_dataset()
    gas_pack = load_gas_datapack()
    elements = dataset["element_reference_g_app_nodes_kJ_mol"]
    printed_crystal_nodes = {
        "fayalite": {"1300": -1056.8, "1400": -1025.2},
        "hercynite": {"1300": -1460.1, "1400": -1421.6},
        "ilmenite": {"1300": -908.5, "1400": -884.2,
                     "1500": -860.2, "1600": -836.3},
        "ulvospinel": {"1300": -1096.9, "1400": -1067.2,
                       "1500": -1037.9, "1600": -1009.0,
                       "1700": -980.6, "1800": -952.5},
        "perovskite": {"1500": -1239.4, "1600": -1211.8,
                       "1700": -1184.1, "1800": -1154.3},
    }
    element_stoichiometry = {
        "fayalite": {"Fe": 2, "Si": 1, "O2": 2},
        "hercynite": {"Fe": 1, "Al": 2, "O2": 2},
        "ilmenite": {"Fe": 1, "Ti": 1, "O2": 1.5},
        "ulvospinel": {"Fe": 2, "Ti": 1, "O2": 2},
        "perovskite": {"Ca": 1, "Ti": 1, "O2": 1.5},
    }
    phases = {row["id"]: row for row in dataset["phases"]}
    for phase_id, crystal_nodes in printed_crystal_nodes.items():
        phase = phases[phase_id]
        assert phase["source_crystal_gibbs_nodes_kJ_mol"] == crystal_nodes
        assert phase["solid_solution_bound"] is (phase_id != "perovskite")
        assert phase["source"]["table_page"]
        for t_text, crystal_g in crystal_nodes.items():
            temp = float(t_text)
            liquid_g_app = sum(
                amount * species_thermo(oxide, "l", temp, gas_pack).G_J_mol / 1000
                for oxide, amount in phase["oxide_stoichiometry"].items()
            )
            element_g_app = sum(
                amount * elements[element][t_text]
                for element, amount in element_stoichiometry[phase_id].items()
            )
            fusion_g = liquid_g_app - element_g_app - crystal_g
            assert phase["fusion_gibbs_nodes_kJ_mol"][t_text] == pytest.approx(
                fusion_g, abs=0.05
            )

    # Fayalite p. 332, 1400 K: pack parent G_app minus the printed JANAF
    # Fe, Si, and O2 reference functions gives formation Gibbs at T. Subtract
    # the printed R&H crystal formation Gibbs on that same reference.
    fayalite = phases["fayalite"]
    temp = 1400.0
    feo = species_thermo("FeO", "l", temp, gas_pack).G_J_mol / 1000
    silica = species_thermo("SiO2", "l", temp, gas_pack).G_J_mol / 1000
    hand = (2 * (feo - elements["Fe"]["1400"] - 0.5 * elements["O2"]["1400"])
            + silica - elements["Si"]["1400"] - elements["O2"]["1400"]
            - fayalite["source_crystal_gibbs_nodes_kJ_mol"]["1400"])
    assert hand == pytest.approx(34.463, abs=0.05)
    assert 0 < hand < 50  # 90 K below fayalite's incongruent melting point.

    # The same conversion on SiO2(l) agrees with the printed JANAF liquid
    # formation value at 1400 K (p. 1754) within the packaged fit residual.
    silica_formation_g = silica - elements["Si"]["1400"] - elements["O2"]["1400"]
    silica_janaf_printed_g = -659.670
    assert silica_formation_g == pytest.approx(-659.378, abs=0.001)
    assert silica_formation_g - silica_janaf_printed_g == pytest.approx(0.292, abs=0.001)

    # Just below congruent SiO2 melting (JANAF, 1996 K), the liquid-minus-
    # crystal Gibbs difference is small and positive on the packaged standard.
    near_melt = 1995.9
    silica_crystal = species_thermo("SiO2", "cr", near_melt, gas_pack).G_J_mol
    silica_liquid = species_thermo("SiO2", "l", near_melt, gas_pack).G_J_mol
    fusion_near_melt = silica_liquid - silica_crystal
    assert 0 < fusion_near_melt < 5000


def test_unsupported_only_composition_returns_typed_notice_instead_of_refusal():
    from openimcc import CrystalStatus, liquidus_bracket, saturation_indices

    result = liquidus_bracket({"NiO": 2.32}, 1323.0, 2100.0)
    assert result.status is CrystalStatus.NO_SUPPORTED_CROSSING
    assert result.notices[0].dropped_oxides_wt_percent == {"NiO": 100.0}
    rows = saturation_indices({"NiO": 2.32}, 1500.0)
    assert all(row.status is CrystalStatus.NO_SUPPORTED_COMPOSITION for row in rows)
    assert all(row.notices == result.notices for row in rows)


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


def test_mixing_enthalpy_reaction_enthalpy_basis_and_scaling():
    from openimcc.model import load_datapack

    pack = load_datapack()
    composition = {"CaO": 1.0, "MgO": 1.0, "SiO2": 2.0}
    row = _association_enthalpy(pack, composition, 1800.0)
    expected = -diagnostics.R * math.log(10.0) * pack.kernel_datapack.B / 1000.0
    assert [item["delta_rH_kJ_per_mol_complex"] for item in row["breakdown"]] == pytest.approx(expected)
    assert row["breakdown"][0]["extent_mol_per_mol_parent"] == pytest.approx(
        row["evaluation"].complex_x[0] / row["evaluation"].D
    )
    scaled = _association_enthalpy(pack, {key: 7.0 * value for key, value in composition.items()}, 1800.0)
    assert scaled["total_kJ"] == pytest.approx(7.0 * row["total_kJ"], rel=2e-10)
    assert scaled["per_parent_mol_kJ"] == pytest.approx(row["per_parent_mol_kJ"], rel=2e-10)


def test_mixing_enthalpy_pure_endpoint_and_additive_standard_cancellation():
    from openimcc.model import load_datapack

    pack = load_datapack()
    composition = {"CaO": 1.0, "MgO": 1.0, "SiO2": 2.0}
    endmembers = {"Di": composition}
    kwargs = {"endmembers": endmembers, "endmember_fractions": {"Di": 1.0}}
    pure = mixing_enthalpy(pack, composition, 1800.0, **kwargs)
    assert pure["excess_enthalpy_kJ"] == pytest.approx(0.0, abs=1e-12)
    mixed_composition = {"CaO": 1.0, "MgO": 0.6, "Al2O3": 0.4, "SiO2": 2.0}
    baseline = mixing_enthalpy(pack, mixed_composition, 1800.0, endmembers={
        "An": {"CaO": 1, "Al2O3": 1, "SiO2": 2},
        "Di": {"CaO": 1, "MgO": 1, "SiO2": 2},
    }, endmember_fractions={"An": 0.4, "Di": 0.6})
    shifted = mixing_enthalpy(pack, mixed_composition, 1800.0, endmembers={
        "An": {"CaO": 1, "Al2O3": 1, "SiO2": 2},
        "Di": {"CaO": 1, "MgO": 1, "SiO2": 2},
    }, endmember_fractions={"An": 0.4, "Di": 0.6},
        oxide_standard_enthalpies_kJ_mol={"CaO": 12, "MgO": -3, "Al2O3": 8, "SiO2": 1.5})
    assert shifted["excess_enthalpy_kJ"] == pytest.approx(baseline["excess_enthalpy_kJ"], abs=1e-12)
    scaled = mixing_enthalpy(pack, {key: 5.0 * value for key, value in mixed_composition.items()},
                             1800.0, endmembers={
                                 "An": {"CaO": 1, "Al2O3": 1, "SiO2": 2},
                                 "Di": {"CaO": 1, "MgO": 1, "SiO2": 2},
                             }, endmember_fractions={"An": 0.4, "Di": 0.6})
    assert scaled["excess_enthalpy_kJ"] == pytest.approx(5.0 * baseline["excess_enthalpy_kJ"], rel=2e-10)


def test_mixing_enthalpy_dissolution_conversion_and_pinned_residuals():
    dataset = load_mixing_enthalpy_dataset()["datasets"][1]
    assert _fusion_enthalpy(dataset, "CaO", 1723.15) == pytest.approx(77.754)
    pack, rows = run_mixing_enthalpy()
    cmas = next(row for row in rows if row["observation_id"] == "CaO")
    assert cmas["measured_mixing_partial_kJ_mol"] == pytest.approx(-4.3 - 77.754, abs=1e-9)
    assert cmas["residual_kJ_mol"] == pytest.approx(11.374626, abs=0.002)
    expected = {
        "An43.5Di49Fo7.5": -1.027633,
        "An42Di58": 0.870437,
        "Di58Ak42-Ak133.1": -2.664967,
        "Di58Ak42-Ak139.8": -0.164967,
        "An46Ak54-Ak133.1": -1.069885,
        "An46Ak54-Ak139.8": 2.530115,
        "Di84Qt16": -2.795509,
        "CaO": 11.374626,
        "SiO2": -3.168378,
        "Al2O3": -40.138551,
        "MgO": -59.411074,
    }
    actual = {row["observation_id"]: row["residual_kJ_mol"] for row in rows}
    assert actual == pytest.approx(expected, abs=0.002)
    assert pack.model_id == "IMCC-SF04-D066"


def test_mixing_enthalpy_cli_receipt_and_csv(tmp_path):
    assert main(["diagnose", "mixing-enthalpy", "--out", str(tmp_path)]) == 0
    with (tmp_path / "mixing-enthalpy.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 11
    assert all(row["standardized_residual"] for row in rows)
    receipt = (tmp_path / "mixing-enthalpy.md").read_text(encoding="utf-8")
    assert "IMCC-SF04-D066" in receipt
    assert "Binding digest:" in receipt
