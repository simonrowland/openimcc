from __future__ import annotations

import csv
import json
import shutil
from importlib import resources
from pathlib import Path

import pytest

from openimcc import diagnostics
from openimcc.gas import ImccGasSourceOverrideWarning
from openimcc.standard_checks import (
    JANAF_PAIR_TOLERANCE_KJ_MOL,
    PACKAGED_SIO2_ROUNDING_TOLERANCE_KJ_MOL,
    SWITCH_BRACKET_TOLERANCE_K,
    StandardCheckRefusal,
    _OXIDES,
    _axis_label,
    _janaf_liquid,
    _janaf_crystal,
    pure_oxide_standard_check,
    pure_oxide_standard_checks,
)


REPO = Path(__file__).resolve().parents[1]
LEGACY_CONDENSATE = REPO / "src/openimcc/data/gas/condensate-feo-janaf-legacy.csv"
# The ±5 K constant-ΔCp relation is only a local sign/scale check; this band
# allows source Cp curvature without tuning the computed JANAF Gibbs function.
LOCAL_FUSION_CHECK_ABS_TOLERANCE_KJ_MOL = 0.03
LOCAL_FUSION_CHECK_REL_TOLERANCE = 0.2
LOCAL_FUSION_CHECK_STEP_K = 5.0


def _enthalpy_at(table_id: str, temperature_K: float) -> float:
    data = json.loads(
        resources.files("openimcc").joinpath(
            "data/diagnostics/janaf-pure-oxide-liquid-pairs.json"
        ).read_text(encoding="utf-8")
    )
    row = next((item for item in data["rows"] if item["table_id"] == table_id), None)
    if row is None:
        crystal_data = json.loads(
            resources.files("openimcc").joinpath(
                "data/diagnostics/janaf-pure-oxide-crystals.json"
            ).read_text(encoding="utf-8")
        )
        row = next(item for item in crystal_data["phases"] if item["crystal_table_id"] == table_id)
    formation_enthalpy_kJ = row["formation_enthalpy_298_kJ_mol"]
    by_temperature = {
        item["T_K"]: item["enthalpy_increment_kJ_mol"]
        for item in row["nodes"] + row.get("transition_rows", [])
    }
    points = sorted(by_temperature.items())
    for left, right in zip(points, points[1:]):
        if left[0] <= temperature_K <= right[0]:
            fraction = (temperature_K - left[0]) / (right[0] - left[0])
            increment = left[1] + fraction * (right[1] - left[1])
            return formation_enthalpy_kJ + increment
    if temperature_K == points[-1][0]:
        return formation_enthalpy_kJ + points[-1][1]
    raise AssertionError(f"no JANAF enthalpy at {temperature_K} K for {table_id}")


def test_janaf_pair_closure_sign_and_local_constant_delta_cp_check():
    checks = {row.oxide: row for row in pure_oxide_standard_checks()}
    paired = [definition for definition in _OXIDES if definition.liquid_table]
    for definition in paired:
        result = checks[definition.oxide]
        assert result.janaf_pair_status == "available"
        assert abs(result.delta_janaf_kJ_mol) <= JANAF_PAIR_TOLERANCE_KJ_MOL

        transition = definition.transition_K
        h_fusion_J_mol = (
            _enthalpy_at(definition.liquid_table, transition)
            - _enthalpy_at(definition.crystal_table, transition)
        ) * 1000.0
        assert h_fusion_J_mol > 0.0
        temperature = transition - LOCAL_FUSION_CHECK_STEP_K
        delta = (
            _janaf_liquid(definition.liquid_table, temperature).G_J_mol
            - _janaf_crystal(definition.oxide, temperature).G_J_mol
        ) / 1000.0
        estimate = h_fusion_J_mol * (1.0 - temperature / transition) / 1000.0
        assert delta > 0.0
        # The shared crystal evaluator stops at the phase transition. The
        # test-side local constant-ΔCp relation checks the expected sign on
        # both sides without evaluating a crystal beyond its allowed domain.
        above_estimate = h_fusion_J_mol * (
            1.0 - (transition + LOCAL_FUSION_CHECK_STEP_K) / transition
        ) / 1000.0
        assert above_estimate < 0.0
        assert abs(delta - estimate) <= max(
            LOCAL_FUSION_CHECK_ABS_TOLERANCE_KJ_MOL,
            LOCAL_FUSION_CHECK_REL_TOLERANCE * abs(estimate),
        )


def test_packaged_silica_residuals_keep_each_crystal_reference_and_pack_identity():
    default = pure_oxide_standard_check("SiO2", 1995.9)
    assert default.pack_id == "condensate-default-v1.csv"
    assert default.pack_digest
    assert default.source_crystal_id == "packaged:O-035"
    assert default.packaged_closure.status == "available"
    assert default.packaged_closure.delta_kJ_mol == pytest.approx(
        0.00048, abs=PACKAGED_SIO2_ROUNDING_TOLERANCE_KJ_MOL
    )

    legacy = pure_oxide_standard_check(
        "SiO2", 1995.9, condensate_path=LEGACY_CONDENSATE
    )
    assert legacy.pack_id == LEGACY_CONDENSATE.name
    assert legacy.pack_digest
    assert legacy.source_crystal_id == "packaged:LAM1987"
    assert legacy.packaged_closure.delta_kJ_mol == pytest.approx(
        2.902, abs=PACKAGED_SIO2_ROUNDING_TOLERANCE_KJ_MOL
    )
    assert legacy.switch_prediction.status == "no_switch_in_range"
    assert legacy.switch_prediction.search_range_K == pytest.approx((1200.0, 1996.0))


def test_pure_oxide_standard_check_warns_for_default_vaporock_tables(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    packaged_gas_pack,
) -> None:
    root = tmp_path / "vaporock"
    gas_target = root / "src" / "vaporock" / "data"
    oxide_target = root / "data"
    gas_target.mkdir(parents=True)
    oxide_target.mkdir(parents=True)
    shutil.copy2(
        packaged_gas_pack.gas_path,
        gas_target / "JANAF-vapor-data-full.csv",
    )
    shutil.copy2(
        packaged_gas_pack.oxide_path,
        oxide_target / "condensate-thermo-data.csv",
    )
    monkeypatch.setenv("OPENIMCC_VAPOROCK_ROOT", str(root))

    with pytest.warns(
        ImccGasSourceOverrideWarning, match="OPENIMCC_VAPOROCK_ROOT"
    ):
        result = pure_oxide_standard_check("SiO2")

    assert result.pack_id == "condensate-thermo-data.csv"


def test_crystal_diagnostics_warn_for_default_vaporock_tables(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    packaged_gas_pack,
) -> None:
    root = tmp_path / "vaporock"
    gas_target = root / "src" / "vaporock" / "data"
    oxide_target = root / "data"
    gas_target.mkdir(parents=True)
    oxide_target.mkdir(parents=True)
    shutil.copy2(
        packaged_gas_pack.gas_path,
        gas_target / "JANAF-vapor-data-full.csv",
    )
    shutil.copy2(
        packaged_gas_pack.oxide_path,
        oxide_target / "condensate-thermo-data.csv",
    )
    monkeypatch.setenv("OPENIMCC_VAPOROCK_ROOT", str(root))

    with pytest.warns(
        ImccGasSourceOverrideWarning, match="OPENIMCC_VAPOROCK_ROOT"
    ):
        gas_pack, reference_pack = diagnostics._crystal_thermo_context(None)

    assert gas_pack is reference_pack
    assert gas_pack.gas_path == gas_target / "JANAF-vapor-data-full.csv"
    assert gas_pack.oxide_path == oxide_target / "condensate-thermo-data.csv"


def test_default_silica_switch_refuses_an_unresolved_endpoint_sign():
    result = pure_oxide_standard_check("SiO2")
    switch = result.switch_prediction
    # The high endpoint residual is smaller than the subtraction's own
    # floating-point resolution; a sign cannot be certified there.
    assert switch.status == "resolution_indeterminate", switch
    assert switch.temperature_K is None
    assert switch.bracket_K is not None
    assert switch.bracket_K[1] - switch.bracket_K[0] <= SWITCH_BRACKET_TOLERANCE_K
    assert switch.bracket_residuals_J_mol is not None
    assert switch.bracket_resolution_bounds_J_mol is not None
    assert switch.bracket_K == pytest.approx((1995.9375, 1996.0))
    assert switch.bracket_residuals_J_mol[1] == pytest.approx(-2.3e-10, abs=2e-10)
    assert abs(switch.bracket_residuals_J_mol[1]) <= switch.bracket_resolution_bounds_J_mol[1]
    assert switch.bracket_resolution_bounds_J_mol[1] == pytest.approx(4.0e-9, rel=0.02)
    assert result.pack_id == "condensate-default-v1.csv"
    assert result.pack_digest


def test_resolved_switch_root_is_bracketed_by_its_pack_data():
    result = pure_oxide_standard_check("Al2O3")
    switch = result.switch_prediction
    assert switch.status == "root"
    assert switch.bracket_K is not None
    assert switch.bracket_K[0] <= switch.temperature_K <= switch.bracket_K[1]
    assert switch.bracket_residuals_J_mol is not None
    assert switch.bracket_resolution_bounds_J_mol is not None
    assert switch.bracket_residuals_J_mol[0] > switch.bracket_resolution_bounds_J_mol[0]
    assert switch.bracket_residuals_J_mol[1] < -switch.bracket_resolution_bounds_J_mol[1]
    assert switch.temperature_K == pytest.approx(2326.97, abs=0.1)
    assert result.pack_id == "condensate-default-v1.csv"
    assert result.pack_digest


def test_requested_temperature_above_transition_has_separate_axis_label():
    result = pure_oxide_standard_check("Al2O3", 2400.0)

    assert result.temperature_K == 2400.0
    assert result.evaluation_temperature_axis_label.startswith("temperature_K;")
    assert "metastable/superheated crystal leg above JANAF transition" in (
        result.evaluation_temperature_axis_label
    )
    assert "metastable/superheated" not in result.switch_prediction.temperature_axis_label


@pytest.mark.parametrize(
    ("oxide", "condensate_path", "expected_root_K"),
    [
        ("TiO2", None, 2130.064914),
        ("TiO2", LEGACY_CONDENSATE, 2130.064914),
        ("FeO", LEGACY_CONDENSATE, 1650.244128),
        ("Na2O", LEGACY_CONDENSATE, 1405.209948),
    ],
)
def test_switch_roots_above_janaf_transition_use_metastable_crystal_leg(
    oxide, condensate_path, expected_root_K
):
    result = pure_oxide_standard_check(oxide, condensate_path=condensate_path)
    switch = result.switch_prediction

    assert switch.status == "root", switch
    assert switch.temperature_K == pytest.approx(expected_root_K, abs=0.1)
    assert "metastable/superheated crystal leg above JANAF transition" in (
        switch.temperature_axis_label
    )


def test_default_sodium_switch_root_is_preserved():
    result = pure_oxide_standard_check("Na2O")

    assert result.switch_prediction.status == "root"
    assert result.switch_prediction.temperature_K == pytest.approx(1405.17, abs=0.1)


def test_k2o_is_a_flagged_construction_and_below_1700_root_is_labelled():
    result = pure_oxide_standard_check("K2O")
    assert result.janaf_pair_status == "construction_case"
    assert result.delta_janaf_kJ_mol is None
    assert result.construction_flag is not None
    assert result.switch_prediction.status == "root"
    assert result.switch_prediction.temperature_K < 1700.0
    assert result.switch_prediction.bracket_K is not None
    assert result.switch_prediction.bracket_residuals_J_mol is not None
    assert (
        result.switch_prediction.bracket_residuals_J_mol[0]
        * result.switch_prediction.bracket_residuals_J_mol[1]
        < 0.0
    )
    assert "extrapolated below 1700 K" in result.switch_prediction.temperature_axis_label


def test_k2o_authority_label_reports_its_composition_scope():
    result = pure_oxide_standard_check("K2O", 1500.0)

    assert "K2O_wt_pct 25–40 wt%" in result.least_authoritative_input
    assert "pure K2O is outside this scope" in result.least_authoritative_input


def test_out_of_range_janaf_temperature_is_a_typed_refusal():
    with pytest.raises(StandardCheckRefusal) as caught:
        pure_oxide_standard_check("CaO", 3601.0)
    assert caught.value.phase == "Ca-027"
    assert caught.value.axis == "temperature_K"
    assert caught.value.valid_range_K == pytest.approx((298.15, 3600.0))

    with pytest.raises(StandardCheckRefusal) as construction_caught:
        pure_oxide_standard_check("K2O", 2001.0)
    assert construction_caught.value.phase == "K-012"
    assert construction_caught.value.axis == "temperature_K"
    assert construction_caught.value.valid_range_K == pytest.approx((298.15, 2000.0))


def test_existing_crystal_closure_fixed_input_is_bit_identical():
    from openimcc.diagnostics import crystal_closure
    from openimcc.model import load_datapack

    pack = load_datapack()
    rows, details = crystal_closure(pack)
    baseline = json.dumps(
        {"rows": rows, "details": details}, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    pure_oxide_standard_check("SiO2", 1995.9)
    after_rows, after_details = crystal_closure(pack)
    after = json.dumps(
        {"rows": after_rows, "details": after_details},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    assert after == baseline


def test_axis_label_marks_temperature_above_liquid_model_domain():
    assert "above 3000 K" in _axis_label(3105.0)
    assert "OUT_OF_DOMAIN" in _axis_label(3105.0)


def test_diagnose_cli_writes_selected_pack_provenance(tmp_path):
    from openimcc.cli import main

    assert main([
        "diagnose", "standard-checks", "--condensate", str(LEGACY_CONDENSATE),
        "--out", str(tmp_path),
    ]) == 0
    with (tmp_path / "pure-oxide-standard-checks.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 8
    silica = next(row for row in rows if row["oxide"] == "SiO2")
    assert silica["pack_id"] == LEGACY_CONDENSATE.name
    assert silica["source_crystal_id"] == "packaged:LAM1987"
    assert silica["switch_status"] == "no_switch_in_range"
    assert "evaluation_temperature_axis_label" in silica
