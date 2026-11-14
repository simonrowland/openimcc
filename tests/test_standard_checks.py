from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from openimcc.standard_checks import (
    JANAF_PAIR_TOLERANCE_KJ_MOL,
    PACKAGED_SIO2_ROUNDING_TOLERANCE_KJ_MOL,
    SWITCH_BRACKET_TOLERANCE_K,
    StandardCheckRefusal,
    _OXIDES,
    _janaf_apparent_gibbs,
    _janaf_table,
    pure_oxide_standard_check,
    pure_oxide_standard_checks,
)


REPO = Path(__file__).resolve().parents[1]
LEGACY_CONDENSATE = REPO / "src/openimcc/data/gas/condensate-feo-janaf-legacy.csv"
# The ±5 K constant-ΔCp relation is only a local sign/scale check; this band
# allows source Cp curvature without tuning the computed JANAF Gibbs function.
LOCAL_FUSION_CHECK_ABS_TOLERANCE_KJ_MOL = 0.03
LOCAL_FUSION_CHECK_REL_TOLERANCE = 0.2


def _enthalpy_at(table_id: str, temperature_K: float) -> float:
    formation_enthalpy_kJ, points = _janaf_table(table_id)
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
        for temperature in (transition - 5.0, transition + 5.0):
            delta = (
                _janaf_apparent_gibbs(definition.liquid_table, temperature)
                - _janaf_apparent_gibbs(definition.crystal_table, temperature)
            ) / 1000.0
            estimate = h_fusion_J_mol * (1.0 - temperature / transition) / 1000.0
            assert delta > 0.0 if temperature < transition else delta < 0.0
            # Local constant-ΔCp relation is a test-side sign and scale check;
            # it is not used to construct either JANAF Gibbs function.
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


def test_default_silica_switch_comes_from_its_sign_changing_bracket():
    result = pure_oxide_standard_check("SiO2")
    switch = result.switch_prediction
    assert switch.status == "root", switch
    assert switch.temperature_K == pytest.approx(1996.0, abs=0.1)
    assert switch.bracket_K is not None
    assert switch.bracket_K[0] <= switch.temperature_K <= switch.bracket_K[1]
    assert switch.bracket_K[1] - switch.bracket_K[0] <= SWITCH_BRACKET_TOLERANCE_K
    assert switch.bracket_residuals_J_mol is not None
    assert switch.bracket_residuals_J_mol[0] * switch.bracket_residuals_J_mol[1] < 0.0
    assert switch.residual_J_mol is not None
    assert switch.uncertainty_K <= SWITCH_BRACKET_TOLERANCE_K
    assert result.pack_id == "condensate-default-v1.csv"
    assert result.pack_digest


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


def test_out_of_range_janaf_temperature_is_a_typed_refusal():
    with pytest.raises(StandardCheckRefusal) as caught:
        pure_oxide_standard_check("CaO", 3601.0)
    assert caught.value.phase == "Ca-027"
    assert caught.value.axis == "temperature_K"
    assert caught.value.valid_range_K == pytest.approx((0.0, 3600.0))

    with pytest.raises(StandardCheckRefusal) as construction_caught:
        pure_oxide_standard_check("K2O", 2001.0)
    assert construction_caught.value.phase == "K-012"
    assert construction_caught.value.axis == "temperature_K"
    assert construction_caught.value.valid_range_K == pytest.approx((298.15, 2000.0))


def test_existing_crystal_closure_fixed_input_is_bit_identical():
    from openimcc.diagnostics import crystal_closure
    from openimcc.model import load_datapack

    rows, details = crystal_closure(load_datapack())
    encoded = json.dumps(
        {"rows": rows, "details": details}, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    assert digest == "fb40d0394b0a8d13cdbe76941c22882ef5b4e2362112b49d71acac4d485f4bf8", (
        f"crystal_closure output changed: {digest}"
    )


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
