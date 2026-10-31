from __future__ import annotations

import json

import numpy as np
import pytest

import openimcc
import openimcc.gas as gas_module
import openimcc.model as model_module
import openimcc.redox as redox_module
from openimcc import evaluate, evaluate_redox
from openimcc.gas import (
    evaluate_gas,
    evaluate_gas_oxygen_balance,
    evaluate_gas_saturation,
    gas_boiling_temperature,
)


_BASALT = {
    "SiO2": 51.85068,
    "MgO": 4.78527,
    "FeO": 13.77307,
    "CaO": 9.02862,
    "Al2O3": 14.80572,
    "TiO2": 1.73824,
    "Na2O": 3.23108,
    "K2O": 0.78732,
}


def _assert_json_receipt(result):
    receipt = result.receipt
    assert receipt is not None
    payload = receipt.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert set(payload) == {
        "request", "data_digests", "code", "dependencies", "solver"
    }
    return receipt


def test_receipts_cover_each_result_entrypoint_and_existing_digests():
    melt = evaluate(_BASALT, 1800.0, basis_type="wt")
    melt_receipt = _assert_json_receipt(melt)
    assert melt_receipt.data_digests == {
        "melt": melt.labels.identity["binding_digest"]
    }

    redox = evaluate_redox({"Fe": 1.0, "O": 1.25}, 2200.0, "closed")
    redox_receipt = _assert_json_receipt(redox)
    assert redox_receipt.data_digests == {
        "melt": redox.melt_pack_binding_digest,
        "redox": redox.redox_pack_digest,
    }

    profile = openimcc.load_profile("openimcc")
    activities = {name: melt.activity(name) for name in melt.parent_oxides}
    gas = evaluate_gas(activities, 1800.0, 1e-10, profile="openimcc")
    gas_receipt = _assert_json_receipt(gas)
    assert gas_receipt.data_digests == {
        "gas": gas.gas_table_digest,
        "condensate": gas.condensate_table_digest,
    }
    assert gas_receipt.request["fO2"] == 1e-10
    assert gas_receipt.request["profile"] == "openimcc"

    _, balanced, _ = evaluate_gas_oxygen_balance(
        activities, 1800.0, profile="openimcc"
    )
    balance_receipt = _assert_json_receipt(balanced)
    assert balance_receipt.data_digests == {
        "gas": balanced.gas_table_digest,
        "condensate": balanced.condensate_table_digest,
    }

    saturation = evaluate_gas_saturation(
        activities, 1800.0, parent_oxides=tuple(activities),
        allow_extrapolation=True, profile="openimcc",
    )
    saturation_receipt = _assert_json_receipt(saturation)
    assert saturation_receipt.data_digests == {
        "gas": saturation.gas_table_digest,
        "condensate": saturation.condensate_table_digest,
    }

    lower = evaluate_gas_saturation(
        activities, 1800.0, parent_oxides=tuple(activities),
        allow_extrapolation=True, profile="openimcc",
    )
    upper = evaluate_gas_saturation(
        activities, 2000.0, parent_oxides=tuple(activities),
        allow_extrapolation=True, profile="openimcc",
    )
    _, boiling = gas_boiling_temperature(
        activities,
        ambient_pressure_bar=(lower.P_sat_bar + upper.P_sat_bar) / 2.0,
        T_bracket=(1800.0, 2000.0),
        parent_oxides=tuple(activities),
        allow_extrapolation=True,
        profile="openimcc",
    )
    boiling_receipt = _assert_json_receipt(boiling)
    assert boiling_receipt.data_digests == {
        "gas": boiling.gas_table_digest,
        "condensate": boiling.condensate_table_digest,
    }


def test_receipt_echoes_melt_request_and_records_solver_settings():
    result = evaluate(_BASALT, 1800.0, basis_type="wt")

    assert result.receipt.request == {
        "composition": _BASALT,
        "T_K": 1800.0,
        "basis_type": "wt",
        "profile": None,
        "options": {
            "basis": None,
            "extra_mol": None,
            "enable_sp_extension": False,
            "allow_extrapolation": False,
            "allow_out_of_envelope": False,
            "tol": 1e-12,
            "max_iter": 100,
            "y_init": None,
        },
    }
    assert result.receipt.solver == {"tol": 1e-12, "max_iter": 100}


def test_version_is_unknown_when_distribution_does_not_own_imported_package(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    class Distribution:
        version = "99.0"

        def locate_file(self, _path):
            return tmp_path / "openimcc" / "__init__.py"

    monkeypatch.setattr(openimcc.metadata, "distribution", lambda _name: Distribution())

    assert openimcc._resolve_version() == "0+unknown"


def test_default_profile_numbers_match_without_receipt_builder(
    monkeypatch: pytest.MonkeyPatch,
):
    with_profile = openimcc.load_profile("openimcc")
    melt_with = evaluate(_BASALT, 1800.0, basis_type="wt")
    redox_with = evaluate_redox({"Fe": 1.0, "O": 1.25}, 2200.0, "closed")
    activities = {name: melt_with.activity(name) for name in melt_with.parent_oxides}
    gas_with = evaluate_gas(activities, 1800.0, 1e-10, profile=with_profile)

    with monkeypatch.context() as patch:
        patch.setattr(model_module, "build_calculation_receipt", lambda **_kwargs: None)
        melt_without = evaluate(_BASALT, 1800.0, basis_type="wt")
    assert np.array_equal(melt_with.parent_activity, melt_without.parent_activity)
    assert melt_with.D.hex() == melt_without.D.hex()

    with monkeypatch.context() as patch:
        patch.setattr(redox_module, "build_calculation_receipt", lambda **_kwargs: None)
        redox_without = evaluate_redox({"Fe": 1.0, "O": 1.25}, 2200.0, "closed")
    assert redox_with.parent_moles == redox_without.parent_moles
    assert redox_with.oxygen_mol.hex() == redox_without.oxygen_mol.hex()

    with monkeypatch.context() as patch:
        patch.setattr(gas_module, "build_calculation_receipt", lambda **_kwargs: None)
        gas_without = evaluate_gas(activities, 1800.0, 1e-10, profile=with_profile)
    assert dict(gas_with) == dict(gas_without)
