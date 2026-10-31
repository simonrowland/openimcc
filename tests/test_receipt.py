from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

import openimcc
import openimcc.receipt as receipt_module
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


def _assert_parent_numbers(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)


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
    # Numerical expectations were generated from parent commit 47f6f01.
    melt = evaluate(_BASALT, 1800.0, basis_type="wt")
    _assert_parent_numbers(
        [melt.D, melt.activity("SiO2")],
        [2.102073306152123, 0.32943033154573187],
    )
    melt_receipt = _assert_json_receipt(melt)
    assert melt_receipt.data_digests == {
        "melt": melt.labels.identity["binding_digest"]
    }

    redox = evaluate_redox({"Fe": 1.0, "O": 1.25}, 2200.0, "closed")
    _assert_parent_numbers(
        [redox.lambda_ln_f_o2, redox.oxygen_mol],
        [-1.5205238540751636, 1.250000000000054],
    )
    redox_receipt = _assert_json_receipt(redox)
    assert redox_receipt.data_digests == {
        "melt": redox.melt_pack_binding_digest,
        "redox": redox.redox_pack_digest,
    }

    profile = openimcc.load_profile("openimcc")
    activities = {name: melt.activity(name) for name in melt.parent_oxides}
    gas = evaluate_gas(activities, 1800.0, 1e-10, profile="openimcc")
    _assert_parent_numbers(
        [gas["O2"], gas["Na"]],
        [1e-10, 0.00019759260187725066],
    )
    gas_receipt = _assert_json_receipt(gas)
    assert gas_receipt.data_digests == {
        "gas": gas.gas_table_digest,
        "condensate": gas.condensate_table_digest,
    }
    assert gas_receipt.request["fO2"] == 1e-10
    assert gas_receipt.request["profile"] == "openimcc"

    balance_pO2, balanced, _ = evaluate_gas_oxygen_balance(
        activities, 1800.0, profile="openimcc"
    )
    _assert_parent_numbers(
        [balance_pO2, balanced["Na"]],
        [4.044113669939114e-06, 1.3933648210712147e-05],
    )
    balance_receipt = _assert_json_receipt(balanced)
    assert balance_receipt.data_digests == {
        "gas": balanced.gas_table_digest,
        "condensate": balanced.condensate_table_digest,
    }
    assert "root_scan_points" not in balance_receipt.solver

    saturation = evaluate_gas_saturation(
        activities, 1800.0, parent_oxides=tuple(activities),
        allow_extrapolation=True, profile="openimcc",
    )
    _assert_parent_numbers(
        [saturation.P_sat_bar, saturation.pO2_bar],
        [1.865008271569144e-05, 3.5987590837640433e-06],
    )
    saturation_receipt = _assert_json_receipt(saturation)
    assert saturation_receipt.data_digests == {
        "gas": saturation.gas_table_digest,
        "condensate": saturation.condensate_table_digest,
    }

    boiling_temperature, boiling = gas_boiling_temperature(
        activities,
        ambient_pressure_bar=4.865657946350167e-05,
        T_bracket=(1800.0, 2000.0),
        parent_oxides=tuple(activities),
        allow_extrapolation=True,
        profile="openimcc",
    )
    _assert_parent_numbers(
        [boiling_temperature, boiling.P_sat_bar],
        [1930.1681888788141, 4.865657946350161e-05],
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

        def read_text(self, _path):
            return None

    monkeypatch.setattr(openimcc.metadata, "distribution", lambda _name: Distribution())

    assert openimcc._resolve_version() == "0+unknown"


def test_version_accepts_wheel_and_editable_ownership(monkeypatch, tmp_path):
    imported_package = Path(openimcc.__file__).resolve()

    class Distribution:
        version = "1.2.3"

        def __init__(self, installed_path, direct_url=None):
            self.installed_path = installed_path
            self.direct_url = direct_url

        def locate_file(self, _path):
            return self.installed_path

        def read_text(self, _path):
            return self.direct_url

    monkeypatch.setattr(
        openimcc.metadata,
        "distribution",
        lambda _name: Distribution(imported_package),
    )
    assert openimcc._resolve_version() == "1.2.3"

    project_dir = imported_package.parents[2]
    editable_url = json.dumps({
        "url": project_dir.as_uri(),
        "dir_info": {"editable": True},
    })
    monkeypatch.setattr(
        openimcc.metadata,
        "distribution",
        lambda _name: Distribution(
            tmp_path / "site-packages/openimcc/__init__.py", editable_url
        ),
    )
    assert openimcc._resolve_version() == "1.2.3"


def test_version_does_not_claim_a_nested_editable_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    project = tmp_path / "project"
    owned_package = project / "src/openimcc"
    nested_package = project / "worktrees/other/src/openimcc"
    owned_package.mkdir(parents=True)
    (owned_package / "__init__.py").write_text("# outer package\n")
    nested_package.mkdir(parents=True)
    imported_package = nested_package / "__init__.py"
    imported_package.write_text("# nested checkout\n")

    class Distribution:
        version = "99.0"

        def locate_file(self, _path):
            return tmp_path / "site-packages/openimcc/__init__.py"

        def read_text(self, _path):
            return json.dumps({
                "url": project.as_uri(),
                "dir_info": {"editable": True},
            })

    monkeypatch.setattr(openimcc.metadata, "distribution", lambda _name: Distribution())
    with monkeypatch.context() as patch:
        patch.setattr(openimcc, "__file__", str(imported_package))
        assert openimcc._resolve_version() == "0+unknown"


def test_version_legacy_develop_install_without_direct_url_is_unknown(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    project = tmp_path / "project"
    imported_package = project / "src/openimcc/__init__.py"
    imported_package.parent.mkdir(parents=True)
    imported_package.write_text("# legacy develop install\n")

    class Distribution:
        version = "99.0"

        def locate_file(self, _path):
            return project / "lib/python/site-packages/openimcc/__init__.py"

        def read_text(self, _path):
            return None

    monkeypatch.setattr(openimcc.metadata, "distribution", lambda _name: Distribution())
    with monkeypatch.context() as patch:
        patch.setattr(openimcc, "__file__", str(imported_package))
        assert openimcc._resolve_version() == "0+unknown"


@pytest.mark.parametrize(
    "direct_url",
    [
        "{malformed json",
        json.dumps({
            "url": "https://example.invalid/project",
            "dir_info": {"editable": True},
        }),
    ],
)
def test_version_ignores_malformed_or_non_file_direct_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path, direct_url
):
    class Distribution:
        version = "99.0"

        def locate_file(self, _path):
            return tmp_path / "site-packages/openimcc/__init__.py"

        def read_text(self, _path):
            return direct_url

    monkeypatch.setattr(openimcc.metadata, "distribution", lambda _name: Distribution())

    assert openimcc._resolve_version() == "0+unknown"


def test_source_identity_ignores_an_enclosing_unrelated_checkout(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    checkout = tmp_path / "project"
    package = checkout / ".venv/lib/python/site-packages/openimcc"
    package.mkdir(parents=True)
    (checkout / ".git").mkdir()
    source_package = checkout / "src/openimcc"
    source_package.mkdir(parents=True)
    (source_package / "__init__.py").write_text("# another package\n")
    imported_init = package / "__init__.py"
    imported_init.write_text("# installed package\n")
    (package / "receipt.py").write_text("# installed receipt\n")

    with monkeypatch.context() as patch:
        patch.setattr(receipt_module, "__file__", str(package / "receipt.py"))
        receipt_module._source_identity.cache_clear()
        assert receipt_module._source_identity() == (None, None)
    receipt_module._source_identity.cache_clear()


def test_result_equality_ignores_the_receipt():
    melt = evaluate(_BASALT, 1800.0, basis_type="wt")
    activities = {name: melt.activity(name) for name in melt.parent_oxides}
    redox = evaluate_redox({"Fe": 1.0, "O": 1.25}, 2200.0, "closed")
    gas = evaluate_gas(activities, 1800.0, 1e-10, profile="openimcc")
    saturation = evaluate_gas_saturation(
        activities, 1800.0, parent_oxides=tuple(activities),
        allow_extrapolation=True, profile="openimcc",
    )

    for result in (melt, redox, gas, saturation):
        receipt_field = next(field for field in fields(result) if field.name == "receipt")
        assert receipt_field.compare is False
        assert receipt_field.hash is False

    for result in (redox, gas, saturation):
        assert result.receipt is not None
        assert result == replace(result, receipt=None)
