"""Behavior-focused coverage gates for the IMCC-SF04 model adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

import openimcc.model as model
from openimcc import (
    ImccComponentOutsideDomainError,
    ImccCompositionIncompleteError,
    ImccDatapack,
    ImccLoadedDatapack,
    ImccMalformedDatapackError,
    ImccSPComponentRequiresExtensionError,
    ImccUnprovenDatapackError,
    evaluate,
    load_datapack,
)


BASE_DATAPACK = Path("src/openimcc/data/packs/imcc-sf04-v1.0.2.json")


def _base_data() -> dict[str, Any]:
    return json.loads(BASE_DATAPACK.read_text())


def _write_core_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[dict[str, Any]], None],
) -> Path:
    data = _base_data()
    mutate(data)
    version = data.get("imcc_sf04_datapack_version")
    model_id = data.get("model_id", "IMCC-SF04")
    expected_hash = model._published_datapack_manifest_hash(
        model._published_core_manifest_payload(
            data,
            model_id=model_id,
            version=version,
        )
    )
    monkeypatch.setattr(model, "_PUBLISHED_DATAPACK_SHA256", expected_hash)
    path = tmp_path / "variant.json"
    path.write_text(json.dumps(data, allow_nan=True))
    return path


def _extension_data() -> dict[str, Any]:
    data = _base_data()
    data.update(
        {
            "model_id": "IMCC-SF04-EXT",
            "imcc_sf04_datapack_version": "1.0.2-ext-sp-coverage",
            "sp_extension": {
                "enable_flag": "enable_sp_extension",
                "parents": ["S", "P2O5"],
                "tier": "EXT-SP",
                "certification": "denied",
                "authority": "screening-only",
                "rows": [
                    {
                        "complex": "FeS",
                        "nu": {"FeO": 1, "S": 1},
                        "A": -1.0,
                        "B": 1000.0,
                        "T_domain_K": [400.0, 3000.0],
                        "T_domain_basis": "coverage extension",
                        "provenance_class": "extension-compound-thermo",
                        "tier": "EXT-SP",
                        "certification": "denied",
                        "provenance": {
                            "source": "coverage",
                            "table_id": "Fe-coverage",
                        },
                    }
                ],
            },
        }
    )
    return data


def _write_extension_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Callable[[dict[str, Any]], None],
) -> Path:
    data = _extension_data()
    mutate(data)
    expected_hash = model._published_datapack_manifest_hash(
        model._published_core_manifest_payload(
            data,
            model_id="IMCC-SF04-EXT",
            version=data["imcc_sf04_datapack_version"],
        )
    )
    monkeypatch.setattr(model, "_PUBLISHED_DATAPACK_SHA256", expected_hash)
    path = tmp_path / "extension-variant.json"
    path.write_text(json.dumps(data, allow_nan=True))
    return path


def _uniform(pack: ImccLoadedDatapack) -> dict[str, float]:
    return {name: 0.125 for name in pack.parent_oxides[:8]}


def test_numeric_parsers_report_invalid_types() -> None:
    with pytest.raises(TypeError, match="cannot parse"):
        model._as_fraction(object())
    with pytest.raises(ImccMalformedDatapackError, match="unknown key"):
        model._build_nu_vector(("A",), {"B": 1}, row_label="coverage row")


def test_load_datapack_rejects_file_root_and_parent_shape_errors(
    tmp_path: Path,
) -> None:
    with pytest.raises(ImccMalformedDatapackError) as missing:
        load_datapack(tmp_path / "missing.json")
    assert missing.value.code == "imcc_malformed_datapack"
    assert "not found" in str(missing.value)

    root = tmp_path / "root.json"
    root.write_text("[]")
    with pytest.raises(ImccMalformedDatapackError) as root_error:
        load_datapack(root)
    assert root_error.value.code == "imcc_malformed_datapack"
    assert "root must be an object" in str(root_error.value)

    for mutate, message in (
        (lambda data: data.pop("imcc_sf04_datapack_version"), "missing"),
        (lambda data: data.update({"parents": ["A"]}), "do not match"),
    ):
        path = tmp_path / f"invalid-{message}.json"
        path.write_text(json.dumps(_base_data()))
        data = json.loads(path.read_text())
        mutate(data)
        path.write_text(json.dumps(data))
        with pytest.raises(ImccMalformedDatapackError) as exc:
            load_datapack(path)
        assert exc.value.code == "imcc_malformed_datapack"
        assert message in str(exc.value)


@pytest.mark.parametrize(
    ("case", "mutate", "message"),
    [
        pytest.param(
            "row-not-object",
            lambda data: data["rows"].__setitem__(0, []),
            "published core row 0",
            id="row-not-object",
        ),
        pytest.param(
            "complex-missing",
            lambda data: data["rows"][0].pop("complex"),
            "missing 'complex'",
            id="complex-missing",
        ),
        pytest.param(
            "nu-not-object",
            lambda data: data["rows"][0].update({"nu": []}),
            "missing 'nu' object",
            id="nu-not-object",
        ),
        pytest.param(
            "thermo-not-numeric",
            lambda data: data["rows"][0].update({"A": []}),
            "A/B must be numeric",
            id="thermo-not-numeric",
        ),
        pytest.param(
            "domain-wrong-shape",
            lambda data: data["rows"][0].update({"T_domain_K": [1.0]}),
            "valid T_domain_K",
            id="domain-wrong-shape",
        ),
        pytest.param(
            "domain-not-numeric",
            lambda data: data["rows"][0].update({"T_domain_K": ["low", 2.0]}),
            "T_domain_K values must be numeric",
            id="domain-not-numeric",
        ),
        pytest.param(
            "basis-missing",
            lambda data: data["rows"][0].pop("T_domain_basis"),
            "missing T_domain_basis",
            id="basis-missing",
        ),
        pytest.param(
            "duplicate-complex",
            lambda data: data["rows"][1].update(
                {"complex": data["rows"][0]["complex"]}
            ),
            "must be unique",
            id="duplicate-complex",
        ),
    ],
)
def test_load_datapack_rejects_core_schema_variants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    del case
    path = _write_core_variant(tmp_path, monkeypatch, mutate)
    with pytest.raises(ImccMalformedDatapackError) as exc:
        load_datapack(path)
    assert exc.value.code == "imcc_malformed_datapack"
    assert message in str(exc.value)


def test_validate_published_core_reports_canonical_serialization_failure() -> None:
    data = _base_data()
    data["rows"][0]["A"] = float("nan")
    with pytest.raises(ImccMalformedDatapackError) as exc:
        model._validate_published_core(
            data,
            model_id="IMCC-SF04",
            version="1.0.2",
        )
    assert exc.value.code == "imcc_malformed_datapack"
    assert "cannot be canonically serialized" in str(exc.value)


@pytest.mark.parametrize(
    ("case", "mutate", "message"),
    [
        pytest.param(
            "model-without-extension",
            lambda data: data.update({"model_id": "IMCC-SF04-EXT"}),
            "requires a recognized extension",
            id="model-without-extension",
        ),
        pytest.param(
            "extension-wrong-model",
            lambda data: data.update({"model_id": "IMCC-SF04"}),
            "requires model_id='IMCC-SF04-EXT'",
            id="extension-wrong-model",
        ),
    ],
)
def test_load_datapack_rejects_model_extension_pairing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    if case == "extension-wrong-model":
        data = _base_data()
        data.update(
            {
                "model_id": "IMCC-SF04",
                "sp_extension": {},
            }
        )
        mutate(data)
        path = tmp_path / "pairing.json"
        path.write_text(json.dumps(data))
    else:
        path = _write_core_variant(tmp_path, monkeypatch, mutate)
    with pytest.raises(ImccMalformedDatapackError) as exc:
        load_datapack(path)
    assert exc.value.code == "imcc_malformed_datapack"
    assert message in str(exc.value)


@pytest.mark.parametrize(
    ("case", "mutate", "message"),
    [
        pytest.param(
            "not-object",
            lambda data: data.update({"sp_extension": []}),
            "must be an object",
            id="not-object",
        ),
        pytest.param(
            "bad-flag",
            lambda data: data["sp_extension"].update({"enable_flag": "no"}),
            "enable_flag",
            id="bad-flag",
        ),
        pytest.param(
            "bad-tier",
            lambda data: data["sp_extension"].update({"tier": "A"}),
            "tier must be 'EXT-SP'",
            id="bad-tier",
        ),
        pytest.param(
            "bad-certification",
            lambda data: data["sp_extension"].update({"certification": "allowed"}),
            "certification must be explicitly denied",
            id="bad-certification",
        ),
        pytest.param(
            "bad-parents",
            lambda data: data["sp_extension"].update({"parents": ["S"]}),
            "parents must be",
            id="bad-parents",
        ),
        pytest.param(
            "empty-rows",
            lambda data: data["sp_extension"].update({"rows": []}),
            "rows must be a non-empty list",
            id="empty-rows",
        ),
        pytest.param(
            "row-not-object",
            lambda data: data["sp_extension"].update({"rows": [[]]}),
            "row 0 is not an object",
            id="row-not-object",
        ),
        pytest.param(
            "missing-provenance",
            lambda data: data["sp_extension"]["rows"][0].update({"provenance": []}),
            "missing provenance object",
            id="missing-provenance",
        ),
        pytest.param(
            "nu-not-object",
            lambda data: data["sp_extension"]["rows"][0].update({"nu": []}),
            "missing 'nu' object",
            id="nu-not-object",
        ),
        pytest.param(
            "unknown-nu-parent",
            lambda data: data["sp_extension"]["rows"][0].update(
                {"nu": {"S": 1, "Unknown": 1}}
            ),
            "unknown parents",
            id="unknown-nu-parent",
        ),
        pytest.param(
            "no-extension-parent",
            lambda data: data["sp_extension"]["rows"][0].update(
                {"nu": {"FeO": 1}}
            ),
            "must consume S or P2O5",
            id="no-extension-parent",
        ),
    ],
)
def test_load_datapack_rejects_extension_schema_variants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    del case
    path = _write_extension_variant(tmp_path, monkeypatch, mutate)
    with pytest.raises(ImccMalformedDatapackError) as exc:
        load_datapack(path)
    assert exc.value.code == "imcc_malformed_datapack"
    assert message in str(exc.value)


def test_evaluate_rejects_unproven_and_invalid_composition_inputs() -> None:
    loaded = load_datapack(BASE_DATAPACK)
    source = loaded.kernel_datapack
    raw = ImccDatapack(
        reactions=source.reactions,
        nu=source.nu,
        A=source.A,
        B=source.B,
        domains=source.domains,
        version=source.version,
        parent_oxides=source.parent_oxides,
    )
    with pytest.raises(ImccUnprovenDatapackError) as unproven:
        evaluate(_uniform(loaded), 2500.0, raw)
    assert unproven.value.code == "imcc_unproven_datapack"

    with pytest.raises(ImccCompositionIncompleteError) as bad_basis_type:
        evaluate(_uniform(loaded), 2500.0, loaded, basis_type="mass")
    assert bad_basis_type.value.code == "imcc_composition_incomplete"

    with pytest.raises(ImccComponentOutsideDomainError) as unknown:
        evaluate({"SiO2": 1.0, "Unknown": 1.0}, 2500.0, loaded)
    assert unknown.value.code == "imcc_component_outside_domain"

    cases = [
        (np.ones((1, 8)), "1-D"),
        (np.full(8, np.nan), "non-finite"),
        (np.zeros(8), "zero"),
    ]
    for composition, message in cases:
        with pytest.raises(ImccCompositionIncompleteError, match=message) as exc:
            evaluate(composition, 2500.0, loaded)
        assert exc.value.code == "imcc_composition_incomplete"

    with pytest.raises(ImccCompositionIncompleteError, match="positive") as exc:
        evaluate(_uniform(loaded), 2500.0, loaded, basis=0.0)
    assert exc.value.code == "imcc_composition_incomplete"

    with pytest.raises(ImccSPComponentRequiresExtensionError) as extra_vector:
        evaluate(np.ones(9), 2500.0, loaded)
    assert extra_vector.value.code == "imcc_sp_extension_required"

    with pytest.raises(ImccSPComponentRequiresExtensionError) as extra_component:
        evaluate(_uniform(loaded), 2500.0, loaded, extra_mol={"S": 0.1})
    assert extra_component.value.code == "imcc_sp_extension_required"


def test_evaluate_extension_vector_guards_and_canonical_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_extension_variant(tmp_path, monkeypatch, lambda data: None)
    pack = load_datapack(path)

    with pytest.raises(ImccCompositionIncompleteError, match="1-D") as ndim:
        evaluate(np.ones((1, 10)), 2500.0, pack, enable_sp_extension=True)
    assert ndim.value.code == "imcc_composition_incomplete"

    with pytest.raises(ImccComponentOutsideDomainError, match="exceeds") as long:
        evaluate(np.ones(11), 2500.0, pack, enable_sp_extension=True)
    assert long.value.code == "imcc_component_outside_domain"

    with pytest.raises(ImccCompositionIncompleteError, match="does not match") as short:
        evaluate(np.ones(9), 2500.0, pack, enable_sp_extension=True)
    assert short.value.code == "imcc_composition_incomplete"

    with pytest.raises(ImccCompositionIncompleteError, match="canonical") as zero:
        evaluate({"S": 1.0}, 2500.0, pack, enable_sp_extension=True)
    assert zero.value.code == "imcc_composition_incomplete"
