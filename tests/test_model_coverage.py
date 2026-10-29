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
                        "fit_residual": None,
                        "accuracy_band": None,
                        "band_basis": {
                            "kind": "none",
                            "explanation": "No row-specific uncertainty band.",
                            "provenance": "coverage",
                        },
                        "independent_check": {
                            "status": "not_found",
                            "equilibrium": None,
                            "primary_source": None,
                            "conditions": None,
                            "measured_value": None,
                            "predicted_value": None,
                            "miss": None,
                            "miss_to_band_ratio": None,
                            "reason": "No row-specific equilibrium check was identified.",
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


def test_below_domain_extrapolation_reports_extension_notice() -> None:
    pack = load_datapack(BASE_DATAPACK)
    below = evaluate(
        _uniform(pack), 1600.0, pack, allow_extrapolation=True
    )
    inside = evaluate(
        _uniform(pack), 1800.0, pack, allow_extrapolation=True
    )

    extension_notices = tuple(
        notice
        for notice in below.labels.notices
        if notice.startswith("temperature-domain-extension:")
    )
    assert len(extension_notices) == 1
    assert "ΔrCp = 0" in extension_notices[0]
    assert "declared domains 1700–3000 K" in extension_notices[0]
    assert "uncertainty band unavailable" in extension_notices[0]
    assert not any(
        notice.startswith("temperature-domain-extension:")
        for notice in inside.labels.notices
    )


def test_below_domain_notice_names_active_sp_row_domain() -> None:
    pack = load_datapack(
        Path("src/openimcc/data/packs/imcc-sf04-d066-ext-v1.json")
    )
    result = evaluate(
        {"FeO": 0.5, "S": 0.5},
        300.0,
        pack,
        enable_sp_extension=True,
        allow_extrapolation=True,
        allow_out_of_envelope=True,
    )
    extension_notices = tuple(
        notice
        for notice in result.labels.notices
        if notice.startswith("temperature-domain-extension:")
    )
    assert len(extension_notices) == 1
    assert "declared domains 400–1500 K" in extension_notices[0]


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


def test_na2si2o5_low_temperature_notice_fires_only_below_its_domain() -> None:
    from openimcc import evaluate

    melt = {"Na2O": 33.0, "SiO2": 67.0}
    below = evaluate(melt, 1500.0, allow_extrapolation=True)
    marker = "low-temperature-uncertain: the Na2Si2O5 association"
    assert sum(marker in notice for notice in below.labels.notices) == 1
    floor = evaluate(melt, 1699.0, allow_extrapolation=True)
    assert sum(marker in notice for notice in floor.labels.notices) == 1
    # Flagging is metadata only: pin solved parent activities at the cutoff.
    assert floor.parent_activity[0] == pytest.approx(
        0.2795921623351865, rel=1e-12, abs=0.0
    )
    measured = next(n for n in below.labels.notices if marker in n)
    assert "Piacente and Matoušek (1973)" in measured
    assert "1.5–3x" in measured
    assert floor.parent_activity[6] == pytest.approx(
        1.3724549799095924e-7, rel=1e-12, abs=0.0
    )
    at_floor = evaluate(melt, 1700.0, allow_extrapolation=True)
    assert not any(marker in notice for notice in at_floor.labels.notices)
    assert at_floor.parent_activity[0] == pytest.approx(
        0.2796869197906698, rel=1e-12, abs=0.0
    )
    assert at_floor.parent_activity[6] == pytest.approx(
        1.3875098072777926e-7, rel=1e-12, abs=0.0
    )
    inside = evaluate(melt, 1800.0, allow_extrapolation=True)
    assert not any(marker in notice for notice in inside.labels.notices)
    sodium_free = evaluate({"MgO": 40.0, "SiO2": 60.0}, 1500.0, allow_extrapolation=True)
    assert not any(marker in notice for notice in sodium_free.labels.notices)
    candidate = load_datapack("src/openimcc/data/packs/imcc-sf04-d066-v2.json")
    other_pack = evaluate(melt, 1500.0, candidate, allow_extrapolation=True)
    assert any(n.startswith("low-temperature-uncertain-unscored:") for n in other_pack.labels.notices)


def _cao_sio2_bias_notices(result) -> tuple[str, ...]:
    return tuple(
        notice
        for notice in result.labels.notices
        if notice.startswith("cao-sio2-activity-bias:")
    )


_CAO_SIO2_MODERATE = (
    "cao-sio2-activity-bias: moderate — effectively binary CaO–SiO2 melt "
    "with x(CaO) in [0.42, 0.56) on the CaO–SiO2 sub-basis at 1873–1933 K; "
    "Kume 2000 and Omori–Sanbongi 1961 show low a(SiO2); this pack raises "
    "predicted log10(a(SiO2)) by 0.045–0.410 dex over x(CaO)=[0.42, 0.56)"
)
_CAO_SIO2_STRONG = (
    "cao-sio2-activity-bias: strong — effectively binary CaO–SiO2 melt "
    "with x(CaO) in [0.56, 0.70] on the CaO–SiO2 sub-basis at 1873–1933 K, "
    "near Ca2SiO4; no Ca3SiO5 associate. Stolyarova 1991 Table IIb's "
    "0.6–1.0 dex low scale conflicts with Table Ib. That comparison's prior "
    "default residuals are not carried as v5 scores; this pack shifts predicted "
    "log10(a(SiO2)) non-monotonically by 0.076–0.479 dex, peaking near "
    "x(CaO)=0.614"
)


def test_cao_sio2_activity_bias_notice_severities_at_1933() -> None:
    """x(CaO)=0.50 is moderate and x(CaO)=0.62 is strong; values stay put."""
    moderate = evaluate({"CaO": 0.50, "SiO2": 0.50}, 1933.0)
    assert _cao_sio2_bias_notices(moderate) == (_CAO_SIO2_MODERATE,)
    assert moderate.parent_activity[
        moderate.parent_oxides.index("SiO2")
    ] == pytest.approx(0.15517084002355458, rel=1e-12, abs=0.0)
    assert moderate.parent_activity[
        moderate.parent_oxides.index("CaO")
    ] == pytest.approx(0.003186326308813352, rel=1e-12, abs=0.0)

    strong = evaluate({"CaO": 0.62, "SiO2": 0.38}, 1933.0)
    assert _cao_sio2_bias_notices(strong) == (_CAO_SIO2_STRONG,)
    assert strong.parent_activity[
        strong.parent_oxides.index("SiO2")
    ] == pytest.approx(0.010778776205447687, rel=1e-12, abs=0.0)
    assert strong.parent_activity[
        strong.parent_oxides.index("CaO")
    ] == pytest.approx(0.024082042568103465, rel=1e-12, abs=0.0)

    # Equal moles from a weight input: masses are the oxide molar masses.
    from openimcc.model import _OXIDE_MOLAR_MASS_G_MOL

    by_weight = evaluate(
        {
            "CaO": _OXIDE_MOLAR_MASS_G_MOL["CaO"],
            "SiO2": _OXIDE_MOLAR_MASS_G_MOL["SiO2"],
        },
        1933.0,
        basis_type="wt",
    )
    assert _cao_sio2_bias_notices(by_weight) == (_CAO_SIO2_MODERATE,)


def test_cao_sio2_activity_bias_notice_skips_lunar_basalts_and_low_xcao() -> None:
    lunar_mare = {
        "Al2O3": 14.2,
        "CaO": 11.4,
        "FeO": 10.9,
        "K2O": 0.1,
        "MgO": 8.1,
        "Na2O": 0.4,
        "SiO2": 45.4,
        "TiO2": 3.2,
    }
    low_ti_mare = {
        "Al2O3": 14.214,
        "CaO": 11.637,
        "FeO": 16.353,
        "MgO": 9.463,
        "Na2O": 0.182,
        "SiO2": 46.234,
        "TiO2": 1.587,
    }
    for composition in (lunar_mare, low_ti_mare):
        result = evaluate(composition, 1933.0, basis_type="wt")
        assert _cao_sio2_bias_notices(result) == ()

    low_x = evaluate({"CaO": 0.30, "SiO2": 0.70}, 1933.0)
    assert _cao_sio2_bias_notices(low_x) == ()


def test_cao_sio2_activity_bias_notice_boundaries() -> None:
    """Closed x(CaO) and T ends fire; the shared 0.56 edge is strong."""
    assert _cao_sio2_bias_notices(
        evaluate({"CaO": 0.42, "SiO2": 0.58}, 1933.0)
    ) == (_CAO_SIO2_MODERATE,)
    assert _cao_sio2_bias_notices(
        evaluate({"CaO": 0.419, "SiO2": 0.581}, 1933.0)
    ) == ()
    assert _cao_sio2_bias_notices(
        evaluate({"CaO": 0.559, "SiO2": 0.441}, 1933.0)
    ) == (_CAO_SIO2_MODERATE,)
    assert _cao_sio2_bias_notices(
        evaluate({"CaO": 0.56, "SiO2": 0.44}, 1933.0)
    ) == (_CAO_SIO2_STRONG,)
    assert _cao_sio2_bias_notices(
        evaluate({"CaO": 0.70, "SiO2": 0.30}, 1933.0)
    ) == (_CAO_SIO2_STRONG,)
    assert _cao_sio2_bias_notices(
        evaluate({"CaO": 0.701, "SiO2": 0.299}, 1933.0)
    ) == ()

    binary = {"CaO": 0.50, "SiO2": 0.50}
    assert _cao_sio2_bias_notices(evaluate(binary, 1873.0)) == (_CAO_SIO2_MODERATE,)
    assert _cao_sio2_bias_notices(evaluate(binary, 1933.0)) == (_CAO_SIO2_MODERATE,)
    assert _cao_sio2_bias_notices(evaluate(binary, 1872.0)) == ()
    assert _cao_sio2_bias_notices(evaluate(binary, 1934.0)) == ()
    # 1973 K is an unvalidated model map, not a notice boundary.
    assert _cao_sio2_bias_notices(evaluate(binary, 1973.0)) == ()

    # Share 0.98 exactly, each other parent below 0.01, x(CaO)=0.50.
    at_share = evaluate(
        {"CaO": 490.0, "SiO2": 490.0, "MgO": 9.0, "FeO": 9.0, "Al2O3": 2.0},
        1933.0,
    )
    assert _cao_sio2_bias_notices(at_share) == (_CAO_SIO2_MODERATE,)
    below_share = evaluate(
        {"CaO": 489.0, "SiO2": 490.0, "MgO": 9.0, "FeO": 9.0, "Al2O3": 3.0},
        1933.0,
    )
    assert _cao_sio2_bias_notices(below_share) == ()

    # Another parent at exactly 0.01 is outside the binary gate.
    at_other = evaluate({"CaO": 495.0, "SiO2": 495.0, "MgO": 10.0}, 1933.0)
    assert _cao_sio2_bias_notices(at_other) == ()
    under_other = evaluate({"CaO": 496.0, "SiO2": 495.0, "MgO": 9.0}, 1933.0)
    assert _cao_sio2_bias_notices(under_other) == (_CAO_SIO2_MODERATE,)


_CAO_SIO2_UNSCORED = (
    "cao-sio2-activity-bias-unscored: the CaO–SiO2 silica-activity bias was "
    f"measured on {model._DEFAULT_D066_V5_IDENTITY[0]} "
    f"(binding digest {model._DEFAULT_D066_V5_IDENTITY[1]}) and has not "
    "been re-scored for this IMCC-SF04-D066 pack"
)


def test_cao_sio2_evidence_identity_matches_the_default_pack() -> None:
    """A new default pack must not silently demote the measured notice."""
    pack = load_datapack()
    loaded = (pack.model_id, pack.binding_digest)
    evidence = model._DEFAULT_D066_V5_IDENTITY
    assert loaded == evidence, (
        "the default melt pack identity no longer matches the CaO-SiO2 "
        f"evidence identity {evidence[0]} {evidence[1]}; loaded "
        f"{loaded[0]} {loaded[1]}. Re-score the CaO-SiO2 evidence and "
        "update the measured identity."
    )
    assert model._melt_notice_score("cao-sio2-activity-bias", loaded) == "scored"


def test_cao_sio2_other_pack_gets_the_unscored_notice() -> None:
    """A different candidate digest does not inherit the measured residuals."""
    candidate = load_datapack("src/openimcc/data/packs/imcc-sf04-d066-v2.json")
    assert candidate.model_id == model._DEFAULT_D066_V5_IDENTITY[0]
    assert candidate.binding_digest != model._DEFAULT_D066_V5_IDENTITY[1]

    scored = evaluate({"CaO": 0.50, "SiO2": 0.50}, 1933.0, candidate)
    assert _cao_sio2_bias_notices(scored) == ()
    unscored = tuple(
        notice
        for notice in scored.labels.notices
        if notice.startswith("cao-sio2-activity-bias-unscored:")
    )
    assert unscored == (_CAO_SIO2_UNSCORED,)

    outside = evaluate({"CaO": 0.30, "SiO2": 0.70}, 1933.0, candidate)
    assert _cao_sio2_bias_notices(outside) == ()
    assert not any(
        notice.startswith("cao-sio2-activity-bias-unscored:")
        for notice in outside.labels.notices
    )


_MULTICOMPONENT_ACTIVITY_PATTERN_MARKERS = (
    "multicomponent-activity-pattern:",
    "multicomponent-activity-pattern-unscored:",
)


def test_multicomponent_pattern_cmas_values_unchanged() -> None:
    composition = {"SiO2": 50.0, "CaO": 25.0, "MgO": 15.0, "Al2O3": 10.0}
    result = evaluate(composition, 1823.0, basis_type="wt")
    notices = tuple(
        notice
        for notice in result.labels.notices
        if notice.startswith(_MULTICOMPONENT_ACTIVITY_PATTERN_MARKERS)
    )
    assert len(notices) == 1
    notice = notices[0]
    assert notice.startswith("multicomponent-activity-pattern:")
    assert "Kume et al. (2000) synthetic CAS/CMS/CMAS slags" in notice
    assert "CAS (Table 2: slope −0.238 dex/NBO/T, n=47)" in notice
    assert "CMS (Table 3: slope −0.362 dex/NBO/T, n=27)" in notice
    assert "Table 4 has no SiO2 rows" in notice
    assert "four-oxide CMAS RMSEs are 0.81 dex (CaO, n=15)" in notice
    assert "0.57 (MgO, n=15)" in notice
    assert "this binary result does not establish the multicomponent pattern" in notice

    # Hex literals pin the calculated outputs from before this notice existed.
    assert tuple(float(value).hex() for value in result.parent_activity) == (
        "0x1.5c1424293252fp-3",
        "0x1.41b5bca62510bp-6",
        "0x0.0p+0",
        "0x1.cababe1126f00p-12",
        "0x1.338d81f5763bfp-6",
        "0x0.0p+0",
        "0x0.0p+0",
        "0x0.0p+0",
    )
    assert tuple(float(value).hex() for value in result.parent_gamma) == (
        "0x1.6d9fbeb88b0e3p-2",
        "0x1.79cccb1c0873fp-4",
        "0x0.0p+0",
        "0x1.c1b7f159d60d1p-10",
        "0x1.56a278fe2f4eap-2",
        "0x0.0p+0",
        "0x0.0p+0",
        "0x0.0p+0",
    )


def test_multicomponent_pattern_skips_binary() -> None:
    result = evaluate({"CaO": 0.50, "SiO2": 0.50}, 1823.0)
    assert not any(
        notice.startswith(_MULTICOMPONENT_ACTIVITY_PATTERN_MARKERS)
        for notice in result.labels.notices
    )


def test_multicomponent_pattern_gate_boundaries() -> None:
    parents = ("SiO2", "CaO", "MgO", "Al2O3")
    identity = model._DEFAULT_D066_V5_IDENTITY
    moles = np.array((0.70, 0.10, 0.10, 0.10))
    notice = model._multicomponent_activity_pattern_notice

    assert notice(parents, moles, 1700.0, *identity).startswith(
        "multicomponent-activity-pattern:"
    )
    assert notice(parents, moles, 3000.0, *identity).startswith(
        "multicomponent-activity-pattern:"
    )
    assert notice(parents, moles, 1699.99, *identity) is None
    assert notice(parents, moles, 3000.01, *identity) is None
    assert notice(parents, np.array((0.97, 0.01, 0.01, 0.01)), 1823.0, *identity) is None


def test_multicomponent_pattern_identity_scope() -> None:
    default = load_datapack()
    evidence = model._DEFAULT_D066_V5_IDENTITY
    assert (default.model_id, default.binding_digest) == evidence
    assert (
        model._melt_notice_score("multicomponent-activity-pattern", evidence)
        == "scored"
    )

    candidate = load_datapack(
        "src/openimcc/data/packs/imcc-sf04-d066-v2.json"
    )
    assert candidate.model_id == evidence[0]
    assert candidate.binding_digest != evidence[1]
    result = evaluate(
        {"SiO2": 50.0, "CaO": 25.0, "MgO": 15.0, "Al2O3": 10.0},
        1823.0,
        candidate,
        basis_type="wt",
    )
    notices = tuple(
        notice
        for notice in result.labels.notices
        if notice.startswith(_MULTICOMPONENT_ACTIVITY_PATTERN_MARKERS)
    )
    assert len(notices) == 1
    assert notices[0].startswith("multicomponent-activity-pattern-unscored:")
    assert "this pack has not been scored" in notices[0]


def test_silicate_vapour_product_notice_gate_identity_and_solve_values() -> None:
    diopside = {"CaO": 1.0, "MgO": 1.0, "SiO2": 2.0}
    result = evaluate(diopside, 1668.0, allow_extrapolation=True)
    notices = [n for n in result.labels.notices if n.startswith("silicate-vapour-product-low-near-liquidus:")]
    assert len(notices) == 1
    assert "−0.53" in notices[0]
    assert "low by 0.53 log10 units" in notices[0]
    assert "split between melt activities and liquid standards unresolved" in notices[0]
    assert "incongruent-melting onset" in notices[0]
    assert "crystal data extended" in notices[0]
    assert result.parent_activity == pytest.approx(
        (0.1060781382929233, 0.02639230008993675, 0.0,
         0.00041927848957406914, 0.0, 0.0, 0.0, 0.0),
        rel=1e-12, abs=0.0,
    )

    phase_cases = (
        ("diopside", diopside, 1668.0, "−0.53"),
        ("akermanite", {"CaO": 2.0, "MgO": 1.0, "SiO2": 2.0}, 1727.0, "+0.11"),
        ("gehlenite", {"CaO": 2.0, "Al2O3": 1.0, "SiO2": 1.0}, 1823.0, "−0.51"),
        ("anorthite", {"CaO": 1.0, "Al2O3": 1.0, "SiO2": 2.0}, 1826.0, "+0.044"),
        ("wollastonite", {"CaO": 1.0, "SiO2": 1.0}, 1817.0, "+0.0028"),
    )
    for phase, composition, melting_point, miss in phase_cases:
        notice = next(
            n for n in evaluate(
                composition, melting_point, allow_extrapolation=True
            ).labels.notices
            if n.startswith("silicate-vapour-product-low-near-liquidus:")
        )
        assert phase in notice
        assert miss in notice
        direction = "low" if miss.startswith("−") else "high"
        assert f"{direction} by {abs(float(miss.replace('−', '-'))):g} log10 units" in notice

    assert any(
        n.startswith("silicate-vapour-product-low-near-liquidus:")
        for n in evaluate(diopside, 1618.0, allow_extrapolation=True).labels.notices
    )
    assert any(
        n.startswith("silicate-vapour-product-low-near-liquidus:")
        for n in evaluate(diopside, 1718.0, allow_extrapolation=True).labels.notices
    )
    assert not any(
        n.startswith("silicate-vapour-product-low-near-liquidus")
        for n in evaluate(diopside, 1718.1, allow_extrapolation=True).labels.notices
    )
    boundary_distance = {"CaO": 1.04, "MgO": 0.96, "SiO2": 2.0}
    outside_distance = {"CaO": 1.046875, "MgO": 0.953125, "SiO2": 2.0}
    assert any(
        n.startswith("silicate-vapour-product-low-near-liquidus:")
        for n in evaluate(boundary_distance, 1668.0, allow_extrapolation=True).labels.notices
    )
    assert not any(
        n.startswith("silicate-vapour-product-low-near-liquidus")
        for n in evaluate(outside_distance, 1668.0, allow_extrapolation=True).labels.notices
    )
    at_share_boundary = {
        "CaO": 0.98, "MgO": 0.98, "SiO2": 1.96,
        "FeO": 0.02, "Na2O": 0.02, "K2O": 0.02, "TiO2": 0.02,
    }
    below_share_boundary = {
        "CaO": 0.979, "MgO": 0.979, "SiO2": 1.958,
        "FeO": 0.021, "Na2O": 0.021, "K2O": 0.021, "TiO2": 0.021,
    }
    assert any(
        n.startswith("silicate-vapour-product-low-near-liquidus:")
        for n in evaluate(at_share_boundary, 1668.0, allow_extrapolation=True).labels.notices
    )
    assert not any(
        n.startswith("silicate-vapour-product-low-near-liquidus")
        for n in evaluate(below_share_boundary, 1668.0, allow_extrapolation=True).labels.notices
    )
    at_other_parent_limit = {"CaO": 0.99, "MgO": 0.99, "SiO2": 1.98, "FeO": 0.04}
    below_other_parent_limit = {"CaO": 0.991, "MgO": 0.991, "SiO2": 1.982, "FeO": 0.036}
    assert not any(
        n.startswith("silicate-vapour-product-low-near-liquidus")
        for n in evaluate(at_other_parent_limit, 1668.0, allow_extrapolation=True).labels.notices
    )
    assert any(
        n.startswith("silicate-vapour-product-low-near-liquidus:")
        for n in evaluate(below_other_parent_limit, 1668.0, allow_extrapolation=True).labels.notices
    )
    other_pack = load_datapack("src/openimcc/data/packs/imcc-sf04-d066-v2.json")
    unscored = evaluate(diopside, 1668.0, other_pack, allow_extrapolation=True)
    assert any(n.startswith("silicate-vapour-product-low-near-liquidus-unscored:") for n in unscored.labels.notices)

    lunar_basalt = {"Al2O3": 14.2, "CaO": 11.4, "FeO": 10.9, "K2O": 0.1, "MgO": 8.1, "Na2O": 0.4, "SiO2": 45.4, "TiO2": 3.2}
    assert not any(n.startswith("silicate-vapour-product-low-near-liquidus") for n in evaluate(lunar_basalt, 1668.0, basis_type="wt", allow_extrapolation=True).labels.notices)
    assert not any(n.startswith("silicate-vapour-product-low-near-liquidus") for n in evaluate(diopside, 1617.9, allow_extrapolation=True).labels.notices)
    assert not any(n.startswith("silicate-vapour-product-low-near-liquidus") for n in evaluate({"CaO": 1.0, "MgO": 1.0, "SiO2": 1.8, "Al2O3": 0.2}, 1668.0, allow_extrapolation=True).labels.notices)


def test_silicate_product_evidence_identity_is_pinned() -> None:
    pack = load_datapack()
    assert (pack.model_id, pack.binding_digest) == (
        model._DEFAULT_D066_V5_IDENTITY
    ), "default pack moved; re-score melt notice evidence before retaining measured claims"
    assert (
        model._melt_notice_score(
            "silicate-vapour-product-low-near-liquidus",
            model._DEFAULT_D066_V5_IDENTITY,
        )
        == "scored"
    )


def test_na2o_activity_bias_notice_gate_and_numeric_identity() -> None:
    marker = "na2o-activity-bias:"
    mathieu_iii1_like = evaluate(
        {"SiO2": 49.1, "Al2O3": 0.05, "CaO": 41.29, "MgO": 5.06, "Na2O": 4.5},
        1673.15, basis_type="wt", allow_extrapolation=True,
    )
    mathieu_like = evaluate(
        {"SiO2": 8.81, "Na2O": 24.0, "CaO": 35.0, "MgO": 24.0, "Al2O3": 8.19},
        1673.15, basis_type="wt", allow_extrapolation=True,
        allow_out_of_envelope=True,
    )
    pak_binary = evaluate(
        {"Na2O": 60.0, "SiO2": 40.0}, 1673.15,
        allow_extrapolation=True, allow_out_of_envelope=True,
    )
    assert any(n.startswith(marker) for n in mathieu_iii1_like.labels.notices)
    assert any(n.startswith(marker) for n in mathieu_like.labels.notices)
    assert any(n.startswith(marker) for n in pak_binary.labels.notices)
    assert mathieu_like.parent_activity[mathieu_like.parent_oxides.index("Na2O")] == 0.1068994643265644
    assert mathieu_like.parent_activity[mathieu_like.parent_oxides.index("SiO2")] == pytest.approx(
        8.65523987822024e-08, rel=1e-12, abs=0.0
    )

    sodium_free = evaluate({"SiO2": 0.4, "CaO": 0.6}, 1673.15, allow_extrapolation=True)
    high_silica = evaluate({"Na2O": 20.0, "SiO2": 80.0}, 1673.15, allow_extrapolation=True)
    albite_like = evaluate(
        {"Na2O": 0.5, "Al2O3": 0.5, "SiO2": 3.0},
        1673.15, allow_extrapolation=True,
    )
    assert not any(n.startswith(marker) for n in sodium_free.labels.notices)
    assert not any(n.startswith(marker) for n in high_silica.labels.notices)
    assert not any(n.startswith(marker) for n in albite_like.labels.notices)


def test_na2o_activity_bias_evidence_identity_is_pinned() -> None:
    pack = load_datapack()
    assert (pack.model_id, pack.binding_digest) == (
        model._DEFAULT_D066_V5_IDENTITY
    ), "default pack moved; re-score Na2O bias evidence before retaining measured claims"
    assert (
        model._melt_notice_score("na2o-activity-bias", model._DEFAULT_D066_V5_IDENTITY)
        == "scored"
    )


@pytest.mark.parametrize(
    "composition,expected",
    [
        ({"Na2O": 0.5, "SiO2": 0.5}, True),
        ({"Na2O": 0.1, "CaO": 0.3, "SiO2": 0.6}, True),
        ({"Na2O": 0.1, "MgO": 0.3, "SiO2": 0.6}, True),
        ({"Na2O": 0.499, "SiO2": 0.501}, False),
        ({"Na2O": 0.1, "CaO": 0.299, "SiO2": 0.601}, False),
        ({"CaO": 0.5, "SiO2": 0.5}, False),
    ],
)
def test_na2o_activity_bias_other_digest_uses_same_gate(
    composition: dict[str, float], expected: bool,
) -> None:
    pack = load_datapack("src/openimcc/data/packs/imcc-sf04-d066-v2.json")
    assert pack.model_id == model._DEFAULT_D066_V5_IDENTITY[0]
    assert pack.binding_digest != model._DEFAULT_D066_V5_IDENTITY[1]
    result = evaluate(
        composition, 1673.15, pack,
        allow_extrapolation=True, allow_out_of_envelope=True,
    )
    notices = tuple(n for n in result.labels.notices if n.startswith("na2o-activity-bias"))
    assert notices == ((
        "na2o-activity-bias-unscored: Na2O activity and Na(g) pressure bias evidence "
        "was measured on the default D066 v5 pack; this pack has not been scored.",
    ) if expected else ())
