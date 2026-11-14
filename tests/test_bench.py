"""IMCC-only empirical residual runner."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import yaml

import pytest

from openimcc._constants import R_PUBLISHED_GAS_J_MOL_K
from openimcc import ImccMalformedDatapackError, evaluate, load_datapack
from openimcc.json_utils import DuplicateJsonKeyError
from openimcc.bench import (
    _PACK_OVERLAY_BY_VERSION,
    _PACK_OVERLAY_MODEL_BY_VERSION,
    _CALLER_STANDARD_STATE_CONVENTIONS,
    _STANDARD_STATE_BY_CONVENTION,
    _binary_subslice,
    _convert_solid_activity_to_liquid,
    _evaluate_imcc,
    _is_research_overlay,
    _standard_state_for_point,
    load_bench_set,
    load_pack,
    main,
    render_report,
    run_bench,
    _solid_to_liquid_log10_shift,
    _controlled_text,
)

DATAPACK_PATH = Path("src/openimcc/data/packs/imcc-sf04-v1.0.2.json")
_REPO_ROOT = Path(__file__).resolve().parents[1]


def _independent_csv_row(
    relative_path: str,
    species: str,
    temperature_K: float,
    *,
    extrapolate_last_row: bool = False,
) -> dict[str, str]:
    with (_REPO_ROOT / relative_path).open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["species_name"] == species
        ]
    in_range = [
        row
        for row in rows
        if float(row["T_min"]) <= temperature_K <= float(row["T_max"])
    ]
    if in_range:
        return max(in_range, key=lambda row: float(row["T_min"]))
    if extrapolate_last_row and rows:
        return max(rows, key=lambda row: float(row["T_max"]))
    raise AssertionError(
        f"no independent table row for {species} at {temperature_K:g} K"
    )


def _independent_janaf_apparent_gibbs_j_mol(
    table_id: str, temperature_K: float
) -> float:
    table_path = _REPO_ROOT / "data-src" / "janaf" / f"{table_id}.yaml"
    table = json.loads(table_path.read_text(encoding="utf-8"))["table"]
    reference_row = next(
        row
        for row in table["values"]
        if float(row["temperature"]["value"]) == 298.15
    )
    formation_enthalpy_298_kj_mol = float(
        reference_row["formation_enthalpy"]["value"]
    )
    nodes = sorted(
        (
            float(row["temperature"]["value"]),
            float(row["enthalpy_increment"]["value"]),
            float(row["entropy"]["value"]),
        )
        for row in table["values"]
        if row["enthalpy_increment"]["value"] is not None
        and row["entropy"]["value"] is not None
    )

    def interpolate(column: int) -> float:
        right = next(
            index
            for index, node in enumerate(nodes)
            if node[0] >= temperature_K
        )
        if nodes[right][0] == temperature_K:
            while right + 1 < len(nodes) and nodes[right + 1][0] == temperature_K:
                right += 1
            return nodes[right][column]
        low_T, low_value = nodes[right - 1][0], nodes[right - 1][column]
        high_T, high_value = nodes[right][0], nodes[right][column]
        return low_value + (high_value - low_value) * (
            (temperature_K - low_T) / (high_T - low_T)
        )

    enthalpy_increment_kj_mol = interpolate(1)
    entropy_j_mol_k = interpolate(2)
    return (
        (formation_enthalpy_298_kj_mol + enthalpy_increment_kj_mol) * 1000.0
        - temperature_K * entropy_j_mol_k
    )


def _independent_condensate_gibbs_j_mol(
    row: dict[str, str], temperature_K: float
) -> float:
    gas_constant = R_PUBLISHED_GAS_J_MOL_K
    t = temperature_K / 1000.0
    phi = sum(
        float(row[f"dG_{coefficient}"]) * t**power
        for power, coefficient in enumerate("ABCDE")
    )
    return -gas_constant * temperature_K * phi + (
        gas_constant * 1000.0 * float(row["dH298_R"])
    )


# In-domain CMAS slag, 8-parent IMCC basis (missing parents are zero).
_CMAS_WT = {
    "SiO2": 50.0,
    "CaO": 20.0,
    "MgO": 15.0,
    "Al2O3": 15.0,
}
_K_CMAS_WT = {**_CMAS_WT, "K2O": 1.0}


@pytest.mark.parametrize(
    "pack_name",
    [
        "imcc-sf04-v1.0.2.json",
        "imcc-sf04-ext-v4.json",
        "imcc-sf04-d066-v1.json",
        "imcc-sf04-d066-v2.json",
        "imcc-sf04-d066-v3.json",
        "imcc-sf04-d066-v1-kcaalsi2o7.json",
        "imcc-sf04-d066-ext-v1.json",
    ],
)
def test_packaged_core_and_extension_packs_run_through_bench(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pack_name: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    engine = load_pack(Path(pack_name))

    assert engine.enable_sp_extension is bool(engine.pack.extension_parents)
    status, activities, _, reason, _, _notices = _evaluate_imcc(
        engine,
        {"SiO2": 100.0},
        2000.0,
    )

    assert status == "ok", reason
    assert activities["SiO2"] > 0.0


@pytest.mark.parametrize(
    ("pack_name", "expected_residuals"),
    [
        (
            "imcc-sf04-v1.0.2.json",
            {
                "hastie_k_1917_186": -1.199993595,
                "hastie_k_1917_282": -1.255882397,
                "hastie_k_1955_825": -1.258877655,
            },
        ),
        (
            "imcc-sf04-ext-v4.json",
            {
                "hastie_k_1917_186": -1.199993595,
                "hastie_k_1917_282": -1.255882397,
                "hastie_k_1955_825": -1.258877655,
            },
        ),
        (
            "imcc-sf04-d066-v1.json",
            {
                "hastie_k_1917_186": -1.166030912,
                "hastie_k_1917_282": -1.221908035,
                "hastie_k_1955_825": -1.220289961,
            },
        ),
        (
            "imcc-sf04-d066-v2.json",
            {
                "hastie_k_1917_186": -1.179372421,
                "hastie_k_1917_282": -1.235244101,
                "hastie_k_1955_825": -1.231518761,
            },
        ),
        (
            "imcc-sf04-d066-v3.json",
            {
                "hastie_k_1917_186": -1.179372421,
                "hastie_k_1917_282": -1.235244101,
                "hastie_k_1955_825": -1.231518761,
            },
        ),
        (
            "imcc-sf04-d066-v1-kcaalsi2o7.json",
            {
                "hastie_k_1917_186": -1.166030912,
                "hastie_k_1917_282": -1.221908035,
                "hastie_k_1955_825": -1.220289961,
            },
        ),
        (
            "imcc-sf04-d066-ext-v1.json",
            {
                "hastie_k_1917_186": -1.166030912,
                "hastie_k_1917_282": -1.221908035,
                "hastie_k_1955_825": -1.220289961,
            },
        ),
    ],
)
def test_hastie_case4_k_residuals_are_pinned_per_pack(
    pack_name: str,
    expected_residuals: dict[str, float],
) -> None:
    report = run_bench(
        Path("benchmarks/sets/basalt-bench-set-v1.yaml"),
        Path("src/openimcc/data/packs") / pack_name,
        populations=("hastie1981_kems",),
        species=("K",),
    )

    rows = {row.point_id: row for row in report.points}
    assert set(rows) == set(expected_residuals)
    for point_id, expected_residual in expected_residuals.items():
        assert rows[point_id].status == "ok"
        assert rows[point_id].residual == pytest.approx(
            expected_residual, abs=5.0e-8
        )


def _write_fixture(path: Path, points: list[dict], compositions: dict | None = None) -> Path:
    payload = {
        "schema_version": "melt-activity-bench.v1",
        "title": "inline IMCC bench runner fixture",
        "compositions": compositions
        or {
            "cmas_in_domain": {
                "material_class": "cmas_slag",
                "composition_wt_pct": dict(_CMAS_WT),
            }
        },
        "points": points,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_excess_candidate(
    path: Path,
    *,
    basis: str = "analytical_parent",
    components: tuple[str, str] | None = None,
) -> Path:
    source = Path("src/openimcc/data/packs/imcc-sf04-d066-v5.json")
    data = json.loads(source.read_text(encoding="utf-8"))
    data["model_id"] = "IMCC-SF04-D066-EXCESS-CANDIDATE"
    components = components or (
        ("Na2O", "CaO") if basis == "analytical_parent"
        else ("Na2O", "Na2SiO3")
    )
    data["excess_gibbs"] = {
        "terms": [{
            "basis": basis,
            "components": list(components),
            "W_H_J_mol": -60000.0 if basis == "analytical_parent" else 1000.0,
            "W_S_J_mol_K": 0.0,
            "provenance": "synthetic bench disclosure test",
            "calibration_role": "notice projection test",
            "valid_composition": {name: [0.0, 1.0] for name in components},
            "T_range_K": [2000.0, 2100.0],
            "fit_status": "empirical-fit",
        }]
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "basis,components",
    [
        ("analytical_parent", ("Na2O", "CaO")),
        ("species_pair", ("Na2O", "Na2SiO3")),
    ],
)
def test_bench_carries_candidate_and_calibration_notices(
    tmp_path: Path, basis: str, components: tuple[str, str]
) -> None:
    composition = {
        "Na2O": 61.9789,
        "CaO": 56.0774,
        "SiO2": 240.3372,
        "MgO": 40.3044,
    }
    point = {
        "id": "candidate_na_activity",
        "population": "candidate",
        "composition_id": "candidate_mix",
        "material_class": "cmas_slag",
        "temperature_K": 1800.0,
        "parent_oxide": "Na2O",
        "species": "Na",
        "observable": "activity",
        "measured": 0.2,
        "units": "dimensionless",
        "score": True,
    }
    fixture = _write_fixture(
        tmp_path / "bench.yaml",
        [point],
        {"candidate_mix": {"material_class": "cmas_slag", "composition_wt_pct": composition}},
    )
    candidate = _write_excess_candidate(
        tmp_path / "candidate.json", basis=basis, components=components
    )

    report = run_bench(fixture, candidate)

    row = report.points[0]
    assert row.status == "ok"
    interaction = "–".join(components)
    assert any(interaction in notice and "empirical-fit" in notice for notice in row.notices)
    assert any(
        notice.startswith("excess-gibbs-calibration-range:") and "T=1800 K" in notice
        for notice in row.notices
    )
    assert report.notices == row.notices
    rendered = render_report(report)
    assert "empirical-fit" in rendered
    assert "excess-gibbs-calibration-range:" in rendered


def _three_point_fixture(path: Path) -> Path:
    return _write_fixture(
        path,
        [
            {
                "id": "ok_sio2_1800",
                "population": "inline_ok",
                "composition_id": "cmas_in_domain",
                "material_class": "cmas_slag",
                "temperature_K": 1800.0,
                "parent_oxide": "SiO2",
                "species": "SiO",
                "observable": "activity",
                "measured": 0.2,
                "units": "dimensionless",
                "score": True,
                "convention": (
                    "a(SiO2) on the parent-oxide formula-unit basis against "
                    "a pure-liquid-SiO2 standard state"
                ),
            },
            {
                "id": "ood_t500",
                "population": "inline_ood",
                "composition_id": "cmas_in_domain",
                "material_class": "cmas_slag",
                "temperature_K": 500.0,
                "parent_oxide": "SiO2",
                "species": "SiO",
                "observable": "activity",
                "measured": 0.2,
                "units": "dimensionless",
                "score": True,
                "convention": "out-of-domain temperature probe",
            },
            {
                "id": "refused_held",
                "population": "inline_refused",
                "composition_id": "cmas_in_domain",
                "material_class": "cmas_slag",
                "temperature_K": 1800.0,
                "parent_oxide": "SiO2",
                "species": "SiO",
                "observable": "activity",
                "measured": 0.2,
                "units": "dimensionless",
                "score": False,
                "dropped_reason": "held from scoring for the unit test",
                "convention": "held point; must not enter RMSE",
            },
        ],
    )


def test_run_bench_ok_out_of_domain_and_refused(tmp_path: Path) -> None:
    fixture = _three_point_fixture(tmp_path / "bench.yaml")
    report = run_bench(fixture, DATAPACK_PATH)

    by_id = {row.point_id: row for row in report.points}
    assert set(by_id) == {"ok_sio2_1800", "ood_t500", "refused_held"}

    ok = by_id["ok_sio2_1800"]
    assert ok.status == "ok"
    assert ok.predicted is not None and ok.predicted > 0.0
    assert ok.residual == math.log10(ok.predicted / ok.measured)
    assert ok.ratio == ok.predicted / ok.measured

    ood = by_id["ood_t500"]
    assert ood.status == "not_converged"
    assert ood.predicted is None
    assert ood.residual is None
    assert ood.domain_flag is not None
    assert "temperature:" in ood.domain_flag
    assert "did not converge" in ood.reason.lower()

    refused = by_id["refused_held"]
    assert refused.status == "refused"
    assert refused.residual is None
    assert "held from scoring" in refused.reason


def test_refused_point_is_counted_and_excluded_from_rmse(tmp_path: Path) -> None:
    fixture = _three_point_fixture(tmp_path / "bench.yaml")
    report = run_bench(fixture, DATAPACK_PATH)

    assert report.n == 3
    assert report.n_ok == 1
    assert report.status_counts["ok"] == 1
    assert report.status_counts["out_of_domain"] == 0
    assert report.status_counts["refused"] == 1
    assert report.status_counts["not_converged"] == 1
    assert report.status_counts["unsupported_observable"] == 0

    ok = next(row for row in report.points if row.status == "ok")
    assert ok.residual is not None
    assert report.rmse == math.sqrt(ok.residual * ok.residual)
    assert report.median_abs_residual == abs(ok.residual)

    refused = next(row for row in report.points if row.status == "refused")
    assert refused.residual is None
    # Flattening RMSE by dropping the refusal would require it not to appear
    # in N / status_counts; both stay visible.
    assert report.n == report.n_ok + report.status_counts["refused"] + report.status_counts[
        "not_converged"
    ]


def test_unsupported_observable_is_not_coerced(tmp_path: Path) -> None:
    point = {
        "population": "inline_unsupported",
        "material_class": "cmas_slag",
        "temperature_K": 1800.0,
        "parent_oxide": "SiO2",
        "species": "SiO",
        "observable": "evaporation_flux",
        "measured": 1.0e-6,
        "units": "mol_m-2_s-1",
        "score": True,
        "convention": "must not be treated as activity or pressure",
    }
    fixture = _write_fixture(
        tmp_path / "bench.yaml",
        [
            {
                **point,
                "id": "flux_not_coerced",
                "composition_id": "cmas_in_domain",
            },
            {
                **point,
                "id": "flux_not_coerced_after_melt_refusal",
                "composition_id": "ferric_melt_refusal",
            },
        ],
        compositions={
            "cmas_in_domain": {
                "material_class": "cmas_slag",
                "composition_wt_pct": dict(_CMAS_WT),
            },
            "ferric_melt_refusal": {
                "material_class": "cmas_slag",
                "composition_wt_pct": {**_CMAS_WT, "Fe2O3": 1.0},
            },
        },
    )
    report = run_bench(fixture, DATAPACK_PATH)
    assert report.n == 2
    assert report.n_ok == 0
    assert [point.status for point in report.points] == [
        "unsupported_observable",
        "unsupported_observable",
    ]
    assert all(point.predicted is None for point in report.points)
    assert all(point.residual is None for point in report.points)
    assert all(
        point.reason == "unsupported observable 'evaporation_flux'"
        for point in report.points
    )
    refusal_notices = report.points[1].notices
    assert len(refusal_notices) == 1
    assert refusal_notices[0].startswith("melt evaluation out_of_domain:")
    assert refusal_notices[0] in report.notices
    assert report.rmse is None
    assert report.status_counts["unsupported_observable"] == 2


def test_research_overlay_controlled_vocabulary_census() -> None:
    assert len(_PACK_OVERLAY_BY_VERSION) == 32
    packs = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((_REPO_ROOT / "src/openimcc/data/packs").glob("*.json"))
        if path.name != "MANIFEST.json"
    ]
    assert len(packs) == len(_PACK_OVERLAY_BY_VERSION)
    assert {
        str(pack.get("imcc_sf04_datapack_version") or pack.get("version"))
        for pack in packs
    } == set(_PACK_OVERLAY_BY_VERSION)
    for pack in packs:
        model = pack.get("model_id") or pack.get("model")
        version = str(
            pack.get("imcc_sf04_datapack_version") or pack.get("version")
        )
        expected_overlay = _PACK_OVERLAY_BY_VERSION[version]
        if expected_overlay:
            assert str(model).casefold() == _PACK_OVERLAY_MODEL_BY_VERSION[
                version
            ].casefold()
        assert _is_research_overlay(pack) is expected_overlay
    assert not _is_research_overlay({"model_id": "IMCC-SF04-CONTEXT"})
    assert not _is_research_overlay(
        {
            "model_id": "IMCC-SF04-CONTEXT",
            "imcc_sf04_datapack_version": "1.0.2-ext-sp-2",
        }
    )
    assert not _is_research_overlay(
        {
            "model_id": "MODEL-EXT",
            "imcc_sf04_datapack_version": "unlisted-ext-version",
        }
    )


def test_load_pack_does_not_fallback_for_context_model_name(tmp_path: Path) -> None:
    raw = json.loads(
        (_REPO_ROOT / "src/openimcc/data/packs/imcc-sf04-ext-v4.json").read_text(
            encoding="utf-8"
        )
    )
    raw["model_id"] = "IMCC-SF04-CONTEXT"
    raw["imcc_sf04_datapack_version"] = "IMCC-SF04-CONTEXT-v1"
    path = tmp_path / "context-pack.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ImccMalformedDatapackError):
        load_pack(path)


def test_standard_state_controlled_vocabulary_census() -> None:
    assert len(_STANDARD_STATE_BY_CONVENTION) == 10
    benchmark = load_bench_set(
        _REPO_ROOT / "benchmarks/sets/basalt-bench-set-v1.yaml"
    )
    actual_pairs = {
        (
            _controlled_text(point.get("convention")),
            str(point.get("observable", "")).strip().casefold(),
        )
        for point in benchmark["points"]
    }
    assert actual_pairs == set(_STANDARD_STATE_BY_CONVENTION)
    actual = [
        _standard_state_for_point(point)
        for point in benchmark["points"]
    ]
    assert len(actual) == 402
    assert Counter(actual) == {
        "pure-solid": 292,
        "pure-liquid": 97,
        "pure-liquid parent": 6,
        "CMAS basis": 3,
        "unspecified": 4,
    }


@pytest.mark.parametrize(
    "convention",
    [
        "formerly pure-solid; now pure-liquid",
        "pure-solid/pure-liquid",
        "pure-solid converted from pure-liquid",
        "pure-solid without conversion",
        "not pure-solid; pure-liquid standard",
    ],
)
def test_unlisted_standard_state_text_is_unspecified(convention: str) -> None:
    assert _standard_state_for_point(
        {"observable": "activity", "convention": convention}
    ) == "unspecified"


@pytest.mark.parametrize(
    ("convention", "expected"),
    [("pure-solid", "pure-solid"), ("pure_liquid", "pure-liquid"), ("PURE-LIQUID", "pure-liquid")],
)
def test_caller_controlled_conventions_classify(convention: str, expected: str) -> None:
    assert set(_CALLER_STANDARD_STATE_CONVENTIONS).isdisjoint(_STANDARD_STATE_BY_CONVENTION)
    assert _standard_state_for_point(
        {"observable": "activity", "convention": convention}
    ) == expected


def test_bench_reports_the_engine_verbatim(tmp_path: Path) -> None:
    """The runner must not transform what the engine returned.

    Upstream this compared against the simulator's multi-engine harness, which
    does not travel. Checking against a direct ``evaluate`` call is a better
    test anyway: it pins the property that actually matters -- the bench is a
    reporting layer, not a second model -- without a second implementation to
    drift against.
    """
    fixture = _three_point_fixture(tmp_path / "bench.yaml")
    report = run_bench(fixture, DATAPACK_PATH)
    row = next(r for r in report.points if r.point_id == "ok_sio2_1800")

    pack = load_datapack(DATAPACK_PATH)
    direct = evaluate(dict(_CMAS_WT), 1800.0, pack, basis_type="wt")
    index = [str(x) for x in direct.parent_oxides].index("SiO2")
    expected = float(direct.parent_activity[index])

    assert row.predicted == expected
    assert row.residual == math.log10(expected / 0.2)


def _partial_pressure_point(
    point_id: str = "pp_k_1800",
    *,
    species: str = "K",
    parent_oxide: str = "K2O",
    temperature_K: float = 1800.0,
    measured: float = 1.0,
    units: str = "Pa",
    fO2_bar: float = 1.0e-10,
) -> dict:
    return {
        "id": point_id,
        "population": "inline_gas",
        "composition_id": "k_cmas_in_domain",
        "material_class": "cmas_slag",
        "temperature_K": temperature_K,
        "parent_oxide": parent_oxide,
        "species": species,
        "observable": "partial_pressure",
        "measured": measured,
        "units": units,
        "score": True,
        "fO2_bar": fO2_bar,
        "convention": f"p({species}) against an independent fO2 pin",
    }


def _partial_pressure_fixture(path: Path, points: list[dict]) -> Path:
    return _write_fixture(
        path,
        points,
        compositions={
            "k_cmas_in_domain": {
                "material_class": "cmas_slag",
                "composition_wt_pct": dict(_K_CMAS_WT),
            }
        },
    )


def test_partial_pressure_uses_parent_basis_and_converts_bar_to_pa(
    tmp_path: Path,
    packaged_gas_pack,
) -> None:
    fixture = _partial_pressure_fixture(
        tmp_path / "bench.yaml",
        [_partial_pressure_point()],
    )
    report = run_bench(fixture, DATAPACK_PATH)
    row = report.points[0]

    assert row.status == "ok", row.reason
    assert row.predicted is not None
    assert row.domain_flag is not None
    assert "liquid_from_solid construction from K-012" in row.domain_flag
    assert row.provenance_class == "estimated_systematic"

    pack = load_datapack(DATAPACK_PATH)
    imcc = evaluate(dict(_K_CMAS_WT), 1800.0, pack, basis_type="wt")
    activities = dict(zip(imcc.parent_oxides, imcc.parent_activity))
    from openimcc.gas import evaluate_gas

    gas_bar = evaluate_gas(
        activities,
        1800.0,
        1.0e-10,
        packaged_gas_pack,
        gas_species=("K",),
    )["K"]
    # The literal is deliberate: this test turns red if the named conversion
    # disappears or is changed to 1e4/1e6.
    assert row.predicted == pytest.approx(gas_bar * 100000.0)


def test_partial_pressure_units_are_a_typed_refusal(tmp_path: Path) -> None:
    fixture = _partial_pressure_fixture(
        tmp_path / "bench.yaml",
        [_partial_pressure_point(units="bar")],
    )
    row = run_bench(fixture, DATAPACK_PATH).points[0]

    assert row.status == "refused"
    assert row.predicted is None
    assert row.residual is None
    assert "expected units 'Pa'" in row.reason
    assert row.provenance_class == "estimated_systematic"


@pytest.mark.parametrize("fO2_mode", ["missing", "null"])
def test_partial_pressure_missing_or_null_fO2_is_a_typed_refusal(
    tmp_path: Path, fO2_mode: str
) -> None:
    point = _partial_pressure_point()
    if fO2_mode == "missing":
        point.pop("fO2_bar")
    else:
        point["fO2_bar"] = None

    row = run_bench(
        _partial_pressure_fixture(tmp_path / "bench.yaml", [point]),
        DATAPACK_PATH,
    ).points[0]

    assert row.status == "refused"
    assert row.predicted is None
    assert row.reason == (
        "gas comparison refused: observation has no independent fO2 pin"
    )
    assert row.provenance_class == "estimated_systematic"


def test_partial_pressure_loads_gas_datapack_once_per_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import openimcc.gas as gas

    calls = 0
    original = gas.load_gas_datapack

    def counted_load():
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(gas, "load_gas_datapack", counted_load)
    fixture = _partial_pressure_fixture(
        tmp_path / "bench.yaml",
        [
            _partial_pressure_point("pp_k_1800_a"),
            _partial_pressure_point("pp_k_1800_b"),
        ],
    )
    report = run_bench(fixture, DATAPACK_PATH)

    assert calls == 1
    assert all(row.status == "ok" for row in report.points)


def test_partial_pressure_refuses_when_pandas_is_unavailable(
    tmp_path: Path,
) -> None:
    fixture = _partial_pressure_fixture(
        tmp_path / "bench.yaml",
        [_partial_pressure_point()],
    )
    repo_root = Path(__file__).resolve().parents[1]
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                [
                    "import sys",
                    'sys.modules["pandas"] = None',
                    "from openimcc.bench import run_bench",
                    "row = run_bench(sys.argv[1], sys.argv[2]).points[0]",
                    'assert row.status == "refused"',
                    "assert row.predicted is None",
                    'assert "pandas" in row.reason',
                    'assert "openimcc[gas]" in row.reason',
                    'assert row.provenance_class == "estimated_systematic"',
                ]
            ),
            str(fixture),
            str(DATAPACK_PATH),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": str(repo_root / "src")},
    )
    assert probe.returncode == 0, probe.stderr


def test_filters_and_limit(tmp_path: Path) -> None:
    fixture = _three_point_fixture(tmp_path / "bench.yaml")
    only_ok = run_bench(fixture, DATAPACK_PATH, populations=["inline_ok"])
    assert only_ok.n == 1
    assert only_ok.points[0].point_id == "ok_sio2_1800"

    limited = run_bench(fixture, DATAPACK_PATH, limit=2)
    assert limited.n == 2
    assert [row.point_id for row in limited.points] == ["ok_sio2_1800", "ood_t500"]

    by_species = run_bench(fixture, DATAPACK_PATH, species=["SiO"])
    assert by_species.n == 3


def test_text_and_json_render(tmp_path: Path) -> None:
    fixture = _three_point_fixture(tmp_path / "bench.yaml")
    report = run_bench(fixture, DATAPACK_PATH)
    text = render_report(report)
    assert "N=3" in text
    assert "N_ok=1" in text
    assert "refused=1" in text
    assert "out_of_domain=0" in text
    assert "arithmetic total across incompatible slices" in text
    assert "ok_sio2_1800" in text

    stdout = io.StringIO()
    rc = main(
        [str(fixture), str(DATAPACK_PATH), "--json"],
        out=stdout,
    )
    assert rc == 0
    payload = json.loads(stdout.getvalue())
    assert payload["n"] == 3
    assert payload["n_ok"] == 1
    assert payload["status_counts"]["refused"] == 1
    ids = {row["point_id"] for row in payload["points"]}
    assert ids == {"ok_sio2_1800", "ood_t500", "refused_held"}
    # JSON must expose only standard-state-separated aggregates. The old
    # per-species/per-population fields mixed incompatible slices silently.
    assert "per_species" not in payload
    assert "per_population" not in payload
    assert "rmse" not in payload
    assert "flagged_rmse" not in payload
    assert "median_abs_residual" not in payload
    assert all(" × " in row["key"] for row in payload["per_slice"])
    assert all(
        {"rmse", "flagged_rmse", "median_abs_residual"} <= row.keys()
        for row in payload["per_slice"]
    )


def test_out_of_domain_gas_is_predicted_flagged_and_excluded_from_headline_rmse(
    tmp_path: Path,
) -> None:
    """The bench opts into a flagged prediction for a gas-domain gap."""
    fixture = _write_fixture(
        tmp_path / "bench.yaml",
        [
            {
                "id": "act_sio2_1800",
                "population": "inline_ok",
                "composition_id": "cmas_in_domain",
                "material_class": "cmas_slag",
                "temperature_K": 1800.0,
                "parent_oxide": "SiO2",
                "species": "SiO",
                "observable": "activity",
                "measured": 0.2,
                "units": "dimensionless",
                "score": True,
                "convention": "a(SiO2), parent-oxide formula-unit basis",
            },
            _partial_pressure_point(
                "pp_sio_2000",
                species="SiO",
                parent_oxide="SiO2",
                temperature_K=2000.0,
            ),
            _partial_pressure_point(
                "pp_sio_1700",
                species="SiO",
                parent_oxide="SiO2",
                temperature_K=1700.0,
                measured=1.0e-3,
            ),
            _partial_pressure_point(
                "pp_mg_2000",
                species="Mg",
                parent_oxide="MgO",
                temperature_K=2000.0,
            ),
        ],
        compositions={
            "cmas_in_domain": {
                "material_class": "cmas_slag",
                "composition_wt_pct": dict(_CMAS_WT),
            },
            "k_cmas_in_domain": {
                "material_class": "cmas_slag",
                "composition_wt_pct": dict(_K_CMAS_WT),
            },
        },
    )
    report = run_bench(fixture, DATAPACK_PATH)
    by_id = {row.point_id: row for row in report.points}

    activity = by_id["act_sio2_1800"]
    assert activity.status == "ok", activity.reason
    assert activity.predicted is not None and math.isfinite(float(activity.predicted))

    # SiO2(l) at 2000 K lies in a corroborated-calculation window, so the
    # row scores with a notice instead of being flagged out of the headline.
    corroborated = by_id["pp_sio_2000"]
    assert corroborated.status == "ok", corroborated.reason
    assert corroborated.residual is not None
    assert corroborated.domain_flag is None

    # MgO(l) is extrapolated at every temperature, so this row stays flagged.
    outside_certified_window = by_id["pp_mg_2000"]
    assert outside_certified_window.status == "ok", outside_certified_window.reason
    assert outside_certified_window.predicted is not None
    assert "MgO(l)" in outside_certified_window.domain_flag

    certified = by_id["pp_sio_1700"]
    assert certified.status == "ok", certified.reason
    assert certified.predicted is not None
    assert certified.residual is not None
    assert certified.domain_flag is None
    assert certified.provenance_class == "janaf_fitted"

    headline_residuals = [activity.residual, corroborated.residual, certified.residual]
    expected_headline_rmse = math.sqrt(
        sum(float(value) ** 2 for value in headline_residuals) / len(headline_residuals)
    )
    assert report.n_ok == 4
    assert report.n_flagged == 1
    assert report.status_counts["refused"] == 0
    assert report.rmse == pytest.approx(expected_headline_rmse)
    assert report.flagged_rmse == pytest.approx(
        abs(float(outside_certified_window.residual))
    )

    rendered = render_report(report)
    assert "flagged=1" in rendered
    assert "flagged_RMSE=" in rendered
    assert "MgO(l)" in rendered


def test_full_bench_slices_keep_standard_states_and_binary_separate() -> None:
    fixture = load_bench_set(Path("benchmarks/sets/basalt-bench-set-v1.yaml"))
    report = run_bench(
        "benchmarks/sets/basalt-bench-set-v1.yaml",
        DATAPACK_PATH,
    )

    kume = [
        row for row in report.per_slice if row.key.startswith("kume2000_slag_si_alloy")
    ]
    assert len(kume) == 1
    assert kume[0].key.endswith("× activity × pure-solid")
    assert kume[0].n == 292
    assert kume[0].median_residual is not None
    assert kume[0].mean_residual is not None
    assert not any(
        row.key.startswith("kume2000_slag_si_alloy")
        and "pure-liquid" in row.key
        for row in report.per_slice
    )

    assert len(report.converted_points) == 292
    assert all(
        row.standard_state
        == "pure-liquid (converted from pure-solid; Gibbs conversion v2)"
        for row in report.converted_points
    )
    converted_by_oxide = {
        row.parent_oxide: row for row in report.converted_points
    }
    assert "closed O-035 SiO2 condensate pair" in converted_by_oxide[
        "SiO2"
    ].convention
    for oxide, crystal_table in (
        ("CaO", "Ca-027"),
        ("MgO", "Mg-008"),
        ("Al2O3", "Al-096"),
    ):
        assert (
            f"JANAF {crystal_table} apparent crystal Gibbs"
            in converted_by_oxide[oxide].convention
        )
    converted_counts = {
        row.key.split(" × ")[1]: row.n for row in report.converted_slices
    }
    assert converted_counts == {
        "Al2O3": 62,
        "CaO": 99,
        "MgO": 45,
        "SiO2": 86,
    }
    assert all(
        "pure-liquid (converted from pure-solid; Gibbs conversion v2)" in row.key
        for row in report.converted_slices
    )
    assert {row.key.split(" × ")[1] for row in report.solid_standard_slices} == {
        "Al2O3",
        "CaO",
        "MgO",
        "SiO2",
    }
    assert all("pure-solid" in row.key for row in report.solid_standard_slices)

    table_slices = {row.key: row for row in report.kume_table_slices}
    for table in ("Table 1", "Table 2", "Table 3", "Table 4"):
        assert any(table in key and "pure-solid" in key for key in table_slices)
        assert any(
            table in key
            and "pure-liquid (converted from pure-solid; Gibbs conversion v2)" in key
            for key in table_slices
        )

    binary_points = [
        row
        for row in report.points
        if row.population
        in {"tsaplin2000_kems_na2o_sio2", "yamaguchi1983_emf_na2o_sio2"}
    ]
    binary_fixture = [
        point
        for point in fixture["points"]
        if point["population"]
        in {"tsaplin2000_kems_na2o_sio2", "yamaguchi1983_emf_na2o_sio2"}
    ]
    assert len(binary_points) == 94
    assert all(point["score"] for point in binary_fixture)
    assert all(
        point["scoring_status"] == "SCORED-ELIGIBLE" for point in binary_fixture
    )
    assert all("HELD-NOT-SCORED" not in str(point) for point in binary_fixture)
    assert all(
        "no authoritative oxide-basis conversion" not in str(point)
        for point in binary_fixture
    )
    assert sum(row.species == "Na" for row in binary_points) == 54
    assert sum(row.species == "SiO" for row in binary_points) == 40
    assert all(row.score for row in binary_points)
    assert all(
        "pure-liquid-Na2O" in row.convention
        for row in binary_points
        if row.species == "Na"
    )
    assert all(row.standard_state == "pure-liquid" for row in binary_points)
    assert all(row.predicted is not None for row in binary_points)
    assert all(row.residual is not None for row in binary_points)
    assert all(row.domain_flag is not None for row in binary_points)
    assert sum(row.extrapolation_flag is not None for row in binary_points) == 12

    assert sum(row.n for row in report.binary_slices) == 94
    assert sum(row.n for row in report.binary_slices if "× Na2O ×" in row.key) == 54
    assert sum(row.n for row in report.binary_slices if "× SiO2 ×" in row.key) == 40
    assert sum(row.n for row in report.binary_slices if "X>0.5" in row.key) == 20
    assert sum(row.n for row in report.binary_slices if "X<=0.5" in row.key) == 74
    assert all(row.n_flagged == row.n for row in report.binary_slices)

    # W3's inclusive wt boundary keeps the ten nominal X=.5 rows inside the
    # envelope. Independent probes therefore yield 74 temperature-only rows
    # and 20 rows needing both overrides, rather than first-exception counts.
    assert report.out_of_domain_counts == {"temperature_only": 74, "both": 20}
    assert sum(report.out_of_domain_counts.values()) == 94

    both = [
        row
        for row in binary_points
        if row.domain_flag is not None
        and "temperature:" in row.domain_flag
        and "envelope:" in row.domain_flag
    ]
    temperature_only = [
        row
        for row in binary_points
        if row.domain_flag is not None
        and "temperature:" in row.domain_flag
        and "envelope:" not in row.domain_flag
    ]
    assert len(temperature_only) == 74
    assert len(both) == 20
    representative = next(
        row
        for row in both
        if row.point_id == "tsaplin2000_a_na2o_x0477_1373"
    )
    assert "temperature:" in representative.domain_flag
    assert "envelope:" in representative.domain_flag
    assert "1373.0 K" in representative.domain_flag
    assert "X_Me2O=" in representative.domain_flag

    affected = {
        row.point_id
        for row in report.points
        if "yamaguchi1983" in row.point_id and "x0500" in row.point_id
    }
    assert affected == {
        "yamaguchi1983_a_sio2_liquid_x0500_1373",
        "yamaguchi1983_a_sio2_liquid_x0500_1473",
        "yamaguchi1983_a_sio2_liquid_x0500_1573",
        "yamaguchi1983_a_sio2_liquid_x0500_1673",
        "yamaguchi1983_a_na2o_x0500_1173",
        "yamaguchi1983_a_na2o_x0500_1273",
        "yamaguchi1983_a_na2o_x0500_1373",
        "yamaguchi1983_a_na2o_x0500_1473",
        "yamaguchi1983_a_na2o_x0500_1573",
        "yamaguchi1983_a_na2o_x0500_1673",
    }

    rendered = render_report(report)
    assert "Per dataset × observable × standard state" in rendered
    assert "Sodium binary predict-and-flag" in rendered
    assert "Kume 2000 activity slices by source table and oxide" in rendered
    assert "Table 4 CaO and Al2O3 activities are derived via Rein–Chipman" in rendered
    assert "unconverted solid standard state; not an accuracy figure for liquid activities" in rendered
    assert "temperature-only=74" in rendered
    assert "envelope-only=0" in rendered
    assert "both=20" in rendered
    assert "median(r)" in rendered
    assert "mean(r)" in rendered


def test_solid_to_liquid_conversion_matches_independent_table_calculation() -> None:
    # These values are calculated directly from the source tables, without
    # calling the production conversion or Gibbs helpers. JANAF crystal
    # rows use G°app = ΔfH°(298) + [H(T)-H(298)] - T*S(T), with the same
    # 298 K element reference as the packaged condensate rows. The chemical-
    # potential balance gives log10(a_liq) = log10(a_solid) -
    # (G_liq - G_cr)/(R*T*ln(10)); checking the converted activity catches a
    # reversed sign as well as a wrong reference row.
    gas_constant = R_PUBLISHED_GAS_J_MOL_K
    temperature_K = 1873.0
    activity_solid = 0.37
    cases = (
        ("MgO", "Mg-008"),
        ("Al2O3", "Al-096"),
    )
    for oxide, crystal_table in cases:
        liquid_row = _independent_csv_row(
            "src/openimcc/data/gas/condensate.csv",
            f"{oxide}(l)",
            temperature_K,
        )
        liquid_gibbs = _independent_condensate_gibbs_j_mol(
            liquid_row, temperature_K
        )
        crystal_gibbs = _independent_janaf_apparent_gibbs_j_mol(
            crystal_table, temperature_K
        )
        expected = 10.0 ** (
            math.log10(activity_solid)
            - (liquid_gibbs - crystal_gibbs)
            / (gas_constant * temperature_K * math.log(10.0))
        )
        assert _convert_solid_activity_to_liquid(
            activity_solid, oxide, temperature_K
        ) == pytest.approx(expected, rel=1.0e-10)


def test_bench_conversion_identity_names_current_gibbs_sources(
    tmp_path: Path,
) -> None:
    temperature_K = 1600.0
    activity_solid = 0.37
    fixture = _write_fixture(
        tmp_path / "bench.yaml",
        [
            {
                "id": "mg_solid_1600",
                "population": "inline_mgo",
                "composition_id": "cmas_in_domain",
                "material_class": "cmas_slag",
                "temperature_K": temperature_K,
                "parent_oxide": "MgO",
                "species": "MgO",
                "observable": "activity",
                "measured": activity_solid,
                "units": "dimensionless",
                "score": True,
                "convention": "pure-solid",
            }
        ],
    )
    converted = run_bench(fixture, DATAPACK_PATH).converted_points[0]

    liquid_row = _independent_csv_row(
        "src/openimcc/data/gas/condensate.csv", "MgO(l)", temperature_K
    )
    liquid_gibbs = _independent_condensate_gibbs_j_mol(
        liquid_row, temperature_K
    )
    crystal_gibbs = _independent_janaf_apparent_gibbs_j_mol(
        "Mg-008", temperature_K
    )
    expected = 10.0 ** (
        math.log10(activity_solid)
        - (liquid_gibbs - crystal_gibbs)
        / (
            R_PUBLISHED_GAS_J_MOL_K
            * temperature_K
            * math.log(10.0)
        )
    )

    assert converted.measured == pytest.approx(expected, rel=1.0e-10)
    assert converted.standard_state == (
        "pure-liquid (converted from pure-solid; Gibbs conversion v2)"
    )
    assert "Gibbs conversion v2" in converted.convention
    assert "liquid Gibbs from packaged condensate rows" in converted.convention
    assert "JANAF Mg-008 apparent crystal Gibbs" in converted.convention


def test_solid_reference_pairs_close_at_each_melting_point() -> None:
    solid_pairs = (
        ("CaO", 3200.0, "Ca-027"),
        ("MgO", 3105.0, "Mg-008"),
        ("Al2O3", 2327.0, "Al-096"),
    )
    for oxide, melting_point_K, crystal_table in solid_pairs:
        # CaO and MgO end their packaged liquid intervals at 3000 K. The last
        # fitted rows are continued here only to measure closure at the source
        # melting markers; the conversion itself still refuses out-of-range T.
        liquid_row = _independent_csv_row(
            "src/openimcc/data/gas/condensate.csv",
            f"{oxide}(l)",
            melting_point_K,
            extrapolate_last_row=True,
        )
        liquid_gibbs = _independent_condensate_gibbs_j_mol(
            liquid_row, melting_point_K
        )
        crystal_gibbs = _independent_janaf_apparent_gibbs_j_mol(
            crystal_table, melting_point_K
        )
        assert liquid_gibbs - crystal_gibbs == pytest.approx(0.0, abs=150.0)

    silica_liquid = _independent_csv_row(
        "src/openimcc/data/gas/condensate-sio2-janaf-cr-candidate.csv",
        "SiO2(l)",
        1996.0,
    )
    silica_crystal = _independent_csv_row(
        "src/openimcc/data/gas/condensate-sio2-janaf-cr-candidate.csv",
        "SiO2(cr)",
        1996.0,
    )
    assert _independent_condensate_gibbs_j_mol(
        silica_liquid, 1996.0
    ) == pytest.approx(
        _independent_condensate_gibbs_j_mol(silica_crystal, 1996.0),
        abs=1.0e-6,
    )
    assert _solid_to_liquid_log10_shift("SiO2", 1996.0) == pytest.approx(
        0.0, abs=1.0e-10
    )


def test_solid_to_liquid_conversion_refuses_out_of_range_temperature() -> None:
    with pytest.raises(ValueError, match="no packaged condensate reference row"):
        _solid_to_liquid_log10_shift("CaO", 3001.0)


def test_load_bench_set_preserves_yaml_12_chemistry_identifiers(tmp_path: Path) -> None:
    # YAML 1.1 would read bare `no`/`NO` as False; chemistry ids must stay strings.
    path = tmp_path / "bench.yaml"
    path.write_text(
        """\
schema_version: melt-activity-bench.v1
compositions:
  no:
    composition_wt_pct:
      SiO2: 100.0
points:
  - id: no
    composition_id: no
    species: NO
    score: true
""",
        encoding="utf-8",
    )

    fixture = load_bench_set(path)
    point = fixture["points"][0]

    assert "no" in fixture["compositions"]
    assert point["id"] == "no"
    assert point["composition_id"] == "no"
    assert point["species"] == "NO"
    assert point["score"] is True


def test_load_bench_set_uses_yaml_12_integer_grammar(tmp_path: Path) -> None:
    path = tmp_path / "bench.yaml"
    path.write_text(
        """\
schema_version: melt-activity-bench.v1
compositions:
  c:
    composition_wt_pct:
      SiO2: 012
      MgO: 0x10
      CaO: 1:30
points:
  - id: yaml12
    composition_id: c
    species: SiO
""",
        encoding="utf-8",
    )

    composition = load_bench_set(path)["compositions"]["c"]["composition_wt_pct"]

    assert composition == {"SiO2": "012", "MgO": "0x10", "CaO": "1:30"}


def test_encoded_composition_error_omits_internal_task_token(tmp_path: Path) -> None:
    point = {"id": "sample_x0500"}

    with pytest.raises(ValueError) as exc_info:
        load_bench_set(_write_fixture(tmp_path / "bench.yaml", [point]))

    token_pattern = r"\bt-" + r"\d{3,4}\b"
    assert re.search(token_pattern, str(exc_info.value)) is None


def test_binary_subslice_does_not_classify_nonfinite_mole_fractions() -> None:
    point = {
        "population": "yamaguchi1983_emf_na2o_sio2",
        "published_mole_fraction": {"Na2O": float("nan")},
    }

    assert _binary_subslice(point) == "X unknown"


def test_bench_rejects_unknown_point_configuration_key(tmp_path: Path) -> None:
    point = {
        "id": "typo_exclusion",
        "population": "inline_ok",
        "composition_id": "cmas_in_domain",
        "material_class": "cmas_slag",
        "temperature_K": 1800.0,
        "parent_oxide": "SiO2",
        "species": "SiO",
        "observable": "activity",
        "measured": 0.5,
        "units": "dimensionless",
        "scored": False,
        "dropped_reason": "intended exclusion",
    }

    with pytest.raises(ValueError, match="unknown point keys.*scored"):
        load_bench_set(_write_fixture(tmp_path / "bench.yaml", [point]))

    # A key the grader never reads must not be accepted either: an
    # "enabled: false" point would otherwise still be graded into RMSE.
    unread = {key: value for key, value in point.items() if key != "scored"}
    unread["enabled"] = False
    with pytest.raises(ValueError, match="unknown point keys.*enabled"):
        load_bench_set(_write_fixture(tmp_path / "bench2.yaml", [unread]))


def test_bench_rejects_quoted_boolean_score(tmp_path: Path) -> None:
    point = {
        "id": "quoted_score",
        "population": "inline_score",
        "composition_id": "cmas_in_domain",
        "material_class": "cmas_slag",
        "temperature_K": 1800.0,
        "parent_oxide": "SiO2",
        "species": "SiO",
        "observable": "activity",
        "measured": 0.5,
        "units": "dimensionless",
        "score": "false",
    }

    with pytest.raises(ValueError, match="score must be a boolean"):
        load_bench_set(_write_fixture(tmp_path / "bench.yaml", [point]))


@pytest.mark.parametrize(
    "score_fields",
    [{"score": True}, {}],
    ids=("true-score", "absent-score"),
)
def test_dropped_reason_always_excludes_bench_point(
    tmp_path: Path,
    score_fields: dict,
) -> None:
    point = {
        "id": "excluded_point",
        "population": "inline_excluded",
        "composition_id": "cmas_in_domain",
        "material_class": "cmas_slag",
        "temperature_K": 1800.0,
        "parent_oxide": "SiO2",
        "species": "SiO",
        "observable": "activity",
        "measured": 0.5,
        "units": "dimensionless",
        "dropped_reason": "held from scoring",
        **score_fields,
    }

    report = run_bench(
        _write_fixture(tmp_path / "bench.yaml", [point]), DATAPACK_PATH
    )
    row = report.points[0]

    assert row.score is False
    assert row.status == "refused"
    assert row.residual is None
    assert row.reason == "held from scoring"


def test_run_bench_cache_distinguishes_inline_compositions(tmp_path: Path) -> None:
    first = {**_CMAS_WT}
    second = {"SiO2": 75.0, "CaO": 5.0, "MgO": 10.0, "Al2O3": 10.0}
    points = [
        {
            "id": f"inline_{index}",
            "population": "inline_cache",
            "composition_id": "same_id",
            "composition_wt_pct": composition,
            "material_class": "cmas_slag",
            "temperature_K": 1800.0,
            "parent_oxide": "SiO2",
            "species": "SiO",
            "observable": "activity",
            "measured": 0.5,
            "units": "dimensionless",
            "score": True,
        }
        for index, composition in enumerate((first, second))
    ]
    report = run_bench(_write_fixture(tmp_path / "bench.yaml", points), DATAPACK_PATH)

    expected = [
        float(
            evaluate(
                composition,
                1800.0,
                load_datapack(DATAPACK_PATH),
                basis_type="wt",
            ).parent_activity[0]
        )
        for composition in (first, second)
    ]
    actual = [row.predicted for row in report.points]

    assert actual == pytest.approx(expected)
    assert actual[0] != pytest.approx(actual[1])


def test_load_pack_resolves_bare_packaged_name(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    packaged = load_pack(Path("imcc-sf04-v1.0.2.json"))
    direct = load_pack(Path(__file__).resolve().parents[1] / DATAPACK_PATH)
    assert (packaged.model_id, packaged.version) == (direct.model_id, direct.version)

    # An unknown bare name is not redirected; it fails as a missing file.
    with pytest.raises(FileNotFoundError):
        load_pack(Path("not-a-pack.json"))


def test_load_pack_does_not_fallback_when_local_path_cannot_be_statted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    original_lstat = Path.lstat

    def unstatable(path: Path):
        if path == Path("imcc-sf04-v1.0.2.json"):
            raise PermissionError("cannot inspect local pack")
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", unstatable)

    with pytest.raises(PermissionError, match="cannot inspect local pack"):
        load_pack(Path("imcc-sf04-v1.0.2.json"))


def test_load_pack_prefers_a_local_file_over_the_packaged_name(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "imcc-sf04-v1.0.2.json").write_text("[]", encoding="utf-8")
    # The local file wins, so its malformed content is what gets loaded.
    with pytest.raises(ImccMalformedDatapackError):
        load_pack(Path("imcc-sf04-v1.0.2.json"))


def test_load_pack_rejects_duplicate_keys_in_research_overlay(tmp_path: Path) -> None:
    path = tmp_path / "overlay.json"
    path.write_text(
        '{"model_id":"IMCC-SF04-EXT",'
        '"imcc_sf04_datapack_version":"IMCC-SF04-EXT-v1",'
        '"parents":["SiO2","Na2O"],'
        '"rows":[{"complex":"Na2SiO3",'
        '"nu":{"SiO2":1,"Na2O":1},"A":0,"A":1,"B":0,'
        '"T_domain_K":[1000,3000]}]}',
        encoding="utf-8",
    )

    with pytest.raises(DuplicateJsonKeyError, match="duplicate JSON key"):
        load_pack(path)
