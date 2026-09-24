"""IMCC-only empirical residual runner."""

from __future__ import annotations

import contextlib
import io
import json
import math
import sys
from pathlib import Path

import yaml

import pytest

from openimcc import evaluate, load_datapack
from openimcc.bench import (
    main,
    render_report,
    run_bench,
)

DATAPACK_PATH = Path("src/openimcc/data/packs/imcc-sf04-v1.0.2.json")

# In-domain CMAS slag, 8-parent IMCC basis (missing parents are zero).
_CMAS_WT = {
    "SiO2": 50.0,
    "CaO": 20.0,
    "MgO": 15.0,
    "Al2O3": 15.0,
}
_K_CMAS_WT = {**_CMAS_WT, "K2O": 1.0}


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
    assert ood.status == "out_of_domain"
    assert ood.predicted is None
    assert ood.residual is None
    assert "outside the declared domain" in ood.reason

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
    assert report.status_counts["out_of_domain"] == 1
    assert report.status_counts["refused"] == 1
    assert report.status_counts["not_converged"] == 0
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
        "out_of_domain"
    ]


def test_unsupported_observable_is_not_coerced(tmp_path: Path) -> None:
    fixture = _write_fixture(
        tmp_path / "bench.yaml",
        [
            {
                "id": "flux_not_coerced",
                "population": "inline_unsupported",
                "composition_id": "cmas_in_domain",
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
        ],
    )
    report = run_bench(fixture, DATAPACK_PATH)
    assert report.n == 1
    assert report.n_ok == 0
    assert report.points[0].status == "unsupported_observable"
    assert report.points[0].predicted is None
    assert report.points[0].residual is None
    assert report.rmse is None
    assert report.status_counts["unsupported_observable"] == 1


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
) -> None:
    fixture = _partial_pressure_fixture(
        tmp_path / "bench.yaml",
        [_partial_pressure_point()],
    )
    report = run_bench(fixture, DATAPACK_PATH)
    row = report.points[0]

    assert row.status == "ok", row.reason
    assert row.predicted is not None
    assert row.domain_flag is None
    assert row.provenance_class == "secondary_transcription_unverified_primary"

    pack = load_datapack(DATAPACK_PATH)
    imcc = evaluate(dict(_K_CMAS_WT), 1800.0, pack, basis_type="wt")
    activities = dict(zip(imcc.parent_oxides, imcc.parent_activity))
    from openimcc.gas import evaluate_gas, load_gas_datapack

    gas_bar = evaluate_gas(
        activities,
        1800.0,
        1.0e-10,
        load_gas_datapack(),
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
    assert row.provenance_class == "secondary_transcription_unverified_primary"


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
    assert row.provenance_class == "secondary_transcription_unverified_primary"


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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _partial_pressure_fixture(
        tmp_path / "bench.yaml",
        [_partial_pressure_point()],
    )
    monkeypatch.delitem(sys.modules, "openimcc.gas", raising=False)
    monkeypatch.setitem(sys.modules, "pandas", None)

    row = run_bench(fixture, DATAPACK_PATH).points[0]

    assert row.status == "refused"
    assert row.predicted is None
    assert "pandas" in row.reason
    assert "openimcc[gas]" in row.reason
    assert row.provenance_class == "secondary_transcription_unverified_primary"


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
    assert "out_of_domain=1" in text
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
                "pp_sio_1900",
                species="SiO",
                parent_oxide="SiO2",
                temperature_K=1900.0,
                measured=1.0e-3,
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

    in_domain = by_id["pp_sio_2000"]
    assert in_domain.status == "ok", in_domain.reason
    assert in_domain.predicted is not None
    assert in_domain.domain_flag is None
    assert in_domain.provenance_class == "lam1987_transcribed"

    flagged = by_id["pp_sio_1900"]
    assert flagged.status == "ok", flagged.reason
    assert flagged.predicted is not None
    assert flagged.residual is not None
    assert "outside declared G(T) interval" in flagged.domain_flag
    assert "SiO2(l) [1996, 3000] K" in flagged.domain_flag
    assert flagged.provenance_class == "lam1987_transcribed"

    headline_residuals = [activity.residual, in_domain.residual]
    expected_headline_rmse = math.sqrt(
        sum(float(value) ** 2 for value in headline_residuals) / 2.0
    )
    assert report.n_ok == 3
    assert report.n_flagged == 1
    assert report.status_counts["refused"] == 0
    assert report.rmse == pytest.approx(expected_headline_rmse)
    assert report.flagged_rmse == pytest.approx(abs(flagged.residual))

    rendered = render_report(report)
    assert "flagged=1" in rendered
    assert "flagged_RMSE=" in rendered
    assert "SiO2(l) [1996, 3000] K" in rendered
