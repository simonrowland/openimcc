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
    _single_cation_gas_activities,
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


def test_single_cation_conversion_is_the_nth_root_of_the_parent_activity() -> None:
    """a_parent = a_single**n for M_nO_m, so a_single = a_parent**(1/n).

    Derivation: the parent oxide M_nO_m is n formula units of the single-cation
    component MO_(m/n). In an ideal mixture of those units the parent activity
    is the product of n identical single-cation activities, so taking the nth
    root inverts it.

    Sanity: SiO2 has one cation, so the value is unchanged; Na2O and Al2O3 have
    two, so each is a square root. A wrong exponent here rescales every gas
    comparison and the residual table still looks plausible, which is why the
    expected values are written out longhand rather than recomputed from the
    same expression the code uses.
    """
    parent = {"SiO2": 0.25, "Al2O3": 0.04, "Na2O": 1.0e-8, "K2O": 4.0e-10, "MgO": 0.1}
    got = _single_cation_gas_activities(parent)

    assert got["SiO2"] == pytest.approx(0.25)          # 1 cation -> unchanged
    assert got["MgO"] == pytest.approx(0.1)            # 1 cation -> unchanged
    assert got["Al2O3"] == pytest.approx(0.2)          # 2 cations -> sqrt(0.04)
    assert got["Na2O"] == pytest.approx(1.0e-4)        # 2 cations -> sqrt(1e-8)
    assert got["K2O"] == pytest.approx(2.0e-5)         # 2 cations -> sqrt(4e-10)


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


def test_activity_scores_while_the_gas_observable_refuses(tmp_path: Path) -> None:
    """The package's current gas posture, pinned.

    Activities are this engine's own output and must score. partial_pressure
    borrowed the simulator's vapour stack, which did not travel, so it refuses
    -- with a reason a user can act on rather than a traceback. When the rewire
    onto openimcc.gas lands (see docs/EXTRACTION.md), this test is the one that
    should change, and it should change to an assertion about a NUMBER in Pa.
    """
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
            {
                "id": "pp_sio_1800",
                "population": "inline_gas",
                "composition_id": "cmas_in_domain",
                "material_class": "cmas_slag",
                "temperature_K": 1800.0,
                "parent_oxide": "SiO2",
                "species": "SiO",
                "observable": "partial_pressure",
                "measured": 1.0,
                "units": "Pa",
                "score": True,
                "fO2_bar": 1.0e-10,
                "convention": "p(SiO) against an independent fO2 pin",
            },
        ],
    )
    report = run_bench(fixture, DATAPACK_PATH)
    by_id = {row.point_id: row for row in report.points}

    activity = by_id["act_sio2_1800"]
    assert activity.status == "ok", activity.reason
    assert activity.predicted is not None and math.isfinite(float(activity.predicted))

    gas_row = by_id["pp_sio_1800"]
    assert gas_row.status == "refused"
    assert gas_row.predicted is None
    assert "not wired in openimcc yet" in gas_row.reason

    # The refusal stays out of the score, like every other refusal.
    assert report.n_ok == 1
    assert report.status_counts["refused"] == 1
