"""Independent published Schaefer & Fegley (2004) reference checks."""

from __future__ import annotations

import csv
import hashlib
import math
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

from tools import build_sf04_reference as sf04_builder
from openimcc import (
    IMCC_GAS_CHANNEL_SPECIES,
    evaluate,
    evaluate_gas,
    load_datapack,
    load_gas_datapack,
)


REFERENCE_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "references" / "schaefer-fegley-2004"
PACK_PATH = Path(__file__).resolve().parents[1] / "src" / "openimcc" / "data" / "packs" / "imcc-sf04-v1.0.2.json"
EXPECTED_DATA_FILES = {
    "README.md",
    "PROVENANCE.yaml",
    "compositions.csv",
    "table9_anchors.csv",
    "fig10_digitized.csv",
}
GRID_TEMPERATURES = {1750, 1875, 1900, 2000, 2125, 2250, 2375}
# Measured maxima from the current independent reference comparison, rounded
# upward to 0.001 dex.  Each species gets its own fixed gate: baseline max +
# this 0.3 dex margin.  Keeping the measured baseline in source prevents a
# regression from raising its own threshold.
SPECIES_MAX_ABS_RESIDUAL_DEX = {
    "Fe": 1.400,
    "FeO": 0.831,
    "K": 0.993,
    "KO": 0.262,
    "Mg": 0.959,
    "MgO": 1.242,
    "Na": 1.417,
    "Na2": 2.212,
    "NaO": 1.827,
    "O": 0.028,
    "O2": 0.000,
    "SiO": 1.076,
    "SiO2": 1.057,
}
SPECIES_RESIDUAL_MARGIN_DEX = 0.300


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(name: str) -> list[dict[str, str]]:
    with (REFERENCE_DIR / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _assert_row_provenance(rows: list[dict[str, str]], *, pressure: bool) -> None:
    for row in rows:
        assert row["source_locator"]
        if "method_class" in row:
            assert row["method_class"]
        if "flux_method_class" in row:
            assert row["flux_method_class"]
            assert row["pressure_method_class"]
            assert row["pressure_source_locator"]
        if pressure:
            assert row["pressure_method_class"]
            assert row["pressure_source_locator"]
        for key, value in row.items():
            if not value or key.endswith("method_class") or key.endswith("locator"):
                continue
            # Every data value is accompanied by a row-level locator and the
            # relevant method class; pressure values have their own locator.
            assert row["source_locator"], (key, row)


def test_sf04_reference_files_and_hashes_are_provenanced() -> None:
    manifest = yaml.safe_load((REFERENCE_DIR / "PROVENANCE.yaml").read_text(encoding="utf-8"))
    entries = {entry["path"]: entry for entry in manifest["files"]}
    assert set(entries) == EXPECTED_DATA_FILES
    assert {path.name for path in REFERENCE_DIR.iterdir() if path.is_file()} == EXPECTED_DATA_FILES
    build_script = REFERENCE_DIR.parents[2] / manifest["build_script"]["path"]
    assert _sha256(build_script) == manifest["build_script"]["sha256"]

    for name, entry in entries.items():
        path = REFERENCE_DIR / name
        assert path.is_file(), name
        if entry.get("sha256") is not None:
            assert _sha256(path) == entry["sha256"], name

    compositions = _rows("compositions.csv")
    anchors = _rows("table9_anchors.csv")
    figure = _rows("fig10_digitized.csv")
    _assert_row_provenance(compositions, pressure=False)
    _assert_row_provenance(anchors, pressure=True)
    _assert_row_provenance(figure, pressure=True)

    assert len(compositions) == 5
    assert {row["rock"] for row in compositions} == {"tho", "aba", "kom", "dun", "cai"}
    assert len(anchors) == 13
    assert {row["pressure_method_class"] for row in anchors} == {"derived_eq11"}
    assert len(figure) == 350
    assert {int(row["T_K"]) for row in figure} <= GRID_TEMPERATURES
    assert {row["method_class"] for row in figure} == {"digitized_figure"}
    assert {row["pressure_method_class"] for row in figure} == {"derived_table7"}


def _compositions() -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for row in _rows("compositions.csv"):
        # IMCC's adapter intentionally accepts FeO-equivalent input, not a
        # ferric parent.  This is the caller's redox conversion; evaluate's
        # wt-to-mol conversion remains the path under test.
        feo_equivalent = float(row["FeO_wt_pct"]) + float(row["Fe2O3_wt_pct"]) * (
            2.0 * 71.844 / 159.688
        )
        result[row["rock"]] = {
            "SiO2": float(row["SiO2_wt_pct"]),
            "MgO": float(row["MgO_wt_pct"]),
            "Al2O3": float(row["Al2O3_wt_pct"]),
            "TiO2": float(row["TiO2_wt_pct"]),
            "FeO": feo_equivalent,
            "CaO": float(row["CaO_wt_pct"]),
            "Na2O": float(row["Na2O_wt_pct"]),
            "K2O": float(row["K2O_wt_pct"]),
        }
    return result


def _activities(
    compositions: dict[str, dict[str, float]],
    imcc_pack: Any,
    rock: str,
    T_K: int,
) -> dict[str, float]:
    result = evaluate(
        compositions[rock],
        T_K,
        imcc_pack,
        basis_type="wt",
        allow_extrapolation=True,
        allow_out_of_envelope=True,
    )
    return dict(zip(result.parent_oxides, result.parent_activity))


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "n": float(len(values)),
        "min": ordered[0],
        "median": ordered[len(ordered) // 2],
        "p95": ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)],
        "max": ordered[-1],
        "absmax": max(abs(value) for value in values),
        "rmse": math.sqrt(sum(value * value for value in values) / len(values)),
    }


def test_sf04_digitization_self_checks() -> None:
    figure = _rows("fig10_digitized.csv")
    sums = {
        rock: sum(
            10.0 ** float(row["log10_x"])
            for row in figure
            if row["rock"] == rock and row["T_K"] == "1900"
        )
        for rock in ("tho", "aba", "kom", "dun", "cai")
    }
    # The measured sums are reported to two decimal places as the 0.80–0.81
    # band; retain the full CSV values for the anchor substitution below.
    assert all(0.80 <= round(total, 2) <= 0.81 for total in sums.values())

    anchors = {row["species"]: row for row in _rows("table9_anchors.csv")}
    tholeiite = {
        row["species"]: row
        for row in figure
        if row["rock"] == "tho" and row["T_K"] == "1900"
    }
    printed_total_bar = 7.72e-5
    deltas = {}
    for species, figure_row in tholeiite.items():
        if species not in anchors:
            continue
        anchor_log10_x = math.log10(float(anchors[species]["partial_pressure_bar"]) / printed_total_bar)
        delta = float(figure_row["log10_x"]) - anchor_log10_x
        deltas[species] = delta

    assert deltas["Na"] == pytest.approx(-0.094, abs=0.01)
    assert deltas["O2"] == pytest.approx(-0.087, abs=0.01)
    assert abs(deltas["Na"]) > float(tholeiite["Na"]["log10_x_uncertainty_dex"])
    assert abs(deltas["O2"]) > float(tholeiite["O2"]["log10_x_uncertainty_dex"])

    figure_sum = sums["tho"]
    substituted_sum = figure_sum
    for species in ("Na", "O2"):
        substituted_sum -= 10.0 ** float(tholeiite[species]["log10_x"])
        substituted_sum += float(anchors[species]["partial_pressure_bar"]) / printed_total_bar
    assert substituted_sum == pytest.approx(1.0, abs=0.01)

    print(f"SF04 Fig. 10 sums at 1900 K: {sums}")
    print(f"SF04 tholeiite/anchor deltas at 1900 K: {deltas}")


@pytest.mark.skipif(
    sf04_builder.PAPER_DEFAULT is None or not sf04_builder.PAPER_DEFAULT.is_file(),
    reason="SF04 source PDF is not available in this checkout",
)
def test_sf04_shifted_seeds_reject_non_curve_ink(tmp_path: Path) -> None:
    pixels = sf04_builder._render_page(sf04_builder.PAPER_DEFAULT, tmp_path)
    original_traces = deepcopy(sf04_builder.FIGURE_LOG10_TRACES)
    baseline = sf04_builder._trace_pixel_rows(pixels)
    cases = (
        ("tho", "Na", 1900),
        ("tho", "O2", 1900),
        ("tho", "O", 1900),
    )
    results = []
    try:
        for rock, species, T_K in cases:
            index = sf04_builder.GRID_TEMPERATURES.index(T_K)
            y0, y1 = sf04_builder.PANEL_CALIBRATION[rock][2:]
            baseline_log10_x = -5.0 * (
                baseline[(rock, species, T_K)] - y0
            ) / (y1 - y0)
            for shift in (-0.2, 0.2):
                sf04_builder.FIGURE_LOG10_TRACES = deepcopy(original_traces)
                values = list(sf04_builder.FIGURE_LOG10_TRACES[rock][species])
                values[index] += shift
                sf04_builder.FIGURE_LOG10_TRACES[rock][species] = tuple(values)
                try:
                    traced = sf04_builder._trace_pixel_rows(pixels)
                except ValueError as exc:
                    assert f"{rock}/{species}/{T_K}" in str(exc)
                    results.append(f"{rock}/{species}/{T_K} {shift:+.1f}: rejected")
                else:
                    measured_log10_x = -5.0 * (
                        traced[(rock, species, T_K)] - y0
                    ) / (y1 - y0)
                    assert measured_log10_x == pytest.approx(
                        baseline_log10_x, abs=0.005
                    )
                    results.append(
                        f"{rock}/{species}/{T_K} {shift:+.1f}: same stroke"
                    )
    finally:
        sf04_builder.FIGURE_LOG10_TRACES = original_traces
    print("SF04 shifted-seed checks: " + "; ".join(results))


def test_sf04_engine_residual_report_and_gate() -> None:
    compositions = _compositions()
    imcc_pack = load_datapack(PACK_PATH)
    gas_pack = load_gas_datapack()
    figure = _rows("fig10_digitized.csv")
    anchors = _rows("table9_anchors.csv")
    residuals: list[tuple[str, str, str, float]] = []
    not_comparable: list[tuple[str, str, str]] = []
    grouped: defaultdict[tuple[str, str], list[float]] = defaultdict(list)

    figure_groups: defaultdict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in figure:
        figure_groups[(row["rock"], int(row["T_K"]))].append(row)

    for (rock, T_K), rows in sorted(figure_groups.items()):
        o2_row = next(row for row in rows if row["species"] == "O2")
        fO2 = 10.0 ** float(o2_row["log10_p_bar"])
        activities = _activities(compositions, imcc_pack, rock, T_K)
        for row in rows:
            species = row["species"]
            method_class = row["pressure_method_class"]
            if species not in IMCC_GAS_CHANNEL_SPECIES:
                not_comparable.append((rock, species, method_class))
                continue
            predicted = evaluate_gas(
                activities,
                T_K,
                fO2,
                gas_pack,
                allow_extrapolation=True,
                gas_species=(species,),
            )[species]
            residual = math.log10(predicted) - float(row["log10_p_bar"])
            residuals.append((rock, species, method_class, residual))
            grouped[("rock", rock)].append(residual)
            grouped[("species", species)].append(residual)
            grouped[("method", method_class)].append(residual)

    anchor_o2 = float(next(row["partial_pressure_bar"] for row in anchors if row["species"] == "O2"))
    anchor_activities = _activities(compositions, imcc_pack, "tho", 1900)
    for row in anchors:
        species = row["species"]
        method_class = row["pressure_method_class"]
        if species not in IMCC_GAS_CHANNEL_SPECIES:
            not_comparable.append(("tho", species, method_class))
            continue
        predicted = evaluate_gas(
            anchor_activities,
            1900,
            anchor_o2,
            gas_pack,
            allow_extrapolation=True,
            gas_species=(species,),
        )[species]
        residual = math.log10(predicted) - math.log10(float(row["partial_pressure_bar"]))
        residuals.append(("tho", species, method_class, residual))
        grouped[("rock", "tho")].append(residual)
        grouped[("species", species)].append(residual)
        grouped[("method", method_class)].append(residual)

    values = [row[3] for row in residuals]
    report = {"all": _summary(values)}
    report.update({f"{kind}:{name}": _summary(vals) for (kind, name), vals in grouped.items()})
    measured_species = {
        key.removeprefix("species:"): value["absmax"]
        for key, value in report.items()
        if key.startswith("species:")
    }
    assert set(measured_species) == set(SPECIES_MAX_ABS_RESIDUAL_DEX)
    species_gates = {
        species: SPECIES_MAX_ABS_RESIDUAL_DEX[species] + SPECIES_RESIDUAL_MARGIN_DEX
        for species in sorted(SPECIES_MAX_ABS_RESIDUAL_DEX)
    }
    for species, measured_max in measured_species.items():
        assert measured_max <= species_gates[species]

    print(
        "SF04 residuals: "
        f"comparable={len(residuals)} not_comparable={len(not_comparable)} "
        f"all={report['all']} margin={SPECIES_RESIDUAL_MARGIN_DEX:.1f} dex"
    )
    print(f"SF04 per-species gates: {species_gates}")
    for key in sorted(report):
        if key != "all" and (
            key.startswith("rock:")
            or key.startswith("method:")
            or key.startswith("species:")
        ):
            print(f"SF04 residual group {key}: {report[key]}")
    not_comparable_counts = Counter(not_comparable)
    print(f"SF04 not comparable counts: {dict(sorted(not_comparable_counts.items()))}")
