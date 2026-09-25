"""Research rails for species-family counterfactuals.

These tests deliberately label every altered in-memory datapack as research
only.  They are sensitivity measurements, not alternative published fits.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Iterable

import numpy as np
import pytest
import yaml

from openimcc import (
    ImccDatapack,
    evaluate,
    evaluate_gas,
    label_research_datapack,
    load_datapack,
    load_gas_datapack,
)


ROOT = Path(__file__).resolve().parents[1]
PACK_PATH = ROOT / "src" / "openimcc" / "data" / "packs" / "imcc-sf04-v1.0.2.json"
SF04_REFERENCE_DIR = ROOT / "benchmarks" / "references" / "schaefer-fegley-2004"
BENCH_PATH = ROOT / "benchmarks" / "sets" / "basalt-bench-set-v1.yaml"

PUBLISHED_PACK = load_datapack(PACK_PATH)
GAS_PACK = load_gas_datapack()

PARENT_MOLAR_MASS = {
    "SiO2": 60.083,
    "Na2O": 61.97853856,
    "K2O": 94.196,
}

# Source citation: Tsaplin, Zaitsev, Shelkova & Mogutnov (2000),
# "Thermodynamic properties and phase equilibria in Na2O-SiO2 and K2O-SiO2
# systems," Molten Slags, Fluxes and Salts 2000, paper 044, Table 1.  Values
# below are the unique rows transcribed in the research evidence, on the
# parent-oxide formula-unit basis.  This fixture is intentionally only the
# binary evidence needed for the two directional rails.
TSAPLIN_NA_ROWS = (
    (0.195, 1473.0, -9.14),
    (0.247, 1273.0, -10.28),
    (0.291, 1673.0, -7.49),
    (0.329, 1173.0, -10.33),
    (0.375, 1573.0, -7.12),
    (0.427, 1473.0, -6.99),
    (0.476, 1573.0, -5.82),
)
TSAPLIN_K_ROWS = (
    (0.152, 1573.0, -10.84),
    (0.189, 1173.0, -14.60),
    (0.230, 1073.0, -15.73),
    (0.278, 1323.0, -11.94),
    (0.278, 1673.0, -9.06),
    (0.326, 1373.0, -10.12),
    (0.370, 1323.0, -9.23),
    (0.370, 1523.0, -7.70),
    (0.409, 1323.0, -8.87),
    (0.457, 1473.0, -7.46),
)


def _research_pack(
    *,
    remove: Iterable[str] = (),
    updates: dict[str, tuple[float, float]] | None = None,
) -> ImccDatapack:
    """Return a labelled, non-published copy of the active kernel rows."""
    source = PUBLISHED_PACK.kernel_datapack
    removed = set(remove)
    keep = [index for index, name in enumerate(source.reactions) if name not in removed]
    updated = updates or {}
    A = source.A[keep].copy()
    B = source.B[keep].copy()
    for name, (new_A, new_B) in updated.items():
        local = [position for position, index in enumerate(keep) if source.reactions[index] == name]
        assert len(local) == 1, (name, local)
        A[local[0]] = new_A
        B[local[0]] = new_B

    raw = ImccDatapack(
        reactions=[source.reactions[index] for index in keep],
        nu=np.array(source.nu[:, keep], copy=True),
        A=A,
        B=B,
        domains=[source.domains[index] for index in keep],
        version="research-species-set-v1",
        parent_oxides=source.parent_oxides,
    )
    # The only path from a counterfactual raw pack to an evaluated pack is the
    # explicit non-published label.  It cannot claim the published model/hash.
    return label_research_datapack(
        raw,
        model_id="IMCC-SF04-RE",
        coverage="RE-species-set",
    )


def _activities(composition: dict[str, float], temperature: float, pack: ImccDatapack) -> dict[str, float]:
    result = evaluate(
        composition,
        temperature,
        pack,
        basis_type="wt",
        allow_extrapolation=True,
        allow_out_of_envelope=True,
    )
    return dict(zip(result.parent_oxides, result.parent_activity))


def _sf04_table9() -> tuple[dict[str, float], float, dict[str, float]]:
    """Read Table 5, Table 9, and the Eq. 11 O2 pin used by the reference test."""
    with (SF04_REFERENCE_DIR / "compositions.csv").open(newline="", encoding="utf-8") as handle:
        composition_row = next(row for row in csv.DictReader(handle) if row["rock"] == "tho")
    # Match tests/test_sf04_published_reference.py: ferric iron is folded into
    # FeO-equivalent input before the wt-to-mol conversion.
    composition = {
        name: float(composition_row[f"{name}_wt_pct"])
        for name in ("SiO2", "MgO", "Al2O3", "TiO2", "CaO", "Na2O", "K2O")
    }
    composition["FeO"] = float(composition_row["FeO_wt_pct"]) + float(
        composition_row["Fe2O3_wt_pct"]
    ) * (2.0 * 71.844 / 159.688)

    with (SF04_REFERENCE_DIR / "table9_anchors.csv").open(newline="", encoding="utf-8") as handle:
        anchors = {row["species"]: row for row in csv.DictReader(handle)}
    # Eq. 11's O2 cell is the independent oxygen-fugacity pin, not a model O2
    # prediction.  The gas layer returns bar, as do these anchor pressures.
    fO2 = float(anchors["O2"]["partial_pressure_bar"])
    pressures = {species: float(anchors[species]["partial_pressure_bar"]) for species in ("K", "Na")}
    return composition, fO2, pressures


def _gas_residual(
    composition: dict[str, float],
    temperature: float,
    fO2: float,
    measured_bar: float,
    species: str,
    pack: ImccDatapack,
) -> float:
    activities = _activities(composition, temperature, pack)
    predicted = evaluate_gas(
        activities,
        temperature,
        fO2,
        GAS_PACK,
        allow_extrapolation=True,
        gas_species=(species,),
    )[species]
    return math.log10(predicted) - math.log10(measured_bar)


def _hastie_case4() -> tuple[dict[str, float], dict[str, float]]:
    data = yaml.safe_load(BENCH_PATH.read_text(encoding="utf-8"))
    composition = data["compositions"]["hastie_case4_k_silicate"]["composition_wt_pct"]
    point = next(row for row in data["points"] if row["id"] == "hastie_k_1917_186")
    return composition, point


def _hastie_k_residual(pack: ImccDatapack) -> float:
    composition, point = _hastie_case4()
    activities = _activities(composition, point["temperature_K"], pack)
    predicted_bar = evaluate_gas(
        activities,
        point["temperature_K"],
        point["fO2_bar"],
        GAS_PACK,
        gas_species=("K",),
    )["K"]
    # The benchmark measurement is Pa; evaluate_gas returns bar.
    return math.log10(predicted_bar * 1.0e5) - math.log10(point["measured"])


def _tsaplin_residuals(rows: tuple[tuple[float, float, float], ...], oxide: str) -> list[float]:
    result = []
    for mole_fraction, temperature, measured_log10_activity in rows:
        composition = {
            "SiO2": (1.0 - mole_fraction) * PARENT_MOLAR_MASS["SiO2"],
            oxide: mole_fraction * PARENT_MOLAR_MASS[oxide],
        }
        activities = _activities(composition, temperature, PUBLISHED_PACK.kernel_datapack)
        result.append(math.log10(activities[oxide]) - measured_log10_activity)
    return result


def test_sf04_table9_species_set_sensitivities() -> None:
    composition, fO2, pressures = _sf04_table9()
    base = _research_pack()
    no_kca = _research_pack(remove=("KCaAlSi2O7",))
    no_na_half = _research_pack(
        remove=("NaAlSiO4", "NaAlSi3O8", "NaAlO2", "NaAlSi2O6")
    )
    # These are the three nu(Na2O)=1 sodium-family rows used by the evidence
    # comparison; their removal is the small-change control.
    no_na_one = _research_pack(remove=("Na2SiO3", "Na2Si2O5", "Na2TiO3"))

    base_k = _gas_residual(composition, 1900.0, fO2, pressures["K"], "K", base)
    no_kca_k = _gas_residual(composition, 1900.0, fO2, pressures["K"], "K", no_kca)
    base_na = _gas_residual(composition, 1900.0, fO2, pressures["Na"], "Na", base)
    no_na_half_na = _gas_residual(
        composition, 1900.0, fO2, pressures["Na"], "Na", no_na_half
    )
    no_na_one_na = _gas_residual(
        composition, 1900.0, fO2, pressures["Na"], "Na", no_na_one
    )

    # SF04 Table 9: measured log10(predicted/reference) is +0.1419 dex with
    # KCaAlSi2O7 and +2.3117 dex without it.  The latter is the species-set
    # sensitivity; it is not a reason to remove the complex.
    assert base_k == pytest.approx(0.14, abs=0.05)
    assert no_kca_k == pytest.approx(2.3, abs=0.1)
    assert base_k > 0.0 < no_kca_k

    # The four nu(Na2O)=0.5 aluminosilicates move Na from -1.4164 to -0.5285
    # dex.  Removing the three nu=1 rows changes it by only 0.0091 dex.
    assert base_na == pytest.approx(-1.42, abs=0.05)
    assert no_na_half_na == pytest.approx(-0.53, abs=0.05)
    assert abs(no_na_one_na - base_na) < 0.05
    print(
        "SF04 species sensitivity: "
        f"K {base_k:+.4f} -> {no_kca_k:+.4f}; "
        f"Na {base_na:+.4f} -> {no_na_half_na:+.4f}; "
        f"nu=1 Na delta {no_na_one_na - base_na:+.4f} dex"
    )


def test_hastie_k_family_and_fc87_supersedes_sensitivities() -> None:
    base = _research_pack()
    no_k_half = _research_pack(
        remove=("KAlSiO4", "KAlSi3O8", "KAlO2", "KAlSi2O6")
    )
    base_hastie = _hastie_k_residual(base)
    no_k_half_hastie = _hastie_k_residual(no_k_half)

    # Hastie case 4's first pinned K point is -0.8916 dex.  Removing the four
    # nu(K2O)=0.5 K-aluminosilicates reverses the error and overshoots to
    # +1.4482 dex; this is a different family from the binary K rows.
    assert base_hastie == pytest.approx(-0.89, abs=0.05)
    assert no_k_half_hastie == pytest.approx(1.4, abs=0.1)
    assert base_hastie < 0.0 < no_k_half_hastie

    pack_data = json.loads(PACK_PATH.read_text(encoding="utf-8"))
    superseded = {
        row["complex"]: (row["supersedes"]["A"], row["supersedes"]["B"])
        for row in pack_data["rows"]
        if row["complex"] in {"K2SiO3", "K2Si2O5"}
    }
    fc87 = _research_pack(updates=superseded)
    fc87_hastie = _hastie_k_residual(fc87)
    composition, fO2, pressures = _sf04_table9()
    base_table9 = _gas_residual(
        composition, 1900.0, fO2, pressures["K"], "K", base
    )
    fc87_table9 = _gas_residual(
        composition, 1900.0, fO2, pressures["K"], "K", fc87
    )
    assert abs(fc87_hastie - base_hastie) < 0.001
    assert abs(fc87_table9 - base_table9) < 0.001
    print(
        "Hastie/FC87 sensitivity: "
        f"K {base_hastie:+.4f} -> {no_k_half_hastie:+.4f}; "
        f"FC87 deltas Hastie {fc87_hastie - base_hastie:+.6f}, "
        f"Table 9 {fc87_table9 - base_table9:+.6f} dex"
    )


def test_tsaplin_binary_activity_sensitivities() -> None:
    na_residuals = _tsaplin_residuals(TSAPLIN_NA_ROWS, "Na2O")
    k_residuals = _tsaplin_residuals(TSAPLIN_K_ROWS, "K2O")

    # Tsaplin a(Na2O), X_Na2O <= 0.476: measured median residual +0.0607
    # dex and the largest absolute residual is 0.2505 dex, inside +/-0.3.
    assert median(na_residuals) == pytest.approx(0.06, abs=0.05)
    assert max(abs(value) for value in na_residuals) <= 0.3

    # The explicit K rows have median +0.6062 dex; the evidence summary reports
    # +0.69 dex across one more rounded row. Keep a band broad enough for that
    # transcript difference, while rejecting a half-dex shift in either direction.
    assert median(k_residuals) == pytest.approx(0.65, abs=0.1)
    print(
        "Tsaplin sensitivity: "
        f"Na median {median(na_residuals):+.4f}, "
        f"Na max abs {max(abs(value) for value in na_residuals):.4f}, "
        f"K median {median(k_residuals):+.4f} dex"
    )
