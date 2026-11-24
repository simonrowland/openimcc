"""Shared builder pieces for the opt-in D066 candidate packs."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CLASS_C_J_MOL_ATOM_K = 8.297530569005671
FIT_METHOD = "least squares on 1001 evenly spaced pointwise re-referenced log10 K values"


def load_d066_pack(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load the D066 base pack and keep an independent copy of its core rows."""
    pack = json.loads(path.read_text(encoding="utf-8"))
    core_rows = copy.deepcopy(pack["rows"])
    pack["rows"] = copy.deepcopy(core_rows)
    return pack, core_rows


def fit_log10_k(
    temperatures: Sequence[float] | np.ndarray,
    values: Sequence[float] | np.ndarray,
) -> tuple[float, float, float, float]:
    """Least-squares fit log10(K)=A+B/T and return A, B, max, and RMS residuals."""
    temperature_values = np.asarray(temperatures, dtype=float)
    log_k_values = np.asarray(values, dtype=float)
    # Premise: log10(K)=A+B/T is linear in [1,1/T]. A is dimensionless,
    # B has units K, and the least-squares residual is dex. Sanity: A+B/2000
    # reproduces the fitted 2000 K value when 2000 K is in the source grid.
    design = np.column_stack(
        (np.ones(len(temperature_values)), 1.0 / temperature_values)
    )
    coefficients, *_ = np.linalg.lstsq(design, log_k_values, rcond=None)
    residual = log_k_values - design @ coefficients
    return (
        float(coefficients[0]),
        float(coefficients[1]),
        float(np.max(np.abs(residual))),
        float(np.sqrt(np.mean(residual**2))),
    )


def candidate_row(
    *,
    name: str,
    index: int,
    parent_oxides: dict[str, float],
    nu: dict[str, float],
    coefficients: tuple[float, float, float, float],
    fit_low_K: float,
    fit_high_K: float,
    paper_domain: list[float],
    source_refs: list[str],
    derivation: str,
    domain_basis: str,
    band_explanation: str,
    independent_check_reason: str,
    flags: list[str] | None = None,
    include_rms_residual: bool = False,
) -> dict[str, Any]:
    """Build the common D066 candidate row shell around candidate science."""
    intercept, slope, max_residual, rms_residual = coefficients
    refit = {
        "parents": parent_oxides,
        "temperature_range_K": [fit_low_K, fit_high_K],
        "fit_method": FIT_METHOD,
        "max_abs_residual_dex": max_residual,
    }
    if include_rms_residual:
        refit["rms_residual_dex"] = rms_residual
    refit["flag_over_0_01_dex"] = bool(max_residual > 0.01)
    return {
        "row": index,
        "complex": name,
        "reaction": " + ".join(
            f"{amount:g} {oxide}(liq)" for oxide, amount in parent_oxides.items()
        ) + f" = {name}(liq)",
        "nu": nu,
        "A": intercept,
        "B": slope,
        "source": source_refs[0],
        "T_domain_K": [fit_low_K, fit_high_K],
        "T_domain_basis": domain_basis,
        "provenance_class": "C-independent-thermochemistry",
        "derivation": derivation,
        "T_domain_paper_demonstrated_K": paper_domain,
        "coverage": "C-independent-thermochemistry",
        **({"flags": flags} if flags is not None else {}),
        "fit_residual": None,
        "accuracy_band": None,
        "band_basis": {
            "kind": "none",
            "explanation": band_explanation,
            "provenance": None,
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
            "reason": independent_check_reason,
        },
        "source_refs": source_refs,
        "standard_state_refit": refit,
    }


def add_write_check_arguments(parser: argparse.ArgumentParser) -> None:
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")


def serialize_candidate_pack(pack: dict[str, Any]) -> str:
    return json.dumps(pack, indent=2, ensure_ascii=False) + "\n"


def write_or_check_candidate_pack(
    destination: Path,
    generated: str,
    *,
    write: bool,
) -> int:
    relative = destination.relative_to(ROOT)
    if write:
        destination.write_text(generated, encoding="utf-8")
        print(f"wrote {relative}")
        return 0
    if (
        not destination.exists()
        or destination.read_text(encoding="utf-8") != generated
    ):
        print(f"{relative} is stale; run with --write")
        return 1
    print(f"{relative} is current")
    return 0


def _array_item_texts(raw: str, key: str) -> list[str]:
    """Return exact JSON source slices for each item in a named array."""
    marker = f'"{key}"'
    marker_start = raw.index(marker)
    array_start = raw.index("[", marker_start + len(marker))
    decoder = json.JSONDecoder()
    cursor = array_start + 1
    items = []
    while True:
        while cursor < len(raw) and raw[cursor].isspace():
            cursor += 1
        if cursor >= len(raw) or raw[cursor] == "]":
            return items
        _value, end = decoder.raw_decode(raw, cursor)
        items.append(raw[cursor:end])
        cursor = end
        while cursor < len(raw) and raw[cursor].isspace():
            cursor += 1
        if cursor < len(raw) and raw[cursor] == ",":
            cursor += 1
        elif cursor < len(raw) and raw[cursor] == "]":
            return items
        else:
            raise ValueError(f"malformed {key} array while checking source bytes")


def assert_core_rows_byte_equal(
    base_path: Path,
    generated: str,
    *,
    expected_count: int,
    message: str,
) -> None:
    base_rows = _array_item_texts(base_path.read_text(encoding="utf-8"), "rows")
    candidate_rows = _array_item_texts(generated, "rows")
    if (
        len(base_rows) != expected_count
        or candidate_rows[:expected_count] != base_rows
    ):
        raise AssertionError(message)
