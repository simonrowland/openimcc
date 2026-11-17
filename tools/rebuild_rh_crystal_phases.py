#!/usr/bin/env python3
"""Rebuild the Robie–Hemingway crystal fusion nodes on an absolute-G basis."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "src/openimcc/data/diagnostics/crystal-phases.json"
SOURCE_COMMIT = "9aacc2f6308e49a5a1988c25487518ba346d9ad0"
RECORD_DIRECTORY = (
    "data/literature/compilations/robie-hemingway-1995-usgs-b2131/records"
)
RECORDS = {
    "fayalite": (332, "FAYALITE"),
    "hercynite": (233, "HERCYNITE"),
    "ilmenite": (235, "ILMENITE"),
    "perovskite": (232, "PEROVSKITE"),
    "ulvospinel": (236, "ULVOSPINEL"),
}
SOURCE_SOLID_LIMITS = {
    "fayalite": {
        "temperature_K": 1490.0,
        "kind": "melting",
        "locator": {"page": 332, "row": "Melting T", "column": "temperature"},
    },
    "ilmenite": {
        "temperature_K": 1640.0,
        "kind": "melting",
        "locator": {
            "page": 235,
            "row": "Rhombohedral crystals 298.15 to 1640 K",
            "column": "phase_as_published",
        },
    },
    "perovskite": {
        "temperature_K": 2188.0,
        "kind": "melting",
        "locator": {"page": 232, "row": "Melting T", "column": "temperature"},
    },
}


def _read_record(source_repository: Path, page: int) -> dict[str, Any]:
    relative_path = f"{RECORD_DIRECTORY}/high-temperature-p{page}.json"
    raw = subprocess.run(
        ["git", "-C", str(source_repository), "show", f"{SOURCE_COMMIT}:{relative_path}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(raw)


def _ambiguous_cells(record: dict[str, Any]) -> set[tuple[int, str]]:
    return {
        (int(ambiguity["row"]), column)
        for ambiguity in record.get("ambiguities", ())
        if ambiguity.get("kind") == "ocr_suspect"
        for column in ambiguity.get("columns", ())
    }


def _value(
    record: dict[str, Any], row_index: int, row: dict[str, Any], column: str,
) -> float:
    cell = row["cells"][column]
    if (row_index, column) in _ambiguous_cells(record) or cell.get("ocr_suspect"):
        raise ValueError(
            f"{record['record_id']} row {row_index} {column} is OCR-ambiguous"
        )
    value = cell.get("value")
    if value is None or not math.isfinite(float(value)):
        raise ValueError(
            f"{record['record_id']} row {row_index} {column} is not numeric"
        )
    value = float(value)
    raw = cell.get("raw")
    try:
        raw_value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"{record['record_id']} row {row_index} {column} has no numeric raw token"
        ) from None
    if not math.isclose(raw_value, value, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"{record['record_id']} row {row_index} {column} differs from its raw token"
        )
    return value


def _temperatures(record: dict[str, Any]) -> list[float]:
    temperatures = []
    for row in record["rows"]:
        value = row["cells"]["temperature"].get("value")
        if value is not None:
            temperatures.append(float(value))
    if any(left >= right for left, right in zip(temperatures, temperatures[1:])):
        raise ValueError(f"{record['record_id']} rows are not in increasing temperature order")
    return temperatures


def _check_hsg(record: dict[str, Any], row_index: int, row: dict[str, Any]) -> None:
    entropy = _value(record, row_index, row, "entropy")
    enthalpy_function = _value(record, row_index, row, "enthalpy_function")
    gibbs_function = _value(record, row_index, row, "gibbs_function")
    # The printed columns are S, (H-H298)/T, and -(G-H298)/T in J/(mol K).
    # Therefore G=H-T*S is the table identity G_function=S-H_function.
    if abs(gibbs_function - (entropy - enthalpy_function)) > 0.021:
        raise ValueError(
            f"{record['record_id']} row {row_index} fails H/S/G consistency"
        )


def _anchor_enthalpy(record: dict[str, Any]) -> float:
    rows = record["rows"]
    anchor_matches = [
        (index, row) for index, row in enumerate(rows)
        if row["cells"]["temperature"].get("value") is not None
        and float(row["cells"]["temperature"]["value"]) == 298.15
    ]
    if len(anchor_matches) != 1:
        raise ValueError(f"{record['record_id']} must have one 298.15 K anchor row")
    index, row = anchor_matches[0]
    temperature = row["cells"]["temperature"]
    if temperature.get("ocr_suspect") or (index, "temperature") in _ambiguous_cells(record):
        # p. 235's anchor is flagged despite its raw token and table sequence.
        # The immediately following 300 K row confirms the standard 298.15 K anchor.
        following = (
            rows[index + 1]["cells"]["temperature"]
            if index + 1 < len(rows) else {}
        )
        if not (
            record["page"] == 235
            and temperature.get("raw", "").strip() == "298.15"
            and float(temperature.get("value")) == 298.15
            and float(following.get("value")) == 300.0
        ):
            raise ValueError(f"{record['record_id']} has an unresolved 298.15 K anchor")
    else:
        _value(record, index, row, "temperature")
    if (index, "formation_enthalpy") in _ambiguous_cells(record):
        raise ValueError(f"{record['record_id']} has an ambiguous ΔfH298 cell")
    h298 = _value(record, index, row, "formation_enthalpy")
    next_h = _value(record, index + 1, rows[index + 1], "formation_enthalpy")
    if abs(next_h - h298) > 2.0:
        raise ValueError(
            f"{record['record_id']} ΔfH changes too sharply after 298.15 K"
        )
    return h298


def _validate_source_solid_limit(record: dict[str, Any], phase_id: str) -> None:
    limit = SOURCE_SOLID_LIMITS.get(phase_id)
    if limit is None:
        return
    description = record["phase_as_published"]
    if phase_id == "ilmenite":
        if "298.15 to 1640 K" not in description:
            raise ValueError("ilmenite source range does not support the 1640 K limit")
        return
    footer_tokens = [
        token
        for item in record.get("metadata", ())
        if item.get("section") == "footer"
        and "Melting T" in item.get("raw", "")
        for token in item.get("tokens", ())
        if token.get("value") is not None and not token.get("ocr_suspect")
    ]
    source_temperature = float(limit["temperature_K"])
    if not any(float(token["value"]) == source_temperature for token in footer_tokens):
        raise ValueError(f"{phase_id} source record does not print its solid limit")
    if phase_id == "fayalite" and "melts incongruently at 1490 K" not in description:
        raise ValueError("fayalite source description does not support its melting limit")


def _node_rows(
    record: dict[str, Any], temperatures: list[float],
) -> dict[float, tuple[float, float]]:
    source_rows = record["rows"]
    if _temperatures(record) != [
        float(row["cells"]["temperature"]["value"])
        for row in source_rows
        if row["cells"]["temperature"].get("value") is not None
    ]:
        raise ValueError(f"{record['record_id']} has missing source temperatures")
    indexes_by_temperature = {
        float(row["cells"]["temperature"]["value"]): index
        for index, row in enumerate(source_rows)
        if row["cells"]["temperature"].get("value") is not None
    }
    nodes = {}
    for temperature in temperatures:
        if temperature not in indexes_by_temperature:
            raise ValueError(
                f"{record['record_id']} has no source row at {temperature:g} K"
            )
        index = indexes_by_temperature[temperature]
        row = source_rows[index]
        _value(record, index, row, "temperature")
        for neighbor_index in (index - 1, index, index + 1):
            if 0 <= neighbor_index < len(source_rows):
                _check_hsg(record, neighbor_index, source_rows[neighbor_index])
        entropy = _value(record, index, row, "entropy")
        enthalpy_function = _value(record, index, row, "enthalpy_function")
        nodes[temperature] = (enthalpy_function * temperature / 1000.0, entropy)
    return nodes


def build(data: dict[str, Any], source_repository: Path) -> dict[str, Any]:
    """Return crystal phases v2 using R&H crystal and packaged parent absolute G."""
    sys.path.insert(0, str(ROOT / "src"))
    from openimcc.diagnostics import _crystal_thermo_context
    from openimcc.gas import species_thermo

    result = json.loads(json.dumps(data))
    result["schema"] = "openimcc-crystal-phases-v2"
    result["source"] = {
        "citation": (
            "Robie and Hemingway (1995), USGS Bulletin 2131, "
            "Thermodynamic Properties of Minerals and Related Substances"
        ),
        "source_commit": SOURCE_COMMIT,
        "record_path_template": f"{RECORD_DIRECTORY}/high-temperature-p{{page}}.json",
    }
    result["units"]["formation_enthalpy_298_kJ_mol"] = (
        "kJ mol-1 crystal formula unit, printed R&H value at 298.15 K"
    )
    result["units"]["source_crystal_enthalpy_increment_nodes_kJ_mol"] = (
        "kJ mol-1 crystal formula unit, H(T)-H(298.15 K)"
    )
    result["units"]["source_crystal_entropy_nodes_J_mol_K"] = (
        "J mol-1 K-1 crystal formula unit, printed R&H entropy"
    )
    result["fusion_node_method"] = {
        "method": (
            "Berman and JANAF pure-oxide-control rows are unchanged. R&H rows "
            "use an absolute-G Hess cycle: G_abs,crystal = ΔfH298 + "
            "[H(T)-H(298.15)] - T*S(T), and ΔG_fus = ΣνG_abs,parent(liquid) "
            "- G_abs,R&H(crystal)."
        ),
        "liquid_standard_source": (
            "The existing packaged parent-liquid Gibbs standards are evaluated "
            "by the gas-layer Gibbs helper, preserving the solver's standard states."
        ),
        "source": (
            "Berman (1988), Tables 2, 3a/3b, and Table 5 where applicable; "
            "Chase (1998), NIST-JANAF Thermochemical Tables, 4th edition; "
            "Robie and Hemingway (1995), USGS Bulletin 2131, cited per phase."
        ),
        "interpolation": (
            "Piecewise linear only between listed source nodes; "
            "no endpoint extrapolation."
        ),
        "element_reference_source": (
            "Chase (1998), NIST-JANAF Thermochemical Tables, 4th edition. "
            "The printed element G_app nodes are retained for audit; R&H fusion "
            "nodes no longer subtract these element-reference functions."
        ),
    }
    # These cached JANAF Si-001 G_app values were transcribed incorrectly in v1.
    result["element_reference_g_app_nodes_kJ_mol"]["Si"]["1700"] = -69.4688
    result["element_reference_g_app_nodes_kJ_mol"]["Si"]["1800"] = -78.7284

    phases = {phase["id"]: phase for phase in result["phases"]}
    gas_pack = _crystal_thermo_context(None)[0]
    for phase_id, (page, expected_name) in RECORDS.items():
        phase = phases[phase_id]
        record = _read_record(source_repository, page)
        if (
            record.get("page") != page
            or record.get("name_as_published") != expected_name
            or record.get("transcription_status") != "transcribed_with_ocr_ambiguities"
        ):
            raise ValueError(f"unexpected R&H record for {phase_id} p. {page}")
        _validate_source_solid_limit(record, phase_id)
        h298 = _anchor_enthalpy(record)
        source_temperatures = [
            float(value) for value in phase["source_node_temperatures_K"]
        ]
        source_nodes = _node_rows(record, source_temperatures)
        phase["source"]["printed_phase_description"] = record["phase_as_published"]
        phase["formation_enthalpy_298_kJ_mol"] = h298
        phase["source_crystal_enthalpy_increment_nodes_kJ_mol"] = {}
        phase["source_crystal_entropy_nodes_J_mol_K"] = {}
        rebuilt = {}
        for temperature in source_temperatures:
            key = str(int(temperature)) if temperature.is_integer() else str(temperature)
            enthalpy_increment, entropy = source_nodes[temperature]
            phase["source_crystal_enthalpy_increment_nodes_kJ_mol"][key] = (
                enthalpy_increment
            )
            phase["source_crystal_entropy_nodes_J_mol_K"][key] = entropy
            # Hf and the enthalpy increment are kJ/mol; T*S is converted from J/mol.
            crystal_g = h298 + enthalpy_increment - temperature * entropy / 1000.0
            parent_g = math.fsum(
                float(coefficient)
                * species_thermo(oxide, "l", temperature, gas_pack).G_J_mol
                / 1000.0
                for oxide, coefficient in phase["oxide_stoichiometry"].items()
            )
            rebuilt[key] = parent_g - crystal_g
        phase["fusion_gibbs_nodes_kJ_mol"] = rebuilt
        if phase_id in SOURCE_SOLID_LIMITS:
            phase["source_solid_limit"] = SOURCE_SOLID_LIMITS[phase_id]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    parser.add_argument(
        "source_repository",
        type=Path,
        help="simulator checkout containing the pinned Robie–Hemingway records",
    )
    args = parser.parse_args()

    original = DATA.read_text(encoding="utf-8")
    rebuilt = json.dumps(
        build(json.loads(original), args.source_repository),
        indent=2,
        ensure_ascii=False,
    ) + "\n"
    if args.check:
        if original != rebuilt:
            print("Robie–Hemingway crystal fusion nodes are stale", file=sys.stderr)
            return 1
        print("Robie–Hemingway crystal fusion nodes are current")
        return 0
    DATA.write_text(rebuilt, encoding="utf-8")
    print("Rebuilt Robie–Hemingway crystal fusion nodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
