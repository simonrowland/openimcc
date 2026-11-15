#!/usr/bin/env python3
"""Regenerate bundled JANAF crystal thermochemistry for declared assemblages."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openimcc.janaf_crystals import _JANAF_OXIDE_PAIRS  # noqa: E402


SOURCE = ROOT / "data-src" / "janaf"
TARGET = (
    ROOT / "src" / "openimcc" / "data" / "diagnostics"
    / "janaf-pure-oxide-crystals.json"
)
LIQUID_TARGET = (
    ROOT / "src" / "openimcc" / "data" / "diagnostics"
    / "janaf-pure-oxide-liquid-pairs.json"
)

def _source_rows(table: dict) -> tuple[list[dict], list[dict]]:
    values = table["values"]
    nodes = [
        {
            "T_K": float(row["temperature"]["value"]),
            "enthalpy_increment_kJ_mol": float(
                row["enthalpy_increment"]["value"]
            ),
            "entropy_J_mol_K": float(row["entropy"]["value"]),
        }
        for row in values
        if row["enthalpy_increment"]["value"] is not None
        and row["entropy"]["value"] is not None
    ]
    nodes.sort(key=lambda node: node["T_K"])
    # JANAF's text parser records phase-transition lines as ambiguities
    # because their formation columns contain labels. Preserve their first
    # five tab-separated thermochemical columns for package-backed evaluation.
    transition_rows = []
    for ambiguity in table.get("parse_ambiguities", []):
        raw_line = ambiguity.get("raw_line", "")
        fields = raw_line.split("\t")
        if len(fields) < 6:
            continue
        try:
            temperature_K, _cp, entropy, _gibbs_function, h_increment = map(
                float, fields[:5]
            )
        except ValueError:
            continue
        label = fields[5].strip()
        if not ("<-->" in label or "TRANSITION" in label.upper()):
            continue
        transition_rows.append(
            {
                "T_K": temperature_K,
                "enthalpy_increment_kJ_mol": h_increment,
                "entropy_J_mol_K": entropy,
                "label": label,
                "source_line_number": ambiguity.get("line_number"),
                "raw_line": raw_line,
            }
        )
    transition_rows.sort(key=lambda row: (row["T_K"], row["label"]))
    return nodes, transition_rows


def _prepare_source_table(table_id: str) -> dict:
    raw = (SOURCE / f"{table_id}.yaml").read_bytes()
    table = json.loads(raw.decode("utf-8"))["table"]
    reference = next(
        row for row in table["values"]
        if float(row["temperature"]["value"]) == 298.15
    )
    h298 = reference["formation_enthalpy"]["value"]
    if h298 is None:
        raise ValueError(f"{table_id} has no 298.15 K formation enthalpy")
    nodes, transition_rows = _source_rows(table)
    source_temperatures = [node["T_K"] for node in nodes] + [
        row["T_K"] for row in transition_rows
    ]
    return {
        "formation_enthalpy_298_kJ_mol": float(h298),
        "nodes": nodes,
        "transition_rows": transition_rows,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_temperature_K": [
            min(source_temperatures), max(source_temperatures)
        ],
    }


def main() -> None:
    phases = []
    for oxide, definition in _JANAF_OXIDE_PAIRS.items():
        table_id = definition.crystal_table
        transition_K = definition.transition_K
        prepared = _prepare_source_table(table_id)
        nodes = prepared["nodes"]
        transition_rows = prepared["transition_rows"]
        transition_at_cap = next(
            (
                row for row in transition_rows
                if row["T_K"] == transition_K
                and "<-->" in row["label"]
                and "LIQUID" in row["label"].upper()
            ),
            None,
        )
        phases.append(
            {
                "id": oxide,
                "formula": oxide,
                "oxide_stoichiometry": {oxide: 1.0},
                "crystal_table_id": table_id,
                "source_authority": "janaf_transcribed",
                "source_sha256": prepared["source_sha256"],
                "formation_enthalpy_298_kJ_mol": prepared[
                    "formation_enthalpy_298_kJ_mol"
                ],
                "source_temperature_K": prepared["source_temperature_K"],
                "crystal_transition_temperature_K": transition_K,
                "nodes": nodes,
                "transition_rows": transition_rows,
                "transition_row_at_cap": transition_at_cap,
                "construction_uncertainty_input_K": (
                    list(definition.construction_uncertainty_input_K)
                    if definition.construction_uncertainty_input_K is not None
                    else None
                ),
            }
        )

    k2o_uncertainty = _JANAF_OXIDE_PAIRS["K2O"].construction_uncertainty_input_K
    assert k2o_uncertainty is not None
    result = {
        "schema": "openimcc-janaf-apparent-crystals-v2",
        "source": "Chase (1998), NIST-JANAF Thermochemical Tables, 4th edition",
        "transformation": (
            "Bundle each source table's Hf(298.15 K), H(T)-H(298.15 K), and S(T) "
            "columns. Runtime apparent G(T) = Hf(298.15 K) + [H(T)-H(298.15 K)] "
            "- T*S(T), converting kJ/mol enthalpies and J/(mol K) entropy to J/mol. "
            "Interpolate source enthalpy increment and entropy separately, linearly. "
            "Also retain finite enthalpy-increment/entropy columns from identified "
            "transition lines in parse_ambiguities and use the crystal-liquid row "
            "at its inclusive cap instead of interpolation."
        ),
        "source_span_rule": (
            "Finite source enthalpy-increment and entropy nodes; evaluation starts "
            "at 298.15 K."
        ),
        "phase_restriction_rule": (
            "A JANAF crystal is valid through its crystal-to-liquid transition "
            "temperature, inclusive; never evaluate it above that transition."
        ),
        "construction_uncertainty_rule": (
            f"K2O's {k2o_uncertainty[0]:g}-{k2o_uncertainty[1]:g} K range is "
            "uncertainty in the selected fusion-temperature "
            "input, not an evaluation-temperature domain."
        ),
        "excluded_fusion_data": {
            "O-038_1700_K": (
                "Prior duplicated fusion node 2.3495 kJ/mol is discarded. The "
                "vendored O-038 raw ambiguous 1700 K row gives liquid formation "
                "G=-609.059 kJ/mol; O-035 crystal formation G=-610.202 kJ/mol, "
                "so direct formation-Gibbs subtraction is 1.143 kJ/mol. O-038 is "
                "not used to construct crystal apparent Gibbs; the runtime liquid "
                "standard is supplied by species_thermo."
            )
        },
        "phases": phases,
    }
    TARGET.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    liquid_rows = []
    for oxide, definition in _JANAF_OXIDE_PAIRS.items():
        table_id = definition.liquid_table
        if table_id is None:
            continue
        prepared = _prepare_source_table(table_id)
        liquid_rows.append(
            {
                "oxide": oxide,
                "table_id": table_id,
                "source_authority": "janaf_transcribed",
                "source_sha256": prepared["source_sha256"],
                "formation_enthalpy_298_kJ_mol": prepared[
                    "formation_enthalpy_298_kJ_mol"
                ],
                "source_temperature_K": prepared["source_temperature_K"],
                "nodes": prepared["nodes"],
                "transition_rows": prepared["transition_rows"],
            }
        )
    liquid_result = {
        "schema": "openimcc-janaf-apparent-liquid-pairs-v1",
        "source": "Chase (1998), NIST-JANAF Thermochemical Tables, 4th edition",
        "transformation": result["transformation"],
        "source_span_rule": (
            "Finite source enthalpy-increment and entropy nodes, including "
            "printed transition rows; evaluation starts at 298.15 K."
        ),
        "rows": liquid_rows,
    }
    LIQUID_TARGET.write_text(
        json.dumps(liquid_result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
