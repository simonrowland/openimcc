#!/usr/bin/env python3
"""Regenerate bundled JANAF crystal thermochemistry for declared assemblages."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data-src" / "janaf"
TARGET = (
    ROOT / "src" / "openimcc" / "data" / "diagnostics"
    / "janaf-pure-oxide-crystals.json"
)

# The cap is the JANAF crystal-to-liquid transition temperature (K), inclusive.
PHASES = {
    "SiO2": ("O-035", 1996.0),
    "MgO": ("Mg-008", 3105.0),
    "CaO": ("Ca-027", 3200.0),
    "Al2O3": ("Al-096", 2327.0),
    "FeO": ("Fe-018", 1650.0),
    "Na2O": ("Na-012", 1405.2),
    "TiO2": ("O-043", 2130.0),
    "K2O": ("K-012", 1013.0),
}


def main() -> None:
    phases = []
    for oxide, (table_id, transition_K) in PHASES.items():
        raw = (SOURCE / f"{table_id}.yaml").read_bytes()
        table = json.loads(raw.decode("utf-8"))["table"]
        values = table["values"]
        reference = next(
            row for row in values
            if float(row["temperature"]["value"]) == 298.15
        )
        h298 = reference["formation_enthalpy"]["value"]
        if h298 is None:
            raise ValueError(f"{table_id} has no 298.15 K formation enthalpy")
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
        phases.append(
            {
                "id": oxide,
                "formula": oxide,
                "oxide_stoichiometry": {oxide: 1.0},
                "crystal_table_id": table_id,
                "source_authority": "janaf_transcribed",
                "source_sha256": hashlib.sha256(raw).hexdigest(),
                "formation_enthalpy_298_kJ_mol": float(h298),
                "source_temperature_K": [nodes[0]["T_K"], nodes[-1]["T_K"]],
                "crystal_transition_temperature_K": transition_K,
                "nodes": nodes,
                "construction_uncertainty_input_K": (
                    [919.0, 1190.0] if oxide == "K2O" else None
                ),
            }
        )

    result = {
        "schema": "openimcc-janaf-apparent-crystals-v1",
        "source": "Chase (1998), NIST-JANAF Thermochemical Tables, 4th edition",
        "transformation": (
            "Bundle each source table's Hf(298.15 K), H(T)-H(298.15 K), and S(T) "
            "columns. Runtime apparent G(T) = Hf(298.15 K) + [H(T)-H(298.15 K)] "
            "- T*S(T), converting kJ/mol enthalpies and J/(mol K) entropy to J/mol. "
            "Interpolate source enthalpy increment and entropy separately, linearly."
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
            "K2O's 919-1190 K range is uncertainty in the selected fusion-temperature "
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


if __name__ == "__main__":
    main()
