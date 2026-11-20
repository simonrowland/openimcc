#!/usr/bin/env python3
"""Build the packaged elemental-reference metal crystals from vendored cards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data-src" / "published-thermo"
TARGET = (
    ROOT
    / "src"
    / "openimcc"
    / "data"
    / "diagnostics"
    / "elemental-reference-crystal-cards.json"
)
SCHEMA = "openimcc-elemental-reference-crystal-cards-v1"
ELEMENTS = (
    ("Au", "gold", "burcat-au.json", "Au(cr)", "BU-0054"),
    ("Pd", "palladium", "burcat-pd.json", "Pd(cr)", "BU-3110"),
    ("Pt", "platinum", "burcat-pt.json", "Pt(cr)", "BU-3120"),
    ("Rh", "rhodium", "pankratz-rh.json", "P672-Rh-cr", "P672-Rh-cr"),
    ("Ir", "iridium", "burcat-ir.json", "Ir(cr)", "BU-2611"),
    ("Os", "osmium", "burcat-os.json", "Os(cr)", "BURCAT-OS-CR"),
    ("Ru", "ruthenium", "pankratz-ru.json", "P672-Ru-cr", "P672-Ru-cr"),
)
BURCAT_FILES = frozenset(
    {"burcat-au.json", "burcat-pd.json", "burcat-pt.json", "burcat-ir.json", "burcat-os.json"}
)


def _burcat_record(
    element: str,
    element_name: str,
    filename: str,
    record_key: str,
    source_card_id: str,
) -> dict[str, Any]:
    source_document = json.loads((SOURCE_DIR / filename).read_text(encoding="utf-8"))
    source = source_document["records"][record_key]
    if source["record_id"] != source_card_id or source["phase"] != "crystal":
        raise ValueError(f"{filename}/{record_key}: unexpected crystal card identity")
    lower_K, upper_K = map(float, source["temperature_range_K"])
    if lower_K >= upper_K:
        raise ValueError(f"{source_card_id}: invalid temperature range")

    low = source.get("low_coefficients", source.get("coefficients"))
    high = source.get("high_coefficients")
    if not isinstance(low, list) or len(low) != 7:
        raise ValueError(f"{source_card_id}: expected a seven-coefficient low block")
    if high is None:
        segments = [{"temperature_range_K": [lower_K, upper_K], "coefficients": low}]
    else:
        if not isinstance(high, list) or len(high) != 7 or lower_K >= 1000.0 or upper_K <= 1000.0:
            raise ValueError(f"{source_card_id}: invalid NASA-7 low/high blocks")
        segments = [
            {"temperature_range_K": [lower_K, 1000.0], "coefficients": low},
            {"temperature_range_K": [1000.0, upper_K], "coefficients": high},
        ]

    return {
        "formula": element,
        "phase": "crystal",
        "elemental_reference": True,
        "polymorph": f"{element_name} elemental reference crystal",
        "source_card_id": source_card_id,
        "source_table_id": source_card_id,
        "source_path": f"data-src/published-thermo/{filename}",
        "source_title": source_document["source"],
        "source_locator": f"BURCAT.THR.txt card {source_card_id}",
        "source_note": source["source_note"],
        "functional_form": "NASA-7 polynomial",
        "form": "nasa7",
        "temperature_range_K": [lower_K, upper_K],
        "source_limit_K": float(source["source_limit_K"]),
        "formation_enthalpy_298_kJ_mol": float(
            source["delta_f_H_298_15_kJ_mol"]
        ),
        "segments": segments,
    }


def _pankratz_record(
    element: str,
    element_name: str,
    filename: str,
    table_id: str,
    source_card_id: str,
) -> dict[str, Any]:
    source_document = json.loads((SOURCE_DIR / filename).read_text(encoding="utf-8"))
    source = source_document["tables"][table_id]
    if (
        source["species"] != f"{element}(cr)"
        or source["phase"] != "crystal"
        or table_id != source_card_id
    ):
        raise ValueError(f"{table_id}: unexpected tabulated crystal identity")
    nodes = source["nodes"]
    lower_K = float(nodes[0][0])
    upper_K = float(nodes[-1][0])
    if lower_K >= upper_K or upper_K != float(source["source_limit_K"]):
        raise ValueError(f"{table_id}: node range does not reach its source limit")

    return {
        "formula": element,
        "phase": "crystal",
        "elemental_reference": True,
        "polymorph": f"{element_name} elemental reference crystal",
        "source_card_id": source_card_id,
        "source_table_id": source_card_id,
        "source_path": f"data-src/published-thermo/{filename}",
        "source_title": source_document["publication"],
        "source_locator": source["table"],
        "source_note": source["source_note"],
        "functional_form": "tabulated Cp, S and H-H298",
        "form": "tabulated",
        "temperature_range_K": [lower_K, upper_K],
        "source_limit_K": float(source["source_limit_K"]),
        "formation_enthalpy_298_kJ_mol": (
            float(source["formation_enthalpy_298_kcal_mol"]) * 4.184
        ),
        "node_columns": source["node_columns"],
        "nodes": nodes,
    }


def build() -> dict[str, Any]:
    records = []
    for element, name, filename, source_key, source_card_id in ELEMENTS:
        if filename in BURCAT_FILES:
            records.append(
                _burcat_record(
                    element, name, filename, source_key, source_card_id
                )
            )
        else:
            records.append(
                _pankratz_record(
                    element, name, filename, source_key, source_card_id
                )
            )
    return {"schema": SCHEMA, "records": records}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the packaged dataset differs from the vendored cards",
    )
    args = parser.parse_args()
    rendered = json.dumps(build(), indent=2, ensure_ascii=False) + "\n"
    if args.check:
        if not TARGET.exists() or TARGET.read_text(encoding="utf-8") != rendered:
            print("elemental reference crystal card dataset is stale", file=sys.stderr)
            return 1
        print("elemental reference crystal card dataset is current")
        return 0
    TARGET.write_text(rendered, encoding="utf-8")
    print("rebuilt elemental reference crystal card dataset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
