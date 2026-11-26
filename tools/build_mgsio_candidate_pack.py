#!/usr/bin/env python3
"""Build the opt-in MgO–SiO2 minimal-associate activity-fit candidate."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from openimcc.model import _d066_core_content_digests, load_datapack
from openimcc.standards import pack_liquid_standards
from candidate_pack import (
    add_write_check_arguments,
    load_d066_pack,
    serialize_candidate_pack,
    write_or_check_candidate_pack,
)


PACKS = ROOT / "src/openimcc/data/packs"
BASE = PACKS / "imcc-sf04-d066-v4.json"
DESTINATION = PACKS / "imcc-sf04-d066-mgsio-candidate-v1.json"
LIQUID_STANDARDS = ROOT / "src/openimcc/data/liquid-standards.json"
VERSION = "1.0.0-d066-mgsio-candidate-v1"
MODEL_ID = "IMCC-SF04-D066"
TARGET = "Mg2SiO4"
TARGET_A = -0.470703
TARGET_B = 6619.71
INACTIVE = "MgSiO3"
INACTIVE_REASON = (
    "removed by the source-balanced fit; the MgO–SiO2 activity data require "
    "a minimal MgO/SiO2/Mg2SiO4 description"
)
FIT_SOURCE_KEYS = (
    "MGSIO-FIT-ARUTYUNYAN-2005",
    "MGSIO-FIT-KAMBAYASHI-KATO-1984",
    "MGSIO-FIT-KAMBAYASHI-1983",
    "MGSIO-FIT-KUME-2000",
    "MGSIO-FIT-MORITA-KUME-2000",
)
FIT_SOURCES = {
    "MGSIO-FIT-ARUTYUNYAN-2005": {
        "citation": "Arutyunyan et al. (2005), MgO–SiO2 liquid activities.",
        "doi": "10.1134/1.2039976",
        "url": "https://repo.library.stonybrook.edu/xmlui/bitstream/handle/11401/70084/DOPv50i8final.pdf?sequence=2",
        "scope": "Joint a(MgO)/a(SiO2) activity-fit observations.",
    },
    "MGSIO-FIT-KAMBAYASHI-KATO-1984": {
        "citation": "Kambayashi and Kato (1984), MgO–SiO2 liquid activities.",
        "doi": "10.1016/0021-9614(84)90085-5",
        "scope": "Joint a(MgO)/a(SiO2) activity-fit observations.",
    },
    "MGSIO-FIT-KAMBAYASHI-1983": {
        "citation": "Kambayashi (1983), MgO–SiO2 liquid activities.",
        "doi": "10.1016/0021-9614(83)90136-2",
        "scope": "Joint a(MgO)/a(SiO2) activity-fit observations.",
    },
    "MGSIO-FIT-KUME-2000": {
        "citation": "Kume et al. (2000), CaO–MgO–SiO2 liquid activities.",
        "doi": "10.2355/isijinternational.40.561",
        "scope": "MgO activity and overlapping silica activity observations.",
    },
    "MGSIO-FIT-MORITA-KUME-2000": {
        "citation": "Morita, Kume, and Sano (2000), silicate-slag SiO2 activities.",
        "doi": "10.2355/isijinternational.40.554",
        "scope": "Overlapping silica activity observations; duplicate Kume values are half-weighted.",
    },
}


def build() -> dict:
    pack, original_rows = load_d066_pack(BASE)
    pack["sources"].update(copy.deepcopy(FIT_SOURCES))
    for row in pack["rows"]:
        if row["complex"] == TARGET:
            if row["A"] != -0.9093214922723483 or row["B"] != 7375.770516093468:
                raise ValueError("unexpected D066 v4 Mg2SiO4 coefficients")
            row["A"] = TARGET_A
            row["B"] = TARGET_B
            row["source_refs"] = [
                *row.get("source_refs", []),
                *FIT_SOURCE_KEYS,
            ]
            row["derivation"] += (
                ". The minimal MgO/SiO2/Mg2SiO4 description jointly refits "
                "A and B to both oxide "
                "activities across the exposed source-balanced observations; "
                "MgSiO3 is inactive in this candidate."
            )
            row["candidate_activity_fit"] = {
                "parameter": "A and B",
                "provisional": True,
                "fit_source_refs": list(FIT_SOURCE_KEYS),
                "B_preserved_from_d066_v4": False,
                "objective": (
                    "Weighted least squares of the signed log10(predicted/measured) "
                    "residuals for both a(MgO) and a(SiO2); 46 activity rows."
                ),
                "weighting": (
                    "Unit weight except coincident Kume and Morita SiO2 "
                    "observations, each assigned one-half weight."
                ),
                "parameter_bounds": {
                    "delta_H_kJ_mol": [-250.0, 100.0],
                    "delta_S_J_mol_K": [-100.0, 100.0],
                },
                "reported_weighted_joint_rms_dex": 0.124698,
                "reported_unweighted_joint_rms_dex": 0.132483,
            }
        elif row["complex"] == INACTIVE:
            row["active"] = False
            row["inactive_reason"] = INACTIVE_REASON

    # For log10 K=A+B/T, ln K=ln(10)A+ln(10)B/T. Matching
    # ln K=Delta S/R-Delta H/(RT) gives Delta S=R ln(10)A and
    # Delta H=-R ln(10)B. R is J mol^-1 K^-1 and B is K, so H is J/mol.
    # Sanity check: these coefficients give H=-126.733 kJ/mol and S=-9.011
    # J mol^-1 K^-1, both inside the declared fit bounds.
    row_digest, standards_digest = _d066_core_content_digests(pack)
    pack["created"] = "2026-11-25"
    pack["imcc_sf04_datapack_version"] = VERSION
    pack["model_id"] = MODEL_ID
    pack["model"] = (
        "Opt-in MgO–SiO2 activity-fit candidate with MgSiO3 inactive and "
        "Mg2SiO4 A/B jointly refit; not the default."
    )
    pack["pack_arithmetic"] = (
        "38 D066 v4 rows retained; MgSiO3 is inactive and Mg2SiO4 A/B are "
        "jointly refit on 46 exposed activity observations."
    )
    pack["candidate_provenance"] = {
        "status": "opt-in provisional candidate; not the default",
        "basis": (
            "D066 v4 with MgSiO3 inactive and a joint Mg2SiO4 A/B activity fit."
        ),
        "fit_set": {
            "n": 46,
            "sources": list(FIT_SOURCE_KEYS),
            "source_roles": {key: "fitted" for key in FIT_SOURCE_KEYS},
            "temperature_range_K": [1873.0, 1973.0],
            "reference_state": "Liquid oxide activities on the exposed source rows.",
            "objective": (
                "Minimize the weighted sum of squared signed log10(predicted/measured) "
                "residuals for a(MgO) and a(SiO2) jointly."
            ),
            "weighting": (
                "Unit weight except coincident Kume and Morita SiO2 observations, "
                "each assigned one-half weight."
            ),
            "excluded_sources": ["Zaitsev 2006 (held out; aggregate reported only)"],
        },
        "parameter_choice": {
            "free_parameters": "Mg2SiO4 A and B; MgSiO3 is inactive.",
            "fixed_parameters": "All other D066 v4 row coefficients and all parent liquid standards.",
            "bounds": {
                "delta_H_kJ_mol": [-250.0, 100.0],
                "delta_S_J_mol_K": [-100.0, 100.0],
            },
            "inactive_row": {"complex": INACTIVE, "reason": INACTIVE_REASON},
        },
        "d066_core_content": {
            "rows_sha256": row_digest,
            "common_parent_liquid_standards_sha256": standards_digest,
            "basis": (
                "The 38 D066 v4 row identities, stoichiometries, parent basis, "
                "and liquid standards are retained; only Mg2SiO4 A/B and the "
                "declared MgSiO3 inactive status change."
            ),
        },
        "fit_coefficients": {TARGET: {"A": TARGET_A, "B": TARGET_B}},
        "fit_thermochemistry": {
            "delta_H_kJ_mol": -126.733,
            "delta_S_J_mol_K": -9.011,
            "derivation": (
                "From log10 K=A+B/T and ln K=Delta S/R-Delta H/(RT): "
                "Delta S=R ln(10) A; Delta H=-R ln(10) B."
            ),
        },
        "validation": {
            "training_joint_rms_weighted_dex": 0.124698,
            "training_joint_rms_unweighted_dex": 0.132483,
            "zaitsev_2006_aggregate_only": {
                "n": 12,
                "median_abs_before_dex": 0.202337,
                "median_abs_after_dex": 0.109047,
                "median_signed_before_dex": -0.177989,
                "median_signed_after_dex": -0.067925,
            },
        },
    }
    pack["errata"].append(
        {
            "version": VERSION,
            "change": "Adds an opt-in minimal MgO–SiO2 associate activity-fit candidate; the default pack is unchanged.",
        }
    )

    if len(pack["rows"]) != len(original_rows):
        raise AssertionError("candidate row count differs from D066 v4")
    for before, after in zip(original_rows, pack["rows"], strict=True):
        if before["complex"] == TARGET:
            expected = copy.deepcopy(before)
            expected["A"] = TARGET_A
            expected["B"] = TARGET_B
            expected["source_refs"] = [*expected["source_refs"], *FIT_SOURCE_KEYS]
            expected["derivation"] += (
                ". The minimal MgO/SiO2/Mg2SiO4 description jointly refits "
                "A and B to both oxide "
                "activities across the exposed source-balanced observations; "
                "MgSiO3 is inactive in this candidate."
            )
            expected["candidate_activity_fit"] = after["candidate_activity_fit"]
            if expected != after:
                raise AssertionError("unexpected Mg2SiO4 candidate-row change")
        elif before["complex"] == INACTIVE:
            expected = copy.deepcopy(before)
            expected["active"] = False
            expected["inactive_reason"] = INACTIVE_REASON
            if expected != after:
                raise AssertionError("unexpected MgSiO3 candidate-row change")
        elif before != after:
            raise AssertionError(f"non-target row changed: {before['complex']}")
    return pack


def write_or_check_liquid_standards(*, write: bool) -> int:
    digest = load_datapack(DESTINATION).binding_digest
    base_digest = load_datapack(BASE).binding_digest
    if not digest or not base_digest:
        print("candidate or D066 v4 pack has no binding digest")
        return 1

    registry = json.loads(LIQUID_STANDARDS.read_text(encoding="utf-8"))
    entries = registry.get("packs")
    if not isinstance(entries, dict):
        print("liquid-standards registry has no packs object")
        return 1
    base_entry = entries.get(base_digest)
    if not isinstance(base_entry, dict):
        print("D066 v4 has no liquid-standards declaration")
        return 1
    candidate_entry = copy.deepcopy(base_entry)
    candidate_entry["file"] = DESTINATION.name
    candidate_entry["model_id"] = MODEL_ID
    candidate_entry["version"] = VERSION
    candidate_entry["standard_citations"] = {
        oxide: citation.replace(BASE.name, DESTINATION.name)
        for oxide, citation in candidate_entry["standard_citations"].items()
    }

    if write:
        for stale_digest, entry in list(entries.items()):
            if isinstance(entry, dict) and entry.get("file") == DESTINATION.name:
                entries.pop(stale_digest)
        entries[digest] = candidate_entry
        LIQUID_STANDARDS.write_text(
            json.dumps(registry, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"registered liquid standards for binding digest {digest}")
    elif entries.get(digest) != candidate_entry:
        print(f"liquid-standards declaration is stale for {DESTINATION.name}")
        return 1

    declared = pack_liquid_standards(DESTINATION)
    if declared.source == "undeclared" or not (
        declared.standards or declared.row_overrides
    ):
        print(f"liquid-standards declaration is empty for {DESTINATION.name}")
        return 1
    print(f"liquid-standards declaration is current for {DESTINATION.name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    add_write_check_arguments(parser)
    args = parser.parse_args()
    generated = serialize_candidate_pack(build())
    pack_status = write_or_check_candidate_pack(
        DESTINATION, generated, write=args.write
    )
    if pack_status:
        return pack_status
    return write_or_check_liquid_standards(write=args.write)


if __name__ == "__main__":
    raise SystemExit(main())
