#!/usr/bin/env python3
"""Build the opt-in fixed Wu MgO–SiO2 MQM candidate pack."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from openimcc.model import _d066_core_content_digests, load_datapack


ROOT = Path(__file__).resolve().parents[1]
PACKS = ROOT / "src/openimcc/data/packs"
BASE = PACKS / "imcc-sf04-d066-v4.json"
DESTINATION = PACKS / "imcc-sf04-d066-mqm-candidate-v1.json"
LIQUID_STANDARDS = ROOT / "src/openimcc/data/liquid-standards.json"
VERSION = "1.0.0-d066-mqm-candidate-v1"
MODEL_ID = "IMCC-SF04-D066"
SOURCE_KEY = "WU1993-MQM"
SOURCE_CITATION = (
    "Wu, P., Eriksson, G., Pelton, A.D. & Blander, M. (1993), "
    "ISIJ International 33, 26–35, DOI 10.2355/isijinternational.33.26; "
    "Eqs. (7)–(8), p. 30."
)
INACTIVE_REASON = (
    "inactive because the separate Wu 1993 free-pool MQM Gibbs term replaces "
    "these two binary Mg–Si associations in this candidate"
)
MQM_BLOCK = {
    "model": "wu-1993-pair-approximation",
    "free_pool": ["MgO", "SiO2"],
    "coordination_numbers": {"MgO": 1.37744375, "SiO2": 2.7548875},
    "omega_J_mol": {
        "constant": -86090.0,
        "Y_Si": -48974.0,
        "Y_Si^7": 328109.0,
    },
    "eta_J_mol_K": {"Y_Si": -37.656, "Y_Si^7": 125.52},
    "T_range_K": [1200.0, 3000.0],
    "replaced_reactions": ["Mg2SiO4", "MgSiO3"],
    "standard_state_conversion": (
        "none in the embedded Gibbs function; Wu-to-OpenIMCC conversion is "
        "applied only by evaluate_mgo_sio2_mqm"
    ),
    "citation": SOURCE_CITATION,
}


def build() -> dict:
    pack = json.loads(BASE.read_text(encoding="utf-8"))
    if pack.get("imcc_sf04_datapack_version") != "1.0.6-d066-v4":
        raise ValueError("unexpected default D066 base pack version")
    if pack.get("model_id") != MODEL_ID or len(pack.get("rows", [])) != 38:
        raise ValueError("unexpected D066 base pack identity or row count")

    core_row_digest, standards_digest = _d066_core_content_digests(pack)
    for row in pack["rows"]:
        if row["complex"] in {"Mg2SiO4", "MgSiO3"}:
            row["active"] = False
            row["inactive_reason"] = INACTIVE_REASON

    pack["created"] = "2026-11-26"
    pack["imcc_sf04_datapack_version"] = VERSION
    pack["model"] = (
        "Opt-in fixed Wu 1993 MgO–SiO2 free-pool MQM candidate; "
        "Mg2SiO4 and MgSiO3 are inactive; not the default."
    )
    pack["pack_arithmetic"] = (
        "D066 v4 coefficients and liquid standards are retained; Mg2SiO4 and "
        "MgSiO3 are inactive and the separate fixed Wu 1993 free-pool Gibbs "
        "block supplies binary MgO–SiO2 mixing."
    )
    pack["sources"][SOURCE_KEY] = {
        "citation": SOURCE_CITATION,
        "doi": "10.2355/isijinternational.33.26",
    }
    pack["mqm_candidate"] = copy.deepcopy(MQM_BLOCK)
    pack["candidate_provenance"] = {
        "status": "opt-in fixed-parameter candidate; not the default",
        "basis": (
            "D066 v4 with Mg2SiO4 and MgSiO3 inactive and the fixed Wu 1993 "
            "pair-approximation Gibbs function added on the free MgO/SiO2 pool."
        ),
        "parameter_status": "published values transcribed; no fitting or refitting",
        "d066_core_content": {
            "rows_sha256": core_row_digest,
            "common_parent_liquid_standards_sha256": standards_digest,
            "basis": (
                "All D066 v4 row coefficients, parent standards, and retained "
                "associates are carried unchanged; only the two binary Mg–Si "
                "association rows are inactive."
            ),
        },
        "inactive_rows": {
            "Mg2SiO4": INACTIVE_REASON,
            "MgSiO3": INACTIVE_REASON,
        },
        "retained_associates": [
            "CaMgSi2O6",
            "Ca2MgSi2O7",
            "Mg2Al4Si5O18",
        ],
        "third_oxide_treatment": (
            "The binary MQM acts on the free MgO/SiO2 subsystem and does not "
            "apply Wu's Toop extension."
        ),
        "source": SOURCE_CITATION,
    }
    pack.setdefault("errata", []).append({
        "version": VERSION,
        "change": (
            "Adds an opt-in fixed Wu 1993 MgO–SiO2 free-pool MQM candidate; "
            "the default pack is unchanged."
        ),
    })
    # Wu's pair contribution uses Δg=ω−ηT directly. By the envelope theorem,
    # d(N_pair*q*Δg)/dT holds the minimized q fixed, so this term contributes
    # ΔH=N_pair*q*ω and ΔS=N_pair*q*η; ω is J/mol of pair reaction and η is
    # J/(mol K). This is not an A+B/T reaction-row rewrite, so no ΔH/ΔS row
    # refit or parent-standard conversion is introduced.
    return pack


def write_or_check_pack(*, write: bool) -> int:
    generated = json.dumps(build(), indent=2, ensure_ascii=False) + "\n"
    if write:
        DESTINATION.write_text(generated, encoding="utf-8")
        print(f"wrote {DESTINATION.relative_to(ROOT)}")
        return 0
    if not DESTINATION.exists() or DESTINATION.read_text(encoding="utf-8") != generated:
        print(f"{DESTINATION.relative_to(ROOT)} is stale; run with --write")
        return 1
    print(f"{DESTINATION.relative_to(ROOT)} is current")
    return 0


def write_or_check_liquid_standards(*, write: bool) -> int:
    digest = load_datapack(DESTINATION).binding_digest
    base_digest = load_datapack(BASE).binding_digest
    if not digest or not base_digest:
        print("candidate or D066 v4 pack has no binding digest")
        return 1
    registry = json.loads(LIQUID_STANDARDS.read_text(encoding="utf-8"))
    entries = registry.get("packs")
    if not isinstance(entries, dict) or not isinstance(entries.get(base_digest), dict):
        print("D066 v4 has no liquid-standards declaration")
        return 1

    declaration = copy.deepcopy(entries[base_digest])
    declaration["file"] = DESTINATION.name
    declaration["model_id"] = MODEL_ID
    declaration["version"] = VERSION
    declaration["standard_citations"] = {
        oxide: citation.replace(BASE.name, DESTINATION.name)
        for oxide, citation in declaration["standard_citations"].items()
    }
    if write:
        for stale_digest, entry in list(entries.items()):
            if isinstance(entry, dict) and entry.get("file") == DESTINATION.name:
                entries.pop(stale_digest)
        entries[digest] = declaration
        LIQUID_STANDARDS.write_text(
            json.dumps(registry, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"registered liquid standards for binding digest {digest}")
        return 0
    if entries.get(digest) != declaration:
        print(f"liquid-standards declaration is stale for {DESTINATION.name}")
        return 1
    print(f"liquid-standards declaration is current for {DESTINATION.name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if write_or_check_pack(write=args.write):
        return 1
    return write_or_check_liquid_standards(write=args.write)


if __name__ == "__main__":
    raise SystemExit(main())
