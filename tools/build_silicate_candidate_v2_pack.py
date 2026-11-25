#!/usr/bin/env python3
"""Build the opt-in D066 Mg-silicate activity-fit candidate pack."""

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
DESTINATION = PACKS / "imcc-sf04-d066-silicate-candidate-v2.json"
LIQUID_STANDARDS = ROOT / "src/openimcc/data/liquid-standards.json"
VERSION = "1.0.0-d066-silicate-candidate-v2"
MODEL_ID = "IMCC-SF04-D066"
FIT_SOURCE_KEYS = (
    "SILICATE-FIT-ARUTYUNYAN-2005",
    "SILICATE-FIT-KAMBAYASHI-KATO-1984",
    "SILICATE-FIT-KAMBAYASHI-1983",
    "SILICATE-FIT-KUME-2000",
)
TARGET_A = {
    "Mg2SiO4": -0.8648532598800563,
    "MgSiO3": 0.3506691491283032,
}
OLD_A = {
    "Mg2SiO4": -0.9093214922723483,
    "MgSiO3": 0.4506785077276513,
}


def build() -> dict:
    pack, original_rows = load_d066_pack(BASE)
    fit_sources = {
        "SILICATE-FIT-ARUTYUNYAN-2005": {
            "citation": "Arutyunyan et al. (2005), MgO-SiO2 liquid activities.",
            "scope": "a(MgO) guard and a(SiO2) FIT rows; exposed FIT observations.",
        },
        "SILICATE-FIT-KAMBAYASHI-KATO-1984": {
            "citation": "Kambayashi and Kato (1984), MgO-SiO2 liquid activities.",
            "scope": "a(MgO) guard and a(SiO2) FIT rows; exposed FIT observations.",
        },
        "SILICATE-FIT-KAMBAYASHI-1983": {
            "citation": "Kambayashi (1983), MgO-SiO2 liquid activities.",
            "scope": "a(MgO) guard and a(SiO2) FIT rows; exposed FIT observations.",
        },
        "SILICATE-FIT-KUME-2000": {
            "citation": "Kume et al. (2000), CaO-MgO-SiO2 liquid activities.",
            "scope": "paired a(MgO) guard and a(SiO2) FIT rows; exposed observations.",
        },
    }
    pack["sources"].update(fit_sources)
    for row in pack["rows"]:
        name = row["complex"]
        if name not in TARGET_A:
            continue
        if row["A"] != OLD_A[name]:
            raise ValueError(f"unexpected D066 v4 coefficient for {name}")
        row["A"] = TARGET_A[name]
        row["source_refs"] = [*row.get("source_refs", []), *FIT_SOURCE_KEYS]
        row["derivation"] += (
            " Candidate A is refit by constrained least squares on the exposed "
            "a(SiO2) activity rows; each selected a(MgO) residual is constrained "
            "to remain within 0.05 dex of D066 v4, and B remains unchanged."
        )
        row["candidate_activity_fit"] = {
            "parameter": "A",
            "provisional": True,
            "fit_source_refs": list(FIT_SOURCE_KEYS),
            "B_preserved_from_d066_v4": True,
            "objective": (
                "Unweighted sum of squared log10(predicted/measured) a(SiO2) "
                "residuals, subject to each selected a(MgO) residual changing "
                "by no more than 0.05 dex from D066 v4."
            ),
            "association_entropy_constraint_J_mol_K": [-50.0, 50.0],
        }

    # The selected activity observations use pure liquid oxide references.
    # For any solid-referenced comparison, equality of chemical potentials
    # gives a_l=a_cr*exp((G_cr-G_l)/(RT)); the log10 correction divides by
    # RT ln(10), with openimcc supplying G(l) and G(cr) on the same basis.
    # The fit objective is sum_i r_i^2 for silica, r_i=log10(a_model/a_measured);
    # the MgO per-row bound prevents the targeted silica correction from moving
    # a well-matched basic-oxide activity by more than 0.05 dex.
    row_digest, standards_digest = _d066_core_content_digests(pack)
    pack["imcc_sf04_datapack_version"] = VERSION
    pack["model_id"] = MODEL_ID
    pack["model"] = (
        "Opt-in constrained Mg-silicate activity-fit candidate retaining the "
        "D066 v4 parent basis and all non-target rows."
    )
    pack["pack_arithmetic"] = (
        "38 D066 v4 rows; only Mg2SiO4 and MgSiO3 A coefficients are refit; "
        "all B coefficients and both Ca-silicate rows are unchanged."
    )
    pack["candidate_provenance"] = {
        "status": "opt-in provisional candidate; not the default",
        "basis": "D066 v4 with constrained A refits for Mg2SiO4 and MgSiO3 only.",
        "fit_set": {
            "n": 118,
            "sources": list(FIT_SOURCE_KEYS),
            "source_roles": {key: "fitted" for key in FIT_SOURCE_KEYS},
            "temperature_range_K": [1873.0, 1973.0],
            "reference_state": "Pure liquid oxide standards; selected FIT activities are liquid-referenced.",
            "objective": (
                "Unweighted least squares on log10 predicted/measured a(SiO2) "
                "residuals, with each a(MgO) FIT residual constrained to change "
                "by at most 0.05 dex from D066 v4."
            ),
            "excluded_sources": [
                "Zaitsev 2006",
                "Shornikov and Archakov 2001",
                "Markova 1984",
                "Mendybaev 2013",
                "Stolyarova 1996 (Ca-only rows; Ca associates remain at D066 v4)",
            ],
        },
        "parameter_choice": {
            "free_parameters": "One A intercept for each of Mg2SiO4 and MgSiO3; both contribute to silica binding in MgO-bearing melts.",
            "fixed_parameters": "All B slopes remain at D066 v4 because the FIT range is only 1873–1973 K; Ca-silicate A and B values remain at D066 v4 to avoid their low-temperature CAS regression.",
            "constraint": "Each fitted a(MgO) residual changes by at most 0.05 dex relative to D066 v4; both association entropies remain within ±50 J mol^-1 K^-1.",
        },
        "d066_core_content": {
            "rows_sha256": row_digest,
            "common_parent_liquid_standards_sha256": standards_digest,
            "basis": "Digest covers the candidate rows, including the two requested coefficient changes; every other row object is copied unchanged from D066 v4.",
        },
        "standard_state_conversion": (
            "For equality G_l+RT ln(a_l)=G_cr+RT ln(a_cr), use "
            "a_l=a_cr exp((G_cr-G_l)/(RT)). The selected FIT rows are "
            "liquid-referenced. Solid-referenced validation values are converted "
            "with openimcc G(l)-G(cr) values on the same oxide basis."
        ),
        "fit_coefficients": {
            name: {"A": TARGET_A[name], "B": row["B"]}
            for name in TARGET_A
            for row in pack["rows"]
            if row["complex"] == name
        },
    }
    pack["errata"].append(
        {
            "version": VERSION,
            "change": "Adds an opt-in D066 v4 Mg-silicate activity-refit candidate; default and published packs remain unchanged.",
        }
    )
    if len(pack["rows"]) != len(original_rows):
        raise AssertionError("candidate row count differs from D066 v4")
    for before, after in zip(original_rows, pack["rows"]):
        if before["complex"] not in TARGET_A:
            if before != after:
                raise AssertionError(f"non-target row changed: {before['complex']}")
            continue
        expected = copy.deepcopy(before)
        expected["A"] = TARGET_A[before["complex"]]
        expected["source_refs"] = [*expected.get("source_refs", []), *FIT_SOURCE_KEYS]
        expected["derivation"] += (
            " Candidate A is refit by constrained least squares on the exposed "
            "a(SiO2) activity rows; each selected a(MgO) residual is constrained "
            "to remain within 0.05 dex of D066 v4, and B remains unchanged."
        )
        expected["candidate_activity_fit"] = after["candidate_activity_fit"]
        if expected != after:
            raise AssertionError(f"unexpected target-row change: {before['complex']}")
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
