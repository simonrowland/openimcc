#!/usr/bin/env python3
"""Rebuild the Barin aluminate fusion nodes on the common absolute-G basis."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "src/openimcc/data/diagnostics/barin-aluminate-crystals.json"
BARIN_EXTRACT = "data/literature/extracts/barin-1995.yaml"


def _read_barin_extract(source_repository: Path, source_commit: str) -> dict:
    raw = subprocess.run(
        [
            "git", "-C", str(source_repository), "show",
            f"{source_commit}:{BARIN_EXTRACT}",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    import yaml

    return yaml.safe_load(raw)


def _barin_formation_enthalpy(
    extract: dict, formula: str,
) -> tuple[float, dict]:
    species = extract["species"].get(formula)
    if species is None:
        raise ValueError(f"Barin extract has no species {formula}")
    anchors = []
    for observation in species["observations"]:
        values = observation.get("values", {})
        if values.get("quantity") != "delta_fH":
            continue
        point = next(
            (point for point in values.get("points", [])
             if float(point["T_K"]) == 298.15),
            None,
        )
        if point is not None:
            anchors.append((float(point["value"]), point["locator"]))
    if len(anchors) != 1:
        raise ValueError(
            f"expected one 298.15 K Barin formation-enthalpy anchor for "
            f"{formula}, found {len(anchors)}"
        )
    value, locator = anchors[0]
    return value, {
        "page": locator["page"],
        "published_page": locator["published_page"],
        "pdf_page_index": locator["pdf_page_index"],
        "row": locator["row"],
        "column": locator["column"],
    }


def _crystal_nodes(phase: dict) -> dict[float, tuple[float, float]]:
    """Return Barin H−H298 and S at each source temperature."""
    values: dict[float, tuple[float, float]] = {}
    for polymorph in phase["polymorphs"]:
        series = polymorph["source_series"]
        enthalpy = {
            float(node["T_K"]): float(node["value"])
            for node in series["H_minus_H298"]["nodes"]
        }
        entropy = {
            float(node["T_K"]): float(node["value"])
            for node in series["S"]["nodes"]
        }
        for node in series["delta_fG"]["nodes"]:
            temperature = float(node["T_K"])
            if temperature not in enthalpy or temperature not in entropy:
                raise ValueError(
                    f"{phase['id']} has no aligned H−H298/S source node at "
                    f"{temperature:g} K"
                )
            # The later polymorph owns a shared transition temperature.
            values[temperature] = (enthalpy[temperature], entropy[temperature])
    return values


def build(data: dict, source_repository: Path | None = None) -> dict:
    """Return a v3 artifact rebuilt from Barin and packaged JANAF H/S data.

    Premise: subtracting formation Gibbs functions from JANAF and Barin keeps
    their different element-reference functions in the fusion node. Algebra:
    form each absolute species G as Hf(298.15)+[H(T)-H(298.15)]-T*S(T)/1000,
    then compute ΔG_fus=ΣνG_abs,JANAF(parent,l)-G_abs,Barin(crystal). The
    formation enthalpy and increment are kJ/mol; T is K and S is J/(mol K),
    so T*S/1000 is kJ/mol and the result is kJ/mol crystal formula. Sanity:
    v3−v2 is the source element-reference correction (about −19.63 kJ/mol for
    C12A7 at 1700 K and −0.025 kJ/mol for CA at 1800 K).
    """
    sys.path.insert(0, str(ROOT / "src"))
    from openimcc.janaf_crystals import evaluate_janaf_apparent_gibbs

    result = json.loads(json.dumps(data))
    result["schema"] = "openimcc-barin-aluminate-crystals-v3"
    conversion = result["conversion"]
    conversion["method"] = (
        "Absolute-G Hess cycle from Barin crystal and JANAF liquid-parent "
        "formation enthalpy, enthalpy increment, and entropy"
    )
    conversion["equation"] = (
        "G_abs,s(T) = delta_fH_s(298.15) + [H_s(T)-H_s(298.15)] "
        "- T*S_s(T)/1000; "
        "deltaG_fus,abs = sum(nu_i * G_abs,JANAF(parent_oxide_i,l)) "
        "- G_abs,Barin(crystal)"
    )
    conversion["unit_check"] = (
        "Formation and enthalpy-increment values are kJ/mol; S is J/(mol K), "
        "so T*S/1000 is kJ/mol. Stoichiometric sums and fusion nodes are "
        "kJ/mol crystal formula."
    )
    parent_rows = {
        row["oxide"]: row for row in json.loads(
            (ROOT / "src/openimcc/data/diagnostics/"
             "janaf-pure-oxide-liquid-pairs.json").read_text(encoding="utf-8")
        )["rows"]
    }
    for oxide, reference in conversion["parent_oxide_references"].items():
        pair = parent_rows[oxide]
        if pair["table_id"] != reference["table_id"]:
            raise ValueError(f"unexpected JANAF pair for {oxide}")
        reference["quantity"] = "absolute_gibbs_energy"
        reference["unit"] = "kJ/mol oxide formula"
        reference["interpolation"] = (
            "linear between source enthalpy-increment and entropy nodes "
            "separately, including transition rows"
        )

    extract = (
        _read_barin_extract(source_repository, result["source"]["source_commit"])
        if source_repository is not None else None
    )
    for phase in result["phases"]:
        if extract is not None:
            h298, locator = _barin_formation_enthalpy(
                extract,
                phase["formula"],
            )
            phase["formation_enthalpy_298_kJ_mol"] = h298
            phase["source"]["formation_enthalpy_locator"] = locator
        if "formation_enthalpy_298_kJ_mol" not in phase:
            raise ValueError(
                f"{phase['id']} is missing Barin formation enthalpy at 298.15 K"
            )
        h298 = float(phase["formation_enthalpy_298_kJ_mol"])
        source_nodes = _crystal_nodes(phase)
        old_fusion_nodes = phase["fusion_gibbs_nodes_kJ_mol"]
        rebuilt = {}
        for key in phase["fusion_gibbs_nodes_kJ_mol"]:
            temperature = float(key)
            if temperature not in source_nodes:
                raise ValueError(
                    f"{phase['id']} has no Barin source node at {temperature:g} K"
                )
            h_increment, entropy = source_nodes[temperature]
            crystal_g = h298 + h_increment - temperature * entropy / 1000.0
            parent_g_j = math.fsum(
                float(coefficient) * evaluate_janaf_apparent_gibbs(
                    conversion["parent_oxide_references"][oxide]["table_id"],
                    temperature,
                ).G_J_mol
                for oxide, coefficient in phase["oxide_stoichiometry"].items()
            )
            rebuilt[key] = parent_g_j / 1000.0 - crystal_g
        if phase["id"] == "CA":
            h298_key = "298.15"
            if h298_key not in rebuilt:
                raise ValueError("CA has no 298.15 K fusion node")
            old_h298 = float(old_fusion_nodes[h298_key])
            if abs(rebuilt[h298_key] - old_h298) > 0.1:
                raise ValueError(
                    "CA absolute-G Hess value differs from its v2 298.15 K "
                    f"node by {rebuilt[h298_key] - old_h298:.6f} kJ/mol"
                )
        phase["fusion_gibbs_nodes_kJ_mol"] = rebuilt
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    parser.add_argument(
        "--source-repository",
        type=Path,
        help="checkout containing the Barin extract at the recorded source commit",
    )
    args = parser.parse_args()

    original = DATA.read_text(encoding="utf-8")
    if args.write and args.source_repository is None:
        parser.error("--write requires --source-repository")
    rebuilt = json.dumps(
        build(json.loads(original), args.source_repository if args.write else None),
        indent=2,
        ensure_ascii=False,
    ) + "\n"
    if args.check:
        if original != rebuilt:
            print("Barin aluminate fusion nodes are stale", file=sys.stderr)
            return 1
        print("Barin aluminate fusion nodes are current")
        return 0
    DATA.write_text(rebuilt, encoding="utf-8")
    print("Rebuilt Barin aluminate fusion nodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
