#!/usr/bin/env python3
"""Build the opt-in CaO–Al2O3 associate candidate pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from candidate_pack import add_write_check_arguments, write_or_check_candidate_pack
from openimcc.model import load_datapack
from openimcc.standards import pack_liquid_standards


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "src/openimcc/data/packs/imcc-sf04-d066-v4.json"
DESTINATION = ROOT / "src/openimcc/data/packs/imcc-sf04-d066-cao-aluminate-candidate-v1.json"
LIQUID_STANDARDS = ROOT / "src/openimcc/data/liquid-standards.json"

# The shared A-shift fraction minimizes equal-source mean-squared dex residuals
# for 14 Allibert binary activities and 345 exposed Stolyarova 1996 rows. For
# factor f in [0, 1], ΔA=f*ΔA_Allibert; each source contributes half the loss,
# regardless of row count. The 0.05 grid is refined by golden-section search.
# Residuals have units dex, so the objective has units dex². At f=0.889630787,
# ΔA=-1.125561024 and J=0.438645513 dex²; B stays at D066 v4 because these
# near-isothermal sources do not identify an independent temperature slope.
UNSHRUNK_ALLIBERT_DELTA_A = -1.2652001714747003
TRANSFER_FACTOR = 0.8896307870036171
SHARED_DELTA_A = UNSHRUNK_ALLIBERT_DELTA_A * TRANSFER_FACTOR
OBJECTIVE_GRID_CURVE = [
    [0.00, 0.4841562962255543], [0.05, 0.47701470911210114],
    [0.10, 0.47069829076469855], [0.15, 0.46515953264180576],
    [0.20, 0.4603418209636342], [0.25, 0.45618592569617633],
    [0.30, 0.4526341421925847], [0.35, 0.4496310260391191],
    [0.40, 0.44712212580878147], [0.45, 0.44505300571426265],
    [0.50, 0.44336979506171126], [0.55, 0.44202091232373353],
    [0.60, 0.44095892106188905], [0.65, 0.44014177068156424],
    [0.70, 0.43953324560318285], [0.75, 0.4391027993100521],
    [0.80, 0.4388250349114341], [0.85, 0.43867904145399467],
    [0.90, 0.4386477139043316], [0.95, 0.4387171206517072],
    [1.00, 0.4388759431609286],
]
REFIT_A = {
    "CaAl4O7": -0.59 + SHARED_DELTA_A,
    "CaAl12O19": -3.79 + SHARED_DELTA_A,
}


def build() -> dict:
    pack = json.loads(BASE.read_text(encoding="utf-8"))
    pack["imcc_sf04_datapack_version"] = "1.0.0-d066-cao-aluminate-candidate-v1"
    pack["created"] = "2026-11-25"
    pack["model"] = (
        "Opt-in D066 v4 candidate with two constrained associate refits in "
        "the CaO–Al2O3 family; all other rows are unchanged."
    )
    pack["pack_arithmetic"] = (
        "38 D066 v4 rows; only CaAl4O7 and CaAl12O19 A intercepts are refitted."
    )
    pack.setdefault("sources", {})["ALLIBERT-1981"] = {
        "citation": "Allibert et al. (1981), CaO–Al2O3 liquid activities, DOI 10.1111/j.1151-2916.1981.tb09608.x",
        "doi": "10.1111/j.1151-2916.1981.tb09608.x",
        "locator": "Binary parent activities at 2060 K; used as exposed fit data.",
    }
    for row in pack["rows"]:
        if row["complex"] in REFIT_A:
            row["A"] = REFIT_A[row["complex"]]
    # Premise: log10(K)=A+B/T, ΔG°=−R*T*ln(K), and ΔG°=ΔH°−T*ΔS°.
    # Algebra: ΔG°=−R*ln(10)*(A*T+B)=(-R*ln(10)*B)-T*(R*ln(10)*A),
    # so ΔH°=−R*ln(10)*B and ΔS°=+R*ln(10)*A. Units: B is K, A is
    # dimensionless, and R is J/(mol K), giving J/mol for ΔH° and J/(mol K)
    # for ΔS°. Sanity: CaAl4O7 at 2060 K has direct ΔG°=−118294.391 J/mol;
    # ΔH°−T*ΔS°=−185953.031 J/mol−2060 K*(−32.844000 J/(mol K)) gives the
    # same −118294.391 J/mol.
    # Holding B fixed therefore preserves inherited enthalpy and changes only
    # the implied entropy.
    pack["candidate_fit"] = {
        "data": "Allibert et al. (1981), 14 exposed binary activities; Stolyarova et al. (1996), 345 exposed numeric rows across melt and gas rails",
        "method": "bounded shared A-shift fit minimizing equal-source mean-squared dex residuals",
        "objective": "0.5 * (mean(Allibert residual_dex^2) + mean(Stolyarova 1996 residual_dex^2))",
        "objective_units": "dex^2",
        "objective_source_counts": {"Allibert": 14, "Stolyarova 1996": 345},
        "transfer_factor_domain": [0.0, 1.0],
        "transfer_factor_interpretation": "fraction of the Allibert shared A correction; the bounded range retains a nonnegative share of the binary correction",
        "transfer_factor_grid_step": 0.05,
        "refinement_method": "golden-section search in the neighboring 0.05-grid interval",
        "objective_grid_curve": OBJECTIVE_GRID_CURVE,
        "objective_at_optimum": 0.43864551293782933,
        "temperature_K": 2060,
        "temperature_slope_policy": "B held at D066 v4 because the exposed sources are near-isothermal and do not justify an independent temperature slope",
        "unshrunk_allibert_delta_A": UNSHRUNK_ALLIBERT_DELTA_A,
        "stolyarova_transfer_factor": TRANSFER_FACTOR,
        "shared_delta_A": SHARED_DELTA_A,
        "temperature_transfer_assumption": (
            "The constrained shared A shift is held constant across the "
            "inherited 1700-3000 K row domain; exposed activity data do not "
            "identify a temperature dependence."
        ),
        "changed_parameters": {
            name: {"A": REFIT_A[name], "B": next(row["B"] for row in pack["rows"] if row["complex"] == name)}
            for name in REFIT_A
        },
    }
    return pack


def write_or_check_liquid_standards(*, write: bool) -> int:
    loaded = load_datapack(DESTINATION)
    digest = loaded.binding_digest
    if not digest:
        print("candidate pack has no binding digest")
        return 1

    registry = json.loads(LIQUID_STANDARDS.read_text(encoding="utf-8"))
    entries = registry.get("packs")
    if not isinstance(entries, dict):
        print("liquid-standards registry has no packs object")
        return 1
    matching = entries.get(digest)
    same_file = [
        (key, value)
        for key, value in entries.items()
        if isinstance(value, dict) and value.get("file") == DESTINATION.name
    ]

    if write:
        if not isinstance(matching, dict):
            if not same_file:
                print(f"no liquid-standards declaration found for {DESTINATION.name}")
                return 1
            matching = same_file[0][1]
        for stale_digest, _ in same_file:
            entries.pop(stale_digest, None)
        entries[digest] = matching
        LIQUID_STANDARDS.write_text(
            json.dumps(registry, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"registered liquid standards for binding digest {digest}")
    elif not isinstance(matching, dict):
        print(
            f"liquid-standards declaration missing for {DESTINATION.name} "
            f"binding digest {digest}"
        )
        return 1

    declared = pack_liquid_standards(DESTINATION)
    if declared.source == "undeclared" or not (
        declared.standards or declared.row_overrides
    ):
        print(
            f"liquid-standards declaration is empty for {DESTINATION.name} "
            f"binding digest {digest}"
        )
        return 1
    print(f"liquid-standards declaration is current for {DESTINATION.name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_write_check_arguments(parser)
    args = parser.parse_args()
    generated = json.dumps(build(), indent=2, ensure_ascii=False) + "\n"
    pack_status = write_or_check_candidate_pack(
        DESTINATION, generated, write=args.write
    )
    if pack_status:
        return pack_status
    return write_or_check_liquid_standards(write=args.write)


if __name__ == "__main__":
    raise SystemExit(main())
