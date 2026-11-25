"""Opt-in CaO–Al2O3 activity-fit pack coverage."""

from __future__ import annotations

import json
import math
from pathlib import Path

from openimcc import evaluate, load_profile
from openimcc._constants import R_J_MOL_K


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "src/openimcc/data/packs/imcc-sf04-d066-v4.json"
CANDIDATE = (
    ROOT / "src/openimcc/data/packs/imcc-sf04-d066-cao-aluminate-candidate-v1.json"
)


def test_candidate_is_opt_in_and_changes_only_two_aluminate_intercepts() -> None:
    base_rows = json.loads(BASE.read_text(encoding="utf-8"))["rows"]
    candidate_rows = json.loads(CANDIDATE.read_text(encoding="utf-8"))["rows"]
    changed = []
    for base_row, candidate_row in zip(base_rows, candidate_rows, strict=True):
        restored = dict(candidate_row)
        if candidate_row["complex"] in {"CaAl4O7", "CaAl12O19"}:
            restored["A"] = base_row["A"]
            changed.append(candidate_row["complex"])
        assert restored == base_row
    assert changed == ["CaAl4O7", "CaAl12O19"]

    default = load_profile("openimcc")
    candidate = load_profile("cao-aluminate-candidate")
    assert default.melt_pack.version == "1.0.6-d066-v4"
    assert candidate.melt_pack.version == "1.0.0-d066-cao-aluminate-candidate-v1"


def test_candidate_profile_applies_to_binary_activity_solve() -> None:
    composition = {"CaO": 0.438, "Al2O3": 0.562}
    default = evaluate(composition, 2060.0, profile="openimcc")
    candidate = evaluate(composition, 2060.0, profile="cao-aluminate-candidate")
    assert candidate.parent_oxides == default.parent_oxides
    assert candidate.parent_activity[3] != default.parent_activity[3]
    assert candidate.parent_activity[4] != default.parent_activity[4]


def test_candidate_association_entropy_sign_reproduces_gibbs_energy() -> None:
    pack = json.loads(CANDIDATE.read_text(encoding="utf-8"))
    temperature = 2060.0
    for row in pack["rows"]:
        if row["complex"] not in {"CaAl4O7", "CaAl12O19"}:
            continue
        delta_h = -R_J_MOL_K * math.log(10.0) * float(row["B"])
        delta_s = R_J_MOL_K * math.log(10.0) * float(row["A"])
        delta_g_from_log_k = -R_J_MOL_K * temperature * math.log(10.0) * (
            float(row["A"]) + float(row["B"]) / temperature
        )

        assert delta_s < 0.0
        assert math.isclose(
            delta_h - temperature * delta_s,
            delta_g_from_log_k,
            rel_tol=1e-12,
            abs_tol=1e-8,
        )
