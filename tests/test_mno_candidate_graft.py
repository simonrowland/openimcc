from __future__ import annotations

import pytest

from openimcc.model import (
    _d066_core_liquid_standards_digest,
    evaluate,
    load_datapack,
)
from openimcc.profiles import load_profile


@pytest.mark.parametrize("profile_name", ["sr-candidate", "p2o5-candidate"])
def test_candidate_core_rows_and_liquid_standards_match_d066_by_content(
    profile_name: str,
) -> None:
    d066_v4 = load_datapack()
    candidate = load_profile(profile_name).melt_pack

    assert candidate._d066_core_row_digest == d066_v4._d066_core_row_digest
    assert candidate._d066_core_row_digest is not None
    assert (
        _d066_core_liquid_standards_digest(candidate)
        == _d066_core_liquid_standards_digest(d066_v4)
    )


@pytest.mark.parametrize(
    ("profile_name", "added_parent"),
    [("sr-candidate", "SrO"), ("p2o5-candidate", "P2O5")],
)
def test_candidate_mno_graft_matches_d066_when_added_parent_is_zero(
    profile_name: str,
    added_parent: str,
) -> None:
    composition = {
        "SiO2": 50.8,
        "MgO": 9.34,
        "FeO": 10.4,
        "CaO": 9.6,
        "Al2O3": 15.0,
        "TiO2": 2.06,
        "Na2O": 0.77,
        "K2O": 0.61,
        "MnO": 0.16,
    }
    d066 = evaluate(composition, 1800.0, profile="openimcc", basis_type="wt")
    candidate_composition = {**composition, added_parent: 0.0}
    candidate = evaluate(
        candidate_composition,
        1800.0,
        profile=load_profile(profile_name),
        basis_type="wt",
    )

    assert candidate.activity("MnO") == d066.activity("MnO")
