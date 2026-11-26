from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

import openimcc.standards as standards_module
from openimcc.model import (
    _d066_core_liquid_standards_digest,
    evaluate,
    load_datapack,
)
from openimcc.profiles import load_profile
from openimcc.redox_pack import RedoxInputError


PACK_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/openimcc/data/packs/imcc-sf04-d066-v4.json"
)


def _mno_composition() -> dict[str, float]:
    return {
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


def _synthetic_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    change_core_row: bool = False,
    change_parent_standard: bool = False,
):
    content = json.loads(PACK_PATH.read_text(encoding="utf-8"))
    content["imcc_sf04_datapack_version"] = "1.0.6-d066-test-candidate"
    if change_core_row:
        row = next(row for row in content["rows"] if row["complex"] == "Mg2SiO4")
        row["A"] = float(row["A"]) + 0.5

    candidate_path = tmp_path / "melt-candidate.json"
    candidate_path.write_text(json.dumps(content), encoding="utf-8")
    candidate = load_datapack(candidate_path)

    registry = copy.deepcopy(standards_module._read_registry())
    default_pack = load_datapack()
    declaration = copy.deepcopy(registry["packs"][default_pack.binding_digest])
    if change_parent_standard:
        declaration["standards"]["SiO2"] = "test alternate SiO2 standard"
    registry["packs"][candidate.binding_digest] = declaration
    monkeypatch.setattr(standards_module, "_read_registry", lambda: registry)
    return candidate


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


def test_changed_core_row_candidate_grafts_and_solves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    d066_v4 = load_datapack()
    candidate = _synthetic_candidate(
        tmp_path, monkeypatch, change_core_row=True
    )

    assert candidate.parent_oxides == d066_v4.parent_oxides
    assert candidate._d066_core_row_digest != d066_v4._d066_core_row_digest
    assert (
        _d066_core_liquid_standards_digest(candidate)
        == _d066_core_liquid_standards_digest(d066_v4)
    )

    result = evaluate(
        _mno_composition(),
        1800.0,
        pack=candidate,
        basis_type="wt",
        allow_extrapolation=True,
        allow_out_of_envelope=True,
    )

    assert np.isfinite(result.activity("MnO"))
    assert {"MnSiO3(l)", "Mn2SiO4(l)", "MnAl2O4(l)"} <= set(
        result.species_names
    )

    # The solve must use the candidate's raised Mg2SiO4 log10 K, not the
    # default core coefficients: a larger association constant at fixed bulk
    # composition gives a larger Mg2SiO4 fraction than the default v4 graft.
    default = evaluate(
        _mno_composition(),
        1800.0,
        pack=d066_v4,
        basis_type="wt",
        allow_extrapolation=True,
        allow_out_of_envelope=True,
    )
    index = result.species_names.index("Mg2SiO4")
    default_index = default.species_names.index("Mg2SiO4")
    assert result.species_x[index] > 1.05 * default.species_x[default_index]


def test_changed_parent_liquid_standard_is_still_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    d066_v4 = load_datapack()
    candidate = _synthetic_candidate(
        tmp_path, monkeypatch, change_parent_standard=True
    )

    assert candidate.parent_oxides == d066_v4.parent_oxides
    assert candidate._d066_core_row_digest == d066_v4._d066_core_row_digest
    assert (
        _d066_core_liquid_standards_digest(candidate)
        != _d066_core_liquid_standards_digest(d066_v4)
    )
    with pytest.raises(
        RedoxInputError,
        match="matching existing-parent liquid standards",
    ):
        evaluate(
            _mno_composition(),
            1800.0,
            pack=candidate,
            basis_type="wt",
            allow_extrapolation=True,
            allow_out_of_envelope=True,
        )


def test_mgsio_candidate_mno_graft_does_not_require_mgsio3() -> None:
    d066 = load_datapack()
    profile = load_profile("mgsio-candidate")
    candidate = profile.melt_pack

    assert candidate.parent_oxides == d066.parent_oxides
    assert (
        _d066_core_liquid_standards_digest(candidate)
        == _d066_core_liquid_standards_digest(d066)
    )
    result = evaluate(
        _mno_composition(),
        1800.0,
        profile=profile,
        basis_type="wt",
        allow_extrapolation=True,
        allow_out_of_envelope=True,
    )

    assert np.isfinite(result.activity("MnO"))
    assert "MgSiO3" not in result.species_names
    assert {"MnSiO3(l)", "Mn2SiO4(l)", "MnAl2O4(l)"} <= set(
        result.species_names
    )
