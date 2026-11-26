from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from openimcc import ImccMalformedDatapackError, evaluate, load_datapack, load_profile


PACK_DIR = Path(__file__).resolve().parents[1] / "src/openimcc/data/packs"
CANDIDATE_PATH = PACK_DIR / "imcc-sf04-d066-mgsio-candidate-v1.json"
DEFAULT_PATH = PACK_DIR / "imcc-sf04-d066-v4.json"
INACTIVE_REASON = (
    "removed by the source-balanced fit; the MgO–SiO2 activity data require "
    "a minimal MgO/SiO2/Mg2SiO4 description"
)


def test_mgsio_candidate_loads_and_solves_with_mgsio3_inactive() -> None:
    profile = load_profile("mgsio-candidate")
    pack = profile.melt_pack
    result = evaluate(
        {"MgO": 50.0, "SiO2": 50.0},
        1873.0,
        profile=profile,
        basis_type="mol",
    )

    inactive = dict(pack.inactive_rows)
    assert inactive["MgSiO3"] == INACTIVE_REASON
    assert inactive["KCaAlSi2O7"] == "E25: out of liquid domain; owner-confirmed"
    assert "MgSiO3" not in pack.kernel_datapack.reactions
    assert "Mg2SiO4" in pack.kernel_datapack.reactions
    assert np.isfinite(result.activity("MgO"))
    assert np.isfinite(result.activity("SiO2"))
    assert "MgSiO3" not in result.species_names


def test_mgsio_candidate_requires_a_reason_for_inactive_row(tmp_path: Path) -> None:
    content = json.loads(CANDIDATE_PATH.read_text(encoding="utf-8"))
    row = next(row for row in content["rows"] if row["complex"] == "MgSiO3")
    row.pop("inactive_reason")
    path = tmp_path / "mgsio-candidate-missing-reason.json"
    path.write_text(json.dumps(content), encoding="utf-8")

    with pytest.raises(
        ImccMalformedDatapackError,
        match="inactive_reason must be a non-empty string",
    ):
        load_datapack(path)


def test_default_d066_still_refuses_an_additional_inactive_row(
    tmp_path: Path,
) -> None:
    content = json.loads(DEFAULT_PATH.read_text(encoding="utf-8"))
    row = next(row for row in content["rows"] if row["complex"] == "CaSiO3")
    row["active"] = False
    row["inactive_reason"] = "test-only extra inactive core row"
    path = tmp_path / DEFAULT_PATH.name
    path.write_text(json.dumps(content), encoding="utf-8")

    with pytest.raises(
        ImccMalformedDatapackError,
        match="D066 row 'CaSiO3' cannot be inactive",
    ):
        load_datapack(path)


def test_renamed_default_d066_still_refuses_an_additional_inactive_row(
    tmp_path: Path,
) -> None:
    content = json.loads(DEFAULT_PATH.read_text(encoding="utf-8"))
    row = next(row for row in content["rows"] if row["complex"] == "CaSiO3")
    row["active"] = False
    row["inactive_reason"] = "test-only extra inactive core row"
    path = tmp_path / "renamed-default.json"
    path.write_text(json.dumps(content), encoding="utf-8")

    with pytest.raises(
        ImccMalformedDatapackError,
        match="D066 row 'CaSiO3' cannot be inactive",
    ):
        load_datapack(path)


def test_candidate_version_without_provenance_refuses_extra_inactive_row(
    tmp_path: Path,
) -> None:
    content = json.loads(CANDIDATE_PATH.read_text(encoding="utf-8"))
    content.pop("candidate_provenance")
    path = tmp_path / "mgsio-candidate-without-provenance.json"
    path.write_text(json.dumps(content), encoding="utf-8")

    with pytest.raises(
        ImccMalformedDatapackError,
        match="D066 row 'MgSiO3' cannot be inactive",
    ):
        load_datapack(path)
