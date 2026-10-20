from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from openimcc import evaluate_redox, load_datapack
from openimcc.model import ImccMalformedDatapackError


ROOT = Path(__file__).resolve().parents[1]
PACKS = ROOT / "src/openimcc/data/packs"
FEO_REDOX = PACKS / "openimcc-redox-feo-liquid-candidate-v2.json"
FEO_MELT = PACKS / "imcc-sf04-d066-feo-liquid-candidate-v2.json"


def _bse_feed() -> dict[str, float]:
    oxide_wt = {
        "SiO2": 46.53,
        "MgO": 38.05,
        "FeO": 8.44,
        "Al2O3": 4.37,
        "CaO": 2.06,
    }
    molar_mass = {
        "SiO2": 60.0843,
        "MgO": 40.3044,
        "FeO": 71.844,
        "Al2O3": 101.9613,
        "CaO": 56.0774,
    }
    cation = {
        "SiO2": ("Si", 1),
        "MgO": ("Mg", 1),
        "FeO": ("Fe", 1),
        "Al2O3": ("Al", 2),
        "CaO": ("Ca", 1),
    }
    feed = {element: 0.0 for element in ("Si", "Mg", "Fe", "Al", "Ca")}
    for oxide, amount in oxide_wt.items():
        element, count = cation[oxide]
        feed[element] += count * amount / molar_mass[oxide]
    feed["O"] = 0.0
    return feed


def test_none_melt_pack_path_preserves_default_result() -> None:
    feed = _bse_feed()
    default = evaluate_redox(
        feed, 2173.0, "imposed", -5.97 * math.log(10.0)
    )
    explicit_none = evaluate_redox(
        feed, 2173.0, "imposed", -5.97 * math.log(10.0), melt_pack_path=None
    )

    assert explicit_none == default
    assert explicit_none.melt_pack_binding_digest == default.melt_pack_binding_digest


def test_feo_melt_candidate_is_loaded_and_reported() -> None:
    feed = _bse_feed()
    default = evaluate_redox(
        feed, 2173.0, "imposed", -5.97 * math.log(10.0)
    )
    candidate = evaluate_redox(
        feed,
        2173.0,
        "imposed",
        -5.97 * math.log(10.0),
        redox_pack_path=FEO_REDOX,
        melt_pack_path=FEO_MELT,
    )
    melt_pack = load_datapack(FEO_MELT)

    assert candidate.melt_pack_binding_digest == melt_pack.binding_digest
    assert candidate.melt_pack_binding_digest != default.melt_pack_binding_digest
    assert candidate.species_moles != default.species_moles


@pytest.mark.parametrize("bad_path_kind", ["missing", "mismatched"])
def test_bad_feo_melt_candidate_path_raises_typed_error(
    tmp_path: Path, bad_path_kind: str
) -> None:
    if bad_path_kind == "missing":
        bad_path = tmp_path / "missing.json"
    else:
        content = json.loads(FEO_MELT.read_text(encoding="utf-8"))
        content["rows"].pop(0)
        bad_path = tmp_path / "mismatched.json"
        bad_path.write_text(json.dumps(content), encoding="utf-8")

    with pytest.raises(ImccMalformedDatapackError):
        evaluate_redox(
            _bse_feed(),
            2173.0,
            "imposed",
            -5.97 * math.log(10.0),
            melt_pack_path=bad_path,
        )
