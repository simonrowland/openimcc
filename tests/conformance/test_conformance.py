"""The contract between this package and anything that pins it.

WHY A SEPARATE SUITE. The unit tests check that the code is right. This suite
checks that the ANSWERS HAVE NOT MOVED. A consumer -- a simulator, a paper's
reproduction script -- pins (package version, datapack manifest hash) and runs
this. If a refactor, a scipy upgrade or an edited coefficient shifts an
activity, that consumer finds out here rather than by noticing months later
that its residuals drifted.

So a failure here is not automatically a bug. It is a question: was the change
intended? If yes, regenerate the goldens IN THE SAME COMMIT as the change and
say why in the message. Never regenerate to make a red suite green.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from openimcc import evaluate, load_datapack

GOLDEN = Path(__file__).parent / "golden-imcc-sf04-v1.0.2.json"
PACK = (
    Path(__file__).parent.parent.parent
    / "src" / "openimcc" / "data" / "packs" / "imcc-sf04-v1.0.2.json"
)

# Tight enough to catch a real coefficient or algorithm change, loose enough to
# survive BLAS/libm differences across platforms. The solve is a least-squares
# root find, so the last couple of digits are not portable.
RTOL = 1e-9


def _golden() -> dict:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def _cases() -> list[str]:
    return sorted(_golden()["cases"])


@pytest.fixture(scope="module")
def pack():
    return load_datapack(PACK)


@pytest.mark.parametrize("case_name", _cases())
def test_activities_have_not_moved(case_name, pack):
    case = _golden()["cases"][case_name]
    result = evaluate(
        case["composition"],
        case["temperature_K"],
        pack,
        basis_type=case["basis_type"],
    )

    assert [str(x) for x in result.parent_oxides] == case["parent_oxides"], (
        "the parent-oxide ORDER changed; every vector below is indexed by it, "
        "so a consumer unpacking by position would silently mis-assign"
    )
    assert result.D == pytest.approx(case["D"], rel=RTOL)
    for key in ("parent_activity", "parent_gamma"):
        got = [float(v) for v in getattr(result, key)]
        assert got == pytest.approx(case[key], rel=RTOL), f"{key} moved"


def test_pure_silica_is_an_exact_limiting_case(pack):
    """A check that needs no golden file to be believed.

    Premise: with one parent oxide present there is nothing for SiO2 to
    associate WITH -- every complex in the pack needs at least two distinct
    parents. So no complex can form.
    Consequence: the species vector is the parent itself, giving
    D = (moles of species) / (moles in) = 1 exactly, and the activity is the
    pure-liquid standard state, a = 1 exactly.
    This is not approximate: if either value drifts off 1.0 by more than
    solver noise, the speciation is producing complexes out of a single
    component and the model is wrong in a way no residual would reveal.
    """
    result = evaluate({"SiO2": 100.0}, 2000.0, pack, basis_type="wt")
    assert result.D == pytest.approx(1.0, abs=1e-12)
    index = [str(x) for x in result.parent_oxides].index("SiO2")
    assert float(result.parent_activity[index]) == pytest.approx(1.0, abs=1e-12)
    assert float(result.parent_gamma[index]) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("case_name", _cases())
def test_every_golden_is_physically_admissible(case_name, pack):
    """Cheap invariants that hold for any valid solve, golden or not.

    Guards against a regenerated golden file that encodes a broken state: if
    someone regenerates after a bad change, the comparison test above would go
    green while the numbers were nonsense. These do not depend on the file's
    recorded values.
    """
    case = _golden()["cases"][case_name]
    result = evaluate(
        case["composition"], case["temperature_K"], pack,
        basis_type=case["basis_type"],
    )
    assert result.D >= 1.0, "association cannot produce MORE moles than went in"
    for name, a, g in zip(
        result.parent_oxides, result.parent_activity, result.parent_gamma
    ):
        assert math.isfinite(float(a)) and float(a) >= 0.0, f"a({name}) = {a}"
        assert math.isfinite(float(g)) and float(g) >= 0.0, f"gamma({name}) = {g}"
        assert float(a) <= 1.0 + 1e-9, f"a({name}) = {a} exceeds the pure standard state"
