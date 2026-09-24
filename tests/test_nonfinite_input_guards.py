"""Non-finite numeric inputs are refused at the input, with the true reason.

The bug class: a bare ``x <= 0.0`` (or ``x < 0.0``) guard is not a validator,
because under IEEE 754 every ordered comparison with nan is False and
``inf <= 0`` is False. Both values therefore pass it.

For ``basis``, that let ``parent_mol / basis`` produce all zeros (inf) or all
nan. The call still refused, but much later, as "no positive parent oxides":
a true refusal with a false reason. The composition was fine; the basis was
the invalid input. A caller reading the message would go looking in the wrong
place.

So these tests pin the REASON, not only the fact of refusal. Every one of
them fails against the pre-fix code, which refused with the misleading
message.
"""

from __future__ import annotations

import math

import pytest

from openimcc import (
    ImccComponentOutsideDomainError,
    ImccCompositionIncompleteError,
    ImccRefusal,
    evaluate,
    load_datapack,
)
from openimcc.kernel import solve_imcc_sf04

PACK = "src/openimcc/data/packs/imcc-sf04-v1.0.2.json"
PARENTS = ("SiO2", "MgO", "FeO", "CaO", "Al2O3", "TiO2", "Na2O", "K2O")
NONFINITE = [
    pytest.param(math.nan, id="nan"),
    pytest.param(math.inf, id="inf"),
    pytest.param(-math.inf, id="neg-inf"),
]
# The pre-fix message. It must never be produced for a bad basis again.
MISLEADING = "no positive parent oxides"


@pytest.fixture(scope="module")
def pack():
    return load_datapack(PACK)


def _uniform():
    return {name: 1.0 for name in PARENTS}


@pytest.mark.parametrize("basis", NONFINITE)
def test_evaluate_refuses_nonfinite_basis_with_the_true_reason(pack, basis):
    with pytest.raises(ImccCompositionIncompleteError) as exc:
        evaluate(_uniform(), 2000.0, pack, basis=basis)
    message = str(exc.value)
    assert "finite" in message, message
    assert MISLEADING not in message, message
    assert exc.value.code == "imcc_composition_incomplete"


@pytest.mark.parametrize("basis", NONFINITE)
def test_kernel_refuses_nonfinite_basis_with_the_true_reason(pack, basis):
    """The kernel carries its own basis check, reachable without model.evaluate.

    The model-layer guard alone would leave direct kernel callers exposed to
    the same misleading refusal.
    """
    parent = [1.0] * len(PARENTS)
    with pytest.raises(ImccCompositionIncompleteError) as exc:
        solve_imcc_sf04(parent, 2000.0, pack.kernel_datapack, basis=basis)
    message = str(exc.value)
    assert "finite" in message, message
    assert MISLEADING not in message, message


@pytest.mark.parametrize("amount", NONFINITE)
def test_nonfinite_extra_component_is_refused_as_nonfinite(pack, amount):
    """Checked before the ferric screen, so the reason names the real problem.

    Fe2O3 is refused whenever it is nonzero, so a nan Fe2O3 used to surface as
    "ferric input unsupported". That is technically a refusal, but not of the
    thing that was actually wrong with the input.
    """
    with pytest.raises(ImccCompositionIncompleteError) as exc:
        evaluate(_uniform(), 2000.0, pack, extra_mol={"Fe2O3": amount})
    message = str(exc.value)
    assert exc.value.code == "imcc_composition_incomplete"
    assert "extra component Fe2O3 has non-finite moles" in message, message


@pytest.mark.parametrize("component", ["Fe2O3", "S", "P2O5"])
@pytest.mark.parametrize("amount", NONFINITE)
def test_nonfinite_mapping_component_is_refused_before_component_screens(
    pack, component, amount
):
    composition = _uniform()
    composition[component] = amount

    with pytest.raises(ImccCompositionIncompleteError) as exc:
        evaluate(composition, 2000.0, pack)

    assert exc.value.code == "imcc_composition_incomplete"
    assert "composition contains non-finite values" in str(exc.value)


@pytest.mark.parametrize("component", ["S", "P2O5"])
@pytest.mark.parametrize("amount", NONFINITE)
def test_nonfinite_sp_extra_is_refused_before_extension_screen(
    pack, component, amount
):
    with pytest.raises(ImccCompositionIncompleteError) as exc:
        evaluate(_uniform(), 2000.0, pack, extra_mol={component: amount})

    assert exc.value.code == "imcc_composition_incomplete"
    assert f"extra component {component} has non-finite moles" in str(exc.value)


def test_zero_ferric_mapping_value_still_reaches_outside_domain(pack):
    with pytest.raises(ImccComponentOutsideDomainError) as exc:
        evaluate({"SiO2": 1.0, "Fe2O3": 0.0}, 2000.0, pack)

    assert exc.value.code == "imcc_component_outside_domain"
    assert "outside IMCC-SF04 domain" in str(exc.value)


@pytest.mark.parametrize("basis", [0.0, -1.0])
def test_nonpositive_finite_basis_still_refused(pack, basis):
    """Regression guard: widening the check must not weaken the original case."""
    with pytest.raises(ImccCompositionIncompleteError) as exc:
        evaluate(_uniform(), 2000.0, pack, basis=basis)
    assert "positive" in str(exc.value)


def test_a_valid_declared_basis_still_solves(pack):
    """Control. The guard must not fire on correct input.

    For a uniform eight-oxide composition of 1 mol each, the parent mole sum
    is 8. Declaring basis=8.0 therefore matches exactly, well inside the 1e-6
    relative slack, and the solve must proceed. D >= 1 is the physical floor:
    association can only lower the species count relative to the parents, and
    D = n_parent / n_species.
    """
    result = evaluate(_uniform(), 2000.0, pack, basis=8.0)
    assert math.isfinite(result.D) and result.D >= 1.0


@pytest.mark.parametrize("basis", NONFINITE)
def test_weight_basis_refuses_nonfinite_declared_basis(pack, basis):
    with pytest.raises(ImccCompositionIncompleteError) as exc:
        evaluate(_uniform(), 2000.0, pack, basis=basis, basis_type="wt")

    assert exc.value.code == "imcc_composition_incomplete"
    assert "declared basis must be a positive finite number" in str(exc.value)


def test_valid_weight_basis_still_solves(pack):
    result = evaluate(_uniform(), 2000.0, pack, basis=8.0, basis_type="wt")
    assert math.isfinite(result.D) and result.D >= 1.0


def test_kernel_refuses_nonfinite_implicit_parent_total(pack):
    with pytest.raises(ImccCompositionIncompleteError) as exc:
        solve_imcc_sf04([1.0e308] * len(PARENTS), 2000.0, pack.kernel_datapack)

    assert exc.value.code == "imcc_composition_incomplete"
    assert "parent mole total is not finite" in str(exc.value)
    assert "declared basis" not in str(exc.value)


def test_evaluate_refuses_nonfinite_implicit_parent_total(pack):
    with pytest.raises(ImccCompositionIncompleteError) as exc:
        evaluate({name: 1.0e308 for name in PARENTS}, 2000.0, pack)

    assert exc.value.code == "imcc_composition_incomplete"
    assert "parent mole total is not finite" in str(exc.value)
    assert "declared basis" not in str(exc.value)


def test_refusals_stay_typed(pack):
    """Every new refusal is catchable as the package's ImccRefusal base, which
    is what makes it usable as a typed result rather than a crash."""
    with pytest.raises(ImccRefusal):
        evaluate(_uniform(), 2000.0, pack, basis=math.inf)


@pytest.mark.parametrize(
    ("where", "name", "raw"),
    [
        pytest.param("composition", "SiO2", "abc", id="parent-string"),
        pytest.param("composition", "SiO2", None, id="parent-none"),
        pytest.param("composition", "Cr2O3", "abc", id="unknown-key-string"),
        pytest.param("composition", "Fe2O3", "abc", id="ferric-key-string"),
        pytest.param("composition", "S", "abc", id="sp-key-string"),
        pytest.param("extra_mol", "Cr2O3", "abc", id="extra-string"),
    ],
)
def test_non_numeric_amount_is_a_typed_refusal_naming_the_entry(pack, where, name, raw):
    """A non-numeric amount is invalid input: typed, and naming the entry.

    Before the pre-check existed, some of these escaped as an untyped
    ValueError/TypeError from float() and others were refused for an
    unrelated reason (ferric, outside domain). All now refuse the same way.
    """
    composition = _uniform()
    extra = None
    if where == "composition":
        composition[name] = raw
    else:
        extra = {name: raw}
    with pytest.raises(ImccCompositionIncompleteError) as exc:
        evaluate(composition, 2000.0, pack, extra_mol=extra)
    message = str(exc.value)
    assert "is not a number" in message, message
    assert name in message, message


@pytest.mark.parametrize("name", ["SiO2", "Cr2O3", "Fe2O3", "S"])
def test_amount_beyond_float_range_is_a_typed_refusal(pack, name):
    """float(10**400) raises OverflowError, not ValueError; it must still be typed."""
    composition = _uniform()
    composition[name] = 10**400
    with pytest.raises(ImccCompositionIncompleteError) as exc:
        evaluate(composition, 2000.0, pack)
    message = str(exc.value)
    assert "too large to represent" in message, message
    assert name in message, message


def test_extra_amount_beyond_float_range_is_a_typed_refusal(pack):
    with pytest.raises(ImccCompositionIncompleteError) as exc:
        evaluate(_uniform(), 2000.0, pack, extra_mol={"Cr2O3": 10**400})
    assert "too large to represent" in str(exc.value)
