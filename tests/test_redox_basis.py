"""Tests for the element-amount inventory contract."""

from __future__ import annotations

import pytest

from openimcc.redox_basis import oxide_inventory_to_elements
from openimcc.redox_pack import load_redox_pack


def test_oxide_splits_map_to_the_same_element_inventory() -> None:
    pack = load_redox_pack()
    formulas = {
        parent["parent_oxide"]: parent["parent_formula_atoms"]
        for parent in pack.parents
    }
    formulas["Fe2O3(l)"] = pack.content["thermo_species"]["Fe2O3(l)"][
        "formula_atoms"
    ]
    formulas["O2(g)"] = pack.content["thermo_species"]["O2(g)"]["formula_atoms"]

    hematite = oxide_inventory_to_elements({"Fe2O3(l)": 1.0}, formulas)
    ferrous_plus_oxygen = oxide_inventory_to_elements(
        {"FeO(l)": 2.0, "O2(g)": 0.5}, formulas
    )

    assert hematite == {"Fe": 2.0, "O": 3.0}
    assert ferrous_plus_oxygen == hematite


@pytest.mark.parametrize(
    ("inventory", "formula_atoms", "expected"),
    [
        ({"FeO": 2.0}, {"FeO": {"Fe": 1, "O": 1}}, {"Fe": 2.0, "O": 2.0}),
        ({"O2": 0.5}, {"O2": {"O": 2}}, {"O": 1.0}),
        ({}, {}, {}),
    ],
)
def test_inventory_mapping_returns_moles_of_each_element(
    inventory: dict[str, float],
    formula_atoms: dict[str, dict[str, int]],
    expected: dict[str, float],
) -> None:
    assert oxide_inventory_to_elements(inventory, formula_atoms) == expected


@pytest.mark.parametrize("amount", [-1.0, float("nan"), float("inf")])
def test_inventory_mapping_rejects_invalid_amounts(amount: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        oxide_inventory_to_elements({"FeO": amount}, {"FeO": {"Fe": 1, "O": 1}})


def test_inventory_mapping_refuses_unregistered_formula() -> None:
    with pytest.raises(ValueError, match="no atom-count record"):
        oxide_inventory_to_elements({"Fe2O3": 1.0}, {"FeO": {"Fe": 1, "O": 1}})
