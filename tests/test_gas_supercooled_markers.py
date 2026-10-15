from __future__ import annotations

import pytest

from openimcc.gas import evaluate_gas, load_gas_datapack


@pytest.mark.parametrize(
    ("parent", "species", "marker", "source", "below", "above"),
    (
        ("FeO", "Fe", 1650.0, "Fe-019", 1500.0, 1700.0),
        ("TiO2", "TiO2", 2130.0, "O-044", 2000.0, 2200.0),
        ("Na2O", "Na2O", 1405.2, "Na-013", 1300.0, 1500.0),
        ("WO3", "WO3", 1745.0, "O-066", 1600.0, 1800.0),
        ("BaO", "BaO", 2286.0, "Ba-035", 2000.0, 2300.0),
        ("SrO", "SrO", 2938.0, "O-014", 2800.0, 3000.0),
        ("ZrO2", "ZrO2", 2950.0, "O-053", 2800.0, 3000.0),
    ),
)
def test_source_tabulated_supercooled_liquid_branches_are_flagged(
    parent: str,
    species: str,
    marker: float,
    source: str,
    below: float,
    above: float,
) -> None:
    pack = load_gas_datapack()
    below_result = evaluate_gas(
        {parent: 1.0e-3}, below, 1.0e-8, pack,
        parent_oxides=(parent,), gas_species=(species,),
    )
    above_result = evaluate_gas(
        {parent: 1.0e-3}, above, 1.0e-8, pack,
        parent_oxides=(parent,), gas_species=(species,),
    )

    below_flag = below_result.domain_flags[species]
    assert below_flag is not None
    assert "source-tabulated supercooled liquid branch" in below_flag
    assert f"{marker:g} K" in below_flag
    assert source in below_flag
    assert "source-tabulated supercooled liquid branch" not in (
        above_result.domain_flags[species] or ""
    )
