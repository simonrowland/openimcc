from __future__ import annotations

import pytest

from openimcc.gas import (
    _SF04_REACTIONS,
    _derived_source_supercooled_liquid_markers,
    _source_model_flag,
    load_gas_datapack,
    evaluate_gas,
)


EXPECTED_SOURCE_SUPERCOOLED_LIQUIDS = {
    "FeO(l)",
    "TiO2(l)",
    "Na2O(l)",
    "WO3(l)",
    "BaO(l)",
    "SrO(l)",
    "ZrO2(l)",
    "Li2O(l)",
    "Cu2O(l)",
    "NbO2(l)",
    "Cr2O3(l)",
    "V2O3(l)",
    "MgO(l)",
    "CaO(l)",
}


def _marker_cases():
    pack = load_gas_datapack()
    markers = _derived_source_supercooled_liquid_markers(pack)
    cases = []
    for species, marker in sorted(markers.items()):
        rows = pack.oxide_df.loc[species]
        if getattr(rows, "ndim", 1) == 1:
            rows = [rows]
        else:
            rows = [row for _, row in rows.iterrows()]
        source_rows = [
            row for row in rows
            if str(row["Ref"]) in {marker["table_id"], "JANAF"}
            and float(row["T_min"]) < marker["fusion_temperature_K"]
        ]
        row = min(source_rows, key=lambda candidate: float(candidate["T_min"]))
        lower = float(row["T_min"])
        upper = min(float(row["T_max"]), float(marker["fusion_temperature_K"]))
        temperature = lower + (upper - lower) / 2.0
        cases.append((species, marker, row, temperature))
    return tuple(cases)


def test_source_liquid_marker_derivation_finds_exact_expected_names() -> None:
    pack = load_gas_datapack()
    assert set(_derived_source_supercooled_liquid_markers(pack)) == (
        EXPECTED_SOURCE_SUPERCOOLED_LIQUIDS
    )


@pytest.mark.parametrize(
    ("species", "marker", "row", "temperature"), _marker_cases(),
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_source_tabulated_liquid_branches_use_source_thresholds(
    species: str, marker: dict, row, temperature: float
) -> None:
    pack = load_gas_datapack()
    parent = species.removesuffix("(l)")
    channel = next(
        name for name, reaction in _SF04_REACTIONS.items()
        if reaction[0] == parent
    )
    result = evaluate_gas(
        {parent: 1.0e-3}, temperature, 1.0e-8, pack,
        parent_oxides=(parent,), gas_species=(channel,),
    )
    flag = result.domain_flags[channel]
    assert flag is not None
    assert marker["table_id"] in flag
    glass_join = marker["glass_join_temperature_K"]
    if glass_join is not None and temperature < glass_join:
        assert "source-tabulated glass cells" in flag
        assert f"{glass_join:g} K glass/liquid join" in flag
    else:
        assert "source-tabulated supercooled liquid branch" in flag
        assert f"{marker['fusion_temperature_K']:g} K fusion marker" in flag

    assert _source_model_flag(
        species, float(marker["fusion_temperature_K"]), row, phase="l"
    ) is None


@pytest.mark.parametrize(
    ("species", "temperature"),
    (("PbO(l)", 1100.0), ("B2O3(l)", 700.0), ("MoO3(l)", 1000.0)),
)
def test_supercooled_branch_flag_is_silent_below_the_packaged_row_floor(
    species: str, temperature: float
) -> None:
    # These rows begin at 1200 K, above or near their fusion markers, so no
    # source-tabulated supercooled cells exist below that floor.
    pack = load_gas_datapack()
    parent = species.removesuffix("(l)")
    channel = next(
        name for name, reaction in _SF04_REACTIONS.items()
        if reaction[0] == parent
    )
    result = evaluate_gas(
        {parent: 1.0e-3}, temperature, 1.0e-8, pack,
        parent_oxides=(parent,), gas_species=(channel,),
        allow_extrapolation=True,
    )
    flag = result.domain_flags[channel] or ""
    assert "source-tabulated supercooled liquid branch" not in flag
    assert "source-tabulated glass cells" not in flag
