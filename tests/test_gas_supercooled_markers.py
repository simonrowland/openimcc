from __future__ import annotations

import pytest

from openimcc.gas import (
    ImccGasDatapack,
    ImccGasValidityMetadataError,
    _SF04_REACTIONS,
    _derived_source_supercooled_liquid_markers,
    _source_model_flag,
    _source_validity_for_row,
    _supercooled_liquid_from_solid_flag,
    load_gas_datapack,
    evaluate_gas,
)


EXPECTED_SOURCE_SUPERCOOLED_LIQUIDS = {
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
    "Ta2O5(l)",
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
    legacy = load_gas_datapack(
        oxide_path="src/openimcc/data/gas/condensate-feo-janaf-legacy.csv"
    )
    assert "FeO(l)" in _derived_source_supercooled_liquid_markers(legacy)


def test_feo_redox_fit_range_flags_default_without_ref_matching() -> None:
    pack = load_gas_datapack()
    feo_row = pack.oxide_df.loc["FeO(l)"]
    result = evaluate_gas(
        {"FeO": 1.0e-3}, 1200.0, 1.0e-8, pack,
        parent_oxides=("FeO",), gas_species=("Fe", "FeO"),
    )
    for channel in ("Fe", "FeO"):
        flag = result.domain_flags[channel]
        assert flag is not None
        assert "D&G/Masson correction extrapolated" in flag
        assert "1649.15-1766.15 K" in flag
        assert "Ref=DG46-MASSON65" in flag

    renamed_row = feo_row.copy()
    renamed_row["Ref"] = "RENAMED-SOURCE"
    renamed_flag = _source_model_flag(
        "FeO(l)", 1200.0, renamed_row, phase="l"
    )
    assert renamed_flag is not None
    assert "Ref=RENAMED-SOURCE" in renamed_flag

    legacy = load_gas_datapack(
        oxide_path="src/openimcc/data/gas/condensate-feo-janaf-legacy.csv"
    )
    legacy_result = evaluate_gas(
        {"FeO": 1.0e-3}, 1200.0, 1.0e-8, legacy,
        parent_oxides=("FeO",), gas_species=("Fe", "FeO"),
    )
    for channel in ("Fe", "FeO"):
        assert "source-tabulated supercooled liquid branch" in (
            legacy_result.domain_flags[channel] or ""
        )

    in_band = evaluate_gas(
        {"FeO": 1.0e-3}, 1700.0, 1.0e-8, pack,
        parent_oxides=("FeO",), gas_species=("Fe", "FeO"),
    )
    assert in_band.domain_flags["Fe"] is None
    assert in_band.domain_flags["FeO"] is None


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


def _row_at(pack, row_name: str, phase: str, temperature: float):
    table = pack.gas_df if phase == "g" else pack.oxide_df
    rows = table.loc[row_name]
    candidates = [rows] if getattr(rows, "ndim", 1) == 1 else [row for _, row in rows.iterrows()]
    return max(
        (row for row in candidates if float(row["T_min"]) <= temperature),
        key=lambda row: float(row["T_min"]),
    )


@pytest.mark.parametrize(
    ("row_name", "phase", "temperature", "expected"),
    (
        (
            "Re2O7(g)", "g", 550.0,
            "constructed Re2O7(g) from Smith 1952 measured vapor pressure and "
            "B677 condensed functions over 503-598 K; maximum log-pressure "
            "fit scatter propagated as ±1.4 kJ/mol at 550 K",
        ),
        (
            "Re2O7(g)", "g", 1000.0,
            "constructed Re2O7(g) from Battles vapor-pressure equilibrium measured "
            "at 961-1087 K; B677 Re2O7(l) Cp held above its 700 K limit for the "
            "condensed-function reconstruction; propagated regression coefficient "
            "band ±5.5 kJ/mol at 1000 K",
        ),
        (
            "Re2O7(g)", "g", 1500.0,
            "constructed Re2O7(g) from Battles vapor-pressure equilibrium measured "
            "at 961-1087 K; held-Cp runtime continuation from 1087 K to 3000 K; "
            "B677 Re2O7(l) Cp held above its 700 K limit for the condensed-function "
            "reconstruction; propagated vapor-pressure regression bound ±6.9 "
            "kJ/mol at 1500 K",
        ),
        ("Hg(l)", "l", 1400.0, "Hg(l) uses a flagged liquid-metal continuation above the 629.839 K source limit"),
        ("Rh(l)", "l", 2600.0, "Rh(l) uses a flagged l-phase continuation above the 2500 K source limit"),
        ("Re(cr)", "s", 2500.0, "Re(cr) uses a flagged s-phase continuation above the 2000 K source limit"),
        ("La2O3(l)", "l", 1800.0, "La2O3(l) uses the printed constant-Cp liquid branch as a supercooled continuation below 2577 K"),
        ("NiO(l)", "l", 1600.0, "NiO(l) uses the TSIV liquid function as a supercooled branch below 2228 K"),
        ("MoO3(l)", "l", 2600.0, "MoO3(l) uses a labelled constant-Cp continuation above 2000 K from Mo-015"),
        ("MnO(l)", "l", 1800.0, "MnO(l) uses a tier-4 liquid_from_solid construction from MnO-IVTAN-1435"),
        (
            "ThO2(l)", "l", 2000.0,
            "ThO2(l) uses a tier-4 liquid_from_solid construction from P672-ThO2-c; "
            "its crystal Cp was extended above 2500 K to the estimated fusion temperature",
        ),
        (
            "TiO2(l)", "l", 1600.0,
            "TiO2(l) uses a source-tabulated supercooled liquid branch below the "
            "2130 K fusion marker (O-044)",
        ),
    ),
)
def test_source_model_flags_keep_text_after_source_swap(
    row_name: str, phase: str, temperature: float, expected: str
) -> None:
    pack = load_gas_datapack()
    row = _row_at(pack, row_name, phase, temperature).copy()
    assert _source_model_flag(row_name, temperature, row, phase=phase) == expected

    row["Ref"] = "SOURCE-SWAP-PROBE"
    assert _source_model_flag(row_name, temperature, row, phase=phase) == expected


def test_lfs_supercooled_flag_uses_typed_fusion_temperature_after_source_swap() -> None:
    pack = load_gas_datapack()
    row = _row_at(pack, "MnO(l)", "l", 1800.0).copy()
    expected = (
        "MnO(l) uses a supercooled liquid_from_solid branch below the "
        "2120 K fusion temperature"
    )
    assert _supercooled_liquid_from_solid_flag("MnO(l)", 1800.0, row) == expected
    row["Ref"] = "SOURCE-SWAP-PROBE"
    assert _supercooled_liquid_from_solid_flag("MnO(l)", 1800.0, row) == expected


def test_supercooled_constant_cp_flag_survives_ref_swap() -> None:
    pack = load_gas_datapack()
    oxide = pack.oxide_df.copy(deep=True)
    oxide.loc[oxide.index == "TiO2(l)", "Ref"] = "SOURCE-SWAP-PROBE"
    swapped = ImccGasDatapack(
        pack.gas_df.copy(deep=True), oxide, pack.gas_path, pack.oxide_path
    )
    result = evaluate_gas(
        {"TiO2": 1.0e-3}, 1400.0, 1.0e-8, swapped,
        parent_oxides=("TiO2",), gas_species=("Ti", "TiO", "TiO2"),
    )
    expected = "TiO2(l) uses a labelled constant-Cp supercooled-liquid continuation from JANAF O-044"
    assert set(result.domain_flags.values()) == {expected}


def test_row_without_typed_validity_columns_fails_loud() -> None:
    pack = load_gas_datapack()
    row = _row_at(pack, "Hg(l)", "l", 1400.0).drop(
        labels=["validity_kind", "continuation_kind", "cp_extension", "construction_method",
                "table_id", "source_label", "source_liquid_marker", "supercooled_continuation",
                "source_fit_min_K", "source_fit_max_K", "source_limit_K", "fusion_temperature_K",
                "supercooled_segment_min_K", "supercooled_segment_max_K", "construction_tier",
                "glass_join_temperature_K"]
    )
    row["Ref"] = "SOURCE-SWAP-PROBE"
    with pytest.raises(ImccGasValidityMetadataError, match="lacks typed source-validity columns"):
        _source_model_flag("Hg(l)", 1400.0, row, phase="l")


def test_source_validity_survives_refit_and_source_swap_on_every_coverage_row() -> None:
    pack = load_gas_datapack()
    coverage_rows = 0
    for phase, table in (("g", pack.gas_df), ("l", pack.oxide_df)):
        for row_name, rows in table.groupby(level=0, sort=False):
            for _, row in rows.iterrows():
                if row["validity_kind"] != "coverage":
                    continue
                coverage_rows += 1
                row_phase = phase if phase == "g" else ("s" if str(row["state"]) == "s" else "l")
                coefficient = "A" if phase == "g" else "dG_A"
                refit = row.copy()
                refit[coefficient] = float(refit[coefficient]) + 0.001
                refit["Ref"] = "SOURCE-SWAP-PROBE"
                temperatures = (
                    float(row["T_min"]) - 1.0,
                    (float(row["T_min"]) + float(row["T_max"])) / 2.0,
                    float(row["T_max"]) + 1.0,
                )
                for temperature in temperatures:
                    assert _source_model_flag(
                        str(row_name), temperature, refit, phase=row_phase
                    ) == _source_model_flag(
                        str(row_name), temperature, row, phase=row_phase
                    )
                assert _source_validity_for_row(str(row_name), refit, phase=row_phase) == (
                    _source_validity_for_row(str(row_name), row, phase=row_phase)
                )
    assert coverage_rows == 204


def test_mno_multi_interval_lfs_refit_keeps_typed_supercooled_flag() -> None:
    pack = load_gas_datapack()
    rows = pack.oxide_df.loc["MnO(l)"]
    assert len(rows) == 3
    expected = (
        "MnO(l) uses a supercooled liquid_from_solid branch below the "
        "2120 K fusion temperature"
    )
    for _, row in rows.iterrows():
        refit = row.copy()
        refit["dG_A"] = float(refit["dG_A"]) + 0.001
        refit["Ref"] = "SOURCE-SWAP-PROBE"
        assert _supercooled_liquid_from_solid_flag("MnO(l)", 1800.0, refit) == expected
