"""Runtime gates for the packaged IMCC-SF04 gas mass-action layer."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from openimcc import evaluate
from tools import build_gas_tables
from openimcc.gas import (
    IMCC_GAS_CHANNEL_SPECIES,
    IMCC_GAS_INCOMPLETE_PARENT_SPECIES,
    IMCC_GAS_NO_JANAF_ROWS,
    IMCC_GAS_PUBLIC_ONLY_PARENT_SPECIES,
    IMCC_GAS_UNAVAILABLE_SPECIES,
    IMCC_GAS_WORKBOOK_EXTRAPOLATION_LABELS,
    IMCC_GAS_WORKBOOK_IN_DOMAIN_SPECIES,
    IMCC_PARENT_OXIDES,
    IMCC_SF04_WORKBOOK_GRID_K,
    R_J_MOL_K,
    ImccGasDatapack,
    ImccGasInvalidFugacityError,
    ImccGasOxygenBalanceError,
    ImccGasResult,
    ImccGasSpeciesNotFoundError,
    ImccGasTemperatureOutsideDomainError,
    default_condensate_database_path,
    default_gas_database_path,
    evaluate_gas,
    evaluate_gas_oxygen_balance,
    oxygen_balance_species_metadata,
    oxygen_balance_from_pressure_model,
    gas_species_provenance,
    load_gas_datapack,
    _REACTION_PARENT_OXIDES,
    _SF04_REACTIONS,
    _nearest_interval_row,
    _SF04_REACTIONS,
)


# These are independent lg10(p/bar) references, rounded to three decimals.
# For every gas record, the source-side calculation uses the JANAF record's
# printed ``formation_enthalpy`` at 298.15 K, ``enthalpy_increment`` and
# ``entropy`` at the test temperature; it never reads gas-shomate.csv.  The
# parent side uses the named shipped condensate.csv row.  The source mappings
# are: Na-005+Na2O(l), K-005+K2O(l), O-012+SiO2(l), Fe-008+FeO(l),
# Fe-021+FeO(l), Mg-005+MgO(l), Mg-011+MgO(l), O-040+SiO2(l), and O-029
# for the caller-pinned O2 reference.
#
# Premise: at unit parent activity and fO2, the reaction equilibrium relation is
# log10(p_gas) = -(n_gas*G_gas + n_O2*G_O2 - G_parent) /
# (n_gas*R*T*ln(10)). Algebra: source kJ/mol values are converted to J/mol
# before the stoichiometric sum. Unit check: the denominator is J/mol. Sanity:
# SiO2(l) -> SiO2(g) at 2000 K rounds to -6.669 (n_O2 = 0); SiO2(l) ->
# SiO(g) + 1/2 O2(g) rounds to -7.759.
_RUNG2B_AGAINST_JANAF_PRINTED_GAS_COLUMNS = {
    2000.0: {
        "Na": -0.312352,
        "K": 0.333,
        "SiO": -7.759,
        "Fe": -7.307,
        "FeO": -5.465,
        "Mg": -7.856,
        "MgO": -7.177,
        "SiO2": -6.669,
        "O2": 0.0,
    }
}

# T-625 source mappings are Al-074/Al-077/Al-092/Al-094/Al-005 plus
# Al2O3(l), Na-011/Na-008 plus Na2O(l), K-011/K-008 plus K2O(l),
# Si-005 plus SiO2(l), Ca-030/Ca-006 plus CaO(l), and O-001 without a parent.
_T625_AGAINST_JANAF_PRINTED_GAS_COLUMNS = {
    2500.0: {
        "O": -1.839,
        "AlO": -7.029309,
        "AlO2": -7.095906,
        "Al2O": -12.292150,
        "Al2O2": -10.749688,
        "Na2": -0.809928,
        "NaO": 0.043715,
        "K2": -0.030,
        "KO": 0.534,
        "Si": -11.905,
        "Al": -9.859116,
        "CaO": -5.475,
        "Ca": -6.258,
    }
}


# Titanium channels against the printed JANAF formation-Gibbs columns at
# 2000 K: Ti-006 Ti(g) 191.423, O-022 TiO(g) -123.256, O-046 TiO2(g) -330.354
# and O-044 TiO2(l) -581.532 kJ/mol (O2 is the reference, 0).  These columns
# are independent of both the fitted gas rows and the fitted TiO2(l) row.
#
# Premise: TiO2(l) = TiO2(g), TiO2(l) = TiO(g) + 1/2 O2 and
# TiO2(l) = Ti(g) + O2, each at unit parent activity and fO2 = 1.  Algebra:
# log10 K = -dG_r/(R T ln 10) with R T ln 10 = 38289.515 J/mol at 2000 K;
# dG_r = 251.178, 458.276 and 772.955 kJ/mol give -6.55997, -11.96871 and
# -20.18712, rounded below to three decimals.  Unit check: kJ/mol * 1000 over
# J/mol is dimensionless.  Sanity: TiO2(g) at -6.560 sits beside the SiO2(g)
# reference of -6.669 above, and TiO overtakes TiO2 only below
# log10 fO2 = 2*(-11.969 + 6.560) = -10.818.
_TI_AGAINST_JANAF_PRINTED_FORMATION_GIBBS = {
    2000.0: {
        "Ti": -20.187,
        "TiO": -11.969,
        "TiO2": -6.560,
    }
}

# Independent JANAF formation-Gibbs checks at 2000 K. The parent activities
# are caller-provided at 1e-3 and fO2 = 1; no generated coefficient appears in
# these references.
_VNB_AGAINST_JANAF_PRINTED_FORMATION_GIBBS = {
    2000.0: {
        "V": -16.456,
        "VO": -9.518,
        "VO2": -3.614,
        "Nb": -25.871,
        "NbO": -15.061,
        "NbO2": -8.176,
    }
}


@pytest.fixture(scope="module")
def gas_pack() -> ImccGasDatapack:
    return load_gas_datapack()


@pytest.fixture
def unit_activities() -> dict[str, float]:
    return {
        "SiO2": 1.0,
        "MgO": 1.0,
        "FeO": 1.0,
        "CaO": 1.0,
        "Al2O3": 1.0,
        "TiO2": 1.0,
        "Na2O": 1.0,
        "K2O": 1.0,
    }


def test_default_tables_are_packaged_and_load_without_environment(
    monkeypatch: pytest.MonkeyPatch, gas_pack: ImccGasDatapack
) -> None:
    monkeypatch.delenv("OPENIMCC_VAPOROCK_ROOT", raising=False)
    gas_path = default_gas_database_path()
    oxide_path = default_condensate_database_path()
    assert gas_path.name == "gas-shomate.csv"
    assert oxide_path.name == "condensate.csv"
    assert gas_path.is_file()
    assert oxide_path.is_file()
    assert len(gas_pack.gas_df) == 62 + len(build_gas_tables.LOW_T_GAS_SPECIES) + 2
    assert len(gas_pack.oxide_df) == 13


def test_janaf_parent_liquid_research_pack_loads_by_path_and_is_in_domain(
    gas_pack: ImccGasDatapack,
) -> None:
    pack_dir = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "openimcc"
        / "data"
        / "packs"
        / "gas-janaf-parent-liquids-research"
    )
    research = load_gas_datapack(
        gas_path=pack_dir / "gas-shomate.csv",
        oxide_path=pack_dir / "condensate.csv",
    )
    interval_1 = gas_pack.gas_df.loc[
        gas_pack.gas_df["T_interval"].astype(int) == 1
    ]
    interval_1 = interval_1.loc[
        ~interval_1.index.isin(tuple(f"{species}(g)" for species in _PS_CHANNELS))
    ]
    low_nasa = gas_pack.gas_df.loc[
        (gas_pack.gas_df["T_interval"].astype(int) == 2)
        & gas_pack.gas_df.index.isin(("Na2O(g)", "K2O(g)"))
    ]
    assert research.gas_df.equals(pd.concat((interval_1, low_nasa)))
    assert research.gas_path == pack_dir / "gas-shomate.csv"
    assert research.oxide_path == pack_dir / "condensate.csv"

    parents = {
        "SiO2": ("SiO2(l)", "Si", 1800.0),
        "Al2O3": ("Al2O3(l)", "Al", 2500.0),
        "MgO": ("MgO(l)", "Mg", 2200.0),
        "CaO": ("CaO(l)", "Ca", 2200.0),
    }
    for oxide, (row_name, gas_species, expected_t_min) in parents.items():
        research_row = research.oxide_df.loc[row_name]
        assert float(research_row["T_min"]) == expected_t_min
        pressures = evaluate_gas(
            {oxide: 1.0},
            expected_t_min,
            1.0e-10,
            research,
            parent_oxides=(oxide,),
            gas_species=(gas_species,),
        )
        assert pressures.domain_flags[gas_species] is None

        default_row = gas_pack.oxide_df.loc[row_name]
        assert default_row["Ref"] == "LAM1987"


def test_explicit_vaporock_override_remains_supported(
    monkeypatch: pytest.MonkeyPatch,
    gas_pack: ImccGasDatapack,
    tmp_path: Path,
) -> None:
    root = tmp_path / "vaporock"
    gas_target = root / "src" / "vaporock" / "data"
    oxide_target = root / "data"
    gas_target.mkdir(parents=True)
    oxide_target.mkdir(parents=True)
    shutil.copy2(gas_pack.gas_path, gas_target / "JANAF-vapor-data-full.csv")
    shutil.copy2(gas_pack.oxide_path, oxide_target / "condensate-thermo-data.csv")
    monkeypatch.setenv("OPENIMCC_VAPOROCK_ROOT", str(root))

    overridden = load_gas_datapack()
    assert overridden.gas_path == gas_target / "JANAF-vapor-data-full.csv"
    assert overridden.oxide_path == oxide_target / "condensate-thermo-data.csv"
    assert overridden.gas_df.equals(gas_pack.gas_df)
    assert overridden.oxide_df.equals(gas_pack.oxide_df)


def test_runtime_schemas_and_interval_ranges(gas_pack: ImccGasDatapack) -> None:
    assert tuple(gas_pack.gas_df.columns) == (
        "state",
        "T_interval",
        "cation",
        "cat_num",
        "oxy_num",
        "T_min",
        "T_max",
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "Ref",
    )
    assert tuple(gas_pack.oxide_df.columns) == (
        "state",
        "cation",
        "cat_num",
        "oxy_num",
        "T_min",
        "T_max",
        "dH298_R",
        "dG_A",
        "dG_B",
        "dG_C",
        "dG_D",
        "dG_E",
        "Ref",
    )
    public_gas_channels = set(IMCC_GAS_CHANNEL_SPECIES) - {
        "MnO",
        "NiO",
        "CoO",
    }
    assert set(gas_pack.gas_df.index) == {
        f"{species}(g)" for species in public_gas_channels
    }
    high_rows = gas_pack.gas_df.loc[
        gas_pack.gas_df["T_interval"].astype(int) == 1
    ]
    low_rows = gas_pack.gas_df.loc[
        gas_pack.gas_df["T_interval"].astype(int) == 2
    ]
    assert len(low_rows) == len(build_gas_tables.LOW_T_GAS_SPECIES) + 2
    assert set(low_rows.index) == {
        f"{species}(g)" for species in build_gas_tables.LOW_T_GAS_SPECIES
    } | {"Na2O(g)", "K2O(g)"}
    assert (high_rows["T_min"] == 1500).all()
    assert (low_rows["T_min"] == build_gas_tables.LOW_FIT_T_MIN).all()
    assert (low_rows["T_max"] == build_gas_tables.LOW_FIT_T_MAX).all()
    expected_t_max = high_rows["Ref"].map(
        lambda table_id: build_gas_tables._FIT_T_MAX_BY_TABLE.get(
            table_id, build_gas_tables.FIT_T_MAX
        )
    )
    assert (high_rows["T_max"].astype(float) == expected_t_max).all()


def test_interval_selection_uses_high_row_at_shared_1500_k_node(
    gas_pack: ImccGasDatapack,
) -> None:
    for species in build_gas_tables.LOW_T_GAS_SPECIES:
        high = _nearest_interval_row(gas_pack.gas_df, f"{species}(g)", 1500.0)
        low = _nearest_interval_row(gas_pack.gas_df, f"{species}(g)", 1499.999)
        assert int(high["T_interval"]) == 1
        assert float(high["T_min"]) == 1500.0
        assert int(low["T_interval"]) == 2
        assert float(low["T_min"]) == build_gas_tables.LOW_FIT_T_MIN


def test_interval_selection_below_every_interval_uses_the_lowest(
    gas_pack: ImccGasDatapack,
) -> None:
    # The 500-1500 K rows are appended after the 1500-3000 K rows, so "first
    # row in file order" would extrapolate from the far interval.  Below every
    # T_min the nearest (lowest) interval must be the one extrapolated, and
    # strict mode must still refuse.
    for species in build_gas_tables.LOW_T_GAS_SPECIES:
        row = _nearest_interval_row(
            gas_pack.gas_df, f"{species}(g)", 400.0, allow_extrapolation=True
        )
        assert int(row["T_interval"]) == 2
        with pytest.raises(ImccGasTemperatureOutsideDomainError):
            _nearest_interval_row(gas_pack.gas_df, f"{species}(g)", 400.0)


def test_interval_selection_keeps_a_valid_zero_t_min_row() -> None:
    # An override table may declare T_min = 0.  At T = 1 K only that row is
    # valid; the selector must not return the invalid T_min = 2 K row.
    frame = pd.DataFrame(
        {"T_min": [2.0, 0.0], "T_max": [5.0, 2.0], "T_interval": [2, 1]},
        index=["X(g)", "X(g)"],
    )
    row = _nearest_interval_row(frame, "X(g)", 1.0)
    assert float(row["T_min"]) == 0.0


def test_channel_coverage_ledger_is_closed() -> None:
    implemented = set(IMCC_GAS_CHANNEL_SPECIES)
    unavailable = set(IMCC_GAS_UNAVAILABLE_SPECIES)
    in_domain = set(IMCC_GAS_WORKBOOK_IN_DOMAIN_SPECIES)
    extrapolated = set(IMCC_GAS_WORKBOOK_EXTRAPOLATION_LABELS)
    assert len(implemented) == 65
    assert len(unavailable) == 5
    assert implemented.isdisjoint(unavailable)
    assert in_domain.isdisjoint(extrapolated)
    assert not set(_P_CHANNELS) & (in_domain | extrapolated)
    assert in_domain | extrapolated | set(_P_CHANNELS) == implemented


_EXTRAPOLATION_REFUSAL_CASES = (
    ("Cr", 1500.0, "Cr2O3(l)"),
    ("CrO", 1500.0, "Cr2O3(l)"),
    ("CrO2", 1500.0, "Cr2O3(l)"),
    ("CrO3", 1500.0, "Cr2O3(l)"),
    ("SiO", 1900.0, "SiO2(l)"),
    ("Mg", 2500.0, "MgO(l)"),
    ("MgO", 2500.0, "MgO(l)"),
    ("SiO2", 1900.0, "SiO2(l)"),
    ("AlO", 2000.0, "Al2O3(l)"),
    ("AlO2", 2000.0, "Al2O3(l)"),
    ("Al2O", 2000.0, "Al2O3(l)"),
    ("Al2O2", 2000.0, "Al2O3(l)"),
    ("Al2", 2000.0, "Al2O3(l)"),
    ("Si", 1900.0, "SiO2(l)"),
    ("Si2", 1900.0, "SiO2(l)"),
    ("Si3", 1900.0, "SiO2(l)"),
    ("Al", 2000.0, "Al2O3(l)"),
    ("CaO", 2500.0, "CaO(l)"),
    ("Ca", 2500.0, "CaO(l)"),
)


@pytest.mark.parametrize(
    ("species", "temperature", "source_species"), _EXTRAPOLATION_REFUSAL_CASES
)
def test_parent_domain_gaps_refuse_with_typed_errors(
    gas_pack: ImccGasDatapack,
    unit_activities: dict[str, float],
    species: str,
    temperature: float,
    source_species: str,
) -> None:
    assert set(IMCC_GAS_WORKBOOK_EXTRAPOLATION_LABELS) == {
        case[0] for case in _EXTRAPOLATION_REFUSAL_CASES
    }
    activities = (
        {**unit_activities, "Cr2O3": 1.0}
        if species in _CR_CHANNELS
        else unit_activities
    )
    with pytest.raises(ImccGasTemperatureOutsideDomainError) as exc:
        evaluate_gas(
            activities,
            temperature,
            1.0,
            gas_pack,
            gas_species=(species,),
            allow_extrapolation=False,
        )
    assert exc.value.code == "imcc_gas_T_outside_domain"
    assert source_species in str(exc.value)


def test_gas_domain_refusal_is_typed(gas_pack: ImccGasDatapack) -> None:
    """A selected gas-row gap still refuses before evaluating the reaction."""
    gas_df = gas_pack.gas_df.copy()
    # Keep O2 available at the diagnostic temperature so the refusal names the
    # selected Fe(g) row rather than the caller-pinned reference row.
    gas_df.loc["O2(g)", "T_min"] = 1000.0
    fe_low = (gas_df.index == "Fe(g)") & (gas_df["T_interval"].astype(int) == 2)
    gas_df.loc[fe_low, "T_max"] = 1300.0
    diagnostic_pack = ImccGasDatapack(
        gas_df=gas_df,
        oxide_df=gas_pack.oxide_df,
        gas_path=gas_pack.gas_path,
        oxide_path=gas_pack.oxide_path,
    )
    with pytest.raises(ImccGasTemperatureOutsideDomainError) as exc:
        evaluate_gas(
            {"FeO": 1.0},
            1400.0,
            fO2=1.0,
            datapack=diagnostic_pack,
            gas_species=("Fe",),
            allow_extrapolation=False,
        )
    assert exc.value.code == "imcc_gas_T_outside_domain"
    assert "Fe(g)" in str(exc.value)


def test_sf04_basalt_at_1800_predicts_and_flags_source_rows(
    gas_pack: ImccGasDatapack,
) -> None:
    from openimcc import evaluate as evaluate_imcc
    from openimcc import load_datapack

    composition = {
        "SiO2": 51.85068,
        "MgO": 4.78527,
        "FeO": 13.77307,
        "CaO": 9.02862,
        "Al2O3": 14.80572,
        "TiO2": 1.73824,
        "Na2O": 3.23108,
        "K2O": 0.78732,
    }
    imcc_pack = load_datapack(Path("src/openimcc/data/packs/imcc-sf04-v1.0.2.json"))
    imcc_result = evaluate_imcc(
        composition,
        T_K=1800.0,
        pack=imcc_pack,
        basis_type="wt",
        allow_extrapolation=True,
    )
    activities = dict(zip(imcc_result.parent_oxides, imcc_result.parent_activity))

    result = evaluate_gas(activities, 1800.0, 1.0e-10, gas_pack)

    assert isinstance(result, ImccGasResult)
    assert set(result) == set(_PRE_GATED_CHANNELS)
    assert all(math.isfinite(value) and value >= 0.0 for value in result.values())
    flagged = {
        "SiO",
        "SiO2",
        "Si",
        "Mg",
        "MgO",
        "CaO",
        "Ca",
        "AlO",
        "AlO2",
        "Al2O",
        "Al2O2",
        "Al2",
        "Al",
        "Si2",
        "Si3",
    }
    assert all(result.domain_flags[species] is not None for species in flagged)
    assert all(
        result.domain_flags[species] is None
        for species in set(result) - flagged
    )
    assert "MgO(l)" in result.domain_flags["Mg"]
    assert "[3100, 3500] K" in result.domain_flags["Mg"]
    assert "T=1800.0 K" in result.domain_flags["SiO"]


def test_o2_pin_is_never_domain_flagged_at_3200_k(
    gas_pack: ImccGasDatapack,
    unit_activities: dict[str, float],
) -> None:
    o2_only = evaluate_gas(
        unit_activities,
        3200.0,
        1.0e-8,
        gas_pack,
        gas_species=("O2",),
    )
    full = evaluate_gas(unit_activities, 3200.0, 1.0e-8, gas_pack)

    assert o2_only.domain_flags["O2"] is None
    assert full.domain_flags["O2"] is None
    # The same O2(g) row remains visible on reactions that actually consume it.
    assert "O2(g)" in full.domain_flags["SiO"]


def test_gas_result_is_frozen_bar_mapping_and_preserves_numbers(
    gas_pack: ImccGasDatapack,
) -> None:
    activities = {
        "SiO2": 1.0,
        "MgO": 1.0,
        "FeO": 1.0,
        "CaO": 1.0,
        "Al2O3": 1.0,
        "Na2O": 1.0,
        "K2O": 1.0,
    }
    result = evaluate_gas(
        activities,
        2000.0,
        1.0,
        gas_pack,
        allow_extrapolation=True,
        gas_species=("Na", "K", "SiO", "O2"),
    )

    assert result.unit == "bar"
    assert dict(result) == {
        "Na": 0.48713404611548444,
        "K": 2.1523860349175448,
        "SiO": 1.741275855555553e-08,
        "O2": 1.0,
    }
    assert result.domain_flags["Na"] is None
    assert result.provenance_class["Na"] == "lam1984_transcribed"
    with pytest.raises(TypeError):
        result.domain_flags["Na"] = "mutated"  # type: ignore[index]
    with pytest.raises((AttributeError, TypeError)):
        result.unit = "Pa"  # type: ignore[misc]


def test_corrected_condensates_leave_other_gas_channels_bit_identical(
    gas_pack: ImccGasDatapack,
) -> None:
    """Compare every other packaged channel with the 1197295 parent rows."""
    baseline = replace(gas_pack, oxide_df=gas_pack.oxide_df.copy(deep=True))
    baseline.oxide_df.loc["Al2O3(l)", "dH298_R"] = -188.14
    baseline.oxide_df.loc[
        "Na2O(l)", ["T_min", "dG_A", "dG_B", "dG_C", "dG_D", "dG_E"]
    ] = [825.0, 4.82, 19.292, -5.267, 0.623, 0.0]
    species = tuple(
        name
        for name in dict.fromkeys(
            (*IMCC_GAS_CHANNEL_SPECIES, *_CR_CHANNELS, *_VNB_CHANNELS)
        )
        if not name.startswith(("Na", "Al"))
        and (
            _SF04_REACTIONS[name][0] is None
            or f"{_SF04_REACTIONS[name][0]}(l)" in gas_pack.oxide_df.index
        )
    )
    activities = {parent: 1.0 for parent in _REACTION_PARENT_OXIDES} | {
        "Cr2O3": 1.0,
        "V2O3": 1.0e-3,
        "NbO2": 1.0e-3,
    }
    for temperature in range(500, 3001, 100):
        before = evaluate_gas(
            activities,
            float(temperature),
            1.0e-10,
            baseline,
            gas_species=species,
        )
        after = evaluate_gas(
            activities,
            float(temperature),
            1.0e-10,
            gas_pack,
            gas_species=species,
        )
        assert dict(after) == dict(before), temperature
        assert dict(after.domain_flags) == dict(before.domain_flags), temperature
        assert dict(after.provenance_class) == dict(before.provenance_class), temperature


def test_negative_fugacity_refuses_with_bar_relative_hint(
    gas_pack: ImccGasDatapack,
) -> None:
    with pytest.raises(
        ImccGasInvalidFugacityError,
        match=r"p_O2/p°.*bar-relative.*not log10 fO2",
    ) as exc:
        evaluate_gas(
            {"Na2O": 1.0},
            1800.0,
            -8.0,
            gas_pack,
            gas_species=("Na",),
        )
    assert exc.value.code == "imcc_gas_invalid_fO2"


def test_full_workbook_grid_runs_for_in_domain_channels(
    gas_pack: ImccGasDatapack,
    synthetic_mnnico_pack: ImccGasDatapack,
    unit_activities: dict[str, float],
) -> None:
    for species in IMCC_GAS_WORKBOOK_IN_DOMAIN_SPECIES:
        for temperature in IMCC_SF04_WORKBOOK_GRID_K:
            datapack = gas_pack
            activities = unit_activities
            if species in _CR_CHANNELS:
                activities = {**unit_activities, "Cr2O3": 1.0}
            elif species in _VNB_CHANNELS:
                activities = {**unit_activities, "V2O3": 1.0e-3, "NbO2": 1.0e-3}
            elif species in _MNNICO_CHANNELS:
                datapack = synthetic_mnnico_pack
                activities = {
                    **unit_activities,
                    "MnO": 1.0,
                    "NiO": 1.0,
                    "CoO": 1.0,
                }
            elif species in _S_CHANNELS:
                activities = {**unit_activities, "S2": 1.0e-3}
            result = evaluate_gas(
                activities,
                temperature,
                1.0,
                datapack,
                gas_species=(species,),
            )
            assert result[species] >= 0.0


def test_all_channels_compute_when_extrapolation_is_explicit(
    gas_pack: ImccGasDatapack, unit_activities: dict[str, float]
) -> None:
    result = evaluate_gas(
        {
            **unit_activities,
            "Cr2O3": 1.0,
            "V2O3": 1.0e-3,
            "NbO2": 1.0e-3,
        },
        2500.0,
        1.0e-10,
        gas_pack,
        allow_extrapolation=True,
    )
    assert set(result) == (
        set(_PRE_GATED_CHANNELS) | set(_CR_CHANNELS) | set(_VNB_CHANNELS)
    )
    assert all(value >= 0.0 for value in result.values())
    assert result["O2"] == 1.0e-10


def test_p1_g1_against_janaf_printed_gas_columns(
    gas_pack: ImccGasDatapack, unit_activities: dict[str, float]
) -> None:
    """The 2000 K rung-2b sample is independent of the fitted gas rows."""
    T = 2000.0
    pressures = evaluate_gas(
        unit_activities, T, fO2=1.0, datapack=gas_pack, allow_extrapolation=True
    )
    # Gate derivation: half the last stored digit is 0.0005 dex; the maximum
    # evaluate_gas residual against the unrounded reference is 5.2e-5 dex,
    # leaving 0.000448 dex of margin to 0.001 dex. Today's worst case is
    # 0.000508 dex (Si), so 0.0005 dex would false-fail. At 2000 K, 0.001 dex
    # is 38.3 J/mol on G_gas. These tests prove the evaluate_gas assembly and
    # stoichiometry path, including the condensate parent; they are not an
    # independent fit (G1 covers the fit).
    worst = 0.0
    nonzero_errors = 0
    for species, expected_log10 in _RUNG2B_AGAINST_JANAF_PRINTED_GAS_COLUMNS[T].items():
        actual_log10 = math.log10(pressures[species])
        error = abs(actual_log10 - expected_log10)
        worst = max(worst, error)
        nonzero_errors += error > 0.0
        assert error <= 0.001, (
            f"{species} at {T} K: |{actual_log10:.6f} - {expected_log10:.6f}| "
            f"= {error:.6f} dex > 0.001 dex"
        )
    assert nonzero_errors >= 1
    assert worst <= 0.001


def test_t625_against_janaf_printed_gas_columns(
    gas_pack: ImccGasDatapack, unit_activities: dict[str, float]
) -> None:
    """The T-625 additions agree with independently rounded printed lgK."""
    T = 2500.0
    refs = _T625_AGAINST_JANAF_PRINTED_GAS_COLUMNS[T]
    pressures = evaluate_gas(
        unit_activities,
        T,
        fO2=1.0,
        datapack=gas_pack,
        allow_extrapolation=True,
        gas_species=tuple(refs),
    )
    assert set(pressures) == set(refs)
    for species, expected_log10 in refs.items():
        actual_log10 = math.log10(pressures[species])
        error = abs(actual_log10 - expected_log10)
        assert 0.0 < error <= 0.001, (
            f"{species} at {T} K: independently rounded reference must have "
            f"nonzero error <= 0.001 dex, got {error:.6f}"
        )


def test_kp_derivation_spot_check_sio(gas_pack: ImccGasDatapack) -> None:
    """Recompute SiO2(l) -> SiO(g) + 1/2 O2(g) at 2000 K by hand."""
    R = 8.314462618
    T = 2000.0
    G_sio2_l = -1129878.93
    G_sio_g = -593626.03
    G_o2_g = -478320.38
    dG = G_sio_g + 0.5 * G_o2_g - G_sio2_l
    Kp_hand = math.exp(-dG / (R * T))

    p_sio = evaluate_gas(
        {"SiO2": 1.0}, T, fO2=1.0, datapack=gas_pack, allow_extrapolation=True
    )["SiO"]

    # The new fit's measured relative residual is 4.27e-5 against these
    # rounded hand values; retain the original 1e-4 check margin.
    assert p_sio == pytest.approx(Kp_hand, rel=1e-4)


def test_authority_query_carries_secondary_k2o_flag() -> None:
    assert gas_species_provenance("Na")["authority"] == "lam1984_transcribed"
    for species in ("K", "K2", "KO"):
        provenance = gas_species_provenance(species)
        assert provenance["authority"] == "secondary_transcription_unverified_primary"
        assert (
            provenance["condensate_authority"]
            == "secondary_transcription_unverified_primary"
        )
    assert gas_species_provenance("O2")["authority"] == "janaf_fitted"
    with pytest.raises(ImccGasSpeciesNotFoundError):
        gas_species_provenance("not-a-channel")


def test_activity_and_fugacity_scaling(
    gas_pack: ImccGasDatapack,
) -> None:
    # Premise: SiO2 -> SiO + 1/2 O2 has n_gas=1, so (5) is linear in a_SiO2.
    # Algebra: p(a/2) = K*(a/2) = p(a)/2 and p(2a) = 2*p(a). Unit check:
    # both activities and p/p° are dimensionless. Sanity: halving a unit
    # activity halves p_SiO.
    p_full = evaluate_gas(
        {"SiO2": 1.0}, 2000.0, 1.0, gas_pack, gas_species=("SiO",)
    )["SiO"]
    p_half = evaluate_gas(
        {"SiO2": 0.5}, 2000.0, 1.0, gas_pack, gas_species=("SiO",)
    )["SiO"]
    p_double = evaluate_gas(
        {"SiO2": 2.0}, 2000.0, 1.0, gas_pack, gas_species=("SiO",)
    )["SiO"]
    assert p_half == pytest.approx(0.5 * p_full, rel=1e-12)
    assert p_double == pytest.approx(2.0 * p_full, rel=1e-12)

    # Premise: K2O -> 2 K + 1/2 O2 has n_gas=2, so (5) is proportional to
    # a_K2O**(1/2). Algebra: p(a/2)/p(a) = sqrt(1/2), and
    # p(2a)/p(a) = sqrt(2). Unit check: the exponent applies to the
    # dimensionless parent activity, not a single-cation conversion. Sanity:
    # a_K2O=0.25 gives the ratios below exactly.
    p_k = evaluate_gas(
        {"K2O": 0.25}, 1800.0, 1.0e-10, gas_pack, gas_species=("K",)
    )["K"]
    p_k_half = evaluate_gas(
        {"K2O": 0.125}, 1800.0, 1.0e-10, gas_pack, gas_species=("K",)
    )["K"]
    p_k_double = evaluate_gas(
        {"K2O": 0.5}, 1800.0, 1.0e-10, gas_pack, gas_species=("K",)
    )["K"]
    assert p_k_half / p_k == pytest.approx(2.0**-0.5, rel=1e-12)
    assert p_k_double / p_k == pytest.approx(2.0**0.5, rel=1e-12)

    # Premise: Na2O -> 2 Na + 1/2 O2 has n_O2/n_gas=1/4. Algebra: changing
    # fO2 from 1 to 1e-4 multiplies p_Na by (1e-4)^(-1/4)=10. Unit check:
    # fO2 is p_O2/p°, hence dimensionless. Sanity: p_low = 10*p_ref.
    p_ref = evaluate_gas(
        {"Na2O": 1.0}, 2000.0, 1.0, gas_pack, allow_extrapolation=True
    )["Na"]
    p_low = evaluate_gas(
        {"Na2O": 1.0}, 2000.0, 1.0e-4, gas_pack, allow_extrapolation=True
    )["Na"]
    assert p_low == pytest.approx(10.0 * p_ref, rel=1e-12)


def test_runtime_evaluation_is_deterministic(
    gas_pack: ImccGasDatapack, unit_activities: dict[str, float]
) -> None:
    run1 = evaluate_gas(
        unit_activities,
        2000.0,
        1.0,
        datapack=gas_pack,
        allow_extrapolation=True,
    )
    run2 = evaluate_gas(
        unit_activities,
        2000.0,
        1.0,
        datapack=gas_pack,
        allow_extrapolation=True,
    )
    assert run1 == run2


def test_missing_gas_row_is_typed_refusal(gas_pack: ImccGasDatapack) -> None:
    stripped = gas_pack.gas_df.drop(index="K(g)")
    stripped_pack = ImccGasDatapack(
        gas_df=stripped,
        oxide_df=gas_pack.oxide_df,
        gas_path=gas_pack.gas_path,
        oxide_path=gas_pack.oxide_path,
    )
    with pytest.raises(ImccGasSpeciesNotFoundError) as exc:
        evaluate_gas(
            {"K2O": 1.0},
            2000.0,
            1.0,
            stripped_pack,
            gas_species=("K",),
            allow_extrapolation=True,
        )
    assert exc.value.code == "imcc_gas_species_not_found"


def test_gas_result_omitted_channels_default_is_empty_and_immutable() -> None:
    result = ImccGasResult({}, "bar", {}, {})

    assert result.omitted_channels == {}
    with pytest.raises(TypeError):
        result.omitted_channels["Na2O"] = "missing from active table: Na2O(g)"


def test_unavailable_species_ledger_names_the_closing_source(
    gas_pack: ImccGasDatapack,
) -> None:
    assert set(IMCC_GAS_NO_JANAF_ROWS) == {
        "Na+",
        "K+",
        "e-",
        "Zn",
        "ZnO",
    }
    assert IMCC_GAS_INCOMPLETE_PARENT_SPECIES == {}
    assert IMCC_GAS_PUBLIC_ONLY_PARENT_SPECIES == {
        "Mn": (
            "public pack has Mn(g) but no MnO(g) or MnO(l); "
            "external pack supplies both"
        ),
        "Ni": (
            "public pack has Ni(g) but no NiO(g) or NiO(l); "
            "external pack supplies both"
        ),
        "Co": (
            "public pack has Co(g) but no CoO(g) or CoO(l); "
            "external pack supplies both"
        ),
    }
    assert all(
        source.startswith("needs") or "; needs" in source
        for source in IMCC_GAS_UNAVAILABLE_SPECIES.values()
    )
    assert all(
        f"{species}(g)" not in gas_pack.gas_df.index
        for species in IMCC_GAS_NO_JANAF_ROWS
    )
    assert {"Mn(g)", "Ni(g)", "Co(g)"} <= set(gas_pack.gas_df.index)
    assert not {"MnO(g)", "NiO(g)", "CoO(g)"} & set(gas_pack.gas_df.index)
    assert set(_MNNICO_CHANNELS) <= set(IMCC_GAS_CHANNEL_SPECIES)
    assert {"Ti(g)", "TiO(g)", "TiO2(g)"} <= set(gas_pack.gas_df.index)
    assert "TiO2(l)" in gas_pack.oxide_df.index
    assert {"Ti", "TiO", "TiO2"} <= set(IMCC_GAS_CHANNEL_SPECIES)


def test_tio2_bearing_melt_returns_ti_pressures(
    gas_pack: ImccGasDatapack,
) -> None:
    """The quickstart basalt carries TiO2, so every Ti channel is evaluated."""
    from openimcc import evaluate as evaluate_imcc

    composition = {
        "SiO2": 51.85068,
        "MgO": 4.78527,
        "FeO": 13.77307,
        "CaO": 9.02862,
        "Al2O3": 14.80572,
        "TiO2": 1.73824,
        "Na2O": 3.23108,
        "K2O": 0.78732,
    }
    melt = evaluate_imcc(composition, 2200.0, basis_type="wt")
    activities = {name: melt.activity(name) for name in melt.parent_oxides}
    assert activities["TiO2"] > 0.0

    # Strict mode: 2200 K lies inside every Ti input row, so nothing refuses.
    result = evaluate_gas(
        activities,
        2200.0,
        1.0e-10,
        gas_pack,
        gas_species=("Ti", "TiO", "TiO2"),
        allow_extrapolation=False,
    )
    assert set(result) == {"Ti", "TiO", "TiO2"}
    for species in ("Ti", "TiO", "TiO2"):
        assert math.isfinite(result[species]) and result[species] > 0.0
        assert result.domain_flags[species] is None
        assert result.provenance_class[species] == "janaf_fitted"
        assert gas_species_provenance(species)["condensate_authority"] == (
            "janaf_fitted"
        )

    # Premise: all three channels share the TiO2(l) parent, so a_TiO2 cancels
    # in the ratios. Algebra: p_TiO/p_TiO2 = K_TiO/K_TiO2 * fO2^(-1/2) and
    # p_Ti/p_TiO = K_Ti/K_TiO * fO2^(-1/2). Unit check: every factor is
    # dimensionless. Sanity: raising fO2 by 1e4 lowers both ratios by 1e2.
    richer = evaluate_gas(
        activities,
        2200.0,
        1.0e-6,
        gas_pack,
        gas_species=("Ti", "TiO", "TiO2"),
    )
    assert richer["TiO2"] == pytest.approx(result["TiO2"], rel=1e-12)
    assert (result["TiO"] / result["TiO2"]) / (
        richer["TiO"] / richer["TiO2"]
    ) == pytest.approx(100.0, rel=1e-9)
    assert (result["Ti"] / result["TiO"]) / (
        richer["Ti"] / richer["TiO"]
    ) == pytest.approx(100.0, rel=1e-9)


_TI_CHANNELS = ("Ti", "TiO", "TiO2")
_CR_CHANNELS = ("Cr", "CrO", "CrO2", "CrO3")
_VNB_CHANNELS = ("V", "VO", "VO2", "Nb", "NbO", "NbO2")
_MNNICO_CHANNELS = ("Mn", "MnO", "Ni", "NiO", "Co", "CoO")
_P_CHANNELS = ("P", "P2", "P4", "PO", "PO2", "P4O6", "P4O10")
_S_CHANNELS = (
    "S", "S2", "S3", "S4", "S5", "S6", "S7", "S8",
    "SO", "SO2", "SO3", "SSO",
)
_PS_CHANNELS = _P_CHANNELS + _S_CHANNELS
_MNNICO_PARENT_PAIRS = (("Mn", "MnO"), ("Ni", "NiO"), ("Co", "CoO"))
_PRE_GATED_CHANNELS = tuple(
    species
    for species in IMCC_GAS_CHANNEL_SPECIES
    if species not in _CR_CHANNELS
    and species not in _VNB_CHANNELS
    and species not in _MNNICO_CHANNELS
    and species not in _PS_CHANNELS
)
_SF04_CHANNELS = tuple(
    species for species in _PRE_GATED_CHANNELS if species not in _TI_CHANNELS
)


def _quickstart_activities(
    T: float, *, allow_extrapolation: bool = False
) -> dict[str, float]:
    from openimcc import evaluate as evaluate_imcc

    basalt = {
        "SiO2": 51.85068,
        "MgO": 4.78527,
        "FeO": 13.77307,
        "CaO": 9.02862,
        "Al2O3": 14.80572,
        "TiO2": 1.73824,
        "Na2O": 3.23108,
        "K2O": 0.78732,
    }
    melt = evaluate_imcc(
        basalt,
        T,
        basis_type="wt",
        allow_extrapolation=allow_extrapolation,
    )
    return {name: melt.activity(name) for name in melt.parent_oxides}


def test_readme_basalt_outputs_match_interval_1_at_and_above_1500_k(
    gas_pack: ImccGasDatapack,
) -> None:
    high_only = replace(
        gas_pack,
        gas_df=gas_pack.gas_df.loc[
            gas_pack.gas_df["T_interval"].astype(int) == 1
        ].copy(),
    )
    for temperature in (1500.0, 1800.0, 2200.0, 2600.0):
        activities = _quickstart_activities(
            temperature,
            allow_extrapolation=True,
        )
        for fO2 in (1.0e-10, 1.0e-6):
            current = evaluate_gas(
                activities,
                temperature,
                fO2,
                gas_pack,
                allow_extrapolation=True,
            )
            interval_1_only = evaluate_gas(
                activities,
                temperature,
                fO2,
                high_only,
                allow_extrapolation=True,
            )
            assert current.unit == interval_1_only.unit
            assert tuple(current) == tuple(interval_1_only)
            assert {
                species: float(current[species]).hex() for species in current
            } == {
                species: float(interval_1_only[species]).hex()
                for species in interval_1_only
            }
            assert dict(current.domain_flags) == dict(interval_1_only.domain_flags)
            assert dict(current.provenance_class) == dict(
                interval_1_only.provenance_class
            )


_OPTIONAL_PS_GAS_SPECIES = (
    "S", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "SO", "SO2", "SO3", "SSO",
    "P", "P2", "P4", "PO", "PO2", "P4O6", "P4O10",
)


def test_optional_ps_rows_do_not_change_default_outputs(
    gas_pack: ImccGasDatapack,
) -> None:
    """Default outputs are identical with and without the optional P/S rows.

    Compared within one environment, so the check is exact without pinning
    platform-dependent bit patterns.
    """
    drop = [f"{name}(g)" for name in _OPTIONAL_PS_GAS_SPECIES]
    assert set(drop) <= set(gas_pack.gas_df.index)
    without_ps = replace(
        gas_pack, gas_df=gas_pack.gas_df.drop(index=drop)
    )
    for temperature in (1500.0, 1800.0, 2200.0, 2600.0):
        activities = _quickstart_activities(
            temperature, allow_extrapolation=True
        )
        for fO2 in (1.0e-10, 1.0e-6):
            full = evaluate_gas(
                activities, temperature, fO2, gas_pack, allow_extrapolation=True
            )
            base = evaluate_gas(
                activities, temperature, fO2, without_ps, allow_extrapolation=True
            )
            assert dict(full.items()) == dict(base.items())
            assert dict(full.domain_flags) == dict(base.domain_flags)
            assert dict(full.provenance_class) == dict(base.provenance_class)
            assert dict(full.omitted_channels) == dict(base.omitted_channels)
            assert not set(_OPTIONAL_PS_GAS_SPECIES) & set(full)


def _janaf_formation_gibbs(table_id: str, temperature: float) -> float:
    source = build_gas_tables._load_record(
        Path(__file__).resolve().parents[1] / "data-src" / "janaf" / f"{table_id}.yaml"
    )
    row = next(
        row
        for row in source["table"]["values"]
        if float(row["temperature"]["value"]) == temperature
    )
    return float(row["formation_gibbs_energy"]["value"]) * 1000.0


def test_sulfur_pressures_match_hand_calculated_janaf_cells(
    gas_pack: ImccGasDatapack,
) -> None:
    """Use JANAF G° cells and nS2*S2 + nO2*O2 = gas, without fitted rows."""
    product_tables = {"S": "S-006", "S2": "S-012", "SO": "O-010", "SO2": "O-034"}
    parent_activity = 1.0e-3
    species = tuple(product_tables)
    for temperature in (1800.0, 2200.0):
        G_S2 = _janaf_formation_gibbs("S-012", temperature)
        G_O2 = _janaf_formation_gibbs("O-029", temperature)
        for fO2 in (1.0e-10, 1.0e-6):
            result = evaluate_gas(
                {"S2": parent_activity},
                temperature,
                fO2,
                gas_pack,
                parent_oxides=(*IMCC_PARENT_OXIDES, "S2"),
                gas_species=species,
                allow_extrapolation=False,
            )
            for gas in species:
                _parent, n_S2, n_O2 = _SF04_REACTIONS[gas]
                dG = (
                    _janaf_formation_gibbs(product_tables[gas], temperature)
                    - n_S2 * G_S2
                    - n_O2 * G_O2
                )
                Kp = math.exp(-dG / (R_J_MOL_K * temperature))
                hand_pressure = (
                    Kp * parent_activity**n_S2 * fO2**n_O2
                )
                assert result[gas] == pytest.approx(hand_pressure, rel=1.5e-3)


def test_sulfur_channels_have_the_declared_fugacity_exponents(
    gas_pack: ImccGasDatapack,
) -> None:
    common = {
        "activities": {"S2": 1.0e-3},
        "T_K": 2200.0,
        "datapack": gas_pack,
        "parent_oxides": (*IMCC_PARENT_OXIDES, "S2"),
        "gas_species": _S_CHANNELS,
    }
    low = evaluate_gas(fO2=1.0e-10, **common)
    high = evaluate_gas(fO2=1.0e-6, **common)
    for gas in _S_CHANNELS:
        _parent, _n_S2, n_O2 = _SF04_REACTIONS[gas]
        assert high[gas] / low[gas] == pytest.approx(1.0e4**n_O2, rel=1e-11)


def test_phosphorus_and_sulfur_channels_are_optional_and_refuse_without_parent(
    gas_pack: ImccGasDatapack,
) -> None:
    activities = _quickstart_activities(2200.0, allow_extrapolation=True)
    default = evaluate_gas(
        activities, 2200.0, 1.0e-10, gas_pack, allow_extrapolation=True
    )
    assert tuple(default) == _PRE_GATED_CHANNELS
    assert not set(_PS_CHANNELS) & set(default)
    assert not set(_PS_CHANNELS) & set(default.omitted_channels)

    p_parent_only = evaluate_gas(
        {**activities, "P2O5": 1.0e-3},
        2200.0,
        1.0e-10,
        gas_pack,
        allow_extrapolation=True,
    )
    assert not set(_P_CHANNELS) & set(p_parent_only)
    assert set(_P_CHANNELS) <= set(p_parent_only.omitted_channels)
    assert all(
        "P2O5(l)" in p_parent_only.omitted_channels[species]
        for species in _P_CHANNELS
    )

    with pytest.raises(ImccGasSpeciesNotFoundError) as p_error:
        evaluate_gas(
            {**activities, "P2O5": 1.0e-3},
            2200.0,
            1.0e-10,
            gas_pack,
            parent_oxides=(*IMCC_PARENT_OXIDES, "P2O5"),
            gas_species=("PO",),
            allow_extrapolation=True,
        )
    assert p_error.value.code == "imcc_gas_species_not_found"
    assert "P2O5(l)" in str(p_error.value)

    for species in ("PO", "P2", "S", "S2"):
        with pytest.raises(ImccGasSpeciesNotFoundError):
            evaluate_gas(
                activities,
                2200.0,
                1.0e-10,
                gas_pack,
                gas_species=(species,),
                allow_extrapolation=True,
            )


@pytest.fixture
def synthetic_mnnico_pack(
    gas_pack: ImccGasDatapack, tmp_path: Path
) -> ImccGasDatapack:
    """Synthetic wiring fixture, not thermodynamic data.

    It copies the public Fe(g), FeO(g), and FeO(l) rows under Mn/Ni/Co names so
    the tests exercise optional-channel wiring with an external pack.
    """
    gas_df = pd.read_csv(gas_pack.gas_path)
    oxide_df = pd.read_csv(gas_pack.oxide_path)
    atomic_species = {f"{gas}(g)" for gas, _parent in _MNNICO_PARENT_PAIRS}
    gas_df = gas_df.loc[~gas_df["species_name"].isin(atomic_species)]
    atomic_gas_template = gas_df.loc[gas_df["species_name"] == "Fe(g)"].iloc[[0]]
    monoxide_gas_template = gas_df.loc[gas_df["species_name"] == "FeO(g)"].iloc[[0]]
    oxide_template = oxide_df.loc[oxide_df["species_name"] == "FeO(l)"].iloc[[0]]
    gas_rows = [
        atomic_gas_template.assign(species_name=f"{gas}(g)")
        for gas, _parent in _MNNICO_PARENT_PAIRS
    ]
    gas_rows.extend(
        monoxide_gas_template.assign(species_name=f"{parent}(g)")
        for _gas, parent in _MNNICO_PARENT_PAIRS
    )
    oxide_rows = [
        oxide_template.assign(species_name=f"{parent}(l)")
        for _gas, parent in _MNNICO_PARENT_PAIRS
    ]
    gas_path = tmp_path / "synthetic-gas.csv"
    oxide_path = tmp_path / "synthetic-condensate.csv"
    pd.concat([gas_df, *gas_rows], ignore_index=True).to_csv(gas_path, index=False)
    pd.concat([oxide_df, *oxide_rows], ignore_index=True).to_csv(
        oxide_path, index=False
    )
    return load_gas_datapack(gas_path=gas_path, oxide_path=oxide_path)


def test_public_pack_skips_mnnico_channels_and_explicit_requests_refuse(
    gas_pack: ImccGasDatapack,
) -> None:
    parent_oxides = (*IMCC_PARENT_OXIDES, "MnO", "NiO", "CoO")
    activities = {name: 1.0 for name in parent_oxides}
    result = evaluate_gas(
        activities,
        2200.0,
        1.0e-10,
        gas_pack,
        parent_oxides=parent_oxides,
    )
    assert not set(_MNNICO_CHANNELS) & set(result)

    for gas, parent in _MNNICO_PARENT_PAIRS:
        with pytest.raises(ImccGasSpeciesNotFoundError) as exc:
            evaluate_gas(
                {parent: 1.0},
                2200.0,
                1.0e-10,
                gas_pack,
                parent_oxides=(parent,),
                gas_species=(gas,),
            )
        assert exc.value.code == "imcc_gas_species_not_found"
        assert f"{parent}(l)" in str(exc.value)


def test_external_pack_activates_mnnico_channels_with_data_free_scaling(
    synthetic_mnnico_pack: ImccGasDatapack,
) -> None:
    activities = {parent: 1.0 for _gas, parent in _MNNICO_PARENT_PAIRS}
    common = {
        "activities": activities,
        "T_K": 2200.0,
        "datapack": synthetic_mnnico_pack,
        "parent_oxides": tuple(activities),
        "gas_species": _MNNICO_CHANNELS,
        "allow_extrapolation": False,
    }
    low_fugacity = evaluate_gas(fO2=1.0e-10, **common)
    high_fugacity = evaluate_gas(fO2=1.0e-6, **common)

    assert tuple(low_fugacity) == _MNNICO_CHANNELS
    assert all(
        math.isfinite(low_fugacity[name]) and low_fugacity[name] > 0.0
        for name in _MNNICO_CHANNELS
    )
    for gas, monoxide in _MNNICO_PARENT_PAIRS:
        assert low_fugacity.provenance_class[gas] == "external_datapack"
        assert low_fugacity.provenance_class[monoxide] == "external_datapack"
        # MO(l) = M(g) + 1/2 O2: p(M) scales as fO2**(-1/2).
        assert low_fugacity[gas] / high_fugacity[gas] == pytest.approx(100.0)
        # MO(l) = MO(g): n_O2 = 0, so its pressure is fO2-independent.
        assert low_fugacity[monoxide] == high_fugacity[monoxide]


def test_override_without_tio2_parent_keeps_the_sf04_default_set(
    monkeypatch: pytest.MonkeyPatch,
    gas_pack: ImccGasDatapack,
    tmp_path: Path,
) -> None:
    """Legacy tables with Ti gas rows but no TiO2(l) row: no default refusal."""
    root = tmp_path / "vaporock"
    gas_target = root / "src" / "vaporock" / "data"
    oxide_target = root / "data"
    gas_target.mkdir(parents=True)
    oxide_target.mkdir(parents=True)
    shutil.copy2(gas_pack.gas_path, gas_target / "JANAF-vapor-data-full.csv")
    condensate = gas_pack.oxide_path.read_text(encoding="utf-8").splitlines(True)
    (oxide_target / "condensate-thermo-data.csv").write_text(
        "".join(line for line in condensate if not line.startswith("TiO2(l),")),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENIMCC_VAPOROCK_ROOT", str(root))
    legacy = load_gas_datapack()
    assert "TiO2(l)" not in legacy.oxide_df.index
    assert {"Ti(g)", "TiO(g)", "TiO2(g)"} <= set(legacy.gas_df.index)

    activities = _quickstart_activities(2200.0)
    result = evaluate_gas(activities, 2200.0, 1.0e-10, legacy)
    packaged = evaluate_gas(activities, 2200.0, 1.0e-10, gas_pack)
    assert len(_SF04_CHANNELS) == 27
    assert tuple(result) == _SF04_CHANNELS
    # Same gas rows and SF04 parent rows, so every value is bit-identical.
    assert dict(result) == {species: packaged[species] for species in _SF04_CHANNELS}
    assert dict(result.domain_flags) == {
        species: packaged.domain_flags[species] for species in _SF04_CHANNELS
    }
    assert result.omitted_channels == {
        species: "missing from active table: TiO2(l)"
        for species in _TI_CHANNELS
    }

    # Naming a Ti channel is an explicit request, so it still refuses, typed.
    for species in _TI_CHANNELS:
        with pytest.raises(ImccGasSpeciesNotFoundError) as exc:
            evaluate_gas(
                activities, 2200.0, 1.0e-10, legacy, gas_species=(species,)
            )
        assert exc.value.code == "imcc_gas_species_not_found"
        assert "TiO2(l)" in str(exc.value)


def _gas_pack_without_gas_rows(
    gas_pack: ImccGasDatapack, rows: set[str], tmp_path: Path
) -> ImccGasDatapack:
    source_lines = gas_pack.gas_path.read_text(encoding="utf-8").splitlines(True)
    removed = {line.partition(",")[0] for line in source_lines} & rows
    assert removed == rows
    gas_path = tmp_path / "gas-without-optional-rows.csv"
    gas_path.write_text(
        "".join(line for line in source_lines if line.partition(",")[0] not in rows),
        encoding="utf-8",
    )
    return load_gas_datapack(gas_path=gas_path, oxide_path=gas_pack.oxide_path)


@pytest.mark.parametrize(
    ("removed_rows", "omitted_channels"),
    [
        (("Na2O(g)", "K2O(g)"), ("Na2O", "K2O")),
        (("Al2(g)", "Si3(g)"), ("Al2", "Si3")),
    ],
)
def test_default_channels_report_missing_optional_gas_rows(
    gas_pack: ImccGasDatapack,
    tmp_path: Path,
    removed_rows: tuple[str, ...],
    omitted_channels: tuple[str, ...],
) -> None:
    alternate = _gas_pack_without_gas_rows(gas_pack, set(removed_rows), tmp_path)
    activities = _quickstart_activities(2200.0)
    packaged = evaluate_gas(activities, 2200.0, 1.0e-8, gas_pack)
    result = evaluate_gas(activities, 2200.0, 1.0e-8, alternate)

    expected_omissions = {
        name: f"missing from active table: {name}(g)"
        for name in omitted_channels
    }
    assert dict(result.omitted_channels) == expected_omissions
    assert {
        name: value.hex() for name, value in result.items()
    } == {
        name: value.hex()
        for name, value in packaged.items()
        if name not in expected_omissions
    }
    assert result.domain_flags == {
        name: packaged.domain_flags[name]
        for name in packaged
        if name not in expected_omissions
    }
    assert result.provenance_class == {
        name: packaged.provenance_class[name]
        for name in packaged
        if name not in expected_omissions
    }

    parents_without_missing_channels = tuple(
        parent
        for parent in activities
        if parent not in {"Na2O", "K2O", "Al2O3", "SiO2"}
    )
    without_parent_activity = evaluate_gas(
        activities,
        2200.0,
        1.0e-8,
        alternate,
        parent_oxides=parents_without_missing_channels,
    )
    assert without_parent_activity.omitted_channels == {}

    if "Na2O" in omitted_channels:
        with pytest.raises(ImccGasSpeciesNotFoundError) as exc:
            evaluate_gas(
                {"Na2O": 1.0}, 2200.0, 1.0e-8, alternate,
                gas_species=("Na2O",),
            )
        assert exc.value.code == "imcc_gas_species_not_found"


def test_oxygen_balance_models_omit_missing_alkali_gas_rows(
    gas_pack: ImccGasDatapack, tmp_path: Path
) -> None:
    alternate = _gas_pack_without_gas_rows(
        gas_pack, {"Na2O(g)", "K2O(g)"}, tmp_path
    )
    activities = _quickstart_activities(2200.0)
    expected_omissions = {
        "Na2O": "missing from active table: Na2O(g)",
        "K2O": "missing from active table: K2O(g)",
    }

    p_o2, pressures, _diagnostics = evaluate_gas_oxygen_balance(
        activities, 2200.0, alternate
    )
    assert p_o2 > 0.0
    assert dict(pressures.omitted_channels) == expected_omissions

    species = oxygen_balance_species_metadata(
        {"O": None, "O2": None, "SiO": "SiO2"}
    )
    p_o2, pressures, _diagnostics = oxygen_balance_from_pressure_model(
        lambda logp: evaluate_gas(
            activities, 2200.0, 10.0**logp, alternate
        ),
        species,
    )
    assert p_o2 > 0.0
    assert dict(pressures.omitted_channels) == expected_omissions


@pytest.mark.skipif(
    not os.environ.get("OPENIMCC_TEST_LEGACY_VAPOROCK_ROOT"),
    reason="set OPENIMCC_VAPOROCK_ROOT to a local VapoRock checkout",
)
def test_vaporock_default_reports_missing_alkali_channels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "OPENIMCC_VAPOROCK_ROOT",
        os.environ["OPENIMCC_TEST_LEGACY_VAPOROCK_ROOT"],
    )
    datapack = load_gas_datapack()
    activities = _quickstart_activities(2200.0)
    missing_alkali = {}
    for name in ("Na2O", "K2O"):
        missing_rows = []
        if f"{name}(g)" not in datapack.gas_df.index:
            missing_rows.append(f"{name}(g)")
        if f"{name}(l)" not in datapack.oxide_df.index:
            missing_rows.append(f"{name}(l)")
        if missing_rows:
            missing_alkali[name] = (
                "missing from active table: " + ", ".join(missing_rows)
            )
    if not missing_alkali:
        pytest.skip("the selected VapoRock table provides both alkali channels")

    result = evaluate_gas(activities, 2200.0, 1.0e-8, datapack)
    assert {
        name: result.omitted_channels[name] for name in missing_alkali
    } == missing_alkali


def test_default_set_follows_parent_oxides_without_key_errors(
    gas_pack: ImccGasDatapack,
) -> None:
    activities = _quickstart_activities(2200.0)
    packaged = evaluate_gas(activities, 2200.0, 1.0e-10, gas_pack)
    assert packaged.omitted_channels == {}
    # The packaged default appends Ti, then the screened association channels
    # and the Na2O/K2O identity channels; Cr, V and Nb stay out until their
    # caller-supplied parent activities are present.
    assert tuple(packaged) == _PRE_GATED_CHANNELS
    assert tuple(packaged)[-8:] == (
        *_TI_CHANNELS,
        "Al2",
        "Si2",
        "Si3",
        "Na2O",
        "K2O",
    )

    seven = tuple(name for name in activities if name != "TiO2")
    for supplied in (
        activities,
        [activities[name] for name in seven],
    ):
        result = evaluate_gas(
            supplied, 2200.0, 1.0e-10, gas_pack, parent_oxides=seven
        )
        assert tuple(result) == _SF04_CHANNELS
        assert result.omitted_channels == {}
        assert dict(result) == {
            species: packaged[species] for species in _SF04_CHANNELS
        }

    # The same rule covers every parent: only channels it can serve.
    silica_only = evaluate_gas(
        {"SiO2": 0.5}, 2200.0, 1.0e-10, gas_pack, parent_oxides=("SiO2",)
    )
    assert tuple(silica_only) == ("SiO", "SiO2", "O", "Si", "O2", "Si2", "Si3")

    # An explicit channel whose parent is not supplied refuses, typed.
    for species, parents in (("Ti", seven), ("Na", ("SiO2",))):
        with pytest.raises(ImccGasSpeciesNotFoundError) as exc:
            evaluate_gas(
                activities,
                2200.0,
                1.0e-10,
                gas_pack,
                parent_oxides=parents,
                gas_species=(species,),
            )
        assert exc.value.code == "imcc_gas_species_not_found"
        assert "not in parent_oxides" in str(exc.value)


def test_tier_a_screened_channels_return_positive_pressures(
    gas_pack: ImccGasDatapack,
) -> None:
    from openimcc import evaluate as evaluate_imcc

    composition = {
        "SiO2": 51.85068,
        "MgO": 4.78527,
        "FeO": 13.77307,
        "CaO": 9.02862,
        "Al2O3": 14.80572,
        "TiO2": 1.73824,
        "Na2O": 3.23108,
        "K2O": 0.78732,
    }
    melt = evaluate_imcc(composition, 2600.0, basis_type="wt")
    activities = {name: melt.activity(name) for name in melt.parent_oxides}
    species = ("Al2", "Si2", "Si3")
    low_fugacity = evaluate_gas(
        activities,
        2600.0,
        1.0e-10,
        gas_pack,
        gas_species=species,
        allow_extrapolation=False,
    )
    high_fugacity = evaluate_gas(
        activities,
        2600.0,
        1.0e-6,
        gas_pack,
        gas_species=species,
        allow_extrapolation=False,
    )
    for name in species:
        assert math.isfinite(low_fugacity[name]) and low_fugacity[name] > 0.0
        assert math.isfinite(high_fugacity[name]) and high_fugacity[name] > 0.0

    # Premise: (5) gives p_gas proportional to fO2^(-n_O2/n_gas) at fixed
    # melt activity and temperature. Algebra: the Al2, Si2, and Si3 tuples
    # have n_O2/n_gas = 3/2, 1/(1/2)=2, and 1/(1/3)=3, respectively. Unit
    # check: the fugacity ratio and pressure ratio are dimensionless. Sanity:
    # lowering fO2 by 1e4 raises these pressures by 1e6, 1e8, and 1e12.
    assert low_fugacity["Al2"] / high_fugacity["Al2"] == pytest.approx(
        1.0e6, rel=1e-12
    )
    assert low_fugacity["Si2"] / high_fugacity["Si2"] == pytest.approx(
        1.0e8, rel=1e-12
    )
    assert low_fugacity["Si3"] / high_fugacity["Si3"] == pytest.approx(
        1.0e12, rel=1e-12
    )


def test_caller_supplied_cr2o3_activity_returns_positive_cr_pressures(
    gas_pack: ImccGasDatapack,
) -> None:
    from openimcc import evaluate as evaluate_imcc

    composition = {
        "SiO2": 51.85068,
        "MgO": 4.78527,
        "FeO": 13.77307,
        "CaO": 9.02862,
        "Al2O3": 14.80572,
        "TiO2": 1.73824,
        "Na2O": 3.23108,
        "K2O": 0.78732,
    }
    melt = evaluate_imcc(composition, 2200.0, basis_type="wt")
    activities = {name: melt.activity(name) for name in melt.parent_oxides}
    assert "Cr2O3" not in IMCC_PARENT_OXIDES
    assert set(evaluate_gas(activities, 2200.0, 1.0e-10, gas_pack)) == set(
        _PRE_GATED_CHANNELS
    )

    activities["Cr2O3"] = 1.0e-3
    result = evaluate_gas(
        activities,
        2200.0,
        1.0e-10,
        gas_pack,
        gas_species=_CR_CHANNELS,
        allow_extrapolation=False,
    )
    assert set(result) == set(_CR_CHANNELS)
    for species in _CR_CHANNELS:
        assert math.isfinite(result[species]) and result[species] > 0.0
        assert result.domain_flags[species] is None
        assert result.provenance_class[species] == "janaf_fitted"


def test_cr_atomic_row_flags_or_refuses_above_declared_endpoint(
    gas_pack: ImccGasDatapack,
) -> None:
    with pytest.raises(ImccGasTemperatureOutsideDomainError, match="Cr\\(g\\)"):
        evaluate_gas(
            {"Cr2O3": 1.0e-3},
            2950.0,
            1.0e-10,
            gas_pack,
            gas_species=("Cr",),
            allow_extrapolation=False,
        )

    result = evaluate_gas(
        {"Cr2O3": 1.0e-3},
        2950.0,
        1.0e-10,
        gas_pack,
        gas_species=("Cr",),
        allow_extrapolation=True,
    )
    assert math.isfinite(result["Cr"]) and result["Cr"] > 0.0
    assert result.domain_flags["Cr"] == (
        "T=2950.0 K outside declared G(T) interval for 'Cr(g)' [1500, 2900] K"
    )


def test_cr_liquid_parent_flags_or_refuses_below_declared_start(
    gas_pack: ImccGasDatapack,
) -> None:
    with pytest.raises(ImccGasTemperatureOutsideDomainError, match="Cr2O3\\(l\\)"):
        evaluate_gas(
            {"Cr2O3": 1.0e-3},
            1500.0,
            1.0e-10,
            gas_pack,
            gas_species=_CR_CHANNELS,
            allow_extrapolation=False,
        )

    result = evaluate_gas(
        {"Cr2O3": 1.0e-3},
        1500.0,
        1.0e-10,
        gas_pack,
        gas_species=_CR_CHANNELS,
        allow_extrapolation=True,
    )
    assert all(
        math.isfinite(result[name]) and result[name] > 0.0 for name in _CR_CHANNELS
    )
    assert all(
        result.domain_flags[name]
        == "T=1500.0 K outside declared G(T) interval for 'Cr2O3(l)' [1900, 3000] K"
        for name in _CR_CHANNELS
    )


def test_caller_supplied_v2o3_and_nbo2_activities_return_positive_pressures(
    gas_pack: ImccGasDatapack,
) -> None:
    species = ("V", "VO", "VO2", "Nb", "NbO", "NbO2")
    for temperature in (1800.0, 2200.0, 2600.0):
        for fugacity in (1.0e-10, 1.0e-6):
            result = evaluate_gas(
                {"V2O3": 1.0e-3, "NbO2": 1.0e-3},
                temperature,
                fugacity,
                gas_pack,
                gas_species=species,
                allow_extrapolation=False,
            )
            assert all(
                math.isfinite(result[name]) and result[name] > 0.0
                for name in species
            )
            assert all(result.domain_flags[name] is None for name in species)


def test_vanadium_and_niobium_channels_match_printed_janaf_gibbs(
    gas_pack: ImccGasDatapack,
) -> None:
    T = 2000.0
    refs = _VNB_AGAINST_JANAF_PRINTED_FORMATION_GIBBS[T]
    pressures = evaluate_gas(
        {"V2O3": 1.0e-3, "NbO2": 1.0e-3},
        T,
        fO2=1.0,
        datapack=gas_pack,
        gas_species=tuple(refs),
        allow_extrapolation=False,
    )
    for species, expected_log10 in refs.items():
        actual_log10 = math.log10(pressures[species])
        assert abs(actual_log10 - expected_log10) <= 0.001, (
            f"{species} at {T} K: |{actual_log10:.6f} - {expected_log10:.3f}| "
            "> 0.001 dex"
        )


def test_ti_channels_against_janaf_printed_formation_gibbs(
    gas_pack: ImccGasDatapack,
) -> None:
    """Ti channels agree with independently rounded JANAF formation Gibbs."""
    T = 2000.0
    refs = _TI_AGAINST_JANAF_PRINTED_FORMATION_GIBBS[T]
    pressures = evaluate_gas(
        {"TiO2": 1.0},
        T,
        fO2=1.0,
        datapack=gas_pack,
        gas_species=tuple(refs),
        allow_extrapolation=False,
    )
    # Gate: 0.001 dex is 38.3 J/mol at 2000 K. The unrounded hand values sit
    # 1.5e-4 dex from the model, mostly the TiO2(l) fit's +5.5 J/mol residual
    # at this node; the three-decimal rounding adds at most 5e-4 dex.
    for species, expected_log10 in refs.items():
        actual_log10 = math.log10(pressures[species])
        assert abs(actual_log10 - expected_log10) <= 0.001, (
            f"{species} at {T} K: |{actual_log10:.6f} - {expected_log10:.3f}| "
            "> 0.001 dex"
        )

    # The TiO/TiO2 balance at a vacuum-like fO2 = 1e-10: the hand value is
    # 10**(-11.969 + 6.560 + 5) = 10**-0.409 = 0.390.
    vacuum = evaluate_gas(
        {"TiO2": 1.0}, T, 1.0e-10, gas_pack, gas_species=("TiO", "TiO2")
    )
    assert math.log10(vacuum["TiO"] / vacuum["TiO2"]) == pytest.approx(
        -0.409, abs=0.001
    )


def test_imcc_adapter_activities_reach_all_channels(
    gas_pack: ImccGasDatapack,
) -> None:
    from openimcc import evaluate as evaluate_imcc
    from openimcc import load_datapack

    imcc_pack = load_datapack(Path("src/openimcc/data/packs/imcc-sf04-v1.0.2.json"))
    composition = {
        "SiO2": 71.39,
        "MgO": 0.27,
        "FeO": 0.04,
        "CaO": 10.75,
        "Al2O3": 2.78,
        "TiO2": 0.0,
        "Na2O": 12.75,
        "K2O": 2.02,
    }
    imcc_result = evaluate_imcc(
        composition,
        T_K=2500.0,
        pack=imcc_pack,
        basis_type="wt",
        allow_extrapolation=True,
    )
    activities = dict(zip(imcc_result.parent_oxides, imcc_result.parent_activity))
    result = evaluate_gas(
        activities,
        2500.0,
        1.0e-10,
        gas_pack,
        allow_extrapolation=True,
    )
    assert set(result) == set(_PRE_GATED_CHANNELS)
    assert all(value >= 0.0 for value in result.values())
    assert result["O2"] == 1.0e-10
    assert result["Na"] > 0.0
    assert result["K"] > 0.0


def test_gas_imports_without_pandas_and_refuses_at_load() -> None:
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                [
                    "import sys",
                    'sys.modules["pandas"] = None',
                    "import openimcc.gas as gas",
                    "from openimcc import ImccRefusal",
                    "try:",
                    "    gas.load_gas_datapack()",
                    "except ImccRefusal as exc:",
                    '    assert \'install "openimcc[gas]" (pandas import failed:\' in str(exc)',
                    "else:",
                    "    raise AssertionError(\"load_gas_datapack unexpectedly succeeded\")",
                ]
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        },
    )
    assert probe.returncode == 0, probe.stderr


def _unread_gas_pack() -> ImccGasDatapack:
    return ImccGasDatapack(
        gas_df=pd.DataFrame(),
        oxide_df=pd.DataFrame(),
        gas_path=Path("."),
        oxide_path=Path("."),
    )


@pytest.mark.parametrize("temperature", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_temperature_refuses_before_table_access(temperature: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        evaluate_gas(
            {"Na2O": 1.0},
            temperature,
            1.0,
            _unread_gas_pack(),
            gas_species=("Na",),
        )


@pytest.mark.parametrize("fugacity", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_fugacity_refuses_before_table_access(fugacity: float) -> None:
    with pytest.raises(
        ImccGasInvalidFugacityError,
        match=r"finite and positive p_O2/p°.*not log10 fO2",
    ):
        evaluate_gas(
            {"Na2O": 1.0},
            2000.0,
            fugacity,
            _unread_gas_pack(),
            gas_species=("Na",),
        )


@pytest.mark.parametrize("activity", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_activity_refuses(activity: float) -> None:
    with pytest.raises(ValueError, match="finite and >= 0"):
        evaluate_gas(
            {"Na2O": activity},
            2000.0,
            1.0,
            _unread_gas_pack(),
            gas_species=("Na",),
        )


def test_pure_silica_oxygen_balance_matches_analytic_flux_limit(
    gas_pack: ImccGasDatapack,
) -> None:
    pressures_o2, gas, diagnostics = evaluate_gas_oxygen_balance(
        {"SiO2": 1.0}, 2000.0, gas_pack, parent_oxides=("SiO2",)
    )
    # For SiO2(l) -> SiO + 1/2 O2, one O atom per SiO molecule leaves the
    # melt. In the O/O2/SiO limit, 2 pO2/sqrt(MO2) + pO/sqrt(MO) equals
    # pSiO/sqrt(MSiO); rearrange that relation to predict pO2. The Si term is
    # retained as its small, explicit stoichiometric correction.
    expected = (
        gas["SiO"] / math.sqrt(44.084)
        + 2.0 * gas["Si"] / math.sqrt(28.085)
        - gas["O"] / math.sqrt(15.999)
    ) * math.sqrt(31.998) / 2.0
    assert pressures_o2 == pytest.approx(expected, rel=1e-10)
    assert diagnostics["residual"] < 1e-10
    assert diagnostics["mode"] == "oxygen_balance_effusion"


def test_plante_k2o_silica_anchor(gas_pack: ImccGasDatapack) -> None:
    # K2O(l) -> 2 K + 1/2 O2 gives nO = nK/2. With K and O2 dominating,
    # flux balance gives pO2/pK = (1/4)*sqrt(MO2/MK) = 0.2262.
    p_o2, gas, diagnostics = evaluate_gas_oxygen_balance(
        {"K2O": 1.0, "SiO2": 0.5}, 1500.0, gas_pack,
        parent_oxides=("K2O", "SiO2"),
    )
    anchor = 0.25 * math.sqrt(31.998 / 39.0983)
    assert p_o2 / gas["K"] == pytest.approx(anchor, rel=0.03)
    assert diagnostics["residual"] < 1e-10


@pytest.mark.parametrize(
    ("temperature", "commanded_log_fO2"),
    [(1700.0, -7.46), (2000.0, -8.360952930984695), (2500.0, -9.862541149292522)],
)
def test_lunar_mare_basalt_oxygen_residual(
    gas_pack: ImccGasDatapack, temperature: float, commanded_log_fO2: float
) -> None:
    # Low-Ti mare basalt, normalized by the kernel from its supported oxide
    # subset. Commanded fO2 values follow the existing Kress91 IW line.
    composition = {
        "Al2O3": 14.214, "CaO": 11.637, "FeO": 16.353, "MgO": 9.463,
        "Na2O": 0.182, "SiO2": 46.234, "TiO2": 1.587,
    }
    melt = evaluate(
        composition, temperature, basis_type="wt", allow_extrapolation=True,
        allow_out_of_envelope=True,
    )
    activities = {name: melt.activity(name) for name in melt.parent_oxides}
    p_o2, _gas, diagnostics = evaluate_gas_oxygen_balance(
        activities, temperature, gas_pack
    )
    assert diagnostics["residual"] < 1e-10
    assert math.log10(p_o2) != pytest.approx(commanded_log_fO2)


def test_unbracketed_oxygen_balance_is_typed_refusal(
    gas_pack: ImccGasDatapack,
) -> None:
    with pytest.raises(ImccGasOxygenBalanceError, match="not bracketed"):
        evaluate_gas_oxygen_balance(
            {"SiO2": 0.0}, 2000.0, gas_pack, parent_oxides=("SiO2",)
        )


def test_oxygen_balance_refuses_root_above_molecular_flow_ceiling() -> None:
    # Hot K2O-rich melt: oxygen demand exceeds supply even at pO2 = 1 bar, so the
    # balancing root lies above the molecular-flow ceiling and must be refused
    # (the effusion law is invalid there), never returned as a number.
    # Take the function and the exception from the SAME module object: other tests
    # reload openimcc.gas, so mixing the lazy top-level export with a fresh
    # submodule import can compare two different exception classes.
    import openimcc.gas as gas_module

    with pytest.raises(gas_module.ImccGasOxygenBalanceError, match="molecular-flow regime"):
        gas_module.evaluate_gas_oxygen_balance(
            {"K2O": 1.0, "SiO2": 0.5},
            2600.0,
            gas_module.load_gas_datapack(),
            parent_oxides=("K2O", "SiO2"),
        )


@pytest.mark.parametrize(
    ("gas_species", "parent", "fixed_species", "fixed_pressure", "expected_ratio"),
    [
        ("K", "K2O", "K", 1e-8, 0.25 * math.sqrt(31.998 / 39.0983)),
        ("SiO", "SiO2", "SiO", 1e-8, 0.5 * math.sqrt(31.998 / 44.084)),
    ],
)
def test_generic_oxygen_balance_analytic_limits(
    gas_species: str,
    parent: str,
    fixed_species: str,
    fixed_pressure: float,
    expected_ratio: float,
) -> None:
    metadata = oxygen_balance_species_metadata(
        {gas_species: parent, "O2": None},
        pO2_exponents={gas_species: 0.0},
    )

    def pressure_model(logp: float) -> dict[str, float]:
        return {fixed_species: fixed_pressure, "O2": 10.0**logp}

    p_o2, _pressures, diagnostics = oxygen_balance_from_pressure_model(
        pressure_model, metadata
    )
    # K2O -> 2 K + 1/2 O2 gives parent demand 1/2 O per K, so
    # (1/2)pK/sqrt(MK) = 2pO2/sqrt(MO2). For SiO2 -> SiO + 1/2 O2,
    # the deficit is 1 O per SiO, so pSiO/sqrt(MSiO) = 2pO2/sqrt(MO2).
    assert p_o2 / fixed_pressure == pytest.approx(expected_ratio, rel=1e-12)
    assert diagnostics["residual"] < 1e-12


def test_generic_oxygen_balance_rejects_nonmonotone_pressure_model() -> None:
    metadata = oxygen_balance_species_metadata({"K": "K2O", "O2": None})

    def pressure_model(logp: float) -> dict[str, float]:
        residual = 0.001 * (
            logp + 1.0 + 0.1 * math.sin(32.0 * math.pi * (logp + 2.0))
        )
        return {
            "O2": 10.0**logp,
            "K": (
                2.0 * 10.0**logp / math.sqrt(metadata["O2"].molar_mass)
                - residual
            ) * math.sqrt(metadata["K"].molar_mass) / 0.5,
        }

    with pytest.raises(ImccGasOxygenBalanceError, match="not a power law"):
        oxygen_balance_from_pressure_model(
            pressure_model, metadata, bracket=(-2.0, 0.0)
        )


def test_generic_oxygen_balance_validates_model_pressures() -> None:
    metadata = oxygen_balance_species_metadata({"K": "K2O", "O2": None})

    with pytest.raises(ImccGasOxygenBalanceError, match="negative pressure"):
        oxygen_balance_from_pressure_model(
            lambda logp: {"K": -1e-8, "O2": 10.0**logp}, metadata
        )
    with pytest.raises(ImccGasOxygenBalanceError, match="missing or has invalid pressure"):
        oxygen_balance_from_pressure_model(
            lambda logp: {"O2": 10.0**logp}, metadata
        )
    with pytest.raises(ImccGasOxygenBalanceError, match="non-finite"):
        oxygen_balance_from_pressure_model(
            lambda logp: {"K": float("nan"), "O2": 10.0**logp}, metadata
        )


def test_generic_oxygen_balance_refuses_custom_bracket_above_one_bar() -> None:
    metadata = oxygen_balance_species_metadata({"K": "K2O", "O2": None})
    with pytest.raises(ImccGasOxygenBalanceError, match="molecular-flow regime"):
        oxygen_balance_from_pressure_model(
            lambda logp: {"K": 1e-8, "O2": 10.0**logp},
            metadata,
            bracket=(-2.0, 0.1),
        )


def test_oxygen_balance_public_api_is_importable() -> None:
    from openimcc import (
        oxygen_balance_from_pressure_model,
        oxygen_balance_species_metadata,
    )

    assert callable(oxygen_balance_from_pressure_model)
    assert callable(oxygen_balance_species_metadata)


def test_oxygen_balance_metadata_exponents_match_all_gas_channels() -> None:
    import openimcc.gas as gas_module

    channels = tuple(gas_module._SF04_REACTIONS.items())
    metadata = oxygen_balance_species_metadata({
        name: parent for name, (parent, _n_gas, _n_O2) in channels
    })
    assert tuple(metadata) == tuple(name for name, _ in channels)
    for name, (parent, n_gas, n_o2) in channels:
        expected = (
            n_o2
            if parent == "S2"
            else -n_o2 / n_gas
            if parent
            else (1.0 if name == "O2" else 0.5)
        )
        assert metadata[name].pO2_exponent == pytest.approx(expected, abs=1e-12)


def test_oxygen_balance_accepts_caller_supplied_s2_gas_parent(
    gas_pack: ImccGasDatapack,
) -> None:
    import openimcc.gas as gas_module

    activities = {
        **_quickstart_activities(1800.0, allow_extrapolation=True),
        "S2": 1e-3,
    }
    p_o2, pressures, diagnostics = gas_module.evaluate_gas_oxygen_balance(
        activities,
        1800.0,
        gas_pack,
        parent_oxides=(*IMCC_PARENT_OXIDES, "S2"),
    )

    assert math.isfinite(p_o2) and p_o2 > 0.0
    assert pressures["S2"] == pytest.approx(1e-3, rel=1e-12)
    assert pressures["SO"] > 0.0
    assert diagnostics["residual"] < 1e-10


def test_oxygen_balance_metadata_accepts_explicit_exponent_override() -> None:
    metadata = oxygen_balance_species_metadata(
        {"K": "K2O", "O2": None},
        pO2_exponents={"K": -0.3},
    )
    assert metadata["K"].pO2_exponent == -0.3


def test_generic_oxygen_balance_rejects_unbracketed_pressure_model() -> None:
    metadata = oxygen_balance_species_metadata({"K": "K2O", "O2": None})

    def pressure_model(logp: float) -> dict[str, float]:
        return {"K": 1e-8, "O2": 10.0**logp}

    with pytest.raises(ImccGasOxygenBalanceError, match="not bracketed"):
        oxygen_balance_from_pressure_model(
            pressure_model, metadata, bracket=(-30.0, -20.0)
        )


# Numeric-output SHA256 pins for the corrected Na2O(l)/Al2O3(l) rows: pO2,
# partial pressures, and diagnostics. Case 20 now refuses because its corrected
# Na pressure moves the oxygen-balance root above 1 bar. The full result is
# pinned separately below, including every ImccGasResult dataclass field.
_OXYGEN_BALANCE_NUMERIC_BASELINE_HEX_SHA256 = (
    "1d29dbab95bc837c6dfac4caa5d3eea739eea52d94484a2b6cee4afffb44c467",
    "650b0e76979d937e98b7c4b16f000718745aa736f08a8f3e12d52d2e353b28e4",
    "4e17b907124ca0da85aa5ca5cf962d531033e2b910b4704892f8b3490f34f145",
    "d3df2539e2255e19e74e94626495e5908c4a81748d0590abdd9c34ab0afc7b05",
    "2ca1cd1169f2b95166b5341233d89df6d17eab026dde3943000ec8438291a8ea",
    "585f8859e0b1bf72328c973c9abe6b4d109609916af301f4de9ede019ec07490",
    "c4a84c82f57b4db015975531ecbd74c8e1a39d64e961bc9ca741d724e04685a5",
    "f592dfad5d8ce13250b734fd285665daed36c09f68dcdb704734e5eee504c397",
    "4297d94edf6d3d76a87be43dcf44ab5ae9d2b93d4cbbf946bf93926268fecd61",
    "6bb036e0bacb82986b029b831ecd40a957391e3d5bc8e5b05af23ede4ff80bb2",
    "731e4c249311a057a0b836160d19d63b4d32d8f99c64e07d1d4c70098f97bf9e",
    "67a09043136b310e1990055863e0ba698cfe085c501ac5fa68f8dd2fa043bef4",
    "d35d6dcc59c64317133d681034ebaecef930a4bd61eb27e63118a751d27678ec",
    "2eab281efb705c6b26122784f20c7f1ef04c8af6adc2c0087755ec33596ef760",
    "707b74ad1767a83c4bddec39ae5ec4625ac8b5d2a19da4c78e7cacc27d58da42",
    "e14972c2096dac66f913013832c2e699e213e605109450ac4cc77af2723eb263",
    "85dbdb25bffe65426bda27d7cbcd0213bf831cf7aaebaa2350d951e3343bcd34",
    "1dc52eb6b0c470424d59b3d565aa925423576790adc468b2fb25492cdd5454f6",
    "d1d0cc3de67dab428864282fd9015ff05ad63ec47eb37f747596930ba56a86ef",
    "aaa02cbb979d27b76a55391692f20af7bacf154960389f6950cb49a032401e02",
    "8e159b9eb44cd0171a03bafdfabe3b0f549f994c70325260fd1883027ebe3a95",
    "83b1e412339d20ba7c80283e1b6f6301ee2d80751b149d667959c37dd91b5b21",
    "db58f4665d7ac808b3093f083af6f1efcb5238b261d6936ed114fb2bc57201aa",
    "9bd3ac2af8aacd35dae2bc739f0c2cb0c7d030d4a75679d7e0114588358ea7f2",
    "ec58dd80a28f9acb1fa9ef0b50c711c3141cf3acd7100a5df0898001a700f419",
    "1f79122c21aec9f84197a5ad322883ea93bd0030b3f23952cf6557466121044e",
    "b86f66c6b2847fe676bf9ef305fe47eeed65205d29259b3aeba678978fd3b0b3",
    "065d99198ea856f805371753cde96cf24f1d9ab3d0e80d089a05c39c35e9febd",
    "29d3694f7fa8473e7b8f0d12913fe5cd60fdaede4ba9cfc54470f9634fb44f76",
    "e4ab384ddf0a1e6b537fd8f3cff328ee6208b8568684d9286f90ed3ba76756d7",
    "7590b4367ff2b0191f8e14bb4f39531fefdacd7f8d4821675318416b06f937c9",
    "62d83d6634e6941e7ef5c116df9783120daf24bae208ac270a41d0356344577f",
    "f74c9779b622b779744d251752ca88440f4e156e7d677ea3f1e52a6e7d41e07f",
    "824ec5c65585f78dc85e0518f113ffb867336b9d2d2ffdb66ca209ad5e30e880",
    "79abe2aae4508e3fb8d49dc3d2ab74655e6d899b1e00b42bad961fd1829f3396",
    "fe0ad875f4b5ab7a95ecd243f736983dff4e973946ffbdf9c0b81fc435b67185",
    "ae56276ffb9a6d8bf4e88f82893320697465e247f6f55c47221c7719871e1014",
    "f1e99183e9753e0e57531c103f0792ed46361f000f4962fdaaf068cdedf36916",
    "5b1d81bbe6f054161f9246752dc399f81fed51e345844a8fe35cf8cd64d17f75",
    "fdb339a56ad2111878b0caf8eef49294d9f37e9f63aa0f7d7bd8605c612ca9a9",
    "654b7d5454a19e3d39c8752e6308d72e143a3d0dca71960f1aefa15f5d0edd49",
    "9c301cf457eec1bdf7a299f20f6b85355a955eaa94055960eab3ff6d234b6312",
    "4ff0203d7f1c9fef5c04d9db5e330779844d6aca14201ad6e2facb8eaba47974",
    "2a82ca18745b6be1d9de93effb3631b965b1b8d81794ce35597eaac0ce1e57a9",
    "609726c45270c673e18425f2c0659a001a371a353c85561d3dac085baa942ac8",
    "44c3beeef8c6e5c50251dc658b89c7cd3df8d9621ab462eadf5fafd568962005",
    "be9d57e3de23e3dd4640b26c258f4dda67b70170419e0b5cf76b0431f728bed5",
    "7673f66041abd726a3f82323bc1863416d13abf859c158707c6bf289e5c883a9",
    "7c98b26a8681c30546de6cb2aba87dc4f7e7c205322d42bc48b56d48c605e7e4",
    "26d198becc1cb87badbf17605e96d228da16f6a1d3cd67187b8b8c8816c10984",
)

# Full dataclass-aware result pins for the corrected parent-liquid rows. The
# numeric projection above pins the same pressures and oxygen-balance outputs.
_OXYGEN_BALANCE_RESULT_BASELINE_HEX_SHA256 = (
    "3d8ecba0274b91b82d5d743e9e280d1bd3b6a5d135de43fee5ce83fc1f5c37b3",
    "bf227faf8d1dc83a3c48dc67b19534e9beb1a44e8d7385b8fcd0dde7790140d0",
    "5eb736c1fa352437dcbbc26c7642c6494a24c3129ad350015e618b91d4821c0b",
    "804a6a522a3698076b99cd879986b36402bdee6b7bba8847a6474cd6223e8d6f",
    "cb4303b5c42b90bf5dc9728a9b0def52c6001859c8e89aec76861afdcf172f9e",
    "f58ef45404c95e87c2c4ef4f98fe9a433485627f0599d90f707a09ecb044ce0a",
    "0b7b196ff4618f59ec424e50f3377eef4bc21e56e3e50a26dcf83a05473659b7",
    "a0a6c2ea4c482dccd3a6b428344093e45a99d1277beea55859723fa659072fcc",
    "4b692e01e5d8cfb7e7ca48262ce972718af8ed8be6f61564313d3e273e7f4518",
    "7e21ef5c5f850792ecc53a0bc461d73a4c3f8277497cbf88b3c981eae68952e2",
    "a52ef908bf28e56d852f36d26c7f79c4797a6e5f8b89da00a96d02afca8c92bc",
    "a90c0423b314026c3271fe85753471537252c61a11641b3f17c8dc21a76b1b99",
    "e2aaa0794f39f61cd59d50111d46ab02731aa5977e9f9be21431c42aa123acf1",
    "dffa7bf65e1415cdf78834c9e847e8362e115780ccfd63a4a5365ca693c9f281",
    "ed48eb3fed861047d3cd9cf1b7b7aed780beaa4751be4727c5c59fa5a52c21d4",
    "89e737ce42960dcc1441181fad2027a460d1302457291d75cce8a01b523ef821",
    "6427ab454c7739b65e6a24a0eadc2a02be8838c5532aa5e8fcb73d638059efa2",
    "db16017913b114928f13ced11c02e08b4577ac4a0e8d3d382715684eda6a6ed9",
    "9b668965d94096f799ff85df860c85c7f91496aed6b3c8b98c58c5dd35d20e86",
    "b20a7a2b5ac03d8470f507577498fffcf09ca95bbc20ae0288818d7c21c081f7",
    "8e159b9eb44cd0171a03bafdfabe3b0f549f994c70325260fd1883027ebe3a95",
    "70fd8106c1fd5416a1345d2c46741f383a8491176358c0e8ece83b6dd20966e4",
    "32e410b10c2b9555e35c5b4523e2515b5daafe3039e587526877e3e54dabb6df",
    "90b0420bfd260d99ebe2a185759939694d856d57d7c11d5451302443c4ac6064",
    "2006103ca9a144bf80a0a9b0a322df83ddb1a136028e152f9923fe4705e0c4e7",
    "ce99b1982f5fe4eac6539991e3cdec0acdf6ac822a5503e7b81904f28275d3b0",
    "3542a99a5417b8a397bdaf397701df2534ae89232b43726ddac460ffbc2d7e28",
    "a34b83c0d53530cf21f90d944f88b76d3745a385505d60ddc54e258d9c2c931d",
    "c66191a0714ddfbad1920b0a6bd6efd05133d8a688af746dd7a9f61eb990b634",
    "3b522be5a2a742643c0c34b036749e067685dd182fef2b6988e6c00bba3406cf",
    "4696c1a869b30a74f5a290c31e43145a6ff349fa921266560b34f593a0c39b60",
    "16b3a5f888d665baeb4dc223d6ecc970e9491f40e779977b6796b2b94f108a95",
    "68f00bc13521717e0d4b7a8855593a7042fda21b634339e1f92b0b7f356ea294",
    "d1a6bc49befa96aecffce31dd458a27b71706ccc0b8961bc410740eb9c498ab8",
    "32fb365c5aad8946cc31b9e940473056b3eb142c9e06e865160dc7dde0e8f73d",
    "6d23bab7810fef7a6736f1f92893023c2e674508de23a2105d3dc5f6f61c0c4b",
    "8436af7f648df1918a30e3ad15eb6097008e2986e00bf09d499b2693ba2aeed0",
    "32c9ab241d3ba052aed57c8e3fa8b5bfb9ae8a593d3b86baed853b6957eb898e",
    "0182b132280bcb563e2e1aec65af5c53557e2971f0063fb49e7d24618fb2af87",
    "2fee4df820262dd304ae8b3f7f746d14b8f1e4326bd2a558514818c6e170e854",
    "cfceea0248cd40c69800dc30224507a5ced4009cbc63b2d8807b8220b457dcb6",
    "39be1577cab0bf36a79aaf6a422646e3549cc81425af3dbd6ebeb5a9b902d195",
    "893d4d5d91fb4469ab1a23ab8d2745555cb707dc765ce1951d07f980e04fe010",
    "42820b60d55e6e3b3dd52e810a6b41774ce37408f1fbb8dcbf4a144326f6e134",
    "7cf90a541077d82369a82fe15d48a7721079d5d78f4ff089891b4d69b84d208e",
    "9ad52deaa8847584a2448d41190b754d823eb68fe7d3939ec9af25dc89cf5650",
    "a2e4593ede269e34cf19710ce22a0a7495f0ff99e67392092e875c0b5b557e05",
    "738e5917f3bb290ea1bffa798e31d1e1ebdd13207728e76eef126c757b07b682",
    "78cfa8c4671e908ac23138dcba3408bdf1dc28dfa9857a19eb7558ac4a18cf08",
    "6a750c7bf0b55f529231622b8c3e3238024dc41b7e93c978cdeefc326080dcf9",
)


def test_oxygen_balance_wrapper_matches_corrected_hash_pins(gas_pack: ImccGasDatapack) -> None:
    compositions = [
        {"SiO2":.45,"MgO":.12,"FeO":.15,"CaO":.11,"Al2O3":.07,"Na2O":.03,"K2O":.01},
        {"SiO2":.47,"MgO":.05,"FeO":.08,"CaO":.14,"Al2O3":.24,"Na2O":.015,"K2O":.005},
        {"SiO2":.50,"MgO":.19,"FeO":.18,"CaO":.08,"Al2O3":.04,"Na2O":.005,"K2O":.005},
        {"SiO2":.43,"MgO":.24,"FeO":.19,"CaO":.08,"Al2O3":.05,"Na2O":.005,"K2O":.005},
        {"SiO2":.45,"MgO":.02,"FeO":.03,"CaO":.18,"Al2O3":.31,"Na2O":.009,"K2O":.001},
        {"SiO2":.49,"MgO":.08,"FeO":.12,"CaO":.10,"Al2O3":.16,"Na2O":.035,"K2O":.015},
        {"SiO2":.48,"MgO":.04,"FeO":.09,"CaO":.10,"Al2O3":.18,"Na2O":.07,"K2O":.04},
        {"SiO2":.40,"MgO":.31,"FeO":.17,"CaO":.07,"Al2O3":.04,"Na2O":.006,"K2O":.004},
        {"SiO2":.65,"MgO":.02,"FeO":.03,"CaO":.04,"Al2O3":.20,"Na2O":.05,"K2O":.01},
        {"SiO2":.55,"MgO":.10,"FeO":.18,"CaO":.10,"Al2O3":.05,"Na2O":.01,"K2O":.01},
    ]
    cases = [(c, t) for c in compositions for t in (1500., 2000., 2400.)]
    for a, b in (("SiO2","MgO"),("SiO2","FeO"),("SiO2","CaO"),("SiO2","Al2O3"),("SiO2","K2O")):
        cases.extend(({a:.6-.05*i,b:.4+.05*i}, t) for i, t in enumerate((1500., 1900., 2200.)))
    cases.extend(({"SiO2":.45+.02*i,"MgO":.15-.01*i,"CaO":.2,"Al2O3":.2}, t)
                 for i, t in enumerate((1500., 2000., 2400., 1800., 2200.)))

    def hex_values(value: object) -> object:
        if isinstance(value, float):
            return {"float.hex": value.hex()}
        if dataclasses.is_dataclass(value):
            return {field.name: hex_values(getattr(value, field.name)) for field in dataclasses.fields(value)}
        if isinstance(value, Mapping):
            return {str(k): hex_values(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [hex_values(v) for v in value]
        if hasattr(value, "__dict__"):
            return {k: hex_values(v) for k, v in vars(value).items()}
        return value

    actual = []
    numeric_actual = []
    refusals = []
    for index, (activities, temperature) in enumerate(cases):
        try:
            p_o2, pressures, diagnostics = evaluate_gas_oxygen_balance(
                activities, temperature, gas_pack, parent_oxides=tuple(activities)
            )
        except ImccGasOxygenBalanceError as exc:
            refusals.append((index, temperature, str(exc)))
            canonical = json.dumps({"refusal": str(exc)}, sort_keys=True)
            digest = hashlib.sha256(canonical.encode()).hexdigest()
            numeric_actual.append(digest)
            actual.append(digest)
            continue
        numeric_canonical = json.dumps(
            hex_values((p_o2, dict(pressures), diagnostics)),
            sort_keys=True,
            separators=(",", ":"),
        )
        numeric_actual.append(hashlib.sha256(numeric_canonical.encode()).hexdigest())
        canonical = json.dumps(
            hex_values((p_o2, pressures, diagnostics)),
            sort_keys=True,
            separators=(",", ":"),
        )
        actual.append(hashlib.sha256(canonical.encode()).hexdigest())
    assert len(actual) == 50
    assert [(index, temperature) for index, temperature, _ in refusals] == [
        (20, 2400.0)
    ]
    assert "root lies above pO2 = 1 bar" in refusals[0][2]
    assert tuple(numeric_actual) == _OXYGEN_BALANCE_NUMERIC_BASELINE_HEX_SHA256
    assert tuple(actual) == _OXYGEN_BALANCE_RESULT_BASELINE_HEX_SHA256

@pytest.mark.parametrize(
    ("activities", "temperature", "parents", "message"),
    [
        (
            {"SiO2": 0.0}, 2000.0, ("SiO2",),
            "oxygen-balance root is not bracketed on (-30.0, 0.0): "
            "F(low)=1.6726738105968946e-19, F(high)=0.35373170703571655",
        ),
        (
            {"K2O": 1.0, "SiO2": 0.5}, 2600.0, ("K2O", "SiO2"),
            "oxygen-balance root lies above pO2 = 1 bar, outside the Knudsen "
            "molecular-flow regime where the effusion law holds; no value returned "
            "(F(1 bar)=-2.6842062105831452)",
        ),
    ],
)
def test_oxygen_balance_refusal_messages_match_base(
    gas_pack: ImccGasDatapack,
    activities: dict[str, float],
    temperature: float,
    parents: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ImccGasOxygenBalanceError) as error:
        evaluate_gas_oxygen_balance(activities, temperature, gas_pack, parent_oxides=parents)
    assert str(error.value) == message
