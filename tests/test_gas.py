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
    _nearest_interval_row,
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
        "Na": -1.890,
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
        "AlO": -5.865,
        "AlO2": -5.932,
        "Al2O": -9.964,
        "Al2O2": -8.422,
        "Na2": -3.725,
        "NaO": -1.414,
        "K2": -0.030,
        "KO": 0.534,
        "Si": -11.905,
        "Al": -8.695,
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
    assert len(gas_pack.gas_df) == 43 + len(build_gas_tables.LOW_T_GAS_SPECIES)
    assert len(gas_pack.oxide_df) == 12


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
    assert research.gas_df.equals(interval_1)
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
    assert len(low_rows) == len(build_gas_tables.LOW_T_GAS_SPECIES)
    assert set(low_rows.index) == {
        f"{species}(g)" for species in build_gas_tables.LOW_T_GAS_SPECIES
    }
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


def test_channel_coverage_ledger_is_closed() -> None:
    implemented = set(IMCC_GAS_CHANNEL_SPECIES)
    unavailable = set(IMCC_GAS_UNAVAILABLE_SPECIES)
    in_domain = set(IMCC_GAS_WORKBOOK_IN_DOMAIN_SPECIES)
    extrapolated = set(IMCC_GAS_WORKBOOK_EXTRAPOLATION_LABELS)
    assert len(implemented) == 46
    assert len(unavailable) == 5
    assert implemented.isdisjoint(unavailable)
    assert in_domain.isdisjoint(extrapolated)
    assert in_domain | extrapolated == implemented


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
        "Na": 0.012891128142910052,
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
_MNNICO_PARENT_PAIRS = (("Mn", "MnO"), ("Ni", "NiO"), ("Co", "CoO"))
_PRE_GATED_CHANNELS = tuple(
    species
    for species in IMCC_GAS_CHANNEL_SPECIES
    if species not in _CR_CHANNELS
    and species not in _VNB_CHANNELS
    and species not in _MNNICO_CHANNELS
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
        expected = -n_o2 / n_gas if parent else (1.0 if name == "O2" else 0.5)
        assert metadata[name].pO2_exponent == pytest.approx(expected, abs=1e-12)


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


# Numeric-output SHA256 pins from 23d7842: pO2, partial pressures, and
# diagnostics. The full result is pinned separately below, including every
# ImccGasResult dataclass field.
_OXYGEN_BALANCE_NUMERIC_BASELINE_HEX_SHA256 = (
    "4d261a81c0bd9e12f153fc1a08ba692bfa2b892573f64f6a0f2074d3877463e5",
    "9881ff4f3cf73b2fd0200591e5ef744b00d428aa01be1c0fd3a793c231edb636",
    "5836acf202077080f3316d557c3bebc2ff0217740c6d0a4234c16024bc75b57c",
    "18af5ea40fc3e8b802bccd6eb69d2710470fb59e22bf9e2de7045ae07f6329f8",
    "79f4f125308e32a9b6209b62e7bf4dfdac7da6dcca45314bf56f8821511a3a38",
    "c4a5af153775d9ebc4a5436bfc82b3e758f7b11b853544c1ef92e1c41f784329",
    "70f1600be054145a3876544596167dbb63df43bca687377e19c87a2c54d44a68",
    "fae604c5cd5c7617fec8010f242fdfc07f35e125cfaefcfea374d56943d306a4",
    "80e20413a7d439e7e1e34107bb9bbf621f2b7cd87aa8483cad8a1231df2cccad",
    "2bf336b1791278f409a376747fd9400af0ec09a7161e85e7ee585dfcc28a3067",
    "1e512733a87c1e6a80be1c39c1a0c58a0931433ba9afaf0fa80675b4e9bec146",
    "a500f0a377f00ab4ebbd21eaa69a0f2c8aa0784c3dc125f52a9d504bfcc24e6e",
    "a2dd8a1fd1065c186ea97812c75badc5d47591d7b2b00402ffd9f6f4391761b6",
    "4a67ce14343418f61108e01390e25cccdf55860f2cc99c42ea3307cfa46b25aa",
    "be2abdeb0115fb3c7318fbf6300cd61c1bbee14f1c5e9614ecb88edd5efad8b3",
    "5f3fc0d837f84968f9d12e79ab106bdd57560f4e39e0df80b384b302ec40db34",
    "fa890723d245fc12acef7669578b6e8b9937dedef663586a8156fbb951fe1e4e",
    "a275d67c35d55d21caba85f77a0c973f16d02caadecb5890133fa565513e861e",
    "3827a41ed9e4b1e7d81d8cbd40a6595210aff922afc56d2e4ff3dca70b26fbed",
    "78dfdd12a77d4d2432c7e6cebc20f6464cd6bf7e096de93f0fb0fadb08062fc2",
    "5fc5005f0e5d09b76e2a4dfa2006273fa38793372f996e4d050119b7fc3d6a3e",
    "8340d056bf589e357ae26966df355f9dba5a2828ca22717ec9ee05bc5be88b9e",
    "4042cab636e73451691c490d9df8de767ae58b86506e3221a8706f897e2d1615",
    "3013a083553ff9384b8f53b83b68b809d2eb89a634412f76c2e09013d99a609a",
    "2b2e041b33d06041bda2f062827397b9e3dae15547703ebb5d92d94c014d90e5",
    "eaa05922aa6e63f46ecf5871734fe09b8507b7eabcafd98e53ab115408483ec7",
    "15d9a3bebd1c1cd93c629b653778c7eac887c642e4ba7bcca302ab9226568276",
    "88cb2b7a3d6115e4c9d1d4dec62e8ed6528b8be457f7f323aea038fea6bc5326",
    "085a600388d82f816af7fa250305a45d0952eaeb38631f4a9dbeda98664d4816",
    "e84bf32be9017b5392c789109a8c86f2677c6f52ff0421f85a087f838e197697",
    "7590b4367ff2b0191f8e14bb4f39531fefdacd7f8d4821675318416b06f937c9",
    "62d83d6634e6941e7ef5c116df9783120daf24bae208ac270a41d0356344577f",
    "f74c9779b622b779744d251752ca88440f4e156e7d677ea3f1e52a6e7d41e07f",
    "824ec5c65585f78dc85e0518f113ffb867336b9d2d2ffdb66ca209ad5e30e880",
    "79abe2aae4508e3fb8d49dc3d2ab74655e6d899b1e00b42bad961fd1829f3396",
    "fe0ad875f4b5ab7a95ecd243f736983dff4e973946ffbdf9c0b81fc435b67185",
    "ae56276ffb9a6d8bf4e88f82893320697465e247f6f55c47221c7719871e1014",
    "f1e99183e9753e0e57531c103f0792ed46361f000f4962fdaaf068cdedf36916",
    "5b1d81bbe6f054161f9246752dc399f81fed51e345844a8fe35cf8cd64d17f75",
    "5abaab557849464f3726375aa91a6ddb9e7cdc9d9e2656a959f20ff83f14c450",
    "0f198777e2475d6949081264b43aa618b26405ce7b62c0a5c1d2b71ad7edfb7e",
    "0be2496c7a39e5f6185cd63f460047b98449b561c7c0f0036a2ef6244adb99e0",
    "4ff0203d7f1c9fef5c04d9db5e330779844d6aca14201ad6e2facb8eaba47974",
    "2a82ca18745b6be1d9de93effb3631b965b1b8d81794ce35597eaac0ce1e57a9",
    "609726c45270c673e18425f2c0659a001a371a353c85561d3dac085baa942ac8",
    "3e7cddeff5529755c45553c6ec638609363178271b94178be734bf2f10f626c7",
    "92575f12914d27eb276247bab3331baccc749a478e4dfb7ba9a4dcf88a03a5ea",
    "2d13a21e22f9442c511d5a2c270a4ff176d2ae8a4fcb3dbcc40ce76f2fe97ea1",
    "d76d62634ac7b8defd9e25f54b4496c802ff11c9ddae1677f9dcdb73ba75b7d4",
    "7d78316eb51b6666eb7122bea5b29803ce4f81208c59e35446e147d0d089cd08",
)

# Full dataclass-aware result pins from cb5117e with the new empty omission map.
# The numeric projection above must continue to match its pre-change pins.
_OXYGEN_BALANCE_RESULT_BASELINE_HEX_SHA256 = (
    "cd9b342310202a1f456780078e81961d4a104adb5e08f7f06d9f295cb68a5918",
    "e48e03ae7f78084ea777cb9caef2eacec79cf3141edf9b5f0afe40e456ed09f3",
    "8b948b244a277890bc0047520d6b41f15edd248974e2cc42c763475cc7a5fcf7",
    "a0236df635076dc07f7e21efad8024cc7aea6ea1875dd32211742c82a68a3b61",
    "b6fe74e4f620e78e21e7f4f8144741ce103b62a735f0eb9852ef8824f09cd224",
    "7d643462cb86aa686dcc8d2773d0d3b72f51d1f52225b181df5b8dee5f6b813d",
    "b30ff620af23f80237f2a6aca2296d7885f556ee538ebc8d0e7bf47402e7214b",
    "fc0115459142f520c8847d523d9884a2f5d7262f39cd469e4a419a98b1d1e7af",
    "5f906d382c1e0a241d887ecfe14329ebbc54cd4c07e7650eb7461a0f7a1e52f3",
    "7617f2c27a90375a8bd0ca16d3f2c1185c1312a025640dee27522d44a2fb05d2",
    "b371590c4ec9c4f19fa8a56f0089b47c361495c2a0af67cc48f9411bcc0347fa",
    "8df1a2d6beaa5c6c3acea46002aa2a00d11a751d9b471d9e40e89cf7921ffa5c",
    "50440c29a57047eb2e3a7b24e44ab1e1f91e029a871539fca67bcd04831b2c75",
    "aac42059e364f84fa3f0ab5e16e3e0b8e5eea2f7270ad14d740fc29e8d43546b",
    "395675a4d0b97400dda6ab7765424ff4800e4104c83a0ac3e2b48493b96618ed",
    "930831905e0353ee2120c668181ea744b62d0b81526fedcef7fee80faf56b29d",
    "508ba2fd4c8899b58f9bc441d1db577cef4c734e12e5ae54bd505ede04975e76",
    "b72bc8058384f75a1058e76e0e2a7d5c5c1339b08511dfa53cd33dbce8ce4e8f",
    "09f5cedcaa1bbe3946900f8f15e3261bc0ab6b453cc5e63445555876994eedca",
    "454e9bdc9c3849009f11075100006afd6e4f2d0261996b12030e08d3ada7f7c8",
    "c929cfb9c71f0cc698b918017527bbda13d082126da93e68d9c0af718a792a52",
    "b13994430629fa963172351acd5f672920acc3ad4398acfae0588ca46a3f95ad",
    "bcef82e57d39f7ed280804557d2c2df14310ee0daf9319f4eeb0dc8609023eb8",
    "a2ec75630194616a505a3294b0299b8cd87bed99def2940312b4e273589e24bc",
    "d0f5b47aa24a8b2774e097eed5ca9be6398164ef9ab2a09b4d99a6a2c76ed7a3",
    "121b1ddf755ea734a609ad6975a624ac7b58472493cad313e2ecd1a42c39605f",
    "5291d46cc4a0feecd88516268c981d8555323f13c946a9999fbcde3170af6e51",
    "2bc324e70b84010bac21a0f24d3f50e93667b7f710fcd139bd60ec10e616968e",
    "34d62b0a0a7ed070785e45387cf937fafca3b58cc45dc11e7abe7654a281969f",
    "fcdbed0769258de81ebceed493e84d9def2c7f4f8c353dd630ff6339a12dfe05",
    "4696c1a869b30a74f5a290c31e43145a6ff349fa921266560b34f593a0c39b60",
    "16b3a5f888d665baeb4dc223d6ecc970e9491f40e779977b6796b2b94f108a95",
    "68f00bc13521717e0d4b7a8855593a7042fda21b634339e1f92b0b7f356ea294",
    "d1a6bc49befa96aecffce31dd458a27b71706ccc0b8961bc410740eb9c498ab8",
    "32fb365c5aad8946cc31b9e940473056b3eb142c9e06e865160dc7dde0e8f73d",
    "6d23bab7810fef7a6736f1f92893023c2e674508de23a2105d3dc5f6f61c0c4b",
    "8436af7f648df1918a30e3ad15eb6097008e2986e00bf09d499b2693ba2aeed0",
    "32c9ab241d3ba052aed57c8e3fa8b5bfb9ae8a593d3b86baed853b6957eb898e",
    "0182b132280bcb563e2e1aec65af5c53557e2971f0063fb49e7d24618fb2af87",
    "6bf761e85b1febaab7a1ccdfdc0e75a7056f1f5b821586cb91725087e19ce664",
    "bc0d11926fe3900adb1a5ed60fd8ad024007d31b22b543d438e4267ac4a99ba1",
    "7f8a5406278817d60d07537cb03f398bfed740934ad7fd572810d09d36585819",
    "893d4d5d91fb4469ab1a23ab8d2745555cb707dc765ce1951d07f980e04fe010",
    "42820b60d55e6e3b3dd52e810a6b41774ce37408f1fbb8dcbf4a144326f6e134",
    "7cf90a541077d82369a82fe15d48a7721079d5d78f4ff089891b4d69b84d208e",
    "52ffd07d48e25e0d721489bbb10bc9184df0964a158edf059ceae9c9004a75b3",
    "cf72f54736b532f70a6d2a5d58dda6689c9649116b242be530c8ca1255f90c41",
    "6436c3dab7ea19e132a9a5b0e14cd5bba75a44530cd75b2dedbe3df151238040",
    "b650d3a2dc129148cef31d3785f6fd92fe882755be752f1c855af8970a34f096",
    "7e3168b62caae5a0c6de67a3a66974e1e4f6d852a8542bdf1f93d2fec77b7b93",
)


def test_oxygen_balance_wrapper_is_bit_identical_to_base(gas_pack: ImccGasDatapack) -> None:
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
    for activities, temperature in cases:
        p_o2, pressures, diagnostics = evaluate_gas_oxygen_balance(
            activities, temperature, gas_pack, parent_oxides=tuple(activities)
        )
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
