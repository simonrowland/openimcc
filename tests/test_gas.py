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
import warnings
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import yaml

from openimcc import evaluate
from tools import build_gas_tables
from openimcc.gas import (
    IMCC_GAS_CHANNEL_SPECIES,
    IMCC_GAS_INCOMPLETE_PARENT_SPECIES,
    IMCC_GAS_NO_JANAF_ROWS,
    IMCC_GAS_UNAVAILABLE_SPECIES,
    IMCC_GAS_WORKBOOK_EXTRAPOLATION_LABELS,
    IMCC_GAS_WORKBOOK_IN_DOMAIN_SPECIES,
    IMCC_PARENT_OXIDES,
    IMCC_SF04_WORKBOOK_GRID_K,
    R_J_MOL_K,
    ImccGasDatapack,
    ImccGasInvalidFugacityError,
    ImccGasIngestionWarning,
    ImccGasInvalidIntervalError,
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
    species_thermo,
    _REACTION_PARENT_OXIDES,
    _SF04_REACTIONS,
    _default_reactions,
    _janaf_gibbs,
    _lamor_gibbs,
    _oxide_row_for_T,
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
        "Na": -0.646234,
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
        "Na2": -1.420546,
        "NaO": -0.261607,
        "K2": -0.030,
        "KO": 0.534,
        "Si": -11.905,
        "Al": -9.859116,
        "CaO": -5.475,
        "Ca": -6.258,
    }
}

# Packaged for the opt-in charge closure, not as default neutral channels.
_ION_GAS_SPECIES = {
    "Na+", "K+", "Ca+", "e-",
    "Na-", "K-", "O-", "Al-", "Fe-", "Si-", "Ti-", "O2-", "AlO-", "AlO2-", "KO-", "NaO-",
    "Cr-", "V-", "Nb-", "Li-", "LiO-", "Rb-", "Pb-", "Cs-", "Cu-",
    "Li+", "Rb+", "Pb+", "Cs+", "Sn+", "Cs2O+",
    "Ga+", "Ge+", "B+", "Ga-", "B-", "BO-", "BO2-",
    "Zn+", "Zn-", "Cd+", "Bi+",
    "Mn+", "Ag+", "Ag-",
    "Mo+", "Mo-", "W+", "W-",
    "Ba+", "Sr+", "Zr+", "Zr-",
    "Ni+", "Ni-", "Co+", "Co-",
    "Hf+", "Hf-", "Th+",
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
    assert len(gas_pack.gas_df) == (
        62
        + len(build_gas_tables.LOW_T_GAS_SPECIES)
        + 2
        + len(_ION_GAS_SPECIES)
        + len(build_gas_tables.TRACE_ION_GAS_NASA_SOURCES)
        + len(build_gas_tables.TRACE_ION_NASA_SOURCES)
        + len(build_gas_tables.TRACE_ION_GAS_YAML_SOURCES)
        + 3
        + 2 * len(build_gas_tables.TRACE_GAS_SPECIES)
        - len(build_gas_tables.HALOGEN_GAS_SPECIES)
        + 1  # Sc+ has an additional interval row.
        + 2 * len(_RARE_EARTH_CHANNELS)
        + sum(
            source[1] in build_gas_tables.HALOGEN_JANAF_TABLES
            for source in build_gas_tables.GAS_SOURCES
        )
        + 6  # Two gas fit intervals for each metal-vapour channel.
    )
    assert len(gas_pack.oxide_df) == 186


_MAJOR_PARENT_CONTINUATIONS = (
    ("SiO2", "SiO", 1700.0, "SiO2(l)"),
    ("Al2O3", "Al", 2400.0, "Al2O3(l)"),
    ("MgO", "Mg", 2100.0, "MgO(l)"),
    ("CaO", "Ca", 2100.0, "CaO(l)"),
)


@pytest.mark.parametrize(
    ("oxide", "species", "temperature", "parent_row"),
    _MAJOR_PARENT_CONTINUATIONS,
)
def test_default_major_parent_continuations_are_flagged(
    gas_pack: ImccGasDatapack,
    oxide: str,
    species: str,
    temperature: float,
    parent_row: str,
) -> None:
    result = evaluate_gas(
        {oxide: 1.0},
        temperature,
        1.0e-10,
        gas_pack,
        parent_oxides=(oxide,),
        gas_species=(species,),
        allow_extrapolation=False,
    )
    assert math.isfinite(result[species])
    assert "constant-Cp supercooled-liquid continuation" in (
        result.domain_flags[species]
    )
    assert parent_row in result.domain_flags[species]
    assert result.provenance_class[species] == "janaf_fitted"


@pytest.mark.parametrize(
    ("parent", "species", "temperature", "runtime_ref"),
    (
        ("Ga2O3", "Ga", 1500.0, "NG-12403"),
        ("GeO2", "Ge", 1200.0, "NG-12434"),
        ("In2O3", "In", 1500.0, "NG-12662"),
    ),
)
def test_trace_parent_continuations_are_flagged(
    gas_pack: ImccGasDatapack,
    parent: str,
    species: str,
    temperature: float,
    runtime_ref: str,
) -> None:
    result = evaluate_gas(
        {parent: 1.0e-3},
        temperature,
        1.0e-8,
        gas_pack,
        parent_oxides=(parent,),
        gas_species=(species,),
        allow_extrapolation=False,
    )
    assert result[species] > 0.0
    assert "constant-Cp supercooled-liquid continuation" in (
        result.domain_flags[species]
    )
    assert runtime_ref in result.domain_flags[species]


@pytest.mark.parametrize(("parent", "species", "extension_limit", "fusion_temperature"), (
    ("ThO2", "Th", 2500.0, 3650.0),
    ("HfO2", "Hf", 1800.0, 3073.15),
))
def test_liquid_from_solid_refs_emit_extension_and_supercooling_flags(
    gas_pack: ImccGasDatapack,
    parent: str,
    species: str,
    extension_limit: float,
    fusion_temperature: float,
) -> None:
    result = evaluate_gas(
        {parent: 1.0e-3}, 2800.0, 1.0e-8, gas_pack,
        parent_oxides=(parent,), gas_species=(species,), allow_extrapolation=False,
    )
    flags = result.domain_flags[species]
    assert f"crystal Cp was extended above {extension_limit:g} K" in flags
    assert f"supercooled liquid_from_solid branch below the {fusion_temperature:g} K" in flags


def test_k2o_liquid_from_solid_construction_is_flagged(
    gas_pack: ImccGasDatapack,
) -> None:
    result = evaluate_gas(
        {"K2O": 1.0e-3}, 1200.0, 1.0e-8, gas_pack,
        parent_oxides=("K2O",), gas_species=("K", "K2", "KO", "K2O"),
    )
    for species in ("K", "K2", "KO", "K2O"):
        assert "tier-4 liquid_from_solid construction from K-012" in (
            result.domain_flags[species]
        )


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


def test_vaporock_override_drops_synthetic_unnamed_columns_with_notice(
    monkeypatch: pytest.MonkeyPatch,
    gas_pack: ImccGasDatapack,
    tmp_path: Path,
) -> None:
    fixture = Path(__file__).parent / "fixtures" / "condensate-thermo-data.csv"
    gas_fixture = Path(__file__).parent / "fixtures" / "JANAF-vapor-data-full.csv"
    condensate_bytes = fixture.read_bytes()
    assert condensate_bytes.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" in condensate_bytes
    assert b"Ref,,,\r\n" in condensate_bytes
    assert b"SYNTHETIC,,,not-source-data\r\n" in condensate_bytes
    root = tmp_path / "vaporock"
    gas_target = root / "src" / "vaporock" / "data"
    oxide_target = root / "data"
    gas_target.mkdir(parents=True)
    oxide_target.mkdir(parents=True)
    shutil.copy2(gas_fixture, gas_target / "JANAF-vapor-data-full.csv")
    shutil.copy2(
        fixture,
        oxide_target / "condensate-thermo-data.csv",
    )
    monkeypatch.setenv("OPENIMCC_VAPOROCK_ROOT", str(root))

    with pytest.warns(ImccGasIngestionWarning) as notices:
        overridden = load_gas_datapack()
    message = str(notices[0].message)
    assert "Unnamed: 14" in message
    assert "Unnamed: 15" in message
    assert "Unnamed: 16" in message
    assert "Unnamed: 16 at row 1 (SYNTHETIC-ROW, SYNTHETIC) = not-source-data" in message
    assert tuple(overridden.oxide_df.columns) == tuple(gas_pack.oxide_df.columns)
    assert tuple(overridden.gas_df.columns) == tuple(gas_pack.gas_df.columns)


def test_vaporock_checkout_real_file_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_value = os.environ.get("OPENIMCC_TEST_VAPOROCK_ROOT")
    if not root_value:
        pytest.skip("set OPENIMCC_TEST_VAPOROCK_ROOT to a VapoRock checkout")
    root = Path(root_value).expanduser().resolve()
    expected_commit = "0159678a8eefb51dfe3f12bb646dc38507a07230"
    # Read the external checkout's own commit: drop any repository-selection
    # variables inherited from the caller (a harness may set GIT_DIR for this
    # repository, and git -C does not override it).
    external_git_env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR"}
    }
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        env=external_git_env,
    ).stdout.strip()
    assert commit == expected_commit
    condensate = root / "data" / "condensate-thermo-data.csv"
    gas = root / "src" / "vaporock" / "data" / "JANAF-vapor-data-full.csv"
    assert hashlib.sha256(condensate.read_bytes()).hexdigest() == (
        "1d8a0a827df3833861c368f06e08554a2e8896d18576195b896fa759931c491d"
    )
    assert hashlib.sha256(gas.read_bytes()).hexdigest() == (
        "d45d76722f21746a0c73202f6b9d70928d0f285025672fa5593c7032ff182f8a"
    )
    monkeypatch.setenv("OPENIMCC_VAPOROCK_ROOT", str(root))
    with pytest.warns(ImccGasIngestionWarning) as notices:
        datapack = load_gas_datapack()
    message = str(notices[0].message)
    assert "Unnamed: 14" in message
    assert "Unnamed: 15" in message
    assert "Unnamed: 16" in message
    assert "Unnamed: 16 at row 6 (Na2O, LAM1984) = -188.14" in message
    result = evaluate_gas(
        {"Na2O": 0.05}, 1800.0, 1.0e-8, datapack,
        parent_oxides=("Na2O",), gas_species=("Na",),
    )
    assert result["Na"] == 0.04667660400876665


def test_packaged_gas_tables_load_without_ingestion_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ImccGasIngestionWarning)
        load_gas_datapack()
    assert not any(
        issubclass(item.category, ImccGasIngestionWarning) for item in caught
    )


def test_vaporock_override_refuses_named_extra_columns(
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
    monkeypatch.setenv("OPENIMCC_VAPOROCK_ROOT", str(root))

    fixture = Path(__file__).parent / "fixtures" / "condensate-thermo-data.csv"
    legacy_csv = fixture.read_text(encoding="utf-8-sig")
    for named_column in ("unexpected", "Unnamed: 99"):
        (oxide_target / "condensate-thermo-data.csv").write_text(
            legacy_csv.replace("Ref,,,", f"Ref,,,{named_column}", 1),
            encoding="utf-8",
        )
        with pytest.warns(ImccGasIngestionWarning):
            with pytest.raises(ImccGasInvalidIntervalError, match="invalid schema") as refusal:
                load_gas_datapack()
        assert f"extra=['{named_column}']" in str(refusal.value)


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
    public_gas_channels = set(IMCC_GAS_CHANNEL_SPECIES)
    assert set(gas_pack.gas_df.index) == {
        f"{species}(g)" for species in public_gas_channels
    } | {f"{species}(g)" for species in _ION_GAS_SPECIES} | {
        source[0] for source in build_gas_tables.TRACE_ION_GAS_NASA_SOURCES
    } | {
        source[0] for source in build_gas_tables.TRACE_ION_GAS_BURCAT_SOURCES
    }
    high_rows = gas_pack.gas_df.loc[
        gas_pack.gas_df["T_interval"].astype(int) == 1
    ]
    low_rows = gas_pack.gas_df.loc[
        gas_pack.gas_df["T_interval"].astype(int) == 2
    ]
    low_trace_species = (
        build_gas_tables.TRACE_GAS_SPECIES - build_gas_tables.HALOGEN_GAS_SPECIES
    )
    assert len(low_rows) == (
        len(build_gas_tables.LOW_T_GAS_SPECIES)
        + 2
        + len(low_trace_species)
        + len(build_gas_tables.TRACE_ION_GAS_NASA_SOURCES)
        + len(build_gas_tables.TRACE_ION_NASA_SOURCES)
        + len(build_gas_tables.TRACE_ION_GAS_YAML_SOURCES)
        + 3
        # Halogen rows use one 1200-3000 K interval and have no low interval.
        + len(_RARE_EARTH_CHANNELS)
        + 3
    )
    assert set(low_rows.index) == {
        f"{species}(g)" for species in build_gas_tables.LOW_T_GAS_SPECIES
    } | {"Na2O(g)", "K2O(g)"} | {
        f"{species}(g)" for species in low_trace_species
    } | {
        f"{species}(g)" for species in _RARE_EARTH_CHANNELS
    } | {f"{species}(g)" for species in ("Hg", "Au", "Pd")} | {
        source[0]
        for source in build_gas_tables.TRACE_ION_GAS_NASA_SOURCES
    } | {
        species for species, *_ in build_gas_tables.TRACE_ION_NASA_SOURCES
    } | {
        source[0] for source in build_gas_tables.TRACE_ION_GAS_YAML_SOURCES
    } | {"Zn+(g)", "Zn-(g)", "Bi+(g)"}
    neutral_high_rows = high_rows.loc[
        ~high_rows.index.isin(tuple(f"{species}(g)" for species in _ION_GAS_SPECIES))
    ]
    ion_rows = gas_pack.gas_df.loc[
        gas_pack.gas_df.index.isin(tuple(f"{species}(g)" for species in _ION_GAS_SPECIES))
    ]
    split_ion_names = {
        source[0] for source in build_gas_tables.TRACE_ION_GAS_NASA_SOURCES
    } | {
        source[0] for source in build_gas_tables.TRACE_ION_NASA_SOURCES
    } | {
        source[0] for source in build_gas_tables.TRACE_ION_GAS_YAML_SOURCES
    } | {"Zn+(g)", "Zn-(g)", "Bi+(g)"}
    split_ion_rows = ion_rows.loc[ion_rows.index.isin(split_ion_names)]
    single_interval_ion_rows = ion_rows.loc[
        ~ion_rows.index.isin(split_ion_names)
    ]
    reviewed_trace_species = {
        f"{source[0]}" for source in build_gas_tables.TRACE_PUBLISHED_GAS_SOURCES
        if source[1].startswith(("P672-Te", "P672-Se", "P672-As", "P672-Sb", "P672-Tl"))
    }
    ordinary_high_rows = neutral_high_rows.loc[
        ~neutral_high_rows.index.isin(reviewed_trace_species)
        & ~neutral_high_rows.index.isin(
            tuple(f"{species}(g)" for species in build_gas_tables.HALOGEN_GAS_SPECIES)
        )
    ]
    assert (ordinary_high_rows["T_min"] == 1500).all()
    for source in build_gas_tables.TRACE_PUBLISHED_GAS_SOURCES:
        if source[1].startswith(("P672-Te", "P672-Se", "P672-As", "P672-Sb", "P672-Tl")):
            split = min(max(source[6], 1500.0), 2000.0)
            assert float(high_rows.loc[source[0], "T_min"]) == split
            assert float(low_rows.loc[source[0], "T_max"]) == split
    assert (single_interval_ion_rows["T_min"] == 1200).all()
    assert (single_interval_ion_rows["T_max"] == 3000).all()
    assert (split_ion_rows.loc[split_ion_rows["T_interval"] == 1, "T_min"] == 1500).all()
    assert (split_ion_rows.loc[split_ion_rows["T_interval"] == 1, "T_max"] == 3000).all()
    assert (split_ion_rows.loc[split_ion_rows["T_interval"] == 2, "T_min"] == 1200).all()
    assert (split_ion_rows.loc[split_ion_rows["T_interval"] == 2, "T_max"] == 1500).all()
    trace_low_species = {
        f"{species}(g)" for species in build_gas_tables.TRACE_GAS_SPECIES
    } | split_ion_names | {f"{species}(g)" for species in _RARE_EARTH_CHANNELS}
    trace_low_species |= {"Hg(g)", "Au(g)", "Pd(g)"}
    legacy_low_rows = low_rows.loc[~low_rows.index.isin(trace_low_species)]
    trace_low_rows = low_rows.loc[low_rows.index.isin(trace_low_species)]
    assert (legacy_low_rows["T_min"] == build_gas_tables.LOW_FIT_T_MIN).all()
    assert (trace_low_rows["T_min"] == 1200.0).all()
    assert (legacy_low_rows["T_max"] == build_gas_tables.LOW_FIT_T_MAX).all()
    expected_t_max = high_rows["Ref"].map(
        lambda table_id: build_gas_tables._DECLARED_T_MAX_BY_TABLE.get(
            table_id,
            build_gas_tables._FIT_T_MAX_BY_TABLE.get(
                table_id, build_gas_tables.FIT_T_MAX
            ),
        )
    )
    assert (high_rows["T_max"].astype(float) == expected_t_max).all()


def test_interval_selection_uses_high_row_at_shared_1500_k_node(
    gas_pack: ImccGasDatapack,
) -> None:
    for species in build_gas_tables.LOW_T_GAS_SPECIES:
        high = _nearest_interval_row(gas_pack.gas_df, f"{species}(g)", 1500.0)
        low = _nearest_interval_row(gas_pack.gas_df, f"{species}(g)", 1499.999)
        pyrolysis = _nearest_interval_row(gas_pack.gas_df, f"{species}(g)", 1200.0)
        assert int(high["T_interval"]) == 1
        assert float(high["T_min"]) == 1500.0
        assert int(low["T_interval"]) == 2
        assert int(pyrolysis["T_interval"]) == 2
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
    assert len(implemented) == (
        64
        + 3
        + len(_RARE_EARTH_CHANNELS)
        + len(build_gas_tables.TRACE_GAS_SPECIES)
        - len({"NiO", "CoO"} & build_gas_tables.TRACE_GAS_SPECIES)
    )
    assert unavailable == {"TeO", "AsO"}
    assert unavailable & implemented == set()
    assert unavailable - implemented == {"TeO", "AsO"}
    assert in_domain.isdisjoint(extrapolated)
    assert not set(_P_CHANNELS) & (in_domain | extrapolated)
    assert (
        in_domain
        | extrapolated
        | set(_P_CHANNELS)
        | set(_RARE_EARTH_CHANNELS)
        | {"Hg", "Au", "Pd"}
        | build_gas_tables.TRACE_GAS_SPECIES
        | (unavailable & implemented)
    ) == implemented


_NEWLY_COVERED_PARENT_CASES = tuple(
    (species, temperature, "Cr2O3(l)")
    for species in ("Cr", "CrO", "CrO2", "CrO3")
    for temperature in (1500.0, 1600.0, 1700.0, 1800.0, 1900.0)
)


@pytest.mark.parametrize(
    ("species", "temperature", "source_species"), _NEWLY_COVERED_PARENT_CASES
)
def test_parent_domain_gaps_use_labelled_extensions(
    gas_pack: ImccGasDatapack,
    unit_activities: dict[str, float],
    species: str,
    temperature: float,
    source_species: str,
) -> None:
    assert set(IMCC_GAS_WORKBOOK_EXTRAPOLATION_LABELS) == set()
    activities = (
        {**unit_activities, "Cr2O3": 1.0}
        if species in _CR_CHANNELS
        else unit_activities
    )
    result = evaluate_gas(
        activities,
        temperature,
        1.0,
        gas_pack,
        gas_species=(species,),
        allow_extrapolation=False,
    )
    assert math.isfinite(result[species])
    flag = result.domain_flags[species] or ""
    if temperature == 1900.0:
        assert "constant-Cp supercooled-liquid continuation" not in flag
    else:
        assert "constant-Cp supercooled-liquid continuation" in flag
        assert source_species in flag


@pytest.mark.parametrize(
    ("oxide", "species"),
    (("SiO2", "SiO"), ("Al2O3", "Al"), ("MgO", "Mg"), ("CaO", "Ca")),
)
def test_major_parent_calls_below_continuations_still_refuse(
    gas_pack: ImccGasDatapack, oxide: str, species: str
) -> None:
    with pytest.raises(ImccGasTemperatureOutsideDomainError):
        evaluate_gas(
            {oxide: 1.0},
            1199.0,
            1.0e-10,
            gas_pack,
            gas_species=(species,),
            allow_extrapolation=False,
        )


@pytest.mark.parametrize(("oxide", "species"), (("TiO2", "Ti"), ("V2O3", "V")))
def test_ti_and_v_gas_rows_use_the_low_fit_below_1500_k(
    gas_pack: ImccGasDatapack, oxide: str, species: str
) -> None:
    result = evaluate_gas(
        {oxide: 1.0},
        1200.0,
        1.0e-10,
        gas_pack,
        gas_species=(species,),
        allow_extrapolation=False,
    )
    assert math.isfinite(result[species])
    assert "extrapolation" not in result.domain_flags[species]


def test_gas_domain_refusal_is_typed(gas_pack: ImccGasDatapack) -> None:
    """A selected gas-row gap still refuses before evaluating the reaction."""
    gas_df = gas_pack.gas_df.copy()
    # Keep O2 available at the diagnostic temperature so the refusal names the
    # selected Fe(g) row rather than the caller-pinned reference row.
    o2_low = (gas_df.index == "O2(g)") & (gas_df["T_interval"].astype(int) == 2)
    gas_df.loc[o2_low, "T_min"] = 1000.0
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


def test_default_janaf_basalt_at_1800_predicts_and_flags_source_rows(
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
        "Ti",
        "TiO",
        "TiO2",
    }
    assert all(result.domain_flags[species] is not None for species in flagged)
    assert all(
        result.domain_flags[species] is None
        for species in set(result) - flagged - {"K", "K2", "KO", "K2O"}
    )
    assert "MgO(l)" in result.domain_flags["Mg"]
    assert "constant-Cp supercooled-liquid continuation" in result.domain_flags["Mg"]
    assert result.domain_flags["SiO"] is None
    assert "2130 K" in result.domain_flags["Ti"]
    assert result.provenance_class["SiO"] == "janaf_fitted"


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
    sf04_gas_pack: ImccGasDatapack,
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
        sf04_gas_pack,
        allow_extrapolation=True,
        gas_species=("Na", "K", "SiO", "O2"),
    )

    assert result.unit == "bar"
    assert dict(result) == {
        "Na": 0.22579882807502985,
        "K": 2.1523860349175448,
        "SiO": 1.741275855555553e-08,
        "O2": 1.0,
    }
    assert result.domain_flags["Na"] is None
    assert result.provenance_class["Na"] == "janaf_fitted"
    with pytest.raises(TypeError):
        result.domain_flags["Na"] = "mutated"  # type: ignore[index]
    with pytest.raises((AttributeError, TypeError)):
        result.unit = "Pa"  # type: ignore[misc]


def test_opt_in_ions_preserve_neutral_outputs_and_close_charge(
    gas_pack: ImccGasDatapack,
) -> None:
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
    outputs = {}
    for temperature in (1200.0, 3000.0):
        melt = evaluate(
            composition,
            T_K=temperature,
            basis_type="wt",
            allow_extrapolation=True,
        )
        activities = dict(zip(melt.parent_oxides, melt.parent_activity))
        activities.update({"Cr2O3": 1.0e-3, "V2O3": 1.0e-3, "NbO2": 1.0e-3})
        default = evaluate_gas(
            activities, temperature, 1.0e-10, gas_pack
        )
        explicit_default = evaluate_gas(
            activities,
            temperature,
            1.0e-10,
            gas_pack,
            include_ions=False,
        )
        with_ions = evaluate_gas(
            activities,
            temperature,
            1.0e-10,
            gas_pack,
            include_ions=True,
        )
        neutral_only = {
            species: pressure
            for species, pressure in with_ions.items()
            if not species.endswith(("+", "-"))
        }
        assert dict(default) == dict(explicit_default) == neutral_only
        outputs[temperature] = with_ions

        positive = sum(with_ions[species] for species in ("Na+", "K+", "Ca+"))
        negative = sum(
            with_ions[species]
            for species in (
                "Na-", "K-", "O-", "Al-", "Fe-", "Si-", "Ti-", "O2-", "AlO-",
                "AlO2-", "KO-", "NaO-", "Cr-", "V-", "Nb-",
            )
        )
        residual = with_ions["e-"] + negative - positive
        scale = max(with_ions["e-"] + negative, positive)
        assert abs(residual) <= 1.0e-12 * scale
        assert with_ions.provenance_class["Na+"] == "janaf_fitted_ionisation"
        assert gas_species_provenance("Na+")["authority"] == (
            "janaf_fitted_ionisation"
        )
        if temperature == 3000.0:
            assert "summed neutral pressure" in with_ions.domain_flags["Na+"]
            assert "exceeds 1 bar" in with_ions.domain_flags["Na+"]
            assert "not credible" in with_ions.domain_flags["Na+"]
            assert "summed neutral pressure" in with_ions.domain_flags["e-"]
            assert "summed neutral pressure" not in (
                with_ions.domain_flags.get("Na") or ""
            )
        else:
            assert with_ions.domain_flags["Na+"] is not None
            assert "1405.2 K" in with_ions.domain_flags["Na+"]

    for element in ("Na", "K", "Ca"):
        low = outputs[1200.0]
        high = outputs[3000.0]
        assert low[element + "+"] < high[element + "+"]
        assert low[element + "+"] / low[element] < high[element + "+"] / high[element]
    low = outputs[1200.0]
    high = outputs[3000.0]
    for species in ("e-", *(name for name in _ION_GAS_SPECIES if name != "e-")):
        if species in low and species in high:
            assert low[species] < high[species], species


def test_ion_subsets_use_full_gas_charge_closure(
    gas_pack: ImccGasDatapack,
) -> None:
    activities = _quickstart_activities(3000.0)
    full = evaluate_gas(
        activities, 3000.0, 1.0e-12, gas_pack, include_ions=True
    )
    potassium = evaluate_gas(
        activities, 3000.0, 1.0e-12, gas_pack,
        gas_species=("K",), include_ions=True,
    )
    silica = evaluate_gas(
        activities, 3000.0, 1.0e-12, gas_pack,
        gas_species=("SiO",), include_ions=True,
    )
    sodium_ion = evaluate_gas(
        activities, 3000.0, 1.0e-12, gas_pack,
        gas_species=("Na+",), include_ions=True,
    )

    assert potassium["K+"] / potassium["K"] == pytest.approx(
        full["K+"] / full["K"], rel=1.0e-12
    )
    assert potassium["e-"] == pytest.approx(full["e-"], rel=1.0e-12)
    assert silica["e-"] == pytest.approx(full["e-"], rel=1.0e-12)
    assert silica["e-"] > 0.0
    assert sodium_ion["Na+"] == full["Na+"]
    assert sodium_ion["e-"] == pytest.approx(full["e-"], rel=1.0e-12)

    with pytest.raises(ImccGasSpeciesNotFoundError, match="include_ions=True"):
        evaluate_gas(
            activities, 3000.0, 1.0e-12, gas_pack, gas_species=("Na+",)
        )


def test_cu_minus_is_opt_in_and_cu_plus_remains_declined(
    gas_pack: ImccGasDatapack,
) -> None:
    activities = _quickstart_activities(2000.0)
    activities["Cu2O"] = 1.0e-3
    neutral = evaluate_gas(activities, 2000.0, 1.0e-8, gas_pack)
    ions = evaluate_gas(
        activities, 2000.0, 1.0e-8, gas_pack, include_ions=True
    )
    assert "Cu-" not in neutral
    assert ions["Cu-"] > 0.0
    assert "Cu+" not in ions
    assert ions.provenance_class["Cu-"] == "janaf_fitted_ionisation"


def test_non_na_gas_channels_match_the_base_na2o_row_bit_for_bit(
    gas_pack: ImccGasDatapack,
) -> None:
    """Compare every channel except Na with the base commit's Na2O(l) row."""
    baseline_oxide = gas_pack.oxide_df.loc[
        gas_pack.oxide_df.index != "Na2O(l)"
    ].copy(deep=True)
    base_na2o = pd.DataFrame(
        [
            {
                    "state": "l",
                    "cation": "Na",
                    "cat_num": 2,
                    "oxy_num": 1,
                    "T_min": 1405,
                    "T_max": 3000,
                "dH298_R": -50.17,
                "dG_A": 7.67,
                "dG_B": 6.193,
                "dG_C": 0.0,
                "dG_D": 0.0,
                "dG_E": 0.0,
                "Ref": "LAM1984",
            }
        ],
        index=pd.Index(["Na2O(l)"], name=gas_pack.oxide_df.index.name),
    )
    baseline_oxide = pd.concat((baseline_oxide, base_na2o))
    baseline = replace(gas_pack, oxide_df=baseline_oxide)
    species = tuple(
        name
        for name in dict.fromkeys(
            (*IMCC_GAS_CHANNEL_SPECIES, *_CR_CHANNELS, *_VNB_CHANNELS)
        )
        if not name.startswith("Na")
        and f"{name}(g)" in gas_pack.gas_df.index
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
            elif species in _MN_CHANNELS:
                activities = {**unit_activities, "MnO": 1.0}
            elif species in _NICO_CHANNELS:
                activities = {
                    **unit_activities,
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
    sf04_gas_pack: ImccGasDatapack, unit_activities: dict[str, float]
) -> None:
    """The 2000 K rung-2b sample is independent of the fitted gas rows."""
    T = 2000.0
    pressures = evaluate_gas(
        unit_activities, T, fO2=1.0, datapack=sf04_gas_pack, allow_extrapolation=True
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
    sf04_gas_pack: ImccGasDatapack, unit_activities: dict[str, float]
) -> None:
    """The T-625 additions agree with independently rounded printed lgK."""
    T = 2500.0
    refs = _T625_AGAINST_JANAF_PRINTED_GAS_COLUMNS[T]
    pressures = evaluate_gas(
        unit_activities,
        T,
        fO2=1.0,
        datapack=sf04_gas_pack,
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


def test_kp_derivation_spot_check_sio(sf04_gas_pack: ImccGasDatapack) -> None:
    """Recompute SiO2(l) -> SiO(g) + 1/2 O2(g) at 2000 K by hand."""
    R = 8.314462618
    T = 2000.0
    G_sio2_l = -1129878.93
    G_sio_g = -593626.03
    G_o2_g = -478320.38
    dG = G_sio_g + 0.5 * G_o2_g - G_sio2_l
    Kp_hand = math.exp(-dG / (R * T))

    p_sio = evaluate_gas(
        {"SiO2": 1.0}, T, fO2=1.0, datapack=sf04_gas_pack, allow_extrapolation=True
    )["SiO"]

    # The new fit's measured relative residual is 4.27e-5 against these
    # rounded hand values; retain the original 1e-4 check margin.
    assert p_sio == pytest.approx(Kp_hand, rel=1e-4)


def test_authority_query_reports_k2o_liquid_construction() -> None:
    assert gas_species_provenance("Na")["authority"] == "janaf_fitted"
    for species in ("K", "K2", "KO"):
        provenance = gas_species_provenance(species)
        assert provenance["authority"] == "estimated_systematic"
        assert (
            provenance["condensate_authority"]
            == "estimated_systematic"
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
    assert set(IMCC_GAS_NO_JANAF_ROWS) == {"TeO", "AsO"}
    assert IMCC_GAS_INCOMPLETE_PARENT_SPECIES == {}
    assert all(
        source.startswith("needs") or "; needs" in source
        for source in IMCC_GAS_UNAVAILABLE_SPECIES.values()
    )
    assert all(
        f"{species}(g)" not in gas_pack.gas_df.index
        for species in IMCC_GAS_UNAVAILABLE_SPECIES
    )
    assert {"Mn(g)", "MnO(g)", "Mn+(g)", "Ni(g)", "NiO(g)", "Ni+(g)", "Ni-(g)", "Co(g)", "CoO(g)", "Co+(g)", "Co-(g)"} <= set(
        gas_pack.gas_df.index
    )
    assert (set(_MN_CHANNELS) | set(_NICO_CHANNELS)) <= set(
        IMCC_GAS_CHANNEL_SPECIES
    )
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
_MN_CHANNELS = ("Mn", "MnO")
_NICO_CHANNELS = ("Ni", "Co")
_P_CHANNELS = ("P", "P2", "P4", "PO", "PO2", "P4O6", "P4O10")
_S_CHANNELS = (
    "S", "S2", "S3", "S4", "S5", "S6", "S7", "S8",
    "SO", "SO2", "SO3", "SSO",
)
_PS_CHANNELS = _P_CHANNELS + _S_CHANNELS
_RARE_EARTH_CHANNELS = ("La", "LaO", "Ce", "CeO", "CeO2", "Pr", "PrO", "Nd", "NdO", "Sm", "SmO", "Eu", "EuO", "Gd", "GdO", "Tb", "TbO", "Dy", "DyO", "Ho", "HoO", "Er", "ErO", "Tm", "TmO", "Yb", "YbO", "Lu", "LuO")
_NICO_PARENT_PAIRS = (("Ni", "NiO"), ("Co", "CoO"))
_PRE_GATED_CHANNELS = tuple(
    species
    for species in IMCC_GAS_CHANNEL_SPECIES
    if species not in _CR_CHANNELS
    and species not in _VNB_CHANNELS
    and species not in _MN_CHANNELS
    and species not in _NICO_CHANNELS
    and species not in {"NiO", "CoO"}
    and species not in _PS_CHANNELS
    and species not in _RARE_EARTH_CHANNELS
    and species not in build_gas_tables.TRACE_GAS_SPECIES
    and species not in {"Hg", "Au", "Pd"}
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


def _janaf_apparent_gibbs(table_id: str, temperature: float) -> float:
    source = build_gas_tables._load_record(
        Path(__file__).resolve().parents[1] / "data-src" / "janaf" / f"{table_id}.yaml"
    )
    rows = source["table"]["values"]
    reference = next(
        row for row in rows if float(row["temperature"]["value"]) == 298.15
    )
    row = next(
        row for row in rows if float(row["temperature"]["value"]) == temperature
    )
    h298 = float(reference["formation_enthalpy"]["value"])
    h_increment = float(row["enthalpy_increment"]["value"])
    entropy = float(row["entropy"]["value"])
    return (h298 + h_increment) * 1000.0 - temperature * entropy


def test_sulfur_pressures_match_hand_calculated_janaf_cells(
    gas_pack: ImccGasDatapack,
) -> None:
    """Use JANAF G° cells and nS2*S2 + nO2*O2 = gas, without fitted rows."""
    product_tables = {
        "S": "S-006", "S2": "S-012", "SO": "O-010", "SO2": "O-034",
        "SSO": "O-011",
    }
    parent_activity = 1.0e-3
    species = tuple(product_tables)
    for temperature in (1800.0, 2200.0):
        G_S2 = _janaf_apparent_gibbs("S-012", temperature)
        G_O2 = _janaf_apparent_gibbs("O-029", temperature)
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
                G_product = _janaf_apparent_gibbs(
                    product_tables[gas], temperature
                )
                dG_apparent = (
                    G_product
                    - n_S2 * G_S2
                    - n_O2 * G_O2
                )
                Kp = math.exp(-dG_apparent / (R_J_MOL_K * temperature))
                hand_pressure = (
                    Kp * parent_activity**n_S2 * fO2**n_O2
                )
                assert result[gas] == pytest.approx(hand_pressure, rel=1.5e-3)
                if gas == "SSO":
                    dG_printed = (
                        _janaf_formation_gibbs("O-011", temperature)
                        - n_S2 * _janaf_formation_gibbs("S-012", temperature)
                        - n_O2 * _janaf_formation_gibbs("O-029", temperature)
                    )
                    if temperature == 2200.0:
                        assert dG_printed - dG_apparent == pytest.approx(
                            -231308.9, abs=0.1
                        )

    gap_298 = (
        _janaf_formation_gibbs("O-011", 298.15)
        - _janaf_formation_gibbs("S-012", 298.15)
        - 0.5 * _janaf_formation_gibbs("O-029", 298.15)
        - (
            _janaf_apparent_gibbs("O-011", 298.15)
            - _janaf_apparent_gibbs("S-012", 298.15)
            - 0.5 * _janaf_apparent_gibbs("O-029", 298.15)
        )
    )
    assert gap_298 == pytest.approx(-9556.67, abs=0.1)


def test_halogen_pressures_match_hand_calculated_source_cells(
    gas_pack: ImccGasDatapack,
) -> None:
    """Use JANAF G cells and NASA NaI cards with the specified reaction balances."""
    janaf_products = {
        "Br2": "Br-040", "Br": "Br-001", "I2": "I-027", "I": "I-001",
        "NaBr": "Br-030", "(NaBr)2": "Br-064", "KBr": "Br-018",
        "(KBr)2": "Br-055", "KI": "I-007", "(KI)2": "I-028",
        "Cl2": "Cl-073", "Cl": "Cl-001", "F2": "F-054", "F": "F-001",
        "NaCl": "Cl-056", "(NaCl)2": "Cl-098", "KCl": "Cl-039",
        "(KCl)2": "Cl-091", "NaF": "F-036", "(NaF)2": "F-082",
        "KF": "F-020", "(KF)2": "F-069",
    }
    halides = {
        "NaBr": ("Na2O", "Br-030", "Br-040", 0.5, 0.5, 0.25),
        "(NaBr)2": ("Na2O", "Br-064", "Br-040", 1.0, 1.0, 0.5),
        "KBr": ("K2O", "Br-018", "Br-040", 0.5, 0.5, 0.25),
        "(KBr)2": ("K2O", "Br-055", "Br-040", 1.0, 1.0, 0.5),
        "NaI": ("Na2O", "NG-0893", "I-027", 0.5, 0.5, 0.25),
        "(NaI)2": ("Na2O", "NG-0904", "I-027", 1.0, 1.0, 0.5),
        "KI": ("K2O", "I-007", "I-027", 0.5, 0.5, 0.25),
        "(KI)2": ("K2O", "I-028", "I-027", 1.0, 1.0, 0.5),
        "NaCl": ("Na2O", "Cl-056", "Cl-073", 0.5, 0.5, 0.25),
        "(NaCl)2": ("Na2O", "Cl-098", "Cl-073", 1.0, 1.0, 0.5),
        "KCl": ("K2O", "Cl-039", "Cl-073", 0.5, 0.5, 0.25),
        "(KCl)2": ("K2O", "Cl-091", "Cl-073", 1.0, 1.0, 0.5),
        "NaF": ("Na2O", "F-036", "F-054", 0.5, 0.5, 0.25),
        "(NaF)2": ("Na2O", "F-082", "F-054", 1.0, 1.0, 0.5),
        "KF": ("K2O", "F-020", "F-054", 0.5, 0.5, 0.25),
        "(KF)2": ("K2O", "F-069", "F-054", 1.0, 1.0, 0.5),
    }
    species = (*janaf_products, *halides)
    parent_activities = {
        "Br2": 2.0e-3, "I2": 3.0e-3, "Cl2": 4.0e-3, "F2": 5.0e-3,
        "Na2O": 2.0e-4, "K2O": 3.0e-5,
    }
    for temperature in (1800.0, 2200.0):
        g_o2 = _janaf_apparent_gibbs("O-029", temperature)
        for f_o2 in (1.0e-10, 1.0e-6):
            result = evaluate_gas(
                parent_activities,
                temperature,
                f_o2,
                gas_pack,
                parent_oxides=(*IMCC_PARENT_OXIDES, "Br2", "I2", "Cl2", "F2"),
                gas_species=species,
                allow_extrapolation=False,
            )
            for gas, table_id in janaf_products.items():
                if gas in {"Br2", "I2", "Cl2", "F2"}:
                    assert result[gas] == parent_activities[gas]
                elif gas in {"Br", "I", "Cl", "F"}:
                    parent = {
                        "Br": "Br-040", "I": "I-027", "Cl": "Cl-073", "F": "F-054"
                    }[gas]
                    n_parent = 0.5
                    delta_g = _janaf_apparent_gibbs(table_id, temperature) - n_parent * _janaf_apparent_gibbs(parent, temperature)
                    parent_activity = parent_activities[parent.removesuffix("-040").removesuffix("-027").removesuffix("-073").removesuffix("-054") + "2"]
                    expected = math.exp(-delta_g / (R_J_MOL_K * temperature)) * parent_activity**n_parent
                    assert result[gas] == pytest.approx(expected, rel=1.5e-3)
            for gas, (oxide, product, parent, n_oxide, n_parent, n_o2) in halides.items():
                if product.startswith("NG-"):
                    card = build_gas_tables._load_record(
                        Path(__file__).resolve().parents[1] / "data-src" / "nasa-glenn" / f"{product}.json"
                    )
                    props = build_gas_tables._nasa7_properties(card, temperature)
                    props_298 = build_gas_tables._nasa7_properties(card, 298.15)
                    h_298 = R_J_MOL_K * 298.15 * props_298["h_rt"]
                    g_product = float(card["delta_f_H_298_15"]["value"]) + (R_J_MOL_K * temperature * props["h_rt"] - h_298) - temperature * R_J_MOL_K * props["s_R"]
                else:
                    g_product = _janaf_apparent_gibbs(product, temperature)
                if oxide == "Na2O":
                    g_oxide = _janaf_apparent_gibbs("Na-013", temperature)
                else:
                    oxide_row = _oxide_row_for_T(
                        gas_pack.oxide_df, "K2O(l)", temperature,
                        allow_extrapolation=False,
                    )
                    g_oxide = _lamor_gibbs(temperature, oxide_row)
                delta_g = (
                    g_product + n_o2 * g_o2
                    - n_oxide * g_oxide
                    - n_parent * _janaf_apparent_gibbs(parent, temperature)
                )
                expected = (
                    math.exp(-delta_g / (R_J_MOL_K * temperature))
                    * parent_activities[oxide] ** n_oxide
                    * parent_activities["Br2" if "Br" in gas else "I2" if "I" in gas else "Cl2" if "Cl" in gas else "F2"] ** n_parent
                    / f_o2**n_o2
                )
                assert result[gas] == pytest.approx(expected, rel=2e-3)


def test_new_halide_pressures_match_hand_calculated_source_cells(
    gas_pack: ImccGasDatapack,
) -> None:
    """Close halide pressures from source cells and family equations, independently of gas.py."""
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def janaf_gibbs(table_id: str, temperature: float) -> float:
        return _janaf_apparent_gibbs(table_id, temperature)

    @lru_cache(maxsize=None)
    def nasa_gibbs(card_id: str, temperature: float) -> float:
        card = build_gas_tables._load_record(
            Path(__file__).resolve().parents[1]
            / "data-src" / "nasa-glenn" / f"{card_id}.json"
        )
        props = build_gas_tables._nasa7_properties(card, temperature)
        props_298 = build_gas_tables._nasa7_properties(card, 298.15)
        h_298 = R_J_MOL_K * 298.15 * props_298["h_rt"]
        return (
            float(card["delta_f_H_298_15"]["value"])
            + R_J_MOL_K * temperature * props["h_rt"]
            - h_298 - temperature * R_J_MOL_K * props["s_R"]
        )
    source_by_species = {
        spec[0].removesuffix("(g)"): spec[1]
        for spec in build_gas_tables.TRACE_JANAF_YAML_GAS_SOURCES
        if spec[1] in build_gas_tables.HALOGEN_JANAF_TABLES
        and spec[1] not in {
            "Br-040", "Br-001", "I-027", "I-001", "Br-030", "Br-064",
            "Br-018", "Br-055", "I-007", "I-028", "Cl-073", "Cl-001",
            "Cl-056", "Cl-098", "Cl-039", "Cl-091", "F-054", "F-001",
            "F-036", "F-082", "F-020", "F-069", "Cl-026", "F-005",
            "Br-010", "H-005",
        }
    }
    source_by_species.update({
        spec[0].removesuffix("(g)"): spec[1]
        for spec in build_gas_tables.TRACE_NASA_GAS_SOURCES
        if spec[1] in build_gas_tables.HALOGEN_NASA_TABLES
        and spec[1] not in {"NG-0893", "NG-0904"}
    })
    assert len(source_by_species) == 59
    species = tuple(source_by_species)
    activities = {
        "F2": 5.0e-3, "Cl2": 4.0e-3, "Br2": 2.0e-3, "I2": 3.0e-3,
        **{oxide: 1.0e-3 for oxide in (
            "BaO", "CaO", "MgO", "SrO", "Li2O", "Na2O", "K2O", "Rb2O", "Cs2O",
        )},
    }
    parents = (
        *IMCC_PARENT_OXIDES, "BaO", "SrO", "Li2O", "Na2O", "K2O", "Rb2O", "Cs2O",
        "F2", "Cl2", "Br2", "I2",
    )
    oxide_stoichiometry = {
        "Ba": ("BaO", 1.0), "Ca": ("CaO", 1.0),
        "Mg": ("MgO", 1.0), "Sr": ("SrO", 1.0),
        "Li": ("Li2O", 0.5), "Na": ("Na2O", 0.5),
        "K": ("K2O", 0.5), "Rb": ("Rb2O", 0.5),
        "Cs": ("Cs2O", 0.5),
    }
    elemental_halogen_tables = {
        "F": "F-054", "Cl": "Cl-073", "Br": "Br-040", "I": "I-027",
    }
    for temperature in (1800.0, 2200.0):
        g_o2 = janaf_gibbs("O-029", temperature)
        for f_o2 in (1.0e-10, 1.0e-6):
            actual = evaluate_gas(
                activities, temperature, f_o2, gas_pack,
                parent_oxides=parents, gas_species=species,
                allow_extrapolation=False,
            )
            for name, source in source_by_species.items():
                if name in {"Br2", "I2", "Cl2", "F2"}:
                    assert actual[name] == activities[name]
                    continue
                if name in {"Br", "I", "Cl", "F"}:
                    parent = elemental_halogen_tables[name]
                    delta_g = janaf_gibbs(source, temperature) - 0.5 * janaf_gibbs(parent, temperature)
                    expected = math.exp(-delta_g / (R_J_MOL_K * temperature)) * activities[f"{name}2"] ** 0.5
                    assert actual[name] == pytest.approx(expected, rel=1.5e-3)
                    continue

                metal = next(
                    symbol for symbol in sorted(oxide_stoichiometry, key=len, reverse=True)
                    if name.startswith(symbol) or name.startswith(f"({symbol}")
                )
                formula = name.removeprefix("(").removesuffix(")2")
                is_dimer = name.startswith("(")
                halogen = next(x for x in ("F", "Cl", "Br", "I") if formula.endswith(x) or formula.endswith(x + "2"))
                dihalide = formula.endswith(halogen + "2")
                scale = 2.0 if is_dimer else 1.0
                oxide, oxide_per_formula = oxide_stoichiometry[metal]
                n_oxide = scale * oxide_per_formula
                n_parent = scale * (1.0 if dihalide else 0.5)
                n_o2 = n_oxide / 2.0
                if source.startswith("NG-"):
                    g_product = nasa_gibbs(source, temperature)
                else:
                    g_product = janaf_gibbs(source, temperature)
                oxide_row = _oxide_row_for_T(
                    gas_pack.oxide_df, f"{oxide}(l)", temperature,
                    allow_extrapolation=False,
                )
                delta_g = (
                    g_product + n_o2 * g_o2
                    - n_oxide * _lamor_gibbs(temperature, oxide_row)
                    - n_parent * janaf_gibbs(elemental_halogen_tables[halogen], temperature)
                )
                expected = (
                    math.exp(-delta_g / (R_J_MOL_K * temperature))
                    * activities[oxide] ** n_oxide
                    * activities[f"{halogen}2"] ** n_parent
                    / f_o2**n_o2
                )
                assert actual[name] == pytest.approx(expected, rel=2e-3)


def test_new_halide_mixed_source_parent_bands_recompute_from_cards(
    gas_pack: ImccGasDatapack,
) -> None:
    import openimcc.gas as gas_module
    from functools import lru_cache

    janaf_gibbs = lru_cache(maxsize=None)(_janaf_apparent_gibbs)

    @lru_cache(maxsize=None)
    def nasa_gas_gibbs(card_id: str, temperature: float) -> float:
        card = build_gas_tables._load_record(
            Path(__file__).resolve().parents[1]
            / "data-src" / "nasa-glenn" / f"{card_id}.json"
        )
        props = build_gas_tables._nasa7_properties(card, temperature)
        at_298 = build_gas_tables._nasa7_properties(card, 298.15)
        h_298 = R_J_MOL_K * 298.15 * at_298["h_rt"]
        return (
            float(card["delta_f_H_298_15"]["value"])
            + R_J_MOL_K * temperature * props["h_rt"]
            - h_298 - temperature * R_J_MOL_K * props["s_R"]
        )

    @lru_cache(maxsize=None)
    def nasa_liquid_card_gibbs(card_id: str, temperature: float) -> float:
        card = build_gas_tables._load_record(
            Path(__file__).resolve().parents[1]
            / "data-src" / "nasa-glenn" / f"{card_id}.json"
        )
        props = build_gas_tables._nasa7_properties(card, temperature)
        return R_J_MOL_K * temperature * (
            props["h_rt"] - props["s_R"]
        )

    provenance = yaml.safe_load(
        (Path(__file__).resolve().parents[1]
         / "src/openimcc/data/gas/PROVENANCE.yaml").read_text()
    )
    rows = {
        row["species_name"].removesuffix("(g)"): row
        for row in provenance["rows"]
        if row["table"] == "gas"
        and (
            row["table_id"] in {
                "NG-1020", "NG-1032", "NG-1019", "NG-1031",
                "NG-1018", "NG-1030", "NG-1022", "NG-1033",
                "NG-0511", "NG-0524", "NG-0515", "NG-0528",
            }
            or row["species_name"] in {
                "CsF(g)", "CsCl(g)", "(CsF)2(g)", "(CsCl)2(g)",
            }
        )
    }
    assert len(rows) == 16
    janaf_parent = {
        "F2": "F-054", "Cl2": "Cl-073", "Br2": "Br-040", "I2": "I-027",
    }
    for name, row in rows.items():
        band = row["accuracy_band"]["value"]
        assert band["kind"] == "mixed_source_parent_scale_inconsistency"
        assert set(map(int, band["values_J_per_mol"])) == set(range(1200, 3001, 100))
        oxide, n_oxide, n_o2 = gas_module._ALKALI_HALIDE_REACTION_OXIDES[name]
        parent, n_parent, _ = gas_module._SF04_REACTIONS[name]
        for temperature in range(1200, 3001, 100):
            nasa_parent = band["comparison_records"]["halogen_parent"]
            nasa_o2 = band["comparison_records"]["oxygen_parent"]
            gap = n_parent * (
                nasa_gas_gibbs(nasa_parent, temperature)
                - janaf_gibbs(janaf_parent[parent], temperature)
            ) + n_o2 * (
                nasa_gas_gibbs(nasa_o2, temperature)
                - janaf_gibbs("O-029", temperature)
            )
            if "oxide_parent" in band["comparison_records"]:
                card_id = band["comparison_records"]["oxide_parent"]
                oxide_row = _oxide_row_for_T(
                    gas_pack.oxide_df, f"{oxide}(l)", temperature,
                    allow_extrapolation=False,
                )
                gap += n_oxide * (
                    nasa_liquid_card_gibbs(card_id, temperature)
                    - _lamor_gibbs(temperature, oxide_row)
                )
            assert band["values_J_per_mol"][temperature] == pytest.approx(
                abs(gap), abs=0.001
            ), (name, temperature, band["comparison_records"], gap)


def test_halogen_activity_and_fugacity_powers_and_optional_family(
    gas_pack: ImccGasDatapack,
) -> None:
    parents = (*IMCC_PARENT_OXIDES, "Br2", "I2", "Cl2", "F2")
    activities = {**_quickstart_activities(2000.0), "Br2": 1.0e-3, "I2": 2.0e-3, "Cl2": 3.0e-3, "F2": 4.0e-3}
    species = ("Br2", "Br", "NaBr", "(NaBr)2", "KBr", "(KBr)2", "I2", "I", "NaI", "(NaI)2", "KI", "(KI)2", "Cl2", "Cl", "NaCl", "(NaCl)2", "KCl", "(KCl)2", "F2", "F", "NaF", "(NaF)2", "KF", "(KF)2")
    low = evaluate_gas(activities, 2000.0, 1.0e-10, gas_pack, parent_oxides=parents, gas_species=species)
    high = evaluate_gas(activities, 2000.0, 1.0e-6, gas_pack, parent_oxides=parents, gas_species=species)
    for gas in species:
        exponent = 0.0 if gas in {"Br2", "Br", "I2", "I", "Cl2", "Cl", "F2", "F"} else -0.25 if gas in {"NaBr", "KBr", "NaI", "KI", "NaCl", "KCl", "NaF", "KF"} else -0.5
        assert high[gas] / low[gas] == pytest.approx(1.0e-6**exponent / 1.0e-10**exponent, rel=1e-12)
    assert low["Br"] == pytest.approx(
        evaluate_gas({**activities, "Na2O": activities["Na2O"] * 10}, 2000.0, 1.0e-10, gas_pack, parent_oxides=parents, gas_species=("Br",))["Br"],
        rel=0.0,
    )
    for halogen in ("Cl", "F"):
        assert low[halogen] == pytest.approx(
            evaluate_gas({**activities, "Na2O": activities["Na2O"] * 10}, 2000.0, 1.0e-10, gas_pack, parent_oxides=parents, gas_species=(halogen,))[halogen],
            rel=0.0,
        )
    assert low["NaBr"] == pytest.approx(
        evaluate_gas({**activities, "Br2": activities["Br2"] * 2}, 2000.0, 1.0e-10, gas_pack, parent_oxides=parents, gas_species=("NaBr",))["NaBr"] / 2**0.5,
        rel=1e-12,
    )
    halide_powers = {
        "NaBr": ("Na2O", 0.5, "Br2", 0.5),
        "(NaBr)2": ("Na2O", 1.0, "Br2", 1.0),
        "KBr": ("K2O", 0.5, "Br2", 0.5),
        "(KBr)2": ("K2O", 1.0, "Br2", 1.0),
        "NaI": ("Na2O", 0.5, "I2", 0.5),
        "(NaI)2": ("Na2O", 1.0, "I2", 1.0),
        "KI": ("K2O", 0.5, "I2", 0.5),
        "(KI)2": ("K2O", 1.0, "I2", 1.0),
        "NaCl": ("Na2O", 0.5, "Cl2", 0.5),
        "(NaCl)2": ("Na2O", 1.0, "Cl2", 1.0),
        "KCl": ("K2O", 0.5, "Cl2", 0.5),
        "(KCl)2": ("K2O", 1.0, "Cl2", 1.0),
        "NaF": ("Na2O", 0.5, "F2", 0.5),
        "(NaF)2": ("Na2O", 1.0, "F2", 1.0),
        "KF": ("K2O", 0.5, "F2", 0.5),
        "(KF)2": ("K2O", 1.0, "F2", 1.0),
    }
    for gas, (oxide, n_oxide, parent, n_parent) in halide_powers.items():
        oxide_scaled = evaluate_gas(
            {**activities, oxide: activities[oxide] * 4.0},
            2000.0, 1.0e-10, gas_pack, parent_oxides=parents,
            gas_species=(gas,),
        )[gas]
        parent_scaled = evaluate_gas(
            {**activities, parent: activities[parent] * 2.0},
            2000.0, 1.0e-10, gas_pack, parent_oxides=parents,
            gas_species=(gas,),
        )[gas]
        assert oxide_scaled / low[gas] == pytest.approx(4.0**n_oxide, rel=1e-12)
        assert parent_scaled / low[gas] == pytest.approx(2.0**n_parent, rel=1e-12)
    default = evaluate_gas(_quickstart_activities(2000.0), 2000.0, 1.0e-8, gas_pack)
    assert not set(species) & set(default)
    with pytest.raises(ImccGasSpeciesNotFoundError, match="Br2"):
        evaluate_gas(_quickstart_activities(2000.0), 2000.0, 1.0e-8, gas_pack, gas_species=("NaBr",))
    with pytest.raises(ImccGasSpeciesNotFoundError, match="Cl2"):
        evaluate_gas(_quickstart_activities(2000.0), 2000.0, 1.0e-8, gas_pack, gas_species=("NaCl",))
    zero_na = evaluate_gas(
        {**activities, "Na2O": 0.0}, 2000.0, 1.0e-8, gas_pack,
        parent_oxides=parents, gas_species=("Br", "Br2", "NaBr"),
    )
    assert zero_na["NaBr"] == 0.0
    assert math.isfinite(zero_na["Br"]) and math.isfinite(zero_na["Br2"])


def test_hydrogen_carbon_and_nitrogen_channels_match_hand_mass_action(
    gas_pack: ImccGasDatapack,
) -> None:
    parents = (*IMCC_PARENT_OXIDES, "H2", "C", "N2")
    activities = {
        **_quickstart_activities(1800.0),
        "H2": 2.0e-3,
        "C": 3.0e-4,
        "N2": 4.0e-3,
    }
    species = (
        "H2", "H", "OH", "H2O", "CO", "CO2",
        "N2", "N", "NO", "NO2",
    )
    for temperature in (1800.0, 2400.0):
        for fugacity in (1.0e-10, 1.0e-6):
            result = evaluate_gas(
                activities,
                temperature,
                fugacity,
                gas_pack,
                parent_oxides=parents,
                gas_species=species,
            )
            g_o2 = _janaf_gibbs(
                temperature,
                _nearest_interval_row(gas_pack.gas_df, "O2(g)", temperature),
            )
            g_h2 = _janaf_gibbs(
                temperature,
                _nearest_interval_row(gas_pack.gas_df, "H2(g)", temperature),
            )
            g_n2 = _janaf_gibbs(
                temperature,
                _nearest_interval_row(gas_pack.gas_df, "N2(g)", temperature),
            )
            g_c = _lamor_gibbs(
                temperature,
                _oxide_row_for_T(gas_pack.oxide_df, "C(cr)", temperature),
            )
            g = {
                name: _janaf_gibbs(
                    temperature,
                    _nearest_interval_row(
                        gas_pack.gas_df, f"{name}(g)", temperature
                    ),
                )
                for name in species
                if name not in {"H2", "N2"}
            }
            k = lambda delta_g: math.exp(
                -delta_g / (R_J_MOL_K * temperature)
            )
            expected = {
                "H2": activities["H2"],
                "H": k(g["H"] - 0.5 * g_h2) * activities["H2"] ** 0.5,
                "OH": k(g["OH"] - 0.5 * g_h2 - 0.5 * g_o2)
                * activities["H2"] ** 0.5 * fugacity**0.5,
                "H2O": k(g["H2O"] - g_h2 - 0.5 * g_o2)
                * activities["H2"] * fugacity**0.5,
                "CO": k(g["CO"] - 0.5 * g_o2 - g_c)
                * activities["C"] * fugacity**0.5,
                "CO2": k(g["CO2"] - g_o2 - g_c)
                * activities["C"] * fugacity,
                "N2": activities["N2"],
                "N": k(g["N"] - 0.5 * g_n2) * activities["N2"] ** 0.5,
                "NO": k(g["NO"] - 0.5 * g_n2 - 0.5 * g_o2)
                * activities["N2"] ** 0.5 * fugacity**0.5,
                "NO2": k(g["NO2"] - 0.5 * g_n2 - g_o2)
                * activities["N2"] ** 0.5 * fugacity,
            }
            for name in species:
                assert result[name] == pytest.approx(expected[name], rel=2e-12)


def test_hydrogen_halides_require_both_gas_parent_keys(
    gas_pack: ImccGasDatapack,
) -> None:
    activities = {
        **_quickstart_activities(2000.0),
        "H2": 1.0e-3,
        "Cl2": 2.0e-3,
        "F2": 3.0e-3,
        "Br2": 4.0e-3,
        "I2": 5.0e-3,
    }
    parents = (*IMCC_PARENT_OXIDES, "H2", "Cl2", "F2", "Br2", "I2")
    result = evaluate_gas(
        activities,
        2000.0,
        1.0e-8,
        gas_pack,
        parent_oxides=parents,
        gas_species=("HCl", "HF", "HBr", "HI"),
    )
    for species, halogen_parent in (
        ("HCl", "Cl2"), ("HF", "F2"), ("HBr", "Br2"), ("HI", "I2")
    ):
        g_product = _janaf_gibbs(
            2000.0,
            _nearest_interval_row(gas_pack.gas_df, f"{species}(g)", 2000.0),
        )
        g_h2 = _janaf_gibbs(
            2000.0, _nearest_interval_row(gas_pack.gas_df, "H2(g)", 2000.0)
        )
        g_x2 = _janaf_gibbs(
            2000.0,
            _nearest_interval_row(
                gas_pack.gas_df, f"{halogen_parent}(g)", 2000.0
            ),
        )
        expected = math.exp(
            -(g_product - 0.5 * g_h2 - 0.5 * g_x2)
            / (R_J_MOL_K * 2000.0)
        ) * activities["H2"] ** 0.5 * activities[halogen_parent] ** 0.5
        assert result[species] == pytest.approx(expected, rel=2e-12)


def test_new_element_families_are_optional_and_imposable_in_oxygen_balance(
    gas_pack: ImccGasDatapack,
) -> None:
    baseline_activities = _quickstart_activities(2000.0)
    default = evaluate_gas(baseline_activities, 2000.0, 1.0e-8, gas_pack)
    assert not {
        "H2", "H", "OH", "H2O", "C", "CO", "CO2", "N2", "N", "NO",
        "NO2", "HCl", "HF", "HBr", "HI",
    } & set(default)
    with pytest.raises(ImccGasSpeciesNotFoundError, match="H2"):
        evaluate_gas(
            baseline_activities, 2000.0, 1.0e-8, gas_pack, gas_species="H2O"
        )

    base_root, _, _ = evaluate_gas_oxygen_balance(
        baseline_activities, 2000.0, gas_pack
    )
    for parent, activity, family, carrier in (
        ("H2", 1.0e-3, "H2", "H2O"),
        ("C", 1.0e-4, "C", "CO2"),
        ("N2", 1.0e-3, "N2", "NO2"),
    ):
        with_family = {**baseline_activities, parent: activity}
        root, pressures, _ = evaluate_gas_oxygen_balance(
            with_family, 2000.0, gas_pack
        )
        imposed_root, imposed_pressures, _ = evaluate_gas_oxygen_balance(
            with_family,
            2000.0,
            gas_pack,
            imposed_parents=(parent,),
        )
        assert root < base_root
        assert imposed_root.hex() == base_root.hex()
        assert carrier in pressures and carrier in imposed_pressures


def test_oxygen_balance_includes_each_dict_only_nonoxide_parent(
    gas_pack: ImccGasDatapack,
) -> None:
    with pytest.raises(TypeError, match="sequence of parent-key strings"):
        evaluate_gas_oxygen_balance(
            _quickstart_activities(1800.0),
            1800.0,
            gas_pack,
            imposed_parents="C",
        )

    activities = _quickstart_activities(1800.0, allow_extrapolation=True)
    no_extra_calls = (
        evaluate_gas_oxygen_balance(activities, 1800.0, gas_pack),
        evaluate_gas_oxygen_balance(
            activities, 1800.0, gas_pack, parent_oxides=tuple(activities)
        ),
        evaluate_gas_oxygen_balance(
            activities, 1800.0, gas_pack, imposed_parents=None
        ),
    )
    assert no_extra_calls[0] == no_extra_calls[1] == no_extra_calls[2]
    assert no_extra_calls[0][0].hex() == "0x1.3d1fbf75274c6p-18"

    # The explicit oxide subset keeps excluding every other oxide activity.
    root, pressures, _ = evaluate_gas_oxygen_balance(
        activities,
        1800.0,
        gas_pack,
        parent_oxides=("SiO2", "MgO"),
    )
    assert root.hex() == "0x1.bfa86c0523f92p-25"
    assert len(pressures) == 9
    assert "Na" not in pressures

    extras = (
        {"S2": 1.0e-10},
        {"Br2": 1.0e-10, "I2": 1.0e-10},
        {"Cl2": 1.0e-10, "F2": 1.0e-10},
    )
    for temperature in (1500.0, 1800.0, 2200.0):
        baseline_root, baseline_pressures, _ = evaluate_gas_oxygen_balance(
            activities, temperature, gas_pack
        )
        baseline_listed = evaluate_gas_oxygen_balance(
            activities,
            temperature,
            gas_pack,
            parent_oxides=tuple(activities),
        )
        baseline_imposed_none = evaluate_gas_oxygen_balance(
            activities, temperature, gas_pack, imposed_parents=None
        )
        baseline_imposed_omitted = evaluate_gas_oxygen_balance(
            activities, temperature, gas_pack
        )
        assert baseline_listed == baseline_imposed_none == baseline_imposed_omitted
        for extra in extras:
            with_extra = {**activities, **extra}
            dict_only = evaluate_gas_oxygen_balance(
                with_extra, temperature, gas_pack
            )
            listed = evaluate_gas_oxygen_balance(
                with_extra,
                temperature,
                gas_pack,
                parent_oxides=(*IMCC_PARENT_OXIDES, *extra),
            )
            imposed = evaluate_gas_oxygen_balance(
                with_extra,
                temperature,
                gas_pack,
                imposed_parents=tuple(extra),
            )
            assert dict_only[0].hex() == listed[0].hex()
            assert dict(dict_only[1]) == dict(listed[1])
            assert {key: value.hex() for key, value in dict_only[1].items()} == {
                key: value.hex() for key, value in dict(listed[1]).items()
            }
            assert imposed[0].hex() == baseline_root.hex()
            assert {
                key: value.hex()
                for key, value in dict(imposed[1]).items()
                if key in baseline_pressures
            } == {
                key: value.hex() for key, value in dict(baseline_pressures).items()
            }


def test_halogen_oxygen_balance_uses_dict_only_parent_activity(
    gas_pack: ImccGasDatapack,
) -> None:
    activities = _quickstart_activities(1800.0, allow_extrapolation=True)
    root_without, result_without, _ = evaluate_gas_oxygen_balance(
        activities, 1800.0, gas_pack
    )
    repeated_root, repeated_result, _ = evaluate_gas_oxygen_balance(
        dict(activities), 1800.0, gas_pack
    )
    assert root_without.hex() == repeated_root.hex()
    assert dict(result_without) == dict(repeated_result)
    root_with, result_with, _ = evaluate_gas_oxygen_balance(
        {**activities, "Br2": 1.0e-20}, 1800.0, gas_pack,
    )
    assert root_with != root_without
    assert result_with["Br2"] == pytest.approx(1.0e-20)
    assert result_with["NaBr"] > 0.0
    listed_root, listed_result, _ = evaluate_gas_oxygen_balance(
        {**activities, "Br2": 1.0e-20}, 1800.0, gas_pack,
        parent_oxides=(*IMCC_PARENT_OXIDES, "Br2"),
    )
    assert root_with.hex() == listed_root.hex()
    assert dict(result_with) == dict(listed_result)
    for parent in ("Cl2", "F2"):
        # F2=1e-8 drives this basalt above the 1-bar Knudsen ceiling; 1e-12
        # keeps the fluorine case inside the solver's admissible bracket.
        activity = 1.0e-12 if parent == "F2" else 1.0e-20
        root, result, _ = evaluate_gas_oxygen_balance(
            {**activities, parent: activity}, 1800.0, gas_pack,
            parent_oxides=(*IMCC_PARENT_OXIDES, parent),
        )
        assert root != root_without
        assert result[parent] == pytest.approx(activity)


def test_side_table_halides_inherit_o2_domain_flag(
    gas_pack: ImccGasDatapack, monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluate_gas_globals = evaluate_gas.__globals__
    original = evaluate_gas_globals["_outside_interval_flag"]

    def flag_o2_extrapolation(name: str, temperature: float, row: pd.Series) -> str | None:
        if name == "O2(g)":
            return "mock O2 source extrapolation"
        return original(name, temperature, row)

    monkeypatch.setitem(
        evaluate_gas_globals, "_outside_interval_flag", flag_o2_extrapolation
    )
    result = evaluate_gas(
        {**_quickstart_activities(1800.0), "F2": 1.0e-8},
        1800.0, 1.0e-10, gas_pack,
        parent_oxides=(*IMCC_PARENT_OXIDES, "F2"),
        gas_species=("CaF",),
    )
    assert "mock O2 source extrapolation" in result.domain_flags["CaF"]


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


def test_public_pack_activates_nico_channels_with_caller_parents(
    gas_pack: ImccGasDatapack,
) -> None:
    parent_oxides = (*IMCC_PARENT_OXIDES, "NiO", "CoO")
    activities = {name: 1.0 for name in parent_oxides}
    no_parent = evaluate_gas(
        _quickstart_activities(2200.0),
        2200.0,
        1.0e-10,
        gas_pack,
    )
    assert not set(_NICO_CHANNELS) & set(no_parent)

    result = evaluate_gas(
        activities,
        2200.0,
        1.0e-10,
        gas_pack,
        parent_oxides=parent_oxides,
    )
    assert set(_NICO_CHANNELS) <= set(result)
    assert all(result.provenance_class[name] == "published_fitted" for name in _NICO_CHANNELS)

    for gas, parent in _NICO_PARENT_PAIRS:
        assert f"{parent}(l)" in gas_pack.oxide_df.index
        with pytest.raises(ImccGasSpeciesNotFoundError) as exc:
            evaluate_gas(
                {},
                2200.0,
                1.0e-10,
                gas_pack,
                parent_oxides=(),
                gas_species=(gas,),
            )
        assert exc.value.code == "imcc_gas_species_not_found"
        assert parent in str(exc.value)


def test_public_mn_channels_use_the_mno_parent_and_janaf_ion(
    gas_pack: ImccGasDatapack,
) -> None:
    no_parent_default = evaluate_gas(
        _quickstart_activities(2000.0), 2000.0, 1.0e-6, gas_pack
    )
    assert not set(_MN_CHANNELS) & set(no_parent_default)
    assert not set(_MN_CHANNELS) & set(no_parent_default.omitted_channels)

    activities = {"MnO": 1.0e-3}
    parent_oxides = ("MnO",)
    default = evaluate_gas(
        activities, 2000.0, 1.0e-6, gas_pack, parent_oxides=parent_oxides
    )
    neutral = evaluate_gas(
        activities,
        2000.0,
        1.0e-6,
        gas_pack,
        parent_oxides=parent_oxides,
        gas_species=_MN_CHANNELS,
        include_ions=False,
    )
    with_ions = evaluate_gas(
        activities,
        2000.0,
        1.0e-6,
        gas_pack,
        parent_oxides=parent_oxides,
        gas_species=_MN_CHANNELS,
        include_ions=True,
    )
    assert set(_MN_CHANNELS) <= set(default)
    assert set(_MN_CHANNELS) <= set(neutral)
    assert "Mn+" not in neutral
    assert "Mn+" in with_ions
    assert neutral.provenance_class["Mn"] == "liquid_from_solid_estimate"
    assert neutral.provenance_class["MnO"] == "liquid_from_solid_estimate"
    for species in ("Mn", "MnO"):
        flags = neutral.domain_flags[species]
        assert flags is not None
        assert "tier-4 liquid_from_solid construction from MnO-IVTAN-1435" in flags
        assert "supercooled liquid_from_solid branch" in flags

    missing_mno_gas = replace(
        gas_pack, gas_df=gas_pack.gas_df.drop(index="MnO(g)")
    )
    partial = evaluate_gas(
        activities,
        2000.0,
        1.0e-6,
        missing_mno_gas,
        parent_oxides=parent_oxides,
    )
    assert "Mn" in partial
    assert "MnO" not in partial
    assert partial.omitted_channels["MnO"] == "missing from active table: MnO(g)"

    missing_parent = replace(
        gas_pack, oxide_df=gas_pack.oxide_df.drop(index="MnO(l)")
    )
    with pytest.raises(ImccGasSpeciesNotFoundError) as refusal:
        evaluate_gas(
            activities,
            2000.0,
            1.0e-6,
            missing_parent,
            parent_oxides=parent_oxides,
            gas_species=("Mn",),
        )
    assert refusal.value.code == "imcc_gas_species_not_found"
    assert "MnO(l)" in str(refusal.value)

    without_mn_ion = replace(
        gas_pack, gas_df=gas_pack.gas_df.drop(index="Mn+(g)")
    )
    neutral_without_ion_row = evaluate_gas(
        activities,
        2000.0,
        1.0e-6,
        without_mn_ion,
        parent_oxides=parent_oxides,
        gas_species=_MN_CHANNELS,
        include_ions=False,
    )
    assert dict(neutral) == dict(neutral_without_ion_row)


def test_mn_evaporation_pressures_match_hand_mass_action_at_2000_k(
    gas_pack: ImccGasDatapack,
) -> None:
    temperature = 2000.0
    activity = 2.5e-3
    fugacity = 1.0e-6
    parent_row = _nearest_interval_row(
        gas_pack.oxide_df, "MnO(l)", temperature, allow_extrapolation=False
    )
    parent_gibbs = _lamor_gibbs(temperature, parent_row)
    oxygen_gibbs = _janaf_gibbs(
        temperature,
        _nearest_interval_row(
            gas_pack.gas_df, "O2(g)", temperature, allow_extrapolation=False
        ),
    )
    expected = {}
    for species, oxygen_count in (("Mn", 0.5), ("MnO", 0.0)):
        gas_gibbs = _janaf_gibbs(
            temperature,
            _nearest_interval_row(
                gas_pack.gas_df,
                f"{species}(g)",
                temperature,
                allow_extrapolation=False,
            ),
        )
        delta_g = gas_gibbs + oxygen_count * oxygen_gibbs - parent_gibbs
        expected[species] = (
            activity
            * math.exp(-delta_g / (R_J_MOL_K * temperature))
            * fugacity ** (-oxygen_count)
        )

    result = evaluate_gas(
        {"MnO": activity},
        temperature,
        fugacity,
        gas_pack,
        parent_oxides=("MnO",),
        gas_species=_MN_CHANNELS,
        include_ions=False,
    )
    assert result["Mn"] == pytest.approx(expected["Mn"], rel=1.0e-12)
    assert result["MnO"] == pytest.approx(expected["MnO"], rel=1.0e-12)


def test_nickel_and_cobalt_evaporation_match_hand_mass_action_at_2000_k(
    gas_pack: ImccGasDatapack,
) -> None:
    temperature = 2000.0
    activity = 1.0e-3
    fugacity = 1.0e-6
    parent_oxides = ("NiO", "CoO")
    activities = {parent: activity for parent in parent_oxides}
    oxygen_gibbs = _janaf_gibbs(
        temperature,
        _nearest_interval_row(
            gas_pack.gas_df, "O2(g)", temperature, allow_extrapolation=False
        ),
    )
    expected = {}
    for element, parent in (("Ni", "NiO"), ("Co", "CoO")):
        parent_gibbs = _lamor_gibbs(
            temperature,
            _nearest_interval_row(
                gas_pack.oxide_df,
                f"{parent}(l)",
                temperature,
                allow_extrapolation=False,
            ),
        )
        gas_gibbs = _janaf_gibbs(
            temperature,
            _nearest_interval_row(
                gas_pack.gas_df,
                f"{element}(g)",
                temperature,
                allow_extrapolation=False,
            ),
        )
        delta_g = gas_gibbs + 0.5 * oxygen_gibbs - parent_gibbs
        expected[element] = (
            activity
            * math.exp(-delta_g / (R_J_MOL_K * temperature))
            * fugacity ** -0.5
        )
    result = evaluate_gas(
        activities,
        temperature,
        fugacity,
        gas_pack,
        parent_oxides=parent_oxides,
        gas_species=_NICO_CHANNELS,
        include_ions=False,
    )
    assert result["Ni"] == pytest.approx(expected["Ni"], rel=1.0e-12)
    assert result["Co"] == pytest.approx(expected["Co"], rel=1.0e-12)


def test_external_p2o5_row_sets_phosphorus_provenance(
    gas_pack: ImccGasDatapack, tmp_path: Path
) -> None:
    oxide_df = pd.read_csv(gas_pack.oxide_path)
    p2o5_row = oxide_df.loc[oxide_df["species_name"] == "FeO(l)"].iloc[[0]]
    oxide_path = tmp_path / "external-condensate.csv"
    pd.concat(
        [oxide_df, p2o5_row.assign(species_name="P2O5(l)")], ignore_index=True
    ).to_csv(oxide_path, index=False)
    external_pack = load_gas_datapack(
        gas_path=gas_pack.gas_path, oxide_path=oxide_path
    )

    result = evaluate_gas(
        {"P2O5": 1.0e-3},
        2200.0,
        1.0e-6,
        external_pack,
        parent_oxides=(*IMCC_PARENT_OXIDES, "P2O5"),
        gas_species=_P_CHANNELS,
    )
    assert tuple(result) == _P_CHANNELS
    assert all(
        result.provenance_class[species] == "external_datapack"
        for species in _P_CHANNELS
    )


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
    reason="set OPENIMCC_TEST_LEGACY_VAPOROCK_ROOT to a local VapoRock checkout",
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
        assert "source-tabulated supercooled liquid branch" in (
            result.domain_flags[species] or ""
        )
        assert "2603 K fusion marker" in result.domain_flags[species]
        assert result.provenance_class[species] == "janaf_fitted"


def test_cr_atomic_row_covers_validated_3000_k_endpoint(
    gas_pack: ImccGasDatapack,
) -> None:
    result = evaluate_gas(
        {"Cr2O3": 1.0e-3},
        2950.0,
        1.0e-10,
        gas_pack,
        gas_species=("Cr",),
        allow_extrapolation=False,
    )
    assert math.isfinite(result["Cr"]) and result["Cr"] > 0.0
    assert result.domain_flags["Cr"] is None


def test_cr_liquid_parent_uses_labelled_extension_below_old_start(
    gas_pack: ImccGasDatapack,
) -> None:
    result = evaluate_gas(
        {"Cr2O3": 1.0e-3},
        1500.0,
        1.0e-10,
        gas_pack,
        gas_species=_CR_CHANNELS,
        allow_extrapolation=False,
    )
    assert all(
        math.isfinite(result[name]) and result[name] > 0.0 for name in _CR_CHANNELS
    )
    assert all(
        "Cr2O3(l) uses a labelled constant-Cp supercooled-liquid continuation from JANAF Cr-015"
        in result.domain_flags[name]
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
            for name in species:
                marker = 2340.0 if name in {"V", "VO", "VO2"} else 2175.0
                if temperature < marker:
                    assert "source-tabulated supercooled liquid branch" in (
                        result.domain_flags[name] or ""
                    )
                    assert f"{marker:g} K fusion marker" in result.domain_flags[name]
                else:
                    assert result.domain_flags[name] is None


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


def test_tantalum_channels_and_oxygen_balance(
    gas_pack: ImccGasDatapack, unit_activities: dict[str, float]
) -> None:
    activities = {"Ta2O5": 1.0e-3}
    result = evaluate_gas(
        activities,
        2000.0,
        1.0e-10,
        gas_pack,
        parent_oxides=("Ta2O5",),
        gas_species=("Ta", "TaO", "TaO2"),
        allow_extrapolation=False,
    )
    assert result._values == pytest.approx(
        {
            "Ta": 1.9379680170131573e-18,
            "TaO": 8.585414315417834e-11,
            "TaO2": 8.51923862658225e-09,
        },
        rel=1.0e-12,
    )
    default = evaluate_gas(unit_activities, 1800.0, 1.0e-10, gas_pack)
    assert not {"Ta", "TaO", "TaO2"}.intersection(default)

    root, balanced, diagnostics = evaluate_gas_oxygen_balance(
        activities, 2000.0, gas_pack, parent_oxides=("Ta2O5",)
    )
    assert root == pytest.approx(1.0907440752895347e-11, rel=1.0e-12)
    assert diagnostics["residual"] < 1.0e-12
    assert {"Ta", "TaO", "TaO2", "O2"}.issubset(balanced)

    unlisted_root, _, _ = evaluate_gas_oxygen_balance(
        {**activities, "SiO2": 1.0}, 2000.0, gas_pack,
        parent_oxides=("SiO2",),
    )
    si_only_root, _, _ = evaluate_gas_oxygen_balance(
        {"SiO2": 1.0}, 2000.0, gas_pack, parent_oxides=("SiO2",)
    )
    assert unlisted_root == pytest.approx(si_only_root, rel=0.0, abs=0.0)


def test_beryllium_channels_and_oxygen_balance(
    gas_pack: ImccGasDatapack, unit_activities: dict[str, float]
) -> None:
    activities = {"BeO": 1.0e-3}
    species = ("Be", "BeO", "Be2O", "(BeO)2", "(BeO)3", "(BeO)4", "Be2")
    result = evaluate_gas(
        activities, 2000.0, 1.0e-10, gas_pack,
        parent_oxides=("BeO",), gas_species=species,
        allow_extrapolation=False,
    )
    assert result._values == pytest.approx(
        {
            "Be": 5.4841642522907187e-11,
            "BeO": 2.839950604885199e-13,
            "Be2O": 3.603203485202284e-16,
            "(BeO)2": 2.528233210150918e-16,
            "(BeO)3": 2.497155230143581e-17,
            "(BeO)4": 6.464750106108906e-20,
            "Be2": 1.699397487178629e-25,
        },
        rel=1.0e-12,
    )
    default = evaluate_gas(unit_activities, 1800.0, 1.0e-10, gas_pack)
    assert not set(species).intersection(default)

    root, balanced, diagnostics = evaluate_gas_oxygen_balance(
        activities, 2000.0, gas_pack, parent_oxides=("BeO",)
    )
    assert root == pytest.approx(1.0897538394182622e-12, rel=1.0e-12)
    assert diagnostics["residual"] < 1.0e-12
    assert {*species, "O2"}.issubset(balanced)

    unlisted_root, _, _ = evaluate_gas_oxygen_balance(
        {**activities, "SiO2": 1.0}, 2000.0, gas_pack,
        parent_oxides=("SiO2",),
    )
    si_only_root, _, _ = evaluate_gas_oxygen_balance(
        {"SiO2": 1.0}, 2000.0, gas_pack, parent_oxides=("SiO2",)
    )
    assert unlisted_root == pytest.approx(si_only_root, rel=0.0, abs=0.0)


def test_thallium_channels_and_oxygen_balance(
    gas_pack: ImccGasDatapack, unit_activities: dict[str, float]
) -> None:
    activities = {"Tl2O": 1.0e-3}
    result = evaluate_gas(
        activities,
        2000.0,
        1.0e-10,
        gas_pack,
        parent_oxides=("Tl2O",),
        gas_species=("Tl", "Tl2O"),
        allow_extrapolation=False,
    )
    assert result._values == pytest.approx(
        {"Tl": 8.624314565304715, "Tl2O": 0.023561701919532385},
        rel=1.0e-12,
    )
    default = evaluate_gas(unit_activities, 1800.0, 1.0e-10, gas_pack)
    assert not {"Tl", "Tl2O"}.intersection(default)

    root, balanced, diagnostics = evaluate_gas_oxygen_balance(
        activities, 2000.0, gas_pack, parent_oxides=("Tl2O",)
    )
    assert root == pytest.approx(0.008771115323502448, rel=1.0e-12)
    assert diagnostics["residual"] < 1.0e-12
    assert {"Tl", "Tl2O", "O2"}.issubset(balanced)

    unlisted_root, _, _ = evaluate_gas_oxygen_balance(
        {**activities, "SiO2": 1.0}, 2000.0, gas_pack,
        parent_oxides=("SiO2",),
    )
    si_only_root, _, _ = evaluate_gas_oxygen_balance(
        {"SiO2": 1.0}, 2000.0, gas_pack, parent_oxides=("SiO2",)
    )
    assert unlisted_root == pytest.approx(si_only_root, rel=0.0, abs=0.0)


def _without_trace_rows(pack: ImccGasDatapack) -> ImccGasDatapack:
    trace_gas = {
        f"{species}(g)" for species in build_gas_tables.TRACE_GAS_SPECIES
    } | {
        source[0] for source in build_gas_tables.TRACE_ION_GAS_TEXT_SOURCES
    } | {
        source[0] for source in build_gas_tables.TRACE_ION_GAS_NASA_SOURCES
    }
    trace_gas |= {
        "Li-(g)", "LiO-(g)", "Rb-(g)", "Pb-(g)", "Mn(g)",
        "Ni(g)", "Co(g)", "Ni-(g)", "Co-(g)",
    }
    trace_gas |= {
        "Li-(g)", "LiO-(g)", "Rb-(g)", "Pb-(g)",
        *(source[0] for source in build_gas_tables.TRACE_ION_NASA_SOURCES),
        *(source[0] for source in build_gas_tables.TRACE_ION_PUBLISHED_SOURCES),
        *(source[0] for source in build_gas_tables.TRACE_ION_GAS_YAML_SOURCES),
    }
    trace_parents = {
        "Li2O(l)", "Rb2O(l)", "PbO(l)", "Cs2O(l)", "Cu2O(l)", "SnO(l)",
        "B2O3(l)", "Ga2O3(l)", "GeO2(l)", "In2O3(l)",
        "ZnO(l)", "CdO(l)", "Bi2O3(l)",
        "TeO2(l)",
        "SeO2(l)",
        "As2O3(l)",
        "Sb2O3(l)",
        "Ag2O(l)",
        "MnO(l)",
        "MoO3(l)", "WO3(l)", "BaO(l)", "SrO(l)", "ZrO2(l)",
        "Sc2O3(l)", "UO2(l)",
        "HfO2(l)", "C(cr)",
    }
    return replace(
        pack,
        gas_df=pack.gas_df.loc[~pack.gas_df.index.isin(trace_gas)],
        oxide_df=pack.oxide_df.loc[~pack.oxide_df.index.isin(trace_parents)],
    )


def test_default_trace_channels_are_optional_and_report_missing_rows(
    gas_pack: ImccGasDatapack,
) -> None:
    import openimcc.gas as gas_module

    halogen_gases = build_gas_tables.HALOGEN_GAS_SPECIES
    trace_gases = set(build_gas_tables.TRACE_GAS_SPECIES) | {
        "Mn", "Ni", "Co", "NiO", "CoO"
    } | halogen_gases
    parents = {
        "MnO": 1.0e-3,
        "CaO": 1.0e-3,
        "MgO": 1.0e-3,
        "Na2O": 1.0e-3,
        "K2O": 1.0e-3,
        "Li2O": 1.0e-3, "Rb2O": 1.0e-3, "PbO": 1.0e-3,
        "Ga2O3": 1.0e-3, "GeO2": 1.0e-3, "B2O3": 1.0e-3,
        "In2O3": 1.0e-3,
        "Cs2O": 1.0e-3,
        "Cu2O": 1.0e-3,
        "SnO": 1.0e-3,
        "ZnO": 1.0e-3,
        "Tl2O": 1.0e-3,
        "Ta2O5": 1.0e-3,
        "BeO": 1.0e-3,
        "CdO": 1.0e-3,
        "Bi2O3": 1.0e-3,
        "TeO2": 1.0e-3,
        "SeO2": 1.0e-3,
        "As2O3": 1.0e-3,
        "Sb2O3": 1.0e-3,
        "Ag2O": 1.0e-3,
        "MoO3": 1.0e-3, "WO3": 1.0e-3,
        "BaO": 1.0e-3, "SrO": 1.0e-3, "ZrO2": 1.0e-3,
        "NiO": 1.0e-3, "CoO": 1.0e-3,
        "Sc2O3": 1.0e-3, "UO2": 1.0e-3, "ThO2": 1.0e-3,
        "HfO2": 1.0e-3, "Y2O3": 1.0e-3,
        "H2": 1.0e-3, "C": 1.0e-3, "N2": 1.0e-3,
        "Br2": 1.0e-3,
        "I2": 1.0e-3,
        "Cl2": 1.0e-3,
        "F2": 1.0e-3,
    }
    default = evaluate_gas({}, 2000.0, 1.0e-8, gas_pack)
    assert not trace_gases & set(default)
    assert not trace_gases & set(default.omitted_channels)
    assert default.omitted_channels == {}

    activated = evaluate_gas(
        parents,
        2000.0,
        1.0e-8,
        gas_pack,
        parent_oxides=(
            *IMCC_PARENT_OXIDES, "H2", "C", "N2",
            "BaO", "SrO", "Li2O", "Rb2O", "Cs2O",
            "F2", "Cl2", "Br2", "I2",
        ),
        gas_species=tuple(trace_gases),
        allow_extrapolation=False,
    )
    assert trace_gases <= set(activated)
    assert all(
        math.isfinite(activated[species]) and activated[species] > 0.0
        for species in trace_gases
    )
    assert not trace_gases & set(activated.omitted_channels)

    missing_rows = evaluate_gas(
        parents,
        2000.0,
        1.0e-8,
        _without_trace_rows(gas_pack),
        parent_oxides=(
            *IMCC_PARENT_OXIDES, "H2", "C", "N2",
            "BaO", "SrO", "Li2O", "Rb2O", "Cs2O",
            "F2", "Cl2", "Br2", "I2",
        ),
        allow_extrapolation=True,
    )
    assert not trace_gases & set(missing_rows)
    assert set(missing_rows.omitted_channels) == trace_gases
    for species in trace_gases:
        assert f"{species}(g)" in missing_rows.omitted_channels[species]
        if species not in {"Ni", "Co", "NiO", "CoO"}:
            reason = missing_rows.omitted_channels[species]
            if species in build_gas_tables.HALOGEN_GAS_SPECIES:
                parent_gas = gas_module._GAS_PHASE_PARENT_ROWS.get(
                    gas_module._SF04_REACTIONS[species][0]
                )
                assert parent_gas is None or parent_gas in reason
                continue
            parent = _SF04_REACTIONS[species][0]
            parent_row = (
                gas_module._GAS_PHASE_PARENT_ROWS.get(parent)
                or gas_module._CONDENSED_PARENT_ROWS.get(parent)
                or f"{parent}(l)"
            )
            assert (
                parent_row in reason
                or species in {
                    "Sc", "ScO", "ScO2", "U", "UO", "UO2", "UO3",
                    "Th", "ThO", "ThO2", "Hf", "Y", "YO",
                    "Ta", "TaO", "TaO2", "Be", "BeO", "Be2O",
                    "(BeO)2", "(BeO)3", "(BeO)4", "Be2", "Tl", "Tl2O",
                }
                and f"missing from active table: {species}(g)" in reason
            )

    trace_species = tuple(sorted(trace_gases))
    trace_ions = evaluate_gas(
        parents,
        2000.0,
        1.0e-8,
        gas_pack,
        parent_oxides=tuple(parents),
        gas_species=trace_species,
        include_ions=True,
        allow_extrapolation=False,
    )
    expected_ions = {
        "Li+", "Rb+", "Pb+", "Li-", "LiO-", "Rb-", "Pb-",
        "Ga+", "Ge+", "B+", "Ga-", "B-", "BO-", "BO2-",
        "Zn+", "Zn-", "Cd+", "Bi+",
        "Mo+", "Mo-", "W+", "W-",
        "Ag+",
        "Ag-",
        "Ba+", "Sr+", "Zr+", "Zr-", "Ni+", "Ni-", "Co+", "Co-",
        "Sc+",
    }
    assert expected_ions <= set(trace_ions)
    assert "In+" not in trace_ions
    assert "Sc+" in trace_ions
    assert "U+" not in trace_ions
    nasa_ions = {"Ge+", "BO-", "Cd+", "Ag+", "Ag-", "Sc+"}
    for species in expected_ions:
        assert trace_ions[species] > 0.0
        assert trace_ions.provenance_class[species] == (
            "nist_partition_fitted_ionisation"
            if species == "Bi+"
            else "nasa_glenn_fitted_ionisation"
            if species in nasa_ions
            else "janaf_fitted_ionisation"
        )
    for species in (
        "Ga", "Ge", "B", "In", "Mo", "MoO3", "(MoO3)5",
        "W", "W3O8", "(WO3)5",
        "Ba", "Sr", "Zr", "Ni", "Co",
    ):
        with pytest.raises(ImccGasSpeciesNotFoundError):
            evaluate_gas(
                {}, 2000.0, 1.0e-8, gas_pack,
                gas_species=(species,), allow_extrapolation=False,
            )

    for element, parent in (("Sc", "Sc2O3"), ("U", "UO2")):
        with pytest.raises(ImccGasSpeciesNotFoundError) as refusal:
            evaluate_gas(
                _quickstart_activities(2000.0, allow_extrapolation=True),
                2000.0,
                1.0e-8,
                gas_pack,
                gas_species=(element,),
                allow_extrapolation=True,
            )
        assert refusal.value.code == "imcc_gas_species_not_found"
        assert parent in str(refusal.value)
    expected_trace_ions = {
        "Li+", "Rb+", "Pb+", "Li-", "LiO-", "Rb-", "Pb-",
        "Cs+", "Sn+", "Cs2O+",
        "Zn+", "Zn-", "Cd+", "Bi+",
        "Mo+", "Mo-", "W+", "W-",
    }
    assert expected_trace_ions <= set(trace_ions)
    for species in expected_trace_ions - {"Sn+", "Cs2O+", "Cd+", "Bi+"}:
        assert trace_ions[species] > 0.0
        assert trace_ions.provenance_class[species] == "janaf_fitted_ionisation"
    for species in ("Sn+", "Cs2O+"):
        assert trace_ions[species] > 0.0
        assert trace_ions.provenance_class[species] == "nasa_glenn_fitted"
    assert trace_ions["Cd+"] > 0.0
    assert trace_ions.provenance_class["Cd+"] == (
        "nasa_glenn_fitted_ionisation"
    )
    assert trace_ions["Bi+"] > 0.0
    assert trace_ions.provenance_class["Bi+"] == "nist_partition_fitted_ionisation"


def test_default_outputs_are_unchanged_when_trace_rows_are_removed(
    gas_pack: ImccGasDatapack,
) -> None:
    baseline_pack = _without_trace_rows(gas_pack)
    activities = {
        "SiO2": 0.45,
        "MgO": 0.15,
        "FeO": 0.08,
        "CaO": 0.12,
        "Al2O3": 0.12,
        "TiO2": 0.02,
        "Na2O": 0.04,
        "K2O": 0.02,
    }
    for temperature in (1200.0, 1499.999, 1500.0, 2000.0, 3000.0):
        for fugacity in (1.0e-12, 1.0e-6, 1.0e-4):
            current = evaluate_gas(
                activities,
                temperature,
                fugacity,
                gas_pack,
                allow_extrapolation=True,
            )
            baseline = evaluate_gas(
                activities,
                temperature,
                fugacity,
                baseline_pack,
                allow_extrapolation=True,
            )
            assert dict(current.items()) == dict(baseline.items())
            assert dict(current.domain_flags) == dict(baseline.domain_flags)
            assert dict(current.provenance_class) == dict(baseline.provenance_class)
            assert dict(current.omitted_channels) == dict(baseline.omitted_channels)


def test_existing_coefficient_rows_match_the_base_pack_exactly(
    gas_pack: ImccGasDatapack,
) -> None:
    """Keep every serialized row from the pre-campaign gas pack unchanged."""
    base = "af0f2a2"
    root = Path(__file__).resolve().parents[1]
    for relpath, current_path in (
        ("src/openimcc/data/gas/gas-shomate.csv", gas_pack.gas_path),
        ("src/openimcc/data/gas/condensate.csv", gas_pack.oxide_path),
    ):
        original = subprocess.run(
            ["git", "show", f"{base}:{relpath}"],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
        ).stdout
        old_lines = original.splitlines()
        current_lines = current_path.read_text(encoding="utf-8").splitlines()
        assert current_lines[0] == old_lines[0]
        assert Counter(old_lines[1:]) <= Counter(current_lines[1:])


def _trace_janaf_apparent_gibbs(table_id: str, temperature: float) -> float:
    source_directory = (
        Path(__file__).resolve().parents[1]
        / "data-src"
        / "janaf"
    )
    if (source_directory / f"{table_id}.yaml").is_file():
        return _janaf_apparent_gibbs(table_id, temperature)
    rows = build_gas_tables._janaf_text_source_rows(source_directory, table_id)
    reference = rows[298.15]["formation_enthalpy"]
    row = rows[temperature]
    return (
        (reference + row["enthalpy_increment"]) * 1000.0
        - temperature * row["entropy"]
    )


def _trace_source_apparent_gibbs(table_id: str, temperature: float) -> float:
    if table_id in {"TSIV-1334", "TSIV-2026"}:
        source_dir = Path(__file__).resolve().parents[1] / "data-src" / "published-thermo"
        record = build_gas_tables._load_record(
            source_dir / f"tsiv-table-{table_id.split('-')[1]}.json"
        )
        source = build_gas_tables._tsiv_liquid_source_rows(record)[temperature]
        return (
            float(record["formation_enthalpy_298_kj_mol"]["value"])
            + source["enthalpy_increment"]
        ) * 1000.0 - temperature * source["entropy"]
    if table_id in {"P672-SeO2-LFS", "P672-As2O3-LFS", "P672-Sb2O3-LFS", "P672-Ag2O-LFS", "P672-HfO2-LFS"}:
        repository = Path(__file__).resolve().parents[1]
        source_dir = repository / "data-src" / "published-thermo"
        config = json.loads(
            (repository / "data-src" / "liquid-from-solid-inputs.json").read_text(
                encoding="utf-8"
            )
        )["rows"][
            {"P672-SeO2-LFS": "SeO2", "P672-As2O3-LFS": "As2O3", "P672-Sb2O3-LFS": "Sb2O3", "P672-Ag2O-LFS": "Ag2O", "P672-HfO2-LFS": "HfO2"}[table_id]
        ]
        species = {"P672-SeO2-LFS": "SeO2", "P672-As2O3-LFS": "As2O3", "P672-Sb2O3-LFS": "Sb2O3", "P672-Ag2O-LFS": "Ag2O", "P672-HfO2-LFS": "HfO2"}[table_id]
        _rows, construction, _alternate = build_gas_tables._fit_liquid_from_solid_parent(
            source_dir, species, config
        )
        return construction.at(temperature).gibbs_kj_mol * 1000.0
    if not table_id.startswith("NG-"):
        source_dir = Path(__file__).resolve().parents[1] / "data-src" / "published-thermo"
        if (
            table_id.startswith(("P672-", "P696-"))
            or table_id in {"TSIV-1335", "TSIV-1436"}
        ) and table_id not in {
            "P672-T102", "P672-SeO2-LFS", "P672-As2O3-LFS",
            "P672-Sb2O3-LFS", "P672-Ag2O-LFS", "P672-ThO2-LFS", "P672-HfO2-LFS",
        }:
            record = build_gas_tables._published_table_record(source_dir, table_id)
            if record.get("phase") == "liquid":
                nodes = {
                    float(T): {
                        "temperature": float(T),
                        "heat_capacity": float(cp) * 4.184,
                        "entropy": float(entropy) * 4.184,
                        "enthalpy_increment": float(h) * 4.184,
                    }
                    for T, cp, entropy, h in record["nodes"]
                }
                nodes = build_gas_tables._constant_cp_rows_through(nodes, temperature)
                reference = float(record["formation_enthalpy_298_kcal_mol"]) * 4.184
                node = nodes[temperature]
                return (reference + node["enthalpy_increment"]) * 1000.0 - (
                    temperature * node["entropy"]
                )
            nodes, reference, _limit = build_gas_tables._published_gas_source_rows(
                source_dir, table_id
            )
            nodes = build_gas_tables._constant_cp_rows_through(nodes, temperature)
            node = nodes[temperature]
            return (reference + node["enthalpy_increment"]) * 1000.0 - (
                temperature * node["entropy"]
            )
        if table_id in {"TSIV-1603", "P672-T102", "P672-SeO2-LFS", "P672-As2O3-LFS", "P672-Sb2O3-LFS", "P672-Ag2O-LFS", "P672-ThO2-LFS", "P672-HfO2-LFS"}:
            repository = Path(__file__).resolve().parents[1]
            species = {
                "TSIV-1603": "ZnO",
                "P672-T102": "CdO",
                "P672-SeO2-LFS": "SeO2",
                "P672-As2O3-LFS": "As2O3",
                "P672-Sb2O3-LFS": "Sb2O3",
                "P672-Ag2O-LFS": "Ag2O",
                "P672-ThO2-LFS": "ThO2",
                "P672-HfO2-LFS": "HfO2",
            }[table_id]
            config = json.loads(
                (repository / "data-src" / "liquid-from-solid-inputs.json").read_text(
                    encoding="utf-8"
                )
            )["rows"][species]
            _rows, construction, _alternate = (
                build_gas_tables._fit_liquid_from_solid_parent(
                    source_dir, species, config
                )
            )
            return construction.at(temperature).gibbs_kj_mol * 1000.0
        return _trace_janaf_apparent_gibbs(table_id, temperature)
    record = build_gas_tables._load_record(
        Path(__file__).resolve().parents[1]
        / "data-src"
        / "nasa-glenn"
        / f"{table_id}.json"
    )
    properties = build_gas_tables._nasa7_properties(record, temperature)
    # NASA-9 H(T)/(R*T), including b1, is already on the assigned elemental
    # reference, so evaluate G directly as H_card(T)-T*S_card(T).
    enthalpy = R_J_MOL_K * temperature * properties["h_rt"]
    entropy = R_J_MOL_K * properties["s_R"]
    return enthalpy - temperature * entropy


def test_trace_channels_match_hand_calculated_mass_action_at_2000_k(
    gas_pack: ImccGasDatapack,
) -> None:
    """Use source H/S cells and reaction stoichiometry, never fitted rows."""
    temperature = 2000.0
    cases = (
        ("Li", "Li2O", "Li-005", "Li-015"),
        ("Rb", "Rb2O", "Rb-005", "NG-1841"),
        ("Pb", "PbO", "Pb-005", "O-007"),
        ("Cs", "Cs2O", "Cs-005", "NG-1842"),
        ("Cu", "Cu2O", "Cu-005", "Cu-020"),
        ("Sn", "SnO", "NG-1845", "NG-1848"),
        ("Ga", "Ga2O3", "Ga-005", "NG-12403"),
        ("Ge", "GeO2", "NG-5186", "NG-12434"),
        ("B", "B2O3", "B-005", "B-096"),
        ("In", "In2O3", "NG-6005", "NG-12662"),
        ("Zn", "ZnO", "Zn-005", "TSIV-1603"),
        ("Cd", "CdO", "NG-4060", "P672-T102"),
        ("Bi", "Bi2O3", "P672-T80", "P672-T82"),
        ("Te", "TeO2", "P672-Te-g", "P672-TeO2-l"),
        ("Se", "SeO2", "P672-Se-g", "P672-SeO2-LFS"),
        ("SeO", "SeO2", "P672-SeO-g", "P672-SeO2-LFS"),
        ("As", "As2O3", "P672-As-g", "P672-As2O3-LFS"),
        ("Sb", "Sb2O3", "P672-Sb-g", "P672-Sb2O3-LFS"),
        ("Ag", "Ag2O", "NG-0074", "P672-Ag2O-LFS"),
        ("MoO2", "MoO3", "Mo-010", "Mo-015"),
        ("WO2", "WO3", "O-048", "O-066"),
        ("Ba", "BaO", "Ba-005", "Ba-035"),
        ("BaO", "BaO", "Ba-037", "Ba-035"),
        ("Sr", "SrO", "Sr-006", "O-014"),
        ("SrO", "SrO", "O-016", "O-014"),
        ("Zr", "ZrO2", "Zr-006", "O-053"),
        ("ZrO", "ZrO2", "O-028", "O-053"),
        ("ZrO2", "ZrO2", "O-055", "O-053"),
        ("Ni", "NiO", "Ni-005", "TSIV-2026"),
        ("NiO", "NiO", "NG-0926", "TSIV-2026"),
        ("Co", "CoO", "Co-005", "TSIV-1334"),
        ("CoO", "CoO", "TSIV-1335", "TSIV-1334"),
        ("Sc", "Sc2O3", "NG-1087", "NG-1857"),
        ("ScO", "Sc2O3", "NG-1090", "NG-1857"),
        ("ScO2", "Sc2O3", "NG-1092", "NG-1857"),
        ("U", "UO2", "NG-1206", "NG-1977"),
        ("UO", "UO2", "NG-1224", "NG-1977"),
        ("UO2", "UO2", "NG-1230", "NG-1977"),
        ("UO3", "UO2", "NG-1235", "NG-1977"),
        ("Th", "ThO2", "P672-Th-g", "P672-ThO2-LFS"),
        ("ThO", "ThO2", "P672-ThO-g", "P672-ThO2-LFS"),
        ("ThO2", "ThO2", "P672-ThO2-g", "P672-ThO2-LFS"),
        ("Hf", "HfO2", "Hf-006", "P672-HfO2-LFS"),
        ("Ta", "Ta2O5", "Ta-005", "O-078", 2000.0),
        ("Be", "BeO", "NG-0185", "NG-BeO-L-1352", 2900.0),
        ("Tl", "Tl2O", "P672-Tl-g", "P672-Tl2O-l", 1200.0),
        ("Tl2O", "Tl2O", "P672-Tl2O-g", "P672-Tl2O-l", 1200.0),
    )
    for case in cases:
        gas_species, parent, gas_table, parent_table = case[:4]
        temperature = case[4] if len(case) == 5 else 2000.0
        reaction_parent, n_gas, n_o2 = _SF04_REACTIONS[gas_species]
        assert reaction_parent == parent
        gas_g = _trace_source_apparent_gibbs(gas_table, temperature)
        parent_g = _trace_source_apparent_gibbs(parent_table, temperature)
        oxygen_g = _janaf_apparent_gibbs("O-029", temperature)
        # Premise: oxide(l) = n_gas*E(g) + n_O2*O2(g). Algebra:
        # log10(p_E/bar) = -DeltaG°/(n_gas*R*T*ln(10)) at unit parent
        # activity and fO2=1. Unit check: every Gibbs term and R*T is J/mol.
        # NASA liquid parents retain their published pure-liquid 1-atm state;
        # the Cu2O NASA tail is anchored to its 2000 K JANAF node.
        delta_g = n_gas * gas_g + n_o2 * oxygen_g - parent_g
        expected_log10 = -delta_g / (
            n_gas * R_J_MOL_K * temperature * math.log(10.0)
        )
        result = evaluate_gas(
            {parent: 1.0},
            temperature,
            1.0,
            gas_pack,
            parent_oxides=(parent,),
            gas_species=(gas_species,),
            allow_extrapolation=False,
        )
        assert math.log10(result[gas_species]) == pytest.approx(
            expected_log10, abs=0.001
        )
        if gas_species == "Zn":
            flag = result.domain_flags[gas_species]
            assert "central liquid uses the printed 2250 K node" in flag
            assert "quadratic crystal-Cp extension is its uncertainty bound" in flag


def test_pankratz_298_k_anchors_reproduce_independent_reaction_table_cells() -> None:
    """Check anchor reference conversions against printed reaction cells.

    The expected values are the independently printed 298.15 K ΔH° cells in
    Bulletin 672, not values derived from the gas fit. Energies are kcal/mol.
    """
    source_path = (
        Path(__file__).resolve().parents[1]
        / "data-src"
        / "published-thermo"
        / "pankratz-bulletin-672.json"
    )
    tables = json.loads(source_path.read_text())["tables"]
    hf_source_path = (
        Path(__file__).resolve().parents[1]
        / "data-src"
        / "published-thermo"
        / "pankratz-hf.json"
    )
    tables.update(json.loads(hf_source_path.read_text())["tables"])
    for table_id, record in tables.items():
        if not table_id.startswith("P672-"):
            continue
        note = record["source_note"].casefold()
        assert note.strip()
        assert "formation_enthalpy_298_kcal_mol" in record

    def anchor(table_id: str) -> float:
        return tables[table_id]["formation_enthalpy_298_kcal_mol"]

    # 0.5 E2(g) = E(g), as separately printed for Te (p. 415) and Se
    # (p. 367). The equality catches either atom anchor left on the E2 basis.
    for atom_id, dimer_id, printed_delta_h in (
        ("P672-Te-g", "P672-Te2-g", 31.435),
        ("P672-Se-g", "P672-Se2-g", 39.600),
    ):
        assert anchor(atom_id) - 0.5 * anchor(dimer_id) == pytest.approx(
            printed_delta_h, abs=0.0005
        )

    # 0.5 Te2(g) + O2(g) = TeO2(l), p. 419. O2 has zero formation
    # enthalpy at 298.15 K, so this independently checks the liquid anchor.
    assert anchor("P672-TeO2-l") - 0.5 * anchor("P672-Te2-g") == pytest.approx(
        -96.265, abs=0.0005
    )

    # Direct 298.15 K cells pin dimer, monoxide, crystal, and arsenic anchors
    # to their printed formation cells. Te(g) and Se(g) are derived anchors:
    # their source notes must preserve the reaction-cell plus half-dimer path.
    for table_id, printed_delta_h in (
        ("P672-Te-g", 50.600),
        ("P672-Te2-g", 38.330),
        ("P672-TeO2-l", -77.100),
        ("P672-Se-g", 56.250),
        ("P672-Se2-g", 33.300),
        ("P672-SeO2-c", -53.860),
        ("P672-As-g", 72.120),
        ("P672-As2-g", 52.820),
        ("P672-As4-g", 36.640),
        ("P672-As2O3-c", -157.020),
        ("P672-As4O6-g", -285.910),
        ("P672-Sb-g", 62.700),
        ("P672-Sb2-g", 56.300),
        ("P672-Sb4-g", 49.000),
        ("P672-Sb4O6-g", -290.520),
        ("P672-Sb2O3-c", -169.400),
        ("P672-Ag-g", 68.090),
        ("P672-Ag2O-c", -7.420),
        ("P672-Th-g", 142.730),
        ("P672-ThO-g", -6.077),
        ("P672-ThO2-c", -293.120),
        ("P672-ThO2-g", -118.670),
        ("P672-Hf-g", 148.000),
        ("P672-HfO2-c", -267.120),
    ):
        assert anchor(table_id) == pytest.approx(printed_delta_h, abs=0.0005)
    provenance = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1]
            / "src/openimcc/data/gas/PROVENANCE.yaml"
        ).read_text()
    )
    source_rows = {
        row["table_id"]: row
        for row in provenance["rows"]
        if row.get("table_id")
    }
    for table_id, printed_page in (
        ("P672-Th-g", 423), ("P672-ThO-g", 424),
        ("P672-ThO2-c", 425), ("P672-ThO2-g", 426),
        ("P672-Hf-g", 187), ("P672-HfO2-c", 188),
    ):
        if table_id == "P672-Hf-g":
            locator = tables[table_id]["source_note"].casefold()
            assert f"printed p. {printed_page}" in locator
        else:
            assert source_rows[table_id]["source_locator"].casefold().startswith(
                f"printed p. {printed_page}"
            )

    # Independent printed reaction cells close against the elemental 298 K
    # formation anchors, with oxygen in its elemental standard state (zero).
    for product, reactant, printed_delta_h in (
        ("P672-ThO-g", "P672-Th-g", -148.807),
        ("P672-ThO2-g", "P672-Th-g", -261.400),
        ("P672-ThO2-c", "P672-Th-g", -435.850),
        ("P672-HfO2-c", "P672-Hf-g", -415.120),
    ):
        assert anchor(product) - anchor(reactant) == pytest.approx(
            printed_delta_h, abs=0.0005
        )
    for table_id, reaction_value, dimer_value in (
        ("P672-Te-g", "31.435", "38.330"),
        ("P672-Se-g", "39.600", "33.300"),
    ):
        note = tables[table_id]["source_note"]
        assert "derived from the printed" in note
        assert f"reaction value, {reaction_value} kcal/mol" in note
        assert f"elemental formation value, {dimer_value} kcal/mol" in note
        assert "not a printed formation-table cell" in note

    # SeO's independently printed 0.5 Se2 + 0.5 O2 reaction value pins the
    # formation anchors to the p. 377 reaction table.
    assert anchor("P672-SeO-g") - 0.5 * anchor("P672-Se2-g") == pytest.approx(
        -3.900, abs=0.0005
    )

    # Independent 1500 K reaction-table ΔG° cells (Bulletin 672 pp. 421 and
    # 381) check the complete elemental reference and the printed oxide rows.
    # Compute G = ΔfH298 + (H-H298) - T*S from the vendored source nodes;
    # subtract 0.5 E2(g) and O2(g), then compare to the separately printed
    # reaction ΔG°. Tolerance covers three-decimal source rounding (0.01 kcal).
    for oxide_id, dimer_id, printed_delta_g_kcal in (
        ("P672-TeO2-g", "P672-Te2-g", -12.826),
        ("P672-SeO2-g", "P672-Se2-g", -19.101),
    ):
        oxide_g = _trace_source_apparent_gibbs(oxide_id, 1500.0)
        dimer_g = _trace_source_apparent_gibbs(dimer_id, 1500.0)
        oxygen_g = _trace_source_apparent_gibbs("O-029", 1500.0)
        # Bulletin 672 reaction tables use the 1-atm gas standard; JANAF's
        # O2 row uses 1 bar. Convert the oxygen chemical potential by RT ln
        # (1.01325) before comparing the independently printed ΔG° cells.
        oxygen_g += R_J_MOL_K * 1500.0 * math.log(101325.0 / 100000.0)
        computed_delta_g_kcal = (
            oxide_g - 0.5 * dimer_g - oxygen_g
        ) / 4184.0
        assert computed_delta_g_kcal == pytest.approx(
            printed_delta_g_kcal, abs=0.01
        )

    # For As, the balanced exchange 2 As2O3(cr) = As4O6(g) cancels both
    # elemental reference contributions. Its independent log K follows from
    # the printed formation-table log Kf cells: 191.361 - 2*100.913. The
    # apparent-G difference from stored 298.15 K enthalpy and entropy rows
    # must reproduce that value. kcal*4184 gives J, cal*4.184 gives J/K;
    # a 0.002 dex tolerance covers the printed three-decimal cells.
    as4o6 = tables["P672-As4O6-g"]
    as2o3 = tables["P672-As2O3-c"]
    as4o6_node = next(row for row in as4o6["nodes"] if row[0] == 298.15)
    as2o3_node = next(row for row in as2o3["nodes"] if row[0] == 298.15)
    as_reaction_g = (
        anchor("P672-As4O6-g") * 4184.0
        - 298.15 * as4o6_node[2] * 4.184
        - 2 * (
            anchor("P672-As2O3-c") * 4184.0
            - 298.15 * as2o3_node[2] * 4.184
        )
    )
    as_reaction_log10_k = -as_reaction_g / (
        R_J_MOL_K * 298.15 * math.log(10.0)
    )
    printed_as_log10_k = 191.361 - 2 * 100.913
    assert as_reaction_log10_k == pytest.approx(printed_as_log10_k, abs=0.002)


def test_sb4_reaction_closes_to_printed_pankratz_cells(
    gas_pack: ImccGasDatapack,
) -> None:
    """Stored Sb gas rows reproduce the separate 2 Sb2(g)=Sb4(g) table."""
    tables = json.loads(
        (Path(__file__).resolve().parents[1] / "data-src/published-thermo/pankratz-bulletin-672.json").read_text()
    )["tables"]
    assert tables["P672-Sb4-g"]["formation_enthalpy_298_kcal_mol"] - 2 * tables["P672-Sb2-g"]["formation_enthalpy_298_kcal_mol"] == pytest.approx(-63.600, abs=0.0005)
    # Printed p. 357 log K cells. The source rows use the elemental 298 K
    # anchors, with kcal-to-J and cal-to-J/K conversion in the fitted G values.
    for temperature, printed_log10_k in ((1200.0, 3.521), (1500.0, 1.280), (2000.0, -0.934)):
        sb2 = _nearest_interval_row(gas_pack.gas_df, "Sb2(g)", temperature)
        sb4 = _nearest_interval_row(gas_pack.gas_df, "Sb4(g)", temperature)
        delta_g = _janaf_gibbs(temperature, sb4) - 2 * _janaf_gibbs(temperature, sb2)
        log10_k = -delta_g / (R_J_MOL_K * temperature * math.log(10.0))
        assert log10_k == pytest.approx(printed_log10_k, abs=0.002)


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


@pytest.mark.parametrize("temperature", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_temperature_refuses_before_table_access(
    temperature: float,
    gas_pack: ImccGasDatapack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _deny_gas_table_reads(monkeypatch, gas_pack)
    with pytest.raises(ValueError, match="finite and positive"):
        evaluate_gas(
            {"Na2O": 1.0},
            temperature,
            1.0,
            gas_pack,
            gas_species=("Na",),
        )


@pytest.mark.parametrize("fugacity", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_fugacity_refuses_before_table_access(
    fugacity: float,
    gas_pack: ImccGasDatapack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _deny_gas_table_reads(monkeypatch, gas_pack)
    with pytest.raises(
        ImccGasInvalidFugacityError,
        match=r"finite and positive p_O2/p°.*not log10 fO2",
    ):
        evaluate_gas(
            {"Na2O": 1.0},
            2000.0,
            fugacity,
            gas_pack,
            gas_species=("Na",),
        )


@pytest.mark.parametrize("activity", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_activity_refuses(
    activity: float,
    gas_pack: ImccGasDatapack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _deny_gas_table_reads(monkeypatch, gas_pack)
    with pytest.raises(ValueError, match="finite and >= 0"):
        evaluate_gas(
            {"Na2O": activity},
            2000.0,
            1.0,
            gas_pack,
            gas_species=("Na",),
        )


def _deny_gas_table_reads(
    monkeypatch: pytest.MonkeyPatch, gas_pack: ImccGasDatapack
) -> None:
    original_getitem = pd.DataFrame.__getitem__

    def guarded_getitem(frame: pd.DataFrame, key: object) -> object:
        if frame is gas_pack.gas_df or frame is gas_pack.oxide_df:
            raise AssertionError("gas-table access sentinel is active")
        return original_getitem(frame, key)

    monkeypatch.setattr(pd.DataFrame, "__getitem__", guarded_getitem)
    with pytest.raises(AssertionError, match="gas-table access sentinel is active"):
        gas_pack.gas_df["state"]


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
        if name in gas_module._ALKALI_HALIDE_REACTION_OXIDES:
            continue  # evaluate_gas_oxygen_balance supplies the side-oxide demand.
        expected = (
            n_o2
            if parent in gas_module._GAS_PHASE_PARENT_ROWS
            else -n_o2 / n_gas
            if parent
            else (1.0 if name == "O2" else 0.5)
        )
        assert metadata[name].pO2_exponent == pytest.approx(expected, abs=1e-12)
        if parent in gas_module._GAS_PHASE_PARENT_ROWS:
            assert metadata[name].parent_oxygen_demand == 0.0


def test_alkali_halides_use_half_oxygen_per_alkali_in_balance_metadata() -> None:
    import openimcc.gas as gas_module

    for species, (oxide, n_oxide, n_o2) in gas_module._ALKALI_HALIDE_REACTION_OXIDES.items():
        atoms = gas_module._formula_atoms(species)
        oxide_atoms = gas_module._formula_atoms(oxide)
        metal_atoms = sum(
            count for element, count in atoms.items()
            if element != "X" and element in oxide_atoms and element != "O"
        )
        oxide_metal_atoms = sum(
            count for element, count in oxide_atoms.items() if element != "O"
        )
        assert n_oxide == pytest.approx(metal_atoms / oxide_metal_atoms)
        assert n_o2 == pytest.approx(n_oxide / 2.0)


def test_oxygen_balance_molar_masses_sum_repeated_formula_tokens() -> None:
    import openimcc.gas as gas_module

    metadata = oxygen_balance_species_metadata(
        {species: None for species in gas_module._SF04_REACTIONS}
    )
    changed = set()
    for species, entry in metadata.items():
        prior_atoms = gas_module._formula_atoms(species)
        prior_mass = sum(
            gas_module._ATOMIC_MASS_G_MOL[element] * count
            for element, count in prior_atoms.items()
        )
        if entry.molar_mass != prior_mass:
            changed.add(species)
    assert changed == set()
    assert metadata["SSO"].molar_mass == pytest.approx(80.119)
    assert metadata["(MoO3)2"].molar_mass == pytest.approx(287.894)


def test_condensed_parent_phase_resolver_switches_at_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import openimcc.gas as gas_module

    monkeypatch.setitem(
        gas_module._CONDENSED_PARENT_PHASE_ROWS,
        "Au",
        (1337.0, "Au(cr)", "Au(l)"),
    )

    assert gas_module._parent_standard_row("Au", 1336.999) == "Au(cr)"
    assert gas_module._parent_standard_row("Au", 1337.0) == "Au(l)"
    assert gas_module._parent_standard_rows("Au") == ("Au(cr)", "Au(l)")
    assert gas_module._parent_standard_row("Pd", 1826.999) == "Pd(cr)"
    assert gas_module._parent_standard_row("Pd", 1827.0) == "Pd(l)"
    assert gas_module._parent_standard_row("Pt", 2041.999) == "Pt(cr)"
    assert gas_module._parent_standard_row("Pt", 2042.0) == "Pt(l)"
    assert gas_module._parent_standard_rows("Pt") == ("Pt(cr)", "Pt(l)")
    assert gas_module._parent_standard_row("Ru", 2606.999) == "Ru(cr)"
    assert gas_module._parent_standard_row("Ru", 2607.0) == "Ru(l)"
    assert gas_module._parent_standard_rows("Ru") == ("Ru(cr)", "Ru(l)")


def test_pt_oxide_gas_uses_unit_positive_oxygen_exponent_and_moves_root_down(
    gas_pack: ImccGasDatapack,
) -> None:
    import openimcc.gas as gas_module

    temperature = 1800.0
    activity = 1.0e14
    base_activities = _quickstart_activities(
        temperature, allow_extrapolation=True
    )
    root_without, _, _ = gas_module.evaluate_gas_oxygen_balance(
        base_activities, temperature, gas_pack,
    )
    root_with, pressures, _ = gas_module.evaluate_gas_oxygen_balance(
        {**base_activities, "Pt": activity}, temperature, gas_pack,
    )
    assert root_with < root_without
    assert pressures["Pt"] > 0.0
    assert pressures["PtO2"] > 0.0
    metadata = oxygen_balance_species_metadata({"PtO2": None})["PtO2"]
    assert metadata.parent_oxygen_demand == 0.0
    assert metadata.pO2_exponent == pytest.approx(1.0)

    lower_f = evaluate_gas(
        {"Pt": 1.0}, temperature, 1.0e-10, gas_pack, gas_species=("PtO2",)
    )["PtO2"]
    higher_f = evaluate_gas(
        {"Pt": 1.0}, temperature, 1.0e-9, gas_pack, gas_species=("PtO2",)
    )["PtO2"]
    assert higher_f / lower_f == pytest.approx(10.0, rel=1e-12)
    metal_at_low_f = evaluate_gas(
        {"Pt": 1.0}, temperature, 1.0e-10, gas_pack, gas_species=("Pt",)
    )["Pt"]
    metal_at_high_f = evaluate_gas(
        {"Pt": 1.0}, temperature, 1.0e-9, gas_pack, gas_species=("Pt",)
    )["Pt"]
    assert metal_at_high_f == metal_at_low_f


@pytest.mark.parametrize(
    ("element", "metal_mass"),
    [("Hg", 200.592), ("Au", 196.96657), ("Pd", 106.42), ("Ru", 101.07), ("Rh", 102.9055), ("Ir", 192.217)],
)
def test_metal_vapour_parent_activity_and_oxygen_root(
    gas_pack: ImccGasDatapack, element: str, metal_mass: float
) -> None:
    import openimcc.gas as gas_module

    temperature = 1800.0
    activity = 1.0e-3
    base_activities = _quickstart_activities(
        temperature, allow_extrapolation=True
    )
    root_without, _, _ = gas_module.evaluate_gas_oxygen_balance(
        base_activities, temperature, gas_pack,
    )
    activities = {**base_activities, element: activity}
    root_with, pressures, _ = gas_module.evaluate_gas_oxygen_balance(
        activities, temperature, gas_pack,
    )
    assert root_with == root_without
    assert pressures[element] > 0.0
    assert gas_module._ATOMIC_MASS_G_MOL[element] == pytest.approx(metal_mass)

    direct = evaluate_gas(
        {element: activity}, temperature, 1.0e-10, gas_pack,
        gas_species=(element,),
    )
    parent_row = gas_module._parent_standard_row(element, temperature)
    metal_g = species_thermo(element, "g", temperature, gas_pack).G_J_mol
    parent_g = species_thermo(
        parent_row.removesuffix("(cr)").removesuffix("(l)"),
        "cr" if parent_row.endswith("(cr)") else "l",
        temperature,
        gas_pack,
    ).G_J_mol
    expected = math.exp(-(metal_g - parent_g) / (R_J_MOL_K * temperature)) * activity
    assert direct[element] == pytest.approx(expected, rel=2e-12)
    assert evaluate_gas(
        {element: activity}, temperature, 1.0e-4, gas_pack,
        gas_species=(element,),
    )[element] == direct[element]

    omitted = evaluate_gas({}, temperature, 1.0e-10, gas_pack)
    assert element not in omitted
    with pytest.raises(ImccGasSpeciesNotFoundError, match="channel"):
        evaluate_gas(
            {}, temperature, 1.0e-10, gas_pack, gas_species=(element,)
        )


def test_hg_liquid_continuation_is_returned_above_one_bar(
    gas_pack: ImccGasDatapack,
) -> None:
    result = evaluate_gas(
        {"Hg": 1.0}, 1200.0, 1.0e-10, gas_pack, gas_species=("Hg",),
        allow_extrapolation=True,
    )
    assert result["Hg"] > 1.0
    assert "continuation" in result.domain_flags["Hg"]


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


# Numeric-output SHA256 pins for the JANAF Na-013 and Al2O3(l) rows: pO2,
# partial pressures, and diagnostics. The full result is pinned separately
# below, including every ImccGasResult dataclass field.
_OXYGEN_BALANCE_SF04_NUMERIC_HEX_SHA256 = (
    "490788de29fddf8e1b567b2ae802743cb70d352107c6094cac6e19e1dac62150",
    "5a11127d8a5707e294e007b97440b15d53eb04cbed16d46aa3800e2bb39a3a91",
    "a01c562e13e903c6d23ef33caef71a911375969c75e3ba0885ae4de21f00e2a1",
    "35dd36c3359c5bc3e7b319be23a225cc05ca9c783e8500eafd65d3abb2793a6b",
    "b609e28617362d126480d0fcaf68c37f6e12086d8305f2e7d8d281b59c36c32d",
    "6f6188435c673d9b566d56b69a4228e9a7ebc0df1fd79713b9699485c15c274f",
    "404a6d5d26660f1c6310385aa681af5b9f9b6dd82d0154e37bc523d826804984",
    "e958063bc882993cdaee7f6608e2d6819809b604bed02d38761f8e0f0bbaba95",
    "12a36b6ba8534904753f6f4c559c846c74ad80689a9aa202cecc1c94277a2f85",
    "58139d3f6b280f546b4c39dc64e80bc0352e024e62d8e7335dc593bf5e5a5112",
    "90527c5001119a9f5ed3ac4d39b8561897dbe515ef83367625f7ac07568dc747",
    "edeeeb192068a2e39c2330f0587b0c5498c7ea9a593432b6a1fbb853a5321c49",
    "b9b6298e6c8533a62fadf1f7c58002a928bbac981169a8705a69000635758951",
    "35c2fea82db7db4879ae89af82cb95d30f56ce3d03cbc21bfd6390ebd445feea",
    "b3bab3e99d78e2352f6e224d6628064fe447ae9177be682a045c830ad1c74645",
    "ea9b76bb38ce7a7332ccc471014e0f94474b19a0b98529733e5b67eb19515053",
    "5a57d4fee949aa23a5d7c888b776c01c0738005ec26cdefc4e54059da948e637",
    "2690ba526d2a08c745449ec8acb459352a6922ddd60918857f285db304f0ea95",
    "f494bec2b07925b041cf2cfe7140a91d8f323b482b18ae8c72a8493d017e35cc",
    "59a88cd3b33a57f85688b82d56aa2497830e5d86e0cd228aa481545a9da4dc5d",
    "c9c3ba7bd39a7d00ce962b8daa26361ccf4b2969ba82621f2711fd8d04b0f4b1",
    "13b7a6ddeeef7c846c7e8b862a19ca60e6d10b0e031be6b8b18e4fe07a3f3b5f",
    "434db6d7deea6653c40d7108608d948d20db6d01ce462ed41b33f950929a3b00",
    "3ce2ec58ad58b9666b710b309eb4b3a389215429c40d09bc5f068c2bf7bc9c58",
    "040d1e1f37526699020a9ebb3de0bbd1a5c21a7d99f2ec74d233263754898235",
    "2eccff8dbc12e9c62dd5537915e4a6bf50ca18eb254bf1776ce8a08685176e3d",
    "8d2abec33f0d7ece586fee5eb0b242be42f97f9eaa8a39c8a7bed3cc2cbb374a",
    "731ba63475a53f0896bcb489b584dbd4ab25f814c6813733d976ffc1421261d4",
    "5c262d714cdf0e4f721ca4add1bacbf46e87d4364a1baf7f5fd0e09474fbf2bc",
    "c39538e917a754b70b55ed0d4a70b1ac91445ed9c97e7762dbfbf3229d969e44",
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

# Full dataclass-aware result pins for the JANAF parent-liquid rows. The
# numeric projection above pins the same pressures and oxygen-balance outputs.
_OXYGEN_BALANCE_SF04_RESULT_HEX_SHA256 = (
    "f3e528a702d090538669d7ae1586a2990073ce9493e71aa3ee19029815738217",
    "9d8dea4f8fe25e7cee544df45a85ae737090c7e481c588dca9ebdfc9eb965917",
    "2c15e8eb142bb0687f2f1dd71cda3060728fc606f179b6a0cdad4e757d112278",
    "8ca5ffedb38c56eb796074e7885d298de964c4612af13d851347caf6c98396ba",
    "4327620f3c6e53911db25a8067eee0bd566b81b2286d76f1d59380f65853dc4c",
    "f68f64c979d24b892aa2de9f2be5f7c18662667d8cdd04aa079db25d2e2155b0",
    "91968b2186a903e7cc7c8c1971bb62533181035b301b667531e6a4c8cf4107c8",
    "1a2cd92c96954fb375d9601d02a898aa76b2c38a6648cdde923c4eff3d770152",
    "888bdaaac97bef24ed1931a5722a2786f8ea82cef9f98df51893395ab8f2e414",
    "05f05ffc2288acc8176635615089536b01986476004c7a699f1a18eb6f870fda",
    "a9b2ac930ad83c64f0153e66a55d4961c445267bae6fea61dc2ca43e870657c5",
    "8a6a5e82384558c743f5eb78c01e4045a2b3ca157056d75065792aea2ee7cbef",
    "62bad3f12b9b0c9ebe80b7afd79f2ebfcb1fa126c33eec2fe80ea09e27260688",
    "208d9ee9afe6607085d1caa7f8db1665a861f96e7c0a90287a5b5ec86c7de853",
    "5c5c923671ae4e9ae896b8632026e159fc9605f207154e4ef04f05b3efda4920",
    "f6d2507d1328f91a8e812460e3bd666bbd8bef18c812e0808c5701c871ed2d02",
    "f1b7495f42361633d99975a6293fabc34c409162a90b52bc7955c0a9e59b0223",
    "9bb2d05218bf7198c653633674e9b13c13d071344f9e39189cb0d85b18089ee2",
    "4b2e224d2b61c887ebbdfba2ba86ca5f49605078cef7b3d50bbb9c6a61bc811b",
    "e569fcd1484e7bf23d192f2d8de122711831e290bf8c7e775502bddec9cddcbd",
    "a95b925db2ce353bd674dc885f238d2ce0502d4afe7648403687fcdc67a8d814",
    "3baa5ce8d276f5f4083f196415a3d1b988507b428e7812391b88b829855113e9",
    "12c5b9411a62179db59391b69ea857da41dd93935df7fe838c2c77c04691cfdc",
    "765464119519e59be6649706fdb11dbba75eb79350f0f477638d0364788921f7",
    "1b6b90f2fd6db50a9f43ace7c54159e26fe3eaa0bdaefcb94d96169805063d31",
    "793b3d7bbd367f1772c8f0ada0d23faf19fdfd4885beeea1699e212dba6bc10b",
    "ec02f606efb851b2bbe3f687a3418d6de2bc67aa988b7c27e5ce77a61ad385cd",
    "0b6226afdb07726bd18c59a7a0d13eaf6f2a85f4f847ff979212f268dae76962",
    "c6abd56d2660ada8ac11f05481977165ca3a4134fa71730f573b4138aa4ea0d8",
    "346afbd887802352b09861a62e02e1b80184ded07e7c96b0e938df6b18458e11",
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
_OXYGEN_BALANCE_DEFAULT_NUMERIC_HEX_SHA256 = (
    "d2033c112069ae98c285af4e89d6abf06f43ca4d6c4cf2e72b4db2042f4076af",
    "9fbd36416e87b74cfb74f09657e3ce0a91d038517e9d7f6bb6814cfc64212ce8",
    "32b848588fe5abc36ce6fb977a8450c0bd1a7c1ff5b1847a1b13b428debb9b92",
    "4e1e182e893756c36ce91e869ce522363dde7012acd74bb7aa872e1f7d2cbf8f",
    "10360077375611067efb345dd22496eba3da6e4d07f649e52998dd4b11138a7a",
    "b3658cfe05b4defd4db03aa16418bb2f129b4b748cba8f9f1fd02108fa10b7c0",
    "faf2d02f6ab9a68803219d785c1fafdb84bf38b5fbf85d535c1182245c6c760e",
    "8a28ca153026b34bf0517ff0a5fe2d213723c8fad4469193bff1bdd018165e4c",
    "92569ada21ba6bf2ab84626408f52f900e069bb53f73a1876044ccc96bebc13b",
    "3a4de033e220cd44456821704fb5800ab5e8e4d341f7a87d643ea0b259700165",
    "6e0bd46f9405d6df79b76915f1c3890eb98304f5db2cdb2afddf76ec52d8e8f4",
    "f20e18b73215bbd822cdd8528cbb72a5a14933864490685b99839a69932afbd6",
    "e7bc5f866fb57c1fa3ce0ced2962ce8e0c6c6bc1ba247cf231178687499add04",
    "72e540ba71251927c5944062c2e48599f5187f055dc7105c3c7ec20fa41312ff",
    "c0ccea5bdb8bae81dc1d1de7f53d0788a3561398da11657c6ce8db20260ae934",
    "503368afa3600c825ec70d4af9d6beb757bc68b8630c9a99bee54409a8033c2d",
    "a30863d423ba09eaa6b7eb176a15cc97f273165d4329f9322127c2930eb8d0b2",
    "63c5337e265d1f152307f4d98e5c86ae68747d4b3c6852aece274ab8343c0ce5",
    "58f64d092d60441c64ebfee1278d6039bafc53f97e658fb619411a6225552e03",
    "2f35951ba8493ca5b2a3d01824ad42d737ff936fe74e90a97c7a1690fe777615",
    "fb612fac3d8a2770eb1f1af2ea100fa4aadc94b032a63f06ee0082596f9931b9",
    "647f4f1e2a8e85dc71996b371ef0511ac0a9166b900c65839f0dbb684bc8bff6",
    "db749be691162a5d5bdae36c16cebfad813b0589556f8b086aed60c95e4cfe19",
    "418196e20c33b54f01f400772cf4a24e325cb7b299352022cf066b2e53d08204",
    "a12460c0a9f3ba210bfc5a2c7c16562a66ddfd68f63647eef8c92a1d02c35b0f",
    "21a389cbe3d5838a5c93d4c879c395aa3647b698761fe575ef2d7d8d703eeb69",
    "a14f057abd29699292f2544d33d98e805b1377c60b4fb8ec86d0facca8260cd9",
    "f208f77bb168794bb94a421c8d706fdfac6344d52b382a3800192deb61bdf817",
    "43a4524f7d5fab23053dc914d51a1189fd5d3f332387974c3bec00c6b85e91fa",
    "b1b842ae6ff91b3938683e5176223c94bc1bb76151e0972d8458a6690d07f9d5",
    "42f187fde5210d1ae5e39594ba766591188df5b5f48e4c6d80245b91582b177d",
    "677233e96078d3f2fcc97bfd2f9b3423f8e697996e8cef8784a2e0b28a31f31a",
    "b3ab07d2de9c109905e665a23c1276631d64768261410714b166645d3bd4dfc7",
    "e740037884d0daa1cbbe5560b975e3848dcc3d258b4614eaf051db60244b6c1b",
    "72e0f81e5881c79d19ec4a29046e6395a10e565d0b9f3191b6328fde6e2976d0",
    "ae08d6c709aaf67f25a7d4c410ed82f3a24cfeddcafdfc5e7c8646fb82eba469",
    "d659052c64f4b6897579f6b737657af2f0c5bd24861212bce9391948ca62738b",
    "da2eb8b5085b094b89ded3938610b3841d670a0a81296425d5062a59d804bf76",
    "c4d772ececfeecca0e5294b1ff8ee237ac9a56d200b40ac7f0602203142108e0",
    "477c3d9a2f5d42ddbaac48071d1b5acbea2c756de861ebf9aa4305a927bf0afe",
    "d6218cd167cd86670ec67874427cb44708bf536234ab519d68c8db2404b481b1",
    "b5349ac39349da704416b7d76f3f44ca1d0d064d60985eaac64bef008f20435e",
    "07cae7f774c9bc8e9377e5199369e6d174218801f64909c8dc7c0e7cd1a66e1a",
    "d1cc814a549304980afb111687797889cb4401952b6bc6649fd97a19520d2416",
    "039923e97084ef0c81419bfbb84233b43db4df411ff82c5ceabecf2977adec1e",
    "0da18af9f9780433338d51d261392e1544f20adcd02f2871d43048e5679ee3b2",
    "2744340b06adbacf5a5f1cc87cbce616d1b543c9fa790cb4f62ac42023ef4793",
    "5ee7a522ac1d1fcb3a90a9b25aa5646b6cb38c3c10eee99eb90e7526c5bbf31b",
    "b9dbf2ef73dd815365dee74b893dcdb0414c372d68530fedadc2ac093af735bf",
    "e72a549b8d9adfd179a87d77d47d11a0eff172c633bf671664becd7ff40099aa",
)

_OXYGEN_BALANCE_DEFAULT_RESULT_HEX_SHA256 = (
    "52b635a53089b0005e06e41fd13a66014e9c8c4309b7386c86ad5e4f96849cbe",
    "3543bb2edc49919fb346754708dbcd6f38f65e0de8e0c53ca5be49c8dc8d66f0",
    "00de01c0a86d3cd5699e38fd075c980649c44b936d14e7dabe84068d4fb97584",
    "da983723a078300d99bb04db02c3808b199405c8cbb9e63a595ab8f224cc7012",
    "503b2adbaae610002566fde986bbb85ab09553f90dbb84c55401326448fc946d",
    "8edf76ef78425d2dfd7e0b741a96df9127f0aa9a102e94737d25385c006b2117",
    "bfea7634e43b2747e472038f87abed229d418612329c99613322f64651869500",
    "6c5b515a85da31eb09c2357058f37cdebb1bf54f5e60bb30f99080248c3abbc2",
    "65fdc59117c9f9e9d0a198acd7918e367d6b4c2e71885652ad12fd85364a848c",
    "5483867bd43ffa83ecb04dfb56d5317d1e7fcaecc94c3a1823bc57ab88048969",
    "94ecac0bf53f603eb7dfbbe55f8312da4791a6499fac69b703748c2831248f96",
    "85b722dad926987a2c4a89f04e61b3fd33209a2aee4acff0392521fa3f70a911",
    "6362f6412e4a5b25d495dd584b745a130fdca4983f48fe2da6dcde5f05f9e34c",
    "a1b2e8f585562afd67ca0c92394de3cf0a072eea2a61805ac1b7e4f542233251",
    "3bc2de392af4438d3e2a26f064d0e7bc3205fabe4b77b8023e8f56af468d94fb",
    "2300c1daf8663cd6da785242a8c41d932f08662fb02e669cf7ced42d9ffd2479",
    "a1991e4a1ba24ac3aba4df3aa63a0613fc222a59ad856f5fb4bfa4394af7d588",
    "75b7292a88471b3047987245da0979178b77dfb2dc964e5920adb26cd54d3983",
    "881fa23f32f36bb925498ce051a337b803db683c06eb5e8b933ae53e96749a68",
    "26eea50a50ab3e539fd8da10ab6cef63f4ae256ea300dab4359cd0d062159457",
    "5d8c4907b8b722021edb7e711457818e0b9ac979facf4bfb5a0844096aa80af0",
    "8a7c9d11df0ed1776a25213665c88b741c759263c24d46e0c6d0748ee7cb6655",
    "4f74f11105ff5a98d43a94db12e66368b5a5f52b2f3bd7a543620f7b6fc07fe3",
    "65908bb3aaa2cfb495ca43084ff6b91c2d4898a568fefa5d68500c47b7fbf932",
    "8a80c89fd0407c0c43867d20b491422682c668f22a458caecf3cafc046238949",
    "fb1ef33a8bc82a34ef38371b803f985f61b8a29a325de43ea79f4163b61adf30",
    "4aa3c79cb513252c8fd922cee39afc9be9633fd182af34602e8f840b626dd006",
    "d6fc42370e79588d657a60dfa465faa6f0b23fc1da1ec241481b3c9265ca015d",
    "ba08a5c07a7d8cfda16cdd026026970b8dbfb4eacdee19912b6e926b001b8d91",
    "16f90662a71b0b4ca8e8ad6722a92f18b2a221d17399ae77c7f63443fcf9c54c",
    "4f372382461374c6fc0b09776bb7a090011aacb32a4e830f2e7aa1f0c8e6ae2b",
    "82d4f986fdacf01854241466711df1ed9e89fab178b188243bd350b0c1864ce7",
    "378a1e6fe81482f7aaf8f6dc39af8cf5c34565dc9d2d54e6b2f62b52dd025f21",
    "d3468aa73397ed6908b5e444106a7867135326a1130920b2d034785b8148ee62",
    "7a07fed3dc6c6250b26f00c17a6c11ccbec99677a682425c599ca15ca944b7a4",
    "18d562189c35e7a0e3a68cca36190c06519d750ac21df1b9c45b9e1db7bf8670",
    "04787b57293492ef8967e8ca0ff59ff4bf9c49686387acfebf9ca570af5ad6ac",
    "8822246789e53d1f6ec177611f8283cf566b5fccd4079807c412310ad8e72cc3",
    "e11906d9debd38c0cfe1cab962b05d23d381acfdf3cb83d5268d0012e3ff7fb1",
    "dbbbd78c424372cb4daedea56816f454659650845440ba9f8c9247b2bd6351c5",
    "d37d13c1fe6093f65061559fbea78a9fa655c59d0ef531d600353b45cbfd3f29",
    "4cc08cb3f03e1fe07fbd76b371ff9c4236abe72eef7ecdda90b0d359c85ccdd4",
    "b3c849813d15569888b43e9ad1c4fac30fd38d3767217e67e39e36655284a571",
    "c261032e5d8f86042142903e6053e77aebdff3aa171adae3a35180755f143a3a",
    "ea50ca29ef5587000f3fca5fdb7af4de76facba6d501369165e62d13335933f4",
    "e4efe715693880b86a0d3e58490205e66be06595b282fa5981ee690736965c1f",
    "d6970c87b3888a6d67add36fb675ded612da755e86cc858aabb40f405ef1733f",
    "ecfc17b1967e837460db7c6c9cdcdb14d431f9e46cd8fc731a997548348eb649",
    "993bdd71676b3351cab991455deff4bf09fdc2d8e89ba2e783f4417525a2b252",
    "9ea364ddd0ab1d0fd8fd3612067e16012a6b51be3742059f6b394c12bd144cae",
)


def _oxygen_balance_hashes(
    gas_pack: ImccGasDatapack,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
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
        result_for_hash = replace(
            pressures,
            domain_flags={
                name: (
                    "; ".join(
                        part
                        for part in flag.split("; ")
                        if "source-tabulated supercooled liquid branch" not in part
                        and "liquid_from_solid construction from K-012" not in part
                    )
                    or None
                )
                if flag is not None
                else None
                for name, flag in pressures.domain_flags.items()
            },
        )
        canonical = json.dumps(
            hex_values((p_o2, result_for_hash, diagnostics)),
            sort_keys=True,
            separators=(",", ":"),
        )
        actual.append(hashlib.sha256(canonical.encode()).hexdigest())
    assert len(actual) == 50
    assert refusals == []
    return tuple(numeric_actual), tuple(actual)



def test_oxygen_balance_sf04_pack_matches_hash_pins(
    sf04_gas_pack: ImccGasDatapack,
) -> None:
    numeric_actual, result_actual = _oxygen_balance_hashes(sf04_gas_pack)
    assert numeric_actual == _OXYGEN_BALANCE_SF04_NUMERIC_HEX_SHA256
    assert result_actual == _OXYGEN_BALANCE_SF04_RESULT_HEX_SHA256


def test_oxygen_balance_default_pack_matches_hash_pins(
    gas_pack: ImccGasDatapack,
) -> None:
    numeric_actual, result_actual = _oxygen_balance_hashes(gas_pack)
    assert numeric_actual == _OXYGEN_BALANCE_DEFAULT_NUMERIC_HEX_SHA256
    assert result_actual == _OXYGEN_BALANCE_DEFAULT_RESULT_HEX_SHA256


def test_oxygen_balance_published_pack_matches_base_in_same_environment(
    sf04_gas_pack: ImccGasDatapack,
    tmp_path: Path,
) -> None:
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
    cases = [(composition, temperature) for composition in compositions for temperature in (1500., 2000., 2400.)]
    for first, second in (("SiO2","MgO"),("SiO2","FeO"),("SiO2","CaO"),("SiO2","Al2O3"),("SiO2","K2O")):
        cases.extend(({first:.6-.05*i,second:.4+.05*i}, temperature) for i, temperature in enumerate((1500., 1900., 2200.)))
    cases.extend(({"SiO2":.45+.02*i,"MgO":.15-.01*i,"CaO":.2,"Al2O3":.2}, temperature)
                 for i, temperature in enumerate((1500., 2000., 2400., 1800., 2200.)))

    base_pack = _load_task_base_gas_pack(tmp_path)
    for activities, temperature in cases:
        parents = tuple(activities)
        base = evaluate_gas_oxygen_balance(
            activities, temperature, base_pack, parent_oxides=parents
        )
        published = evaluate_gas_oxygen_balance(
            activities, temperature, sf04_gas_pack, parent_oxides=parents
        )
        assert base[0] == published[0]
        assert dict(base[1]) == dict(published[1])
        assert dict(base[1].domain_flags) == dict(published[1].domain_flags)
        assert dict(base[1].provenance_class) == dict(published[1].provenance_class)
        assert base[2] == published[2]
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
    sf04_gas_pack: ImccGasDatapack,
    activities: dict[str, float],
    temperature: float,
    parents: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ImccGasOxygenBalanceError) as error:
        evaluate_gas_oxygen_balance(
            activities, temperature, sf04_gas_pack, parent_oxides=parents
        )
    assert str(error.value) == message


def test_constant_cp_parent_extensions_are_flagged_at_interval_boundaries() -> None:
    pack = load_gas_datapack()
    extension_rows = pack.oxide_df.loc[
        pack.oxide_df["Ref"].astype(str).str.endswith("-SC-CP")
    ]
    assert set(extension_rows.index) == {
        "TiO2(l)", "Cr2O3(l)", "V2O3(l)", "SiO2(l)",
        "Al2O3(l)", "MgO(l)", "CaO(l)",
        "Ga2O3(l)", "GeO2(l)", "In2O3(l)",
        "Sc2O3(l)", "UO2(l)",
        "Al2O3(l)", "MgO(l)", "CaO(l)", "SnO(l)",
        "BeO(l)",
    }
    for row_name, row in extension_rows.iterrows():
        species = row_name.removesuffix("(l)")
        row_t_min = float(row["T_min"])
        thermo = species_thermo(species, "l", row_t_min, pack)
        source_table = str(row["Ref"]).removesuffix("-SC-CP")
        assert thermo.source_table_id == source_table
        assert thermo.source_row_id == str(row["Ref"])
        assert (thermo.T_min, thermo.T_max) == (
            row_t_min, float(row["T_max"])
        )

    low = evaluate_gas(
        {"TiO2": 1.0}, 1200.0, 1.0e-10, pack, gas_species=("Ti",)
    )
    assert "constant-Cp supercooled-liquid continuation" in low.domain_flags["Ti"]
    assert "JANAF O-044" in low.domain_flags["Ti"]
    low_li = evaluate_gas(
        {"Li2O": 1.0}, 1200.0, 1.0e-10, pack,
        gas_species=("Li",), allow_extrapolation=True,
    )
    assert "constant-Cp supercooled-liquid continuation" not in (
        low_li.domain_flags["Li"] or ""
    )
    low_v = evaluate_gas(
        {"V2O3": 1.0}, 1200.0, 1.0e-10, pack,
        gas_species=("V",), allow_extrapolation=True,
    )
    assert "constant-Cp supercooled-liquid continuation" in low_v.domain_flags["V"]
    low_sn = evaluate_gas(
        {"SnO": 1.0}, 1200.0, 1.0e-10, pack, gas_species=("Sn",)
    )
    assert "constant-Cp supercooled-liquid continuation" in low_sn.domain_flags["Sn"]
    assert "NASA Glenn card NG-1848" in low_sn.domain_flags["Sn"]
    at_1500_v = evaluate_gas(
        {"V2O3": 1.0}, 1500.0, 1.0e-10, pack,
        gas_species=("V",), allow_extrapolation=False,
    )
    assert "constant-Cp supercooled-liquid continuation" in (
        at_1500_v.domain_flags["V"]
    )
    at_1700_v = evaluate_gas(
        {"V2O3": 1.0}, 1700.0, 1.0e-10, pack,
        gas_species=("V",), allow_extrapolation=False,
    )
    assert "source-tabulated supercooled liquid branch" in (
        at_1700_v.domain_flags["V"] or ""
    )
    assert "2340 K fusion marker" in at_1700_v.domain_flags["V"]

    without_extensions = replace(
        pack,
        oxide_df=pack.oxide_df.loc[
            ~pack.oxide_df["Ref"].astype(str).str.endswith("-SC-CP")
        ],
    )
    parent_channels = {
        "TiO2(l)": ("TiO2", "Ti"),
        "Cr2O3(l)": ("Cr2O3", "Cr"),
        "V2O3(l)": ("V2O3", "V"),
        "SnO(l)": ("SnO", "Sn"),
    }
    for row_name, (oxide, channel) in parent_channels.items():
        high = pack.oxide_df.loc[row_name]
        high = high.loc[~high["Ref"].astype(str).str.endswith("-SC-CP")]
        high = high.iloc[0]
        start = float(high["T_min"])
        temperatures = {
            start,
            float(high["T_max"]),
            *(temperature for temperature in IMCC_SF04_WORKBOOK_GRID_K
              if start <= temperature <= float(high["T_max"])),
        }
        for temperature in sorted(temperatures):
            current = evaluate_gas(
                {oxide: 1.0}, temperature, 1.0e-10, pack,
                gas_species=(channel,), allow_extrapolation=True,
            )
            previous = evaluate_gas(
                {oxide: 1.0}, temperature, 1.0e-10, without_extensions,
                gas_species=(channel,), allow_extrapolation=True,
            )
            assert dict(current.items()) == dict(previous.items())
            assert dict(current.domain_flags) == dict(previous.domain_flags)


def _load_task_base_gas_pack(tmp_path: Path) -> ImccGasDatapack:
    root = Path(__file__).resolve().parents[1]
    baseline_dir = tmp_path / "task-base-gas"
    baseline_dir.mkdir()
    baseline_paths = {}
    for filename, repository_path in (
        ("gas-shomate.csv", "src/openimcc/data/gas/gas-shomate.csv"),
        ("condensate.csv", "src/openimcc/data/gas/condensate.csv"),
    ):
        payload = subprocess.run(
            ["git", "show", f"5380e46:{repository_path}"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        path = baseline_dir / filename
        path.write_bytes(payload)
        baseline_paths[filename] = path
    return load_gas_datapack(
        gas_path=baseline_paths["gas-shomate.csv"],
        oxide_path=baseline_paths["condensate.csv"],
    )


@pytest.mark.parametrize("collection_type", (tuple, list))
def test_sf04_pack_reproduces_base_default_and_default_keeps_other_channels(
    sf04_gas_pack: ImccGasDatapack,
    collection_type,
    tmp_path: Path,
) -> None:
    """Compare every 1 K result with the base tables in this environment."""
    base_pack = _load_task_base_gas_pack(tmp_path)
    default_pack = load_gas_datapack()
    parents = collection_type(
        sorted(parent for parent in _REACTION_PARENT_OXIDES if parent != "Ag2O")
    )
    activities = {parent: 1.0 for parent in parents}
    selected, _omitted = _default_reactions(parents, base_pack)
    channels = collection_type(name for name, _reaction in selected)
    switched_parent_oxides = {
        "TiO2", "Cr2O3", "V2O3", "SiO2",
        "Al2O3", "MgO", "CaO", "K2O",
    }
    switched_channels = {
        name for name, reaction in selected
        if reaction[0] in switched_parent_oxides
    }
    new_low_temperature_channels = {
        "Al2", "Co", "Cr", "CrO", "CrO2", "CrO3", "Mn", "Nb", "NbO",
        "NbO2", "Ni", "Si2", "Si3", "Ti", "TiO", "TiO2", "V", "VO", "VO2",
    }

    for temperature in range(1200, 3001):
        args = {
            "parent_oxides": parents,
            "gas_species": channels,
            "allow_extrapolation": True,
        }
        base = evaluate_gas(
            activities, float(temperature), 1.0e-10, base_pack, **args
        )
        published = evaluate_gas(
            activities, float(temperature), 1.0e-10, sf04_gas_pack, **args
        )
        newly_fitted_channels = (
            {name for name in channels if name in new_low_temperature_channels}
            if temperature < 1500
            else set()
        )
        unchanged_against_base = set(base) - newly_fitted_channels
        assert {name: published[name] for name in unchanged_against_base} == {
            name: base[name] for name in unchanged_against_base
        }
        expected_base_flags = dict(base.domain_flags)
        if 2900 < temperature <= 3000 and "Cr" in unchanged_against_base:
            expected_base_flags["Cr"] = None
        assert {
            name: published.domain_flags[name] for name in unchanged_against_base
        } == {
            name: expected_base_flags[name] for name in unchanged_against_base
        }
        assert {
            name: published.provenance_class[name] for name in unchanged_against_base
        } == {name: base.provenance_class[name] for name in unchanged_against_base}
        assert dict(published.omitted_channels) == dict(base.omitted_channels)

        current = evaluate_gas(
            activities, float(temperature), 1.0e-10, default_pack, **args
        )
        unchanged = set(base) - switched_channels - newly_fitted_channels
        assert {name: current[name] for name in unchanged} == {
            name: base[name] for name in unchanged
        }
        assert {name: current.domain_flags[name] for name in unchanged} == {
            name: base.domain_flags[name] for name in unchanged
        }
        assert {name: current.provenance_class[name] for name in unchanged} == {
            name: base.provenance_class[name] for name in unchanged
        }


def test_la_channels_use_caller_parent_and_hand_mass_action(
    gas_pack: ImccGasDatapack,
) -> None:
    temperature = 2000.0
    activity = 1.0e-3
    oxygen_fugacity = 1.0e-4
    result = evaluate_gas(
        {"La2O3": activity}, temperature, oxygen_fugacity, gas_pack,
        parent_oxides=("La2O3",), gas_species=("La", "LaO"),
    )
    assert tuple(result) == ("La", "LaO")
    assert "supercooled continuation below 2577 K" in result.domain_flags["La"]
    parent_g = species_thermo("La2O3", "l", temperature, gas_pack).G_J_mol
    oxygen_g = species_thermo("O2", "g", temperature, gas_pack).G_J_mol
    for species, n_oxygen in (("La", 1.5), ("LaO", 0.5)):
        gas_g = species_thermo(species, "g", temperature, gas_pack).G_J_mol
        delta_g = 2.0 * gas_g + n_oxygen * oxygen_g - parent_g
        log10_k = -delta_g / (R_J_MOL_K * temperature * math.log(10.0))
        hand_log10_pressure = (
            log10_k + math.log10(activity)
            - n_oxygen * math.log10(oxygen_fugacity)
        ) / 2.0
        hand = 10.0**hand_log10_pressure
        assert result[species] == pytest.approx(hand, rel=2e-12)


@pytest.mark.parametrize(
    ("parent", "species", "n_gas", "n_oxygen"),
    [
        ("Ce2O3", "Ce", 2, 1.5), ("Ce2O3", "CeO", 2, 0.5),
        ("Pr2O3", "Pr", 2, 1.5), ("Pr2O3", "PrO", 2, 0.5),
        ("Nd2O3", "Nd", 2, 1.5), ("Nd2O3", "NdO", 2, 0.5),
        ("Sm2O3", "Sm", 2, 1.5), ("Sm2O3", "SmO", 2, 0.5),
        ("Eu2O3", "Eu", 2, 1.5), ("Eu2O3", "EuO", 2, 0.5),
        ("Gd2O3", "Gd", 2, 1.5), ("Gd2O3", "GdO", 2, 0.5),
        ("Tb2O3", "Tb", 2, 1.5), ("Tb2O3", "TbO", 2, 0.5),
        ("Dy2O3", "Dy", 2, 1.5), ("Dy2O3", "DyO", 2, 0.5),
        ("Ho2O3", "Ho", 2, 1.5), ("Ho2O3", "HoO", 2, 0.5),
        ("Er2O3", "Er", 2, 1.5), ("Er2O3", "ErO", 2, 0.5),
        ("Tm2O3", "Tm", 2, 1.5), ("Tm2O3", "TmO", 2, 0.5),
        ("Yb2O3", "Yb", 2, 1.5), ("Yb2O3", "YbO", 2, 0.5),
        ("Lu2O3", "Lu", 2, 1.5), ("Lu2O3", "LuO", 2, 0.5),
    ],
)
def test_rare_earth_channels_match_hand_mass_action(
    gas_pack: ImccGasDatapack,
    parent: str,
    species: str,
    n_gas: int,
    n_oxygen: float,
) -> None:
    temperature = 2000.0
    activity = 1.0e-3
    oxygen_fugacity = 1.0e-4
    result = evaluate_gas(
        {parent: activity}, temperature, oxygen_fugacity, gas_pack,
        parent_oxides=(parent,), gas_species=(species,),
    )
    parent_g = species_thermo(parent, "l", temperature, gas_pack).G_J_mol
    gas_g = species_thermo(species, "g", temperature, gas_pack).G_J_mol
    oxygen_g = species_thermo("O2", "g", temperature, gas_pack).G_J_mol
    delta_g = n_gas * gas_g + n_oxygen * oxygen_g - parent_g
    log10_k = -delta_g / (R_J_MOL_K * temperature * math.log(10.0))
    log10_pressure = (
        log10_k + math.log10(activity)
        - n_oxygen * math.log10(oxygen_fugacity)
    ) / n_gas
    assert result[species] == pytest.approx(10.0**log10_pressure, rel=2e-12)


def test_rare_earth_channels_require_the_caller_parent(
    gas_pack: ImccGasDatapack, tmp_path: Path
) -> None:
    pairs = tuple(
        (element, f"{element}2O3", species)
        for element in ("Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu")
        for species in (element, f"{element}O")
    )
    oxide_df = pd.read_csv(gas_pack.oxide_path)
    missing_parents = {f"{parent}(l)" for _, parent, _ in pairs}
    oxide_path = tmp_path / "condensate-without-rare-earth-parents.csv"
    oxide_df.loc[~oxide_df["species_name"].isin(missing_parents)].to_csv(
        oxide_path, index=False
    )
    pack_without_parents = load_gas_datapack(
        gas_path=gas_pack.gas_path, oxide_path=oxide_path
    )
    for _element, parent, species in pairs:
        with pytest.raises(ImccGasSpeciesNotFoundError) as exc:
            evaluate_gas(
                {parent: 1.0e-3}, 2000.0, 1.0e-4, pack_without_parents,
                parent_oxides=(parent,), gas_species=(species,),
            )
        assert exc.value.code == "imcc_gas_species_not_found"


def test_cerium_dioxide_channel_uses_caller_supplied_ceo2_liquid(
    gas_pack: ImccGasDatapack,
) -> None:
    temperature = 2000.0
    activity = 1.0e-3
    result = evaluate_gas(
        {"CeO2": activity}, temperature, 1.0e-4, gas_pack,
        parent_oxides=("CeO2",), gas_species=("CeO2",),
    )
    parent_g = species_thermo("CeO2", "l", temperature, gas_pack).G_J_mol
    gas_g = species_thermo("CeO2", "g", temperature, gas_pack).G_J_mol
    expected = activity * math.exp(-(gas_g - parent_g) / (R_J_MOL_K * temperature))
    assert result["CeO2"] == pytest.approx(expected, rel=2e-12)
    assert "supercooled continuation below 3083 K" in result.domain_flags["CeO2"]
