"""Runtime gates for the packaged IMCC-SF04 gas mass-action layer."""

from __future__ import annotations

import csv
import dataclasses
from enum import Enum
import hashlib
import inspect
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

import numpy as np
import pandas as pd
import pytest
import yaml

from openimcc import LiquidReferenceAuthority, evaluate
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
    ImccGasSourceOverrideWarning,
    ImccGasInvalidIntervalError,
    ImccGasOxygenBalanceError,
    ImccGasResult,
    ImccGasSpeciesNotFoundError,
    ImccGasTemperatureOutsideDomainError,
    ImccGasUnknownActivityError,
    ImccGasUnrecognizedSourceError,
    ImccGasEquilibriumRangeError,
    OxygenBalanceSpeciesMetadata,
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
    "Au+", "Au-", "Hg+", "Pd+", "Pt+", "Pt-",
}
_ION_GAS_SPECIES.update(
    species.removesuffix("(g)")
    for species, _element in build_gas_tables.CONSTRUCTED_SAHA_ION_SOURCES
)


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


def test_default_reactions_name_signature_and_return_shape_are_stable(gas_pack) -> None:
    assert _default_reactions.__name__ == "_default_reactions"
    signature = inspect.signature(_default_reactions)
    assert tuple(signature.parameters) == ("available_parents", "datapack", "T_K")
    assert signature.parameters["T_K"].default is None
    selected, omitted = _default_reactions(("K2O",), gas_pack)
    assert isinstance(selected, tuple) and selected
    assert all(
        isinstance(item, tuple)
        and len(item) == 2
        and isinstance(item[0], str)
        and isinstance(item[1], tuple)
        and len(item[1]) == 3
        for item in selected
    )
    assert isinstance(omitted, dict)
    assert all(
        isinstance(name, str) and isinstance(reason, str)
        for name, reason in omitted.items()
    )


@pytest.mark.parametrize("value", [True, False, np.bool_(True), np.bool_(False)])
def test_evaluate_gas_refuses_boolean_fugacity(gas_pack, value):
    with pytest.raises(ImccGasInvalidFugacityError):
        evaluate_gas(
            {"Na2O": 0.1}, 1800.0, value, gas_pack, gas_species=("Na",)
        )


@pytest.mark.parametrize("value", [True, False, np.bool_(True), np.bool_(False)])
def test_evaluate_gas_refuses_boolean_activity(gas_pack, value):
    with pytest.raises(ValueError, match="activity.*real number"):
        evaluate_gas(
            {"Na2O": value}, 1800.0, 1.0e-8, gas_pack, gas_species=("Na",)
        )


def test_evaluate_gas_refuses_boolean_temperature(gas_pack):
    with pytest.raises(ValueError, match="temperature must be a real number"):
        evaluate_gas(
            {"Na2O": 0.1}, True, 1.0e-8, gas_pack, gas_species=("Na",)
        )


def test_evaluate_gas_refuses_temperature_beyond_float_range(gas_pack):
    with pytest.raises(ValueError, match="temperature must be finite and positive"):
        evaluate_gas(
            {"Na2O": 0.1}, 10**400, 1.0e-8, gas_pack, gas_species=("Na",)
        )


def test_evaluate_gas_refuses_activity_beyond_float_range(gas_pack):
    with pytest.raises(ValueError, match="activity.*too large to represent"):
        evaluate_gas(
            {"Na2O": 10**400}, 1800.0, 1.0e-8, gas_pack, gas_species=("Na",)
        )


def test_evaluate_gas_oxygen_balance_refuses_temperature_beyond_float_range(gas_pack):
    with pytest.raises(ValueError, match="temperature must be finite and positive"):
        evaluate_gas_oxygen_balance({}, 10**400, gas_pack)


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
        63
        + len(build_gas_tables.LOW_T_GAS_SPECIES)
        + 2
        + len(_ION_GAS_SPECIES)
        + len(build_gas_tables.TRACE_ION_GAS_NASA_SOURCES)
        + len(build_gas_tables.TRACE_ION_NASA_SOURCES)
        + len(build_gas_tables.TRACE_ION_GAS_YAML_SOURCES)
        + len(build_gas_tables.CONSTRUCTED_SAHA_ION_SOURCES)
        + 3
        + 2 * len(build_gas_tables.TRACE_GAS_SPECIES)
        - len(build_gas_tables.HALOGEN_GAS_SPECIES)
        + 1  # Sc+ has an additional interval row.
        + 2 * len(_RARE_EARTH_CHANNELS)
        + sum(
            source[1] in build_gas_tables.HALOGEN_JANAF_TABLES
            for source in build_gas_tables.GAS_SOURCES
        )
        + 6  # Two gas fit intervals for Hg, Au, and Pd.
        + 5  # Pt and PtO2, including PtO2's continuation interval.
        + 2  # Ir gas rows; Ru and Rh are counted with trace gas species.
        + 2  # Os gas fit intervals.
        + 4  # Four Re2O7 constructed intervals.
        + 11  # Ten low JANAF halogen fits and the low Hg Burcat interval.
    )
    assert len(gas_pack.oxide_df) == 262


_MAJOR_PARENT_CONTINUATIONS = (
    ("Al2O3", "Al", 1700.0, "Al2O3(l)"),
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
    if oxide == "CaO":
        assert result.domain_flags[species] is None
        return
    assert "constant-Cp supercooled-liquid continuation" in (
        result.domain_flags[species]
    )
    assert parent_row in result.domain_flags[species]
    assert result.provenance_class[species] == "janaf_fitted"


def test_default_silica_parent_uses_promoted_liquid_construction(
    gas_pack: ImccGasDatapack,
) -> None:
    result = evaluate_gas(
        {"SiO2": 1.0}, 1700.0, 1.0e-10, gas_pack,
        parent_oxides=("SiO2",), gas_species=("SiO",),
        allow_extrapolation=False,
    )
    assert math.isfinite(result["SiO"])
    assert result.domain_flags["SiO"] is None


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

    with pytest.warns(
        ImccGasSourceOverrideWarning, match="OPENIMCC_VAPOROCK_ROOT"
    ) as notices:
        overridden = load_gas_datapack()
    assert str(root) in str(notices[0].message)
    assert overridden.gas_path == gas_target / "JANAF-vapor-data-full.csv"
    assert overridden.oxide_path == oxide_target / "condensate-thermo-data.csv"
    assert overridden.gas_df.equals(gas_pack.gas_df)
    assert overridden.oxide_df.equals(gas_pack.oxide_df)

    with pytest.warns(
        ImccGasSourceOverrideWarning, match="OPENIMCC_VAPOROCK_ROOT"
    ):
        load_gas_datapack(gas_path=overridden.gas_path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ImccGasSourceOverrideWarning)
        assert default_gas_database_path() == overridden.gas_path
        assert default_condensate_database_path() == overridden.oxide_path
        explicit = load_gas_datapack(
            gas_path=overridden.gas_path,
            oxide_path=overridden.oxide_path,
        )
    assert not any(
        issubclass(item.category, ImccGasSourceOverrideWarning) for item in caught
    )
    assert explicit.gas_df.equals(overridden.gas_df)
    assert explicit.oxide_df.equals(overridden.oxide_df)


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
    assert b"Ref,,,,validity_kind" in condensate_bytes
    assert b"SYNTHETIC,,,not-source-data,none,none,false" in condensate_bytes
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

    with (
        pytest.warns(ImccGasSourceOverrideWarning, match="OPENIMCC_VAPOROCK_ROOT"),
        pytest.warns(ImccGasIngestionWarning) as notices,
    ):
        overridden = load_gas_datapack()
    ingestion = [
        item for item in notices
        if issubclass(item.category, ImccGasIngestionWarning)
    ]
    message = str(ingestion[0].message)
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
    with (
        pytest.warns(ImccGasSourceOverrideWarning, match="OPENIMCC_VAPOROCK_ROOT"),
        pytest.warns(ImccGasIngestionWarning) as notices,
    ):
        datapack = load_gas_datapack()
    ingestion = [
        item for item in notices
        if issubclass(item.category, ImccGasIngestionWarning)
    ]
    message = str(ingestion[0].message)
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
        warnings.simplefilter("always", ImccGasSourceOverrideWarning)
        load_gas_datapack()
    assert not any(
        issubclass(item.category, ImccGasIngestionWarning) for item in caught
    )
    assert not any(
        issubclass(item.category, ImccGasSourceOverrideWarning) for item in caught
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
        with (
            pytest.warns(ImccGasSourceOverrideWarning, match="OPENIMCC_VAPOROCK_ROOT"),
            pytest.warns(ImccGasIngestionWarning),
        ):
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
        *build_gas_tables.SOURCE_VALIDITY_COLUMNS,
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
        *build_gas_tables.SOURCE_VALIDITY_COLUMNS,
    )
    public_gas_channels = set(IMCC_GAS_CHANNEL_SPECIES)
    assert set(gas_pack.gas_df.index) == {
        f"{species}(g)" for species in public_gas_channels
    } | {f"{species}(g)" for species in _ION_GAS_SPECIES} | {
        source[0] for source in build_gas_tables.TRACE_ION_GAS_NASA_SOURCES
    } | {
        source[0] for source in build_gas_tables.TRACE_ION_GAS_BURCAT_SOURCES
    } | {
        source[0] for source in build_gas_tables.CONSTRUCTED_SAHA_ION_SOURCES
    }
    high_rows = gas_pack.gas_df.loc[
        (gas_pack.gas_df["T_interval"].astype(int) == 1)
        & (gas_pack.gas_df.index != "Re2O7(g)")
    ]
    low_rows = gas_pack.gas_df.loc[
        (gas_pack.gas_df["T_interval"].astype(int) == 2)
        & (gas_pack.gas_df.index != "Re2O7(g)")
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
        + len(build_gas_tables.CONSTRUCTED_SAHA_ION_SOURCES)
        + 3
        + 10  # JANAF HCl/HF and alkali-halide low rows.
        + len(_RARE_EARTH_CHANNELS)
        + 6
        + 1  # Os(g) has one low interval; Re2O7(g) is excluded above.
    )
    assert set(low_rows.index) == {
        f"{species}(g)" for species in build_gas_tables.LOW_T_GAS_SPECIES
    } | {"Na2O(g)", "K2O(g)"} | {
        f"{species}(g)" for species in low_trace_species
    } | {
        f"{species}(g)" for species in (
            "HCl", "HF", "NaCl", "(NaCl)2", "KCl", "(KCl)2",
            "NaF", "(NaF)2", "KF", "(KF)2",
        )
    } | {
        f"{species}(g)" for species in _RARE_EARTH_CHANNELS
    } | {f"{species}(g)" for species in ("Hg", "Au", "Pd", "Pt", "PtO2", "Ir")} | {
        source[0]
        for source in build_gas_tables.TRACE_ION_GAS_NASA_SOURCES
    } | {
        species for species, *_ in build_gas_tables.TRACE_ION_NASA_SOURCES
    } | {
        source[0] for source in build_gas_tables.TRACE_ION_GAS_YAML_SOURCES
    } | {"Zn+(g)", "Zn-(g)", "Bi+(g)"} | {
        source[0] for source in build_gas_tables.CONSTRUCTED_SAHA_ION_SOURCES
    } | {"Os(g)"}
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
    } | {"Zn+(g)", "Zn-(g)", "Bi+(g)"} | {
        source[0] for source in build_gas_tables.CONSTRUCTED_SAHA_ION_SOURCES
    }
    split_ion_rows = ion_rows.loc[ion_rows.index.isin(split_ion_names)]
    single_interval_ion_rows = ion_rows.loc[
        ~ion_rows.index.isin(split_ion_names)
    ]
    reviewed_trace_species = {
        f"{source[0]}" for source in build_gas_tables.TRACE_PUBLISHED_GAS_SOURCES
        if source[1].startswith(("P672-Te", "P672-Se", "P672-As", "P672-Sb", "P672-Tl"))
        or source[1] == "Kelley-584-SbO"
    }
    reviewed_trace_species.add("Rh(g)")
    reviewed_trace_species.add("Re(g)")
    ordinary_high_rows = neutral_high_rows.loc[
        ~neutral_high_rows.index.isin(reviewed_trace_species)
        & ~neutral_high_rows.index.isin(
            tuple(f"{species}(g)" for species in build_gas_tables.HALOGEN_GAS_SPECIES)
        )
    ]
    assert (ordinary_high_rows["T_min"] == 1500).all()
    for source in build_gas_tables.TRACE_PUBLISHED_GAS_SOURCES:
        if (
            source[1].startswith(("P672-Te", "P672-Se", "P672-As", "P672-Sb", "P672-Tl"))
            or source[1] in {"Kelley-584-SbO", "P677-Re-g"}
        ):
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
    trace_low_species |= {
        "Hg(g)", "Au(g)", "Pd(g)", "Pt(g)", "PtO2(g)", "Ir(g)", "Os(g)"
    }
    lowered_trace_species = {
        "H2O(g)", "Zn(g)", "Cd(g)", "Pb(g)", "PbO(g)", "PbO2(g)",
        "HCl(g)", "HF(g)", "NaCl(g)", "(NaCl)2(g)", "KCl(g)", "(KCl)2(g)",
        "NaF(g)", "(NaF)2(g)", "KF(g)", "(KF)2(g)",
    }
    legacy_low_rows = low_rows.loc[~low_rows.index.isin(trace_low_species)]
    trace_low_rows = low_rows.loc[
        low_rows.index.isin(trace_low_species - lowered_trace_species)
    ]
    lowered_trace_rows = low_rows.loc[
        low_rows.index.isin(lowered_trace_species)
    ]
    assert (legacy_low_rows["T_min"] == build_gas_tables.LOW_FIT_T_MIN).all()
    assert (trace_low_rows["T_min"] == 1200.0).all()
    assert (lowered_trace_rows["T_min"] == 1000.0).all()
    low_halide_rows = lowered_trace_rows.loc[
        lowered_trace_rows.index.isin(
            {
                "HCl(g)", "HF(g)", "NaCl(g)", "(NaCl)2(g)", "KCl(g)",
                "(KCl)2(g)", "NaF(g)", "(NaF)2(g)", "KF(g)", "(KF)2(g)",
            }
        )
    ]
    assert (low_halide_rows["T_max"] == 1200.0).all()
    assert (legacy_low_rows["T_max"] == build_gas_tables.LOW_FIT_T_MAX).all()
    expected_t_max = high_rows["Ref"].map(
        lambda table_id: build_gas_tables._DECLARED_T_MAX_BY_TABLE.get(
            table_id,
            build_gas_tables._FIT_T_MAX_BY_TABLE.get(
                table_id, build_gas_tables.FIT_T_MAX
            ),
        )
    )
    expected_t_max.loc["PtO2(g)"] = 1900.0
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
        65
        + 6
        + 2  # Os and Re2O7 are exposed metal-vapor channels.
        + len(_RARE_EARTH_CHANNELS)
        + len(build_gas_tables.TRACE_GAS_SPECIES)
        - len({"NiO", "CoO"} & build_gas_tables.TRACE_GAS_SPECIES)
    )
    assert unavailable == {"AsO"}
    assert unavailable & implemented == set()
    assert unavailable - implemented == {"AsO"}
    assert in_domain.isdisjoint(extrapolated)
    assert not set(_P_CHANNELS) & (in_domain | extrapolated)
    assert (
        in_domain
        | extrapolated
        | set(_P_CHANNELS)
        | {"PS"}
        | set(_RARE_EARTH_CHANNELS)
        | {"Hg", "Au", "Pd", "Pt", "PtO2", "Ir", "Os", "Re2O7"}
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
    result = evaluate_gas(
        {oxide: 1.0},
        1200.0 if oxide == "SiO2" else 1199.0,
        1.0e-10,
        gas_pack,
        gas_species=(species,),
        allow_extrapolation=False,
    )
    assert math.isfinite(result[species])
    flag = result.domain_flags[species] or ""
    if oxide == "SiO2":
        assert flag == ""
    else:
        assert "constant-Cp supercooled-liquid continuation" in flag
    with pytest.raises(ImccGasTemperatureOutsideDomainError):
        evaluate_gas(
            {oxide: 1.0}, 999.0, 1.0e-10, gas_pack,
            gas_species=(species,), allow_extrapolation=False,
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


def test_stored_gas_tables_refuse_edits_that_would_diverge_from_their_digest(
    gas_pack: ImccGasDatapack,
) -> None:
    """A public cell write must not change pressures under a cached digest."""
    digest = gas_pack.gas_table_digest
    condensate_digest = gas_pack.condensate_table_digest
    pressure = evaluate_gas(
        {"SiO2": 1.0},
        1800.0,
        1e-8,
        gas_pack,
        parent_oxides=("SiO2",),
        gas_species=("SiO",),
        allow_extrapolation=False,
    )["SiO"]

    with pytest.raises((TypeError, ValueError)):
        gas_pack.gas_df.loc["SiO(g)", "A"] = gas_pack.gas_df.loc["SiO(g)", "A"] + 1
    with pytest.raises(TypeError, match="immutable gas table"):
        gas_pack.gas_df["A"] = gas_pack.gas_df["A"] + 1
    with pytest.raises(TypeError, match="immutable gas table"):
        gas_pack.gas_df.index = gas_pack.gas_df.index
    with pytest.raises((TypeError, ValueError)):
        gas_pack.oxide_df.loc["SiO2(l)", "dG_A"] = (
            gas_pack.oxide_df.loc["SiO2(l)", "dG_A"] + 1
        )

    again = evaluate_gas(
        {"SiO2": 1.0},
        1800.0,
        1e-8,
        gas_pack,
        parent_oxides=("SiO2",),
        gas_species=("SiO",),
        allow_extrapolation=False,
    )["SiO"]
    assert again == pressure
    assert gas_pack.gas_table_digest == digest
    assert gas_pack.condensate_table_digest == condensate_digest

    # A copy is the supported way to build a replacement table.
    editable = gas_pack.gas_df.copy()
    editable.loc["SiO(g)", "A"] = editable.loc["SiO(g)", "A"] + 1
    replaced = replace(
        gas_pack,
        gas_df=editable,
    )
    assert replaced.gas_table_digest != digest
    assert gas_pack.gas_table_digest == digest


def test_rejected_gas_pack_reinitialization_preserves_table_identity(
    gas_pack: ImccGasDatapack,
) -> None:
    digest = gas_pack.gas_table_digest
    condensate_digest = gas_pack.condensate_table_digest
    rows = len(gas_pack.gas_df)
    bad = gas_pack.gas_df.iloc[:1].copy()
    bad.loc[:, "T_min"] = 5000.0
    bad.loc[:, "T_max"] = 1000.0

    with pytest.raises(TypeError, match="cannot be initialized more than once"):
        gas_pack.__init__(
            gas_df=bad,
            oxide_df=gas_pack.oxide_df,
            gas_path=gas_pack.gas_path,
            oxide_path=gas_pack.oxide_path,
        )

    assert len(gas_pack.gas_df) == rows
    assert gas_pack.gas_table_digest == digest
    assert gas_pack.condensate_table_digest == condensate_digest
    assert evaluate_gas(
        {"SiO2": 1.0},
        1800.0,
        1e-8,
        gas_pack,
        parent_oxides=("SiO2",),
        gas_species=("SiO",),
        allow_extrapolation=False,
    )["SiO"] == evaluate_gas(
        {"SiO2": 1.0},
        1800.0,
        1e-8,
        load_gas_datapack(),
        parent_oxides=("SiO2",),
        gas_species=("SiO",),
        allow_extrapolation=False,
    )["SiO"]


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

    result = evaluate_gas(
        activities, 1800.0, 1.0e-10, gas_pack,
        composition_wt_pct=composition,
    )

    assert isinstance(result, ImccGasResult)
    assert set(result) == set(_PRE_GATED_CHANNELS)
    assert all(math.isfinite(value) and value >= 0.0 for value in result.values())
    flagged = {
        "Mg",
        "MgO",
        "Fe",
        "FeO",
        "Ti",
        "TiO",
        "TiO2",
        "K",
        "K2",
        "KO",
        "K2O",
        "AlO",
        "AlO2",
        "Al2O",
        "Al2O2",
        "Al2",
        "Al",
    }
    assert all(result.domain_flags[species] is not None for species in flagged)
    assert all(
        result.domain_flags[species] is None
        for species in set(result) - flagged
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
        "Na": 0.22579835176799098,
        "K": 2.1523835203777817,
        "SiO": 1.741267474160606e-08,
        "O2": 1.0,
    }
    assert result.domain_flags["Na"] is None
    assert result.liquid_reference_status["Na"].authority is LiquidReferenceAuthority.CORROBORATED_CALCULATION
    assert result.liquid_reference_status["Na"].band_value == pytest.approx(0.21)
    assert result.provenance_class["Na"] == "external_datapack"
    with pytest.raises(TypeError):
        result.domain_flags["Na"] = "mutated"  # type: ignore[index]
    with pytest.raises((AttributeError, TypeError)):
        result.unit = "Pa"  # type: ignore[misc]


def test_provenance_class_follows_the_coefficients_that_set_the_pressure(
    gas_pack: ImccGasDatapack,
    sf04_gas_pack: ImccGasDatapack,
) -> None:
    """A changed Gibbs coefficient must not keep the packaged class.

    Dropping an unused row leaves the selected cells packaged, so that
    channel keeps its class. A LAM1984 or LAM1987 parent ref keeps the
    class taken from the loaded row.
    """
    call = {
        "parent_oxides": ("SiO2",),
        "gas_species": ("SiO",),
        "allow_extrapolation": False,
    }
    packaged = evaluate_gas({"SiO2": 1.0}, 2000.0, 1.0e-8, gas_pack, **call)
    assert packaged["SiO"] == 0.0002081379974361049
    assert packaged.provenance_class["SiO"] == "janaf_fitted"

    edited = gas_pack.gas_df.copy()
    sio = edited.index == "SiO(g)"
    assert bool(sio.any())
    edited.loc[sio, "A"] = edited.loc[sio, "A"] + 50.0
    shifted = evaluate_gas(
        {"SiO2": 1.0},
        2000.0,
        1.0e-8,
        replace(gas_pack, gas_df=edited),
        **call,
    )
    assert shifted["SiO"] == 3.2881353908841685e-05
    assert shifted.provenance_class["SiO"] == "external_datapack"

    oxide = gas_pack.oxide_df.copy()
    silica = oxide.index == "SiO2(l)"
    assert bool(silica.any())
    oxide.loc[silica, "dG_A"] = oxide.loc[silica, "dG_A"] + 5.0
    oxide_result = evaluate_gas(
        {"SiO2": 1.0},
        2000.0,
        1.0e-8,
        replace(gas_pack, oxide_df=oxide),
        **call,
    )
    assert oxide_result["SiO"] != packaged["SiO"]
    assert oxide_result.provenance_class["SiO"] == "external_datapack"

    without_sodium = replace(
        gas_pack, gas_df=gas_pack.gas_df.drop(index=["Na(g)"])
    )
    kept = evaluate_gas(
        {"SiO2": 1.0}, 2000.0, 1.0e-8, without_sodium, **call
    )
    assert kept["SiO"] == packaged["SiO"]
    assert kept.provenance_class["SiO"] == "janaf_fitted"

    published = evaluate_gas(
        {"SiO2": 1.0, "K2O": 1.0, "Na2O": 1.0},
        2000.0,
        1.0,
        sf04_gas_pack,
        parent_oxides=("SiO2", "K2O", "Na2O"),
        gas_species=("SiO", "K", "Na"),
        allow_extrapolation=True,
    )
    assert published.provenance_class["SiO"] == "lam1987_transcribed"
    assert published.provenance_class["K"] == "lam1984_transcribed"
    assert published.provenance_class["Na"] == "external_datapack"

    ions = gas_pack.gas_df.copy()
    sodium_ion = ions.index == "Na+(g)"
    assert bool(sodium_ion.any())
    ions.loc[sodium_ion, "A"] = ions.loc[sodium_ion, "A"] + 50.0
    with_ions = evaluate_gas(
        {"Na2O": 1.0},
        2000.0,
        1.0e-8,
        replace(gas_pack, gas_df=ions),
        parent_oxides=("Na2O",),
        gas_species=("Na",),
        include_ions=True,
    )
    assert with_ions.provenance_class["Na"] == "janaf_fitted"
    assert with_ions.provenance_class["Na+"] == "external_datapack"
    assert with_ions.provenance_class["e-"] == "janaf_fitted_ionisation"


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
                **build_gas_tables._empty_source_validity_fields(),
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
    assert set(IMCC_GAS_NO_JANAF_ROWS) == {"AsO"}
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

    # Strict mode: 2200 K lies inside every Ti input row, so nothing refuses;
    # the TiO2(l) standard state remains extrapolated because O-044 is estimated.
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
        assert "TiO2(l) liquid reference is extrapolated" in (
            result.domain_flags[species] or ""
        )
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
_PS_CHANNELS = _P_CHANNELS + ("PS",) + _S_CHANNELS
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
    and species not in {"Hg", "Au", "Pd", "Pt", "PtO2", "Ru", "Rh", "Ir", "Os", "Re2O7"}
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
    "P", "P2", "P4", "PO", "PO2", "P4O6", "P4O10", "PS",
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
                    oxide_row = _oxide_row_for_T(
                        gas_pack.oxide_df, "Na2O(l)", temperature,
                        allow_extrapolation=False,
                    )
                    g_oxide = _lamor_gibbs(temperature, oxide_row)
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
    # The default D066 v5 association refit changes the basalt activities
    # supplied to this oxygen balance; the gas data and balance method are unchanged.
    assert no_extra_calls[0][0] == pytest.approx(4.044113669939106e-06, rel=1e-12)

    # The explicit oxide subset keeps excluding every other oxide activity.
    root, pressures, _ = evaluate_gas_oxygen_balance(
        activities,
        1800.0,
        gas_pack,
        parent_oxides=("SiO2", "MgO"),
    )
    assert root == pytest.approx(5.4130789404751755e-08, rel=1e-12)
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
    assert set(_P_CHANNELS) <= set(p_parent_only)
    assert all(p_parent_only[species] > 0.0 for species in _P_CHANNELS)
    assert "PS" not in p_parent_only
    assert not set(_P_CHANNELS) & set(p_parent_only.omitted_channels)

    requested_p = evaluate_gas(
        {**activities, "P2O5": 1.0e-3},
        2200.0,
        1.0e-10,
        gas_pack,
        parent_oxides=(*IMCC_PARENT_OXIDES, "P2O5"),
        gas_species=("PO",),
        allow_extrapolation=True,
    )
    assert tuple(requested_p) == ("PO",)

    with pytest.raises(ImccGasSpeciesNotFoundError) as ps_error:
        evaluate_gas(
            {**activities, "P2O5": 1.0e-3},
            2200.0,
            1.0e-10,
            gas_pack,
            parent_oxides=(*IMCC_PARENT_OXIDES, "P2O5"),
            gas_species=("PS",),
            allow_extrapolation=True,
        )
    assert "PS" in str(ps_error.value)

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


def test_published_p2o5_row_sets_phosphorus_provenance(
    gas_pack: ImccGasDatapack,
) -> None:
    result = evaluate_gas(
        {"P2O5": 1.0e-3},
        2200.0,
        1.0e-6,
        gas_pack,
        parent_oxides=(*IMCC_PARENT_OXIDES, "P2O5"),
        gas_species=_P_CHANNELS,
    )
    assert tuple(result) == _P_CHANNELS
    assert all(
        result.provenance_class[species] == "published_fitted"
        for species in _P_CHANNELS
    )
    assert species_thermo("P2O5", "l", 2200.0, gas_pack).source_table_id == (
        "Jung-Hudon-2012"
    )

    low = evaluate_gas(
        {"P2O5": 1.0e-3},
        1100.0,
        1.0e-6,
        gas_pack,
        parent_oxides=(*IMCC_PARENT_OXIDES, "P2O5"),
        gas_species=_P_CHANNELS,
        allow_extrapolation=True,
    )
    assert all("P2O5(l)" in (low.domain_flags[name] or "") for name in _P_CHANNELS)
    with pytest.raises(ImccGasTemperatureOutsideDomainError):
        evaluate_gas(
            {"P2O5": 1.0e-3},
            1100.0,
            1.0e-6,
            gas_pack,
            parent_oxides=(*IMCC_PARENT_OXIDES, "P2O5"),
            gas_species=("P",),
            allow_extrapolation=False,
        )


def test_ps_uses_both_parent_activities_and_propagates_source_domains(
    gas_pack: ImccGasDatapack,
) -> None:
    temperature = 2200.0
    fugacity = 1.0e-6
    p2o5_activity = 1.0e-3
    s2_activity = 2.5e-2
    parents = ("P2O5", "S2")
    activities = {"P2O5": p2o5_activity, "S2": s2_activity}

    p2o5_g = species_thermo("P2O5", "l", temperature, gas_pack).G_J_mol
    ps_g = species_thermo("PS", "g", temperature, gas_pack).G_J_mol
    s2_g = species_thermo("S2", "g", temperature, gas_pack).G_J_mol
    oxygen_g = species_thermo("O2", "g", temperature, gas_pack).G_J_mol
    delta_g = 2.0 * ps_g + 2.5 * oxygen_g - p2o5_g - s2_g
    expected = math.sqrt(
        math.exp(-delta_g / (R_J_MOL_K * temperature))
        * p2o5_activity
        * s2_activity
        / fugacity**2.5
    )

    result = evaluate_gas(
        activities,
        temperature,
        fugacity,
        gas_pack,
        parent_oxides=parents,
        gas_species=("PS",),
    )
    assert result["PS"] == pytest.approx(expected, rel=1.0e-12)
    assert result["PS"] > 1.0
    assert result.notices[0].code == "p_gas_exceeds_one_bar_complete_volatilisation"
    assert result.domain_flags["PS"] is not None
    assert gas_species_provenance("PS")["authority"] == "published_fitted"

    balance_metadata = oxygen_balance_species_metadata({"PS": "P2O5"})["PS"]
    assert balance_metadata.parent_oxygen_demand == pytest.approx(2.5)
    assert balance_metadata.pO2_exponent == pytest.approx(-1.25)
    _root, balance_pressures, diagnostics = evaluate_gas_oxygen_balance(
        activities,
        temperature,
        gas_pack,
        parent_oxides=parents,
    )
    assert balance_pressures["PS"] > 0.0
    assert diagnostics["residual"] < 1.0e-12

    low = evaluate_gas(
        activities,
        1100.0,
        fugacity,
        gas_pack,
        parent_oxides=parents,
        gas_species=("PS",),
        allow_extrapolation=True,
    )
    assert "P2O5(l)" in (low.domain_flags["PS"] or "")
    with pytest.raises(ImccGasTemperatureOutsideDomainError):
        evaluate_gas(
            activities,
            1100.0,
            fugacity,
            gas_pack,
            parent_oxides=parents,
            gas_species=("PS",),
            allow_extrapolation=False,
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
    with pytest.warns(ImccGasSourceOverrideWarning, match="OPENIMCC_VAPOROCK_ROOT"):
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
    activities_without_missing_channels = {
        parent: activities[parent] for parent in parents_without_missing_channels
    }
    without_parent_activity = evaluate_gas(
        activities_without_missing_channels,
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
    # The checkout may also drop unnamed columns. That notice is covered
    # elsewhere; here it must not hide the source-override warning.
    with warnings.catch_warnings():
        warnings.simplefilter("always", ImccGasIngestionWarning)
        with pytest.warns(
            ImccGasSourceOverrideWarning, match="OPENIMCC_VAPOROCK_ROOT"
        ):
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


def test_overflowing_equilibrium_constant_is_a_typed_refusal(
    gas_pack: ImccGasDatapack,
) -> None:
    with pytest.raises(ImccGasEquilibriumRangeError) as exc:
        evaluate_gas(
            {"Na2O": 1.0}, 1.0, 1.0, gas_pack, include_ions=True
        )
    assert exc.value.code == "imcc_gas_equilibrium_out_of_range"
    with pytest.raises(ImccGasEquilibriumRangeError):
        evaluate_gas({"Na2O": 1.0}, 1.0, 1.0, gas_pack)
    # A temperature inside the tabulated rows still returns a finite pressure.
    pressure = evaluate_gas(
        {"Na2O": 1.0},
        1800.0,
        1.0e-8,
        gas_pack,
        gas_species=("Na",),
        allow_extrapolation=False,
    )["Na"]
    assert math.isfinite(pressure) and pressure > 0.0


def test_unknown_activity_key_names_itself_and_omitted_parent_stays_zero(
    gas_pack: ImccGasDatapack,
) -> None:
    with pytest.raises(ImccGasUnknownActivityError, match="K20") as exc:
        evaluate_gas(
            {"SiO2": 0.5, "K20": 0.1},
            1800.0,
            1e-8,
            gas_pack,
            gas_species=("K", "SiO"),
        )
    assert exc.value.code == "imcc_gas_unknown_activity"
    with pytest.raises(ImccGasUnknownActivityError, match="NotAnOxide") as exc:
        evaluate_gas(
            {"sio2": 1.0, "NotAnOxide": 5.0},
            1800.0,
            1e-8,
            gas_pack,
            parent_oxides=("SiO2",),
            gas_species=("SiO",),
        )
    assert "sio2" in str(exc.value)
    with pytest.raises(ImccGasUnknownActivityError, match="K20"):
        evaluate_gas_oxygen_balance(
            {"SiO2": 0.5, "K20": 0.1}, 1800.0, gas_pack
        )

    omitted = evaluate_gas(
        {"SiO2": 0.5},
        1800.0,
        1e-8,
        gas_pack,
        gas_species=("K", "SiO"),
    )
    explicit_zero = evaluate_gas(
        {"SiO2": 0.5, "K2O": 0.0},
        1800.0,
        1e-8,
        gas_pack,
        gas_species=("K", "SiO"),
    )
    assert omitted["K"] == 0.0
    assert explicit_zero["K"] == omitted["K"]
    assert explicit_zero["SiO"] == omitted["SiO"]
    assert omitted["SiO"] > 0.0

    # Recognised mapping parents outside an explicit selection must be listed.
    with pytest.raises(ValueError, match=r"Na2O.*add them to parent_oxides"):
        evaluate_gas(
            {"SiO2": 1.0, "Na2O": 1.0},
            1800.0,
            1e-8,
            gas_pack,
            parent_oxides=("SiO2",),
            gas_species=("SiO",),
        )

    # S is the gas channel whose parent is S2. A melt activity map may
    # carry that species name without supplying the S2 parent.
    sulfur = evaluate_gas(
        {"K2O": 1.0, "S": 0.2},
        1800.0,
        1.0e-8,
        gas_pack,
        gas_species=("K",),
        allow_extrapolation=False,
    )
    without_sulfur = evaluate_gas(
        {"K2O": 1.0},
        1800.0,
        1.0e-8,
        gas_pack,
        gas_species=("K",),
        allow_extrapolation=False,
    )
    assert sulfur["K"] == without_sulfur["K"]


def test_mapping_parent_selection_is_explicit_but_none_auto_includes(
    gas_pack: ImccGasDatapack,
) -> None:
    activities = {"FeO": 1.0, "Cr2O3": 1.0}

    with pytest.raises(ValueError, match=r"Cr2O3.*add them to parent_oxides"):
        evaluate_gas(
            activities,
            1800.0,
            1.0,
            gas_pack,
            parent_oxides=IMCC_PARENT_OXIDES,
        )

    explicit = evaluate_gas(
        activities,
        1800.0,
        1.0,
        gas_pack,
        parent_oxides=(*IMCC_PARENT_OXIDES, "Cr2O3"),
    )
    assert {"Cr", "CrO", "CrO2", "CrO3"}.issubset(explicit)

    automatic = evaluate_gas(activities, 1800.0, 1.0, gas_pack)
    assert {"Cr", "CrO", "CrO2", "CrO3"}.issubset(automatic)


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
    with pytest.raises(ValueError, match=r"TiO2.*add them to parent_oxides"):
        evaluate_gas(
            activities, 2200.0, 1.0e-10, gas_pack, parent_oxides=seven
        )
    vector_result = evaluate_gas(
        [activities[name] for name in seven],
        2200.0,
        1.0e-10,
        gas_pack,
        parent_oxides=seven,
    )
    assert tuple(vector_result) == _SF04_CHANNELS
    assert vector_result.omitted_channels == {}
    assert dict(vector_result) == {
        species: packaged[species] for species in _SF04_CHANNELS
    }

    # The same rule covers every parent: only channels it can serve.
    silica_only = evaluate_gas(
        {"SiO2": 0.5}, 2200.0, 1.0e-10, gas_pack, parent_oxides=("SiO2",)
    )
    assert tuple(silica_only) == ("SiO", "SiO2", "O", "Si", "O2", "Si2", "Si3")

    # An explicit channel whose parent is not supplied refuses, typed.
    for species, parents in (("Ti", seven), ("Na", ("SiO2",))):
        selected_activities = {
            name: activities[name] for name in parents if name in activities
        }
        with pytest.raises(ImccGasSpeciesNotFoundError) as exc:
            evaluate_gas(
                selected_activities,
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
    assert set(result._values) == {"Ta", "TaO", "TaO2"}
    assert result._values["Ta"] == pytest.approx(1.9379680170131573e-18, rel=1.0e-12)
    assert result._values["TaO"] == pytest.approx(8.585414315417834e-11, rel=1.0e-12)
    assert result._values["TaO2"] == pytest.approx(8.51923862658225e-09, rel=1.0e-12)
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
    expected = {
        "Be": 5.4841642522907187e-11,
        "BeO": 2.839950604885199e-13,
        "Be2O": 3.603203485202284e-16,
        "(BeO)2": 2.528233210150918e-16,
        "(BeO)3": 2.497155230143581e-17,
        "(BeO)4": 6.464750106108906e-20,
        "Be2": 1.699397487178629e-25,
    }
    assert set(result._values) == set(expected)
    assert [result._values[name] for name in expected] == pytest.approx(
        list(expected.values()), rel=1.0e-12
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
    expected = {"Tl": 8.624304489890012, "Tl2O": 0.023561701919532385}
    assert set(result._values) == set(expected)
    assert [result._values[name] for name in expected] == pytest.approx(
        list(expected.values()), rel=1.0e-12
    )
    default = evaluate_gas(unit_activities, 1800.0, 1.0e-10, gas_pack)
    assert not {"Tl", "Tl2O"}.intersection(default)

    root, balanced, diagnostics = evaluate_gas_oxygen_balance(
        activities, 2000.0, gas_pack, parent_oxides=("Tl2O",)
    )
    assert root == pytest.approx(0.008771107026886694, rel=1.0e-12)
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
        "Ru(g)", "Rh(g)", "Ir(g)",
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
    trace_gases = (set(build_gas_tables.TRACE_GAS_SPECIES) - {"Ru", "Rh", "Re"}) | {
        "Mn", "Ni", "Co", "NiO", "CoO", "Ru", "Rh", "Ir"
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
        "Ru": 1.0e-3, "Rh": 1.0e-3, "Ir": 1.0e-3,
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
        parent_oxides=tuple(parents),
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
        parent_oxides=tuple(parents),
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
                    "Ru", "Rh", "Ir",
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
        "Sc+", "Ru+", "Rh+", "Ir+",
    }
    constructed_trace_ions = {
        f"{element}+"
        for element in build_gas_tables.CONSTRUCTED_SAHA_ION_ELEMENTS
        if element in trace_gases
    }
    expected_ions.update(constructed_trace_ions)
    assert expected_ions <= set(trace_ions)
    assert "In+" not in trace_ions
    assert "Sc+" in trace_ions
    nasa_ions = {"Ge+", "BO-", "Cd+", "Ag+", "Ag-", "Sc+"}
    for species in expected_ions:
        assert trace_ions[species] > 0.0
        assert trace_ions.provenance_class[species] == (
            "nist_partition_fitted_ionisation"
            if species == "Bi+" or species in constructed_trace_ions
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
        "Cs+", "Cs-", "Sn+", "Cs2O+",
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
    """Keep untouched rows exact while extending only their lower bounds."""
    base = "ccabd0b"
    rebuilt_lfs = {"MnO(l)", "As2O3(l)", "Sb2O3(l)", "Ag2O(l)"}
    lowered_gas_rows = {"H2O(g)", "Zn(g)", "Cd(g)", "Pb(g)", "PbO(g)", "PbO2(g)"}
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
        old_columns = old_lines[0].split(",")
        assert current_lines[0].split(",")[:len(old_columns)] == old_columns
        current_records = list(csv.DictReader(current_lines))
        unchanged = []
        for old_row in csv.DictReader(old_lines):
            fields = [old_row[column] for column in old_columns]
            species = old_row["species_name"]
            if species in rebuilt_lfs | {"FeO(l)", "Na2O(l)", "SiO2(l)"}:
                continue
            if relpath.endswith("gas-shomate.csv") and species in lowered_gas_rows and fields[2] == "2":
                continue
            if relpath.endswith("condensate.csv") and species == "Na2O(l)" and fields[5] == "1200":
                continue
            low = float(old_row["T_min"])
            high = float(old_row["T_max"])
            if any(
                other["species_name"] == species
                and other["state"] == old_row["state"]
                and other["Ref"] == old_row["Ref"]
                and (
                    abs(float(other["T_max"]) - low) <= 1.0e-6
                    or abs(float(other["T_min"]) - high) <= 1.0e-6
                )
                for other in current_records
            ):
                continue
            unchanged.append(tuple(fields[:len(old_columns)]))
        current_rows = [
            tuple(row[column] for column in old_columns)
            for row in current_records
        ]
        assert Counter(unchanged) <= Counter(current_rows)


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


def test_teo_matches_nureg_table_30_and_anchored_mass_action(
    gas_pack: ImccGasDatapack,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    source_path = repository / "data-src/published-thermo/nureg-cr-4308-teo.json"
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes)
    provenance = yaml.safe_load(
        (repository / "src/openimcc/data/gas/PROVENANCE.yaml").read_text(
            encoding="utf-8"
        )
    )
    rows = [
        row
        for row in provenance["rows"]
        if row.get("table_id") == "NUREG-4308-TeO"
    ]
    assert len(rows) == 2
    assert all(
        hashlib.sha256(source_bytes).hexdigest() == row["source_sha256"]
        for row in rows
    )
    anchor_correction = float(rows[0]["anchor_correction"]["value_kJ_mol"]) * 1000.0
    inferred_hf = float(
        rows[0]["anchor_correction"]["inferred_nureg_Hf298_kJ_mol"]
    )
    selected_hf = float(rows[0]["anchor_correction"]["selected_Hf298_kJ_mol"])

    dutton_path = repository / "data-src/published-thermo/dutton-cooper-1966-teo.json"
    dutton_bytes = dutton_path.read_bytes()
    dutton = json.loads(dutton_bytes)
    published_provenance = yaml.safe_load(
        (repository / "data-src/published-thermo/PROVENANCE.yaml").read_text(
            encoding="utf-8"
        )
    )
    dutton_metadata = next(
        record
        for record in published_provenance["records"]
        if record["record_id"] == "Dutton-Cooper-1966-TeO"
    )
    assert dutton_metadata["path"] == "data-src/published-thermo/dutton-cooper-1966-teo.json"
    assert dutton_metadata["source_sha256"] == hashlib.sha256(dutton_bytes).hexdigest()
    assert dutton_metadata["source_limit_K"] == 2000.0

    cp = dutton["heat_capacity_cal_mol_K"]
    a = float(cp["a"])
    b = float(cp["b_per_K"])
    c = float(cp["c_K2"])
    reference_temperature = float(dutton["source_reference_temperature_K"])
    source_limit = float(cp["source_limit_K"])

    def dutton_increments(temperature: float) -> tuple[float, float]:
        # Integrate the printed Cp = a + b*T + c/T^2 correlation from 298.16 K.
        h_increment = (
            a * (temperature - reference_temperature)
            + 0.5 * b * (temperature**2 - reference_temperature**2)
            - c * (1.0 / temperature - 1.0 / reference_temperature)
        )
        s_increment = (
            a * math.log(temperature / reference_temperature)
            + b * (temperature - reference_temperature)
            - 0.5 * c * (1.0 / temperature**2 - 1.0 / reference_temperature**2)
        )
        return h_increment, s_increment

    # Premise: Dutton and Cooper report H-H298 in kcal/mol and S-S298 in
    # cal/(mol K), with their Cp equation accurate through 2000 K. Algebra:
    # integrate Cp and compare against each printed source node. Units convert
    # kcal to cal only for the H-node closure. Sanity: the largest printed-node
    # closure residual is below 0.05 kcal/mol and 0.05 cal/(mol K).
    for temperature, h_node_kcal, s_node in dutton["nodes"]:
        h_increment, s_increment = dutton_increments(float(temperature))
        assert h_increment / 1000.0 == pytest.approx(float(h_node_kcal), abs=0.05)
        assert s_increment == pytest.approx(float(s_node), abs=0.05)

    h_at_limit, s_at_limit = dutton_increments(source_limit)
    cp_at_limit = a + b * source_limit + c / source_limit**2

    def dutton_gibbs(temperature: float) -> float:
        h_increment, s_increment = dutton_increments(
            min(temperature, source_limit)
        )
        if temperature > source_limit:
            # Dutton does not validate Cp beyond 2000 K. Hold only the
            # comparison Cp at its source-limit value and integrate H/S.
            h_increment = h_at_limit + cp_at_limit * (temperature - source_limit)
            s_increment = s_at_limit + cp_at_limit * math.log(
                temperature / source_limit
            )
        return (
            float(dutton["formation_enthalpy_kcal_mol"]["value"]) * 1000.0
            + h_increment
        ) * 4.184 - temperature * (
            float(dutton["entropy_cal_mol_K"]["value"]) + s_increment
        ) * 4.184

    assert anchor_correction == pytest.approx(abs(selected_hf - inferred_hf) * 1000.0)
    for row in rows:
        low, high = map(int, row["T_range_K"])
        knots = set(range(low, high + 1, 100))
        anchor_values = {
            int(temperature): float(value)
            for temperature, value in row["accuracy_band"]["value"][
                "values_J_per_mol"
            ].items()
        }
        assert set(anchor_values) == knots
        assert all(value == pytest.approx(7400.0) for value in anchor_values.values())
        mixed_bands = row["accuracy_band"]["value"]["additional_uncertainty_bands"]
        assert len(mixed_bands) == 1
        band = mixed_bands[0]
        assert band["kind"] == "mixed_source_parent_scale_inconsistency"
        assert band["comparison_record"] == dutton_metadata["record_id"]
        assert band["comparison_source_limit_K"] == source_limit
        extrapolated = sorted(
            temperature for temperature in knots if temperature > source_limit
        )
        assert band["comparison_extrapolated_temperatures_K"] == extrapolated
        values = {
            int(temperature): float(value)
            for temperature, value in band["values_J_per_mol"].items()
        }
        assert set(values) == knots
        assert band["comparison_extrapolation"]["source_data"] is False

        # Compare the two complete parent reactions on the same P672 TeO2(l)
        # and JANAF O2 references. Those terms cancel, leaving the source-route
        # difference between the selected NUREG and alternate Dutton TeO data.
        for temperature in knots:
            t = float(temperature)
            teo_row = _nearest_interval_row(gas_pack.gas_df, "TeO(g)", t)
            oxygen_row = _nearest_interval_row(gas_pack.gas_df, "O2(g)", t)
            parent_row = _oxide_row_for_T(gas_pack.oxide_df, "TeO2(l)", t)
            selected_reaction_g = (
                _janaf_gibbs(t, teo_row)
                + 0.5 * _janaf_gibbs(t, oxygen_row)
                - _lamor_gibbs(t, parent_row)
            )
            alternate_reaction_g = (
                dutton_gibbs(t)
                + 0.5 * _janaf_gibbs(t, oxygen_row)
                - _lamor_gibbs(t, parent_row)
            )
            assert values[temperature] == pytest.approx(
                abs(selected_reaction_g - alternate_reaction_g), abs=0.001
            )

    # Premise: Table 30's difference is the Te(g)+1/2 O2(g)=TeO(g)
    # reaction Gibbs function, while the later measured enthalpy anchor shifts
    # that function uniformly. Algebra: add the reaction Gibbs to the active
    # Te/O2 reference rows; compare each printed node to TeO's fitted G_app.
    # Unit check: source cal/mol becomes J/mol with 4.184; the anchor is kJ/mol.
    # Sanity: all 19 source nodes close within the recorded 10 J/mol fit gate.
    for temperature, teo_delta_g_cal, te_delta_g_cal in source["nodes"]:
        teo_row = _nearest_interval_row(gas_pack.gas_df, "TeO(g)", temperature)
        te_row = _nearest_interval_row(gas_pack.gas_df, "Te(g)", temperature)
        oxygen_row = _nearest_interval_row(gas_pack.gas_df, "O2(g)", temperature)
        expected_gibbs = (
            _janaf_gibbs(temperature, te_row)
            + 0.5 * _janaf_gibbs(temperature, oxygen_row)
            + (teo_delta_g_cal - te_delta_g_cal) * 4.184
            + anchor_correction
        )
        assert _janaf_gibbs(temperature, teo_row) == pytest.approx(
            expected_gibbs, abs=10.0
        )

    # The fit's F coefficient carries the measured 298.15 K enthalpy anchor.
    high_row = _nearest_interval_row(gas_pack.gas_df, "TeO(g)", 2000.0)
    t298 = 298.15 / 1000.0
    h298 = (
        float(high_row["A"]) * t298
        + float(high_row["B"]) * t298**2 / 2.0
        + float(high_row["C"]) * t298**3 / 3.0
        + float(high_row["D"]) * t298**4 / 4.0
        - float(high_row["E"]) / t298
        + float(high_row["F"])
        - float(high_row["H"])
    )
    assert h298 == pytest.approx(
        source["anchor"]["formation_enthalpy_kj_mol"], abs=1e-10
    )

    # Premise: TeO2(l)=TeO(g)+1/2 O2(g) is TeO's active parent reaction.
    # Algebra: compare the direct runtime result to Kp from the same active
    # species Gibbs values at multiple source-grid nodes.
    for temperature in (1200.0, 1500.0, 2000.0, 2500.0, 3000.0):
        product_row = _nearest_interval_row(
            gas_pack.gas_df, "TeO(g)", temperature
        )
        oxygen_row = _nearest_interval_row(gas_pack.gas_df, "O2(g)", temperature)
        parent_row = _oxide_row_for_T(
            gas_pack.oxide_df, "TeO2(l)", temperature
        )
        delta_g = (
            _janaf_gibbs(temperature, product_row)
            + 0.5 * _janaf_gibbs(temperature, oxygen_row)
            - _lamor_gibbs(temperature, parent_row)
        )
        expected_log10_pressure = -delta_g / (
            R_J_MOL_K * temperature * math.log(10.0)
        )
        result = evaluate_gas(
            {"TeO2": 1.0},
            temperature,
            1.0,
            gas_pack,
            parent_oxides=("TeO2",),
            gas_species=("TeO",),
            allow_extrapolation=False,
        )
        assert math.log10(result["TeO"]) == pytest.approx(
            expected_log10_pressure, abs=1e-12
        )


def test_sbo_matches_kelley_nbs_sohn_and_records_nureg_disagreement(
    gas_pack: ImccGasDatapack,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    source_path = repository / "data-src/published-thermo/kelley-1960-sbo.json"
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes)
    published_provenance = yaml.safe_load(
        (repository / "data-src/published-thermo/PROVENANCE.yaml").read_text(
            encoding="utf-8"
        )
    )
    source_record = next(
        record
        for record in published_provenance["records"]
        if record["record_id"] == "Kelley-584-SbO"
    )
    assert source_record["source_sha256"] == hashlib.sha256(source_bytes).hexdigest()

    gas_provenance = yaml.safe_load(
        (repository / "src/openimcc/data/gas/PROVENANCE.yaml").read_text(
            encoding="utf-8"
        )
    )
    rows = [
        row
        for row in gas_provenance["rows"]
        if row.get("table_id") == "Kelley-584-SbO"
    ]
    assert len(rows) == 2
    assert all(
        row["source_sha256"] == hashlib.sha256(source_bytes).hexdigest()
        for row in rows
    )
    low = next(row for row in rows if row["T_interval"] == 2)
    high = next(row for row in rows if row["T_interval"] == 1)
    assert low["T_range_K"] == [1200, 2000]
    assert high["T_range_K"] == [2000, 3000]
    assert high["extrapolation"] is True
    assert high["extrapolation_source_limit_K"] == 2000
    assert "constant-Cp continuation above 2000 K" in high["flags"]

    def h_increment(row: pd.Series, temperature: float) -> float:
        value = temperature / 1000.0
        reference = 298.15 / 1000.0
        return 1000.0 * (
            float(row["A"]) * (value - reference)
            + float(row["B"]) * (value**2 - reference**2) / 2.0
            + float(row["C"]) * (value**3 - reference**3) / 3.0
            + float(row["D"]) * (value**4 - reference**4) / 4.0
            - float(row["E"]) * (1.0 / value - 1.0 / reference)
        )

    def s_increment(row: pd.Series, temperature: float) -> float:
        value = temperature / 1000.0
        reference = 298.15 / 1000.0
        return (
            float(row["A"]) * math.log(value / reference)
            + float(row["B"]) * (value - reference)
            + float(row["C"]) * (value**2 - reference**2) / 2.0
            + float(row["D"]) * (value**3 - reference**3) / 3.0
            - float(row["E"]) * (1.0 / value**2 - 1.0 / reference**2) / 2.0
        )

    def absolute_h(row: pd.Series, temperature: float) -> float:
        value = temperature / 1000.0
        return (
            float(row["A"]) * value
            + float(row["B"]) * value**2 / 2.0
            + float(row["C"]) * value**3 / 3.0
            + float(row["D"]) * value**4 / 4.0
            - float(row["E"]) / value
            + float(row["F"])
        )

    def absolute_s(row: pd.Series, temperature: float) -> float:
        value = temperature / 1000.0
        return (
            float(row["A"]) * math.log(value)
            + float(row["B"]) * value
            + float(row["C"]) * value**2 / 2.0
            + float(row["D"]) * value**3 / 3.0
            - float(row["E"]) / (2.0 * value**2)
            + float(row["G"])
        )

    low_runtime = _nearest_interval_row(gas_pack.gas_df, "SbO(g)", 1500.0)
    high_runtime = _nearest_interval_row(gas_pack.gas_df, "SbO(g)", 2500.0)
    anchor = source["formation_enthalpy_anchor"]
    assert h_increment(low_runtime, 298.15) == pytest.approx(0.0, abs=1e-12)
    assert (
        float(low_runtime["A"]) * (298.15 / 1000.0)
        + float(low_runtime["B"]) * (298.15 / 1000.0) ** 2 / 2.0
        + float(low_runtime["C"]) * (298.15 / 1000.0) ** 3 / 3.0
        + float(low_runtime["D"]) * (298.15 / 1000.0) ** 4 / 4.0
        - float(low_runtime["E"]) / (298.15 / 1000.0)
        + float(low_runtime["F"])
    ) == pytest.approx(anchor["formation_enthalpy_kj_mol"], abs=1e-10)

    thermal_h_residuals = []
    thermal_s_residuals = []
    for temperature, source_h_cal, source_s_cal in source["thermal_source"]["nodes"]:
        row = low_runtime
        thermal_h_residuals.append(
            h_increment(row, float(temperature)) - float(source_h_cal) * 4.184
        )
        thermal_s_residuals.append(
            s_increment(row, float(temperature)) - float(source_s_cal) * 4.184
        )
    thermal_fit = low["thermal_increment_fit"]
    assert max(map(abs, thermal_h_residuals)) == pytest.approx(
        thermal_fit["max_enthalpy_increment_residual_J_per_mol"], abs=1e-8
    )
    assert max(map(abs, thermal_s_residuals)) == pytest.approx(
        thermal_fit["max_entropy_increment_residual_J_per_mol_K"], abs=1e-8
    )

    t298 = 298.15 / 1000.0
    s298 = (
        float(low_runtime["A"]) * math.log(t298)
        + float(low_runtime["B"]) * t298
        + float(low_runtime["C"]) * t298**2 / 2.0
        + float(low_runtime["D"]) * t298**3 / 3.0
        - float(low_runtime["E"]) / (2.0 * t298**2)
        + float(low_runtime["G"])
    )
    assert s298 == pytest.approx(
        low["inferred_formation_entropy_J_per_mol_K"], abs=1e-10
    )

    reaction = source["reaction_source"]["gibbs_function"]
    nureg_nodes = {
        int(node[0]): tuple(float(value) for value in node[1:])
        for node in source["comparison_source"]["nodes"]
    }
    source_gibbs_residuals = []
    for temperature in range(1200, 2001, 100):
        sb_g, sb_l, _ = nureg_nodes[temperature]
        selected_source_delta_g = (
            float(reaction["constant_J_mol"])
            + float(reaction["slope_J_mol_K"]) * temperature
        ) - (sb_g - sb_l) * 4.184
        expected_gibbs = (
            _janaf_gibbs(temperature, _nearest_interval_row(gas_pack.gas_df, "Sb(g)", temperature))
            + 0.5
            * _janaf_gibbs(
                temperature, _nearest_interval_row(gas_pack.gas_df, "O2(g)", temperature)
            )
            + selected_source_delta_g
        )
        actual_gibbs = _janaf_gibbs(
            temperature, _nearest_interval_row(gas_pack.gas_df, "SbO(g)", temperature)
        )
        source_gibbs_residuals.append(actual_gibbs - expected_gibbs)
    assert max(map(abs, source_gibbs_residuals)) == pytest.approx(
        low["fit_residual"]["value"]["max_residual_J_per_mol"], abs=1e-7
    )

    low_band = low["accuracy_band"]["value"]
    high_band = high["accuracy_band"]["value"]
    assert low_band["comparison_record"] == "NUREG-4308-SbO"
    assert high_band["comparison_record"] == "NUREG-4308-SbO"
    recorded_differences = {
        int(temperature): float(value)
        for band in (low_band, high_band)
        for temperature, value in band["values_J_per_mol"].items()
    }
    assert set(recorded_differences) == set(nureg_nodes)
    for temperature, (sb_g, sb_l, sbo_g) in nureg_nodes.items():
        selected_delta_f_g = (
            _janaf_gibbs(
                temperature, _nearest_interval_row(gas_pack.gas_df, "SbO(g)", temperature)
            )
            - _janaf_gibbs(
                temperature, _nearest_interval_row(gas_pack.gas_df, "Sb(g)", temperature)
            )
            - 0.5
            * _janaf_gibbs(
                temperature, _nearest_interval_row(gas_pack.gas_df, "O2(g)", temperature)
            )
            + (sb_g - sb_l) * 4.184
        )
        assert recorded_differences[temperature] == pytest.approx(
            abs(selected_delta_f_g - (sbo_g - sb_l) * 4.184), abs=0.001
        )

    low_cp_at_limit = (
        float(low_runtime["A"])
        + float(low_runtime["B"]) * 2.0
        + float(low_runtime["C"]) * 2.0**2
        + float(low_runtime["D"]) * 2.0**3
        + float(low_runtime["E"]) / 2.0**2
    )
    assert float(high_runtime["A"]) == pytest.approx(low_cp_at_limit, abs=1e-12)
    assert absolute_h(high_runtime, 2000.0) == pytest.approx(
        absolute_h(low_runtime, 2000.0), abs=1e-8
    )
    assert absolute_s(high_runtime, 2000.0) == pytest.approx(
        absolute_s(low_runtime, 2000.0), abs=1e-8
    )
    for temperature in (1200.0, 1500.0, 2000.0, 2500.0, 3000.0):
        parent_row = _oxide_row_for_T(gas_pack.oxide_df, "Sb2O3(l)", temperature)
        product_row = _nearest_interval_row(gas_pack.gas_df, "SbO(g)", temperature)
        oxygen_row = _nearest_interval_row(gas_pack.gas_df, "O2(g)", temperature)
        delta_g = (
            2.0 * _janaf_gibbs(temperature, product_row)
            + 0.5 * _janaf_gibbs(temperature, oxygen_row)
            - _lamor_gibbs(temperature, parent_row)
        )
        expected_log10_pressure = -delta_g / (
            2.0 * R_J_MOL_K * temperature * math.log(10.0)
        )
        result = evaluate_gas(
            {"Sb2O3": 1.0},
            temperature,
            1.0,
            gas_pack,
            parent_oxides=("Sb2O3",),
            gas_species=("SbO",),
        )
        assert math.log10(result["SbO"]) == pytest.approx(
            expected_log10_pressure, abs=1e-12
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


def test_oxygen_balance_records_whether_the_total_pressure_is_knudsen(
    gas_pack: ImccGasDatapack,
) -> None:
    activities = {
        "SiO2": 0.45,
        "MgO": 0.10,
        "FeO": 0.15,
        "CaO": 0.15,
        "Al2O3": 0.15,
    }
    low_p, low_gas, low_diag = evaluate_gas_oxygen_balance(
        activities, 1800.0, gas_pack, allow_extrapolation=False
    )
    high_p, high_gas, high_diag = evaluate_gas_oxygen_balance(
        activities, 3000.0, gas_pack, allow_extrapolation=False
    )
    low_total = math.fsum(float(value) for value in low_gas.values())
    high_total = math.fsum(float(value) for value in high_gas.values())
    assert low_diag["mode"] == "oxygen_balance_effusion"
    assert high_diag["mode"] == "oxygen_balance_effusion"
    assert low_diag["total_pressure_bar"] == low_total
    assert high_diag["total_pressure_bar"] == high_total
    assert low_total <= 1e-3
    assert high_total > 1e-3
    assert low_diag["knudsen_regime"] is True
    assert high_diag["knudsen_regime"] is False
    assert high_p > low_p > 0.0


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
    assert diagnostics["root_solver"]["path"] == "direct"
    assert diagnostics["root_solver"]["refinement_evaluations"] == 0


def _generic_oxygen_balance_relative_residual(
    pressures: Mapping[str, float],
    species: Mapping[str, OxygenBalanceSpeciesMetadata],
) -> float:
    oxygen_flux = math.fsum(
        data.oxygen_atoms * pressures[name] / math.sqrt(data.molar_mass)
        for name, data in species.items()
    )
    parent_flux = math.fsum(
        data.parent_oxygen_demand * pressures[name] / math.sqrt(data.molar_mass)
        for name, data in species.items()
    )
    return abs(oxygen_flux - parent_flux) / max(oxygen_flux, parent_flux, 1e-300)


def test_generic_oxygen_balance_solves_sine_on_k_pressure_model() -> None:
    metadata = oxygen_balance_species_metadata(
        {"K": "K2O", "O2": None}, pO2_exponents={"K": 0.0}
    )
    power_law_root = math.log10(
        0.25 * 1e-8
        * math.sqrt(metadata["O2"].molar_mass / metadata["K"].molar_mass)
    )

    def pressure_model(logp: float) -> dict[str, float]:
        return {
            "K": 1e-8 * (
                1.0 + 0.8 * math.sin(math.pi * (logp - power_law_root) / 0.1)
            ),
            "O2": 10.0**logp,
        }

    p_o2, pressures, diagnostics = oxygen_balance_from_pressure_model(
        pressure_model, metadata
    )
    assert abs(math.log10(p_o2) - power_law_root) > 0.01
    assert _generic_oxygen_balance_relative_residual(pressures, metadata) <= (
        diagnostics["root_solver"]["residual_tolerance"]
    )


def test_generic_oxygen_balance_accepts_monotone_piecewise_pressure_model() -> None:
    metadata = oxygen_balance_species_metadata(
        {"K": "K2O", "O2": None, "MgO": "MgO"},
        pO2_exponents={"K": 0.0, "MgO": -0.5},
    )
    power_law_root = math.log10(
        0.25 * 1e-8
        * math.sqrt(metadata["O2"].molar_mass / metadata["K"].molar_mass)
    )

    def pressure_model(logp: float) -> dict[str, float]:
        offset = logp - power_law_root
        inner = max(-0.05, min(0.05, offset))
        outer = offset - inner
        return {
            "K": 1e-8,
            "O2": 10.0**logp,
            "MgO": 10.0 ** (-30.0 - 0.9 * inner - 0.1 * outer),
        }

    _p_o2, pressures, diagnostics = oxygen_balance_from_pressure_model(
        pressure_model, metadata
    )
    assert _generic_oxygen_balance_relative_residual(pressures, metadata) <= (
        diagnostics["root_solver"]["residual_tolerance"]
    )


@pytest.mark.parametrize("spacing", [0.2, 0.1, 0.05])
def test_generic_oxygen_balance_accepts_linear_in_pressure_table(
    spacing: float,
) -> None:
    metadata = oxygen_balance_species_metadata(
        {"K": "K2O", "O2": None, "MgO": "MgO"},
        pO2_exponents={"K": 0.0, "MgO": -0.5},
    )
    power_law_root = math.log10(
        0.25 * 1e-8
        * math.sqrt(metadata["O2"].molar_mass / metadata["K"].molar_mass)
    )
    # The spectator follows a true -0.5 law at each knot; interpolation is
    # linear in p, so the chord's two endpoints can miss its interior bend.
    logp_nodes = np.arange(-30.0, 0.0001, spacing)
    pressure_nodes = 10.0**logp_nodes
    mgo_pressures = 1e-30 * 10.0 ** (-0.5 * (logp_nodes - power_law_root))

    def pressure_model(logp: float) -> dict[str, float]:
        return {
            "K": 1e-8,
            "O2": 10.0**logp,
            "MgO": float(np.interp(10.0**logp, pressure_nodes, mgo_pressures)),
        }

    p_o2, pressures, diagnostics = oxygen_balance_from_pressure_model(
        pressure_model, metadata
    )
    assert math.log10(p_o2) == pytest.approx(power_law_root, abs=1e-12)
    assert _generic_oxygen_balance_relative_residual(pressures, metadata) <= (
        diagnostics["root_solver"]["residual_tolerance"]
    )


def test_generic_oxygen_balance_reports_refinement_cost_for_curved_model() -> None:
    metadata = oxygen_balance_species_metadata(
        {"K": "K2O", "O2": None}, pO2_exponents={"K": 0.0}
    )
    power_law_root = math.log10(
        0.25 * 1e-8
        * math.sqrt(metadata["O2"].molar_mass / metadata["K"].molar_mass)
    )
    model_evaluations = 0

    def pressure_model(logp: float) -> dict[str, float]:
        nonlocal model_evaluations
        model_evaluations += 1
        offset = max(-0.01, min(0.01, logp - power_law_root))
        return {"K": 1e-8 * 10.0 ** (-1000.0 * offset), "O2": 10.0**logp}

    _p_o2, pressures, diagnostics = oxygen_balance_from_pressure_model(
        pressure_model, metadata
    )
    solver_diagnostics = diagnostics["root_solver"]
    assert solver_diagnostics["path"] == "refined"
    assert solver_diagnostics["refinement_evaluations"] > 0
    assert solver_diagnostics["model_evaluations"] == model_evaluations
    assert _generic_oxygen_balance_relative_residual(pressures, metadata) <= (
        solver_diagnostics["residual_tolerance"]
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
# Closed gas and condensate seams move equilibrium oxygen-balance results.
_OXYGEN_BALANCE_SF04_NUMERIC_HEX_SHA256 = ('84568009b644523fb3c2df932903f027b740964c06b46b6012b33905a1bc7593', 'b0e65cb285efddc766d272d93463360bf973c5eb7d707dd58845d943888dcb06', '8a5fcbd6205233c4bf4158ca1d3a16f1e6598fa62f35661377dec3706719fd33', 'fe8c2ef1b13106bde221b1e5eb90f51ecc8e651e840cedc4788c43d2336babf8', 'c047f1e38fde0ec3b80c109b3a00e4080170fde7549750e319de735b6743c26a', '4cead1131184147a21d5e0933fad77e672d1a596b92576a80e88e8b2e839e247', '4c6b8fdc1b57a22aaec3968d33511b478f272b484f1513fd87b41e89683575b5', 'f2acdfa62cae74539be4a4b29eeafb30b0bb23e8a648d311f7dcbb6ca71a12d3', '5a52f149eec431266f1c2f5ec618014ac4dc3a24be1a4ec8657035d0027a72b1', '275be352891012572c866619868d7078a95c48f508e99f84f5abbcbfe6cdbb17', '5f283d0c97d5b70dabb8a056a8fe079ce9fe82a2fa6ced9125d7eff96f3d7c6c', 'd7e854c91a574cb443241b9f35644f8dcb2618baacf9187547c8879976a622ae', '0fe793a84527f91eb70678f2d92ffc64e540a63ba89676074480c0f3053f21b9', '77aa95ecd1326e5c64afb0fd067b216afca9b41cf0c55e9f44b4e3ea90da19ac', '5042f11064b8d32253b1f918d2c77d8c9656fa6d64c1e78a5e825c26c6946c7a', 'e4c61dd624cd0e685d68794076da4c16562b4e9f589dd83ad452de0c4dda3a2c', '0be9e66e4d6cf5e902bc28a62210ed68e55a1e0f99fff79e880e63fb306da08c', 'cc7642adfa85090efb7f89db633e2ec51ef0e205242b2ea3454070adefae61e2', '656c5321205fcc1e4fb5a8c2e9ff65be8db99b1f479f3cf8b4f4d4358a01d261', 'f457245fa21ab5f789d4d11bcb2ed8c87d9a38604b71fbe15bf39211dc1df6d9', '8581fc489703d30f9520a12cc824a69f65479f14933c3c779efcef3d9d528364', '867c923135a4020b22f9da9819b5833d78290f65113c456faf2414bb2eba6b41', 'da6c3083bb88df1e566086a76c7229c86c84749ba37095e96f52df8d18fd57a7', '9ecf81bd0fe7853554f38e91d14c65d363e4dc8a48d850e760b7533d739b24f7', '5f2d6461764057853883cd947840e82e1062dc6f2a6032146c8207fff1236db0', 'c6dd85d18c20b66cf1559e3bfa94bb3621023c240d2c65f2fe61b576add67917', '3586e9e8608fcb280a0abed2d9c849f58c6aac388b0829e0f67385c4a7390737', '1d71c9e73d1d701352922da8471260db82610ddd805e647f5aa6d8deafb5b2bf', '8f3bd87f822f210fc05334ec2250088e3279bef874f99372340bb4f8a9fc3324', '614988cad0e2f5faeda39418f9330c5a0649fbf41208b6ca45ed49752d937160', '380143af6a95a0b8b14d9f73c3aa66e6345e7f6574d4e5f3dddb6ab23eac6ffc', '30be849a7445476de25ea6ddc302f6012e8f45a3618a44b88ff4d52919f4cea4', '402fc8477fdc133abea74c6ece5a0383035a9893d4bcf62fb8b778b762e679de', 'e7ad6388b91d9def6e4452dafde537108beccece12a32546eb2e9fad760f3f16', 'ac9a59abad9eff0b7b7f074bf87b1057bde623aa1d1f642fa50456ac48d07dd3', '8830a4e10ae6120a12fa7bbc2e67ff682db81264c7a0d6f88d20dcc60a0cf142', '393e861930abd571659e83e913d6052bd1445ed5f371ee718c2df382d3009f79', '51e6b121c8a5216742123a8afd03565b81c4b562178719abbbeb33753e975463', '9e55c9d2a1356ec26bc3a745633e8a5cd2f8538fcbb9771c4876e550600ebac9', '4df0ea79b65cb1e0864cd1ef635808c9c1cd0d3f40a5de381b3a0e761e85b410', '03f88c7c2bb315139eea080b524e779ef15f66b59eece40d7ce6398bfbf4f5b6', '3ea4439161b57c9253a571d8481e241a4581734b2bf0c381e29f49d2029c46f5', '8067e14492e3778ffcde526c9d2288e3747846a8d8ac6de7e466a6148e03e7f2', 'ec9b6a23dcc29333f184757211713848d366dab3aec9deed0118ff8d400703e6', '085e834cd0287ea5fe2dc1e2b2a9b638c6f48b84edc39d8c078568d29df385f8', '9bdce17b2cfed26865a29764946c04bb8a6c3543217133e3365712a52e5b86d2', '0832be76830f2169fa301e8c27e6fa3d16f573e71fb718cc81c33785cec00982', '2c96f6e1a1bdd5f47805f8042724327860d8f40aa72f2220e7ea035ff3e813bc', '461c4d52ad2cffc6b4a5c90b30a2a10794f322fe2ddca60fae98cb08255adeb3', '313b0375e195d22cfaeb8e3cab2109aa60b7ee73268dccb17a07ab26a094a32b')
_OXYGEN_BALANCE_SF04_RESULT_HEX_SHA256 = ('5e0f66865371e04ec121959dfd854705e98524b4116ce5b902f8fd3befe0c232', 'ad3e519e4067e25d8dc53afcddec1e6034feed2c47c47eb2982b2eb86203bfbb', '6768da82c75efd2ba8380b22883680016c3e3e3d200e610a8fbe5618618dcd87', 'ee399785fac0e2efad55b818a4fcc5551191b1706b1a7412684ad6f6b7030077', '394548d20994d438abfec7b59c240936a5fa6088aeaa48357aef93daee0f900e', '1920b4c044e6464247773bab3261b371b2c4e80ef63f61bb97d7038cf31915dd', '6f5d209673ddfb653bcb01c31eddca7a8e5f2ce46877bc7e44bc54c89f6595ee', 'e4c2706d54b7e803d50b254cbfecb622f9babb00b30cbb4b49f744379417503f', '3ae5e3f03cc2d18ad04a4cb4fe2aec9f0ef705883cfc6bbbce3536c60b344b1c', '27c9016810bab799bc556b8f11b01b50500332e42c4ef144f1d654aba12614e9', 'b233639221a38a3b136ef390ab95214a907e6330f0b84cfa23ba00c3c05903df', '8982a0033a958d38bb29dd0a9c52748c1f0dd7e6d9a70b093898646c4446c75f', '8c930868da5464193d5582fa5a7a01662b1536efad3d535c666cd7d4641171bb', 'b234ced0832f64bc584470b580205b2b4f55a9043917127c2df300c7f5678223', 'b4db2a2a63d3c6162125cd9210c1dc1cb65094977db495decd6fcdef4ad9baec', 'd138454d8d5260dbbdaa2e823d734e3768e28e5d1d7cd29507877ac47f0270f9', '91dd70e648c054bdb34e01e3a0bd2edeeac72c465711a33362161184d7ffeba7', 'd91b38affd33fb9bfa0407edb12af329183e1389f969a7736017ffe716d8fe2a', '789fb7c349c476c06e8d41fbb85aa07ab2a82794e14d32e4ec7662aa4124c462', '4f081c8cd716db0827c3432256b8cea73dad90050dd6a30f228cb34dc8628491', '4c153b380110ca02650a7e37fe7411ccf3765cde9c6fc7c794270d24bc370232', '8be78c40eeb77c07307920bd8d74efff5d8482628dc213578b80f6c18f1510dd', '88b74cf7b8a7ade2a4a05b932a147f7a69198295a548a574766e4217f50328b4', '4b44695903f52d814aab913b72ce0414c897f1030d97a04e83eb581e68531742', '091ab932af955e4e2fe9334152f557b5cd692b40bc1131b8c12c63f43750403c', '6e1711d5152f54967101aca1e59e2e951f0934dc258ac9a5f015a35e2d6e4342', '855e617ead7d3b13d9c73f32dfa5dcd93375920f6e317a5b0465dc6b0d555291', '32b28a480bf8a9eb96dd438a84753129052eaf88d44dab2ee7ff835ee45f461d', '08b099b067f54c8630c6267d2096f5a78e4471090dba8f5c53d328b530f818e8', 'bd20726a34279138e461c7f06875c0052a99a4e189174c048630cf861d2c254b', '2018a283cd335bbe4d4de80408fe40a09f2a053cfce69d0bc14801502bf95173', '56134cd8915599dd703c16af552409fb6bebcc6effce59cd245ce3257f5574f2', '6460c1ce2abedca0d09ee1c23832f1567f205b991b8d0b0e85d4395a6a7f091d', '90a66808e82083a36fc2fcd971983bcd8710c68c80256d2bba0d8c24731725a9', '0e177cd10113d9c90e67ab32cf61a0bb15cdf78fe56ea720055544a8c4a9e91e', '1485fc84efdbb0f0432c04f49ef47cbc0c67096bf7f2af3794a9d8efc9ec1433', '866d6db120d46cf7e2b3608875d9232199ec2ce9c0f8a7f4f70f524d58ae99c0', 'eb4a617ae7fbec63b4f0d6fa409894d1979e05547805cce616ea3a1a845b72d8', 'ce99ab71793031746fad805b000a42f70d73cdc8f47b89857c2b35c996114618', '559bac6a8853fdf84b9d428aa73e6d17343523dfce8baedf5f632c5527cfc7e9', '8462380b4367372c201e65d14b7a4d09e9991dc81a40b0a955181297a182ccad', 'd3af9198e25410b3f7e480d4da43774c81de14730b3871f9ce1a1e14790676ab', '417f7f6a10335e5b3c1666132290b62ab1043b8fda32c9ac4458814bd9a40ba1', 'a19ceab269ce8c32a6db5a54d1536df3473b3333508d150b404b487621e611c9', '0a61c52a43a6703057d4985ac972861ffc20db4859e8c92cfeffbf4ee7e3fba2', '0ca73d7b6ec5178cd70cfef4a3f74adcd3c4306c190f29a6b7c0e0aa2ec99d64', 'e20d11b0300370a4e63cc2a8e30af8282d90764d98aba08dd2435a5331740df4', 'baf50187f2723eaec5b0800e1c8dff5cabb95ec44de08f0d368ad716aa752f68', '4bce5f9969851644c0a2de49e6c053ddebb0c42af4fa470c87deff17915073ae', '40ed4faa4ca9920557fb27c89b73026de5227edaee9c71de9bc14b4e36858770')
_OXYGEN_BALANCE_DEFAULT_NUMERIC_HEX_SHA256 = ('ae8572790f312f96bb8e2d275a7695abf4bcfaea975bf121581ab24b6192a957', 'f6ecb08052b4a27b513f966f3085399d4a1ee58db2b83b35135adc910bceea62', 'b7b4b18b11eb7c931a6db12f39c68e1f54680ffe6938542f96acc68d3acd6677', '819bd7ec07bece92e9b2359d51d72429e93d2ae04f1446638c1e230e4a473c6d', 'a448783800eac17449a3aa0c3f808c7f8dbcb2d5fdd98028a0e0dd97f2cf853f', 'a24e0d42aee12e9d498922a9d9c4ee6c68d3a0f9f227c76e86e36b7f448b7821', '011bab00cb73effd4e9e1dacd75a07bb6874113dcc125b3d5c89fadfeef295d6', '01018c5a41a67a3cd083632bb94dd7fc1704fa700e4f6bda02899f8bd90e8a5c', 'e967a08ac9468c40af0897cac5b7a42ae962be8ca61b63f6cbb3f2bde79d4a34', '92edb220e9c82746b9578cd994bdcf1b36c112dd6f640c70449dc46e169df9ee', 'af07e6e5a393c9a4b9aac8728ff970bda4398b3b0868c09c6da7bd09bbd23f84', 'e649d70d5278a669cd83a6fb74f3ce4b357d08aff73bbfb3fa04348b332c6bb5', '54b2d0878e4e65716f86dfa1a28d95c5f27f937f8af9d3a6593f5420db143cd3', '7b6d388e8aa3ae4c179e427b83ffdd2269d479ac47fc7fffbb2cb1cce98197c9', '6a8a622fa66b1a27c8052cbd3e6e79223d8258356eaf81fbfd2c1e36166e38e1', '39d610344fdf831325a98aa4b2f1e3931563876a1adcf2a2677f83bc992bda84', '760afc4e3435c02ebf915144c66934e99de3b3775f0fba9895e54a7858a78b2c', '68117e6a6584bd5bc7dd218811a32e006e4b01d4a2da8a9f60bcfeeb92158a3b', '113e188277373e4d17393d6dd9015c8d43c157a6386c51f7ff7ae4cd3960dc75', '4bec7c0a51954dc825c5a5c1893695f9d74040257e18102ba3b3ee7ff9e8ba2b', 'a290e4f2007851dc711dfc4119b1e35950b36182dd9bf986a84d29882d9794e0', 'a2e5f1158d149e9dba05dd42b81df773c6b99005cd2e03bc65a243d2f7ba0bba', '127a6087caba5eab4d5aa3c50f95020bde227f3cfa1f509df266a8d14bdbb263', '86757483768c7a0af96d0302f4dc1365eb1d1ce93d5b0d3a4e87cc55fb995b56', '9ecc9bb6acb8bd3c8c1b03570fd1697f5538646596493ca94ed9dd7273492e1c', '491acb8321bfa980cf75460627d11a853fd304825b9514de950b06dc0868bbda', '10669df2b80169afb5e628a1e61f10543d6373b4f4dccbc25c6dd3c7d94fe6a3', '9c962083ded5ad20427e40f3a4eca65c4b8810f3e3d1e435ba9b41f7347e7210', '2548022b038396e9752f26032d3030f2af9dd2dcab6b73cb4bbd7288dc73a837', 'b171e3e85d96a7b3534840f28a6029e729acf775847760aa10818d0c9efa733e', '38ce9a2f96786313e889f5398e55c1414124c7039b339166db384191a198263d', '81614006757e7e1d1b06f03cc5e5859d07a3694d66ec0589ed7eba6aa98235a1', 'f2f9406a3cf38082fa55b26fd2a62c45e309d552b6f5c36c7b6cd60706609e8a', '9681bcb7ca95b5140a2afd6750a6f603a4e842a180bbefa64b988c2a2fe17cba', 'f630b4482ca9ce2ac90f330f0e3795701ef21f8d8438ba0a235f812c5cb5076d', '3213716f42914e013b5b3005820f23adf29889b2d0ede1b3cf7308dfd1fd1768', '7a4ca07dd3545173a300193aee0fe55a9dbceff61e2b32e50704385b1a665b50', 'a7d5972c52feae0a41460915708b035eeba3c4a21c08ac01ea0c321cbb8fb6ca', 'ff6dc4409483ad820d04d57e04bfec028143322ba63266537908a8c5e8e32aeb', 'f30bb8bf618a5968f527478f8bc511c07b182c47eff477d884e632f8707ad260', '345bb7487b8317c5d9fb8f578aca5a78fb86cd68533d99c9986aa7c0d3c6dd01', '84b9f001c8b90f0ab4300b833b58fa7f8f745355bc989f9b45e0a557111c8f53', '99c74d05e0c4a53c68c481bd55dca6edf7a23222144d8b45b373e749e0f75a0b', '213441400b1aebcf90b4d20c3928fa1ea5c07d28508e042c099ed528512fd139', '86b5726d34dc13fd4c5320d169bfd5f1f51c8b1bbc86e5585201ecdc2b1b99ca', '915bbd2580d872593fa0cfd141d9cab8f80b441a31eca1456f9211d4445ac02b', 'fe3666568e0c5ab050b4d5ef16fe10b7ac6ba517865cb3e2b7787e84218959ff', '4dd3ab378a5668a427ecaa0ab17e5ad981bc9c86d7365c4b829e540fb4ae80bc', '2f59a19365cd617e1c8ad79963da36d71d73b3af46963e5f0f9ca73cc94fa45e', 'fc84bf0ce2b8b6884ae2a55c37c8a2e1121c14eb3722d0920b18728e531c186f')
_OXYGEN_BALANCE_DEFAULT_RESULT_HEX_SHA256 = ('497da4d64cd8156fc968bd5605350340772ba6f47ebde94e94fba808659e1688', 'b2c41611d8511bda266f0bf7bd2b6702e0cab34835c75096b53cecd6a81d8897', '90a23c71038858fb23d6cb28f4848e829ebb5c3913f0a2b9902b86ebcfa106e6', '16724fed1ae5d3d0f307fba7086247da0a92fe13c6bc9e85f7542c570387abe0', 'e086a312c40e7f65714508164ca0d9eeaf451d4bc6c606094a8111e7751bdf97', 'c43fec3eb71fbe27ca4bb39da2e90b290e121de8af792264ae6af2ff7df24837', '8f627ba293db40e43ac2a4de64109cadb07c9e9ac6671d8e4b5d4df68d675732', 'efe90094b2cf377d14b5a374da5e500d813dc229533c84f075fe6fde5dcebc5b', '8dc732bdb3cb2d07c851a35eb7a909652178e453bfb9a420763543c9b83b85c0', 'c498ce9105ad1eefdd5700b29c5d6ab45fb2d03726f067ee5d9f2e1413ca9d61', 'd8789c42243cd6014e11afa62045b4c68998de7963bda39813bafbcd003ab3f3', '4b4fb07f0036ae2f417388eee97932dd55bc35cc737c1f31f27bc66921ff5360', '0e91d84430e1647e5cf8da837fb28e8c855aca0108ae4b5036502d11e0881c53', 'f93b5bef7d3ae606b5846ee3fc10b88b4222eee9fbdbc5d94e0515f7b4d32680', '32f5be3de341fe52d3ce50e16995b3da01e8ac87a7e62ecf530e05ebe800be05', 'd4372d9dbe113df4337c81b752b4ffbadc5a0175c9d6115a291ac7f7a42e1f48', 'b3a085b3ccf46b2128298b7a59ba1b1e4bbf62978633cf66873482dc30bd19df', 'bc112bb0b1ff1a44e5eefcbb0c730307a99712ff2b3d2995b3a3c4d09c63c838', '7283ed7738ded155015b2f1ca17f08ce7d437786e818f28f9508d134501c6594', '184078b3d631794fb4af8d31e412dd682632b1ff50fe7d01831887e50e9cdf3d', '9eaa667b744cfb6f57b982b85e9ef95e56dce857ce13dd38d87669875c8e3d36', '2dfdec2839966f222b6e34752933480cdcebc60caf0b1f0461b7c6ea8895d4ec', '7852646e35d08280bbf6769f8c3d7870563e28d1d771f45bbe40ef987ba881b5', '89623c8c39beede40c2472502b048f6265f820c8129b578696a8f3cd0feeec64', '56ec10aac0c3431ae899b8ee6aeb3bf7c890dabde83c49b25fca0e73c3eb5520', 'd3b8b8b1ff280ab14384d9eccc4e1b1efc9f0a328c83998622bf697a15fcf703', 'e6252fee11c2eceaf75b948398d6f0926264d12d377303dc87f4306f240b17c0', 'b9a814291ff04bc598b8a817de275b45beaea6566c09dc6d0336d2b968ccb78b', '3ca21cac03f1a643b14ec8c343e2551ff188737871048facc0ddc8bcf63f15b5', 'ce3b6c05fb91ac8701aed828821f0b05eabd43aa5dacf3a2e7d725886a473332', '3846577d2d746714f63bf37d36766c33b459dcaf728b6093aa121bc4f83f8f7e', '4417cfb51a6e179c2ecce8098ef0b098abdc8a82dfff7e77df2a13929e2c6395', '93f56e3ffb29784784a2e4933482392bad64c11e66ae724e95983df70a331f7e', 'cbe7d5e0ffbb70078f31947a80aa7e63386bd6f98a19dac2fe116fd6f9fb6f4a', '41ee6813c49d5eb39d7ad9052b6f504a48d0253a360c9453def6283a288133e4', 'c544e733037e9fd70448097eb4ee9375e15128037372eecb53375d168e0672c7', 'f8220c0d8d310c2b68ad5d125580d7db5fdc2ed81b354eb7ac021a67c80bdbb3', 'df2a659e360e3e63a0f9c8893fe93496b9a90d90a03fc023e5b715e45d51e652', '0035f8f4ec82335e37935270fb65cb93f80aeb7fcf45f00a8a15f7c05a35d0cd', '20f998970d9b3490d8fc0e4f8be5d8af1719e4f433a2ddac35bb9e8558bfd955', '7016dc2fe13073a2374fa09b0076a442fa5287206aeec36690bdd39c1f6ab1d1', '3177e80a3a995c20bcfeb19ef4878142da91aee9d51c0c0486748afddf4b7bb4', 'a24a73faf7ffa23de5096940804bdc47cbf7784e38ccca6848726058372aef12', '3303be0a6d0e819783a7b659f35b1dd040ed618098a379b7e19dbffd1538a769', '90d93324b7156a40c48bea14e08b97e140b1a88140a479f4137570a96ee48113', '290ac69bdefe61bb1aef824227352380bfdf2c3f45f026a7183152061d61d400', 'ab0dc2e30506180a5ab88dbcecc94a51002b470983e949252e5e062656a49acc', '693ebaaf47a95adf829646a452d537559d0643d86867b4c5935f995f8e36b306', '0b40f608d9d856fc2c213f4c9daf0701d2133b202b7df16e3f263aa4c7fe05c0', 'e0dac58b4a39e2b22df7ba6b79c682dd0b1328b4975d863d9d2b1b09feea84d5')

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
        if isinstance(value, Enum):
            return value.value
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
        assert diagnostics["root_solver"]["path"] == "direct"
        assert diagnostics["root_solver"]["refinement_evaluations"] == 0
        # Preserve the pre-existing output pins; root_solver is the one new,
        # additive diagnostic and has focused path/evaluation assertions above.
        pinned_diagnostics = {
            key: value for key, value in diagnostics.items() if key != "root_solver"
        }
        numeric_canonical = json.dumps(
            hex_values((p_o2, dict(pressures), pinned_diagnostics)),
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
            hex_values((p_o2, result_for_hash, pinned_diagnostics)),
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


def test_oxygen_balance_ion_mode_is_opt_in_and_reports_high_temperature_shifts(
    gas_pack: ImccGasDatapack,
) -> None:
    basalt_composition = {
        "Al2O3": 14.214, "CaO": 11.637, "FeO": 16.353, "MgO": 9.463,
        "Na2O": .182, "K2O": .35, "SiO2": 46.234, "TiO2": 1.587,
    }
    melt = evaluate(
        basalt_composition, 3000.0, basis_type="wt",
        allow_extrapolation=True, allow_out_of_envelope=True,
    )
    basalt = {name: melt.activity(name) for name in melt.parent_oxides}
    neutral_3000, _, _ = evaluate_gas_oxygen_balance(
        basalt, 3000.0, gas_pack, parent_oxides=tuple(basalt)
    )
    ion_3000, _, low_diag = evaluate_gas_oxygen_balance(
        basalt, 3000.0, gas_pack, parent_oxides=tuple(basalt), include_ions=True
    )
    assert abs(math.log10(ion_3000 / neutral_3000)) < 1e-3
    assert low_diag["ions_included"] is True
    assert set(low_diag["ion_fraction_by_element"]) >= {"Na", "K"}

    high_temperature_pack = load_gas_datapack(
        gas_path=(
            Path(__file__).parents[1]
            / "src/openimcc/data/gas/gas-shomate-hight-candidate-v2.csv"
        ),
        oxide_path=(
            Path(__file__).parents[1]
            / "src/openimcc/data/gas/condensate-hight-candidate-v2.csv"
        ),
    )
    # A dilute SiO2-Na2O-K2O activity vector keeps both extrapolated
    # roots inside the documented pO2 <= 1 bar molecular-flow bracket.
    high_temperature_activities = {
        "SiO2": 1e-5, "Na2O": 1e-7, "K2O": 1e-7,
    }
    for temperature in (4500.0, 6000.0):
        neutral, _, _ = evaluate_gas_oxygen_balance(
            high_temperature_activities, temperature, high_temperature_pack,
            parent_oxides=tuple(high_temperature_activities),
            allow_extrapolation=True,
        )
        ion, pressures, diagnostics = evaluate_gas_oxygen_balance(
            high_temperature_activities, temperature, high_temperature_pack,
            parent_oxides=tuple(high_temperature_activities),
            allow_extrapolation=True, include_ions=True,
        )
        assert diagnostics["ions_included"] is True
        assert diagnostics["residual"] < 1e-8
        assert diagnostics["ion_fraction_by_element"]["Na"] > 0.0
        assert diagnostics["ion_fraction_by_element"]["K"] > 0.0
        assert pressures["e-"] >= 0.0
        positive_charge = math.fsum(
            value for species, value in pressures.items() if species.endswith("+")
        )
        negative_charge = pressures["e-"] + math.fsum(
            value for species, value in pressures.items()
            if species.endswith("-") and species != "e-"
        )
        assert math.isclose(positive_charge, negative_charge, rel_tol=1e-12)
        # The pO2 shift is retained in the assertion message for reproducible
        # focused-test evidence at both requested temperatures.
        shift_dex = math.log10(ion / neutral)
        assert math.isfinite(shift_dex), (temperature, shift_dex)


def test_oxygen_balance_published_pack_keeps_same_channels(
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
        # Closed joins intentionally move roots, pressures, and validity flags.
        # The refreshed oxygen-balance pins above cover those outputs.
        assert set(base[1]) == set(published[1])
@pytest.mark.parametrize(
    ("activities", "temperature", "parents", "message"),
    [
        (
            {"SiO2": 0.0}, 2000.0, ("SiO2",),
            "oxygen-balance root is not bracketed on (-30.0, 0.0): "
            "F(low)=1.6726777188290725e-19, F(high)=0.35373170742653975",
        ),
        (
            {"K2O": 1.0, "SiO2": 0.5}, 2600.0, ("K2O", "SiO2"),
            "oxygen-balance root lies above pO2 = 1 bar, outside the Knudsen "
            "molecular-flow regime where the effusion law holds; no value returned "
            "(F(1 bar)=-2.6841950488205346)",
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


def test_condensate_source_table_follows_the_selected_row(
    gas_pack: ImccGasDatapack,
) -> None:
    """A phase-blind species map must not rename the row that supplied G."""
    corrected = (
        ("BaO", "l", 1350.0, "Ba-035", "Ba-035", -673304.1866616248),
        ("SrO", "l", 1350.0, "O-014", "O-014", -676460.3790433418),
        ("ZrO2", "l", 1350.0, "O-053", "O-053", -1190328.7307094275),
        ("Rh", "cr", 2000.0, "P672-Rh-cr", "P672-Rh-cr", -123994.38790011544),
        ("Ru", "cr", 1903.5, "P672-Ru-cr", "P672-Ru-cr", -107072.09380351192),
        ("MnO", "l", 2200.0, "LFS[MnO-IVTAN-1435;Tf=2120]", "MnO-liquid-construction", -642945.1452175163),
        ("As2O3", "l", 1400.0, "P672-As2O3-LFS", "P672-As2O3", -966087.500692369),
        ("Ag2O", "l", 1400.0, "P672-Ag2O-LFS", "P672-Ag2O-c", -303029.4914097916),
        (
            "Sb2O3", "l", 1400.0, "P672-Sb2O3-LFS", "P672-Sb2O3-c",
            -1020170.3103370364,
        ),
    )
    for species, phase, temperature, reference, table_id, gibbs in corrected:
        thermo = species_thermo(species, phase, temperature, gas_pack)
        assert thermo.source_row_id == reference
        assert thermo.source_table_id == table_id
        assert thermo.G_J_mol == gibbs

    unchanged = (
        ("BeO", "l", 2900.0, "NG-BeO-L-1352"),
        ("FeO", "l", 2200.0, "DG46-MASSON65"),
        ("P2O5", "l", 2200.0, "Jung-Hudon-2012"),
        ("HfO2", "l", 2000.0, "P672-HfO2-c"),
        ("ThO2", "l", 2500.0, "P672-ThO2-c"),
        ("Y2O3", "l", 2000.0, "P672-Y2O3-c"),
        ("K2O", "l", 1500.0, "K-012"),
    )
    for species, phase, temperature, table_id in unchanged:
        assert species_thermo(
            species, phase, temperature, gas_pack
        ).source_table_id == table_id
    legacy = load_gas_datapack(
        oxide_path=Path("src/openimcc/data/gas/condensate-feo-janaf-legacy.csv")
    )
    assert species_thermo("FeO", "l", 2200.0, legacy).source_table_id == "Fe-019"
    with pytest.raises(ImccGasTemperatureOutsideDomainError):
        species_thermo("Rh", "l", 2000.0, gas_pack)


def test_unrecognised_condensate_reference_does_not_inherit_the_packaged_table(
    gas_pack: ImccGasDatapack, tmp_path: Path
) -> None:
    source = gas_pack.oxide_path.read_text(encoding="utf-8").splitlines(True)
    rewritten = []
    ref_column = source[0].rstrip("\r\n").split(",").index("Ref")
    for line in source:
        if line.startswith("MgO(l),"):
            ending = line[len(line.rstrip("\r\n")):]
            fields = line.rstrip("\r\n").split(",")
            fields[ref_column] = "CUSTOM-THERMO-42"
            rewritten.append(",".join(fields) + ending)
        else:
            rewritten.append(line)
    oxide_path = tmp_path / "condensate.csv"
    oxide_path.write_text("".join(rewritten), encoding="utf-8")
    custom = load_gas_datapack(gas_path=gas_pack.gas_path, oxide_path=oxide_path)

    with pytest.raises(ImccGasUnrecognizedSourceError, match="CUSTOM-THERMO-42") as exc:
        species_thermo("MgO", "l", 1800.0, custom)
    assert exc.value.code == "imcc_gas_unrecognized_source"
    packaged = evaluate_gas(
        {"MgO": 1.0}, 1800.0, 1.0e-8, gas_pack,
        parent_oxides=("MgO",), gas_species=("Mg",),
    )
    overridden = evaluate_gas(
        {"MgO": 1.0}, 1800.0, 1.0e-8, custom,
        parent_oxides=("MgO",), gas_species=("Mg",),
    )
    assert overridden["Mg"] == packaged["Mg"]

    silica = []
    for line in source:
        if line.startswith("SiO2(l),"):
            ending = line[len(line.rstrip("\r\n")):]
            fields = line.rstrip("\r\n").split(",")
            fields[ref_column] = "O-999-NOT-JANAF"
            silica.append(",".join(fields) + ending)
        else:
            silica.append(line)
    silica_path = tmp_path / "silica.csv"
    silica_path.write_text("".join(silica), encoding="utf-8")
    silica_pack = load_gas_datapack(
        gas_path=gas_pack.gas_path, oxide_path=silica_path
    )
    with pytest.raises(ImccGasUnrecognizedSourceError, match="O-999-NOT-JANAF"):
        species_thermo("SiO2", "l", 2000.0, silica_pack)


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
            ["git", "show", f"2064476:{repository_path}"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        path = baseline_dir / filename
        path.write_bytes(payload)
        baseline_paths[filename] = path
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            old_columns = list(reader.fieldnames or [])
            rows = list(reader)
        defaults = build_gas_tables._empty_source_validity_fields()
        columns = list(old_columns)
        if not set(build_gas_tables.SOURCE_VALIDITY_COLUMNS) <= set(columns):
            columns.extend(build_gas_tables.SOURCE_VALIDITY_COLUMNS)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows({**defaults, **row} for row in rows)
    markers = build_gas_tables.build_source_liquid_markers(root / "data-src/janaf")
    validity = build_gas_tables.build_source_liquid_validity_rows(
        baseline_paths["gas-shomate.csv"],
        baseline_paths["condensate.csv"],
        markers=markers,
    )
    build_gas_tables.write_source_validity_columns(
        baseline_paths["gas-shomate.csv"], validity, phase="gas"
    )
    build_gas_tables.write_source_validity_columns(
        baseline_paths["condensate.csv"], validity, phase="condensate"
    )
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
        sorted(
            parent
            for parent in _REACTION_PARENT_OXIDES
            if parent not in {"Ag2O", "Os", "Re"}
            and f"{parent}(l)" in sf04_gas_pack.oxide_df.index
        )
    )
    activities = {parent: 1.0 for parent in parents}
    selected, _omitted = _default_reactions(parents, base_pack)
    channels = collection_type(name for name, _reaction in selected)
    switched_parent_oxides = {
        "TiO2", "Cr2O3", "V2O3", "SiO2",
        "Al2O3", "MgO", "CaO", "K2O", "FeO", "Na2O",
    }
    switched_channels = {
        name for name, reaction in selected
        if reaction[0] in switched_parent_oxides
    }
    new_low_temperature_channels = {
        "Al2", "Co", "Cr", "CrO", "CrO2", "CrO3", "Mn", "Nb", "NbO",
        "NbO2", "Ni", "Si2", "Si3", "Ti", "TiO", "TiO2", "V", "VO", "VO2",
        "H2O", "Zn", "Cd", "Pb", "PbO", "PbO2",
        "Na", "Na2", "NaO", "Na2O",
    }
    gas_coefficients = ("A", "B", "C", "D", "E", "F", "G", "H")
    seam_changed_channels = set()
    for species, base_rows in base_pack.gas_df.groupby(level=0, sort=False):
        current_rows = sf04_gas_pack.gas_df.loc[[species]]
        for _, base_row in base_rows.iterrows():
            matching = current_rows.loc[
                (current_rows["T_interval"] == base_row["T_interval"])
                & (current_rows["T_min"] == base_row["T_min"])
                & (current_rows["T_max"] == base_row["T_max"])
                & (current_rows["Ref"] == base_row["Ref"])
            ]
            if len(matching) != 1 or any(
                float(matching.iloc[0][column]) != float(base_row[column])
                for column in gas_coefficients
            ):
                seam_changed_channels.add(str(species).removesuffix("(g)"))
                break
    if "O2" in seam_changed_channels:
        seam_changed_channels.add("O")

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
        newly_fitted_channels = seam_changed_channels | (
            {name for name in channels if name in new_low_temperature_channels}
            if temperature < 1500
            else set()
        )
        unchanged_against_base = (
            set(base) - switched_channels - newly_fitted_channels
        )
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


def test_oxygen_balance_explicit_parents_keep_unknown_key_refusal(
    gas_pack: ImccGasDatapack,
) -> None:
    # An explicit parent list drops unlisted recognised parents only; an
    # unknown key must still be refused rather than silently ignored.
    with pytest.raises(ImccGasUnknownActivityError, match="NotAParent"):
        evaluate_gas_oxygen_balance(
            {"SiO2": 1.0, "NotAParent": 1.0},
            1800.0,
            gas_pack,
            parent_oxides=("SiO2",),
            allow_extrapolation=True,
        )

    activities = _quickstart_activities(1800.0, allow_extrapolation=True)
    base = evaluate_gas_oxygen_balance(
        activities, 1800.0, gas_pack, parent_oxides=("SiO2", "MgO"),
    )
    with_extra = evaluate_gas_oxygen_balance(
        {**activities, "Cr2O3": 1.0e-3},
        1800.0,
        gas_pack,
        parent_oxides=("SiO2", "MgO"),
    )
    assert with_extra[0].hex() == base[0].hex()
    assert with_extra[1] == base[1]
    assert not any(name.startswith("Cr") for name in with_extra[1])


def test_default_oxygen_balance_counts_auto_included_oxide_parents(
    gas_pack: ImccGasDatapack,
) -> None:
    # parent_oxides=None auto-includes a mapping's oxide parents in the
    # returned pressures, so the balance must count their oxygen too: the
    # default call must match the explicit call that lists the same parents.
    base = _quickstart_activities(1800.0, allow_extrapolation=True)
    activities = {**base, "ZnO": 1.0e-2}
    default = evaluate_gas_oxygen_balance(
        activities, 1800.0, gas_pack, allow_extrapolation=True,
    )
    explicit = evaluate_gas_oxygen_balance(
        activities, 1800.0, gas_pack,
        parent_oxides=tuple(activities), allow_extrapolation=True,
    )
    without_zno = evaluate_gas_oxygen_balance(
        base, 1800.0, gas_pack, allow_extrapolation=True,
    )
    assert any(name.startswith("Zn") for name in default[1])
    assert default[0].hex() == explicit[0].hex()
    assert default[1] == explicit[1]
    # ZnO gas carries real oxygen demand at this activity, so the root moves.
    assert abs(math.log10(default[0] / without_zno[0])) > 0.1


def test_refining_temperature_seam_rows_evaluate_with_supercooling_notices() -> None:
    pack = load_gas_datapack()
    gas_rows = (
        "H2O", "HCl", "HF", "NaCl", "(NaCl)2", "KCl", "(KCl)2",
        "NaF", "(NaF)2", "KF", "(KF)2", "Hg",
        "Zn", "Cd", "Pb", "PbO", "PbO2",
    )
    oxide_channels = (
        ("Na2O", ("Na",), "tier-1 liquid_from_solid construction"),
        ("K2O", ("K",), "tier-4 liquid_from_solid construction"),
        ("CaO", ("Ca",), "constant-Cp supercooled-liquid continuation"),
        ("MgO", ("Mg",), "constant-Cp supercooled-liquid continuation"),
        ("Al2O3", ("Al",), "constant-Cp supercooled-liquid continuation"),
    )
    for temperature in (1033.0, 1053.0, 1093.0):
        for species in gas_rows:
            value = species_thermo(species, "g", temperature, pack)
            assert math.isfinite(value.G_J_mol)
            assert math.isfinite(value.Cp_J_molK)
        for parent, channels, expected_notice in oxide_channels:
            liquid = species_thermo(parent, "l", temperature, pack)
            assert math.isfinite(liquid.G_J_mol)
            assert math.isfinite(liquid.Cp_J_molK)
            result = evaluate_gas(
                {parent: 1.0}, temperature, 1.0e-10, pack, gas_species=channels
            )
            for channel in channels:
                assert math.isfinite(result[channel])
                assert expected_notice in result.domain_flags[channel]
                if parent in {"CaO", "MgO", "SiO2", "Al2O3"}:
                    assert "hypothetical reference" in result.domain_flags[channel]
                    assert (
                        "not a prediction of a liquid condensate"
                        in result.domain_flags[channel]
                    )
                elif parent == "Na2O":
                    assert "supercooled liquid_from_solid branch" in result.domain_flags[channel]
                elif parent == "K2O":
                    assert "1190 ± 50 K fusion estimate" in result.domain_flags[channel]

    below_selected_k2o_fusion = evaluate_gas(
        {"K2O": 1.0}, 1000.0, 1.0e-10, pack, gas_species=("K",)
    )
    assert "supercooled liquid_from_solid branch below the 1013 K fusion temperature" in (
        below_selected_k2o_fusion.domain_flags["K"]
    )
    assert species_thermo("K2O", "l", 1033.0, pack).source_table_id == "K-012"
