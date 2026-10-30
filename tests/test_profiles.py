"""Matched pack-profile selection tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import struct
from types import SimpleNamespace

import pytest

from openimcc import (
    ProfileConflictError,
    ProfileRedoxUnavailableError,
    evaluate,
    evaluate_gas,
    evaluate_redox,
    load_datapack,
    load_profile,
)
from openimcc.gas import (
    evaluate_gas_oxygen_balance,
    evaluate_gas_saturation,
    gas_boiling_temperature,
    load_gas_datapack,
)


_BASALT = {
    "SiO2": 51.85068,
    "MgO": 4.78527,
    "FeO": 13.77307,
    "CaO": 9.02862,
    "Al2O3": 14.80572,
    "TiO2": 1.73824,
    "Na2O": 3.23108,
    "K2O": 0.78732,
}


@pytest.mark.parametrize(
    ("name", "melt_version", "has_redox"),
    [
        ("openimcc", "1.0.7-d066-v5", True),
        ("published", "1.0.2", False),
    ],
)
def test_named_profiles_load(name: str, melt_version: str, has_redox: bool) -> None:
    profile = load_profile(name)

    assert profile.name == name
    assert profile.melt_pack.version == melt_version
    assert profile.gas_datapack.gas_path.name == "gas-shomate.csv"
    assert profile.provenance
    assert (profile.redox_pack_path is not None) is has_redox
    if name == "published":
        assert profile.gas_datapack.oxide_path.name == "condensate.csv"
        assert "sf04-published" in str(profile.gas_datapack.oxide_path)
    else:
        assert profile.gas_datapack.oxide_path.name == "condensate.csv"
        assert profile.redox_pack_path.name == "openimcc-redox-v4.json"
    with pytest.raises(FrozenInstanceError):
        profile.name = "published"


def test_openimcc_profile_matches_readme_quickstart_values_bit_for_bit() -> None:
    profile = load_profile("openimcc")
    default_melt = evaluate(_BASALT, 1800.0, basis_type="wt")
    profile_melt = evaluate(
        _BASALT, 1800.0, basis_type="wt", profile=profile
    )

    for field in (
        "parent_mol",
        "parent_x",
        "parent_x_star",
        "parent_activity",
        "parent_gamma",
        "complex_x",
        "species_x",
    ):
        assert getattr(profile_melt, field).tobytes() == getattr(
            default_melt, field
        ).tobytes()
    assert struct.pack("!d", profile_melt.D) == struct.pack("!d", default_melt.D)
    assert {
        key: value
        for key, value in profile_melt.labels.identity.items()
        if key != "profile"
    } == dict(default_melt.labels.identity)
    assert profile_melt.labels.identity["profile"] == "openimcc"

    activities = {
        name: profile_melt.activity(name) for name in profile_melt.parent_oxides
    }
    default_gas = evaluate_gas(
        activities, 1800.0, 1e-10, load_gas_datapack()
    )
    profile_gas = evaluate_gas(
        activities, 1800.0, 1e-10, profile=profile
    )
    assert dict(profile_gas) == dict(default_gas)
    assert profile_gas.gas_table_digest == default_gas.gas_table_digest
    assert profile_gas.condensate_table_digest == default_gas.condensate_table_digest
    assert profile_gas.profile == "openimcc"


def test_explicit_packs_and_profiles_are_mutually_exclusive() -> None:
    with pytest.raises(ProfileConflictError):
        evaluate(_BASALT, 1800.0, load_datapack(), profile="published")
    with pytest.raises(ProfileConflictError):
        evaluate_gas(
            {"SiO2": 0.5}, 1800.0, 1e-10, load_gas_datapack(),
            profile="published",
        )
    with pytest.raises(ProfileConflictError):
        evaluate_redox(
            {"Fe": 1.0, "O": 1.25}, 2200.0, "closed",
            redox_pack_path="openimcc-redox-v4.json", profile="openimcc",
        )


def test_profile_selects_gas_oxygen_balance_and_saturation_packs() -> None:
    profile = load_profile("openimcc")
    melt = evaluate(_BASALT, 1800.0, basis_type="wt", profile=profile)
    activities = {name: melt.activity(name) for name in melt.parent_oxides}

    default_root, _, _ = evaluate_gas_oxygen_balance(
        activities, 1800.0, profile.gas_datapack
    )
    profile_root, profile_pressures, _ = evaluate_gas_oxygen_balance(
        activities, 1800.0, profile=profile
    )
    assert profile_root.hex() == default_root.hex()
    assert profile_pressures.profile == "openimcc"
    assert (
        profile_pressures.gas_table_digest
        == profile.gas_datapack.gas_table_digest
    )

    saturation = evaluate_gas_saturation(
        {"CaO": 1.0}, 1800.0, parent_oxides=("CaO",),
        allow_extrapolation=True, profile=profile,
    )
    assert saturation.profile == "openimcc"
    assert saturation.condensate_table_digest == (
        profile.gas_datapack.condensate_table_digest
    )


def test_gas_boiling_temperature_accepts_profile_without_datapack(monkeypatch) -> None:
    import openimcc.gas as gas_module

    def fake_saturation(_activities, temperature, _datapack, **_kwargs):
        return SimpleNamespace(
            P_sat_bar=temperature,
            diagnostics={},
            pO2_bar=1.0,
            species_pressures_bar={},
            channel_pressures_bar={},
            mole_fractions={},
            mode="synthetic",
            domain_flags={},
        )

    monkeypatch.setattr(gas_module, "evaluate_gas_saturation", fake_saturation)
    profile = load_profile("openimcc")
    temperature, result = gas_boiling_temperature(
        {}, ambient_pressure_bar=1.5, T_bracket=(1.0, 2.0), profile=profile
    )

    assert temperature == pytest.approx(1.5)
    assert result.profile == "openimcc"
    assert result.gas_table_digest == profile.gas_datapack.gas_table_digest


def test_published_profile_refuses_redox_because_sf04_has_no_redox_model() -> None:
    with pytest.raises(ProfileRedoxUnavailableError, match="no redox model"):
        evaluate_redox(
            {"Fe": 1.0, "O": 1.25}, 2200.0, "closed", profile="published"
        )


def test_openimcc_profile_selects_the_default_redox_pack() -> None:
    default = evaluate_redox({"Fe": 1.0, "O": 1.25}, 2200.0, "closed")
    profiled = evaluate_redox(
        {"Fe": 1.0, "O": 1.25}, 2200.0, "closed", profile="openimcc"
    )

    assert profiled.lambda_ln_f_o2 == default.lambda_ln_f_o2
    assert profiled.redox_pack_digest == default.redox_pack_digest
    assert profiled.melt_pack_binding_digest == default.melt_pack_binding_digest
