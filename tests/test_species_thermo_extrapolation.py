"""Opt-in species thermochemistry continuation follows the gas runtime path."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from openimcc.gas import (
    ImccGasTemperatureOutsideDomainError,
    SpeciesThermo,
    _janaf_gibbs,
    _lamor_gibbs,
    _nearest_interval_row,
    _oxide_row_for_T,
    load_gas_datapack,
    species_thermo,
)


def _value_hexes(value: SpeciesThermo) -> dict[str, str]:
    return {
        name: getattr(value, name).hex()
        for name in (
            "Cp_J_molK",
            "S_J_molK",
            "H_app_kJ_mol",
            "G_J_mol",
        )
    }


def test_species_thermo_in_range_values_remain_byte_identical() -> None:
    legacy = SpeciesThermo(1.0, 2.0, 3.0, 4.0, "row", None, None, 10.0, 20.0)
    assert legacy.extrapolated is False
    assert legacy.continuation is None
    assert legacy.authority is None

    pack = load_gas_datapack()
    gas = species_thermo("O2", "g", 2000.0, pack)
    liquid = species_thermo("SiO2", "l", 2000.0, pack)

    assert _value_hexes(gas) == {
        "Cp_J_molK": "0x1.2def84315b766p+5",
        "S_J_molK": "0x1.0cbf4b7f5a86ep+8",
        "H_app_kJ_mol": "0x1.d9652d0829742p+5",
        "G_J_mol": "-0x1.d31c05b57dbffp+18",
    }
    assert _value_hexes(liquid) == {
        "Cp_J_molK": "0x1.4574931af766bp+6",
        "S_J_molK": "0x1.5994ed1502996p+7",
        "H_app_kJ_mol": "-0x1.86aa470158b5ep+9",
        "G_J_mol": "-0x1.1320008d49734p+20",
    }
    assert gas.extrapolated is False
    assert gas.declared_interval_K == (1500.0, 3000.0)
    assert gas.continuation is None
    assert gas.authority
    assert liquid.extrapolated is False
    assert liquid.declared_interval_K == (1996.0, 3000.0)
    assert liquid.continuation is None
    assert liquid.authority

    with pytest.raises(FrozenInstanceError):
        gas.extrapolated = True


def test_species_thermo_extrapolation_reuses_gas_runtime_rows() -> None:
    pack = load_gas_datapack()
    with pytest.raises(ImccGasTemperatureOutsideDomainError):
        species_thermo("O2", "g", 499.0, pack)
    with pytest.raises(ImccGasTemperatureOutsideDomainError):
        species_thermo("SiO2", "l", 1199.0, pack)

    gas = species_thermo("O2", "g", 499.0, pack, allow_extrapolation=True)
    gas_row = _nearest_interval_row(
        pack.gas_df, "O2(g)", 499.0, allow_extrapolation=True
    )
    assert gas.G_J_mol == _janaf_gibbs(499.0, gas_row)
    assert gas.source_row_id == str(gas_row["Ref"])
    assert gas.extrapolated is True
    assert gas.declared_interval_K == (
        float(gas_row["T_min"]), float(gas_row["T_max"])
    )
    assert gas.continuation == "Shomate fit continued"
    assert gas.authority

    liquid = species_thermo(
        "SiO2", "l", 1199.0, pack, allow_extrapolation=True
    )
    liquid_row = _oxide_row_for_T(
        pack.oxide_df, "SiO2(l)", 1199.0, allow_extrapolation=True
    )
    assert liquid.G_J_mol == _lamor_gibbs(1199.0, liquid_row)
    assert liquid.source_row_id == str(liquid_row["Ref"])
    assert liquid.extrapolated is True
    assert liquid.declared_interval_K == (
        float(liquid_row["T_min"]), float(liquid_row["T_max"])
    )
    assert liquid.continuation is not None
    assert liquid.authority


def test_species_thermo_refuses_crystal_extrapolation_without_runtime_path() -> None:
    with pytest.raises(
        ImccGasTemperatureOutsideDomainError,
        match="evaluate_gas has no crystal-phase continuation",
    ):
        species_thermo(
            "SiO2", "cr", 3000.0, allow_extrapolation=True
        )
