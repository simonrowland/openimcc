"""Opt-in species thermochemistry continuation follows the gas runtime path."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from openimcc.gas import (
    ImccGasTemperatureOutsideDomainError,
    SpeciesThermo,
    _janaf_gibbs,
    _lamor_gibbs,
    _nearest_interval_row,
    _oxide_row_for_T,
    evaluate_gas,
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
    assert legacy.phase_identity is None
    assert legacy.formation_enthalpy_298_kJ_mol is None
    assert legacy.formation_enthalpy_source_id is None

    pack = load_gas_datapack()
    gas = species_thermo("O2", "g", 2000.0, pack)
    liquid = species_thermo("SiO2", "l", 2000.0, pack)
    crystal = species_thermo("SiO2", "cr", 1500.0, pack)

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
    assert _value_hexes(crystal) == {
        "Cp_J_molK": "0x1.240610b27dc5cp+6",
        "S_J_molK": "0x1.25a4de21ba0e6p+7",
        "H_app_kJ_mol": "-0x1.9ddd493cd318dp+9",
        "G_J_mol": "-0x1.ffb339de40090p+19",
    }
    assert gas.extrapolated is False
    assert gas.declared_interval_K == (1500.0, 3000.0)
    assert gas.continuation is None
    assert gas.authority
    assert liquid.extrapolated is False
    assert liquid.declared_interval_K == (1996.0, 3000.0)
    assert liquid.continuation is None
    assert liquid.authority
    assert crystal.extrapolated is False
    assert crystal.declared_interval_K == (1000.0, 1996.0)
    assert crystal.continuation is None
    assert crystal.authority

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


def test_species_thermo_crystal_phase_identity_and_formation_anchors() -> None:
    pack = load_gas_datapack()
    high_cristobalite = species_thermo("SiO2", "cr", 1500.0, pack)
    assert high_cristobalite.phase_identity is not None
    assert high_cristobalite.phase_identity["kind"] == "single_polymorph"
    assert high_cristobalite.phase_identity["polymorph"] == "high cristobalite"
    assert high_cristobalite.phase_identity["transition_temperatures_K"][0][
        "temperature_K"
    ] == 1079
    assert high_cristobalite.formation_enthalpy_298_kJ_mol == pytest.approx(
        -905.489
    )
    assert high_cristobalite.formation_enthalpy_source_id == "O-035"

    legacy_table = (
        Path(__file__).parents[1]
        / "src/openimcc/data/gas/condensate-feo-janaf-legacy.csv"
    )
    lam_pack = load_gas_datapack(oxide_path=legacy_table)
    lam_cristobalite = species_thermo("SiO2", "cr", 1500.0, lam_pack)
    assert lam_cristobalite.source_table_id == "LH87 Table 2"
    assert lam_cristobalite.phase_identity is not None
    assert lam_cristobalite.phase_identity["polymorph"] == (
        "cristobalite (low/high subtype not specified)"
    )

    gas = species_thermo("H", "g", 2000.0, pack)
    assert gas.formation_enthalpy_298_kJ_mol == pytest.approx(217.999)
    assert gas.formation_enthalpy_source_id == "H-001"


def test_species_thermo_crystal_constant_cp_continuation_is_continuous() -> None:
    pack = load_gas_datapack()
    edge = species_thermo("Os", "cr", 3000.0, pack)
    continued = species_thermo(
        "Os", "cr", 3100.0, pack, allow_extrapolation=True
    )

    assert continued.extrapolated is True
    assert continued.declared_interval_K == (1200.0, 3000.0)
    assert continued.phase_identity is not None
    polymorph = continued.phase_identity["polymorph"]
    assert continued.continuation == (
        f"constant-Cp crystal continuation within {polymorph} stability range"
    )
    assert continued.Cp_J_molK == edge.Cp_J_molK
    assert continued.H_app_kJ_mol == pytest.approx(
        edge.H_app_kJ_mol + edge.Cp_J_molK * 100.0 / 1000.0
    )
    assert continued.S_J_molK == pytest.approx(
        edge.S_J_molK + edge.Cp_J_molK * math.log(3100.0 / 3000.0)
    )
    assert continued.G_J_mol == pytest.approx(
        1000.0 * continued.H_app_kJ_mol - 3100.0 * continued.S_J_molK
    )


def test_species_thermo_refuses_transition_and_melting_crossings() -> None:
    with pytest.raises(
        ImccGasTemperatureOutsideDomainError,
        match="outside its source-printed stability range.*transition at 1079 K",
    ):
        species_thermo(
            "SiO2", "cr", 999.0, allow_extrapolation=True
        )

    with pytest.raises(
        ImccGasTemperatureOutsideDomainError,
        match="above its source-printed melting point of 1996 K",
    ):
        species_thermo(
            "SiO2", "cr", 1997.0, allow_extrapolation=True
        )


@pytest.mark.parametrize(
    ("species", "phase", "temperature", "channel", "activities", "expected"),
    (
        (
            "SiO2", "l", 1199.0, "Si", {"SiO2": 1.0e-3},
            "SiO2(l) uses a tier-1 liquid_from_solid construction from "
            "Richet-Bottinga-Tequi-1984",
        ),
        (
            "Al2O3", "l", 999.0, "Al", {"Al2O3": 1.0e-3},
            "Al2O3(l) uses a labelled constant-Cp supercooled-liquid "
            "continuation from JANAF Al-100; this is a hypothetical reference, "
            "not a prediction of a liquid condensate",
        ),
        (
            "UO2", "l", 3001.0, "U", {"UO2": 1.0e-3},
            "UO2(l) uses a labelled constant-Cp supercooled-liquid "
            "continuation from NASA Glenn card NG-1977",
        ),
        (
            "ZnO", "l", 3001.0, "Zn", {"ZnO": 1.0e-3},
            "ZnO(l) uses a tier-4 liquid_from_solid construction from "
            "TSIV-1603; the central liquid uses the printed 2250 K node, and "
            "the quadratic crystal-Cp extension is its uncertainty bound",
        ),
        (
            "Re2O7", "g", 499.0, "Re2O7", {"Re": 1.0},
            "constructed Re2O7(g) from Smith 1952 measured vapor pressure "
            "and B677 condensed functions over 503.15-598.15 K; maximum "
            "log-pressure fit scatter propagated as ±1.3 kJ/mol at 499 K",
        ),
    ),
)
def test_continuation_label_matches_evaluate_gas_flag(
    packaged_gas_pack,
    species: str,
    phase: str,
    temperature: float,
    channel: str,
    activities: dict[str, float],
    expected: str,
) -> None:
    thermo = species_thermo(
        species,
        phase,
        temperature,
        packaged_gas_pack,
        allow_extrapolation=True,
    )
    result = evaluate_gas(
        activities,
        temperature,
        1.0e-8,
        packaged_gas_pack,
        gas_species=(channel,),
        allow_extrapolation=True,
    )

    assert thermo.continuation == expected
    runtime_flag = result.domain_flags[channel]
    assert runtime_flag is not None
    assert expected in runtime_flag
