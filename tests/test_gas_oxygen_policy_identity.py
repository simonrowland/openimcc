from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json

import openimcc.gas as gas_module

_POLICY_METADATA_SHA256 = (
    "016bef813538af54a82ada7a82944460b73e21f4178e91693d9dac856664fb47",
) * 3
_POLICY_GAS_SHA256 = (
    "c4ee8cda715d3a1abc95de6154e7e2bcefa294677b55828d921adde33d5630ee",
    "fa5374fbef12758c5937a99b4690a882c7c88ed7046943083524aa2f543855ec",
    "87a8576eda35f5302a05466788851959d9dd2d192273c8ad10764efea56be9ed",
    "8b8b4f1aaf861bef451fe33d92b46f6532a2090cfa43d300d81083f6b39141ba",
    "9a9472db901753a7d8bdcf096948ff5f60aa7436ef73c08faa2a91a97e8eee96",
    "cd7b228aa71f93f115365d222ea34af1ddbef238c0bc5fb71db0cfcc68feb35b",
    "e7edf965f595022742e6c0f0831ebb749d7a90063823696f59a7f5fd56eeb4f9",
    "e5b95a8550ce1bfa11562cc041c67dc2c2f7d8d1018fb006ef8efe9d3701074c",
    "bcb5b3562321e23a01cde98990158f4b4d0cc558b2539bbbd547d1b42c6ba07c",
)
_POLICY_BALANCE_SHA256 = (
    "92b827d183667cdf64c915a0543173d6d1a3cbf4222402ff37d226dbfb734248",
    "ddc2778f58dec4095fc26689a7619a7768dd3d33f4d33e1a1894f94d9aa8b356",
    "7db4e50d814874c2d2c0f40ff73b9cc62dbdcdd70661e502b3b72052766b693b",
)
_POLICY_SATURATION_SHA256 = (
    "4a834e20d559bdd3ae3037088cf10da18eab2b256c029d3d1cb4eb2bfb145cf0",
    "0a7d9b429580e1ae9cfcc7995bb40ff4edde5c36f786e00c2e476afed3934351",
    "5d6e86ed45a815282a514fc8a2881f434800f6b22872f47ac371fdc799186ba0",
)


def _digest(value: object) -> str:
    def normalize(item: object) -> object:
        if isinstance(item, float):
            return item.hex()
        if isinstance(item, Mapping):
            return {
                str(key): normalize(value)
                for key, value in sorted(item.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(item, (tuple, list)):
            return [normalize(value) for value in item]
        return item

    payload = json.dumps(normalize(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def test_reaction_oxygen_outputs_match_pre_refactor_bits(packaged_gas_pack) -> None:
    parents = tuple(dict.fromkeys(
        [parent for parent, _, _ in gas_module._GAS_REACTIONS.values() if parent]
        + [
            side_parent
            for species in gas_module._GAS_REACTIONS
            if (side_parent := gas_module._side_parent(species))
        ]
    ))
    activities = dict.fromkeys(parents, 0.0)
    activities.update({
        "SiO2": 0.45,
        "MgO": 0.12,
        "FeO": 0.15,
        "CaO": 0.11,
        "Al2O3": 0.07,
        "Na2O": 0.03,
        "K2O": 0.01,
    })
    temperatures = (1500.0, 2000.0, 2400.0)
    f_o2_values = (1e-12, 1e-8, 1e-4)
    gas_index = 0
    for temperature_index, temperature in enumerate(temperatures):
        metadata = gas_module.gas_channel_metadata(
            activities, temperature, packaged_gas_pack, parent_oxides=parents
        )
        assert {"AlSiO", "NaCl", "BaCl", "HCl"} <= metadata.keys()
        assert _digest({
            name: (
                item.oxygen_atoms,
                item.parent_oxygen_demand,
                item.charge,
                item.available,
                item.neutral_pO2_exponent,
            )
            for name, item in metadata.items()
        }) == _POLICY_METADATA_SHA256[temperature_index]
        assert metadata["AlSiO"].parent_oxygen_demand == 3.5
        assert metadata["AlSiO"].neutral_pO2_exponent == -1.25
        assert metadata["NaCl"].parent_oxygen_demand == 0.5
        assert metadata["NaCl"].neutral_pO2_exponent == -0.25
        assert metadata["BaCl"].parent_oxygen_demand == 1.0
        assert metadata["BaCl"].neutral_pO2_exponent == -0.5
        assert metadata["HCl"].parent_oxygen_demand == 0.0
        assert metadata["HCl"].neutral_pO2_exponent == 0.0
        for species, expected in (
            ("AlSiO", (3.5, -1.25, -2.5)),
            ("NaCl", (0.5, -0.25, 0.0)),
            ("BaCl", (1.0, -0.5, 0.0)),
            ("HCl", (0.0, 0.0, 0.0)),
        ):
            policy = gas_module._reaction_oxygen_policy(
                species, gas_module._GAS_REACTIONS[species]
            )
            assert (
                policy.parent_oxygen_demand,
                policy.pO2_exponent,
                policy.saturation_oxygen_weight,
            ) == expected
        selected, omitted = gas_module._default_reactions(
            parents, packaged_gas_pack, temperature
        )
        selected_names = {name for name, _ in selected}
        assert set(metadata) == selected_names | set(omitted)
        assert {name for name, item in metadata.items() if item.available} == selected_names
        for f_o2 in f_o2_values:
            gas = gas_module.evaluate_gas(
                activities,
                temperature,
                f_o2,
                packaged_gas_pack,
                parent_oxides=parents,
            )
            assert set(gas) == selected_names
            assert _digest(dict(gas)) == _POLICY_GAS_SHA256[gas_index]
            gas_index += 1
        p_o2, pressures, _ = gas_module.evaluate_gas_oxygen_balance(
            activities, temperature, packaged_gas_pack, parent_oxides=parents
        )
        assert set(pressures) == selected_names
        assert _digest((p_o2, dict(pressures))) == _POLICY_BALANCE_SHA256[temperature_index]
        saturation = gas_module.evaluate_gas_saturation(
            activities,
            temperature,
            packaged_gas_pack,
            parent_oxides=parents,
            allow_extrapolation=True,
        )
        assert set(saturation.channel_pressures_bar) == selected_names
        assert _digest((
            saturation.pO2_bar,
            dict(saturation.species_pressures_bar),
            dict(saturation.channel_pressures_bar),
            saturation.P_sat_bar,
            dict(saturation.mole_fractions),
        )) == _POLICY_SATURATION_SHA256[temperature_index]
