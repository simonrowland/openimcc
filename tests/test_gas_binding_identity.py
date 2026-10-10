"""Golden pins for gas-layer numerical behavior and packaged CSV loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openimcc import evaluate, load_datapack
from openimcc.gas import (
    IMCC_PARENT_OXIDES,
    evaluate_gas,
    evaluate_gas_oxygen_balance,
    load_gas_datapack,
)


PACK_DIR = Path("src/openimcc/data/packs")
PACKAGED_MELTS = {
    "v1": (
        "imcc-sf04-v1.0.2.json",
        {
            "SiO2": 0.45,
            "MgO": 0.20,
            "FeO": 0.10,
            "CaO": 0.08,
            "Al2O3": 0.08,
            "TiO2": 0.02,
            "Na2O": 0.05,
            "K2O": 0.02,
        },
    ),
    "ext": (
        "imcc-sf04-ext-v4.json",
        {
            "SiO2": 0.42,
            "MgO": 0.18,
            "FeO": 0.10,
            "CaO": 0.08,
            "Al2O3": 0.08,
            "TiO2": 0.02,
            "Na2O": 0.05,
            "K2O": 0.02,
            "S": 0.02,
            "P2O5": 0.03,
        },
    ),
}

GOLDEN = json.loads(
    Path(__file__).with_name("golden-gas-binding-identity-evaluations.json").read_text(
        encoding="utf-8"
    )
)
EVALUATION_RTOL = 1.0e-12


def _gas_activity_mapping(melt_pack, composition: dict[str, float], temperature: float):
    melt = evaluate(
        composition,
        temperature,
        melt_pack,
        allow_out_of_envelope=True,
        allow_extrapolation=True,
        enable_sp_extension=bool(melt_pack.extension_parents),
    )
    return {
        oxide: float(melt.parent_activity[index])
        for index, oxide in enumerate(melt_pack.parent_oxides)
        if oxide in IMCC_PARENT_OXIDES
    }


def test_packaged_gas_outputs_match_per_value_pins() -> None:
    gas_pack = load_gas_datapack()
    for label, (filename, composition) in PACKAGED_MELTS.items():
        melt_pack = load_datapack(PACK_DIR / filename)
        for temperature in (1800.0, 2500.0):
            activities = _gas_activity_mapping(melt_pack, composition, temperature)
            for fugacity in (1.0e-10, 1.0e-8):
                gas = evaluate_gas(activities, temperature, fugacity, gas_pack)
                key = f"{label}:{int(temperature)}:{fugacity:.0e}"
                assert dict(gas) == pytest.approx(
                    GOLDEN["gas"][key], rel=EVALUATION_RTOL, abs=0.0
                )

            p_o2, pressures, diagnostics = evaluate_gas_oxygen_balance(
                activities, temperature, gas_pack
            )
            key = f"{label}:{int(temperature)}"
            expected = GOLDEN["oxygen_balance"][key]
            assert float(p_o2) == pytest.approx(
                expected["pO2_bar"], rel=EVALUATION_RTOL, abs=0.0
            )
            assert dict(pressures) == pytest.approx(
                expected["partial_pressures"], rel=EVALUATION_RTOL, abs=0.0
            )
            assert diagnostics["mode"] == "oxygen_balance_effusion"
            assert 0.0 <= float(diagnostics["residual"]) <= 1.0e-12
            assert 0 <= int(diagnostics["iterations"]) <= 100
