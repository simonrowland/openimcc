"""Pins for published-core integrity and packaged numeric output."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from openimcc import ImccMalformedDatapackError, evaluate, load_datapack
import openimcc.kernel as kernel


PACK_DIR = Path("src/openimcc/data/packs")
PUBLISHED_CORE_SHA256 = (
    "f2b479cd54e3c82704a5863fcc06836f72045375d9a8c7f8d2fad19e98f75d05"
)
COMPOSITIONS = (
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
    {
        "SiO2": 0.60,
        "MgO": 0.10,
        "FeO": 0.08,
        "CaO": 0.06,
        "Al2O3": 0.08,
        "TiO2": 0.02,
        "Na2O": 0.04,
        "K2O": 0.02,
    },
    {
        "SiO2": 0.30,
        "MgO": 0.15,
        "FeO": 0.10,
        "CaO": 0.10,
        "Al2O3": 0.10,
        "TiO2": 0.05,
        "Na2O": 0.12,
        "K2O": 0.08,
    },
)
NUMERIC_OUTPUT_SHA256 = {
    ("v1", 0, 1800): "9d7afe03be9c53a3ad8c1702d23df05f3e6786b4f244e1f239b51d2925fd88f2",
    ("v1", 0, 2500): "acbca440b60e1af5892df0b90adf8fd6403def748336266d52ff30d6674464c0",
    ("v1", 1, 1800): "01d40931bbbd3e85042fd64eb7c956a7b970221f15ae9d4be7681dfdf9945828",
    ("v1", 1, 2500): "ca44ce6ecf40673813f21012258a94cf70c56e57b32f02c768e510589a54c1cd",
    ("v1", 2, 1800): "8deb0353ad5dc056d6452e459f14cc61c4c619b12d062579912fd63b09e9588e",
    ("v1", 2, 2500): "8a69102b3ba406cb707cd7626150b2f115d5a8d9f7acf52348549aec5766624e",
    ("ext", 0, 1800): "54c9cb829f05289b7663f5d3e2a5d6e64837651a974d82e05fad01d3b3ff9c4e",
    ("ext", 0, 2500): "3c93a206fe1be1f87d61eace4cf4888112aaa906e63bc3ba93cc8c7b10338f7d",
    ("ext", 1, 1800): "05e9db464ee26d772b6d52461e67bf1183b879ea08a9fc09de6b7bb6e1ae798a",
    ("ext", 1, 2500): "693944a475ca9ecd532397cc91fcd102fc4a734dad777579fe01bbf3edcdc04d",
    ("ext", 2, 1800): "34b34a70fd6dffe665ad1f58dac8581fcd102c0cd29cb1bd4681e7f7fbaac00a",
    ("ext", 2, 2500): "94a2fa7ce4931c32c380b1605b2484e0cb6d813b6b4ac0051ad613e40a8b4b82",
}


def _numeric_output_digest(result: object) -> str:
    values: list[str] = []
    for field in (
        "parent_mol",
        "parent_x",
        "parent_x_star",
        "parent_activity",
        "parent_gamma",
        "complex_x",
        "species_x",
    ):
        values.extend(
            float(value).hex()
            for value in np.asarray(getattr(result, field)).flat
        )
    values.extend(
        float(value).hex()
        for value in (
            result.temperature_K,
            result.basis,
            result.D,
            result.convergence.residual_inf,
            result.convergence.residual_l2,
            result.convergence.total_displacement,
        )
    )
    return hashlib.sha256(
        json.dumps(values, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def test_published_core_integrity_hash_and_refusal_are_pinned(
    tmp_path: Path,
) -> None:
    assert kernel._PUBLISHED_DATAPACK_SHA256 == PUBLISHED_CORE_SHA256

    manifest_path = PACK_DIR / "imcc-sf04-v1.0.2.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["rows"][0]["A"] += 0.5
    altered_path = tmp_path / "altered-core.json"
    altered_path.write_text(json.dumps(manifest), encoding="utf-8")

    try:
        load_datapack(altered_path)
    except ImccMalformedDatapackError as exc:
        assert exc.code == "imcc_malformed_datapack"
        assert str(exc).startswith(
            "published IMCC datapack canonical hash mismatch: expected "
            f"{PUBLISHED_CORE_SHA256}, got "
        )
    else:
        raise AssertionError("modified published core row was accepted")


def test_packaged_evaluation_outputs_keep_their_float_hex_values() -> None:
    packs = {
        "v1": load_datapack(PACK_DIR / "imcc-sf04-v1.0.2.json"),
        "ext": load_datapack(PACK_DIR / "imcc-sf04-ext-v4.json"),
    }
    for label, pack in packs.items():
        for composition_index, composition in enumerate(COMPOSITIONS):
            for temperature in (1800.0, 2500.0):
                result = evaluate(
                    composition,
                    temperature,
                    pack,
                    allow_out_of_envelope=True,
                    allow_extrapolation=True,
                    enable_sp_extension=(label == "ext"),
                )
                key = (label, composition_index, int(temperature))
                assert _numeric_output_digest(result) == NUMERIC_OUTPUT_SHA256[key]
