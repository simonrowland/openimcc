"""Pins for published-core integrity and packaged numeric output."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

from openimcc import ImccMalformedDatapackError, evaluate, load_datapack
import openimcc.kernel as kernel
import openimcc.model as model


PACK_DIR = Path("src/openimcc/data/packs")
PUBLISHED_CORE_SHA256 = (
    "f2b479cd54e3c82704a5863fcc06836f72045375d9a8c7f8d2fad19e98f75d05"
)
LEGACY_PACK_SHA256 = {
    "imcc-sf04-v1.0.2.json": "313ab57ffb45f25d762b76f54cba70c4bac5df06cb8eca59dba4a7ce046c89bf",
    "imcc-sf04-ext-v4.json": "fbae18acb13acb18404581183ab9bf91ef657992ec9273d0ae83f924695c0922",
}
LEGACY_BINDING_SHA256 = {
    "imcc-sf04-v1.0.2.json": "f2b479cd54e3c82704a5863fcc06836f72045375d9a8c7f8d2fad19e98f75d05",
    "imcc-sf04-ext-v4.json": "4cbec2ee85cd95314a5f62f5ce4e00cbe660935bde12df3a7e5daa425612ad03",
}
D066_MANIFEST_SHA256 = {
    "imcc-sf04-d066-v1.json": (
        "d894f4e5ce79ba393f7068ec7d9032336eda8276293c4f46cafb18944a8b247e"
    ),
    "imcc-sf04-d066-v1-kcaalsi2o7.json": (
        "e737f191a6b12df89bf8c3814e65e25ee0d37745ad8142d12ed0c8614d9f55d5"
    ),
    "imcc-sf04-d066-ext-v1.json": (
        "90b3fb016073b415e7f505b4739fb193e9cf4680232307d642db730694bcb44b"
    ),
}
D066_BINDING_SHA256 = D066_MANIFEST_SHA256
D066_DEFAULT_K_PRESSURE_BAR = float.fromhex("0x1.edb56ea3c124cp-12")
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
EVALUATION_EXPECTATIONS = json.loads(
    Path(__file__).with_name("golden-binding-digest-evaluations.json").read_text(
        encoding="utf-8"
    )
)["cases"]
# Modelled on tests/conformance/test_conformance.py and its JSON golden file.
# These values were produced by Python 3.14.5 with NumPy 2.5.3. A relative
# tolerance of 1e-12 admits last-digit solver differences between NumPy builds
# while remaining tight enough to catch the association-constant mutations
# exercised during review. Exact zero expectations stay exact.
EVALUATION_RTOL = 1.0e-12
SOLVER_TOL = 1.0e-12
EXACT_PIN_ENVIRONMENT = ("3.14.5", "2.4.4")
EXACT_EVALUATION_SHA256 = {
    "v1:0:1800": "1159f55beb5a6469215ca21a88d89a32fe1ce4583c5ee1a982d67cedd979b890",
    "v1:0:2500": "94a0d3d89e3bc1775c7c0af4c1e615e0afbbd33d71f892d97fab6e99c8046cff",
    "v1:1:1800": "f9259c684768b6a181fd0676babf701f5de053086065a9a780ca995453315e14",
    "v1:1:2500": "6bb1575fbbf88356e5c05395e0075a6cec3243d7858d3fb76b7c8fe92e3060c5",
    "v1:2:1800": "71553eed4862c3fe07033e4a4a324d1de5253309a6d5d6adf80e838a0da04f45",
    "v1:2:2500": "a7a541109b1f1fc2e3a6e6f2cdd1e5c0bdec356f357dea63b4a02775eaf6a750",
    "ext:0:1800": "206cb9032b5d469349a4385e252976c7c3c292941ceb5e288f3eb914a0fe0cf7",
    "ext:0:2500": "fe0c6481d162d9841225ea5b1a883984a4c65eaa43e090fb138ed656ec2b5328",
    "ext:1:1800": "33df36559650a9e21efa2e521bb3ec9a8a1d3a2d9dc6f108573f408b6e4e894b",
    "ext:1:2500": "7bf2e6cd70e398db60d7f2c4f5cde6f781ff79256eb95a7b929f4335c80fb077",
    "ext:2:1800": "006ae56686e6d2e9ab3dfd8e374411024cc8fa9096d4d1dddb89da02355eef0a",
    "ext:2:2500": "f707bfcc1960d8c76ca1820e78f04ab8ee909b3808dc9d3e7d03e03f4f9e681a",
}


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


def test_legacy_pack_bytes_are_pinned() -> None:
    import hashlib

    for filename, expected in LEGACY_PACK_SHA256.items():
        raw = (PACK_DIR / filename).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected


def test_legacy_canonical_binding_digests_are_pinned() -> None:
    for filename, expected in LEGACY_BINDING_SHA256.items():
        pack = load_datapack(PACK_DIR / filename)
        assert pack.kernel_datapack.binding_digest == expected


def test_d066_manifest_and_binding_digests_are_pinned() -> None:
    # Intended data identity: FC87 K rows translated to JANAF K2O(l), KCa
    # disabled per E25 in primary, with a separate active sensitivity arm.
    for filename, expected_manifest in D066_MANIFEST_SHA256.items():
        path = PACK_DIR / filename
        data = json.loads(path.read_text(encoding="utf-8"))
        _, actual_manifest = model._validate_d066_core(data)
        pack = load_datapack(path)
        assert pack.model_id == "IMCC-SF04-D066"
        assert actual_manifest == expected_manifest
        assert pack.kernel_datapack.binding_digest == D066_BINDING_SHA256[filename]


def test_default_d066_k_pressure_is_pinned_after_data_change() -> None:
    from openimcc.gas import evaluate_gas, load_gas_datapack

    melt = load_datapack()
    result = evaluate(
        COMPOSITIONS[0],
        2000.0,
        melt,
        allow_out_of_envelope=True,
    )
    pressure = evaluate_gas(
        dict(zip(result.parent_oxides, result.parent_activity)),
        2000.0,
        1.0e-6,
        load_gas_datapack(),
        gas_species=("K",),
    )["K"]
    # The previous CI matrix showed about 5e-17 backend movement in one
    # legacy activity. Keep the existing portable-golden convention: a tight
    # relative bound with no absolute allowance; pack and digest pins stay exact.
    assert float(pressure) == pytest.approx(
        D066_DEFAULT_K_PRESSURE_BAR,
        rel=EVALUATION_RTOL,
        abs=0.0,
    )


def test_legacy_evaluation_outputs_match_portable_pins() -> None:
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
                key = f"{label}:{composition_index}:{int(temperature)}"
                expected = EVALUATION_EXPECTATIONS[key]
                exact_values = []
                for field in (
                    "parent_mol",
                    "parent_x",
                    "parent_x_star",
                    "parent_activity",
                    "parent_gamma",
                    "complex_x",
                    "species_x",
                ):
                    actual = [float(value) for value in getattr(result, field)]
                    exact_values.extend(actual)
                    assert actual == pytest.approx(
                        expected[field], rel=EVALUATION_RTOL, abs=0.0
                    ), f"{key} {field} moved"
                scalars = {
                    "temperature_K": result.temperature_K,
                    "basis": result.basis,
                    "D": result.D,
                }
                # Solver diagnostics depend on the numerical library build, so
                # they are checked against the solver's contract, not pinned.
                # The solve accepts on the infinity norm of the residual
                # (kernel tol, 1e-12). For a residual vector f of length n,
                #   |f|_inf <= |f|_2 <= sqrt(n) * |f|_inf,
                # and n is at most the number of parent oxides, so
                #   |f|_2 <= sqrt(n_parents) * tol.
                # Check: eight components of 5e-13 give inf 5e-13 (accepted)
                # and L2 sqrt(8) * 5e-13 = 1.41e-12, inside sqrt(8) * 1e-12.
                # The displacement is the log-space distance from the initial
                # guess (order 10, not small); only its sanity is checked.
                convergence = result.convergence
                residual_inf = float(convergence.residual_inf)
                residual_l2 = float(convergence.residual_l2)
                l2_bound = math.sqrt(len(result.parent_x)) * SOLVER_TOL
                assert 0.0 <= residual_inf <= SOLVER_TOL, f"{key} {residual_inf}"
                assert 0.0 <= residual_l2 <= l2_bound, f"{key} {residual_l2}"
                displacement = float(convergence.total_displacement)
                assert math.isfinite(displacement) and displacement >= 0.0, key
                for name, actual in scalars.items():
                    exact_values.append(float(actual))
                    assert float(actual) == pytest.approx(
                        expected["scalars"][name],
                        rel=EVALUATION_RTOL,
                        abs=0.0,
                    ), f"{key} {name} moved"
                # These hashes pin the IEEE-754 outputs from this exact Python
                # and NumPy environment; tolerance pins above remain portable.
                if (
                    f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                    np.__version__,
                ) == EXACT_PIN_ENVIRONMENT:
                    packed = b"".join(struct.pack("<d", value) for value in exact_values)
                    assert hashlib.sha256(packed).hexdigest() == EXACT_EVALUATION_SHA256[key]
