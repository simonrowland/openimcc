"""Pins for published-core integrity and packaged numeric output."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from openimcc import ImccMalformedDatapackError, evaluate, load_datapack
import openimcc.kernel as kernel
import openimcc.model as model


PACK_DIR = Path("src/openimcc/data/packs")
PUBLISHED_CORE_SHA256 = (
    "ae979b68295f6e653fb1c37d6b381f06cb0be8d4c5b421e673420a6e4596357c"
)
LEGACY_PACK_SHA256 = {
    "imcc-sf04-v1.0.2.json": "87be28815642e8ec65f09edd791525fef8dfe10622e61772dee6047f4ad027b8",
    "imcc-sf04-ext-v4.json": "4c8e3c89cce921909b70005b6c28e0a5bd75f4487acf12f33e02be55bb730828",
}
LEGACY_BINDING_SHA256 = {
    "imcc-sf04-v1.0.2.json": "ae979b68295f6e653fb1c37d6b381f06cb0be8d4c5b421e673420a6e4596357c",
    "imcc-sf04-ext-v4.json": "de51a9bcf52724fd4b06e9cb639779f31189d86ecf304073fd72fc3019f29770",
}
D066_MANIFEST_SHA256 = {
    "imcc-sf04-d066-sio2-na2o-candidate.json": (
        "51179602bfd1bbc8be071bead0f40f9d8ff407227e5a4a904ab8a158fe471efd"
    ),
    "imcc-sf04-d066-v3.json": (
        "252f8705a3d04799ab9131836785992ffbedb6e01c98780c849498f4836804f2"
    ),
    "imcc-sf04-d066-v4.json": (
        "0879f5a8e94dfd844ed5cfcd900677b496a89e717a85f320b3e0b374ebcdc3b0"
    ),
    "imcc-sf04-d066-v5.json": (
        "13a185563870ec1a9c20d51bd25fb58df08c9bbbbdf7b2212b0f43ead64738cf"
    ),
    "imcc-sf04-d066-v2.json": (
        "ac88aa2272c50d1cbf300634737c5d0071bddee0cd0f0483a6530c4a4992c393"
    ),
    "imcc-sf04-d066-v1.json": (
        "02c88c04fe09f8f494adda8a76e1f95fee536d11b2b6d66639aaea097fce5e75"
    ),
    "imcc-sf04-d066-v1-kcaalsi2o7.json": (
        "45e12de67ee8effa56fe3935e12e66e0dda2ea73ce877dc598ccda3faaea453f"
    ),
    "imcc-sf04-d066-ext-v1.json": (
        "25125d24b9a7f3907773811c245b8137f40e46794caba5ce12f9ea26b7a49ce8"
    ),
}
D066_BINDING_SHA256 = D066_MANIFEST_SHA256
D066_DEFAULT_K_PRESSURE_BAR = 0.0005914655746591656
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


def test_d066_v5_changes_only_the_five_source_refit_a_values() -> None:
    previous = load_datapack(PACK_DIR / "imcc-sf04-d066-v4.json").kernel_datapack
    current = load_datapack(PACK_DIR / "imcc-sf04-d066-v5.json").kernel_datapack
    moved = {"Mg2SiO4", "CaSiO3", "CaAl2Si2O8", "Ca2MgSi2O7", "MgAl2O4"}
    changed = {
        name for i, name in enumerate(current.reactions)
        if current.A[i] != previous.A[i]
    }
    assert changed == moved
    np.testing.assert_array_equal(current.B, previous.B)
    expected_delta_A = {
        "Mg2SiO4": -1.106834313836493,
        "CaSiO3": -0.2765176335401777,
        "CaAl2Si2O8": -0.94161923030176,
        "Ca2MgSi2O7": -0.81455303243265,
        "MgAl2O4": -0.670271821871233,
    }
    for i, name in enumerate(current.reactions):
        if name not in moved:
            assert current.A[i] == previous.A[i]
        else:
            assert current.A[i] - previous.A[i] == pytest.approx(expected_delta_A[name])


def test_d066_source_refs_are_complete_and_do_not_change_numeric_results(
    tmp_path: Path,
) -> None:
    path = PACK_DIR / "imcc-sf04-d066-v3.json"
    content = json.loads(path.read_text(encoding="utf-8"))
    sources = content["sources"]
    expected_dois = {
        "FC87": "10.1016/0012-821X(87)90196-8",
        "SF04": "10.1016/j.icarus.2003.08.023",
        "Shornikov2017": "10.1134/S0036024417020303",
        "DarkenGurry1946": "10.1021/ja01209a030",
        "Masson1965": "10.1098/rspa.1965.0176",
    }
    assert {key: sources[key]["doi"] for key in expected_dois} == expected_dois

    for row in content["rows"]:
        name = row["complex"]
        if name == "MgAl2O4":
            expected_refs = ["FC87", "Shornikov2017"]
        elif name in {"FeTiO3", "Fe2SiO4", "FeAl2O4"}:
            expected_refs = ["FC87", "DarkenGurry1946", "Masson1965"]
        elif row["source"].startswith("SF04 "):
            expected_refs = ["SF04"]
        else:
            expected_refs = ["FC87"]
        assert row["source_refs"] == expected_refs
        assert all(ref in sources for ref in row["source_refs"])
        assert all(sources[ref].get("doi") for ref in row["source_refs"])

    without_refs = json.loads(json.dumps(content))
    for row in without_refs["rows"]:
        row.pop("source_refs")
    without_refs_path = tmp_path / "d066-without-source-refs.json"
    without_refs_path.write_text(
        json.dumps(without_refs, ensure_ascii=False),
        encoding="utf-8",
    )

    with_refs = load_datapack(path)
    without_refs_pack = load_datapack(without_refs_path)
    with_refs_result = evaluate(
        COMPOSITIONS[0], 2000.0, with_refs, allow_out_of_envelope=True
    )
    without_refs_result = evaluate(
        COMPOSITIONS[0], 2000.0, without_refs_pack, allow_out_of_envelope=True
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
        np.testing.assert_array_equal(
            getattr(with_refs_result, field),
            getattr(without_refs_result, field),
        )
    assert with_refs_result.D == without_refs_result.D


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
    # This intentionally moves with the v2 MgAl2O4 and FeO(l) corrections.
    # Keep the portable-golden convention: a tight relative bound with no
    # absolute allowance; pack and digest pins stay exact.
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
                    assert float(actual) == pytest.approx(
                        expected["scalars"][name],
                        rel=EVALUATION_RTOL,
                        abs=0.0,
                    ), f"{key} {name} moved"


def test_modified_published_core_refuses_on_hash_before_band_field_validation(
    tmp_path: Path,
) -> None:
    # A modified published core that also lacks the band fields must be
    # refused by the published-identity gate (canonical hash), not by
    # field-level band validation. (D066 packs have no published-hash gate;
    # their rows are checked structurally, so the ordering applies here.)
    manifest = json.loads(
        (PACK_DIR / "imcc-sf04-v1.0.2.json").read_text(encoding="utf-8")
    )
    manifest["rows"][0]["A"] += 0.5
    for row in manifest["rows"]:
        for field in ("fit_residual", "accuracy_band", "band_basis", "independent_check"):
            row.pop(field, None)
    altered_path = tmp_path / "altered.json"
    altered_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ImccMalformedDatapackError, match="hash mismatch"):
        load_datapack(altered_path)
