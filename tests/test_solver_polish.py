"""Regressions for the bounded analytic Newton polish in SF04 solves."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from openimcc import evaluate, load_datapack
from openimcc import kernel


ROOT = Path(__file__).resolve().parents[1]
V1_PACK_PATH = (
    ROOT / "src" / "openimcc" / "data" / "packs" / "imcc-sf04-d066-v1.json"
)


def _depleted_feo_composition(pack: object) -> dict[str, float]:
    return {
        name: (1.0e-12 if name == "FeO" else (1.0 - 1.0e-12) / 7.0)
        for name in pack.parent_oxides
    }


def _fingerprint(result: object) -> str:
    arrays = (
        "parent_mol",
        "parent_x",
        "parent_x_star",
        "parent_activity",
        "parent_gamma",
        "complex_x",
        "species_x",
    )
    payload = {
        name: getattr(result, name).tobytes().hex()
        for name in arrays
    }
    convergence = result.convergence
    payload.update(
        {
            "T": result.temperature_K.hex(),
            "basis": result.basis.hex(),
            "D": result.D.hex(),
            "ri": convergence.residual_inf.hex(),
            "rl": convergence.residual_l2.hex(),
            "disp": convergence.total_displacement.hex(),
            "it": convergence.iterations,
            "status": convergence.status,
            "path": convergence.solver_path,
            "stages": convergence.continuation_stages,
        }
    )
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_corrected_mgal2o4_depleted_feo_trace_root_is_polished(
    tmp_path: Path,
) -> None:
    # Keep this solver commit independent of the later pack commit: substitute
    # the corrected v2 MgAl2O4 A/B row into the unchanged v1 test pack.
    pack_data = json.loads(V1_PACK_PATH.read_text(encoding="utf-8"))
    mgal2o4 = next(
        row for row in pack_data["rows"] if row.get("complex") == "MgAl2O4"
    )
    mgal2o4["A"] = -1.7842579418420448
    mgal2o4["B"] = 7601.933123955643
    test_pack_path = tmp_path / "d066-v2-mgal2o4-row.json"
    test_pack_path.write_text(json.dumps(pack_data), encoding="utf-8")
    pack = load_datapack(test_pack_path)

    result = evaluate(
        _depleted_feo_composition(pack),
        1710.0,
        pack,
        basis_type="mol",
        allow_extrapolation=True,
        allow_out_of_envelope=True,
    )

    assert result.convergence.solver_path == "direct"
    assert result.convergence.continuation_stages == 0
    assert result.convergence.residual_inf <= 1.0e-12
    assert result.parent_activity[result.parent_oxides.index("FeO")] == pytest.approx(
        1.6045765472734952e-12, rel=2.0e-12
    )


def test_highland_normal_run_trace_root_is_polished_and_certified() -> None:
    # Parent-oxide projection of the captured normal-run bulk, with ferric
    # iron already folded into the FeO-equivalent input. Keep the trace Na2O
    # amount: this case fails at the pre-polish solver revision.
    composition = {
        "SiO2": 278.41036377176334,
        "MgO": 1462.4326109015108,
        "FeO": 0.025620374070120855,
        "CaO": 2780.0948143537,
        "Al2O3": 2614.2535001541282,
        "TiO2": 56.660319985140646,
        "Na2O": 4.2184025412391715e-11,
        "K2O": 8.46987830123445,
    }
    pack = load_datapack()
    result = evaluate(
        composition,
        2473.15,
        pack,
        basis_type="mol",
        enable_sp_extension=False,
        allow_extrapolation=False,
        allow_out_of_envelope=False,
    )

    assert result.convergence.status == "converged"
    assert result.convergence.residual_inf <= 1.0e-12

    active = result.parent_x > 0.0
    balance = (
        result.parent_x_star
        + pack.kernel_datapack.nu @ result.complex_x
        - result.D * result.parent_x
    )
    relative_inventory_error = (
        result.parent_x_star + pack.kernel_datapack.nu @ result.complex_x
    ) / (result.D * result.parent_x) - 1.0
    assert np.linalg.norm(balance, ord=np.inf) <= 1.0e-12
    assert np.max(np.abs(relative_inventory_error[active])) <= 1.0e-12
    assert np.all(
        result.parent_activity[active]
        <= result.D * result.parent_x[active]
    )


def test_v1_depleted_feo_1710k_uses_seven_continuation_stages() -> None:
    pack = load_datapack(V1_PACK_PATH)
    result = evaluate(
        _depleted_feo_composition(pack),
        1710.0,
        pack,
        basis_type="mol",
        allow_extrapolation=True,
        allow_out_of_envelope=True,
    )

    baseline_activities = np.array(
        [
            0.00016125002005635012,
            0.18269196648457398,
            1.5838876114056108e-12,
            0.01938238394154021,
            0.00047289701462967557,
            0.0001556742804076289,
            2.745176400003109e-05,
            1.3334941867533056e-05,
        ]
    )
    activity_shift = np.max(np.abs(result.parent_activity - baseline_activities))

    assert result.convergence.solver_path == "composition-continuation"
    assert result.convergence.continuation_stages == 7
    assert result.convergence.residual_inf <= 1.0e-12
    assert activity_shift == pytest.approx(2.7755575615628914e-17, abs=1.0e-30)


@pytest.mark.parametrize(
    ("composition", "temperature", "basis_type", "path", "digest"),
    (
        (
            {
                "SiO2": 45.0,
                "MgO": 25.0,
                "FeO": 15.0,
                "CaO": 6.0,
                "Al2O3": 5.0,
                "TiO2": 2.0,
                "Na2O": 1.0,
                "K2O": 1.0,
            },
            2000.0,
            "wt",
            "direct",
            "2e3d4ec280ce223d0b425d90688e35a6b60d4f070402a4a75220b2b573510214",
        ),
        (
            {
                "SiO2": 46.2,
                "TiO2": 5.5,
                "Al2O3": 12.9,
                "FeO": 12.9,
                "MgO": 2.7,
                "CaO": 3.2,
                "Na2O": 3.0,
            },
            2200.0,
            "wt",
            "continuation",
            "d32ff4d9db0ac6a0e8e557aff3d7c160649194ab82a6407b148c0a6180f06e47",
        ),
    ),
)
def test_ordinary_v1_root_keeps_its_pre_polish_bits(
    composition: dict[str, float],
    temperature: float,
    basis_type: str,
    path: str,
    digest: str,
) -> None:
    pack = load_datapack(V1_PACK_PATH)
    result = evaluate(
        composition,
        temperature,
        pack,
        basis_type=basis_type,
        allow_extrapolation=True,
        allow_out_of_envelope=True,
    )

    assert result.convergence.solver_path == path
    assert _fingerprint(result) == digest


def test_polish_linear_solve_failure_uses_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pack = load_datapack(V1_PACK_PATH)
    original_solve = kernel.np.linalg.solve
    solve_calls = 0

    def fail_first_polish_solve(matrix: np.ndarray, residual: np.ndarray) -> np.ndarray:
        nonlocal solve_calls
        solve_calls += 1
        if solve_calls == 1:
            raise kernel.np.linalg.LinAlgError("forced polish failure")
        return original_solve(matrix, residual)

    monkeypatch.setattr(kernel.np.linalg, "solve", fail_first_polish_solve)
    result = evaluate(
        _depleted_feo_composition(pack),
        1710.0,
        pack,
        basis_type="mol",
        allow_extrapolation=True,
        allow_out_of_envelope=True,
    )

    assert solve_calls > 1
    assert result.convergence.solver_path == "composition-continuation"
    assert result.convergence.continuation_stages == 7
    assert result.convergence.residual_inf <= 1.0e-12
