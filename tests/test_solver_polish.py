"""Regressions for the bounded analytic Newton polish in SF04 solves."""

from __future__ import annotations

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

    solver_path = result.convergence.solver_path
    assert solver_path in {
        "direct",
        "continuation",
        "composition-continuation",
        "temperature-continuation",
        "relative-log-x",
    }
    if solver_path == "direct":
        assert result.convergence.continuation_stages == 0
    elif solver_path != "relative-log-x":
        assert result.convergence.continuation_stages > 0
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


def test_v1_depleted_feo_1710k_is_certified() -> None:
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

    solver_path = result.convergence.solver_path
    assert solver_path in {
        "direct",
        "continuation",
        "composition-continuation",
        "temperature-continuation",
        "relative-log-x",
    }
    if solver_path == "direct":
        assert result.convergence.continuation_stages == 0
    elif solver_path != "relative-log-x":
        assert result.convergence.continuation_stages > 0
    assert result.convergence.residual_inf <= 1.0e-12
    assert result.parent_activity[result.parent_oxides.index("FeO")] == pytest.approx(
        1.5838876114056108e-12, rel=2.0e-5, abs=0.0
    )
    assert activity_shift <= 1.0e-10


_ORDINARY_ROOT_EXPECTATIONS = {
    "direct": {
        "basis": 1.78583892132001,
        "parent_mol": (
            0.7489477284415397,
            0.6202796716983754,
            0.20878570235510274,
            0.10699497480268344,
            0.049038213518266247,
            0.025041945258307666,
            0.016134523200637636,
            0.010616162045097457,
        ),
        "parent_x": (
            0.4193814568045991,
            0.347332373761847,
            0.11691183334764478,
            0.0599130042051036,
            0.027459483009822327,
            0.014022510630352826,
            0.00903470240681715,
            0.00594463583381332,
        ),
        "parent_activity": (
            0.1325387755112024,
            0.04656106257425901,
            0.160701142016943,
            0.00014968663070267257,
            0.005465596509698274,
            0.0034967697333541745,
            1.954108483328543e-08,
            1.1076696075801892e-11,
        ),
        "complex_x": (
            0.17194467065280078,
            0.23706412487324355,
            0.006571438535325022,
            0.005066288502001032,
            4.873679339425474e-05,
            0.0028557014369819714,
            1.6036094612099636e-07,
            0.0011293292594590897,
            8.259583492541143e-05,
            1.2166316966454675e-32,
            0.041833670067914386,
            0.002821671737450364,
            0.06171515210212003,
            0.003602258017567325,
            0.00037613742509195797,
            0.0014667295690272437,
            0.00020450737763532501,
            0.004973853636549876,
            0.010572702147182738,
            0.028568068602858893,
            0.010706722099607621,
            1.3092072962185533e-10,
            7.998999507837905e-05,
            0.0010430175693927994,
            0.0006616595390308998,
            0.019281289378500096,
            0.011623100168074166,
            0.0012396329707520897,
            0.00015035424858780196,
            0.0011027549772492603,
            6.371668807869071e-06,
            9.585088950567317e-06,
            0.0059869815294837,
            0.002747222445938117,
            0.0002551907711896561,
            0.0152950489858118,
            2.2897183014825984e-07,
        ),
        "parent_gamma": (
            0.3160339432292919,
            0.13405333361232896,
            1.3745498416664796,
            0.0024983996828174698,
            0.19904222187079107,
            0.24936830682696307,
            2.162891919775982e-06,
            1.8633094415636388e-09,
        ),
        "D": 2.045273747084461,
    },
    "continuation": {
        "basis": 1.3163170729362774,
        "parent_mol": (
            0.7689196678666474,
            0.06699020454342454,
            0.17955570402538837,
            0.05706398656143117,
            0.1265185908771269,
            0.06886534946034607,
            0.04840356960191291,
            0.0,
        ),
        "parent_x": (
            0.5841447198974913,
            0.05089214895161321,
            0.13640763894739866,
            0.04335124700171204,
            0.0961155890768057,
            0.05231668788336063,
            0.03677196824161842,
            0.0,
        ),
        "parent_activity": (
            0.4484457419335304,
            0.003929924990798973,
            0.11023673848118967,
            4.2478048473777986e-05,
            0.0312204166986516,
            0.030307391935336592,
            1.2417036098961474e-09,
            0.0,
        ),
        "complex_x": (
            0.0019035721233552445,
            0.05305551439153717,
            0.003018089034323436,
            0.0026386839289411,
            0.00022943687035577401,
            0.0001183162402202797,
            0.0001282624500767952,
            0.000638749427944881,
            0.00027672089060851873,
            3.4273198168701247e-37,
            0.0224276387300125,
            0.029986940071204126,
            0.006962846300411438,
            5.6357453936264204e-05,
            0.00019083565967504057,
            0.001723976348212127,
            2.3093702270262163e-05,
            0.014426145193673173,
            0.04326608987433151,
            0.03287132128027311,
            0.023122659189844904,
            1.2105383321684366e-07,
            0.008245090679494043,
            5.251378272269329e-05,
            9.654070818879653e-05,
            0.01889699593410886,
            0.10811936542320827,
            0.000542285908774767,
            1.6444514438127382e-05,
            0.0027826995043390462,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        "parent_gamma": (
            0.767696303087745,
            0.07722065331796919,
            0.8081419730730698,
            0.000979857591457529,
            0.3248215715944211,
            0.5793063965155157,
            3.3767667853329385e-08,
            0.0,
        ),
        "D": 1.7767929419214863,
    },
}


@pytest.mark.parametrize(
    (
        "composition",
        "temperature",
        "basis_type",
        "path",
        "continuation_stages",
    ),
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
            0,
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
            5,
        ),
    ),
)
def test_ordinary_v1_root_keeps_its_pre_polish_bits(
    composition: dict[str, float],
    temperature: float,
    basis_type: str,
    path: str,
    continuation_stages: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pack = load_datapack(V1_PACK_PATH)
    solve_calls = 0
    original_solve = kernel.np.linalg.solve

    def tracked_solve(matrix: np.ndarray, residual: np.ndarray) -> np.ndarray:
        nonlocal solve_calls
        solve_calls += 1
        return original_solve(matrix, residual)

    if path == "direct":
        monkeypatch.setattr(kernel.np.linalg, "solve", tracked_solve)
    result = evaluate(
        composition,
        temperature,
        pack,
        basis_type=basis_type,
        allow_extrapolation=True,
        allow_out_of_envelope=True,
    )

    assert result.convergence.solver_path == path
    assert result.convergence.continuation_stages == continuation_stages
    if path == "direct":
        assert solve_calls == 0
    expected = _ORDINARY_ROOT_EXPECTATIONS[path]
    assert result.temperature_K == temperature
    assert result.basis == expected["basis"]
    np.testing.assert_array_equal(result.parent_mol, expected["parent_mol"])
    np.testing.assert_array_equal(result.parent_x, expected["parent_x"])
    assert result.parent_activity == pytest.approx(
        expected["parent_activity"],
        rel=1e-12,
        abs=0.0,
    )
    assert result.complex_x == pytest.approx(
        expected["complex_x"],
        rel=1e-12,
        abs=0.0,
    )
    assert result.parent_gamma == pytest.approx(
        expected["parent_gamma"],
        rel=1e-12,
        abs=0.0,
    )
    assert result.D == pytest.approx(
        expected["D"],
        rel=1e-12,
        abs=0.0,
    )


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
    solver_path = result.convergence.solver_path
    assert solver_path in {
        "continuation",
        "composition-continuation",
        "temperature-continuation",
    }
    assert result.convergence.continuation_stages > 0
    assert result.convergence.residual_inf <= 1.0e-12
    assert result.parent_activity[result.parent_oxides.index("FeO")] == pytest.approx(
        1.5838876114056108e-12, rel=2.0e-5, abs=0.0
    )
