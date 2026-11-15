"""Focused synthetic and in-domain checks for the melt/crystal solver."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import median
from time import process_time

import numpy as np
import pytest

from openimcc import diagnostics
from openimcc._constants import R_J_MOL_K
from openimcc.equilibrium import (
    CrystalCandidate,
    _CandidateState,
    _active_residual,
    _solve_active_equalities,
    equilibrate,
)
from openimcc.gas import species_thermo
from openimcc.kernel import ImccDatapack, label_research_datapack
from openimcc.model import evaluate, load_datapack


@pytest.fixture(scope="module")
def ideal_pack():
    parents = tuple(load_datapack().parent_oxides)
    raw = ImccDatapack(
        reactions=(),
        nu=np.zeros((len(parents), 0)),
        A=np.zeros(0),
        B=np.zeros(0),
        domains=(),
        version="synthetic-ideal-control",
        parent_oxides=parents,
    )
    return label_research_datapack(
        raw, model_id="synthetic-ideal-control", coverage="synthetic",
    )


@pytest.fixture(scope="module")
def gas_pack():
    return diagnostics._crystal_thermo_context(None)[0]


def _candidate_from_log_activities(phase, stoichiometry, log_activities, temperature, gas_pack):
    gibbs = math.fsum(
        float(nu) * (
            species_thermo(parent, "l", temperature, gas_pack).G_J_mol
            + R_J_MOL_K * temperature * float(log_activities[parent])
        )
        for parent, nu in stoichiometry.items()
    )
    return CrystalCandidate(
        phase_id=phase,
        formula=phase,
        stoichiometry=stoichiometry,
        G_crystal_J_mol=gibbs,
        source_authority="external_datapack",
    )


def _one_parent_candidate(phase, temperature, delta_rt, gas_pack):
    liquid_g = species_thermo("SiO2", "l", temperature, gas_pack).G_J_mol
    return CrystalCandidate(
        phase,
        {"SiO2": 1.0},
        liquid_g + delta_rt * R_J_MOL_K * temperature,
        formula="SiO2",
    )


def test_binary_eutectic_singular_jacobian_is_consistent(ideal_pack, gas_pack):
    temperature = 2000.0
    parents = ("CaO", "SiO2")
    candidates = (
        _candidate_from_log_activities(
        "solid-CaO", {"CaO": 1.0}, {"CaO": math.log(0.5)}, temperature, gas_pack
        ),
        _candidate_from_log_activities(
        "solid-SiO2", {"SiO2": 1.0}, {"SiO2": math.log(0.5)}, temperature, gas_pack
        ),
    )
    states = tuple(_CandidateState(item, item.source_authority, None, True) for item in candidates)
    A = np.eye(2)
    b = np.ones(2)
    liquid_g = np.asarray([
        species_thermo(parent, "l", temperature, gas_pack).G_J_mol
        for parent in parents
    ])
    # At the eutectic, both liquid activities are 0.5 and the melt fraction is
    # positive. The extent direction that consumes both solids equally leaves
    # those activities unchanged, so the active Jacobian has rank one.
    start = np.asarray([0.25, 0.35])
    solved, converged, reason, _solver_diagnostics = _solve_active_equalities(
        start, [0, 1], A, b, states, parents, liquid_g,
        temperature, ideal_pack, 1.0e-8, 1.0e-10, 1.0e-12, 100,
    )
    assert converged, reason
    at_solution = _active_residual(
        solved, [0, 1], solved, A, b, states, parents, liquid_g,
        temperature, ideal_pack, 1.0e-12, 100, 1.0e-10,
    )
    assert at_solution is not None
    assert np.linalg.norm(at_solution[0], ord=np.inf) <= 1.0e-8
    jac = np.empty((2, 2))
    for j in range(2):
        plus = solved.copy()
        minus = solved.copy()
        plus[j] += 1.0e-5
        minus[j] -= 1.0e-5
        probe_plus = _active_residual(
            plus, [0, 1], solved, A, b, states, parents, liquid_g,
            temperature, ideal_pack, 1.0e-12, 100, 1.0e-10,
        )
        probe_minus = _active_residual(
            minus, [0, 1], solved, A, b, states, parents, liquid_g,
            temperature, ideal_pack, 1.0e-12, 100, 1.0e-10,
        )
        assert probe_plus is not None and probe_minus is not None
        jac[:, j] = (probe_plus[0] - probe_minus[0]) / 2.0e-5
    singular_values = np.linalg.svd(jac, compute_uv=False)
    # Relative rank is the physical check: the lever-rule direction is null,
    # while an absolute cutoff would vary with the finite-difference step units.
    assert singular_values[-1] <= 1.0e-4 * singular_values[0]

    liquid = b - A @ solved
    assert liquid.sum() > 1.0
    assert liquid[0] / liquid.sum() == pytest.approx(0.5, abs=2.0e-6)


def test_peritectic_replacement_drops_resolved_parent_phase(ideal_pack, gas_pack):
    temperature = 2000.0
    parents = ("CaO", "SiO2")
    parent_phase = _candidate_from_log_activities(
        "CaO-solid", {"CaO": 1.0}, {"CaO": math.log(0.25)}, temperature, gas_pack
    )
    compound = _candidate_from_log_activities(
        "CaSiO3-solid",
        {"CaO": 1.0, "SiO2": 1.0},
        {"CaO": math.log((1.0 - math.sqrt(1.0 - 0.72)) / 2.0),
         "SiO2": math.log(1.0 - (1.0 - math.sqrt(1.0 - 0.72)) / 2.0)},
        temperature,
        gas_pack,
    )
    result = equilibrate(
        {"CaO": 1.0, "SiO2": 2.0}, temperature,
        candidates=(parent_phase, compound), pack=ideal_pack,
    )
    assert result.phase_presence["CaO-solid"] == "absent"
    assert result.phase_presence["CaSiO3-solid"] == "present"
    assert result.kkt_recheck["present_affinity_inf_ln"] <= 1.0e-8
    assert result.kkt_recheck["absent_affinity_violation_inf_ln"] <= 1.0e-8
    decisions = result.receipt.solver["deterministic_tie_breaks"]
    assert any("replacement:CaO-solid->CaSiO3-solid" in item for item in decisions)


def test_competing_polymorphs_switch_and_share_at_crossing(ideal_pack, gas_pack):
    crossing = 2000.0
    results = {}
    for temperature in (1900.0, crossing, 2100.0):
        liquid_g = species_thermo("SiO2", "l", temperature, gas_pack).G_J_mol
        delta = 0.01 * (temperature - crossing) * R_J_MOL_K * temperature
        alpha = CrystalCandidate("poly-alpha", {"SiO2": 1.0}, liquid_g + delta)
        beta = CrystalCandidate("poly-beta", {"SiO2": 1.0}, liquid_g - delta)
        results[temperature] = equilibrate(
            {"SiO2": 1.0}, temperature, candidates=(alpha, beta), pack=ideal_pack,
        )
    assert results[1900.0].phase_presence["poly-alpha"] == "present"
    assert results[1900.0].phase_presence["poly-beta"] == "absent"
    assert results[2100.0].phase_presence["poly-beta"] == "present"
    assert results[2100.0].phase_presence["poly-alpha"] == "absent"
    at_crossing = results[crossing]
    assert at_crossing.refusal is not None
    assert at_crossing.refusal.code == "incipient_liquid_unresolved"
    assert set(at_crossing.saturated_candidates) == {"poly-alpha", "poly-beta"}
    assert at_crossing.phase_amounts_unique is False
    assert at_crossing.phase_presence["poly-alpha"] == "unresolved"
    assert at_crossing.phase_presence["poly-beta"] == "unresolved"


def test_exact_zero_parent_excludes_only_incompatible_crystals(ideal_pack, gas_pack):
    temperature = 2000.0
    silica = _one_parent_candidate("solid-SiO2", temperature, -5.0, gas_pack)
    incompatible = CrystalCandidate(
        "lower-G-CaSiO3", {"CaO": 1.0, "SiO2": 1.0},
        silica.G_crystal_J_mol - 100_000.0,
    )
    result = equilibrate(
        {"SiO2": 1.0}, temperature,
        candidates=(silica, incompatible), pack=ideal_pack,
    )
    assert result.phase_presence["solid-SiO2"] == "present"
    assert result.phase_presence["lower-G-CaSiO3"] == "absent"


def test_active_set_cycle_refuses_deterministically(monkeypatch, ideal_pack, gas_pack):
    temperature = 2000.0
    phase = _candidate_from_log_activities(
        "CaO-solid", {"CaO": 1.0}, {"CaO": math.log(0.25)}, temperature, gas_pack
    )
    monkeypatch.setattr("openimcc.equilibrium._MAX_ACTIVE_RESTARTS", 3)
    monkeypatch.setattr(
        "openimcc.equilibrium._solve_active_equalities",
        lambda amounts, *_args, **_kwargs: (amounts, True, "repeated-state", {}),
    )
    first = equilibrate(
        {"CaO": 1.0, "SiO2": 1.0}, temperature,
        candidates=(phase,), pack=ideal_pack,
    )
    second = equilibrate(
        {"CaO": 1.0, "SiO2": 1.0}, temperature,
        candidates=(phase,), pack=ideal_pack,
    )
    assert first.refusal is not None and first.refusal.code == "active_set_cycle"
    assert second.refusal is not None and second.refusal.code == first.refusal.code
    assert first.active_set_history == ((), ("CaO-solid",), ("CaO-solid",))
    assert first.active_set_history == second.active_set_history


def test_single_parent_no_associate_control_has_q_one_at_unit_activity(ideal_pack, gas_pack):
    temperature = 2000.0
    candidate = _one_parent_candidate("pure-SiO2", temperature, 0.0, gas_pack)
    result = equilibrate(
        {"SiO2": 1.0}, temperature, candidates=(candidate,), pack=ideal_pack,
    )
    assert result.identifiable_activities["SiO2"] == pytest.approx(1.0, abs=1.0e-12)
    assert result.logQ == pytest.approx(0.0, abs=1.0e-12)
    assert result.logQ_interval[0] <= 0.0 <= result.logQ_interval[1]
    assert result.refusal.code == "incipient_liquid_unresolved"


def test_fully_solid_branch_certifies_logq_below_zero(ideal_pack, gas_pack):
    temperature = 2000.0
    candidate = _one_parent_candidate("solid-SiO2", temperature, -5.0, gas_pack)
    result = equilibrate(
        {"SiO2": 1.0}, temperature, candidates=(candidate,), pack=ideal_pack,
    )
    assert result.liquid_parent_mol is None
    assert result.logQ_interval[1] < 0.0
    assert result.phase_presence["solid-SiO2"] == "present"
    assert result.kkt_recheck["absent_affinity_violation_inf_ln"] <= 1.0e-8


def test_default_pack_positive_liquid_state_passes_independent_kkt(gas_pack):
    temperature = 1900.0
    pack = load_datapack()
    target_liquid = {"CaO": 1.0, "Al2O3": 1.0, "SiO2": 1.0}
    target = evaluate(
        target_liquid, temperature, pack, basis_type="mol",
        allow_out_of_envelope=True,
    )
    liquid_g = species_thermo("CaO", "l", temperature, gas_pack).G_J_mol
    phase = CrystalCandidate(
        "synthetic-CaO-solid",
        {"CaO": 1.0},
        liquid_g + R_J_MOL_K * temperature * (math.log(target.activity("CaO")) - 0.1),
        formula="CaO",
    )
    result = equilibrate(
        {"CaO": 1.0, "Al2O3": 1.0, "SiO2": 1.0},
        temperature,
        candidates=(phase,),
        pack=pack,
    )
    assert result.liquid_parent_mol is not None, result.refusal
    assert 0.0 < result.liquid_parent_mol["CaO"] < 1.0
    assert result.melt is not None
    assert result.kkt_recheck["balance_relative_inf"] <= 1.0e-10
    assert result.kkt_recheck["present_affinity_inf_ln"] <= 1.0e-8
    assert result.kkt_recheck["inner_balance_relative_inf"] <= 1.0e-12


def test_refusals_for_nonconvex_ferric_out_of_domain_and_nonfinite(
    tmp_path, ideal_pack, gas_pack
):
    source = Path(__file__).parents[1] / "src/openimcc/data/packs/imcc-sf04-d066-v4.json"
    data = json.loads(source.read_text())
    data["model_id"] = "IMCC-SF04-D066-EXCESS-CANDIDATE"
    data["excess_gibbs"] = {
        "terms": [{
            "basis": "analytical_parent",
            "components": ["CaO", "Al2O3"],
            "W_H_J_mol": 1900.0,
            "W_S_J_mol_K": 1.0,
            "provenance": {"source": "synthetic test coefficient"},
            "calibration_role": "test",
            "valid_composition": {"CaO": [0.0, 1.0], "Al2O3": [0.0, 1.0]},
            "T_range_K": [300.0, 5000.0],
            "fit_status": "empirical-fit",
        }],
    }
    excess_path = tmp_path / "excess.json"
    excess_path.write_text(json.dumps(data))
    excess = load_datapack(excess_path)
    refusal = equilibrate(
        {"CaO": 1.0}, 1900.0, candidates=(), pack=excess,
    )
    assert refusal.refusal.code == "unsupported_nonconvex_model"

    ferric = CrystalCandidate("hematite", {"FeO": 1.0}, 0.0, formula="Fe2O3")
    refusal = equilibrate(
        {"FeO": 1.0}, 1900.0, candidates=(ferric,), pack=ideal_pack,
    )
    assert refusal.refusal.code == "unsupported_redox_phase"

    low_candidate = _one_parent_candidate("low-SiO2", 1600.0, -5.0, gas_pack)
    refusal = equilibrate(
        {"SiO2": 1.0}, 1600.0, candidates=(low_candidate,), pack=ideal_pack,
    )
    assert refusal.labels.validity == "OUT_OF_DOMAIN"
    assert refusal.labels.out_of_domain_axes == ("temperature_K",)
    assert refusal.crystal_only_minimum is True

    refusal = equilibrate(
        {"CaO": float("nan")}, 1900.0, candidates=(), pack=ideal_pack,
    )
    assert refusal.refusal.code == "nonfinite_input"


def test_cpu_median_per_equilibrium_point(ideal_pack, gas_pack):
    temperature = 2000.0
    solid_candidate = _one_parent_candidate(
        "solid-SiO2", temperature, -5.0, gas_pack
    )
    pack = load_datapack()
    target = evaluate(
        {"CaO": 1.0, "Al2O3": 1.0, "SiO2": 1.0},
        1900.0,
        pack,
        basis_type="mol",
        allow_out_of_envelope=True,
    )
    liquid_g = species_thermo("CaO", "l", 1900.0, gas_pack).G_J_mol
    liquid_candidate = CrystalCandidate(
        "synthetic-CaO-solid",
        {"CaO": 1.0},
        liquid_g + R_J_MOL_K * 1900.0 * (math.log(target.activity("CaO")) - 0.1),
        formula="CaO",
    )
    cases = (
        (
            "crystal-only",
            lambda: equilibrate(
                {"SiO2": 1.0}, temperature,
                candidates=(solid_candidate,), pack=ideal_pack,
            ),
        ),
        (
            "positive-liquid",
            lambda: equilibrate(
                {"CaO": 1.0, "Al2O3": 1.0, "SiO2": 1.0},
                1900.0,
                candidates=(liquid_candidate,),
                pack=pack,
            ),
        ),
    )
    for name, solve in cases:
        elapsed = []
        for _ in range(5):
            started = process_time()
            result = solve()
            elapsed.append(process_time() - started)
            assert result.kkt_recheck, result.refusal
        print(
            f"CPU median per equilibrium point ({name}, n=5): "
            f"{median(elapsed):.6f} s; design projection: 0.2–1 CPU s"
        )
