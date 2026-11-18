"""Focused synthetic and in-domain checks for the melt/crystal solver."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from statistics import median
from time import process_time

import numpy as np
import pytest
from scipy.optimize import brentq as independent_brentq

from openimcc import diagnostics
from openimcc import gas as gas_module
from openimcc._constants import R_J_MOL_K
from openimcc.declared_assemblage import EquilibriumRefusal
from openimcc.equilibrium import (
    CrystalCandidate,
    _CandidateState,
    _PURE_RESOURCE,
    _load_json,
    _inner_activity_uncertainty,
    _active_residual,
    _parse_bulk,
    _lp_solid_minimum,
    _minimize_logq_on_dual_face,
    _positive_branch,
    _solve_active_equalities,
    equilibrate,
    equilibrate_sweep,
    evaluate_janaf_crystal,
    phase_state_contract_version,
    phase_boundary_brackets,
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


def _install_synthetic_default_candidates(monkeypatch, candidates, gas_pack):
    from openimcc import equilibrium as equilibrium_module

    states = tuple(
        _CandidateState(
            candidate, candidate.source_authority, candidate.temperature_range_K, True
        )
        for candidate in candidates
    )

    def synthetic_defaults(_temperature, selected_gas_pack=None):
        return (
            states, "synthetic-crystal-data", "synthetic-pure-data",
            gas_pack if selected_gas_pack is None else selected_gas_pack, (),
        )

    monkeypatch.setattr(
        equilibrium_module, "_default_candidate_states", synthetic_defaults
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
    assert converged, (reason, solver_diagnostics)
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


def test_rank_zero_inconsistent_affinities_get_typed_refusal(monkeypatch, ideal_pack, gas_pack):
    temperature = 2000.0
    candidates = (
        _candidate_from_log_activities(
            "solid-CaO", {"CaO": 1.0}, {"CaO": math.log(0.5)}, temperature, gas_pack
        ),
        _candidate_from_log_activities(
            "solid-SiO2", {"SiO2": 1.0}, {"SiO2": math.log(0.5)}, temperature, gas_pack
        ),
    )
    states = tuple(_CandidateState(item, item.source_authority, None, True) for item in candidates)

    def inconsistent_residual(active_n, active, full_n, *_args, **_kwargs):
        return np.asarray([1.0e-4, -1.0e-4]), full_n.copy(), None

    monkeypatch.setattr(
        "openimcc.equilibrium._active_residual", inconsistent_residual
    )
    solved, converged, reason, residuals = _solve_active_equalities(
        np.asarray([0.25, 0.35]), [0, 1], np.eye(2), np.ones(2), states,
        ("CaO", "SiO2"), np.zeros(2), temperature, ideal_pack,
        1.0e-8, 1.0e-10, 1.0e-12, 100,
    )
    assert not converged
    assert reason == "singular_inconsistent_affinity"
    assert residuals["active_jacobian_rank"] == 0.0
    assert residuals["rank_inconsistent_affinity_inf_ln"] == pytest.approx(1.0e-4)


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
    assert at_crossing.refusal.code == "missing_phase_data"
    assert set(at_crossing.saturated_candidates) == {"poly-alpha", "poly-beta"}
    assert at_crossing.phase_amount_ranges_mol["poly-alpha"] == pytest.approx((0.0, 1.0))
    assert at_crossing.phase_amount_ranges_mol["poly-beta"] == pytest.approx((0.0, 1.0))
    assert at_crossing.phase_amounts_unique is False
    assert at_crossing.liquid_parent_mol is None
    assert at_crossing.liquid_mass_fraction is None
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
    assert result.phase_presence["lower-G-CaSiO3"] == "infeasible"


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


def test_replacement_preserves_injected_resolution_error(monkeypatch, ideal_pack, gas_pack):
    from openimcc import equilibrium as equilibrium_module

    original = equilibrium_module._active_residual

    def inject_tighter_activity_difference(*args, **kwargs):
        result = original(*args, **kwargs)
        probe_tol = args[10]
        if result is not None and probe_tol < 1.0e-12:
            residual, trial, melt = result
            return residual + np.full_like(residual, 1.0e-6), trial, melt
        return result

    monkeypatch.setattr(
        "openimcc.equilibrium._active_residual",
        inject_tighter_activity_difference,
    )
    temperature = 2000.0
    candidate = _candidate_from_log_activities(
        "injected-phase", {"CaO": 1.0}, {"CaO": math.log(0.25)},
        temperature, gas_pack,
    )
    state = _CandidateState(candidate, candidate.source_authority, None, True)
    parents = ("CaO", "SiO2")
    liquid_g = np.asarray([
        species_thermo(parent, "l", temperature, gas_pack).G_J_mol
        for parent in parents
    ])
    result = _positive_branch(
        bulk={"CaO": 1.0, "SiO2": 1.0}, parents=parents, states=(state,),
        A=np.asarray([[1.0], [0.0]]), g=np.asarray([candidate.G_crystal_J_mol]),
        liquid_gibbs=liquid_g, temperature_K=temperature,
        pack=ideal_pack, gas_pack=gas_pack, candidates_all=(state,),
        aliases={}, rejected_polymorphs={}, crystal_digest="synthetic-crystals",
        pure_digest="synthetic-pure", explicit_candidates=False, missing=(),
        balance_tol=1.0e-10, affinity_tol_ln=1.0e-8,
        inner_tol=1.0e-12, max_iter=100,
    )
    assert result.refusal is not None
    assert result.refusal.code == "resolution_unresolved", (
        result.refusal.residuals, result.active_set_history,
        result.receipt.solver.get("outer_error_budget_history"),
    )
    assert result.refusal.residuals["inner_correction_error_ln"] == pytest.approx(
        1.0e-6, rel=1.0e-3
    )


def test_single_parent_no_associate_control_has_q_one_at_unit_activity(ideal_pack, gas_pack):
    temperature = 2000.0
    candidate = _one_parent_candidate("pure-SiO2", temperature, 0.0, gas_pack)
    result = equilibrate(
        {"SiO2": 1.0}, temperature, candidates=(candidate,), pack=ideal_pack,
    )
    assert result.logQ == pytest.approx(0.0, abs=1.0e-12)
    assert result.logQ_interval[0] <= 0.0 <= result.logQ_interval[1]
    assert result.refusal.code == "missing_phase_data"
    assert result.solid_mol is None
    assert result.liquid_mass_fraction is None
    assert result.liquid_parent_mol is None
    assert not result.phase_amounts_unique
    assert result.phase_presence["pure-SiO2"] == "unresolved"
    assert result.phase_amount_ranges_mol["pure-SiO2"] == pytest.approx((0.0, 1.0))
    assert result.crystal_only_minimum is False


def test_dual_face_logq_analytic_two_parent_minimum(ideal_pack, gas_pack):
    temperature = 2000.0
    rt = R_J_MOL_K * temperature
    parents = ("CaO", "SiO2")
    liquid_g = np.asarray([
        species_thermo(parent, "l", temperature, gas_pack).G_J_mol
        for parent in parents
    ])
    A = np.asarray([[1.0], [1.0]])
    b = np.ones(2)
    g = np.asarray([float(liquid_g.sum() - 2.0 * rt)])
    lp = _lp_solid_minimum(A, g, b)
    assert lp is not None
    mu, logq, interval, certified, diagnostics_map = _minimize_logq_on_dual_face(
        A, g, b, lp.objective_J, lp.chemical_potentials, liquid_g,
        temperature, ideal_pack,
        [tuple(load_datapack().parent_oxides).index(parent) for parent in parents],
    )
    expected = math.log(2.0) - 1.0
    assert mu == pytest.approx(liquid_g - rt, abs=1.0e-5)
    assert logq == pytest.approx(expected, abs=1.0e-8)
    assert interval[1] < 0.0
    assert certified
    assert diagnostics_map["logQ_stationarity_scaled"] <= 1.0e-8


def test_dual_face_logq_respects_variable_crystal_inequalities(ideal_pack, gas_pack):
    temperature = 2000.0
    rt = R_J_MOL_K * temperature
    parents = ("CaO", "SiO2")
    liquid_g = np.asarray([
        species_thermo(parent, "l", temperature, gas_pack).G_J_mol
        for parent in parents
    ])
    A = np.asarray([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    g = np.asarray([
        liquid_g[0] - 0.5 * rt,
        liquid_g[1] - 1.5 * rt,
        liquid_g.sum() - 2.0 * rt,
    ])
    b = np.ones(2)
    lp = _lp_solid_minimum(A, g, b)
    assert lp is not None
    mu, logq, interval, certified, diagnostics_map = _minimize_logq_on_dual_face(
        A, g, b, lp.objective_J, lp.chemical_potentials, liquid_g,
        temperature, ideal_pack,
        [tuple(load_datapack().parent_oxides).index(parent) for parent in parents],
    )
    expected_logs = np.asarray([-0.5, -1.5])
    expected = float(np.logaddexp(expected_logs[0], expected_logs[1]))
    assert mu == pytest.approx(liquid_g + rt * expected_logs, abs=1.0e-5)
    assert logq == pytest.approx(expected, abs=1.0e-8)
    assert interval[0] == pytest.approx(expected, abs=1.0e-8)
    assert certified
    assert diagnostics_map["logQ_stationarity_scaled"] <= 1.0e-8


def test_fully_solid_nonunique_extents_have_unique_activities(
    monkeypatch, ideal_pack, gas_pack
):
    temperature = 2000.0
    activity_logs = {"CaO": math.log(0.25), "SiO2": math.log(0.25)}
    ca = _candidate_from_log_activities(
        "solid-CaO", {"CaO": 1.0}, activity_logs, temperature, gas_pack
    )
    silica = _candidate_from_log_activities(
        "solid-SiO2", {"SiO2": 1.0}, activity_logs, temperature, gas_pack
    )
    compound = CrystalCandidate(
        "solid-CaSiO3", {"CaO": 1.0, "SiO2": 1.0},
        ca.G_crystal_J_mol + silica.G_crystal_J_mol,
    )
    _install_synthetic_default_candidates(monkeypatch, (ca, silica, compound), gas_pack)

    result = equilibrate({"CaO": 1.0, "SiO2": 1.0}, temperature, pack=ideal_pack)

    assert result.labels.validity == "IN"
    assert result.refusal is None
    assert result.phase_amounts_unique is False
    assert result.solid_mol is None
    for phase in ("solid-CaO", "solid-SiO2", "solid-CaSiO3"):
        assert result.phase_amount_ranges_mol[phase] == pytest.approx((0.0, 1.0))
    for parent in ("CaO", "SiO2"):
        assert result.activity_ranges_ln[parent] == pytest.approx(
            (math.log(0.25), math.log(0.25)), abs=2.0e-9
        )
        assert result.identifiable_activities[parent] == pytest.approx(0.25, abs=2.0e-9)


def test_fully_solid_dual_face_ranges_are_scale_invariant(
    monkeypatch, ideal_pack, gas_pack
):
    temperature = 2000.0
    activity_logs = {"CaO": math.log(0.25), "SiO2": math.log(0.25)}
    ca = _candidate_from_log_activities(
        "solid-CaO", {"CaO": 1.0}, activity_logs, temperature, gas_pack
    )
    silica = _candidate_from_log_activities(
        "solid-SiO2", {"SiO2": 1.0}, activity_logs, temperature, gas_pack
    )
    compound = CrystalCandidate(
        "solid-CaSiO3", {"CaO": 1.0, "SiO2": 1.0},
        ca.G_crystal_J_mol + silica.G_crystal_J_mol,
    )
    _install_synthetic_default_candidates(monkeypatch, (ca, silica, compound), gas_pack)

    results = [
        equilibrate(
            {"CaO": scale, "SiO2": scale}, temperature, pack=ideal_pack,
        )
        for scale in (1.0, 1.0e-8, 1.0e-10)
    ]
    reference = results[0]
    assert all(result.labels.validity == "IN" for result in results)
    assert all(result.refusal is None for result in results)
    for result in results:
        for parent in ("CaO", "SiO2"):
            assert result.activity_ranges_ln[parent] == pytest.approx(
                reference.activity_ranges_ln[parent], abs=2.0e-9
            )
        for phase in reference.phase_amount_ranges_mol:
            expected = tuple(
                endpoint * result.bulk_parent_mol["CaO"]
                for endpoint in reference.phase_amount_ranges_mol[phase]
            )
            assert result.phase_amount_ranges_mol[phase] == pytest.approx(
                expected, abs=2.0e-18
            )


def test_fully_solid_unique_extent_can_have_unbounded_activity_ranges(
    monkeypatch, ideal_pack, gas_pack
):
    temperature = 2000.0
    rt = R_J_MOL_K * temperature
    liquid_g = math.fsum(
        species_thermo(parent, "l", temperature, gas_pack).G_J_mol
        for parent in ("CaO", "SiO2")
    )
    compound = CrystalCandidate(
        "solid-CaSiO3", {"CaO": 1.0, "SiO2": 1.0}, liquid_g - 2.0 * rt
    )
    _install_synthetic_default_candidates(monkeypatch, (compound,), gas_pack)

    result = equilibrate({"CaO": 1.0, "SiO2": 1.0}, temperature, pack=ideal_pack)

    assert result.labels.validity == "IN"
    assert result.phase_amounts_unique is True
    assert result.phase_amount_ranges_mol["solid-CaSiO3"] == pytest.approx((1.0, 1.0))
    assert result.solid_mol["solid-CaSiO3"] == pytest.approx(1.0)
    assert result.identifiable_activities == {}
    assert result.activity_ranges_ln["CaO"] == (-math.inf, math.inf)
    assert result.activity_ranges_ln["SiO2"] == (-math.inf, math.inf)


def test_eutectic_invariant_reports_lever_ranges_and_brackets(
    monkeypatch, ideal_pack, gas_pack
):
    invariant_K = 2000.0
    liquid_activity = math.log(0.5)
    ca = _candidate_from_log_activities(
        "solid-CaO", {"CaO": 1.0}, {"CaO": liquid_activity}, invariant_K, gas_pack
    )
    silica = _candidate_from_log_activities(
        "solid-SiO2", {"SiO2": 1.0}, {"SiO2": liquid_activity}, invariant_K, gas_pack
    )
    _install_synthetic_default_candidates(monkeypatch, (ca, silica), gas_pack)

    invariant = equilibrate(
        {"CaO": 1.0, "SiO2": 1.0}, invariant_K, pack=ideal_pack,
    )
    assert invariant.labels.validity == "IN"
    assert invariant.refusal is None
    assert invariant.saturated_candidates == ("solid-CaO", "solid-SiO2")
    assert invariant.phase_amounts_unique is False
    assert invariant.solid_mol is None
    assert invariant.liquid_parent_mol is None
    assert invariant.liquid_mass_fraction is None
    for phase in ("solid-CaO", "solid-SiO2"):
        assert invariant.phase_amount_ranges_mol[phase] == pytest.approx((0.0, 1.0))
        assert invariant.phase_presence[phase] == "unresolved"

    boundaries = phase_boundary_brackets(
        {"CaO": 1.0, "SiO2": 1.0}, 1990.0, 2010.0,
        pack=ideal_pack, scan_points=4,
    )
    assert boundaries.solidus_status == "AVAILABLE", boundaries.to_dict()
    assert boundaries.liquidus_status == "AVAILABLE"
    for crossing in (*boundaries.solidus, *boundaries.liquidus):
        assert crossing.root_temperature_K == pytest.approx(invariant_K, abs=0.1)
        assert crossing.bracket_K[0] <= invariant_K <= crossing.bracket_K[1]
        assert crossing.bracket_K[1] - crossing.bracket_K[0] <= 0.1
        assert crossing.lower_endpoint_status == "IN"
        assert crossing.upper_endpoint_status == "IN"


def test_peritectic_solidus_and_liquidus_match_independent_analytics(
    monkeypatch, ideal_pack, gas_pack
):
    invariant_K = 2000.0
    liquid_x_ca = 0.25
    ca = _candidate_from_log_activities(
        "phase-A", {"CaO": 1.0},
        {"CaO": math.log(liquid_x_ca)}, invariant_K, gas_pack,
    )
    compound = _candidate_from_log_activities(
        "phase-AB", {"CaO": 1.0, "SiO2": 1.0},
        {"CaO": math.log(liquid_x_ca), "SiO2": math.log(1.0 - liquid_x_ca)},
        invariant_K, gas_pack,
    )
    _install_synthetic_default_candidates(monkeypatch, (ca, compound), gas_pack)
    bulk = {"CaO": 0.6, "SiO2": 0.4}

    # At the invariant, L has x(CaO)=0.25 and AB has x(CaO)=0.5, so AB
    # lies between the liquid and A. With bulk x(CaO)=0.6, balances give
    # n_A=0.2+0.5*n_L and n_AB=0.4-0.75*n_L, with 0≤n_L≤8/15.
    invariant = equilibrate(bulk, invariant_K, pack=ideal_pack)
    assert invariant.labels.validity == "IN"
    assert invariant.phase_amount_ranges_mol["phase-A"] == pytest.approx((0.2, 7.0 / 15.0))
    assert invariant.phase_amount_ranges_mol["phase-AB"] == pytest.approx((0.0, 0.4))
    assert invariant.receipt.solver["liquid_parent_scale_range_mol"] == pytest.approx(
        (0.0, 8.0 / 15.0)
    )
    assert invariant.phase_amounts_unique is False

    def independent_a_saturation(temperature):
        liquid_g = species_thermo("CaO", "l", temperature, gas_pack).G_J_mol
        return math.log(bulk["CaO"]) + (liquid_g - ca.G_crystal_J_mol) / (
            R_J_MOL_K * temperature
        )

    def independent_ab_saturation(temperature):
        liquid_g = math.fsum(
            species_thermo(parent, "l", temperature, gas_pack).G_J_mol
            for parent in ("CaO", "SiO2")
        )
        return math.log(bulk["CaO"] * bulk["SiO2"]) + (
            liquid_g - compound.G_crystal_J_mol
        ) / (R_J_MOL_K * temperature)

    a_disappears_K = independent_brentq(
        independent_a_saturation, invariant_K, 3000.0
    )
    ab_disappears_K = independent_brentq(
        independent_ab_saturation, invariant_K, 3000.0
    )
    # All-liquid stability requires both affinities nonpositive; A sets the
    # higher-temperature root for this invariant composition and bulk.
    analytic_liquidus = max(a_disappears_K, ab_disappears_K)
    upper = 2100.0
    boundaries = phase_boundary_brackets(
        bulk, 1990.0, upper, pack=ideal_pack, scan_points=12,
    )
    detail = (
        boundaries.refusal,
        boundaries.receipt.solver.get("scan_endpoint_statuses"),
        boundaries.receipt.solver.get("scan_liquid_parent_scale_ranges_mol"),
        boundaries.receipt.solver.get("scan_phase_sets"),
        boundaries.receipt.solver.get("boundary_refinement_errors"),
    )
    assert boundaries.solidus_status == "AVAILABLE", detail
    assert boundaries.liquidus_status == "AVAILABLE", detail
    assert boundaries.solidus[0].root_temperature_K == pytest.approx(invariant_K, abs=0.1)
    assert boundaries.liquidus[0].root_temperature_K == pytest.approx(analytic_liquidus, abs=0.1)
    assert boundaries.solidus[0].lower_phase_set
    assert boundaries.liquidus[0].upper_phase_set == ()


def test_refused_refinement_point_cannot_certify_boundary(
    monkeypatch, ideal_pack, gas_pack
):
    invariant_K = 2000.0
    activity_logs = {"CaO": math.log(0.5), "SiO2": math.log(0.5)}
    candidates = (
        _candidate_from_log_activities(
            "solid-CaO", {"CaO": 1.0}, activity_logs, invariant_K, gas_pack
        ),
        _candidate_from_log_activities(
            "solid-SiO2", {"SiO2": 1.0}, activity_logs, invariant_K, gas_pack
        ),
    )
    _install_synthetic_default_candidates(monkeypatch, candidates, gas_pack)
    from openimcc import equilibrium as equilibrium_module

    actual_equilibrate = equilibrium_module.equilibrate

    def refused_root(bulk, temperature, **kwargs):
        result = actual_equilibrate(bulk, temperature, **kwargs)
        if abs(temperature - invariant_K) < 1.0e-6:
            refusal = EquilibriumRefusal(
                "injected_refinement_refusal", "Synthetic failed refinement certificate."
            )
            return replace(
                result,
                labels=replace(result.labels, validity="DEPENDS"),
                refusal=refusal,
                kkt_recheck={"independent_kkt_passed": 0.0},
            )
        return result

    monkeypatch.setattr(equilibrium_module, "equilibrate", refused_root)
    boundaries = phase_boundary_brackets(
        {"CaO": 1.0, "SiO2": 1.0}, 1990.0, 2010.0,
        pack=ideal_pack, scan_points=2,
    )
    assert boundaries.labels.validity == "DEPENDS"
    assert boundaries.refusal is not None
    assert boundaries.solidus_status == "DEPENDS"
    assert boundaries.liquidus_status == "DEPENDS"
    assert boundaries.solidus == ()
    assert boundaries.liquidus == ()


def test_phase_boundaries_refine_cooling_direction_crossing(
    monkeypatch, ideal_pack, gas_pack
):
    from openimcc import equilibrium as equilibrium_module

    def temperature_dependent_default(temperature, selected_gas_pack=None):
        selected_gas_pack = gas_pack if selected_gas_pack is None else selected_gas_pack
        liquid_g = species_thermo("SiO2", "l", temperature, selected_gas_pack).G_J_mol
        candidate = CrystalCandidate(
            "temperature-dependent-SiO2", {"SiO2": 1.0},
            liquid_g + R_J_MOL_K * temperature * (2000.0 - temperature) / 1000.0,
        )
        state = _CandidateState(
            candidate, candidate.source_authority,
            candidate.temperature_range_K, True,
        )
        return (state,), "synthetic-crystal-data", "synthetic-pure-data", selected_gas_pack, ()

    monkeypatch.setattr(
        equilibrium_module, "_default_candidate_states", temperature_dependent_default
    )
    monkeypatch.setattr(
        equilibrium_module, "_source_solid_limits_K", lambda _candidates: {},
    )
    boundaries = phase_boundary_brackets(
        {"SiO2": 1.0}, 1990.0, 2010.0, pack=ideal_pack, scan_points=3,
    )
    assert boundaries.solidus_status == "AVAILABLE", boundaries.to_dict()
    assert boundaries.liquidus_status == "AVAILABLE", boundaries.to_dict()
    assert boundaries.solidus[0].root_temperature_K == pytest.approx(2000.0, abs=0.1)
    assert boundaries.liquidus[0].root_temperature_K == pytest.approx(2000.0, abs=0.1)


@pytest.mark.parametrize("coincident_crossing", (False, True))
def test_source_cap_transition_is_not_a_model_boundary_in_either_sweep_order(
    monkeypatch, ideal_pack, gas_pack, coincident_crossing
):
    from openimcc import equilibrium as equilibrium_module

    cap_K = 2000.0
    compound_id = "synthetic-multioxide"
    alternate_id = "available-alternate"
    bulk = {"CaO": 1.0, "SiO2": 1.0}

    def synthetic_defaults(temperature, selected_gas_pack=None):
        selected_gas_pack = gas_pack if selected_gas_pack is None else selected_gas_pack
        liquid_reference = math.fsum(
            species_thermo(parent, "l", temperature, selected_gas_pack).G_J_mol
            for parent in bulk
        )
        common = {
            "stoichiometry": {"CaO": 1.0, "SiO2": 1.0},
            "formula": "CaSiO3",
            "source_authority": "external_datapack",
            "temperature_range_K": (1700.0, 3000.0),
        }
        # At all-liquid x=(0.5, 0.5), the capped phase's affinity is its Gibbs
        # offset minus ln(4), so offset 4 is positive while the alternate is
        # undersaturated. Its removal changes liquid presence, the solidus-side
        # analogue of positive affinity at an incipient liquid composition.
        affinity_offset = (
            math.log(4.0) + 0.01 * (cap_K - temperature)
            if coincident_crossing else 4.0
        )
        compound = CrystalCandidate(
            phase_id=compound_id,
            G_crystal_J_mol=(
                liquid_reference - affinity_offset * R_J_MOL_K * temperature
            ),
            **common,
        )
        alternate = CrystalCandidate(
            phase_id=alternate_id,
            G_crystal_J_mol=liquid_reference + 2.0 * R_J_MOL_K * temperature,
            **common,
        )
        capped = temperature > cap_K
        states = (
            _CandidateState(
                compound, compound.source_authority, compound.temperature_range_K,
                not capped, "above_source_melting_point" if capped else None,
            ),
            _CandidateState(
                alternate, alternate.source_authority,
                alternate.temperature_range_K, True,
            ),
        )
        return (
            states, "synthetic-crystal-data", "synthetic-pure-data",
            selected_gas_pack, (),
        )

    monkeypatch.setattr(
        equilibrium_module, "_default_candidate_states", synthetic_defaults
    )
    monkeypatch.setattr(
        equilibrium_module,
        "_source_solid_limits_K",
        lambda _candidates: {compound_id: cap_K},
    )
    just_above_cap = float(np.nextafter(cap_K, math.inf))

    boundaries = phase_boundary_brackets(
        bulk, 1990.0, 2010.0, pack=ideal_pack, scan_points=2,
    )
    assert boundaries.labels.validity == "DEPENDS", boundaries.to_dict()
    assert boundaries.solidus_status == "DEPENDS", boundaries.to_dict()
    assert boundaries.liquidus_status == "DEPENDS", boundaries.to_dict()
    assert boundaries.solidus == ()
    assert boundaries.liquidus == ()
    assert f"source_cap:{compound_id}@{cap_K:g}K" in boundaries.labels.notices
    assert not any(
        notice.startswith("source_cap_unresolved:")
        for notice in boundaries.labels.notices
    )

    temperatures = (1990.0, cap_K, just_above_cap, 2010.0)
    forward = equilibrate_sweep(
        bulk, temperatures, pack=ideal_pack,
    )
    reverse = equilibrate_sweep(
        bulk, tuple(reversed(temperatures)), pack=ideal_pack,
    )
    for sweep in (forward, reverse):
        ordered = sorted(sweep, key=lambda result: result.temperature_K)
        cap_transitions = []
        for left, right in zip(ordered, ordered[1:]):
            phases, _phase_set_changed, _liquid_presence_changed = (
                equilibrium_module._source_cap_transition_phases(
                    left, right, 1.0e-10,
                )
            )
            if phases:
                cap_transitions.append(
                    (left.temperature_K, right.temperature_K, phases)
                )
        assert cap_transitions == [(cap_K, just_above_cap, (compound_id,))]
    reverse_by_temperature = {result.temperature_K: result for result in reverse}
    forward_by_temperature = {result.temperature_K: result for result in forward}
    for temperature in (cap_K, just_above_cap):
        result = forward_by_temperature[temperature]
        counterpart = reverse_by_temperature[temperature]
        assert result.phase_presence == counterpart.phase_presence
        if temperature == just_above_cap:
            assert result.labels.validity == counterpart.labels.validity == "IN"
            assert result.labels.notices == counterpart.labels.notices
            assert result.receipt.solver["crystals_above_source_melting_point"] == (
                compound_id,
            )
            assert f"crystals_above_source_melting_point:{compound_id}" in result.labels.notices
        elif coincident_crossing:
            assert result.labels.validity in ("IN", "DEPENDS")
            assert counterpart.labels.validity in ("IN", "DEPENDS")


def test_explicit_multioxide_candidate_obeys_source_cap(ideal_pack, gas_pack):
    cap_K = 2000.0
    phase_id = "explicit-source-capped-multioxide"
    bulk = {"CaO": 1.0, "SiO2": 1.0}
    parent_liquid_gibbs = math.fsum(
        species_thermo(parent, "l", cap_K, gas_pack).G_J_mol
        for parent in bulk
    )
    candidate = CrystalCandidate(
        phase_id,
        {"CaO": 1.0, "SiO2": 1.0},
        parent_liquid_gibbs - 40.0 * R_J_MOL_K * cap_K,
        formula="CaSiO3",
        temperature_range_K=(1700.0, 3000.0),
        source={"melting_point": {"temperature_K": cap_K}},
    )

    above_cap = equilibrate(
        bulk, cap_K + 1.0, candidates=(candidate,), pack=ideal_pack,
    )
    assert above_cap.phase_presence[phase_id] == "absent"
    assert above_cap.labels.coverage[phase_id] == "above_source_melting_point"
    assert above_cap.receipt.solver["crystals_above_source_melting_point"] == (
        phase_id,
    )
    assert (
        f"crystals_above_source_melting_point:{phase_id}"
        in above_cap.labels.notices
    )
    assert phase_id not in above_cap.receipt.solver["missing_candidate_phase_ids"]
    assert "missing_phase_data" not in above_cap.labels.notices
    assert "restricted_candidate_set" in above_cap.labels.notices

    boundaries = phase_boundary_brackets(
        bulk, cap_K - 10.0, cap_K + 10.0, candidates=(candidate,),
        pack=ideal_pack, scan_points=2,
    )
    assert boundaries.labels.validity == "DEPENDS", boundaries.to_dict()
    assert boundaries.solidus_status == "DEPENDS", boundaries.to_dict()
    assert boundaries.liquidus_status == "DEPENDS", boundaries.to_dict()
    assert boundaries.solidus == ()
    assert boundaries.liquidus == ()
    assert f"source_cap:{phase_id}@{cap_K:g}K" in boundaries.labels.notices


def test_polymorph_extents_do_not_hide_the_liquidus(
    monkeypatch, ideal_pack, gas_pack
):
    coexistence_K = 2000.0
    candidate = _candidate_from_log_activities(
        "poly-alpha", {"CaO": 1.0}, {"CaO": math.log(1.0 / 3.0)},
        coexistence_K, gas_pack,
    )
    polymorph = CrystalCandidate(
        "poly-beta", {"CaO": 1.0}, candidate.G_crystal_J_mol,
    )
    _install_synthetic_default_candidates(
        monkeypatch, (candidate, polymorph), gas_pack,
    )
    bulk = {"CaO": 1.0, "SiO2": 1.0}

    def all_liquid_residual(temperature):
        activity = math.log(0.5)
        liquid_g = species_thermo("CaO", "l", temperature, gas_pack).G_J_mol
        return activity + (liquid_g - candidate.G_crystal_J_mol) / (
            R_J_MOL_K * temperature
        )

    analytic_liquidus = independent_brentq(
        all_liquid_residual, coexistence_K, 3000.0,
    )
    boundaries = phase_boundary_brackets(
        bulk, 1990.0, analytic_liquidus + 5.0,
        pack=ideal_pack, scan_points=3,
    )
    assert boundaries.liquidus_status == "AVAILABLE", boundaries.to_dict()
    assert boundaries.liquidus[0].root_temperature_K == pytest.approx(
        analytic_liquidus, abs=0.1
    )
    assert boundaries.liquidus[0].upper_phase_set == ()


def test_warm_sweep_agrees_in_both_directions_away_from_invariant(
    monkeypatch, ideal_pack, gas_pack
):
    crossing = 2000.0
    candidate = _one_parent_candidate("solid-SiO2", crossing, 0.0, gas_pack)
    _install_synthetic_default_candidates(monkeypatch, (candidate,), gas_pack)
    temperatures = (1800.0, 1900.0, 2100.0, 2200.0)
    forward = equilibrate_sweep({"SiO2": 1.0}, temperatures, pack=ideal_pack)
    reverse = equilibrate_sweep(
        {"SiO2": 1.0}, tuple(reversed(temperatures)), pack=ideal_pack
    )
    by_temperature = {result.temperature_K: result for result in reverse}
    for result in forward:
        comparison = by_temperature[result.temperature_K]
        assert result.labels.validity == comparison.labels.validity == "IN"
        assert result.phase_presence == comparison.phase_presence
        assert result.phase_amount_ranges_mol == pytest.approx(
            comparison.phase_amount_ranges_mol
        )
        assert result.liquid_mass_fraction == pytest.approx(comparison.liquid_mass_fraction)
        assert result.kkt_recheck["independent_kkt_passed"] == 1.0
        assert comparison.kkt_recheck["independent_kkt_passed"] == 1.0


def test_default_anorthite_boundary_remains_out_of_domain_below_1700():
    bulk = {"CaO": 1.0, "Al2O3": 1.0, "SiO2": 2.0}
    low = equilibrate(bulk, 1699.0)
    assert low.labels.validity == "OUT_OF_DOMAIN"
    assert low.labels.out_of_domain_axes == ("temperature_K",)
    assert low.crystal_only_minimum is True

    boundaries = phase_boundary_brackets(
        bulk, 1690.0, 1800.0, scan_points=5,
    )
    assert "temperature_K" in boundaries.labels.out_of_domain_axes
    assert all(
        crossing.bracket_K[0] >= 1700.0
        for crossing in (*boundaries.solidus, *boundaries.liquidus)
    )

    forsterite = phase_boundary_brackets(
        {"MgO": 2.0, "SiO2": 1.0}, 1750.0, 1850.0, scan_points=6,
    )
    assert forsterite.solidus_status == "AVAILABLE", forsterite.to_dict()
    assert forsterite.solidus[0].root_temperature_K == pytest.approx(
        1798.7, abs=0.2
    )
    assert forsterite.solidus[0].bracket_K[0] >= 1700.0
    assert forsterite.liquidus_status == "NO_CROSSING"


def test_cpu_median_per_warm_sweep_point(ideal_pack, gas_pack, monkeypatch):
    crossing = 2000.0
    candidate = _one_parent_candidate("solid-SiO2", crossing, 0.0, gas_pack)
    _install_synthetic_default_candidates(monkeypatch, (candidate,), gas_pack)
    temperatures = (1850.0, 1900.0, 2100.0)
    elapsed = []
    for _ in range(5):
        started = process_time()
        results = equilibrate_sweep(
            {"SiO2": 1.0}, temperatures, pack=ideal_pack,
        )
        elapsed.append((process_time() - started) / len(temperatures))
        assert len(results) == len(temperatures)
        assert all(result.kkt_recheck.get("independent_kkt_passed") == 1.0 for result in results)
    print(
        "CPU median per warm sweep point (n=5 sweeps): "
        f"{median(elapsed):.6f} s"
    )


def test_default_and_explicit_candidate_scope_labels(ideal_pack):
    default = equilibrate({"SiO2": 1.0}, 1990.0, pack=ideal_pack)
    assert "restricted_candidate_set" not in default.labels.notices
    assert default.receipt.solver["missing_candidate_phase_ids"]

    explicit = equilibrate({"SiO2": 1.0}, 1990.0, candidates=(), pack=ideal_pack)
    assert "restricted_candidate_set" in explicit.labels.notices
    assert explicit.refusal is not None
    assert explicit.refusal.code == "missing_phase_data"


def test_phase_amount_ranges_are_invariant_to_bulk_scale(ideal_pack, gas_pack):
    temperature = 2000.0
    parents = ("CaO", "SiO2")
    # With liquid CaO activity 1/9 at bulk ratio (1,2), the pure CaO phase
    # amount is 3/4; scaling the inventory must scale every extent equally.
    activity = 1.0 / 9.0
    liquid_g = species_thermo("CaO", "l", temperature, gas_pack).G_J_mol
    phase = CrystalCandidate(
        "CaO-solid", {"CaO": 1.0},
        liquid_g + R_J_MOL_K * temperature * math.log(activity),
    )
    base = equilibrate(
        {"CaO": 1.0, "SiO2": 2.0}, temperature,
        candidates=(phase,), pack=ideal_pack,
    )
    tiny = equilibrate(
        {"CaO": 1.0e-8, "SiO2": 2.0e-8}, temperature,
        candidates=(phase,), pack=ideal_pack,
    )
    assert base.phase_amounts_unique and tiny.phase_amounts_unique
    assert base.phase_amount_ranges_mol["CaO-solid"][0] == pytest.approx(0.75, abs=1e-8)
    assert tiny.phase_amount_ranges_mol["CaO-solid"][0] == pytest.approx(0.75e-8, rel=1e-8)


def test_default_anorthite_case_excludes_zero_bulk_phases():
    bulk = {"CaO": 1.0, "Al2O3": 1.0, "SiO2": 2.0}
    result = equilibrate(bulk, 1800.0)
    assert result.liquid_parent_mol == pytest.approx(
        {"CaO": 1.0, "Al2O3": 1.0, "SiO2": 2.0}
    )
    assert result.solid_mol is not None
    assert all(amount == pytest.approx(0.0, abs=1.0e-12) for amount in result.solid_mol.values())
    assert result.liquid_mass_fraction == pytest.approx(1.0)
    assert result.kkt_recheck["independent_kkt_passed"] == 1.0
    assert result.refusal is not None
    assert result.refusal.code == "missing_phase_data"
    assert result.labels.validity == "DEPENDS"
    missing = result.receipt.solver["missing_candidate_phase_ids"]
    assert set(missing) == {"C12A7", "CA6"}
    dropped = {
        "akermanite", "diopside", "fayalite", "FeO", "hercynite",
        "ilmenite", "monticellite",
    }
    for phase in dropped:
        assert result.phase_presence[phase] == "infeasible"
        assert result.phase_amount_ranges_mol[phase] == (0.0, 0.0)

    hot = equilibrate(bulk, 1900.0)
    hot_missing = hot.receipt.solver["missing_candidate_phase_ids"]
    assert hot.labels.validity == "DEPENDS"
    assert {"CA6", "C12A7"} <= set(hot_missing)
    assert "anorthite" not in hot_missing
    assert "anorthite" in hot.receipt.solver["crystals_above_source_melting_point"]
    assert any(
        notice.startswith("crystals_above_source_melting_point:")
        and "anorthite" in notice
        for notice in hot.labels.notices
    )
    print(
        "anorthite-1800 returned liquid:", dict(result.liquid_parent_mol),
        "solid amounts:", dict(result.solid_mol),
        "liquid mass fraction:", result.liquid_mass_fraction,
        "infeasible phases:", ", ".join(sorted(dropped)),
        "anorthite affinity:", result.phase_affinities_ln["anorthite"],
    )


def test_feasible_candidate_without_gibbs_still_depends(monkeypatch):
    from openimcc import equilibrium as equilibrium_module

    original = equilibrium_module._default_candidate_states

    def missing_feasible_anorthite(temperature, gas_pack=None):
        states, crystal_digest, pure_digest, resolved_gas, unavailable = original(
            temperature, gas_pack
        )
        altered = tuple(
            replace(state, available=False, reason="standard_out_of_range")
            if state.candidate.phase_id == "anorthite" else state
            for state in states
        )
        return (
            altered, crystal_digest, pure_digest, resolved_gas,
            tuple(sorted((*unavailable, "anorthite"))),
        )

    monkeypatch.setattr(
        "openimcc.equilibrium._default_candidate_states",
        missing_feasible_anorthite,
    )
    result = equilibrate(
        {"CaO": 1.0, "Al2O3": 1.0, "SiO2": 2.0}, 1800.0,
    )
    assert result.labels.validity == "DEPENDS"
    assert result.refusal is not None and result.refusal.code == "missing_phase_data"
    assert "anorthite" in result.receipt.solver["missing_candidate_phase_ids"]
    assert result.phase_presence["anorthite"] == "unresolved"


def test_result_to_dict_round_trips_through_json(ideal_pack, gas_pack):
    candidate = _one_parent_candidate("serialization-control", 2000.0, -5.0, gas_pack)
    result = equilibrate(
        {"SiO2": 1.0}, 2000.0, candidates=(candidate,), pack=ideal_pack,
    )
    encoded = json.dumps(result.to_dict())
    decoded = json.loads(encoded)
    assert decoded["temperature_K"] == 2000.0
    assert decoded["bulk_parent_mol"] == {"SiO2": 1.0}
    assert decoded["phase_saturation_indices"]


@pytest.mark.parametrize("value", [True, np.bool_(True)])
def test_parse_bulk_rejects_boolean_parent_amount(value: object, ideal_pack) -> None:
    with pytest.raises(ValueError, match="numeric"):
        _parse_bulk({"SiO2": value}, ideal_pack.parent_oxides)


@pytest.mark.parametrize("value", [True, np.bool_(True)])
def test_equilibrate_refuses_boolean_temperature(value: object, ideal_pack) -> None:
    result = equilibrate({"SiO2": 1.0}, value, candidates=(), pack=ideal_pack)
    assert result.refusal is not None
    assert result.refusal.code == "invalid_input"


@pytest.mark.parametrize("settings", [{"tol": True}, {"max_iter": True}])
def test_equilibrate_refuses_boolean_solver_settings(
    settings: dict[str, object], ideal_pack,
) -> None:
    result = equilibrate(
        {"SiO2": 1.0}, 1990.0, candidates=(), pack=ideal_pack, **settings
    )
    assert result.refusal is not None
    assert result.refusal.code == "invalid_input"


def test_fully_liquid_vapour_matches_direct_gas_evaluation(ideal_pack, gas_pack):
    fO2 = 1.0e-8
    result = equilibrate(
        {"MgO": 1.0, "SiO2": 1.0}, 2000.0,
        candidates=(), pack=ideal_pack, fO2=fO2,
    )
    expected = gas_module.evaluate_gas(
        result.identifiable_activities, result.temperature_K, fO2, gas_pack,
        parent_oxides=tuple(result.identifiable_activities),
    )

    assert result.liquid_mass_fraction == pytest.approx(1.0)
    assert dict(result.gas) == dict(expected)


def test_crystal_vapour_withholds_nonidentifiable_channels(
    monkeypatch, ideal_pack, gas_pack
):
    temperature = 2000.0
    stoichiometry = {"CaO": 1.0, "SiO2": 1.0}
    liquid_gibbs = math.fsum(
        species_thermo(parent, "l", temperature, gas_pack).G_J_mol
        for parent in stoichiometry
    )
    # A 2 RT lower crystal Gibbs energy creates a certified crystal-only state.
    crystal_gibbs = liquid_gibbs - 2.0 * R_J_MOL_K * temperature
    candidate = CrystalCandidate(
        "solid-CaSiO3", stoichiometry, crystal_gibbs,
    )
    _install_synthetic_default_candidates(monkeypatch, (candidate,), gas_pack)
    result = equilibrate(
        {"CaO": 1.0, "SiO2": 1.0}, temperature,
        pack=ideal_pack, fO2=1.0e-8,
    )

    assert result.phase_presence["solid-CaSiO3"] == "present"
    assert result.identifiable_activities == {}
    assert "CaO" in result.gas_depends
    assert "activity_not_identifiable" in result.gas_depends["CaO"]
    assert "CaO" not in result.gas
    assert result.labels.validity == "IN"
    assert result.to_dict()["phase_state_contract"]["output"]["vapour"]["status"] == "DEPENDS"


def test_vapour_status_is_available_when_source_molten_candidates_clear_dependencies():
    result = equilibrate(
        {"CaO": 3.0, "SiO2": 1.0}, 1900.0, fO2=1.0e-8,
    )

    assert result.labels.validity == "IN"
    assert result.refusal is None
    assert result.to_dict()["phase_state_contract"]["output"]["vapour"]["status"] == "AVAILABLE"
    assert result.receipt.solver["crystals_above_source_melting_point"]


def test_vapour_domain_does_not_change_condensed_equilibrium_labels():
    bulk = {"CaO": 1.0, "Al2O3": 1.0, "SiO2": 2.0}
    condensed = equilibrate(bulk, 1800.0)
    with_vapour = equilibrate(bulk, 1800.0, fO2=1.0e-8)
    vapour = with_vapour.to_dict()["phase_state_contract"]["output"]["vapour"]

    assert condensed.labels.validity == "DEPENDS"
    assert set(condensed.labels.out_of_domain_axes) == {
        "crystal.C12A7.temperature_K", "crystal.CA6.temperature_K",
    }
    assert with_vapour.labels == condensed.labels
    assert with_vapour.labels.validity == "DEPENDS"
    assert set(with_vapour.labels.out_of_domain_axes) == {
        "crystal.C12A7.temperature_K", "crystal.CA6.temperature_K",
    }
    assert vapour["status"] == "OUT_OF_DOMAIN"
    assert set(vapour["out_of_domain_axes"]) == {
        f"gas_species.{species}.temperature_K"
        for species in ("AlO", "AlO2", "Al2O", "Al2O2", "Al", "Al2", "AlSiO")
    }


def test_phase_state_contract_schema_round_trips_through_json(ideal_pack):
    result = equilibrate(
        {"MgO": 1.0, "SiO2": 1.0}, 2000.0,
        candidates=(), pack=ideal_pack, fO2=1.0e-8,
    )
    decoded = json.loads(json.dumps(result.to_dict()))
    contract = decoded["phase_state_contract"]
    output = contract["output"]

    assert contract["name"] == "openimcc.phase-state"
    assert contract["version"] == phase_state_contract_version == 1
    assert contract["input"]["bulk_basis"] == "parent_oxide_mol"
    assert contract["input"]["bulk_parent_mol"] == {"MgO": 1.0, "SiO2": 1.0}
    assert contract["input"]["candidates"] == []
    assert contract["input"]["fO2"] == 1.0e-8
    assert set(output) == {
        "phase_presence", "phase_amount_ranges_mol", "phase_amounts_unique",
        "liquid_mass_fraction", "identifiable_activities", "labels",
        "refusal", "receipt", "vapour",
    }
    assert output["liquid_mass_fraction"]["certification_status"] == "CERTIFIED"
    assert output["vapour"]["status"] in {"AVAILABLE", "DEPENDS", "OUT_OF_DOMAIN"}
    assert output["vapour"]["fO2"] == 1.0e-8
    assert output["vapour"]["unit"] == "bar"


def test_phase_state_contract_preserves_equilibrium_refusal(ideal_pack):
    candidate = CrystalCandidate("magnetite", {"FeO": 3.0}, 0.0)
    result = equilibrate(
        {"FeO": 1.0}, 1800.0, candidates=(candidate,), pack=ideal_pack,
        fO2=1.0e-8,
    )
    contract = result.to_dict()["phase_state_contract"]

    assert result.refusal is not None
    assert result.refusal.code == "unsupported_redox_phase"
    assert contract["output"]["refusal"]["code"] == "unsupported_redox_phase"


def test_absent_by_construction_uses_selected_liquid_standard(ideal_pack):
    result = equilibrate(
        {"K2O": 1.0, "Na2O": 1.0, "FeO": 1.0}, 1800.0, pack=ideal_pack,
    )
    coverage = result.labels.coverage
    assert coverage["K2O"] == "absent_by_construction"
    assert coverage["Na2O"] == "absent_by_construction"
    assert coverage["FeO"] == "temperature_out_of_range"
    assert "FeO" in result.receipt.solver["missing_candidate_phase_ids"]
    gas_pack = diagnostics._crystal_thermo_context(None)[0]
    pure, _ = _load_json(_PURE_RESOURCE)
    rows = {row["id"]: row for row in pure["phases"]}
    for phase, expected_kJ in (("K2O", -20.14), ("Na2O", -13.37), ("FeO", 5.85)):
        oxide = next(iter(rows[phase]["oxide_stoichiometry"]))
        uncapped = evaluate_janaf_crystal(phase, 1800.0, apply_phase_cap=False)
        selected_liquid = species_thermo(oxide, "l", 1800.0, gas_pack).G_J_mol
        delta_kJ = (selected_liquid - uncapped.G_J_mol) / 1000.0
        assert delta_kJ == pytest.approx(expected_kJ, abs=0.02)


def test_absent_by_construction_is_absent_in_liquid_result():
    result = equilibrate({"K2O": 1.0, "Na2O": 1.0}, 1800.0)

    assert result.labels.validity == "IN"
    assert result.refusal is None
    assert result.liquid_parent_mol == {"K2O": 1.0, "Na2O": 1.0}
    for phase in ("K2O", "Na2O"):
        assert result.phase_presence[phase] == "absent"
        assert result.phase_amount_ranges_mol[phase] == (0.0, 0.0)


def test_absent_by_construction_is_absent_in_solid_result(
    monkeypatch, gas_pack,
):
    temperature = 1800.0
    available = tuple(
        _CandidateState(
            _candidate_from_log_activities(
                f"synthetic-{phase}-solid",
                {phase: 1.0},
                {phase: math.log(0.25)},
                temperature,
                gas_pack,
            ),
            "external_datapack",
            None,
            True,
        )
        for phase in ("K2O", "Na2O")
    )
    proven_absent = tuple(
        _CandidateState(
            CrystalCandidate(phase, {phase: 1.0}, 0.0),
            "external_datapack",
            None,
            False,
            "absent_by_construction",
        )
        for phase in ("K2O", "Na2O")
    )
    states = (*available, *proven_absent)
    monkeypatch.setattr(
        "openimcc.equilibrium._resolve_candidates",
        lambda *_args, **_kwargs: (
            states, "control-crystals", "control-pure", gas_pack, (), False,
        ),
    )

    result = equilibrate({"K2O": 1.0, "Na2O": 1.0}, temperature)

    assert result.labels.validity == "IN", result.refusal
    assert result.liquid_parent_mol is None
    for phase in ("K2O", "Na2O"):
        assert result.phase_presence[phase] == "absent"
        assert result.phase_amount_ranges_mol[phase] == (0.0, 0.0)


def test_duplicate_candidate_refusal_preserves_validated_bulk():
    bulk = {"MgO": 1.0, "SiO2": 2.0}

    result = equilibrate(bulk, 1800.0, candidates=("forsterite", "forsterite"))

    assert result.refusal is not None
    assert result.refusal.code == "missing_phase_data"
    assert result.bulk_parent_mol == bulk
    assert result.receipt.request["bulk_parent_mol"] == bulk


def test_inner_melt_arithmetic_failure_has_explicit_code(monkeypatch, ideal_pack):
    def arithmetic_failure(*_args, **_kwargs):
        raise ArithmeticError("synthetic arithmetic failure")

    monkeypatch.setattr("openimcc.equilibrium._positive_melt", arithmetic_failure)
    result = equilibrate(
        {"SiO2": 1.0}, 2000.0, candidates=(), pack=ideal_pack,
    )

    assert result.refusal is not None
    assert result.refusal.code == "arithmetic_failure"
    assert result.to_dict()["refusal"]["code"] == "arithmetic_failure"


def test_inner_activity_uncertainty_covers_trace_residual_and_scale(ideal_pack):
    parents = ("SiO2", "MgO")
    liquid = np.asarray((1.0e-6, 1.0))
    temperature = 1800.0

    def evaluate_liquid(values):
        return evaluate(
            dict(zip(parents, map(float, values), strict=True)),
            temperature,
            ideal_pack,
            basis_type="mol",
            allow_out_of_envelope=True,
        )

    melt = evaluate_liquid(liquid)
    residual_melt = replace(
        melt,
        convergence=replace(melt.convergence, residual_inf=1.0e-10),
    )
    bound, _jacobian_norm = _inner_activity_uncertainty(
        liquid, parents, temperature, ideal_pack, residual_melt, 1.0e-12, 100,
    )

    normalized = liquid / math.fsum(map(float, liquid))
    perturbation = 1.0e-10
    shifted = evaluate_liquid(
        (normalized[0] + perturbation, normalized[1] - perturbation)
    )
    measured = abs(
        math.log(shifted.activity("SiO2")) - math.log(melt.activity("SiO2"))
    )
    analytic = math.log((normalized[0] + perturbation) / normalized[0])
    assert measured == pytest.approx(analytic, rel=1.0e-7)
    assert measured == pytest.approx(9.99951e-5, rel=2.0e-5)
    assert bound >= measured

    scale = 1.0e6
    scaled_liquid = liquid * scale
    scaled_melt = evaluate_liquid(scaled_liquid)
    scaled_residual_melt = replace(
        scaled_melt,
        convergence=replace(scaled_melt.convergence, residual_inf=1.0e-10),
    )
    scaled_bound, _ = _inner_activity_uncertainty(
        scaled_liquid, parents, temperature, ideal_pack,
        scaled_residual_melt, 1.0e-12, 100,
    )
    assert scaled_bound == pytest.approx(bound, rel=2.0e-6)


def test_default_crystal_thermo_context_loads_gas_pack_once(monkeypatch):
    calls = []
    loaded = object()

    def instrumented_load(*args, **kwargs):
        calls.append((args, kwargs))
        return loaded

    monkeypatch.setattr(gas_module, "load_gas_datapack", instrumented_load)

    first, second = diagnostics._crystal_thermo_context(None)

    assert len(calls) == 1
    assert first is loaded
    assert second is loaded


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
