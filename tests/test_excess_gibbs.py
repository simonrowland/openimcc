"""Opt-in analytical-parent excess-Gibbs candidate machinery."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from openimcc import (
    ImccMalformedDatapackError,
    ImccRefusal,
    ImccSpeciesPairMultipleRootsError,
    ImccSpeciesPairUnstableError,
    ImccUnprovenDatapackError,
    RedoxExcessGibbsUnsupportedError,
    evaluate,
    evaluate_redox,
    list_excess_gibbs_interactions,
    load_datapack,
)
from openimcc.model import (
    _analytical_parent_excess_state,
    _species_pair_excess_state,
)
from openimcc.kernel import solve_imcc_sf04


ROOT = Path(__file__).parents[1]
CANDIDATE_SOURCE = ROOT / "src/openimcc/data/packs/imcc-sf04-d066-v5.json"
PUBLISHED_SOURCE = ROOT / "src/openimcc/data/packs/imcc-sf04-v1.0.2.json"
EXCESS_MODEL_ID = "IMCC-SF04-D066-EXCESS-CANDIDATE"


def _term(
    *,
    W_H_J_mol: float = -60_000.0,
    W_S_J_mol_K: float = 0.0,
    components: tuple[str, str] = ("Na2O", "CaO"),
) -> dict:
    return {
        "basis": "analytical_parent",
        "components": list(components),
        "W_H_J_mol": W_H_J_mol,
        "W_S_J_mol_K": W_S_J_mol_K,
        "provenance": {"source": "synthetic test coefficient"},
        "calibration_role": "activity surface test",
        "valid_composition": {
            components[0]: [0.0, 1.0],
            components[1]: [0.0, 1.0],
        },
        "T_range_K": [300.0, 5000.0],
        "fit_status": "empirical-fit",
    }


def _species_term(
    *,
    W_H_J_mol: float = 4_000.0,
    W_S_J_mol_K: float = 0.0,
    components: tuple[str, str] = ("FeO", "Fe2SiO4"),
) -> dict:
    return {
        "basis": "species_pair",
        "components": list(components),
        "W_H_J_mol": W_H_J_mol,
        "W_S_J_mol_K": W_S_J_mol_K,
        "provenance": {"source": "synthetic species-pair test coefficient"},
        "calibration_role": "species-pair regression test",
        "valid_composition": {name: [0.0, 1.0] for name in components},
        "T_range_K": [300.0, 5000.0],
        "fit_status": "empirical-fit",
    }


def _candidate(tmp_path: Path, terms: list[dict], name: str = "candidate.json"):
    data = json.loads(CANDIDATE_SOURCE.read_text(encoding="utf-8"))
    data["model_id"] = EXCESS_MODEL_ID
    data["excess_gibbs"] = {"terms": terms}
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return load_datapack(path), path


def _same_legacy_numbers(left, right) -> None:
    for field in (
        "parent_mol",
        "parent_x",
        "parent_x_star",
        "parent_activity",
        "parent_gamma",
        "complex_x",
        "species_x",
    ):
        assert np.asarray(getattr(left, field)).tobytes() == np.asarray(
            getattr(right, field)
        ).tobytes()
    assert left.D == right.D
    assert left.basis == right.basis
    assert left.convergence == right.convergence
    assert left.excess_gibbs_J == right.excess_gibbs_J == 0.0
    assert left.excess_enthalpy_J == right.excess_enthalpy_J == 0.0


def _same_speciation_numbers(left, right) -> None:
    for field in (
        "parent_mol",
        "parent_x",
        "parent_x_star",
        "parent_activity",
        "parent_gamma",
        "complex_x",
        "species_x",
    ):
        assert np.asarray(getattr(left, field)).tobytes() == np.asarray(
            getattr(right, field)
        ).tobytes()
    assert left.D == right.D
    assert left.basis == right.basis
    assert left.convergence == right.convergence


@pytest.mark.parametrize("terms", [[], [_term(W_H_J_mol=0.0, W_S_J_mol_K=0.0)]])
def test_empty_or_zero_excess_terms_preserve_legacy_numeric_outputs(
    tmp_path: Path, terms: list[dict]
) -> None:
    candidate, _path = _candidate(tmp_path, terms)
    composition = {
        "Na2O": 1.0,
        "CaO": 1.0,
        "SiO2": 6.0,
        "MgO": 2.0,
        "Al2O3": 0.5,
    }
    legacy = evaluate(composition, 1800.0, allow_out_of_envelope=True)
    result = evaluate(
        composition,
        1800.0,
        candidate,
        allow_out_of_envelope=True,
    )
    _same_legacy_numbers(legacy, result)
    assert not any(
        notice.startswith("excess-gibbs-candidate:")
        for notice in result.labels.notices
    )


@pytest.mark.parametrize("with_analytical_parent_term", [False, True])
def test_zero_species_pair_preserves_existing_solver_bits(
    tmp_path: Path, with_analytical_parent_term: bool
) -> None:
    composition = {"FeO": 1.0, "SiO2": 1.0, "Na2O": 0.25}
    pair_zero = _species_term(W_H_J_mol=0.0)
    if with_analytical_parent_term:
        analytical = _term(W_H_J_mol=-12_000.0)
        reference, _ = _candidate(tmp_path, [analytical], "parent-only.json")
        candidate, _ = _candidate(
            tmp_path, [analytical, pair_zero], "parent-and-zero-pair.json"
        )
        expected = evaluate(
            composition, 1800.0, reference, allow_out_of_envelope=True
        )
        actual = evaluate(
            composition, 1800.0, candidate, allow_out_of_envelope=True
        )
        _same_speciation_numbers(expected, actual)
        assert expected.excess_gibbs_J == actual.excess_gibbs_J
        assert expected.excess_enthalpy_J == actual.excess_enthalpy_J
    else:
        candidate, _ = _candidate(tmp_path, [pair_zero])
        expected = evaluate(composition, 1800.0, allow_out_of_envelope=True)
        actual = evaluate(
            composition, 1800.0, candidate, allow_out_of_envelope=True
        )
        _same_legacy_numbers(expected, actual)


def test_species_pair_potentials_obey_finite_difference_euler_and_gibbs_duhem(
    tmp_path: Path,
) -> None:
    candidate, _ = _candidate(
        tmp_path,
        [_species_term(W_H_J_mol=42_000.0, W_S_J_mol_K=5.0)],
    )
    species_names = candidate.parent_oxides + tuple(candidate.kernel_datapack.reactions)
    amounts = np.linspace(0.5, 3.0, len(species_names))
    temperature = 2200.0
    terms = list_excess_gibbs_interactions(candidate)
    gibbs, _enthalpy, mu = _species_pair_excess_state(
        species_names, amounts / amounts.sum(), float(amounts.sum()),
        temperature, terms,
    )

    step = 1.0e-5
    for index in range(len(species_names)):
        plus = amounts.copy()
        minus = amounts.copy()
        plus[index] += step
        minus[index] -= step
        g_plus = _species_pair_excess_state(
            species_names, plus / plus.sum(), float(plus.sum()), temperature, terms
        )[0]
        g_minus = _species_pair_excess_state(
            species_names, minus / minus.sum(), float(minus.sum()), temperature, terms
        )[0]
        assert mu[index] == pytest.approx(
            (g_plus - g_minus) / (2.0 * step), rel=2.0e-8, abs=1.0e-5
        )
    assert float(amounts @ mu) == pytest.approx(gibbs, rel=2.0e-14)

    fractions = amounts / amounts.sum()
    direction = np.linspace(-0.01, 0.01, len(species_names))
    direction -= direction.mean()
    epsilon = 1.0e-5
    plus_mu = _species_pair_excess_state(
        species_names, fractions + epsilon * direction, 1.0, temperature, terms
    )[2]
    minus_mu = _species_pair_excess_state(
        species_names, fractions - epsilon * direction, 1.0, temperature, terms
    )[2]
    assert float(fractions @ ((plus_mu - minus_mu) / (2.0 * epsilon))) == pytest.approx(
        0.0, abs=2.0e-8
    )


def test_species_pair_binary_limit_matches_analytic_regular_solution(
    tmp_path: Path,
) -> None:
    candidate, _ = _candidate(tmp_path, [_species_term(W_H_J_mol=48_000.0)])
    species_names = candidate.parent_oxides + tuple(candidate.kernel_datapack.reactions)
    first, second = (species_names.index(name) for name in ("FeO", "Fe2SiO4"))
    fractions = np.zeros(len(species_names))
    fractions[first], fractions[second] = 0.25, 0.75
    gibbs, enthalpy, mu = _species_pair_excess_state(
        species_names, fractions, 3.5, 1800.0,
        list_excess_gibbs_interactions(candidate),
    )
    pair_g = 48_000.0 * 0.25 * 0.75
    assert gibbs == pytest.approx(3.5 * pair_g)
    assert enthalpy == pytest.approx(3.5 * pair_g)
    assert mu[first] == pytest.approx(48_000.0 * 0.75**2)
    assert mu[second] == pytest.approx(48_000.0 * 0.25**2)
    spectator = species_names.index("SiO2")
    assert mu[spectator] == pytest.approx(-pair_g)


def test_species_pair_mass_action_activity_and_enthalpy_use_joules(
    tmp_path: Path,
) -> None:
    candidate, _ = _candidate(tmp_path, [_species_term(W_H_J_mol=4_000.0)])
    composition = {"FeO": 1.0, "SiO2": 1.0}
    temperature = 1800.0
    result = evaluate(
        composition, temperature, candidate, allow_out_of_envelope=True
    )
    species_moles = sum(composition.values()) / result.D
    gibbs, enthalpy, mu = _species_pair_excess_state(
        result.species_names,
        result.species_x,
        species_moles,
        temperature,
        candidate.excess_gibbs_terms,
    )
    assert result.excess_gibbs_J == pytest.approx(gibbs)
    assert result.excess_enthalpy_J == pytest.approx(enthalpy)
    feo = result.parent_oxides.index("FeO")
    assert result.parent_activity[feo] == pytest.approx(
        result.parent_x_star[feo] * np.exp(mu[feo] / (8.31446261815324 * temperature))
    )
    from openimcc.diagnostics import mixing_enthalpy

    enthalpy_diagnostic = mixing_enthalpy(candidate, composition, temperature)
    assert enthalpy_diagnostic["excess_interaction_enthalpy_kJ"] == pytest.approx(
        result.excess_enthalpy_J / 1000.0
    )
    assert enthalpy_diagnostic["total_enthalpy_kJ"] == pytest.approx(
        enthalpy_diagnostic["association_enthalpy_kJ"]
        + enthalpy_diagnostic["excess_interaction_enthalpy_kJ"]
    )

    pack = candidate.kernel_datapack
    active_parent = result.parent_mol > 0.0
    active_complex = ~np.any((pack.nu > 0.0) & (~active_parent[:, None]), axis=0)
    for j in np.flatnonzero(active_complex):
        needed = pack.nu[:, j] > 0.0
        ln_k = np.log(10.0) * (pack.A[j] + pack.B[j] / temperature)
        residual = (
            np.log(result.complex_x[j])
            - float(pack.nu[needed, j] @ np.log(result.parent_x_star[needed]))
            - ln_k
            + (mu[len(result.parent_oxides) + j] - float(pack.nu[:, j] @ mu[:len(result.parent_oxides)]))
            / (8.31446261815324 * temperature)
        )
        assert residual == pytest.approx(0.0, abs=2.0e-10)
def test_public_kernel_refuses_excess_candidate_and_loaded_adapter_applies_it(
    tmp_path: Path,
) -> None:
    candidate, _path = _candidate(tmp_path, [_term()])
    composition = {"Na2O": 1.0, "CaO": 1.0, "SiO2": 4.0, "MgO": 1.0}
    parent_mol = [composition.get(name, 0.0) for name in candidate.parent_oxides]

    with pytest.raises(ImccRefusal, match="ideal-only.*loaded wrapper") as error:
        solve_imcc_sf04(parent_mol, 1800.0, candidate.kernel_datapack)
    assert error.value.code == "imcc_excess_gibbs_requires_loaded_adapter"

    corrected = evaluate(composition, 1800.0, candidate, allow_out_of_envelope=True)
    ideal = evaluate(composition, 1800.0, allow_out_of_envelope=True)
    assert corrected.excess_gibbs_J != 0.0
    assert corrected.activity("Na2O") != ideal.activity("Na2O")
    assert any(
        "Na2O–CaO" in notice and "empirical-fit" in notice
        for notice in corrected.labels.notices
    )

    species_pair_candidate, _ = _candidate(
        tmp_path, [_species_term()], "species-pair-kernel-refusal.json"
    )
    pair_parent_mol = [
        1.0 if name in {"FeO", "SiO2"} else 0.0
        for name in species_pair_candidate.parent_oxides
    ]
    with pytest.raises(ImccRefusal) as pair_error:
        solve_imcc_sf04(
            pair_parent_mol, 1800.0, species_pair_candidate.kernel_datapack
        )
    assert pair_error.value.code == "imcc_excess_gibbs_requires_loaded_adapter"
    with pytest.raises(ImccUnprovenDatapackError, match="loaded wrapper"):
        evaluate(
            {"FeO": 1.0, "SiO2": 1.0},
            1800.0,
            species_pair_candidate.kernel_datapack,
            allow_out_of_envelope=True,
        )
    pair_result = evaluate(
        {"FeO": 1.0, "SiO2": 1.0},
        1800.0,
        species_pair_candidate,
        allow_out_of_envelope=True,
    )
    assert pair_result.convergence.solver_path == "species-pair-multistart"


@pytest.mark.parametrize(
    "components", [("FeO", "Fe2SiO4"), ("FeO", "SiO2")]
)
def test_species_pair_trace_parent_cold_start_and_zero_limit(
    tmp_path: Path, components: tuple[str, str]
) -> None:
    composition = {
        "SiO2": 0.011138886562497141,
        "MgO": 2288.8042997487846,
        "FeO": 87.38220230346212,
        "CaO": 2014.8644234379233,
        "Al2O3": 1360.0091131723866,
        "TiO2": 192.82496476933426,
        "Na2O": 5.3731722769972105e-09,
        "K2O": 8.844980280923895,
    }
    temperature = 2445.65
    ideal = evaluate(composition, temperature)

    candidate, _ = _candidate(
        tmp_path,
        [_species_term(W_H_J_mol=100.0, components=components)],
        "trace-pair.json",
    )
    paired = evaluate(
        composition, temperature, candidate, allow_out_of_envelope=True
    )
    assert paired.convergence.residual_inf <= 1.0e-12
    assert paired.convergence.solver_path == "species-pair-multistart"
    assert paired.parent_activity[paired.parent_oxides.index("Na2O")] > 0.0
    assert np.allclose(
        paired.parent_activity, ideal.parent_activity, rtol=1.0e-2, atol=1.0e-12
    )

    tiny, _ = _candidate(
        tmp_path,
        [_species_term(W_H_J_mol=1.0e-3, components=components)],
        "trace-pair-tiny.json",
    )
    near_ideal = evaluate(
        composition, temperature, tiny, allow_out_of_envelope=True
    )
    assert np.allclose(
        near_ideal.parent_activity, ideal.parent_activity, rtol=1.0e-7, atol=1.0e-12
    )


def test_excess_parent_potentials_are_gibbs_derivatives_and_obey_euler_and_gd(
    tmp_path: Path,
) -> None:
    candidate, _path = _candidate(
        tmp_path,
        [
            _term(
                W_H_J_mol=42_000.0,
                W_S_J_mol_K=5.0,
                components=("Na2O", "CaO"),
            ),
            _term(
                W_H_J_mol=-18_000.0,
                W_S_J_mol_K=-1.5,
                components=("MgO", "SiO2"),
            ),
        ],
    )
    pack = candidate
    parents = tuple(pack.parent_oxides)
    amounts = np.array([5.0, 3.0, 2.0, 4.0, 2.0, 1.0, 2.0, 3.0])
    temperature = 2200.0
    terms = list_excess_gibbs_interactions(pack)
    gibbs, enthalpy, mu = _analytical_parent_excess_state(
        parents, amounts, temperature, terms
    )

    step = 1.0e-5
    for index in range(len(parents)):
        plus = amounts.copy()
        minus = amounts.copy()
        plus[index] += step
        minus[index] -= step
        g_plus = _analytical_parent_excess_state(
            parents, plus, temperature, terms
        )[0]
        g_minus = _analytical_parent_excess_state(
            parents, minus, temperature, terms
        )[0]
        finite_difference = (g_plus - g_minus) / (2.0 * step)
        assert mu[index] == pytest.approx(finite_difference, rel=2.0e-8, abs=1.0e-5)

    assert float(amounts @ mu) == pytest.approx(gibbs, rel=2.0e-14)
    assert enthalpy == pytest.approx(
        float(np.sum(amounts))
        * sum(
            term.W_H_J_mol
            * (amounts[parents.index(term.components[0])] / amounts.sum())
            * (amounts[parents.index(term.components[1])] / amounts.sum())
            for term in terms
        )
    )

    fractions = amounts / amounts.sum()
    direction = np.array([0.01, -0.02, 0.015, -0.005, 0.01, -0.015, 0.005, 0.0])
    assert float(direction.sum()) == pytest.approx(0.0)
    epsilon = 1.0e-5
    plus_mu = _analytical_parent_excess_state(
        parents, fractions + epsilon * direction, temperature, terms
    )[2]
    minus_mu = _analytical_parent_excess_state(
        parents, fractions - epsilon * direction, temperature, terms
    )[2]
    directional_derivative = (plus_mu - minus_mu) / (2.0 * epsilon)
    assert float(fractions @ directional_derivative) == pytest.approx(0.0, abs=2.0e-7)


def test_binary_interaction_limits_and_candidate_evaluation_keep_speciation(
    tmp_path: Path,
) -> None:
    candidate, _path = _candidate(tmp_path, [_term()])
    parents = tuple(candidate.parent_oxides)
    term = list_excess_gibbs_interactions(candidate)
    temperature = 1800.0

    ca_free = np.zeros(len(parents))
    ca_free[parents.index("SiO2")] = 1.0
    ca_free[parents.index("Na2O")] = 1.0
    g_ca_free, _h, mu_ca_free = _analytical_parent_excess_state(
        parents, ca_free, temperature, term
    )
    assert g_ca_free == 0.0
    assert mu_ca_free[parents.index("SiO2")] == 0.0
    assert mu_ca_free[parents.index("Na2O")] == 0.0

    na_free = np.zeros(len(parents))
    na_free[parents.index("SiO2")] = 1.0
    na_free[parents.index("CaO")] = 1.0
    g_na_free, _h, mu_na_free = _analytical_parent_excess_state(
        parents, na_free, temperature, term
    )
    assert g_na_free == 0.0
    assert mu_na_free[parents.index("SiO2")] == 0.0
    assert mu_na_free[parents.index("CaO")] == 0.0

    binary = np.zeros(len(parents))
    binary[parents.index("Na2O")] = 1.0
    binary[parents.index("CaO")] = 1.0
    binary_g, _h, binary_mu = _analytical_parent_excess_state(
        parents, binary, temperature, term
    )
    assert binary_g == pytest.approx(term[0].W_J_mol(temperature) / 2.0)
    assert binary_mu[parents.index("Na2O")] == pytest.approx(
        term[0].W_J_mol(temperature) / 4.0
    )
    assert binary_mu[parents.index("CaO")] == pytest.approx(
        term[0].W_J_mol(temperature) / 4.0
    )

    composition = {"Na2O": 1.0, "CaO": 1.0, "SiO2": 4.0, "MgO": 1.0}
    ideal = evaluate(composition, temperature, allow_out_of_envelope=True)
    corrected = evaluate(
        composition,
        temperature,
        candidate,
        allow_out_of_envelope=True,
    )
    with pytest.raises(ImccUnprovenDatapackError, match="loaded wrapper"):
        evaluate(
            composition,
            temperature,
            candidate.kernel_datapack,
            allow_out_of_envelope=True,
        )
    assert corrected.parent_x_star.tobytes() == ideal.parent_x_star.tobytes()
    assert corrected.species_x.tobytes() == ideal.species_x.tobytes()
    assert corrected.parent_activity[parents.index("Na2O")] != ideal.parent_activity[
        parents.index("Na2O")
    ]
    assert corrected.parent_gamma[parents.index("Na2O")] == pytest.approx(
        corrected.parent_activity[parents.index("Na2O")]
        / corrected.parent_x[parents.index("Na2O")]
    )
    expected_g, expected_h, _mu = _analytical_parent_excess_state(
        parents, corrected.parent_mol, temperature, term
    )
    assert corrected.excess_gibbs_J == pytest.approx(expected_g)
    assert corrected.excess_enthalpy_J == pytest.approx(expected_h)


def test_calibration_range_notice_uses_wt_converted_mole_fractions(
    tmp_path: Path,
) -> None:
    term = _term()
    term["valid_composition"] = {
        "Na2O": [0.0, 0.1],
        "CaO": [0.2, 0.5],
    }
    term["T_range_K"] = [1900.0, 2000.0]
    candidate, _path = _candidate(tmp_path, [term])
    weight_composition = {
        "Na2O": 61.9789,
        "CaO": 56.0774,
        "SiO2": 4.0 * 60.0843,
        "MgO": 40.3044,
    }
    result = evaluate(
        weight_composition,
        1800.0,
        candidate,
        basis_type="wt",
        allow_extrapolation=True,
        allow_out_of_envelope=True,
    )
    notice = next(
        item
        for item in result.labels.notices
        if item.startswith("excess-gibbs-calibration-range:")
    )
    assert "Na2O–CaO (analytical_parent)" in notice
    assert "X(Na2O)=" in notice and "above upper bound 0.1" in notice
    assert "X(CaO)=" in notice and "below lower bound 0.2" in notice
    assert "T=1800 K below lower bound 1900 K" in notice
    assert result.excess_enthalpy_J != 0.0


def test_candidate_hash_disclosure_accessor_gas_and_cli(tmp_path: Path) -> None:
    low, _ = _candidate(tmp_path, [_term(W_H_J_mol=-40_000.0)], "low.json")
    high, _ = _candidate(tmp_path, [_term(W_H_J_mol=-41_000.0)], "high.json")
    assert low.binding_digest != high.binding_digest

    composition = {"Na2O": 1.0, "CaO": 1.0, "SiO2": 4.0, "MgO": 1.0}
    result = evaluate(
        composition,
        1800.0,
        low,
        allow_out_of_envelope=True,
    )
    assert list_excess_gibbs_interactions(low) == low.excess_gibbs_terms
    notice = next(
        notice
        for notice in result.labels.notices
        if notice.startswith("excess-gibbs-candidate:")
    )
    assert "Na2O–CaO" in notice
    assert "analytical_parent" in notice
    assert "empirical-fit" in notice

    from openimcc import evaluate_gas, load_gas_datapack
    from openimcc.cli import _result_payload

    gas_pack = load_gas_datapack()
    activity = result.activity("Na2O")
    gas_result = evaluate_gas(
        {"Na2O": activity},
        1800.0,
        1.0e-8,
        gas_pack,
        parent_oxides=("Na2O",),
        gas_species=("Na",),
    )
    assert gas_result["Na"] > 0.0
    payload = _result_payload(
        result,
        has_active_excess_terms=any(term.active for term in low.excess_gibbs_terms),
    )
    assert payload["excess_gibbs_J"] == result.excess_gibbs_J
    assert payload["excess_enthalpy_J"] == result.excess_enthalpy_J


@pytest.mark.parametrize(
    "source", [PUBLISHED_SOURCE, CANDIDATE_SOURCE], ids=["published", "existing-d066"]
)
def test_existing_pack_identities_reject_excess_gibbs(
    tmp_path: Path, source: Path
) -> None:
    data = json.loads(source.read_text(encoding="utf-8"))
    data["excess_gibbs"] = {"terms": []}
    path = tmp_path / f"{source.stem}-modified.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ImccMalformedDatapackError, match="new candidate identity"):
        load_datapack(path)


def test_redox_refuses_active_excess_candidate(tmp_path: Path) -> None:
    _candidate_pack, candidate_path = _candidate(tmp_path, [_term()])
    with pytest.raises(RedoxExcessGibbsUnsupportedError) as exc:
        evaluate_redox(
            {"Fe": 1.0, "O": 1.0},
            2200.0,
            "imposed",
            0.0,
            melt_pack_path=candidate_path,
        )
    assert exc.value.code == "redox_excess_gibbs_unsupported"
    _pair_pack, pair_path = _candidate(
        tmp_path, [_species_term()], "redox-species-pair-candidate.json"
    )
    with pytest.raises(RedoxExcessGibbsUnsupportedError) as pair_exc:
        evaluate_redox(
            {"Fe": 1.0, "O": 1.0},
            2200.0,
            "imposed",
            0.0,
            melt_pack_path=pair_path,
        )
    assert pair_exc.value.code == "redox_excess_gibbs_unsupported"


@pytest.mark.parametrize("terms", [[], [_term()]])
@pytest.mark.parametrize(
    "composition",
    [
        {"Na2O": 0.5, "SiO2": 0.5},
        {"Na2O": 0.1, "CaO": 0.3, "SiO2": 0.6},
    ],
)
def test_candidate_does_not_inherit_measured_bias_notices(
    tmp_path: Path, terms: list[dict], composition: dict[str, float],
) -> None:
    candidate, _path = _candidate(tmp_path, terms)
    result = evaluate(
        composition, 1673.15, candidate,
        allow_extrapolation=True, allow_out_of_envelope=True,
    )
    notices = tuple(n for n in result.labels.notices if n.startswith("na2o-activity-bias"))
    assert notices == (
        "na2o-activity-bias-unscored: Na2O activity and Na(g) pressure bias evidence "
        "was measured on the default D066 v5 pack; this pack has not been scored.",
    )
    assert (
        "K predictions against Hastie 1981 KEMS pressures (case 4) have "
        f"not been evaluated for this {EXCESS_MODEL_ID} pack; see "
        "https://github.com/simonrowland/openimcc"
    ) in result.labels.notices


def test_species_pair_identity_hash_and_disclosure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    low_term = _species_term(W_H_J_mol=4_000.0)
    low_term["valid_composition"] = {"FeO": [0.0, 0.0], "Fe2SiO4": [0.0, 0.0]}
    low, low_path = _candidate(tmp_path, [low_term], "pair-low.json")
    high, _ = _candidate(tmp_path, [_species_term(W_H_J_mol=4_100.0)], "pair-high.json")
    assert low.binding_digest != high.binding_digest
    result = evaluate(
        {"FeO": 1.0, "SiO2": 1.0}, 1800.0, low, allow_out_of_envelope=True
    )
    notice = next(
        item for item in result.labels.notices
        if item.startswith("excess-gibbs-candidate:")
    )
    assert "FeO–Fe2SiO4 (species_pair; empirical-fit)" in notice
    assert list_excess_gibbs_interactions(low)[0].basis == "species_pair"
    range_notice = next(
        item for item in result.labels.notices
        if item.startswith("excess-gibbs-calibration-range:")
    )
    assert "X(FeO)=" in range_notice and "above upper bound 0" in range_notice

    from openimcc.cli import _result_payload, main

    payload = _result_payload(result, has_active_excess_terms=True)
    assert payload["excess_gibbs_J"] == result.excess_gibbs_J
    assert any(
        "FeO–Fe2SiO4 (species_pair; empirical-fit)" in item
        for item in payload["labels"]["notices"]
    )
    assert main([
        "solve", "--pack", str(low_path), "--temperature", "1800",
        "--oxide", "FeO=1", "--oxide", "SiO2=1",
        "--allow-out-of-envelope", "--json",
    ]) == 0
    cli_payload = json.loads(capsys.readouterr().out)
    assert cli_payload["status"] == "ok"
    assert any(
        "FeO–Fe2SiO4 (species_pair; empirical-fit)" in item
        for item in cli_payload["labels"]["notices"]
    )


def test_species_pair_refuses_unstable_and_seed_dependent_roots(tmp_path: Path) -> None:
    composition = {"FeO": 1.0, "SiO2": 1.0}
    unstable, _ = _candidate(
        tmp_path,
        [_species_term(W_H_J_mol=1_600_000.0)],
        "unstable.json",
    )
    with pytest.raises(ImccSpeciesPairUnstableError) as error:
        evaluate(composition, 1800.0, unstable, allow_out_of_envelope=True)
    assert error.value.code == "imcc_species_pair_unstable"

    multiple, _ = _candidate(
        tmp_path,
        [_species_term(W_H_J_mol=200_000.0)],
        "multiple.json",
    )
    with pytest.raises(ImccSpeciesPairMultipleRootsError) as error:
        evaluate(composition, 1800.0, multiple, allow_out_of_envelope=True)
    assert error.value.code == "imcc_species_pair_multiple_roots"
