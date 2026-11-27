from __future__ import annotations

import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from openimcc import (
    ImccMQMCandidateNonconvexError,
    evaluate,
    evaluate_mgo_sio2_mqm,
    evaluate_redox,
    load_datapack,
    load_profile,
)
from openimcc._constants import R_J_MOL_K
from openimcc.equilibrium import CrystalCandidate, equilibrate
from openimcc.gas import load_gas_datapack, species_thermo
from openimcc.kernel import (
    _mqm_excess_gibbs_j_mol,
    _mqm_free_pool_state,
)
from openimcc.mqm_mgo_sio2 import (
    _mixing_excess_gibbs_derivative_j_mol,
    _wu_liquid_gibbs_j_mol,
)

ROOT = Path(__file__).resolve().parents[1]


_MARKOVA_PANEL_VI_WT = (
    (2069.15, {"SiO2": 35.57, "Al2O3": 7.39, "FeO": 1.11, "MgO": 44.67, "CaO": 11.27}),
    (2156.15, {"SiO2": 13.60, "Al2O3": 13.09, "FeO": 1.79, "MgO": 46.61, "CaO": 16.34}),
)
_OXIDE_MOLAR_MASS = {
    "SiO2": 60.0843, "Al2O3": 101.9613, "FeO": 71.8444,
    "MgO": 40.3044, "CaO": 56.0774,
}
_HASHIMOTO_START = {
    "SiO2": 0.3379238246485715,
    "Al2O3": 0.01776054124964631,
    "FeO": 0.2794942400441881,
    "MgO": 0.3389669610365858,
    "CaO": 0.02585443302100831,
}
_CAS_ZERO_MG = {"SiO2": 0.50, "Al2O3": 0.20, "CaO": 0.20, "Na2O": 0.10}
_MNO_BASALT = {
    "SiO2": 30.0,
    "MgO": 40.0,
    "FeO": 10.4,
    "CaO": 8.0,
    "Al2O3": 8.0,
    "TiO2": 2.06,
    "Na2O": 0.77,
    "K2O": 0.61,
    "MnO": 0.16,
}


def _mole_fractions_from_wt(weight_percent: dict[str, float]) -> dict[str, float]:
    moles = {
        oxide: amount / _OXIDE_MOLAR_MASS[oxide]
        for oxide, amount in weight_percent.items()
    }
    total = math.fsum(moles.values())
    return {oxide: amount / total for oxide, amount in moles.items()}


def _mqm_edge_series_reference(
    y_sio2: float, temperature: float
) -> tuple[float, float, float, float, float]:
    """Derived O(t^2) edge series: gE, dgE/dy, gE'', muMgE, muSiE."""
    z_mgo, z_sio2 = 1.37744375, 2.7548875
    rt = R_J_MOL_K * temperature
    if y_sio2 < 0.5:
        t = y_sio2
        delta_0 = -86090.0
        d_delta_dy_0 = -48974.0 + 37.656 * temperature
        pair_constant = math.exp(-delta_0 / rt)
        a = z_sio2 * delta_0 / 2.0
        b = z_sio2**2 / (2.0 * z_mgo) * (
            d_delta_dy_0 - delta_0 / pair_constant
            + rt * (1.0 - (1.0 + math.log(pair_constant)) / pair_constant)
        )
        derivative = a + 2.0 * b * t
        mu_mgo = -b * t**2
        mu_sio2 = derivative - b * t**2
    else:
        t = 1.0 - y_sio2
        delta_1 = 193045.0 - 87.864 * temperature
        d_delta_dy_1 = 2247789.0 - 840.984 * temperature
        pair_constant = math.exp(-delta_1 / rt)
        a = z_mgo * delta_1 / 2.0
        b = z_mgo**2 / (2.0 * z_sio2) * (
            -d_delta_dy_1 - delta_1 / pair_constant
            + rt * (1.0 - (1.0 + math.log(pair_constant)) / pair_constant)
        )
        derivative = -a - 2.0 * b * t
        mu_mgo = -derivative - b * t**2
        mu_sio2 = -b * t**2
    return a * t + b * t**2, derivative, 2.0 * b, mu_mgo, mu_sio2


def test_binary_join_matches_wu_standard_mixing_before_conversion() -> None:
    temperature = 1873.15
    profile = load_profile("mqm-candidate")
    standards = load_gas_datapack()
    for x_sio2 in (0.0, 0.10, 0.30, 0.50, 0.563, 1.0):
        composition = {"MgO": 1.0 - x_sio2, "SiO2": x_sio2}
        result = evaluate(
            composition, temperature, profile=profile, allow_extrapolation=True,
        )
        binary = evaluate_mgo_sio2_mqm(composition, temperature)
        for oxide, candidate_activity, converted_activity in (
            ("MgO", result.activity("MgO"), binary.a_mgo),
            ("SiO2", result.activity("SiO2"), binary.a_sio2),
        ):
            standard_shift = (
                _wu_liquid_gibbs_j_mol(oxide, temperature)
                - species_thermo(oxide, "l", temperature, standards).G_J_mol
            )
            wu_activity = converted_activity * math.exp(
                -standard_shift / (R_J_MOL_K * temperature)
            )
            assert candidate_activity == pytest.approx(
                wu_activity, rel=1.0e-9, abs=0.0
            )


def test_mg_free_cas_na_face_is_exactly_the_default_including_silica_activity() -> None:
    temperature = 1800.0
    default = evaluate(_CAS_ZERO_MG, temperature, allow_extrapolation=True)
    candidate = evaluate(
        _CAS_ZERO_MG, temperature, profile="mqm-candidate",
        allow_extrapolation=True,
    )

    assert np.array_equal(candidate.parent_activity, default.parent_activity)
    assert np.array_equal(candidate.parent_gamma, default.parent_gamma)
    assert np.array_equal(candidate.parent_x_star, default.parent_x_star)
    assert candidate.D == default.D
    assert candidate.activity("SiO2") == default.activity("SiO2")


def test_low_temperature_near_binary_mqm_recovery_solves_the_regression_case() -> None:
    result = evaluate(
        {
            "MgO": 1.0, "SiO2": 0.95, "CaO": 0.01,
            "Al2O3": 0.01, "FeO": 0.02, "Na2O": 0.01,
        },
        1300.0,
        profile="mqm-candidate",
        allow_extrapolation=True,
        allow_out_of_envelope=True,
    )

    assert result.convergence.status == "converged"
    assert result.convergence.residual_inf <= 2.0e-12
    assert result.D == pytest.approx(1.052, abs=0.005)
    smallest_complex_log = float(
        np.min(np.log(result.complex_x[result.complex_x > 0.0]))
    )
    assert smallest_complex_log == pytest.approx(-189.4, abs=2.0)


def test_excess_potentials_are_derivatives_of_one_gibbs_function() -> None:
    temperature = 2069.15
    n_mgo, n_sio2 = 0.73, 0.41
    total = n_mgo + n_sio2
    y = n_sio2 / total
    g_ex, mu_mgo, mu_sio2, _gpp, reported_y = _mqm_free_pool_state(
        n_mgo, n_sio2, temperature
    )
    assert reported_y == y
    assert (1.0 - y) * mu_mgo + y * mu_sio2 == pytest.approx(
        g_ex, abs=1.0e-9
    )

    def extensive_gibbs(mg: float, silica: float) -> float:
        amount = mg + silica
        return amount * _mqm_excess_gibbs_j_mol(silica / amount, temperature)

    step = 1.0e-4
    fd_mgo = (
        extensive_gibbs(n_mgo + step, n_sio2)
        - extensive_gibbs(n_mgo - step, n_sio2)
    ) / (2.0 * step)
    fd_sio2 = (
        extensive_gibbs(n_mgo, n_sio2 + step)
        - extensive_gibbs(n_mgo, n_sio2 - step)
    ) / (2.0 * step)
    assert mu_mgo == pytest.approx(fd_mgo, abs=0.05)
    assert mu_sio2 == pytest.approx(fd_sio2, abs=0.05)


@pytest.mark.parametrize(
    "y_sio2",
    (1.0e-8, 1.0e-7, 1.0e-5, 1.0e-3, 0.999, 1.0 - 1.0e-8),
)
def test_excess_potentials_remain_accurate_at_dilute_edges(y_sio2: float) -> None:
    temperature = 1873.15
    n_mgo, n_sio2 = 1.0 - y_sio2, y_sio2
    _g, mu_mgo, mu_sio2, _gpp, _y = _mqm_free_pool_state(
        n_mgo, n_sio2, temperature
    )
    step = min(n_mgo, n_sio2) * 1.0e-3

    def extensive_gibbs(mg: float, silica: float) -> float:
        amount = mg + silica
        return amount * _mqm_excess_gibbs_j_mol(
            silica / amount, temperature
        )

    def five_point_derivative(index: int) -> float:
        values = []
        for multiplier in (-2, -1, 1, 2):
            amounts = [n_mgo, n_sio2]
            amounts[index] += multiplier * step
            values.append(extensive_gibbs(*amounts))
        return (
            values[0] - 8.0 * values[1] + 8.0 * values[2] - values[3]
        ) / (12.0 * step)

    assert mu_mgo == pytest.approx(five_point_derivative(0), abs=1.0)
    assert mu_sio2 == pytest.approx(five_point_derivative(1), abs=1.0)


@pytest.mark.parametrize(
    "y_sio2",
    (
        1.0e-14, 1.0e-12, 1.0e-11,
        1.0 - 1.0e-11, 1.0 - 1.0e-12, 1.0 - 1.0e-14,
    ),
)
def test_excess_potentials_match_derived_high_precision_edge_series(
    y_sio2: float,
) -> None:
    temperature = 1873.15
    expected_g, _expected_dg, _expected_gpp, expected_mu_mgo, expected_mu_sio2 = (
        _mqm_edge_series_reference(y_sio2, temperature)
    )
    g, mu_mgo, mu_sio2, _gpp, _y = _mqm_free_pool_state(
        1.0 - y_sio2, y_sio2, temperature
    )

    assert g == pytest.approx(expected_g, rel=1.0e-14, abs=1.0e-20)
    assert mu_mgo == pytest.approx(expected_mu_mgo, abs=1.0e-15)
    assert mu_sio2 == pytest.approx(expected_mu_sio2, abs=1.0e-15)


@pytest.mark.parametrize(
    ("temperature", "y_sio2"),
    (
        (1200.0, 1.0e-8), (1200.0, 1.0 - 1.0e-8),
        (1873.15, 1.0e-8), (1873.15, 1.0 - 1.0e-8),
        (3000.0, 1.0e-8), (3000.0, 1.0 - 1.0e-8),
    ),
)
def test_regular_edge_derivative_matches_series_at_switch(
    temperature: float, y_sio2: float
) -> None:
    expected = _mqm_edge_series_reference(y_sio2, temperature)[1]
    actual = _mixing_excess_gibbs_derivative_j_mol(y_sio2, temperature)
    assert actual == pytest.approx(expected, abs=1.0)


def test_excess_potentials_have_finite_limits_when_free_pool_ratio_underflows() -> None:
    temperature = 1873.15
    smallest = math.nextafter(0.0, 1.0)
    dilute_silica = _mqm_free_pool_state(1.0e308, smallest, temperature)
    dilute_magnesia = _mqm_free_pool_state(smallest, 1.0e308, temperature)
    silica_endpoint_derivative = _mixing_excess_gibbs_derivative_j_mol(
        0.0, temperature
    )
    magnesia_endpoint_derivative = _mixing_excess_gibbs_derivative_j_mol(
        1.0, temperature
    )

    assert dilute_silica[4] == 0.0
    assert dilute_silica[0] == 0.0
    assert dilute_silica[1] == 0.0
    assert dilute_silica[2] == silica_endpoint_derivative
    assert dilute_silica[3] == 0.0
    assert all(math.isfinite(value) for value in dilute_silica)

    assert dilute_magnesia[4] == 1.0
    assert dilute_magnesia[0] == 0.0
    assert dilute_magnesia[1] == -magnesia_endpoint_derivative
    assert dilute_magnesia[2] == 0.0
    assert dilute_magnesia[3] == 0.0
    assert all(math.isfinite(value) for value in dilute_magnesia)


def test_mno_graft_keeps_and_uses_the_candidate_block() -> None:
    profile = load_profile("mqm-candidate")
    default = evaluate(
        _MNO_BASALT, 1800.0, profile="openimcc", basis_type="wt",
        allow_extrapolation=True, allow_out_of_envelope=True,
    )
    candidate = evaluate(
        _MNO_BASALT, 1800.0, profile=profile, basis_type="wt",
        allow_extrapolation=True, allow_out_of_envelope=True,
    )

    assert profile.melt_pack.mqm_candidate is not None
    assert any(flag.startswith("mqm-candidate:") for flag in candidate.labels.flags)
    assert {"MnSiO3(l)", "Mn2SiO4(l)", "MnAl2O4(l)"} <= set(
        candidate.species_names
    )
    assert candidate.activity("MgO") != default.activity("MgO")


def test_nonconvex_refusal_is_limited_to_the_wu_silica_rich_gap() -> None:
    with pytest.raises(ImccMQMCandidateNonconvexError):
        evaluate(
            {"MgO": 0.30, "SiO2": 0.70}, 1960.0,
            profile="mqm-candidate", allow_extrapolation=True,
        )

    markova_results = [
        evaluate(
            _mole_fractions_from_wt(weight_percent), temperature,
            profile="mqm-candidate", allow_extrapolation=True,
            allow_out_of_envelope=True,
        )
        for temperature, weight_percent in _MARKOVA_PANEL_VI_WT
    ]
    hashimoto = evaluate(
        _HASHIMOTO_START, 1973.15, profile="mqm-candidate",
        allow_extrapolation=True, allow_out_of_envelope=True,
    )
    assert all(result.activity("SiO2") > 0.0 for result in markova_results)
    assert hashimoto.activity("SiO2") > 0.0


def test_default_output_is_bitwise_unchanged_by_candidate_evaluation() -> None:
    temperature = 1973.15
    before = evaluate(
        _HASHIMOTO_START, temperature, allow_extrapolation=True,
        allow_out_of_envelope=True,
    )
    evaluate(
        _HASHIMOTO_START, temperature, profile="mqm-candidate",
        allow_extrapolation=True, allow_out_of_envelope=True,
    )
    after = evaluate(
        _HASHIMOTO_START, temperature, allow_extrapolation=True,
        allow_out_of_envelope=True,
    )

    assert before.parent_activity.tobytes() == after.parent_activity.tobytes()
    assert before.parent_gamma.tobytes() == after.parent_gamma.tobytes()
    assert before.parent_x_star.tobytes() == after.parent_x_star.tobytes()
    assert before.complex_x.tobytes() == after.complex_x.tobytes()
    assert before.species_x.tobytes() == after.species_x.tobytes()
    assert before.D == after.D
    assert before.labels.flags == after.labels.flags


def test_redox_carries_the_candidate_and_equilibrate_refuses_it_explicitly() -> None:
    inventory = {
        "Si": 0.1, "Mg": 1.5, "Fe": 0.15, "O": 1.85,
    }
    default_redox = evaluate_redox(inventory, 3000.0, "closed")
    redox = evaluate_redox(
        inventory, 3000.0, "closed", profile="mqm-candidate"
    )
    assert redox.profile == "mqm-candidate"
    assert any(flag.startswith("mqm-candidate:") for flag in redox.flags)
    assert redox.melt_pack_binding_digest == load_profile(
        "mqm-candidate"
    ).melt_pack.binding_digest
    assert redox.parent_activity["MgO"] != default_redox.parent_activity["MgO"]
    assert redox.parent_activity["MgO"] > 0.0
    assert redox.parent_activity["SiO2"] > 0.0

    equilibrium = equilibrate(
        {"MgO": 0.30, "SiO2": 0.70}, 1800.0,
        pack=load_profile("mqm-candidate").melt_pack,
    )
    assert equilibrium.refusal is not None
    assert equilibrium.refusal.code == "unsupported_nonconvex_model"
    above_consolute = equilibrate(
        {"MgO": 0.30, "SiO2": 0.70}, 2300.0,
        candidates=(), pack=load_profile("mqm-candidate").melt_pack,
    )
    assert above_consolute.refusal is not None
    assert above_consolute.refusal.code == "unsupported_nonconvex_model"

    temperature = 2000.0
    silica_liquid_g = species_thermo(
        "SiO2", "l", temperature, load_gas_datapack()
    ).G_J_mol
    mg_free = equilibrate(
        {"SiO2": 1.0}, temperature,
        candidates=(CrystalCandidate(
            "high-G-silica-control", {"SiO2": 1.0}, silica_liquid_g + 100_000.0,
        ),),
        pack=load_profile("mqm-candidate").melt_pack,
    )
    assert mg_free.refusal is None
    assert mg_free.liquid_mass_fraction == pytest.approx(1.0)

    from openimcc.equilibrium import (
        _EquilibriumModelRefusalError,
        _boundary_context,
    )

    mg_free_boundary = _boundary_context(
        {"SiO2": 1.0}, 2000.0, (), load_profile("mqm-candidate").melt_pack
    )
    assert mg_free_boundary.pack.mqm_candidate is not None
    with pytest.raises(_EquilibriumModelRefusalError) as refusal:
        _boundary_context(
            {"MgO": 1.0, "SiO2": 1.0}, 2000.0, (),
            load_profile("mqm-candidate").melt_pack,
        )
    assert refusal.value.code == "unsupported_nonconvex_model"


def test_loaded_default_pack_has_no_candidate_block() -> None:
    assert load_datapack().mqm_candidate is None


def test_candidate_pack_and_liquid_standards_builders_are_current() -> None:
    result = subprocess.run(
        [sys.executable, "tools/build_mqm_candidate_pack.py", "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
